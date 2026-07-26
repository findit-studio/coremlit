"""Shared loader, pins, driver, and asserts for the granite CoreML conversion.

Source of truth: the OFFICIAL public checkpoint
``ibm-granite/granite-embedding-97m-multilingual-r2`` pinned to ``REV`` below
(Apache-2.0, ungated). The recipe converts FROM this official model; nothing is
consumed from the published ``FinDIT-Studio/embedkit-coreml`` artifact repo.
Local staging only.

All filesystem paths come from the environment (never hardcoded):

  GRANITE_CONV        base scratch dir (required)
  GRANITE_SRC_MODEL   downloaded checkpoint dir     (default $GRANITE_CONV/src-model)
  GRANITE_STAGE       conversion staging dir        (default $GRANITE_CONV/granite/staging)
  GRANITE_MODELS_OUT  final gitignored Models tree  (staging/verify steps only)
  GRANITE_GOLDENS     committed goldens dir         (goldens step only)

The checkpoint is loaded from the local snapshot (GRANITE_SRC_MODEL) and every
source file's SHA-256 is asserted against the pins below on load — a stricter,
offline-reproducible form of ``from_pretrained(..., revision=REV)``.

REPLAYABILITY SCOPE: this recipe re-derives an artifact EQUIVALENT to the
published one to the floors in ``verify_granite.py``. It is not, and does not
claim to be, bit-reproducible — the CoreML compiler's output and torch's fp32
reduction order are not pinned by these versions. See README.md for the measured
byte-identity result against the published manifest.
"""
import datetime
import hashlib
import json
import os
import platform
import uuid

import numpy as np
import torch

MODEL_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"
REV = "835ad14087e140460703cf0fae09f97d469d65c2"

# Per-source-file SHA-256 at REV (verified 2026-07-26). model.safetensors and
# tokenizer.json are the load-bearing weight/tokenizer identities (tokenizer.json
# is ALSO the `coremlit::embeddings::granite::BUNDLED_TOKENIZER` pin, asserted in
# tests/granite/tokenizer_identity.rs); the JSON configs are pinned because the
# graph shape (layer_types, rope thetas, local_attention, norm_eps) and the
# pooling/prompt contract are read from them, not hardcoded here.
SOURCE_SHA256 = {
    "model.safetensors": "f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be",
    "tokenizer.json": "4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f",
    "tokenizer_config.json": "6ed69389e30a8ecabfce2f9ebcdf0c908b34056f24d994340f2f216521c057d5",
    "special_tokens_map.json": "013787ee251ff611722479197c00853b62113ad303cb0a36524231783c676c69",
    "config.json": "de948b0bdc6f356afad7a84b276d8dd7e7fe10fb9add1bb5e610621c28e41ebc",
    "config_sentence_transformers.json": "93a59cbc7d82a47a7148719d9b21c0f2f111121e495b3918143184f4cd0ea25e",
    "sentence_bert_config.json": "967ef958285e4a7a37d8ff1832473d967edd913b4e48572f31c3d3ea361d5327",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "1_Pooling/config.json": "8bc5c9a40814fcf48d2fbe7cfeff4bee6736c3c2a823ba0ce098985c59d12ab7",
}

# The frozen shipped contract (mirrors src/embeddings/granite + tests/granite).
SEQ_LEN = 512               # input_ids / attention_mask [1, 512]
EMBED_DIM = 384             # embedding [1, 384]
MODEL_STEM = "granite_97m_512"
BUNDLE_SUBDIR = "granite-97m-multilingual-r2"

# The compute units verify_granite.py must exercise, named ONCE so the matrix it
# runs and the evidence gate that reads its output cannot drift apart.
COMPUTE_UNIT_NAMES = ("CpuOnly", "CpuAndGpu", "CpuAndNeuralEngine", "All")

# The published artifact's EXACT file set (the enumerate-then-hash discipline
# `tests/granite/model_io.rs` applies to the bundle). The model card is part of
# the distributed set but is staged, never generated here; the two names below it
# are this recipe's own outputs and are excluded from discovery.
MODEL_CARD = "README.md"
GENERATED_AT_ROOT = ("CHECKSUMS.sha256", "MANIFEST.json")
FP32_REFERENCE = f"{MODEL_STEM}_fp32.mlpackage"
SHIPPED_PACKAGE = f"{MODEL_STEM}.mlpackage"
SHIPPED_BUNDLE = f"{MODEL_STEM}.mlmodelc"
VERIFY_METRICS = "verify_metrics.json"
PRODUCER_RECORD = "producer.json"
STAGED_CROSSCHECK = "driver_crosscheck.json"

# Every attestation this recipe writes carries the id of the run that produced
# it, and every consumer requires that id to match. The stages are separately
# invocable, so without one identity a manifest can be assembled from a producer
# record, a verification and a crosscheck that each describe a DIFFERENT run —
# each internally consistent, the combination describing nothing that ever
# existed.
RUN_ID_KEY = "run_id"

# SHA-256 of the published model card. The card belongs to the distributed set
# but is staged rather than generated, so its bytes are pinned like any other
# source input instead of trusted from wherever they were copied.
MODEL_CARD_SHA256 = "a9bacaf49d780b5a6de07043557805a183f3eb4a191600bd7f89785ad3d90796"

EXPECTED_BUNDLE_FILES = sorted([
    f"./{MODEL_STEM}.mlmodelc/analytics/coremldata.bin",
    f"./{MODEL_STEM}.mlmodelc/coremldata.bin",
    f"./{MODEL_STEM}.mlmodelc/metadata.json",
    f"./{MODEL_STEM}.mlmodelc/model.mil",
    f"./{MODEL_STEM}.mlmodelc/weights/weight.bin",
    f"./{MODEL_STEM}.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    f"./{MODEL_STEM}.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    f"./{MODEL_STEM}.mlpackage/Manifest.json",
])
EXPECTED_ARTIFACT_FILES = sorted(EXPECTED_BUNDLE_FILES + [f"./{MODEL_CARD}"])

# The additive attention-mask "blocked" value. LOAD-BEARING, and NOT
# ``torch.finfo(dtype).min``: the shipped artifact is fp16, and -3.4e38 overflows
# to -inf in coremltools' fp16 cast. A fully-padded query row is then all -inf,
# whose softmax is NaN — measured on the CpuOnly backend as 15/16 corpus entries
# returning non-finite output, while CpuAndGpu/ANE/All stayed clean (they
# evidently evaluate the row differently), so this fails on ONE arm only. -1e4 is
# exactly representable in fp16 and far below any attention logit this graph
# produces, so blocked keys underflow to zero in the softmax and an all-blocked
# row degrades to a uniform, finite distribution instead of NaN. It is also the
# value in the published artifact's mask constants (allow 0.0, block -10000.0,
# read back from its weights/weight.bin).
MASK_BLOCK = -1e4

# Faithfulness / accuracy floors. Breaching one is a FINDING, never a number to
# loosen. Measured values live in README.md and staging/verify_metrics.json.
DRIVER_FLOOR = 0.99999997   # driver wrapper vs the UNMODIFIED canonical pipeline
FP32_FLOOR = 0.99999998     # CoreML fp32 (CPUOnly) vs the committed fp32 goldens
FP16_FLOOR = 0.99996        # CoreML fp16 vs CoreML fp32, on EVERY compute unit

# The committed embeddings are unit-normalized; 7-decimal serialization of 384
# components moves the norm by ~1e-7, so this bounds rounding while still
# rejecting a rescaled oracle.
UNIT_NORM_TOLERANCE = 1e-5

# Floor on the largest per-component difference between the normalized driver
# vector and the canonical one — the positive evidence that the two sides were
# computed independently, which the cosine cannot supply.
#
# MEASURED separation, and it is narrower than it looks: two genuinely distinct
# fp32 computations of this embedding differ by at least 9.28e-08 per component
# over the corpus, while a driver that merely returned the canonical vector
# RESCALED still differs by 1.2e-08 to 2.6e-08 purely from fp32 quantization of
# the normalize-and-compare round trip. Only ~4x separates them, so this floor is
# a secondary band, not the primary defense — an exactly-reproducing stand-in is
# rejected outright by the byte-identity and floor checks in driver_crosscheck().
DISTINCTNESS_FLOOR = 5e-8

# The toolchain this recipe is pinned to. ``python`` is a MAJOR.MINOR pin; every
# other entry is exact. ``observed_toolchain()`` reads what is ACTUALLY running
# and refuses to continue on a mismatch, so nothing can record a version nobody
# ran.
REQUIRED_TOOLCHAIN = {
    "python": "3.11",
    "torch": "2.6.0",
    "transformers": "5.14.0",
    "sentence_transformers": "5.6.0",
    "coremltools": "9.0",
    "numpy": "1.26.4",
}

# Manifest key -> installed distribution name, where they differ.
_DISTRIBUTION_NAMES = {"sentence_transformers": "sentence-transformers"}


def _env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"required environment variable {name} is unset")
    return val


def conv_dir():
    return _env("GRANITE_CONV", required=True)


def src_dir():
    return _env("GRANITE_SRC_MODEL", os.path.join(conv_dir(), "src-model"))


def stage_dir():
    d = _env("GRANITE_STAGE", os.path.join(conv_dir(), "granite", "staging"))
    os.makedirs(d, exist_ok=True)
    return d


def model_root():
    """The shipped bundle's parent dir under the gitignored Models tree."""
    return os.path.join(_env("GRANITE_MODELS_OUT", required=True), BUNDLE_SUBDIR)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_rel(base, prefix=""):
    """EVERY real file under ``base`` as a sorted forward-slash relative path.

    Nothing is filtered by name. An earlier version skipped ``.DS_Store`` and
    ``._``-prefixed AppleDouble files so discovery would match the Rust
    ``collect_files_rel`` exclusions, but that made a real file invisible to both
    the exact-set gate and the checksums — and macOS creates those files on its
    own on non-native filesystems, so it is ordinary operation, not tampering.
    Discovery now sees them and ``assert_no_os_sidecars`` refuses to continue."""
    out = []
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            rel = os.path.relpath(os.path.join(dirpath, name), base).replace(os.sep, "/")
            out.append(prefix + rel)
    return sorted(out)


def assert_no_os_sidecars(root, discovered):
    """Fail-closed on macOS AppleDouble / Finder sidecars under the artifact root.

    These are never part of the artifact and the CoreML loader never reads them,
    but they are real files: leaving them in place would either break the
    exact-set gate or (if excused by name) hide whatever sits behind that name.
    Removing files under someone's tree is not this script's call, so it names
    them and the remedy instead."""
    sidecars = [
        rel for rel in discovered
        if rel.rsplit("/", 1)[-1].startswith("._") or rel.rsplit("/", 1)[-1] == ".DS_Store"
    ]
    if sidecars:
        raise SystemExit(
            f"OS SIDECAR FILES under {root} — refusing to publish a tree containing them:\n"
            f"  {sidecars}\n"
            f"  These appear when the tree lives on exFAT/FAT/SMB. Remove them "
            f"(`dot_clean {root}`, or `find {root} -name '._*' -o -name .DS_Store -delete`) "
            f"and re-run; staging onto a native APFS/HFS+ volume avoids them entirely."
        )


def enumerate_artifact_root(root):
    """Every file under the artifact ROOT, ``./``-relative, minus this recipe's own
    outputs.

    Recursive from the root rather than from the two known bundle names: a stray
    file dropped anywhere under the root — beside the bundles, or nested in one —
    must be discoverable, otherwise the "exact set" contract is only enforced
    where someone thought to look."""
    skip = {f"./{name}" for name in GENERATED_AT_ROOT}
    return sorted(rel for rel in _walk_rel(root, "./") if rel not in skip)


def assert_artifact_file_set(root, expected=None):
    """Fail-closed: the artifact root must hold EXACTLY ``expected``.

    A missing path means staging dropped something; an extra means the toolchain
    (or a stray hand edit) added something. Either way a manifest cut from that
    tree is no longer set-comparable with the published CHECKSUMS.sha256."""
    want = sorted(EXPECTED_ARTIFACT_FILES if expected is None else expected)
    got = enumerate_artifact_root(root)
    assert_no_os_sidecars(root, got)
    if got != want:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        raise SystemExit(
            f"ARTIFACT FILE SET MISMATCH under {root} — refusing to proceed with a tree that is "
            f"not set-comparable with the published manifest:\n"
            f"  missing {missing}\n  unexpected {extra}"
        )
    print(f"[ok] artifact file set: {len(got)} files, exactly the published set")
    return got


def digest_files(root, relatives):
    """``{relative path: sha256}`` for an explicit list of files under ``root``."""
    return {rel: sha256_file(os.path.join(root, rel)) for rel in sorted(relatives)}


def digest_tree(base):
    """``{relative path: sha256}`` for every file under ``base``."""
    return {rel: sha256_file(os.path.join(base, rel)) for rel in _walk_rel(base)}


def assert_staged_matches_staging(root, stage):
    """The shipped tree must be a byte-for-byte copy of what staging holds.

    Verification reads the shipped ``.mlmodelc`` and ``.mlpackage`` from ``root``
    while conversion writes them to ``stage``; the runner copies one to the other.
    Re-running conversion alone rebuilds the staging packages (CoreML emits a
    fresh package UUID every run) and leaves ``root`` holding the previous build,
    whose digests are still internally consistent. Comparing them here is what
    makes "verified" mean "verified THESE shipped bytes"."""
    for sub in (SHIPPED_BUNDLE, SHIPPED_PACKAGE):
        staged = os.path.join(root, sub)
        built = os.path.join(stage, sub)
        if not os.path.isdir(built):
            raise SystemExit(f"missing staging build {built} — run convert_granite.py")
        if not os.path.isdir(staged):
            raise SystemExit(f"missing staged copy {staged} — the staging step did not run")
        if digest_tree(staged) != digest_tree(built):
            raise SystemExit(
                f"STAGED/STAGING DIVERGENCE for {sub}: {staged} is not a copy of {built}.\n"
                f"  The staging step did not run for this build, so verification would measure "
                f"one package and publish another. Re-run run_granite.sh."
            )
    print("[ok] shipped tree is byte-identical to the staging build")


def evidence_digests(root, stage, corpus_path):
    """The bytes ``verify_granite.py``'s numbers were measured from.

    All three parts matter and together they cover the whole matrix: the per-unit
    fp16 arms run the staged ``.mlmodelc`` under ``root`` (a byte copy of the
    staging one), the I/O contract reads the staged fp16 ``.mlpackage``, the fp32
    oracle arm runs the fp32 reference package — which lives ONLY in staging and
    is never shipped, so it is digested from there — and every cosine is scored
    against ``corpus.json``, which is itself regenerable. Without the corpus
    digest, regenerating the oracle would leave a manifest describing comparisons
    against goldens that no longer exist under that name."""
    return {
        "artifact": digest_files(root, EXPECTED_BUNDLE_FILES),
        "fp32_reference": digest_tree(os.path.join(stage, FP32_REFERENCE)),
        "corpus": sha256_file(corpus_path),
    }


def replace_file_atomic(path, text):
    """Write ``text`` to a sibling temp file, then rename it into place.

    A reader either sees the previous complete file or the new complete one. A
    crashed or failing writer cannot leave a half-written record that a later
    stage would read as valid."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def discard_file(path):
    """Remove ``path`` if present. Used to invalidate a previous run's evidence
    before a new run starts, so a failing run cannot leave the old verdict
    standing as though it described the new build."""
    if os.path.exists(path):
        os.remove(path)


def begin_run(stage):
    """Open a new conversion run: mint its id and INVALIDATE every attestation a
    previous run left in ``stage``.

    This happens before any failure-prone work. Conversion loads the checkpoint,
    checks the driver and traces — any of which can fail — and if a previous
    run's producer record, verification and crosscheck were still lying around,
    a failed conversion would leave them intact and downstream stages would
    happily certify the packages they describe. Invalidating first means a failed
    run leaves nothing that can be mistaken for a result."""
    for name in (PRODUCER_RECORD, VERIFY_METRICS, STAGED_CROSSCHECK):
        discard_file(os.path.join(stage, name))
    run_id = uuid.uuid4().hex
    print(f"[ok] run {run_id[:12]} started; prior producer/verification/crosscheck discarded")
    return run_id


def require_run_id(record, run_id, what, path):
    """Every attestation must belong to the run being described."""
    got = record.get(RUN_ID_KEY)
    if got != run_id:
        raise SystemExit(
            f"RUN IDENTITY MISMATCH: {what} ({path}) carries run {got!r}, but this build is run "
            f"{run_id!r}. These records describe different runs; re-run the pipeline in order."
        )


def write_producer_record(stage, run_id, produced):
    """Record WHICH run, WHEN, and under WHICH toolchain ``produced`` was made.

    The timestamp is taken here, in the process that actually converts. The
    manifest writer is independently rerunnable — the next day, or across
    midnight — so a date it sampled itself would name a "conversion date" that
    is not one.

    ``write_manifest.py`` must not resample its own environment: the manifest
    describes the artifact's producer, and the writer can trivially be a
    different shell — 3.11.14 converts and verifies, 3.11.15 writes a manifest
    claiming 3.11.15. The record is bound to the digests of what was produced so
    it cannot be read as describing some other build."""
    record = {
        RUN_ID_KEY: run_id,
        "converted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "toolchain": observed_toolchain(),
        "produced": produced,
    }
    replace_file_atomic(os.path.join(stage, PRODUCER_RECORD),
                        json.dumps(record, indent=2) + "\n")
    print(f"[ok] recorded producer toolchain for run {run_id[:12]} "
          f"over {len(produced)} produced files")
    return record


def read_producer_record(stage, produced):
    """Load the conversion-time producer record and REQUIRE it to describe
    ``produced`` exactly. Aborts if it is missing or bound to other bytes."""
    path = os.path.join(stage, PRODUCER_RECORD)
    if not os.path.exists(path):
        raise SystemExit(
            f"MISSING PRODUCER RECORD: {path} does not exist.\n"
            f"  Run convert_granite.py first — the manifest records the toolchain that "
            f"PRODUCED the artifact, which cannot be recovered after the fact."
        )
    with open(path) as f:
        record = json.load(f)
    for field in (RUN_ID_KEY, "converted_utc"):
        if not record.get(field):
            raise SystemExit(
                f"PRODUCER RECORD ({path}) carries no `{field}`; re-run conversion."
            )
    if record.get("produced") != produced:
        raise SystemExit(
            f"PRODUCER RECORD IS STALE ({path}): it was written for different bytes than the "
            f"packages now in {stage}. Re-run convert_granite.py."
        )
    return record


def write_staged_crosscheck(stage, run_id, crosscheck):
    """Persist the CONVERSION-TIME crosscheck so the goldens step can commit that
    exact measurement instead of recomputing its own."""
    record = dict(crosscheck)
    record[RUN_ID_KEY] = run_id
    replace_file_atomic(os.path.join(stage, STAGED_CROSSCHECK),
                        json.dumps(record, indent=2) + "\n")
    return record


def read_staged_crosscheck(stage, run_id):
    """Load the conversion-time crosscheck for ``run_id``.

    ``generate_goldens.py`` COMMITS this; it must not recompute one of its own.
    The committed fixture is supposed to be the pre-trace faithfulness proof, and
    a second computation made later — however similar its numbers — is a
    different measurement of a different graph instance."""
    path = os.path.join(stage, STAGED_CROSSCHECK)
    if not os.path.exists(path):
        raise SystemExit(
            f"MISSING CONVERSION-TIME CROSSCHECK: {path} does not exist.\n"
            f"  Run convert_granite.py first. The committed driver_crosscheck.json IS the "
            f"pre-trace measurement; this step publishes it, it does not produce it."
        )
    with open(path) as f:
        record = json.load(f)
    require_run_id(record, run_id, "the staged crosscheck", path)
    return record


def expected_metric_keys():
    """The exact key set ``verify_metrics.json`` must carry — one per gated arm,
    so a metrics file that simply omits an arm cannot pass as complete, plus the
    non-numeric checks whose absence would otherwise be silently acceptable."""
    keys = {"fp32_CpuOnly_vs_committed_goldens", "floors", "evidence", "checks",
            "toolchain", RUN_ID_KEY}
    for unit in COMPUTE_UNIT_NAMES:
        keys.add(f"fp16_{unit}_vs_fp32")
        keys.add(f"fp16_{unit}_nonfinite_entries")
    return keys


# Non-numeric verifications whose success must be recorded explicitly. A metrics
# file that merely omits them must not read as "nothing failed".
REQUIRED_CHECKS = ("io_contract", "tokenizer_identity", "corpus_identity")


def require_verify_evidence(root, stage, corpus_path, run_id):
    """Fail-closed: REQUIRE complete, passing, artifact-bound verification evidence.

    Downstream steps must not treat missing evidence as permission to proceed.
    This asserts, in order: the file exists; its key set is exactly
    ``expected_metric_keys()``; the recorded floors equal the constants in force
    now (so evidence cut under loosened floors cannot certify); every worst is
    finite and clears its floor; every non-finite counter is zero; and the
    recorded digests still match the bytes on disk, so metrics from an earlier
    run cannot certify a different bundle."""
    path = os.path.join(stage, VERIFY_METRICS)
    if not os.path.exists(path):
        raise SystemExit(
            f"MISSING VERIFICATION EVIDENCE: {path} does not exist.\n"
            f"  Run verify_granite.py first — an unverified artifact must never be published."
        )
    with open(path) as f:
        metrics = json.load(f)

    problems = []
    want_keys = expected_metric_keys()
    missing = sorted(want_keys - set(metrics))
    extra = sorted(set(metrics) - want_keys)
    if missing:
        problems.append(f"incomplete evidence, missing metrics {missing}")
    if extra:
        problems.append(f"unrecognized metrics {extra}")

    if not missing:
        checks = metrics["checks"]
        for name in REQUIRED_CHECKS:
            if checks.get(name) is not True:
                problems.append(f"check `{name}` did not record success (got {checks.get(name)!r})")
        want_floors = {"fp32_vs_goldens": FP32_FLOOR, "fp16_vs_fp32": FP16_FLOOR}
        if metrics["floors"] != want_floors:
            problems.append(
                f"evidence was cut against floors {metrics['floors']}, not {want_floors}"
            )
        gated = [("fp32_CpuOnly_vs_committed_goldens", FP32_FLOOR)]
        gated += [(f"fp16_{u}_vs_fp32", FP16_FLOOR) for u in COMPUTE_UNIT_NAMES]
        for key, floor in gated:
            worst = metrics[key]
            if not isinstance(worst, (int, float)) or not np.isfinite(worst):
                problems.append(f"{key} is not a finite number ({worst!r})")
            elif not worst >= floor:
                problems.append(f"{key} = {worst:.8f} is below its floor {floor}")
        for unit in COMPUTE_UNIT_NAMES:
            key = f"fp16_{unit}_nonfinite_entries"
            if metrics[key] != 0:
                problems.append(f"{key} = {metrics[key]!r}, expected 0")
        if metrics.get(RUN_ID_KEY) != run_id:
            problems.append(
                f"evidence belongs to run {metrics.get(RUN_ID_KEY)!r}, not this build's {run_id!r}"
            )
        got = evidence_digests(root, stage, corpus_path)
        if metrics["evidence"] != got:
            problems.append(
                "evidence is STALE: the recorded digests do not match the bytes on disk, so "
                "these metrics describe a different build than the one being published"
            )

    if problems:
        raise SystemExit(
            f"VERIFICATION EVIDENCE REJECTED ({path}):\n  " + "\n  ".join(problems)
        )
    print(f"[ok] verification evidence complete, passing, and bound to this build")
    return metrics


def observed_toolchain():
    """Return the versions ACTUALLY running this recipe, after asserting each one
    matches ``REQUIRED_TOOLCHAIN``. Aborts (SystemExit) on any mismatch.

    The manifest must never record a claimed version. A venv that resolved, say,
    numpy 1.26.3 would otherwise complete the whole recipe and then be written
    down as 1.26.4 — a provenance record nobody can replay, which is the exact
    defect this recipe exists to remove. The returned dict carries the observed
    strings (python at full MAJOR.MINOR.PATCH), not the pins."""
    from importlib.metadata import PackageNotFoundError, version

    observed = {"python": platform.python_version()}
    mismatches = []
    want_python = REQUIRED_TOOLCHAIN["python"]
    if observed["python"].rsplit(".", 1)[0] != want_python:
        mismatches.append(f"python: observed {observed['python']}, pinned {want_python}.x")
    for key, want in REQUIRED_TOOLCHAIN.items():
        if key == "python":
            continue
        try:
            got = version(_DISTRIBUTION_NAMES.get(key, key))
        except PackageNotFoundError:
            got = None
        observed[key] = got
        if got != want:
            mismatches.append(f"{key}: observed {got!r}, pinned {want!r}")
    if mismatches:
        raise SystemExit(
            "TOOLCHAIN MISMATCH — refusing to record versions that were not run:\n  "
            + "\n  ".join(mismatches)
        )
    print(f"[ok] toolchain observed and matches the pins (python {observed['python']})")
    return observed


def assert_corpus_identity(entries, source):
    """Fail-closed: ``entries`` must be EXACTLY the committed fixture corpus, in
    order, with the right shape.

    Two distinct failures are covered. An empty or truncated corpus would make
    every downstream loop iterate zero times, leaving each running ``worst`` at
    its 1.0 seed and every non-finite counter at 0 — a verification that made no
    prediction at all would exit 0. And ids alone are not coverage: keeping all
    16 ids while replacing every ``text`` with the shortest one yields 16
    repetitions of a single easy prediction that still clears every floor. So the
    full ORDERED ``(id, text)`` mapping is compared, not the id set.

    Shape is validated too, because the ids and texts can be right while the
    stored oracle is not: ``token_ids`` must agree with ``n_tokens``, embeddings
    must be finite and ``EMBED_DIM`` long, and the over-length entry must still
    sit at exactly ``SEQ_LEN`` so the truncation regime is genuinely exercised."""
    from _fixtures import CORPUS

    problems = []
    got_pairs = [(e.get("id"), e.get("text")) for e in entries]
    want_pairs = [(e["id"], e["text"]) for e in CORPUS]
    if got_pairs != want_pairs:
        got_ids = [i for i, _ in got_pairs]
        want_ids = [i for i, _ in want_pairs]
        if len(got_ids) != len(want_ids):
            problems.append(f"{len(got_ids)} entries, expected {len(want_ids)}")
        dupes = sorted({i for i in got_ids if got_ids.count(i) > 1})
        if dupes:
            problems.append(f"duplicate ids {dupes}")
        missing = sorted(set(want_ids) - set(got_ids))
        extra = sorted(set(got_ids) - set(want_ids))
        if missing:
            problems.append(f"missing ids {missing}")
        if extra:
            problems.append(f"unexpected ids {extra}")
        want_text = dict(want_pairs)
        altered = sorted(i for i, t in got_pairs if i in want_text and t != want_text[i])
        if altered:
            problems.append(f"text differs from the fixture for ids {altered}")
        if not (problems or dupes):
            problems.append("entries are the right set but in a different order")

    for entry in entries:
        eid = entry.get("id", "<no id>")
        token_ids = entry.get("token_ids")
        n_tokens = entry.get("n_tokens")
        embedding = entry.get("embedding")
        if not isinstance(token_ids, list) or not token_ids:
            problems.append(f"`{eid}`: token_ids missing or empty")
        elif token_ids != list(token_ids[:SEQ_LEN]):
            problems.append(f"`{eid}`: {len(token_ids)} token_ids exceeds the {SEQ_LEN} window")
        elif n_tokens != len(token_ids):
            problems.append(f"`{eid}`: n_tokens {n_tokens} != len(token_ids) {len(token_ids)}")
        if not isinstance(embedding, list) or len(embedding) != EMBED_DIM:
            got = len(embedding) if isinstance(embedding, list) else type(embedding).__name__
            problems.append(f"`{eid}`: embedding is {got}, expected {EMBED_DIM} floats")
        elif not all(isinstance(v, (int, float)) and np.isfinite(v) for v in embedding):
            problems.append(f"`{eid}`: embedding holds a non-finite value")
        else:
            # The oracle is documented UNIT-NORMALIZED, so a rescaled vector is a
            # different oracle even though it has the same cosine to everything.
            norm = float(np.linalg.norm(np.asarray(embedding, np.float64)))
            if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
                problems.append(
                    f"`{eid}`: embedding norm {norm:.9f} is not unit within "
                    f"{UNIT_NORM_TOLERANCE:g} — the goldens are L2-normalized by contract"
                )

    # Tie the length witness to the specific fixture that is over-length, so it
    # cannot be satisfied by some other entry having grown to 512.
    witness = max(CORPUS, key=lambda e: len(e["text"]))["id"]
    at_full = {e.get("id") for e in entries if e.get("n_tokens") == SEQ_LEN}
    if witness not in at_full:
        problems.append(
            f"`{witness}` is not at the full {SEQ_LEN}-token window (entries at {SEQ_LEN}: "
            f"{sorted(at_full) or 'none'}) — the truncation regime the fixed-length graph "
            f"exists for is not exercised by the entry that is supposed to exercise it"
        )

    if problems:
        raise SystemExit(
            f"CORPUS MISMATCH in {source} — refusing to proceed against it:\n  "
            + "\n  ".join(problems)
        )
    print(f"[ok] corpus identity: {len(entries)} entries, ordered (id, text) and shape "
          f"match the committed fixtures")


def verify_source_sha():
    """Fail-closed: assert every pinned source file matches its SHA-256 at REV.
    Aborts (SystemExit) on any mismatch so a wrong or corrupt snapshot cannot cut
    goldens or artifacts."""
    src = src_dir()
    for name, want in SOURCE_SHA256.items():
        path = os.path.join(src, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing pinned source file {name} under {src}")
        got = sha256_file(path)
        if got != want:
            raise SystemExit(
                f"SOURCE SHA-256 MISMATCH for {name}:\n  got  {got}\n  want {want}\n"
                f"  (snapshot {src} is not {MODEL_ID}@{REV[:12]})"
            )
    print(f"[ok] source SHA-256 verified against {MODEL_ID}@{REV[:12]} ({len(SOURCE_SHA256)} files)")


def assert_config_contract(config):
    """Assert the config fields the exported graph shape depends on. A silent
    upstream or transformers default drift here would change the graph without
    changing any code, so it is caught before an artifact is cut."""
    head_dim = config.hidden_size // config.num_attention_heads
    checks = [
        ("model_type", config.model_type, "modernbert"),
        ("hidden_size", config.hidden_size, EMBED_DIM),
        ("num_hidden_layers", config.num_hidden_layers, 12),
        ("num_attention_heads", config.num_attention_heads, 12),
        ("head_dim", head_dim, 32),
        ("intermediate_size", config.intermediate_size, 1536),
        ("vocab_size", config.vocab_size, 180000),
        ("norm_eps", float(config.norm_eps), 1e-5),
        ("norm_bias", config.norm_bias, False),
        ("attention_bias", config.attention_bias, False),
        ("mlp_bias", config.mlp_bias, False),
        ("hidden_activation", config.hidden_activation, "silu"),
        ("classifier_pooling", config.classifier_pooling, "cls"),
        ("local_attention", config.local_attention, 128),
        ("sliding_window", config.sliding_window, 64),
        ("global_attn_every_n_layers", config.global_attn_every_n_layers, 3),
        ("pad_token_id", config.pad_token_id, 179935),
        ("cls_token_id", config.cls_token_id, 179934),
    ]
    for name, got, want in checks:
        assert got == want, f"config contract drift: {name} = {got!r}, expected {want!r}"

    # Dual RoPE: global theta 150000 on the full-attention layers, local theta
    # 160000 on the sliding-window ones.
    rope = {lt: p["rope_theta"] for lt, p in config.rope_parameters.items()}
    assert rope == {"full_attention": 150000.0, "sliding_attention": 160000.0}, rope

    # Full attention on layers 0/3/6/9, sliding elsewhere.
    want_types = [
        "full_attention" if i % config.global_attn_every_n_layers == 0 else "sliding_attention"
        for i in range(config.num_hidden_layers)
    ]
    assert list(config.layer_types) == want_types, list(config.layer_types)
    print("[ok] config contract asserted (12x384 ModernBERT, dual RoPE 150000/160000, "
          "window 128 with full attention on 0/3/6/9, CLS pooling)")


def load_encoder():
    """Return the UNMODIFIED ``ModernBertModel`` (eval, fp32, eager attention)
    from the SHA-verified local snapshot.

    fp32 is a lossless upcast of the bf16 checkpoint. ``eager`` attention is the
    pin that makes the exported graph deterministic: it takes an ADDITIVE float
    mask, which is what the driver below supplies as a fixed-512 constant. Under
    ``sdpa`` transformers hands the layers a boolean mask (and elides it entirely
    when nothing is padded), so the traced graph would depend on the example."""
    from transformers import AutoModel

    verify_source_sha()
    model = AutoModel.from_pretrained(
        src_dir(), dtype=torch.float32, attn_implementation="eager"
    ).eval()
    assert_config_contract(model.config)
    return model


def load_sentence_transformer():
    """Return the canonical sentence-transformers pipeline (eval, fp32) — the
    ORACLE. Modules: Transformer -> Pooling(cls) -> Normalize, so its output is
    the UNIT-NORMALIZED embedding the committed goldens store.

    ``max_seq_length`` is forced to the export length: the checkpoint declares
    32768, but the CoreML contract is a fixed 512 window and the goldens are cut
    at 512."""
    from sentence_transformers import SentenceTransformer

    verify_source_sha()
    st = SentenceTransformer(src_dir(), device="cpu", model_kwargs={"dtype": torch.float32}).eval()
    st.max_seq_length = SEQ_LEN
    names = [type(m).__name__ for m in st]
    assert names == ["Transformer", "Pooling", "Normalize"], names
    pooling = st[1].get_config_dict()
    assert pooling["pooling_mode"] == "cls", (
        f"granite pools the CLS token, got {pooling['pooling_mode']!r} — a pooling-mode drift "
        "would silently change the oracle"
    )
    assert pooling["embedding_dimension"] == EMBED_DIM, pooling
    print(f"[ok] canonical pipeline: {' -> '.join(names)} @ max_seq_length {st.max_seq_length} "
          f"(pooling {pooling['pooling_mode']}, dim {pooling['embedding_dimension']})")
    return st


def assert_prompt_free(st):
    """granite r2 retrieval is PROMPT-FREE — ``config_sentence_transformers.json``
    carries empty query/document prompts. The corpus is fed raw, so a checkpoint
    that grew a non-empty prompt would silently shift every golden."""
    prompts = getattr(st, "prompts", {}) or {}
    non_empty = {k: v for k, v in prompts.items() if v}
    assert not non_empty, f"expected prompt-free retrieval, got prompts {non_empty}"
    assert getattr(st, "default_prompt_name", None) is None
    print("[ok] prompt-free (empty query/document prompts, no default prompt)")


def official_window_bool(model, seq_len=SEQ_LEN):
    """The pure sliding-window geometry as a ``[1, 1, S, S]`` bool tensor, taken
    from the checkpoint's OWN mask builder — not a reimplementation of ``|i-j|
    <= local_attention // 2``.

    ``create_bidirectional_sliding_window_mask`` returns ``window AND non_pad``
    (and elides the mask entirely when nothing is padded), so it is probed twice
    — once with the LAST position padded, once with the FIRST — and the two
    results OR-ed. Their non_pad vectors are True everywhere except one distinct
    column each, so the union is exactly ``window``."""
    from transformers.masking_utils import create_bidirectional_sliding_window_mask

    inputs_embeds = torch.zeros(1, seq_len, model.config.hidden_size)
    probes = []
    for pad_at in (seq_len - 1, 0):
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        attention_mask[0, pad_at] = 0
        m = create_bidirectional_sliding_window_mask(
            config=model.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
        if m is None:
            raise SystemExit("official sliding-window mask builder returned None for a padded input")
        probes.append((m == 0) if m.is_floating_point() else m.to(torch.bool))
    window = probes[0] | probes[1]
    assert tuple(window.shape) == (1, 1, seq_len, seq_len), tuple(window.shape)

    # Independent cross-check of the geometry the model card documents (+-64 at
    # local_attention 128). This does not REPLACE the official extraction above —
    # it proves the extraction did not silently degrade to something else.
    half = model.config.local_attention // 2
    idx = torch.arange(seq_len)
    expect = (idx[:, None] - idx[None, :]).abs() <= half
    assert torch.equal(window[0, 0], expect), "extracted window is not the documented +-64 band"
    print(f"[ok] official sliding window extracted: |i-j| <= {half} "
          f"({int(window[0, 0][seq_len // 2].sum())} keys per interior row)")
    return window


class GraniteGraph(torch.nn.Module):
    """The traced CoreML graph source: ``(input_ids, attention_mask)`` both
    ``[1, 512]`` -> ``embedding [1, 384]``, PRE-L2-norm.

    Every weight-bearing step is the STOCK ``ModernBertModel`` submodule
    (``embeddings``, each ``ModernBertEncoderLayer``, ``final_norm``) called
    unchanged. Only the two things transformers computes dynamically per call are
    hoisted to fixed-512 constants, which is what makes the graph static:

    * **RoPE** — ``cos``/``sin`` per layer type, produced by the checkpoint's own
      ``rotary_emb`` at positions ``0..511`` and held as fp32 buffers. The model
      card's "RoPE sin/cos precomputed as fp32 constants".
    * **attention masks** — the sliding-window geometry is a constant bool
      (``official_window_bool``); only the pad component depends on the runtime
      ``attention_mask`` input, so the additive masks are rebuilt in-graph as
      ``where(allowed, 0, MASK_BLOCK)``. Full-attention layers get the
      ``[1, 1, 1, S]`` pad-only mask; sliding layers get ``window AND non_pad``
      at ``[1, 1, S, S]``.

    L2 normalization is deliberately NOT in the graph — the Rust caller
    normalizes, which keeps the fp16 rsqrt guard class out of the artifact.
    """

    def __init__(self, encoder, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("window", official_window_bool(encoder, seq_len), persistent=False)
        self.layer_types = sorted(set(encoder.config.layer_types))
        position_ids = torch.arange(seq_len).unsqueeze(0)
        inputs_embeds = torch.zeros(1, seq_len, encoder.config.hidden_size)
        for lt in self.layer_types:
            cos_t, sin_t = encoder.rotary_emb(inputs_embeds, position_ids, lt)
            self.register_buffer(f"cos_{lt}", cos_t.to(torch.float32), persistent=False)
            self.register_buffer(f"sin_{lt}", sin_t.to(torch.float32), persistent=False)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder.embeddings(input_ids=input_ids)
        non_pad = attention_mask[:, None, None, :].to(torch.bool)
        allow = torch.zeros((), dtype=hidden.dtype)
        block = torch.full((), MASK_BLOCK, dtype=hidden.dtype)
        masks = {
            "full_attention": torch.where(non_pad, allow, block),
            "sliding_attention": torch.where(self.window & non_pad, allow, block),
        }
        rope = {lt: (getattr(self, f"cos_{lt}"), getattr(self, f"sin_{lt}"))
                for lt in self.layer_types}
        for layer in self.encoder.layers:
            hidden = layer(
                hidden,
                attention_mask=masks[layer.attention_type],
                position_embeddings=rope[layer.attention_type],
            )
        return self.encoder.final_norm(hidden)[:, 0]


def padded_inputs(tokenizer, text, pad_id, seq_len=SEQ_LEN):
    """The exact fixed-window ``(input_ids, attention_mask)`` the Rust embedder
    feeds the graph: tokenize with specials, truncate at ``seq_len``, right-pad
    the ids with ``pad_id`` and the mask with 0. Returns numpy int32 ``[1, S]``
    arrays plus the unpadded id list (the committed ``token_ids`` golden)."""
    ids = list(tokenizer(text, truncation=True, max_length=seq_len)["input_ids"])
    assert len(ids) <= seq_len, len(ids)
    input_ids = np.full((1, seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((1, seq_len), dtype=np.int32)
    input_ids[0, : len(ids)] = ids
    attention_mask[0, : len(ids)] = 1
    return input_ids, attention_mask, ids


def cos(a, b):
    """Cosine of two vectors in float64, NO epsilon guard — a non-finite artifact
    MUST propagate to NaN (fail-closed), never be masked to a finite-looking value."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def worst_update(worst, c):
    """Fold a cosine into a running worst, POISONING to NaN if either side is NaN.

    The ONLY correct way to accumulate a measured worst here. Plain ``min`` is
    silently wrong: ``nan < 1.0`` is False, so ``min(1.0, nan)`` returns 1.0 and
    a run whose every cosine was NaN would clear its floor. Once poisoned, the
    ``worst >= FLOOR`` comparison is False and the gate reds."""
    if worst != worst or c != c:
        return float("nan")
    return min(worst, c)


# The divergence budget the crosscheck reports against. It is a REPORTING
# threshold recorded in the golden, deliberately looser than DRIVER_FLOOR, which
# is the number the recipe actually gates on.
CROSSCHECK_STOP_DIVERGENCE = 1e-4


def driver_crosscheck(st, net):
    """Per-entry cosine of the PRE-TRACE driver against the UNMODIFIED canonical
    pipeline, over the whole committed corpus.

    Called ONLY by ``convert_granite.py``, which gates the conversion on the
    result and stages it; ``generate_goldens.py`` publishes that staged record
    rather than calling this again, so the committed golden is the measurement
    the conversion actually gated on. The driver emits the PRE-L2-norm CLS
    vector and the canonical
    pipeline emits the unit-normalized one; cosine is scale-invariant, so they
    are compared directly.

    The recorded cosines can sit a few ULP ABOVE 1.0 — float32 vectors that agree
    to ~7 significant digits, reduced in float64, land either side of the
    mathematical bound. That drift is real and is preserved rather than clamped.

    A cosine near 1.0 is NOT evidence that two computations were independent: a
    vector scored against ITSELF also lands at 1.0 up to reduction roundoff. So a
    positive distinctness statistic is recorded alongside it — the largest
    component difference between the unit-normalized driver vector and the
    canonical one — and a byte-identical pair aborts outright."""
    from _fixtures import CORPUS

    tokenizer = st.tokenizer
    pad_id = st[0].auto_model.config.pad_token_id
    per_entry = []
    worst = 1.0
    min_delta = float("inf")
    for entry in CORPUS:
        input_ids, attention_mask, _ids = padded_inputs(tokenizer, entry["text"], pad_id)
        with torch.no_grad():
            driven = net(
                torch.from_numpy(input_ids).to(torch.long),
                torch.from_numpy(attention_mask).to(torch.long),
            ).numpy()
            canonical = st.encode([entry["text"]], convert_to_numpy=True,
                                  normalize_embeddings=True, batch_size=1)[0]
        driven = np.asarray(driven, np.float64).ravel()
        canonical = np.asarray(canonical, np.float64).ravel()
        if np.array_equal(driven, canonical):
            raise SystemExit(
                f"SELF-COMPARISON on `{entry['id']}`: the driver output is byte-identical to the "
                f"canonical output. The crosscheck is comparing one computation with itself and "
                f"proves nothing; the driver is not being exercised."
            )
        unit = driven / np.linalg.norm(driven)
        delta = float(np.max(np.abs(unit - canonical)))
        if delta < DISTINCTNESS_FLOOR:
            raise SystemExit(
                f"DEGENERATE CROSSCHECK on `{entry['id']}`: the normalized driver vector differs "
                f"from the canonical one by at most {delta:.3e}, under the {DISTINCTNESS_FLOOR:.0e} "
                f"floor. That is the signature of the driver returning the canonical vector (or a "
                f"rescaling of it) rather than computing its own; the crosscheck would prove "
                f"nothing."
            )
        c = cos(driven, canonical)
        per_entry.append({
            "id": entry["id"],
            "cosine_canonical_vs_driver": c,
            "max_abs_component_delta": delta,
        })
        worst = worst_update(worst, c)
        min_delta = min(min_delta, delta)
    return {
        "worst_cosine_canonical_vs_driver": worst,
        "min_max_abs_component_delta": min_delta,
        "stop_threshold_divergence": CROSSCHECK_STOP_DIVERGENCE,
        "verdict": "AGREE" if (1.0 - worst) <= CROSSCHECK_STOP_DIVERGENCE else "DIVERGE",
        "per_entry": per_entry,
    }
