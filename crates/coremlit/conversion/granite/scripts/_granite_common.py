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
import re
import shlex
import shutil
import subprocess
import uuid

import numpy as np
import torch

MODEL_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"
REV = "835ad14087e140460703cf0fae09f97d469d65c2"

# Per-source-file SHA-256 at REV (verified 2026-07-26). model.safetensors and
# tokenizer.json are the load-bearing weight/tokenizer identities. tokenizer.json
# is ALSO the file this recipe STAGES INTO the published artifact (the crate no
# longer embeds it: `TextEmbedder::load` reads it from beside the .mlmodelc and
# checks it against `contract::TOKENIZER_SHA256_HEX`, which is this same digest),
# and it is asserted in tests/granite/tokenizer_identity.rs. The JSON configs are
# pinned because the graph shape (layer_types, rope thetas, local_attention,
# norm_eps) and the pooling/prompt contract are read from them, not hardcoded
# here.
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
# `tests/granite/model_io.rs` applies to the bundle). The model card and the
# tokenizer are part of the distributed set but are staged, never generated here;
# the two generated names below them are this recipe's own outputs and are
# excluded from discovery.
MODEL_CARD = "README.md"
# The runtime tokenizer sidecar. It ships WITH the model because the Rust crate
# stopped embedding it (a 24 MB include_bytes! in a crates.io package), so
# `TextEmbedder::load` reads `<artifact root>/tokenizer.json` beside the
# .mlmodelc. Dropping it from the artifact breaks every default constructor, so
# it belongs in the exact-set gate and in CHECKSUMS.sha256 like any other
# distributed file. Its bytes are the pinned SOURCE tokenizer, copied unmodified:
# SOURCE_SHA256["tokenizer.json"] IS the identity the crate enforces at load.
TOKENIZER_FILE = "tokenizer.json"
CHECKSUMS_FILE = "CHECKSUMS.sha256"
MANIFEST_FILE = "MANIFEST.json"
GENERATED_AT_ROOT = (CHECKSUMS_FILE, MANIFEST_FILE)
FP32_REFERENCE = f"{MODEL_STEM}_fp32.mlpackage"
SHIPPED_PACKAGE = f"{MODEL_STEM}.mlpackage"
SHIPPED_BUNDLE = f"{MODEL_STEM}.mlmodelc"
FP32_BUNDLE = f"{MODEL_STEM}_fp32.mlmodelc"
VERIFY_METRICS = "verify_metrics.json"
PRODUCER_RECORD = "producer.json"
STAGED_CROSSCHECK = "driver_crosscheck.json"
COMPILE_RECORD = "compile.json"

# The two committed goldens, written and consumed as a PAIR: the crosscheck names
# the corpus bytes it was published beside, so neither half says anything
# trustworthy about the other's contents on its own.
GOLDEN_CORPUS = "corpus.json"
GOLDEN_CROSSCHECK = "driver_crosscheck.json"

# Every ``(mlpackage -> mlmodelc)`` pair the compile step produces, named ONCE so
# the step that compiles them, the run-start invalidation that removes them, and
# the record that binds them cannot drift apart.
COMPILED_PAIRS = (
    (SHIPPED_PACKAGE, SHIPPED_BUNDLE),
    (FP32_REFERENCE, FP32_BUNDLE),
)

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
EXPECTED_ARTIFACT_FILES = sorted(
    EXPECTED_BUNDLE_FILES + [f"./{MODEL_CARD}", f"./{TOKENIZER_FILE}"]
)

# The additive attention-mask "blocked" value. LOAD-BEARING, and NOT
# ``torch.finfo(dtype).min``: the shipped artifact is fp16, and -3.4e38 overflows
# to -inf in coremltools' fp16 cast. A fully-padded query row is then all -inf,
# whose softmax is NaN — measured on the CpuOnly backend as 15/16 corpus entries
# returning non-finite output, while CpuAndGpu/CpuAndNeuralEngine/All stayed clean (they
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
    them and the remedy instead.

    The remedies are built with ``shlex.join`` for the reason spelled out in
    ``assert_no_stale_publication_temps``: an unquoted root containing a space
    silently splits into two arguments in whatever the operator pastes. The
    ``find`` branches are parenthesised so ``-delete`` applies to BOTH names —
    unparenthesised, ``-o`` binds looser and only ``.DS_Store`` is removed.

    The root is made ABSOLUTE, which is a DIFFERENT fix from quoting. A relative
    root beginning with ``-`` survives ``shlex`` intact and is then read as an
    option by the tool receiving it (measured: ``dot_clean: invalid option -- a``,
    ``find: illegal option -- a``). Every absolute path starts with ``/`` and can
    never be an option. ``--`` is passed as well; both tools were verified
    non-vacuously to accept it and still act on the operand."""
    sidecars = [
        rel for rel in discovered
        if rel.rsplit("/", 1)[-1].startswith("._") or rel.rsplit("/", 1)[-1] == ".DS_Store"
    ]
    if sidecars:
        raise SystemExit(
            f"OS SIDECAR FILES under {root} — refusing to publish a tree containing them:\n"
            f"  {sidecars}\n"
            f"  These appear when the tree lives on exFAT/FAT/SMB. Remove them and re-run;\n"
            f"  staging onto a native APFS/HFS+ volume avoids them entirely.\n"
            f"    {shlex.join(['dot_clean', '--', os.path.abspath(root)])}\n"
            f"    {shlex.join(['find', '--', os.path.abspath(root), '(', '-name', '._*',
                               '-o', '-name', '.DS_Store', ')', '-delete'])}"
        )


def assert_no_stale_publication_temps(root, discovered):
    """Fail-closed on a temp left behind by an interrupted publication.

    These appear when a publication write, a model-card copy or a tokenizer copy
    is hard-killed between creating the temp and renaming it — no handler runs, so the temp
    survives. It can never be adopted (every generated name is unique), but it
    fails the exact-file-set gate on every retry, and that gate says only
    "unexpected", which sends the operator looking for a fault in the artifact.

    This REPORTS and refuses. It does NOT delete, and that is a deliberate
    reversal: an earlier version swept these automatically and was wrong three
    times about which names only this recipe could have produced. Prefix matching
    removed ``README.md.notes.tmp``; a ``<32 hex>`` match still removed
    ``README.md.00000000000000000000000000000000.tmp``, which ``uuid4`` cannot
    emit because it pins the version nibble to ``4`` and the variant to
    ``[89ab]``; and a UUIDv4 grammar would still be a GUESS about provenance.
    Each iteration narrowed an irreversible blast radius without eliminating it,
    and what it destroys is someone else's file.

    Removing files under someone's tree is not this recipe's call — the same
    judgement ``assert_no_os_sidecars`` already makes for AppleDouble files. So
    the pattern below chooses only WHICH refusal the operator reads: a name it
    misjudges still stops at the exact-set gate, and nothing is destroyed either
    way.

    The suggested command is built with ``shlex.join``, not string concatenation.
    A configured root of ``/tmp/work tree`` otherwise emits ``rm /tmp/work
    tree/...``, which hands ``rm`` two operands and deletes ``/tmp/work``. A path
    is data and a command line is code; that is what ``shlex`` is for.

    The paths are also made ABSOLUTE, and ``--`` guards a leading ``-``. These are
    two separate hazards that both present as "the path broke the command":
    quoting (a space splits one operand into two) and option parsing (a relative
    root beginning with ``-`` is read as a flag). ``shlex`` fixes only the first —
    a leading-dash path round-trips through it perfectly and is still misread."""
    generated_tmp = re.compile(
        "|".join(rf"{re.escape(n)}\.[0-9a-f]{{32}}\.tmp"
                 for n in tuple(GENERATED_AT_ROOT) + (MODEL_CARD, TOKENIZER_FILE))
    )
    stale = [rel for rel in discovered if generated_tmp.fullmatch(rel.rsplit("/", 1)[-1])]
    if stale:
        raise SystemExit(
            f"STALE PUBLICATION TEMP under {root} — refusing to publish over a tree that still "
            f"holds a half-finished write:\n"
            f"  {stale}\n"
            f"  A publication, model-card or tokenizer copy was killed between writing the "
            f"temp and renaming it. Nothing here removes them: this recipe does not delete "
            f"files it cannot prove it created.\n"
            f"  Confirm they are not yours, remove them, and re-run:\n"
            f"    {shlex.join(['rm', '--',
                               *(os.path.abspath(os.path.join(root, rel[2:]))
                                 for rel in stale)])}"
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
    assert_no_stale_publication_temps(root, got)
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


def evidence_digests(root, stage, goldens):
    """The bytes ``verify_granite.py``'s numbers were measured from, and the
    goldens they were measured against.

    Every part matters and together they cover the whole matrix: the per-unit
    fp16 arms run the staged ``.mlmodelc`` under ``root`` (a byte copy of the
    staging one), the I/O contract reads the staged fp16 ``.mlpackage``, the fp32
    oracle arm runs the fp32 reference package — which lives ONLY in staging and
    is never shipped, so it is digested from there — and every cosine is scored
    against ``corpus.json``, which is itself regenerable. Without the corpus
    digest, regenerating the oracle would leave a manifest describing comparisons
    against goldens that no longer exist under that name.

    ``driver_crosscheck.json`` is digested for exactly the same reason, and it was
    the gap: it is the corpus's PAIR, and until it was bonded here NO later stage
    read or digested it. A regeneration interrupted between the two renames could
    therefore leave the corpus from one run beside the crosscheck from another,
    and verification, the manifest and the checksums would every one of them
    attest successfully to a pair the Rust gate rejects."""
    return {
        "artifact": digest_files(root, EXPECTED_BUNDLE_FILES),
        "fp32_reference": digest_tree(os.path.join(stage, FP32_REFERENCE)),
        "corpus": sha256_file(os.path.join(goldens, GOLDEN_CORPUS)),
        "crosscheck": sha256_file(os.path.join(goldens, GOLDEN_CROSSCHECK)),
    }


def fsync_dir(path):
    """Push ``path``'s own changes — the renames and unlinks made IN it — out of
    host memory. Best effort, and NOT a power-loss ordering primitive.

    What it does. A rename is a change to the DIRECTORY rather than to the file,
    so a directory whose new entries live only in the buffer cache is a directory
    a kernel panic loses. ``fsync`` hands them to the drive, which bounds that
    window.

    What it does NOT do, spelled out because both the name and the placement
    suggest otherwise: order two renames across a POWER loss. macOS ``fsync(2)``
    says outright that once the data reaches the drive, the drive "may not
    physically write the data to the platters for quite some time and it may be
    written in an out-of-order sequence", so "later writes may be present, while
    earlier writes are not" — and adds "This is not a theoretical edge case."
    Ordering there needs
    ``F_FULLFSYNC``/``F_BARRIERFSYNC``, and even those would not make a
    multi-rename, multi-directory publication recover as a unit. No caller may
    read this as a crash-consistency barrier. The publication invariants this
    recipe does hold are PROCESS-crash invariants, and ``os.replace``'s atomicity
    gives those on its own.

    Fail-closed like every other precondition here. A directory whose sync fails
    is reporting a real I/O problem on the volume being published to."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path):
    """Push ``path``'s CONTENTS out of host memory. Same scope as ``fsync_dir``:
    best effort, not a power-loss primitive.

    A file is not on the drive because ``write`` or ``shutil.copyfile`` returned.
    Both places here that rename a temp into place flush it first —
    ``write_manifest.stage_model_card`` through this helper, and
    ``replace_file_atomic`` inline — so neither reads as an oversight beside the
    other."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(base):
    """Push every file and directory under ``base`` out of host memory.

    A copied tree is not on the drive because ``shutil.copytree`` returned, and
    promotion renames it into the artifact root. Files first, then their
    directory, matching ``fsync_dir``'s contract — and carrying ``fsync_dir``'s
    scope: this bounds what a kernel panic can lose and claims nothing about a
    power cut."""
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            fsync_file(os.path.join(dirpath, name))
        fsync_dir(dirpath)


def replace_file_atomic(path, text):
    """Serialize ``text`` COMPLETELY to a unique temp file, then rename it into place.

    A reader either sees the previous complete file or the new complete one; no
    reader ever sees a prefix. The whole payload is passed in rather than
    streamed, so an interrupted writer cannot leave a plausible SUBSET — the
    failure mode that matters for a checksum list, where a truncated file is
    still one ``shasum -c`` accepts.

    The temp name is unique, so a concurrent or previously-killed writer cannot
    have its partial file adopted by this one, and it is removed on any failure
    rather than left beside the artifact.

    The all-or-nothing property comes from the RENAME, not from the fsyncs: a
    process killed at any point leaves either the previous complete file or the
    new complete one, because ``os.replace`` is atomic and the page cache outlives
    the process. The two fsyncs — contents before the rename, parent directory
    after it — are the best-effort flush ``fsync_dir`` describes, and neither is a
    power-loss guarantee."""
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    parent = os.path.dirname(os.path.abspath(path))
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(parent)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def discard_file(path):
    """Remove ``path`` if present. Used to invalidate a previous run's evidence
    before a new run starts, so a failing run cannot leave the old verdict
    standing as though it described the new build."""
    if os.path.exists(path):
        os.remove(path)


def discard_tree(path):
    """Remove directory ``path`` and everything under it, if present. The
    directory-shaped counterpart of ``discard_file`` — CoreML artifacts are
    directories, so invalidating one is a tree removal."""
    if os.path.isdir(path):
        shutil.rmtree(path)


def discard_publication_outputs(root):
    """Remove the previous ``CHECKSUMS.sha256`` and ``MANIFEST.json``.

    Two exact names, nothing else. This is not a sweeper: stale temps are
    REPORTED by ``assert_no_stale_publication_temps`` and never deleted, because
    a recovery step that guesses which files it created eventually guesses wrong
    and the loss is irreversible.

    Call this IMMEDIATELY BEFORE publication — after every precondition has
    passed and after both payloads are fully built — never on entry. Invalidating
    on entry made the independently-runnable writer destroy a valid attestation
    whenever any precondition failed: an unset ``GRANITE_GOLDENS`` or a mistyped
    stage path removed both files and only then aborted, and the sync below
    pushed that loss straight out to the drive. A failed run must leave an
    existing valid publication exactly as it found it.

    The pipeline still clears both at the other genuine mutation point, by a
    different mechanism and not through this function: ``stage_artifact.py``
    renames them out of the root into its scratch, in the phase before it
    replaces the bundles. So a staging step that succeeds followed by a
    verification that fails cannot leave a complete attestation standing over
    bytes that are no longer there.

    The removals are fsynced on the same best-effort terms as every other flush
    here — see ``fsync_dir``. What keeps a failed run non-destructive is WHERE
    this is called from, not the flush."""
    if not os.path.isdir(root):
        return
    removed = [name for name in GENERATED_AT_ROOT if os.path.exists(os.path.join(root, name))]
    for name in removed:
        os.remove(os.path.join(root, name))
    if removed:
        fsync_dir(root)
        print(f"[ok] invalidated the previous publication under {root}: {removed}")


def begin_run(stage):
    """Open a new conversion run: mint its id and INVALIDATE every attestation and
    every compiled bundle a previous run left in ``stage``.

    This happens before any failure-prone work. Conversion loads the checkpoint,
    checks the driver and traces — any of which can fail — and if a previous
    run's producer record, verification and crosscheck were still lying around,
    a failed conversion would leave them intact and downstream stages would
    happily certify the packages they describe. Invalidating first means a failed
    run leaves nothing that can be mistaken for a result.

    The staged ``.mlmodelc`` bundles go for a sharper reason: compilation happens
    AFTER ``producer.json`` is written, and that record binds only the two
    ``.mlpackage``s. A bundle left by an earlier run therefore satisfies every
    later check — the staging copy makes root and staging equal, and the manifest
    stamps THIS run's identity onto compiled bytes another run produced. Deleting
    them means a resume that skips compilation fails loudly at the copy instead."""
    for name in (PRODUCER_RECORD, VERIFY_METRICS, STAGED_CROSSCHECK, COMPILE_RECORD):
        discard_file(os.path.join(stage, name))
    for _package, bundle in COMPILED_PAIRS:
        discard_tree(os.path.join(stage, bundle))
    run_id = uuid.uuid4().hex
    print(f"[ok] run {run_id[:12]} started; prior producer/verification/crosscheck/compilation "
          f"records and every staged .mlmodelc discarded")
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


def observed_compiler():
    """The toolchain that turns an ``.mlpackage`` into an ``.mlmodelc``, read from
    what is ACTUALLY installed.

    README records that the residual run-to-run instability in this recipe is the
    CoreML compiler and the package UUID, not the conversion — the compiled
    ``coremldata.bin`` files move between two runs whose ``model.mil`` and
    ``weight.bin`` are identical. Compiled bytes are therefore only attributable
    if the compiler that emitted them is recorded beside them; ``producer.json``
    describes torch and coremltools, neither of which compiled anything.

    Every probe is fail-closed. The same ``xcrun`` that answers them is the one
    that compiles, so a probe that cannot run means the compile could not have
    either, and a record naming no compiler must not be written."""
    def probe(argv):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            raise SystemExit(
                f"COMPILER PROBE FAILED for `{' '.join(argv)}`: {e}\n"
                f"  The compilation record must name the toolchain that produced the bundle. "
                f"Install the Xcode command line tools and re-run."
            ) from e
        return done.stdout.strip()

    return {
        "coremlcompiler": probe(["xcrun", "coremlcompiler", "version"]),
        "developer_dir": probe(["xcode-select", "-p"]),
        "macos_product_version": probe(["sw_vers", "-productVersion"]),
        "macos_build_version": probe(["sw_vers", "-buildVersion"]),
    }


def compiled_digests(stage):
    """What each ``.mlmodelc`` in ``stage`` was compiled FROM and what it is, read
    from disk right now — one entry per ``COMPILED_PAIRS`` member."""
    out = {}
    for package, bundle in COMPILED_PAIRS:
        for sub in (package, bundle):
            path = os.path.join(stage, sub)
            if not os.path.isdir(path):
                raise SystemExit(
                    f"missing {path} — run convert_granite.py then compile_granite.py, in order"
                )
        out[bundle] = {
            "input_package": package,
            "input_package_sha256": digest_tree(os.path.join(stage, package)),
            "output_bundle_sha256": digest_tree(os.path.join(stage, bundle)),
        }
    return out


def write_compile_record(stage, run_id, compiled):
    """Record WHICH run compiled the staged bundles, from WHICH packages, under
    WHICH compiler.

    ``producer.json`` binds only the two ``.mlpackage``s, because compilation
    happens after it is written. Without this second record the compiled bundle —
    which is what actually ships and what every Rust gate loads — has no
    run-bound provenance at all, and the equality checks downstream all pass on a
    bundle that some earlier run produced."""
    record = {
        RUN_ID_KEY: run_id,
        "compiled_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "compiler": observed_compiler(),
        "bundles": compiled,
    }
    replace_file_atomic(os.path.join(stage, COMPILE_RECORD),
                        json.dumps(record, indent=2) + "\n")
    print(f"[ok] recorded compilation for run {run_id[:12]} over {len(compiled)} bundles")
    return record


def require_compile_record(stage, run_id):
    """Fail-closed: REQUIRE a run-bound compilation record that still describes the
    packages and bundles on disk.

    This is what makes the compiled bytes attributable. Its absence means the
    compile step did not run for this conversion; a foreign ``run_id`` means the
    bundle came from another run; a digest mismatch means the packages were
    re-converted after compiling, or a bundle from a different build is staged."""
    path = os.path.join(stage, COMPILE_RECORD)
    if not os.path.exists(path):
        raise SystemExit(
            f"MISSING COMPILATION RECORD: {path} does not exist.\n"
            f"  Run compile_granite.py for this conversion. producer.json binds only the two "
            f"mlpackages, so without this record the compiled bundle — the thing that actually "
            f"ships — carries no evidence that this run built it from these packages."
        )
    with open(path) as f:
        record = json.load(f)
    require_run_id(record, run_id, "the compilation record", path)
    if not record.get("compiler"):
        raise SystemExit(
            f"COMPILATION RECORD ({path}) names no compiler environment; re-run "
            f"compile_granite.py."
        )
    got = compiled_digests(stage)
    if record.get("bundles") != got:
        raise SystemExit(
            f"COMPILATION RECORD IS STALE ({path}): it does not describe the packages and "
            f"compiled bundles now in {stage}.\n"
            f"  Either the packages were re-converted without recompiling, or a bundle from "
            f"another build is staged. Re-run compile_granite.py."
        )
    return record


def corpus_input_sha256(pairs):
    """SHA-256 over the ORDERED ``(id, text)`` inputs a measurement consumed.

    This is the digest that makes a measurement's INPUT checkable. The published
    ``corpus_sha256`` hashes the serialized ``corpus.json`` bytes, which do not
    exist while the crosscheck is being measured — it binds a publication pair
    and nothing more. This one is computed FROM the inputs as they are consumed,
    so the same value is reproducible by anyone holding the same fixtures, and a
    later step can prove it is writing goldens for the corpus that was measured
    rather than relabelling an older measurement.

    Order-sensitive and separator-fixed on purpose: a reordered corpus is a
    different ordered input, and an ``indent``/``ensure_ascii`` change in some
    other serializer must not move this digest."""
    payload = json.dumps([[i, t] for i, t in pairs], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
            "toolchain", "compile", RUN_ID_KEY}
    for unit in COMPUTE_UNIT_NAMES:
        keys.add(f"fp16_{unit}_vs_fp32")
        keys.add(f"fp16_{unit}_nonfinite_entries")
    return keys


# Non-numeric verifications whose success must be recorded explicitly. A metrics
# file that merely omits them must not read as "nothing failed".
REQUIRED_CHECKS = ("io_contract", "tokenizer_identity", "corpus_identity", "golden_pair")


def require_published_crosscheck(goldens):
    """Fail-closed: the two published goldens must actually be a PAIR.

    Takes no run id on purpose. There is no sound unconditional run-identity
    check here — a plain replay legitimately scores against a committed pair from
    an earlier conversion — and the conditional one this used to carry is exactly
    what made the check evadable. The bindings below hold regardless of which run
    cut the pair.

    ``generate_goldens.py`` puts ``corpus.json`` and ``driver_crosscheck.json``
    down as two renames, and nothing downstream used to read the crosscheck at
    all — it was absent from ``evidence_digests`` and neither verification nor the
    manifest step referenced it. So a regeneration interrupted between those two
    renames left the corpus from one run beside the crosscheck from another, and
    verification, ``MANIFEST.json`` and ``CHECKSUMS.sha256`` all attested
    successfully to a pair that ``tests/granite/driver_crosscheck.rs`` rejects.
    That is a failed run leaving a stale file — inside this recipe's stated
    boundary, reachable with no tampering — so it is checked here.

    ``corpus_sha256`` is REQUIRED and must name the corpus bytes on disk. That is
    the binding that catches the split pair.

    ``corpus_input_sha256`` is REQUIRED UNCONDITIONALLY, and that is load-bearing.
    An earlier version demanded it only when the crosscheck's ``run_id`` matched
    the run being verified, reasoning that a plain replay legitimately scores
    against a committed pair from an earlier run. That exemption is evadable
    exactly where it matters: the committed pair carries a ``run_id`` and no
    digest, so on ANY fresh run the ids differ AND the field is absent — neither
    condition fires, and the pair is accepted. It also accepts a record the old
    generator republished over an EDITED corpus, because that generator stamped
    the new ``corpus_sha256`` on while leaving the measurement untouched. A
    conditional guard whose condition is false in the common case is not a guard.

    The consequence is accepted rather than worked around: the pair committed to
    this repository predates the field, so verification REFUSES to attest until it
    is regenerated by a conversion run. Loudly unrunnable is the correct failure;
    the alternative is quietly certifying a pair that nothing ties to its
    measurement.

    What the two digests together CATCH: a stale pair, a mismatched pair, a
    regeneration interrupted between the two renames, and a fixture edit published
    without re-running the conversion. Those are mistakes and failed runs, which is
    what this recipe's gates claim.

    What they do NOT catch: a digest computed from the corpus it is stamped beside
    and written into the record by hand. It matches by construction, and nothing
    here can distinguish it from one a conversion recorded. That is circumvention
    — editing a fixture specifically to defeat the refusal above — not a mistake
    or a failed run, and it sits outside this recipe's boundary alongside the two
    scenarios README.md already declines to claim. Binding a record to a real
    conversion of these exact texts would need either re-deriving the measurement
    (a conversion run) or a signed attestation; neither exists here, and neither is
    claimed."""
    corpus_path = os.path.join(goldens, GOLDEN_CORPUS)
    path = os.path.join(goldens, GOLDEN_CROSSCHECK)
    if not os.path.exists(path):
        raise SystemExit(
            f"MISSING PUBLISHED CROSSCHECK: {path} does not exist.\n"
            f"  It is the corpus's pair and the conversion's faithfulness evidence; an artifact "
            f"must not be verified or published against a corpus whose crosscheck is gone."
        )
    with open(path) as f:
        record = json.load(f)

    problems = []
    want_pair = sha256_file(corpus_path)
    if record.get("corpus_sha256") != want_pair:
        problems.append(
            f"it was published beside a corpus.json hashing to "
            f"{record.get('corpus_sha256')!r}, but {GOLDEN_CORPUS} here hashes to {want_pair!r}"
        )
    measured = record.get("corpus_input_sha256")
    if not measured:
        problems.append(
            "it carries no `corpus_input_sha256`, so NOTHING ties the measurement it records to "
            "the corpus it is published with — the record could have been measured over any "
            "other text and republished here"
        )
    else:
        with open(corpus_path) as f:
            entries = json.load(f)["entries"]
        want_inputs = corpus_input_sha256([(e.get("id"), e.get("text")) for e in entries])
        if measured != want_inputs:
            problems.append(
                f"it measured ordered inputs {measured!r}, but {GOLDEN_CORPUS} here holds "
                f"{want_inputs!r}"
            )

    if problems:
        raise SystemExit(
            f"SPLIT GOLDEN PAIR ({goldens}) — refusing to attest to goldens that do not belong "
            f"together:\n  " + "\n  ".join(problems) + "\n"
            f"  REMEDY: regenerate BOTH goldens in one run — `GRANITE_REGEN_GOLDENS=1 bash "
            f"crates/coremlit/conversion/granite/run_granite.sh`.\n"
            f"  That needs a full CONVERSION run (the goldens step PUBLISHES the conversion's "
            f"crosscheck, it does not compute one), which is owner-gated.\n"
            f"  If the missing field is `corpus_input_sha256`: the pair committed to this "
            f"repository predates it, so it fails here until it is regenerated. That is "
            f"deliberate."
        )
    print("[ok] published golden pair: the crosscheck names this corpus")
    return record


def require_verify_evidence(root, stage, goldens, run_id):
    """Fail-closed: REQUIRE complete, passing, artifact-bound verification evidence.

    Downstream steps must not treat missing evidence as permission to proceed.
    This asserts, in order: the file exists; its key set is exactly
    ``expected_metric_keys()``; the recorded floors equal the constants in force
    now (so evidence cut under loosened floors cannot certify); every worst is
    finite and clears its floor; every non-finite counter is zero; the recorded
    compilation record is the one still on disk for this run; and the recorded
    digests still match the bytes on disk, so metrics from an earlier run cannot
    certify a different bundle."""
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
        # The compiled bundle is what ships, and it is not covered by the
        # producer record. Re-read its compilation record here rather than trust
        # the copy inside the metrics: a bundle recompiled after verification
        # would leave the metrics' copy intact while the record on disk moved.
        if metrics["compile"] != require_compile_record(stage, run_id):
            problems.append(
                "the compilation record inside the evidence is not the one on disk — the bundle "
                "was recompiled after it was verified"
            )
        # The goldens are the oracle every number above was scored against, and
        # they are a pair. Re-validate the pair here as well as in verification:
        # the digests below catch a golden that MOVED, this catches one that was
        # never consistent with its partner in the first place.
        require_published_crosscheck(goldens)
        got = evidence_digests(root, stage, goldens)
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
    canonical one — and a byte-identical pair aborts outright.

    ``corpus_input_sha256`` records the ORDERED ``(id, text)`` inputs this
    measurement consumed. It is the one field here computed from the measurement
    input rather than stamped on afterwards, and it is what stops a later
    goldens-only re-run from republishing this record over a different corpus."""
    from _fixtures import CORPUS

    tokenizer = st.tokenizer
    pad_id = st[0].auto_model.config.pad_token_id
    per_entry = []
    consumed = []
    worst = 1.0
    min_delta = float("inf")
    for entry in CORPUS:
        # Accumulated from the entries this loop actually feeds the two models,
        # so the recorded digest describes the measurement's input rather than
        # some other reading of the fixture module.
        consumed.append((entry["id"], entry["text"]))
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
        "corpus_input_sha256": corpus_input_sha256(consumed),
        "stop_threshold_divergence": CROSSCHECK_STOP_DIVERGENCE,
        "verdict": "AGREE" if (1.0 - worst) <= CROSSCHECK_STOP_DIVERGENCE else "DIVERGE",
        "per_entry": per_entry,
    }
