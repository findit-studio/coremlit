"""Fail-closed conversion-verification matrix for the granite artifact.

  (0) PRECONDITIONS — the producer record and the run-bound COMPILATION record
      must both describe the packages and bundles on disk, and the loaded
      ``corpus.json`` must be exactly the committed fixture set (same ids, same
      count, no duplicates), all asserted before any model is loaded. Every
      measurement below is a fold seeded at 1.0, so an empty or truncated corpus
      would otherwise clear every floor having made no prediction at all; and
      the fp16 matrix runs the COMPILED bundle, which ``producer.json`` does not
      cover, so without the compilation record these numbers could be measured
      from a bundle an earlier run built. The two committed goldens are checked
      as a PAIR too — the crosscheck must name the corpus bytes beside it —
      because an interrupted regeneration otherwise leaves the corpus from one
      run next to the crosscheck from another, and every attestation below still
      succeeds over goldens the Rust gate rejects.
  (a) I/O CONTRACT — the shipped fp16 bundle's input/output names, dtypes, and
      shapes must be EXACTLY ``input_ids`` int32 [1,512] + ``attention_mask``
      int32 [1,512] -> ``embedding`` fp32 [1,384]. A contract drift is a hard
      failure; every Rust gate reads this shape from the artifact.
  (b) ORACLE — CoreML fp32 on CPUOnly vs the COMMITTED transformers-fp32
      goldens (``$GRANITE_GOLDENS/corpus.json``, unit-normalized; cosine is
      scale-invariant so the pre-L2-norm graph output compares directly), over
      all 16 entries -> worst cosine, floor ``FP32_FLOOR``.
  (c) PRECISION x COMPUTE-UNIT REQUEST — CoreML fp16 vs CoreML fp32-CPUOnly on EVERY
      compute unit {CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All} -> worst cosine,
      floor ``FP16_FLOOR`` on each, and ZERO non-finite outputs on each. Unlike
      siglip, no arm here is merely characterized: granite ships fp16 on the
      default planner, so every arm is gated.

FAIL-CLOSED: a non-finite cosine POISONS worst to NaN (fails the floor); any
load/predict exception is a HARD failure; any breach exits NONZERO so
run_granite.sh (set -e) halts. A breach is a FINDING, never a floor to loosen.

REPLAYABILITY SCOPE: clearing this matrix proves the re-derived artifact is
EQUIVALENT to the published one to the measured floors. It is not a claim of
byte-identity — ``write_manifest.py`` records the produced hashes so a
``shasum -a 256 -c`` against a published CHECKSUMS.sha256 can be run and reported
separately.
"""
import json
import os
import sys

import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from _granite_common import (
    COMPUTE_UNIT_NAMES,
    EMBED_DIM,
    FP16_FLOOR,
    FP32_FLOOR,
    MODEL_STEM,
    SEQ_LEN,
    REQUIRED_CHECKS,
    RUN_ID_KEY,
    VERIFY_METRICS,
    assert_corpus_identity,
    assert_staged_matches_staging,
    cos,
    digest_tree,
    discard_file,
    evidence_digests,
    load_sentence_transformer,
    model_root,
    observed_toolchain,
    padded_inputs,
    read_producer_record,
    replace_file_atomic,
    require_compile_record,
    require_producer_toolchain,
    require_published_crosscheck,
    stage_dir,
    worst_update,
)
from _fixtures import goldens_dir

# Keyed off COMPUTE_UNIT_NAMES so the arms measured here and the arms the
# evidence gate demands cannot drift apart.
_CT_UNITS = {
    "CpuOnly": ct.ComputeUnit.CPU_ONLY,
    "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
    "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "All": ct.ComputeUnit.ALL,
}
UNITS = {name: _CT_UNITS[name] for name in COMPUTE_UNIT_NAMES}

# CoreML ``ArrayFeatureType.ArrayDataType`` enum values for the contract assert.
ARRAY_INT32 = 131104
ARRAY_FLOAT32 = 65568


def check(failures, label, worst, floor):
    ok = bool(worst >= floor)
    print(f"    [{'OK' if ok else 'FAIL'}] {label}: worst {worst:.8f}  (floor {floor})")
    if not ok:
        failures.append(f"{label}: worst {worst:.8f} < floor {floor} (or non-finite)")
    return ok


def assert_contract(model, failures):
    spec = model.get_spec()
    got_in = {i.name: (i.type.multiArrayType.dataType, list(i.type.multiArrayType.shape))
              for i in spec.description.input}
    got_out = {o.name: (o.type.multiArrayType.dataType, list(o.type.multiArrayType.shape))
               for o in spec.description.output}
    want_in = {"input_ids": (ARRAY_INT32, [1, SEQ_LEN]),
               "attention_mask": (ARRAY_INT32, [1, SEQ_LEN])}
    want_out = {"embedding": (ARRAY_FLOAT32, [1, EMBED_DIM])}
    for label, got, want in (("inputs", got_in, want_in), ("outputs", got_out, want_out)):
        if got != want:
            failures.append(f"I/O contract {label} mismatch: got {got}, want {want}")
            print(f"    [FAIL] I/O contract {label}: {got}")
        else:
            print(f"    [OK] I/O contract {label}: {got}")


def main():
    stage = stage_dir()
    root = model_root()
    failures = []
    metrics = {}
    checks = dict.fromkeys(REQUIRED_CHECKS, False)

    # A previous run's verdict must never outlive it: discard it before measuring
    # anything, so a failing run cannot leave manifest-acceptable evidence behind.
    discard_file(os.path.join(stage, VERIFY_METRICS))
    toolchain = observed_toolchain()
    produced = {
        name: digest_tree(os.path.join(stage, name))
        for name in (f"{MODEL_STEM}.mlpackage", f"{MODEL_STEM}_fp32.mlpackage")
    }
    producer = read_producer_record(stage, produced)
    run_id = producer[RUN_ID_KEY]
    require_producer_toolchain(producer, toolchain, "VERIFIER", "verify_granite.py")
    # The fp16 matrix below runs the COMPILED bundle, which producer.json does
    # not bind. Require the compilation record for this run before measuring
    # anything, so the numbers cannot describe a bundle another run built.
    compile_record = require_compile_record(stage, run_id)
    assert_staged_matches_staging(root, stage)

    goldens = goldens_dir()
    corpus_path = os.path.join(goldens, "corpus.json")
    # The goldens are a PAIR and the crosscheck names the corpus it belongs to.
    # Checked before the corpus is even parsed: scoring against a corpus whose
    # crosscheck describes a different one is the split-pair state an interrupted
    # regeneration leaves, and nothing downstream used to look.
    require_published_crosscheck(goldens)
    checks["golden_pair"] = True
    entries = json.load(open(corpus_path))["entries"]
    # Before any model is loaded: the oracle must be the full committed corpus.
    # Every measurement below is a fold over these entries seeded at 1.0, so an
    # empty or truncated corpus would clear every floor without predicting once.
    assert_corpus_identity(entries, corpus_path)
    checks["corpus_identity"] = True
    st = load_sentence_transformer()
    tokenizer = st.tokenizer
    pad_id = st[0].auto_model.config.pad_token_id
    feeds = []
    for entry in entries:
        input_ids, attention_mask, ids = padded_inputs(tokenizer, entry["text"], pad_id)
        if list(ids) != entry["token_ids"]:
            failures.append(f"token_ids drift on `{entry['id']}` (tokenizer is not the golden one)")
        feeds.append((entry["id"], {"input_ids": input_ids, "attention_mask": attention_mask},
                      np.asarray(entry["embedding"], np.float32)))
    checks["tokenizer_identity"] = not failures
    print(f"  {len(feeds)} corpus entries from {corpus_path}")

    # ---- (a) I/O contract on the SHIPPED bundle ----
    print("\n=== (a) I/O contract (the SHIPPED fp16 mlpackage under the artifact root) ===")
    before = len(failures)
    fp16_pkg = ct.models.MLModel(os.path.join(root, f"{MODEL_STEM}.mlpackage"),
                                 compute_units=ct.ComputeUnit.CPU_ONLY)
    assert_contract(fp16_pkg, failures)
    checks["io_contract"] = len(failures) == before

    # ---- (b) CoreML fp32 (CPUOnly) vs the committed transformers-fp32 goldens ----
    print("\n=== (b) CoreML fp32 [CpuOnly] vs the committed transformers-fp32 goldens ===")
    fp32 = ct.models.MLModel(os.path.join(stage, f"{MODEL_STEM}_fp32.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
    fp32_out = []
    worst = 1.0
    for name, feed, golden in feeds:
        out = np.asarray(fp32.predict(feed)["embedding"]).ravel()
        fp32_out.append(out)
        c = cos(out, golden)
        print(f"  {name:<10} cos = {c:.10f}")
        worst = worst_update(worst, c)
    metrics["fp32_CpuOnly_vs_committed_goldens"] = worst
    check(failures, "fp32 [CpuOnly] vs committed goldens", worst, FP32_FLOOR)

    # ---- (c) CoreML fp16 vs CoreML fp32-CPUOnly, per compute unit ----
    print("\n=== (c) CoreML fp16 vs CoreML fp32 [CpuOnly], per compute unit ===")
    bundle = os.path.join(root, f"{MODEL_STEM}.mlmodelc")
    for uname, unit in UNITS.items():
        try:
            m = ct.models.CompiledMLModel(bundle, unit)
            worst = 1.0
            nonfinite = 0
            for (name, feed, _golden), ref in zip(feeds, fp32_out):
                out = np.asarray(m.predict(feed)["embedding"]).ravel()
                if not np.all(np.isfinite(out)):
                    nonfinite += 1
                worst = worst_update(worst, cos(out, ref))
        except Exception as e:  # noqa: BLE001 — a load/predict failure is HARD
            print(f"  fp16 [{uname:18s}] ERROR {type(e).__name__}: {str(e)[:150]}")
            failures.append(f"fp16 [{uname}] load/predict failed: {type(e).__name__}")
            continue
        metrics[f"fp16_{uname}_vs_fp32"] = worst
        metrics[f"fp16_{uname}_nonfinite_entries"] = nonfinite
        print(f"  fp16 [{uname:18s}] worst vs fp32 = {worst:.8f}  non-finite {nonfinite}/{len(feeds)}")
        check(failures, f"fp16 [{uname}] vs fp32", worst, FP16_FLOOR)
        if nonfinite:
            failures.append(f"fp16 [{uname}]: {nonfinite}/{len(feeds)} entries produced non-finite output")

    if failures:
        print("\nVERIFY FAILED — the artifact did not clear its pinned floors:")
        for f in failures:
            print("  -", f)
        print(f"  no {VERIFY_METRICS} was written; nothing downstream may treat this build "
              f"as verified")
        sys.exit(1)

    # Only now, with `failures` empty: record the verdict, the checks whose
    # success would otherwise be unrepresented, the producing toolchain, and the
    # digests of the bytes every number above was measured from.
    metrics["floors"] = {"fp32_vs_goldens": FP32_FLOOR, "fp16_vs_fp32": FP16_FLOOR}
    metrics[RUN_ID_KEY] = run_id
    metrics["checks"] = checks
    metrics["toolchain"] = producer["toolchain"]
    metrics["compile"] = compile_record
    metrics["evidence"] = evidence_digests(root, stage, goldens)
    replace_file_atomic(os.path.join(stage, VERIFY_METRICS),
                        json.dumps(metrics, indent=2) + "\n")
    print(f"\n  wrote {VERIFY_METRICS} (bound to this build's digests)")
    print("\nDONE verify — contract, fp32-vs-goldens, and every fp16 compute-unit arm held")


if __name__ == "__main__":
    main()
