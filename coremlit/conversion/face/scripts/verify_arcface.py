"""Parity and known-pairs, on the committed fixture faces. Usage: ``python verify_arcface.py``.

Fail-closed, and **every floor below is declared here, above the code that measures against
it.** A floor chosen after seeing the number is not a floor.

Four measurements, in the order a failure is most useful:

1. **fp32 CoreML vs onnxruntime** (``PARITY_FP32``). Same precision on both sides, so this
   is the conversion itself — the ONNX -> PyTorch -> MIL path — with no precision story to
   hide behind. Anything below ``0.9999`` here is a converted graph that is not the graph.
2. **fp16 CoreML per compute arm vs the same fp32 ONNX** (``PARITY_FP16``). This is issue
   #115's acceptance gate (cosine >= 0.99 between the CoreML and ONNX outputs on the
   fixture faces), and 0.99 is where that issue's census put it on measured grounds: the
   ANE's own fp16 floor for an IResNet is ``1 - cos ~ 0.0015`` typical / ``0.0025`` worst,
   while the cheapest REAL preprocessing bug costs ``0.083``. 0.99 sits ~4x above the noise
   and ~8x below the cheapest bug. **Do not tighten it to 0.999** — that number fails
   constantly on the ANE and says nothing about correctness.
3. **Channel order, decided numerically.** ``ArcFaceONNX`` feeds RGB (``blobFromImages(...,
   swapRB=True)`` over an OpenCV BGR crop), and ``probe_onnx_contract.py`` records that
   reading of their source. This arm makes it a measurement: the SAME crops are embedded
   RGB-fed and BGR-fed, and the known-pairs separation is reported under both. A model fed
   the wrong order still returns a plausible-looking 512-d vector; what it loses is the
   separation, which is the only thing that can tell the two apart.
4. **Known pairs**, at a threshold this recipe did not invent. InsightFace's own
   recognition demo (``web-demos/src_recognition/main.py`` @ ``f8aa2c17e18044a86bbfa04be40e
   00cd2ff40a4f``, sha256 ``24a94180…9509``) rules ``sim >= 0.28`` "They ARE the same
   person", ``sim < 0.2`` "They are NOT the same person", and the band between the two
   "LIKELY TO be". Those two constants are ``SAME_MIN`` and ``DIFFERENT_MAX`` below. The
   fixture set is 6 identities — far too small to estimate a false-accept rate, so no FAR
   is claimed; what is claimed is that at InsightFace's own operating point every
   same-person pair is accepted, every different-person pair is rejected, and the two
   populations do not touch.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (BUNDLE, EMBED_DIM, INPUT_NAME, MLPACKAGE_FP32, OUTPUT_NAME,
                             conv_dir, cos, fixtures_dir, models_out_dir, observed_toolchain,
                             preprocess, require_source, staging_dir)

#: fp32 CoreML must BE the ONNX. Both sides are fp32 and the graph is arithmetic-identical,
#: so the residual is float-association noise in the MIL scheduler and nothing else.
PARITY_FP32 = 0.9999
#: fp16, any compute arm, against the fp32 ONNX. See the module doc; this is issue #115's
#: own gate and its own justification.
PARITY_FP16 = 0.99
#: InsightFace's operating point, quoted above. Not ours, and not fitted to this set.
SAME_MIN = 0.28
DIFFERENT_MAX = 0.20

ARMS = ("All", "CpuAndGpu", "CpuOnly", "CpuAndNeuralEngine")


def load_fixtures():
    manifest = json.loads((fixtures_dir() / "faces" / "manifest.json").read_text())
    crops, labels, names = [], [], []
    for row in manifest["faces"]:
        raw = (fixtures_dir() / "faces" / row["crop"]).read_bytes()
        if len(raw) != 112 * 112 * 3:
            raise SystemExit(f"{row['crop']}: {len(raw)} bytes, expected {112 * 112 * 3}")
        crops.append(np.frombuffer(raw, np.uint8).reshape(112, 112, 3))
        labels.append(row["person"])
        names.append(row["id"])
    return manifest, np.stack(crops), labels, names


def embed_onnx(x_nchw):
    import onnxruntime as ort

    session = ort.InferenceSession(str(require_source()), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.stack([session.run(None, {name: x[None]})[0].ravel() for x in x_nchw])


def embed_coreml(path, units, x_nchw, compiled):
    import coremltools as ct

    if compiled:
        model = ct.models.CompiledMLModel(str(path), units)
    else:
        model = ct.models.MLModel(str(path), compute_units=units)
    return np.stack([np.asarray(model.predict({INPUT_NAME: x[None]})[OUTPUT_NAME],
                                np.float64).ravel() for x in x_nchw])


def pair_stats(embeddings, labels):
    """(min same-person cosine, max different-person cosine, counts, the worst of each)."""
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = unit @ unit.T
    same, diff = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (same if labels[i] == labels[j] else diff).append((float(sims[i, j]), i, j))
    return {
        "same_pairs": len(same), "different_pairs": len(diff),
        "min_same": min(same)[0], "max_same": max(same)[0],
        "min_different": min(diff)[0], "max_different": max(diff)[0],
        "worst_same": min(same), "worst_different": max(diff),
        "margin": min(same)[0] - max(diff)[0],
    }


def main():
    observed = observed_toolchain()
    manifest, crops, labels, names = load_fixtures()
    print(f"[ok] {len(crops)} aligned crops, {len(set(labels))} identities")

    x = preprocess(crops)
    reference = embed_onnx(x)
    if reference.shape != (len(crops), EMBED_DIM):
        raise SystemExit(f"onnxruntime returned {reference.shape}")
    norms = np.linalg.norm(reference, axis=1)
    print(f"[ok] onnxruntime fp32 reference: ‖e‖ = {norms.min():.2f}..{norms.max():.2f} "
          f"(RAW — nowhere near 1, so the door's L2 is not a double normalisation)")

    failures, report = [], {"floors": {"parity_fp32": PARITY_FP32, "parity_fp16": PARITY_FP16,
                                       "same_min": SAME_MIN, "different_max": DIFFERENT_MAX},
                            "reference_norms": {"min": float(norms.min()),
                                                "max": float(norms.max()),
                                                "median": float(np.median(norms))}}

    # --- 1. fp32 CoreML vs the ONNX ---------------------------------------------------
    import coremltools as ct
    fp32 = embed_coreml(staging_dir() / MLPACKAGE_FP32, ct.ComputeUnit.CPU_ONLY, x, False)
    cs = [cos(a, b) for a, b in zip(reference, fp32)]
    report["fp32_vs_onnx"] = {"min": min(cs), "median": float(np.median(cs)), "max": max(cs)}
    print(f"  fp32 CoreML (CpuOnly) vs onnxruntime: min {min(cs):.7f} median "
          f"{np.median(cs):.7f} max {max(cs):.7f}   floor {PARITY_FP32}")
    if min(cs) < PARITY_FP32:
        failures.append(f"fp32 parity {min(cs)} < {PARITY_FP32}")

    # --- 2. fp16 per arm --------------------------------------------------------------
    bundle = models_out_dir() / BUNDLE
    if not bundle.is_dir():
        raise SystemExit(f"missing compiled bundle {bundle} — run the compile step first")
    report["fp16_vs_onnx"] = {}
    fp16_embeddings = {}
    for arm in ARMS:
        units = {"All": ct.ComputeUnit.ALL, "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
                 "CpuOnly": ct.ComputeUnit.CPU_ONLY,
                 "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE}[arm]
        e = embed_coreml(bundle, units, x, True)
        fp16_embeddings[arm] = e
        cs = [cos(a, b) for a, b in zip(reference, e)]
        report["fp16_vs_onnx"][arm] = {"min": min(cs), "median": float(np.median(cs)),
                                       "max": max(cs), "worst_1_minus_cos": 1.0 - min(cs)}
        print(f"  fp16 {arm:20s} vs onnxruntime: min {min(cs):.7f} median "
              f"{np.median(cs):.7f} max {max(cs):.7f}   1-cos worst {1.0 - min(cs):.2e}")
        if min(cs) < PARITY_FP16:
            failures.append(f"fp16 {arm} parity {min(cs)} < {PARITY_FP16}")

    # --- 3. channel order, measured ---------------------------------------------------
    bgr = embed_onnx(preprocess(crops[:, :, :, ::-1]))
    report["channel_order"] = {
        "rgb": pair_stats(reference, labels),
        "bgr": pair_stats(bgr, labels),
        "mean_1_minus_cos_rgb_vs_bgr": float(np.mean([1.0 - cos(a, b)
                                                      for a, b in zip(reference, bgr)])),
    }
    r, b = report["channel_order"]["rgb"], report["channel_order"]["bgr"]
    print(f"  channel order   RGB-fed: margin {r['margin']:+.4f} "
          f"(min same {r['min_same']:.4f}, max different {r['max_different']:.4f})")
    print(f"                  BGR-fed: margin {b['margin']:+.4f} "
          f"(min same {b['min_same']:.4f}, max different {b['max_different']:.4f})")
    print(f"                  mean 1-cos between the two feedings: "
          f"{report['channel_order']['mean_1_minus_cos_rgb_vs_bgr']:.4f}")
    if b["margin"] >= r["margin"]:
        failures.append("BGR feeding separates identities at least as well as RGB — the "
                        "channel order this recipe read off InsightFace's source is not the "
                        "one the weights want")

    # --- 4. known pairs, on the SHIPPED artifact --------------------------------------
    report["known_pairs"] = {"onnx_fp32": pair_stats(reference, labels)}
    for arm, e in fp16_embeddings.items():
        report["known_pairs"][f"coreml_fp16_{arm}"] = pair_stats(e, labels)
    for key, st in report["known_pairs"].items():
        ok = st["min_same"] >= SAME_MIN and st["max_different"] < DIFFERENT_MAX
        print(f"  known pairs {key:28s} same {st['same_pairs']:3d} / different "
              f"{st['different_pairs']:4d}  min same {st['min_same']:.4f}  max different "
              f"{st['max_different']:.4f}  margin {st['margin']:+.4f}  {'OK' if ok else 'FAIL'}")
        if st["min_same"] < SAME_MIN:
            i, j = st["worst_same"][1], st["worst_same"][2]
            failures.append(f"{key}: same-person pair ({names[i]}, {names[j]}) scores "
                            f"{st['min_same']:.4f} < {SAME_MIN}")
        if st["max_different"] >= DIFFERENT_MAX:
            i, j = st["worst_different"][1], st["worst_different"][2]
            failures.append(f"{key}: different-person pair ({names[i]}, {names[j]}) scores "
                            f"{st['max_different']:.4f} >= {DIFFERENT_MAX}")

    report["fixtures"] = {"count": len(crops), "identities": sorted(set(labels)),
                          "manifest_revision": manifest.get("revision")}
    report["toolchain"] = observed
    (conv_dir() / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[ok] wrote {conv_dir() / 'verify.json'}")
    if failures:
        raise SystemExit("VERIFY FAILED:\n  " + "\n  ".join(failures))
    print("VERIFY OK")


if __name__ == "__main__":
    main()
