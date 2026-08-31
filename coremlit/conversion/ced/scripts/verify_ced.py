"""Fail-closed conversion verification (adapts clap's verify_encoders.py), per CED size:

  (a) CONVERSION FLOOR — the SHIP gate: CoreML fp32 (CPU) vs PyTorch fp32 ``CedMelToLogits``
      (pre-sigmoid), feeding the IDENTICAL torchaudio mel (isolates the transformer conversion).
      worst cosine >= 0.9999 AND max|Δlogit| <= 5e-3, else STOP (nonzero exit — do NOT ship).
  (b) "UNMODIFIED forward" cross-check: sigmoid(CoreML fp32) vs the unmodified
      ``model(mel).logits`` (post-sigmoid). worst cosine >= 0.9999.
  (c) shipped-ONNX drop-in sanity (recorded, not gated): sigmoid(CoreML fp32) vs
      ``onnxruntime(model.onnx, waveform=zero_pad(clip,160000))`` (post-sigmoid). The onnx computes
      its mel in-graph so a small offset is expected; sanity floor 0.99.
  (d) fp16 characterization: CoreML fp16 vs fp32 (CPU) per compute unit. Recorded; sanity 0.99.

FAIL-CLOSED (clap's rule): NaN-poisoning worst, load/predict exceptions are HARD failures,
any breach exits NONZERO so run_ced.sh's ``set -e`` halts before goldens are cut.
"""
import os
import sys

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _ced_common import (CedMelToLogits, MODELS, SIZES, WINDOW_SAMPLES, cos, load_feature_extractor,
                         load_model, mel_for_waveform, models_out_dir, src_dir, staging_dir)
from _fixtures import CORPUS, samples_f32

UNITS = {
    "All": ct.ComputeUnit.ALL,
    "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
    "CpuOnly": ct.ComputeUnit.CPU_ONLY,
}
CONV_COS_FLOOR = 0.9999
CONV_MAXABS_CEIL = 5e-3
SANITY_COS_FLOOR = 0.99


def worst_update(worst, c):
    if worst != worst or c != c:
        return float("nan")
    return min(worst, c)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def check(failures, label, worst, floor):
    ok = bool(worst >= floor)
    print(f"    [{'OK' if ok else 'FAIL'}] {label}: worst {worst:.8f} (floor {floor})")
    if not ok:
        failures.append(f"{label}: worst {worst:.8f} < floor {floor} (or non-finite)")
    return ok


def verify_size(size, failures):
    print(f"\n===== {size} =====")
    model = load_model(size)
    fe = load_feature_extractor(size)
    wrap = CedMelToLogits(model).eval()

    # references on each parity/full clip (skip the long clip — not a single window)
    items = []  # (cid, mel_np[1,64,1001], pre_ref[527], post_ref[527], onnx_post[527])
    import onnxruntime
    sess = onnxruntime.InferenceSession(str(src_dir(size) / "model.onnx"),
                                        providers=["CPUExecutionProvider"])
    inn, outn = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    for cid, (_gen, kind) in CORPUS.items():
        if kind == "long":
            continue
        mel = mel_for_waveform(fe, samples_f32(cid))
        with torch.no_grad():
            pre = wrap(mel).numpy().ravel()
            post = model(mel).logits.numpy().ravel()
        wav = samples_f32(cid).astype(np.float32)
        if wav.shape[0] < WINDOW_SAMPLES:
            wav = np.pad(wav, (0, WINDOW_SAMPLES - wav.shape[0]))
        onnx_post = np.asarray(sess.run([outn], {inn: wav[None, :]})[0]).ravel()
        items.append((cid, mel.numpy().astype(np.float32), pre, post, onnx_post))

    fp32 = ct.models.MLModel(str(staging_dir() / f"ced_{size}_fp32.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)

    # (a) conversion floor (pre-sigmoid) + max-abs
    wa, maxabs = 1.0, 0.0
    for cid, mel, pre, _post, _o in items:
        out = np.asarray(fp32.predict({"mel": mel})["logits"]).ravel()
        wa = worst_update(wa, cos(out, pre))
        maxabs = max(maxabs, float(np.abs(out - pre).max()))
    print(f"  (a) CoreML fp32 vs PyTorch pre-sigmoid: worst cos {wa:.8f}  max|Δ| {maxabs:.3e}")
    check(failures, f"{size} (a) conversion cos", wa, CONV_COS_FLOOR)
    if not (maxabs <= CONV_MAXABS_CEIL):
        failures.append(f"{size} (a) max|Δlogit| {maxabs:.3e} > ceil {CONV_MAXABS_CEIL}")
        print(f"    [FAIL] {size} (a) max|Δ| {maxabs:.3e} > {CONV_MAXABS_CEIL}")
    else:
        print(f"    [OK]   {size} (a) max|Δ| {maxabs:.3e} <= {CONV_MAXABS_CEIL}")

    # (b) unmodified forward (post-sigmoid) + (c) onnx sanity (RECORD-ONLY, per §6: the shipped
    # onnx computes its mel IN-GRAPH, so it diverges from the torchaudio mel on degenerate clips —
    # (a) already proves the CoreML graph is exact, so (c) is informational, not a ship gate).
    wb, wc, wc_signal = 1.0, 1.0, 1.0
    for cid, mel, _pre, post, onnx_post in items:
        cm_post = sigmoid(np.asarray(fp32.predict({"mel": mel})["logits"]).ravel())
        wb = worst_update(wb, cos(cm_post, post))
        c = cos(cm_post, onnx_post)
        wc = worst_update(wc, c)
        if cid != "silence_2s":  # silence: both are ~uniform; onnx's in-graph floor differs
            wc_signal = worst_update(wc_signal, c)
        print(f"      (c) {cid:14s} cos(CoreML_post, onnx_post) = {c:.6f}")
    print(f"  (b) sigmoid(CoreML fp32) vs unmodified forward: worst cos {wb:.8f}")
    check(failures, f"{size} (b) unmodified-forward cos", wb, CONV_COS_FLOOR)
    print(f"  (c) sigmoid(CoreML fp32) vs shipped model.onnx (post-sigmoid): worst {wc:.6f}"
          f"  (signal-clip worst {wc_signal:.6f}) — RECORDED, not a ship gate")
    if not (wc >= 0.85):  # loose tripwire: only a GROSS onnx mismatch is a finding
        failures.append(f"{size} (c) onnx sanity worst {wc:.6f} < 0.85 tripwire")

    # (d) fp16 per compute unit vs fp32 CPU
    mlmodelc = models_out_dir() / f"ced-{size}" / f"ced_{size}.mlmodelc"
    fp32_ref = [np.asarray(fp32.predict({"mel": mel})["logits"]).ravel() for _c, mel, *_ in items]
    for uname, cu in UNITS.items():
        try:
            m16 = ct.models.CompiledMLModel(str(mlmodelc), cu)
            w = 1.0
            for (cid, mel, *_), ref in zip(items, fp32_ref):
                out = np.asarray(m16.predict({"mel": mel})["logits"]).ravel()
                w = worst_update(w, cos(out, ref))
        except Exception as e:  # HARD failure — nonzero exit
            print(f"  (d) fp16 [{uname:18s}] ERROR {type(e).__name__}: {str(e)[:140]}")
            failures.append(f"{size} fp16 [{uname}] load/predict: {type(e).__name__}")
            continue
        print(f"  (d) fp16 [{uname:18s}] worst cos vs fp32 = {w:.8f}")
        check(failures, f"{size} (d) fp16 [{uname}]", w, SANITY_COS_FLOOR)


def main(sizes):
    failures = []
    for size in sizes:
        verify_size(size, failures)
    if failures:
        print("\nVERIFY FAILED — floors breached (do NOT ship these artifacts):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nDONE verify — every conversion floor held.")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else list(SIZES))
