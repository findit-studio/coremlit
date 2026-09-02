"""Fail-closed conversion verification for ReDimNet-B5.

  (a) PARITY FLOOR — the SHIP gate: CoreML fp32 (CPU) vs the PyTorch fp32 model, on the
      SAME mel, which is itself computed by the checkpoint's own `MelBanks`. The reference
      is the UNMODIFIED `ReDimNetWrap.forward` on the waveform, so (a) measures the whole
      published function even though the graph starts one module later.
  (b) fp16 characterization: the shipped fp16 bundle vs the fp32 reference on CPU.
  (c) discrimination sanity: refuses the version where the CoreML graph alone collapses.
      The corpus deliberately contains a DEGENERATE PAIR — `silence` and `dc_offset` both
      reduce to an all-zero mel, because a mean-normalized log-mel of any STATIONARY signal
      is identically zero — so an absolute "no two clips may be identical" floor is wrong
      here and was measured to fire on a correct graph. The check therefore compares the
      CoreML cross-clip cosine MATRIX against PyTorch's entry for entry: identical pairs
      are allowed exactly where PyTorch has them.

WHY THESE FLOORS. The house precedent for a parity claim is >= 0.99
(`tests/*/placement.rs::SANITY_COS`, `conversion/*/verify_*.py::SANITY_COS_FLOOR`), and
that is where the fp16 arms are held. The fp32-vs-fp32 floor is set much tighter
(CONV_COS_FLOOR) because the two implementations compute the same arithmetic in the same
precision and should agree to near machine precision — a loose floor there would let a
real conversion defect through. Both were chosen BEFORE measuring and neither was moved
afterwards; the measured values live in README.md, and the floors are what must not move.
The waveform-in variant of this graph does NOT clear the 0.99 fp16 floor on any compute
unit, which is why the contract starts at the mel — see `probe_waveform_contract.py`.
"""
import sys

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import (EMBED_DIM, INPUT_NAME, N_FRAMES, N_MELS, OUTPUT_NAME,
                              WINDOW_SAMPLES, cos, load_model, mel_for_waveform, models_out_dir,
                              observed_toolchain, staging_dir, worst_update)
from _fixtures import CORPUS, samples_f32

UNITS = {
    "All": ct.ComputeUnit.ALL,
    "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
    "CpuOnly": ct.ComputeUnit.CPU_ONLY,
    "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
}
CONV_COS_FLOOR = 0.9999      # fp32 vs fp32: same arithmetic, different backend.
SANITY_COS_FLOOR = 0.99      # the house floor for a parity claim (fp16 arms).
# How far a CoreML cross-clip cosine may sit from PyTorch's for the same pair. Purely a
# collapse detector, not a quality claim.
CROSS_CLIP_TOL = 1e-3


def main():
    observed_toolchain()
    model, _cfg = load_model()

    clips = list(CORPUS)
    wavs = {c: samples_f32(c, WINDOW_SAMPLES)[None, :] for c in clips}
    # The graph's input, computed by the checkpoint's own front end; the Rust door must
    # reproduce this from MEL_FRONT_END.
    xs = {c: mel_for_waveform(model, w).numpy().astype(np.float32) for c, w in wavs.items()}
    for c, x in xs.items():
        if x.shape != (1, N_MELS, N_FRAMES):
            raise SystemExit(f"{c}: mel {x.shape}, expected (1, {N_MELS}, {N_FRAMES})")
    # The reference is the UNMODIFIED forward on the waveform, so (a) still measures the
    # published function rather than only the piece that was converted.
    with torch.no_grad():
        torch_ref = {c: model(torch.from_numpy(w)).numpy().ravel() for c, w in wavs.items()}
    for c, e in torch_ref.items():
        if e.shape != (EMBED_DIM,):
            raise SystemExit(f"{c}: PyTorch embedding {e.shape}, expected ({EMBED_DIM},)")

    failures = []
    fp32 = ct.models.MLModel(str(staging_dir() / "redimnet_b5_fp32.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)

    # (a) parity floor.
    print("\n(a) CoreML fp32 (CPU) vs PyTorch fp32 — the SHIP gate")
    worst, maxabs, cm_ref = 1.0, 0.0, {}
    for c in clips:
        out = np.asarray(fp32.predict({INPUT_NAME: xs[c]})[OUTPUT_NAME], np.float64).ravel()
        cm_ref[c] = out
        cc = cos(out, torch_ref[c])
        worst = worst_update(worst, cc)
        maxabs = max(maxabs, float(np.abs(out - torch_ref[c]).max()))
        print(f"    {c:12s} cos {cc:.8f}  max|Δ| {float(np.abs(out - torch_ref[c]).max()):.3e}")
    print(f"  worst cos {worst:.8f}  worst max|Δ| {maxabs:.3e}  (floor {CONV_COS_FLOOR})")
    if not (worst >= CONV_COS_FLOOR):
        failures.append(f"(a) parity worst cos {worst:.8f} < {CONV_COS_FLOOR}")

    # (c) discrimination sanity — refuse a collapsed CoreML graph.
    pairs = [(a, b) for i, a in enumerate(clips) for b in clips[i + 1:]]
    worst_pair, worst_delta = None, 0.0
    for a, b in pairs:
        d = abs(cos(cm_ref[a], cm_ref[b]) - cos(torch_ref[a], torch_ref[b]))
        if d > worst_delta:
            worst_delta, worst_pair = d, (a, b)
    t_pairs = [cos(torch_ref[a], torch_ref[b]) for a, b in pairs]
    c_pairs = [cos(cm_ref[a], cm_ref[b]) for a, b in pairs]
    print(f"\n(c) cross-clip cosine range: PyTorch [{min(t_pairs):+.4f}, {max(t_pairs):+.4f}], "
          f"CoreML [{min(c_pairs):+.4f}, {max(c_pairs):+.4f}]; worst per-pair Δ "
          f"{worst_delta:.2e} {worst_pair}")
    if not (worst_delta <= CROSS_CLIP_TOL):
        failures.append(f"(c) CoreML cross-clip geometry differs from PyTorch by "
                        f"{worst_delta:.2e} > {CROSS_CLIP_TOL} at {worst_pair}")

    # (b) fp16 per compute unit.
    print("\n(b) fp16 .mlmodelc vs the fp32 CPU reference, per compute unit")
    bundle = models_out_dir() / "redimnet_b5.mlmodelc"
    for uname, cu in UNITS.items():
        try:
            m16 = ct.models.CompiledMLModel(str(bundle), cu)
            w = 1.0
            for c in clips:
                out = np.asarray(m16.predict({INPUT_NAME: xs[c]})[OUTPUT_NAME], np.float64).ravel()
                if not np.isfinite(out).all():
                    failures.append(f"(b) fp16 [{uname}] non-finite output on {c}")
                w = worst_update(w, cos(out, cm_ref[c]))
        except Exception as exc:                     # a load/predict failure is HARD
            print(f"    [{uname:18s}] ERROR {type(exc).__name__}: {str(exc)[:160]}")
            failures.append(f"(b) fp16 [{uname}] load/predict: {type(exc).__name__}")
            continue
        ok = bool(w >= SANITY_COS_FLOOR)
        print(f"    [{uname:18s}] worst cos {w:.8f}  {'OK' if ok else 'FAIL'} "
              f"(floor {SANITY_COS_FLOOR})")
        if not ok:
            failures.append(f"(b) fp16 [{uname}] worst cos {w:.8f} < {SANITY_COS_FLOOR}")

    if failures:
        print("\nVERIFY FAILED — do NOT ship these artifacts:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nDONE verify — every floor held.")


if __name__ == "__main__":
    main()
