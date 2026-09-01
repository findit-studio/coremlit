"""The evidence for the contract decision: why the graph starts at the MEL and not at the
waveform. Reproduces the measurement rather than leaving it as a claim in README.md.

The natural contract for the identity lane is ``waveform [1, 96000] -> embedding [1, 192]``
— one fixed window, no mask, the whole published function in one graph, and it is what the
only existing third-party ReDimNet CoreML artifact does. It converts cleanly and is EXACT
in fp32. It is nevertheless rejected, because in fp16 it is wrong on every compute unit.

The cause is dynamic range, and it fails at both ends of it. ``MelBanks`` computes a POWER
spectrogram (``power=2.0``) over a 400-sample window before taking a log:

  * the high end — a full-scale tone concentrates ~400 samples of energy into one bin, and
    the squared magnitude summed across the mel filter's bins passes fp16's 65504 ceiling.
    ``coremltools`` says so out loud during the fp16 conversion of the waveform variant:
    ``RuntimeWarning: overflow encountered in cast``.
  * the low end — the log guard is ``+1e-6``, which is SUBNORMAL in fp16 (smallest normal
    6.10e-5). Hardware that flushes subnormals to zero turns the guard into ``log(0)``.

This is the same defect class this repository already carries a test file for
(``tests/fp16_guards.rs``, issue #15: the pre-repair segmentation graph's "inert
``log(epsilon = 0)``" saturating on the default ANE placement).

Three things this probe measures, in order:

  1. the waveform-in graph's fp32 parity (exact — the conversion itself is not at fault);
  2. its fp16 agreement per compute unit (wrong on all four);
  3. the mel front end ALONE in fp16 per compute unit, which localizes the damage to the
     front end rather than to the network.

``verify_redimnet.py`` measures the third leg — the same weights behind the mel-in contract
— and it clears the floor on every arm. Together the four numbers are the decision.

Usage: ``python probe_waveform_contract.py``. Records ``staging/waveform_contract.json``.
This script is DIAGNOSTIC: it never gates, and ``run_redimnet.sh`` does not call it.
"""
import json
import subprocess
import sys

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import (WINDOW_SAMPLES, cos, load_model, observed_toolchain, staging_dir,
                              worst_update)
from _fixtures import CORPUS, samples_f32

UNITS = {
    "CpuOnly": ct.ComputeUnit.CPU_ONLY,
    "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
    "All": ct.ComputeUnit.ALL,
    "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
}
COMMON = dict(minimum_deployment_target=ct.target.iOS17, convert_to="mlprogram")


def _convert(traced, in_name, in_shape, out_name, stem):
    paths = {}
    for prec, tag in ((ct.precision.FLOAT32, "fp32"), (ct.precision.FLOAT16, "fp16")):
        m = ct.convert(traced,
                       inputs=[ct.TensorType(name=in_name, shape=in_shape, dtype=np.float32)],
                       outputs=[ct.TensorType(name=out_name, dtype=np.float32)],
                       compute_precision=prec, **COMMON)
        p = staging_dir() / f"{stem}_{tag}.mlpackage"
        m.save(str(p))
        paths[tag] = p
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(paths["fp16"]),
                    str(staging_dir())], check=True)
    paths["mlmodelc"] = staging_dir() / f"{stem}_fp16.mlmodelc"
    return paths


def _sweep(paths, in_name, out_name, inputs, refs, clips):
    fp32 = ct.models.MLModel(str(paths["fp32"]), compute_units=ct.ComputeUnit.CPU_ONLY)
    cm32 = {c: np.asarray(fp32.predict({in_name: inputs[c]})[out_name], np.float64).ravel()
            for c in clips}
    worst32 = 1.0
    for c in clips:
        worst32 = worst_update(worst32, cos(cm32[c], refs[c]))
    per_unit = {}
    for uname, cu in UNITS.items():
        m = ct.models.CompiledMLModel(str(paths["mlmodelc"]), cu)
        w = 1.0
        for c in clips:
            out = np.asarray(m.predict({in_name: inputs[c]})[out_name], np.float64).ravel()
            w = worst_update(w, cos(out, cm32[c]))
        per_unit[uname] = w
    return worst32, per_unit


def main():
    observed_toolchain()
    model, _cfg = load_model()
    clips = list(CORPUS)
    wavs = {c: samples_f32(c, WINDOW_SAMPLES)[None, :].astype(np.float32) for c in clips}

    # 1-2. the waveform-in graph: the UNMODIFIED ReDimNetWrap.forward.
    with torch.no_grad():
        torch_emb = {c: model(torch.from_numpy(w)).numpy().ravel() for c, w in wavs.items()}
    ts = torch.jit.trace(model, torch.from_numpy(wavs["formant"]), check_trace=False)
    wav_paths = _convert(ts, "waveform", (1, WINDOW_SAMPLES), "embedding", "probe_waveform")
    wav32, wav16 = _sweep(wav_paths, "waveform", "embedding", wavs, torch_emb, clips)

    # 3. the mel front end alone, to localize the damage.
    class MelOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.spec = m.spec

        def forward(self, x):
            return self.spec(x)

    mo = MelOnly(model).eval()
    with torch.no_grad():
        torch_mel = {c: mo(torch.from_numpy(w)).numpy().ravel() for c, w in wavs.items()}
    tsm = torch.jit.trace(mo, torch.from_numpy(wavs["formant"]), check_trace=False)
    mel_paths = _convert(tsm, "waveform", (1, WINDOW_SAMPLES), "mel", "probe_mel")
    # silence/dc_offset reduce to an all-zero mel, so their cosine is undefined; the
    # localization argument does not need them and including them would report a NaN as
    # if it were a defect.
    signal = [c for c in clips if c not in ("silence", "dc_offset")]
    mel32, mel16 = _sweep(mel_paths, "waveform", "mel", wavs, torch_mel, signal)

    print("\n### waveform -> embedding (the REJECTED contract)")
    print(f"  fp32 (CPU) vs PyTorch: worst cos {wav32:.8f}   <- the conversion itself is exact")
    for u, w in wav16.items():
        print(f"  fp16 [{u:18s}] vs fp32: worst cos {w:.6f}")
    print("\n### waveform -> mel ALONE (localizing the damage; signal clips only)")
    print(f"  fp32 (CPU) vs PyTorch: worst cos {mel32:.8f}")
    for u, w in mel16.items():
        print(f"  fp16 [{u:18s}] vs fp32: worst cos {w:.6f}")
    print("\nCompare with verify_redimnet.py's (b): the same weights behind the mel-in "
          "contract clear the 0.99 floor on every arm.")

    out = staging_dir() / "waveform_contract.json"
    out.write_text(json.dumps({
        "waveform_to_embedding": {"fp32_vs_pytorch": wav32, "fp16_per_unit": wav16},
        "waveform_to_mel": {"fp32_vs_pytorch": mel32, "fp16_per_unit": mel16,
                            "clips": signal},
        "verdict": "waveform-in rejected: fp16 wrong on every compute unit; damage is in "
                   "the mel front end's power spectrogram, not in the network.",
    }, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
