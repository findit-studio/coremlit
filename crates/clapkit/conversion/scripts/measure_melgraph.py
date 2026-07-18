"""Concrete parity numbers for the mel-in-graph candidate, on real audio:
  A. melgraph fp32 (raw audio in) vs the shipped spectrogram-input fp32 (HF mel in)
     -> does the in-graph STFT reproduce HF's mel path end-to-end?
  B. melgraph fp16 vs melgraph fp32 -> does the fp16 STFT survive (overflow)?"""
import os
import sys
import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from _clap_common import load_processor, TARGET_SAMPLES
from _fixtures import audio_clips, input_features

STAGE = "/private/tmp/claude-501/-Users-al-Developer-findit-studio-coremlit/2e543e17-c5e2-4187-be75-b6b4fafe4418/scratchpad/conv/clapkit/staging"


def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


proc = load_processor()
melg_f32 = ct.models.MLModel(os.path.join(STAGE, "clap_audio_melgraph.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
spec_f32 = ct.models.MLModel(os.path.join(STAGE, "clap_audio_fp32.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
melg_f16 = ct.models.CompiledMLModel(os.path.join(STAGE, "clap_audio_melgraph.mlmodelc"),
                                     ct.ComputeUnit.CPU_ONLY)

print(f"{'clip':24s} {'A:melgraph-f32 vs spec-f32':>28s} {'B:melgraph f16 vs f32':>24s} {'f16 finite?':>12s}")
wa, wb = 1.0, 1.0
for name, samples in audio_clips():
    a = samples[:TARGET_SAMPLES].astype(np.float32)
    if len(a) < TARGET_SAMPLES:
        a = np.pad(a, (0, TARGET_SAMPLES - len(a))).astype(np.float32)
    audio_in = a.reshape(1, TARGET_SAMPLES)
    inf, _ = input_features(proc, samples)
    e_melg = melg_f32.predict({"audio": audio_in})["audio_embeds"]
    e_spec = spec_f32.predict({"input_features": inf.numpy().astype(np.float32)})["audio_embeds"]
    e_m16 = np.asarray(melg_f16.predict({"audio": audio_in})["audio_embeds"])
    A = cos(e_melg, e_spec)
    B = cos(e_m16, e_melg)
    finite = bool(np.isfinite(e_m16).all())
    wa, wb = min(wa, A), min(wb, B)
    print(f"{name:24s} {A:28.6f} {B:24.6f} {str(finite):>12s}")
print(f"\nWORST A (melgraph-fp32 vs spectrogram-fp32) = {wa:.6f}")
print(f"WORST B (melgraph fp16 vs fp32)             = {wb:.6f}")
print("DONE")
