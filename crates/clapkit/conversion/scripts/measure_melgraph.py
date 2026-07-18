"""Concrete parity numbers for the mel-in-graph candidate, on real audio:
  A. melgraph fp32 (raw audio in) vs the shipped spectrogram-input fp32 (HF mel
     in) -> does the in-graph STFT reproduce HF's mel path end-to-end at fp32?
  B. melgraph fp16 vs melgraph fp32 -> does the fp16 in-graph STFT survive
     (power |X|^2 can exceed the fp16 max and overflow)?

Two fixes over the original probe:
  * arm A now loads the SEPARATELY-converted fp32 melgraph (not a compiled copy of
    the fp16 one), and arm B compares the fp16 melgraph against that fp32 melgraph
    — two DIFFERENT conversions, not two executions of one;
  * both arms receive the IDENTICAL 480k waveform (repeat-tiled to fill, exactly
    as clapkit's mel `repeatpad`s), so the only variable is the mel computation.
Both melgraphs load as mlpackages on CPU (no stale/absent `.mlmodelc`)."""
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


def nan_prop_min(worst, c):
    """NaN-propagating min so a non-finite fp16 arm poisons the worst rather than
    being silently dropped by min()."""
    if worst != worst or c != c:
        return float("nan")
    return min(worst, c)


def to_480k(clip):
    """Repeat-tile (or head-truncate) a clip to EXACTLY TARGET_SAMPLES, matching
    clapkit's mel `repeatpad`. The SAME 480k waveform feeds both arms, so arm A
    isolates the mel computation (in-graph fp32 STFT vs HF float64 mel), not a
    padding difference."""
    a = np.asarray(clip, np.float32)
    if len(a) >= TARGET_SAMPLES:
        return a[:TARGET_SAMPLES].astype(np.float32)
    reps = -(-TARGET_SAMPLES // len(a))  # ceil division
    return np.tile(a, reps)[:TARGET_SAMPLES].astype(np.float32)


proc = load_processor()
melg_f32 = ct.models.MLModel(os.path.join(STAGE, "clap_audio_melgraph_fp32.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
melg_f16 = ct.models.MLModel(os.path.join(STAGE, "clap_audio_melgraph_fp16.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
spec_f32 = ct.models.MLModel(os.path.join(STAGE, "clap_audio_fp32.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)

print(f"{'clip':24s} {'A:melg-f32 vs spec-f32':>24s} {'B:melg f16 vs f32':>20s} {'f16 finite?':>12s}")
wa, wb = 1.0, 1.0
for name, samples in audio_clips():
    audio_480k = to_480k(samples)                       # SAME input to both arms
    audio_in = audio_480k.reshape(1, TARGET_SAMPLES)
    inf, _ = input_features(proc, audio_480k)           # HF mel of the SAME 480k
    e_melg32 = melg_f32.predict({"audio": audio_in})["audio_embeds"]
    e_spec = spec_f32.predict({"input_features": inf.numpy().astype(np.float32)})["audio_embeds"]
    e_m16 = np.asarray(melg_f16.predict({"audio": audio_in})["audio_embeds"])
    A = cos(e_melg32, e_spec)   # faithfulness: fp32 in-graph STFT vs HF mel path
    B = cos(e_m16, e_melg32)    # fp16 survival: fp16 melgraph vs fp32 melgraph
    finite = bool(np.isfinite(e_m16).all())
    wa, wb = nan_prop_min(wa, A), nan_prop_min(wb, B)
    print(f"{name:24s} {A:24.6f} {B:20.6f} {str(finite):>12s}")
print(f"\nWORST A (melgraph-fp32 vs spectrogram-fp32) = {wa:.6f}")
print(f"WORST B (melgraph fp16 vs fp32)             = {wb:.6f}")
print("DONE")
