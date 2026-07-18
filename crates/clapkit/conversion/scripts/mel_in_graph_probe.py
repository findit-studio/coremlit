"""MEL-IN-GRAPH DECISION PROBE (spec §2: 'decided at conversion time by measurement').

Attempt to ride the STFT + Slaney-mel + power_to_dB frontend INSIDE the audio
graph (raw 480k audio -> embedding), then judge by three measurements:
  1. does coremltools 9.0 convert torch.stft at all?
  2. is the fp16 graph guard-clean? (the power_to_dB floor amin=1e-10 is 1680x
     below 2^-24 -> a textbook fp16-vanishing guard)
  3. fp16 parity vs the spectrogram-input path.

The fallback (spec-sanctioned) is the shipped spectrogram-input graph + a Rust mel
port validated against textclap's mel.rs. This script RECORDS which path the
measurement selects; it does not itself ship anything.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
import _custom_ops  # noqa: F401
from _clap_common import load_model, load_processor, patch_bicubic_resize, TARGET_SAMPLES, N_MELS, T_FRAMES
from _fixtures import audio_clips

STAGE = "/private/tmp/claude-501/-Users-al-Developer-findit-studio-coremlit/2e543e17-c5e2-4187-be75-b6b4fafe4418/scratchpad/conv/clapkit/staging"
N_FFT, HOP, AMIN = 1024, 480, 1e-10


class MelFrontend(nn.Module):
    """raw audio [1, 480000] -> dB mel [1, 1, 1001, 64], mirroring HF
    ClapFeatureExtractor (periodic Hann, |STFT|^2, Slaney filterbank, 10*log10
    with amin=1e-10 floor). Uses HF's exact filterbank so parity is meaningful."""

    def __init__(self, mel_filters):
        super().__init__()
        # HF mel_filters: [n_freq=513, n_mels=64]. Register as constant buffers.
        self.register_buffer("fb", torch.tensor(mel_filters.T, dtype=torch.float32))  # [64, 513]
        win = torch.hann_window(N_FFT, periodic=True, dtype=torch.float32)
        self.register_buffer("win", win)

    def forward(self, audio):  # audio [1, 480000]
        spec = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                          window=self.win, center=True, pad_mode="reflect",
                          return_complex=True)               # [1, 513, T]
        power = spec.real ** 2 + spec.imag ** 2               # [1, 513, T]
        power = power[:, :, :T_FRAMES]
        mel = torch.matmul(self.fb, power)                    # [1, 64, T]
        mel = torch.clamp(mel, min=AMIN)
        db = 10.0 * torch.log10(mel)                          # [1, 64, T]
        return db.transpose(1, 2).unsqueeze(1)                # [1, 1, T, 64]


def main():
    model = load_model()
    proc = load_processor()
    fe = proc.feature_extractor
    mel_filters = np.asarray(fe.mel_filters, dtype=np.float32)
    print(f"[info] HF mel_filters shape = {mel_filters.shape}")

    front = MelFrontend(mel_filters).eval()

    # 1. mel fidelity: in-graph dB mel vs HF feature extractor input_features (fp32 torch).
    name, samples = next(audio_clips())
    if len(samples) < TARGET_SAMPLES:
        samples = np.pad(samples, (0, TARGET_SAMPLES - len(samples)))
    samples = samples[:TARGET_SAMPLES].astype(np.float32)
    audio_t = torch.from_numpy(samples).unsqueeze(0)
    with torch.no_grad():
        mine = front(audio_t)
    hf = fe(samples, sampling_rate=48000, return_tensors="pt")["input_features"]
    dmax = float((mine - hf).abs().max())
    print(f"[mel-fidelity] in-graph dB mel vs HF fe: max|Δ| = {dmax:.4f} dB  (shapes {tuple(mine.shape)} / {tuple(hf.shape)})")

    # Full raw-audio tower = MelFrontend + patched audio tower.
    patch_bicubic_resize(model)

    class RawAudioTower(nn.Module):
        def __init__(s):
            super().__init__(); s.front = front; s.am = model.audio_model; s.ap = model.audio_projection
        def forward(s, audio):
            inf = s.front(audio)
            return s.ap(s.am(input_features=inf, is_longer=None).pooler_output)

    raw = RawAudioTower().eval()
    with torch.no_grad():
        emb = raw(audio_t)
    print(f"[raw-tower] torch output shape {tuple(emb.shape)}  (fp32 forward works)")
    ts = torch.jit.trace(raw, (audio_t,), check_trace=False)

    # 2. Convert BOTH precisions from the SAME trace, so the measurement compares
    #    an honest fp32 mel-in-graph against the shipped fp32 spectrogram path
    #    (arm A, faithfulness) AND the fp16 mel-in-graph against its own fp32
    #    (arm B, fp16 survival) — not two executions of one conversion.
    #    FAIL-CLOSED: a conversion error exits nonzero (a legitimate "reject
    #    mel-in-graph" outcome, but recorded HARD so measure_melgraph can never
    #    read a stale/absent artifact and print a bogus cosine).
    for prec_name, prec in [("fp32", ct.precision.FLOAT32), ("fp16", ct.precision.FLOAT16)]:
        print(f"\n[convert] attempting {prec_name} mel-in-graph conversion ...")
        try:
            ml = ct.convert(
                ts,
                inputs=[ct.TensorType(name="audio", shape=(1, TARGET_SAMPLES), dtype=np.float32)],
                outputs=[ct.TensorType(name="audio_embeds", dtype=np.float32)],
                minimum_deployment_target=ct.target.iOS17,
                compute_precision=prec,
                convert_to="mlprogram",
            )
            out = os.path.join(STAGE, f"clap_audio_melgraph_{prec_name}.mlpackage")
            ml.save(out)
            print(f"[convert] {prec_name} SUCCESS -> {out}")
        except Exception as e:
            print(f"[convert] {prec_name} FAILED: {type(e).__name__}: {str(e)[:400]}")
            print("MEL-DECISION: in-graph conversion FAILED -> spectrogram-input (mel in Rust).")
            sys.exit(1)
    print("[convert] both precisions converted; run measure_melgraph.py next.")
    print("DONE mel-probe")


if __name__ == "__main__":
    main()
