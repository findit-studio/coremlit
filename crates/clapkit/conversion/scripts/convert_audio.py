"""Convert the CLAP audio tower (HTSAT + audio_projection) -> CoreML.

Source: laion/clap-htsat-unfused (transformers ClapModel), revision-pinned.
Contract (spectrogram-input; the mel/STFT frontend is a Rust port validated
bit-level against textclap's mel.rs — see mel_in_graph_probe.py for the
in-graph attempt and its recorded rejection):
  input : input_features fp32 [1, 1, 1001, 64]  (HF ClapFeatureExtractor mel)
  output: audio_embeds   fp32 [1, 512]           (projection output, PRE-L2-norm)

L2 normalization is intentionally OUT of the graph (clapkit normalizes in Rust),
which keeps the fp16 rsqrt-guard class out of the audio graph entirely. fusion is
disabled (enable_fusion=False) so is_longer is baked to None -> input_features is
the sole input.

Produces BOTH precisions:
  clap_audio.mlpackage       compute_precision=FLOAT16 (shipped candidate)
  clap_audio_fp32.mlpackage  compute_precision=FLOAT32 (verification reference)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
import _custom_ops  # noqa: F401  registers the `new_ones` torch op
from _clap_common import (
    load_model, load_processor, patch_bicubic_resize, TARGET_SAMPLES, N_MELS, T_FRAMES,
)

STAGE = "/private/tmp/claude-501/-Users-al-Developer-findit-studio-coremlit/2e543e17-c5e2-4187-be75-b6b4fafe4418/scratchpad/conv/clapkit/staging"
os.makedirs(STAGE, exist_ok=True)


class AudioTower(nn.Module):
    """input_features [1,1,1001,64] -> audio_embeds [1,512] (pre-norm)."""

    def __init__(self, m):
        super().__init__()
        self.audio_model = m.audio_model
        self.audio_projection = m.audio_projection

    def forward(self, input_features):
        pooled = self.audio_model(input_features=input_features, is_longer=None).pooler_output
        return self.audio_projection(pooled)


def main():
    model = load_model()
    proc = load_processor()

    rng = np.random.RandomState(0)
    audio = rng.randn(TARGET_SAMPLES).astype(np.float32) * 0.1
    feats = proc.feature_extractor(audio, sampling_rate=48000, return_tensors="pt")
    inf = feats["input_features"]
    assert tuple(inf.shape) == (1, 1, T_FRAMES, N_MELS), inf.shape

    # Ground truth from the UN-patched model (true CLAP bicubic), captured before
    # the resize shim is installed.
    with torch.no_grad():
        ref = model.get_audio_features(input_features=inf, is_longer=feats.get("is_longer"))
        ref = ref if torch.is_tensor(ref) else ref.pooler_output

    # Install the exact bicubic->matmul shim and prove it reproduces ground truth.
    patch_bicubic_resize(model)
    net = AudioTower(model).eval()
    with torch.no_grad():
        pre = net(inf)
        cos = F.cosine_similarity(F.normalize(pre, dim=-1), ref, dim=-1).min().item()
    print(f"[CHECK] audio pre-norm shape {tuple(pre.shape)}  "
          f"bicubic-shim normalize(wrapper)-vs-unpatched get_audio_features cos = {cos:.8f}")
    assert cos > 0.99999, f"bicubic resize shim unfaithful: {cos}"

    ts = torch.jit.trace(net, (inf,), check_trace=False)

    for tag, prec in (("", ct.precision.FLOAT16), ("_fp32", ct.precision.FLOAT32)):
        ml = ct.convert(
            ts,
            inputs=[ct.TensorType(name="input_features",
                                  shape=(1, 1, T_FRAMES, N_MELS), dtype=np.float32)],
            outputs=[ct.TensorType(name="audio_embeds", dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=prec,
            convert_to="mlprogram",
        )
        ml.author = "clapkit T1: laion/clap-htsat-unfused audio tower (HTSAT + audio_projection), pre-norm"
        ml.short_description = ("CLAP audio encoder (spectrogram-input): mel [1,1,1001,64] -> "
                                "512-d joint embedding, L2-norm applied by the caller")
        out = os.path.join(STAGE, f"clap_audio{tag}.mlpackage")
        ml.save(out)
        print(f"SAVED {out}  ({prec})")
    print("DONE audio")


if __name__ == "__main__":
    main()
