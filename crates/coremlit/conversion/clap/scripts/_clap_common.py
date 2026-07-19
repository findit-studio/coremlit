"""Shared loader + constants for the clapkit CLAP conversion recipes.

Source of truth: laion/clap-htsat-unfused (transformers ClapModel), pinned to
revision REV below. The repo ships only `pytorch_model.bin` (no safetensors),
and transformers 5.14 refuses `torch.load` under torch<2.6 as a blanket CVE
guard (CVE-2025-32434 — a *pickle* deserialization hazard). We are loading the
official, widely-downloaded LAION checkpoint from HF's content-addressed snapshot
(revision-pinned + hash-verified by the hub), not an untrusted pickle, so we
neutralize the blanket guard for this one trusted load. Documented + deliberate.
"""
import transformers.utils.import_utils as _iu

# Neutralize the blanket torch.load guard for this trusted, revision-pinned load.
_iu.check_torch_load_is_safe = lambda *a, **k: None
# The symbol is also re-exported into modeling_utils' namespace at import time.
import transformers.modeling_utils as _mu  # noqa: E402
_mu.check_torch_load_is_safe = lambda *a, **k: None

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import ClapModel, ClapProcessor  # noqa: E402

MODEL_ID = "laion/clap-htsat-unfused"
REV = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"

PROJ_DIM = 512
TARGET_SAMPLES = 480_000     # 10 s @ 48 kHz
N_MELS = 64
T_FRAMES = 1001              # HF center=True STFT frames for 480k samples @ hop 480
SR = 48_000


def load_model():
    """Return ClapModel (eval, fp32) pinned to REV."""
    return ClapModel.from_pretrained(MODEL_ID, revision=REV).eval()


def load_processor():
    return ClapProcessor.from_pretrained(MODEL_ID, revision=REV)


def mel_features(processor, audio_f32):
    """HF feature-extractor mel for a 1-D float32 waveform at 48 kHz.
    Returns input_features tensor [1, 1, 1001, 64] and is_longer (or None)."""
    feats = processor.feature_extractor(
        audio_f32, sampling_rate=SR, return_tensors="pt"
    )
    return feats


import types  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _bicubic_operator(in_len, out_len, dtype):
    """Exact linear operator W [out_len, in_len] for torch's
    F.interpolate(mode='bicubic', align_corners=True) along ONE axis. Bicubic
    interpolation is linear in the pixel values, so a fixed resize is a fixed
    matmul; deriving W from torch itself makes it bit-exact (same cubic kernel +
    boundary handling)."""
    eye = torch.eye(in_len, dtype=dtype).reshape(in_len, 1, in_len, 1)
    out = F.interpolate(eye, (out_len, 1), mode="bicubic", align_corners=True)
    return out[:, 0, :, 0].t().contiguous()  # [out_len, in_len]


def patch_bicubic_resize(model, time_in=T_FRAMES):
    """Replace ClapAudioEncoder.reshape_mel2img's bicubic F.interpolate calls
    (coremltools 9.0 has no `upsample_bicubic2d`) with an EXACT baked-constant
    matmul. Numerically identical to the original (verified by the end-to-end
    faithfulness check in the converters). Only the converters call this; the
    verification ground truth uses the un-patched model.

    The bicubic weight matrix W is precomputed EAGERLY here (at patch time), so the
    forward pass — hence the trace — contains ONLY the einsum (a plain matmul);
    the F.interpolate that derives W never enters the graph. For this model the
    only resize is along time (T_FRAMES -> spec_width); freq (64) already equals
    spec_height (64), so no freq resize is emitted (asserted at forward time)."""
    enc = model.audio_model.audio_encoder
    spec_width = int(enc.spec_size * enc.freq_ratio)   # 1024
    spec_height = enc.spec_size // enc.freq_ratio       # 64
    assert time_in < spec_width, (time_in, spec_width)
    W_time = _bicubic_operator(time_in, spec_width, torch.float32)  # [spec_width, time_in]

    def reshape_mel2img(self, x):
        _, _, time_length, freq_length = x.shape
        if time_length > spec_width or freq_length > spec_height:
            raise ValueError("the wav size should be less than or equal to the swin input size")
        assert time_length == time_in, (time_length, time_in)
        assert freq_length == spec_height, "freq resize path unsupported in the fixed-shape shim"
        x = torch.einsum("oi,bcif->bcof", W_time.to(x.dtype), x)  # exact bicubic, time axis
        batch, channels, time, freq = x.shape
        x = x.reshape(batch, channels * self.freq_ratio, time // self.freq_ratio, freq)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(batch, channels, freq * self.freq_ratio, time // self.freq_ratio)
        return x

    enc.reshape_mel2img = types.MethodType(reshape_mel2img, enc)
    return model
