"""Shared loader + constants for the coremlit CED conversion recipes.

Source of truth: the four OFFICIAL public checkpoints ``mispeech/ced-{tiny,mini,small,base}``
(Apache-2.0), each pinned to the revision in ``MODELS`` below and hash-verified against the
recorded ``model.safetensors`` SHA-256 at load. The four are contract-identical for coremlit —
one ``mel [1, 64, 1001]`` f32 -> ``logits [1, 527]`` f32 (PRE-sigmoid) graph — differing only in
transformer width (embed_dim / num_heads).

The CoreML graph is converted from the PyTorch ``CedForAudioClassification`` (trust_remote_code):
we trace the ``mel -> logits`` sub-forward and DROP the final sigmoid (``pooling="mean"`` ends in
``.sigmoid()``), because the coremlit contract is PRE-sigmoid (Rust applies the sigmoid). The
extracted wrapper is the exact pre-sigmoid of the unmodified forward
(``sigmoid(wrapper) == forward.logits``), which is ALSO the golden oracle (PyTorch fp32).

Paths are env-driven (no hardcoded scratchpad — the clap recipe's wart is fixed here):
  CED_CONV        working dir holding ``src-<size>`` snapshots + ``staging`` (default:
                  ``~/.cache/coremlit-ced-conv``).
  CED_MODELS_OUT  where the shipped fp16 ``.mlmodelc`` bundles are staged (default: the repo's
                  gitignored ``Models/ced``).
  HF_HOME         Hugging Face cache (default: ``$CED_CONV/hf-cache``).
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

# --- pinned sources (revision + expected model.safetensors/model.onnx SHA-256) -------------
# (repo, revision, safetensors_sha256, onnx_sha256, embed_dim, num_heads)
MODELS = {
    "tiny": ("mispeech/ced-tiny", "ace276d29dd0bb3f3517b0fa8cf300738c409019",
             "0e086f0cd62814c6def89001f3f25193f75955696f6975ef6800af31d00d6dd7",
             "3890f227660be71886ff48633d11a731993e0206db8a486781412c28a0873435", 192, 3),
    "mini": ("mispeech/ced-mini", "26c3ebcae85d4330f4fc26763f029539a3afcda0",
             "e4070d02ca53ec7a1df83bbf779a0e70bc2bbc0a9bacde6330e1dc37c990b215",
             "2a86374dbe1fa03e96b0acf6a02618ac71852d12aa3d72a030b062ff1e5c5434", 256, 4),
    "small": ("mispeech/ced-small", "06bb40c5ec089e96867ebc5246be02441f4a71e4",
              "a495e319a620dec4c8b7ca7bb927667ed68169a9efc1a943ccbf5d5c6f9c0c71",
              "18fa4fa30c1872c322c6b08f2824c9dd6f7fe149b8aa21320ddceb77007cff75", 384, 6),
    "base": ("mispeech/ced-base", "db3e14a8db4c21b56b165261c39649741a900e7f",
             "314935693ed1dcef07576ca0c41277c51c642f3847bc5eb03918c5277eb79af9",
             "1cb33c4300b6c52ae099a5af72058982e673ec79862855961b8b8c10eeaba74c", 768, 12),
}

# Frozen family contract (matches src/audio/ced: NUM_CLASSES / mel geometry) --------------------
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 160_000          # 10 s @ 16 kHz — the fixed inference window
N_MELS = 64
N_FRAMES = 1001                   # 1 + WINDOW_SAMPLES/HOP (hop 160, center=True)
NUM_CLASSES = 527
SIZES = ("tiny", "mini", "small", "base")

_REPO_ROOT = Path(__file__).resolve().parents[5]   # scripts/ -> ced -> conversion -> coremlit -> crates -> <root>


def conv_dir() -> Path:
    return Path(os.environ.get("CED_CONV", str(Path.home() / ".cache" / "coremlit-ced-conv")))


def models_out_dir() -> Path:
    return Path(os.environ.get("CED_MODELS_OUT", str(_REPO_ROOT / "Models" / "ced")))


def staging_dir() -> Path:
    d = conv_dir() / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def src_dir(size: str) -> Path:
    return conv_dir() / f"src-{size}"


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(size: str):
    """Snapshot the pinned revision into ``src-<size>`` and FAIL CLOSED on a SHA mismatch."""
    os.environ.setdefault("HF_HOME", str(conv_dir() / "hf-cache"))
    from huggingface_hub import snapshot_download
    repo, rev, st_sha, onnx_sha, _, _ = MODELS[size]
    d = Path(snapshot_download(repo, revision=rev, local_dir=str(src_dir(size))))
    for fname, want in (("model.safetensors", st_sha), ("model.onnx", onnx_sha)):
        got = sha256_file(d / fname)
        if got != want:
            raise SystemExit(f"FATAL {size}: {fname} sha256 {got} != pinned {want}")
    return d


def load_model(size: str):
    """Return ``CedForAudioClassification`` (eval, fp32) for ``size``, config asserted."""
    from transformers import AutoModelForAudioClassification
    repo, rev, _, _, embed_dim, num_heads = MODELS[size]
    m = AutoModelForAudioClassification.from_pretrained(
        str(src_dir(size)), trust_remote_code=True).eval()
    c = m.config
    assert c.name == f"ced-{size}", (c.name, size)
    assert (c.embed_dim, c.num_heads) == (embed_dim, num_heads), (c.embed_dim, c.num_heads)
    assert c.outputdim == NUM_CLASSES and c.n_mels == N_MELS, (c.outputdim, c.n_mels)
    assert c.pooling == "mean", c.pooling
    return m


def load_feature_extractor(size: str):
    from transformers import AutoFeatureExtractor
    return AutoFeatureExtractor.from_pretrained(str(src_dir(size)), trust_remote_code=True)


class CedMelToLogits(torch.nn.Module):
    """``mel [1, 64, 1001]`` f32 -> ``logits [1, 527]`` f32 (PRE-sigmoid).

    Reuses the REAL encoder forward (unsqueeze -> permute -> ``init_bn`` BatchNorm2d eval ->
    permute -> T=1001<=1012 so no split -> ``forward_features``) then mean-pools the patches and
    applies ``outputlayer`` WITHOUT the final sigmoid (== ``forward_head`` for pooling="mean"
    minus its ``.sigmoid()``)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, mel):
        feats = self.m.encoder(mel).logits          # [1, 248, embed_dim]
        return self.m.outputlayer(feats.mean(1))    # [1, 527] pre-sigmoid


def mel_for_waveform(fe, wav_f32):
    """torchaudio ``CedFeatureExtractor`` mel of a 1-D f32 waveform, zero-padded to the fixed
    WINDOW_SAMPLES first — mirroring ``Classifier::raw_scores`` (a sub-window clip is padded to
    the 10 s window before the mel, giving the same 1001-frame mel the Rust path builds).
    Returns a torch tensor ``[1, 64, 1001]``."""
    w = np.asarray(wav_f32, dtype=np.float32)
    if w.shape[0] > WINDOW_SAMPLES:
        raise ValueError(f"waveform {w.shape[0]} > WINDOW_SAMPLES {WINDOW_SAMPLES}")
    if w.shape[0] < WINDOW_SAMPLES:
        w = np.pad(w, (0, WINDOW_SAMPLES - w.shape[0]))
    feat = fe(w, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"]
    assert tuple(feat.shape) == (1, N_MELS, N_FRAMES), tuple(feat.shape)
    return feat


def cos(a, b) -> float:
    """Cosine WITHOUT an eps guard — a non-finite artifact MUST propagate to NaN (clap's
    fail-closed rule), never be masked into a finite-looking value."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
