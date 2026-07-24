"""Shared loader, pins, and asserts for the siglip2-naflex CoreML conversion.

Source of truth: the OFFICIAL public checkpoint ``google/siglip2-base-patch16-naflex``
pinned to ``REV`` below (Apache-2.0). The recipes convert FROM this official model;
nothing is consumed from any pre-uploaded artifact repo. Local staging only.

All filesystem paths come from the environment (never hardcoded — the clapkit
recipes' stale-absolute-path mistake):

  SIGLIP_CONV        base scratch dir (required)
  SIGLIP_SRC_MODEL   downloaded checkpoint dir      (default $SIGLIP_CONV/src-model)
  SIGLIP_STAGE       conversion staging dir         (default $SIGLIP_CONV/siglip/staging)
  SIGLIP_MODELS_OUT  final gitignored Models tree   (staging step only)
  SIGLIP_GOLDENS     committed goldens dir          (goldens step only)

The checkpoint is loaded from the local SHA-verified snapshot (SIGLIP_SRC_MODEL),
and every source file's SHA-256 is asserted against the pins below on load — a
stricter, offline-reproducible form of ``from_pretrained(..., revision=REV)``.
"""
import hashlib
import os

import numpy as np
import torch
from transformers import GemmaTokenizerFast, Siglip2ImageProcessor, Siglip2Model
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

MODEL_ID = "google/siglip2-base-patch16-naflex"
REV = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"

# Per-source-file SHA-256 at REV (verified on 2026-07-24). model.safetensors and
# tokenizer.json are the load-bearing weights + tokenizer identity; tokenizer.model
# is an advisory sentencepiece cross-check.
SOURCE_SHA256 = {
    "model.safetensors": "ac5f28bbdf92c0c1696ccbd3ce716426049cd67ad8045b66d0d938b0f9c8bbec",
    "tokenizer.json": "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
}

# Shipped shape tier (D2 in the port plan): the artifact shapes carry P/T; the Rust
# runtime resolves them from the loaded model, nothing is a code constant.
PATCH_BUDGET = 512          # vision pixel_values [1, P, 768]
TEXT_WINDOW = 64            # text input_ids [1, T]
EMBED_DIM = 768
POS_GRID_SIDE = 16          # base position grid 16x16 (num_patches=256)
PATCH_DIM = 3 * 16 * 16     # 768


def _env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"required environment variable {name} is unset")
    return val


def conv_dir():
    return _env("SIGLIP_CONV", required=True)


def src_dir():
    return _env("SIGLIP_SRC_MODEL", os.path.join(conv_dir(), "src-model"))


def stage_dir():
    d = _env("SIGLIP_STAGE", os.path.join(conv_dir(), "siglip", "staging"))
    os.makedirs(d, exist_ok=True)
    return d


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_sha():
    """Fail-closed: assert every pinned source file matches its SHA-256 at REV.
    Aborts (SystemExit) on any mismatch so a wrong/corrupt snapshot cannot cut
    goldens or artifacts."""
    src = src_dir()
    for name, want in SOURCE_SHA256.items():
        path = os.path.join(src, name)
        got = _sha256_file(path)
        if got != want:
            raise SystemExit(
                f"SOURCE SHA-256 MISMATCH for {name}:\n  got  {got}\n  want {want}\n"
                f"  (snapshot {src} is not {MODEL_ID}@{REV[:12]})"
            )
    print(f"[ok] source SHA-256 verified against {MODEL_ID}@{REV[:12]}")


def load_model(attn_implementation="sdpa"):
    """Return Siglip2Model (eval, fp32) from the SHA-verified local snapshot.

    Defaults to the checkpoint's ``sdpa`` attention (what the conversion probe
    measured); the converters may re-load with ``eager`` if coremltools 9.0 rejects
    an op in the sdpa lowering (exactness re-proven by the fp32 floor either way)."""
    verify_source_sha()
    model = Siglip2Model.from_pretrained(
        src_dir(), attn_implementation=attn_implementation
    ).eval()
    assert_config_defaults(model)
    return model


def load_slow_image_processor():
    """The SLOW ``Siglip2ImageProcessor`` (explicit class, NOT AutoProcessor / the
    declared Fast one): its uint8 PIL resize (pillow BILINEAR) is what the Rust
    colconv q8 path is bit-exact against. use_fast is irrelevant for an explicit
    slow class."""
    return Siglip2ImageProcessor.from_pretrained(src_dir())


def load_fast_tokenizer():
    """The FAST ``GemmaTokenizerFast`` — it executes the same tokenizer.json the
    Rust ``tokenizers`` crate bundles, so its ids are the Rust-side truth."""
    tok = GemmaTokenizerFast.from_pretrained(src_dir())
    assert_pad_side(tok)
    return tok


def assert_config_defaults(model):
    """Assert the library-default dims the minimal config.json relies on (§2), so a
    silent transformers-default drift is caught before it poisons an artifact."""
    v = model.config.vision_config
    t = model.config.text_config
    checks = [
        ("vision.hidden_size", v.hidden_size, EMBED_DIM),
        ("vision.num_hidden_layers", v.num_hidden_layers, 12),
        ("vision.num_attention_heads", v.num_attention_heads, 12),
        ("vision.patch_size", v.patch_size, 16),
        ("vision.num_patches", v.num_patches, 256),
        ("vision.num_channels", v.num_channels, 3),
        ("vision.layer_norm_eps", float(v.layer_norm_eps), 1e-6),
        ("text.hidden_size", t.hidden_size, EMBED_DIM),
        ("text.num_hidden_layers", t.num_hidden_layers, 12),
        ("text.num_attention_heads", t.num_attention_heads, 12),
        ("text.max_position_embeddings", t.max_position_embeddings, TEXT_WINDOW),
        ("text.projection_size", t.projection_size, EMBED_DIM),
        ("text.vocab_size", t.vocab_size, 256000),
        ("text.layer_norm_eps", float(t.layer_norm_eps), 1e-6),
    ]
    for name, got, want in checks:
        assert got == want, f"config default drift: {name} = {got!r}, expected {want!r}"
    print("[ok] config library-defaults asserted (dims 768/12/16, num_patches 256, T=64)")


def assert_pad_side(tokenizer):
    """The D6 pin: SigLIP2 pads RIGHT. transformers v5's Siglip2Tokenizer pads LEFT
    — cutting goldens under that would silently break the frozen Wave-A contract."""
    assert (
        tokenizer.padding_side == "right"
    ), f"padding_side must be 'right' (got {tokenizer.padding_side!r}); refuse to cut goldens"
    print("[ok] tokenizer padding_side == 'right'")


def base_pos_grid_f32(model):
    """The base 16x16x768 position grid as a row-major [16, 16, 768] float32 numpy
    array — ``vision_model.embeddings.position_embedding.weight`` ([256, 768])
    reshaped exactly as the official ``resize_positional_embeddings`` reshapes it."""
    w = model.vision_model.embeddings.position_embedding.weight  # [256, 768]
    grid = w.reshape(POS_GRID_SIDE, POS_GRID_SIDE, EMBED_DIM)
    return grid.detach().to(torch.float32).cpu().numpy()


def official_lift(model, spatial_shapes, max_length=PATCH_BUDGET):
    """The OFFICIAL host-side position-embedding lift: call the checkpoint's own
    ``Siglip2VisionEmbeddings.resize_positional_embeddings`` (bilinear, antialias,
    align_corners=False) — NOT a reimplementation — so the lifted tensor is exactly
    what the stock ``get_image_features`` computes internally.

    ``spatial_shapes`` is a [B, 2] LongTensor of (grid_h, grid_w). Returns a
    [B, max_length, 768] float32 tensor (pad rows filled with resized[0], the
    reference convention; the Rust port zero-fills pads — attention-masked, so the
    embedding is invariant)."""
    w = model.vision_model.embeddings.position_embedding.weight.detach()
    grid = w.reshape(POS_GRID_SIDE, POS_GRID_SIDE, EMBED_DIM).to(torch.float32)
    return Siglip2VisionEmbeddings.resize_positional_embeddings(
        grid, spatial_shapes, max_length=max_length
    )


def padded_ids(fast_tokenizer, text, window=TEXT_WINDOW):
    """The exact [window] padded input_ids the module feeds the text graph.

    The lowercase protocol (§7, byte-parity-critical): 4.53.3's GemmaTokenizerFast
    does NOT lowercase, but the module composes a ``tokenizers`` ``Lowercase``
    normalizer ahead of the loaded one. We therefore lowercase here with the SAME
    ``tokenizers.normalizers.Lowercase`` the Rust side composes, assert it equals
    ``text.lower()`` (true for ASCII), then tokenize with padding=max_length,
    truncation, and add_eos (the checkpoint's ``padding_side: right`` /
    ``add_eos_token: true``). Returns (ids[window] list, lowercased text)."""
    import tokenizers

    lower = tokenizers.normalizers.Lowercase().normalize_str(text)
    assert lower == text.lower(), f"lowercase divergence: {lower!r} != {text.lower()!r}"
    enc = fast_tokenizer(lower, padding="max_length", truncation=True, max_length=window)
    ids = list(enc["input_ids"])
    assert len(ids) == window, f"padded window length {len(ids)} != {window}"
    return ids, lower


def cos(a, b):
    """Cosine of two vectors in float64, NO epsilon guard — a non-finite artifact
    MUST propagate to NaN (fail-closed), never be masked to a finite-looking value."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
