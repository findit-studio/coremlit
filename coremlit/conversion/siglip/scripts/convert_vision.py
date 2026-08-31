"""Convert the SigLIP2 vision tower -> CoreML (the NaFlex -> fixed-shape crux).

Contract (the position-embedding LIFT is host-side; the graph is fully static):
  inputs : pixel_values        fp32 [1, 512, 768]  (patchified, normalized)
           position_embeddings fp32 [1, 512, 768]  (the OFFICIAL lift, host-side)
           attention_mask      fp32 [1, 512]       (1.0 real prefix, 0.0 pad)
  output : image_features      fp32 [1, 768]       (pre-L2-norm; the caller normalizes)

The stock Siglip2VisionEmbeddings runs a per-image F.interpolate(antialias) of the
position grid whose target size is DATA (spatial_shapes) — that cannot trace to ONE
static graph. We hoist that resize OUT: the Rust runtime computes it per image and
feeds it as position_embeddings. This wrapper is byte-for-byte the stock
Siglip2VisionTransformer.forward with the position embeddings supplied instead of
recomputed — proven by the pre-trace faithfulness assert (>= 0.999999) vs the
UNMODIFIED model.get_image_features over every fixture image.

Also emits the base position-grid sidecar pos_embed_16x16x768.f32le.bin (786432 B).
Produces BOTH precisions: fp16 (shipped) + fp32 (verification reference).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

sys.path.insert(0, os.path.dirname(__file__))
from _siglip_common import (
    EMBED_DIM,
    MODEL_ID,
    PATCH_BUDGET,
    PATCH_DIM,
    POS_GRID_SIDE,
    REV,
    base_pos_grid_f32,
    cos,
    load_model,
    load_slow_image_processor,
    official_lift,
    stage_dir,
)
from _fixtures import IMAGES, load_pil

FAITHFUL_FLOOR = 0.999999


class VisionTower(nn.Module):
    """pixel_values, position_embeddings, attention_mask -> image_features (pre-norm).

    Exactly Siglip2VisionTransformer.forward with the position embeddings lifted to
    an input: patch_embedding(pv) + position_embeddings -> encoder(additive 4d mask)
    -> post_layernorm -> multihead-attention-pooling head (raw [1, P] mask)."""

    def __init__(self, m):
        super().__init__()
        vm = m.vision_model
        self.patch_embedding = vm.embeddings.patch_embedding  # Linear 768 -> 768
        self.encoder = vm.encoder
        self.post_layernorm = vm.post_layernorm
        self.head = vm.head  # Siglip2MultiheadAttentionPoolingHead

    def forward(self, pixel_values, position_embeddings, attention_mask):
        h = self.patch_embedding(pixel_values) + position_embeddings
        enc_mask = _prepare_4d_attention_mask(attention_mask, h.dtype)
        h = self.encoder(inputs_embeds=h, attention_mask=enc_mask).last_hidden_state
        h = self.post_layernorm(h)
        return self.head(h, attention_mask)


def fixture_tensors(proc, model):
    """For every corpus image, the slow-processor tensors at the 512 budget plus the
    OFFICIAL lifted position embeddings: (id, pv[1,512,768], pos[1,512,768],
    mask[1,512] f32, spatial_shapes[1,2])."""
    out = []
    for entry in IMAGES:
        img = load_pil(entry["id"])
        feats = proc(images=[img], max_num_patches=PATCH_BUDGET, return_tensors="pt")
        pv = feats["pixel_values"].to(torch.float32)
        mask = feats["pixel_attention_mask"].to(torch.float32)
        ss = feats["spatial_shapes"]
        assert tuple(pv.shape) == (1, PATCH_BUDGET, PATCH_DIM), pv.shape
        assert tuple(mask.shape) == (1, PATCH_BUDGET), mask.shape
        pos = official_lift(model, ss, max_length=PATCH_BUDGET).to(torch.float32)
        out.append((entry["id"], pv, pos, mask, ss))
    return out


def build_and_convert(attn):
    model = load_model(attn_implementation=attn)
    proc = load_slow_image_processor()
    net = VisionTower(model).eval()

    fixtures = fixture_tensors(proc, model)

    # Pre-trace faithfulness (mandatory): the lift-wrapper == the stock forward.
    worst = 1.0
    for iid, pv, pos, mask, ss in fixtures:
        with torch.no_grad():
            wrap = net(pv, pos, mask).numpy()
            stock = model.get_image_features(
                pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=ss
            ).numpy()
        c = cos(wrap, stock)
        worst = min(worst, c)
        print(f"  [faithful] {iid:9s} wrapper-vs-get_image_features cos = {c:.8f}")
    print(f"[CHECK] vision pre-trace worst faithfulness cos = {worst:.8f} (attn={attn})")
    if not (worst >= FAITHFUL_FLOOR):
        raise SystemExit(
            f"vision lift-wrapper UNFAITHFUL: worst {worst:.8f} < {FAITHFUL_FLOOR}"
        )

    # Trace on a real fixture (exact input shapes), then re-assert traced vs eager.
    _, pv0, pos0, mask0, _ = fixtures[0]
    ts = torch.jit.trace(net, (pv0, pos0, mask0), check_trace=False)
    worst_tr = 1.0
    for _, pv, pos, mask, _ in fixtures:
        with torch.no_grad():
            worst_tr = min(worst_tr, cos(ts(pv, pos, mask).numpy(), net(pv, pos, mask).numpy()))
    print(f"[CHECK] vision traced-vs-eager worst cos = {worst_tr:.8f}")
    if not (worst_tr >= FAITHFUL_FLOOR):
        raise SystemExit(f"vision trace UNFAITHFUL: {worst_tr:.8f} < {FAITHFUL_FLOOR}")

    stage = stage_dir()
    for tag, prec in (("", ct.precision.FLOAT16), ("_fp32", ct.precision.FLOAT32)):
        ml = ct.convert(
            ts,
            inputs=[
                ct.TensorType(name="pixel_values", shape=(1, PATCH_BUDGET, PATCH_DIM), dtype=np.float32),
                ct.TensorType(name="position_embeddings", shape=(1, PATCH_BUDGET, EMBED_DIM), dtype=np.float32),
                ct.TensorType(name="attention_mask", shape=(1, PATCH_BUDGET), dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="image_features", dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=prec,
            convert_to="mlprogram",
        )
        ml.author = f"coremlit siglip: {MODEL_ID}@{REV[:12]} vision tower (NaFlex, host-lifted pos-emb), pre-norm"
        ml.short_description = (
            "SigLIP2 vision encoder: pixel_values/position_embeddings/attention_mask "
            "[1,512,*] -> 768-d joint embedding; L2-norm applied by the caller"
        )
        out = os.path.join(stage, f"siglip2_vision_512{tag}.mlpackage")
        ml.save(out)
        print(f"SAVED {out}  ({prec})")

    # Base position-grid sidecar (row-major 16x16x768 f32 LE, 786432 bytes).
    grid = base_pos_grid_f32(model)  # [16, 16, 768]
    assert grid.shape == (POS_GRID_SIDE, POS_GRID_SIDE, EMBED_DIM), grid.shape
    sidecar = os.path.join(stage, "pos_embed_16x16x768.f32le.bin")
    grid.astype("<f4").tofile(sidecar)
    nbytes = os.path.getsize(sidecar)
    assert nbytes == POS_GRID_SIDE * POS_GRID_SIDE * EMBED_DIM * 4 == 786_432, nbytes
    print(f"SAVED {sidecar}  ({nbytes} bytes)")

    # Record which attention lowering produced the artifact (for MANIFEST).
    with open(os.path.join(stage, "attn_impl_vision.txt"), "w") as f:
        f.write(attn + "\n")
    print(f"DONE vision (attn={attn}, faithfulness {worst:.8f})")


def main():
    forced = os.environ.get("SIGLIP_ATTN")
    order = [forced] if forced else ["sdpa", "eager"]
    last = None
    for attn in order:
        try:
            build_and_convert(attn)
            return
        except SystemExit:
            raise  # a faithfulness breach is a finding, not a fallback trigger
        except Exception as e:  # noqa: BLE001 — only a converter/op failure falls back
            last = e
            print(f"[fallback] vision convert with attn={attn} failed: {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(f"vision conversion failed on all attention impls: {last}")


if __name__ == "__main__":
    main()
