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
import copy
import math
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
    assert_toolchain_pins,
    load_model,
    load_slow_image_processor,
    official_lift,
    stage_dir,
    write_toolchain_sidecar,
)
from _fixtures import IMAGES, load_pil

FAITHFUL_FLOOR = 0.999999


class ClampedTanhGelu(nn.Module):
    """``gelu_pytorch_tanh`` written out elementwise for the Apple Neural Engine.

    The tanh is evaluated on ``clamp(x, -10, 10)`` and the tails are selected
    exactly: ``x`` for ``x > 10``, ``0`` for ``x < -10``. MEASURED in the fp16 CoreML
    pipeline against torch's ``F.gelu(approximate="tanh")`` over ±1200 (dense on
    ±40, negative tail included; ``README.md`` §"The ANE rewrite"): ``max|Δ| = 0`` for
    ``x < -20``; ``0.0083`` on ``|x| <= 20`` (fp16 arithmetic of the elementwise form;
    the stock MIL ``gelu`` op measures 4.8e-7 there); and the fp16 rounding of ``x``
    itself for ``x > 20`` (``0.498`` at ``x = 1111.5``, the input's own half-ulp). It is
    NOT bit-identical to the stock op in the mid-range — the tower-level cost on the
    GPU is cos 0.99999488 → 0.99999429 (max|Δ| 0.0065 on both), a systematic and
    measured change, not rounding noise.

    Why it exists (issue #51): the stock MIL ``gelu`` op is exact on every compute
    unit (it does not overflow anywhere in ±1200 — only THIS decomposed cubic would,
    at ``|x| ≈ 40``, which is what the clamp is for, ``fc1`` reaching ±1172 in layer
    9), but on the ANE the fused ``linear → gelu`` path is coarser than the
    elementwise ops, and over the twelve encoder layers that alone holds the ANE arm
    at cos 0.99796 (below the 0.99917 gate) with the head already fixed. Pinning
    only LayerNorm and softmax to fp32 changes nothing; this form takes the ANE arm
    to 0.99992. Forms measured and rejected: ``x·sigmoid(2·inner)`` (exact negative
    tail, but the ANE arm drops to 0.99899); the plain clamped tanh without the tail
    selects (emits ``x/2048`` for ``x < -10``, because ``1 + tanh`` never reaches 0 in
    fp16); a clamp-ramp gate (leaves ≤ 0.005 on ``(-10, -9)``). The ``where`` selects
    cost ANE latency (≈ 51 ms/image vs ≈ 38 ms without them, this host), not GPU."""

    def forward(self, x):
        xc = x.clamp(-10.0, 10.0)
        inner = math.sqrt(2.0 / math.pi) * (xc + 0.044715 * xc * xc * xc)
        g = 0.5 * x * (1.0 + torch.tanh(inner))
        return torch.where(x > 10.0, x, torch.where(x < -10.0, torch.zeros_like(x), g))


class ManualMAPHead(nn.Module):
    """``Siglip2MultiheadAttentionPoolingHead`` with its ``nn.MultiheadAttention``
    written out explicitly — the SAME parameters (``probe``, packed ``in_proj``,
    ``out_proj``, ``layernorm``, ``mlp``), the same math, one probe query over the
    ``P`` patch tokens under the additive pad mask. Verified identical to
    ``nn.MultiheadAttention`` at torch 2.5.1 / transformers 4.53.3 (the structural
    asserts below are what that identity rests on; a future revision that changes
    the module fails loudly here instead of silently converting something else).

    Why (issue #51): on this host's Neural Engine coremltools' lowering of
    ``F.multi_head_attention_forward`` for this single-query head computes WRONG —
    head-only fp16 graph cos 0.751 vs fp32 (0.999999 on the GPU), with or without
    the pad mask and at every mask magnitude — and that is the entire vision
    collapse (encoder residual stream on the ANE cos ≥ 0.998 at every layer;
    ``features`` 0.31). Measured with the stock head unchanged and its averaged
    attention weights exported as a second output: on the ANE they are a proper
    distribution (non-negative, sum 1.0000–1.0005 over ``P``, exactly 0 on every
    pad token) but the WRONG one (cos 0.316 vs fp32) — so the fault is in the
    pre-softmax scores, not in the softmax or the masking. It was not isolated
    below that: every piece of the stock lowering — the ``(P, B, D)`` packed
    ``linear``, its 5-D-transpose K/V split, the ``perm=(1,2,0)`` K transpose, the
    constant-``q`` matmul, the softmax — is correct on the ANE in isolation, and a
    step-for-step mirror with intermediate outputs is correct too; the fused whole
    is not. Every reformulation tried is correct (0.999995 head-only on the ANE);
    this one lowers to plain ``linear``/``matmul``/``softmax`` ops.

    The mask is the finite ``(1 - mask) * -1e4`` additive form rather than
    ``masked_fill(finfo.min)``: the two are equal only for a strictly binary mask,
    which is what this graph receives — the processor's ``pixel_attention_mask`` is
    0/1, and the Rust door refuses anything else before predicting
    (``validate_mask`` in ``src/embeddings/siglip/image/mod.rs``); ``exp(-1e4)`` is
    exactly 0 in fp16 and fp32, and a finite constant survives every cast."""

    def __init__(self, head):
        super().__init__()
        attn = head.attention
        D = head.probe.shape[-1]
        assert isinstance(attn, nn.MultiheadAttention), type(attn)
        assert attn.batch_first is True, "the explicit form assumes (B, L, D) inputs"
        assert attn.dropout == 0.0, attn.dropout
        assert attn._qkv_same_embed_dim, "packed in_proj assumed (q, k, v share D)"
        assert not attn.add_zero_attn, "add_zero_attn would append a key the explicit form lacks"
        assert attn.bias_k is None and attn.bias_v is None, "bias_k/bias_v not modelled"
        assert attn.in_proj_weight.shape == (3 * D, D), attn.in_proj_weight.shape
        assert attn.in_proj_bias is not None and attn.in_proj_bias.shape == (3 * D,)
        assert attn.num_heads * attn.head_dim == D, (attn.num_heads, attn.head_dim, D)
        assert tuple(head.probe.shape) == (1, 1, D), head.probe.shape
        self.probe = head.probe
        self.num_heads = attn.num_heads
        self.in_proj_weight = attn.in_proj_weight
        self.in_proj_bias = attn.in_proj_bias
        self.out_proj = attn.out_proj
        self.layernorm = head.layernorm
        self.mlp = head.mlp

    def forward(self, h, attention_mask):
        B, P, D = h.shape
        H = self.num_heads
        d = D // H
        W, bb = self.in_proj_weight, self.in_proj_bias
        probe = self.probe.repeat(B, 1, 1)
        q = F.linear(probe, W[:D], bb[:D]) * (d ** -0.5)
        k = F.linear(h, W[D : 2 * D], bb[D : 2 * D])
        v = F.linear(h, W[2 * D :], bb[2 * D :])
        q = q.view(B, 1, H, d).transpose(1, 2)
        k = k.view(B, P, H, d).transpose(1, 2)
        v = v.view(B, P, H, d).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) + ((1.0 - attention_mask) * -1e4)[:, None, None, :]
        ctx = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, 1, D)
        a = self.out_proj(ctx)
        return (a + self.mlp(self.layernorm(a)))[:, 0]


class VisionTower(nn.Module):
    """pixel_values, position_embeddings, attention_mask -> image_features (pre-norm).

    Exactly Siglip2VisionTransformer.forward with the position embeddings lifted to
    an input: patch_embedding(pv) + position_embeddings -> encoder(additive 4d mask)
    -> post_layernorm -> multihead-attention-pooling head (raw [1, P] mask) — with
    two weight-preserving rewrites for the Apple Neural Engine (issue #51, see
    ClampedTanhGelu and ManualMAPHead): every MLP activation is the elementwise
    tanh-GELU with exact tails, and the pooling head is written out explicitly. The
    WHOLE vision model is deep-copied once, so nothing here shares a module with
    the model the pre-trace faithfulness assert compares against — that assert can
    never compare the model to itself."""

    def __init__(self, m):
        super().__init__()
        vm = copy.deepcopy(m.vision_model)
        self.patch_embedding = vm.embeddings.patch_embedding  # Linear 768 -> 768
        self.encoder = vm.encoder
        for layer in self.encoder.layers:
            layer.mlp.activation_fn = ClampedTanhGelu()
        self.post_layernorm = vm.post_layernorm
        vm.head.mlp.activation_fn = ClampedTanhGelu()
        self.head = ManualMAPHead(vm.head)

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
    assert_toolchain_pins()
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
    print(f"SAVED {write_toolchain_sidecar(stage, 'vision')}  (observed toolchain, #97)")
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
