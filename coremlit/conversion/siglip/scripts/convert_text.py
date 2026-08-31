"""Convert the SigLIP2 text tower -> CoreML.

Contract (single input — the SigLIP no-attention-mask convention; the checkpoint's
tokenizer model_input_names == ["input_ids"]):
  input : input_ids     int32 [1, 64]
  output: text_features fp32  [1, 768]   (pre-L2-norm; the caller normalizes)

attention_mask=None gives full bidirectional attention over the padded window; the
tower pools the FINAL position (sticky-EOS convention). Faithfulness is proven vs
the UNMODIFIED model.get_text_features over every corpus text before tracing.

Produces BOTH precisions: fp16 (shipped) + fp32 (verification reference). The
256000x768 token embedding makes the fp16 weight ~393 MB and is WHY the whole-graph
ANECCompile fails (known, characterized) — CoreML runs it on the GPU.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from _siglip_common import (
    EMBED_DIM,
    MODEL_ID,
    REV,
    TEXT_WINDOW,
    cos,
    load_fast_tokenizer,
    load_model,
    padded_ids,
    stage_dir,
)
from _fixtures import TEXTS

FAITHFUL_FLOOR = 0.999999


class TextTower(nn.Module):
    """input_ids [1, 64] -> text_features [1, 768] (pre-norm)."""

    def __init__(self, m):
        super().__init__()
        self.text_model = m.text_model  # Siglip2TextTransformer

    def forward(self, input_ids):
        return self.text_model(input_ids=input_ids, attention_mask=None).pooler_output


def build_and_convert(attn):
    model = load_model(attn_implementation=attn)
    tok = load_fast_tokenizer()
    net = TextTower(model).eval()

    fixtures = []
    for entry in TEXTS:
        ids, _lower = padded_ids(tok, entry["text"], TEXT_WINDOW)
        fixtures.append((entry["id"], torch.tensor([ids], dtype=torch.long)))

    worst = 1.0
    for tid, ids in fixtures:
        with torch.no_grad():
            wrap = net(ids).numpy()
            stock = model.get_text_features(input_ids=ids).numpy()
        c = cos(wrap, stock)
        worst = min(worst, c)
        print(f"  [faithful] {tid:15s} wrapper-vs-get_text_features cos = {c:.8f}")
    print(f"[CHECK] text pre-trace worst faithfulness cos = {worst:.8f} (attn={attn})")
    if not (worst >= FAITHFUL_FLOOR):
        raise SystemExit(f"text tower UNFAITHFUL: worst {worst:.8f} < {FAITHFUL_FLOOR}")

    ids0 = fixtures[0][1]
    ts = torch.jit.trace(net, (ids0,), check_trace=False)
    worst_tr = 1.0
    for _, ids in fixtures:
        with torch.no_grad():
            worst_tr = min(worst_tr, cos(ts(ids).numpy(), net(ids).numpy()))
    print(f"[CHECK] text traced-vs-eager worst cos = {worst_tr:.8f}")
    if not (worst_tr >= FAITHFUL_FLOOR):
        raise SystemExit(f"text trace UNFAITHFUL: {worst_tr:.8f} < {FAITHFUL_FLOOR}")

    stage = stage_dir()
    for tag, prec in (("", ct.precision.FLOAT16), ("_fp32", ct.precision.FLOAT32)):
        ml = ct.convert(
            ts,
            inputs=[ct.TensorType(name="input_ids", shape=(1, TEXT_WINDOW), dtype=np.int32)],
            outputs=[ct.TensorType(name="text_features", dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=prec,
            convert_to="mlprogram",
        )
        ml.author = f"coremlit siglip: {MODEL_ID}@{REV[:12]} text tower (Gemma-tokenized), pre-norm"
        ml.short_description = (
            "SigLIP2 text encoder: input_ids [1,64] -> 768-d joint embedding "
            "(no attention_mask); L2-norm applied by the caller"
        )
        out = os.path.join(stage, f"siglip2_text_64{tag}.mlpackage")
        ml.save(out)
        print(f"SAVED {out}  ({prec})")

    with open(os.path.join(stage, "attn_impl_text.txt"), "w") as f:
        f.write(attn + "\n")
    print(f"DONE text (attn={attn}, faithfulness {worst:.8f})")


def main():
    forced = os.environ.get("SIGLIP_ATTN")
    order = [forced] if forced else ["sdpa", "eager"]
    last = None
    for attn in order:
        try:
            build_and_convert(attn)
            return
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[fallback] text convert with attn={attn} failed: {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(f"text conversion failed on all attention impls: {last}")


if __name__ == "__main__":
    main()
