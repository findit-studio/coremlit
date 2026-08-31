"""Convert the CLAP text tower (RoBERTa + text_projection) -> CoreML.

Source: laion/clap-htsat-unfused (transformers ClapModel), revision-pinned.
Contract (tokenized ids/mask as inputs; fixed length 512 = the model's max, which
reproduces natural-length embeddings exactly via the attention mask + RoBERTa
position derivation — verified cos=1.0 in probe_text_pad.py):
  inputs : input_ids       int32 [1, 512]
           attention_mask   int32 [1, 512]
  output : text_embeds      fp32  [1, 512]   (projection output, PRE-L2-norm)

L2 normalization is intentionally OUT of the graph (clapkit normalizes in Rust).

Produces BOTH precisions:
  clap_text.mlpackage       compute_precision=FLOAT16 (shipped candidate)
  clap_text_fp32.mlpackage  compute_precision=FLOAT32 (verification reference)
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
from _clap_common import load_model, load_processor

STAGE = "/private/tmp/claude-501/-Users-al-Developer-findit-studio-coremlit/2e543e17-c5e2-4187-be75-b6b4fafe4418/scratchpad/conv/clapkit/staging"
SEQ = 512
os.makedirs(STAGE, exist_ok=True)


class TextTower(nn.Module):
    """input_ids/attention_mask [1,512] -> text_embeds [1,512] (pre-norm)."""

    def __init__(self, m):
        super().__init__()
        self.text_model = m.text_model
        self.text_projection = m.text_projection

    def forward(self, input_ids, attention_mask):
        pooled = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.text_projection(pooled)


def main():
    model = load_model()
    proc = load_processor()
    net = TextTower(model).eval()

    tok = proc.tokenizer(["a dog barking loudly in the distance"],
                         padding="max_length", truncation=True, max_length=SEQ, return_tensors="pt")
    ids, mask = tok["input_ids"], tok["attention_mask"]
    assert tuple(ids.shape) == (1, SEQ), ids.shape

    with torch.no_grad():
        pre = net(ids, mask)
        ref = model.get_text_features(input_ids=ids, attention_mask=mask)
        ref = ref if torch.is_tensor(ref) else ref.pooler_output
        cos = F.cosine_similarity(F.normalize(pre, dim=-1), ref, dim=-1).min().item()
    print(f"[CHECK] text pre-norm shape {tuple(pre.shape)}  "
          f"normalize(wrapper)-vs-get_text_features cos = {cos:.8f}")
    assert cos > 0.9999, f"text wrapper unfaithful: {cos}"

    ts = torch.jit.trace(net, (ids, mask), check_trace=False)

    for tag, prec in (("", ct.precision.FLOAT16), ("_fp32", ct.precision.FLOAT32)):
        ml = ct.convert(
            ts,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, SEQ), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, SEQ), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="text_embeds", dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=prec,
            convert_to="mlprogram",
        )
        ml.author = "clapkit T1: laion/clap-htsat-unfused text tower (RoBERTa + text_projection), pre-norm"
        ml.short_description = ("CLAP text encoder: input_ids/attention_mask [1,512] -> "
                                "512-d joint embedding, L2-norm applied by the caller")
        out = os.path.join(STAGE, f"clap_text{tag}.mlpackage")
        ml.save(out)
        print(f"SAVED {out}  ({prec})")
    print("DONE text")


if __name__ == "__main__":
    main()
