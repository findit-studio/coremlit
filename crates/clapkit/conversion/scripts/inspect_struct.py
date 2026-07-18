"""Module structure + I/O contract of both CLAP towers (trusted load)."""
import numpy as np
import torch
from _clap_common import load_model, load_processor, mel_features

model = load_model()
proc = load_processor()

print("=== logit scales ===")
print("logit_scale_a (exp):", float(model.logit_scale_a.exp()))
print("logit_scale_t (exp):", float(model.logit_scale_t.exp()))
print("audio_projection:", model.audio_projection)
print("text_projection:", model.text_projection)

print("\n=== AUDIO TOWER children ===")
for n, m in model.audio_model.named_children():
    print(" ", n, type(m).__name__)
print("=== TEXT TOWER children ===")
for n, m in model.text_model.named_children():
    print(" ", n, type(m).__name__)

# count layer_norms + their eps in text tower
print("\n=== TEXT layer_norm eps values ===")
eps_set = set()
for n, m in model.text_model.named_modules():
    if isinstance(m, torch.nn.LayerNorm):
        eps_set.add(m.eps)
print("  distinct eps:", eps_set, " (count LN modules:",
      sum(isinstance(m, torch.nn.LayerNorm) for _, m in model.text_model.named_modules()), ")")
print("=== AUDIO layer_norm eps values ===")
eps_a = set()
for n, m in model.audio_model.named_modules():
    if isinstance(m, torch.nn.LayerNorm):
        eps_a.add(m.eps)
print("  distinct eps:", eps_a, " (count:",
      sum(isinstance(m, torch.nn.LayerNorm) for _, m in model.audio_model.named_modules()), ")")

# Feature extractor output + get_audio_features / get_text_features shapes.
rng = np.random.RandomState(0)
audio = rng.randn(480000).astype(np.float32) * 0.1
feats = mel_features(proc, audio)
print("\n=== feature extractor output ===")
for k, v in feats.items():
    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}",
          ("val=" + str(v.item()) if v.numel() == 1 else ""))

with torch.no_grad():
    af = model.get_audio_features(input_features=feats["input_features"],
                                  is_longer=feats.get("is_longer"))
print("audio_features:", tuple(af.shape), "norm:", float(af.norm(dim=-1)[0]))

tok = proc.tokenizer(["a dog barking", "rain"], padding=True, return_tensors="pt")
print("tok input_ids:", tuple(tok["input_ids"].shape), "mask:", tuple(tok["attention_mask"].shape))
print("input_ids[0]:", tok["input_ids"][0].tolist())
with torch.no_grad():
    tf = model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
print("text_features:", tuple(tf.shape), "norm[0]:", float(tf.norm(dim=-1)[0]))

# Does get_text_features depend on attention_mask? Compare padded vs unpadded single.
tok1 = proc.tokenizer(["a dog barking"], padding=False, return_tensors="pt")
with torch.no_grad():
    tf1 = model.get_text_features(input_ids=tok1["input_ids"], attention_mask=tok1["attention_mask"])
    # same text but as row 0 of the padded batch:
cos = torch.nn.functional.cosine_similarity(tf[0:1], tf1[0:1]).item()
print("cos(padded-batch row0 vs unpadded single) for 'a dog barking':", cos)
print("DONE")
