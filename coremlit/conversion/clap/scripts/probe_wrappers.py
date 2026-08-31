"""Validate the pre-norm tower wrappers + traceability BEFORE writing converters.
- AudioTower(input_features)->[1,512] pre-norm; TextTower(ids,mask)->[1,512] pre-norm
- normalize(wrapper) must match model.get_*_features (post-norm) to cosine ~1
- confirm is_longer=None works (fusion off) and matches is_longer=False tensor
- confirm torch.jit.trace succeeds on both."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _clap_common import load_model, load_processor, mel_features

model = load_model()
proc = load_processor()


class AudioTower(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.audio_model = m.audio_model
        self.audio_projection = m.audio_projection

    def forward(self, input_features):
        pooled = self.audio_model(input_features=input_features, is_longer=None).pooler_output
        return self.audio_projection(pooled)  # [B, 512] pre-norm


class TextTower(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.text_model = m.text_model
        self.text_projection = m.text_projection

    def forward(self, input_ids, attention_mask):
        pooled = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.text_projection(pooled)  # [B, 512] pre-norm


atower = AudioTower(model).eval()
ttower = TextTower(model).eval()

rng = np.random.RandomState(0)
audio = rng.randn(480000).astype(np.float32) * 0.1
feats = mel_features(proc, audio)
inf = feats["input_features"]
is_longer = feats.get("is_longer")

with torch.no_grad():
    a_pre = atower(inf)
    # reference post-norm from the model API
    a_ref = model.get_audio_features(input_features=inf, is_longer=is_longer)
    if not torch.is_tensor(a_ref):
        a_ref = a_ref.pooler_output
    a_norm = F.normalize(a_pre, dim=-1)
    a_cos = F.cosine_similarity(a_norm, a_ref, dim=-1).min().item()
    # is_longer None vs explicit False tensor
    a_pre_false = atower.audio_model(input_features=inf,
                                     is_longer=torch.zeros(1, 1, dtype=torch.bool)).pooler_output
    a_false = atower.audio_projection(a_pre_false)
    a_ilcos = F.cosine_similarity(a_pre, a_false, dim=-1).min().item()
print(f"AUDIO pre-norm shape {tuple(a_pre.shape)}  norm={float(a_pre.norm(dim=-1)[0]):.4f}")
print(f"AUDIO normalize(wrapper) vs get_audio_features cosine (min) = {a_cos:.8f}")
print(f"AUDIO is_longer None vs False-tensor cosine (min) = {a_ilcos:.8f}")

tok = proc.tokenizer(["a dog barking", "gentle rain on a tin roof"],
                     padding="max_length", max_length=64, truncation=True, return_tensors="pt")
ids, mask = tok["input_ids"], tok["attention_mask"]
with torch.no_grad():
    t_pre = ttower(ids, mask)
    t_ref = model.get_text_features(input_ids=ids, attention_mask=mask)
    if not torch.is_tensor(t_ref):
        t_ref = t_ref.pooler_output
    t_norm = F.normalize(t_pre, dim=-1)
    t_cos = F.cosine_similarity(t_norm, t_ref, dim=-1).min().item()
print(f"TEXT pre-norm shape {tuple(t_pre.shape)}  norm={float(t_pre.norm(dim=-1)[0]):.4f}")
print(f"TEXT normalize(wrapper) vs get_text_features cosine (min) = {t_cos:.8f}")

# does text depend on attention_mask? zero-out mask beyond real tokens vs all-ones
with torch.no_grad():
    t_allones = ttower(ids, torch.ones_like(mask))
    t_mask_cos = F.cosine_similarity(F.normalize(t_pre, dim=-1),
                                     F.normalize(t_allones, dim=-1), dim=-1).min().item()
print(f"TEXT real-mask vs all-ones-mask cosine (min) = {t_mask_cos:.6f} "
      f"(low => mask matters; must feed real mask)")

# Traceability
print("\n=== trace ===")
ats = torch.jit.trace(atower, (inf,), check_trace=False)
print("audio trace OK")
tts = torch.jit.trace(ttower, (ids, mask), check_trace=False)
print("text trace OK")
print("DONE")
