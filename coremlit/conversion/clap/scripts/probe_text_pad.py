"""Decide the text graph's sequence contract: does padding to a fixed length with
attention_mask reproduce the natural-length embedding? (validates static-shape).
Also report the pooler kind."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from _clap_common import load_model, load_processor

model = load_model()
proc = load_processor()
print("pooler:", model.text_model.pooler)

PAD = model.config.text_config.pad_token_id
print("pad_token_id:", PAD)


class TextTower(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.text_model = m.text_model
        self.text_projection = m.text_projection

    def forward(self, input_ids, attention_mask):
        pooled = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.text_projection(pooled)


tt = TextTower(model).eval()
prompts = ["a dog barking", "gentle rain on a tin roof", "a short one",
           "the sound of a violin playing a slow melody in a concert hall"]

worst = 1.0
for FIX in (64, 512):
    cmin = 1.0
    for p in prompts:
        nat = proc.tokenizer([p], padding=False, truncation=True, max_length=512, return_tensors="pt")
        pad = proc.tokenizer([p], padding="max_length", truncation=True, max_length=FIX, return_tensors="pt")
        with torch.no_grad():
            e_nat = F.normalize(tt(nat["input_ids"], nat["attention_mask"]), dim=-1)
            e_pad = F.normalize(tt(pad["input_ids"], pad["attention_mask"]), dim=-1)
        c = F.cosine_similarity(e_nat, e_pad, dim=-1).item()
        cmin = min(cmin, c)
    print(f"FIX={FIX}: natural-vs-padded cosine min = {cmin:.8f}")
    worst = min(worst, cmin)

# int32 vs int64 ids equivalence (CoreML will feed int32)
p = "a dog barking"
tok = proc.tokenizer([p], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    e64 = tt(tok["input_ids"].long(), tok["attention_mask"].long())
    e32 = tt(tok["input_ids"].int().long(), tok["attention_mask"].int().long())
print("int32->int64 ids cosine:", F.cosine_similarity(F.normalize(e64,dim=-1), F.normalize(e32,dim=-1), dim=-1).item())
print("WORST natural-vs-padded:", worst)
print("DONE")
