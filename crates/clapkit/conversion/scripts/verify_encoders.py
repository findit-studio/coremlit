"""Conversion-verification chain (measure-then-record), per encoder:
  (a) PyTorch fp32 (un-patched source) vs CoreML fp32 on CPU  -> worst cosine
  (b) CoreML fp16 vs CoreML fp32, per compute unit (All/CpuAndGpu/CpuOnly)

Ground truth is the UN-patched ClapModel (true CLAP bicubic). Inputs are the HF
feature-extractor mel (audio) and tokenizer ids/mask padded to 512 (text) — the
SAME inputs feed torch and CoreML, so the mel frontend is excluded and this
isolates model-graph parity. Cosine is normalization-invariant, so comparing the
pre-norm graph outputs is exactly the post-norm embedding agreement clapkit sees.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from _clap_common import load_model, load_processor
from _fixtures import audio_clips, input_features, TEXT_PROMPTS

STAGE = "/private/tmp/claude-501/-Users-al-Developer-findit-studio-coremlit/2e543e17-c5e2-4187-be75-b6b4fafe4418/scratchpad/conv/clapkit/staging"
SEQ = 512
UNITS = {"All": ct.ComputeUnit.ALL, "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU, "CpuOnly": ct.ComputeUnit.CPU_ONLY}


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    model = load_model()  # UN-patched ground truth
    proc = load_processor()

    # ---- build torch fp32 references (pre-norm) ----
    class AT(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.am = model.audio_model; s.ap = model.audio_projection
        def forward(s, x):
            return s.ap(s.am(input_features=x, is_longer=None).pooler_output)

    class TT(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.tm = model.text_model; s.tp = model.text_projection
        def forward(s, i, m):
            return s.tp(s.tm(input_ids=i, attention_mask=m).pooler_output)

    at, tt = AT().eval(), TT().eval()

    print("=== building fixtures ===")
    audio_inputs = []  # (name, input_features np [1,1,1001,64], torch_ref [512])
    for name, samples in audio_clips():
        inf, _ = input_features(proc, samples)
        with torch.no_grad():
            ref = at(inf)[0].numpy()
        audio_inputs.append((name, inf.numpy().astype(np.float32), ref))
    print(f"  audio fixtures: {len(audio_inputs)}")

    text_inputs = []
    for p in TEXT_PROMPTS:
        tok = proc.tokenizer([p], padding="max_length", truncation=True, max_length=SEQ, return_tensors="pt")
        ids, mask = tok["input_ids"], tok["attention_mask"]
        with torch.no_grad():
            ref = tt(ids, mask)[0].numpy()
        text_inputs.append((p, ids.numpy().astype(np.int32), mask.numpy().astype(np.int32), ref))
    print(f"  text fixtures: {len(text_inputs)}")

    # ---- (a) CoreML fp32 CPU vs PyTorch ----
    print("\n=== (a) PyTorch fp32 vs CoreML fp32 (CPU) ===")
    a_fp32 = ct.models.MLModel(os.path.join(STAGE, "clap_audio_fp32.mlpackage"),
                               compute_units=ct.ComputeUnit.CPU_ONLY)
    worst_a = 1.0
    for name, inf, ref in audio_inputs:
        out = a_fp32.predict({"input_features": inf})["audio_embeds"]
        c = cos(out, ref); worst_a = min(worst_a, c)
    print(f"AUDIO worst cosine (fp32 CPU vs torch) = {worst_a:.8f}")

    t_fp32 = ct.models.MLModel(os.path.join(STAGE, "clap_text_fp32.mlpackage"),
                               compute_units=ct.ComputeUnit.CPU_ONLY)
    worst_t = 1.0
    for p, ids, mask, ref in text_inputs:
        out = t_fp32.predict({"input_ids": ids, "attention_mask": mask})["text_embeds"]
        c = cos(out, ref); worst_t = min(worst_t, c)
    print(f"TEXT  worst cosine (fp32 CPU vs torch) = {worst_t:.8f}")

    # ---- (b) CoreML fp16 vs CoreML fp32, per compute unit ----
    print("\n=== (b) CoreML fp16 vs CoreML fp32 per compute unit ===")
    # fp32 CPU baseline outputs (per input) reused as the reference.
    a_ref = [a_fp32.predict({"input_features": inf})["audio_embeds"].ravel() for _, inf, _ in audio_inputs]
    t_ref = [t_fp32.predict({"input_ids": ids, "attention_mask": mask})["text_embeds"].ravel()
             for _, ids, mask, _ in text_inputs]

    for enc, path, feed_fn, refs, items in [
        ("AUDIO", "clap_audio.mlmodelc", lambda x: {"input_features": x[1]}, a_ref, audio_inputs),
        ("TEXT", "clap_text.mlmodelc", lambda x: {"input_ids": x[1], "attention_mask": x[2]}, t_ref, text_inputs),
    ]:
        for uname, cu in UNITS.items():
            try:
                m = ct.models.CompiledMLModel(os.path.join(STAGE, path), cu)
                worst = 1.0
                out_key = "audio_embeds" if enc == "AUDIO" else "text_embeds"
                for item, ref in zip(items, refs):
                    out = np.asarray(m.predict(feed_fn(item))[out_key]).ravel()
                    worst = min(worst, cos(out, ref))
                print(f"{enc:5s} fp16 [{uname:9s}] worst cosine vs fp32 = {worst:.8f}")
            except Exception as e:
                print(f"{enc:5s} fp16 [{uname:9s}] ERROR {type(e).__name__}: {str(e)[:120]}")
    print("DONE verify")


if __name__ == "__main__":
    main()
