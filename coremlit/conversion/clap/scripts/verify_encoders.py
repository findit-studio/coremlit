"""Conversion-verification chain (measure-then-ASSERT), per encoder:
  (a) PyTorch fp32 (UN-patched source) vs CoreML fp32 on CPU  -> worst cosine,
      asserted >= a pinned near-1 floor. Ground truth is the UN-patched
      ClapModel (true CLAP bicubic, real new_ones), so this floor is the
      ARTIFACT-level proof that the baked bicubic-resize matmul (audio) and the
      `new_ones` lowering (text) are faithful in the CONVERTED graph — not merely
      in the pre-conversion PyTorch wrappers `convert_*.py` already check.
  (b) CoreML fp16 vs CoreML fp32, per compute unit (All/CpuAndNeuralEngine/
      CpuAndGpu/CpuOnly) -> worst cosine, asserted >= a pinned floor on EVERY
      unit (the audio graph falls back off the ANE; the floor still holds).

FAIL-CLOSED (the point of this rewrite):
  * a non-finite cosine POISONS the worst to NaN, so a NaN artifact fails the
    floor (`NaN >= floor` is False) instead of `min(1.0, NaN)` printing perfect
    agreement;
  * an artifact load / prediction exception is a recorded HARD failure, not a
    caught-and-ignored `ERROR` line;
  * any floor breach or exception exits NONZERO, so `set -e` in run_clapkit.sh
    halts the pipeline before the downstream steps consume a bad artifact.

Inputs are the HF feature-extractor mel (audio) and tokenizer ids/mask padded to
512 (text) — the SAME inputs feed torch and CoreML, so the mel frontend is
excluded and this isolates model-graph parity. Cosine is normalization-invariant,
so comparing the pre-norm graph outputs is exactly the post-norm embedding
agreement clapkit sees.
"""
import os
import sys
import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from _clap_common import load_model, load_processor
from _fixtures import audio_clips, input_features, TEXT_PROMPTS

STAGE = "/private/tmp/claude-501/-Users-al-Developer-findit-studio-coremlit/2e543e17-c5e2-4187-be75-b6b4fafe4418/scratchpad/conv/clapkit/staging"
SEQ = 512

# Every public compute unit, including the ANE-naming one the previous matrix
# omitted (the audio graph falls back off the ANE, so its floor still holds and
# is now measured rather than assumed).
UNITS = {
    "All": ct.ComputeUnit.ALL,
    "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
    "CpuOnly": ct.ComputeUnit.CPU_ONLY,
}

# Pinned near-1 floors (measure-then-pin against the shipped T1 artifacts; set
# below the measured worst with margin). The fp32-vs-torch floors are the
# artifact-level conversion + shim faithfulness gate; the fp16-vs-fp32 floors are
# the per-placement gate. A breach is a finding, not a threshold to loosen.
AUDIO_FP32_FLOOR = 0.9999
TEXT_FP32_FLOOR = 0.9999
AUDIO_FP16_FLOOR = 0.999
TEXT_FP16_FLOOR = 0.999


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    # NO +eps guard: a non-finite artifact MUST propagate to NaN here, not be
    # masked into a finite-looking value.
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def worst_update(worst, c):
    """NaN-propagating min: a NaN cosine poisons `worst` to NaN so the floor
    assertion (`worst >= floor`) becomes False. Plain `min(1.0, NaN)` returns
    1.0 on many platforms, which is exactly the bug this rewrite kills."""
    if worst != worst or c != c:
        return float("nan")
    return min(worst, c)


def check(failures, label, worst, floor):
    """Record a failure unless `worst >= floor` (NaN-safe: a NaN worst is < any
    floor by this comparison, so a poisoned worst is a failure)."""
    ok = bool(worst >= floor)
    status = "OK" if ok else "FAIL"
    print(f"    [{status}] {label}: worst {worst:.8f}  (floor {floor})")
    if not ok:
        failures.append(f"{label}: worst {worst:.8f} < floor {floor} (or non-finite)")
    return ok


def main():
    model = load_model()  # UN-patched ground truth (true bicubic, real new_ones)
    proc = load_processor()
    failures = []

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

    # ---- (a) CoreML fp32 CPU vs PyTorch — artifact-level conversion + shim proof ----
    print("\n=== (a) PyTorch fp32 vs CoreML fp32 (CPU) — conversion + bicubic/new_ones shim floor ===")
    try:
        a_fp32 = ct.models.MLModel(os.path.join(STAGE, "clap_audio_fp32.mlpackage"),
                                   compute_units=ct.ComputeUnit.CPU_ONLY)
        worst_a = 1.0
        for name, inf, ref in audio_inputs:
            out = a_fp32.predict({"input_features": inf})["audio_embeds"]
            worst_a = worst_update(worst_a, cos(out, ref))
        print(f"  AUDIO worst cosine (fp32 CPU vs torch) = {worst_a:.8f}")
        check(failures, "AUDIO fp32-vs-torch (bicubic-resize shim)", worst_a, AUDIO_FP32_FLOOR)
    except Exception as e:
        print(f"  AUDIO fp32 load/predict ERROR {type(e).__name__}: {e}")
        failures.append(f"AUDIO fp32 artifact load/predict failed: {e}")

    try:
        t_fp32 = ct.models.MLModel(os.path.join(STAGE, "clap_text_fp32.mlpackage"),
                                   compute_units=ct.ComputeUnit.CPU_ONLY)
        worst_t = 1.0
        for p, ids, mask, ref in text_inputs:
            out = t_fp32.predict({"input_ids": ids, "attention_mask": mask})["text_embeds"]
            worst_t = worst_update(worst_t, cos(out, ref))
        print(f"  TEXT  worst cosine (fp32 CPU vs torch) = {worst_t:.8f}")
        check(failures, "TEXT fp32-vs-torch (new_ones shim)", worst_t, TEXT_FP32_FLOOR)
    except Exception as e:
        print(f"  TEXT fp32 load/predict ERROR {type(e).__name__}: {e}")
        failures.append(f"TEXT fp32 artifact load/predict failed: {e}")

    # ---- (b) CoreML fp16 vs CoreML fp32, per compute unit ----
    print("\n=== (b) CoreML fp16 vs CoreML fp32 per compute unit ===")
    # fp32 CPU baseline outputs (per input) reused as the reference. If the fp32
    # artifacts failed to load above, these references cannot be built and the
    # already-recorded failures will exit nonzero.
    a_ref = [a_fp32.predict({"input_features": inf})["audio_embeds"].ravel() for _, inf, _ in audio_inputs]
    t_ref = [t_fp32.predict({"input_ids": ids, "attention_mask": mask})["text_embeds"].ravel()
             for _, ids, mask, _ in text_inputs]

    for enc, path, feed_fn, refs, items, out_key, floor in [
        ("AUDIO", "clap_audio.mlmodelc", lambda x: {"input_features": x[1]}, a_ref, audio_inputs, "audio_embeds", AUDIO_FP16_FLOOR),
        ("TEXT", "clap_text.mlmodelc", lambda x: {"input_ids": x[1], "attention_mask": x[2]}, t_ref, text_inputs, "text_embeds", TEXT_FP16_FLOOR),
    ]:
        for uname, cu in UNITS.items():
            try:
                m = ct.models.CompiledMLModel(os.path.join(STAGE, path), cu)
                worst = 1.0
                for item, ref in zip(items, refs):
                    out = np.asarray(m.predict(feed_fn(item))[out_key]).ravel()
                    worst = worst_update(worst, cos(out, ref))
            except Exception as e:
                # A load/prediction failure is a HARD failure (nonzero exit), not
                # a caught-and-ignored ERROR line that lets the pipeline pass.
                print(f"  {enc:5s} fp16 [{uname:18s}] ERROR {type(e).__name__}: {str(e)[:160]}")
                failures.append(f"{enc} fp16 [{uname}] load/predict failed: {type(e).__name__}: {str(e)[:160]}")
                continue
            print(f"  {enc:5s} fp16 [{uname:18s}] worst cosine vs fp32 = {worst:.8f}")
            check(failures, f"{enc} fp16 [{uname}]", worst, floor)

    if failures:
        print("\nVERIFY FAILED — the shipped artifacts did not clear their pinned floors:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nDONE verify — every fp32-vs-torch and fp16-vs-fp32 floor held")


if __name__ == "__main__":
    main()
