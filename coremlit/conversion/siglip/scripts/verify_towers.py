"""Fail-closed conversion-verification matrix (§4.7), per tower.

  (a) PyTorch fp32 (UNMODIFIED get_image_features/get_text_features) vs CoreML fp32
      on CPU, over ALL fixtures -> worst cosine, asserted >= 0.9999. The
      artifact-level proof the lift-wrapper + conversion are faithful IN the graph.
  (b) CoreML fp16 vs CoreML fp32-CPU per compute unit {All, CpuAndNeuralEngine,
      CpuAndGpu, CpuOnly} -> worst cosine. CpuAndGpu is THE floor-gated arm
      (>= 0.99917); ANE/All/CpuOnly are RECORDED (ANE below floor by design, never
      gated). The All arm also records whether it is GPU-identical (D1 evidence).
  (c) CoreML fp16 CpuAndGpu vs the committed torch goldens -> the end-to-end number
      the Rust parity gate reproduces (informational; the Rust test pins the band).

FAIL-CLOSED: a non-finite cosine POISONS worst to NaN (fails the floor); any
load/predict exception is a HARD failure; any breach exits NONZERO so run_siglip.sh
(set -e) halts. A breach is a FINDING, never a floor to loosen.
"""
import json
import os
import sys

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, os.path.dirname(__file__))
from _siglip_common import (
    PATCH_BUDGET,
    TEXT_WINDOW,
    cos,
    load_fast_tokenizer,
    load_model,
    load_slow_image_processor,
    official_lift,
    padded_ids,
    stage_dir,
)
from _fixtures import IMAGES, TEXTS, load_pil

MODELS_OUT = os.environ.get("SIGLIP_MODELS_OUT")
MODEL_ROOT = os.path.join(MODELS_OUT, "siglip2-base-patch16-naflex-512") if MODELS_OUT else None
STAGE = stage_dir()

VISION_FP32_FLOOR = 0.9999
TEXT_FP32_FLOOR = 0.9999
GPU_FLOOR = 0.99917       # THE gate (CpuAndGpu), both towers
ANE_SANITY = 0.995        # recorded sanity only — the ANE is never floor-gated

UNITS = {
    "All": ct.ComputeUnit.ALL,
    "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
    "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
    "CpuOnly": ct.ComputeUnit.CPU_ONLY,
}


def worst_update(worst, c):
    if worst != worst or c != c:
        return float("nan")
    return min(worst, c)


def check(failures, label, worst, floor):
    ok = bool(worst >= floor)
    print(f"    [{'OK' if ok else 'FAIL'}] {label}: worst {worst:.8f}  (floor {floor})")
    if not ok:
        failures.append(f"{label}: worst {worst:.8f} < floor {floor} (or non-finite)")
    return ok


def main():
    if not MODEL_ROOT:
        raise SystemExit("SIGLIP_MODELS_OUT unset")
    model = load_model()  # UNMODIFIED ground truth
    proc = load_slow_image_processor()
    tok = load_fast_tokenizer()
    failures = []
    metrics = {}

    # ---- fixtures + torch refs ----
    print("=== building fixtures ===")
    vis = []  # (id, feed{dict np}, torch_ref[768])
    for e in IMAGES:
        img = load_pil(e["id"])
        feats = proc(images=[img], max_num_patches=PATCH_BUDGET, return_tensors="pt")
        pv = feats["pixel_values"].to(torch.float32)
        mask = feats["pixel_attention_mask"]
        ss = feats["spatial_shapes"]
        pos = official_lift(model, ss, PATCH_BUDGET).to(torch.float32)
        with torch.no_grad():
            ref = model.get_image_features(pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=ss).numpy()
        feed = {
            "pixel_values": pv.numpy().astype(np.float32),
            "position_embeddings": pos.numpy().astype(np.float32),
            "attention_mask": mask.to(torch.float32).numpy().astype(np.float32),
        }
        vis.append((e["id"], feed, ref.ravel()))
    txt = []
    for e in TEXTS:
        ids, _ = padded_ids(tok, e["text"], TEXT_WINDOW)
        arr = np.array([ids], dtype=np.int32)
        with torch.no_grad():
            ref = model.get_text_features(input_ids=torch.tensor(arr, dtype=torch.long)).numpy()
        txt.append((e["id"], {"input_ids": arr}, ref.ravel()))
    print(f"  vision {len(vis)}, text {len(txt)}")

    # ---- (a) CoreML fp32-CPU vs torch fp32 ----
    print("\n=== (a) PyTorch fp32 vs CoreML fp32 (CPU) — artifact faithfulness floor ===")
    v_fp32 = ct.models.MLModel(os.path.join(STAGE, "siglip2_vision_512_fp32.mlpackage"),
                               compute_units=ct.ComputeUnit.CPU_ONLY)
    t_fp32 = ct.models.MLModel(os.path.join(STAGE, "siglip2_text_64_fp32.mlpackage"),
                               compute_units=ct.ComputeUnit.CPU_ONLY)
    v_ref_fp32, t_ref_fp32 = [], []
    worst = 1.0
    for iid, feed, ref in vis:
        out = v_fp32.predict(feed)["image_features"].ravel()
        v_ref_fp32.append(out)
        worst = worst_update(worst, cos(out, ref))
    metrics["vision_fp32_vs_torch"] = worst
    check(failures, "VISION fp32-CPU vs torch", worst, VISION_FP32_FLOOR)
    worst = 1.0
    for tid, feed, ref in txt:
        out = t_fp32.predict(feed)["text_features"].ravel()
        t_ref_fp32.append(out)
        worst = worst_update(worst, cos(out, ref))
    metrics["text_fp32_vs_torch"] = worst
    check(failures, "TEXT fp32-CPU vs torch", worst, TEXT_FP32_FLOOR)

    # ---- (b) CoreML fp16 vs CoreML fp32-CPU, per unit ----
    print("\n=== (b) CoreML fp16 vs CoreML fp32-CPU per compute unit ===")
    per_unit_out = {"vision": {}, "text": {}}
    for tower, path, items, refs, out_key in [
        ("vision", "siglip2_vision_512.mlmodelc", vis, v_ref_fp32, "image_features"),
        ("text", "siglip2_text_64.mlmodelc", txt, t_ref_fp32, "text_features"),
    ]:
        for uname, cu in UNITS.items():
            try:
                m = ct.models.CompiledMLModel(os.path.join(MODEL_ROOT, path), cu)
                outs, worst = [], 1.0
                for (name, feed, _), r32 in zip(items, refs):
                    out = np.asarray(m.predict(feed)[out_key]).ravel()
                    outs.append(out)
                    worst = worst_update(worst, cos(out, r32))
            except Exception as e:  # noqa: BLE001 — a load/predict failure is HARD
                print(f"  {tower:6s} fp16 [{uname:18s}] ERROR {type(e).__name__}: {str(e)[:150]}")
                failures.append(f"{tower} fp16 [{uname}] load/predict failed: {type(e).__name__}")
                continue
            per_unit_out[tower][uname] = outs
            metrics[f"{tower}_fp16_{uname}_vs_fp32"] = worst
            print(f"  {tower:6s} fp16 [{uname:18s}] worst vs fp32 = {worst:.8f}")
            if uname == "CpuAndGpu":
                check(failures, f"{tower} fp16 [CpuAndGpu] GATE", worst, GPU_FLOOR)
            elif uname == "CpuAndNeuralEngine":
                ok = bool(worst >= ANE_SANITY)
                print(f"      [{'ok' if ok else 'LOW'}] ANE recorded (sanity {ANE_SANITY}, never gated)")
        # D1: is the All arm GPU-identical? (planner dispatch evidence)
        gpu = per_unit_out[tower].get("CpuAndGpu")
        alll = per_unit_out[tower].get("All")
        if gpu and alll:
            d = max(float(np.max(np.abs(a - g))) for a, g in zip(alll, gpu))
            metrics[f"{tower}_All_vs_CpuAndGpu_maxabs"] = d
            print(f"  {tower:6s} All-vs-CpuAndGpu max|delta| = {d:.3e}  "
                  f"({'GPU-identical' if d == 0.0 else 'differs -> ANE-influenced'})")

    # ---- (c) CoreML fp16 CpuAndGpu vs committed torch goldens (Rust-parity proxy) ----
    print("\n=== (c) CoreML fp16 CpuAndGpu vs torch goldens (the Rust parity gate reproduces this) ===")
    for tower, items, torch_refs in [
        ("vision", vis, [r for _, _, r in vis]),
        ("text", txt, [r for _, _, r in txt]),
    ]:
        outs = per_unit_out[tower].get("CpuAndGpu")
        if not outs:
            continue
        worst = 1.0
        for out, ref in zip(outs, torch_refs):
            worst = worst_update(worst, cos(out, ref))
        metrics[f"{tower}_fp16_CpuAndGpu_vs_torch"] = worst
        print(f"  {tower:6s} fp16 CpuAndGpu vs torch golden worst = {worst:.8f}")

    with open(os.path.join(STAGE, "verify_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  wrote verify_metrics.json")

    if failures:
        print("\nVERIFY FAILED — shipped artifacts did not clear their pinned floors:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nDONE verify — fp32-vs-torch and the CpuAndGpu fp16 gate held on both towers")


if __name__ == "__main__":
    main()
