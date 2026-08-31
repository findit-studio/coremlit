"""Generate the committed siglip goldens + staged NaFlex fixtures.

Oracle model : Siglip2Model.from_pretrained(REV) fp32 eval (the SHA-verified local
               snapshot). Image tensors come from the SLOW Siglip2ImageProcessor at
               the 512 budget; text ids from the FAST GemmaTokenizerFast (lowercased
               per the module contract). Embeddings are L2-normalized (f64 accumulate
               -> f32) before commit.

Writes (all paths from env):
  $SIGLIP_GOLDENS/corpus.json        committed image+text goldens (embeddings, ids)
  $SIGLIP_GOLDENS/preprocess.json    committed small preprocessing oracles (hermetic)
  $SIGLIP_MODELS_OUT/.../fixtures/preprocess/<id>.{pixel_values,attention_mask,
      position_embeddings,spatial_shapes}.npy   staged full-tensor NaFlex fixtures
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from _siglip_common import (
    EMBED_DIM,
    PATCH_BUDGET,
    PATCH_DIM,
    TEXT_WINDOW,
    load_fast_tokenizer,
    load_model,
    load_slow_image_processor,
    official_lift,
    padded_ids,
    src_dir,
)
from _fixtures import IMAGES, TEXTS, goldens_dir, load_pil

EOS_ID = 1
PAD_ID = 0
PATCH_SIZE = 16


def rust_fit_to_patch_budget(image_height, image_width, patch=PATCH_SIZE, budget=PATCH_BUDGET):
    """A line-for-line Python mirror of the Rust `fit_to_patch_budget` binary
    search, so the generator can CROSS-CHECK the slow processor's spatial_shapes
    against the coremlit port (not just against transformers)."""
    eps = 1e-5
    patch_f = float(patch)

    def scaled(scale, size):
        s = (scale * size) / patch_f
        s = math.ceil(s) * patch_f
        s = max(s, patch_f)
        return int(s)

    def grid(t):
        return t // patch

    smin, smax = eps / 10.0, 100.0
    while (smax - smin) >= eps:
        scale = (smin + smax) / 2.0
        if grid(scaled(scale, image_height)) * grid(scaled(scale, image_width)) <= budget:
            smin = scale
        else:
            smax = scale
    return grid(scaled(smin, image_height)), grid(scaled(smin, image_width))


def unit_norm(vec):
    """L2-normalize a [768] tensor: f64 accumulate, then f32 — exactly the Rust
    `Embedding::from_slice_normalizing` contract. Returns a python float list whose
    reprs round-trip the exact f32 (f32->f64 is exact)."""
    e = np.asarray(vec, np.float64).ravel()
    n = math.sqrt(float(e @ e))
    assert n > 0.0, "zero-magnitude embedding"
    unit = (e / n).astype(np.float32)
    return [float(x) for x in unit]


def resize_bilinear_antialias_rustalg(src, out_h, out_w):
    """A numpy mirror of the Rust f64 antialiased-bilinear `resize_bilinear_antialias`
    (single channel) — used ONLY to MEASURE the torch-vs-Rust-algorithm delta so the
    committed pos_lift tolerance is measured-then-pinned (not guessed). The committed
    oracle itself is torch's F.interpolate output."""
    src = np.asarray(src, np.float64)
    src_h, src_w = src.shape

    def coeffs(in_size, out_size):
        scale = in_size / out_size
        fscale = scale if scale >= 1.0 else 1.0
        support = fscale
        inv = 1.0 / fscale
        out = []
        for o in range(out_size):
            center = (o + 0.5) * scale
            xmin = max(int(math.floor(center - support + 0.5)), 0)
            xmax = min(int(math.floor(center + support + 0.5)), in_size)
            if xmax <= xmin:
                xmin = min(max(xmin, 0), in_size - 1)
                xmax = xmin + 1
            ws = []
            s = 0.0
            for k in range(xmax - xmin):
                x = xmin + k
                t = abs((x - center + 0.5) * inv)
                w = 1.0 - t if t < 1.0 else 0.0
                ws.append(w)
                s += w
            if s != 0.0:
                ws = [w / s for w in ws]
            out.append((xmin, ws))
        return out

    wc = coeffs(src_w, out_w)
    tmp = np.zeros((src_h, out_w), np.float64)
    for y in range(src_h):
        for ox, (start, ws) in enumerate(wc):
            tmp[y, ox] = sum(w * src[y, start + k] for k, w in enumerate(ws))
    hc = coeffs(src_h, out_h)
    out = np.zeros((out_h, out_w), np.float64)
    for oy, (start, ws) in enumerate(hc):
        for x in range(out_w):
            out[oy, x] = sum(w * tmp[start + k, x] for k, w in enumerate(ws))
    return out


def torch_resize_2d(grid2d, out_h, out_w):
    """The OFFICIAL resize kernel (per channel): F.interpolate(bilinear, antialias,
    align_corners=False) — exactly what resize_positional_embeddings runs."""
    t = torch.tensor(np.asarray(grid2d, np.float32))[None, None]
    r = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=True)
    return r[0, 0].numpy()


def build_corpus(model, proc, tok):
    images_out = []
    texts_by_id = {}

    # ---- texts ----
    for entry in TEXTS:
        ids, lower = padded_ids(tok, entry["text"], TEXT_WINDOW)  # asserts lowercase eq
        n_real = sum(1 for i in ids if i != PAD_ID)
        with torch.no_grad():
            emb = model.get_text_features(input_ids=torch.tensor([ids], dtype=torch.long)).numpy()
        texts_by_id[entry["id"]] = {
            "id": entry["id"],
            "text": entry["text"],
            "token_ids_padded": [int(i) for i in ids],
            "n_real": int(n_real),
            "embedding": unit_norm(emb),
        }

    # lowercase non-vacuity pair: mixedcase twin's window == its lowercase twin's.
    assert texts_by_id["mixedcase_cat"]["token_ids_padded"] == texts_by_id["cap_cat"]["token_ids_padded"], \
        "MixedCase twin window must equal the lowercase caption window"
    # sticky-EOS truncation proof: the long entry fills the window, ends in EOS, no pad.
    lt = texts_by_id["long_truncated"]
    assert lt["n_real"] == TEXT_WINDOW, f"long entry must fill the window (n_real={lt['n_real']})"
    assert lt["token_ids_padded"][TEXT_WINDOW - 1] == EOS_ID, "sticky-EOS: id[63] must be <eos>"
    assert PAD_ID not in lt["token_ids_padded"], "truncated long entry must have no pad"

    # ---- images ----
    for entry in IMAGES:
        img = load_pil(entry["id"])
        w, h = img.size
        feats = proc(images=[img], max_num_patches=PATCH_BUDGET, return_tensors="pt")
        pv = feats["pixel_values"].to(torch.float32)          # [1, 512, 768]
        mask = feats["pixel_attention_mask"]                  # [1, 512] (int)
        ss = feats["spatial_shapes"]                          # [1, 2]
        hp, wp = int(ss[0, 0]), int(ss[0, 1])
        # cross-check the slow processor's grid against the coremlit Rust solver.
        assert (hp, wp) == rust_fit_to_patch_budget(h, w), \
            f"{entry['id']}: processor grid ({hp},{wp}) != Rust solver {rust_fit_to_patch_budget(h, w)}"
        with torch.no_grad():
            emb = model.get_image_features(
                pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=ss
            ).numpy()
        images_out.append({
            "id": entry["id"],
            "file": f"images/{entry['id']}.png",
            "source": entry["source"],
            "license": entry["license"],
            "width": int(w),
            "height": int(h),
            "spatial_shapes": [hp, wp],
            "caption_id": entry["caption_id"],
            "embedding": unit_norm(emb),
        })
        assert entry["caption_id"] in texts_by_id, entry["caption_id"]

    # the committed 320x240 -> (19,26) cross-link to the Rust budget-solver oracle.
    cat = next(i for i in images_out if i["id"] == "cat")
    assert (cat["width"], cat["height"]) == (320, 240) and cat["spatial_shapes"] == [19, 26], cat

    texts_out = [texts_by_id[e["id"]] for e in TEXTS]
    return {"images": images_out, "texts": texts_out}


def build_preprocess_oracle():
    """Small hermetic oracles (no model needed by the Rust-side test): budget table,
    uint8 PIL resize grids, normalize samples, position-lift on small grids, and a
    tiny patchify tensor. Each is re-derivable by an independent in-test computation,
    so the committed torch/PIL reference is cross-checked, not merely stored."""
    from PIL import Image

    # (1) budget table: transformers/coremlit-solver grid for a spread of (H,W).
    budget_rows = [
        (240, 320), (480, 640), (120, 160),          # 1.333 landscape (incl. 320x240 twin)
        (640, 425), (640, 480), (640, 586),           # probe portrait aspects
        (483, 640), (427, 640), (426, 640),           # probe landscape aspects
        (512, 512), (64, 64), (1000, 1000),           # square
        (16, 16000), (16000, 16),                     # extreme (1-patch clamp)
        (427, 640), (608, 608), (640, 427), (640, 440), (640, 487),  # corpus dims
    ]
    budget_table = []
    seen = set()
    for (h, w) in budget_rows:
        if (h, w) in seen:
            continue
        seen.add((h, w))
        gh, gw = rust_fit_to_patch_budget(h, w)
        budget_table.append({"height": h, "width": w, "grid": [gh, gw]})

    # (2) uint8 PIL BILINEAR resize (the slow processor's image-resize semantics —
    # resize on u8, per-pass rounding). Single-channel oracles hand-verified against
    # the committed in-lib E3 grids; generated here via pillow so a pillow drift is
    # caught, then asserted equal by the Rust test to the known grids.
    def pil_resize_mono(src, sh, sw, dh, dw):
        im = Image.frombytes("L", (sw, sh), bytes(src))
        r = im.resize((dw, dh), Image.BILINEAR)
        return list(r.tobytes())

    resize_u8 = [
        {  # checker upscale 2x2 -> 4x4
            "src_h": 2, "src_w": 2, "dst_h": 4, "dst_w": 4,
            "src": [0, 255, 255, 0],
            "dst": pil_resize_mono([0, 255, 255, 0], 2, 2, 4, 4),
        },
        {  # per-pass rounding discriminant 2x2 -> 4x4
            "src_h": 2, "src_w": 2, "dst_h": 4, "dst_w": 4,
            "src": [0, 1, 255, 255],
            "dst": pil_resize_mono([0, 1, 255, 255], 2, 2, 4, 4),
        },
        {  # a genuine downscale 4x4 -> 2x2 (antialias low-pass)
            "src_h": 4, "src_w": 4, "dst_h": 2, "dst_w": 2,
            "src": [0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255],
            "dst": pil_resize_mono([0, 0, 255, 255] * 4, 4, 4, 2, 2),
        },
    ]

    # (3) normalize samples: ((v/255)-0.5)/0.5, exact f32.
    def norm_u8(v):
        return float((np.float32(float(v) * (1.0 / 255.0)) - np.float32(0.5)) / np.float32(0.5))

    normalize = {
        "rescale_factor": 0.00392156862745098,
        "mean": 0.5,
        "std": 0.5,
        "samples": [{"u8": v, "f32": norm_u8(v)} for v in (0, 1, 64, 127, 128, 200, 254, 255)],
    }

    # (4) position lift on SMALL single-channel grids (the resize-semantics pin): the
    # OFFICIAL torch F.interpolate(bilinear, antialias, align_corners=False) — one
    # upscale (the lift's real 16->grid regime) and one downscale (antialias active).
    # Stored input + output so the Rust test can recompute with its own f64 kernel;
    # the tolerance is MEASURED against a numpy mirror of that kernel and pinned.
    pos_cases = []
    max_delta = 0.0
    inputs = [
        ("upscale", [[float((y * 4 + x) % 7) for x in range(4)] for y in range(4)], 7, 9),
        ("downscale", [[float((y * 8 + x) * 0.5) for x in range(8)] for y in range(8)], 3, 2),
        ("near16", [[float((y * 16 + x) % 13) - 6.0 for x in range(16)] for y in range(16)], 19, 26),
    ]
    for name, grid2d, oh, ow in inputs:
        ref = torch_resize_2d(grid2d, oh, ow)
        rust = resize_bilinear_antialias_rustalg(grid2d, oh, ow)
        max_delta = max(max_delta, float(np.max(np.abs(ref - rust))))
        pos_cases.append({
            "name": name,
            "src_h": len(grid2d), "src_w": len(grid2d[0]), "dst_h": oh, "dst_w": ow,
            "input": [float(v) for row in grid2d for v in row],
            "values": [float(v) for v in ref.ravel()],  # torch reference, row-major
        })
    # measured-then-pinned: a safety factor above the observed torch-vs-Rust-alg delta.
    pos_tol = max(1e-6, round(max_delta * 4.0 + 1e-7, 9))
    print(f"  [pos_lift] measured torch-vs-Rustalg max delta = {max_delta:.3e}, pinned tol = {pos_tol:.3e}")

    # (5) patchify: a tiny exact tensor pinning the (patch_row, patch_col, py, px, ch)
    # flatten order. grid 2x2 -> image 32x32x3; value = y*10000 + x*10 + c.
    gh, gw = 2, 2
    ih, iw = gh * PATCH_SIZE, gw * PATCH_SIZE
    pixels = np.zeros((ih, iw, 3), np.int64)
    for y in range(ih):
        for x in range(iw):
            for c in range(3):
                pixels[y, x, c] = y * 10000 + x * 10 + c
    patches = np.zeros((gh * gw, PATCH_DIM), np.int64)
    for ph in range(gh):
        for pw in range(gw):
            k = 0
            for py in range(PATCH_SIZE):
                for px in range(PATCH_SIZE):
                    for c in range(3):
                        patches[ph * gw + pw, k] = pixels[ph * PATCH_SIZE + py, pw * PATCH_SIZE + px, c]
                        k += 1
    patchify = {
        "grid_h": gh, "grid_w": gw,
        "pixels": [int(v) for v in pixels.ravel()],   # [ih*iw*3] row-major HWC
        "patches": [int(v) for v in patches.ravel()],  # [gh*gw*768] row-major
    }

    return {
        "budget_table": budget_table,
        "resize_u8": resize_u8,
        "normalize": normalize,
        "pos_lift": {"tolerance": pos_tol, "cases": pos_cases},
        "patchify": patchify,
    }


def write_npy_fixtures(model, proc):
    out_root = os.environ.get("SIGLIP_MODELS_OUT")
    if not out_root:
        raise SystemExit("SIGLIP_MODELS_OUT unset (needed to stage .npy fixtures)")
    fdir = os.path.join(out_root, "fixtures", "preprocess")
    os.makedirs(fdir, exist_ok=True)
    for entry in IMAGES:
        img = load_pil(entry["id"])
        feats = proc(images=[img], max_num_patches=PATCH_BUDGET, return_tensors="pt")
        pv = feats["pixel_values"].to(torch.float32)[0].numpy()     # [512, 768]
        mask = feats["pixel_attention_mask"][0].to(torch.float32).numpy()  # [512]
        ss = feats["spatial_shapes"]                                # [1, 2]
        pos = official_lift(model, ss, max_length=PATCH_BUDGET)[0].to(torch.float32).numpy()  # [512, 768]
        iid = entry["id"]
        np.save(os.path.join(fdir, f"{iid}.pixel_values.npy"), pv.astype(np.float32))
        np.save(os.path.join(fdir, f"{iid}.attention_mask.npy"), mask.astype(np.float32))
        np.save(os.path.join(fdir, f"{iid}.position_embeddings.npy"), pos.astype(np.float32))
        np.save(os.path.join(fdir, f"{iid}.spatial_shapes.npy"),
                np.array([int(ss[0, 0]), int(ss[0, 1])], dtype=np.int64))
    print(f"  staged .npy fixtures -> {fdir}")


def advisory_slow_vs_fast_tokenizer(fast_tok):
    """Non-blocking advisory (§7): the SLOW sentencepiece GemmaTokenizer should
    produce the same ids as the FAST tokenizer.json on the ASCII corpus. Printed,
    never asserted — the Rust path only ever executes tokenizer.json (== the fast
    tokenizer), pinned by SHA + the byte-parity identity gate; this is a defensive
    cross-check against a fast-vs-slow HF skew, not a gate."""
    try:
        from transformers import GemmaTokenizer

        slow = GemmaTokenizer.from_pretrained(src_dir())
    except Exception as e:  # noqa: BLE001 — advisory only
        print(f"  [advisory] slow tokenizer unavailable ({type(e).__name__}); skipped")
        return
    import tokenizers

    mismatches = 0
    for entry in TEXTS:
        lower = tokenizers.normalizers.Lowercase().normalize_str(entry["text"])
        fast_ids = fast_tok(lower, add_special_tokens=True)["input_ids"]
        slow_ids = slow(lower, add_special_tokens=True)["input_ids"]
        if fast_ids != slow_ids:
            mismatches += 1
            print(f"  [advisory] slow!=fast ids for {entry['id']!r} (non-blocking)")
    print(f"  [advisory] slow-vs-fast tokenizer ids: {len(TEXTS) - mismatches}/{len(TEXTS)} agree")


def main():
    model = load_model()
    proc = load_slow_image_processor()
    tok = load_fast_tokenizer()
    advisory_slow_vs_fast_tokenizer(tok)

    gdir = goldens_dir()
    os.makedirs(gdir, exist_ok=True)

    print("=== corpus.json ===")
    corpus = build_corpus(model, proc, tok)
    with open(os.path.join(gdir, "corpus.json"), "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"  {len(corpus['images'])} images, {len(corpus['texts'])} texts -> corpus.json")

    print("=== preprocess.json ===")
    pp = build_preprocess_oracle()
    with open(os.path.join(gdir, "preprocess.json"), "w") as f:
        json.dump(pp, f, indent=2, ensure_ascii=False)
    print(f"  budget_table {len(pp['budget_table'])} rows, resize_u8 {len(pp['resize_u8'])}, "
          f"pos_lift {len(pp['pos_lift']['cases'])} -> preprocess.json")

    print("=== staged .npy fixtures ===")
    write_npy_fixtures(model, proc)

    print("DONE goldens")


if __name__ == "__main__":
    main()
