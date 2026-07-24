"""Emit CHECKSUMS.sha256 + MANIFEST.json for the staged shipped bundle.

Walks the two shipped .mlmodelc bundles + the pos-emb sidecar under
$SIGLIP_MODELS_OUT/siglip2-base-patch16-naflex-512, writing forward-slash-relative
SHA-256 lines (the exact-set manifest the Rust gates pin from) and a MANIFEST.json
recording source repo+REV, per-source-file SHA-256, toolchain pins, the attention
lowering used, the measured verify numbers, and the checkpoint's logit_scale/bias
(recorded now for a future score(); rank() ships cosine-only)."""
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _siglip_common import MODEL_ID, REV, SOURCE_SHA256, src_dir, stage_dir

MODELS_OUT = os.environ.get("SIGLIP_MODELS_OUT")
if not MODELS_OUT:
    raise SystemExit("SIGLIP_MODELS_OUT unset")
MODEL_ROOT = os.path.join(MODELS_OUT, "siglip2-base-patch16-naflex-512")
STAGE = stage_dir()

SHIPPED = ["siglip2_vision_512.mlmodelc", "siglip2_text_64.mlmodelc"]
SIDECAR = "pos_embed_16x16x768.f32le.bin"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_files(root, sub):
    base = os.path.join(root, sub) if sub else root
    out = []
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            if name.startswith("._") or name == ".DS_Store":
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            out.append(rel)
    return sorted(out)


def read_scalar(name):
    """Read a 1-element tensor (logit_scale/logit_bias) from the safetensors."""
    from safetensors import safe_open

    with safe_open(os.path.join(src_dir(), "model.safetensors"), framework="np") as f:
        keys = [k for k in f.keys() if k.split(".")[-1] == name]
        if not keys:
            return None
        return float(f.get_tensor(keys[0]).ravel()[0])


def main():
    import math

    # 1. CHECKSUMS.sha256 over every shipped file (both bundles + sidecar).
    rels = []
    for sub in SHIPPED:
        rels += rel_files(MODEL_ROOT, sub)
    rels.append(SIDECAR)
    rels = sorted(rels)
    checks = {rel: sha256_file(os.path.join(MODEL_ROOT, rel)) for rel in rels}
    with open(os.path.join(MODEL_ROOT, "CHECKSUMS.sha256"), "w") as f:
        for rel in rels:
            f.write(f"{checks[rel]}  {rel}\n")
    print(f"[ok] CHECKSUMS.sha256: {len(rels)} files")

    # 2. MANIFEST.json.
    verify_path = os.path.join(STAGE, "verify_metrics.json")
    verify = json.load(open(verify_path)) if os.path.exists(verify_path) else {}
    attn = {}
    for tower in ("vision", "text"):
        p = os.path.join(STAGE, f"attn_impl_{tower}.txt")
        attn[tower] = open(p).read().strip() if os.path.exists(p) else "sdpa"
    logit_scale = read_scalar("logit_scale")
    logit_bias = read_scalar("logit_bias")

    manifest = {
        "source": {
            "repo": MODEL_ID,
            "revision": REV,
            "license": "Apache-2.0",
            "files_sha256": SOURCE_SHA256,
        },
        "toolchain": {
            "python": "3.11", "torch": "2.5.1", "transformers": "4.53.3",
            "coremltools": "9.0", "numpy": "1.26.4", "pillow": "12.3.0",
            "tokenizers": "0.21.2",
        },
        "attention_impl": attn,
        "conversion_date": datetime.date.today().isoformat(),
        "contract": {
            "vision": {
                "inputs": {
                    "pixel_values": "float32 [1, 512, 768]",
                    "position_embeddings": "float32 [1, 512, 768]",
                    "attention_mask": "float32 [1, 512]",
                },
                "output": {"image_features": "float32 [1, 768] (pre-L2-norm)"},
                "sidecar": f"{SIDECAR} (16x16x768 f32 LE, 786432 bytes)",
                "patch_budget": 512,
            },
            "text": {
                "inputs": {"input_ids": "int32 [1, 64]"},
                "output": {"text_features": "float32 [1, 768] (pre-L2-norm)"},
                "window": 64,
            },
            "note": "L2 normalization is applied by the caller (Rust), OUT of both graphs.",
        },
        "shipped_files_sha256": checks,
        "verify": verify,
        "scoring": {
            "logit_scale": logit_scale,
            "logit_scale_exp": math.exp(logit_scale) if logit_scale is not None else None,
            "logit_bias": logit_bias,
            "note": "recorded for a future sigmoid score(); v1 rank() ships cosine only.",
        },
    }
    with open(os.path.join(MODEL_ROOT, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] MANIFEST.json (attn vision={attn['vision']} text={attn['text']}, "
          f"logit_scale.exp={manifest['scoring']['logit_scale_exp']})")


if __name__ == "__main__":
    main()
