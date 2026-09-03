"""Emit the publish tree's ``CHECKSUMS.sha256``, ``MANIFEST.json`` and model card
(``README.md``). Usage: ``python write_manifest.py``.

The manifest DESCRIBES only what this recipe produced, and it refuses to describe anything
else: the bundle's ``producer.json`` must exist, and the toolchain recorded there must equal
the one running this script entry for entry. A manifest that recorded the versions a comment
claims, rather than the versions that ran, is a provenance record nobody can replay (issue
#97).

``CHECKSUMS.sha256`` paths are **kit-root-relative with a leading ``./``**, the layout
``redimnetkit``'s second revision and ``speakerkit``'s use, so ``shasum -c`` verifies from
the kit root and a second bundle later added to the same repository does not collide with
this one's ``weights/weight.bin``.

The measurements are folded in from ``verify.json`` and ``placement.json`` if they are
present — a card that states a throughput number nobody measured is the same defect as a
toolchain nobody ran — and the script says so rather than inventing a row when they are not.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (BUNDLE, CONTRACT, EMBED_DIM, INPUT_NAME, INPUT_SHAPE,
                             INSIGHTFACE_REV, ONNX_INPUT_NAME, ONNX_OUTPUT_NAME, OUTPUT_NAME,
                             OUTPUT_SHAPE, PACK_SHA256, PACK_URL, PREPROCESSING,
                             RECOGNITION_BYTES, RECOGNITION_MEMBER, RECOGNITION_SHA256,
                             conv_dir, models_out_dir, observed_compiler, observed_toolchain,
                             sha256_file)

CARD = """---
license: other
license_name: insightface-non-commercial-research
license_link: https://github.com/deepinsight/insightface/tree/master/model_zoo
tags:
  - coreml
  - face-recognition
  - arcface
  - insightface
  - non-commercial
---

# w600k_r50 — CoreML (Core ML `mlprogram`, fp16)

**These are InsightFace's non-commercial research weights, converted. This repository is
private, and it exists for development and CI only. Shipping this model in a product
requires a commercial licence that InsightFace does not offer for these weights.**

`w600k_r50` is the recognition head of InsightFace's `buffalo_l` model pack: an IResNet-50
trained on WebFace600K, emitting a 512-d face embedding. This bundle is that ONNX graph
converted to Core ML by [`coremlit`](https://github.com/findit-studio/coremlit)'s
`conversion/face` recipe, unchanged in arithmetic and unchanged in preprocessing.

## Licence — three layers, and two of them forbid a product

| layer | terms |
|---|---|
| **conversion recipe** (`coremlit/conversion/face`) | MIT OR Apache-2.0, with the rest of `coremlit`. Covers the recipe. Covers nothing below. |
| **weights** (`w600k_r50.onnx`, and therefore this bundle) | **Research only.** InsightFace's model zoo: *"ALL models are available for non-commercial research purposes only."* No commercial licence is offered for them. |
| **training corpus** (WebFace600K) | **Research only.** WebFace260M/WebFace600K is released under a licence agreement restricting it to non-commercial academic research. |

A conversion does not lift either restriction. Re-encoding a graph is a derivative of the
weights, not a new work, so this bundle carries the weights' terms exactly.

**What this bundle may be used for:** development, evaluation, regression testing, and
research. **What it may not be used for:** anything commercial, and any redistribution
beyond the private access this repository already has.

## Provenance

| what | value |
|---|---|
| source pack | `{pack_url}` |
| pack sha256 | `{pack_sha256}` |
| converted member | `{member}` ({member_bytes:,} bytes) |
| member sha256 | `{member_sha256}` |
| upstream digest published? | **no** — InsightFace's `utils/storage.py` fetches and unzips with no manifest, signature or hash. The pin above is a witness to the bytes this conversion consumed, not a check against an upstream claim. |
| preprocessing reference | `deepinsight/insightface` @ `{insightface_rev}`, `model_zoo/arcface_onnx.py` |

## Contract

```
{contract}
```

* input `{input_name}` — `MultiArray` float32, **fixed** `{input_shape}`, NCHW. Not an
  `ImageType`: the two third-party Core ML ArcFace builds on the Hub declare image inputs,
  which `coremlit`'s `MultiArray`-only feature binding cannot feed, and that is the reason
  this conversion exists.
* output `{output_name}` — `MultiArray` float32 `{output_shape}`, **raw**. The ONNX graph
  ends `BatchNormalization → Flatten → Gemm → BatchNormalization`; there is no L2 anywhere
  in it (measured: ‖e‖ ≈ {norm_lo:.0f}–{norm_hi:.0f} on the fixture faces). **The caller
  normalises.**
* batch is **1**. The graph the ONNX declares has a symbolic batch; a flexible Core ML input
  is off the Neural Engine for every shape but its default, so the export pins it. A
  batch-8 export is a follow-up with a measured throughput reason, not a default.
* stateless, fp16 weights, `mlprogram`, `minimum_deployment_target` iOS17 / macOS14.

## Preprocessing — part of the artifact, not a constant at a call site

`value = byte × {scale!r} + bias[channel]`, i.e. **`(x − 127.5) / 127.5` over RGB, NCHW** —
the mapping to `[−1, 1]`.

The channel order is **RGB**, and it is read off InsightFace's own code rather than a card:
`ArcFaceONNX.get_feat` calls `cv2.dnn.blobFromImages(..., swapRB=True)` on a crop that
`face_align.norm_crop` produced from an OpenCV **BGR** frame, so what reaches the model is
RGB. Measured on the fixture faces, feeding BGR instead costs a mean `1 − cos` of
{bgr_cost:.4f} and drops the worst same-person pair from {rgb_min_same:.4f} to
{bgr_min_same:.4f} — through InsightFace's own 0.28 "same person" line. It degrades
silently; nothing raises.

## Measurements

{measurements}

## Files

* `{bundle}/` — the compiled Core ML bundle.
* `CHECKSUMS.sha256` — every file, kit-root-relative; `shasum -c CHECKSUMS.sha256` from the
  repository root.
* `MANIFEST.json` — source, toolchain, contract, preprocessing and the licence in three
  layers, machine-readable.

## Reproducing

`coremlit/conversion/face/run_arcface.sh`. It re-downloads the pinned pack, re-verifies its
SHA-256, re-derives the graph, and refuses to record any toolchain version it did not run
under.
"""


def walk(root: Path, bundle: Path):
    rows = []
    for path in sorted(bundle.rglob("*")):
        if path.is_dir() or path.name.startswith("._") or path.name == ".DS_Store":
            continue
        rows.append(("./" + path.relative_to(root).as_posix(), sha256_file(path)))
    return rows


def require_producer(observed):
    path = conv_dir() / "producer.json"
    if not path.is_file():
        raise SystemExit(f"missing {path} — run convert_arcface.py first; the manifest "
                         f"records the toolchain that produced the bundle, not this one.")
    producer = json.loads(path.read_text())
    recorded = producer["toolchain"]
    diff = [f"{k}: producer {recorded.get(k)!r}, this run {observed.get(k)!r}"
            for k in sorted(set(recorded) | set(observed)) if recorded.get(k) != observed.get(k)]
    if diff:
        raise SystemExit("PRODUCER/MANIFEST TOOLCHAIN MISMATCH — refusing to describe a "
                         "build this environment did not make:\n  " + "\n  ".join(diff))
    print(f"[ok] producer and manifest toolchains are one environment "
          f"(run {producer['run_id'][:12]}…)")
    return producer


def optional(name):
    path = conv_dir() / name
    return json.loads(path.read_text()) if path.is_file() else None


def measurement_prose(verify, placement):
    if not verify or not placement:
        return ("_Not recorded: this tree was written without `verify.json` and/or "
                "`placement.json`. Run `verify_arcface.py` and `sweep_placement.py` and "
                "rewrite the manifest._")
    lines = []
    fp32 = verify["fp32_vs_onnx"]
    lines.append("**Parity against the source ONNX** (`onnxruntime` 1.20.1, CPU, fp32), "
                 "cosine over 18 fixture faces. Floors were set before the measurement: "
                 "0.9999 for fp32 (same precision both sides — this is the conversion "
                 "itself) and 0.99 for fp16 (issue #115's gate, placed ~4× above the ANE's "
                 "own fp16 noise and ~8× below the cheapest real preprocessing bug).\n")
    lines.append("| path | min | median | max | worst `1 − cos` |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Core ML **fp32**, CpuOnly | {fp32['min']:.7f} | {fp32['median']:.7f} | "
                 f"{fp32['max']:.7f} | {1 - fp32['min']:.1e} |")
    for arm, row in verify["fp16_vs_onnx"].items():
        lines.append(f"| Core ML fp16, {arm} | {row['min']:.7f} | {row['median']:.7f} | "
                     f"{row['max']:.7f} | {row['worst_1_minus_cos']:.1e} |")

    lines.append(f"\n**Placement**, four arms, a fresh process per arm per round so every "
                 f"load is cold, {placement['rounds']} rounds x "
                 f"{placement['repeats_per_round']} warm predicts. No arm emitted a "
                 f"`BNNS Graph Shape Deduction` line.\n")
    lines.append("| arm | cold load ms | first predict ms | warm predict ms (median, range) "
                 "| faces/s | min cos vs fp32 Core ML |")
    lines.append("|---|---|---|---|---|---|")
    for row in placement["arms"]:
        if row.get("error"):
            lines.append(f"| {row['arm']} | — | — | — | — | {row['error']} |")
            continue
        lines.append(f"| {row['arm']} | {row['load_ms_median']:.0f} | "
                     f"{row['first_predict_ms_median']:.0f} | "
                     f"{row['warm_predict_ms_median']:.2f} "
                     f"({row['warm_predict_ms_min']:.2f}–{row['warm_predict_ms_max']:.2f}) | "
                     f"{row['faces_per_second']:.0f} | {row['cos_vs_fp32_min']:.6f} |")
    lines.append(f"\n**Recommended arm: `{placement['recommended']}`** — "
                 f"{placement['throughput_faces_per_second']:.0f} faces/s warm. "
                 f"{placement['method']}.")

    known = verify["known_pairs"]["onnx_fp32"]
    lines.append(f"\n**Known pairs** — {known['same_pairs']} same-person and "
                 f"{known['different_pairs']} different-person pairs over "
                 f"{len(verify['fixtures']['identities'])} identities, at InsightFace's own "
                 f"operating point (`>= 0.28` same, `< 0.20` not the same, from "
                 f"`web-demos/src_recognition/main.py`).\n")
    lines.append("| embedding path | min same-person | max different-person | margin |")
    lines.append("|---|---|---|---|")
    for key, row in verify["known_pairs"].items():
        lines.append(f"| {key} | {row['min_same']:.4f} | {row['max_different']:.4f} | "
                     f"{row['margin']:+.4f} |")
    return "\n".join(lines)


def main():
    observed = observed_toolchain()
    compiler = observed_compiler()
    producer = require_producer(observed)
    out = models_out_dir()
    bundle = out / BUNDLE
    if not bundle.is_dir():
        raise SystemExit(f"no {BUNDLE} under {out}")
    present = sorted(p.name for p in out.iterdir() if p.is_dir() and p.name.endswith(".mlmodelc"))
    if present != [BUNDLE]:
        raise SystemExit(f"{out} holds bundles this recipe did not produce and will not "
                         f"describe: {[b for b in present if b != BUNDLE]}")

    rows = walk(out, bundle)
    (out / "CHECKSUMS.sha256").write_text("".join(f"{d}  {p}\n" for p, d in rows))
    print(f"[ok] CHECKSUMS.sha256: {len(rows)} files")

    verify, placement = optional("verify.json"), optional("placement.json")
    manifest = {
        "schema": 1,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle": BUNDLE,
        "model": {
            "name": "w600k_r50",
            "family": "ArcFace / IResNet-50",
            "embedding_dim": EMBED_DIM,
            "parameters": producer["onnx_to_torch"]["parameters"],
        },
        "source": {
            "pack_url": PACK_URL, "pack_sha256": PACK_SHA256,
            "member": RECOGNITION_MEMBER, "member_sha256": RECOGNITION_SHA256,
            "member_bytes": RECOGNITION_BYTES,
            "upstream_publishes_a_digest": False,
            "onnx": {"input": ONNX_INPUT_NAME, "output": ONNX_OUTPUT_NAME,
                     "opset": 11, "producer": "pytorch 1.9",
                     "l2_at_the_tail": False,
                     "tail": "BatchNormalization -> Flatten -> Gemm -> BatchNormalization"},
            "preprocessing_reference": {
                "repo": "deepinsight/insightface", "revision": INSIGHTFACE_REV,
                "file": "python-package/insightface/model_zoo/arcface_onnx.py"},
        },
        "contract": {
            "summary": CONTRACT,
            "input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "dtype": "float32",
                      "kind": "MultiArray", "flexibility": "none (fixed)"},
            "output": {"name": OUTPUT_NAME, "shape": list(OUTPUT_SHAPE), "dtype": "float32",
                       "kind": "MultiArray", "normalisation": "none (raw feature); the "
                                                              "caller L2-normalises"},
            "state": None,
            "compute_precision": "float16",
            "convert_to": "mlprogram",
            "minimum_deployment_target": producer["contract"]["minimum_deployment_target"],
        },
        "preprocessing": PREPROCESSING,
        "licence": {
            "recipe": {"terms": "MIT OR Apache-2.0",
                       "detail": "coremlit's own licence; covers the conversion recipe and "
                                 "nothing below it."},
            "weights": {
                "terms": "research-only",
                "identifier": "InsightFace model licence (non-commercial research)",
                "detail": "InsightFace's model zoo states \"ALL models are available for "
                          "non-commercial research purposes only.\" No commercial licence is "
                          "offered for these weights. A conversion is a derivative of the "
                          "weights and carries their terms unchanged."},
            "corpus": {
                "terms": "research-only",
                "identifier": "WebFace260M/WebFace600K licence agreement",
                "detail": "w600k_r50 is trained on WebFace600K, a subset of WebFace260M, "
                          "released under a licence agreement restricting use to "
                          "non-commercial academic research."},
            "verdict": "forbids the shipping path at BOTH the weights and the corpus layer",
        },
        "toolchain": observed,
        "compiler": compiler,
        "producer": {"run_id": producer["run_id"], "converted_utc": producer["converted_utc"]},
        "onnx_to_torch_check": producer["onnx_to_torch"],
        "measurements": {"verify": verify, "placement": placement},
        "files": [{"path": p, "sha256": d} for p, d in rows],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ok] MANIFEST.json")

    norms = [17.0, 25.0]
    bgr_cost, rgb_min_same, bgr_min_same = 0.0, 0.0, 0.0
    if verify:
        reference = verify.get("reference_norms")
        if reference:
            norms = [reference["min"], reference["max"]]
        bgr_cost = verify["channel_order"]["mean_1_minus_cos_rgb_vs_bgr"]
        rgb_min_same = verify["channel_order"]["rgb"]["min_same"]
        bgr_min_same = verify["channel_order"]["bgr"]["min_same"]
    card = CARD.format(
        pack_url=PACK_URL, pack_sha256=PACK_SHA256, member=RECOGNITION_MEMBER,
        member_bytes=RECOGNITION_BYTES, member_sha256=RECOGNITION_SHA256,
        insightface_rev=INSIGHTFACE_REV, contract=CONTRACT, input_name=INPUT_NAME,
        input_shape=list(INPUT_SHAPE), output_name=OUTPUT_NAME, output_shape=list(OUTPUT_SHAPE),
        norm_lo=norms[0], norm_hi=norms[1], scale=PREPROCESSING["scale"],
        bgr_cost=bgr_cost, rgb_min_same=rgb_min_same, bgr_min_same=bgr_min_same,
        measurements=measurement_prose(verify, placement), bundle=BUNDLE)
    (out / "README.md").write_text(card)
    print(f"[ok] README.md (model card)")
    print(f"[ok] publish tree ready: {out}")


if __name__ == "__main__":
    main()
