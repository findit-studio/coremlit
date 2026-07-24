#!/bin/bash
# siglip2-naflex CoreML conversion pipeline — re-derives both SigLIP2 towers
# deterministically from the OFFICIAL public checkpoint
# google/siglip2-base-patch16-naflex @ b53b807d3a2d5e2b3911292f2d69e5341cdc064c
# (Apache-2.0). Toolchain: the conv/ phase venv (python 3.11, torch 2.5.1,
# transformers 4.53.3, coremltools 9.0, numpy 1.26.4, pillow 12.3.0,
# tokenizers 0.21.2). See scripts/_siglip_common.py for the pins + SHA verify.
#
# ALL paths come from the environment (no hardcoded absolute paths). Set:
#   SIGLIP_CONV        base scratch dir with the .venv + src-model (required)
#   SIGLIP_GOLDENS     committed goldens dir (crates/coremlit/tests/siglip/fixtures/goldens)
#   SIGLIP_MODELS_OUT  gitignored Models/siglip2-naflex tree
# Optional: SIGLIP_SRC_MODEL, SIGLIP_STAGE, SIGLIP_ATTN (sdpa|eager).
set -euo pipefail

: "${SIGLIP_CONV:?set SIGLIP_CONV to the scratch dir holding .venv and src-model}"
: "${SIGLIP_GOLDENS:?set SIGLIP_GOLDENS to crates/coremlit/tests/siglip/fixtures/goldens}"
: "${SIGLIP_MODELS_OUT:?set SIGLIP_MODELS_OUT to the gitignored Models/siglip2-naflex tree}"
export SIGLIP_STAGE="${SIGLIP_STAGE:-$SIGLIP_CONV/siglip/staging}"
export TOKENIZERS_PARALLELISM=false
PY="$SIGLIP_CONV/.venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SIGLIP_MODELS_OUT/siglip2-base-patch16-naflex-512"

echo "== 1/6 convert both towers -> fp16 (shipped) + fp32 (reference) mlpackages + sidecar =="
"$PY" -u "$HERE/scripts/convert_vision.py"
"$PY" -u "$HERE/scripts/convert_text.py"

echo "== 2/6 compile every mlpackage -> mlmodelc (the gates consume model.mil) =="
cd "$SIGLIP_STAGE"
for p in siglip2_vision_512 siglip2_text_64 siglip2_vision_512_fp32 siglip2_text_64_fp32; do
  [ -d "$p.mlpackage" ] && xcrun coremlcompiler compile "$p.mlpackage" .
done

echo "== 3/6 stage shipped fp16 bundles + sidecar into the Models tree =="
mkdir -p "$ROOT"
cp -R "$SIGLIP_STAGE/siglip2_vision_512.mlmodelc" "$ROOT/"
cp -R "$SIGLIP_STAGE/siglip2_text_64.mlmodelc" "$ROOT/"
cp "$SIGLIP_STAGE/pos_embed_16x16x768.f32le.bin" "$ROOT/"

echo "== 4/6 CHECKSUMS.sha256 + MANIFEST.json over the shipped bundle =="
"$PY" -u "$HERE/scripts/stage_manifest.py"

echo "== 5/6 fail-closed verify matrix (fp32-vs-torch >= 0.9999; CpuAndGpu fp16 gate >= 0.99917) =="
"$PY" -u "$HERE/scripts/verify_towers.py"

echo "== 6/6 committed goldens (corpus.json + preprocess.json) + staged .npy fixtures =="
"$PY" -u "$HERE/scripts/generate_goldens.py"

echo "siglip2-naflex conversion pipeline complete."
