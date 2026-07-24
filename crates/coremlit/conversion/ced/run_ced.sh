#!/bin/bash
# CED ×4 conversion pipeline — re-derives all four mel->logits CoreML graphs deterministically.
# Toolchain: the conv-phase venv (torch 2.5.1, torchaudio 2.5.1, transformers 4.53.3,
# coremltools 9.0, onnxruntime, python 3.11). Sources: the four OFFICIAL public checkpoints
# mispeech/ced-{tiny,mini,small,base}, revision-pinned + SHA-verified (see scripts/_ced_common.py).
#
# Env (all optional; defaults are portable):
#   CED_PY         python interpreter of the conv venv (default: python3)
#   CED_CONV       working dir: src-<size> snapshots + staging (default: ~/.cache/coremlit-ced-conv)
#   CED_MODELS_OUT staged fp16 .mlmodelc root (default: <repo>/Models/ced)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${CED_PY:-python3}"
SIZES="${1:-tiny mini small base}"
CONV="${CED_CONV:-$HOME/.cache/coremlit-ced-conv}"
OUT_ROOT="${CED_MODELS_OUT:-$(cd "$HERE/../../../.." && pwd)/Models/ced}"
export CED_CONV="$CONV" CED_MODELS_OUT="$OUT_ROOT"

for size in $SIZES; do
  echo "=== $size: convert (fp16 + fp32) ==="
  "$PY" -u "$HERE/scripts/convert_ced.py" "$size"

  echo "=== $size: compile fp16 -> $OUT_ROOT/ced-$size/ced_$size.mlmodelc ==="
  DST="$OUT_ROOT/ced-$size"
  mkdir -p "$DST"
  rm -rf "$DST/ced_$size.mlmodelc"
  xcrun coremlcompiler compile "$CONV/staging/ced_$size.mlpackage" "$DST"

  echo "=== $size: CHECKSUMS.sha256 + MANIFEST.json ==="
  "$PY" -u "$HERE/scripts/write_manifest.py" "$size"
done

echo "=== verify (fail-closed: PyTorch fp32 vs CoreML fp32 floor, fp16 characterization) ==="
"$PY" -u "$HERE/scripts/verify_ced.py" $SIZES

echo "=== goldens (corpus.json per size + shared WAV fixtures) ==="
"$PY" -u "$HERE/scripts/generate_goldens.py"

echo "CED conversion pipeline complete."
