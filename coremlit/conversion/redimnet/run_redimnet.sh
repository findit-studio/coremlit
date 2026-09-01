#!/bin/bash
# ReDimNet-B5 -> CoreML, for coremlit's IDENTITY lane (issue #123). Re-derives the
# waveform->embedding graph deterministically from the OFFICIAL public release asset
# `b5-vox2-ft_lm.pt` (IDRnD/redimnet), pinned by SHA-256 because its release tag is
# literally named `latest` and is mutable.
#
# Toolchain: python 3.11, torch 2.5.0, torchaudio 2.5.0, coremltools 8.3.0, numpy 1.26.4.
# coremltools 8.3.0 is not a preference — it is the version that produced the graph this
# crate already ships (Models/speakerkit/wespeaker.mlmodelc/model.mil). Every stage
# OBSERVES its toolchain and refuses to record a version it did not run under.
#
# Env (all optional; defaults are portable):
#   REDIMNET_PY          python interpreter of the conv venv (default: python3)
#   REDIMNET_CONV        working dir: pinned source + staging (default:
#                        ~/.cache/coremlit-redimnet-conv)
#   REDIMNET_MODELS_OUT  staged fp16 .mlmodelc root (default: <repo>/Models/redimnet)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${REDIMNET_PY:-python3}"
CONV="${REDIMNET_CONV:-$HOME/.cache/coremlit-redimnet-conv}"

# The repository root is FOUND, not counted: walk up to the directory carrying
# `MODELS_LOCK`, a file that exists only at this repository's root. A `../` hop count
# encodes this script's depth in the tree and fails silently when that depth changes.
REPO="$HERE"
while [[ "$REPO" != "/" && ! -f "$REPO/MODELS_LOCK" ]]; do
  REPO="$(dirname "$REPO")"
done
[[ -f "$REPO/MODELS_LOCK" ]] || {
  echo "error: no MODELS_LOCK at or above $HERE — cannot locate the repository root" >&2
  exit 1
}
OUT_ROOT="${REDIMNET_MODELS_OUT:-$REPO/Models/redimnet}"
export REDIMNET_CONV="$CONV" REDIMNET_MODELS_OUT="$OUT_ROOT"

echo "=== convert (fp16 + fp32) ==="
"$PY" -u "$HERE/scripts/convert_redimnet.py"

echo "=== compile fp16 -> $OUT_ROOT/redimnet_b5.mlmodelc ==="
mkdir -p "$OUT_ROOT"
rm -rf "$OUT_ROOT/redimnet_b5.mlmodelc"
xcrun coremlcompiler compile "$CONV/staging/redimnet_b5.mlpackage" "$OUT_ROOT"

echo "=== CHECKSUMS.sha256 + MANIFEST.json ==="
"$PY" -u "$HERE/scripts/write_manifest.py"

echo "=== verify (fail-closed: PyTorch fp32 vs CoreML fp32 floor, fp16 per unit) ==="
"$PY" -u "$HERE/scripts/verify_redimnet.py"

echo "=== placement sweep (All / CpuAndGpu / CpuOnly / CpuAndNeuralEngine) ==="
"$PY" -u "$HERE/scripts/sweep_placement.py"

echo "ReDimNet-B5 conversion pipeline complete."
