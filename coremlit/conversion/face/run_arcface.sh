#!/bin/bash
# InsightFace `w600k_r50` (the `buffalo_l` recognition head) -> CoreML, for coremlit's
# `embeddings::face` door (issue #115).
#
#   run_arcface.sh                 the whole recipe
#   run_arcface.sh fixtures        re-cut the committed known-pairs fixtures only
#   run_arcface.sh reference       re-cut the committed ONNX reference embeddings only
#                                  (needs only numpy + onnxruntime, no torch/coremltools)
#
# RESEARCH USE ONLY. The weights are InsightFace's non-commercial research terms and the
# corpus (WebFace600K) is research-only as well. This recipe converts them so CI has a real
# artifact to run against — a use the owner ruled acceptable on the standing basis that
# coremlit never redistributes weight bytes — and the artifact it produces is published to a
# PRIVATE repository behind a `commercial-`prefixed feature that is never in `default`.
# See LICENCE_ROW.md.
#
# Toolchain: python 3.11, torch 2.5.0, coremltools 8.3.0, onnx 1.17.0, onnxruntime 1.20.1,
# onnx2torch 1.5.15, numpy 1.26.4, Pillow 11.0.0. coremltools 8.3.0 is not a preference — it
# is the version that produced every other bundle this crate ships. Every stage OBSERVES its
# toolchain and refuses to record a version it did not run under.
#
# Env (all optional; defaults are portable):
#   ARCFACE_PY          python interpreter of the conversion venv (default: python3)
#   ARCFACE_CONV        working dir: pinned source + staging (default:
#                       ~/.cache/coremlit-arcface-conv)
#   ARCFACE_MODELS_OUT  staged fp16 .mlmodelc root (default: <repo>/Models/facekit)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ARCFACE_PY:-python3}"

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
CONV="${ARCFACE_CONV:-$HOME/.cache/coremlit-arcface-conv}"
OUT_ROOT="${ARCFACE_MODELS_OUT:-$REPO/Models/facekit}"
export ARCFACE_CONV="$CONV" ARCFACE_MODELS_OUT="$OUT_ROOT"
BUNDLE="w600k_r50.mlmodelc"

case "${1:-all}" in
  all) ;;
  fixtures)
    echo "=== fetch source (pinned) ==="
    "$PY" -u "$HERE/scripts/fetch_source.py"
    echo "=== rebuild the committed known-pairs fixtures ==="
    "$PY" -u "$HERE/scripts/build_fixtures.py"
    exit 0
    ;;
  reference)
    echo "=== fetch source (pinned) ==="
    "$PY" -u "$HERE/scripts/fetch_source.py"
    echo "=== rebuild the committed ONNX reference embeddings ==="
    "$PY" -u "$HERE/scripts/write_onnx_reference.py"
    exit 0
    ;;
  *) echo "usage: $0 [all|fixtures|reference]" >&2; exit 2 ;;
esac

echo "=== fetch source: buffalo_l.zip, verified against its pin ==="
"$PY" -u "$HERE/scripts/fetch_source.py"

echo "=== probe the ONNX contract (L2 at the tail? I/O? preprocessing?) ==="
"$PY" -u "$HERE/scripts/probe_onnx_contract.py"

echo "=== convert (fp16 + fp32), after checking the ONNX -> torch rebuild ==="
"$PY" -u "$HERE/scripts/convert_arcface.py"

echo "=== compile fp16 -> $OUT_ROOT/$BUNDLE ==="
mkdir -p "$OUT_ROOT"
rm -rf "${OUT_ROOT:?}/$BUNDLE"
xcrun coremlcompiler compile "$CONV/staging/w600k_r50.mlpackage" "$OUT_ROOT"

echo "=== fixtures: NASA public-domain faces, aligned by align_oracle.py ==="
"$PY" -u "$HERE/scripts/build_fixtures.py"

echo "=== verify (fail-closed: fp32 vs ONNX, fp16 per unit, channel order, known pairs) ==="
"$PY" -u "$HERE/scripts/verify_arcface.py"

echo "=== ONNX reference embeddings -> tests/face/fixtures/onnx_reference.json ==="
"$PY" -u "$HERE/scripts/write_onnx_reference.py"

echo "=== placement sweep (All / CpuAndGpu / CpuOnly / CpuAndNeuralEngine) + throughput ==="
"$PY" -u "$HERE/scripts/sweep_placement.py"

echo "=== CHECKSUMS.sha256 + MANIFEST.json + model card ==="
"$PY" -u "$HERE/scripts/write_manifest.py"

echo "ArcFace w600k_r50 conversion pipeline complete."
