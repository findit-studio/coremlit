#!/bin/bash
# ReDimNet -> CoreML, for coremlit's IDENTITY lane (issue #123). ONE recipe, several
# artifacts: the variant is the first argument and there is deliberately no default.
#
#   run_redimnet.sh b5       b5-vox2-ft_lm.pt  -> redimnet_b5.mlmodelc
#   run_redimnet.sh b2       b2-vox2-ft_lm.pt  -> redimnet_b2.mlmodelc
#   run_redimnet.sh b2_ptn   b2-vox2-ptn.pt    -> redimnet_b2_ptn.mlmodelc   (2 s pretrain,
#                                                  NO published metrics of any kind)
#
# Each re-derives the mel->embedding graph deterministically from the OFFICIAL public
# release asset (IDRnD/redimnet), pinned by SHA-256 because the release tag is literally
# named `latest` and is mutable. Every variant shares one front end, one window and one
# I/O contract, and the recipe asserts that against each checkpoint rather than assuming
# it — which is what lets one Rust door serve them all.
#
# Toolchain: python 3.11, torch 2.5.0, torchaudio 2.5.0, coremltools 8.3.0, numpy 1.26.4.
# coremltools 8.3.0 is not a preference — it is the version that produced the graph this
# crate already ships (Models/speakerkit/wespeaker.mlmodelc/model.mil). Every stage
# OBSERVES its toolchain and refuses to record a version it did not run under.
#
# The manifest step describes EVERY bundle staged under the output root, so to publish a
# tree with several artifacts run the recipe once per variant into the same output root
# and the last run's manifest covers them all.
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

VARIANT="${1:-}"
case "$VARIANT" in
  b5|b2|b2_ptn) ;;
  *)
    echo "usage: $0 <b5|b2|b2_ptn>   (no default: a recipe that converts a size nobody" >&2
    echo "       asked for records provenance nobody can replay)" >&2
    exit 2
    ;;
esac

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
export REDIMNET_VARIANT="$VARIANT" REDIMNET_CONV="$CONV" REDIMNET_MODELS_OUT="$OUT_ROOT"
BUNDLE="redimnet_${VARIANT}"

echo "=== [$VARIANT] convert (fp16 + fp32) ==="
"$PY" -u "$HERE/scripts/convert_redimnet.py"

echo "=== [$VARIANT] compile fp16 -> $OUT_ROOT/$BUNDLE.mlmodelc ==="
mkdir -p "$OUT_ROOT"
rm -rf "$OUT_ROOT/$BUNDLE.mlmodelc"
xcrun coremlcompiler compile "$CONV/staging/$BUNDLE.mlpackage" "$OUT_ROOT"

echo "=== CHECKSUMS.sha256 + MANIFEST.json (every bundle under $OUT_ROOT) ==="
"$PY" -u "$HERE/scripts/write_manifest.py"

echo "=== [$VARIANT] verify (fail-closed: PyTorch fp32 vs CoreML fp32 floor, fp16 per unit) ==="
"$PY" -u "$HERE/scripts/verify_redimnet.py"

echo "=== [$VARIANT] placement sweep (All / CpuAndGpu / CpuOnly / CpuAndNeuralEngine) ==="
"$PY" -u "$HERE/scripts/sweep_placement.py"

echo "ReDimNet [$VARIANT] conversion pipeline complete."
