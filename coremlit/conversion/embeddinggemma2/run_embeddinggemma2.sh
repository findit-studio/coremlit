#!/bin/bash
# EmbeddingGemma 2's fp32 audio package: converted from the original checkpoint
# google/embeddinggemma-2 @ f9972ac22e696a105c482202646979e33e4b5a1e, verified,
# and staged with its CHECKSUMS.sha256 for upload. It uploads nothing: the
# upload is a separate step (knife 2's row U), never run together with this.
#
# "Converted" means re-derivable to the floors in scripts/verify_audio_fp32.py,
# NOT bit-reproducible: neither the Core ML compiler nor torch's fp32 reduction
# order is pinned by the toolchain's versions.
#
# Every path comes from the environment:
#   EG2_CONV        the scratch directory holding the recipe's .venv (required)
#   EG2_SRC         the pinned checkpoint (default: $EG2_CONV/src-model)
#   EG2_MODELS      FluidInference's staged tree, read only, for the fp16
#                   package's contract (required)
#   EG2_TESTS       coremlit/tests, for the committed clips (required)
#   EG2_MODELS_OUT  the directory the package is staged into for upload (required)
#   EG2_GOLDENS     coremlit/tests/embeddinggemma2/fixtures/goldens, rewritten
#                   only with EG2_REGEN_GOLDENS=1
#   EG2_STAGE       the conversion's scratch (default: $EG2_CONV/eg2-stage)
set -euo pipefail

: "${EG2_CONV:?set EG2_CONV to the scratch directory holding .venv}"
: "${EG2_MODELS:?set EG2_MODELS to the staged FluidInference tree}"
: "${EG2_TESTS:?set EG2_TESTS to coremlit/tests}"
: "${EG2_MODELS_OUT:?set EG2_MODELS_OUT to the directory to stage the package into}"
# Absolute before anything uses them: a relative value beginning with `-` would
# be read as an option.
abspath_of() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$PWD/$1" ;; esac; }
EG2_CONV="$(abspath_of "$EG2_CONV")"
EG2_SRC="$(abspath_of "${EG2_SRC:-$EG2_CONV/src-model}")"
EG2_MODELS="$(abspath_of "$EG2_MODELS")"
EG2_TESTS="$(abspath_of "$EG2_TESTS")"
EG2_MODELS_OUT="$(abspath_of "$EG2_MODELS_OUT")"
EG2_STAGE="$(abspath_of "${EG2_STAGE:-$EG2_CONV/eg2-stage}")"
export EG2_CONV EG2_SRC EG2_MODELS EG2_TESTS EG2_MODELS_OUT EG2_STAGE
export TOKENIZERS_PARALLELISM=false
PY="$EG2_CONV/.venv/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${EG2_REGEN_GOLDENS:-0}" = "1" ]; then
  : "${EG2_GOLDENS:?set EG2_GOLDENS to rewrite the goldens}"
  : "${EG2_FLUIDUSE:?set EG2_FLUIDUSE to the FluidUse checkout at b51f7ad}"
  export EG2_GOLDENS="$(abspath_of "$EG2_GOLDENS")"
  echo "== 0/3 regenerate the committed R1 goldens =="
  "$PY" -u "$HERE/scripts/generate_goldens.py"
fi

echo "== 1/3 faithfulness gate, trace and convert the audio tower at FLOAT32 =="
"$PY" -u "$HERE/scripts/convert_audio_fp32.py"

echo "== 2/3 verify: the fp16 package's contract; token-level parity against PyTorch fp32 =="
"$PY" -u "$HERE/scripts/verify_audio_fp32.py"

echo "== 3/3 stage the package and write CHECKSUMS.sha256 =="
"$PY" -u "$HERE/scripts/stage_manifest.py"

echo "the fp32 audio recipe is complete: $EG2_MODELS_OUT"
