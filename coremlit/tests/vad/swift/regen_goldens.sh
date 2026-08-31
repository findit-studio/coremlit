#!/usr/bin/env bash
#
# Regenerates the FluidAudio Swift reference traces read by
# `coremlit/tests/vad/parity_swift.rs` (design spec §6 model-layer gate).
#
# The oracle is FluidAudio's OWN Swift — `VadManager.process([Float])` — driven
# by the XCTest in `Tests/VadTraceDump/DumpVadTraces.swift`. That method
# performs the exact 4096-sample chunking, 64-sample context stitching,
# repeat-last final-chunk padding and LSTM state carry-forward that
# `vadkit::VadModel` ports, so its per-chunk probabilities are the ground truth
# the Rust port must reproduce.
#
# The FluidAudio checkout is used READ-ONLY, as a SwiftPM path dependency.
# Nothing is written into it; this package's own `.build/` (here, gitignored)
# takes every build product.
#
#   Usage:  coremlit/tests/vad/swift/regen_goldens.sh
#
#   FLUIDAUDIO_SRC     FluidAudio checkout        [default: the checkout's sibling FluidAudio]
#   VADKIT_TEST_MODELS vadkit model artifacts dir [default: <workspace>/Models/vadkit]
#
# Both defaults match `tests/common/mod.rs`'s `models_dir()` and the sibling-
# checkout layout this repo is developed in. First run builds FluidAudio (a few
# minutes); seconds after.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The repository root is FOUND, not counted. A `../` hop count encodes this
# script's depth in the tree and fails SILENTLY when that depth changes — `cd
# ..` always succeeds, so a wrong count lands on a real directory that merely
# holds no models, and this script would stage its work somewhere else without
# a word. So walk up to the directory carrying `MODELS_LOCK`, a file that
# exists only at this repository's root, and refuse when there is none.
#
# The Rust side searches for the `[workspace]` manifest instead
# (`coremlit/tests/support/workspace_root.rs`); this script cannot, because
# `golden_provenance.rs` forbids it from naming Rust's build tool at all — the
# tripwire that keeps a "fall back to the Rust path" arm out of a script whose
# only source of numbers may be the external oracle.
workspace="$here"
while [[ "$workspace" != "/" && ! -f "$workspace/MODELS_LOCK" ]]; do
  workspace="$(dirname "$workspace")"
done
if [[ ! -f "$workspace/MODELS_LOCK" ]]; then
  echo "error: no MODELS_LOCK at or above $here — cannot locate the repository root" >&2
  exit 1
fi

fluidaudio_src="${FLUIDAUDIO_SRC:-$(cd "$(dirname "$workspace")/FluidAudio" 2>/dev/null && pwd || true)}"
if [[ -z "$fluidaudio_src" || ! -f "$fluidaudio_src/Package.swift" ]]; then
  echo "error: no FluidAudio checkout; set FLUIDAUDIO_SRC=<path>" >&2
  exit 1
fi

models="${VADKIT_TEST_MODELS:-$workspace/Models/vadkit}"
model="$models/silero-vad-unified-256ms-v6.2.1.mlmodelc"
if [[ ! -d "$model" ]]; then
  echo "error: no vadkit model at $model" >&2
  echo "       hf download FluidInference/silero-vad-coreml \\" >&2
  echo "         --include \"silero-vad-unified-256ms-v6.2.1*\" \\" >&2
  echo "         --revision b419383c55c110e2c9271fa6ee0ea83d03c70d96 --local-dir Models/vadkit" >&2
  exit 1
fi

# The fixture set: two real-speech clips from dia's parity corpus, borrowed by
# path from the speakerkit crate (`tests/common/mod.rs`'s FIXTURES). Together
# they exceed 40 chunks over >= 2 clips and exercise the short-final-chunk
# repeat-last padding path.
audio="$workspace/coremlit/tests/speaker/fixtures/audio"
fixtures="02_pyannote_sample=$audio/02_pyannote_sample.wav"
fixtures="$fixtures;07_yuhewei_dongbei_english=$audio/07_yuhewei_dongbei_english.wav"

revision="$(git -C "$fluidaudio_src" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "FluidAudio  : $fluidaudio_src @ $revision"
echo "model       : $model"
echo "goldens     : $here/../fixtures/golden_swift/"

FLUIDAUDIO_SRC="$fluidaudio_src" \
FLUIDAUDIO_REVISION="$revision" \
VADKIT_MODEL="$model" \
VADKIT_FIXTURES="$fixtures" \
VADKIT_GOLDEN_OUT="$here/../fixtures/golden_swift" \
  swift test --package-path "$here" --filter DumpVadTraces
