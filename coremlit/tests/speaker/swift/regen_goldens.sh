#!/usr/bin/env bash
#
# Regenerates the argmax Swift reference goldens read by
# `coremlit/tests/speaker/parity_argmax_swift.rs` (design spec §5.1).
#
# The oracle is argmax's OWN Swift — `SpeakerSegmenterModel.predict` +
# `SpeakerEmbedderModel.embed` — driven by the XCTest in
# `Tests/ArgmaxTensorDump/DumpArgmaxTensors.swift`. argmax's `DiarizeCLI`
# cannot serve: it emits only post-clustering RTTM, never a tensor. See that
# file's header for why a TEST target (and not a plain executable) is the
# only way in: everything the dump needs is `internal` to `SpeakerKit`.
#
# The `argmax-oss-swift` checkout is used READ-ONLY, as a SwiftPM path
# dependency. Nothing is written into it; this package's own `.build/` (here,
# gitignored) takes every build product.
#
#   Usage:  coremlit/tests/speaker/swift/regen_goldens.sh
#
#   ARGMAX_SWIFT_SRC    argmax-oss-swift checkout   [default: the checkout's sibling argmax-oss-swift]
#   ARGMAX_TEST_MODELS  speakerkit-coreml artifacts [default: <workspace>/Models/argmax-speakerkit]
#
# Both defaults match `tests/common/mod.rs`'s `argmax_models_dir()` and the
# sibling-checkout layout this repo is developed in. Runtime is a few minutes
# on first run (it builds ArgmaxCore + WhisperKit + SpeakerKit), seconds after.

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

argmax_src="${ARGMAX_SWIFT_SRC:-$(cd "$(dirname "$workspace")/argmax-oss-swift" 2>/dev/null && pwd || true)}"
if [[ -z "$argmax_src" || ! -f "$argmax_src/Package.swift" ]]; then
  echo "error: no argmax-oss-swift checkout; set ARGMAX_SWIFT_SRC=<path>" >&2
  exit 1
fi

models="${ARGMAX_TEST_MODELS:-$workspace/Models/argmax-speakerkit}"
if [[ ! -d "$models" ]]; then
  echo "error: no argmax models at $models" >&2
  echo "       hf download argmaxinc/speakerkit-coreml --local-dir Models/argmax-speakerkit" >&2
  exit 1
fi

# The fixture set. The first two are speakerkit's own committed parity clips
# (`tests/common/mod.rs`'s FIXTURES); `ted_60` is whisperkit's, borrowed
# rather than copied — it is the only one long enough to produce MORE THAN
# ONE argmax 30 s chunk, and so the only one that exercises the `c = k*21 + w`
# grid mapping's `k >= 1` branch at all.
fixtures="02_pyannote_sample=$here/../fixtures/audio/02_pyannote_sample.wav"
fixtures="$fixtures;07_yuhewei_dongbei_english=$here/../fixtures/audio/07_yuhewei_dongbei_english.wav"
fixtures="$fixtures;ted_60=$workspace/coremlit/tests/whisper/fixtures/audio/ted_60.wav"

revision="$(git -C "$argmax_src" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "argmax-oss-swift : $argmax_src @ $revision"
echo "models           : $models"
echo "goldens          : $here/../fixtures/golden_argmax_swift/"

ARGMAX_SWIFT_SRC="$argmax_src" \
ARGMAX_SWIFT_REVISION="$revision" \
ARGMAX_TEST_MODELS="$models" \
SPEAKERKIT_FIXTURES="$fixtures" \
SPEAKERKIT_GOLDEN_OUT="$here/../fixtures/golden_argmax_swift" \
  swift test --package-path "$here" --filter DumpArgmaxTensors
