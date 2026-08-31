#!/usr/bin/env bash
#
# Regenerates the three committed whisper goldens read by
# `coremlit/tests/whisper/{parity_es,parity_jfk}.rs` (and, for its host
# provenance only, `streaming.rs`).
#
#   fixtures/golden/es_tiny_golden.json         es_test_clip.wav, --language es
#   fixtures/golden/jfk_tiny_golden.json        jfk.wav, auto-detect
#   fixtures/golden/jfk_tiny_words_golden.json  jfk.wav, --word-timestamps
#
# ─────────────────────────────────────────────────────────────────────────────
# THE ONE RULE: the goldens come from `whisperkit-cli`, NEVER from coremlit.
#
# A golden's only value is that something other than the code under test
# produced it. Re-baselining one against coremlit's own output would leave the
# parity gates asserting that coremlit agrees with coremlit — green forever,
# proving nothing. So this script has exactly one source of numbers: the
# `--report` JSON that `whisperkit-cli` writes. Every `text`, `language`,
# `tokens`, `segments` and `words` value below is a `jq` projection of that
# file.
#
# It follows that this script must never build, link, or run coremlit. It
# invokes `whisperkit-cli` and `jq`, and nothing else. Rust's build tool is not
# spelled anywhere in this file, and the hermetic test
# `whisper_golden_provenance::regen_script_cannot_emit_coremlits_own_output`
# greps for it and fails the suite if it ever appears — so the day someone adds
# a "just fall back to the Rust path" arm here, CI refuses it on the next PR.
#
# Regenerating IS legitimate — that is why this script exists. What makes it
# legitimate is that the oracle stays external and the host is recorded:
#
#   1. produced by whisperkit-cli (this script; there is no other path),
#   2. stamped with `generationHost` (below; read off the running machine),
#   3. reviewed by a human as a diff, because changed oracle output is news.
#
# A tolerance is never the answer, on any host. See `parity_es.rs`'s module doc.
# ─────────────────────────────────────────────────────────────────────────────
#
#   Usage:  coremlit/tests/whisper/swift/regen_goldens.sh
#
#   WHISPERKIT_TEST_MODELS  model root       [default: <workspace>/Models]
#   WHISPER_GOLDEN_OUT      output directory [default: the committed fixtures]
#
# `WHISPER_GOLDEN_OUT` is what the CI regeneration job sets: it points the
# script at a scratch directory and uploads the result as an artifact, so the
# job can produce goldens without being able to commit them.
#
# whisperkit-cli is not vendored — `brew install whisperkit-cli`. The tiny model
# is the one MODELS_LOCK's first table stages.

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

die() { echo "error: $*" >&2; exit 1; }

command -v whisperkit-cli >/dev/null 2>&1 || die "whisperkit-cli is not on PATH.
       brew install whisperkit-cli
       There is deliberately no fallback: a golden this script could produce
       without the Swift oracle would not be a golden."
command -v jq >/dev/null 2>&1 || die "jq is not on PATH (brew install jq)"

models="${WHISPERKIT_TEST_MODELS:-$workspace/Models}"
model="$models/whisperkit-coreml/openai_whisper-tiny"
[[ -d "$model" ]] || die "no tiny model at $model
       hf download argmaxinc/whisperkit-coreml --include 'openai_whisper-tiny/*' \\
         --revision <MODELS_LOCK table 1 revision> --local-dir Models/whisperkit-coreml"

audio="$here/../fixtures/audio"
out="${WHISPER_GOLDEN_OUT:-$here/../fixtures/golden}"
mkdir -p "$out"

# ── generationHost: read off THIS machine, never hardcoded ──────────────────
#
# The same four values, read from the same three sysctl keys, that the test-side
# `HostClass::running()` reads (`tests/support/host_class.rs`) — so a golden
# stamped here compares equal on the machine that produced it. `uname -m`
# already spells Apple Silicon `arm64`, which is the spelling `HostClass`
# normalizes Rust's `aarch64` to; under Rosetta both say `x86_64`.
os_build="$(/usr/sbin/sysctl -n kern.osversion)"
os_version="$(/usr/sbin/sysctl -n kern.osproductversion)"
chip="$(/usr/sbin/sysctl -n machdep.cpu.brand_string)"
arch="$(uname -m)"
for v in os_build os_version chip arch; do
  [[ -n "${!v}" ]] || die "empty host-class field '$v' — cannot stamp generationHost"
done

# whisperkit-cli loads a tokenizer of its own (the model bundle carries none).
# Left unset it picks its own Hub cache location; pinned here to a directory
# OUTSIDE the repository so a regeneration never leaves untracked files in the
# checkout — which the CI job then asserts, and which would otherwise be
# indistinguishable from the job having written a golden in tree. Kept across
# runs rather than inside `$raw` so repeat runs do not re-download it.
#
# This tokenizer detokenizes `text`; the `tokens` are model output and do not
# depend on it. MODELS_LOCK's table 2 pins the tokenizer the RUST side reads, and
# the two are not forced to agree — so a regenerated golden whose `text` moved
# while its `tokens` did not is this, not the port.
tokenizer_cache="${WHISPER_TOKENIZER_CACHE:-${TMPDIR:-/tmp}/whisperkit-cli-tokenizers}"
mkdir -p "$tokenizer_cache"

cli_version="$(whisperkit-cli --version 2>/dev/null | head -1 | tr -d '\n' || true)"
source_field="whisperkit-cli @ argmax-oss-swift${cli_version:+ ($cli_version)}"

echo "whisperkit-cli : $(command -v whisperkit-cli) ${cli_version:-(version unknown)}"
echo "model          : $model"
echo "tokenizer cache: $tokenizer_cache"
echo "generationHost : macOS $os_version (build $os_build), $chip, $arch"
echo "goldens        : $out"
echo

raw="$(mktemp -d "${TMPDIR:-/tmp}/whisper-goldens.XXXXXX")"
trap 'rm -rf "$raw"' EXIT


# ── The pinned invocation ────────────────────────────────────────────────────
#
# The CLI's own defaults differ from the library's, so every knob the goldens
# were captured under is passed explicitly (`TranscribeCLIArguments.swift`
# leaves the thresholds nil and `usePrefillPrompt` false). `--temperature 0`
# with the fallback ladder pinned means pure argmax and no RNG, i.e. a
# reproducible decode on a fixed host. `=-1.0` / `=-1.5` forms because
# swift-argument-parser eats bare negative values.
#
# COMPUTE PATH — do not "fix" this to cpuOnly. The CLI defaults both
# audioEncoderComputeUnits and textDecoderComputeUnits to cpuAndNeuralEngine,
# which is exactly what coremlit ships (DEFAULT_ENCODER_COMPUTE_UNITS /
# DEFAULT_DECODER_COMPUTE_UNITS) and what the parity gates assert they run on.
# The goldens are an ANE capture compared against an ANE decode; pinning either
# side to CpuOnly would validate a path nobody runs. That is also precisely why
# these goldens need `generationHost`: ANE fp16 is where the drift lives.
#
# The fallback COUNT is deliberately not in the shared set: the two token
# goldens were captured at 5 and the word-timestamp golden at 0
# (`temperatureFallbackCount=0` in its recorded `options`, from the SwiftPM
# driver's `DecodingOptions(...)`). Passing both would leave the winner up to
# swift-argument-parser's last-wins behaviour, which is not something a golden
# should rest on.
common_flags=(
  --model-path "$model"
  --download-tokenizer-path "$tokenizer_cache"
  --use-prefill-prompt
  --temperature 0
  --temperature-increment-on-fallback 0.2
  --compression-ratio-threshold 2.4
  --logprob-threshold=-1.0
  --first-token-log-prob-threshold=-1.5
  --no-speech-threshold 0.6
  --report --report-path "$raw"
)

# `jq` filters. Both project the CLI report and add nothing of our own except
# the provenance fields — there is no path here from a coremlit value into a
# golden.
host_obj='{osBuild:$osBuild, osProductVersion:$osProductVersion, chip:$chip, arch:$arch}'

# Key order matches the committed goldens so a regeneration diffs as content,
# not as a reshuffle; `generationHost` slots in after `source`, where provenance
# belongs.
tokens_filter="{ model: \$model,
                 source: \$source,
                 generationHost: $host_obj,
                 text: .text,
                 language: .language,
                 tokens: [.segments[].tokens[]],
                 segments: [.segments[] | {id, start, end, text, tokens}] }"

words_filter="{ model: \$model,
                source: \$source,
                generationHost: $host_obj,
                options: \$options,
                text: .text,
                language: .language,
                totalDecodingLoops: .timings.totalDecodingLoops,
                totalDecodingWindows: .timings.totalDecodingWindows,
                segments: [.segments[] | {avgLogprob, compressionRatio, end, id, seek,
                                          start, temperature, text, tokens,
                                          words: [(.words // [])[] |
                                                  {end, probability, start, tokens, word}]}] }"

emit() { # <report-basename> <jq-filter> <golden-name> [extra jq --arg pairs...]
  local report="$raw/$1.json" filter="$2" golden="$3"
  shift 3
  [[ -s "$report" ]] || die "whisperkit-cli wrote no report at $report — nothing to build a golden from"
  jq --arg model openai_whisper-tiny \
     --arg source "$source_field" \
     --arg osBuild "$os_build" \
     --arg osProductVersion "$os_version" \
     --arg chip "$chip" \
     --arg arch "$arch" \
     "$@" \
     "$filter" "$report" > "$out/$golden"
  echo "wrote $out/$golden"
}

echo "== es_test_clip.wav (--language es) =="
whisperkit-cli transcribe --audio-path "$audio/es_test_clip.wav" --language es \
  --temperature-fallback-count 5 "${common_flags[@]}"
emit es_test_clip "$tokens_filter" es_tiny_golden.json

echo "== jfk.wav (auto-detect) =="
whisperkit-cli transcribe --audio-path "$audio/jfk.wav" \
  --temperature-fallback-count 5 "${common_flags[@]}"
emit jfk "$tokens_filter" jfk_tiny_golden.json

# The word-timestamp golden. `parity_jfk.rs::oracle_options` mirrors these
# exactly, so the two must move together. `--first-token-log-prob-threshold=-1.5`
# above is load-bearing here: the original capture went through a SwiftPM driver
# calling `DecodingOptions(...)`, whose default for that knob is -1.5, while the
# CLI's flag defaults to nil and is passed straight through
# (`TranscribeCLIUtils.swift:64`). Omit it and the fallback ladder can behave
# differently from the committed capture for a reason that has nothing to do
# with the host.
echo "== jfk.wav (--word-timestamps) =="
whisperkit-cli transcribe --audio-path "$audio/jfk.wav" --word-timestamps \
  --temperature-fallback-count 0 --skip-special-tokens \
  --concurrent-worker-count 1 --chunking-strategy none \
  "${common_flags[@]}"
emit jfk "$words_filter" jfk_tiny_words_golden.json \
  --arg options "task=transcribe language=nil temperature=0 temperatureFallbackCount=0 usePrefillPrompt=true skipSpecialTokens=true withoutTimestamps=false wordTimestamps=true concurrentWorkerCount=1 chunkingStrategy=none; ModelComputeOptions(audioEncoderCompute: .cpuAndNeuralEngine, textDecoderCompute: .cpuAndNeuralEngine)"

echo
echo "Done. Now READ THE DIFF before committing:"
echo "  git diff -- coremlit/tests/whisper/fixtures/golden/"
echo
echo "A change in tokens or text is the oracle saying something different than it"
echo "did before. Explain it — a new host-class, a new whisperkit-cli, a new model"
echo "revision — do not commit it as routine."
