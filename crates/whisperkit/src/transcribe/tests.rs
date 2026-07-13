use std::path::PathBuf;

use super::*;
use crate::{
  audio::vad::VoiceActivityDetector,
  backend::{ModelDims, mock::MockBackend},
  error::SegmentError,
  options::{ChunkingStrategy, DecodingOptions},
  tokenizer::{SpecialTokens, WhisperTokenizer},
};

fn tiny_tokenizer() -> WhisperTokenizer {
  let root = std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  );
  WhisperTokenizer::from_folder(root.join("tokenizers/whisper-tiny")).unwrap()
}

fn special() -> SpecialTokens {
  SpecialTokens::whisper_defaults()
}

fn ts(index: u32) -> u32 {
  special().time_token_begin() + index
}

/// One clean scripted window: prompt predictions + " Hello" + closing
/// timestamps + EOT. Timestamp pair <|0.00|> .. <|2.00|>.
fn script_clean_window(mock: &mut MockBackend, word: u32) {
  let s = special();
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    ts(0),
    word,
    ts(100),
    ts(100),
    s.end_token(),
  ]);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn audio_shorter_than_window_clip_time_yields_no_windows() {
  // Swift-faithful guard: `while seek < seekClipEnd - windowPadding`
  // (TranscribeTask.swift:113-116) never runs for audio shorter than
  // windowClipTime (1 s default) — port with guarded usize subtraction.
  let t = tiny_tokenizer();
  let mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 14_400], &DecodingOptions::new())
    .unwrap();
  assert!(result.segments_slice().is_empty());
  assert_eq!(result.text(), "");
  assert_eq!(mock.counters().encode_calls(), 0);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn single_window_run_produces_segments_and_text() {
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  // 2 s of audio, 1 s mock windows (window_samples 16_000): window at seek 0
  // runs (0 < 32000 - 16000); its <|2.00|> ending advances seek to 32000,
  // which fails the guard -> exactly one window.
  let result = task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();
  assert_eq!(result.text(), "Hello"); // decoded word tokens, trimmed (TranscribeTask.swift:304-305)
  assert_eq!(result.segments_slice().len(), 1);
  let segment = &result.segments_slice()[0];
  assert!((segment.start() - 0.0).abs() < 1e-4);
  assert!((segment.end() - 2.0).abs() < 1e-4);
  assert_eq!(mock.counters().resets(), 1, "state reset after the window");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn windowing_advances_seek_by_last_timestamp_and_decodes_again() {
  let t = tiny_tokenizer();
  // 3 s of audio with 1 s windows: the scripted window ends with a
  // <|1.00|> pair -> seek advances 1 s per window; the state reset between
  // windows rewinds the mock's script cursor (it lives in the state), so
  // one scripted window replays for every window.
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  let s = special();
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    ts(0),
    hello,
    ts(50), // <|1.00|>
    ts(50),
    s.end_token(),
  ]);
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 48_000], &DecodingOptions::new())
    .unwrap();
  // window 0 covers [0, 16000) -> seek 16000; window 1 -> seek 32000;
  // guard: 32000 < 48000 - 16000 is false -> stop.
  assert_eq!(result.segments_slice().len(), 2);
  assert!(
    (result.segments_slice()[1].start() - 1.0).abs() < 1e-4,
    "time offset applied"
  );
  assert_eq!(mock.counters().encode_calls(), 2);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn fallback_ladder_retries_with_rising_temperature_then_accepts() {
  let t = tiny_tokenizer();
  let s = special();
  // A window whose avg logprob is catastrophically low. without_timestamps
  // keeps TimestampRulesFilter out (near-flat logits would otherwise trip
  // its sum-vs-max forcing) and makes the seeker take the lump branch with
  // a FULL segment_size seek advance — guaranteed loop progress.
  // Per-attempt script (prompt [SOT, en, transcribe, <|notimestamps|>]):
  // 3 forced-prompt predictions + 1 sampled step + EOT = 5 steps.
  // The sampled step's logits are flat-with-EOT-pinned-low: every candidate
  // equally (un)likely at lp = ln(1/51864) ~ -10.86, and EOT can never land
  // in the top-k -> deterministic step COUNTS at any temperature.
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  for _ in 0..3 {
    for _ in 0..4 {
      let mut flat = vec![0.0f32; 51865];
      flat[s.end_token() as usize] = -20.0;
      mock.push_step(flat);
    }
    mock.push_token_step(s.end_token());
  }
  // avg logprob over SOT..=EOT: (4 prompt zeros + -10.86 + EOT 0)/6 ~ -1.81
  // < -1.0 default -> LogProbThreshold fallback on EVERY attempt; the
  // ladder exhausts and the final attempt's result stands
  // (TranscribeTask.swift:394-410).
  let options = DecodingOptions::new()
    .with_without_timestamps()
    .maybe_first_token_logprob_threshold(None)
    .maybe_compression_ratio_threshold(None)
    .with_temperature_fallback_count(2); // 3 attempts total
  let task = TranscribeTask::new(&mock, &t);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  let counters = mock.counters();
  // Swift resets on every needsFallback, INCLUDING the exhausted final
  // attempt (TranscribeTask.swift:394-404), plus the per-window reset:
  assert_eq!(
    counters.resets(),
    3 + 1,
    "3 fallback resets + 1 window reset"
  );
  assert_eq!(
    result.segments_slice().len(),
    1,
    "lump branch: one full-window segment"
  );
  assert!(
    (result.segments_slice()[0].temperature() - 0.4).abs() < 1e-3,
    "final attempt at temperature 0.0 + 2 * 0.2"
  );
}

/// Scripts the exact catastrophic-avg-logprob window
/// `fallback_ladder_retries_with_rising_temperature_then_accepts` uses to
/// force the fallback ladder to exhaust all 3 attempts regardless of
/// temperature (see that test's own comments for the avg-logprob
/// derivation). Reused here because it is a PROVEN fallback-forcing
/// script, and because its 4 "flat" (all-zero-but-EOT) steps per attempt
/// are genuine top-k multinomial draws at temperature > 0: the sampled
/// TOKEN is RNG-dependent even though the STEP COUNT is not, which is
/// exactly the observable coremlit issue #9's seed-reproducibility
/// contract needs.
fn script_exhausting_fallback_window(mock: &mut MockBackend, end_token: u32) {
  for _ in 0..3 {
    for _ in 0..4 {
      let mut flat = vec![0.0f32; 51865];
      flat[end_token as usize] = -20.0;
      mock.push_step(flat);
    }
    mock.push_token_step(end_token);
  }
}

/// The fallback-exhausting options recipe, with a nonzero base
/// `temperature` (so even ATTEMPT 0 draws from the RNG — attempt 0 at the
/// crate's default `temperature() == 0.0` would sample by argmax alone,
/// never touching the seed) and `seed` set from the given value.
fn exhausting_fallback_options(seed: Option<u64>) -> DecodingOptions {
  DecodingOptions::new()
    .with_temperature(0.3)
    .with_without_timestamps()
    .maybe_first_token_logprob_threshold(None)
    .maybe_compression_ratio_threshold(None)
    .with_temperature_fallback_count(2) // 3 attempts total
    .maybe_seed(seed)
}

/// Runs the exhausting-fallback scenario end to end against a freshly
/// built [`MockBackend`]/[`TranscribeTask`] pair, so repeated calls are
/// fully independent runs (no shared mutable state whatsoever) — the
/// same shape a real caller re-invoking [`WhisperKit::transcribe`] twice
/// would see.
fn run_exhausting_fallback(t: &WhisperTokenizer, seed: Option<u64>) -> TranscriptionResult {
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_exhausting_fallback_window(&mut mock, special().end_token());
  let task = TranscribeTask::new(&mock, t);
  let audio = vec![0.1f32; 32_000];
  task
    .run(&audio, &exhausting_fallback_options(seed))
    .unwrap()
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn seeded_fallback_ladder_is_bit_reproducible_across_runs() {
  // Closes coremlit issue #9's final open item: with a seed set, the
  // fallback ladder can stay fully enabled AND be bit-reproducible.
  // N (here 4) entirely independent runs (fresh backend/task every time)
  // at the same seed, each forcing the SAME 3-attempt exhausting ladder,
  // must all sample byte-identical tokens.
  let t = tiny_tokenizer();
  let runs: Vec<TranscriptionResult> = (0..4)
    .map(|_| run_exhausting_fallback(&t, Some(7)))
    .collect();
  let reference = &runs[0];
  assert_eq!(reference.segments_slice().len(), 1);
  let reference_tokens = reference.segments_slice()[0].tokens_slice();
  for (index, run) in runs.iter().enumerate().skip(1) {
    assert_eq!(run.segments_slice().len(), 1);
    assert_eq!(
      run.segments_slice()[0].tokens_slice(),
      reference_tokens,
      "run {index} diverged from run 0: same seed, same fallback ladder -> byte-identical sampled tokens"
    );
    assert_eq!(run.text(), reference.text(), "run {index} text diverged");
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn seeded_fallback_ladder_differs_across_seeds() {
  // Proves the seed is actually threaded into the pipeline's sampler
  // construction, not silently ignored: a DIFFERENT base seed over the
  // identical scripted scenario must sample different tokens. (If this
  // ever flakes because two specific seed literals happen to coincide,
  // that is a signal to pick different literals, not to delete the
  // property -- verified empirically for the literals below before
  // trusting them.)
  let t = tiny_tokenizer();
  let a = run_exhausting_fallback(&t, Some(7));
  let b = run_exhausting_fallback(&t, Some(99));
  assert_ne!(
    a.segments_slice()[0].tokens_slice(),
    b.segments_slice()[0].tokens_slice(),
    "different seeds must not sample the same tokens"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn unseeded_fallback_ladder_still_runs_without_threading_a_seed() {
  // `seed = None` (the default): the fallback ladder must still run to
  // completion exactly as it did before this knob existed. Deliberately
  // NOT asserting any determinism/nondeterminism property here (an
  // OS-seeded run could coincidentally repeat) -- only that this code
  // path is exercised successfully. `options.seed()` returning `None`
  // means `decode_with_fallback`'s `if let Some(seed) = ...` body never
  // runs, so no seed is threaded into the sampler at all; that skip is a
  // structural (code-path) guarantee, not something this test can
  // observe from the outside without breaking the "don't assert
  // (non)determinism" rule above.
  let t = tiny_tokenizer();
  let result = run_exhausting_fallback(&t, None);
  assert_eq!(result.segments_slice().len(), 1, "lump branch, as scripted");
  assert!(
    (result.segments_slice()[0].temperature() - (0.3 + 2.0 * 0.2)).abs() < 1e-3,
    "ladder still exhausts to the final (highest-temperature) attempt"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn segment_discovery_callback_fires_per_window() {
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let discovered = std::sync::Mutex::new(Vec::<usize>::new());
  let callback: &(dyn Fn(&[crate::result::TranscriptionSegment]) + Sync) =
    &|segments| discovered.lock().unwrap().push(segments.len());
  let task = TranscribeTask::new(&mock, &t).with_segment_callback(callback);
  task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();
  assert_eq!(discovered.lock().unwrap().as_slice(), &[1]);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn failed_probe_rederives_language_from_that_attempts_decode() {
  // Regression (task-11 review, Important): TranscribeTask.swift:351-352
  // assigns the probe's outcome unconditionally (`try?` yields nil on
  // failure), so a later attempt's FAILED probe must clear the earlier
  // value and re-derive from that attempt's own decode result — never
  // stay sticky at the stale probe language.
  let t = tiny_tokenizer();
  let es = t.token_to_id("<|es|>").unwrap();
  let hello = t.encode(" Hello").unwrap()[0];
  let s = special();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  // Replayed from position 0 after every reset: serves attempt 0's probe
  // (one step) AND, after the probe's internal rewind, the decode itself.
  mock.push_token_steps(&[
    es,
    s.transcribe_token(),
    ts(0),
    hello,
    ts(100),
    ts(100),
    s.end_token(),
  ]);
  // Call ordinals (measured): probe0 = call 1 (succeeds, "es");
  // attempt-0 decode = call 2 only — decode_text's own first-token early
  // stop fires (forced "es" logprob ~-1.21 < -0.5 at t = 0) -> fallback;
  // probe1 = call 3 -> scripted to fail. Attempt 1's decode then rebuilds
  // the prompt with the default language and early-stops the same way,
  // so the ladder exhausts with attempt 1's result.
  mock.fail_on_call(3);

  let options = DecodingOptions::new()
    .with_detect_language()
    .with_temperature_fallback_count(1)
    .maybe_first_token_logprob_threshold(Some(-0.5))
    .maybe_logprob_threshold(None);
  let task = TranscribeTask::new(&mock, &t);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(
    result.language(),
    "en",
    "stale probe language must not survive a failed re-probe"
  );
  // The only enabled fallback trigger is the first-token comparison; it
  // fires on attempt 0 (stored counter = that fallback's 0-based attempt
  // index, TranscribeTask.swift:397) and attempt 1 accepts. The reset
  // count pins the full structure — probe0 + fallback + probe1 + window
  // — so a broken first-token comparison (no fallback, no re-probe)
  // fails here even though 0.0 is also the counter's default.
  assert_eq!(result.timings().total_decoding_fallbacks(), 0.0);
  assert_eq!(mock.counters().resets(), 4);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn detect_language_pinned_deviation_actually_runs_the_probe() {
  // Closes the loop on `DecodingOptions::detect_language`'s pinned
  // deviation (also pinned at the options-getter level by
  // `options::tests::detect_language_pinned_construction_vs_mutation_histories`):
  // an unset `detect_language` with `use_prefill_prompt` mutated to
  // `false` in place resolves `true` on this port (Swift's equivalent
  // history stays `false`). This test proves that resolution is not
  // just a getter-level curiosity — it actually gates whether
  // `TranscribeTask::run` invokes the language-detection probe against
  // the backend, observed via `MockCounters` deltas between two
  // otherwise-identical runs.
  let t = tiny_tokenizer();
  let hello = t.encode(" Hello").unwrap()[0];

  // Shared recipe: prefill OFF (so `initial_prompt == [SOT]` on both
  // sides — one token, no task token anywhere in `current_tokens`,
  // which keeps `TimestampRulesFilter` a no-op for a multilingual model
  // per its own `effective_sample_begin` doc, so the scripted logits
  // reach the sampler unmodified), one fallback attempt
  // (`temperature_fallback_count(0)`), and the same first-token-logprob
  // trick `failed_probe_rederives_language_from_that_attempts_decode`
  // above uses: an unmasked one-hot step samples with logprob ~-1.21
  // against this tiny vocab's softmax denominator, comfortably below
  // -0.5, so `decode_text` breaks after exactly one `decode_step` call.
  // One scripted step serves either run: the probe (if it runs) and the
  // main decode both read script index 0 — the probe's own
  // unconditional reset rewinds the mock's script cursor before the
  // main decode starts (`decode::detect_language`'s own doc).
  let base_options = || {
    DecodingOptions::new()
      .maybe_use_prefill_prompt(false)
      .with_temperature_fallback_count(0)
      .maybe_first_token_logprob_threshold(Some(-0.5))
      .maybe_logprob_threshold(None)
  };

  // Probe disabled: explicit `false` beats the coupling even with
  // prefill off.
  let mut no_probe = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  no_probe.push_token_step(hello);
  let options_no_probe = base_options().maybe_detect_language(false);
  assert!(!options_no_probe.detect_language());
  TranscribeTask::new(&no_probe, &t)
    .run(&vec![0.1; 32_000], &options_no_probe)
    .unwrap();

  // Probe enabled: `detect_language` left unset, resolving `true` via
  // the pinned deviation — Swift's equivalent history
  // (`DecodingOptions()` then `usePrefillPrompt = false`) would stay
  // `false` and never run this probe at all.
  let mut with_probe = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  with_probe.push_token_step(hello);
  let options_with_probe = base_options();
  assert!(options_with_probe.detect_language());
  TranscribeTask::new(&with_probe, &t)
    .run(&vec![0.1; 32_000], &options_with_probe)
    .unwrap();

  // The delta IS the probe: one extra `decode_step` (the probe's own
  // one-shot draw) and one extra `reset_decoder_state`
  // (`decode::detect_language`'s own unconditional reset).
  let base = no_probe.counters();
  let probed = with_probe.counters();
  assert_eq!(
    probed.decode_steps(),
    base.decode_steps() + 1,
    "probe adds exactly one decode_step call"
  );
  assert_eq!(
    probed.resets(),
    base.resets() + 1,
    "probe adds exactly one reset_decoder_state call"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn transcribe_all_preserves_order_across_scoped_threads() {
  let t = tiny_tokenizer();
  // The mock's script cursor lives in each worker's OWN MockDecoderState
  // (Task 2), so concurrent workers replay the same script independently;
  // only the counters are shared. One scripted window serves both audios.
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let kit = WhisperKit::with_backend(mock, t);
  let a = vec![0.1f32; 32_000];
  let b = vec![0.1f32; 32_000];
  let results = kit.transcribe_all(&[&a, &b], &DecodingOptions::new());
  assert_eq!(results.len(), 2);
  for result in results {
    assert_eq!(result.unwrap().text(), "Hello");
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn vad_chunked_transcribe_reanchors_and_merges() {
  // Regression (task-12 review, Important): the VAD branch itself —
  // chunk_all splitting, per-chunk clip clearing, apply_seek_offsets
  // re-anchoring, and merge_transcription_results — had no direct test.
  // Silence frames at 32_000..35_200 and 64_000..67_200 make
  // split_on_middle_of_longest_silence cut at 33_600 and 65_600, giving
  // three chunks (offsets 0 / 33_600 / 65_600), each decoding one
  // scripted "Hello" window.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(48_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let kit = WhisperKit::with_backend(mock, t);
  let mut audio = vec![0.1f32; 96_000];
  audio[32_000..35_200].fill(0.0);
  audio[64_000..67_200].fill(0.0);
  let options = DecodingOptions::new().with_chunking_strategy(ChunkingStrategy::Vad);
  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(result.text(), "Hello Hello Hello", "per-chunk texts joined");
  let segments = result.segments_slice();
  assert_eq!(segments.len(), 3);
  let ids: Vec<usize> = segments.iter().map(|s| s.id()).collect();
  assert_eq!(
    ids,
    vec![0, 1, 2],
    "merge re-ids result_index + segment_index"
  );
  let starts: Vec<f32> = segments.iter().map(|s| s.start()).collect();
  assert!((starts[0] - 0.0).abs() < 1e-3);
  // 33_600 / 16_000 = 2.1 s and 65_600 / 16_000 = 4.1 s: the .1 offsets
  // are the chunk boundaries and CANNOT come from the un-chunked path
  // (whose window seeks land on whole timestamps, 2.0 / 4.0).
  assert!(
    (starts[1] - 2.1).abs() < 1e-3,
    "chunk 2 re-anchored, got {}",
    starts[1]
  );
  assert!(
    (starts[2] - 4.1).abs() < 1e-3,
    "chunk 3 re-anchored, got {}",
    starts[2]
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn vad_detector_swap_changes_chunk_boundaries() {
  // coremlit issue #9 ("Make VAD strategy pluggable or configurable
  // rather than locking product behavior to the default energy VAD"):
  // reuses `vad_chunked_transcribe_reanchors_and_merges`'s exact audio
  // (two silent stretches inside 96_000 samples, 48_000-sample windows)
  // but swaps in a detector that never reports silence. With no silence
  // to split on, `split_on_middle_of_longest_silence` falls through to
  // its `None => end` arm every time instead of cutting at a silence
  // midpoint, so chunking lands on whole-window boundaries (2 chunks of
  // 48_000 samples each, second chunk starting at exactly 3.0 s) instead
  // of the default `EnergyVad`'s silence-cut boundaries (3 chunks,
  // starting at 0 / 2.1 / 4.1 s, pinned above) -- an observable chunking
  // difference driven purely by which detector is plugged in.
  //
  // Unit struct: no fields, so it is trivially `Send + Sync + 'static`
  // (auto traits with nothing in the type to violate them) -- it
  // satisfies `set_vad_detector`'s actual bounds, and compiling below is
  // the positive-case proof of that, complementing that method's
  // `compile_fail` doctest, which rejects a detector holding non-`Send`
  // state.
  struct AlwaysActiveVad;
  impl VoiceActivityDetector for AlwaysActiveVad {
    fn voice_activity(&self, samples: &[f32]) -> Vec<bool> {
      vec![true; samples.len().div_ceil(self.frame_length_samples())]
    }
    fn frame_length_samples(&self) -> usize {
      crate::audio::vad::DEFAULT_FRAME_LENGTH_SAMPLES
    }
  }

  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(48_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let kit = WhisperKit::with_backend(mock, t).with_vad_detector(Box::new(AlwaysActiveVad));
  let mut audio = vec![0.1f32; 96_000];
  audio[32_000..35_200].fill(0.0);
  audio[64_000..67_200].fill(0.0);
  let options = DecodingOptions::new().with_chunking_strategy(ChunkingStrategy::Vad);
  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(result.text(), "Hello Hello", "2 whole-window chunks, not 3");
  let segments = result.segments_slice();
  assert_eq!(segments.len(), 2);
  assert!((segments[0].start() - 0.0).abs() < 1e-3);
  assert!(
    (segments[1].start() - 3.0).abs() < 1e-3,
    "chunk 2 starts at the whole-window boundary (48_000 / 16_000), got {}",
    segments[1].start()
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn zero_max_window_seek_terminates() {
  // Regression (phase-gate round 3): `Some(0)` reached the seek clamp as
  // `min(seek, previous_seek + 0)`, re-decoding the same window forever.
  // The cap now floors at one sample of progress: audio 3 samples past
  // the window guard yields exactly 3 one-sample windows, then stops.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new().with_max_window_seek(0);
  let result = task.run(&vec![0.1; 16_003], &options).unwrap();
  assert_eq!(
    result.segments_slice().len(),
    3,
    "one window per floored sample"
  );
  // Saturation half: a cap near usize::MAX must not overflow the sum.
  let options = DecodingOptions::new().with_max_window_seek(usize::MAX);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(
    result.segments_slice().len(),
    1,
    "uncapped-in-practice advance"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn out_of_range_clip_end_terminates_at_physical_audio() {
  // Regression (phase-gate round 4, High): a clip end beyond the audio
  // keeps `clip_guard` unsatisfied after `seek` reaches the physical
  // end; `segment_size` hits zero and — `without_timestamps`, where the
  // seek advance IS `segment_size` — the loop re-decoded padded silence
  // forever. The zero-size guard now breaks to the next clip.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  let s = special();
  // without_timestamps prompt: [sot, en, transcribe, no_ts]; forced
  // predictions consume script[0..2], then sampled hello + EOT.
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.no_timestamps_token(),
    hello,
    s.end_token(),
  ]);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new()
    .with_without_timestamps()
    .with_clip_timestamps(vec![0.0, 4.0]); // 64_000 samples of clip over 32_000 of audio
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  // Two real 1-second windows fit the physical audio; the third
  // iteration's zero segment breaks instead of decoding padding forever.
  assert_eq!(result.segments_slice().len(), 2);
  assert_eq!(
    mock.counters().encode_calls(),
    2,
    "no empty-window inference"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn early_stop_does_not_leak_into_fallback_retries() {
  // Regression (phase-gate round 5): the early-stop latch was allocated
  // once per window, so a callback-stopped attempt whose partial result
  // triggered an ordinary fallback handed the still-true flag to the
  // retry, truncating it after one step — Swift initializes a fresh
  // latch per decodeText (TextDecoder.swift:570).
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let calls = std::sync::Mutex::new(0usize);
  let callback: &(dyn Fn(&crate::result::TranscriptionProgress) -> Option<bool> + Sync) =
    &|_progress| {
      let mut seen = calls.lock().unwrap();
      *seen += 1;
      // Call 6 = attempt 0's third sampled step (after hello + both
      // timestamps landed): request the stop. Every other call continues.
      Some(*seen != 6)
    };
  let task = TranscribeTask::new(&mock, &t).with_progress_callback(callback);
  // At t = 0 the window averages ~-0.19 (hello's one-hot logprob -1.21
  // diluted by prefill/EOT zeros and the mass-rule-boosted timestamp
  // logprobs of ~-0.07); at t = 0.2 every draw scores ~0. A -0.1
  // threshold therefore fails attempt 0's stopped partial and accepts
  // the full retry. A leaked stale flag instead breaks the retry after
  // one step. (The first-token threshold stays disabled: it would
  // complete attempt 0 at its very first iteration, before any callback
  // can fire.)
  let options = DecodingOptions::new()
    .with_temperature_fallback_count(1)
    .maybe_first_token_logprob_threshold(None)
    .maybe_logprob_threshold(Some(-0.1));
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(result.text(), "Hello", "the retry ran to completion");
  assert_eq!(
    mock.counters().decode_steps(),
    13,
    "6 stopped steps + 7 full-retry steps"
  );
  assert_eq!(result.timings().total_decoding_fallbacks(), 0.0);
}

fn one_hot(token: u32) -> Vec<f32> {
  let mut logits = vec![0.0f32; 51865];
  logits[token as usize] = 10.0;
  logits
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn window_loop_attaches_word_timings_when_enabled() {
  let t = tiny_tokenizer();
  let s = special();
  let hello = t.encode(" Hello").unwrap()[0];
  // Mock with alignment rows: dims n_audio_ctx 100 -> 100-col rows; each
  // step's row peaks 10 frames (0.2 s) later than the last.
  let mut mock = MockBackend::new().with_dims(
    ModelDims::new()
      .with_window_samples(16_000)
      .with_n_audio_ctx(100),
  );
  let script = [
    s.english_token(),
    s.transcribe_token(),
    ts(0),
    hello,
    ts(100),
    ts(100),
    s.end_token(),
  ];
  for (step, token) in script.iter().enumerate() {
    let mut row = vec![0.0f32; 100];
    row[(step + 1) * 10] = 1.0;
    mock.push_step_with_alignment(one_hot(*token), row);
  }
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new().with_word_timestamps();
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(result.segments_slice().len(), 1);
  let words = result.segments_slice()[0].words_slice();
  assert!(!words.is_empty(), "word timings attached");
  let joined: String = words.iter().map(|w| w.word()).collect();
  assert_eq!(crate::text::normalized(&joined), "hello");
  for word in words {
    assert!(word.end() >= word.start());
    assert!(
      (0.0..=2.5).contains(&word.end()),
      "timings inside the window"
    );
  }
  // Segment boundary follows the last word (update heuristics ran).
  assert!((result.segments_slice()[0].end() - words.last().unwrap().end()).abs() < 1e-4);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_timestamps_off_leaves_segments_wordless() {
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();
  assert!(result.segments_slice()[0].words_slice().is_empty());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn silence_skipped_window_with_word_timestamps_surfaces_a_segment_error() {
  // Self-review finding (task-5 review): the brief's insert-code comment
  // describes a silence-skipped window's `None` `current_segments`
  // becoming `Some(vec![])` under word timestamps, mirroring Swift's
  // `currentSegments ?? []`. That is not what the already-shipped
  // `add_word_timestamps` (task 4) does: called with an empty `segments`
  // slice, its prefix-take alignment always ends up zero rows (sized off
  // `segments`' own token count, independent of the real alignment
  // matrix's shape), which `find_alignment`'s DTW guard unconditionally
  // rejects as `SegmentError::InvalidAlignmentShape`. `run` therefore
  // surfaces a `TranscribeError::Segment`, not a silent empty segment
  // list — the faithful analogue of Swift's own unconditional crash on
  // the same input (`SegmentSeeker.swift:208`/`:211`'s unguarded `1...0`
  // range) as a typed, recoverable error instead of a hard process abort.
  //
  // Reached here via an explicit negative `no_speech_threshold`, since
  // `no_speech_prob` is permanently `0.0` (`decode/mod.rs`'s own
  // faithfully-ported upstream TODO) — no positive threshold, including
  // the default, can trigger the silence skip through a real decode.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new()
    .with_word_timestamps()
    .with_no_speech_threshold(-0.1)
    .maybe_logprob_threshold(None);
  let err = task.run(&vec![0.1; 32_000], &options).unwrap_err();
  assert!(
    matches!(
      err,
      TranscribeError::Segment(SegmentError::InvalidAlignmentShape { rows: 0, .. })
    ),
    "got: {err:?}"
  );
}

// ---------------------------------------------------------------------
// drop_blank_audio (coremlit issue #14)
// ---------------------------------------------------------------------

/// The BPE tokens a Whisper model samples to spell out `[BLANK_AUDIO]` for
/// silence — several ordinary text tokens, not one special token (see
/// `constants::BLANK_AUDIO_MARKER`'s doc). Encoded with the leading space
/// the real model emits, which the drop filter's Swift-whitespace trim
/// normalizes away before matching.
fn blank_audio_tokens(t: &WhisperTokenizer) -> Vec<u32> {
  t.encode(&format!(" {}", crate::constants::BLANK_AUDIO_MARKER))
    .unwrap()
}

/// One window decoding to nothing but `[BLANK_AUDIO]` — the scripted
/// analogue of the 5 s-of-silence pipeline golden, one segment spanning
/// <|0.00|>..<|2.00|>.
fn script_blank_audio_window(mock: &mut MockBackend, t: &WhisperTokenizer) {
  let s = special();
  let mut steps = vec![s.english_token(), s.transcribe_token(), ts(0)];
  steps.extend(blank_audio_tokens(t));
  steps.extend([ts(100), ts(100), s.end_token()]);
  mock.push_token_steps(&steps);
}

/// One window decoding to speech / `[BLANK_AUDIO]` / speech as three
/// consecutive-timestamp-delimited segments: <|0.00|>..<|1.00|> " Hello",
/// <|1.00|>..<|2.00|> " [BLANK_AUDIO]", <|2.00|>..<|3.00|> " World".
fn script_speech_blank_speech_window(mock: &mut MockBackend, t: &WhisperTokenizer) {
  let s = special();
  let hello = t.encode(" Hello").unwrap()[0];
  let world = t.encode(" World").unwrap()[0];
  let mut steps = vec![
    s.english_token(),
    s.transcribe_token(),
    ts(0),
    hello,
    ts(50),
    ts(50),
  ];
  steps.extend(blank_audio_tokens(t));
  steps.extend([ts(100), ts(100), world, ts(150), ts(150), s.end_token()]);
  mock.push_token_steps(&steps);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn blank_audio_segment_is_dropped_by_default() {
  // The DEFAULT path (`drop_blank_audio == true`): a window that decodes to
  // nothing but the marker collapses to a genuinely empty result — zero
  // segments AND empty text, not a segment-less transcript still carrying
  // "[BLANK_AUDIO]" in its aggregate text. Deliberately diverges from
  // Swift, which emits the marker (see the drop=false twin below).
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_blank_audio_window(&mut mock, &t);
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();
  assert!(
    result.segments_slice().is_empty(),
    "got: {:?}",
    result.segments_slice()
  );
  assert_eq!(result.text(), "", "got: {:?}", result.text());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn blank_audio_segment_is_emitted_when_drop_is_cleared() {
  // MUTATION EVIDENCE for the test above, and the exact Swift-parity escape
  // hatch: the identical script with `drop_blank_audio == false` keeps the
  // segment and reports the marker verbatim — byte-identical to the
  // behavior that predates the option.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_blank_audio_window(&mut mock, &t);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new().maybe_drop_blank_audio(false);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();
  assert_eq!(result.segments_slice().len(), 1);
  assert_eq!(result.text(), crate::constants::BLANK_AUDIO_MARKER);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn blank_audio_drop_keeps_surrounding_speech_and_renumbers() {
  // A blank stretch BETWEEN speech: the default drops only the blank
  // segment, the speech on either side survives untouched, the marker
  // leaves the aggregate text with it, and the survivors are renumbered to
  // a contiguous 0..N id range (the same contiguity a merged result has).
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_speech_blank_speech_window(&mut mock, &t);
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 32_000], &DecodingOptions::new())
    .unwrap();

  let segments = result.segments_slice();
  assert_eq!(
    segments.len(),
    2,
    "blank dropped, speech kept: {segments:?}"
  );
  assert_eq!(
    segments.iter().map(|s| s.id()).collect::<Vec<_>>(),
    vec![0, 1],
    "survivors renumbered contiguously"
  );
  for segment in segments {
    assert!(
      !segment
        .text()
        .contains(crate::constants::BLANK_AUDIO_MARKER),
      "no surviving segment carries the marker: {segment:?}"
    );
  }
  assert_eq!(result.text(), "Hello World", "got: {:?}", result.text());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn blank_audio_between_speech_is_kept_when_drop_is_cleared() {
  // MUTATION EVIDENCE for the mixed case: the identical script with the
  // filter off keeps all three segments, marker included — proving the two
  // outcomes above are the option's doing and not the script's.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_speech_blank_speech_window(&mut mock, &t);
  let task = TranscribeTask::new(&mock, &t);
  let options = DecodingOptions::new().maybe_drop_blank_audio(false);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();

  assert_eq!(result.segments_slice().len(), 3);
  assert!(
    result.text().contains(crate::constants::BLANK_AUDIO_MARKER),
    "got: {:?}",
    result.text()
  );
}

// ---------------------------------------------------------------------
// drop_blank_audio x the VAD merge join (issue #14 review, C1)
// ---------------------------------------------------------------------

/// A chunk result carrying `text` and no segments — the shape
/// [`TranscribeTask::run`] returns for a chunk the blank-audio drop
/// emptied (and, independently of the drop, for any chunk shorter than
/// `window_clip_time`).
fn chunk_text(text: &str) -> TranscriptionResult {
  TranscriptionResult::new(text, Vec::new(), "en", TranscriptionTimings::new())
}

#[test]
fn emptied_chunks_never_become_bare_separators_in_the_join() {
  // The C1 REGRESSION, at the exact function `transcribe`'s VAD branch
  // substitutes for the merge's own join under `drop_blank_audio`.
  //
  // The bug: the merge joins EVERY result's text with `" "`, so an emptied
  // chunk contributes a bare separator — `["a", "", "b"].join(" ")` is
  // `"a  b"`. All three placements are the same defect, and all three are
  // covered here because the mock's script cursor rewinds on every
  // `reset_decoder_state`, which makes every chunk of a scripted VAD run
  // decode IDENTICALLY: the mixed speech/silence/speech shape is not
  // expressible end-to-end against `MockBackend`, so it is pinned here,
  // against the real joining code, and the wiring is pinned end-to-end by
  // `vad_chunked_blank_audio_does_not_leave_bare_separators` below.

  // Interior: silence BETWEEN two speech runs -> no doubled space.
  assert_eq!(
    join_non_empty_texts(&[
      chunk_text("Hello world."),
      chunk_text(""),
      chunk_text("Goodbye."),
    ]),
    "Hello world. Goodbye."
  );
  // Trailing: silence after the speech -> no trailing space(s).
  assert_eq!(
    join_non_empty_texts(&[chunk_text("Hello world."), chunk_text(""), chunk_text("")]),
    "Hello world."
  );
  // Leading: silence before the speech -> no leading space.
  assert_eq!(
    join_non_empty_texts(&[chunk_text(""), chunk_text("Hello world.")]),
    "Hello world."
  );
  // Wholly silent: nothing at all, not a string of separators.
  assert_eq!(
    join_non_empty_texts(&[chunk_text(""), chunk_text(""), chunk_text("")]),
    ""
  );
  // Speech only: the join is untouched — a single separator per gap.
  assert_eq!(
    join_non_empty_texts(&[chunk_text("Hello"), chunk_text("world.")]),
    "Hello world."
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn vad_chunked_blank_audio_does_not_leave_bare_separators() {
  // END-TO-END wiring proof for C1, through `WhisperKit::transcribe`'s VAD
  // branch. Same audio as `vad_chunked_transcribe_reanchors_and_merges`
  // (three chunks), but every chunk decodes to nothing but the blank
  // marker, so the default drop empties all three. Before the fix the
  // merge joined `["", "", ""]` into TWO BARE SPACES; the transcript of a
  // silent recording must be genuinely empty.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(48_000));
  script_blank_audio_window(&mut mock, &t);
  let kit = WhisperKit::with_backend(mock, t);
  let mut audio = vec![0.1f32; 96_000];
  audio[32_000..35_200].fill(0.0);
  audio[64_000..67_200].fill(0.0);
  let options = DecodingOptions::new().with_chunking_strategy(ChunkingStrategy::Vad);

  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(
    result.text(),
    "",
    "three emptied chunks must not join into bare separators, got {:?}",
    result.text()
  );
  assert!(result.segments_slice().is_empty());
  // The chunks were MERGED, not skipped: dropping an emptied result from
  // the merge input would have taken its timings with it, and
  // `total_audio_processing_runs` sums one per decoded window per chunk.
  assert_eq!(
    result.timings().total_audio_processing_runs(),
    3.0,
    "every chunk's timings must still be in the merged sums"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn vad_chunked_blank_audio_is_joined_verbatim_when_drop_is_cleared() {
  // MUTATION EVIDENCE + the Swift-parity pin: the identical audio and
  // script with `drop_blank_audio == false` keeps all three markers and
  // joins them with the merge's own single separator, byte-for-byte as
  // before this option existed. The C1 repair above must be INERT here —
  // it is gated on `drop_blank_audio` precisely because an empty-text
  // result is reachable WITHOUT the drop (see
  // `audio_shorter_than_window_clip_time_yields_no_windows`), and the
  // merge must keep joining those as Swift does.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(48_000));
  script_blank_audio_window(&mut mock, &t);
  let kit = WhisperKit::with_backend(mock, t);
  let mut audio = vec![0.1f32; 96_000];
  audio[32_000..35_200].fill(0.0);
  audio[64_000..67_200].fill(0.0);
  let options = DecodingOptions::new()
    .with_chunking_strategy(ChunkingStrategy::Vad)
    .maybe_drop_blank_audio(false);

  let result = kit.transcribe(&audio, &options).unwrap();
  let marker = crate::constants::BLANK_AUDIO_MARKER;
  assert_eq!(result.text(), format!("{marker} {marker} {marker}"));
  assert_eq!(result.segments_slice().len(), 3);
}
