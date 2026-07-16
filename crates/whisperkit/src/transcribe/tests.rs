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
fn zero_window_run_observes_no_language_in_provenance() {
  // F3 (codex round 2), end to end. The same zero-window run as above:
  // audio shorter than the padding threshold decodes NO window
  // (encode_calls == 0), so the pipeline observes no language. The result
  // still carries the Swift-compat `"en"` DISPLAY fallback, but the recorded
  // observation -- and therefore `Provenance::for_result` -- must be `None`,
  // not a fabricated language the pipeline never saw.
  let t = tiny_tokenizer();
  let mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let task = TranscribeTask::new(&mock, &t);
  let result = task
    .run(&vec![0.1; 14_400], &DecodingOptions::new())
    .unwrap();

  assert_eq!(mock.counters().encode_calls(), 0, "no window decoded");
  assert_eq!(result.language(), "en", "the display fallback is kept");
  assert_eq!(
    result.task_facts().observed_language(),
    None,
    "nothing was observed, so the result records no detected language"
  );

  // F3 (codex round 8). A zero-window run POSITIVELY knows it drew nothing and
  // was truncated by nothing: no window decoded, so no attempt could draw or be
  // stopped. Seeding the fact sink OBSERVED-CLEAN (not `unknown()`) records the
  // honest `Some(false)`/`Some(false)`, so the run is reproducible -- where the
  // pre-fix `unknown()` seed left it conservatively non-reproducible.
  //
  // Mutation proof: revert the sink seed to `TaskFacts::unknown()` and both
  // booleans read back `None`, failing `is_reproducible()` below.
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(false),
    "a run that decoded no window POSITIVELY drew nothing"
  );
  assert_eq!(
    result.task_facts().early_stopped(),
    Some(false),
    "and was truncated by nothing"
  );

  let provenance = crate::provenance::Provenance::for_result(
    &DecodingOptions::new(),
    &crate::options::ComputeOptions::new(),
    &result,
  );
  assert_eq!(
    provenance.task_facts().observed_language(),
    None,
    "and neither does the provenance -- absent, not fabricated"
  );
  assert!(
    provenance.is_reproducible(),
    "an honest zero-window run reproduces byte-for-byte -- it did nothing to redo"
  );
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
fn genuine_observation_survives_a_later_failed_probe() {
  // F2 (codex round 4), opposite direction of the zero-iteration bug. Window
  // 1's probe genuinely detects Spanish; window 2's probe FAILS. A failed
  // probe witnessed nothing, so it must NOT erase the earlier genuine
  // observation -- first-genuine-observation wins. Pre-fix the failed probe
  // cleared `observed_language` to `None`, and with the finalize fix in place
  // window 2's forced default `<|en|>` no longer re-observes either, so the
  // whole run reported NO detection even though window 1 plainly detected
  // "es".
  let t = tiny_tokenizer();
  let s = special();
  let es = t.token_to_id("<|es|>").unwrap();
  let hello = t.encode(" Hello").unwrap()[0];
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  // script[0] = <|es|>: the probe's language filter picks it as the detected
  // language. The main decode's prompt is then re-derived with a
  // `<|transcribe|>` task token, so `TimestampRulesFilter` masks language
  // tokens at the sampling position -- the decode itself predicts NO language
  // token, making the probe the sole observation. The ts(50) pair ends each
  // window and advances seek 1 s, giving two windows over 3 s of audio.
  mock.push_token_steps(&[
    es,
    s.transcribe_token(),
    ts(0),
    hello,
    ts(50),
    ts(50),
    s.end_token(),
  ]);
  // Window 1: probe = call 1 (succeeds, "es"), decode = calls 2..=8.
  // Window 2: probe = call 9 -> scripted to fail; its decode (calls 10..=16)
  // still runs. `fail_on_call` is reset-immune, so only window 2's probe fails.
  mock.fail_on_call(9);
  let options = DecodingOptions::new().with_detect_language();
  let task = TranscribeTask::new(&mock, &t);
  let result = task.run(&vec![0.1; 48_000], &options).unwrap();
  assert_eq!(mock.counters().encode_calls(), 2, "two windows decoded");
  assert_eq!(
    result.task_facts().observed_language(),
    Some("es"),
    "window 1's genuine detection must survive window 2's failed probe"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_predicted_language_is_recorded_over_the_forced_display_language() {
  // F1 (codex round 5). The default multilingual prefill FORCES `<|en|>` into
  // the prompt, and `without_timestamps` (a public config) drops the
  // `TimestampRulesFilter`, so the model can freely PREDICT `<|es|>` at the
  // first free position after it. The Swift-faithful DISPLAY language is the
  // FIRST language token in the whole slice -- the forced `<|en|>` -- while the
  // OBSERVATION is the PREDICTED `<|es|>`.
  //
  // Pre-fix, `decode_with_fallback` reconstructed the observation from the
  // display `result.language()` (the forced `<|en|>`), recording `"en"` for a
  // run that plainly detected `"es"`; the observation now flows from the
  // decode's own `observed_language`, so the display stays `"en"` and the
  // detection is `"es"`.
  let t = tiny_tokenizer();
  let s = special();
  let es = t.token_to_id("<|es|>").unwrap();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  // Prefill [SOT, <|en|>, <|transcribe|>, <|notimestamps|>] forces positions
  // 0..=3; the first FREE prediction is `<|es|>`, then a text token, then EOT.
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.no_timestamps_token(),
    es,
    2425,
    s.end_token(),
  ]);
  // Default `detect_language()` is false (use_prefill_prompt is set), so NO
  // probe runs: the observation can only come from the decode's predicted token.
  let options = DecodingOptions::new().with_without_timestamps();
  assert!(
    !options.detect_language(),
    "no probe: prefill is on by default"
  );
  let task = TranscribeTask::new(&mock, &t);
  let result = task.run(&vec![0.1; 32_000], &options).unwrap();

  assert_eq!(
    result.language(),
    "en",
    "the DISPLAY language is the forced-prefill <|en|>, kept Swift-faithful"
  );
  assert_eq!(
    result.task_facts().observed_language(),
    Some("es"),
    "the DETECTION is the PREDICTED <|es|>, not the forced display <|en|>"
  );
  assert_eq!(
    crate::provenance::Provenance::for_result(
      &options,
      &crate::options::ComputeOptions::new(),
      &result,
    )
    .task_facts()
    .observed_language(),
    Some("es"),
    "provenance records the predicted detection, never the forced display"
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
  // All three chunks survived, so their ordered coordinates concatenate through
  // the fixed schedule merge (round 10, F2) -- a fully-known ordered attribution,
  // NOT the pre-fix first-child-only [0].
  assert_eq!(
    result.task_facts().worker_schedule(),
    Some([0, 1, 2].as_slice()),
    "every chunk survived -- the ordered coordinates concatenate to [0, 1, 2]",
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn worker_schedule_is_unknown_when_a_vad_chunk_errors() {
  // ADJUDICATED (round 10, F2): a VAD chunk that ERRORS and is dropped contributes
  // an UNKNOWN (`None`) schedule, not a missing coordinate -- and under the
  // absorbing-`None` schedule law that taints the whole run's schedule to `None`,
  // distinct from the surviving chunks' hand-selected `[1, 2]`. A run that lost a
  // chunk cannot report a fully-known ordered worker attribution.
  //
  // Same 3-chunk audio as `vad_chunked_transcribe_reanchors_and_merges`, but
  // chunk 0's first decode_step fails (`fail_on_call(1)`), so its whole run errors
  // and the VAD branch drops it; chunks 1 and 2 replay the script from their own
  // resets and survive with known coordinates [1] and [2].
  //
  // Mutation proof: seed the schedule fold from the merged survivors (hand-select
  // the successes) instead of folding the errored chunk's `None`, and this reads
  // back `Some([1, 2])` instead of the adjudicated `None`.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(48_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  mock.fail_on_call(1); // chunk 0's first decode_step errors -> its run is dropped
  let kit = WhisperKit::with_backend(mock, t);
  let mut audio = vec![0.1f32; 96_000];
  audio[32_000..35_200].fill(0.0);
  audio[64_000..67_200].fill(0.0);
  let options = DecodingOptions::new().with_chunking_strategy(ChunkingStrategy::Vad);
  let result = kit.transcribe(&audio, &options).unwrap();
  assert_eq!(
    result.text(),
    "Hello Hello",
    "chunk 0 errored and was dropped; only chunks 1 and 2 survive",
  );
  assert_eq!(
    result.task_facts().worker_schedule(),
    None,
    "an errored chunk taints the ordered schedule to unknown -- NOT the survivors' [1, 2]",
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

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_rejected_attempts_early_stop_survives_the_fallback_selection() {
  // R6-F1 (codex round 6). A callback truncates attempt 0, whose bad partial then
  // CROSSES the logprob threshold and falls back; attempt 1 runs to completion
  // and is accepted WITHOUT stopping. The accepted attempt was not truncated, so
  // the pre-fix accepted-only read (`decoding_result.early_stopped()`) recorded
  // early_stopped=false -- losing the history that a callback stop fired at all.
  // The unified sink OR-s the rejected attempt's stop before the fallback drops
  // it, so the fact survives to the accepted result, and the two runs -- one with
  // the callback, one without -- leave DISTINCT records even though their text is
  // identical: the truncated one is not reproducible from the options alone.
  //
  // Mutation proof: revert `decode_with_fallback` to merge `false` for the early
  // stop (or read only the accepted `DecodingResult::early_stopped`) and the
  // `early_stopped()` assertion below fails, reading back false.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let compute = crate::options::ComputeOptions::new();
  // Same fallback-forcing configuration as the leak test above, but with a ZERO
  // fallback increment so the accepted retry stays greedy (temperature 0.0) and
  // never DRAWS -- isolating the early-stop fact as the only thing that can move
  // the reproducibility answer between the two runs below.
  let options = DecodingOptions::new()
    .with_temperature_fallback_count(1)
    .with_temperature_increment_on_fallback(0.0)
    .maybe_first_token_logprob_threshold(None)
    .maybe_logprob_threshold(Some(-0.1));

  // WITHOUT a callback: attempt 0 runs full, still fails the threshold, falls
  // back, and the greedy attempt 1 is accepted -- an un-truncated, non-sampling,
  // reproducible run.
  let uncut = TranscribeTask::new(&mock, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert_eq!(uncut.text(), "Hello");
  assert_eq!(uncut.task_facts().early_stopped(), Some(false));
  assert_eq!(
    uncut.task_facts().drew_from_rng(),
    Some(false),
    "greedy retry never draws"
  );
  let uncut_prov = crate::provenance::Provenance::for_result(&options, &compute, &uncut);
  assert!(uncut_prov.is_reproducible());

  // WITH a callback that stops attempt 0's third sampled step (call 6, exactly as
  // the leak test): the rejected attempt is truncated, attempt 1 completes.
  let calls = std::sync::Mutex::new(0usize);
  let callback: &(dyn Fn(&crate::result::TranscriptionProgress) -> Option<bool> + Sync) =
    &|_progress| {
      let mut seen = calls.lock().unwrap();
      *seen += 1;
      Some(*seen != 6)
    };
  let truncated = TranscribeTask::new(&mock, &t)
    .with_progress_callback(callback)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert_eq!(
    truncated.text(),
    "Hello",
    "the accepted retry still ran to completion"
  );
  assert_eq!(
    truncated.task_facts().early_stopped(),
    Some(true),
    "the REJECTED attempt's early stop must survive the fallback selection"
  );
  assert_eq!(
    truncated.task_facts().drew_from_rng(),
    Some(false),
    "the greedy retry never draws, so early_stopped is the ONLY differing fact"
  );
  let trunc_prov = crate::provenance::Provenance::for_result(&options, &compute, &truncated);
  assert_eq!(trunc_prov.task_facts().early_stopped(), Some(true));
  assert!(
    !trunc_prov.is_reproducible(),
    "a callback truncation -- even of a rejected attempt -- is not reproducible from options alone"
  );
  assert_ne!(
    uncut_prov, trunc_prov,
    "the surviving early-stop fact distinguishes two runs whose text is identical"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_callback_truncation_is_recorded_and_is_not_reproducible() {
  // F4a (codex round 5). A progress callback returning Some(false) TRUNCATES the
  // transcript -- a caller CONTROL action -- but the callback is a closure the
  // record cannot name. Two runs differing ONLY in that callback must NOT leave
  // byte-identical, both-reproducible provenance: the truncated one records the
  // early-stop OUTCOME (library-known) and is not reproducible from the recorded
  // options+seed alone, while the un-truncated one is (greedy). Pre-fix both
  // recorded nothing of the callback and both claimed `is_reproducible()`.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let options = DecodingOptions::new();
  let compute = crate::options::ComputeOptions::new();

  // WITHOUT a callback: the full greedy transcript, reproducible from options.
  let full = TranscribeTask::new(&mock, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert_eq!(
    full.task_facts().early_stopped(),
    Some(false),
    "no callback truncated the full run"
  );
  let full_prov = crate::provenance::Provenance::for_result(&options, &compute, &full);
  assert!(
    full_prov.is_reproducible(),
    "a greedy, un-truncated run reproduces from options alone"
  );

  // WITH a callback that stops at the first non-prefill step: the decode breaks
  // early, so the transcript is a truncation of the full one.
  let stop: &(dyn Fn(&crate::result::TranscriptionProgress) -> Option<bool> + Sync) =
    &|_progress| Some(false);
  let truncated = TranscribeTask::new(&mock, &t)
    .with_progress_callback(stop)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert_eq!(
    truncated.task_facts().early_stopped(),
    Some(true),
    "the callback's Some(false) truncated this run"
  );
  let trunc_prov = crate::provenance::Provenance::for_result(&options, &compute, &truncated);
  assert_eq!(
    trunc_prov.task_facts().early_stopped(),
    Some(true),
    "provenance records the early-stop outcome"
  );
  assert!(
    !trunc_prov.is_reproducible(),
    "a callback-truncated transcript is not reproducible from the record alone"
  );
  // The two runs differ ONLY in the callback, yet their records are now distinct.
  assert_ne!(
    full_prov, trunc_prov,
    "the early-stop outcome distinguishes two runs that differ only in the callback"
  );

  // F1 (codex round 6 post-consolidation). The same greedy segment that
  // `for_result` — handed the whole transcript, which POSITIVELY observed
  // not-truncated — calls reproducible is NOT reproducible through `for_segment`,
  // which is handed only the segment and structurally cannot see whether a
  // callback truncated the decode. The pre-fix constructor fabricated a
  // `not-truncated` and claimed reproducible — the exact false promise a
  // genuinely callback-truncated segment (as here) would have inherited.
  //
  // Mutation proof: make `TaskFacts::is_reproducible_under` treat an unknown
  // `early_stopped` optimistically (as `not-truncated`) and this assertion fails,
  // reading back reproducible for a record that cannot know it was un-truncated.
  let segment = full
    .segments_slice()
    .first()
    .expect("the full greedy run produced a segment");
  let seg_prov = crate::provenance::Provenance::for_segment(&options, &compute, segment, false);
  assert!(
    !seg_prov.is_reproducible(),
    "for_segment cannot observe the truncation, so it must not promise reproducibility"
  );
  // Supplying the COMPLETE facts — via `for_result` on the un-truncated
  // transcript, which carries the observed not-truncated — is what earns the
  // promise back.
  assert!(
    full_prov.is_reproducible(),
    "the complete facts (for_result on the un-truncated run) do promise it"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn window_id_offset_is_recorded_in_the_result_and_provenance() {
  // F4b (codex round 5), now carried in the task facts' worker schedule. The
  // worker coordinate is the WORKER input to the seeded fallback ladder's
  // sub-seed derivation, so under a seed two runs at different offsets draw
  // different RNG streams and land different transcripts. A single run records it
  // as a one-element schedule `[offset]` (an explicit KNOWN coordinate, never a
  // fabricated 0), or two such runs leave byte-identical records for text that
  // genuinely differs.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let options = DecodingOptions::new();
  let compute = crate::options::ComputeOptions::new();

  let worker0 = TranscribeTask::new(&mock, &t)
    .with_window_id_offset(0)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  let worker3 = TranscribeTask::new(&mock, &t)
    .with_window_id_offset(3)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert_eq!(worker0.task_facts().worker_schedule(), Some([0].as_slice()));
  assert_eq!(worker3.task_facts().worker_schedule(), Some([3].as_slice()));

  let prov0 = crate::provenance::Provenance::for_result(&options, &compute, &worker0);
  let prov3 = crate::provenance::Provenance::for_result(&options, &compute, &worker3);
  assert_eq!(prov0.task_facts().worker_schedule(), Some([0].as_slice()));
  assert_eq!(prov3.task_facts().worker_schedule(), Some([3].as_slice()));
  assert_ne!(
    prov0, prov3,
    "the worker coordinate distinguishes two runs that differ only in it"
  );
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

/// Runs the F1 word-timestamp scenario under the given `drop_blank_audio` and
/// returns the surviving segment ids. Two windows each decode
/// speech/bare-timestamp/speech; the middle segment is a zero-length wordless
/// slice the word-timestamp filter removes AFTER id allocation. Shared by the
/// drop-ON unique-id pin and the drop-OFF Swift-parity duplicate pin so both run
/// byte-identical scripting and differ only in the id-base advance.
fn word_timestamp_removed_segment_ids(drop_blank_audio: bool) -> Vec<usize> {
  let t = tiny_tokenizer();
  let s = special();
  let hello = 2425u32;
  let world = 1002u32;
  // n_audio_ctx 200 -> 200-col alignment rows; each step's peak marches 15
  // frames later so the two real words get distinct, non-zero-length timings.
  let mut mock = MockBackend::new().with_dims(
    ModelDims::new()
      .with_window_samples(16_000)
      .with_n_audio_ctx(200),
  );
  // `without_timestamps` drops the TimestampRulesFilter, so the scripted
  // timestamps are decoded verbatim (with it on, the filter masks a third
  // consecutive timestamp and no bare pair ever forms). Prompt is then
  // [SOT, en, transcribe, no_ts]; the free predictions ts(0) hello ts(25)
  // ts(25) ts(25) world ts(50) ts(50) EOT slice into three segments --
  // [hello 0-0.5s | ts(25) 0.5s (a zero-length wordless bare pair) | world
  // 0.5-1s] -- and `word_timestamps` removes the middle one after id allocation.
  let script = [
    s.english_token(),
    s.transcribe_token(),
    s.no_timestamps_token(), // step 2 (overridden anyway; no_ts is forced)
    ts(0),
    hello,
    ts(25),
    ts(25),
    ts(25),
    world,
    ts(50),
    ts(50),
    s.end_token(),
  ];
  for (step, token) in script.iter().enumerate() {
    let mut row = vec![0.0f32; 200];
    // Peaks kept near the window start so the word-timestamp end stays inside
    // the window and its seek re-anchoring does not overshoot the next window.
    row[step + 1] = 1.0;
    mock.push_step_with_alignment(one_hot(*token), row);
  }
  let task = TranscribeTask::new(&mock, &t);
  // Each window's last timestamp is ts(50) = 1.0 s, so seek advances 16_000
  // samples per window; 48_000 samples of audio therefore run exactly two
  // windows (0..16_000, 16_000..32_000; the third would need seek < 32_000).
  let options = DecodingOptions::new()
    .with_word_timestamps()
    .with_without_timestamps()
    .maybe_drop_blank_audio(drop_blank_audio);
  let result = task.run(&vec![0.1; 48_000], &options).unwrap();
  assert_eq!(mock.counters().encode_calls(), 2, "two windows decoded");
  result
    .segments_slice()
    .iter()
    .map(TranscriptionSegment::id)
    .collect()
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_timestamps_removing_a_segment_keeps_ids_unique_and_monotonic() {
  // F2 (codex round 5) / F1 (codex round 9), the TRUE-path pin. Under
  // `drop_blank_audio` (the default, the deliberate unique-id hardening) the next
  // window ids its segments off the count the decode ALLOCATED (3), not the count
  // that SURVIVED (2) -- otherwise window 2's first survivor renumbers back onto
  // window 1's second survivor (both id 2), leaving the pipeline-local ids
  // non-unique/non-monotonic before the merge ever runs. Window 1 allocates
  // [0, 1, 2], drops the zero-length middle id 1 -> [0, 2], advancing the base by
  // ALL 3; window 2 bases at 3, allocates [3, 4, 5], drops id 4 -> [3, 5].
  //
  // Mutation proof: advance the drop-ON branch by the survivor count and these
  // ids collapse to the drop-OFF [0, 2, 2, 4], failing uniqueness/monotonicity.
  let ids = word_timestamp_removed_segment_ids(true);
  assert_eq!(
    ids,
    vec![0, 2, 3, 5],
    "survivor ids stay unique and monotonic across the removed segment (drop ON)"
  );
  let unique: std::collections::HashSet<usize> = ids.iter().copied().collect();
  assert_eq!(unique.len(), ids.len(), "no id collision across windows");
  assert!(
    ids.windows(2).all(|w| w[0] < w[1]),
    "survivor ids stay strictly monotonic"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_timestamps_drop_cleared_reproduces_swifts_duplicate_ids() {
  // F1 (codex round 9), the FALSE-path Swift oracle. Clearing `drop_blank_audio`
  // restores EXACT Swift parity: `findSeekPointAndSegments(allSegmentsCount:)`
  // bases each window off the running SURVIVOR total Swift appends
  // (`TranscribeTask.swift:181`), filters zero-length (`:217`), and appends only
  // survivors (`:262`). Window 1 allocates [0, 1, 2], drops the middle -> [0, 2]
  // (survivor count 2); window 2 bases at 2, allocates [2, 3, 4], drops id 3 ->
  // [2, 4] -- so id 2 DUPLICATES, byte-for-byte as Swift's do. The pre-fix code
  // advanced by the allocated count regardless of the option, yielding the
  // unique-id [0, 2, 3, 5] here and violating the exact-parity contract.
  //
  // Mutation proof: advance the drop-OFF branch by the allocated count and these
  // ids become the drop-ON [0, 2, 3, 5], failing the duplicate-id assertion.
  let ids = word_timestamp_removed_segment_ids(false);
  assert_eq!(
    ids,
    vec![0, 2, 2, 4],
    "clearing drop_blank_audio reproduces Swift's duplicate survivor ids (drop OFF)"
  );
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
fn blank_audio_drop_keeps_surrounding_speech_and_preserves_ids() {
  // A blank stretch BETWEEN speech: the default drops only the blank
  // segment, the speech on either side survives untouched, and the marker
  // leaves the aggregate text with it.
  //
  // The survivors KEEP THEIR ORIGINAL IDS -- [0, 2], with the gap where
  // the blank segment 1 was. A drop removes, it does not relabel: an id is
  // an ordinal decode position (`all_segments_count + segments.len()`),
  // not an index into this vec, and nothing in the crate looks a segment
  // up by one. Renumbering to a dense [0, 1] would make id 1 mean "the
  // blank" with the filter off and "World" with it on, so a consumer
  // diffing the two settings could not correlate them; the gap, by
  // contrast, is self-describing.
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
    vec![0, 2],
    "survivors keep their decoded ids; the dropped segment leaves a gap"
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
// drop_blank_audio x the merge join (issue #14)
// ---------------------------------------------------------------------
//
// The join rule itself lives on `merge_transcription_results_with_options`
// and is pinned against the public merge in `result::tests` (all four
// placements: interior / trailing / leading / wholly-empty, plus the
// `false` twin and the timing-sum invariance). The tests here pin the two
// PIPELINE doors that reach it: `transcribe`'s VAD branch, and a hand-folded
// `transcribe_all` batch.

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

/// `transcribe_all` over three clips whose middle one comes back EMPTY, and
/// the batch hand-folded through the public merge — i.e. the composition a
/// consumer batching audio actually writes, and the second public door onto
/// the bare-separator defect. Both `transcribe_all` and the merge are
/// `pub`, so gating the repair on `transcribe`'s VAD branch alone left this
/// path broken under the DEFAULT options; the fix belongs in the merge, and
/// this is the test that says so.
///
/// The middle clip is 14 400 samples — under `window_clip_time` (1.0 s /
/// 16 000 samples), so `TranscribeTask::run` executes **zero** windows and
/// returns a no-segment, empty-text result (the drop-INDEPENDENT empty
/// pinned by `audio_shorter_than_window_clip_time_yields_no_windows`). That
/// is deliberate: it makes the pair below assert the exact semantics the
/// merge promises — with `drop_blank_audio` set, *any* empty text is kept
/// out of the join, whatever emptied it.
fn short_clip_batch(
  kit: &WhisperKit<MockBackend>,
  options: &DecodingOptions,
) -> TranscriptionResult {
  let speech = vec![0.1f32; 32_000];
  let too_short = vec![0.1f32; 14_400];
  let results = kit.transcribe_all(&[&speech, &too_short, &speech], options);
  let results: Vec<TranscriptionResult> = results.into_iter().map(Result::unwrap).collect();
  assert_eq!(results[1].text(), "", "the middle clip runs no window");
  merge_transcription_results_with_options(&results, options)
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn transcribe_all_batch_merged_by_hand_has_no_bare_separators() {
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let kit = WhisperKit::with_backend(mock, t);

  let merged = short_clip_batch(&kit, &DecodingOptions::new());
  assert_eq!(
    merged.text(),
    "Hello Hello",
    "an empty result must not become a doubled space, got {:?}",
    merged.text()
  );
  // Merged, not skipped: the empty clip contributed no text and every clip
  // still contributed its timings.
  assert_eq!(merged.timings().total_audio_processing_runs(), 2.0);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn transcribe_all_batch_keeps_the_bare_separator_when_the_drop_is_cleared() {
  // MUTATION EVIDENCE + the Swift-parity pin for the batch door: cleared,
  // the identical batch joins the empty text as the bare separator Swift
  // gives it. The repair is INERT under `false` — which is exactly why it
  // could not be an unconditional filter inside the merge.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let hello = t.encode(" Hello").unwrap()[0];
  script_clean_window(&mut mock, hello);
  let kit = WhisperKit::with_backend(mock, t);

  let options = DecodingOptions::new().maybe_drop_blank_audio(false);
  let merged = short_clip_batch(&kit, &options);
  assert_eq!(
    merged.text(),
    "Hello  Hello",
    "the bare separator must survive the cleared drop (Swift parity)"
  );
  assert_eq!(merged.timings().total_audio_processing_runs(), 2.0);
}

// ---------------------------------------------------------------------
// The unseeded-sampling invariant (coremlit issue #14, codex round 1 /
// finding 2)
// ---------------------------------------------------------------------
//
// `Provenance::is_reproducible` used to reconstruct the decode's effective
// temperature from the SURVIVING segments. Every filter between a decoded
// window and a surviving segment is lossy, so a window that sampled from an
// unseeded RNG could be erased entirely and leave a transcript that looked
// greedy — and therefore reproducible — when it was not.
//
// The fix accumulates the fact in `TranscribeTask::run`, at the one point
// where the accepted temperature is still knowable and ahead of every filter
// that could erase it. These tests drive the real pipeline into each of those
// filters and pin that the fact survives.

/// Options that make the blank window (and ONLY the blank window) fall back,
/// landing it on exactly temperature `0.2`.
///
/// The scripted speech window's decode compresses to 0.667 and the blank
/// window's to 0.952, so a `compression_ratio_threshold` of 0.8 sits between
/// them: the blank window trips the ladder on every attempt and the speech
/// window never does — under ONE shared `DecodingOptions`, exactly as
/// `WhisperKit::transcribe`'s VAD branch decodes all of its chunks. With
/// `temperature_fallback_count == 1` the ladder's last rung is
/// `0.0 + 1 * 0.2`, so the blank window is accepted at 0.2 and its segment
/// carries that temperature.
///
/// Nothing here touches the sampler's determinism: the mock's logits are
/// one-hot at 10.0, so even at 0.2 (`1/t == 5`, giving the target token a
/// softmax mass of `1 - 5e-11` inside the top-k) the draw lands on the
/// scripted token. The decode is stochastic in KIND — it consults the RNG —
/// which is the entire point; it is simply peaked enough to script.
fn blank_falls_back_to_point_two() -> DecodingOptions {
  DecodingOptions::new()
    .with_compression_ratio_threshold(0.8)
    .with_temperature_fallback_count(1)
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_window_accepted_above_zero_can_decode_the_blank_marker_and_be_dropped() {
  // THE REACHABILITY PROOF, and the exact question the review posed: can a
  // window accepted at temperature > 0 decode to exactly the blank marker
  // and then be dropped? Yes — constructed here, not argued.
  //
  // First, with the drop CLEARED, observe what the window actually decodes:
  // `[BLANK_AUDIO]`, at temperature 0.2, from an unseeded sampler.
  let t = tiny_tokenizer();
  let options = blank_falls_back_to_point_two();
  assert_eq!(options.seed(), None, "unseeded is the default");

  let mut observed_mock =
    MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_blank_audio_window(&mut observed_mock, &t);
  let observed = TranscribeTask::new(&observed_mock, &t)
    .run(
      &vec![0.1; 32_000],
      &options.clone().maybe_drop_blank_audio(false),
    )
    .unwrap();
  assert_eq!(observed.text(), crate::constants::BLANK_AUDIO_MARKER);
  assert_eq!(
    observed.segments_slice()[0].temperature(),
    0.2,
    "the ladder must have climbed, or this proves nothing"
  );

  // Now the same window under the DEFAULT drop: the segment is deleted, and
  // with it every trace of the temperature it was decoded at. The result has
  // nothing left to read a temperature off — and yet the decode really did
  // draw from an unseeded RNG, so the transcript is NOT reproducible.
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_blank_audio_window(&mut mock, &t);
  let result = TranscribeTask::new(&mock, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();

  assert!(result.segments_slice().is_empty(), "the drop emptied it");
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(true),
    "the sampling must survive the segment that carried it"
  );

  let provenance = crate::provenance::Provenance::for_result(
    &options,
    &crate::options::ComputeOptions::new(),
    &result,
  );
  assert!(
    !provenance.is_reproducible(),
    "an unseeded sampled window was dropped: this transcript cannot be \
     promised byte-for-byte"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_rejected_nonzero_attempt_is_recorded_even_when_the_window_is_accepted_greedily() {
  // F2 (codex round 3), history 1. base temperature -0.2, increment 0.2,
  // fallback 1: attempt 0 runs at -0.2 and DRAWS from the RNG, is rejected on
  // its first-token logprob, and attempt 1 runs at exactly 0.0 (greedy) and is
  // accepted as the ladder's last rung. The accepted temperature is 0.0, so
  // the OLD temperature-inferred fact called this greedy and reproducible --
  // but attempt 0's unseeded draw already happened, and a re-run may keep its
  // output, so the transcript is NOT reproducible.
  //
  // `without_timestamps` empties the filter chain (no token is masked to
  // -inf), so the -0.2 attempt samples over finite logits without tripping the
  // negative-temperature/masked-token corner. Flat logits make the first-token
  // logprob ~ln(1/V) at EITHER temperature, so both attempts break at their
  // first token deterministically -- attempt 0 rejected, attempt 1 accepted as
  // the last rung -- with no probabilistic assertion.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  for _ in 0..24 {
    mock.push_step(vec![0.0f32; 51865]);
  }
  let options = DecodingOptions::new()
    .with_language("en") // no probe: keep the RNG accounting to the fallback ladder
    .with_without_timestamps()
    .with_temperature(-0.2)
    .with_temperature_fallback_count(1) // attempts: -0.2, then exactly 0.0
    .with_first_token_logprob_threshold(-1.5)
    .maybe_compression_ratio_threshold(None)
    .maybe_logprob_threshold(None);
  assert_eq!(options.seed(), None, "unseeded is the default");

  // 32_000 samples clears the 1 s (16_000-sample) window padding, so exactly
  // one window actually decodes (a shorter clip would be skipped outright).
  let result = TranscribeTask::new(&mock, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert!(
    mock.counters().encode_calls() > 0,
    "a window must actually decode, or this proves nothing"
  );
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(true),
    "attempt 0 drew at -0.2, even though the window was accepted greedily at 0.0"
  );
  let provenance = crate::provenance::Provenance::for_result(
    &options,
    &crate::options::ComputeOptions::new(),
    &result,
  );
  assert!(
    !provenance.is_reproducible(),
    "an unseeded rejected draw cannot be promised byte-for-byte"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_nonzero_temperature_that_never_samples_records_no_sampling() {
  // F2 (codex round 3), the inverse history. A non-zero temperature with
  // `sample_length == 0` runs ZERO decode iterations, so the sampler is never
  // consulted and no RNG draw happens. The OLD fact inferred sampling from the
  // non-zero temperature and wrongly claimed the transcript was sampled; the
  // drew-from-rng fact records the truth -- nothing was drawn.
  let t = tiny_tokenizer();
  let mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  let options = DecodingOptions::new()
    .with_language("en") // no probe (a probe would itself draw)
    .with_without_timestamps() // lump-branch seek advance, so the window terminates cleanly
    .with_temperature(0.3) // non-zero...
    .with_sample_length(0) // ...but zero iterations, so `sample` is never called
    .with_temperature_fallback_count(0);

  // 32_000 samples clears the window padding, so the window really decodes
  // (with zero sampling iterations) rather than being skipped -- otherwise the
  // old temperature-inferred fact would never have been consulted either, and
  // the test would pass vacuously.
  let result = TranscribeTask::new(&mock, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert!(
    mock.counters().encode_calls() > 0,
    "the window must actually decode (at temperature 0.3) for the fact to matter"
  );
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(false),
    "a zero-iteration decode never draws, whatever the temperature -- the old \
     `temperature != 0.0` inference wrongly recorded this window as sampled"
  );

  // F3 (codex round 4). The lump-branch segment DID land at 0.3: segment
  // discovery copies the accepted rung into the segment even though ZERO
  // sampling iterations ran. This is the exact history the old
  // `for_result` OR-ed `segment.temperature() != 0.0` on, reporting a false
  // "sampled" and a false non-reproducible. `for_result` must now read only
  // the carried draw fact -- no draw, so no sampling and reproducible, despite
  // the 0.3 segment sitting right there.
  assert_eq!(
    result.segments_slice().len(),
    1,
    "the zero-iteration window still lumps into one segment"
  );
  assert_eq!(
    result.segments_slice()[0].temperature(),
    0.3,
    "the segment carries the accepted 0.3 rung, though nothing was drawn"
  );
  let provenance = crate::provenance::Provenance::for_result(
    &options,
    &crate::options::ComputeOptions::new(),
    &result,
  );
  assert_eq!(
    provenance.task_facts().drew_from_rng(),
    Some(false),
    "the 0.3 segment must not be read as a draw"
  );
  assert!(
    provenance.is_reproducible(),
    "a zero-iteration decode drew nothing, so it is reproducible despite the 0.3 segment"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn unseeded_sampling_survives_the_blank_audio_drop() {
  // THE FULL FAILING HISTORY, end to end through the merge every VAD-chunked
  // `WhisperKit::transcribe` runs:
  //
  //   1. chunk A decodes speech greedily at 0.0 and survives;
  //   2. chunk B falls back, is accepted at 0.2, and samples exactly
  //      `[BLANK_AUDIO]` from an unseeded RNG;
  //   3. the default blank-drop deletes chunk B's only segment;
  //   4. the merged transcript is chunk A's "Hello" — every surviving
  //      segment reads 0.0.
  //
  // BEFORE the fix, `for_result` inferred the effective temperature from
  // those survivors, saw only 0.0, and answered `is_reproducible() == true`.
  // A re-run redraws chunk B's unseeded sample, and text that is not the
  // marker SURVIVES the drop and changes the transcript — so that guarantee
  // was false.
  let t = tiny_tokenizer();
  let options = blank_falls_back_to_point_two();

  let mut speech = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_clean_window(&mut speech, t.encode(" Hello").unwrap()[0]);
  let mut blank = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_blank_audio_window(&mut blank, &t);

  let chunk_a = TranscribeTask::new(&speech, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  let chunk_b = TranscribeTask::new(&blank, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();

  assert_eq!(
    chunk_a.segments_slice()[0].temperature(),
    0.0,
    "A is greedy"
  );
  assert_eq!(chunk_a.task_facts().drew_from_rng(), Some(false));
  assert!(chunk_b.segments_slice().is_empty(), "B was emptied");
  assert_eq!(
    chunk_b.task_facts().drew_from_rng(),
    Some(true),
    "B sampled, and said so"
  );

  let merged =
    crate::result::merge_transcription_results_with_options(&[chunk_a, chunk_b], &options);
  assert_eq!(merged.text(), "Hello");
  assert!(
    merged
      .segments_slice()
      .iter()
      .all(|segment| segment.temperature() == 0.0),
    "every SURVIVING segment is greedy — which is exactly why inferring the \
     answer from them was wrong"
  );

  // The merge OR-ed the fact out of the chunk whose segments are gone.
  assert_eq!(merged.task_facts().drew_from_rng(), Some(true));

  let compute = crate::options::ComputeOptions::new();
  let provenance = crate::provenance::Provenance::for_result(&options, &compute, &merged);
  assert_eq!(
    provenance.effective_temperature(),
    Some(0.0),
    "the surviving segments really do all say 0.0 — the fix must NOT come \
     from changing this"
  );
  assert!(
    !provenance.is_reproducible(),
    "REGRESSION: an unseeded sampled window was filtered out, and the record \
     went back to promising byte-reproducibility it cannot honor"
  );

  // The seeded twin: the very same history, replayable, so the promise is
  // real this time.
  let seeded = options.clone().with_seed(7);
  let mut speech = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_clean_window(&mut speech, t.encode(" Hello").unwrap()[0]);
  let mut blank = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_blank_audio_window(&mut blank, &t);
  let merged_seeded = crate::result::merge_transcription_results_with_options(
    &[
      TranscribeTask::new(&speech, &t)
        .run(&vec![0.1; 32_000], &seeded)
        .unwrap(),
      TranscribeTask::new(&blank, &t)
        .run(&vec![0.1; 32_000], &seeded)
        .unwrap(),
    ],
    &seeded,
  );
  assert_eq!(merged_seeded.task_facts().drew_from_rng(), Some(true));
  assert!(
    crate::provenance::Provenance::for_result(&seeded, &compute, &merged_seeded).is_reproducible(),
    "a seed makes the same sampled window replayable"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn unseeded_sampling_survives_a_no_speech_window_with_no_segments() {
  // SIBLING (`transcribe/mod.rs`'s no-speech `continue`): a window can
  // produce no segments at all, without any filter running, and the same
  // reasoning applies — the temperature it decoded at is nowhere in the
  // output.
  //
  // A no-speech window never falls back (`needs_fallback` short-circuits on
  // the same comparison that skips it), so the way to reach this state with
  // sampling is a non-zero BASE temperature: the very first attempt draws
  // from the sampler and is accepted as-is.
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_clean_window(&mut mock, t.encode(" Hello").unwrap()[0]);
  // `no_speech_threshold(-0.1)` makes every window read as silent, and
  // clearing `logprob_threshold` stops the skip being un-set again by a
  // healthy average log probability (`segment::find_seek_point_and_segments`
  // :96-99) — the same pairing `silence_skipped_window_with_word_timestamps_
  // surfaces_a_segment_error` above needs, for the same reason.
  let options = DecodingOptions::new()
    .with_temperature(0.5)
    .with_no_speech_threshold(-0.1)
    .maybe_logprob_threshold(None);
  assert_eq!(options.seed(), None);

  let result = TranscribeTask::new(&mock, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();

  assert!(
    result.segments_slice().is_empty(),
    "the no-speech window contributed nothing"
  );
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(true),
    "it still sampled at 0.5, and the record has to know"
  );
  assert!(
    !crate::provenance::Provenance::for_result(
      &options,
      &crate::options::ComputeOptions::new(),
      &result,
    )
    .is_reproducible()
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_greedy_run_stays_reproducible_through_the_drop() {
  // The other direction, so the fix is not just "always say no": an
  // ALL-GREEDY run whose every segment the blank-drop deleted is still
  // perfectly reproducible — nothing ever drew from the sampler. (The old
  // inference-from-survivors rule could not tell this apart from the case
  // above: both leave zero segments, and it guessed `false` for both.)
  let t = tiny_tokenizer();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(16_000));
  script_blank_audio_window(&mut mock, &t);
  let options = DecodingOptions::new();

  let result = TranscribeTask::new(&mock, &t)
    .run(&vec![0.1; 32_000], &options)
    .unwrap();
  assert!(result.segments_slice().is_empty());
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(false),
    "greedy throughout: the sampler was never consulted"
  );
  assert!(
    crate::provenance::Provenance::for_result(
      &options,
      &crate::options::ComputeOptions::new(),
      &result,
    )
    .is_reproducible(),
    "an empty greedy transcript reproduces exactly"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn unseeded_draw_survives_an_errored_vad_chunk_drop() {
  // F3 (codex round 4), the loss path. A VAD chunk decodes at a non-zero
  // temperature (drawing from an unseeded RNG), then a later decode step
  // errors, so its whole `run` errors and `WhisperKit::transcribe`'s VAD branch
  // DROPS it. The draw fact must still reach the merged transcript:
  // `decode_with_fallback` captures it BEFORE the error can propagate, into a
  // sink the VAD branch owns across chunks. Pre-fix the errored chunk's draw
  // was read only AFTER `?`, so it vanished with the dropped chunk and the
  // empty merged result claimed a byte-reproducibility a re-run (which redraws
  // that chunk's unseeded sample, and may not error) could not honor.
  let t = tiny_tokenizer();
  let s = special();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(32_000));
  // Call 1 (position 0's sample) draws at 0.2; call 2 errors, aborting the
  // chunk after the draw already happened. `fail_on_call` counts decode_step
  // calls across resets, and there is a single chunk here, so call 2 is that
  // chunk's second step.
  mock.push_token_steps(&[2425, 1002, 2425, 1002, s.end_token()]);
  mock.fail_on_call(2);
  let kit = WhisperKit::with_backend(mock, t);
  let options = DecodingOptions::new()
    .with_chunking_strategy(ChunkingStrategy::Vad)
    .with_language("en") // no probe: keep the RNG accounting to the decode
    .with_temperature(0.2);
  assert_eq!(options.seed(), None, "unseeded is the default");

  // 40_000 samples > one 32_000-sample window -> the VAD branch; no silence in
  // the 0.1 (voiced) audio -> a single chunk of 32_000 samples. That chunk
  // clears the 16_000-sample window padding, so it decodes one window, which
  // draws (call 1) then errors (call 2) and is dropped, leaving an empty
  // merged result.
  let result = kit.transcribe(&vec![0.1; 40_000], &options).unwrap();
  assert!(
    result.segments_slice().is_empty(),
    "the errored chunk was dropped, so nothing survives"
  );
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(true),
    "the dropped chunk's unseeded draw must survive into the merged result"
  );
  assert!(
    !crate::provenance::Provenance::for_result(
      &options,
      &crate::options::ComputeOptions::new(),
      &result,
    )
    .is_reproducible(),
    "an unseeded draw happened (in a dropped chunk), so the transcript is not reproducible"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn probe_detection_survives_an_errored_vad_chunk_drop() {
  // F3 (codex round 5), the same loss shape as the RNG-draw sink above. A VAD
  // chunk's language probe genuinely detects Spanish (call 1), then the main
  // decode errors (call 2), so the whole `run` errors and the VAD branch DROPS
  // the chunk. The detection must still reach the merged transcript:
  // `decode_with_fallback` records it into a sink the VAD branch owns across
  // chunks BEFORE the error can propagate. Pre-fix the observation lived only in
  // the dropped chunk's discarded `run`, so the merged result read
  // `detected_language == None` -- violating that field's own contract for a run
  // that plainly observed "es".
  let t = tiny_tokenizer();
  let s = special();
  let es = t.token_to_id("<|es|>").unwrap();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(32_000));
  // Call 1 is the probe: its language-filtered argmax is <|es|>. Call 2 is the
  // main decode's first step, scripted to fail -> the chunk's whole run errors.
  mock.push_token_steps(&[es, 1002, 2425, s.end_token()]);
  mock.fail_on_call(2);
  let kit = WhisperKit::with_backend(mock, t);
  // Empty language + multilingual + detect_language -> the probe runs.
  let options = DecodingOptions::new()
    .with_chunking_strategy(ChunkingStrategy::Vad)
    .with_detect_language();

  // 40_000 samples > one 32_000-sample window -> the VAD branch; one voiced
  // chunk that probes (call 1, detects "es") then errors (call 2) and is dropped.
  let result = kit.transcribe(&vec![0.1; 40_000], &options).unwrap();
  assert!(
    result.segments_slice().is_empty(),
    "the errored chunk was dropped, so nothing survives"
  );
  assert_eq!(
    result.task_facts().observed_language(),
    Some("es"),
    "the dropped chunk's probe detection must survive into the merged result"
  );
  assert_eq!(
    crate::provenance::Provenance::for_result(
      &options,
      &crate::options::ComputeOptions::new(),
      &result,
    )
    .task_facts()
    .observed_language(),
    Some("es"),
    "and provenance records the detection the dropped chunk made"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn predicted_language_survives_an_errored_vad_chunk_drop() {
  // F2 (codex round 6 post-consolidation), the sibling of the probe test above
  // with NO probe. Under auto-detect a language token becomes a genuine
  // OBSERVATION the instant it is SAMPLED into the predicted region — the sampler
  // draw from that same partial attempt already survives an errored drop — but
  // the observation's STRING was only built at successful finalization, so a
  // chunk that PREDICTED `<|es|>` (rather than probing it) then errored on a
  // LATER step lost it: the merged result read `observed_language == None` for a
  // run that plainly predicted "es". The fix recognizes the token at sampling
  // time into a cell the sink reads before the error can propagate.
  //
  // Mutation proof: move the capture back to finalization — disable the
  // recognition-time `observed_language_token.set(..)` in `decode::decode_text`
  // (so the string is only built when finalization succeeds) and the
  // `Some("es")` assertions below fail, reading back `None`.
  let t = tiny_tokenizer();
  let es = t.token_to_id("<|es|>").unwrap();
  let mut mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(32_000));
  // NO probe. The multilingual prefill `[SOT, <|en|>, <|transcribe|>,
  // <|notimestamps|>]` is forced over calls 1-3; call 4 samples the FIRST free
  // prediction, scripted to `<|es|>` (recognized as the observation there); call
  // 5 is the next step, scripted to fail -> the chunk's whole run errors.
  mock.push_token_steps(&[2425, 2425, 2425, es]);
  mock.fail_on_call(5);
  let kit = WhisperKit::with_backend(mock, t);
  // Empty language, detect_language OFF (the prefill-coupled default), and
  // `without_timestamps` so the model freely predicts `<|es|>` at the first free
  // position (no timestamp filter masks it).
  let options = DecodingOptions::new()
    .with_chunking_strategy(ChunkingStrategy::Vad)
    .with_without_timestamps();
  assert!(
    !options.detect_language(),
    "no probe runs, so <|es|> is a PREDICTION"
  );

  // 40_000 samples > one 32_000-sample window -> the VAD branch; one voiced chunk
  // that predicts "es" (call 4) then errors (call 5) and is dropped.
  let result = kit.transcribe(&vec![0.1; 40_000], &options).unwrap();
  assert!(
    result.segments_slice().is_empty(),
    "the errored chunk was dropped, so nothing survives"
  );
  assert_eq!(
    result.task_facts().observed_language(),
    Some("es"),
    "the dropped chunk's PREDICTED language must survive into the merged result"
  );
  assert_eq!(
    crate::provenance::Provenance::for_result(
      &options,
      &crate::options::ComputeOptions::new(),
      &result,
    )
    .task_facts()
    .observed_language(),
    Some("es"),
    "and provenance records the prediction the dropped chunk made"
  );
}

#[test]
fn recover_vad_run_facts_carries_sink_facts_with_explicit_schedule_and_span() {
  // Round 10 refactor (extends codex round 9's F3). `recover_vad_run_facts` takes
  // the sink's error-fragile facts VERBATIM — draw, early-stop, and the run's
  // FIRST-observed language (the sink accumulated these across every chunk in
  // ingestion order, a dropped chunk's "es" included, so no later survivor can
  // overwrite it) — and sets the worker schedule and decoded span EXPLICITLY.
  // Under the absorbing-None laws (F2/F3) the sink's own None schedule/span can no
  // longer be merged from the survivors without absorbing them, so the caller
  // hands them in: the schedule it folded over all chunks, and the merged
  // surviving result's own span.
  //
  // Mutation proof: swap the two `with_*` calls in `recover_vad_run_facts` for a
  // `merge` of a survivor-facts record and the explicit schedule/span are
  // absorbed to None under the round-10 laws.
  let sink = TaskFacts::observed_clean().with_observed_language(Some("es".into()));
  let facts = recover_vad_run_facts(sink, Some(vec![1]), Some(1));
  assert_eq!(
    facts.observed_language(),
    Some("es"),
    "the sink's earliest ingested language is carried, even from a dropped chunk",
  );
  assert_eq!(facts.drew_from_rng(), Some(false), "and its draw watch");
  assert_eq!(
    facts.early_stopped(),
    Some(false),
    "and its early-stop watch"
  );
  assert_eq!(
    facts.worker_schedule(),
    Some([1].as_slice()),
    "the caller-folded schedule is set explicitly, not absorbed to None",
  );
  assert_eq!(
    facts.decoded_span(),
    Some(1),
    "and the merged surviving result's id span, likewise",
  );
}

#[test]
fn recover_vad_run_facts_keeps_a_zero_chunk_run_clean_and_known_empty() {
  // F4 (codex round 9) + round 10 (F2), the VAD facts assembly in isolation for a
  // genuine zero-chunk run. The sink is observed_clean (Some(false)/Some(false) —
  // the run watched and saw no draw or truncation), the caller folds the schedule
  // to the known-empty Some([]) (zero chunks = zero workers OBSERVED), and the
  // empty merge carries no span (None). The record stays reproducible AND records
  // a KNOWN-empty schedule, distinct from the unknown None a run that cannot see
  // its workers would carry.
  //
  // Mutation proof: pass `None` for the schedule (the pre-round-10 zero-chunk
  // value) and the known-empty assertion below fails.
  let facts = recover_vad_run_facts(TaskFacts::observed_clean(), Some(Vec::new()), None);
  assert_eq!(facts.drew_from_rng(), Some(false));
  assert_eq!(facts.early_stopped(), Some(false));
  let known_empty: &[usize] = &[];
  assert_eq!(
    facts.worker_schedule(),
    Some(known_empty),
    "a zero-chunk run KNOWS zero workers ran -- Some([]), never unknown None",
  );
  assert!(
    facts.is_reproducible_under(false),
    "a zero-chunk run drew nothing and was truncated by nothing -- reproducible",
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn vad_run_with_zero_chunks_is_known_clean_not_unknown() {
  // F4 (codex round 9), end to end. Audio LONGER than the model window takes the
  // VAD branch, but a clip SHORTER than the chunker's 16_000-sample padding yields
  // ZERO chunks (`AudioChunker.swift`'s signed `startIndex < end - padding`
  // guard). No chunk decodes and no chunk errors, so the run KNOWS nothing
  // happened: the shared sink stays observed_clean and, with no survivors, is
  // carried verbatim rather than poisoned to unknown by the empty merge's
  // unknown() (which pre-fix left both booleans None and non-reproducible).
  let t = tiny_tokenizer();
  // window_samples 8_000 < the 16_000 padding, so a 12_000-sample clip is longer
  // than the window (VAD branch taken) yet shorter than the padding (zero chunks).
  let mock = MockBackend::new().with_dims(ModelDims::new().with_window_samples(8_000));
  let kit = WhisperKit::with_backend(mock, t);
  let options = DecodingOptions::new().with_chunking_strategy(ChunkingStrategy::Vad);
  let result = kit.transcribe(&vec![0.1; 12_000], &options).unwrap();

  assert_eq!(
    kit.backend().counters().encode_calls(),
    0,
    "no chunk decoded -- a genuine zero-chunk VAD run",
  );
  assert!(result.segments_slice().is_empty());
  assert_eq!(
    result.task_facts().drew_from_rng(),
    Some(false),
    "the run watched and POSITIVELY drew nothing",
  );
  assert_eq!(
    result.task_facts().early_stopped(),
    Some(false),
    "and was truncated by nothing",
  );
  // The run KNOWS zero workers ran: a known-empty schedule Some([]), the identity
  // of the merge law, NOT the unknown None a run that cannot see its workers
  // carries (round 10, F2). Result AND provenance carry it.
  let known_empty: &[usize] = &[];
  assert_eq!(
    result.task_facts().worker_schedule(),
    Some(known_empty),
    "a zero-chunk VAD run observed zero workers -- Some([]), never unknown None",
  );
  let provenance = crate::provenance::Provenance::for_result(
    &options,
    &crate::options::ComputeOptions::new(),
    &result,
  );
  assert_eq!(
    provenance.task_facts().worker_schedule(),
    Some(known_empty),
    "and provenance carries the known-empty schedule verbatim",
  );
  assert!(
    provenance.is_reproducible(),
    "a zero-chunk run did nothing to redo -- reproducible",
  );
}
