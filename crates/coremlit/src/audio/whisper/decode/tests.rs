use std::{
  cell::Cell,
  path::PathBuf,
  sync::{Mutex, atomic::AtomicBool},
};

use super::*;
use crate::audio::whisper::{
  backend::{InferenceBackend, mock::MockBackend},
  decode::sampler::GreedyTokenSampler,
  options::DecodingOptions,
  result::TranscriptionTimings,
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

/// SOT + en + transcribe + <|0.00|>: the default multilingual prefill.
fn default_prompt(s: &SpecialTokens) -> Vec<u32> {
  vec![
    s.start_of_transcript_token(),
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
  ]
}

fn run_mock(
  mock: &MockBackend,
  prompt: &[u32],
  options: &DecodingOptions,
  tokenizer: &WhisperTokenizer,
) -> crate::audio::whisper::result::DecodingResult {
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 16]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut sampler = GreedyTokenSampler::new(options.temperature(), special().end_token(), options);
  let mut timings = TranscriptionTimings::new();
  decode_text(
    mock,
    &encoded,
    &mut state,
    prompt,
    &mut sampler,
    options,
    tokenizer,
    &mut timings,
    &AtomicBool::new(false),
    &Cell::new(None),
    None,
  )
  .unwrap()
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn prefill_tokens_multilingual_default_shape() {
  let t = tiny_tokenizer();
  let s = special();
  let options = DecodingOptions::new();
  assert_eq!(prefill_tokens(&options, &t, true), default_prompt(&s));
  // without_timestamps flips the final token
  let options = DecodingOptions::new().with_without_timestamps();
  assert_eq!(
    prefill_tokens(&options, &t, true).last(),
    Some(&s.no_timestamps_token())
  );
  // monolingual model: no language/task tokens
  let options = DecodingOptions::new();
  assert_eq!(
    prefill_tokens(&options, &t, false),
    vec![s.start_of_transcript_token(), s.time_token_begin()]
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn prefill_prompt_tokens_truncate_and_prepend_previous() {
  // maxPromptLen = 224/2 - 1 = 111; SUFFIX first, THEN specials filtered
  // (TextDecoder.swift:198-201). Prompt [0..=199, EOT]: suffix(111) =
  // [90..=199, EOT] (111 items), filter < specialTokenBegin drops EOT ->
  // 110 word tokens 90..=199.
  let t = tiny_tokenizer();
  let s = special();
  let long_prompt: Vec<u32> = (0..200u32).chain([s.end_token()]).collect();
  let options = DecodingOptions::new().with_prompt_tokens(long_prompt);
  let tokens = prefill_tokens(&options, &t, true);
  assert_eq!(tokens[0], s.start_of_previous_token());
  assert_eq!(tokens[1], 90);
  assert_eq!(tokens[110], 199);
  assert_eq!(tokens[111], s.start_of_transcript_token());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn loop_forces_prompt_then_samples_to_eot() {
  let t = tiny_tokenizer();
  let s = special();
  let prompt = default_prompt(&s);
  // Steps consumed while forcing the 4-token prompt: predictions at
  // positions 0..3 are overridden (except none here), then free-running:
  // hello(2425), world(1002), <|1.00|>(ts 50), EOT.
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    s.english_token(),    // pos 0 prediction: overridden by prompt[1]
    s.transcribe_token(), // pos 1: overridden by prompt[2]
    s.time_token_begin(), // pos 2: overridden by prompt[3]
    2425,                 // pos 3: first sampled token
    1002,
    s.time_token_begin() + 50,
    s.end_token(),
  ]);
  let result = run_mock(&mock, &prompt, &DecodingOptions::new(), &t);
  // Result tokens = SOT..=EOT inclusive (TextDecoder.swift:780-783).
  let expected: Vec<u32> = prompt
    .iter()
    .copied()
    .chain([2425, 1002, s.time_token_begin() + 50, s.end_token()])
    .collect();
  assert_eq!(result.tokens_slice(), expected.as_slice());
  assert!(result.avg_logprob() < 0.0); // one-hot 10.0-vs-0.0 softmax < 1
  assert_eq!(result.temperature(), 0.0);
  // KV consumed exactly positions 0..6 with the forced/sampled inputs.
  // (decode_step calls: prompt forcing feeds tokens[i] at position i.)
  let counters = mock.counters();
  assert_eq!(counters.decode_steps(), 7);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn negative_temperature_decode_with_timestamp_filter_does_not_panic() {
  // F1 (codex round 4, High), at the pipeline level. `TimestampRulesFilter`
  // masks `<|notimestamps|>` (and, in pairs, whole token ranges) with `-inf`
  // on every post-prefill step (`filter/mod.rs`, ports
  // `LogitsFilter.swift:81`). At a NEGATIVE temperature the pre-fix sampler
  // scaled that `-inf` by `1/T < 0` into `+inf`, so the masked index became
  // the scaled max and the stabilized softmax collapsed to NaN, panicking
  // `random_range`. The whole decode loop -- real filter chain, real sampler,
  // negative temperature -- must now run to completion with a finite result.
  let t = tiny_tokenizer();
  let s = special();
  let mut mock = MockBackend::new();
  // Enough one-hot steps that the loop stays inside the script whatever the
  // (unseeded) negative-temperature draws pick; `sample_length` bounds the
  // loop well below the script length, so no EOT is required to terminate.
  mock.push_token_steps(&[100u32; 24]);
  let options = DecodingOptions::new()
    .with_temperature(-0.2)
    .with_sample_length(8);
  // The mask is present from the first post-prefill step regardless of the
  // RNG, so the failure regime is reached deterministically even though the
  // draws themselves are not seeded here.
  let result = run_mock(&mock, &default_prompt(&s), &options, &t);
  assert!(
    result.avg_logprob().is_finite(),
    "a negative-temperature decode over masked logits must finish with a finite avg log-prob"
  );
  assert!(
    (result.temperature() - (-0.2)).abs() < 1e-6,
    "the accepted temperature is the configured negative one"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn language_observed_only_for_a_predicted_language_token() {
  // F2 (codex round 4) / F1 (codex round 5). `observed_language` is `Some(code)`
  // ONLY when the model PREDICTS a `<|lang|>` token at a position at/after the
  // forced prompt, and it carries the PREDICTED code — never the Swift display
  // `language`, which follows the first-in-the-whole-slice rule and so reports a
  // forced prefill `<|en|>`. A CONFIGURED language, the "en" fallback, and a
  // FORCED prefill `<|lang|>` are all inputs/defaults, not detections.
  let t = tiny_tokenizer();
  let s = special();
  let es = t.token_to_id("<|es|>").unwrap();

  // Branch 1 -- CONFIGURED "es": the language is taken from the option, so it
  // is NOT observed no matter what decodes.
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    es,
    s.transcribe_token(),
    s.time_token_begin(),
    2425,
    1002,
    s.time_token_begin() + 50,
    s.end_token(),
  ]);
  let prompt_es = vec![
    s.start_of_transcript_token(),
    es,
    s.transcribe_token(),
    s.time_token_begin(),
  ];
  let configured = run_mock(
    &mock,
    &prompt_es,
    &DecodingOptions::new().with_language("es"),
    &t,
  );
  assert_eq!(configured.language(), "es");
  assert_eq!(
    configured.observed_language(),
    None,
    "a configured language is copied, not detected"
  );

  // Branch 2 -- FORCED prompt token, NOT predicted (F2, codex round 4): the
  // default multilingual prefill FORCES `<|en|>` into the prompt, and the
  // model predicts only text/timestamps after it -- no `<|lang|>` token in the
  // predicted region. The forced token is an INPUT, not a detection, so it
  // must NOT be observed, even though it IS the Swift-faithful display
  // language. (Pre-fix this asserted `true`: the mislabeled targeting test
  // called the forced token "decoded" and scanned the full prompt+output
  // slice, so it could not fail on the bug it existed to catch.)
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    s.english_token(),    // pos 0 prediction: overridden by prompt[1]
    s.transcribe_token(), // overridden by prompt[2]
    s.time_token_begin(), // overridden by prompt[3]
    2425,                 // first PREDICTED token -- text, not a language token
    1002,
    s.time_token_begin() + 50,
    s.end_token(),
  ]);
  let forced = run_mock(&mock, &default_prompt(&s), &DecodingOptions::new(), &t);
  assert_eq!(
    forced.language(),
    "en",
    "forced <|en|> is still the display language"
  );
  assert_eq!(
    forced.observed_language(),
    None,
    "a FORCED prefill <|en|> is an input, not a detection -- never observed"
  );

  // Branch 2c -- FORCED `<|en|>` prefill, but the model PREDICTS `<|es|>` (F1,
  // codex round 5): the divergence the whole finding turns on. `without_timestamps`
  // drops the `TimestampRulesFilter`, so the model freely predicts `<|es|>` at
  // the first free position AFTER the forced `[SOT, <|en|>, <|transcribe|>,
  // <|notimestamps|>]`. The DISPLAY language stays the Swift-faithful FIRST
  // language token in the whole slice -- the forced `<|en|>` -- while the
  // OBSERVATION is the PREDICTED `<|es|>`. Pre-fix, `observed_language` was a
  // mere boolean and the pipeline reconstructed the string from the display
  // `language`, recording `"en"` for a run that plainly detected `"es"`.
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    s.english_token(),       // pos 0: overridden by prompt[1]
    s.transcribe_token(),    // pos 1: overridden by prompt[2]
    s.no_timestamps_token(), // pos 2: overridden by prompt[3]
    es,                      // first PREDICTED token: a language token, after the prompt
    2425,
    s.end_token(),
  ]);
  let forced_en_predicts_es = run_mock(
    &mock,
    &[
      s.start_of_transcript_token(),
      s.english_token(),
      s.transcribe_token(),
      s.no_timestamps_token(),
    ],
    &DecodingOptions::new().with_without_timestamps(),
    &t,
  );
  assert_eq!(
    forced_en_predicts_es.language(),
    "en",
    "the DISPLAY language is the forced-prefill <|en|>, first in the whole slice"
  );
  assert_eq!(
    forced_en_predicts_es.observed_language(),
    Some("es"),
    "the OBSERVATION is the PREDICTED <|es|>, never the forced display <|en|>"
  );

  // Branch 2d -- FORCED `<|en|>` prefill AND a CONFIGURED `language="en"`, but the
  // model STILL predicts `<|es|>` (round 10, F1): the exact failing history the
  // finding turns on. Duplicates branch 2c with `.with_language("en")`, so
  // `options.language()` is NON-empty. Pre-fix the observation gate ALSO required
  // `options.language().is_empty()`, so a configured language SUPPRESSED the
  // genuine prediction and `observed_language` wrongly read `None` for a run that
  // plainly detected `es`. The display language is unchanged (the Swift-faithful
  // forced `<|en|>`), proving the observation is decoupled from the configured
  // input -- an observation is a probe or a PREDICTED token, never the config.
  //
  // Mutation proof: restore the `&& options.language().is_empty()` conjunct and
  // this reads back `None`; branch 2c (no configured language) still passes, so
  // ONLY the configured case catches the bug the conjunct caused.
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    s.english_token(),       // pos 0: overridden by prompt[1]
    s.transcribe_token(),    // pos 1: overridden by prompt[2]
    s.no_timestamps_token(), // pos 2: overridden by prompt[3]
    es,                      // first PREDICTED token: a language token, after the prompt
    2425,
    s.end_token(),
  ]);
  let configured_en_predicts_es = run_mock(
    &mock,
    &[
      s.start_of_transcript_token(),
      s.english_token(),
      s.transcribe_token(),
      s.no_timestamps_token(),
    ],
    &DecodingOptions::new()
      .with_without_timestamps()
      .with_language("en"),
    &t,
  );
  assert_eq!(
    configured_en_predicts_es.language(),
    "en",
    "the DISPLAY language is the configured/forced <|en|>, unchanged by the fix"
  );
  assert_eq!(
    configured_en_predicts_es.observed_language(),
    Some("es"),
    "a configured language must NOT suppress a genuinely PREDICTED <|es|> observation"
  );

  // Branch 2b -- GENUINELY PREDICTED token (F2, codex round 4): a bare `[SOT]`
  // prompt (nothing forced past it) with `without_timestamps` (so no
  // `TimestampRulesFilter` masks the language token at the sampling position),
  // and the model PREDICTS `<|es|>` at the first free position. That prediction
  // sits at/after `initial_prompt_index`, so it IS a genuine observation. This
  // is the case a broken "always false" over-correction would fail on.
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[es, 2425, s.end_token()]);
  let predicted = run_mock(
    &mock,
    &[s.start_of_transcript_token()],
    &DecodingOptions::new().with_without_timestamps(),
    &t,
  );
  assert_eq!(
    predicted.language(),
    "es",
    "the predicted <|es|> is the display language"
  );
  assert_eq!(
    predicted.observed_language(),
    Some("es"),
    "a <|lang|> token PREDICTED after the prompt is a genuine detection"
  );

  // Branch 3 -- FALLBACK: empty configured language and NO <|lang|> token in
  // the decoded tokens, so the language defaults to "en" and is NOT observed.
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    s.transcribe_token(),
    s.time_token_begin(),
    2425,
    1002,
    s.time_token_begin() + 50,
    s.end_token(),
  ]);
  let prompt_no_lang = vec![
    s.start_of_transcript_token(),
    s.transcribe_token(),
    s.time_token_begin(),
  ];
  let fallback = run_mock(&mock, &prompt_no_lang, &DecodingOptions::new(), &t);
  assert_eq!(
    fallback.language(),
    crate::audio::whisper::constants::DEFAULT_LANGUAGE_CODE
  );
  assert_eq!(
    fallback.observed_language(),
    None,
    "the \"en\" fallback is a default, not a detection"
  );

  // Branch 4 -- LOW-FIRST-TOKEN-LOGPROB completion (codex round 11, M1): the model
  // genuinely SAMPLES `<|es|>` at the first free position of a bare `[SOT]` prompt
  // (no prefill, no probe, `temperature_fallback_count` at its default 0), but its
  // logprob falls below the default -1.5 first-token threshold, so the decode
  // COMPLETES on that very first step. The observation must still latch: the
  // recognition now runs BEFORE the completion break, so the sampled `<|es|>`
  // reaches BOTH the finalized `DecodingResult` AND the `observed_language_token`
  // cell the attempt sink carries into the task facts -- rather than being dropped
  // because the threshold broke the loop first. `without_timestamps` frees the
  // language slot (no `TimestampRulesFilter`), and a lone positive logit at `<|es|>`
  // is the argmax at a probability far under `e^-1.5` (`ln P ~= -9.86`).
  //
  // Mutation proof: move the recognition back after the `is_segment_completed`
  // break (its pre-round-11 position, inside the `!is_prefill` push) and both the
  // `DecodingResult` and cell assertions below read back `None`/`None` -- the
  // threshold completion skips the latch entirely.
  let mut mock = MockBackend::new();
  let mut low_confidence_es = vec![0.0_f32; mock.dims().vocab()];
  low_confidence_es[es as usize] = 1.0; // the only positive logit: argmax `<|es|>`, low prob
  mock.push_step(low_confidence_es);
  // `temperature_fallback_count = 0` per the disclosed history -- `decode_text`
  // runs a single decode and never consults it, but modelling it keeps the branch
  // faithful to the reported transcribe-layer scenario.
  let options = DecodingOptions::new()
    .with_without_timestamps()
    .with_temperature_fallback_count(0);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 16]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut sampler = GreedyTokenSampler::new(options.temperature(), s.end_token(), &options);
  let mut timings = TranscriptionTimings::new();
  let observed_cell: Cell<Option<u32>> = Cell::new(None);
  let low_first = decode_text(
    &mock,
    &encoded,
    &mut state,
    &[s.start_of_transcript_token()],
    &mut sampler,
    &options,
    &t,
    &mut timings,
    &AtomicBool::new(false),
    &observed_cell,
    None,
  )
  .unwrap();
  assert!(
    low_first.first_token_log_prob() < -1.5,
    "the sampled <|es|> is below the -1.5 first-token threshold, got {}",
    low_first.first_token_log_prob(),
  );
  assert_eq!(
    mock.counters().decode_steps(),
    1,
    "the below-threshold first token completes the decode on the first step",
  );
  assert_eq!(
    low_first.observed_language(),
    Some("es"),
    "a first token below the threshold still latches its PREDICTED language onto the DecodingResult",
  );
  assert_eq!(
    observed_cell.get(),
    Some(es),
    "and into the cell the attempt sink carries into the task facts -- latched BEFORE the break",
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn zero_iteration_decode_forces_english_but_observes_nothing() {
  // F2 (codex round 4). A `sample_length` of 0 runs ZERO decode iterations, so
  // `current_tokens` stays exactly the forced multilingual prefill
  // `[SOT, <|en|>, <|transcribe|>, <|0.00|>]` -- the model predicts nothing at
  // all. The display language is still the Swift-faithful `<|en|>` off that
  // forced prompt, but nothing was OBSERVED: the predicted region is empty.
  // Pre-fix, finalization scanned the whole prompt+output slice and reported
  // the FORCED `<|en|>` as observed, so a zero-step decode fabricated an
  // English detection.
  let t = tiny_tokenizer();
  let s = special();
  let mut mock = MockBackend::new();
  mock.push_token_step(2425); // never reached: the loop runs 0 times
  let result = run_mock(
    &mock,
    &default_prompt(&s),
    &DecodingOptions::new().with_sample_length(0),
    &t,
  );
  assert_eq!(mock.counters().decode_steps(), 0, "zero decoder steps ran");
  assert_eq!(
    result.language(),
    "en",
    "the display language is the forced <|en|>"
  );
  assert_eq!(
    result.observed_language(),
    None,
    "a zero-iteration decode predicts nothing, so it observes no language"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn last_prefill_timestamp_keeps_model_prediction() {
  // TextDecoder.swift:581-594: if the LAST prompt token is a timestamp and
  // the model also predicted a timestamp, the model's wins (skip-force).
  let t = tiny_tokenizer();
  let s = special();
  let prompt = default_prompt(&s); // ends in <|0.00|>
  let predicted_ts = s.time_token_begin() + 25; // model predicts <|0.50|>
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    predicted_ts, // pos 2 predicts a timestamp for the last prompt slot
    100,          // then free text
    s.end_token(),
  ]);
  let result = run_mock(&mock, &prompt, &DecodingOptions::new(), &t);
  assert_eq!(
    result.tokens_slice()[3],
    predicted_ts,
    "model timestamp kept"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn first_token_logprob_below_threshold_stops_immediately() {
  // TextDecoder.swift:662-671: checked at tokenIndex == prefilledIndex (the
  // FIRST inference), even though prompt forcing overrides that token.
  let t = tiny_tokenizer();
  let s = special();
  let mut mock = MockBackend::new();
  // Flat logits: TimestampRulesFilter's mass-comparison rule (`filter/
  // mod.rs`'s `timestamp_mass_exceeds_text`) actually fires on a uniform
  // distribution (combined mass of 1501 timestamp tokens beats any single
  // text token when every logit is equal), so the true logprob is
  // ln(1/1501) ~= -7.31, not the naive unfiltered ln(1/51865) ~= -10.86 —
  // either way, comfortably under the -1.5 default threshold.
  mock.push_step(vec![0.0; 51865]);
  let result = run_mock(&mock, &default_prompt(&s), &DecodingOptions::new(), &t);
  assert_eq!(
    mock.counters().decode_steps(),
    1,
    "stopped after first step"
  );
  // Result carries the channel needs_fallback reads (Plan-2 assumption (b)).
  assert!(result.first_token_log_prob() < -1.5);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn early_stop_flag_breaks_loop_and_callback_sets_it() {
  let t = tiny_tokenizer();
  let s = special();
  let mut mock = MockBackend::new();
  mock.push_token_steps(&[
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
    100,
    101,
    102,
    103,
    104,
    s.end_token(),
  ]);
  let steps_seen = Mutex::new(0usize);
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let options = DecodingOptions::new();
  let mut sampler = GreedyTokenSampler::new(0.0, s.end_token(), &options);
  let mut timings = TranscriptionTimings::new();
  let callback: &(
     dyn Fn(&crate::audio::whisper::result::TranscriptionProgress) -> Option<bool> + Sync
   ) = &|_progress| {
    let mut seen = steps_seen.lock().unwrap();
    *seen += 1;
    // Prefill steps get callbacks too, so `seen` reaches 5 on the 2nd
    // non-prefill step; the stop reply is honored there.
    Some(*seen < 5)
  };
  let result = decode_text(
    &mock,
    &encoded,
    &mut state,
    &default_prompt(&s),
    &mut sampler,
    &options,
    &t,
    &mut timings,
    &AtomicBool::new(false),
    &Cell::new(None),
    Some(callback),
  )
  .unwrap();
  assert!(
    result.tokens_slice().len() < 4 + 5 + 1,
    "stopped before scripted EOT"
  );
  assert_eq!(
    *result.tokens_slice().last().unwrap(),
    s.end_token(),
    "finalize appends EOT"
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn detect_language_single_step_and_resets_state() {
  let t = tiny_tokenizer();
  let es_token = t.token_to_id("<|es|>").unwrap();
  let mut mock = MockBackend::new();
  mock.push_token_step(es_token);
  mock.push_token_step(es_token); // proves replay-from-0 after internal reset
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut timings = TranscriptionTimings::new();
  let mut sampler = GreedyTokenSampler::new(
    0.0,
    SpecialTokens::whisper_defaults().end_token(),
    &DecodingOptions::new(),
  );
  let result =
    detect_language(&mock, &encoded, &mut state, &t, &mut sampler, &mut timings).unwrap();
  assert_eq!(result.language(), "es");
  assert!(
    result
      .language_probs_slice()
      .iter()
      .any(|(code, _)| code == "es")
  );
  assert!(result.tokens_slice().is_empty()); // TextDecoder.swift:525-538
  assert_eq!(mock.counters().resets(), 1, "state reset after probe");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn detect_language_resets_state_even_when_the_step_fails() {
  // Regression (task-5 review, Important): a probe that errors partway
  // may already have advanced KV/masks; the documented reset must run on
  // error paths too, not only on success.
  let t = tiny_tokenizer();
  let mock = MockBackend::new(); // zero scripted steps -> ScriptExhausted
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut timings = TranscriptionTimings::new();
  let s = SpecialTokens::whisper_defaults();
  let mut sampler = GreedyTokenSampler::new(0.0, s.end_token(), &DecodingOptions::new());
  let err =
    detect_language(&mock, &encoded, &mut state, &t, &mut sampler, &mut timings).unwrap_err();
  assert!(matches!(err, DecodeError::Backend(_)));
  assert_eq!(mock.counters().resets(), 1, "state reset despite the error");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn detect_language_samples_through_the_callers_sampler() {
  // Regression (phase-gate round 1, High): Swift threads the attempt's
  // own sampler into the probe (TranscribeTask.swift:337-343) and draws
  // through it (TextDecoder.swift:500) — at nonzero temperature the
  // language pick is a top-k draw, and it consumes exactly one draw from
  // the attempt's RNG stream.
  let t = tiny_tokenizer();
  let es = t.token_to_id("<|es|>").unwrap();
  let de = t.token_to_id("<|de|>").unwrap();
  let mut mock = MockBackend::new();
  // Two viable languages: close logits so a t = 0.7 draw genuinely
  // consults the RNG (argmax would always pick es).
  let mut logits = vec![0.0f32; crate::audio::whisper::backend::ModelDims::new().vocab()];
  logits[es as usize] = 10.0;
  logits[de as usize] = 9.5;
  mock.push_step(logits.clone());
  let encoded = mock
    .encode(&mock.extract_features(&[0.0; 4]).unwrap())
    .unwrap();
  let mut state = mock.new_decoder_state().unwrap();
  let mut timings = TranscriptionTimings::new();
  let s = SpecialTokens::whisper_defaults();

  // Reference stream: an identically seeded sampler draws directly from
  // the identically filtered buffer.
  let mut reference =
    GreedyTokenSampler::new(0.7, s.end_token(), &DecodingOptions::new()).with_seed(7);
  let filter =
    crate::audio::whisper::decode::filter::LanguageLogitsFilter::new(t.all_language_tokens(), 0);
  let mut reference_logits = logits;
  filter.filter(&mut reference_logits, &[s.start_of_transcript_token()]);
  let expected = reference.sample(&reference_logits);
  let expected_language = t
    .language_for_token(expected.token())
    .expect("draw lands on a language token");

  let mut probe_sampler =
    GreedyTokenSampler::new(0.7, s.end_token(), &DecodingOptions::new()).with_seed(7);
  let result = detect_language(
    &mock,
    &encoded,
    &mut state,
    &t,
    &mut probe_sampler,
    &mut timings,
  )
  .unwrap();
  assert_eq!(
    result.language(),
    expected_language,
    "probe = the caller's draw"
  );

  // Exactly one draw consumed: both streams must continue in lockstep.
  let plain = [1.0f32, 2.0, 3.0, 2.5];
  for _ in 0..5 {
    assert_eq!(
      probe_sampler.sample(&plain).token(),
      reference.sample(&plain).token(),
      "streams diverged: the probe consumed a different number of draws"
    );
  }
}
