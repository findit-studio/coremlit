use std::sync::{Mutex, atomic::AtomicBool};

use super::*;
use crate::{
  backend::{InferenceBackend, mock::MockBackend},
  decode::sampler::GreedyTokenSampler,
  options::DecodingOptions,
  result::TranscriptionTimings,
  tokenizer::{SpecialTokens, WhisperTokenizer},
};

fn tiny_tokenizer() -> WhisperTokenizer {
  let root = std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    std::path::PathBuf::from,
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
) -> crate::result::DecodingResult {
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
  let callback: &(dyn Fn(&crate::result::TranscriptionProgress) -> Option<bool> + Sync) =
    &|_progress| {
      let mut seen = steps_seen.lock().unwrap();
      *seen += 1;
      Some(*seen < 5) // request stop at the 5th non-prefill callback
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
  let result = detect_language(&mock, &encoded, &mut state, &t, &mut timings).unwrap();
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
