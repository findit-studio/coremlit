use super::*;
use crate::options::DecodingOptions;

// ---------------------------------------------------------------------
// FallbackReason / needs_fallback
// ---------------------------------------------------------------------

/// Builds a `DecodingResult` with the four fields `needs_fallback` reads.
/// `first_token_lp` becomes the sole `token_log_probs` entry: Swift has no
/// separate stored "first token logprob" field either — `TextDecoder.
/// swift:788-791` builds `tokenLogProbs` as one `[token: logprob]` dict per
/// decode step, so its first entry already *is* the first sampled token's
/// logprob, and `needs_fallback` reads it the same way.
fn result_with(
  avg_logprob: f32,
  no_speech: f32,
  compression: f32,
  first_token_lp: f32,
) -> DecodingResult {
  DecodingResult::new()
    .with_avg_logprob(avg_logprob)
    .with_no_speech_prob(no_speech)
    .with_compression_ratio(compression)
    .with_token_log_probs(vec![(0u32, first_token_lp)])
}

#[test]
fn fallback_decision_order_matches_swift() {
  // Models.swift:357-381 `DecodingFallback.init?` — order matters (the
  // source's own comment, line 365); every comparison is strict (`<`/`>`,
  // never `<=`/`>=`).
  let opts = DecodingOptions::new();

  // 1. first-token logprob below threshold wins outright, before any other
  //    check runs (TextDecoder.swift:662-663; Models.swift:366-367).
  let r = result_with(-0.5, 0.1, 1.0, -2.0);
  assert_eq!(
    needs_fallback(&r, &opts),
    Some(FallbackReason::FirstTokenLogProbThreshold)
  );

  // 2. silence: `no_speech_prob > threshold` alone -> None. NOTE: this
  //    task's brief encoded an exploration reading that silence *also*
  //    required `avg_logprob < threshold`; Models.swift:368-370 has no
  //    such condition (`else if let threshold = options.noSpeechThreshold,
  //    noSpeechProb > threshold`) — avg_logprob is never consulted by this
  //    branch. This particular case's *outcome* happens to match either
  //    reading; see `fallback_silence_short_circuits_regardless_of_avg_logprob`
  //    below for a case that actually discriminates between them.
  let r = result_with(-1.5, 0.9, 1.0, 0.0);
  assert_eq!(needs_fallback(&r, &opts), None);

  // 3. compression ratio over threshold -> repetition fallback.
  let r = result_with(-0.5, 0.1, 3.0, 0.0);
  assert_eq!(
    needs_fallback(&r, &opts),
    Some(FallbackReason::CompressionRatioThreshold)
  );

  // 4. avg logprob under threshold -> quality fallback.
  let r = result_with(-1.5, 0.1, 1.0, 0.0);
  assert_eq!(
    needs_fallback(&r, &opts),
    Some(FallbackReason::LogProbThreshold)
  );

  // 5. clean result -> no fallback.
  let r = result_with(-0.2, 0.1, 1.0, 0.0);
  assert_eq!(needs_fallback(&r, &opts), None);

  // disabled thresholds (None) disable their own checks; nothing else
  // objects to a compression ratio of 3.0 here.
  let opts = DecodingOptions::new().maybe_compression_ratio_threshold(None);
  let r = result_with(-0.5, 0.1, 3.0, 0.0);
  assert_eq!(needs_fallback(&r, &opts), None);
}

#[test]
fn fallback_silence_short_circuits_regardless_of_avg_logprob() {
  // Discriminates the corrected reading from the brief's original one.
  // avg_logprob = -0.2 would NOT itself trigger LogProbThreshold, and
  // compression = 3.0 WOULD trigger CompressionRatioThreshold on its own —
  // but no_speech_prob (0.9) exceeds its threshold (0.6, default), and per
  // Models.swift:368-370 that alone short-circuits to "silence" (None)
  // *before* the compression-ratio check ever runs. Under the brief's
  // original (avg_logprob-gated) reading of "silence", this case would
  // have fallen through to the compression check instead and returned
  // `Some(CompressionRatioThreshold)`.
  let opts = DecodingOptions::new();
  let r = result_with(-0.2, 0.9, 3.0, 0.0);
  assert_eq!(needs_fallback(&r, &opts), None);
}

#[test]
fn fallback_thresholds_use_strict_inequality() {
  // Exactly-at-threshold never triggers (Models.swift uses `<`/`>`, never
  // `<=`/`>=`, at every step).
  let opts = DecodingOptions::new();
  // first_token_logprob_threshold default is Some(-1.5); exactly -1.5 must
  // not trigger.
  let r = result_with(-0.2, 0.1, 1.0, -1.5);
  assert_eq!(needs_fallback(&r, &opts), None);
  // no_speech_threshold default is Some(0.6); exactly 0.6 must not trigger
  // silence.
  let r = result_with(-0.2, 0.6, 1.0, 0.0);
  assert_eq!(needs_fallback(&r, &opts), None);
  // compression_ratio_threshold default is Some(2.4); exactly 2.4 must not
  // trigger.
  let r = result_with(-0.2, 0.1, 2.4, 0.0);
  assert_eq!(needs_fallback(&r, &opts), None);
  // logprob_threshold default is Some(-1.0); exactly -1.0 must not trigger.
  let r = result_with(-1.0, 0.1, 1.0, 0.0);
  assert_eq!(needs_fallback(&r, &opts), None);
}

#[test]
fn fallback_first_token_check_ignores_empty_token_log_probs() {
  // A `DecodingResult` with no token_log_probs at all has no "first token"
  // to check; needs_fallback must skip that branch rather than panic, and
  // fall through to the remaining checks.
  let opts = DecodingOptions::new();
  let r = DecodingResult::new().with_avg_logprob(-1.5); // logprob-threshold-worthy
  assert!(r.token_log_probs_slice().is_empty());
  assert_eq!(
    needs_fallback(&r, &opts),
    Some(FallbackReason::LogProbThreshold)
  );
}

#[test]
fn fallback_all_thresholds_disabled_never_triggers() {
  let opts = DecodingOptions::new()
    .maybe_first_token_logprob_threshold(None)
    .maybe_no_speech_threshold(None)
    .maybe_compression_ratio_threshold(None)
    .maybe_logprob_threshold(None);
  // Values that would trip every single check if thresholds were active.
  let r = result_with(-9.0, 1.0, 9.0, -9.0);
  assert_eq!(needs_fallback(&r, &opts), None);
}

#[test]
fn fallback_reason_as_str_matches_swift_strings() {
  // Models.swift:367,373,376 fallbackReason string literals.
  assert_eq!(
    FallbackReason::FirstTokenLogProbThreshold.as_str(),
    "firstTokenLogProbThreshold"
  );
  assert_eq!(
    FallbackReason::CompressionRatioThreshold.as_str(),
    "compressionRatioThreshold"
  );
  assert_eq!(
    FallbackReason::LogProbThreshold.as_str(),
    "logProbThreshold"
  );
  assert_eq!(
    FallbackReason::FirstTokenLogProbThreshold.to_string(),
    "firstTokenLogProbThreshold"
  );
  assert!(FallbackReason::LogProbThreshold.is_log_prob_threshold());
  assert!(!FallbackReason::LogProbThreshold.is_compression_ratio_threshold());
}

// ---------------------------------------------------------------------
// WordTiming
// ---------------------------------------------------------------------

#[test]
fn segment_duration_and_word_duration() {
  // NOTE: the brief's literal snippet called `.into()` on "hi"; against
  // `WordTiming::new`'s generic `impl Into<String>` parameter that is
  // ambiguous (E0283 - `&str` implements `Into<T>` for several `T`), and
  // `.into()` is redundant besides (`&str: Into<String>` already holds).
  let w = WordTiming::new("hi", vec![1], 1.0, 1.5, 0.9);
  assert_eq!(w.duration(), 0.5);
}

#[test]
fn word_timing_accessors_match_constructor() {
  // Binary-exact fractions (quarters/eighths) so `duration()`'s
  // subtraction can be compared with `==` without float rounding noise.
  let w = WordTiming::new("hello", vec![15339u32], 0.25, 0.75, 0.875);
  assert_eq!(w.word(), "hello");
  assert_eq!(w.tokens_slice(), &[15339u32]);
  assert_eq!(w.start(), 0.25);
  assert_eq!(w.end(), 0.75);
  assert_eq!(w.probability(), 0.875);
  assert_eq!(w.duration(), 0.5);
}

// ---------------------------------------------------------------------
// TranscriptionSegment
// ---------------------------------------------------------------------

#[test]
fn transcription_segment_defaults_match_swift() {
  // Models.swift:593-606 `TranscriptionSegment.init` defaults.
  let s = TranscriptionSegment::new();
  assert_eq!(s.id(), 0);
  assert_eq!(s.seek(), 0);
  assert_eq!(s.start(), 0.0);
  assert_eq!(s.end(), 0.0);
  assert!(s.text().is_empty());
  assert!(s.tokens_slice().is_empty());
  assert!(s.token_log_probs_slice().is_empty());
  assert_eq!(s.temperature(), 1.0); // NOT 0.0 - Swift default, Models.swift:601
  assert_eq!(s.avg_logprob(), 0.0);
  assert_eq!(s.compression_ratio(), 1.0); // NOT 0.0 - Swift default, Models.swift:603
  assert_eq!(s.no_speech_prob(), 0.0);
  assert!(s.words_slice().is_empty());
  assert_eq!(s.duration(), 0.0);
  assert_eq!(TranscriptionSegment::default(), TranscriptionSegment::new());
}

#[test]
fn transcription_segment_builder_vocabulary() {
  let s = TranscriptionSegment::new()
    .with_id(3)
    .with_seek(48_000)
    .with_start(1.0)
    .with_end(2.5)
    .with_text("hello world")
    .with_tokens(vec![50364u32, 15339])
    .with_token_log_probs(vec![(50364u32, -0.1), (15339, -0.2)])
    .with_temperature(0.2)
    .with_avg_logprob(-0.3)
    .with_compression_ratio(1.8)
    .with_no_speech_prob(0.01)
    .with_words(vec![WordTiming::new(
      "hello",
      vec![15339u32],
      1.0,
      1.5,
      0.9,
    )]);
  assert_eq!(s.id(), 3);
  assert_eq!(s.seek(), 48_000);
  assert_eq!(s.duration(), 1.5); // end(2.5) - start(1.0)
  assert_eq!(s.text(), "hello world");
  assert_eq!(s.tokens_slice(), &[50364u32, 15339]);
  assert_eq!(
    s.token_log_probs_slice(),
    &[(50364u32, -0.1), (15339, -0.2)]
  );
  assert_eq!(s.temperature(), 0.2);
  assert_eq!(s.avg_logprob(), -0.3);
  assert_eq!(s.compression_ratio(), 1.8);
  assert_eq!(s.no_speech_prob(), 0.01);
  assert_eq!(s.words_slice().len(), 1);

  let mut m = TranscriptionSegment::new();
  m.set_id(7).set_text("mutated");
  assert_eq!(m.id(), 7);
  assert_eq!(m.text(), "mutated");
}

// ---------------------------------------------------------------------
// TranscriptionTimings
// ---------------------------------------------------------------------

#[test]
fn timings_defaults_match_swift() {
  // Models.swift:778-843 `TranscriptionTimings.init` defaults: every
  // duration/count is zero except the two "not yet reached" sentinels and
  // the audio-seconds floor.
  let t = TranscriptionTimings::new();
  assert_eq!(t.pipeline_start(), f64::MAX);
  assert_eq!(t.first_token_time(), f64::MAX);
  assert_eq!(t.input_audio_seconds(), 0.001);
  assert_eq!(t.model_loading(), 0.0);
  assert_eq!(t.prewarm_load_time(), 0.0);
  assert_eq!(t.encoder_load_time(), 0.0);
  assert_eq!(t.decoder_load_time(), 0.0);
  assert_eq!(t.encoder_specialization_time(), 0.0);
  assert_eq!(t.decoder_specialization_time(), 0.0);
  assert_eq!(t.tokenizer_load_time(), 0.0);
  assert_eq!(t.audio_loading(), 0.0);
  assert_eq!(t.audio_processing(), 0.0);
  assert_eq!(t.logmels(), 0.0);
  assert_eq!(t.encoding(), 0.0);
  assert_eq!(t.decoding_init(), 0.0);
  assert_eq!(t.decoding_loop(), 0.0);
  assert_eq!(t.decoding_predictions(), 0.0);
  assert_eq!(t.decoding_filtering(), 0.0);
  assert_eq!(t.decoding_sampling(), 0.0);
  assert_eq!(t.decoding_fallback(), 0.0);
  assert_eq!(t.decoding_windowing(), 0.0);
  assert_eq!(t.decoding_kv_caching(), 0.0);
  assert_eq!(t.decoding_word_timestamps(), 0.0);
  assert_eq!(t.decoding_non_prediction(), 0.0);
  assert_eq!(t.total_audio_processing_runs(), 0.0);
  assert_eq!(t.total_logmel_runs(), 0.0);
  assert_eq!(t.total_encoding_runs(), 0.0);
  assert_eq!(t.total_decoding_loops(), 0.0);
  assert_eq!(t.total_kv_update_runs(), 0.0);
  assert_eq!(t.total_timestamp_alignment_runs(), 0.0);
  assert_eq!(t.total_decoding_fallbacks(), 0.0);
  assert_eq!(t.total_decoding_windows(), 0.0);
  assert_eq!(t.full_pipeline(), 0.0);
  assert_eq!(TranscriptionTimings::default(), TranscriptionTimings::new());
}

#[test]
fn timings_projections() {
  let mut t = TranscriptionTimings::new();
  t.set_full_pipeline(2.0)
    .set_total_decoding_loops(100.0)
    .set_input_audio_seconds(10.0);
  assert_eq!(t.tokens_per_second(), 50.0);
  assert_eq!(t.real_time_factor(), 0.2);
  assert_eq!(t.speed_factor(), 5.0);
}

#[test]
fn timings_projections_guard_division_by_zero() {
  let mut t = TranscriptionTimings::new();
  t.set_full_pipeline(0.0);
  assert_eq!(t.tokens_per_second(), 0.0); // would be NaN/inf unguarded
  assert_eq!(t.speed_factor(), 0.0);
  t.set_full_pipeline(5.0).set_input_audio_seconds(0.0);
  assert_eq!(t.real_time_factor(), 0.0);
}

#[test]
fn timings_setters_mutate_in_place_and_chain() {
  let mut t = TranscriptionTimings::new();
  t.set_model_loading(1.2)
    .set_encoder_load_time(0.4)
    .set_decoder_load_time(0.6)
    .set_total_decoding_windows(3.0);
  assert_eq!(t.model_loading(), 1.2);
  assert_eq!(t.encoder_load_time(), 0.4);
  assert_eq!(t.decoder_load_time(), 0.6);
  assert_eq!(t.total_decoding_windows(), 3.0);
}

// ---------------------------------------------------------------------
// TranscriptionResult
// ---------------------------------------------------------------------

#[test]
fn transcription_result_requires_core_fields_and_defaults_seek_time() {
  let timings = TranscriptionTimings::new();
  let r = TranscriptionResult::new("hello world", Vec::new(), "en", timings.clone());
  assert_eq!(r.text(), "hello world");
  assert!(r.segments_slice().is_empty());
  assert_eq!(r.language(), "en");
  assert_eq!(r.timings(), &timings);
  assert_eq!(r.seek_time(), None);
}

#[test]
fn transcription_result_seek_time_option_vocabulary() {
  let r =
    TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new()).with_seek_time(12.5);
  assert_eq!(r.seek_time(), Some(12.5));
  let mut r = r;
  r.clear_seek_time();
  assert_eq!(r.seek_time(), None);
  r.update_seek_time(Some(3.0));
  assert_eq!(r.seek_time(), Some(3.0));
  let r = r.maybe_seek_time(None);
  assert_eq!(r.seek_time(), None);
}

// ---------------------------------------------------------------------
// DecodingResult
// ---------------------------------------------------------------------

#[test]
fn decoding_result_defaults_match_swift_empty_results() {
  // Models.swift:397-410 `DecodingResult.emptyResults`.
  let r = DecodingResult::new();
  assert!(r.language().is_empty());
  assert!(r.language_probs_slice().is_empty());
  assert!(r.tokens_slice().is_empty());
  assert!(r.token_log_probs_slice().is_empty());
  assert!(r.text().is_empty());
  assert_eq!(r.avg_logprob(), 0.0);
  assert_eq!(r.no_speech_prob(), 0.0);
  assert_eq!(r.temperature(), 0.0); // unlike TranscriptionSegment's 1.0 default
  assert_eq!(r.compression_ratio(), 0.0); // unlike TranscriptionSegment's 1.0 default
  assert_eq!(DecodingResult::default(), DecodingResult::new());
}

#[test]
fn decoding_result_builder_vocabulary() {
  let r = DecodingResult::new()
    .with_language("en")
    .with_language_probs(vec![("en".to_string(), 0.98)])
    .with_tokens(vec![50364u32, 15339])
    .with_token_log_probs(vec![(50364u32, -0.05)])
    .with_text("hello")
    .with_avg_logprob(-0.4)
    .with_no_speech_prob(0.02)
    .with_temperature(0.2)
    .with_compression_ratio(1.6);
  assert_eq!(r.language(), "en");
  assert_eq!(r.language_probs_slice(), &[("en".to_string(), 0.98)]);
  assert_eq!(r.tokens_slice(), &[50364u32, 15339]);
  assert_eq!(r.token_log_probs_slice(), &[(50364u32, -0.05)]);
  assert_eq!(r.text(), "hello");
  assert_eq!(r.avg_logprob(), -0.4);
  assert_eq!(r.no_speech_prob(), 0.02);
  assert_eq!(r.temperature(), 0.2);
  assert_eq!(r.compression_ratio(), 1.6);

  let mut m = DecodingResult::new();
  m.set_text("mutated").set_avg_logprob(-1.0);
  assert_eq!(m.text(), "mutated");
  assert_eq!(m.avg_logprob(), -1.0);
}

// ---------------------------------------------------------------------
// serde
// ---------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn word_timing_serde_round_trips_and_requires_every_field() {
  let w = WordTiming::new("hi", vec![1u32], 1.0, 1.5, 0.9);
  let json = serde_json::to_string(&w).unwrap();
  assert_eq!(serde_json::from_str::<WordTiming>(&json).unwrap(), w);
  // No defaults: a payload missing a field is an error (matches Swift
  // Codable's synthesis, which has no init-default fallback either).
  assert!(serde_json::from_str::<WordTiming>(r#"{"word":"hi"}"#).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn transcription_segment_serde_skips_empty_words_and_fills_defaults() {
  let s = TranscriptionSegment::new().with_text("hi");
  let json = serde_json::to_string(&s).unwrap();
  let value: serde_json::Value = serde_json::from_str(&json).unwrap();
  assert!(!value.as_object().unwrap().contains_key("words"));
  assert!(!value.as_object().unwrap().contains_key("tokens"));
  assert_eq!(
    serde_json::from_str::<TranscriptionSegment>(&json).unwrap(),
    s
  );
  // Partial config still resolves temperature/compression_ratio to
  // Swift's non-zero defaults, not f32::default().
  let partial: TranscriptionSegment = serde_json::from_str("{}").unwrap();
  assert_eq!(partial, TranscriptionSegment::new());
  assert_eq!(partial.temperature(), 1.0);
  assert_eq!(partial.compression_ratio(), 1.0);
}

#[cfg(feature = "serde")]
#[test]
fn transcription_timings_serde_round_trips_and_fills_sentinel_defaults() {
  let t = TranscriptionTimings::new();
  let json = serde_json::to_string(&t).unwrap();
  assert_eq!(
    serde_json::from_str::<TranscriptionTimings>(&json).unwrap(),
    t
  );
  let partial: TranscriptionTimings = serde_json::from_str("{}").unwrap();
  assert_eq!(partial.pipeline_start(), f64::MAX);
  assert_eq!(partial.input_audio_seconds(), 0.001);
}

#[cfg(feature = "serde")]
#[test]
fn transcription_result_serde_skips_absent_seek_time() {
  let r = TranscriptionResult::new("hi", Vec::new(), "en", TranscriptionTimings::new());
  let json = serde_json::to_string(&r).unwrap();
  assert!(!json.contains("seek_time"));
  assert_eq!(
    serde_json::from_str::<TranscriptionResult>(&json).unwrap(),
    r
  );
  let with_seek = r.with_seek_time(1.5);
  let json = serde_json::to_string(&with_seek).unwrap();
  assert!(json.contains("seek_time"));
  assert_eq!(
    serde_json::from_str::<TranscriptionResult>(&json).unwrap(),
    with_seek
  );
}

#[cfg(feature = "serde")]
#[test]
fn decoding_result_serde_round_trips_and_skips_empty_collections() {
  let r = DecodingResult::new().with_text("hi");
  let json = serde_json::to_string(&r).unwrap();
  let value: serde_json::Value = serde_json::from_str(&json).unwrap();
  let object = value.as_object().unwrap();
  assert!(!object.contains_key("language"));
  assert!(!object.contains_key("tokens"));
  assert!(!object.contains_key("token_log_probs"));
  assert!(!object.contains_key("language_probs"));
  assert_eq!(serde_json::from_str::<DecodingResult>(&json).unwrap(), r);
  assert_eq!(
    serde_json::from_str::<DecodingResult>("{}").unwrap(),
    DecodingResult::new()
  );
}

#[cfg(feature = "serde")]
#[test]
fn fallback_reason_serde_uses_swift_strings() {
  assert_eq!(
    serde_json::to_string(&FallbackReason::FirstTokenLogProbThreshold).unwrap(),
    "\"firstTokenLogProbThreshold\""
  );
  assert_eq!(
    serde_json::from_str::<FallbackReason>("\"logProbThreshold\"").unwrap(),
    FallbackReason::LogProbThreshold
  );
}
