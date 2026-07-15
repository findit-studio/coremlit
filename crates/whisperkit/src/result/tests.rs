use super::*;
use crate::options::DecodingOptions;

// ---------------------------------------------------------------------
// merge_transcription_results
// ---------------------------------------------------------------------

#[test]
fn merge_transcription_results_concatenates_and_reids() {
  // NOTE: the brief's literal snippet called `TranscriptionResult::new()`
  // with no arguments; the shipped constructor requires all four fields
  // (text, segments, language, timings) with no defaulted/zero-arg form
  // (this module's own doc: "no honest default means no Default" applies
  // equally to a bare `new()`). Built blank here, then mutated via the
  // `set_*` calls the brief's snippet already used. Likewise `.into()` on
  // the string literals is dropped: against `set_text`/`set_language`'s
  // generic `impl Into<String>` parameter it is ambiguous (E0283 - `&str`
  // implements `Into<T>` for several `T`), the same fix already applied
  // to `WordTiming::new`'s call site above.
  let mut first = TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new());
  let mut seg0 = TranscriptionSegment::new();
  seg0.set_id(0).set_start(0.0).set_end(1.0);
  first
    .set_text("hello")
    .set_segments(vec![seg0])
    .set_language("en");
  let mut second = TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new());
  let mut seg1 = TranscriptionSegment::new();
  seg1.set_id(0).set_start(30.0).set_end(31.0);
  second.set_text("world").set_segments(vec![seg1]);
  let merged = merge_transcription_results(&[first, second]);
  assert_eq!(merged.text(), "hello world");
  assert_eq!(merged.segments_slice().len(), 2);
  assert_eq!(merged.segments_slice()[1].id(), 1); // resultIndex + segmentIndex (:89-94)
  assert_eq!(merged.language(), "en");
}

#[test]
fn merge_preserves_survivor_ids_when_dropping_blanks() {
  // F4 (codex round 2). The VAD path ALWAYS routes chunk results through
  // `merge_transcription_results_with_options` (transcribe::transcribe), and
  // that merge reindexed every survivor to `result_index + segment_index` --
  // silently collapsing the [0, 2] gap a blank-audio drop leaves back to
  // [0, 1], so `drop_blank_audio`'s documented "survivors keep their decoded
  // ids" promise held only on the unmerged single-chunk path.
  //
  // One chunk, speech-blank-speech, the blank already dropped in the task ->
  // survivors carry decode ids 0 and 2. Match survivors by tokens, assert ids.
  let mut speech0 = TranscriptionSegment::new();
  speech0
    .set_id(0)
    .set_start(0.0)
    .set_end(1.0)
    .set_text(" Hello")
    .set_tokens(vec![10]);
  let mut speech2 = TranscriptionSegment::new();
  speech2
    .set_id(2)
    .set_start(2.0)
    .set_end(3.0)
    .set_text(" World")
    .set_tokens(vec![20]);
  let chunk = TranscriptionResult::new(
    "Hello World",
    vec![speech0, speech2],
    "en",
    TranscriptionTimings::new(),
  );

  // Dropping ON (the default): ids preserved, the [0, 2] hole intact.
  let dropped =
    merge_transcription_results_with_options(std::slice::from_ref(&chunk), &DecodingOptions::new());
  assert_eq!(
    dropped
      .segments_slice()
      .iter()
      .map(TranscriptionSegment::id)
      .collect::<Vec<_>>(),
    vec![0, 2],
    "survivors keep their decode ids; the dropped segment's gap is preserved"
  );
  // Survivors matched by tokens, NOT id: the second is still " World".
  assert_eq!(dropped.segments_slice()[1].tokens_slice(), &[20]);
  assert_eq!(dropped.segments_slice()[1].start(), 2.0);

  // Dropping OFF: EXACTLY Swift's `result_index + segment_index` reindexing
  // -- the false path stays byte-for-byte Swift, so the same survivors come
  // back densely renumbered [0, 1].
  let swift = merge_transcription_results_with_options(
    std::slice::from_ref(&chunk),
    &DecodingOptions::new().maybe_drop_blank_audio(false),
  );
  assert_eq!(
    swift
      .segments_slice()
      .iter()
      .map(TranscriptionSegment::id)
      .collect::<Vec<_>>(),
    vec![0, 1],
    "Swift-exact reindexing (result_index + segment_index) when dropping is off"
  );
  assert_eq!(
    swift.segments_slice()[1].tokens_slice(),
    &[20],
    "still the same survivor, only its id differs"
  );

  // Multiple chunks, dropping ON: each VAD chunk is its own decode with its
  // own id space, so both lone survivors carry decode id 0. The
  // `result_index` offset is what keeps them from COLLIDING to [0, 0] -- the
  // id preservation must still disambiguate across chunks, landing [0, 1].
  let mut chunk_a_seg = TranscriptionSegment::new();
  chunk_a_seg
    .set_id(0)
    .set_text(" Hello")
    .set_tokens(vec![10]);
  let mut chunk_b_seg = TranscriptionSegment::new();
  chunk_b_seg
    .set_id(0)
    .set_text(" World")
    .set_tokens(vec![20]);
  let chunk_a = TranscriptionResult::new(
    "Hello",
    vec![chunk_a_seg],
    "en",
    TranscriptionTimings::new(),
  );
  let chunk_b = TranscriptionResult::new(
    "World",
    vec![chunk_b_seg],
    "en",
    TranscriptionTimings::new(),
  );
  let two_chunks =
    merge_transcription_results_with_options(&[chunk_a, chunk_b], &DecodingOptions::new());
  assert_eq!(
    two_chunks
      .segments_slice()
      .iter()
      .map(TranscriptionSegment::id)
      .collect::<Vec<_>>(),
    vec![0, 1],
    "each chunk's lone `id() == 0` must be offset by result_index, not collapsed to [0, 0]"
  );
  assert_eq!(two_chunks.segments_slice()[1].tokens_slice(), &[20]);
}

/// A result carrying nothing but text — the shape `transcribe_all` returns
/// for a chunk/clip whose segments were all emptied (or, independently of
/// the blank-audio drop, for any clip shorter than `window_clip_time`).
fn spoken(text: &str) -> TranscriptionResult {
  TranscriptionResult::new(text, Vec::new(), "en", TranscriptionTimings::new())
}

#[test]
fn merge_joins_an_empty_text_as_a_bare_separator() {
  // PARITY PIN (issue #14). The options-BLIND merge deliberately does NOT
  // skip empty-text results: Swift's `validResults` `compactMap`s away
  // only *nil* elements, never empty-text ones, so
  // `["a", "", "b"].joined(separator: " ")` is `"a  b"` there and must be
  // `"a  b"` here.
  //
  // It is tempting to "fix" this here, because
  // `DecodingOptions::drop_blank_audio` (default `true`) makes an emptied
  // chunk common. DON'T — an empty-text result is reachable with NO
  // involvement from that option (any audio shorter than
  // `window_clip_time` runs no window and returns one; see
  // `transcribe::tests::audio_shorter_than_window_clip_time_yields_no_windows`,
  // which predates the option), so filtering unconditionally here would
  // silently change the `drop_blank_audio == false` path — the path whose
  // whole purpose is to be byte-for-byte Swift. The skip belongs to the
  // option, and therefore to `merge_transcription_results_with_options`
  // (below); this test is what keeps it from creeping down here.
  assert_eq!(
    merge_transcription_results(&[spoken("a"), spoken(""), spoken("b")]).text(),
    "a  b",
    "interior empty stays a bare separator (Swift parity)"
  );
  assert_eq!(
    merge_transcription_results(&[spoken("a"), spoken("")]).text(),
    "a ",
    "trailing empty stays a bare separator (Swift parity)"
  );
}

#[test]
fn merge_with_options_skips_empty_texts_when_blank_audio_is_dropped() {
  // THE REGRESSION, at the public door a consumer actually uses: fold a
  // `transcribe_all` batch through the merge under the DEFAULT options
  // (`drop_blank_audio == true`). An emptied result must contribute no
  // separator at all — the merge's own `["a", "", "b"].join(" ")` would
  // make it `"a  b"`.
  let options = DecodingOptions::new();
  assert!(options.drop_blank_audio(), "this is the default path");
  let text = |results: &[TranscriptionResult]| {
    merge_transcription_results_with_options(results, &options)
      .text()
      .to_string()
  };

  // Interior: an emptied chunk BETWEEN two speech runs -> no doubled space.
  assert_eq!(
    text(&[spoken("Hello world."), spoken(""), spoken("Goodbye.")]),
    "Hello world. Goodbye."
  );
  // Trailing: emptied chunks after the speech -> no trailing space(s).
  assert_eq!(
    text(&[spoken("Hello world."), spoken(""), spoken("")]),
    "Hello world."
  );
  // Leading: an emptied chunk before the speech -> no leading space.
  assert_eq!(text(&[spoken(""), spoken("Hello world.")]), "Hello world.");
  // Wholly emptied: nothing at all, not a string of bare separators.
  assert_eq!(text(&[spoken(""), spoken(""), spoken("")]), "");
  // Speech only: the join is untouched — one separator per gap.
  assert_eq!(text(&[spoken("Hello"), spoken("world.")]), "Hello world.");
  // Empty input: still the empty string (`[].join(" ")`).
  assert_eq!(text(&[]), "");
}

#[test]
fn merge_with_options_joins_empty_texts_verbatim_when_the_drop_is_cleared() {
  // The `false` TWIN of the test above, and the parity pin on this entry
  // point: cleared, it must reproduce `merge_transcription_results` — bare
  // separators and all — byte for byte. This is what makes the skip above
  // provably attributable to the option rather than to the new function.
  let options = DecodingOptions::new().maybe_drop_blank_audio(false);
  let results = [spoken("Hello world."), spoken(""), spoken("Goodbye.")];

  let merged = merge_transcription_results_with_options(&results, &options);
  assert_eq!(
    merged.text(),
    "Hello world.  Goodbye.",
    "the bare separator must SURVIVE when the drop is cleared (Swift parity)"
  );
  assert_eq!(
    merged.text(),
    merge_transcription_results(&results).text(),
    "cleared, this entry point IS the options-blind merge"
  );
  assert_eq!(
    merge_transcription_results_with_options(&[spoken("a"), spoken("")], &options).text(),
    "a ",
    "trailing bare separator survives too"
  );
}

#[test]
fn merge_with_options_keeps_every_result_in_the_timing_sums() {
  // The skip is a JOIN rule, not a merge-input filter. Dropping an emptied
  // result from the merge instead would take its `input_audio_seconds` /
  // `audio_processing` / every other summed timing out with it — silently
  // corrupting the merged metrics, and the RTF derived from them, to fix a
  // spacing bug. Every field except `text` must therefore be INVARIANT
  // under the option.
  let timed = |text: &str, audio_seconds: f64| {
    let mut timings = TranscriptionTimings::new();
    timings
      .set_input_audio_seconds(audio_seconds)
      .set_audio_processing(audio_seconds / 10.0)
      .set_total_audio_processing_runs(1.0)
      .set_full_pipeline(audio_seconds / 4.0);
    TranscriptionResult::new(text, Vec::new(), "en", timings)
  };
  // The middle chunk is 30 s of silence the blank-audio drop emptied: no
  // text, but 30 s of audio that really was processed.
  let results = [
    timed("Hello world.", 30.0),
    timed("", 30.0),
    timed("Goodbye.", 20.0),
  ];

  let dropped = merge_transcription_results_with_options(
    &results,
    &DecodingOptions::new().with_drop_blank_audio(),
  );
  let kept = merge_transcription_results_with_options(
    &results,
    &DecodingOptions::new().maybe_drop_blank_audio(false),
  );
  let blind = merge_transcription_results(&results);

  // Only the text moves.
  assert_eq!(dropped.text(), "Hello world. Goodbye.");
  assert_eq!(kept.text(), "Hello world.  Goodbye.");

  // Everything else is byte-identical across all three doors — including
  // the EMPTIED chunk's 30 s, which must still be in the sums.
  for other in [&kept, &blind] {
    assert_eq!(
      dropped.timings().input_audio_seconds(),
      other.timings().input_audio_seconds()
    );
    assert_eq!(
      dropped.timings().audio_processing(),
      other.timings().audio_processing()
    );
    assert_eq!(
      dropped.timings().total_audio_processing_runs(),
      other.timings().total_audio_processing_runs()
    );
    assert_eq!(
      dropped.timings().full_pipeline(),
      other.timings().full_pipeline()
    );
    assert_eq!(
      dropped.timings().real_time_factor(),
      other.timings().real_time_factor()
    );
    assert_eq!(dropped.segments_slice().len(), other.segments_slice().len());
    assert_eq!(dropped.language(), other.language());
  }

  // ...and the sums are the REAL ones, not the ones a skipped result leaves
  // behind: 30 + 30 + 20, not 30 + 20.
  assert_eq!(dropped.timings().input_audio_seconds(), 80.0);
  assert_eq!(dropped.timings().total_audio_processing_runs(), 3.0);
  assert_eq!(dropped.timings().full_pipeline(), 20.0);
  assert_eq!(dropped.timings().real_time_factor(), 20.0 / 80.0);
}

#[test]
fn merge_full_pipeline_sums_when_pipeline_start_is_never_stamped() {
  // Regression (task-12 review): with every pipeline_start at the
  // "never stamped" sentinel (f64::MAX) — which is what every result this
  // sync port produces looks like — the merged full_pipeline must be the
  // sum of the per-result full_pipelines. The naive Swift formula
  // degenerates here: f64::MAX + full_pipeline ABSORBS (the ULP at that
  // magnitude is ~2e292, so the sum rounds back to exactly f64::MAX, it
  // does NOT overflow to infinity), making user_pipeline_duration
  // f64::MAX - f64::MAX == 0.0 and min() zero out the real sum.
  let mut timings_a = TranscriptionTimings::new();
  timings_a
    .set_full_pipeline(2.0)
    .set_total_decoding_loops(10.0);
  let a = TranscriptionResult::new("a", Vec::new(), "en", timings_a);
  let mut timings_b = TranscriptionTimings::new();
  timings_b
    .set_full_pipeline(3.0)
    .set_total_decoding_loops(20.0);
  let b = TranscriptionResult::new("b", Vec::new(), "en", timings_b);
  let merged = merge_transcription_results(&[a, b]);
  assert_eq!(merged.timings().full_pipeline(), 5.0);
  // The derived projections must therefore be live, not zeroed.
  assert_eq!(merged.timings().tokens_per_second(), 30.0 / 5.0);
  // The sentinel itself survives the merge (min of sentinels), matching
  // Swift's own formula on the same input.
  assert_eq!(merged.timings().pipeline_start(), f64::MAX);
  assert_eq!(merged.timings().first_token_time(), f64::MAX);
}

#[test]
fn merge_full_pipeline_takes_wall_clock_span_when_starts_are_real() {
  // The general Swift formula (TranscriptionUtilities.swift:110-114) on
  // results that DO carry real pipeline_start stamps: two overlapping
  // concurrent pipelines, user span = (101 + 3) - 100 = 4, system sum =
  // 2 + 3 = 5, merged full_pipeline = min(4, 5) = 4.
  let mut timings_a = TranscriptionTimings::new();
  timings_a.set_pipeline_start(100.0).set_full_pipeline(2.0);
  let a = TranscriptionResult::new("a", Vec::new(), "en", timings_a);
  let mut timings_b = TranscriptionTimings::new();
  timings_b.set_pipeline_start(101.0).set_full_pipeline(3.0);
  let b = TranscriptionResult::new("b", Vec::new(), "en", timings_b);
  let merged = merge_transcription_results(&[a, b]);
  assert_eq!(merged.timings().full_pipeline(), 4.0);
  assert_eq!(merged.timings().pipeline_start(), 100.0);
}

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
  //    check runs (TextDecoder.swift:662-667; Models.swift:366-367).
  //    first_token_lp=-2.0 < threshold=-1.5, so flag=true.
  let r = result_with(-0.5, 0.1, 1.0, -2.0);
  assert_eq!(
    needs_fallback(true, &r, &opts),
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
  //    first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let r = result_with(-1.5, 0.9, 1.0, 0.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);

  // 3. compression ratio over threshold -> repetition fallback.
  //    first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let r = result_with(-0.5, 0.1, 3.0, 0.0);
  assert_eq!(
    needs_fallback(false, &r, &opts),
    Some(FallbackReason::CompressionRatioThreshold)
  );

  // 4. avg logprob under threshold -> quality fallback.
  //    first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let r = result_with(-1.5, 0.1, 1.0, 0.0);
  assert_eq!(
    needs_fallback(false, &r, &opts),
    Some(FallbackReason::LogProbThreshold)
  );

  // 5. clean result -> no fallback.
  //    first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let r = result_with(-0.2, 0.1, 1.0, 0.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);

  // disabled thresholds (None) disable their own checks; nothing else
  // objects to a compression ratio of 3.0 here.
  let opts = DecodingOptions::new().maybe_compression_ratio_threshold(None);
  let r = result_with(-0.5, 0.1, 3.0, 0.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);
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
  // first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let opts = DecodingOptions::new();
  let r = result_with(-0.2, 0.9, 3.0, 0.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);
}

#[test]
fn fallback_thresholds_use_strict_inequality() {
  // Exactly-at-threshold never triggers (Models.swift uses `<`/`>`, never
  // `<=`/`>=`, at every step).
  let opts = DecodingOptions::new();
  // first_token_logprob_threshold default is Some(-1.5); exactly -1.5 must
  // not trigger. first_token_lp=-1.5 == threshold, so flag=false.
  let r = result_with(-0.2, 0.1, 1.0, -1.5);
  assert_eq!(needs_fallback(false, &r, &opts), None);
  // no_speech_threshold default is Some(0.6); exactly 0.6 must not trigger
  // silence. first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let r = result_with(-0.2, 0.6, 1.0, 0.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);
  // compression_ratio_threshold default is Some(2.4); exactly 2.4 must not
  // trigger. first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let r = result_with(-0.2, 0.1, 2.4, 0.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);
  // logprob_threshold default is Some(-1.0); exactly -1.0 must not trigger.
  // first_token_lp=0.0 >= threshold=-1.5, so flag=false.
  let r = result_with(-1.0, 0.1, 1.0, 0.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);
}

#[test]
fn empty_word_tokens_do_not_trigger_compression_fallback() {
  // PARITY (coremlit issue #9), decision level. An empty word-token window
  // (decode/mod.rs feeds `compression_ratio_of_tokens(&word_tokens)`, and
  // `word_tokens` can be empty) yields a compression ratio of 0.0 — Swift's
  // value, since its tokens overload has no empty guard (see
  // `text::tests::compression_ratio_of_tokens_empty_is_zero_matching_swift`).
  // Threaded through `needs_fallback` at the DEFAULT threshold (Some(2.4)),
  // `0.0 > 2.4` is false, so the compression check does not fire and no
  // repetition fallback is requested — matching Swift. Before this fix the
  // ratio was f32::INFINITY, `INFINITY > 2.4` was true, and this same empty
  // window would have (wrongly) forced a fallback: the exact parity bug this
  // guards against.
  let empty_ratio = crate::text::compression_ratio_of_tokens(&[]);
  assert_eq!(empty_ratio, 0.0);
  // Other signals kept clean so the compression branch is the one under
  // test: no_speech below its 0.6 default, avg_logprob above its -1.0
  // default, first-token flag false.
  let opts = DecodingOptions::new();
  let r = result_with(-0.2, 0.1, empty_ratio, 0.0);
  assert_ne!(
    needs_fallback(false, &r, &opts),
    Some(FallbackReason::CompressionRatioThreshold)
  );
  assert_eq!(needs_fallback(false, &r, &opts), None);
}

#[test]
fn fallback_first_token_check_ignores_empty_token_log_probs() {
  // A `DecodingResult` with no token_log_probs at all still requires the
  // caller to compute first_token_log_prob_too_low from the loop-local
  // first token; this test passes false (no first-token fallback) and
  // verifies the function continues to check other thresholds.
  let opts = DecodingOptions::new();
  let r = DecodingResult::new().with_avg_logprob(-1.5); // logprob-threshold-worthy
  assert!(r.token_log_probs_slice().is_empty());
  assert_eq!(
    needs_fallback(false, &r, &opts),
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
  // Pass false for the first_token flag since thresholds are disabled anyway.
  let r = result_with(-9.0, 1.0, 9.0, -9.0);
  assert_eq!(needs_fallback(false, &r, &opts), None);
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
  // Rust-only addition beyond Swift's field set (T5/decode loop assumption
  // (b), see `needs_fallback`'s doc): the raw first-sampled-token logprob,
  // threaded out of the loop so a fallback-ladder caller can recompute
  // `first_token_log_prob_too_low` without decode_text changing its return
  // type.
  assert_eq!(r.first_token_log_prob(), 0.0);
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
    .with_compression_ratio(1.6)
    .with_first_token_log_prob(-0.8);
  assert_eq!(r.language(), "en");
  assert_eq!(r.language_probs_slice(), &[("en".to_string(), 0.98)]);
  assert_eq!(r.tokens_slice(), &[50364u32, 15339]);
  assert_eq!(r.token_log_probs_slice(), &[(50364u32, -0.05)]);
  assert_eq!(r.text(), "hello");
  assert_eq!(r.avg_logprob(), -0.4);
  assert_eq!(r.no_speech_prob(), 0.02);
  assert_eq!(r.temperature(), 0.2);
  assert_eq!(r.compression_ratio(), 1.6);
  assert_eq!(r.first_token_log_prob(), -0.8);

  let mut m = DecodingResult::new();
  m.set_text("mutated").set_avg_logprob(-1.0);
  assert_eq!(m.text(), "mutated");
  assert_eq!(m.avg_logprob(), -1.0);
}

// ---------------------------------------------------------------------
// TranscriptionProgress
// ---------------------------------------------------------------------

#[test]
fn transcription_progress_defaults_match_swift() {
  // Models.swift:643-660 `TranscriptionProgress.init` defaults: the
  // optional trio starts `nil`, `windowId` starts `0`.
  let timings = TranscriptionTimings::new();
  let p = TranscriptionProgress::new(timings.clone(), "hello", vec![50364u32, 15339]);
  assert_eq!(p.timings(), &timings);
  assert_eq!(p.text(), "hello");
  assert_eq!(p.tokens_slice(), &[50364u32, 15339]);
  assert_eq!(p.temperature(), None);
  assert_eq!(p.avg_logprob(), None);
  assert_eq!(p.compression_ratio(), None);
  assert_eq!(p.window_id(), 0);
}

#[test]
fn transcription_progress_builder_vocabulary() {
  let p = TranscriptionProgress::new(TranscriptionTimings::new(), "hi", Vec::new())
    .with_temperature(0.2)
    .with_avg_logprob(-0.3)
    .with_compression_ratio(1.4)
    .with_window_id(2);
  assert_eq!(p.temperature(), Some(0.2));
  assert_eq!(p.avg_logprob(), Some(-0.3));
  assert_eq!(p.compression_ratio(), Some(1.4));
  assert_eq!(p.window_id(), 2);

  let mut m = p.clone();
  m.clear_temperature();
  assert_eq!(m.temperature(), None);
  m.update_avg_logprob(Some(-0.9));
  assert_eq!(m.avg_logprob(), Some(-0.9));
  m.set_text("mutated").set_tokens(vec![1u32]);
  assert_eq!(m.text(), "mutated");
  assert_eq!(m.tokens_slice(), &[1u32]);
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
fn sampled_at_nonzero_temperature_is_required_on_deserialize() {
  // F1 (codex round 2). The flag must never silently default to `false`
  // ("never sampled", the optimistic answer) when a persisted record drops
  // it: a blank-dropped result whose sampled window was filtered away carries
  // the fact ONLY here, and a `false` default would hand
  // `Provenance::is_reproducible` a guarantee the run never earned. Mirrors
  // the same requirement on `Provenance`'s carried flag
  // (`provenance::tests::a_record_missing_a_library_known_field_is_rejected`).
  let sampled_empty = TranscriptionResult::new("", Vec::new(), "en", TranscriptionTimings::new())
    .with_sampled_at_nonzero_temperature();
  let value: serde_json::Value = serde_json::to_value(&sampled_empty).unwrap();
  // The intact record round-trips, or the removal below proves nothing.
  assert_eq!(
    serde_json::from_value::<TranscriptionResult>(value.clone()).unwrap(),
    sampled_empty
  );

  // Drop the key: it must FAIL, not default to `false`.
  let mut without = value;
  assert!(
    without
      .as_object_mut()
      .unwrap()
      .remove("sampled_at_nonzero_temperature")
      .is_some(),
    "the flag is always serialized, so the key must have been present"
  );
  assert!(
    serde_json::from_value::<TranscriptionResult>(without).is_err(),
    "a dropped `sampled_at_nonzero_temperature` must be rejected, not read back false"
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
fn transcription_progress_serde_round_trips_and_skips_absent_optionals() {
  let p = TranscriptionProgress::new(TranscriptionTimings::new(), "hi", Vec::new());
  let json = serde_json::to_string(&p).unwrap();
  let value: serde_json::Value = serde_json::from_str(&json).unwrap();
  let object = value.as_object().unwrap();
  assert!(!object.contains_key("temperature"));
  assert!(!object.contains_key("avg_logprob"));
  assert!(!object.contains_key("compression_ratio"));
  assert!(!object.contains_key("tokens"));
  assert_eq!(
    serde_json::from_str::<TranscriptionProgress>(&json).unwrap(),
    p
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

// ---------------------------------------------------------------------
// all_words / format_segments / merge_transcription_results_with_words
// ---------------------------------------------------------------------

fn timed_word(text: &str, start: f32, end: f32) -> WordTiming {
  // See the analogous NOTE on `word()` in `text/tests.rs`: `.into()` on
  // `text` is dropped here for the same E0283 ambiguity reason already
  // documented earlier in this file's own `WordTiming::new` call sites.
  WordTiming::new(text, vec![1], start, end, 0.9)
}

fn segment_with_words(
  start: f32,
  end: f32,
  text: &str,
  words: Vec<WordTiming>,
) -> TranscriptionSegment {
  let mut segment = TranscriptionSegment::new();
  segment
    .set_start(start)
    .set_end(end)
    .set_text(text)
    .set_words(words);
  segment
}

#[test]
fn all_words_flattens_segments_in_order() {
  // Models.swift:566-570
  let mut result = TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new());
  result.set_segments(vec![
    segment_with_words(0.0, 1.0, " Hi", vec![timed_word(" Hi", 0.0, 0.5)]),
    segment_with_words(
      1.0,
      2.0,
      " there now",
      vec![timed_word(" there", 1.0, 1.4), timed_word(" now", 1.4, 1.9)],
    ),
  ]);
  let words = result.all_words();
  assert_eq!(words.len(), 3);
  assert_eq!(words[1].word(), " there");
  // Segments without words contribute nothing (empty-means-absent).
  let mut bare = TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new());
  bare.set_segments(vec![segment_with_words(0.0, 1.0, " Hi", vec![])]);
  assert!(bare.all_words().is_empty());
}

#[test]
fn format_segments_renders_timestamps_and_raw_text() {
  // TranscriptionUtilities.swift:16-27 + Logging.formatTimestamp ("%.2f").
  let segments = [segment_with_words(0.0, 2.5, " Hello", vec![])];
  assert_eq!(
    format_segments(&segments, true),
    vec!["[0.00 --> 2.50]  Hello".to_string()]
  );
  assert_eq!(
    format_segments(&segments, false),
    vec![" Hello".to_string()]
  );
}

#[test]
fn merge_with_confirmed_words_overrides_text_only() {
  // TranscriptionUtilities.swift:76-82 — confirmed words joined with NO
  // separator; segments/language/timings identical to the plain merge.
  let mut first = TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new());
  first
    .set_text("hello")
    .set_language("en")
    .set_segments(vec![segment_with_words(0.0, 1.0, "hello", vec![])]);
  let mut second = TranscriptionResult::new("", Vec::new(), "", TranscriptionTimings::new());
  second
    .set_text("world")
    .set_segments(vec![segment_with_words(30.0, 31.0, "world", vec![])]);
  let results = [first, second];
  let confirmed = [timed_word(" And", 0.0, 0.4), timed_word(" so", 0.4, 0.7)];

  let with_words = merge_transcription_results_with_words(&results, &confirmed);
  let plain = merge_transcription_results(&results);
  assert_eq!(with_words.text(), " And so");
  assert_eq!(plain.text(), "hello world");
  assert_eq!(with_words.segments_slice(), plain.segments_slice());
  assert_eq!(with_words.language(), plain.language());
}

#[test]
fn merge_ors_the_sampling_fact_across_results() {
  // The VAD-chunk instance of finding 2: a chunk the blank-audio drop
  // emptied contributes NO segments, so its accepted temperature is nowhere
  // in the merged segment list. The merge has to carry the fact out of it
  // anyway, or the merged transcript looks greedy and claims a
  // byte-reproducibility it cannot honor.
  let greedy = TranscriptionResult::new(
    "Hello",
    vec![TranscriptionSegment::new().with_temperature(0.0)],
    "en",
    TranscriptionTimings::new(),
  );
  // The emptied chunk: zero segments, and the only witness to its own
  // sampling is the flag itself.
  let emptied = TranscriptionResult::new("", Vec::new(), "en", TranscriptionTimings::new())
    .with_sampled_at_nonzero_temperature();
  assert!(emptied.segments_slice().is_empty());

  let merged = merge_transcription_results(&[greedy.clone(), emptied.clone()]);
  assert!(
    merged
      .segments_slice()
      .iter()
      .all(|segment| segment.temperature() == 0.0),
    "no surviving segment carries the evidence"
  );
  assert!(
    merged.sampled_at_nonzero_temperature(),
    "and yet the merge must still know"
  );

  // Same through the options-aware door `WhisperKit::transcribe` actually uses.
  assert!(
    merge_transcription_results_with_options(&[greedy.clone(), emptied], &DecodingOptions::new())
      .sampled_at_nonzero_temperature()
  );

  // All-greedy merges stay honest in the other direction.
  assert!(!merge_transcription_results(&[greedy.clone(), greedy]).sampled_at_nonzero_temperature());
  assert!(!merge_transcription_results(&[]).sampled_at_nonzero_temperature());
}
