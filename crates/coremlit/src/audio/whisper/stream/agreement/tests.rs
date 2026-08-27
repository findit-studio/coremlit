use super::*;
use crate::audio::whisper::{
  result::{TranscriptionResult, TranscriptionSegment, TranscriptionTimings, WordTiming},
  task_facts::TaskFacts,
};

fn word(text: &str, start: f32, end: f32) -> WordTiming {
  WordTiming::new(text, vec![start as u32 + 1], start, end, 0.9)
}

/// The instant an empty-holdback advance anchors at when the word it settles
/// last is ZERO-DURATION and sits at 1.0 s: the exact time of sample 16001, the
/// first sample strictly after the one 1.0 s clips to.
///
/// Written as the sample's own instant rather than by calling the engine's
/// `past_the_settled_instant`, so these assertions state which SAMPLE the
/// anchor has to clear rather than repeating the arithmetic that produced it.
/// The `f32::next_up` anchor this replaced was `1.0000001` — a real step in
/// seconds and no step at all in samples, so the next stride re-read the
/// settled word's own audio while the module doc claimed it had been clipped
/// away (#94, codex round 7 on PR #95, finding 2). The pin that the two
/// coordinate systems agree is
/// `the_watermark_clears_the_settled_sample_not_just_the_settled_instant`.
const PAST_ONE_SECOND: f32 = 1.0000625;

/// The same anchor for a zero-duration settled word at 2.0 s — sample 32001's
/// own instant. See [`PAST_ONE_SECOND`].
const PAST_TWO_SECONDS: f32 = 2.0000625;

/// A word carrying `token_count` distinct plain-vocabulary token ids — for the
/// prefill-budget cases, where what matters is how many TOKENS the holdback
/// costs, not how many words it holds. Ids stay far below any Whisper
/// vocabulary's special range, so `prefill_tokens`' other reduction cannot
/// confuse the measurement.
fn word_of_tokens(text: &str, start: f32, end: f32, token_count: usize) -> WordTiming {
  let first = start as u32 * 1000 + 1;
  let tokens: Vec<u32> = (0..token_count as u32).map(|index| first + index).collect();
  WordTiming::new(text, tokens, start, end, 0.9)
}

/// The tiny model's tokenizer — the same artifact `decode`'s own tests read, and
/// the only way to assert what `decode_text` is actually handed rather than what
/// [`DecodingOptions`] merely records.
fn tiny_tokenizer() -> crate::audio::whisper::tokenizer::WhisperTokenizer {
  let root = std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    std::path::PathBuf::from,
  );
  crate::audio::whisper::tokenizer::WhisperTokenizer::from_folder(
    root.join("tokenizers/whisper-tiny"),
  )
  .unwrap()
}

/// The initial prompt `WhisperKit::transcribe` builds from `options`, exactly as
/// it builds it (`transcribe/mod.rs:394-405`): `[<|startoftranscript|>]` unless
/// [`DecodingOptions::use_prefill_prompt`] is set, in which case
/// [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens) derives the
/// whole thing. This is the layer the prefill contract has to be pinned at —
/// `DecodingOptions::prefix_tokens` is only what the engine RECORDED, and
/// `prefill_tokens` is free to trim and filter it before `decode_text` ever sees
/// a token.
fn initial_prompt_for(
  options: &DecodingOptions,
  tokenizer: &crate::audio::whisper::tokenizer::WhisperTokenizer,
) -> Vec<u32> {
  if options.use_prefill_prompt() {
    crate::audio::whisper::decode::prefill_tokens(options, tokenizer, true)
  } else {
    vec![tokenizer.special_tokens().start_of_transcript_token()]
  }
}

// NOTE: this task's own brief's literal snippet called `TranscriptionResult::
// new()` with no arguments, then chained `.set_segments(...)`/
// `.set_language(...)`. The shipped constructor is four-argument
// (`TranscriptionResult::new(text, segments, language, timings)` — that
// type's own doc: "Builds a result from its four required fields ... has no
// defaults for these either") — same brief-vs-shipped-API fix as
// `tests/pipeline.rs`'s `tiny_options`/`tests/parity_jfk.rs`. Both call sites
// below pass the real values directly instead.
/// Ingest the way [`LocalAgreementTranscriber::push_samples`] does. The driver
/// retargets its options through
/// [`LocalAgreement::decoding_options_for_next`] between strides; the engine
/// itself takes only the result, so this is a one-line alias kept for the
/// streaming-shaped tests to read as what they model.
trait IngestStreamed {
  fn ingest_streamed(&mut self, result: TranscriptionResult) -> AgreementOutcome;
}

impl IngestStreamed for LocalAgreement {
  fn ingest_streamed(&mut self, result: TranscriptionResult) -> AgreementOutcome {
    self.ingest(result)
  }
}

fn result_with_words(words: Vec<WordTiming>) -> TranscriptionResult {
  let mut segment = TranscriptionSegment::new();
  segment
    .set_start(0.0)
    .set_end(words.last().map_or(0.0, |w| w.end()));
  segment.set_text(
    words
      .iter()
      .map(|w| w.word().to_string())
      .collect::<String>(),
  );
  segment.set_words(words);
  TranscriptionResult::new("", vec![segment], "en", TranscriptionTimings::new())
}

#[test]
fn agreement_confirms_the_common_prefix_minus_the_agreed_tail() {
  // TranscribeCLI.swift:370-394 with agreementCountNeeded = 2.
  let mut agreement = LocalAgreement::new();
  let first = result_with_words(vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.7),
    word(" my", 0.7, 1.0),
  ]);
  assert!(
    agreement.ingest_streamed(first).is_awaiting_agreement(),
    "first result: nothing to agree with"
  );
  assert_eq!(
    agreement.results_slice().len(),
    1,
    "first result IS appended (:408-410)"
  );

  let second = result_with_words(vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.7),
    word(" my", 0.7, 1.0),
    word(" fellow", 1.0, 1.5),
  ]);
  assert!(agreement.ingest_streamed(second).is_advanced());
  assert_eq!(agreement.results_slice().len(), 2);
  // common = [And, so, my]; last agreed = suffix(2) = [so, my];
  // confirmed += prefix(1) = [And]; watermark = " so".start.
  assert_eq!(agreement.confirmed_words_slice().len(), 1);
  assert_eq!(agreement.confirmed_words_slice()[0].word(), " And");
  assert!((agreement.last_agreed_seconds() - 0.4).abs() < 1e-6);

  // Options for the next stride carry the watermark + agreed prefix tokens
  // (:364-367).
  let next =
    agreement.decoding_options_for_next(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(next.clip_timestamps_slice(), &[0.4]);
  assert_eq!(next.prefix_tokens_slice().len(), 2);
}

#[test]
fn disagreement_skips_the_result_and_keeps_the_watermark() {
  // TranscribeCLI.swift:395-400 (skipAppend).
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.7),
  ]));
  let disagreeing = result_with_words(vec![word(" But", 0.0, 0.4), word(" then", 0.4, 0.7)]);
  assert!(
    agreement
      .ingest_streamed(disagreeing)
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.results_slice().len(),
    1,
    "disagreeing result NOT appended"
  );
  assert_eq!(agreement.last_agreed_seconds(), 0.0);
  assert!(agreement.confirmed_words_slice().is_empty());
}

#[test]
fn wordless_results_are_flagged_but_still_appended() {
  // TranscribeCLI.swift:403-409.
  let mut agreement = LocalAgreement::new();
  let mut segment = TranscriptionSegment::new();
  segment.set_text("hi");
  let wordless = TranscriptionResult::new("hi", vec![segment], "en", TranscriptionTimings::new());
  assert!(agreement.ingest_streamed(wordless).is_no_word_timings());
  assert_eq!(agreement.results_slice().len(), 1);
}

#[test]
fn finalize_appends_agreed_tail_plus_different_suffix_and_merges() {
  // TranscribeCLI.swift:418-421.
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.7),
    word(" my", 0.7, 1.0),
  ]));
  agreement.ingest_streamed(result_with_words(vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.7),
    word(" my", 0.7, 1.0),
    word(" fellow", 1.0, 1.5),
  ]));
  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  // confirmed [And] + lastAgreed [so, my] + differentSuffix(prev, hyp) [fellow]
  assert_eq!(final_result.text(), " And so my fellow");
  assert_eq!(final_result.language(), "en");
  assert_eq!(
    final_result.segments_slice().len(),
    2,
    "merged from the two appended results"
  );
}

#[test]
fn finalize_threads_options_so_dropped_ids_survive() {
  // F5 (codex round 3), the finalize half. `finalize` delegated to the
  // options-blind confirmed-words merge, so the default streaming path lost a
  // survivor id gap [0, 2] back to a dense [0, 1] at finalization. Threading
  // the driver's own options through must preserve it.
  let seg = |id: usize, start: f32, end: f32| {
    let mut s = TranscriptionSegment::new();
    s.set_id(id).set_start(start).set_end(end);
    s
  };
  // One ingested result carrying an internal dropped-id gap [0, 2] (a
  // wordless result is still appended on first ingest -- see
  // `wordless_results_are_flagged_but_still_appended`).
  let result = TranscriptionResult::new(
    "A B",
    vec![seg(0, 0.0, 1.0), seg(2, 1.0, 2.0)],
    "en",
    TranscriptionTimings::new(),
  );
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result);
  assert_eq!(
    agreement.results_slice().len(),
    1,
    "first result is appended"
  );

  // drop-ON (the default): the gap must survive finalization.
  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result
      .segments_slice()
      .iter()
      .map(TranscriptionSegment::id)
      .collect::<Vec<_>>(),
    vec![0, 2],
    "finalize must pass drop_blank_audio through, not collapse [0, 2] to [0, 1]"
  );
}

#[test]
fn agreement_count_needed_is_configurable() {
  // The brief's own tests only ever exercise the default of 2
  // (DEFAULT_AGREEMENT_COUNT_NEEDED); this pins that the options-pattern
  // knob itself actually changes ingest's threshold, not just its
  // constructor/accessor plumbing.
  let mut agreement = LocalAgreement::new().with_agreement_count_needed(1);
  assert_eq!(agreement.agreement_count_needed(), 1);
  agreement.ingest_streamed(result_with_words(vec![word(" And", 0.0, 0.4)]));
  let second = result_with_words(vec![word(" And", 0.0, 0.4), word(" so", 0.4, 0.7)]);
  // A single-word common prefix ([And]) already meets a threshold of 1 —
  // it would NOT at the default threshold of 2.
  assert!(agreement.ingest_streamed(second).is_advanced());
  assert!(agreement.confirmed_words_slice().is_empty());
  assert_eq!(agreement.last_agreed_words_slice().len(), 1);
  assert_eq!(agreement.last_agreed_words_slice()[0].word(), " And");
}

#[test]
fn agreement_count_needed_zero_is_clamped_to_one_and_never_panics() {
  // Regression (self-review, Critical): `common[split..]` with `split ==
  // common.len()` is always empty when agreement_count_needed is 0, so an
  // unclamped 0 would index `last_agreed_words[0]` on an empty Vec inside
  // `ingest` and panic. Swift's hardcoded `agreementCountNeeded = 2`
  // (`TranscribeCLI.swift:349`) never exposes this knob, so it never
  // reaches this state; this port's builder/setter do expose it, so the
  // setter clamps instead.
  let mut agreement = LocalAgreement::new().with_agreement_count_needed(0);
  assert_eq!(agreement.agreement_count_needed(), 1);
  agreement.ingest_streamed(result_with_words(vec![word(" And", 0.0, 0.4)]));
  let second = result_with_words(vec![word(" And", 0.0, 0.4), word(" so", 0.4, 0.7)]);
  agreement.ingest_streamed(second); // must not panic
}

#[test]
fn later_segment_words_satisfy_the_any_segment_gate() {
  // Review follow-up pinning the documented deviation (module doc): the
  // gate is "ANY segment carries words", not Swift's first-segment-only
  // nil probe — a wordless first segment with a worded second one must
  // NOT be flagged NoWordTimings.
  let mut wordless = TranscriptionSegment::new();
  wordless.set_start(0.0).set_end(0.5);
  let mut worded = TranscriptionSegment::new();
  worded
    .set_start(0.5)
    .set_end(1.0)
    .set_words(vec![word(" hi", 0.5, 1.0)]);
  let result = TranscriptionResult::new(
    "",
    vec![wordless, worded],
    "en",
    TranscriptionTimings::new(),
  );
  let mut agreement = LocalAgreement::new();
  let outcome = agreement.ingest_streamed(result);
  assert!(
    !outcome.is_no_word_timings(),
    "any-segment gate: later words count"
  );
}

#[test]
fn tied_word_starts_never_confirm_twice() {
  // Regression (phase-gate round 1): the timestamp-only watermark
  // re-admitted already-confirmed words whose start ties the watermark
  // (B holds the watermark at A's shared start), confirming A again on
  // the next pass. Three-pass history from the finding, agreement 2.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
  let c = || word(" C", 1.0, 2.0);
  let d = || word(" D", 2.0, 3.0);
  let e = || word(" E", 3.0, 4.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c(), d()]));
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c(), d(), e()]));
  let confirmed: Vec<&str> = agreement
    .confirmed_words_slice()
    .iter()
    .map(|w| w.word())
    .collect();
  assert_eq!(confirmed, vec![" A", " B"], "confirmed once and stable");
  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  for token in ["A", "B", "C", "D", "E"] {
    assert_eq!(
      text.matches(token).count(),
      1,
      "{token} must appear exactly once in {text:?}"
    );
  }
}

#[test]
fn omitting_a_confirmed_tied_word_does_not_drop_provisional_words() {
  // Regression (phase-gate round 2): the count-based readmit skip dropped
  // B whenever a rewrite OMITTED confirmed A (tied start) and shifted the
  // hypothesis left. Rule W keeps the tie out of the offered list in the first
  // place -- it widens the split past the tie, so the watermark is 1.0 and
  // neither A nor B is ever re-offered -- so there is nothing here for a
  // rewrite to shift into.
  //
  // Mutation proof: delete Rule W's widening loop (the `while split <
  // common.len()` in `ingest`'s advance) and B is never confirmed -- the
  // re-admitted A sits at the front of BOTH filtered lists, the prefix
  // comparison lines up on it instead of on B, and `confirmed_words_slice()`
  // reads back [" A"]. The same mutation costs
  // `tied_word_starts_never_confirm_twice` its stability, reading back
  // [" A", " A", " B"].
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
  let c = || word(" C", 1.0, 2.0);
  let d = || word(" D", 2.0, 3.0);
  let e = || word(" E", 3.0, 4.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c(), d()])); // confirms A, holds B,C
  // The rewrite omits A entirely: B must survive to be confirmed next.
  agreement.ingest_streamed(result_with_words(vec![b(), c(), d(), e()]));
  let confirmed: Vec<&str> = agreement
    .confirmed_words_slice()
    .iter()
    .map(|w| w.word())
    .collect();
  assert!(
    confirmed.contains(&" B"),
    "B lost to a re-admitted tied word: {confirmed:?}"
  );
  assert_eq!(
    confirmed.iter().filter(|w| **w == " B").count(),
    1,
    "and confirmed exactly once"
  );
  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  for token in ["A", "B", "C", "D", "E"] {
    assert_eq!(text.matches(token).count(), 1, "{token} once in {text:?}");
  }
}

#[test]
fn the_split_never_cuts_at_a_tied_start() {
  // RULE W's POSTCONDITION, swept rather than pinned to one fixture, and TOTAL:
  // the last CONFIRMED word starts strictly before the watermark, with NO
  // condition on the holdback.
  //
  // That inequality is the whole of #94. `watermark_filtered` offers every
  // hypothesis word whose `start >= watermark`, so a confirmed word that TIES
  // the watermark passes that filter and can come back at the head of the next
  // hypothesis -- and there it is byte-identical to the stream's own second
  // occurrence of the same text, which is the issue's impossibility result.
  // Every defeated rule in this module's ledger tried to DECIDE that state.
  // Rule W refuses to create it: the split lands only on a boundary whose
  // preceding word starts strictly earlier, and where no such boundary exists at
  // or after the requested split it backs OFF rather than widening off the end.
  //
  // THE SHAPE THIS TEST USED TO HAVE SKIPPED THE EMPTY HOLDBACK, and the state
  // it skipped is the one the property did not hold in: with nothing held back
  // the watermark came from `common.last().end()`, which for a zero-duration
  // word is that word's own start (codex round 1 on PR #95 -- the skip read as
  // coverage). The postcondition is now asserted on EVERY observation with a
  // non-empty confirmed list, and the sweep DRIVES the empty holdback rather
  // than avoiding it: some trials raise the prefill cost of one word past
  // `MAX_HOLDBACK_PREFILL_TOKENS`, and some build a TIED RUN whose tokens
  // exceed it in aggregate -- the two shapes that empty the holdback -- and
  // `empty_holdbacks` below is the non-vacuity proof that it really happened.
  //
  // The shape then still had ONE oversized word as its only route past the
  // budget, and that could not build the AGGREGATE trigger (codex round 3 on
  // PR #95): a TIED RUN of ordinary words whose TOTAL exceeds the budget, where
  // the floor lands inside the run, every reachable boundary ties, and the split
  // runs off the end of `common`. Measured on the shape this test had before:
  // 25 such rounds of 1662 observations, all of them reached through the
  // oversized word rather than through a run, and none of them DISTINGUISHED.
  // `aggregate_fallbacks` is that half's own non-vacuity proof, and it counts
  // the ROUTE rather than the arm: a fallback round in which no word of `common`
  // is over budget on its own can only have got there through the run.
  //
  // A SECOND POSTCONDITION now rides the same sweep, and reaching the state it
  // speaks about needed one more shape change (codex round 3 on PR #95, second
  // finding). `nothing_unconfirmed_falls_below_the_watermark` is the claim that
  // an advance may not push the watermark past a word of its own hypothesis it
  // did not confirm; the state that tests it is the FORCED arm reached with a
  // live suffix beyond `common`, and offering every stride TWICE could not build
  // it -- measured at 0 rounds of 1353. `repeats` below offers a stride once in
  // three, which leaves the next stride's first ingest comparing consecutive
  // GROWING lists, and `forced_strands` is that half's non-vacuity proof.
  //
  // A FOURTH shape change makes two consecutive hypotheses CONTRADICT each
  // other past the agreed prefix (codex round 4 on PR #95). Everything above
  // offers PREFIXES, so every hypothesis is a prefix of the next and `common`
  // grows on every round; `alternating` rewrites the LAST offered word on every
  // other offering, which pins `common` in front of it and holds the engine on
  // the fallback arm round after round. `contradictions` is that half's
  // non-vacuity proof, and `tie_strands` proves the second postcondition's one
  // exception is reached rather than merely written down.
  //
  // A FIFTH drifts a whole offering's TIMINGS while leaving its texts alone
  // (`retiming`), so the normalized prefix two hypotheses agree on is unchanged
  // while the instant an empty-holdback advance anchors past MOVES.
  // `drifted_advances` is its non-vacuity proof. This half was added for a
  // deferral bound that no longer exists (codex round 5 on PR #95, finding 2)
  // and is KEPT because the shape is otherwise unswept: it is also the shape
  // this module's residual 6 records as unconstrained -- at 0.03 s per offering
  // the drift stays inside the gap in front of the watermark, so the sweep
  // reaches the re-timing without ever reaching the whole-run jump past it.
  //
  // A SIXTH shape draws BACKWARDS starts (codex round 7 on PR #95, finding 1).
  // Every shape above keeps its starts non-decreasing, which is the premise the
  // second postcondition used to be free under on an interior split -- and
  // `segment::update_segments_with_word_timings` does not honour it. The
  // backwards draw reproduces that function's own clamp; `backward_truths` is
  // this half's non-vacuity proof, and `backward_strands` proves the half of the
  // exception only a backwards start can reach.
  //
  // Measured on the shape below, at 512 trials: 477 tied truths, 2488
  // observations, 410 empty holdbacks, 140 fallback rounds (40 of them through
  // the aggregate route), 28 forced-arm rounds with a live suffix, 1041
  // contradicting rounds, 166 backwards truths, 26 tie strands, 4 backwards
  // strands, 696 drifted advances, and 13 rounds that erased a word from the
  // published transcript.
  //
  // The DEFERRAL comparison that removed the wait was measured on the draw this
  // test carried at `4b259ef`, BEFORE the backwards half existed: 26 words
  // erased from the published transcript against this fallback's 10 on that
  // identical draw, and 29 tie strands against 38. Adding a shape re-rolls every
  // later draw, so those two columns cannot be re-derived from the numbers
  // above and are not restated as if they could; the decision they support is
  // in the engine module's doc, "Why there is no deferral", with the tree it
  // compared against.
  //
  // The sweep draws starts from a coarse grid whose repeats make consecutive
  // words tie -- the `a=[0,0.5)/b=[0,1.0)` shape both #94 regressions are built
  // from, generalized -- and drives growing prefixes through `ingest_streamed`,
  // the MARKED path the driver uses, so every advance carries the prefill
  // premise the engine issues for itself. The draw is non-decreasing BEFORE the
  // backwards clamp, the way `find_alignment` guarantees (`segment::tests`:
  // `w[i].end() <= w[i+1].start() + 1e-4`), and the clamp is the only thing that
  // breaks it -- which is the pipeline's own arrangement. One stride in four
  // omits the leading word, which is the rewrite
  // `omitting_a_confirmed_tied_word_does_not_drop_provisional_words` is built
  // from.
  //
  // Mutation proof, every row RUN on this shape. Make the boundary non-strict
  // (`<=` for `<` in `split_at_a_strict_boundary`) and the FIRST postcondition
  // reds on a swept tie at trial 0. Drop the
  // `.max(past_the_settled_instant(settled_high))` from `empty_holdback_anchor`
  // and it reds on a swept over-budget zero-duration word. Keep the sparing fold
  // but drop its `*start > settled_high` filter and it reds again, the watermark
  // having landed on a confirmed word's own start. Return `common.len() - 1`
  // from the final `unwrap_or` and it reds a third time. It does NOT red when
  // the back-off arm is deleted outright: the empty-holdback anchor still holds
  // the postcondition on that path, which is why the back-off is pinned in
  // `a_trailing_tied_run_never_confirms_itself_twice_at_the_default_count`
  // (measured: that test alone reds, 384 of 385 still green). It also does not
  // red when the boundary is read against the ADJACENT predecessor rather than
  // the running maximum: nothing this shape draws puts two backwards steps close
  // enough together to need the difference, and
  // `a_backwards_start_two_words_back_still_cannot_be_re_admitted` is that
  // mutation's sole falsifier.
  //
  // The SECOND postcondition is one clause now, and the ARM gate that used to
  // ride beside it is gone. Return `anchor` from `sparing_watermark` and it reds
  // on a strand ABOVE the highest confirmed start. Confirm one word FEWER on the
  // advance (`common[..split.saturating_sub(1)]`) and it reds again, on the word
  // the round declined to settle. The gate said WHICH ARM may strand -- only a
  // split that ran off the end of `common` -- and a backwards start makes that
  // false: an interior split confirms past a word behind it and strands it too
  // (`a_backward_start_from_the_segment_pipeline_does_not_strand_a_later_word`
  // is the pipeline-built witness). What actually bounds the exception is the
  // highest confirmed start, on either arm, so that is what is asserted.
  //
  // The SHAPE rows, each forced off and re-measured. `aggregate` off:
  // `aggregate_fallbacks` reds at `0 of 134`, the tied-run route to the empty
  // holdback being unreachable while the oversized-word route still supplies
  // 134. `retiming` off: `drifted_advances` reds at `0 rounds`. `alternating`
  // off: `contradictions` reds at `18 rounds` against its floor of 128 -- but
  // only with `nothing_the_stream_still_says_is_erased` neutralized first, since
  // that helper fires at trial 92 on a re-timing its `offered` membership test
  // cannot tell from an erasure (the transcript keeps one `(" A", 0.5)` and the
  // result offers one; the multiset difference blames the retained copy). The
  // `alternating` shape hides that on the shipped draw. `repeats` pinned at 2 --
  // the row that used to red `forced_strands` at `0` -- now reds NOTHING:
  // `forced_strands` measures 32 against its floor of 4, the forced arm with a
  // live suffix being reachable without the one-offering cadence once an
  // agreeing round always advances. The cadence is kept anyway, being the
  // driver's own; the row is recorded as DOMINATED rather than deleted.
  const TEXTS: [&str; 4] = [" A", " B", " C", " D"];
  // Repeated 0.0 entries are the ties; the rest keep the grid coarse enough for
  // two words to share an instant often.
  const STEPS: [f32; 7] = [0.0, 0.0, 0.0, 0.1, 0.5, 0.5, 1.0];
  // The 0.0 duration is a zero-length word, which is the one shape that could
  // satisfy `start >= watermark` from inside the confirmed list without a tie
  // between two distinct starts. The 0.55 is the BACKWARDS draw's own entry: the
  // segment-start clamp fires only for a word spanning more than half a second
  // and only moves it back by `CLAMP_MEDIAN - duration`, so 0.55 is the one
  // duration here that both opens the branch and lands behind where it started
  // -- by 0.15 s, which is wider than the 0.1 STEP so the clamped word can duck
  // below a word the same round confirms rather than merely tying it.
  const DURATIONS: [f32; 5] = [0.0, 0.2, 0.5, 0.55, 1.0];
  /// `calculate_word_duration_constraints`' own ceiling (`median.min(0.7)`), and
  /// the value `SegmentSeeker.swift:635-640` subtracts from a drifted first
  /// word's END to get its new start.
  const CLAMP_MEDIAN: f32 = 0.7;

  /// `None` when nothing is confirmed yet and the postcondition has nothing to
  /// speak about; otherwise `Some(the holdback was empty)`, which is the state
  /// this test's earlier shape skipped.
  fn postcondition(agreement: &LocalAgreement, trial: u32, stride: usize) -> Option<bool> {
    let high = highest_start(agreement.confirmed_words_slice())?;
    assert!(
      high < agreement.last_agreed_seconds(),
      "trial {trial}, stride {stride}: the confirmed list {:?} reaches {high}, \
       which is not strictly before the {} s watermark -- that word passes \
       `watermark_filtered`'s own `start >= watermark` and can be re-admitted. \
       The holdback here is {:?}",
      confirmed_texts(agreement),
      agreement.last_agreed_seconds(),
      held_back_texts(agreement),
    );
    Some(agreement.last_agreed_words_slice().is_empty())
  }

  /// RULE W'S SECOND POSTCONDITION, and the one the forced empty holdback broke
  /// (codex round 3 on PR #95, second finding): an advance may not push the
  /// watermark past a word of its own hypothesis that it did not CONFIRM.
  ///
  /// Every word of `hypothesis_words` past the split is either a word this
  /// round appended to the confirmed list or a word the stream can still offer
  /// again; anything else is a strand. It is TOTAL for the same reason the first
  /// postcondition is: a round that does not advance appends nothing and moves
  /// no watermark, and every word of `hypothesis_words` cleared `start >=
  /// last_agreed_seconds` to be there at all, so there is nothing below it.
  ///
  /// `appended` is measured across the call rather than read off a split, so
  /// this cannot be satisfied by the same arithmetic that produced it. The
  /// unconfirmed words are `hypothesis_words[appended..]` exactly: `common` is a
  /// PREFIX of `hypothesis_words` and the advance appends `common[..split]`, so
  /// the index is the split. It used to be read as a filtered PREFIX instead,
  /// which was only equivalent while starts were non-decreasing.
  ///
  /// IT HAS ONE EXCEPTION, and the exception is as narrow as the impossibility
  /// that forces it: a stranded word may sit at or below the HIGHEST confirmed
  /// start. There no instant serves both -- every watermark strictly past that
  /// start (which the first postcondition requires) filters the strand, and
  /// every watermark that spares the strand re-admits the settled word. It is
  /// this module's residual 1, and `sparing_watermark` is what keeps every other
  /// overlap out of it.
  ///
  /// The exception is read at `<=` rather than `==` because BOTH halves are
  /// reachable and they are reached differently. AT the highest start is the TIE
  /// -- a zero-duration word an empty-holdback advance settles last, which is
  /// what residual 1 has always been about. BELOW it is a BACKWARDS start:
  /// `segment::update_segments_with_word_timings` can put a later word behind an
  /// earlier one (codex round 7 on PR #95, finding 1), and an interior split can
  /// then confirm past it. `tie_strands` and `backward_strands` count the two
  /// separately so neither can pass by being unreachable.
  ///
  /// The gate used to read "this round ESCAPED a repeating deferral" (codex
  /// round 4 on PR #95), then "this round's split ran off the END of `common`".
  /// The ARM gate is gone: with backwards starts an INTERIOR split strands too,
  /// so the arm was never what bounded this -- the highest confirmed start is.
  /// Returns whether the exception was taken.
  fn nothing_unconfirmed_falls_below_the_watermark(
    agreement: &LocalAgreement,
    appended: usize,
    trial: u32,
    stride: usize,
  ) -> Option<bool> {
    let watermark = agreement.last_agreed_seconds();
    let stranded: Vec<(&str, f32)> = agreement.hypothesis_words[appended..]
      .iter()
      .filter(|word| word.start() < watermark)
      .map(|word| (word.word(), word.start()))
      .collect();
    if stranded.is_empty() {
      return None;
    }
    let settled = highest_start(agreement.confirmed_words_slice());
    assert!(
      stranded
        .iter()
        .all(|(_, start)| settled.is_some_and(|settled| *start <= settled)),
      "trial {trial}, stride {stride}: {stranded:?} of the hypothesis are \
       below the {watermark} s watermark and were NOT confirmed this round \
       (which confirmed {appended}) -- they are STRANDED: the next worded \
       ingest filters them out of both hypotheses at once and `finalize` can no \
       longer reach them, after this round's own `finalize` already published \
       them. Only a word at or below the highest confirmed start {settled:?} \
       may go that way, which is where no watermark serves both claims",
    );
    Some(
      stranded
        .iter()
        .any(|(_, start)| settled.is_some_and(|settled| *start == settled)),
    )
  }

  /// THE PUBLISHED TRANSCRIPT, with its timings — `LocalAgreement::finalize`'s
  /// own word list rather than its merged text, which is why
  /// `take_finalized_words` exists as a function at all.
  fn published(agreement: &LocalAgreement) -> Vec<(String, f32)> {
    agreement
      .clone()
      .take_finalized_words()
      .into_iter()
      .map(|word| (word.word().to_string(), word.start()))
      .collect()
  }

  /// THE TRANSCRIPT'S OWN RETENTION, swept across every round (#94, codex round
  /// 5 on PR #95, closing note). Both findings that round raised hid in the same
  /// gap: `LocalAgreement::confirmed_words_slice` is append-only, so a word that
  /// reaches the transcript through a round's PROVISIONAL tail and then
  /// disappears is invisible to every property written over the confirmed list —
  /// and disappearing is exactly what a strand does one round after the advance
  /// that stranded it, when `watermark_filtered` drops it from both hypotheses
  /// at once.
  ///
  /// Plain monotonicity is FALSE here and must not be asserted: revising an
  /// unconfirmed word is the whole point of LocalAgreement-2, and a revision
  /// removes the reading it replaces — including by RE-TIMING it, which this
  /// module's residual 3 already names. So the claim is scoped to the words the
  /// newest result STILL SAYS: a word the transcript published, that this very
  /// result offers again at the same text and instant, and that the watermark
  /// has since put out of reach. Nothing revised it; the engine simply cannot
  /// see it any more, and no later `finalize` can reach it.
  ///
  /// Each such word must sit at or below the HIGHEST confirmed start, the range
  /// where no watermark both clears every confirmed start (#94) and spares the
  /// strand — this module's residual 1. Anything above that is a word the engine
  /// erased while it could still have held it. It reads the high-water start
  /// rather than the last confirmed word's because word starts inside one
  /// hypothesis are not non-decreasing (codex round 7 on PR #95, finding 1).
  ///
  /// **What this does NOT catch, stated rather than implied.** Every retraction
  /// an empty-holdback advance can cause is AT the settled instant, so all of
  /// them are identical in SHAPE to the residual-1 retraction this permits, and
  /// no property over the transcript alone can separate a necessary one from an
  /// avoidable one. What this DOES buy is the bound: retraction is confined to
  /// that one instant on every round of every shape the sweep drives, and
  /// COUNTED so the confinement is not vacuous. The count is also the number
  /// that removed the deferral — 10 erasures here against the deferral's 26 on
  /// the identical draw (see the engine module's doc, "Why there is no
  /// deferral").
  ///
  /// **It is DOMINATED, and by which assertion.** Every transcript
  /// `LocalAgreement::finalize` publishes is a subset of
  /// `confirmed_words ++ hypothesis_words` -- the holdback and
  /// `find_longest_different_suffix`'s output are both slices of the latter --
  /// so a word this can see leaving the transcript was in `hypothesis_words`
  /// below the new watermark on the round it left, which is exactly what
  /// `nothing_unconfirmed_falls_below_the_watermark` reads one step earlier.
  /// This is kept because it is the SOLE DISCRIMINATOR once that assertion is
  /// neutralized -- drop the sparing fold from `empty_holdback_watermark` and
  /// neutralize it, and this reds on `[(" C", 2.0), (" C", 2.0)]` erased below a
  /// 2.5 s watermark at a 1.5 s settled instant -- and because it reads
  /// `finalize`'s OWN output rather than the engine's internal word lists, so a
  /// defect on the publication path has somewhere to show. Conditional on
  /// mutants, not on inputs; the same standing the exception clauses above have.
  ///
  /// Returns whether an erasure happened.
  fn nothing_the_stream_still_says_is_erased(
    before: &[(String, f32)],
    after: &[(String, f32)],
    offered: &[(String, f32)],
    agreement: &LocalAgreement,
    trial: u32,
    stride: usize,
  ) -> bool {
    let watermark = agreement.last_agreed_seconds();
    let settled = highest_start(agreement.confirmed_words_slice());
    // MULTISET, not set: the sweep's alphabet is four texts over a coarse grid,
    // so one transcript can carry the same (text, start) twice and a set-shaped
    // comparison would call the second copy retained by the first.
    let mut kept: Vec<&(String, f32)> = after.iter().collect();
    let erased: Vec<&(String, f32)> = before
      .iter()
      .filter(|word| match kept.iter().position(|held| held == word) {
        Some(index) => {
          kept.swap_remove(index);
          false
        }
        None => true,
      })
      .filter(|(_, start)| *start < watermark)
      .filter(|word| offered.contains(word))
      .collect();
    assert!(
      erased
        .iter()
        .all(|(_, start)| settled.is_some_and(|settled| *start <= settled)),
      "trial {trial}, stride {stride}: {erased:?} left the published transcript \
       while this very result still offers them, and they sit below the \
       {watermark} s watermark -- so nothing revised them, no hypothesis can \
       carry them again and no `finalize` can reach them, after this engine had \
       already published them. Only a word at or below the highest confirmed \
       start {settled:?} may go that way. Transcript before: {before:?}; after: \
       {after:?}; offered: {offered:?}; confirmed: {:?}",
      confirmed_texts(agreement),
    );
    !erased.is_empty()
  }

  let mut state: u64 = 0x2545_F491_4F6C_DD1D;
  let mut next = move || {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
  };
  let mut checked = 0u32;
  let mut empty_holdbacks = 0u32;
  let mut tied_truths = 0u32;
  let mut forced_strands = 0u32;
  let mut contradictions = 0u32;
  let mut fallbacks = 0u32;
  let mut aggregate_fallbacks = 0u32;
  let mut tie_strands = 0u32;
  let mut backward_strands = 0u32;
  let mut backward_truths = 0u32;
  let mut drifted_advances = 0u32;
  let mut erasures = 0u32;

  for trial in 0..512u32 {
    let length = 4 + (next() % 5) as usize;
    // TWO ways to blow the prefill budget, and the sweep drives BOTH.
    //
    // ONE OVERSIZED WORD is the route residual 1 needs: it is the only thing
    // that can still leave the holdback EMPTY, and it is placed anywhere,
    // including last, where the budget floor reaches `common.len()`.
    //
    // The AGGREGATE route needs no oversized word at all, and this test could
    // not reach it before (codex round 3 on PR #95): ORDINARY words whose TIED
    // RUN totals more than the budget put the floor strictly INSIDE the run, so
    // every boundary the forward search and the back-off can reach ties while
    // split 0 -- the boundary a tied run always leaves legal -- is below the
    // floor. `AGGREGATE_TOKENS` is sized so any THREE such words exceed
    // `MAX_HOLDBACK_PREFILL_TOKENS` and any two fit, which puts the floor two
    // words from the end of every `common` and makes a three-word tie at the
    // tail the trigger. That round runs the split off the END of `common`;
    // `fallbacks` below is its non-vacuity proof.
    const AGGREGATE_TOKENS: usize = MAX_HOLDBACK_PREFILL_TOKENS / 3 + 1;
    let aggregate = next() % 4 == 0;
    let over_budget = if !aggregate && next() % 3 == 0 {
      Some((next() as usize) % length)
    } else {
      None
    };
    let mut truth: Vec<WordTiming> = Vec::with_capacity(length);
    let mut start = 0.0f32;
    for index in 0..length {
      let text = TEXTS[(next() % TEXTS.len() as u64) as usize];
      start += STEPS[(next() % STEPS.len() as u64) as usize];
      let end = start + DURATIONS[(next() % DURATIONS.len() as u64) as usize];
      truth.push(if over_budget == Some(index) {
        word_of_tokens(text, start, end, MAX_HOLDBACK_PREFILL_TOKENS + 1)
      } else if aggregate {
        word_of_tokens(text, start, end, AGGREGATE_TOKENS)
      } else {
        word(text, start, end)
      });
    }
    // THE BACKWARDS DRAW (codex round 7 on PR #95, finding 1). Everything above
    // generates non-decreasing starts, which is the premise Rule W's second
    // postcondition used to be free under on an interior split -- and
    // `segment::update_segments_with_word_timings` does not honour it. Its
    // segment-start preference (`SegmentSeeker.swift:635-640`) pulls a segment's
    // first word back to `end - constrained_median` whenever the segment's own
    // timestamp start ran more than half a second ahead of it, and that lands
    // BEHIND the word in front of it.
    // `a_backward_start_from_the_segment_pipeline_does_not_strand_a_later_word`
    // is the proof built through that function; this reproduces the SHAPE so the
    // sweep can drive it at every split position and every count.
    //
    // The clamp's own arithmetic, not a free-hand backwards step: the branch
    // needs the word to span more than half a second, and it lowers the start to
    // `end - CLAMP_MEDIAN` -- a step back of `CLAMP_MEDIAN - duration`, under
    // 0.2 s, on ONE word per draw. `DURATIONS`' 0.6 is the only entry that both
    // opens the branch and moves the word backwards; the others leave this a
    // no-op, which is why `backward_truths` below is the non-vacuity proof.
    let spin = (next() as usize) % length;
    let clamped_at = (1..truth.len())
      .map(|offset| 1 + (spin + offset) % (truth.len() - 1))
      .find(|&at| {
        truth[at].end() - truth[at].start() > 0.5
          && truth[at].end() - CLAMP_MEDIAN < truth[at].start()
      });
    if next() % 2 == 0
      && let Some(at) = clamped_at
    {
      let target = &truth[at];
      truth[at] = WordTiming::new(
        target.word(),
        target.tokens_slice().to_vec(),
        target.end() - CLAMP_MEDIAN,
        target.end(),
        target.probability(),
      );
      backward_truths += 1;
    }
    if truth
      .windows(2)
      .any(|pair| pair[0].start() >= pair[1].start())
    {
      tied_truths += 1;
    }

    // Both counts the engine can be driven at: 1 makes every agreement an
    // advance, 2 is the driver's own `DEFAULT_AGREEMENT_COUNT_NEEDED`.
    let mut agreement =
      LocalAgreement::new().with_agreement_count_needed(1 + (next() % 2) as usize);
    let alternating = next() % 2 == 0;
    // The SHIFT shape, exclusive with the rewrite above because both act on the
    // same word: `alternating` takes the last offered word OUT of `common` by
    // changing what it says, and this leaves it IN by changing only WHEN it is.
    let retiming = !alternating && next() % 2 == 0;
    let mut offering = 0u32;
    // A fresh engine publishes nothing, so the first round is compared against
    // the empty transcript and can only ADD.
    let mut transcript: Vec<(String, f32)> = Vec::new();
    for stride in 2..=truth.len() {
      let omit_head = stride > 2 && next() % 4 == 0;
      let offered = if omit_head {
        truth[1..stride].to_vec()
      } else {
        truth[..stride].to_vec()
      };
      // ONE offering of a stride, sometimes, instead of two. TWO is what the
      // sweep did everywhere, and it can never put a live suffix at the FORCED
      // arm: the second ingest of a stride compares a hypothesis against itself,
      // so `common` is the whole filtered list with nothing beyond it, and where
      // that list ENDS on the over-budget word the forced arm confirms it there
      // -- one stride BEFORE the growth that would have given it a suffix.
      // Measured on the two-everywhere shape: 0 such rounds in 256 trials.
      //
      // Offering a stride once leaves the next stride's FIRST ingest comparing
      // `truth[..s]` against `truth[..s + 1]`, so `common` ends on `truth[s - 1]`
      // with `truth[s]` beyond it -- the reproduction's own shape, reached
      // whenever the over-budget word lands on that boundary.
      let repeats = if next() % 3 == 0 { 1 } else { 2 };
      for _ in 0..repeats {
        // ALTERNATION, the shape codex round 4 on PR #95 needed and neither
        // half above could build. Growing prefixes and re-offered strides make
        // every hypothesis a PREFIX of the next, so `common` grows on every
        // round and the fallback arm is left almost as soon as it is entered.
        // Rewriting the LAST offered word on every other offering instead makes
        // two consecutive hypotheses CONTRADICT each other there, which pins
        // `common` at everything in front of it for as long as the rewriting
        // continues. `contradictions` below is this half's non-vacuity proof,
        // and `fallbacks` is the proof that the arm it pins the engine on is
        // actually taken.
        let mut offered = offered.clone();
        if alternating
          && offering % 2 == 1
          && let Some(last) = offered.last_mut()
        {
          *last = WordTiming::new(
            " Z",
            last.tokens_slice().to_vec(),
            last.start(),
            last.end(),
            last.probability(),
          );
        }
        // THE SHIFT. `alternating` rewrites the last word's TEXT, which pins
        // `common` in FRONT of it; this rewrites TIMINGS and leaves every text
        // alone, so the words stay inside `common` and the NORMALIZED PREFIX
        // two consecutive hypotheses agree on is unchanged while the instant an
        // empty-holdback advance anchors past MOVES. Added for codex round 5's
        // finding 2 on PR #95 and kept past the deferral it measured: it is the
        // only re-timing this sweep drives, and this module's residual 6 is
        // about the whole-run re-timing it does NOT reach.
        //
        // The WHOLE offering drifts, by a step that only ever grows. Two weaker
        // shapes were tried and are wrong: shifting the last word alone breaks
        // it out of the trailing tie and OPENS a legal boundary above the floor,
        // relieving the state instead of building it; shifting the trailing tied
        // run alternately moves those words BACK again on the next offering,
        // which is this module's residual 3 (drift wider than the gap in front
        // of the watermark) and loses words for a reason that has nothing to do
        // with the split. Drifting everything monotonically keeps
        // every start non-decreasing across offerings as well as within one, so
        // the only thing that moves is the instant itself.
        // OFF the 0.5 grid the starts are drawn from, deliberately: a drift of
        // one grid step makes a re-timed word land exactly where a DIFFERENT
        // word of the round before sat, and every property that identifies a
        // word by its text and instant then confuses the two. A real re-decode's
        // timings are not on the offered grid either.
        let drift = if retiming {
          0.03 * offering as f32
        } else {
          0.0
        };
        if drift > 0.0 {
          for word in &mut offered {
            *word = WordTiming::new(
              word.word(),
              word.tokens_slice().to_vec(),
              word.start() + drift,
              word.end() + drift,
              word.probability(),
            );
          }
        }
        offering += 1;
        // The RAW offering, kept past the ingest that consumes it: what the
        // stream is still saying is half of what separates a revision from an
        // erasure.
        let still_offered: Vec<(String, f32)> = offered
          .iter()
          .map(|word| (word.word().to_string(), word.start()))
          .collect();
        let before = agreement.confirmed_words_slice().len();
        // Captured BEFORE the call, since `ingest` overwrites it: it is one of
        // the inputs the split decision is actually made from.
        let confirmed_last_before = agreement
          .confirmed_words_slice()
          .last()
          .map(WordTiming::start);
        let outcome = agreement.ingest_streamed(result_with_words(offered));
        // The RE-TIMED half's own non-vacuity: a drifted offering must actually
        // reach the advance path, not merely be constructed and disagreed with.
        drifted_advances += u32::from(drift > 0.0 && outcome.is_advanced());
        forced_strands += u32::from(forced_arm_with_a_live_suffix(&agreement));
        let common_len = common_prefix_len(&agreement);
        contradictions += u32::from(
          common_len >= agreement.agreement_count_needed()
            && common_len < agreement.prev_words.len()
            && common_len < agreement.hypothesis_words.len(),
        );
        // EXACT, rather than inferred from the outcome or read off the
        // engine's own holdback: re-ask `split_at_a_strict_boundary` WHERE it
        // put this round's split, from the same inputs the engine handed it.
        // `common.len()` is arm 3 -- the FALLBACK, the one position that leaves
        // the holdback empty, and so the only one that can strand anything.
        //
        // Reading `last_agreed_words_slice().is_empty()` instead would be
        // LOOSER in two directions and both of them weaken the gate below: it is
        // true on a round that did not advance at all, and true on a round that
        // inherited an empty holdback from an earlier one. This asks the
        // function.
        let ran_off_the_end = common_len >= agreement.agreement_count_needed()
          && split_at_a_strict_boundary(
            &agreement.hypothesis_words[..common_len],
            common_len - agreement.agreement_count_needed(),
            confirmed_last_before,
          ) == common_len;
        fallbacks += u32::from(ran_off_the_end);
        // WHICH route emptied the holdback. A fallback round in which NO word
        // of `common` is over budget on its own is the AGGREGATE route: the
        // floor landed strictly inside a tied run whose words exceed the budget
        // only between them, every boundary at or above it tied, and split 0 --
        // the boundary a tied run always leaves legal -- was below the floor.
        // That is the shape `add_word_timestamps` produces from an all-zero
        // alignment matrix, and the one the oversized-word half cannot build.
        aggregate_fallbacks += u32::from(
          ran_off_the_end
            && agreement.hypothesis_words[..common_len]
              .iter()
              .all(|word| word.tokens_slice().len() <= MAX_HOLDBACK_PREFILL_TOKENS),
        );
        match nothing_unconfirmed_falls_below_the_watermark(
          &agreement,
          agreement.confirmed_words_slice().len() - before,
          trial,
          stride,
        ) {
          Some(true) => tie_strands += 1,
          Some(false) => backward_strands += 1,
          None => {}
        }
        // Read AFTER the ingest and kept for the next round, so every round is
        // compared against the transcript the round before it published.
        let after = published(&agreement);
        erasures += u32::from(nothing_the_stream_still_says_is_erased(
          &transcript,
          &after,
          &still_offered,
          &agreement,
          trial,
          stride,
        ));
        transcript = after;
        if let Some(empty) = postcondition(&agreement, trial, stride) {
          checked += 1;
          empty_holdbacks += u32::from(empty);
        }
      }
    }
  }

  // Non-vacuity, all three halves: the sweep really did build tied truths, the
  // postcondition really was READ against a non-empty confirmed list rather than
  // skipped, and the EMPTY-HOLDBACK state -- the one the earlier shape of this
  // test skipped, and the one #94 was re-opened from -- really was reached.
  assert!(
    tied_truths > 128,
    "the sweep must actually produce tied starts: {tied_truths} of 512 trials",
  );
  assert!(
    checked > 256,
    "the postcondition must actually be reachable: {checked} observations",
  );
  assert!(
    empty_holdbacks > 64,
    "the postcondition must be read against an EMPTY holdback too, which is \
     the state this test used to skip: {empty_holdbacks} of {checked} \
     observations",
  );
  assert!(
    fallbacks > 64,
    "and arm 3 -- the FALLBACK, where no legal boundary sits at or above the \
     budget floor and the split runs off the end -- must actually be taken, \
     since it is the only arm that can strand anything: {fallbacks} rounds",
  );
  assert!(
    aggregate_fallbacks > 16,
    "and it must be reached through the AGGREGATE route -- a tied run whose \
     tokens exceed the budget between them, with no single word over it, which \
     is the shape `add_word_timestamps` produces from an all-zero alignment \
     matrix and the one the oversized-word half cannot build: \
     {aggregate_fallbacks} of {fallbacks} fallback rounds",
  );
  assert!(
    forced_strands > 4,
    "and the FORCED arm must be reached with a live suffix beyond `common` -- \
     the state the second postcondition above speaks about, and the one an \
     unconditional forced advance strands: {forced_strands} rounds",
  );
  assert!(
    contradictions > 128,
    "and two consecutive hypotheses must actually CONTRADICT each other past \
     `common` -- growing prefixes alone make every hypothesis a prefix of the \
     next, which relieves the fallback arm before it can be read: \
     {contradictions} rounds",
  );
  assert!(
    backward_truths > 32,
    "and the sweep must actually produce BACKWARDS starts -- the shape \
     `segment::update_segments_with_word_timings` emits and the one every \
     non-decreasing generator misses: {backward_truths} of 512 trials",
  );
  assert!(
    tie_strands > 0,
    "and the second postcondition's exception must be reached rather than \
     merely written down, on BOTH halves. AT the highest confirmed start is the \
     tie an empty-holdback advance leaves: {tie_strands} rounds",
  );
  assert!(
    backward_strands > 0,
    "and BELOW it is the backwards start, which an interior split can confirm \
     past -- the half that did not exist while the generator kept its starts \
     non-decreasing: {backward_strands} rounds",
  );
  assert!(
    drifted_advances > 0,
    "and the RE-TIMED half must actually reach the advance path rather than \
     only being constructed -- a whole offering whose instants drifted while \
     its texts did not, agreed with and advanced over: {drifted_advances} \
     rounds",
  );
  assert!(
    erasures > 0,
    "and the TRANSCRIPT's own retention must be read against a round that \
     actually erases something below the watermark, which is the gap both of \
     round 5's findings hid in and the one `confirmed_words`' append-only \
     guarantee cannot see: {erasures} rounds",
  );
}
#[test]
fn a_trailing_tied_run_never_confirms_itself_twice_at_the_default_count() {
  // #94, codex round 1 on PR #95 -- the ORIGINAL duplicate-confirmation defect,
  // back on the DEFAULT driver path. Rule W's widening runs to `common.len()`
  // whenever the agreed prefix ENDS in a tied run: the holdback empties, the
  // watermark falls back to `common.last().end()`, and for a zero-duration word
  // that EQUALS its own start -- so `watermark_filtered`'s `start >= watermark`
  // re-admits it and every later stride confirms the whole run AGAIN.
  //
  // It needs neither an over-budget word nor a non-default
  // `agreement_count_needed`, which is what separates it from the empty-holdback
  // state `a_zero_duration_word_at_an_empty_holdback_is_not_re_confirmed` reaches
  // through the prefill budget: two ingests of one four-word hypothesis whose
  // last three words tie is enough.
  //
  // The repair has two separable halves and this test pins the SECOND. The
  // watermark's sample-domain anchor (see
  // `a_zero_duration_word_at_an_empty_holdback_is_not_re_confirmed`) is what
  // closes the DUPLICATION: with it in place the re-offered run is filtered out
  // even when the holdback is empty. The back-off is what keeps Rule W from
  // emptying the holdback in the first place, so the tied run stays revisable,
  // the next stride still gets a prefill anchor, and a genuinely new word at the
  // run's own instant is not filtered away with it.
  //
  // Mutation proof: delete the back-off arm (the `.or_else(...)` in
  // `split_at_a_strict_boundary`) and this reds -- `([" X", " A", " B", " C"],
  // [], 2.0000002)` against `([" X"], [" A", " B", " C"], 2.0)`. Take the
  // EARLIEST legal boundary instead of the latest (drop the `.rev()`) and it
  // reds the other way, confirming nothing at all: `([], [" X", " A", " B",
  // " C"], 0.0)`.
  //
  // Before either half existed, this read `[" X", " A", " B", " C", " A", " B",
  // " C", " A", " B", " C"]` after the four ingests below, growing by the whole
  // run on every further stride.
  let hypothesis = || {
    result_with_words(vec![
      word(" X", 0.0, 1.0),
      word(" A", 2.0, 2.0),
      word(" B", 2.0, 2.0),
      word(" C", 2.0, 2.0),
    ])
  };
  let mut agreement = LocalAgreement::new();
  assert_eq!(
    agreement.agreement_count_needed(),
    DEFAULT_AGREEMENT_COUNT_NEEDED,
    "non-vacuous: the DEFAULT count, the only one the driver reaches",
  );
  agreement.ingest_streamed(hypothesis());
  assert!(agreement.ingest_streamed(hypothesis()).is_advanced());
  assert_eq!(
    (
      confirmed_texts(&agreement),
      held_back_texts(&agreement),
      agreement.last_agreed_seconds(),
    ),
    (vec![" X"], vec![" A", " B", " C"], 2.0),
    "Rule W may not empty the holdback: with no legal boundary at or after the \
     requested split it backs off to the last one BEFORE the tied run, so the \
     run stays provisional instead of being confirmed against a watermark that \
     sits on its own start",
  );

  // The re-decode reproduces the holdback and nothing else -- the exact shape
  // `decoding_options_for_next` forces, clipped at the 2.0 s watermark.
  let reproduction = || {
    result_with_words(vec![
      word(" A", 2.0, 2.0),
      word(" B", 2.0, 2.0),
      word(" C", 2.0, 2.0),
    ])
  };
  agreement.ingest_streamed(reproduction());
  agreement.ingest_streamed(reproduction());
  assert_eq!(
    (confirmed_texts(&agreement), held_back_texts(&agreement)),
    (vec![" X"], vec![" A", " B", " C"]),
    "and NO stride re-confirms the run: the reproduction offers nothing that \
     starts strictly later, so the same back-off holds it exactly once",
  );
  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert_eq!(
    text, " X A B C",
    "and the finalized face agrees with the streaming one -- the holdback the \
     back-off kept is still emitted, so nothing is lost by holding it",
  );
  for token in ["X", "A", "B", "C"] {
    assert_eq!(
      text.matches(token).count(),
      1,
      "{token} must appear exactly once in {text:?}"
    );
  }
}

#[test]
fn an_over_budget_tied_run_strands_its_suffix_at_the_settled_instant() {
  // CHARACTERIZATION of Rule W's fallback arm, not a property that holds. This
  // pins what the engine does TODAY; the CORRECT answer is in the failure
  // messages below, so the day the trade is revisited this test goes red and
  // hands the next author the expectation.
  //
  // #94, codex round 3 on PR #95 -- the OTHER way the holdback empties, and the
  // one Rule W's back-off cannot reach. The back-off may not cross the prefill
  // budget FLOOR, and a tied run that is itself over budget puts that floor
  // strictly inside the run: every boundary at or above it ties, split 0 -- the
  // one boundary a tied run always leaves legal -- is below it, and the forward
  // search and the back-off therefore BOTH fail. The split runs off the end,
  // confirming the whole run and emptying the holdback.
  //
  // WHAT THAT COSTS: the watermark anchors at `past_the_settled_instant`,
  // strictly past
  // the run's instant, so any word the NEWER hypothesis produced at that same
  // instant beyond `common` -- words nothing ever confirmed -- fails the offered
  // filter on the next worded ingest, drops out of both hypotheses at once, and
  // `finalize` has nothing left to recover them from. This round's own
  // `finalize` has ALREADY published them, so it is a RETRACTION of transcript
  // and `confirmed_words`' append-only guarantee cannot see it.
  //
  // It takes no over-budget WORD, which is what separates it from
  // `a_zero_duration_word_at_an_empty_holdback_is_not_re_confirmed`: 113
  // ORDINARY one-token words sharing one start are 113 tokens against a
  // 112-token budget, and `add_word_timestamps` produces exactly that shape from
  // an ALL-ZERO alignment matrix -- the zero-fill
  // `add_word_timestamps_zero_pads_missing_rows` pins -- because DTW's tie-break
  // walks the path down column 0 and every text-index step there records the
  // same boundary time. Measured on that stack: 130 words, all at 0.0, all
  // zero-duration, one token each.
  //
  // WHY IT IS ACCEPTED. To a timestamp filter a genuinely new word at the run's
  // instant and a re-offer of the run's own last word are the same value, which
  // is this issue's impossibility result; every watermark that spares the strand
  // re-admits the settled word, which is #94. Between `6987bec` and `b3ec5c6`
  // this round DEFERRED instead, waiting for `common` to grow. Measured over the
  // accumulated counterexample suite and the 512-trial sweep, that wait cost 26
  // words erased from the published transcript where this fallback costs 10, and
  // its liveness bound was defeated by `At(0)`; see the engine module's doc,
  // "Why there is no deferral". So the loss below is this module's residual 1,
  // reached on the shape the pipeline can actually produce.
  //
  // WHICH HALF of residual 1 this is. The second postcondition's exception is
  // "at or below the highest confirmed start", and this test drives the AT half
  // -- the TIE, where the stranded word shares the settled word's own instant
  // and the empty holdback is what creates it. The BELOW half needs no empty
  // holdback and no tie: a backwards word start lets an INTERIOR split confirm
  // past a word it leaves unconfirmed
  // (`a_backward_start_from_the_segment_pipeline_does_not_strand_a_later_word`,
  // codex round 7 on PR #95, finding 1). The name of this test says "at the
  // settled instant" because that is the half it drives, not because the
  // exception is only that wide.
  const RUN: usize = MAX_HOLDBACK_PREFILL_TOKENS + 1;
  let tied: Vec<WordTiming> = (0..RUN)
    .map(|index| word(&format!(" w{index:03}"), 2.0, 2.0))
    .collect();
  assert_eq!(
    (
      tied.len(),
      tied
        .iter()
        .map(|word| word.tokens_slice().len())
        .sum::<usize>(),
      tied.iter().all(|word| word.tokens_slice().len() == 1),
    ),
    (113, MAX_HOLDBACK_PREFILL_TOKENS + 1, true),
    "non-vacuous: ORDINARY one-token words, and it is their SUM that exceeds \
     the budget -- no single word here is over budget",
  );

  // Two more words at the run's own instant, beyond the prefix the two
  // hypotheses agree on. Nothing confirms these; the watermark is the only thing
  // deciding whether they can still be offered.
  let suffix = || vec![word(" x0", 2.0, 2.0), word(" x1", 2.0, 2.0)];
  let older = || result_with_words(tied.clone());
  let newer = || result_with_words([tied.clone(), suffix()].concat());

  let mut agreement = LocalAgreement::new();
  assert_eq!(
    agreement.agreement_count_needed(),
    DEFAULT_AGREEMENT_COUNT_NEEDED,
    "non-vacuous: the DEFAULT count, the only one the driver reaches",
  );
  agreement.ingest_streamed(older());
  assert!(
    agreement.ingest_streamed(newer()).is_advanced(),
    "the agreeing round ADVANCES: an agreeing round always does",
  );
  assert_eq!(
    (
      confirmed_texts(&agreement).len(),
      held_back_texts(&agreement).len(),
      agreement.last_agreed_seconds(),
      // The consequence, read through the filter that consumes the watermark:
      // nothing the newer hypothesis produced at the run's instant is offerable
      // any more -- the whole run AND the two words beyond it. Folded into this
      // assertion rather than standing beside it, since it is a function of the
      // watermark above and could never red first.
      LocalAgreement::watermark_filtered(&newer(), agreement.last_agreed_seconds()).len(),
      // The hypotheses AGREED, so Swift KEEPS the result (`:408-410`,
      // `!skipAppend`) and it reaches the `finalize` merge as a segment source.
      agreement.results_slice().len(),
    ),
    (RUN, 0, PAST_TWO_SECONDS, 0, 2),
    "CHARACTERIZATION (https://github.com/findit-studio/coremlit/issues/94): no \
     legal boundary sits at or above the budget floor, so the split runs off \
     the END of `common` -- the whole 113-word run is confirmed, the holdback \
     is empty, and the watermark anchors strictly past the run's instant. The \
     CORRECT answer holds the run back and keeps \" x0\"/\" x1\" offerable; no \
     watermark can do both, since one strictly past the run's start filters \
     them and one at or below it re-admits the run's own last word. If the \
     trade was revisited, assert (0, 0, 0.0, RUN + 2, 2) here and re-check the \
     retraction below.",
  );

  // FINALIZE POINT ONE. The retraction is only visible across two of these, and
  // this is the one that PUBLISHES the words: the round confirmed `common` and
  // `find_longest_different_suffix` adds `[" x0", " x1"]` on top.
  let published = agreement
    .clone()
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert_eq!(
    published.split_whitespace().count(),
    RUN + 2,
    "a stream that ENDS on this round publishes every word it produced, the \
     two at the settled instant included: {published:?}",
  );

  // FINALIZE POINT TWO, one hypothesis later. `" x0"` and `" x1"` are below the
  // watermark by now, so the filter drops them from the hypothesis AND from the
  // re-read previous result -- both sides at once. The grown tail carries TWO
  // words starting strictly later, which is what the default count needs to
  // agree over anything again.
  let grown = || {
    result_with_words(
      [
        tied.clone(),
        suffix(),
        vec![word(" y", 3.0, 4.0), word(" z", 4.0, 5.0)],
      ]
      .concat(),
    )
  };
  agreement.ingest_streamed(grown());
  let retracted = agreement
    .clone()
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert_eq!(
    (
      retracted.split_whitespace().count(),
      retracted
        .split_whitespace()
        .rev()
        .take(3)
        .collect::<Vec<_>>(),
    ),
    (RUN + 2, vec!["z", "y", "w112"]),
    "CHARACTERIZATION, and a RETRACTION -- this module's non-preferred \
     direction (https://github.com/findit-studio/coremlit/issues/94). The \
     CORRECT answer is {} words with [\"x0\", \"x1\"] still between \"w112\" \
     and \"y\": both hypotheses produced them, nothing contradicted them, and \
     the round before this one already published them. They are gone because \
     the empty holdback's watermark passed their instant. Accepted as \
     residual 1 -- to a timestamp filter they are indistinguishable from a \
     re-offer of the run's own last word, which is what #94 is. If that was \
     revisited, assert {} words here and delete this message. Transcript: \
     {retracted:?}",
    RUN + 4,
    RUN + 4,
  );

  // AND IT IS NOT A STALL: the grown tail reaches `common` on the next ingest
  // and the stream keeps moving.
  assert!(
    agreement.ingest_streamed(grown()).is_advanced(),
    "the grown tail reaches `common` and the round advances",
  );
  assert_eq!(
    (
      confirmed_texts(&agreement).len(),
      held_back_texts(&agreement),
      agreement.last_agreed_seconds(),
    ),
    (RUN, vec![" y", " z"], 3.0),
    "and the tail is held back under an ordinary interior split, with the run \
     confirmed exactly once",
  );
}

#[test]
fn a_dropped_disagreeing_hypothesiss_draw_survives_into_finalize() {
  // F1 (codex round 8). A three-hypothesis history where the MIDDLE hypothesis
  // disagrees and is dropped from `results` (`:395-400`, skipAppend) but is
  // retained as `prev_result` to CONTROL the next agreement comparison. Its
  // unseeded draw decided which words R3 agreed on, so it must reach `finalize`'s
  // reproducibility answer even though its segments never survive the merge.
  //
  // Mutation proof: remove the `ingested_facts` accumulation in `ingest` (or its
  // merge in `finalize`) and the dropped R2's `Some(true)` draw vanishes -- the
  // final record reads `Some(false)` and `is_reproducible()` flips true, failing
  // the assertions below.
  let r1 = result_with_words(vec![word(" And", 0.0, 0.4), word(" so", 0.4, 0.7)])
    .with_task_facts(TaskFacts::observed_clean());
  // R2 disagrees with R1 (no common prefix) AND drew from an unseeded sampler.
  let r2 = result_with_words(vec![word(" But", 0.0, 0.4), word(" then", 0.4, 0.7)])
    .with_task_facts(TaskFacts::observed_clean().with_drew_from_rng(true));
  // R3 agrees with the retained R2 control hypothesis, advancing the watermark.
  let r3 = result_with_words(vec![
    word(" But", 0.0, 0.4),
    word(" then", 0.4, 0.7),
    word(" folks", 0.7, 1.0),
  ])
  .with_task_facts(TaskFacts::observed_clean());

  let mut agreement = LocalAgreement::new();
  assert!(agreement.ingest_streamed(r1).is_awaiting_agreement());
  assert!(
    agreement.ingest_streamed(r2).is_awaiting_agreement(),
    "R2 disagrees with R1 and is dropped from results",
  );
  assert!(
    agreement.ingest_streamed(r3).is_advanced(),
    "R3 agrees with the retained R2 control hypothesis",
  );

  // R2 is absent from the kept results: only R1 (2 words) and R3 (3 words) remain.
  assert_eq!(agreement.results_slice().len(), 2, "R2 was dropped");
  assert_eq!(agreement.results_slice()[0].all_words().len(), 2, "R1 kept");
  assert_eq!(
    agreement.results_slice()[1].all_words().len(),
    3,
    "R3 kept -- the 2-word R2 is not here",
  );

  let options = crate::audio::whisper::options::DecodingOptions::new();
  let compute = crate::audio::whisper::options::ComputeOptions::new();
  let finalized = agreement.finalize(&options);
  assert_eq!(
    finalized.task_facts().drew_from_rng(),
    Some(true),
    "the dropped control hypothesis's unseeded draw survives into finalize",
  );
  assert!(
    !crate::audio::whisper::provenance::Provenance::for_result(&options, &compute, &finalized)
      .is_reproducible(),
    "an unseeded draw happened (in a dropped hypothesis), so it is not reproducible",
  );
  // ORACLE CORRECTION (codex round 13, M2): a seed does NOT make the recovered
  // draw replayable here. `finalize` leaves the worker schedule at the `None` the
  // agreement strip produces (its confirmed text interleaves MULTIPLE hypotheses,
  // so no single ordered attribution survives -- round 10, F2), and a draw whose
  // domain-separating coordinate is unknown can land different text at a different
  // coordinate even under the same seed. This is the LocalAgreement history through
  // which the seed-plus-unknown-schedule case is reachable; the focused unit is
  // `task_facts::tests::seeded_draw_with_unknown_worker_schedule_is_not_reproducible`.
  //
  // Mutation proof: drop the `&& schedule_known` guard from
  // `is_reproducible_under`'s `Some(true)` draw arm and this reads back reproducible.
  assert_eq!(
    finalized.task_facts().worker_schedule(),
    None,
    "agreement strips the schedule, so the seeded draw's coordinate is unknown",
  );
  assert!(
    !crate::audio::whisper::provenance::Provenance::for_result(
      &options.clone().with_seed(11),
      &compute,
      &finalized,
    )
    .is_reproducible(),
    "a seed cannot replay the recovered draw whose worker coordinate agreement stripped",
  );
}

#[test]
fn finalize_keeps_the_earliest_ingested_language_over_a_later_survivor() {
  // F3 (codex round 9). The ingest-ordered sink observed a MIDDLE hypothesis's
  // "es" (which disagreed and was dropped from `results`) BEFORE a later
  // surviving hypothesis's "fr". Folding that sink as a trailing contributor let
  // the survivor's "fr" win first-genuine; seeding the finalize fold FROM the
  // sink keeps the earliest genuine observation, "es".
  //
  // Mutation proof: revert `finalize` to fold the sink LAST
  // (`merged.task_facts_mut().merge(&ingested)`) and this reads back Some("fr").
  //
  // R1: kept (first ever), observes NO language. R2: disagrees with R1 (no common
  // prefix), observes "es", dropped from results but retained as the control. R3:
  // agrees with the retained R2 control, observes "fr", kept.
  let r1 = result_with_words(vec![word(" And", 0.0, 0.4), word(" so", 0.4, 0.7)])
    .with_task_facts(TaskFacts::observed_clean());
  let r2 = result_with_words(vec![word(" But", 0.0, 0.4), word(" then", 0.4, 0.7)])
    .with_task_facts(TaskFacts::observed_clean().with_observed_language(Some("es".into())));
  let r3 = result_with_words(vec![
    word(" But", 0.0, 0.4),
    word(" then", 0.4, 0.7),
    word(" folks", 0.7, 1.0),
  ])
  .with_task_facts(TaskFacts::observed_clean().with_observed_language(Some("fr".into())));

  let mut agreement = LocalAgreement::new();
  assert!(agreement.ingest_streamed(r1).is_awaiting_agreement());
  assert!(
    agreement.ingest_streamed(r2).is_awaiting_agreement(),
    "R2 disagrees with R1 and is dropped from results",
  );
  assert!(
    agreement.ingest_streamed(r3).is_advanced(),
    "R3 agrees with the retained R2 control hypothesis",
  );
  assert_eq!(
    agreement.results_slice().len(),
    2,
    "only R1 and R3 are kept; the es-observing R2 was dropped",
  );

  let options = crate::audio::whisper::options::DecodingOptions::new();
  let compute = crate::audio::whisper::options::ComputeOptions::new();
  let finalized = agreement.finalize(&options);
  assert_eq!(
    finalized.task_facts().observed_language(),
    Some("es"),
    "the earliest ingested genuine language wins, even from a dropped hypothesis",
  );
  assert_eq!(
    crate::audio::whisper::provenance::Provenance::for_result(&options, &compute, &finalized)
      .task_facts()
      .observed_language(),
    Some("es"),
    "and provenance carries that earliest observation",
  );
}

#[test]
fn finalize_reports_an_unknown_worker_schedule() {
  // ADJUDICATED (round 10, F2): agreement-confirmed text interleaves words from
  // MULTIPLE hypotheses, so no single ordered worker attribution is knowable --
  // the finalized record's worker_schedule is None even when every ingested
  // hypothesis carried a DISTINCT, known coordinate. The strip at `ingest` makes
  // every contributor's schedule None, and the absorbing-None merge law keeps the
  // aggregate None (the surviving results' own [0, 2] cannot pass through it).
  //
  // Mutation proof: drop the `.with_worker_schedule(None)` strip in `ingest` and
  // the ingested coordinates accumulate, so the finalized schedule reads back a
  // non-None Some(...) instead of the adjudicated None.
  let r1 = result_with_words(vec![word(" And", 0.0, 0.4), word(" so", 0.4, 0.7)])
    .with_task_facts(TaskFacts::observed_clean().with_worker(0));
  let r2 = result_with_words(vec![word(" But", 0.0, 0.4), word(" then", 0.4, 0.7)])
    .with_task_facts(TaskFacts::observed_clean().with_worker(1));
  let r3 = result_with_words(vec![
    word(" But", 0.0, 0.4),
    word(" then", 0.4, 0.7),
    word(" folks", 0.7, 1.0),
  ])
  .with_task_facts(TaskFacts::observed_clean().with_worker(2));

  let mut agreement = LocalAgreement::new();
  assert!(agreement.ingest_streamed(r1).is_awaiting_agreement());
  assert!(
    agreement.ingest_streamed(r2).is_awaiting_agreement(),
    "R2 disagrees with R1 and is dropped from results",
  );
  assert!(
    agreement.ingest_streamed(r3).is_advanced(),
    "R3 agrees with the retained R2 control hypothesis",
  );
  // The surviving results R1 (worker 0) and R3 (worker 2) carry a knowable [0, 2],
  // but the confirmed transcript mixes their words -- attribution is unknown.
  let finalized = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    finalized.task_facts().worker_schedule(),
    None,
    "agreement-confirmed text has no single knowable worker attribution -- unknown, not [0, 2]",
  );
}

#[test]
fn a_disagreeing_final_hypothesis_replaces_the_holdback_instead_of_duplicating_it() {
  // DIVERGENCE from `TranscribeCLI.swift:418-419` (module doc). Swift's `let
  // final = lastAgreedWords + findLongestDifferentSuffix(prevWords,
  // hypothesisWords)` composes correctly only when the LAST hypothesis AGREED:
  // `find_longest_common_prefix` returns elements from `current`, so on an
  // advance `last_agreed_words` IS the final hypothesis's own
  // `[split..common.len()]` slice and the sum reconstructs
  // `hypothesis_words[split..]` word-for-word. When the last hypothesis
  // DISAGREED, `last_agreed_words` belongs to the hypothesis that one just
  // SUPERSEDED, while `hypothesis_words` re-covers the same span (it is filtered
  // to `start >= last_agreed_words[0].start`) carrying the revision -- and Swift
  // emits both.
  //
  // Mutation proof: drop the `holdback_superseded` guard in `finalize` and this
  // reads back " alpha bravo charlie xray charlie" -- " charlie" twice, and the
  // superseded " bravo" sitting beside its own revision " xray".
  let mut agreement = LocalAgreement::new();
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" alpha", 0.0, 0.4),
        word(" bravo", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_awaiting_agreement(),
    "first result: nothing to agree with",
  );
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" alpha", 0.0, 0.4),
        word(" bravo", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_advanced(),
    "identical rerun agrees on all three: confirms alpha, holds bravo+charlie",
  );
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" alpha"],
  );
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" bravo", " charlie"],
  );

  // The LAST hypothesis revises the held-back " bravo" to " xray", so it shares
  // no common prefix with the previous one and is dropped from `results`.
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" xray", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_awaiting_agreement(),
    "the revision disagrees with the holdback",
  );
  assert_eq!(
    agreement.results_slice().len(),
    2,
    "the revision is dropped"
  );

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " alpha xray charlie",
    "the final hypothesis's own words replace the holdback it superseded",
  );
}

#[test]
fn a_disagreeing_final_pair_keeps_the_words_both_hypotheses_agreed_on() {
  // The same defect with an EMPTY holdback -- its other face. With no advance
  // yet, `last_agreed_words` is empty, so Swift's `lastAgreedWords +
  // differentSuffix(prevWords, hypothesisWords)` contributes only
  // `hypothesis_words[common.len()..]` and DROPS the `common.len()` leading
  // words BOTH hypotheses produced (up to `agreement_count_needed - 1` of them).
  // `last_agreed_words` was what was supposed to supply them, and on this path
  // it holds nothing.
  //
  // Mutation proof: drop the `holdback_superseded` guard in `finalize` and this
  // reads back " then" -- " and", which BOTH hypotheses produced, subtracted
  // away by `find_longest_different_suffix` with nothing in the (empty)
  // holdback to put it back.
  //
  // Round 4 had recorded this falsifier as unreachable-rather-than-guarded: at
  // the time, `ingest` STORED its remainder and `finalize` no longer subtracted
  // anything, so both branches coincided here and the guard mutation left this
  // green. That reasoning was true of the recorded-remainder shape and does not
  // survive its removal -- `finalize` subtracts again, so the guard is
  // load-bearing again and the direct mutation is the falsifier once more.
  let mut agreement = LocalAgreement::new();
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" and", 0.0, 0.4),
        word(" so", 0.4, 0.7),
      ]))
      .is_awaiting_agreement()
  );
  // A one-word common prefix is short of the default threshold of 2, so this
  // disagrees even though both hypotheses produced " and".
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" and", 0.0, 0.4),
        word(" then", 0.4, 0.7),
      ]))
      .is_awaiting_agreement()
  );
  assert!(agreement.confirmed_words_slice().is_empty());
  assert!(agreement.last_agreed_words_slice().is_empty());

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(final_result.text(), " and then");
}

#[test]
fn a_final_hypothesis_with_nothing_past_the_watermark_keeps_the_holdback() {
  // The GUARD on the divergence above, and the reason it is `holdback_superseded
  // && !hypothesis_words.is_empty()` rather than `holdback_superseded` alone. A
  // result whose every word lands BEFORE the watermark still clears the
  // `has_words` gate (`:379-386` checks `segment.words_slice()`, not the
  // watermark filter), so it reaches the agreement comparison with an EMPTY
  // `hypothesis_words`, disagrees, and marks the holdback superseded -- but it
  // re-covers nothing, so there is no revision to prefer and the provisional
  // holdback is still the only estimate for that span. This case must stay
  // byte-identical to `TranscribeCLI.swift:418-419`.
  //
  // Mutation proof: drop the `!self.hypothesis_words.is_empty()` conjunct from
  // `finalize`'s guard and this reads back " alpha" -- the held-back
  // " bravo charlie" silently lost.
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![
    word(" alpha", 0.0, 0.4),
    word(" bravo", 0.4, 0.7),
    word(" charlie", 0.7, 1.0),
  ]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" alpha", 0.0, 0.4),
        word(" bravo", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_advanced()
  );
  assert!((agreement.last_agreed_seconds() - 0.4).abs() < 1e-6);

  // Words, but every one of them before the 0.4 s watermark.
  let outcome = agreement.ingest_streamed(result_with_words(vec![word(" alpha", 0.0, 0.4)]));
  assert!(
    outcome.is_awaiting_agreement(),
    "words present, so NOT NoWordTimings -- it disagrees instead: {outcome}",
  );

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " alpha bravo charlie",
    "nothing superseded the holdback, so it is still flushed",
  );
}

#[test]
fn an_agreement_after_a_disagreement_restores_the_swift_shape() {
  // The recovery path, and the falsifier for `holdback_superseded`'s CLEAR. A
  // disagreement marks the holdback superseded; the NEXT hypothesis corroborates
  // the revision and advances, which re-anchors `last_agreed_words` in that same
  // final hypothesis — so `finalize` must be back on `TranscribeCLI.swift:418-419`
  // verbatim.
  //
  // Mutation proof: make the flag sticky (`if skip_append { self
  // .holdback_superseded = true; }` instead of the unconditional assignment) and
  // this reads back " alpha xray xray charlie delta echo" -- " xray" both
  // confirmed and re-emitted, because the divergence branch fires on an ingest
  // that actually agreed. The other three-hypothesis finalize tests are blind to
  // it: they all split at 0, where `confirmed_words` absorbs none of
  // `hypothesis_words` and the two branches coincide. (The two
  // `..._repeats_a_held_back_word_...` tests do catch it, from a longer
  // history — this is the minimal one.)
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![
    word(" alpha", 0.0, 0.4),
    word(" bravo", 0.4, 0.7),
    word(" charlie", 0.7, 1.0),
  ]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" alpha", 0.0, 0.4),
        word(" bravo", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_advanced()
  );
  // Revises the held-back " bravo": disagrees, dropped, holdback superseded.
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" xray", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
        word(" delta", 1.0, 1.3),
      ]))
      .is_awaiting_agreement()
  );
  // Corroborates the revision on three words: confirms " xray", holds
  // " charlie delta", and the holdback is the final hypothesis's own again.
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        word(" xray", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
        word(" delta", 1.0, 1.3),
        word(" echo", 1.3, 1.6),
      ]))
      .is_advanced()
  );
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" alpha", " xray"],
  );

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(final_result.text(), " alpha xray charlie delta echo");
}

#[test]
fn a_revision_that_repeats_a_held_back_word_is_not_duplicated() {
  // Codex round 4, F2, at the engine. The stream settles " A", holds " B C", and
  // the next hypothesis revises the span to " B X B C D" -- it INSERTS " X" and
  // then repeats " B" behind it. The repeat is what makes the shape adversarial:
  // the held-back " B" and the revision's SECOND " B" carry the same text, so
  // the two occurrences are distinguishable only by what produced them.
  //
  // On `TranscribeCLI.swift:418-419` this reads back " A B C X B C D": the
  // held-back " B C" emitted once as the superseded reading and again inside the
  // revision that replaced it. `holdback_superseded` collapses that to the
  // revision alone.
  //
  // Mutation proof: drop the `holdback_superseded` guard in `finalize` and this
  // reads back " A B C X B C D".
  let settled = || {
    result_with_words(vec![
      word(" A", 0.0, 0.005),
      word(" B", 0.0, 0.5),
      word(" C", 1.0, 1.5),
    ])
  };
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(settled());
  assert!(agreement.ingest_streamed(settled()).is_advanced());
  // Rule W (#94): the split may not cut at a tied start, so it widens past the
  // tie -- one word moves from `last_agreed_words_slice()` into
  // `confirmed_words_slice()` and the watermark moves to the first word past
  // the tie. `confirmed ++ holdback` is unchanged, and the finalized text this
  // test asserts is measured byte-identical either way.
  assert_eq!(confirmed_texts(&agreement), vec![" A", " B"]);
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" C"],
  );

  let revision = || {
    result_with_words(vec![
      word(" A", 0.0, 0.005),
      word(" B", 0.0, 0.4),
      word(" X", 1.0, 1.4),
      word(" B", 1.5, 1.9),
      word(" C", 2.0, 2.4),
      word(" D", 2.5, 2.9),
    ])
  };
  assert!(
    agreement
      .ingest_streamed(revision())
      .is_awaiting_agreement(),
    "one shared word is short of the threshold of 2, so this disagrees",
  );

  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A B X B C D",
    "the revision replaces the holdback it superseded, and nothing is emitted twice",
  );
}

#[test]
fn a_corroborated_revision_that_repeats_a_held_back_word_reaches_the_transcript() {
  // Codex round 4, F2, de-confounded: the SAME revision, offered twice, so the
  // run ends on an agreement and `finalize`'s divergence branch is not involved
  // at all. What is under test here is `ingest` -- the revision must reach
  // `confirmed_words_slice()` on the streaming path a caller reads between
  // pushes, with the watermark left where the words it confirmed actually are.
  //
  // This is the row round 4 used to separate the two candidate rules: an
  // occurrence-blind rule that anchors on TEXT strips the revision's leading
  // " B X" as though it reproduced the holdback, falsely ADVANCES on the
  // remainder, and loses " B X" permanently -- they never reach
  // `confirmed_words_slice()` and the watermark moves past them.
  let settled = || {
    result_with_words(vec![
      word(" A", 0.0, 0.005),
      word(" B", 0.0, 0.5),
      word(" C", 1.0, 1.5),
    ])
  };
  let revision = || {
    result_with_words(vec![
      word(" A", 0.0, 0.005),
      word(" B", 0.0, 0.4),
      word(" X", 1.0, 1.4),
      word(" B", 1.5, 1.9),
      word(" C", 2.0, 2.4),
      word(" D", 2.5, 2.9),
    ])
  };
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(settled());
  assert!(agreement.ingest_streamed(settled()).is_advanced());
  agreement.ingest_streamed(revision());
  assert!(
    agreement.ingest_streamed(revision()).is_advanced(),
    "the revision corroborates itself",
  );

  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B", " X", " B"],
    "the revised span is confirmed, not skipped over",
  );
  assert!(
    (agreement.last_agreed_seconds() - 2.0).abs() < 1e-6,
    "the watermark sits on the holdback's own first word, not past the revision: {}",
    agreement.last_agreed_seconds(),
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A B X B C D",
  );
}

// ---------------------------------------------------------------------
// The re-admission ledger: the sequences that closed #94
// ---------------------------------------------------------------------
//
// Every test below is a sequence that slid past `watermark_filtered`'s old
// leading-run strip and confirmed a settled word a second time -- or, in the
// last two, DELETED a word the stream genuinely produced. Each was a
// CHARACTERIZATION while <https://github.com/findit-studio/coremlit/issues/94>
// was open: it asserted the wrong answer the strip produced and carried the
// right one in its failure message. RULE W (`LocalAgreement::ingest`'s advance)
// closed the class at its source -- the split never cuts at a tied start, so a
// confirmed word can never pass the offered filter -- and `watermark_filtered`
// went back to Swift's bare `:372` line. The two `rule_w_deletes_*` entries are
// the cost of that trade and stay CHARACTERIZATION; the rest now assert the
// property their names state.
//
// They belong beside the constraints in the next section: these pin what the
// rule must now DO, those pin what it must still NOT do. Run both halves at
// once:
//
//     cargo test -p coremlit --features whisper --lib -- \
//         audio::whisper::stream::agreement::tests::
//
// NOT `#[ignore]`. An early revision of this ledger marked its ten entries
// `#[ignore]` and called them "red on purpose". That is the wrong marker and
// must not be restored, for two independent reasons:
//
//   - libtest's `--ignored` is ignored-ONLY, not skip-the-ignored: it SELECTS
//     every ignored test in the target and runs it. This repository's model
//     gates (.github/workflows/ci.yml, the `whisper|@all` row) and its sharded
//     coverage legs (.github/workflows/coverage.yml, the same row) both invoke
//     `-- --ignored`, so an `#[ignore]`d test is not parked -- it is scheduled.
//   - An ignored test never executes, so it rots silently as the code moves
//     under it.
//
// `#[ignore]` here marks a test CI cannot run unconditionally: it needs an
// artifact this checkout may not have (a staged model, a tokenizer sidecar, a
// fixture tree), or one specific host, or a cost CI will not pay. What it must
// never mean is "this test is expected to fail" -- `--ignored` runs it.
//
// Four adversarial-review rounds each defeated a different predicate over
// (confirmed list, offered list, watermark) before establishing that NO such
// predicate can decide these: two runs reach byte-identical triples and need
// opposite answers, so occurrence identity is not recoverable from those three.
// A fifth -- a longest-common-subsequence alignment of `settled ++ holdback`
// against the offered list -- decided them, and deleted the words BETWEEN two
// occurrences of a recurring phrase while doing it
// (`a_recurring_phrase_does_not_delete_the_words_between_its_occurrences`).
// Rule W stops trying to decide the state and refuses to create it. The
// defeated approaches and the falsifier that killed each are in the issue.

#[test]
fn a_later_re_use_of_a_confirmed_word_is_not_mistaken_for_a_re_admission() {
  // THE STANDING BOUND on any re-admission rule, and phase-gate round 2's rule
  // restated for it: a word the stream genuinely re-utters must cost nothing.
  // " A" is confirmed at 0.0 s and the clip says " A" again at 2.0 s -- a
  // different word at a different instant, with provisional words in front of
  // it. Under Rule W nothing looks for it at all: the watermark is 1.0 s, the
  // offered filter is Swift's bare `start >= watermark`, and a rule that SCANS
  // the offered list for a confirmed word's TEXT would find the 2.0 s " A" and
  // take the words in front of it down too. Kept green as the constraint any
  // future change is measured against, not because the current code has a
  // branch that could get it wrong.
  //
  // Mutation proof: put a text SCAN back in front of the agreement comparison
  // (`hypothesis_words.rposition(|w| confirmed texts contain normalized(w))`,
  // then strip through it -- the naive rule phase-gate round 2 defeated) and
  // this reads back " A B D E": " C" and the re-uttered " A", both agreed by two
  // consecutive hypotheses, gone. Measured, and it reds five other pins with it
  // (`a_recurring_phrase_does_not_delete_the_words_between_its_occurrences`
  // among them).
  let a0 = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with the confirmed A
  let c = || word(" C", 1.0, 2.0);
  let a2 = || word(" A", 2.0, 2.5); // the same TEXT, a different word
  let d = || word(" D", 3.0, 3.5);
  let e = || word(" E", 4.0, 4.5);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a0(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![a0(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![b(), c(), a2(), d()]));
  agreement.ingest_streamed(result_with_words(vec![b(), c(), a2(), d(), e()]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B C A D E",
    "both A's belong in the transcript, and neither costs the words before it",
  );
}

// ---------------------------------------------------------------------
// What the re-admission rule must NOT break
// ---------------------------------------------------------------------
//
// The pins below were green on `main` and on every defeated candidate rule, and
// they are green under Rule W: they are the constraints a replacement has to
// keep satisfying, recorded next to the ledger above so any future change is
// measured against both halves at once. Deleting a word the stream genuinely
// produced is the failure mode a stricter rule falls into, and it is worse than
// the duplication the ledger records -- `tests/whisper/streaming.rs`'s portable
// prefix property tolerates a truncation and forbids a rewrite. Rule W pays
// exactly that cost on two tied fixtures, which is why those two are
// characterization tests carrying the correct answer rather than pins.

#[test]
fn a_stutter_at_the_watermark_keeps_both_occurrences() {
  // Codex round 3, finding 2, first half -- the half every candidate rule got
  // right, kept green so the second half
  // (`a_distinct_repetition_of_a_confirmed_word_survives_the_continuing_stream`)
  // is pinned from both ends. A hypothesis that STUTTERS at the watermark's own
  // instant -- " A" twice, same text, same start, same end -- straddles the
  // advance's split: the first " A" is confirmed, the second is held back, and
  // `finalize` flushes it. Nothing distinguishes that survivor from a
  // reproduction of the word just confirmed, which is why the rule must not run
  // a second time over words a first pass already cleared.
  //
  // Mutation proof: make the offered filter STRICT (`start > watermark` in
  // `watermark_filtered`) and the advance never happens -- both stuttered A's
  // sit exactly at the 0.0 s watermark, the hypothesis comes back as the single
  // word " B", and `is_advanced()` is false. The non-strict endpoint is what
  // lets a hypothesis re-offer the instant the watermark sits on, which is the
  // whole of what Rule W then has to keep safe.
  let hypothesis = || {
    result_with_words(vec![
      word(" A", 0.0, 0.5),
      word(" A", 0.0, 0.5),
      word(" B", 1.0, 1.5),
    ])
  };
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(hypothesis());
  assert!(agreement.ingest_streamed(hypothesis()).is_advanced());
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A A B",
    "both stuttered A's are the hypothesis's own words, not a re-admission",
  );
}

/// `last_agreed_words_slice()` as text -- the still-provisional half of the
/// streaming face, asserted beside `confirmed_texts` wherever a word could be in
/// one list rather than the other.
/// Whether the round just ingested reached `split_at_a_strict_boundary`'s FORCED
/// arm -- the budget floor at `common.len()` -- with words still beyond `common`
/// that the advance's own watermark would put out of reach.
///
/// Written LONGHAND from the engine's recorded comparison state rather than by
/// calling `budgeted_split`, so a mutation to it cannot mutate this counter
/// along with it:
///
/// `find_longest_common_prefix`'s answer for the round the engine has just
/// finished, recomputed from the two word lists it left behind. The sweep's
/// alphabet is five distinct plain texts, so `normalized` comparison is text
/// equality and this is the engine's own `common.len()`.
fn common_prefix_len(agreement: &LocalAgreement) -> usize {
  agreement
    .prev_words
    .iter()
    .zip(&agreement.hypothesis_words)
    .take_while(|(previous, current)| previous.word() == current.word())
    .count()
}

/// - `budgeted_split(common, 0) == common.len()` is exactly "`common`'s last
///   word alone exceeds `MAX_HOLDBACK_PREFILL_TOKENS`". The loop subtracts words
///   from the front while the holdback is over budget, so it can only run off
///   the end from `split == common.len() - 1`, where the holdback is that one
///   word.
/// - the sweep's alphabet is five distinct plain texts, so
///   `find_longest_common_prefix`'s `normalized` comparison is text equality.
fn forced_arm_with_a_live_suffix(agreement: &LocalAgreement) -> bool {
  let common_len = common_prefix_len(agreement);
  if common_len < agreement.agreement_count_needed() {
    return false;
  }
  let last = &agreement.hypothesis_words[common_len - 1];
  if last.tokens_slice().len() <= MAX_HOLDBACK_PREFILL_TOKENS {
    return false;
  }
  let watermark = empty_holdback_anchor(last, last.start());
  agreement.hypothesis_words[common_len..]
    .iter()
    .any(|word| word.start() < watermark)
}

fn held_back_texts(agreement: &LocalAgreement) -> Vec<&str> {
  agreement
    .last_agreed_words_slice()
    .iter()
    .map(WordTiming::word)
    .collect()
}

/// `confirmed_words_slice()` as text — the face a streaming caller reads
/// between pushes, asserted alongside the finalized text so neither can pass on
/// the strength of the other.
fn confirmed_texts(agreement: &LocalAgreement) -> Vec<&str> {
  agreement
    .confirmed_words_slice()
    .iter()
    .map(WordTiming::word)
    .collect()
}

#[test]
fn a_distinct_repetition_of_a_confirmed_word_survives_the_continuing_stream() {
  // #94 (codex round 3, finding 2) -- the ledger's OTHER face, a DELETION rather
  // than a duplication, and the reason a stricter TEXT-MATCHING rule was never
  // the fix. A hypothesis that STUTTERS at the watermark's own instant straddles
  // the advance: the first " A" is confirmed and the second -- a DISTINCT
  // occurrence with identical text, start and end -- is held back. Every later
  // hypothesis correctly omits the confirmed one and re-offers the held-back
  // one; the old strip read that survivor as a reproduction (it IS a
  // front-of-list text match) and stripped it on every filter, and the next
  // advance moved the watermark past it.
  //
  // This is the issue's two-run construction, live: a run of ` A A B` and a run
  // of ` A B C` reach byte-identical (confirmed, offered, watermark) and need
  // OPPOSITE answers, so no predicate over those three can be right. Round 5,
  // finding 1 re-reported THIS sequence and argued the other reading -- that the
  // offered " A" is the SETTLED one coming back, so a rule that keeps it
  // duplicates a word. Rule W answers neither reading: the stutter is a start
  // TIE (both A's at 0.0 s), so the split widens past both instead of cutting
  // between them. They are confirmed together, the watermark lands on " B" at
  // 1.0 s, and no later hypothesis can offer either one back -- the state the
  // two readings disagreed about is never created.
  //
  // Mutation proof: delete Rule W's widening loop (the `while split <
  // common.len()` in `ingest`'s advance) and this reads back [" A"] instead of
  // [" A", " A"] at the first checkpoint -- the split cuts between the two
  // stuttered A's, the second is held back at a watermark tied to the first, and
  // the sequence is back in the undecidable state. Measured; the same mutation
  // reds `tied_word_starts_never_confirm_twice`,
  // `omitting_a_confirmed_tied_word_does_not_drop_provisional_words`,
  // `the_split_never_cuts_at_a_tied_start` and both `rule_w_deletes_*`
  // characterizations.
  //
  //   ingest    [A@0.0, A@0.0, B@1.0] twice -> confirmed [A], holding [A, B]
  //   then      [A@0.0, B@1.0, C@2.0, D@3.0] twice
  //   confirmed_words_slice() == [" A", " A", " B"], text " A A B C D"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 2.0, 2.5);
  let d = || word(" D", 3.0, 3.5);
  let stutter = || result_with_words(vec![a(), a(), b()]);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(stutter());
  assert!(agreement.ingest_streamed(stutter()).is_advanced());
  // Rule W (#94): the split may not cut at a tied start, so it widens past the
  // tie -- one word moves from `last_agreed_words_slice()` into
  // `confirmed_words_slice()` and the watermark moves to the first word past
  // the tie. `confirmed ++ holdback` is unchanged, and the finalized text this
  // test asserts is measured byte-identical either way.
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " A"],
    "the advance settles both stuttered A's rather than cutting between them",
  );

  // The stream CONTINUES past that advance -- which is exactly what
  // `a_stutter_at_the_watermark_keeps_both_occurrences` does not do, and why
  // that pin cannot see this.
  let onward = || result_with_words(vec![a(), b(), c(), d()]);
  agreement.ingest_streamed(onward());
  assert!(agreement.ingest_streamed(onward()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " A", " B"],
    "the held-back A is the stream's own second occurrence, not a re-offer of \
     the confirmed first one",
  );

  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A A B C D",
    "both occurrences the stream produced",
  );
}

#[test]
fn rule_w_deletes_a_tied_insertion_that_reproduces_nothing_confirmed() {
  // CHARACTERIZATION of Rule W, not a property that holds. This pins what the
  // engine does TODAY on a tied fixture; the CORRECT answer is in the failure
  // message below, so the day the trade is revisited this test goes red and
  // hands the next author the expectation.
  //
  // TRIGGER: two adjacent agreed words with EQUAL starts -- here `" A"@[0.0,0.5)`
  // and `" B"@[0.0,1.0)`. On words this crate's own pipeline produces that
  // needs a zero-duration word, since `find_alignment` guarantees
  // `w[i].end() <= w[i + 1].start() + 1e-4` (see
  // `crate::audio::whisper::segment`'s tests); `LocalAgreementTranscriber`
  // never reaches it, and the committed jfk golden carries no start tie at all.
  //
  // WHAT MOVES: Rule W widens the first advance's split past the tie, so `" B"`
  // is confirmed a round early and the watermark lands at 1.0 s instead of
  // 0.0 s. The later insertion `" X"@[0.0,0.3)` then falls BEFORE the watermark
  // and `watermark_filtered` drops it from every hypothesis -- it is never
  // offered, never held, never confirmed. Finalized `" A X B C D"` becomes
  // `" A B C D"`.
  //
  // WHY IT WAS ACCEPTED: the alignment frontier Rule W replaced deleted words
  // on PHRASE RECURRENCE -- a shape real transcripts produce, and present at two
  // watermark positions of this crate's own canonical jfk phrase -- while this
  // deletes only inside a degenerate tie the driver cannot reach. Recorded as a
  // behaviour change in this module's `Documented deviations`.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A -- the trigger
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "CHARACTERIZATION (https://github.com/findit-studio/coremlit/issues/94): \
     Rule W widened this advance's split past the tie between \" A\"@[0.0,0.5) \
     and \" B\"@[0.0,1.0), so \" B\" is confirmed here and the watermark is 1.0 \
     rather than 0.0. Without the rule the split stops at 1 and this reads \
     [\" A\"]. If you changed the rule, that is the expectation -- assert \
     [\" A\"] and re-check the insertion below.",
  );
  assert_eq!(
    agreement.last_agreed_seconds(),
    1.0,
    "the watermark is the tie's far side, which is what deletes the insertion \
     below",
  );

  let inserted = || result_with_words(vec![word(" X", 0.0, 0.3), b(), c(), word(" D", 2.0, 2.5)]);
  agreement.ingest_streamed(inserted());
  assert!(agreement.ingest_streamed(inserted()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "\" X\"@[0.0,0.3) starts before the 1.0 s watermark, so it never reaches a \
     hypothesis at all",
  );
  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert_eq!(
    text, " A B C D",
    "CHARACTERIZATION, and a DELETION -- this module's non-preferred direction. \
     The CORRECT answer is \" A X B C D\": \" X\" reproduces nothing confirmed, \
     both hypotheses agreed on it, and nothing contradicted it, so no rule may \
     drop it. It is dropped because Rule W moved the watermark past \" X\"'s \
     span one advance earlier. Accepted at \
     https://github.com/findit-studio/coremlit/issues/94 as the price of the \
     postcondition (`confirmed.last().start() < last_agreed_seconds` strictly, \
     which makes re-admission unrepresentable): the rule it replaced deleted \
     words on phrase recurrence, which real transcripts produce, while this \
     needs a zero-duration word the driver never emits. If that trade was \
     revisited, assert \" A X B C D\" here and delete this message.",
  );
}

#[test]
fn a_recurring_phrase_does_not_delete_the_words_between_its_occurrences() {
  // #94, the re-admission FRONTIER's own deletion failure mode -- the direction
  // this section calls "worse than the duplication the ledger records", reached
  // by the rule that closed the ledger. It fires whenever a phrase RECURS inside
  // one decode window, which is the shape Whisper's repetition loop manufactures
  // and which the crate's own canonical phrase contains twice.
  //
  // The frontier ALIGNED `settled ++ holdback` as a WHOLE, and a globally
  // optimal alignment is free to explain the settled word by an EARLIER
  // occurrence and the whole holdback by a LATER one. The seam then cut past
  // both, and every word BETWEEN the two occurrences was refused -- never
  // confirmed, never held, and never re-offerable, because the advance that
  // follows moves the watermark past them. This test was RED at 8f30eda and is
  // green now that the frontier is gone; it is the falsifier that decided the
  // rule had to be deleted rather than repaired.
  //
  //   settled  [" the"]                       confirmed, end == watermark
  //   holdback [" cat", " sat"]               forced into the next decode
  //   offered  [" cat"," sat"," on"," the"," cat"," sat"," down"]
  //   scores   [2, 2, 2, 2, 3, 2, 1, 1]       best 3, smallest optimal seam 4
  //   REFUSED  [" cat"," sat"," on"," the"]   kept [" cat"," sat"," down"]
  //
  // `prev_words` is `[" cat", " sat"]` and the kept head is `[" cat", " sat"]`,
  // so the two still AGREE. The advance fires, installs the LATER occurrence as
  // the holdback, drops the earlier one, and moves the watermark from 1.0 s to
  // 5.0 s -- past " on" and past the second " the", which the stream produced
  // and nothing revised.
  //
  //   ingest    [the@0, cat@1, sat@2] twice        -> confirmed [the], held [cat, sat]
  //   ingest    [the@0, cat@1, sat@2, on@3, the@4, cat@5, sat@6, down@7]
  let the = |start: f32| word(" the", start, start + 1.0);
  let cat = |start: f32| word(" cat", start, start + 1.0);
  let sat = |start: f32| word(" sat", start, start + 1.0);
  let opening = || result_with_words(vec![the(0.0), cat(1.0), sat(2.0)]);

  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" the"],
    "confirmed [the], holding [cat, sat] at a 1.0 s watermark",
  );

  // The window has grown far enough ahead of the watermark to contain the
  // phrase a second time -- the only thing the green golden run lacks.
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
        the(0.0),
        cat(1.0),
        sat(2.0),
        word(" on", 3.0, 4.0),
        the(4.0),
        cat(5.0),
        sat(6.0),
        word(" down", 7.0, 8.0),
      ]))
      .is_advanced(),
    "the refused prefix leaves both sides agreeing, so the watermark still moves",
  );

  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert!(
    text.contains(" on"),
    "the word between the two occurrences was deleted: {text:?}"
  );
  assert_eq!(
    text.matches(" the").count(),
    2,
    "both occurrences of the recurring word must reach the transcript: {text:?}"
  );
}

// ---------------------------------------------------------------------
// What Rule W deliberately LEAVES, and what it costs
// ---------------------------------------------------------------------
//
// Documented residuals and accepted costs, each with a test, rather than a
// claim of totality. See `LocalAgreement::watermark_filtered` and this
// module's `Documented deviations`.

#[test]
fn rule_w_deletes_an_unaccounted_repeat_of_a_settled_word() {
  // CHARACTERIZATION of Rule W, not a property that holds. The CORRECT answer
  // is in the failure message below.
  //
  // This sequence is the issue's own two-run construction with the HOLDBACKS
  // equal too: the engine holds `confirmed = [A]`, `holdback = [B, C]`, and is
  // offered `[A, A, B, C]`. The clip stuttering (" A A B C" is right) and the
  // decode duplicating (" A B C" is right) produce byte-identical inputs, down
  // to every `WordTiming` field, so nothing the engine holds can separate them.
  // The adjudicated bias was to KEEP: `tests/whisper/streaming.rs`'s portable
  // prefix property tolerates a truncation and forbids a rewrite, so a
  // duplicate is the recoverable error and a deletion is not.
  //
  // TRIGGER: the same tie as
  // `rule_w_deletes_a_tied_insertion_that_reproduces_nothing_confirmed` --
  // `" A"@[0.0,0.5)` and `" B"@[0.0,1.0)` share a start, which on this crate's
  // own words needs a zero-duration word and which
  // `LocalAgreementTranscriber` never produces.
  //
  // WHAT MOVES: Rule W widens the first advance past the tie, so the watermark
  // is 1.0 s and BOTH copies of `" A"@[0.0,0.5)` -- the settled one and the
  // stream's surplus one -- fall behind it. The surplus copy is filtered out
  // rather than kept, the repeated hypothesis carries only `" C"`, which is one
  // word short of `agreement_count_needed`, so the pair never agrees again and
  // `finalize` takes its `holdback_superseded` path. Finalized `" A A B C"`
  // becomes `" A B C"`.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A -- the trigger
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  assert_eq!(
    (confirmed_texts(&agreement), agreement.last_agreed_seconds()),
    (vec![" A", " B"], 1.0),
    "CHARACTERIZATION (https://github.com/findit-studio/coremlit/issues/94): \
     Rule W widened past the tie, so \" B\" is confirmed here and the watermark \
     is 1.0. Without the rule this reads ([\" A\"], 0.0) and the surplus repeat \
     below is still visible to the engine.",
  );

  let repeated = || result_with_words(vec![a(), a(), b(), c()]);
  assert_eq!(
    (
      agreement.ingest_streamed(repeated()),
      agreement.ingest_streamed(repeated())
    ),
    (
      AgreementOutcome::AwaitingAgreement,
      AgreementOutcome::AwaitingAgreement
    ),
    "both copies of \" A\" and \" B\" are behind the 1.0 s watermark, so each \
     hypothesis is the single word \" C\" -- short of `agreement_count_needed`, \
     so neither pair advances",
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A B C",
    "CHARACTERIZATION, and a DELETION -- this module's non-preferred direction, \
     taken here against its own adjudicated bias. The CORRECT answer is \
     \" A A B C\": nothing the engine holds can tell a stuttering clip from a \
     duplicating decode, and the standing bias is to KEEP the surplus copy \
     because a duplicate is recoverable through the portable prefix property \
     and a deletion is not. It is deleted because Rule W put the watermark past \
     both copies one advance earlier. Accepted at \
     https://github.com/findit-studio/coremlit/issues/94 as the price of the \
     postcondition (`confirmed.last().start() < last_agreed_seconds` strictly): \
     the alignment frontier it replaced deleted words on phrase recurrence, \
     which real transcripts produce, while this needs a zero-duration word the \
     driver never emits. If that trade was revisited, assert \" A A B C\" here \
     and delete this message.",
  );
}

// ---------------------------------------------------------------------
// The scope of `settled`: which EDGE of a confirmed word faces the span
// ---------------------------------------------------------------------

#[test]
fn a_re_utterance_separated_from_the_watermark_keeps_its_word() {
  // The COST BOUNDARY of the scope above, and the reason it is `end >=
  // watermark` rather than "always take the last confirmed word". This is
  // `a_word_whose_extent_reaches_the_watermark_is_in_the_alignment`'s mirror,
  // byte-identical in shape -- one word ahead of a full reproduction of the
  // holdback, whose text is a confirmed word's -- and it needs the OPPOSITE
  // answer, because here the stream genuinely says " B" again.
  //
  // What separates them is the only evidence there is: the confirmed " B" ENDS
  // 0.1 s before the watermark, with another word's audio (or silence) between
  // it and the re-decoded span, while the sibling's " A" ends ON it. The gap is
  // load-bearing. Widen the scope past it -- to the whole confirmed list, or by
  // an unconditional `.max(1)` on the run length -- and this reads back
  // [" A", " B", " C"], the stream's second " B" deleted, while the sibling
  // still passes. Both mutations were run. `.max(1)` reds exactly this test in
  // this module; the whole-list widening reds this one and
  // `a_confirmed_word_from_before_the_watermark_is_out_of_the_alignment`, which
  // is the pair of them bounding the run's near and far ends.
  //
  //   confirmed [A@0.0-0.4, B@0.5-0.9], holding [C@1.0, D@1.5], watermark 1.0
  //   ingest    [B@1.0, C@1.1, D@1.5, E@2.0] twice
  let a0 = || word(" A", 0.0, 0.4);
  let b0 = || word(" B", 0.5, 0.9);
  let c0 = || word(" C", 1.0, 1.4);
  let d0 = || word(" D", 1.5, 1.9);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a0(), b0(), c0(), d0()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a0(), b0(), c0(), d0()]))
      .is_advanced()
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "confirmed [A@0.0, B@0.5], holding [C@1.0, D@1.5] at a 1.0 s watermark",
  );

  let re_uttered = || {
    result_with_words(vec![
      word(" B", 1.0, 1.05),
      word(" C", 1.1, 1.4),
      word(" D", 1.5, 1.9),
      word(" E", 2.0, 2.4),
    ])
  };
  agreement.ingest_streamed(re_uttered());
  assert!(agreement.ingest_streamed(re_uttered()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B", " B", " C"],
    "the later \" B\" is the stream's own, and costs nothing that follows it",
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A B B C D E",
  );
}

#[test]
fn the_next_strides_prefill_is_the_holdback_verbatim() {
  // THE PREMISE, pinned as the fact it is: the engine WRITES the holdback into
  // the next hypothesis rather than asking for it. That is what makes the next
  // hypothesis a RE-AGREEMENT over the span the holdback covers rather than an
  // independent reading of it, and it is what `budgeted_split`'s whole budget
  // argument is about. This asserts the two halves of that contract that live in
  // this module:
  //
  //   * `prefix_tokens` is the holdback's own tokens, in order, concatenated --
  //     not a count, not a summary. `decode::prefill_tokens` appends them to the
  //     initial prompt and `decode::decode_text` forces every prompt position
  //     (`next_token = current_tokens[token_index]`), so they are copied into
  //     the hypothesis rather than predicted by it.
  //   * `use_prefill_prompt` is SET, because `WhisperKit::transcribe` calls
  //     `prefill_tokens` only when it is -- without it the tokens above are
  //     silently inert and the retargeting is a no-op.
  //
  // The prefill BUDGET is deliberately NOT asserted here: with the default
  // two-word agreement the prefix is two tokens, so `len() <=
  // MAX_HOLDBACK_PREFILL_TOKENS` would hold whatever the code did -- true by
  // construction is a gap, not a pass. It is pinned where it can fail instead,
  // by `an_over_budget_holdback_is_capped_rather_than_silently_truncated`.
  //
  // THESE ARE THE OPTIONS-LAYER FACTS ONLY -- what the engine RECORDED. What
  // `decode_text` is actually handed, after `prefill_tokens`' trim and
  // special-id filter, is pinned by
  // `the_prefill_reaches_decode_text_as_the_whole_holdback`, which needs the
  // tokenizer artifact and so runs under `--ignored` (codex round 6, finding 2:
  // this test alone was checking the wrong layer). Both are kept: this one is
  // hermetic and runs everywhere.
  //
  // Mutation proof: drop `.with_use_prefill_prompt()` from
  // `decoding_options_for_next` and the flag assertion reds. Prefill
  // `confirmed_words` instead of `last_agreed_words` and the token assertion
  // reds. Both were run.
  let mut agreement = LocalAgreement::new();
  let words = vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.8),
    word(" my", 0.8, 1.2),
  ];
  agreement.ingest_streamed(result_with_words(words.clone()));
  assert!(
    agreement
      .ingest_streamed(result_with_words(words))
      .is_advanced()
  );

  // The base has the prompt turned OFF, which is what makes the flag assertion
  // below non-vacuous: on the default base it would hold with no code at all.
  let base = crate::audio::whisper::options::DecodingOptions::new().maybe_use_prefill_prompt(false);
  assert!(!base.use_prefill_prompt());
  let next = agreement.decoding_options_for_next(&base);
  assert!(
    next.use_prefill_prompt(),
    "without this the prefix tokens below never reach the decoder",
  );
  assert_eq!(
    next.clip_timestamps_slice(),
    &[agreement.last_agreed_seconds()]
  );

  let holdback_tokens: Vec<u32> = agreement
    .last_agreed_words_slice()
    .iter()
    .flat_map(|word| word.tokens_slice().iter().copied())
    .collect();
  assert_eq!(
    next.prefix_tokens_slice(),
    holdback_tokens.as_slice(),
    "the prefill IS the holdback, token for token",
  );
  assert!(
    !holdback_tokens.is_empty(),
    "non-vacuous: an empty holdback would satisfy the equality above trivially",
  );
}

// ---------------------------------------------------------------------
// The prefill budget
// ---------------------------------------------------------------------

/// Nine abutting 30-token words: `[w0 .. w8]`, `wN` spanning `[N, N+1)`.
/// 30 tokens apiece puts four of them (120) over
/// [`MAX_HOLDBACK_PREFILL_TOKENS`] and three (90) under, so a five-word holdback
/// request straddles the budget.
fn budget_words() -> Vec<WordTiming> {
  (0..9u32)
    .map(|index| word_of_tokens(&format!(" w{index}"), index as f32, index as f32 + 1.0, 30))
    .collect()
}

#[test]
fn an_over_budget_holdback_is_capped_rather_than_silently_truncated() {
  // #94 (codex round 6, finding 2). `agreement_count_needed` has no upper bound
  // and `prefill_tokens` keeps only the last `MAX_TOKEN_CONTEXT / 2` prefix
  // tokens, so a wide enough agreement asks for a holdback the decoder will
  // never be given whole. The head of such a holdback is then neither reproduced
  // (the decoder never sees those tokens) nor confirmed (the next advance
  // REPLACES the holdback with the new `common[split..]`) -- it just vanishes.
  //
  // Holding back only what the prefill can carry, and confirming the rest, is
  // what closes it: `common` is already the prefix two consecutive hypotheses
  // agreed on, so those words meet LocalAgreement-2's whole criterion, and a
  // word outside the prefill budget is one no third hypothesis could revise
  // anyway.
  //
  // Mutation proof: make `budgeted_split` return its `requested` argument
  // unchanged and both faces red -- the confirmed list stalls at two words
  // instead of four, and the finalized text loses " w2 w3" entirely. Stop the
  // budget loop one word short (`split + 1 < common.len()`, round 7's own
  // defect) and `the_split_holds_back_exactly_what_the_prefill_budget_carries`
  // reds on the single-oversized-word row.
  let words = budget_words();
  let mut agreement = LocalAgreement::new().with_agreement_count_needed(5);

  let first = || result_with_words(words[..7].to_vec());
  agreement.ingest_streamed(first());
  assert!(agreement.ingest_streamed(first()).is_advanced());

  let requested: usize = agreement
    .last_agreed_words_slice()
    .iter()
    .map(|word| word.tokens_slice().len())
    .sum();
  assert!(
    requested <= MAX_HOLDBACK_PREFILL_TOKENS,
    "the holdback must fit the prefill budget: {requested} tokens",
  );
  assert_eq!(
    agreement.last_agreed_words_slice().len(),
    3,
    "five words were asked for; three is what 112 tokens can carry",
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" w0", " w1", " w2", " w3"],
    "the two words that could not be held are CONFIRMED, not dropped",
  );
  // The face that matters to the decoder: what the next stride will prefill.
  assert_eq!(
    agreement
      .decoding_options_for_next(&DecodingOptions::new())
      .prefix_tokens_slice()
      .len(),
    90,
    "the whole prefix survives `prefill_tokens`' 112-token trim",
  );

  // The stream continues, and the next hypotheses begin at the holdback -- which
  // is exactly what a prefill the decoder receives WHOLE produces, and what a
  // truncated one does not.
  let onward = || result_with_words(words[4..].to_vec());
  agreement.ingest_streamed(onward());
  assert!(agreement.ingest_streamed(onward()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" w0", " w1", " w2", " w3", " w4", " w5"],
    "confirmation kept moving, and nothing was retracted",
  );

  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " w0 w1 w2 w3 w4 w5 w6 w7 w8",
    "every word the stream produced reaches the transcript",
  );
}

#[test]
fn a_holdback_word_the_prefill_cannot_carry_is_confirmed_rather_than_held() {
  // #94 (codex round 7, finding 2). THE CAP MUST ALWAYS CAP. `budgeted_split`
  // moved the split later until the holdback fit -- but stopped while one word
  // was still left, so a single word whose OWN tokens exceed
  // `MAX_HOLDBACK_PREFILL_TOKENS` was held anyway and the cap silently did not
  // cap. `decoding_options_for_next` then issued a prefix `prefill_tokens`
  // trims, the decoder was fed the word's TAIL, and the hypothesis came back
  // with a word that is not the held one.
  //
  // What follows is DATA LOSS, not a stall -- the truncated hypothesis
  // disagrees, `holdback_superseded` fires, and `finalize` replaces the intact
  // held word with the truncation.
  //
  // Holding it is what cannot be done; CONFIRMING it always can. `common` is
  // already the prefix two consecutive hypotheses agreed on -- LocalAgreement-2's
  // whole criterion -- and a word outside the prefill budget is one no third
  // hypothesis could revise, being behind both the forced prefill and the clip
  // window. So the split runs all the way to `common.len()` when it has to, and
  // the holdback the advance leaves can be EMPTY.
  //
  // Mutation proof: stop `budgeted_split`'s loop one word early again
  // (`split + 1 < common.len()`) and every assertion below reds -- the oversized
  // word is held instead of confirmed, the issued prefix is 113 tokens against a
  // 112-token budget, and the transcript reads " A Htail X" with the intact
  // " H" gone. The empty PENDING half: make the advance's
  // `self.last_agreed_words.is_empty()` arm DEFER (`0` in place of
  // `widened.len()`) and both words wait for an anchor that can never arrive.
  //
  // This is also the state that says the forced arm may not simply refuse to
  // advance (see
  // `a_forced_empty_holdback_retracts_its_suffix_at_the_settled_instant`, the
  // round where refusing WOULD have saved a word): make that arm return `0`
  // instead of `common.len()` and the `is_advanced` assertion below reds.
  // Nothing lies beyond `common` here, so there is no anchor a wait could ever
  // be waiting for -- round 7's finding again.
  let a = || word_of_tokens(" A", 1.0, 2.0, 1);
  let huge = || word_of_tokens(" H", 2.0, 3.0, MAX_HOLDBACK_PREFILL_TOKENS + 1);
  assert!(
    huge().tokens_slice().len() > MAX_HOLDBACK_PREFILL_TOKENS,
    "non-vacuous: one word, and it alone cannot fit the prefill budget",
  );

  let mut agreement = LocalAgreement::new();
  let pair = || result_with_words(vec![a(), huge()]);
  agreement.ingest_streamed(pair());
  assert!(agreement.ingest_streamed(pair()).is_advanced());

  // The streaming face, read between pushes: both agreed words are settled,
  // because the second one is a word the engine can never hand a decoder whole.
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " H"],
    "a word the prefill cannot carry is CONFIRMED, never held",
  );
  assert!(
    agreement.last_agreed_words_slice().is_empty(),
    "and the holdback is empty rather than over budget",
  );

  // The face the decoder sees: whatever the engine issues, `prefill_tokens`
  // keeps ALL of it. This is the postcondition the cap now actually has.
  let next = agreement.decoding_options_for_next(&DecodingOptions::new());
  assert!(
    next.prefix_tokens_slice().len() <= MAX_HOLDBACK_PREFILL_TOKENS,
    "the issued prefix is {} tokens against a {MAX_HOLDBACK_PREFILL_TOKENS}-token \
     budget",
    next.prefix_tokens_slice().len(),
  );
  assert_eq!(
    agreement.last_agreed_seconds(),
    3.0,
    "with nothing held back, the still-open span starts where the confirmed one \
     ends",
  );

  // The continuation a truncated prefill would produce: fed " H"'s last 112
  // tokens, the decoder emits some other word over that same extent. Under the
  // cap that word is outside the clip window and outside the offered span
  // entirely, so it cannot displace the settled " H".
  let truncated = || {
    result_with_words(vec![
      word_of_tokens(" Htail", 2.0, 3.0, MAX_HOLDBACK_PREFILL_TOKENS),
      word_of_tokens(" X", 3.0, 4.0, 1),
    ])
  };
  assert!(
    agreement
      .ingest_streamed(truncated())
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A H X",
    "the finalized face: an intact confirmed word is not replaced by a \
     truncation of itself",
  );
}

#[test]
fn the_split_holds_back_exactly_what_the_prefill_budget_carries() {
  // `budgeted_split`'s POSTCONDITION and its MINIMALITY, written out longhand
  // rather than by calling the code under test, so a mutation cannot mutate its
  // own falsifier with it: the holdback the split leaves fits
  // `MAX_HOLDBACK_PREFILL_TOKENS`, and no EARLIER split would have.
  //
  // Mutation proof, every row enumerated by running it: make `budgeted_split`
  // the identity (`requested`) and rows 0, 1 and 2 red on the postcondition;
  // return `common.len()` unconditionally and rows 3 and 4 red on the MINIMALITY
  // clause instead; stop the loop at `split + 1 < common.len()` (round 7's
  // defect, the cap that did not cap) and row 2 reds.
  let plain =
    |text: &str, start: f32, count: usize| word_of_tokens(text, start, start + 1.0, count);

  // `(common, requested)`.
  let cases: Vec<(Vec<WordTiming>, usize)> = vec![
    // 0: one over-budget word at the holdback head.
    (
      vec![
        plain(" A", 0.0, 1),
        plain(" B", 1.0, MAX_HOLDBACK_PREFILL_TOKENS),
        plain(" C", 2.0, MAX_HOLDBACK_PREFILL_TOKENS),
      ],
      1,
    ),
    // 1: the budget blown by the SUM rather than by any single word.
    (
      vec![
        plain(" A", 0.0, 1),
        plain(" B", 1.0, MAX_HOLDBACK_PREFILL_TOKENS / 2 + 1),
        plain(" C", 2.0, MAX_HOLDBACK_PREFILL_TOKENS / 2 + 1),
      ],
      1,
    ),
    // 2: ONE word whose own tokens exceed the budget, so the split has to run to
    //    `common.len()` and leave the holdback EMPTY (codex round 7, finding 2 --
    //    a loop that stopped one word short held it anyway).
    (
      vec![
        plain(" A", 0.0, 1),
        plain(" B", 1.0, MAX_HOLDBACK_PREFILL_TOKENS + 1),
      ],
      1,
    ),
    // 3: nothing to do -- the whole holdback is in budget.
    (
      vec![
        plain(" A", 0.0, 1),
        plain(" B", 1.0, 1),
        plain(" C", 2.0, 1),
      ],
      1,
    ),
    // 4: nothing to do with an empty holdback either.
    (vec![plain(" A", 0.0, 1), plain(" B", 1.0, 1)], 2),
  ];

  let holdable = |words: &[WordTiming]| {
    words
      .iter()
      .map(|word| word.tokens_slice().len())
      .sum::<usize>()
      <= MAX_HOLDBACK_PREFILL_TOKENS
  };

  for (row, (common, requested)) in cases.iter().enumerate() {
    let split = budgeted_split(common, *requested);
    let texts: Vec<&str> = common.iter().map(WordTiming::word).collect();
    assert!(
      (*requested..=common.len()).contains(&split),
      "row {row} ({texts:?}): split {split} left the requested-to-end range",
    );
    assert!(
      holdable(&common[split..]),
      "row {row} ({texts:?}): the holdback at {split} is over \
       MAX_HOLDBACK_PREFILL_TOKENS, so `prefill_tokens` would silently trim it",
    );
    for earlier in *requested..split {
      assert!(
        !holdable(&common[earlier..]),
        "row {row} ({texts:?}): split {split} confirms more than it has to -- \
         {earlier} already holds a within-budget holdback",
      );
    }
  }
}

// ---------------------------------------------------------------------
// The prompt itself, at the layer the decoder reads it
// ---------------------------------------------------------------------

#[test]
fn the_first_strides_options_keep_the_callers_prefill_flag() {
  // #94 (codex round 6, finding 3). Before the first advance the engine holds no
  // holdback, so there is no premise to enforce -- and
  // forcing `use_prefill_prompt` there would silently swap the caller's bare
  // `<|startoftranscript|>` prompt for the full multilingual prefill.
  //
  // Mutation proof: force the flag unconditionally (the single
  // `.with_use_prefill_prompt()` this replaced) and the first assertion reds;
  // never force it and the last one reds.
  let base = DecodingOptions::new().maybe_use_prefill_prompt(false);
  assert!(!base.use_prefill_prompt());

  let mut agreement = LocalAgreement::new();
  let first = agreement.decoding_options_for_next(&base);
  assert!(
    !first.use_prefill_prompt(),
    "nothing is held back yet, so the caller's own flag stands",
  );
  assert!(
    first.prefix_tokens_slice().is_empty(),
    "and there is nothing for the prompt to have carried anyway",
  );
  assert_eq!(
    first.clip_timestamps_slice(),
    &[0.0],
    "the clip retarget is unconditional -- Swift `:364-367` sets it too",
  );

  let words = vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.8),
    word(" my", 0.8, 1.2),
  ];
  agreement.ingest_streamed(result_with_words(words.clone()));
  assert!(
    agreement
      .ingest_streamed(result_with_words(words))
      .is_advanced()
  );
  assert!(
    agreement
      .decoding_options_for_next(&base)
      .use_prefill_prompt(),
    "once there IS a holdback the flag is forced, or the prefix is inert and \
     the engine's own retarget promises text the decoder was never given",
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn the_first_strides_prompt_is_the_bare_start_of_transcript() {
  // The decoder-layer face of the test above: not what `DecodingOptions`
  // records, but the `initial_prompt` `WhisperKit::transcribe` builds from it
  // (`transcribe/mod.rs:394-405`) and hands to `decode_text`.
  //
  // Mutation proof: force `use_prefill_prompt` unconditionally in
  // `decoding_options_for_next` and the first prompt below becomes the
  // four-token multilingual prefill -- the very divergence this pins against.
  let tokenizer = tiny_tokenizer();
  let special = *tokenizer.special_tokens();
  let base = DecodingOptions::new().maybe_use_prefill_prompt(false);

  let agreement = LocalAgreement::new();
  let first = agreement.decoding_options_for_next(&base);
  assert_eq!(
    initial_prompt_for(&first, &tokenizer),
    vec![special.start_of_transcript_token()],
    "the caller asked for a bare SOT and the first stride still gives it one",
  );
  // Non-vacuous: this IS a different prompt, not the same tokens by luck.
  assert_eq!(
    crate::audio::whisper::decode::prefill_tokens(&first, &tokenizer, true),
    vec![
      special.start_of_transcript_token(),
      special.english_token(),
      special.transcribe_token(),
      special.time_token_begin(),
    ],
    "what the caller would have been given instead",
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn the_prefill_reaches_decode_text_as_the_whole_holdback() {
  // The contract `budgeted_split`'s budget argument rests on, pinned at the
  // layer that decides it. `DecodingOptions::prefix_tokens` is
  // only what the engine RECORDED; `prefill_tokens` keeps just the last
  // `MAX_TOKEN_CONTEXT / 2` of them and drops every id at or above
  // `special_token_begin` before `decode_text` sees a single token. What the
  // tie-break needs is that the WHOLE holdback survives that -- so this asserts
  // the assembled initial prompt ENDS with the holdback's tokens, in order.
  //
  // Mutation proof: prefill `confirmed_words` instead of `last_agreed_words` and
  // the tail below stops matching; make `budgeted_split` return `requested`
  // unchanged and the over-budget half loses its head tokens to the trim.
  let tokenizer = tiny_tokenizer();

  let mut agreement = LocalAgreement::new();
  let words = vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.8),
    word(" my", 0.8, 1.2),
  ];
  agreement.ingest_streamed(result_with_words(words.clone()));
  assert!(
    agreement
      .ingest_streamed(result_with_words(words))
      .is_advanced()
  );

  let holdback_tokens: Vec<u32> = agreement
    .last_agreed_words_slice()
    .iter()
    .flat_map(|word| word.tokens_slice().iter().copied())
    .collect();
  assert!(!holdback_tokens.is_empty(), "non-vacuous");
  let prompt = initial_prompt_for(
    &agreement.decoding_options_for_next(&DecodingOptions::new()),
    &tokenizer,
  );
  assert!(
    prompt.ends_with(&holdback_tokens),
    "decode_text is forced through the whole holdback: prompt {prompt:?} must \
     end with {holdback_tokens:?}",
  );

  // The wide-agreement case, where the budget is genuinely in play: five words
  // were asked for, and every token of what is actually held still lands.
  let budget = budget_words();
  let mut wide = LocalAgreement::new().with_agreement_count_needed(5);
  let first = || result_with_words(budget[..7].to_vec());
  wide.ingest_streamed(first());
  assert!(wide.ingest_streamed(first()).is_advanced());
  let wide_holdback: Vec<u32> = wide
    .last_agreed_words_slice()
    .iter()
    .flat_map(|word| word.tokens_slice().iter().copied())
    .collect();
  assert_eq!(
    wide_holdback.len(),
    90,
    "non-vacuous: a real multi-word prefill"
  );
  let wide_prompt = initial_prompt_for(
    &wide.decoding_options_for_next(&DecodingOptions::new()),
    &tokenizer,
  );
  assert!(
    wide_prompt.ends_with(&wide_holdback),
    "no token of the held-back words is left behind by the 112-token trim",
  );
}

#[test]
fn a_zero_duration_word_at_an_empty_holdback_is_not_re_confirmed() {
  // RULE W'S POSTCONDITION ON THE ONE PATH THAT STILL EMPTIES THE HOLDBACK.
  // With nothing held back there is no held word to measure the watermark
  // against, so `ingest` anchors it at the last confirmed word's END -- and for
  // a ZERO-DURATION word that is its own START, which used to leave it passing
  // `watermark_filtered`'s `start >= watermark` against its own confirmation and
  // being confirmed a SECOND time (#94 residual 1, characterized here until
  // codex round 1 on PR #95 found the same state on the default path).
  //
  // Closed, not characterized: the watermark is
  // `end.max(past_the_settled_instant(settled_high))`, the first instant
  // strictly past the settled start in BOTH the seconds `watermark_filtered`
  // compares and the samples `clip_timestamps` is rounded to. It refuses exactly
  // the settled word's own sample rather than moving a cliff, which is what an
  // `end + epsilon` tolerance would have done -- and unlike the `f32::next_up`
  // this used to take, it actually moves the clip (codex round 7 on PR #95,
  // finding 2).
  //
  // Only the PREFILL BUDGET can reach this state now. Rule W's own widening no
  // longer empties the holdback (it backs off instead --
  // `a_trailing_tied_run_never_confirms_itself_twice_at_the_default_count`), and
  // the back-off may not cross the budget floor, which here sits at
  // `common.len()`: `" Z"` alone exceeds `MAX_HOLDBACK_PREFILL_TOKENS`, so there
  // is nothing the prefill could carry whole (codex round 7, finding 2). It also
  // needs a non-default `agreement_count_needed` (here 1) and a zero-duration
  // word; `add_word_timestamps` never emits a 112-token word.
  //
  // The forced arm ADVANCES here with nothing to lose: both hypotheses are
  // `[" A", " Z"]`, so nothing lies beyond `common` for the advance to strand,
  // and there is no anchor a wait could ever be waiting for. The word this
  // costs -- `" B"` below -- arrives one hypothesis LATER, which is outside
  // what any split can see. Where the strand IS already visible the same
  // advance retracts it, which is the characterization in
  // `a_forced_empty_holdback_retracts_its_suffix_at_the_settled_instant`.
  //
  // Mutation proof: drop the `.max(past_the_settled_instant(settled_high))`
  // from `empty_holdback_anchor` and this reds with `" Z"` confirmed twice.
  let a = || word(" A", 0.0, 1.0);
  let zero = || word_of_tokens(" Z", 1.0, 1.0, MAX_HOLDBACK_PREFILL_TOKENS + 1);
  let b = || word(" B", 1.0, 2.0);

  let mut agreement = LocalAgreement::new().with_agreement_count_needed(1);
  let opening = || result_with_words(vec![a(), zero()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (
      confirmed_texts(&agreement),
      agreement.last_agreed_words_slice().len(),
    ),
    (vec![" A", " Z"], 0),
    "non-vacuous: the budget could hold nothing back, so this is the \
     empty-holdback watermark rather than a held word's start",
  );
  assert_eq!(
    agreement.last_agreed_seconds(),
    PAST_ONE_SECOND,
    "and \" Z\"'s own END is 1.0, so `end` alone would have put the watermark \
     on \" Z\"'s own start",
  );
  let last_confirmed_start = agreement
    .confirmed_words_slice()
    .last()
    .map(WordTiming::start)
    .expect("the confirmed list is non-empty");
  assert!(
    last_confirmed_start < agreement.last_agreed_seconds(),
    "the postcondition is TOTAL: {last_confirmed_start} is strictly before the \
     {} s watermark even with an empty holdback",
    agreement.last_agreed_seconds(),
  );

  // The re-decode offers " Z" back at the head of its word list. Nothing
  // displaces it: the holdback is empty, so `decoding_options_for_next` attaches
  // an empty prefix and the decoder was never fed a reproduction to begin with.
  // The watermark is the only thing standing between " Z" and a second
  // confirmation, and it now does stand there.
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![zero(), b()]))
      .is_awaiting_agreement()
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " Z"],
    "\" Z\" is one word the stream produced once and it reaches the confirmed \
     list once",
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A Z",
    "COST, recorded as this rule's residual: \" B\" is a genuinely NEW word \
     that begins at the same instant the zero-duration \" Z\" occupies, and no \
     watermark can admit it while refusing \" Z\" -- the two are byte-identical \
     to a timestamp filter. It is dropped. The trade is the module's own \
     adjudicated bias applied to the evidence: an unbounded re-confirmation \
     REWRITES the confirmed text and breaks the portable prefix property, while \
     this leaves a truncation, which that property tolerates.",
  );
}
#[test]
fn a_forced_empty_holdback_retracts_its_suffix_at_the_settled_instant() {
  // CHARACTERIZATION of Rule W's FORCED arm, not a property that holds. This
  // pins what the engine does TODAY; the CORRECT answer is in the failure
  // messages below.
  //
  // #94, codex round 3 on PR #95, SECOND FINDING. Where the budget FLOOR itself
  // reaches `common.len()` the split runs off the end unconditionally: it takes
  // a LAST word whose own tokens exceed `MAX_HOLDBACK_PREFILL_TOKENS`, since
  // nothing else runs `budgeted_split`'s loop off the end, and there is no
  // holdback the prefill could carry at any split. Confirming that word is round
  // 7 finding 2's own repair -- leaving it held is the data loss that finding
  // recorded.
  //
  // WHAT THAT COSTS: the advance empties the holdback, anchors the watermark at
  // `past_the_settled_instant` -- strictly past `" H"`'s instant -- and `" X"`, which the
  // newer hypothesis produced AT that instant beyond `common`, fails the offered
  // filter from then on. That is not a deletion of something never emitted: this
  // round's OWN `finalize` emits it, through `differentSuffix(prev,
  // hypothesis)`. What the next hypothesis costs is a RETRACTION of published
  // transcript, and `confirmed_words`' monotonicity -- the #89 property -- cannot
  // see it, because the retracted word was never confirmed.
  //
  // WHY IT IS ACCEPTED. Deferring here is what this branch tried between
  // `6987bec` and `b3ec5c6`: the round waited while a word beyond `common` would
  // be stranded, under two bounds that ended the wait. Measured over the
  // accumulated counterexample suite and the 512-trial sweep, the wait erased 26
  // words from the published transcript where this arm erases 10, and its count
  // bound was defeated by `At(0)` -- see the engine module's doc, "Why there is
  // no deferral". The narrower repair is kept: `sparing_watermark`'s
  // sparing fold lowers the anchor to spare every word beyond `common` that
  // starts strictly later
  // (`a_word_starting_strictly_later_lowers_the_watermark_instead_of_being_stranded`),
  // so what is lost is exactly the TIE -- residual 1's AT half, the one instant
  // no watermark serves once the settled start is fixed.
  //
  // WHICH HALF, and why the name is narrower than the exception. The second
  // postcondition's exception reads "at or below the highest confirmed start".
  // This test and `an_over_budget_tied_run_strands_its_suffix_at_the_settled_
  // instant` drive the AT half, which is what their names say. BELOW is the
  // other half and needs neither an empty holdback nor a tie -- a backwards word
  // start (codex round 7 on PR #95, finding 1) lets an INTERIOR split confirm
  // past a word it leaves unconfirmed, pinned in
  // `a_backward_start_from_the_segment_pipeline_does_not_strand_a_later_word`
  // and counted by the sweep as `backward_strands`.
  //
  // The state this rule may NOT swallow is
  // `a_holdback_word_the_prefill_cannot_carry_is_confirmed_rather_than_held`:
  // there the budget forces the empty holdback and nothing lies beyond `common`,
  // so there is nothing to lose and nothing to wait for either.
  //
  // Mutation proof, every row run: drop
  // `.max(past_the_settled_instant(settled_high))` from `empty_holdback_anchor`
  // and the state row reds at a 1.0 watermark, `" H"`
  // re-admissible. Drop the sparing FOLD and this stays green while
  // `a_word_starting_strictly_later_lowers_the_watermark_instead_of_being_stranded`
  // reds -- the fold cannot help an exact tie, which is why that test exists
  // beside this one.
  let a = || word(" A", 0.0, 1.0);
  let over = || word_of_tokens(" H", 1.0, 1.0, MAX_HOLDBACK_PREFILL_TOKENS + 1);
  let tied = || word(" X", 1.0, 1.0);
  let anchor = || word(" Y", 2.0, 3.0);
  // A SECOND word starting strictly later: the default count needs two agreed
  // words to advance over anything at all, and the tail below is what proves
  // this arm is not a stall.
  let anchor2 = || word(" Z", 3.0, 4.0);
  assert_eq!(
    (
      over().tokens_slice().len() > MAX_HOLDBACK_PREFILL_TOKENS,
      over().start() == tied().start(),
      over().end() == over().start(),
      tied().end() == tied().start(),
    ),
    (true, true, true, true),
    "non-vacuous: ONE word over the budget -- the forced arm's own condition, \
     not the tied run's aggregate -- and the word beyond `common` shares its \
     zero-duration instant",
  );

  let older = || result_with_words(vec![a(), over()]);
  let newer = || result_with_words(vec![a(), over(), tied()]);
  let later = || result_with_words(vec![a(), over(), tied(), anchor(), anchor2()]);

  let mut agreement = LocalAgreement::new();
  assert_eq!(
    agreement.agreement_count_needed(),
    DEFAULT_AGREEMENT_COUNT_NEEDED,
    "non-vacuous: the DEFAULT count, the only one the driver reaches",
  );
  agreement.ingest_streamed(older());
  assert!(
    agreement.ingest_streamed(newer()).is_advanced(),
    "the agreeing round ADVANCES: an agreeing round always does",
  );
  assert_eq!(
    (
      confirmed_texts(&agreement),
      held_back_texts(&agreement),
      agreement.last_agreed_seconds(),
    ),
    (vec![" A", " H"], Vec::new(), PAST_ONE_SECOND),
    "CHARACTERIZATION (https://github.com/findit-studio/coremlit/issues/94): \
     the budget floor reaches `common.len()`, so the split runs off the end -- \
     `[\" A\", \" H\"]` is confirmed, the holdback is empty, and the watermark \
     anchors strictly past `\" H\"`'s instant, which is `\" X\"`'s instant too. \
     The CORRECT answer keeps `\" X\"` offerable, and no watermark does that \
     while also clearing `\" H\"`'s own start. If the trade was revisited, \
     assert ([], [], 0.0) here and re-check the retraction below.",
  );

  // FINALIZE POINT ONE. The retraction is only visible across two of these, and
  // this is the one that publishes the word: the advance confirmed `common` and
  // `find_longest_different_suffix` adds `" X"` on top.
  assert_eq!(
    agreement
      .clone()
      .finalize(&DecodingOptions::new())
      .text()
      .to_string(),
    " A H X",
    "a stream ENDING here publishes \" X\" -- which is what makes losing it a \
     retraction rather than a word never emitted",
  );

  // FINALIZE POINT TWO, after a hypothesis that repeats " X" and carries an
  // anchor starting strictly later. `" X"` is below the watermark by now, so
  // the filter drops it from the hypothesis AND from the re-read previous
  // result -- both sides at once.
  agreement.ingest_streamed(later());
  assert_eq!(
    agreement
      .clone()
      .finalize(&DecodingOptions::new())
      .text()
      .to_string(),
    " A H Y Z",
    "CHARACTERIZATION, and a RETRACTION -- this module's non-preferred \
     direction (https://github.com/findit-studio/coremlit/issues/94). The \
     CORRECT answer is \" A H X Y Z\": both hypotheses produced \" X\", \
     nothing contradicted it, and the round before this one already published \
     it. It is gone because the empty holdback's watermark passed its instant, \
     and no watermark both passes \" H\"'s start and spares a word AT it. \
     Accepted as residual 1. If that was revisited, assert \" A H X Y Z\" \
     here and delete this message.",
  );

  // AND IT IS NOT A STALL: the anchor reaches `common` on the next ingest and
  // an ordinary interior split takes over.
  assert!(
    agreement.ingest_streamed(later()).is_advanced(),
    "the anchor's strictly later start opens an ordinary interior boundary",
  );
  assert_eq!(
    (
      confirmed_texts(&agreement)
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>(),
      held_back_texts(&agreement)
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>(),
      agreement.last_agreed_seconds(),
      agreement
        .finalize(&DecodingOptions::new())
        .text()
        .to_string(),
    ),
    (
      vec![" A".to_string(), " H".to_string()],
      vec![" Y".to_string(), " Z".to_string()],
      2.0,
      " A H Y Z".to_string(),
    ),
    "and the stream keeps moving: the tail is held under an interior split and \
     nothing is confirmed twice",
  );
}

/// One round of the STREAMING face — the outcome a caller reads back plus the
/// three pieces of engine state it can see between pushes. Built as a value so a
/// whole run of rounds compares in ONE assertion: a stall is a face that repeats,
/// and a repeating face is only visible across rounds.
fn streaming_face(
  outcome: AgreementOutcome,
  agreement: &LocalAgreement,
) -> (String, Vec<String>, Vec<String>, f32) {
  (
    outcome.to_string(),
    confirmed_texts(agreement)
      .into_iter()
      .map(str::to_string)
      .collect(),
    held_back_texts(agreement)
      .into_iter()
      .map(str::to_string)
      .collect(),
    agreement.last_agreed_seconds(),
  )
}

#[test]
fn an_alternating_suffix_advances_instead_of_stalling() {
  // #94, codex round 4 on PR #95 -- LIVENESS on the forced arm, kept past the
  // deferral it was written against.
  //
  // Hypotheses that ALTERNATE past the agreed prefix -- `[A, H, X]` then
  // `[A, H, Y]` then `[A, H, X]` -- pin `common` at `[A, H]` forever: the forced
  // arm sees the same over-budget `" H"` at the end of the same `common` every
  // round, and the same live suffix beyond it. Whatever the split does with that
  // state, it must not do it FOREVER: the watermark has to move, or the caller
  // reads `awaiting_agreement` while the driver's clip keeps re-decoding from a
  // boundary that never advances and the buffer grows past it.
  //
  // The split runs off the END of `common` on the first such round, so the
  // stream is a stream: `[" A", " H"]` is confirmed at the only instant that
  // clears `" H"`'s own start, the alternation stops blocking the tail, and
  // `[" T", " U"]` is held back under an ordinary interior split from then on.
  //
  // Between `6987bec` and `b3ec5c6` this round DEFERRED instead and the escape
  // came one round later; every face below is that same run shifted by one
  // ingest, and the two transcripts at the end are byte-identical either way.
  // The measurement that removed the deferral is in the engine module's doc
  // ("Why there is no deferral"): the wait cost 26 published erasures against
  // this fallback's 10 over the same 512 trials.
  //
  // WHAT THIS TRADES, stated in the direction that costs something. `" X"` and
  // `" Y"` sit at `" H"`'s own instant, so no watermark strictly past `" H"`'s
  // start can spare them (`sparing_watermark` lowers the anchor as far as
  // it can, and here it cannot lower it at all) -- this module's residual 1. A
  // stream that ENDS on the round before keeps the disputed word; a stream that
  // CONTINUES gets a transcript instead of a frozen one. Both are asserted
  // below. The disputed word is also the one thing no wait could ever have
  // delivered: `find_longest_common_prefix` stops at it by construction, so it
  // is uncorroborated on every round it is offered, and `finalize` already
  // published a DIFFERENT one each round.
  //
  // Mutation proof, every row run: drop the `.or_else(...)` back-off from
  // `split_at_a_strict_boundary` and this stays green (the forced arm never
  // reaches either search), which is why the back-off is pinned in
  // `a_trailing_tied_run_never_confirms_itself_twice_at_the_default_count`
  // instead. Drop `.max(past_the_settled_instant(settled_high))` from
  // `empty_holdback_anchor`
  // and the face reds at round 1 with a 1.0 watermark, `" H"` re-admissible.
  // Drop the sparing FOLD and the face reds at round 2 instead, `[" T", " U"]`
  // filtered away by a 3.0 anchor they never earned.
  let a = || word(" A", 0.0, 1.0);
  let over = || word_of_tokens(" H", 1.0, 1.0, MAX_HOLDBACK_PREFILL_TOKENS + 1);
  let x = || word(" X", 1.0, 1.0);
  let y = || word(" Y", 1.0, 1.0);
  // Two words a WAIT would have called relief. They start strictly later, they
  // are in every hypothesis, and they cannot reach `common` while the words in
  // front of them disagree -- which is why waiting for them never ended.
  let tail = || vec![word(" T", 2.0, 3.0), word(" U", 3.0, 4.0)];
  let odd = || result_with_words([vec![a(), over(), x()], tail()].concat());
  let even = || result_with_words([vec![a(), over(), y()], tail()].concat());

  assert_eq!(
    (
      over().tokens_slice().len() > MAX_HOLDBACK_PREFILL_TOKENS,
      x().start() == over().start(),
      y().start() == over().start(),
      x().word() == y().word(),
      tail()[0].start() > over().start(),
    ),
    (true, true, true, false, true),
    "non-vacuous: ONE word over the budget (the forced arm's own condition), \
     two DIFFERENT words alternating at its exact instant, and a tail that does \
     start strictly later",
  );

  let mut agreement = LocalAgreement::new();
  assert_eq!(
    agreement.agreement_count_needed(),
    DEFAULT_AGREEMENT_COUNT_NEEDED,
    "non-vacuous: the DEFAULT count, the only one the driver reaches",
  );

  const ROUNDS: usize = 8;
  let mut face = Vec::with_capacity(ROUNDS);
  let mut forced_arm = Vec::with_capacity(ROUNDS);
  let mut fallback_transcript = String::new();
  for round in 0..ROUNDS {
    let outcome = agreement.ingest_streamed(if round % 2 == 0 { odd() } else { even() });
    face.push(streaming_face(outcome, &agreement));
    forced_arm.push(forced_arm_with_a_live_suffix(&agreement));
    if round == 1 {
      fallback_transcript = agreement
        .clone()
        .finalize(&DecodingOptions::new())
        .text()
        .to_string();
    }
  }

  // NON-VACUOUS, per round: the forced arm really is the arm being driven, with
  // a live suffix beyond `common`, on the round that takes the fallback.
  // Without this the face below could be produced by an engine that never
  // reached the state at all.
  assert_eq!(
    forced_arm,
    vec![false, true, false, false, false, false, false, false],
    "the round that runs the split off the end must BE the forced arm with a \
     live suffix; round 0 has no previous hypothesis and rounds 2+ have \
     advanced past the over-budget word",
  );

  let awaiting = |confirmed: &[&str], held: &[&str], watermark: f32| {
    (
      "awaiting_agreement".to_string(),
      confirmed.iter().map(|w| (*w).to_string()).collect(),
      held.iter().map(|w| (*w).to_string()).collect(),
      watermark,
    )
  };
  let advanced = |confirmed: &[&str], held: &[&str], watermark: f32| {
    (
      "advanced".to_string(),
      confirmed.iter().map(|w| (*w).to_string()).collect(),
      held.iter().map(|w| (*w).to_string()).collect(),
      watermark,
    )
  };
  assert_eq!(
    face,
    vec![
      // No previous hypothesis to agree with.
      awaiting(&[], &[], 0.0),
      // THE FALLBACK. `common` is `[" A", " H"]` and the budget floor reaches
      // its end, so the empty holdback is taken at the only instant that clears
      // `" H"`'s own start.
      advanced(&[" A", " H"], &[], PAST_ONE_SECOND),
      // And the stream is a stream: the tail the alternation was blocking
      // reaches `common` and is held back under an ordinary interior split.
      advanced(&[" A", " H"], &[" T", " U"], 2.0),
      advanced(&[" A", " H"], &[" T", " U"], 2.0),
      advanced(&[" A", " H"], &[" T", " U"], 2.0),
      advanced(&[" A", " H"], &[" T", " U"], 2.0),
      advanced(&[" A", " H"], &[" T", " U"], 2.0),
      advanced(&[" A", " H"], &[" T", " U"], 2.0),
    ],
    "the alternation must not freeze the stream",
  );

  // THE TRADE, both directions, so neither can be reported as free. A stream
  // that ENDS on the round that took the fallback still publishes the disputed
  // word, through `find_longest_different_suffix`; one that CONTINUES past it
  // does not, and gets everything after it instead.
  assert_eq!(
    (
      fallback_transcript.as_str(),
      agreement
        .finalize(&DecodingOptions::new())
        .text()
        .to_string()
        .as_str(),
    ),
    (" A H Y T U", " A H T U"),
    "the empty holdback drops the word at the settled instant -- residual 1 -- \
     and keeps the transcript moving",
  );
}

#[test]
fn a_tied_run_above_the_budget_floor_advances_instead_of_stalling() {
  // #94, codex round 4 on PR #95 -- the SAME liveness question on the arm this
  // crate's own pipeline can actually reach, kept past the deferral it was
  // written against.
  //
  // The alternating shape above needs a single word carrying more than
  // `MAX_HOLDBACK_PREFILL_TOKENS` tokens. This one needs no over-budget word, no
  // alternation, and nothing beyond `common` at all: 113 ORDINARY one-token
  // words sharing one instant, offered UNCHANGED on every stride. Their SUM puts
  // the budget floor strictly inside the run, every boundary at or above it
  // ties, and the back-off may not cross the floor -- so both searches fail and
  // the split runs off the END of `common`. A repeated hypothesis grows nothing,
  // so the state is the same again next round and the round after.
  // `add_word_timestamps` produces exactly that shape from an ALL-ZERO alignment
  // matrix (`add_word_timestamps_zero_pads_missing_rows`; measured at 130 such
  // words), which is why this is the reachability answer for the finding rather
  // than the shape above.
  //
  // Nothing is stranded on the FIRST half: `beyond_common` is EMPTY, so the
  // watermark filters nothing away that any hypothesis had produced, and BOTH of
  // Rule W's postconditions hold across it unweakened. What a wait would have
  // been protecting is the HOLDBACK -- the tied run stays revisable and the next
  // stride keeps its prefill anchor -- and a repeated hypothesis never delivers
  // it.
  //
  // The SECOND half drives the same arm with a suffix that ALTERNATES at the
  // run's own instant, which is where a strand-conditional wait would have gone
  // on deferring forever. Both halves are here because they are one arm;
  // splitting them would let a repair pass on the half it happened to cover.
  //
  // Between `6987bec` and `b3ec5c6` the first agreeing round DEFERRED and the
  // advance came one round later; both faces below are that same run shifted by
  // one ingest, and the finalized transcript is byte-identical either way. See
  // the engine module's doc, "Why there is no deferral", for the measurement.
  //
  // Mutation proof, every row run: return `common.len() - 1` from
  // `split_at_a_strict_boundary`'s `unwrap_or` and the STABLE face reds at
  // round 1 -- 112 confirmed against 113, and a watermark of 2.0 that re-admits
  // the run's own last word. Drop `.max(past_the_settled_instant(settled_high))`
  // from `empty_holdback_anchor` and both faces red at a 2.0 watermark with 113
  // words still offerable, which is the unbounded re-confirmation #94 is about.
  // Let the back-off cross the floor (`0..widened` for `floor..widened`) and the
  // stable face reds at round 1 with a two-word holdback the prefill cannot
  // carry.
  const RUN: usize = MAX_HOLDBACK_PREFILL_TOKENS + 1;
  let tied: Vec<WordTiming> = (0..RUN)
    .map(|index| word(&format!(" w{index:03}"), 2.0, 2.0))
    .collect();
  assert_eq!(
    (
      tied.len(),
      tied.iter().all(|word| word.tokens_slice().len() == 1),
      tied
        .iter()
        .map(|word| word.tokens_slice().len())
        .sum::<usize>()
        > MAX_HOLDBACK_PREFILL_TOKENS,
      tied.iter().all(|word| word.start() == 2.0),
    ),
    (113, true, true, true),
    "non-vacuous: ORDINARY one-token words, no single one over budget, and it \
     is their SUM at ONE instant that puts the floor inside the run",
  );
  let stable = || result_with_words(tied.clone());

  let mut agreement = LocalAgreement::new();
  assert_eq!(
    agreement.agreement_count_needed(),
    DEFAULT_AGREEMENT_COUNT_NEEDED,
    "non-vacuous: the DEFAULT count, the only one the driver reaches",
  );

  const ROUNDS: usize = 6;
  let mut face = Vec::with_capacity(ROUNDS);
  for _ in 0..ROUNDS {
    let outcome = agreement.ingest_streamed(stable());
    face.push((
      outcome.to_string(),
      confirmed_texts(&agreement).len(),
      held_back_texts(&agreement).len(),
      agreement.last_agreed_seconds(),
      // Nothing this hypothesis produced falls below the watermark that is not
      // confirmed: the SECOND postcondition, read on every round of this shape
      // rather than swept, since the empty holdback is where it could have
      // broken.
      LocalAgreement::watermark_filtered(&stable(), agreement.last_agreed_seconds()).len()
        + agreement.confirmed_words_slice().len(),
    ));
  }
  assert_eq!(
    face,
    vec![
      ("awaiting_agreement".to_string(), 0, 0, 0.0, RUN),
      // THE FALLBACK, at the only instant that clears the run's own.
      ("advanced".to_string(), RUN, 0, PAST_TWO_SECONDS, RUN),
      // The run is behind the watermark now, so a hypothesis that keeps
      // repeating it offers nothing -- which is the input being degenerate, not
      // the engine stalling: every word it has is at one instant.
      (
        "awaiting_agreement".to_string(),
        RUN,
        0,
        PAST_TWO_SECONDS,
        RUN
      ),
      (
        "awaiting_agreement".to_string(),
        RUN,
        0,
        PAST_TWO_SECONDS,
        RUN
      ),
      (
        "awaiting_agreement".to_string(),
        RUN,
        0,
        PAST_TWO_SECONDS,
        RUN
      ),
      (
        "awaiting_agreement".to_string(),
        RUN,
        0,
        PAST_TWO_SECONDS,
        RUN
      ),
    ],
    "a hypothesis that simply REPEATS must not freeze the stream, and the \
     advance must strand nothing: every word is either still offerable or \
     confirmed",
  );

  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert_eq!(
    text.split_whitespace().count(),
    RUN,
    "and the whole run reaches the transcript, each word once: {text:?}",
  );

  // ── SECOND HALF: the same arm, with a live strand at the run's own instant.
  //
  // Two words alternate past the run, both at the run's instant, so no watermark
  // that clears the run's own start can spare either -- and a rule that refused
  // to advance while a strand existed would wait here forever, since the strand
  // is renewed by the next hypothesis every round.
  let x0 = || word(" x0", 2.0, 2.0);
  let x1 = || word(" x1", 2.0, 2.0);
  let odd = || result_with_words([tied.clone(), vec![x0()]].concat());
  let even = || result_with_words([tied.clone(), vec![x1()]].concat());
  assert_eq!(
    (
      x0().start() == tied[RUN - 1].start(),
      x0().word() == x1().word()
    ),
    (true, false),
    "non-vacuous: two DIFFERENT words alternating at the run's own instant,      which no watermark strictly past that instant can spare",
  );

  let mut alternating = LocalAgreement::new();
  let mut alternating_face = Vec::with_capacity(ROUNDS);
  for round in 0..ROUNDS {
    let outcome = alternating.ingest_streamed(if round % 2 == 0 { odd() } else { even() });
    alternating_face.push((
      outcome.to_string(),
      confirmed_texts(&alternating).len(),
      held_back_texts(&alternating).len(),
      alternating.last_agreed_seconds(),
    ));
  }
  assert_eq!(
    alternating_face,
    vec![
      ("awaiting_agreement".to_string(), 0, 0, 0.0),
      ("advanced".to_string(), RUN, 0, PAST_TWO_SECONDS),
      ("awaiting_agreement".to_string(), RUN, 0, PAST_TWO_SECONDS),
      ("awaiting_agreement".to_string(), RUN, 0, PAST_TWO_SECONDS),
      ("awaiting_agreement".to_string(), RUN, 0, PAST_TWO_SECONDS),
      ("awaiting_agreement".to_string(), RUN, 0, PAST_TWO_SECONDS),
    ],
    "an alternating suffix at the run's own instant must not freeze this arm \
     either -- the run is confirmed and the disputed words at its instant are \
     the residual, not a reason to wait",
  );
}

#[test]
fn a_word_starting_strictly_later_lowers_the_watermark_instead_of_being_stranded() {
  // #94, codex round 4 on PR #95 -- the SPARING fold, which keeps the empty
  // holdback from stranding anything it does not have to.
  //
  // The forced empty holdback used to anchor at `common.last().end()` flatly, so
  // every word the hypothesis had already produced beyond `common` that started
  // before that edge was filtered away. But word ENDS inside a hypothesis are
  // not monotone (`an_overlapping_agreed_word_is_confirmed_on_the_mainline_path_
  // too`), so `end` can reach past a word that is already there -- and nothing
  // needs it to. All the watermark owes #94 is to clear the last CONFIRMED
  // word's START. So `sparing_watermark` lowers the anchor to the
  // earliest such word's start when that still clears it, and the word stays
  // offerable instead of being lost.
  //
  // What that leaves is the EXACT TIE alone, which is the case no instant
  // serves -- this module's residual 1, characterized in
  // `a_forced_empty_holdback_retracts_its_suffix_at_the_settled_instant`.
  //
  // The fold is SEPARABLE from the deferral this branch briefly carried and was
  // kept when that was removed: it is the whole of what confines the loss to the
  // tie, and it is measured here rather than argued.
  //
  // Mutation proof, every row run: drop the `fold` and return
  // `past_the_confirmed_start` and this reds -- round 1 confirms `[" A", " H"]`
  // at a 3.0 watermark and `" X"` is gone from round 2's holdback, stranded by
  // an edge it never earned. Keep the fold but drop its `*start > last.start()`
  // filter and this stays green while `the_split_never_cuts_at_a_tied_start`
  // reds on a swept exact tie (`" C" at 1.5, which is not strictly before the
  // 1.5 s watermark`) -- the filter is postcondition ONE, asserted where it is
  // swept.
  let a = || word(" A", 0.0, 1.0);
  let over = || word_of_tokens(" H", 1.0, 3.0, MAX_HOLDBACK_PREFILL_TOKENS + 1);
  let strand = || word(" X", 2.0, 2.5);
  let tail = || word(" T", 3.5, 4.0);
  assert_eq!(
    (
      over().tokens_slice().len() > MAX_HOLDBACK_PREFILL_TOKENS,
      over().start() < strand().start(),
      strand().start() < over().end(),
    ),
    (true, true, true),
    "non-vacuous: the forced arm's own condition, and a word beyond `common` \
     that starts strictly AFTER the last agreed word's start and strictly \
     BEFORE its end -- the overlap the flat anchor used to strand",
  );

  let older = || result_with_words(vec![a(), over()]);
  let newer = || result_with_words(vec![a(), over(), strand(), tail()]);

  let mut agreement = LocalAgreement::new();
  assert_eq!(
    agreement.agreement_count_needed(),
    DEFAULT_AGREEMENT_COUNT_NEEDED,
    "non-vacuous: the DEFAULT count, the only one the driver reaches",
  );
  let mut face = Vec::new();
  for round in 0..3u32 {
    let outcome = agreement.ingest_streamed(if round == 0 { older() } else { newer() });
    face.push(streaming_face(outcome, &agreement));
  }
  assert_eq!(
    face,
    vec![
      (
        "awaiting_agreement".to_string(),
        Vec::new(),
        Vec::new(),
        0.0
      ),
      // The anchor is the strand's OWN start rather than `" H"`'s far edge, so
      // the advance clears the settled start without reaching past a word the
      // hypothesis has already produced.
      (
        "advanced".to_string(),
        vec![" A".to_string(), " H".to_string()],
        Vec::new(),
        2.0,
      ),
      // And `" X"` was never filtered away, so it is held back like any other
      // agreed word one round later.
      (
        "advanced".to_string(),
        vec![" A".to_string(), " H".to_string()],
        vec![" X".to_string(), " T".to_string()],
        2.0,
      ),
    ],
    "a word beyond `common` that starts strictly later is SPARED by lowering \
     the anchor, not stranded by the last agreed word's far edge",
  );

  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A H X T",
    "and nothing is lost",
  );
}

#[test]
fn an_overlapping_agreed_word_is_confirmed_on_the_mainline_path_too() {
  // #94 (codex round 13, finding 1), REFUTED RATHER THAN FIXED, and this is the
  // row that says why. The finding reads "a word is confirmed even though a
  // later decode could re-read its audio" as a hazard some particular arm of the
  // advance introduces. It is not: `common[..split]` -- the MAINLINE
  // confirmation, the one Swift has and this port has never touched -- appends
  // with no overlap test of any kind.
  //
  // Word ends inside a hypothesis are not monotone, so an agreed word can extend
  // past the first held-back word's start. Here " P" runs 0.0..5.0 while the
  // watermark lands at " Q"'s 1.0, and the " Y" that follows re-reads 1.0..5.0.
  // Both are in the transcript, overlapping.
  //
  // So "a confirmed word overlaps a later hypothesis's word" is the
  // LocalAgreement-2 contract itself, which confirms on agreement between two
  // consecutive hypotheses and is append-only. Whether an offered word is a
  // settled one coming BACK is a different question, and RULE W makes it
  // unaskable rather than answering it: the split never puts the watermark at a
  // tied start, so no confirmed word can pass the offered filter, and " Y" is
  // simply new text over a span " P" also covers. P/Q/R have distinct starts, so
  // Rule W does not fire here at all.
  //
  // Mutation proof: apply the finding's recommendation to the mainline --
  // confirm only the `position(|word| word.end() > self.last_agreed_seconds)`
  // prefix of `common[..split]` -- and the state below reads back [] : " P"
  // never reaches `confirmed_words_slice()` at all.
  let p = || word(" P", 0.0, 5.0);
  let q = || word(" Q", 1.0, 2.0);
  let r = || word(" R", 2.0, 3.0);
  let y = || word(" Y", 1.0, 5.0);
  let z = || word(" Z", 5.0, 6.0);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![p(), q(), r()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(agreement.last_agreed_seconds(), 1.0);
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" P"],
    "non-vacuous: \" P\" is confirmed by the mainline `common[..split]`",
  );
  assert!(
    p().end() > agreement.last_agreed_seconds(),
    "non-vacuous: the mainline path confirmed a word that OVERLAPS the \
     still-open span, with no overlap test anywhere on it",
  );

  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![y(), z()]))
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " P Y Z",
    "the overlapping confirmed word and the later re-reading of its audio are \
     BOTH in the transcript -- the append-only contract",
  );
}

// ── The two coordinate systems the watermark is read in ──────────────────────

#[test]
fn the_watermark_clears_the_settled_sample_not_just_the_settled_instant() {
  // #94, codex round 7 on PR #95, finding 2 -- the ARITHMETIC half, pinned
  // before the driver-level regression below reads its consequence.
  //
  // The empty-holdback anchor used to be `end.max(start.next_up())`, and
  // `f32::next_up` is the IMMEDIATE successor: a real step in SECONDS, which is
  // the coordinate `watermark_filtered` compares in. The same value is also
  // handed to `clip_timestamps`, where `chunker::prepare_seek_clips` rounds it
  // to a SAMPLE -- and one ULP of a small `f32` is worth a few thousandths of a
  // sample, so the step vanished. The rows below are the measurement: every
  // whole second's `next_up` clips to the settled word's OWN sample, so the
  // "strictly past" guarantee was real in float space and vacuous in sample
  // space, and the next stride re-read the audio the doc claimed it had clipped
  // away.
  //
  // Mutation proof: replace `past_the_settled_instant`'s body with
  // `settled.next_up()` and the `clip_seek_sample` column below reds on every
  // row.
  for settled in [0.0f32, 0.5, 1.0, 2.0, 3.0, 10.0, 0.07, 1.44] {
    let settled_sample = chunker::clip_seek_sample(settled);
    assert_eq!(
      chunker::clip_seek_sample(settled.next_up()),
      settled_sample,
      "the ULP step is INERT in sample space, which is the finding: \
       {settled} and {} both clip to sample {settled_sample}",
      settled.next_up(),
    );
    let boundary = past_the_settled_instant(settled);
    assert!(
      boundary > settled,
      "strictly past in SECONDS, which is what `watermark_filtered` compares: \
       {boundary} vs {settled}",
    );
    assert!(
      chunker::clip_seek_sample(boundary) > settled_sample,
      "and strictly past in SAMPLES, which is what `clip_timestamps` reads: \
       {boundary} clips to {} and {settled} clips to {settled_sample}",
      chunker::clip_seek_sample(boundary),
    );
  }
  // The two constants the empty-holdback characterizations assert, derived here
  // from the sample index rather than trusted at their literal value.
  assert_eq!(past_the_settled_instant(1.0), PAST_ONE_SECOND);
  assert_eq!(past_the_settled_instant(2.0), PAST_TWO_SECONDS);
  assert_eq!(chunker::clip_seek_sample(PAST_ONE_SECOND), 16_001);
  assert_eq!(chunker::clip_seek_sample(PAST_TWO_SECONDS), 32_001);
  // TOTAL on the degenerate inputs the fold can hand it: `+inf` is what
  // `f32::next_up` mapped to itself, so this keeps that answer rather than
  // spinning, and a NEGATIVE instant saturates to sample 0 on the way in.
  assert_eq!(past_the_settled_instant(f32::INFINITY), f32::INFINITY);
  assert!(past_the_settled_instant(-1.0) > 0.0);
}

/// A [`crate::audio::whisper::backend::mock::MockBackend`] that RECORDS the
/// first sample of every window `TranscribeTask` asks it to extract features
/// from, then delegates.
///
/// `TranscribeTask` calls `extract_features` on
/// `pad_or_trim(&audio[seek..seek + segment_size], window_samples)`, so over
/// RAMP audio (`samples[i] == i as f32`) that first sample IS the seek index —
/// the exact audio each stride re-read. That is the observation #94's codex
/// round 7 finding 2 is about, and it cannot be read off the engine's own
/// state: the watermark is in SECONDS and the re-read is in SAMPLES.
struct WindowRecordingBackend {
  inner: crate::audio::whisper::backend::mock::MockBackend,
  window_starts: std::sync::Mutex<Vec<usize>>,
}

impl WindowRecordingBackend {
  fn new(inner: crate::audio::whisper::backend::mock::MockBackend) -> Self {
    Self {
      inner,
      window_starts: std::sync::Mutex::new(Vec::new()),
    }
  }

  fn window_starts(&self) -> Vec<usize> {
    self
      .window_starts
      .lock()
      .expect("window-start record lock poisoned")
      .clone()
  }
}

impl InferenceBackend for WindowRecordingBackend {
  type Features = <crate::audio::whisper::backend::mock::MockBackend as InferenceBackend>::Features;
  type EncoderOutput =
    <crate::audio::whisper::backend::mock::MockBackend as InferenceBackend>::EncoderOutput;
  type DecoderState =
    <crate::audio::whisper::backend::mock::MockBackend as InferenceBackend>::DecoderState;

  fn extract_features(
    &self,
    audio: &[f32],
  ) -> Result<Self::Features, crate::audio::whisper::backend::BackendError> {
    if let Some(&first) = audio.first() {
      self
        .window_starts
        .lock()
        .expect("window-start record lock poisoned")
        .push(first as usize);
    }
    self.inner.extract_features(audio)
  }

  fn encode(
    &self,
    features: &Self::Features,
  ) -> Result<Self::EncoderOutput, crate::audio::whisper::backend::BackendError> {
    self.inner.encode(features)
  }

  fn new_decoder_state(
    &self,
  ) -> Result<Self::DecoderState, crate::audio::whisper::backend::BackendError> {
    self.inner.new_decoder_state()
  }

  fn reset_decoder_state(&self, state: &mut Self::DecoderState) {
    self.inner.reset_decoder_state(state);
  }

  fn decode_step(
    &self,
    token: u32,
    position: usize,
    encoder_output: &Self::EncoderOutput,
    state: &mut Self::DecoderState,
    logits: &mut Vec<f32>,
  ) -> Result<(), crate::audio::whisper::backend::BackendError> {
    self
      .inner
      .decode_step(token, position, encoder_output, state, logits)
  }

  fn commit_alignment_row(&self, state: &mut Self::DecoderState) {
    self.inner.commit_alignment_row(state);
  }

  fn alignment_weights<'state>(
    &self,
    state: &'state Self::DecoderState,
  ) -> Option<crate::audio::whisper::backend::AlignmentView<'state>> {
    self.inner.alignment_weights(state)
  }

  fn dims(&self) -> crate::audio::whisper::backend::ModelDims {
    self.inner.dims()
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn the_driver_does_not_re_read_the_settled_words_own_sample() {
  // #94, codex round 7 on PR #95, finding 2 -- the DRIVER-level regression, and
  // the reason the arithmetic pin above is not enough on its own. The finding is
  // not that a float compared wrong; it is that the value the engine calls "the
  // first instant strictly past the settled start" is ALSO the next stride's
  // clip start, and `chunker::prepare_seek_clips` rounds it to a sample -- where
  // the `f32::next_up` step it used to take moved nothing. So the driver handed
  // the decoder the settled word's OWN audio again while this module's residuals
  // 3 and 6 claimed that audio was outside the clip window. Those notes were
  // wrong for a concrete arithmetic reason and are corrected with this test.
  //
  // The state is reached THROUGH the driver, not asserted about it: the scripted
  // window decodes `" Hi" " World" " Helloaaa..."`, whose last word carries 114
  // tokens against a 112-token prefill budget and lands ZERO-DURATION at 1.0 s
  // (its own alignment column and the closing timestamps' are the same). So
  // `budgeted_split`'s floor reaches `common.len()`, the advance confirms all
  // three words, the holdback empties, and `empty_holdback_anchor` -- not a held
  // word's start -- is what the next clip is drawn from. That is the ONE arm
  // where the anchor's own sample matters: give the last word any real duration
  // and `end` already clears the settled sample by itself.
  //
  // Mutation proof: replace `past_the_settled_instant`'s body with
  // `settled.next_up()` and the final window start below reds at 16_000 -- the
  // settled word's own sample -- against 16_001.
  use crate::audio::whisper::{
    backend::{ModelDims, mock::MockBackend},
    tokenizer::SpecialTokens,
    transcribe::WhisperKit,
  };

  let tokenizer = tiny_tokenizer();
  let specials = SpecialTokens::whisper_defaults();
  let hi = tokenizer.encode(" Hi").unwrap()[0];
  let world = tokenizer.encode(" World").unwrap()[0];
  let hello = tokenizer.encode(" Hello").unwrap()[0];
  // No leading space, so `split_to_word_tokens` appends every one of these to
  // `" Hello"` rather than starting a new word -- which is how one WORD comes to
  // carry more tokens than the prefill can hold.
  let filler = tokenizer.encode("a").unwrap()[0];
  let one_hot = |token: u32| {
    let mut logits = vec![0.0f32; 51_865];
    logits[token as usize] = 10.0;
    logits
  };

  let mut mock = MockBackend::new().with_dims(
    ModelDims::new()
      .with_window_samples(16_000)
      .with_n_audio_ctx(100),
  );
  // `(token, alignment peak column)`. 100 columns over the window put one column
  // at 0.02 s, so column 50 is 1.0 s. `" Hello"` and all 113 fillers peak on THAT
  // column and the closing timestamps peak past it, which is what leaves the
  // last word zero-duration at exactly 1.0 s.
  let mut script: Vec<(u32, usize)> = vec![
    (specials.english_token(), 1),
    (specials.transcribe_token(), 2),
    (specials.time_token_begin(), 3),
    (hi, 10),
    (world, 50),
    (hello, 50),
  ];
  script.extend(std::iter::repeat_n((filler, 50), 113));
  script.push((specials.time_token_begin() + 100, 60));
  script.push((specials.time_token_begin() + 100, 60));
  script.push((specials.end_token(), 60));
  for (token, peak) in &script {
    let mut row = vec![0.0f32; 100];
    row[*peak] = 1.0;
    mock.push_step_with_alignment(one_hot(*token), row);
  }

  // The temperature LADDER is switched off and the sampler SEEDED, so this
  // scripted window decodes the same way every run. The 113-token filler run
  // trips `needs_fallback`'s compression-ratio check, and a fallback attempt
  // samples at a non-zero temperature -- from an unseeded draw by default, which
  // made the assertions below depend on the draw rather than on the split.
  let kit = WhisperKit::with_backend(WindowRecordingBackend::new(mock), tokenizer);
  let mut streamer = kit
    .local_agreement_transcriber(
      DecodingOptions::new()
        .with_temperature_fallback_count(0)
        .with_seed(94),
    )
    .with_agreement_count_needed(1);
  // A RAMP, so the backend's record of each window's first sample is that
  // window's seek index.
  let ramp: Vec<f32> = (0..64_000).map(|index| index as f32).collect();
  let mut chunks = ramp.chunks(STRIDE_SAMPLES);
  for chunk in chunks.by_ref().take(3) {
    streamer.push_samples(chunk).unwrap();
  }

  let settled = streamer
    .agreement()
    .confirmed_words_slice()
    .last()
    .expect("the advance confirmed the whole agreed prefix")
    .clone();
  assert_eq!(
    (
      streamer.agreement().confirmed_words_slice().len(),
      streamer.agreement().last_agreed_words_slice().len(),
      settled.tokens_slice().len() > MAX_HOLDBACK_PREFILL_TOKENS,
      settled.start(),
      settled.end(),
    ),
    (3, 0, true, 1.0, 1.0),
    "non-vacuous: the DRIVER reached the empty-holdback arm, and the word it \
     settled last is the ZERO-DURATION one -- the only shape where the anchor's \
     own sample decides the next clip",
  );
  assert_eq!(
    chunker::clip_seek_sample(settled.start()),
    16_000,
    "the settled word's own audio is sample 16_000",
  );

  let before = kit.backend().window_starts();
  streamer.push_samples(chunks.next().unwrap()).unwrap();
  let after = kit.backend().window_starts();
  let fresh: Vec<usize> = after[before.len()..].to_vec();
  assert!(
    !fresh.is_empty(),
    "non-vacuous: the last stride actually decoded a window",
  );
  assert_eq!(
    fresh[0], 16_001,
    "the stride after the empty-holdback advance must start at the sample AFTER \
     the settled word's own -- {fresh:?}. At 16_000 the driver re-reads the very \
     audio the watermark claims to have settled, which is what an `f32::next_up` \
     anchor left it doing: a real step in seconds and none at all in samples.",
  );
  assert!(
    fresh
      .iter()
      .all(|&start| start > chunker::clip_seek_sample(settled.start())),
    "and no window of that stride reaches back into it: {fresh:?}",
  );
}

// ── Backwards word starts, from the pipeline that emits them ─────────────────

/// The three segments and the monotone alignment that make
/// `segment::update_segments_with_word_timings` emit a BACKWARDS word start,
/// plus the words it emits. Shared by the engine regression below and by the
/// sweep's own shape note.
///
/// Nothing here is hand-timed: every value goes in as an alignment
/// `find_alignment` could produce (`w[i].end() <= w[i + 1].start()`, the
/// guarantee `segment::tests` pins) and comes out through Swift's own
/// segment-start preference, `SegmentSeeker.swift:635-640`. That branch prefers
/// the SEGMENT's start over a first word the DTW drifted more than half a second
/// earlier, and clamps to `w0.end - constrained_median` — which for the last
/// segment here is `1.51 - 0.70 = 0.81`, BELOW the `0.99` of the word in front
/// of it.
///
/// The 0.70 median is `calculate_word_duration_constraints`' own ceiling
/// (`median.min(0.7)`), which any speech whose median word runs 0.7 s or longer
/// reaches.
fn a_pipeline_result_with_a_backwards_start(
  tokenizer: &crate::audio::whisper::tokenizer::WhisperTokenizer,
) -> Vec<WordTiming> {
  use crate::audio::whisper::{
    result::TranscriptionSegment, segment::update_segments_with_word_timings,
  };

  let ids: Vec<u32> = [" A", " B", " C", " D"]
    .iter()
    .map(|text| tokenizer.encode(text).unwrap()[0])
    .collect();
  let plain = |tokens: Vec<u32>, start: f32, end: f32| {
    let mut segment = TranscriptionSegment::new();
    segment.set_tokens(tokens).set_start(start).set_end(end);
    segment
  };
  let segments = [
    plain(vec![ids[0]], 0.50, 0.70),
    plain(vec![ids[1], ids[2]], 0.80, 0.99),
    // The drifted one: the timestamp tokens put this segment at 1.50 while DTW
    // put its first word at 0.99 — a 0.51 s disagreement, which is what opens
    // `:635-640`.
    plain(vec![ids[3]], 1.50, 1.60),
  ];
  let alignment = [
    WordTiming::new(" A", vec![ids[0]], 0.50, 0.70, 0.9),
    WordTiming::new(" B", vec![ids[1]], 0.80, 0.99, 0.9),
    WordTiming::new(" C", vec![ids[2]], 0.99, 0.99, 0.9),
    WordTiming::new(" D", vec![ids[3]], 0.99, 1.51, 0.9),
  ];
  assert!(
    alignment
      .windows(2)
      .all(|pair| pair[0].end() <= pair[1].start() + 1e-4),
    "the INPUT is what `find_alignment` guarantees: non-decreasing and \
     non-overlapping. Everything backwards below is the post-processing's doing.",
  );
  update_segments_with_word_timings(&segments, &alignment, 0, 0.0, 0.70, 1.40, tokenizer)
    .unwrap()
    .iter()
    .flat_map(TranscriptionSegment::words_slice)
    .cloned()
    .collect()
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn a_backward_start_from_the_segment_pipeline_does_not_strand_a_later_word() {
  // #94, codex round 7 on PR #95, finding 1. Rule W's SECOND postcondition —
  // an advance may not push the watermark past a word of its own hypothesis it
  // did not confirm — used to be free on an INTERIOR split, and the reason given
  // was that "word starts inside one hypothesis are non-decreasing, so
  // everything from `split` on is at or past the watermark". That premise is
  // FALSE, and this test is the disproof: `update_segments_with_word_timings`
  // emits `[0.50, 0.80, 0.99, 0.81]` from a strictly non-decreasing alignment.
  //
  // What the premise cost: the split lands at index 2 (`0.80 < 0.99` is a legal
  // strict boundary), so the watermark used to be `" C"`'s own `0.99` — and
  // `" D"` at `0.81`, a word this round CONFIRMED NOTHING about and is still
  // holding back, falls below it. The next worded ingest filters it out of both
  // hypotheses at once and `finalize` can never reach it again, after this
  // round's own `finalize` has already published it. That is a RETRACTION on an
  // interior split, outside the split-at-the-end exception the doc claimed was
  // the only one.
  //
  // The repair is `sparing_watermark`, which the empty-holdback arm already had:
  // the anchor drops to the earliest unconfirmed start it can still clear. Here
  // that is `" D"`'s own `0.81`, which is above the `0.80` this round settled
  // last, so nothing is stranded at all.
  //
  // Mutation proof: return `anchor` from `sparing_watermark` (drop the fold) and
  // this reds with a `0.99` watermark and `" D"` unreachable.
  let tokenizer = tiny_tokenizer();
  let words = a_pipeline_result_with_a_backwards_start(&tokenizer);
  let starts: Vec<f32> = words.iter().map(WordTiming::start).collect();
  assert_eq!(
    (
      words.iter().map(WordTiming::word).collect::<Vec<_>>(),
      starts[0],
      starts[1],
      starts[2],
    ),
    (vec![" A", " B", " C", " D"], 0.50, 0.80, 0.99),
    "non-vacuous: the first three words are where the alignment put them",
  );
  assert!(
    starts[3] < starts[2] && starts[3] > starts[1],
    "THE FINDING: the pipeline's own post-processing put `\" D\"` at {} — behind \
     `\" C\"`'s {} and ahead of `\" B\"`'s {}. Word starts inside one hypothesis \
     are NOT non-decreasing.",
    starts[3],
    starts[2],
    starts[1],
  );

  let offered = || result_with_words(words.clone());
  let mut agreement = LocalAgreement::new();
  assert_eq!(
    agreement.agreement_count_needed(),
    DEFAULT_AGREEMENT_COUNT_NEEDED,
    "non-vacuous: the DEFAULT count, which is the driver's own",
  );
  agreement.ingest_streamed(offered());
  assert!(agreement.ingest_streamed(offered()).is_advanced());
  assert_eq!(
    (
      confirmed_texts(&agreement),
      held_back_texts(&agreement),
      agreement.last_agreed_seconds(),
    ),
    (vec![" A", " B"], vec![" C", " D"], starts[3]),
    "the watermark is the BACKWARDS word's own start, not the first held word's: \
     an interior split may not pass a word it is still holding back. Before this \
     repair it was {} — `\" C\"`'s start — and `\" D\"` was stranded below it.",
    starts[2],
  );

  // The two postconditions, read on this round rather than argued about.
  let settled_high = highest_start(agreement.confirmed_words_slice()).unwrap();
  assert!(
    settled_high < agreement.last_agreed_seconds(),
    "FIRST postcondition: every confirmed word starts strictly before the \
     watermark. Highest confirmed start {settled_high}, watermark {}",
    agreement.last_agreed_seconds(),
  );
  let below: Vec<(&str, f32)> = agreement
    .hypothesis_words
    .iter()
    .skip(agreement.confirmed_words_slice().len())
    .filter(|word| word.start() < agreement.last_agreed_seconds())
    .map(|word| (word.word(), word.start()))
    .collect();
  assert!(
    below.is_empty(),
    "SECOND postcondition: nothing this round left unconfirmed is below the \
     watermark, so nothing was stranded: {below:?}",
  );

  // And the round trip: the very next hypothesis still carries `" D"`, which is
  // the whole point — a stranded word disappears from BOTH hypotheses at once.
  assert!(
    LocalAgreement::watermark_filtered(&offered(), agreement.last_agreed_seconds())
      .iter()
      .any(|held| held.word() == " D"),
    "`\" D\"` is still offerable, so a later hypothesis can still revise or \
     corroborate it",
  );
  assert!(agreement.ingest_streamed(offered()).is_advanced());
  assert!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text()
      .contains('D'),
    "and it reaches the published transcript rather than being retracted out of \
     it",
  );
}

#[test]
fn a_backwards_start_two_words_back_still_cannot_be_re_admitted() {
  // THE SCOPE of the first postcondition, and the reason it is stated over the
  // WHOLE confirmed list rather than over its last word (#94, codex round 7 on
  // PR #95, finding 1).
  //
  // `split_at_a_strict_boundary` used to compare each candidate boundary
  // against its ADJACENT predecessor, and the doc justified that with "word
  // starts inside one hypothesis are non-decreasing, so an earlier confirmed
  // word starts at or before the last one". `a_backward_start_from_the_segment_
  // pipeline_does_not_strand_a_later_word` disproves the premise. This pins what
  // the premise was carrying: with starts `[0.50, 1.15, 0.95, 1.10]` the
  // adjacent test passes at index 3 (`0.95 < 1.10`), the advance confirms
  // `" Q"` at `1.15`, and the watermark lands at `1.10` — so `" Q"` satisfies
  // `watermark_filtered`'s own `start >= watermark` against its own confirmation
  // and can head the next hypothesis. That is #94 itself, reached from the
  // backwards side.
  //
  // REACHABILITY, stated rather than implied. This exact shape is NOT one
  // `update_segments_with_word_timings` can emit: its only backwards mover is
  // the `SegmentSeeker.swift:635-640` clamp, which needs `w0` to span more than
  // 0.5 s and lowers it only to `w0.end - median`, and every word after it
  // starts at or after `w0`'s own alignment END — which is at or after
  // everything in front of it. So the word AFTER a backwards one cannot also
  // duck below the word two back, which is what this shape needs. The check is
  // therefore a STRENGTHENING that removes a premise rather than a repair for a
  // demonstrated route: `ingest` is `pub(crate)` and residual 4 already records
  // that an in-crate caller can order its calls freely, and re-deriving a new
  // bound from the clamp's arithmetic is the kind of reasoning this issue has
  // punished twice. The falsifier is hermetic because the input is.
  //
  // Mutation proof: restore the adjacent-predecessor test in
  // `split_at_a_strict_boundary` (`common[at - 1].start()` for
  // `settled_before[at]`) and this reds — alone, at 503 of 504 green.
  let offered = || {
    result_with_words(vec![
      word(" P", 0.50, 0.60),
      word(" Q", 1.15, 1.20),
      word(" R", 0.95, 1.05),
      word(" S", 1.10, 1.30),
    ])
  };
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(offered());
  assert!(agreement.ingest_streamed(offered()).is_advanced());
  assert_eq!(
    (
      confirmed_texts(&agreement),
      held_back_texts(&agreement),
      agreement.last_agreed_seconds(),
    ),
    (vec![" P"], vec![" Q", " R", " S"], 0.95),
    "the forward search finds no boundary at or above the requested split -- \
     both `0.95` and `1.10` are below the `1.15` already inside `common` -- so \
     the back-off lands at index 1, and the watermark is the lowest unconfirmed \
     start it can still clear",
  );
  for confirmed in agreement.confirmed_words_slice() {
    assert!(
      confirmed.start() < agreement.last_agreed_seconds(),
      "EVERY confirmed word starts strictly before the {} s watermark, not just \
       the last: {:?} at {}",
      agreement.last_agreed_seconds(),
      confirmed.word(),
      confirmed.start(),
    );
  }
  assert!(
    LocalAgreement::watermark_filtered(&offered(), agreement.last_agreed_seconds())
      .iter()
      .all(|offered| offered.word() != " P"),
    "and nothing confirmed can head the next hypothesis",
  );
}
