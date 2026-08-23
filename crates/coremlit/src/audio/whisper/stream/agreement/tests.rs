use super::*;
use crate::audio::whisper::{
  result::{TranscriptionResult, TranscriptionSegment, TranscriptionTimings, WordTiming},
  task_facts::TaskFacts,
};

fn word(text: &str, start: f32, end: f32) -> WordTiming {
  WordTiming::new(text, vec![start as u32 + 1], start, end, 0.9)
}

// NOTE: this task's own brief's literal snippet called `TranscriptionResult::
// new()` with no arguments, then chained `.set_segments(...)`/
// `.set_language(...)`. The shipped constructor is four-argument
// (`TranscriptionResult::new(text, segments, language, timings)` — that
// type's own doc: "Builds a result from its four required fields ... has no
// defaults for these either") — same brief-vs-shipped-API fix as
// `tests/pipeline.rs`'s `tiny_options`/`tests/parity_jfk.rs`. Both call sites
// below pass the real values directly instead.
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
    agreement.ingest(first).is_awaiting_agreement(),
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
  assert!(agreement.ingest(second).is_advanced());
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
  agreement.ingest(result_with_words(vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.7),
  ]));
  let disagreeing = result_with_words(vec![word(" But", 0.0, 0.4), word(" then", 0.4, 0.7)]);
  assert!(agreement.ingest(disagreeing).is_awaiting_agreement());
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
  assert!(agreement.ingest(wordless).is_no_word_timings());
  assert_eq!(agreement.results_slice().len(), 1);
}

#[test]
fn finalize_appends_agreed_tail_plus_different_suffix_and_merges() {
  // TranscribeCLI.swift:418-421.
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![
    word(" And", 0.0, 0.4),
    word(" so", 0.4, 0.7),
    word(" my", 0.7, 1.0),
  ]));
  agreement.ingest(result_with_words(vec![
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
  agreement.ingest(result);
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
  agreement.ingest(result_with_words(vec![word(" And", 0.0, 0.4)]));
  let second = result_with_words(vec![word(" And", 0.0, 0.4), word(" so", 0.4, 0.7)]);
  // A single-word common prefix ([And]) already meets a threshold of 1 —
  // it would NOT at the default threshold of 2.
  assert!(agreement.ingest(second).is_advanced());
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
  agreement.ingest(result_with_words(vec![word(" And", 0.0, 0.4)]));
  let second = result_with_words(vec![word(" And", 0.0, 0.4), word(" so", 0.4, 0.7)]);
  agreement.ingest(second); // must not panic
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
  let outcome = agreement.ingest(result);
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
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  agreement.ingest(result_with_words(vec![a(), b(), c(), d()]));
  agreement.ingest(result_with_words(vec![a(), b(), c(), d(), e()]));
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
  // hypothesis left — `watermark_filtered_with`'s strip only removes words that
  // ACTUALLY reproduce the confirmed tail, so a candidate the hypothesis does
  // not reproduce costs nothing.
  //
  // Mutation proof: replace that strip with the unconditional count-skip
  // (`let strip = readmit_candidates.len().min(filtered.len());`) and B is never
  // confirmed at all.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
  let c = || word(" C", 1.0, 2.0);
  let d = || word(" D", 2.0, 3.0);
  let e = || word(" E", 3.0, 4.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  agreement.ingest(result_with_words(vec![a(), b(), c(), d()])); // confirms A, holds B,C
  // The rewrite omits A entirely: B must survive to be confirmed next.
  agreement.ingest(result_with_words(vec![b(), c(), d(), e()]));
  let confirmed: Vec<&str> = agreement
    .confirmed_words_slice()
    .iter()
    .map(|w| w.word())
    .collect();
  assert!(
    confirmed.contains(&" B"),
    "B lost to the positional skip: {confirmed:?}"
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
  assert!(agreement.ingest(r1).is_awaiting_agreement());
  assert!(
    agreement.ingest(r2).is_awaiting_agreement(),
    "R2 disagrees with R1 and is dropped from results",
  );
  assert!(
    agreement.ingest(r3).is_advanced(),
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
  assert!(agreement.ingest(r1).is_awaiting_agreement());
  assert!(
    agreement.ingest(r2).is_awaiting_agreement(),
    "R2 disagrees with R1 and is dropped from results",
  );
  assert!(
    agreement.ingest(r3).is_advanced(),
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
  assert!(agreement.ingest(r1).is_awaiting_agreement());
  assert!(
    agreement.ingest(r2).is_awaiting_agreement(),
    "R2 disagrees with R1 and is dropped from results",
  );
  assert!(
    agreement.ingest(r3).is_advanced(),
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
      .ingest(result_with_words(vec![
        word(" alpha", 0.0, 0.4),
        word(" bravo", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_awaiting_agreement(),
    "first result: nothing to agree with",
  );
  assert!(
    agreement
      .ingest(result_with_words(vec![
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
      .ingest(result_with_words(vec![
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
      .ingest(result_with_words(vec![
        word(" and", 0.0, 0.4),
        word(" so", 0.4, 0.7),
      ]))
      .is_awaiting_agreement()
  );
  // A one-word common prefix is short of the default threshold of 2, so this
  // disagrees even though both hypotheses produced " and".
  assert!(
    agreement
      .ingest(result_with_words(vec![
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
  agreement.ingest(result_with_words(vec![
    word(" alpha", 0.0, 0.4),
    word(" bravo", 0.4, 0.7),
    word(" charlie", 0.7, 1.0),
  ]));
  assert!(
    agreement
      .ingest(result_with_words(vec![
        word(" alpha", 0.0, 0.4),
        word(" bravo", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_advanced()
  );
  assert!((agreement.last_agreed_seconds() - 0.4).abs() < 1e-6);

  // Words, but every one of them before the 0.4 s watermark.
  let outcome = agreement.ingest(result_with_words(vec![word(" alpha", 0.0, 0.4)]));
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
  agreement.ingest(result_with_words(vec![
    word(" alpha", 0.0, 0.4),
    word(" bravo", 0.4, 0.7),
    word(" charlie", 0.7, 1.0),
  ]));
  assert!(
    agreement
      .ingest(result_with_words(vec![
        word(" alpha", 0.0, 0.4),
        word(" bravo", 0.4, 0.7),
        word(" charlie", 0.7, 1.0),
      ]))
      .is_advanced()
  );
  // Revises the held-back " bravo": disagrees, dropped, holdback superseded.
  assert!(
    agreement
      .ingest(result_with_words(vec![
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
      .ingest(result_with_words(vec![
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
  agreement.ingest(settled());
  assert!(agreement.ingest(settled()).is_advanced());
  assert_eq!(confirmed_texts(&agreement), vec![" A"]);
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" B", " C"],
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
    agreement.ingest(revision()).is_awaiting_agreement(),
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
  agreement.ingest(settled());
  assert!(agreement.ingest(settled()).is_advanced());
  agreement.ingest(revision());
  assert!(
    agreement.ingest(revision()).is_advanced(),
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
// The re-admission ledger: CHARACTERIZATIONS of a defect that is OPEN
// ---------------------------------------------------------------------
//
// `watermark_filtered`'s strip only catches a reproduction the hypothesis puts
// at the very FRONT of its post-watermark list. Every test below is a sequence
// that slides past it and confirms a settled word a second time (or, in the last
// one, deletes a word the stream genuinely produced).
//
// Each one asserts WHAT THE CODE DOES TODAY -- the wrong answer -- and carries
// the CORRECT expectation in its failure message. They are green on this tree,
// and they go RED the day someone fixes
// <https://github.com/findit-studio/coremlit/issues/94>, handing the fixer the
// value to change the expectation to. Run the whole module, both halves at
// once:
//
//     cargo test -p coremlit --features whisper --lib -- \
//         audio::whisper::stream::agreement::tests::
//
// WHY CHARACTERIZATION RATHER THAN `#[ignore]`. An earlier revision marked
// these ten `#[ignore]` and called them "red on purpose". That is the wrong
// marker, for two independent reasons -- do not revert to it:
//
//   - libtest's `--ignored` is ignored-ONLY, not skip-the-ignored: it SELECTS
//     every ignored test in the target and runs it. This repository's model
//     gates (.github/workflows/ci.yml, the `whisper|@all` row) and its sharded
//     coverage legs (.github/workflows/coverage.yml, the same row) both invoke
//     `-- --ignored`, so an `#[ignore]`d test is not parked -- it is scheduled,
//     and ten deliberately-red tests turn every one of those runs red.
//   - An ignored test never executes, so it rots silently as the code moves
//     under it. A characterization test runs on every push: it pins today's
//     behaviour exactly, it cannot drift unnoticed, and its going red is the
//     signal that the defect is gone.
//
// `#[ignore]` here marks a test CI cannot run unconditionally: it needs an
// artifact this checkout may not have (a staged model, a tokenizer sidecar, a
// fixture tree), or one specific host, or a cost CI will not pay. What it must
// never mean is "this test is expected to fail" -- `--ignored` runs it.
//
// Four adversarial-review rounds established that no predicate over (confirmed
// list, offered list, watermark) can decide these: occurrence identity is not
// recoverable from what the rule sees. The argument, the three defeated
// approaches and the recommended direction are in the issue.

#[test]
fn an_insertion_before_a_reproduced_confirmed_word_reconfirms_it_today() {
  // OPEN (codex round 2; the round-4 F1 "all tied at the watermark" variant).
  // `watermark_filtered_with` zips the confirmed tail against the offered list
  // from BOTH fronts, so it only removes a reproduction the hypothesis puts at
  // the very FRONT of its post-watermark list. Insert a word at that same
  // instant AHEAD of the reproduction and the zip mismatches at offset 0, strips
  // nothing, and the already-confirmed " A" rides through into
  // `hypothesis_words` -- which `finalize` then appends wholesale.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //   ingest    [X@0.0, A@0.0, B@0.0, C@1.0]
  //   TODAY     " A X A B C"   <- " A" confirmed twice; asserted below
  //   CORRECT   " A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
  let c = || word(" C", 1.0, 2.0);
  let x = || word(" X", 0.0, 0.3); // an insertion at that same instant
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A"],
    "confirmed [A], holding [B, C] at a 0.0 s watermark",
  );

  agreement.ingest(result_with_words(vec![x(), a(), b(), c()]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A X A B C",
    "CHARACTERIZATION of open defect #94, not a requirement -- this is what the \
     re-admission rule produces today. CORRECT is \" A B C\": the settled A wins \
     the instant it was confirmed at, and the insertion is not smuggled in \
     beside it. RED here means you fixed #94 -- change this expectation to \
     \" A B C\" and the count below to 1.",
  );
  assert_eq!(
    final_result.text().matches('A').count(),
    2,
    "CHARACTERIZATION of open defect #94: \" A\" reaches the transcript TWICE \
     today. CORRECT is 1. Text: {:?}",
    final_result.text(),
  );
}

#[test]
fn an_insertion_before_a_reproduced_confirmed_word_reconfirms_it_on_the_advance_today() {
  // OPEN. The same defect WITHOUT `finalize`: `ingest`'s advance folds
  // `common[..split]` -- a slice of `hypothesis_words` -- straight into
  // `confirmed_words`. Two consecutive hypotheses that both insert ahead of the
  // reproduced confirmed word agree on the whole insertion, so the duplicate
  // reaches `confirmed_words_slice()` on the streaming path a caller reads
  // between pushes, not only in the finalized text. Fixing `finalize` alone
  // would not reach this one.
  //
  //   ingest [A,B,C] [A,B,C] [X,A,B,C] [X,A,B,C]  (all tied at 0.0 but C)
  //   TODAY     confirmed_words_slice() == [" A", " X", " A"]
  //   CORRECT   [" A"]
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0);
  let c = || word(" C", 1.0, 2.0);
  let x = || word(" X", 0.0, 0.3);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  agreement.ingest(result_with_words(vec![x(), a(), b(), c()]));
  agreement.ingest(result_with_words(vec![x(), a(), b(), c()]));

  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " X", " A"],
    "CHARACTERIZATION of open defect #94, not a requirement -- no finalize \
     involved, so fixing `finalize` alone would not reach this one. CORRECT is \
     [\" A\"]: the advance path must not re-confirm A either. RED here means you \
     fixed #94 -- change this expectation to vec![\" A\"].",
  );
}

#[test]
fn a_holdback_over_a_reproduced_confirmed_word_reconfirms_it_today() {
  // OPEN. The sibling at `finalize`'s OTHER append: `confirmed_words.append(&mut
  // last_agreed_words)`. `last_agreed_words` is `common[split..]`, itself a
  // slice of `hypothesis_words`, so an already-confirmed word the strip missed
  // reaches `confirmed_words` through the HOLDBACK rather than through the
  // divergence branch. With `split == 0` nothing is confirmed at the advance and
  // the whole agreed prefix -- insertion and re-admitted " A" alike -- is held
  // back, then flushed by `finalize`.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //   ingest    [X@0.0, A@0.0] twice -- they agree on [X, A], split is 0
  //   TODAY     " A X A"       <- " A" confirmed twice, " B C" lost with it
  //   CORRECT   " A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0);
  let c = || word(" C", 1.0, 2.0);
  let x = || word(" X", 0.0, 0.3);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  // Two hypotheses that agree on exactly [X, A]: `split` is 0, so the pair lands
  // in the holdback rather than in `confirmed_words`.
  agreement.ingest(result_with_words(vec![x(), a()]));
  agreement.ingest(result_with_words(vec![x(), a()]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A X A",
    "CHARACTERIZATION of open defect #94, not a requirement -- this is what the \
     holdback flushes today. CORRECT is \" A B C\": the holdback must not flush a \
     second copy of the confirmed A, and it must not lose \" B C\" doing it. RED \
     here means you fixed #94 -- change this expectation to \" A B C\".",
  );
}

#[test]
fn a_different_suffix_over_a_reproduced_confirmed_word_reconfirms_it_today() {
  // OPEN. The sibling at `finalize`'s THIRD append: `find_longest_different_suffix(
  // &prev_words, &hypothesis_words)`, also a slice of `hypothesis_words`. Here
  // the re-admitted " A" sits PAST the common prefix -- two hypotheses agree on
  // an inserted [P, Q] and then diverge, and the newer one's reproduction of the
  // confirmed " A" rides in on the differing suffix.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //   ingest    [P@0.0, Q@0.0, Z@0.0] then [P@0.0, Q@0.0, A@0.0]
  //   TODAY     " A P Q A"     <- " A" confirmed twice
  //   CORRECT   " A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0);
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  agreement.ingest(result_with_words(vec![
    word(" P", 0.0, 0.1),
    word(" Q", 0.0, 0.2),
    word(" Z", 0.0, 0.3),
  ]));
  agreement.ingest(result_with_words(vec![
    word(" P", 0.0, 0.1),
    word(" Q", 0.0, 0.2),
    a(),
  ]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A P Q A",
    "CHARACTERIZATION of open defect #94, not a requirement -- this is what the \
     differing suffix flushes today. CORRECT is \" A B C\": the differing suffix \
     must not flush a second copy of the confirmed A. RED here means you fixed \
     #94 -- change this expectation to \" A B C\".",
  );
}

#[test]
fn a_later_re_use_of_a_confirmed_word_is_not_mistaken_for_a_re_admission() {
  // The BOUND on `watermark_filtered_with`'s strip, and phase-gate round 2's
  // rule restated for it: a candidate the hypothesis does not reproduce must
  // cost nothing. " A" is confirmed at 0.0 s and the clip says " A" again at
  // 2.0 s — a different word at a different instant, with two provisional words
  // in front of it. Zipping the confirmed tail against the offered list from
  // both fronts stops at the first mismatch, so the later " A" is out of reach;
  // a rule that SCANS for a text match instead would find it and take " B" and
  // " C" down with it. That is the failure mode a stricter replacement rule has
  // to avoid, which is why this pin is green rather than ignored.
  //
  // Mutation proof: scan the whole offered list for the last text match
  // (`filtered.iter().rposition(|w| candidates.contains(&normalized(w.word())))
  // .map_or(0, |i| i + 1)`) and this reads back " A D E" -- " B" and " C",
  // agreed by both hypotheses, gone.
  let a0 = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with the confirmed A
  let c = || word(" C", 1.0, 2.0);
  let a2 = || word(" A", 2.0, 2.5); // the same TEXT, a different word
  let d = || word(" D", 3.0, 3.5);
  let e = || word(" E", 4.0, 4.5);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a0(), b(), c()]));
  agreement.ingest(result_with_words(vec![a0(), b(), c()]));
  agreement.ingest(result_with_words(vec![b(), c(), a2(), d()]));
  agreement.ingest(result_with_words(vec![b(), c(), a2(), d(), e()]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B C A D E",
    "both A's belong in the transcript, and neither costs the words before it",
  );
}

// ---------------------------------------------------------------------
// The ledger, continued: two shapes with TWO confirmed words tied at the
// watermark
// ---------------------------------------------------------------------

/// The shared history for the two ledger entries below: two identical hypotheses
/// whose first THREE words tie at 0.0 s, so the advance confirms `[A, B]`,
/// holds `[C, D]`, and leaves the watermark at 0.0 s with TWO confirmed words
/// still at or past it.
fn tied_pair_confirmed() -> LocalAgreement {
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![
    tied_a(),
    tied_b(),
    tied_c(),
    tied_d(),
  ]));
  assert!(
    agreement
      .ingest(result_with_words(vec![
        tied_a(),
        tied_b(),
        tied_c(),
        tied_d()
      ]))
      .is_advanced(),
  );
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B"],
    "confirmed [A, B], holding [C, D] at a 0.0 s watermark",
  );
  agreement
}

fn tied_a() -> WordTiming {
  word(" A", 0.0, 0.5)
}
fn tied_b() -> WordTiming {
  word(" B", 0.0, 0.6)
}
fn tied_c() -> WordTiming {
  word(" C", 0.0, 0.7)
}
fn tied_d() -> WordTiming {
  word(" D", 1.0, 1.5)
}

#[test]
fn a_confirmed_tail_reproduced_without_its_head_is_reconfirmed_today() {
  // OPEN (codex round 2, counterexample 1) -- the CONFIRMED-TAIL OMISSION.
  // `watermark_filtered_with` compares the confirmed tail from ITS OWN front, so
  // a hypothesis that OMITS " A" and reproduces only " B" mismatches at offset 0,
  // strips nothing, and " B" rides through into `hypothesis_words`.
  //
  //   confirmed [A@0.0, B@0.0], holding [C@0.0, D@1.0], watermark 0.0
  //   ingest    [B@0.0, C@0.0, D@1.0]
  //   TODAY     " A B B C D"   <- " B" confirmed twice
  //   CORRECT   " A B C D"
  let mut agreement = tied_pair_confirmed();
  agreement.ingest(result_with_words(vec![tied_b(), tied_c(), tied_d()]));
  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B B C D",
    "CHARACTERIZATION of open defect #94, not a requirement -- this is what the \
     confirmed-tail omission produces today. CORRECT is \" A B C D\": the \
     confirmed B must not be confirmed a second time by a reproduction that \
     skipped A. RED here means you fixed #94 -- change this expectation to \
     \" A B C D\".",
  );
}

#[test]
fn a_confirmed_tail_reproduced_without_its_head_is_reconfirmed_on_the_advance_today() {
  // OPEN. The same omission WITHOUT `finalize`. Repeating the hypothesis makes
  // the two agree, and `ingest`'s advance folds `common[..split]` -- which still
  // carries the reproduced " B" -- straight into the caller-visible
  // `confirmed_words_slice()` a streaming caller reads between pushes.
  //
  //   ingest [B,C,D] twice on top of the same history
  //   TODAY     confirmed_words_slice() == [" A", " B", " B"]
  //   CORRECT   [" A", " B"]
  let mut agreement = tied_pair_confirmed();
  agreement.ingest(result_with_words(vec![tied_b(), tied_c(), tied_d()]));
  agreement.ingest(result_with_words(vec![tied_b(), tied_c(), tied_d()]));
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B", " B"],
    "CHARACTERIZATION of open defect #94, not a requirement -- no finalize \
     involved, so fixing `finalize` alone would not reach this one. CORRECT is \
     [\" A\", \" B\"]: the advance path must not re-confirm B either. RED here \
     means you fixed #94 -- change this expectation to vec![\" A\", \" B\"].",
  );
}

#[test]
fn a_reproduction_behind_a_shorter_false_match_is_reconfirmed_today() {
  // OPEN (codex round 2, counterexample 2) -- the PARTIAL FRONT MATCH.
  // `watermark_filtered_with` zips the confirmed tail [A, B] against the offered
  // list, matches " A" at offset 0, mismatches " B" against the inserted " X",
  // and stops -- so it strips one word and lets the real, longer " A B"
  // reproduction sitting at offset 2 ride through.
  //
  //   confirmed [A@0.0, B@0.0], holding [C@0.0, D@1.0], watermark 0.0
  //   ingest    [A@0.0, X@0.0, A@0.0, B@0.0, C@0.0, D@1.0]
  //   TODAY     " A B X A B C D"   <- " A" and " B" confirmed twice
  //   CORRECT   " A B C D"
  let mut agreement = tied_pair_confirmed();
  agreement.ingest(result_with_words(vec![
    tied_a(),
    word(" X", 0.0, 0.3),
    tied_a(),
    tied_b(),
    tied_c(),
    tied_d(),
  ]));
  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B X A B C D",
    "CHARACTERIZATION of open defect #94, not a requirement -- this is what the \
     partial front match produces today. CORRECT is \" A B C D\": the longer \
     reproduction behind the false match must still be stripped. RED here means \
     you fixed #94 -- change this expectation to \" A B C D\".",
  );
}

#[test]
fn a_reproduction_behind_a_shorter_false_match_is_reconfirmed_on_the_advance_today() {
  // OPEN. The same partial front match WITHOUT `finalize` -- two hypotheses that
  // agree on the whole `[A, X, A, B]` run advance, and `common[..split]` carries
  // both already-confirmed words into `confirmed_words_slice()`.
  //
  //   TODAY     confirmed_words_slice() == [" A", " B", " X", " A", " B"]
  //   CORRECT   [" A", " B"]
  let hypothesis = || {
    result_with_words(vec![
      tied_a(),
      word(" X", 0.0, 0.3),
      tied_a(),
      tied_b(),
      tied_c(),
      tied_d(),
    ])
  };
  let mut agreement = tied_pair_confirmed();
  agreement.ingest(hypothesis());
  agreement.ingest(hypothesis());
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B", " X", " A", " B"],
    "CHARACTERIZATION of open defect #94, not a requirement -- no finalize \
     involved, so fixing `finalize` alone would not reach this one. CORRECT is \
     [\" A\", \" B\"]: the advance path must not re-confirm A or B either. RED \
     here means you fixed #94 -- change this expectation to \
     vec![\" A\", \" B\"].",
  );
}

// ---------------------------------------------------------------------
// What the re-admission rule must NOT break
// ---------------------------------------------------------------------
//
// The three pins below are green on this tree AND on `main`: they are the
// constraints any replacement rule has to keep satisfying, recorded next to the
// characterizations above so a future fix is measured against both halves at
// once -- the ten go RED when #94 is fixed, these three must STAY green.
// Deleting a word the stream genuinely produced is the failure mode a
// stricter rule falls into, and it is worse than the duplication above --
// `tests/whisper/streaming.rs`'s portable prefix property tolerates a
// truncation and forbids a rewrite.

#[test]
fn a_stutter_at_the_watermark_keeps_both_occurrences() {
  // Codex round 3, finding 2, first half -- the half `watermark_filtered` gets
  // right, kept green so the characterized second half
  // (`a_distinct_repetition_of_a_confirmed_word_is_deleted_by_the_continuing_stream_today`)
  // is pinned from both ends. A hypothesis that STUTTERS at the watermark's own
  // instant -- " A" twice, same text, same start, same end -- straddles the
  // advance's split: the first " A" is confirmed, the second is held back, and
  // `finalize` flushes it. Nothing distinguishes that survivor from a
  // reproduction of the word just confirmed, which is why the rule must not run
  // a second time over words a first pass already cleared.
  //
  // Mutation proof: re-run `watermark_filtered_with`'s strip over `finalize`'s
  // flushed words (the tempting "defence in depth") and this reads back " A B",
  // one " A" short.
  let hypothesis = || {
    result_with_words(vec![
      word(" A", 0.0, 0.5),
      word(" A", 0.0, 0.5),
      word(" B", 1.0, 1.5),
    ])
  };
  let mut agreement = LocalAgreement::new();
  agreement.ingest(hypothesis());
  assert!(agreement.ingest(hypothesis()).is_advanced());
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A A B",
    "both stuttered A's are the hypothesis's own words, not a re-admission",
  );
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
fn a_distinct_repetition_of_a_confirmed_word_is_deleted_by_the_continuing_stream_today() {
  // OPEN (codex round 3, finding 2) -- the ledger's OTHER face, a DELETION
  // rather than a duplication, and the reason a stricter text-matching rule is
  // not the fix. A hypothesis that STUTTERS at the watermark's own instant
  // straddles the advance: the first " A" is confirmed and the second -- a
  // DISTINCT occurrence with identical text, start and end -- is held back.
  // Every later hypothesis correctly omits the confirmed one and re-offers the
  // held-back one; `watermark_filtered_with` reads that survivor as a
  // reproduction (it IS a front-of-list text match) and strips it on every
  // filter, and the next advance moves the watermark past it.
  //
  // This is the two-run construction from the issue, live: a run of ` A A B` and
  // a run of ` A B C` reach byte-identical (confirmed, offered, watermark) here
  // and need OPPOSITE answers, so no predicate over those three can be right.
  //
  //   ingest    [A@0.0, A@0.0, B@1.0] twice -> confirmed [A], holding [A, B]
  //   then      [A@0.0, B@1.0, C@2.0, D@3.0] twice
  //   TODAY     confirmed_words_slice() == [" A", " B"]
  //             <- the stream's second " A" is deleted; text " A B C D"
  //   CORRECT   [" A", " A", " B"], text " A A B C D"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 2.0, 2.5);
  let d = || word(" D", 3.0, 3.5);
  let stutter = || result_with_words(vec![a(), a(), b()]);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(stutter());
  assert!(agreement.ingest(stutter()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A"],
    "the advance confirms the first A and holds the distinct second one",
  );

  // The stream CONTINUES past that advance -- which is exactly what
  // `a_stutter_at_the_watermark_keeps_both_occurrences` does not do, and why
  // that green pin cannot see this.
  let onward = || result_with_words(vec![a(), b(), c(), d()]);
  agreement.ingest(onward());
  assert!(agreement.ingest(onward()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "CHARACTERIZATION of open defect #94, not a requirement -- the stream's own \
     second \" A\" is DELETED today, the ledger's other face. CORRECT is \
     [\" A\", \" A\", \" B\"]: the held-back A is the stream's own second \
     occurrence, not a re-offer of the confirmed first one. RED here means you \
     fixed #94 -- change this expectation to vec![\" A\", \" A\", \" B\"].",
  );

  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A B C D",
    "CHARACTERIZATION of open defect #94, not a requirement. CORRECT is \
     \" A A B C D\" -- both occurrences the stream produced. RED here means you \
     fixed #94 -- change this expectation to \" A A B C D\".",
  );
}

#[test]
fn a_reproduction_shifted_off_the_watermark_is_reconfirmed_today() {
  // OPEN (codex round 3, finding 1; re-reported verbatim as ROUND 4, FINDING 1
  // -- "F1 as reported" in the round-4 measurement, where `main` and the
  // apparatus branch were byte-identical at every ingest). The re-decode nudges
  // every word a hair PAST the 0.0 s watermark and inserts " X" ahead of the
  // reproduced " A", so nothing the hypothesis offers matches the confirmed tail
  // at offset 0 and the strip is 0. Two such hypotheses agree with each other
  // and confirm [X, A] on top of the settled A.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //   ingest    [X@0.01, A@0.02, B@1.0, C@2.0] twice
  //   TODAY     confirmed_words_slice() == [" A", " X", " A"], text " A X A B C"
  //   CORRECT   [" A"], text " A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A: the watermark stays 0.0
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A"],
    "confirmed [A], holding [B, C] at a 0.0 s watermark",
  );

  let shifted = || {
    result_with_words(vec![
      word(" X", 0.01, 0.3),
      word(" A", 0.02, 0.5),
      word(" B", 1.0, 1.5),
      word(" C", 2.0, 2.5),
    ])
  };
  agreement.ingest(shifted());
  agreement.ingest(shifted());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " X", " A"],
    "CHARACTERIZATION of open defect #94, not a requirement -- this is what the \
     shifted reproduction reaches the confirmed list as today. CORRECT is \
     [\" A\"]: it must not reach the confirmed list on the advance path either. \
     RED here means you fixed #94 -- change this expectation to vec![\" A\"].",
  );

  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert_eq!(
    text, " A X A B C",
    "CHARACTERIZATION of open defect #94, not a requirement. CORRECT is \
     \" A B C\". RED here means you fixed #94 -- change this expectation to \
     \" A B C\" and the count below to 1.",
  );
  assert_eq!(
    text.matches('A').count(),
    2,
    "CHARACTERIZATION of open defect #94: \" A\" reaches the transcript TWICE \
     today. CORRECT is 1. Text: {text:?}",
  );
}

#[test]
fn an_insertion_that_reproduces_nothing_confirmed_keeps_its_word() {
  // The second constraint on any replacement rule: an insertion that reproduces
  // NOTHING confirmed must keep its word. Refusing what sits ahead of a
  // reproduction is a forced choice -- the settled " A" is already in the
  // caller's hands, so an " X" put BEFORE a reproduction of it can go neither
  // before " A" (the list is append-only) nor after it (that would rewrite the
  // confirmed prefix). Take the reproduction away and the choice disappears, so
  // refusing " X" would be an unforced deletion. Here two hypotheses insert " X"
  // ahead of the held-back [B, C] and reproduce nothing confirmed.
  //
  // Mutation proof: replace the strip with the unconditional count-skip
  // (`let strip = readmit_candidates.len().min(filtered.len());`) and
  // `confirmed_words_slice()` reads back [" A", " B"] -- " X", agreed on by both
  // hypotheses and contradicting nothing, gone. Same mutation as
  // `omitting_a_confirmed_tied_word_does_not_drop_provisional_words`, from the
  // opposite direction: that one loses a word to an OMISSION shifting the
  // hypothesis left, this one to an INSERTION shifting it right.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  assert_eq!(confirmed_texts(&agreement), vec![" A"]);

  let inserted = || result_with_words(vec![word(" X", 0.0, 0.3), b(), c(), word(" D", 2.0, 2.5)]);
  agreement.ingest(inserted());
  assert!(agreement.ingest(inserted()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " X", " B"],
    "nothing confirmed is re-offered, so the insertion costs nothing",
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A X B C D",
  );
}
