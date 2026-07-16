use super::*;
use crate::{
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
  let next = agreement.decoding_options_for_next(&crate::options::DecodingOptions::new());
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
  let final_result = agreement.finalize(&crate::options::DecodingOptions::new());
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
  let final_result = agreement.finalize(&crate::options::DecodingOptions::new());
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
    .finalize(&crate::options::DecodingOptions::new())
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
  // hypothesis left — the match-based strip only removes words that
  // actually reproduce the confirmed tail.
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
    .finalize(&crate::options::DecodingOptions::new())
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

  let options = crate::options::DecodingOptions::new();
  let compute = crate::options::ComputeOptions::new();
  let finalized = agreement.finalize(&options);
  assert_eq!(
    finalized.task_facts().drew_from_rng(),
    Some(true),
    "the dropped control hypothesis's unseeded draw survives into finalize",
  );
  assert!(
    !crate::provenance::Provenance::for_result(&options, &compute, &finalized).is_reproducible(),
    "an unseeded draw happened (in a dropped hypothesis), so it is not reproducible",
  );
  // A seed makes that same recovered draw replayable -- the promise becomes real.
  assert!(
    crate::provenance::Provenance::for_result(
      &options.clone().with_seed(11),
      &compute,
      &finalized,
    )
    .is_reproducible(),
    "a seed makes the recovered draw replayable",
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

  let options = crate::options::DecodingOptions::new();
  let compute = crate::options::ComputeOptions::new();
  let finalized = agreement.finalize(&options);
  assert_eq!(
    finalized.task_facts().observed_language(),
    Some("es"),
    "the earliest ingested genuine language wins, even from a dropped hypothesis",
  );
  assert_eq!(
    crate::provenance::Provenance::for_result(&options, &compute, &finalized)
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
  let finalized = agreement.finalize(&crate::options::DecodingOptions::new());
  assert_eq!(
    finalized.task_facts().worker_schedule(),
    None,
    "agreement-confirmed text has no single knowable worker attribution -- unknown, not [0, 2]",
  );
}
