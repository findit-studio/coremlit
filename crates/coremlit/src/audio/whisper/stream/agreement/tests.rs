use super::*;
use crate::audio::whisper::{
  result::{TranscriptionResult, TranscriptionSegment, TranscriptionTimings, WordTiming},
  task_facts::TaskFacts,
};

fn word(text: &str, start: f32, end: f32) -> WordTiming {
  WordTiming::new(text, vec![start as u32 + 1], start, end, 0.9)
}

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
  // RULE W's POSTCONDITION, swept rather than pinned to one fixture: whenever
  // the holdback is non-empty, the last CONFIRMED word starts strictly before
  // the watermark.
  //
  // That inequality is the whole of #94. `watermark_filtered` offers every
  // hypothesis word whose `start >= watermark`, so a confirmed word that TIES
  // the watermark passes that filter and can come back at the head of the next
  // hypothesis -- and there it is byte-identical to the stream's own second
  // occurrence of the same text, which is the issue's impossibility result.
  // Every defeated rule in this module's ledger tried to DECIDE that state.
  // Rule W refuses to create it: the split widens past a tie instead of cutting
  // at one, so the state is unreachable and needs no decision.
  //
  // The sweep draws starts from a coarse grid whose repeats make consecutive
  // words tie -- the `a=[0,0.5)/b=[0,1.0)` shape both #94 regressions are built
  // from, generalized -- keeps them non-decreasing the way `find_alignment`
  // guarantees (`segment::tests`: `w[i].end() <= w[i+1].start() + 1e-4`), and
  // drives growing prefixes through `ingest_streamed`, the MARKED path the
  // driver uses, so every advance carries the prefill premise the engine issues
  // for itself. One stride in four omits the leading word, which is the rewrite
  // `omitting_a_confirmed_tied_word_does_not_drop_provisional_words` is built
  // from.
  //
  // Mutation proof: drop Rule W's widening loop from `ingest` (leaving the bare
  // `budgeted_split`) and this reds on the first swept tie, reporting the
  // confirmed word whose start equals the watermark.
  const TEXTS: [&str; 4] = [" A", " B", " C", " D"];
  // Repeated 0.0 entries are the ties; the rest keep the grid coarse enough for
  // two words to share an instant often.
  const STEPS: [f32; 6] = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0];
  // The 0.0 duration is a zero-length word, which is the one shape that could
  // satisfy `start >= watermark` from inside the confirmed list without a tie
  // between two distinct starts.
  const DURATIONS: [f32; 4] = [0.0, 0.2, 0.5, 1.0];

  fn postcondition(agreement: &LocalAgreement, trial: u32, stride: usize) -> bool {
    if agreement.last_agreed_words_slice().is_empty() {
      return false;
    }
    let Some(last) = agreement.confirmed_words_slice().last() else {
      return false;
    };
    assert!(
      last.start() < agreement.last_agreed_seconds(),
      "trial {trial}, stride {stride}: the confirmed list {:?} ends on {:?} at \
       {}, which is not strictly before the {} s watermark -- that word passes \
       `watermark_filtered`'s own `start >= watermark` and can be re-admitted",
      confirmed_texts(agreement),
      last.word(),
      last.start(),
      agreement.last_agreed_seconds(),
    );
    true
  }

  let mut state: u64 = 0x2545_F491_4F6C_DD1D;
  let mut next = move || {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
  };
  let mut checked = 0u32;
  let mut tied_truths = 0u32;

  for trial in 0..256u32 {
    let length = 4 + (next() % 5) as usize;
    let mut truth: Vec<WordTiming> = Vec::with_capacity(length);
    let mut start = 0.0f32;
    for _ in 0..length {
      let text = TEXTS[(next() % TEXTS.len() as u64) as usize];
      start += STEPS[(next() % STEPS.len() as u64) as usize];
      let end = start + DURATIONS[(next() % DURATIONS.len() as u64) as usize];
      truth.push(word(text, start, end));
    }
    if truth
      .windows(2)
      .any(|pair| pair[0].start() >= pair[1].start())
    {
      tied_truths += 1;
    }

    let mut agreement = LocalAgreement::new();
    for stride in 2..=truth.len() {
      let omit_head = stride > 2 && next() % 4 == 0;
      let offered = if omit_head {
        truth[1..stride].to_vec()
      } else {
        truth[..stride].to_vec()
      };
      for _ in 0..2 {
        agreement.ingest_streamed(result_with_words(offered.clone()));
        if postcondition(&agreement, trial, stride) {
          checked += 1;
        }
      }
    }
  }

  // Non-vacuity, both halves: the sweep really did build tied truths, and the
  // postcondition really was READ against a non-empty holdback and a non-empty
  // confirmed list rather than skipped.
  assert!(
    tied_truths > 64,
    "the sweep must actually produce tied starts: {tied_truths} of 256 trials",
  );
  assert!(
    checked > 256,
    "the postcondition must actually be reachable: {checked} observations",
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
fn an_empty_holdback_leaves_a_zero_duration_word_at_the_watermark() {
  // RULE W's ONE RESIDUAL, characterized rather than closed. The postcondition
  // -- `confirmed.last().start() < last_agreed_seconds` strictly -- is a claim
  // about a NON-EMPTY holdback, because the watermark is then the first held
  // word's start and the rule widens past any tie with it. When
  // `budgeted_split` empties the holdback the watermark falls back to
  // `common.last().end()` instead, and Rule W has no `common[split]` to anchor
  // against: if that last word has `start == end` the watermark EQUALS its
  // start, the word passes `watermark_filtered`'s own `start >= watermark`, and
  // a re-decode that offers it back -- with no holdback, so no prefill to
  // displace it -- confirms it a SECOND time.
  //
  // Reaching it needs three things at once, and the driver supplies none of
  // them: a non-default `agreement_count_needed` (here 1), a word whose OWN
  // tokens exceed `MAX_HOLDBACK_PREFILL_TOKENS` so the split runs to
  // `common.len()`, and a ZERO-DURATION word in that position.
  // `LocalAgreementTranscriber` never leaves `DEFAULT_AGREEMENT_COUNT_NEEDED`,
  // and `add_word_timestamps` never emits a 112-token word.
  //
  // Recorded, not repaired: the alternatives are anchoring the empty-holdback
  // watermark at `end + epsilon` (a threshold with the same cliff one instant
  // further on, and one that makes the watermark a value no word's `start` ever
  // equals) or refusing the advance (the blocking policy this issue's ledger
  // already refuted for deadlock).
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
      agreement.last_agreed_seconds()
    ),
    (vec![" A", " Z"], 0, 1.0),
    "non-vacuous: the budget emptied the holdback, so the watermark is \
     \" Z\"'s END -- which for a zero-duration word is also its START",
  );
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .last()
      .map(WordTiming::start),
    Some(agreement.last_agreed_seconds()),
    "the postcondition is VACUOUS here: with no held-back word the last \
     confirmed word's start EQUALS the watermark instead of preceding it",
  );

  // The re-decode offers " Z" back at the head of its word list. Nothing
  // displaces it: the holdback is empty, so `decoding_options_for_next` attaches
  // an empty prefix and the decoder was never fed a reproduction to begin with.
  //
  // One ingest is enough: the previous hypothesis already contributed " Z" at
  // this watermark, so the pair agrees on it immediately.
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![zero(), b()]))
      .is_advanced()
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " Z", " Z"],
    "CHARACTERIZATION (https://github.com/findit-studio/coremlit/issues/94, \
     residual 1): the CORRECT answer is [\" A\", \" Z\"] -- \" Z\" is one word \
     the stream produced once and it reaches the confirmed list twice. Rule W \
     cannot help: it widens the split past a tie with `common[split]`, and here \
     the split ran to `common.len()` so there is no `common[split]` to widen \
     past. If this state is ever closed, assert [\" A\", \" Z\"] and delete this \
     message.",
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
