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
/// vocabulary's `special_token_begin`, so `prefill_tokens`' other filter cannot
/// confuse the measurement.
fn word_of_tokens(text: &str, start: f32, end: f32, token_count: usize) -> WordTiming {
  let first = start as u32 * 1000 + 1;
  let tokens: Vec<u32> = (0..token_count as u32).map(|index| first + index).collect();
  WordTiming::new(text, tokens, start, end, 0.9)
}

/// A word whose tokens `prefill_tokens` ERASES: every id is at or above
/// [`MIN_SPECIAL_TOKEN_BEGIN`], so the filter drops the lot and the word
/// contributes NOTHING to the initial prompt. `add_word_timestamps` never emits
/// one — it strips exactly those ids and skips an alignment entry that has
/// nothing left — so this is a hand-built shape, which is precisely the call
/// shape [`LocalAgreement::ingest`]'s `decoded_under` parameter exists for.
fn special_only_word(text: &str, start: f32, end: f32) -> WordTiming {
  WordTiming::new(text, vec![MIN_SPECIAL_TOKEN_BEGIN], start, end, 0.9)
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
/// Ingest the way [`LocalAgreementTranscriber::push_samples`] does: under the
/// options the engine itself issued for its current state, so the prefill
/// premise `LocalAgreement::watermark_filtered`'s ambiguity rule is decided by
/// is ESTABLISHED rather than assumed.
///
/// Every test below that is modelling the streaming loop goes through this. The
/// bare [`LocalAgreement::ingest`] is the UNMARKED path — a result decoded some
/// other way — and the tests that mean that call it directly, with the options
/// they mean.
trait IngestStreamed {
  fn ingest_streamed(&mut self, result: TranscriptionResult) -> AgreementOutcome;
}

impl IngestStreamed for LocalAgreement {
  fn ingest_streamed(&mut self, result: TranscriptionResult) -> AgreementOutcome {
    let options = self.decoding_options_for_next(&DecodingOptions::new());
    self.ingest(result, &options)
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
  // hypothesis left. The frontier rule refuses only what the alignment
  // ATTRIBUTES to the settled half, and a deletion costs it nothing beyond the
  // match it forgoes, so a candidate the hypothesis does not reproduce costs
  // nothing here either.
  //
  // Mutation proof: refuse nothing at all (a frontier of 0, Swift's bare
  // timestamp filter at `:372`) and B is never confirmed -- the re-admitted A
  // sits at the front of BOTH filtered lists, the prefix comparison lines up on
  // it instead of on B, and `confirmed_words_slice()` reads back [" A"]. The
  // same mutation costs `tied_word_starts_never_confirm_twice` its stability,
  // reading back [" A", " A", " B"].
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
    "B lost to the re-admission frontier: {confirmed:?}"
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
  // That inequality is the whole of #94. `watermark_filtered_with` offers every
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
// last one, DELETED a word the stream genuinely produced. Each was a
// CHARACTERIZATION while <https://github.com/findit-studio/coremlit/issues/94>
// was open: it asserted the wrong answer the strip produced and carried the
// right one in its failure message. The frontier rule
// (`LocalAgreement::watermark_filtered`) replaced that strip, so each now
// asserts the property its name states, at exactly the value its old failure
// message specified.
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
// What separates those two runs is the HOLDBACK -- which is why the rule that
// closed them aligns `settled ++ holdback` as a whole instead of scanning the
// offered list for a text match. The four defeated approaches and the
// falsifier that killed each are in the issue.

#[test]
fn an_insertion_before_a_reproduced_confirmed_word_is_refused_with_it() {
  // #94 (codex round 2; also the round-4 F1 "all tied at the watermark"
  // variant). The old leading-run strip zipped the confirmed tail against the
  // offered list from BOTH fronts, so it only removed a reproduction the
  // hypothesis put at the very FRONT of its post-watermark list. An insertion at
  // that same instant AHEAD of the reproduction mismatched at offset 0, stripped
  // nothing, and the already-confirmed " A" rode through into `hypothesis_words`
  // -- which `finalize` then appended wholesale, reading back " A X A B C".
  //
  // The frontier rule aligns the span as a whole: `settled ++ holdback` is
  // [A, B, C] against the offered [X, A, B, C], and the only optimal alignment
  // matches A, B and C at offsets 1, 2 and 3 -- so the seam sits at offset 1 and
  // " X" is refused WITH the reproduction it hides behind. That is forced rather
  // than incidental: the settled " A" is already in the caller's hands and
  // `confirmed_words_slice()` is append-only, so " X" can go neither before it
  // nor after it. Take the reproduction away and the choice disappears, which is
  // what `an_insertion_that_reproduces_nothing_confirmed_keeps_its_word` pins
  // from the other side.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //   ingest    [X@0.0, A@0.0, B@0.0, C@1.0]  ->  " A B C", " A" once
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back " A X A B C" -- the answer the ledger recorded while #94 was open.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
  let c = || word(" C", 1.0, 2.0);
  let x = || word(" X", 0.0, 0.3); // an insertion at that same instant
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  // Rule W (#94): the split may not cut at a tied start, so it widens past the
  // tie -- one word moves from `last_agreed_words_slice()` into
  // `confirmed_words_slice()` and the watermark moves to the first word past
  // the tie. `confirmed ++ holdback` is unchanged, and the finalized text this
  // test asserts is measured byte-identical either way.
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B"],
    "confirmed [A, B], holding [C] at a 1.0 s watermark",
  );

  agreement.ingest_streamed(result_with_words(vec![x(), a(), b(), c()]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B C",
    "the settled A wins the instant it was confirmed at, and the insertion is \
     not smuggled in beside it",
  );
  assert_eq!(
    final_result.text().matches('A').count(),
    1,
    "\" A\" reaches the transcript exactly once. Text: {:?}",
    final_result.text(),
  );
}

#[test]
fn an_insertion_before_a_reproduced_confirmed_word_is_refused_on_the_advance() {
  // #94, the same defect WITHOUT `finalize`. `ingest`'s advance folds
  // `common[..split]` -- a slice of `hypothesis_words` -- straight into
  // `confirmed_words`, so two consecutive hypotheses that both insert ahead of
  // the reproduced confirmed word used to agree on the whole insertion and put
  // the duplicate into `confirmed_words_slice()`, the face a streaming caller
  // reads between pushes. Fixing `finalize` alone would never have reached this
  // one, which is why the ledger pins both paths.
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back [" A", " X", " A"] -- the answer the ledger recorded while #94 was open.
  //
  //   ingest [A,B,C] [A,B,C] [X,A,B,C] [X,A,B,C]  (all tied at 0.0 but C)
  //   confirmed_words_slice() == [" A"]
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0);
  let c = || word(" C", 1.0, 2.0);
  let x = || word(" X", 0.0, 0.3);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![x(), a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![x(), a(), b(), c()]));

  // Rule W (#94): the split may not cut at a tied start, so it widens past the
  // tie -- one word moves from `last_agreed_words_slice()` into
  // `confirmed_words_slice()` and the watermark moves to the first word past
  // the tie. `confirmed ++ holdback` is unchanged, and the finalized text this
  // test asserts is measured byte-identical either way.
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B"],
    "the advance path must not re-confirm A either",
  );
}

#[test]
fn a_holdback_over_a_reproduced_confirmed_word_keeps_the_settled_span() {
  // #94 at `finalize`'s OTHER append: `confirmed_words.append(&mut
  // last_agreed_words)`. `last_agreed_words` is `common[split..]`, itself a
  // slice of `hypothesis_words`, so an already-confirmed word the old strip
  // missed reached `confirmed_words` through the HOLDBACK rather than through
  // the `holdback_superseded` branch: with `split == 0` nothing is confirmed at
  // the advance and the whole agreed prefix -- insertion and re-admitted " A"
  // alike -- was held back and then flushed, reading back " A X A" and losing
  // " B C" with it.
  //
  // Under the frontier rule these two hypotheses offer NOTHING past the seam:
  // aligning [A, B, C] against [X, A] matches only the A at offset 1, so the
  // seam is at offset 2 and `hypothesis_words` is empty. They therefore disagree
  // with the holdback, and an empty `hypothesis_words` is exactly the case
  // `a_final_hypothesis_with_nothing_past_the_watermark_keeps_the_holdback`
  // guards: nothing supersedes the holdback, so it is still flushed.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back " A X A" -- the answer the ledger recorded while #94 was open.
  //   ingest    [X@0.0, A@0.0] twice  ->  " A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0);
  let c = || word(" C", 1.0, 2.0);
  let x = || word(" X", 0.0, 0.3);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![x(), a()]));
  agreement.ingest_streamed(result_with_words(vec![x(), a()]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B C",
    "the holdback must not flush a second copy of the confirmed A, and it must \
     not lose \" B C\" doing it",
  );
}

#[test]
fn a_different_suffix_over_a_reproduced_confirmed_word_keeps_the_settled_span() {
  // #94 at `finalize`'s THIRD append: `find_longest_different_suffix(&prev_words,
  // &hypothesis_words)`, also a slice of `hypothesis_words`. Here the
  // re-admitted " A" sat PAST the common prefix -- two hypotheses agree on an
  // inserted [P, Q] and then diverge, and the newer one's reproduction of the
  // confirmed " A" rode in on the differing suffix, reading back " A P Q A".
  //
  // The frontier rule refuses [P, Q, A] wholesale: aligning [A, B, C] against
  // [P, Q, A] matches only the A at offset 2, so the seam is at offset 3.
  // Refusing " P" and " Q" is the same forced choice as
  // `an_insertion_before_a_reproduced_confirmed_word_is_refused_with_it` -- they
  // sit ahead of a reproduction of a word the caller already holds.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back " A P Q A" -- the answer the ledger recorded while #94 was open.
  //   ingest    [P@0.0, Q@0.0, Z@0.0] then [P@0.0, Q@0.0, A@0.0]  ->  " A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0);
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  agreement.ingest_streamed(result_with_words(vec![
    word(" P", 0.0, 0.1),
    word(" Q", 0.0, 0.2),
    word(" Z", 0.0, 0.3),
  ]));
  agreement.ingest_streamed(result_with_words(vec![
    word(" P", 0.0, 0.1),
    word(" Q", 0.0, 0.2),
    a(),
  ]));

  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B C",
    "the differing suffix must not flush a second copy of the confirmed A",
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
  // Mutation proof: drop the HOLDBACK from the alignment (align `settled`
  // alone) and this reads back " A D E" -- " B" and " C", agreed by both
  // hypotheses, gone. Without the holdback to score against, matching the
  // settled " A" at offset 2 is the alignment's whole objective, so the seam
  // lands behind it. That is the identical answer the naive rposition scan gave
  // (`filtered.iter().rposition(|w| candidates.contains(&normalized(w.word())))
  // .map_or(0, |i| i + 1)`), which is the point: what keeps the frontier off
  // the later occurrence is the holdback, not the search.
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
// The ledger, continued: two shapes with TWO confirmed words tied at the
// watermark
// ---------------------------------------------------------------------

/// The shared history for the two ledger entries below: two identical hypotheses
/// whose first THREE words tie at 0.0 s, so the advance confirms `[A, B]`,
/// holds `[C, D]`, and leaves the watermark at 0.0 s with TWO confirmed words
/// still at or past it.
fn tied_pair_confirmed() -> LocalAgreement {
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![
    tied_a(),
    tied_b(),
    tied_c(),
    tied_d(),
  ]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![
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
    vec![" A", " B", " C"],
    "Rule W widens past the 0.0 s tie, so this is confirmed [A, B, C] holding \
     [D] at a 1.0 s watermark -- `confirmed ++ holdback` unchanged",
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
fn a_confirmed_tail_reproduced_without_its_head_is_still_refused() {
  // #94 (codex round 2, counterexample 1) -- the CONFIRMED-TAIL OMISSION. The
  // old strip compared the confirmed tail from ITS OWN front, so a hypothesis
  // that OMITS " A" and reproduces only " B" mismatched at offset 0, stripped
  // nothing, and " B" rode through into `hypothesis_words` for " A B B C D".
  //
  // A deletion costs the frontier alignment nothing beyond the match it forgoes:
  // [A, B, C, D] against [B, C, D] simply drops the A, matches B/C/D at offsets
  // 0/1/2, and puts the seam after offset 0. The reproduction is refused even
  // though it does not start where the confirmed tail does.
  //
  //   confirmed [A@0.0, B@0.0], holding [C@0.0, D@1.0], watermark 0.0
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back " A B B C D" -- the answer the ledger recorded while #94 was open.
  //   ingest    [B@0.0, C@0.0, D@1.0]  ->  " A B C D"
  let mut agreement = tied_pair_confirmed();
  agreement.ingest_streamed(result_with_words(vec![tied_b(), tied_c(), tied_d()]));
  let final_result = agreement.finalize(&crate::audio::whisper::options::DecodingOptions::new());
  assert_eq!(
    final_result.text(),
    " A B C D",
    "the confirmed B must not be confirmed a second time by a reproduction \
     that skipped A",
  );
}

#[test]
fn a_confirmed_tail_reproduced_without_its_head_is_still_refused_on_the_advance() {
  // #94, the same omission WITHOUT `finalize`. Repeating the hypothesis makes
  // the two agree, and `ingest`'s advance folds `common[..split]` straight into
  // the caller-visible `confirmed_words_slice()` a streaming caller reads
  // between pushes -- so this path needs its own pin.
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back [" A", " B", " B"] -- the answer the ledger recorded while #94 was open.
  //
  //   ingest [B,C,D] twice on top of the same history
  //   confirmed_words_slice() == [" A", " B"]
  let mut agreement = tied_pair_confirmed();
  agreement.ingest_streamed(result_with_words(vec![tied_b(), tied_c(), tied_d()]));
  agreement.ingest_streamed(result_with_words(vec![tied_b(), tied_c(), tied_d()]));
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B", " C"],
    "the advance path must not re-confirm B either (\" C\" is Rule W's widening, \
     not a re-admission)",
  );
}

#[test]
fn a_reproduction_behind_a_shorter_false_match_is_still_refused() {
  // #94 (codex round 2, counterexample 2) -- the PARTIAL FRONT MATCH. The old
  // strip zipped the confirmed tail [A, B] against the offered list, matched
  // " A" at offset 0, mismatched " B" against the inserted " X", and stopped --
  // stripping one word and letting the real, longer " A B" reproduction at
  // offset 2 ride through, for " A B X A B C D".
  //
  // The frontier alignment has no front to be trapped at. [A, B, C, D] against
  // [A, X, A, B, C, D] scores four matches only by taking B at offset 3 (there
  // is no other B), so every optimal alignment puts the seam after it: the
  // frontier is 4 and the false match, the insertion and the real reproduction
  // are all refused together.
  //
  //   confirmed [A@0.0, B@0.0], holding [C@0.0, D@1.0], watermark 0.0
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back " A B X A B C D" -- the answer the ledger recorded while #94 was open.
  //   ingest    [A@0.0, X@0.0, A@0.0, B@0.0, C@0.0, D@1.0]  ->  " A B C D"
  let mut agreement = tied_pair_confirmed();
  agreement.ingest_streamed(result_with_words(vec![
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
    " A B C D",
    "the longer reproduction behind the false match is stripped too",
  );
}

#[test]
fn a_reproduction_behind_a_shorter_false_match_is_still_refused_on_the_advance() {
  // #94, the same partial front match WITHOUT `finalize` -- two hypotheses that
  // agree on the whole `[A, X, A, B]` run used to advance, carrying both
  // already-confirmed words into `confirmed_words_slice()`.
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back [" A", " B", " X", " A", " B"] -- the answer the ledger recorded while
  // #94 was open.
  //
  //   confirmed_words_slice() == [" A", " B"]
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
  agreement.ingest_streamed(hypothesis());
  agreement.ingest_streamed(hypothesis());
  assert_eq!(
    agreement
      .confirmed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B", " C"],
    "the advance path must not re-confirm A or B either (\" C\" is Rule W's \
     widening, not a re-admission)",
  );
}

// ---------------------------------------------------------------------
// What the re-admission rule must NOT break
// ---------------------------------------------------------------------
//
// The pins below were green on `main` and on every defeated candidate rule, and
// they are green on the frontier rule: they are the constraints a replacement
// has to keep satisfying, recorded next to the ledger above so any future change
// is measured against both halves at once. Deleting a word the stream genuinely
// produced is the failure mode a stricter rule falls into, and it is worse than
// the duplication the ledger records -- `tests/whisper/streaming.rs`'s portable
// prefix property tolerates a truncation and forbids a rewrite. That asymmetry
// is why the frontier rule breaks an ambiguous seam toward KEEPING: it refuses
// only the offered prefix every optimal alignment settles.

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
  // Mutation proof: run `settled_frontier` a second time over `finalize`'s
  // flushed holdback (the tempting "defence in depth") and this reads back
  // " A B", one " A" short -- the frontier must be taken once, on the offered
  // list, and never again on words a first pass already cleared.
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

/// `pending_words_slice()` as text — the agreed-but-not-yet-irrevocable head of
/// the holdback. Always read in the SAME assertion as `confirmed_texts`, so the
/// boundary between the two is pinned rather than implied: a mutation that moves
/// a word across it fails the pair even though the concatenation is unchanged.
fn pending_texts(agreement: &LocalAgreement) -> Vec<&str> {
  agreement
    .pending_words_slice()
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
  // OPPOSITE answers, so no predicate over those three can be right. What
  // separates them is the HOLDBACK -- `[A, B]` here against `[B, C]` there --
  // and the frontier rule reads it, because it aligns `settled ++ holdback` as a
  // whole. [A, A, B] against [A, B, C, D] scores two matches with the seam at
  // either offset 0 or offset 1, so the seam is AMBIGUOUS; taking the smallest
  // keeps the leading " A", which is the stream's own second occurrence.
  //
  // Round 5, finding 1 re-reported THIS sequence -- the same two ingests, the
  // same two optimal seams, the same [" A", " A", " B"] -- and argued the other
  // reading: that the offered " A" is the SETTLED one coming back and the
  // provisional " A" was deleted, so seam 1 is right and the minimum rule
  // duplicates a word. That reading is refuted by how the offered list is
  // produced, not by this bias. `decoding_options_for_next` prefills the decoder
  // with the holdback's own tokens and `decode_text` forces every prompt
  // position, so a hypothesis decoded from those options BEGINS with the
  // provisional " A"; deleting it would mean dropping a token the decoder was
  // fed rather than asked for. The minimum seam is that reading.
  // `the_next_strides_prefill_is_the_holdback_verbatim` pins the contract, and
  // `LocalAgreement::watermark_filtered`'s "Ambiguity rule" carries why blocking
  // on the tie instead would deadlock this exact shape on every stride.
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back [" A", " B"] -- the answer the ledger recorded while #94 was open.
  // Two more, for the two halves of the rule this one turns on. Take the
  // LARGEST optimal seam instead of the smallest and it reads back [" A", " B"]
  // again -- the ambiguity is real, and the direction the tie breaks IS the
  // answer. Drop the holdback from the alignment (align `settled` alone) and it
  // reads back [" A", " B"] once more, because the holdback is the only thing
  // that separates this run from the one in the issue's table. That same
  // mutation takes the sibling pin
  // `a_later_re_use_of_a_confirmed_word_is_not_mistaken_for_a_re_admission`
  // down to " A D E" -- the naive-scan failure, from the other direction.
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
fn a_reproduction_shifted_off_the_watermark_is_still_refused() {
  // #94 (codex round 3, finding 1; re-reported verbatim as ROUND 4, FINDING 1 --
  // "F1 as reported" in the round-4 three-tree measurement, where `main` and the
  // apparatus branch were byte-identical at every ingest). The re-decode nudges
  // every word a hair PAST the 0.0 s watermark and inserts " X" ahead of the
  // reproduced " A", so nothing the hypothesis offered matched the confirmed
  // tail at offset 0, the old strip was 0, and two such hypotheses agreed with
  // each other and confirmed [X, A] on top of the settled A.
  //
  // This is also why the frontier rule reads TEXT and not timestamps. The
  // defeated approach 3 drew a match window from the watermark, and this exact
  // sequence escaped it by sitting a hundredth of a second outside; a re-decode
  // is free to move every timestamp it emits, so timings are evidence about the
  // span, not about occurrence identity within it. The alignment of [A, B, C]
  // against [X, A, B, C] does not care where the words landed: its only optimal
  // reading matches A/B/C at offsets 1/2/3, seam at 1.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //
  // Mutation proof: put the old leading-run strip back in place of the frontier
  // (`settled.iter().zip(&offered).take_while(text-equal).count()`) and this reads
  // back [" A", " X", " A"] -- the answer the ledger recorded while #94 was open.
  //
  //   ingest    [X@0.01, A@0.02, B@1.0, C@2.0] twice
  //   confirmed_words_slice() == [" A"], text " A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A: the watermark stays 0.0
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  // Rule W (#94): the split may not cut at a tied start, so it widens past the
  // tie -- one word moves from `last_agreed_words_slice()` into
  // `confirmed_words_slice()` and the watermark moves to the first word past
  // the tie. `confirmed ++ holdback` is unchanged, and the finalized text this
  // test asserts is measured byte-identical either way.
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "confirmed [A, B], holding [C] at a 1.0 s watermark",
  );

  let shifted = || {
    result_with_words(vec![
      word(" X", 0.01, 0.3),
      word(" A", 0.02, 0.5),
      word(" B", 1.0, 1.5),
      word(" C", 2.0, 2.5),
    ])
  };
  agreement.ingest_streamed(shifted());
  agreement.ingest_streamed(shifted());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "the shifted reproduction must not reach the confirmed list on the advance \
     path either (\" B\" is Rule W's widening, not the reproduction)",
  );

  let text = agreement
    .finalize(&crate::audio::whisper::options::DecodingOptions::new())
    .text()
    .to_string();
  assert_eq!(text, " A B C");
  assert_eq!(
    text.matches('A').count(),
    1,
    "\" A\" reaches the transcript exactly once. Text: {text:?}",
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
  // This is also the direct falsifier for the frontier's TIE-BREAK. The seam
  // here is ambiguous -- " X" matches nothing settled, so an optimal alignment
  // may put it on either side -- and the smallest admissible seam keeps it.
  //
  // Mutation proof: take the LARGEST optimal seam instead of the smallest
  // (`.rev()` on `settled_frontier`'s final `find`) and `confirmed_words_slice()`
  // reads back [" A", " B"] -- " X", agreed on by both hypotheses and
  // contradicting nothing, gone. That is the deletion the bias exists to
  // avoid, and it is the same loss
  // `omitting_a_confirmed_tied_word_does_not_drop_provisional_words` records
  // from the opposite direction: that one loses a word to an OMISSION shifting
  // the hypothesis left, this one to an INSERTION shifting it right.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
  let c = || word(" C", 1.0, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  assert_eq!(confirmed_texts(&agreement), vec![" A"]);

  let inserted = || result_with_words(vec![word(" X", 0.0, 0.3), b(), c(), word(" D", 2.0, 2.5)]);
  agreement.ingest_streamed(inserted());
  assert!(agreement.ingest_streamed(inserted()).is_advanced());
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

#[test]
fn a_confirmed_word_from_before_the_watermark_is_out_of_the_alignment() {
  // The third constraint, and the SCOPE of `settled`: only the trailing run of
  // the confirmed list whose extent REACHES the watermark is aligned against the
  // offered words. Confirmed words that end before it have another word's audio,
  // or silence, between them and the re-decoded span, and putting them in the
  // alignment would let them match a genuinely later re-use of the same text --
  // taking the provisional words behind it down too, exactly the scan-based
  // failure `a_later_re_use_of_a_confirmed_word_is_not_mistaken_for_a_re_admission`
  // pins. That sibling cannot see this, because there the watermark is still
  // 0.0 s and the trailing run IS the whole confirmed list; here the watermark
  // has moved past " A" and the clip says " A" again at the new frontier.
  //
  // The GAPS below are load-bearing, and were not when this test was written:
  // " B" ends at 0.9 with the watermark at 1.0, so it is out of scope by 0.1 s.
  // Its sibling `a_re_utterance_separated_from_the_watermark_keeps_its_word`
  // re-utters " B" rather than " A" and so pins that boundary directly; this
  // one pins the word two places back, which stays out however the run's near
  // end is drawn.
  //
  // Mutation proof: widen `settled` to the whole confirmed list
  // (`let settled_len = confirmed.len();`) and the alignment matches the
  // re-used " A" at offset 0 against the confirmed " A" at 0.0 s, putting the
  // seam behind it -- `confirmed_words_slice()` reads back [" A", " B", " C"],
  // the stream's second " A" deleted. Every other test in this module stays
  // green under that mutation.
  let a0 = || word(" A", 0.0, 0.4);
  let b = || word(" B", 0.5, 0.9);
  let c0 = || word(" C", 1.0, 1.4);
  let d0 = || word(" D", 1.5, 1.9);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a0(), b(), c0(), d0()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a0(), b(), c0(), d0()]))
      .is_advanced()
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "confirmed [A@0.0, B@0.5], holding [C@1.0, D@1.5] at a 1.0 s watermark",
  );
  assert!((agreement.last_agreed_seconds() - 1.0).abs() < 1e-6);

  // The clip says " A" again, at the watermark itself -- a different word at a
  // different instant, with the confirmed one two words behind the watermark.
  let re_used = || {
    result_with_words(vec![
      word(" A", 1.0, 1.05),
      word(" C", 1.1, 1.4),
      word(" D", 1.5, 1.9),
      word(" E", 2.0, 2.4),
    ])
  };
  agreement.ingest_streamed(re_used());
  assert!(agreement.ingest_streamed(re_used()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B", " A", " C"],
    "the later \" A\" is the stream's own, and costs nothing that follows it",
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A B A C D E",
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
  // The frontier aligns `settled ++ holdback` as a WHOLE, and a globally optimal
  // alignment is free to explain the settled word by an EARLIER occurrence and
  // the whole holdback by a LATER one. The seam then cuts past both, and every
  // word BETWEEN the two occurrences is refused -- never confirmed, never held,
  // and never re-offerable, because the advance that follows moves the watermark
  // past them.
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
// What the frontier rule deliberately LEAVES
// ---------------------------------------------------------------------
//
// A documented residual with a test, rather than a claim of totality. See
// `LocalAgreement::watermark_filtered`'s "What this deliberately leaves".

#[test]
fn an_unaccounted_repeat_of_a_settled_word_is_kept_as_the_streams_own() {
  // RESIDUAL of #94, held open on purpose. The frontier rule aligns
  // `settled ++ holdback` -- the engine's whole record of the span at and past
  // the watermark -- against the offered words. When the offered list carries
  // MORE copies of a settled word than that record accounts for, the surplus
  // copy matches nothing on the settled side, no alignment can attribute it
  // there, and it is kept: the transcript ends up with the settled " A" twice.
  //
  // That is not a missed case, it is the tie broken deliberately. The engine
  // holds `settled = [A]`, `holdback = [B, C]`, offered `[A, A, B, C]`; the clip
  // stuttering (" A A B C" is right) and the decode duplicating (" A B C" is
  // right) produce byte-identical inputs, down to every `WordTiming` field. The
  // holdback separates the issue's two-run construction, but it cannot separate
  // THIS one -- the holdbacks are equal too. With nothing left to decide on, the
  // rule keeps: `tests/whisper/streaming.rs`'s portable prefix property
  // tolerates a truncation and forbids a rewrite, so a duplicate is the
  // recoverable error and a deletion is not.
  //
  // Its mirror image is
  // `a_distinct_repetition_of_a_confirmed_word_survives_the_continuing_stream`:
  // there the offered list carries FEWER copies than the record, the same tie is
  // broken the same way, and keeping is the RIGHT answer. One rule, one bias,
  // two truths -- which is exactly why the boundary is undecidable here and the
  // bias is the whole of the answer.
  //
  // Refusing to advance instead would not help: this shape reproduces on every
  // subsequent stride, `settled` and `holdback` only move on an advance, so the
  // stream would stall forever rather than for a stride.
  //
  //   confirmed [A@0.0], holding [B@0.0, C@1.0], watermark 0.0
  //
  // Mutation proof: take the LARGEST optimal seam instead of the smallest and
  // this reads back [" A"] -- the surplus copy refused, which is the deletion
  // this bias exists to avoid. Drop the refusal entirely (a frontier of 0,
  // Swift's bare timestamp filter at `:372`) and it reads back
  // [" A", " A", " A"] -- three copies, so the rule is doing real work here
  // even though it cannot do all of it.
  //
  //   ingest    [A@0.0, A@0.0, B@0.0, C@1.0] twice
  //   confirmed_words_slice() == [" A", " A"], text " A A B C"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 0.0, 1.0); // tied start with A
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
    vec![" A"],
    "confirmed [A], holding [B, C] at a 0.0 s watermark",
  );

  let repeated = || result_with_words(vec![a(), a(), b(), c()]);
  agreement.ingest_streamed(repeated());
  assert!(agreement.ingest_streamed(repeated()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " A"],
    "the surplus A is kept and confirmed -- the documented residual, not a \
     regression of the ledger above",
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A A B C",
  );
}

// ---------------------------------------------------------------------
// The scope of `settled`: which EDGE of a confirmed word faces the span
// ---------------------------------------------------------------------

#[test]
fn a_word_whose_extent_reaches_the_watermark_is_in_the_alignment() {
  // #94 round 5, finding 2 -- CROSS-WATERMARK DRIFT, which bypassed the
  // alignment entirely rather than losing an argument inside it. `settled` was
  // the trailing run of confirmed words with `start >= watermark`; here the
  // confirmed " A" ENDS at the watermark and starts a hundredth of a second
  // before it, so the run was EMPTY, `watermark_filtered_with` took its
  // `settled_len == 0` fast path, and the whole offered list came back
  // unfiltered. Two such hypotheses then agreed and confirmed " A" a second
  // time. Nothing in the frontier rule ever ran.
  //
  // The endpoint was the bug. Each list is tested against the watermark on the
  // edge that FACES the span -- `offered` on `start`, `settled` on `end` -- and
  // testing `start` on both asked whether a confirmed word BEGINS inside the
  // span, which a word straddling the boundary never does. The last confirmed
  // word straddles it on essentially every advance: DTW word timings abut, so
  // `confirmed.last().end() == holdback[0].start() == watermark` (measured in
  // `jfk_tiny_words_golden.json`: 20 of its 21 consecutive word pairs share an
  // instant exactly, the one exception a 0.5 s silence). Under the old rule that
  // word was in scope only in the degenerate tie where it ALSO had a zero-length
  // overlap with the holdback -- which is why `tied_word_starts_never_confirm_twice`
  // passed while this did not.
  //
  //   confirmed [A@0.99-1.00], holding [B@1.00, C@1.50], watermark 1.00
  //   ingest    [A@1.01, B@1.11, C@1.61, D@2.20] twice
  //
  // The alignment then has a UNIQUE optimum -- [A] ++ [B, C] against
  // [A, B, C, D] matches all three with the seam at 1, and no other seam scores
  // 3 -- so unlike the ambiguous shapes above, nothing here is a tie-break.
  //
  // Mutation proof: put the endpoint back (`take_while(|word| word.start() >=
  // watermark)`) and both assertions below red at once, to [" A", " A", " B"]
  // and " A A B C D" -- the finding as reported. Dropping the fast path's early
  // return instead changes nothing, which is the point: the fast path was never
  // wrong, its INPUT was.
  let a = || word(" A", 0.99, 1.0);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 1.5, 2.0);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(result_with_words(vec![a(), b(), c()]));
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a(), b(), c()]))
      .is_advanced()
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A"],
    "confirmed [A@0.99], holding [B@1.00, C@1.50] at a 1.00 s watermark",
  );
  assert!((agreement.last_agreed_seconds() - 1.0).abs() < 1e-6);

  // The re-decode moves every timestamp it emits, and " A" lands PAST the
  // watermark it used to end on.
  let drifted = || {
    result_with_words(vec![
      word(" A", 1.01, 1.1),
      word(" B", 1.11, 1.6),
      word(" C", 1.61, 2.1),
      word(" D", 2.2, 2.5),
    ])
  };
  agreement.ingest_streamed(drifted());
  assert!(agreement.ingest_streamed(drifted()).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "the drifted reproduction is refused; only the words past it advance",
  );
  assert_eq!(
    agreement
      .finalize(&crate::audio::whisper::options::DecodingOptions::new())
      .text(),
    " A B C D",
    "\" A\" reaches the transcript exactly once",
  );
}

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
  // What decides the ambiguity rule, pinned as the fact it is. The smallest
  // optimal seam attributes a disputed offered head to the HOLDBACK rather than
  // to `settled`, and the justification is not that keeping errs safe -- it is
  // that the engine WROTE the holdback into the next hypothesis. This asserts
  // the two halves of that contract that live in this module:
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
// The unmarked path: no prefill premise, so the other reading of the seam
// ---------------------------------------------------------------------

#[test]
fn a_direct_ingest_with_no_prefill_refuses_the_disputed_head() {
  // #94 (codex round 6, finding 1). THE SAME SEQUENCE as
  // `a_distinct_repetition_of_a_confirmed_word_survives_the_continuing_stream`,
  // with the opposite PROVENANCE -- and it must reach the opposite answer.
  //
  // There, the continuing results are decoded under the engine's own retarget,
  // so the decoder was FORCED to emit the holdback and the disputed leading " A"
  // can only be the holdback's own first word. Here the caller decoded some
  // other way: no clip retarget, no forced prefill. Nothing then says whether
  // that " A" is the held-back second occurrence coming back or the SETTLED
  // first one being re-offered while the held one was dropped -- and
  // `confirmed_words_slice()` is append-only, so guessing "held" and being wrong
  // confirms a settled word a second time with no way to retract it. The
  // unattributable reading refuses it.
  //
  // Mutation proof: read `seams.prefilled` on the unmarked path too (the single
  // policy this replaced) and BOTH faces below read back the marked path's
  // answers -- [" A", " A", " B"] and " A A B C D".
  //
  //   ingest_streamed [A@0.0, A@0.0, B@1.0] twice -> confirmed [A], holding [A, B]
  //   then BARE ingest [A@0.0, B@1.0, C@2.0, D@3.0] twice, caller's own options
  //   confirmed_words_slice() == [" A", " B"], text " A B C D"
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 2.0, 2.5);
  let d = || word(" D", 3.0, 3.5);
  let stutter = || result_with_words(vec![a(), a(), b()]);
  let mut agreement = LocalAgreement::new();
  agreement.ingest_streamed(stutter());
  assert!(agreement.ingest_streamed(stutter()).is_advanced());
  assert_eq!(confirmed_texts(&agreement), vec![" A"]);

  // The caller's OWN options: whatever it decoded with, it was not this engine's
  // retarget. `DecodingOptions::new()` already fails the premise on two clauses
  // at once -- no `clip_timestamps` at the watermark, no `prefix_tokens` at all.
  let unmarked = DecodingOptions::new();
  assert_ne!(
    unmarked.clip_timestamps_slice(),
    &[agreement.last_agreed_seconds()],
    "non-vacuous: these options are NOT the retarget",
  );
  let onward = || result_with_words(vec![a(), b(), c(), d()]);
  agreement.ingest(onward(), &unmarked);
  assert!(agreement.ingest(onward(), &unmarked).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "the disputed leading A could be the settled one coming back, so it is not \
     confirmed a second time",
  );

  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A B C D",
    "the finalized face of the same refusal",
  );
}

#[test]
fn a_stale_retarget_does_not_buy_the_prefilled_reading() {
  // The premise is checked against the CURRENT state, not against "these look
  // like retarget options". A caller that keeps re-offering the options the
  // engine issued a watermark ago has not prefilled the decoder with THIS
  // holdback, so it gets the unattributable reading -- the same answer as a
  // caller with no retarget at all.
  //
  // Mutation proof: drop the `clip_timestamps`/`prefix_tokens` clauses from
  // `prefill_reproduces_holdback` (check only `use_prefill_prompt`) and the
  // stale options buy the prefilled reading, reading back [" A", " A", " B"].
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 2.0, 2.5);
  let d = || word(" D", 3.0, 3.5);
  let mut agreement = LocalAgreement::new();
  // Captured BEFORE the advance: at this point the holdback is empty and the
  // watermark is 0.0, so these are the FIRST stride's options.
  let stale = agreement.decoding_options_for_next(&DecodingOptions::new());
  let stutter = || result_with_words(vec![a(), a(), b()]);
  agreement.ingest_streamed(stutter());
  assert!(agreement.ingest_streamed(stutter()).is_advanced());
  assert!(
    !agreement.last_agreed_words_slice().is_empty(),
    "non-vacuous: the state moved on from what `stale` targets",
  );

  let onward = || result_with_words(vec![a(), b(), c(), d()]);
  agreement.ingest(onward(), &stale);
  assert!(agreement.ingest(onward(), &stale).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "options that prefill an EMPTY holdback cannot make the offered head the \
     holdback's own reproduction",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A B C D"
  );
}

#[test]
fn an_unmarked_results_frontier_is_not_re_read_under_a_later_marked_one() {
  // #94 (codex round 7, finding 1). THE UNMARKED REFUSAL MUST OUTLIVE THE CALL
  // THAT MADE IT. `prev_result` is stored RAW and re-filtered on the next
  // ingest; if that re-filter reads the frontier under the NEXT result's
  // provenance, an unmarked result whose disputed head was refused gets that
  // head handed back one call later, on nothing more than the next caller
  // decoding honestly.
  //
  // Nobody lies here. The unmarked result really was decoded some other way, and
  // the marked one really was decoded under the retarget. Each is entitled to
  // its own frontier -- and re-reading the stored one under a foreign premise
  // manufactures an agreement neither hypothesis actually offered:
  // `find_longest_common_prefix` returns elements of the CURRENT hypothesis, but
  // its LENGTH is set by both sides, and `prev_words` read too permissively
  // makes that length LONGER, not shorter. Length is what decides whether the
  // watermark advances and how much lands in the append-only
  // `confirmed_words_slice()`.
  //
  // Mutation proof: filter `prev_result` with the CURRENT result's flag (the
  // shape this replaced -- pass `prefilled` where the stored provenance is read)
  // and both faces red: the fourth ingest reads back `Advanced` with
  // [" A", " A", " B"] confirmed, and the transcript becomes " A A B C D" --
  // " B" confirmed irrevocably on an agreement that never happened, so the
  // genuine later revision " Z" can never be applied.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 2.0, 2.5);
  let d = || word(" D", 3.0, 3.5);
  // " Z" revises the held-back " B", at " B"'s own extent.
  let z = || word(" Z", 1.0, 1.5);

  let mut agreement = LocalAgreement::new();
  let stutter = || result_with_words(vec![a(), a(), b()]);
  agreement.ingest_streamed(stutter());
  assert!(agreement.ingest_streamed(stutter()).is_advanced());
  assert_eq!(confirmed_texts(&agreement), vec![" A"]);
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A", " B"],
    "the disputed shape: one settled \" A\", and a holdback that starts with \
     another",
  );

  // (1) UNMARKED. The head is unattributable, so it is cut -- the answer
  // `a_direct_ingest_with_no_prefill_refuses_the_disputed_head` pins. This
  // result is now `prev_result`, stored raw.
  let unmarked = DecodingOptions::new();
  let onward = || result_with_words(vec![a(), b(), c(), d()]);
  assert!(
    agreement
      .ingest(onward(), &unmarked)
      .is_awaiting_agreement(),
    "[B, C, D] against the stutter's [A, B] shares no prefix",
  );

  // (2) MARKED, and the very same words. Its OWN frontier is the prefilled one
  // -- but the raw result stored at (1) keeps the unmarked frontier it arrived
  // with, so the two lists still disagree and nothing is confirmed.
  assert!(
    agreement.ingest_streamed(onward()).is_awaiting_agreement(),
    "the stored result's head is still refused, so [B, C, D] against \
     [A, B, C, D] still shares no prefix",
  );
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A"],
    "the streaming face: the unmarked refusal is not undone one call later",
  );

  // (3) The cost of getting (2) wrong, one revision later. " B" is still HELD,
  // so a hypothesis that revises it to " Z" supersedes it; had (2) confirmed
  // " B" the watermark would already be past " Z" and the revision could never
  // be applied at all.
  assert!(
    agreement
      .ingest_streamed(result_with_words(vec![a(), z(), c(), d()]))
      .is_awaiting_agreement(),
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A A Z C D",
    "the finalized face: the revision reaches the transcript because the word it \
     revises was never confirmed on a manufactured agreement",
  );
}

#[test]
fn a_recurring_ambiguity_would_never_clear_on_the_unmarked_path() {
  // THE REFUTATION of "block on an ambiguous frontier, since without the
  // prefill nothing re-creates the ambiguity every stride". The prefill is not
  // what re-creates it. The disputed columns' scores are a function of
  // `(settled, holdback, offered)` alone; `settled` and `holdback` move only on
  // an advance -- which is exactly what a block prevents -- and growing the
  // offered list at the TAIL leaves the disputed columns tied, because
  // `before[j]` is untouched there and `after[j]` rises for every `j` at once.
  //
  // So a caller re-decoding a growing window off unchanged audio re-offers the
  // same head every stride whether or not it prefills: a deterministic decoder
  // does not need to be forced to repeat itself. A block would re-arm on every
  // one of the strides below and the stream would never confirm another word --
  // trading one forgone repetition for the entire rest of the transcript.
  //
  // Mutation proof: this is an assertion about `settled_seams` itself, so its
  // falsifier is the claim's negation -- assert the seams CONVERGE at some
  // stride and every row below reds.
  let settled = [word(" A", 0.0, 0.5)];
  let holdback = [word(" A", 0.0, 0.5), word(" B", 1.0, 1.5)];
  let mut offered = vec![word(" A", 0.0, 0.5), word(" B", 1.0, 1.5)];
  for stride in 0..6usize {
    let seams = settled_seams(&settled, &holdback, &offered);
    assert_eq!(
      (seams.prefilled, seams.unattributed),
      (0, 1),
      "stride {stride}: the seam is still ambiguous with {} offered words, so a \
       blocking policy re-arms here too",
      offered.len(),
    );
    // The stream grows at the tail, which is the only way it CAN grow while the
    // watermark is pinned by the block.
    let next = 2.0 + stride as f32;
    offered.push(word(&format!(" W{stride}"), next, next + 0.5));
  }

  // Liveness, for contrast: the unattributable reading advances on this exact
  // shape rather than stalling on it -- see
  // `a_direct_ingest_with_no_prefill_refuses_the_disputed_head`, whose stream
  // reaches `Advanced` and confirms " B" while a block would still be waiting.
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
  // instead of four, and the finalized text loses " w2 w3" entirely. The empty
  // PENDING half has its own falsifier: relax the advance's `word.end() >
  // self.last_agreed_seconds` to `>=` and " w3" is deferred instead of
  // confirmed, though nothing offered can ever overlap it.
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
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" w0", " w1", " w2", " w3"], Vec::<&str>::new()),
    "the two words that could not be held are CONFIRMED, not dropped -- and \
     confirmed OUTRIGHT, not left pending: both end at or before the watermark, \
     so no word the engine will ever be offered again can overlap them (codex \
     round 12, finding 1)",
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
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (
      vec![" w0", " w1", " w2", " w3", " w4", " w5"],
      Vec::<&str>::new()
    ),
    "confirmation kept moving, and nothing was retracted",
  );

  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " w0 w1 w2 w3 w4 w5 w6 w7 w8",
    "every word the stream produced reaches the transcript",
  );
}

#[test]
fn a_prefill_over_the_budget_does_not_buy_the_prefilled_reading() {
  // A caller assembling its own options can present a prefix that ENDS with the
  // holdback behind a head of its own. `prefill_tokens` keeps the last
  // `MAX_HOLDBACK_PREFILL_TOKENS` of it, so `decode_text` is forced to emit that
  // head too and the hypothesis does NOT begin with the holdback -- the premise
  // the smallest-seam reading rests on is simply false. Equality is what refuses
  // it, and equality is what the premise needs.
  //
  // Mutation proof: weaken the comparison in `prefill_reproduces_holdback` to
  // `decoded_under.prefix_tokens_slice().ends_with(...)` -- the plausible
  // mistake, since the decoder does receive the holdback at the tail -- and the
  // over-budget options below buy the prefilled reading, reading back
  // [" A", " A", " B"] / " A A B C D".
  //
  // ROUND 7 CORRECTION: this test used to record its falsifier as dropping the
  // `holdback.len() <= MAX_HOLDBACK_PREFILL_TOKENS` clause. That mutation reds
  // NOTHING, here or anywhere -- the padded prefix differs from the holdback in
  // LENGTH as well as content, so the equality already refuses it and the length
  // clause never decided this case. It cannot decide any case now: the clause
  // read the ENGINE's holdback, which `budgeted_split` keeps inside the budget
  // after every advance, so it was a conjunct that could not be false. It is
  // gone, and this is the mutation that actually falsifies what is left.
  let a = || word(" A", 0.0, 0.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 2.0, 2.5);
  let d = || word(" D", 3.0, 3.5);
  let mut agreement = LocalAgreement::new();
  let stutter = || result_with_words(vec![a(), a(), b()]);
  agreement.ingest_streamed(stutter());
  assert!(agreement.ingest_streamed(stutter()).is_advanced());

  // The engine's own retarget, then the prefix swapped for an over-budget one
  // that still ENDS with the holdback -- so `prefill_tokens` would hand
  // `decode_text` a tail that reproduces it, but nothing forces the head.
  let honest = agreement.decoding_options_for_next(&DecodingOptions::new());
  let mut padded: Vec<u32> = (10_000..10_000 + MAX_HOLDBACK_PREFILL_TOKENS as u32).collect();
  padded.extend_from_slice(honest.prefix_tokens_slice());
  assert!(padded.len() > MAX_HOLDBACK_PREFILL_TOKENS);
  let over_budget = honest.clone().with_prefix_tokens(padded);

  let onward = || result_with_words(vec![a(), b(), c(), d()]);
  agreement.ingest(onward(), &over_budget);
  assert!(agreement.ingest(onward(), &over_budget).is_advanced());
  assert_eq!(
    confirmed_texts(&agreement),
    vec![" A", " B"],
    "a prefix the decoder will trim is not a reproduction of the holdback",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A B C D"
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
  // The unmarked seam does not repair that: it is the right frontier for a
  // hypothesis the engine did not write, and it cannot put back a token the
  // engine itself dropped. What follows is DATA LOSS, not a stall -- the
  // truncated hypothesis disagrees, `holdback_superseded` fires, and `finalize`
  // replaces the intact held word with the truncation.
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
  // Settled OUTRIGHT rather than left pending -- the watermark is " H"'s own
  // end, so the still-open span starts where " H" stops and can never overlap
  // it (codex round 12, finding 1).
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " H"], Vec::<&str>::new()),
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
fn a_holdback_word_the_prefill_would_filter_is_confirmed_rather_than_held() {
  // #94 (codex round 8, finding 1). THE CAP MUST CAP THE OTHER FILTER TOO.
  // `prefill_tokens` reduces `prefix_tokens` twice on its way to the decoder: it
  // keeps only the last `MAX_HOLDBACK_PREFILL_TOKENS` ids, AND it drops every id
  // at or above the vocabulary's `special_token_begin`. `budgeted_split` bounded
  // only the first, and `prefill_reproduces_holdback` proves the caller's prefix
  // EQUALS the holdback -- which is exactly as true for a holdback whose tokens
  // the second filter erases. So a hand-built stream can settle a word, hold a
  // word made of ids the decoder never receives, pass the retarget
  // `decoding_options_for_next` just issued, and have `ProvenancedResult` record
  // `prefilled = true` for a hypothesis the engine did NOT write the holdback
  // into.
  //
  // REACHABILITY, since round 3 classified this half unreachable. That
  // classification's evidence is correct -- `update_segments_with_word_timings`
  // strips `id >= special_token_begin` from every `WordTiming` and emits no word
  // at all for an all-special entry, so this crate's pipeline never produces
  // one. What it does not cover is `ingest` itself, which is public, takes a
  // hand-built `TranscriptionResult`, and sets `last_agreed_words` to
  // `common[split..]` -- elements of the caller's own hypothesis, tokens
  // included. `WordTiming::new` takes the token vector directly. Nothing between
  // there and `holdback_prefill_tokens` looks at an id.
  //
  // NEITHER SEAM IS THE REPAIR, which is why the fix is in the split. Read the
  // arithmetic of this very stream three ways:
  //
  //   - held and read MARKED (the defect): " S" is never confirmed, the
  //     re-offer supersedes the holdback that carried it, and the transcript
  //     loses it -- " A A Y C".
  //   - held and read UNATTRIBUTED (refuse the marked reading): the offered head
  //     is cut as well, so the stream loses " S" AND the " A" that replaced it --
  //     " A Y C". Refusing is not the conservative choice here; the state is the
  //     defect, not the reading of it.
  //   - CONFIRMED instead of held (this fix): " A S A Y C", every word two
  //     hypotheses agreed on, and the later revision still applied.
  //
  // Confirming is no weaker a claim than any other confirmation carries --
  // `common` is the prefix two consecutive hypotheses agreed on, LocalAgreement-2's
  // entire criterion -- and holding buys nothing, because whatever the next
  // hypothesis produces over that extent was decoded from a prefix the held word
  // is not in, so it is neither a corroboration of it nor a revision of it.
  //
  // Mutation proof: drop `budgeted_split`'s `rposition` floor (equivalently,
  // weaken `prefill_carries_whole` to `true`) and every assertion below reds --
  // " S" is held rather than confirmed, the issued prefix carries the id the
  // filter erases, and the transcript reads " A A Y C" with " S" gone. The two
  // guards have their own falsifiers, named where they stand: widen
  // `prefill_carries_whole` to `false` for every word and the issued prefill goes
  // EMPTY, and take the plain maximum seam and the ambiguity pin reads (0, 3).
  let a0 = || word(" A", 0.5, 1.0);
  let s = || special_only_word(" S", 1.0, 1.5);
  let a1 = || word(" A", 1.5, 2.0);
  let b = || word(" B", 2.0, 2.5);
  let c = || word(" C", 2.5, 3.0);
  // " Y" revises " B", at " B"'s own extent.
  let y = || word(" Y", 2.0, 2.5);
  assert!(
    s()
      .tokens_slice()
      .iter()
      .all(|&id| id >= MIN_SPECIAL_TOKEN_BEGIN),
    "non-vacuous: the held word is made only of ids `prefill_tokens` drops",
  );

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![a0(), s(), a1()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());

  // The streaming face, read between pushes: the word the prefill cannot carry
  // is SETTLED, and the holdback is the one word that survives the filter whole.
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " S"], Vec::<&str>::new()),
    "a word the decoder can never be given is CONFIRMED, never held -- and \
     outright, since \" S\" ends exactly where the watermark starts and no \
     offered word can overlap it (codex round 12, finding 1)",
  );
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" A"],
    "and the holdback keeps only what the prefill reproduces",
  );
  assert_eq!(
    agreement.last_agreed_seconds(),
    1.5,
    "the watermark moves to the first word still held",
  );

  // The face the decoder sees: every id the engine issues survives BOTH of
  // `prefill_tokens`'s reductions, so the equality
  // `prefill_reproduces_holdback` checks really does establish that the
  // hypothesis was written to begin with the whole holdback.
  let next = agreement.decoding_options_for_next(&DecodingOptions::new());
  assert!(
    next
      .prefix_tokens_slice()
      .iter()
      .all(|&id| id < MIN_SPECIAL_TOKEN_BEGIN),
    "the issued prefix carries an id the decoder's filter erases: {:?}",
    next.prefix_tokens_slice(),
  );
  assert!(
    !next.prefix_tokens_slice().is_empty(),
    "and it is not empty, so there is something for the decoder to reproduce",
  );

  // The AMBIGUOUS frontier this fix makes unreachable, stated as the shape it
  // is: had " S" stayed in the holdback, a re-offer of the settled word's text
  // would have left the seam genuinely disputed, and the marked reading would
  // have taken column 0 on a premise the decoder never established.
  let re_offer = || result_with_words(vec![a1(), b(), c()]);
  assert_eq!(
    {
      let seams = settled_seams(&[a0()], &[s(), a1()], &[a1(), b(), c()]);
      (seams.prefilled, seams.unattributed)
    },
    (0, 1),
    "non-vacuous: the state the fix removes really was an ambiguous frontier",
  );

  // The continuation. The re-offer reproduces the one held word and carries the
  // stream on; the pair agrees on the next stride and settles it. Outcome and
  // settled span are read TOGETHER at each step, so the step itself has a
  // falsifier rather than only the state after it.
  assert_eq!(
    (
      agreement.ingest_streamed(re_offer()),
      confirmed_texts(&agreement)
    ),
    (AgreementOutcome::AwaitingAgreement, vec![" A", " S"]),
    "one reproduced word is short of `agreement_count_needed`, so nothing new \
     settles here",
  );
  assert_eq!(
    (
      agreement.ingest_streamed(re_offer()),
      confirmed_texts(&agreement)
    ),
    (AgreementOutcome::Advanced, vec![" A", " S", " A"]),
    "the settled span grows over the word the prefill could not carry, never \
     through it",
  );

  // The genuine later revision, which must still be applied: " B" is only HELD,
  // so a hypothesis that revises it to " Y" supersedes it.
  let revised = agreement.ingest_streamed(result_with_words(vec![y(), c()]));
  assert_eq!(
    (
      revised,
      agreement
        .finalize(&DecodingOptions::new())
        .text()
        .to_string()
    ),
    (
      AgreementOutcome::AwaitingAgreement,
      " A S A Y C".to_string()
    ),
    "the finalized face: the unreproducible word survives BECAUSE it was \
     confirmed, and the later revision is still free to land",
  );
}

#[test]
fn the_split_holds_back_exactly_what_the_prefill_carries_whole() {
  // The postcondition `budgeted_split` now has, and its MINIMALITY -- written out
  // longhand rather than through `prefill_carries_whole`, so a mutation of that
  // predicate cannot mutate its own falsifier along with it.
  //
  // Four things a row here pins that the end-to-end counterexample does not: a
  // word with NO tokens at all (it contributes nothing to `prefix_tokens`, so the
  // decoder is never given it either -- the same defect, with no vocabulary
  // knowledge required to see it); a filtered word BEHIND a carriable one, which
  // is why the floor reads `rposition` and not `position` (the FIRST such word is
  // not the one the split has to clear); the two reductions side by side, so
  // neither is enforced only by cancelling the other; and the id filter read
  // against a THRESHOLD rather than against the constant, rows 6 and 7 differing
  // in nothing else (codex round 12, finding 2).
  //
  // Mutation proof, every row enumerated by running it: drop `!tokens.is_empty()`
  // and row 1 reds; use `position` for the floor and row 3 reds; drop the floor
  // entirely and rows 0-3 and 6 red on the postcondition; make `budgeted_split`
  // the identity and rows 0-4 and 6 do; return `common.len()` unconditionally and
  // rows 0, 1, 3, 4, 5, 6 and 7 red on the MINIMALITY clause instead; read
  // `MIN_SPECIAL_TOKEN_BEGIN` in `budgeted_split` instead of the threshold it is
  // given and row 6 reds alone.
  let plain =
    |text: &str, start: f32, count: usize| word_of_tokens(text, start, start + 1.0, count);
  let filtered = |text: &str, start: f32| special_only_word(text, start, start + 1.0);
  let tokenless =
    |text: &str, start: f32| WordTiming::new(text, Vec::new(), start, start + 1.0, 0.9);
  // An id BELOW `MIN_SPECIAL_TOKEN_BEGIN` and so carriable under the floor, but
  // at or above the lower threshold rows 6/7 configure -- the artifact shape
  // codex round 12, finding 2 is about.
  let below_floor =
    |text: &str, start: f32| WordTiming::new(text, vec![45_000], start, start + 1.0, 0.9);

  // `(common, requested, special_token_begin)`.
  let cases: Vec<(Vec<WordTiming>, usize, u32)> = vec![
    // 0: the counterexample's own shape -- a filtered word at the holdback head.
    (
      vec![
        plain(" A", 0.0, 1),
        filtered(" S", 1.0),
        plain(" B", 2.0, 1),
      ],
      1,
      MIN_SPECIAL_TOKEN_BEGIN,
    ),
    // 1: no tokens at all, in the same position.
    (
      vec![
        plain(" A", 0.0, 1),
        tokenless(" S", 1.0),
        plain(" B", 2.0, 1),
      ],
      1,
      MIN_SPECIAL_TOKEN_BEGIN,
    ),
    // 2: a filtered word at the holdback TAIL -- the split has to clear that one
    //    too, so the holdback comes back empty.
    (
      vec![
        plain(" A", 0.0, 1),
        plain(" B", 1.0, 1),
        filtered(" S", 2.0),
      ],
      1,
      MIN_SPECIAL_TOKEN_BEGIN,
    ),
    // 3: a carriable word BETWEEN two filtered ones. `position` stops at the
    //    first and leaves the second held; `rposition` clears both.
    (
      vec![
        plain(" A", 0.0, 1),
        filtered(" S", 1.0),
        plain(" B", 2.0, 1),
        filtered(" T", 3.0),
        plain(" C", 4.0, 1),
      ],
      1,
      MIN_SPECIAL_TOKEN_BEGIN,
    ),
    // 4: the length reduction, unchanged by any of the above.
    (
      vec![
        plain(" A", 0.0, 1),
        plain(" B", 1.0, MAX_HOLDBACK_PREFILL_TOKENS),
        plain(" C", 2.0, MAX_HOLDBACK_PREFILL_TOKENS),
      ],
      1,
      MIN_SPECIAL_TOKEN_BEGIN,
    ),
    // 5: nothing to do -- every word carriable, the whole holdback in budget.
    (
      vec![
        plain(" A", 0.0, 1),
        plain(" B", 1.0, 1),
        plain(" C", 2.0, 1),
      ],
      1,
      MIN_SPECIAL_TOKEN_BEGIN,
    ),
    // 6: an id the FLOOR calls carriable and a lower-threshold vocabulary
    //    filters. The split must clear it against the threshold it was GIVEN,
    //    not against the constant (codex round 12, finding 2).
    (
      vec![
        plain(" A", 0.0, 1),
        below_floor(" S", 1.0),
        plain(" B", 2.0, 1),
      ],
      1,
      40_000,
    ),
    // 7: row 6's own control -- the same three words under the default floor,
    //    where the same id IS carriable and nothing needs clearing. Rows 6 and 7
    //    differ only in the threshold, so a split that ignored it would fail one
    //    of them whatever it returned.
    (
      vec![
        plain(" A", 0.0, 1),
        below_floor(" S", 1.0),
        plain(" B", 2.0, 1),
      ],
      1,
      MIN_SPECIAL_TOKEN_BEGIN,
    ),
  ];

  let holdable = |words: &[WordTiming], special_token_begin: u32| {
    words.iter().all(|word| {
      !word.tokens_slice().is_empty()
        && word
          .tokens_slice()
          .iter()
          .all(|&id| id < special_token_begin)
    }) && words
      .iter()
      .map(|word| word.tokens_slice().len())
      .sum::<usize>()
      <= MAX_HOLDBACK_PREFILL_TOKENS
  };

  for (row, (common, requested, special_token_begin)) in cases.iter().enumerate() {
    let split = budgeted_split(common, *requested, *special_token_begin);
    let texts: Vec<&str> = common.iter().map(WordTiming::word).collect();
    assert!(
      (*requested..=common.len()).contains(&split),
      "row {row} ({texts:?}): split {split} left the requested-to-end range",
    );
    assert!(
      holdable(&common[split..], *special_token_begin),
      "row {row} ({texts:?}): the holdback at {split} is not one the prefill \
       carries whole",
    );
    for earlier in *requested..split {
      assert!(
        !holdable(&common[earlier..], *special_token_begin),
        "row {row} ({texts:?}): split {split} confirms more than it has to -- \
         {earlier} already holds only carriable words within budget",
      );
    }
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn the_prefill_token_ceiling_is_below_the_vocabularys_special_range() {
  // `MIN_SPECIAL_TOKEN_BEGIN` is the bound `budgeted_split` tests against with no
  // tokenizer in hand, and it is only sound while it is at or below the LOADED
  // vocabulary's own `special_token_begin` -- otherwise an id `prefill_tokens`
  // filters would be held anyway. Asserted against the real artifact rather than
  // argued from the constant's doc.
  //
  // Mutation proof: raise the constant above the shipped vocabulary's threshold
  // (`50_258`) and this reds.
  let special_token_begin = tiny_tokenizer().special_tokens().special_token_begin();
  assert!(
    special_token_begin >= MIN_SPECIAL_TOKEN_BEGIN,
    "the vocabulary reserves ids from {special_token_begin}, below the \
     {MIN_SPECIAL_TOKEN_BEGIN} the holdback rule assumes",
  );
}

#[test]
fn with_nothing_held_back_both_frontiers_are_the_same_column() {
  // The property `prefill_reproduces_holdback`'s vacuous `true` now rests on.
  // Before `budgeted_split` could leave an EMPTY holdback, an empty holdback
  // implied an empty `confirmed_words` too, so `watermark_filtered` took its
  // `settled_len == 0` fast path and no frontier was read at all -- the premise
  // could not matter because nothing consulted it. It can now be empty beside a
  // NON-empty settled set, so a frontier IS read; it still cannot matter,
  // because with no holdback the two policies name the same column.
  //
  // With an empty holdback `after[j]` is identically zero (`settled_seams`'
  // second DP runs its outer loop over an empty pattern), so `score(j) ==
  // before[j]`, which is non-decreasing. `best == before[len]`; the smallest
  // optimal seam is the first column to reach it; and every larger column has
  // `before[j] == before[j - 1]`, so none of them is match-SUPPORTED.
  //
  // Mutation proof: take the plain maximum for `unattributed` (drop the
  // `before[j] > before[j - 1]` clause) and the rows whose settled word matches
  // nothing red -- the plain maximum is `offered.len()` there, against a
  // smallest seam of 0.
  let settled = [word(" A", 0.0, 0.5)];
  for offered in [
    Vec::new(),
    vec![word(" A", 0.0, 0.5)],
    vec![word(" X", 0.0, 0.5), word(" Y", 1.0, 1.5)],
    vec![word(" A", 0.0, 0.5), word(" X", 1.0, 1.5)],
    vec![
      word(" X", 0.0, 0.5),
      word(" A", 1.0, 1.5),
      word(" A", 2.0, 2.5),
    ],
  ] {
    let seams = settled_seams(&settled, &[], &offered);
    assert_eq!(
      seams.unattributed,
      seams.prefilled,
      "with an empty holdback the premise cannot change the frontier: {:?}",
      offered.iter().map(WordTiming::word).collect::<Vec<_>>(),
    );
  }
}

// ---------------------------------------------------------------------
// The prompt itself, at the layer the decoder reads it
// ---------------------------------------------------------------------

#[test]
fn the_first_strides_options_keep_the_callers_prefill_flag() {
  // #94 (codex round 6, finding 3). Before the first advance the engine holds no
  // holdback, so there is no premise to enforce and no frontier to read -- and
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
     the frontier's tie-break has nothing behind it",
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
  // The contract `LocalAgreement::watermark_filtered`'s ambiguity rule rests on,
  // pinned at the layer that decides it. `DecodingOptions::prefix_tokens` is
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
fn an_unmarked_hypothesis_that_reproduces_nothing_settled_is_kept_whole() {
  // The other half of the unattributable reading, and the reason it is the
  // largest MATCH-SUPPORTED optimal seam rather than the largest one. When
  // `settled` matches nothing in the offered list, every seam scores the same
  // and the plain maximum is `offered.len()` -- the entire hypothesis refused,
  // on a stream that re-admitted nothing at all. Repeat that each stride and the
  // engine never confirms another word: a permanent stall wearing caution's
  // clothes. Requiring the word before the seam to be matched to a settled word
  // pins the frontier at 0 here, and the stream keeps moving.
  //
  // Mutation proof: drop the `frontier == 0 || before[frontier] >
  // before[frontier - 1]` conjunct from `settled_seams` (take the plain maximum)
  // and both faces red -- the confirmed list stops at [" A"] and the transcript
  // loses " X Y Z".
  let mut agreement = LocalAgreement::new();
  let settled = || {
    result_with_words(vec![
      word(" A", 0.0, 1.0),
      word(" B", 1.0, 2.0),
      word(" C", 2.0, 3.0),
    ])
  };
  agreement.ingest_streamed(settled());
  assert!(agreement.ingest_streamed(settled()).is_advanced());
  assert_eq!(confirmed_texts(&agreement), vec![" A"]);
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" B", " C"],
    "non-vacuous: `settled` is [\" A\"] and REACHES the watermark, so the \
     alignment is live -- this is not the settled_len == 0 fast path",
  );

  // Wholly different text, decoded some other way: nothing here can be a
  // settled word coming back, and nothing may be refused as one.
  let unmarked = DecodingOptions::new();
  let onward = || {
    result_with_words(vec![
      word(" X", 1.0, 2.0),
      word(" Y", 2.0, 3.0),
      word(" Z", 3.0, 4.0),
    ])
  };
  agreement.ingest(onward(), &unmarked);
  assert!(
    agreement.ingest(onward(), &unmarked).is_advanced(),
    "the unmarked path must still make progress, not stall on unrelated text",
  );
  assert_eq!(confirmed_texts(&agreement), vec![" A", " X"]);
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A X Y Z",
  );
}

#[test]
fn a_widened_word_is_not_confirmed_before_an_unmarked_continuation_can_revise_it() {
  // #94 (codex round 12, finding 1). THE WIDENING'S OWN PREMISE IS ABOUT THE
  // NEXT RESULT, AND THE SPLIT RUNS BEFORE IT EXISTS. `budgeted_split` widens
  // past a word the prefill cannot carry, and round 8's argument for settling it
  // on the spot -- "whatever the next hypothesis produces over that extent was
  // decoded from a prefix the word is not in, and from audio the clip excludes"
  // -- is true of a MARKED continuation and of nothing else. A caller that does
  // not use `decoding_options_for_next` is subject to neither reduction: its
  // decoder saw that audio and can revise the word. Confirming at the advance
  // strands the stale reading beside the revision, which is the exact defect
  // `finalize`'s `holdback_superseded` branch exists to prevent -- reached one
  // word earlier and through the append-only list, where nothing can take it
  // back.
  //
  // The split still has to widen: with " S" left in the holdback the retarget
  // would prefill " B" while clipping at " S"'s own start, so a marked
  // continuation would be forced to contradict the audio it was given and would
  // disagree every stride (round 8's arithmetic). What is deferrable is the
  // widened-past word's DESTINATION, not the split, so it goes to
  // `pending_words` and `ingest` settles it on the first hypothesis whose window
  // starts at the watermark.
  //
  // " S" is tokenless, so `budgeted_split` widens past it; " B" ties with it at
  // 1.0, so the watermark stays at 1.0 and " S" OVERLAPS the still-open span --
  // which is what leaves it revisable at all (see the advance's own `end >
  // watermark` split, and `the_split_settles_a_widened_word_the_span_can_never_
  // reach` for the other side of that line).
  //
  // Mutation proof, both faces: put the widened-past words straight into
  // `confirmed_words` at the advance (`extend_from_slice(&common[..split])`,
  // with no `pending_words`) and the state reads back ([" A", " S"], []) and the
  // transcript " A S Y B C" -- " S" and its own replacement " Y" side by side.
  // The transcript face has a second falsifier of its own: append
  // `pending_words` ahead of `hypothesis_words` on `finalize`'s superseded path
  // instead of dropping them, and the same " A S Y B C" comes back.
  let a = || word(" A", 0.0, 1.0);
  let s = || special_only_word(" S", 1.0, 1.5);
  let b = || word(" B", 1.0, 1.5);
  // " Y" revises " S", at " S"'s own extent.
  let y = || word(" Y", 1.0, 1.5);
  let c = || word(" C", 1.5, 2.0);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![a(), s(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());

  // The streaming face, read between pushes: " S" is agreed and out of the
  // holdback, but NOT yet irrevocable.
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A"], vec![" S"]),
    "the widened-past word waits for the provenance that decides it",
  );
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" B"],
    "non-vacuous: the split really did widen -- \" S\" is out of the holdback, \
     so the prefill below carries no id the decoder's filter erases",
  );
  assert!(
    agreement
      .decoding_options_for_next(&DecodingOptions::new())
      .prefix_tokens_slice()
      .iter()
      .all(|&id| id < MIN_SPECIAL_TOKEN_BEGIN),
    "and the retarget is the widened one, unchanged by the deferral",
  );
  // Finalized HERE, with nothing having superseded it, the pending word is still
  // in the transcript: deferring is not dropping. (Mutation: delete
  // `finalize`'s Swift-shaped `append(&mut self.pending_words)` and this reads
  // " A B".)
  assert_eq!(
    agreement.clone().finalize(&DecodingOptions::new()).text(),
    " A S B",
    "nothing superseded it, so it is still the only estimate for its span",
  );

  // The continuation the widening's premise does not cover: decoded some other
  // way, so it saw " S"'s audio and revised it.
  let unmarked = DecodingOptions::new();
  assert!(
    agreement
      .ingest(result_with_words(vec![y(), b(), c()]), &unmarked)
      .is_awaiting_agreement(),
    "the revision disagrees with the hypothesis it revises",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A Y B C",
    "the finalized face: the revision replaces the word it revises rather than \
     landing beside it",
  );
}

#[test]
fn a_marked_continuation_settles_the_widened_word_it_could_not_have_revised() {
  // The other half of the pair above, and the reason the deferral is not a
  // regression on round 8's fix: a hypothesis decoded under the retarget could
  // not have revised the widened-past word (no prefill token of it, and its
  // audio outside the clip), so the word settles the moment one arrives -- one
  // ingest later than round 8 settled it, and irrevocably from then on.
  //
  // Mutation proof: delete the promotion drain from `ingest` and the SECOND
  // assertion reads back ([" A"], [" S"]) -- the word never settles, and the
  // stream that could not have revised it is treated as though it might.
  let a = || word(" A", 0.0, 1.0);
  let s = || special_only_word(" S", 1.0, 1.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 1.5, 2.0);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![a(), s(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A"], vec![" S"]),
    "non-vacuous: there is a pending word for the next stride to settle",
  );

  // `ingest_streamed` decodes under the engine's own retarget, so its window
  // starts at the watermark.
  agreement.ingest_streamed(result_with_words(vec![b(), c()]));
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " S"], Vec::<&str>::new()),
    "a hypothesis clipped at the watermark settles it",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A S B C",
    "and it stays in the transcript, exactly as round 8 required",
  );
}

#[test]
fn an_empty_holdback_leaves_nothing_pending_because_nothing_could_ever_clear_it() {
  // The second half of the advance's reachability split, and the reason the
  // promotion above can read `prefill_reproduces_holdback` without ever reading
  // it VACUOUSLY. That predicate answers TRUE for anything when the holdback is
  // empty -- a state `budgeted_split` reaches by design (round 7, finding 2) --
  // and its vacuity was only ever justified for the FRONTIER, where both
  // readings name the same column
  // (`with_nothing_held_back_both_frontiers_are_the_same_column`).
  //
  // Rather than bolt a clause on for a state that must not arise, the state is
  // removed: a pending word waits for a hypothesis this engine ANCHORED, and
  // with nothing held back no future result can ever be one. Waiting would be an
  // indefinite hold, and `finalize`'s superseded path would end it by deleting a
  // word the stream produced -- exactly the loss round 7 finding 2 removed. So
  // the whole widened-past run is settled at the advance, and
  // `pending_words` non-empty implies `last_agreed_words` non-empty for good.
  //
  // The shape: " L" runs 1.0..5.0 and " B" 2.0..3.0, so the budget forces the
  // split to the end (an empty holdback, watermark at 3.0) and " L" OVERLAPS the
  // still-open span -- the one case the overlap test alone would defer. Word ends
  // inside a hypothesis are not monotone, which is also why the advance takes a
  // `position` prefix.
  //
  // Mutation proof, one face each, and the second one is codex round 13 finding
  // 1's recommendation run end to end. Dropping the
  // `self.last_agreed_words.is_empty()` arm from the advance's second split reds
  // the STATE face at ([], [" L", " B"]) -- and stops there: the transcript
  // survives, because `prefill_reproduces_holdback` answers the unanchored
  // continuation below TRUE vacuously and the promotion puts both words back.
  // (Round 12 recorded that mutation as reaching the transcript too; it does
  // not, and the vacuity is why.) Add the other two clauses that finding asks
  // for -- promote only on a NON-VACUOUS anchor (`&& !self.last_agreed_words
  // .is_empty()`), and read `seams.unattributed` while the holdback is empty --
  // and the transcript face reds at " Y Z": " L" and " B" deleted, which is
  // round 7 finding 2's loss returning. Nothing else in this module reds with
  // all three applied, so that deletion is the whole of what the
  // recommendation buys.
  let long = || word_of_tokens(" L", 1.0, 5.0, 1);
  let big = || word_of_tokens(" B", 2.0, 3.0, MAX_HOLDBACK_PREFILL_TOKENS + 1);
  let y = || word(" Y", 3.0, 5.0);
  let z = || word(" Z", 5.0, 6.0);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![long(), big()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert!(
    agreement.last_agreed_words_slice().is_empty(),
    "non-vacuous: the budget left NOTHING held back, so there is no anchor a \
     later result could be checked against",
  );
  assert_eq!(agreement.last_agreed_seconds(), 3.0);
  assert!(
    long().end() > agreement.last_agreed_seconds(),
    "non-vacuous: \" L\" DOES overlap the still-open span, so the overlap test \
     alone would have deferred it",
  );
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" L", " B"], Vec::<&str>::new()),
    "with no anchor to wait for, the widened-past run is settled at the advance",
  );

  // And it stays settled through a continuation the engine did not write.
  let unmarked = DecodingOptions::new();
  assert!(
    agreement.prefill_reproduces_holdback(&unmarked),
    "non-vacuous: these options satisfy the prefill premise VACUOUSLY -- the \
     answer that would have been read had anything been left pending",
  );
  assert!(
    agreement
      .ingest(result_with_words(vec![y(), z()]), &unmarked)
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " L B Y Z",
    "the finalized face: nothing the stream agreed on was dropped by the wait \
     it never took",
  );
}

#[test]
fn an_overlapping_agreed_word_is_confirmed_on_the_mainline_path_too() {
  // #94 (codex round 13, finding 1), REFUTED HERE RATHER THAN FIXED, and this is
  // the row that says why. The finding reads the empty-holdback arm of the
  // advance's second split as a NEW hazard: `overlapping` is forced to
  // `widened.len()`, so a widened-past word whose `end` is past the watermark is
  // confirmed even though a later unanchored decode could re-read its audio, and
  // `an_empty_holdback_leaves_nothing_pending_because_nothing_could_ever_clear_
  // it` is offered as the counterexample (" L" 1.0..5.0 confirmed, then " Y"
  // 3.0..5.0 landing beside it).
  //
  // That reading of the state IS what that test asserts. What makes it not a
  // defect is this row: `common[..requested]` -- the MAINLINE confirmation, the
  // one Swift has and this port has never touched -- appends with no overlap
  // test of any kind, and lands in exactly the same place. Word ends inside a
  // hypothesis are not monotone, so an agreed word can extend past the first
  // held-back word's start; here " P" runs 0.0..5.0 while the watermark lands at
  // " Q"'s 1.0, and the unmarked " Y" that follows re-reads 1.0..5.0. Both are
  // in the transcript, overlapping.
  //
  // So "a confirmed word overlaps a later hypothesis's word" is not a property
  // the empty-holdback arm introduces; it is the LocalAgreement-2 contract, which
  // confirms on agreement between two consecutive hypotheses and is append-only.
  // The frontier is what decides whether an offered word is a settled one coming
  // BACK (`settled_seams`), and it decides that by alignment; it said " Y" is not
  // " P". The alternative the finding proposes -- hold the word until an anchor
  // certifies it -- has no terminating condition when the holdback is empty, and
  // ends in the deletion round 7 finding 2 removed. See that test's own comment.
  //
  // Mutation proof: this row's whole point is that no mutation of the
  // empty-holdback arm can red it -- the arm is not on this path. The mutation
  // that DOES red it is the finding's own recommendation carried to its logical
  // end: take the same `position(|word| word.end() > self.last_agreed_seconds)`
  // prefix of `common[..requested]` that the widened run takes, and the state
  // below reads back ([], []) -- " P" never reaches `confirmed_words_slice()` at
  // all. Eighteen other rows in this module red with it, which is the measure of
  // how much of the port that recommendation actually moves.
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
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" P"], Vec::<&str>::new()),
    "non-vacuous: \" P\" came from `common[..requested]`, not from the widened \
     run -- nothing was ever pending here",
  );
  assert!(
    p().end() > agreement.last_agreed_seconds(),
    "non-vacuous: the mainline path confirmed a word that OVERLAPS the \
     still-open span, with no overlap test anywhere on it",
  );

  let unmarked = DecodingOptions::new();
  assert!(
    agreement
      .ingest(result_with_words(vec![y(), z()]), &unmarked)
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " P Y Z",
    "the overlapping confirmed word and the later re-reading of its audio are \
     BOTH in the transcript -- the append-only contract, reached without the \
     empty-holdback arm being involved at all",
  );
}

#[test]
fn a_retarget_with_a_foreign_prefix_does_not_settle_a_pending_word() {
  // The promotion's premise is the whole prefill contract, not the clip half of
  // it. A caller can clip exactly where the engine asked and still hand the
  // decoder a prefix of its own -- and then nothing forces the hypothesis to
  // BEGIN with the holdback, so it is free to put a word in front of that
  // reproduction, over the very span the pending word occupies. (Here " S" runs
  // 1.0..1.5 and the clip is at 1.0, so its audio is inside that window: the
  // clip alone excludes nothing.)
  //
  // Mutation proof, both faces: promote on the clip clause alone
  // (`decoded_under.clip_timestamps_slice() == [self.last_agreed_seconds]`,
  // threaded in beside `decoded_under`) and the state reads back
  // ([" A", " S"], []) and the transcript " A S Y B C".
  let a = || word(" A", 0.0, 1.0);
  let s = || special_only_word(" S", 1.0, 1.5);
  let b = || word(" B", 1.0, 1.5);
  let y = || word(" Y", 1.0, 1.5);
  let c = || word(" C", 1.5, 2.0);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![a(), s(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A"], vec![" S"]),
    "non-vacuous: there is a pending word for the retarget to settle or not",
  );

  // The engine's own clip, with someone else's prefix.
  let honest = agreement.decoding_options_for_next(&DecodingOptions::new());
  let foreign = honest.clone().with_prefix_tokens(vec![9_001, 9_002]);
  assert_eq!(
    foreign.clip_timestamps_slice(),
    [agreement.last_agreed_seconds()],
    "non-vacuous: the clip clause alone IS satisfied",
  );
  assert!(
    !agreement.prefill_reproduces_holdback(&foreign),
    "and the contract as a whole is not",
  );
  assert!(
    s().start() < agreement.last_agreed_seconds() + 0.001
      && s().end() > agreement.last_agreed_seconds(),
    "non-vacuous: the pending word's audio straddles the clip point, so the \
     clip excludes none of it",
  );

  assert!(
    agreement
      .ingest(result_with_words(vec![y(), b(), c()]), &foreign)
      .is_awaiting_agreement()
  );
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A"], vec![" S"]),
    "the retarget settled nothing -- the word is still revisable, and the \
     disagreement leaves `finalize` to say what became of it",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A Y B C",
    "and there it is superseded by the hypothesis that re-covered its span",
  );
}

#[test]
fn a_late_clipped_disagreement_does_not_supersede_the_span_it_never_decoded() {
  // #94 (codex round 13, finding 2). THE ENGINE ASSUMED EVERY RESULT SAW THE
  // AUDIO IT WAS JUDGING. `finalize`'s `holdback_superseded` branch replaces the
  // engine's own provisional record of the still-open span -- `pending_words`
  // then `last_agreed_words` -- with the final hypothesis's own post-watermark
  // words, on the premise that hypothesis RE-COVERED that span carrying a
  // revision. For a marked continuation the clip guarantees it; for an unmarked
  // one it was simply assumed, and `ingest` documents no requirement that an
  // unmarked result's window reach the watermark at all.
  //
  // A window that begins AFTER the words it supersedes falsifies it outright.
  // " S" and " B" both run 1.0..1.5 and this decode's `clip_timestamps` begin at
  // 1.5, so it shares no instant with either: it produced no revision of them,
  // it produced nothing over their span at all, and the branch's whole
  // justification -- prefer the revision to the reading it replaced -- has no
  // referent. Dropping them deletes two words the stream agreed on and nothing
  // contradicted.
  //
  // The fact was already in hand: `decoded_under` carries `clip_timestamps`, and
  // `ProvenancedResult` already pairs a result with what was known when it
  // arrived, so the window travels with the result exactly as its prefill
  // premise does (`ProvenancedResult::window`).
  //
  // Mutation proof, the finalized face: make `finalize` drop the whole record
  // regardless (pass `0` where it passes `open_record_split`) and the first row
  // reads back " A C" -- " S" and " B" gone. The second row is the falsifier for
  // how the declared window is READ: `clip_timestamps` holds `(start, end)`
  // pairs, and reading only the last point as a start (`[2.0, ..)`) refuses a
  // window that plainly covered the span.
  let a = || word(" A", 0.0, 1.0);
  let s = || special_only_word(" S", 1.0, 1.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 1.5, 2.0);

  let settled = || {
    let mut agreement = LocalAgreement::new();
    let opening = || result_with_words(vec![a(), s(), b()]);
    agreement.ingest_streamed(opening());
    assert!(agreement.ingest_streamed(opening()).is_advanced());
    assert_eq!(
      (confirmed_texts(&agreement), pending_texts(&agreement)),
      (vec![" A"], vec![" S"]),
      "non-vacuous: there is a pending word AND a holdback for the branch to \
       drop",
    );
    assert_eq!(agreement.last_agreed_seconds(), 1.0);
    agreement
  };

  for (clip, expected, why) in [
    // Honest options for a decoder that resumed at 1.5 -- STRICTLY past the
    // watermark, and at or past the end of every word the branch would drop.
    (
      vec![1.5],
      " A S B C",
      "a decode that never covered the span does not get to replace the only \
       estimate it has",
    ),
    // A whole `(start, end)` pair whose START precedes the record: this window
    // DID cover 1.0..1.5, so the branch fires exactly as it always did.
    (
      vec![0.5, 2.0],
      " A C",
      "a window that covered the span still supersedes it -- the guard is a \
       coverage test, not a blanket refusal of unmarked results",
    ),
  ] {
    let mut agreement = settled();
    let late = DecodingOptions::new().with_clip_timestamps(clip.clone());
    assert!(
      !agreement.prefill_reproduces_holdback(&late),
      "non-vacuous: unmarked, so nothing is promoted on arrival and the pending \
       word is still there to be dropped ({clip:?})",
    );
    assert!(
      agreement
        .ingest(result_with_words(vec![c()]), &late)
        .is_awaiting_agreement(),
      "it disagrees, so `holdback_superseded` fires ({clip:?})",
    );
    // The streaming face is unmoved on both rows -- the decision is
    // `finalize`'s alone, and this pins that the ingest above neither promoted
    // nor dropped anything.
    assert_eq!(
      (confirmed_texts(&agreement), pending_texts(&agreement)),
      (vec![" A"], vec![" S"]),
      "{clip:?}",
    );
    assert_eq!(
      agreement.finalize(&DecodingOptions::new()).text(),
      expected,
      "the finalized face, clipped at {clip:?}: {why}",
    );
  }
  assert!(
    s().end() <= 1.5 && b().end() <= 1.5,
    "non-vacuous: the refusing row's window shares NO instant with either word \
     it would supersede",
  );
}

#[test]
fn the_still_open_span_begins_where_the_engines_own_record_does() {
  // Round 13, finding 2's span, from both ends. `open_record_split` scans the
  // record the engine would DROP, and that record is `pending_words ++
  // last_agreed_words` -- so it begins at the pending head whenever anything is
  // pending, and at the holdback otherwise. Reading only one of the two is a
  // live hole in each direction, and neither is visible in the shapes above,
  // where the pending word and the holdback tie at the same start.
  //
  // The FIRST case is also round 14 finding 2's boundary seen from the pending
  // side: the window opens at 2.0, which is inside " S1" (1.5..3.0) and at the
  // head of " B" (2.0..3.0), so the record splits between them -- the pending
  // head is preserved because no clip re-read it whole, and the holdback is
  // superseded because one did. A verdict applied to the whole record cannot
  // express that in either direction.
  //
  // Mutation proof, one per case, each mutating `open_record` AND
  // `open_record_len` together -- moving only one of them leaves the split
  // arithmetic inconsistent and the mutation self-cancelling, which is how the
  // first attempt at this pair passed. Make the record the HOLDBACK alone and
  // the FIRST case reads back " A S0 C": " S1" is no longer in the record, so
  // nothing protects it and it goes with " B". Make it PENDING alone and the
  // SECOND case reads back " P D": nothing is pending, so the record is empty,
  // every word of it is vacuously re-read, and the holdback goes.

  // A pending word that starts STRICTLY before the holdback -- the shape
  // `the_split_settles_a_widened_word_the_span_can_never_reach` builds, where
  // the split widens past two words and only the straddling one stays pending.
  let a = || word(" A", 0.0, 1.0);
  let s0 = || special_only_word(" S0", 1.0, 2.0);
  let s1 = || special_only_word(" S1", 1.5, 3.0);
  let b = || word(" B", 2.0, 3.0);
  let c = || word(" C", 3.0, 4.0);

  let mut agreement = LocalAgreement::new().with_agreement_count_needed(3);
  let opening = || result_with_words(vec![a(), s0(), s1(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " S0"], vec![" S1"]),
    "non-vacuous: the record's head is PENDING, and it starts at 1.5 while the \
     holdback starts at 2.0",
  );
  assert_eq!(agreement.last_agreed_seconds(), 2.0);

  // Exactly at the watermark, which is exactly where the holdback begins -- so
  // measuring the holdback alone would call this covering. It is not: " S1"
  // runs from 1.5.
  let at_the_holdback = DecodingOptions::new().with_clip_timestamps(vec![2.0]);
  assert!(
    agreement
      .ingest(result_with_words(vec![c()]), &at_the_holdback)
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A S0 S1 C",
    "the pending head is part of the record and no clip re-read it whole, so it \
     survives a window that superseded the holdback behind it",
  );

  // The other end: nothing pending at all, so the holdback IS the record.
  let p = || word(" P", 0.0, 1.0);
  let q = || word(" Q", 1.0, 2.0);
  let r = || word(" R", 2.0, 3.0);
  let d = || word(" D", 3.5, 4.5);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![p(), q(), r()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" P"], Vec::<&str>::new()),
    "non-vacuous: NOTHING is pending, so the holdback is the whole record",
  );
  assert_eq!(agreement.last_agreed_seconds(), 1.0);

  let past_the_holdback = DecodingOptions::new().with_clip_timestamps(vec![3.5]);
  assert!(
    agreement
      .ingest(result_with_words(vec![d()]), &past_the_holdback)
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " P Q R D",
    "with nothing pending the holdback is still a record to protect, not an \
     absent one",
  );
}

#[test]
fn a_late_clipped_agreement_confirms_the_span_it_could_not_re_read() {
  // Round 13, finding 2's OTHER consumer, and the reason the coverage fact is a
  // property of the result rather than a clause bolted onto `finalize`. The
  // advance replaces the same provisional record -- `pending_words.clear()` and
  // `last_agreed_words = common[split..]` -- on the same unchecked premise, and
  // its own comment stated it outright ("this hypothesis saw its audio"). Two
  // consecutive late-clipped hypotheses agree on " C D" at 1.5 onwards; `common`
  // is their words, not a re-reading of 1.0..1.5, so replacing " S" and " B"
  // with it deletes them -- and here it deletes them from the STREAMING face
  // too, where nothing downstream can put them back.
  //
  // Confirming is what is left, and it is the same claim every other confirmed
  // word carries: two consecutive hypotheses agreed on them, and no hypothesis
  // that could see their audio has contradicted them since. Holding them instead
  // has no terminating condition -- the watermark has moved past them, so no
  // future result can be offered over their span at all.
  //
  // Mutation proof, both faces: drop the `covers` guard from the advance's
  // replacement and the streaming face reads back ([" A"], []) and the
  // transcript " A C D".
  let a = || word(" A", 0.0, 1.0);
  let s = || special_only_word(" S", 1.0, 1.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 1.5, 2.0);
  let d = || word(" D", 2.0, 2.5);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![a(), s(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A"], vec![" S"]),
    "non-vacuous: there is a pending word AND a holdback for the advance to \
     replace",
  );

  let late = DecodingOptions::new().with_clip_timestamps(vec![1.5]);
  let resumed = || result_with_words(vec![c(), d()]);
  assert!(
    agreement.ingest(resumed(), &late).is_awaiting_agreement(),
    "the first one has nothing to agree with over this span",
  );
  assert!(
    agreement.ingest(resumed(), &late).is_advanced(),
    "the second corroborates it: an advance whose `common` is entirely past 1.5",
  );

  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " S", " B"], Vec::<&str>::new()),
    "the streaming face: what the advance could not re-read, it confirms rather \
     than drops",
  );
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" C", " D"],
    "and the holdback is the agreeing pair's own words, as always",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A S B C D",
    "the finalized face: nothing the stream agreed on was lost to a window that \
     excluded it",
  );
}

#[test]
fn a_gap_between_clip_ranges_does_not_supersede_the_span_no_range_reached() {
  // #94 (codex round 14, finding 1). COVERAGE IS A SET OF INTERVALS, NOT A
  // SINGLE HALF-LINE. Round 13 recorded the window a result arrived under as its
  // FIRST `clip_timestamps` entry and read it as `[decoded_from, ..)`.
  // `clip_timestamps` is not a start: its own doc says "Explicit `(start, end)`-pair
  // timestamps ... to split the audio into segments before transcription", and
  // `WhisperKit::transcribe` hands it straight to `chunker::prepare_seek_clips`,
  // which pairs the points and decodes each pair as its own clip.
  //
  // `[0.0, 0.5, 3.0]` therefore decodes `[0.0, 0.5)` and `[3.0, end)` and NOTHING
  // between them. The still-open record here is " S1" (1.5..3.0) then " B"
  // (2.0..3.0), which sits entirely inside that gap -- yet the half-line reading
  // starts the window at 0.0 and calls the record covered, so a word from the
  // SECOND range supersedes a span the decoder demonstrably never saw. Two words
  // the stream agreed on, deleted for a revision that does not exist.
  // Mutation proof, one per row. Read the first clip point as a half-line
  // (`clip_timestamps.first()`, then `[start, ..)`, which is what round 13 did)
  // and ROW 1 reads back " A S0 C" -- " S1" and " B" gone. Make the trailing
  // comparison STRICT (`word.end() < end`) and ROW 2 reads back
  // " A S0 S1 B Y": a clip ending exactly where the record does stops covering
  // it. Test only the word's START against the clip end
  // (`start <= word.start() && word.start() <= end`) and ROW 3 reads back
  // " A S0 Y" -- half a word inside the clip counted as re-read. Full OVERLAP
  // (`start <= word.end() && word.start() <= end`) reds row 1 first, " A S0 C",
  // because the `[3.0, ..)` clip then touches both record words at the instant
  // 3.0, and reds five other tests besides.
  //
  // The three rows below also pin the two ENDS of a clip against each other. A
  // range is half-open and a word is covered by CONTAINMENT in ONE of them, so
  // the trailing comparison is non-strict at a clip that ends exactly where the
  // record does (row 2) and refuses a clip that ends one word short of it (row
  // 3) -- overlap would have accepted both, and a strict end would have refused
  // both. Rows 2 and 3 differ in NOTHING but that clip end.
  let a = || word(" A", 0.0, 1.0);
  let s0 = || special_only_word(" S0", 1.0, 2.0);
  let s1 = || special_only_word(" S1", 1.5, 3.0);
  let b = || word(" B", 2.0, 3.0);
  let c = || word(" C", 3.0, 4.0);
  let y = || word(" Y", 2.0, 2.5);

  let settled = || {
    let mut agreement = LocalAgreement::new().with_agreement_count_needed(3);
    let opening = || result_with_words(vec![a(), s0(), s1(), b()]);
    agreement.ingest_streamed(opening());
    assert!(agreement.ingest_streamed(opening()).is_advanced());
    assert_eq!(
      (confirmed_texts(&agreement), pending_texts(&agreement)),
      (vec![" A", " S0"], vec![" S1"]),
      "non-vacuous: the record spans both buckets -- a pending head and a \
       holdback",
    );
    assert_eq!(agreement.last_agreed_seconds(), 2.0);
    agreement
  };

  assert_eq!(
    crate::audio::whisper::audio::chunker::prepare_seek_clips(&[0.0, 0.5, 3.0], 16_000 * 10)
      .unwrap(),
    vec![(0, 8_000), (48_000, 160_000)],
    "non-vacuous: the crate's own range derivation reads these three points as \
     two disjoint clips, and 1.5..3.0 is in neither",
  );
  assert!(
    s1().start() >= 0.5 && b().end() <= 3.0,
    "non-vacuous: the whole record lies strictly inside the gap",
  );

  for (clip, hypothesis, expected, why) in [
    // The gap. Every word of the record is between the two clips, and the only
    // word offered comes out of the SECOND one.
    (
      vec![0.0, 0.5, 3.0],
      vec![c()],
      " A S0 S1 B C",
      "a word decoded in the range AFTER the gap supersedes nothing inside it",
    ),
    // One clip containing the whole record, ending EXACTLY where it ends. The
    // branch fires as it always did -- this is a coverage test, not a blanket
    // refusal of a `(start, end)` pair.
    (
      vec![0.0, 3.0],
      vec![y()],
      " A S0 Y",
      "a clip that ends exactly where the record does still decoded it whole",
    ),
    // The same clip half a second shorter, so it stops between the record's two
    // words' starts and their shared end. Neither word was decoded WHOLE, so
    // neither is superseded.
    (
      vec![0.0, 2.5],
      vec![y()],
      " A S0 S1 B Y",
      "a clip that only reaches PART of each held word re-read neither of them",
    ),
  ] {
    let mut agreement = settled();
    let clipped = DecodingOptions::new().with_clip_timestamps(clip.clone());
    assert!(
      !agreement.prefill_reproduces_holdback(&clipped),
      "non-vacuous: unmarked, so nothing is promoted on arrival ({clip:?})",
    );
    assert!(
      agreement
        .ingest(result_with_words(hypothesis), &clipped)
        .is_awaiting_agreement(),
      "it disagrees, so `holdback_superseded` fires ({clip:?})",
    );
    assert_eq!(
      (confirmed_texts(&agreement), pending_texts(&agreement)),
      (vec![" A", " S0"], vec![" S1"]),
      "the streaming face is unmoved -- the decision is `finalize`'s alone \
       ({clip:?})",
    );
    assert_eq!(
      agreement.finalize(&DecodingOptions::new()).text(),
      expected,
      "the finalized face, clipped at {clip:?}: {why}",
    );
  }
  assert!(
    y().start() >= 2.0 && y().end() <= 2.5,
    "non-vacuous: the offered word is inside BOTH of the last two rows' clips, \
     so the rows differ only in what the clip reached of the RECORD",
  );
}

#[test]
fn an_agreement_from_past_a_clip_gap_confirms_the_span_no_range_reached() {
  // #94 (codex round 14, finding 1), the advance's face. Same gapped schedule,
  // same record, and the same deletion reached without `finalize` at all: two
  // consecutive hypotheses whose every word comes from the `[3.0, end)` range
  // agree, and the advance installs `common` over `pending_words` and
  // `last_agreed_words`. `common` is not a re-reading of 1.5..3.0 -- no decode
  // in this schedule ever looked there -- so the record is deleted from the
  // STREAMING face, where nothing downstream can put it back.
  //
  // Mutation proof: read the first clip point as a half-line and the streaming
  // face reads back ([" A", " S0"], []) with the transcript " A S0 C E F". The
  // two halves of the confirmed prefix have their own falsifiers -- drop the
  // holdback half of `confirm_the_unread_prefix_and_drop_the_rest` and it reads
  // ([" A", " S0", " S1"], []); drop the pending half and it reads
  // ([" A", " S0", " B"], []), which also pins their ORDER.
  let a = || word(" A", 0.0, 1.0);
  let s0 = || special_only_word(" S0", 1.0, 2.0);
  let s1 = || special_only_word(" S1", 1.5, 3.0);
  let b = || word(" B", 2.0, 3.0);
  let c = || word(" C", 3.0, 4.0);
  let e = || word(" E", 4.0, 5.0);
  let f = || word(" F", 5.0, 6.0);

  let mut agreement = LocalAgreement::new().with_agreement_count_needed(3);
  let opening = || result_with_words(vec![a(), s0(), s1(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " S0"], vec![" S1"]),
    "non-vacuous: there is a pending word AND a holdback for the advance to \
     replace",
  );

  let gapped = DecodingOptions::new().with_clip_timestamps(vec![0.0, 0.5, 3.0]);
  let resumed = || result_with_words(vec![c(), e(), f()]);
  assert!(
    agreement.ingest(resumed(), &gapped).is_awaiting_agreement(),
    "the first one has nothing to agree with over this span",
  );
  assert!(
    agreement.ingest(resumed(), &gapped).is_advanced(),
    "the second corroborates it: an advance whose `common` is entirely past the \
     gap",
  );
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " S0", " S1", " B"], Vec::<&str>::new()),
    "the streaming face: what no clip in the schedule could re-read, the advance \
     confirms rather than drops",
  );
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" C", " E", " F"],
    "and the holdback is the agreeing pair's own words, as always",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A S0 S1 B C E F",
    "the finalized face: nothing the stream agreed on was lost to a schedule \
     that skipped it",
  );
}

#[test]
fn a_window_that_opens_inside_the_record_replaces_only_the_part_it_re_read() {
  // #94 (codex round 14, finding 2). ONE VERDICT FOR A WHOLE RECORD IS WRONG IN
  // BOTH DIRECTIONS. Round 13 asked whether the window reached the record's
  // FIRST word and applied that answer to every word in it. Here " Q" (1.0..2.0)
  // and " R" (2.0..3.0) are held and the continuation clips at exactly 2.0: " Q"
  // is outside its window and " R" is wholly inside it. The head-only reading
  // answers "not covered" and CONFIRMS both -- so " R" becomes irrevocable on the
  // streaming face at the very moment a hypothesis that did re-read it says it is
  // " X" instead, and the transcript carries the stale " R" beside its own
  // revision.
  //
  // That is the exact mirror of round 13's finding: there a late-clipped result
  // was allowed to supersede a record it never saw; here the same conservatism
  // confirms the portion it DID see and revise. The record splits at the
  // coverage boundary instead: the uncovered prefix is preserved (or, on the
  // advance, confirmed) and only the covered suffix is replaced.
  //
  // Mutation proof, four ways and both faces. Make the leading comparison STRICT
  // (`start < word.start()`) and the finalized row reads back " P Q R X D" --
  // the round-13 answer, and the reason that comparison cannot be tightened.
  // Make `finalize` drop the whole record (pass `0` where it passes
  // `open_record_split`) and it reads " P X D"; make the ADVANCE do the same and
  // the streaming face reads ([" P"], []). Read the first clip point as a
  // half-line, or test only the word's START, or scan FORWARD for the first
  // covered word (`position`) instead of back over the covered suffix, and the
  // last row reads back " P Y" -- " Q" and " R" both replaced, the second of
  // them from behind a word no clip reached.
  let p = || word(" P", 0.0, 1.0);
  let q = || word(" Q", 1.0, 2.0);
  let r = || word(" R", 2.0, 3.0);
  let x = || word(" X", 2.0, 3.0);
  let d = || word(" D", 3.0, 4.0);

  let settled = || {
    let mut agreement = LocalAgreement::new();
    let opening = || result_with_words(vec![p(), q(), r()]);
    agreement.ingest_streamed(opening());
    assert!(agreement.ingest_streamed(opening()).is_advanced());
    assert_eq!(
      (confirmed_texts(&agreement), pending_texts(&agreement)),
      (vec![" P"], Vec::<&str>::new()),
      "non-vacuous: TWO words are held, and the window below opens between them",
    );
    assert_eq!(agreement.last_agreed_seconds(), 1.0);
    agreement
  };

  // Exactly the second held word's start, which is strictly past the record's
  // own head. `" R"` is wholly inside `[2.0, ..)`; `" Q"` is wholly outside it.
  let inside = || DecodingOptions::new().with_clip_timestamps(vec![2.0]);
  assert!(
    q().start() < 2.0 && r().start() >= 2.0,
    "non-vacuous: the window opens strictly inside the record",
  );

  // The finalized face: one disagreeing hypothesis.
  let mut agreement = settled();
  assert!(
    agreement
      .ingest(result_with_words(vec![x(), d()]), &inside())
      .is_awaiting_agreement(),
    "it disagrees, so `holdback_superseded` fires",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " P Q X D",
    "the revised half is replaced by its revision and the unreachable half is \
     kept -- not both readings of 2.0..3.0 side by side",
  );

  // The streaming face: two agreeing hypotheses, so the advance decides.
  let mut agreement = settled();
  let resumed = || result_with_words(vec![x(), d()]);
  assert!(
    agreement
      .ingest(resumed(), &inside())
      .is_awaiting_agreement()
  );
  assert!(agreement.ingest(resumed(), &inside()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" P", " Q"], Vec::<&str>::new()),
    "the streaming face: only the half the pair could not re-read is made \
     irrevocable -- confirming \" R\" here would strand it against its own \
     revision",
  );
  assert_eq!(
    agreement
      .last_agreed_words_slice()
      .iter()
      .map(WordTiming::word)
      .collect::<Vec<_>>(),
    vec![" X", " D"],
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " P Q X D",
    "and the transcript agrees with the streaming face word for word",
  );

  // THE OTHER DIRECTION, and why the split is the start of the longest COVERED
  // SUFFIX rather than the first covered word. A clip that CLOSES inside the
  // record re-read " Q" and not " R", so the covered word is the one in FRONT.
  // Replacing from it would take " R" with it -- a deletion, from behind a word
  // the window never reached -- so the boundary moves past both and the whole
  // record is preserved. The cost is the erring-wide one this rule is drawn to
  // pay: " Q" reaches the transcript beside its own revision " Y". A repetition,
  // not a deletion, and the same direction the advance's `position` split takes
  // for the same reason.
  let closing_inside = DecodingOptions::new().with_clip_timestamps(vec![0.0, 2.0]);
  let y = || word(" Y", 1.0, 2.0);
  assert!(
    q().end() <= 2.0 && r().end() > 2.0,
    "non-vacuous: the clip contains the FIRST held word whole and the second \
     only in part",
  );
  let mut agreement = settled();
  assert!(
    agreement
      .ingest(result_with_words(vec![y()]), &closing_inside)
      .is_awaiting_agreement(),
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " P Q R Y",
    "a covered word in front of an uncovered one is kept, so the uncovered one \
     is never replaced out from behind it",
  );
}

#[test]
fn a_final_pair_that_agreed_past_an_unreachable_record_keeps_what_they_agreed_on() {
  // #94 (codex round 14). The record and the TAIL are two questions, and only
  // the record's half belongs to coverage. Round 13's `finalize` guard made
  // "this hypothesis re-read the record" decide both at once; round 14's split
  // answers the record's half by itself -- a record nothing re-read gets
  // `open_record_split == len` and is kept whole either way -- leaving the guard
  // deciding only whether the tail is `hypothesis_words` or Swift's
  // `findLongestDifferentSuffix(prevWords, hypothesisWords)`.
  //
  // Nothing in the suite could tell the two apart, and every shape that CAN
  // makes the subtraction wrong. Here two consecutive late-clipped hypotheses
  // both produce " M" at 3.0..4.0 -- a one-word common prefix, short of the
  // threshold of 2, so it disagrees and " M" is never confirmed. Swift subtracts
  // it on the premise that `lastAgreedWords` supplies it; `lastAgreedWords` is
  // " Q R" at 1.0..3.0, which has nothing to do with " M", so subtracting drops
  // a word BOTH hypotheses produced and nothing puts it back. That is
  // `a_disagreeing_final_pair_keeps_the_words_both_hypotheses_agreed_on`'s
  // defect reached with a NON-empty holdback.
  //
  // Mutation proof: restore the conjunct (`&& (record_len == 0 ||
  // open_record_split < record_len)`, with `record_len` re-derived from
  // `open_record_len()`) and this reads back " P Q R Z" -- " M" gone.
  let p = || word(" P", 0.0, 1.0);
  let q = || word(" Q", 1.0, 2.0);
  let r = || word(" R", 2.0, 3.0);
  let m = || word(" M", 3.0, 4.0);
  let n = || word(" N", 4.0, 5.0);
  let z = || word(" Z", 4.0, 5.0);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![p(), q(), r()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" P"], Vec::<&str>::new()),
    "non-vacuous: the holdback is NON-empty, which is what separates this from \
     the empty-holdback face of the same defect",
  );

  // Strictly past every word of the record, so nothing in it is re-read and the
  // record survives whichever tail is taken.
  let late = DecodingOptions::new().with_clip_timestamps(vec![3.0]);
  assert!(
    agreement
      .ingest(result_with_words(vec![m(), n()]), &late)
      .is_awaiting_agreement()
  );
  assert!(
    agreement
      .ingest(result_with_words(vec![m(), z()]), &late)
      .is_awaiting_agreement(),
    "a ONE-word common prefix is short of the threshold, so this disagrees too",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " P Q R M Z",
    "the unreachable record is kept whole AND the word both hypotheses produced \
     survives -- the subtraction has no holdback to justify it here",
  );
}

#[test]
fn the_split_settles_a_widened_word_the_span_can_never_reach() {
  // The line the advance's second split draws, asserted from both sides in one
  // stream. `budgeted_split` widens past BOTH " S" words here; the still-open
  // span begins at " B"'s start, and only the word whose extent crosses that
  // point is still in play. The other one is settled outright -- every word the
  // engine will ever be offered again starts at or after the watermark, so
  // nothing can overlap it, and no provenance can change that.
  //
  // Deferring it anyway would be strictly worse than the round-8 behaviour it
  // replaces: `finalize`'s superseded branch would drop a word the stream
  // produced and nothing revised.
  //
  // Mutation proof: relax the advance's `word.end() > self.last_agreed_seconds`
  // to `>=` and the first assertion reads back ([" A"], [" S0", " S1"]); test
  // `word.start() >= self.last_agreed_seconds` instead -- the plausible mistake,
  // since `offered` is filtered on `start` -- and it reads
  // ([" A", " S0", " S1"], []), which the second half then falsifies too: the
  // unmarked revision lands beside " S1" rather than replacing it,
  // " A S0 S1 Y B C".
  //
  // `agreement_count_needed` is 3 so that BOTH " S" words fall inside
  // `common[requested..split]`: at the default 2 the first of them would be
  // confirmed by `common[..requested]` regardless, and the boundary under test
  // would decide nothing.
  let a = || word(" A", 0.0, 1.0);
  // Ends exactly where the still-open span begins: unreachable from inside it.
  let s0 = || special_only_word(" S0", 1.0, 2.0);
  // STRADDLES that point -- starts before it, ends after it -- so an offered
  // word can overlap it and it is still in play.
  let s1 = || special_only_word(" S1", 1.5, 3.0);
  let b = || word(" B", 2.0, 3.0);
  let y = || word(" Y", 2.0, 3.0);
  let c = || word(" C", 3.0, 4.0);

  let mut agreement = LocalAgreement::new().with_agreement_count_needed(3);
  let opening = || result_with_words(vec![a(), s0(), s1(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(agreement.last_agreed_seconds(), 2.0);
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A", " S0"], vec![" S1"]),
    "the widened-past words split again, on whether the still-open span can \
     reach them",
  );

  let unmarked = DecodingOptions::new();
  assert!(
    agreement
      .ingest(result_with_words(vec![y(), b(), c()]), &unmarked)
      .is_awaiting_agreement()
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A S0 Y B C",
    "the unreachable word survives the revision; the reachable one is replaced \
     by it",
  );
}

#[test]
fn the_holdback_filter_reads_the_configured_special_range_not_the_floor() {
  // #94 (codex round 12, finding 2). THE FLOOR IS AN ASSUMPTION, NOT AN
  // INVARIANT. `MIN_SPECIAL_TOKEN_BEGIN` is correct for the 50256/50257
  // families, but `WhisperTokenizer::from_folder` loads any parseable
  // `tokenizer.json` and probes `<|endoftext|>` for that artifact's OWN
  // threshold, rejecting nothing below the floor. For such an artifact the
  // engine calls an id carriable that `prefill_tokens` erases, the equality
  // `prefill_reproduces_holdback` checks is satisfied by a prefix the decoder is
  // given only part of, and round 8's defect is back with no caller lying.
  //
  // Threading the real value closes it where the value is known and costs
  // nothing where it is not: the default is exactly the floor, so no existing
  // caller moves, and `LocalAgreementTranscriber` sets it from the very
  // tokenizer whose `prefill_tokens` will apply the filter (see
  // `the_driver_takes_its_special_range_from_the_loaded_vocabulary`). Rejecting
  // the artifact at load instead would refuse a vocabulary this crate otherwise
  // decodes correctly, over a premise only the streaming engine has a stake in.
  //
  // Mutation proof, both faces: read `MIN_SPECIAL_TOKEN_BEGIN` in
  // `budgeted_split` instead of the threshold it is given and the configured
  // engine's holdback reads back [" S", " B"] carrying id 40005 -- an id the
  // artifact's own filter erases.
  const LOW: u32 = 40_000;
  let a = || word(" A", 0.0, 0.5);
  // Below the floor, so carriable under the default; at or above LOW, so not
  // carriable for an artifact that reserves ids from there.
  let s = || WordTiming::new(" S", vec![LOW + 5], 0.5, 1.0, 0.9);
  let b = || word(" B", 1.0, 1.5);
  assert!(
    s()
      .tokens_slice()
      .iter()
      .all(|&id| id < MIN_SPECIAL_TOKEN_BEGIN),
    "non-vacuous: the floor calls every one of these ids carriable",
  );

  let run = |engine: LocalAgreement| {
    let mut engine = engine;
    let opening = || result_with_words(vec![a(), s(), b()]);
    engine.ingest_streamed(opening());
    assert!(engine.ingest_streamed(opening()).is_advanced());
    let holdback: Vec<String> = engine
      .last_agreed_words_slice()
      .iter()
      .map(|word| word.word().to_string())
      .collect();
    let prefill = engine
      .decoding_options_for_next(&DecodingOptions::new())
      .prefix_tokens_slice()
      .to_vec();
    (holdback, prefill)
  };

  // The default engine: the floor, and the artifact's filter erases what it
  // issues.
  let engine = LocalAgreement::new();
  assert_eq!(engine.special_token_begin(), MIN_SPECIAL_TOKEN_BEGIN);
  let (holdback, prefill) = run(engine);
  assert_eq!(holdback, vec![" S".to_string(), " B".to_string()]);
  assert!(
    prefill.iter().any(|&id| id >= LOW),
    "non-vacuous: this is the defect -- the issued prefix {prefill:?} carries \
     an id a vocabulary reserving from {LOW} drops",
  );

  // Told the artifact's own threshold: the same word is widened past instead,
  // and every id the engine issues survives that vocabulary's filter.
  let engine = LocalAgreement::new().with_special_token_begin(LOW);
  assert_eq!(engine.special_token_begin(), LOW);
  let (holdback, prefill) = run(engine);
  assert_eq!(
    holdback,
    vec![" B".to_string()],
    "the holdback keeps only what THAT vocabulary's prefill carries whole",
  );
  assert!(
    !prefill.is_empty() && prefill.iter().all(|&id| id < LOW),
    "and the issued prefix {prefill:?} survives its filter intact",
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn the_driver_takes_its_special_range_from_the_loaded_vocabulary() {
  // The path that needs nothing remembered: the driver holds the very tokenizer
  // whose `prefill_tokens` applies the filter `budgeted_split` is standing in
  // for, so it hands the engine the exact threshold instead of the floor. Read
  // off the real artifact rather than argued from the constant.
  //
  // Mutation proof: drop the `with_special_token_begin(...)` from
  // `LocalAgreementTranscriber::new` and this reads back
  // `MIN_SPECIAL_TOKEN_BEGIN`.
  let tokenizer = tiny_tokenizer();
  let vocabulary = tokenizer.special_tokens().special_token_begin();
  assert_ne!(
    vocabulary, MIN_SPECIAL_TOKEN_BEGIN,
    "non-vacuous: the shipped vocabulary's threshold is not the floor, so the \
     two answers are distinguishable",
  );

  let kit = crate::audio::whisper::transcribe::WhisperKit::with_backend(
    crate::audio::whisper::backend::mock::MockBackend::new(),
    tokenizer,
  );
  let streamer = kit.local_agreement_transcriber(DecodingOptions::new());
  assert_eq!(
    streamer.agreement().special_token_begin(),
    vocabulary,
    "the driver's engine reads the loaded vocabulary, not the floor",
  );
}

#[test]
fn an_unmarked_hypothesis_may_re_agree_a_pending_word_because_it_is_not_confirmed() {
  // Where the pending words sit in the ALIGNMENT, pinned by the case that can
  // tell. They are not confirmed, so they belong to `settled_seams`' revisable
  // half beside the holdback: an offered word that re-covers one is a
  // reproduction the engine WANTS -- a second hypothesis agreeing over that span
  // is how the word earns its place. Filing them in the settled half instead
  // would refuse exactly that reproduction, which is the right treatment for a
  // word already in the caller's hands and the wrong one for a word that is not.
  //
  // Mutation proof: put `pending` in `watermark_filtered_with`'s settled half
  // (concatenate it onto `confirmed` and leave `revisable` as the holdback) and
  // both faces red -- the outcome becomes `awaiting_agreement`, the state
  // ([" A"], [" S"]) survives for the wrong reason, and the transcript loses
  // " S" entirely (" A B C"). Separately, drop `self.pending_words.clear()` from
  // the advance and the state reads back ([" A"], [" S", " S"]).
  let a = || word(" A", 0.0, 1.0);
  let s = || special_only_word(" S", 1.0, 1.5);
  let b = || word(" B", 1.0, 1.5);
  let c = || word(" C", 1.5, 2.0);

  let mut agreement = LocalAgreement::new();
  let opening = || result_with_words(vec![a(), s(), b()]);
  agreement.ingest_streamed(opening());
  assert!(agreement.ingest_streamed(opening()).is_advanced());
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A"], vec![" S"]),
    "non-vacuous: there is a pending word for the re-offer to re-agree",
  );

  // Decoded some other way, and reproducing the pending word rather than
  // revising it.
  let unmarked = DecodingOptions::new();
  assert!(
    agreement
      .ingest(result_with_words(vec![s(), b(), c()]), &unmarked)
      .is_advanced(),
    "the re-offer agrees with the hypothesis that produced the pending word",
  );
  assert_eq!(
    (confirmed_texts(&agreement), pending_texts(&agreement)),
    (vec![" A"], vec![" S"]),
    "and the re-agreement leaves it pending, not doubled and not dropped: it is \
     still a word the prefill cannot carry",
  );
  assert_eq!(
    agreement.finalize(&DecodingOptions::new()).text(),
    " A S B C",
  );
}
