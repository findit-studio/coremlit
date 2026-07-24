use std::num::NonZeroUsize;

use super::*;
use crate::audio::whisper::{
  audio::{
    chunker::{AudioChunk, VadChunker, prepare_seek_clips},
    vad::{EnergyVad, VoiceActivityDetector},
  },
  options::{ChunkingStrategy, Task, WordGrouping},
  result::TranscriptionTimings,
  task_facts::{SpanKnowledge, TaskFacts},
};

// ---------------------------------------------------------------------
// The completeness table (coremlit issue #14, codex round 1 / finding 1)
// ---------------------------------------------------------------------

/// Every knob of [`DecodingOptions`], paired with a mutation that moves it
/// **off** [`DecodingOptions::new`]'s value.
///
/// This is the fixture behind the two tests that make provenance capture
/// *provably* complete rather than complete-by-assertion:
///
/// - `provenance_records_every_decoding_option` applies each mutation and
///   demands the resulting [`Provenance`] differ from the baseline's. A knob
///   the record does not carry produces a byte-identical record, and its row
///   fails by name.
/// - `mutation_table_covers_every_decoding_option` checks this list against
///   [`DecodingOptions`]' **own serialized key set**, so the list cannot
///   silently fall behind the type. Add a 31st option and that test fails
///   until a row for it appears here — which is exactly the guard the
///   previous, hand-curated shape lacked: it asserted only over the fields
///   already present in the record, so it passed while the record was
///   missing 19 of 30 knobs, `drop_blank_audio` and `word_grouping` (the two
///   this branch itself added) among them.
///
/// The mutations are deliberately *value* changes, not merely
/// presence changes, so `Provenance`'s `PartialEq` has something to see.
///
/// One row per [`DecodingOptions`] field: its serde key, and a mutation
/// moving it off the default.
type OptionMutation = (&'static str, fn(DecodingOptions) -> DecodingOptions);

fn mutations() -> Vec<OptionMutation> {
  vec![
    ("task", |o| o.with_task(Task::Translate)),
    ("language", |o| o.with_language("es")),
    ("temperature", |o| o.with_temperature(0.4)),
    ("temperature_increment_on_fallback", |o| {
      o.with_temperature_increment_on_fallback(0.3)
    }),
    ("temperature_fallback_count", |o| {
      o.with_temperature_fallback_count(9)
    }),
    ("sample_length", |o| o.with_sample_length(64)),
    ("top_k", |o| o.with_top_k(11)),
    ("seed", |o| o.with_seed(42)),
    ("use_prefill_prompt", |o| o.maybe_use_prefill_prompt(false)),
    ("detect_language", |o| o.maybe_detect_language(true)),
    ("skip_special_tokens", |o| o.with_skip_special_tokens()),
    ("without_timestamps", |o| o.with_without_timestamps()),
    ("word_timestamps", |o| o.with_word_timestamps()),
    ("max_initial_timestamp", |o| {
      o.with_max_initial_timestamp(2.5)
    }),
    ("max_window_seek", |o| o.with_max_window_seek(1_234)),
    ("clip_timestamps", |o| {
      o.with_clip_timestamps(vec![0.5, 3.0])
    }),
    ("window_clip_time", |o| o.with_window_clip_time(2.0)),
    ("prompt_tokens", |o| {
      o.with_prompt_tokens(vec![101_u32, 102])
    }),
    ("prefix_tokens", |o| o.with_prefix_tokens(vec![201_u32])),
    ("suppress_blank", |o| o.with_suppress_blank()),
    ("suppress_tokens", |o| o.with_suppress_tokens(vec![301_u32])),
    ("compression_ratio_threshold", |o| {
      // `None` (the check DISABLED), not another number: this is the
      // mutation that catches a lossy `Option` encoding, since the four
      // thresholds are the only knobs whose default is `Some(_)`.
      o.maybe_compression_ratio_threshold(None)
    }),
    ("logprob_threshold", |o| o.maybe_logprob_threshold(None)),
    ("first_token_logprob_threshold", |o| {
      o.maybe_first_token_logprob_threshold(None)
    }),
    ("no_speech_threshold", |o| o.maybe_no_speech_threshold(None)),
    ("concurrent_worker_count", |o| {
      o.with_concurrent_worker_count(NonZeroUsize::new(3).unwrap())
    }),
    ("chunking_strategy", |o| {
      o.with_chunking_strategy(ChunkingStrategy::Vad)
    }),
    ("verbose", |o| o.with_verbose()),
    // The two knobs THIS branch added, and the two the old projection
    // silently dropped — pinned by name below as well as by the table.
    ("drop_blank_audio", |o| o.maybe_drop_blank_audio(false)),
    ("word_grouping", |o| {
      // SwiftParity is the default after #41, so mutate to FineGrained to
      // keep this row an actual value change (else the completeness gate below
      // sees a byte-identical record and rightly fails).
      o.with_word_grouping(WordGrouping::FineGrained)
    }),
  ]
}

#[test]
fn provenance_records_every_decoding_option() {
  // THE completeness gate. For each knob in turn: change only that knob,
  // and the record must change with it. A knob the capture drops produces a
  // record byte-identical to the baseline's, and that is precisely the
  // defect this test exists to make impossible — two runs whose transcripts
  // differ, described by the same record.
  let compute = ComputeOptions::new();
  let baseline = DecodingOptions::new();
  let baseline_record = Provenance::from_options(&baseline, &compute, 0.0, false);

  for (field, mutate) in mutations() {
    let mutated = mutate(baseline.clone());
    assert_ne!(
      mutated, baseline,
      "`{field}`'s row does not actually change the options, so its assertion \
       below would prove nothing"
    );

    let record = Provenance::from_options(&mutated, &compute, 0.0, false);
    assert_ne!(
      record, baseline_record,
      "`{field}` is NOT represented in the provenance record: two runs \
       differing only in it leave byte-identical records"
    );
    // ... and not merely "different": the record holds the exact options.
    assert_eq!(
      record.decoding(),
      &mutated,
      "`{field}` must be captured verbatim, not approximated"
    );
  }
}

#[test]
fn drop_blank_audio_and_word_grouping_are_recorded() {
  // The two knobs this branch added, pinned by name and not only by the
  // table above — they are the ones the old 11-of-30 projection dropped,
  // and both visibly change the transcript.
  let compute = ComputeOptions::new();
  let record =
    |decoding: &DecodingOptions| Provenance::from_options(decoding, &compute, 0.0, false);

  // `drop_blank_audio`: `true` yields "Hello World" / segment ids [0, 2];
  // `false` yields three segments including `[BLANK_AUDIO]`
  // (`transcribe::tests`' speech/blank/speech script). Two different
  // transcripts — so two different records.
  let dropping = DecodingOptions::new();
  let emitting = DecodingOptions::new().maybe_drop_blank_audio(false);
  assert!(dropping.drop_blank_audio(), "dropping is the default");
  assert_ne!(
    record(&dropping),
    record(&emitting),
    "drop_blank_audio must be legible in the record"
  );
  assert!(record(&dropping).decoding().drop_blank_audio());
  assert!(!record(&emitting).decoding().drop_blank_audio());

  // `word_grouping`: FineGrained vs SwiftParity carve a CJK segment's words
  // differently. Same story — but after #41 SwiftParity is the default, so
  // `fine` is now the explicit opt-in and `swift` is `new()`.
  let fine = DecodingOptions::new().with_word_grouping(WordGrouping::FineGrained);
  let swift = DecodingOptions::new();
  assert_eq!(fine.word_grouping(), WordGrouping::FineGrained);
  assert_eq!(
    swift.word_grouping(),
    WordGrouping::SwiftParity,
    "swift-parity is the #41 default"
  );
  assert_ne!(
    record(&fine),
    record(&swift),
    "word_grouping must be legible in the record"
  );
  assert_eq!(
    record(&fine).decoding().word_grouping(),
    WordGrouping::FineGrained
  );
}

#[test]
fn mutation_table_covers_every_decoding_option() {
  // NON-CIRCULAR by construction (codex round 3, F7). The field roster comes
  // from `DecodingOptions` ITSELF -- `options::DECODING_OPTION_FIELD_NAMES`,
  // backed by a compile-time exhaustive destructure (no `..`) in the `options`
  // module -- NOT from `serde_json::to_value(fully_populated())`, which folded
  // the very `mutations()` table this test verifies. Under the old anchor a new
  // field whose default is skip-serialized was absent from BOTH sides and
  // slipped through all three table-driven tests. Now adding a field breaks the
  // destructure's compilation until it is named in the roster, and then fails
  // HERE as an uncovered name until it also gets a mutation row.
  let expected: std::collections::BTreeSet<&str> =
    crate::audio::whisper::options::DECODING_OPTION_FIELD_NAMES
      .iter()
      .copied()
      .collect();
  let covered: std::collections::BTreeSet<&str> =
    mutations().iter().map(|(field, _)| *field).collect();

  assert_eq!(
    covered, expected,
    "the provenance mutation table has fallen out of step with \
     DecodingOptions -- every knob needs a row (see `mutations`)"
  );
}

// ---------------------------------------------------------------------
// The task-fact completeness table (coremlit issue #14, codex round 5)
// ---------------------------------------------------------------------

/// One TASK-LEVEL fact `Provenance::for_result` reads off the transcript,
/// paired with a mutation that moves it off a baseline result. This is the
/// analogue of the `DecodingOptions` `mutations()` table above, for the facts a
/// RUN controls rather than the options configure: the whole
/// [`TaskFacts`](crate::audio::whisper::task_facts::TaskFacts) sub-record (observed language,
/// RNG draw, early stop, worker schedule, id span) plus the one derived outcome
/// that lives on `Provenance` beside it, the effective temperature.
type TaskFactMutation = (&'static str, fn(TranscriptionResult) -> TranscriptionResult);

/// A baseline transcript with every task fact at its `new` default: one
/// segment at temperature 0.0 (so `effective_temperature` is `Some(0.0)`) and
/// [`TaskFacts::unknown`] — no observation, greedy, not truncated, an unknown
/// worker schedule, no id span. Each mutation moves exactly one.
fn baseline_task_result() -> TranscriptionResult {
  TranscriptionResult::new(
    "x",
    vec![TranscriptionSegment::new().with_temperature(0.0)],
    "en",
    TranscriptionTimings::new(),
  )
}

/// The derived outcome fact `for_result` computes rather than reading off the
/// carried record — kept separate from the [`TaskFacts`](crate::audio::whisper::task_facts::TaskFacts)
/// sub-facts in the coverage arithmetic below.
const DERIVED_TASK_FACT: &str = "effective_temperature";

fn task_fact_mutations() -> Vec<TaskFactMutation> {
  vec![
    // The five carried `TaskFacts` sub-facts, keyed by their record field
    // names, each set on an otherwise-unknown record.
    ("observed_language", |r| {
      r.with_task_facts(TaskFacts::unknown().with_observed_language(Some("es".to_string())))
    }),
    ("drew_from_rng", |r| {
      r.with_task_facts(TaskFacts::unknown().with_drew_from_rng(true))
    }),
    ("early_stopped", |r| {
      r.with_task_facts(TaskFacts::unknown().with_early_stopped(true))
    }),
    ("had_swallowed_error", |r| {
      r.with_task_facts(TaskFacts::unknown().with_had_swallowed_error(true))
    }),
    ("worker_schedule", |r| {
      r.with_task_facts(TaskFacts::unknown().with_worker(3))
    }),
    ("decoded_span", |r| {
      r.with_task_facts(TaskFacts::unknown().with_decoded_span(SpanKnowledge::Exact(1)))
    }),
    // The derived one: moving a segment temperature changes the unanimous
    // effective temperature `for_result` computes.
    (DERIVED_TASK_FACT, |mut r| {
      r.set_segments(vec![TranscriptionSegment::new().with_temperature(0.6)]);
      r
    }),
  ]
}

#[test]
fn provenance_records_every_task_fact() {
  // THE task-fact completeness gate, mirroring `provenance_records_every_decoding_option`
  // for the facts a RUN controls. Move one task fact off the baseline result and
  // the `for_result` record must change with it; a fact `for_result` fails to
  // read leaves a byte-identical record -- two runs whose transcripts differ,
  // described the same. This is exactly what catches a fact the consolidated
  // `TaskFacts` clone silently drops (the round-5/6 omissions of
  // early_stopped/worker_schedule/decoded_span among them).
  let decoding = DecodingOptions::new();
  let compute = ComputeOptions::new();
  let baseline = baseline_task_result();
  let baseline_record = Provenance::for_result(&decoding, &compute, &baseline);

  for (field, mutate) in task_fact_mutations() {
    let mutated = mutate(baseline.clone());
    assert_ne!(
      mutated, baseline,
      "`{field}`'s row does not actually change the result, so its assertion \
       below would prove nothing"
    );
    let record = Provenance::for_result(&decoding, &compute, &mutated);
    assert_ne!(
      record, baseline_record,
      "task fact `{field}` is NOT recorded in the provenance: two runs \
       differing only in it leave byte-identical records"
    );
  }
}

#[test]
fn task_fact_table_covers_every_provenance_task_fact() {
  // NON-CIRCULAR by construction, on TWO exhaustive rosters:
  //
  // 1. `Provenance`'s own field set (`PROVENANCE_FIELD_NAMES`, from the
  //    compile-time destructure) must partition into the non-task-facts (the
  //    embedded options + the consumer-supplied identity), the one DERIVED
  //    outcome, and the composite `task_facts` sub-record. Add a field to
  //    `Provenance` and this fails until it is placed.
  // 2. the composite is then covered field-by-field by `TaskFacts`' OWN roster
  //    (`TASK_FACTS_FIELD_NAMES`) against the mutation rows. Add a field to
  //    `TaskFacts` and its destructure forces it into that roster, and this
  //    fails HERE as an uncovered name until it also gets a mutation row above.
  const NON_TASK_FACTS: &[&str] = &[
    "decoding", // embedded DecodingOptions, covered by `mutations()`
    "compute",  // embedded ComputeOptions
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "vad_detector",
  ];

  // (1) Every Provenance field is accounted for.
  let provenance_task_layer: std::collections::BTreeSet<&str> =
    crate::audio::whisper::provenance::PROVENANCE_FIELD_NAMES
      .iter()
      .copied()
      .filter(|field| !NON_TASK_FACTS.contains(field))
      .collect();
  let provenance_expected: std::collections::BTreeSet<&str> =
    [DERIVED_TASK_FACT, "task_facts"].into_iter().collect();
  assert_eq!(
    provenance_task_layer, provenance_expected,
    "a Provenance field is neither a non-task-fact, the derived outcome, nor the \
     carried `task_facts` record -- place it in this test's partition"
  );

  // (2) Every carried TaskFacts sub-fact (plus the derived outcome) has a row.
  let expected: std::collections::BTreeSet<&str> =
    crate::audio::whisper::task_facts::TASK_FACTS_FIELD_NAMES
      .iter()
      .copied()
      .chain(std::iter::once(DERIVED_TASK_FACT))
      .collect();
  let covered: std::collections::BTreeSet<&str> = task_fact_mutations()
    .iter()
    .map(|(field, _)| *field)
    .collect();
  assert_eq!(
    covered, expected,
    "the provenance task-fact table has fallen out of step with TaskFacts -- \
     every carried sub-fact needs a mutation row (see `task_fact_mutations`)"
  );
}

#[cfg(feature = "serde")]
#[test]
fn every_decoding_option_survives_the_provenance_round_trip() {
  // Capture is only half of it: a record that cannot be READ BACK intact is
  // no more use than one that never held the field. This is the test that
  // catches a lossy `Option` encoding -- a DISABLED threshold that
  // serialized as "absent" would read back RE-ENABLED at its default, and
  // the record would then quietly assert a check the run never performed.
  let compute = ComputeOptions::new();
  let baseline = DecodingOptions::new();
  let baseline_json =
    serde_json::to_string(&Provenance::from_options(&baseline, &compute, 0.0, false))
      .expect("baseline serializes");

  for (field, mutate) in mutations() {
    let mutated = mutate(baseline.clone());
    let record = Provenance::from_options(&mutated, &compute, 0.0, false);

    let json = serde_json::to_string(&record).expect("record serializes");
    assert_ne!(
      json, baseline_json,
      "`{field}` leaves no trace in the SERIALIZED record"
    );
    assert_eq!(
      serde_json::from_str::<Provenance>(&json).expect("record deserializes"),
      record,
      "`{field}` does not survive the round trip"
    );
  }
}

// ---------------------------------------------------------------------
// Capture semantics
// ---------------------------------------------------------------------

/// A deliberately all-non-default decode configuration.
fn distinctive_decoding() -> DecodingOptions {
  DecodingOptions::new()
    .with_task(Task::Translate)
    .with_language("es")
    .with_skip_special_tokens()
    .with_word_timestamps()
    .with_chunking_strategy(ChunkingStrategy::Vad)
    .with_temperature(0.2)
    .with_temperature_increment_on_fallback(0.3)
    .with_temperature_fallback_count(9)
    .with_seed(42)
}

fn distinctive_compute() -> ComputeOptions {
  ComputeOptions::new()
    .with_mel(ComputeUnits::CpuOnly)
    .with_encoder(ComputeUnits::All)
    .with_decoder(ComputeUnits::CpuAndGpu)
}

#[test]
fn from_options_captures_the_options_and_invents_nothing_else() {
  let decoding = distinctive_decoding();
  let compute = distinctive_compute();
  let provenance = Provenance::from_options(&decoding, &compute, 0.6, false);

  // The decode configuration, whole and verbatim -- one assertion, because
  // there is one field. (`provenance_records_every_decoding_option` above is
  // what proves that field really carries all 30 knobs.)
  assert_eq!(provenance.decoding(), &decoding);
  assert_eq!(provenance.compute(), compute);
  assert_eq!(provenance.encoder_compute_units(), ComputeUnits::All);
  assert_eq!(provenance.effective_temperature(), Some(0.6));

  // Options alone cannot observe a DETECTION outcome, so this constructor
  // does not pretend to: `for_result` is the one that records it.
  assert_eq!(provenance.task_facts().observed_language(), None);

  // What the library genuinely cannot observe is never invented -- the
  // identity, and (though `chunking_strategy` above proves VAD ran) the
  // detector that drove it.
  assert_eq!(provenance.model_id(), None);
  assert_eq!(provenance.model_revision(), None);
  assert_eq!(provenance.tokenizer_id(), None);
  assert_eq!(provenance.tokenizer_revision(), None);
  assert_eq!(provenance.vad_detector(), None);
}

#[test]
fn detect_language_reads_back_resolved_not_raw() {
  // The record must answer with what the pipeline ACTED on, which is the
  // resolved `detect_language ?? !use_prefill_prompt` coupling. Embedding
  // the options keeps BOTH halves of that coupling in the record, so
  // `DecodingOptions::detect_language` -- the crate's single resolution
  // point -- re-resolves to exactly the value the decode used, and the raw
  // tri-state survives for a reader who wants to know the caller never
  // chose.
  let compute = ComputeOptions::new();

  let prefilled = DecodingOptions::new();
  assert!(prefilled.use_prefill_prompt());
  assert!(
    !Provenance::from_options(&prefilled, &compute, 0.0, false)
      .decoding()
      .detect_language()
  );

  let mut no_prefill = DecodingOptions::new();
  no_prefill.clear_use_prefill_prompt();
  let provenance = Provenance::from_options(&no_prefill, &compute, 0.0, false);
  assert!(
    provenance.decoding().detect_language(),
    "the resolved coupling, not the unset raw tri-state"
  );
  assert!(!provenance.decoding().use_prefill_prompt());
}

#[test]
fn for_segment_reads_the_effective_temperature_off_the_segment() {
  // The effective temperature is per-segment (the fallback ladder runs per
  // window), which is exactly what this constructor exists to capture: a
  // segment that climbed to 0.4 must not be recorded as the base 0.0.
  let decoding = DecodingOptions::new();
  let compute = ComputeOptions::new();
  let segment = TranscriptionSegment::new().with_temperature(0.4);

  let provenance = Provenance::for_segment(&decoding, &compute, &segment, false);
  assert_eq!(provenance.effective_temperature(), Some(0.4));
  assert_eq!(
    provenance.decoding().temperature(),
    0.0,
    "the BASE temperature stays what was configured"
  );
  assert_eq!(
    provenance,
    Provenance::from_options(&decoding, &compute, 0.4, false),
    "for_segment is from_options with the segment's temperature and draw fact"
  );
}

/// A result of `segments` decoded at the given temperatures, reporting
/// `language` as the DETECTED one.
fn result_at(language: &str, temperatures: &[f32]) -> TranscriptionResult {
  TranscriptionResult::new(
    "Hello world.",
    temperatures
      .iter()
      .map(|&t| TranscriptionSegment::new().with_temperature(t))
      .collect::<Vec<_>>(),
    language,
    TranscriptionTimings::new(),
  )
  // `new` no longer seeds the observation from the display language (F3, codex
  // round 3), so model a real decode that ran to completion: it genuinely
  // OBSERVED `language`, POSITIVELY observed it was NOT truncated
  // (`early_stopped = Some(false)`, the honest fact of a full run — F1, codex
  // round 6 post-consolidation), and — the COMMON case — drew from the RNG if
  // any window landed on a non-zero temperature. `for_result` reads THIS carried
  // record, never the segment temperatures (F3, codex round 4); the
  // zero-iteration EXCEPTION -- a non-zero-temperature segment that never drew --
  // is exercised on its own in
  // `for_result_reads_the_carried_flag_not_the_segment_temperature`.
  .with_task_facts(
    TaskFacts::unknown()
      .with_observed_language((!language.is_empty()).then(|| language.to_string()))
      .with_early_stopped(false)
      // Swallowed nothing (codex round 11, M2): a full decode that watched its
      // child steps and hid no error, the observed-clean fact a reproducible run
      // must positively carry alongside the not-truncated one.
      .with_had_swallowed_error(false)
      .with_drew_from_rng(temperatures.iter().any(|&t| t != 0.0)),
  )
}

#[test]
fn for_result_records_the_detected_language_not_the_configured_one() {
  // The whole point (issue #14 review, m4). Under the DEFAULT auto-detect
  // the configured language is "" — so a record built from options alone
  // names no language at all, which is precisely the fact issue #9 wanted
  // recorded. Only the result knows what was actually detected, and
  // `for_segment` structurally cannot reach it (it takes a segment).
  let decoding = DecodingOptions::new();
  let compute = ComputeOptions::new();
  assert_eq!(decoding.language(), "", "auto-detect is the default");

  let provenance = Provenance::for_result(&decoding, &compute, &result_at("es", &[0.0]));

  assert_eq!(
    provenance.decoding().language(),
    "",
    "the CONFIGURED language, verbatim"
  );
  assert_eq!(
    provenance.task_facts().observed_language(),
    Some("es"),
    "the DETECTED language — the fact the record exists to carry"
  );
  // Never inferred: without a result there is nothing to read it from.
  assert_eq!(
    Provenance::from_options(&decoding, &compute, 0.0, false)
      .task_facts()
      .observed_language(),
    None
  );
}

#[test]
fn for_result_detected_language_is_absent_when_no_window_observed_one() {
  // F3 (codex round 2). A run that decoded ZERO windows (audio too short)
  // observes no language, yet its result still carries the Swift-compat
  // `"en"` DISPLAY fallback on `TranscriptionResult::language`. for_result
  // must record `detected_language = None` -- reading the observation, not
  // promoting the display string it never witnessed.
  //
  // Mutation proof: revert for_result to `Some(result.language())` and the
  // first assertion (None) fails, reporting Some("en").
  let compute = ComputeOptions::new();
  let opts = DecodingOptions::new();

  // The pipeline's zero-window shape: display "en", observation None.
  let unobserved = TranscriptionResult::new("", Vec::new(), "en", TranscriptionTimings::new())
    .with_task_facts(TaskFacts::unknown().with_observed_language(None));
  assert_eq!(
    unobserved.language(),
    "en",
    "the Swift-compat display fallback is kept on the result"
  );
  assert_eq!(
    Provenance::for_result(&opts, &compute, &unobserved)
      .task_facts()
      .observed_language(),
    None,
    "nothing was observed -- the detected language is absent, not fabricated"
  );

  // The guard against a wrong 'empty result means absent' fix: a window DID
  // run and genuinely decoded English, then the blank-audio drop emptied its
  // segments. The observation survives even though no segment does.
  let dropped_english = TranscriptionResult::new("", Vec::new(), "en", TranscriptionTimings::new())
    .with_task_facts(TaskFacts::unknown().with_observed_language(Some("en".to_string())));
  assert!(dropped_english.segments_slice().is_empty());
  assert_eq!(
    Provenance::for_result(&opts, &compute, &dropped_english)
      .task_facts()
      .observed_language(),
    Some("en"),
    "a genuinely observed language must survive its segments being dropped"
  );
}

#[test]
fn for_result_temperature_is_some_only_when_every_segment_agrees() {
  // m3: a result-level `f32` would have had to misrepresent a per-window
  // fallback ladder that split the segments. `Option<f32>` is honest AND
  // usable: the overwhelmingly common no-fallback case still answers
  // `Some(0.0)`.
  let decoding = DecodingOptions::new();
  let compute = ComputeOptions::new();
  let for_result = |temperatures: &[f32]| {
    Provenance::for_result(&decoding, &compute, &result_at("en", temperatures))
      .effective_temperature()
  };

  assert_eq!(
    for_result(&[0.0, 0.0, 0.0]),
    Some(0.0),
    "no fallback anywhere"
  );
  assert_eq!(
    for_result(&[0.4, 0.4]),
    Some(0.4),
    "the same rung throughout"
  );
  assert_eq!(
    for_result(&[0.0, 0.2]),
    None,
    "the ladder split the segments: no single temperature describes this"
  );
  assert_eq!(
    for_result(&[]),
    None,
    "no segments (silence, once drop_blank_audio empties it) landed anywhere"
  );
}

#[test]
fn is_reproducible_requires_greedy_or_a_seed() {
  let compute = ComputeOptions::new();
  let unseeded = DecodingOptions::new();
  let seeded = DecodingOptions::new().with_seed(7);

  // The greedy/seed axis is exercised through `for_result`, which reads a
  // transcript's POSITIVELY observed facts — the only door that can answer
  // reproducible at all (F1, codex round 6 post-consolidation): `result_at`
  // carries an observed not-truncated (`early_stopped = Some(false)`), so the
  // draw and the seed are the only variables left.
  // A real drawing decode carries its worker coordinate, and a seed replays its
  // draw only WITH it (codex round 13, M2). `result_at` models the worker-less
  // shape the serde tests need, so layer a KNOWN coordinate on here — the seed rows
  // below are the reproducible-WITH-schedule case (the schedule-less case is
  // `task_facts::tests::seeded_draw_with_unknown_worker_schedule_is_not_reproducible`).
  let repro = |decoding: &DecodingOptions, temps: &[f32]| {
    let result = result_at("en", temps);
    let facts = result.task_facts().clone().with_worker(0);
    Provenance::for_result(decoding, &compute, &result.with_task_facts(facts)).is_reproducible()
  };

  // Greedy (0.0, never drew) -> deterministic, seed or not.
  assert!(repro(&unseeded, &[0.0]));
  assert!(repro(&seeded, &[0.0]));

  // Sampled (drew at 0.2): only a seed makes the draws replayable.
  assert!(
    !repro(&unseeded, &[0.2]),
    "a fallback climb with no seed is not reproducible"
  );
  assert!(repro(&seeded, &[0.2]));

  // A SPLIT ladder (None effective temperature) is conservatively not
  // reproducible without a seed: the rungs only ascend, so a split means at
  // least one segment climbed off 0.0 and sampled.
  assert!(!repro(&unseeded, &[0.0, 0.2]));
  assert!(repro(&seeded, &[0.0, 0.2]));

  // `from_options`, by contrast, can NEVER answer reproducible — it cannot see
  // whether a callback truncated the decode, so its `early_stopped` is an
  // explicit unknown and the predicate stays conservative, whatever the draw
  // fact or the seed. This is the F1 fix: the old constructor fabricated a
  // `not-truncated` and handed out a promise it could not back.
  assert!(!Provenance::from_options(&unseeded, &compute, 0.0, false).is_reproducible());
  assert!(
    !Provenance::from_options(&seeded, &compute, 0.0, false).is_reproducible(),
    "a seed cannot rescue what the constructor never observed — the truncation"
  );
  assert!(!Provenance::from_options(&seeded, &compute, 0.2, true).is_reproducible());
}

#[test]
fn identity_uses_the_full_option_vocabulary() {
  let base = Provenance::from_options(&DecodingOptions::new(), &ComputeOptions::new(), 0.0, false);

  let built = base
    .clone()
    .with_model_id("openai_whisper-tiny")
    .with_model_revision("a1b2c3d")
    .with_tokenizer_id("openai/whisper-tiny")
    .with_tokenizer_revision("deadbeef");
  assert_eq!(built.model_id(), Some("openai_whisper-tiny"));
  assert_eq!(built.model_revision(), Some("a1b2c3d"));
  assert_eq!(built.tokenizer_id(), Some("openai/whisper-tiny"));
  assert_eq!(built.tokenizer_revision(), Some("deadbeef"));

  let mut m = built.clone();
  m.clear_model_id().clear_tokenizer_revision();
  assert_eq!(m.model_id(), None);
  assert_eq!(m.tokenizer_revision(), None);
  m.set_model_id("other").update_tokenizer_id(None);
  assert_eq!(m.model_id(), Some("other"));
  assert_eq!(m.tokenizer_id(), None);

  let via_maybe = base.maybe_model_revision(Some("feedface".to_string()));
  assert_eq!(via_maybe.model_revision(), Some("feedface"));
}

#[test]
fn provenance_never_infers_the_vad_detector() {
  // The detector is consumer-supplied for the same reason the model and
  // tokenizer identity are: this crate cannot observe it. `WhisperKit`
  // holds it as a `Box<dyn VoiceActivityDetector>` — a trait object with
  // no identity to read — and it lives on the pipeline, not in
  // `DecodingOptions`/`ComputeOptions`, so no constructor here is ever
  // handed it. `chunking_strategy` records THAT VAD ran; nothing records
  // WHICH detector ran unless the caller says so.

  // A real detector, at `WhisperKit::set_vad_detector`'s own boxed type,
  // and a genuinely behavior-changing one: it never reports silence, so
  // the chunker finds no silence midpoint to cut at and falls through to
  // whole-window boundaries.
  struct AlwaysActiveVad;
  impl VoiceActivityDetector for AlwaysActiveVad {
    fn voice_activity(&self, samples: &[f32]) -> Vec<bool> {
      vec![true; samples.len().div_ceil(self.frame_length_samples())]
    }
    fn frame_length_samples(&self) -> usize {
      crate::audio::whisper::audio::vad::DEFAULT_FRAME_LENGTH_SAMPLES
    }
  }
  let installed: Box<dyn VoiceActivityDetector + Send + Sync> = Box::new(AlwaysActiveVad);

  // The premise, driven through the exact seam `WhisperKit::transcribe`
  // drives (`VadChunker::chunk_all(self.vad_detector.as_ref(), ..)`):
  // which detector is installed decides where the chunks fall, and the
  // chunk boundaries decide the text. Two silent stretches inside 96_000
  // samples, 48_000-sample windows.
  let mut audio = vec![0.1f32; 96_000];
  audio[32_000..35_200].fill(0.0);
  audio[64_000..67_200].fill(0.0);
  let clips = prepare_seek_clips(&[], audio.len()).unwrap();
  let boundaries = |vad: &(dyn VoiceActivityDetector + Send + Sync)| {
    VadChunker::new()
      .chunk_all(vad, &audio, 48_000, &clips)
      .iter()
      .map(AudioChunk::seek_offset)
      .collect::<Vec<_>>()
  };
  assert_ne!(
    boundaries(&EnergyVad::new()),
    boundaries(installed.as_ref()),
    "the swap must move the chunk boundaries, or the rest proves nothing"
  );

  // ... and yet the record cannot see any of it. Both runs are described
  // by the same options, and the options are ALL the constructor is given,
  // so the best a consumer can do for two runs whose transcripts differ is
  // two byte-identical records. That is the gap this field closes.
  let decoding = DecodingOptions::new().with_chunking_strategy(ChunkingStrategy::Vad);
  let compute = ComputeOptions::new();
  let result = result_at("en", &[0.0]);
  let default_run = Provenance::for_result(&decoding, &compute, &result);
  let installed_run = Provenance::for_result(&decoding, &compute, &result);
  assert_eq!(
    default_run, installed_run,
    "there is no constructor parameter the detector could arrive through"
  );

  // So it stays `None` — never a name derived from the concrete type, from
  // `type_name`, or from the bare fact that VAD chunking ran.
  assert_eq!(
    installed_run.decoding().chunking_strategy(),
    ChunkingStrategy::Vad
  );
  assert_eq!(
    installed_run.vad_detector(),
    None,
    "never inferred: not `AlwaysActiveVad`, not the default `EnergyVad`"
  );

  // Only the consumer — the one party that knows what it installed — can
  // close the gap, through the same option vocabulary the identity fields
  // use.
  let named =
    Provenance::for_result(&decoding, &compute, &result).with_vad_detector("AlwaysActiveVad");
  assert_eq!(named.vad_detector(), Some("AlwaysActiveVad"));
  assert_ne!(
    named, default_run,
    "supplied, the two runs are finally distinguishable"
  );
  assert_eq!(
    Provenance::for_result(&decoding, &compute, &result)
      .maybe_vad_detector(Some("AlwaysActiveVad".to_string())),
    named
  );

  let mut mutated =
    Provenance::for_result(&decoding, &compute, &result).with_vad_detector("AlwaysActiveVad");
  mutated.clear_vad_detector();
  assert_eq!(mutated.vad_detector(), None);
  mutated.set_vad_detector("EnergyVad");
  assert_eq!(mutated.vad_detector(), Some("EnergyVad"));
  mutated.update_vad_detector(None);
  assert_eq!(mutated.vad_detector(), None);

  // Unset is ABSENT on the wire, exactly like the identity pairs: an
  // unsupplied detector must never read back as a known `null`. Supplied,
  // it round-trips.
  #[cfg(feature = "serde")]
  {
    let unsupplied: serde_json::Value = serde_json::to_value(&installed_run).unwrap();
    assert!(
      !unsupplied.as_object().unwrap().contains_key("vad_detector"),
      "an unsupplied detector is absent, not null"
    );
    assert_eq!(
      serde_json::from_str::<Provenance>(&unsupplied.to_string()).unwrap(),
      installed_run,
      "and it reads back `None`, never a guess"
    );

    let supplied: serde_json::Value = serde_json::to_value(&named).unwrap();
    assert_eq!(supplied["vad_detector"], "AlwaysActiveVad");
    assert_eq!(
      serde_json::from_str::<Provenance>(&supplied.to_string()).unwrap(),
      named
    );
  }
}

// ---------------------------------------------------------------------
// serde
// ---------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_every_field() {
  let full = Provenance::from_options(&distinctive_decoding(), &distinctive_compute(), 0.6, false)
    .with_model_id("openai_whisper-tiny")
    .with_model_revision("a1b2c3d")
    .with_tokenizer_id("openai/whisper-tiny")
    .with_tokenizer_revision("deadbeef")
    .with_vad_detector("SileroVad");

  let json = serde_json::to_string(&full).unwrap();
  assert_eq!(serde_json::from_str::<Provenance>(&json).unwrap(), full);
}

#[cfg(feature = "serde")]
#[test]
fn unset_identity_serializes_as_absent_not_null() {
  // A provenance record must not claim a `null` model revision it never
  // knew: unset identity is ABSENT from the wire form entirely.
  let bare = Provenance::from_options(&DecodingOptions::new(), &ComputeOptions::new(), 0.0, false);
  let value: serde_json::Value = serde_json::to_value(&bare).unwrap();
  let object = value.as_object().unwrap();

  for absent in [
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "vad_detector",
  ] {
    assert!(
      !object.contains_key(absent),
      "unset `{absent}` must be absent, not null"
    );
  }
  // An unset `seed` keeps the same absent-not-null contract inside the
  // embedded options (`DecodingOptions`' own wire form owns that rule now).
  assert!(
    !object["decoding"].as_object().unwrap().contains_key("seed"),
    "an unset seed must be absent, not null"
  );

  // The library-known facts are always written, so a persisted record is
  // never silently missing the settings that produced it. The derived
  // `effective_temperature` and the carried `task_facts` record are written
  // even when their contents are `None` — as an explicit `null`, NOT omitted
  // like the identity above, because for them `None` is itself the fact ("the
  // ladder split the segments" / "no language observed" / "unknown worker"),
  // and a reader must be able to tell that from "the writer dropped the field".
  for present in ["decoding", "compute", "effective_temperature", "task_facts"] {
    assert!(object.contains_key(present), "`{present}` must be recorded");
  }
  // The explicit-unknown outcomes inside the carried record are present-null,
  // not omitted — including the WORKER SCHEDULE, which must never read back as a
  // fabricated `0` (R6-F2).
  let facts = object["task_facts"].as_object().unwrap();
  assert!(
    facts["observed_language"].is_null(),
    "an unobserved detection is an explicit null, not an omission"
  );
  assert!(
    facts["worker_schedule"].is_null(),
    "an unknown worker schedule is an explicit null, never a fabricated 0"
  );

  assert_eq!(
    serde_json::from_str::<Provenance>(&value.to_string()).unwrap(),
    bare
  );
}

#[cfg(feature = "serde")]
#[test]
fn a_record_missing_a_library_known_field_is_rejected() {
  // Deliberately NOT `serde(default)` on the captured facts: unlike a
  // config, where an omitted knob sensibly falls back to its default, a
  // provenance record that silently invented a whole missing `decoding`
  // block would be a lie about what actually ran. Only the
  // consumer-supplied identity may be omitted.
  //
  // The rule reaches INTO the carried `task_facts`: serde's derive silently
  // treats a missing `Option` field as `None` even with no `serde(default)` on
  // it, which for `observed_language`/`worker_schedule` would forge "no language
  // observed" / "unknown worker" (and, worse, a dropped `drew_from_rng` reads
  // back `false`) out of a field the writer merely dropped. Each names a
  // `deserialize_with` (`TaskFacts`' `required_option`) to defeat that path, and
  // `effective_temperature` the finite-float `with` helper (codex round 3, F6).
  let full = Provenance::for_result(
    &distinctive_decoding(),
    &distinctive_compute(),
    &result_at("es", &[0.6]),
  );
  let value: serde_json::Value = serde_json::to_value(&full).unwrap();
  assert_eq!(
    serde_json::from_str::<Provenance>(&value.to_string()).unwrap(),
    full,
    "the intact record must round-trip, or the removals below prove nothing"
  );

  // Top-level required fields: dropping any is rejected.
  for required in ["decoding", "compute", "effective_temperature", "task_facts"] {
    let mut without = value.clone();
    without.as_object_mut().unwrap().remove(required).unwrap();
    assert!(
      serde_json::from_str::<Provenance>(&without.to_string()).is_err(),
      "a missing `{required}` must fail, not default"
    );
  }

  // The carried record's OWN required facts, reached inside the nested object —
  // the WORKER SCHEDULE among them, so a dropped coordinate is rejected rather
  // than read back as a fabricated `0` (R6-F2), and a dropped `drew_from_rng` is
  // rejected rather than read back `false` (the optimistic direction that hands
  // `is_reproducible` a guarantee the run never earned).
  for required in [
    "drew_from_rng",
    "observed_language",
    "early_stopped",
    "worker_schedule",
  ] {
    let mut without = value.clone();
    without
      .as_object_mut()
      .unwrap()
      .get_mut("task_facts")
      .unwrap()
      .as_object_mut()
      .unwrap()
      .remove(required)
      .unwrap();
    assert!(
      serde_json::from_str::<Provenance>(&without.to_string()).is_err(),
      "a missing `task_facts.{required}` must fail, not default"
    );
  }

  // Present-but-null is the honest, ACCEPTED encoding of "no such fact": it is
  // the omission that is rejected, not the null. `effective_temperature` and the
  // carried explicit-unknown facts all read back their `None`.
  let mut nulled = value;
  nulled.as_object_mut().unwrap()["effective_temperature"] = serde_json::Value::Null;
  {
    let facts = nulled
      .as_object_mut()
      .unwrap()
      .get_mut("task_facts")
      .unwrap()
      .as_object_mut()
      .unwrap();
    facts["observed_language"] = serde_json::Value::Null;
    facts["worker_schedule"] = serde_json::Value::Null;
  }
  let read: Provenance = serde_json::from_str(&nulled.to_string()).unwrap();
  assert_eq!(read.effective_temperature(), None);
  assert_eq!(read.task_facts().observed_language(), None);
  assert_eq!(
    read.task_facts().worker_schedule(),
    None,
    "a null worker schedule reads back explicit unknown, never [0]"
  );
}

// ---------------------------------------------------------------------
// The unseeded-sampling invariant (codex round 1 / finding 2)
// ---------------------------------------------------------------------

#[test]
fn for_result_reads_the_carried_sampling_fact_not_the_surviving_segments() {
  // The heart of finding 2, at the `Provenance` boundary. A result whose
  // sampled window was FILTERED AWAY has no segment left to betray it: every
  // survivor reads 0.0, and `effective_temperature` honestly reports
  // `Some(0.0)`. Only the carried flag knows better -- and
  // `is_reproducible` must believe the flag, not the survivors.
  let compute = ComputeOptions::new();
  let unseeded = DecodingOptions::new();

  // A greedy transcript whose decode ALSO ran an unseeded sampled window
  // that got dropped: exactly what the blank-audio drop leaves behind.
  let filtered = TranscriptionResult::new(
    "Hello world.",
    vec![TranscriptionSegment::new().with_temperature(0.0)],
    "en",
    TranscriptionTimings::new(),
  )
  .with_task_facts(
    TaskFacts::unknown()
      .with_drew_from_rng(true)
      .with_early_stopped(false)
      // Observed-clean on the swallow axis so a seed alone earns reproducibility
      // back below (codex round 11, M2).
      .with_had_swallowed_error(false)
      // A KNOWN worker coordinate — a real drawing decode always carries one — so
      // the seed can replay it (codex round 13, M2).
      .with_worker(0),
  );
  let record = Provenance::for_result(&unseeded, &compute, &filtered);
  assert_eq!(
    record.effective_temperature(),
    Some(0.0),
    "the survivors really are all greedy -- the fix must not come from here"
  );
  assert_eq!(record.task_facts().drew_from_rng(), Some(true));
  assert!(
    !record.is_reproducible(),
    "an unseeded sampled window happened, even though nothing survived to say so"
  );

  // Seeded, the same history replays exactly (its worker schedule is known).
  assert!(
    Provenance::for_result(&DecodingOptions::new().with_seed(3), &compute, &filtered)
      .is_reproducible()
  );

  // And with nothing sampled anywhere, a segment-less transcript is
  // reproducible -- the answer the old infer-from-survivors rule could not
  // give, because zero segments left it with no evidence and it had to guess.
  let empty = result_at("en", &[]);
  assert_eq!(empty.task_facts().drew_from_rng(), Some(false));
  let greedy = Provenance::for_result(&unseeded, &compute, &empty);
  assert_eq!(greedy.effective_temperature(), None);
  assert!(
    greedy.is_reproducible(),
    "nothing ever drew from the sampler, so there is nothing to replay"
  );
}

#[test]
fn from_options_and_for_segment_record_the_explicit_draw_fact_not_the_temperature() {
  // F3 (codex round 4). `from_options`/`for_segment` take the draw fact as an
  // EXPLICIT argument and record it verbatim -- they never re-derive it from
  // the temperature they are also handed. The two come apart: a zero-iteration
  // decode lands a NON-zero-temperature segment that never drew, and a caller
  // that ran it says so. The old `temperature != 0.0` inference would have
  // called the 0.3 rows sampled and the 0.0-with-draw row greedy -- both wrong.
  let compute = ComputeOptions::new();
  let decoding = DecodingOptions::new();

  // Non-zero temperature, explicit NO draw (the zero-iteration shape).
  assert_eq!(
    Provenance::from_options(&decoding, &compute, 0.3, false)
      .task_facts()
      .drew_from_rng(),
    Some(false),
    "an explicit no-draw at 0.3 is recorded as not sampled, not inferred from 0.3"
  );
  // Zero temperature, explicit draw (proves no inference the other way either).
  assert_eq!(
    Provenance::from_options(&decoding, &compute, 0.0, true)
      .task_facts()
      .drew_from_rng(),
    Some(true),
    "an explicit draw is recorded even at temperature 0.0"
  );

  // for_segment reads the segment's temperature but takes the draw fact
  // explicitly: a 0.3 segment that never drew is not sampled.
  let never_drew = TranscriptionSegment::new().with_temperature(0.3);
  let record = Provenance::for_segment(&decoding, &compute, &never_drew, false);
  assert_eq!(
    record.effective_temperature(),
    Some(0.3),
    "the segment's rung is still recorded"
  );
  assert_eq!(
    record.task_facts().drew_from_rng(),
    Some(false),
    "a 0.3 segment that never drew is not sampled -- for_segment must not infer from 0.3"
  );
  assert_eq!(
    Provenance::for_segment(&decoding, &compute, &never_drew, true)
      .task_facts()
      .drew_from_rng(),
    Some(true),
    "and an explicit draw on that same segment IS recorded"
  );
}

#[test]
fn provenance_records_the_real_draw_fact_never_the_temperature() {
  // F3 (codex round 4). The token sampler argmax-decodes ONLY at exactly
  // `temperature == 0.0` and draws from the RNG for every other value --
  // NEGATIVES included -- exactly as Swift's `temperature != 0.0` guard does
  // (`TokenSampler.swift:49,110,140`). The reproducibility fact keys on that
  // REAL draw (`GreedyTokenSampler::drew_from_rng`), fed in explicitly, and is
  // NEVER re-derived from the temperature: the two come apart for a
  // zero-iteration decode, and a temperature-inferred fact would misreport it.
  let compute = ComputeOptions::new();

  for &temperature in &[-0.2f32, -0.0, 0.0, 0.2] {
    // Run the REAL sampler and read whether it actually drew. Wide logits
    // (`[-10, 10, ..]`) keep the negative-temperature scale-then-softmax in
    // range (codex round 3/4, F1); a narrow `[0, 1, 0.5]` would not.
    let mut sampler = crate::audio::whisper::decode::sampler::GreedyTokenSampler::new(
      temperature,
      99,
      &DecodingOptions::new(),
    );
    let token = sampler.sample(&[-10.0f32, 10.0, 0.5]).token();
    let drew = sampler.drew_from_rng();
    // `-0.0 == 0.0` in IEEE, so the sampler's `== 0.0` arm catches it too.
    assert_eq!(
      drew,
      temperature != 0.0,
      "temperature {temperature}: the sampler draws iff temperature != 0.0"
    );
    if !drew {
      assert_eq!(
        token, 1,
        "temperature {temperature} must decode greedily (argmax)"
      );
    }

    // `from_options` records THAT fact verbatim -- never inferred from the
    // temperature it is also handed.
    let unseeded = Provenance::from_options(&DecodingOptions::new(), &compute, temperature, drew);
    assert_eq!(
      unseeded.task_facts().drew_from_rng(),
      Some(drew),
      "temperature {temperature}: the recorded fact is the explicit draw"
    );
    // `from_options` cannot observe a truncation, so it is conservatively
    // non-reproducible whatever the draw (F1) -- the reproducible-iff-greedy
    // logic is exercised through `for_result`, which reads a transcript's
    // observed facts. Model that transcript: it drew iff `drew`, and POSITIVELY
    // ran to completion (`early_stopped = Some(false)`).
    assert!(
      !unseeded.is_reproducible(),
      "temperature {temperature}: from_options never promises reproducibility"
    );
    let ran_to_completion = TranscriptionResult::new(
      "x",
      vec![TranscriptionSegment::new().with_temperature(temperature)],
      "en",
      TranscriptionTimings::new(),
    )
    .with_task_facts(
      TaskFacts::unknown()
        .with_drew_from_rng(drew)
        .with_early_stopped(false)
        // Observed-clean on the swallow axis so only the draw drives the answer
        // (codex round 11, M2).
        .with_had_swallowed_error(false)
        // A KNOWN worker coordinate, so a seed can replay a real draw (codex round
        // 13, M2 — the schedule-less draw is not seed-reproducible).
        .with_worker(0),
    );
    assert_eq!(
      Provenance::for_result(&DecodingOptions::new(), &compute, &ran_to_completion)
        .is_reproducible(),
      !drew,
      "unseeded temperature {temperature}: reproducible iff nothing was drawn"
    );

    // A seed makes even a real draw replayable (its worker schedule is known).
    assert!(
      Provenance::for_result(
        &DecodingOptions::new().with_seed(7),
        &compute,
        &ran_to_completion
      )
      .is_reproducible(),
      "seeded temperature {temperature} replays exactly"
    );
  }
}

#[test]
fn for_result_reads_the_carried_flag_not_the_segment_temperature() {
  // F3 (codex round 4). `for_result` takes the draw fact ONLY from the
  // result's carried `task_facts().drew_from_rng()`, never from a segment
  // temperature. Segment discovery copies the accepted rung into a segment even
  // when ZERO sampling iterations ran (a `sample_length == 0` window at 0.3
  // lands a 0.3-temperature segment that never drew), so the old
  // `segment.temperature() != 0.0` OR reported a false "sampled" and a false
  // non-reproducible.
  let compute = ComputeOptions::new();
  let decoding = DecodingOptions::new();

  // A result whose only segment is at 0.3, but whose carried flag is false:
  // the exact shape of a zero-iteration decode.
  let never_drew = TranscriptionResult::new(
    "Hello world.",
    vec![TranscriptionSegment::new().with_temperature(0.3)],
    "en",
    TranscriptionTimings::new(),
  )
  .with_task_facts(
    // The zero-iteration decode's honest, POSITIVELY observed facts: it never
    // drew (Some(false)), ran to completion (Some(false)), and swallowed no child
    // error (Some(false) — codex round 11, M2).
    TaskFacts::unknown()
      .with_drew_from_rng(false)
      .with_early_stopped(false)
      .with_had_swallowed_error(false),
  );
  assert_eq!(
    never_drew.task_facts().drew_from_rng(),
    Some(false),
    "the result itself carries no draw"
  );
  let record = Provenance::for_result(&decoding, &compute, &never_drew);
  assert_eq!(
    record.task_facts().drew_from_rng(),
    Some(false),
    "the 0.3 segment must NOT be read as a draw -- the carried flag is the only witness"
  );
  assert!(
    record.is_reproducible(),
    "nothing drew, so it is reproducible despite the 0.3 segment"
  );

  // The converse: a run that DID draw but whose surviving segments are all
  // greedy (the blank-drop history) is still recorded as sampled and
  // non-reproducible -- from the carried flag alone.
  let drew_greedy_survivors = TranscriptionResult::new(
    "Hello",
    vec![TranscriptionSegment::new().with_temperature(0.0)],
    "en",
    TranscriptionTimings::new(),
  )
  .with_task_facts(
    TaskFacts::unknown()
      .with_drew_from_rng(true)
      .with_early_stopped(false)
      // Observed-clean on the other two axes so ONLY the carried draw drives the
      // non-reproducible result below (codex round 11, M2).
      .with_had_swallowed_error(false),
  );
  let record = Provenance::for_result(&decoding, &compute, &drew_greedy_survivors);
  assert_eq!(
    record.task_facts().drew_from_rng(),
    Some(true),
    "a carried draw survives even when every remaining segment reads 0.0"
  );
  assert!(!record.is_reproducible());
}

#[cfg(feature = "serde")]
#[test]
fn effective_temperature_non_finite_is_rejected_by_serde() {
  // Codex round 3, F6. A non-finite effective temperature must not serialize
  // to the lossy `null` serde_json emits — the required-field deserialize would
  // read that back as a forged `None` ("the ladder split the segments"), the
  // same silent round-trip disable the embedded `DecodingOptions` closes. It is
  // refused on serialize instead; a finite record still round-trips.
  let compute = ComputeOptions::new();
  for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    let record = Provenance::from_options(&DecodingOptions::new(), &compute, bad, false);
    assert!(record.effective_temperature().is_some());
    assert!(serde_json::to_string(&record).is_err());
  }
  let finite = Provenance::from_options(&DecodingOptions::new(), &compute, 0.7, false);
  let json = serde_json::to_string(&finite).unwrap();
  assert_eq!(serde_json::from_str::<Provenance>(&json).unwrap(), finite);
}
