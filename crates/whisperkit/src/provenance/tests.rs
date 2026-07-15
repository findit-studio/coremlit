use std::num::NonZeroUsize;

use super::*;
use crate::{
  audio::{
    chunker::{AudioChunk, VadChunker, prepare_seek_clips},
    vad::{EnergyVad, VoiceActivityDetector},
  },
  options::{ChunkingStrategy, Task, WordGrouping},
  result::TranscriptionTimings,
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
      o.with_word_grouping(WordGrouping::SwiftParity)
    }),
  ]
}

/// A [`DecodingOptions`] with every knob set to a non-default, **non-skipped**
/// value — the fixture `mutation_table_covers_every_decoding_option` derives
/// the true field list from. Every collection is non-empty and every
/// `Option` whose default is `None` is `Some`, so nothing is elided by
/// `skip_serializing_if` and the serialized map has exactly one key per
/// field.
#[cfg(feature = "serde")]
fn fully_populated() -> DecodingOptions {
  mutations()
    .into_iter()
    .fold(DecodingOptions::new(), |options, (_, mutate)| {
      mutate(options)
    })
    // The four thresholds' mutations DISABLE them (`None`), which is the
    // right probe for the record but the wrong one for a key census: they
    // serialize as `null` rather than being skipped, so they are present
    // either way. Set them back to real numbers so this fixture reads as
    // what it claims to be — "every knob populated".
    .with_compression_ratio_threshold(3.0)
    .with_logprob_threshold(-2.0)
    .with_first_token_logprob_threshold(-2.5)
    .with_no_speech_threshold(0.7)
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
  let baseline_record = Provenance::from_options(&baseline, &compute, 0.0);

  for (field, mutate) in mutations() {
    let mutated = mutate(baseline.clone());
    assert_ne!(
      mutated, baseline,
      "`{field}`'s row does not actually change the options, so its assertion \
       below would prove nothing"
    );

    let record = Provenance::from_options(&mutated, &compute, 0.0);
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
  let record = |decoding: &DecodingOptions| Provenance::from_options(decoding, &compute, 0.0);

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

  // `word_grouping`: FineGrained vs Phrase carve a CJK segment's words
  // differently. Same story.
  let fine = DecodingOptions::new();
  let phrase = DecodingOptions::new().with_word_grouping(WordGrouping::SwiftParity);
  assert_eq!(fine.word_grouping(), WordGrouping::FineGrained);
  assert_ne!(
    record(&fine),
    record(&phrase),
    "word_grouping must be legible in the record"
  );
  assert_eq!(
    record(&phrase).decoding().word_grouping(),
    WordGrouping::SwiftParity
  );
}

#[cfg(feature = "serde")]
#[test]
fn mutation_table_covers_every_decoding_option() {
  // What makes the table above a GATE rather than a list somebody has to
  // remember: the field roster is read off `DecodingOptions` itself (its
  // serialized key set, one key per field once nothing is skipped), so a
  // knob added tomorrow lands here as an uncovered name and fails this test
  // until it is exercised. The old completeness test had no such anchor —
  // it iterated the fields the record already had, which is why it passed
  // while 19 were missing.
  let value: serde_json::Value = serde_json::to_value(fully_populated()).unwrap();
  let fields: std::collections::BTreeSet<&str> = value
    .as_object()
    .expect("DecodingOptions serializes as a map")
    .keys()
    .map(String::as_str)
    .collect();
  let covered: std::collections::BTreeSet<&str> =
    mutations().iter().map(|(field, _)| *field).collect();

  assert_eq!(
    covered, fields,
    "the provenance mutation table has fallen out of step with \
     DecodingOptions -- every knob needs a row (see `mutations`)"
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
  let baseline_json = serde_json::to_string(&Provenance::from_options(&baseline, &compute, 0.0))
    .expect("baseline serializes");

  for (field, mutate) in mutations() {
    let mutated = mutate(baseline.clone());
    let record = Provenance::from_options(&mutated, &compute, 0.0);

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
  let provenance = Provenance::from_options(&decoding, &compute, 0.6);

  // The decode configuration, whole and verbatim -- one assertion, because
  // there is one field. (`provenance_records_every_decoding_option` above is
  // what proves that field really carries all 30 knobs.)
  assert_eq!(provenance.decoding(), &decoding);
  assert_eq!(provenance.compute(), compute);
  assert_eq!(provenance.encoder_compute_units(), ComputeUnits::All);
  assert_eq!(provenance.effective_temperature(), Some(0.6));

  // Options alone cannot observe a DETECTION outcome, so this constructor
  // does not pretend to: `for_result` is the one that records it.
  assert_eq!(provenance.detected_language(), None);

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
    !Provenance::from_options(&prefilled, &compute, 0.0)
      .decoding()
      .detect_language()
  );

  let mut no_prefill = DecodingOptions::new();
  no_prefill.clear_use_prefill_prompt();
  let provenance = Provenance::from_options(&no_prefill, &compute, 0.0);
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

  let provenance = Provenance::for_segment(&decoding, &compute, &segment);
  assert_eq!(provenance.effective_temperature(), Some(0.4));
  assert_eq!(
    provenance.decoding().temperature(),
    0.0,
    "the BASE temperature stays what was configured"
  );
  assert_eq!(
    provenance,
    Provenance::from_options(&decoding, &compute, 0.4),
    "for_segment is from_options with the segment's temperature"
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
    provenance.detected_language(),
    Some("es"),
    "the DETECTED language — the fact the record exists to carry"
  );
  // Never inferred: without a result there is nothing to read it from.
  assert_eq!(
    Provenance::from_options(&decoding, &compute, 0.0).detected_language(),
    None
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

  // Greedy (0.0) never draws from the sampler -> deterministic, seed or not.
  assert!(Provenance::from_options(&unseeded, &compute, 0.0).is_reproducible());
  assert!(Provenance::from_options(&seeded, &compute, 0.0).is_reproducible());

  // Sampled: only a seed makes the draws replayable.
  assert!(
    !Provenance::from_options(&unseeded, &compute, 0.2).is_reproducible(),
    "a fallback climb with no seed is not reproducible"
  );
  assert!(Provenance::from_options(&seeded, &compute, 0.2).is_reproducible());

  // A SPLIT ladder (None) is conservatively not reproducible without a
  // seed: the rungs only ascend, so a split means at least one segment
  // climbed off 0.0 and sampled. A segment-less result carries no evidence
  // either way, and this predicate must never claim what it cannot back.
  let split = |decoding: &DecodingOptions| {
    Provenance::for_result(decoding, &compute, &result_at("en", &[0.0, 0.2])).is_reproducible()
  };
  assert!(!split(&unseeded));
  assert!(split(&seeded));
}

#[test]
fn identity_uses_the_full_option_vocabulary() {
  let base = Provenance::from_options(&DecodingOptions::new(), &ComputeOptions::new(), 0.0);

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
      crate::audio::vad::DEFAULT_FRAME_LENGTH_SAMPLES
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
  let full = Provenance::from_options(&distinctive_decoding(), &distinctive_compute(), 0.6)
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
  let bare = Provenance::from_options(&DecodingOptions::new(), &ComputeOptions::new(), 0.0);
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
  // never silently missing the settings that produced it. The two OUTCOME
  // fields are written even when they are `None` — as an explicit `null`,
  // NOT omitted like the identity above, because for them `None` is itself
  // the fact ("the ladder split the segments" / "this record was built
  // without a result"), and a reader must be able to tell that from "the
  // writer dropped the field".
  for present in [
    "decoding",
    "compute",
    "detected_language",
    "effective_temperature",
    "sampled_at_nonzero_temperature",
  ] {
    assert!(object.contains_key(present), "`{present}` must be recorded");
  }
  assert!(
    object["detected_language"].is_null(),
    "an unobserved detection outcome is an explicit null, not an omission"
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
  // The two `Option` OUTCOME fields need `required_option` to hold this
  // line at all: serde's derive silently treats a missing `Option` field
  // as `None` even with no `serde(default)` on it, which for these two
  // would forge a meaning ("the ladder split the segments" / "no result
  // was observed") out of a field the writer merely dropped. They are
  // asserted here alongside the plain ones — this is the test that proves
  // the `deserialize_with` actually defeats that path.
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

  for required in [
    "decoding",
    "compute",
    "detected_language",
    "effective_temperature",
    // The optimistic direction is the dangerous one: a dropped
    // `sampled_at_nonzero_temperature` would read back `false` ("never
    // sampled") and hand `is_reproducible` a guarantee it never earned.
    "sampled_at_nonzero_temperature",
  ] {
    let mut without = value.clone();
    without.as_object_mut().unwrap().remove(required).unwrap();
    assert!(
      serde_json::from_str::<Provenance>(&without.to_string()).is_err(),
      "a missing `{required}` must fail, not default"
    );
  }

  // Present-but-null is the honest, ACCEPTED encoding of "no such fact":
  // it is the omission that is rejected, not the null.
  let mut nulled = value;
  nulled.as_object_mut().unwrap()["effective_temperature"] = serde_json::Value::Null;
  assert_eq!(
    serde_json::from_str::<Provenance>(&nulled.to_string())
      .unwrap()
      .effective_temperature(),
    None
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
  let filtered = result_at("en", &[0.0]).with_sampled_at_nonzero_temperature();
  let record = Provenance::for_result(&unseeded, &compute, &filtered);
  assert_eq!(
    record.effective_temperature(),
    Some(0.0),
    "the survivors really are all greedy -- the fix must not come from here"
  );
  assert!(record.sampled_at_nonzero_temperature());
  assert!(
    !record.is_reproducible(),
    "an unseeded sampled window happened, even though nothing survived to say so"
  );

  // Seeded, the same history replays exactly.
  assert!(
    Provenance::for_result(&DecodingOptions::new().with_seed(3), &compute, &filtered)
      .is_reproducible()
  );

  // And with nothing sampled anywhere, a segment-less transcript is
  // reproducible -- the answer the old infer-from-survivors rule could not
  // give, because zero segments left it with no evidence and it had to guess.
  let empty = result_at("en", &[]);
  assert!(!empty.sampled_at_nonzero_temperature());
  let greedy = Provenance::for_result(&unseeded, &compute, &empty);
  assert_eq!(greedy.effective_temperature(), None);
  assert!(
    greedy.is_reproducible(),
    "nothing ever drew from the sampler, so there is nothing to replay"
  );
}

#[test]
fn a_hand_built_result_cannot_hide_a_sampled_segment() {
  // `TranscriptionResult::new` starts the flag `false`, so a result the
  // pipeline did not produce claims no sampling. The segment scan in
  // `for_result` is what keeps that from becoming a free "reproducible":
  // evidence can only ADD sampling, never retract it.
  let compute = ComputeOptions::new();
  let unseeded = DecodingOptions::new();

  let hand_built = result_at("en", &[0.0, 0.2]);
  assert!(
    !hand_built.sampled_at_nonzero_temperature(),
    "no decode path set the flag"
  );
  let record = Provenance::for_result(&unseeded, &compute, &hand_built);
  assert!(
    record.sampled_at_nonzero_temperature(),
    "a VISIBLE sampled segment is evidence too"
  );
  assert!(!record.is_reproducible());
}

#[test]
fn from_options_and_for_segment_record_their_own_temperature() {
  let compute = ComputeOptions::new();
  let decoding = DecodingOptions::new();

  assert!(
    !Provenance::from_options(&decoding, &compute, 0.0).sampled_at_nonzero_temperature(),
    "greedy: the sampler was never consulted"
  );
  assert!(Provenance::from_options(&decoding, &compute, 0.2).sampled_at_nonzero_temperature());

  let sampled_segment = TranscriptionSegment::new().with_temperature(0.4);
  assert!(
    Provenance::for_segment(&decoding, &compute, &sampled_segment).sampled_at_nonzero_temperature()
  );
}

#[test]
fn sampling_predicate_matches_the_samplers_nonzero_branch() {
  // F2 (codex round 2). The token sampler argmax-decodes ONLY at exactly
  // `temperature == 0.0` and draws from the RNG for every other value --
  // NEGATIVES included -- exactly as Swift's `temperature != 0.0` guard does
  // (`TokenSampler.swift:49,110,140`). The reproducibility predicate must key
  // on the same `!= 0.0`; the old `> 0.0` called an unseeded negative-temp
  // draw "greedy" and thus reproducible, when a re-run redraws it.
  //
  // Mutation proof: restore any of the three predicates to `> 0.0` and the
  // `-0.2` row's `sampled`/`is_reproducible` assertions fail.
  let compute = ComputeOptions::new();

  for &temperature in &[-0.2f32, -0.0, 0.0, 0.2] {
    // `-0.0 == 0.0` in IEEE, so the sampler's `== 0.0` arm catches it too;
    // this expectation IS the sampler's branch, restated.
    let samples = temperature != 0.0;

    // The sampler's own branch agrees. At a temperature the `== 0.0` arm
    // catches, decoding is argmax (index 1 of these logits) and never touches
    // the RNG; the sampled temps may or may not land on argmax for a given
    // draw, so only the greedy rows are pinned to a token.
    let mut sampler =
      crate::decode::sampler::GreedyTokenSampler::new(temperature, 99, &DecodingOptions::new());
    let token = sampler.sample(&[0.0f32, 1.0, 0.5]).token();
    if !samples {
      assert_eq!(
        token, 1,
        "temperature {temperature} must decode greedily (argmax)"
      );
    }

    // from_options predicate (`provenance/mod.rs`, `effective_temperature`).
    let unseeded = Provenance::from_options(&DecodingOptions::new(), &compute, temperature);
    assert_eq!(
      unseeded.sampled_at_nonzero_temperature(),
      samples,
      "temperature {temperature}: sampled flag must match the sampler's != 0.0 branch"
    );
    assert_eq!(
      unseeded.is_reproducible(),
      !samples,
      "unseeded temperature {temperature}: reproducible iff the sampler was never drawn"
    );

    // A seed makes even a sampled draw replayable.
    assert!(
      Provenance::from_options(&DecodingOptions::new().with_seed(7), &compute, temperature)
        .is_reproducible(),
      "seeded temperature {temperature} replays exactly"
    );

    // for_result's surviving-segment scan (the third predicate site).
    assert_eq!(
      Provenance::for_result(
        &DecodingOptions::new(),
        &compute,
        &result_at("en", &[temperature])
      )
      .sampled_at_nonzero_temperature(),
      samples,
      "temperature {temperature}: for_result's segment scan must agree"
    );
  }
}
