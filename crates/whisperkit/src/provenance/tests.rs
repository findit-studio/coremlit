use super::*;
use crate::result::TranscriptionTimings;

/// A deliberately all-non-default decode configuration: every field the
/// capture below asserts differs from `DecodingOptions::new()`, so a
/// constructor that silently dropped one (or read the wrong knob) cannot
/// pass by coincidentally matching the default.
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
fn from_options_captures_every_library_known_field() {
  let decoding = distinctive_decoding();
  let compute = distinctive_compute();
  let provenance = Provenance::from_options(&decoding, &compute, 0.6);

  assert_eq!(provenance.task(), Task::Translate);
  assert_eq!(provenance.language(), "es");
  assert!(provenance.use_prefill_prompt());
  assert!(provenance.skip_special_tokens());
  assert!(provenance.word_timestamps());
  assert_eq!(provenance.chunking_strategy(), ChunkingStrategy::Vad);
  assert_eq!(provenance.temperature(), 0.2);
  assert_eq!(provenance.temperature_increment_on_fallback(), 0.3);
  assert_eq!(provenance.temperature_fallback_count(), 9);
  assert_eq!(provenance.seed(), Some(42));
  assert_eq!(provenance.compute(), compute);
  assert_eq!(provenance.encoder_compute_units(), ComputeUnits::All);
  assert_eq!(provenance.effective_temperature(), Some(0.6));

  // Options alone cannot observe a DETECTION outcome, so this constructor
  // does not pretend to: `for_result` is the one that records it.
  assert_eq!(provenance.detected_language(), None);

  // The identity the library genuinely cannot observe is never invented.
  assert_eq!(provenance.model_id(), None);
  assert_eq!(provenance.model_revision(), None);
  assert_eq!(provenance.tokenizer_id(), None);
  assert_eq!(provenance.tokenizer_revision(), None);
}

#[test]
fn detect_language_is_captured_resolved_not_raw() {
  // The record must hold what the pipeline ACTED on, which is the resolved
  // `detect_language ?? !use_prefill_prompt` coupling — not the raw
  // tri-state. Prefill on (the default) with detection never set resolves
  // to `false`; clearing prefill flips it to `true` with no explicit
  // detection choice anywhere.
  let compute = ComputeOptions::new();

  let prefilled = DecodingOptions::new();
  assert!(prefilled.use_prefill_prompt());
  assert!(!Provenance::from_options(&prefilled, &compute, 0.0).detect_language());

  let mut no_prefill = DecodingOptions::new();
  no_prefill.clear_use_prefill_prompt();
  let provenance = Provenance::from_options(&no_prefill, &compute, 0.0);
  assert!(
    provenance.detect_language(),
    "the resolved coupling, not the unset raw tri-state"
  );
  assert!(!provenance.use_prefill_prompt());
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
    provenance.temperature(),
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
    provenance.language(),
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

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_every_field() {
  let full = Provenance::from_options(&distinctive_decoding(), &distinctive_compute(), 0.6)
    .with_model_id("openai_whisper-tiny")
    .with_model_revision("a1b2c3d")
    .with_tokenizer_id("openai/whisper-tiny")
    .with_tokenizer_revision("deadbeef");

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
    "seed",
  ] {
    assert!(
      !object.contains_key(absent),
      "unset `{absent}` must be absent, not null"
    );
  }

  // The library-known facts are always written, so a persisted record is
  // never silently missing the settings that produced it. The two OUTCOME
  // fields are written even when they are `None` — as an explicit `null`,
  // NOT omitted like the identity above, because for them `None` is itself
  // the fact ("the ladder split the segments" / "this record was built
  // without a result"), and a reader must be able to tell that from "the
  // writer dropped the field".
  for present in [
    "task",
    "language",
    "detect_language",
    "use_prefill_prompt",
    "skip_special_tokens",
    "word_timestamps",
    "chunking_strategy",
    "temperature",
    "temperature_increment_on_fallback",
    "temperature_fallback_count",
    "compute",
    "detected_language",
    "effective_temperature",
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
  // provenance record that silently invented `use_prefill_prompt: false`
  // for a missing field would be a lie about what actually ran. Only the
  // consumer-supplied identity (and `seed`, whose absence honestly means
  // "unseeded") may be omitted.
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
    "use_prefill_prompt",
    "detected_language",
    "effective_temperature",
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
