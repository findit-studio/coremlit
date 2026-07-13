use super::*;

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
  assert_eq!(provenance.effective_temperature(), 0.6);

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
  assert_eq!(provenance.effective_temperature(), 0.4);
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
  // never silently missing the settings that produced it.
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
    "effective_temperature",
  ] {
    assert!(object.contains_key(present), "`{present}` must be recorded");
  }

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
  let full = Provenance::from_options(&distinctive_decoding(), &distinctive_compute(), 0.6);
  let mut value: serde_json::Value = serde_json::to_value(&full).unwrap();
  value
    .as_object_mut()
    .unwrap()
    .remove("use_prefill_prompt")
    .unwrap();

  assert!(
    serde_json::from_str::<Provenance>(&value.to_string()).is_err(),
    "a missing library-known field must fail, not default"
  );
}
