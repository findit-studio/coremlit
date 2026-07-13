use super::*;

use asry::emissions::{EmissionsFailure, EnglishNormalizer};

fn normalizer() -> DynTextNormalizer {
  Box::new(EnglishNormalizer::new())
}

// ---------------------------------------------------------------------
// AlignerOptions (rust-options-pattern)
// ---------------------------------------------------------------------

#[test]
fn options_new_matches_documented_defaults() {
  let o = AlignerOptions::new();
  assert_eq!(o.hop_samples(), DEFAULT_HOP_SAMPLES);
  assert_eq!(o.hop_samples().get(), 320);
  assert_eq!(o.min_speech_coverage(), DEFAULT_MIN_SPEECH_COVERAGE);
  assert_eq!(o.min_speech_coverage(), 0.5);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
  // The production placement stays `All` — model-gated tests opt down to
  // CpuOnly explicitly, they do not change the default.
  assert_eq!(o.compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(o.compute(), ComputeUnits::All);
}

#[test]
fn options_compute_overrides() {
  let o = AlignerOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(o.compute(), ComputeUnits::CpuOnly);

  let mut o = AlignerOptions::new();
  o.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(o.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(AlignerOptions::default(), AlignerOptions::new());
}

#[test]
fn options_with_builders_override() {
  let hop = NonZeroU32::new(160).unwrap();
  let o = AlignerOptions::new()
    .with_hop_samples(hop)
    .with_min_speech_coverage(0.75)
    .with_max_intra_silent_run(Duration::from_millis(120));
  assert_eq!(o.hop_samples(), hop);
  assert_eq!(o.min_speech_coverage(), 0.75);
  assert_eq!(o.max_intra_silent_run(), Duration::from_millis(120));
}

#[test]
fn options_set_in_place() {
  let mut o = AlignerOptions::new();
  o.set_hop_samples(NonZeroU32::new(640).unwrap());
  o.set_min_speech_coverage(0.25);
  o.set_max_intra_silent_run(Duration::from_millis(40));
  assert_eq!(o.hop_samples().get(), 640);
  assert_eq!(o.min_speech_coverage(), 0.25);
  assert_eq!(o.max_intra_silent_run(), Duration::from_millis(40));
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_fields_default() {
  let o: AlignerOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(o, AlignerOptions::new());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_partial_fills_defaults() {
  let o: AlignerOptions = serde_json::from_str(r#"{"min_speech_coverage":0.7}"#).unwrap();
  assert_eq!(o.hop_samples(), DEFAULT_HOP_SAMPLES);
  assert_eq!(o.min_speech_coverage(), 0.7);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips() {
  let o = AlignerOptions::new()
    .with_hop_samples(NonZeroU32::new(160).unwrap())
    .with_max_intra_silent_run(Duration::from_millis(120))
    .with_compute(ComputeUnits::CpuOnly);
  let json = serde_json::to_string(&o).unwrap();
  assert!(json.contains("cpu_only"), "round-tripped json: {json}");
  let back: AlignerOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(o, back);
}

// ---------------------------------------------------------------------
// Seam construction / blank-id wiring (DECISION 5) — hermetic: these
// build the asry seam alone (bundled tokenizer bytes + a normalizer), no
// CoreML model, so they run without ALIGNKIT_TEST_MODELS.
// ---------------------------------------------------------------------

#[test]
fn build_seam_wires_blank_id_zero_and_vocab_29() {
  let seam = build_seam(Lang::En, normalizer(), &AlignerOptions::new())
    .expect("bundled tokenizer + explicit blank id builds");
  assert_eq!(seam.blank_token_id(), crate::vocab::BLANK_ID);
  assert_eq!(seam.blank_token_id(), 0);
  assert_eq!(seam.vocab_size().get(), crate::vocab::VOCAB_SIZE);
  assert_eq!(seam.hop_samples(), DEFAULT_HOP_SAMPLES);
}

#[test]
fn build_seam_threads_options_into_the_seam() {
  let options = AlignerOptions::new()
    .with_hop_samples(NonZeroU32::new(160).unwrap())
    .with_max_intra_silent_run(Duration::from_millis(120));
  let seam = build_seam(Lang::En, normalizer(), &options).expect("builds");
  assert_eq!(seam.hop_samples(), options.hop_samples());
  assert_eq!(seam.max_intra_silent_run(), options.max_intra_silent_run());
}

#[test]
fn bundled_tokenizer_has_no_autodetectable_blank() {
  // Proves the explicit `.blank_token_id(BLANK_ID)` in `build_seam` is
  // load-bearing: WITHOUT it, asry's default `<pad>` / `[PAD]` / `<blank>`
  // auto-detect finds nothing in the chordai vocab and construction FAILS.
  // A mutant dropping that override would regress to exactly this error.
  let result = EmissionsAligner::builder(Lang::En, crate::vocab::tokenizer_json_bytes())
    .normalizer(normalizer())
    .build();
  assert!(
    matches!(result, Err(EmissionsError::Config(_))),
    "auto-detect must fail without an explicit blank id"
  );
}

// ---------------------------------------------------------------------
// Recoverable-subset mapping — the `align_chunk` policy, tested directly.
// ---------------------------------------------------------------------

fn failure(message: &str) -> EmissionsFailure {
  EmissionsFailure::new(message.into())
}

#[test]
fn recover_maps_no_alignment_path_to_empty_words() {
  let result =
    recover_or_error(EmissionsError::NoAlignmentPath(failure("no finite path"))).unwrap();
  assert!(result.words().is_empty());
}

#[test]
fn recover_maps_semantic_oov_to_empty_words() {
  let result = recover_or_error(EmissionsError::SemanticOutOfVocab(failure(
    "fail-closed OOV",
  )))
  .unwrap();
  assert!(result.words().is_empty());
}

#[test]
fn recover_propagates_non_recoverable_errors() {
  // A config / abort failure is a HARD error, never empty words — the exact
  // distinction that stops a broken setup from silently emitting empty
  // alignments forever.
  assert!(matches!(
    recover_or_error(EmissionsError::Config(failure("blank id >= V"))),
    Err(AlignError::Alignment(EmissionsError::Config(_)))
  ));
  assert!(matches!(
    recover_or_error(EmissionsError::Aborted(failure("aborted"))),
    Err(AlignError::Alignment(EmissionsError::Aborted(_)))
  ));
}
