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
  assert_eq!(o.min_speech_coverage(), DEFAULT_MIN_SPEECH_COVERAGE);
  assert_eq!(o.min_speech_coverage(), 0.5);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
  // The shipping placement is CpuOnly, and it is a correctness requirement:
  // the ANE placements underflow this model's fp16 `log(softmax(·))` tail to a
  // `-45440` sentinel. See `DEFAULT_ENCODER_COMPUTE`.
  assert_eq!(o.compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(o.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_compute_overrides() {
  // Override with placements that are NOT the default, or this would pass
  // against a no-op `with_compute`.
  let o = AlignerOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(o.compute(), ComputeUnits::CpuAndGpu);

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
  let o = AlignerOptions::new()
    .with_min_speech_coverage(0.75)
    .with_max_intra_silent_run(Duration::from_millis(120));
  assert_eq!(o.min_speech_coverage(), 0.75);
  assert_eq!(o.max_intra_silent_run(), Duration::from_millis(120));
}

#[test]
fn options_set_in_place() {
  let mut o = AlignerOptions::new();
  o.set_min_speech_coverage(0.25);
  o.set_max_intra_silent_run(Duration::from_millis(40));
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
  assert_eq!(o.min_speech_coverage(), 0.7);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
  assert_eq!(o.compute(), DEFAULT_ENCODER_COMPUTE);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips() {
  // A non-default compute, so the round-trip proves the field actually
  // survives serialization rather than being re-defaulted on the way back.
  let o = AlignerOptions::new()
    .with_max_intra_silent_run(Duration::from_millis(120))
    .with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&o).unwrap();
  assert!(json.contains("cpu_and_gpu"), "round-tripped json: {json}");
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
}

#[test]
fn build_seam_threads_options_into_the_seam() {
  let options = AlignerOptions::new().with_max_intra_silent_run(Duration::from_millis(120));
  let seam = build_seam(Lang::En, normalizer(), &options).expect("builds");
  assert_eq!(seam.max_intra_silent_run(), options.max_intra_silent_run());
}

#[test]
fn seam_stride_is_the_encoder_stride() {
  // THE one-stride invariant. The stride that TIMES the words (asry's seam)
  // and the stride that TRUNCATES the emissions
  // (`encode::truncated_frame_count`, which divides by
  // `encode::HOP_SAMPLES`) must be the same number, or every word is skewed
  // in proportion to the difference. They are not independently checked
  // downstream: asry's `validate_stride_extent` allows `chunk_extent ± 2·hop`,
  // which on jfk.wav accepts 319, 320 AND 321 without error.
  //
  // This held only by coincidence while `AlignerOptions::hop_samples` existed
  // (it fed the seam, never the encoder); it now holds by construction, since
  // `SEAM_HOP_SAMPLES` is DERIVED from `encode::HOP_SAMPLES`. A mutant that
  // re-spells the seam's stride as a literal fails here.
  let seam = build_seam(Lang::En, normalizer(), &AlignerOptions::new()).expect("builds");
  assert_eq!(seam.hop_samples(), SEAM_HOP_SAMPLES);
  assert_eq!(
    seam.hop_samples().get() as usize,
    crate::encode::HOP_SAMPLES,
    "the seam's word-timing stride must equal the encoder's truncation stride"
  );
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
