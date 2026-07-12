use super::*;

#[test]
fn aligner_error_wraps_load_via_from() {
  let inner = coremlit::LoadError::NotFound {
    path: "base960h_aligner.mlmodelc".into(),
  };
  let e: AlignerError = inner.into();
  assert!(matches!(e, AlignerError::Load(_)));
}

#[test]
fn aligner_error_load_displays_inner_message() {
  let e: AlignerError = coremlit::LoadError::NotFound {
    path: "/tmp/missing.mlmodelc".into(),
  }
  .into();
  assert!(e.to_string().contains("/tmp/missing.mlmodelc"));
}

#[test]
fn aligner_error_contract_mismatch_displays_feature_and_shapes() {
  let e = AlignerError::ContractMismatch {
    feature: "emissions",
    expected: "[1, 2999, 29] f32".to_string(),
    actual: "[1, 2999, 32] f32".to_string(),
  };
  let rendered = e.to_string();
  assert!(rendered.contains("emissions"));
  assert!(rendered.contains("2999, 29"));
  assert!(rendered.contains("2999, 32"));
}

#[test]
fn aligner_error_is_equatable_and_cloneable() {
  let a = AlignerError::ContractMismatch {
    feature: "waveform",
    expected: "[1, 960000] f32".to_string(),
    actual: "[1, 480000] f32".to_string(),
  };
  let b = a.clone();
  assert_eq!(a, b);
}

#[test]
fn align_error_wraps_prediction_via_from() {
  let inner = coremlit::PredictionError::MissingOutput {
    name: "emissions".to_string(),
  };
  let e: AlignError = inner.into();
  assert!(matches!(e, AlignError::Prediction(_)));
  assert!(e.to_string().contains("emissions"));
}

#[test]
fn align_error_wraps_tensor_via_from() {
  let inner = coremlit::TensorError::ShapeMismatch {
    expected: 960_000,
    actual: 100,
  };
  let e: AlignError = inner.into();
  assert!(matches!(e, AlignError::Tensor(_)));
  assert!(e.to_string().contains("960000") || e.to_string().contains("960_000"));
}

#[test]
fn align_error_wraps_alignment_via_from_and_is_transparent() {
  let inner = asry::AlignmentError::EmptyText(asry::AlignmentFailure::new(
    "nothing to align".into(),
    asry::Lang::En,
  ));
  let displayed_inner = inner.to_string();
  let e: AlignError = inner.into();
  assert!(matches!(e, AlignError::Alignment(_)));
  // `#[error(transparent)]` forwards Display verbatim, no extra wrapper text.
  assert_eq!(e.to_string(), displayed_inner);
}

#[test]
fn align_error_input_too_long_displays_both_counts() {
  let e = AlignError::InputTooLong {
    got: 1_000_000,
    max: crate::encode::ENCODER_WINDOW_SAMPLES,
  };
  let rendered = e.to_string();
  assert!(rendered.contains("1000000"));
  assert!(rendered.contains("960000"));
}
