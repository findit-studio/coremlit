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
fn align_error_wraps_emissions_via_from_and_is_transparent() {
  let inner = asry::emissions::EmissionsError::NoAlignmentPath(
    asry::emissions::EmissionsFailure::new("no finite path".into()),
  );
  let displayed_inner = inner.to_string();
  let e: AlignError = inner.clone().into();
  assert!(matches!(e, AlignError::Alignment(_)));
  // `#[error(transparent)]` forwards Display verbatim, no extra wrapper text.
  assert_eq!(e.to_string(), displayed_inner);
}

#[test]
fn align_error_wraps_span_via_from_and_is_transparent() {
  let inner = asry::emissions::SpanError::Timebase {
    expected: 16_000,
    num: 1,
    den: 1_000,
  };
  let displayed_inner = inner.to_string();
  let e: AlignError = inner.into();
  assert!(matches!(e, AlignError::Span(_)));
  assert_eq!(e.to_string(), displayed_inner);
}

#[test]
fn aligner_error_wraps_seam_via_from() {
  let inner = asry::emissions::EmissionsError::Config(asry::emissions::EmissionsFailure::new(
    "bad tokenizer".into(),
  ));
  let e: AlignerError = inner.into();
  assert!(matches!(e, AlignerError::Seam(_)));
  assert!(e.to_string().contains("bad tokenizer"));
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

#[test]
fn align_error_corrupt_emissions_names_the_placement_and_the_floor() {
  // The error exists to be SELF-DIAGNOSING: a caller who flipped
  // `with_compute` must be able to read the cause straight off the message,
  // without knowing anything about fp16 subnormals. So the placement, the
  // floor that was tripped, the observed minimum and the blast radius all
  // have to survive into Display. The real ANE numbers, measured on jfk.wav.
  let e = AlignError::CorruptEmissions {
    compute: coremlit::ComputeUnits::All,
    min: -45_440.0,
    cells: 2_667,
    total: 15_950,
  };
  let rendered = e.to_string();
  assert!(
    rendered.contains("All"),
    "must name the placement: {rendered}"
  );
  assert!(
    rendered.contains("-45440"),
    "must report the min: {rendered}"
  );
  assert!(
    rendered.contains("2667"),
    "must report the blast radius: {rendered}"
  );
  assert!(
    rendered.contains("15950"),
    "must report the total: {rendered}"
  );
  assert!(
    rendered.contains(&crate::encode::LOG_PROB_FLOOR.to_string()),
    "must name the floor it tripped: {rendered}"
  );
  assert!(
    rendered.contains("DEFAULT_ENCODER_COMPUTE"),
    "must name the way out: {rendered}"
  );
}
