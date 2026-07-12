use super::*;

#[test]
fn model_error_wraps_load_via_from() {
  let inner = coremlit::LoadError::NotFound {
    path: "seg.mlmodelc".into(),
  };
  let e: ModelError = inner.into();
  assert!(matches!(e, ModelError::Load(_)));
}

#[test]
fn model_error_contract_mismatch_displays_feature_and_shapes() {
  let e = ModelError::ContractMismatch {
    feature: "segments",
    expected: "[1, 589, 7] f32".to_string(),
    actual: "[1, 592, 7] f32".to_string(),
  };
  let rendered = e.to_string();
  assert!(rendered.contains("segments"));
  assert!(rendered.contains("589"));
  assert!(rendered.contains("592"));
}

#[test]
fn infer_error_wraps_prediction_and_tensor_via_from() {
  let e: InferError = coremlit::PredictionError::StateUnsupported.into();
  assert!(matches!(e, InferError::Prediction(_)));

  let e: InferError = coremlit::TensorError::ShapeMismatch {
    expected: 4,
    actual: 2,
  }
  .into();
  assert!(matches!(e, InferError::Tensor(_)));
}

#[test]
fn infer_error_non_finite_output_displays_index() {
  let e = InferError::NonFiniteOutput { index: 42 };
  assert_eq!(
    e.to_string(),
    "output contains a non-finite value at index 42"
  );
}

#[test]
fn infer_error_input_length_displays_got_and_expected() {
  let e = InferError::InputLength {
    got: 100,
    expected: 160_000,
  };
  let rendered = e.to_string();
  assert!(rendered.contains("100"));
  assert!(rendered.contains("160000"));
}

#[test]
fn infer_error_output_shape_displays_got_and_expected() {
  // Missing pin (T2 review-queue item): every other variant has a Display
  // test, but `OutputShape` (added in fix round 1, commit fcbce74) never
  // got one.
  let e = InferError::OutputShape {
    got: vec![1, 7, 589],
    expected: vec![1, 589, 7],
  };
  let rendered = e.to_string();
  assert!(rendered.contains("[1, 7, 589]"));
  assert!(rendered.contains("[1, 589, 7]"));
}

#[test]
fn infer_error_non_finite_input_displays_index() {
  let e = InferError::NonFiniteInput { index: 7 };
  assert_eq!(
    e.to_string(),
    "input contains a non-finite value at index 7"
  );
}

#[test]
fn infer_error_empty_mask_displays_message() {
  let e = InferError::EmptyMask;
  assert_eq!(e.to_string(), "mask has no active (true) frame");
}

#[test]
fn extract_error_composes_model_arm() {
  let model_err: ModelError = coremlit::LoadError::NotFound {
    path: "seg.mlmodelc".into(),
  }
  .into();
  let e: ExtractError = model_err.into();
  assert!(matches!(e, ExtractError::Model(ModelError::Load(_))));
}

#[test]
fn extract_error_composes_infer_arm() {
  let infer_err: InferError = coremlit::TensorError::ShapeMismatch {
    expected: 4,
    actual: 2,
  }
  .into();
  let e: ExtractError = infer_err.into();
  assert!(matches!(e, ExtractError::Infer(InferError::Tensor(_))));
}
