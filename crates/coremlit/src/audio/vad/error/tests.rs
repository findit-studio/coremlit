use super::*;

#[test]
fn model_error_wraps_load_via_from() {
  let inner = crate::LoadError::NotFound("silero-vad.mlmodelc".into());
  let e: ModelError = inner.into();
  assert!(matches!(e, ModelError::Load(_)));
}

#[test]
fn model_error_contract_mismatch_displays_feature_and_shapes() {
  let e = ModelError::ContractMismatch(ContractMismatch::new(
    "audio_input",
    "[1, 4160] float32".to_string(),
    "[1, 512] float32".to_string(),
  ));
  let rendered = e.to_string();
  assert!(rendered.contains("audio_input"));
  assert!(rendered.contains("4160"));
  assert!(rendered.contains("512"));
}

#[test]
fn infer_error_wraps_prediction_and_tensor_via_from() {
  let e: InferError = crate::PredictionError::StateUnsupported.into();
  assert!(matches!(e, InferError::Prediction(_)));

  let e: InferError =
    crate::TensorError::ShapeMismatch(crate::ShapeMismatch::new(4160, 4096)).into();
  assert!(matches!(e, InferError::Tensor(_)));
}

#[test]
fn infer_error_chunk_too_long_displays_got_and_max() {
  let e = InferError::ChunkTooLong(ChunkTooLong::new(8_192, 4_096));
  let rendered = e.to_string();
  assert!(rendered.contains("8192"));
  assert!(rendered.contains("4096"));
}

#[test]
fn infer_error_non_finite_input_displays_index() {
  let e = InferError::NonFiniteInput(7);
  assert_eq!(
    e.to_string(),
    "input contains a non-finite value at index 7"
  );
}

#[test]
fn infer_error_output_shape_displays_feature_and_shapes() {
  let e = InferError::OutputShape(OutputShape::new("vad_output", vec![1, 1], vec![1, 1, 1]));
  let rendered = e.to_string();
  assert!(rendered.contains("vad_output"));
  assert!(rendered.contains("[1, 1]"));
  assert!(rendered.contains("[1, 1, 1]"));
}

#[test]
fn infer_error_non_finite_output_displays_feature_and_index() {
  let e = InferError::NonFiniteOutput(NonFiniteOutput::new("new_hidden_state", 42));
  let rendered = e.to_string();
  assert!(rendered.contains("new_hidden_state"));
  assert!(rendered.contains("index 42"));
}
