use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch {
    feature: "input_features",
    expected: "[1, 1, 1001, 64] float32".to_string(),
    actual: "[1, 1, 1001, 64] float16".to_string(),
  };
  let msg = e.to_string();
  assert!(msg.contains("input_features"), "{msg}");
  assert!(msg.contains("float16"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape {
    got: vec![512, 1],
    expected: vec![1, 512],
  };
  let msg = e.to_string();
  assert!(
    msg.contains("[512, 1]") && msg.contains("[1, 512]"),
    "{msg}"
  );
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into clapkit's Error.
  let e = Error::from(coremlit::PredictionError::MissingOutput {
    name: "audio_embeds".to_string(),
  });
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn non_finite_variants_carry_index() {
  assert!(Error::NonFiniteInput { index: 7 }.to_string().contains('7'));
  assert!(
    Error::NonFiniteEmbedding { component_index: 3 }
      .to_string()
      .contains('3')
  );
}
