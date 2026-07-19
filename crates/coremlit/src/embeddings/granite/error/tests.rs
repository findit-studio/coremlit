use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch {
    feature: "input_ids",
    expected: "[1, 512] int32".to_string(),
    actual: "[1, 512] float16".to_string(),
  };
  let msg = e.to_string();
  assert!(msg.contains("input_ids"), "{msg}");
  assert!(msg.contains("float16"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape {
    got: vec![384, 1],
    expected: vec![1, 384],
  };
  let msg = e.to_string();
  assert!(
    msg.contains("[384, 1]") && msg.contains("[1, 384]"),
    "{msg}"
  );
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into granite's Error.
  let e = Error::from(crate::PredictionError::MissingOutput {
    name: "embedding".to_string(),
  });
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn non_finite_variants_carry_index() {
  assert!(
    Error::NonFiniteOutput { index: 7 }
      .to_string()
      .contains('7')
  );
  assert!(
    Error::NonFiniteEmbedding { component_index: 3 }
      .to_string()
      .contains('3')
  );
}
