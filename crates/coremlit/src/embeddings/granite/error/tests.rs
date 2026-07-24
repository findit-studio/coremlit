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

#[test]
fn tokenizer_contract_mismatch_display_names_check() {
  let e = Error::TokenizerContractMismatch {
    check: "vocab size",
    expected: "180000".to_string(),
    actual: "32".to_string(),
  };
  let msg = e.to_string();
  assert!(msg.contains("vocab size"), "{msg}");
  assert!(msg.contains("180000") && msg.contains("32"), "{msg}");
}

#[test]
fn input_too_large_display_shows_sizes() {
  let e = Error::InputTooLarge {
    got: 8_388_608,
    max: 1_048_576,
  };
  let msg = e.to_string();
  assert!(msg.contains("8388608") && msg.contains("1048576"), "{msg}");
}

#[test]
fn contentless_input_over_budget_display_shows_span_and_counts() {
  let e = Error::ContentlessInputOverBudget {
    start: 1,
    end: 100_001,
    tokens: 784,
    max: 512,
  };
  let msg = e.to_string();
  assert!(
    msg.contains("100001") && msg.contains("784") && msg.contains("512"),
    "{msg}"
  );
}
