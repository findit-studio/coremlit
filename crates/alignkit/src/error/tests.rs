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
