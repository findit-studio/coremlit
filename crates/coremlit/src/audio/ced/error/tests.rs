use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch {
    feature: "mel",
    expected: "[1, 64, 1001] float32".to_string(),
    actual: "[1, 1001, 64] float32".to_string(),
  };
  let msg = e.to_string();
  assert!(msg.contains("mel"), "{msg}");
  assert!(msg.contains("[1, 1001, 64]"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape {
    got: vec![527],
    expected: vec![1, 527],
  };
  let msg = e.to_string();
  assert!(msg.contains("[527]") && msg.contains("[1, 527]"), "{msg}");
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into ced's Error.
  let e = Error::from(crate::PredictionError::MissingOutput {
    name: "logits".to_string(),
  });
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn windit_errors_convert_via_from() {
  let e = Error::from(WinditError::Empty);
  assert!(matches!(e, Error::Windowing(_)), "got {e:?}");
}

#[test]
fn input_variants_render_their_payloads() {
  assert!(Error::EmptyAudio.to_string().contains("empty"));
  let too_long = Error::AudioTooLong {
    len: 160_001,
    max: 160_000,
  }
  .to_string();
  assert!(
    too_long.contains("160001") && too_long.contains("160000"),
    "{too_long}"
  );
  assert!(
    Error::NonFiniteInput { index: 42 }
      .to_string()
      .contains("42")
  );
  assert!(
    Error::NonFiniteOutput { index: 7 }
      .to_string()
      .contains('7')
  );
  assert!(Error::EmptyWindows.to_string().contains("window"));
  assert!(
    Error::UnknownClassIndex { index: 527 }
      .to_string()
      .contains("527")
  );
}
