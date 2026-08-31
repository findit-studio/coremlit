use super::*;

#[test]
fn contract_mismatch_display_names_feature() {
  let e = Error::ContractMismatch(ContractMismatch::new(
    "mel",
    "[1, 64, 1001] float32".to_string(),
    "[1, 1001, 64] float32".to_string(),
  ));
  let msg = e.to_string();
  assert!(msg.contains("mel"), "{msg}");
  assert!(msg.contains("[1, 1001, 64]"), "{msg}");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape(OutputShape::new(vec![527], vec![1, 527]));
  let msg = e.to_string();
  assert!(msg.contains("[527]") && msg.contains("[1, 527]"), "{msg}");
}

#[test]
fn coremlit_errors_convert_via_from() {
  // `#[from]` lets `?` lift coremlit errors into ced's Error.
  let e = Error::from(crate::PredictionError::MissingOutput("logits".to_string()));
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
  let too_long = Error::AudioTooLong(AudioTooLong::new(160_001, 160_000)).to_string();
  assert!(
    too_long.contains("160001") && too_long.contains("160000"),
    "{too_long}"
  );
  assert!(Error::NonFiniteInput(42).to_string().contains("42"));
  assert!(Error::NonFiniteOutput(7).to_string().contains('7'));
  assert!(Error::EmptyWindows.to_string().contains("window"));
  assert!(Error::UnknownClassIndex(527).to_string().contains("527"));
}

#[test]
fn class_count_mismatch_display_shows_got_then_expected() {
  // `new` takes (expected, got); the message prints GOT first.
  let msg = Error::ClassCountMismatch(ClassCountMismatch::new(527, 3)).to_string();
  assert_eq!(
    msg,
    "confidence vector has 3 values, expected exactly 527 (one per class)"
  );
}

#[test]
fn invalid_confidence_display_shows_index_and_value() {
  let msg = Error::InvalidConfidence(InvalidConfidence::new(9, 1.5)).to_string();
  assert_eq!(
    msg,
    "confidence at class index 9 is 1.5, not a finite value in [0, 1]"
  );
}
