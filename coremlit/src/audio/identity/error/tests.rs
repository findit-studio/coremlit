use super::*;

#[test]
fn contract_mismatch_display_names_feature_and_both_sides() {
  let e = Error::ContractMismatch(ContractMismatch::new(
    "mel",
    "[1, 72, 401] float32".to_string(),
    "[1, 401, 72] float32".to_string(),
  ));
  let msg = e.to_string();
  assert!(msg.contains("mel"), "{msg}");
  assert!(msg.contains("[1, 72, 401]"), "{msg}");
  assert!(msg.contains("[1, 401, 72]"), "{msg}");
}

#[test]
fn contract_mismatch_accessors_round_trip() {
  let m = ContractMismatch::new("embedding", "a".to_string(), "b".to_string());
  assert_eq!(m.feature(), "embedding");
  assert_eq!(m.expected(), "a");
  assert_eq!(m.actual(), "b");
}

#[test]
fn output_shape_display_shows_both() {
  let e = Error::OutputShape(OutputShape::new(vec![192], vec![1, 192]));
  let msg = e.to_string();
  assert!(msg.contains("[192]") && msg.contains("[1, 192]"), "{msg}");
  let s = OutputShape::new(vec![192], vec![1, 192]);
  assert_eq!(s.got(), [192]);
  assert_eq!(s.expected(), [1, 192]);
}

/// The window error carries BOTH counts and says the door neither pads nor
/// truncates — the whole point of the variant is that a caller learns what
/// happened to their clip, which is nothing.
#[test]
fn window_length_display_carries_both_counts_and_the_policy() {
  let e = Error::WindowLength(WindowLength::new(48_000, 96_000));
  let msg = e.to_string();
  assert!(msg.contains("48000"), "{msg}");
  assert!(msg.contains("96000"), "{msg}");
  assert!(msg.contains("neither padded nor truncated"), "{msg}");
  let w = WindowLength::new(48_000, 96_000);
  assert_eq!((w.got(), w.expected()), (48_000, 96_000));
}

#[test]
fn coremlit_errors_convert_via_from() {
  let e = Error::from(crate::PredictionError::MissingOutput(
    "embedding".to_string(),
  ));
  assert!(matches!(e, Error::Prediction(_)), "got {e:?}");
}

#[test]
fn non_finite_variants_render_their_index() {
  assert!(Error::NonFiniteInput(41).to_string().contains("41"));
  assert!(Error::NonFiniteOutput(7).to_string().contains("7"));
}
