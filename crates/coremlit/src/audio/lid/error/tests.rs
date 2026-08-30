use super::*;
use crate::audio::lid::{MAX_FRAMES, MAX_SAMPLES, MIN_FRAMES, MIN_SAMPLES, frame_count};

/// The rejection carries the caller's own units — samples AND frames, with both
/// bound pairs — so acting on it never requires re-deriving the geometry or
/// parsing a message.
#[test]
fn frame_count_out_of_range_carries_both_unit_systems() {
  let detail = FrameCountOutOfRange::for_samples(1_000_000);
  assert_eq!(detail.samples(), 1_000_000);
  assert_eq!(detail.frames(), frame_count(1_000_000));
  assert_eq!(detail.min_frames(), MIN_FRAMES);
  assert_eq!(detail.max_frames(), MAX_FRAMES);
  assert_eq!(detail.min_samples(), MIN_SAMPLES);
  assert_eq!(detail.max_samples(), MAX_SAMPLES);
  assert!(!detail.is_too_short());

  assert!(FrameCountOutOfRange::for_samples(0).is_too_short());
  assert!(FrameCountOutOfRange::for_samples(MIN_SAMPLES - 1).is_too_short());
  assert!(!FrameCountOutOfRange::for_samples(MIN_SAMPLES).is_too_short());
}

/// The rendered message names every number a caller needs to correct the clip,
/// and does NOT leak the runtime's own axis-indexed wording.
#[test]
fn frame_count_out_of_range_renders_the_actionable_numbers() {
  let rendered = Error::from(FrameCountOutOfRange::for_samples(800)).to_string();
  for needle in ["800 samples", "6 mel frames", "10..=3001", "1440..=480159"] {
    assert!(
      rendered.contains(needle),
      "{needle:?} missing from {rendered:?}"
    );
  }
  assert!(
    !rendered.contains("dimension"),
    "the CoreML runtime's axis wording must not surface: {rendered}"
  );
}

/// The contract-mismatch payload keeps the feature name and both renderings.
#[test]
fn contract_mismatch_reports_the_feature_and_both_sides() {
  let detail = ContractMismatch::new(
    "mel_features",
    "[1, 10..=3001, 60] float32".to_owned(),
    "[1, 301, 80] float32".to_owned(),
  );
  assert_eq!(detail.feature(), "mel_features");
  assert_eq!(detail.expected(), "[1, 10..=3001, 60] float32");
  assert_eq!(detail.actual(), "[1, 301, 80] float32");

  let rendered = Error::from(detail).to_string();
  assert!(rendered.contains("mel_features"), "{rendered}");
  assert!(rendered.contains("[1, 301, 80] float32"), "{rendered}");
}

/// The predict-time shape payload keeps both shapes, so a caller can tell an
/// axis swap from a width change.
#[test]
fn output_shape_reports_both_shapes() {
  let detail = OutputShape::new(vec![107, 1], vec![1, 107]);
  assert_eq!(detail.got(), [107, 1]);
  assert_eq!(detail.expected(), [1, 107]);
  assert!(Error::from(detail).to_string().contains("[107, 1]"));
}

/// Every payload-carrying variant is a NEWTYPE, so it can be destructured with
/// one binding and its payload handled on its own. This test is the shape
/// contract itself: adding a struct-shaped variant would not compile here.
#[test]
fn every_payload_variant_is_a_newtype() {
  let cases: Vec<Error> = vec![
    ContractMismatch::new("mel_features", "a".to_owned(), "b".to_owned()).into(),
    OutputShape::new(vec![2], vec![1]).into(),
    FrameCountOutOfRange::for_samples(0).into(),
    Error::NonFiniteInput(3),
    Error::NonFiniteOutput(4),
    Error::UnknownLanguageIndex(999),
  ];

  for error in cases {
    let rendered = error.to_string();
    assert!(!rendered.is_empty(), "every variant must render");
    match error {
      Error::ContractMismatch(detail) => assert_eq!(detail.feature(), "mel_features"),
      Error::OutputShape(detail) => assert_eq!(detail.expected(), [1]),
      Error::FrameCountOutOfRange(detail) => assert!(detail.is_too_short()),
      Error::NonFiniteInput(index) => assert_eq!(index, 3),
      Error::NonFiniteOutput(index) => assert_eq!(index, 4),
      Error::UnknownLanguageIndex(index) => assert_eq!(index, 999),
      other => panic!("unexpected variant {other:?}"),
    }
  }
}

/// The index-carrying variants render their index, so a log line is enough to
/// locate the offending sample or column.
#[test]
fn index_variants_render_their_index() {
  assert!(Error::NonFiniteInput(1_234).to_string().contains("1234"));
  assert!(Error::NonFiniteOutput(56).to_string().contains("56"));
  assert!(Error::UnknownLanguageIndex(107).to_string().contains("107"));
}

/// The foreign errors arrive through `#[from]` and keep their own message, so
/// `?` works and no cause is flattened into a string here.
#[test]
fn foreign_errors_convert_and_keep_their_message() {
  let load = crate::LoadError::NotFound {
    path: std::path::PathBuf::from("/nonexistent/lid.mlmodelc"),
  };
  let inner = load.to_string();
  let error = Error::from(load);
  assert!(matches!(error, Error::Load(_)));
  assert!(error.to_string().contains(&inner), "{error}");

  let tensor = crate::TensorError::ShapeMismatch {
    expected: 60,
    actual: 61,
  };
  assert!(matches!(Error::from(tensor), Error::Tensor(_)));
}
