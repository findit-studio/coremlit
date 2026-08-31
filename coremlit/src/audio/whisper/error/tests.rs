use super::*;

#[test]
fn transcribe_error_composes_via_from() {
  let e: TranscribeError = AudioError::EmptyInput.into();
  assert!(matches!(e, TranscribeError::Audio(AudioError::EmptyInput)));
  let e: TranscribeError = ModelError::InvalidState(InvalidState::new("loaded", "unloaded")).into();
  assert!(e.to_string().contains("loaded"));
}

#[test]
fn tokenizer_missing_token_displays_name() {
  let e = TokenizerError::MissingToken("<|endoftext|>");
  assert_eq!(
    e.to_string(),
    "tokenizer vocabulary is missing required token `<|endoftext|>`"
  );
}

#[test]
fn coreml_errors_wrap_typed() {
  let inner = crate::TensorError::ShapeMismatch(crate::ShapeMismatch::new(4, 2));
  let e: DecodeError = inner.into();
  assert!(matches!(e, DecodeError::Tensor(_)));
}

#[test]
fn transcribe_error_composes_tokenizer_and_decode_arms() {
  let e: TranscribeError = TokenizerError::MissingToken("<|endoftext|>").into();
  assert!(matches!(e, TranscribeError::Tokenizer(_)));
  let e: TranscribeError = DecodeError::MissingAlignment.into();
  assert!(matches!(e, TranscribeError::Decode(_)));
}

#[test]
fn decode_error_composes_tokenizer_arm() {
  let e: DecodeError = TokenizerError::MissingToken("<|endoftext|>").into();
  assert!(matches!(e, DecodeError::Tokenizer(_)));
}

#[test]
fn segment_error_composes_tokenizer_arm() {
  let e: SegmentError = TokenizerError::MissingToken("<|endoftext|>").into();
  assert!(matches!(e, SegmentError::Tokenizer(_)));
}

#[test]
fn alignment_pitch_errors_name_the_shape_and_the_explicit_way_out() {
  // Both arms are FAIL-CLOSED refusals of the OPT-IN
  // `AlignmentGather::SwiftParity`, so their whole job is telling a caller
  // what could not be measured and which option gathers every row instead.
  // Neither is reachable on a supported host (the probe allocation is the same
  // one the decoder's f16 tensors use), which is exactly why the messages are
  // pinned here rather than by a path that can produce them. The default
  // gather returns neither: it allocates no surface and measures nothing.
  let unavailable = SegmentError::AlignmentPitchUnavailable(AlignmentPitchUnavailable::new(
    30,
    1500,
    crate::TensorError::SurfaceUnsupported,
  ));
  let text = unavailable.to_string();
  assert!(text.contains("30 x 1500"), "{text}");
  assert!(text.contains("AlignmentGather::Complete"), "{text}");
  assert!(
    std::error::Error::source(&unavailable).is_some(),
    "the tensor failure must survive as the source"
  );

  let unexpected = SegmentError::AlignmentPitchUnexpectedLayout(
    AlignmentPitchUnexpectedLayout::new(30, 1500, vec![1504, 2]),
  );
  let text = unexpected.to_string();
  assert!(text.contains("[1504, 2]"), "{text}");
  assert!(text.contains("AlignmentGather::Complete"), "{text}");

  let composed: TranscribeError = unexpected.into();
  assert!(matches!(composed, TranscribeError::Segment(_)));
  let composed: TranscribeError = unavailable.into();
  assert!(matches!(composed, TranscribeError::Segment(_)));
}

#[test]
fn transcribe_error_composes_segment_arm() {
  let e: TranscribeError =
    SegmentError::InvalidAlignmentShape(InvalidAlignmentShape::new(4, 8, 16)).into();
  assert!(matches!(e, TranscribeError::Segment(_)));
  assert!(e.to_string().contains("16"));
}

#[test]
fn search_path_errors_display_the_paths_they_searched() {
  // A path with a SPACE, so a `Display` rendering could not masquerade as the
  // `Debug` one these two deliberately use for a whole `Vec<PathBuf>`.
  let paths = vec![PathBuf::from("/tmp/a b/tiny.mlmodelc")];
  assert_eq!(
    ModelError::NotFound(paths.clone()).to_string(),
    "model not found (searched [\"/tmp/a b/tiny.mlmodelc\"])"
  );
  assert_eq!(
    TokenizerError::FileNotFound(paths).to_string(),
    "tokenizer file not found (searched [\"/tmp/a b/tiny.mlmodelc\"])"
  );
}

#[test]
fn clip_range_and_alignment_shape_display_their_payloads() {
  assert_eq!(
    AudioError::InvalidClipRange(InvalidClipRange::new(1.5, 0.5)).to_string(),
    "invalid clip range: start 1.5, end 0.5"
  );
  assert_eq!(
    SegmentError::InvalidAlignmentShape(InvalidAlignmentShape::new(2, 3, 7)).to_string(),
    "invalid alignment matrix shape: 2 rows x 3 cols, but data has 7 elements"
  );
}
