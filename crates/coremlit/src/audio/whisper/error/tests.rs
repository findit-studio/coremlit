use super::*;

#[test]
fn transcribe_error_composes_via_from() {
  let e: TranscribeError = AudioError::EmptyInput.into();
  assert!(matches!(e, TranscribeError::Audio(AudioError::EmptyInput)));
  let e: TranscribeError = ModelError::InvalidState {
    expected: "loaded",
    actual: "unloaded",
  }
  .into();
  assert!(e.to_string().contains("loaded"));
}

#[test]
fn tokenizer_missing_token_displays_name() {
  let e = TokenizerError::MissingToken {
    token: "<|endoftext|>",
  };
  assert_eq!(
    e.to_string(),
    "tokenizer vocabulary is missing required token `<|endoftext|>`"
  );
}

#[test]
fn coreml_errors_wrap_typed() {
  let inner = crate::TensorError::ShapeMismatch {
    expected: 4,
    actual: 2,
  };
  let e: DecodeError = inner.into();
  assert!(matches!(e, DecodeError::Tensor(_)));
}

#[test]
fn transcribe_error_composes_tokenizer_and_decode_arms() {
  let e: TranscribeError = TokenizerError::MissingToken {
    token: "<|endoftext|>",
  }
  .into();
  assert!(matches!(e, TranscribeError::Tokenizer(_)));
  let e: TranscribeError = DecodeError::MissingAlignment.into();
  assert!(matches!(e, TranscribeError::Decode(_)));
}

#[test]
fn decode_error_composes_tokenizer_arm() {
  let e: DecodeError = TokenizerError::MissingToken {
    token: "<|endoftext|>",
  }
  .into();
  assert!(matches!(e, DecodeError::Tokenizer(_)));
}

#[test]
fn segment_error_composes_tokenizer_arm() {
  let e: SegmentError = TokenizerError::MissingToken {
    token: "<|endoftext|>",
  }
  .into();
  assert!(matches!(e, SegmentError::Tokenizer(_)));
}

#[test]
fn transcribe_error_composes_segment_arm() {
  let e: TranscribeError = SegmentError::InvalidAlignmentShape {
    rows: 4,
    cols: 8,
    len: 16,
  }
  .into();
  assert!(matches!(e, TranscribeError::Segment(_)));
  assert!(e.to_string().contains("16"));
}
