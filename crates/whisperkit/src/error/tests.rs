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
  let inner = coremlit::TensorError::ShapeMismatch {
    expected: 4,
    actual: 2,
  };
  let e: DecodeError = inner.into();
  assert!(matches!(e, DecodeError::Tensor(_)));
}
