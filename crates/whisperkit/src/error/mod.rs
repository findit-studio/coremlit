//! Structured, per-domain error types for the WhisperKit pipeline (spec
//! §6.4). Foreign errors from `coremlit`/`tokenizers` are wrapped as typed
//! `#[from]` variants; [`TranscribeError`] composes every domain error at
//! the top level.

use std::path::PathBuf;

/// Failure locating, loading, or using a CoreML-backed Whisper model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
  /// None of the searched paths contained the model.
  #[error("model not found (searched {searched:?})")]
  NotFound {
    /// Paths that were checked.
    searched: Vec<PathBuf>,
  },
  /// The model was used from a lifecycle state that does not support the
  /// requested operation.
  #[error("model is in state `{actual}`, expected `{expected}`")]
  InvalidState {
    /// State the operation required.
    expected: &'static str,
    /// State the model was actually in.
    actual: &'static str,
  },
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] coremlit::LoadError),
  /// A [`crate::model::ModelInfo`] was constructed with an empty name.
  #[error("model info name must not be empty")]
  EmptyName,
  /// A [`crate::model::SupportConfig`] JSON document was malformed or had
  /// an unexpected shape. Carries a rendered message rather than the
  /// originating `serde_json::Error` because that type implements
  /// neither `Clone` nor `PartialEq`/`Eq`, which this enum otherwise
  /// derives uniformly across every variant.
  #[error("invalid support config: {0}")]
  InvalidSupportConfig(String),
}

/// Failure loading or using the BPE tokenizer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TokenizerError {
  /// None of the searched paths contained a tokenizer file.
  #[error("tokenizer file not found (searched {searched:?})")]
  FileNotFound {
    /// Paths that were checked.
    searched: Vec<PathBuf>,
  },
  /// The `tokenizers` crate failed to load or run the tokenizer.
  #[error("tokenizer backend failed: {0}")]
  Backend(#[from] tokenizers::Error),
  /// A token required by the pipeline is absent from the tokenizer's
  /// vocabulary.
  #[error("tokenizer vocabulary is missing required token `{token}`")]
  MissingToken {
    /// The missing token's text.
    token: &'static str,
  },
}

/// Failure preparing or validating audio input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AudioError {
  /// The audio window exceeds the model's maximum supported length.
  #[error("audio window of {got} samples exceeds the maximum of {max}")]
  WindowTooLarge {
    /// Samples provided.
    got: usize,
    /// Maximum samples supported.
    max: usize,
  },
  /// No audio samples were provided.
  #[error("audio input is empty")]
  EmptyInput,
  /// A clip's timestamp range is invalid (inverted or out of bounds).
  #[error("invalid clip range: start {start}, end {end}")]
  InvalidClipRange {
    /// Clip start time, in seconds.
    start: f32,
    /// Clip end time, in seconds.
    end: f32,
  },
}

/// Failure running or interpreting a decoder step.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DecodeError {
  /// The CoreML runtime failed to run the decoder model.
  #[error("decoder prediction failed: {0}")]
  Prediction(#[from] coremlit::PredictionError),
  /// A decoder tensor failed to construct or view.
  #[error("decoder tensor failed: {0}")]
  Tensor(#[from] coremlit::TensorError),
  /// The decoder's logits tensor has an unexpected shape.
  #[error("logits shape mismatch: expected {expected}, got {actual}")]
  LogitsShape {
    /// Elements the decode step expected.
    expected: usize,
    /// Elements the logits tensor actually had.
    actual: usize,
  },
  /// Cross-attention alignment data required for word timestamps is
  /// missing.
  #[error("decoder output is missing cross-attention alignment data")]
  MissingAlignment,
  /// The inference backend failed.
  #[error("backend failure: {0}")]
  Backend(#[from] crate::backend::BackendError),
}

/// Top-level transcription failure, composing every domain error (spec
/// §6.4).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TranscribeError {
  /// A model lifecycle failure.
  #[error("model error: {0}")]
  Model(#[from] ModelError),
  /// A tokenizer failure.
  #[error("tokenizer error: {0}")]
  Tokenizer(#[from] TokenizerError),
  /// An audio-input failure.
  #[error("audio error: {0}")]
  Audio(#[from] AudioError),
  /// A decode-step failure.
  #[error("decode error: {0}")]
  Decode(#[from] DecodeError),
}

#[cfg(test)]
mod tests;
