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
  Load(#[from] crate::LoadError),
  /// A [`crate::audio::whisper::model::ModelInfo`] was constructed with an empty name.
  #[error("model info name must not be empty")]
  EmptyName,
  /// A [`crate::audio::whisper::model::SupportConfig`] JSON document was malformed or had
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
///
/// Not `Clone`/`PartialEq`/`Eq` (unlike its sibling domain-error enums):
/// [`Self::Tokenizer`] wraps [`TokenizerError`], which itself wraps the
/// foreign `tokenizers::Error` (`Box<dyn std::error::Error + Send +
/// Sync>`) and so cannot implement any of the three.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DecodeError {
  /// The CoreML runtime failed to run the decoder model.
  #[error("decoder prediction failed: {0}")]
  Prediction(#[from] crate::PredictionError),
  /// A decoder tensor failed to construct or view.
  #[error("decoder tensor failed: {0}")]
  Tensor(#[from] crate::TensorError),
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
  Backend(#[from] crate::audio::whisper::backend::BackendError),
  /// Converting sampled token ids back to text failed (the decode loop's
  /// per-step progress callback and its final result both decode through
  /// the tokenizer).
  #[error("tokenizer decode failed: {0}")]
  Tokenizer(#[from] TokenizerError),
}

/// Failure seeking to the next decode window or slicing a window's decode
/// result into segments.
///
/// Not `Clone`/`PartialEq`/`Eq` (unlike its sibling domain-error enums, same
/// reason as [`DecodeError`]): [`Self::Tokenizer`] wraps [`TokenizerError`],
/// which itself wraps the foreign `tokenizers::Error` and so cannot
/// implement any of the three.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SegmentError {
  /// A word-alignment matrix did not have the expected 2D shape, or its
  /// flattened element count did not match `rows * cols`.
  #[error("invalid alignment matrix shape: {rows} rows x {cols} cols, but data has {len} elements")]
  InvalidAlignmentShape {
    /// Expected row count (text tokens).
    rows: usize,
    /// Expected column count (audio tokens).
    cols: usize,
    /// Actual flattened element count.
    len: usize,
  },
  /// This host's CoreVideo row pitch for the Float16 surface Swift's
  /// alignment gather allocates could not be measured, so
  /// [`AlignmentGather::SwiftParity`](crate::audio::whisper::options::AlignmentGather::SwiftParity)
  /// — whose whole content is replicating what that pitch truncates —
  /// cannot be honored.
  ///
  /// Fail-closed by design: the alternative, quietly gathering every row in
  /// full, is the behavior
  /// [`AlignmentGather::Complete`](crate::audio::whisper::options::AlignmentGather::Complete)
  /// names, and substituting it under a `SwiftParity` request would be the
  /// same silent, host-dependent swap of transcript-changing behavior that
  /// whisper #41 exists to remove. A caller who prefers timings over parity
  /// asks for `Complete` explicitly.
  #[error(
    "cannot measure this host's CoreVideo Float16 row pitch for the {rows} x {cols} alignment \
     gather, so the Swift-parity gather cannot be reproduced (select \
     `AlignmentGather::Complete` to gather every row in full instead): {source}"
  )]
  AlignmentPitchUnavailable {
    /// Rows the gather would have allocated (gathered tokens).
    rows: usize,
    /// Columns the gather would have allocated (audio tokens).
    cols: usize,
    /// Why the probe allocation could not supply a pitch.
    #[source]
    source: crate::TensorError,
  },
  /// The probe allocation behind
  /// [`AlignmentGather::SwiftParity`](crate::audio::whisper::options::AlignmentGather::SwiftParity)
  /// succeeded but reported a layout the gather's model cannot describe —
  /// anything other than `[pitch, 1]` element strides with `pitch >= cols`,
  /// i.e. rows padded only *between* each other.
  ///
  /// Fail-closed for the same reason as [`Self::AlignmentPitchUnavailable`]:
  /// an unmodellable layout means the truncation Swift's gather performs is
  /// unknown, not absent.
  #[error(
    "this host's CoreVideo Float16 surface for the {rows} x {cols} alignment gather reports \
     element strides {strides:?}, which is not the row-padded row-major layout the Swift-parity \
     gather models (select `AlignmentGather::Complete` to gather every row in full instead)"
  )]
  AlignmentPitchUnexpectedLayout {
    /// Rows the gather allocated (gathered tokens).
    rows: usize,
    /// Columns the gather allocated (audio tokens).
    cols: usize,
    /// The element strides the probe allocation actually reported.
    strides: Vec<usize>,
  },
  /// Decoding a slice's tokens back to text failed.
  #[error("tokenizer decode failed: {0}")]
  Tokenizer(#[from] TokenizerError),
}

/// Failure running the pluggable voice-activity detector during
/// [`ChunkingStrategy::Vad`](crate::audio::whisper::options::ChunkingStrategy::Vad)
/// chunking.
///
/// The detector's per-frame contract
/// ([`voice_activity`](crate::audio::whisper::audio::vad::VoiceActivityDetector::voice_activity))
/// is infallible — it returns `Vec<bool>`, with no channel for a hard
/// model/runtime failure. A learned detector backed by a model (e.g. the
/// `vadkit`-gated Silero detector) therefore *latches* its first inference
/// failure and the transcription pipeline surfaces it here after driving
/// the detector, rather than letting a swallowed failure masquerade as
/// silence and silently corrupt the chunk boundaries.
///
/// Not `Clone`/`PartialEq`/`Eq` (unlike its sibling domain-error enums):
/// it carries the detector's own error as an erased
/// `Box<dyn std::error::Error + Send + Sync>`, so the pipeline stays
/// decoupled from any particular detector's error type.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VadError {
  /// The voice-activity detector reported a hard inference failure
  /// mid-stream (e.g. the Silero CoreML model failed to run a frame, or
  /// returned a non-finite probability). Its own error is preserved as
  /// the [`source`](std::error::Error::source).
  #[error("voice-activity detection failed: {0}")]
  Detection(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
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
  /// A segment-seeking or slicing failure.
  #[error("segment error: {0}")]
  Segment(#[from] SegmentError),
  /// A voice-activity-detection failure during VAD chunking: the
  /// pluggable detector hit a hard model/runtime failure that would
  /// otherwise have been swallowed into false silence.
  #[error("vad error: {0}")]
  Vad(#[from] VadError),
}

#[cfg(test)]
mod tests;
