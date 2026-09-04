//! Structured, per-domain error types for the WhisperKit pipeline (spec
//! §6.4). Foreign errors from `coremlit`/`tokenizers` are wrapped as typed
//! `#[from]` variants; [`TranscribeError`] composes every domain error at
//! the top level.

use std::path::PathBuf;

/// The model was used from a lifecycle state that does not support the
/// requested operation.
///
/// Payload of [`ModelError::InvalidState`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidState {
  /// State the operation required.
  expected: &'static str,
  /// State the model was actually in.
  actual: &'static str,
}

impl InvalidState {
  /// Construct from the state the operation required and the state the
  /// model was actually in.
  #[inline(always)]
  pub const fn new(expected: &'static str, actual: &'static str) -> Self {
    Self { expected, actual }
  }

  /// State the operation required.
  #[inline(always)]
  pub const fn expected(&self) -> &'static str {
    self.expected
  }

  /// State the model was actually in.
  #[inline(always)]
  pub const fn actual(&self) -> &'static str {
    self.actual
  }
}

/// A model name is not one plain path component.
///
/// Payload of [`ModelError::ModelName`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelName {
  /// The name the caller passed.
  name: String,
  /// Why it is not one path component, as a phrase completing
  /// `` `{name}` ... ``.
  reason: &'static str,
}

impl ModelName {
  /// Construct from the offending name and the phrase naming its defect.
  #[inline(always)]
  pub const fn new(name: String, reason: &'static str) -> Self {
    Self { name, reason }
  }

  /// The name the caller passed.
  #[inline(always)]
  pub fn name(&self) -> &str {
    &self.name
  }

  /// Why it is not one path component, as a phrase completing
  /// `` `{name}` ... ``.
  #[inline(always)]
  pub const fn reason(&self) -> &'static str {
    self.reason
  }
}

/// Failure locating, loading, or using a CoreML-backed Whisper model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
  /// None of the searched paths contained the model. Carries the paths
  /// that were checked.
  #[error("model not found (searched {0:?})")]
  NotFound(Vec<PathBuf>),
  /// The model was used from a lifecycle state that does not support the
  /// requested operation.
  #[error("model is in state `{}`, expected `{}`", .0.actual(), .0.expected())]
  InvalidState(InvalidState),
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] crate::LoadError),
  /// A [`crate::audio::whisper::model::ModelInfo`] was constructed with an empty name.
  #[error("model info name must not be empty")]
  EmptyName,
  /// The model name handed to
  /// [`detect_model_url`](crate::audio::whisper::model::detect_model_url) is
  /// not one plain path component, so `{folder}/{name}.mlmodelc` would resolve
  /// somewhere other than the folder the caller named. Nothing on disk was
  /// looked at. Distinct from [`Self::EmptyName`], which is
  /// [`ModelInfo`](crate::audio::whisper::model::ModelInfo)'s own constructor
  /// guard over a different name.
  #[error("model name `{}` {}", .0.name(), .0.reason())]
  ModelName(ModelName),
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
  /// None of the searched paths contained a tokenizer file. Carries the
  /// paths that were checked.
  #[error("tokenizer file not found (searched {0:?})")]
  FileNotFound(Vec<PathBuf>),
  /// The `tokenizers` crate failed to load or run the tokenizer.
  #[error("tokenizer backend failed: {0}")]
  Backend(#[from] tokenizers::Error),
  /// A token required by the pipeline is absent from the tokenizer's
  /// vocabulary. Carries the missing token's text.
  #[error("tokenizer vocabulary is missing required token `{0}`")]
  MissingToken(&'static str),
}

/// The audio window exceeds the model's maximum supported length.
///
/// Payload of [`AudioError::WindowTooLarge`].
#[derive(Debug, Clone, PartialEq)]
pub struct WindowTooLarge {
  /// Samples provided.
  got: usize,
  /// Maximum samples supported.
  max: usize,
}

impl WindowTooLarge {
  /// Construct from the samples provided and the maximum supported.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// Samples provided.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// Maximum samples supported.
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A clip's timestamp range is invalid (inverted or out of bounds).
///
/// Payload of [`AudioError::InvalidClipRange`].
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidClipRange {
  /// Clip start time, in seconds.
  start: f32,
  /// Clip end time, in seconds.
  end: f32,
}

impl InvalidClipRange {
  /// Construct from the clip's start and end times, in seconds.
  #[inline(always)]
  pub const fn new(start: f32, end: f32) -> Self {
    Self { start, end }
  }

  /// Clip start time, in seconds.
  #[inline(always)]
  pub const fn start(&self) -> f32 {
    self.start
  }

  /// Clip end time, in seconds.
  #[inline(always)]
  pub const fn end(&self) -> f32 {
    self.end
  }
}

/// Failure preparing or validating audio input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AudioError {
  /// The audio window exceeds the model's maximum supported length.
  #[error("audio window of {} samples exceeds the maximum of {}", .0.got(), .0.max())]
  WindowTooLarge(WindowTooLarge),
  /// No audio samples were provided.
  #[error("audio input is empty")]
  EmptyInput,
  /// A clip's timestamp range is invalid (inverted or out of bounds).
  #[error("invalid clip range: start {}, end {}", .0.start(), .0.end())]
  InvalidClipRange(InvalidClipRange),
}

/// The decoder's logits tensor has an unexpected shape.
///
/// Payload of [`DecodeError::LogitsShape`].
#[derive(Debug)]
pub struct LogitsShape {
  /// Elements the decode step expected.
  expected: usize,
  /// Elements the logits tensor actually had.
  actual: usize,
}

impl LogitsShape {
  /// Construct from the element count the decode step expected and the
  /// count the logits tensor actually had.
  #[inline(always)]
  pub const fn new(expected: usize, actual: usize) -> Self {
    Self { expected, actual }
  }

  /// Elements the decode step expected.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Elements the logits tensor actually had.
  #[inline(always)]
  pub const fn actual(&self) -> usize {
    self.actual
  }
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
  #[error("logits shape mismatch: expected {}, got {}", .0.expected(), .0.actual())]
  LogitsShape(LogitsShape),
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
  /// A logits filter was given an id it cannot mask: it is not a position in
  /// the step's logits. Reached from caller-supplied
  /// [`DecodingOptions::suppress_tokens`](crate::audio::whisper::options::DecodingOptions::set_suppress_tokens)
  /// and from a token slice whose ids outrun the decoder's vocabulary.
  #[error(
    "logits filter cannot mask token {} in a {}-wide vocabulary",
    .0.token(), .0.vocab()
  )]
  UnmaskableToken(crate::audio::whisper::decode::filter::UnmaskableToken),
}

/// A word-alignment matrix did not have the expected 2D shape, or its
/// flattened element count did not match `rows * cols`.
///
/// Payload of [`SegmentError::InvalidAlignmentShape`].
#[derive(Debug)]
pub struct InvalidAlignmentShape {
  /// Expected row count (text tokens).
  rows: usize,
  /// Expected column count (audio tokens).
  cols: usize,
  /// Actual flattened element count.
  len: usize,
}

// `len` is the flattened element count that FAILED to match `rows * cols`,
// not a container length, so an `is_empty` companion would be meaningless.
#[allow(clippy::len_without_is_empty)]
impl InvalidAlignmentShape {
  /// Construct from the expected row and column counts and the actual
  /// flattened element count.
  #[inline(always)]
  pub const fn new(rows: usize, cols: usize, len: usize) -> Self {
    Self { rows, cols, len }
  }

  /// Expected row count (text tokens).
  #[inline(always)]
  pub const fn rows(&self) -> usize {
    self.rows
  }

  /// Expected column count (audio tokens).
  #[inline(always)]
  pub const fn cols(&self) -> usize {
    self.cols
  }

  /// Actual flattened element count.
  #[inline(always)]
  pub const fn len(&self) -> usize {
    self.len
  }
}

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
/// whisper #41 exists to remove. `SwiftParity` is opt-in, so refusing an
/// environment where it cannot be honored costs a caller nothing they did
/// not explicitly ask for — a caller who prefers timings over parity simply
/// does not opt in, and gets `Complete` by default.
///
/// Payload of [`SegmentError::AlignmentPitchUnavailable`].
///
/// Unlike its sibling payloads this one derives [`std::error::Error`] and
/// owns both the variant's message and its `#[source]`: the variant is
/// `#[error(transparent)]`, which forwards `Display` *and* `source()`
/// straight through to this struct rather than inserting it as a link. The
/// rendered message and the error chain are therefore exactly what the
/// struct-shaped variant produced — one link, to the [`crate::TensorError`]
/// below — not one link deeper.
///
/// Note that the inherent [`source`](Self::source) getter returns the
/// concrete `&`[`crate::TensorError`] the struct pattern used to expose, and
/// so shadows [`std::error::Error::source`] for method-call syntax; call the
/// trait method by path (`std::error::Error::source(&e)`) to walk the chain.
#[derive(Debug, thiserror::Error)]
#[error(
  "cannot measure this host's CoreVideo Float16 row pitch for the {rows} x {cols} alignment \
   gather, so the Swift-parity gather cannot be reproduced (select \
   `AlignmentGather::Complete` to gather every row in full instead): {source}"
)]
pub struct AlignmentPitchUnavailable {
  /// Rows the gather would have allocated (gathered tokens).
  rows: usize,
  /// Columns the gather would have allocated (audio tokens).
  cols: usize,
  /// Why the probe allocation could not supply a pitch.
  #[source]
  source: crate::TensorError,
}

impl AlignmentPitchUnavailable {
  /// Construct from the rows and columns the gather would have allocated
  /// and the reason the probe allocation could not supply a pitch.
  #[inline(always)]
  pub const fn new(rows: usize, cols: usize, source: crate::TensorError) -> Self {
    Self { rows, cols, source }
  }

  /// Rows the gather would have allocated (gathered tokens).
  #[inline(always)]
  pub const fn rows(&self) -> usize {
    self.rows
  }

  /// Columns the gather would have allocated (audio tokens).
  #[inline(always)]
  pub const fn cols(&self) -> usize {
    self.cols
  }

  /// Why the probe allocation could not supply a pitch.
  #[inline(always)]
  pub const fn source(&self) -> &crate::TensorError {
    &self.source
  }
}

/// The probe allocation behind
/// [`AlignmentGather::SwiftParity`](crate::audio::whisper::options::AlignmentGather::SwiftParity)
/// succeeded but reported a layout the gather's model cannot describe —
/// anything other than `[pitch, 1]` element strides with `pitch >= cols`,
/// i.e. rows padded only *between* each other.
///
/// Fail-closed for the same reason as
/// [`SegmentError::AlignmentPitchUnavailable`]: an unmodellable layout means
/// the truncation Swift's gather performs is unknown, not absent.
///
/// Payload of [`SegmentError::AlignmentPitchUnexpectedLayout`].
#[derive(Debug)]
pub struct AlignmentPitchUnexpectedLayout {
  /// Rows the gather allocated (gathered tokens).
  rows: usize,
  /// Columns the gather allocated (audio tokens).
  cols: usize,
  /// The element strides the probe allocation actually reported.
  strides: Vec<usize>,
}

impl AlignmentPitchUnexpectedLayout {
  /// Construct from the rows and columns the gather allocated and the
  /// element strides the probe allocation actually reported.
  #[inline(always)]
  pub const fn new(rows: usize, cols: usize, strides: Vec<usize>) -> Self {
    Self {
      rows,
      cols,
      strides,
    }
  }

  /// Rows the gather allocated (gathered tokens).
  #[inline(always)]
  pub const fn rows(&self) -> usize {
    self.rows
  }

  /// Columns the gather allocated (audio tokens).
  #[inline(always)]
  pub const fn cols(&self) -> usize {
    self.cols
  }

  /// The element strides the probe allocation actually reported.
  #[inline(always)]
  pub fn strides(&self) -> &[usize] {
    &self.strides
  }
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
  #[error(
    "invalid alignment matrix shape: {} rows x {} cols, but data has {} elements",
    .0.rows(),
    .0.cols(),
    .0.len()
  )]
  InvalidAlignmentShape(InvalidAlignmentShape),
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
  /// whisper #41 exists to remove. `SwiftParity` is opt-in, so refusing an
  /// environment where it cannot be honored costs a caller nothing they did
  /// not explicitly ask for — a caller who prefers timings over parity simply
  /// does not opt in, and gets `Complete` by default.
  ///
  /// The message and the `#[source]` live on
  /// [`AlignmentPitchUnavailable`], which this variant forwards both of
  /// through `#[error(transparent)]`.
  #[error(transparent)]
  AlignmentPitchUnavailable(#[from] AlignmentPitchUnavailable),
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
    "this host's CoreVideo Float16 surface for the {} x {} alignment gather reports \
     element strides {:?}, which is not the row-padded row-major layout the Swift-parity \
     gather models (select `AlignmentGather::Complete` to gather every row in full instead)",
    .0.rows(),
    .0.cols(),
    .0.strides()
  )]
  AlignmentPitchUnexpectedLayout(AlignmentPitchUnexpectedLayout),
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
