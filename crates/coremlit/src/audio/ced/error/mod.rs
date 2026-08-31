//! The module's single error type and `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants.
//! Model-contract, input-validation, and classification failures are their own
//! variants so callers can match on cause. Mirrors granite's error module,
//! re-cut for the audio-classifier surface (input validation gains the clap
//! audio variants; the embedding variants are gone). (Plain-text references —
//! ced builds without the `granite`/`clap` features, so its docs must not link
//! across them.)

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// Re-exported so callers (and tests) can name and match the typed error
/// [`Error::Windowing`] carries from the windit windowed-sequence engine.
pub use windit::WinditError;

/// A loaded model's input or output feature does not match the shape/dtype
/// contract this module was built against (the pinned ground truth lives in
/// `tests/ced/model_io.rs`).
///
/// Payload of [`Error::ContractMismatch`].
#[derive(Debug)]
pub struct ContractMismatch {
  /// Name of the input/output feature that mismatched.
  feature: &'static str,
  /// The contract this module expects, rendered for display.
  expected: String,
  /// What the loaded model actually declares, rendered for display.
  actual: String,
}

impl ContractMismatch {
  /// Construct from the mismatched feature, the expected contract, and what
  /// the loaded model actually declares.
  #[inline(always)]
  pub const fn new(feature: &'static str, expected: String, actual: String) -> Self {
    Self {
      feature,
      expected,
      actual,
    }
  }

  /// Name of the input/output feature that mismatched.
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }

  /// The contract this module expects, rendered for display.
  #[inline(always)]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// What the loaded model actually declares, rendered for display.
  #[inline(always)]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// A predict-time output tensor's shape diverged from the contract validated
/// at construction. [`crate::MultiArray::copy_into`] alone validates only
/// total element count, so an axes-swapped output would otherwise pass
/// silently — the CoreML runtime is re-checked on every call.
///
/// Payload of [`Error::OutputShape`].
#[derive(Debug)]
pub struct OutputShape {
  /// Shape the runtime tensor actually had.
  got: Vec<usize>,
  /// Shape the construction-time contract declares.
  expected: Vec<usize>,
}

impl OutputShape {
  /// Construct from the runtime tensor's shape and the shape the
  /// construction-time contract declares.
  #[inline(always)]
  pub const fn new(got: Vec<usize>, expected: Vec<usize>) -> Self {
    Self { got, expected }
  }

  /// Shape the runtime tensor actually had.
  #[inline(always)]
  pub fn got(&self) -> &[usize] {
    &self.got
  }

  /// Shape the construction-time contract declares.
  #[inline(always)]
  pub fn expected(&self) -> &[usize] {
    &self.expected
  }
}

/// A per-window input exceeded the fixed window. Never silently truncated —
/// long clips are windowed explicitly (`classify_windows`/`classify_long`).
///
/// Payload of [`Error::AudioTooLong`].
#[derive(Debug)]
pub struct AudioTooLong {
  /// Number of samples the caller supplied.
  len: usize,
  /// The fixed window length (`WINDOW_SAMPLES`).
  max: usize,
}

// `len` names the sample count the caller SUPPLIED, against the fixed window
// bound it overran — not a collection length this payload owns, so there is
// nothing for an `is_empty` to mean here.
#[allow(clippy::len_without_is_empty)]
impl AudioTooLong {
  /// Construct from the samples the caller supplied and the fixed window
  /// length they exceeded.
  #[inline(always)]
  pub const fn new(len: usize, max: usize) -> Self {
    Self { len, max }
  }

  /// Number of samples the caller supplied.
  #[inline(always)]
  pub const fn len(&self) -> usize {
    self.len
  }

  /// The fixed window length (`WINDOW_SAMPLES`).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A caller-built confidence vector did not have exactly one entry per class.
///
/// Payload of [`Error::ClassCountMismatch`].
#[derive(Debug)]
pub struct ClassCountMismatch {
  /// The required length ([`NUM_CLASSES`](crate::audio::ced::NUM_CLASSES)).
  expected: usize,
  /// The length the caller supplied.
  got: usize,
}

impl ClassCountMismatch {
  /// Construct from the required length and the length the caller supplied.
  #[inline(always)]
  pub const fn new(expected: usize, got: usize) -> Self {
    Self { expected, got }
  }

  /// The required length ([`NUM_CLASSES`](crate::audio::ced::NUM_CLASSES)).
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// The length the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }
}

/// A caller-built confidence was not a finite value in `[0, 1]`. Same origin
/// as [`Error::ClassCountMismatch`]: the model path gets the range for free
/// from sigmoid, so only a hand-built vector can violate it, and
/// [`Confidences`](crate::audio::ced::Confidences) states that range as its
/// invariant.
///
/// Payload of [`Error::InvalidConfidence`].
#[derive(Debug)]
pub struct InvalidConfidence {
  /// Class index of the offending value.
  index: usize,
  /// The value the caller supplied.
  value: f32,
}

impl InvalidConfidence {
  /// Construct from the offending class index and the value the caller
  /// supplied.
  #[inline(always)]
  pub const fn new(index: usize, value: f32) -> Self {
    Self { index, value }
  }

  /// Class index of the offending value.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }

  /// The value the caller supplied.
  #[inline(always)]
  pub const fn value(&self) -> f32 {
    self.value
  }
}

/// Any failure loading the CED classifier, running inference, or constructing
/// predictions.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// The CoreML runtime failed to load a compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] crate::LoadError),

  /// A CoreML prediction call failed.
  #[error("prediction failed: {0}")]
  Prediction(#[from] crate::PredictionError),

  /// A tensor failed to construct or view.
  #[error("tensor failed: {0}")]
  Tensor(#[from] crate::TensorError),

  /// A loaded model's input or output feature does not match the shape/dtype
  /// contract this module was built against (the pinned ground truth lives in
  /// `tests/ced/model_io.rs`).
  #[error(
    "model contract mismatch on `{}`: expected {}, got {}",
    .0.feature(),
    .0.expected(),
    .0.actual()
  )]
  ContractMismatch(ContractMismatch),

  /// A predict-time output tensor's shape diverged from the contract validated
  /// at construction. [`crate::MultiArray::copy_into`] alone validates only
  /// total element count, so an axes-swapped output would otherwise pass
  /// silently — the CoreML runtime is re-checked on every call.
  #[error("output shape mismatch: expected {:?}, got {:?}", .0.expected(), .0.got())]
  OutputShape(OutputShape),

  /// A model output logit was NaN or infinite — model corruption, caught
  /// before it can poison sigmoid confidences or the ranking heap.
  ///
  /// Carries the flat index (class index) of the offending logit.
  #[error("model output contains a non-finite value at index {0}")]
  NonFiniteOutput(usize),

  /// The caller passed an empty clip; there is nothing to classify.
  #[error("audio input is empty")]
  EmptyAudio,

  /// A per-window input exceeded the fixed window. Never silently truncated —
  /// long clips are windowed explicitly (`classify_windows`/`classify_long`).
  #[error("audio input has {} samples, exceeding the fixed {}-sample window", .0.len(), .0.max())]
  AudioTooLong(AudioTooLong),

  /// An input sample was NaN or infinite (it would silently poison the mel).
  ///
  /// Carries the index of the offending sample.
  #[error("audio input contains a non-finite sample at index {0}")]
  NonFiniteInput(usize),

  /// `aggregate_windows` was called with an empty window slice; there is
  /// nothing to aggregate.
  #[error("no windows to aggregate")]
  EmptyWindows,

  /// A caller-built confidence vector did not have exactly one entry per
  /// class. Raised only by
  /// [`Confidences::try_from_slice`](crate::audio::ced::Confidences::try_from_slice) —
  /// the model path is shape-checked at the graph boundary long before it
  /// reaches confidence space, so this is the hand-built path's error and no
  /// other.
  #[error(
    "confidence vector has {} values, expected exactly {} (one per class)",
    .0.got(),
    .0.expected()
  )]
  ClassCountMismatch(ClassCountMismatch),

  /// A caller-built confidence was not a finite value in `[0, 1]`. Same origin
  /// as [`Self::ClassCountMismatch`]: the model path gets the range for free
  /// from sigmoid, so only a hand-built vector can violate it, and
  /// [`Confidences`](crate::audio::ced::Confidences) states that range as its
  /// invariant.
  #[error(
    "confidence at class index {} is {}, not a finite value in [0, 1]",
    .0.index(),
    .0.value()
  )]
  InvalidConfidence(InvalidConfidence),

  /// A windowed-sequence operation failed. Carries windit's own typed error
  /// unchanged ([`WinditError`] is `#[non_exhaustive]`, so match it with a
  /// wildcard arm). Constructed by the `WindowPlan::spans` resource rail:
  /// (a) [`WinditError::TooManyWindows`] manufactured by the O(1) cap pre-check
  /// when the planned window count exceeds `max_windows`, whose `got` is the
  /// FULL planned count — deviating from windit's own abort-at-`max + 1`
  /// convention, matching granite's post-windit raise; (b)
  /// [`WinditError::AllocFailed`] propagated from windit's planner (including its
  /// defense-in-depth cap); (c) [`WinditError::AllocFailed`] manufactured when a
  /// span or per-window result buffer's `try_reserve_exact` is refused.
  #[error("windowed-sequence processing failed: {0}")]
  Windowing(#[from] WinditError),

  /// A class index had no `RatedSoundEvent` row. Defensive: the compile-time
  /// `NUM_CLASSES == RatedSoundEvent::events().len()` assert makes
  /// `RatedSoundEvent::from_index` `None` unreachable for in-range indices —
  /// a typed error, never a panic (the granite `TokenCount` posture).
  ///
  /// Carries the offending class index.
  #[error("class index {0} has no rated AudioSet event")]
  UnknownClassIndex(usize),
}

#[cfg(test)]
mod tests;
