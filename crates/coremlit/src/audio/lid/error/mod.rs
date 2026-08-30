//! The module's single error type, its payload structs, and the `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants.
//! Model-contract, input-validation, and identification failures are their own
//! variants so callers can match on cause.
//!
//! # Why every payload is a newtype, not a struct variant
//!
//! [`Error`] deliberately carries **unit and newtype variants only**. A
//! multi-field payload lives in its own named, documented, accessor-bearing
//! struct ([`ContractMismatch`], [`OutputShape`], [`FrameCountOutOfRange`])
//! that the variant then wraps. Struct-shaped enum variants are the shape this
//! crate is moving away from (the older doors still use them; that sweep is
//! tracked separately), and this door adds none. The practical gain is that a
//! payload is constructible, matchable, and `Display`-able on its own —
//! [`FrameCountOutOfRange`] in particular is the guard callers reach for most,
//! and it answers "how much audio may I pass?" without a live error in hand.

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// A loaded model's input or output feature does not match the shape/dtype
/// contract this module was built against (the pinned ground truth lives in
/// `tests/lid/model_io.rs`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("model contract mismatch on `{feature}`: expected {expected}, got {actual}")]
pub struct ContractMismatch {
  feature: &'static str,
  expected: String,
  actual: String,
}

impl ContractMismatch {
  pub(crate) fn new(feature: &'static str, expected: String, actual: String) -> Self {
    Self {
      feature,
      expected,
      actual,
    }
  }

  /// Name of the input/output feature that mismatched.
  #[inline]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }

  /// The contract this module expects, rendered for display.
  #[inline]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// What the loaded model actually declares, rendered for display.
  #[inline]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// A predict-time output tensor's shape diverged from the contract validated at
/// construction. [`crate::MultiArray::copy_into`] alone validates only total
/// element count, so an axes-swapped output would otherwise pass silently — the
/// CoreML runtime is re-checked on every call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("output shape mismatch: expected {expected:?}, got {got:?}")]
pub struct OutputShape {
  got: Vec<usize>,
  expected: Vec<usize>,
}

impl OutputShape {
  pub(crate) fn new(got: Vec<usize>, expected: Vec<usize>) -> Self {
    Self { got, expected }
  }

  /// Shape the runtime tensor actually had.
  #[inline]
  pub fn got(&self) -> &[usize] {
    &self.got
  }

  /// Shape the construction-time contract declares.
  #[inline]
  pub fn expected(&self) -> &[usize] {
    &self.expected
  }
}

/// The clip's mel frame count falls outside the graph's accepted range.
///
/// Raised **before** the model is called, so the CoreML runtime's own
/// `"Size (9) of dimension (1) is not in allowed range (10..3001)"` never
/// reaches a caller: it names an internal axis index and would have to be
/// string-matched. This carries the same fact in the caller's own units —
/// samples as well as frames, and the sample bounds that would have been
/// accepted.
///
/// ```
/// use coremlit::audio::lid::{Error, FrameCountOutOfRange, MAX_SAMPLES, MIN_SAMPLES};
///
/// // Constructible without a model, so the bounds are readable up front.
/// let too_short = FrameCountOutOfRange::for_samples(MIN_SAMPLES - 1);
/// assert_eq!(too_short.samples(), 1_439);
/// assert_eq!(too_short.frames(), 9);
/// assert_eq!(too_short.min_samples(), MIN_SAMPLES);
/// assert_eq!(too_short.max_samples(), MAX_SAMPLES);
/// assert!(too_short.is_too_short());
///
/// let too_long = FrameCountOutOfRange::for_samples(MAX_SAMPLES + 1);
/// assert!(!too_long.is_too_short());
///
/// // The bounds travel with the error, so a caller never has to string-match
/// // the CoreML runtime's own axis-indexed complaint.
/// let rendered = Error::from(too_long).to_string();
/// assert!(rendered.contains("480160 samples"), "{rendered}");
/// assert!(rendered.contains("10..=3001 frames"), "{rendered}");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
  "audio has {samples} samples ({frames} mel frames), outside the model's accepted \
   {min_frames}..={max_frames} frames ({min_samples}..={max_samples} samples at 16 kHz)"
)]
pub struct FrameCountOutOfRange {
  samples: usize,
  frames: usize,
  min_frames: usize,
  max_frames: usize,
  min_samples: usize,
  max_samples: usize,
}

impl FrameCountOutOfRange {
  /// Describe the rejection of a clip of `samples` 16 kHz samples, filling in
  /// the frame count and both bound pairs from the module's own geometry.
  ///
  /// Public because the bounds are worth reading without provoking a failure;
  /// it does not itself check that `samples` is actually out of range.
  #[must_use]
  pub const fn for_samples(samples: usize) -> Self {
    Self {
      samples,
      frames: super::frame_count(samples),
      min_frames: super::MIN_FRAMES,
      max_frames: super::MAX_FRAMES,
      min_samples: super::MIN_SAMPLES,
      max_samples: super::MAX_SAMPLES,
    }
  }

  /// Number of samples the caller supplied.
  #[inline]
  pub const fn samples(&self) -> usize {
    self.samples
  }

  /// Mel frames those samples produce
  /// ([`frame_count`](super::frame_count)).
  #[inline]
  pub const fn frames(&self) -> usize {
    self.frames
  }

  /// Smallest frame count the graph accepts ([`MIN_FRAMES`](super::MIN_FRAMES)).
  #[inline]
  pub const fn min_frames(&self) -> usize {
    self.min_frames
  }

  /// Largest frame count the graph accepts ([`MAX_FRAMES`](super::MAX_FRAMES)).
  #[inline]
  pub const fn max_frames(&self) -> usize {
    self.max_frames
  }

  /// Smallest sample count the graph accepts
  /// ([`MIN_SAMPLES`](super::MIN_SAMPLES)).
  #[inline]
  pub const fn min_samples(&self) -> usize {
    self.min_samples
  }

  /// Largest sample count the graph accepts
  /// ([`MAX_SAMPLES`](super::MAX_SAMPLES)).
  #[inline]
  pub const fn max_samples(&self) -> usize {
    self.max_samples
  }

  /// Whether the clip was too SHORT (as opposed to too long) — the two
  /// rejections call for opposite fixes, so callers should not have to
  /// re-derive which one they got.
  #[inline]
  pub const fn is_too_short(&self) -> bool {
    self.frames < self.min_frames
  }
}

/// Any failure loading the language identifier, running inference, or
/// constructing scores.
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

  /// A loaded model's I/O contract does not match this module's.
  #[error(transparent)]
  ContractMismatch(#[from] ContractMismatch),

  /// A predict-time output tensor's shape diverged from the contract validated
  /// at construction.
  #[error(transparent)]
  OutputShape(#[from] OutputShape),

  /// The clip's mel frame count falls outside the graph's accepted range —
  /// raised before the model is called.
  #[error(transparent)]
  FrameCountOutOfRange(#[from] FrameCountOutOfRange),

  /// An input sample was NaN or infinite, carrying its index (it would
  /// silently poison the mel).
  #[error("audio input contains a non-finite sample at index {0}")]
  NonFiniteInput(usize),

  /// A model output log-probability was NaN or infinite, carrying its language
  /// index — model corruption, caught before it can reach the ranking heap
  /// (where `total_cmp` would silently sort a NaN) or `exp`.
  #[error("model output contains a non-finite log-probability at index {0}")]
  NonFiniteOutput(usize),

  /// A language index had no roster row, carrying that index. Defensive: the
  /// compile-time `NUM_LANGUAGES == languages().len()` assert makes this
  /// unreachable for in-range indices — a typed error, never a panic.
  #[error("language index {0} has no roster entry")]
  UnknownLanguageIndex(usize),
}

#[cfg(test)]
mod tests;
