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
//! struct ([`ContractMismatch`], [`OutputShape`], [`FrameCountOutOfRange`],
//! [`InvalidLogProbability`], [`NotADistribution`]) that the variant then
//! wraps. Struct-shaped enum variants are the shape this crate is moving away
//! from (the older doors still use them; that sweep is tracked separately), and
//! this door adds none. The practical gain is that a
//! payload is constructible, matchable, and `Display`-able on its own —
//! [`FrameCountOutOfRange`] in particular is the guard callers reach for most,
//! and it answers "how much audio may I pass?" without a live error in hand.

use super::ScorePooling;

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// The windowed-sequence engine's own error, re-exported because
/// [`Error::Windowing`] carries it (`audio::ced`'s convention).
pub use windit::WinditError;

/// A value offered to [`LogProbabilities::try_from_slice`] is not a natural-log
/// probability: it is NaN, or it is greater than zero.
///
/// `-∞` is deliberately ACCEPTED — it is the exact log of a zero probability,
/// which [`ScorePooling::Vote`] genuinely produces for a language no window
/// chose. Only NaN (which would sort silently under `total_cmp`) and positive
/// values (which no log-softmax output can take) are rejected.
///
/// [`LogProbabilities::try_from_slice`]: super::LogProbabilities::try_from_slice
/// [`ScorePooling::Vote`]: super::ScorePooling::Vote
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("log-probability at index {index} is {value}, which is not a value <= 0")]
pub struct InvalidLogProbability {
  index: usize,
  value: f32,
}

impl InvalidLogProbability {
  pub(crate) const fn new(index: usize, value: f32) -> Self {
    Self { index, value }
  }

  /// Position of the offending value in the supplied row.
  #[inline]
  pub const fn index(&self) -> usize {
    self.index
  }

  /// The offending value itself.
  #[inline]
  pub const fn value(&self) -> f32 {
    self.value
  }
}

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

/// Aggregation produced a row whose probabilities do not sum to 1, carrying the
/// [`ScorePooling`] that produced it and the mass it actually left.
///
/// Every pooling normalizes the row it folds, so this is a defect report rather
/// than a description of any input: it is the fold's postcondition catching an
/// arithmetic slip in the crate. The two it was written for are recorded in
/// `aggregate`'s tests — a normalizer that loses its constant to rounding
/// against a huge shift (mass 2), and a mixture that counts a window in its
/// denominator but not in its numerator (mass 0.5).
///
/// [`Error::ZeroMassAggregate`] is the one deviation that is NOT a defect — a
/// logarithmic pool over windows with disjoint supports honestly leaves nothing
/// — so it keeps its own variant and its own explanation.
///
/// [`ScorePooling`]: super::ScorePooling
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error(
  "{pooling:?} pooling produced a row whose probabilities sum to {mass}, not 1, \
   so it is not a distribution"
)]
pub struct NotADistribution {
  pooling: ScorePooling,
  mass: f64,
}

impl NotADistribution {
  pub(crate) const fn new(pooling: ScorePooling, mass: f64) -> Self {
    Self { pooling, mass }
  }

  /// The pooling whose fold produced the row.
  #[inline]
  pub const fn pooling(&self) -> ScorePooling {
    self.pooling
  }

  /// Total probability mass the row actually carries — `exp` summed over it.
  #[inline]
  pub const fn mass(&self) -> f64 {
    self.mass
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

  /// The windowing plan for a long clip could not be built — it exceeded
  /// [`WindowPlan::max_windows`] ([`WinditError::TooManyWindows`], `got`
  /// carrying the FULL planned count) or a span buffer could not be allocated
  /// ([`WinditError::AllocFailed`]).
  ///
  /// [`WinditError`] is `#[non_exhaustive]`, so match it with a wildcard arm.
  ///
  /// [`WindowPlan::max_windows`]: super::WindowPlan::max_windows
  #[error("windowing failed: {0}")]
  Windowing(#[from] WinditError),

  /// Aggregation was asked to fold an empty window list. Unreachable through
  /// [`Identifier::identify_long`] — a clip long enough to reach the model
  /// always plans at least one span — so this only reaches a caller who called
  /// [`aggregate_windows`] with an empty slice.
  ///
  /// [`Identifier::identify_long`]: super::Identifier::identify_long
  /// [`aggregate_windows`]: super::aggregate_windows
  #[error("cannot aggregate an empty window list")]
  EmptyWindows,

  /// Pooling produced a row that assigns probability zero to EVERY language,
  /// carrying the [`ScorePooling`] that produced it.
  ///
  /// Not an arithmetic slip. The logarithmic pool is a geometric mean, so a
  /// language ANY window scored at `-∞` is zero in the pool; windows certain of
  /// different languages therefore zero out every language between them, and
  /// the pool's honest answer is that nothing is possible. It is refused rather
  /// than returned because a row whose exponentials sum to zero is not a
  /// distribution: ranking it reports arbitrary languages at probability zero.
  ///
  /// Unreachable through [`Identifier::identify_long`] — a model row is
  /// all-finite, so no `-∞` enters the fold — and reachable through
  /// [`aggregate_windows`] only from hand-built rows, `-∞` being a value
  /// [`LogProbabilities::try_from_slice`] deliberately accepts. Each of those
  /// rows must still have been normalizable on its own: a window that ruled
  /// every language out is [`Error::UnnormalizableWindow`], refused before the
  /// fold.
  ///
  /// [`Identifier::identify_long`]: super::Identifier::identify_long
  /// [`aggregate_windows`]: super::aggregate_windows
  /// [`LogProbabilities::try_from_slice`]: super::LogProbabilities::try_from_slice
  #[error(
    "{0:?} pooling left no probability mass: every language pooled to probability \
     zero, so the result is not a distribution and its ranking would be arbitrary"
  )]
  ZeroMassAggregate(ScorePooling),

  /// A window offered to the fold has a maximum that is not FINITE, so no
  /// shift makes it a distribution and no pooling can fold it. Carries the
  /// window's position in the pushed sequence (its index in the slice
  /// [`aggregate_windows`] was given).
  ///
  /// Exactly two rows reach it:
  ///
  /// - `-∞` in EVERY column. Such a row is not evidence about which language
  ///   was spoken — it is the statement that none was — and no pooling has a
  ///   meaningful answer for it. The logarithmic pool zeroes the whole clip
  ///   out; the linear pool would count the window's duration in its
  ///   denominator while its terms contribute to no numerator, diluting every
  ///   other window; and [`ScorePooling::Vote`] would cast the window's vote
  ///   for whatever column the ranking tie-break surfaces, handing a share of
  ///   the clip to a language nothing chose.
  /// - `+∞` ANYWHERE. `exp` over such a row sums to `∞`, so it is not a
  ///   log-probability row at all and there is no constant that makes it one.
  ///
  /// What this is NOT is a judgement on a row's absolute SCALE. A row whose
  /// largest value is `-800` says exactly what one whose largest value is `0`
  /// says, column for column, and both are folded — the shift comes off before
  /// anything else, so how far from zero a row happens to sit is arithmetic
  /// noise rather than evidence and decides nothing here. See `aggregate`'s
  /// module docs, "A row's own scale is not evidence".
  ///
  /// Unreachable through [`Identifier::identify_long`]: a model row is a
  /// log-softmax, and [`Identifier::log_probabilities`] refuses a non-finite
  /// score outright. Reachable through [`aggregate_windows`] from hand-built
  /// rows, `-∞` being a value [`LogProbabilities::try_from_slice`] deliberately
  /// accepts; `+∞` is not, so that half guards this crate's own unvalidated
  /// internal constructor rather than a caller.
  ///
  /// [`Identifier::identify_long`]: super::Identifier::identify_long
  /// [`Identifier::log_probabilities`]: super::Identifier::log_probabilities
  /// [`aggregate_windows`]: super::aggregate_windows
  /// [`ScorePooling::Vote`]: super::ScorePooling::Vote
  /// [`LogProbabilities::try_from_slice`]: super::LogProbabilities::try_from_slice
  #[error(
    "window {0}'s largest log-probability is not finite, so no shift makes the row a \
     distribution: it is either -inf throughout, which rules every language out, or \
     +inf somewhere, which is not a log-probability row at all"
  )]
  UnnormalizableWindow(usize),

  /// The fold produced a row that is not a distribution — its probabilities do
  /// not sum to 1 — which is a defect in this crate rather than a property of
  /// the caller's rows. See [`NotADistribution`].
  #[error(transparent)]
  NotADistribution(#[from] NotADistribution),

  /// A hand-built log-probability row was not exactly
  /// [`NUM_LANGUAGES`](super::NUM_LANGUAGES) values long, carrying the length
  /// supplied.
  #[error("expected a row of exactly {n} log-probabilities, got {0}", n = super::NUM_LANGUAGES)]
  LanguageCountMismatch(usize),

  /// A hand-built log-probability row carried a value that is not a natural-log
  /// probability (NaN, or greater than zero).
  #[error(transparent)]
  InvalidLogProbability(#[from] InvalidLogProbability),

  /// A language index had no roster row, carrying that index. Defensive: the
  /// compile-time `NUM_LANGUAGES == languages().len()` assert makes this
  /// unreachable for in-range indices — a typed error, never a panic.
  #[error("language index {0} has no roster entry")]
  UnknownLanguageIndex(usize),
}

#[cfg(test)]
mod tests;
