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
  #[error("model contract mismatch on `{feature}`: expected {expected}, got {actual}")]
  ContractMismatch {
    /// Name of the input/output feature that mismatched.
    feature: &'static str,
    /// The contract this module expects, rendered for display.
    expected: String,
    /// What the loaded model actually declares, rendered for display.
    actual: String,
  },

  /// A predict-time output tensor's shape diverged from the contract validated
  /// at construction. [`crate::MultiArray::copy_into`] alone validates only
  /// total element count, so an axes-swapped output would otherwise pass
  /// silently — the CoreML runtime is re-checked on every call.
  #[error("output shape mismatch: expected {expected:?}, got {got:?}")]
  OutputShape {
    /// Shape the runtime tensor actually had.
    got: Vec<usize>,
    /// Shape the construction-time contract declares.
    expected: Vec<usize>,
  },

  /// A model output logit was NaN or infinite — model corruption, caught
  /// before it can poison sigmoid confidences or the ranking heap.
  #[error("model output contains a non-finite value at index {index}")]
  NonFiniteOutput {
    /// Flat index (class index) of the offending logit.
    index: usize,
  },

  /// The caller passed an empty clip; there is nothing to classify.
  #[error("audio input is empty")]
  EmptyAudio,

  /// A per-window input exceeded the fixed window. Never silently truncated —
  /// long clips are windowed explicitly (`classify_windows`/`classify_long`).
  #[error("audio input has {len} samples, exceeding the fixed {max}-sample window")]
  AudioTooLong {
    /// Number of samples the caller supplied.
    len: usize,
    /// The fixed window length (`WINDOW_SAMPLES`).
    max: usize,
  },

  /// An input sample was NaN or infinite (it would silently poison the mel).
  #[error("audio input contains a non-finite sample at index {index}")]
  NonFiniteInput {
    /// Index of the offending sample.
    index: usize,
  },

  /// `aggregate_windows` was called with an empty window slice; there is
  /// nothing to aggregate.
  #[error("no windows to aggregate")]
  EmptyWindows,

  /// A windowed-sequence operation failed inside the windit engine. Carries
  /// windit's own typed error unchanged ([`WinditError`] is
  /// `#[non_exhaustive]`, so match it with a wildcard arm). Not constructed by
  /// any Wave-A path (`WindowPlan::spans` is infallible by construction and
  /// windit's aggregation engine is deliberately unused) — the taxonomy's
  /// forward seam, spec §4.
  #[error("windowed-sequence processing failed: {0}")]
  Windowing(#[from] WinditError),

  /// A class index had no `RatedSoundEvent` row. Defensive: the compile-time
  /// `NUM_CLASSES == RatedSoundEvent::events().len()` assert makes
  /// `RatedSoundEvent::from_index` `None` unreachable for in-range indices —
  /// a typed error, never a panic (the granite `TokenCount` posture).
  #[error("class index {index} has no rated AudioSet event")]
  UnknownClassIndex {
    /// The offending class index.
    index: usize,
  },
}

#[cfg(test)]
mod tests;
