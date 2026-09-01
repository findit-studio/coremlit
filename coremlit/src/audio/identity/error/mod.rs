//! The module's single error type and `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants;
//! model-contract and input-validation failures are their own variants so
//! callers can match on cause. Cut from `audio::ced`'s error module — the
//! nearest door in shape (a Rust mel front end in front of one fixed-shape
//! `mel` graph) — with the classifier-specific variants dropped and
//! [`WindowLength`] added, because this door's window is **exact** rather than
//! a ceiling (see [`WindowLength`] for why padding is refused rather than
//! performed). Plain-text cross-references only: this module builds without the
//! `ced` / `speaker` features, so its docs must not link across them.

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// A loaded model's input or output feature does not match the shape/dtype
/// contract this module was built against (the pinned ground truth lives in
/// `tests/identity/model_io.rs`).
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

/// A predict-time output tensor's shape diverged from the contract validated at
/// construction. [`crate::MultiArray::copy_into`] alone validates only total
/// element count, so an axes-swapped output would otherwise pass silently — the
/// CoreML runtime is re-checked on every call.
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

/// The caller's clip was not exactly one window long.
///
/// Payload of [`Error::WindowLength`].
///
/// # Why neither padding nor truncation is offered
///
/// The other audio doors in this crate accept a short clip and pad it, because
/// their front ends are *local*: a padded frame changes that frame and no
/// other. This one's is not. Its last stage subtracts, per mel bin, the mean
/// over all `N_FRAMES` frames, so appending silence pulls every bin's mean
/// toward `ln(1e-6)` and shifts every REAL frame's value — a padded clip is a
/// different function of the speech, not a truncated one, and the shift grows
/// with the amount padded. Nothing in the conversion recipe measured that
/// regime; its whole evidence base is the exact 6 s window.
///
/// So a caller who has less than a window decides what to do about it —
/// gather more audio, tile the clip, or decline to enrol — and a caller who has
/// more windows it explicitly. Neither choice is one this crate can make
/// silently on their behalf, which is what a default padding policy would be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowLength {
  /// Samples the caller supplied.
  got: usize,
  /// The exact window length (`WINDOW_SAMPLES`).
  expected: usize,
}

impl WindowLength {
  /// Construct from the supplied and required sample counts.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// Samples the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The exact window length (`WINDOW_SAMPLES`).
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// Any failure loading the identity embedder, running inference, or computing
/// its mel front end.
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
  /// `tests/identity/model_io.rs`).
  #[error(
    "model contract mismatch on `{}`: expected {}, got {}",
    .0.feature(),
    .0.expected(),
    .0.actual()
  )]
  ContractMismatch(ContractMismatch),

  /// A predict-time output tensor's shape diverged from the contract validated
  /// at construction.
  #[error("output shape mismatch: expected {:?}, got {:?}", .0.expected(), .0.got())]
  OutputShape(OutputShape),

  /// The caller's clip was not exactly one window long — never padded and
  /// never truncated; see [`WindowLength`] for why.
  #[error(
    "audio input has {} samples, but this door takes exactly {} (one window); \
     it is neither padded nor truncated",
    .0.got(),
    .0.expected()
  )]
  WindowLength(WindowLength),

  /// An input sample was NaN or infinite (it would silently poison the mel,
  /// and a NaN mel bin propagates through the whole embedding).
  ///
  /// Carries the index of the offending sample.
  #[error("audio input contains a non-finite sample at index {0}")]
  NonFiniteInput(usize),

  /// The loaded graph declares a REQUIRED input this door never supplies, so
  /// every prediction through it would fail.
  ///
  /// Carries the offending feature name. An OPTIONAL extra input is not this:
  /// CoreML runs a prediction that omits one, so only a required input the
  /// caller cannot fill makes the contract unsatisfiable.
  #[error(
    "model declares a required input `{0}` that this door never supplies; \
     it sends `mel` and nothing else, so every prediction would fail"
  )]
  UnsatisfiableInput(String),

  /// The loaded graph declares CoreML STATE buffers, and this door predicts
  /// through the stateless API.
  ///
  /// Carries the offending state feature name. A stateful model must receive an
  /// `MLState` on every prediction; a door that never makes one either fails
  /// the prediction outright or silently discards the persistence the graph was
  /// built around. Neither is something to discover at predict time.
  #[error(
    "model declares the state buffer `{0}`, and this door predicts through the \
     stateless API; a stateful graph needs an `MLState` on every prediction"
  )]
  UnsatisfiableState(String),

  /// A model output component was NaN or infinite — model corruption, caught
  /// before the raw vector reaches a caller's normalization, where it would
  /// turn the whole embedding into NaNs.
  ///
  /// Carries the index of the offending component.
  #[error("model output contains a non-finite value at index {0}")]
  NonFiniteOutput(usize),
}

#[cfg(test)]
mod tests;
