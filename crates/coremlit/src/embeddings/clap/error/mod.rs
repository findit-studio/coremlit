//! The crate's single error type and `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants;
//! tokenizer errors preserve their `#[source]` chain. Model-contract and
//! embedding-invariant failures are their own variants so callers can match on
//! cause.

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// Re-exported so callers (and tests) can name and match the typed error
/// [`Error::Windowing`] carries from the windit windowed-sequence engine (the
/// long-audio window geometry and aggregation).
pub use windit::WinditError;

/// A loaded model's input or output feature does not match the shape/dtype
/// contract this crate was built against (the pinned ground truth lives in
/// `tests/clap/model_io.rs` / `tests/clap/text_model_io.rs`).
///
/// Payload of [`Error::ContractMismatch`].
#[derive(Debug)]
pub struct ContractMismatch {
  /// Name of the input/output feature that mismatched.
  feature: &'static str,
  /// The contract this crate expects, rendered for display.
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

  /// The contract this crate expects, rendered for display.
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

/// [`AudioEncoder::embed_window`](crate::embeddings::clap::AudioEncoder::embed_window) received
/// more than [`TARGET_SAMPLES`](crate::embeddings::clap::audio::TARGET_SAMPLES) samples. That
/// method embeds exactly one fixed 480 000-sample window, so a longer clip must
/// be hopped into windows by
/// [`AudioEncoder::embed_windows`](crate::embeddings::clap::AudioEncoder::embed_windows) (the
/// long-audio pipeline) rather than silently head-truncated: HF's
/// `ClapFeatureExtractor` is configured for `rand_trunc`, so truncating a longer
/// clip here would be both non-deterministic and unfaithful to HF, which clapkit
/// refuses to do behind the caller's back.
///
/// Payload of [`Error::AudioTooLong`].
#[derive(Debug)]
pub struct AudioTooLong {
  /// Number of samples the caller supplied.
  len: usize,
  /// The per-window limit ([`TARGET_SAMPLES`](crate::embeddings::clap::audio::TARGET_SAMPLES)).
  max: usize,
}

// `len` names the sample count the caller SUPPLIED, against the per-window
// bound it overran — not a collection length this payload owns, so there is
// nothing for an `is_empty` to mean here.
#[allow(clippy::len_without_is_empty)]
impl AudioTooLong {
  /// Construct from the samples the caller supplied and the per-window limit
  /// they exceeded.
  #[inline(always)]
  pub const fn new(len: usize, max: usize) -> Self {
    Self { len, max }
  }

  /// Number of samples the caller supplied.
  #[inline(always)]
  pub const fn len(&self) -> usize {
    self.len
  }

  /// The per-window limit ([`TARGET_SAMPLES`](crate::embeddings::clap::audio::TARGET_SAMPLES)).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// An embedding slice did not have the expected dimension.
///
/// Payload of [`Error::EmbeddingDimMismatch`].
#[derive(Debug)]
pub struct EmbeddingDimMismatch {
  /// The required dimension ([`crate::embeddings::clap::embedding::EMBEDDING_DIM`]).
  expected: usize,
  /// The dimension the caller supplied.
  got: usize,
}

impl EmbeddingDimMismatch {
  /// Construct from the required dimension and the dimension the caller
  /// supplied.
  #[inline(always)]
  pub const fn new(expected: usize, got: usize) -> Self {
    Self { expected, got }
  }

  /// The required dimension ([`crate::embeddings::clap::embedding::EMBEDDING_DIM`]).
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// The dimension the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }
}

/// Any failure loading a CLAP encoder, running inference, tokenizing text, or
/// constructing an [`crate::embeddings::clap::Embedding`].
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
  /// contract this crate was built against (the pinned ground truth lives in
  /// `tests/clap/model_io.rs` / `tests/clap/text_model_io.rs`).
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

  /// The caller's audio input contained a NaN or infinite value before
  /// inference ran. An unchecked non-finite sample would otherwise propagate
  /// through the mel front-end into a finite-looking but garbage embedding.
  ///
  /// Carries the flat index of the offending sample.
  #[error("audio input contains a non-finite value at index {0}")]
  NonFiniteInput(usize),

  /// A model output component was NaN or infinite.
  ///
  /// Carries the flat index of the offending element.
  #[error("model output contains a non-finite value at index {0}")]
  NonFiniteOutput(usize),

  /// The caller passed an empty audio slice; there is nothing to embed.
  #[error("audio input is empty")]
  EmptyAudio,

  /// [`AudioEncoder::embed_window`](crate::embeddings::clap::AudioEncoder::embed_window) received
  /// more than [`TARGET_SAMPLES`](crate::embeddings::clap::audio::TARGET_SAMPLES) samples. That
  /// method embeds exactly one fixed 480 000-sample window, so a longer clip must
  /// be hopped into windows by
  /// [`AudioEncoder::embed_windows`](crate::embeddings::clap::AudioEncoder::embed_windows) (the
  /// long-audio pipeline) rather than silently head-truncated: HF's
  /// `ClapFeatureExtractor` is configured for `rand_trunc`, so truncating a longer
  /// clip here would be both non-deterministic and unfaithful to HF, which clapkit
  /// refuses to do behind the caller's back.
  #[error(
    "audio window has {} samples, over the {}-sample per-window limit; use \
     `AudioEncoder::embed_windows` for long audio",
    .0.len(),
    .0.max()
  )]
  AudioTooLong(AudioTooLong),

  /// The caller passed an empty text string; there is nothing to embed.
  #[error("text input is empty")]
  EmptyText,

  /// An embedding slice did not have the expected dimension.
  #[error("embedding dimension mismatch: expected {}, got {}", .0.expected(), .0.got())]
  EmbeddingDimMismatch(EmbeddingDimMismatch),

  /// An embedding component was NaN or infinite.
  ///
  /// Carries the index of the offending component.
  #[error("embedding contains a non-finite value at component {0}")]
  NonFiniteEmbedding(usize),

  /// An embedding to be normalized had zero magnitude (undefined direction).
  #[error("embedding has zero magnitude and cannot be normalized")]
  EmbeddingZero,

  /// A trusted-path embedding was not unit-norm within the crate's norm budget
  /// (`crate::embeddings::clap::embedding::NORM_BUDGET`).
  ///
  /// Carries `(norm² − 1).abs()`, the amount by which the invariant was
  /// violated.
  #[error("embedding is not unit-norm: |norm² − 1| = {0}")]
  EmbeddingNotUnitNorm(f32),

  /// The tokenizer failed to load from its JSON definition.
  #[error("failed to load tokenizer: {0}")]
  TokenizerLoad(#[source] tokenizers::Error),

  /// Configuring the tokenizer (truncation) failed.
  #[error("failed to configure tokenizer: {0}")]
  TokenizerConfig(#[source] tokenizers::Error),

  /// Encoding text into token ids failed.
  #[error("failed to tokenize text: {0}")]
  Tokenize(#[source] tokenizers::Error),

  /// [`aggregate`](crate::embeddings::clap::aggregate::aggregate) was asked to
  /// combine zero window embeddings. Every policy needs at least one window to
  /// produce a direction; the caller should skip aggregation (or handle the
  /// empty clip) instead.
  ///
  /// This variant is produced if and only if the `windows` slice passed to
  /// [`aggregate`](crate::embeddings::clap::aggregate::aggregate) was itself
  /// empty — checked before windit's engine (and therefore before any policy)
  /// ever runs. It is NOT produced by matching windit's returned error: a
  /// custom [`AggregatePolicy`](crate::embeddings::clap::aggregate::AggregatePolicy)
  /// may itself return [`WinditError::Empty`] from `aggregate_values` for a
  /// NONEMPTY `windows` slice (reporting an aggregation failure, not "there
  /// were no windows"), and that reaches the caller as [`Error::Windowing`],
  /// never as this variant. The blanket [`From<WinditError>`](Error::from) impl
  /// does NOT produce this variant either — it is total and never
  /// special-cases a variant — so nothing that returns a bare [`WinditError`]
  /// (a downstream `?` included) can produce it.
  #[error("cannot aggregate zero window embeddings")]
  EmptyWindows,

  /// A windowed-sequence operation failed inside the windit engine (an
  /// aggregation domain / determinacy gate, geometry validation, or an allocator
  /// refusal). Carries windit's own typed error unchanged ([`WinditError`] is
  /// `#[non_exhaustive]`, so match it with a wildcard arm). Notably
  /// `WinditError::NonFinite` here is windit's determinacy gate — an aggregate
  /// whose windows cancel exactly has no direction at working precision (the
  /// pre-windit code reported the same condition as [`Error::EmbeddingZero`]);
  /// and `WinditError::AlphaOutOfRange` is an out-of-range [`EmaRenormalized`](crate::embeddings::clap::aggregate::EmaRenormalized)
  /// smoothing factor.
  ///
  /// It is also the window planner's resource rail (see
  /// [`WindowPlan::spans`](crate::embeddings::clap::window::WindowPlan::spans)):
  /// (a) [`WinditError::TooManyWindows`] manufactured by the O(1) cap pre-check
  /// when the planned window count exceeds
  /// [`max_windows`](crate::embeddings::clap::window::WindowPlan::max_windows), whose `got` is
  /// the FULL planned count — deviating from windit's own abort-at-`max + 1`
  /// convention, matching granite's post-windit raise; and (b)
  /// [`WinditError::AllocFailed`] propagated from windit's planner or manufactured
  /// when the multi-tail continuation's `try_reserve_exact` is refused. This is
  /// what makes an untrusted clip length + hop a typed refusal rather than an
  /// unbounded allocation or a panic.
  #[error("windowed-sequence processing failed: {0}")]
  Windowing(#[source] WinditError),
}

impl From<WinditError> for Error {
  /// A total, lossless wrap of any [`WinditError`] into [`Error::Windowing`] —
  /// this impl makes NO special case for any variant, `Empty` included.
  ///
  /// It has to be total *because* it is reached by more than the call this
  /// crate writes: [`SmoothPolicy`](crate::embeddings::clap::smooth::SmoothPolicy)
  /// and [`Smoother`](crate::embeddings::clap::smooth::Smoother) are
  /// re-exported with their windit method signatures intact (returning
  /// [`WinditError`] directly), so a downstream caller's own
  /// `policy.smooth(&windows)?` or `smoother.push(w)?` — in a function
  /// returning this crate's [`Result`] — lifts through this same impl on a
  /// plain `?`, no different from any in-crate call. A special case here (e.g.
  /// collapsing [`WinditError::Empty`] onto [`Error::EmptyWindows`]) would
  /// silently reinterpret every such caller's error under `aggregate`'s
  /// taxonomy, including callers with nothing to do with aggregation — the
  /// [`smooth`](crate::embeddings::clap::smooth::smooth) wrapper had exactly
  /// this bug until its own call site stopped routing through this impl; going
  /// through the public re-exports instead of that wrapper reached the same
  /// special case regardless.
  ///
  /// [`aggregate`](crate::embeddings::clap::aggregate::aggregate) is the one
  /// place that reports [`Error::EmptyWindows`], but it does not reach that
  /// variant through this impl, and not by matching a RETURNED
  /// [`WinditError::Empty`] either: that variant alone cannot distinguish "the
  /// engine saw no windows" from "the policy refused" (a custom policy may
  /// return `Empty` for a NONEMPTY `windows` slice). It checks its own
  /// `windows` argument before ever calling into windit, and reports
  /// [`Error::EmptyWindows`] directly when that argument is empty — leaving
  /// every [`WinditError`] windit itself returns, `Empty` included, to this
  /// impl.
  fn from(e: WinditError) -> Self {
    Error::Windowing(e)
  }
}

#[cfg(test)]
mod tests;
