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
  #[error("model contract mismatch on `{feature}`: expected {expected}, got {actual}")]
  ContractMismatch {
    /// Name of the input/output feature that mismatched.
    feature: &'static str,
    /// The contract this crate expects, rendered for display.
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

  /// The caller's audio input contained a NaN or infinite value before
  /// inference ran. An unchecked non-finite sample would otherwise propagate
  /// through the mel front-end into a finite-looking but garbage embedding.
  #[error("audio input contains a non-finite value at index {index}")]
  NonFiniteInput {
    /// Flat index of the offending sample.
    index: usize,
  },

  /// A model output component was NaN or infinite.
  #[error("model output contains a non-finite value at index {index}")]
  NonFiniteOutput {
    /// Flat index of the offending element.
    index: usize,
  },

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
    "audio window has {len} samples, over the {max}-sample per-window limit; use \
     `AudioEncoder::embed_windows` for long audio"
  )]
  AudioTooLong {
    /// Number of samples the caller supplied.
    len: usize,
    /// The per-window limit ([`TARGET_SAMPLES`](crate::embeddings::clap::audio::TARGET_SAMPLES)).
    max: usize,
  },

  /// The caller passed an empty text string; there is nothing to embed.
  #[error("text input is empty")]
  EmptyText,

  /// An embedding slice did not have the expected dimension.
  #[error("embedding dimension mismatch: expected {expected}, got {got}")]
  EmbeddingDimMismatch {
    /// The required dimension ([`crate::embeddings::clap::embedding::EMBEDDING_DIM`]).
    expected: usize,
    /// The dimension the caller supplied.
    got: usize,
  },

  /// An embedding component was NaN or infinite.
  #[error("embedding contains a non-finite value at component {component_index}")]
  NonFiniteEmbedding {
    /// Index of the offending component.
    component_index: usize,
  },

  /// An embedding to be normalized had zero magnitude (undefined direction).
  #[error("embedding has zero magnitude and cannot be normalized")]
  EmbeddingZero,

  /// A trusted-path embedding was not unit-norm within the crate's norm budget
  /// (`crate::embeddings::clap::embedding::NORM_BUDGET`).
  #[error("embedding is not unit-norm: |norm² − 1| = {norm_sq_deviation}")]
  EmbeddingNotUnitNorm {
    /// `(norm² − 1).abs()`, the amount by which the invariant was violated.
    norm_sq_deviation: f32,
  },

  /// The tokenizer failed to load from its JSON definition.
  #[error("failed to load tokenizer: {0}")]
  TokenizerLoad(#[source] tokenizers::Error),

  /// Configuring the tokenizer (truncation) failed.
  #[error("failed to configure tokenizer: {0}")]
  TokenizerConfig(#[source] tokenizers::Error),

  /// Encoding text into token ids failed.
  #[error("failed to tokenize text: {0}")]
  Tokenize(#[source] tokenizers::Error),

  /// An [`crate::embeddings::clap::aggregate::AggregatePolicy`] was asked to combine zero window
  /// embeddings. Every policy needs at least one window to produce a direction;
  /// the caller should skip aggregation (or handle the empty clip) instead. This
  /// is the one windit error given its own clap variant:
  /// [`From<WinditError>`](Error::from) maps [`WinditError::Empty`] here so the
  /// pinned empty-aggregation taxonomy is stable across the windit port.
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
  #[error("windowed-sequence processing failed: {0}")]
  Windowing(#[source] WinditError),
}

impl From<WinditError> for Error {
  /// The ONE outward translation from windit into clap's taxonomy.
  /// [`WinditError::Empty`] maps onto the pinned [`Error::EmptyWindows`] variant
  /// (same meaning, kept so the empty-aggregation taxonomy is stable across the
  /// port); every other windit error is wrapped losslessly in
  /// [`Error::Windowing`].
  fn from(e: WinditError) -> Self {
    match e {
      WinditError::Empty => Error::EmptyWindows,
      other => Error::Windowing(other),
    }
  }
}

#[cfg(test)]
mod tests;
