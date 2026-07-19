//! The module's single error type and `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants;
//! tokenizer errors preserve their `#[source]` chain. Model-contract and
//! embedding-invariant failures are their own variants so callers can match on
//! cause. Mirrors `clap`'s error module, pared to the text-only surface (no
//! audio / window / aggregate variants). (Plain-text reference — granite builds
//! without the `clap` feature, so its docs must not link across it.)

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// Any failure loading the granite text embedder, running inference, tokenizing
/// text, or constructing an [`crate::embeddings::granite::Embedding`].
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
  /// `tests/granite/model_io.rs`).
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

  /// A model output component was NaN or infinite.
  #[error("model output contains a non-finite value at index {index}")]
  NonFiniteOutput {
    /// Flat index of the offending element.
    index: usize,
  },

  /// The caller passed an empty text string; there is nothing to embed.
  #[error("text input is empty")]
  EmptyText,

  /// An embedding slice did not have the expected dimension.
  #[error("embedding dimension mismatch: expected {expected}, got {got}")]
  EmbeddingDimMismatch {
    /// The required dimension ([`crate::embeddings::granite::embedding::EMBEDDING_DIM`]).
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

  /// A trusted-path embedding was not unit-norm within the module's norm budget
  /// (`crate::embeddings::granite::embedding::NORM_BUDGET`).
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
}

#[cfg(test)]
mod tests;
