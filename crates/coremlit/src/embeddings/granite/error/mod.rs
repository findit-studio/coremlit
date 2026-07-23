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

/// Re-exported so callers (and tests) can name and match the typed error
/// [`Error::Windowing`] carries from the windit windowed-sequence engine
/// (`embed_long`'s content-aware chunking and window aggregation).
pub use windit::WinditError;

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

  /// The tokenized input exceeded the fixed
  /// [`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS) window. Every
  /// constructor forces truncation at that length and disables the tokenizer's
  /// own padding, so this is a defensive backstop — returned instead of an
  /// out-of-bounds panic — against a tokenizer that still yields more ids than
  /// the window (e.g. a padding policy that survived configuration).
  #[error("tokenized input has {got} tokens, exceeding the fixed {max}-token window")]
  TokenCount {
    /// Number of token ids the tokenizer produced.
    got: usize,
    /// The fixed window length
    /// ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
    max: usize,
  },

  /// A token id did not fit the model's `int32` `input_ids` tensor. granite's
  /// vocabulary is far below `i32::MAX`, so this only fires for a foreign
  /// tokenizer with an out-of-range id — returned instead of a silently
  /// wrapping cast.
  #[error("token id {id} exceeds the model's int32 input range")]
  TokenIdRange {
    /// The offending token id.
    id: u32,
  },

  /// A windowed-sequence operation ([`TextEmbedder::embed_long`]'s content-aware
  /// chunking or window aggregation) failed inside the windit engine. Carries
  /// windit's own typed error unchanged ([`WinditError`] is `#[non_exhaustive]`,
  /// so match it with a wildcard arm). Notably `WinditError::NonFinite` here is
  /// windit's determinacy gate — an aggregate whose per-chunk embeddings cancel
  /// exactly has no direction at working precision.
  ///
  /// [`TextEmbedder::embed_long`]: crate::embeddings::granite::TextEmbedder::embed_long
  #[error("windowed-sequence processing failed: {0}")]
  Windowing(#[from] WinditError),

  /// [`TextEmbedder::embed_long_with`] was configured with a per-chunk token
  /// budget above the model's fixed input window ([`MAX_TOKENS`]), so every chunk
  /// would be silently truncated by the tokenizer. Rejected before any chunking
  /// or prediction runs.
  ///
  /// [`TextEmbedder::embed_long_with`]: crate::embeddings::granite::TextEmbedder::embed_long_with
  /// [`MAX_TOKENS`]: crate::embeddings::granite::MAX_TOKENS
  #[error("embed_long window budget {window} exceeds the model's fixed {max}-token input window")]
  WindowOverBudget {
    /// The requested per-chunk token budget (`opts.window()`).
    window: usize,
    /// The model's fixed input window ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
    max: usize,
  },
}

#[cfg(test)]
mod tests;
