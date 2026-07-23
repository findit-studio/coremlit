//! The module's single error type and `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants;
//! tokenizer errors preserve their `#[source]` chain. Model-contract,
//! image-preprocessing, and embedding-invariant failures are their own variants
//! so callers can match on cause. Mirrors `granite`'s error module, extended
//! with the vision-tower image / position-embedding variants (siglip is a
//! dual-tower image+text surface). (Plain-text reference — siglip builds without
//! the `granite`/`clap` features, so its docs must not link across them.)

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// Any failure loading a siglip tower, preprocessing an image, running
/// inference, tokenizing text, or constructing an
/// [`crate::embeddings::siglip::Embedding`].
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
  /// `tests/siglip/model_io.rs` / `tests/siglip/text_model_io.rs`).
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

  /// An [`crate::embeddings::siglip::Rgb8Image`] view had a zero dimension, a
  /// `width · height · 3` byte length overflowing `usize`, or an axis exceeding
  /// the preprocessing bound
  /// [`crate::embeddings::siglip::image::MAX_IMAGE_AXIS`] (which keeps every
  /// accepted extent inside Pillow's `f32` box envelope and bounds resize
  /// working memory). A real decoded RGB image has non-zero, in-bound
  /// dimensions and a length that fits.
  #[error(
    "invalid image dimensions: {width}×{height} (zero, over the per-axis maximum, or size overflow)"
  )]
  ImageDimensions {
    /// The width supplied.
    width: usize,
    /// The height supplied.
    height: usize,
  },

  /// An [`crate::embeddings::siglip::Rgb8Image`] view's backing slice length did
  /// not equal `width · height · 3` (row-major, RGB-interleaved, 3 bytes/pixel).
  #[error("image data length mismatch: expected {expected} bytes (w·h·3), got {got}")]
  ImageDataLength {
    /// The backing slice length the caller supplied.
    got: usize,
    /// The required `width · height · 3` length.
    expected: usize,
  },

  /// Reading the base position-embedding grid sidecar
  /// (`pos_embed_16x16x768.f32le.bin`) failed.
  #[error("failed to read position-embedding grid: {0}")]
  PosEmbedLoad(#[source] std::io::Error),

  /// The base position-embedding grid sidecar's byte length did not equal the
  /// exact `16 · 16 · 768 · 4` raw little-endian f32 grid the vision tower
  /// requires (the load-time hard-validation of D5). A short or long file is a
  /// wrong or corrupt artifact.
  #[error("position-embedding grid length mismatch: expected {expected} bytes, got {got}")]
  PosEmbedLength {
    /// The sidecar's actual byte length.
    got: usize,
    /// The required `16 · 16 · 768 · 4` byte length.
    expected: usize,
  },

  /// Preprocessing produced more real patches than the resolved patch budget
  /// `P`. The budget solver caps `h_p · w_p ≤ P` by construction, so this is a
  /// defensive backstop — returned instead of an out-of-bounds write — against a
  /// future solver/plumbing bug.
  #[error("preprocessing produced {got} patches, exceeding the {max}-patch budget")]
  PatchCount {
    /// Number of real patches produced.
    got: usize,
    /// The resolved patch budget `P`.
    max: usize,
  },

  /// Preprocessing could not allocate a resize working buffer of `bytes` bytes
  /// (a pathologically large source geometry, or memory exhaustion). Returned
  /// instead of aborting the process on allocator failure. `bytes` is
  /// [`usize::MAX`] when the buffer's element count overflowed `usize` (a
  /// geometry that could never be allocated).
  #[error("image preprocessing failed to allocate a {bytes}-byte resize buffer")]
  PreprocessAllocation {
    /// The size of the refused allocation, or [`usize::MAX`] on size overflow.
    bytes: usize,
  },

  /// The caller passed an empty text string; there is nothing to embed.
  #[error("text input is empty")]
  EmptyText,

  /// An embedding slice did not have the expected dimension.
  #[error("embedding dimension mismatch: expected {expected}, got {got}")]
  EmbeddingDimMismatch {
    /// The required dimension
    /// ([`crate::embeddings::siglip::embedding::EMBEDDING_DIM`]).
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
  /// (`crate::embeddings::siglip::embedding::NORM_BUDGET`).
  #[error("embedding is not unit-norm: |norm² − 1| = {norm_sq_deviation}")]
  EmbeddingNotUnitNorm {
    /// `(norm² − 1).abs()`, the amount by which the invariant was violated.
    norm_sq_deviation: f32,
  },

  /// The tokenizer failed to load from its JSON definition.
  #[error("failed to load tokenizer: {0}")]
  TokenizerLoad(#[source] tokenizers::Error),

  /// The bundled `tokenizer.json` is still the build-time placeholder (its
  /// vocab carries the `PLACEHOLDER_…_IN_WAVE_B` sentinel), which maps every
  /// ordinary word to `<pad>` — embedding with it would silently produce
  /// meaningless vectors. Stage the source-revision Gemma tokenizer bytes (the
  /// golden-generation step of the port plan), or supply a real tokenizer via
  /// [`crate::embeddings::siglip::TextEmbedder::from_files`].
  #[error("bundled tokenizer is the build-time placeholder; stage the real Gemma tokenizer.json")]
  TokenizerPlaceholder,

  /// Configuring the tokenizer (truncation) failed.
  #[error("failed to configure tokenizer: {0}")]
  TokenizerConfig(#[source] tokenizers::Error),

  /// Encoding text into token ids failed.
  #[error("failed to tokenize text: {0}")]
  Tokenize(#[source] tokenizers::Error),

  /// The tokenized input exceeded the fixed text window
  /// ([`max_tokens`](crate::embeddings::siglip::TextEmbedder::max_tokens)).
  /// Every constructor forces truncation at that length and disables the
  /// tokenizer's own padding, so this is a defensive backstop — returned instead
  /// of an out-of-bounds panic — against a tokenizer that still yields more ids
  /// than the window.
  #[error("tokenized input has {got} tokens, exceeding the fixed {max}-token window")]
  TokenCount {
    /// Number of token ids the tokenizer produced.
    got: usize,
    /// The fixed window length (the text tower's resolved `T`).
    max: usize,
  },

  /// A token id did not fit the model's `int32` `input_ids` tensor. siglip's
  /// Gemma vocabulary (256000) is far below `i32::MAX`, so this only fires for a
  /// foreign tokenizer with an out-of-range id — returned instead of a silently
  /// wrapping cast.
  #[error("token id {id} exceeds the model's int32 input range")]
  TokenIdRange {
    /// The offending token id.
    id: u32,
  },

  /// A [`crate::embeddings::siglip::PreprocessedImage`] patch budget was zero,
  /// or so large that the `[P · 768]` tensor lengths would overflow `usize`.
  /// A real budget is the loaded model's resolved `P` (e.g. 512) — small and
  /// non-zero.
  #[error("invalid preprocessed patch budget {max_num_patches} (zero, or tensor lengths overflow)")]
  PreprocessedPatchBudget {
    /// The budget supplied to `try_new`.
    max_num_patches: usize,
  },

  /// A caller-supplied preprocessed tensor's length did not match the padded
  /// contract at the supplied patch budget (`pixel_values` = `P · 768`,
  /// `position_embeddings` = `P · 768`, `attention_mask` = `P`).
  #[error("preprocessed `{feature}` length mismatch: expected {expected}, got {got}")]
  PreprocessedLength {
    /// The model input feature (`pixel_values` / `position_embeddings` /
    /// `attention_mask`) whose length mismatched.
    feature: &'static str,
    /// The length the caller supplied.
    got: usize,
    /// The required length at the supplied budget.
    expected: usize,
  },

  /// A caller-supplied preprocessed tensor contained a NaN or infinite value
  /// — caller-data corruption, classified apart from the model-output
  /// counterpart ([`Error::NonFiniteOutput`]).
  #[error("preprocessed `{feature}` contains a non-finite value at index {index}")]
  PreprocessedNonFinite {
    /// The model input feature containing the non-finite value.
    feature: &'static str,
    /// Flat index of the first non-finite element.
    index: usize,
  },

  /// A preprocessed attention-mask entry was not exactly `0.0` or `1.0`. The
  /// NaFlex pipeline emits an exact binary real/pad mask; anything else is
  /// not its output. (A NaN mask entry is classified here rather than as
  /// [`Error::PreprocessedNonFinite`] — the mask's domain check subsumes
  /// finiteness.)
  #[error("preprocessed attention mask entry {index} is {value}, not exactly 0.0 or 1.0")]
  PreprocessedMaskValue {
    /// Index of the offending entry.
    index: usize,
    /// The offending value.
    value: f32,
  },

  /// A preprocessed attention mask had a real (`1.0`) entry after a pad
  /// (`0.0`). The NaFlex pipeline packs real patches as a contiguous prefix
  /// with pads only at the tail; a non-prefix mask is not its output.
  #[error("preprocessed attention mask has a real (1.0) entry at index {index} after a pad")]
  PreprocessedMaskOrder {
    /// Index of the out-of-order `1.0`.
    index: usize,
  },

  /// A preprocessed attention mask had no real (`1.0`) entries. The budget
  /// solver guarantees at least one real patch; an all-pad input would make
  /// the graph attend over nothing.
  #[error("preprocessed attention mask has no real (1.0) entries")]
  PreprocessedMaskEmpty,

  /// A padded (mask `0.0`) row of a preprocessed tensor contained a nonzero
  /// value. The NaFlex pipeline zero-fills pad rows and the module's parity
  /// evidence covers only zero pads, so nonzero pad content is rejected
  /// fail-closed rather than trusted to be masked out by the graph.
  #[error("preprocessed `{feature}` has a nonzero value at index {index} inside a padded row")]
  PreprocessedPadNonZero {
    /// The model input feature (`pixel_values` / `position_embeddings`).
    feature: &'static str,
    /// Flat index of the first nonzero pad element.
    index: usize,
  },

  /// A [`crate::embeddings::siglip::PreprocessedImage`] was validated against
  /// a different patch budget than the loaded model resolved at load (e.g. a
  /// 256-tier bundle fed to a 512-tier model). Rebuild the bundle with this
  /// embedder's
  /// [`crate::embeddings::siglip::ImageEmbedder::max_num_patches`].
  #[error("preprocessed patch budget {input} does not match the model's resolved budget {model}")]
  PatchBudgetMismatch {
    /// The budget the input bundle was validated against.
    input: usize,
    /// The budget the loaded model resolved at load.
    model: usize,
  },
}

#[cfg(test)]
mod tests;
