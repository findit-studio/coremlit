//! The module's single error type and `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants;
//! tokenizer errors preserve their `#[source]` chain. Model-contract,
//! image-preprocessing, and embedding-invariant failures are their own variants
//! so callers can match on cause. Mirrors `granite`'s error module, extended
//! with the vision-tower image / position-embedding variants (siglip is a
//! dual-tower image+text surface). (Plain-text reference — siglip builds without
//! the `granite`/`clap` features, so its docs must not link across them.)

use crate::model::contract::{ContractViolation, Rendered};

/// Re-exported so callers can name and match the reason
/// [`Error::PostProcessorTemplate`] carries. The check itself is shared with the
/// crate's other text doors; the reasons are the same for all of them.
pub use crate::embeddings::tokenizer_guard::PostProcessorTemplate;

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// A loaded model's input or output feature does not match the shape/dtype
/// contract this module was built against (the pinned ground truth lives in
/// `tests/siglip/model_io.rs` / `tests/siglip/text_model_io.rs`).
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
  /// Construct from the shape the runtime tensor actually had and the shape
  /// the construction-time contract declares.
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

/// An [`crate::embeddings::siglip::Rgb8Image`] view had a zero dimension, a
/// `width · height · 3` byte length overflowing `usize`, or an axis exceeding
/// the preprocessing bound
/// [`crate::embeddings::siglip::image::MAX_IMAGE_AXIS`] (which keeps every
/// accepted extent inside Pillow's `f32` box envelope and bounds resize
/// working memory). A real decoded RGB image has non-zero, in-bound
/// dimensions and a length that fits.
///
/// Payload of [`Error::ImageDimensions`].
#[derive(Debug)]
pub struct ImageDimensions {
  /// The width supplied.
  width: usize,
  /// The height supplied.
  height: usize,
}

impl ImageDimensions {
  /// Construct from the width and the height supplied.
  #[inline(always)]
  pub const fn new(width: usize, height: usize) -> Self {
    Self { width, height }
  }

  /// The width supplied.
  #[inline(always)]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// The height supplied.
  #[inline(always)]
  pub const fn height(&self) -> usize {
    self.height
  }
}

/// An [`crate::embeddings::siglip::Rgb8Image`] view's backing slice length did
/// not equal `width · height · 3` (row-major, RGB-interleaved, 3 bytes/pixel).
///
/// Payload of [`Error::ImageDataLength`].
#[derive(Debug)]
pub struct ImageDataLength {
  /// The backing slice length the caller supplied.
  got: usize,
  /// The required `width · height · 3` length.
  expected: usize,
}

impl ImageDataLength {
  /// Construct from the backing slice length the caller supplied and the
  /// required `width · height · 3` length.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// The backing slice length the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The required `width · height · 3` length.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// The base position-embedding grid sidecar's byte length did not equal the
/// exact `16 · 16 · 768 · 4` raw little-endian f32 grid the vision tower
/// requires (the load-time hard-validation of D5). A short or long file is a
/// wrong or corrupt artifact.
///
/// Payload of [`Error::PosEmbedLength`].
#[derive(Debug)]
pub struct PosEmbedLength {
  /// The sidecar's actual byte length.
  got: usize,
  /// The required `16 · 16 · 768 · 4` byte length.
  expected: usize,
}

impl PosEmbedLength {
  /// Construct from the sidecar's actual byte length and the required
  /// `16 · 16 · 768 · 4` byte length.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// The sidecar's actual byte length.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The required `16 · 16 · 768 · 4` byte length.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// Preprocessing produced more real patches than the resolved patch budget
/// `P`. The budget solver caps `h_p · w_p ≤ P` by construction, so this is a
/// defensive backstop — returned instead of an out-of-bounds write — against a
/// future solver/plumbing bug.
///
/// Payload of [`Error::PatchCount`].
#[derive(Debug)]
pub struct PatchCount {
  /// Number of real patches produced.
  got: usize,
  /// The resolved patch budget `P`.
  max: usize,
}

impl PatchCount {
  /// Construct from the number of real patches produced and the resolved
  /// patch budget `P`.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// Number of real patches produced.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The resolved patch budget `P`.
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
  /// The required dimension
  /// ([`crate::embeddings::siglip::embedding::EMBEDDING_DIM`]).
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

  /// The required dimension
  /// ([`crate::embeddings::siglip::embedding::EMBEDDING_DIM`]).
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

/// The model artifact's `tokenizer.json` sidecar could not be read.
///
/// [`crate::embeddings::siglip::TextEmbedder::load`] reads the tokenizer from
/// the directory CONTAINING the text `.mlmodelc`, where the published bundle
/// stages it. This is the "the artifact tree is incomplete" failure — an older
/// download that predates the sidecar, a partial fetch, or a `.mlmodelc` copied
/// out of its artifact directory. Distinct from [`Error::TokenizerLoad`], which
/// means the bytes were read but are not a valid tokenizer.
///
/// Payload of [`Error::ArtifactTokenizerRead`].
///
/// Unlike its sibling payloads this one derives [`std::error::Error`] and owns
/// both the variant's message and its `#[source]`: the variant is
/// `#[error(transparent)]`, which forwards `Display` *and* `source()` straight
/// through to this struct rather than inserting it as a link. The rendered
/// message and the error chain are therefore exactly what the struct-shaped
/// variant produced — one link, to the [`std::io::Error`] below — not one link
/// deeper.
///
/// Note that the inherent [`source`](Self::source) getter returns the concrete
/// `&`[`std::io::Error`] the struct pattern used to expose, and so shadows
/// [`std::error::Error::source`] for method-call syntax; call the trait method
/// by path (`std::error::Error::source(&e)`) to walk the chain.
#[derive(Debug, thiserror::Error)]
#[error("failed to read the artifact tokenizer `{path}`: {source}")]
pub struct ArtifactTokenizerRead {
  /// The sidecar path that could not be read.
  path: std::path::PathBuf,
  /// The underlying I/O failure.
  #[source]
  source: std::io::Error,
}

impl ArtifactTokenizerRead {
  /// Construct from the sidecar path that could not be read and the underlying
  /// I/O failure.
  #[inline(always)]
  pub const fn new(path: std::path::PathBuf, source: std::io::Error) -> Self {
    Self { path, source }
  }

  /// The sidecar path that could not be read.
  #[inline(always)]
  pub fn path(&self) -> &std::path::Path {
    &self.path
  }

  /// The underlying I/O failure.
  #[inline(always)]
  pub const fn source(&self) -> &std::io::Error {
    &self.source
  }
}

/// The `tokenizer.json` read from the model artifact directory is not the
/// pinned source-revision Gemma artifact (SHA-256).
///
/// SigLIP 2 NaFlex is a fixed model with exactly one correct tokenizer, and the
/// tokenizer now travels with the artifact instead of being compiled into the
/// crate — so its identity is checked at load rather than assumed. A wrong,
/// truncated, or re-serialized sidecar would otherwise produce finite but
/// meaningless embeddings. Supply your own bytes through
/// [`crate::embeddings::siglip::TextEmbedder::from_files`] /
/// [`crate::embeddings::siglip::TextEmbedder::from_memory`] if you deliberately
/// want a different tokenizer.
///
/// Payload of [`Error::ArtifactTokenizerIdentity`].
#[derive(Debug)]
pub struct ArtifactTokenizerIdentity {
  /// The sidecar path whose bytes were hashed.
  path: std::path::PathBuf,
  /// The pinned SHA-256 (lowercase hex).
  expected: &'static str,
  /// The SHA-256 (lowercase hex) of the bytes actually read.
  actual: String,
}

impl ArtifactTokenizerIdentity {
  /// Construct from the sidecar path whose bytes were hashed, the pinned
  /// SHA-256, and the SHA-256 of the bytes actually read.
  #[inline(always)]
  pub const fn new(path: std::path::PathBuf, expected: &'static str, actual: String) -> Self {
    Self {
      path,
      expected,
      actual,
    }
  }

  /// The sidecar path whose bytes were hashed.
  #[inline(always)]
  pub fn path(&self) -> &std::path::Path {
    &self.path
  }

  /// The pinned SHA-256 (lowercase hex).
  #[inline(always)]
  pub const fn expected(&self) -> &'static str {
    self.expected
  }

  /// The SHA-256 (lowercase hex) of the bytes actually read.
  #[inline(always)]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// The tokenized input exceeded the fixed text window
/// ([`max_tokens`](crate::embeddings::siglip::TextEmbedder::max_tokens)).
/// Every constructor forces truncation at that length and disables the
/// tokenizer's own padding, so this is a defensive backstop — returned instead
/// of an out-of-bounds panic — against a tokenizer that still yields more ids
/// than the window.
///
/// Payload of [`Error::TokenCount`].
#[derive(Debug)]
pub struct TokenCount {
  /// Number of token ids the tokenizer produced.
  got: usize,
  /// The fixed window length (the text tower's resolved `T`).
  max: usize,
}

impl TokenCount {
  /// Construct from the number of token ids the tokenizer produced and the
  /// fixed window length.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// Number of token ids the tokenizer produced.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The fixed window length (the text tower's resolved `T`).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A caller-supplied tokenizer's post-processor adds at least as many special
/// tokens as the text window holds, so no text token can fit.
///
/// `tokenizers::Tokenizer::with_truncation` computes its effective window as
/// `max_length - post_processor.added_tokens(false)` with an UNCHECKED `usize`
/// subtraction, and repeats that subtraction on every `encode(_, true)`. A
/// post-processor that over-fills the window therefore panics inside the
/// dependency under overflow checks, and under a release profile wraps to a
/// near-`usize::MAX` window that never truncates — leaving every later embed to
/// fail the [`Error::TokenCount`] backstop instead. Refused here, before that
/// subtraction, naming both numbers.
///
/// # Why `added >= window` and not the dependency's `added > window`
///
/// `added > window` is only the arithmetic precondition. `added == window`
/// subtracts cleanly, to an effective text window of **zero**, and the encoding
/// is then the special tokens alone: a two-special post-processor at a
/// two-token window encodes any text to those two ids and nothing else. That is
/// a silently wrong answer rather than a reported failure, so the equal case is
/// refused alongside the overflowing one.
///
/// Payload of [`Error::SpecialTokenOverhead`].
#[derive(Debug)]
pub struct SpecialTokenOverhead {
  /// Special tokens the post-processor adds to a single sequence.
  added: usize,
  /// The fixed window length (the text tower's resolved `T`).
  window: usize,
}

impl SpecialTokenOverhead {
  /// Construct from the post-processor's single-sequence special-token count
  /// and the fixed window length.
  #[inline(always)]
  pub const fn new(added: usize, window: usize) -> Self {
    Self { added, window }
  }

  /// Special tokens the post-processor adds to a single sequence.
  #[inline(always)]
  pub const fn added(&self) -> usize {
    self.added
  }

  /// The fixed window length (the text tower's resolved `T`).
  #[inline(always)]
  pub const fn window(&self) -> usize {
    self.window
  }
}

/// A caller-supplied preprocessed tensor's length did not match the padded
/// contract at the supplied patch budget (`pixel_values` = `P · 768`,
/// `position_embeddings` = `P · 768`, `attention_mask` = `P`).
///
/// Payload of [`Error::PreprocessedLength`].
#[derive(Debug)]
pub struct PreprocessedLength {
  /// The model input feature (`pixel_values` / `position_embeddings` /
  /// `attention_mask`) whose length mismatched.
  feature: &'static str,
  /// The length the caller supplied.
  got: usize,
  /// The required length at the supplied budget.
  expected: usize,
}

impl PreprocessedLength {
  /// Construct from the model input feature whose length mismatched, the
  /// length the caller supplied, and the required length at the supplied
  /// budget.
  #[inline(always)]
  pub const fn new(feature: &'static str, got: usize, expected: usize) -> Self {
    Self {
      feature,
      got,
      expected,
    }
  }

  /// The model input feature (`pixel_values` / `position_embeddings` /
  /// `attention_mask`) whose length mismatched.
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }

  /// The length the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The required length at the supplied budget.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// A caller-supplied preprocessed tensor contained a NaN or infinite value
/// — caller-data corruption, classified apart from the model-output
/// counterpart ([`Error::NonFiniteOutput`]).
///
/// Payload of [`Error::PreprocessedNonFinite`].
#[derive(Debug)]
pub struct PreprocessedNonFinite {
  /// The model input feature containing the non-finite value.
  feature: &'static str,
  /// Flat index of the first non-finite element.
  index: usize,
}

impl PreprocessedNonFinite {
  /// Construct from the model input feature containing the non-finite value
  /// and the flat index of the first non-finite element.
  #[inline(always)]
  pub const fn new(feature: &'static str, index: usize) -> Self {
    Self { feature, index }
  }

  /// The model input feature containing the non-finite value.
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }

  /// Flat index of the first non-finite element.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }
}

/// A preprocessed attention-mask entry was not exactly `0.0` or `1.0`. The
/// NaFlex pipeline emits an exact binary real/pad mask; anything else is
/// not its output. (A NaN mask entry is classified here rather than as
/// [`Error::PreprocessedNonFinite`] — the mask's domain check subsumes
/// finiteness.)
///
/// Payload of [`Error::PreprocessedMaskValue`].
#[derive(Debug)]
pub struct PreprocessedMaskValue {
  /// Index of the offending entry.
  index: usize,
  /// The offending value.
  value: f32,
}

impl PreprocessedMaskValue {
  /// Construct from the index of the offending entry and the offending value.
  #[inline(always)]
  pub const fn new(index: usize, value: f32) -> Self {
    Self { index, value }
  }

  /// Index of the offending entry.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }

  /// The offending value.
  #[inline(always)]
  pub const fn value(&self) -> f32 {
    self.value
  }
}

/// A padded (mask `0.0`) row of a preprocessed tensor contained a nonzero
/// value. The NaFlex pipeline zero-fills pad rows and the module's parity
/// evidence covers only zero pads, so nonzero pad content is rejected
/// fail-closed rather than trusted to be masked out by the graph.
///
/// Payload of [`Error::PreprocessedPadNonZero`].
#[derive(Debug)]
pub struct PreprocessedPadNonZero {
  /// The model input feature (`pixel_values` / `position_embeddings`).
  feature: &'static str,
  /// Flat index of the first nonzero pad element.
  index: usize,
}

impl PreprocessedPadNonZero {
  /// Construct from the model input feature and the flat index of the first
  /// nonzero pad element.
  #[inline(always)]
  pub const fn new(feature: &'static str, index: usize) -> Self {
    Self { feature, index }
  }

  /// The model input feature (`pixel_values` / `position_embeddings`).
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }

  /// Flat index of the first nonzero pad element.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }
}

/// A [`crate::embeddings::siglip::PreprocessedImage`] was validated against
/// a different patch budget than the loaded model resolved at load (e.g. a
/// 256-tier bundle fed to a 512-tier model). Rebuild the bundle with this
/// embedder's
/// [`crate::embeddings::siglip::ImageEmbedder::max_num_patches`].
///
/// Payload of [`Error::PatchBudgetMismatch`].
#[derive(Debug)]
pub struct PatchBudgetMismatch {
  /// The budget the input bundle was validated against.
  input: usize,
  /// The budget the loaded model resolved at load.
  model: usize,
}

impl PatchBudgetMismatch {
  /// Construct from the budget the input bundle was validated against and the
  /// budget the loaded model resolved at load.
  #[inline(always)]
  pub const fn new(input: usize, model: usize) -> Self {
    Self { input, model }
  }

  /// The budget the input bundle was validated against.
  #[inline(always)]
  pub const fn input(&self) -> usize {
    self.input
  }

  /// The budget the loaded model resolved at load.
  #[inline(always)]
  pub const fn model(&self) -> usize {
    self.model
  }
}

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
  #[error("model contract mismatch on `{}`: expected {}, got {}", .0.feature(), .0.expected(), .0.actual())]
  ContractMismatch(ContractMismatch),

  /// The loaded graph declares a REQUIRED input the door that opened it never
  /// supplies, so every prediction through it would fail.
  ///
  /// Carries the offending feature name. An OPTIONAL extra input is not this:
  /// CoreML runs a prediction that omits one, so only a required input the
  /// door cannot fill makes the contract unsatisfiable.
  #[error(
    "model declares a required input `{0}` that this door never supplies, so \
     every prediction would fail"
  )]
  UnsatisfiableInput(String),

  /// The loaded graph declares CoreML STATE buffers, and this crate's doors
  /// predict through the stateless API.
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

  /// A predict-time output tensor's shape diverged from the contract validated
  /// at construction. [`crate::MultiArray::copy_into`] alone validates only
  /// total element count, so an axes-swapped output would otherwise pass
  /// silently — the CoreML runtime is re-checked on every call.
  #[error("output shape mismatch: expected {:?}, got {:?}", .0.expected(), .0.got())]
  OutputShape(OutputShape),

  /// A model output component was NaN or infinite.
  ///
  /// Carries the flat index of the offending element.
  #[error("model output contains a non-finite value at index {0}")]
  NonFiniteOutput(usize),

  /// An [`crate::embeddings::siglip::Rgb8Image`] view had a zero dimension, a
  /// `width · height · 3` byte length overflowing `usize`, or an axis exceeding
  /// the preprocessing bound
  /// [`crate::embeddings::siglip::image::MAX_IMAGE_AXIS`] (which keeps every
  /// accepted extent inside Pillow's `f32` box envelope and bounds resize
  /// working memory). A real decoded RGB image has non-zero, in-bound
  /// dimensions and a length that fits.
  #[error(
    "invalid image dimensions: {}×{} (zero, over the per-axis maximum, or size overflow)",
    .0.width(),
    .0.height()
  )]
  ImageDimensions(ImageDimensions),

  /// An [`crate::embeddings::siglip::Rgb8Image`] view's backing slice length did
  /// not equal `width · height · 3` (row-major, RGB-interleaved, 3 bytes/pixel).
  #[error("image data length mismatch: expected {} bytes (w·h·3), got {}", .0.expected(), .0.got())]
  ImageDataLength(ImageDataLength),

  /// Reading the base position-embedding grid sidecar
  /// (`pos_embed_16x16x768.f32le.bin`) failed.
  #[error("failed to read position-embedding grid: {0}")]
  PosEmbedLoad(#[source] std::io::Error),

  /// The base position-embedding grid sidecar's byte length did not equal the
  /// exact `16 · 16 · 768 · 4` raw little-endian f32 grid the vision tower
  /// requires (the load-time hard-validation of D5). A short or long file is a
  /// wrong or corrupt artifact.
  #[error("position-embedding grid length mismatch: expected {} bytes, got {}", .0.expected(), .0.got())]
  PosEmbedLength(PosEmbedLength),

  /// Preprocessing produced more real patches than the resolved patch budget
  /// `P`. The budget solver caps `h_p · w_p ≤ P` by construction, so this is a
  /// defensive backstop — returned instead of an out-of-bounds write — against a
  /// future solver/plumbing bug.
  #[error("preprocessing produced {} patches, exceeding the {}-patch budget", .0.got(), .0.max())]
  PatchCount(PatchCount),

  /// Preprocessing could not allocate a resize working buffer of the carried
  /// size (a pathologically large source geometry, or memory exhaustion).
  /// Returned instead of aborting the process on allocator failure.
  ///
  /// Carries the size of the refused allocation, which is [`usize::MAX`] when
  /// the buffer's element count overflowed `usize` (a geometry that could
  /// never be allocated).
  #[error("image preprocessing failed to allocate a {0}-byte resize buffer")]
  PreprocessAllocation(usize),

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

  /// A trusted-path embedding was not unit-norm within the module's norm budget
  /// (`crate::embeddings::siglip::embedding::NORM_BUDGET`).
  ///
  /// Carries `(norm² − 1).abs()`, the amount by which the invariant was
  /// violated.
  #[error("embedding is not unit-norm: |norm² − 1| = {0}")]
  EmbeddingNotUnitNorm(f32),

  /// The tokenizer failed to load from its JSON definition.
  #[error("failed to load tokenizer: {0}")]
  TokenizerLoad(#[source] tokenizers::Error),

  /// The model artifact's `tokenizer.json` sidecar could not be read.
  ///
  /// [`crate::embeddings::siglip::TextEmbedder::load`] reads the tokenizer from
  /// the directory CONTAINING the text `.mlmodelc`, where the published bundle
  /// stages it. This is the "the artifact tree is incomplete" failure — an older
  /// download that predates the sidecar, a partial fetch, or a `.mlmodelc` copied
  /// out of its artifact directory. Distinct from [`Error::TokenizerLoad`], which
  /// means the bytes were read but are not a valid tokenizer.
  ///
  /// The message and the `source` live on [`ArtifactTokenizerRead`], which this variant
  /// forwards both of through `#[error(transparent)]`.
  #[error(transparent)]
  ArtifactTokenizerRead(#[from] ArtifactTokenizerRead),

  /// The `tokenizer.json` read from the model artifact directory is not the
  /// pinned source-revision Gemma artifact (SHA-256).
  ///
  /// SigLIP 2 NaFlex is a fixed model with exactly one correct tokenizer, and the
  /// tokenizer now travels with the artifact instead of being compiled into the
  /// crate — so its identity is checked at load rather than assumed. A wrong,
  /// truncated, or re-serialized sidecar would otherwise produce finite but
  /// meaningless embeddings. Supply your own bytes through
  /// [`crate::embeddings::siglip::TextEmbedder::from_files`] /
  /// [`crate::embeddings::siglip::TextEmbedder::from_memory`] if you deliberately
  /// want a different tokenizer.
  #[error(
    "artifact tokenizer `{}` is not the pinned Gemma tokenizer: expected sha-256 {}, got {}",
    .0.path().display(),
    .0.expected(),
    .0.actual()
  )]
  ArtifactTokenizerIdentity(ArtifactTokenizerIdentity),

  /// A `tokenizer.json` is still the build-time placeholder (its vocab carries
  /// the `PLACEHOLDER_…_IN_WAVE_B` sentinel), which maps every ordinary word to
  /// `<pad>` — embedding with it would silently produce meaningless vectors.
  /// Stage the source-revision Gemma tokenizer bytes beside the model bundle, or
  /// supply a real tokenizer via
  /// [`crate::embeddings::siglip::TextEmbedder::from_files`].
  #[error("tokenizer is the build-time placeholder; stage the real Gemma tokenizer.json")]
  TokenizerPlaceholder,

  /// Configuring the tokenizer (truncation) failed.
  #[error("failed to configure tokenizer: {0}")]
  TokenizerConfig(#[source] tokenizers::Error),

  /// The tokenizer's post-processor adds at least as many special tokens as the
  /// resolved text window holds, so no text token can fit. Refused before the
  /// dependency's unchecked `max_length - added_tokens` subtraction; see
  /// [`SpecialTokenOverhead`] for why the equal case is refused too.
  #[error(
    "tokenizer post-processor adds {} special tokens, leaving no room for text in the {}-token window",
    .0.added(),
    .0.window()
  )]
  SpecialTokenOverhead(SpecialTokenOverhead),

  /// The tokenizer's post-processor is not one this door's single-sequence
  /// `encode` can be trusted with: a `TemplateProcessing` it reaches is
  /// internally inconsistent, its single template places the text other than
  /// exactly once, or the chain hands a token-adding post-processor a number of
  /// encodings other than one. The tokenizers crate's deserializer skips its own
  /// builder's validation, so such a post-processor parses and then PANICS
  /// inside the dependency on the first `encode`, silently drops the text, or
  /// returns more tokens than the window its truncation was sized for; this door
  /// refuses it at construction instead. See [`PostProcessorTemplate`] for the
  /// rules and why they stop where they do.
  #[error("tokenizer post-processor is inconsistent: {0}")]
  PostProcessorTemplate(PostProcessorTemplate),

  /// Encoding text into token ids failed.
  #[error("failed to tokenize text: {0}")]
  Tokenize(#[source] tokenizers::Error),

  /// The tokenized input exceeded the fixed text window
  /// ([`max_tokens`](crate::embeddings::siglip::TextEmbedder::max_tokens)).
  /// Every constructor forces truncation at that length and disables the
  /// tokenizer's own padding, so this is a defensive backstop — returned instead
  /// of an out-of-bounds panic — against a tokenizer that still yields more ids
  /// than the window.
  #[error("tokenized input has {} tokens, exceeding the fixed {}-token window", .0.got(), .0.max())]
  TokenCount(TokenCount),

  /// A token id did not fit the model's `int32` `input_ids` tensor. siglip's
  /// Gemma vocabulary (256000) is far below `i32::MAX`, so this only fires for a
  /// foreign tokenizer with an out-of-range id — returned instead of a silently
  /// wrapping cast.
  ///
  /// Carries the offending token id.
  #[error("token id {0} exceeds the model's int32 input range")]
  TokenIdRange(u32),

  /// A [`crate::embeddings::siglip::PreprocessedImage`] patch budget was zero,
  /// or so large that the `[P · 768]` tensor lengths would overflow `usize`.
  /// A real budget is the loaded model's resolved `P` (e.g. 512) — small and
  /// non-zero.
  ///
  /// Carries the budget supplied to `try_new`.
  #[error("invalid preprocessed patch budget {0} (zero, or tensor lengths overflow)")]
  PreprocessedPatchBudget(usize),

  /// A caller-supplied preprocessed tensor's length did not match the padded
  /// contract at the supplied patch budget (`pixel_values` = `P · 768`,
  /// `position_embeddings` = `P · 768`, `attention_mask` = `P`).
  #[error(
    "preprocessed `{}` length mismatch: expected {}, got {}",
    .0.feature(),
    .0.expected(),
    .0.got()
  )]
  PreprocessedLength(PreprocessedLength),

  /// A caller-supplied preprocessed tensor contained a NaN or infinite value
  /// — caller-data corruption, classified apart from the model-output
  /// counterpart ([`Error::NonFiniteOutput`]).
  #[error("preprocessed `{}` contains a non-finite value at index {}", .0.feature(), .0.index())]
  PreprocessedNonFinite(PreprocessedNonFinite),

  /// A preprocessed attention-mask entry was not exactly `0.0` or `1.0`. The
  /// NaFlex pipeline emits an exact binary real/pad mask; anything else is
  /// not its output. (A NaN mask entry is classified here rather than as
  /// [`Error::PreprocessedNonFinite`] — the mask's domain check subsumes
  /// finiteness.)
  #[error("preprocessed attention mask entry {} is {}, not exactly 0.0 or 1.0", .0.index(), .0.value())]
  PreprocessedMaskValue(PreprocessedMaskValue),

  /// A preprocessed attention mask had a real (`1.0`) entry after a pad
  /// (`0.0`). The NaFlex pipeline packs real patches as a contiguous prefix
  /// with pads only at the tail; a non-prefix mask is not its output.
  ///
  /// Carries the index of the out-of-order `1.0`.
  #[error("preprocessed attention mask has a real (1.0) entry at index {0} after a pad")]
  PreprocessedMaskOrder(usize),

  /// A preprocessed attention mask had no real (`1.0`) entries. The budget
  /// solver guarantees at least one real patch; an all-pad input would make
  /// the graph attend over nothing.
  #[error("preprocessed attention mask has no real (1.0) entries")]
  PreprocessedMaskEmpty,

  /// A padded (mask `0.0`) row of a preprocessed tensor contained a nonzero
  /// value. The NaFlex pipeline zero-fills pad rows and the module's parity
  /// evidence covers only zero pads, so nonzero pad content is rejected
  /// fail-closed rather than trusted to be masked out by the graph.
  #[error(
    "preprocessed `{}` has a nonzero value at index {} inside a padded row",
    .0.feature(),
    .0.index()
  )]
  PreprocessedPadNonZero(PreprocessedPadNonZero),

  /// A [`crate::embeddings::siglip::PreprocessedImage`] was validated against
  /// a different patch budget than the loaded model resolved at load (e.g. a
  /// 256-tier bundle fed to a 512-tier model). Rebuild the bundle with this
  /// embedder's
  /// [`crate::embeddings::siglip::ImageEmbedder::max_num_patches`].
  #[error(
    "preprocessed patch budget {} does not match the model's resolved budget {}",
    .0.input(),
    .0.model()
  )]
  PatchBudgetMismatch(PatchBudgetMismatch),
}

#[cfg(test)]
mod tests;

/// Map a [`ContractViolation`] into this module's error vocabulary.
///
/// Shared by both siglip doors, because both hold a `Checked` and both report
/// into this one [`Error`]. The two "unsatisfiable" clauses keep their own
/// variants — they are about what a door cannot SUPPLY, not about a feature's
/// declared shape — and the per-feature clauses all land in
/// [`Error::ContractMismatch`], which already carries a feature name and a
/// rendered expected/actual pair. An output the model declares OPTIONAL is one
/// of those: it is a fact about the named feature's declaration, so "expected a
/// required output, got optional" is the shape that pair was made for.
///
/// `ContractViolation::rendered` performs that reduction, so a clause added to
/// the checker later lands in the `Feature` arm rather than breaking this
/// function and its five siblings at once.
pub(crate) fn contract_violation(violation: ContractViolation) -> Error {
  match violation.rendered() {
    Rendered::UnsatisfiableInput(name) => Error::UnsatisfiableInput(name),
    Rendered::UnsatisfiableState(name) => Error::UnsatisfiableState(name),
    Rendered::Feature(feature) => Error::ContractMismatch(ContractMismatch::new(
      feature.feature(),
      feature.clone().expected(),
      feature.actual(),
    )),
  }
}
