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

/// A loaded model's input or output feature does not match the shape/dtype
/// contract this module was built against (the pinned ground truth lives in
/// `tests/granite/model_io.rs`).
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

/// An embedding slice did not have the expected dimension.
///
/// Payload of [`Error::EmbeddingDimMismatch`].
#[derive(Debug)]
pub struct EmbeddingDimMismatch {
  /// The required dimension ([`crate::embeddings::granite::embedding::EMBEDDING_DIM`]).
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

  /// The required dimension ([`crate::embeddings::granite::embedding::EMBEDDING_DIM`]).
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
/// [`crate::embeddings::granite::TextEmbedder::load`] reads the tokenizer from
/// the directory CONTAINING the `.mlmodelc`, where the published bundle stages
/// it. This is the "the artifact tree is incomplete" failure — an older
/// download that predates the sidecar, a partial fetch, or a `.mlmodelc`
/// copied out of its artifact directory. Distinct from
/// [`Error::TokenizerLoad`], which means the bytes were read but are not a
/// valid tokenizer.
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

/// A tokenizer parsed but does not match the Granite tokenizer/model contract
/// (vocabulary size, special-token ids, the model's id range, or the pinned
/// sentinel encoding), so it would produce finite but semantically meaningless
/// embeddings — or out-of-vocabulary ids the model can only gather as zeros;
/// OR it is behaviorally valid but not byte-identical (SHA-256) to the pinned
/// granite `tokenizer.json`, catching corruption or version skew outside the
/// behavioral checks' coverage. Checked at construction, fail-closed, by every
/// constructor — BOTH stages, for the artifact sidecar `load` reads and for
/// caller-supplied bytes (`from_memory` / `from_files`) alike.
///
/// Payload of [`Error::TokenizerContractMismatch`].
#[derive(Debug)]
pub struct TokenizerContractMismatch {
  /// Name of the contract check that failed (e.g. `vocab size`,
  /// `special token <|startoftext|>`, `max token id`, `sentinel encoding`,
  /// `tokenizer identity (sha-256)`, `artifact tokenizer identity (sha-256)`).
  check: &'static str,
  /// The contract this module expects, rendered for display.
  expected: String,
  /// What the supplied tokenizer actually declares/produces, rendered for
  /// display (`missing` for an absent token).
  actual: String,
}

impl TokenizerContractMismatch {
  /// Construct from the failed contract check, the expected contract, and what
  /// the supplied tokenizer actually declares/produces.
  #[inline(always)]
  pub const fn new(check: &'static str, expected: String, actual: String) -> Self {
    Self {
      check,
      expected,
      actual,
    }
  }

  /// Name of the contract check that failed (e.g. `vocab size`,
  /// `special token <|startoftext|>`, `max token id`, `sentinel encoding`,
  /// `tokenizer identity (sha-256)`, `artifact tokenizer identity (sha-256)`).
  #[inline(always)]
  pub const fn check(&self) -> &'static str {
    self.check
  }

  /// The contract this module expects, rendered for display.
  #[inline(always)]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// What the supplied tokenizer actually declares/produces, rendered for
  /// display (`missing` for an absent token).
  #[inline(always)]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// The tokenized input exceeded the fixed
/// [`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS) window. Every
/// constructor forces truncation at that length and disables the tokenizer's
/// own padding, so this is a defensive backstop — returned instead of an
/// out-of-bounds panic — against a tokenizer that still yields more ids than
/// the window (e.g. a padding policy that survived configuration).
///
/// Payload of [`Error::TokenCount`].
#[derive(Debug)]
pub struct TokenCount {
  /// Number of token ids the tokenizer produced.
  got: usize,
  /// The fixed window length
  /// ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
  max: usize,
}

impl TokenCount {
  /// Construct from the number of token ids the tokenizer produced and the
  /// fixed window length it exceeded.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// Number of token ids the tokenizer produced.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The fixed window length
  /// ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A caller-supplied tokenizer's post-processor adds at least as many special
/// tokens as the fixed
/// [`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS) window holds, so no
/// text token can fit.
///
/// `tokenizers::Tokenizer::with_truncation` computes its effective window as
/// `max_length - post_processor.added_tokens(false)` with an UNCHECKED `usize`
/// subtraction, and repeats that subtraction on every `encode(_, true)`. A
/// post-processor that over-fills the window therefore panics inside the
/// dependency under overflow checks, and under a release profile wraps to a
/// near-`usize::MAX` window that never truncates — leaving every later embed to
/// fail the [`Error::TokenCount`] backstop instead. Refused at configuration
/// time, before that subtraction and before
/// [`Error::TokenizerContractMismatch`]'s sentinel encode, naming both numbers.
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
  /// The fixed window length
  /// ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
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

  /// The fixed window length
  /// ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
  #[inline(always)]
  pub const fn window(&self) -> usize {
    self.window
  }
}

/// [`TextEmbedder::embed_long_with`] was configured with a per-chunk token
/// budget above the model's fixed input window ([`MAX_TOKENS`]), so every chunk
/// would be silently truncated by the tokenizer. Rejected before any chunking
/// or prediction runs.
///
/// Payload of [`Error::WindowOverBudget`].
///
/// [`TextEmbedder::embed_long_with`]: crate::embeddings::granite::TextEmbedder::embed_long_with
/// [`MAX_TOKENS`]: crate::embeddings::granite::MAX_TOKENS
#[derive(Debug)]
pub struct WindowOverBudget {
  /// The requested per-chunk token budget (`opts.window()`).
  window: usize,
  /// The model's fixed input window ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
  max: usize,
}

impl WindowOverBudget {
  /// Construct from the requested per-chunk token budget and the model's fixed
  /// input window it exceeded.
  #[inline(always)]
  pub const fn new(window: usize, max: usize) -> Self {
    Self { window, max }
  }

  /// The requested per-chunk token budget (`opts.window()`).
  #[inline(always)]
  pub const fn window(&self) -> usize {
    self.window
  }

  /// The model's fixed input window ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// The text handed to [`TextEmbedder::embed_long_with`] exceeds the
/// caller-configured input byte limit
/// ([`LongTextOptions::max_input_bytes`]). Enforced BEFORE any tokenizer or
/// chunker work, so the reject path's cost is independent of the input size —
/// the limit to set when embedding untrusted text.
///
/// Payload of [`Error::InputTooLarge`].
///
/// [`TextEmbedder::embed_long_with`]: crate::embeddings::granite::TextEmbedder::embed_long_with
/// [`LongTextOptions::max_input_bytes`]: crate::embeddings::granite::LongTextOptions::max_input_bytes
#[derive(Debug)]
pub struct InputTooLarge {
  /// The input length, in UTF-8 bytes.
  got: usize,
  /// The configured limit, in UTF-8 bytes.
  max: usize,
}

impl InputTooLarge {
  /// Construct from the input length and the configured byte limit it
  /// exceeded.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// The input length, in UTF-8 bytes.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The configured limit, in UTF-8 bytes.
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A contentless (separator-only) byte run that must be embedded as one whole
/// window — the ENTIRE input when it has no tokenizable content, or a
/// pure-separator gap between packed chunks that neither neighbor can absorb —
/// measures more tokens than the model's fixed window can hold, so embedding
/// it would silently drop its suffix tokens. Refused instead (measured with
/// the non-truncating tokenizer, special tokens included).
///
/// Payload of [`Error::ContentlessInputOverBudget`].
#[derive(Debug)]
pub struct ContentlessInputOverBudget {
  /// Start of the offending run, as a UTF-8 byte offset into the input.
  start: usize,
  /// One past the end of the offending run, as a UTF-8 byte offset.
  end: usize,
  /// The run's untruncated token count (special tokens included).
  tokens: usize,
  /// The fixed window ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
  max: usize,
}

impl ContentlessInputOverBudget {
  /// Construct from the offending run's byte span, its untruncated token
  /// count, and the fixed window it exceeded.
  #[inline(always)]
  pub const fn new(start: usize, end: usize, tokens: usize, max: usize) -> Self {
    Self {
      start,
      end,
      tokens,
      max,
    }
  }

  /// Start of the offending run, as a UTF-8 byte offset into the input.
  #[inline(always)]
  pub const fn start(&self) -> usize {
    self.start
  }

  /// One past the end of the offending run, as a UTF-8 byte offset.
  #[inline(always)]
  pub const fn end(&self) -> usize {
    self.end
  }

  /// The run's untruncated token count (special tokens included).
  #[inline(always)]
  pub const fn tokens(&self) -> usize {
    self.tokens
  }

  /// The fixed window ([`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS)).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

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

  /// A model output component was NaN or infinite.
  ///
  /// Carries the flat index of the offending element.
  #[error("model output contains a non-finite value at index {0}")]
  NonFiniteOutput(usize),

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
  /// (`crate::embeddings::granite::embedding::NORM_BUDGET`).
  ///
  /// Carries `(norm² − 1).abs()`, the amount by which the invariant was
  /// violated.
  #[error("embedding is not unit-norm: |norm² − 1| = {0}")]
  EmbeddingNotUnitNorm(f32),

  /// The model artifact's `tokenizer.json` sidecar could not be read.
  ///
  /// [`crate::embeddings::granite::TextEmbedder::load`] reads the tokenizer from
  /// the directory CONTAINING the `.mlmodelc`, where the published bundle stages
  /// it. This is the "the artifact tree is incomplete" failure — an older
  /// download that predates the sidecar, a partial fetch, or a `.mlmodelc`
  /// copied out of its artifact directory. Distinct from
  /// [`Error::TokenizerLoad`], which means the bytes were read but are not a
  /// valid tokenizer.
  ///
  /// The message and the `source` live on [`ArtifactTokenizerRead`], which this
  /// variant forwards both of through `#[error(transparent)]`.
  #[error(transparent)]
  ArtifactTokenizerRead(#[from] ArtifactTokenizerRead),

  /// The tokenizer failed to load from its JSON definition.
  #[error("failed to load tokenizer: {0}")]
  TokenizerLoad(#[source] tokenizers::Error),

  /// Configuring the tokenizer (truncation) failed.
  #[error("failed to configure tokenizer: {0}")]
  TokenizerConfig(#[source] tokenizers::Error),

  /// The tokenizer's post-processor adds at least as many special tokens as the
  /// fixed [`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS) window holds,
  /// so no text token can fit. Refused before the dependency's unchecked
  /// `max_length - added_tokens` subtraction; see [`SpecialTokenOverhead`] for
  /// why the equal case is refused too.
  #[error(
    "tokenizer post-processor adds {} special tokens, leaving no room for text in the {}-token window",
    .0.added(),
    .0.window()
  )]
  SpecialTokenOverhead(SpecialTokenOverhead),

  /// Encoding text into token ids failed.
  #[error("failed to tokenize text: {0}")]
  Tokenize(#[source] tokenizers::Error),

  /// A tokenizer parsed but does not match the Granite tokenizer/model contract
  /// (vocabulary size, special-token ids, the model's id range, or the pinned
  /// sentinel encoding), so it would produce finite but semantically meaningless
  /// embeddings — or out-of-vocabulary ids the model can only gather as zeros;
  /// OR it is behaviorally valid but not byte-identical (SHA-256) to the pinned
  /// granite `tokenizer.json`, catching corruption or version skew outside the
  /// behavioral checks' coverage. Checked at construction, fail-closed, by every
  /// constructor — BOTH stages, for the artifact sidecar `load` reads and for
  /// caller-supplied bytes (`from_memory` / `from_files`) alike.
  #[error(
    "tokenizer contract mismatch on `{}`: expected {}, got {}",
    .0.check(),
    .0.expected(),
    .0.actual()
  )]
  TokenizerContractMismatch(TokenizerContractMismatch),

  /// The tokenized input exceeded the fixed
  /// [`MAX_TOKENS`](crate::embeddings::granite::MAX_TOKENS) window. Every
  /// constructor forces truncation at that length and disables the tokenizer's
  /// own padding, so this is a defensive backstop — returned instead of an
  /// out-of-bounds panic — against a tokenizer that still yields more ids than
  /// the window (e.g. a padding policy that survived configuration).
  #[error(
    "tokenized input has {} tokens, exceeding the fixed {}-token window",
    .0.got(),
    .0.max()
  )]
  TokenCount(TokenCount),

  /// A token id did not fit the model's `int32` `input_ids` tensor. granite's
  /// vocabulary is far below `i32::MAX`, so this only fires for a foreign
  /// tokenizer with an out-of-range id — returned instead of a silently
  /// wrapping cast.
  ///
  /// Carries the offending token id.
  #[error("token id {0} exceeds the model's int32 input range")]
  TokenIdRange(u32),

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
  #[error(
    "embed_long window budget {} exceeds the model's fixed {}-token input window",
    .0.window(),
    .0.max()
  )]
  WindowOverBudget(WindowOverBudget),

  /// The text handed to [`TextEmbedder::embed_long_with`] exceeds the
  /// caller-configured input byte limit
  /// ([`LongTextOptions::max_input_bytes`]). Enforced BEFORE any tokenizer or
  /// chunker work, so the reject path's cost is independent of the input size —
  /// the limit to set when embedding untrusted text.
  ///
  /// [`TextEmbedder::embed_long_with`]: crate::embeddings::granite::TextEmbedder::embed_long_with
  /// [`LongTextOptions::max_input_bytes`]: crate::embeddings::granite::LongTextOptions::max_input_bytes
  #[error(
    "text input is {} bytes, exceeding the configured {}-byte limit",
    .0.got(),
    .0.max()
  )]
  InputTooLarge(InputTooLarge),

  /// A contentless (separator-only) byte run that must be embedded as one whole
  /// window — the ENTIRE input when it has no tokenizable content, or a
  /// pure-separator gap between packed chunks that neither neighbor can absorb —
  /// measures more tokens than the model's fixed window can hold, so embedding
  /// it would silently drop its suffix tokens. Refused instead (measured with
  /// the non-truncating tokenizer, special tokens included).
  #[error(
    "contentless text at bytes {}..{} measures {} tokens, exceeding the model's fixed {}-token window",
    .0.start(),
    .0.end(),
    .0.tokens(),
    .0.max()
  )]
  ContentlessInputOverBudget(ContentlessInputOverBudget),
}

#[cfg(test)]
mod tests;
