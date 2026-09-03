//! The crate's single error type and `Result` alias.
//!
//! Foreign errors from [`crate`] are wrapped as typed `#[from]` variants;
//! tokenizer errors preserve their `#[source]` chain. Model-contract and
//! embedding-invariant failures are their own variants so callers can match on
//! cause.

use crate::model::contract::{ContractViolation, Rendered};

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// Re-exported so callers (and tests) can name and match the typed error
/// [`Error::Windowing`] carries from the windit windowed-sequence engine (the
/// long-audio window geometry and aggregation).
pub use windit::WinditError;

/// Re-exported so callers can name and match the reason
/// [`Error::PostProcessorTemplate`] carries. The check itself is shared with the
/// crate's other text doors; the reasons are the same for all of them.
pub use crate::embeddings::tokenizer_guard::PostProcessorTemplate;

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

/// A caller-supplied tokenizer's post-processor adds at least as many special
/// tokens as the fixed text window
/// ([`TEXT_MAX_TOKENS`](crate::embeddings::clap::text::TEXT_MAX_TOKENS)) holds,
/// so no text token can fit.
///
/// `tokenizers::Tokenizer::with_truncation` computes its effective window as
/// `max_length - post_processor.added_tokens(false)` with an UNCHECKED `usize`
/// subtraction, and repeats that subtraction on every `encode(_, true)`. A
/// post-processor that over-fills the window therefore panics inside the
/// dependency under overflow checks, and under a release profile wraps to a
/// near-`usize::MAX` window that never truncates — leaving the over-long ids to
/// fail downstream instead. Refused at configuration time, before that
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
  /// The fixed window length
  /// ([`TEXT_MAX_TOKENS`](crate::embeddings::clap::text::TEXT_MAX_TOKENS)).
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
  /// ([`TEXT_MAX_TOKENS`](crate::embeddings::clap::text::TEXT_MAX_TOKENS)).
  #[inline(always)]
  pub const fn window(&self) -> usize {
    self.window
  }
}

/// The tokenized input exceeded the fixed text window
/// ([`TEXT_MAX_TOKENS`](crate::embeddings::clap::text::TEXT_MAX_TOKENS)).
/// Every constructor forces truncation at that length and disables the
/// tokenizer's own padding, so this is a defensive backstop — returned instead
/// of the out-of-bounds write a fixed-size window would otherwise take — against
/// a tokenizer that still yields more ids than the window.
///
/// Payload of [`Error::TokenCount`].
#[derive(Debug)]
pub struct TokenCount {
  /// Number of token ids the tokenizer produced.
  got: usize,
  /// The fixed window length
  /// ([`TEXT_MAX_TOKENS`](crate::embeddings::clap::text::TEXT_MAX_TOKENS)).
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

  /// The fixed window length
  /// ([`TEXT_MAX_TOKENS`](crate::embeddings::clap::text::TEXT_MAX_TOKENS)).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
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

  /// The tokenizer's post-processor adds at least as many special tokens as the
  /// fixed text window holds, so no text token can fit. Refused before the
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
  /// ([`TEXT_MAX_TOKENS`](crate::embeddings::clap::text::TEXT_MAX_TOKENS)).
  /// Every constructor forces truncation at that length and disables the
  /// tokenizer's own padding, so this is a defensive backstop — returned instead
  /// of the out-of-bounds write a fixed-size window would otherwise take.
  #[error("tokenized input has {} tokens, exceeding the fixed {}-token window", .0.got(), .0.max())]
  TokenCount(TokenCount),

  /// A token id did not fit the model's `int32` `input_ids` tensor. CLAP's
  /// RoBERTa vocabulary (50265) is far below `i32::MAX`, so this only fires for
  /// a foreign tokenizer with an out-of-range id — returned instead of a
  /// silently wrapping cast that would hand CoreML a NEGATIVE token id.
  ///
  /// Carries the offending token id.
  #[error("token id {0} exceeds the model's int32 input range")]
  TokenIdRange(u32),

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

/// Map a [`ContractViolation`] into this module's error vocabulary.
///
/// Shared by both CLAP doors, because both hold a `Checked` and both report
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
