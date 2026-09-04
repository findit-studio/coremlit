//! The CLAP [`TextEncoder`]: the pinned Xenova tokenizer around the fp16 CoreML
//! RoBERTa graph, with L2 normalization applied in Rust.

use std::path::Path;

use crate::{
  ComputeUnits, DataType, Model, MultiArray,
  model::contract::{Checked, Dim, FeatureContract, LoadContract, StateContract},
};
use tokenizers::{
  PostProcessor, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

use crate::embeddings::clap::{
  embedding::{EMBEDDING_DIM, Embedding, check_finite_output},
  error::{Error, OutputShape, Result, SpecialTokenOverhead, TokenCount, contract_violation},
};

/// Declared feature names on `clap_text.mlmodelc` (pinned by
/// `tests/clap/text_model_io.rs`).
mod names {
  pub const INPUT_IDS: &str = "input_ids";
  pub const ATTENTION_MASK: &str = "attention_mask";
  pub const TEXT_EMBEDS: &str = "text_embeds";
}

/// Fixed token-sequence length the RoBERTa graph was converted at (the model's
/// max, `[1, 512]`). Shorter inputs are right-padded to this length with the mask
/// zeroed on the pad positions, which reproduces the natural-length embedding
/// EXACTLY (T1 verified cos = 1.0); longer inputs are truncated at this length,
/// so they can never index past the position table.
pub const TEXT_MAX_TOKENS: usize = 512;

/// Default [`TextEncoderOptions::compute`]: [`ComputeUnits::CpuAndGpu`] — a
/// **measure-then-pin** default, moved off `All` by the issue #30 perf pass (the
/// committed `clap_encode` bench, `benches/clap/encode.rs`).
///
/// The RoBERTa text graph is tiny (a fixed `[1, 512]` window). On `All`, CoreML
/// pulls the ANE into the schedule, and for a graph this small the per-dispatch
/// coordination cost dominates the actual compute — so `All` is *slower* than
/// scheduling on the GPU alone. Measured **warm-median** latency (Apple M1 Max,
/// macOS 26.5 25F71, fp16, 25 runs, consistent across 3 sweeps):
///
/// | unit                      | warm median | cosine vs `CpuOnly` |
/// |---------------------------|-------------|---------------------|
/// | `CpuAndGpu` (new default) | ~16.8 ms    | 0.999956            |
/// | `CpuAndNeuralEngine`      | ~28.6 ms    | 0.999950            |
/// | `All` (former default)    | ~29.7 ms    | 0.999947            |
/// | `CpuOnly`                 | ~42.1 ms    | 1.000000 (ref)      |
///
/// `CpuAndGpu` is ~43 % faster than the former `All` default and holds the
/// cross-placement parity floor the `tests/clap/placement.rs` gate pins (0.9999).
/// Only the *default* moves — every unit stays selectable via
/// [`TextEncoderOptions::with_compute`] / [`TextEncoderOptions::set_compute`]
/// (`All` and `CpuOnly` remain parity-clean). The text graph **does** compile for
/// the ANE (unlike the audio graph); placement is characterized, not asserted.
pub const DEFAULT_TEXT_COMPUTE: ComputeUnits = ComputeUnits::CpuAndGpu;

#[cfg(feature = "serde")]
fn default_text_compute() -> ComputeUnits {
  DEFAULT_TEXT_COMPUTE
}

/// Construction options for [`TextEncoder`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextEncoderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_text_compute",
      with = "crate::embeddings::clap::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for TextEncoderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl TextEncoderOptions {
  /// Options matching the crate default: [`DEFAULT_TEXT_COMPUTE`].
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_TEXT_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the text graph on.
  #[inline]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }

  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }

  /// Sets [`Self::compute`] in place.
  #[inline]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// CLAP text encoder: a `&str` in, a unit-norm 512-d [`Embedding`] out.
///
/// Tokenizes with the pinned Xenova tokenizer (truncation `LongestFirst` at
/// [`TEXT_MAX_TOKENS`], matching textclap so token ids are identical), right-pads
/// to the fixed `[1, 512]` window with an attention mask, runs the fp16 CoreML
/// RoBERTa graph, and L2-normalizes the pre-normalization projection.
#[derive(Debug)]
pub struct TextEncoder {
  /// A [`Checked`], never a bare [`Model`]: [`text_contract`] is the only
  /// contract this door states and [`Checked::new`] is the only way one is
  /// built, so removing the check from [`Self::from_parts`] does not compile.
  model: Checked,
  tokenizer: Tokenizer,
  /// Right-padding token id for the fixed-length window. The pad positions are
  /// masked to 0, so their embedding is never read (T1 verified pad-to-512 +
  /// mask reproduces the natural-length embedding exactly); this only needs to
  /// be a valid vocabulary index. Resolved from `<pad>` at load, else RoBERTa's
  /// conventional pad id `1`.
  pad_id: i32,
}

impl TextEncoder {
  /// Loads `clap_text.mlmodelc` from `model_path` using the crate's bundled
  /// tokenizer ([`crate::embeddings::clap::BUNDLED_TOKENIZER`]) and [`TextEncoderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::from_files`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::from_bundled_tokenizer(model_path, TextEncoderOptions::new())
  }

  /// Loads the model from `model_path` with the bundled tokenizer and custom
  /// options.
  ///
  /// # Errors
  /// As [`Self::from_files`] (with the bundled tokenizer bytes).
  pub fn from_bundled_tokenizer(
    model_path: impl AsRef<Path>,
    options: TextEncoderOptions,
  ) -> Result<Self> {
    let tokenizer = Tokenizer::from_bytes(crate::embeddings::clap::BUNDLED_TOKENIZER)
      .map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  /// Loads the model and a `tokenizer.json` from separate file paths.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model / [`Error::ContractMismatch`]
  /// if its I/O contract mismatches; [`Error::TokenizerLoad`] if the tokenizer
  /// JSON is unreadable/invalid; [`Error::PostProcessorTemplate`] /
  /// [`Error::SpecialTokenOverhead`] / [`Error::TokenizerConfig`] if the
  /// tokenizer cannot be configured to this door's fixed window
  /// (`configure_tokenizer`); [`Error::TokenIdRange`] if its `<pad>` id does
  /// not fit the model's `int32` `input_ids`.
  pub fn from_files(
    model_path: impl AsRef<Path>,
    tokenizer_json_path: impl AsRef<Path>,
    options: TextEncoderOptions,
  ) -> Result<Self> {
    let tokenizer =
      Tokenizer::from_file(tokenizer_json_path.as_ref()).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  /// Loads the model from a path and the tokenizer from caller-supplied bytes.
  ///
  /// # Errors
  /// As [`Self::from_files`].
  pub fn from_memory(
    model_path: impl AsRef<Path>,
    tokenizer_json_bytes: &[u8],
    options: TextEncoderOptions,
  ) -> Result<Self> {
    let tokenizer = Tokenizer::from_bytes(tokenizer_json_bytes).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  fn from_parts(
    model_path: impl AsRef<Path>,
    mut tokenizer: Tokenizer,
    options: TextEncoderOptions,
  ) -> Result<Self> {
    configure_tokenizer(&mut tokenizer)?;
    let pad_id = resolve_pad_id(&tokenizer)?;

    let model = Model::load(model_path, options.compute())?;
    let model = Checked::new(model, &text_contract()).map_err(contract_violation)?;

    Ok(Self {
      model,
      tokenizer,
      pad_id,
    })
  }

  /// The real token-id sequence for `text` (post-truncation at
  /// [`TEXT_MAX_TOKENS`], pre-padding, RoBERTa special tokens included) — the
  /// sequence that is identity-comparable to textclap (`tests/clap/tokenizer_identity.rs`).
  ///
  /// Tokenization runs over the whole of `text` before any truncation, so the
  /// cost is linear in the input's length and the input budget is the caller's
  /// (#118).
  ///
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure.
  pub fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
    if text.is_empty() {
      return Err(Error::EmptyText);
    }
    let encoding = self.tokenizer.encode(text, true).map_err(Error::Tokenize)?;
    Ok(encoding.get_ids().to_vec())
  }

  /// Embeds one text query into a unit-norm [`Embedding`].
  ///
  /// Tokenization runs over the whole of `text` before any truncation, so the
  /// cost is linear in the input's length and the input budget is the caller's
  /// (#118).
  ///
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure; [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML
  /// failure; [`Error::OutputShape`] if the predicted `text_embeds` shape
  /// diverges from `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] if the
  /// model output has a NaN/infinite component — model corruption, classified
  /// apart from a caller's own non-finite embedding data
  /// ([`Error::NonFiniteEmbedding`]); [`Error::EmbeddingZero`] if the (finite)
  /// projection has zero magnitude; [`Error::TokenCount`] /
  /// [`Error::TokenIdRange`] on a window guard (defensive — truncation caps the
  /// count and the vocabulary is far inside `int32`).
  pub fn embed(&self, text: &str) -> Result<Embedding> {
    let ids = self.token_ids(text)?;
    let (input_ids, attention_mask) = build_window(&ids, self.pad_id)?;

    let ids_tensor = MultiArray::from_slice(&[1, TEXT_MAX_TOKENS], &input_ids)?;
    let mask_tensor = MultiArray::from_slice(&[1, TEXT_MAX_TOKENS], &attention_mask)?;
    let mut outputs = self.model.predict_with(&[
      (names::INPUT_IDS, &ids_tensor),
      (names::ATTENTION_MASK, &mask_tensor),
    ])?;
    let embeds = outputs
      .take(names::TEXT_EMBEDS)
      .ok_or_else(|| crate::PredictionError::MissingOutput(names::TEXT_EMBEDS.to_string()))?;
    if embeds.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape(OutputShape::new(
        embeds.shape().to_vec(),
        vec![1, EMBEDDING_DIM],
      )));
    }

    let mut row = [0.0f32; EMBEDDING_DIM];
    embeds.copy_into::<f32>(&mut row)?;
    // Classify a NaN/∞ the CoreML runtime produced as model-output corruption
    // (`NonFiniteOutput`) before it reaches `from_slice_normalizing`, which would
    // otherwise mislabel it as caller-supplied embedding data
    // (`NonFiniteEmbedding`).
    check_finite_output(&row)?;
    Embedding::from_slice_normalizing(&row)
  }

  /// Runs one throwaway [`Self::embed`] to fully specialize the prediction path,
  /// so the first user-facing request is warm.
  ///
  /// Construction ([`Self::from_file`] &c.) already pays the model *load* /
  /// device specialization; what it does **not** pay is the first prediction's
  /// own graph specialization. The `clap_encode` bench measures that first
  /// inference at several times the warm latency (e.g. ~120 ms first vs ~17 ms
  /// warm on `CpuAndGpu`), so calling `prewarm` once — after construction, before
  /// serving — moves that one-time cost off the first real query. Then **reuse**
  /// this same encoder for every request (it is `&self`, so it stays resident and
  /// there is nothing to reconstruct).
  ///
  /// This is the whole prewarm delta over the construct-once-and-reuse pattern:
  /// the load is the constructor's job, and this is deliberately *only* the dummy
  /// inference the reuse pattern otherwise leaves for the first live request.
  ///
  /// # Errors
  /// As [`Self::embed`] (the warm-up query is a fixed non-empty string, so the
  /// empty-text path cannot fire); a failure here surfaces a broken model at
  /// prewarm time rather than on the first request.
  pub fn prewarm(&self) -> Result<()> {
    self.embed("warmup")?;
    Ok(())
  }
}

/// Overrides the loaded tokenizer's truncation and padding policy to this
/// module's fixed-window contract, so the contract holds for ANY tokenizer
/// (bundled or caller-supplied) regardless of what it carried:
///
/// * **Truncation** `LongestFirst` at [`TEXT_MAX_TOKENS`], stride 0, right
///   direction — identical to textclap's `force_max_length_truncation`
///   (`textclap/src/text.rs`), so clapkit's token ids match textclap's on the
///   identical tokenizer artifact. The position table is a hard model
///   constraint, not a knob.
/// * **Padding disabled** (`with_padding(None)`) — this module does its own
///   fixed-window right-padding in [`build_window`] and masks the pad positions.
///   The tokenizer pads AFTER truncating, so an inherited `Fixed` (or
///   `pad_to_multiple_of`) policy longer than the window pushes
///   [`TextEncoder::token_ids`] past `[1, 512]` entirely, and a shorter one
///   hands [`build_window`] pad ids marked as real tokens with mask `1`.
///
/// The tokenizer is a caller input on the `from_files` / `from_memory` paths, so
/// this is also where it is checked against the window it has to agree with:
/// first the post-processor's STRUCTURE (see
/// [`crate::embeddings::tokenizer_guard`]), then its special-token overhead (see
/// [`SpecialTokenOverhead`]). The overhead is a number computed FROM the
/// template, and `count_added` scores an undeclared `SpecialToken` id as ZERO,
/// so a malformed template does not read as a large overhead — it reads as no
/// overhead at all. The two guards therefore refuse the same set of tokenizers
/// whichever order they run in; judging the STRUCTURE first is what makes a
/// tokenizer that breaks both rules report its root defect rather than a derived
/// count the caller would try to shrink.
///
/// The structural check is also what makes the overhead reading COMPLETE. It
/// admits only chains whose every token-adding post-processor runs in single
/// mode and whose template places the text exactly once, and for those
/// `added_tokens(false)` is exactly what the chain adds and the text appears
/// exactly once — so a raw encoding truncated to `max_length - added`
/// post-processes to at most `max_length`, which is what the test below is
/// worth.
///
/// # Errors
/// [`Error::PostProcessorTemplate`] if the post-processor is not one this door's
/// single-sequence `encode` can be trusted with — it would panic inside the
/// dependency, drop the text, place it more than once, or run in a mode whose
/// overhead `added_tokens(false)` does not report;
/// [`Error::SpecialTokenOverhead`] if the tokenizer's post-processor adds at
/// least [`TEXT_MAX_TOKENS`] special tokens, leaving no room for text;
/// [`Error::TokenizerConfig`] if the tokenizer rejects the truncation policy.
fn configure_tokenizer(tokenizer: &mut Tokenizer) -> Result<()> {
  crate::embeddings::tokenizer_guard::check_post_processor(tokenizer)
    .map_err(Error::PostProcessorTemplate)?;
  // `Tokenizer::with_truncation` computes the effective text window as
  // `max_length - post_processor.added_tokens(false)` with an UNCHECKED usize
  // subtraction, and `encode(_, true)` — which this door always calls — repeats
  // it. Read the same number off the public `PostProcessor` trait and refuse the
  // tokenizer while the arithmetic is still ours. `>=` rather than the
  // dependency's `>`: the equal case subtracts cleanly to a ZERO-token text
  // window, whose every encoding is the special tokens alone.
  let added = tokenizer
    .get_post_processor()
    .map_or(0, |post| post.added_tokens(false));
  if added >= TEXT_MAX_TOKENS {
    return Err(Error::SpecialTokenOverhead(SpecialTokenOverhead::new(
      added,
      TEXT_MAX_TOKENS,
    )));
  }
  tokenizer
    .with_truncation(Some(TruncationParams {
      max_length: TEXT_MAX_TOKENS,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .map_err(Error::TokenizerConfig)?;
  tokenizer.with_padding(None);
  Ok(())
}

/// RoBERTa's conventional padding id, used when the tokenizer's vocabulary has
/// no `<pad>` entry to resolve.
const FALLBACK_PAD_ID: i32 = 1;

/// The right-padding token id for the fixed window, resolved from the
/// tokenizer's `<pad>` entry, else [`FALLBACK_PAD_ID`].
///
/// The tokenizer is a caller input on the `from_files` / `from_memory` paths, so
/// the vocabulary index is CONVERTED rather than cast: `token_to_id` yields a
/// `u32` and the model's `input_ids` tensor is `int32`, so an id above
/// `i32::MAX` would otherwise wrap to a NEGATIVE id and gather the wrong row of
/// the embedding table.
///
/// # Errors
/// [`Error::TokenIdRange`] if `<pad>` resolves to an id outside `int32`.
fn resolve_pad_id(tokenizer: &Tokenizer) -> Result<i32> {
  tokenizer
    .token_to_id("<pad>")
    .map_or(Ok(FALLBACK_PAD_ID), |id| {
      i32::try_from(id).map_err(|_| Error::TokenIdRange(id))
    })
}

/// Builds the fixed `[1, `[`TEXT_MAX_TOKENS`]`]` `input_ids` / `attention_mask`
/// window from the real token `ids`: the real tokens occupy the prefix (mask
/// `1`) and the remainder is right-padded with `pad_id` (mask `0`).
///
/// [`configure_tokenizer`] forces truncation at [`TEXT_MAX_TOKENS`] and disables
/// the tokenizer's own padding, so `ids` is already real and within the window;
/// this still returns a typed [`Error`] rather than panicking should that
/// contract ever be violated. Both halves earn their place: writing an over-long
/// `ids` into this fixed-size window is an out-of-bounds index, which a
/// `debug_assert!` documents but does not prevent in release, and an
/// out-of-range id becomes a negative one under a wrapping cast.
///
/// # Errors
/// [`Error::TokenCount`] if `ids` exceeds [`TEXT_MAX_TOKENS`];
/// [`Error::TokenIdRange`] if a token id does not fit the model's `int32`
/// `input_ids` tensor.
fn build_window(
  ids: &[u32],
  pad_id: i32,
) -> Result<([i32; TEXT_MAX_TOKENS], [i32; TEXT_MAX_TOKENS])> {
  if ids.len() > TEXT_MAX_TOKENS {
    return Err(Error::TokenCount(TokenCount::new(
      ids.len(),
      TEXT_MAX_TOKENS,
    )));
  }
  let mut input_ids = [pad_id; TEXT_MAX_TOKENS];
  let mut attention_mask = [0i32; TEXT_MAX_TOKENS];
  for (i, &id) in ids.iter().enumerate() {
    input_ids[i] = i32::try_from(id).map_err(|_| Error::TokenIdRange(id))?;
    attention_mask[i] = 1;
  }
  Ok((input_ids, attention_mask))
}

/// Test-only seam: the crate's actual tokenizer configuration, without loading a
/// CoreML model — so `tests` can exercise the real tokenization path
/// hermetically (the tokenizer-identity gate).
#[cfg(test)]
pub(crate) fn configured_tokenizer_from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
  let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(Error::TokenizerLoad)?;
  configure_tokenizer(&mut tokenizer)?;
  Ok(tokenizer)
}

/// The load contract this door states: `input_ids` and `attention_mask` both
/// `[1, 512]` i32 in, `text_embeds` `[1, 512]` f32 out, no state.
///
/// Data rather than a sequence of checks, and the ONLY thing
/// [`TextEncoder::from_parts`] does to the model beyond calling
/// [`Model::load`]. The six hand-written comparisons this replaced — a
/// presence test and a shape-and-dtype test per feature — were each a check
/// the constructor could forget to make, and deleting any of them failed no
/// runnable test. A [`Checked`] field turns that mutation into a compile error.
///
/// Every axis is [`Dim::Exactly`], and that buys more than the numbers.
/// [`crate::FeatureInfo::shape`] reports the DEFAULT shape of a flexible
/// input, so a `RangeDims` graph converted at `[1, 512]` declares this
/// contract's exact numbers. An all-`Exactly` contract therefore requires the
/// whole feature to be [`crate::ShapeConstraint::Fixed`], which is the only
/// thing that separates the two — and which matters here twice over, because
/// the window is what this door PADS to: a graph that would also accept a
/// shorter sequence is one whose exports differ in what the mask means.
/// Nothing is read back off the artifact: the fp16 and int8 tiers are
/// contract-identical, so every number is this door's own.
fn text_contract() -> LoadContract {
  let window = vec![Dim::Exactly(1), Dim::Exactly(TEXT_MAX_TOKENS)];
  LoadContract::new(
    vec![
      FeatureContract::new(names::INPUT_IDS, DataType::I32, window.clone()),
      FeatureContract::new(names::ATTENTION_MASK, DataType::I32, window),
    ],
    vec![FeatureContract::new(
      names::TEXT_EMBEDS,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(EMBEDDING_DIM)],
    )],
    StateContract::None,
  )
}

#[cfg(test)]
mod tests;
