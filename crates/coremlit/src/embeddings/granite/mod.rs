//! Native CoreML **granite** text embeddings — a general sentence-embedding
//! surface whose first model is IBM's
//! `granite-embedding-97m-multilingual-r2` (a ModernBERT encoder with CLS
//! pooling projecting to a 384-dim space).
//!
//! A `&str` in, a unit-norm 384-dim [`Embedding`] out ([`TextEmbedder::embed`]):
//! the bundled granite tokenizer around the fp16 CoreML ModernBERT graph, with
//! L2 normalization applied in Rust.
//!
//! Design spec: `docs/superpowers/specs/2026-07-18-embedkit-design.md`
//! (Amendment 3: granite confirmed, `coremlit::embeddings::granite`, prompt-free,
//! committed-golden oracle).
//!
//! macOS only (built on [`crate`]).
//!
//! # Prompt-free (raw strings)
//!
//! granite-embedding r2 retrieval is **prompt-free**: its
//! `config_sentence_transformers.json` query/document prompts are empty. Feed
//! **raw strings** — no task prefixes. (This is the model's documented retrieval
//! contract; it differs from instruction-tuned embedders.)
//!
//! # Model artifacts
//!
//! The CoreML graph is distributed on the Hugging Face Hub at
//! [`FinDIT-Studio/embedkit-coreml`](https://huggingface.co/FinDIT-Studio/embedkit-coreml),
//! revision `81852f70`, converted from
//! [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
//! (**Apache-2.0**; see the crate `NOTICE`). It is a gitignored dev-time
//! download under `Models/embedkit-granite/`; its per-file SHA-256 and I/O
//! contract are pinned by `tests/granite/model_io.rs`.
//!
//! # Rust front-end around an fp16 CoreML graph
//!
//! The graph emits the **pre-normalization** CLS embedding (`hidden_states[:,
//! 0]` after the final LayerNorm); this module applies the final L2
//! normalization in Rust (keeping the fp16 rsqrt-guard class out of the graph,
//! the workspace convention). The graph takes tokenized `input_ids` /
//! `attention_mask` (`[1, 512]` int32), produced from the bundled granite
//! tokenizer (see [`BUNDLED_TOKENIZER`]).
//!
//! # Committed-golden oracle (no ort)
//!
//! Parity is scored against **committed transformers-fp32 fixtures**
//! (`tests/granite/fixtures/goldens/corpus.json`), never a live ONNX crate — the
//! embedkit "no ort anywhere, not even dev" rule. The hermetic
//! `tests/granite/tokenizer_identity.rs` proves the bundled tokenizer is
//! byte-correct (token-ids match the goldens exactly, no model needed);
//! `tests/granite/parity_embed.rs` scores the CoreML embeddings against the
//! fp32 goldens by cosine (model-gated).
//!
//! # Compute placement (measured, never marketed)
//!
//! Placement is characterized, not asserted (`tests/granite/placement.rs`).
//! Unlike CLAP's audio tower, the granite ModernBERT graph **does** compile for
//! the ANE (the T1 probe measured ~97.8% ANE residency, fp16 cosine 0.99996 vs a
//! `CpuOnly` reference). [`crate::ComputeUnits::All`] (the default) lets CoreML
//! schedule it; the module characterizes the placement rather than claiming it.

pub mod embedding;
pub mod error;

#[cfg(feature = "serde")]
mod compute_units_serde;

pub use embedding::Embedding;
pub use error::Error;

/// windit's window geometry, re-exported as the one windit type in granite's
/// public surface: the per-chunk token budget, overlap, and window cap
/// [`TextEmbedder::embed_long_with`] accepts.
pub use windit::plan::WindowOptions;

use std::{path::Path, sync::OnceLock};

use crate::{ComputeUnits, DataType, Model, MultiArray};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::embeddings::granite::{
  embedding::{EMBEDDING_DIM, check_finite_output},
  error::Result,
};

/// Bytes of the bundled granite `tokenizer.json` compiled into the crate.
///
/// This is the tokenizer from the source model repo
/// [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2),
/// revision `835ad14087e140460703cf0fae09f97d469d65c2` (SHA-256
/// `4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f`) — the
/// exact tokenizer that produced the committed token-id goldens, proven
/// byte-correct by `tests/granite/tokenizer_identity.rs`. Exposed for callers
/// who construct [`TextEmbedder`] via [`TextEmbedder::from_memory`]; the
/// [`TextEmbedder::load`] / [`TextEmbedder::from_file`] constructors use it
/// directly.
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("assets/tokenizer.json");

/// Declared feature names on the granite `.mlmodelc` (pinned by
/// `tests/granite/model_io.rs`).
mod names {
  pub const INPUT_IDS: &str = "input_ids";
  pub const ATTENTION_MASK: &str = "attention_mask";
  pub const EMBEDDING: &str = "embedding";
}

/// Fixed token-sequence length the ModernBERT graph was converted at (the
/// export sequence length, `[1, 512]`). Shorter inputs are right-padded to this
/// length with the mask zeroed on the pad positions; longer inputs are truncated
/// at this length. RoPE makes any fixed length sound, and CLS pooling reads
/// position 0 (never a pad), so the pad token value never reaches the output.
pub const MAX_TOKENS: usize = 512;

/// Default [`TextEmbedderOptions::compute`]: [`ComputeUnits::All`]. The granite
/// ModernBERT graph compiles for the ANE (T1); `All` lets CoreML schedule it.
/// Placement is characterized, not asserted (`tests/granite/placement.rs`).
pub const DEFAULT_COMPUTE: ComputeUnits = ComputeUnits::All;

#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_COMPUTE
}

/// Construction options for [`TextEmbedder`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextEmbedderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_compute",
      with = "crate::embeddings::granite::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for TextEmbedderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl TextEmbedderOptions {
  /// Options matching the module default: [`DEFAULT_COMPUTE`].
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the graph on.
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

/// granite text embedder: a `&str` in, a unit-norm 384-d [`Embedding`] out.
///
/// Tokenizes with the bundled granite tokenizer (truncation `LongestFirst` at
/// [`MAX_TOKENS`] and the tokenizer's own padding disabled, matching the goldens'
/// convention so token ids are identical), right-pads to the fixed `[1, 512]`
/// window with an attention mask, runs the fp16 CoreML ModernBERT graph, and
/// L2-normalizes the pre-normalization CLS projection.
#[derive(Debug)]
pub struct TextEmbedder {
  model: Model,
  tokenizer: Tokenizer,
  /// Right-padding token id for the fixed-length window. The pad positions are
  /// masked to 0, so their embedding is never read, and CLS pooling reads
  /// position 0 (never a pad); this only needs to be a valid vocabulary index.
  /// Resolved from the tokenizer's pad token `<|endoftext|>` at load, else `0`
  /// (a guaranteed-valid vocabulary index).
  pad_id: i32,
  /// Lazily built clone of `tokenizer` with truncation DISABLED — the tokenizer
  /// [`Self::embed_long`] measures chunk lengths with. The embed path's
  /// `tokenizer` truncates at [`MAX_TOKENS`], so its id counts saturate at 512
  /// and would tell the content-aware chunker that EVERY document fits one
  /// window; measurement must see the true, untruncated count. Lazy so
  /// embed-only callers pay nothing, and shared across every `embed_long` call.
  measure_tokenizer: OnceLock<Tokenizer>,
}

impl TextEmbedder {
  /// Loads the granite `.mlmodelc` from `model_path` with the bundled tokenizer
  /// and custom `options` — the primary constructor. Pins the model's I/O
  /// contract against the metadata at load.
  ///
  /// # Errors
  /// As [`Self::from_files`] (with the bundled tokenizer bytes).
  pub fn load(model_path: impl AsRef<Path>, options: TextEmbedderOptions) -> Result<Self> {
    let tokenizer = Tokenizer::from_bytes(BUNDLED_TOKENIZER).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  /// Loads the granite `.mlmodelc` from `model_path` using the bundled tokenizer
  /// and [`TextEmbedderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::from_files`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, TextEmbedderOptions::new())
  }

  /// Loads the model and a `tokenizer.json` from separate file paths.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model / [`Error::ContractMismatch`]
  /// if its I/O contract mismatches; [`Error::TokenizerLoad`] if the tokenizer
  /// JSON is unreadable/invalid; [`Error::TokenizerConfig`] if truncation cannot
  /// be configured.
  pub fn from_files(
    model_path: impl AsRef<Path>,
    tokenizer_json_path: impl AsRef<Path>,
    options: TextEmbedderOptions,
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
    options: TextEmbedderOptions,
  ) -> Result<Self> {
    let tokenizer = Tokenizer::from_bytes(tokenizer_json_bytes).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  fn from_parts(
    model_path: impl AsRef<Path>,
    mut tokenizer: Tokenizer,
    options: TextEmbedderOptions,
  ) -> Result<Self> {
    configure_tokenizer(&mut tokenizer)?;
    // The pad positions are attention-masked to 0 and CLS pooling reads position
    // 0 (never a pad), so the pad token value is immaterial to the output; a
    // valid vocabulary index is all that is required. Resolve granite's pad
    // token `<|endoftext|>` (checked into `i32` range), else fall back to `0`
    // (the BOS/CLS index, always valid).
    let pad_id = tokenizer
      .token_to_id("<|endoftext|>")
      .and_then(|id| i32::try_from(id).ok())
      .unwrap_or(0);

    let model = Model::load(model_path, options.compute())?;
    let description = model.description();

    let ids_expected = format!("[1, {MAX_TOKENS}] int32");
    for name in [names::INPUT_IDS, names::ATTENTION_MASK] {
      let input = description
        .input(name)
        .ok_or_else(|| Error::ContractMismatch {
          feature: name,
          expected: ids_expected.clone(),
          actual: "missing".to_string(),
        })?;
      if input.shape() != [1, MAX_TOKENS] || input.data_type() != Some(DataType::I32) {
        return Err(Error::ContractMismatch {
          feature: name,
          expected: ids_expected.clone(),
          actual: describe(input.shape(), input.data_type()),
        });
      }
    }

    let output_expected = format!("[1, {EMBEDDING_DIM}] float32");
    let output = description
      .output(names::EMBEDDING)
      .ok_or_else(|| Error::ContractMismatch {
        feature: names::EMBEDDING,
        expected: output_expected.clone(),
        actual: "missing".to_string(),
      })?;
    if output.shape() != [1, EMBEDDING_DIM] || output.data_type() != Some(DataType::F32) {
      return Err(Error::ContractMismatch {
        feature: names::EMBEDDING,
        expected: output_expected,
        actual: describe(output.shape(), output.data_type()),
      });
    }

    Ok(Self {
      model,
      tokenizer,
      pad_id,
      measure_tokenizer: OnceLock::new(),
    })
  }

  /// The real token-id sequence for `text` (post-truncation at [`MAX_TOKENS`],
  /// pre-padding, granite special tokens included) — the sequence that is
  /// identity-comparable to the committed goldens
  /// (`tests/granite/tokenizer_identity.rs`).
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

  /// Embeds one text into a unit-norm [`Embedding`]. Prompt-free: feed the raw
  /// string.
  ///
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure; [`Error::TokenCount`] if the tokenized input exceeds [`MAX_TOKENS`]
  /// or [`Error::TokenIdRange`] if a token id is out of `int32` range (both
  /// defensive — the tokenizer config makes neither reachable in practice);
  /// [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML
  /// failure; [`Error::OutputShape`] if the predicted `embedding` shape diverges
  /// from `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] if the model
  /// output has a NaN/infinite component — model corruption, classified apart
  /// from a caller's own non-finite embedding data
  /// ([`Error::NonFiniteEmbedding`]); [`Error::EmbeddingZero`] if the (finite)
  /// projection has zero magnitude.
  pub fn embed(&self, text: &str) -> Result<Embedding> {
    let ids = self.token_ids(text)?;
    self.embed_tokenized(&ids)
  }

  /// Everything after tokenization: right-pads `ids` to the fixed `[1, 512]`
  /// window, runs the CoreML graph, checks the output is finite, and
  /// L2-normalizes it. [`Self::embed`] is [`Self::token_ids`] composed with this;
  /// [`Self::embed_long`] runs it once per content-aware chunk.
  ///
  /// # Errors
  /// As the tensor / prediction / output tail of [`Self::embed`]:
  /// [`Error::TokenCount`] if `ids` exceeds [`MAX_TOKENS`] or
  /// [`Error::TokenIdRange`] if a token id is out of `int32` range (both
  /// defensive); [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML
  /// failure; [`Error::OutputShape`] on a shape divergence;
  /// [`Error::NonFiniteOutput`] on a NaN/infinite model output;
  /// [`Error::EmbeddingZero`] if the projection has zero magnitude.
  fn embed_tokenized(&self, ids: &[u32]) -> Result<Embedding> {
    // Right-pad to the fixed [1, 512] window; real tokens masked 1, pads 0. The
    // tokenizer config guarantees `ids` is real and within the window, but
    // `build_window` still guards it with a typed error instead of a panic.
    let (input_ids, attention_mask) = build_window(ids, self.pad_id)?;

    let ids_tensor = MultiArray::from_slice(&[1, MAX_TOKENS], &input_ids)?;
    let mask_tensor = MultiArray::from_slice(&[1, MAX_TOKENS], &attention_mask)?;
    let mut outputs = self.model.predict_with(&[
      (names::INPUT_IDS, &ids_tensor),
      (names::ATTENTION_MASK, &mask_tensor),
    ])?;
    let embeds =
      outputs
        .take(names::EMBEDDING)
        .ok_or_else(|| crate::PredictionError::MissingOutput {
          name: names::EMBEDDING.to_string(),
        })?;
    if embeds.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape {
        got: embeds.shape().to_vec(),
        expected: vec![1, EMBEDDING_DIM],
      });
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

  /// A clone of the stored tokenizer with truncation DISABLED, built once and
  /// cached in [`Self::measure_tokenizer`]. This is the tokenizer
  /// [`Self::embed_long`] measures chunk lengths with: the embed path's
  /// tokenizer truncates at [`MAX_TOKENS`], so its counts saturate at 512 and
  /// would report every long document as fitting a single window.
  ///
  /// # Errors
  /// [`Error::TokenizerConfig`] if truncation cannot be reconfigured.
  fn measuring_tokenizer(&self) -> Result<&Tokenizer> {
    if let Some(t) = self.measure_tokenizer.get() {
      return Ok(t);
    }
    // Padding is already disabled on the stored tokenizer (construction), and the
    // clone inherits that; only truncation is lifted.
    let mut t = self.tokenizer.clone();
    t.with_truncation(None).map_err(Error::TokenizerConfig)?;
    // Racing initializers build identical values; the loser's clone is dropped.
    let _ = self.measure_tokenizer.set(t);
    Ok(
      self
        .measure_tokenizer
        .get()
        .expect("measure_tokenizer was set just above, on this thread or another"),
    )
  }

  /// Embeds arbitrarily long text: splits it into content-aware chunks of at
  /// most [`MAX_TOKENS`] tokens (respecting paragraph, sentence, and word
  /// boundaries), embeds each chunk with one CoreML prediction, and combines the
  /// per-chunk embeddings by a coverage-weighted spherical mean into one
  /// unit-norm [`Embedding`]. The chunks jointly cover every byte of `text` —
  /// separator bytes the content-aware splitter leaves at chunk boundaries
  /// (paragraph breaks; inter-word punctuation under its oversized-sentence
  /// fallback) are reattached to an adjacent chunk before embedding — so the
  /// aggregate represents the caller's whole text, as `embed` does within one
  /// window. Prompt-free, like [`Self::embed`], and equivalent to
  /// `embed_long_with(text, &WindowOptions::new(MAX_TOKENS))`.
  ///
  /// Text that fits a single window returns exactly [`Self::embed`]'s embedding.
  ///
  /// # Errors
  /// As [`Self::embed_long_with`].
  pub fn embed_long(&self, text: &str) -> Result<Embedding> {
    self.embed_long_with(text, &WindowOptions::new(MAX_TOKENS))
  }

  /// [`Self::embed_long`] with caller-controlled chunk geometry: `opts.window()`
  /// is the per-chunk token budget (must be `1..=`[`MAX_TOKENS`]), the overlap
  /// sets the repeated-token budget between consecutive chunks, and
  /// `opts.max_windows()` caps the final chunk count — separator reattachment
  /// included — and thereby the number of per-chunk CoreML predictions.
  /// `opts.tail()` is ignored — content-aware chunking has no ragged-tail
  /// concept, the final chunk is simply short.
  ///
  /// The per-chunk token budget counts granite's special tokens (`[CLS]`/`[SEP]`,
  /// +2), because both the length measurement and each chunk's embedding run
  /// `encode(s, add_special_tokens = true)` — self-consistent by construction, so
  /// the effective content budget is `window − 2`.
  ///
  /// With `overlap == 0` the chunks partition `text` (the first starts at byte 0,
  /// each begins where the previous ends, the last ends at `text.len()`); a
  /// non-zero overlap additionally repeats trailing regions. Reattached
  /// separators are re-measured against the budget; a pure-separator run neither
  /// neighbor can absorb becomes a chunk of its own and can exceed `window` only
  /// when the run alone measures past it — the same recorded escape as windit's
  /// lone oversized `char`, kept sound by the embed path's truncation guard.
  /// Such insertions count against `opts.max_windows()`: the cap is enforced on
  /// the repaired chunk list, never silently exceeded.
  ///
  /// # Errors
  /// [`Error::WindowOverBudget`] if `opts.window()` exceeds [`MAX_TOKENS`] (every
  /// chunk would be silently truncated); [`Error::EmptyText`] for `""` (as
  /// [`Self::embed`]); [`Error::Tokenize`] on a tokenizer failure;
  /// [`Error::Windowing`] carrying a [`WinditError`](crate::embeddings::granite::error::WinditError)
  /// from chunking (e.g. `ZeroWindow`, `OverlapGeWindow`, `TooManyWindows` — the
  /// `max_windows` cap binds the final, post-reattachment chunk count, `got`
  /// reporting that full count) or
  /// aggregation (e.g. `NonFinite` when the per-chunk embeddings cancel exactly);
  /// plus any per-chunk tensor / prediction / output error (the same set
  /// [`Self::embed`] can raise).
  pub fn embed_long_with(&self, text: &str, opts: &WindowOptions) -> Result<Embedding> {
    validate_long_options(opts)?;
    let chunks = chunk_long(self.measuring_tokenizer()?, text, opts)?;
    match chunks.len() {
      // Empty or whitespace-only text yields no chunks; `embed` defines the
      // contract there (EmptyText for "", a specials-only embedding for
      // whitespace), so delegate to keep those semantics identical.
      0 => self.embed(text),
      // After gap reattachment a single chunk always spans `[0, text.len())`, so
      // this is `embed` on the same bytes — and skipping the one-window
      // aggregation keeps it numerically identical rather than
      // identical-up-to-an-f64-renormalization-rounding.
      1 => {
        let s = chunks[0].as_str(text).expect(
          "chunk_long yields char-aligned boundaries (windit cuts, or 0/len from attach_gaps)",
        );
        self.embed_tokenized(&self.token_ids(s)?)
      }
      _ => {
        let mut windows = Vec::with_capacity(chunks.len());
        // Cumulative token offset; informational only — aggregation reads
        // coverage, not position. Under overlap the offsets overstate positions
        // (overlapped tokens counted twice), which is exactly the double-weighting
        // overlap is meant to express.
        let mut offset = 0usize;
        for chunk in &chunks {
          let s = chunk.as_str(text).expect(
            "chunk_long yields char-aligned boundaries (windit cuts, or 0/len from attach_gaps)",
          );
          let ids = self.token_ids(s)?;
          let embedding = self.embed_tokenized(&ids)?;
          // `embed_tokenized` just proved `ids.len() <= MAX_TOKENS` (build_window's
          // typed guard) and a chunk is never empty, so `Span::new` cannot panic.
          let span = windit::plan::Span::new(offset, ids.len(), MAX_TOKENS);
          windows.push(windit::windowed::Windowed::new(embedding, span));
          offset += ids.len();
        }
        Ok(windit::aggregate::aggregate(
          &windit::aggregate::CoverageWeightedMean,
          &windows,
        )?)
      }
    }
  }
}

/// Overrides the loaded tokenizer's truncation and padding policy to this
/// module's fixed-window contract, so the contract holds for ANY tokenizer
/// (bundled or caller-supplied) regardless of what it carried:
///
/// * **Truncation** `LongestFirst` at [`MAX_TOKENS`], stride 0, right direction —
///   the convention the committed goldens were tokenized under (fixed 512, right
///   truncation), so this module's token ids match them exactly. The export
///   sequence length is a hard model constraint, not a knob.
/// * **Padding disabled** (`with_padding(None)`) — this module does its own
///   fixed-window right-padding in [`build_window`] and masks the pad positions.
///   Leaving an inherited padding policy in place would let pad ids reach
///   [`TextEmbedder::token_ids`] marked as real tokens (corrupt mask), push the
///   CLS token off position 0 under left-padding (wrong CLS pooling), or overflow
///   the window under fixed-padding beyond 512.
fn configure_tokenizer(tokenizer: &mut Tokenizer) -> Result<()> {
  tokenizer
    .with_truncation(Some(TruncationParams {
      max_length: MAX_TOKENS,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .map_err(Error::TokenizerConfig)?;
  tokenizer.with_padding(None);
  Ok(())
}

/// Builds the fixed `[1, `[`MAX_TOKENS`]`]` `input_ids` / `attention_mask` window
/// from the real token `ids`: the real tokens occupy the prefix (mask `1`) and
/// the remainder is right-padded with `pad_id` (mask `0`). CLS therefore stays at
/// position 0 and no pad position is ever attended.
///
/// [`configure_tokenizer`] forces truncation at [`MAX_TOKENS`] and disables the
/// tokenizer's own padding, so `ids` is already real and within the window; this
/// still returns a typed [`Error`] rather than panicking should that contract
/// ever be violated (an over-long or out-of-range id must not become an
/// out-of-bounds write or a wrapping cast).
///
/// # Errors
/// [`Error::TokenCount`] if `ids` exceeds [`MAX_TOKENS`]; [`Error::TokenIdRange`]
/// if a token id does not fit the model's `int32` `input_ids` tensor.
fn build_window(ids: &[u32], pad_id: i32) -> Result<([i32; MAX_TOKENS], [i32; MAX_TOKENS])> {
  if ids.len() > MAX_TOKENS {
    return Err(Error::TokenCount {
      got: ids.len(),
      max: MAX_TOKENS,
    });
  }
  let mut input_ids = [pad_id; MAX_TOKENS];
  let mut attention_mask = [0i32; MAX_TOKENS];
  for (i, &id) in ids.iter().enumerate() {
    input_ids[i] = i32::try_from(id).map_err(|_| Error::TokenIdRange { id })?;
    attention_mask[i] = 1;
  }
  Ok((input_ids, attention_mask))
}

/// Rejects an [`TextEmbedder::embed_long`] chunk budget above the model's fixed
/// input window before any chunking or prediction: a larger budget would let
/// [`TextEmbedder::token_ids`] silently truncate every chunk. Factored out so the
/// check is hermetically testable. `window == 0` and `overlap >= window` are left
/// to windit's own validation (surfacing as [`Error::Windowing`]).
///
/// # Errors
/// [`Error::WindowOverBudget`] if `opts.window()` exceeds [`MAX_TOKENS`].
fn validate_long_options(opts: &WindowOptions) -> Result<()> {
  if opts.window() > MAX_TOKENS {
    return Err(Error::WindowOverBudget {
      window: opts.window(),
      max: MAX_TOKENS,
    });
  }
  Ok(())
}

/// The pure text-splitting stage of [`TextEmbedder::embed_long`]: token-budgeted,
/// boundary-aware byte ranges over `text`, measured with `measure_tok` (the
/// truncation-disabled tokenizer). Model-free, so the chunk geometry is
/// hermetically testable.
///
/// The chunks jointly cover `text`: windit's `ContentAware` extracts tokenized
/// content only, leaving separator bytes (paragraph breaks, whitespace-only
/// interiors, and word-fallback inter-word punctuation) uncovered at chunk
/// boundaries, so [`attach_gaps`] reattaches every such gap — re-measuring the
/// repaired substring against the window — before the chunks are returned. With
/// `overlap == 0` the chunks partition `text` (the first starts at byte 0, each
/// begins where the previous ends, the last ends at `text.len()`); a non-zero
/// overlap covers `text` while preserving its repeats.
///
/// Measurement and per-chunk embedding run the SAME tokenization
/// (`encode(s, add_special_tokens = true)`) on the SAME substring, so a chunk
/// measured at `<= window <= MAX_TOKENS` re-tokenizes to exactly the counted ids
/// and [`build_window`] never truncates or rejects it. Two escapes can exceed the
/// window: windit's lone `char` whose own measure exceeds it, and a
/// pure-separator gap whose run alone measures past it (emitted as its own chunk
/// when neither neighbor can absorb it). Both are unreachable for a real
/// tokenizer on natural text, and even then `token_ids`' truncation plus
/// [`build_window`]'s typed guard keep them sound.
///
/// # Errors
/// [`Error::Windowing`] carrying whatever windit's `ContentAware::chunk` rejects
/// (a zero window, an overlap at or above the window, or a `max_windows`
/// overrun), or `TooManyWindows` raised here when gap reattachment grows the
/// repaired list past `opts.max_windows()` — the cap binds the FINAL chunk
/// count, `got` reporting that full count (windit's own raise aborts at
/// `max + 1`; this one reports the whole overrun).
fn chunk_long(
  measure_tok: &Tokenizer,
  text: &str,
  opts: &WindowOptions,
) -> Result<Vec<windit::split::Chunk>> {
  // Blanket `MeasureText` impl over any `Fn(&str) -> usize`. A tokenizer error
  // folds to `usize::MAX` ("does not fit"), so the chunker descends to a smaller
  // range; a persistent failure resurfaces as `Error::Tokenize` from the
  // per-chunk `token_ids` in `embed_long_with`. The closure cannot stop early
  // (the tokenizers crate exposes no incremental token count on its stable
  // surface), so a giant untrusted input is scanned a few times even under a
  // `max_windows` cap — a recorded limitation, not silently fine.
  let measure = |s: &str| -> usize {
    measure_tok
      .encode(s, true)
      .map(|e| e.get_ids().len())
      .unwrap_or(usize::MAX)
  };
  let chunks = windit::split::ContentAware::new(&measure)
    .chunk(text, opts)
    .map_err(Error::from)?;
  let repaired = attach_gaps(text, chunks, &measure, opts.window());
  // windit enforced `max_windows` on ITS output; each own-chunk the repair
  // inserts grows the count past that check, and every chunk costs one CoreML
  // prediction, so the cap re-binds on the final list. Fail-closed: coverage
  // and the cap cannot both hold here, and silently exceeding the caller's
  // work bound (or silently dropping bytes) would be worse than a typed
  // refusal. `got` is the full repaired count, not windit's abort count.
  if let Some(max) = opts.max_windows()
    && repaired.len() > max
  {
    return Err(Error::Windowing(windit::WinditError::TooManyWindows {
      got: repaired.len(),
      max,
    }));
  }
  Ok(repaired)
}

/// Reattaches the byte gaps windit leaves between chunks, so [`chunk_long`]'s
/// output covers every byte of `text`. windit's `ContentAware` extracts
/// tokenized content only: paragraph separators (`\n\n` runs), whitespace-only
/// paragraph interiors, and — under its oversized-sentence word fallback — the
/// whitespace and punctuation between words are excluded, so a gap opens wherever
/// such bytes fall on a chunk boundary (including a leading gap before the first
/// chunk and a trailing gap after the last).
///
/// A single left-to-right sweep closes every positive gap by re-measuring the
/// exact candidate substring against `window` (BPE is not additive — the repaired
/// range is re-measured, never assumed to gain a fixed token count), trying in
/// order:
///
/// 1. append the gap to the left neighbor if the extended range still fits —
///    left-first because terminal punctuation and paragraph breaks belong to the
///    preceding content, and it keeps every chunk starting where content starts;
/// 2. otherwise prepend it to the right neighbor if that range fits;
/// 3. otherwise emit the gap as its own chunk (pure separator bytes), reachable
///    only when both neighbors are already packed to exactly `window`.
///
/// With `overlap == 0` the result partitions `text`: the first chunk starts at
/// byte 0, each starts where the previous ends, the last ends at `text.len()`,
/// and the chunks concatenate back to `text`. With `overlap > 0` the pre-existing
/// overlaps are negative gaps, left untouched, so coverage is completed without
/// disturbing the repeats. The sweep never fuses two input chunks — each maps to
/// exactly one output chunk — so the output count is the input count plus one
/// per own-chunk emitted; [`chunk_long`] re-enforces `max_windows` on that
/// final count.
///
/// Every accepted attachment re-measures within `window`; the sole escape is a
/// pure-separator own-chunk whose run alone measures past `window` — the same
/// shape as windit's lone oversized-`char` escape, and kept sound the same way:
/// [`TextEmbedder::token_ids`] truncates at [`MAX_TOKENS`] and [`build_window`]
/// guards with a typed error rather than panicking. Every constructed boundary is
/// a windit cut or `0`/`text.len()`, all on `char` boundaries, so `Chunk::new`
/// never straddles a `char` and `as_str` never returns `None`.
fn attach_gaps(
  text: &str,
  chunks: Vec<windit::split::Chunk>,
  measure: &dyn Fn(&str) -> usize,
  window: usize,
) -> Vec<windit::split::Chunk> {
  use windit::split::Chunk;
  let Some(&first) = chunks.first() else {
    return chunks;
  };
  let mut out = Vec::with_capacity(chunks.len());
  let mut cur = first;
  // Leading gap: extend the first chunk left to byte 0, else emit the gap alone.
  if cur.start() > 0 {
    if measure(&text[..cur.end()]) <= window {
      cur = Chunk::new(0, cur.end());
    } else {
      out.push(Chunk::new(0, cur.start()));
    }
  }
  for mut next in chunks.into_iter().skip(1) {
    let (gap_start, gap_end) = (cur.end(), next.start());
    if gap_start < gap_end {
      if measure(&text[cur.start()..gap_end]) <= window {
        cur = Chunk::new(cur.start(), gap_end);
      } else if measure(&text[gap_start..next.end()]) <= window {
        next = Chunk::new(gap_start, next.end());
      } else {
        out.push(cur);
        out.push(Chunk::new(gap_start, gap_end));
        cur = next;
        continue;
      }
    }
    out.push(cur);
    cur = next;
  }
  // Trailing gap: extend the last chunk to `text.len()`, else emit the gap alone.
  if cur.end() < text.len() {
    if measure(&text[cur.start()..]) <= window {
      cur = Chunk::new(cur.start(), text.len());
    } else {
      let tail = Chunk::new(cur.end(), text.len());
      out.push(cur);
      cur = tail;
    }
  }
  out.push(cur);
  out
}

/// Test-only seam: the module's actual tokenizer configuration, without loading
/// a CoreML model — so `tests` can exercise the real tokenization path
/// hermetically (the tokenizer-identity gate).
#[cfg(test)]
pub(crate) fn configured_tokenizer_from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
  let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(Error::TokenizerLoad)?;
  configure_tokenizer(&mut tokenizer)?;
  Ok(tokenizer)
}

/// Test-only seam: the module's MEASURING tokenizer — the production
/// configuration ([`configured_tokenizer_from_bytes`]) with truncation then
/// DISABLED — without loading a CoreML model, so `tests` can exercise the real
/// `chunk_long` measurement path (and pin the truncation hazard) hermetically.
#[cfg(test)]
pub(crate) fn measuring_tokenizer_from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
  let mut tokenizer = configured_tokenizer_from_bytes(bytes)?;
  tokenizer
    .with_truncation(None)
    .map_err(Error::TokenizerConfig)?;
  Ok(tokenizer)
}

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

#[cfg(test)]
mod tests;
