//! The siglip [`TextEmbedder`]: the bundled Gemma tokenizer around the fp16
//! CoreML text graph, with L2 normalization applied in Rust.
//!
//! Text is lowercased before tokenization (the SigLIP2 training convention;
//! checkpoint `do_lower_case: true`), mirroring transformers `Siglip2Tokenizer`.
//!
//! Unlike `granite`/`clap`, the SigLIP text graph takes **only** `input_ids`
//! (`[1, T]` int32) — the processor emits no attention mask (canonical SigLIP
//! attends all `T` positions) and the tower pools the final position. Because
//! every position is attended and the pooled token is positional, the pad id AND
//! pad side are semantically load-bearing (D6); the built window is compared
//! byte-for-byte against the committed goldens by the Wave B token-identity gate,
//! which pins them empirically.

use std::path::Path;

use crate::{ComputeUnits, DataType, Model, ModelDescription, MultiArray};
use tokenizers::{
  Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
  normalizers::{Lowercase, NormalizerWrapper, Sequence as NormalizerSequence},
};

use crate::embeddings::siglip::{
  embedding::{EMBEDDING_DIM, Embedding, check_finite_output},
  error::{Error, Result},
};

/// Declared feature names on the siglip text `.mlmodelc` (pinned by
/// `tests/siglip/text_model_io.rs`). There is deliberately no `attention_mask`
/// — the graph has a single input.
mod names {
  pub const INPUT_IDS: &str = "input_ids";
  pub const TEXT_FEATURES: &str = "text_features";
}

/// Sentinel embedded in the Wave-A placeholder `assets/tokenizer.json`; the real
/// source-revision Gemma artifact cannot contain it. Kept after Wave B as a
/// regression guard against re-committing the placeholder.
const PLACEHOLDER_SENTINEL: &[u8] =
  b"PLACEHOLDER_REPLACE_WITH_SOURCE_REVISION_GEMMA_TOKENIZER_IN_WAVE_B";

/// Fails closed if `bytes` is the build-time placeholder tokenizer, whose vocab
/// maps every ordinary word to `<pad>` (so embedding with it would silently
/// yield meaningless vectors). Called before any tokenizer parse or model load
/// so the failure is deterministic and hermetically testable.
///
/// # Errors
/// [`Error::TokenizerPlaceholder`] if `bytes` carries the placeholder sentinel.
fn ensure_not_placeholder(bytes: &[u8]) -> Result<()> {
  // The real Gemma artifact is tens of MB; only a small file can be the
  // placeholder, so the scan is skipped once the real bytes are staged.
  if bytes.len() < 1_000_000
    && bytes
      .windows(PLACEHOLDER_SENTINEL.len())
      .any(|w| w == PLACEHOLDER_SENTINEL)
  {
    return Err(Error::TokenizerPlaceholder);
  }
  Ok(())
}

/// Default [`TextEmbedderOptions::compute`]: [`ComputeUnits::CpuAndGpu`] — the
/// measured floor-holding placement.
///
/// The conversion probe measured the text tower's whole-graph ANE compile as
/// **failing** (`ANECCompile() FAILED`), so CoreML runs it on the GPU regardless;
/// forcing [`ComputeUnits::CpuAndNeuralEngine`] is **7–10× slower** (58.5 ms vs
/// 6.0 ms at batch 1) as it re-attempts the failing compile on every load. On the
/// GPU the fp16 parity is granite-class (**0.999998**). `CpuAndGpu` pins the
/// floor-holding GPU path and skips the ANE-dispatch cost (mirroring `clap`'s
/// measure-then-pin `text` default). Every unit stays selectable via
/// [`TextEmbedderOptions::with_compute`] / [`TextEmbedderOptions::set_compute`];
/// placement is characterized, not asserted (`tests/siglip/placement.rs`).
pub const DEFAULT_TEXT_COMPUTE: ComputeUnits = ComputeUnits::CpuAndGpu;

#[cfg(feature = "serde")]
fn default_text_compute() -> ComputeUnits {
  DEFAULT_TEXT_COMPUTE
}

/// Construction options for [`TextEmbedder`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextEmbedderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_text_compute",
      with = "crate::embeddings::siglip::compute_units_serde"
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
  /// Options matching the module default: [`DEFAULT_TEXT_COMPUTE`].
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

/// Which side of the fixed window the padding occupies. SigLIP's final-position
/// pooling makes this semantically load-bearing (D6); the concrete value is
/// pinned empirically by the Wave B token-identity goldens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PadSide {
  /// Real tokens occupy the prefix; pads fill the suffix.
  Right,
  /// Pads fill the prefix; real tokens occupy the suffix. Reserved for the
  /// Wave B pinned convention (production currently pads [`PadSide::Right`]);
  /// exercised by the hermetic `build_window` tests.
  #[allow(dead_code)]
  Left,
}

/// siglip text embedder: a `&str` in, a unit-norm 768-d [`Embedding`] out — the
/// same joint-space [`Embedding`] the image tower emits.
///
/// Lowercases the text (SigLIP2 convention; checkpoint `do_lower_case: true`),
/// tokenizes with the bundled Gemma tokenizer (truncation `LongestFirst` at the
/// resolved window `T`, the tokenizer's own padding disabled), builds the fixed
/// `[1, T]` padded window (side/id per D6), runs the single-input fp16 CoreML
/// graph, and L2-normalizes the pre-normalization projection.
#[derive(Debug)]
pub struct TextEmbedder {
  model: Model,
  tokenizer: Tokenizer,
  /// Padding token id for the fixed-length window. SigLIP attends every position
  /// and pools the final one, so this is semantically load-bearing (D6);
  /// resolved from the tokenizer's `<pad>` at load, else `0`. Pinned by the
  /// Wave B token-identity goldens.
  pad_id: i32,
  /// Padding side for the fixed-length window (D6). Provisionally [`PadSide::Right`];
  /// pinned by the Wave B token-identity goldens.
  pad_side: PadSide,
  /// The text window length `T` resolved from the loaded model's `input_ids [1,
  /// T]` contract (D2 — never a code constant).
  max_tokens: usize,
}

impl TextEmbedder {
  /// Loads the text `.mlmodelc` from `model_path` with the bundled tokenizer and
  /// custom `options` — the primary constructor. Resolves the window `T` and
  /// validates the I/O contract against the metadata at load.
  ///
  /// # Errors
  /// [`Error::TokenizerPlaceholder`] if the bundled `tokenizer.json` is still the
  /// build-time placeholder (fails closed before any I/O); otherwise as
  /// [`Self::from_files`] (with the bundled tokenizer bytes).
  pub fn load(model_path: impl AsRef<Path>, options: TextEmbedderOptions) -> Result<Self> {
    ensure_not_placeholder(crate::embeddings::siglip::BUNDLED_TOKENIZER)?;
    let tokenizer = Tokenizer::from_bytes(crate::embeddings::siglip::BUNDLED_TOKENIZER)
      .map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  /// Loads the text `.mlmodelc` from `model_path` using the bundled tokenizer and
  /// [`TextEmbedderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::from_files`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, TextEmbedderOptions::new())
  }

  /// Loads the model and a `tokenizer.json` from separate file paths. The
  /// caller-supplied file is deliberately NOT placeholder-checked — a
  /// caller-chosen tokenizer is the caller's contract; the placeholder ships
  /// only as the bundled bytes that [`Self::load`] / [`Self::from_memory`] guard.
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
  /// [`Error::TokenizerPlaceholder`] if `tokenizer_json_bytes` is the build-time
  /// placeholder (e.g. the current [`crate::embeddings::siglip::BUNDLED_TOKENIZER`];
  /// fails closed before any I/O); otherwise as [`Self::from_files`].
  pub fn from_memory(
    model_path: impl AsRef<Path>,
    tokenizer_json_bytes: &[u8],
    options: TextEmbedderOptions,
  ) -> Result<Self> {
    ensure_not_placeholder(tokenizer_json_bytes)?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_json_bytes).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  fn from_parts(
    model_path: impl AsRef<Path>,
    mut tokenizer: Tokenizer,
    options: TextEmbedderOptions,
  ) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let max_tokens = resolve_text_window(model.description())?;
    configure_tokenizer(&mut tokenizer, max_tokens)?;
    let pad_id = tokenizer
      .token_to_id("<pad>")
      .and_then(|id| i32::try_from(id).ok())
      .unwrap_or(0);
    Ok(Self {
      model,
      tokenizer,
      pad_id,
      pad_side: PadSide::Right,
      max_tokens,
    })
  }

  /// The text window length `T` this model was converted at — resolved from the
  /// loaded `input_ids [1, T]` contract (D2), not a code constant.
  #[inline]
  pub const fn max_tokens(&self) -> usize {
    self.max_tokens
  }

  /// The fixed `[T]` **padded** `input_ids` window for `text` (lowercased, then
  /// post-truncation, then padded to `T` on the pinned side with the pad id) —
  /// the exact sequence fed to the graph, and the one the Wave B token-identity
  /// gate compares byte-for-byte against the committed goldens.
  ///
  /// This deliberately differs from `granite::token_ids` (which returns the
  /// UNPADDED ids): SigLIP attends every position and pools the final one, so the
  /// pad positions are part of the semantic input and belong in the window (D6).
  ///
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure; [`Error::TokenCount`] if the tokenized input exceeds the window
  /// (defensive — truncation caps it); [`Error::TokenIdRange`] if a token id is
  /// out of `int32` range.
  pub fn token_ids(&self, text: &str) -> Result<Vec<i32>> {
    if text.is_empty() {
      return Err(Error::EmptyText);
    }
    let encoding = self.tokenizer.encode(text, true).map_err(Error::Tokenize)?;
    build_window(
      encoding.get_ids(),
      self.pad_id,
      self.pad_side,
      self.max_tokens,
    )
  }

  /// Embeds one text into a unit-norm [`Embedding`].
  ///
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure; [`Error::TokenCount`] / [`Error::TokenIdRange`] on a window guard;
  /// [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML failure;
  /// [`Error::OutputShape`] if the predicted `text_features` shape diverges from
  /// `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] if the model output
  /// has a NaN/infinite component — model corruption, classified apart from a
  /// caller's own non-finite embedding data ([`Error::NonFiniteEmbedding`]);
  /// [`Error::EmbeddingZero`] if the (finite) projection has zero magnitude.
  pub fn embed(&self, text: &str) -> Result<Embedding> {
    let ids = self.token_ids(text)?;
    let ids_tensor = MultiArray::from_slice(&[1, self.max_tokens], &ids)?;
    // Single input: no attention_mask (the SigLIP text graph has none).
    let mut outputs = self
      .model
      .predict_with(&[(names::INPUT_IDS, &ids_tensor)])?;
    let feats =
      outputs
        .take(names::TEXT_FEATURES)
        .ok_or_else(|| crate::PredictionError::MissingOutput {
          name: names::TEXT_FEATURES.to_string(),
        })?;
    if feats.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape {
        got: feats.shape().to_vec(),
        expected: vec![1, EMBEDDING_DIM],
      });
    }

    let mut row = [0.0f32; EMBEDDING_DIM];
    feats.copy_into::<f32>(&mut row)?;
    // Classify a NaN/∞ the CoreML runtime produced as model-output corruption
    // (`NonFiniteOutput`) before it reaches `from_slice_normalizing`.
    check_finite_output(&row)?;
    Embedding::from_slice_normalizing(&row)
  }

  /// Runs one throwaway [`Self::embed`] to fully specialize the prediction path,
  /// so the first user-facing request is warm. Construction pays the model load;
  /// this pays the first prediction's graph specialization. Then **reuse** this
  /// same embedder for every request (it is `&self`).
  ///
  /// # Errors
  /// As [`Self::embed`] (the warm-up query is a fixed non-empty string, so the
  /// empty-text path cannot fire); a failure surfaces a broken model at prewarm
  /// time rather than on the first request.
  pub fn prewarm(&self) -> Result<()> {
    self.embed("warmup")?;
    Ok(())
  }
}

/// Resolves the text window `T` from the loaded model's `input_ids [1, T]`
/// contract (D2) and validates the `text_features [1, 768]` output. The
/// exact-input-SET assertion (that `input_ids` is the ONLY input — no
/// `attention_mask`) is the Wave C `tests/siglip/text_model_io.rs` gate.
fn resolve_text_window(description: &ModelDescription) -> Result<usize> {
  let ids_expected = "[1, T] int32";
  let input = description
    .input(names::INPUT_IDS)
    .ok_or_else(|| Error::ContractMismatch {
      feature: names::INPUT_IDS,
      expected: ids_expected.to_string(),
      actual: "missing".to_string(),
    })?;
  let shape = input.shape();
  if shape.len() != 2 || shape[0] != 1 || input.data_type() != Some(DataType::I32) {
    return Err(Error::ContractMismatch {
      feature: names::INPUT_IDS,
      expected: ids_expected.to_string(),
      actual: describe(shape, input.data_type()),
    });
  }
  let t = shape[1];
  if t == 0 {
    return Err(Error::ContractMismatch {
      feature: names::INPUT_IDS,
      expected: ids_expected.to_string(),
      actual: describe(shape, input.data_type()),
    });
  }

  let out_expected = format!("[1, {EMBEDDING_DIM}] float32");
  let output = description
    .output(names::TEXT_FEATURES)
    .ok_or_else(|| Error::ContractMismatch {
      feature: names::TEXT_FEATURES,
      expected: out_expected.clone(),
      actual: "missing".to_string(),
    })?;
  if output.shape() != [1, EMBEDDING_DIM] || output.data_type() != Some(DataType::F32) {
    return Err(Error::ContractMismatch {
      feature: names::TEXT_FEATURES,
      expected: out_expected,
      actual: describe(output.shape(), output.data_type()),
    });
  }

  Ok(t)
}

/// Overrides the loaded tokenizer's normalization, truncation, and padding
/// policy to this module's fixed-window contract: a `Lowercase` normalizer
/// composed ahead of the loaded one, `LongestFirst` truncation at `max_tokens`,
/// stride 0, right direction (the export window is a hard model constraint), and
/// the tokenizer's own padding DISABLED — the module builds its own padded
/// window in [`build_window`] on the pinned side (D6), so an inherited padding
/// policy must not leak into the ids.
fn configure_tokenizer(tokenizer: &mut Tokenizer, max_tokens: usize) -> Result<()> {
  // SigLIP2 lowercases text before tokenization (checkpoint tokenizer_config
  // `do_lower_case: true`; transformers `Siglip2Tokenizer` composes
  // `normalizers.Lowercase()` ahead of the loaded tokenizer.json normalizer).
  // `Lowercase` here IS the same Rust implementation the Python reference calls.
  // Unlike upstream's defensive `is not None` guard, the composition applies
  // even when the loaded file carries no normalizer — the lowercase contract is
  // the module's, not the file's. Special/added tokens are matched before
  // normalization, so this cannot corrupt them.
  let lowercased: NormalizerWrapper = match tokenizer.get_normalizer() {
    Some(existing) => NormalizerSequence::new(vec![Lowercase.into(), existing.clone()]).into(),
    None => Lowercase.into(),
  };
  tokenizer
    .with_normalizer(Some(lowercased))
    .map_err(Error::TokenizerConfig)?;
  tokenizer
    .with_truncation(Some(TruncationParams {
      max_length: max_tokens,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .map_err(Error::TokenizerConfig)?;
  tokenizer.with_padding(None);
  Ok(())
}

/// Builds the fixed `[max_tokens]` padded `input_ids` window from the real token
/// `ids`: the real tokens occupy the prefix (`Right` pad) or suffix (`Left` pad),
/// and the remainder is filled with `pad_id`. Returns the full padded window (D6
/// — SigLIP attends and pools over pads, so they are part of the input).
///
/// [`configure_tokenizer`] forces truncation and disables the tokenizer's own
/// padding, so `ids` is already within the window; this still returns a typed
/// [`Error`] rather than panicking should that contract be violated.
///
/// # Errors
/// [`Error::TokenCount`] if `ids` exceeds `max_tokens`; [`Error::TokenIdRange`]
/// if a token id does not fit the model's `int32` `input_ids` tensor.
fn build_window(
  ids: &[u32],
  pad_id: i32,
  pad_side: PadSide,
  max_tokens: usize,
) -> Result<Vec<i32>> {
  if ids.len() > max_tokens {
    return Err(Error::TokenCount {
      got: ids.len(),
      max: max_tokens,
    });
  }
  let mut window = vec![pad_id; max_tokens];
  let offset = match pad_side {
    PadSide::Right => 0,
    PadSide::Left => max_tokens - ids.len(),
  };
  for (i, &id) in ids.iter().enumerate() {
    window[offset + i] = i32::try_from(id).map_err(|_| Error::TokenIdRange { id })?;
  }
  Ok(window)
}

/// Test-only seam: the module's actual tokenizer configuration, without loading
/// a CoreML model — so `tests` can exercise the real tokenization path
/// hermetically with a caller-supplied tokenizer and window `T`.
#[cfg(test)]
pub(crate) fn configured_tokenizer_from_bytes(
  bytes: &[u8],
  max_tokens: usize,
) -> Result<Tokenizer> {
  let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(Error::TokenizerLoad)?;
  configure_tokenizer(&mut tokenizer, max_tokens)?;
  Ok(tokenizer)
}

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

#[cfg(test)]
mod tests;
