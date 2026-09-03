//! The siglip [`TextEmbedder`]: the artifact's Gemma tokenizer around the fp16
//! CoreML text graph, with L2 normalization applied in Rust.
//!
//! Text is lowercased before tokenization (the SigLIP2 training convention;
//! checkpoint `do_lower_case: true`), mirroring transformers `Siglip2Tokenizer`.
//!
//! Unlike `granite`/`clap`, the SigLIP text graph takes **only** `input_ids`
//! (`[1, T]` int32) — the processor emits no attention mask (canonical SigLIP
//! attends all `T` positions) and the tower pools the final position. That is a
//! clause of this door's load contract (`text_contract`), so a graph that
//! grew a mask is refused at load rather than asserted about by a model-gated
//! test. Because
//! every position is attended and the pooled token is positional, the pad id AND
//! pad side are semantically load-bearing (D6); the built window is compared
//! byte-for-byte against the committed goldens by the Wave B token-identity gate,
//! which pins them empirically.

use std::path::Path;

use crate::{
  ComputeUnits, DataType, Model, ModelDescription, MultiArray,
  model::contract::{Checked, Dim, FeatureContract, LoadContract, StateContract},
};
use tokenizers::{
  Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
  normalizers::{Lowercase, NormalizerWrapper, Sequence as NormalizerSequence},
};

use crate::embeddings::siglip::{
  embedding::{EMBEDDING_DIM, Embedding, check_finite_output},
  error::{
    ArtifactTokenizerIdentity, ArtifactTokenizerRead, ContractMismatch, Error, OutputShape, Result,
    TokenCount, contract_violation,
  },
};

/// Declared feature names on the siglip text `.mlmodelc` (pinned by
/// `tests/siglip/text_model_io.rs`). There is deliberately no `attention_mask`
/// — the graph has a single input.
mod names {
  pub const INPUT_IDS: &str = "input_ids";
  pub const TEXT_FEATURES: &str = "text_features";
}

/// The tokenizer identity contract. SigLIP 2 NaFlex is a FIXED model, so exactly
/// one tokenizer artifact is correct — the source-revision Gemma `tokenizer.json`
/// that cut the committed token-id goldens.
///
/// This pin is what keeps the artifact SIDECAR from being weaker than the bytes
/// the crate used to embed: a compiled-in asset was identity-by-construction, a
/// file on disk is not. [`TextEmbedder::load`] hashes what it read and refuses
/// anything else, fail-closed, before any model load.
mod contract {
  /// SHA-256 (lowercase hex) of `google/siglip2-base-patch16-naflex`'s
  /// `tokenizer.json` at revision `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` —
  /// the bytes `tests/siglip/tokenizer_identity.rs` proves reproduce every
  /// committed golden token window.
  pub const TOKENIZER_SHA256_HEX: &str =
    "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0";
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

/// The artifact-root path [`TextEmbedder::load`] reads its tokenizer from: the
/// directory CONTAINING `model_path`, joined with the sidecar file name.
///
/// The published bundle lays the artifact out with `tokenizer.json` beside the
/// `.mlmodelc` bundles and the pos-emb sidecar, all under the tier directory the
/// `CHECKSUMS.sha256` is rooted at. A `model_path` with no parent component
/// yields the bare file name, i.e. the current directory — the same place the
/// bundle itself would resolve from.
fn artifact_tokenizer_path(model_path: &Path) -> std::path::PathBuf {
  model_path
    .parent()
    .unwrap_or_else(|| Path::new(""))
    .join(crate::embeddings::siglip::TOKENIZER_FILE_NAME)
}

/// Fails closed unless `bytes` is byte-identical (SHA-256) to the pinned
/// source-revision Gemma `tokenizer.json`.
///
/// The placeholder scan above only catches the ONE stub this repo ever shipped;
/// it says nothing about a truncated download, a different checkpoint's
/// tokenizer, or a re-serialized copy whose vocab drifted. When the tokenizer was
/// compiled in, byte identity held by construction and a dev-time test was
/// enough. Read from a staged artifact directory, it has to be checked at load —
/// otherwise moving the file out of the crate would trade a guaranteed-correct
/// tokenizer for an unverified one, which is strictly worse.
///
/// # Errors
/// [`Error::ArtifactTokenizerIdentity`] if the digest differs from the pin.
fn ensure_pinned_identity(bytes: &[u8], path: &Path) -> Result<()> {
  use sha2::{Digest, Sha256};
  let actual: String = Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
  if actual == contract::TOKENIZER_SHA256_HEX {
    return Ok(());
  }
  Err(Error::ArtifactTokenizerIdentity(
    ArtifactTokenizerIdentity::new(path.to_path_buf(), contract::TOKENIZER_SHA256_HEX, actual),
  ))
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
/// tokenizes with the Gemma tokenizer (truncation `LongestFirst` at the
/// resolved window `T`, the tokenizer's own padding disabled), builds the fixed
/// `[1, T]` padded window (side/id per D6), runs the single-input fp16 CoreML
/// graph, and L2-normalizes the pre-normalization projection.
#[derive(Debug)]
pub struct TextEmbedder {
  /// A [`Checked`], never a bare [`Model`]: [`text_contract`] is the only
  /// contract this door states and [`Checked::new`] is the only way one is
  /// built, so removing the check from [`Self::from_parts`] does not compile.
  model: Checked,
  tokenizer: Tokenizer,
  /// Padding token id for the fixed-length window. SigLIP attends every position
  /// and pools the final one, so this is semantically load-bearing (D6);
  /// resolved from the tokenizer's `<pad>` at load, else `0`. Pinned by the
  /// Wave B token-identity goldens.
  pad_id: i32,
  /// Padding side for the fixed-length window (D6). Provisionally [`PadSide::Right`];
  /// pinned by the Wave B token-identity goldens.
  pad_side: PadSide,
  /// The text window length `T` READ BACK off the checked model's `input_ids
  /// [1, T]` contract (D2 — never a code constant). See [`text_contract`] for
  /// why the reading happens after the check.
  max_tokens: usize,
}

impl TextEmbedder {
  /// Loads the text `.mlmodelc` from `model_path` with the artifact's own
  /// [`TOKENIZER_FILE_NAME`](crate::embeddings::siglip::TOKENIZER_FILE_NAME)
  /// sidecar and custom `options` — the primary constructor.
  ///
  /// The model is checked against this door's load contract (`text_contract`)
  /// and held as a crate-internal `Checked` wrapper whose only constructor runs
  /// that check:
  ///
  /// ```text
  /// input   input_ids      i32  [1, T]    T AnyFixed, the batch Exactly
  /// output  text_features  f32  [1, 768]  every axis Exactly
  /// state   none
  /// ```
  ///
  /// `T` is the one number this door reads back rather than requires — the
  /// conversion pinned it, and [`Self::max_tokens`] is that value taken off the
  /// CHECKED model, which is what makes it the graph's only window rather than
  /// the DEFAULT shape of a flexible one this door would then pad every request
  /// to. It is also the tokenizer's truncation length, so the two cannot drift.
  ///
  /// The tokenizer is read from the model artifact's ROOT — the directory
  /// *containing* `model_path`, where the published bundle places
  /// `tokenizer.json` beside the `.mlmodelc` bundles. Both guards run on the
  /// bytes actually read, before any model load: the placeholder sentinel scan
  /// and the SHA-256 identity pin.
  ///
  /// # Errors
  /// [`Error::ArtifactTokenizerRead`] if the sidecar is missing or unreadable;
  /// [`Error::TokenizerPlaceholder`] if it is the build-time placeholder;
  /// [`Error::ArtifactTokenizerIdentity`] if it is not the pinned Gemma
  /// artifact; [`Error::ContractMismatch`] if a named feature's type or
  /// geometry is not the contract's or `input_ids` declares a zero window;
  /// [`Error::UnsatisfiableInput`] if the graph requires an input this door
  /// never sends — an `attention_mask` in particular;
  /// [`Error::UnsatisfiableState`] if it declares a state buffer; otherwise as
  /// [`Self::from_files`].
  pub fn load(model_path: impl AsRef<Path>, options: TextEmbedderOptions) -> Result<Self> {
    let model_path = model_path.as_ref();
    let tokenizer_path = artifact_tokenizer_path(model_path);
    let bytes = std::fs::read(&tokenizer_path).map_err(|source| {
      Error::ArtifactTokenizerRead(ArtifactTokenizerRead::new(tokenizer_path.clone(), source))
    })?;
    // The guard the embedded bytes used to carry, applied to the file that is
    // ACTUALLY loaded — a placeholder staged into an artifact tree must fail
    // exactly as a placeholder compiled into the crate did.
    ensure_not_placeholder(&bytes)?;
    ensure_pinned_identity(&bytes, &tokenizer_path)?;
    let tokenizer = Tokenizer::from_bytes(&bytes).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  /// Loads the text `.mlmodelc` from `model_path` using the artifact's
  /// [`TOKENIZER_FILE_NAME`](crate::embeddings::siglip::TOKENIZER_FILE_NAME)
  /// sidecar and [`TextEmbedderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, TextEmbedderOptions::new())
  }

  /// Loads the model and a `tokenizer.json` from separate file paths. The
  /// caller-supplied file is deliberately NOT placeholder-checked — a
  /// caller-chosen tokenizer is the caller's contract. [`Self::load`] guards the
  /// artifact sidecar it reads, and [`Self::from_memory`] guards the bytes handed
  /// to it; this constructor deliberately does neither.
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
  /// placeholder (fails closed before any I/O); otherwise as
  /// [`Self::from_files`].
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
    let model = Checked::new(model, &text_contract()).map_err(contract_violation)?;
    // Read BACK off the checked model: `input_ids`' second axis is
    // `Dim::AnyFixed`, so after the check the feature is `Fixed` and this
    // number is the graph's only window rather than the default shape of a
    // flexible one.
    let max_tokens = read_text_window(model.description())?;
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
    let feats = outputs
      .take(names::TEXT_FEATURES)
      .ok_or_else(|| crate::PredictionError::MissingOutput(names::TEXT_FEATURES.to_string()))?;
    if feats.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape(OutputShape::new(
        feats.shape().to_vec(),
        vec![1, EMBEDDING_DIM],
      )));
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

/// The load contract this door states: `input_ids` `[1, T]` i32 in,
/// `text_features` `[1, 768]` f32 out, no state.
///
/// # Which axis is READ and which is REQUIRED
///
/// `T` is the window the conversion pinned (`TEXT_WINDOW` in
/// `conversion/siglip/scripts/_siglip_common.py`), not a number this crate
/// chose, so `input_ids`' second axis is [`Dim::AnyFixed`]: the door asks only
/// that the graph pin exactly ONE size there, and [`TextEmbedder::max_tokens`]
/// is that size, read back off the CHECKED model — where the feature is known
/// [`crate::ShapeConstraint::Fixed`], so the number is the graph's only window
/// rather than the DEFAULT shape of a flexible one that this door would then
/// pad every request to. Nothing else in the contract depends on it, so unlike
/// `embeddings::siglip::image` this contract takes no parameter: one feature
/// carries the window and one feature reads it back.
///
/// **The input SET is a clause now, not a test.** This door's docs used to say
/// the "`input_ids` is the ONLY input — no `attention_mask`" assertion was
/// delegated to `tests/siglip/text_model_io.rs`, which is `#[ignore]`d without
/// a staged artifact, and none is staged. `check_load_contract` refuses any
/// REQUIRED input the contract does not name, so a graph that grew a mask —
/// which this door never supplies, and whose absence would fail every
/// prediction — is refused at load, hermetically.
fn text_contract() -> LoadContract {
  LoadContract::new(
    vec![FeatureContract::new(
      names::INPUT_IDS,
      DataType::I32,
      // Not `Exactly`: this door does not require a window, it reads back
      // whichever one the graph pins.
      vec![Dim::Exactly(1), Dim::AnyFixed],
    )],
    vec![FeatureContract::new(
      names::TEXT_FEATURES,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(EMBEDDING_DIM)],
    )],
    StateContract::None,
  )
}

/// The window `input_ids` pins, read back off a model [`Checked::new`] has
/// already accepted against [`text_contract`], and the one refusal that check
/// cannot make.
///
/// A declared window of ZERO is refused here rather than by the contract,
/// because the contract cannot express it: [`Dim::AnyFixed`] asks only that the
/// axis admit exactly one size, and zero is one size. A zero-token window is
/// one this door can build no tensor for.
///
/// # Errors
/// [`Error::ContractMismatch`] naming `input_ids` for a window of zero.
///
/// # Panics
/// Never, for a description [`Checked::new`] accepted against [`text_contract`]:
/// the check established that `input_ids` is declared and has exactly two axes.
fn read_text_window(description: &ModelDescription) -> Result<usize> {
  let window = description
    .input(names::INPUT_IDS)
    .and_then(|declared| declared.shape().get(1).copied())
    .expect("the load contract established `input_ids` and its rank");
  if window == 0 {
    return Err(Error::ContractMismatch(ContractMismatch::new(
      names::INPUT_IDS,
      "[1, T] int32 with T >= 1".to_string(),
      "[1, 0]".to_string(),
    )));
  }
  Ok(window)
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
    return Err(Error::TokenCount(TokenCount::new(ids.len(), max_tokens)));
  }
  let mut window = vec![pad_id; max_tokens];
  let offset = match pad_side {
    PadSide::Right => 0,
    PadSide::Left => max_tokens - ids.len(),
  };
  for (i, &id) in ids.iter().enumerate() {
    window[offset + i] = i32::try_from(id).map_err(|_| Error::TokenIdRange(id))?;
  }
  Ok(window)
}

/// Hermetic seam: the module's actual tokenizer configuration (the composed
/// `Lowercase` normalizer + `LongestFirst` truncation at `T` + disabled padding),
/// without loading a CoreML model — so the real tokenization path can be exercised
/// with a caller-supplied tokenizer and window `T`. Exposed (rather than
/// `#[cfg(test)]`) for the Wave B token-identity integration gate
/// (`tests/siglip/tokenizer_identity.rs`), which builds each golden window from
/// the artifact's staged `tokenizer.json` with no model load; hidden from docs
/// because it is a test seam, not part of the supported surface.
#[doc(hidden)]
pub fn configured_tokenizer_from_bytes(bytes: &[u8], max_tokens: usize) -> Result<Tokenizer> {
  let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(Error::TokenizerLoad)?;
  configure_tokenizer(&mut tokenizer, max_tokens)?;
  Ok(tokenizer)
}

#[cfg(test)]
mod tests;
