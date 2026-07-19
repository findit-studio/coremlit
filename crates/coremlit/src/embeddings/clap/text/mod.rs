//! The CLAP [`TextEncoder`]: the pinned Xenova tokenizer around the fp16 CoreML
//! RoBERTa graph, with L2 normalization applied in Rust.

use std::path::Path;

use crate::{ComputeUnits, DataType, Model, MultiArray};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::embeddings::clap::{
  embedding::{EMBEDDING_DIM, Embedding, check_finite_output},
  error::{Error, Result},
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

/// Default [`TextEncoderOptions::compute`]. [`ComputeUnits::All`]. As converted
/// (T1), the RoBERTa text graph **does** compile for the ANE (unlike the audio
/// graph); placement is still characterized, not asserted (`tests/clap/placement.rs`).
pub const DEFAULT_TEXT_COMPUTE: ComputeUnits = ComputeUnits::All;

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
  model: Model,
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
  /// JSON is unreadable/invalid; [`Error::TokenizerConfig`] if truncation cannot
  /// be configured.
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
    configure_truncation(&mut tokenizer)?;
    let pad_id = tokenizer.token_to_id("<pad>").map_or(1, |id| id as i32);

    let model = Model::load(model_path, options.compute())?;
    let description = model.description();

    let ids_expected = format!("[1, {TEXT_MAX_TOKENS}] int32");
    for name in [names::INPUT_IDS, names::ATTENTION_MASK] {
      let input = description
        .input(name)
        .ok_or_else(|| Error::ContractMismatch {
          feature: name,
          expected: ids_expected.clone(),
          actual: "missing".to_string(),
        })?;
      if input.shape() != [1, TEXT_MAX_TOKENS] || input.data_type() != Some(DataType::I32) {
        return Err(Error::ContractMismatch {
          feature: name,
          expected: ids_expected.clone(),
          actual: describe(input.shape(), input.data_type()),
        });
      }
    }

    let output_expected = format!("[1, {EMBEDDING_DIM}] float32");
    let output = description
      .output(names::TEXT_EMBEDS)
      .ok_or_else(|| Error::ContractMismatch {
        feature: names::TEXT_EMBEDS,
        expected: output_expected.clone(),
        actual: "missing".to_string(),
      })?;
    if output.shape() != [1, EMBEDDING_DIM] || output.data_type() != Some(DataType::F32) {
      return Err(Error::ContractMismatch {
        feature: names::TEXT_EMBEDS,
        expected: output_expected,
        actual: describe(output.shape(), output.data_type()),
      });
    }

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
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure; [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML
  /// failure; [`Error::OutputShape`] if the predicted `text_embeds` shape
  /// diverges from `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] if the
  /// model output has a NaN/infinite component — model corruption, classified
  /// apart from a caller's own non-finite embedding data
  /// ([`Error::NonFiniteEmbedding`]); [`Error::EmbeddingZero`] if the (finite)
  /// projection has zero magnitude.
  pub fn embed(&self, text: &str) -> Result<Embedding> {
    let ids = self.token_ids(text)?;
    debug_assert!(
      ids.len() <= TEXT_MAX_TOKENS,
      "truncation caps ids at the window"
    );

    // Right-pad to the fixed [1, 512] window; mask real tokens 1, pads 0.
    let mut input_ids = [self.pad_id; TEXT_MAX_TOKENS];
    let mut attention_mask = [0i32; TEXT_MAX_TOKENS];
    for (i, &id) in ids.iter().enumerate() {
      input_ids[i] = id as i32;
      attention_mask[i] = 1;
    }

    let ids_tensor = MultiArray::from_slice(&[1, TEXT_MAX_TOKENS], &input_ids)?;
    let mask_tensor = MultiArray::from_slice(&[1, TEXT_MAX_TOKENS], &attention_mask)?;
    let mut outputs = self.model.predict_with(&[
      (names::INPUT_IDS, &ids_tensor),
      (names::ATTENTION_MASK, &mask_tensor),
    ])?;
    let embeds =
      outputs
        .take(names::TEXT_EMBEDS)
        .ok_or_else(|| crate::PredictionError::MissingOutput {
          name: names::TEXT_EMBEDS.to_string(),
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
}

/// Applies the fixed truncation config shared by every constructor — identical
/// to textclap's `force_max_length_truncation` (`textclap/src/text.rs`):
/// `LongestFirst` at [`TEXT_MAX_TOKENS`], stride 0, right direction. This is a
/// hard model constraint (the position table cannot be exceeded), not a knob, so
/// clapkit's token ids match textclap's on the identical tokenizer artifact.
fn configure_truncation(tokenizer: &mut Tokenizer) -> Result<()> {
  tokenizer
    .with_truncation(Some(TruncationParams {
      max_length: TEXT_MAX_TOKENS,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .map_err(Error::TokenizerConfig)?;
  Ok(())
}

/// Test-only seam: the crate's actual tokenizer configuration, without loading a
/// CoreML model — so `tests` can exercise the real tokenization path
/// hermetically (the tokenizer-identity gate).
#[cfg(test)]
pub(crate) fn configured_tokenizer_from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
  let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(Error::TokenizerLoad)?;
  configure_truncation(&mut tokenizer)?;
  Ok(tokenizer)
}

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

#[cfg(test)]
mod tests;
