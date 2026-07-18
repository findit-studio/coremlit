//! The CLAP [`AudioEncoder`]: a Rust log-mel front-end (the private `mel`
//! submodule) around the fp16 CoreML HTSAT graph, with L2 normalization applied
//! in Rust.

mod mel;

use std::path::Path;

use coremlit::{ComputeUnits, DataType, Model, MultiArray};

use crate::{
  embedding::{EMBEDDING_DIM, Embedding},
  error::{Error, Result},
};

pub use self::mel::{N_MELS, T_FRAMES, TARGET_SAMPLES};

/// Declared feature names on `clap_audio.mlmodelc` (pinned by
/// `tests/model_io.rs`).
mod names {
  pub const INPUT_FEATURES: &str = "input_features";
  pub const AUDIO_EMBEDS: &str = "audio_embeds";
}

/// Default [`AudioEncoderOptions::compute`]. [`ComputeUnits::All`] lets CoreML
/// schedule across the available hardware.
///
/// As converted (T1), the HTSAT audio graph does **not** compile for the ANE and
/// CoreML falls back to GPU/CPU (fp16-clean there); the text graph does compile
/// for the ANE. `All` is the honest default — it never *asserts* ANE residency,
/// which `tests/placement.rs` characterizes rather than claims.
pub const DEFAULT_AUDIO_COMPUTE: ComputeUnits = ComputeUnits::All;

#[cfg(feature = "serde")]
fn default_audio_compute() -> ComputeUnits {
  DEFAULT_AUDIO_COMPUTE
}

/// Construction options for [`AudioEncoder`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioEncoderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_audio_compute", with = "crate::compute_units_serde")
  )]
  compute: ComputeUnits,
}

impl Default for AudioEncoderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl AudioEncoderOptions {
  /// Options matching the crate default: [`DEFAULT_AUDIO_COMPUTE`].
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_AUDIO_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the audio graph on.
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

/// CLAP audio encoder: 48 kHz mono `&[f32]` in, a unit-norm 512-d
/// [`Embedding`] out.
///
/// The front-end is a Rust log-mel port (the private `mel` submodule); the fp16
/// CoreML HTSAT graph maps the `[1, 1, 1001, 64]` spectrogram to a
/// pre-normalization 512-d projection, which this encoder L2-normalizes.
///
/// `&self` inference (no mutable scratch): the FFT plan and filterbank are built
/// once at load and per-call buffers are local, so fan-out means one
/// [`AudioEncoder`] per worker over a `Send` [`coremlit::Model`]
/// (`coremlit::Model` is deliberately `!Sync`).
#[derive(Debug)]
pub struct AudioEncoder {
  model: Model,
  mel: mel::MelExtractor,
}

impl AudioEncoder {
  /// Loads `clap_audio.mlmodelc` with [`AudioEncoderOptions::new`]
  /// ([`ComputeUnits::All`]).
  ///
  /// # Errors
  /// As [`Self::from_file_with`].
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
    Self::from_file_with(path, AudioEncoderOptions::new())
  }

  /// Loads the model with custom options, validating its I/O contract against
  /// the ground truth pinned by `tests/model_io.rs`.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model.
  /// [`Error::ContractMismatch`] if the loaded model's `input_features` input
  /// isn't `[1, 1, `[`T_FRAMES`]`, `[`N_MELS`]`]` f32 or its `audio_embeds`
  /// output isn't `[1, `[`EMBEDDING_DIM`]`]` f32.
  pub fn from_file_with(path: impl AsRef<Path>, options: AudioEncoderOptions) -> Result<Self> {
    let model = Model::load(path, options.compute())?;
    let description = model.description();

    let input_expected = format!("[1, 1, {T_FRAMES}, {N_MELS}] float32");
    let input =
      description
        .input(names::INPUT_FEATURES)
        .ok_or_else(|| Error::ContractMismatch {
          feature: names::INPUT_FEATURES,
          expected: input_expected.clone(),
          actual: "missing".to_string(),
        })?;
    if input.shape() != [1, 1, T_FRAMES, N_MELS] || input.data_type() != Some(DataType::F32) {
      return Err(Error::ContractMismatch {
        feature: names::INPUT_FEATURES,
        expected: input_expected,
        actual: describe(input.shape(), input.data_type()),
      });
    }

    let output_expected = format!("[1, {EMBEDDING_DIM}] float32");
    let output =
      description
        .output(names::AUDIO_EMBEDS)
        .ok_or_else(|| Error::ContractMismatch {
          feature: names::AUDIO_EMBEDS,
          expected: output_expected.clone(),
          actual: "missing".to_string(),
        })?;
    if output.shape() != [1, EMBEDDING_DIM] || output.data_type() != Some(DataType::F32) {
      return Err(Error::ContractMismatch {
        feature: names::AUDIO_EMBEDS,
        expected: output_expected,
        actual: describe(output.shape(), output.data_type()),
      });
    }

    Ok(Self {
      model,
      mel: mel::MelExtractor::new(),
    })
  }

  /// Embeds one audio window into a unit-norm [`Embedding`].
  ///
  /// `samples` is 48 kHz mono. Any non-empty length is accepted: the mel
  /// front-end `repeatpad`s (or head-truncates) it to the model's fixed
  /// [`TARGET_SAMPLES`] window exactly as HF's `ClapFeatureExtractor` does. The
  /// long-audio pipeline (a later task) produces properly-hopped 480 000-sample
  /// windows; this method is the per-window primitive.
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples` is empty.
  /// [`Error::NonFiniteInput`] if any sample is NaN/infinite (it would
  /// otherwise propagate through the mel into a garbage embedding).
  /// [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML failure.
  /// [`Error::OutputShape`] if the predicted `audio_embeds` shape diverges from
  /// `[1, `[`EMBEDDING_DIM`]`]`. [`Error::NonFiniteEmbedding`] /
  /// [`Error::EmbeddingZero`] if the projection cannot be normalized.
  pub fn embed_window(&self, samples: &[f32]) -> Result<Embedding> {
    if samples.is_empty() {
      return Err(Error::EmptyAudio);
    }
    if let Some(index) = first_non_finite(samples) {
      return Err(Error::NonFiniteInput { index });
    }

    let mut features = vec![0.0f32; N_MELS * T_FRAMES];
    self.mel.extract_into(samples, &mut features)?;

    // Time-major mel [1001, 64] maps directly onto the row-major
    // `input_features [1, 1, 1001, 64]` contract (T1).
    let input = MultiArray::from_slice(&[1, 1, T_FRAMES, N_MELS], &features)?;
    let mut outputs = self
      .model
      .predict_with(&[(names::INPUT_FEATURES, &input)])?;
    let embeds = outputs.take(names::AUDIO_EMBEDS).ok_or_else(|| {
      coremlit::PredictionError::MissingOutput {
        name: names::AUDIO_EMBEDS.to_string(),
      }
    })?;
    if embeds.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape {
        got: embeds.shape().to_vec(),
        expected: vec![1, EMBEDDING_DIM],
      });
    }

    let mut row = [0.0f32; EMBEDDING_DIM];
    embeds.copy_into::<f32>(&mut row)?;
    Embedding::from_slice_normalizing(&row)
  }
}

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

/// Flat index of the first non-finite (NaN/±∞) sample, if any.
fn first_non_finite(samples: &[f32]) -> Option<usize> {
  samples.iter().position(|v| !v.is_finite())
}

#[cfg(test)]
mod tests;
