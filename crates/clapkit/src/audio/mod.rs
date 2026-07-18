//! The CLAP [`AudioEncoder`]: a Rust log-mel front-end (the private `mel`
//! submodule) around the fp16 CoreML HTSAT graph, with L2 normalization applied
//! in Rust.

mod mel;

use std::path::Path;

use coremlit::{ComputeUnits, DataType, Model, MultiArray};

use crate::{
  embedding::{EMBEDDING_DIM, Embedding, check_finite_output},
  error::{Error, Result},
  window::{WindowEmbedding, WindowPlan},
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
  /// `samples` is 48 kHz mono and must be `1..=`[`TARGET_SAMPLES`] long. A
  /// shorter clip is `repeatpad`ed up to the fixed 480 000-sample window (exactly
  /// as HF's `ClapFeatureExtractor` does); a **longer** clip is rejected with
  /// [`Error::AudioTooLong`] rather than silently head-truncated. This is the
  /// per-window primitive — feed a longer clip to [`Self::embed_windows`] (the
  /// long-audio pipeline), which hops it into 480 000-sample windows first. (HF is
  /// configured for `rand_trunc`, so head-truncating here would be neither
  /// deterministic nor HF-faithful; clapkit will not truncate behind your back.)
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples` is empty.
  /// [`Error::AudioTooLong`] if `samples.len()` exceeds [`TARGET_SAMPLES`] (use
  /// [`Self::embed_windows`] for long audio).
  /// [`Error::NonFiniteInput`] if any sample is NaN/infinite (it would
  /// otherwise propagate through the mel into a garbage embedding).
  /// [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML failure.
  /// [`Error::OutputShape`] if the predicted `audio_embeds` shape diverges from
  /// `[1, `[`EMBEDDING_DIM`]`]`. [`Error::NonFiniteOutput`] if the model output
  /// has a NaN/infinite component — model corruption, classified apart from a
  /// caller's own non-finite embedding data ([`Error::NonFiniteEmbedding`]).
  /// [`Error::EmbeddingZero`] if the (finite) projection has zero magnitude.
  pub fn embed_window(&self, samples: &[f32]) -> Result<Embedding> {
    if samples.is_empty() {
      return Err(Error::EmptyAudio);
    }
    check_window_len(samples.len())?;
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
    // Classify a NaN/∞ the CoreML runtime produced as model-output corruption
    // (`NonFiniteOutput`) before it reaches `from_slice_normalizing`, which would
    // otherwise mislabel it as caller-supplied embedding data
    // (`NonFiniteEmbedding`).
    check_finite_output(&row)?;
    Embedding::from_slice_normalizing(&row)
  }

  /// Embeds a long clip as overlapped windows per `plan`, one
  /// [`WindowEmbedding`] (embedding + its span + coverage) per
  /// [`WindowSpan`](crate::window::WindowSpan).
  ///
  /// This is the long-audio pipeline entry: it slices `samples` at the plan's
  /// offsets and runs [`Self::embed_window`] on each (which `repeatpad`s a short
  /// tail to the fixed window). The per-window embeddings are RETURNED, not
  /// hidden inside aggregation, so a caller can aggregate them with an
  /// [`AggregatePolicy`](crate::aggregate::AggregatePolicy), score each window
  /// ([`crate::score::score_windows`]), or both.
  ///
  /// Runs sequentially: [`coremlit::Model`] is `!Sync`, so windows share one
  /// encoder on one thread (fan out with one [`AudioEncoder`] per worker for
  /// parallelism).
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples` is empty; otherwise any error
  /// [`Self::embed_window`] raises for a window.
  pub fn embed_windows(&self, samples: &[f32], plan: &WindowPlan) -> Result<Vec<WindowEmbedding>> {
    if samples.is_empty() {
      return Err(Error::EmptyAudio);
    }
    let spans = plan.spans(samples.len());
    let mut out = Vec::with_capacity(spans.len());
    for span in spans {
      let embedding = self.embed_window(&samples[span.start()..span.end()])?;
      out.push(WindowEmbedding::new(embedding, span));
    }
    Ok(out)
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

/// Reject a per-window sample count over [`TARGET_SAMPLES`]. The mel front-end
/// maps exactly one 480 000-sample window per inference, so an over-length clip
/// is a caller error ([`Error::AudioTooLong`]) — it must be hopped into windows
/// by [`AudioEncoder::embed_windows`], never silently head-truncated. Empty input
/// is rejected separately by the caller ([`Error::EmptyAudio`]).
fn check_window_len(len: usize) -> Result<()> {
  if len > TARGET_SAMPLES {
    return Err(Error::AudioTooLong {
      len,
      max: TARGET_SAMPLES,
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests;
