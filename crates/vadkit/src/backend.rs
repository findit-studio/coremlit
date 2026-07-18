//! The CoreML implementation of silero's [`VadBackend`] seam, plus the
//! one-shot [`detect_speech`] entry point wired over it (design spec §2-§4).
//!
//! This is the whole of vadkit's "detector" surface: a [`CoreMlBackend`] that
//! turns one 256 ms (4096-sample) chunk of audio into one speech probability
//! by running the FluidInference unified Silero VAD graph through
//! [`crate::VadModel`], and a thin [`detect_speech`] that hands that backend to
//! silero's backend-agnostic [`silero::detect_speech_with`]. Every rule that
//! turns probabilities into segments — thresholding, the start/end hysteresis,
//! `min_speech`/`min_silence`, `speech_pad`, force-splitting — lives in the
//! published `silero` crate and stays there (spec §2-§3). vadkit authors NONE
//! of it; `tests/reexport.rs`'s `src_authors_no_detection_logic` grep gate
//! pins that.
//!
//! # Geometry and the seam contract
//!
//! [`CoreMlBackend`] declares [`frame_samples`](VadBackend::frame_samples) =
//! [`CHUNK_SAMPLES`] (4096) at 16 kHz — an 8× coarser frame than the ONNX
//! backend's 512, which silero's geometry-parameterized detector consumes
//! unchanged (spec §3). Its [`predict`](VadBackend::predict) advances the
//! model's recurrent [`VadState`](crate::VadState) in place, so successive
//! calls form one logical stream until [`reset`](VadBackend::reset); this is
//! exactly the streaming contract [`silero::detect_speech_with`] and
//! [`silero::SpeechSegmenter`] drive.
//!
//! # Error bridging
//!
//! The seam's associated error is bridged into [`silero::Error`] the way its
//! trait doc prescribes for an out-of-tree backend: [`CoreMlBackend::Error`] is
//! vadkit's own [`InferError`], and `impl From<InferError> for silero::Error`
//! wraps it in the transparent [`silero::Error::Backend`] variant, whose
//! `Display`/`source` delegate to the wrapped error.

use std::path::Path;

use silero::{SampleRate, SpeechOptions, SpeechSegment, VadBackend};

use crate::{
  error::{InferError, ModelError},
  model::{CHUNK_SAMPLES, VadModel, VadModelOptions},
};

/// A [`silero::VadBackend`] over the CoreML FluidInference unified Silero VAD
/// graph: one 256 ms chunk in, one speech probability out, recurrent state
/// carried across calls by the wrapped [`VadModel`].
///
/// Construct one and hand it to [`detect_speech`] (one-shot) or drive it
/// through [`silero::SpeechSegmenter`] / [`silero::detect_speech_with`]
/// directly for streaming — both re-exported at the crate root. Because the
/// backend owns recurrent state, a single value is a single logical stream:
/// call [`VadBackend::reset`] (or build a fresh backend) to start another.
#[derive(Debug)]
pub struct CoreMlBackend {
  model: VadModel,
}

impl CoreMlBackend {
  /// Loads the CoreML VAD model at `path` with the default compute units
  /// ([`VadModelOptions::new`]) and wraps it as a backend.
  ///
  /// # Errors
  /// As [`VadModel::load`] ([`ModelError::Load`] / [`ModelError::ContractMismatch`]).
  pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
    Ok(Self {
      model: VadModel::load(path)?,
    })
  }

  /// Loads the CoreML VAD model at `path` with custom [`VadModelOptions`]
  /// (e.g. [`coremlit::ComputeUnits::CpuOnly`] for deterministic runs) and
  /// wraps it as a backend.
  ///
  /// # Errors
  /// As [`VadModel::load_with`].
  pub fn load_with(path: impl AsRef<Path>, options: VadModelOptions) -> Result<Self, ModelError> {
    Ok(Self {
      model: VadModel::load_with(path, options)?,
    })
  }

  /// Wraps an already-loaded [`VadModel`] as a backend — the seam a caller that
  /// already holds a model (or shares one across detector and streaming uses)
  /// constructs through.
  #[inline(always)]
  pub const fn from_model(model: VadModel) -> Self {
    Self { model }
  }

  /// The wrapped model, for read access to its recurrent
  /// [`state`](VadModel::state) or a direct
  /// [`predict_chunk_with_state`](VadModel::predict_chunk_with_state) call.
  #[inline(always)]
  pub const fn model(&self) -> &VadModel {
    &self.model
  }

  /// Unwraps the backend back into its [`VadModel`].
  #[inline(always)]
  pub fn into_model(self) -> VadModel {
    self.model
  }
}

impl VadBackend for CoreMlBackend {
  /// vadkit's own inference error, bridged into [`silero::Error`] via
  /// [`silero::Error::Backend`] (see the `From` impl below).
  type Error = InferError;

  /// [`CHUNK_SAMPLES`] (4096) — 256 ms at 16 kHz, the unified artifact's new
  /// samples per chunk.
  #[inline(always)]
  fn frame_samples(&self) -> usize {
    CHUNK_SAMPLES
  }

  /// [`SampleRate::Rate16k`] — the only rate the unified artifact is trained
  /// for (design spec §4).
  #[inline(always)]
  fn sample_rate(&self) -> SampleRate {
    SampleRate::Rate16k
  }

  /// Runs one 4096-sample frame through the CoreML graph, advancing the model's
  /// recurrent [`VadState`](crate::VadState) in place and returning the speech
  /// probability in `[0, 1]`.
  ///
  /// Delegates to [`VadModel::predict_chunk`]; the detector always hands
  /// exactly [`frame_samples`](Self::frame_samples) samples (zero-padding the
  /// trailing partial frame itself), which the model consumes with no further
  /// padding.
  ///
  /// # Errors
  /// As [`VadModel::predict_chunk`] ([`InferError`]).
  fn predict(&mut self, frame: &[f32]) -> Result<f32, InferError> {
    self.model.predict_chunk(frame)
  }

  /// Clears the model's recurrent state back to
  /// [`VadState::initial`](crate::VadState::initial) — the next
  /// [`predict`](Self::predict) starts a fresh logical stream.
  #[inline(always)]
  fn reset(&mut self) {
    self.model.reset();
  }
}

/// Bridges vadkit's [`InferError`] into [`silero::Error`] through the
/// transparent [`silero::Error::Backend`] variant — the out-of-tree backend
/// pattern silero's [`VadBackend::Error`] doc prescribes. This is what lets
/// [`CoreMlBackend::Error`] satisfy the trait's `Into<silero::Error>` bound and
/// lets a backend failure surface from [`detect_speech`] /
/// [`silero::detect_speech_with`] as a single `silero::Error`.
impl From<InferError> for silero::Error {
  #[inline]
  fn from(error: InferError) -> Self {
    silero::Error::Backend(Box::new(error))
  }
}

/// One-shot offline speech detection over the CoreML backend: the CoreML
/// counterpart to `silero::detect_speech` (which runs the bundled ONNX
/// backend). Chunks `samples` into 4096-sample (256 ms) frames, runs `backend`
/// once per frame, and applies silero's segmentation rules — a pure forward to
/// [`silero::detect_speech_with`], authoring nothing.
///
/// `backend` is NOT reset first: pass a freshly built or
/// [`reset`](VadBackend::reset) backend to start a new stream (mirroring
/// `silero::detect_speech_with`'s own contract).
///
/// For streaming, drive a [`CoreMlBackend`] through the re-exported
/// [`silero::SpeechSegmenter`] (`push_probability` per
/// [`predict`](VadBackend::predict)) instead.
///
/// # Errors
/// Any frame's [`InferError`], bridged into [`silero::Error`] via
/// [`silero::Error::Backend`].
#[inline]
pub fn detect_speech(
  backend: &mut CoreMlBackend,
  samples: &[f32],
  options: SpeechOptions,
) -> silero::Result<Vec<SpeechSegment>> {
  silero::detect_speech_with(backend, samples, options)
}
