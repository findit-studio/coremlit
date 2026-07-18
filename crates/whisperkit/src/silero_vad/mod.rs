//! Silero VAD as a whisperkit
//! [`VoiceActivityDetector`](crate::audio::vad::VoiceActivityDetector) — the
//! opt-in alternative to the default [`EnergyVad`](crate::audio::vad::EnergyVad),
//! behind the `vadkit` feature.
//!
//! Runs the FluidInference unified Silero VAD graph (via the sibling `vadkit`
//! crate's CoreML model layer) and reports one activity flag per 256 ms
//! (4096-sample) frame: the learned speech probability thresholded at
//! [`SileroVad::threshold`](crate::silero_vad::SileroVad::threshold). This is a
//! drop-in for whisperkit's frame-level
//! VAD seam — the same abstraction `EnergyVad` fills, learned instead of RMS —
//! whose flags whisperkit's own
//! [`active_chunks`](crate::audio::vad::VoiceActivityDetector::active_chunks)
//! then merge into the long-form [`ChunkingStrategy::Vad`] chunk boundaries. It
//! does NOT re-home silero's segment logic (that stays in `silero`, re-exported
//! by `vadkit`); it plugs the Silero *model* into whisperkit's existing VAD
//! contract.
//!
//! [`ChunkingStrategy::Vad`]: crate::options::ChunkingStrategy::Vad
//!
//! # Opt-in, energy VAD untouched
//!
//! The default pipeline still uses `EnergyVad` (Swift-parity). A caller opts in
//! by swapping the detector — the deviation-knob convention (like
//! `drop_blank_audio`), through the existing runtime seam:
//!
//! ```no_run
//! # use whisperkit::{options::Options, transcribe::WhisperKit, silero_vad::SileroVad};
//! # fn f(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
//! let kit = WhisperKit::new(options)?
//!   .with_vad_detector(Box::new(SileroVad::load("Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc")?));
//! # Ok(()) }
//! ```
//!
//! # Concurrency
//!
//! [`coremlit::Model`] is [`Send`] but deliberately not [`Sync`] (Apple's "one
//! `MLModel` on one thread at a time"), yet whisperkit stores the detector as
//! `Box<dyn VoiceActivityDetector + Send + Sync>`. So the model is held behind a
//! [`Mutex`](std::sync::Mutex), which supplies both the [`Sync`] the seam
//! requires and the exact serialization Apple's contract asks for —
//! [`coremlit::Model`]'s own doc names this ("serialized behind an external
//! `Mutex`") as the way to fan a model across threads. Each
//! [`voice_activity`](crate::audio::vad::VoiceActivityDetector::voice_activity)
//! call locks once and drives the whole buffer as one logical stream.

use std::sync::Mutex;

use vadkit::{CHUNK_SAMPLES, ModelError, VadModel, VadModelOptions, VadState};

use crate::audio::vad::VoiceActivityDetector;

/// Default speech threshold: a 256 ms frame is active when its Silero
/// probability is ≥ this. Matches silero's own `start_threshold` default (0.5).
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.5;

/// Silero-VAD [`VoiceActivityDetector`] over the `vadkit` CoreML model layer.
///
/// Follows the options pattern: [`Self::load`] is the default configuration
/// ([`DEFAULT_SPEECH_THRESHOLD`], the crate-default compute units). Holds the
/// model behind a [`Mutex`] so the detector is `Send + Sync`
/// (see the module doc's concurrency note).
#[derive(Debug)]
pub struct SileroVad {
  model: Mutex<VadModel>,
  threshold: f32,
}

impl SileroVad {
  /// Samples per analysis frame: [`vadkit::CHUNK_SAMPLES`] (4096 = 256 ms at
  /// 16 kHz), the unified artifact's chunk size. Exposed as an associated const
  /// so the geometry can be asserted without a loaded model;
  /// [`frame_length_samples`](crate::audio::vad::VoiceActivityDetector::frame_length_samples)
  /// returns it.
  pub const FRAME_LENGTH_SAMPLES: usize = CHUNK_SAMPLES;

  /// Loads the CoreML VAD model at `path` with the crate-default compute units
  /// and [`DEFAULT_SPEECH_THRESHOLD`].
  ///
  /// # Errors
  /// As [`vadkit::VadModel::load`] ([`ModelError`]).
  pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ModelError> {
    Ok(Self::from_model(VadModel::load(path)?))
  }

  /// Loads the CoreML VAD model at `path` with custom
  /// [`VadModelOptions`] (e.g.
  /// [`coremlit::ComputeUnits::CpuOnly`] for a deterministic run) and
  /// [`DEFAULT_SPEECH_THRESHOLD`].
  ///
  /// # Errors
  /// As [`vadkit::VadModel::load_with`] ([`ModelError`]).
  pub fn load_with(
    path: impl AsRef<std::path::Path>,
    options: VadModelOptions,
  ) -> Result<Self, ModelError> {
    Ok(Self::from_model(VadModel::load_with(path, options)?))
  }

  /// Wraps an already-loaded [`VadModel`] as a detector with
  /// the default threshold.
  pub fn from_model(model: VadModel) -> Self {
    Self {
      model: Mutex::new(model),
      threshold: DEFAULT_SPEECH_THRESHOLD,
    }
  }

  /// The speech threshold: a frame is active when its probability is ≥ this.
  #[inline(always)]
  pub const fn threshold(&self) -> f32 {
    self.threshold
  }

  /// Sets the speech threshold.
  #[inline(always)]
  pub const fn set_threshold(&mut self, threshold: f32) -> &mut Self {
    self.threshold = threshold;
    self
  }

  /// Consuming form of [`Self::set_threshold`].
  #[must_use]
  #[inline(always)]
  pub fn with_threshold(mut self, threshold: f32) -> Self {
    self.threshold = threshold;
    self
  }
}

impl VoiceActivityDetector for SileroVad {
  /// One activity flag per 256 ms frame: the Silero probability ≥
  /// [`Self::threshold()`]. The final partial frame is included (the model
  /// repeat-pads a short chunk, [`vadkit::VadModel::predict_chunk_with_state`]).
  ///
  /// The buffer is processed as one logical stream from a fresh recurrent
  /// state, so the result is independent of any previous call. A per-frame
  /// inference failure — a CoreML runtime error or the non-finite corruption
  /// `vadkit` exists to catch — is treated as **inactive** for that frame:
  /// [`voice_activity`](crate::audio::vad::VoiceActivityDetector::voice_activity)
  /// is infallible by contract (as is
  /// [`EnergyVad`](crate::audio::vad::EnergyVad), whose RMS on non-finite input
  /// likewise yields `false`), so a hard failure has no channel here. Callers
  /// that need inference errors surfaced should drive
  /// [`vadkit::detect_speech`] directly instead.
  fn voice_activity(&self, samples: &[f32]) -> Vec<bool> {
    let model = self.model.lock().expect("SileroVad model mutex poisoned");
    let mut state = VadState::initial();
    samples
      .chunks(CHUNK_SAMPLES)
      .map(
        |chunk| match model.predict_chunk_with_state(chunk, &state) {
          Ok((probability, next)) => {
            state = next;
            probability >= self.threshold
          }
          Err(_) => false,
        },
      )
      .collect()
  }

  /// [`Self::FRAME_LENGTH_SAMPLES`] (4096 = 256 ms at 16 kHz).
  #[inline(always)]
  fn frame_length_samples(&self) -> usize {
    Self::FRAME_LENGTH_SAMPLES
  }
}
