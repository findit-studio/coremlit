//! Silero VAD as a whisperkit
//! [`VoiceActivityDetector`](crate::audio::whisper::audio::vad::VoiceActivityDetector) — the
//! opt-in alternative to the default [`EnergyVad`](crate::audio::whisper::audio::vad::EnergyVad),
//! behind the `vadkit` feature.
//!
//! Runs the FluidInference unified Silero VAD graph (via the sibling `vadkit`
//! crate's CoreML model layer) and reports one activity flag per 256 ms
//! (4096-sample) frame: the learned speech probability thresholded at
//! [`SileroVad::threshold`](crate::audio::whisper::silero_vad::SileroVad::threshold). This is a
//! drop-in for whisperkit's frame-level
//! VAD seam — the same abstraction `EnergyVad` fills, learned instead of RMS —
//! whose flags whisperkit's own
//! [`active_chunks`](crate::audio::whisper::audio::vad::VoiceActivityDetector::active_chunks)
//! then merge into the long-form [`ChunkingStrategy::Vad`] chunk boundaries. It
//! does NOT re-home silero's segment logic (that stays in `silero`, re-exported
//! by `vadkit`); it plugs the Silero *model* into whisperkit's existing VAD
//! contract.
//!
//! [`ChunkingStrategy::Vad`]: crate::audio::whisper::options::ChunkingStrategy::Vad
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
//! [`crate::Model`] is [`Send`] but deliberately not [`Sync`] (Apple's "one
//! `MLModel` on one thread at a time"), yet whisperkit stores the detector as
//! `Box<dyn VoiceActivityDetector + Send + Sync>`. So the model is held behind a
//! [`Mutex`](std::sync::Mutex), which supplies both the [`Sync`] the seam
//! requires and the exact serialization Apple's contract asks for —
//! [`crate::Model`]'s own doc names this ("serialized behind an external
//! `Mutex`") as the way to fan a model across threads. Each
//! [`voice_activity`](crate::audio::whisper::audio::vad::VoiceActivityDetector::voice_activity)
//! call locks once and drives the whole buffer as one logical stream.
//!
//! A shared detector may also be driven by *simultaneous* `transcribe` calls.
//! The hard-failure latch is built for that: a monotonic generation counter
//! (not a drainable error slot) is each caller's source of truth, so
//! concurrent runs can't clear one another's failures — see
//! [`detection_generation`](crate::audio::whisper::audio::vad::VoiceActivityDetector::detection_generation).

use std::sync::Mutex;

use crate::audio::vad::{
  CHUNK_SAMPLES, InferError, ModelError, VadModel, VadModelOptions, VadState,
};

use crate::audio::whisper::audio::vad::VoiceActivityDetector;

/// Default speech threshold: a 256 ms frame is active when its Silero
/// probability is ≥ this. Matches silero's own `start_threshold` default (0.5).
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.5;

/// The hard-failure latch behind [`SileroVad`]'s infallible
/// [`voice_activity`](VoiceActivityDetector::voice_activity): a monotonic
/// `generation` bumped once per latched inference failure, plus the most
/// recent such error for reporting. One [`Mutex`] guards both so a reader
/// sees them consistently. See the module's concurrency note.
#[derive(Debug, Default)]
struct DetectionLatch {
  generation: u64,
  last_error: Option<InferError>,
}

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
  // Hard-failure latch (interior-mutable: the trait takes `&self`). A
  // per-frame model failure is infallible at the trait boundary, so rather
  // than swallow it into false silence `voice_activity` records it here —
  // bumping `generation` and storing the error. `detection_generation` /
  // `last_detection_error` expose it non-destructively, so concurrent
  // `transcribe` calls can't clear each other's failures (module doc).
  latch: Mutex<DetectionLatch>,
}

impl SileroVad {
  /// Samples per analysis frame: [`crate::audio::vad::CHUNK_SAMPLES`] (4096 = 256 ms at
  /// 16 kHz), the unified artifact's chunk size. Exposed as an associated const
  /// so the geometry can be asserted without a loaded model;
  /// [`frame_length_samples`](crate::audio::whisper::audio::vad::VoiceActivityDetector::frame_length_samples)
  /// returns it.
  pub const FRAME_LENGTH_SAMPLES: usize = CHUNK_SAMPLES;

  /// Loads the CoreML VAD model at `path` with the crate-default compute units
  /// and [`DEFAULT_SPEECH_THRESHOLD`].
  ///
  /// # Errors
  /// As [`crate::audio::vad::VadModel::load`] ([`ModelError`]).
  pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ModelError> {
    Ok(Self::from_model(VadModel::load(path)?))
  }

  /// Loads the CoreML VAD model at `path` with custom
  /// [`VadModelOptions`] (e.g.
  /// [`crate::ComputeUnits::CpuOnly`] for a deterministic run) and
  /// [`DEFAULT_SPEECH_THRESHOLD`].
  ///
  /// # Errors
  /// As [`crate::audio::vad::VadModel::load_with`] ([`ModelError`]).
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
      latch: Mutex::new(DetectionLatch::default()),
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
  /// repeat-pads a short chunk, [`crate::audio::vad::VadModel::predict_chunk_with_state`]).
  ///
  /// The buffer is processed as one logical stream from a fresh recurrent
  /// state, so the result is independent of any previous call. A per-frame
  /// inference failure — a CoreML runtime error or the non-finite corruption
  /// `vadkit` exists to catch — is reported as **inactive** for that frame
  /// (the trait's
  /// [`voice_activity`](crate::audio::whisper::audio::vad::VoiceActivityDetector::voice_activity)
  /// is infallible, returning `Vec<bool>`), but unlike
  /// [`EnergyVad`](crate::audio::whisper::audio::vad::EnergyVad) — whose RMS is a total
  /// function of any finite input — a hard *model* failure here is not a
  /// benign "no speech": it is **latched** instead of swallowed, bumping a
  /// monotonic
  /// [`detection_generation`](crate::audio::whisper::audio::vad::VoiceActivityDetector::detection_generation)
  /// and recording the error for
  /// [`last_detection_error`](crate::audio::whisper::audio::vad::VoiceActivityDetector::last_detection_error).
  /// `WhisperKit::transcribe` snapshots that generation across chunking and
  /// surfaces a [`crate::audio::whisper::error::VadError`], so a hard failure during VAD is
  /// observable from the same transcription call rather than silently
  /// degrading the chunk boundaries.
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
          Err(error) => {
            // Infallible per the trait, so this frame is reported inactive —
            // but the hard failure is LATCHED rather than lost: bump the
            // monotonic generation and record the error (most recent wins).
            // `transcribe` detects the generation advance and surfaces it; a
            // concurrent run reading the latch can't clear this one (module
            // doc). `state` is not advanced; later frames continue from the
            // last good state.
            let mut latch = self
              .latch
              .lock()
              .expect("SileroVad detection latch poisoned");
            latch.generation = latch.generation.wrapping_add(1);
            latch.last_error = Some(error);
            false
          }
        },
      )
      .collect()
  }

  /// The monotonic count of hard inference failures [`Self::voice_activity`]
  /// has latched (see its note). `transcribe` snapshots this before chunking
  /// and compares afterward; any advance is a failure it must surface. Never
  /// reset, so a concurrent run can't hide this one.
  fn detection_generation(&self) -> u64 {
    self
      .latch
      .lock()
      .expect("SileroVad detection latch poisoned")
      .generation
  }

  /// The most recent latched inference failure, cloned and boxed as the
  /// trait's erased error, without clearing it — non-destructive, so
  /// simultaneous callers each observing a generation advance can all read
  /// it. `None` when nothing has been latched.
  fn last_detection_error(&self) -> Option<Box<dyn std::error::Error + Send + Sync + 'static>> {
    self
      .latch
      .lock()
      .expect("SileroVad detection latch poisoned")
      .last_error
      .clone()
      .map(|error| Box::new(error) as Box<dyn std::error::Error + Send + Sync + 'static>)
  }

  /// [`Self::FRAME_LENGTH_SAMPLES`] (4096 = 256 ms at 16 kHz).
  #[inline(always)]
  fn frame_length_samples(&self) -> usize {
    Self::FRAME_LENGTH_SAMPLES
  }
}
