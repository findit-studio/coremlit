//! The CoreML implementation of zuoer's [`VadBackend`] seam, plus the
//! one-shot [`detect_speech`] entry point wired over it (design spec §2-§4).
//!
//! This is the whole of vadkit's "detector" surface: a [`CoreMlBackend`] that
//! turns 256 ms (4096-sample) chunks of audio into speech probabilities by
//! running the FluidInference unified Silero VAD graph through
//! [`crate::audio::vad::VadModel`], and a thin [`detect_speech`] that hands that backend to
//! zuoer's backend-agnostic [`zuoer::detect_speech_with`]. Every rule that
//! turns probabilities into segments — thresholding, the start/end hysteresis,
//! `min_speech`/`min_silence`, `speech_pad`, force-splitting — lives in the
//! published `zuoer` crate and stays there (spec §2-§3). vadkit authors NONE
//! of it; `tests/reexport.rs`'s `src_authors_no_detection_logic` grep gate
//! pins that.
//!
//! # Geometry and the seam contract
//!
//! [`CoreMlBackend`] declares [`frame_hop`](VadBackend::frame_hop) =
//! [`CHUNK_SAMPLES`] (4096) at 16 kHz — the model's analysis window equals its
//! hop, so the hop IS the frame size, an 8× coarser frame than the ONNX
//! backend's 512, which zuoer's geometry-parameterized detector consumes
//! unchanged (spec §3). [`push`](VadBackend::push) advances the model's
//! recurrent [`VadState`](crate::audio::vad::VadState) in place once per
//! completed frame, so successive calls form one logical stream until
//! [`reset`](VadBackend::reset); this is exactly the streaming contract
//! [`zuoer::detect_speech_with`] and [`zuoer::SpeechSegmenter`] drive.
//!
//! # Input windowing and the end-of-stream policy
//!
//! Under zuoer's push-based seam the BACKEND owns input windowing and the
//! trailing-frame policy, so [`CoreMlBackend`] buffers whatever PCM does not
//! yet complete a 4096-sample frame and carries it into the next
//! [`push`](VadBackend::push). At end of stream
//! [`finish`](VadBackend::finish) applies the Silero policy — zero-pad a
//! NON-EMPTY trailing partial frame and emit its probability; emit nothing
//! when the stream ended on an exact frame boundary. That reproduces, sample
//! for sample, what the pre-zuoer detector did on vadkit's behalf, so a clip
//! whose length is not a whole number of frames still contributes its final
//! partial frame and segments still close at the padded frame boundary.
//! `tests/vad/reexport.rs`'s
//! `coreml_backend_frames_match_a_hand_chunked_zero_padded_reference` is the
//! gate on that: it replays the pre-zuoer detector's chunking loop straight
//! through [`VadModel::predict_chunk`] and requires this backend's
//! `push` + `finish` probability stream to match it bit for bit. (The
//! end-to-end `detect_speech_on_real_audio_is_pinned` boundary pin does NOT
//! substitute for it — its deliberate ±1-frame fp16 band is exactly wide
//! enough to swallow a dropped tail.) The mock-backend `finish` scenarios in
//! the same file pin the policy hermetically.
//!
//! # Error bridging
//!
//! The seam's associated error is bridged into [`zuoer::Error`] the way its
//! trait doc prescribes for an out-of-tree backend: [`CoreMlBackend::Error`] is
//! vadkit's own [`InferError`], and `impl From<InferError> for zuoer::Error`
//! wraps it in the transparent [`zuoer::Error::Backend`] variant, whose
//! `Display`/`source` delegate to the wrapped error.

use std::path::Path;

use zuoer::{SampleRate, SpeechOptions, SpeechSegment, VadBackend};

use crate::audio::vad::{
  error::{InferError, ModelError},
  model::{CHUNK_SAMPLES, VadModel, VadModelOptions},
};

/// A [`zuoer::VadBackend`] over the CoreML FluidInference unified Silero VAD
/// graph: 256 ms chunks in, one speech probability per chunk out, recurrent
/// state carried across chunks by the wrapped [`VadModel`].
///
/// Construct one and hand it to [`detect_speech`] (one-shot) or drive it
/// through [`zuoer::SpeechSegmenter`] / [`zuoer::detect_speech_with`]
/// directly for streaming — both re-exported at the crate root. Because the
/// backend owns recurrent state AND the un-chunked PCM tail, a single value is
/// a single logical stream: call [`VadBackend::reset`] (or build a fresh
/// backend) to start another.
#[derive(Debug)]
pub struct CoreMlBackend {
  model: VadModel,
  /// PCM accepted by [`VadBackend::push`] that does not yet complete a
  /// [`CHUNK_SAMPLES`] frame. Always shorter than one frame between calls;
  /// consumed by the next `push` or zero-padded and flushed by
  /// [`VadBackend::finish`].
  pending: Vec<f32>,
}

impl CoreMlBackend {
  /// Loads the CoreML VAD model at `path` with the default compute units
  /// ([`VadModelOptions::new`]) and wraps it as a backend.
  ///
  /// # Errors
  /// As [`VadModel::load`] ([`ModelError::Load`] / [`ModelError::ContractMismatch`]).
  pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
    Ok(Self::from_model(VadModel::load(path)?))
  }

  /// Loads the CoreML VAD model at `path` with custom [`VadModelOptions`]
  /// (e.g. [`crate::ComputeUnits::CpuOnly`] for deterministic runs) and
  /// wraps it as a backend.
  ///
  /// # Errors
  /// As [`VadModel::load_with`].
  pub fn load_with(path: impl AsRef<Path>, options: VadModelOptions) -> Result<Self, ModelError> {
    Ok(Self::from_model(VadModel::load_with(path, options)?))
  }

  /// Wraps an already-loaded [`VadModel`] as a backend — the seam a caller that
  /// already holds a model (or shares one across detector and streaming uses)
  /// constructs through. The PCM buffer starts empty, i.e. at the start of a
  /// logical stream.
  #[inline(always)]
  pub const fn from_model(model: VadModel) -> Self {
    Self {
      model,
      pending: Vec::new(),
    }
  }

  /// The wrapped model, for read access to its recurrent
  /// [`state`](VadModel::state) or a direct
  /// [`predict_chunk_with_state`](VadModel::predict_chunk_with_state) call.
  #[inline(always)]
  pub const fn model(&self) -> &VadModel {
    &self.model
  }

  /// Unwraps the backend back into its [`VadModel`], DISCARDING any buffered
  /// partial-frame PCM (call [`VadBackend::finish`] first to flush it).
  #[inline(always)]
  pub fn into_model(self) -> VadModel {
    self.model
  }
}

impl VadBackend for CoreMlBackend {
  /// vadkit's own inference error, bridged into [`zuoer::Error`] via
  /// [`zuoer::Error::Backend`] (see the `From` impl below).
  type Error = InferError;

  /// [`CHUNK_SAMPLES`] (4096) — 256 ms at 16 kHz. The unified artifact's
  /// analysis window equals its hop, so its new-samples-per-chunk IS the hop
  /// the detector advances its timeline by.
  #[inline(always)]
  fn frame_hop(&self) -> usize {
    CHUNK_SAMPLES
  }

  /// [`SampleRate::Rate16k`] — the only rate the unified artifact is trained
  /// for (design spec §4).
  #[inline(always)]
  fn sample_rate(&self) -> SampleRate {
    SampleRate::Rate16k
  }

  /// Feeds `samples` into the stream, running the CoreML graph once per
  /// completed [`CHUNK_SAMPLES`] frame — advancing the model's recurrent
  /// [`VadState`](crate::audio::vad::VadState) in place — and handing each
  /// frame's probability in `[0, 1]` to `sink`.
  ///
  /// Whatever trailing PCM does not complete a frame is buffered and consumed
  /// by the next call, so a caller may push arbitrary-sized blocks. One call
  /// therefore invokes `sink` zero, one, or many times.
  ///
  /// # Errors
  /// As [`VadModel::predict_chunk`] ([`InferError`]). Probabilities emitted
  /// earlier in the same call have already reached `sink`.
  fn push(&mut self, samples: &[f32], sink: &mut dyn FnMut(f32)) -> Result<(), InferError> {
    let mut rest = samples;

    // Top up a partial frame left by an earlier call before touching the
    // caller's slice as frames, so frame boundaries stay tied to the stream,
    // not to how the caller happened to block its input.
    if !self.pending.is_empty() {
      let take = (CHUNK_SAMPLES - self.pending.len()).min(rest.len());
      self.pending.extend_from_slice(&rest[..take]);
      rest = &rest[take..];
      if self.pending.len() < CHUNK_SAMPLES {
        return Ok(());
      }
      let probability = self.model.predict_chunk(&self.pending)?;
      self.pending.clear();
      sink(probability);
    }

    // Whole frames run straight off the caller's slice — no copy.
    let (frames, remainder) = rest.as_chunks::<CHUNK_SAMPLES>();
    for frame in frames {
      let probability = self.model.predict_chunk(frame)?;
      sink(probability);
    }
    self.pending.extend_from_slice(remainder);
    Ok(())
  }

  /// Marks end-of-stream with the Silero trailing-frame policy: a NON-EMPTY
  /// buffered partial frame is zero-padded to [`CHUNK_SAMPLES`], run, and its
  /// probability handed to `sink`; a stream that ended on an exact frame
  /// boundary emits nothing.
  ///
  /// Call [`reset`](Self::reset) before reusing the backend for another
  /// stream.
  ///
  /// # Errors
  /// As [`VadModel::predict_chunk`] ([`InferError`]).
  fn finish(&mut self, sink: &mut dyn FnMut(f32)) -> Result<(), InferError> {
    if self.pending.is_empty() {
      return Ok(());
    }
    self.pending.resize(CHUNK_SAMPLES, 0.0);
    let probability = self.model.predict_chunk(&self.pending)?;
    self.pending.clear();
    sink(probability);
    Ok(())
  }

  /// Clears the model's recurrent state back to
  /// [`VadState::initial`](crate::audio::vad::VadState::initial) AND drops any
  /// buffered partial-frame PCM — the next [`push`](Self::push) starts a fresh
  /// logical stream.
  #[inline(always)]
  fn reset(&mut self) {
    self.model.reset();
    self.pending.clear();
  }
}

/// Bridges vadkit's [`InferError`] into [`zuoer::Error`] through the
/// transparent [`zuoer::Error::Backend`] variant — the out-of-tree backend
/// pattern zuoer's [`VadBackend::Error`] doc prescribes. This is what lets
/// [`CoreMlBackend::Error`] satisfy the trait's `Into<zuoer::Error>` bound and
/// lets a backend failure surface from [`detect_speech`] /
/// [`zuoer::detect_speech_with`] as a single `zuoer::Error`.
impl From<InferError> for zuoer::Error {
  #[inline]
  fn from(error: InferError) -> Self {
    zuoer::Error::Backend(Box::new(error))
  }
}

/// One-shot offline speech detection over the CoreML backend: the CoreML
/// counterpart to `silero::detect_speech` (which runs the bundled ONNX
/// backend). Pushes `samples` through the backend in 4096-sample (256 ms)
/// frames, flushes the zero-padded trailing partial frame, and applies zuoer's
/// segmentation rules — a pure forward to [`zuoer::detect_speech_with`],
/// authoring nothing.
///
/// `backend` is NOT reset first: pass a freshly built or
/// [`reset`](VadBackend::reset) backend to start a new stream (mirroring
/// `zuoer::detect_speech_with`'s own contract).
///
/// For streaming, drive a [`CoreMlBackend`] through the re-exported
/// [`zuoer::SpeechSegmenter`] (`push_probability` per probability
/// [`push`](VadBackend::push) emits) instead.
///
/// # Errors
/// Any frame's [`InferError`], bridged into [`zuoer::Error`] via
/// [`zuoer::Error::Backend`].
#[inline]
pub fn detect_speech(
  backend: &mut CoreMlBackend,
  samples: &[f32],
  options: SpeechOptions,
) -> zuoer::Result<Vec<SpeechSegment>> {
  zuoer::detect_speech_with(backend, samples, options)
}
