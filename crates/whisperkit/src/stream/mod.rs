//! Push-based streaming vocabulary (spec §5.3 `stream` row): session
//! state, options, and the early-stop gate a live transcription driver
//! needs.
//!
//! Swift's `AudioStreamTranscriber` (`AudioStreamTranscriber.swift:26-228`)
//! is an `actor` that owns microphone capture, a permission request
//! (`AudioProcessor.requestRecordPermission`), and a `Task.sleep`-driven
//! polling loop (`realtimeLoop`) that re-checks the input buffer every
//! 100 ms. None of that has a home at this crate's sans-I/O boundary
//! (`crate::audio`'s module doc): there is no microphone, no actor
//! isolation, and no async runtime here. This module ports the pure
//! vocabulary the actor drove instead: [`StreamState`] (Swift's
//! `AudioStreamTranscriber.State`), [`AudioStreamOptions`] (the actor's
//! constructor defaults), [`StreamUpdate`], [`should_stop_early`] (the
//! actor's static `shouldStopEarly`), and the energy-history bookkeeping
//! (`EnergyTracker`) behind `crate::audio::is_voice_detected`'s input.
//! Callers push audio samples and drive transitions themselves; the
//! orchestrating state machine that actually calls these pieces in a loop
//! (`AudioStreamTranscriber`'s Rust counterpart) is a later task — this
//! module compiles and is fully tested standing alone.

use crate::{
  audio::{relative_energy, signal_energy},
  options::DecodingOptions,
  result::{TranscriptionProgress, TranscriptionSegment},
  text::compression_ratio_of_tokens,
};

#[cfg(test)]
mod tests;

/// Samples per energy frame: 0.1 s at 16 kHz (Swift `AudioProcessor.
/// minBufferLength`, `AudioProcessor.swift:215`).
pub const ENERGY_FRAME_SAMPLES: usize = 1_600;

/// Trailing frame count the crate's `EnergyTracker` streaming helper mins
/// over for its relative-energy reference (Swift `AudioProcessor.
/// relativeEnergyWindow`,
/// `AudioProcessor.swift:209`).
pub const RELATIVE_ENERGY_WINDOW: usize = 20;

// ---------------------------------------------------------------------
// StreamState
// ---------------------------------------------------------------------

/// Live snapshot of a streaming transcription session — ports Swift
/// `AudioStreamTranscriber.State` (`AudioStreamTranscriber.swift:7-17`)
/// minus `isRecording`: microphone on/off is mic-lifecycle bookkeeping
/// with no meaning at this sans-I/O boundary (this module's doc), so it
/// is dropped rather than ported.
///
/// [`Self::new`] and the accessors are this type's only public surface;
/// every field mutates through a `pub(crate)` `set_*` family instead of a
/// public one — a later state machine (Plan 4 T8) owns every transition
/// Swift's actor applied via `didSet` (`AudioStreamTranscriber.swift:
/// 27-31`), so outside callers only ever read a session's state.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamState {
  current_fallbacks: usize,
  last_buffer_size: usize,
  last_confirmed_segment_end_seconds: f32,
  buffer_energy: Vec<f32>,
  current_text: String,
  confirmed_segments: Vec<TranscriptionSegment>,
  unconfirmed_segments: Vec<TranscriptionSegment>,
  unconfirmed_text: Vec<String>,
}

impl Default for StreamState {
  fn default() -> Self {
    Self::new()
  }
}

impl StreamState {
  /// A fresh session state: every count zero, every collection empty
  /// (Swift's all-default `State.init`, `AudioStreamTranscriber.swift:
  /// 8-16`).
  pub const fn new() -> Self {
    Self {
      current_fallbacks: 0,
      last_buffer_size: 0,
      last_confirmed_segment_end_seconds: 0.0,
      buffer_energy: Vec::new(),
      current_text: String::new(),
      confirmed_segments: Vec::new(),
      unconfirmed_segments: Vec::new(),
      unconfirmed_text: Vec::new(),
    }
  }

  // -- current_fallbacks ---------------------------------------------------
  /// Temperature-fallback count of the in-flight (or most recent) decode.
  #[inline(always)]
  pub const fn current_fallbacks(&self) -> usize {
    self.current_fallbacks
  }
  /// Sets [`Self::current_fallbacks`] in place. `pub(crate)`: see this
  /// struct's doc.
  // Only the state machine (Plan 4 T8, not yet built) calls the `set_*`
  // family; `dead_code` would otherwise flag every one of them in a
  // plain (non-test) build, since `stream::tests` is this crate's only
  // caller today (mirrors `tests/common/mod.rs`'s `tokenizer_dir`).
  #[allow(dead_code)]
  #[inline(always)]
  pub(crate) const fn set_current_fallbacks(&mut self, current_fallbacks: usize) -> &mut Self {
    self.current_fallbacks = current_fallbacks;
    self
  }

  // -- last_buffer_size ------------------------------------------------------
  /// Sample count of the buffer the last transcription pass ran over.
  #[inline(always)]
  pub const fn last_buffer_size(&self) -> usize {
    self.last_buffer_size
  }
  /// Sets [`Self::last_buffer_size`] in place. `pub(crate)`: see this
  /// struct's doc.
  #[allow(dead_code)] // see `set_current_fallbacks`'s comment
  #[inline(always)]
  pub(crate) const fn set_last_buffer_size(&mut self, last_buffer_size: usize) -> &mut Self {
    self.last_buffer_size = last_buffer_size;
    self
  }

  // -- last_confirmed_segment_end_seconds -------------------------------------
  /// End time, in seconds, of the last segment promoted to confirmed.
  #[inline(always)]
  pub const fn last_confirmed_segment_end_seconds(&self) -> f32 {
    self.last_confirmed_segment_end_seconds
  }
  /// Sets [`Self::last_confirmed_segment_end_seconds`] in place.
  /// `pub(crate)`: see this struct's doc.
  #[allow(dead_code)] // see `set_current_fallbacks`'s comment
  #[inline(always)]
  pub(crate) const fn set_last_confirmed_segment_end_seconds(
    &mut self,
    last_confirmed_segment_end_seconds: f32,
  ) -> &mut Self {
    self.last_confirmed_segment_end_seconds = last_confirmed_segment_end_seconds;
    self
  }

  // -- buffer_energy (Vec<f32>) ----------------------------------------------
  /// Relative energy per frame of the current buffer (Swift
  /// `AudioProcessor.relativeEnergy`, mirrored into state on every buffer
  /// callback, `AudioStreamTranscriber.swift:109-111`) — the input
  /// `crate::audio::is_voice_detected` reads.
  #[inline(always)]
  pub const fn buffer_energy_slice(&self) -> &[f32] {
    self.buffer_energy.as_slice()
  }
  /// Sets [`Self::buffer_energy_slice`] in place. `pub(crate)`: see this
  /// struct's doc.
  #[allow(dead_code)] // see `set_current_fallbacks`'s comment
  #[inline(always)]
  pub(crate) fn set_buffer_energy(&mut self, buffer_energy: impl Into<Vec<f32>>) -> &mut Self {
    self.buffer_energy = buffer_energy.into();
    self
  }

  // -- current_text ------------------------------------------------------------
  /// In-progress decoded text for the window currently being transcribed.
  #[inline(always)]
  pub fn current_text(&self) -> &str {
    self.current_text.as_str()
  }
  /// Sets [`Self::current_text`] in place. `pub(crate)`: see this struct's
  /// doc.
  #[allow(dead_code)] // see `set_current_fallbacks`'s comment
  #[inline(always)]
  pub(crate) fn set_current_text(&mut self, current_text: impl Into<String>) -> &mut Self {
    self.current_text = current_text.into();
    self
  }

  // -- confirmed_segments (Vec<TranscriptionSegment>) -------------------------
  /// Segments promoted to confirmed — stable output that will not be
  /// revised by a later re-transcription pass.
  #[inline(always)]
  pub const fn confirmed_segments_slice(&self) -> &[TranscriptionSegment] {
    self.confirmed_segments.as_slice()
  }
  /// Sets [`Self::confirmed_segments_slice`] in place. `pub(crate)`: see
  /// this struct's doc.
  #[allow(dead_code)] // see `set_current_fallbacks`'s comment
  #[inline(always)]
  pub(crate) fn set_confirmed_segments(
    &mut self,
    confirmed_segments: impl Into<Vec<TranscriptionSegment>>,
  ) -> &mut Self {
    self.confirmed_segments = confirmed_segments.into();
    self
  }

  // -- unconfirmed_segments (Vec<TranscriptionSegment>) ------------------------
  /// Segments from the latest pass still subject to revision by the next
  /// one.
  #[inline(always)]
  pub const fn unconfirmed_segments_slice(&self) -> &[TranscriptionSegment] {
    self.unconfirmed_segments.as_slice()
  }
  /// Sets [`Self::unconfirmed_segments_slice`] in place. `pub(crate)`: see
  /// this struct's doc.
  #[allow(dead_code)] // see `set_current_fallbacks`'s comment
  #[inline(always)]
  pub(crate) fn set_unconfirmed_segments(
    &mut self,
    unconfirmed_segments: impl Into<Vec<TranscriptionSegment>>,
  ) -> &mut Self {
    self.unconfirmed_segments = unconfirmed_segments.into();
    self
  }

  // -- unconfirmed_text (Vec<String>) -------------------------------------------
  /// Superseded `current_text` snapshots kept after a fallback retry
  /// shortened the decoded text (Swift `AudioStreamTranscriber.
  /// onProgressCallback`, `AudioStreamTranscriber.swift:113-124`).
  #[inline(always)]
  pub const fn unconfirmed_text_slice(&self) -> &[String] {
    self.unconfirmed_text.as_slice()
  }
  /// Sets [`Self::unconfirmed_text_slice`] in place. `pub(crate)`: see
  /// this struct's doc.
  #[allow(dead_code)] // see `set_current_fallbacks`'s comment
  #[inline(always)]
  pub(crate) fn set_unconfirmed_text(
    &mut self,
    unconfirmed_text: impl Into<Vec<String>>,
  ) -> &mut Self {
    self.unconfirmed_text = unconfirmed_text.into();
    self
  }
}

// ---------------------------------------------------------------------
// StateChangeCallback
// ---------------------------------------------------------------------

/// `(old, new)` state pair delivered once per state assignment — ports
/// Swift's `didSet` observer firing `stateChangeCallback?(oldValue,
/// state)` after every mutation (`AudioStreamTranscriber.swift:27-31`).
///
/// `Sync` (not `Send`) on the callback itself mirrors Swift's `@Sendable`
/// closure requirement (`AudioStreamTranscriberCallback`,
/// `AudioStreamTranscriber.swift:20-23`) via the same shape this crate's
/// decode-loop progress sink already uses
/// ([`TranscriptionProgressCallback`](crate::decode::TranscriptionProgressCallback)):
/// a `&F` reference is itself `Send` exactly when `F: Sync`, so a shared
/// reference to a `Sync` closure is the reference-based equivalent of a
/// thread-safe callback value.
pub type StateChangeCallback<'a> = &'a (dyn Fn(&StreamState, &StreamState) + Sync);

// ---------------------------------------------------------------------
// AudioStreamOptions
// ---------------------------------------------------------------------

/// Default [`AudioStreamOptions::required_segments_for_confirmation`]
/// (Swift `AudioStreamTranscriber.init`'s
/// `requiredSegmentsForConfirmation` parameter, `AudioStreamTranscriber.
/// swift:51`).
pub const DEFAULT_REQUIRED_SEGMENTS_FOR_CONFIRMATION: usize = 2;
/// Default [`AudioStreamOptions::silence_threshold`] (`:52`).
pub const DEFAULT_SILENCE_THRESHOLD: f32 = 0.3;
/// Default [`AudioStreamOptions::compression_check_window`] (`:53`).
pub const DEFAULT_COMPRESSION_CHECK_WINDOW: usize = 60;
/// Default [`AudioStreamOptions::use_vad`] (`:54`).
pub const DEFAULT_USE_VAD: bool = true;

/// Streaming-session knobs — ports the defaulted parameters of Swift's
/// `AudioStreamTranscriber.init` (`AudioStreamTranscriber.swift:43-74`;
/// its non-defaulted parameters are model/pipeline plumbing this port
/// threads separately, not option fields). `new()`/`Default` apply
/// Swift's defaults verbatim, matching [`DecodingOptions`]'s own
/// reference implementation of this crate's options pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioStreamOptions {
  required_segments_for_confirmation: usize,
  silence_threshold: f32,
  compression_check_window: usize,
  use_vad: bool,
}

impl Default for AudioStreamOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl AudioStreamOptions {
  /// Streaming options matching Swift's constructor defaults.
  pub const fn new() -> Self {
    Self {
      required_segments_for_confirmation: DEFAULT_REQUIRED_SEGMENTS_FOR_CONFIRMATION,
      silence_threshold: DEFAULT_SILENCE_THRESHOLD,
      compression_check_window: DEFAULT_COMPRESSION_CHECK_WINDOW,
      use_vad: DEFAULT_USE_VAD,
    }
  }

  // -- required_segments_for_confirmation -------------------------------------
  /// Segments kept behind the confirmation watermark before being
  /// promoted to [`StreamState::confirmed_segments_slice`].
  #[inline(always)]
  pub const fn required_segments_for_confirmation(&self) -> usize {
    self.required_segments_for_confirmation
  }
  /// Builder form of [`Self::set_required_segments_for_confirmation`].
  #[must_use]
  #[inline(always)]
  pub const fn with_required_segments_for_confirmation(
    mut self,
    required_segments_for_confirmation: usize,
  ) -> Self {
    self.set_required_segments_for_confirmation(required_segments_for_confirmation);
    self
  }
  /// Sets [`Self::required_segments_for_confirmation`] in place.
  #[inline(always)]
  pub const fn set_required_segments_for_confirmation(
    &mut self,
    required_segments_for_confirmation: usize,
  ) -> &mut Self {
    self.required_segments_for_confirmation = required_segments_for_confirmation;
    self
  }

  // -- silence_threshold ----------------------------------------------------
  /// Relative-energy threshold `crate::audio::is_voice_detected` gates on.
  #[inline(always)]
  pub const fn silence_threshold(&self) -> f32 {
    self.silence_threshold
  }
  /// Builder form of [`Self::set_silence_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_silence_threshold(mut self, silence_threshold: f32) -> Self {
    self.set_silence_threshold(silence_threshold);
    self
  }
  /// Sets [`Self::silence_threshold`] in place.
  #[inline(always)]
  pub const fn set_silence_threshold(&mut self, silence_threshold: f32) -> &mut Self {
    self.silence_threshold = silence_threshold;
    self
  }

  // -- compression_check_window ----------------------------------------------
  /// Trailing token-window width [`should_stop_early`] checks compression
  /// ratio over.
  #[inline(always)]
  pub const fn compression_check_window(&self) -> usize {
    self.compression_check_window
  }
  /// Builder form of [`Self::set_compression_check_window`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compression_check_window(mut self, compression_check_window: usize) -> Self {
    self.set_compression_check_window(compression_check_window);
    self
  }
  /// Sets [`Self::compression_check_window`] in place.
  #[inline(always)]
  pub const fn set_compression_check_window(
    &mut self,
    compression_check_window: usize,
  ) -> &mut Self {
    self.compression_check_window = compression_check_window;
    self
  }

  // -- use_vad (bool) ---------------------------------------------------------
  /// Gate transcription on `crate::audio::is_voice_detected` rather than
  /// running on every buffer.
  #[inline(always)]
  pub const fn use_vad(&self) -> bool {
    self.use_vad
  }
  /// Builder form of [`Self::set_use_vad`].
  #[must_use]
  #[inline(always)]
  pub const fn with_use_vad(mut self) -> Self {
    self.set_use_vad();
    self
  }
  /// Sets [`Self::use_vad`] to `true`.
  #[inline(always)]
  pub const fn set_use_vad(&mut self) -> &mut Self {
    self.use_vad = true;
    self
  }
  /// Builder form of [`Self::update_use_vad`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_use_vad(mut self, use_vad: bool) -> Self {
    self.update_use_vad(use_vad);
    self
  }
  /// Assigns [`Self::use_vad`] directly.
  #[inline(always)]
  pub const fn update_use_vad(&mut self, use_vad: bool) -> &mut Self {
    self.use_vad = use_vad;
    self
  }
  /// Sets [`Self::use_vad`] to `false`.
  #[inline(always)]
  pub const fn clear_use_vad(&mut self) -> &mut Self {
    self.use_vad = false;
    self
  }
}

// ---------------------------------------------------------------------
// StreamUpdate
// ---------------------------------------------------------------------

/// A streaming step's outcome — the vocabulary a later push-based driver
/// (Plan 4 T8) reports after each pushed buffer: not enough audio yet,
/// enough audio but no voice in it, or a transcription pass ran. Swift
/// has no equivalent type; `AudioStreamTranscriber.transcribeCurrentBuffer`
/// (`AudioStreamTranscriber.swift:126-193`) expresses the same three
/// outcomes as early `return`s rather than a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum StreamUpdate {
  /// Not enough new audio has arrived yet to attempt a transcription pass
  /// (Swift's `nextBufferSeconds > 1` guard, `AudioStreamTranscriber.
  /// swift:135`).
  AwaitingAudio,
  /// Enough audio arrived, but VAD found no voice in it (Swift's
  /// `voiceDetected` guard, `AudioStreamTranscriber.swift:142-157`).
  AwaitingVoice,
  /// A transcription pass ran over the buffer.
  Transcribed,
}

impl StreamUpdate {
  /// Stable snake_case name of the variant — the crate's own `as_str`
  /// convention (matching [`crate::options::Task`]/
  /// [`crate::model::ModelState`]).
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::AwaitingAudio => "awaiting_audio",
      Self::AwaitingVoice => "awaiting_voice",
      Self::Transcribed => "transcribed",
    }
  }
}

// ---------------------------------------------------------------------
// should_stop_early
// ---------------------------------------------------------------------

/// Whether a decode window should stop early — ports the static
/// `shouldStopEarly` (`AudioStreamTranscriber.swift:208-227`), meant to be
/// called from a
/// [`TranscriptionProgressCallback`](crate::decode::TranscriptionProgressCallback)
/// on every decode step: `Some(false)` there requests early stop, while
/// `Some(true)`/`None` continue — so `None` here means "keep decoding,"
/// not "checked and clean."
///
/// Two independent checks, either of which returns `Some(false)`:
/// 1. Once `progress.tokens_slice()` grows past `compression_check_window`,
///    the compression ratio of its **trailing** `compression_check_window`
///    tokens is compared against `options.compression_ratio_threshold()`.
///    **Faithful quirk** (Swift `?? 0.0`, `:217`): unlike every other
///    optional threshold in this crate, a disabled (`None`) threshold does
///    not skip this check — it compares against `0.0` instead, so any
///    token run long enough to reach the window (with the compression
///    ratio always positive for non-empty input) trips the stop.
/// 2. `progress.avg_logprob()` below `options.logprob_threshold()`, when
///    both are present.
pub fn should_stop_early(
  progress: &TranscriptionProgress,
  options: &DecodingOptions,
  compression_check_window: usize,
) -> Option<bool> {
  let tokens = progress.tokens_slice();
  if tokens.len() > compression_check_window {
    let window = &tokens[tokens.len() - compression_check_window..];
    let compression_ratio = compression_ratio_of_tokens(window);
    if compression_ratio > options.compression_ratio_threshold().unwrap_or(0.0) {
      return Some(false);
    }
  }
  if let Some(avg_logprob) = progress.avg_logprob()
    && let Some(threshold) = options.logprob_threshold()
    && avg_logprob < threshold
  {
    return Some(false);
  }
  None
}

// ---------------------------------------------------------------------
// EnergyTracker
// ---------------------------------------------------------------------

/// Per-frame relative-energy history for the streaming VAD gate.
///
/// Ports the energy bookkeeping `AudioProcessor.processBuffer` performs
/// incrementally as audio streams in (`AudioProcessor.swift:904-926`),
/// minus the mic-driven `audioBufferCallback`/logging side effects this
/// sans-I/O crate has no home for (this module's doc). Rather than mirror
/// Swift's one-call-per-buffer shape, [`Self::absorb`] takes the caller's
/// entire accumulated sample history each call and works out for itself
/// how many new [`ENERGY_FRAME_SAMPLES`]-sized frames have completed
/// since the last call — callers just keep passing a growing buffer.
///
/// `pub(crate)`, and every method below is `#[allow(dead_code)]`: the only
/// caller today is `stream::tests` (a plain, non-test build has none), and
/// the real caller — a later push-based driver, Plan 4 T8 — does not exist
/// yet. Not a bug; mirrors `tests/common/mod.rs`'s `tokenizer_dir`.
#[derive(Debug, Clone, Default, PartialEq)]
#[allow(dead_code)]
pub(crate) struct EnergyTracker {
  /// `(relative, average)` energy per completed frame, oldest first.
  frames: Vec<(f32, f32)>,
  /// How many leading samples of the caller's buffer have already been
  /// folded into `frames`.
  consumed_samples: usize,
}

impl EnergyTracker {
  /// Folds every newly completed [`ENERGY_FRAME_SAMPLES`] window in
  /// `buffer` into the tracked history. `buffer` is the caller's entire
  /// accumulated sample history (not just the newly arrived slice) —
  /// ports `AudioProcessor.audioSamples`'s append-only growth
  /// (`AudioProcessor.swift:908`). Frames already folded from an earlier
  /// call are skipped; a short trailing remainder waits for a future call
  /// to complete it.
  #[allow(dead_code)]
  fn absorb(&mut self, buffer: &[f32]) {
    while self.consumed_samples + ENERGY_FRAME_SAMPLES <= buffer.len() {
      let frame = &buffer[self.consumed_samples..self.consumed_samples + ENERGY_FRAME_SAMPLES];
      let avg = signal_energy(frame);
      let rel = if self.frames.is_empty() {
        // Swift's empty-history reference is `+∞` — the `Float.infinity`
        // fold seed reduced over zero prior entries
        // (`self.audioEnergy.suffix(20).reduce(Float.infinity) { min($0,
        // $1.avg) }`, `AudioProcessor.swift:911`) — which sends
        // `calculateRelativeEnergy`'s dB math (`AudioProcessor.swift:
        // 724-741`) through `(finite - ∞) / (0 - ∞)` = `-∞ / -∞` = `NaN`.
        // Swift's `Swift.min`/`Swift.max` (both defined via `<`/`>=`,
        // which are always `false` against `NaN`) clamp that `NaN` down
        // to `0.0` (`:740`, `max(0, min(normalizedEnergy, 1))`).
        //
        // Rust's equivalent, `relative_energy`'s `normalized.clamp(0.0,
        // 1.0)` (`audio/mod.rs`), does NOT reproduce that clamp-to-0 —
        // verified by tracing `f32::clamp`'s own implementation (`if self
        // < min { min } else if self > max { max } else { self }`), both
        // comparisons `false` against `NaN`, so it falls through and
        // returns `self` unchanged: `NaN`, not `0.0` (confirmed
        // empirically, not just by reading). A chained `.min(1.0).
        // max(0.0)` — the literal transliteration of Swift's `max(0,
        // min(x, 1))` nesting — would take a *third*, still-wrong path:
        // `f32::min`/`f32::max` ignore `NaN` and return the other
        // operand, so that chain silently produces `1.0` instead. Three
        // different Rust idioms, three different wrong answers if this
        // case is left to fall out of them — so it is special-cased
        // explicitly instead.
        0.0
      } else {
        let start = self.frames.len().saturating_sub(RELATIVE_ENERGY_WINDOW);
        let reference = self.frames[start..]
          .iter()
          .map(|&(_, avg)| avg)
          .fold(f32::INFINITY, f32::min);
        relative_energy(avg, reference)
      };
      self.frames.push((rel, avg));
      self.consumed_samples += ENERGY_FRAME_SAMPLES;
    }
  }

  /// Relative energy per completed frame, oldest first — the history
  /// `crate::audio::is_voice_detected` reads (Swift `AudioProcessor.
  /// relativeEnergy`, `AudioProcessor.swift:210-212`).
  #[allow(dead_code)]
  fn relative_energies(&self) -> Vec<f32> {
    self.frames.iter().map(|&(rel, _)| rel).collect()
  }
}
