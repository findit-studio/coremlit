//! Push-based streaming (spec §5.3 `stream` row): [`AudioStreamTranscriber`],
//! the state machine that drives it, and the supporting vocabulary
//! ([`StreamState`], [`AudioStreamOptions`], [`StreamUpdate`],
//! [`should_stop_early`], `EnergyTracker`) it and its tests are built
//! from.
//!
//! Swift's `AudioStreamTranscriber` (`AudioStreamTranscriber.swift:26-228`)
//! is an `actor` that owns microphone capture, a permission request
//! (`AudioProcessor.requestRecordPermission`), and a `Task.sleep`-driven
//! polling loop (`realtimeLoop`) that re-checks the input buffer every
//! 100 ms. None of that has a home at this crate's sans-I/O boundary
//! (`crate::audio`'s module doc): there is no microphone, no actor
//! isolation, and no async runtime here. This module ports the pure state
//! machine the actor drove instead: [`AudioStreamTranscriber::push_samples`]
//! is the caller-driven port of `transcribeCurrentBuffer`
//! (`AudioStreamTranscriber.swift:126-193`) — a caller pushes samples as
//! they arrive and drives the loop itself; a non-[`StreamUpdate::Transcribed`]
//! return means "not enough yet, call again with more," replacing Swift's
//! sleep-and-retry `realtimeLoop`.
//!
//! **Documented deviations**, beyond the mic lifecycle above:
//! [`AudioStreamTranscriber::push_samples`] constructs a fresh
//! [`crate::transcribe::TranscribeTask`] every call, so
//! [`crate::result::TranscriptionTimings::total_decoding_fallbacks`]
//! counts fallbacks within that one run rather than accumulating across
//! the whole stream the way Swift's single reused `transcribeTask` would
//! (`AudioStreamTranscriber.swift:57-66`) — the fallback comparison
//! `push_samples`' progress callback makes is itself within-run only
//! (it compares against the immediately preceding progress update), so
//! this is unaffected. Errors propagate as a `Result` out of
//! `push_samples` instead of `realtimeLoop`'s log-and-break (`:98-107`).
//! Every dropped `Logging.*` call along the way (`:150`'s VAD-skip debug
//! log, `:119`'s fallback info log) follows this crate's established
//! precedent of not wiring Swift's instrumentation calls to `crate::log`
//! (see `crate::transcribe`'s module doc, "Not ported").

use std::sync::Mutex;

use crate::{
  audio::{is_voice_detected, relative_energy, signal_energy},
  backend::InferenceBackend,
  constants::SAMPLE_RATE,
  error::TranscribeError,
  options::DecodingOptions,
  result::{TranscriptionProgress, TranscriptionResult, TranscriptionSegment},
  text::compression_ratio_of_tokens,
  tokenizer::WhisperTokenizer,
  transcribe::TranscribeTask,
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
  /// Sets [`Self::confirmed_segments_slice`] in place, replacing the
  /// whole collection. `pub(crate)`: see this struct's doc.
  ///
  /// Unlike its `set_*` siblings, [`AudioStreamTranscriber::push_samples`]
  /// (Plan 4 T8) never calls this one: it only ever *grows* confirmed
  /// segments, via [`Self::confirmed_segments_mut`]
  /// (`AudioStreamTranscriber.swift:183`'s `append(contentsOf:)`, never a
  /// wholesale replace), so this whole-collection setter stays exercised
  /// only by this module's own tests in a plain (non-test) build — not a
  /// bug, mirrors `tests/common/mod.rs`'s `tokenizer_dir`.
  #[allow(dead_code)]
  #[inline(always)]
  pub(crate) fn set_confirmed_segments(
    &mut self,
    confirmed_segments: impl Into<Vec<TranscriptionSegment>>,
  ) -> &mut Self {
    self.confirmed_segments = confirmed_segments.into();
    self
  }
  /// Mutable access to the raw confirmed-segments vector, so a caller can
  /// grow it in place (e.g. `extend_from_slice`) instead of reading,
  /// cloning, and rebuilding the whole collection through
  /// [`Self::set_confirmed_segments`] on every promotion — ports Swift's
  /// `state.confirmedSegments.append(contentsOf:)`
  /// (`AudioStreamTranscriber.swift:183`). `pub(crate)`: see this
  /// struct's doc.
  #[inline(always)]
  pub(crate) const fn confirmed_segments_mut(&mut self) -> &mut Vec<TranscriptionSegment> {
    &mut self.confirmed_segments
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
  /// Sets [`Self::unconfirmed_text_slice`] in place, replacing the whole
  /// collection. `pub(crate)`: see this struct's doc.
  #[inline(always)]
  pub(crate) fn set_unconfirmed_text(
    &mut self,
    unconfirmed_text: impl Into<Vec<String>>,
  ) -> &mut Self {
    self.unconfirmed_text = unconfirmed_text.into();
    self
  }
  /// Mutable access to the raw unconfirmed-text vector, so a caller can
  /// `push` a superseded snapshot in place instead of reading, cloning,
  /// and rebuilding the whole collection through
  /// [`Self::set_unconfirmed_text`] on every fallback restart — ports
  /// Swift's `state.unconfirmedText.append(_:)`
  /// (`AudioStreamTranscriber.swift:117`). `pub(crate)`: see this
  /// struct's doc.
  #[inline(always)]
  pub(crate) const fn unconfirmed_text_mut(&mut self) -> &mut Vec<String> {
    &mut self.unconfirmed_text
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioStreamOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_required_segments_for_confirmation")
  )]
  required_segments_for_confirmation: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_silence_threshold"))]
  silence_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_compression_check_window"))]
  compression_check_window: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_use_vad"))]
  use_vad: bool,
}

#[cfg(feature = "serde")]
fn default_required_segments_for_confirmation() -> usize {
  DEFAULT_REQUIRED_SEGMENTS_FOR_CONFIRMATION
}
#[cfg(feature = "serde")]
fn default_silence_threshold() -> f32 {
  DEFAULT_SILENCE_THRESHOLD
}
#[cfg(feature = "serde")]
fn default_compression_check_window() -> usize {
  DEFAULT_COMPRESSION_CHECK_WINDOW
}
#[cfg(feature = "serde")]
fn default_use_vad() -> bool {
  DEFAULT_USE_VAD
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
/// `pub(crate)`: [`AudioStreamTranscriber::push_samples`] (Plan 4 T8) is
/// this crate's real caller; `stream::tests` exercises it directly too,
/// ahead of and alongside that consumer.
#[derive(Debug, Clone, Default, PartialEq)]
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
  fn relative_energies(&self) -> Vec<f32> {
    self.frames.iter().map(|&(rel, _)| rel).collect()
  }
}

// ---------------------------------------------------------------------
// AudioStreamTranscriber
// ---------------------------------------------------------------------

/// Placeholder [`StreamState::current_text`] shown while waiting for
/// enough new audio, or for voice within it — Swift's literal, verified
/// byte-for-byte against both of its call sites
/// (`AudioStreamTranscriber.swift:137`, `:152`; identical at both).
const WAITING_FOR_SPEECH_TEXT: &str = "Waiting for speech...";

/// Clones `state`'s current value, applies `mutate`, then — if `callback`
/// is set — fires it with `(old, new)`. The single routing point every
/// individual [`StreamState`] mutation in this module goes through,
/// giving each one Swift's per-assignment `didSet` parity
/// (`AudioStreamTranscriber.swift:27-31`: assigning any one field of
/// Swift's `state` re-triggers `didSet` on the whole property, by value
/// semantics), whether the mutation happens through `&mut self.state`
/// directly or through the `Mutex<StreamState>`
/// [`AudioStreamTranscriber::transcribe_audio_samples`] locks for the
/// duration of a run. Fires unconditionally after `mutate` runs, even if
/// `mutate` leaves `state` unchanged — Swift's `didSet` does too, since it
/// observes the assignment, not a value comparison.
fn apply(
  state: &mut StreamState,
  callback: Option<StateChangeCallback<'_>>,
  mutate: impl FnOnce(&mut StreamState),
) {
  let old = state.clone();
  mutate(state);
  if let Some(callback) = callback {
    callback(&old, state);
  }
}

/// Whether `needle` appears in `haystack` as a contiguous run — ports
/// Swift's SE-0357 `Array.contains(_:)` subsequence check
/// (`AudioStreamTranscriber.swift:182`,
/// `if !state.confirmedSegments.contains(confirmedSegmentsArray)`).
///
/// `needle` is always non-empty at this module's one call site (guarded
/// by `segments.len() > required`, so the newly confirmed run has at
/// least one segment); the `!needle.is_empty()` guard here exists so this
/// helper never calls [`slice::windows`] with a size of `0`, which panics,
/// rather than because an empty needle's result is otherwise ambiguous.
fn contains_subsequence(
  haystack: &[TranscriptionSegment],
  needle: &[TranscriptionSegment],
) -> bool {
  !needle.is_empty()
    && haystack
      .windows(needle.len())
      .any(|window| window == needle)
}

/// Ports `onProgressCallback` (`AudioStreamTranscriber.swift:113-124`):
/// records this decode step's fallback count and, when this step's text
/// is SHORTER than the previously recorded [`StreamState::current_text`]
/// (a fallback restart discarding a partial decode, not the normal
/// monotonic growth of a completing one) at the SAME fallback count as
/// before (an in-progress retry, not one that just occurred), stashes the
/// superseded text onto [`StreamState::unconfirmed_text_slice`] before
/// overwriting it. Swift's parallel `else` branch
/// (`Logging.info("Fallback occured: \(fallbacks)")`, `:119`) is a
/// dropped `Logging.*` call (this module's doc) with no state effect, so
/// it collapses into this function's single combined condition rather
/// than a nested `if`/dead `else`.
///
/// Swift compares `String.count` (grapheme clusters); [`str::chars`]
/// (Unicode scalars) is the documented close-enough port (this crate has
/// no grapheme-cluster segmentation dependency).
fn on_progress_callback(
  state: &mut StreamState,
  callback: Option<StateChangeCallback<'_>>,
  progress: &TranscriptionProgress,
) {
  let fallbacks = progress.timings().total_decoding_fallbacks() as usize;
  if progress.text().chars().count() < state.current_text().chars().count()
    && fallbacks == state.current_fallbacks()
  {
    let stale_text = state.current_text().to_string();
    apply(state, callback, |s| {
      s.unconfirmed_text_mut().push(stale_text);
    });
  }
  let text = progress.text().to_string();
  apply(state, callback, |s| {
    s.set_current_text(text);
  });
  apply(state, callback, |s| {
    s.set_current_fallbacks(fallbacks);
  });
}

/// The push-based streaming state machine — ports the `actor`
/// `AudioStreamTranscriber` (`AudioStreamTranscriber.swift:26-228`) minus
/// its microphone lifecycle (this module's doc). A caller owns the audio
/// source (microphone, file, network stream, ...) and calls
/// [`Self::push_samples`] whenever new samples arrive, reading
/// [`Self::state`] for the session's live transcript.
///
/// Bare struct, no bounds — bounds live on the `impl` blocks below,
/// narrowed to just [`Self::push_samples`] and its private helpers, the
/// only members needing `B: InferenceBackend` (golden §8; mirrors
/// [`crate::transcribe::TranscribeTask`]/[`crate::transcribe::WhisperKit`]'s
/// own two-impl-block split).
pub struct AudioStreamTranscriber<'ctx, B> {
  backend: &'ctx B,
  tokenizer: &'ctx WhisperTokenizer,
  decoding_options: DecodingOptions,
  stream_options: AudioStreamOptions,
  state: StreamState,
  state_callback: Option<StateChangeCallback<'ctx>>,
  buffer: Vec<f32>,
  energy: EnergyTracker,
}

// Construction and field accessors: no bound — none of these touch an
// `InferenceBackend` method, so none may demand one (golden §8's "where
// clauses on the methods/impls that need them"; the same split
// `crate::transcribe::WhisperKit`'s own impl blocks already demonstrate).
impl<'ctx, B> AudioStreamTranscriber<'ctx, B> {
  /// A fresh streaming session over `backend`/`tokenizer`, with default
  /// [`AudioStreamOptions`], a fresh [`StreamState`], no state-change
  /// callback, and an empty sample buffer — ports the pipeline-component
  /// slice of Swift's `AudioStreamTranscriber.init`
  /// (`AudioStreamTranscriber.swift:43-74`; see
  /// [`crate::transcribe::WhisperKit::audio_stream_transcriber`] for the
  /// convenience constructor mirroring Swift's call site).
  pub fn new(
    backend: &'ctx B,
    tokenizer: &'ctx WhisperTokenizer,
    decoding_options: DecodingOptions,
  ) -> Self {
    Self {
      backend,
      tokenizer,
      decoding_options,
      stream_options: AudioStreamOptions::new(),
      state: StreamState::new(),
      state_callback: None,
      buffer: Vec::new(),
      energy: EnergyTracker::default(),
    }
  }

  /// Builder form of [`Self::set_stream_options`].
  #[must_use]
  #[inline(always)]
  pub const fn with_stream_options(mut self, stream_options: AudioStreamOptions) -> Self {
    self.set_stream_options(stream_options);
    self
  }
  /// Replaces the streaming-session knobs (confirmation window, VAD
  /// gating, silence threshold, compression-check window).
  #[inline(always)]
  pub const fn set_stream_options(&mut self, stream_options: AudioStreamOptions) -> &mut Self {
    self.stream_options = stream_options;
    self
  }

  /// Builder form of [`Self::set_state_callback`].
  #[must_use]
  #[inline(always)]
  pub const fn with_state_callback(mut self, state_callback: StateChangeCallback<'ctx>) -> Self {
    self.set_state_callback(state_callback);
    self
  }
  /// Installs the callback fired with `(old, new)` after every
  /// [`StreamState`] mutation — Swift's `stateChangeCallback`
  /// (`AudioStreamTranscriber.swift:33`, `:55`).
  #[inline(always)]
  pub const fn set_state_callback(
    &mut self,
    state_callback: StateChangeCallback<'ctx>,
  ) -> &mut Self {
    self.state_callback = Some(state_callback);
    self
  }

  /// This session's live state.
  #[inline(always)]
  pub const fn state(&self) -> &StreamState {
    &self.state
  }

  /// Total samples accumulated in the session buffer so far (Swift
  /// `audioProcessor.audioSamples.count`, read indirectly through
  /// `AudioStreamTranscriber.swift`'s `currentBuffer.count`).
  #[inline(always)]
  pub const fn buffer_len(&self) -> usize {
    self.buffer.len()
  }

  /// This session's streaming knobs.
  #[inline(always)]
  pub const fn stream_options(&self) -> &AudioStreamOptions {
    &self.stream_options
  }
}

impl<B> AudioStreamTranscriber<'_, B>
where
  B: InferenceBackend,
{
  /// Pushes newly captured `samples` onto the session buffer and, if
  /// enough new voiced audio has accumulated, runs one transcription pass
  /// over the whole buffer. Ports `transcribeCurrentBuffer`
  /// (`AudioStreamTranscriber.swift:126-193`) with the caller as the loop
  /// instead of Swift's `Task.sleep`-driven `realtimeLoop` (`:98-107`,
  /// dropped — this module's doc): a non-[`StreamUpdate::Transcribed`]
  /// return is this port's "come back with more audio" signal, replacing
  /// Swift's 100 ms sleep-and-retry.
  ///
  /// 1. Absorbs `samples` into the buffer and the energy tracker, then
  ///    publishes [`StreamState::buffer_energy_slice`] (ports
  ///    `onAudioBufferCallback`, `:109-111`).
  /// 2. If fewer than (or exactly) 1 s of audio arrived since the last
  ///    transcription pass, sets the waiting placeholder text when
  ///    [`StreamState::current_text`] is still empty and returns
  ///    [`StreamUpdate::AwaitingAudio`] (`:131-140`).
  /// 3. When [`AudioStreamOptions::use_vad`] and
  ///    [`crate::audio::is_voice_detected`] finds no voice in the new
  ///    audio, sets the same placeholder and returns
  ///    [`StreamUpdate::AwaitingVoice`] (`:142-157`).
  /// 4. Otherwise runs the transcription (see the private
  ///    `transcribe_audio_samples` helper), clears
  ///    [`StreamState::current_text`]/[`StreamState::unconfirmed_text_slice`],
  ///    promotes segments past the confirmation watermark, and returns
  ///    [`StreamUpdate::Transcribed`] (`:159-192`).
  ///
  /// # Errors
  /// Whatever [`crate::transcribe::TranscribeTask::run`] returns,
  /// propagated directly. **Documented deviation:** Swift's
  /// `realtimeLoop` instead logs the error and breaks its loop silently
  /// (`:102-104`); this port surfaces the failure to the caller as a
  /// `Result` rather than swallowing it.
  pub fn push_samples(&mut self, samples: &[f32]) -> Result<StreamUpdate, TranscribeError> {
    // :109-111 (onAudioBufferCallback) — folded into every push; no
    // separate mic callback exists at this sans-I/O boundary.
    self.buffer.extend_from_slice(samples);
    self.energy.absorb(&self.buffer);
    let buffer_energy = self.energy.relative_energies();
    apply(&mut self.state, self.state_callback, |s| {
      s.set_buffer_energy(buffer_energy);
    });

    // :131-140. `saturating_sub` (not Swift's signed `Int` subtraction):
    // `last_buffer_size` is only ever a past `buffer.len()` and the
    // buffer never shrinks, so this never actually saturates — matches
    // Swift's own guarantee, just expressed for `usize` instead of `Int`.
    let next_buffer_samples = self
      .buffer
      .len()
      .saturating_sub(self.state.last_buffer_size());
    let next_buffer_seconds = next_buffer_samples as f32 / SAMPLE_RATE as f32;
    if next_buffer_seconds <= 1.0 {
      self.set_waiting_text_if_empty();
      return Ok(StreamUpdate::AwaitingAudio);
    }

    // :142-157.
    if self.stream_options.use_vad()
      && !is_voice_detected(
        self.state.buffer_energy_slice(),
        next_buffer_seconds,
        self.stream_options.silence_threshold(),
      )
    {
      self.set_waiting_text_if_empty();
      return Ok(StreamUpdate::AwaitingVoice);
    }

    // :160.
    let buffer_len = self.buffer.len();
    apply(&mut self.state, self.state_callback, |s| {
      s.set_last_buffer_size(buffer_len);
    });

    let transcription = self.transcribe_audio_samples()?;

    // :164-165.
    apply(&mut self.state, self.state_callback, |s| {
      s.set_current_text("");
    });
    apply(&mut self.state, self.state_callback, |s| {
      s.set_unconfirmed_text(Vec::new());
    });

    // :168-192 — confirmation logic: past `required` segments, the
    // earliest run is promoted to confirmed once its end advances the
    // watermark (unless already present as a contiguous run); the rest
    // (or, under `required`, every segment) stays unconfirmed.
    let segments = transcription.segments_slice();
    let required = self.stream_options.required_segments_for_confirmation();
    if segments.len() > required {
      let confirm_count = segments.len() - required;
      let (confirmed, remaining) = segments.split_at(confirm_count);
      if let Some(last_confirmed) = confirmed.last()
        && last_confirmed.end() > self.state.last_confirmed_segment_end_seconds()
      {
        let watermark = last_confirmed.end();
        apply(&mut self.state, self.state_callback, |s| {
          s.set_last_confirmed_segment_end_seconds(watermark);
        });
        if !contains_subsequence(self.state.confirmed_segments_slice(), confirmed) {
          apply(&mut self.state, self.state_callback, |s| {
            s.confirmed_segments_mut().extend_from_slice(confirmed);
          });
        }
      }
      apply(&mut self.state, self.state_callback, |s| {
        s.set_unconfirmed_segments(remaining);
      });
    } else {
      apply(&mut self.state, self.state_callback, |s| {
        s.set_unconfirmed_segments(segments);
      });
    }

    Ok(StreamUpdate::Transcribed)
  }

  /// Sets [`StreamState::current_text`] to [`WAITING_FOR_SPEECH_TEXT`]
  /// when it's still empty — shared by both of [`Self::push_samples`]'s
  /// "not enough voiced audio yet" branches
  /// (`AudioStreamTranscriber.swift:136-138`, `:151-153`; byte-identical
  /// text both places, guarded by the same `state.currentText == ""`
  /// check both places).
  fn set_waiting_text_if_empty(&mut self) {
    if self.state.current_text().is_empty() {
      apply(&mut self.state, self.state_callback, |s| {
        s.set_current_text(WAITING_FOR_SPEECH_TEXT);
      });
    }
  }

  /// Runs one [`crate::transcribe::TranscribeTask`] over the whole
  /// accumulated buffer, clipped from the last confirmed watermark.
  /// Ports `transcribeAudioSamples` (`AudioStreamTranscriber.swift:
  /// 195-206`).
  ///
  /// The live [`StreamState`] is moved into a [`Mutex`] for the duration
  /// of the run: [`crate::transcribe::TranscribeTask::run`]'s progress
  /// callback is a `&dyn Fn(..) + Sync` that may be called repeatedly
  /// while `run` holds only `&self`, so it cannot capture `&mut
  /// self.state` the way the rest of [`Self::push_samples`] mutates it
  /// directly — the `Mutex` supplies the interior mutability the callback
  /// needs instead, exactly the way [`Self::state_callback`] itself is
  /// already `Sync` rather than `FnMut`. [`Self::state`] briefly holds
  /// [`StreamState::default`] as a placeholder for the swap; nothing can
  /// observe that, since `push_samples` never returns control to a caller
  /// between the swap-out here and the swap-back-in below.
  fn transcribe_audio_samples(&mut self) -> Result<TranscriptionResult, TranscribeError> {
    let mut options = self.decoding_options.clone();
    options.set_clip_timestamps(vec![self.state.last_confirmed_segment_end_seconds()]);
    let compression_check_window = self.stream_options.compression_check_window();
    let state_callback = self.state_callback;

    let state_mutex = Mutex::new(std::mem::take(&mut self.state));
    let progress_callback = |progress: &TranscriptionProgress| -> Option<bool> {
      {
        let mut guard = state_mutex.lock().expect("stream state mutex poisoned");
        on_progress_callback(&mut guard, state_callback, progress);
      }
      should_stop_early(progress, &options, compression_check_window)
    };

    let result = TranscribeTask::new(self.backend, self.tokenizer)
      .with_progress_callback(&progress_callback)
      .run(&self.buffer, &options);

    self.state = state_mutex
      .into_inner()
      .expect("stream state mutex poisoned");

    result
  }
}
