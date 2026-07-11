//! The transcription window loop: turns the decode stack into full-audio
//! transcription. Ports `TranscribeTask` (`TranscribeTask.swift:57-411`) —
//! [`TranscribeTask::run`] is the seek/window loop (:57-296), including the
//! optional word-timestamp re-anchoring step
//! (`options.word_timestamps()`, :196-233) that runs
//! [`crate::segment::add_word_timestamps`] against the accepted attempt's
//! alignment snapshot; its private `decode_with_fallback` is the
//! temperature-fallback ladder (:316-411) that produces that snapshot
//! (:198).
//!
//! **Documented deviation — parameterless reset:** Swift's
//! `decoderInputs.reset(maxTokenContext:)` (:271-273) takes the window's
//! `sampleLength`/`Constants.maxTokenContext` bound as a parameter, because
//! Swift's decoder tensors are allocated once at the model's absolute
//! maximum and `reset` re-derives a *mask* over that fixed allocation from
//! the given bound. This port's
//! [`crate::backend::InferenceBackend::reset_decoder_state`] takes no such
//! parameter: the backend's decoder state is sized once at allocation
//! ([`crate::backend::InferenceBackend::new_decoder_state`]), and reset
//! always restores that same fixed allocation's initial state, so there is
//! nothing left for a per-call bound to vary.
//!
//! **Not ported:** Swift's `Progress`/`Logging.beginSignpost`
//! instrumentation (:62-63, 101-103, 110, 276-277, 282) and its
//! `Task.checkCancellation()` cooperative-cancellation checks (:135, 144,
//! 165) have no equivalent here — this is a sync library with no `Progress`
//! type and no structured-concurrency task tree to cancel.
//! [`crate::decode::TranscriptionProgressCallback`]'s `Some(false)`-triggered
//! [`AtomicBool`] early stop is a different, narrower mechanism (ends the
//! *current window's* decode loop only, like reaching end-of-text) and is
//! the only cancellation-adjacent hook this port exposes. Swift's `open`
//! `windowPreprocess`/`windowPostProcess` subclass hooks (:42-55) likewise
//! have no port: this crate has no subclassing extension point.
//!
//! # `WhisperKit`
//!
//! [`WhisperKit<B>`] composes everything above (and everything shipped by
//! earlier tasks) into the public pipeline entry point: single-audio
//! transcription with VAD dispatch ([`WhisperKit::transcribe`], ports
//! `WhisperKit.swift:867-931`), batch transcription
//! ([`WhisperKit::transcribe_all`], ports `transcribeWithOptions`,
//! `WhisperKit.swift:716-812`), and one-shot language detection
//! ([`WhisperKit::detect_language`], ports `WhisperKit.swift:534-581`).
//! [`WhisperKit::<CoreMlBackend>::new`] loads models and tokenizer from
//! [`crate::options::Options`] and detects the model variant the same way
//! `loadTokenizerIfNeeded` does (`WhisperKit.swift:444-457`);
//! [`WhisperKit::with_backend`] is the mock/e2e-hermetic construction seam
//! every test above this layer uses instead.
//!
//! **Concurrency note — why `transcribe_all` has no `CoreMlBackend`
//! instance:** [`WhisperKit::transcribe_all`] batches workers over
//! `std::thread::scope`, which requires `B: Sync` (every worker holds a
//! `&self.backend` shared across threads at once). `coremlit::Model` is
//! documented `Send` but deliberately **not** `Sync` — Apple's contract is
//! "use an `MLModel` instance on one thread ... at a time" — so
//! [`crate::backend::coreml::CoreMlBackend`], which owns three `Model`s,
//! is not `Sync` either, and `WhisperKit<CoreMlBackend>::transcribe_all`
//! is simply never a callable method (there is no `impl` providing it for
//! that `B`). This is by design, not a gap: real concurrent batch
//! transcription over CoreML needs one backend — hence one `WhisperKit` —
//! per worker thread, each driven independently, not one shared backend
//! serving many. [`crate::backend::mock::MockBackend`] IS `Sync` (its
//! mutable state lives behind `Arc<Mutex<_>>`), so hermetic batch tests
//! exercise the real code path. [`WhisperKit::transcribe`], by contrast,
//! never needs `B: Sync`: its VAD-chunked branch runs each chunk's
//! [`TranscribeTask`] sequentially rather than reusing
//! [`WhisperKit::transcribe_all`]'s worker pool, specifically so it stays
//! callable on `WhisperKit<CoreMlBackend>` — see that method's own doc for
//! the full reasoning.

use std::{sync::atomic::AtomicBool, time::Instant};

use unicode_categories::UnicodeCategories;

use crate::{
  audio::{self, chunker},
  backend::{AlignmentMatrix, InferenceBackend, coreml::CoreMlBackend},
  constants::{APPEND_PUNCTUATION, DEFAULT_LANGUAGE_CODE, PREPEND_PUNCTUATION, SAMPLE_RATE},
  decode::{self, TranscriptionProgressCallback, sampler::GreedyTokenSampler},
  error::{DecodeError, ModelError, TranscribeError},
  model::{ModelVariant, detect_variant, manager::ModelManager},
  options::{DecodingOptions, Options},
  result::{
    DecodingResult, TranscriptionProgress, TranscriptionResult, TranscriptionSegment,
    TranscriptionTimings, merge_transcription_results, needs_fallback,
  },
  segment,
  tokenizer::WhisperTokenizer,
};

#[cfg(test)]
mod tests;

/// Trims Swift's `.whitespaces` `CharacterSet` (Unicode general category
/// `Zs` plus U+0009 CHARACTER TABULATION; no newlines) off both ends of `s`
/// — ports `String.trimmingCharacters(in: .whitespaces)`
/// (`TranscribeTask.swift:305`). Rust's [`str::trim`] is narrower-scoped in
/// the wrong direction for this: it also strips newlines, which Swift's
/// `.whitespaces` (unlike `.whitespacesAndNewlines`) deliberately does not.
/// Duplicated from [`crate::segment`]'s private helper of the same name and
/// behavior, since that one isn't visible outside its own module.
fn trim_swift_whitespaces(s: &str) -> &str {
  s.trim_matches(|c: char| c.is_separator_space() || c == '\u{0009}')
}

// ---------------------------------------------------------------------
// SegmentDiscoveryCallback
// ---------------------------------------------------------------------

/// Fired once per decoded window, with that window's segments only (not the
/// running total). Ports Swift `SegmentDiscoveryCallback`
/// (`Models.swift:667-669`).
pub type SegmentDiscoveryCallback<'a> = &'a (dyn Fn(&[TranscriptionSegment]) + Sync);

// ---------------------------------------------------------------------
// TranscribeTask
// ---------------------------------------------------------------------

/// Drives one full-audio transcription: the seek/window loop and its
/// per-window temperature-fallback ladder. Ports `TranscribeTask`
/// (`TranscribeTask.swift:57-411`); see the module docs for what's deferred
/// or deliberately not ported.
pub struct TranscribeTask<'ctx, B> {
  backend: &'ctx B,
  tokenizer: &'ctx WhisperTokenizer,
  segment_callback: Option<SegmentDiscoveryCallback<'ctx>>,
  progress_callback: Option<TranscriptionProgressCallback<'ctx>>,
  window_id_offset: usize,
}

impl<'ctx, B> TranscribeTask<'ctx, B> {
  /// A task against `backend`/`tokenizer` with no callbacks and a zero
  /// window-id offset.
  pub const fn new(backend: &'ctx B, tokenizer: &'ctx WhisperTokenizer) -> Self {
    Self {
      backend,
      tokenizer,
      segment_callback: None,
      progress_callback: None,
      window_id_offset: 0,
    }
  }

  /// Builder form of [`Self::set_segment_callback`].
  #[must_use]
  #[inline(always)]
  pub const fn with_segment_callback(
    mut self,
    segment_callback: SegmentDiscoveryCallback<'ctx>,
  ) -> Self {
    self.set_segment_callback(segment_callback);
    self
  }
  /// Sets the callback fired with each window's segments as they're
  /// discovered.
  #[inline(always)]
  pub const fn set_segment_callback(
    &mut self,
    segment_callback: SegmentDiscoveryCallback<'ctx>,
  ) -> &mut Self {
    self.segment_callback = Some(segment_callback);
    self
  }

  /// Builder form of [`Self::set_progress_callback`].
  #[must_use]
  #[inline(always)]
  pub const fn with_progress_callback(
    mut self,
    progress_callback: TranscriptionProgressCallback<'ctx>,
  ) -> Self {
    self.set_progress_callback(progress_callback);
    self
  }
  /// Sets the callback fired with per-step decode progress.
  #[inline(always)]
  pub const fn set_progress_callback(
    &mut self,
    progress_callback: TranscriptionProgressCallback<'ctx>,
  ) -> &mut Self {
    self.progress_callback = Some(progress_callback);
    self
  }

  /// Builder form of [`Self::set_window_id_offset`].
  #[must_use]
  #[inline(always)]
  pub const fn with_window_id_offset(mut self, window_id_offset: usize) -> Self {
    self.set_window_id_offset(window_id_offset);
    self
  }
  /// Sets the chunk-worker id [`TranscriptionProgress::window_id`] updates
  /// are offset by — lets a caller running several [`TranscribeTask`]s
  /// concurrently over different audio chunks keep each worker's window ids
  /// distinct.
  #[inline(always)]
  pub const fn set_window_id_offset(&mut self, window_id_offset: usize) -> &mut Self {
    self.window_id_offset = window_id_offset;
    self
  }
}

impl<B> TranscribeTask<'_, B>
where
  B: InferenceBackend,
{
  /// Runs the seek/window loop over `audio`, producing one
  /// [`TranscriptionResult`]. Ports `TranscribeTask.run`
  /// (`TranscribeTask.swift:57-296`).
  ///
  /// Per seek clip (`chunker::prepare_seek_clips`), decodes consecutive
  /// [`crate::backend::ModelDims::window_samples`]-sized (or shorter, for
  /// the final partial window) windows until fewer than
  /// `options.window_clip_time()` seconds of the clip remain, feeding each
  /// window's decode through the private temperature-fallback ladder and
  /// then [`crate::segment::find_seek_point_and_segments`] to turn it into
  /// the next seek offset and zero or more segments. The final transcript
  /// is every segment's tokens, filtered to non-special ids and decoded,
  /// trimmed the same way Swift trims it (`.trimmingCharacters(in:
  /// .whitespaces)`).
  ///
  /// # Errors
  /// [`TranscribeError::Audio`] if `options`' clip timestamps are malformed;
  /// [`TranscribeError::Decode`] if a backend feature-extraction, encode, or
  /// decode step fails; [`TranscribeError::Segment`] if seeking a decoded
  /// window into segments fails, or — when `options.word_timestamps()` is
  /// set — if aligning that window's words fails (see the word-timestamp
  /// block below for when this fires); [`TranscribeError::Tokenizer`] if
  /// the final transcript decode fails.
  pub fn run(
    &self,
    audio: &[f32],
    options: &DecodingOptions,
  ) -> Result<TranscriptionResult, TranscribeError> {
    let pipeline_start = Instant::now();
    let mut timings = TranscriptionTimings::new();

    // :74 — content_frames / sampleRate, less the first explicit clip
    // start (clip_timestamps stores seconds, not samples).
    let content_frames = audio.len();
    let clip_start_seconds = options
      .clip_timestamps_slice()
      .first()
      .copied()
      .unwrap_or(0.0);
    timings.set_input_audio_seconds(
      content_frames as f64 / f64::from(SAMPLE_RATE) - f64::from(clip_start_seconds),
    );

    let mut all_segments: Vec<TranscriptionSegment> = Vec::new();
    let mut detected_language: Option<String> = None;

    // :82-85 — decoder init timing covers only state allocation, matching
    // Swift's `decoderInitTime` scope exactly (the prefill call right below
    // is untimed in Swift too).
    let decoder_init_start = Instant::now();
    let mut state = self
      .backend
      .new_decoder_state()
      .map_err(DecodeError::from)?;
    timings.set_decoding_init(decoder_init_start.elapsed().as_secs_f64());

    // :83+90-93 — `[SOT]` unless prefill is requested, in which case
    // `prefill_tokens` derives the full prompt; see that function's own doc
    // for why there's no separate "no options" parameter here.
    let mut initial_prompt: Vec<u32> =
      vec![self.tokenizer.special_tokens().start_of_transcript_token()];
    if options.use_prefill_prompt() {
      initial_prompt = decode::prefill_tokens(
        options,
        self.tokenizer,
        self.backend.dims().is_multilingual(),
      );
    }

    let seek_clips = chunker::prepare_seek_clips(options.clip_timestamps_slice(), content_frames)?;
    // :113 — samples clipped from a clip's end to avoid trailing-silence
    // hallucinations; computed once here rather than once per clip
    // (Swift's own placement), since its value never varies by clip.
    let window_padding = (options.window_clip_time() * SAMPLE_RATE as f32) as usize;
    let window_samples = self.backend.dims().window_samples();

    let decode_loop_start = Instant::now();
    for (seek_clip_start, seek_clip_end) in seek_clips {
      // :116 — Swift's signed `seek < seekClipEnd - windowPadding` is
      // always false once `seekClipEnd <= windowPadding`; ported as a
      // guarded skip rather than a subtraction that would underflow.
      if seek_clip_end <= window_padding {
        continue;
      }
      let clip_guard = seek_clip_end - window_padding;
      let mut seek = seek_clip_start;

      while seek < clip_guard {
        // :120 — bounded with `saturating_sub`, not `-`: an explicit
        // `clip_timestamps` end beyond `content_frames`, combined with a
        // model hallucinating a large timestamp near a short final window,
        // could otherwise underflow here. Swift's signed `Int` subtraction
        // has no such trap, but also has no defined behavior on that same
        // malformed input.
        let segment_size = window_samples
          .min(content_frames.saturating_sub(seek))
          .min(seek_clip_end.saturating_sub(seek));

        // Nothing left to decode: `seek` reached (or a hallucinated
        // timestamp pushed it past) the physical audio while an
        // out-of-range clip end keeps `clip_guard` unsatisfied. Swift has
        // no such guard — a `without_timestamps` window then advances by
        // this zero and re-decodes padded silence forever, and a
        // past-the-end `seek` crashes its slice — but a non-terminating
        // or aborting publicly reachable configuration is not benign
        // parity (documented deviation): move to the next clip instead.

        if segment_size == 0 {
          break;
        }

        // :125-133 — no `windowPreprocess` hook (see module docs).
        let audio_processing_start = Instant::now();
        let padded = audio::pad_or_trim(&audio[seek..seek + segment_size], window_samples);
        timings.set_audio_processing(
          timings.audio_processing() + audio_processing_start.elapsed().as_secs_f64(),
        );
        timings.set_total_audio_processing_runs(timings.total_audio_processing_runs() + 1.0);

        // :136-142.
        let logmel_start = Instant::now();
        let features = self
          .backend
          .extract_features(&padded)
          .map_err(DecodeError::from)?;
        timings.set_logmels(timings.logmels() + logmel_start.elapsed().as_secs_f64());
        timings.set_total_logmel_runs(timings.total_logmel_runs() + 1.0);

        // :144-151.
        let encoder_start = Instant::now();
        let encoder_output = self.backend.encode(&features).map_err(DecodeError::from)?;
        timings.set_encoding(timings.encoding() + encoder_start.elapsed().as_secs_f64());
        timings.set_total_encoding_runs(timings.total_encoding_runs() + 1.0);

        // windowId attribution happens inside `decode_with_fallback`,
        // which already receives `timings`; the early-stop flag is that
        // function's own per-attempt concern. `captured_alignment` is the
        // accepted attempt's alignment-weight snapshot (see that
        // function's own doc); `None` unless `options.word_timestamps()`.
        let (decoding_result, captured_alignment) = self.decode_with_fallback(
          &encoder_output,
          &mut state,
          &mut initial_prompt,
          &mut detected_language,
          options,
          &mut timings,
        )?;

        // :178-194.
        let windowing_start = Instant::now();
        let previous_seek = seek;
        let (new_seek, mut current_segments) = segment::find_seek_point_and_segments(
          &decoding_result,
          options,
          all_segments.len(),
          seek,
          segment_size,
          self.tokenizer,
        )?;
        seek = seek.max(new_seek);

        // :196-233 — optional word-timestamp re-anchoring, run against the
        // accepted attempt's alignment snapshot.
        if options.word_timestamps()
          && let Some(matrix) = &captured_alignment
        {
          let word_timestamps_start = Instant::now();
          let language = detected_language
            .as_deref()
            .unwrap_or(DEFAULT_LANGUAGE_CODE);
          let with_words = segment::add_word_timestamps(
            current_segments.as_deref().unwrap_or(&[]), // Swift quirk: nil -> [] (:202)
            &matrix.view(),
            self.tokenizer,
            language,
            previous_seek,
            PREPEND_PUNCTUATION,
            APPEND_PUNCTUATION,
            previous_seek as f32 / SAMPLE_RATE as f32, // :209
          )?;
          timings.set_decoding_word_timestamps(
            timings.decoding_word_timestamps() + word_timestamps_start.elapsed().as_secs_f64(),
          );
          timings
            .set_total_timestamp_alignment_runs(timings.total_timestamp_alignment_runs() + 1.0);
          // :217-218 — drop zero-length segments.
          let filtered: Vec<TranscriptionSegment> = with_words
            .into_iter()
            .filter(|segment| segment.end() > segment.start())
            .collect();
          // :221-223 — refine seek with the (more accurate) last word end.
          if let Some(last_end) = filtered.last().map(TranscriptionSegment::end) {
            seek = seek.max((last_end * SAMPLE_RATE as f32) as usize);
          }
          // CORRECTION (this task's brief describes a silence-skipped
          // window's `None` becoming `Some(vec![])` here, mirroring
          // Swift's `currentSegments ?? []`): `current_segments` is `None`
          // only via `find_seek_point_and_segments`'s silence-skip branch,
          // which makes `segments` above `&[]`; `add_word_timestamps` on
          // an empty `segments` slice always returns
          // `SegmentError::InvalidAlignmentShape` (its prefix-take
          // alignment is sized off `segments`' own token count — zero
          // here — independent of `matrix`'s real shape; see that
          // function's `# Errors` doc), which the `?` above propagates out
          // of `run` as `TranscribeError::Segment` instead of reaching
          // this assignment. That is the faithful analogue of Swift's own
          // unconditional crash on the same input
          // (`SegmentSeeker.swift:208`/`:211`'s unguarded `1...0`
          // `ClosedRange` traps for a zero-row filtered matrix) — a typed,
          // recoverable error in place of a hard process abort, never a
          // silent `Some(vec![])`. This path is dormant today regardless:
          // `no_speech_prob` is permanently `0.0`
          // (`crate::decode::decode_text`'s own faithfully-ported upstream
          // TODO), so no positive `no_speech_threshold` — including the
          // default — can ever reach it.
          current_segments = Some(filtered);
        }

        // :236-239 — hardened beyond Swift's bare `previous_seek + max`:
        // the sum saturates (a huge configured cap must not overflow),
        // and the cap floors at one sample of forward progress so the
        // publicly reachable `Some(0)` degrades to a slow-but-finite
        // loop instead of decoding the same window forever.
        if let Some(max_window_seek) = options.max_window_seek() {
          seek = seek.min(previous_seek.saturating_add(max_window_seek.max(1)));
        }

        // :241-244 — a silent window: skip straight to the next iteration
        // *before* the reset below, exactly like Swift's `continue` here
        // (the decoder state is left as the just-finished decode call
        // leaves it, not explicitly reset, until the next `Some` window's
        // own reset catches it back up).
        let Some(current_segments) = current_segments else {
          continue;
        };

        // :252-265 — no `windowPostProcess` hook (see module docs); the
        // callback fires with this window's segments only, before they're
        // folded into the running total.
        if let Some(callback) = self.segment_callback {
          callback(&current_segments);
        }
        all_segments.extend(current_segments);

        timings.set_decoding_windowing(
          timings.decoding_windowing() + windowing_start.elapsed().as_secs_f64(),
        );
        timings.set_total_decoding_windows(timings.total_decoding_windows() + 1.0);

        // :270-273 — see module docs for the parameterless-reset deviation.
        self.backend.reset_decoder_state(&mut state);
      }
    }
    timings.set_decoding_loop(decode_loop_start.elapsed().as_secs_f64());

    // :298-305 — every segment's tokens, filtered to non-special ids,
    // decoded, and trimmed. `all_tokens` isn't tracked as its own running
    // accumulator (unlike Swift's `allTokens`): since nothing here diverges
    // `allSegments` from the tokens their own segments carry (no
    // `windowPostProcess` hook to make them differ), deriving the token
    // list from `all_segments` once at the end is equivalent and avoids a
    // redundant parallel Vec.
    let special_token_begin = self.tokenizer.special_tokens().special_token_begin();
    let word_tokens: Vec<u32> = all_segments
      .iter()
      .flat_map(|segment| segment.tokens_slice().iter().copied())
      .filter(|&token| token < special_token_begin)
      .collect();
    let text = self.tokenizer.decode(&word_tokens, false)?;
    let trimmed_text = trim_swift_whitespaces(&text);

    timings.set_full_pipeline(pipeline_start.elapsed().as_secs_f64());

    Ok(TranscriptionResult::new(
      trimmed_text,
      all_segments,
      detected_language.unwrap_or_else(|| DEFAULT_LANGUAGE_CODE.to_string()),
      timings,
    ))
  }

  /// The per-window temperature-fallback ladder: retries decoding at
  /// increasing temperatures (`options.temperature() + attempt as f32 *
  /// options.temperature_increment_on_fallback()`, `attempt` from `0` up to
  /// and including `options.temperature_fallback_count()`) until a decode
  /// does not [`needs_fallback`], or every attempt is exhausted, in which
  /// case the last (worst) attempt's result stands regardless. Ports
  /// `decodeWithFallback` (`TranscribeTask.swift:316-411`).
  ///
  /// Two corrections against this task's own brief, checked directly
  /// against `Models.swift`/`TranscribeTask.swift`:
  /// - [`needs_fallback`] takes a caller-computed
  ///   `first_token_log_prob_too_low` flag as its own first parameter, not
  ///   a two-argument call — recomputed here from
  ///   [`DecodingResult::first_token_log_prob`] against
  ///   `options.first_token_logprob_threshold()`, the same strict
  ///   comparison `TextDecoder.swift:662-667` makes.
  /// - the language-detection probe's error is swallowed, not propagated:
  ///   Swift's call is `try?` (`TranscribeTask.swift:342`), and
  ///   [`decode::detect_language`]'s own doc says it resets `state`
  ///   unconditionally, errors included, specifically so a caller can carry
  ///   on after a failed probe — matching that contract instead of `?`.
  ///
  /// **Faithfully-ported Swift quirk:** `timings.total_decoding_fallbacks`
  /// is *assigned* the 0-based `attempt` index on every fallback
  /// (`TranscribeTask.swift:397`, `totalDecodingFallbacks = Double(i)`), not
  /// accumulated. After a multi-window run this field therefore reflects
  /// only the most recent fallback's attempt index, not a running total
  /// across windows — kept exactly as Swift computes it rather than
  /// "fixed" into an accumulator.
  ///
  /// Also returns an owned alignment-weight snapshot alongside the
  /// [`DecodingResult`], taken immediately after each attempt's
  /// `decode_text` returns and before that same attempt's own
  /// fallback-triggered reset, if any (`TranscribeTask.swift:198`'s
  /// per-attempt `decodingResult.cache?.alignmentWeights` capture). A later
  /// attempt's snapshot always overwrites an earlier one, so by the time
  /// this function returns, the snapshot belongs to the accepted (last-run)
  /// attempt — it cannot instead be taken once, after the loop, because
  /// [`InferenceBackend::reset_decoder_state`] zeroes the live accumulator
  /// [`InferenceBackend::alignment_weights`] borrows from in place on every
  /// fallback and again before the caller's next window (see
  /// [`AlignmentMatrix`]'s own doc). `None` when `options.word_timestamps()`
  /// is `false` or the backend has no alignment data for this window.
  #[allow(clippy::too_many_arguments)] // Mirrors Swift's decodeWithFallback argument
  // surface (mirroring decode_text's own precedent for this exact lint, per
  // its doc comment); no natural subset of these forms a cohesive struct
  // without inventing one purely to dodge the lint.
  fn decode_with_fallback(
    &self,
    encoder_output: &B::EncoderOutput,
    state: &mut B::DecoderState,
    initial_prompt: &mut Vec<u32>,
    detected_language: &mut Option<String>,
    options: &DecodingOptions,
    timings: &mut TranscriptionTimings,
  ) -> Result<(DecodingResult, Option<AlignmentMatrix>), TranscribeError> {
    let special = *self.tokenizer.special_tokens();

    // :156-158 — windowId for progress attribution, computed once before
    // this window's attempts touch `timings`' window/fallback counters (Swift
    // computes this in `run`, immediately before calling this function; doing
    // it here instead is equivalent, since `timings` hasn't changed for this
    // window yet either way).
    // The `.max(0.0)` diverges from Swift's plain Int arithmetic, which
    // can go negative under heavy early-window fallback; `usize` forces
    // the clamp, and 0 is the sanest floor for progress metadata.
    let window_id = self.window_id_offset
      + (timings.total_decoding_windows() - timings.total_decoding_fallbacks()).max(0.0) as usize;
    let stamped = self.progress_callback.map(|callback| {
      move |progress: &TranscriptionProgress| -> Option<bool> {
        let mut with_id = progress.clone();
        with_id.set_window_id(window_id);
        callback(&with_id)
      }
    });
    let window_callback: Option<TranscriptionProgressCallback<'_>> = stamped
      .as_ref()
      .map(|wrapper| wrapper as &(dyn Fn(&TranscriptionProgress) -> Option<bool> + Sync));

    let mut decoding = None;
    let mut captured_alignment: Option<AlignmentMatrix> = None;
    for attempt in 0..=options.temperature_fallback_count() {
      let attempt_start = Instant::now();
      let temperature =
        options.temperature() + attempt as f32 * options.temperature_increment_on_fallback();
      let mut sampler = GreedyTokenSampler::new(temperature, special.end_token(), options);
      // A FRESH early-stop latch per attempt: Swift initializes a new
      // early-stop entry for every decodeText invocation
      // (TextDecoder.swift:570), so a callback-stopped attempt whose
      // partial result triggers an ordinary fallback must not truncate
      // the retry (phase-gate round-5 finding).
      let early_stop = AtomicBool::new(false);

      // :340-365 — for a multilingual model with no explicit language and
      // detection requested, probe the language once, patch a per-attempt
      // options clone, and re-derive the prefill prompt from it.
      let mut window_options = options.clone();
      if self.backend.dims().is_multilingual()
        && options.language().is_empty()
        && options.detect_language()
      {
        // :351-352 — the probe's outcome is assigned UNCONDITIONALLY
        // (Swift's `try?` yields nil on failure), so a failed probe
        // clears any earlier window's/attempt's value and the
        // post-decode re-derivation below fires for THIS attempt —
        // last-write-wins, never sticky-to-first-success.
        // :337-343 — the probe samples through THIS attempt's sampler:
        // at nonzero temperature the language pick is a top-k draw, and
        // the draw advances the same RNG stream the attempt's text
        // tokens continue from.
        match decode::detect_language(
          self.backend,
          encoder_output,
          state,
          self.tokenizer,
          &mut sampler,
          timings,
        ) {
          Ok(probe) => {
            window_options.set_language(probe.language().to_string());
            *detected_language = Some(probe.language().to_string());
          }
          Err(_) => *detected_language = None,
        }
        if options.use_prefill_prompt() {
          *initial_prompt = decode::prefill_tokens(&window_options, self.tokenizer, true);
        }
      }

      let result = decode::decode_text(
        self.backend,
        encoder_output,
        state,
        initial_prompt.as_slice(),
        &mut sampler,
        &window_options,
        self.tokenizer,
        timings,
        &early_stop,
        window_callback,
      )?;

      // TranscribeTask.swift:198 — snapshot THIS attempt's alignment
      // weights now, before the fallback branch below can reset (and
      // thereby zero) the live accumulator they borrow from; see this
      // function's own doc for why the snapshot can't just be read once
      // after the loop instead.
      if options.word_timestamps() {
        captured_alignment = self
          .backend
          .alignment_weights(state)
          .map(|view| view.to_matrix());
      }

      // :375-378 — only used if language detection above never ran/set it.
      if detected_language.is_none() {
        *detected_language = Some(result.language().to_string());
      }

      let is_first_token_log_prob_too_low = options
        .first_token_logprob_threshold()
        .is_some_and(|threshold| result.first_token_log_prob() < threshold);
      let fallback = needs_fallback(is_first_token_log_prob_too_low, &result, options);
      decoding = Some(result);

      match fallback {
        Some(_reason) => {
          timings.set_decoding_fallback(
            timings.decoding_fallback() + attempt_start.elapsed().as_secs_f64(),
          );
          self.backend.reset_decoder_state(state);
          timings.set_total_decoding_fallbacks(attempt as f64);
        }
        None => break,
      }
    }
    Ok((
      decoding.expect("the loop runs at least once (0..=count always yields >= 1 attempt)"),
      captured_alignment,
    ))
  }
}

// ---------------------------------------------------------------------
// LanguageDetection
// ---------------------------------------------------------------------

/// The result of a one-shot language-detection probe — Swift's
/// `(language: String, langProbs: [String: Float])` tuple return from
/// `detectLanguage`/`detectLangauge` (`WhisperKit.swift:520-581`).
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageDetection {
  language: String,
  probs: Vec<(String, f32)>,
}

impl LanguageDetection {
  /// Builds a detection result from its resolved language and per-language
  /// probabilities.
  pub fn new(language: impl Into<String>, probs: impl Into<Vec<(String, f32)>>) -> Self {
    Self {
      language: language.into(),
      probs: probs.into(),
    }
  }

  /// The detected (or default-fallback) language, as an ISO code.
  #[inline(always)]
  pub fn language(&self) -> &str {
    self.language.as_str()
  }

  /// Per-language detection probabilities — Swift's `langProbs`
  /// dictionary, as an order-preserving list. In practice this holds at
  /// most one entry: [`decode::detect_language`] only ever records the
  /// single argmax-sampled language token's probability (see its own doc).
  #[inline(always)]
  pub const fn probs_slice(&self) -> &[(String, f32)] {
    self.probs.as_slice()
  }
}

// ---------------------------------------------------------------------
// WhisperKit
// ---------------------------------------------------------------------

/// Top-level transcription pipeline: owns an inference backend, tokenizer,
/// and (when detectable) the loaded model's variant, and composes
/// [`TranscribeTask`] into this crate's public entry points. See the
/// module doc's "`WhisperKit`" section for the citations and the
/// `transcribe`/`transcribe_all` concurrency split.
///
/// Bare struct, no bounds — bounds live on the `impl` blocks below,
/// narrowed further per method where only some of them need `B: Sync`
/// (golden §8).
pub struct WhisperKit<B> {
  backend: B,
  tokenizer: WhisperTokenizer,
  variant: Option<ModelVariant>,
}

impl WhisperKit<CoreMlBackend> {
  /// Builds a pipeline from `options`: resolves and loads the three CoreML
  /// models (prewarming first when requested), builds the
  /// [`CoreMlBackend`], loads the tokenizer, and detects the model variant
  /// from the loaded backend's dimensions. Ports the local-folder,
  /// eager-load slice of `WhisperKit.init`/`loadModels`/
  /// `loadTokenizerIfNeeded` (`WhisperKit.swift:354-470`) — `vocab` is the
  /// decoder logits dimension Swift calls `logitsDim`, `embed_dim` is the
  /// encoder's `encoderDim` (`WhisperKit.swift:455-457`).
  ///
  /// # Errors
  /// [`TranscribeError::Model`] if prewarming or loading the models fails,
  /// or if `options.load()` is `false`: this constructor always loads at
  /// construction time (Swift's `WhisperKitConfig.load` resolves to `true`
  /// whenever a model folder is given, `options::Options`'s own doc), so a
  /// `load = false` [`Options`] has no honest behavior for it to run —
  /// that is Plan 4's deferred lazy-load construction path instead, not a
  /// silent no-op here. Reported as [`ModelError::InvalidState`]: the
  /// closest existing variant to "operation not valid in this
  /// configuration" ([`ModelError`]'s own doc), and truer than inventing a
  /// new variant for what is really the same shape as the
  /// monolingual-model case [`Self::detect_language`] reports the same
  /// way.
  /// [`TranscribeError::Decode`] if the loaded models' introspected
  /// dimensions are inconsistent (wraps [`crate::backend::BackendError`]
  /// via [`CoreMlBackend::from_loaded`]).
  /// [`TranscribeError::Tokenizer`] if the tokenizer fails to load.
  pub fn new(options: &Options) -> Result<Self, TranscribeError> {
    // Checked before the prewarm, not in Swift's statement order (prewarm
    // first, then the load decision): in Swift `load = false` is a valid
    // lazy-load configuration whose prewarm is useful later, while here
    // it is an outright construction error — running a potentially
    // expensive ANE prewarm pass before reporting it would be pure waste.
    if !options.load() {
      return Err(
        ModelError::InvalidState {
          expected: "load = true (WhisperKit::new always loads at construction)",
          actual: "load = false",
        }
        .into(),
      );
    }
    let mut manager = ModelManager::new(options.model_folder(), options.compute());
    if options.prewarm() {
      manager.prewarm()?;
    }
    let models = manager.into_loaded()?;
    let backend = CoreMlBackend::from_loaded(models).map_err(DecodeError::from)?;
    let tokenizer = WhisperTokenizer::from_folder(options.tokenizer_folder())?;
    let dims = backend.dims();
    let variant = detect_variant(dims.vocab(), dims.embed_dim());
    Ok(Self {
      backend,
      tokenizer,
      variant,
    })
  }
}

// Construction and field accessors: no bound — none of these touch an
// `InferenceBackend` method, so none may demand one (golden §8's
// "where clauses on the methods/impls that need them"; the same split
// `TranscribeTask`'s two impl blocks above already demonstrate).
impl<B> WhisperKit<B> {
  /// Wraps an already-constructed `backend`/`tokenizer` directly, with no
  /// model-loading step — the seam [`crate::backend::mock::MockBackend`]-driven
  /// hermetic tests (and any other non-CoreML/e2e backend) use to build a
  /// pipeline. [`Self::variant`] starts `None`: variant detection is
  /// [`WhisperKit::<CoreMlBackend>::new`]'s job specifically, since it
  /// needs a loaded model's real dimensions to detect anything from —
  /// there is no Swift analogue of a variant-less `WhisperKit` to defer
  /// to instead.
  pub fn with_backend(backend: B, tokenizer: WhisperTokenizer) -> Self {
    Self {
      backend,
      tokenizer,
      variant: None,
    }
  }

  /// The wrapped inference backend.
  #[inline(always)]
  pub const fn backend(&self) -> &B {
    &self.backend
  }

  /// The wrapped tokenizer.
  #[inline(always)]
  pub const fn tokenizer(&self) -> &WhisperTokenizer {
    &self.tokenizer
  }

  /// The detected model variant, if [`WhisperKit::<CoreMlBackend>::new`]
  /// could resolve one from the loaded backend's dimensions. `None` for
  /// [`Self::with_backend`] pipelines, or for dimensions
  /// [`detect_variant`] doesn't recognize.
  #[inline(always)]
  pub const fn variant(&self) -> Option<ModelVariant> {
    self.variant
  }
}

impl<B> WhisperKit<B>
where
  B: InferenceBackend,
{
  /// Transcribes `audio` end to end: VAD-chunks long-form audio and merges
  /// the per-chunk results, or runs a single [`TranscribeTask`] for audio
  /// no longer than one window. Ports `WhisperKit.transcribe(audioArray:
  /// decodeOptions:callback:segmentCallback:)` (`WhisperKit.swift:
  /// 867-931`).
  ///
  /// **Documented deviation — this port merges where Swift's library
  /// layer doesn't:** Swift's `transcribe` returns `[TranscriptionResult]`
  /// (unmerged: one element per chunk, or a single-element array for the
  /// unchunked case, `runTranscribeTask`'s `return [transcribeTaskResult]`
  /// at :1010) — `WhisperKit.swift` itself never calls
  /// `TranscriptionUtilities.mergeTranscriptionResults` anywhere; that
  /// function is only ever called from application code layered on top
  /// (`Sources/ArgmaxCLI/TranscribeCLI.swift`,
  /// `Sources/ArgmaxCLI/Server/OpenAIHandler.swift`). This port's
  /// [`TranscribeTask::run`] already committed (Task 11) to a single
  /// [`TranscriptionResult`] return rather than Swift's redundant
  /// one-element-array wrapping, so by the time `transcribe`'s VAD branch
  /// has several chunk results to reconcile, folding them through
  /// [`merge_transcription_results`] right here is what keeps this
  /// method's own signature single-valued — pulling a call Swift leaves to
  /// its CLI down into the library, not adding behavior Swift lacks.
  ///
  /// When `options.chunking_strategy()` is
  /// [`ChunkingStrategy::Vad`](crate::options::ChunkingStrategy::Vad)
  /// **and** `audio.len()` exceeds the backend's
  /// [`ModelDims::window_samples`](crate::backend::ModelDims::window_samples)
  /// (:876-878): [`audio::vad::EnergyVad`] +
  /// [`chunker::VadChunker::chunk_all`] split `audio` along silence
  /// boundaries (:880-886 — Swift's injectable `voiceActivityDetector ??
  /// EnergyVAD()` field, :880, has no seam here yet: the energy VAD is
  /// the only detector this port ships, so it is constructed directly),
  /// with `clip_timestamps` cleared per chunk since chunking already
  /// consumed them (:892-894); each chunk runs its own
  /// [`TranscribeTask::run`], window-id-offset by its position in the
  /// chunk list (Swift's `audioIndex + batchIndex * batchSize`, :750 —
  /// which collapses to a flat running index here, see the deviation note
  /// below); a chunk whose task errored is dropped rather than failing the
  /// whole call, matching `updateSeekOffsetsForResults`'s `.failure`
  /// branch (`AudioChunker.swift:34-36` — Swift logs and skips, never
  /// rethrows); every surviving chunk's segments and
  /// [`TranscriptionResult::seek_time`] are re-anchored to the original
  /// timeline by [`chunker::apply_result_seek_offset`] before all chunk
  /// results are folded into one via [`merge_transcription_results`].
  /// Zero chunks (every clip shorter than the chunker's window padding —
  /// [`chunker::VadChunker::chunk_all`]'s own documented outcome), or
  /// every chunk erroring, therefore yields an `Ok` result with empty
  /// text/segments, exactly as Swift's own pipeline returns an empty
  /// (never-error) result array on the same inputs.
  ///
  /// Otherwise (:912-919): a single [`TranscribeTask::run`] over the whole
  /// input, unchunked.
  ///
  /// **Documented deviation — sequential chunk loop, not a thread pool:**
  /// Swift's VAD branch recurses into the exact same `transcribeWithOptions`
  /// worker-pooled batch machinery any multi-audio call goes through
  /// (`WhisperKit.swift:898-903`). This port deliberately does *not* call
  /// [`Self::transcribe_all`] here, even though the per-chunk unit of work
  /// (one [`TranscribeTask::run`] per chunk, with a distinct window-id
  /// offset) is identical: [`Self::transcribe_all`] additionally requires
  /// `B: Sync` (see its own doc), which
  /// [`crate::backend::coreml::CoreMlBackend`] does not satisfy, and this
  /// method must stay callable on `WhisperKit<CoreMlBackend>` — the only
  /// backend real (non-mock, non-streaming-e2e) callers ever transcribe
  /// through. Chunks are therefore processed one at a time in this
  /// method's own loop, needing only `B: InferenceBackend`. The
  /// observable per-chunk work, error handling, and merge step are
  /// otherwise identical either way; only the parallelism differs, and
  /// Swift's own concurrency model (concurrent tasks sharing one plain
  /// class instance through an `@unchecked Sendable` wrapper —
  /// `ConcurrencyUtilities.swift:131` — not actor isolation) has no
  /// `Sync`-shaped restriction forcing the same tradeoff there.
  ///
  /// # Errors
  /// [`TranscribeError::Audio`] if `options`' clip timestamps are
  /// malformed (from [`chunker::prepare_seek_clips`]). Otherwise as
  /// [`TranscribeTask::run`] — a single-window call's error propagates
  /// directly; a VAD-chunked call never fails from an individual chunk's
  /// error (dropped instead, see above).
  pub fn transcribe(
    &self,
    audio: &[f32],
    options: &DecodingOptions,
  ) -> Result<TranscriptionResult, TranscribeError> {
    let window_samples = self.backend.dims().window_samples();
    if options.chunking_strategy().is_vad() && audio.len() > window_samples {
      let vad = audio::vad::EnergyVad::new();
      let vad_chunker = chunker::VadChunker::new();
      let clip_ranges = chunker::prepare_seek_clips(options.clip_timestamps_slice(), audio.len())?;
      let chunks = vad_chunker.chunk_all(&vad, audio, window_samples, &clip_ranges);
      let chunk_options = options.clone().with_clip_timestamps(Vec::new());

      let mut chunk_results = Vec::with_capacity(chunks.len());
      for (chunk_index, chunk) in chunks.iter().enumerate() {
        let outcome = TranscribeTask::new(&self.backend, &self.tokenizer)
          .with_window_id_offset(chunk_index)
          .run(chunk.samples_slice(), &chunk_options);
        if let Ok(mut result) = outcome {
          chunker::apply_result_seek_offset(&mut result, chunk.seek_offset());
          chunk_results.push(result);
        }
      }
      return Ok(merge_transcription_results(&chunk_results));
    }

    TranscribeTask::new(&self.backend, &self.tokenizer).run(audio, options)
  }

  /// Runs one [`TranscribeTask`] per entry of `audios`, batched
  /// `options.concurrent_worker_count()` at a time and parallelized within
  /// each batch via [`std::thread::scope`]. Ports `transcribeWithOptions`
  /// (`WhisperKit.swift:716-812`), minus its per-audio
  /// `decodeOptionsArray` (every audio in this port shares one `options`)
  /// and `seekOffsets` (only [`Self::transcribe`]'s VAD branch needs
  /// seek-offset re-anchoring, and it does that itself after this call
  /// returns rather than threading an offset array through a worker
  /// callback the way Swift's `batchedSegmentCallback` does).
  ///
  /// Each worker's window ids are offset by its global index
  /// (`audio_index + batch_index * batch_size`, :750) via
  /// [`TranscribeTask::with_window_id_offset`], so
  /// [`crate::result::TranscriptionProgress::window_id`] stays distinct
  /// across concurrently-running workers. Results come back in input
  /// order: this function spawns each batch's workers in `audios`' order
  /// and joins the returned handles in that same order, which
  /// `std::thread::scope` preserves regardless of which worker finishes
  /// first — the structural equivalent of Swift's own post-hoc
  /// `batchResult.sort(by: { $0.index < $1.index })` (:801) over its
  /// unordered `TaskGroup` results.
  ///
  /// `B: Sync` is required because every worker in a batch borrows
  /// `&self.backend` from a different thread at once. See the module
  /// doc's "`WhisperKit`" section for why this makes
  /// `WhisperKit<CoreMlBackend>::transcribe_all` uncallable by design, and
  /// [`Self::transcribe`]'s doc for how its own VAD-chunked path avoids
  /// the same restriction by not calling this method.
  pub fn transcribe_all(
    &self,
    audios: &[&[f32]],
    options: &DecodingOptions,
  ) -> Vec<Result<TranscriptionResult, TranscribeError>>
  where
    B: Sync,
  {
    let batch_size = options.concurrent_worker_count().get();
    let mut results = Vec::with_capacity(audios.len());

    for (batch_index, batch) in audios.chunks(batch_size).enumerate() {
      let batch_results = std::thread::scope(|scope| {
        let handles: Vec<_> = batch
          .iter()
          .enumerate()
          .map(|(audio_index, &audio)| {
            let global_index = audio_index + batch_index * batch_size;
            scope.spawn(move || {
              TranscribeTask::new(&self.backend, &self.tokenizer)
                .with_window_id_offset(global_index)
                .run(audio, options)
            })
          })
          .collect();
        handles
          .into_iter()
          .map(|handle| handle.join().expect("transcribe worker thread panicked"))
          .collect::<Vec<_>>()
      });
      results.extend(batch_results);
    }

    results
  }

  /// One-shot language-detection probe over the first
  /// [`window_samples`](crate::backend::ModelDims::window_samples)-worth
  /// of `audio`. Ports `WhisperKit.detectLangauge(audioArray:)`
  /// (`WhisperKit.swift:534-581`, Swift's own misspelling — not repeated
  /// here).
  ///
  /// Pads or trims `audio` to one window ([`audio::pad_or_trim`],
  /// :554-558 — Swift's separate `detectLanguage(audioPath:)` overload
  /// clips to 30 s before this point instead, :525; this port takes
  /// samples directly, so only the pad/trim step applies here), extracts
  /// mel features, encodes, and runs [`decode::detect_language`] against a
  /// fresh decoder state (:550-563).
  ///
  /// # Errors
  /// [`TranscribeError::Model`] (as [`ModelError::InvalidState`] — see
  /// [`Self::new`]'s doc for why this is the variant this port reuses
  /// rather than inventing a new one) when
  /// `self.backend.dims().is_multilingual()` is `false` (:542-544 — Swift
  /// throws `WhisperError.decodingFailed("Language detection not
  /// supported for this model")` for a monolingual model). Otherwise as
  /// feature extraction, encoding, or [`decode::detect_language`] can
  /// fail.
  pub fn detect_language(&self, audio: &[f32]) -> Result<LanguageDetection, TranscribeError> {
    let dims = self.backend.dims();
    if !dims.is_multilingual() {
      return Err(
        ModelError::InvalidState {
          expected: "a multilingual model",
          actual: "a monolingual (English-only) model",
        }
        .into(),
      );
    }

    let padded = audio::pad_or_trim(audio, dims.window_samples());
    let features = self
      .backend
      .extract_features(&padded)
      .map_err(DecodeError::from)?;
    let encoder_output = self.backend.encode(&features).map_err(DecodeError::from)?;
    let mut state = self
      .backend
      .new_decoder_state()
      .map_err(DecodeError::from)?;
    let mut timings = TranscriptionTimings::new();

    // WhisperKit.swift:569-575 — the standalone path builds its own
    // zero-temperature sampler (argmax; never consults the RNG).
    let special = *self.tokenizer.special_tokens();
    let mut sampler =
      decode::sampler::GreedyTokenSampler::new(0.0, special.end_token(), &DecodingOptions::new());
    let probe = decode::detect_language(
      &self.backend,
      &encoder_output,
      &mut state,
      &self.tokenizer,
      &mut sampler,
      &mut timings,
    )?;

    Ok(LanguageDetection::new(
      probe.language(),
      probe.language_probs_slice().to_vec(),
    ))
  }
}
