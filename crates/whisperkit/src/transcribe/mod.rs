//! The transcription window loop: turns the decode stack into full-audio
//! transcription. Ports `TranscribeTask` (`TranscribeTask.swift:57-411`) —
//! [`TranscribeTask::run`] is the seek/window loop (:57-296) and its
//! private `decode_with_fallback` is the temperature-fallback ladder
//! (:316-411).
//!
//! **Deferred to Plan 4:** word-timestamp integration
//! (`options.word_timestamps()`, `TranscribeTask.swift:196-233`). The
//! alignment data already flows through
//! [`crate::backend::InferenceBackend::alignment_weights`] (Task 9) and the
//! pure math it needs ships in [`crate::segment`] (`find_alignment`,
//! `dynamic_time_warping`, `merge_punctuations`, Task 10), but
//! `addWordTimestamps`'s orchestration — re-anchoring word times against a
//! window's seek offset, clamping against `lastSpeechTimestamp`, writing the
//! results back onto segments, and the seek re-adjustment that follows
//! (:196-224) — is not called from this loop yet.
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

use std::{sync::atomic::AtomicBool, time::Instant};

use unicode_categories::UnicodeCategories;

use crate::{
  audio::{self, chunker},
  backend::InferenceBackend,
  constants::{DEFAULT_LANGUAGE_CODE, SAMPLE_RATE},
  decode::{self, TranscriptionProgressCallback, sampler::GreedyTokenSampler},
  error::{DecodeError, TranscribeError},
  options::DecodingOptions,
  result::{
    DecodingResult, TranscriptionProgress, TranscriptionResult, TranscriptionSegment,
    TranscriptionTimings, needs_fallback,
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
  pub const fn with_segment_callback(
    mut self,
    segment_callback: SegmentDiscoveryCallback<'ctx>,
  ) -> Self {
    self.set_segment_callback(segment_callback);
    self
  }
  /// Sets the callback fired with each window's segments as they're
  /// discovered.
  pub const fn set_segment_callback(
    &mut self,
    segment_callback: SegmentDiscoveryCallback<'ctx>,
  ) -> &mut Self {
    self.segment_callback = Some(segment_callback);
    self
  }

  /// Builder form of [`Self::set_progress_callback`].
  #[must_use]
  pub const fn with_progress_callback(
    mut self,
    progress_callback: TranscriptionProgressCallback<'ctx>,
  ) -> Self {
    self.set_progress_callback(progress_callback);
    self
  }
  /// Sets the callback fired with per-step decode progress.
  pub const fn set_progress_callback(
    &mut self,
    progress_callback: TranscriptionProgressCallback<'ctx>,
  ) -> &mut Self {
    self.progress_callback = Some(progress_callback);
    self
  }

  /// Builder form of [`Self::set_window_id_offset`].
  #[must_use]
  pub const fn with_window_id_offset(mut self, window_id_offset: usize) -> Self {
    self.set_window_id_offset(window_id_offset);
    self
  }
  /// Sets the chunk-worker id [`TranscriptionProgress::window_id`] updates
  /// are offset by — lets a caller running several [`TranscribeTask`]s
  /// concurrently over different audio chunks keep each worker's window ids
  /// distinct.
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
  /// window into segments fails; [`TranscribeError::Tokenizer`] if the final
  /// transcript decode fails.
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
        // malformed input (the resulting negative-length slice below would
        // itself crash) — this is a strictly safer, not looser, port.
        let segment_size = window_samples
          .min(content_frames.saturating_sub(seek))
          .min(seek_clip_end.saturating_sub(seek));

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

        // :156-173 — a fresh early-stop flag per window (matching this
        // port's own decode-loop test convention of a fresh `AtomicBool`
        // per `decode_text` call); windowId attribution happens inside
        // `decode_with_fallback`, which already receives `timings`.
        let early_stop = AtomicBool::new(false);
        let decoding_result = self.decode_with_fallback(
          &encoder_output,
          &mut state,
          &mut initial_prompt,
          &mut detected_language,
          options,
          &mut timings,
          &early_stop,
        )?;

        // :178-194.
        let windowing_start = Instant::now();
        let previous_seek = seek;
        let (new_seek, current_segments) = segment::find_seek_point_and_segments(
          &decoding_result,
          options,
          all_segments.len(),
          seek,
          segment_size,
          self.tokenizer,
        )?;
        seek = seek.max(new_seek);

        // Word-timestamp re-adjustment (:196-233) is deferred — see module
        // docs.

        // :236-239.
        if let Some(max_window_seek) = options.max_window_seek() {
          seek = seek.min(previous_seek + max_window_seek);
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
    early_stop: &AtomicBool,
  ) -> Result<DecodingResult, TranscribeError> {
    let special = *self.tokenizer.special_tokens();

    // :156-158 — windowId for progress attribution, computed once before
    // this window's attempts touch `timings`' window/fallback counters (Swift
    // computes this in `run`, immediately before calling this function; doing
    // it here instead is equivalent, since `timings` hasn't changed for this
    // window yet either way).
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
    for attempt in 0..=options.temperature_fallback_count() {
      let attempt_start = Instant::now();
      let temperature =
        options.temperature() + attempt as f32 * options.temperature_increment_on_fallback();
      let mut sampler = GreedyTokenSampler::new(temperature, special.end_token(), options);

      // :340-365 — for a multilingual model with no explicit language and
      // detection requested, probe the language once, patch a per-attempt
      // options clone, and re-derive the prefill prompt from it.
      let mut window_options = options.clone();
      if self.backend.dims().is_multilingual()
        && options.language().is_empty()
        && options.detect_language()
      {
        if let Ok(probe) =
          decode::detect_language(self.backend, encoder_output, state, self.tokenizer, timings)
        {
          window_options.set_language(probe.language().to_string());
          *detected_language = Some(probe.language().to_string());
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
        early_stop,
        window_callback,
      )?;

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
    Ok(decoding.expect("the loop runs at least once (0..=count always yields >= 1 attempt)"))
  }
}
