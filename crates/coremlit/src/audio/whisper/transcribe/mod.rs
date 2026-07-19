//! The transcription window loop: turns the decode stack into full-audio
//! transcription. Ports `TranscribeTask` (`TranscribeTask.swift:57-411`) —
//! [`TranscribeTask::run`] is the seek/window loop (:57-296), including the
//! optional word-timestamp re-anchoring step
//! (`options.word_timestamps()`, :196-233) that runs
//! [`crate::audio::whisper::segment::add_word_timestamps`] against the accepted attempt's
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
//! [`crate::audio::whisper::backend::InferenceBackend::reset_decoder_state`] takes no such
//! parameter: the backend's decoder state is sized once at allocation
//! ([`crate::audio::whisper::backend::InferenceBackend::new_decoder_state`]), and reset
//! always restores that same fixed allocation's initial state, so there is
//! nothing left for a per-call bound to vary.
//!
//! **Not ported:** Swift's `Progress`/`Logging.beginSignpost`
//! instrumentation (:62-63, 101-103, 110, 276-277, 282) and its
//! `Task.checkCancellation()` cooperative-cancellation checks (:135, 144,
//! 165) have no equivalent here — this is a sync library with no `Progress`
//! type and no structured-concurrency task tree to cancel.
//! [`crate::audio::whisper::decode::TranscriptionProgressCallback`]'s `Some(false)`-triggered
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
//! [`crate::audio::whisper::options::Options`] and detects the model variant the same way
//! `loadTokenizerIfNeeded` does (`WhisperKit.swift:444-457`);
//! [`WhisperKit::with_backend`] is the mock/e2e-hermetic construction seam
//! every test above this layer uses instead.
//!
//! **Concurrency note — why `transcribe_all` has no `CoreMlBackend`
//! instance:** [`WhisperKit::transcribe_all`] batches workers over
//! `std::thread::scope`, which requires `B: Sync` (every worker holds a
//! `&self.backend` shared across threads at once). `crate::Model` is
//! documented `Send` but deliberately **not** `Sync` — Apple's contract is
//! "use an `MLModel` instance on one thread ... at a time" — so
//! [`crate::audio::whisper::backend::coreml::CoreMlBackend`], which owns three `Model`s,
//! is not `Sync` either, and `WhisperKit<CoreMlBackend>::transcribe_all`
//! is simply never a callable method (there is no `impl` providing it for
//! that `B`). This is by design, not a gap: real concurrent batch
//! transcription over CoreML needs one backend — hence one `WhisperKit` —
//! per worker thread, each driven independently, not one shared backend
//! serving many. [`crate::audio::whisper::backend::mock::MockBackend`] IS `Sync` (its
//! mutable state lives behind `Arc<Mutex<_>>`), so hermetic batch tests
//! exercise the real code path. [`WhisperKit::transcribe`], by contrast,
//! never needs `B: Sync`: its VAD-chunked branch runs each chunk's
//! [`TranscribeTask`] sequentially rather than reusing
//! [`WhisperKit::transcribe_all`]'s worker pool, specifically so it stays
//! callable on `WhisperKit<CoreMlBackend>` — see that method's own doc for
//! the full reasoning.

use std::{
  cell::Cell,
  sync::{
    Mutex, PoisonError,
    atomic::{AtomicBool, Ordering},
  },
  time::{Duration, Instant},
};

use unicode_categories::UnicodeCategories;

use crate::audio::whisper::{
  audio::{self, chunker},
  backend::{AlignmentMatrix, InferenceBackend, coreml::CoreMlBackend},
  constants::{
    APPEND_PUNCTUATION, BLANK_AUDIO_MARKER, DEFAULT_LANGUAGE_CODE, PREPEND_PUNCTUATION, SAMPLE_RATE,
  },
  decode::{
    self, TranscriptionProgressCallback,
    sampler::{self, GreedyTokenSampler},
  },
  error::{DecodeError, ModelError, TranscribeError, VadError},
  model::{
    ModelVariant, detect_variant,
    manager::{ModelLoadTimings, ModelManager},
  },
  options::{DecodingOptions, Options},
  result::{
    DecodingResult, TranscriptionProgress, TranscriptionResult, TranscriptionSegment,
    TranscriptionTimings, merge_transcription_results_with_options, needs_fallback,
  },
  segment,
  stream::{AudioStreamTranscriber, agreement::LocalAgreementTranscriber},
  task_facts::{SpanKnowledge, TaskFacts},
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
/// Duplicated from [`crate::audio::whisper::segment`]'s private helper of the same name and
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
///
/// **Observes segments BEFORE the [`DecodingOptions::drop_blank_audio`]
/// filter** (codex round 3, adjudicated). This is a real-time, raw-segment
/// surface: it fires the instant a window finishes decoding, inside the window
/// loop, so a window that decoded to nothing but `[BLANK_AUDIO]` is reported
/// here even when `drop_blank_audio` is set and that segment is later dropped
/// from the final [`TranscriptionResult`]. The drop governs the assembled
/// result; this callback governs what was actually decoded. A consumer that
/// wants only the surviving segments must filter its own copy — the drop runs
/// once every window has decoded (see [`TranscribeTask::run`]), necessarily
/// after this has already fired per-window.
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
  /// Optional caller-owned [`TaskFacts`] sink [`Self::run`] merges each
  /// attempt's error-fragile facts — the RNG draw, the early-stop truncation,
  /// and the genuine language observation — into, the instant an attempt settles
  /// inside `decode_with_fallback` and *before* any error can propagate out. It
  /// outlives the task, so an errored [`Self::run`] whose whole result the caller
  /// discards still leaves those facts behind: [`WhisperKit::transcribe`]'s VAD
  /// branch installs one shared sink across its chunks so a chunk that sampled,
  /// truncated, or detected a language then errored and was DROPPED still
  /// contributes those facts to the merged transcript (coremlit issue #14, codex
  /// rounds 4–6 — the early-stop fact is the round-6 R6-F1 addition the two
  /// former per-fact sinks lacked). `None` on the non-VAD/`transcribe_all` paths,
  /// which surface a chunk's error directly rather than dropping it, so `run`
  /// accumulates into a task-local sink instead. The worker schedule and id span
  /// are set on the result directly, never through this sink.
  facts_sink: Option<&'ctx Mutex<TaskFacts>>,
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
      facts_sink: None,
    }
  }

  /// Installs a caller-owned [`TaskFacts`] sink that [`Self::run`] merges this
  /// task's error-fragile facts (RNG draw, early-stop truncation, language
  /// observation) into before any error propagates — see the field's own doc.
  /// `pub(crate)`: only [`WhisperKit::transcribe`]'s VAD branch needs it, to
  /// keep a dropped-because-errored chunk's facts from vanishing from the merged
  /// transcript.
  #[must_use]
  #[inline(always)]
  pub(crate) const fn with_facts_sink(mut self, sink: &'ctx Mutex<TaskFacts>) -> Self {
    self.facts_sink = Some(sink);
    self
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
  /// distinct. It doubles as the `worker_index` coordinate of the seeded
  /// fallback ladder's sub-seed derivation (see
  /// [`sampler::derive_attempt_seed`]), which is what keeps distinct
  /// chunks/workers on distinct RNG streams even where their task-local
  /// window indices coincide.
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
  /// [`crate::audio::whisper::backend::ModelDims::window_samples`]-sized (or shorter, for
  /// the final partial window) windows until fewer than
  /// `options.window_clip_time()` seconds of the clip remain, feeding each
  /// window's decode through the private temperature-fallback ladder and
  /// then [`crate::audio::whisper::segment::find_seek_point_and_segments`] to turn it into
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
    // Monotonic count of segment-id ordinals this run has consumed so far — the
    // base each window's `find_seek_point_and_segments` ids its segments off. How
    // it advances per window depends on [`DecodingOptions::drop_blank_audio`]
    // (see the F1 advance in the loop below):
    //
    // - `true` (the default) advances by ALL allocated ordinals, BEFORE the
    //   word-timestamp zero-length filter removes any — a survivor-count base
    //   would reissue an id a survivor of an earlier window still carries, making
    //   the pipeline-local ids non-unique/non-monotonic (coremlit issue #14,
    //   codex round 5). This is the deliberate unique-id hardening.
    // - `false` advances by the SURVIVING count Swift appends
    //   (`allSegmentsCount: allSegments.count`, `TranscribeTask.swift:181`), so
    //   the ids DUPLICATE across a removed segment exactly as Swift's do — the
    //   exact-parity contract of clearing the option (F1, codex round 9).
    let mut decoded_segment_span = 0usize;
    // The ordinal-ALLOCATION fact carried onto the result as its `decoded_span`,
    // kept SEPARATE from the id base above (codex round 13, M3). It advances by
    // every window's PRE-FILTER allocation count regardless of the option, so it
    // counts ordinals ALLOCATED — dropped segments included, the public definition
    // of `SpanKnowledge::Exact` — where the `false`-path id base counts only
    // SURVIVORS. On the `true` path the two coincide (both advance by the
    // allocation); on the `false` path they diverge, and recording the id base as
    // the span there fabricated an `Exact` that was neither the allocation count
    // nor the survivor extent (it under-counted the ordinals the run consumed).
    let mut allocated_ordinals = 0usize;
    let mut detected_language: Option<String> = None;
    // The GENUINE observation, kept SEPARATE from `detected_language` above:
    // that one is the Swift-faithful DISPLAY language and includes a
    // configured or fallback (`"en"`) value, whereas this is `Some` only when
    // a probe ran or a `<|lang|>` token was actually decoded. This is what the
    // result's `TaskFacts::observed_language` records — a configured or
    // defaulted language was never *detected* (F3, codex round 3). It is THIS
    // task's own observation; the shared sink below carries it cross-chunk.
    let mut observed_language: Option<String> = None;
    // The ONE sink for this task's error-fragile facts — the RNG draw (from the
    // sampler's own `drew_from_rng`, never inferred from a temperature), the
    // early-stop truncation (OR-ed across every attempt including a REJECTED one
    // whose stop changed which attempt was selected — R6-F1, coremlit issue #14
    // codex round 6), and the language observation. `decode_with_fallback` merges
    // each attempt's facts in the instant it settles, BEFORE any error can
    // propagate. When the caller installed a sink (the VAD branch), that shared
    // sink is used so a chunk that drew/stopped/observed then ERRORED and is
    // dropped still records the facts; otherwise a task-local sink, discarded
    // with the error the caller surfaces anyway.
    //
    // Seeded **observed-clean** (`Some(false)` for the draw and early-stop), not
    // `unknown()`: this run IS watching, so before any window decodes it has
    // POSITIVELY seen no draw and no truncation. A window that draws or a
    // callback that truncates flips the fact to `Some(true)` under the Kleene OR
    // (for which `Some(false)` is the identity, so a greedy attempt's `Some(false)`
    // leaves the sink `Some(false)` rather than nulling it). A run that decodes NO
    // window at all (audio shorter than one window) therefore finalizes the honest
    // `Some(false)`/`Some(false)` and is reproducible, where `unknown()` would have
    // left it conservatively non-reproducible (codex round 8, F3).
    let local_facts = Mutex::new(TaskFacts::observed_clean());
    let facts_sink: &Mutex<TaskFacts> = self.facts_sink.unwrap_or(&local_facts);

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

    // Strictly monotonic, 0-based count of windows this `run` call has
    // attempted so far (incremented once per `decode_with_fallback` call,
    // below) — the seed-derivation input `sampler::derive_attempt_seed`'s
    // doc calls `window_index`. Deliberately NOT the `window_id` computed
    // inside `decode_with_fallback` for progress-callback attribution:
    // that value is derived from `timings` counters a prior window's
    // fallback can leave in a stale state (see its own doc), so it is not
    // guaranteed distinct across windows the way seed derivation needs.
    let mut window_index: u64 = 0;

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
          &mut observed_language,
          facts_sink,
          options,
          &mut timings,
          window_index,
        )?;
        window_index += 1;

        // THE reproducibility invariant, now unified. The RNG-draw AND early-stop
        // facts were merged into `facts_sink` INSIDE `decode_with_fallback` just
        // now, per attempt, from the sampler's own `GreedyTokenSampler::drew_from_rng`
        // and the per-attempt early-stop latch — the honest facts of whether an
        // RNG draw or a callback truncation actually happened, NOT inferred here
        // from `decoding_result` (its temperature, or its accepted-attempt-only
        // `early_stopped`). The draw inference was wrong twice over (F2, codex
        // round 3) and the accepted-only early-stop read lost a REJECTED attempt's
        // truncation that changed which attempt was selected (R6-F1, codex round
        // 6). Merging both into the ladder also keeps them ahead of every step
        // that can erase the evidence — the word-timestamp zero-length filter, the
        // no-speech `continue`, the blank-audio drop, or (on the VAD path) a chunk
        // that contributed nothing to the merge OR whose whole `run` errored and
        // was dropped (the facts are captured before that error can propagate,
        // into the sink the caller owns) — so a `Provenance` can never read the
        // effective temperature back off the SURVIVING segments and declare the
        // transcript reproducible. (Constructed, not hypothesized: `transcribe::tests`'
        // `unseeded_sampling_survives_the_blank_audio_drop` scripts a window
        // accepted at 0.2 that decodes to exactly `[BLANK_AUDIO]` and is then
        // dropped, and `unseeded_draw_survives_an_errored_vad_chunk_drop` a
        // chunk that draws then errors.)

        // :178-194.
        let windowing_start = Instant::now();
        let previous_seek = seek;
        let (new_seek, mut current_segments) = segment::find_seek_point_and_segments(
          &decoding_result,
          options,
          decoded_segment_span,
          seek,
          segment_size,
          self.tokenizer,
        )?;
        // The count this window ALLOCATED, captured before the word-timestamp
        // zero-length filter below can remove any — the `drop_blank_audio == true`
        // advance (see the F1 advance after that filter, and the
        // `decoded_segment_span` declaration). A silence-skipped window allocates
        // nothing (`None`).
        let allocated_this_window = current_segments.as_ref().map_or(0, Vec::len);
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
            options.word_grouping(), // coremlit issue #14; default: fine-grained
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
          // (`crate::audio::whisper::decode::decode_text`'s own faithfully-ported upstream
          // TODO), so no positive `no_speech_threshold` — including the
          // default — can ever reach it.
          current_segments = Some(filtered);
        }

        // F1 (codex round 9): advance the decoded-ordinal base for the NEXT
        // window's ids, now that the word-timestamp zero-length filter above has
        // run. The two paths differ only when that filter removed something:
        //
        // - `drop_blank_audio == true` (the default) advances by ALL allocated
        //   ordinals — the deliberate unique-id hardening that keeps every
        //   survivor id unique and monotonic across a removed segment
        //   ([0, 2, 3, 5] rather than a duplicate).
        // - `drop_blank_audio == false` advances by the SURVIVOR count Swift
        //   appends — `findSeekPointAndSegments(allSegmentsCount:)` reads the
        //   running SURVIVOR total (`TranscribeTask.swift:181`), which filters
        //   zero-length (`:217`) and appends only survivors (`:262`), so its ids
        //   DUPLICATE across a removed segment ([0, 2, 2, 4]) — exact Swift
        //   parity, the whole contract of clearing the option.
        //
        // Without word timestamps the filter never runs, so survivors ==
        // allocated and the two coincide. A silence-skipped window (`None`)
        // allocated nothing and advances by zero either way.
        decoded_segment_span = decoded_segment_span.saturating_add(if options.drop_blank_audio() {
          allocated_this_window
        } else {
          current_segments.as_ref().map_or(0, Vec::len)
        });
        // The allocation FACT advances by the PRE-FILTER count on BOTH paths
        // (codex round 13, M3): it records ordinals ALLOCATED, not survivors, so
        // the `false`-path survivor-counting id base above no longer under-reports
        // the stored `decoded_span`. On the `true` path this equals the advance
        // above; on the `false` path it exceeds it whenever the filter dropped a
        // segment.
        allocated_ordinals = allocated_ordinals.saturating_add(allocated_this_window);

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

    let special_token_begin = self.tokenizer.special_tokens().special_token_begin();

    // coremlit issue #14 — the blank-audio drop, as a POST-decode filter
    // over the fully-assembled segments and nothing more: it runs after
    // every window has decoded, so it cannot perturb decoding itself, and
    // it is skipped outright when the caller opts out. Sequencing it
    // BEFORE the text assembly below is what makes a dropped segment's
    // tokens vanish from `TranscriptionResult::text` too, rather than
    // leaving the marker stranded in the aggregate text with no segment
    // behind it — pure silence collapses to a genuinely empty result.
    // Speech decodes no blank segment, so this is a no-op on the golden
    // parity inputs under either setting (see
    // `DecodingOptions::drop_blank_audio`).
    if options.drop_blank_audio() {
      self.drop_blank_audio_segments(&mut all_segments, special_token_begin)?;
    }

    // :298-305 — every segment's tokens, filtered to non-special ids,
    // decoded, and trimmed. `all_tokens` isn't tracked as its own running
    // accumulator (unlike Swift's `allTokens`): since nothing here diverges
    // `allSegments` from the tokens their own segments carry (no
    // `windowPostProcess` hook to make them differ), deriving the token
    // list from `all_segments` once at the end is equivalent and avoids a
    // redundant parallel Vec.
    let word_tokens: Vec<u32> = all_segments
      .iter()
      .flat_map(|segment| segment.tokens_slice().iter().copied())
      .filter(|&token| token < special_token_begin)
      .collect();
    let text = self.tokenizer.decode(&word_tokens, false)?;
    let trimmed_text = trim_swift_whitespaces(&text);

    timings.set_full_pipeline(pipeline_start.elapsed().as_secs_f64());

    Ok(
      TranscriptionResult::new(
        trimmed_text,
        all_segments,
        // Swift-compat DISPLAY fallback: `"en"` when the run detected
        // nothing. Kept verbatim on `TranscriptionResult::language`.
        detected_language
          .clone()
          .unwrap_or_else(|| DEFAULT_LANGUAGE_CODE.to_string()),
        timings,
      )
      // The decode-time facts this run controlled, assembled into the ONE
      // carried record (coremlit issue #14, codex round 6):
      //
      // - the RNG-draw and early-stop facts come from `facts_sink`, accumulated
      //   across every attempt (rejected included) BEFORE any filter or error
      //   could erase them — NOT derived from `all_segments`, which by this point
      //   may have lost the very window that sampled or was truncated;
      // - the genuine language observation is THIS task's own (the shared sink
      //   carries it cross-chunk for the VAD merge instead), `None` for a
      //   configured/fallback language or a zero-window run that witnessed
      //   nothing — `Provenance::for_result` reads THIS, never the display `"en"`;
      // - the worker coordinate is a single-run schedule `[offset]` (an explicit
      //   KNOWN coordinate), and the decode's own id-ordinal span (dropped
      //   segments included) is what the merge advances its running base by.
      .with_task_facts({
        let facts = facts_sink.lock().unwrap_or_else(PoisonError::into_inner);
        // Carry the sink's ACCUMULATED draw/early-stop facts verbatim — each an
        // explicit `Some`/`None`, never collapsed to a bool (F1, codex round 6
        // post-consolidation) — and layer THIS task's own observation, worker
        // coordinate, and id span on top. The per-attempt merge only ever sets
        // draw/early-stop/observation on the sink, so its worker schedule and
        // span are still `None`; the `with_*` calls below define them, and
        // `with_observed_language` overrides the sink's cross-chunk observation
        // with this task's own (the sink carries it cross-chunk for the VAD merge).
        facts
          .clone()
          .with_observed_language(observed_language)
          .with_worker(self.window_id_offset)
          // A real decode task KNOWS the exact ordinal count it allocated
          // (dropped segments included — the pre-filter allocation total, NOT the
          // `false`-path survivor-counting id base; codex round 13, M3).
          .with_decoded_span(SpanKnowledge::Exact(allocated_ordinals))
      }),
    )
  }

  /// Removes every blank-audio segment from `segments` in place, for
  /// [`DecodingOptions::drop_blank_audio`] (coremlit issue #14). A segment
  /// is blank-audio when its CLEAN text — its own tokens with the
  /// special/timestamp ids (`>= special_token_begin`) stripped, decoded,
  /// and Swift-whitespace-trimmed — is exactly
  /// [`BLANK_AUDIO_MARKER`]. That projection is deliberately the
  /// per-segment analogue of the aggregate text [`Self::run`] assembles
  /// from the survivors, which is why the match is against the *clean*
  /// text and not [`TranscriptionSegment::text`]: the latter still carries
  /// its `<|startoftranscript|>`/timestamp tokens under the default
  /// `skip_special_tokens == false`, so equality against the bare marker
  /// would never hold there (see [`BLANK_AUDIO_MARKER`]'s own doc). It is
  /// computed only to decide the drop and never reused to build the result
  /// text — that stays the single aggregate decode of the surviving
  /// segments' concatenated tokens, so per-segment isolation cannot change
  /// the transcript's spacing.
  ///
  /// **Only the exact [`BLANK_AUDIO_MARKER`] literal is dropped.** The
  /// other non-speech markers a Whisper model samples — `[APPLAUSE]`,
  /// `[MUSIC]`, and friends; `[APPLAUSE]` shows up in this crate's own jfk
  /// fixture decode — are left in the transcript, by design. This is a
  /// blank-*audio* filter, not a general non-speech-annotation stripper:
  /// silence carries no information a consumer could want, whereas
  /// `[APPLAUSE]` is a genuine (if non-lexical) event, and deciding which
  /// of those a product wants to keep is not this crate's call to make.
  ///
  /// **Survivors keep the ids they were decoded with, gaps and all** — a
  /// drop REMOVES, it does not relabel. A segment id is
  /// `all_segments_count + segments.len()`
  /// ([`segment::find_seek_point_and_segments`]), i.e. an *ordinal decode
  /// position*, not an index into this vec, and nothing in the crate looks
  /// a segment up by it. Renumbering the survivors to a dense `0..N` would
  /// only make the two settings harder to compare: id 1 would mean "the
  /// blank" under `drop_blank_audio == false` and "the second speech
  /// segment" under `true`, so a consumer diffing the two runs could not
  /// correlate them. Left alone, the ids are stable across the toggle and
  /// the hole is self-describing ("segment 1 was dropped"). There is also
  /// no id contiguity to "restore" in the first place:
  /// [`merge_transcription_results`](crate::audio::whisper::result::merge_transcription_results)
  /// re-ids every segment it merges to
  /// `result_index + segment_index` — a faithfully-ported upstream quirk
  /// that is neither dense nor unique (two results of two segments give
  /// `[0, 1, 1, 2]`) — and it does so unconditionally, so on the VAD path
  /// any renumbering here would be overwritten anyway.
  ///
  /// When nothing is blank (every speech input, the golden parity clips
  /// included) this returns having touched nothing at all.
  ///
  /// # Errors
  /// [`TranscribeError::Tokenizer`] if a segment's tokens fail to decode.
  fn drop_blank_audio_segments(
    &self,
    segments: &mut Vec<TranscriptionSegment>,
    special_token_begin: u32,
  ) -> Result<(), TranscribeError> {
    let mut blank = Vec::with_capacity(segments.len());
    for segment in segments.iter() {
      let clean_tokens: Vec<u32> = segment
        .tokens_slice()
        .iter()
        .copied()
        .filter(|&token| token < special_token_begin)
        .collect();
      let clean_text = self.tokenizer.decode(&clean_tokens, false)?;
      blank.push(trim_swift_whitespaces(&clean_text) == BLANK_AUDIO_MARKER);
    }

    if !blank.contains(&true) {
      return Ok(());
    }

    let survivors: Vec<TranscriptionSegment> = segments
      .drain(..)
      .zip(blank)
      .filter_map(|(segment, is_blank)| (!is_blank).then_some(segment))
      .collect();
    *segments = survivors;
    Ok(())
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
  ///
  /// **Rust-only addition, no Swift equivalent:** each attempt's sampler is
  /// seeded from `options.seed()` when set, via
  /// [`sampler::derive_attempt_seed`] over three domain-separated
  /// coordinates — `self.window_id_offset` (this task's per-chunk/worker
  /// id), `window_index` (this call's own position in [`Self::run`]'s
  /// strictly monotonic per-`run` window counter, local to the task), and
  /// the loop's own `attempt` index — passed distinctly, never summed; see
  /// that function's doc for the full mixing contract. `options.seed()`
  /// unset takes the exact same [`GreedyTokenSampler::new`] OS-seeded path
  /// as before this knob existed: the default is byte-unchanged.
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
    observed_language: &mut Option<String>,
    facts_sink: &Mutex<TaskFacts>,
    options: &DecodingOptions,
    timings: &mut TranscriptionTimings,
    window_index: u64,
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
      // `options.seed()` unset (the default): no-op, leaving `sampler`'s
      // OS-seeded RNG exactly as `GreedyTokenSampler::new` just built it —
      // byte-identical to the pre-seed-knob behavior. Set: reseed with this
      // (worker, window, attempt) triple's own derived sub-seed, so the
      // whole transcription reproduces bit-for-bit across runs while every
      // window/attempt still draws an independent stream (see
      // `sampler::derive_attempt_seed`'s doc for the full contract).
      if let Some(seed) = options.seed() {
        sampler = sampler.with_seed(sampler::derive_attempt_seed(
          seed,
          self.window_id_offset as u64,
          window_index,
          attempt as u64,
        ));
      }
      // A FRESH early-stop latch per attempt: Swift initializes a new
      // early-stop entry for every decodeText invocation
      // (TextDecoder.swift:570), so a callback-stopped attempt whose
      // partial result triggers an ordinary fallback must not truncate
      // the retry (phase-gate round-5 finding).
      let early_stop = AtomicBool::new(false);
      // A FRESH per-attempt cell for the predicted-language OBSERVATION,
      // recognized at token-sampling time inside `decode_text` and set BEFORE any
      // later fallible step — exactly like `early_stop` and the sampler's own
      // `drew_from_rng`, so a decode that recognizes `<|lang|>` then errors on a
      // LATER step still surfaces the detection into the sink below (F2, codex
      // round 6 post-consolidation).
      let observed_language_token: Cell<Option<u32>> = Cell::new(None);
      // Whether THIS attempt's automatic-language probe failed and was SWALLOWED
      // (Swift's `try?` → nil). A failed probe silently alters the prompt/language
      // this attempt decodes from — the same class of hidden, transcript-controlling
      // error as a dropped VAD chunk — so it is recorded into the sink's
      // `had_swallowed_error` below, forcing the run non-reproducible (codex round
      // 11, M2). Stays `false` when no probe runs or the probe succeeds.
      let mut language_probe_swallowed = false;

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
            // A probe that actually SAMPLED a language token IS a genuine
            // detection (the observation), even when the sampled code fell back
            // to the display default. But an all-masked probe -- a valid
            // tokenizer with no `<|lang|>` entries makes `LanguageLogitsFilter`
            // mask everything, so the sampler returns the degenerate token 0 at
            // `-inf` and the probe's `language_probs` stays EMPTY -- sampled
            // NOTHING; promoting its `"en"` DISPLAY default fabricates an
            // observation the model never made, which first-wins merging then
            // lets SUPPRESS a genuine later `"es"` (F3, codex round 14). Gate on
            // the probe having a predicted-language entry, exactly the invariant
            // `decode::decode_text` already holds for a decode's own prediction
            // (`decode/tests.rs`). FIRST genuine observation wins (the round-3
            // merge rule), so an earlier window's or attempt's observation is
            // never overwritten by a later probe (F2, codex round 4).
            if observed_language.is_none() && !probe.language_probs_slice().is_empty() {
              *observed_language = Some(probe.language().to_string());
            }
          }
          Err(_) => {
            // DISPLAY: Swift's `try?` yields nil — last-write-wins.
            *detected_language = None;
            // OBSERVATION: a FAILED probe witnessed nothing, so it must NOT
            // erase an earlier genuine observation (F2, codex round 4). Left
            // untouched; the post-decode promotion below still records THIS
            // attempt's own observation if its decode predicts a `<|lang|>`
            // token.
            // SWALLOWED ERROR: the probe's failure was hidden, yet it changed the
            // prompt/language this attempt (and a retry) decodes from — a
            // transcript-controlling error with no outcome fact until now. Mark it
            // for the sink so `is_reproducible` reflects the hidden failure (codex
            // round 11, M2).
            language_probe_swallowed = true;
          }
        }
        if options.use_prefill_prompt() {
          *initial_prompt = decode::prefill_tokens(&window_options, self.tokenizer, true);
        }
      }

      // Start this attempt's swallowed-compression-error window clean. Every
      // compression that controls this attempt's transcript/fallback — the
      // decode's finalize and progress ratios, and any streaming early-stop
      // callback's window ratio — funnels through `text::zlib_compressed_len` on
      // this thread between here and the fact merge below, latching the flag on
      // an erased OS error. Reading it there records `had_swallowed_error`
      // honestly, so a swallowed error's `+inf` ratio that drove a fallback (a
      // different transcript) can no longer read back as reproducible (coremlit
      // issue #14, codex round 14).
      crate::audio::whisper::text::clear_compression_error_swallowed();
      let outcome = decode::decode_text(
        self.backend,
        encoder_output,
        state,
        initial_prompt.as_slice(),
        &mut sampler,
        &window_options,
        self.tokenizer,
        timings,
        &early_stop,
        &observed_language_token,
        window_callback,
      );

      // Merge THIS attempt's error-fragile facts into `facts_sink` BEFORE
      // propagating any error (coremlit issue #14, codex rounds 4–6). Captured
      // ahead of the `?` below so a VAD chunk that drew, was truncated, or
      // observed a language then errored on a LATER step and was DROPPED still
      // contributes those facts to the merged transcript. OR-ed across every
      // attempt — REJECTED as well as retained — because a rejected attempt's
      // unseeded draw still decides which attempt is kept (F2, codex round 3),
      // and a rejected attempt's early stop still changed which attempt the
      // ladder selected (R6-F1, codex round 6): reading only the accepted
      // attempt's `DecodingResult::early_stopped` lost exactly that history.
      // One `sampler` owns both the language probe's draw and every text token's,
      // so its single `drew_from_rng` covers the whole attempt; a zero-iteration
      // decode at a non-zero temperature never draws and correctly leaves it
      // unset. `early_stop` is THIS attempt's own fresh latch (see its reset).
      //
      // The observation is first-wins (the merge law's rule) — a later attempt
      // cannot overwrite an earlier genuine detection — and lives in the sink so a
      // dropped chunk's detection still reaches the merged transcript. It is
      // recovered here from `observed_language_token`, the cell `decode_text`
      // recognized the predicted `<|lang|>` into at sampling time: a chunk that
      // predicts `<|es|>` (no probe) then errors on a LATER step used to lose it,
      // because its string was only built at successful finalization (F2, codex
      // round 6 post-consolidation). The task-level `observed_language` (a probe's
      // detection, or an earlier window's) takes precedence, preserving first-wins.
      let attempt_observation = observed_language.clone().or_else(|| {
        observed_language_token.get().and_then(|token| {
          self.tokenizer.decode(&[token], false).ok().map(|decoded| {
            crate::audio::whisper::text::trim_special_token_chars(&decoded).to_string()
          })
        })
      });
      facts_sink
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .merge(
          &TaskFacts::unknown()
            .with_drew_from_rng(sampler.drew_from_rng())
            .with_early_stopped(early_stop.load(Ordering::Relaxed))
            .with_had_swallowed_error(
              language_probe_swallowed
                || crate::audio::whisper::text::take_compression_error_swallowed(),
            )
            .with_observed_language(attempt_observation),
        );
      let result = outcome?;

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

      // :375-378 — the DISPLAY language: promote the decode's language when no
      // probe set it. Kept for ALL branches (configured / decoded token /
      // fallback) so the display stays Swift-faithful.
      if detected_language.is_none() {
        *detected_language = Some(result.language().to_string());
      }
      // The GENUINE observation: the language the decode actually PREDICTED
      // (`DecodingResult::observed_language`), promoted ONLY when it predicted a
      // `<|lang|>` token — never the forced prefill `<|en|>` nor a
      // configured/`"en"`-fallback (F2, codex round 4). Read the predicted CODE,
      // NOT the display `language`: a forced `<|en|>` prefill is the display even
      // when the model predicted a different language after it, so promoting the
      // display would record `"en"` for a run that detected `"es"` (F1, codex
      // round 5). FIRST genuine observation wins, so a later window/attempt
      // cannot overwrite an earlier detection and a failed probe's cleared
      // display does not drag the observation down with it.
      if observed_language.is_none()
        && let Some(predicted) = result.observed_language()
      {
        *observed_language = Some(predicted.to_string());
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
          // Records the ZERO-BASED `attempt` index of the latest fallback:
          // the first fallback writes 0.0, which is also this counter's init
          // value, so it alone cannot tell "never fell back" from "one
          // fallback at attempt 0" (a true fallback count would be
          // `attempt + 1`). Kept as-is: the field is public and summed
          // across merged results, so re-defining it is out of scope here.
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
// LoadTimings
// ---------------------------------------------------------------------

/// The one-time model/tokenizer load durations [`WhisperKit::<CoreMlBackend>::new`]
/// measures at construction and [`WhisperKit`] then stamps into every run's
/// fresh [`TranscriptionTimings`] — the port's stand-in for Swift's
/// persistent `currentTimings`, whose load fields (populated in
/// `loadModels`) ride into each result the same way
/// (`WhisperKit.swift:396-441`, plumbed through `setupTranscribeTask`).
/// [`Default`] (all [`Duration::ZERO`]) is the honest value for a
/// [`WhisperKit::with_backend`] pipeline, which loads no models to time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LoadTimings {
  /// Whole-construction load total: prewarm pass + real model load + tokenizer
  /// (Swift's `modelLoading = load-pass + prewarmLoadTime`,
  /// `WhisperKit.swift:439`).
  model_loading: Duration,
  /// The prewarm pass as a whole; [`Duration::ZERO`] when prewarm was off.
  prewarm_load_time: Duration,
  /// Real-load-pass encoder load (`ModelLoadTimings::encoder_load`).
  encoder_load: Duration,
  /// Real-load-pass decoder load (`ModelLoadTimings::decoder_load`).
  decoder_load: Duration,
  /// Prewarm-pass encoder specialization; [`Duration::ZERO`] without prewarm.
  encoder_specialization: Duration,
  /// Prewarm-pass decoder specialization; [`Duration::ZERO`] without prewarm.
  decoder_specialization: Duration,
  /// Tokenizer load.
  tokenizer_load_time: Duration,
}

impl LoadTimings {
  /// Writes all seven load durations (as fractional seconds, the unit
  /// [`TranscriptionTimings`] stores) into `timings`. The load fields are
  /// never touched by [`TranscribeTask::run`], so stamping them onto a
  /// run's result overwrites only their zero defaults, leaving every
  /// compute-stage timing the run accumulated intact.
  fn stamp(&self, timings: &mut TranscriptionTimings) {
    timings
      .set_model_loading(self.model_loading.as_secs_f64())
      .set_prewarm_load_time(self.prewarm_load_time.as_secs_f64())
      .set_encoder_load_time(self.encoder_load.as_secs_f64())
      .set_decoder_load_time(self.decoder_load.as_secs_f64())
      .set_encoder_specialization_time(self.encoder_specialization.as_secs_f64())
      .set_decoder_specialization_time(self.decoder_specialization.as_secs_f64())
      .set_tokenizer_load_time(self.tokenizer_load_time.as_secs_f64());
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
  /// VAD detector [`Self::transcribe`]'s VAD-chunked branch drives.
  /// Defaults to [`audio::vad::EnergyVad`] — Swift's own default for its
  /// counterpart `voiceActivityDetector` field (`WhisperKit.swift:880`) —
  /// and is swappable via [`Self::with_vad_detector`]/
  /// [`Self::set_vad_detector`] (coremlit issue #9: "Make VAD strategy
  /// pluggable or configurable rather than locking product behavior to
  /// the default energy VAD").
  vad_detector: Box<dyn audio::vad::VoiceActivityDetector + Send + Sync>,
  /// One-time load durations measured at construction, stamped into every
  /// run's [`TranscriptionTimings`] (see [`LoadTimings`]). All-zero for a
  /// [`Self::with_backend`] pipeline, which loads no models.
  load_timings: LoadTimings,
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
  /// dimensions are inconsistent (wraps [`crate::audio::whisper::backend::BackendError`]
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
    // Time the load the same way the compute pipeline already times its
    // stages (`Instant::elapsed`), populating the seven load fields Swift
    // stamps into `currentTimings` inside `loadModels`
    // (`WhisperKit.swift:396-441`). These are DURATIONS of a construction
    // step that inherently loads models from disk, not the absolute
    // wall-clock stamps the sans-I/O transcription rule (and #37.2) reject:
    // the whole prewarm pass is `prewarm_load_time`; the encoder/decoder
    // per-model splits come from the manager; and `model_loading` folds the
    // real load pass and the prewarm total together as Swift does (`:439`).
    let prewarm_load_time = if options.prewarm() {
      let start = Instant::now();
      manager.prewarm()?;
      start.elapsed()
    } else {
      Duration::ZERO
    };
    let model_load_start = Instant::now();
    let (models, model_splits): (_, ModelLoadTimings) = manager.into_loaded()?;
    let backend = CoreMlBackend::from_loaded(models).map_err(DecodeError::from)?;
    let tokenizer_start = Instant::now();
    let tokenizer = WhisperTokenizer::from_folder(options.tokenizer_folder())?;
    let tokenizer_load_time = tokenizer_start.elapsed();
    let dims = backend.dims();
    let variant = detect_variant(dims.vocab(), dims.embed_dim());
    // Swift's `modelLoading = now - modelLoadStart + prewarmLoadTime`
    // (`:439`): the whole real-load pass (models, tokenizer, and the dims
    // introspection variant detection needs) plus the earlier prewarm pass.
    let model_loading = model_load_start.elapsed() + prewarm_load_time;
    Ok(Self {
      backend,
      tokenizer,
      variant,
      vad_detector: Box::new(audio::vad::EnergyVad::new()),
      load_timings: LoadTimings {
        model_loading,
        prewarm_load_time,
        encoder_load: model_splits.encoder_load(),
        decoder_load: model_splits.decoder_load(),
        encoder_specialization: model_splits.encoder_specialization(),
        decoder_specialization: model_splits.decoder_specialization(),
        tokenizer_load_time,
      },
    })
  }
}

// Construction and field accessors: no bound — none of these touch an
// `InferenceBackend` method, so none may demand one (golden §8's
// "where clauses on the methods/impls that need them"; the same split
// `TranscribeTask`'s two impl blocks above already demonstrate).
impl<B> WhisperKit<B> {
  /// Wraps an already-constructed `backend`/`tokenizer` directly, with no
  /// model-loading step — the seam [`crate::audio::whisper::backend::mock::MockBackend`]-driven
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
      vad_detector: Box::new(audio::vad::EnergyVad::new()),
      // No models loaded through this seam, so nothing to time: the honest
      // load timings are all-zero (see [`LoadTimings`]).
      load_timings: LoadTimings::default(),
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

  /// The VAD detector [`Self::transcribe`]'s VAD-chunked branch drives.
  /// Defaults to [`audio::vad::EnergyVad`]; see [`Self::with_vad_detector`]/
  /// [`Self::set_vad_detector`] to swap it.
  #[inline(always)]
  pub fn vad_detector(&self) -> &(dyn audio::vad::VoiceActivityDetector + Send + Sync) {
    self.vad_detector.as_ref()
  }

  /// Builder form of [`Self::set_vad_detector`].
  #[must_use]
  #[inline(always)]
  pub fn with_vad_detector(
    mut self,
    detector: Box<dyn audio::vad::VoiceActivityDetector + Send + Sync>,
  ) -> Self {
    self.set_vad_detector(detector);
    self
  }

  /// Replaces the VAD detector [`Self::transcribe`]'s VAD-chunked branch
  /// drives — the seam matching Swift's injectable `voiceActivityDetector`
  /// field (`WhisperKit.swift:880`). [`audio::vad::EnergyVad`] (this
  /// pipeline's default) is one valid detector to construct here; any
  /// type implementing [`audio::vad::VoiceActivityDetector`] works too,
  /// provided it also satisfies this parameter's actual bounds —
  /// `Send + Sync + 'static` (the `'static` is implicit: the default
  /// object-lifetime bound for a `Box<dyn Trait + Send + Sync>` with no
  /// lifetime written out) — because the boxed detector is stored on
  /// `self` for the pipeline's full lifetime and must be safe to
  /// send/share across threads along with it.
  ///
  /// # Examples
  ///
  /// A detector holding non-`Send` state is rejected at compile time
  /// (this example is intentionally `compile_fail`):
  ///
  /// ```compile_fail
  /// use std::{cell::Cell, rc::Rc};
  /// use coremlit::audio::whisper::audio::vad::VoiceActivityDetector;
  /// use coremlit::audio::whisper::transcribe::WhisperKit;
  ///
  /// struct NotSendVad(Rc<Cell<u32>>);
  ///
  /// impl VoiceActivityDetector for NotSendVad {
  ///   fn voice_activity(&self, samples: &[f32]) -> Vec<bool> {
  ///     vec![false; samples.len()]
  ///   }
  ///   fn frame_length_samples(&self) -> usize {
  ///     160
  ///   }
  /// }
  ///
  /// fn reject<B>(kit: &mut WhisperKit<B>) {
  ///   // error[E0277]: `Rc<Cell<u32>>` cannot be sent between threads safely
  ///   kit.set_vad_detector(Box::new(NotSendVad(Rc::new(Cell::new(0)))));
  /// }
  /// ```
  #[inline(always)]
  pub fn set_vad_detector(
    &mut self,
    detector: Box<dyn audio::vad::VoiceActivityDetector + Send + Sync>,
  ) -> &mut Self {
    self.vad_detector = detector;
    self
  }

  /// Builds an [`AudioStreamTranscriber`] over this pipeline's own
  /// backend/tokenizer — the convenience constructor mirroring how Swift
  /// wires a streamer from the same pipeline components `WhisperKit.init`
  /// already assembled (`AudioStreamTranscriber.swift:43-74`). Swift's
  /// constructor takes the encoder/feature-extractor/segment-seeker/
  /// decoder as separate dependencies and assembles its own internal
  /// `TranscribeTask` from them (`:57-66`); this port's [`TranscribeTask`]
  /// is already the assembled unit callers construct directly, so
  /// [`AudioStreamTranscriber::push_samples`] builds one per run instead
  /// (see that module's doc) and this constructor needs only
  /// `backend`/`tokenizer`, exactly like [`TranscribeTask::new`] itself.
  #[inline(always)]
  pub fn audio_stream_transcriber(
    &self,
    decoding_options: DecodingOptions,
  ) -> AudioStreamTranscriber<'_, B> {
    AudioStreamTranscriber::new(self.backend(), self.tokenizer(), decoding_options)
  }

  /// Builds a [`LocalAgreementTranscriber`] over this pipeline — the
  /// simulated-stream driver for LocalAgreement-2 confirmation. Mirrors
  /// [`Self::audio_stream_transcriber`]'s role as the convenience
  /// constructor for this pipeline's other streaming driver; see
  /// [`crate::audio::whisper::stream::agreement`] for the port this wraps
  /// (`TranscribeCLI.swift:322-424`). Unlike
  /// [`AudioStreamTranscriber::new`], which takes `backend`/`tokenizer`
  /// separately, [`LocalAgreementTranscriber::new`] holds `self` directly:
  /// it calls [`Self::transcribe`] (Swift's own `whisperKit.transcribe`
  /// call site, `:369`) rather than assembling a
  /// [`TranscribeTask`] itself.
  #[inline(always)]
  pub fn local_agreement_transcriber(
    &self,
    options: DecodingOptions,
  ) -> LocalAgreementTranscriber<'_, B> {
    LocalAgreementTranscriber::new(self, options)
  }

  /// Folds this pipeline's construction-time [`LoadTimings`] into `result`'s
  /// timings — the port's counterpart to Swift carrying `currentTimings`'
  /// load fields into every `TranscriptionResult`
  /// (`WhisperKit.swift:396-441`, plumbed via `setupTranscribeTask`). Called
  /// on every result [`Self::transcribe`]/[`Self::transcribe_all`] hands
  /// back, after [`TranscribeTask::run`] built it: the run leaves the seven
  /// load fields at their zero defaults, so this only fills them in. No
  /// backend bound — this touches no [`InferenceBackend`] method.
  fn stamp_load_timings(&self, result: &mut TranscriptionResult) {
    let mut timings = result.timings().clone();
    self.load_timings.stamp(&mut timings);
    result.set_timings(timings);
  }
}

/// Assembles the record a VAD-merged transcript carries from the run's three
/// fact sources: the shared `sink`, the run's derived `worker_schedule`, and the
/// merged surviving result's `decoded_span`.
///
/// The `sink` is authoritative for the run's error-fragile facts (codex rounds
/// 4–9; round 11 adds the swallowed-error): it accumulated the per-attempt draw,
/// early-stop, language, and had-swallowed-error of EVERY chunk in ingestion order
/// — dropped-because-errored ones included, captured before their error could
/// propagate — so its Kleene-OR'd draw/early-stop/swallowed-error and its
/// FIRST-observed language are already the whole run's. A chunk that observed `es`
/// then errored and was dropped keeps `es` over a later surviving chunk's `fr`
/// (F3); a genuine zero-survivor run keeps the sink's own observed-clean
/// `Some(false)` draw/early-stop rather than the `unknown()` an empty merge folds
/// to (F4); and a chunk whose error was DROPPED leaves `had_swallowed_error =
/// Some(true)` in the sink, so the merged record is not reproducible (codex round
/// 11, M2). Those facts are taken from the sink verbatim; the merged surviving
/// result carries no draw/early-stop/language/swallowed-error the sink has not
/// already seen.
///
/// The `worker_schedule` and `decoded_span` are set EXPLICITLY rather than merged
/// from the surviving result (round 10 refactor): under the absorbing-`None`
/// schedule/span laws (F2/F3) the sink's stripped `None`s would otherwise ABSORB
/// the merged `Some`s away. The `worker_schedule` is the aggregate the caller
/// folded over ALL chunks (a dropped chunk's coordinate never reached a result,
/// so it taints the ordered schedule to `None`; zero chunks record the known-empty
/// `Some([])`), and the `decoded_span` is the merged surviving result's own — the
/// id-ordinal count its segments consumed, which drives a staged re-merge's ids —
/// EXCEPT that once any chunk errored the caller passes an
/// [`AtLeast`](SpanKnowledge::AtLeast) of the survivors' KNOWN sum, since the
/// dropped chunk's contribution is unknown but the survivors' ordinals still
/// lower-bound the run's total (codex round 11, M2; round 12).
fn recover_vad_run_facts(
  sink: TaskFacts,
  worker_schedule: Option<Vec<usize>>,
  decoded_span: SpanKnowledge,
) -> TaskFacts {
  sink
    .with_worker_schedule(worker_schedule)
    .with_decoded_span(decoded_span)
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
  /// [`merge_transcription_results_with_options`] right here is what keeps
  /// this method's own signature single-valued — pulling a call Swift
  /// leaves to its CLI down into the library, not adding behavior Swift
  /// lacks.
  ///
  /// When `options.chunking_strategy()` is
  /// [`ChunkingStrategy::Vad`](crate::audio::whisper::options::ChunkingStrategy::Vad)
  /// **and** `audio.len()` exceeds the backend's
  /// [`ModelDims::window_samples`](crate::audio::whisper::backend::ModelDims::window_samples)
  /// (:876-878): [`Self::vad_detector`] + [`chunker::VadChunker::chunk_all`]
  /// split `audio` along silence boundaries (:880-886 — Swift's injectable
  /// `voiceActivityDetector ?? EnergyVAD()` field, :880, is
  /// [`Self::vad_detector`]'s own counterpart: it defaults to
  /// [`audio::vad::EnergyVad`], same as Swift, and
  /// [`Self::with_vad_detector`]/[`Self::set_vad_detector`] are the seam
  /// that swaps it), with `clip_timestamps` cleared per chunk since
  /// chunking already consumed them (:892-894); each chunk runs its own
  /// [`TranscribeTask::run`], window-id-offset by its position in the
  /// chunk list (Swift's `audioIndex + batchIndex * batchSize`, :750 —
  /// which collapses to a flat running index here, see the deviation note
  /// below); a chunk whose task errored is dropped rather than failing the
  /// whole call, matching `updateSeekOffsetsForResults`'s `.failure`
  /// branch (`AudioChunker.swift:34-36` — Swift logs and skips, never
  /// rethrows); every surviving chunk's segments and
  /// [`TranscriptionResult::seek_time`] are re-anchored to the original
  /// timeline by [`chunker::apply_result_seek_offset`] before all chunk
  /// results are folded into one via
  /// [`merge_transcription_results_with_options`] — passed this call's own
  /// `options`, so a chunk the blank-audio drop emptied contributes no bare
  /// separator to the joined text while still contributing its timings to
  /// the merged sums (see that function's doc).
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
  /// [`crate::audio::whisper::backend::coreml::CoreMlBackend`] does not satisfy, and this
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
      let vad_chunker = chunker::VadChunker::new();
      let clip_ranges = chunker::prepare_seek_clips(options.clip_timestamps_slice(), audio.len())?;
      // Snapshot the detector's monotonic hard-failure generation before
      // driving it. Comparing this across chunking — rather than draining a
      // shared error slot — is what keeps the check correct when the detector
      // is shared across concurrent `transcribe` calls: no other run can clear
      // a failure this one is about to observe (see
      // `VoiceActivityDetector::detection_generation`).
      let detection_generation = self.vad_detector.detection_generation();
      let chunks = vad_chunker.chunk_all(
        self.vad_detector.as_ref(),
        audio,
        window_samples,
        &clip_ranges,
      );
      // The detector is infallible per frame (`voice_activity -> Vec<bool>`),
      // so a hard model/runtime failure during chunking latches inside it
      // rather than surfacing — bumping that generation. If it advanced, a
      // swallowed VAD failure produced degraded chunk boundaries (speech
      // misread as silence), so fail the whole transcription with a typed
      // `VadError` instead of returning an `Ok` transcript off a
      // silently-corrupted segmentation. The generation is the authority on
      // *whether* to fail; `last_detection_error` only supplies the detail,
      // which a concurrent run may have raced in to read (and, for a
      // destructive detector, clear) — so fail closed even if it is `None`.
      if self.vad_detector.detection_generation() != detection_generation {
        let source = self
          .vad_detector
          .last_detection_error()
          .unwrap_or_else(|| "hard model inference failure during VAD chunking".into());
        return Err(TranscribeError::Vad(VadError::Detection(source)));
      }
      let chunk_options = options.clone().with_clip_timestamps(Vec::new());

      // One [`TaskFacts`] sink shared across every chunk: `TranscribeTask::run`
      // merges each window's error-fragile facts — the RNG draw, the early-stop
      // truncation, and the language observation — into it the instant an attempt
      // settles, BEFORE any error can propagate (see `TranscribeTask::with_facts_sink`).
      // A chunk whose task ERRORS is dropped below rather than merged, so those
      // facts would otherwise vanish — leaving the merged transcript reading
      // reproducible off the surviving greedy chunks while a re-run redraws that
      // dropped chunk's unseeded sample and may land different surviving text, or
      // reading `detected_language == None` for a run that plainly observed one,
      // or claiming reproducibility for a callback-truncated run (coremlit issue
      // #14, codex rounds 4–6 — the early-stop recovery is the round-6 R6-F1
      // addition). Neither the worker schedule nor the id span rides this sink:
      // the schedule is folded over EVERY chunk separately (below, so an errored
      // chunk taints it to the adjudicated `None`, round 10 F2), and the id span
      // is the merged surviving result's own.
      //
      // Seeded **observed-clean** like the per-run sink (codex round 8, F3): the
      // VAD run as a whole is watching every chunk, so before any chunk draws it
      // has seen no draw and no truncation. Under the Kleene OR `Some(false)` is
      // the identity, so a greedy chunk's `Some(false)` leaves the sink
      // `Some(false)`; a chunk that draws flips it to `Some(true)`, recovered into
      // the merged record below even when that chunk errored and was dropped.
      let facts_sink = Mutex::new(TaskFacts::observed_clean());
      let mut chunk_results = Vec::with_capacity(chunks.len());
      // The worker schedule is folded over EVERY chunk through the fixed merge law
      // (round 10, F2), seeded known-empty (`Some([])`): a surviving chunk
      // contributes its known coordinate `[chunk_index]`, and a chunk that errored
      // and was dropped contributes an UNKNOWN schedule (`None`) that — the
      // schedule law being absorbing-`None` — taints the ordered aggregate to
      // `None` rather than letting the survivors pass for the whole schedule. Zero
      // chunks keep the `Some([])` seed: the run observed zero workers. This fact
      // cannot ride the per-attempt `facts_sink` (that would duplicate a
      // coordinate per fallback), so it is derived here where every chunk's fate
      // is visible.
      let mut schedule = TaskFacts::unknown().with_worker_schedule(Some(Vec::new()));
      // Whether ANY chunk errored and was dropped. A dropped chunk hid an error
      // that controlled the returned transcript (the sink records the swallow) and
      // may have allocated id ordinals before erroring, so the run's decoded span
      // becomes honestly unknown once it happens — the decoded-span analogue of the
      // schedule's absorbing-`None` fold (codex round 11, M2).
      let mut any_chunk_dropped = false;
      for (chunk_index, chunk) in chunks.iter().enumerate() {
        let outcome = TranscribeTask::new(&self.backend, &self.tokenizer)
          .with_window_id_offset(chunk_index)
          .with_facts_sink(&facts_sink)
          .run(chunk.samples_slice(), &chunk_options);
        let coordinate = if let Ok(mut result) = outcome {
          chunker::apply_result_seek_offset(&mut result, chunk.seek_offset());
          chunk_results.push(result);
          TaskFacts::unknown().with_worker(chunk_index)
        } else {
          // An errored chunk is dropped here — its error-fragile draw/early-stop/
          // language already reached the sink before the error, but its coordinate
          // never reached a result, so its schedule contribution is unknown.
          //
          // Record the DROP itself as a swallowed error into the shared sink (codex
          // round 11, M2): an observed `Some(true)` `had_swallowed_error` that
          // forces `is_reproducible` false, since the hidden error controlled the
          // transcript and a re-run of the same audio/options need not reproduce the
          // drop. The observed-clean marker leaves the sink's draw/early-stop/
          // language untouched (`Some(false)` is the Kleene-OR identity, and it
          // carries no language) while OR-ing the swallow flag to `Some(true)`.
          facts_sink
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .merge(&TaskFacts::observed_clean().with_had_swallowed_error(true));
          any_chunk_dropped = true;
          TaskFacts::unknown()
        };
        schedule.merge(&coordinate);
      }
      // `_with_options`, not the plain merge: the blank-audio drop can
      // empty a whole chunk (a wholly-silent one decodes to nothing but
      // `[BLANK_AUDIO]`, which the filter then removes), and `chunk_all` is
      // a CONTIGUOUS chunker — `start = end` marches across the clip and
      // nothing is ever skipped — so a long enough silence really does
      // become a chunk of its own. Swift's join would land it in the
      // transcript as a bare separator. Passing the very `options` the
      // chunks were decoded with is what keeps the merged text and the
      // decode from disagreeing; every chunk is still merged, so no chunk's
      // timings leave the sums.
      let mut merged = merge_transcription_results_with_options(&chunk_results, options);
      // Assemble the merged record's facts (round 10 refactor of codex rounds
      // 4–9's VAD recovery; extended round 11): the shared sink is authoritative
      // for the error-fragile draw/early-stop/language/swallowed-error it watched
      // across EVERY chunk (dropped ones included — the drop itself is recorded as
      // a swallow above), while the worker schedule folded over all chunks above
      // and the run's decoded span are set explicitly — the sink's stripped
      // schedule would otherwise absorb the merged coordinates, and its span seed
      // is only the wholly-unknown default. The decoded span is the merged
      // surviving result's own — an `Exact` sum of the surviving chunks' counts
      // (or `Exact(0)` for a genuine zero-chunk run) — degraded to an `AtLeast` of
      // the survivors' KNOWN sum once ANY chunk errored: the dropped chunk may
      // have allocated ordinals before erroring, so the exact total is unknown,
      // yet the survivors' ordinals still lower-bound the run's span. That lower
      // bound is the round-12 replacement for the pre-round-12 `None`, which threw
      // the survivors' known sum away (codex round 11, M2; round 12). See
      // [`recover_vad_run_facts`].
      let sink_facts = facts_sink
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
      let decoded_span = if any_chunk_dropped {
        // The dropped chunk's contribution is unknown; the surviving merge's own
        // lower bound is the run's known floor (round 12, replacing the pre-round-12
        // `None` that threw the survivors' known sum away).
        SpanKnowledge::AtLeast(merged.task_facts().decoded_span().lower_bound())
      } else if chunk_results.is_empty() {
        // A zero-chunk run allocated exactly nothing — a KNOWN-empty `Exact(0)`,
        // distinct from the wholly-unknown value a run that cannot see would carry.
        SpanKnowledge::Exact(0)
      } else {
        merged.task_facts().decoded_span()
      };
      let recovered = recover_vad_run_facts(
        sink_facts,
        schedule.worker_schedule().map(|s| s.to_vec()),
        decoded_span,
      );
      *merged.task_facts_mut() = recovered;
      // Stamps the merged result once, not each chunk before merging — and
      // the two are equivalent only for the five max-combined load fields.
      // Load timings are identical for every chunk in this `WhisperKit`, so
      // stamping post-merge lands the same value pre-merge stamping (then
      // max-combining identical values) would. The two specialization
      // fields differ: the merge (`merge_results`, `result/mod.rs:2381-2389`)
      // always leaves them at zero regardless of what the inputs held, so
      // pre-merge stamping would still merge away to 0.0 — stamping the
      // merged result here is what carries real specialization values
      // through.
      self.stamp_load_timings(&mut merged);
      return Ok(merged);
    }

    let mut result = TranscribeTask::new(&self.backend, &self.tokenizer).run(audio, options)?;
    self.stamp_load_timings(&mut result);
    Ok(result)
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
  /// [`crate::audio::whisper::result::TranscriptionProgress::window_id`] stays distinct
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

    // Stamp construction-time load timings onto every successful result,
    // mirroring Swift's `currentTimings` riding into each task's result.
    for result in results.iter_mut().flatten() {
      self.stamp_load_timings(result);
    }
    results
  }

  /// One-shot language-detection probe over the first
  /// [`window_samples`](crate::audio::whisper::backend::ModelDims::window_samples)-worth
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
