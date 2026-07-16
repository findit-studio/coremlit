//! LocalAgreement-2 streaming confirmation: the hypothesis-agreement
//! engine ([`LocalAgreement`]) and the simulated-stream driver that wraps
//! it ([`LocalAgreementTranscriber`]) — ports the CLI's
//! `transcribeStreamSimulated` loop (`TranscribeCLI.swift:322-424`,
//! specifically its LocalAgreement-2 bookkeeping and loop body at
//! `:346-421`).
//!
//! [`LocalAgreement`] is pure: it consumes already-decoded
//! [`TranscriptionResult`]s (word timings and text, no backend, no I/O)
//! and is fully hermetic to test. [`LocalAgreementTranscriber`] is the
//! thin driver around it that owns a growing sample buffer and calls
//! [`crate::transcribe::WhisperKit::transcribe`] once per stride.
//!
//! **Documented deviations** from `TranscribeCLI.swift`:
//!
//! - **Gate semantics** (Swift `:371`, `if let result = result, let _ =
//!   result.segments.first?.words`): Swift's check is "the first
//!   segment's `words` property is non-nil" — optional-typed in Swift,
//!   so nil (alignment weights unavailable) and `[]` (computed, zero
//!   words) are distinguishable there. This port's
//!   [`crate::result::TranscriptionSegment::words_slice`] is never
//!   optional (empty-means-absent, that module's own doc), so nil and
//!   `[]` already collapse to the same representation before
//!   [`LocalAgreement::ingest`] ever sees it — "any segment has a
//!   non-empty `words_slice`" is the closest faithful gate reachable
//!   from that representation, checking every segment rather than only
//!   the first since there is no cheaper-but-still-correct equivalent of
//!   Swift's specifically-first-segment check.
//! - **Errors propagate.** Swift's per-stride `catch` logs and continues
//!   (`:411-415`); [`LocalAgreementTranscriber::push_samples`] instead
//!   returns `Result` and stops at the first error, leaving the caller to
//!   decide whether to retry or abandon the stream.
//! - **`word_timestamps` is forced.** [`LocalAgreementTranscriber::new`]
//!   sets [`DecodingOptions::word_timestamps`] on its own options copy
//!   unconditionally; Swift leaves this to a user-supplied CLI flag
//!   (`TranscribeCLIUtils.createDecodingOptions`). LocalAgreement-2 has
//!   no signal to agree over without word timings — every ingested
//!   result would otherwise hit the [`AgreementOutcome::NoWordTimings`]
//!   gate.
//! - **Stride cadence starts from zero, not one stride in.** Swift's `for
//!   seekSample in stride(from: 16000, to: audioArray.count, by: 16000)`
//!   (`:357`) starts its induction variable at `16000`, so its *first*
//!   transcribed window is `[0, 32000)` (2 s) and audio no longer than
//!   1 s is never transcribed at all (the stride sequence is empty
//!   whenever `audioArray.count <= 16000`). This port's
//!   [`LocalAgreementTranscriber`] cursor starts at `0` instead, so its
//!   first window is `[0, 16000)` (1 s) and any audio of at least 1 s
//!   produces at least one stride. Swift loops once over a fully
//!   buffered static array and derives its induction variable from that;
//!   this port has no such array, only a growing buffer crossing
//!   [`STRIDE_SAMPLES`]-sized thresholds as samples are pushed in — a
//!   deliberate regularization for that push-based shape, not a
//!   byte-for-byte port of Swift's off-by-one starting point.
//! - **`push_samples` needs only `B: InferenceBackend`, not `+ Sync`.**
//!   Its only backend-touching call is
//!   [`crate::transcribe::WhisperKit::transcribe`], whose own `impl`
//!   block bound is `B: InferenceBackend` alone — `Sync` is
//!   `WhisperKit::transcribe_all`'s addition, for its concurrent worker
//!   pool (`crate::transcribe`'s module doc, "Concurrency note"), and
//!   [`InferenceBackend`] itself has no `Sync` supertrait either. This is
//!   a correction against this task's own brief, which specified `B:
//!   InferenceBackend + Sync` here.

use crate::{
  backend::InferenceBackend,
  constants::SAMPLE_RATE,
  error::TranscribeError,
  options::DecodingOptions,
  result::{TranscriptionResult, WordTiming, merge_transcription_results_with_words},
  task_facts::TaskFactsAccumulator,
  text::{find_longest_common_prefix, find_longest_different_suffix},
  transcribe::WhisperKit,
};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// AgreementOutcome
// ---------------------------------------------------------------------

/// One [`LocalAgreement::ingest`] call's outcome — whether the new result
/// advanced the confirmation watermark, merely awaits a future result to
/// agree with, or carried no word timings to agree over at all. Swift
/// expresses these same three outcomes as local bookkeeping (`skipAppend`,
/// the no-words `else` branch) rather than a value
/// (`TranscribeCLI.swift:370-410`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum AgreementOutcome {
  /// The new result's hypothesis agreed with the previous one on at least
  /// [`LocalAgreement::agreement_count_needed`] words: the confirmation
  /// watermark advanced and the result was kept.
  Advanced,
  /// Either there is no previous result to agree with yet (the first
  /// ingested result), or the new hypothesis disagreed with the previous
  /// one — the watermark is unchanged and, in the disagreement case, the
  /// result was dropped rather than kept.
  AwaitingAgreement,
  /// The result carried no word timings to agree over; it was still kept
  /// (Swift `:403-409` falls through to the unconditional append).
  NoWordTimings,
}

impl AgreementOutcome {
  /// Stable snake_case name of the variant.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Advanced => "advanced",
      Self::AwaitingAgreement => "awaiting_agreement",
      Self::NoWordTimings => "no_word_timings",
    }
  }
}

// ---------------------------------------------------------------------
// LocalAgreement
// ---------------------------------------------------------------------

/// Default [`LocalAgreement::agreement_count_needed`] — Swift's
/// `agreementCountNeeded` local (`TranscribeCLI.swift:349`).
pub const DEFAULT_AGREEMENT_COUNT_NEEDED: usize = 2;

/// The LocalAgreement-2 hypothesis-confirmation engine: consumes one
/// [`TranscriptionResult`] per call and tracks the growing prefix two
/// consecutive hypotheses agree on. Pure — no backend, no I/O, fully
/// hermetic to test; ports the bookkeeping locals and loop body of
/// `transcribeStreamSimulated` (`TranscribeCLI.swift:346-421`) minus the
/// transcription call itself, which is
/// [`LocalAgreementTranscriber::push_samples`]'s job.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalAgreement {
  agreement_count_needed: usize,
  last_agreed_seconds: f32,
  prev_result: Option<TranscriptionResult>,
  prev_words: Vec<WordTiming>,
  hypothesis_words: Vec<WordTiming>,
  last_agreed_words: Vec<WordTiming>,
  confirmed_words: Vec<WordTiming>,
  results: Vec<TranscriptionResult>,
  /// A sink for the reproducibility facts of EVERY ingested hypothesis —
  /// including the disagreeing ones dropped from [`Self::results`] but retained
  /// as [`Self::prev_result`] to CONTROL the next agreement comparison (codex
  /// round 8, F1). The same error-drop-sink pattern the VAD branch uses: a
  /// dropped hypothesis's unseeded draw (or callback truncation) still decided
  /// which words the surviving hypotheses agreed on, so it must reach
  /// [`Self::finalize`]'s reproducibility answer even though its segments never
  /// survive into the merge. Only the draw/early-stop/language facts are folded
  /// (the worker schedule and id span are stripped, so the merged result's own
  /// — from the surviving results — are left intact).
  ingested_facts: TaskFactsAccumulator,
}

impl Default for LocalAgreement {
  fn default() -> Self {
    Self::new()
  }
}

impl LocalAgreement {
  /// A fresh engine: no prior result, a zero watermark, every collection
  /// empty, [`DEFAULT_AGREEMENT_COUNT_NEEDED`] words required to confirm
  /// (Swift's all-default locals, `TranscribeCLI.swift:346-353`).
  pub const fn new() -> Self {
    Self {
      agreement_count_needed: DEFAULT_AGREEMENT_COUNT_NEEDED,
      last_agreed_seconds: 0.0,
      prev_result: None,
      prev_words: Vec::new(),
      hypothesis_words: Vec::new(),
      last_agreed_words: Vec::new(),
      confirmed_words: Vec::new(),
      results: Vec::new(),
      ingested_facts: TaskFactsAccumulator::new(),
    }
  }

  // -- agreement_count_needed -----------------------------------------------
  /// Consecutive agreeing words required to advance the confirmation
  /// watermark.
  #[inline(always)]
  pub const fn agreement_count_needed(&self) -> usize {
    self.agreement_count_needed
  }
  /// Builder form of [`Self::set_agreement_count_needed`].
  #[must_use]
  #[inline(always)]
  pub const fn with_agreement_count_needed(mut self, agreement_count_needed: usize) -> Self {
    self.set_agreement_count_needed(agreement_count_needed);
    self
  }
  /// Sets [`Self::agreement_count_needed`] in place, clamped up to at
  /// least `1`. Zero would hold back no words at all on an advance
  /// (`Self::ingest`'s `common[split..]` slice with `split ==
  /// common.len()`), leaving no first held-back word to anchor
  /// [`Self::last_agreed_seconds`] to — an algorithmically degenerate
  /// configuration Swift's hardcoded `agreementCountNeeded = 2`
  /// (`TranscribeCLI.swift:349`) never reaches, since Swift never exposes
  /// this knob as configurable at all; its own `lastAgreedWords.first!`
  /// (`:385`) would force-unwrap-crash on the same input if it somehow
  /// did. This setter is the only way to reach [`Self::agreement_count_needed`]
  /// from outside the module, so clamping here keeps `ingest` panic-free
  /// for every value this type can actually hold.
  #[inline(always)]
  pub const fn set_agreement_count_needed(&mut self, agreement_count_needed: usize) -> &mut Self {
    self.agreement_count_needed = if agreement_count_needed == 0 {
      1
    } else {
      agreement_count_needed
    };
    self
  }

  // -- last_agreed_seconds ---------------------------------------------------
  /// The confirmation watermark, in seconds: word timings before this
  /// point are settled and will not be revisited.
  #[inline(always)]
  pub const fn last_agreed_seconds(&self) -> f32 {
    self.last_agreed_seconds
  }

  // -- last_agreed_words (Vec<WordTiming>) -----------------------------------
  /// The most recent agreement's trailing [`Self::agreement_count_needed`]
  /// words — held back from [`Self::confirmed_words_slice`] since a
  /// still-later hypothesis could yet revise them.
  #[inline(always)]
  pub const fn last_agreed_words_slice(&self) -> &[WordTiming] {
    self.last_agreed_words.as_slice()
  }

  // -- confirmed_words (Vec<WordTiming>) -------------------------------------
  /// Word timings settled so far: every agreement's leading remainder,
  /// ahead of that agreement's own [`Self::agreement_count_needed`]-word
  /// holdback.
  #[inline(always)]
  pub const fn confirmed_words_slice(&self) -> &[WordTiming] {
    self.confirmed_words.as_slice()
  }

  // -- results (Vec<TranscriptionResult>) ------------------------------------
  /// Every ingested result kept for the eventual [`Self::finalize`] merge
  /// — every result except the ones a disagreeing hypothesis caused to be
  /// dropped (`TranscribeCLI.swift:395-400`, `skipAppend`).
  #[inline(always)]
  pub const fn results_slice(&self) -> &[TranscriptionResult] {
    self.results.as_slice()
  }

  /// `base`, retargeted at the next stride: the clip start moved to
  /// [`Self::last_agreed_seconds`] and the decoder prefilled with
  /// [`Self::last_agreed_words_slice`]'s tokens — ports
  /// `TranscribeCLI.swift:364-367` (`streamOptions.clipTimestamps =
  /// [lastAgreedSeconds]`; `streamOptions.prefixTokens =
  /// lastAgreedWords.flatMap { $0.tokens }`).
  pub fn decoding_options_for_next(&self, base: &DecodingOptions) -> DecodingOptions {
    let prefix_tokens: Vec<u32> = self
      .last_agreed_words
      .iter()
      .flat_map(|word| word.tokens_slice().iter().copied())
      .collect();
    base
      .clone()
      .with_clip_timestamps(vec![self.last_agreed_seconds])
      .with_prefix_tokens(prefix_tokens)
  }

  /// The agreement view of a result's words: everything at or past the
  /// watermark, MINUS the already-confirmed words a tied start would
  /// re-admit. The watermark is the first held-back word's start; a word
  /// confirmed in the previous round can share that exact start (DTW row
  /// steps without a column advance, then centisecond rounding — ties are
  /// pipeline-reachable), and a bare timestamp filter would pull it back
  /// in and confirm it AGAIN (phase-gate round-1 finding; Swift shares
  /// the bug, and "confirmed once and stable" wins over parity here).
  /// Confirmed words are time-ordered, so the re-admitted ones are
  /// exactly the trailing run with `start >= watermark`, and they sit at
  /// the front of the filtered list — skipping that count restores the
  /// invariant on both sides of the prefix comparison.
  fn watermark_filtered(&self, result: &TranscriptionResult) -> Vec<WordTiming> {
    Self::watermark_filtered_with(result, self.last_agreed_seconds, &self.confirmed_words)
  }

  fn watermark_filtered_with(
    result: &TranscriptionResult,
    watermark: f32,
    confirmed: &[WordTiming],
  ) -> Vec<WordTiming> {
    // The confirmed tail that a tied start would re-admit, in order.
    let tail_start = confirmed
      .iter()
      .rev()
      .take_while(|word| word.start() >= watermark)
      .count();
    let readmit_candidates = &confirmed[confirmed.len() - tail_start..];
    let filtered: Vec<WordTiming> = result
      .all_words()
      .into_iter()
      .filter(|word| word.start() >= watermark)
      .collect();
    // Strip only an ACTUALLY MATCHING prefix (normalized, the agreement's
    // own equality — `find_longest_common_prefix` compares the same way):
    // an unconditional count-skip dropped a PROVISIONAL word whenever a
    // rewrite omitted a confirmed tied word and shifted everything left
    // (phase-gate round-2 finding).
    let strip = readmit_candidates
      .iter()
      .zip(&filtered)
      .take_while(|(confirmed_word, candidate)| {
        crate::text::normalized(confirmed_word.word()) == crate::text::normalized(candidate.word())
      })
      .count();
    filtered[strip..].to_vec()
  }

  /// Folds one freshly-decoded `result` into the engine. Ports
  /// `TranscribeCLI.swift:370-410`:
  ///
  /// - If no segment of `result` carries a word timing, `result` is kept
  ///   in [`Self::results_slice`] anyway (`:403-409`: the `else` branch
  ///   still falls through to the unconditional `!skipAppend` append) and
  ///   this returns [`AgreementOutcome::NoWordTimings`] — see this
  ///   module's doc for why "any segment" replaces Swift's
  ///   first-segment-only check.
  /// - Otherwise, `result.all_words()` filtered to `start >=
  ///   last_agreed_seconds()` becomes the new hypothesis (`:372`). With no
  ///   previous result yet (the first call ever, or the first call after
  ///   [`Self::new`]), there is nothing to compare against: `result` is
  ///   kept and this returns [`AgreementOutcome::AwaitingAgreement`] —
  ///   Swift runs no agreement logic on this path either (`:374`'s `if
  ///   let prevResult = prevResult` is simply not entered).
  /// - With a previous result, its own `all_words()` (filtered the same
  ///   way, `:375`) and the new hypothesis feed
  ///   [`crate::text::find_longest_common_prefix`] (`:376`). A common
  ///   prefix at least [`Self::agreement_count_needed`] words long
  ///   advances the watermark: its trailing `agreement_count_needed`
  ///   words become the new [`Self::last_agreed_words_slice`] (whose
  ///   first word's start is the new [`Self::last_agreed_seconds`]), its
  ///   leading remainder is folded into [`Self::confirmed_words_slice`],
  ///   `result` is kept, and this returns [`AgreementOutcome::Advanced`]
  ///   (`:383-394`). Otherwise the hypotheses disagree: the watermark is
  ///   unchanged, `result` is **dropped** rather than kept (`:395-400`,
  ///   `skipAppend`), and this returns
  ///   [`AgreementOutcome::AwaitingAgreement`].
  ///
  /// Either way — agreeing, disagreeing, or no previous result — `result`
  /// becomes the new previous result for the next call (`:402`, outside
  /// the agreement `if`/`else` but still inside the has-words branch).
  pub fn ingest(&mut self, result: TranscriptionResult) -> AgreementOutcome {
    // Accumulate THIS hypothesis's reproducibility facts BEFORE any gate or
    // branch, so a hypothesis dropped from `results` on disagreement (:395-400,
    // `skipAppend`) still contributes them to `finalize` (codex round 8, F1). It
    // controlled which words the surviving hypotheses agreed on — a re-run that
    // redraws its unseeded sample may land different confirmed text — so its draw
    // must not vanish with its segments. Worker schedule and id span are stripped
    // to `None`: those come from the SURVIVING results via the merge, and folding
    // a dropped hypothesis's coordinate/span in would corrupt them.
    self.ingested_facts.merge(
      &result
        .task_facts()
        .clone()
        .with_worker_schedule(None)
        .with_decoded_span(None),
    );

    // :371 gate — see this module's doc for "any segment" vs. Swift's
    // first-segment-only nil check.
    let has_words = result
      .segments_slice()
      .iter()
      .any(|segment| !segment.words_slice().is_empty());
    if !has_words {
      self.results.push(result);
      return AgreementOutcome::NoWordTimings;
    }

    // :372 — plus the readmit skip (see `watermark_filtered`).
    self.hypothesis_words = self.watermark_filtered(&result);

    let mut advanced = false;
    let mut skip_append = false;
    // :374 — absent on the first-ever call, so nothing below runs and
    // this falls through to the `AwaitingAgreement` append below.
    if let Some(prev_result) = &self.prev_result {
      // :375-376 — the SAME filter as the hypothesis, so the two sides
      // stay index-aligned for the prefix comparison.
      self.prev_words =
        Self::watermark_filtered_with(prev_result, self.last_agreed_seconds, &self.confirmed_words);
      let common = find_longest_common_prefix(&self.prev_words, &self.hypothesis_words);
      if common.len() >= self.agreement_count_needed {
        // :383-394 — advance the watermark.
        let split = common.len() - self.agreement_count_needed;
        self.confirmed_words.extend_from_slice(&common[..split]);
        self.last_agreed_words = common[split..].to_vec();
        self.last_agreed_seconds = self.last_agreed_words[0].start();
        advanced = true;
      } else {
        // :395-400 — disagreement; `result` is dropped below.
        skip_append = true;
      }
    }

    // :402 (unconditional) + :408-410 (`!skipAppend`).
    if skip_append {
      self.prev_result = Some(result);
    } else {
      self.prev_result = Some(result.clone());
      self.results.push(result);
    }

    if advanced {
      AgreementOutcome::Advanced
    } else {
      AgreementOutcome::AwaitingAgreement
    }
  }

  /// Consumes the engine and produces the final merged transcript. Ports
  /// `TranscribeCLI.swift:418-421`: the last (still-provisional)
  /// agreement's [`Self::last_agreed_words_slice`], then whatever the
  /// final hypothesis added beyond the final previous result
  /// ([`crate::text::find_longest_different_suffix`] over the last
  /// ingested pair), both folded onto [`Self::confirmed_words_slice`];
  /// [`merge_transcription_results_with_words`] then merges every kept
  /// [`Self::results_slice`] result with that word list as the merged
  /// text, under `options` — the same options the kept results were decoded
  /// with, so the merged segments honor
  /// [`DecodingOptions::drop_blank_audio`]'s id mapping (which the confirmed
  /// text override does not touch, but the segments still carry).
  ///
  /// The reproducibility facts of EVERY ingested hypothesis — including the
  /// disagreeing ones dropped from [`Self::results_slice`] — are then folded
  /// onto the merged record from `Self::ingested_facts` (codex round 8, F1),
  /// so a dropped control hypothesis's unseeded draw or callback truncation is
  /// not lost from the finalized transcript's reproducibility answer. Worker
  /// schedule and id span were stripped at ingest, so this touches only the
  /// draw/early-stop/language facts; the merged result's own schedule and span
  /// (from the surviving results) are untouched.
  pub fn finalize(mut self, options: &DecodingOptions) -> TranscriptionResult {
    self.confirmed_words.append(&mut self.last_agreed_words);
    let suffix = find_longest_different_suffix(&self.prev_words, &self.hypothesis_words);
    self.confirmed_words.extend_from_slice(suffix);
    let mut merged =
      merge_transcription_results_with_words(&self.results, &self.confirmed_words, options);
    merged
      .task_facts_mut()
      .merge(&self.ingested_facts.into_facts());
    merged
  }
}

// ---------------------------------------------------------------------
// LocalAgreementTranscriber
// ---------------------------------------------------------------------

/// Samples per stride: 1 s at [`SAMPLE_RATE`] — Swift's `16000` stride
/// literal (`TranscribeCLI.swift:357`). See this module's doc for how this
/// port's cursor start differs from Swift's induction variable.
pub const STRIDE_SAMPLES: usize = SAMPLE_RATE as usize;

/// The simulated-stream driver: feeds a growing audio buffer through
/// [`crate::transcribe::WhisperKit::transcribe`] one [`STRIDE_SAMPLES`]
/// stride at a time, folding each result through a [`LocalAgreement`].
/// Ports the loop shell of `transcribeStreamSimulated`
/// (`TranscribeCLI.swift:357-369`) — see this module's doc for the
/// `word_timestamps`-forcing and error-propagation deviations, and
/// [`LocalAgreement::ingest`] for the per-result confirmation logic this
/// driver doesn't itself implement.
///
/// Bare struct, no bounds — bounds live on the `impl` blocks below,
/// narrowed to just [`Self::push_samples`], the only member needing `B:
/// InferenceBackend` (golden §8; mirrors
/// [`crate::stream::AudioStreamTranscriber`]'s own two-impl-block split).
pub struct LocalAgreementTranscriber<'ctx, B> {
  kit: &'ctx WhisperKit<B>,
  options: DecodingOptions,
  agreement: LocalAgreement,
  buffer: Vec<f32>,
  transcribed_samples: usize,
}

impl<'ctx, B> LocalAgreementTranscriber<'ctx, B> {
  /// A fresh driver over `kit`, with a fresh [`LocalAgreement`] and an
  /// empty buffer. Forces [`DecodingOptions::word_timestamps`] on its own
  /// copy of `options` — see this module's doc for why (LocalAgreement-2
  /// has nothing to agree over without word timings); Swift leaves this to
  /// a user-supplied CLI flag instead.
  pub fn new(kit: &'ctx WhisperKit<B>, options: DecodingOptions) -> Self {
    Self {
      kit,
      options: options.with_word_timestamps(),
      agreement: LocalAgreement::new(),
      buffer: Vec::new(),
      transcribed_samples: 0,
    }
  }

  /// The live confirmation engine — read
  /// [`LocalAgreement::confirmed_words_slice`] for the settled transcript
  /// so far without waiting for [`Self::finalize`].
  #[inline(always)]
  pub const fn agreement(&self) -> &LocalAgreement {
    &self.agreement
  }

  /// Total samples accumulated in the session buffer so far.
  #[inline(always)]
  pub const fn buffer_len(&self) -> usize {
    self.buffer.len()
  }

  /// Consumes the driver and produces the final merged transcript.
  /// Delegates to [`LocalAgreement::finalize`], passing this driver's own
  /// (word-timestamp-forced) [`DecodingOptions`] so the merge honors the
  /// same [`DecodingOptions::drop_blank_audio`] the streamed results decoded
  /// under.
  pub fn finalize(self) -> TranscriptionResult {
    self.agreement.finalize(&self.options)
  }
}

impl<B> LocalAgreementTranscriber<'_, B>
where
  B: InferenceBackend,
{
  /// Appends `samples` to the session buffer, then runs one transcription
  /// pass per complete [`STRIDE_SAMPLES`] stride that has newly
  /// accumulated (zero, one, or several, depending on how much of
  /// `samples` was pending — arbitrary push sizes coalesce to the same
  /// fixed cadence). Each pass transcribes the buffer from the start
  /// through that stride's end
  /// ([`crate::transcribe::WhisperKit::transcribe`], with options
  /// retargeted per [`LocalAgreement::decoding_options_for_next`]) and
  /// folds the result through [`LocalAgreement::ingest`]. Ports
  /// `TranscribeCLI.swift:357-369`.
  ///
  /// # Errors
  /// Whatever [`crate::transcribe::WhisperKit::transcribe`] returns,
  /// propagated directly and immediately — a later stride is never
  /// attempted after an earlier one fails. **Documented deviation:**
  /// Swift's per-stride `catch` instead logs the error and continues to
  /// the next stride (`TranscribeCLI.swift:411-415`).
  ///
  /// A failing stride does not roll back the strides that already
  /// succeeded earlier in the *same* call: [`Self::agreement`]'s
  /// watermark/confirmed words and [`Self::buffer_len`]'s progress already
  /// reflect them, even though their [`AgreementOutcome`]s are not in the
  /// `Vec` this call returns (the `Err` replaces it). Call
  /// [`Self::agreement`] to inspect what happened so far before deciding
  /// whether to retry with more samples or abandon the stream.
  pub fn push_samples(
    &mut self,
    samples: &[f32],
  ) -> Result<Vec<AgreementOutcome>, TranscribeError> {
    self.buffer.extend_from_slice(samples);
    let mut outcomes = Vec::new();
    // `saturating_sub` (not a bare `-`): `transcribed_samples` is only
    // ever a past `buffer.len()` and the buffer never shrinks, so this
    // never actually saturates — same reasoning as
    // `AudioStreamTranscriber::push_samples`'s own `last_buffer_size`
    // comparison.
    while self.buffer.len().saturating_sub(self.transcribed_samples) >= STRIDE_SAMPLES {
      let end = (self.transcribed_samples + STRIDE_SAMPLES).min(self.buffer.len());
      let options = self.agreement.decoding_options_for_next(&self.options);
      let result = self.kit.transcribe(&self.buffer[..end], &options)?;
      outcomes.push(self.agreement.ingest(result));
      self.transcribed_samples = end;
    }
    Ok(outcomes)
  }
}
