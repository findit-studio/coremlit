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
//! [`crate::audio::whisper::transcribe::WhisperKit::transcribe`] once per stride.
//!
//! **Documented deviations** from `TranscribeCLI.swift`:
//!
//! - **Gate semantics** (Swift `:371`, `if let result = result, let _ =
//!   result.segments.first?.words`): Swift's check is "the first
//!   segment's `words` property is non-nil" — optional-typed in Swift,
//!   so nil (alignment weights unavailable) and `[]` (computed, zero
//!   words) are distinguishable there. This port's
//!   [`crate::audio::whisper::result::TranscriptionSegment::words_slice`] is never
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
//! - **The split may not cut at a tied start (Rule W), so a confirmed word is
//!   never re-offered** (Swift `:372`/`:375`). Swift's hypothesis view is a bare
//!   timestamp filter, `start >= lastAgreedSeconds`, and the watermark is the
//!   first held-back word's start — so a word confirmed in the previous round
//!   that shares that exact start (DTW row steps without a column advance, then
//!   centisecond rounding; ties are pipeline-reachable) is pulled back in and
//!   confirmed AGAIN. This port keeps Swift's filter verbatim and closes the
//!   hole at its SOURCE instead: an advance refuses to put the watermark at a
//!   word whose start ties the confirmed one in front of it, widening past the
//!   tie rather than cutting the clip boundary INSIDE a span already settled
//!   (see the Rule W comment at [`LocalAgreement::ingest`]'s advance).
//!
//!   **Postcondition** — whenever [`LocalAgreement::last_agreed_words_slice`] is
//!   non-empty, `confirmed_words.last().start() < last_agreed_seconds`
//!   STRICTLY. No confirmed word can satisfy the offered filter's own
//!   `start >= watermark`, so none can ever head a hypothesis and the
//!   re-admission question is unrepresentable rather than defended against.
//!   Adjudicated: Swift shares the bug, and "confirmed once and stable" wins
//!   over parity here.
//!
//!   **[BEHAVIOUR CHANGE]** the rule confirms one word EARLIER on a tied input,
//!   trading one round of revisability for a clip boundary that does not bisect
//!   a settled span — the same trade `budgeted_split` already makes, and
//!   [`LocalAgreement::agreement_count_needed`] becomes a maximum slightly more
//!   often. **Trigger:** two adjacent agreed words with equal `start`. On words
//!   this crate's own pipeline produces that requires a ZERO-DURATION word:
//!   `find_alignment` guarantees `w[i].end() <= w[i + 1].start() + 1e-4`
//!   (pinned in `crate::audio::whisper::segment`'s tests), so
//!   `w[i].start() >= w[i + 1].start()` forces `w[i].end() <= w[i].start() +
//!   1e-4`. The committed jfk golden carries no start tie at all and is
//!   byte-identical under the rule. Two hermetic sequences DO move, both
//!   LOSING a word, and both are pinned as characterization rather than
//!   repaired: `rule_w_deletes_a_tied_insertion_that_reproduces_nothing_
//!   confirmed` (finalized `" A X B C D"` becomes `" A B C D"`) and
//!   `rule_w_deletes_an_unaccounted_repeat_of_a_settled_word` (`" A A B C"`
//!   becomes `" A B C"`). Deletion is this module's non-preferred direction and
//!   the cost was weighed rather than missed: the alignment frontier Rule W
//!   replaces deleted words on PHRASE RECURRENCE — a shape real transcripts
//!   produce, present at two watermark positions of this crate's own canonical
//!   jfk phrase — while Rule W deletes only in a degenerate tie the driver
//!   cannot reach. Each test names
//!   <https://github.com/findit-studio/coremlit/issues/94> and carries the
//!   CORRECT behaviour in its failure message, so the day the trade is
//!   revisited the suite hands the next author the expectation.
//! - **The holdback holds only what the prefill carries WHOLE, with no
//!   residual.** Swift's `agreementCountNeeded` is a hardcoded `2` it never
//!   exposes, so the length case cannot arise there. Here
//!   [`LocalAgreement::agreement_count_needed`] is settable with no upper bound,
//!   while `prefill_tokens` keeps only the last `MAX_TOKEN_CONTEXT / 2` prefix
//!   tokens — so a large enough holdback would be silently truncated, and its
//!   head would be neither re-offered nor confirmed (codex round 6, finding 2).
//!   An advance therefore holds back only what fits [`MAX_HOLDBACK_PREFILL_TOKENS`]
//!   and CONFIRMS the rest; see `budgeted_split`.
//!
//!   The same split now also clears the prefill's OTHER reduction (codex round
//!   8, finding 1): `prefill_tokens` drops every id at or above the vocabulary's
//!   `special_token_begin`, and a word with no tokens contributes nothing to the
//!   prefix at all, so either kind of held word leaves the engine reasoning
//!   about text the decoder was never given — the premise
//!   [`LocalAgreement::decoding_options_for_next`]'s retarget makes. It was
//!   recorded as a residual for as long as the
//!   argument was "`add_word_timestamps` strips those ids from everything this
//!   crate emits", which is true and does not reach
//!   [`LocalAgreement::ingest`]'s public hand-built path. **[BEHAVIOUR CHANGE]**
//!   an advance takes such a word OUT of the holdback instead of holding it, so
//!   more words leave the holdback per advance and `finalize`'s text keeps a
//!   word it previously let a later hypothesis supersede. Unreachable through
//!   [`LocalAgreementTranscriber`], whose words come from
//!   `add_word_timestamps`.
//!
//!   **The threshold is the engine's, not a constant**
//!   ([`LocalAgreement::special_token_begin`] **[new public accessor,
//!   builder, and setter]**). It defaults to [`MIN_SPECIAL_TOKEN_BEGIN`]
//!   **[public const]**, the minimum `special_token_begin` over the vocabularies
//!   this crate is expected to load — a bound rather than the threshold, which is
//!   what a hermetic engine can hold on its own and the only direction that errs
//!   safely. Nothing makes that bound hold, though (codex round 12, finding 2):
//!   [`crate::audio::whisper::tokenizer::WhisperTokenizer::from_folder`] accepts
//!   any parseable `tokenizer.json` and probes `<|endoftext|>` for its own
//!   threshold, so an artifact below the floor walks straight back into round 8's
//!   defect. [`LocalAgreementTranscriber::new`] therefore hands the engine the
//!   loaded vocabulary's own value — exact, and nothing to remember; a caller
//!   driving `ingest` directly against such an artifact sets it itself.
//!   Rejecting the artifact at LOAD would close the same hole by refusing a
//!   vocabulary the rest of this crate decodes correctly, over a premise only
//!   this module has a stake in, so the loader is left alone. The default is
//!   today's constant, so no existing caller moves.
//!
//!   **A widened-past word is CONFIRMED on the spot.** The argument is round 8's:
//!   a word the prefill cannot carry is neither corroborable nor revisable by a
//!   continuation decoded under
//!   [`LocalAgreement::decoding_options_for_next`], being behind both the clip
//!   and the forced prefill, and the watermark passes it on the very next line,
//!   so no future result can ever be offered over its span. Holding it instead
//!   would be an indefinite wait ending in the deletion codex round 7 finding 2
//!   removed. A caller driving [`LocalAgreement::ingest`] with a result decoded
//!   some OTHER way could have re-read that audio, and for that caller the word
//!   is confirmed while a revision of it is still possible — the transcript then
//!   carries both readings. That is not a property of this arm:
//!   `common[..split]`, the mainline confirmation Swift has and this port has
//!   never touched, appends with no overlap test of any kind and lands in the
//!   same place whenever word ends inside a hypothesis are non-monotone
//!   (`an_overlapping_agreed_word_is_confirmed_on_the_mainline_path_too`). It is
//!   the LocalAgreement-2 contract itself — confirmation follows agreement
//!   between two consecutive hypotheses and is append-only.
//!
//!   That split runs all the way to `common.len()` when it has to, so **an
//!   advance can leave the holdback EMPTY** and the watermark anchored at the
//!   last confirmed word's end instead of the first held one's start. It has to:
//!   stopping while one word remained left a single word whose OWN tokens exceed
//!   the budget held anyway, and the cap silently did not cap (codex round 7,
//!   finding 2). What followed was data loss rather than a stall — the next
//!   hypothesis was decoded from a prefix `prefill_tokens` trims, came back with
//!   a word that is not the held one, disagreed, and
//!   [`LocalAgreement::finalize`]'s `holdback_superseded` path replaced the
//!   intact held word with that truncation. Confirming such a word is always
//!   possible and is no weaker a claim: `common` is the prefix two hypotheses
//!   agreed on, and a word outside the prefill budget is one no third hypothesis
//!   decoded from that prefill could revise — see the widened-past entry above
//!   for the qualifier that carries. Unreachable through
//!   [`LocalAgreementTranscriber`], which never leaves
//!   [`DEFAULT_AGREEMENT_COUNT_NEEDED`].
//! - **The final hypothesis's holdback** (Swift `:418-419`, `let final =
//!   lastAgreedWords + findLongestDifferentSuffix(prevWords,
//!   hypothesisWords)`). That decomposition is only valid when the LAST
//!   hypothesis agreed. `findLongestCommonPrefix` returns elements from its
//!   second argument, so on an advance `lastAgreedWords` *is* the final
//!   hypothesis's own `[split..commonPrefix.count]` slice and the sum
//!   reconstructs `hypothesisWords[split...]` exactly. When the last
//!   hypothesis DISAGREED, `lastAgreedWords` belongs to the hypothesis it just
//!   superseded, while `hypothesisWords` — filtered to `start >=
//!   lastAgreedSeconds` — already re-covers that same span carrying the
//!   revision, and Swift emits BOTH: the revised word lands beside the reading
//!   it replaced, and every word the two share is transcribed twice. With an
//!   empty holdback the same expression fails the other way, dropping the
//!   `commonPrefix.count` leading words both hypotheses actually produced.
//!   [`LocalAgreement::finalize`] instead emits the final hypothesis's own
//!   post-watermark words on that path (`holdback_superseded` is the flag), and
//!   keeps Swift's shape everywhere else — including when the final hypothesis
//!   contributes nothing at or past the watermark, where nothing supersedes the
//!   holdback. How much of the holdback that path actually replaces is the
//!   window's question, below; what is NOT replaced is emitted ahead of the
//!   hypothesis's words rather than sending the whole thing back to Swift's
//!   expression, whose prefix subtraction has no holdback to justify it once the
//!   holdback is the thing being kept (codex round 14). Same adjudication as the re-admission divergence recorded on
//!   `watermark_filtered`: Swift shares the bug, and a word confirmed once and
//!   stable wins over parity. Nothing in this repo pins the streaming transcript
//!   against a Swift capture (`tests/whisper/streaming.rs` "owns no golden"; the
//!   token-for-token goldens are the BATCH decode's), so the divergence costs no
//!   oracle comparison.
//! - **`use_prefill_prompt` is forced on the retargeted options, but only once
//!   there is a holdback to reproduce** (Swift `:364-367` sets only
//!   `clipTimestamps` and `prefixTokens`).
//!   [`DecodingOptions::prefix_tokens`](crate::audio::whisper::options::DecodingOptions::prefix_tokens_slice)
//!   reaches the decoder through exactly one call,
//!   [`crate::audio::whisper::decode::prefill_tokens`], which
//!   [`crate::audio::whisper::transcribe::WhisperKit::transcribe`] makes only when
//!   [`DecodingOptions::use_prefill_prompt`](crate::audio::whisper::options::DecodingOptions::use_prefill_prompt)
//!   is set — so on a base with the prompt off, the prefix
//!   [`LocalAgreement::decoding_options_for_next`] attaches is silently dropped
//!   and the stream re-decodes each span with no anchor at all. Both this port
//!   and Swift default the flag on, so this only diverges for a caller that
//!   turned it off; it is forced for the same reason
//!   [`LocalAgreementTranscriber::new`] forces `word_timestamps`: a prefix the
//!   decoder is never given makes `budgeted_split`'s whole budget argument
//!   vacuous, and leaves the next hypothesis with no anchor over the span the
//!   holdback covers.
//!
//!   It is forced only while [`LocalAgreement::last_agreed_words_slice`] is
//!   NON-EMPTY (codex round 6, finding 3). Before the first advance there is no
//!   holdback and the prefix is empty, so there is nothing for the flag to
//!   carry. Forcing the flag on those strides would
//!   change the caller's prompt from a bare `<|startoftranscript|>` to the full
//!   multilingual language/task/timestamp prefill for nothing, and a streaming
//!   caller would get different decoding behaviour before LocalAgreement had
//!   produced any state that justified the deviation.
//! - **`push_samples` needs only `B: InferenceBackend`, not `+ Sync`.**
//!   Its only backend-touching call is
//!   [`crate::audio::whisper::transcribe::WhisperKit::transcribe`], whose own `impl`
//!   block bound is `B: InferenceBackend` alone — `Sync` is
//!   `WhisperKit::transcribe_all`'s addition, for its concurrent worker
//!   pool (`crate::audio::whisper::transcribe`'s module doc, "Concurrency note"), and
//!   [`InferenceBackend`] itself has no `Sync` supertrait either. This is
//!   a correction against this task's own brief, which specified `B:
//!   InferenceBackend + Sync` here.

use crate::audio::whisper::{
  backend::InferenceBackend,
  constants::{MAX_TOKEN_CONTEXT, SAMPLE_RATE},
  error::TranscribeError,
  options::DecodingOptions,
  result::{TranscriptionResult, WordTiming, merge_transcription_results_with_words},
  task_facts::{SpanKnowledge, TaskFactsAccumulator},
  text::{find_longest_common_prefix, find_longest_different_suffix},
  tokenizer::MIN_SPECIAL_TOKEN_BEGIN,
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

/// The most holdback tokens a prefill can carry and have EVERY one of them
/// reach the decoder: [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens)
/// keeps only the LAST `MAX_TOKEN_CONTEXT / 2` elements of
/// [`DecodingOptions::prefix_tokens`](DecodingOptions::prefix_tokens_slice)
/// (`prefix_tokens.len().saturating_sub(MAX_TOKEN_CONTEXT / 2)` — Swift
/// `TextDecoder.swift:203`'s `.suffix`). Everything before that point is
/// dropped before the initial prompt is even assembled, so it never enters
/// `decode_text`'s `current_tokens` and never appears in the hypothesis.
///
/// That trim is silent, and the holdback is what
/// [`LocalAgreement::decoding_options_for_next`] promises the next hypothesis
/// will be WRITTEN with — so a holdback that cannot survive this budget is one
/// the decoder is never given, and the words the trim erases would be neither
/// re-offered nor confirmed. [`LocalAgreement::ingest`] therefore holds back
/// only what fits, and
/// `budgeted_split` guarantees that for EVERY input rather than for every input
/// but one: where nothing can be held it holds nothing, and the advance confirms
/// the whole agreed prefix.
///
/// The trim is a LENGTH bound only, and it is not the only way `prefill_tokens`
/// reduces a prefix: it also drops every id at or above the vocabulary's
/// `special_token_begin`, and contributes nothing at all for a word carrying no
/// tokens. `budgeted_split` tests each candidate word against
/// [`LocalAgreement::special_token_begin`] — the loaded vocabulary's own value
/// where the engine was told it, and [`MIN_SPECIAL_TOKEN_BEGIN`] otherwise — and
/// widens past every word that fails, exactly as it widens past an over-budget
/// one. So no prefill this engine issues is ever trimmed OR filtered.
///
/// That id filter used to be recorded here as a residual, on the evidence that
/// `add_word_timestamps` strips exactly those ids from every [`WordTiming`] it
/// emits (`segment::update_segments_with_word_timings`, Swift
/// `SegmentSeeker.swift:551-554`; an all-special timing emits no word at all).
/// The evidence is correct — this crate's pipeline never produces such a word —
/// but it does not cover [`LocalAgreement::ingest`], which is public and takes a
/// hand-built [`TranscriptionResult`]. That path could hold back a filtered
/// word, honestly pass the retarget, and still be recorded `prefilled`
/// (codex round 8, finding 1); see `budgeted_split`.
pub const MAX_HOLDBACK_PREFILL_TOKENS: usize = MAX_TOKEN_CONTEXT / 2;

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
  /// The previous hypothesis, kept RAW so each ingest can re-filter it against
  /// the watermark that is current then rather than the one current when it
  /// arrived.
  prev_result: Option<TranscriptionResult>,
  prev_words: Vec<WordTiming>,
  hypothesis_words: Vec<WordTiming>,
  last_agreed_words: Vec<WordTiming>,
  /// Whether the most recent WORDED hypothesis failed to corroborate
  /// [`Self::last_agreed_words`] — i.e. that holdback belongs to a hypothesis
  /// the latest one has since superseded. [`Self::finalize`] needs this and
  /// cannot recover it from the word lists alone; see the divergence recorded
  /// there and in this module's doc.
  ///
  /// Maintained ONLY on the worded path of [`Self::ingest`], alongside
  /// [`Self::prev_words`]/[`Self::hypothesis_words`]/[`Self::last_agreed_words`]
  /// themselves: the [`AgreementOutcome::NoWordTimings`] early return leaves all
  /// four untouched, so this keeps describing the last hypothesis that actually
  /// had words to agree over — exactly the pair `finalize` reasons about.
  holdback_superseded: bool,
  confirmed_words: Vec<WordTiming>,
  results: Vec<TranscriptionResult>,
  /// A sink for the reproducibility facts of EVERY ingested hypothesis —
  /// including the disagreeing ones dropped from [`Self::results`] but retained
  /// as [`Self::prev_result`] to CONTROL the next agreement comparison (codex
  /// round 8, F1). The same error-drop-sink pattern the VAD branch uses: a
  /// dropped hypothesis's unseeded draw (or callback truncation) still decided
  /// which words the surviving hypotheses agreed on, so it must reach
  /// [`Self::finalize`]'s reproducibility answer even though its segments never
  /// survive into the merge. Only the draw/early-stop/language facts are folded;
  /// the worker schedule and id span are stripped to `None` (see the strip in
  /// [`Self::ingest`]) — the finalized schedule is the adjudicated `None` and the
  /// finalized span is restored from the merged surviving result (round 10).
  ingested_facts: TaskFactsAccumulator,
  /// The vocabulary threshold `budgeted_split` tests a candidate held-back
  /// word's ids against — see [`Self::special_token_begin`].
  special_token_begin: u32,
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
      holdback_superseded: false,
      confirmed_words: Vec::new(),
      results: Vec::new(),
      ingested_facts: TaskFactsAccumulator::new(),
      special_token_begin: MIN_SPECIAL_TOKEN_BEGIN,
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
  /// least `1`. Zero would hold back no words at all on EVERY advance
  /// (`Self::ingest`'s `common[split..]` slice with `split ==
  /// common.len()`), so no hypothesis would ever be given an anchor to
  /// re-decode from and LocalAgreement-2's second round of corroboration would
  /// be switched off wholesale — an algorithmically degenerate
  /// configuration Swift's hardcoded `agreementCountNeeded = 2`
  /// (`TranscribeCLI.swift:349`) never reaches, since Swift never exposes
  /// this knob as configurable at all; its own `lastAgreedWords.first!`
  /// (`:385`) would force-unwrap-crash on the same input if it somehow
  /// did. This port would not: an empty holdback is a state `ingest` handles
  /// (`budgeted_split` can produce one for a word the prefill cannot carry, and
  /// the watermark anchor covers it), so the clamp is about keeping the
  /// ALGORITHM meaningful rather than about keeping it from panicking.
  #[inline(always)]
  pub const fn set_agreement_count_needed(&mut self, agreement_count_needed: usize) -> &mut Self {
    self.agreement_count_needed = if agreement_count_needed == 0 {
      1
    } else {
      agreement_count_needed
    };
    self
  }

  // -- special_token_begin ---------------------------------------------------
  /// The vocabulary's first special-token id, as this engine understands it:
  /// `budgeted_split` holds back a word only when every one of its token ids is
  /// BELOW this, because
  /// [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens) drops the
  /// rest before the decoder is given a single one of them.
  ///
  /// Defaults to [`MIN_SPECIAL_TOKEN_BEGIN`], the minimum over the vocabularies
  /// this crate is expected to load — a BOUND, which is all a hermetic engine
  /// can hold on its own and is the only direction that errs safely (see that
  /// constant's own doc). The bound is not an invariant, though:
  /// [`crate::audio::whisper::tokenizer::WhisperTokenizer::from_folder`] loads
  /// any parseable `tokenizer.json` and probes `<|endoftext|>` for its own
  /// threshold, so an artifact whose threshold is LOWER makes the default an
  /// over-estimate — the engine would hold a word whose ids that artifact's
  /// `prefill_tokens` erases, so the prefill the engine promises the next
  /// hypothesis is one the decoder is given only part of (codex round 12,
  /// finding 2).
  ///
  /// So it is a value rather than a constant. [`LocalAgreementTranscriber::new`]
  /// sets it from the pipeline's own loaded vocabulary, which is exact and needs
  /// nothing remembered; a caller driving [`Self::ingest`] directly against an
  /// unusual artifact sets it with [`Self::with_special_token_begin`]. Leaving
  /// it alone is exactly the behaviour every release before this one had.
  #[inline(always)]
  pub const fn special_token_begin(&self) -> u32 {
    self.special_token_begin
  }
  /// Builder form of [`Self::set_special_token_begin`].
  #[must_use]
  #[inline(always)]
  pub const fn with_special_token_begin(mut self, special_token_begin: u32) -> Self {
    self.set_special_token_begin(special_token_begin);
    self
  }
  /// Sets [`Self::special_token_begin`] in place. Unclamped, and deliberately:
  /// a value ABOVE the deciding vocabulary's own threshold makes the engine hold
  /// a word the decoder never receives, and a value below it only confirms a
  /// word a round earlier than it had to. The caller owns the artifact, so the
  /// caller supplies the fact, and it is checked where it can be:
  /// [`LocalAgreementTranscriber`] reads it off the vocabulary itself.
  #[inline(always)]
  pub const fn set_special_token_begin(&mut self, special_token_begin: u32) -> &mut Self {
    self.special_token_begin = special_token_begin;
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
  /// Word timings settled so far: every agreement's leading remainder, ahead of
  /// that agreement's own [`Self::agreement_count_needed`]-word holdback.
  ///
  /// Append-only across the life of the engine: nothing here is ever rewritten,
  /// reordered, or taken back, which is why the trailing words of an agreement
  /// wait in [`Self::last_agreed_words_slice`] instead. RULE W (see
  /// [`Self::ingest`]'s advance) additionally guarantees that while that
  /// holdback is non-empty, this list's last word starts STRICTLY before
  /// [`Self::last_agreed_seconds`] — so nothing here can be re-offered to the
  /// agreement comparison and confirmed a second time.
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
  ///
  /// # The prefill is a contract, not a hint
  ///
  /// [`DecodingOptions::prefix_tokens`](DecodingOptions::prefix_tokens_slice) is
  /// read by exactly one place,
  /// [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens), and
  /// [`crate::audio::whisper::transcribe::WhisperKit::transcribe`] only CALLS
  /// that function when [`DecodingOptions::use_prefill_prompt`] is set — so on a
  /// `base` with the prompt turned off, the prefix tokens this method attaches
  /// are silently inert and the returned options are not retargeted at all.
  /// This therefore sets it, the same way
  /// [`LocalAgreementTranscriber::new`] forces
  /// [`DecodingOptions::word_timestamps`] and for the same kind of reason: the
  /// option is not a preference here, it is what makes the returned value mean
  /// what its name says. **Documented deviation** — Swift `:364-367` sets only
  /// the two fields and inherits `usePrefillPrompt` from the CLI flag, so a
  /// `whisperkit-cli` run without `--use-prefill-prompt` streams with an inert
  /// `prefixTokens` exactly as described. It defaults `true` on both sides, so
  /// this only diverges for a caller that turned it off.
  ///
  /// What the prefill buys is that the next hypothesis REPRODUCES the holdback
  /// rather than predicting it — which is what makes an advance's `common` a
  /// re-agreement over the same span rather than a fresh reading of it, and what
  /// `budgeted_split`'s budget argument is about. `prefill_tokens`
  /// appends these tokens to the initial prompt, and `decode_text` FORCES every
  /// prompt position
  /// (`next_token = current_tokens[token_index]` for `token_index <
  /// initial_prompt_index`) before `finalize_decoding_result` keeps the whole
  /// `SOT..=EOT` span — so the holdback is not something the next hypothesis
  /// might predict, it is text the engine wrote into that hypothesis. Combined
  /// with `clip_timestamps`, which puts the audio before the watermark outside
  /// the decoded window entirely, a hypothesis produced from these options
  /// BEGINS with a reproduction of the holdback, and nothing already confirmed
  /// can precede it.
  ///
  /// # Before the first advance
  ///
  /// With an empty [`Self::last_agreed_words_slice`] there is nothing to
  /// reproduce: the prefix is empty, `clip_timestamps` is at the watermark, and
  /// [`DecodingOptions::use_prefill_prompt`] is left exactly as `base` had it.
  /// The forcing above exists so the prefix the engine attaches actually
  /// reaches the decoder, and there is no prefix on this path. Overriding the
  /// caller's flag here
  /// would change the initial prompt from a bare `<|startoftranscript|>` to the
  /// full multilingual language/task/timestamp prefill for no reason the engine's
  /// own state can point at (codex round 6, finding 3), and there would be no
  /// prefix for it to carry in any case.
  pub fn decoding_options_for_next(&self, base: &DecodingOptions) -> DecodingOptions {
    let retargeted = base
      .clone()
      .with_clip_timestamps(vec![self.last_agreed_seconds])
      .with_prefix_tokens(self.holdback_prefill_tokens());
    if self.last_agreed_words.is_empty() {
      // NOTHING IS HELD BACK, so the prefix is empty and the caller's own flag
      // stands. Forcing the flag here would change the prompt
      // from whatever the caller asked for to the full multilingual
      // language/task/timestamp prefill, on strides where the engine holds no
      // state that needs the deviation (codex round 6, finding 3). The prefix
      // above is empty on this path, so nothing is silently dropped by leaving
      // the prompt off either.
      retargeted
    } else {
      retargeted.with_use_prefill_prompt()
    }
  }

  /// The holdback as one token sequence — the exact value
  /// [`Self::decoding_options_for_next`] attaches as
  /// [`DecodingOptions::prefix_tokens`](DecodingOptions::prefix_tokens_slice).
  fn holdback_prefill_tokens(&self) -> Vec<u32> {
    self
      .last_agreed_words
      .iter()
      .flat_map(|word| word.tokens_slice().iter().copied())
      .collect()
  }

  /// The agreement view of a result's words: everything at or past the
  /// watermark — `TranscribeCLI.swift:372`/`:375` verbatim,
  /// `result.allWords.filter { $0.start >= lastAgreedSeconds }`, with no strip,
  /// no scope and no alignment behind it.
  ///
  /// It needs none. The watermark is the first held-back word's start, and
  /// RULE W (see [`Self::ingest`]'s advance) refuses to put it at a word whose
  /// start ties the confirmed one in front of it — so whenever there is a
  /// holdback at all, `confirmed_words.last().start() < last_agreed_seconds`
  /// STRICTLY and no confirmed word can pass this filter. The re-admission the
  /// issue is about is unrepresentable rather than detected, which is why this
  /// is the Swift line and not a rule.
  ///
  /// What it deliberately leaves is the same short list the postcondition
  /// bounds, recorded in this module's doc and each with a named test:
  /// a repeat the engine's record cannot account for is read as the stream's
  /// own; an empty holdback anchors the watermark at the last confirmed word's
  /// END, so a zero-duration word there can still tie it; and a re-decode free
  /// to move every timestamp it emits can push a settled word past the
  /// watermark, where it reads as new speech.
  fn watermark_filtered(result: &TranscriptionResult, watermark: f32) -> Vec<WordTiming> {
    result
      .all_words()
      .into_iter()
      .filter(|word| word.start() >= watermark)
      .collect()
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
  ///   [`crate::audio::whisper::text::find_longest_common_prefix`] (`:376`). A common
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
    // must not vanish with its segments.
    //
    // Worker schedule and id span are stripped here. For the SCHEDULE this is the
    // ADJUDICATED agreement contract (round 10, F2): agreement-confirmed text
    // interleaves words from MULTIPLE hypotheses, so no single ordered worker
    // attribution is knowable — every contributor is `None`, which under the
    // absorbing-`None` law is exactly what the finalized aggregate must read back
    // (`finalize` leaves it there). For the SPAN the strip to the wholly-unknown
    // `AtLeast(0)` keeps `ingested_facts` from summing dropped hypotheses'
    // ordinals; `finalize` overwrites it with the merged surviving result's own
    // span, the authoritative id-ordinal count.
    self.ingested_facts.merge(
      &result
        .task_facts()
        .clone()
        .with_worker_schedule(None)
        .with_decoded_span(SpanKnowledge::wholly_unknown()),
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

    // :372 verbatim — see `watermark_filtered`, and Rule W below for why the
    // bare filter is sound.
    self.hypothesis_words = Self::watermark_filtered(&result, self.last_agreed_seconds);

    let mut advanced = false;
    let mut skip_append = false;
    // :374 — absent on the first-ever call, so nothing below runs and
    // this falls through to the `AwaitingAgreement` append below.
    if let Some(previous) = &self.prev_result {
      // :375-376 — the same filter as the hypothesis, against the CURRENT
      // watermark, so the two sides stay index-aligned for the prefix
      // comparison. `prev_result` is kept RAW for exactly this: the watermark it
      // is re-read against is the one current NOW, not the one current when it
      // arrived.
      self.prev_words = Self::watermark_filtered(previous, self.last_agreed_seconds);
      let common = find_longest_common_prefix(&self.prev_words, &self.hypothesis_words);
      if common.len() >= self.agreement_count_needed {
        // :383-394 — advance the watermark.
        let requested = common.len() - self.agreement_count_needed;
        let mut split = budgeted_split(common, requested, self.special_token_begin);
        // RULE W -- THE SPLIT MAY NOT CUT AT A TIED START (#94, at its source).
        //
        // The watermark is the first held-back word's start, and it is also the
        // CLIP this engine hands its own next decoder. Cutting at a word whose
        // start TIES the last confirmed one puts that boundary INSIDE a span
        // already settled: the confirmed word then satisfies the offered filter's
        // own `start >= watermark`, and the next hypothesis can re-offer it at
        // the head of its word list. That is the state every re-admission defence
        // in this issue's history was built to survive -- and the one that cannot
        // be decided from the offered list, because a re-offered settled word and
        // the stream's own second occurrence of the same text are byte-identical
        // there. Refuse to CREATE it: widen past the tie instead.
        //
        // Postcondition: whenever `last_agreed_words` is non-empty,
        // `confirmed_words.last().start() < last_agreed_seconds` strictly, so no
        // confirmed word can ever pass the offered filter again.
        //
        // The anchor is the word the watermark would be measured against: the
        // last word the split has already moved past (`common[split - 1]`) when
        // it has moved past anything, and otherwise the engine's own last
        // confirmed word, which is what the watermark would then sit beside. It
        // is carried forward through the loop, so a RUN of tied starts is
        // cleared in one pass rather than one word of it.
        //
        // Composes with `budgeted_split` in one direction only, which is why it
        // runs after it: widening can only SHRINK the holdback `common[split..]`,
        // so the token budget it just established still holds, and its id-filter
        // floor is likewise never re-crossed (see `budgeted_split`). Where the
        // tie runs to the end of `common` the holdback empties and the watermark
        // falls back to `common.last().end()`, which is at or past every start
        // in it -- the postcondition is then vacuous, and the next advance
        // re-seeds the anchor from the confirmed list.
        let mut anchor = if split > 0 {
          Some(common[split - 1].start())
        } else {
          self.confirmed_words.last().map(WordTiming::start)
        };
        while split < common.len() && anchor.is_some_and(|tied| tied >= common[split].start()) {
          anchor = Some(common[split].start());
          split += 1;
        }
        // `common` REPLACES the still-open record: it is the span two consecutive
        // hypotheses have just re-agreed over it, and `last_agreed_words` is the
        // one this hypothesis has superseded.
        self.confirmed_words.extend_from_slice(&common[..split]);
        self.last_agreed_words = common[split..].to_vec();
        // The watermark is the first held-back word's start -- except when the
        // budget could hold NOTHING (see `budgeted_split`), where the still-open
        // span begins where the confirmed one ends. `common` is non-empty here
        // (its length is at least `agreement_count_needed`, clamped to at least
        // one), so the final fallback is unreachable and only keeps this total.
        // Monotone either way: every word of `common` starts at or past the old
        // watermark, and `end >= start`.
        //
        // That empty-holdback anchor is Rule W's one gap, and it is recorded as
        // a residual rather than closed: `common.last().end()` can EQUAL
        // `common.last().start()` for a zero-duration word, and the
        // postcondition is then vacuous because there is no held-back word for
        // it to speak about. Reaching it needs a non-default
        // `agreement_count_needed`, a word whose own tokens exceed
        // `MAX_HOLDBACK_PREFILL_TOKENS`, and a zero-duration word, all at once
        // (`an_empty_holdback_leaves_a_zero_duration_word_at_the_watermark`).
        self.last_agreed_seconds = self.last_agreed_words.first().map_or_else(
          || {
            common
              .last()
              .map_or(self.last_agreed_seconds, WordTiming::end)
          },
          WordTiming::start,
        );
        advanced = true;
      } else {
        // :395-400 — disagreement; `result` is dropped below.
        skip_append = true;
      }
    }

    // The holdback is stale exactly when THIS hypothesis failed to corroborate
    // it — the same condition as `skip_append`, recorded under its own name
    // because its consumer is `finalize`, not `results`. Set here rather than in
    // the branches above so the first-ever call (no `prev_result`, no agreement
    // run at all) also lands on `false`: nothing has been held back yet, so
    // nothing can have been superseded.
    self.holdback_superseded = skip_append;

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
  /// ([`crate::audio::whisper::text::find_longest_different_suffix`] over the last
  /// ingested pair), both folded onto [`Self::confirmed_words_slice`] —
  /// **except when the final hypothesis DISAGREED**, where this port emits that
  /// hypothesis's own post-watermark words instead of the superseded holdback
  /// (see this module's doc, "The final hypothesis's holdback");
  /// [`merge_transcription_results_with_words`] then merges every kept
  /// [`Self::results_slice`] result with that word list as the merged
  /// text, under `options` — the same options the kept results were decoded
  /// with, so the merged segments honor
  /// [`DecodingOptions::drop_blank_audio`]'s id mapping (which the confirmed
  /// text override does not touch, but the segments still carry).
  ///
  /// The reproducibility facts of EVERY ingested hypothesis — including the
  /// disagreeing ones dropped from [`Self::results_slice`] — are carried on the
  /// finalized record from `Self::ingested_facts` (codex round 8, F1), so a
  /// dropped control hypothesis's unseeded draw or callback truncation is not
  /// lost from the reproducibility answer. That ingest-ordered sink is the fold
  /// BASE, with the merged result's own facts folded IN (codex round 9): its
  /// FIRST-observed language then wins over a later surviving result's, where
  /// folding the sink last let the survivor's win (F3). The worker schedule is
  /// the adjudicated `None` (agreement attribution is unknown, round 10, F2) and
  /// the decoded span is restored from the merged surviving result (round 10, F3;
  /// the ingest strip carries only the wholly-unknown span, so the fold cannot
  /// supply the exact count, round 12); see [`Self::ingest`].
  pub fn finalize(mut self, options: &DecodingOptions) -> TranscriptionResult {
    if self.holdback_superseded && !self.hypothesis_words.is_empty() {
      // DIVERGENCE from `:418-419` — see this module's doc for the full
      // argument. Swift's `lastAgreedWords + differentSuffix(prevWords,
      // hypothesisWords)` is only a valid decomposition when the final
      // hypothesis AGREED. Here it did not: `last_agreed_words` belongs to the
      // hypothesis this one just superseded, while `hypothesis_words` — filtered
      // to `start >= last_agreed_words[0].start()` — already re-covers that exact
      // span carrying the revision. Emitting both duplicates the span and strands
      // the superseded reading beside its own replacement; emitting only the
      // SUFFIX would instead drop the leading words both hypotheses produced,
      // which is the same defect's other face when the holdback is empty.
      //
      // The non-empty guard is load-bearing: a result whose every word falls
      // BEFORE the watermark still clears the `has_words` gate (`:371`) and
      // disagrees, but it re-covers nothing, so there is no revision to prefer
      // and the provisional holdback remains the only estimate for that span.
      // That case stays on the Swift shape below, byte-identical.
      self.confirmed_words.append(&mut self.hypothesis_words);
    } else {
      // `:418-419` verbatim.
      self.confirmed_words.append(&mut self.last_agreed_words);
      let suffix = find_longest_different_suffix(&self.prev_words, &self.hypothesis_words);
      self.confirmed_words.extend_from_slice(suffix);
    }
    let mut merged =
      merge_transcription_results_with_words(&self.results, &self.confirmed_words, options);
    // Fold the merged (surviving-result) facts INTO the ingest-ordered sink, not
    // the other way round (codex round 9): the sink observed EVERY hypothesis in
    // ingest order — the disagreeing dropped ones included — so its FIRST-observed
    // language must win over a later surviving result's, which merging the sink
    // last reversed (F3). The draw/early-stop Kleene OR is commutative, so their
    // answer is unchanged by the order.
    //
    // The DECODED SPAN is then restored from the merged surviving result: the sink
    // stripped its span to the wholly-unknown `AtLeast(0)` at ingest, so the fold
    // would only lower-bound the merged span (round 12: the strip no longer
    // absorbs, but it also carries no exact count), losing the authoritative
    // id-ordinal count a staged re-merge needs. Overwriting with the merged
    // surviving result's own span restores it exactly. The WORKER SCHEDULE is
    // deliberately left at the `None` the strip and the absorbing merge produce —
    // ADJUDICATED (round 10, F2): agreement-confirmed text interleaves multiple
    // hypotheses, so no single ordered worker attribution is knowable. See the
    // strip site in [`Self::ingest`].
    let merged_span = merged.task_facts().decoded_span();
    let mut facts = self.ingested_facts.into_facts();
    facts.merge(merged.task_facts());
    *merged.task_facts_mut() = facts.with_decoded_span(merged_span);
    merged
  }
}

// ---------------------------------------------------------------------
// The confirmed/holdback split
// ---------------------------------------------------------------------

/// Where an advance splits `common` into the part that is CONFIRMED and the
/// part that is HELD BACK, given the requested split — moved later until every
/// word still held is one [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens)
/// carries into the initial prompt WHOLE.
///
/// The holdback is not merely "the last few agreed words": it is the text
/// [`LocalAgreement::decoding_options_for_next`] forces into the next
/// hypothesis, and `prefill_tokens` reduces `prefix_tokens` on its way there in
/// two independent ways — it keeps only the last [`MAX_HOLDBACK_PREFILL_TOKENS`]
/// ids, and it drops every id at or above the vocabulary's
/// `special_token_begin`. A holdback the decoder cannot be given whole is not a
/// holdback at all — the words the filter erases would be neither reproduced
/// (the decoder never sees their tokens) nor confirmed (an advance replaces the
/// holdback with the new `common[split..]`), so they would simply vanish from
/// the transcript.
///
/// **Both filters are enforced here, and the id one is enforced against
/// `special_token_begin`** (codex round 8, finding 1) — the deciding
/// vocabulary's own threshold, which
/// [`LocalAgreement::special_token_begin`] carries and
/// [`LocalAgreementTranscriber::new`] reads off the pipeline's loaded tokenizer.
/// A word with no tokens at all fails the same test for a different reason: it
/// contributes nothing to `prefix_tokens`, so the decoder is never given it
/// either, and no threshold is needed to see that.
///
/// The engine holds no tokenizer of its own, so with nothing supplied that value
/// is [`MIN_SPECIAL_TOKEN_BEGIN`] — a BOUND rather than the threshold, which is
/// what a hermetic engine can hold and which errs in the only safe direction.
/// The bound is not self-enforcing (codex round 12, finding 2):
/// [`crate::audio::whisper::tokenizer::WhisperTokenizer::from_folder`] accepts
/// any parseable `tokenizer.json` and probes `<|endoftext|>` for its threshold,
/// so an artifact below the floor turns the bound into an over-estimate and
/// walks straight back into round 8's defect — a word held on the strength of a
/// filter that will erase it. Rejecting such an artifact at load would fix it by
/// refusing a vocabulary this crate otherwise decodes correctly, for a premise
/// only this module has a stake in; supplying the real value fixes it where the
/// value is known and costs nothing where it is not. See
/// [`LocalAgreement::special_token_begin`].
///
/// That this engine could not evaluate the id filter used to be recorded as a
/// residual, on the evidence that `add_word_timestamps` strips exactly those ids
/// from every [`WordTiming`] it emits (and emits no word at all for an
/// all-special alignment entry). The evidence is correct and the pipeline really
/// is clean. What it does not cover is [`LocalAgreement::ingest`] itself, which
/// is public and takes a hand-built [`TranscriptionResult`], so a caller can
/// hold back a word carrying filtered ids and then use
/// [`LocalAgreement::decoding_options_for_next`] honestly, leaving the engine
/// promising the decoder text it will never be given.
///
/// Widening the split instead takes that head OUT of the holdback and CONFIRMS
/// it. That is not a weaker claim than any other agreed word carries: `common`
/// is the prefix two consecutive hypotheses agreed on, which is the whole of
/// LocalAgreement-2's criterion, and [`LocalAgreement::finalize`] already
/// appends the entire holdback to
/// [`LocalAgreement::confirmed_words_slice`] unconditionally on its Swift-shaped
/// path. What the holdback buys on top of that is one more round in which a
/// third hypothesis could revise it — and a word the prefill cannot carry cannot
/// be revised by one *that was decoded from the prefill*, because whatever such
/// a hypothesis produces over that extent came from a DIFFERENT prefix and from
/// audio the clip excludes, and is therefore neither a corroboration of the held
/// word nor a revision of it. A caller driving [`LocalAgreement::ingest`] with a
/// result decoded some OTHER way is subject to neither reduction, so for it the
/// word is revisable after all and the confirmation lands beside the revision —
/// the same append-only cost `common[..split]` already carries on every path
/// (`an_overlapping_agreed_word_is_confirmed_on_the_mainline_path_too`).
///
/// Widening is the repair because the defect is the STATE, not any reading of
/// it. Leaving the unreproducible word IN the holdback is what round 7's
/// finding 2 recorded: the next unanchored hypothesis disagrees with it and
/// [`LocalAgreement::finalize`]'s `holdback_superseded` path deletes it.
///
/// The split runs all the way to `common.len()` when it has to, so the holdback
/// this leaves can be EMPTY. It has to (codex round 7, finding 2): stopping while
/// one word remained still held a single word whose OWN tokens exceed the budget,
/// and the cap silently did not cap. What followed was data
/// loss, not a stall: the next hypothesis came back with the truncated word
/// rather than the held one, disagreed, and
/// [`LocalAgreement::finalize`]'s `holdback_superseded` path replaced the intact
/// held word with that truncation. Made impossible here rather than refused
/// downstream, because a refusal on a public, infallible `ingest` has no path to
/// report on, and this needs none: taking the word out of the holdback is always
/// available and is exactly the argument above.
///
/// Where the holdback comes back empty, [`LocalAgreement::ingest`] anchors the
/// watermark at the last confirmed word's END rather than the first held word's
/// start — see the anchor at its advance branch.
/// [`LocalAgreement::agreement_count_needed`] is then a maximum that reached
/// zero for that round, the same way it becomes a maximum for any holdback the
/// budget shortens.
///
/// **Documented deviation**: with `agreement_count_needed` at its
/// [`DEFAULT_AGREEMENT_COUNT_NEEDED`] — the only value
/// [`LocalAgreementTranscriber`] can reach, since it exposes
/// [`LocalAgreementTranscriber::agreement`] by shared reference only — a
/// two-word holdback is nowhere near 112 tokens and this is the identity. It
/// bites only for a direct caller that raised
/// [`LocalAgreement::agreement_count_needed`] far enough, and for that caller
/// the count becomes a maximum rather than an exact width.
fn budgeted_split(common: &[WordTiming], requested: usize, special_token_begin: u32) -> usize {
  // THE ID FILTER FIRST, as a floor. `prefill_tokens` drops ids at or above the
  // vocabulary's `special_token_begin` and contributes nothing at all for a word
  // with no tokens, so a holdback containing either is one the decoder is not
  // given whole -- and the split has to clear the LAST such word, not the first,
  // since the holdback is `common[split..]` and only a split past a word removes
  // it. Widening past it can never re-introduce one, so this floor and the
  // budget loop below compose in one direction: the loop only ever moves `split`
  // further right.
  let mut split = common[requested..]
    .iter()
    .rposition(|word| !prefill_carries_whole(word, special_token_begin))
    .map_or(requested, |last| requested + last + 1);
  let mut tokens: usize = common[split..]
    .iter()
    .map(|word| word.tokens_slice().len())
    .sum();
  // `split < common.len()`, not `split + 1 < common.len()`: stopping while one
  // word was still held is what let a single oversized word through (codex round
  // 7, finding 2). The empty holdback costs zero tokens, so this loop's
  // postcondition -- `tokens <= MAX_HOLDBACK_PREFILL_TOKENS` -- now holds for
  // every input rather than for every input but one.
  while tokens > MAX_HOLDBACK_PREFILL_TOKENS && split < common.len() {
    tokens -= common[split].tokens_slice().len();
    split += 1;
  }
  split
}

/// Whether [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens)
/// carries this word's tokens into the initial prompt WHOLE — the property every
/// held-back word has to have for
/// [`LocalAgreement::decoding_options_for_next`]'s retarget to promise the
/// decoder text it will actually be given.
///
/// Two ways to fail, and the empty one needs no vocabulary knowledge at all: a
/// word with NO tokens contributes nothing to `prefix_tokens`, so a prefix equal
/// to the holdback is equal to a sequence that never mentions it. The other is
/// the id filter, tested against `special_token_begin` — the deciding
/// vocabulary's own threshold where the engine was told it
/// ([`LocalAgreement::special_token_begin`]), and otherwise the floor
/// [`MIN_SPECIAL_TOKEN_BEGIN`]; see `budgeted_split` for why over-estimating the
/// special range is the safe direction.
fn prefill_carries_whole(word: &WordTiming, special_token_begin: u32) -> bool {
  let tokens = word.tokens_slice();
  !tokens.is_empty() && tokens.iter().all(|&id| id < special_token_begin)
}

// ---------------------------------------------------------------------
// LocalAgreementTranscriber
// ---------------------------------------------------------------------

/// Samples per stride: 1 s at [`SAMPLE_RATE`] — Swift's `16000` stride
/// literal (`TranscribeCLI.swift:357`). See this module's doc for how this
/// port's cursor start differs from Swift's induction variable.
pub const STRIDE_SAMPLES: usize = SAMPLE_RATE as usize;

/// The simulated-stream driver: feeds a growing audio buffer through
/// [`crate::audio::whisper::transcribe::WhisperKit::transcribe`] one [`STRIDE_SAMPLES`]
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
/// [`crate::audio::whisper::stream::AudioStreamTranscriber`]'s own two-impl-block split).
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
  ///
  /// Also hands the engine `kit`'s own vocabulary threshold. The engine's
  /// default, [`MIN_SPECIAL_TOKEN_BEGIN`], is a bound over the vocabularies this
  /// crate is expected to load, and nothing makes an artifact honour it — but
  /// this driver holds the very tokenizer whose
  /// [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens) call will
  /// apply the filter the bound is standing in for, so on this path the exact
  /// value is free and nothing has to be remembered (codex round 12,
  /// finding 2). See [`LocalAgreement::special_token_begin`].
  pub fn new(kit: &'ctx WhisperKit<B>, options: DecodingOptions) -> Self {
    Self {
      kit,
      options: options.with_word_timestamps(),
      agreement: LocalAgreement::new()
        .with_special_token_begin(kit.tokenizer().special_tokens().special_token_begin()),
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
  /// ([`crate::audio::whisper::transcribe::WhisperKit::transcribe`], with options
  /// retargeted per [`LocalAgreement::decoding_options_for_next`]) and
  /// folds the result through [`LocalAgreement::ingest`]. Ports
  /// `TranscribeCLI.swift:357-369`.
  ///
  /// # Errors
  /// Whatever [`crate::audio::whisper::transcribe::WhisperKit::transcribe`] returns,
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
