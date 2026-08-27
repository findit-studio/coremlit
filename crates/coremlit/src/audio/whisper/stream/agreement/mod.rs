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
//!   `LocalAgreement::ingest` ever sees it — "any segment has a
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
//!   word whose start ties the confirmed one in front of it, cutting the clip
//!   boundary INSIDE a span already settled (see `split_at_a_strict_boundary`).
//!
//!   **Postcondition (TOTAL)** — after every advance,
//!   `confirmed_words.last().start() < last_agreed_seconds` STRICTLY, with no
//!   condition on the holdback, so the LAST confirmed word cannot satisfy the
//!   offered filter's own `start >= watermark` and cannot head a hypothesis. The
//!   re-admission question is unrepresentable rather than defended against.
//!   Adjudicated: Swift shares the bug, and "confirmed once and stable" wins
//!   over parity here.
//!
//!   Two things carry it, and they are separable. `split_at_a_strict_boundary`
//!   puts an INTERIOR split only on a boundary whose preceding word starts
//!   strictly earlier — searching forward from the requested split first, then
//!   BACKING OFF when the forward search would run off the end of `common`,
//!   because widening past a tied run that reaches that end would empty the
//!   holdback and anchor the watermark on the run's own last word, and finally
//!   DEFERRING the round outright — where the prefill budget floor sits at or
//!   above every legal boundary, which is where the back-off has nowhere legal
//!   to land, and where the budget FORCES the empty holdback but the watermark
//!   that advance would set lies past a word the newer hypothesis produced
//!   beyond `common`. And where the holdback is empty anyway — a single word the
//!   prefill cannot carry WHOLE is the only thing that can still do that, and
//!   only while it strands nothing — `LocalAgreement::ingest` anchors the
//!   watermark at `empty_holdback_watermark`'s `end.max(start.next_up())` rather
//!   than at `end`, since `end == start` for a zero-duration word. `next_up` is
//!   the IMMEDIATE f32 successor: it refuses exactly the one instant the
//!   confirmed word occupies and no span of instants, which an `end + epsilon`
//!   tolerance would not have managed.
//!
//!   The DEFERRAL closes a DELETION rather than a re-confirmation (codex round 3
//!   on PR #95). A tied run whose own tokens exceed
//!   [`MAX_HOLDBACK_PREFILL_TOKENS`] — 113 ORDINARY one-token words sharing a
//!   start, which `add_word_timestamps` produces from an ALL-ZERO alignment
//!   matrix, measured at 130 such words — puts the budget floor strictly inside
//!   the run, so every boundary the forward search and the back-off can reach
//!   ties while split `0` is below the floor. Widening off the end there
//!   confirmed the whole run, emptied the holdback and anchored the watermark
//!   strictly PAST the run's instant: every word the newest hypothesis produced
//!   at that same instant beyond `common` — words nothing had confirmed — was
//!   then filtered out of both hypotheses on the next worded ingest and lost.
//!   `an_over_budget_tied_run_defers_rather_than_stranding_its_suffix` is the
//!   falsifier. Deferring costs nothing on the round itself, since
//!   `LocalAgreement::finalize` emits the same words either way, and TAIL growth
//!   relieves it.
//!
//!   The FORCED empty holdback — the arm that first repair deliberately left
//!   alone — strands the same way and needs the same guard (codex round 3 on
//!   PR #95, second finding). Where the budget floor itself reaches
//!   `common.len()` the split ran off the end unconditionally, deciding on
//!   `common` alone and never looking at what the newer hypothesis produced past
//!   it. What that costs is a RETRACTION rather than a deletion: the round's own
//!   `LocalAgreement::finalize` PUBLISHES the stranded word through
//!   `find_longest_different_suffix`, and only the NEXT hypothesis loses it, so
//!   `confirmed_words`' append-only guarantee is intact throughout and cannot
//!   see it. The forced advance is therefore taken exactly while it strands
//!   nothing, which is also the whole of what keeps its original case alive: a
//!   `common` with nothing beyond it has no anchor to wait for, and deferring
//!   there would wait forever.
//!   `a_forced_empty_holdback_defers_rather_than_retracting_its_suffix` is the
//!   falsifier, and it needs both of its `finalize` points — the retraction is
//!   only visible across two.
//!
//!   The first shape of this rule widened unconditionally and left the empty
//!   holdback anchored at `end`, which put the ORIGINAL duplicate-confirmation
//!   defect back on the DEFAULT driver path: two ingests of one hypothesis whose
//!   agreed prefix ends in a zero-duration tied run confirmed that whole run,
//!   emptied the holdback, and re-confirmed the run on every later stride
//!   without bound (codex round 1 on PR #95;
//!   `a_trailing_tied_run_never_confirms_itself_twice_at_the_default_count`).
//!   The lesson is recorded with it: the patch had covered the ROUTE to that
//!   state (an over-budget word) and not the STATE, and the property test's own
//!   shape skipped the state, which read as coverage.
//!
//!   It is a claim about the LAST confirmed word, and that is as strong as it
//!   needs to be exactly while word starts inside one hypothesis are
//!   non-decreasing: `find_alignment` guarantees
//!   `w[i].end() <= w[i + 1].start() + 1e-4`, so an earlier confirmed word
//!   starts at or before the last one and so also before the watermark. It is
//!   NOT a claim about the whole list under a hypothesis whose starts run
//!   BACKWARDS — offered `[P@5.0, Q@1.0, R@2.0]` confirms `[P, Q]` at a 2.0 s
//!   watermark, and `P` then passes the offered filter while `Q` does not
//!   (measured). That input is unreachable now: `ingest` is `pub(crate)` and its
//!   only in-crate caller is [`LocalAgreementTranscriber::push_samples`], whose
//!   words come from `add_word_timestamps`. It is one more thing the seal below
//!   is holding up, and one more reason the grep gate exists.
//!
//!   **[BEHAVIOUR CHANGE]** the rule confirms one word EARLIER on a tied input
//!   the forward search clears, trading one round of revisability for a clip
//!   boundary that does not bisect a settled span — the same trade
//!   `budgeted_split` already makes — and holds one word LONGER where it has to
//!   back off instead. So [`LocalAgreement::agreement_count_needed`] is a target
//!   rather than an exact width in either direction: the budget can shorten the
//!   holdback and the back-off can lengthen it. **Trigger:** two adjacent agreed
//!   words with equal `start`. On words
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
//!   The split bounds the LENGTH trim only. `prefill_tokens` reduces a prefix a
//!   second way — it drops every id at or above the loaded vocabulary's
//!   `special_token_begin`, and a word carrying no tokens contributes nothing at
//!   all — and this engine does not model that (codex round 8, finding 1;
//!   round 12, finding 2). It does not have to:
//!   `segment::update_segments_with_word_timings` strips exactly those ids from
//!   every [`WordTiming`] this crate emits and emits no word at all for an
//!   all-special alignment entry, so the pipeline cannot produce such a word,
//!   and `LocalAgreement::new`/`LocalAgreement::ingest` are `pub(crate)`
//!   (see the seal), so no caller outside this crate can hand one in either.
//!   The residual the id filter used to carry is closed by the API surface
//!   rather than by a vocabulary threshold the hermetic engine cannot know.
//!
//!   **A widened-past word is CONFIRMED on the spot.** The argument is round 8's:
//!   a word the prefill cannot carry is neither corroborable nor revisable by a
//!   continuation decoded under
//!   `LocalAgreement::decoding_options_for_next`, being behind both the clip
//!   and the forced prefill, and the watermark passes it on the very next line,
//!   so no future result can ever be offered over its span. Holding it instead
//!   would be an indefinite wait ending in the deletion codex round 7 finding 2
//!   removed. A caller driving `LocalAgreement::ingest` with a result decoded
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
//!   advance can leave the holdback EMPTY** and the watermark anchored on the
//!   last confirmed word's own far edge instead of the first held one's start.
//!   A single word whose OWN tokens exceed the budget is now the only thing that
//!   can do that, and only on a round where nothing the newer hypothesis
//!   produced beyond `common` starts before the watermark it would set — Rule
//!   W's own widening backs off, and DEFERS both where the back-off has nowhere
//!   legal to land and where the forced advance would strand such a word, rather
//!   than emptying (see `split_at_a_strict_boundary`) — and the
//!   anchor is `empty_holdback_watermark`'s `end.max(start.next_up())`, so a
//!   zero-duration word there is still strictly behind the watermark. It has to:
//!   stopping while one word remained left a single word whose OWN tokens exceed
//!   the budget held anyway, and the cap silently did not cap (codex round 7,
//!   finding 2). What followed was data loss rather than a stall — the next
//!   hypothesis was decoded from a prefix `prefill_tokens` trims, came back with
//!   a word that is not the held one, disagreed, and
//!   `LocalAgreement::finalize`'s `holdback_superseded` path replaced the
//!   intact held word with that truncation. Confirming such a word is always
//!   possible and is no weaker a claim: `common` is the prefix two hypotheses
//!   agreed on, and a word outside the prefill budget is one no third hypothesis
//!   decoded from that prefill could revise — see the widened-past entry above
//!   for the qualifier that carries. It costs one thing, recorded as this
//!   module's residual 1: a genuinely NEW word beginning at the same instant a
//!   zero-duration confirmed word occupies is filtered out with the
//!   re-offer it cannot be told apart from.
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
//!   `LocalAgreement::finalize` instead emits the final hypothesis's own
//!   post-watermark words on that path (`holdback_superseded` is the flag), and
//!   on the DEFERRED one too (`split_deferred`), where the hypotheses agreed but
//!   the only split available was one Rule W refuses — nothing legal at or above
//!   the prefill budget floor, or a forced empty holdback that would strand the
//!   hypothesis's own suffix — so the holdback is an earlier agreement's and
//!   Swift's sum would drop everything between it and `common`'s end. It keeps Swift's shape everywhere else — including when
//!   the final hypothesis contributes nothing at or past the watermark, where
//!   nothing supersedes the holdback. How much of the holdback that path actually replaces is the
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
//!   `LocalAgreement::decoding_options_for_next` attaches is silently dropped
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
//! - **[API BREAK] the engine's mutating surface is sealed to this crate**, an
//!   unconditional and authorized break against `main` rather than a deviation
//!   from Swift — Swift has no library surface here at all. It removes one
//!   caller shape with no migration; the record, the reasoning and the design
//!   for the verified contract that could restore it are in the next section.
//!
//! # The engine's mutating surface is `pub(crate)`
//!
//! [`LocalAgreement`] is `pub` and fully READABLE —
//! [`LocalAgreement::confirmed_words_slice`],
//! [`LocalAgreement::last_agreed_words_slice`],
//! [`LocalAgreement::last_agreed_seconds`], [`LocalAgreement::results_slice`],
//! [`LocalAgreement::agreement_count_needed`] — and everything that MOVES its
//! state is crate-internal: its constructor, `ingest`, `finalize`,
//! `decoding_options_for_next`, and the count knob (rehomed to
//! [`LocalAgreementTranscriber::with_agreement_count_needed`], the only side
//! that can order the engine's calls correctly). `impl Default` is deliberately
//! ABSENT: a public trait impl is a public constructor.
//!
//! What that removes is one shape and one shape only: **bring your own
//! TRANSCRIPT.** A caller can still bring its own DECODER — the extension seam
//! is [`InferenceBackend`], which is public, unsealed, and has two impls in this
//! crate, and
//! [`WhisperKit::local_agreement_transcriber`](crate::audio::whisper::transcribe::WhisperKit::local_agreement_transcriber)
//! sits on `impl<B> WhisperKit<B>` with no bound at all, so a custom backend
//! inherits this entire stack, Rule W included. Only the caller that wanted to
//! hand `ingest` hypotheses this crate did not decode loses anything, and that
//! is the one shape which inherits NONE of the correctness above: the holdback
//! reproduction that makes an advance a re-agreement, the budget that keeps the
//! prefill whole, and Rule W's postcondition are all facts about hypotheses
//! [`LocalAgreementTranscriber`] produced. Removing it declines a promise that
//! was never true rather than withdrawing a working mode; the issue's own
//! impossibility argument is that no substitute oracle exists for it.
//!
//! **This is an AUTHORIZED, unconditional public API break** against `main`,
//! recorded here rather than left incidental. On `main` `LocalAgreement`
//! published `Default`, its constructor, count mutation, retargeting, `ingest`
//! and `finalize`; code outside this crate that fed stored, remote or
//! precomputed [`TranscriptionResult`]s into the engine no longer compiles, and
//! [`InferenceBackend`] is NOT an equivalent seam for it — that trait exposes
//! feature/encoder/decoder-step operations and cannot be handed an existing
//! transcript. **Migration: there is none today.** A caller that has a
//! transcript and wants confirmation over it has no supported path; the design
//! for one is recorded on <https://github.com/findit-studio/coremlit/issues/94>
//! and is a VERIFIED contract rather than a trusted one — the engine would check
//! that the returned result BEGINS with the holdback, which is decidable in one
//! pass, unlike the occurrence identity the impossibility result rules out. The
//! crate is unpublished, so no semver ceremony applies; the break is deliberate
//! and visible instead.
//!
//! `the_engine_exposes_no_public_mutator` in `tests/whisper/streaming.rs` is the
//! falsifier: it greps this file and reds if any of those names is re-published.
//!
//! # Residuals
//!
//! Stated with the sequence that reaches each, rather than claimed closed. Rule
//! W's postcondition IS total (see its entry above); the module claims no
//! totality beyond it.
//!
//! 1. **A new word at an empty holdback's own instant is DROPPED.** With nothing
//!    held back the watermark is the last confirmed word's `end`, raised to
//!    `start.next_up()` where that word has no duration — so a word the stream
//!    genuinely produces at that same instant fails the offered filter and never
//!    reaches a hypothesis. It cannot be helped: to a timestamp filter that word
//!    and a re-offer of the settled one are the same value, which is the issue's
//!    impossibility result, and the alternative is the unbounded
//!    re-confirmation #94 is about. A truncation is what the portable prefix
//!    property tolerates; a rewrite is not. Needs the prefill budget to empty the
//!    holdback — Rule W's own widening no longer does — over a ZERO-DURATION word,
//!    which takes a single word at the end of `common` whose own tokens exceed
//!    [`MAX_HOLDBACK_PREFILL_TOKENS`]. It does NOT additionally take a
//!    non-default [`LocalAgreement::agreement_count_needed`]: an earlier form of
//!    this entry listed one, and the default count reaches the same state
//!    whenever that one word is the last of the agreed prefix (measured).
//!    `add_word_timestamps` emitting a 112-token word is the whole of the gate —
//!    and only since the deferral: an AGGREGATE tied run over budget used to
//!    reach the same empty holdback with no oversized word anywhere, and what it
//!    cost there was the DELETION of the words at the run's own instant rather
//!    than this residual's single dropped word. That route now defers instead
//!    (codex round 3 on PR #95;
//!    `an_over_budget_tied_run_defers_rather_than_stranding_its_suffix`), which
//!    is what restores this sentence.
//!
//!    It is narrower again since round 3's second finding: the word this drops
//!    must be one no hypothesis had produced YET at the moment of the advance.
//!    A word already visible past `common` when the split runs is not dropped —
//!    the forced arm defers rather than stranding it
//!    (`a_forced_empty_holdback_defers_rather_than_retracting_its_suffix`). So
//!    what remains is exactly the word the NEXT decode invents at an instant
//!    already settled, which is the one case no watermark can tell from a
//!    re-offer.
//!    `a_zero_duration_word_at_an_empty_holdback_is_not_re_confirmed` drives it
//!    at count 1 — its dropped `" B"` arrives one hypothesis later, which is why
//!    it is still dropped — and `the_split_never_cuts_at_a_tied_start` sweeps
//!    both counts.
//! 2. **A repeat the engine's record cannot account for** is the stream's own,
//!    on the untied input — and on a TIED one Rule W deletes it instead. Both
//!    directions are pinned:
//!    `a_distinct_repetition_of_a_confirmed_word_survives_the_continuing_stream`
//!    and `rule_w_deletes_an_unaccounted_repeat_of_a_settled_word`.
//! 3. **Drift wider than the gap in front of the watermark.** A re-decode free
//!    to move every timestamp it emits can push a settled word past the
//!    watermark, where it reads as new speech rather than as a re-admission.
//!    Pre-existing on `main`; under
//!    `LocalAgreement::decoding_options_for_next` such a word is outside the
//!    clip window and behind the forced prefill, so the driver cannot reach it.
//! 4. **A crate-internal caller could still order the calls wrongly.** The seal
//!    is privacy plus a grep gate, not a type. There is one call site —
//!    [`LocalAgreementTranscriber::push_samples`], three lines in this file —
//!    and nothing stops a future in-crate caller from handing `ingest` a result
//!    it did not decode from `decoding_options_for_next`. Inverting the seam
//!    (an engine-side `step(|options| decode(options))`) would make the ordering
//!    unbreakable in-crate; flagged, not taken.
//! 5. **A VAD-dropped chunk at [`LocalAgreementTranscriber::finalize`].** A
//!    caller setting `chunking_strategy = Vad` on a stream longer than one
//!    window can lose a chunk covering the holdback's span; the final hypothesis
//!    then disagrees, the `holdback_superseded` path fires, and the record is
//!    replaced by words that never re-read it. Pre-existing on `main` and
//!    unchanged here — the coverage model this branch briefly carried did not
//!    close it either, because the nominal clip schedule is not the effective
//!    coverage. `task_facts().had_swallowed_error()` is a fact the PIPELINE
//!    recorded rather than caller testimony and could preserve the record;
//!    flagged, not taken.

use crate::audio::whisper::{
  backend::InferenceBackend,
  constants::{MAX_TOKEN_CONTEXT, SAMPLE_RATE},
  error::TranscribeError,
  options::DecodingOptions,
  result::{TranscriptionResult, WordTiming, merge_transcription_results_with_words},
  task_facts::{SpanKnowledge, TaskFactsAccumulator},
  text::{find_longest_common_prefix, find_longest_different_suffix},
  transcribe::WhisperKit,
};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// AgreementOutcome
// ---------------------------------------------------------------------

/// One `LocalAgreement::ingest` call's outcome — whether the new result
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
  /// The watermark is unchanged. Three routes reach it: there is no previous
  /// result to agree with yet (the first ingested result); the new hypothesis
  /// disagreed with the previous one, in which case the result was dropped
  /// rather than kept; or the two agreed and the round was DEFERRED because
  /// every split Rule W would accept is one the prefill budget refuses, or the
  /// one the budget forces would strand the hypothesis's own suffix (see
  /// `split_at_a_strict_boundary`), in which case the result was kept like any
  /// other agreeing one. The deferral is deliberately not its own variant: what
  /// this value reports is whether the watermark moved, and no in-crate caller
  /// branches on the reason.
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
/// `LocalAgreement::decoding_options_for_next` promises the next hypothesis
/// will be WRITTEN with — so a holdback that cannot survive this budget is one
/// the decoder is never given, and the words the trim erases would be neither
/// re-offered nor confirmed. `LocalAgreement::ingest` therefore holds back
/// only what fits, and
/// `budgeted_split` guarantees that for EVERY input rather than for every input
/// but one: where nothing can be held it holds nothing, and the advance confirms
/// the whole agreed prefix.
///
/// The trim is a LENGTH bound only. `prefill_tokens` also drops every id at or
/// above the loaded vocabulary's `special_token_begin`, and contributes nothing
/// at all for a word carrying no tokens — which `budgeted_split` does not model,
/// because it cannot arise: `add_word_timestamps` strips exactly those ids from
/// every [`WordTiming`] this crate emits and emits no word at all for an
/// all-special alignment entry (`segment::update_segments_with_word_timings`,
/// Swift `SegmentSeeker.swift:551-554`), and the engine's constructor and
/// `ingest` are `pub(crate)`, so no caller outside this crate can hand one in
/// (codex round 8, finding 1; codex round 12, finding 2).
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
  /// the latest one has since superseded. `Self::finalize` needs this and
  /// cannot recover it from the word lists alone; see the divergence recorded
  /// there and in this module's doc.
  ///
  /// Maintained ONLY on the worded path of `Self::ingest`, alongside
  /// [`Self::prev_words`]/[`Self::hypothesis_words`]/[`Self::last_agreed_words`]
  /// themselves: the [`AgreementOutcome::NoWordTimings`] early return leaves all
  /// four untouched, so this keeps describing the last hypothesis that actually
  /// had words to agree over — exactly the pair `finalize` reasons about.
  holdback_superseded: bool,
  /// Whether the most recent WORDED hypothesis AGREED with the previous one and
  /// the round still did not advance, because `split_at_a_strict_boundary` had
  /// no acceptable boundary to advance to: none legal at or above the prefill
  /// budget floor, or a forced empty holdback whose watermark would have
  /// stranded a word this hypothesis produced beyond `common` (#94, codex
  /// round 3 on PR #95, both findings). `Self::finalize` needs
  /// it for the same reason it needs [`Self::holdback_superseded`]: on such a
  /// round the holdback belongs to an EARLIER agreement while
  /// [`Self::hypothesis_words`] already re-covers that whole span and more, so
  /// Swift's `lastAgreedWords + differentSuffix` decomposition would drop
  /// everything between them.
  ///
  /// Maintained on exactly the same schedule as `Self::holdback_superseded` —
  /// only on the worded path, so the [`AgreementOutcome::NoWordTimings`] early
  /// return leaves it describing the last hypothesis that had words.
  split_deferred: bool,
  confirmed_words: Vec<WordTiming>,
  results: Vec<TranscriptionResult>,
  /// A sink for the reproducibility facts of EVERY ingested hypothesis —
  /// including the disagreeing ones dropped from [`Self::results`] but retained
  /// as [`Self::prev_result`] to CONTROL the next agreement comparison (codex
  /// round 8, F1). The same error-drop-sink pattern the VAD branch uses: a
  /// dropped hypothesis's unseeded draw (or callback truncation) still decided
  /// which words the surviving hypotheses agreed on, so it must reach
  /// `Self::finalize`'s reproducibility answer even though its segments never
  /// survive into the merge. Only the draw/early-stop/language facts are folded;
  /// the worker schedule and id span are stripped to `None` (see the strip in
  /// `Self::ingest`) — the finalized schedule is the adjudicated `None` and the
  /// finalized span is restored from the merged surviving result (round 10).
  ingested_facts: TaskFactsAccumulator,
}

impl LocalAgreement {
  /// A fresh engine: no prior result, a zero watermark, every collection
  /// empty, [`DEFAULT_AGREEMENT_COUNT_NEEDED`] words required to confirm
  /// (Swift's all-default locals, `TranscribeCLI.swift:346-353`).
  pub(crate) const fn new() -> Self {
    Self {
      agreement_count_needed: DEFAULT_AGREEMENT_COUNT_NEEDED,
      last_agreed_seconds: 0.0,
      prev_result: None,
      prev_words: Vec::new(),
      hypothesis_words: Vec::new(),
      last_agreed_words: Vec::new(),
      holdback_superseded: false,
      split_deferred: false,
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
  /// Builder form of `Self::set_agreement_count_needed`.
  #[must_use]
  #[inline(always)]
  pub(crate) const fn with_agreement_count_needed(mut self, agreement_count_needed: usize) -> Self {
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
  pub(crate) const fn set_agreement_count_needed(
    &mut self,
    agreement_count_needed: usize,
  ) -> &mut Self {
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
  /// Word timings settled so far: every agreement's leading remainder, ahead of
  /// that agreement's own [`Self::agreement_count_needed`]-word holdback.
  ///
  /// Append-only across the life of the engine: nothing here is ever rewritten,
  /// reordered, or taken back, which is why the trailing words of an agreement
  /// wait in [`Self::last_agreed_words_slice`] instead. RULE W (see
  /// `split_at_a_strict_boundary`) additionally guarantees, with no condition on
  /// that holdback, that this list's last word starts STRICTLY before
  /// [`Self::last_agreed_seconds`] — so nothing here can be re-offered to the
  /// agreement comparison and confirmed a second time.
  #[inline(always)]
  pub const fn confirmed_words_slice(&self) -> &[WordTiming] {
    self.confirmed_words.as_slice()
  }

  // -- results (Vec<TranscriptionResult>) ------------------------------------
  /// Every ingested result kept for the eventual `Self::finalize` merge
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
  pub(crate) fn decoding_options_for_next(&self, base: &DecodingOptions) -> DecodingOptions {
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
  /// `Self::decoding_options_for_next` attaches as
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
  /// It needs none. RULE W (see `split_at_a_strict_boundary`) refuses to put the
  /// watermark at a start already settled — so `confirmed_words.last().start() <
  /// last_agreed_seconds` STRICTLY after every advance, with no condition on the
  /// holdback, and no confirmed word can pass this filter. The re-admission the
  /// issue is about is unrepresentable rather than detected, which is why this
  /// is the Swift line and not a rule.
  ///
  /// What it deliberately leaves is the same short list the postcondition
  /// bounds, recorded in this module's doc and each with a named test: a repeat
  /// the engine's record cannot account for is read as the stream's own; a word
  /// the stream genuinely produces at an empty holdback's own instant is
  /// filtered out with the re-offer it cannot be told apart from; and a
  /// re-decode free to move every timestamp it emits can push a settled word
  /// past the watermark, where it reads as new speech.
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
  ///   `Self::new`), there is nothing to compare against: `result` is
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
  pub(crate) fn ingest(&mut self, result: TranscriptionResult) -> AgreementOutcome {
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
    let mut deferred = false;
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
        // RULE W'S DEFERRAL (#94, codex round 3 on PR #95, both findings).
        // `None` is "the two hypotheses agreed, but every advance available is
        // one Rule W refuses", and two states reach it. A tied run whose OWN
        // tokens exceed the budget puts the floor strictly inside the run, so
        // every boundary the forward search and the back-off can reach ties, and
        // split 0, the one boundary a tied run always leaves legal, is below the
        // floor. Or the budget FORCES the empty holdback and the watermark that
        // advance would anchor lies past a word this hypothesis produced beyond
        // `common` -- which is why `hypothesis_words` past `common` is passed
        // in: on `common` alone that arm cannot see what it would strand.
        //
        // Advancing in either state empties the holdback and anchors the
        // watermark strictly PAST the last confirmed instant. Every word the
        // newest hypothesis produced at that same instant beyond `common` is
        // then stranded: nothing confirmed it, the next worded ingest filters it
        // out of BOTH hypotheses at once, and `finalize` has nothing left to
        // recover it from -- after THIS round's `finalize` already published it
        // through `find_longest_different_suffix`. A deletion, and on the forced
        // arm a retraction, which is this module's non-preferred direction.
        //
        // So the round simply does not advance. It costs nothing that round --
        // `finalize` emits `confirmed ++ hypothesis_words` either way, see
        // `Self::finalize` -- and it is not the blocking policy this issue's
        // ledger refuted for deadlock: TAIL growth relieves it, the same relief
        // `split_at_a_strict_boundary`'s back-off already relies on, since one
        // word starting strictly later opens a boundary above the floor and one
        // word joining `common` gives an interior split something to hold.
        let boundary = split_at_a_strict_boundary(
          common,
          &self.hypothesis_words[common.len()..],
          requested,
          self.confirmed_words.last().map(WordTiming::start),
        );
        if let Some(split) = boundary {
          // `common` REPLACES the still-open record: it is the span two consecutive
          // hypotheses have just re-agreed over it, and `last_agreed_words` is the
          // one this hypothesis has superseded.
          self.confirmed_words.extend_from_slice(&common[..split]);
          self.last_agreed_words = common[split..].to_vec();
          // RULE W'S WATERMARK (#94). The watermark is the first held-back word's
          // start, which `split_at_a_strict_boundary` has already placed STRICTLY
          // past the last confirmed word's start.
          //
          // With NOTHING held back -- which takes a single word whose OWN
          // tokens exceed the prefill budget, the one thing that pushes the
          // budget floor to `common.len()`, on a round where nothing beyond
          // `common` would be stranded by the result; an over-budget tied RUN
          // and a live strand both defer instead (see
          // `split_at_a_strict_boundary`) -- there is no held word to measure
          // against, and `empty_holdback_watermark` is the answer. That is the
          // SAME function the deferral decision above consults, deliberately:
          // the guard and the value it guards may not drift apart.
          //
          // Monotone: every word of `common` starts at or past the old watermark,
          // `end >= start` for any word the pipeline emits, and `next_up` only
          // increases. A NaN start cannot break the postcondition either -- `max`
          // returns the non-NaN side, and `start >= watermark` is false for a NaN
          // start, so such a word is never offered back in the first place.
          //
          // `common` is non-empty here (its length is at least
          // `agreement_count_needed`, clamped to at least one), so the final
          // fallback is unreachable and only keeps this total.
          self.last_agreed_seconds = self.last_agreed_words.first().map_or_else(
            || {
              common
                .last()
                .map_or(self.last_agreed_seconds, empty_holdback_watermark)
            },
            WordTiming::start,
          );
          advanced = true;
        } else {
          // The hypotheses AGREED, so `result` is KEPT below (`:402`/`:408-410`,
          // no `skipAppend`) and the next round compares against it -- that is
          // what lets tail growth reach `common` and relieve this.
          deferred = true;
        }
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
    // Assigned on EVERY worded ingest for the same reason, so an advance or a
    // disagreement clears a deferral the round before it.
    self.split_deferred = deferred;

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
  /// **except when the holdback is not the final estimate of its own span** —
  /// the final hypothesis DISAGREED with it, or the final round was DEFERRED so
  /// the holdback belongs to an earlier agreement — where this port emits that
  /// hypothesis's own post-watermark words instead
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
  /// supply the exact count, round 12); see `Self::ingest`.
  pub(crate) fn finalize(mut self, options: &DecodingOptions) -> TranscriptionResult {
    if (self.holdback_superseded || self.split_deferred) && !self.hypothesis_words.is_empty() {
      // DIVERGENCE from `:418-419` — see this module's doc for the full
      // argument. Swift's `lastAgreedWords + differentSuffix(prevWords,
      // hypothesisWords)` is only a valid decomposition when the final round
      // ADVANCED, so that `last_agreed_words` is the holdback that round itself
      // produced. On the two paths here it is not.
      //
      // DISAGREED: `last_agreed_words` belongs to the
      // hypothesis this one just superseded, while `hypothesis_words` — filtered
      // to `start >= last_agreed_words[0].start()` — already re-covers that exact
      // span carrying the revision. Emitting both duplicates the span and strands
      // the superseded reading beside its own replacement; emitting only the
      // SUFFIX would instead drop the leading words both hypotheses produced,
      // which is the same defect's other face when the holdback is empty.
      //
      // DEFERRED (`split_deferred`, codex round 3 on PR #95): the two hypotheses
      // AGREED but no legal split existed at or above the prefill budget floor,
      // so `last_agreed_words` is an EARLIER agreement's holdback and Swift's sum
      // would drop everything between it and `common`'s end. `hypothesis_words`
      // is the latest full reading of the whole post-watermark span and subsumes
      // both. It is also byte-identical to what the widen-off-the-end fallback
      // this replaced produced on the same round: that fallback confirmed
      // `common`, left the holdback empty and finalized
      // `confirmed ++ common ++ hypothesis-beyond-common`, which is
      // `confirmed ++ hypothesis_words` written out. The whole divergence is in
      // what LATER ingests can still see, not in this round's transcript.
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
    // strip site in `Self::ingest`.
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

/// RULE W (#94, at its source): where an advance splits `common`, moved off any
/// boundary that would put the watermark AT a start already settled.
///
/// The watermark is the first held-back word's start, and it is also the CLIP
/// this engine hands its own next decoder. Cutting at a word whose start TIES
/// the last confirmed one puts that boundary INSIDE a span already settled: the
/// confirmed word then satisfies `LocalAgreement::watermark_filtered`'s own
/// `start >= watermark`, and the next hypothesis can re-offer it at the head of
/// its word list. That is the state every re-admission defence in this issue's
/// history was built to survive -- and the one that cannot be DECIDED from the
/// offered list, because a re-offered settled word and the stream's own second
/// occurrence of the same text are byte-identical there. Refuse to CREATE it.
///
/// A split at `at` is legal exactly when the word that would then END the
/// confirmed list starts STRICTLY before the word that would HEAD the holdback.
/// At `at == 0` nothing of `common` is confirmed, so the preceding word is the
/// engine's own last confirmed one -- the word the watermark would sit beside.
/// That arm is provably never the blocking one: the postcondition below gives
/// `confirmed.last().start() < last_agreed_seconds`, and every word of `common`
/// cleared `start >= last_agreed_seconds` to be offered at all, so
/// `confirmed.last().start() < common[0].start()` already. It is written as a
/// CHECK rather than assumed so this function is correct on its own terms rather
/// than on its caller's induction — but the proof is why it carries NO
/// falsifier: replacing `confirmed_last_start` with `None` reds nothing in this
/// crate, and no sequence through `LocalAgreement::ingest` can construct the
/// state it guards, because that state IS the postcondition's negation. It was
/// testable while the postcondition was conditional (through the empty-holdback
/// residual, which the `next_up` anchor has since closed) and its test went with
/// that state.
///
/// # Which legal boundary
///
/// `requested` (`common.len() - agreement_count_needed`) is where the split
/// WANTS to be, and `budgeted_split`'s floor is where the prefill budget lets it
/// be at the earliest. From `max(requested, floor)`:
///
/// 1. **Forward** to the first legal boundary. This confirms one word earlier
///    than Swift on a tied input -- the [BEHAVIOUR CHANGE] recorded in this
///    module's doc.
/// 2. **Backward**, if there is no legal boundary at or after that point, to the
///    LAST legal one still at or above the budget floor. Rule W may not empty
///    the holdback: widening past a run that reaches the END of `common` would
///    leave the watermark anchored at that run's own last word, which for a
///    zero-duration word is that word's own start -- the very state this rule
///    exists to refuse, re-entered from the other side (codex round 1 on PR #95;
///    `a_trailing_tied_run_never_confirms_itself_twice_at_the_default_count`,
///    where two ingests of ONE default-count hypothesis re-confirmed its whole
///    tied tail on every later stride, without bound).
///
///    Backing off never fails for want of a boundary while the budget floor is
///    `0`: `at == 0` is legal by the argument above, so a `common` that is one
///    tied run from end to end is held WHOLE rather than confirmed, and the next
///    stride -- which grows the hypothesis at its TAIL -- opens a boundary as
///    soon as one word starts strictly later. This is NOT the blocking policy
///    this issue's ledger refuted for deadlock: that one blocked on a predicate
///    over the HEAD of the offered list, which only an advance can move, whereas
///    this defers a split position that TAIL growth relieves, and it advances
///    the holdback rather than refusing the round.
/// 3. **`None` -- DEFER**, on either of two states. The first is where the
///    budget floor is `1` or more and no legal
///    boundary sits at or above it. A tied run whose OWN tokens exceed
///    `MAX_HOLDBACK_PREFILL_TOKENS` is the shape: the floor lands strictly
///    inside the run, every boundary the search and the back-off can reach ties,
///    and split `0` -- the boundary a tied run always leaves legal -- is below
///    the floor. Neither of the two positions that ARE available is acceptable.
///    Crossing the floor issues a prefill `prefill_tokens` truncates at the
///    head, and the head then exists in no hypothesis and in no confirmed list
///    (codex round 7, finding 2, exactly). Widening off the end -- what this
///    used to do -- confirms the whole run, empties the holdback and anchors the
///    watermark strictly PAST the run's instant, stranding every word the newest
///    hypothesis produced at that same instant beyond `common`: the next worded
///    ingest filters them from both hypotheses at once and `LocalAgreement::
///    finalize` cannot recover them (codex round 3 on PR #95).
///
///    So the round does not advance at all, and `LocalAgreement::ingest` records
///    it for `LocalAgreement::finalize`. It is the same policy as the back-off
///    above -- Rule W may not empty the holdback -- carried into the state where
///    the back-off has nowhere legal to land, and it is relieved the same way,
///    by TAIL growth: one word starting strictly later opens a boundary at or
///    above the floor, because a single word's tokens raise the floor by at most
///    that word's own cost. It costs nothing on the deferring round itself --
///    `finalize` emits `confirmed ++ hypothesis_words` either way (see
///    `LocalAgreement::finalize`) -- and the divergence is entirely in what
///    LATER ingests can still see.
///
///    The second state is arm 4's, refused: the budget FORCES the empty
///    holdback, and the watermark that advance would set --
///    `empty_holdback_watermark(common.last())` -- lies strictly past a word of
///    `beyond_common`. That word is stranded exactly as above, and what it costs
///    is one step worse: this round's own `finalize` PUBLISHES it, through
///    `find_longest_different_suffix`, so losing it on the next ingest RETRACTS
///    transcript rather than merely never emitting it. `confirmed_words`'
///    append-only guarantee never sees it -- the word was never confirmed. The
///    same relief applies: tail growth either moves the strand into `common`,
///    where an ordinary interior split can hold it, or opens a boundary above
///    the floor.
/// 4. **`Some(common.len())`** -- tested FIRST in the code, since its condition
///    is a property of the budget alone and neither search runs on it. Only
///    where the BUDGET FLOOR itself reaches `common.len()` is that length
///    returned and the holdback left empty. That takes a LAST word whose own
///    tokens exceed the budget, since nothing else runs `budgeted_split`'s loop
///    off the end, and it is the one state where the empty holdback is FORCED:
///    no split leaves a holdback the prefill could carry, so there is no anchor
///    to wait for and deferring would wait forever. Confirming is round 7
///    finding 2's own repair, and `LocalAgreement::ingest`'s watermark then
///    anchors strictly past the last confirmed start rather than at its `end`.
///    This arm is the WHOLE of what reaches the empty holdback.
///
///    "There is no anchor to wait for" is a claim about `common` alone, and it
///    is why this arm may not simply defer -- but the ADVANCE is not a property
///    of `common` alone, which is what `beyond_common` is here to supply. Where
///    a word past `common` would be stranded by this arm's own watermark, arm 3
///    takes the round instead; where nothing is, this arm advances exactly as
///    before. The guard is that narrow on purpose: widening it to "defer
///    whenever the budget forces the empty holdback" re-opens round 7 finding
///    2, which is the wait that never ends.
///
///    An EMPTY `common` reaches this arm too (`floor == 0 == common.len()`) and
///    advances: there is no word to anchor a watermark on, so there is nothing
///    to strand, and `LocalAgreement::ingest`'s own fallback leaves the
///    watermark where it was. Like the `at == 0` check above it carries NO
///    falsifier -- `LocalAgreement::ingest` only calls this with
///    `common.len() >= agreement_count_needed`, which is clamped to at least one
///    -- and, like it, it is written out so this function is total on its own
///    terms rather than on its caller's clamp. Flipping it to defer reds nothing
///    in this crate.
///
/// So `agreement_count_needed` is a target rather than an exact width in EITHER
/// direction: the budget can shorten the holdback (see `budgeted_split`) and
/// this can lengthen it.
///
/// # Postcondition (TOTAL)
///
/// After every advance, `confirmed_words.last().start() < last_agreed_seconds`
/// STRICTLY -- with no condition on the holdback. Where the split is interior
/// the boundary above is exactly that inequality; where it is `common.len()`
/// `LocalAgreement::ingest`'s `next_up` anchor is. So no confirmed word can pass
/// the offered filter, and the re-admission question is unrepresentable rather
/// than defended against.
///
/// It stays TOTAL under both `None` states rather than becoming conditional
/// again: a deferred round is not an advance, so it moves neither side of the
/// inequality and there is no state for the claim to be excluded from. `None`
/// REMOVES reachable advances rather than adding unguarded ones, and the second
/// state removes a strict subset of arm 4's, which the `next_up` anchor already
/// carried.
///
/// # A second postcondition
///
/// After every advance, every word of the driving hypothesis that the advance
/// did NOT confirm still satisfies `LocalAgreement::watermark_filtered`'s own
/// `start >= last_agreed_seconds`, so the round cannot put a word it left
/// unsettled out of the next round's reach. Where the split is interior this is
/// free -- the watermark is `common[split].start()` and word starts inside one
/// hypothesis are non-decreasing, so everything from `split` on is at or past
/// it. Where the split is `common.len()` the watermark is strictly PAST the last
/// confirmed start, so it is not free, and the `beyond_common` test in arm 4 is
/// what buys it. `the_split_never_cuts_at_a_tied_start` sweeps it beside the
/// first postcondition.
///
/// It is a claim about the LAST confirmed word, and that is as strong as it needs
/// to be exactly while word starts inside one hypothesis are non-decreasing --
/// see this module's doc for the `find_alignment` guarantee that supplies it and
/// for the backwards-starts input the `pub(crate)` seal excludes.
fn split_at_a_strict_boundary(
  common: &[WordTiming],
  beyond_common: &[WordTiming],
  requested: usize,
  confirmed_last_start: Option<f32>,
) -> Option<usize> {
  let strictly_after_the_preceding_start = |at: usize| {
    let preceding = if at == 0 {
      confirmed_last_start
    } else {
      Some(common[at - 1].start())
    };
    preceding.is_none_or(|preceding| preceding < common[at].start())
  };
  // `budgeted_split` with nothing requested IS the budget's own floor: the
  // earliest split whose holdback fits `MAX_HOLDBACK_PREFILL_TOKENS`.
  let floor = budgeted_split(common, 0);
  if floor == common.len() {
    // The budget FORCES the empty holdback: no split leaves a non-empty one the
    // prefill could carry, which takes a LAST word whose own tokens exceed
    // `MAX_HOLDBACK_PREFILL_TOKENS` (that is the only way the loop in
    // `budgeted_split` runs off the end). Confirming it is always available and
    // is round 7 finding 2's own repair; deferring where nothing would be
    // stranded would wait for an anchor that can never arrive.
    //
    // But the advance would anchor the watermark at
    // `empty_holdback_watermark(common.last())`, and every word the newer
    // hypothesis produced BEFORE that instant beyond `common` is stranded by it
    // (codex round 3 on PR #95, second finding): the next worded ingest filters
    // it out of both hypotheses at once, and it is a word THIS round's
    // `LocalAgreement::finalize` already published through
    // `find_longest_different_suffix`. So the forced advance is available
    // exactly while it strands nothing, and the strand is what
    // `beyond_common` is here to see -- without it this arm decides on `common`
    // alone and cannot know what lies past it.
    //
    // `start < watermark` is `watermark_filtered`'s own `start >= watermark`
    // negated, and totally so: every word of `beyond_common` reached this call
    // through that filter, so none of their starts is NaN.
    let strands_a_suffix = common.last().is_some_and(|last| {
      let watermark = empty_holdback_watermark(last);
      beyond_common.iter().any(|word| word.start() < watermark)
    });
    if strands_a_suffix {
      return None;
    }
    return Some(common.len());
  }
  let widened = requested.max(floor);
  (widened..common.len())
    .find(|&at| strictly_after_the_preceding_start(at))
    .or_else(|| {
      (floor..widened)
        .rev()
        .find(|&at| strictly_after_the_preceding_start(at))
    })
}

/// Where an advance splits `common` into the part that is CONFIRMED and the
/// part that is HELD BACK, given the requested split — moved later until every
/// word still held is one [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens)
/// carries into the initial prompt WHOLE.
///
/// The holdback is not merely "the last few agreed words": it is the text
/// `LocalAgreement::decoding_options_for_next` forces into the next
/// hypothesis, and
/// [`prefill_tokens`](crate::audio::whisper::decode::prefill_tokens) keeps only
/// the last [`MAX_HOLDBACK_PREFILL_TOKENS`] ids of it. A holdback the decoder
/// cannot be given whole is not a holdback at all — the words the trim erases
/// would be neither reproduced (the decoder never sees their tokens) nor
/// confirmed (an advance replaces the holdback with the new `common[split..]`),
/// so they would simply vanish from the transcript.
///
/// `prefill_tokens` reduces a prefix a SECOND way — it drops every id at or
/// above the loaded vocabulary's `special_token_begin` — and this does not model
/// that (codex round 8, finding 1). The premise that made it a residual is now
/// total rather than partial: `segment::update_segments_with_word_timings`
/// strips exactly those ids from every [`WordTiming`] this crate emits and emits
/// no word at all for an all-special alignment entry, so the pipeline cannot
/// produce such a word, and `LocalAgreement::new`/`LocalAgreement::ingest`
/// are `pub(crate)`, so no caller outside this crate can hand one in. See this
/// module's doc.
///
/// Widening the split instead takes that head OUT of the holdback and CONFIRMS
/// it. That is not a weaker claim than any other agreed word carries: `common`
/// is the prefix two consecutive hypotheses agreed on, which is the whole of
/// LocalAgreement-2's criterion, and `LocalAgreement::finalize` already
/// appends the entire holdback to
/// [`LocalAgreement::confirmed_words_slice`] unconditionally on its Swift-shaped
/// path. What the holdback buys on top of that is one more round in which a
/// third hypothesis could revise it — and a word the prefill cannot carry cannot
/// be revised by one *that was decoded from the prefill*, because whatever such
/// a hypothesis produces over that extent came from a DIFFERENT prefix and from
/// audio the clip excludes, and is therefore neither a corroboration of the held
/// word nor a revision of it. A caller driving `LocalAgreement::ingest` with a
/// result decoded some OTHER way is subject to neither reduction, so for it the
/// word is revisable after all and the confirmation lands beside the revision —
/// the same append-only cost `common[..split]` already carries on every path
/// (`an_overlapping_agreed_word_is_confirmed_on_the_mainline_path_too`).
///
/// Widening is the repair because the defect is the STATE, not any reading of
/// it. Leaving the unreproducible word IN the holdback is what round 7's
/// finding 2 recorded: the next unanchored hypothesis disagrees with it and
/// `LocalAgreement::finalize`'s `holdback_superseded` path deletes it.
///
/// The split runs all the way to `common.len()` when it has to, so the holdback
/// this leaves can be EMPTY. It has to (codex round 7, finding 2): stopping while
/// one word remained still held a single word whose OWN tokens exceed the budget,
/// and the cap silently did not cap. What followed was data
/// loss, not a stall: the next hypothesis came back with the truncated word
/// rather than the held one, disagreed, and
/// `LocalAgreement::finalize`'s `holdback_superseded` path replaced the intact
/// held word with that truncation. Made impossible here rather than refused
/// downstream, because a refusal on a public, infallible `ingest` has no path to
/// report on, and this needs none: taking the word out of the holdback is always
/// available and is exactly the argument above.
///
/// Where the holdback comes back empty, `LocalAgreement::ingest` anchors the
/// watermark on the last confirmed word's own far edge — `end`, raised to
/// `start.next_up()` where that word has no duration — rather than at the first
/// held word's start; see the anchor at its advance branch.
/// [`LocalAgreement::agreement_count_needed`] is then a maximum that reached
/// zero for that round, the same way it becomes a maximum for any holdback the
/// budget shortens.
///
/// Called with `requested == 0` this IS the budget's own FLOOR — the earliest
/// split whose holdback fits at all. `split_at_a_strict_boundary` needs that
/// value because its back-off moves the split EARLIER, and the floor is the one
/// line it may not cross: below it `prefill_tokens` silently truncates and the
/// erased words are neither re-offered nor confirmed, which is the whole of
/// round 7's finding 2. Where nothing legal sits at or above the floor, that
/// rule DEFERS the round rather than crossing the floor or widening off the end
/// — both of which delete a word — and the floor stays hard.
///
/// **Documented deviation**: with `agreement_count_needed` at its
/// [`DEFAULT_AGREEMENT_COUNT_NEEDED`] a two-word holdback of words
/// `add_word_timestamps` emits is nowhere near 112 tokens, and this is the
/// identity. What makes it bite is a HOLDBACK too expensive for the prefill, and
/// the count is only one of the two ways to get one: raise the count far enough,
/// or leave the count alone and have a SINGLE word at the end of `common` whose
/// own tokens exceed the budget — the split then runs to `common.len()` at the
/// default count too (measured). Either way the count becomes a maximum rather
/// than an exact width. (Rule W's back-off moves it the other way for any
/// caller — see `split_at_a_strict_boundary`.)
///
/// The count is NOT out of a public caller's reach, and an earlier form of this
/// note said it was: [`LocalAgreementTranscriber::with_agreement_count_needed`]
/// is `pub`, having been rehomed there when the engine's own knob was sealed. It
/// is `LocalAgreement::set_agreement_count_needed` that a caller outside this
/// crate cannot call, and the driver's builder does the same job.
fn budgeted_split(common: &[WordTiming], requested: usize) -> usize {
  let mut split = requested;
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

/// The watermark an advance that holds NOTHING back anchors at: the first
/// instant strictly past `last`'s start, where `last` is the word that advance
/// confirms last.
///
/// `end` is the answer whenever that word has any duration at all; where it does
/// NOT, `end == start` and the word would satisfy
/// `LocalAgreement::watermark_filtered`'s own `start >= watermark` against its
/// own confirmation, which is the whole of #94. `next_up` is the IMMEDIATE `f32`
/// successor, so nothing representable lies between the result and the start it
/// excludes: exactly one instant is refused rather than a span, which an
/// `end + epsilon` tolerance would not have managed.
///
/// It is a named function rather than an expression at its one obvious site
/// because it has TWO callers with opposite jobs and they may not disagree.
/// `LocalAgreement::ingest` SETS this watermark; `split_at_a_strict_boundary`
/// decides whether setting it would strand a word the newer hypothesis produced
/// beyond `common`, which is a question about this exact value. Computing it
/// twice would let the guard drift off the thing it guards, silently.
fn empty_holdback_watermark(last: &WordTiming) -> f32 {
  last.end().max(last.start().next_up())
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
/// `LocalAgreement::ingest` for the per-result confirmation logic this
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
  pub fn new(kit: &'ctx WhisperKit<B>, options: DecodingOptions) -> Self {
    Self {
      kit,
      options: options.with_word_timestamps(),
      agreement: LocalAgreement::new(),
      buffer: Vec::new(),
      transcribed_samples: 0,
    }
  }

  /// How many consecutive agreeing words this driver's engine needs before it
  /// confirms — [`LocalAgreement::agreement_count_needed`], set from the only
  /// side that can order the engine's calls correctly.
  ///
  /// Clamped up to at least `1`: zero would hold back no words at all on every
  /// advance, so no hypothesis would ever be given an anchor to re-decode from
  /// and LocalAgreement-2's second round of corroboration would be switched off
  /// wholesale. Swift hardcodes `2` and never exposes the knob
  /// (`TranscribeCLI.swift:349`).
  ///
  /// It is a TARGET rather than an exact width in either direction: raising it
  /// far enough makes [`MAX_HOLDBACK_PREFILL_TOKENS`] bite and the holdback
  /// comes back SHORTER (see `budgeted_split`), while a tied run at the end of
  /// an agreed prefix makes Rule W back its split off and the holdback comes
  /// back LONGER (see `split_at_a_strict_boundary`).
  #[must_use]
  #[inline(always)]
  pub fn with_agreement_count_needed(mut self, agreement_count_needed: usize) -> Self {
    self.agreement = self
      .agreement
      .with_agreement_count_needed(agreement_count_needed);
    self
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
  /// Delegates to `LocalAgreement::finalize`, passing this driver's own
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
  /// retargeted per `LocalAgreement::decoding_options_for_next`) and
  /// folds the result through `LocalAgreement::ingest`. Ports
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
