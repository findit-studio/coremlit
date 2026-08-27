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
//!   **Postcondition (TOTAL)** — after every advance, EVERY confirmed word
//!   starts strictly before `last_agreed_seconds`, with no condition on the
//!   holdback, so no confirmed word can satisfy the offered filter's own
//!   `start >= watermark` and none can head a hypothesis. The re-admission
//!   question is unrepresentable rather than defended against. Adjudicated:
//!   Swift shares the bug, and "confirmed once and stable" wins over parity
//!   here.
//!
//!   It is stated over the WHOLE list, and it used to be stated over its LAST
//!   word with "starts inside one hypothesis are non-decreasing" carrying the
//!   rest. That premise is false — see "Word starts run backwards" below — so
//!   the claim now rests on nothing outside this module: the split and the
//!   anchor both measure against `highest_start(confirmed_words)`, the
//!   high-water settled start, rather than against the last word.
//!
//!   Two things carry it, and they are separable. `split_at_a_strict_boundary`
//!   puts an INTERIOR split only on a boundary whose start is strictly past
//!   every settled start behind it — searching forward from the requested split
//!   first, then BACKING OFF when the forward search would run off the end of
//!   `common`, because widening past a tied run that reaches that end would
//!   empty the holdback and anchor the watermark on the run's own last word —
//!   and widening off the END only where the prefill budget floor sits at or
//!   above every legal boundary, which is where the back-off has nowhere legal
//!   to land. And where the holdback is empty, `LocalAgreement::ingest` anchors
//!   at `empty_holdback_anchor`: the last confirmed word's own far edge, raised
//!   to `past_the_settled_instant` where that word has no duration — since
//!   `end == start` for a zero-duration word.
//!
//!   `past_the_settled_instant` is a SAMPLE-domain step, and it used to be
//!   `f32::next_up`. The watermark is read in two coordinate systems:
//!   `watermark_filtered` compares it against word starts in SECONDS, and
//!   `LocalAgreement::decoding_options_for_next` hands the same value to
//!   `clip_timestamps`, where `chunker::prepare_seek_clips` rounds it to a
//!   SAMPLE. One ULP is a real step in the first and none at all in the second —
//!   `2.0f32.next_up()` and `2.0` both clip to sample `32000` — so the anchor
//!   moved the filter and left the CLIP where it was, and the next stride
//!   re-read the settled word's own audio (codex round 7 on PR #95, finding 2;
//!   `the_driver_does_not_re_read_the_settled_words_own_sample`). The anchor is
//!   now the first instant strictly past the settled one in BOTH, which is the
//!   sample after the one it clips to. What that widens: the filter refuses
//!   every start inside the settled word's own sample rather than only the
//!   settled instant. Every word `segment::update_segments_with_word_timings`
//!   emits is centisecond-rounded — 160 samples — so nothing the pipeline can
//!   produce falls in the widened gap.
//!
//!   The SPARING fold (`sparing_watermark`) is what keeps the cost to the
//!   impossibility rather than to the policy, and it now runs on BOTH arms: it
//!   lowers the anchor to the earliest start among the words this round did not
//!   confirm — the holdback and everything past `common` — so every such word
//!   strictly after the highest settled start stays offerable
//!   (`a_word_starting_strictly_later_lowers_the_watermark_instead_of_being_stranded`).
//!   On an interior split with non-decreasing starts it is the identity; it
//!   bites where the starts run backwards. What no watermark can spare is a word
//!   at or below the settled high-water start itself, and that is residual 1.
//!
//!   **Word starts run backwards, and the pipeline is where they come from.**
//!   `segment::update_segments_with_word_timings` prefers a SEGMENT's own start
//!   over a first word the DTW drifted more than half a second earlier
//!   (`SegmentSeeker.swift:635-640`) and clamps that word to
//!   `end - constrained_median`, which can land BEHIND the word in front of it —
//!   measured, from a strictly non-decreasing alignment, in
//!   `a_backward_start_from_the_segment_pipeline_does_not_strand_a_later_word`
//!   (`[0.50, 0.80, 0.99, 0.81]`). `find_alignment`'s
//!   `w[i].end() <= w[i + 1].start() + 1e-4` is a guarantee about its OWN
//!   output, and the post-processing that follows it is not bound by it. Two
//!   claims used to rest on the premise and neither does now: the postcondition
//!   above reads the high-water start, and the second postcondition's exception
//!   is stated at `<=` that start rather than at the tie.
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
//!   The claim runs over the whole confirmed list, so a hypothesis whose starts
//!   run backwards does not escape it. `[P@1.15, Q@0.95, R@1.10]` is where the
//!   difference shows: the adjacent-predecessor test this rule used to make
//!   passes at `R` (`0.95 < 1.10`), confirms `P` at `1.15` and sets a `1.10`
//!   watermark, and `P` then passes the offered filter against its own
//!   confirmation — #94 reached from the backwards side. Against the running
//!   maximum the same round backs off to a legal boundary instead
//!   (`a_backwards_start_two_words_back_still_cannot_be_re_admitted`). That
//!   exact shape is not one the segment pipeline emits — its clamp needs a word
//!   spanning more than half a second and every word after a clamped one starts
//!   at or after that word's own alignment end — so this is a STRENGTHENING that
//!   removes a premise rather than a repair for a demonstrated route, and its
//!   falsifier is hermetic because the input is.
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
//!   advance can leave the holdback EMPTY** and the watermark anchored past the
//!   last confirmed word's own start instead of at the first held one's. Two
//!   things reach it: a single word whose OWN tokens exceed the budget, which
//!   pushes the budget floor itself to `common.len()`; and a tied run whose own
//!   tokens exceed the budget, which puts that floor strictly inside the run so
//!   that no legal boundary is left at or above it (see
//!   `split_at_a_strict_boundary`). Rule W's own widening backs off rather than
//!   emptying wherever a legal boundary remains above the floor. The anchor is
//!   `empty_holdback_anchor`, never below `past_the_settled_instant`, so a
//!   zero-duration word there is still strictly behind the watermark in BOTH the
//!   seconds the filter reads and the samples the clip does; `sparing_watermark`
//!   then lowers it, so a word the hypothesis already produced past `common` is
//!   spared wherever any instant could spare it. It has to:
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
//!   post-watermark words on that path (`holdback_superseded` is the flag). It
//!   keeps Swift's shape everywhere else — including when
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
//! - **[API BREAK] `MAX_CONSECUTIVE_DEFERRALS` is gone**, a `pub const` this
//!   branch itself added at `4f2a3c9` and removed at the commit that removed the
//!   deferral. It named the bound on a wait that no longer exists, so nothing
//!   replaces it; the next section is why. It never reached `main`, so the break
//!   is against this branch's own intermediate surface rather than against a
//!   released one.
//!
//! # Why there is no deferral
//!
//! Between `6987bec` and `b3ec5c6` an agreeing round could decline to advance.
//! Where `split_at_a_strict_boundary` found no legal boundary at or above the
//! prefill budget floor — or where the floor forced the empty holdback and the
//! watermark that advance would set lay past a word the hypothesis had already
//! produced beyond `common` — the round DEFERRED, waiting for `common` to grow,
//! under two bounds (a repeating `DeferralSignature` and a
//! `MAX_CONSECUTIVE_DEFERRALS` count) that ended the wait. What it was protecting
//! is real: the empty holdback strands a word at the settled instant, and on the
//! forced arm this round's own `finalize` has already PUBLISHED that word, so
//! losing it RETRACTS transcript rather than merely never emitting it — the
//! direction `c6fc2e1` named as the non-preferred one.
//!
//! It was removed because it made that direction WORSE, measured rather than
//! argued. Three trees — the deferral, this fallback, and `main` — were driven
//! over the accumulated counterexample suite of this issue and over
//! `the_split_never_cuts_at_a_tied_start`'s 512 fixed-seed trials, reading the
//! published transcript after every round:
//!
//! | 512 fixed-seed trials, drawn at `4b259ef` | deferral | fallback |
//! |---|---|---|
//! | words ERASED from the published transcript | 26 | 10 |
//! | strands, all at the settled instant | 29 | 38 |
//! | rounds with no legal boundary that took the empty holdback | 43 | 141 |
//! | rounds with no legal boundary that WAITED instead | 113 | — |
//!
//! Those columns are of `the_split_never_cuts_at_a_tied_start`'s draw AS IT WAS
//! at `4b259ef`, and they are not re-derivable from the sweep's current numbers:
//! the backwards-start half added for codex round 7's finding 1 re-rolls every
//! later draw, and the deferral tree is gone, so the comparison cannot be
//! re-run. It is recorded with the commit it was measured on rather than
//! restated as if it still described the shipped shape.
//!
//! The deferral produced 2.6x the published retractions. What it BOUGHT over the
//! same suite was four words on the growing-tied-prefix row and one word each on
//! two others — every one of them at the SETTLED instant, which is the class
//! this module already accepts as residual 1 and already ships two
//! characterization tests for. Over 141 fallback rounds and 38 strands the
//! sweep's own oracle held every time: nothing unconfirmed fell below the
//! watermark except at or before the settled start.
//!
//! And the wait had a liveness hole its own count bound could not close. A split
//! at `0` is an ADVANCE that confirms nothing — legal whenever `common` opens on
//! a boundary and the floor is `0` — yet it reset `deferrals_since_advance`. An
//! `L, L, S` cycle (two long hypotheses, then a short one that agrees on two
//! words) therefore held the engine at ZERO words confirmed for 30 rounds,
//! `results` growing one per round, the published transcript oscillating between
//! 2 and 113 words, while this fallback confirms 113 words on round 2 and is
//! stable forever. The count bound was defeated by the very split it was meant
//! to backstop.
//!
//! What the fallback keeps from that work, and what it does not. The SPARING
//! FOLD (`4f2a3c9`, now `sparing_watermark` and shared with the interior arm) is
//! kept: it is separable from the deferral, it is what confines the loss to the
//! settled high-water start, and dropping it reds
//! `a_word_starting_strictly_later_lowers_the_watermark_instead_of_being_stranded`.
//! What is not kept is the wait, its two bounds, the deferred flag and
//! `finalize`'s clause for it. `jfk_simulated_stream_confirms_the_transcript` is
//! byte-identical across all three trees, so no measured stream in this repo
//! distinguishes them at all.
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
//! 1. **A word at or below the settled high-water start is DROPPED.** The
//!    watermark is strictly past every confirmed start (postcondition 1), so a
//!    word this round did not confirm that starts at or below the highest
//!    confirmed one fails the offered filter and can never reach a hypothesis
//!    again. It cannot be helped: to a timestamp filter that word and a re-offer
//!    of the settled one are the same value, which is the issue's impossibility
//!    result, and the alternative is the unbounded re-confirmation #94 is about.
//!    A truncation is what the portable prefix property tolerates; a rewrite is
//!    not.
//!
//!    TWO shapes reach it, and they are the two halves of the second
//!    postcondition's exception. AT the settled start is the TIE, which needs an
//!    empty holdback: the prefill budget must run the split off the end of
//!    `common` — Rule W's own widening backs off wherever a legal boundary
//!    remains above the floor — and two things do that, a single word at the end
//!    of `common` whose own tokens exceed [`MAX_HOLDBACK_PREFILL_TOKENS`], and a
//!    TIED RUN whose tokens exceed it in aggregate, which `add_word_timestamps`
//!    produces from an all-zero alignment matrix. It does NOT additionally take
//!    a non-default [`LocalAgreement::agreement_count_needed`]: an earlier form
//!    of this entry listed one, and the default count reaches the same state
//!    whenever that one word is the last of the agreed prefix (measured).
//!
//!    BELOW the settled start is a BACKWARDS start, and it needs no empty
//!    holdback at all: `segment::update_segments_with_word_timings` can put a
//!    later word behind an earlier one (see "Word starts run backwards" above),
//!    and an INTERIOR split that confirms past it strands it. This half was
//!    invisible while the postcondition assumed non-decreasing starts (codex
//!    round 7 on PR #95, finding 1); `the_split_never_cuts_at_a_tied_start`
//!    counts it as `backward_strands` and reached it 4 times in 512 trials.
//!
//!    It covers a word the hypothesis had ALREADY produced past `common` as well
//!    as one a later decode invents. Between `6987bec` and `b3ec5c6` the first
//!    kind was excluded — the round DEFERRED rather than stranding it — and
//!    removing the deferral restores it, on measured evidence that the deferral
//!    cost more published transcript than it saved (see "Why there is no
//!    deferral" above). What is NOT covered is any other instant: the sparing
//!    fold in `sparing_watermark` lowers the anchor to the earliest unconfirmed
//!    word it can, so a word starting strictly later stays offerable
//!    (`a_word_starting_strictly_later_lowers_the_watermark_instead_of_being_stranded`).
//!    `a_zero_duration_word_at_an_empty_holdback_is_not_re_confirmed` drives the
//!    invented-later kind at count 1 — its dropped `" B"` arrives one hypothesis
//!    later, which is why it is still dropped;
//!    `an_over_budget_tied_run_strands_its_suffix_at_the_settled_instant` and
//!    `a_forced_empty_holdback_retracts_its_suffix_at_the_settled_instant` drive
//!    the already-visible kind on each of the two shapes above;
//!    `a_backward_start_from_the_segment_pipeline_does_not_strand_a_later_word`
//!    is the pipeline-built witness for the backwards half, and shows the case
//!    the sparing fold SAVES; and `the_split_never_cuts_at_a_tied_start` sweeps
//!    both counts and COUNTS the strands (`tie_strands`, `backward_strands`) and
//!    the erasures, so neither half can be reported as unreachable.
//! 2. **A repeat the engine's record cannot account for** is the stream's own,
//!    on the untied input — and on a TIED one Rule W deletes it instead. Both
//!    directions are pinned:
//!    `a_distinct_repetition_of_a_confirmed_word_survives_the_continuing_stream`
//!    and `rule_w_deletes_an_unaccounted_repeat_of_a_settled_word`.
//! 3. **Drift wider than the gap in front of the watermark.** A re-decode free
//!    to move every timestamp it emits can push a settled word past the
//!    watermark, where it reads as new speech rather than as a re-admission.
//!    Pre-existing on `main`, and DRIVER-REACHABLE — an earlier form of this
//!    entry said it was not, on the ground that `decoding_options_for_next` puts
//!    such a word "outside the clip window and behind the forced prefill". The
//!    second half stands; the first is false in general (codex round 7 on PR
//!    #95, finding 2). The clip begins at `clip_seek_sample(watermark)`, so the
//!    audio it excludes is what lies strictly before that SAMPLE — and a settled
//!    word whose own end reaches the watermark, or which has no duration at all,
//!    keeps its audio inside the next clip. Where the settled word has real
//!    duration the exclusion is real, which is the case the earlier note
//!    generalized from.
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
//! 6. **A whole hypothesis RE-TIMED past the watermark is confirmed twice, and
//!    the shape cannot be built without also building a re-admission.** Offer a
//!    113-word tied run at 2 s twice, then the SAME 113 words at 3 s: the first
//!    pair advances and anchors just past 2 s, the second pair is entirely
//!    strictly past that anchor, so it agrees with itself and is confirmed as
//!    well — 226 confirmed words where the stream said 113 (measured; `main`
//!    confirms 222 on the same input, so this is not new here).
//!
//!    It is AMBIGUOUS BY CONSTRUCTION, and the two readings give opposite
//!    verdicts. Read as RE-TIMESTAMPING, the second confirmation is a duplicate
//!    of the first and this is #94's own defect in another dress — the words are
//!    the same words, moved. Read by this module's documented contract, a word
//!    STRICTLY PAST the watermark is new speech (`watermark_filtered`, and
//!    residual 2's "a repeat the engine's record cannot account for is the
//!    stream's own"), so confirming it is exactly right and refusing it would be
//!    the re-admission defence this issue's ledger already refuted. There is no
//!    third reading available from the offered list: the two are byte-identical
//!    there, which is the issue's impossibility result reached from the drift
//!    side — residual 3's territory.
//!
//!    **It is DRIVER-REACHABLE, and an earlier form of this entry said it was
//!    not** (codex round 7 on PR #95, finding 2). The claim borrowed residual
//!    3's "outside the clip window", and that is exactly the shape it does not
//!    hold for: a 113-word tied run at 2 s is ZERO-DURATION, so the empty
//!    holdback's anchor is `past_the_settled_instant(2.0)` and the next clip
//!    begins one sample later — sample `32001` against the run's own `32000`.
//!    One sample is not the run's audio. No boundary derived from a
//!    zero-duration word's OWN timestamps can exclude the speech it came from,
//!    because the word claims no extent, so this is not something the anchor can
//!    close — and it was not closed by the `f32::next_up` anchor either, which
//!    did not move the clip at all. What the sample-domain anchor buys here is
//!    an HONEST boundary, not this residual.
//!
//!    **Unconstrained by any current assertion.** No test in this file, and no
//!    shape the 512-trial sweep draws, pins either verdict: the sweep's
//!    `retiming` half drifts a whole offering by 0.03 s per round, which is
//!    smaller than the gap in front of the watermark and so never jumps a
//!    settled run past it. Neither the deferral tree nor this one flags it. The
//!    deferral masked it on exactly this shape by declining the first advance,
//!    which is not a fix and did not generalize. Recorded rather than decided,
//!    because deciding it needs the identity oracle #94 proves does not exist.

use crate::audio::whisper::{
  audio::chunker,
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
  /// The watermark is unchanged. Two routes reach it: there is no previous
  /// result to agree with yet (the first ingested result), or the new hypothesis
  /// disagreed with the previous one, in which case the result was dropped
  /// rather than kept.
  ///
  /// Two hypotheses that AGREE always advance the watermark. A third route --
  /// an agreeing round that DEFERRED rather than advancing -- existed on this
  /// branch between `6987bec` and `b3ec5c6` and was removed on measured evidence
  /// (#94; see this module's doc, "Why there is no deferral"). What this value
  /// reports is what it always reported: whether the watermark moved.
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
  /// that holdback, that EVERY word here starts STRICTLY before
  /// [`Self::last_agreed_seconds`] — so nothing here can be re-offered to the
  /// agreement comparison and confirmed a second time. Over the whole list, not
  /// merely its last word: the last one need not be the latest, since
  /// `crate::audio::whisper::segment::update_segments_with_word_timings` emits
  /// backwards word starts.
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
  /// watermark at a start already settled — so EVERY confirmed word starts
  /// strictly before `last_agreed_seconds` after every advance, with no
  /// condition on the holdback, and no confirmed word can pass this filter. The
  /// re-admission the issue is about is unrepresentable rather than detected,
  /// which is why this is the Swift line and not a rule.
  ///
  /// The threshold it compares against is also the next stride's CLIP start
  /// (`Self::decoding_options_for_next`), and the two read it in different
  /// coordinates — seconds here, samples there. `past_the_settled_instant` is
  /// what keeps a step that moves this filter from being inert in the other one;
  /// what it costs this filter is that a word starting inside the settled word's
  /// own sample is refused, where the `f32::next_up` anchor refused only the
  /// settled instant. Pipeline word starts are centisecond-rounded, so nothing
  /// this crate emits falls in that gap (codex round 7 on PR #95, finding 2).
  ///
  /// What it deliberately leaves is the same short list the postcondition
  /// bounds, recorded in this module's doc and each with a named test: a repeat
  /// the engine's record cannot account for is read as the stream's own; a word
  /// the stream genuinely produces at or below the settled high-water start is
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
        let split =
          split_at_a_strict_boundary(common, requested, highest_start(&self.confirmed_words));
        // `common` REPLACES the still-open record: it is the span two consecutive
        // hypotheses have just re-agreed over it, and `last_agreed_words` is the
        // one this hypothesis has superseded.
        self.confirmed_words.extend_from_slice(&common[..split]);
        self.last_agreed_words = common[split..].to_vec();
        // RULE W'S WATERMARK (#94). Two steps, and BOTH postconditions are
        // properties of the second one.
        //
        // The ANCHOR is where the advance would like to put the clip boundary.
        // With a holdback it is the first held-back word's start, which
        // `split_at_a_strict_boundary` has already placed strictly past every
        // settled start. With NOTHING held back -- the fallback, reached where
        // neither the forward search nor the back-off found a legal boundary at
        // or above the prefill budget floor, and where the floor itself reaches
        // `common.len()` -- there is no held word to measure against and
        // `empty_holdback_anchor` supplies one: the last confirmed word's own
        // far edge, raised to the first instant past the settled start in the
        // SAMPLE domain the clip is read in.
        //
        // `sparing_watermark` then LOWERS that anchor to spare every word this
        // hypothesis produced that the advance did not settle -- the holdback
        // and everything past `common` alike. On a non-decreasing hypothesis
        // that changes nothing on the interior arm (every unconfirmed word
        // starts at or past the first held one) and is the fold the empty arm
        // has carried since `4f2a3c9`. It bites where the starts run BACKWARDS,
        // which `update_segments_with_word_timings` can produce (codex round 7
        // on PR #95, finding 1): there the adjacent boundary is not the latest
        // settled start, and an unconfirmed word can sit below the anchor.
        //
        // `settled_high` is read AFTER the append, so it covers what this very
        // round confirmed. Postcondition 1 follows: the anchor is strictly past
        // it and every folded candidate is filtered to be, so the minimum is --
        // and it is the HIGHEST confirmed start, not merely the last, so no
        // confirmed word can pass the offered filter. A NaN start cannot break
        // it either: `highest_start` skips NaN, `f32::max`/`f32::min` return the
        // non-NaN side, and `start >= watermark` is false for a NaN start, so
        // such a word is never offered back in the first place.
        //
        // `common` is non-empty here (its length is at least
        // `agreement_count_needed`, clamped to at least one), so the final
        // fallback is unreachable and only keeps this total.
        let settled_high = highest_start(&self.confirmed_words);
        let anchor = self.last_agreed_words.first().map_or_else(
          || {
            common.last().map_or(self.last_agreed_seconds, |last| {
              empty_holdback_anchor(last, settled_high.unwrap_or(f32::NEG_INFINITY))
            })
          },
          WordTiming::start,
        );
        self.last_agreed_seconds = sparing_watermark(
          anchor,
          settled_high.unwrap_or(f32::NEG_INFINITY),
          &self.hypothesis_words[split..],
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
  /// **except when the holdback is not the final estimate of its own span** —
  /// the final hypothesis DISAGREED with it — where this port emits that
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
  /// The exact word list `Self::finalize` publishes, taken out of the engine —
  /// `Self::confirmed_words_slice` with the round's own provisional tail folded
  /// on, under the two shapes described on `Self::finalize`.
  ///
  /// Split out of `Self::finalize` so the TRANSCRIPT can be observed with its
  /// timings intact (#94, codex round 5 on PR #95, closing note).
  /// `merge_transcription_results_with_words` keeps this list only as merged
  /// TEXT, and the merged segments carry the decoders' own words instead — so
  /// from `Self::finalize`'s return value alone there is no way to ask when a
  /// published word started, which is the question a retraction is answered by.
  /// `the_split_never_cuts_at_a_tied_start` reads this every round and asserts
  /// that a word which LEAVES the transcript is either still offerable or at the
  /// settled instant; the last two findings on this branch both hid in exactly
  /// that gap, where `confirmed_words`' append-only guarantee cannot see.
  fn take_finalized_words(&mut self) -> Vec<WordTiming> {
    if self.holdback_superseded && !self.hypothesis_words.is_empty() {
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
      // A DEFERRED round used to reach this branch too, between `6987bec` and
      // `b3ec5c6`, and no longer exists: an agreeing round always advances (#94;
      // see this module's doc, "Why there is no deferral"). It made no
      // difference to the round's own transcript -- a deferred round finalized
      // `confirmed ++ hypothesis_words` and the advance it refused finalizes
      // `confirmed ++ common ++ hypothesis-beyond-common`, which is the same
      // list -- so removing the clause moves no word out of any single round's
      // `finalize`.
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
    core::mem::take(&mut self.confirmed_words)
  }

  pub(crate) fn finalize(mut self, options: &DecodingOptions) -> TranscriptionResult {
    let words = self.take_finalized_words();
    let mut merged = merge_transcription_results_with_words(&self.results, &words, options);
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
/// The watermark is drawn from the first held-back word's start, and it is also
/// the CLIP this engine hands its own next decoder. Cutting at a word whose
/// start TIES a confirmed one puts that boundary INSIDE a span already settled:
/// the confirmed word then satisfies `LocalAgreement::watermark_filtered`'s own
/// `start >= watermark`, and the next hypothesis can re-offer it at the head of
/// its word list. That is the state every re-admission defence in this issue's
/// history was built to survive -- and the one that cannot be DECIDED from the
/// offered list, because a re-offered settled word and the stream's own second
/// occurrence of the same text are byte-identical there. Refuse to CREATE it.
///
/// A split at `at` is legal exactly when `common[at]` starts STRICTLY after
/// every start a split there would settle -- everything already confirmed, plus
/// `common[..at]`. That is a running MAXIMUM, not the adjacent predecessor:
/// word starts inside one hypothesis are not non-decreasing (see the
/// postconditions below), so the word immediately in front of the boundary is
/// not necessarily the latest one behind it.
///
/// At `at == 0` nothing of `common` is confirmed, so the maximum is the engine's
/// own `confirmed_start_high` -- the latest start the watermark would sit
/// beside. That arm is provably never the blocking one: the postcondition below
/// gives `high < last_agreed_seconds`, and every word of `common` cleared
/// `start >= last_agreed_seconds` to be offered at all, so
/// `high < common[0].start()` already. It is written as a CHECK rather than
/// assumed so this function is correct on its own terms rather than on its
/// caller's induction — but the proof is why it carries NO falsifier: replacing
/// `confirmed_start_high` with `None` reds nothing in this crate, and no
/// sequence through `LocalAgreement::ingest` can construct the state it guards,
/// because that state IS the postcondition's negation. It was testable while the
/// postcondition was conditional (through the empty-holdback residual, which the
/// anchor has since closed) and its test went with that state.
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
/// 3. **`common.len()` -- the FALLBACK**, where neither search found a legal
///    boundary at or above the budget floor. The floor is never crossed, because
///    below it `prefill_tokens` silently truncates the prefill and the erased
///    words are neither re-offered nor confirmed (codex round 7, finding 2), so
///    the only position left is off the end: `common` is confirmed WHOLE, the
///    holdback goes empty, and `LocalAgreement::ingest` draws the watermark from
///    `empty_holdback_anchor` rather than from a held word's start. Two shapes reach it, and neither is exotic. The
///    budget FLOOR itself can reach `common.len()`, which takes a LAST word
///    whose own tokens exceed `MAX_HOLDBACK_PREFILL_TOKENS` -- nothing else runs
///    `budgeted_split`'s loop off the end -- and there the empty holdback is
///    forced outright, no split leaving one the prefill could carry. Or the
///    floor lands strictly INSIDE a tied run whose own tokens exceed the budget,
///    where every boundary the forward search and the back-off can reach ties
///    and split `0` -- the boundary a tied run always leaves legal -- is below
///    the floor. `add_word_timestamps` produces the second shape from an
///    ALL-ZERO alignment matrix (113 ordinary one-token words at one instant,
///    measured at 130), so it is the reachable one.
///
///    **What this costs, and why it is not a DEFERRAL** (#94, measured on this
///    branch; see this module's doc, "Why there is no deferral"). The empty
///    holdback anchors strictly past the settled high-water start, so a word the
///    newest hypothesis produced beyond `common` AT that same start is stranded:
///    the next worded ingest filters it out of both hypotheses at once, and
///    `LocalAgreement::finalize` cannot reach it afterwards -- after THIS
///    round's `finalize` already published it through
///    `find_longest_different_suffix`. That is the TIE half of this module's
///    residual 1, and it is exactly that narrow: `sparing_watermark` lowers the
///    anchor to spare every word at any HIGHER instant
///    (`a_word_starting_strictly_later_lowers_the_watermark_instead_of_being_stranded`).
///
///    Between `6987bec` and `b3ec5c6` this arm did not advance at all: it
///    DEFERRED, waiting for `common` to grow, under two bounds that ended the
///    wait. Measured against exactly this fallback over the accumulated
///    counterexample suite and the 512-trial sweep, the deferral erased 26 words
///    from the published transcript where this erases 10, and its count bound
///    was defeated by `At(0)` -- an advance that confirms nothing yet resets the
///    counter -- so an `L, L, S` cycle held the engine at zero words confirmed
///    for 30 rounds while `results` grew one per round. What the wait bought
///    over the same suite was four words on one row and one word each on two
///    others, every one of them at the settled instant. The trade is recorded in
///    full in this module's doc; the fallback is what this returns.
///
/// So `agreement_count_needed` is a target rather than an exact width in EITHER
/// direction: the budget can shorten the holdback (see `budgeted_split`) and
/// this can lengthen it.
///
/// # Postcondition (TOTAL)
///
/// After every advance, EVERY confirmed word starts strictly before
/// `last_agreed_seconds` -- with no condition on the holdback. `sparing_watermark`
/// is what delivers it on both arms: it folds a minimum over candidates it has
/// already filtered to be strictly above `highest_start(confirmed_words)`,
/// starting from an anchor that is strictly above it too. Where the split is
/// interior the anchor is `common[split].start()` and the boundary rule above is
/// exactly that inequality; where the split is `common.len()` the anchor is
/// `empty_holdback_anchor`, which clears the settled high-water start by
/// construction. So no confirmed word can pass the offered filter, and the
/// re-admission question is unrepresentable rather than defended against.
///
/// It is stated over the whole list because the LAST confirmed word need not be
/// the LATEST: `segment::update_segments_with_word_timings` emits backwards word
/// starts (see this module's doc, "Word starts run backwards"). The claim
/// therefore rests on no premise about the pipeline at all.
///
/// The two arms above are the whole of the function, so there is no third state
/// to exclude the claim from: every round with `common.len() >=
/// agreement_count_needed` ADVANCES, and it advances to one of those two
/// positions. That is what removing the deferral restores (#94) -- while it
/// existed, totality had to be re-argued for a non-advancing state, and the
/// argument was that a deferred round moves neither side of the inequality.
///
/// # A second postcondition
///
/// After every advance, every word of the driving hypothesis that the advance
/// did NOT confirm still satisfies `LocalAgreement::watermark_filtered`'s own
/// `start >= last_agreed_seconds`, so the round cannot put a word it left
/// unsettled out of the next round's reach. `sparing_watermark` is again what
/// buys it, on both arms: the watermark is the LOWEST start among the words this
/// round did not confirm that any legal watermark could still spare.
///
/// **It has ONE exception, and the exception is the impossibility rather than a
/// policy** (#94). A stranded word starts AT OR BELOW the highest confirmed
/// start. There no value serves both claims -- the first postcondition demands a
/// watermark strictly past that start, and every such watermark filters the
/// strand -- so this is this module's residual 1.
///
/// Both halves of `<=` are reachable and they are reached differently. AT is the
/// TIE, and it takes an empty holdback: a zero-duration word the fallback arm
/// settles last, with something the same hypothesis produced beyond `common` at
/// that same instant. BELOW is a BACKWARDS start, and it takes no empty holdback
/// at all -- an INTERIOR split can confirm a word that starts later than one it
/// leaves unconfirmed, and then no watermark can clear the first while sparing
/// the second. The exception was written at `==` and gated on the arm until
/// codex round 7 on PR #95, finding 1; the gate was never what bounded this.
///
/// The exception's GATE moved twice. While the deferral existed it read "this
/// round escaped a repeating wait"; removing the deferral made it "this round's
/// split ran off the END of `common`" (measured over the 512-trial sweep at
/// `4b259ef`: 38 strands where the deferral had 29, and 10 words erased from the
/// published transcript where the deferral erased 26 -- the direction that
/// decided it, this module's doc, "Why there is no deferral"). The ARM gate is
/// now gone, because a backwards start strands from an interior split too.
/// `the_split_never_cuts_at_a_tied_start` sweeps both postconditions and both
/// halves of the exception, counts the empty-holdback rounds, the tie strands
/// and the backwards strands so none can pass by being unreachable -- and reads
/// the published TRANSCRIPT across every round besides, which is where a
/// retraction the confirmed list cannot see shows up.
fn split_at_a_strict_boundary(
  common: &[WordTiming],
  requested: usize,
  confirmed_start_high: Option<f32>,
) -> usize {
  // The HIGH-WATER settled start at each candidate boundary: everything already
  // confirmed, plus everything a split at `at` would confirm. Read as a running
  // maximum rather than as `common[at - 1].start()` because word starts inside
  // one hypothesis are NOT non-decreasing -- `update_segments_with_word_timings`
  // can pull a later segment's first word BACK past the word before it (codex
  // round 7 on PR #95, finding 1; `a_backward_start_from_the_segment_pipeline_
  // does_not_strand_a_later_word`), and against a backwards start the adjacent
  // predecessor is not the word the postcondition has to clear.
  //
  // `NEG_INFINITY` is the nothing-confirmed-yet base, where every boundary is
  // legal; `f32::max` returns the non-NaN side, so a NaN start neither poisons
  // the running maximum nor makes a boundary look legal (`high < start` is false
  // for a NaN `start`).
  let mut running = confirmed_start_high.unwrap_or(f32::NEG_INFINITY);
  let mut settled_before: Vec<f32> = Vec::with_capacity(common.len() + 1);
  settled_before.push(running);
  for word in common {
    running = running.max(word.start());
    settled_before.push(running);
  }
  let strictly_after_every_settled_start = |at: usize| settled_before[at] < common[at].start();
  // `budgeted_split` with nothing requested IS the budget's own floor: the
  // earliest split whose holdback fits `MAX_HOLDBACK_PREFILL_TOKENS`.
  let floor = budgeted_split(common, 0);
  let widened = requested.max(floor);
  (widened..common.len())
    .find(|&at| strictly_after_every_settled_start(at))
    .or_else(|| {
      (floor..widened)
        .rev()
        .find(|&at| strictly_after_every_settled_start(at))
    })
    .unwrap_or(common.len())
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
/// `past_the_settled_instant` where that word has no duration — rather than at
/// the first held word's start; see the anchor at its advance branch.
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
/// rule widens off the END rather than crossing the floor — the one thing it
/// never does, the floor being hard — and the empty holdback that leaves is its
/// arm 3.
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

/// The ANCHOR an advance that holds NOTHING back starts from: the last
/// confirmed word's own far edge, raised to the first instant past the settled
/// start where that word has no duration.
///
/// Strictly past `settled_high` is the hard part and the whole of #94: `end` is
/// the answer whenever `last` has any duration at all; where it does NOT,
/// `end == start` and the word would satisfy
/// `LocalAgreement::watermark_filtered`'s own `start >= watermark` against its
/// own confirmation. `past_the_settled_instant` is what supplies the rest — and
/// it is a SAMPLE-domain step rather than the `f32::next_up` this used to take,
/// because the same value is handed to `clip_timestamps` and one ULP of it
/// rounds to the settled word's own sample (see that function).
///
/// `settled_high` rather than `last.start()`: with backwards starts reachable
/// (see `split_at_a_strict_boundary`) the last word of `common` need not be the
/// LATEST one confirmed, and it is the latest that the first postcondition has
/// to clear. The two coincide on every non-decreasing input, which is why this
/// takes the value rather than re-deriving it from `last`.
///
/// This is only the anchor. `sparing_watermark` then lowers it to spare the
/// words the same hypothesis produced beyond `common`, and what neither can
/// spare is a word at or below `settled_high` — this module's residual 1, which
/// is the impossibility rather than a policy. `split_at_a_strict_boundary` is
/// what decides whether to advance into that at all.
///
/// Total, never NaN, and always strictly greater than `settled_high`:
/// `past_the_settled_instant` is, `f32::max` returns the non-NaN side so a NaN
/// `end` falls through to it rather than poisoning the result, and every start
/// that reached here did so through `LocalAgreement::watermark_filtered`, whose
/// `start >= watermark` is false for a NaN start.
fn empty_holdback_anchor(last: &WordTiming, settled_high: f32) -> f32 {
  last.end().max(past_the_settled_instant(settled_high))
}

/// The highest `start` in `words`, or `None` where `words` is empty — the
/// high-water settled start the two postconditions are stated against.
///
/// A MAXIMUM rather than `words.last()`: word starts inside one hypothesis are
/// not non-decreasing (see `split_at_a_strict_boundary`), so the last confirmed
/// word need not be the latest one, and it is the LATEST that a watermark has to
/// clear before `LocalAgreement::watermark_filtered` can be trusted to refuse
/// every confirmed word rather than only the final one.
///
/// `f32::max` returns the non-NaN side, so a NaN start is skipped rather than
/// absorbing the fold.
fn highest_start(words: &[WordTiming]) -> Option<f32> {
  words
    .iter()
    .map(WordTiming::start)
    .reduce(f32::max)
    .filter(|high| !high.is_nan())
}

/// The lowest instant strictly past `settled` in BOTH coordinate systems the
/// watermark is read in — seconds and SAMPLES.
///
/// The watermark has two consumers with two different granularities, and #94's
/// codex round 7 finding 2 is that a step which satisfies one can be inert in
/// the other. `LocalAgreement::watermark_filtered` compares it against word
/// starts in SECONDS, where `f32::next_up` — the immediate successor — is
/// enough to refuse exactly one instant.
/// `LocalAgreement::decoding_options_for_next` hands the SAME value to
/// [`DecodingOptions::clip_timestamps`](crate::audio::whisper::options::DecodingOptions::clip_timestamps_slice),
/// where `chunker::prepare_seek_clips` rounds it to a sample index — and one ULP
/// of a small `f32` is worth far less than half a sample, so `next_up` there
/// moves NOTHING. Measured: `2.0f32.next_up()` is `2.000000238418579`, and both
/// it and `2.0` round to sample `32000`. The "strictly past" guarantee was real
/// in float space and vacuous in sample space, so the next stride re-read the
/// settled word's own audio while the doc claimed it had been clipped away.
///
/// This closes that by asking `chunker::clip_seek_sample` — the rounding
/// `prepare_seek_clips` itself applies — where `settled` lands, and returning an
/// instant that lands strictly later. The first candidate is the exact time of
/// the NEXT sample, which sits half a sample above the round-half-away-from-zero
/// threshold; the loop then corrects for the quotient's own narrowing, which
/// costs a whole sample only once `settled` is past roughly 500 s and `f32`
/// seconds can no longer resolve one.
///
/// What it changes for the FILTER, stated rather than implied: the watermark
/// this anchors now refuses every word starting within the settled word's own
/// SAMPLE, where `next_up` refused only the settled instant itself. Every word
/// [`crate::audio::whisper::segment::update_segments_with_word_timings`] emits
/// is rounded to a centisecond (`rounded_to_places(_, 2)`), which is 160 samples
/// — so no word the pipeline can produce falls in the widened gap, and the
/// engine's residual 1 widens only for a caller synthesizing sub-centisecond
/// starts (`the_watermark_clears_the_settled_sample_not_just_the_settled_instant`).
///
/// Total and terminating. `next_up` is strictly increasing on the finite
/// floats, and `clip_seek_sample` is unbounded on them, so the loop exits; the
/// `is_finite` guard is what keeps a `settled` of `+inf` — which
/// `prepare_seek_clips` rejects outright and which `next_up` maps to itself —
/// from spinning. On that input this returns `+inf`, exactly as the `next_up`
/// anchor it replaces did.
fn past_the_settled_instant(settled: f32) -> f32 {
  let settled_sample = chunker::clip_seek_sample(settled);
  let mut boundary = (settled_sample as f32 + 1.0) / SAMPLE_RATE as f32;
  while boundary.is_finite()
    && (boundary <= settled || chunker::clip_seek_sample(boundary) <= settled_sample)
  {
    boundary = boundary.next_up();
  }
  boundary
}

/// The watermark an advance actually sets: `anchor`, lowered to spare every
/// UNCONFIRMED word of the driving hypothesis that any legal watermark could
/// spare.
///
/// `settled_high` is the highest start in the confirmed list AFTER this
/// round's append (`highest_start`), and it is the whole of the first
/// postcondition: the result must be strictly greater than it, or a confirmed
/// word passes `LocalAgreement::watermark_filtered`'s own `start >= watermark`
/// and can be re-admitted. `anchor` already is — see its two callers in
/// `LocalAgreement::ingest` — and every candidate this folds in is filtered to
/// be, so the minimum is too.
///
/// `unconfirmed` is `hypothesis_words[split..]`: the holdback plus everything
/// the same hypothesis produced past the agreed prefix. Every one of those words
/// is a word this round did NOT settle, so pushing the watermark past it would
/// filter it out of both hypotheses on the next worded ingest and leave
/// `LocalAgreement::finalize` unable to reach it — after this round's own
/// `finalize` has already published it. That is the second postcondition, and
/// the fold is what buys it on BOTH arms rather than only on the empty-holdback
/// one. On an interior split with non-decreasing starts it changes nothing: the
/// anchor is `common[split].start()` and every later word starts at or past it,
/// so the minimum is the anchor. It bites exactly where the starts run
/// BACKWARDS, which `update_segments_with_word_timings` can produce (codex
/// round 7 on PR #95, finding 1).
///
/// SKIPPING rather than abandoning on an unsparable word is deliberate and
/// predates this generalization: a word at or below `settled_high` cannot be
/// spared by ANY watermark the first postcondition permits, but the words
/// BEHIND it can, and an anchor that gave up on all of them because one was
/// unsparable stranded them as collateral (measured on the sweep: a strand at
/// 2.5 s lost to a tie at 2.0 s).
fn sparing_watermark(anchor: f32, settled_high: f32, unconfirmed: &[WordTiming]) -> f32 {
  unconfirmed
    .iter()
    .map(WordTiming::start)
    .filter(|start| *start > settled_high)
    .fold(anchor, f32::min)
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
