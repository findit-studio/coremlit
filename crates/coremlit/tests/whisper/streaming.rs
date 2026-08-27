//! Simulated-stream LocalAgreement-2 on jfk.wav / tiny (ports the
//! whisperkit-cli `transcribeStreamSimulated` loop, TranscribeCLI.swift:322-424).
//!
//! # Three portable properties, and two host-scoped measurements
//!
//! The PORTABLE properties are asserted on every host, because each compares
//! this run against ITSELF on the same machine — host fp16 drift moves both
//! sides of the comparison together, so none can red for hardware reasons:
//!
//! - **Truncation, never rewriting.** [`finalize`](
//!   coremlit::audio::whisper::stream::agreement::LocalAgreementTranscriber::finalize)'s
//!   confirmed text is a word-for-word prefix of the SAME kit's batch transcript
//!   of the whole clip. Confirming fewer words than the clip contains is
//!   LocalAgreement-2 working as designed — it holds back everything two
//!   consecutive hypotheses have not yet agreed on. Confirming a word the batch
//!   decode contradicts is not.
//! - **Monotone confirmation.** Across the pushes, `confirmed_words_slice()`
//!   only grows, `last_agreed_seconds()` never decreases, and no already-
//!   confirmed word is revised.
//! - **Honest outcome labels.** Every [`AgreementOutcome`] a push reported is
//!   the one an INDEPENDENT reconstruction of that push says it had to be. The
//!   reconstruction ([`Route`]) reads engine state — which hypotheses
//!   `results_slice()` retained, whether the retained one carried word timings,
//!   and the common-prefix length recomputed from the results themselves — and
//!   never the label under test. WHICH strides agree is host-scoped; that a
//!   label matches its own push's route is not.
//!
//! All three are non-vacuous by construction: the prefix check refuses a
//! confirmed text too short to have completed one agreement round (every string
//! starts with the empty string, so an unconstrained prefix check on an empty
//! confirmation asserts nothing), the monotonicity check refuses a run in which
//! nothing was ever confirmed, and the label check refuses a run in which no
//! stride ever progressed. The falsifiers for each are the hermetic tests at the
//! bottom of this file; they are NOT `#[ignore]`d, so the predicates are gated
//! on every host, model or no model.
//!
//! The MEASURED observations are TWO, both descriptions of one machine rather
//! than properties of the port, so both ride `tests/support/measured_band.rs`'s
//! three-way host gate — asserted on [`CHARACTERIZED_ON`], computed and PRINTED
//! everywhere else:
//!
//! - whether the confirmed stream ever reaches the clip's canonical phrase,
//!   [`CANONICAL_PHRASE`];
//! - the per-stride outcome sequence, [`RECORDED_OUTCOMES`] — the belt to the
//!   label check's braces. The label check proves each label matches its own
//!   push's route on every host; this pins WHICH routes this clip takes here,
//!   and so catches a stride whose behaviour changed while its label stayed
//!   self-consistent.
//!
//! # Why the phrase describes a host rather than the port
//!
//! An earlier revision of this file asserted the phrase unconditionally and
//! justified it by claiming the streaming decode "drifts with exactly the same
//! fp16 argmax flips" as `whisper_parity_jfk`. **That claim was false**, and PR
//! #89's CI runner disproved it: `jfk_tiny_golden`'s tokens are byte-identical
//! on the runner and on the development machine — the BATCH decode does not
//! diverge at all — while the confirmed stream on the runner stops at
//! `"and so my fellow americans"`.
//!
//! The two paths are not the same decode with different numbers. Every stride
//! after the first re-enters the decoder with `prefix_tokens` and
//! `clip_timestamps` that the batch path never sets
//! (`LocalAgreement::decoding_options_for_next`, crate-internal since the M1
//! seal — [`LocalAgreementTranscriber`](
//! coremlit::audio::whisper::stream::agreement::LocalAgreementTranscriber) is
//! what drives it),
//! and that changes which code runs, not just what it computes on. Once a
//! stride decodes `"And so my fellow Americans!"` — an exclamation mark the
//! batch decode never produces; `jfk_tiny_golden.json`'s own text has none —
//! the held-back words prefill the next window with a sentence-final `"!"`, and
//! `TimestampRulesFilter`'s timestamp-mass rule
//! (`src/audio/whisper/decode/filter/mod.rs:233`, `timestamp_mass_exceeds_text`
//! at `:306`) masks ALL text logits and forces a closing timestamp. The window
//! then re-emits only its own prefix, agrees with the previous hypothesis on
//! exactly `agreement_count_needed` words, confirms nothing, and re-establishes
//! the prefix that caused the stall. The `[stride ...]` lines this test prints
//! show it directly: a run of strides pinned at a 1.4 s watermark with three
//! confirmed words, then one stride that escapes. The `[outcome ...]` lines
//! beside them name the same thing in one word — the pinned strides that move
//! nothing report `stationary`.
//!
//! That is a BOOLEAN control-flow gate riding a sub-nat logit margin, not a
//! cascade of argmax flips, and it is sensitive to PLACEMENT on a single
//! machine. The diagnosis that established the mechanism reports the mass
//! rule's own margin at the 7 s buffer as `+0.230` (ANE, fires), `+0.280`
//! (`CpuAndGpu`, fires) and `-0.049` (`CpuOnly`, does NOT fire), with under one
//! nat of margin at the last stride that can still escape — inside this repo's
//! documented ~1.0 worst cross-placement logit delta (`parity_jfk.rs`, "Numeric
//! drift"). Those logit numbers are that diagnosis's, not this file's; what THIS
//! file measured is their consequence, and it agrees: on the host recorded in
//! [`CHARACTERIZED_ON`], `CpuOnly` escapes the stall a whole stride earlier than
//! either accelerated placement (see the table there). A gate that a placement
//! change moves on ONE machine is not a portable contract on every machine.
//!
//! # Not the golden's host
//!
//! This gate deliberately no longer borrows `jfk_tiny_golden.json`'s
//! `generationHost`. It owns no golden, and `common::golden_host_note` applies
//! GOLDEN semantics: a foreign host PANICS before any measurement. That is right
//! for a committed-oracle comparison, where the comparison is the whole test,
//! and wrong here — after the goldens were stamped with the CI runner's host it
//! hard-red this gate on every development machine without ever reaching a
//! CoreML number. The phrase's provenance is this machine's own measurement,
//! recorded in [`CHARACTERIZED_ON`].

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use coremlit::audio::whisper::{
  options::{DecodingOptions, Options},
  result::WordTiming,
  stream::agreement::{
    AgreementOutcome, DEFAULT_AGREEMENT_COUNT_NEEDED, LocalAgreement, STRIDE_SAMPLES,
  },
  text::{find_longest_common_prefix, normalized},
  transcribe::WhisperKit,
};

/// The clip's canonical phrase, as `whisperkit-cli` transcribes jfk.wav on
/// tiny. MEASURED, not portable: whether the confirmed stream reaches it is
/// decided by the timestamp-mass gate described in this module's docs, which a
/// same-host placement change already flips. Never weaken, widen or delete it —
/// it moves behind a host gate, it does not get easier.
const CANONICAL_PHRASE: &str = "ask not what your country can do for you";

/// The host class [`CANONICAL_PHRASE`] was measured on.
///
/// RECORDED, unlike the siglip bands — because unlike them, this one could be
/// measured rather than guessed. Sweeping the encoder and decoder together with
/// mel left on its shipping `CpuAndGpu`, every placement this machine can drive
/// confirms the WHOLE 22-word transcript, phrase included:
///
/// | encoder/decoder | phrase | confirmed | escapes the stall between strides |
/// | --- | --- | --- | --- |
/// | `CpuAndNeuralEngine` (shipping default) | present | 22/22 words | 9 -> 10 (watermark 1.44 s -> 8.68 s) |
/// | `CpuAndGpu` | present | 22/22 words | 9 -> 10 (watermark 1.44 s -> 8.68 s) |
/// | `CpuOnly` | present | 22/22 words | 8 -> 9 (watermark 1.44 s -> 6.92 s) |
///
/// So the phrase is not a knife-edge here: it survives a placement change, and
/// the only thing placement moves is which stride escapes the timestamp-mass
/// stall. On the CI runner it is not merely close — the stream stalls at
/// `"and so my fellow americans"` and never escapes at all. Arming it there
/// would assert something known to be false, which is exactly what the
/// [`common::BandVerdict::Foreign`] path exists to prevent.
///
/// To arm a different machine: run the command the band-gate banner prints, read
/// the `[band]` line, and point this at THAT host class. Never widen the phrase
/// to span both.
const CHARACTERIZED_ON: Option<common::CharacterizedHost> = Some(common::CharacterizedHost {
  os_build: "25F71",
  os_product_version: "26.5",
  chip: "Apple M1 Max",
  arch: "arm64",
});

/// The per-stride [`AgreementOutcome`] sequence the shipping JFK stream reports
/// on [`CHARACTERIZED_ON`].
///
/// MEASURED, exactly like [`CANONICAL_PHRASE`] and for the same reason: which
/// strides agree, and which of the agreeing ones move anything, is decided by
/// the host-scoped timestamp-mass gate this module's docs describe. It
/// therefore rides the same three-way band gate — asserted on the recorded host
/// class, computed and PRINTED everywhere else — and not a portable `assert!`.
///
/// # Why record a sequence at all, next to a portable relation
///
/// [`check_outcomes_match_independent_evidence`] is the portable braces and
/// this is the belt. The relation proves each label matches the route its push
/// took; this pins WHICH routes this clip takes on this machine. They fail on
/// different things: the relation catches a label that contradicts the engine
/// on any host, and this catches a stride whose behaviour changed while its
/// label stayed self-consistent — the stationary strides 7 and 9 becoming
/// agreeing-and-moving ones, say, or stride 1 finding word timings it did not
/// have.
///
/// The `[outcome ...]` trace this is compared against is also digested outside
/// the test, and the two must agree by construction:
///
/// ```text
/// cargo test -p coremlit --features whisper --test whisper_streaming -- --ignored --nocapture \
///   | grep -E '^\[outcome' | shasum -a 256
/// 9ce93ed6b510bba3c685061c03110cae2108cf0a6e813528fc47ee41c3e51911
/// ```
///
/// To re-record: run that command, read the `[outcome ...]` lines, and replace
/// this list — but only from a host that matches [`CHARACTERIZED_ON`]. A
/// sequence measured anywhere else must not be armed here, for the reason the
/// [`common::BandVerdict::Foreign`] path exists.
const RECORDED_OUTCOMES: &[AgreementOutcome] = &[
  AgreementOutcome::NoWordTimings,
  AgreementOutcome::AwaitingAgreement,
  AgreementOutcome::Progressed,
  AgreementOutcome::Progressed,
  AgreementOutcome::Progressed,
  AgreementOutcome::Progressed,
  AgreementOutcome::Stationary,
  AgreementOutcome::Progressed,
  AgreementOutcome::Stationary,
  AgreementOutcome::Progressed,
  AgreementOutcome::Progressed,
];

/// [`RECORDED_OUTCOMES`] as the `[outcome ...]` trace spells it, for a band
/// line and a failure message.
fn joined_labels(outcomes: &[AgreementOutcome]) -> String {
  outcomes
    .iter()
    .map(ToString::to_string)
    .collect::<Vec<_>>()
    .join(",")
}

/// The exact command that re-measures BOTH host-scoped measurements — the
/// phrase and [`RECORDED_OUTCOMES`] — on THIS machine, quoted into every
/// band-gate banner so a log names its own fix.
fn recharacterize_command() -> String {
  "cargo test -p coremlit --features whisper --test whisper_streaming -- --ignored --nocapture\n                \
   then read the printed `[band]` lines: if the phrase was present, set\n                \
   CHARACTERIZED_ON in crates/coremlit/tests/whisper/streaming.rs to the `this host`\n                \
   line above and replace RECORDED_OUTCOMES with the `[outcome ...]` labels the same\n                \
   run printed; if the phrase was ABSENT, leave BOTH alone — a host that does not\n                \
   produce the phrase must not arm either of them."
    .to_string()
}

// ---------------------------------------------------------------------
// The portable properties, as pure predicates
// ---------------------------------------------------------------------

/// One completed push's view of the confirmation state.
///
/// The TRANSCRIPT fields and the OUTCOME field are deliberately separate
/// concerns, and the test prints them on separate lines for the same reason: the
/// `[stride ...]` trace exists to prove that the watermark work changes no
/// transcript, and mixing the outcome label into it made an honest-signal change
/// disturb a guard that is not about signals. See
/// [`check_outcomes_match_independent_evidence`] for what the labels are
/// asserted against instead.
#[derive(Debug, Clone)]
struct Confirmation {
  /// 1-based stride index; also the buffered seconds, at a 1 s stride.
  stride: usize,
  /// [`LocalAgreement::last_agreed_seconds`] after this push.
  last_agreed_seconds: f32,
  /// [`LocalAgreement::confirmed_words_slice`], normalized — settled words.
  confirmed: Vec<String>,
  /// [`LocalAgreement::last_agreed_words_slice`], normalized — agreed but held
  /// back, still revisable by a later hypothesis.
  held_back: Vec<String>,
  /// Every [`AgreementOutcome`] this push reported, in order. A push runs one
  /// ingest per complete stride that newly accumulated, so at a 1 s stride and
  /// 1 s pushes this is one element — but it is a list because `push_samples`
  /// returns one.
  outcomes: Vec<AgreementOutcome>,
  /// What this push looked like from OUTSIDE [`Self::outcomes`] — the oracle's
  /// input. See [`OutcomeEvidence`].
  evidence: OutcomeEvidence,
}

impl Confirmation {
  fn observe(
    stride: usize,
    agreement: &LocalAgreement,
    outcomes: Vec<AgreementOutcome>,
    evidence: OutcomeEvidence,
  ) -> Self {
    Self {
      stride,
      last_agreed_seconds: agreement.last_agreed_seconds(),
      confirmed: normalized_words(agreement.confirmed_words_slice()),
      held_back: normalized_words(agreement.last_agreed_words_slice()),
      outcomes,
      evidence,
    }
  }

  /// The outcome labels this push reported, comma-joined — the `[outcome ...]`
  /// trace's payload.
  fn outcome_labels(&self) -> String {
    self
      .outcomes
      .iter()
      .map(ToString::to_string)
      .collect::<Vec<_>>()
      .join(",")
  }
}

/// Word timings as the normalized word strings the agreement compares.
fn normalized_words(words: &[WordTiming]) -> Vec<String> {
  words.iter().map(|word| normalized(word.word())).collect()
}

/// `words` filtered the way [`LocalAgreement::ingest`] filters BOTH sides of
/// its agreement comparison — `start >= watermark`. The engine's own
/// `watermark_filtered` is crate-internal; this is that filter rebuilt from the
/// public parts, so [`OutcomeEvidence::common_prefix`] compares the same two
/// lists the engine compared.
fn at_or_past(words: &[WordTiming], watermark: f32) -> Vec<WordTiming> {
  words
    .iter()
    .filter(|word| word.start() >= watermark)
    .cloned()
    .collect()
}

// ---------------------------------------------------------------------
// The agreement oracle: what a push DID, decided without reading what it SAID
// ---------------------------------------------------------------------

/// One push's route through [`LocalAgreement::ingest`], reconstructed from
/// engine state rather than from the [`AgreementOutcome`] under test.
///
/// # Why the label cannot be its own witness
///
/// The relation this replaced asked the outcome whether the round had agreed
/// (`outcome.agreed()`) and then checked progress against that answer. On a
/// round where nothing moved, THREE labels satisfy such a relation —
/// `stationary` through `is_progressed() == moved`, and `awaiting_agreement`
/// and `no_word_timings` through the not-agreed branch — so a regression that
/// swapped a stalled stride's `stationary` for `awaiting_agreement`, or the
/// reverse, passed. The transcript trace cannot catch it either: it is blind to
/// labels by construction (that is the point of the two-trace split above).
///
/// These four routes are genuinely different events, and each leaves a
/// signature outside the returned enum:
///
/// | route | previous hypothesis | word timings | result | oracle reads |
/// | --- | --- | --- | --- | --- |
/// | [`Self::Agreed`] | yes | yes | KEPT | `results_slice` grew, the kept result has words, and the recomputed common prefix reached `agreement_count_needed` |
/// | [`Self::FirstHypothesis`] | no | yes | KEPT | `results_slice` grew and the kept result has words, but no worded result preceded it |
/// | [`Self::Disagreed`] | yes | yes | DROPPED | `results_slice` did NOT grow |
/// | [`Self::NoWordTimings`] | — | no | KEPT | `results_slice` grew and the kept result carries no word timings |
///
/// Retention is the load-bearing one, and it is a channel the label cannot
/// reach: `ingest` drops a hypothesis on exactly one route, and
/// [`LocalAgreement::results_slice`] is the list [`LocalAgreement::finalize`]
/// merges. `common_prefix` is a SECOND, independent leg over the same question
/// — see [`OutcomeEvidence::common_prefix`] for the defect it catches that
/// retention alone does not.
///
/// No production accessor was added for any of this: `results_slice`,
/// `TranscriptionResult::segments_slice`/`all_words`,
/// `TranscriptionSegment::words_slice`, `last_agreed_seconds`,
/// `agreement_count_needed` and
/// [`find_longest_common_prefix`] were already public.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
  /// A previous hypothesis existed, this result carried word timings, their
  /// common prefix reached `agreement_count_needed`, and the result was KEPT.
  /// The only route that runs an advance, and so the only one that may move
  /// anything.
  Agreed,
  /// The first WORDED result — however many wordless ones preceded it. There is
  /// no previous hypothesis to compare against, so no agreement logic runs and
  /// the result is KEPT.
  ///
  /// It is the first WORDED one rather than the first one at all because
  /// `ingest`'s no-timings route returns BEFORE the assignment that installs a
  /// previous hypothesis, so a wordless result never becomes one. The shipping
  /// JFK run is exactly this shape: stride 1 is wordless, and stride 2 takes
  /// this route.
  FirstHypothesis,
  /// A previous hypothesis existed and the common prefix fell short, so the
  /// result was DROPPED. The one route on which `results_slice` does not grow.
  Disagreed,
  /// The result carried no word timings to agree over. Kept, and the engine
  /// returned before it could look at a previous hypothesis at all.
  NoWordTimings,
}

impl Route {
  /// The label a push on this route MUST report. `moved` is consulted only on
  /// [`Self::Agreed`], the one route that runs an advance.
  const fn expected_label(self, moved: bool) -> AgreementOutcome {
    match self {
      Self::Agreed if moved => AgreementOutcome::Progressed,
      Self::Agreed => AgreementOutcome::Stationary,
      Self::FirstHypothesis | Self::Disagreed => AgreementOutcome::AwaitingAgreement,
      Self::NoWordTimings => AgreementOutcome::NoWordTimings,
    }
  }

  /// Whether this route ran the advance — the only one that may move the
  /// watermark or the confirmed prefix.
  const fn advanced(self) -> bool {
    matches!(self, Self::Agreed)
  }

  /// The evidence that put a push on this route, for a failure message.
  const fn evidenced_by(self) -> &'static str {
    match self {
      Self::Agreed => {
        "the result was KEPT, it carried word timings, and a worded hypothesis \
         preceded it"
      }
      Self::FirstHypothesis => {
        "the result was KEPT and carried word timings, and no worded hypothesis \
         preceded it"
      }
      Self::Disagreed => "the result was DROPPED, which `ingest` does on no other route",
      Self::NoWordTimings => "the result was KEPT and carried no word timings",
    }
  }
}

/// What one push looked like from OUTSIDE the value under test.
///
/// [`check_outcomes_match_independent_evidence`] decides which [`Route`] a push
/// took from these three facts alone, works out the label that route demands,
/// and only then looks at what the push actually reported. Nothing here reads
/// that report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutcomeEvidence {
  /// [`LocalAgreement::results_slice`]'s length AFTER this push. It grows by
  /// one for a KEPT result and not at all for a dropped one, and `ingest` drops
  /// on exactly one route — a hypothesis whose common prefix with the previous
  /// one fell short. So this length alone separates [`Route::Disagreed`] from
  /// the three kept routes.
  kept_results: usize,
  /// Whether the result this push newly KEPT carried word timings, recomputed
  /// with `ingest`'s own gate: ANY segment with a non-empty `words_slice`.
  ///
  /// `None` when the push kept nothing. A dropped result is not in
  /// `results_slice` to look at — and it does not need to be, since it is
  /// necessarily worded: the no-timings route returns before any comparison
  /// that could drop one.
  kept_has_words: Option<bool>,
  /// The common-prefix length `ingest`'s agreement gate reads, recomputed here
  /// from the two RESULTS with [`find_longest_common_prefix`] over
  /// [`at_or_past`]-filtered words — the engine's own comparison, rebuilt from
  /// public parts.
  ///
  /// This is the second leg, and it catches a defect retention alone cannot: a
  /// broken agreement gate that KEEPS a hypothesis whose prefix fell short. The
  /// retention leg would call that round [`Route::Agreed`] and rubber-stamp
  /// whatever progress label it reported; this leg reds.
  ///
  /// `None` where the engine's previous hypothesis is out of the test's reach.
  /// Two ways in: no worded result has been ingested yet, or the previous
  /// ingested result was DROPPED and so never reached `results_slice`. Both are
  /// legitimate runs, so the coverage this leg achieves is REPORTED per stride
  /// in the `[evidence ...]` trace rather than floored — a run whose every
  /// agreeing round follows a dropped hypothesis would have no reachable
  /// predecessor anywhere, and refusing it would red a correct stream.
  common_prefix: Option<usize>,
}

impl OutcomeEvidence {
  /// A push that KEPT a result carrying word timings.
  const fn kept_worded(kept_results: usize, common_prefix: Option<usize>) -> Self {
    Self {
      kept_results,
      kept_has_words: Some(true),
      common_prefix,
    }
  }

  /// A push that KEPT a result with no word timings.
  const fn kept_wordless(kept_results: usize) -> Self {
    Self {
      kept_results,
      kept_has_words: Some(false),
      common_prefix: None,
    }
  }

  /// A push whose result was DROPPED — `kept_results` is therefore the length
  /// it already had.
  const fn dropped(kept_results: usize) -> Self {
    Self {
      kept_results,
      kept_has_words: None,
      common_prefix: None,
    }
  }

  /// The `[evidence ...]` trace's payload.
  fn trace(&self) -> String {
    format!(
      "kept {:>2}  words {}  common {}",
      self.kept_results,
      match self.kept_has_words {
        Some(true) => "yes    ",
        Some(false) => "none   ",
        None => "DROPPED",
      },
      match self.common_prefix {
        Some(length) => length.to_string(),
        None => "-".to_string(),
      },
    )
  }
}

/// PORTABLE: `confirmed` is a word-for-word prefix of `batch`, and long enough
/// to be evidence.
///
/// `min_words` is the run's own [`LocalAgreement::agreement_count_needed`]: one
/// completed agreement round confirms at least that many words into the
/// finalized text (a round whose common prefix is exactly that long holds all of
/// them back, and `finalize` folds the holdback in), so a shorter confirmation
/// means no two consecutive hypotheses ever agreed at all. Without this floor
/// the prefix check would pass vacuously on an empty confirmation, since every
/// string starts with the empty one — a green light that asserts nothing is
/// worse than the host-specific assertion it replaced.
///
/// # Errors
/// A confirmation shorter than `min_words`, or one that rewrites rather than
/// truncates, with the first diverging word named.
fn check_confirmed_is_prefix_of_batch(
  confirmed: &str,
  batch: &str,
  min_words: usize,
) -> Result<(), String> {
  let confirmed_words: Vec<&str> = confirmed.split_whitespace().collect();
  let batch_words: Vec<&str> = batch.split_whitespace().collect();

  if confirmed_words.len() < min_words {
    return Err(format!(
      "the stream confirmed {} word(s), fewer than the {min_words} that one completed \
       agreement round puts into the finalized text.\n  \
       confirmed : {confirmed:?}\n  \
       batch     : {batch:?}\n\
       LocalAgreement-2 is ALLOWED to stop early — that is what holding words back \
       means — but reaching here says no two consecutive hypotheses agreed on \
       {min_words} words anywhere in the clip, and the prefix check below would then \
       pass vacuously (every string starts with the empty string). This is a \
       confirmation stall, not a tolerance to relax.",
      confirmed_words.len()
    ));
  }

  if batch_words.starts_with(&confirmed_words) {
    return Ok(());
  }

  let mut report = format!(
    "the confirmed stream REWRITES the batch transcript rather than truncating it.\n  \
     confirmed : {} word(s) {confirmed:?}\n  \
     batch     : {} word(s) {batch:?}\n",
    confirmed_words.len(),
    batch_words.len(),
  );
  match confirmed_words
    .iter()
    .zip(&batch_words)
    .position(|(ours, theirs)| ours != theirs)
  {
    Some(index) => report.push_str(&format!(
      "  first divergence at word {index}: confirmed {:?} vs batch {:?}\n",
      confirmed_words[index], batch_words[index],
    )),
    // Every shared position matched, so the confirmation ran PAST the batch
    // transcript — the signature of a word confirmed twice. Two mechanisms
    // produce it, and both are guarded with hermetic falsifiers in
    // `stream/agreement/tests.rs`: `LocalAgreement::finalize` folding in BOTH
    // the held-back words and the last pair's differing suffix
    // (`holdback_superseded`), and a word confirmed twice across strides. The
    // second is closed at its source by RULE W (issue #94) — the advance never
    // puts the watermark at a word whose start ties the last confirmed one, so
    // `confirmed.last().start() < last_agreed_seconds` strictly and no
    // confirmed word can pass `watermark_filtered`'s `start >= watermark`
    // again. `the_split_never_cuts_at_a_tied_start` sweeps that postcondition.
    // The rule leaves ONE way back in, recorded there: with an EMPTY holdback
    // the watermark anchors at the last confirmed word's END, so a
    // zero-duration word there still ties it. Check that residual first, then
    // the decode.
    None => report.push_str(&format!(
      "  every shared position matched, but the confirmation is {} word(s) LONGER \
       than the batch transcript — i.e. it emitted words the batch decode never \
       produced. Suspect a word confirmed TWICE before suspecting the decode: \
       Rule W (issue #94) makes a re-admission unrepresentable while the \
       holdback is non-empty, but leaves one documented residual — an EMPTY \
       holdback anchors the watermark at the last confirmed word's END, so a \
       zero-duration word there ties it; see \
       `the_split_never_cuts_at_a_tied_start` in \
       `stream/agreement/tests.rs` — and `LocalAgreement::finalize` folds in \
       both the held-back words and the last pair's differing suffix behind the \
       `holdback_superseded` guard.\n",
      confirmed_words.len().saturating_sub(batch_words.len()),
    )),
  }
  report.push_str(
    "\nTruncating is fine and expected; contradicting a word already handed to the \
     caller is not.\nThis compares the run against ITSELF on this machine — same kit, \
     same clip, same compute\npath — so host fp16 drift moves both sides together and \
     cannot cause it. Do not add a\ntolerance: there is no number here to relax.",
  );
  Err(report)
}

/// PORTABLE: every reported [`AgreementOutcome`] is the one an INDEPENDENT
/// reconstruction of that push says it had to be.
///
/// The outcome sequence is real behaviour and must stay asserted, but the
/// LABELS are not portable — which strides agree, and which of the agreeing ones
/// move anything, is decided by the same host-scoped timestamp-mass gate this
/// file's docs describe. What IS portable is that each label matches the route
/// its push actually took, because that compares this run against itself.
///
/// The route comes from [`Route`], which reads engine state and never the
/// outcome under test: which hypotheses [`LocalAgreement::results_slice`]
/// retained, whether the retained one carried word timings, and the
/// common-prefix length recomputed from the results themselves. That
/// independence is the whole guard — see [`Route`] for the three-label
/// ambiguity a self-referential relation leaves open on a round that moved
/// nothing.
///
/// Progress within the agreeing route is still measured the same way `ingest`
/// measures it: [`LocalAgreement::last_agreed_seconds`] by BITS and the
/// confirmed prefix by length. Rounding the watermark to the two decimals the
/// trace prints would report a 0.02 s step as a stall, which is precisely the
/// reading the outcome exists to replace.
///
/// `agreement_count_needed` is the run's own
/// [`LocalAgreement::agreement_count_needed`], the threshold the recomputed
/// common prefix is read against.
///
/// # Errors
/// Evidence that cannot describe one ingest; a round the oracle says ran no
/// advance that moved something anyway; the first stride whose label is not the
/// one its route demands, naming both; or a run in which no stride ever
/// progressed, which would leave the agreeing route asserted only on its
/// stationary side.
fn check_outcomes_match_independent_evidence(
  steps: &[Confirmation],
  agreement_count_needed: usize,
) -> Result<(), String> {
  let mut progressed = 0usize;
  // `LocalAgreement`'s private `prev_result`, reconstructed as a boolean: the
  // no-timings route returns BEFORE the assignment that installs one, so a
  // wordless result never becomes a hypothesis however many of them arrive.
  // Both other kept routes and the dropped one do install theirs.
  let mut worded_before = false;
  for (index, after) in steps.iter().enumerate() {
    // The first push has no predecessor; the engine starts at watermark 0.0
    // with nothing confirmed and no results kept, which is what a synthetic
    // zeroth snapshot would say.
    let (was_watermark, was_confirmed, was_kept) =
      index
        .checked_sub(1)
        .map_or((0.0f32, 0usize, 0usize), |previous| {
          (
            steps[previous].last_agreed_seconds,
            steps[previous].confirmed.len(),
            steps[previous].evidence.kept_results,
          )
        });
    let moved = after.last_agreed_seconds.to_bits() != was_watermark.to_bits()
      || after.confirmed.len() != was_confirmed;
    // One push, one ingest at this stride and cadence — so the push's labels
    // describe exactly the transition measured above. A push that ran several
    // ingests would need per-ingest snapshots to say anything this sharp.
    let [outcome] = after.outcomes.as_slice() else {
      return Err(format!(
        "stride {} reported {} outcome(s), and this check reads the state ONCE \
         per push: {:?}.\nAt a 1 s stride and 1 s pushes exactly one ingest \
         runs per push. If the cadence changed, snapshot the engine per ingest \
         rather than weakening this.",
        after.stride,
        after.outcomes.len(),
        after.outcome_labels(),
      ));
    };

    // ── The evidence must describe ONE ingest, or it is not evidence for one.
    let evidence = &after.evidence;
    let kept = match evidence.kept_results.checked_sub(was_kept) {
      Some(0) => false,
      Some(1) => true,
      _ => {
        return Err(format!(
          "stride {}: the kept-result count went {was_kept} -> {}, which one \
           ingest cannot do — it keeps exactly one result or none, and the list \
           is append-only.\n\
           Either the cadence changed (see the one-outcome check above) or the \
           evidence was not read from the same engine the outcome came from.",
          after.stride, evidence.kept_results,
        ));
      }
    };
    if kept != evidence.kept_has_words.is_some() {
      return Err(format!(
        "stride {}: the kept-result count says the result was {}, and the \
         word-timing evidence says it was {}.\n\
         `kept_has_words` is read FROM the newly kept result, so it is present \
         exactly when one was kept. Fix the observation, not this check.",
        after.stride,
        if kept { "KEPT" } else { "DROPPED" },
        if evidence.kept_has_words.is_some() {
          "KEPT"
        } else {
          "DROPPED"
        },
      ));
    }

    // ── The route, from that evidence and nothing the push reported.
    let this_worded = evidence.kept_has_words != Some(false);
    let route = if !kept {
      if !worded_before {
        return Err(format!(
          "stride {}: the result was DROPPED, but no worded hypothesis had been \
           ingested yet.\n\
           `ingest` drops on ONE route — a common prefix too short — and that \
           comparison only runs against a previous hypothesis. With none there \
           is nothing to disagree with and the result is kept. This evidence \
           describes a state the engine cannot be in.",
          after.stride,
        ));
      }
      Route::Disagreed
    } else if evidence.kept_has_words == Some(false) {
      Route::NoWordTimings
    } else if worded_before {
      Route::Agreed
    } else {
      Route::FirstHypothesis
    };
    worded_before |= this_worded;

    // ── Second leg: the common prefix, where the previous hypothesis is still
    // reachable. It answers the SAME question as retention through a different
    // channel, so a disagreement between them is an engine defect either way.
    if let Some(common) = evidence.common_prefix {
      let agreed_by_prefix = common >= agreement_count_needed;
      if agreed_by_prefix != route.advanced() {
        return Err(format!(
          "stride {}: the two independent readings of whether this round AGREED \
           contradict each other.\n  \
           retention  : {} — so the round {}\n  \
           common prefix: {common} word(s) vs the {agreement_count_needed} this \
           run requires — so the round {}\n\
           `ingest` keeps a hypothesis exactly when that prefix reaches the \
           threshold, so these cannot differ. Suspect the agreement gate itself: \
           a gate that keeps a result whose prefix fell short reads as an \
           agreement to the retention leg and would rubber-stamp whatever \
           progress label followed.",
          after.stride,
          route.evidenced_by(),
          if route.advanced() {
            "agreed"
          } else {
            "did not agree"
          },
          if agreed_by_prefix {
            "agreed"
          } else {
            "did not agree"
          },
        ));
      }
    }

    // ── A round that ran no advance may not have moved anything, whatever it
    // reported. This is about the ENGINE, not about the label.
    if !route.advanced() && moved {
      return Err(format!(
        "stride {} did not agree — {} — so it ran no advance, yet the watermark \
         went {was_watermark} s -> {} s ({:#x} -> {:#x}) and the confirmed \
         prefix {was_confirmed} -> {} words.\n\
         Only the agreeing route touches either channel.\n\
         This compares the run against ITSELF on this machine, so host fp16 \
         drift moves both sides together and cannot cause it.",
        after.stride,
        route.evidenced_by(),
        after.last_agreed_seconds,
        was_watermark.to_bits(),
        after.last_agreed_seconds.to_bits(),
        after.confirmed.len(),
      ));
    }

    // ── The label, against the route's demand rather than against itself.
    let expected = route.expected_label(moved);
    if *outcome != expected {
      return Err(format!(
        "stride {} reported `{outcome}`, but the evidence says it had to report \
         `{expected}`.\n  \
         route     : {route:?} — {}\n  \
         watermark : {was_watermark} s -> {} s ({:#x} -> {:#x})\n  \
         confirmed : {was_confirmed} -> {} words\n  \
         kept      : {was_kept} -> {} result(s)\n  \
         common    : {}\n\
         The route is read from the engine's own state and NEVER from the \
         outcome, so this cannot be satisfied by relabelling: on a round that \
         moved nothing, `stationary`, `awaiting_agreement` and \
         `no_word_timings` are three different claims and exactly one of them \
         is true.\n\
         This compares the run against ITSELF on this machine, so host fp16 \
         drift moves both sides together and cannot cause it.",
        after.stride,
        route.evidenced_by(),
        after.last_agreed_seconds,
        was_watermark.to_bits(),
        after.last_agreed_seconds.to_bits(),
        after.confirmed.len(),
        evidence.kept_results,
        evidence.common_prefix.map_or_else(
          || "unreachable — the previous hypothesis was dropped, or none had \
              been ingested yet"
            .to_string(),
          |common| format!("{common} word(s) vs the {agreement_count_needed} this run requires"),
        ),
      ));
    }
    progressed += usize::from(outcome.is_progressed());
  }

  if progressed == 0 {
    return Err(format!(
      "no stride reported `progressed` across all {} stride(s), so the agreeing \
       route above was only ever read on its stationary side and asserts \
       nothing about the moving one.\n\
       A run that never progressed is a stall to diagnose, not evidence.",
      steps.len(),
    ));
  }
  Ok(())
}

/// PORTABLE: confirmation only ever moves forward across the pushes.
///
/// Checks three ways of going backwards — a receding
/// [`LocalAgreement::last_agreed_seconds`] watermark, a shrinking confirmed
/// list, and a rewritten confirmed word — and then refuses a run that confirmed
/// nothing at all, which would satisfy all three trivially.
///
/// # Errors
/// The first backwards step, naming both snapshots; or a run with no pushes or
/// no confirmation to be monotone about.
fn check_monotone_confirmation(steps: &[Confirmation]) -> Result<(), String> {
  let Some(last) = steps.last() else {
    return Err(
      "no pushes were observed, so monotonicity was asserted over an empty sequence \
       and could not have failed. The clip must produce at least one stride."
        .to_string(),
    );
  };

  for pair in steps.windows(2) {
    let (before, after) = (&pair[0], &pair[1]);
    if after.last_agreed_seconds < before.last_agreed_seconds {
      return Err(format!(
        "the agreement watermark RECEDED between stride {} and stride {}: {} s -> {} s.\n\
         Words before the watermark are settled and are never revisited, so moving it \
         back re-opens\ntext the caller was already told was final.",
        before.stride, after.stride, before.last_agreed_seconds, after.last_agreed_seconds,
      ));
    }
    if after.confirmed.len() < before.confirmed.len() {
      return Err(format!(
        "the confirmed word list SHRANK between stride {} and stride {}: {} -> {} words.\n  \
         stride {} : {:?}\n  stride {} : {:?}\n\
         Confirmed words are handed to the caller as final; retracting one is a defect, \
         not drift.",
        before.stride,
        after.stride,
        before.confirmed.len(),
        after.confirmed.len(),
        before.stride,
        before.confirmed,
        after.stride,
        after.confirmed,
      ));
    }
    if !after.confirmed.starts_with(&before.confirmed) {
      let index = before
        .confirmed
        .iter()
        .zip(&after.confirmed)
        .position(|(was, now)| was != now)
        .unwrap_or(before.confirmed.len());
      return Err(format!(
        "a CONFIRMED word was rewritten between stride {} and stride {}, at index {index}: \
         {:?} -> {:?}.\n  stride {} : {:?}\n  stride {} : {:?}\n\
         Confirmation is append-only; a later hypothesis may extend it and may never \
         revise it.",
        before.stride,
        after.stride,
        before.confirmed.get(index),
        after.confirmed.get(index),
        before.stride,
        before.confirmed,
        after.stride,
        after.confirmed,
      ));
    }
  }

  if last.confirmed.is_empty() && last.held_back.is_empty() {
    return Err(format!(
      "the stream confirmed and held back NOTHING across all {} stride(s), so every \
       monotonicity check above passed on empty sequences and asserted nothing.\n\
       A run with no confirmation is a stall to diagnose, not a monotone run.",
      steps.len(),
    ));
  }
  Ok(())
}

// ---------------------------------------------------------------------
// The model-gated measurement
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_simulated_stream_confirms_the_transcript() {
  // Opened BEFORE any CoreML number, so the log leads with whether the phrase is
  // asserted here, on which host it was measured, and how to re-measure.
  let gate = common::BandGate::open(
    "whisper streaming confirmed phrase",
    CHARACTERIZED_ON,
    recharacterize_command(),
  );

  // `Options::new` takes both folders directly (two-arg constructor, not a
  // zero-arg `new()` plus `with_model_folder`/`with_tokenizer_folder`
  // builders) — same brief-vs-shipped-API fix as tests/pipeline.rs's
  // `tiny_options`/tests/parity_jfk.rs.
  let kit = WhisperKit::new(&Options::new(common::tiny_dir(), common::tokenizer_dir())).unwrap();
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));

  // The reference for the prefix property: the SAME kit decoding the SAME clip
  // in one batch, under the same word-timestamped options the streamer forces on
  // its own copy. Holding `word_timestamps` equal is what leaves the per-stride
  // `prefix_tokens`/`clip_timestamps` retargeting as the only difference between
  // the two — which is exactly the thing under test.
  let batch_options = DecodingOptions::new().with_word_timestamps();
  let batch = normalized(kit.transcribe(&audio, &batch_options).unwrap().text());
  println!("[batch] {batch:?}");

  let mut streamer = kit.local_agreement_transcriber(DecodingOptions::new());
  // 1 s pushes — 11 strides, each re-transcribing the grown prefix.
  let mut steps = Vec::new();
  // TWO TRACES, on separate lines and deliberately so. `[batch]`/`[stride
  // ...]`/`[confirmed]` carry the TRANSCRIPT and the confirmation state, and
  // their extract (`grep -E '^\[(batch|stride|confirmed)'`) is the digest this
  // watermark work is measured against: it must not move. `[outcome ...]`
  // carries the labels, which this work DOES move, and digests separately.
  // While the two shared one line, an honest-signal fix disturbed a guard that
  // is not about signals.
  // THE ORACLE'S ONE PIECE OF STATE. `LocalAgreement::ingest` compares the new
  // hypothesis against a PRIVATE `prev_result`, and this tracks the half of it
  // the test can still see, by the engine's own three rules: a worded result
  // becomes the previous hypothesis; a WORDLESS one does not (that route returns
  // before the assignment); and a DROPPED one does become it but never reaches
  // `results_slice`, so the test loses sight of it and says so with `None`
  // rather than guessing. See `OutcomeEvidence::common_prefix`.
  let mut previous_hypothesis: Option<Vec<WordTiming>> = None;
  let mut kept_before = 0usize;
  let mut watermark_before = 0.0f32;
  for (index, chunk) in audio.chunks(STRIDE_SAMPLES).enumerate() {
    let outcomes = streamer.push_samples(chunk).unwrap();
    let agreement = streamer.agreement();

    // ── The evidence, read from engine state and never from `outcomes`.
    let kept_results = agreement.results_slice().len();
    let newly_kept = (kept_results > kept_before).then(|| {
      agreement
        .results_slice()
        .last()
        .expect("a kept result is in the list")
    });
    // `ingest`'s own gate (`:371`), recomputed: ANY segment with a word timing.
    let kept_has_words = newly_kept.map(|result| {
      result
        .segments_slice()
        .iter()
        .any(|segment| !segment.words_slice().is_empty())
    });
    let kept_words = newly_kept
      .filter(|_| kept_has_words == Some(true))
      .map(coremlit::audio::whisper::result::TranscriptionResult::all_words);
    // The engine's own comparison, over the watermark that was current when
    // this ingest ran — the one this push started from, not the one it ended
    // with.
    let common_prefix =
      previous_hypothesis
        .as_ref()
        .zip(kept_words.as_ref())
        .map(|(previous, hypothesis)| {
          find_longest_common_prefix(
            &at_or_past(previous, watermark_before),
            &at_or_past(hypothesis, watermark_before),
          )
          .len()
        });
    let evidence = OutcomeEvidence {
      kept_results,
      kept_has_words,
      common_prefix,
    };

    let step = Confirmation::observe(index + 1, agreement, outcomes, evidence);
    println!(
      "[stride {:>2}] watermark {:>6.2} s  confirmed {:>2}  held {:>2}  {:?}",
      step.stride,
      step.last_agreed_seconds,
      step.confirmed.len(),
      step.held_back.len(),
      step.confirmed.join(" "),
    );
    println!("[outcome {:>2}] {}", step.stride, step.outcome_labels());
    // A THIRD trace line, on its own prefix so neither digest above moves: what
    // the oracle read, so a failure can be diagnosed from a log alone.
    println!("[evidence {:>2}] {}", step.stride, step.evidence.trace());

    // Maintain the tracker by the three rules above, then the two baselines.
    match kept_has_words {
      Some(false) => {}
      Some(true) => previous_hypothesis = kept_words,
      None => previous_hypothesis = None,
    }
    kept_before = kept_results;
    watermark_before = step.last_agreed_seconds;
    steps.push(step);
  }
  // Read before `finalize` consumes the driver: the floor the prefix check needs
  // is this run's own agreement width, not a literal.
  let agreement_count_needed = streamer.agreement().agreement_count_needed();
  let confirmed = normalized(streamer.finalize().text());
  println!("[confirmed] {confirmed:?}");

  // ── Portable, asserted on EVERY host, and asserted FIRST so a real
  // confirmation regression is never reported from behind a host-scoped gate.
  if let Err(why) = check_monotone_confirmation(&steps) {
    panic!("streaming confirmation was not monotone: {why}");
  }
  if let Err(why) = check_confirmed_is_prefix_of_batch(&confirmed, &batch, agreement_count_needed) {
    panic!("streaming confirmation is not a prefix of the batch transcript: {why}");
  }
  // The outcome labels, asserted against an INDEPENDENT reconstruction of what
  // each push did — the labels themselves are host-scoped, the relation between
  // a label and its push's route is not.
  if let Err(why) = check_outcomes_match_independent_evidence(&steps, agreement_count_needed) {
    panic!("a reported agreement outcome did not match what the engine did: {why}");
  }

  // ── Measured, asserted only on the host class that produced it.
  //
  // The recorded label sequence is the belt to the relation's braces: the
  // relation proves every label matches its own push's route on every host, and
  // this pins WHICH routes this clip takes on the host that was characterized.
  // On that host it catches the one thing a self-consistent relabelling could
  // still hide — a stationary stride becoming a moving one, or stride 1 finding
  // word timings it did not have.
  let observed: Vec<AgreementOutcome> = steps
    .iter()
    .flat_map(|step| step.outcomes.iter().copied())
    .collect();
  gate.check_holds(
    "per-stride agreement outcome sequence",
    observed == RECORDED_OUTCOMES,
    &format!(
      "the {} recorded label(s) {}",
      RECORDED_OUTCOMES.len(),
      joined_labels(RECORDED_OUTCOMES),
    ),
    &format!(
      "the per-stride outcome sequence is not the one recorded on this host \
       class.\n  recorded : {} label(s) {}\n  observed : {} label(s) {}\n{}\n\
       The portable relation above already passed, so every label still matches \
       its own push's route —\nwhat moved is WHICH route a stride takes. That is \
       either a real behaviour change to\nunderstand or a host drift to \
       re-record; it is not a sequence to widen.",
      RECORDED_OUTCOMES.len(),
      joined_labels(RECORDED_OUTCOMES),
      observed.len(),
      joined_labels(&observed),
      RECORDED_OUTCOMES
        .iter()
        .zip(&observed)
        .position(|(recorded, seen)| recorded != seen)
        .map_or_else(
          || "  the shared prefix matched, so the two differ only in LENGTH — \
              the clip produced a different number of strides"
            .to_string(),
          |index| format!(
            "  first divergence at stride {}: recorded `{}`, observed `{}`",
            index + 1,
            RECORDED_OUTCOMES[index],
            observed[index],
          ),
        ),
    ),
  );

  gate.check_holds(
    "confirmed stream reaches the clip's canonical phrase",
    confirmed.contains(CANONICAL_PHRASE),
    &format!("contains {CANONICAL_PHRASE:?}"),
    &format!(
      "the confirmed stream never reached the canonical phrase on the host that \
       characterized it.\n  confirmed : {confirmed:?}\n  batch     : {batch:?}\n\
       The portable properties above already passed, so the confirmation is still a \
       faithful\ntruncation — what changed is HOW FAR it got before the timestamp-mass \
       gate stalled it\n(see this file's module docs). Do NOT weaken, widen or delete \
       the phrase to make this\npass; re-measure and re-record CHARACTERIZED_ON, or \
       diagnose the stall."
    ),
  );
}

// ---------------------------------------------------------------------
// Falsifiers for the portable predicates — hermetic, NOT model-gated
// ---------------------------------------------------------------------

/// Snapshots from normalized word lists, for the hermetic cases below.
///
/// No outcome and no evidence: the monotonicity cases are about the state
/// sequence alone. [`step_reporting`] is the one the outcome cases use.
fn step(stride: usize, watermark: f32, confirmed: &[&str], held_back: &[&str]) -> Confirmation {
  Confirmation {
    stride,
    last_agreed_seconds: watermark,
    confirmed: confirmed.iter().map(|w| (*w).to_string()).collect(),
    held_back: held_back.iter().map(|w| (*w).to_string()).collect(),
    outcomes: Vec::new(),
    evidence: OutcomeEvidence::dropped(0),
  }
}

/// The same snapshot carrying the label this push reported AND the independent
/// evidence the oracle reads. The two are supplied separately on purpose: every
/// falsifier below moves exactly one of them and leaves the other alone, which
/// is what makes each case a real minimal pair.
fn step_reporting(
  stride: usize,
  outcome: AgreementOutcome,
  watermark: f32,
  confirmed: &[&str],
  evidence: OutcomeEvidence,
) -> Confirmation {
  Confirmation {
    outcomes: vec![outcome],
    evidence,
    ..step(stride, watermark, confirmed, &[])
  }
}

/// The JFK run's own shape, as the labels a correct engine reports for it AND
/// the evidence that same engine leaves behind: a wordless stride, the first
/// worded hypothesis, then rounds that move the watermark and one that does
/// not.
///
/// The evidence is the shipping run's, not an invention — stride 1 keeps a
/// wordless result, stride 2 keeps the first worded one (it cannot agree: the
/// no-timings route never installed a hypothesis for it to agree with), and
/// every stride after that keeps a worded result whose recomputed common prefix
/// clears the default width of 2.
fn honest_run() -> Vec<Confirmation> {
  vec![
    step_reporting(
      1,
      AgreementOutcome::NoWordTimings,
      0.0,
      &[],
      OutcomeEvidence::kept_wordless(1),
    ),
    step_reporting(
      2,
      AgreementOutcome::AwaitingAgreement,
      0.0,
      &[],
      OutcomeEvidence::kept_worded(2, None),
    ),
    step_reporting(
      3,
      AgreementOutcome::Progressed,
      1.36,
      &["and", "so", "my"],
      OutcomeEvidence::kept_worded(3, Some(5)),
    ),
    step_reporting(
      4,
      AgreementOutcome::Progressed,
      1.38,
      &["and", "so", "my"],
      OutcomeEvidence::kept_worded(4, Some(5)),
    ),
    step_reporting(
      5,
      AgreementOutcome::Stationary,
      1.38,
      &["and", "so", "my"],
      OutcomeEvidence::kept_worded(5, Some(2)),
    ),
  ]
}

/// The width [`honest_run`]'s evidence is read at. Bound to the shipping
/// [`DEFAULT_AGREEMENT_COUNT_NEEDED`] rather than written as `2`, so a change to
/// the default moves these cases with it instead of leaving them asserting
/// against a literal the engine no longer uses.
const HERMETIC_AGREEMENT_WIDTH: usize = DEFAULT_AGREEMENT_COUNT_NEEDED;

/// The honest shape passes, and it is the JFK stall's own: a `stationary` round
/// with the watermark bit-identical to the round before it.
#[test]
fn an_outcome_sequence_that_matches_the_state_is_accepted() {
  assert_eq!(
    check_outcomes_match_independent_evidence(&honest_run(), HERMETIC_AGREEMENT_WIDTH),
    Ok(())
  );
}

/// THE FALSIFIER the honest-signal fix exists for: a round that moved neither
/// SETTLED channel — not the watermark, not the confirmed prefix — and claimed
/// progress. This is what every stalled stride reported before the two agreeing
/// cases were told apart.
#[test]
fn a_progressed_label_on_a_stalled_stride_reds() {
  let mut steps = honest_run();
  steps[4].outcomes = vec![AgreementOutcome::Progressed];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a stalled stride claiming progress must red");
  assert!(why.contains("stride 5"), "{why}");
  assert!(why.contains("reported `progressed`"), "{why}");
  assert!(why.contains("had to report `stationary`"), "{why}");
  assert!(why.contains("1.38"), "{why}");
}

/// And the other direction: a round that DID move and claimed to be stationary.
/// A `split != 0` progress test produces exactly this on a zero-split round
/// whose timings drifted.
#[test]
fn a_stationary_label_on_a_moving_stride_reds() {
  let mut steps = honest_run();
  steps[3].outcomes = vec![AgreementOutcome::Stationary];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a moving stride claiming to be stationary must red");
  assert!(why.contains("stride 4"), "{why}");
  assert!(why.contains("reported `stationary`"), "{why}");
  assert!(why.contains("had to report `progressed`"), "{why}");
}

// ── The three swaps a self-referential relation could not see ────────────────
//
// All three land on rounds where NOTHING moved, which is exactly where the
// relation this replaced went blind: it asked the outcome whether the round had
// agreed and then checked progress against that answer, so `stationary`
// (`is_progressed() == moved`, false == false), `awaiting_agreement` (`!moved`)
// and `no_word_timings` (`!moved`) all satisfied it. The oracle reads the route
// from `results_slice` and the kept result's own timings instead, so exactly one
// of the three is true on any given round.

/// SWAP 1: a stationary stride relabelled `awaiting_agreement`. The stride kept
/// a worded result behind a worded predecessor, so it AGREED; a round that
/// agreed cannot report the label of one that did not.
#[test]
fn a_stationary_stride_relabelled_awaiting_agreement_reds() {
  let mut steps = honest_run();
  // Non-vacuous: the round this replaces is the stall, and nothing moved across
  // it — the exact shape the old relation accepted under either label.
  assert_eq!(
    steps[4].last_agreed_seconds.to_bits(),
    steps[3].last_agreed_seconds.to_bits(),
  );
  assert_eq!(steps[4].confirmed.len(), steps[3].confirmed.len());

  steps[4].outcomes = vec![AgreementOutcome::AwaitingAgreement];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("an agreeing stall relabelled `awaiting_agreement` must red");
  assert!(why.contains("stride 5"), "{why}");
  assert!(why.contains("reported `awaiting_agreement`"), "{why}");
  assert!(why.contains("had to report `stationary`"), "{why}");
  assert!(why.contains("the result was KEPT"), "{why}");
}

/// SWAP 2: the same stationary stride relabelled `no_word_timings`. Its kept
/// result carried word timings, so the no-timings route is not one it could
/// have taken.
#[test]
fn a_stationary_stride_relabelled_no_word_timings_reds() {
  let mut steps = honest_run();
  steps[4].outcomes = vec![AgreementOutcome::NoWordTimings];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("an agreeing stall relabelled `no_word_timings` must red");
  assert!(why.contains("stride 5"), "{why}");
  assert!(why.contains("reported `no_word_timings`"), "{why}");
  assert!(why.contains("had to report `stationary`"), "{why}");
  assert!(why.contains("carried word timings"), "{why}");
}

/// SWAP 3, the reverse: a genuinely non-agreeing round relabelled `stationary`.
/// Stride 2 is the first WORDED hypothesis — the no-timings stride before it
/// never installed one to agree with — so no agreement logic ran at all, and it
/// moved nothing, which is what let the old relation take `stationary` for it.
#[test]
fn a_non_agreeing_stride_relabelled_stationary_reds() {
  let mut steps = honest_run();
  assert_eq!(steps[1].last_agreed_seconds.to_bits(), 0.0f32.to_bits());
  assert!(steps[1].confirmed.is_empty());

  steps[1].outcomes = vec![AgreementOutcome::Stationary];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a first-hypothesis round relabelled `stationary` must red");
  assert!(why.contains("stride 2"), "{why}");
  assert!(why.contains("reported `stationary`"), "{why}");
  assert!(why.contains("had to report `awaiting_agreement`"), "{why}");
  assert!(why.contains("FirstHypothesis"), "{why}");

  // And the OTHER non-agreeing route reads the same way: a DROPPED result is a
  // disagreement whatever label it carries.
  let mut steps = honest_run();
  steps[2].evidence = OutcomeEvidence::dropped(2);
  steps[2].last_agreed_seconds = steps[1].last_agreed_seconds;
  steps[2].confirmed = Vec::new();
  steps[2].outcomes = vec![AgreementOutcome::Stationary];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a dropped round relabelled `stationary` must red");
  assert!(why.contains("stride 3"), "{why}");
  assert!(why.contains("had to report `awaiting_agreement`"), "{why}");
  assert!(why.contains("the result was DROPPED"), "{why}");
}

/// A wordless stride relabelled as one of the agreeing ones reds too — the
/// fourth corner of the same square.
#[test]
fn a_wordless_stride_relabelled_stationary_reds() {
  let mut steps = honest_run();
  steps[0].outcomes = vec![AgreementOutcome::Stationary];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a wordless stride relabelled `stationary` must red");
  assert!(why.contains("stride 1"), "{why}");
  assert!(why.contains("had to report `no_word_timings`"), "{why}");
  assert!(why.contains("carried no word timings"), "{why}");
}

/// A round that did not agree ran no advance, so it may not have moved either.
/// This is a statement about the ENGINE rather than about the label, so the
/// evidence is what moves here: the round is made a DISAGREEMENT while its
/// watermark still steps.
#[test]
fn a_non_agreeing_round_that_moved_reds() {
  let mut steps = honest_run();
  // Stride 4 keeps its 1.36 -> 1.38 step and its `awaiting_agreement` label,
  // but its result was dropped — so no advance ran and nothing may have moved.
  steps[3].evidence = OutcomeEvidence::dropped(3);
  steps[3].outcomes = vec![AgreementOutcome::AwaitingAgreement];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a non-agreeing stride that moved the watermark must red");
  assert!(why.contains("stride 4"), "{why}");
  assert!(why.contains("did not agree"), "{why}");
  assert!(why.contains("ran no advance"), "{why}");
}

/// The SECOND leg. A gate that KEEPS a hypothesis whose common prefix fell
/// short reads as an agreement to the retention leg, which would then
/// rubber-stamp whatever progress label followed. The recomputed prefix refuses
/// it.
#[test]
fn a_kept_round_whose_prefix_fell_short_reds() {
  let mut steps = honest_run();
  steps[4].evidence = OutcomeEvidence::kept_worded(5, Some(HERMETIC_AGREEMENT_WIDTH - 1));
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a kept round whose prefix fell short must red");
  assert!(why.contains("stride 5"), "{why}");
  assert!(why.contains("contradict each other"), "{why}");
  assert!(why.contains("common prefix"), "{why}");

  // Non-vacuous: the SAME step at the full width is accepted, so the case turns
  // on the prefix length and on nothing else.
  steps[4].evidence = OutcomeEvidence::kept_worded(5, Some(HERMETIC_AGREEMENT_WIDTH));
  assert_eq!(
    check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH),
    Ok(())
  );
}

/// Evidence that cannot have come from one ingest is refused rather than
/// interpreted. Both directions: a kept-count that jumped, and a kept-count
/// that disagrees with whether a result was there to look at.
#[test]
fn evidence_that_cannot_describe_one_ingest_reds() {
  let mut steps = honest_run();
  steps[2].evidence = OutcomeEvidence::kept_worded(4, Some(5));
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a kept-count that jumped by two must red");
  assert!(why.contains("stride 3"), "{why}");
  assert!(why.contains("one ingest cannot do"), "{why}");

  let mut steps = honest_run();
  steps[2].evidence = OutcomeEvidence {
    kept_results: 2,
    kept_has_words: Some(true),
    common_prefix: Some(5),
  };
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a dropped result that was somehow inspected must red");
  assert!(why.contains("stride 3"), "{why}");
  assert!(why.contains("DROPPED"), "{why}");
  assert!(why.contains("Fix the observation"), "{why}");
}

/// Nothing can be dropped before there is a hypothesis to disagree with, so
/// evidence claiming it describes a state the engine cannot reach.
#[test]
fn a_drop_before_the_first_hypothesis_reds() {
  let steps = [step_reporting(
    1,
    AgreementOutcome::AwaitingAgreement,
    0.0,
    &[],
    OutcomeEvidence::dropped(0),
  )];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a drop with no previous hypothesis must red");
  assert!(why.contains("stride 1"), "{why}");
  assert!(why.contains("cannot be in"), "{why}");
}

/// Rounding is what the bit comparison is there to refuse: 1.38 and a watermark
/// one ULP away print identically at two decimals, and a `stationary` claim over
/// that step is still false.
#[test]
fn a_sub_printed_precision_step_is_still_a_move() {
  let mut steps = honest_run();
  steps[4].last_agreed_seconds = f32::from_bits(steps[3].last_agreed_seconds.to_bits() + 1);
  assert_eq!(
    format!("{:.2}", steps[4].last_agreed_seconds),
    format!("{:.2}", steps[3].last_agreed_seconds),
    "non-vacuous: the two print the same at the trace's own precision",
  );
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a one-ULP step is a move, and calling it stationary must red");
  assert!(why.contains("stride 5"), "{why}");
}

/// A run in which nothing ever progressed reads the agreeing route on its
/// stationary side only, so the predicate must refuse it rather than report a
/// green.
#[test]
fn a_run_that_never_progresses_is_not_outcome_evidence() {
  let steps = [
    step_reporting(
      1,
      AgreementOutcome::AwaitingAgreement,
      0.0,
      &[],
      OutcomeEvidence::kept_worded(1, None),
    ),
    step_reporting(
      2,
      AgreementOutcome::Stationary,
      0.0,
      &[],
      OutcomeEvidence::kept_worded(2, Some(HERMETIC_AGREEMENT_WIDTH)),
    ),
  ];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a run with no progress at all must red");
  assert!(why.contains("no stride reported `progressed`"), "{why}");
}

/// The check reads the engine's state once per PUSH, so it is only sound while a
/// push runs exactly one ingest. A push that ran several must red rather than
/// silently compare one transition against several labels.
#[test]
fn a_push_with_more_than_one_ingest_reds_rather_than_guessing() {
  let mut steps = honest_run();
  steps[2].outcomes = vec![AgreementOutcome::Progressed, AgreementOutcome::Stationary];
  let why = check_outcomes_match_independent_evidence(&steps, HERMETIC_AGREEMENT_WIDTH)
    .expect_err("a multi-ingest push must red");
  assert!(why.contains("reported 2 outcome(s)"), "{why}");
  assert!(why.contains("progressed,stationary"), "{why}");
}

/// The RECORDED sequence is the belt, so it must be able to red: a stride whose
/// route changed while its label stayed self-consistent passes the portable
/// relation and is caught only here.
#[test]
fn the_recorded_outcome_sequence_notices_a_changed_route() {
  // The shipping shape, and a copy in which stride 7's stall became a moving
  // round. Both are internally honest — the relation would pass either.
  let mut moved_stride_7 = RECORDED_OUTCOMES.to_vec();
  assert_eq!(moved_stride_7[6], AgreementOutcome::Stationary);
  moved_stride_7[6] = AgreementOutcome::Progressed;

  assert_eq!(RECORDED_OUTCOMES.to_vec(), RECORDED_OUTCOMES);
  assert_ne!(moved_stride_7, RECORDED_OUTCOMES);
  assert_eq!(
    moved_stride_7
      .iter()
      .zip(RECORDED_OUTCOMES)
      .position(|(seen, recorded)| seen != recorded),
    Some(6),
    "and the divergence the failure message names is stride 7",
  );

  // The recorded list is the trace's own payload, so a log and this constant
  // cannot drift apart silently.
  assert_eq!(
    joined_labels(RECORDED_OUTCOMES),
    "no_word_timings,awaiting_agreement,progressed,progressed,progressed,\
     progressed,stationary,progressed,stationary,progressed,progressed",
  );
}

const BATCH: &str = "and so my fellow americans ask not what your country can do for you";

/// A genuine truncation is what LocalAgreement-2 is FOR, and must pass.
#[test]
fn a_truncated_confirmation_is_accepted() {
  assert_eq!(
    check_confirmed_is_prefix_of_batch("and so my fellow americans", BATCH, 2),
    Ok(())
  );
  // The degenerate-but-legal ends: exactly the agreement width, and the whole
  // transcript.
  assert_eq!(
    check_confirmed_is_prefix_of_batch("and so", BATCH, 2),
    Ok(())
  );
  assert_eq!(check_confirmed_is_prefix_of_batch(BATCH, BATCH, 2), Ok(()));
}

/// THE falsifier for the prefix property: a confirmation that REWRITES a word
/// instead of stopping short must red, and must name the word.
#[test]
fn a_rewritten_word_is_not_a_truncation() {
  let why = check_confirmed_is_prefix_of_batch("and so my fellow australians", BATCH, 2)
    .expect_err("a rewritten word must not pass the prefix check");
  assert!(why.contains("REWRITES"), "{why}");
  assert!(why.contains("first divergence at word 4"), "{why}");
  assert!(why.contains("australians"), "{why}");

  // A word DROPPED from the middle is a rewrite too: everything after it shifts
  // left, so position 3 no longer holds `fellow`.
  let why = check_confirmed_is_prefix_of_batch("and so my americans", BATCH, 2)
    .expect_err("a dropped middle word must not pass the prefix check");
  assert!(why.contains("first divergence at word 3"), "{why}");
}

/// A confirmation that runs PAST the batch transcript is not a truncation
/// either — the shape a word confirmed TWICE produces, whether from
/// `LocalAgreement::finalize`'s double fold with its `holdback_superseded` guard
/// not doing its job, or from the re-admission strip letting an
/// already-confirmed word back into a hypothesis.
#[test]
fn a_confirmation_longer_than_the_batch_is_not_a_truncation() {
  let why = check_confirmed_is_prefix_of_batch(&format!("{BATCH} for you"), BATCH, 2)
    .expect_err("a confirmation longer than the batch must not pass");
  assert!(why.contains("LONGER"), "{why}");
  assert!(why.contains("finalize"), "{why}");
}

/// An EMPTY confirmation is a prefix of everything, so without a floor the
/// prefix check would be decorative. It must red instead.
#[test]
fn an_empty_confirmation_is_not_evidence() {
  let why = check_confirmed_is_prefix_of_batch("", BATCH, 2)
    .expect_err("an empty confirmation must not pass the prefix check");
  assert!(why.contains("confirmed 0 word(s)"), "{why}");
  assert!(why.contains("stall"), "{why}");

  // One word short of a completed agreement round is refused for the same
  // reason — the floor is the run's own `agreement_count_needed`, not zero.
  let why = check_confirmed_is_prefix_of_batch("and", BATCH, 2)
    .expect_err("a sub-agreement-width confirmation must not pass");
  assert!(why.contains("confirmed 1 word(s)"), "{why}");
}

/// A growing confirmation with a rising watermark is the healthy shape.
#[test]
fn a_growing_confirmation_is_accepted() {
  let steps = [
    step(1, 0.0, &[], &[]),
    step(2, 0.0, &[], &["and", "so"]),
    step(3, 0.62, &["and"], &["so", "my"]),
    step(4, 0.94, &["and", "so"], &["my", "fellow"]),
  ];
  assert_eq!(check_monotone_confirmation(&steps), Ok(()));
}

/// THE falsifier for monotonicity: a confirmed word rewritten by a later stride.
#[test]
fn a_retracted_confirmed_word_reds_monotonicity() {
  let steps = [
    step(1, 0.62, &["and", "so"], &["my", "fellow"]),
    step(2, 0.94, &["and", "then"], &["my", "fellow"]),
  ];
  let why = check_monotone_confirmation(&steps)
    .expect_err("a rewritten confirmed word must red monotonicity");
  assert!(why.contains("CONFIRMED word was rewritten"), "{why}");
  assert!(why.contains("at index 1"), "{why}");
  assert!(why.contains("then"), "{why}");
}

/// A confirmed list that gets SHORTER is a retraction with its own message.
#[test]
fn a_shrinking_confirmed_list_reds_monotonicity() {
  let steps = [
    step(1, 0.62, &["and", "so", "my"], &["fellow", "americans"]),
    step(2, 0.94, &["and", "so"], &["my", "fellow"]),
  ];
  let why = check_monotone_confirmation(&steps).expect_err("a shrinking confirmed list must red");
  assert!(why.contains("SHRANK"), "{why}");
  assert!(why.contains("3 -> 2 words"), "{why}");
}

/// A watermark that moves back re-opens settled text.
#[test]
fn a_receding_watermark_reds_monotonicity() {
  let steps = [
    step(1, 1.50, &["and", "so"], &["my", "fellow"]),
    step(2, 0.94, &["and", "so"], &["my", "fellow"]),
  ];
  let why = check_monotone_confirmation(&steps).expect_err("a receding watermark must red");
  assert!(why.contains("RECEDED"), "{why}");
  assert!(why.contains("1.5 s -> 0.94 s"), "{why}");
}

/// A run that never confirms anything satisfies every monotonicity check
/// trivially, so the predicate must refuse it rather than report a green.
#[test]
fn a_run_that_confirms_nothing_is_not_monotone_evidence() {
  let steps = [step(1, 0.0, &[], &[]), step(2, 0.0, &[], &[])];
  let why = check_monotone_confirmation(&steps).expect_err("a run with no confirmation must red");
  assert!(why.contains("confirmed and held back NOTHING"), "{why}");

  let why = check_monotone_confirmation(&[]).expect_err("a run with no pushes at all must red");
  assert!(why.contains("no pushes were observed"), "{why}");
}

// ---------------------------------------------------------------------
// The phrase gate's three host verdicts — hermetic, NOT model-gated
// ---------------------------------------------------------------------

/// A synthetic running host, deliberately not this machine's, so these run the
/// same everywhere.
fn synthetic_running_host() -> common::HostClass {
  common::HostClass {
    os_build: "25F71".to_string(),
    os_product_version: "26.5".to_string(),
    chip: "Apple M9 Ultra".to_string(),
    arch: "arm64".to_string(),
  }
}

/// The recorded constant matching [`synthetic_running_host`].
const SYNTHETIC_SAME_HOST: common::CharacterizedHost = common::CharacterizedHost {
  os_build: "25F71",
  os_product_version: "26.5",
  chip: "Apple M9 Ultra",
  arch: "arm64",
};

/// A recorded constant for a different chip — the axis the Neural Engine's fp16
/// arithmetic actually varies along, and the one separating this machine from
/// the CI runner.
const SYNTHETIC_OTHER_HOST: common::CharacterizedHost = common::CharacterizedHost {
  os_build: "24G720",
  os_product_version: "15.7.7",
  chip: "Apple M1 (Virtual)",
  arch: "arm64",
};

fn phrase_gate(recorded: Option<common::CharacterizedHost>) -> common::BandGate {
  common::BandGate::open_with(
    "whisper streaming confirmed phrase (hermetic)",
    recorded,
    synthetic_running_host(),
    recharacterize_command(),
  )
}

/// Calls the phrase gate exactly as the model-gated test does, catching the
/// panic an ARMED failure raises.
fn drive_phrase_gate(gate: &common::BandGate, present: bool) -> Result<String, ()> {
  catch_unwind(AssertUnwindSafe(|| {
    gate.check_holds(
      "confirmed stream reaches the clip's canonical phrase",
      present,
      &format!("contains {CANONICAL_PHRASE:?}"),
      "the confirmed stream never reached the canonical phrase",
    )
  }))
  .map_err(|_| ())
}

/// ARMED: on the host the phrase was measured on the gate is exactly as strict
/// as the unconditional assertion it replaced.
#[test]
fn an_armed_phrase_gate_is_strict() {
  let gate = phrase_gate(Some(SYNTHETIC_SAME_HOST));
  assert_eq!(gate.verdict(), common::BandVerdict::Armed);
  assert!(gate.armed());
  assert!(gate.banner().contains("ARE ASSERTED"), "{}", gate.banner());

  let line = drive_phrase_gate(&gate, true).expect("a present phrase must not panic");
  assert!(line.contains("[ASSERTED]"), "{line}");
  assert!(!line.contains("BAND NOT ASSERTED"), "{line}");

  assert!(
    drive_phrase_gate(&gate, false).is_err(),
    "an ARMED gate must still red on a missing phrase — otherwise host-scoping the \
     phrase would have deleted it"
  );
}

/// FOREIGN: the CI runner's case. Reported with both hosts named, never
/// asserted — arming a phrase on a machine known not to produce it would assert
/// something false.
#[test]
fn a_foreign_phrase_gate_reports_and_cannot_red() {
  let gate = phrase_gate(Some(SYNTHETIC_OTHER_HOST));
  assert_eq!(gate.verdict(), common::BandVerdict::Foreign);
  assert!(!gate.armed());

  let banner = gate.banner();
  assert!(banner.contains("NOT ASSERTED"), "{banner}");
  assert!(banner.contains("Apple M1 (Virtual)"), "{banner}");
  assert!(banner.contains("Apple M9 Ultra"), "{banner}");

  let line = drive_phrase_gate(&gate, false).expect("a foreign gate must never panic");
  assert!(line.contains("MISSING"), "{line}");
  assert!(line.contains("BAND NOT ASSERTED"), "{line}");
  assert!(
    line.contains("characterized on a different host class"),
    "{line}"
  );
}

/// UNRECORDED: no host in source, so nothing is known about which machine
/// produced the phrase. Reported, never asserted.
#[test]
fn an_unrecorded_phrase_gate_reports_and_cannot_red() {
  let gate = phrase_gate(None);
  assert_eq!(gate.verdict(), common::BandVerdict::Unrecorded);
  assert!(!gate.armed());
  assert!(
    gate.banner().contains("CHARACTERIZED_ON = None"),
    "{}",
    gate.banner()
  );

  let line = drive_phrase_gate(&gate, false).expect("an unrecorded gate must never panic");
  assert!(line.contains("MISSING"), "{line}");
  assert!(line.contains("no characterization host recorded"), "{line}");
}

/// Only MEASUREMENTS ride the host gate: the portable properties are asserted
/// bare, so no verdict can silence them. Checked against this file's real
/// source, the way siglip's `band_provenance` checks its own.
#[test]
fn only_measurements_are_host_scoped() {
  let source = std::fs::read_to_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/whisper/streaming.rs"
  ))
  .expect("this test file is readable");

  // Scoped to the model-gated test's own body, so this scan cannot match itself
  // or the hermetic verdict tests below.
  let body = source
    .split_once("fn jfk_simulated_stream_confirms_the_transcript(")
    .expect("the model-gated test is still here")
    .1
    .split_once("// Falsifiers for the portable predicates")
    .expect("the falsifier section still follows it")
    .0;

  // Exactly TWO measurements ride the gate, and both are named here: the
  // canonical phrase and the recorded outcome sequence. A third `gate.check_`
  // would mean a portable property had been moved behind a verdict that can
  // switch it off — which is the failure this scan exists to catch, and the
  // reason it counts rather than merely looking for the two it knows about.
  let gated: Vec<&str> = body
    .lines()
    .filter(|line| line.contains("gate.check_"))
    .collect();
  assert_eq!(
    gated.len(),
    2,
    "exactly two measurements may ride the host gate — the canonical phrase and \
     the recorded outcome sequence: {gated:?}"
  );
  for line in &gated {
    assert!(line.contains("check_holds"), "{line:?}");
  }
  // Named, so that swapping one of them for a portable property under the same
  // call shape does not slip past the count above.
  for measurement in [
    "\"per-stride agreement outcome sequence\",",
    "\"confirmed stream reaches the clip's canonical phrase\",",
  ] {
    assert!(
      body.contains(measurement),
      "the host-gated measurement `{measurement}` is gone; the count above \
       cannot tell a replacement from the original"
    );
  }

  // Both portable properties are called bare, so no verdict can silence them.
  for portable in [
    "check_monotone_confirmation(&steps)",
    "check_confirmed_is_prefix_of_batch(&confirmed, &batch, agreement_count_needed)",
    "check_outcomes_match_independent_evidence(&steps, agreement_count_needed)",
  ] {
    assert!(
      body.contains(portable),
      "portable property `{portable}` is no longer asserted in the model-gated test"
    );
  }

  // Host-scoping moved the phrase; it did not relax it.
  assert!(
    source.contains("const CANONICAL_PHRASE: &str = \"ask not what your country can do for you\";"),
    "the phrase must not be weakened, widened or deleted — it is host-scoped, not relaxed"
  );
}

// ── The seal: the engine exposes no public mutator ───────────────────────────

/// Every name on [`LocalAgreement`] that MOVES its state. Each is `pub(crate)`,
/// so the only way to drive the engine from outside this crate is through
/// [`LocalAgreementTranscriber`](coremlit::audio::whisper::stream::agreement::LocalAgreementTranscriber),
/// which orders the calls correctly by construction.
///
const SEALED_MUTATORS: &[&str] = &[
  "fn new(",
  "fn ingest(",
  "fn finalize(",
  "fn decoding_options_for_next(",
  "fn with_agreement_count_needed(",
  "fn set_agreement_count_needed(",
];

/// **THE SEAL'S FALSIFIER** (issue #94, M1). `LocalAgreement` stays `pub` and
/// fully readable — `confirmed_words_slice`, `last_agreed_words_slice`,
/// `last_agreed_seconds`, `results_slice`, `agreement_count_needed` are all
/// public — but nothing that MOVES its state is, because the correctness this
/// module argues for is a property of hypotheses the driver produced: the
/// holdback reproduction that makes an advance a re-agreement, the prefill
/// budget, and Rule W's postcondition all assume it.
///
/// A caller that wants its own DECODER still has one: `InferenceBackend` is
/// public and unsealed, and `WhisperKit::local_agreement_transcriber` sits on
/// `impl<B> WhisperKit<B>` with no bound, so a custom backend inherits this
/// whole stack. What the seal removes is "bring your own TRANSCRIPT", the one
/// shape that inherits none of it.
///
/// Grep rather than a compile-fail fixture, in the convention of
/// `tests/vad/reexport.rs::src_authors_no_detection_logic`: publishing any of
/// these again is a one-word edit, and a one-word edit should red a test.
#[test]
fn the_engine_exposes_no_public_mutator() {
  let source = std::fs::read_to_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/audio/whisper/stream/agreement/mod.rs"
  ))
  .expect("the engine's source is readable");

  // Scoped to the ENGINE's own inherent impl block. `LocalAgreementTranscriber`
  // shares three of these names (`new`, `finalize`,
  // `with_agreement_count_needed`) and is legitimately public — it IS the public
  // surface the seal leaves.
  let engine = source
    .split_once("\nimpl LocalAgreement {\n")
    .expect("the engine's inherent impl block is still here")
    .1
    .split_once("\n}\n")
    .expect("that impl block still closes at column 0")
    .0;

  // Non-vacuous: the block really does declare each of these, `pub(crate)`.
  // Without this a rename would silently empty the scan.
  for name in SEALED_MUTATORS {
    assert!(
      engine.contains(name),
      "`{name}` is gone from `impl LocalAgreement` — if it was renamed, rename \
       it in SEALED_MUTATORS too; this gate is worthless if it scans for nothing",
    );
  }

  let mut violations = Vec::new();
  for (lineno, line) in engine.lines().enumerate() {
    // Code only: a doc comment that NAMES a sealed item (this module's own docs
    // do, repeatedly) is not a re-publication.
    let code = line.split("//").next().unwrap_or("");
    for name in SEALED_MUTATORS {
      // `pub ` immediately in front, which `pub(crate) fn` does not contain.
      if code.contains(&format!("pub {name}")) || code.contains(&format!("pub const {name}")) {
        violations.push(format!(
          "impl LocalAgreement, line {}: `{name}` — {}",
          lineno + 1,
          line.trim(),
        ));
      }
    }
  }

  // A public trait impl IS a public constructor: `LocalAgreement::default()`
  // would hand out a fresh engine with no `new` in sight, and `Default` is
  // reachable through every generic bound that asks for it. Scanned over the
  // WHOLE file, since a trait impl lives outside the inherent block.
  for (lineno, line) in source.lines().enumerate() {
    if line
      .split("//")
      .next()
      .unwrap_or("")
      .contains("impl Default for LocalAgreement")
    {
      violations.push(format!(
        "mod.rs:{}: `impl Default for LocalAgreement` is a PUBLIC CONSTRUCTOR — {}",
        lineno + 1,
        line.trim(),
      ));
    }
  }

  assert!(
    violations.is_empty(),
    "the LocalAgreement engine must expose NO public mutator (issue #94, M1): \
     the correctness of `ingest` is a property of hypotheses \
     `LocalAgreementTranscriber` produced, and a caller handing in its own \
     inherits none of it. Found {} re-publication(s):\n{}",
    violations.len(),
    violations.join("\n"),
  );
}
