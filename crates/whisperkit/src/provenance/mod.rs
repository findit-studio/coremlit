//! Structured transcription provenance (coremlit issue #14, following
//! issue #9's "record what produced this transcript" recommendation):
//! [`Provenance`] bundles, in one serde-serializable record, the decode
//! facts that determine — or merely describe — a transcript, so a consumer
//! can persist them alongside the text instead of hand-assembling the
//! bundle from [`DecodingOptions`]/[`ComputeOptions`]/
//! [`TranscriptionSegment`] at every call site.
//!
//! # What this type can and cannot know
//!
//! This module is deliberately honest about the boundary, because a
//! provenance record that quietly invents a field is worse than one that
//! admits the gap:
//!
//! - **Library-known** — the **whole resolved [`DecodingOptions`]**, embedded
//!   verbatim ([`Provenance::decoding`]); the [`ComputeOptions`] the pipeline
//!   was built with; and — from the transcript itself — the language the
//!   decode actually **detected** and the *effective* temperature it actually
//!   landed on. [`Provenance::from_options`] fills in everything the options
//!   alone settle; [`Provenance::for_result`] adds the two outcome facts, and
//!   is the constructor to reach for when you have a transcript in hand.
//!
//!   Note *whole*, and note **embedded** rather than projected. An earlier
//!   shape hand-copied a curated subset of [`DecodingOptions`]' knobs into
//!   flat fields here, and it did what every hand-maintained projection does:
//!   it drifted. It reached 30 options captured as 11, and the 19 it silently
//!   dropped included
//!   [`DecodingOptions::drop_blank_audio`] and [`DecodingOptions::word_grouping`]
//!   — two knobs that visibly change the transcript, so two runs that differed
//!   only in them left **byte-identical records**. Embedding the struct makes
//!   completeness true *by construction*: a knob added to [`DecodingOptions`]
//!   tomorrow is captured here with no edit to this file, and cannot be
//!   forgotten. `provenance::tests`' mutation table enforces it — its coverage
//!   check is derived from [`DecodingOptions`]' own serialized key set, so a
//!   new field fails the suite until it is exercised.
//! - **Consumer-supplied** — three facts this crate cannot observe. All
//!   start `None`, stay `None` until the caller sets them, and are never
//!   guessed:
//!   - The model identity ([`Provenance::model_id`]/
//!     [`Provenance::model_revision`]) and the tokenizer identity
//!     ([`Provenance::tokenizer_id`]/[`Provenance::tokenizer_revision`]).
//!     These are *load-time* facts: this crate loads models and tokenizers
//!     from plain local folders ([`crate::options::Options`] holds nothing
//!     but two [`std::path::Path`]s; model auto-download is deferred, spec
//!     §4.7), so nothing in the pipeline ever sees a Hub repo id or a git
//!     revision. Only the caller — who *does* know which artifact it put
//!     in those folders — can say. This crate will not fabricate a
//!     revision it cannot observe.
//!   - The VAD detector ([`Provenance::vad_detector`]). It is
//!     **doubly** unobservable: the pipeline holds it as a
//!     `Box<dyn VoiceActivityDetector>`
//!     ([`crate::transcribe::WhisperKit::vad_detector`]), and a trait
//!     object carries no identity to read; and it lives on
//!     [`WhisperKit`](crate::transcribe::WhisperKit), not in
//!     [`DecodingOptions`] or [`ComputeOptions`], so the constructors here
//!     could not reach it even if it had a name. That is why the record's
//!     [`DecodingOptions::chunking_strategy`] says only *whether* VAD ran.
//!     Supplying the detector matters because swapping it
//!     ([`crate::transcribe::WhisperKit::set_vad_detector`]) moves the
//!     chunk boundaries, and the boundaries move the transcript — so two
//!     runs differing *only* in detector would otherwise leave
//!     byte-identical records with no trace of what made their text
//!     differ.
//!
//! # Why these fields are worth recording
//!
//! - **The compute unit changes the output.** CoreML's CPU, GPU, and
//!   Neural Engine paths do not produce bit-identical floating-point
//!   results, so a transcript is only reproducible against the same
//!   [`ComputeOptions`] — this crate's own golden baselines are pinned
//!   per-unit for exactly that reason. Recording the units is what makes a
//!   later mismatch diagnosable instead of mysterious.
//! - **A non-zero temperature is not reproducible without a seed.** The
//!   temperature-fallback ladder samples stochastically once it climbs off
//!   `0.0`, and [`DecodingOptions::seed`] is unset by default (OS-seeded,
//!   matching Swift's own unseeded draw). [`Provenance::effective_temperature`]
//!   is therefore the field that tells you whether the decode was
//!   greedy/deterministic (`Some(0.0)`) or sampled, and
//!   [`DecodingOptions::seed`] — embedded with the rest of the options —
//!   tells you whether that sampling can be replayed at all. Record both, or
//!   a re-run that disagrees is uninvestigable.
//! - **Auto-detect makes the *configured* language useless as a record.**
//!   It is `""` whenever the decoder is left to detect (the default
//!   pairing), so a record built from options alone names no language at
//!   all. The carried [`Provenance::task_facts`]'
//!   [`observed_language`](crate::task_facts::TaskFacts::observed_language) is
//!   the fact that carries what was actually spoken, and only
//!   [`Provenance::for_result`] — which is handed the transcript — can fill it in.

use coremlit::ComputeUnits;

use crate::{
  options::{ComputeOptions, DecodingOptions},
  result::{TranscriptionResult, TranscriptionSegment},
  task_facts::TaskFacts,
};

#[cfg(test)]
mod tests;

/// The one temperature every segment of a result agrees on, or `None` when
/// they do not — backing [`Provenance::effective_temperature`] for
/// [`Provenance::for_result`].
///
/// `None` for an empty slice too: the temperature-fallback ladder runs
/// per *window*, so a result is only describable by a single effective
/// temperature when every segment in it actually landed on the same rung,
/// and a result with no segments has nothing to land. (Both cases are
/// ordinary now — [`DecodingOptions::drop_blank_audio`] empties a silent
/// chunk outright.) Claiming a number here for either would be a
/// fabrication; `None` says the honest thing.
fn unanimous_temperature(segments: &[TranscriptionSegment]) -> Option<f32> {
  let first = segments.first()?.temperature();
  segments
    .iter()
    .all(|segment| segment.temperature() == first)
    .then_some(first)
}

/// A serde-serializable record of what produced a transcript: the resolved
/// decode configuration, the compute units it ran on, the language it
/// detected and the effective temperature it landed on, and — when the
/// caller supplies them — the model and tokenizer identity and the VAD
/// detector.
///
/// Build it with [`Self::for_result`] when you have the transcript (the
/// form that records the detected language and the effective temperature);
/// with [`Self::for_segment`] to record one segment's own rung of the
/// fallback ladder; or with [`Self::from_options`] from the configuration
/// alone. Then attach what the library cannot know:
///
/// ```
/// use whisperkit::{
///   options::{ComputeOptions, DecodingOptions},
///   provenance::Provenance,
/// };
///
/// let decoding = DecodingOptions::new().with_language("en");
/// let compute = ComputeOptions::new();
///
/// // `0.0` is the effective temperature the segment decoded at, and `false`
/// // the explicit draw fact (greedy at 0.0 never draws) — supply both from
/// // the decode in real use, never inferring the draw from the temperature.
/// let provenance = Provenance::from_options(&decoding, &compute, 0.0, false)
///   .with_model_id("openai_whisper-tiny")
///   .with_model_revision("a1b2c3d");
///
/// assert_eq!(provenance.decoding().language(), "en");
/// assert_eq!(provenance.model_id(), Some("openai_whisper-tiny"));
/// // Never fabricated: the tokenizer identity was not supplied.
/// assert_eq!(provenance.tokenizer_revision(), None);
/// // Nor is the VAD detector ever guessed — it is a `dyn` trait object on
/// // `WhisperKit`, so only the caller that installed it can name it.
/// assert_eq!(provenance.vad_detector(), None);
/// ```
///
/// The library-known fields are captured facts, so they are read-only
/// (there are no setters for them — reconstruct from the options instead);
/// only the five consumer-supplied fields are settable — the two identity
/// pairs and [`Self::vad_detector`] — and each serializes as **absent**
/// while unset rather than as `null`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Provenance {
  // -- library-known: the resolved decode configuration -----------------
  /// The **entire** resolved [`DecodingOptions`] the decode ran under,
  /// embedded verbatim — every knob, including the ones nobody thought to
  /// list.
  ///
  /// Embedded rather than projected into flat fields, and that is the whole
  /// design: a hand-curated projection is a list somebody has to remember to
  /// extend, and the one this replaced had already drifted to 11 of 30 knobs
  /// (see the module doc). Reading a knob costs one hop —
  /// `provenance.decoding().drop_blank_audio()` — and in exchange the record
  /// cannot be incomplete. [`DecodingOptions::detect_language`] is the
  /// resolved getter, so the tri-state is stored raw and still reads back
  /// resolved; `use_prefill_prompt` travels with it, so the coupling
  /// re-resolves to exactly what the pipeline acted on.
  ///
  /// Required on deserialize, and lossless: [`DecodingOptions`]' own wire
  /// form round-trips every value exactly (see that module's doc), so a
  /// persisted record reconstructs the run's configuration rather than
  /// approximating it.
  decoding: DecodingOptions,

  // -- library-known: compute + outcome ---------------------------------
  /// The per-stage CoreML compute units the pipeline ran on. Recorded
  /// because they change the output (see this module's docs).
  compute: ComputeOptions,
  /// The temperature the decode **actually landed on** — the fallback
  /// ladder's accepted attempt, read off
  /// [`TranscriptionSegment::temperature`]. Equal to
  /// [`DecodingOptions::temperature`] when no fallback was needed; higher
  /// when the ladder climbed.
  /// `Some(0.0)` means greedy/argmax and therefore deterministic; the
  /// overwhelmingly common no-fallback case.
  ///
  /// `None` means **no single temperature describes this transcript**. The
  /// ladder runs per *window*, so two segments of one result can
  /// legitimately land on different rungs; when they do, any single number
  /// here would be a lie about at least one of them. A result with no
  /// segments at all (silence, once
  /// [`DecodingOptions::drop_blank_audio`] has emptied it) is `None` for
  /// the same reason: nothing landed anywhere.
  /// [`Self::from_options`]/[`Self::for_segment`] are always `Some` — they
  /// are handed the one temperature they record.
  ///
  /// The one outcome fact NOT in [`Self::task_facts`]: it is *derived* from
  /// the surviving segments (their unanimous temperature), where the record's
  /// facts are *carried* out of the decode. Required on deserialize, and
  /// finite: bridged through `crate::options::finite_f32_option`, which keeps
  /// the field required (naming a `with` defeats serde's
  /// missing-`Option`-is-`None` special case) and refuses a non-finite value
  /// on both sides of the wire (codex round 3, F6). Without it, a `Some(-inf)`
  /// that `serde_json` collapses to `null` would read back as a forged `None`
  /// ("the ladder split the segments").
  #[cfg_attr(feature = "serde", serde(with = "crate::options::finite_f32_option"))]
  effective_temperature: Option<f32>,
  /// The decode-time facts this run **controlled** — the RNG draw, the
  /// genuinely observed language, the early-stop truncation, the worker
  /// coordinates, and the allocated id span — embedded verbatim as one record
  /// (coremlit issue #14, codex round 6). [`Self::for_result`] clones it whole
  /// off the [`TranscriptionResult`], so a fact the transcript carries cannot be
  /// dropped or fabricated on the way into the provenance the way its five
  /// former flat fields were; and [`Self::is_reproducible`] reads the two facts
  /// it rests on ([`TaskFacts::drew_from_rng`] and [`TaskFacts::early_stopped`])
  /// straight off it.
  ///
  /// [`Self::from_options`]/[`Self::for_segment`] — options with no decode to
  /// speak for — record [`TaskFacts::unknown`] with only the caller-supplied
  /// draw fact set: no observed language, an **unknown** truncation (the callback
  /// is unobservable, so `early_stopped` stays `None` — F1), an **unknown**
  /// worker schedule (never a fabricated `0`, R6-F2), and no id span.
  ///
  /// Required on deserialize, with its own explicit-unknown serde contract (see
  /// [`TaskFacts`]): the reproducibility facts inside it must never silently
  /// default to their optimistic values, the one direction this must not fail in.
  task_facts: TaskFacts,

  // -- consumer-supplied: load-time identity ----------------------------
  /// The model's identity (e.g. a Hub repo id), if the caller supplied it.
  /// Never inferred — see this module's docs.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  model_id: Option<String>,
  /// The model's revision (e.g. a git sha), if the caller supplied it.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  model_revision: Option<String>,
  /// The tokenizer's identity, if the caller supplied it.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  tokenizer_id: Option<String>,
  /// The tokenizer's revision, if the caller supplied it.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  tokenizer_revision: Option<String>,

  // -- consumer-supplied: the VAD detector --------------------------------
  /// Which VAD detector drove the chunking, if the caller supplied it.
  ///
  /// Never inferred — not from the detector's concrete type, not from
  /// [`std::any::type_name`], and not from
  /// [`DecodingOptions::chunking_strategy`] being
  /// [`ChunkingStrategy::Vad`](crate::options::ChunkingStrategy::Vad). The
  /// pipeline holds the detector as a
  /// `Box<dyn VoiceActivityDetector>`
  /// ([`crate::transcribe::WhisperKit::vad_detector`]), which exposes no
  /// identity to read, and it lives on
  /// [`WhisperKit`](crate::transcribe::WhisperKit) rather than in
  /// [`DecodingOptions`]/[`ComputeOptions`] — so no constructor here can
  /// reach it. Only the caller that installed it knows what it is.
  ///
  /// Worth supplying whenever it is not the default
  /// [`EnergyVad`](crate::audio::vad::EnergyVad): the detector decides
  /// where the chunk boundaries fall, and the boundaries decide the text,
  /// so two runs that differ *only* in detector yield different
  /// transcripts from records that are otherwise identical.
  /// [`DecodingOptions::chunking_strategy`] alone cannot tell them apart.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  vad_detector: Option<String>,
}

/// Names every field of [`Provenance`] exactly once, generating (for tests) both
/// a `PROVENANCE_FIELD_NAMES` roster and a compile-time exhaustiveness guard that
/// destructures `Provenance` WITHOUT `..`.
///
/// This is what extends the DecodingOptions-only completeness mechanism to the
/// TASK-LEVEL facts (coremlit issue #14, codex round 5). Add a field to the
/// struct and the guard below fails to compile until it is named here — and it
/// then lands in `provenance::tests`' `provenance_records_every_task_fact` /
/// `task_fact_mutations` coverage as an uncovered name until it is either
/// exercised by a mutation (a task fact) or listed among the non-task-fact
/// partitions (the embedded options and the consumer-supplied identity). So a
/// new outcome a run controls cannot be added to the record and left unread by
/// [`Provenance::for_result`]: the mutation that moves it would produce a record
/// identical to the baseline, failing the suite.
macro_rules! provenance_field_names {
  ($($field:ident),+ $(,)?) => {
    /// The full field set of [`Provenance`], one entry per field. Kept
    /// exhaustive at compile time by the guard in the same macro expansion.
    /// Consumed by the provenance task-fact coverage test.
    #[cfg(test)]
    #[allow(dead_code)] // used only by the provenance coverage test
    pub(crate) const PROVENANCE_FIELD_NAMES: &[&str] = &[$(stringify!($field)),+];

    /// A pure compile-time exhaustiveness check: destructuring without `..`
    /// forces every `Provenance` field to be named in the list above. Never
    /// called.
    #[cfg(test)]
    #[allow(dead_code)]
    fn _provenance_field_exhaustiveness_guard(provenance: Provenance) {
      let Provenance { $($field: _),+ } = provenance;
    }
  };
}

provenance_field_names!(
  decoding,
  compute,
  effective_temperature,
  task_facts,
  model_id,
  model_revision,
  tokenizer_id,
  tokenizer_revision,
  vad_detector,
);

impl Provenance {
  /// The shared capture: the resolved `decoding`/`compute`, the derived
  /// `effective_temperature`, and the carried [`TaskFacts`] — with the identity
  /// left `None`.
  fn capture(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    effective_temperature: Option<f32>,
    task_facts: TaskFacts,
  ) -> Self {
    Self {
      // The WHOLE options value, not a field-by-field projection — the one
      // line that makes this capture complete by construction, and keeps it
      // complete for every knob added after today (see the field's doc).
      decoding: decoding.clone(),
      compute: *compute,
      effective_temperature,
      // The WHOLE task-facts record, likewise — every run-controlled fact
      // travels in one carried value with one merge law, so none can be dropped
      // or fabricated on the way in (coremlit issue #14, codex round 6).
      task_facts,
      model_id: None,
      model_revision: None,
      tokenizer_id: None,
      tokenizer_revision: None,
      // Structurally unreachable from here, and never guessed: the
      // detector lives on `WhisperKit`, behind a `dyn` trait object with
      // no identity to read (see the field's doc).
      vad_detector: None,
    }
  }

  /// Captures every library-known fact from `decoding`/`compute` plus the
  /// `effective_temperature` a decode landed on, leaving the model and
  /// tokenizer identity unset (`None`) for the caller to fill in — this
  /// crate loads from bare local folders and genuinely does not know it
  /// (see the module docs).
  ///
  /// The [`Self::task_facts`] observed language is left `None`: options alone
  /// cannot know what language was detected. Reach for [`Self::for_result`] when
  /// you have the transcript — that is the constructor that records it.
  ///
  /// `effective_temperature` is per-*segment*, not per-result: the
  /// fallback ladder runs per window, so two segments of one transcript
  /// can legitimately land on different temperatures. Pass
  /// [`TranscriptionSegment::temperature`] for the segment being recorded
  /// — or use [`Self::for_segment`], which does exactly that.
  ///
  /// `sampled_at_nonzero_temperature` is the **explicit** draw fact — whether
  /// the decode actually drew from the token sampler — and is **never
  /// inferred** from `effective_temperature` here. A temperature is not a
  /// witness of a draw: a decode that ran zero sampling iterations (or only
  /// the all-masked degenerate path) lands a non-zero temperature yet never
  /// touched the RNG, so `effective_temperature != 0.0` would over-report it
  /// as sampled and non-reproducible (F3, codex round 4). Options-and-a-
  /// -temperature alone cannot observe the draw, so the caller — who ran the
  /// decode — supplies the fact; a caller with only the configuration and no
  /// decode to speak for should pass `false`.
  ///
  /// This constructor cannot observe an early-stop truncation either (it is
  /// never handed the progress callback), so it records
  /// [`TaskFacts::early_stopped`] as an explicit unknown and the resulting
  /// record is conservatively NOT [`reproducible`](Self::is_reproducible),
  /// whatever the draw fact or [`DecodingOptions::seed`] (F1, codex round 6
  /// post-consolidation). [`Self::for_result`] is the constructor to reach for a
  /// reproducibility answer: it fills the draw AND the truncation from the
  /// transcript's own carried, POSITIVELY observed facts.
  pub fn from_options(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    effective_temperature: f32,
    sampled_at_nonzero_temperature: bool,
  ) -> Self {
    Self::capture(
      decoding,
      compute,
      Some(effective_temperature),
      // Options with no decode to speak for: only the caller-supplied draw fact
      // is known. No observed language, an UNKNOWN truncation (the constructor
      // is never handed the progress callback, so `early_stopped` stays `None`,
      // never a fabricated `not-truncated` — F1), an UNKNOWN worker schedule
      // (never a fabricated `0` — R6-F2), and no id span. Because the truncation
      // is unknown, such a record is conservatively NOT reproducible; `for_result`
      // fills the honest facts in from the transcript's own carried record.
      TaskFacts::unknown().with_drew_from_rng(sampled_at_nonzero_temperature),
    )
  }

  /// [`Self::from_options`] with the effective temperature read straight
  /// off a decoded `segment` — the ergonomic form when recording
  /// provenance for one segment of a transcript you already have.
  ///
  /// Per-segment for the reason spelled out on [`Self::from_options`]:
  /// only the segment knows which rung of the fallback ladder its decode
  /// was accepted at. For the whole transcript, use [`Self::for_result`].
  ///
  /// `sampled_at_nonzero_temperature` stays an **explicit** argument rather
  /// than being read off the segment: a [`TranscriptionSegment`] carries its
  /// temperature but not the draw fact, and the two come apart (a
  /// zero-iteration decode lands a non-zero-temperature segment that never
  /// drew), so inferring it from `segment.temperature()` is exactly the F3
  /// bug this constructor must not reintroduce. Only the caller that ran the
  /// decode knows it.
  pub fn for_segment(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    segment: &TranscriptionSegment,
    sampled_at_nonzero_temperature: bool,
  ) -> Self {
    Self::from_options(
      decoding,
      compute,
      segment.temperature(),
      sampled_at_nonzero_temperature,
    )
  }

  /// The **result-level** capture: [`Self::from_options`]'s options, the
  /// derived effective temperature, and the whole [`TaskFacts`] record the
  /// transcript carries.
  ///
  /// - [`Self::task_facts`] is **cloned whole** off the result
  ///   ([`TranscriptionResult::task_facts`]): the observed language (never
  ///   [`TranscriptionResult::language`]'s Swift-compat `"en"` display fallback
  ///   — "record what produced the transcript, invent nothing"), the carried
  ///   draw fact (the sampler's own [`drew_from_rng`](TaskFacts::drew_from_rng),
  ///   accumulated across every attempt before any filter could delete the
  ///   window that sampled — never inferred from a segment temperature, F3 codex
  ///   round 4), the early-stop truncation (R6-F1), the worker schedule, and the
  ///   id span. One clone replaces the five separate reads this used to
  ///   fabricate or drop facts through, and is a fact [`Self::for_segment`]
  ///   structurally cannot reach (it is handed a segment, not the result).
  /// - [`Self::effective_temperature`] becomes `Some(t)` **iff every
  ///   segment landed on the same `t`** — the overwhelmingly common
  ///   no-fallback case, which yields `Some(0.0)` — and `None` when the
  ///   per-window fallback ladder split them, or when the result has no
  ///   segments at all to agree (silence, after
  ///   [`DecodingOptions::drop_blank_audio`] empties it). A result-level
  ///   `f32` would have had to invent a number for both.
  ///
  /// The model/tokenizer identity is still the caller's to supply — a
  /// result cannot know it either.
  ///
  /// ```
  /// use whisperkit::{
  ///   options::{ComputeOptions, DecodingOptions},
  ///   provenance::Provenance,
  ///   result::{TranscriptionResult, TranscriptionTimings},
  ///   task_facts::TaskFacts,
  /// };
  ///
  /// // Auto-detect: the CONFIGURED language is empty, while the run OBSERVED
  /// // English. The pipeline records the observation on the result's task facts;
  /// // here it is set explicitly, since this is a hand-built one (a genuine
  /// // observation is never inferred from the "en" display string — F3).
  /// let decoding = DecodingOptions::new();
  /// let compute = ComputeOptions::new();
  /// let result = TranscriptionResult::new(
  ///   "Hello world.",
  ///   Vec::new(),
  ///   "en",
  ///   TranscriptionTimings::new(),
  /// )
  /// .with_task_facts(TaskFacts::unknown().with_observed_language(Some("en".to_string())));
  ///
  /// let provenance = Provenance::for_result(&decoding, &compute, &result);
  /// assert_eq!(provenance.decoding().language(), "");
  /// // ... and the OBSERVED one is the fact actually worth persisting.
  /// assert_eq!(provenance.task_facts().observed_language(), Some("en"));
  /// // No segments landed anywhere, so no single temperature describes it.
  /// assert_eq!(provenance.effective_temperature(), None);
  /// ```
  pub fn for_result(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    result: &TranscriptionResult,
  ) -> Self {
    Self::capture(
      decoding,
      compute,
      unanimous_temperature(result.segments_slice()),
      // The WHOLE carried record, cloned off the transcript: every run-controlled
      // fact travels in one value with one merge law, so none can be dropped or
      // fabricated on the way into the provenance (coremlit issue #14, codex
      // round 6) the way its five former flat reads were.
      result.task_facts().clone(),
    )
  }

  // -- decoding -----------------------------------------------------------
  /// The **whole** resolved [`DecodingOptions`] the decode ran under — the
  /// single door to every decode-time fact this record carries.
  ///
  /// There are deliberately no per-knob projections beside it
  /// (`provenance.decoding().task()`, not `provenance.task()`): a projection
  /// is a list that has to be kept in step with [`DecodingOptions`], and the
  /// one this replaced fell 19 knobs behind (module doc). One accessor cannot.
  ///
  /// ```
  /// use whisperkit::{
  ///   options::{ComputeOptions, DecodingOptions, WordGrouping},
  ///   provenance::Provenance,
  /// };
  ///
  /// let decoding = DecodingOptions::new()
  ///   .maybe_drop_blank_audio(false)
  ///   .with_word_grouping(WordGrouping::SwiftParity);
  /// let provenance = Provenance::from_options(&decoding, &ComputeOptions::new(), 0.0, false);
  ///
  /// // Knobs the old projection dropped on the floor, now recorded:
  /// assert!(!provenance.decoding().drop_blank_audio());
  /// assert_eq!(provenance.decoding().word_grouping(), WordGrouping::SwiftParity);
  /// // ... alongside the ones it did keep.
  /// assert_eq!(provenance.decoding(), &decoding);
  /// ```
  #[inline(always)]
  pub const fn decoding(&self) -> &DecodingOptions {
    &self.decoding
  }

  // -- compute ------------------------------------------------------------
  /// The per-stage CoreML compute units the pipeline ran on.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeOptions {
    self.compute
  }

  /// The audio encoder's compute units — the stage whose unit most visibly
  /// moves the output, and the one a baseline is usually pinned against.
  #[inline(always)]
  pub const fn encoder_compute_units(&self) -> ComputeUnits {
    self.compute.encoder()
  }

  // -- effective_temperature ----------------------------------------------
  /// The temperature the decode actually landed on. `Some(0.0)` means
  /// greedy and therefore deterministic; anything higher was sampled, and
  /// is only reproducible if [`DecodingOptions::seed`] is set. `None` means no single
  /// temperature describes the transcript — the per-window fallback ladder
  /// split its segments, or it has no segments. See the field's doc.
  #[inline(always)]
  pub const fn effective_temperature(&self) -> Option<f32> {
    self.effective_temperature
  }

  // -- task_facts ---------------------------------------------------------
  /// The decode-time facts this run **controlled**, as one carried record —
  /// the RNG draw, the genuinely observed language, the early-stop truncation,
  /// a swallowed child error, the worker coordinates, and the allocated id span.
  /// Read
  /// [`provenance.task_facts().observed_language()`](TaskFacts::observed_language),
  /// [`.drew_from_rng()`](TaskFacts::drew_from_rng),
  /// [`.early_stopped()`](TaskFacts::early_stopped),
  /// [`.had_swallowed_error()`](TaskFacts::had_swallowed_error),
  /// [`.worker_schedule()`](TaskFacts::worker_schedule), and
  /// [`.decoded_span()`](TaskFacts::decoded_span) off it. See the field's doc.
  #[inline(always)]
  pub const fn task_facts(&self) -> &TaskFacts {
    &self.task_facts
  }

  /// Whether this transcript can be reproduced byte-for-byte by re-running
  /// the same audio through the same options: true when the decode never
  /// drew from the sampler (every window accepted greedily at `0.0`), or
  /// when a [`DecodingOptions::seed`] makes the draws it did make
  /// replayable — AND no progress callback truncated it.
  ///
  /// # It reads recorded facts, and deliberately does not infer them
  ///
  /// The predicate rests on [`TaskFacts::drew_from_rng`] — a fact the decode
  /// path *carried out* of the window loop — and **not** on
  /// [`Self::effective_temperature`], which describes only the segments that
  /// survived to the end.
  ///
  /// Inferring it from the survivors is exactly the bug this replaced. The
  /// blank-audio drop, the word-timestamp zero-length filter, a no-speech
  /// window, and an emptied VAD chunk each delete a whole window's segments;
  /// a window accepted at `0.2` that sampled `[BLANK_AUDIO]` from an
  /// unseeded RNG therefore leaves a transcript whose every surviving
  /// segment reads `0.0`. Reconstructed from those, the answer was
  /// `Some(0.0)` -> "greedy" -> **`true`** — a byte-reproducibility
  /// guarantee the run could not honor, since a re-run redraws that window
  /// and the text it lands on next time may well survive the filter.
  ///
  /// Because the fact is now carried rather than reconstructed, this is also
  /// *more* precise than the old conservative fallbacks: an all-greedy run
  /// whose every segment was dropped (pure silence, blank-dropped) is
  /// correctly reproducible, where the old `None`-means-unknown rule had to
  /// guess `false`.
  ///
  /// A seed makes *this port's* output reproducible; it cannot make that
  /// output match Swift's, which has no seed knob and always draws
  /// unseeded (see [`DecodingOptions::seed`]).
  ///
  /// # A callback truncation — or an unobservable one — is never reproducible
  ///
  /// An observed [`TaskFacts::early_stopped`] of `Some(true)` independently
  /// forces `false`. A progress callback returning `Some(false)` truncates the
  /// transcript — a CONTROL action — but the callback is a closure with no
  /// readable identity, so it is not part of this record. A re-run from the
  /// recorded options and seed alone therefore cannot reproduce the truncation:
  /// two runs differing only in that callback otherwise both claimed
  /// reproducibility (coremlit issue #14, codex round 5). This keys on the
  /// OUTCOME (whether a stop actually fired), not the presence of a callback, so
  /// a callback that only observed and never truncated (`Some(false)`) leaves
  /// reproducibility untouched — its transcript IS the un-truncated one the
  /// options and seed reproduce.
  ///
  /// An UNOBSERVED truncation (`None`) is treated the same as an observed one:
  /// conservatively non-reproducible (F1, codex round 6 post-consolidation).
  /// [`Self::from_options`]/[`Self::for_segment`] cannot see the callback — they
  /// are handed options and at most one segment — so they record `None`, and a
  /// record built through them is never reproducible, whatever the temperature
  /// or seed. Reach for [`Self::for_result`] for a reproducibility answer: it
  /// reads the transcript's carried, POSITIVELY observed facts. The whole
  /// predicate is [`TaskFacts::is_reproducible_under`], to which this supplies
  /// whether a [`DecodingOptions::seed`] is set.
  #[inline(always)]
  pub const fn is_reproducible(&self) -> bool {
    self
      .task_facts
      .is_reproducible_under(self.decoding.seed().is_some())
  }

  // -- model_id (Option<String>) ------------------------------------------
  /// The model's identity, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn model_id(&self) -> Option<&str> {
    self.model_id.as_deref()
  }
  /// Builder form of [`Self::set_model_id`].
  #[must_use]
  #[inline(always)]
  pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
    self.set_model_id(model_id);
    self
  }
  /// Sets [`Self::model_id`] to `Some(model_id)`.
  #[inline(always)]
  pub fn set_model_id(&mut self, model_id: impl Into<String>) -> &mut Self {
    self.model_id = Some(model_id.into());
    self
  }
  /// Builder form of [`Self::update_model_id`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_model_id(mut self, model_id: Option<String>) -> Self {
    self.update_model_id(model_id);
    self
  }
  /// Assigns [`Self::model_id`] directly.
  #[inline(always)]
  pub fn update_model_id(&mut self, model_id: Option<String>) -> &mut Self {
    self.model_id = model_id;
    self
  }
  /// Sets [`Self::model_id`] to `None`.
  #[inline(always)]
  pub fn clear_model_id(&mut self) -> &mut Self {
    self.model_id = None;
    self
  }

  // -- model_revision (Option<String>) ------------------------------------
  /// The model's revision, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn model_revision(&self) -> Option<&str> {
    self.model_revision.as_deref()
  }
  /// Builder form of [`Self::set_model_revision`].
  #[must_use]
  #[inline(always)]
  pub fn with_model_revision(mut self, model_revision: impl Into<String>) -> Self {
    self.set_model_revision(model_revision);
    self
  }
  /// Sets [`Self::model_revision`] to `Some(model_revision)`.
  #[inline(always)]
  pub fn set_model_revision(&mut self, model_revision: impl Into<String>) -> &mut Self {
    self.model_revision = Some(model_revision.into());
    self
  }
  /// Builder form of [`Self::update_model_revision`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_model_revision(mut self, model_revision: Option<String>) -> Self {
    self.update_model_revision(model_revision);
    self
  }
  /// Assigns [`Self::model_revision`] directly.
  #[inline(always)]
  pub fn update_model_revision(&mut self, model_revision: Option<String>) -> &mut Self {
    self.model_revision = model_revision;
    self
  }
  /// Sets [`Self::model_revision`] to `None`.
  #[inline(always)]
  pub fn clear_model_revision(&mut self) -> &mut Self {
    self.model_revision = None;
    self
  }

  // -- tokenizer_id (Option<String>) --------------------------------------
  /// The tokenizer's identity, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn tokenizer_id(&self) -> Option<&str> {
    self.tokenizer_id.as_deref()
  }
  /// Builder form of [`Self::set_tokenizer_id`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokenizer_id(mut self, tokenizer_id: impl Into<String>) -> Self {
    self.set_tokenizer_id(tokenizer_id);
    self
  }
  /// Sets [`Self::tokenizer_id`] to `Some(tokenizer_id)`.
  #[inline(always)]
  pub fn set_tokenizer_id(&mut self, tokenizer_id: impl Into<String>) -> &mut Self {
    self.tokenizer_id = Some(tokenizer_id.into());
    self
  }
  /// Builder form of [`Self::update_tokenizer_id`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_tokenizer_id(mut self, tokenizer_id: Option<String>) -> Self {
    self.update_tokenizer_id(tokenizer_id);
    self
  }
  /// Assigns [`Self::tokenizer_id`] directly.
  #[inline(always)]
  pub fn update_tokenizer_id(&mut self, tokenizer_id: Option<String>) -> &mut Self {
    self.tokenizer_id = tokenizer_id;
    self
  }
  /// Sets [`Self::tokenizer_id`] to `None`.
  #[inline(always)]
  pub fn clear_tokenizer_id(&mut self) -> &mut Self {
    self.tokenizer_id = None;
    self
  }

  // -- tokenizer_revision (Option<String>) --------------------------------
  /// The tokenizer's revision, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn tokenizer_revision(&self) -> Option<&str> {
    self.tokenizer_revision.as_deref()
  }
  /// Builder form of [`Self::set_tokenizer_revision`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokenizer_revision(mut self, tokenizer_revision: impl Into<String>) -> Self {
    self.set_tokenizer_revision(tokenizer_revision);
    self
  }
  /// Sets [`Self::tokenizer_revision`] to `Some(tokenizer_revision)`.
  #[inline(always)]
  pub fn set_tokenizer_revision(&mut self, tokenizer_revision: impl Into<String>) -> &mut Self {
    self.tokenizer_revision = Some(tokenizer_revision.into());
    self
  }
  /// Builder form of [`Self::update_tokenizer_revision`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_tokenizer_revision(mut self, tokenizer_revision: Option<String>) -> Self {
    self.update_tokenizer_revision(tokenizer_revision);
    self
  }
  /// Assigns [`Self::tokenizer_revision`] directly.
  #[inline(always)]
  pub fn update_tokenizer_revision(&mut self, tokenizer_revision: Option<String>) -> &mut Self {
    self.tokenizer_revision = tokenizer_revision;
    self
  }
  /// Sets [`Self::tokenizer_revision`] to `None`.
  #[inline(always)]
  pub fn clear_tokenizer_revision(&mut self) -> &mut Self {
    self.tokenizer_revision = None;
    self
  }

  // -- vad_detector (Option<String>) ---------------------------------------
  /// Which VAD detector drove the chunking, if the caller supplied it.
  /// Never inferred — see the field's doc.
  #[inline(always)]
  pub fn vad_detector(&self) -> Option<&str> {
    self.vad_detector.as_deref()
  }
  /// Builder form of [`Self::set_vad_detector`].
  #[must_use]
  #[inline(always)]
  pub fn with_vad_detector(mut self, vad_detector: impl Into<String>) -> Self {
    self.set_vad_detector(vad_detector);
    self
  }
  /// Sets [`Self::vad_detector`] to `Some(vad_detector)`.
  #[inline(always)]
  pub fn set_vad_detector(&mut self, vad_detector: impl Into<String>) -> &mut Self {
    self.vad_detector = Some(vad_detector.into());
    self
  }
  /// Builder form of [`Self::update_vad_detector`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_vad_detector(mut self, vad_detector: Option<String>) -> Self {
    self.update_vad_detector(vad_detector);
    self
  }
  /// Assigns [`Self::vad_detector`] directly.
  #[inline(always)]
  pub fn update_vad_detector(&mut self, vad_detector: Option<String>) -> &mut Self {
    self.vad_detector = vad_detector;
    self
  }
  /// Sets [`Self::vad_detector`] to `None`.
  #[inline(always)]
  pub fn clear_vad_detector(&mut self) -> &mut Self {
    self.vad_detector = None;
    self
  }
}
