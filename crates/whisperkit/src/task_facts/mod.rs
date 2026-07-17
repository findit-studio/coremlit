//! The decode-time facts a transcription **run controls** — the RNG draw, the
//! genuinely observed language, the early-stop truncation, the worker
//! coordinates its RNG streams rode, and the id-ordinal span it allocated —
//! consolidated into ONE carried record (coremlit issue #14, codex round 6).
//!
//! # Why one record
//!
//! Each of these facts is *born* at the attempt/decode layer, must *survive*
//! every aggregation boundary (temperature-fallback selection, the blank-audio
//! drop, a VAD chunk merge, a streaming finalize, an errored-and-dropped
//! chunk), and is finally *read* by [`Provenance`](crate::provenance::Provenance)
//! to answer [`Provenance::is_reproducible`](crate::provenance::Provenance::is_reproducible)
//! and to describe the run. Carrying them as five separate fields — plus two
//! ad-hoc decode-layer sinks and a scatter of `.any()`/`.find_map()`/`.first()`
//! re-aggregations at each merge — is exactly how three of them came to be lost
//! or fabricated at a boundary across six review rounds:
//!
//! - a rejected-because-early-stopped fallback attempt's truncation was read
//!   only off the *accepted* attempt, so it vanished when a later attempt was
//!   selected (R6-F1);
//! - a merge kept only the *first* child's worker coordinate, collapsing
//!   `[0, 2]` and `[0, 1]` to the same record, and a missing/hand-built
//!   coordinate defaulted to a fabricated worker `0` (R6-F2);
//! - a merge advanced its running id base per child but never *stored* the
//!   aggregate, so a staged re-merge (VAD result → streaming finalize)
//!   renumbered segments differently than a one-shot merge (R6-F3).
//!
//! [`TaskFacts`] gives the five facts one home, one [merge law](TaskFacts::merge)
//! that every merge entry point calls, and one serde contract — so a boundary
//! can no longer silently drop or invent one.
//!
//! # Explicit unknown, never a fabricated default
//!
//! Every fact this record carries has an **explicit-unknown** state distinct
//! from any observed value: `None` for [`TaskFacts::observed_language`] and
//! [`TaskFacts::worker_schedule`], and — since codex round 6's post-consolidation
//! F1 — `Option<bool>`'s `None` for [`TaskFacts::drew_from_rng`],
//! [`TaskFacts::early_stopped`], and (codex round 11) [`TaskFacts::had_swallowed_error`]
//! too. A record built from options with no decode to
//! speak for ([`Provenance::from_options`](crate::provenance::Provenance::from_options)),
//! or a transcript assembled by hand, knows no worker coordinate, witnessed no
//! language, and **cannot observe** whether the decode drew from the sampler, a
//! callback truncated it, or a child error was swallowed — and says exactly that,
//! rather than the worker `0`, `""`-language, or the optimistic `drew = false` /
//! `not-truncated` / `nothing-swallowed` a default would forge.
//!
//! The distinction is not cosmetic, and it is why the merge is **Kleene**
//! three-valued logic rather than a free monoid over `Option<bool>`.
//!
//! - As an **epistemic unknown**, an unknown boolean forces
//!   [`is_reproducible_under`](TaskFacts::is_reproducible_under) to answer
//!   CONSERVATIVELY: a record that cannot know whether a transcript-controlling
//!   event (an RNG draw, a callback truncation) occurred must NOT promise
//!   byte-reproducibility. The old `false`-means-both representation handed that
//!   promise to a genuinely truncated segment recorded through
//!   [`Provenance::for_segment`](crate::provenance::Provenance::for_segment),
//!   whose constructor cannot see the truncation (F1). Only a fact POSITIVELY
//!   observed as `Some(false)` — the shape a real decode carries out of the
//!   window loop — earns the optimistic answer.
//! - Under [the merge law](TaskFacts::merge) the same unknown OR-s by
//!   **Kleene's** rule (`kleene_or`): `Some(true)` absorbs, `Some(false) |
//!   Some(false)` stays `Some(false)`, and an unknown mixed with
//!   anything-but-true stays unknown (`None | Some(false) = None`) — a
//!   contributor that cannot see the fact cannot certify the other's `false`.
//!   The OR identity is therefore `Some(false)` (observed-clean), NOT `None`
//!   (codex round 8, F2): the pre-round-8 free monoid that treated `None` as the
//!   identity let `or_unknown(None, Some(false))` forge an observed-`false` out
//!   of an unobserved contributor, so a merge of a genuinely-unknown record into
//!   a known-clean one read back reproducible it never earned.
//!
//! Because `None` is no longer the OR identity, [`TaskFacts::unknown`] is no
//! longer the identity of a CONTRIBUTOR fold either. A fold seeds from a
//! separate "no contributor yet" (`TaskFactsAccumulator`) identity and takes the
//! first contributor verbatim, so an all-unknown `unknown()` can never
//! masquerade as the neutral element and silently null a known observation. A
//! **real run** that is watching starts its own fact sink at
//! [`TaskFacts::observed_clean`] (`Some(false)` for both booleans — it has
//! POSITIVELY seen nothing happen yet), so a run that decodes no window at all
//! keeps the reproducible `Some(false)`, while `unknown()` stays reserved for
//! the hand-built, deserialized-absent, and segment-/options-only records that
//! genuinely cannot see (codex round 8, F3).

#[cfg(test)]
mod tests;

/// Deserializes a **required but nullable** `Option` field — the same
/// required-field helper [`Provenance`](crate::provenance::Provenance) uses, so
/// a dropped key is rejected rather than silently read back as `None`.
///
/// Serde's derive special-cases a *missing* `Option` field to `None` even with
/// no `serde(default)` on it, which for [`Self`](TaskFacts)'s explicit-unknown
/// fields would forge "no language observed" / "no worker coordinate" out of a
/// field the writer merely dropped. Naming a `deserialize_with` sends the derive
/// down its required-field path instead: the key must be present, and `null`
/// then carries its real meaning — explicit unknown — and nothing else.
#[cfg(feature = "serde")]
fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
  D: serde::Deserializer<'de>,
  T: serde::Deserialize<'de>,
{
  <Option<T> as serde::Deserialize<'de>>::deserialize(deserializer)
}

/// **Kleene** (strong three-valued) logical OR over [`TaskFacts`]'s
/// explicit-unknown booleans: `Some(true)` absorbs, `Some(false) | Some(false)`
/// stays `Some(false)`, and an unknown (`None`) mixed with anything-but-true
/// stays unknown — `None | Some(false) = None`, because a contributor that
/// cannot see the fact cannot certify the OTHER contributor's `false` (codex
/// round 8, F2). The full table:
///
/// | a \ b       | `Some(true)` | `Some(false)` | `None` |
/// |-------------|--------------|---------------|--------|
/// | `Some(true)`  | `Some(true)` | `Some(true)`  | `Some(true)` |
/// | `Some(false)` | `Some(true)` | `Some(false)` | `None` |
/// | `None`        | `Some(true)` | `None`        | `None` |
///
/// The OR **identity is `Some(false)`** (observed-clean), NOT `None`: a genuine
/// unknown poisons the fold, so it can no longer double as the neutral element
/// the way the pre-round-8 free monoid (`None`-as-identity) did — the very
/// conflation that let a merge forge an observed-`false` out of an unobserved
/// contributor. Kleene OR is nonetheless associative and commutative, which is
/// what keeps [the contributor merge](TaskFacts::merge) associative over the
/// tri-state; the fold's own identity moves to `TaskFactsAccumulator`.
const fn kleene_or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
  match (a, b) {
    (Some(true), _) | (_, Some(true)) => Some(true),
    (Some(false), Some(false)) => Some(false),
    // `Some(false) | None`, `None | Some(false)`, `None | None` — an unknown
    // that cannot be certified `false` by a mere observed `false` beside it.
    _ => None,
  }
}

/// What a transcript's decode knows about the number of segment-id ordinals it
/// **allocated** — the id span [`merge_transcription_results_with_options`](crate::result::merge_transcription_results_with_options)
/// advances its running base by. **Two states**, because the fact has to tell an
/// EXACTLY-known total apart from a known LOWER BOUND on an otherwise-unknown
/// one, and the pre-round-12 `Option<usize>` could express only "exactly `n`"
/// (`Some(n)`) and "wholly unknown" (`None`) — never "unknown total, but at
/// least `k`" (coremlit issue #14, codex round 12).
///
/// That third state is not hypothetical. A VAD run that drops an errored chunk
/// keeps its SURVIVING chunks' known ordinal count as a lower bound while the
/// dropped chunk's contribution stays unknown; the pre-round-12 `None` — which
/// **absorbed** under [the merge](TaskFacts::merge) — threw that lower bound
/// away, so a staged re-merge, unable to recover the dropped chunk's ordinal
/// from the survivors' extent alone, renumbered a trailing chunk onto an id a
/// one-shot merge left free (the round-12 regression
/// `drop_on_merge_preserves_known_empty_span_after_unknown_prefix`: records
/// `A`=unknown-span-with-a-survivor, `B`=known-empty-span, `T`=a trailing
/// survivor, numbered `[0, 2]` one-shot but `[0, 1]` staged). Carrying the lower
/// bound explicitly makes [the merge](Self::merge) associative **by
/// construction** — every grouping stores the same value, and the id-base
/// advance derives from it — so one-shot and staged merges number identically.
///
/// - [`Exact(n)`](Self::Exact) — the decode allocated exactly `n` ordinals (the
///   old `Some(n)`): a real decode task's own count, or a sum of exacts.
/// - [`AtLeast(k)`](Self::AtLeast) — the exact total is unknown, but at least
///   `k` ordinals were allocated. [`AtLeast(0)`](Self::wholly_unknown) is the
///   WHOLLY-unknown value (the old `None` — a hand-built or deserialized-absent
///   record); an `AtLeast(k)` with `k > 0` carries a genuine lower bound (a
///   dropped-chunk VAD run's surviving span, or an overflowed sum's saturated
///   floor).
///
/// **Not** a reproducibility fact — an in-process merge coordinate — so it plays
/// no part in [`TaskFacts::is_reproducible_under`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpanKnowledge {
  /// The decode allocated **exactly** this many segment-id ordinals — the old
  /// `Some(n)`. A real decode task's own count (dropped segments included, since
  /// the count is captured before the drop) and the sum of two exact spans.
  Exact(usize),
  /// The exact total is unknown, but **at least** this many ordinals were
  /// allocated. [`AtLeast(0)`](Self::wholly_unknown) is the wholly-unknown value
  /// (the old `None`); a positive bound is a genuine floor — the KNOWN survivors'
  /// sum of a VAD run whose errored chunk was dropped, or the saturated floor of
  /// a sum that overflowed `usize`.
  AtLeast(usize),
}

impl SpanKnowledge {
  /// The **wholly-unknown** span, [`AtLeast(0)`](Self::AtLeast) — the old `None`:
  /// no lower bound is known. The value a hand-built or deserialized-absent
  /// record carries (the [`TaskFacts::unknown`] span and the serde default), and
  /// the serde-skipped one (see [`Self::is_wholly_unknown`]).
  #[inline]
  pub const fn wholly_unknown() -> Self {
    Self::AtLeast(0)
  }

  /// The known **lower bound** on the ordinal count — `n` for both `Exact(n)` and
  /// `AtLeast(n)`. The single value [the merge's id-base advance](crate::result::merge_transcription_results_with_options)
  /// derives from (floored at the survivors' own extent at read time), so every
  /// grouping advances by the same stored number and staged and one-shot merges
  /// number identically.
  #[inline]
  pub const fn lower_bound(&self) -> usize {
    match *self {
      Self::Exact(n) | Self::AtLeast(n) => n,
    }
  }

  /// Whether the **exact** total is known (`Exact`), as opposed to a mere lower
  /// bound (`AtLeast`).
  #[inline]
  pub const fn is_exact(&self) -> bool {
    matches!(self, Self::Exact(_))
  }

  /// Whether this is the wholly-unknown [`AtLeast(0)`](Self::wholly_unknown) — no
  /// lower bound at all. The serde `skip_serializing_if` predicate: the transient
  /// span is omitted from the wire form when it carries no information, and a
  /// missing key reads back as this value. `Exact(0)` (a KNOWN-empty span, e.g. a
  /// zero-chunk VAD run) is distinct and is NOT skipped.
  #[inline]
  pub const fn is_wholly_unknown(&self) -> bool {
    matches!(self, Self::AtLeast(0))
  }

  /// Merges two spans — the id-ordinal counts of two transcripts being joined —
  /// into the span of the join. **Associative and commutative by construction**,
  /// with [`Exact(0)`](Self::Exact) the identity, which is what keeps
  /// [`TaskFacts::merge`] associative over the span and a staged re-merge
  /// numbering identically to a one-shot one (coremlit issue #14, codex round
  /// 12):
  ///
  /// - **`Exact + Exact`** sums with a **checked** add — the join allocated
  ///   exactly the two counts' sum. A sum that overflows `usize` has no exact
  ///   answer, so it degrades to [`AtLeast(usize::MAX)`](Self::AtLeast) (the
  ///   saturated lower bound), never a wrapped or a fabricated exact count.
  /// - **any `AtLeast` participation** yields [`AtLeast`](Self::AtLeast) of the
  ///   **saturating** sum of the two lower bounds: joining an exactly-known count
  ///   onto an at-least-known one — or two at-least-known ones — can only
  ///   lower-bound the total, at the sum of what each side is known to have
  ///   allocated. Saturating, not checked: a lower bound that overflows `usize`
  ///   is still a lower bound (`usize::MAX`), not a loss of the bound.
  ///
  /// The closed form both properties fall out of: the result is `Exact(T)` when
  /// every merged leaf is `Exact` and their true total `T` fits `usize`, and
  /// `AtLeast(min(T, usize::MAX))` otherwise — associative because `T` and
  /// "every leaf `Exact`" are both grouping-independent.
  #[must_use]
  pub const fn merge(self, other: Self) -> Self {
    match (self, other) {
      (Self::Exact(a), Self::Exact(b)) => match a.checked_add(b) {
        Some(sum) => Self::Exact(sum),
        // No exact `usize` sum: degrade to the saturated lower bound, never a
        // wrapped or fabricated exact count a staged re-merge would trust.
        None => Self::AtLeast(usize::MAX),
      },
      // Any `AtLeast` makes the total a lower bound: the saturating sum of the
      // two known floors.
      (Self::Exact(a), Self::AtLeast(b))
      | (Self::AtLeast(a), Self::Exact(b))
      | (Self::AtLeast(a), Self::AtLeast(b)) => Self::AtLeast(a.saturating_add(b)),
    }
  }
}

/// A carried record of the decode-time facts a transcription **run controls**,
/// as opposed to the ones its [`DecodingOptions`](crate::options::DecodingOptions)
/// configure: whether it drew from the token sampler, the language it genuinely
/// observed, whether a progress callback truncated it, the worker coordinates
/// its RNG streams rode, and the segment-id span its decode allocated.
///
/// Lives on every [`TranscriptionResult`](crate::result::TranscriptionResult)
/// and is embedded verbatim in [`Provenance`](crate::provenance::Provenance).
/// Build the identity/hand-built value with [`Self::unknown`] and layer facts on
/// with the `with_*` builders; merge two with [`Self::merge`].
///
/// ```
/// use whisperkit::task_facts::{SpanKnowledge, TaskFacts};
///
/// // A single run at worker 2 that observed Spanish, drew from the sampler,
/// // was not truncated, and allocated exactly 3 segment ordinals.
/// let facts = TaskFacts::unknown()
///   .with_drew_from_rng(true)
///   .with_observed_language(Some("es".to_string()))
///   .with_worker(2)
///   .with_decoded_span(SpanKnowledge::Exact(3));
/// assert_eq!(facts.decoded_span(), SpanKnowledge::Exact(3));
/// assert_eq!(facts.drew_from_rng(), Some(true));
/// assert_eq!(facts.observed_language(), Some("es"));
/// assert_eq!(facts.worker_schedule(), Some([2].as_slice()));
/// // A hand-built record knows no worker coordinate and cannot observe the draw,
/// // a truncation, or a swallowed error — explicit unknown throughout, never a
/// // fabricated 0/false.
/// assert_eq!(TaskFacts::unknown().worker_schedule(), None);
/// assert_eq!(TaskFacts::unknown().drew_from_rng(), None);
/// assert_eq!(TaskFacts::unknown().early_stopped(), None);
/// assert_eq!(TaskFacts::unknown().had_swallowed_error(), None);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TaskFacts {
  /// Whether the decode ever **drew from the token sampler** — any window
  /// accepted (or rejected) at a **non-zero** temperature, from the sampler's
  /// own [`GreedyTokenSampler::drew_from_rng`](crate::decode::sampler::GreedyTokenSampler::drew_from_rng),
  /// OR-ed across every attempt including rejected ones and captured before any
  /// error could propagate. Backs [`Self::is_reproducible_under`]; a *carried*
  /// fact, never re-derived from the surviving segments' temperatures (which a
  /// filter can empty). See [`crate::provenance::Provenance::is_reproducible`].
  ///
  /// **`Option<bool>`, with `None` the explicit unknown** (F1, codex round 6
  /// post-consolidation): `Some(true)`/`Some(false)` is a decode that OBSERVED
  /// whether it drew, `None` a record that cannot know (an options-only
  /// [`Provenance::from_options`](crate::provenance::Provenance::from_options),
  /// or the [`Self::unknown`] merge identity). A `false` and an unknown are NOT
  /// the same fact: [`Self::is_reproducible_under`] trusts an observed
  /// `Some(false)` (deterministic) but treats `None` conservatively.
  ///
  /// Required on deserialize (present, nullable, via [`required_option`]): a
  /// dropped flag would read back `None` and — were `None` optimistic — hand a
  /// byte-reproducibility guarantee to a run that never earned it; keeping it
  /// present and conservative closes the one direction this must never silently
  /// fail in.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  drew_from_rng: Option<bool>,
  /// The language a window **actually observed** (a probe ran or a `<|lang|>`
  /// token was predicted), or `None` when the run observed none. The
  /// **outcome**, distinct from the configured
  /// [`DecodingOptions::language`](crate::options::DecodingOptions::language) and
  /// from the Swift-compat display fallback
  /// [`TranscriptionResult::language`](crate::result::TranscriptionResult::language)
  /// carries — this is never that `"en"` display string, and never inferred.
  ///
  /// Required on deserialize (present, nullable): `None` is itself the fact
  /// ("this run witnessed no language"), so a reader must be able to tell it
  /// from a dropped key — hence the [`required_option`] bridge.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  observed_language: Option<String>,
  /// Whether a progress callback **truncated** the decode with an early stop
  /// (any attempt of any window ended on a `Some(false)` callback rather than
  /// an ordinary EOT) — a caller CONTROL action, OR-ed across every attempt
  /// including a **rejected** one whose truncation changed which attempt the
  /// fallback ladder selected (R6-F1), and captured before any error could
  /// propagate. An observed `Some(false)` forces nothing; an observed
  /// `Some(true)` independently forces [`Self::is_reproducible_under`] false: a
  /// closure has no readable identity, so a re-run from the recorded
  /// options+seed alone cannot reproduce the truncation.
  ///
  /// **`Option<bool>`, with `None` the explicit unknown** (F1, codex round 6
  /// post-consolidation): a constructor that cannot see the truncation —
  /// [`Provenance::for_segment`](crate::provenance::Provenance::for_segment) and
  /// [`Provenance::from_options`](crate::provenance::Provenance::from_options),
  /// which are handed options and at most one segment, never the callback —
  /// records `None`, and [`Self::is_reproducible_under`] then refuses to promise
  /// reproducibility rather than fabricating a `not-truncated`. That fabrication
  /// was the bug: a genuinely callback-truncated segment recorded through
  /// `for_segment` used to read back reproducible.
  ///
  /// Required on deserialize (present, nullable, via [`required_option`]), like
  /// [`Self::drew_from_rng`]: a dropped flag must not silently become the
  /// optimistic answer.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  early_stopped: Option<bool>,
  /// Whether the run **silently swallowed a child error** whose hidden outcome
  /// controlled the returned transcript — a VAD chunk that errored and was
  /// dropped, or an automatic-language probe that failed and was ignored — OR-ed
  /// across every such site and captured before the surviving result is
  /// assembled. An observed `Some(true)` independently forces
  /// [`Self::is_reproducible_under`] false: the error was hidden, so a re-run of
  /// the same audio and options need not hit it again (the mock's own
  /// transient-failure semantics make this concrete — a second identical call can
  /// return DIFFERENT text), and a transcript that depended on the drop cannot
  /// promise byte-reproducibility (codex round 11, M2).
  ///
  /// **`Option<bool>`, with `None` the explicit unknown**, exactly like
  /// [`Self::drew_from_rng`]/[`Self::early_stopped`]: `Some(false)` is a run that
  /// POSITIVELY watched its child fallible steps and saw none swallowed (the
  /// [`Self::observed_clean`] seed a real run starts from), `Some(true)` a run
  /// that hid at least one, and `None` a record that cannot know — a hand-built or
  /// options-/segment-only [`Provenance`](crate::provenance::Provenance) that
  /// never watched a child step. [`Self::is_reproducible_under`] treats that `None`
  /// like the other booleans' — conservatively non-reproducible — rather than
  /// fabricating an optimistic "nothing was swallowed".
  ///
  /// Required on deserialize (present, nullable, via [`required_option`]), like
  /// the other two reproducibility booleans: a dropped flag must not silently
  /// become the optimistic answer and hand back a byte-reproducibility the run
  /// never earned.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  had_swallowed_error: Option<bool>,
  /// The ordered worker/chunk coordinates whose RNG streams produced this
  /// transcript's segments — a single decode task's own
  /// [`window_id_offset`](crate::transcribe::TranscribeTask::set_window_id_offset)
  /// as `[offset]`, a merge's the ordered concatenation of its children's
  /// (coordinates `[0]` and `[2]` merge to `[0, 2]`). Each coordinate
  /// domain-separates the seeded fallback ladder's sub-seed derivation
  /// ([`crate::decode::sampler::derive_attempt_seed`]), so under a
  /// [`DecodingOptions::seed`](crate::options::DecodingOptions::seed) two runs at
  /// different coordinates draw different streams and land different text.
  ///
  /// **Explicit unknown (`None`) for a hand-built or options-only record, never
  /// a fabricated `[0]`** (R6-F2): a value nobody observed must not masquerade
  /// as "worker zero". A known-empty `Some([])` — a run that observed zero
  /// workers, e.g. a zero-chunk VAD run — is DISTINCT from that unknown, and
  /// under [the merge law](Self::merge) `None` is ABSORBING while `Some([])` is
  /// the identity (round 10, F2): an unknown contributor taints the ordered
  /// aggregate to unknown, where a known-empty one leaves it unchanged. Required
  /// on deserialize (present, nullable, via [`required_option`]) so a dropped key
  /// is rejected rather than read back as unknown — and so removing a known
  /// coordinate fails or yields explicit unknown, never zero.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  worker_schedule: Option<Vec<usize>>,
  /// What this transcript's decode knows about the number of segment-id ordinals
  /// it **allocated** — a [`SpanKnowledge`], carried separately from the surviving
  /// segments because a filter (the blank-audio drop, the word-timestamp
  /// zero-length filter) removes some *after* their ids are allocated, so the
  /// survivors' count under-reports the span.
  /// [`merge_transcription_results_with_options`](crate::result::merge_transcription_results_with_options)
  /// advances its running id base by this span's [lower bound](SpanKnowledge::lower_bound)
  /// (floored at the survivors' extent, not inferred from it) so a wholly-dropped
  /// chunk still shifts the next chunk's ids past the ordinals it consumed, and
  /// **stores the merged aggregate on the merged result** so a staged re-merge
  /// renumbers identically to a one-shot merge (R6-F3).
  ///
  /// Two states rather than the pre-round-12 `Option<usize>` (codex round 12):
  /// [`SpanKnowledge::Exact`] is a fully-known count, [`SpanKnowledge::AtLeast`]
  /// a known lower bound on an unknown total —
  /// [`AtLeast(0)`](SpanKnowledge::wholly_unknown) the wholly-unknown value a
  /// hand-built or deserialized result carries. Because [the merge](Self::merge)
  /// SUMS lower bounds rather than absorbing an unknown to a bound-less `None`, a
  /// dropped chunk's unknown contribution no longer erases its surviving
  /// siblings' KNOWN ordinals, and every grouping stores the same value — the
  /// associativity the old absorbing-`None` could not give a staged re-merge. **Not**
  /// a reproducibility fact — an in-process merge coordinate — so it defaults to
  /// the wholly-unknown value on a missing key rather than being required, and is
  /// omitted from the wire form when it carries no lower bound.
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "SpanKnowledge::wholly_unknown",
      skip_serializing_if = "SpanKnowledge::is_wholly_unknown"
    )
  )]
  decoded_span: SpanKnowledge,
}

/// Names every field of [`TaskFacts`] exactly once, generating (for tests) both
/// a `TASK_FACTS_FIELD_NAMES` roster and a compile-time exhaustiveness guard
/// that destructures `TaskFacts` WITHOUT `..`.
///
/// This is what keeps the record's completeness gates honest (coremlit issue
/// #14, codex round 6): add a field to the struct and this guard fails to
/// compile until it is named here, and the new name then lands in
/// `provenance::tests`' task-fact coverage as uncovered until it is exercised by
/// a mutation. So a new run-controlled fact cannot be added to the record and
/// left unread by [`Provenance::for_result`](crate::provenance::Provenance::for_result).
macro_rules! task_facts_field_names {
  ($($field:ident),+ $(,)?) => {
    /// The full field set of [`TaskFacts`], one entry per field. Kept
    /// exhaustive at compile time by the guard in the same macro expansion.
    /// Consumed by the provenance task-fact coverage test.
    #[cfg(test)]
    #[allow(dead_code)] // used only by the provenance coverage test
    pub(crate) const TASK_FACTS_FIELD_NAMES: &[&str] = &[$(stringify!($field)),+];

    /// A pure compile-time exhaustiveness check: destructuring without `..`
    /// forces every `TaskFacts` field to be named in the list above. Never
    /// called.
    #[cfg(test)]
    #[allow(dead_code)]
    fn _task_facts_field_exhaustiveness_guard(facts: TaskFacts) {
      let TaskFacts { $($field: _),+ } = facts;
    }
  };
}

task_facts_field_names!(
  drew_from_rng,
  observed_language,
  early_stopped,
  had_swallowed_error,
  worker_schedule,
  decoded_span,
);

impl TaskFacts {
  /// The all-unknown record: an **unknown** draw, truncation, and swallowed-error
  /// (`None`, never the optimistic `Some(false)`), no observed language, an
  /// **unknown** worker schedule (`None`, never `[0]`), and a wholly-unknown span
  /// ([`AtLeast(0)`](SpanKnowledge::wholly_unknown)). The value a
  /// hand-built [`TranscriptionResult`](crate::result::TranscriptionResult)
  /// carries, an options-only [`Provenance`](crate::provenance::Provenance)
  /// records, and — for zero contributors — what a
  /// `TaskFactsAccumulator` folds down to. **Not** the identity of
  /// [`Self::merge`]: under the Kleene OR (`kleene_or`) the boolean identity is
  /// `Some(false)`, so folding `unknown()` into a `Some(false)` observation
  /// would null it; a contributor fold seeds from `TaskFactsAccumulator::Empty`
  /// instead. Layer POSITIVELY observed facts on with the `with_*` builders — a
  /// genuinely greedy, un-truncated decode is [`Self::observed_clean`], which
  /// [`Self::is_reproducible_under`] can then trust, where the bare `unknown()`
  /// cannot.
  #[inline]
  pub const fn unknown() -> Self {
    Self {
      drew_from_rng: None,
      observed_language: None,
      early_stopped: None,
      had_swallowed_error: None,
      worker_schedule: None,
      decoded_span: SpanKnowledge::wholly_unknown(),
    }
  }

  /// The **observed-clean** record a real run starts watching from: it has
  /// POSITIVELY seen no RNG draw, no early-stop truncation, and no swallowed child
  /// error yet (`Some(false)` for [`Self::drew_from_rng`], [`Self::early_stopped`],
  /// and [`Self::had_swallowed_error`]), with no observed
  /// language, an unknown worker schedule, and a wholly-unknown span. Distinct from
  /// [`Self::unknown`] — which cannot see those facts and is conservatively
  /// non-reproducible — this is the honest initial state of a decode that IS
  /// watching: a window that draws, a callback that truncates, or a swallowed
  /// child error flips the corresponding fact to `Some(true)` under the
  /// [merge law](Self::merge), and
  /// a run that decodes NO window at all keeps the `Some(false)` and earns the
  /// byte-reproducibility a zero-window run genuinely has (codex round 8, F3).
  ///
  /// It is also the correct **seed for a per-attempt fact sink**: under the
  /// Kleene OR (`kleene_or`) `Some(false)` is the OR identity, so a greedy
  /// attempt (`Some(false)`) folded onto this sink leaves it `Some(false)`,
  /// where seeding at `unknown()`'s `None` would instead null it to unknown
  /// (`None | Some(false) = None`).
  #[inline]
  pub const fn observed_clean() -> Self {
    Self::unknown()
      .with_drew_from_rng(false)
      .with_early_stopped(false)
      .with_had_swallowed_error(false)
  }

  // -- drew_from_rng ------------------------------------------------------
  /// Whether the decode ever drew from the token sampler, or `None` when
  /// unobserved. See the field's doc.
  #[inline(always)]
  pub const fn drew_from_rng(&self) -> Option<bool> {
    self.drew_from_rng
  }
  /// Builder recording [`Self::drew_from_rng`] as a POSITIVELY observed fact
  /// (`Some(drew_from_rng)`) — the shape a decode carries out of the window
  /// loop. Leave it at [`Self::unknown`]'s `None` to say the draw was
  /// unobserved.
  #[must_use]
  #[inline(always)]
  pub const fn with_drew_from_rng(mut self, drew_from_rng: bool) -> Self {
    self.drew_from_rng = Some(drew_from_rng);
    self
  }

  // -- observed_language --------------------------------------------------
  /// The language a window actually observed, or `None`. See the field's doc.
  #[inline(always)]
  pub fn observed_language(&self) -> Option<&str> {
    self.observed_language.as_deref()
  }
  /// Builder setting [`Self::observed_language`].
  #[must_use]
  #[inline(always)]
  pub fn with_observed_language(mut self, observed_language: Option<String>) -> Self {
    self.observed_language = observed_language;
    self
  }

  // -- early_stopped ------------------------------------------------------
  /// Whether a progress callback truncated the decode, or `None` when
  /// unobserved. See the field's doc.
  #[inline(always)]
  pub const fn early_stopped(&self) -> Option<bool> {
    self.early_stopped
  }
  /// Builder recording [`Self::early_stopped`] as a POSITIVELY observed fact
  /// (`Some(early_stopped)`). Leave it at [`Self::unknown`]'s `None` when the
  /// constructor cannot see the callback — the honest state for
  /// [`Provenance::for_segment`](crate::provenance::Provenance::for_segment).
  #[must_use]
  #[inline(always)]
  pub const fn with_early_stopped(mut self, early_stopped: bool) -> Self {
    self.early_stopped = Some(early_stopped);
    self
  }

  // -- had_swallowed_error ------------------------------------------------
  /// Whether the run silently swallowed a transcript-controlling child error, or
  /// `None` when unobserved. See the field's doc.
  #[inline(always)]
  pub const fn had_swallowed_error(&self) -> Option<bool> {
    self.had_swallowed_error
  }
  /// Builder recording [`Self::had_swallowed_error`] as a POSITIVELY observed
  /// fact (`Some(had_swallowed_error)`) — `Some(false)` the observed-clean seed a
  /// watching run starts from, `Some(true)` set at a swallow site. Leave it at
  /// [`Self::unknown`]'s `None` when the constructor cannot see the child steps.
  #[must_use]
  #[inline(always)]
  pub const fn with_had_swallowed_error(mut self, had_swallowed_error: bool) -> Self {
    self.had_swallowed_error = Some(had_swallowed_error);
    self
  }

  // -- worker_schedule ----------------------------------------------------
  /// The ordered worker/chunk coordinates, or `None` when unknown. See the
  /// field's doc.
  #[inline(always)]
  pub fn worker_schedule(&self) -> Option<&[usize]> {
    self.worker_schedule.as_deref()
  }
  /// Builder assigning [`Self::worker_schedule`] directly.
  #[must_use]
  #[inline(always)]
  pub fn with_worker_schedule(mut self, worker_schedule: Option<Vec<usize>>) -> Self {
    self.worker_schedule = worker_schedule;
    self
  }
  /// Builder setting [`Self::worker_schedule`] to a **single** known worker
  /// coordinate `[worker]` — the shape a single decode task carries.
  #[must_use]
  #[inline(always)]
  pub fn with_worker(mut self, worker: usize) -> Self {
    self.worker_schedule = Some(vec![worker]);
    self
  }

  // -- decoded_span -------------------------------------------------------
  /// What the decode knows about the number of segment-id ordinals it allocated —
  /// [`SpanKnowledge::wholly_unknown`] when untracked. See the field's doc.
  #[inline(always)]
  pub const fn decoded_span(&self) -> SpanKnowledge {
    self.decoded_span
  }
  /// Builder assigning [`Self::decoded_span`] directly.
  #[must_use]
  #[inline(always)]
  pub const fn with_decoded_span(mut self, decoded_span: SpanKnowledge) -> Self {
    self.decoded_span = decoded_span;
    self
  }

  /// Folds `other`'s facts into `self` under the **one** merge law every merge
  /// entry point calls (coremlit issue #14, codex round 6). Associative — a
  /// one-shot merge over a slice and a staged left-fold produce byte-identical
  /// records, which is what makes a VAD result safe to re-merge at streaming
  /// finalize (R6-F3) — but NOT with [`Self::unknown`] as its identity: under
  /// the Kleene OR (`kleene_or`) the boolean identity is `Some(false)`, so a fold
  /// over contributors seeds from the separate `TaskFactsAccumulator` identity
  /// (codex round 8, F2). The per-field laws:
  ///
  /// - **[`drew_from_rng`](Self::drew_from_rng)** / **[`early_stopped`](Self::early_stopped)**
  ///   / **[`had_swallowed_error`](Self::had_swallowed_error)** — Kleene
  ///   three-valued OR (`kleene_or`): `Some(true)` absorbs, two `Some(false)` stay
  ///   `Some(false)`, and an unknown (`None`) mixed with anything-but-true stays
  ///   unknown (`None | Some(false) = None`). The merge is `Some(true)` iff some
  ///   child observed `true`, `Some(false)` iff EVERY child observed the fact and
  ///   all saw `false`, and `None` as soon as any child could not observe it
  ///   (unless another observed `true`). A child that cannot see the fact
  ///   therefore no longer lets a `Some(false)` beside it pass for an observed
  ///   clean. `had_swallowed_error` rides this same law so a VAD run's swallowed
  ///   chunk-drop, merged in beside the surviving chunks' clean `Some(false)`,
  ///   OR-s the aggregate to `Some(true)` (codex round 11, M2).
  /// - **[`observed_language`](Self::observed_language)** — first genuine
  ///   observation wins: `self`'s is kept when present, else `other`'s is
  ///   adopted. A scalar cannot hold two conflicting observations, and a
  ///   left-fold over the children in order makes "first" well defined.
  /// - **[`worker_schedule`](Self::worker_schedule)** — two KNOWN schedules
  ///   **concatenate in order**, so a merge of coordinates `[0]` and `[2]`
  ///   records `[0, 2]`, distinct from `[0, 1]` (R6-F2); `Some([])` (known-empty)
  ///   is the identity. An unknown (`None`) child is ABSORBING (round 10, F2): it
  ///   taints the aggregate to `None` rather than passing the known side through
  ///   as the whole schedule — a coordinate nobody could report must not read
  ///   back as a fully-known ordering. Absorbing-`None` over a free monoid is
  ///   associative, the same lattice the Kleene booleans use.
  /// - **[`decoded_span`](Self::decoded_span)** — the two spans **sum under
  ///   [`SpanKnowledge::merge`]**, so the merged result stores the aggregate
  ///   ordinal count its children allocated (R6-F3). Two exacts checked-add (an
  ///   overflow degrades to a saturated `AtLeast(usize::MAX)`, never a wrapped or
  ///   fabricated exact count); once any child is a mere `AtLeast`, the total is
  ///   a saturating lower bound. Unlike the pre-round-12 absorbing-`None` (codex
  ///   round 12), an unknown-but-with-a-known-sibling total is `AtLeast(sibling's
  ///   count)`, NOT a bound-less unknown — a dropped chunk's unknown contribution
  ///   no longer erases its siblings' KNOWN ordinals, which is what let a staged
  ///   re-merge renumber a trailing chunk differently than a one-shot one. The
  ///   sum is associative by construction (`Exact(0)` its identity), the same
  ///   grouping-independence the worker schedule and Kleene booleans have.
  pub fn merge(&mut self, other: &Self) {
    self.drew_from_rng = kleene_or(self.drew_from_rng, other.drew_from_rng);
    self.early_stopped = kleene_or(self.early_stopped, other.early_stopped);
    self.had_swallowed_error = kleene_or(self.had_swallowed_error, other.had_swallowed_error);
    if self.observed_language.is_none() {
      self.observed_language = other.observed_language.clone();
    }
    self.worker_schedule = match (
      self.worker_schedule.take(),
      other.worker_schedule.as_deref(),
    ) {
      // Both KNOWN: ordered concatenation. `Some([])` is the identity — a known
      // run of zero workers contributes no coordinates but does not taint the
      // aggregate.
      (Some(mut schedule), Some(more)) => {
        schedule.extend_from_slice(more);
        Some(schedule)
      }
      // An unknown (`None`) contributor is ABSORBING (round 10, F2): a child that
      // cannot report its ordered coordinates taints the aggregate to unknown,
      // rather than letting the known side pass for the WHOLE schedule. The
      // pre-round-10 law treated `None` as the identity, so `None + Some([7])`
      // read back `Some([7])` — partial knowledge presented as fully known.
      (None, _) | (_, None) => None,
    };
    // Sum the two spans under [`SpanKnowledge::merge`] — checked-add over exacts,
    // saturating lower bounds once any part is a mere `AtLeast`, associative by
    // construction (codex round 12). This replaces the pre-round-12 absorbing-`None`
    // fold, which threw away a dropped chunk's siblings' KNOWN ordinals and left a
    // staged re-merge unable to recover them.
    self.decoded_span = self.decoded_span.merge(other.decoded_span);
  }

  /// Whether a transcript carrying these facts can be reproduced byte-for-byte
  /// by re-running the same audio through the same options — `true` only when
  /// the decode POSITIVELY observed that it never drew from the sampler (or
  /// `seeded` makes the draws it did make replayable) AND POSITIVELY observed
  /// that no progress callback truncated it AND POSITIVELY observed that it
  /// swallowed no transcript-controlling child error (codex round 11, M2).
  ///
  /// **Conservative on the explicit unknown** (F1, codex round 6
  /// post-consolidation): an unobserved draw, truncation, or swallowed error
  /// (`None`) answers `false`, never the optimistic value. A record that cannot
  /// see whether a transcript-controlling event happened — an options-only
  /// [`Provenance::from_options`](crate::provenance::Provenance::from_options),
  /// or a segment-only [`Provenance::for_segment`](crate::provenance::Provenance::for_segment)
  /// that is handed the truncated segment but never the callback — must not
  /// promise byte-reproducibility. Only a real decode's carried `Some(false)`
  /// facts, read by [`Provenance::for_result`](crate::provenance::Provenance::for_result)
  /// off the transcript, earns the optimistic answer. The bare [`Self::unknown`]
  /// is therefore NOT reproducible; a genuinely clean run is
  /// [`Self::observed_clean`] (all three booleans positively `Some(false)`).
  ///
  /// The reproducibility predicate [`Provenance::is_reproducible`](crate::provenance::Provenance::is_reproducible)
  /// is built on: it reads [`Self::drew_from_rng`], [`Self::early_stopped`], and
  /// [`Self::had_swallowed_error`] here and supplies whether
  /// [`DecodingOptions::seed`](crate::options::DecodingOptions::seed)
  /// is set as `seeded`. Kept on the record so the facts it rests on have
  /// one home; see the [`Provenance`](crate::provenance::Provenance) method for
  /// the full rationale.
  #[inline]
  pub const fn is_reproducible_under(&self, seeded: bool) -> bool {
    let not_truncated = matches!(self.early_stopped, Some(false));
    // A swallowed child error (`Some(true)`) forces false — its hidden outcome
    // controlled the transcript, unreproducibly — and an UNOBSERVED swallow
    // (`None`) is treated exactly like an unobserved truncation: conservatively
    // non-reproducible, never the optimistic answer (codex round 11, M2).
    let no_swallowed_error = matches!(self.had_swallowed_error, Some(false));
    let draw_replayable = match self.drew_from_rng {
      Some(false) => true,
      Some(true) => seeded,
      None => false,
    };
    not_truncated && no_swallowed_error && draw_replayable
  }
}

/// The **fold identity** for a merge over CONTRIBUTORS, kept distinct from
/// [`TaskFacts::unknown`] so an unknown contributor can never masquerade as the
/// neutral element (codex round 8, F2). Under the Kleene OR (`kleene_or`) the
/// boolean identity is `Some(false)`, not `None`, so `unknown()` is no longer a
/// left identity of [`TaskFacts::merge`] (`unknown().merge(Some(false))` nulls
/// to `None`). [`Empty`](Self::Empty) is the honest "no contributor merged yet"
/// element: the first contributor is taken **verbatim**, each subsequent one
/// folds in under the merge law, and [`into_facts`](Self::into_facts) of an
/// `Empty` accumulator is `unknown()` — the correct "nothing was observed"
/// answer for zero contributors.
///
/// Consumed by [`merge_transcription_results_with_options`](crate::result::merge_transcription_results_with_options)'s
/// task-facts fold and by [`LocalAgreement`](crate::stream::agreement::LocalAgreement)'s
/// finalize sink (codex round 8, F1).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TaskFactsAccumulator {
  /// No contributor has been merged yet — the true fold identity.
  Empty,
  /// The running merge of every contributor folded in so far.
  Merged(TaskFacts),
}

impl TaskFactsAccumulator {
  /// A fresh [`Empty`](Self::Empty) accumulator.
  #[inline]
  pub(crate) const fn new() -> Self {
    Self::Empty
  }

  /// Folds one `contributor` in: the first becomes the seed verbatim (so no
  /// `unknown()` identity ever nulls a known `Some(false)`), each subsequent one
  /// merges under [`TaskFacts::merge`].
  #[inline]
  pub(crate) fn merge(&mut self, contributor: &TaskFacts) {
    match self {
      Self::Empty => *self = Self::Merged(contributor.clone()),
      Self::Merged(facts) => facts.merge(contributor),
    }
  }

  /// The accumulated facts, or [`TaskFacts::unknown`] when no contributor was
  /// ever merged.
  #[inline]
  pub(crate) fn into_facts(self) -> TaskFacts {
    match self {
      Self::Empty => TaskFacts::unknown(),
      Self::Merged(facts) => facts,
    }
  }
}
