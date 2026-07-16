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
//! [`TaskFacts::worker_schedule`] and [`TaskFacts::observed_language`] each carry
//! an **explicit-unknown** state (`None`) distinct from any value. A record built
//! from options with no decode to speak for
//! ([`Provenance::from_options`](crate::provenance::Provenance::from_options)),
//! or a transcript assembled by hand, knows no worker coordinate and witnessed
//! no language — and says exactly that, rather than the worker `0` /
//! `""`-language a default would forge. The merge law treats that unknown as an
//! identity (it contributes nothing), which is what keeps the law associative.

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
/// use whisperkit::task_facts::TaskFacts;
///
/// // A single run at worker 2 that observed Spanish, drew from the sampler,
/// // was not truncated, and allocated 3 segment ordinals.
/// let facts = TaskFacts::unknown()
///   .with_drew_from_rng(true)
///   .with_observed_language(Some("es".to_string()))
///   .with_worker(2)
///   .with_decoded_span(Some(3));
/// assert!(facts.drew_from_rng());
/// assert_eq!(facts.observed_language(), Some("es"));
/// assert_eq!(facts.worker_schedule(), Some([2].as_slice()));
/// // A hand-built record knows no worker coordinate — explicit unknown, never 0.
/// assert_eq!(TaskFacts::unknown().worker_schedule(), None);
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
  /// Required on deserialize: a dropped flag would read back `false` ("never
  /// sampled"), the optimistic answer that hands a byte-reproducibility
  /// guarantee to a run that never earned it — the one direction this must
  /// never silently fail in.
  drew_from_rng: bool,
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
  /// propagate. Independently forces [`Self::is_reproducible_under`] false: a
  /// closure has no readable identity, so a re-run from the recorded
  /// options+seed alone cannot reproduce the truncation.
  ///
  /// Required on deserialize, like [`Self::drew_from_rng`]: a dropped flag
  /// reads back `false` ("not truncated"), the optimistic answer.
  early_stopped: bool,
  /// The ordered worker/chunk coordinates whose RNG streams produced this
  /// transcript's segments — a single decode task's own
  /// [`window_id_offset`](crate::transcribe::TranscribeTask::set_window_id_offset)
  /// as `[offset]`, a merge's the concatenation of its children's in order
  /// (`[0, 2]` for a VAD merge that dropped the middle chunk). Each coordinate
  /// domain-separates the seeded fallback ladder's sub-seed derivation
  /// ([`crate::decode::sampler::derive_attempt_seed`]), so under a
  /// [`DecodingOptions::seed`](crate::options::DecodingOptions::seed) two runs at
  /// different coordinates draw different streams and land different text.
  ///
  /// **Explicit unknown (`None`) for a hand-built or options-only record, never
  /// a fabricated `[0]`** (R6-F2): a value nobody observed must not masquerade
  /// as "worker zero". Required on deserialize (present, nullable, via
  /// [`required_option`]) so a dropped key is rejected rather than read back as
  /// unknown — and so removing a known coordinate fails or yields explicit
  /// unknown, never zero.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  worker_schedule: Option<Vec<usize>>,
  /// The number of segment-id ordinals this transcript's decode **allocated** —
  /// carried separately from the surviving segments because a filter (the
  /// blank-audio drop, the word-timestamp zero-length filter) removes some
  /// *after* their ids are allocated, so the survivors' count under-reports the
  /// span. [`merge_transcription_results_with_options`](crate::result::merge_transcription_results_with_options)
  /// advances its running id base by this (not the survivors' extent) so a
  /// wholly-dropped chunk still shifts the next chunk's ids past the ordinals it
  /// consumed, and — the R6-F3 fix — **stores the summed aggregate on the merged
  /// result** so a staged re-merge renumbers identically to a one-shot merge.
  ///
  /// `None` when untracked (a hand-built or deserialized result); the merge then
  /// falls back to the survivors' own extent. **Not** a reproducibility fact —
  /// an in-process merge coordinate — so it defaults to `None` on a missing key
  /// rather than being required, and is omitted from the wire form when absent.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  decoded_span: Option<usize>,
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
  worker_schedule,
  decoded_span,
);

impl TaskFacts {
  /// The all-unknown record: no draw, no observed language, not truncated, an
  /// **unknown** worker schedule (`None`, never `[0]`) and an untracked span
  /// (`None`). The value a hand-built [`TranscriptionResult`](crate::result::TranscriptionResult)
  /// carries, an options-only [`Provenance`](crate::provenance::Provenance)
  /// records, and the **identity** of [`Self::merge`].
  #[inline]
  pub const fn unknown() -> Self {
    Self {
      drew_from_rng: false,
      observed_language: None,
      early_stopped: false,
      worker_schedule: None,
      decoded_span: None,
    }
  }

  // -- drew_from_rng ------------------------------------------------------
  /// Whether the decode ever drew from the token sampler. See the field's doc.
  #[inline(always)]
  pub const fn drew_from_rng(&self) -> bool {
    self.drew_from_rng
  }
  /// Builder setting [`Self::drew_from_rng`].
  #[must_use]
  #[inline(always)]
  pub const fn with_drew_from_rng(mut self, drew_from_rng: bool) -> Self {
    self.drew_from_rng = drew_from_rng;
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
  /// Whether a progress callback truncated the decode. See the field's doc.
  #[inline(always)]
  pub const fn early_stopped(&self) -> bool {
    self.early_stopped
  }
  /// Builder setting [`Self::early_stopped`].
  #[must_use]
  #[inline(always)]
  pub const fn with_early_stopped(mut self, early_stopped: bool) -> Self {
    self.early_stopped = early_stopped;
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
  /// The number of segment-id ordinals the decode allocated, or `None` when
  /// untracked. See the field's doc.
  #[inline(always)]
  pub const fn decoded_span(&self) -> Option<usize> {
    self.decoded_span
  }
  /// Builder assigning [`Self::decoded_span`] directly.
  #[must_use]
  #[inline(always)]
  pub const fn with_decoded_span(mut self, decoded_span: Option<usize>) -> Self {
    self.decoded_span = decoded_span;
    self
  }

  /// Folds `other`'s facts into `self` under the **one** merge law every merge
  /// entry point calls (coremlit issue #14, codex round 6). Associative with
  /// [`Self::unknown`] as its identity, so a one-shot merge over a slice and a
  /// staged left-fold produce byte-identical records — which is what makes a
  /// VAD result safe to re-merge at streaming finalize (R6-F3):
  ///
  /// - **[`drew_from_rng`](Self::drew_from_rng)** / **[`early_stopped`](Self::early_stopped)**
  ///   — logical OR. Either fact is true of the merge if it was true of any
  ///   child.
  /// - **[`observed_language`](Self::observed_language)** — first genuine
  ///   observation wins: `self`'s is kept when present, else `other`'s is
  ///   adopted. A scalar cannot hold two conflicting observations, and a
  ///   left-fold over the children in order makes "first" well defined.
  /// - **[`worker_schedule`](Self::worker_schedule)** — **concatenated in
  ///   order**, so a merge of coordinates `[0]` and `[2]` records `[0, 2]`,
  ///   distinct from `[0, 1]` (R6-F2). An unknown (`None`) child is the
  ///   identity: it contributes no coordinates rather than poisoning the
  ///   schedule, which is what keeps the concatenation associative.
  /// - **[`decoded_span`](Self::decoded_span)** — summed, so the merged result
  ///   stores the aggregate ordinal count its children allocated (R6-F3). A
  ///   `None` child contributes nothing; the sum is `None` only when every
  ///   child is untracked.
  pub fn merge(&mut self, other: &Self) {
    self.drew_from_rng |= other.drew_from_rng;
    self.early_stopped |= other.early_stopped;
    if self.observed_language.is_none() {
      self.observed_language = other.observed_language.clone();
    }
    match (
      self.worker_schedule.as_mut(),
      other.worker_schedule.as_deref(),
    ) {
      (Some(schedule), Some(more)) => schedule.extend_from_slice(more),
      (None, Some(more)) => self.worker_schedule = Some(more.to_vec()),
      // `other` unknown (or both): keep `self`'s schedule as-is (identity).
      (_, None) => {}
    }
    self.decoded_span = match (self.decoded_span, other.decoded_span) {
      (Some(a), Some(b)) => Some(a.saturating_add(b)),
      (some @ Some(_), None) | (None, some @ Some(_)) => some,
      (None, None) => None,
    };
  }

  /// Whether a transcript carrying these facts can be reproduced byte-for-byte
  /// by re-running the same audio through the same options — `true` when the
  /// decode never drew from the sampler, or when `seeded` makes the draws it did
  /// make replayable, AND no progress callback truncated it.
  ///
  /// The reproducibility predicate [`Provenance::is_reproducible`](crate::provenance::Provenance::is_reproducible)
  /// is built on: it reads [`Self::drew_from_rng`] and [`Self::early_stopped`]
  /// here and supplies whether [`DecodingOptions::seed`](crate::options::DecodingOptions::seed)
  /// is set as `seeded`. Kept on the record so the two facts it rests on have
  /// one home; see the [`Provenance`](crate::provenance::Provenance) method for
  /// the full rationale.
  #[inline]
  pub const fn is_reproducible_under(&self, seeded: bool) -> bool {
    !self.early_stopped && (!self.drew_from_rng || seeded)
  }
}
