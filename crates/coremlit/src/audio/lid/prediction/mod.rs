//! The prediction vocabulary: [`LanguageScore`] (one ranked language) and the
//! min-heap top-k the identify path runs.
//!
//! Ranking follows the crate's existing `RankedScore` contract (ced's, which is
//! soundevents'): `f32::total_cmp` descending on the score, ties broken by
//! **ascending language index**. The score ranked on is the model's raw
//! natural-log probability; because `exp` is strictly monotonic, ranking in log
//! space and reporting in either space give the same order, and no
//! `NUM_LANGUAGES`-element sort is needed.

use core::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::audio::lid::{
  NUM_LANGUAGES,
  error::{Error, InvalidLogProbability, Result},
  labels::Language,
};

#[cfg(test)]
mod tests;

/// One window's log-probability row paired with the [`Span`] it was scored
/// over — `windit`'s own value type, so the per-window output composes with the
/// windit post-processing stack (smoothing, segmentation) with no adapter.
///
/// [`Span`]: crate::audio::lid::Span
pub type WindowLogProbabilities = windit::windowed::Windowed<LogProbabilities>;

/// A full row of natural-log probabilities — one window's, or a whole clip's
/// after aggregation: always exactly [`NUM_LANGUAGES`] values, indexed by model
/// column, each `<= 0` and never NaN.
///
/// # What the invariant does and does not promise
///
/// Straight off the graph the row is a log-SOFTMAX: `exp` over it sums to 1 —
/// to the graph's own fp16 accuracy, which puts it as much as 7.7e-3 away from
/// 1 on [`ComputeUnits::CpuOnly`] — on EITHER side, the deviation being signed
/// and measured in both directions on every compute unit by
/// `the_graphs_largest_output_reaches_zero_and_never_passes_it`. Aggregation
/// makes that exact: `Vote` divides shares, the other three close with a
/// renormalization, and the result is
/// checked against 1 before it is returned. Where a fold cannot produce a
/// distribution it FAILS — [`Error::ZeroMassAggregate`] on rows whose honest
/// pool assigns every language probability zero — rather than hand back a row
/// that is not one.
///
/// The invariant this TYPE enforces is only the pointwise one (`<= 0`, not
/// NaN), because that is the part a hand-built row can be held to without
/// choosing a floating-point tolerance for "sums to 1". Both doors that ADMIT a
/// row apply it from `is_log_probability`, this module's single definition of
/// it: [`Self::try_from_slice`] to a caller's row, and
/// [`Identifier::log_probabilities`] to the graph's own output, so a model that
/// meets the feature-name/shape/dtype contract and then emits a positive score
/// is refused rather than ranked into a `probability()` above 1. The one thing
/// [`aggregate_windows`] additionally requires of a row it is given is that it
/// have a FINITE maximum ([`Error::UnnormalizableWindow`]) — a bound that is
/// finite and that the whole row sits under — which is exactly the condition
/// under which the row normalizes: a row that is `-∞` in every column rules
/// every language out, which is not evidence about any of them. That door
/// re-establishes the pointwise invariant rather than assuming it, because the
/// crate-internal constructor does not check it: a row reaching the fold from
/// inside this crate holding a `+∞` or a NaN is refused there too. It does NOT
/// require the row to sit at any particular scale — a row whose largest value
/// is `-800` folds exactly as one shifted up to `0` does.
///
/// [`ComputeUnits::CpuOnly`]: crate::ComputeUnits::CpuOnly
/// [`Identifier::log_probabilities`]: crate::audio::lid::Identifier::log_probabilities
/// [`aggregate_windows`]: crate::audio::lid::aggregate_windows
/// [`Error::ZeroMassAggregate`]: crate::audio::lid::Error::ZeroMassAggregate
/// [`Error::UnnormalizableWindow`]: crate::audio::lid::Error::UnnormalizableWindow
///
/// `-∞` is a legal value: it is the exact log of a zero probability, which
/// [`ScorePooling::Vote`] produces for any language no window chose.
/// [`LanguageScore::probability`] maps it to exactly `0.0`.
///
/// [`NUM_LANGUAGES`]: crate::audio::lid::NUM_LANGUAGES
/// [`ScorePooling`]: crate::audio::lid::ScorePooling
/// [`ScorePooling::Vote`]: crate::audio::lid::ScorePooling::Vote
#[derive(Debug, Clone, PartialEq)]
pub struct LogProbabilities {
  values: Vec<f32>,
}

impl LogProbabilities {
  /// Wrap an already-validated row.
  ///
  /// # Panics
  /// If `values.len() != NUM_LANGUAGES` — an internal invariant (every producer
  /// is post-shape-check), not a caller-reachable path.
  pub(crate) fn new(values: Vec<f32>) -> Self {
    assert!(
      values.len() == NUM_LANGUAGES,
      "LogProbabilities requires exactly NUM_LANGUAGES values, got {}",
      values.len()
    );
    Self { values }
  }

  /// Build a row by hand, for a consumer's own tests and for driving
  /// [`aggregate_windows`] with no staged `.mlmodelc` and no inference.
  ///
  /// `values` is read positionally as natural-log probabilities that are
  /// ALREADY in log space, one per model column, indexed exactly as
  /// [`Self::as_slice`] returns them. Nothing is transformed: no softmax, no
  /// `ln`, no renormalization. The slice is copied rather than adopted.
  ///
  /// # Errors
  /// [`Error::LanguageCountMismatch`] if `values.len() != `[`NUM_LANGUAGES`];
  /// [`Error::InvalidLogProbability`] on a NaN or a value above zero.
  ///
  /// # Examples
  /// ```
  /// use coremlit::audio::lid::{Error, LogProbabilities, NUM_LANGUAGES};
  ///
  /// // A near-certain call on Thai (model column 94).
  /// let mut row = vec![-14.0f32; NUM_LANGUAGES];
  /// row[94] = -0.01;
  /// let scores = LogProbabilities::try_from_slice(&row)?;
  /// assert_eq!(scores.as_slice()[94], -0.01);
  /// assert_eq!(scores.top_k(1)?[0].code(), "th");
  ///
  /// // A zero probability is representable; a positive one is not.
  /// assert!(LogProbabilities::try_from_slice(&vec![f32::NEG_INFINITY; NUM_LANGUAGES]).is_ok());
  /// row[94] = 0.5;
  /// assert!(matches!(
  ///   LogProbabilities::try_from_slice(&row),
  ///   Err(Error::InvalidLogProbability(d)) if d.index() == 94
  /// ));
  /// assert!(matches!(
  ///   LogProbabilities::try_from_slice(&row[..NUM_LANGUAGES - 1]),
  ///   Err(Error::LanguageCountMismatch(got)) if got == NUM_LANGUAGES - 1
  /// ));
  /// # Ok::<(), Error>(())
  /// ```
  ///
  /// [`NUM_LANGUAGES`]: crate::audio::lid::NUM_LANGUAGES
  /// [`aggregate_windows`]: crate::audio::lid::aggregate_windows
  pub fn try_from_slice(values: &[f32]) -> Result<Self> {
    if values.len() != NUM_LANGUAGES {
      return Err(Error::LanguageCountMismatch(values.len()));
    }
    for (index, &value) in values.iter().enumerate() {
      // The ONE definition of "is a natural-log probability", shared with the
      // model door so the two cannot drift apart: `is_log_probability` carries
      // why it is written as the accepting form, why `-inf` is legal, and why
      // exactly zero is.
      if !is_log_probability(value) {
        return Err(InvalidLogProbability::new(index, value).into());
      }
    }
    Ok(Self::new(values.to_vec()))
  }

  /// The per-language natural-log probabilities, indexed by model column
  /// ([`LanguageScore::index`]).
  #[inline]
  pub fn as_slice(&self) -> &[f32] {
    &self.values
  }

  /// The top `k` languages, descending, ties broken by ascending model column —
  /// the same ranking [`Identifier::identify`] applies to a single-window row.
  /// `k == 0` yields an empty vec; `k` above the roster size saturates.
  ///
  /// # Errors
  /// [`Error::UnknownLanguageIndex`] — defensive only.
  ///
  /// [`Identifier::identify`]: crate::audio::lid::Identifier::identify
  pub fn top_k(&self, k: usize) -> Result<Vec<LanguageScore>> {
    top_k_from_scores(self.values.iter().copied().enumerate(), k)
  }
}

/// The pointwise invariant every [`LogProbabilities`] row holds, as **one
/// definition** that both doors admitting a row read: a natural-log
/// probability is at most zero.
///
/// [`LogProbabilities::try_from_slice`] holds a CALLER's row to it;
/// [`Identifier::log_probabilities`] holds the GRAPH's output to it, through
/// [`is_finite_log_probability`], which adds finiteness and nothing else. The
/// two report different errors — a caller's bad row and a corrupt model output
/// are different diagnoses — but they must not disagree about what a
/// log-probability IS, and calling one function is what makes that structural
/// rather than remembered.
///
/// **It is written as the predicate that ACCEPTS**, `value <= 0.0`, rather
/// than as the `value.is_nan() || value > 0.0` that refuses. Those two are the
/// same test only over an ORDERED domain and `f32` is not one: every ordered
/// comparison against a NaN is false, so the accepting form rejects a NaN with
/// no second clause a later edit could leave behind. `aggregate`'s
/// postcondition is written this way for the same reason.
///
/// **`-∞` is accepted.** It is the exact log of a zero probability, which
/// [`ScorePooling::Vote`] genuinely produces for a language no window chose.
///
/// **Exactly zero is accepted, and that is a measurement rather than a
/// courtesy.** A log-softmax output is non-positive by construction, so
/// refusing `>= 0` at the model door would look free — and it would refuse
/// real audio. These rows are narrowed through fp16, and this graph really does
/// emit `0.0`: 22 of the 50 076 values in `lid_long_clip`'s published sweep, all
/// of them on [`ComputeUnits::CpuOnly`], whose narrowing is the loosest of the
/// four and rounds a top language already near probability 1 the rest of the
/// way up. Nothing in that sweep sits ABOVE zero on
/// any of the four compute units — the other three peak near `-1.4e-3`.
/// `the_graphs_largest_output_reaches_zero_and_never_passes_it` gates both
/// halves, because the predicate rests on both.
///
/// [`Identifier::log_probabilities`]: crate::audio::lid::Identifier::log_probabilities
/// [`ScorePooling::Vote`]: crate::audio::lid::ScorePooling::Vote
/// [`ComputeUnits::CpuOnly`]: crate::ComputeUnits::CpuOnly
#[inline]
pub(crate) fn is_log_probability(value: f32) -> bool {
  value <= 0.0
}

/// [`is_log_probability`] **plus finiteness** — the model door's form of the
/// same rule, and the only thing that door adds.
///
/// A caller may legitimately hand in `-∞`; a log-softmax GRAPH emitting one is
/// corruption, as `+∞` and a NaN are. That extra requirement is written here as
/// an addition to the shared predicate rather than as a second copy of it, so
/// the `<= 0` half still exists in exactly one place and the difference between
/// the two doors is a single readable conjunct rather than two rules to
/// compare.
#[inline]
pub(crate) fn is_finite_log_probability(value: f32) -> bool {
  is_log_probability(value) && value.is_finite()
}

/// One ranked language: the roster row plus the model's natural-log
/// probability for it.
///
/// # Why a struct and not `(usize, f32)`
///
/// A bare tuple would be smaller to write and worse to use, on three counts
/// that all bite silently:
///
/// - **The number's meaning is not guessable.** This graph emits values that
///   are already log-softmaxed — natural log, summing to 1 under `exp`, all
///   `<= 0`. A tuple's `f32` gives a reader no way to tell that from a logit or
///   a probability, and the two plausible wrong readings (thresholding a
///   log-prob at 0.5, or `exp`-ing something that was already a probability)
///   both produce numbers rather than errors. [`Self::log_probability`] and
///   [`Self::probability`] each say what they are, and only one of them needs
///   an `exp`.
/// - **A raw index invites the wrong roster.** `.0` is meaningful only against
///   THIS door's [`languages`](crate::audio::lid::languages) table; carrying
///   the resolved [`Language`] means the code and name travel with the score
///   and cannot be looked up in something else.
/// - **A tuple's shape is frozen.** Adding a field to a struct is additive;
///   widening a tuple is a break for every destructuring caller.
///
/// It is also the shape this crate already uses for a ranked prediction
/// (`audio::ced`'s `EventPrediction`), so the two doors read alike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanguageScore {
  language: &'static Language,
  log_probability: f32,
}

impl LanguageScore {
  /// Resolve `index` to its roster row.
  ///
  /// # Errors
  /// [`Error::UnknownLanguageIndex`] if the index has no row — defensive: the
  /// roster is a `[Language; NUM_LANGUAGES]`, so this is unreachable for
  /// in-range indices.
  pub(crate) fn new(index: usize, log_probability: f32) -> Result<Self> {
    let language = Language::from_index(index).ok_or(Error::UnknownLanguageIndex(index))?;
    Ok(Self {
      language,
      log_probability,
    })
  }

  /// The roster row for this language.
  #[inline]
  pub const fn language(&self) -> &'static Language {
    self.language
  }

  /// The model output column this language occupies.
  #[inline]
  pub const fn index(&self) -> usize {
    self.language.index()
  }

  /// The language code as upstream spells it, e.g. `"th"` — see
  /// [`Language::code`] on the legacy `iw`/`jw` spellings.
  #[inline]
  pub const fn code(&self) -> &'static str {
    self.language.code()
  }

  /// The English language name, e.g. `"Thai"`.
  #[inline]
  pub const fn name(&self) -> &'static str {
    self.language.name()
  }

  /// The model's score for this language, as it comes out of the graph: a
  /// **natural-log probability**, always `<= 0`, and already normalized (the
  /// graph's last op is a log-softmax, so `exp` over all
  /// [`NUM_LANGUAGES`] columns sums to 1).
  ///
  /// This is the value ranking compares, and the one to threshold on when a
  /// confident answer matters: log space keeps the resolution that `exp`
  /// flattens away near 1.
  #[inline]
  pub const fn log_probability(&self) -> f32 {
    self.log_probability
  }

  /// [`Self::log_probability`] mapped back through `exp` into a probability in
  /// `[0, 1]`.
  ///
  /// Convenience for display. Prefer the log form for comparisons: two
  /// languages at −0.0101 and −4.60 are three orders of magnitude apart in
  /// evidence, which reads as 0.990 versus 0.010, but two at −18.7 and −25.0
  /// both round to 0.000.
  #[inline]
  #[must_use]
  pub fn probability(&self) -> f32 {
    self.log_probability.exp()
  }
}

/// Ranking key: score under `f32::total_cmp`, ties broken by ascending
/// language index (a smaller index compares GREATER at equal scores, so it
/// surfaces first in descending output) — the crate's existing `RankedScore`
/// contract.
#[derive(Debug, Clone, Copy)]
struct RankedScore {
  index: usize,
  score: f32,
}

impl PartialEq for RankedScore {
  fn eq(&self, other: &Self) -> bool {
    self.index == other.index && self.score.total_cmp(&other.score) == Ordering::Equal
  }
}

impl Eq for RankedScore {}

impl PartialOrd for RankedScore {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for RankedScore {
  fn cmp(&self, other: &Self) -> Ordering {
    self
      .score
      .total_cmp(&other.score)
      .then_with(|| other.index.cmp(&self.index))
  }
}

/// Select the top `k` of `scores` (`(index, log_probability)` pairs) without a
/// full sort: a size-`k` min-heap of [`Reverse`]d [`RankedScore`]s, replacing
/// the smallest whenever a larger candidate arrives.
///
/// `k == 0` yields an empty vec; `k` above the roster size saturates. Capacity
/// is clamped to the roster size so a caller's "give me everything" sentinel
/// (`usize::MAX`) cannot overflow the pre-allocation.
///
/// # Errors
/// [`Error::UnknownLanguageIndex`] if a surviving index has no roster row
/// (defensive; unreachable for in-range indices).
pub(crate) fn top_k_from_scores(
  scores: impl IntoIterator<Item = (usize, f32)>,
  k: usize,
) -> Result<Vec<LanguageScore>> {
  if k == 0 {
    return Ok(Vec::new());
  }

  let mut heap = BinaryHeap::with_capacity(k.min(super::NUM_LANGUAGES));
  for (index, score) in scores {
    let candidate = Reverse(RankedScore { index, score });
    if heap.len() < k {
      heap.push(candidate);
      continue;
    }
    if heap.peek().is_some_and(|smallest| candidate.0 > smallest.0) {
      heap.pop();
      heap.push(candidate);
    }
  }

  let mut ranked = Vec::with_capacity(heap.len());
  while let Some(entry) = heap.pop() {
    ranked.push(LanguageScore::new(entry.0.index, entry.0.score)?);
  }
  ranked.reverse();
  Ok(ranked)
}
