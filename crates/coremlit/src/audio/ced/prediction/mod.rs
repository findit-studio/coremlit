//! The prediction vocabulary: [`EventPrediction`] (one ranked class),
//! [`Confidences`] (the per-class sigmoid-confidence vector), and the
//! min-heap top-k shared by every classify path.
//!
//! Ranking is the soundevents `RankedScore` contract, pinned by the sibling
//! tests as soundevents-identical: `f32::total_cmp` descending on score, ties
//! broken by **ascending class index**. Single-window ranking runs the heap
//! over raw logits and maps sigmoid at extraction (monotonic ⇒ identical
//! ranking, no 527-element sort — soundevents' exact trick); long-clip ranking
//! runs the same heap over aggregated confidences with the identity map. Ties
//! in the raw logit are broken by ascending class index; distinct logits that
//! saturate to equal f32 confidences keep logit order (the tie-break key is
//! always the pre-sigmoid score, never the extracted confidence).

use core::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::audio::ced::{
  NUM_CLASSES,
  error::{Error, Result},
};

/// The 527-class rated AudioSet vocabulary, re-exported from the ort-free
/// `soundevents-dataset` data crate so callers can name event rows without a
/// direct dependency.
pub use soundevents_dataset::RatedSoundEvent;

#[cfg(test)]
mod tests;

/// Per-window classification output: a [`Confidences`] vector paired with the
/// [`Span`](crate::audio::ced::window::Span) of input it was computed from
/// (`windit::windowed::Windowed<Confidences>`). Build with
/// [`WindowConfidences::new`](windit::windowed::Windowed::new); read with
/// [`value`](windit::windowed::Windowed::value) /
/// [`span`](windit::windowed::Windowed::span). Carrying the span is what makes
/// time-localized tagging ("when did the dog bark") a caller-side read — no
/// second API needed. To build one in a test without running a model, get the
/// value from [`Confidences::try_from_slice`].
pub type WindowConfidences = windit::windowed::Windowed<Confidences>;

/// One ranked AudioSet prediction: a rated event row plus its sigmoid
/// confidence — the soundevents surface, coremlit-native.
#[derive(Debug, Clone, Copy)]
pub struct EventPrediction {
  event: &'static RatedSoundEvent,
  confidence: f32,
}

impl EventPrediction {
  /// Resolve `class_index` to its rated event row.
  ///
  /// # Errors
  /// [`Error::UnknownClassIndex`] if the index has no rated row — defensive:
  /// the compile-time `NUM_CLASSES == events().len()` assert makes this
  /// unreachable for in-range indices.
  pub(crate) fn from_confidence(class_index: usize, confidence: f32) -> Result<Self> {
    let event = RatedSoundEvent::from_index(class_index)
      .ok_or(Error::UnknownClassIndex { index: class_index })?;
    Ok(Self { event, confidence })
  }

  /// The full rated AudioSet event row.
  #[inline]
  pub const fn event(&self) -> &'static RatedSoundEvent {
    self.event
  }

  /// The model output index of this class.
  #[inline]
  pub const fn index(&self) -> usize {
    self.event.index()
  }

  /// Human-readable class name, e.g. `"Speech"`.
  #[inline]
  pub const fn name(&self) -> &'static str {
    self.event.name()
  }

  /// Stable AudioSet identifier, e.g. `"/m/09x0r"`.
  #[inline]
  pub const fn id(&self) -> &'static str {
    self.event.id()
  }

  /// Confidence after applying a sigmoid to the model's raw logit (or, for a
  /// long clip, after Mean/Max aggregation of per-window confidences).
  #[inline]
  pub const fn confidence(&self) -> f32 {
    self.confidence
  }
}

/// The per-class sigmoid-confidence vector for one window (or one aggregated
/// clip): always exactly [`NUM_CLASSES`] finite values in `[0, 1]`. Finiteness
/// is established at the model boundary (`raw_scores` rejects non-finite
/// logits before sigmoid) and preserved by Mean/Max aggregation.
#[derive(Debug, Clone, PartialEq)]
pub struct Confidences {
  values: Vec<f32>,
}

impl Confidences {
  /// Wrap an already-confidence-space vector.
  ///
  /// # Panics
  /// If `values.len() != NUM_CLASSES` — an internal invariant (every producer
  /// is post-shape-check), not a caller-reachable path.
  pub(crate) fn new(values: Vec<f32>) -> Self {
    assert!(
      values.len() == NUM_CLASSES,
      "Confidences requires exactly NUM_CLASSES values, got {}",
      values.len()
    );
    Self { values }
  }

  /// Map raw logits (already finite-checked at the model boundary) through the
  /// sigmoid into confidence space.
  ///
  /// # Panics
  /// As [`Self::new`], on a wrong-length slice (internal invariant).
  pub(crate) fn from_logits(logits: &[f32]) -> Self {
    Self::new(logits.iter().copied().map(sigmoid).collect())
  }

  /// Build a confidence vector by hand, for a consumer's own tests: it is what
  /// lets downstream event logic — class projection, smoothing, segmentation —
  /// be unit-tested on synthetic window scores with no staged `.mlmodelc` and
  /// no inference. It is not a general-purpose builder, and the classify path
  /// does not go through it.
  ///
  /// `values` is read positionally as confidences that are ALREADY in
  /// confidence space, one per class, indexed exactly as [`Self::as_slice`]
  /// returns them ([`EventPrediction::index`] / [`RatedSoundEvent::index`]).
  /// How the model path reaches those numbers is deliberately not part of this
  /// contract: it takes no logits and applies no sigmoid, so the internal
  /// logit→confidence step stays free to change. The slice is copied rather
  /// than adopted, so neither is the storage `Confidences` chooses.
  ///
  /// Pair it with [`WindowConfidences::new`](windit::windowed::Windowed::new)
  /// to stand in for one window of [`Classifier::classify_windows`](crate::audio::ced::Classifier::classify_windows)
  /// output; `audio::ced`'s module docs run the whole pipeline that way.
  ///
  /// # Errors
  /// [`Error::ClassCountMismatch`] if `values.len() != `[`NUM_CLASSES`];
  /// [`Error::InvalidConfidence`] on any value outside `[0, 1]` — this type's
  /// stated invariant, which the model path gets for free from sigmoid and a
  /// hand-built vector has to be checked for.
  pub fn try_from_slice(values: &[f32]) -> Result<Self> {
    if values.len() != NUM_CLASSES {
      return Err(Error::ClassCountMismatch {
        expected: NUM_CLASSES,
        got: values.len(),
      });
    }
    for (index, &value) in values.iter().enumerate() {
      // NaN and both infinities fail this containment test too, so it is the
      // whole "finite and in [0, 1]" check, not just the bounds half.
      if !(0.0..=1.0).contains(&value) {
        return Err(Error::InvalidConfidence { index, value });
      }
    }
    Ok(Self::new(values.to_vec()))
  }

  /// The per-class confidences, indexed by class index
  /// ([`EventPrediction::index`] / [`RatedSoundEvent::index`]).
  #[inline]
  pub fn as_slice(&self) -> &[f32] {
    &self.values
  }

  /// The top `k` classes by confidence, descending, ties broken by ascending
  /// class index. Unlike the single-window logit path, ranking runs directly
  /// on these confidence values (the identity map), so what is compared IS
  /// what is returned — no separate raw key, hence no f32-saturation subtlety
  /// here. `k == 0` yields an empty vec; `k > NUM_CLASSES` saturates.
  ///
  /// # Errors
  /// [`Error::UnknownClassIndex`] — defensive only (see
  /// `EventPrediction::from_confidence`, `pub(crate)` so not doc-linkable).
  pub fn top_k(&self, k: usize) -> Result<Vec<EventPrediction>> {
    top_k_from_scores(self.values.iter().copied().enumerate(), k, |c| c)
  }
}

/// The soundevents sigmoid, verbatim: `1 / (1 + e^{-x})` in f32.
pub(crate) fn sigmoid(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}

/// Ranking key: score under `f32::total_cmp`, ties broken by ascending class
/// index — soundevents' `RankedScore` contract (a smaller index compares
/// GREATER at equal scores, so it surfaces first in descending output).
#[derive(Debug, Clone, Copy)]
struct RankedScore {
  class_index: usize,
  score: f32,
}

impl PartialEq for RankedScore {
  fn eq(&self, other: &Self) -> bool {
    self.class_index == other.class_index && self.score.total_cmp(&other.score) == Ordering::Equal
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
      .then_with(|| other.class_index.cmp(&self.class_index))
  }
}

/// Select the top `k` of `scores` (pairs of `(class_index, score)`) without a
/// full sort: a size-`k` min-heap of [`Reverse`]d [`RankedScore`]s, replacing
/// the smallest whenever a larger candidate arrives — soundevents'
/// `top_k_from_scores`, verbatim. `map_score` maps each surviving raw score at
/// extraction (sigmoid for the logit path, identity for confidences).
///
/// # Errors
/// [`Error::UnknownClassIndex`] if a surviving `class_index` has no rated row
/// (defensive; unreachable for in-range indices).
pub(crate) fn top_k_from_scores(
  scores: impl IntoIterator<Item = (usize, f32)>,
  k: usize,
  map_score: impl Fn(f32) -> f32,
) -> Result<Vec<EventPrediction>> {
  if k == 0 {
    return Ok(Vec::new());
  }

  // Capacity is `k.min(NUM_CLASSES)`, not the raw caller `k`: every
  // in-module score stream holds at most NUM_CLASSES items, so the heap
  // never needs more regardless of `k`. The unclamped `k` previously let a
  // natural "give me everything" sentinel like `usize::MAX` panic on
  // capacity overflow (and `k ~= 2^40` abort via `handle_alloc_error`)
  // before the saturation loop below ever ran. Output is unchanged for
  // `k <= NUM_CLASSES`; only this pre-allocation is clamped.
  let mut heap = BinaryHeap::with_capacity(k.min(NUM_CLASSES));
  for (class_index, score) in scores {
    let candidate = Reverse(RankedScore { class_index, score });
    if heap.len() < k {
      heap.push(candidate);
      continue;
    }
    if heap.peek().is_some_and(|smallest| candidate.0 > smallest.0) {
      heap.pop();
      heap.push(candidate);
    }
  }

  let mut predictions = Vec::with_capacity(heap.len());
  while let Some(entry) = heap.pop() {
    let ranked = entry.0;
    predictions.push(EventPrediction::from_confidence(
      ranked.class_index,
      map_score(ranked.score),
    )?);
  }
  predictions.reverse();
  Ok(predictions)
}
