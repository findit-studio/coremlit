//! Per-window confidence aggregation for long clips:
//! [`ChunkAggregation`] `{ Mean, Max }` + [`aggregate_windows`] — soundevents'
//! chunked-inference semantics exactly.
//!
//! Both the batch [`aggregate_windows`] and `Classifier::classify_long` fold
//! through one streaming `Accumulator`, so Mean/Max are single-sourced: the
//! long-clip path never materializes a per-window vector for every window, yet
//! its result is bit-identical to aggregating the materialized slice.
//!
//! Aggregation runs in **confidence space** (per-window sigmoid confidences),
//! never logit space: sigmoid is nonlinear, so the two disagree, and
//! soundevents defines the contract as sigmoid-then-aggregate (pinned by a
//! mutation-red test in the sibling `tests.rs`). Mean is the equal-weight
//! arithmetic mean (f32 accumulation, one divide at the end — a single window
//! aggregates to itself bit-exactly); Max is the elementwise peak.
//!
//! windit's aggregation engine is deliberately NOT used here: its built-ins
//! are renormalizing unit-vector policies, the wrong domain for independent
//! per-class probabilities (spec §2).

use crate::audio::ced::{
  error::{Error, Result},
  prediction::{Confidences, WindowConfidences},
};

#[cfg(test)]
mod tests;

/// Controls how a long clip's per-window confidences combine into one
/// clip-level [`Confidences`] — soundevents' `ChunkAggregation`, spelling and
/// semantics (wire spellings `"mean"` / `"max"`, golden-pinned).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ChunkAggregation {
  /// Equal-weight arithmetic mean of each class's confidence across windows.
  /// The default.
  #[default]
  Mean,
  /// The peak confidence each class reached in any window.
  Max,
}

/// Streaming Mean/Max fold shared by [`aggregate_windows`] and
/// `Classifier::classify_long`, one window at a time — `classify_long` folds
/// each window's confidences in and never materializes the per-window vectors,
/// so a long clip retains O([`NUM_CLASSES`]) state rather than one 527-float
/// vector per window.
///
/// Bit-identical to the batch fold by construction: the SAME op sequence — copy
/// the first window, then `+=` (Mean) / `max` (Max) in window order, one
/// trailing divide gated on `count > 1` — over the same f32 storage, so the
/// golden aggregation values do not shift. Do NOT "improve" to f64 accumulation:
/// "f32 accumulation, one divide at the end" is the golden-pinned contract.
///
/// [`NUM_CLASSES`]: crate::audio::ced::NUM_CLASSES
#[derive(Debug, Clone)]
pub(crate) struct Accumulator {
  aggregation: ChunkAggregation,
  /// Empty until the first [`Self::push`]; then exactly `NUM_CLASSES` long
  /// (the first window's vector, folded in place).
  values: Vec<f32>,
  count: usize,
}

impl Accumulator {
  /// An empty fold under `aggregation`. [`Self::finish`] on it is
  /// [`Error::EmptyWindows`] until at least one window is pushed.
  pub(crate) fn new(aggregation: ChunkAggregation) -> Self {
    Self {
      aggregation,
      values: Vec::new(),
      count: 0,
    }
  }

  /// Fold one window's confidences in. The first window is copied verbatim (so
  /// a single-window fold is the bit-exact identity); each later window is
  /// combined elementwise in window order — summed for Mean, kept-if-greater
  /// for Max.
  pub(crate) fn push(&mut self, window: &Confidences) {
    if self.count == 0 {
      self.values = window.as_slice().to_vec();
    } else {
      match self.aggregation {
        ChunkAggregation::Mean => {
          for (a, &c) in self.values.iter_mut().zip(window.as_slice()) {
            *a += c;
          }
        }
        ChunkAggregation::Max => {
          for (a, &c) in self.values.iter_mut().zip(window.as_slice()) {
            *a = a.max(c);
          }
        }
      }
    }
    self.count += 1;
  }

  /// Finish the fold into one clip-level [`Confidences`]. Mean divides by the
  /// window count only when more than one window was pushed (a single window
  /// aggregates to itself bit-exactly); Max needs no scaling.
  ///
  /// # Errors
  /// [`Error::EmptyWindows`] if no window was pushed.
  pub(crate) fn finish(self) -> Result<Confidences> {
    if self.count == 0 {
      return Err(Error::EmptyWindows);
    }
    let mut values = self.values;
    if matches!(self.aggregation, ChunkAggregation::Mean) && self.count > 1 {
      let denominator = self.count as f32;
      for a in values.iter_mut() {
        *a /= denominator;
      }
    }
    Ok(Confidences::new(values))
  }
}

/// Combine per-window confidences into one clip-level [`Confidences`] under
/// `aggregation`, folding them through the shared `Accumulator`. Every
/// window's vector is [`NUM_CLASSES`]-long by type invariant, so no length
/// reconciliation is needed; finiteness is preserved (mean/max of finite
/// `[0, 1]` values is finite).
///
/// A single window aggregates to itself bit-exactly (soundevents divides only
/// when the window count exceeds 1).
///
/// # Errors
/// [`Error::EmptyWindows`] if `windows` is empty. (Unreachable through
/// `classify_long` — a nonempty clip always plans at least one span.)
///
/// [`NUM_CLASSES`]: crate::audio::ced::NUM_CLASSES
pub fn aggregate_windows(
  aggregation: ChunkAggregation,
  windows: &[WindowConfidences],
) -> Result<Confidences> {
  let mut acc = Accumulator::new(aggregation);
  for w in windows {
    acc.push(w.value());
  }
  acc.finish()
}
