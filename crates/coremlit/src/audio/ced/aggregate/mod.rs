//! Per-window confidence aggregation for long clips:
//! [`ChunkAggregation`] `{ Mean, Max }` + [`aggregate_windows`] — soundevents'
//! chunked-inference semantics exactly.
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

/// Combine per-window confidences into one clip-level [`Confidences`] under
/// `aggregation`. Every window's vector is [`NUM_CLASSES`]-long by type
/// invariant, so no length reconciliation is needed; finiteness is preserved
/// (mean/max of finite `[0, 1]` values is finite).
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
  let Some((first, rest)) = windows.split_first() else {
    return Err(Error::EmptyWindows);
  };
  let mut acc: Vec<f32> = first.value().as_slice().to_vec();
  for w in rest {
    match aggregation {
      ChunkAggregation::Mean => {
        for (a, &c) in acc.iter_mut().zip(w.value().as_slice()) {
          *a += c;
        }
      }
      ChunkAggregation::Max => {
        for (a, &c) in acc.iter_mut().zip(w.value().as_slice()) {
          *a = a.max(c);
        }
      }
    }
  }
  if matches!(aggregation, ChunkAggregation::Mean) && windows.len() > 1 {
    let denominator = windows.len() as f32;
    for a in acc.iter_mut() {
      *a /= denominator;
    }
  }
  Ok(Confidences::new(acc))
}
