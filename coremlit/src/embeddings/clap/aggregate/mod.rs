//! Window-embedding aggregation: coremlit re-exports windit's aggregation engine
//! — the object-safe [`AggregatePolicy`] seam and its built-in strategies
//! ([`CoverageWeightedMean`], [`MeanRenormalized`], [`EmaRenormalized`]) — and
//! adds a thin clap-typed [`aggregate`] wrapper plus the serde-able
//! [`AggregatePolicyKind`] selector for config surfaces.
//!
//! A long clip becomes a list of [`WindowEmbedding`]s (one per
//! [`Span`](crate::embeddings::clap::window::Span) produced by
//! [`WindowPlan`](crate::embeddings::clap::window::WindowPlan) and embedded by
//! [`AudioEncoder::embed_windows`](crate::embeddings::clap::AudioEncoder::embed_windows));
//! [`aggregate`] combines them into one clip-level [`Embedding`] under any
//! [`AggregatePolicy`]. The seam is windit's object-safe trait, so end users
//! implement it for strategies the built-ins don't cover.
//!
//! Per-window embeddings are always exposed upstream (see
//! [`AudioEncoder::embed_windows`](crate::embeddings::clap::AudioEncoder::embed_windows)) and
//! per-window zero-shot scores via
//! [`score_windows`](crate::embeddings::clap::score::score_windows), so score-level smoothing or
//! voting needs no second trait seam (the deliberate cut recorded in the spec
//! amendment).
//!
//! windit's own differently-spelled `AggregatePolicyKind` never defines the
//! wire form; the golden-pinned wire spellings live on the crate's shared
//! [`AggregatePolicyKind`] (re-exported below), mapped to windit policies in
//! [`AggregatePolicyKind::into_policy`]. `SaliencyWeighted` is deliberately not
//! re-exported: [`aggregate`] feeds already-unit embeddings, where saliency
//! degenerates to the mean, so exposing it would ship a misleading option
//! (experts can reach it via windit directly).

use crate::embeddings::clap::{
  embedding::Embedding,
  error::{Error, Result},
  window::WindowEmbedding,
};

pub use crate::embeddings::aggregate::{AggregatePolicyKind, EmaRenormalizedOptions};
pub use windit::aggregate::{
  AggregatePolicy, CoverageWeightedMean, EmaRenormalized, MeanRenormalized,
};

#[cfg(test)]
mod tests;

/// Aggregate per-window embeddings into one clip-level [`Embedding`] under
/// `policy`, translating windit's errors into clap's ([`Error::EmptyWindows`]
/// for an empty window slice, [`Error::Windowing`] otherwise).
///
/// This is the clap-typed wrapper over [`windit::aggregate::aggregate`]: the
/// generic `P` mirrors windit's, so both a concrete policy
/// (`&CoverageWeightedMean`) and a boxed one (`kind.into_policy().as_ref()`) fit.
///
/// # Errors
/// [`Error::EmptyWindows`] if `windows` is empty; [`Error::Windowing`] carrying
/// windit's typed error for any aggregation failure (an out-of-range
/// [`EmaRenormalized`] alpha, a determinacy-gate `NonFinite`, an allocator
/// refusal, …).
///
/// # Implementing a custom policy
///
/// The set is open. windit's trait is slice-level — values arrive already
/// widened to the `f64` compute domain and unit-normalized, and [`aggregate`]
/// reconstructs the [`Embedding`] from what the policy returns — so a custom
/// policy implements [`AggregatePolicy`] over `&[&[f64]]`, and reads the window
/// coverages as `&[f64]` — the domain [`Span::coverage`] itself resolves in, so
/// nothing the fold multiplies an embedding by is rounded through a narrower
/// grid first. Here one that trusts only the highest-coverage window, exercised
/// through the public seam, no model required:
///
/// [`Span::coverage`]: crate::embeddings::clap::window::Span::coverage
///
/// ```
/// use coremlit::embeddings::clap::aggregate::{AggregatePolicy, aggregate};
/// use coremlit::embeddings::clap::embedding::Embedding;
/// use coremlit::embeddings::clap::window::{Span, WindowEmbedding, WINDOW_SAMPLES};
/// use coremlit::embeddings::clap::error::WinditError;
///
/// struct MostCovered;
///
/// impl AggregatePolicy for MostCovered {
///     fn aggregate_values(
///         &self,
///         embeddings: &[&[f64]],
///         coverages: &[f64],
///         dim: usize,
///     ) -> Result<Vec<f64>, WinditError> {
///         let (best, _) = coverages
///             .iter()
///             .enumerate()
///             .max_by(|a, b| a.1.total_cmp(b.1))
///             .ok_or(WinditError::Empty)?;
///         let e = embeddings[best];
///         if e.len() != dim {
///             return Err(WinditError::DimMismatch { got: e.len(), expected: dim });
///         }
///         Ok(e.to_vec())
///     }
/// }
///
/// let mut a = [0.0f32; 512];
/// a[0] = 1.0;
/// let mut b = [0.0f32; 512];
/// b[1] = 1.0;
/// let windows = vec![
///     WindowEmbedding::new(
///         Embedding::from_slice_normalizing(&a)?,
///         Span::new(0, 120_000, WINDOW_SAMPLES),
///     ),
///     WindowEmbedding::new(
///         Embedding::from_slice_normalizing(&b)?,
///         Span::new(120_000, WINDOW_SAMPLES, WINDOW_SAMPLES),
///     ),
/// ];
///
/// let clip = aggregate(&MostCovered, &windows)?;
/// assert_eq!(clip.as_slice()[1], 1.0); // the full-coverage window won
/// # Ok::<(), coremlit::embeddings::clap::Error>(())
/// ```
pub fn aggregate<P>(policy: &P, windows: &[WindowEmbedding]) -> Result<Embedding>
where
  P: windit::aggregate::AggregatePolicy + ?Sized,
{
  // This crate's ONE place that produces `Error::EmptyWindows`, keyed on the
  // OBSERVED INPUT, not on the error windit returns: `WinditError::Empty` alone
  // cannot distinguish "the engine saw no windows" from "the policy refused". A
  // custom `AggregatePolicy` may return `WinditError::Empty` from
  // `aggregate_values` for a NONEMPTY `windows` slice — windit's own rustdoc
  // example for implementing a custom policy does exactly this
  // (`embeddings.first().ok_or(WinditError::Empty)?`), and a filtering policy
  // that delegates a filtered subset to a windit built-in gets a
  // contract-correct `Empty` from windit's `check_inputs` when nothing in the
  // subset qualifies — so matching on the RETURNED error would misreport that
  // as "zero windows were supplied".
  //
  // Guarding on the input sidesteps the ambiguity instead of relocating it:
  // windit's own `aggregate` runs the identical `windows.is_empty()` check
  // before invoking ANY policy, built-in or custom, so this guard fires on
  // exactly the inputs that used to reach `Empty` through the engine's own
  // short-circuit — behaviour is unchanged for an empty slice and for every
  // built-in policy. A nonempty slice always reaches the policy now, and
  // whatever it returns — `Empty` included — is reported faithfully as
  // `Error::Windowing`, never smuggled into the blanket `From<WinditError>`
  // impl — see that impl's doc for why it must stay total.
  if windows.is_empty() {
    return Err(Error::EmptyWindows);
  }
  windit::aggregate::aggregate(policy, windows).map_err(Error::Windowing)
}
