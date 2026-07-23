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
//! windit's `serde` feature is deliberately NOT enabled, so its own
//! differently-spelled `AggregatePolicyKind` never compiles; the golden-pinned
//! wire spellings live on clap's own [`AggregatePolicyKind`] below, mapped to
//! windit policies in [`AggregatePolicyKind::into_policy`]. `SaliencyWeighted` is
//! deliberately not re-exported: [`aggregate`] feeds already-unit embeddings,
//! where saliency degenerates to the mean, so exposing it would ship a
//! misleading knob (experts can reach it via windit directly).

use crate::embeddings::clap::{
  embedding::Embedding,
  error::{Error, Result},
  window::WindowEmbedding,
};

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
/// policy implements [`AggregatePolicy`] over `&[&[f64]]`. Here one that trusts
/// only the highest-coverage window, exercised through the public seam, no model
/// required:
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
///         coverages: &[f32],
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
  windit::aggregate::aggregate(policy, windows).map_err(Error::from)
}

/// A serde-able closed enum over the built-in policies, for config surfaces
/// (a file, CLI flag, or env var that names the aggregation strategy).
///
/// Custom policies use [`AggregatePolicy`] directly — this wrapper exists only
/// so the *built-ins* survive a round trip through text.
/// [`Self::into_policy`] converts a deserialized value into the trait object the
/// pipeline runs. The wire spellings are clap-owned and pinned (windit's `serde`
/// feature is off, so its own kind enum never compiles); the mapping to windit
/// policies happens in [`Self::into_policy`].
///
/// # Golden-enum contract (what the tests actually force)
///
/// A wildcard-free golden test (`serde` feature) serializes each representative
/// in the test-only `REPRESENTATIVES` roster to a pinned JSON literal, round-trips
/// it, and rejects a non-`snake_case` spelling. Two exhaustive, no-`_` matches
/// stop a new variant being added half-way in the ways that matter at runtime:
///
/// - [`Self::into_policy`] has no `_` arm, so a new variant fails to compile until
///   it is dispatched to a policy.
/// - The golden test's `match kind` has no `_` arm, so a new variant fails to
///   compile until its expected JSON literal is written.
///
/// What is **not** compiler-enforced is roster completeness: the round-trip
/// iterates the hand-maintained test-only `REPRESENTATIVES` slice, so *executing* a
/// new variant's round-trip still requires adding it there (keep it complete).
/// This is weaker than alignkit's `define_alignment_fallback!`, which
/// co-generates the enum and its roster in one macro; the struct-carrying
/// [`Self::EmaRenormalized`] is why the roster is hand-written here, so its
/// completeness is a maintained invariant rather than a compile-time guarantee.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum AggregatePolicyKind {
  /// Selects [`MeanRenormalized`].
  MeanRenormalized,
  /// Selects [`EmaRenormalized`] with the given smoothing factor.
  EmaRenormalized {
    /// The EMA smoothing factor, forwarded to [`EmaRenormalized::new`].
    alpha: f32,
  },
  /// Selects [`CoverageWeightedMean`].
  CoverageWeightedMean,
}

impl AggregatePolicyKind {
  /// Convert to the boxed trait object [`aggregate`] runs.
  ///
  /// Infallible: [`Self::EmaRenormalized`]'s `alpha` is validated when the policy
  /// runs (through [`aggregate`]), so a config that names a built-in always
  /// yields a policy, and a bad `alpha` fails loudly at aggregation as
  /// [`Error::Windowing`] carrying `WinditError::AlphaOutOfRange` rather than
  /// here.
  pub fn into_policy(self) -> Box<dyn AggregatePolicy + Send + Sync> {
    match self {
      Self::MeanRenormalized => Box::new(MeanRenormalized),
      Self::EmaRenormalized { alpha } => Box::new(EmaRenormalized::new(alpha)),
      Self::CoverageWeightedMean => Box::new(CoverageWeightedMean),
    }
  }

  /// One representative per variant, in declaration order — the hand-maintained
  /// roster the golden serde round-trip iterates. Keep it complete: the golden
  /// test's exhaustive `match` forces a new variant's expected JSON to be written,
  /// but only a roster entry here makes that variant's round-trip actually run.
  #[cfg(all(test, feature = "serde"))]
  pub(crate) const REPRESENTATIVES: &'static [Self] = &[
    Self::MeanRenormalized,
    Self::EmaRenormalized { alpha: 0.5 },
    Self::CoverageWeightedMean,
  ];
}
