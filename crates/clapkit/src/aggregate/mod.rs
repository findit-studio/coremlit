//! Window-embedding aggregation: the open [`AggregatePolicy`] seam, its three
//! shipped built-ins, and the serde-able [`AggregatePolicyKind`] wrapper for
//! config surfaces.
//!
//! A long clip becomes a list of [`WindowEmbedding`]s (one per
//! [`WindowSpan`](crate::window::WindowSpan) produced by
//! [`WindowPlan`](crate::window::WindowPlan) and embedded by
//! [`AudioEncoder::embed_windows`](crate::AudioEncoder::embed_windows)); a policy
//! combines them into one clip-level [`Embedding`]. The set is deliberately
//! **open** — end users implement [`AggregatePolicy`] for strategies the
//! built-ins don't cover — so the seam is a trait, mirroring whisperkit's
//! object-safe `VoiceActivityDetector`.
//!
//! Per-window embeddings are always exposed upstream (see
//! [`AudioEncoder::embed_windows`](crate::AudioEncoder::embed_windows)) and
//! per-window zero-shot scores via
//! [`score_windows`](crate::score::score_windows), so score-level smoothing or
//! voting needs no second trait seam (the deliberate cut recorded in the spec
//! amendment).

use crate::{
  embedding::{EMBEDDING_DIM, Embedding},
  error::{Error, Result},
  window::WindowEmbedding,
};

#[cfg(test)]
mod tests;

/// Combines per-window embeddings into a single clip-level [`Embedding`].
///
/// This is the customization seam: the built-ins ([`MeanRenormalized`],
/// [`EmaRenormalized`], [`CoverageWeightedMean`]) implement it, and so can your
/// own type. It is **object-safe** — one `&self` method taking a slice and
/// returning an owned [`Embedding`], no generics and no `Self`-typed return — so
/// a config surface can hold one behind `Box<dyn AggregatePolicy + Send + Sync>`
/// (what [`AggregatePolicyKind::into_policy`] hands back) and a caller can pass
/// `&dyn AggregatePolicy` without monomorphizing the pipeline.
///
/// Each [`WindowEmbedding`] carries its embedding **and** its span, so a policy
/// can weight by time, overlap, or tail coverage — not only average uniformly.
///
/// # Contract
///
/// - Aggregating zero windows returns [`Error::EmptyWindows`]; every policy
///   needs at least one window to define a direction.
/// - The returned embedding is unit-norm (the built-ins renormalize; a custom
///   policy should return a value from an [`Embedding`] constructor, which
///   enforces the invariant).
///
/// # Implementing a custom policy
///
/// The set is open. Here a policy that trusts only the highest-coverage window
/// — exercised through the public trait, no model required:
///
/// ```
/// use clapkit::aggregate::AggregatePolicy;
/// use clapkit::embedding::Embedding;
/// use clapkit::window::{WindowEmbedding, WindowSpan};
/// use clapkit::Error;
///
/// struct MostCovered;
///
/// impl AggregatePolicy for MostCovered {
///     fn aggregate(&self, windows: &[WindowEmbedding]) -> Result<Embedding, Error> {
///         windows
///             .iter()
///             .max_by(|a, b| a.coverage().total_cmp(&b.coverage()))
///             .map(|w| w.embedding().clone())
///             .ok_or(Error::EmptyWindows)
///     }
/// }
///
/// let mut a = [0.0f32; 512];
/// a[0] = 1.0;
/// let mut b = [0.0f32; 512];
/// b[1] = 1.0;
/// let windows = vec![
///     WindowEmbedding::new(Embedding::from_slice_normalizing(&a)?, WindowSpan::new(0, 120_000)),
///     WindowEmbedding::new(Embedding::from_slice_normalizing(&b)?, WindowSpan::new(120_000, 480_000)),
/// ];
///
/// let clip = MostCovered.aggregate(&windows)?;
/// assert_eq!(clip.as_slice()[1], 1.0); // the full-coverage window won
/// # Ok::<(), clapkit::Error>(())
/// ```
pub trait AggregatePolicy {
  /// Combine `windows` into one unit-norm [`Embedding`].
  ///
  /// # Errors
  /// [`Error::EmptyWindows`] if `windows` is empty; otherwise any error the
  /// policy's own normalization raises (e.g. [`Error::EmbeddingZero`] if the
  /// combined vector cancels to zero magnitude).
  fn aggregate(&self, windows: &[WindowEmbedding]) -> Result<Embedding>;
}

/// Component-wise mean of the window embeddings, renormalized to unit length —
/// the default policy.
///
/// Because every window embedding is already unit-norm, this is a spherical mean
/// (average the unit vectors, then renormalize). Equal weight per window,
/// regardless of coverage or overlap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeanRenormalized;

impl AggregatePolicy for MeanRenormalized {
  fn aggregate(&self, windows: &[WindowEmbedding]) -> Result<Embedding> {
    if windows.is_empty() {
      return Err(Error::EmptyWindows);
    }
    // Accumulate in f64 for order-independent stability, then renormalize (the
    // /n is immaterial to direction but keeps the value a true mean).
    let mut acc = [0.0f64; EMBEDDING_DIM];
    for w in windows {
      for (a, &v) in acc.iter_mut().zip(w.embedding().as_slice()) {
        *a += v as f64;
      }
    }
    let inv_n = 1.0 / windows.len() as f64;
    let mean: Vec<f32> = acc.iter().map(|&x| (x * inv_n) as f32).collect();
    Embedding::from_slice_normalizing(&mean)
  }
}

/// Exponential moving average across windows in order, renormalized.
///
/// `ema₀ = windows[0]`, then `emaᵢ = alpha·windowᵢ + (1−alpha)·emaᵢ₋₁`, and the
/// result is L2-renormalized. Temporal smoothing that leans on later windows as
/// `alpha → 1` (`alpha == 1` reduces to the last window; `alpha == 0` to the
/// first). `alpha` is validated finite in `[0, 1]` when [`Self::aggregate`] runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmaRenormalized {
  /// Smoothing factor in `[0, 1]`: higher weights recent windows more.
  pub alpha: f32,
}

impl EmaRenormalized {
  /// An EMA policy with smoothing factor `alpha` (validated at aggregation
  /// time, not here — construction is infallible so this stays a `const fn`).
  pub const fn new(alpha: f32) -> Self {
    Self { alpha }
  }
}

impl AggregatePolicy for EmaRenormalized {
  fn aggregate(&self, windows: &[WindowEmbedding]) -> Result<Embedding> {
    if windows.is_empty() {
      return Err(Error::EmptyWindows);
    }
    if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
      return Err(Error::InvalidPolicyParameter {
        policy: "EmaRenormalized",
        param: "alpha",
        value: self.alpha,
      });
    }
    let a = self.alpha as f64;
    let mut ema = [0.0f64; EMBEDDING_DIM];
    for (e, &v) in ema.iter_mut().zip(windows[0].embedding().as_slice()) {
      *e = v as f64;
    }
    for w in &windows[1..] {
      for (e, &v) in ema.iter_mut().zip(w.embedding().as_slice()) {
        *e = a * (v as f64) + (1.0 - a) * *e;
      }
    }
    let out: Vec<f32> = ema.iter().map(|&x| x as f32).collect();
    Embedding::from_slice_normalizing(&out)
  }
}

/// Coverage-weighted mean, renormalized: each window contributes in proportion
/// to its [`WindowEmbedding::coverage`], so a `repeatpad`-padded tail is
/// down-weighted relative to full interior windows.
///
/// `Σ(coverageᵢ · windowᵢ) / Σ coverageᵢ`, then renormalized. With every window
/// at full coverage this equals [`MeanRenormalized`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoverageWeightedMean;

impl AggregatePolicy for CoverageWeightedMean {
  fn aggregate(&self, windows: &[WindowEmbedding]) -> Result<Embedding> {
    if windows.is_empty() {
      return Err(Error::EmptyWindows);
    }
    let mut acc = [0.0f64; EMBEDDING_DIM];
    let mut weight_sum = 0.0f64;
    for w in windows {
      let cov = w.coverage() as f64; // in (0, 1]; never zero (real_len >= 1)
      weight_sum += cov;
      for (a, &v) in acc.iter_mut().zip(w.embedding().as_slice()) {
        *a += cov * v as f64;
      }
    }
    // weight_sum > 0: windows is non-empty and every coverage is > 0.
    let inv = 1.0 / weight_sum;
    let weighted: Vec<f32> = acc.iter().map(|&x| (x * inv) as f32).collect();
    Embedding::from_slice_normalizing(&weighted)
  }
}

/// A serde-able closed enum over the built-in policies, for config surfaces
/// (a file, CLI flag, or env var that names the aggregation strategy).
///
/// Custom policies use [`AggregatePolicy`] directly — this wrapper exists only
/// so the *built-ins* survive a round trip through text.
/// [`Self::into_policy`] converts a deserialized value into the trait object the
/// pipeline runs.
///
/// # Golden-enum contract
///
/// The wire spelling of every variant is pinned by a wildcard-free golden test
/// (`serde` feature): each representative in the test-only `REPRESENTATIVES` roster serializes
/// to a pinned JSON literal and round-trips, and a non-`snake_case` spelling must
/// not deserialize. Adding a variant forces both that test's match and
/// [`Self::into_policy`]'s match to grow (neither has a `_` arm), so a variant
/// cannot be half-added — the same lockstep guarantee alignkit's
/// `define_alignment_fallback!` gives, adapted for the struct-carrying
/// [`Self::EmaRenormalized`] a fieldless-spelling macro can't express.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum AggregatePolicyKind {
  /// Selects [`MeanRenormalized`].
  MeanRenormalized,
  /// Selects [`EmaRenormalized`] with the given smoothing factor.
  EmaRenormalized {
    /// The EMA smoothing factor; see [`EmaRenormalized::alpha`].
    alpha: f32,
  },
  /// Selects [`CoverageWeightedMean`].
  CoverageWeightedMean,
}

impl AggregatePolicyKind {
  /// Convert to the boxed trait object the pipeline aggregates with.
  ///
  /// Infallible: [`Self::EmaRenormalized`]'s `alpha` is validated when the
  /// policy runs ([`AggregatePolicy::aggregate`]), so a config that names a
  /// built-in always yields a policy, and a bad `alpha` fails loudly at
  /// aggregation with [`Error::InvalidPolicyParameter`] rather than here.
  pub fn into_policy(self) -> Box<dyn AggregatePolicy + Send + Sync> {
    match self {
      Self::MeanRenormalized => Box::new(MeanRenormalized),
      Self::EmaRenormalized { alpha } => Box::new(EmaRenormalized { alpha }),
      Self::CoverageWeightedMean => Box::new(CoverageWeightedMean),
    }
  }

  /// One representative per variant, in declaration order — the roster the
  /// golden serde test iterates so a new variant must pin its wire form.
  #[cfg(all(test, feature = "serde"))]
  pub(crate) const REPRESENTATIVES: &'static [Self] = &[
    Self::MeanRenormalized,
    Self::EmaRenormalized { alpha: 0.5 },
    Self::CoverageWeightedMean,
  ];
}
