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

/// The configuration [`AggregatePolicyKind::EmaRenormalized`] carries: the
/// smoothing factor an [`EmaRenormalized`] policy is built with, once a value
/// has been read off a config surface.
///
/// Construction is infallible and the range is checked where the value is used,
/// the same contract [`EmaRenormalized::new`] itself keeps — see [`Self::new`].
///
/// # Why it is not named `EmaRenormalized`
///
/// A variant and its payload struct may otherwise share a name, living in
/// different namespaces; here they cannot, because [`EmaRenormalized`] is
/// already bound in this module as windit's re-exported policy type. A bare
/// `EmaOptions` would clear that collision and still be wrong: the flat
/// [`embeddings::clap`] namespace this type is re-exported into holds a SECOND
/// ema whose knob is also an `alpha` — the streaming [`VectorEma`] — so the name
/// has to say which ema it configures. `…Options` is this crate's suffix for a
/// configuration carrier ([`AudioEncoderOptions`], [`TextEncoderOptions`], …),
/// and is what windit calls the same infallible-construction,
/// validated-where-used shape ([`WindowOptions`]).
///
/// [`embeddings::clap`]: crate::embeddings::clap
/// [`VectorEma`]: crate::embeddings::clap::smooth::VectorEma
/// [`AudioEncoderOptions`]: crate::embeddings::clap::audio::AudioEncoderOptions
/// [`TextEncoderOptions`]: crate::embeddings::clap::text::TextEncoderOptions
/// [`WindowOptions`]: windit::plan::WindowOptions
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmaRenormalizedOptions {
  /// The EMA smoothing factor, forwarded to [`EmaRenormalized::new`].
  ///
  /// `f64`, matching the compute domain the factor multiplies: this is the
  /// *wire* type, deserialized from a config surface before any embedding
  /// exists, and a decimal in a file has no compute domain of its own.
  /// [`aggregate`] widens clap's `f32` storage to `f64` before folding a
  /// single component, so an `f32` field here would have resolved the weight
  /// at `2^-24` inside a sum that rounds at `2^-53` — the defect windit fixed
  /// in its own selector.
  ///
  /// What that width change does to a legacy JSON document whose `alpha` was
  /// written for the old `f32` field, measured by parsing the same literal as
  /// `f32` then as `f64`:
  /// - **Structure** is preserved: an old document still deserializes here
  ///   (pinned by the golden round-trip), because the object shape does not
  ///   depend on the field's width. A fixed-width binary format is a separate
  ///   question from JSON: bincode encodes an `f32` and an `f64` at different
  ///   byte widths on the wire, and
  ///   [`AggregatePolicyKind::EmaRenormalized`]'s own shape change affects a
  ///   format like RON regardless of this field's type (see that variant's
  ///   docs).
  /// - **Value** is not preserved: `0.30000000000000004` parses as `f32` to
  ///   `0.3`, which reserializes to `0.3`; the same literal parses as `f64`
  ///   to `0.30000000000000004` and reserializes unchanged — different
  ///   bytes.
  /// - **Behaviour** is not preserved either, and this is the one a caller
  ///   can be hurt by: `1.00000001` parses as `f32` to exactly `1.0`, inside
  ///   the `[0, 1]` range [`aggregate`] accepts; the same literal parses as
  ///   `f64` to `1.00000001`, outside it. A configuration that used to
  ///   aggregate now fails there instead, as [`Error::Windowing`] carrying
  ///   `WinditError::AlphaOutOfRange`.
  ///
  /// None of this touches a configuration authored for this `f64` field: a
  /// configured `0.3` reaches the fold at full `f64` precision rather than
  /// through the `f32` grid, which moves an EMA aggregate in its eighth
  /// significant digit. Pass `f64::from(0.3f32)` to reproduce the pre-0.3
  /// weights bit for bit.
  ///
  /// Required on the wire — deliberately no `serde(default)`, because a config
  /// that forgets `alpha` is a misconfiguration and not a request for `0.0`,
  /// the value that pins the fold to the first window.
  alpha: f64,
}

impl EmaRenormalizedOptions {
  /// An EMA configuration with the given smoothing factor.
  ///
  /// Infallible, like the [`EmaRenormalized::new`] it feeds: an `alpha` outside
  /// `[0, 1]` (or a NaN) is reported by [`aggregate`] as [`Error::Windowing`]
  /// carrying `WinditError::AlphaOutOfRange`, never here — which is what lets
  /// [`AggregatePolicyKind::into_policy`] have no error channel of its own.
  /// `const`, so a configuration can be named in a `const` or `static`.
  pub const fn new(alpha: f64) -> Self {
    Self { alpha }
  }

  /// The configured smoothing factor: larger values track recent windows more.
  #[inline]
  pub const fn alpha(&self) -> f64 {
    self.alpha
  }
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
/// Every variant is unit or newtype — the EMA knob lives in
/// [`EmaRenormalizedOptions`] rather than loose in the variant — and no
/// `is_`/`unwrap_`/`try_unwrap_` face is generated over them: every consumer
/// here matches exhaustively on purpose, which is what the two no-`_` matches
/// below buy, so a helper triple would be unspent public surface.
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
/// co-generates the enum and its roster in one macro; the payload-carrying
/// [`Self::EmaRenormalized`] is why the roster is hand-written here, so its
/// completeness is a maintained invariant rather than a compile-time guarantee.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum AggregatePolicyKind {
  /// Selects [`MeanRenormalized`].
  MeanRenormalized,
  /// Selects [`EmaRenormalized`], configured by [`EmaRenormalizedOptions`].
  ///
  /// Before this type existed, the payload lived directly in a struct variant
  /// (`EmaRenormalized { alpha: f32 }`). For a self-describing map format the
  /// wire form is unchanged: JSON renders a struct variant and a newtype
  /// variant wrapping the same fields identically, so
  /// `{"ema_renormalized":{"alpha":0.5}}` is still exactly what a config file
  /// holds, and that old-shape document still deserializes here (both
  /// measured, and pinned by the golden round-trip). That does **not**
  /// generalize: serde's data model distinguishes a struct variant from a
  /// newtype variant wrapping one, and a shape-sensitive format shows it — RON
  /// measures as `ema_renormalized(alpha:0.5)` for the old shape and
  /// `ema_renormalized((alpha:0.5))` for this one, and the old string fails to
  /// deserialize here. A fixed-width binary format such as bincode is
  /// shape-sensitive the same way. Accepted deliberately: every variant here
  /// is unit or newtype only (see above), never struct; this crate has never
  /// published a release, so no old-shape document exists to break.
  EmaRenormalized(EmaRenormalizedOptions),
  /// Selects [`CoverageWeightedMean`].
  CoverageWeightedMean,
}

impl AggregatePolicyKind {
  /// Convert to the boxed trait object [`aggregate`] runs.
  ///
  /// Infallible: [`Self::EmaRenormalized`]'s
  /// [`alpha`](EmaRenormalizedOptions::alpha) is validated when the policy runs
  /// (through [`aggregate`]), so a config that names a built-in always yields a
  /// policy, and a bad `alpha` fails loudly at aggregation as
  /// [`Error::Windowing`] carrying `WinditError::AlphaOutOfRange` rather than
  /// here.
  pub fn into_policy(self) -> Box<dyn AggregatePolicy + Send + Sync> {
    match self {
      Self::MeanRenormalized => Box::new(MeanRenormalized),
      Self::EmaRenormalized(ema) => Box::new(EmaRenormalized::new(ema.alpha())),
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
    Self::EmaRenormalized(EmaRenormalizedOptions::new(0.5)),
    Self::CoverageWeightedMean,
  ];
}
