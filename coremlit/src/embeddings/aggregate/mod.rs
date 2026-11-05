//! The window-aggregation selector the embedding doors share: the serde-able
//! [`AggregatePolicyKind`] and the [`EmaRenormalizedOptions`] it carries.
//!
//! A door embeds a long input as several windows and combines them under one of
//! windit's `windit::aggregate::AggregatePolicy` strategies. This module is the
//! part of that surface a config file, CLI flag or environment variable
//! touches: a closed enum naming the built-in strategies, whose wire spellings
//! are owned and pinned here rather than by windit (windit's own `serde`
//! feature is deliberately not what produces them, so its differently-spelled
//! kind enum never defines the wire form). Each consuming door re-exports both
//! types at its own path.

use windit::aggregate::{AggregatePolicy, CoverageWeightedMean, EmaRenormalized, MeanRenormalized};

/// The configuration [`AggregatePolicyKind::EmaRenormalized`] carries: the
/// smoothing factor a `windit::aggregate::EmaRenormalized` policy is built
/// with, once a value has been read off a config surface.
///
/// Construction is infallible and the range is checked where the value is used,
/// the same contract [`EmaRenormalized::new`] itself keeps — see [`Self::new`].
///
/// # Why it is not named `EmaRenormalized`
///
/// A variant and its payload struct may otherwise share a name, living in
/// different namespaces; here they cannot, because `EmaRenormalized` is
/// already bound in this module as windit's policy type. A bare `EmaOptions`
/// would clear that collision and still be wrong: a door's flat namespace may
/// hold a SECOND ema whose parameter is also an `alpha` — a streaming vector
/// smoother — so the name has to say which ema it configures. `…Options` is
/// this crate's suffix for a configuration carrier, and is what windit calls
/// the same infallible-construction, validated-where-used shape
/// ([`WindowOptions`]).
///
/// [`WindowOptions`]: windit::plan::WindowOptions
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmaRenormalizedOptions {
  /// The EMA smoothing factor, forwarded to [`EmaRenormalized::new`].
  ///
  /// `f64`, matching the compute domain the factor multiplies: this is the
  /// *wire* type, deserialized from a config surface before any embedding
  /// exists, and a decimal in a file has no compute domain of its own.
  /// Aggregation widens a door's `f32` storage to `f64` before folding a
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
  ///   the `[0, 1]` range aggregation accepts; the same literal parses as
  ///   `f64` to `1.00000001`, outside it. A configuration that used to
  ///   aggregate now fails there instead, as the door's windowing error
  ///   carrying `WinditError::AlphaOutOfRange`.
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
  /// `[0, 1]` (or a NaN) is reported by aggregation as the door's windowing
  /// error carrying `WinditError::AlphaOutOfRange`, never here — which is what
  /// lets [`AggregatePolicyKind::into_policy`] have no error channel of its
  /// own. `const`, so a configuration can be named in a `const` or `static`.
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
/// Custom policies use `windit::aggregate::AggregatePolicy` directly — this
/// wrapper exists only so the *built-ins* survive a round trip through text.
/// [`Self::into_policy`] converts a deserialized value into the trait object the
/// pipeline runs. The wire spellings are owned here and pinned (windit's own
/// kind enum never defines them); the mapping to windit policies happens in
/// [`Self::into_policy`].
///
/// Every variant is unit or newtype — the EMA parameter lives in
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
  /// Selects `windit::aggregate::MeanRenormalized`.
  MeanRenormalized,
  /// Selects `windit::aggregate::EmaRenormalized`, configured by
  /// [`EmaRenormalizedOptions`].
  ///
  /// The payload is a newtype variant wrapping [`EmaRenormalizedOptions`],
  /// not a struct variant (`EmaRenormalized { alpha }`). For a self-describing
  /// map format the wire form is the same either way: JSON renders a struct
  /// variant and a newtype variant wrapping the same fields identically, so
  /// `{"ema_renormalized":{"alpha":0.5}}` is exactly what a config file holds,
  /// and a struct-variant-shaped document deserializes here too (both
  /// measured, and pinned by the golden round-trip). That does **not**
  /// generalize: serde's data model distinguishes a struct variant from a
  /// newtype variant wrapping one, and a shape-sensitive format shows it — RON
  /// measures as `ema_renormalized(alpha:0.5)` for the struct shape and
  /// `ema_renormalized((alpha:0.5))` for this one, and the struct-shaped string
  /// fails to deserialize here. A fixed-width binary format such as bincode is
  /// shape-sensitive the same way. Accepted deliberately: every variant here
  /// is unit or newtype only (see above), never struct.
  EmaRenormalized(EmaRenormalizedOptions),
  /// Selects `windit::aggregate::CoverageWeightedMean`.
  CoverageWeightedMean,
}

impl AggregatePolicyKind {
  /// Convert to the boxed trait object aggregation runs.
  ///
  /// Infallible: [`Self::EmaRenormalized`]'s
  /// [`alpha`](EmaRenormalizedOptions::alpha) is validated when the policy runs
  /// (through the door's aggregation), so a config that names a built-in always
  /// yields a policy, and a bad `alpha` fails loudly at aggregation as the
  /// door's windowing error carrying `WinditError::AlphaOutOfRange` rather than
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

#[cfg(test)]
mod tests;
