//! The runtime clustering config surface: [`ClusterBackend`] and its
//! [`Offline`](ClusterBackend::Offline) hyperparameters ([`OfflineOptions`]).
//!
//! [`crate::extract::Extraction::diarize_with`] takes a [`ClusterBackend`] and
//! runs the named engine over an [`crate::extract::Extraction`];
//! [`crate::extract::Extraction::diarize`] is the same thing at the default
//! backend ([`ClusterBackend::default`]), the path every parity harness drives.
//!
//! # One engine today: `Offline`
//!
//! [`ClusterBackend`] is `#[non_exhaustive]` and carries exactly one variant so
//! far — [`Offline`](ClusterBackend::Offline), which wraps dia's
//! pyannote-community-1 offline pipeline
//! ([`dia::offline::diarize_offline`]). The online (streaming) engine is a
//! genuinely different algorithm class (a greedy centroid matcher, not
//! AHC→VBx) and lands as its own variant in a later task; there is no
//! not-yet-implemented stub variant here (honest surface: what compiles, runs).
//!
//! ## Which dia entry point `Offline` wraps (and which it does NOT)
//!
//! dia has TWO disjoint offline entry points, and this is the subtle one:
//!
//! - [`dia::offline::diarize_offline`] — the pyannote-parity PIPELINE (AHC
//!   initialization → VBx refinement; `threshold = 0.6`; the
//!   `fa`/`fb`/`max_iters`/`min_duration_off` hyperparameters carried inline on
//!   [`dia::offline::OfflineInput`]). This is what every DER gate validates and
//!   what [`OfflineOptions`] configures.
//! - `dia::cluster::cluster_offline` — a separate BATCH clusterer
//!   (agglomerative/spectral, `similarity_threshold = 0.5`, `target_speakers`,
//!   `seed`), a DIFFERENT algorithm surface never validated against the parity
//!   corpus. Its `OfflineClusterOptions`/`OfflineMethod`/`Linkage` vocabulary is
//!   deliberately NOT part of speakerkit's surface (see [`crate`]'s own
//!   root-doc note on the removed re-export). If a batch-clustering mode is
//!   ever wanted it arrives as its own [`ClusterBackend`] variant with its own
//!   gates.
//!
//! # The [`OfflineOptions`] knob set == dia's `OfflineInput` hyperparameters
//!
//! [`OfflineOptions`] mirrors, one-for-one, the five community-1
//! hyperparameters [`dia::offline::OfflineInput`] exposes through its `with_*`
//! builders — [`threshold`](OfflineOptions::threshold),
//! [`fa`](OfflineOptions::fa), [`fb`](OfflineOptions::fb),
//! [`max_iters`](OfflineOptions::max_iters), and
//! [`min_duration_off`](OfflineOptions::min_duration_off) — and every default
//! equals dia's, which equals pyannote's (`cluster::defaults_equal_dia`
//! pins this against dia's OWN `OfflineInput` accessors, so a drift on EITHER
//! side fails to compile the assertion). [`OfflineOptions::default`] therefore
//! produces byte-identical clustering to feeding a bare
//! [`dia::offline::OfflineInput`] — the property
//! [`crate::extract::Extraction::diarize`] relies on.
//!
//! Two `OfflineInput` fields are deliberately NOT surfaced:
//! - `smoothing_epsilon: Option<f32>` — its only documented non-default use is
//!   dia's own `OwnedDiarizationPipeline` audio entry point, which speakerkit
//!   does not use (speakerkit feeds the tensor set directly). Its default,
//!   `None`, is "bit-exact pyannote argmax", the only value meaningful for the
//!   direct-tensor path, so exposing it would add a knob with exactly one
//!   sensible setting.
//! - `spill_options` — a memory/spill BACKEND configuration (mmap threshold, a
//!   temp-dir `PathBuf`), not a clustering hyperparameter; surfacing an I/O path
//!   here would also cut against this crate's sans-I/O config surface. dia's
//!   default is used unchanged.

#[cfg(test)]
mod tests;

/// Default [`OfflineOptions::threshold`] — dia's community-1 AHC linkage
/// threshold. Matches [`dia::offline::OfflineInput`]'s `threshold` default
/// (`diarization/src/offline/algo.rs`, `OfflineInput::new`'s community-1
/// block), which is pyannote-community-1's `clustering.threshold`.
pub const DEFAULT_THRESHOLD: f64 = 0.6;

/// Default [`OfflineOptions::fa`] — dia's community-1 VBx `Fa`. Matches
/// [`dia::offline::OfflineInput`]'s `fa` default (`OfflineInput::new`).
pub const DEFAULT_FA: f64 = 0.07;

/// Default [`OfflineOptions::fb`] — dia's community-1 VBx `Fb`. Matches
/// [`dia::offline::OfflineInput`]'s `fb` default (`OfflineInput::new`).
pub const DEFAULT_FB: f64 = 0.8;

/// Default [`OfflineOptions::max_iters`] — dia's community-1 VBx
/// max-iterations cap. Matches [`dia::offline::OfflineInput`]'s `max_iters`
/// default (`OfflineInput::new`).
pub const DEFAULT_MAX_ITERS: usize = 20;

/// Default [`OfflineOptions::min_duration_off`] — dia's community-1 gap-merging
/// threshold for span post-processing. Matches
/// [`dia::offline::OfflineInput`]'s `min_duration_off` default
/// (`OfflineInput::new`).
pub const DEFAULT_MIN_DURATION_OFF: f64 = 0.0;

#[cfg(feature = "serde")]
fn default_threshold() -> f64 {
  DEFAULT_THRESHOLD
}
#[cfg(feature = "serde")]
fn default_fa() -> f64 {
  DEFAULT_FA
}
#[cfg(feature = "serde")]
fn default_fb() -> f64 {
  DEFAULT_FB
}
#[cfg(feature = "serde")]
fn default_max_iters() -> usize {
  DEFAULT_MAX_ITERS
}
#[cfg(feature = "serde")]
fn default_min_duration_off() -> f64 {
  DEFAULT_MIN_DURATION_OFF
}

// ---------------------------------------------------------------------
// Non-finite float rejection at the serde boundary (whisperkit round-3 F6)
// ---------------------------------------------------------------------

/// The error a non-finite `f64` raises on either side of the `serde` boundary.
///
/// `serde_json` has no JSON form for `NaN`/±∞ and silently writes `null` for
/// each, so a non-finite hyperparameter would serialize to a lossy `null` and
/// (for [`OfflineOptions::min_duration_off`], whose `serde(default)` reads a
/// missing/`null` field as the default) could silently round-trip to a
/// DIFFERENT value. Rejecting non-finite on both sides keeps the round trip
/// lossless — the whisperkit round-3 F6 lesson, applied to this crate's f64
/// knobs.
#[cfg(feature = "serde")]
const NON_FINITE_FLOAT_MSG: &str = "non-finite float (NaN or infinity) is not representable in \
  JSON and is rejected to keep the serde round trip lossless";

/// The error a non-finite OR negative `f64` raises at the `serde` boundary for
/// [`OfflineOptions::min_duration_off`], whose dia consumer
/// ([`dia::offline::OfflineInput::with_min_duration_off`]) PANICS on a
/// non-finite or negative value. Rejecting the same predicate here (and in the
/// builder) means no `OfflineOptions` value — however constructed, serde
/// included — can drive dia into that panic.
#[cfg(feature = "serde")]
const NEGATIVE_OR_NON_FINITE_MSG: &str = "min_duration_off must be a finite, non-negative float \
  (seconds); NaN, infinity, and negative values are rejected";

/// `serde` bridge for an `f64` hyperparameter that must round-trip losslessly:
/// a non-finite value is refused on BOTH serialize (so the lossy `null` is
/// never produced) and deserialize. In-memory construction is deliberately NOT
/// guarded for these knobs — dia's own [`dia::offline::OfflineInput`] `with_*`
/// builders for `threshold`/`fa`/`fb` accept any `f64` unchecked, and
/// [`OfflineOptions`] mirrors that contract; only the serde wire boundary is
/// hardened. See [`NON_FINITE_FLOAT_MSG`].
#[cfg(feature = "serde")]
pub(crate) mod finite_f64 {
  use serde::{Deserialize, Deserializer, Serialize, Serializer};

  pub(crate) fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if !value.is_finite() {
      return Err(serde::ser::Error::custom(super::NON_FINITE_FLOAT_MSG));
    }
    value.serialize(serializer)
  }

  pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let value = f64::deserialize(deserializer)?;
    if !value.is_finite() {
      return Err(serde::de::Error::custom(super::NON_FINITE_FLOAT_MSG));
    }
    Ok(value)
  }
}

/// `serde` bridge for [`OfflineOptions::min_duration_off`]: refuses a
/// non-finite OR negative value on both sides of the boundary — the exact
/// predicate dia's [`dia::offline::OfflineInput::with_min_duration_off`]
/// asserts (`check_min_duration_off`, see [`super::check_min_duration_off`]),
/// so a serde-deserialized `OfflineOptions` (which bypasses the panicking
/// builder) can never carry a value that would later panic dia. See
/// [`NEGATIVE_OR_NON_FINITE_MSG`].
#[cfg(feature = "serde")]
pub(crate) mod finite_nonneg_f64 {
  use serde::{Deserialize, Deserializer, Serialize, Serializer};

  pub(crate) fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if !super::check_min_duration_off(*value) {
      return Err(serde::ser::Error::custom(super::NEGATIVE_OR_NON_FINITE_MSG));
    }
    value.serialize(serializer)
  }

  pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let value = f64::deserialize(deserializer)?;
    if !super::check_min_duration_off(value) {
      return Err(serde::de::Error::custom(super::NEGATIVE_OR_NON_FINITE_MSG));
    }
    Ok(value)
  }
}

/// dia's `min_duration_off` validity predicate: finite and `>= 0`. Exact copy
/// of dia's `check_min_duration_off`
/// (`diarization/src/offline/algo.rs`), including its const-fn-safe NaN check —
/// `f64::is_finite` is not yet usable in a `const fn` at this crate's MSRV (the
/// same reason [`crate::window`]'s `check_onset` is hand-rolled), so the check
/// is phrased via the `v != v` NaN idiom plus a direct `+∞` rejection.
/// `v >= 0.0` already rejects `-∞` and every negative.
#[inline]
const fn check_min_duration_off(v: f64) -> bool {
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && v >= 0.0 && v != f64::INFINITY
}

/// Hyperparameters for the offline pyannote-community-1 clustering pipeline —
/// the payload of [`ClusterBackend::Offline`]. Mirrors, field-for-field, the
/// five community-1 knobs [`dia::offline::OfflineInput`] exposes; see this
/// module's own doc for the full correspondence, every default's dia citation,
/// and the two `OfflineInput` fields deliberately not surfaced.
///
/// Composed per the rust-options-pattern: [`Self::new`] (a `const fn` equal to
/// [`Default`]) is the single source of the defaults, with a
/// getter / `with_*` builder / `set_*` in-place setter per knob. No `Eq`: the
/// four `f64` knobs make it unsound, exactly as [`crate::window::WindowOptions`]
/// (whose `f32 onset` is the same story) forgoes `Eq`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OfflineOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_threshold", with = "finite_f64")
  )]
  threshold: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_fa", with = "finite_f64"))]
  fa: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_fb", with = "finite_f64"))]
  fb: f64,
  #[cfg_attr(feature = "serde", serde(default = "default_max_iters"))]
  max_iters: usize,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_min_duration_off", with = "finite_nonneg_f64")
  )]
  min_duration_off: f64,
}

impl Default for OfflineOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl OfflineOptions {
  /// Options matching dia's community-1 defaults: [`DEFAULT_THRESHOLD`] (0.6),
  /// [`DEFAULT_FA`] (0.07), [`DEFAULT_FB`] (0.8), [`DEFAULT_MAX_ITERS`] (20),
  /// and [`DEFAULT_MIN_DURATION_OFF`] (0.0) — each equal to
  /// [`dia::offline::OfflineInput`]'s own default for the same knob.
  pub const fn new() -> Self {
    Self {
      threshold: DEFAULT_THRESHOLD,
      fa: DEFAULT_FA,
      fb: DEFAULT_FB,
      max_iters: DEFAULT_MAX_ITERS,
      min_duration_off: DEFAULT_MIN_DURATION_OFF,
    }
  }

  /// The AHC linkage threshold. Fed to
  /// [`dia::offline::OfflineInput::with_threshold`].
  #[inline(always)]
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }
  /// The VBx `Fa` hyperparameter. Fed to
  /// [`dia::offline::OfflineInput::with_fa`].
  #[inline(always)]
  pub const fn fa(&self) -> f64 {
    self.fa
  }
  /// The VBx `Fb` hyperparameter. Fed to
  /// [`dia::offline::OfflineInput::with_fb`].
  #[inline(always)]
  pub const fn fb(&self) -> f64 {
    self.fb
  }
  /// The VBx max-iterations cap. Fed to
  /// [`dia::offline::OfflineInput::with_max_iters`].
  #[inline(always)]
  pub const fn max_iters(&self) -> usize {
    self.max_iters
  }
  /// The gap-merging threshold (seconds) for span post-processing. Fed to
  /// [`dia::offline::OfflineInput::with_min_duration_off`].
  #[inline(always)]
  pub const fn min_duration_off(&self) -> f64 {
    self.min_duration_off
  }

  /// Builder form of [`Self::set_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_threshold(mut self, threshold: f64) -> Self {
    self.set_threshold(threshold);
    self
  }
  /// Sets [`Self::threshold`] in place. Unchecked, mirroring dia's own
  /// [`dia::offline::OfflineInput::with_threshold`] (which range-checks
  /// nothing); a non-finite value is refused only at the serde boundary (the
  /// crate-private `finite_f64` serde helper).
  #[inline(always)]
  pub const fn set_threshold(&mut self, threshold: f64) -> &mut Self {
    self.threshold = threshold;
    self
  }
  /// Builder form of [`Self::set_fa`].
  #[must_use]
  #[inline(always)]
  pub const fn with_fa(mut self, fa: f64) -> Self {
    self.set_fa(fa);
    self
  }
  /// Sets [`Self::fa`] in place. Unchecked, mirroring dia's own
  /// [`dia::offline::OfflineInput::with_fa`].
  #[inline(always)]
  pub const fn set_fa(&mut self, fa: f64) -> &mut Self {
    self.fa = fa;
    self
  }
  /// Builder form of [`Self::set_fb`].
  #[must_use]
  #[inline(always)]
  pub const fn with_fb(mut self, fb: f64) -> Self {
    self.set_fb(fb);
    self
  }
  /// Sets [`Self::fb`] in place. Unchecked, mirroring dia's own
  /// [`dia::offline::OfflineInput::with_fb`].
  #[inline(always)]
  pub const fn set_fb(&mut self, fb: f64) -> &mut Self {
    self.fb = fb;
    self
  }
  /// Builder form of [`Self::set_max_iters`].
  #[must_use]
  #[inline(always)]
  pub const fn with_max_iters(mut self, max_iters: usize) -> Self {
    self.set_max_iters(max_iters);
    self
  }
  /// Sets [`Self::max_iters`] in place.
  #[inline(always)]
  pub const fn set_max_iters(&mut self, max_iters: usize) -> &mut Self {
    self.max_iters = max_iters;
    self
  }
  /// Builder form of [`Self::set_min_duration_off`].
  ///
  /// # Panics
  /// As [`Self::set_min_duration_off`].
  #[must_use]
  #[inline(always)]
  pub const fn with_min_duration_off(mut self, min_duration_off: f64) -> Self {
    self.set_min_duration_off(min_duration_off);
    self
  }
  /// Sets [`Self::min_duration_off`] in place.
  ///
  /// # Panics
  /// Panics if `min_duration_off` is NaN, `±∞`, or negative — mirroring dia's
  /// [`dia::offline::OfflineInput::with_min_duration_off`]
  /// (`diarization/src/offline/algo.rs`), which asserts the identical
  /// predicate (this module's `check_min_duration_off`): RTTM span-merge
  /// consumes this as a non-negative seconds quantity, and `+∞` merges every
  /// same-cluster gap while `NaN` silently disables the merge. The serde
  /// boundary rejects the same values (the crate-private `finite_nonneg_f64`
  /// serde helper), so no `OfflineOptions` ever reaches dia's assert.
  #[inline(always)]
  pub const fn set_min_duration_off(&mut self, min_duration_off: f64) -> &mut Self {
    assert!(
      check_min_duration_off(min_duration_off),
      "min_duration_off must be finite and >= 0"
    );
    self.min_duration_off = min_duration_off;
    self
  }

  /// Apply these five hyperparameters onto a [`dia::offline::OfflineInput`],
  /// returning the tuned input ready for [`dia::offline::diarize_offline`].
  ///
  /// The SINGLE place [`OfflineOptions`] maps onto dia's `OfflineInput`
  /// hyperparameter fields — one `with_*` builder per knob, in field order.
  /// [`crate::extract::Extraction::diarize_with`] calls this over
  /// [`crate::extract::Extraction::into_offline_input`]; the
  /// `apply_to_maps_each_knob_to_its_dia_field` test pins each knob to its dia
  /// field so a swapped mapping fails.
  ///
  /// With [`Self::default`] every applied value equals dia's own default (see
  /// this module's `defaults_equal_dia` pin), so the returned input is
  /// field-identical to the untouched one — the no-op
  /// [`crate::extract::Extraction::diarize`] relies on for byte-identical
  /// default clustering.
  ///
  /// Cannot panic dia's `with_min_duration_off`: [`Self::min_duration_off`] is
  /// finite and `>= 0` for every `OfflineOptions` (rejected at both the builder
  /// and the serde boundary), so the assert it re-checks always holds.
  #[must_use]
  pub(crate) fn apply_to<'a>(
    &self,
    input: dia::offline::OfflineInput<'a>,
  ) -> dia::offline::OfflineInput<'a> {
    input
      .with_threshold(self.threshold)
      .with_fa(self.fa)
      .with_fb(self.fb)
      .with_max_iters(self.max_iters)
      .with_min_duration_off(self.min_duration_off)
  }
}

/// Emits [`ClusterBackend`], its [`ClusterBackend::as_str`], and its
/// [`FromStr`](core::str::FromStr) parser from ONE table of
/// `Variant(Payload) => "spelling"` rows — the workspace golden-enum contract
/// with the alignkit round-4 lesson baked in: the roster (enum variants), the
/// spellings (`as_str`), and the parser (`FromStr`) cannot drift apart because
/// they are all generated from the same rows, and a row is a compile ERROR
/// unless it carries all three of variant, payload, and spelling — so adding a
/// variant with an incomplete mapping fails to compile at this macro, never
/// merely goes missing from a hand-maintained list.
///
/// The discriminant is what round-trips through `as_str`/`FromStr`/`Display`
/// (and the serde tag); each variant's payload is configured separately.
/// `FromStr` yields the named variant with a DEFAULT payload — the string form
/// selects an engine, not its tuning.
macro_rules! define_cluster_backend {
  (
    $( #[$enum_meta:meta] )*
    pub enum ClusterBackend {
      $(
        $( #[$variant_meta:meta] )*
        $variant:ident ( $payload:ty ) => $spelling:literal
      ),+ $(,)?
    }
  ) => {
    $( #[$enum_meta] )*
    pub enum ClusterBackend {
      $(
        $( #[$variant_meta] )*
        $variant($payload),
      )+
    }

    impl ClusterBackend {
      /// The stable snake_case name of this backend's discriminant, ignoring
      /// its payload. Inverse of [`FromStr`](core::str::FromStr), and equal to
      /// both the [`Display`](core::fmt::Display) rendering and the serde
      /// discriminant tag.
      #[inline(always)]
      pub const fn as_str(&self) -> &'static str {
        match self {
          $( Self::$variant(..) => $spelling, )+
        }
      }
    }

    impl core::str::FromStr for ClusterBackend {
      type Err = ParseClusterBackendError;

      /// Parses a backend by its [`ClusterBackend::as_str`] discriminant name,
      /// yielding that variant with a DEFAULT payload. Unknown names return the
      /// opaque [`ParseClusterBackendError`].
      fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
          $( $spelling => Self::$variant(<$payload>::default()), )+
          _ => return Err(ParseClusterBackendError(())),
        })
      }
    }

    /// The single-table roster of every [`ClusterBackend`] discriminant
    /// spelling, generated alongside the enum/`as_str`/`FromStr` so the
    /// golden-enum contract test iterates the same source of truth (adding a
    /// variant extends the test automatically).
    #[cfg(test)]
    pub(crate) const CLUSTER_BACKEND_SPELLINGS: &[&str] = &[ $( $spelling ),+ ];
  };
}

define_cluster_backend! {
  /// The runtime clustering engine selection (design spec §Architecture) — the
  /// backend [`crate::extract::Extraction::diarize_with`] runs. `#[non_exhaustive]`
  /// because a second engine (streaming/online) is planned; match it with a
  /// wildcard-free arm and the compiler will force that variant on you when it
  /// lands.
  ///
  /// Golden-enum contract (workspace convention): stable snake_case
  /// [`Self::as_str`], derived [`Display`](core::fmt::Display), total
  /// [`FromStr`](core::str::FromStr) with the opaque [`ParseClusterBackendError`],
  /// and (behind `serde`) `rename_all = "snake_case"` on the discriminant tag —
  /// all generated from one table (the crate-private `define_cluster_backend!`
  /// macro).
  #[derive(Debug, Clone, Copy, PartialEq, derive_more::Display)]
  #[display("{}", self.as_str())]
  #[non_exhaustive]
  #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
  #[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
  pub enum ClusterBackend {
    /// dia's pyannote-community-1 offline pipeline
    /// ([`dia::offline::diarize_offline`]), tuned by [`OfflineOptions`]. The
    /// default backend, and the one every DER parity gate drives.
    Offline(OfflineOptions) => "offline",
  }
}

impl Default for ClusterBackend {
  /// [`ClusterBackend::Offline`] with default [`OfflineOptions`] — dia's
  /// community-1 hyperparameters, i.e. byte-identical clustering to a bare
  /// [`dia::offline::OfflineInput`]. This is the backend
  /// [`crate::extract::Extraction::diarize`] uses.
  fn default() -> Self {
    Self::Offline(OfflineOptions::new())
  }
}

/// Error parsing a [`ClusterBackend`] discriminant name (opaque, per the
/// workspace golden-enum convention — mirrors `ParseTaskError` /
/// `ParseComputeUnitsError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown cluster backend name")]
pub struct ParseClusterBackendError(());
