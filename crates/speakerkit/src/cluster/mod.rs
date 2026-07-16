//! The runtime clustering config surface: [`ClusterBackend`] and its
//! [`Offline`](ClusterBackend::Offline) hyperparameters ([`OfflineOptions`]).
//!
//! [`crate::extract::Extraction::diarize_with`] takes a [`ClusterBackend`] and
//! runs the named engine over an [`crate::extract::Extraction`];
//! [`crate::extract::Extraction::diarize`] is the same thing at the default
//! backend ([`ClusterBackend::default`]), the path every parity harness drives.
//!
//! # Two engines: `Offline` and `Online`
//!
//! [`ClusterBackend`] is `#[non_exhaustive]` and carries two variants:
//! - [`Offline`](ClusterBackend::Offline) wraps dia's pyannote-community-1
//!   offline pipeline ([`dia::offline::diarize_offline`], AHC→VBx over
//!   PLDA-projected embeddings) — the default, and the backend every DER parity
//!   gate drives. Tuned by [`OfflineOptions`].
//! - [`Online`](ClusterBackend::Online) wraps FluidAudio's greedy online
//!   centroid matcher ([`dia::cluster::online::OnlineClusterer`]) — a genuinely
//!   DIFFERENT algorithm class (streaming assign-as-you-go, not AHC→VBx):
//!   order-dependent by design, matched on RAW cosine embeddings with NO PLDA,
//!   and gated against FluidAudio's Swift `SpeakerManager` rather than pyannote
//!   DER. Tuned by [`OnlineOptions`]; run via
//!   [`crate::extract::Extraction::diarize_online`]. `#[non_exhaustive]`
//!   remains for a future third engine — there is no stub variant (honest
//!   surface: what compiles, runs).
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
//!   corpus.
//!
//! Its `OfflineClusterOptions`/`OfflineMethod`/`Linkage` vocabulary is
//! deliberately NOT part of speakerkit's surface. T1 briefly re-exported those
//! three types at the crate root, expecting [`ClusterBackend::Offline`] to wrap
//! them; T2 removed that re-export once T1 discovered the runtime path drives the
//! OTHER entry point ([`dia::offline::diarize_offline`], above) — surfacing the
//! batch clusterer's vocabulary as speakerkit's clustering surface would have
//! been misleading for an unpublished crate whose only validated offline path is
//! the pipeline (design spec AMENDMENT 2026-07-16). A caller who genuinely wants
//! dia's batch clusterer can still reach it through the `dia` dependency
//! directly; a first-class batch mode, if ever wanted, would arrive as its own
//! [`ClusterBackend`] variant with its own gates.
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
//!
//! # Example: selecting and tuning a backend
//!
//! The config surface is pure and model-free — construct and inspect it without
//! loading any model:
//!
//! ```
//! use speakerkit::{ClusterBackend, OfflineOptions, OnlineOptions};
//!
//! // The default backend is the offline pyannote-community-1 pipeline with
//! // dia's community-1 hyperparameters.
//! assert_eq!(
//!   ClusterBackend::default(),
//!   ClusterBackend::Offline(OfflineOptions::default()),
//! );
//! assert_eq!(ClusterBackend::default().as_str(), "offline");
//!
//! // Tune the offline knobs (every default already equals dia's = pyannote's).
//! let offline = OfflineOptions::default().with_threshold(0.7);
//! assert_eq!(offline.threshold(), 0.7);
//!
//! // Select the online engine — raw cosine, NO PLDA, order-dependent by design.
//! let online = ClusterBackend::Online(OnlineOptions::from_clustering_threshold(0.7));
//! assert_eq!(online.as_str(), "online");
//! ```

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

// =====================================================================
// Online engine options — FluidAudio SpeakerManager knobs, ported by dia's
// `cluster::online::OnlineClusterOptions`. Mirror dia's contract 1:1 (defaults,
// range predicates), then harden the serde boundary the same way OfflineOptions
// does, so no OnlineOptions can drive dia's validating setters into a panic.
// =====================================================================

/// Default [`OnlineOptions::speaker_threshold`] — the assignment cosine
/// DISTANCE a bare FluidAudio `SpeakerManager()` uses. Equals dia's
/// [`dia::cluster::online::DEFAULT_SPEAKER_THRESHOLD`] (0.65), which ports
/// `Clustering/SpeakerManager.swift:46`. A distance in `[0.0, 2.0]` (0
/// identical … 2 antipodal), NOT a similarity.
pub const DEFAULT_SPEAKER_THRESHOLD: f32 = 0.65;

/// Default [`OnlineOptions::embedding_threshold`] — the centroid-update cosine
/// distance a bare `SpeakerManager()` uses. Equals dia's
/// [`dia::cluster::online::DEFAULT_EMBEDDING_THRESHOLD`] (0.45)
/// (`Clustering/SpeakerManager.swift:47`).
pub const DEFAULT_EMBEDDING_THRESHOLD: f32 = 0.45;

/// Default [`OnlineOptions::min_speech_duration`] (seconds) — the minimum
/// segment length to spawn a new speaker a bare `SpeakerManager()` uses. Equals
/// dia's [`dia::cluster::online::DEFAULT_MIN_SPEECH_DURATION`] (1.0)
/// (`Clustering/SpeakerManager.swift:48`).
pub const DEFAULT_MIN_SPEECH_DURATION: f32 = 1.0;

#[cfg(feature = "serde")]
fn default_speaker_threshold() -> f32 {
  DEFAULT_SPEAKER_THRESHOLD
}
#[cfg(feature = "serde")]
fn default_embedding_threshold() -> f32 {
  DEFAULT_EMBEDDING_THRESHOLD
}
#[cfg(feature = "serde")]
fn default_min_speech_duration() -> f32 {
  DEFAULT_MIN_SPEECH_DURATION
}

/// dia's `validate_threshold` predicate as a `const fn`: a finite cosine
/// distance in `[0.0, 2.0]`. `v >= 0.0` rejects NaN and `-∞`; `v <= 2.0`
/// rejects NaN, `+∞`, and any value past `cosine_distance`'s codomain — so no
/// separate NaN clause is needed (unlike [`check_min_speech_duration`], whose
/// upper bound is `+∞`). Matches the range dia's
/// [`dia::cluster::online::OnlineClusterOptions`] threshold setters assert
/// (`diarization/src/cluster/online/options.rs`), so a value passing this
/// cannot panic dia's setter in [`OnlineOptions::to_dia_options`].
#[inline]
#[allow(clippy::manual_range_contains)] // const fn: RangeInclusive::contains is not const at MSRV.
const fn check_online_threshold(v: f32) -> bool {
  v >= 0.0 && v <= 2.0
}

/// dia's `validate_duration` predicate as a `const fn`: a finite, non-negative
/// number of seconds. `v >= 0.0` rejects NaN and `-∞`; the explicit
/// `!= f32::INFINITY` rejects `+∞` — hand-rolled with the `v != v` NaN idiom
/// because `f32::is_finite` is not usable in a `const fn` at this crate's MSRV
/// (the same reason [`check_min_duration_off`] is hand-rolled).
#[inline]
const fn check_min_speech_duration(v: f32) -> bool {
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && v >= 0.0 && v != f32::INFINITY
}

/// The error a non-finite / out-of-range online threshold raises at the `serde`
/// boundary. dia's threshold setters PANIC outside `[0.0, 2.0]`; rejecting the
/// same predicate here (and in the builder) means no serde-deserialized
/// `OnlineOptions` can later panic dia in [`OnlineOptions::to_dia_options`].
#[cfg(feature = "serde")]
const ONLINE_THRESHOLD_MSG: &str = "online cluster threshold must be a finite cosine distance in \
  [0.0, 2.0]; NaN, infinity, and out-of-range values are rejected";

/// The error a non-finite / negative `min_speech_duration` raises at the
/// `serde` boundary — the predicate dia's duration setter asserts.
#[cfg(feature = "serde")]
const ONLINE_DURATION_MSG: &str = "min_speech_duration must be a finite, non-negative float \
  (seconds); NaN, infinity, and negative values are rejected";

/// `serde` bridge for the two [`OnlineOptions`] cosine-distance thresholds:
/// refuses a value outside a finite `[0.0, 2.0]` on BOTH sides of the boundary,
/// mirroring [`finite_nonneg_f64`]. See [`ONLINE_THRESHOLD_MSG`].
#[cfg(feature = "serde")]
pub(crate) mod finite_threshold_f32 {
  use serde::{Deserialize, Deserializer, Serialize, Serializer};

  pub(crate) fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
    if !super::check_online_threshold(*value) {
      return Err(serde::ser::Error::custom(super::ONLINE_THRESHOLD_MSG));
    }
    value.serialize(serializer)
  }

  pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    let value = f32::deserialize(deserializer)?;
    if !super::check_online_threshold(value) {
      return Err(serde::de::Error::custom(super::ONLINE_THRESHOLD_MSG));
    }
    Ok(value)
  }
}

/// `serde` bridge for [`OnlineOptions::min_speech_duration`]: refuses a
/// non-finite OR negative value on both sides — the exact predicate dia's
/// duration setter asserts. See [`ONLINE_DURATION_MSG`].
#[cfg(feature = "serde")]
pub(crate) mod finite_nonneg_f32 {
  use serde::{Deserialize, Deserializer, Serialize, Serializer};

  pub(crate) fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
    if !super::check_min_speech_duration(*value) {
      return Err(serde::ser::Error::custom(super::ONLINE_DURATION_MSG));
    }
    value.serialize(serializer)
  }

  pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    let value = f32::deserialize(deserializer)?;
    if !super::check_min_speech_duration(value) {
      return Err(serde::de::Error::custom(super::ONLINE_DURATION_MSG));
    }
    Ok(value)
  }
}

/// Hyperparameters for the online (streaming) greedy centroid clusterer — the
/// payload of [`ClusterBackend::Online`]. Mirrors, field-for-field, the three
/// knobs FluidAudio's `SpeakerManager` assignment path consults, exactly as
/// dia's [`dia::cluster::online::OnlineClusterOptions`] ports them; every
/// default equals dia's, which equals FluidAudio's bare `SpeakerManager()` (the
/// `cluster::online_defaults_equal_dia` pin reads dia's OWN
/// `OnlineClusterOptions::default` accessors, so a drift on EITHER side fails to
/// compile the assertion).
///
/// # Cosine space, no PLDA
/// The online engine matches RAW L2-normalized WeSpeaker embeddings by cosine
/// distance; the PLDA projection the offline pipeline applies has NO part here
/// (design spec §Architecture point 3; dia's `cluster::online` module doc; T4's
/// semantics table). The thresholds are therefore cosine DISTANCES in
/// `[0.0, 2.0]`, and [`crate::extract::Extraction::diarize_online`] takes NO
/// `plda` argument.
///
/// # The two thresholds gate different decisions
/// - [`speaker_threshold`](Self::speaker_threshold): assignment — reuse the
///   nearest existing speaker vs. spawn a new one (strict `<`).
/// - [`embedding_threshold`](Self::embedding_threshold): centroid update —
///   whether an assigned segment folds into the speaker's running centroid.
///
/// Because the update threshold is (by default) the smaller, there is a band
/// `[embedding_threshold, speaker_threshold)` where a segment is assigned but
/// does not move its centroid.
///
/// Composed per the rust-options-pattern: [`Self::new`] (a `const fn` equal to
/// [`Default`]) is the single source of the defaults, with a getter / `with_*`
/// builder / `set_*` in-place setter per knob. Unlike [`OfflineOptions`]'s
/// unchecked `threshold`/`fa`/`fb` (which mirror dia's unchecked `OfflineInput`
/// setters), ALL three online setters panic-validate — because dia's
/// `OnlineClusterOptions` setters do, and [`Self::to_dia_options`] drives them.
/// No `Eq`: the three `f32` knobs make it unsound, exactly as [`OfflineOptions`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OnlineOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_speaker_threshold", with = "finite_threshold_f32")
  )]
  speaker_threshold: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_embedding_threshold", with = "finite_threshold_f32")
  )]
  embedding_threshold: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_min_speech_duration", with = "finite_nonneg_f32")
  )]
  min_speech_duration: f32,
}

impl Default for OnlineOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl OnlineOptions {
  /// Options matching dia's / FluidAudio's bare `SpeakerManager()` defaults:
  /// [`DEFAULT_SPEAKER_THRESHOLD`] (0.65), [`DEFAULT_EMBEDDING_THRESHOLD`]
  /// (0.45), and [`DEFAULT_MIN_SPEECH_DURATION`] (1.0) — each equal to
  /// [`dia::cluster::online::OnlineClusterOptions`]'s own default for the same
  /// knob.
  ///
  /// This is NOT the production `DiarizerManager` wiring, which derives the
  /// thresholds from `clusteringThreshold = 0.7` as `0.84` / `0.56` — reproduce
  /// that with [`Self::from_clustering_threshold`].
  pub const fn new() -> Self {
    Self {
      speaker_threshold: DEFAULT_SPEAKER_THRESHOLD,
      embedding_threshold: DEFAULT_EMBEDDING_THRESHOLD,
      min_speech_duration: DEFAULT_MIN_SPEECH_DURATION,
    }
  }

  /// Construct the way production `DiarizerManager` does
  /// (`Core/DiarizerManager.swift:29,32`, via dia's
  /// [`dia::cluster::online::OnlineClusterOptions::from_clustering_threshold`]):
  /// from a single base `clusteringThreshold`, deriving `speaker_threshold =
  /// base × 1.2` and `embedding_threshold = base × 0.8`. `min_speech_duration`
  /// keeps its default. Passing `base = 0.7` reproduces the shipping FluidAudio
  /// diarizer's thresholds (`0.84` / `0.56`).
  ///
  /// # Panics
  /// Panics if either derived threshold is non-finite or outside `[0.0, 2.0]`
  /// (e.g. `base > 1.666…` overflows `speaker_threshold` past `2.0`) — the same
  /// predicate dia asserts.
  #[must_use]
  pub const fn from_clustering_threshold(base: f32) -> Self {
    Self::new()
      .with_speaker_threshold(base * 1.2)
      .with_embedding_threshold(base * 0.8)
  }

  /// The assignment cosine-distance threshold. Fed to
  /// [`dia::cluster::online::OnlineClusterOptions::with_speaker_threshold`] by
  /// [`Self::to_dia_options`].
  #[inline(always)]
  pub const fn speaker_threshold(&self) -> f32 {
    self.speaker_threshold
  }
  /// The centroid-update cosine-distance threshold. Fed to
  /// [`dia::cluster::online::OnlineClusterOptions::with_embedding_threshold`].
  #[inline(always)]
  pub const fn embedding_threshold(&self) -> f32 {
    self.embedding_threshold
  }
  /// The minimum new-speaker speech duration (seconds). Fed to
  /// [`dia::cluster::online::OnlineClusterOptions::with_min_speech_duration`].
  #[inline(always)]
  pub const fn min_speech_duration(&self) -> f32 {
    self.min_speech_duration
  }

  /// Builder form of [`Self::set_speaker_threshold`].
  ///
  /// # Panics
  /// As [`Self::set_speaker_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_speaker_threshold(mut self, speaker_threshold: f32) -> Self {
    self.set_speaker_threshold(speaker_threshold);
    self
  }
  /// Sets [`Self::speaker_threshold`] in place.
  ///
  /// # Panics
  /// Panics if `speaker_threshold` is NaN, `±∞`, or outside `[0.0, 2.0]` —
  /// mirroring dia's [`dia::cluster::online::OnlineClusterOptions`] threshold
  /// setter (a cosine distance has codomain `[0.0, 2.0]`). The serde boundary
  /// rejects the same values (the crate-private `finite_threshold_f32` helper),
  /// so no `OnlineOptions` ever reaches dia's assert.
  #[inline(always)]
  pub const fn set_speaker_threshold(&mut self, speaker_threshold: f32) -> &mut Self {
    assert!(
      check_online_threshold(speaker_threshold),
      "speaker_threshold must be a finite cosine distance in [0.0, 2.0]"
    );
    self.speaker_threshold = speaker_threshold;
    self
  }
  /// Builder form of [`Self::set_embedding_threshold`].
  ///
  /// # Panics
  /// As [`Self::set_embedding_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_embedding_threshold(mut self, embedding_threshold: f32) -> Self {
    self.set_embedding_threshold(embedding_threshold);
    self
  }
  /// Sets [`Self::embedding_threshold`] in place.
  ///
  /// # Panics
  /// Panics if `embedding_threshold` is NaN, `±∞`, or outside `[0.0, 2.0]` — as
  /// [`Self::set_speaker_threshold`].
  #[inline(always)]
  pub const fn set_embedding_threshold(&mut self, embedding_threshold: f32) -> &mut Self {
    assert!(
      check_online_threshold(embedding_threshold),
      "embedding_threshold must be a finite cosine distance in [0.0, 2.0]"
    );
    self.embedding_threshold = embedding_threshold;
    self
  }
  /// Builder form of [`Self::set_min_speech_duration`].
  ///
  /// # Panics
  /// As [`Self::set_min_speech_duration`].
  #[must_use]
  #[inline(always)]
  pub const fn with_min_speech_duration(mut self, min_speech_duration: f32) -> Self {
    self.set_min_speech_duration(min_speech_duration);
    self
  }
  /// Sets [`Self::min_speech_duration`] in place.
  ///
  /// # Panics
  /// Panics if `min_speech_duration` is NaN, `±∞`, or negative — mirroring
  /// dia's duration setter. The serde boundary rejects the same values (the
  /// crate-private `finite_nonneg_f32` helper).
  #[inline(always)]
  pub const fn set_min_speech_duration(&mut self, min_speech_duration: f32) -> &mut Self {
    assert!(
      check_min_speech_duration(min_speech_duration),
      "min_speech_duration must be finite and >= 0"
    );
    self.min_speech_duration = min_speech_duration;
    self
  }

  /// Map these three knobs onto dia's
  /// [`dia::cluster::online::OnlineClusterOptions`] — the input to
  /// [`dia::cluster::online::OnlineClusterer::new`].
  ///
  /// The SINGLE place [`OnlineOptions`] maps onto dia's online options (one
  /// `with_*` builder per knob, in field order), the online analogue of
  /// [`OfflineOptions`]'s `apply_to`.
  /// [`crate::extract::Extraction::diarize_online`] builds the clusterer from
  /// this, and the out-of-crate Swift-trace oracle
  /// (`tests/parity_online_swift.rs`) drives the engine through it — so the gate
  /// exercises the REAL wiring, not a re-implementation of it. The
  /// `online_to_dia_options_maps_each_knob` test pins each knob to its dia
  /// field.
  ///
  /// Cannot panic dia's validating setters: every `OnlineOptions` field
  /// satisfies dia's predicate (finite thresholds in `[0.0, 2.0]`, finite
  /// non-negative duration), enforced at both this crate's builder and its
  /// serde boundary. With [`Self::default`] the result equals
  /// [`dia::cluster::online::OnlineClusterOptions::default`].
  #[must_use]
  pub fn to_dia_options(&self) -> dia::cluster::online::OnlineClusterOptions {
    dia::cluster::online::OnlineClusterOptions::new()
      .with_speaker_threshold(self.speaker_threshold)
      .with_embedding_threshold(self.embedding_threshold)
      .with_min_speech_duration(self.min_speech_duration)
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
  /// because a THIRD engine may yet land; match it with a wildcard-free arm and
  /// the compiler will force any future variant on you when it does.
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
    /// FluidAudio's greedy online centroid matcher
    /// ([`dia::cluster::online::OnlineClusterer`]), tuned by [`OnlineOptions`].
    /// A DIFFERENT algorithm class from [`Offline`](Self::Offline) (streaming
    /// greedy assignment, not AHC→VBx): order-dependent by design, matched on
    /// RAW cosine embeddings with NO PLDA, and gated against FluidAudio's Swift
    /// `SpeakerManager` (never pyannote DER). Run it with
    /// [`crate::extract::Extraction::diarize_online`], or via
    /// [`diarize_with`](crate::extract::Extraction::diarize_with) — which
    /// ignores its `plda` argument for this backend.
    Online(OnlineOptions) => "online",
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
