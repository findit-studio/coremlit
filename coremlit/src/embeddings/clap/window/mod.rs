//! Overlapped long-audio chunking: [`WindowPlan`] turns a clip length into a
//! list of [`Span`]s the [`AudioEncoder`](crate::embeddings::clap::AudioEncoder) embeds one at a
//! time, and [`WindowEmbedding`] pairs each resulting embedding with the [`Span`]
//! it came from (start, real length, and tail-padding-aware coverage) so an
//! [`AggregatePolicy`](crate::embeddings::clap::aggregate::AggregatePolicy) can weight by time,
//! overlap, or coverage.
//!
//! # windit engine + two clap-contract guards
//!
//! The window GEOMETRY and per-window AGGREGATION are the generic `windit`
//! windowed-sequence engine: [`Span`] is `windit::plan::Span`, [`WindowEmbedding`]
//! is `windit::windowed::WindowEmbedding<Embedding>`, and [`WindowPlan::spans`]
//! plans the head through `windit::plan::WindowPlan`. Two behaviours are clap's
//! own contract, reproduced as thin guards on top of the windit plan so the
//! pinned geometry stays bit-for-bit what it always was:
//!
//! 1. **Short clip** (`total <= WINDOW_SAMPLES`): exactly one span, whatever the
//!    hop AND tail policy — windit's `DropBelowMin` would drop the sole span of a
//!    short clip, but clap never drops a clip's only representation.
//! 2. **Multi-tail continuation**: windit stops at the first ragged tail, whereas
//!    clap's overlapped plan (`hop < window`) keeps striding, emitting
//!    progressively shorter tails until the stride passes the clip end.
//!
//! For `total > W`, windit visits starts `0, H, 2H, …` and stops at the first
//! `start` with `total − start ≤ W`, i.e. `first_tail_start = ceil((total − W)/H)·H`;
//! head spans and that first tail match clap's old loop term for term, and the
//! continuation reproduces its remaining iterations verbatim.
//!
//! # What windit does NOT replace: the mel `repeatpad`
//!
//! windit owns geometry and aggregation ONLY. The audio path still slices
//! `&samples[span.start()..span.end()]` and hands the (possibly short) slice to
//! the encoder, whose mel front-end `repeatpad`s it up to the fixed window —
//! windit's constant-right-pad helper is not used, and the mel path is untouched.
//!
//! # Wire format
//!
//! [`TailPolicy`] and [`WindowPlan`] keep their own clap-owned serde
//! representations (windit's `serde` feature is off), so the validated
//! deserialization is clap's own and was unchanged by the windit port. The
//! [`TailPolicy`] SPELLINGS did change once since: they were realigned onto the
//! adjacently tagged form windit 0.4 adopted, so a consumer holding both reads
//! one shape. See that type's "Wire form".
//!
//! The window length is **fixed** at [`WINDOW_SAMPLES`] (480 000 = 10 s at
//! 48 kHz) — the model's geometry, not a knob. The hop, the tail policy, and the
//! [`WindowPlan::max_windows`] resource cap are configurable. This module holds
//! no audio and touches no model, so its offsets and coverages are hermetically
//! pinned (see the sibling `tests.rs`).
//!
//! # Resource cap
//!
//! [`WindowPlan::spans`] counts its plan in O(1) and refuses one exceeding
//! [`WindowPlan::max_windows`] ([`DEFAULT_MAX_WINDOWS`], default-on) with a typed
//! [`Error::Windowing`]`(`[`WinditError::TooManyWindows`]`)` BEFORE materializing
//! any span — so a serde-supplied `hop_samples: 1` over a modest clip (a hop
//! every sample plans ~`total_samples` windows, gigabytes of retained embeddings
//! and one CoreML inference each) is a typed refusal, not an unbounded
//! allocation or a `.expect()` panic on the allocator's refusal.

use crate::embeddings::clap::{
  audio::TARGET_SAMPLES,
  embedding::Embedding,
  error::{Error, Result, WinditError},
};

/// windit's window span (`windit::plan::Span`), re-exported as clap's window
/// geometry unit — the half-open real range `[start, end)` a [`WindowPlan`]
/// plans and the [`AudioEncoder`](crate::embeddings::clap::AudioEncoder) embeds. Every
/// clap-produced span carries `window() == `[`WINDOW_SAMPLES`], so
/// [`Span::coverage`] is the padding-aware `real length / 480_000` fraction a
/// coverage-weighting policy uses. (`Span::new` is 3-arg — `(start, len,
/// window)` — and reports `len()`/`end()`; there is no `real_len()`.)
pub use windit::plan::Span;

/// A per-window embedding paired with the [`Span`] it was computed from — the
/// input unit to aggregation, `windit::windowed::WindowEmbedding<Embedding>`.
/// Carrying the span (and thus [`Span::coverage`]) alongside the embedding is
/// what lets a policy weight windows by time, overlap, or tail coverage. Build
/// one with [`WindowEmbedding::new`](windit::windowed::Windowed::new); read it
/// back with [`value`](windit::windowed::Windowed::value) and
/// [`span`](windit::windowed::Windowed::span).
pub type WindowEmbedding = windit::windowed::WindowEmbedding<Embedding>;

#[cfg(test)]
mod tests;

/// The fixed inference-window length in samples (480 000 = 10 s at 48 kHz).
///
/// The CLAP HTSAT graph consumes exactly this many samples per inference (via
/// the mel front-end, which `repeatpad`s a shorter tail up to it), so it is the
/// window every [`Span`] is measured against — the geometry, not a tunable
/// preference. Equal to [`crate::embeddings::clap::audio::TARGET_SAMPLES`].
pub const WINDOW_SAMPLES: usize = TARGET_SAMPLES;

/// Default [`WindowPlan::hop_samples`]: one full window (no overlap), so the
/// default plan tiles a clip into back-to-back 10 s chunks — matching textclap's
/// `ChunkingOptions` default (`window == hop == 480_000`).
pub const DEFAULT_HOP_SAMPLES: u32 = WINDOW_SAMPLES as u32;

/// Default minimum real length (samples) for [`TailPolicy::DropBelowMin`]: a
/// quarter window (120 000 = 2.5 s), matching textclap's `embed_chunked`
/// `window / 4` keep threshold.
pub const DEFAULT_TAIL_MIN_SAMPLES: u32 = (WINDOW_SAMPLES / 4) as u32;

/// Default [`WindowPlan::max_windows`]: 100 000 windows.
///
/// The cap is a resource rail, not a latency policy: each planned window costs
/// one full CoreML inference, and
/// [`AudioEncoder::embed_windows`](crate::embeddings::clap::AudioEncoder::embed_windows) retains a
/// [`Embedding`] ([`EMBEDDING_DIM`](crate::embeddings::clap::embedding::EMBEDDING_DIM) = 512 floats,
/// ~2 KiB) per window, so 100 000 caps that retention at ~200 MiB. It admits
/// every realistic clip — 24 h of audio at a 1 s hop is 86 400 windows; at the
/// default no-overlap hop the cap is ~11 days of audio — while rejecting
/// hop-abuse: at `hop_samples == 1` ANY clip long enough to window at all
/// (> 10 s) plans more than 480 000 windows and fails typed. Latency-sensitive
/// services should lower it; raising it is a deliberate opt-in to more memory
/// and inference work. Mirrors the CED classifier's identical rail.
pub const DEFAULT_MAX_WINDOWS: u32 = 100_000;

/// Drop a final chunk whose real length is below `min_samples`, so a trailing
/// sliver dominated by padding never contributes. A chunk at or above the
/// threshold is kept. The single window a clip shorter than one full window
/// produces is never dropped (there is nothing else to represent it).
///
/// A payload STRUCT, not a bare `u32` newtype: [`TailPolicy`] carries a serde
/// representation, and a struct puts the threshold on the wire under its own
/// `min_samples` name — `{"kind":"drop_below_min","value":{"min_samples":N}}`
/// — rather than as a bare integer whose meaning lives only in the variant
/// tag, and it leaves room for a second field without reshaping the document.
///
/// Payload of [`TailPolicy::DropBelowMin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DropBelowMin {
  /// The keep threshold in real samples; validated into `1..=WINDOW_SAMPLES`
  /// by [`WindowPlan`]'s checked setters and serde path.
  min_samples: u32,
}

impl DropBelowMin {
  /// Construct from the keep threshold in real samples. Unvalidated on its
  /// own: [`WindowPlan`]'s checked setters and its serde path hold the
  /// `1..=WINDOW_SAMPLES` range, exactly as they did for the struct variant's
  /// public field.
  #[inline(always)]
  pub const fn new(min_samples: u32) -> Self {
    Self { min_samples }
  }

  /// The keep threshold in real samples; validated into `1..=WINDOW_SAMPLES`
  /// by [`WindowPlan`]'s checked setters and serde path.
  #[inline(always)]
  pub const fn min_samples(&self) -> u32 {
    self.min_samples
  }
}

/// What [`WindowPlan`] does with a final chunk whose real samples fall short of a
/// full [`WINDOW_SAMPLES`] window.
///
/// A short tail is embedded by `repeatpad`ing it up to the fixed window, so a
/// kept tail's [`Span::coverage`] is `< 1.0` — the padding-aware fraction a
/// coverage-weighting policy uses to down-weight it. This policy chooses whether
/// such a tail is kept at all.
///
/// # Wire form
///
/// Under the `serde` feature this has TWO representations, chosen by
/// [`is_human_readable`](serde::Serializer::is_human_readable):
///
/// - **Human-readable** (JSON, TOML — the formats a config file is written in):
///   **adjacently tagged**, snake_case, `kind` naming the variant and `value`
///   carrying [`DropBelowMin`]'s payload — `{"kind":"pad"}` and
///   `{"kind":"drop_below_min","value":{"min_samples":N}}`. That is the form
///   `windit` 0.4 gave its own `TailPolicy`, adopted here (and by `audio::ced::window::TailPolicy`, which is feature-gated
///   separately) so a
///   consumer holding several of them reads one shape. It REPLACES the
///   externally tagged `"pad"` / `{"drop_below_min":{"min_samples":N}}` form: a
///   document written against the old spelling no longer deserializes, and
///   `tail_policy_wire_spellings_are_pinned` asserts both halves of that — the
///   new form round-trips, the old one is refused.
/// - **Binary** (postcard and every other non-self-describing format): the
///   plain enum protocol — a variant index, plus the payload for
///   [`DropBelowMin`](Self::DropBelowMin). serde's adjacent tagging cannot
///   survive such a format at all: it writes the tag as a struct FIELD and
///   reads it back through `deserialize_identifier`, which a format carrying no
///   field names refuses, so EVERY variant fails to deserialize — not only the
///   unit one, whose content serde additionally reads through
///   `deserialize_any`. Without this branch a default [`WindowPlan`] would
///   serialize to bytes it could not then read back.
///
/// Only the choice is hand-written: each branch delegates to a derived mirror
/// in `tail_policy_serde`, so the document above is still serde's own spelling
/// and this enum stays the single source of truth for the variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TailPolicy {
  /// Keep the final short chunk (any tail with ≥ 1 real sample). Its coverage is
  /// `real_len / WINDOW_SAMPLES < 1.0`; nothing is dropped, so the whole clip is
  /// covered. The default.
  #[default]
  Pad,
  /// Drop a final chunk whose real length is below `min_samples`, so a trailing
  /// sliver dominated by padding never contributes. A chunk at or above the
  /// threshold is kept. The single window a clip shorter than one full window
  /// produces is never dropped (there is nothing else to represent it).
  DropBelowMin(DropBelowMin),
}

/// The two derived mirrors of [`TailPolicy`]: its human-readable document and
/// its binary form. [`TailPolicy`]'s own impls pick between them on
/// `is_human_readable` — see that type's "Wire form" for why one representation
/// cannot serve both.
///
/// Mirrors rather than hand-written `Serialize`/`Deserialize` impls: the
/// adjacently tagged document stays serde's own spelling rather than a
/// hand-rolled imitation of it, and the wildcard-free conversions below fail to
/// compile until a new variant is spelled in both forms.
#[cfg(feature = "serde")]
mod tail_policy_serde {
  use super::{DropBelowMin, TailPolicy};

  /// JSON and TOML: `{"kind":"pad"}` /
  /// `{"kind":"drop_below_min","value":{"min_samples":N}}`.
  #[derive(serde::Serialize, serde::Deserialize)]
  #[serde(rename_all = "snake_case", tag = "kind", content = "value")]
  pub(super) enum Document {
    Pad,
    DropBelowMin(DropBelowMin),
  }

  /// postcard and friends: a variant index plus the payload, with no field
  /// name or variant string to look up.
  #[derive(serde::Serialize, serde::Deserialize)]
  pub(super) enum Binary {
    Pad,
    DropBelowMin(DropBelowMin),
  }

  impl From<TailPolicy> for Document {
    fn from(policy: TailPolicy) -> Self {
      match policy {
        TailPolicy::Pad => Self::Pad,
        TailPolicy::DropBelowMin(d) => Self::DropBelowMin(d),
      }
    }
  }

  impl From<Document> for TailPolicy {
    fn from(doc: Document) -> Self {
      match doc {
        Document::Pad => Self::Pad,
        Document::DropBelowMin(d) => Self::DropBelowMin(d),
      }
    }
  }

  impl From<TailPolicy> for Binary {
    fn from(policy: TailPolicy) -> Self {
      match policy {
        TailPolicy::Pad => Self::Pad,
        TailPolicy::DropBelowMin(d) => Self::DropBelowMin(d),
      }
    }
  }

  impl From<Binary> for TailPolicy {
    fn from(binary: Binary) -> Self {
      match binary {
        Binary::Pad => Self::Pad,
        Binary::DropBelowMin(d) => Self::DropBelowMin(d),
      }
    }
  }
}

/// The adjacently tagged document in a human-readable format, the plain enum
/// protocol in every other — see the type's "Wire form".
#[cfg(feature = "serde")]
impl serde::Serialize for TailPolicy {
  fn serialize<S: serde::Serializer>(
    &self,
    serializer: S,
  ) -> core::result::Result<S::Ok, S::Error> {
    if serializer.is_human_readable() {
      serde::Serialize::serialize(&tail_policy_serde::Document::from(*self), serializer)
    } else {
      serde::Serialize::serialize(&tail_policy_serde::Binary::from(*self), serializer)
    }
  }
}

/// The inverse of [`Serialize`](serde::Serialize) above, split on the same
/// question, so what a format wrote is what it reads back.
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TailPolicy {
  fn deserialize<D: serde::Deserializer<'de>>(
    deserializer: D,
  ) -> core::result::Result<Self, D::Error> {
    if deserializer.is_human_readable() {
      <tail_policy_serde::Document as serde::Deserialize>::deserialize(deserializer).map(Self::from)
    } else {
      <tail_policy_serde::Binary as serde::Deserialize>::deserialize(deserializer).map(Self::from)
    }
  }
}

/// Whether `hop_samples` is in the valid `1..=WINDOW_SAMPLES` range: positive
/// (a zero hop never advances) and no larger than one window (a hop past the
/// window would leave gaps of un-embedded audio between chunks). `hop ==
/// WINDOW_SAMPLES` means contiguous, non-overlapping chunks; a smaller hop
/// overlaps.
const fn check_hop_samples(v: u32) -> bool {
  v > 0 && v as usize <= WINDOW_SAMPLES
}

/// Whether a [`TailPolicy::DropBelowMin`] `min_samples` is in `1..=WINDOW_SAMPLES`
/// (a zero threshold would drop nothing yet mean "drop below one sample", and a
/// threshold above the window can never be met by a sub-window tail).
const fn check_tail(tail: TailPolicy) -> bool {
  match tail {
    TailPolicy::Pad => true,
    TailPolicy::DropBelowMin(d) => {
      d.min_samples() > 0 && d.min_samples() as usize <= WINDOW_SAMPLES
    }
  }
}

/// Whether `max_windows` is a usable cap: strictly positive. A zero cap would
/// admit no plan at all (even the single-span short clip), so a default-carrying
/// field that can never embed anything is a misconfiguration, the same class as
/// `hop == 0`. `u32::MAX` is the deliberate "effectively uncapped" escape hatch.
const fn check_max_windows(v: u32) -> bool {
  v > 0
}

/// Overlapped-chunking plan: a validated hop and tail policy over the fixed
/// [`WINDOW_SAMPLES`] window, plus a [`Self::max_windows`] resource cap
/// (rust-options-pattern).
///
/// [`Self::spans`] is the pure-geometry core — it maps a clip length to the list
/// of [`Span`]s to embed, with no audio and no model involved, so the offsets
/// and coverages are hermetically testable. `max_windows` bounds that count in
/// O(1) BEFORE any span is materialized, so an untrusted length + hop cannot
/// expand into an out-of-memory allocation or a flood of inferences.
///
/// # Validated deserialization
///
/// `Deserialize` routes through a private `WindowPlanRepr` via
/// `serde(try_from)`, holding a config-file or hand-written `WindowPlan` to the
/// SAME `hop_samples`/`min_samples`/`max_windows` invariants the checked setters
/// enforce. Deriving `Deserialize` on the fields directly would bypass
/// [`Self::set_hop_samples`]: `{"hop_samples": 0}` would deserialize and then
/// loop forever (a zero hop never advances), `{"hop_samples": 960000}` would
/// silently leave 10 s gaps of un-embedded audio between chunks, and
/// `{"max_windows": 0}` could never embed anything. Invalid input now fails to
/// deserialize instead (mirrors speakerkit's `WindowOptions`). An omitted
/// `max_windows` fills [`DEFAULT_MAX_WINDOWS`], so the cap is default-on for
/// every deserialized plan.
///
/// UNKNOWN KEYS ARE REFUSED. Defaulted fields and a tolerated stray key compose
/// into a silent hole: `{"max_window": 1}` — the plural dropped — would
/// otherwise deserialize with the typo discarded and `max_windows` filled from
/// [`DEFAULT_MAX_WINDOWS`], so a caller capping this door at ONE window would
/// get 100 000 and a misspelled RESOURCE LIMIT would silently become up to
/// 100 000 CoreML inferences; a misspelled `hop_samples` or `tail` would
/// silently change the embedded geometry the same way. The misspelling is a
/// hard error naming the key instead.
///
/// That refusal makes this type UNFLATTENABLE: serde's `deny_unknown_fields`
/// and `flatten` do not compose (a flattened field sees the outer struct's
/// other keys and rejects them), so a config type composing a plan must NEST it
/// under a key of its own — `window_plan = { … }` — not `#[serde(flatten)]` it
/// into itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "WindowPlanRepr"))]
pub struct WindowPlan {
  hop_samples: u32,
  tail: TailPolicy,
  max_windows: u32,
}

/// The plain wire form [`WindowPlan`]'s `Deserialize` deserializes FIRST
/// (carrying the field defaults), before [`WindowPlan::try_from`] applies the
/// range checks. Its whole purpose is to make the validated setters unbypassable
/// via serde — it is never constructed or exposed otherwise.
///
/// `deny_unknown_fields` lives HERE rather than on [`WindowPlan`], because this
/// is the type whose fields serde actually visits: the public plan's
/// `Deserialize` is a `try_from` wrapper around this one.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WindowPlanRepr {
  #[serde(default = "default_hop_samples")]
  hop_samples: u32,
  #[serde(default)]
  tail: TailPolicy,
  #[serde(default = "default_max_windows")]
  max_windows: u32,
}

#[cfg(feature = "serde")]
fn default_hop_samples() -> u32 {
  DEFAULT_HOP_SAMPLES
}

#[cfg(feature = "serde")]
fn default_max_windows() -> u32 {
  DEFAULT_MAX_WINDOWS
}

#[cfg(feature = "serde")]
impl TryFrom<WindowPlanRepr> for WindowPlan {
  type Error = String;

  /// Applies [`check_hop_samples`], [`check_tail`], and [`check_max_windows`] —
  /// the exact invariants [`WindowPlan::set_hop_samples`] /
  /// [`WindowPlan::set_tail_policy`] / [`WindowPlan::set_max_windows`] assert —
  /// as fallible checks, so a serde-deserialized plan can never construct the
  /// infinite-loop (`hop == 0`) or audio-skipping (`hop > window`) geometry the
  /// builders reject, nor an embed-nothing (`max_windows == 0`) cap.
  fn try_from(r: WindowPlanRepr) -> core::result::Result<Self, Self::Error> {
    if !check_hop_samples(r.hop_samples) {
      return Err(format!(
        "hop_samples ({}) must be > 0 and <= WINDOW_SAMPLES ({WINDOW_SAMPLES})",
        r.hop_samples
      ));
    }
    if !check_tail(r.tail) {
      // Interpolate the PAYLOAD, not the whole policy. `DropBelowMin`'s payload
      // struct shares its variant's name and field name, so its `Debug` is
      // exactly what the struct-shaped variant used to render — this keeps the
      // message byte-identical to the pre-newtype one instead of doubling the
      // name to `DropBelowMin(DropBelowMin { .. })`. `Pad` never fails
      // `check_tail`, so its arm exists only to keep the match total.
      let tail: &dyn core::fmt::Debug = match &r.tail {
        TailPolicy::DropBelowMin(min) => min,
        TailPolicy::Pad => &r.tail,
      };
      return Err(format!(
        "tail DropBelowMin.min_samples must be > 0 and <= WINDOW_SAMPLES ({WINDOW_SAMPLES}), got {tail:?}"
      ));
    }
    if !check_max_windows(r.max_windows) {
      return Err(format!("max_windows ({}) must be > 0", r.max_windows));
    }
    Ok(Self {
      hop_samples: r.hop_samples,
      tail: r.tail,
      max_windows: r.max_windows,
    })
  }
}

impl Default for WindowPlan {
  fn default() -> Self {
    Self::new()
  }
}

impl WindowPlan {
  /// A plan with [`DEFAULT_HOP_SAMPLES`] (no overlap), [`TailPolicy::Pad`]
  /// (keep every tail), and [`DEFAULT_MAX_WINDOWS`] (the resource cap). Tiles a
  /// clip into back-to-back 10 s windows, the last `repeatpad`-padded.
  pub const fn new() -> Self {
    Self {
      hop_samples: DEFAULT_HOP_SAMPLES,
      tail: TailPolicy::Pad,
      max_windows: DEFAULT_MAX_WINDOWS,
    }
  }

  /// Distance in samples between successive window starts. `<`
  /// [`WINDOW_SAMPLES`] means overlapping windows; `==` means contiguous.
  #[inline]
  pub const fn hop_samples(&self) -> u32 {
    self.hop_samples
  }

  /// The configured tail policy.
  #[inline]
  pub const fn tail_policy(&self) -> TailPolicy {
    self.tail
  }

  /// Builder form of [`Self::set_hop_samples`].
  ///
  /// # Panics
  /// If `hop_samples` is not in `1..=`[`WINDOW_SAMPLES`].
  #[must_use]
  pub const fn with_hop_samples(mut self, hop_samples: u32) -> Self {
    self.set_hop_samples(hop_samples);
    self
  }

  /// Sets [`Self::hop_samples`] in place.
  ///
  /// # Panics
  /// If `hop_samples` is not in `1..=`[`WINDOW_SAMPLES`] — a zero hop never
  /// advances and a hop past the window leaves gaps of un-embedded audio. The
  /// serde path reports the same violation as a deserialize error instead.
  pub const fn set_hop_samples(&mut self, hop_samples: u32) -> &mut Self {
    assert!(
      check_hop_samples(hop_samples),
      "hop_samples must be > 0 and <= WINDOW_SAMPLES (480_000)"
    );
    self.hop_samples = hop_samples;
    self
  }

  /// Builder form of [`Self::set_tail_policy`].
  ///
  /// # Panics
  /// If `tail` is [`TailPolicy::DropBelowMin`] with `min_samples` not in
  /// `1..=`[`WINDOW_SAMPLES`].
  #[must_use]
  pub const fn with_tail_policy(mut self, tail: TailPolicy) -> Self {
    self.set_tail_policy(tail);
    self
  }

  /// Sets [`Self::tail_policy`] in place.
  ///
  /// # Panics
  /// If `tail` is [`TailPolicy::DropBelowMin`] with `min_samples` not in
  /// `1..=`[`WINDOW_SAMPLES`].
  pub const fn set_tail_policy(&mut self, tail: TailPolicy) -> &mut Self {
    assert!(
      check_tail(tail),
      "TailPolicy::DropBelowMin.min_samples must be > 0 and <= WINDOW_SAMPLES (480_000)"
    );
    self.tail = tail;
    self
  }

  /// The maximum number of windows [`Self::spans`] may plan before it refuses
  /// the clip with [`WinditError::TooManyWindows`]. See [`DEFAULT_MAX_WINDOWS`].
  #[inline]
  pub const fn max_windows(&self) -> u32 {
    self.max_windows
  }

  /// Builder form of [`Self::set_max_windows`].
  ///
  /// # Panics
  /// If `max_windows` is `0`.
  #[must_use]
  pub const fn with_max_windows(mut self, max_windows: u32) -> Self {
    self.set_max_windows(max_windows);
    self
  }

  /// Sets [`Self::max_windows`] in place.
  ///
  /// # Panics
  /// If `max_windows` is `0` — a zero cap would refuse every clip, even the
  /// single-span short one. The serde path reports the same violation as a
  /// deserialize error instead.
  pub const fn set_max_windows(&mut self, max_windows: u32) -> &mut Self {
    assert!(check_max_windows(max_windows), "max_windows must be > 0");
    self.max_windows = max_windows;
    self
  }

  /// The windit [`WindowOptions`](windit::plan::WindowOptions) that reproduce
  /// clap's head + first-tail geometry: the fixed [`WINDOW_SAMPLES`] window, this
  /// plan's hop, and the tail policy mapped to windit's. `Pad` maps to `PadFull`
  /// (whose spans are identical to `KeepWithCoverage`, chosen because it
  /// documents clap's intent — the mel front-end `repeatpad`s the kept tail).
  ///
  /// The cap is passed through as `with_max_windows` for defense in depth: the
  /// O(1) pre-check in [`Self::spans`] already refuses an over-cap plan before
  /// windit is reached, so windit's kept count is always `<= max` here and its
  /// own [`WinditError::TooManyWindows`]/[`WinditError::AllocFailed`] never fire
  /// — but if `planned_windows` ever undercounted (a bug), windit would fail
  /// typed at `max + 1` rather than over-materialize.
  fn windit_options(&self) -> windit::plan::WindowOptions {
    windit::plan::WindowOptions::new(WINDOW_SAMPLES)
      .with_hop(self.hop_samples as usize)
      .with_tail(match self.tail {
        TailPolicy::Pad => windit::plan::TailPolicy::PadFull,
        TailPolicy::DropBelowMin(d) => {
          windit::plan::TailPolicy::DropBelowMin(d.min_samples() as usize)
        }
      })
      .with_max_windows(self.max_windows as usize)
  }

  /// Exactly `spans(total_samples).len()` for an admissible plan, in O(1) — the
  /// cap check must never materialize-then-count. Both branches count the same
  /// starts [`Self::spans`] keeps: under `Pad` every hop-multiple start in
  /// `[0, total)` (`⌈total / hop⌉`); under `DropBelowMin` the hop-multiples in
  /// `[0, total - min]` (a full window is always kept; a tail is kept iff its
  /// real length meets `min`, i.e. its start is `<= total - min`). Pinned
  /// against the real construction by `planned_windows_matches_materialized_len`
  /// and the `debug_assert_eq!` at the end of [`Self::spans`].
  ///
  /// No arithmetic here can overflow: `div_ceil` never overflows on `usize`,
  /// and in the `DropBelowMin` arm `total_samples > WINDOW_SAMPLES >= min_samples >= 1`
  /// gives `(total_samples - min_samples) / hop + 1 <= total_samples`.
  fn planned_windows(&self, total_samples: usize) -> usize {
    if total_samples == 0 {
      return 0;
    }
    // clap contract 1: a short clip is exactly one span, any hop/tail.
    if total_samples <= WINDOW_SAMPLES {
      return 1;
    }
    let hop = self.hop_samples as usize;
    match self.tail {
      TailPolicy::Pad => total_samples.div_ceil(hop),
      TailPolicy::DropBelowMin(d) => (total_samples - d.min_samples() as usize) / hop + 1,
    }
  }

  /// Map a clip of `total_samples` to the [`Span`]s to embed.
  ///
  /// The planned window count is bounded FIRST, in O(1), by
  /// [`Self::max_windows`]: an untrusted `total_samples` and small hop that would
  /// expand into millions of spans is refused before a single span (or CoreML
  /// inference) is materialized, so the plan can never become an out-of-memory or
  /// inference-flood lever.
  ///
  /// Geometry (window `W` = [`WINDOW_SAMPLES`], hop `H` = [`Self::hop_samples`]):
  ///
  /// - `total_samples == 0` → no windows (an empty clip has nothing to embed).
  /// - `total_samples <= W` → exactly one span `[0, total_samples)`, coverage
  ///   `total_samples / W` (`≤ 1.0`) — a short clip is embedded once,
  ///   `repeatpad`-padded, regardless of hop AND tail policy (clap contract 1:
  ///   windit's `DropBelowMin` would drop a short clip's sole span, but clap
  ///   never drops a clip's only representation).
  /// - `total_samples > W` → the windit plan (spans at `0, H, 2H, …` up to the
  ///   first ragged tail), then clap's multi-tail continuation (clap contract 2:
  ///   windit stops at the first tail, clap keeps striding, emitting
  ///   progressively shorter tails, each kept iff its real length meets the
  ///   policy threshold).
  ///
  /// The output is bit-for-bit clap's pre-windit geometry; see the module docs
  /// for the equivalence argument.
  ///
  /// # Errors
  /// [`Error::Windowing`] carrying [`WinditError::TooManyWindows`] if the planned
  /// count exceeds [`Self::max_windows`] — `got` is the FULL planned count,
  /// following granite's post-windit convention (windit's own raise aborts at
  /// `max + 1`) — or [`WinditError::AllocFailed`] if the span buffer cannot be
  /// allocated.
  pub fn spans(&self, total_samples: usize) -> Result<Vec<Span>> {
    // Cap FIRST, before any branch or allocation: the O(1) planned count is the
    // full would-be span count, so an over-cap clip dies here — no gigabyte
    // buffer, no flood of pushes, no `.expect()` panic on the allocator's
    // refusal, no inferences.
    let planned = self.planned_windows(total_samples);
    let max = self.max_windows as usize;
    if planned > max {
      return Err(Error::Windowing(WinditError::TooManyWindows {
        got: planned,
        max,
      }));
    }
    if total_samples == 0 {
      return Ok(Vec::new());
    }
    // clap contract 1 (SHORT CLIP): total <= window ⇒ exactly one span,
    // regardless of hop AND tail policy.
    if total_samples <= WINDOW_SAMPLES {
      return Ok(vec![Span::new(0, total_samples, WINDOW_SAMPLES)]);
    }
    let mut spans = windit::plan::WindowPlan::spans(&self.windit_options(), total_samples)?;
    // clap contract 2 (MULTI-TAIL): windit stops at the first span that reaches
    // the clip end; clap's overlapped plan (hop < window) keeps striding,
    // emitting progressively shorter tails until the stride passes the end. The
    // first tail start is derived arithmetically because DropBelowMin may have
    // dropped that span from the windit plan.
    let hop = self.hop_samples as usize;
    let min_keep = match self.tail {
      TailPolicy::Pad => 1,
      TailPolicy::DropBelowMin(d) => d.min_samples() as usize,
    };
    // Contract 2 appends exactly `planned - spans.len()` more spans (windit's
    // kept spans are a subset of the full plan), so reserve that exact count up
    // front: the pushes then stay within capacity, never an infallible growth
    // that would abort under an allocator refusal.
    let extra = planned - spans.len();
    spans
      .try_reserve_exact(extra)
      .map_err(|_| Error::Windowing(WinditError::AllocFailed { elements: extra }))?;
    let first_tail_start = (total_samples - WINDOW_SAMPLES).div_ceil(hop) * hop;
    let mut start = first_tail_start + hop;
    while start < total_samples {
      let len = total_samples - start; // < WINDOW_SAMPLES here, >= 1
      if len >= min_keep {
        spans.push(Span::new(start, len, WINDOW_SAMPLES));
      }
      start += hop;
    }
    debug_assert_eq!(
      spans.len(),
      planned,
      "planned_windows drifted from construction"
    );
    Ok(spans)
  }
}
