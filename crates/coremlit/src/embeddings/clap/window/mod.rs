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
//! representations (windit's `serde` feature is off), so the golden-pinned wire
//! spellings and the validated deserialization are unchanged by the windit port.
//!
//! The window length is **fixed** at [`WINDOW_SAMPLES`] (480 000 = 10 s at
//! 48 kHz) — the model's geometry, not a knob. Only the hop and the tail policy
//! are configurable. This module holds no audio and touches no model, so its
//! offsets and coverages are hermetically pinned (see the sibling `tests.rs`).

use crate::embeddings::clap::{audio::TARGET_SAMPLES, embedding::Embedding};

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

/// What [`WindowPlan`] does with a final chunk whose real samples fall short of a
/// full [`WINDOW_SAMPLES`] window.
///
/// A short tail is embedded by `repeatpad`ing it up to the fixed window, so a
/// kept tail's [`Span::coverage`] is `< 1.0` — the padding-aware fraction a
/// coverage-weighting policy uses to down-weight it. This policy chooses whether
/// such a tail is kept at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
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
  DropBelowMin {
    /// The keep threshold in real samples; validated into `1..=WINDOW_SAMPLES`
    /// by [`WindowPlan`]'s checked setters and serde path.
    min_samples: u32,
  },
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
    TailPolicy::DropBelowMin { min_samples } => {
      min_samples > 0 && min_samples as usize <= WINDOW_SAMPLES
    }
  }
}

/// Overlapped-chunking plan: a validated hop and tail policy over the fixed
/// [`WINDOW_SAMPLES`] window (rust-options-pattern).
///
/// [`Self::spans`] is the pure-geometry core — it maps a clip length to the list
/// of [`Span`]s to embed, with no audio and no model involved, so the
/// offsets and coverages are hermetically testable.
///
/// # Validated deserialization
///
/// `Deserialize` routes through a private `WindowPlanRepr` via
/// `serde(try_from)`, holding a config-file or hand-written `WindowPlan` to the
/// SAME `hop_samples`/`min_samples` invariants the checked setters enforce.
/// Deriving `Deserialize` on the fields directly would bypass
/// [`Self::set_hop_samples`]: `{"hop_samples": 0}` would deserialize and then
/// loop forever (a zero hop never advances), and `{"hop_samples": 960000}` would
/// silently leave 10 s gaps of un-embedded audio between chunks. Invalid input
/// now fails to deserialize instead (mirrors speakerkit's `WindowOptions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "WindowPlanRepr"))]
pub struct WindowPlan {
  hop_samples: u32,
  tail: TailPolicy,
}

/// The plain wire form [`WindowPlan`]'s `Deserialize` deserializes FIRST
/// (carrying the field defaults), before [`WindowPlan::try_from`] applies the
/// range checks. Its whole purpose is to make the validated setters unbypassable
/// via serde — it is never constructed or exposed otherwise.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct WindowPlanRepr {
  #[serde(default = "default_hop_samples")]
  hop_samples: u32,
  #[serde(default)]
  tail: TailPolicy,
}

#[cfg(feature = "serde")]
fn default_hop_samples() -> u32 {
  DEFAULT_HOP_SAMPLES
}

#[cfg(feature = "serde")]
impl TryFrom<WindowPlanRepr> for WindowPlan {
  type Error = String;

  /// Applies [`check_hop_samples`] and [`check_tail`] — the exact invariants
  /// [`WindowPlan::set_hop_samples`] / [`WindowPlan::set_tail_policy`] assert —
  /// as fallible checks, so a serde-deserialized plan can never construct the
  /// infinite-loop (`hop == 0`) or audio-skipping (`hop > window`) geometry the
  /// builders reject.
  fn try_from(r: WindowPlanRepr) -> Result<Self, Self::Error> {
    if !check_hop_samples(r.hop_samples) {
      return Err(format!(
        "hop_samples ({}) must be > 0 and <= WINDOW_SAMPLES ({WINDOW_SAMPLES})",
        r.hop_samples
      ));
    }
    if !check_tail(r.tail) {
      return Err(format!(
        "tail DropBelowMin.min_samples must be > 0 and <= WINDOW_SAMPLES ({WINDOW_SAMPLES}), got {:?}",
        r.tail
      ));
    }
    Ok(Self {
      hop_samples: r.hop_samples,
      tail: r.tail,
    })
  }
}

impl Default for WindowPlan {
  fn default() -> Self {
    Self::new()
  }
}

impl WindowPlan {
  /// A plan with [`DEFAULT_HOP_SAMPLES`] (no overlap) and [`TailPolicy::Pad`]
  /// (keep every tail). Tiles a clip into back-to-back 10 s windows, the last
  /// `repeatpad`-padded.
  pub const fn new() -> Self {
    Self {
      hop_samples: DEFAULT_HOP_SAMPLES,
      tail: TailPolicy::Pad,
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

  /// The windit [`WindowOptions`](windit::plan::WindowOptions) that reproduce
  /// clap's head + first-tail geometry: the fixed [`WINDOW_SAMPLES`] window, this
  /// plan's hop, and the tail policy mapped to windit's. `Pad` maps to `PadFull`
  /// (whose spans are identical to `KeepWithCoverage`, chosen because it
  /// documents clap's intent — the mel front-end `repeatpad`s the kept tail).
  fn windit_options(&self) -> windit::plan::WindowOptions {
    windit::plan::WindowOptions::new(WINDOW_SAMPLES)
      .with_hop(self.hop_samples as usize)
      .with_tail(match self.tail {
        TailPolicy::Pad => windit::plan::TailPolicy::PadFull,
        TailPolicy::DropBelowMin { min_samples } => {
          windit::plan::TailPolicy::DropBelowMin(min_samples as usize)
        }
      })
  }

  /// Map a clip of `total_samples` to the [`Span`]s to embed.
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
  #[must_use]
  pub fn spans(&self, total_samples: usize) -> Vec<Span> {
    if total_samples == 0 {
      return Vec::new();
    }
    // clap contract 1 (SHORT CLIP): total <= window ⇒ exactly one span,
    // regardless of hop AND tail policy.
    if total_samples <= WINDOW_SAMPLES {
      return vec![Span::new(0, total_samples, WINDOW_SAMPLES)];
    }
    let mut spans = windit::plan::WindowPlan::spans(&self.windit_options(), total_samples).expect(
      "windit options are valid by construction: WINDOW_SAMPLES is a non-zero const, \
       hop is setter/serde-validated into 1..=WINDOW_SAMPLES, and no max_windows cap is set; \
       the only remaining failure is allocator refusal, where the pre-windit Vec growth aborted too",
    );
    // clap contract 2 (MULTI-TAIL): windit stops at the first span that reaches
    // the clip end; clap's overlapped plan (hop < window) keeps striding,
    // emitting progressively shorter tails until the stride passes the end. The
    // first tail start is derived arithmetically because DropBelowMin may have
    // dropped that span from the windit plan.
    let hop = self.hop_samples as usize;
    let min_keep = match self.tail {
      TailPolicy::Pad => 1,
      TailPolicy::DropBelowMin { min_samples } => min_samples as usize,
    };
    let first_tail_start = (total_samples - WINDOW_SAMPLES).div_ceil(hop) * hop;
    let mut start = first_tail_start + hop;
    while start < total_samples {
      let len = total_samples - start; // < WINDOW_SAMPLES here, >= 1
      if len >= min_keep {
        spans.push(Span::new(start, len, WINDOW_SAMPLES));
      }
      start += hop;
    }
    spans
  }
}
