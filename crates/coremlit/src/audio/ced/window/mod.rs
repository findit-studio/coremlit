//! Overlapped long-clip chunking: [`WindowPlan`] turns a clip length into the
//! list of [`Span`]s the classifier scores one at a time.
//!
//! # windit engine + two clap-precedent guards (spec §2)
//!
//! The window GEOMETRY is the generic `windit` engine: [`Span`] is
//! `windit::plan::Span` and [`WindowPlan::spans`] plans the head through
//! `windit::plan::WindowPlan`. Two behaviours are this module's own contract,
//! reproduced as thin guards on top of the windit plan (the clap precedent,
//! re-cut for CED's 160 000-sample window):
//!
//! 1. **Short clip** (`total <= WINDOW_SAMPLES`): exactly one span, whatever
//!    the hop AND tail policy — a clip's only representation is never dropped.
//! 2. **Multi-tail continuation**: windit stops at the first ragged tail; an
//!    overlapped plan (`hop < window`) keeps striding, emitting progressively
//!    shorter tails until the stride passes the clip end — matching
//!    soundevents' `chunk_slices` overlapped semantics for `total > window`.
//!
//! windit's *aggregation* engine is deliberately NOT used: its built-ins are
//! renormalizing unit-vector policies, the wrong domain for independent
//! per-class probabilities (see the sibling `aggregate` module).
//!
//! The window length is **fixed** at [`WINDOW_SAMPLES`] (160 000 = 10 s at
//! 16 kHz) — model geometry, not a knob. The hop, the tail policy, and the
//! [`WindowPlan::max_windows`] resource cap are configurable. This module holds
//! no audio and touches no model, so its offsets and coverages are hermetically
//! pinned (see the sibling `tests.rs`).
//!
//! # Resource cap (spec §4)
//!
//! [`WindowPlan::spans`] counts its plan in O(1) and refuses one exceeding
//! [`WindowPlan::max_windows`] ([`DEFAULT_MAX_WINDOWS`], default-on) with a
//! typed [`WinditError::TooManyWindows`] BEFORE materializing any span — so a
//! serde-supplied `hop_samples: 1` over a modest clip (320 000 spans, ~643 MiB
//! and 320 000 inferences at the default) is a typed refusal, not a panic/OOM.

use crate::audio::ced::{
  WINDOW_SAMPLES,
  error::{Error, Result, WinditError},
};

/// windit's window span (`windit::plan::Span`), re-exported as this module's
/// geometry unit — the half-open real range `[start, end)` a [`WindowPlan`]
/// plans and the classifier scores. Every CED-produced span carries
/// `window() == `[`WINDOW_SAMPLES`], so [`Span::coverage`] is the
/// padding-aware `real length / 160_000` fraction. (`Span::new` is 3-arg —
/// `(start, len, window)` — and reports `len()`/`end()`.)
pub use windit::plan::Span;

#[cfg(test)]
mod tests;

/// Default [`WindowPlan::hop_samples`]: one full window (no overlap), so the
/// default plan tiles a clip into back-to-back 10 s chunks — matching
/// soundevents' `ChunkingOptions` default (`window == hop`).
pub const DEFAULT_HOP_SAMPLES: u32 = WINDOW_SAMPLES as u32;

/// Default [`WindowPlan::max_windows`]: 100 000 windows.
///
/// The cap is a resource rail, not a latency policy: each planned window costs
/// one full CoreML inference, and `classify_windows` retains a 527-float
/// confidence vector (~2.1 KiB) per window, so 100 000 caps that retention at
/// ~206 MiB. It admits every realistic clip — 24 h of audio at a 1 s hop is
/// 86 400 windows; at the default no-overlap hop the cap is ~11 days of audio —
/// while rejecting hop-abuse: at `hop_samples == 1` ANY clip long enough to
/// window at all (> 10 s) plans more than 160 000 windows and fails typed.
/// Latency-sensitive services should lower it; raising it is a deliberate
/// opt-in to more memory and inference work.
pub const DEFAULT_MAX_WINDOWS: u32 = 100_000;

/// What [`WindowPlan`] does with a final chunk whose real samples fall short of
/// a full [`WINDOW_SAMPLES`] window.
///
/// A kept short tail is scored by zero-padding it up to the fixed window
/// (the believed sub-window policy, probe-pinned), so its [`Span::coverage`]
/// is `< 1.0`. This policy chooses whether such a tail is kept at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum TailPolicy {
  /// Keep the final short chunk (any tail with ≥ 1 real sample). Nothing is
  /// dropped, so the whole clip is covered. The default.
  #[default]
  Pad,
  /// Drop a final chunk whose real length is below `min_samples`, so a
  /// trailing sliver dominated by padding never contributes. A chunk at or
  /// above the threshold is kept. The single window a clip shorter than one
  /// full window produces is never dropped (there is nothing else to
  /// represent it).
  DropBelowMin {
    /// The keep threshold in real samples; validated into
    /// `1..=WINDOW_SAMPLES` by [`WindowPlan`]'s checked setters and serde
    /// path. No default — soundevents has no drop policy, so there is no
    /// upstream value to mirror.
    min_samples: u32,
  },
}

/// Whether `hop_samples` is in the valid `1..=WINDOW_SAMPLES` range: positive
/// (a zero hop never advances) and no larger than one window (a hop past the
/// window would leave gaps of un-classified audio — soundevents' sparse-skim
/// mode is a recorded non-goal).
const fn check_hop_samples(v: u32) -> bool {
  v > 0 && v as usize <= WINDOW_SAMPLES
}

/// Whether `max_windows` is a usable cap: strictly positive. A zero cap would
/// admit no plan at all (even the single-span short clip), so a default-carrying
/// field that can never score anything is a misconfiguration, the same class as
/// `hop == 0`. `u32::MAX` is the deliberate "effectively uncapped" escape hatch.
const fn check_max_windows(v: u32) -> bool {
  v > 0
}

/// Whether a [`TailPolicy::DropBelowMin`] `min_samples` is in
/// `1..=WINDOW_SAMPLES` (a zero threshold would mean "drop below one sample"
/// yet drop nothing, and a threshold above the window can never be met by a
/// sub-window tail).
const fn check_tail(tail: TailPolicy) -> bool {
  match tail {
    TailPolicy::Pad => true,
    TailPolicy::DropBelowMin { min_samples } => {
      min_samples > 0 && min_samples as usize <= WINDOW_SAMPLES
    }
  }
}

/// Overlapped-chunking plan: a validated hop and tail policy over the fixed
/// [`WINDOW_SAMPLES`] window, plus a [`Self::max_windows`] resource cap
/// (rust-options-pattern).
///
/// [`Self::spans`] is the pure-geometry core — it maps a clip length to the
/// list of [`Span`]s to score, with no audio and no model involved, so the
/// offsets and coverages are hermetically testable. `max_windows` bounds that
/// count in O(1) BEFORE any span is materialized, so an untrusted length + hop
/// cannot expand into an out-of-memory allocation or a flood of inferences.
///
/// # Validated deserialization
///
/// `Deserialize` routes through a private `WindowPlanRepr` via
/// `serde(try_from)`, holding a config-file `WindowPlan` to the SAME
/// `hop_samples`/`min_samples`/`max_windows` invariants the checked setters
/// enforce: `{"hop_samples": 0}` would loop forever, `{"hop_samples": 320000}`
/// would silently skip audio, and `{"max_windows": 0}` could never score
/// anything; all fail to deserialize instead (the clap unbypassable-setter
/// pattern). An omitted `max_windows` fills [`DEFAULT_MAX_WINDOWS`], so the cap
/// is default-on for every deserialized plan.
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
/// range checks. Its whole purpose is to make the validated setters
/// unbypassable via serde — it is never constructed or exposed otherwise.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
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
  /// the exact invariants the checked setters assert — as fallible checks, so a
  /// serde-deserialized plan can never construct the infinite-loop (`hop == 0`)
  /// or audio-skipping (`hop > window`) geometry the builders reject, nor a
  /// score-nothing (`max_windows == 0`) cap.
  fn try_from(r: WindowPlanRepr) -> core::result::Result<Self, Self::Error> {
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
  /// clip into back-to-back 10 s windows, the last zero-padded.
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
  /// advances and a hop past the window leaves gaps of un-classified audio.
  /// The serde path reports the same violation as a deserialize error instead.
  pub const fn set_hop_samples(&mut self, hop_samples: u32) -> &mut Self {
    assert!(
      check_hop_samples(hop_samples),
      "hop_samples must be > 0 and <= WINDOW_SAMPLES (160_000)"
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
      "TailPolicy::DropBelowMin.min_samples must be > 0 and <= WINDOW_SAMPLES (160_000)"
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
  /// the head + first-tail geometry: the fixed [`WINDOW_SAMPLES`] window, this
  /// plan's hop, and the tail policy mapped to windit's. `Pad` maps to
  /// `PadFull` (span-identical to `KeepWithCoverage`; chosen because it
  /// documents the intent — the classifier zero-pads the kept tail).
  ///
  /// The cap is passed through as `with_max_windows` for defense in depth: the
  /// O(1) pre-check in [`Self::spans`] already refuses an over-cap plan before
  /// windit is reached, so windit's kept count is always `<= max` here and its
  /// own [`WinditError::TooManyWindows`]/[`WinditError::AllocFailed`] never
  /// fire — but if `planned_windows` ever undercounted (a bug), windit would
  /// fail typed at `max + 1` rather than over-materialize.
  fn windit_options(&self) -> windit::plan::WindowOptions {
    windit::plan::WindowOptions::new(WINDOW_SAMPLES)
      .with_hop(self.hop_samples as usize)
      .with_tail(match self.tail {
        TailPolicy::Pad => windit::plan::TailPolicy::PadFull,
        TailPolicy::DropBelowMin { min_samples } => {
          windit::plan::TailPolicy::DropBelowMin(min_samples as usize)
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
    // Guard 1: a short clip is exactly one span, any hop/tail.
    if total_samples <= WINDOW_SAMPLES {
      return 1;
    }
    let hop = self.hop_samples as usize;
    match self.tail {
      TailPolicy::Pad => total_samples.div_ceil(hop),
      TailPolicy::DropBelowMin { min_samples } => (total_samples - min_samples as usize) / hop + 1,
    }
  }

  /// Map a clip of `total_samples` to the [`Span`]s to score.
  ///
  /// The planned window count is bounded FIRST, in O(1), by
  /// [`Self::max_windows`]: an untrusted `total_samples` and small hop that
  /// would expand into millions of spans is refused before a single span (or
  /// CoreML inference) is materialized, so the plan can never become an
  /// out-of-memory or inference-flood lever.
  ///
  /// Geometry (window `W` = [`WINDOW_SAMPLES`], hop `H` = [`Self::hop_samples`]):
  ///
  /// - `total_samples == 0` → no windows (an empty clip has nothing to score).
  /// - `total_samples <= W` → exactly one span `[0, total_samples)`, coverage
  ///   `total_samples / W` (`≤ 1.0`) — a short clip is scored once,
  ///   zero-padded, regardless of hop AND tail policy (guard 1: windit's
  ///   `DropBelowMin` would drop a short clip's sole span; a clip's only
  ///   representation is never dropped).
  /// - `total_samples > W` → the windit plan (spans at `0, H, 2H, …` up to the
  ///   first ragged tail), then the multi-tail continuation (guard 2: windit
  ///   stops at the first tail; this plan keeps striding, emitting
  ///   progressively shorter tails, each kept iff its real length meets the
  ///   policy threshold — soundevents' `chunk_slices` overlapped semantics,
  ///   pinned against a naive reference in the sibling tests).
  ///
  /// # Errors
  /// [`Error::Windowing`] carrying [`WinditError::TooManyWindows`] if the
  /// planned count exceeds [`Self::max_windows`] — `got` is the FULL planned
  /// count, following granite's post-windit convention (windit's own raise
  /// aborts at `max + 1`) — or [`WinditError::AllocFailed`] if the span buffer
  /// cannot be allocated.
  pub fn spans(&self, total_samples: usize) -> Result<Vec<Span>> {
    // Cap FIRST, before any branch or allocation: the O(1) planned count is the
    // full would-be span count, so an over-cap clip dies here — no 643 MiB
    // buffer, no 320 000 pushes, no inferences.
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
    // Guard 1 (SHORT CLIP): total <= window ⇒ exactly one span, regardless of
    // hop AND tail policy.
    if total_samples <= WINDOW_SAMPLES {
      return Ok(vec![Span::new(0, total_samples, WINDOW_SAMPLES)]);
    }
    let mut spans = windit::plan::WindowPlan::spans(&self.windit_options(), total_samples)?;
    // Guard 2 (MULTI-TAIL): windit stops at the first span that reaches the
    // clip end; an overlapped plan (hop < window) keeps striding, emitting
    // progressively shorter tails until the stride passes the end. The first
    // tail start is derived arithmetically because DropBelowMin may have
    // dropped that span from the windit plan.
    let hop = self.hop_samples as usize;
    let min_keep = match self.tail {
      TailPolicy::Pad => 1,
      TailPolicy::DropBelowMin { min_samples } => min_samples as usize,
    };
    // Guard 2 appends exactly `planned - spans.len()` more spans (windit's kept
    // spans are a subset of the full plan), so reserve that exact count up
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
