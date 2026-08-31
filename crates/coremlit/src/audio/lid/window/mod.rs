//! Long-clip chunking geometry: [`WindowPlan`] turns a clip length into the
//! list of [`Span`]s the identifier scores one at a time.
//!
//! # windit engine + this door's own tail rule
//!
//! The window GEOMETRY is the generic `windit` engine — [`Span`] is
//! `windit::plan::Span` and the *head* (every full-length window) is planned by
//! `windit::plan::WindowPlan` under `DropBelowMin(window)`. What windit does
//! with the leftover is replaced wholesale, because this graph's tail options
//! are not the ones windit (or `audio::ced`) has: its time axis is a
//! `RangeDims`, so a short tail can be scored AT ITS OWN LENGTH rather than
//! padded, and a full-length window can be slid backwards over audio already
//! read. See [`TailPolicy`].
//!
//! windit's *aggregation* engine is deliberately NOT used either — see the
//! sibling `aggregate` module for why this domain needs its own.
//!
//! # Exactly one tail span, deliberately
//!
//! `audio::ced` continues striding past the first ragged tail, emitting
//! progressively shorter ones, because soundevents' `chunk_slices` defines
//! that. This door has no upstream chunker to mirror, so it stops at one: after
//! the head, ONE tail span (or none) covers everything the head left, and every
//! sample is still inside at least one span. A second, shorter tail over audio
//! the first already covered would add an inference and a noisier vector for no
//! new evidence.
//!
//! # Geometry is a knob here, not model shape
//!
//! `audio::ced`'s window is fixed at the graph's only accepted input length.
//! This graph accepts [`MIN_FRAMES`]..=[`MAX_FRAMES`], so the window IS a
//! choice — [`DEFAULT_WINDOW_SAMPLES`] is a measured default, not a constant of
//! the model, and it can move without reshaping this API.
//!
//! # Resource cap
//!
//! [`WindowPlan::spans`] counts its plan in O(1) and refuses one exceeding
//! [`WindowPlan::max_windows`] ([`DEFAULT_MAX_WINDOWS`], default-on) with a
//! typed [`WinditError::TooManyWindows`] BEFORE materializing any span — so a
//! serde-supplied `hop_samples: 1` over a modest clip is a typed refusal, not
//! an OOM and not a flood of inferences.
//!
//! [`MIN_FRAMES`]: crate::audio::lid::MIN_FRAMES
//! [`MAX_FRAMES`]: crate::audio::lid::MAX_FRAMES

use crate::audio::lid::{
  MAX_SAMPLES, MIN_SAMPLES,
  error::{Error, Result, WinditError},
};

/// windit's window span (`windit::plan::Span`), re-exported as this module's
/// geometry unit — the half-open sample range `[start, end)` a [`WindowPlan`]
/// plans and the identifier scores. Every span a plan produces carries
/// `window() == `[`WindowPlan::window_samples`], so [`Span::coverage`] is the
/// `real length / window` fraction — `1.0` for every span except a
/// [`TailPolicy::Partial`] tail and the sole span of a clip shorter than one
/// window.
pub use windit::plan::Span;

#[cfg(test)]
mod tests;

/// Default [`WindowPlan::window_samples`]: 160 000 samples — **10 s** at 16 kHz,
/// which is 1 001 mel frames.
///
/// Chosen on two measurements, in this order:
///
/// 1. **It is the frame count [`prewarm`] already specializes.** Every unseen
///    frame count costs a one-off 55–97 ms graph specialization against a 9–23 ms
///    steady state (see the module docs' performance notes). A fixed window
///    means the long path pays that ONCE, and pinning the default at the length
///    `prewarm` warms means it is paid off the first real request for free.
///    `default_window_is_the_length_prewarm_warms` holds the two together.
/// 2. **Self-consistency improves with window length up to it, and
///    code-switch resolution gets worse above it.** Reproducing the single-shot
///    top-1 from windows scored 81 % at 3 s windows, 87 % at 5 s and 91 % at
///    10 s; at 30 s windows a third of a clip in another language stops winning
///    any window at all. The module docs carry the full table.
///
/// Both halves are contingent, which is why this is a `pub const` and
/// [`WindowPlan::with_geometry`] exists: a future export with enumerated shapes
/// would change the specialization economics, and a caller who needs finer
/// code-switch resolution should shorten it.
///
/// [`prewarm`]: crate::audio::lid::Identifier::prewarm
pub const DEFAULT_WINDOW_SAMPLES: u32 = 160_000;

/// Default [`WindowPlan::hop_samples`]: one full window, so the default plan
/// tiles a clip into back-to-back 10 s chunks with no overlap and no sample
/// scored twice.
pub const DEFAULT_HOP_SAMPLES: u32 = DEFAULT_WINDOW_SAMPLES;

/// Default [`WindowPlan::max_windows`]: 100 000 windows.
///
/// A resource rail, not a latency policy: each planned window costs one full
/// CoreML inference, and the per-window path retains a
/// [`NUM_LANGUAGES`](crate::audio::lid::NUM_LANGUAGES)-float row (428 B), so
/// 100 000 caps that retention at ~41 MiB. It admits every realistic clip — at
/// the default 10 s hop it is ~11.5 days of audio — while still refusing
/// hop-abuse: at `hop_samples == 1` any clip longer than one window plans more
/// windows than there are samples in a second of audio, and a 30 s one plans
/// 320 001. Latency-sensitive services should lower it; raising it is a
/// deliberate opt-in to more memory and inference work.
pub const DEFAULT_MAX_WINDOWS: u32 = 100_000;

/// What [`WindowPlan`] does with the audio a plan's full-length windows leave
/// uncovered at the end of a clip.
///
/// All three variants are unit-shaped on purpose: the payload a threshold would
/// carry has exactly one defensible value here ([`MIN_SAMPLES`], the shortest
/// clip the graph accepts at all), so it is a constant of the model rather than
/// a knob.
///
/// # Why "pad the tail" is not among them
///
/// `audio::ced` pads, because its graph accepts one input length and there is
/// nothing else to do. This one accepts a range, and padding is measurably
/// worse than both alternatives below. The fused in-graph mean subtraction
/// reduces over the time axis and therefore SEES the padding: measured against
/// the same audio scored honestly, padding a tail up to the 10 s default window
/// shifts log-probabilities by 2.5 nats at a 9 s tail and 19 nats at a 1 s one,
/// and **changes the reported language** for every tail of 3 s or less (module
/// docs, "Clips longer than 30 s"). [`Self::SlideBack`] gets a full-length
/// window with no padding at all, and [`Self::Partial`] gets an honest short
/// one; neither pays that shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum TailPolicy {
  /// Slide one final FULL-length window backwards to end at the clip end, so
  /// it overlaps its predecessor instead of running short. The default.
  ///
  /// Every span in the plan is then exactly one window long: one graph
  /// specialization, no padding, and no window whose vector was computed from
  /// less audio than the others — which is also what makes duration weighting
  /// a no-op under this policy (see the `aggregate` module). The price is that
  /// the overlapped samples are read twice, so the clip's last
  /// `window - (total mod hop)` samples carry slightly more weight in the
  /// aggregate than the middle of the clip does.
  #[default]
  SlideBack,
  /// Score the ragged tail at its own length — the graph's `RangeDims` time
  /// axis accepts it, so nothing is padded and nothing is read twice.
  ///
  /// The cost is one extra graph specialization for that one frame count
  /// (55–97 ms, once per distinct tail length the process sees), and a window
  /// whose vector rests on less audio than its neighbours' — which is exactly
  /// what the aggregator's duration weighting exists to discount.
  ///
  /// A tail shorter than [`MIN_SAMPLES`] (0.09 s) is dropped rather than
  /// scored: the graph refuses it outright. That leaves under 0.09 s of a clip
  /// unrepresented, and only when the clip length lands in that sliver.
  Partial,
  /// Drop the uncovered tail entirely; every scored window is full length and
  /// none is read twice.
  ///
  /// What is discarded is always shorter than one hop — the tail exists only
  /// when the head could not stride again — so at the default no-overlap hop
  /// this drops up to 10 s from the end of a clip. Choose it when a partial or
  /// re-read window would be worse than no window; otherwise prefer
  /// [`Self::SlideBack`], which covers the same audio at the same cost per
  /// window.
  Drop,
}

/// Whether `window_samples` is a length the graph can score:
/// [`MIN_SAMPLES`]..=[`MAX_SAMPLES`]. A window outside it could never be
/// predicted on, so a plan built from one would fail per window rather than at
/// configuration time.
const fn check_window_samples(v: u32) -> bool {
  v as usize >= MIN_SAMPLES && v as usize <= MAX_SAMPLES
}

/// Whether `hop_samples` is valid against `window_samples`: positive (a zero
/// hop never advances) and no larger than one window (a hop past the window
/// would stride over un-scored audio — a sparse-skim mode is a recorded
/// non-goal).
const fn check_hop_samples(hop: u32, window: u32) -> bool {
  hop > 0 && hop <= window
}

/// Whether `max_windows` is a usable cap: strictly positive. A zero cap would
/// admit no plan at all, even the single-span short clip.
const fn check_max_windows(v: u32) -> bool {
  v > 0
}

/// Long-clip chunking plan: a jointly validated window/hop geometry, a
/// [`TailPolicy`], and a [`Self::max_windows`] resource cap
/// (rust-options-pattern).
///
/// [`Self::spans`] is the pure-geometry core — it maps a clip length to the
/// list of [`Span`]s to score, with no audio and no model involved, so offsets
/// and coverages are hermetically testable. `max_windows` bounds that count in
/// O(1) BEFORE any span is materialized, so an untrusted length plus a small
/// hop cannot expand into an out-of-memory allocation or a flood of inferences.
///
/// # One geometry setter, not two
///
/// `window_samples` and `hop_samples` are validated against each other, so they
/// are set together by [`Self::with_geometry`]. A per-field window setter would
/// make the most natural call on a default plan
/// (`WindowPlan::new().with_window_samples(48_000)`, shrinking the window below
/// the default 160 000-sample hop) a panic, and a per-field setter that
/// silently moved the other field would be worse.
///
/// # Validated deserialization
///
/// `Deserialize` routes through a private `WindowPlanRepr` via
/// `serde(try_from)`, holding a config-file `WindowPlan` to the SAME invariants
/// the checked setters enforce: `{"hop_samples": 0}` would loop forever,
/// `{"hop_samples": 320000}` at the default window would skip audio,
/// `{"window_samples": 1000000}` could never be predicted on, and
/// `{"max_windows": 0}` could never score anything. All four fail to
/// deserialize instead. Every field is optional and fills its `DEFAULT_*`, so
/// the cap is default-on for every deserialized plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "WindowPlanRepr"))]
pub struct WindowPlan {
  window_samples: u32,
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
  #[serde(default = "default_window_samples")]
  window_samples: u32,
  #[serde(default = "default_hop_samples")]
  hop_samples: u32,
  #[serde(default)]
  tail: TailPolicy,
  #[serde(default = "default_max_windows")]
  max_windows: u32,
}

#[cfg(feature = "serde")]
fn default_window_samples() -> u32 {
  DEFAULT_WINDOW_SAMPLES
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

  /// Applies [`check_window_samples`], [`check_hop_samples`] and
  /// [`check_max_windows`] — the exact invariants the checked setters assert —
  /// as fallible checks, so a deserialized plan can never carry an
  /// unpredictable window, an infinite-loop (`hop == 0`) or audio-skipping
  /// (`hop > window`) stride, or a score-nothing (`max_windows == 0`) cap.
  fn try_from(r: WindowPlanRepr) -> core::result::Result<Self, Self::Error> {
    if !check_window_samples(r.window_samples) {
      return Err(format!(
        "window_samples ({}) must be in {MIN_SAMPLES}..={MAX_SAMPLES}",
        r.window_samples
      ));
    }
    if !check_hop_samples(r.hop_samples, r.window_samples) {
      return Err(format!(
        "hop_samples ({}) must be > 0 and <= window_samples ({})",
        r.hop_samples, r.window_samples
      ));
    }
    if !check_max_windows(r.max_windows) {
      return Err(format!("max_windows ({}) must be > 0", r.max_windows));
    }
    Ok(Self {
      window_samples: r.window_samples,
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
  /// A plan with [`DEFAULT_WINDOW_SAMPLES`] and [`DEFAULT_HOP_SAMPLES`] (10 s
  /// windows, no overlap), [`TailPolicy::SlideBack`], and
  /// [`DEFAULT_MAX_WINDOWS`].
  #[must_use]
  pub const fn new() -> Self {
    Self {
      window_samples: DEFAULT_WINDOW_SAMPLES,
      hop_samples: DEFAULT_HOP_SAMPLES,
      tail: TailPolicy::SlideBack,
      max_windows: DEFAULT_MAX_WINDOWS,
    }
  }

  /// Length in samples of one full window — the amount of audio each scored
  /// vector rests on. See [`DEFAULT_WINDOW_SAMPLES`].
  #[inline]
  pub const fn window_samples(&self) -> u32 {
    self.window_samples
  }

  /// Distance in samples between successive window starts. `<`
  /// [`Self::window_samples`] means overlapping windows; `==` means
  /// back-to-back.
  #[inline]
  pub const fn hop_samples(&self) -> u32 {
    self.hop_samples
  }

  /// The configured tail policy.
  #[inline]
  pub const fn tail_policy(&self) -> TailPolicy {
    self.tail
  }

  /// The maximum number of windows [`Self::spans`] may plan before it refuses
  /// the clip with [`WinditError::TooManyWindows`]. See [`DEFAULT_MAX_WINDOWS`].
  #[inline]
  pub const fn max_windows(&self) -> u32 {
    self.max_windows
  }

  /// Builder form of [`Self::set_geometry`].
  ///
  /// # Panics
  /// As [`Self::set_geometry`].
  #[must_use]
  pub const fn with_geometry(mut self, window_samples: u32, hop_samples: u32) -> Self {
    self.set_geometry(window_samples, hop_samples);
    self
  }

  /// Sets [`Self::window_samples`] and [`Self::hop_samples`] together, in
  /// place.
  ///
  /// # Panics
  /// If `window_samples` is outside
  /// [`MIN_SAMPLES`]..=[`MAX_SAMPLES`]
  /// — the graph could not score such a window — or if `hop_samples` is not in
  /// `1..=window_samples`: a zero hop never advances, and a hop past the window
  /// strides over un-scored audio. The serde path reports the same violations
  /// as deserialize errors instead.
  pub const fn set_geometry(&mut self, window_samples: u32, hop_samples: u32) -> &mut Self {
    assert!(
      check_window_samples(window_samples),
      "window_samples must be in MIN_SAMPLES..=MAX_SAMPLES (1_440..=480_159)"
    );
    assert!(
      check_hop_samples(hop_samples, window_samples),
      "hop_samples must be > 0 and <= window_samples"
    );
    self.window_samples = window_samples;
    self.hop_samples = hop_samples;
    self
  }

  /// Builder form of [`Self::set_tail_policy`].
  #[must_use]
  pub const fn with_tail_policy(mut self, tail: TailPolicy) -> Self {
    self.set_tail_policy(tail);
    self
  }

  /// Sets [`Self::tail_policy`] in place. Every variant is valid at every
  /// geometry, so this cannot fail.
  pub const fn set_tail_policy(&mut self, tail: TailPolicy) -> &mut Self {
    self.tail = tail;
    self
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

  /// Start of the LAST full-length window, and whether any audio lies beyond
  /// its end. Only meaningful for `total_samples > window_samples`, which every
  /// caller checks first.
  ///
  /// `last_full_start` is the largest hop multiple `s` with
  /// `s + window <= total`; `s + window <= total` also makes the sum
  /// overflow-free.
  const fn head_end(&self, total_samples: usize) -> (usize, bool) {
    let (window, hop) = (self.window_samples as usize, self.hop_samples as usize);
    let last_full_start = ((total_samples - window) / hop) * hop;
    (last_full_start, last_full_start + window < total_samples)
  }

  /// Exactly `spans(total_samples).len()` for an admissible plan, in O(1) — the
  /// cap check must never materialize-then-count.
  ///
  /// The head is every hop multiple `s` with `s + window <= total`, i.e.
  /// `(total - window) / hop + 1` of them. A tail span exists only when the
  /// head leaves audio uncovered (`last_full_start + window < total`), and then
  /// [`TailPolicy`] decides whether it is emitted. Pinned against the real
  /// construction by `planned_windows_matches_materialized_len` and the
  /// `debug_assert_eq!` at the end of [`Self::spans`].
  fn planned_windows(&self, total_samples: usize) -> usize {
    if total_samples == 0 {
      return 0;
    }
    let (window, hop) = (self.window_samples as usize, self.hop_samples as usize);
    // Guard: a clip no longer than one window is exactly one span, whatever the
    // hop and tail policy — its only representation is never dropped.
    if total_samples <= window {
      return 1;
    }
    let full = (total_samples - window) / hop + 1;
    let (last_full_start, uncovered) = self.head_end(total_samples);
    let tail = match self.tail {
      TailPolicy::Drop => false,
      TailPolicy::SlideBack => uncovered,
      TailPolicy::Partial => uncovered && total_samples - last_full_start - hop >= MIN_SAMPLES,
    };
    full + usize::from(tail)
  }

  /// Map a clip of `total_samples` to the [`Span`]s to score.
  ///
  /// The planned window count is bounded FIRST, in O(1), by
  /// [`Self::max_windows`]: an untrusted `total_samples` and small hop that
  /// would expand into millions of spans is refused before a single span (or
  /// CoreML inference) is materialized.
  ///
  /// Geometry (window `W` = [`Self::window_samples`], hop `H` =
  /// [`Self::hop_samples`], `S` = the last hop multiple with `S + W <= total`):
  ///
  /// - `total_samples == 0` → no windows.
  /// - `total_samples <= W` → exactly one span `[0, total_samples)`, coverage
  ///   `total / W` (`<= 1.0`), whatever the hop and tail policy. This is the
  ///   guard that makes the long path agree with the single-shot one on a clip
  ///   that already fits.
  /// - `total_samples > W` → the head, `⌊(total − W) / H⌋ + 1` full-length
  ///   spans at `0, H, 2H, …, S`; then, IF `S + W < total` (the head left audio
  ///   uncovered), at most one tail span:
  ///   [`TailPolicy::SlideBack`] → `[total − W, total)`, full length;
  ///   [`TailPolicy::Partial`] → `[S + H, total)`, ragged, emitted only when it
  ///   reaches [`MIN_SAMPLES`];
  ///   [`TailPolicy::Drop`] → none.
  ///
  /// Under `SlideBack` and `Partial` every sample of the clip lies in at least
  /// one span (`Partial` can leave under 0.09 s at the very end, when the tail
  /// is shorter than the graph accepts); under `Drop` the final `total − S − W`
  /// samples — always fewer than one hop — are not scored.
  ///
  /// # Errors
  /// [`Error::Windowing`] carrying [`WinditError::TooManyWindows`] if the
  /// planned count exceeds [`Self::max_windows`] (`got` is the FULL planned
  /// count, the house convention — windit's own raise aborts at `max + 1`), or
  /// [`WinditError::AllocFailed`] if the span buffer cannot be allocated.
  pub fn spans(&self, total_samples: usize) -> Result<Vec<Span>> {
    // Cap FIRST, before any branch or allocation: the O(1) planned count is the
    // full would-be span count, so an over-cap clip dies here — no buffer, no
    // pushes, no inferences.
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
    let (window, hop) = (self.window_samples as usize, self.hop_samples as usize);
    if total_samples <= window {
      return Ok(vec![Span::new(0, total_samples, window)]);
    }
    // The head: windit's planner under `DropBelowMin(window)` keeps exactly the
    // full-length spans and drops the ragged one it would otherwise stop on.
    let mut spans = windit::plan::WindowPlan::spans(&self.windit_options(), total_samples)?;
    let (last_full_start, uncovered) = self.head_end(total_samples);
    if uncovered {
      let tail = match self.tail {
        TailPolicy::Drop => None,
        TailPolicy::SlideBack => Some((total_samples - window, window)),
        TailPolicy::Partial => {
          let start = last_full_start + hop;
          let len = total_samples - start;
          (len >= MIN_SAMPLES).then_some((start, len))
        }
      };
      if let Some((start, len)) = tail {
        spans
          .try_reserve_exact(1)
          .map_err(|_| Error::Windowing(WinditError::AllocFailed { elements: 1 }))?;
        spans.push(Span::new(start, len, window));
      }
    }
    debug_assert_eq!(
      spans.len(),
      planned,
      "planned_windows drifted from construction"
    );
    Ok(spans)
  }

  /// The windit [`WindowOptions`](windit::plan::WindowOptions) that reproduce
  /// the head: this plan's window and hop, and `DropBelowMin(window)` so only
  /// full-length spans survive — the ragged one windit would stop on is this
  /// module's own business (see [`TailPolicy`]).
  ///
  /// The cap is passed through for defense in depth: the O(1) pre-check in
  /// [`Self::spans`] already refuses an over-cap plan before windit is reached,
  /// so windit's kept count is always `<= max` here and its own raise never
  /// fires — but if `planned_windows` ever undercounted (a bug), windit would
  /// fail typed at `max + 1` rather than over-materialize.
  fn windit_options(&self) -> windit::plan::WindowOptions {
    windit::plan::WindowOptions::new(self.window_samples as usize)
      .with_hop(self.hop_samples as usize)
      .with_tail(windit::plan::TailPolicy::DropBelowMin(
        self.window_samples as usize,
      ))
      .with_max_windows(self.max_windows as usize)
  }
}
