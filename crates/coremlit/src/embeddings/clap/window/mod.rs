//! Overlapped long-audio chunking: [`WindowPlan`] turns a clip length into a
//! list of [`WindowSpan`]s the [`AudioEncoder`](crate::embeddings::clap::AudioEncoder) embeds one
//! at a time, and [`WindowEmbedding`] pairs each resulting embedding with the
//! span it came from (start, real length, and tail-padding-aware coverage) so an
//! [`AggregatePolicy`](crate::embeddings::clap::aggregate::AggregatePolicy) can weight by time,
//! overlap, or coverage.
//!
//! The window length is **fixed** at [`WINDOW_SAMPLES`] (480 000 = 10 s at
//! 48 kHz) — the model's geometry, not a knob. Only the hop and the tail policy
//! are configurable. This module is pure geometry: it holds no audio and touches
//! no model, so its offsets and coverages are hermetically pinned
//! (`tests/`-free — see the sibling `tests.rs`).

use crate::embeddings::clap::{audio::TARGET_SAMPLES, embedding::Embedding};

#[cfg(test)]
mod tests;

/// The fixed inference-window length in samples (480 000 = 10 s at 48 kHz).
///
/// The CLAP HTSAT graph consumes exactly this many samples per inference (via
/// the mel front-end, which `repeatpad`s a shorter tail up to it), so it is the
/// window every [`WindowSpan`] is measured against — the geometry, not a tunable
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
/// kept tail's [`WindowSpan::coverage`] is `< 1.0` — the padding-aware fraction a
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

/// One planned inference window over a caller's sample buffer: where it starts
/// and how many **real** samples it covers.
///
/// Interior windows cover a full [`WINDOW_SAMPLES`]; a kept tail (or a clip
/// shorter than one window) covers fewer, and [`Self::coverage`] reports the
/// padding-aware fraction. The half-open real range is `[start, end)` with
/// `end == start + real_len` — exactly the slice
/// [`AudioEncoder::embed_windows`](crate::embeddings::clap::AudioEncoder::embed_windows) hands to
/// the encoder (which `repeatpad`s it to the fixed window).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSpan {
  start: usize,
  real_len: usize,
}

impl WindowSpan {
  /// A span starting at `start` covering `real_len` real samples.
  ///
  /// `real_len` is expected in `1..=`[`WINDOW_SAMPLES`] (every span
  /// [`WindowPlan::spans`] produces is), so [`Self::coverage`] lands in
  /// `(0, 1]`; a larger `real_len` is not rejected but yields a coverage above 1.
  pub const fn new(start: usize, real_len: usize) -> Self {
    Self { start, real_len }
  }

  /// First sample index this window covers.
  #[inline]
  pub const fn start(&self) -> usize {
    self.start
  }

  /// Number of **real** (non-padding) samples this window covers.
  #[inline]
  pub const fn real_len(&self) -> usize {
    self.real_len
  }

  /// One past the last real sample: `start + real_len`. The end of the slice fed
  /// to the encoder.
  #[inline]
  pub const fn end(&self) -> usize {
    self.start + self.real_len
  }

  /// Real-coverage fraction `real_len / `[`WINDOW_SAMPLES`] — `1.0` for a full
  /// interior window, `< 1.0` for a `repeatpad`-padded tail. The weight a
  /// coverage-aware [`AggregatePolicy`](crate::embeddings::clap::aggregate::AggregatePolicy) uses.
  #[inline]
  pub fn coverage(&self) -> f32 {
    self.real_len as f32 / WINDOW_SAMPLES as f32
  }
}

/// A per-window embedding paired with the [`WindowSpan`] it was computed from —
/// the input unit to an [`AggregatePolicy`](crate::embeddings::clap::aggregate::AggregatePolicy).
///
/// Carrying the span (and thus [`Self::coverage`]) alongside the embedding is
/// what lets a custom policy weight windows by time, overlap, or tail coverage
/// rather than treating them all equally.
#[derive(Debug, Clone)]
pub struct WindowEmbedding {
  embedding: Embedding,
  span: WindowSpan,
}

impl WindowEmbedding {
  /// Pair `embedding` with the `span` it was computed from.
  pub const fn new(embedding: Embedding, span: WindowSpan) -> Self {
    Self { embedding, span }
  }

  /// The window's unit-norm embedding.
  #[inline]
  pub const fn embedding(&self) -> &Embedding {
    &self.embedding
  }

  /// The span this embedding was computed from.
  #[inline]
  pub const fn span(&self) -> WindowSpan {
    self.span
  }

  /// The window's real-coverage fraction — [`WindowSpan::coverage`] of
  /// [`Self::span`].
  #[inline]
  pub fn coverage(&self) -> f32 {
    self.span.coverage()
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
    TailPolicy::DropBelowMin { min_samples } => {
      min_samples > 0 && min_samples as usize <= WINDOW_SAMPLES
    }
  }
}

/// Overlapped-chunking plan: a validated hop and tail policy over the fixed
/// [`WINDOW_SAMPLES`] window (rust-options-pattern).
///
/// [`Self::spans`] is the pure-geometry core — it maps a clip length to the list
/// of [`WindowSpan`]s to embed, with no audio and no model involved, so the
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

  /// Map a clip of `total_samples` to the [`WindowSpan`]s to embed.
  ///
  /// Geometry (window `W` = [`WINDOW_SAMPLES`], hop `H` = [`Self::hop_samples`]):
  ///
  /// - `total_samples == 0` → no windows (an empty clip has nothing to embed).
  /// - `total_samples <= W` → exactly one span `[0, total_samples)`, coverage
  ///   `total_samples / W` (`≤ 1.0`) — so a short clip is embedded once,
  ///   `repeatpad`-padded, regardless of hop, matching textclap's single-chunk
  ///   rule (a smaller hop would otherwise re-embed the same content).
  /// - `total_samples > W` → spans at `0, H, 2H, …` while the start is below
  ///   `total_samples`. Each covers `min(W, total_samples - start)` real
  ///   samples; the interior ones are full windows and the final one may be a
  ///   short tail, kept or dropped per [`Self::tail_policy`]. The first span is
  ///   always kept.
  #[must_use]
  pub fn spans(&self, total_samples: usize) -> Vec<WindowSpan> {
    if total_samples == 0 {
      return Vec::new();
    }
    if total_samples <= WINDOW_SAMPLES {
      return vec![WindowSpan::new(0, total_samples)];
    }

    let hop = self.hop_samples as usize;
    let min_keep = match self.tail {
      TailPolicy::Pad => 1,
      TailPolicy::DropBelowMin { min_samples } => min_samples as usize,
    };

    let mut spans = Vec::new();
    let mut start = 0;
    while start < total_samples {
      let real_len = (total_samples - start).min(WINDOW_SAMPLES);
      // Keep a full interior window always; keep a short tail only if it meets
      // the policy threshold, and always keep the very first span so a clip
      // longer than one window can never plan to nothing.
      if real_len == WINDOW_SAMPLES || real_len >= min_keep || spans.is_empty() {
        spans.push(WindowSpan::new(start, real_len));
      }
      start += hop;
    }
    spans
  }
}
