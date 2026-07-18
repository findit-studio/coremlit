//! Sliding-window geometry: chunk-start scheduling over long audio, plus
//! per-output-frame speaker-count aggregation across overlapping chunk
//! segmentations.
//!
//! This is the pure-geometry layer a future `Extractor` (plan Task 5,
//! `docs/superpowers/plans/2026-07-12-dia-coreml.md`) orchestrates: no
//! CoreML model, no I/O — [`chunk_starts`] schedules where each
//! [`crate::segment::SegmentModel`]/[`crate::embed::EmbedModel`] chunk
//! starts over a long recording, and [`count_from_segmentations`]
//! aggregates the resulting per-chunk [`crate::segment::multilabel`]
//! outputs into the per-output-frame `count` tensor dia's
//! `offline::OfflineInput::new` requires.
//!
//! # `SlidingWindow`: the visibility DECISION
//!
//! dia's own `SlidingWindow` (`diarization/src/reconstruct/algo.rs:
//! 44-113`) is a `pub` type re-exported from `diarization::reconstruct`
//! (`diarization/src/reconstruct/mod.rs:26`) — but its three fields
//! (`start: f64, duration: f64, step: f64`, `algo.rs:46-50`) carry no
//! `pub` keyword at all: it is fully opaque, constructible and readable
//! ONLY through its public `new`/`start`/`duration`/`step`/`with_start`/
//! `with_duration`/`with_step` API (all `pub const fn`, `algo.rs:52-113`).
//!
//! This crate builds WITHOUT the `dia` feature (T1's established
//! contract — see `crates/speakerkit/Cargo.toml`'s optional `dia`
//! dependency), so [`SlidingWindow`] cannot simply BE a re-export of
//! dia's own type: that path only exists when the feature is on. The
//! decision (mirror-struct-always, the shape already established by
//! this crate's other feature-gated boundaries): this module defines
//! its OWN [`SlidingWindow`] — same three private `f64` fields, same
//! public `new`/accessor/builder surface, field-for-field and
//! method-for-method identical to dia's — UNCONDITIONALLY, and adds
//! `dia`-feature-gated `From` conversions in both directions, built
//! entirely through dia's own public accessor API (its fields are
//! private, so there is no other way to reach them). The conversions
//! are lossless and infallible: both types are plain `(f64, f64, f64)`
//! tuples underneath, `Copy`, with no invariants enforced at
//! construction on EITHER side (dia's own `SlidingWindow::new` performs
//! no validation either — validation happens one layer up, at the
//! aggregate/reconstruct function boundaries that consume a
//! `SlidingWindow`; see [`count_from_segmentations`]'s own doc).
//!
//! # Chunking parameters (cited, not trusted from the task brief)
//!
//! - **Chunk length**: 160 000 samples (10 s @ 16 kHz) — dia's
//!   `WINDOW_SAMPLES` (`diarization/src/segment/options.rs:18`), already
//!   pinned in this crate as [`crate::segment::SEG_CHUNK_SAMPLES`]
//!   (`crates/speakerkit/src/segment/mod.rs:126`, verified against the
//!   real `pyannote_segmentation.mlmodelc` contract by T1/T2). This
//!   module reuses that constant rather than redefining it.
//! - **Chunk step default**: 16 000 samples (1 s) — dia's
//!   `OwnedPipelineOptions::new`'s `step_samples` field
//!   (`diarization/src/offline/owned.rs:143`: `step_samples: 16_000, //
//!   1 s — community-1 config`). [`DEFAULT_STEP_SAMPLES`].
//! - **Output-frame duration**: `0.0619375` s — dia's
//!   `PYANNOTE_FRAME_DURATION_S` (`diarization/src/segment/options.rs:
//!   37`, `≈ 991 / 16_000`). [`FRAME_DURATION_S`].
//! - **Output-frame step**: `0.016875` s — dia's `PYANNOTE_FRAME_STEP_S`
//!   (`diarization/src/segment/options.rs:32`, `= 270 / 16_000`).
//!   [`FRAME_STEP_S`].
//! - **Onset default**: `0.5` — dia's `OwnedPipelineOptions::new`'s
//!   `onset` field (`diarization/src/offline/owned.rs:144`) and dia's
//!   own `count_pyannote` community-1 call site
//!   (`diarization/src/aggregate/parity_tests.rs:50`, `0.5, // pyannote
//!   community-1 onset`). [`DEFAULT_ONSET`].
//!
//! # `chunk_starts`: dia's offline chunking rule, and the final-chunk rule
//!
//! Ported from `OwnedDiarizationPipeline::run`'s stage-1 chunk loop
//! (`diarization/src/offline/owned.rs:447-475`), NOT from the streaming
//! segmenter's `segment::window::plan_starts`
//! (`diarization/src/segment/window.rs`) — a DIFFERENT function for a
//! DIFFERENT pipeline (see "Contrast with the streaming planner" below).
//! dia's own doc comment for `ShapeError::StepSamplesExceedsWindow`
//! (`diarization/src/offline/algo.rs:82-89`) independently confirms this
//! is dia's documented, general "owned/streaming chunk planner" contract
//! for the offline side, not an incidental code shape:
//!
//! > "The owned/streaming chunk planners use `start = c * step` and stop
//! > after `(samples.len() - win).div_ceil(step) + 1` chunks."
//!
//! Concretely (`owned.rs:447-451, 466-467`):
//! ```text
//! num_chunks = if total_samples <= WINDOW_SAMPLES { 1 }
//!              else { (total_samples - WINDOW_SAMPLES).div_ceil(step) + 1 };
//! starts[c]  = c * step,  for c in 0..num_chunks
//! ```
//!
//! **Final-chunk rule: dia PADS, never drops, never snaps back.** The
//! grid is perfectly regular — every start is `c * step`, unconditionally
//! — so the LAST chunk's window `[start, start + WINDOW_SAMPLES)` may
//! (and, whenever `step` does not evenly divide `total_samples -
//! WINDOW_SAMPLES`, does) extend past `total_samples`. dia's caller
//! zero-pads whatever falls outside the buffer rather than dropping the
//! chunk or shrinking it (`owned.rs:468-475`: `padded_chunk.fill(0.0)`,
//! then only the in-range `samples[lo..end]` slice is copied over the
//! zeroed buffer). [`chunk_starts`] is pure geometry — it returns ONLY
//! the `c * step` start offsets; zero-padding a short final chunk's
//! samples is left to the caller (a future `Extractor`), exactly as it
//! is `OwnedDiarizationPipeline::run`'s own job one layer above this
//! arithmetic, not the arithmetic's own.
//!
//! **Contrast with the streaming planner.** `segment::window::plan_starts`
//! (`diarization/src/segment/window.rs:1-41`) is a DIFFERENT function
//! for dia's STREAMING segmenter: it anchors a final "tail" window to
//! `total_samples - WINDOW_SAMPLES` so the window NEVER runs past the
//! buffer (no padding needed, at the cost of extra overlap with the
//! previous window). [`chunk_starts`] intentionally does NOT replicate
//! that tail-anchor — the brief's target is the OFFLINE pipeline
//! specifically, and the two rules produce genuinely different start
//! offsets whenever `total_samples - WINDOW_SAMPLES` isn't a multiple of
//! `step` (compare `plan_starts(230_000, 40_000) == [0, 40_000,
//! 70_000]`, `window.rs:76-82`, tail-anchored at `70_000`, against this
//! module's own [`chunk_starts`] on the same input, which is NOT
//! anchored and keeps advancing by the regular `40_000` stride).
//!
//! **`total_samples == 0`**: the formula above, applied literally, gives
//! `num_chunks = 1` (since `0 <= WINDOW_SAMPLES`), i.e. `chunk_starts(0,
//! ..) == vec![0]` — one fully-zero-padded chunk. dia's OWN
//! `OwnedDiarizationPipeline::run` never actually reaches this
//! arithmetic for empty audio: it rejects `samples.is_empty()` ONE LAYER
//! ABOVE, before computing `num_chunks` at all (`owned.rs:369-371`,
//! `Error::Shape(ShapeError::EmptySamples)`). [`chunk_starts`] is a
//! total, `Result`-free function over its documented domain (per the
//! brief's pinned signature) and computes the same well-defined answer
//! the formula gives for any input, including zero; a future
//! `Extractor` is expected to replicate dia's OWN `EmptySamples`
//! rejection at ITS OWN layer (mirroring `owned.rs`'s guard) before ever
//! calling [`chunk_starts`], exactly matching dia's own layering.
//!
//! # `count_from_segmentations`: the hairiest numeric match
//!
//! Pure-Rust, `Vec`-based port of dia's `count_pyannote`
//! (`diarization/src/aggregate/count.rs:579-600`, the infallible
//! wrapper — mirrored here because the brief pins this function's return
//! type to a bare `Vec<u8>`, not a `Result`) delegating to
//! `try_count_pyannote`'s full algorithm
//! (`diarization/src/aggregate/count.rs:607-807`). This is EXACTLY the
//! computation that produces `OfflineInput::new`'s `count` parameter —
//! `OwnedDiarizationPipeline::run` calls `try_count_pyannote` for
//! precisely this purpose (`owned.rs:663-673`) and threads its output
//! straight into `OfflineInput::new` (`owned.rs:677-687`).
//!
//! dia's own module doc (`diarization/src/aggregate/count.rs:1-35`)
//! states the pyannote formula this function mirrors bit-exactly:
//!
//! ```text
//! trimmed = Inference.trim(binarized_segmentations, warm_up=(0.1, 0.1))
//! count = Inference.aggregate(np.sum(trimmed, axis=-1, keepdims=True),
//!                              frames, hamming=False, missing=0.0,
//!                              skip_average=False)
//! count.data = np.rint(count.data).astype(np.uint8)
//! ```
//!
//! — except community-1 overrides `warm_up` to `(0.0, 0.0)`
//! (`count.rs:722-730`), so in practice NO frame is ever trimmed; every
//! frame of every chunk contributes.
//!
//! ## `[c][f][s]` shape
//!
//! `segmentations` is `[num_chunks][num_frames_per_chunk][num_speakers]`
//! flattened row-major, speakers innermost. dia documents this identical
//! layout at BOTH the aggregate boundary (`count.rs:563-564`: "flattened
//! row-major in the `[c][f][s]` order pyannote uses") and the offline
//! boundary (`diarization/src/offline/algo.rs:209-210`: "per-(chunk,
//! frame, speaker) activity flattened `[c][f][s]`") — it is the SAME
//! tensor [`crate::segment::multilabel`] produces per chunk (frame-major,
//! `[frame][slot]`, i.e. one chunk's `[f][s]` slab), so a future
//! `Extractor` concatenates each chunk's `multilabel` output, in chunk
//! order, to build the full `[c][f][s]` buffer this function expects.
//!
//! ## Threshold, combine, and rounding semantics — every rule cited
//!
//! | Sub-decision | This port | dia citation |
//! |---|---|---|
//! | Onset comparison | `v >= onset` (inclusive) | `count.rs:715`: `if v >= onset { 1.0_f64 } else { 0.0_f64 }` |
//! | Per-chunk frame weight | uniform `1.0`, no trim (community-1 `warm_up=(0,0)`) | `count.rs:722-734` |
//! | Combine across overlapping chunks | unweighted SUM of active-speaker counts ÷ SUM of contributing-chunk count (arithmetic mean, NOT Hamming-weighted, NOT max) | `count.rs:762-777` (accumulate), `count.rs:792-801` (divide) |
//! | Chunk→output-frame index | `start_frame(c) = round_ties_even(c * chunk_step / frame_step) as i64`; `ofr = start_frame(c) + f` | `count.rs:736-747, 763-764` |
//! | Output-frame count | `round_ties_even((chunk_duration + (num_chunks-1)*chunk_step) / frame_step) + 1` | `count.rs:486-547` (`num_output_frames_pyannote`) |
//! | Rounding | `round_ties_even` (banker's rounding) of the mean | `count.rs:797` |
//! | Clamp | `[0.0, 255.0]` before the `as u8` cast | `count.rs:797` |
//! | Zero-coverage cells | `count[t] = 0`, not NaN / div-by-zero | `count.rs:792-801`, pyannote's `missing=0.0` |
//!
//! `hamming_aggregate` (`count.rs:211-263`) is a DIFFERENT function for a
//! DIFFERENT tensor — per-speaker activation aggregation during
//! RECONSTRUCTION, `hamming=True, skip_average=True` — and is
//! deliberately NOT ported here; it has nothing to do with
//! `OfflineInput::new`'s `count` parameter (see dia's own module doc,
//! `count.rs:1-46`, "Importantly, this is NOT the same aggregation...").
//!
//! ## Who validates dims
//!
//! [`count_from_segmentations`] panics (never returns `Result` — matching
//! the brief's pinned return type and this crate's established
//! assert-on-shape-mismatch convention,
//! [`crate::segment::multilabel`]) on the SAME preconditions dia's
//! infallible `count_pyannote` wrapper inherits via its own
//! `.expect(..)` on `try_count_pyannote`'s `Result` (`count.rs:589-600`):
//! `num_chunks`/`num_frames_per_chunk`/`num_speakers` all `>= 1`
//! (`count.rs:624-632`), `chunks_sw`/`frames_sw`'s duration and step all
//! positive and finite (`count.rs:637-648`), `onset` finite
//! (`count.rs:649-651`), `segmentations.len() == num_chunks *
//! num_frames_per_chunk * num_speakers` with the product itself
//! overflow-checked (`count.rs:652-658`), and every `segmentations`
//! value finite (`count.rs:659-671` — an unchecked NaN/Inf cell would
//! otherwise compare `false` against `onset` and silently masquerade as
//! "inactive speaker" rather than surfacing the corrupted input). NOT
//! ported: `try_count_pyannote`'s `MAX_OUTPUT_FRAMES` overflow/OOM cap
//! and its `SpillBytesMut` file-backed scratch-buffer fallback
//! (`count.rs:41-55, 690-761`) — both are dia's OWN hardening against
//! multi-gigabyte adversarial inputs via its `ops::spill` subsystem,
//! which this crate does not depend on; out of scope for porting the
//! numeric ALGORITHM itself, which is this task's explicit focus.
//!
//! Additionally guarded: the output-frame-count computation itself
//! (`last_chunk_end / frame_step`, rounded and cast to `usize`) is
//! checked for overflow rather than left to saturate — see
//! `try_num_output_frames`'s own doc (private to this module) for the
//! exact bound, ported from dia's `ShapeError::OutputFrameCountOverflow`
//! guard (`count.rs:504-509, 522-533`).
//!
//! ## Downstream re-validation: what dia actually re-checks
//!
//! This function's output — the `count` tensor, plus
//! [`chunk_sliding_window`]/[`frame_sliding_window`]'s `SlidingWindow`
//! values — ultimately feeds dia's `offline::OfflineInput::new`. That
//! constructor (`diarization/src/offline/algo.rs:216-248`) is a bare
//! `pub const fn` performing ONLY field assignment: it validates
//! NOTHING. The real re-validation happens one call further in, inside
//! `offline::diarize_offline` itself (`diarization/src/offline/algo.rs:
//! 494-596`): it re-checks `num_chunks`/`num_frames_per_chunk`/
//! `num_speakers` are non-zero, re-checks the `raw_embeddings`/
//! `segmentations` length products (overflow-checked), and re-checks
//! `count.len() == num_output_frames` plus a `MAX_COUNT_PER_FRAME` cap —
//! mirroring `reconstruct`'s own boundary checks so a malformed `count`
//! tensor fails before the expensive AHC/VBx/PLDA stages run rather than
//! after. `reconstruct::reconstruct`
//! (`diarization/src/reconstruct/algo.rs:400-406`) separately re-checks
//! `chunks_sw`/`frames_sw` finiteness AND positivity (`w.duration`/
//! `step`/`start` `.is_finite()`, `w.duration`/`step` `> 0.0`).
//!
//! **`onset` is the one exception — nothing downstream re-checks it.**
//! dia's `OfflineInput` doesn't even carry an `onset` field: by the time
//! execution reaches `OfflineInput::new`, onset's effect is already
//! pre-baked into the caller-supplied `segmentations` (hard-zeroed
//! column-wise against `>= onset`, `diarization/src/offline/owned.rs:
//! 544-570`) and `count` (thresholded internally by `try_count_pyannote`,
//! `count.rs:715`) tensors. dia validates the raw `onset` scalar at three
//! points, all upstream of or INSIDE this computation: at
//! `OwnedPipelineOptions::with_onset` (builder-time panic,
//! `owned.rs:222-235`), at `OwnedDiarizationPipeline::run`'s own
//! defense-in-depth re-check (`owned.rs:388-392`), and inside
//! `try_count_pyannote` itself, which re-checks `onset.is_finite()`
//! (`count.rs:649-651`) — the exact check this function's own
//! `onset.is_finite()` assert ports. It is never re-validated DOWNSTREAM
//! of `OfflineInput::new` (dia's `OfflineInput` carries no `onset` field
//! at all). So in THIS crate's port, [`count_from_segmentations`]'s own
//! `onset.is_finite()` assert is LOAD-BEARING: it is the port of dia's
//! last-line `try_count_pyannote` finiteness check, and nothing
//! DOWNSTREAM of this function would catch a non-finite `onset` otherwise.

use crate::segment::SEG_CHUNK_SAMPLES;

/// Audio sample rate this module's second-based geometry assumes —
/// 16 kHz. Matches dia's `SAMPLE_RATE_HZ`
/// (`diarization/src/segment/options.rs:11`) and this crate's own model
/// contracts ([`crate::segment::SEG_CHUNK_SAMPLES`] is exactly `10 *
/// SAMPLE_RATE_HZ`).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Pyannote community-1 output-frame receptive-field duration, in
/// seconds. Matches dia's `PYANNOTE_FRAME_DURATION_S`
/// (`diarization/src/segment/options.rs:37`, `≈ 991 / 16_000`).
pub const FRAME_DURATION_S: f64 = 0.0619375;

/// Pyannote community-1 output-frame stride, in seconds. Matches dia's
/// `PYANNOTE_FRAME_STEP_S` (`diarization/src/segment/options.rs:32`,
/// `= 270 / 16_000`).
pub const FRAME_STEP_S: f64 = 0.016875;

/// Chunk duration in seconds: [`crate::segment::SEG_CHUNK_SAMPLES`]
/// samples at [`SAMPLE_RATE_HZ`] = `10.0` s. Matches dia's own
/// derivation (`diarization/src/offline/owned.rs:653`: `WINDOW_SAMPLES
/// as f64 / SAMPLE_RATE_HZ as f64`).
pub const CHUNK_DURATION_S: f64 = SEG_CHUNK_SAMPLES as f64 / SAMPLE_RATE_HZ as f64;

/// Default [`WindowOptions::step_samples`] — matches dia's
/// `OwnedPipelineOptions::new`'s `step_samples` default
/// (`diarization/src/offline/owned.rs:143`).
pub const DEFAULT_STEP_SAMPLES: u32 = 16_000;

/// Default [`WindowOptions::onset`] — matches dia's
/// `OwnedPipelineOptions::new`'s `onset` default
/// (`diarization/src/offline/owned.rs:144`).
pub const DEFAULT_ONSET: f32 = 0.5;

#[cfg(feature = "serde")]
fn default_step_samples() -> u32 {
  DEFAULT_STEP_SAMPLES
}
#[cfg(feature = "serde")]
fn default_onset() -> f32 {
  DEFAULT_ONSET
}

/// Mirror of dia's `SlidingWindow` (`diarization/src/reconstruct/algo.rs:
/// 44-113`): `(start, duration, step)`, all in seconds — see the module
/// doc's "`SlidingWindow`: the visibility DECISION" section for why this
/// crate keeps its own copy of this type rather than depending on dia's
/// directly, even where the `dia` feature is off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlidingWindow {
  start: f64,
  duration: f64,
  step: f64,
}

impl SlidingWindow {
  /// Construct a sliding window. All values in seconds. Performs no
  /// validation — matches dia's own `SlidingWindow::new`
  /// (`diarization/src/reconstruct/algo.rs:52-60`), which is likewise
  /// unchecked; validation happens at the aggregate/reconstruct function
  /// boundaries that consume a `SlidingWindow`, not at construction (see
  /// [`count_from_segmentations`]'s own "Who validates dims" doc).
  pub const fn new(start: f64, duration: f64, step: f64) -> Self {
    Self {
      start,
      duration,
      step,
    }
  }

  /// First-frame center offset, in seconds. Matches dia's
  /// `SlidingWindow::start` (`diarization/src/reconstruct/algo.rs:
  /// 62-65`).
  #[inline(always)]
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// Per-frame receptive-field length, in seconds. Matches dia's
  /// `SlidingWindow::duration` (`diarization/src/reconstruct/algo.rs:
  /// 67-70`).
  #[inline(always)]
  pub const fn duration(&self) -> f64 {
    self.duration
  }

  /// Stride between consecutive frame centers, in seconds. Matches dia's
  /// `SlidingWindow::step` (`diarization/src/reconstruct/algo.rs:72-75`).
  #[inline(always)]
  pub const fn step(&self) -> f64 {
    self.step
  }

  /// Builder form of [`Self::with_start`]'s underlying field-replace.
  /// Matches dia's `SlidingWindow::with_start`
  /// (`diarization/src/reconstruct/algo.rs:77-82`).
  #[must_use]
  pub const fn with_start(mut self, start: f64) -> Self {
    self.start = start;
    self
  }

  /// Builder: replace [`Self::duration`]. Matches dia's
  /// `SlidingWindow::with_duration`
  /// (`diarization/src/reconstruct/algo.rs:84-89`).
  #[must_use]
  pub const fn with_duration(mut self, duration: f64) -> Self {
    self.duration = duration;
    self
  }

  /// Builder: replace [`Self::step`]. Matches dia's
  /// `SlidingWindow::with_step` (`diarization/src/reconstruct/algo.rs:
  /// 91-96`).
  #[must_use]
  pub const fn with_step(mut self, step: f64) -> Self {
    self.step = step;
    self
  }
}

/// `dia`-feature-gated conversion INTO dia's own `SlidingWindow` — see
/// the module doc's "`SlidingWindow`: the visibility DECISION" section.
/// Lossless and infallible: both types are unchecked `(f64, f64, f64)`
/// tuples.
#[cfg(feature = "dia")]
impl From<SlidingWindow> for dia::reconstruct::SlidingWindow {
  fn from(value: SlidingWindow) -> Self {
    Self::new(value.start, value.duration, value.step)
  }
}

/// `dia`-feature-gated conversion FROM dia's own `SlidingWindow` — the
/// reverse of the `From` impl above. Built entirely through dia's public
/// `start`/`duration`/`step` accessors: dia's fields are private, so
/// there is no other way to reach them.
#[cfg(feature = "dia")]
impl From<dia::reconstruct::SlidingWindow> for SlidingWindow {
  fn from(value: dia::reconstruct::SlidingWindow) -> Self {
    Self::new(value.start(), value.duration(), value.step())
  }
}

/// dia's onset validity predicate: finite and in `(0.0, 1.0]` (lower
/// bound EXCLUSIVE, upper bound inclusive). Exact copy of dia's
/// `check_onset` (`diarization/src/offline/owned.rs:52-56`), including
/// its const-fn-safe NaN check: `f32::is_finite` is not yet usable in a
/// `const fn` at this crate's MSRV (dia's own comment at
/// `diarization/src/segment/options.rs:54-56` explains why: it awaits
/// the unstable `const_float_classify` feature, still unstable at this
/// crate's and dia's now-shared `rust-version` floor), so the check is
/// phrased by hand via the `v != v` NaN idiom plus direct comparisons.
///
/// `pub(crate)` so [`crate::extract::Extractor::extract`]'s own onset
/// preflight can reuse this exact predicate (mirroring dia's
/// `OwnedDiarizationPipeline::run` re-checking `check_onset` at
/// `owned.rs:388-392`) rather than re-deriving the range test.
#[inline]
pub(crate) const fn check_onset(v: f32) -> bool {
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && v > 0.0 && v <= 1.0
}

/// [`WindowOptions::step_samples`]'s validity predicate: `> 0` and `<=
/// SEG_CHUNK_SAMPLES` — the exact invariant [`WindowOptions::set_step_samples`]
/// asserts. Named here (mirroring [`check_onset`]) so the serde deserialize
/// path can enforce the SAME rule the checked setter does, rather than
/// bypassing it. A `step_samples > SEG_CHUNK_SAMPLES` opens silent audio gaps
/// of `step - SEG_CHUNK_SAMPLES` samples between consecutive chunks that no
/// window covers (see [`WindowOptions::set_step_samples`]'s own doc); `0` would
/// hang [`chunk_starts`]'s `div_ceil`. `serde`-gated: it exists to hold the
/// deserialize path to this invariant, and the builder setters carry their own
/// (message-distinct) asserts, so nothing references it without `serde`.
#[cfg(feature = "serde")]
#[inline]
const fn check_step_samples(v: u32) -> bool {
  v > 0 && v <= SEG_CHUNK_SAMPLES as u32
}

/// Construction options for [`chunk_starts`] and (via
/// [`chunk_sliding_window`]) [`count_from_segmentations`]
/// (rust-options-pattern). Mirrors dia's `OwnedPipelineOptions`'s
/// identical `step_samples`/`onset` pair
/// (`diarization/src/offline/owned.rs:75-101`) — same defaults, same
/// validated ranges — but scoped to only the two fields this crate's
/// pure-geometry layer needs (dia's clustering/reconstruction
/// hyperparameters on that same options type have no analog here).
///
/// # Validated deserialization
///
/// `Deserialize` routes through a private `WindowOptionsRepr` via
/// `serde(try_from)`, so a config-file or hand-written `WindowOptions` is held
/// to the SAME `step_samples`/`onset` invariants the checked setters enforce.
/// Deriving
/// `Deserialize` directly on the fields would bypass [`Self::set_step_samples`]
/// entirely: `{"step_samples":200000}` on 320 000 samples would deserialize
/// silently and then produce windows `[0, 160000) + [200000, 360000)`, omitting
/// `[160000, 200000)` — audio dropped with no error (L1). Invalid input now
/// fails to deserialize instead.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "WindowOptionsRepr"))]
pub struct WindowOptions {
  step_samples: u32,
  onset: f32,
}

/// The plain wire form [`WindowOptions`]'s `Deserialize` deserializes FIRST
/// (carrying the same field defaults), before [`WindowOptions::try_from`]
/// applies the range checks. Its whole purpose is to make the validated
/// setters unbypassable via serde (see [`WindowOptions`]'s "Validated
/// deserialization" doc) — it is never constructed or exposed otherwise.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct WindowOptionsRepr {
  #[serde(default = "default_step_samples")]
  step_samples: u32,
  #[serde(default = "default_onset")]
  onset: f32,
}

#[cfg(feature = "serde")]
impl TryFrom<WindowOptionsRepr> for WindowOptions {
  type Error = String;

  /// Applies [`check_step_samples`] and [`check_onset`] — the exact invariants
  /// [`WindowOptions::set_step_samples`]/[`WindowOptions::set_onset`] assert —
  /// as fallible checks, so a serde-deserialized value can never construct the
  /// audio-dropping geometry (or degenerate onset) the builders reject.
  fn try_from(r: WindowOptionsRepr) -> Result<Self, Self::Error> {
    if !check_step_samples(r.step_samples) {
      return Err(format!(
        "step_samples ({}) must be > 0 and <= SEG_CHUNK_SAMPLES ({SEG_CHUNK_SAMPLES})",
        r.step_samples
      ));
    }
    if !check_onset(r.onset) {
      return Err(format!("onset ({}) must be finite in (0.0, 1.0]", r.onset));
    }
    Ok(Self {
      step_samples: r.step_samples,
      onset: r.onset,
    })
  }
}

impl Default for WindowOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl WindowOptions {
  /// Options matching dia's own community-1 defaults:
  /// [`DEFAULT_STEP_SAMPLES`] (16 000, 1 s) and [`DEFAULT_ONSET`] (0.5).
  pub const fn new() -> Self {
    Self {
      step_samples: DEFAULT_STEP_SAMPLES,
      onset: DEFAULT_ONSET,
    }
  }

  /// Sliding-window step in samples between successive chunk starts —
  /// see [`chunk_starts`].
  #[inline(always)]
  pub const fn step_samples(&self) -> u32 {
    self.step_samples
  }

  /// Onset threshold [`count_from_segmentations`] compares hard
  /// segmentation values against (`v >= onset`, inclusive).
  #[inline(always)]
  pub const fn onset(&self) -> f32 {
    self.onset
  }

  /// Builder form of [`Self::set_step_samples`].
  ///
  /// # Panics
  /// As [`Self::set_step_samples`].
  #[must_use]
  pub const fn with_step_samples(mut self, v: u32) -> Self {
    self.set_step_samples(v);
    self
  }

  /// Sets [`Self::step_samples`] in place.
  ///
  /// # Panics
  /// Panics if `v == 0` or `v > SEG_CHUNK_SAMPLES` (160 000) — mirrors
  /// dia's identical validation on the SAME field, present in TWO of
  /// dia's own options types: `SegmentOptions::with_step_samples`
  /// (`diarization/src/segment/options.rs:181-197`) and
  /// `OwnedPipelineOptions::with_step_samples`
  /// (`diarization/src/offline/owned.rs:205-221`). Both document the
  /// same failure modes a violation causes: zero would hang the chunk
  /// planner (a divide-by-zero inside [`chunk_starts`]'s own
  /// `div_ceil`); `step > SEG_CHUNK_SAMPLES` causes silent audio gaps of
  /// `step - SEG_CHUNK_SAMPLES` samples between consecutive chunks that
  /// no chunk's `[start, start + SEG_CHUNK_SAMPLES)` window ever covers
  /// (`diarization/src/offline/algo.rs:82-89`,
  /// `ShapeError::StepSamplesExceedsWindow`'s own doc).
  pub const fn set_step_samples(&mut self, v: u32) -> &mut Self {
    assert!(v > 0, "step_samples must be > 0");
    assert!(
      v <= SEG_CHUNK_SAMPLES as u32,
      "step_samples must be <= SEG_CHUNK_SAMPLES (160_000)"
    );
    self.step_samples = v;
    self
  }

  /// Builder form of [`Self::set_onset`].
  ///
  /// # Panics
  /// As [`Self::set_onset`].
  #[must_use]
  pub const fn with_onset(mut self, v: f32) -> Self {
    self.set_onset(v);
    self
  }

  /// Sets [`Self::onset`] in place.
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `(0.0, 1.0]` — mirrors dia's
  /// `OwnedPipelineOptions::with_onset`
  /// (`diarization/src/offline/owned.rs:222-235`) exactly, including its
  /// EXCLUSIVE lower bound: `0.0` itself is invalid, because the hard
  /// segmentation mask `seg >= onset` would then treat every zero cell
  /// as "active", corrupting frame masks, embeddings, and counts alike
  /// (see `check_onset`'s own doc).
  pub const fn set_onset(&mut self, v: f32) -> &mut Self {
    assert!(check_onset(v), "onset must be finite in (0.0, 1.0]");
    self.onset = v;
    self
  }
}

/// Chunk-start sample offsets over `total_samples` — dia's offline
/// pipeline's exact chunking arithmetic. See the module doc's
/// "`chunk_starts`: dia's offline chunking rule, and the final-chunk
/// rule" section for the full derivation, citations, and edge-case
/// decisions (including `total_samples == 0` and the deliberate
/// contrast with the streaming `plan_starts` planner).
///
/// Each returned `starts[c]` is a chunk's start sample; every chunk
/// spans `[starts[c], starts[c] + SEG_CHUNK_SAMPLES)`. The LAST chunk's
/// span may extend past `total_samples` — zero-padding that overhang is
/// the caller's job (a future `Extractor`), not this function's: this
/// is pure geometry, matching dia's own `OwnedDiarizationPipeline::run`,
/// which computes the identical `num_chunks`/`start = c * step`
/// arithmetic BEFORE ever touching sample data
/// (`diarization/src/offline/owned.rs:447-451, 466-467`).
///
/// # Panics
/// Panics if `options.step_samples() == 0`. [`WindowOptions`]'s own
/// setters already reject this before it can reach this function
/// through the builder path (see [`WindowOptions::set_step_samples`]),
/// so this is defense-in-depth against a `WindowOptions` value built
/// some other way (e.g. `serde`-deserialized, which bypasses the
/// builder) — mirroring both the identical `assert!(step > 0, ..)` in
/// dia's OWN sibling geometry function, `segment::window::plan_starts`
/// (`diarization/src/segment/window.rs:24`), and the equivalent
/// defense-in-depth re-check `OwnedDiarizationPipeline::run` itself
/// performs one layer up (`diarization/src/offline/owned.rs:374-376`,
/// `ShapeError::ZeroStepSamples`) for the identical reason.
pub fn chunk_starts(total_samples: usize, options: &WindowOptions) -> Vec<usize> {
  let step = options.step_samples() as usize;
  assert!(step > 0, "step_samples must be > 0");

  let num_chunks = if total_samples <= SEG_CHUNK_SAMPLES {
    1
  } else {
    (total_samples - SEG_CHUNK_SAMPLES).div_ceil(step) + 1
  };

  (0..num_chunks).map(|c| c * step).collect()
}

/// The chunk-grid [`SlidingWindow`] matching `options`: `start = 0.0`,
/// `duration = `[`CHUNK_DURATION_S`]` (10.0 s), `step =
/// options.step_samples() / `[`SAMPLE_RATE_HZ`]. Matches dia's own
/// `chunks_sw` construction (`diarization/src/offline/owned.rs:
/// 653-655`).
pub fn chunk_sliding_window(options: &WindowOptions) -> SlidingWindow {
  let step_s = f64::from(options.step_samples()) / f64::from(SAMPLE_RATE_HZ);
  SlidingWindow::new(0.0, CHUNK_DURATION_S, step_s)
}

/// The pyannote community-1 output-frame-grid [`SlidingWindow`]: fixed
/// `start = 0.0`, `duration = `[`FRAME_DURATION_S`]` (`0.0619375` s),
/// `step = `[`FRAME_STEP_S`]` (`0.016875` s) — no [`WindowOptions`]
/// dependency, since dia treats this grid as a FIXED property of the
/// segmentation model, never a tunable. Matches dia's own
/// `frames_sw_template` construction
/// (`diarization/src/offline/owned.rs:656-657`).
///
/// This is also bit-identical to what dia's own `try_count_pyannote`
/// returns as ITS OWN `frames_sw` (`count.rs:804`:
/// `SlidingWindow::new(0.0, frame_duration, frame_step)`, derived from
/// whatever `frames_sw_template` was passed in) whenever the template's
/// `start` was already `0.0` going in — which dia's own call site
/// always satisfies (`owned.rs:656-657`). So this function's return
/// value can stand in for both "the template to pass in" and "the grid
/// `count_from_segmentations` effectively describes its own output
/// against".
pub fn frame_sliding_window() -> SlidingWindow {
  SlidingWindow::new(0.0, FRAME_DURATION_S, FRAME_STEP_S)
}

/// Internal failure for [`try_num_output_frames`]'s guard, surfaced by
/// [`try_count_from_segmentations`]. Not part of this crate's public
/// error taxonomy (`crate::error::{ModelError, InferError, ExtractError}`,
/// design spec §5) — [`count_from_segmentations`]'s established public
/// contract stays `Result`-free (module doc, "Who validates dims"), so
/// this type exists purely to make the overflow guard's control flow, and
/// this module's regression tests, typed rather than an unguarded
/// arithmetic op. [`crate::extract::Extractor::extract`] converts it to
/// `crate::error::ExtractError::OutputFrameCountOverflow` via an
/// exhaustive manual match (NOT a `From` impl — see that variant's own
/// doc for why); it stays crate-private either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum WindowError {
  /// `num_output_frames` would not fit in `usize` — see
  /// [`try_num_output_frames`]'s own doc for the exact bound. Message
  /// text matches dia's own `ShapeError::OutputFrameCountOverflow`
  /// (`diarization/src/aggregate/count.rs:114-117`).
  #[error(
    "num_output_frames overflows usize (chunk_duration / frame_step too large \
     to represent or saturated past usize::MAX)"
  )]
  OutputFrameCountOverflow,
}

/// Guarded computation of [`count_from_segmentations`]'s
/// `num_output_frames`. Ports dia's `try_num_output_frames_pyannote`'s
/// overflow check (`diarization/src/aggregate/count.rs:510-547`)
/// bit-for-bit, scoped to the two inputs this function's one call site
/// has already validated (`last_chunk_end`, `frame_step`) — dia's own
/// `num_chunks == 0` / `frame_step` finiteness-and-positivity re-checks
/// (`count.rs:516-521`) are not repeated here because
/// [`count_from_segmentations`] already asserts both before computing
/// `last_chunk_end` (unreachable at this call site). This helper's whole
/// job is the ONE check dia has that this crate's port was missing: the
/// division/rounding/cast sequence itself.
///
/// dia's exact bound (`count.rs:522-533`):
/// ```text
/// let frames_f = (last_chunk_end / frame_step).round_ties_even();
/// if !frames_f.is_finite() || frames_f < 0.0 || frames_f >= usize::MAX as f64 {
///     return Err(ShapeError::OutputFrameCountOverflow);
/// }
/// let n = (frames_f as usize).checked_add(1).ok_or(ShapeError::OutputFrameCountOverflow)?;
/// ```
///
/// Two adversarial-but-finite geometries this guards against (both
/// reachable through the public, unchecked [`SlidingWindow::new`]):
/// `chunk_duration = 1e300, frame_step = 1e-300` divides straight to
/// `+inf` (caught by `!frames_f.is_finite()`); `chunk_duration = 1e20,
/// frame_step = 1.0` divides to a large but FINITE value that would
/// still saturate `as usize` to a value near `usize::MAX` (caught by
/// `frames_f >= usize::MAX as f64` — dia's own comment notes `usize::MAX
/// as f64` rounds UP to exactly `2.0f64.powi(64)`, so this comparison
/// stays monotonic). The trailing `checked_add(1)` is defense-in-depth
/// for the residual case where `frames_f` is finite and just under that
/// threshold, yet still casts to a `usize` within 1 of `usize::MAX`
/// (`f64`'s ~53-bit mantissa has no integer resolution at this
/// magnitude, so "just under the threshold" can still land exactly on
/// `usize::MAX` after the cast).
///
/// # Errors
/// [`WindowError::OutputFrameCountOverflow`] if `(last_chunk_end /
/// frame_step).round_ties_even()` is non-finite, negative, `>=
/// usize::MAX as f64`, or whose `+ 1` would overflow `usize`.
fn try_num_output_frames(last_chunk_end: f64, frame_step: f64) -> Result<usize, WindowError> {
  let frames_f = (last_chunk_end / frame_step).round_ties_even();
  if !frames_f.is_finite() || frames_f < 0.0 || frames_f >= usize::MAX as f64 {
    return Err(WindowError::OutputFrameCountOverflow);
  }
  (frames_f as usize)
    .checked_add(1)
    .ok_or(WindowError::OutputFrameCountOverflow)
}

/// dia's `count_pyannote` (`diarization/src/aggregate/count.rs:
/// 579-807`) ported to a plain, `Vec`-based, `Result`-free function —
/// see the module doc's "`count_from_segmentations`: the hairiest
/// numeric match" section for the full semantics table, every
/// threshold/combine/rounding rule cited against dia's own source, and
/// exactly which preconditions this function validates.
///
/// Thin wrapper over `try_count_from_segmentations` (private to this
/// module): it unwraps the one typed failure that variant can return
/// (the derived `num_output_frames`
/// overflowing `usize`) into the panic direct callers already contract
/// for — mirroring dia's own infallible `count_pyannote` wrapper, which
/// likewise `.expect(..)`s its fallible `try_count_pyannote`
/// (`count.rs:589-600`). All shape/precondition validation lives in the
/// try_ variant.
///
/// # Panics
/// See the module doc's "Who validates dims" section: panics if
/// `num_chunks`/`num_frames_per_chunk`/`num_speakers` is `0`, if
/// `chunks_sw`/`frames_sw`'s duration or step is non-positive or
/// non-finite, if `onset` is non-finite, if `segmentations.len() !=
/// num_chunks * num_frames_per_chunk * num_speakers` (or that product
/// overflows `usize`), if any `segmentations` value is NaN/infinite, or
/// if the derived `num_output_frames` would not fit in `usize` (see
/// `try_num_output_frames`, private to this module).
pub fn count_from_segmentations(
  segmentations: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_speakers: usize,
  onset: f32,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Vec<u8> {
  try_count_from_segmentations(
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    onset,
    chunks_sw,
    frames_sw,
  )
  .expect("num_output_frames must fit in usize")
}

/// Fallible core of [`count_from_segmentations`]: identical body and
/// identical shape/precondition asserts, but returns the
/// `num_output_frames`-overflow guard as a typed
/// [`WindowError::OutputFrameCountOverflow`] instead of unwrapping it.
///
/// The seam exists so [`crate::extract::Extractor::extract`] can convert
/// that one overflow case into its own `crate::error::ExtractError`
/// (a `Result`-typed public API) while direct callers of the public
/// [`count_from_segmentations`] keep the identical panic contract. Every
/// OTHER precondition here stays an `assert!` rather than a `Result` arm:
/// those are invariants `extract` already guarantees at its own boundary
/// before calling this (`num_chunks >= 1` via [`chunk_starts`],
/// length-consistent `segmentations` it assembled itself, an `onset` it
/// already ran through [`check_onset`]), so surfacing them as typed
/// errors would add unreachable variants to `WindowError` and to
/// `ExtractError`'s match.
///
/// # Panics
/// On the SAME shape/precondition violations as
/// [`count_from_segmentations`] — see that function and the module doc's
/// "Who validates dims" section.
///
/// # Errors
/// [`WindowError::OutputFrameCountOverflow`] if the derived
/// `num_output_frames` would not fit in `usize` (see
/// `try_num_output_frames`).
pub(crate) fn try_count_from_segmentations(
  segmentations: &[f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_speakers: usize,
  onset: f32,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Result<Vec<u8>, WindowError> {
  assert!(num_chunks > 0, "num_chunks must be at least 1");
  assert!(
    num_frames_per_chunk > 0,
    "num_frames_per_chunk must be at least 1"
  );
  assert!(num_speakers > 0, "num_speakers must be at least 1");

  let chunk_duration = chunks_sw.duration();
  let chunk_step = chunks_sw.step();
  let frame_duration = frames_sw.duration();
  let frame_step = frames_sw.step();
  assert!(
    chunk_duration.is_finite() && chunk_duration > 0.0,
    "chunks_sw.duration() must be a positive finite scalar"
  );
  assert!(
    chunk_step.is_finite() && chunk_step > 0.0,
    "chunks_sw.step() must be a positive finite scalar"
  );
  assert!(
    frame_duration.is_finite() && frame_duration > 0.0,
    "frames_sw.duration() must be a positive finite scalar"
  );
  assert!(
    frame_step.is_finite() && frame_step > 0.0,
    "frames_sw.step() must be a positive finite scalar"
  );
  assert!(onset.is_finite(), "onset must be finite");

  let expected = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_speakers));
  assert_eq!(
    Some(segmentations.len()),
    expected,
    "segmentations.len() must equal num_chunks * num_frames_per_chunk * num_speakers"
  );
  assert!(
    segmentations.iter().all(|v| v.is_finite()),
    "segmentations must not contain NaN/infinite values"
  );

  let onset = f64::from(onset);

  // ── 1. Per-(chunk, frame) integer active-speaker count ──────────
  // `v >= onset`, inclusive — count.rs:715.
  let mut chunk_count = vec![0.0f64; num_chunks * num_frames_per_chunk];
  for c in 0..num_chunks {
    for f in 0..num_frames_per_chunk {
      let mut active = 0.0f64;
      for s in 0..num_speakers {
        let v = segmentations[(c * num_frames_per_chunk + f) * num_speakers + s];
        if v >= onset {
          active += 1.0;
        }
      }
      chunk_count[c * num_frames_per_chunk + f] = active;
    }
  }

  // ── 2. Output-frame count — count.rs:486-547 ─────────────────────
  let last_chunk_end = chunk_duration + (num_chunks - 1) as f64 * chunk_step;
  let num_output_frames = try_num_output_frames(last_chunk_end, frame_step)?;

  // ── 3. Aggregate: uniform-weight sum + covering-chunk count ──────
  // count.rs:762-777.
  let mut aggregated = vec![0.0f64; num_output_frames];
  let mut overlapping_count = vec![0.0f64; num_output_frames];
  for c in 0..num_chunks {
    let chunk_start_t = c as f64 * chunk_step;
    let start_frame = (chunk_start_t / frame_step).round_ties_even() as i64;
    for f in 0..num_frames_per_chunk {
      let ofr = start_frame + f as i64;
      if ofr < 0 || (ofr as usize) >= num_output_frames {
        continue;
      }
      let ofr = ofr as usize;
      aggregated[ofr] += chunk_count[c * num_frames_per_chunk + f];
      overlapping_count[ofr] += 1.0;
    }
  }

  // ── 4. count[t] = round(aggregated[t] / overlapping_count[t]) ────
  // count.rs:792-801: missing=0.0 for zero-coverage cells.
  let epsilon = 1e-12_f64;
  Ok(
    (0..num_output_frames)
      .map(|t| {
        if overlapping_count[t] > 0.0 {
          let avg = aggregated[t] / overlapping_count[t].max(epsilon);
          avg.round_ties_even().clamp(0.0, u8::MAX as f64) as u8
        } else {
          0
        }
      })
      .collect(),
  )
}

#[cfg(test)]
mod tests;
