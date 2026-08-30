//! The extraction bridge: run segmentation + embedding over a whole clip
//! and assemble the exact tensor set diaric's offline diarizer consumes.
//!
//! [`Extractor::extract`] is the composition layer over Tasks 2-4
//! ([`crate::audio::speaker::segment`], [`crate::audio::speaker::embed`], [`crate::audio::speaker::window`]): it ports the
//! data-plane of dia's `OwnedDiarizationPipeline::run`
//! (`diarization/src/offline/owned.rs:361-697`) — everything from the
//! input guards through the `count` tensor — stopping exactly where dia
//! hands off to `diarize_offline`. Its output, [`Extraction`], exposes
//! precisely `diaric::offline::OfflineInput::new`'s parameter list
//! (`diarization/src/offline/algo.rs:206-227`) and converts into it
//! directly (`Extraction::into_offline_input`) — `diaric` is a runtime
//! dependency, so that bridge (and the clustering it feeds) is always
//! available.
//!
//! # Stage structure (ported from `owned.rs`)
//!
//! 1. **Input guards** (`owned.rs:369-393`): empty samples, `step_samples`
//!    range, and `onset` range — see [`Extractor::extract`]'s own step
//!    list. One guard has no dia analog: [`crate::audio::speaker::error::ExtractError::FrameCountMismatch`].
//! 2. **Chunk grid + zero-padding** (`owned.rs:447-475`): [`crate::audio::speaker::window::chunk_starts`]
//!    schedules `start = c * step`; each chunk is copied into a reused
//!    `SEG_CHUNK_SAMPLES` buffer with the out-of-range tail left zero
//!    (`fill_padded_chunk`).
//! 3. **Segment → multilabel** (`owned.rs:477-498`): [`crate::audio::speaker::segment::SegmentModel::infer`]
//!    then [`crate::audio::speaker::segment::multilabel`] (whose own module doc proves it
//!    equals dia's inline `softmax_row` + `powerset_to_speakers_hard`
//!    decode). Each chunk's `[f][s]` slab is written into the flat
//!    `segmentations` buffer at `chunk_segmentation_range` — dia's
//!    `segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s]` layout
//!    (`owned.rs:496`).
//! 4. **Mask derivation** (`owned.rs:507-591`): the overlap-exclusion rule
//!    (`derive_slot_plans`). See "The critical port" below.
//! 5. **Masked embedding + drop paths** (`owned.rs:600-632`):
//!    [`crate::audio::speaker::embed::EmbedModel::embed_chunk`], the non-finite hard error
//!    (`owned.rs:611-618`), and the PLDA-norm drop (`owned.rs:619-630`).
//! 6. **Count tensor + sliding windows** (`owned.rs:653-674`):
//!    `crate::audio::speaker::window::try_count_from_segmentations` over the
//!    POST-drop-zeroing `segmentations` buffer, plus
//!    [`crate::audio::speaker::window::chunk_sliding_window`] / [`crate::audio::speaker::window::frame_sliding_window`].
//!
//! Layouts, all pinned against dia: `segmentations` is `[c][f][s]` f64
//! (`owned.rs:496`, `algo.rs:209-210`); `raw_embeddings` is `[c][s][d]`
//! f32 written at offset `(c * SLOTS_PER_CHUNK + s) * EMBEDDING_DIM`
//! (`owned.rs:631`, `algo.rs:207-208`); `count` is `[t]` u8 whose length
//! IS `num_output_frames` (`owned.rs:663-674`).
//!
//! **Count runs after all zeroing.** dia computes `count` from the
//! `segmentations` buffer only after Stage 2 has finished zeroing every
//! dropped `(chunk, slot)` column (`owned.rs:663-673` reads the
//! post-Stage-2 buffer); this port preserves that ordering — the fused
//! per-chunk loop finishes all of a chunk's zeroing before the next
//! chunk, and `try_count_from_segmentations` runs only after the whole
//! loop.
//!
//! # The critical port: overlap-exclusion mask derivation (`owned.rs:507-591`)
//!
//! `derive_slot_plans` reproduces pyannote's `embedding_exclude_overlap`
//! (community-1 default) bit-for-bit. Per chunk:
//!
//! - A per-frame "clean" indicator is computed ONCE, over all
//!   [`crate::audio::speaker::segment::SEG_NUM_SLOTS`] slots, BEFORE the per-slot loop:
//!   `clean_frame[f] = active_count < 2`, where a slot is active iff
//!   `seg[f][s] >= onset` — INCLUSIVE `>=` (`owned.rs:536-549`; dia's
//!   prose comment at `owned.rs:552` says "> onset" but its CODE at
//!   `owned.rs:557` is `>=`, and this port matches the code).
//! - Per slot: the raw active mask is `frame_mask[f] = seg[f][s] >= onset`.
//!   If NO frame is active, the slot is `SlotPlan::Skip` — no embed call,
//!   and its segmentation column is zeroed (`owned.rs:561-571`).
//! - Otherwise `used_mask = frame_mask AND clean_frame`, and its true
//!   frames are counted as `clean_count`. The overlap-excluded mask is
//!   used ONLY when it has strictly more than
//!   `EXCLUDE_OVERLAP_MIN_FRAMES` clean frames: `if clean_count <=
//!   EXCLUDE_OVERLAP_MIN_FRAMES { used_mask = frame_mask; }`
//!   (`owned.rs:573-591`). The fallback comparison is `<=`, and it is
//!   PER-SLOT — it sits inside the `for s` loop and replaces only that
//!   slot's mask with that slot's own raw active mask, never the whole
//!   chunk's.
//!
//! `clean_frame` is derived from the PRE-zeroing segmentation values;
//! later column-zeroing (Skip or norm-drop) never feeds back into mask
//! derivation, because each slot reads only its own column and
//! `clean_frame` is already frozen (`derive_slot_plans` computes every
//! slot's plan before `extract` zeroes anything).
//!
//! # Deliberate adaptations from `owned.rs`
//!
//! - **Fused per-chunk loop.** dia runs Stage 1 (segment every chunk),
//!   THEN Stage 2 (embed every chunk) as two passes over `num_chunks`
//!   (`owned.rs:466-499` then `:524-634`). This port fuses them: each
//!   chunk is segmented, masked, and embedded before the next. The output
//!   is identical — every data dependency is within a single chunk (a
//!   chunk's masks read only that chunk's `segmentations` slab; a chunk's
//!   embeddings read only that chunk's masks), and `count` runs after ALL
//!   chunks in both orderings.
//! - **Batched embed with placeholder masks.** dia embeds one
//!   `(chunk, slot)` at a time and never calls embed for a skipped slot
//!   (`owned.rs:561-571,600`). This crate's model is inherently batch-3
//!   ([`crate::audio::speaker::embed`]'s "Batching design"), and an all-false mask row is
//!   the known statistics-pooling divide-by-zero NaN mode
//!   ([`crate::audio::speaker::embed`]'s "NonFinite-output scan scope";
//!   [`crate::audio::speaker::error::InferError::EmptyMask`]'s doc). So a chunk with at
//!   least one planned (`SlotPlan::Embed`) slot makes ONE batched call
//!   in which every `SlotPlan::Skip` slot's mask row borrows the first
//!   planned slot's mask (a non-degenerate placeholder), and those
//!   placeholder OUTPUT rows are discarded — the corresponding
//!   `raw_embeddings` rows stay zero, identical to dia's pre-zeroed,
//!   never-written rows (`owned.rs:502-505`). A chunk with NO planned slot
//!   makes no embed call at all (= dia's zero calls for such a chunk).
//!
//!   Divergence: `embed_chunk`'s 768-wide non-finite scan
//!   ([`crate::audio::speaker::embed`]'s "NonFinite-output scan scope") also covers the
//!   placeholder rows, so a NaN confined to a placeholder row would
//!   hard-error here where dia computes no such row. Accepted, because the
//!   placeholder mask is bit-identical to a real slot's mask over the same
//!   audio, and dia hard-errors on exactly that mask + audio anyway
//!   (`owned.rs:616-618`).
//! - **No `InvalidClip` / `DegenerateEmbedding` recoverable paths.** dia
//!   silently drops a slot on those two embed errors (`owned.rs:602-608`).
//!   Neither exists here: [`crate::audio::speaker::embed::EmbedModel::embed_chunk`]
//!   repeat-pads any length (no clip-length error) and the CoreML path has
//!   no sliding-window aggregation (no degenerate-aggregation error).
//!   `NonFiniteOutput` stays a HARD error (`owned.rs:616-618`), never a
//!   silent drop.
//! - **`!any_active` column-zeroing is ported even though it is a
//!   provable no-op here.** On the hard 0/1 multilabel this crate feeds in
//!   (values exactly `0.0` / `1.0`) with `onset` in `(0.0, 1.0]`, a slot
//!   with no `>= onset` frame also has no nonzero cell, so zeroing its
//!   column changes nothing (`owned.rs:561-571`). It is kept for
//!   structural fidelity to dia and robustness to any future soft
//!   multilabel where sub-onset noise (`0.0001` from softmax) could be
//!   nonzero.

use std::sync::OnceLock;

use crate::audio::speaker::{
  cluster::{ClusterBackend, OnlineOptions},
  embed::{EMBED_SLOTS, EMBEDDING_DIM, EmbedModel},
  error::ExtractError,
  segment::{SEG_CHUNK_SAMPLES, SEG_NUM_SLOTS, SegmentModel},
  source::Source,
  window::{SlidingWindow, WindowOptions},
};

/// pyannote's `embedding_exclude_overlap` minimum clean-frame count: the
/// overlap-excluded mask is used only when its clean-active frame count is
/// STRICTLY greater than this, else the slot falls back to its raw active
/// mask. Matches dia's `EXCLUDE_OVERLAP_MIN_FRAMES`
/// (`diarization/src/offline/owned.rs:522`; pyannote's `min_num_frames =
/// ceil(589 * 400 / (10 * 16000)) = 2`).
///
/// `pub` (not `pub(crate)`) for two independent reasons rather than one:
/// [`crate::audio::speaker::source::ArgmaxSource`] applies the SAME rule to argmax's own
/// tensors (its module doc's "The overlap-exclusion fallback" section), and
/// `tests/parity_argmax_swift.rs` — a separate crate, so `pub(crate)` cannot
/// reach it — asserts the fallback never fires on any consumed slot. All
/// three name this ONE constant rather than each declaring their own `2`, so
/// none of them can drift apart.
pub const EXCLUDE_OVERLAP_MIN_FRAMES: usize = 2;

/// The largest output-frame grid [`Extraction::try_from_parts`] will assemble:
/// `2^22` frames.
///
/// A RESOURCE bound, and the only check in that constructor not derived from the
/// caller's own parts. `num_output_frames` sizes every grid-shaped buffer
/// downstream, and the two largest are plain heap `Vec<f64>`s this crate
/// allocates itself: [`crate::audio::speaker::window::count_from_segmentations`]'s
/// `aggregated` / `overlapping_count` pair, which [`Extraction::diarize_online`]
/// builds on every call and which `try_from_parts` builds once to validate
/// `count`. At this cap that pair is exactly 64 MiB — `diaric`'s own default
/// `SpillOptions::threshold_bytes`, the size above which `diaric` stops keeping a
/// grid in RAM at all (`diarization/src/reconstruct/algo.rs:30-42`). Above the
/// cap a caller could turn an `n`-byte `count` into `16n` bytes of scratch
/// (measured: a 50 MB `count` reached 726 MB peak RSS) with no bound but `usize`.
///
/// At [`crate::audio::speaker::window::FRAME_STEP_S`] — the community-1 frame
/// grid, the only one this crate's models produce — the cap is `4_194_304 *
/// 0.016875 s`, i.e. **19.6 hours** of audio in a single extraction. The offline
/// backend's clustering is quadratic in `num_chunks`, so an extraction that long
/// is already far outside what either backend can finish; the cap refuses
/// geometries that were never going to run, before they can allocate.
///
/// # Why a cap, and not fallible allocation or a short circuit
/// `Vec::try_reserve` would turn the abort into a typed error, but only when the
/// allocator actually fails — on a machine with memory to spare it grants the
/// 16x amplification and calls it success, which IS the denial of service rather
/// than a fix for it. Short-circuiting an all-unmatched input covers only that
/// one shape and leaves every partially-matched grid unbounded. A cap is the
/// only one of the three that bounds the work BEFORE it is attempted, in `O(1)`,
/// for every input — and it is the shape `diaric` itself already uses for this
/// class of hazard (`MAX_RECONSTRUCT_GRID_CELLS`,
/// `diarization/src/reconstruct/algo.rs:42`).
pub const MAX_OUTPUT_FRAMES: usize = 1 << 22;

/// PLDA's minimum raw-embedding L2 norm: `diaric` refuses a raw row below it at
/// the PLDA boundary itself — `plda::transform::RAW_EMBEDDING_MIN_NORM = 0.01`,
/// checked in `RawEmbedding::from_raw_array`
/// (`diarization/src/plda/transform.rs:72,152-165`) and reached from
/// `diarization/src/offline/algo.rs:738`. That constant is `pub(crate)` in
/// `diaric`, so this is the only name a `coremlit` caller has for the number.
///
/// # NECESSARY, NOT SUFFICIENT
///
/// This is the LOWER edge of the band [`Extraction::try_from_parts`] admits for
/// an active slot, not the admission test. `raw_embedding_reaches_plda` no
/// longer reads it: that predicate CALLS the two backend functions instead of
/// re-deriving them, and one of them —
/// [`diaric::embed::Embedding::normalize_from`] — narrows the norm to `f32`, so
/// a finite row whose norm overflows `f32` (`[f32::MAX, f32::MAX, 0.0, …]`,
/// norm `4.8e38`) clears `0.01` in `f64` and is still refused. A caller who
/// wants the exact contract calls those two `diaric` functions, both public;
/// this constant only says how small a row may be, never that a row is usable.
///
/// `pub` for the same reason [`MAX_OUTPUT_FRAMES`] is: it is a number a caller
/// assembling an [`ExtractionParts`] has to know and cannot otherwise reach.
/// Because nothing in-crate reads it any more, it is anchored to `diaric`'s
/// real boundary by test rather than by use —
/// `plda_min_norm_is_diarics_own_floor_measured_not_copied` binary-searches
/// `RawEmbedding::from_wespeaker` for the floor it actually enforces and
/// requires this constant to equal it, so a `diaric` change turns the published
/// number into a test failure instead of a silent lie.
pub const PLDA_MIN_NORM: f64 = 0.01;

/// The ONE [`diaric::plda::PldaTransform`] this crate validates rows against,
/// built at most once per process.
///
/// [`raw_embedding_reaches_plda`] has to run the offline route's PROJECTION,
/// not just its raw-input boundary, and `project` needs a transform. That
/// transform is a process-wide constant, not a caller's choice:
/// `PldaTransform::new()` is `diaric`'s ONLY public constructor, it takes no
/// arguments, and it decodes weight blobs `include_bytes!`d into the binary
/// (`diarization/src/plda/loader.rs:17-36`). Every `PldaTransform` a caller can
/// hand [`Extraction::diarize_with`] is therefore this same transform, so
/// validating against a cached one is validating against theirs.
///
/// Cached in a [`OnceLock`] because building it is ~0.15 ms (blob decode plus
/// nalgebra allocation) against ~8.6 µs for one `project` — ~17 rows' worth of
/// projection per build, measured `--release` on this host — so a per-row build
/// would turn a 1 773-row extraction's 15 ms of validation into 270 ms.
///
/// # Errors
/// [`ExtractError::PldaTransformUnavailable`] if `PldaTransform::new()` fails.
/// It cannot today — its body has no fallible step, the generalized
/// eigendecomposition its doc mentions is pre-computed and shipped, and
/// `plda_transform_is_available` pins that — but it is declared fallible, so
/// this is a TYPED refusal rather than a silent one. The alternative, treating
/// a failed load as "no row is usable", would make [`Extractor::extract`] drop
/// every slot and return an empty diarization: exactly the silent degradation
/// this predicate's whole history is about.
///
/// A cached failure is still reported EVERY time: the `OnceLock` holds an
/// `Option<PldaTransform>`, so a `None` yields a fresh
/// [`ExtractError::PldaTransformUnavailable`] on this and every later call. What
/// the cache cannot repeat is the CAUSE — see that variant's doc.
pub(crate) fn shared_plda_transform() -> Result<&'static diaric::plda::PldaTransform, ExtractError>
{
  static PLDA: OnceLock<Option<diaric::plda::PldaTransform>> = OnceLock::new();
  PLDA
    .get_or_init(|| diaric::plda::PldaTransform::new().ok())
    .as_ref()
    .ok_or(ExtractError::PldaTransformUnavailable)
}

/// Whether a raw WeSpeaker row can reach the clustering BOTH backends run.
///
/// The ONE predicate for "this row is usable", shared by every site that needs
/// it: [`Extractor::extract`] and
/// [`crate::audio::speaker::source::argmax::ArgmaxSource`] both DROP a slot
/// whose row fails it (zeroing that slot's segmentation column, so nothing
/// downstream reads the row at all), and [`Extraction::try_from_parts`]
/// REFUSES parts whose ACTIVE slot carries one.
///
/// `plda` is [`shared_plda_transform`]'s value, hoisted out of the caller's
/// per-row loop.
///
/// # It CALLS both backends rather than describing them
///
/// An `Extraction` is handed to whichever backend the caller picks, so the row
/// standard is the INTERSECTION of the two. Four earlier revisions of this
/// function wrote that intersection out as a norm comparison, and each was a
/// different approximation of it with a different corner escaping:
///
/// - a bare norm floor, with no finiteness clause — `+inf` has an infinite
///   norm, which passes `norm >= floor`;
/// - the ONLINE floor (`NORM_EPSILON`, `1e-12`,
///   `diarization/src/embed/options.rs:30`) — a row at `[0.005, 0.0, …]` is
///   normalized into a speaker by online while offline fails the whole
///   extraction with `Plda(DegenerateInput)`;
/// - the OFFLINE floor (`0.01`) computed in `f64` — which is what
///   `RawEmbedding::from_raw_array` does, but NOT what the online engine does:
///   `Embedding::normalize_from` narrows the norm to `f32` before comparing, so
///   a finite row whose norm overflows `f32` (`[f32::MAX, f32::MAX, 0.0, …]`,
///   `f64` norm `4.8e38`) clears the `f64` floor and normalizes to `None` —
///   online's DROPPED-slot sentinel, which silently yields no speaker at all.
///
/// Every one of those was a better approximation, and the approximation itself
/// was the defect. So this calls the real functions:
///
/// - [`diaric::embed::Embedding::normalize_from`] returning `Some` — the WHOLE
///   of what [`Extraction::diarize_online`] does to a row, with its own `f32`
///   narrowing and its own epsilon, whatever they become;
/// - `diaric::plda::RawEmbedding::from_wespeaker` returning `Ok` — the PLDA RAW
///   boundary [`Extraction::diarize_with`] reaches, `from_raw_array` under a
///   public name (`diarization/src/plda/transform.rs:152-190`), with its own
///   finiteness scan, its own `f64` accumulation and its own floor;
/// - [`diaric::plda::PldaTransform::project`] returning `Ok` — the stage the
///   offline route runs IMMEDIATELY AFTER that boundary
///   (`diarization/src/offline/algo.rs:735-737`, `from_raw_array(arr)?` then
///   `plda.project(&raw)?`). Composing only the first of the two was the
///   round-5 gap: `project` rejects again, on `‖row - mean1‖ <
///   XVEC_CENTERED_MIN_NORM` (`0.1`) and on a degenerate post-LDA intermediate
///   (`diarization/src/plda/transform.rs:315,436,450`). The `f32` cast of
///   `mean1` itself is the witness — raw norm `1.42`, forty times PLDA's raw
///   floor, so both admission functions take it — and it fails the WHOLE
///   offline extraction with `Plda(DegenerateInput)` while online happily
///   emits a speaker
///   (`try_from_parts_rejects_an_active_row_plda_projection_refuses`).
///
/// A future change to any of those functions' epsilons, precision or ordering
/// is picked up here automatically, because this predicate has no opinion of
/// its own to drift from theirs.
///
/// # Where this stops, per backend
///
/// - ONLINE: `normalize_from` is the only thing between a row and
///   `OnlineClusterer::assign`; nothing downstream can reject it. The predicate
///   composes online's chain WHOLE.
/// - OFFLINE: the chain is `filter_embeddings` (which slot is in the PLDA train
///   subset) → `from_raw_array` → `project` → `assign_embeddings`. This
///   composes the two ROW gates. It deliberately does NOT model
///   `filter_embeddings`, whose `clean_frames >= 0.2 * num_frames_per_chunk`
///   selection is a function of the segmentations rather than the row, so a
///   row is held to what offline WOULD do with it, not to whether this
///   particular geometry routes it there. And `assign_embeddings`'s own scan is
///   implied for any row this admits: it rejects a non-finite value (already
///   refused by `from_raw_array`'s finiteness scan) and a squared row norm that
///   overflows `f64` (`ShapeError::RowNormOverflow`, unreachable from `f32`
///   input — `256 · f32::MAX² ≈ 3e79`).
///
/// The three calls take an owned `[f32; EMBEDDING_DIM]`; every call site slices
/// exactly `embedding_range`'s `EMBEDDING_DIM` elements, so the length
/// conversion cannot fail in-crate, and a wrong-length row is refused rather
/// than panicked on. The array types also tie `coremlit`'s [`EMBEDDING_DIM`] to
/// `diaric`'s `embed::EMBEDDING_DIM` and `plda::EMBEDDING_DIMENSION` at COMPILE
/// time: if any of the three moves, this stops building.
pub(crate) fn raw_embedding_reaches_plda(plda: &diaric::plda::PldaTransform, row: &[f32]) -> bool {
  let Ok(row) = <[f32; EMBEDDING_DIM]>::try_from(row) else {
    return false;
  };
  if diaric::embed::Embedding::normalize_from(row).is_none() {
    return false;
  }
  let Ok(raw) = diaric::plda::RawEmbedding::from_wespeaker(row) else {
    return false;
  };
  plda.project(&raw).is_ok()
}

#[cfg(feature = "serde")]
fn default_segmenter_compute() -> crate::ComputeUnits {
  crate::audio::speaker::segment::DEFAULT_SEGMENT_COMPUTE
}

#[cfg(feature = "serde")]
fn default_embedder_compute() -> crate::ComputeUnits {
  crate::audio::speaker::embed::DEFAULT_EMBED_COMPUTE
}

/// Which hardware CoreML may schedule each model on (rust-options-pattern).
///
/// These live on the extractor's [`Options`] even though
/// [`Extractor::extract`] takes already-loaded models and never reads
/// them: `Options` is the one serializable configuration surface a
/// consumer reads to LOAD the two models in the first place (design spec
/// §5, `docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md`)
/// — `segmenter` feeds [`crate::audio::speaker::segment::SegmentModelOptions`], `embedder`
/// feeds [`crate::audio::speaker::embed::EmbedModelOptions`]. Keeping them here lets a
/// single deserialized `Options` drive both the model loads and the
/// extraction geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_segmenter_compute",
      with = "crate::audio::speaker::compute_units_serde"
    )
  )]
  segmenter: crate::ComputeUnits,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_embedder_compute",
      with = "crate::audio::speaker::compute_units_serde"
    )
  )]
  embedder: crate::ComputeUnits,
}

impl Default for ComputeOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ComputeOptions {
  /// Options matching the crate defaults:
  /// [`crate::audio::speaker::segment::DEFAULT_SEGMENT_COMPUTE`] for the segmenter and
  /// [`crate::audio::speaker::embed::DEFAULT_EMBED_COMPUTE`] for the embedder (both
  /// `ComputeUnits::All`).
  pub const fn new() -> Self {
    Self {
      segmenter: crate::audio::speaker::segment::DEFAULT_SEGMENT_COMPUTE,
      embedder: crate::audio::speaker::embed::DEFAULT_EMBED_COMPUTE,
    }
  }

  /// Hardware the segmentation model may be scheduled on.
  #[inline(always)]
  pub const fn segmenter(&self) -> crate::ComputeUnits {
    self.segmenter
  }
  /// Hardware the embedding model may be scheduled on.
  #[inline(always)]
  pub const fn embedder(&self) -> crate::ComputeUnits {
    self.embedder
  }

  /// Builder form of [`Self::set_segmenter`].
  #[must_use]
  #[inline(always)]
  pub const fn with_segmenter(mut self, segmenter: crate::ComputeUnits) -> Self {
    self.set_segmenter(segmenter);
    self
  }
  /// Sets [`Self::segmenter`] in place.
  #[inline(always)]
  pub const fn set_segmenter(&mut self, segmenter: crate::ComputeUnits) -> &mut Self {
    self.segmenter = segmenter;
    self
  }
  /// Builder form of [`Self::set_embedder`].
  #[must_use]
  #[inline(always)]
  pub const fn with_embedder(mut self, embedder: crate::ComputeUnits) -> Self {
    self.set_embedder(embedder);
    self
  }
  /// Sets [`Self::embedder`] in place.
  #[inline(always)]
  pub const fn set_embedder(&mut self, embedder: crate::ComputeUnits) -> &mut Self {
    self.embedder = embedder;
    self
  }
}

/// Full [`Extractor`] configuration: the sliding-window geometry
/// ([`WindowOptions`]) plus the per-model compute placement
/// ([`ComputeOptions`]) plus the selected model [`Source`], composed per
/// rust-options-pattern.
///
/// No `Eq`: [`WindowOptions`] carries an `f32` `onset`.
///
/// `source` is NOT read by [`Extractor::extract`] — that method IS the
/// FluidAudio orchestration and always runs it, whatever this field says.
/// The field is read by [`crate::audio::speaker::source::AnySource::load`], the dispatcher
/// that builds the named source; an `Extractor` obtained by other means
/// simply ignores it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Options {
  #[cfg_attr(feature = "serde", serde(default))]
  window: WindowOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  compute: ComputeOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  source: Source,
}

impl Default for Options {
  fn default() -> Self {
    Self::new()
  }
}

impl Options {
  /// Options composing [`WindowOptions::new`], [`ComputeOptions::new`],
  /// and [`crate::audio::speaker::source::DEFAULT_SOURCE`] — each component's own default
  /// is the single source of truth (the `serde(default)` on each field
  /// defers to it; nested partial configs are covered by each component's
  /// own per-field serde defaults).
  pub const fn new() -> Self {
    Self {
      window: WindowOptions::new(),
      compute: ComputeOptions::new(),
      source: crate::audio::speaker::source::DEFAULT_SOURCE,
    }
  }

  /// The sliding-window geometry ([`crate::audio::speaker::window::chunk_starts`] step and
  /// `onset`).
  #[inline(always)]
  pub const fn window(&self) -> WindowOptions {
    self.window
  }
  /// The per-model compute placement.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeOptions {
    self.compute
  }
  /// The selected model [`Source`] — read by
  /// [`crate::audio::speaker::source::AnySource::load`], not by [`Extractor::extract`] (see
  /// this field's struct-level doc).
  #[inline(always)]
  pub const fn source(&self) -> Source {
    self.source
  }

  /// Builder form of [`Self::set_window`].
  #[must_use]
  #[inline(always)]
  pub const fn with_window(mut self, window: WindowOptions) -> Self {
    self.set_window(window);
    self
  }
  /// Sets [`Self::window`] in place.
  #[inline(always)]
  pub const fn set_window(&mut self, window: WindowOptions) -> &mut Self {
    self.window = window;
    self
  }
  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compute(mut self, compute: ComputeOptions) -> Self {
    self.set_compute(compute);
    self
  }
  /// Sets [`Self::compute`] in place.
  #[inline(always)]
  pub const fn set_compute(&mut self, compute: ComputeOptions) -> &mut Self {
    self.compute = compute;
    self
  }
  /// Builder form of [`Self::set_source`].
  #[must_use]
  #[inline(always)]
  pub const fn with_source(mut self, source: Source) -> Self {
    self.set_source(source);
    self
  }
  /// Sets [`Self::source`] in place.
  #[inline(always)]
  pub const fn set_source(&mut self, source: Source) -> &mut Self {
    self.source = source;
    self
  }
}

/// Runs segmentation + embedding over a clip and assembles diaric's offline
/// tensor set (design spec §5). Holds only [`Options`] — the models
/// themselves are passed to [`Self::extract`], so one `Extractor` can
/// drive many `(SegmentModel, EmbedModel)` pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extractor {
  options: Options,
}

impl Default for Extractor {
  fn default() -> Self {
    Self::new()
  }
}

impl Extractor {
  /// An extractor with default [`Options`].
  pub const fn new() -> Self {
    Self {
      options: Options::new(),
    }
  }

  /// An extractor with the given [`Options`].
  #[must_use]
  pub const fn with_options(options: Options) -> Self {
    Self { options }
  }

  /// The extractor's [`Options`].
  #[inline(always)]
  pub const fn options_ref(&self) -> &Options {
    &self.options
  }

  /// Runs the full extraction over `samples` (16 kHz mono f32) using the
  /// pre-loaded `seg` and `embed` models, producing the [`Extraction`]
  /// diaric's offline diarizer consumes.
  ///
  /// Ports the data-plane of dia's `OwnedDiarizationPipeline::run`
  /// (`diarization/src/offline/owned.rs:361-697`) — see the module doc for
  /// the stage-by-stage structure and every deliberate adaptation.
  ///
  /// # Errors
  /// - [`ExtractError::EmptySamples`] if `samples` is empty
  ///   (`owned.rs:369-371`).
  /// - [`ExtractError::ZeroStepSamples`] if the configured `step_samples`
  ///   is `0` (`owned.rs:374-376`).
  /// - [`ExtractError::StepSamplesExceedsWindow`] if `step_samples >
  ///   SEG_CHUNK_SAMPLES` (`owned.rs:377-387`).
  /// - [`ExtractError::OnsetOutOfRange`] if `onset` is not finite in
  ///   `(0.0, 1.0]` (`owned.rs:388-393`).
  /// - [`ExtractError::FrameCountMismatch`] if the two models disagree on
  ///   the per-chunk frame count (this crate's own guard — see the
  ///   variant's doc).
  /// - [`ExtractError::MisalignedChunkPlacement`] if the chunk grid
  ///   `step_samples` and `samples.len()` derive is one the `count`
  ///   aggregation and `diaric::reconstruct` place differently (this crate's
  ///   own guard — see `window::first_misaligned_chunk`
  ///   for the exact class and [`Extraction::try_from_parts`]'s check 8 for
  ///   why the two must agree). Raised BEFORE any inference: it depends only
  ///   on `samples.len()` and the configured `step_samples`, so a geometry
  ///   this crate cannot diarize honestly costs no model time. It is the same
  ///   refusal `try_from_parts` makes, at the same standard — without it
  ///   `extract` would emit, through the crate-private unchecked assembly,
  ///   precisely the `Extraction` its own public constructor rejects.
  /// - [`ExtractError::Infer`] (via `#[from]`) if either model's inference
  ///   fails (`owned.rs:477,600`).
  /// - [`ExtractError::OutputFrameCountOverflow`] if the derived
  ///   `num_output_frames` would not fit in `usize` (converted from
  ///   [`crate::audio::speaker::window`]'s `WindowError` by exhaustive match —
  ///   unreachable through `extract`'s own geometry, kept typed per this
  ///   crate's no-panic-on-untrusted-config posture; `owned.rs:663-673`).
  pub fn extract(
    &self,
    seg: &SegmentModel,
    embed: &EmbedModel,
    samples: &[f32],
  ) -> Result<Extraction, ExtractError> {
    // ── 1-4. Input guards (owned.rs:369-393) ──────────────────────────
    if samples.is_empty() {
      return Err(ExtractError::EmptySamples);
    }
    let w = self.options.window();
    if w.step_samples() == 0 {
      return Err(ExtractError::ZeroStepSamples);
    }
    if w.step_samples() as usize > SEG_CHUNK_SAMPLES {
      return Err(ExtractError::StepSamplesExceedsWindow {
        step: w.step_samples(),
        window: SEG_CHUNK_SAMPLES,
      });
    }
    if !crate::audio::speaker::window::check_onset(w.onset()) {
      return Err(ExtractError::OnsetOutOfRange { onset: w.onset() });
    }

    // ── 5. Cross-model frame-count agreement (no dia analog) ──────────
    let num_frames = seg.num_frames();
    if num_frames != embed.num_mask_frames() {
      return Err(ExtractError::FrameCountMismatch {
        segmenter: num_frames,
        embedder: embed.num_mask_frames(),
      });
    }

    // ── 6-7. Chunk grid + zero-cleared output buffers ─────────────────
    let starts = crate::audio::speaker::window::chunk_starts(samples.len(), &w); // owned.rs:447-451
    let num_chunks = starts.len();

    // Both timing grids, derived HERE (dia derives them at owned.rs:653-657,
    // after the chunk loop) so the placement guard below can run before any
    // inference. Nothing between here and step 9-11 reads them, so the move is
    // a hoist only.
    let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&w); // owned.rs:653-655
    let frames_sw = crate::audio::speaker::window::frame_sliding_window(); // owned.rs:656-657

    // ── 7b. The two grids must place every chunk at the SAME frame ────
    // No dia analog, and the guard `Extraction::try_from_parts` applies as its
    // check 8: the `count` built at step 9-11 is written on the AGGREGATION's
    // frame grid, while `diaric::reconstruct` — which both cluster backends
    // feed — places the same chunk's activations by `closest_frame`. Where the
    // two disagree the count marks frames the activations never reach and
    // suppresses the ones they do, and `diarize_online` re-derives its own
    // count through the same aggregation, so no choice of `count` repairs it.
    //
    // Assembling through the crate-private, UNCHECKED `from_parts` is what
    // makes this guard `extract`'s own job: without it this method emits the
    // exact `Extraction` its public constructor refuses. See
    // `window::first_misaligned_chunk` for which geometries are affected —
    // none of them reachable with the default `step_samples`.
    if let Some(m) =
      crate::audio::speaker::window::first_misaligned_chunk(num_chunks, chunks_sw, frames_sw)
    {
      return Err(ExtractError::MisalignedChunkPlacement(m));
    }

    // The transform the per-slot row guard below validates against, resolved
    // ONCE before any inference — and BEFORE it, so an unavailable transform
    // refuses the call rather than surfacing after the first chunk's models
    // have already run. It is process-wide (see `shared_plda_transform`), so a
    // per-row build would repay ~0.15 ms for a constant.
    let plda = shared_plda_transform()?;

    let onset = f64::from(w.onset());
    // `segmentations` [c][f][s] f64 (owned.rs:461-464), `raw_embeddings`
    // [c][s][d] f32 pre-zeroed so dropped slots stay zero (owned.rs:502-505).
    let mut segmentations = vec![0.0f64; num_chunks * num_frames * SEG_NUM_SLOTS];
    let mut raw_embeddings = vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM];
    // Reused across chunks (owned.rs:453-455): fixed SEG_CHUNK_SAMPLES.
    let mut padded = vec![0.0f32; SEG_CHUNK_SAMPLES];

    // ── 8. Fused per-chunk segment → mask → embed (module doc) ────────
    for (c, &start) in starts.iter().enumerate() {
      // a. Build the (possibly zero-padded) chunk window (owned.rs:469-475).
      fill_padded_chunk(&mut padded, samples, start);

      // b-d. Segment → multilabel → write this chunk's [f][s] slab
      // (owned.rs:477-498).
      let logits = seg.infer(&padded)?;
      let slab = crate::audio::speaker::segment::multilabel(&logits, num_frames);
      segmentations[chunk_segmentation_range(c, num_frames)].copy_from_slice(&slab);

      // e. Per-slot embedding plans from the overlap-exclusion rule
      // (owned.rs:507-591).
      let plans = derive_slot_plans(
        &segmentations[chunk_segmentation_range(c, num_frames)],
        num_frames,
        onset,
      );

      // f. Zero every Skip slot's segmentation column (owned.rs:561-571).
      for (s, plan) in plans.iter().enumerate() {
        if matches!(plan, SlotPlan::Skip) {
          zero_slot_column(
            &mut segmentations[chunk_segmentation_range(c, num_frames)],
            num_frames,
            s,
          );
        }
      }

      // g. One batched embed call if any slot is planned; Skip slots
      // borrow the first planned slot's mask as a non-degenerate
      // placeholder and their output rows are discarded (module doc).
      let placeholder = plans.iter().find_map(|p| match p {
        SlotPlan::Embed(mask) => Some(mask.as_slice()),
        SlotPlan::Skip => None,
      });
      if let Some(placeholder) = placeholder {
        let masks: [&[bool]; EMBED_SLOTS] = core::array::from_fn(|s| match &plans[s] {
          SlotPlan::Embed(mask) => mask.as_slice(),
          SlotPlan::Skip => placeholder,
        });
        let rows = embed.embed_chunk(&padded, &masks)?;
        for s in 0..SEG_NUM_SLOTS {
          if matches!(plans[s], SlotPlan::Skip) {
            continue;
          }
          // dia's per-slot norm pre-check (owned.rs:619-630), through the ONE
          // predicate every site shares (`raw_embedding_reaches_plda`, which
          // calls the backends rather than restating their thresholds). Its
          // finiteness clause cannot fire here — `embed_chunk` hard-scans its
          // own output — so what this call site exercises is the norm band
          // (too small for PLDA below, past `f32`'s range for the online
          // engine's narrowing above) and PLDA's centered-norm ball, which no
          // real WeSpeaker row lands in: `diaric` calibrated its `0.1` at ~13x
          // below the smallest centered norm across its captured distribution
          // (`diarization/src/plda/transform.rs:273-315`). A collapsed embedder
          // that DID land there is better dropped one slot at a time here than
          // left to fail the caller's WHOLE offline extraction later.
          if !raw_embedding_reaches_plda(plda, &rows[s]) {
            zero_slot_column(
              &mut segmentations[chunk_segmentation_range(c, num_frames)],
              num_frames,
              s,
            );
          } else {
            raw_embeddings[embedding_range(c, s)].copy_from_slice(&rows[s]); // owned.rs:631-632
          }
        }
      }
    }

    // ── 9-11. Count tensor + timing over the post-zeroing buffer ──────
    // `chunks_sw` / `frames_sw` were derived at step 6-7 so the placement
    // guard could run ahead of inference; they are the same two values
    // `owned.rs:653-657` builds here.
    // Manual exhaustive match, deliberately not a `From` impl — see
    // `ExtractError::OutputFrameCountOverflow`'s doc. Unreachable through
    // extract's own geometry (num_chunks * step ≈ samples.len()), kept
    // typed regardless (owned.rs:663-673).
    let count = crate::audio::speaker::window::try_count_from_segmentations(
      &segmentations,
      num_chunks,
      num_frames,
      SEG_NUM_SLOTS,
      w.onset(),
      chunks_sw,
      frames_sw,
    )
    .map_err(|e| match e {
      crate::audio::speaker::window::WindowError::OutputFrameCountOverflow => {
        ExtractError::OutputFrameCountOverflow
      }
    })?;
    Ok(Extraction::from_parts(
      raw_embeddings,
      segmentations,
      count,
      num_chunks,
      num_frames,
      chunks_sw,
      frames_sw,
    ))
  }
}

/// The seven values an [`Extraction`] is assembled from — the exact input set
/// [`Extraction::into_offline_input`] forwards to
/// `diaric::offline::OfflineInput::new`, minus the two it derives.
///
/// This is the "put it back together" half of `Extraction`'s API. mediagraph
/// decomposes diarization into three autonomous nodes (`segmentation → embed →
/// cluster`, issue #110); the cluster node accumulates these parts from TWO
/// upstream stages across many messages and rebuilds an `Extraction` at track
/// end via [`Extraction::try_from_parts`], then calls the same
/// [`Extraction::diarize_with`] / [`Extraction::diarize_online`] every in-process
/// caller does. `Extraction` stays the single carrier: no parallel free
/// `cluster()` function to keep in step with it.
///
/// # Not parameters
/// - `num_speakers` is the fixed [`SEG_NUM_SLOTS`] (3) — the powerset
///   segmenter's slot count, not a caller choice
///   ([`Extraction::num_speakers`]).
/// - `num_output_frames` IS `count.len()`
///   (`diarization/src/offline/owned.rs:674`), derived by the constructor so the
///   two cannot disagree — the same property the crate-private `from_parts` has
///   always had.
///
/// # Why public fields
/// Every field is REQUIRED, has no default, no presence semantics, and no
/// per-field invariant: this is a transparent data carrier whose representation
/// IS the API (`rust-type-conventions`, "Structs and accessors" → Representation
/// and presence). The invariants are all CROSS-field and belong to
/// [`Extraction`], which keeps its own fields private and validates on the way
/// in. Private fields here would add fourteen accessors and setters that each
/// protect nothing, and either a positional `new` or a `Default`-seeded builder
/// — both of which reopen the hole this struct exists to close.
///
/// The hole: `num_chunks`/`num_frames_per_chunk` are both `usize` and
/// `chunks_sw`/`frames_sw` are both [`SlidingWindow`], so a seven-argument
/// positional constructor lets either pair be TRANSPOSED and still compile. A
/// struct literal names every field, so the compiler checks each one.
///
/// This is deliberately NOT the shape of this module's `*Options` types
/// ([`Options`], [`ComputeOptions`], [`WindowOptions`]): those are
/// configuration — every field defaultable, `new()` the canonical default,
/// `with_*`/`set_*` expressing partial override, per-field `serde(default)`
/// meaningful. None of that applies to a required tensor set, and copying the
/// shape would hand out a `Default` that is an invalid extraction
/// (`num_chunks = 0`). The in-crate precedent for THIS shape is
/// `embeddings::siglip::image::preprocess`'s `VisionInputs`, the analogous
/// tensor bundle, likewise public-field.
///
/// No `#[non_exhaustive]`, for the same reason: it would forbid the struct
/// literal outside this crate, which is the entire point.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionParts {
  /// Pre-PLDA WeSpeaker raw embeddings, flattened `[c][s][d]`. Must have length
  /// `num_chunks * num_speakers * EMBEDDING_DIM`, and EVERY value must be finite
  /// — including the rows of inactive slots, which the offline backend scans and
  /// the online one never reads ([`Extraction::try_from_parts`]'s check 11).
  /// Dropped `(chunk, slot)` rows are all-zero, which satisfies that. See
  /// [`Extraction::raw_embeddings`].
  pub raw_embeddings: Vec<f32>,
  /// Per-`(chunk, frame, speaker)` activity, flattened `[c][f][s]`. Must have
  /// length `num_chunks * num_frames_per_chunk * num_speakers`. See
  /// [`Extraction::segmentations`].
  pub segmentations: Vec<f64>,
  /// Per-output-frame instantaneous speaker count, `[t]`. Its length becomes
  /// [`Extraction::num_output_frames`], and its VALUES must be exactly what
  /// `segmentations` derive through
  /// [`crate::audio::speaker::window::count_from_segmentations`] over
  /// `seg > 0.0` — [`Extraction::try_from_parts`]'s check 10, an equality in
  /// both directions. See [`Extraction::count`].
  pub count: Vec<u8>,
  /// Number of sliding-window chunks. See [`Extraction::num_chunks`].
  pub num_chunks: usize,
  /// Frames per chunk (the segmentation model's declared frame count). See
  /// [`Extraction::num_frames_per_chunk`].
  pub num_frames_per_chunk: usize,
  /// Outer (chunk-level) sliding window. See [`Extraction::chunks_sw`].
  pub chunks_sw: SlidingWindow,
  /// Inner (frame-level) sliding window. See [`Extraction::frames_sw`].
  pub frames_sw: SlidingWindow,
}

/// The assembled diaric offline-input tensor set produced by
/// [`Extractor::extract`]. Its accessors expose exactly
/// `diaric::offline::OfflineInput::new`'s parameter list (minus `plda`, which
/// the consumer supplies) — see `Self::into_offline_input`.
///
/// Storage is plain `Vec` (spec §9 open item resolved: a desktop consumer
/// clones once if it fans out; `Arc` is premature).
#[derive(Debug, Clone, PartialEq)]
pub struct Extraction {
  raw_embeddings: Vec<f32>,
  segmentations: Vec<f64>,
  count: Vec<u8>,
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_output_frames: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
}

impl Extraction {
  /// The single ASSEMBLY site for an [`Extraction`], shared by every
  /// [`crate::audio::speaker::source::ModelSource`] and by the public
  /// [`Self::try_from_parts`] — each source builds the identical layout its own
  /// way (see [`crate::audio::speaker::source::argmax`], which decodes in-graph
  /// instead of host-side), and every one of them lands here.
  ///
  /// Crate-private and UNCHECKED, deliberately: its in-crate callers produce a
  /// self-consistent tensor set by construction, so re-validating them would be
  /// dead weight on the hot path. Anything assembled outside this crate goes
  /// through [`Self::try_from_parts`], which validates and then delegates here.
  ///
  /// `num_output_frames` is not a parameter: it IS `count.len()`
  /// (`owned.rs:674`), so deriving it here — at the one site both paths reach —
  /// makes the two impossible to disagree.
  pub(crate) fn from_parts(
    raw_embeddings: Vec<f32>,
    segmentations: Vec<f64>,
    count: Vec<u8>,
    num_chunks: usize,
    num_frames_per_chunk: usize,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
  ) -> Self {
    let num_output_frames = count.len(); // owned.rs:674
    Self {
      raw_embeddings,
      segmentations,
      count,
      num_chunks,
      num_frames_per_chunk,
      num_output_frames,
      chunks_sw,
      frames_sw,
    }
  }

  /// The PUBLIC construction site: validate an [`ExtractionParts`] and assemble
  /// the [`Extraction`] it describes.
  ///
  /// The crate-private `from_parts` trusts its in-crate callers — every
  /// [`crate::audio::speaker::source::ModelSource`] builds a self-consistent
  /// tensor set by construction. This one cannot: mediagraph's cluster node
  /// accumulates the same seven values from TWO upstream stages across many
  /// messages (issue #110), so a dropped or misordered message reaches here as a
  /// geometry that does not describe its own tensors. Every check below exists
  /// so that failure surfaces HERE, naming the disagreeing part, instead of
  /// producing silently wrong clusters or panicking deep inside `diaric`.
  ///
  /// `num_output_frames` and `num_speakers` are not parameters — see
  /// [`ExtractionParts`]'s "Not parameters". Assembly itself is delegated to
  /// that same crate-private `from_parts`, so the `num_output_frames ==
  /// count.len()` derivation still lives at exactly one place.
  ///
  /// # The standard this constructor holds to
  ///
  /// This is the ONE place the two backends' requirements meet. An
  /// [`Extraction`] assembled here can be handed to [`Self::diarize_with`]
  /// (offline) or [`Self::diarize_online`], and the caller chooses which — so
  /// the checks below are the INTERSECTION of what both consumers need, never
  /// the union of what each individually tolerates. An input only one of them
  /// mishandles is still an input this constructor must refuse.
  ///
  /// Every omission in "What is deliberately NOT checked" therefore names the
  /// consumer it was verified against, both of them.
  ///
  /// # What is checked
  ///
  /// 1. `num_chunks`, `num_frames_per_chunk` and `count.len()` are all non-zero.
  /// 2. Both sliding windows are usable timing grids: `start` finite,
  ///    `duration`/`step` finite and `> 0`.
  /// 3. `raw_embeddings.len() == num_chunks * num_speakers * EMBEDDING_DIM` and
  ///    `segmentations.len() == num_chunks * num_frames_per_chunk *
  ///    num_speakers`, each product computed with `checked_mul` FIRST: an
  ///    unchecked product can wrap on hostile dimensions (`num_chunks = 2^32,
  ///    num_frames_per_chunk = 2^32, num_speakers = 3` wraps to `0`) and a short
  ///    or empty slice would then satisfy a naive equality.
  /// 4. The output-frame count [`Self::diarize_online`] re-derives from
  ///    `(chunks_sw, frames_sw, num_chunks)` does not overflow `usize`.
  /// 5. `count.len()` EQUALS that derived count — the value check 4 already
  ///    computes. [`Self::diarize_online`] refuses anything else
  ///    (`CountLenMismatch`) while the offline route accepts it and emits speech
  ///    past the audio the chunks describe, so without this the two backends
  ///    disagree about the same `Extraction`.
  /// 6. That derived count is at most [`MAX_OUTPUT_FRAMES`] — a resource bound,
  ///    see that constant. The last `O(1)` check: everything below is `O(n)`
  ///    over buffers this one bounds.
  /// 7. `frames_sw.step()` stays finite and `> 0` through the `f32` narrowing
  ///    [`Self::diarize_online`] applies to it when it builds the online speech
  ///    duration. See [`ExtractError::FrameStepNotRepresentableInF32`].
  /// 8. The COUNT aggregation and `diaric::reconstruct` place every chunk at the
  ///    SAME output frame. The aggregation reads neither window origin; the
  ///    reconstruction reads both, and routes the chunk start through
  ///    `+ frames_sw.duration / 2` and back out. Equal origins are neither
  ///    necessary nor sufficient for the two to agree, so the mappings
  ///    themselves are compared, chunk by chunk, through the one shared
  ///    `window::first_misaligned_chunk` that
  ///    [`Extractor::extract`] also runs before it touches a model. See
  ///    [`ExtractError::MisalignedChunkPlacement`].
  /// 9. Every `(chunk, slot)` whose segmentation column is active (`seg > 0.0`,
  ///    the activity rule both backends use) carries a raw-embedding row that
  ///    `raw_embedding_reaches_plda` accepts — which is to say a row BOTH
  ///    backends' row chains accept, because that predicate calls them:
  ///    [`diaric::embed::Embedding::normalize_from`] (what
  ///    [`Self::diarize_online`] runs, `f32`-narrowed norm, `1e-12` floor) must
  ///    return `Some`, AND `diaric::plda::RawEmbedding::from_wespeaker` (the
  ///    PLDA raw boundary [`Self::diarize_with`] reaches, `f64` norm,
  ///    [`PLDA_MIN_NORM`] floor) must return `Ok`, AND
  ///    [`diaric::plda::PldaTransform::project`] — the stage offline runs
  ///    immediately after that boundary, with its own `0.1` centered-norm
  ///    rejection — must return `Ok` too. None alone: a row only online accepts
  ///    makes it create a speaker where offline fails the extraction, a row
  ///    only offline accepts is silently dropped online, and a row both
  ///    ADMISSION functions take but projection refuses is the same split one
  ///    stage further in. See [`ExtractError::ActiveSlotWithoutEmbedding`] and
  ///    [`ExtractError::PldaTransformUnavailable`].
  /// 10. `count` EQUALS the count the supplied `segmentations` derive, through
  ///     the same overlap-add aggregation
  ///     [`crate::audio::speaker::window::count_from_segmentations`] runs over
  ///     `seg > 0.0`. Not a bound: offline consumes this field and online
  ///     derives its own, so a `count` above the derived one fabricates
  ///     speakers offline and a `count` below it makes offline silent where
  ///     online speaks. See [`ExtractError::CountNotSegmentationDerived`].
  /// 11. EVERY `raw_embeddings` value is finite — the whole buffer, not only
  ///     the rows check 9 reaches. An INACTIVE slot's row has no active column
  ///     to bring it to check 9, and the two backends read it in opposite ways:
  ///     offline's `diaric::pipeline::assign_embeddings` scans the WHOLE matrix
  ///     (train subset or not, active or not — stage 6 scores every row) and
  ///     fails the extraction with `NonFiniteField::Embeddings`
  ///     (`diarization/src/pipeline/algo.rs:443-455`), while
  ///     [`Self::diarize_online`] skips the inactive column before it copies the
  ///     row and returns `Ok`. Ordered after check 9 so an ACTIVE slot's
  ///     non-finite row keeps that check's more specific `(chunk, slot)`
  ///     diagnosis. Finiteness is the WHOLE of what that offline scan can find
  ///     in an `f32` buffer: its companion refusal `ShapeError::RowNormOverflow`
  ///     needs `Σ v²` to overflow `f64`, and `256 · f32::MAX² ≈ 3e79` cannot.
  ///     See [`ExtractError::NonFiniteRawEmbedding`].
  ///
  /// Checks 1, 2 and 4 are the PANIC-preventing ones: `window`'s
  /// `try_aggregate_output_frame_count` asserts the first two with bare
  /// `assert!`s and [`Self::diarize_online`] `.expect(..)`s the third, so
  /// without them a publicly-assembled `Extraction` could panic far from its
  /// cause. Check 3 is what keeps every `[c][s][d]` / `[c][f][s]` index inside
  /// its buffer. Checks 5 and 7-10 are the CROSS-PART ones: each is a pair of
  /// parts that are individually well-formed and jointly describe something the
  /// producing pipeline cannot have produced. Check 11 is neither: it holds ONE
  /// part to a standard only ONE consumer enforces, which is the same failure —
  /// the backends disagreeing about an identical `Extraction` — arrived at
  /// without a second part being involved at all.
  ///
  /// # What is deliberately NOT checked
  ///
  /// - **`count[t] <= diaric::reconstruct::MAX_COUNT_PER_FRAME`.** Now IMPLIED
  ///   by check 10 and kept unchecked for that reason rather than by deferral:
  ///   check 10 makes `count` EQUAL the derived value, and that derivation is
  ///   `round_ties_even(Σ_c active(c, f) / covering_chunks(t))` over
  ///   per-`(chunk, frame)` active-slot counts each at most `SEG_NUM_SLOTS` (3)
  ///   — an average of values `<= 3`, which rounds to `<= 3`, comfortably under
  ///   `diaric`'s 64. *Verified against both:* the OFFLINE route re-checks it
  ///   anyway as a typed `ShapeError::CountAboveMax`, ahead of every stage
  ///   (`diarization/src/offline/algo.rs:612-617`); the ONLINE route never reads
  ///   this `count` at all — it derives its own distinct-cluster count, likewise
  ///   bounded by `SEG_NUM_SLOTS`, which `diaric::reconstruct` then re-checks
  ///   against the same constant
  ///   (`diarization/src/reconstruct/algo.rs:395-399`). Neither can consume an
  ///   over-count silently, and neither is relied on: the bound holds by
  ///   construction for both.
  /// - **Finiteness of `segmentations`.** An `O(n)` scan that would duplicate a
  ///   typed refusal BOTH routes reach through the SAME function. *Verified
  ///   against both:* every path ends in `diaric::reconstruct`, which scans the
  ///   whole tensor and raises `NonFiniteField::Segmentations`
  ///   (`diarization/src/reconstruct/algo.rs:504-508`) — ONLINE hands it
  ///   `self.segmentations` directly, OFFLINE hands it the same slice at its
  ///   stage 5 (`diarization/src/offline/algo.rs:808`). OFFLINE additionally
  ///   meets `diaric::pipeline::assign_embeddings`' own copy of that scan first
  ///   (`diarization/src/pipeline/algo.rs:456-460`), so the two refuse with
  ///   different typed variants — `Pipeline(NonFinite(Segmentations))` offline,
  ///   `Reconstruct(NonFinite(Segmentations))` online — and neither returns
  ///   `Ok`. This is precisely what check 11's `raw_embeddings` case was NOT:
  ///   there the offline scan had no online counterpart, here it is
  ///   belt-and-braces over a refusal both share.
  ///
  ///   The two also agree on the way to that refusal. `NaN > 0.0` is false, so a
  ///   NaN entry is INACTIVE to every activity predicate in play — this
  ///   constructor's check-9 scan and check-10 derivation, `diarize_online`'s
  ///   own activity count, and dia's `filter_embeddings` — so it cannot make one
  ///   engine see speech the other does not. It only shifts the check-10
  ///   equality DOWN at that frame: the derived count and the caller's `count`
  ///   are required to agree on whatever that scan sees.
  /// - **An INACTIVE slot carrying a usable embedding row** — the converse of
  ///   check 9, and deliberately allowed. *Verified against both:* ONLINE,
  ///   [`Self::diarize_online`] skips the slot on the SAME `seg > 0.0` activity
  ///   test before the row is copied, so the row is not read at all. OFFLINE is
  ///   the half an earlier revision under-stated: it DOES read the row, in three
  ///   places, and is output-blind at every one.
  ///
  ///   1. `filter_embeddings` never routes the slot into the PLDA TRAIN subset —
  ///      that needs `clean_frames >= 0.2 * num_frames_per_chunk` over
  ///      singly-active frames, and an all-zero column sums to `0`
  ///      (`diarization/src/offline/algo.rs:645-679`). This is where the earlier
  ///      reasoning stopped, and on its own it is not enough.
  ///   2. Stage 6 cosine-scores EVERY row against every centroid, this one
  ///      included (`diarization/src/pipeline/algo.rs:636-684`) — but stage 7
  ///      then OVERWRITES the whole soft row of any `(chunk, slot)` whose
  ///      segmentations sum to `0` with `soft.min() - 1.0`
  ///      (`diarization/src/pipeline/algo.rs:685-712`), so the row's own scores
  ///      never survive into the assignment.
  ///   3. What survives is that those pre-mask scores took part in the
  ///      `soft.min()` that constant is built from, so the row CAN move it. It
  ///      cannot move the ANSWER: the constant lands on every inactive row at
  ///      once and on every column of each, and a linear assignment problem is
  ///      invariant under a per-row constant shift. Whatever label an inactive
  ///      slot then draws, its activation is `0` at every frame, so
  ///      `diaric::reconstruct` writes nothing for it.
  ///
  ///   `an_inactive_slots_row_cannot_change_the_offline_result` pins 2 and 3 on
  ///   a three-cluster geometry whose inactive slots DO draw labels.
  ///
  ///   Back to ONLINE: that skip is load-bearing, and its absence was a live
  ///   defect (round 6). An earlier revision argued this shape safe because the row
  ///   would be assigned "with a speech duration of `0`, dropped by any
  ///   `min_speech_duration > 0`". That reasoning covers whether the row
  ///   produces a SPAN; it does not cover whether it perturbs STATE first.
  ///   `OnlineClusterer::assign` reads `min_speech_duration` only in its
  ///   NEW-speaker arm — it matches the nearest centroid and, inside
  ///   `speaker_threshold`, MOVES that centroid and returns before any duration
  ///   gate — so a zero-duration row could shift a speaker far enough that the
  ///   next slot spawned a second one. The fix is the skip, in the one place
  ///   that reads the row; a constructor refusal would not have helped
  ///   [`Extractor::extract`]'s own output (which assembles through the
  ///   crate-private `from_parts`) and would refuse parts that are now
  ///   output-irrelevant to BOTH engines. Both halves are pinned by
  ///   `an_inactive_slots_row_cannot_change_the_online_result`.
  ///
  ///   So the shape stays a consequence of the caller's own data ("this slot
  ///   has an embedding but no speech"), not of a part disagreeing with
  ///   another; `tiny_extraction`'s third slot is exactly it.
  /// - **`num_output_frames` covering the last chunk's last frame.** With checks
  ///   5 and 8 the grid is the derived one and every chunk is placed
  ///   identically by both mappings, but a chunk whose declared `duration`
  ///   spans fewer frame-steps than `num_frames_per_chunk` still derives a grid
  ///   shorter than the chunk it must hold. *Verified against both:* both routes
  ///   end in the same `diaric::reconstruct`, which raises the typed
  ///   `ShapeError::OutputFrameCountTooSmall` before allocating the grid
  ///   (`diarization/src/reconstruct/algo.rs:465-495`). They differ in the WORK
  ///   that precedes it, not in the outcome: ONLINE reaches that call directly,
  ///   OFFLINE only at its stage 5, after AHC and VBx have already run
  ///   (`diarization/src/offline/algo.rs:808`). Neither can return `Ok`.
  ///   Re-deriving the bound here would duplicate `closest_frame`'s float
  ///   arithmetic in a second place, which is how the two grids drift apart.
  /// - **That the parts came from the SAME track.** Every check here compares
  ///   parts to each OTHER, and no comparison carries provenance: parts that are
  ///   each well-formed and mutually consistent are accepted even when
  ///   `raw_embeddings` came from one track and `segmentations` from another of
  ///   identical geometry — silently changing the clustering
  ///   (`try_from_parts_cannot_detect_mutually_inconsistent_parts` pins it).
  ///   *Verified against both:* neither backend can see it either — each
  ///   consumes the tensors it is given and produces a well-formed answer for
  ///   them. A caller joining parts from several messages must carry its own
  ///   track identity and match on it before assembling an [`ExtractionParts`].
  /// - **Which slots offline routes into the PLDA TRAIN subset.** Check 9 now
  ///   composes offline's whole ROW chain — `from_raw_array`'s admission AND
  ///   the `project` that follows it — but not `filter_embeddings`, which
  ///   decides WHICH `(chunk, slot)` that chain is ever run on:
  ///   `clean_frames >= 0.2 * num_frames_per_chunk` over singly-active frames
  ///   (`diarization/src/offline/algo.rs:645-679`). Check 9 therefore holds a
  ///   row to what offline WOULD do with it, not to whether this particular
  ///   geometry routes it there, and is in that direction stricter than
  ///   offline. *Verified against both:* ONLINE has no selection to model —
  ///   [`Self::diarize_online`] runs its row chain on EVERY active slot — so
  ///   check 9 examines exactly the rows online examines, and a superset of the
  ///   rows offline trains on. The asymmetry runs the safe way: a check that
  ///   examined FEWER rows than a backend reads is the failure mode, and it is
  ///   the one check 11 just closed. *Deliberate:* the alternative is a
  ///   constructor whose row standard changes with the segmentations, so the
  ///   same row is accepted in one extraction and refused in another — and the
  ///   corner it would buy is a row `diaric` calibrated out of existence anyway
  ///   (the `0.1` sits ~13x below the smallest centered norm in its captured
  ///   distribution, `diarization/src/plda/transform.rs:273-315`).
  ///
  ///   An earlier revision left the projection out ENTIRELY, on the grounds
  ///   that it needs a [`diaric::plda::PldaTransform`] this constructor is not
  ///   given and that no witness was constructible. Both were wrong (round 6):
  ///   `PldaTransform::new()` is public, takes no arguments, and loads
  ///   compile-time-embedded weights, so the transform IS available here
  ///   (cached once — `shared_plda_transform`), and the `f32` cast of `mean1`
  ///   is a witness built from those same shipped bytes
  ///   (`try_from_parts_rejects_an_active_row_plda_projection_refuses`).
  /// - **That both backends agree an active slot deserves a SPEAKER.** Check 9
  ///   is about the ROW; whether a slot with an acceptable row becomes a speaker
  ///   is settled later, and NOT by two symmetric duration gates — an earlier
  ///   revision described it that way and was wrong on both halves.
  ///   *Verified against both:* OFFLINE has no speech-duration gate on the span
  ///   path at all. Its `0.2 * num_frames_per_chunk` ratio selects the PLDA
  ///   TRAIN subset only (`diarization/src/offline/algo.rs:645-679`); a slot
  ///   that FAILS it is still cosine-scored at stage 6, still assigned a cluster
  ///   by `constrained_argmax` at stage 7 (whose mask is `Σ seg == 0`, which it
  ///   is not), and still contributes its frames to `diaric::reconstruct`'s grid
  ///   — so it emits a span. ONLINE does gate on duration, but in ONE arm only:
  ///   `OnlineClusterer::assign` matches the nearest centroid FIRST and returns
  ///   `Existing` whenever that match is inside `speaker_threshold`, reading
  ///   `min_speech_duration` only when nothing matched
  ///   (`diarization/src/cluster/online/algo.rs:274-304`) — the same structure
  ///   round 6's finding 1 turned on. So a short slot is dropped online exactly
  ///   when it would also be a NEW speaker. That is still a real split: on the
  ///   shipping grid the default `min_speech_duration` of `1.0` s is `60` frames
  ///   of `589`, so the first slot of a speaker who talks for `50` of them
  ///   (`0.84` s) yields ONE offline span and NONE online for the identical
  ///   `Extraction`. *Not checkable here:* the online
  ///   gate is a function of `OnlineOptions`, supplied at
  ///   [`Self::diarize_online`] and not at construction, so any threshold this
  ///   constructor assumed would be wrong for some caller — and the input is
  ///   ordinary audio [`Extractor::extract`] itself produces (a speaker who
  ///   talks briefly in a 10 s window), which it must not refuse. Choosing
  ///   between the two clustering engines is the caller's, and their disagreement
  ///   on sparse speech is a property of that choice, not of malformed parts.
  ///
  /// # Errors
  /// - [`ExtractError::ZeroExtractionDimension`] — check 1.
  /// - [`ExtractError::InvalidSlidingWindow`] — check 2.
  /// - [`ExtractError::ExtractionGeometryOverflow`] /
  ///   [`ExtractError::ExtractionLenMismatch`] — check 3.
  /// - [`ExtractError::OutputFrameCountOverflow`] — check 4.
  /// - [`ExtractError::ExtractionLenMismatch`] with
  ///   [`ExtractionPart::Count`](crate::audio::speaker::error::ExtractionPart::Count)
  ///   — check 5.
  /// - [`ExtractError::OutputFrameCountTooLarge`] — check 6.
  /// - [`ExtractError::FrameStepNotRepresentableInF32`] — check 7.
  /// - [`ExtractError::MisalignedChunkPlacement`] — check 8.
  /// - [`ExtractError::ActiveSlotWithoutEmbedding`] — check 9.
  /// - [`ExtractError::PldaTransformUnavailable`] — check 9's transform could
  ///   not be built, so the row standard cannot be applied at all.
  /// - [`ExtractError::CountNotSegmentationDerived`] — check 10.
  /// - [`ExtractError::NonFiniteRawEmbedding`] — check 11.
  ///
  /// # Examples
  /// ```
  /// use coremlit::audio::speaker::{
  ///   embed::EMBEDDING_DIM,
  ///   error::{ExtractError, ExtractionPart},
  ///   extract::{Extraction, ExtractionParts},
  ///   segment::SEG_NUM_SLOTS,
  ///   window::{FRAME_STEP_S, WindowOptions, chunk_sliding_window, frame_sliding_window},
  /// };
  ///
  /// let parts = ExtractionParts {
  ///   raw_embeddings: vec![0.5; SEG_NUM_SLOTS * EMBEDDING_DIM],
  ///   segmentations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  ///   // One speaker per frame, and only the two frames the single chunk covers:
  ///   // `count[t]` may not exceed what `segmentations` puts at output frame `t`.
  ///   count: vec![1, 1, 0, 0],
  ///   num_chunks: 1,
  ///   num_frames_per_chunk: 2,
  ///   // A chunk three frame-steps long, so the geometry derives exactly the
  ///   // four output frames `count` declares. (The nominal 10 s chunk duration
  ///   // derives 594 of them, which this `count` would contradict.)
  ///   chunks_sw: chunk_sliding_window(&WindowOptions::new()).with_duration(3.0 * FRAME_STEP_S),
  ///   frames_sw: frame_sliding_window(),
  /// };
  /// let extraction = Extraction::try_from_parts(parts.clone()).expect("self-consistent parts");
  /// assert_eq!(extraction.num_output_frames(), 4); // == count.len()
  /// assert_eq!(extraction.num_speakers(), SEG_NUM_SLOTS);
  ///
  /// // A message-assembly bug names the part that disagreed.
  /// let mut broken = parts;
  /// broken.segmentations.pop();
  /// let err = Extraction::try_from_parts(broken).unwrap_err();
  /// let ExtractError::ExtractionLenMismatch(m) = err else {
  ///   panic!("expected a length mismatch, got {err}")
  /// };
  /// assert_eq!(m.part(), ExtractionPart::Segmentations);
  /// assert_eq!((m.got(), m.expected()), (5, 6));
  /// ```
  pub fn try_from_parts(parts: ExtractionParts) -> Result<Self, ExtractError> {
    use crate::audio::speaker::error::{
      ExtractionGeometryOverflow, ExtractionLenMismatch, ExtractionPart, InvalidSlidingWindow,
    };

    let ExtractionParts {
      raw_embeddings,
      segmentations,
      count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    } = parts;

    // ── 1. Non-zero dimensions ────────────────────────────────────────
    // `count.len()` IS num_output_frames, so an empty `count` is the
    // ZeroExtractionDimension(Count) case, not a length mismatch.
    if num_chunks == 0 {
      return Err(ExtractError::ZeroExtractionDimension(
        ExtractionPart::NumChunks,
      ));
    }
    if num_frames_per_chunk == 0 {
      return Err(ExtractError::ZeroExtractionDimension(
        ExtractionPart::NumFramesPerChunk,
      ));
    }
    if count.is_empty() {
      return Err(ExtractError::ZeroExtractionDimension(ExtractionPart::Count));
    }

    // ── 2. Both sliding windows are usable timing grids ───────────────
    for (part, w) in [
      (ExtractionPart::ChunksSw, chunks_sw),
      (ExtractionPart::FramesSw, frames_sw),
    ] {
      let usable = w.start().is_finite()
        && w.duration().is_finite()
        && w.duration() > 0.0
        && w.step().is_finite()
        && w.step() > 0.0;
      if !usable {
        return Err(ExtractError::InvalidSlidingWindow(
          InvalidSlidingWindow::new(part, w),
        ));
      }
    }

    // ── 3. Geometry products, CHECKED, before the length equalities ───
    // Order matters twice over. Overflow before equality, because a wrapped
    // product can land on a length a short slice happens to have. And
    // raw_embeddings' product before segmentations', because the two share
    // `num_chunks`: a `num_chunks` large enough to overflow one makes the
    // other's required length unallocatable too, so whichever is checked
    // first is the one that can be exercised in isolation.
    let expected_embeddings = num_chunks
      .checked_mul(SEG_NUM_SLOTS)
      .and_then(|n| n.checked_mul(EMBEDDING_DIM))
      .ok_or_else(|| {
        ExtractError::ExtractionGeometryOverflow(ExtractionGeometryOverflow::new(
          ExtractionPart::RawEmbeddings,
          num_chunks,
          num_frames_per_chunk,
        ))
      })?;
    let chunk_frames = num_chunks.checked_mul(num_frames_per_chunk);
    let expected_segmentations = chunk_frames
      .and_then(|n| n.checked_mul(SEG_NUM_SLOTS))
      .ok_or_else(|| {
        ExtractError::ExtractionGeometryOverflow(ExtractionGeometryOverflow::new(
          ExtractionPart::Segmentations,
          num_chunks,
          num_frames_per_chunk,
        ))
      })?;
    // `chunk_frames` is `Some` whenever the product above did not overflow:
    // `x * y * z` overflowing only at the last factor still yields `x * y`.
    let chunk_frames = chunk_frames.expect("checked_mul chain proved the inner product fits");
    if raw_embeddings.len() != expected_embeddings {
      return Err(ExtractError::ExtractionLenMismatch(
        ExtractionLenMismatch::new(
          ExtractionPart::RawEmbeddings,
          raw_embeddings.len(),
          expected_embeddings,
        ),
      ));
    }
    if segmentations.len() != expected_segmentations {
      return Err(ExtractError::ExtractionLenMismatch(
        ExtractionLenMismatch::new(
          ExtractionPart::Segmentations,
          segmentations.len(),
          expected_segmentations,
        ),
      ));
    }

    // ── 4. The output-frame count `diarize_online` will re-derive ─────
    // Same helper, same two arguments, same deterministic f64 arithmetic as
    // `window::try_aggregate_output_frame_count` runs there, so proving it
    // returns `Ok` here proves that method's `.expect(..)` is unreachable.
    // `num_chunks >= 1` (check 1) makes the `- 1` safe; the windows are
    // finite and positive (check 2), so `last_chunk_end` is the only
    // quantity left that can drive the division out of range.
    let last_chunk_end = chunks_sw.duration() + (num_chunks - 1) as f64 * chunks_sw.step();
    let derived_output_frames =
      crate::audio::speaker::window::try_num_output_frames(last_chunk_end, frames_sw.step())
        .map_err(|e| match e {
          crate::audio::speaker::window::WindowError::OutputFrameCountOverflow => {
            ExtractError::OutputFrameCountOverflow
          }
        })?;

    // ── 5. `count.len()` IS the grid the geometry derives ─────────────
    // The value check 4 computes is the answer, not a by-product: keeping it
    // is the whole fix. `diarize_online` re-derives this identical number and
    // refuses any `Extraction` whose `num_output_frames` differs
    // (`CountLenMismatch`), while `diaric`'s offline reconstruct requires only
    // that the grid COVER the last chunk — so a longer `count` makes the two
    // backends disagree about the same `Extraction`, offline emitting speech
    // past the end of the audio the chunks describe. Reported as a `Count`
    // length mismatch: `count.len()` is `num_output_frames`, so this is the
    // same diagnosis as any other tensor whose length the geometry contradicts.
    if count.len() != derived_output_frames {
      return Err(ExtractError::ExtractionLenMismatch(
        ExtractionLenMismatch::new(ExtractionPart::Count, count.len(), derived_output_frames),
      ));
    }

    // ── 6. The grid is one this crate is willing to allocate for ──────
    // The RESOURCE bound (see `MAX_OUTPUT_FRAMES`), and the last O(1) check:
    // everything below is O(n) over buffers this bound now limits.
    if derived_output_frames > MAX_OUTPUT_FRAMES {
      return Err(ExtractError::OutputFrameCountTooLarge(
        derived_output_frames,
      ));
    }

    // ── 7. `frames_sw.step()` survives the narrowing `diarize_online` does ──
    // That method builds the online speech duration in `f32`
    // (`active_frames as f32 * frames_sw.step() as f32`, FluidAudio's own
    // arithmetic, pinned by `tests/parity_online_swift.rs`). A step below
    // `f32`'s smallest subnormal narrows to `0.0` and a step above `f32::MAX`
    // to `+inf`, either of which hands the engine a duration its geometry did
    // not declare. Checked AFTER 4 so a step that also overflows the derived
    // count still reports the overflow.
    let frame_step_f32 = frames_sw.step() as f32;
    if !frame_step_f32.is_finite() || frame_step_f32 <= 0.0 {
      return Err(ExtractError::FrameStepNotRepresentableInF32(
        InvalidSlidingWindow::new(ExtractionPart::FramesSw, frames_sw),
      ));
    }

    // ── 8. Both grids must place every chunk at the SAME output frame ──
    // The `count` this constructor validates (check 10) is written on the
    // aggregation's grid: `window::aggregate_chunk_start_frame` places chunk `c`
    // at `round(c * chunk_step / frame_step)` and reads NEITHER window origin.
    // `diaric::reconstruct` — which BOTH backends feed — places the same chunk
    // at `closest_frame(chunks_sw.start + c * chunk_step + frames_sw.duration /
    // 2)`, mirrored here as `window::reconstruct_chunk_start_frame`. Where the
    // two differ, the count marks frames the activations never reach and
    // suppresses the ones they do: speech silently shifted.
    //
    // Comparing the two mappings is the check; testing the origins for `0.0` is
    // not. Zero origins do NOT imply agreement — the reconstruction route adds
    // `frames_sw.duration / 2` and subtracts it again, and `(x + h) - h != x` in
    // binary floating point (`chunk_step = 0.04218750000000001` over the
    // community-1 frame grid puts chunk 1 at frame 3 aggregating and frame 2
    // reconstructing) — and non-zero origins do NOT imply disagreement (equal
    // origins cancel exactly). See `ExtractError::MisalignedChunkPlacement`.
    //
    // Ordered after every O(1) check, and it MUST stay after check 3: this is
    // the first O(num_chunks) work, and check 3 is what bounds `num_chunks` by
    // a buffer the caller actually allocated. Ahead of it, a declared
    // `num_chunks` of `2^60` would spin here before anything refused it.
    //
    // `window::first_misaligned_chunk` is the ONE definition of the comparison,
    // shared with `Extractor::extract`'s own pre-inference guard: written out a
    // second time it would be a second expression that is algebraically equal
    // and numerically different — the exact failure mode this check exists for.
    if let Some(m) =
      crate::audio::speaker::window::first_misaligned_chunk(num_chunks, chunks_sw, frames_sw)
    {
      return Err(ExtractError::MisalignedChunkPlacement(m));
    }

    // ── 9. An active slot must carry an embedding BOTH engines can use ──
    // The activity predicate is `seg > 0.0` — the same "any nonzero entry is
    // binary-active" rule `diarize_online` applies and dia's
    // `filter_embeddings` uses (`diarization/src/offline/algo.rs:656-660`).
    //
    // The ROW predicate is `raw_embedding_reaches_plda`, which does not describe
    // what the two backends accept — it CALLS them: `normalize_from`, then
    // `from_wespeaker`, then `PldaTransform::project`, and requires all three.
    // Every hand-written stand-in for that conjunction has had a corner escape
    // it (see the predicate's own doc); the last two were the floor — offline
    // compares the norm in `f64`, online narrows it to `f32` first, so
    // `[f32::MAX, f32::MAX, 0.0, …]` clears `0.01` in `f64` and normalizes to
    // `None`, online's DROPPED-slot sentinel read as "no speaker here" under an
    // ACTIVE column — and the composition, which stopped at offline's RAW
    // boundary and missed the projection that boundary feeds.
    //
    // Both directions matter. A row only ONLINE accepts (norm in `[1e-12, 0.01)`,
    // or inside PLDA's `0.1` centered ball) makes online manufacture a speaker
    // where offline fails the whole extraction with `Plda(DegenerateInput)`; a
    // row only OFFLINE accepts (`f64` norm past `f32`'s range) reaches PLDA
    // while online silently drops the slot. Both in-crate producers drop such a
    // row through this very same predicate, so this constructor requires no
    // more of a caller than the crate requires of itself.
    //
    // The transform is resolved once, ahead of the loop: it is process-wide
    // (see `shared_plda_transform`) and building it costs ~0.15 ms against
    // ~8.6 µs per `project`.
    let plda = shared_plda_transform()?;
    for c in 0..num_chunks {
      for s in 0..SEG_NUM_SLOTS {
        let active = (0..num_frames_per_chunk)
          .any(|f| segmentations[(c * num_frames_per_chunk + f) * SEG_NUM_SLOTS + s] > 0.0);
        if !active {
          continue;
        }
        if !raw_embedding_reaches_plda(plda, &raw_embeddings[embedding_range(c, s)]) {
          return Err(ExtractError::ActiveSlotWithoutEmbedding(
            crate::audio::speaker::error::ActiveSlotWithoutEmbedding::new(c, s),
          ));
        }
      }
    }

    // ── 10. `count` must BE the count these segmentations derive ──────
    // EQUALITY, not a bound. The two backends read this field differently:
    // offline consumes it verbatim, and `diarize_online` ignores it and derives
    // its own from the same segmentations (its own comment says why it must).
    // Any supplied `count` that differs from the derived one therefore makes
    // the two disagree about the same `Extraction` — an inflated `count[t]`
    // makes offline select zero-activation padded columns and emit phantom
    // speakers, a deflated one makes offline silent where online emits the
    // speaker. Bounding one direction leaves the other open, which is why this
    // is `!=` and not `>`.
    //
    // The derived value is the caller's own data run through the SAME
    // overlap-add aggregation `count_from_segmentations` uses, over `seg > 0.0`
    // — the activity predicate BOTH backends apply to a segmentation column
    // (`diarize_online`'s activity scan, dia's `filter_embeddings`,
    // `diarization/src/offline/algo.rs:656-660`). It is also exactly what
    // `extract()` produces: `extract()` aggregates `seg >= onset` over a hard
    // `0.0`/`1.0` multilabel, and on those values `>= onset` and `> 0.0` select
    // the same slots for every `onset` in `(0.0, 1.0]` (`check_onset`'s range).
    //
    // The only check that ALLOCATES, so it is ordered behind every check that
    // bounds what it would allocate — `chunk_frames` past check 3 and the
    // derived grid past check 6. Check 11 follows it only because that one is
    // allocation-free and deliberately yields to check 9's diagnosis; nothing
    // below this point can make this aggregation cheaper.
    let mut chunk_count = vec![0.0f64; chunk_frames];
    for (cf, slot) in chunk_count.iter_mut().enumerate() {
      let base = cf * SEG_NUM_SLOTS;
      *slot = segmentations[base..base + SEG_NUM_SLOTS]
        .iter()
        .filter(|v| **v > 0.0)
        .count() as f64;
    }
    let derived = crate::audio::speaker::window::try_aggregate_output_frame_count(
      &chunk_count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    )
    .map_err(|e| match e {
      crate::audio::speaker::window::WindowError::OutputFrameCountOverflow => {
        ExtractError::OutputFrameCountOverflow
      }
    })?;
    // Check 5 made `derived.len() == count.len()`: both are the same
    // `try_num_output_frames(last_chunk_end, frames_sw.step())`. The `zip`
    // below TRUNCATES to the shorter of the two, so that equality is what stops
    // a short `count` from skipping frames it never declared.
    debug_assert_eq!(
      derived.len(),
      count.len(),
      "check 5 must have equated count.len() with the derived grid"
    );
    for (t, (&got, &expected)) in count.iter().zip(derived.iter()).enumerate() {
      if got != expected {
        return Err(ExtractError::CountNotSegmentationDerived(
          crate::audio::speaker::error::CountNotSegmentationDerived::new(t, got, expected),
        ));
      }
    }

    // ── 11. EVERY raw_embeddings value is finite, active or not ───────
    // Check 9 stops at the rows of ACTIVE slots, and that is exactly one
    // buffer position short of the split it exists to prevent. Under an
    // all-zero segmentation column the two backends read the same row
    // differently: dia's `assign_embeddings` scans the WHOLE matrix — train
    // subset or not, active or not, because its stage-6 cosine scoring reads
    // every row — and fails the offline extraction with
    // `NonFiniteField::Embeddings` (`diarization/src/pipeline/algo.rs:443-455`),
    // while `diarize_online` skips an inactive column before it ever copies the
    // row and returns `Ok`. Fatal to one engine, invisible to the other, for
    // the identical `Extraction`.
    //
    // The WHOLE buffer, not a per-inactive-row loop: the property is "no value
    // in this tensor is non-finite", which is what offline's scan asserts, and
    // stating it over the buffer cannot drift from the `[c][s][d]` indexing the
    // way a hand-rolled row walk can.
    //
    // Ordered AFTER check 9 on purpose, and it costs nothing to do so. Ahead of
    // it, this blanket scan would swallow every ACTIVE slot's non-finite row —
    // the round-1 falsifier's NaN included — and report a bare buffer offset
    // where `ActiveSlotWithoutEmbedding` names the `(chunk, slot)` whose column
    // claims speech, the more specific diagnosis and the more actionable one.
    // The scan is ~1% of check 9 on a realistic extraction (10 minutes of audio
    // is 1 773 rows: ~0.17 ms to scan the 1.73 MiB buffer against ~16.4 ms for
    // the row chain, release, this host), so the ordering buys that diagnosis
    // for no measurable cost.
    //
    // Refuses nothing either in-crate producer emits. Both pre-zero this buffer
    // and `0.0` is finite, so an unwritten row passes; a written row passed
    // `raw_embedding_reaches_plda`, whose `from_wespeaker` clause has its own
    // finiteness scan (`no_producer_can_emit_a_buffer_the_finiteness_check_refuses`).
    if let Some(i) = raw_embeddings.iter().position(|v| !v.is_finite()) {
      return Err(ExtractError::NonFiniteRawEmbedding(i));
    }

    Ok(Self::from_parts(
      raw_embeddings,
      segmentations,
      count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    ))
  }

  /// Pre-PLDA WeSpeaker raw embeddings, flattened `[c][s][d]`. Length
  /// `num_chunks * num_speakers * EMBEDDING_DIM`. Dropped `(chunk, slot)`
  /// rows are all-zero. Matches `OfflineInput::raw_embeddings`
  /// (`diarization/src/offline/algo.rs:207-208,324-326`).
  #[inline(always)]
  pub fn raw_embeddings(&self) -> &[f32] {
    &self.raw_embeddings
  }
  /// Number of sliding-window chunks. Matches `OfflineInput::num_chunks`
  /// (`diarization/src/offline/algo.rs:328-330`).
  #[inline(always)]
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Speaker slots per chunk — the fixed [`SEG_NUM_SLOTS`] (3). Mirrors
  /// `OfflineInput::new`'s `num_speakers` parameter, which dia's own
  /// pipeline supplies as `SLOTS_PER_CHUNK` (`owned.rs:680`); accessor
  /// matches `OfflineInput::num_speakers`
  /// (`diarization/src/offline/algo.rs:332-334`).
  #[inline(always)]
  pub const fn num_speakers(&self) -> usize {
    SEG_NUM_SLOTS
  }
  /// Per-`(chunk, frame, speaker)` activity, flattened `[c][f][s]`. Length
  /// `num_chunks * num_frames_per_chunk * num_speakers`. Matches
  /// `OfflineInput::segmentations`
  /// (`diarization/src/offline/algo.rs:209-210,336-338`).
  #[inline(always)]
  pub fn segmentations(&self) -> &[f64] {
    &self.segmentations
  }
  /// Frames per chunk (the segmentation model's declared frame count).
  /// Matches `OfflineInput::num_frames_per_chunk`
  /// (`diarization/src/offline/algo.rs:340-342`).
  #[inline(always)]
  pub const fn num_frames_per_chunk(&self) -> usize {
    self.num_frames_per_chunk
  }
  /// Per-output-frame instantaneous speaker count, `[t]`. Length
  /// `num_output_frames`. Matches `OfflineInput::count`
  /// (`diarization/src/offline/algo.rs:211-212,344-346`).
  #[inline(always)]
  pub fn count(&self) -> &[u8] {
    &self.count
  }
  /// Output-frame grid length (`== count().len()`). Matches
  /// `OfflineInput::num_output_frames`
  /// (`diarization/src/offline/algo.rs:348-350`).
  #[inline(always)]
  pub const fn num_output_frames(&self) -> usize {
    self.num_output_frames
  }
  /// Outer (chunk-level) sliding window. Matches `OfflineInput::chunks_sw`
  /// (`diarization/src/offline/algo.rs:352-354`, likewise by value).
  #[inline(always)]
  pub const fn chunks_sw(&self) -> SlidingWindow {
    self.chunks_sw
  }
  /// Inner (frame-level) sliding window. Matches `OfflineInput::frames_sw`
  /// (`diarization/src/offline/algo.rs:356-358`, likewise by value).
  #[inline(always)]
  pub const fn frames_sw(&self) -> SlidingWindow {
    self.frames_sw
  }

  /// Borrow this extraction (plus a caller-supplied `plda`) as a
  /// `diaric::offline::OfflineInput`, ready for `diaric::offline::diarize_offline`.
  ///
  /// Fills `OfflineInput::new`'s 10-parameter signature verbatim (pinned
  /// at `diarization/src/offline/algo.rs:216-227`); the returned value
  /// carries diaric's community-1 hyperparameter defaults (`threshold = 0.6`
  /// etc., `algo.rs:239-246`), each overridable via diaric's own `with_*`
  /// builders on the returned value.
  ///
  /// `plda` is spelled `diaric::plda::PldaTransform` — dia exports it there
  /// (`diarization/src/plda/mod.rs:39`), NOT at its crate root, so the
  /// plan's `diaric::PldaTransform` shorthand is written out in full here.
  /// The two [`SlidingWindow`] values convert into diaric's own via
  /// [`crate::audio::speaker::window`]'s `From` impls (`window/mod.rs`); `OfflineInput::new`
  /// takes `diaric::reconstruct::SlidingWindow` by value (`algo.rs:11,224-225`).
  ///
  /// Un-gated: `diaric` is a runtime dependency and `diaric::offline` is part of
  /// its ort-free clustering surface, so this bridge is always available.
  ///
  /// # Why the projection is Rust arithmetic and not the vendor's CoreML PLDA
  ///
  /// The speakerkit model repo ships the same community-1 projection as CoreML
  /// graphs (`PLDA.mlmodelc`, `PldaRho.mlmodelc`), and this crate deliberately
  /// loads neither. Both normalize with `clip(x, 1e-12) -> sqrt` and then divide
  /// by the result, at TWO sites each. 1e-12 is 1.7e-5x fp16's smallest
  /// subnormal (`2^-24`), so IF those ops are lowered to fp16 — which is what an
  /// ANE placement under the default [`crate::ComputeUnits::All`] would mean —
  /// the clip floor rounds to zero, leaving `sqrt(0)` and a silent divide by
  /// zero.
  ///
  /// That antecedent is exactly where the evidence stops. `fp16_guards` reads
  /// the MIL text STATICALLY, so the fp16 consequence is established, but its
  /// PREMISE is UNTESTED: nothing loads these graphs, so no run has observed
  /// where CoreML actually places their `clip`/`sqrt` ops, nor whether it
  /// demotes them to fp16 under `ComputeUnits::All`. Read this as a decision to
  /// decline an unnecessary risk, not as a reproduced divide-by-zero.
  ///
  /// [`diaric::plda::PldaTransform`] instead `include_bytes!`s the fitted LDA +
  /// PLDA weights and projects in f64 on the host, so no compute-unit choice can
  /// demote the arithmetic and the projection is bit-reproducible across
  /// placements.
  ///
  /// Both graphs stay pinned in `tests/fp16_guards.rs`'s `KNOWN_DEFECTS`. That
  /// pin is what makes this decision revisitable rather than folkloric: the gate
  /// fails if a re-converted graph ever repairs the epsilon, forcing the choice
  /// to be re-made deliberately instead of a fixed model going unnoticed.
  pub fn into_offline_input<'a>(
    &'a self,
    plda: &'a diaric::plda::PldaTransform,
  ) -> diaric::offline::OfflineInput<'a> {
    diaric::offline::OfflineInput::new(
      self.raw_embeddings.as_slice(),
      self.num_chunks,
      SEG_NUM_SLOTS,
      self.segmentations.as_slice(),
      self.num_frames_per_chunk,
      self.count.as_slice(),
      self.num_output_frames,
      self.chunks_sw.into(),
      self.frames_sw.into(),
      plda,
    )
  }

  /// Cluster this extraction into speaker-labelled RTTM spans at the DEFAULT
  /// backend — [`ClusterBackend::default`], i.e. diaric's offline
  /// pyannote-community-1 pipeline with its community-1 hyperparameters. Exactly
  /// [`self.diarize_with(plda, ClusterBackend::default())`](Self::diarize_with).
  ///
  /// This is the SINGLE default runtime clustering path: every parity harness
  /// scores exactly this method's output rather than re-plumbing
  /// `into_offline_input → diarize_offline` (or re-selecting a backend) itself,
  /// so the public API and the tested path cannot diverge (the alignkit
  /// canonical-wiring lesson). Because [`ClusterBackend::default`] applies
  /// diaric's own defaults, the assembled [`diaric::offline::OfflineInput`] is
  /// field-identical to the bare [`Self::into_offline_input`], so this is
  /// byte-identical to feeding diaric directly.
  ///
  /// # Errors
  /// As [`Self::diarize_with`].
  pub fn diarize(
    &self,
    plda: &diaric::plda::PldaTransform,
  ) -> Result<diaric::offline::OfflineOutput, diaric::offline::Error> {
    self.diarize_with(plda, ClusterBackend::default())
  }

  /// Cluster this extraction into speaker-labelled RTTM spans via the selected
  /// [`ClusterBackend`] — the crate's runtime clustering entry point.
  ///
  /// For [`ClusterBackend::Offline`], assembles the
  /// [`diaric::offline::OfflineInput`] bridge ([`Self::into_offline_input`]) with
  /// the variant's [`OfflineOptions`](crate::audio::speaker::cluster::OfflineOptions) applied
  /// over it (its crate-private `apply_to`) and runs
  /// [`diaric::offline::diarize_offline`] over the result. For
  /// [`ClusterBackend::Online`], delegates to [`Self::diarize_online`] with the
  /// variant's [`OnlineOptions`]. The `match` on
  /// `backend` is wildcard-free: any future engine variant forces a new arm
  /// here rather than silently routing to an existing path.
  ///
  /// # `plda` is consumed by `Offline` only
  /// `plda` threads into the offline bridge (see [`Self::into_offline_input`]);
  /// the [`Online`](ClusterBackend::Online) route IGNORES it. FluidAudio's
  /// greedy matcher works on RAW cosine embeddings with no PLDA projection
  /// (design spec §Architecture point 3; T4's semantics table), so
  /// `diarize_with(plda, ClusterBackend::Online(opts))` is exactly
  /// `self.diarize_online(opts)` with `plda` unused. Prefer
  /// [`Self::diarize_online`] directly when you want the online engine and have
  /// no PLDA to supply — its signature takes none, so the absence is a fact of
  /// the API rather than an argument quietly discarded.
  ///
  /// The returned [`diaric::offline::OfflineOutput`] carries the speaker-labelled
  /// spans ([`diaric::offline::OfflineOutput::spans_slice`]) plus the frame-level
  /// diarization grid and per-chunk hard assignments. `plda` is the frozen
  /// community-1 PLDA projection ([`diaric::plda::PldaTransform`]); see
  /// [`Self::into_offline_input`] for how it threads through the bridge.
  ///
  /// Un-gated: `diaric` is a runtime dependency and `diaric::offline` is part of its
  /// ort-free clustering surface, so this runs without `ort` (the parity crate's
  /// `speaker-oracle` test feature only adds dia's ONNX reference oracle, never a
  /// runtime requirement).
  ///
  /// # Errors
  ///
  /// Propagates [`diaric::offline::diarize_offline`]'s typed
  /// [`diaric::offline::Error`] verbatim: a tensor-shape mismatch, a degenerate
  /// (zero-norm/NaN) raw embedding rejected by PLDA, a non-finite
  /// segmentation, or a clustering bail-out — e.g. the deliberate
  /// `Pipeline(Centroid(AmbiguousAliveCluster { .. }))` refusal when a
  /// cluster's alive-value lands in the SIMD guard band around the threshold.
  /// Keeping the error TYPED (not stringified) is load-bearing: the
  /// shipping-DER suite matches that exact variant rather than `is_err`.
  pub fn diarize_with(
    &self,
    plda: &diaric::plda::PldaTransform,
    backend: ClusterBackend,
  ) -> Result<diaric::offline::OfflineOutput, diaric::offline::Error> {
    match backend {
      ClusterBackend::Offline(opts) => {
        diaric::offline::diarize_offline(&opts.apply_to(self.into_offline_input(plda)))
      }
      // `plda` is deliberately NOT forwarded: the online engine matches raw
      // cosine embeddings, not PLDA-projected ones (see the doc's "`plda` is
      // consumed by `Offline` only" and [`Self::diarize_online`]).
      ClusterBackend::Online(opts) => self.diarize_online(opts),
    }
  }

  /// Cluster this extraction into speaker-labelled spans with the ONLINE
  /// (streaming) engine — FluidAudio's greedy centroid matcher, ported in diaric as
  /// [`diaric::cluster::online::OnlineClusterer`] — tuned by
  /// [`OnlineOptions`]. This is
  /// [`Self::diarize_with`]'s [`ClusterBackend::Online`] route, exposed directly
  /// because the online engine takes NO `plda`: it matches RAW L2-normalized
  /// WeSpeaker embeddings by cosine distance, and the PLDA projection the
  /// offline pipeline applies has no part in it (design spec §Architecture
  /// point 3; T4's semantics table, "Cosine on raw WeSpeaker embeddings, no
  /// PLDA"). Making the absence of `plda` a fact of the signature — rather than
  /// an argument silently ignored — is the honest surface.
  ///
  /// # What it does
  /// Feeds each `(chunk, slot)`'s raw embedding to the clusterer in **chunk
  /// order, then slot order within the chunk** — the exact order FluidAudio's
  /// `DiarizerManager` feeds `SpeakerManager` (`Core/DiarizerManager.swift:351`)
  /// and the ONE order this order-DEPENDENT engine is defined at here
  /// (deterministic given a fixed extraction). Per slot:
  /// - a slot whose segmentation column is INACTIVE (no frame with `seg > 0.0`)
  ///   is skipped and left unmatched, whatever its row holds. Its speech
  ///   duration would be `0`, but the engine's duration gate is not what would
  ///   stop it: `assign` matches the nearest centroid and updates it BEFORE
  ///   that gate is read, so an inactive row would move a speaker's centroid.
  ///   The offline route excludes the same slot at `filter_embeddings`
  ///   (`diarization/src/offline/algo.rs:645-679`), so both engines now agree
  ///   an empty column contributes nothing;
  /// - a dropped slot (all-zero raw-embedding row —
  ///   [`diaric::embed::Embedding::normalize_from`] rejects its zero norm) is
  ///   likewise skipped and left unmatched;
  /// - otherwise the row is L2-normalized into a [`diaric::embed::Embedding`] and
  ///   assigned, with a speech duration of `active_frame_count ×
  ///   frames_sw.step` seconds — FluidAudio's `Float(activity) *
  ///   slidingWindow.step` (`DiarizerManager.swift:357`), where `activity` is
  ///   the slot's nonzero-segmentation frame count — which gates new-speaker
  ///   creation vs. drop inside the engine.
  ///
  /// The per-slot speaker labels become the `hard_clusters` fed to the SAME
  /// reconstruction the offline path uses ([`diaric::reconstruct::reconstruct`] →
  /// [`diaric::reconstruct::try_discrete_to_spans`]); only the cluster labels come
  /// from a different engine. The result is a [`diaric::offline::OfflineOutput`]
  /// (the type name refers to diaric's `offline` module, not the engine — here it
  /// carries the online greedy assignment) with the speaker-labelled spans, the
  /// frame-level grid, and the per-chunk hard assignment.
  ///
  /// Online ids are the engine's dense `u64` from 1; they are mapped to the
  /// 0-based cluster indices [`diaric::reconstruct::reconstruct`] expects.
  ///
  /// # NOT pyannote-parity
  /// The online engine is order-dependent and its gate is parity with
  /// FluidAudio's Swift `SpeakerManager` (`tests/parity_online_swift.rs`), never
  /// DER against pyannote. See
  /// [`OnlineOptions`] and diaric's `cluster::online`.
  ///
  /// # Errors
  /// Every failure routes through [`diaric::offline::Error::Reconstruct`]: a
  /// non-finite segmentation, invalid sliding-window timing, a
  /// `ShapeError::CountLenMismatch` when the output-frame grid this method
  /// re-derives from `(chunks_sw, frames_sw, num_chunks)` is not
  /// [`Self::num_output_frames`] long — raised BEFORE that grid is allocated,
  /// see the check's own comment — or, only for a degenerate input that spawns
  /// more than [`diaric::reconstruct::MAX_CLUSTER_ID`] + 1 speakers, an
  /// out-of-range cluster id. The PLDA / pipeline / segment / embed error arms
  /// of [`diaric::offline::Error`] cannot fire here: the online path runs none
  /// of them.
  pub fn diarize_online(
    &self,
    opts: OnlineOptions,
  ) -> Result<diaric::offline::OfflineOutput, diaric::offline::Error> {
    use diaric::cluster::{
      hungarian::UNMATCHED,
      online::{Assignment, OnlineClusterer},
    };

    // `to_dia_options` builds the options through diaric's validating `with_*`
    // setters, so `try_new` cannot fail here; `diarize_online`'s
    // `diaric::offline::Error` has no arm for an online-options error anyway.
    let mut clusterer = OnlineClusterer::try_new(opts.to_dia_options())
      .expect("to_dia_options yields validated OnlineClusterOptions");
    let frame_step = self.frames_sw.step() as f32;

    // One `[i32; SEG_NUM_SLOTS]` row per chunk (dia's `ChunkAssignment`),
    // UNMATCHED (-2) for every slot until the engine labels it.
    let mut hard_clusters: Vec<diaric::pipeline::ChunkAssignment> =
      vec![[UNMATCHED; SEG_NUM_SLOTS]; self.num_chunks];

    // Feed each (chunk, slot) in chunk order, then slot order within the chunk
    // (iterating `hard_clusters` itself is that order and lets the label be
    // written straight into the slot). Self's tensors are read by the `(c, s)`
    // index alongside.
    for (c, chunk_row) in hard_clusters.iter_mut().enumerate() {
      for (s, slot) in chunk_row.iter_mut().enumerate() {
        // Speech duration = active-frame count × frame step (FluidAudio's
        // `Float(activity) * slidingWindow.step`, DiarizerManager.swift:357).
        // Binarized segmentations are 0/1; count nonzero frames — dia's own
        // `filter_embeddings` "any nonzero entry is binary-active" convention.
        let mut activity = 0usize;
        for f in 0..self.num_frames_per_chunk {
          if self.segmentations[(c * self.num_frames_per_chunk + f) * SEG_NUM_SLOTS + s] > 0.0 {
            activity += 1;
          }
        }

        // An INACTIVE column (no frame with `seg > 0.0`) contributes nothing,
        // and that has to be decided BEFORE the row reaches `assign` rather
        // than by the duration it would carry. `OnlineClusterer::assign`
        // consults `min_speech_duration` only in its NEW-speaker arm: it first
        // matches the nearest centroid and, when that match is inside
        // `speaker_threshold`, runs `update_existing` — MOVING that centroid —
        // and returns before any duration gate is read. So a zero-duration row
        // still perturbs the engine's state, and a later slot that would have
        // joined the unpolluted speaker can then fall outside the threshold and
        // spawn a second one: one speaker silently becomes two
        // (`an_inactive_slots_row_cannot_change_the_online_result`).
        //
        // Skipping here makes the online route read exactly what the offline
        // route reads: dia's `filter_embeddings` requires
        // `clean_frames >= 0.2 * num_frames_per_chunk` before a `(chunk, slot)`
        // reaches PLDA at all (`diarization/src/offline/algo.rs:645-679`), and
        // an all-zero column sums to `0`. The slot stays UNMATCHED — where it
        // already ended up whenever the row was the all-zero one both in-crate
        // producers write into every dropped slot — so this changes nothing for
        // any `Extraction` [`Extractor::extract`] or the argmax source
        // produced, only for one a caller ASSEMBLED with a live row under a
        // dead column.
        if activity == 0 {
          continue;
        }

        // Raw embedding row for (c, s). A dropped slot's row is all-zero, so
        // `normalize_from` rejects it (zero norm) and the slot stays UNMATCHED.
        let range = embedding_range(c, s);
        let mut row = [0.0f32; EMBEDDING_DIM];
        row.copy_from_slice(&self.raw_embeddings[range]);
        let Some(embedding) = diaric::embed::Embedding::normalize_from(row) else {
          continue;
        };

        let speech_duration = activity as f32 * frame_step;

        match clusterer.assign(&embedding, speech_duration) {
          Assignment::New(id) => {
            // The online engine just appended global speaker `id` (dense u64
            // from 1); its 0-based cluster label is `id - 1`. diaric's
            // `reconstruct` rejects any label `> MAX_CLUSTER_ID`
            // (reconstruct/algo.rs). Cap the loop at that ceiling HERE — the
            // moment a newly-created speaker would exceed it — instead of
            // labelling on and letting `reconstruct` reject late. Without this,
            // a pathological but VALID option set (`speaker_threshold = 0` and
            // `min_speech_duration = 0`, both accepted: with threshold 0 the
            // cosine `distance >= 0` never matches, and with min-speech 0 every
            // row's `duration >= 0` spawns a NEW speaker) creates one speaker
            // per active slot, and each further `assign` keeps a centroid and
            // rescans all priors — unbounded O(N^2) work / O(N) heap before the
            // inevitable typed error. The error returned here is the identical
            // `HardClustersIdAboveMax` that `reconstruct` would raise; only its
            // timing changes, so the online loop retains at most
            // `MAX_CLUSTER_ID + 1` (1024) speakers. Computed in u64 so `id - 1`
            // cannot overflow i32 before the check.
            if id - 1 > diaric::reconstruct::MAX_CLUSTER_ID as u64 {
              return Err(diaric::offline::Error::Reconstruct(
                diaric::reconstruct::Error::Shape(
                  diaric::reconstruct::ShapeError::HardClustersIdAboveMax,
                ),
              ));
            }
            // Past the guard `id - 1 <= MAX_CLUSTER_ID` (1023), so it fits i32.
            *slot = i32::try_from(id - 1).expect("id - 1 <= MAX_CLUSTER_ID fits i32");
          }
          Assignment::Existing(id) => {
            // Existing matched a speaker whose label was already accepted
            // (<= MAX_CLUSTER_ID): the New arm returns before any speaker past
            // the ceiling is labelled, so an existing id cannot exceed it.
            *slot = i32::try_from(id - 1).expect("existing id - 1 <= MAX_CLUSTER_ID fits i32");
          }
          Assignment::Dropped => {} // stays UNMATCHED
        }
      }
    }

    // The online path CANNOT reuse `self.count`. `self.count` is the
    // segmentation-derived count of active LOCAL slots per frame
    // (`count_from_segmentations`), but the online engine can DROP a slot
    // (`Assignment::Dropped` leaves it UNMATCHED, in no cluster) and can COLLIDE
    // two local slots onto ONE global cluster, so `self.count[t]` can EXCEED the
    // number of distinct global clusters active at frame `t`. Offline `reconstruct`
    // treats its count as an injective per-cluster count (its top-K binarize marks
    // exactly `count[t]` clusters active by descending activation), so any `count[t]`
    // above the real cluster count would select a zero-activation PADDED column and
    // emit a phantom speaker/span. (The offline path is safe: its `count` and
    // `hard_clusters` come from the SAME pyannote assignment, so the injective
    // assumption holds.) Derive the count from the CLUSTERED assignment instead: the
    // number of DISTINCT active clusters per frame.
    //
    // `num_clusters_from_hard = (max non-negative label) + 1`, matching diaric's own
    // `max_cluster + 1` (reconstruct/algo.rs); UNMATCHED (-2) slots are excluded. It
    // is the ceiling of the distinct-cluster count derived below (and per (chunk,
    // frame) that count is additionally `<= SEG_NUM_SLOTS`); kept as the invariant's
    // ceiling, NOT to size any buffer — the direct count below allocates no cluster
    // axis.
    let num_clusters_from_hard = hard_clusters
      .iter()
      .flatten()
      .filter(|&&k| k >= 0)
      .max()
      .map_or(0, |&k| k as usize + 1);

    // Per (chunk, frame): the number of DISTINCT non-negative cluster labels among
    // the active slots (`seg > 0.0`). This equals, cell for cell, the column count
    // the deleted dense `clustered_seg` tensor produced — a column `k` was `1.0`
    // iff some active slot carried label `k`, and the count helper counted the
    // `1.0` columns — so `online_count` below is byte-for-byte identical to the old
    // buffer approach; the only difference is that we never materialize a
    // `chunks × frames × clusters` tensor (the process-OOM the old buffer's
    // unchecked `num_chunks * num_frames_per_chunk * num_clusters_from_hard`
    // allocation risked before diaric's own cluster cap could fire). At most
    // `SEG_NUM_SLOTS` labels are live at one cell, so an inline dedup over a fixed
    // `[i32; SEG_NUM_SLOTS]` scratch is O(slots²) with `slots <= 3` — no global
    // cluster axis, no allocation beyond this `chunks × frames` vector.
    let mut chunk_count = vec![0.0f64; self.num_chunks * self.num_frames_per_chunk];
    for c in 0..self.num_chunks {
      let row = hard_clusters[c];
      for f in 0..self.num_frames_per_chunk {
        let mut seen = [i32::MIN; SEG_NUM_SLOTS];
        let mut n_seen = 0usize;
        for (s, &k) in row.iter().enumerate() {
          if k < 0 {
            continue; // dropped/unmatched slot: in no cluster
          }
          if self.segmentations[(c * self.num_frames_per_chunk + f) * SEG_NUM_SLOTS + s] > 0.0
            && !seen[..n_seen].contains(&k)
          {
            seen[n_seen] = k;
            n_seen += 1;
          }
        }
        chunk_count[c * self.num_frames_per_chunk + f] = n_seen as f64;
      }
    }

    // The output-frame grid length the aggregator below will build, derived in
    // O(1) from the SAME `(last_chunk_end, frames_sw.step())` pair it uses.
    // `try_from_parts` runs this very call as its check 4, and `extract()`'s own
    // geometry keeps `num_output_frames` far below `usize::MAX`, so no
    // construction path can make it fail here.
    let last_chunk_end =
      self.chunks_sw.duration() + (self.num_chunks - 1) as f64 * self.chunks_sw.step();
    let derived_output_frames =
      crate::audio::speaker::window::try_num_output_frames(last_chunk_end, self.frames_sw.step())
        .expect(
          "every construction path proves this derivation succeeds: extract() bounds the geometry \
       by samples.len(), and try_from_parts runs this identical call as its check 4",
        );

    // Refuse a grid that does not match `self.num_output_frames` BEFORE building
    // it. `reconstruct` below requires `count.len() == num_output_frames` and
    // would raise this same `CountLenMismatch` anyway — but only after
    // `try_aggregate_output_frame_count` had allocated TWO `f64` buffers of the
    // derived length. `extract()` derives `count` from this very geometry, so
    // the two always agree there; a publicly assembled `Extraction`
    // ([`Self::try_from_parts`]) may instead declare finite, strictly positive
    // windows whose derived length is astronomically larger than its `count`
    // (`chunks_sw.duration = 1e13` over `frames_sw.step = 0.01` derives 1e15
    // frames = 8 PB per buffer), turning a guaranteed typed refusal into an
    // allocation-failure abort. Same shape as the MAX_CLUSTER_ID early cap
    // above: the identical typed error `reconstruct` would raise, only sooner.
    if derived_output_frames != self.num_output_frames {
      return Err(diaric::offline::Error::Reconstruct(
        diaric::reconstruct::Error::Shape(diaric::reconstruct::ShapeError::CountLenMismatch),
      ));
    }

    // The SAME overlap-add + rounding `count_from_segmentations` runs, over the
    // distinct-cluster chunk count. The geometry (`num_chunks`/`num_frames_per_chunk`/
    // `chunks_sw`/`frames_sw`) is IDENTICAL to the one `extract()` already ran to
    // derive `self.num_output_frames`, so `online_count.len() ==
    // self.num_output_frames` and the output-frame count cannot overflow. The
    // all-dropped case falls out naturally: every label negative → every `chunk_count`
    // cell `0` → `online_count` all-zero (length `self.num_output_frames`), and
    // reconstruct's `max_cluster < 0` early-return fires regardless.
    let online_count = crate::audio::speaker::window::try_aggregate_output_frame_count(
      &chunk_count,
      self.num_chunks,
      self.num_frames_per_chunk,
      self.chunks_sw,
      self.frames_sw,
    )
    .expect(
      "the identical derivation over the identical (last_chunk_end, frames_sw.step()) pair \
       returned Ok a few lines above, so this one cannot overflow",
    );

    // Invariant preserved from the deleted buffer approach: distinct labels at any
    // one (chunk, frame) ≤ total distinct labels = `num_clusters_from_hard`, and the
    // overlap-add average + `round_ties_even` is monotone (a mean of integers each
    // `<= K` rounds to `<= K`), so `max(online_count) <= num_clusters_from_hard`. No
    // padded column, no phantom speaker — the M1 correctness fix is fully preserved.
    debug_assert!(
      online_count
        .iter()
        .all(|&t| usize::from(t) <= num_clusters_from_hard),
      "distinct-cluster online_count must not exceed num_clusters_from_hard"
    );

    // The SAME reconstruction the offline path runs — only the cluster labels came
    // from the online engine instead of AHC→VBx, and the count is the
    // clustered-segmentation count derived just above (NOT `self.count`).
    // `reconstruct` derives its own cluster count from `hard_clusters` + `count`.
    let recon_input = diaric::reconstruct::ReconstructInput::new(
      self.segmentations.as_slice(),
      self.num_chunks,
      self.num_frames_per_chunk,
      SEG_NUM_SLOTS,
      &hard_clusters,
      online_count.as_slice(),
      self.num_output_frames,
      self.chunks_sw.into(),
      self.frames_sw.into(),
    );
    let discrete = diaric::reconstruct::reconstruct(&recon_input)?;

    // The grid is `num_output_frames × num_clusters` row-major, so its width IS
    // the cluster count — the single source of truth for both the span
    // conversion and the stored metadata. Deriving it from the grid (rather than
    // recomputing dia's `num_clusters_from_hard.max(max_count.max(1))`) is always
    // shape-consistent, INCLUDING reconstruct's all-UNMATCHED zero-return path
    // (width 1), which a `count`-inflated recomputation would mismatch — that is
    // the reachable "every slot dropped" outcome for a short clip at the default
    // `min_speech_duration`. `num_output_frames > 0` holds here: `reconstruct`
    // rejects a zero-frame grid before returning `Ok`.
    let num_clusters = discrete.as_slice().len() / self.num_output_frames;
    let spans = diaric::reconstruct::try_discrete_to_spans(
      discrete.as_slice(),
      self.num_output_frames,
      num_clusters,
      self.frames_sw.into(),
      // Online exposes no gap-merge knob; 0.0 = no merge, dia's own default.
      0.0,
    )
    .map_err(diaric::reconstruct::Error::from)?;

    Ok(diaric::offline::OfflineOutput::new(
      std::sync::Arc::from(hard_clusters),
      discrete,
      num_clusters,
      std::sync::Arc::from(spans),
    ))
  }
}

/// The flat `segmentations` sub-slice for chunk `c`: `c * F * S .. (c + 1)
/// * F * S`, where `F = num_frames` and `S = SEG_NUM_SLOTS`. Indexes the
/// `[c][f][s]` buffer at dia's `owned.rs:496` layout.
fn chunk_segmentation_range(c: usize, num_frames: usize) -> core::ops::Range<usize> {
  let stride = num_frames * SEG_NUM_SLOTS;
  c * stride..(c + 1) * stride
}

/// The flat `raw_embeddings` sub-slice for `(chunk c, slot s)`: `(c * S +
/// s) * EMBEDDING_DIM .. + EMBEDDING_DIM`. dia's write offset `dst = (c *
/// SLOTS_PER_CHUNK + s) * EMBEDDING_DIM` (`owned.rs:631`).
fn embedding_range(c: usize, s: usize) -> core::ops::Range<usize> {
  let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
  base..base + EMBEDDING_DIM
}

/// Copies the chunk window starting at sample `start` into `padded`,
/// zero-clearing first and leaving any out-of-range tail zero. Exact shape
/// of dia's per-chunk build (`owned.rs:469-475`), including the `.min`
/// clamps that keep a `start` at or beyond `samples.len()` from panicking
/// (it yields an all-zero padded chunk).
fn fill_padded_chunk(padded: &mut [f32], samples: &[f32], start: usize) {
  padded.fill(0.0);
  let end = (start + SEG_CHUNK_SAMPLES).min(samples.len());
  let lo = start.min(samples.len());
  let n = end - lo;
  if n > 0 {
    padded[..n].copy_from_slice(&samples[lo..end]);
  }
}

/// Zeroes exactly slot `s`'s column across all `num_frames` frames of one
/// chunk's `[f][s]` slab, leaving the other slots untouched. dia's
/// column-zero on a dropped `(chunk, slot)` (`owned.rs:567-569,626-628`).
fn zero_slot_column(chunk_segs: &mut [f64], num_frames: usize, s: usize) {
  for f in 0..num_frames {
    chunk_segs[f * SEG_NUM_SLOTS + s] = 0.0;
  }
}

/// The per-slot embedding decision for one chunk, from the
/// overlap-exclusion rule (`owned.rs:507-591`): either [`Self::Skip`] (no
/// active frame — no embed, column zeroed) or [`Self::Embed`] with the
/// exact per-frame boolean mask to pool over.
#[derive(Debug, PartialEq)]
enum SlotPlan {
  /// No frame is active for this slot; it is dropped (no embed call, its
  /// segmentation column is zeroed).
  Skip,
  /// Embed this slot with the given per-frame mask (`num_frames` long) —
  /// the overlap-excluded clean mask, or (via the `<=`-fallback) the raw
  /// active mask.
  Embed(Vec<bool>),
}

/// The overlap-exclusion mask derivation for one chunk's `[f][s]` slab —
/// THE critical port (`owned.rs:507-591`; see the module doc's "The
/// critical port" section for the adjudicated semantics). Returns one
/// [`SlotPlan`] per slot.
///
/// `chunk_segs` is `num_frames * SEG_NUM_SLOTS` f64 values, frame-major
/// (`chunk_segs[f * SEG_NUM_SLOTS + s]`) — one chunk's [`crate::audio::speaker::segment::multilabel`]
/// output. `onset` is the (already-validated) f64 threshold.
///
/// # Panics
/// Panics if `chunk_segs.len() != num_frames * SEG_NUM_SLOTS`.
fn derive_slot_plans(
  chunk_segs: &[f64],
  num_frames: usize,
  onset: f64,
) -> [SlotPlan; SEG_NUM_SLOTS] {
  assert_eq!(
    chunk_segs.len(),
    num_frames * SEG_NUM_SLOTS,
    "chunk_segs.len() must equal num_frames * SEG_NUM_SLOTS"
  );

  // Per-frame "clean" indicator: fewer than 2 of the SEG_NUM_SLOTS slots
  // active (`seg >= onset`, inclusive). Computed ONCE over all slots,
  // BEFORE the per-slot loop, from the pre-zeroing values (owned.rs:536-549).
  let mut clean_frame = vec![false; num_frames];
  for f in 0..num_frames {
    let mut active_count = 0u8;
    for s in 0..SEG_NUM_SLOTS {
      if chunk_segs[f * SEG_NUM_SLOTS + s] >= onset {
        active_count += 1;
      }
    }
    clean_frame[f] = active_count < 2;
  }

  let mut plans: [SlotPlan; SEG_NUM_SLOTS] = core::array::from_fn(|_| SlotPlan::Skip);
  for s in 0..SEG_NUM_SLOTS {
    // Raw active mask for this slot (owned.rs:552-560).
    let mut frame_mask = vec![false; num_frames];
    let mut any_active = false;
    for f in 0..num_frames {
      let active = chunk_segs[f * SEG_NUM_SLOTS + s] >= onset;
      frame_mask[f] = active;
      any_active |= active;
    }
    if !any_active {
      // No active frame → drop (owned.rs:561-571). plans[s] stays Skip.
      continue;
    }

    // Overlap-excluded clean mask + clean-active frame count
    // (owned.rs:573-591).
    let mut used_mask = vec![false; num_frames];
    let mut clean_count = 0usize;
    for f in 0..num_frames {
      let v = frame_mask[f] && clean_frame[f];
      used_mask[f] = v;
      if v {
        clean_count += 1;
      }
    }
    // Per-slot fallback: use the raw mask when too few clean frames remain
    // (`<=`, per owned.rs:589). Replaces only THIS slot's mask.
    if clean_count <= EXCLUDE_OVERLAP_MIN_FRAMES {
      used_mask = frame_mask;
    }
    plans[s] = SlotPlan::Embed(used_mask);
  }
  plans
}

#[cfg(test)]
mod tests;
