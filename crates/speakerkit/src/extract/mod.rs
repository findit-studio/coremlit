//! The extraction bridge: run segmentation + embedding over a whole clip
//! and assemble the exact tensor set diaric's offline diarizer consumes.
//!
//! [`Extractor::extract`] is the composition layer over Tasks 2-4
//! ([`crate::segment`], [`crate::embed`], [`crate::window`]): it ports the
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
//!    list. One guard has no dia analog: [`crate::error::ExtractError::FrameCountMismatch`].
//! 2. **Chunk grid + zero-padding** (`owned.rs:447-475`): [`crate::window::chunk_starts`]
//!    schedules `start = c * step`; each chunk is copied into a reused
//!    `SEG_CHUNK_SAMPLES` buffer with the out-of-range tail left zero
//!    (`fill_padded_chunk`).
//! 3. **Segment → multilabel** (`owned.rs:477-498`): [`crate::segment::SegmentModel::infer`]
//!    then [`crate::segment::multilabel`] (whose own module doc proves it
//!    equals dia's inline `softmax_row` + `powerset_to_speakers_hard`
//!    decode). Each chunk's `[f][s]` slab is written into the flat
//!    `segmentations` buffer at `chunk_segmentation_range` — dia's
//!    `segs[(c * FRAMES_PER_WINDOW + f) * SLOTS_PER_CHUNK + s]` layout
//!    (`owned.rs:496`).
//! 4. **Mask derivation** (`owned.rs:507-591`): the overlap-exclusion rule
//!    (`derive_slot_plans`). See "The critical port" below.
//! 5. **Masked embedding + drop paths** (`owned.rs:600-632`):
//!    [`crate::embed::EmbedModel::embed_chunk`], the non-finite hard error
//!    (`owned.rs:611-618`), and the PLDA-norm drop (`owned.rs:619-630`).
//! 6. **Count tensor + sliding windows** (`owned.rs:653-674`):
//!    `crate::window::try_count_from_segmentations` over the
//!    POST-drop-zeroing `segmentations` buffer, plus
//!    [`crate::window::chunk_sliding_window`] / [`crate::window::frame_sliding_window`].
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
//!   [`crate::segment::SEG_NUM_SLOTS`] slots, BEFORE the per-slot loop:
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
//!   ([`crate::embed`]'s "Batching design"), and an all-false mask row is
//!   the known statistics-pooling divide-by-zero NaN mode
//!   ([`crate::embed`]'s "NonFinite-output scan scope";
//!   [`crate::error::InferError::EmptyMask`]'s doc). So a chunk with at
//!   least one planned (`SlotPlan::Embed`) slot makes ONE batched call
//!   in which every `SlotPlan::Skip` slot's mask row borrows the first
//!   planned slot's mask (a non-degenerate placeholder), and those
//!   placeholder OUTPUT rows are discarded — the corresponding
//!   `raw_embeddings` rows stay zero, identical to dia's pre-zeroed,
//!   never-written rows (`owned.rs:502-505`). A chunk with NO planned slot
//!   makes no embed call at all (= dia's zero calls for such a chunk).
//!
//!   Divergence: `embed_chunk`'s 768-wide non-finite scan
//!   ([`crate::embed`]'s "NonFinite-output scan scope") also covers the
//!   placeholder rows, so a NaN confined to a placeholder row would
//!   hard-error here where dia computes no such row. Accepted, because the
//!   placeholder mask is bit-identical to a real slot's mask over the same
//!   audio, and dia hard-errors on exactly that mask + audio anyway
//!   (`owned.rs:616-618`).
//! - **No `InvalidClip` / `DegenerateEmbedding` recoverable paths.** dia
//!   silently drops a slot on those two embed errors (`owned.rs:602-608`).
//!   Neither exists here: [`crate::embed::EmbedModel::embed_chunk`]
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

use crate::{
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
/// [`crate::source::ArgmaxSource`] applies the SAME rule to argmax's own
/// tensors (its module doc's "The overlap-exclusion fallback" section), and
/// `tests/parity_argmax_swift.rs` — a separate crate, so `pub(crate)` cannot
/// reach it — asserts the fallback never fires on any consumed slot. All
/// three name this ONE constant rather than each declaring their own `2`, so
/// none of them can drift apart.
pub const EXCLUDE_OVERLAP_MIN_FRAMES: usize = 2;

/// PLDA minimum raw-embedding L2 norm: a slot whose raw embedding has a
/// smaller norm is dropped (its column zeroed, its row left zero) before
/// it can reach PLDA. Matches dia's inline `0.01` guard
/// (`diarization/src/offline/owned.rs:619-630`), which pre-validates the
/// norm `RawEmbedding::from_raw_array` would otherwise reject downstream.
const PLDA_MIN_NORM: f64 = 0.01;

#[cfg(feature = "serde")]
fn default_segmenter_compute() -> coremlit::ComputeUnits {
  crate::segment::DEFAULT_SEGMENT_COMPUTE
}

#[cfg(feature = "serde")]
fn default_embedder_compute() -> coremlit::ComputeUnits {
  crate::embed::DEFAULT_EMBED_COMPUTE
}

/// Which hardware CoreML may schedule each model on (rust-options-pattern).
///
/// These live on the extractor's [`Options`] even though
/// [`Extractor::extract`] takes already-loaded models and never reads
/// them: `Options` is the one serializable configuration surface a
/// consumer reads to LOAD the two models in the first place (design spec
/// §5, `docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md`)
/// — `segmenter` feeds [`crate::segment::SegmentModelOptions`], `embedder`
/// feeds [`crate::embed::EmbedModelOptions`]. Keeping them here lets a
/// single deserialized `Options` drive both the model loads and the
/// extraction geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_segmenter_compute",
      with = "crate::compute_units_serde"
    )
  )]
  segmenter: coremlit::ComputeUnits,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_embedder_compute",
      with = "crate::compute_units_serde"
    )
  )]
  embedder: coremlit::ComputeUnits,
}

impl Default for ComputeOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ComputeOptions {
  /// Options matching the crate defaults:
  /// [`crate::segment::DEFAULT_SEGMENT_COMPUTE`] for the segmenter and
  /// [`crate::embed::DEFAULT_EMBED_COMPUTE`] for the embedder (both
  /// `ComputeUnits::All`).
  pub const fn new() -> Self {
    Self {
      segmenter: crate::segment::DEFAULT_SEGMENT_COMPUTE,
      embedder: crate::embed::DEFAULT_EMBED_COMPUTE,
    }
  }

  /// Hardware the segmentation model may be scheduled on.
  #[inline(always)]
  pub const fn segmenter(&self) -> coremlit::ComputeUnits {
    self.segmenter
  }
  /// Hardware the embedding model may be scheduled on.
  #[inline(always)]
  pub const fn embedder(&self) -> coremlit::ComputeUnits {
    self.embedder
  }

  /// Builder form of [`Self::set_segmenter`].
  #[must_use]
  #[inline(always)]
  pub const fn with_segmenter(mut self, segmenter: coremlit::ComputeUnits) -> Self {
    self.set_segmenter(segmenter);
    self
  }
  /// Sets [`Self::segmenter`] in place.
  #[inline(always)]
  pub const fn set_segmenter(&mut self, segmenter: coremlit::ComputeUnits) -> &mut Self {
    self.segmenter = segmenter;
    self
  }
  /// Builder form of [`Self::set_embedder`].
  #[must_use]
  #[inline(always)]
  pub const fn with_embedder(mut self, embedder: coremlit::ComputeUnits) -> Self {
    self.set_embedder(embedder);
    self
  }
  /// Sets [`Self::embedder`] in place.
  #[inline(always)]
  pub const fn set_embedder(&mut self, embedder: coremlit::ComputeUnits) -> &mut Self {
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
/// The field is read by [`crate::source::AnySource::load`], the dispatcher
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
  /// and [`crate::source::DEFAULT_SOURCE`] — each component's own default
  /// is the single source of truth (the `serde(default)` on each field
  /// defers to it; nested partial configs are covered by each component's
  /// own per-field serde defaults).
  pub const fn new() -> Self {
    Self {
      window: WindowOptions::new(),
      compute: ComputeOptions::new(),
      source: crate::source::DEFAULT_SOURCE,
    }
  }

  /// The sliding-window geometry ([`crate::window::chunk_starts`] step and
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
  /// [`crate::source::AnySource::load`], not by [`Extractor::extract`] (see
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
  /// - [`ExtractError::Infer`] (via `#[from]`) if either model's inference
  ///   fails (`owned.rs:477,600`).
  /// - [`ExtractError::OutputFrameCountOverflow`] if the derived
  ///   `num_output_frames` would not fit in `usize` (converted from
  ///   [`crate::window`]'s `WindowError` by exhaustive match —
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
    if !crate::window::check_onset(w.onset()) {
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
    let starts = crate::window::chunk_starts(samples.len(), &w); // owned.rs:447-451
    let num_chunks = starts.len();
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
      let slab = crate::segment::multilabel(&logits, num_frames);
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
          // Exact f64 arithmetic shape of dia's norm pre-check
          // (owned.rs:619-630). Finite by `embed_chunk`'s own hard scan,
          // so `< PLDA_MIN_NORM` is the only branch that can fire here.
          let norm_sq: f64 = rows[s].iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
          if norm_sq.sqrt() < PLDA_MIN_NORM {
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
    let chunks_sw = crate::window::chunk_sliding_window(&w); // owned.rs:653-655
    let frames_sw = crate::window::frame_sliding_window(); // owned.rs:656-657
    // Manual exhaustive match, deliberately not a `From` impl — see
    // `ExtractError::OutputFrameCountOverflow`'s doc. Unreachable through
    // extract's own geometry (num_chunks * step ≈ samples.len()), kept
    // typed regardless (owned.rs:663-673).
    let count = crate::window::try_count_from_segmentations(
      &segmentations,
      num_chunks,
      num_frames,
      SEG_NUM_SLOTS,
      w.onset(),
      chunks_sw,
      frames_sw,
    )
    .map_err(|e| match e {
      crate::window::WindowError::OutputFrameCountOverflow => {
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
  /// The single construction site for an [`Extraction`], shared by every
  /// [`crate::source::ModelSource`] (crate-private: the field set is an
  /// implementation detail, and each source assembles it its own way — see
  /// [`crate::source::argmax`], which builds the identical layout from
  /// argmax's in-graph-decoded tensors instead of a host-side decode).
  ///
  /// `num_output_frames` is not a parameter: it IS `count.len()`
  /// (`owned.rs:674`), so deriving it here makes the two impossible to
  /// disagree.
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
  /// [`crate::window`]'s `From` impls (`window/mod.rs`); `OfflineInput::new`
  /// takes `diaric::reconstruct::SlidingWindow` by value (`algo.rs:11,224-225`).
  ///
  /// Un-gated: `diaric` is a runtime dependency and `diaric::offline` is part of
  /// its ort-free clustering surface, so this bridge is always available.
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
  /// the variant's [`OfflineOptions`](crate::cluster::OfflineOptions) applied
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
  /// ort-free clustering surface, so this runs without `ort` (the `dia-oracle`
  /// test feature only adds dia's ONNX reference oracle, never a runtime
  /// requirement).
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
  /// - a dropped slot (all-zero raw-embedding row —
  ///   [`diaric::embed::Embedding::normalize_from`] rejects its zero norm) is
  ///   skipped and left unmatched;
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
  /// non-finite segmentation, invalid sliding-window timing, or — only for a
  /// degenerate input that spawns more than
  /// [`diaric::reconstruct::MAX_CLUSTER_ID`] + 1 speakers — an out-of-range cluster
  /// id. The PLDA / pipeline / segment / embed error arms of
  /// [`diaric::offline::Error`] cannot fire here: the online path runs none of them.
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
        // Raw embedding row for (c, s). A dropped slot's row is all-zero, so
        // `normalize_from` rejects it (zero norm) and the slot stays UNMATCHED.
        let range = embedding_range(c, s);
        let mut row = [0.0f32; EMBEDDING_DIM];
        row.copy_from_slice(&self.raw_embeddings[range]);
        let Some(embedding) = diaric::embed::Embedding::normalize_from(row) else {
          continue;
        };

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
        let speech_duration = activity as f32 * frame_step;

        match clusterer.assign(&embedding, speech_duration) {
          Assignment::New(id) | Assignment::Existing(id) => {
            // Dense u64 ids from 1 → 0-based cluster indices. `try_from`
            // cannot fail for realistic speaker counts; a pathological
            // overflow surfaces later as reconstruct's HardClustersIdAboveMax
            // (a typed error), never a panic here.
            *slot = i32::try_from(id - 1).unwrap_or(i32::MAX);
          }
          Assignment::Dropped => {} // stays UNMATCHED
        }
      }
    }

    // The SAME reconstruction the offline path runs — only the cluster labels
    // came from the online engine instead of AHC→VBx. `reconstruct` derives its
    // own cluster count from `hard_clusters` + `count`.
    let recon_input = diaric::reconstruct::ReconstructInput::new(
      self.segmentations.as_slice(),
      self.num_chunks,
      self.num_frames_per_chunk,
      SEG_NUM_SLOTS,
      &hard_clusters,
      self.count.as_slice(),
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
/// (`chunk_segs[f * SEG_NUM_SLOTS + s]`) — one chunk's [`crate::segment::multilabel`]
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
