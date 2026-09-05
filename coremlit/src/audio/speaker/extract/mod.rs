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
//! # The split pipeline's three doors
//!
//! [`Extractor::extract`] is the FUSED path: one call, both models, a whole
//! clip. A caller that instead runs `segmentation → embed → cluster` as three
//! autonomous nodes ([`ExtractionParts`]' doc, issue #110) drives stages 3-6
//! through three public doors, and each one is a face of this module's own
//! implementation rather than a second copy of it:
//!
//! | node | door |
//! |---|---|
//! | segmentation | [`crate::audio::speaker::segment::SegmentModel::infer`] + [`crate::audio::speaker::segment::multilabel`], on the chunk grid [`crate::audio::speaker::window::chunk_starts`] schedules |
//! | embed | [`Extractor::extract_chunk_embeddings`] — stage 4 and stage 5 for ONE chunk |
//! | cluster | [`crate::audio::speaker::window::count_from_segmentations`] then [`Extraction::try_from_parts`], at the track's end |
//!
//! **Stage 2 (chunk grid + zero-padding) has no door: it is not shared, it is
//! rebuilt.** A split-pipeline caller still constructs each
//! `SEG_CHUNK_SAMPLES` window itself — zero-padding the final, partial chunk
//! exactly as the fused loop's private `fill_padded_chunk` does — before
//! calling the segmentation door, and again, identically, before calling
//! [`Extractor::extract_chunk_embeddings`] (whose own `samples` doc explains
//! why: a short slice is refused rather than padded there, because
//! [`crate::audio::speaker::embed::EmbedModel::embed_chunk`] REPEAT-pads
//! whatever it is given, so accepting one here would silently embed
//! different audio than the fused path computed). That copy is small
//! (`fill_padded_chunk` is seven lines) but it is a real one, not a face of
//! anything published — this module's shared surface is stages 3-6, not the
//! whole pipeline.
//!
//! The embed door is per-CHUNK because that is the granularity at which the
//! stage decides anything, and because it lets `extract`'s loop and the split
//! node call ONE function (`embed_chunk_slots`) instead of two implementations
//! of the masking policy this crate parity-gates against dia (issue #127).
//! Publishing `derive_slot_plans` instead would have handed the caller the
//! decision and kept the loop — the drop paths, the placeholder-mask batching,
//! the zeroing order — as something they had to rebuild correctly.
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
//!
//! # The resource envelope (round 11)
//!
//! Three bounds hold this module's producer path, one per quantity that can
//! actually be exhausted, each enforced from GEOMETRY ALONE before a tensor is
//! allocated or a model is called:
//!
//! | bound | axis | what it holds |
//! |---|---|---|
//! | [`MAX_OUTPUT_FRAMES`] | output grid | the grid-shaped buffers — `count`, and the `aggregated` / `overlapping_count` pair rebuilt on every `diarize_online` |
//! | [`MAX_EXTRACTION_CHUNKS`] | chunk grid | COMPUTE: the model-call count, and `chunk_starts`' `8 * num_chunks` |
//! | [`MAX_EXTRACTION_TENSOR_BYTES`] | chunk grid x frame grid | MEMORY: `segmentations` + `raw_embeddings`, and through them the per-chunk frame count |
//!
//! The last two are both on the chunk axis and neither implies the other: the
//! byte ceiling is scaled by the loaded segmenter's frame count, so at the
//! loader's floor of one frame per chunk it still admits 393 349 chunks and
//! 786 698 model calls; the chunk ceiling cannot see the frame axis at all, so
//! it would admit a single chunk of any declared width.
//!
//! What a caller can vary, and which bound covers it:
//!
//! - **`samples.len()`** — derives `num_output_frames` (held by the frame cap)
//!   and `num_chunks` (held by the chunk cap, and the tensors it implies by the
//!   byte cap). The residual is the `O(samples.len())` pass each producer makes
//!   over the slice itself; it allocates nothing and is proportional to memory
//!   the caller already holds, the same argument
//!   [`Extraction::try_from_parts`]' check 3 makes for caller-supplied tensors.
//! - **`step_samples`** (`1..=SEG_CHUNK_SAMPLES`) — scales `num_chunks` and
//!   nothing else; it cancels out of the output grid entirely. Chunk cap, then
//!   byte cap.
//! - **`onset`** (finite, `(0.0, 1.0]`) — a scalar threshold read once per
//!   `(chunk, frame, slot)`. Scales no allocation and no loop count, so it needs
//!   no bound.
//! - **Compute units** — selects where a fixed number of calls run. Scales
//!   nothing this crate allocates or counts.
//! - **The loaded models' declared shapes** — only ONE dimension is free.
//!   [`crate::audio::speaker::segment::SegmentModel`] pins the `audio` input at
//!   `[1, 1, SEG_CHUNK_SAMPLES]` and `segments` at `[1, >=1, POWERSET_CLASSES]`,
//!   so `shape[1]` — `num_frames_per_chunk` — is the only caller-chosen value;
//!   [`crate::audio::speaker::embed::EmbedModel`] pins `embedding` at
//!   `[EMBED_SLOTS, EMBEDDING_DIM]`, and `extract`'s check 5 forces its free
//!   `mask[1]` to EQUAL the segmenter's. That one dimension is held by the byte
//!   cap: `num_chunks >= 1` always, so `24 * F + 3_072 <= 1_226_302_560` forces
//!   `F <= 51_095_812`. Every per-chunk temporary is `O(F)` with a small
//!   constant — `SegmentModel::infer`'s `28F`, `multilabel`'s `24F`,
//!   `build_masks`' `12F`, `derive_slot_plans`' `8F` plus `3F` of `bool` — so
//!   bounding the two persistent tensors bounds the transient ones with it.
//!   [`crate::audio::speaker::source::ArgmaxSource`] has no free dimension at
//!   all: every shape it consumes is pinned to its compiled graph.
//! - **How many times `extract` is called** — out of scope, and nothing
//!   accumulates across calls: the only process-wide state is the `OnceLock`
//!   holding one [`diaric::plda::PldaTransform`].
//!
//! No other quantity on this path scales with anything a caller supplies.

use std::sync::OnceLock;

use crate::audio::speaker::{
  cluster::{ClusterBackend, OnlineOptions},
  embed::{EMBED_SLOTS, EMBEDDING_DIM, EmbedModel},
  error::{ExtractError, FrameCountMismatch, StepSamplesExceedsWindow},
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

/// The largest output-frame grid this crate will assemble an [`Extraction`] on:
/// `2^22` frames. Enforced at EVERY construction path — [`Extraction::try_from_parts`]
/// and, since round 8, every in-crate
/// [`crate::audio::speaker::source::ModelSource`] through
/// `Extraction::assemble_checked`.
///
/// Since round 9 it is enforced BEFORE the work it bounds, not only at
/// assembly. The grid is a function of `num_chunks` and the two sliding windows
/// alone, so `checked_output_frame_count` derives it and applies this cap at
/// each producer's first opportunity — ahead of the extraction tensors, ahead
/// of every model call, and ahead of the count buffers `assemble_checked`
/// builds. The late check remains, because [`Extraction::try_from_parts`] has
/// no earlier point to run it at.
///
/// A RESOURCE bound, and the only check in that sequence not derived from the
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

/// The largest chunk grid this crate will run a producer over: `70_770` chunks.
/// Enforced from GEOMETRY ALONE in every producer's `checked_geometry` seam —
/// ahead of the `O(num_chunks)` placement scan, ahead of the two tensors, and
/// ahead of every model call.
///
/// # The axis a byte ceiling cannot see
///
/// [`MAX_EXTRACTION_TENSOR_BYTES`] bounds MEMORY, and memory per chunk is
/// `num_frames_per_chunk * SEG_NUM_SLOTS * size_of::<f64>() + SEG_NUM_SLOTS *
/// EMBEDDING_DIM * size_of::<f32>()` — a figure the LOADED SEGMENTER scales.
/// [`crate::audio::speaker::segment::SegmentModel`] accepts any declared
/// `segments` shape with `shape[1] >= 1`, so that figure bottoms out at
/// `SEG_NUM_SLOTS * (size_of::<f64>() + EMBEDDING_DIM * size_of::<f32>())` =
/// 3 096 bytes for a segmenter emitting ONE frame per ten-second chunk. Divide
/// the byte ceiling by that and the chunk axis reopens:
///
/// | quantity | value |
/// |---|---|
/// | `samples.len()` | 946 695 — 59.17 s, 3 786 780 bytes of `f32` |
/// | `step_samples` | 2 |
/// | `num_chunks` | 393 349 |
/// | tensors, total | 1 217 808 504 B — 1 656 B BELOW the byte ceiling |
/// | derived output frames | 3 507, against a cap of 4 194 304 |
/// | model calls | **786 698**, for 59.17 s of audio |
///
/// Every guard passed it, the byte ceiling included. Compute is not memory: a
/// cheap chunk is still a chunk, and [`Extractor::extract`] issues one
/// segmentation call and at most one batched embedding call for each one. So
/// this bound is model-INDEPENDENT by construction — it reads `num_chunks` and
/// nothing else — where the byte ceiling is necessarily model-relative.
///
/// # The number, derived
///
/// The FIRST chunk count [`MAX_OUTPUT_FRAMES`] refuses at the shipped
/// [`crate::audio::speaker::window::DEFAULT_STEP_SAMPLES`], admitted
/// INCLUSIVELY. That is not a new budget either: the output cap already declines
/// every grid past 70 769 chunks at the stride this crate ships and the stride
/// [`crate::audio::speaker::source::ArgmaxSource`]'s graph compiles in, so what
/// this constant does is make that same allowance hold at EVERY stride instead
/// of only at those two. `max_extraction_chunks_is_the_frame_caps_own_allowance_derived_not_copied`
/// searches for the number through
/// `checked_output_frame_count` rather than restating it, so a change to either
/// cap becomes a test failure instead of a silent drift.
///
/// Inclusive because round 10's byte ceiling was `70_770 * 17_208` — the
/// footprint OF that boundary count, so it admitted the count itself. Keeping
/// the chunk allowance inclusive keeps the shipped 589-frame grid's accepted set
/// byte-for-byte what round 10 accepted; an exclusive bound would newly refuse
/// the one geometry at exactly 70 770 chunks.
///
/// # What it bounds, per producer
///
/// - [`Extractor::extract`]: one `SegmentModel::infer` per chunk plus at most
///   one batched `EmbedModel` call per chunk — at most **141 540** model calls.
/// - [`crate::audio::speaker::source::ArgmaxSource`]: three calls per
///   21-window argmax chunk (`ARGMAX_WINDOWS_PER_CHUNK`), i.e.
///   `3 * 70_770.div_ceil(21)` = at most **10 110**.
///
/// Both are proportional to `num_chunks` with a fixed per-producer constant, so
/// bounding the chunk count IS bounding the call count; a separate constant per
/// producer would be the same number written twice.
///
/// # NOT part of the fourteen-check assembly sequence
///
/// For the reason [`MAX_EXTRACTION_TENSOR_BYTES`] is not: see that constant's
/// own doc. [`Extraction::try_from_parts`] calls no model at all, so there is no
/// compute for this bound to hold there.
pub const MAX_EXTRACTION_CHUNKS: usize = 70_770;

/// The largest `segmentations` + `raw_embeddings` footprint this crate will
/// allocate for, in bytes: `1_226_302_560` (1.142 GiB). Enforced from GEOMETRY
/// ALONE in every producer's `checked_geometry` seam — ahead of the
/// `O(num_chunks)` placement scan, ahead of the two tensors, and ahead of every
/// model call.
///
/// # The axis [`MAX_OUTPUT_FRAMES`] cannot see
///
/// That cap bounds the OUTPUT grid, and the output grid is a function of the
/// clip's DURATION: `num_output_frames = round(last_chunk_end / FRAME_STEP_S) +
/// 1`, with `last_chunk_end = CHUNK_DURATION_S + (num_chunks - 1) *
/// step_samples / SAMPLE_RATE_HZ`. The two `step_samples` factors cancel, so the
/// derived grid is very nearly `samples.len() / 270` whatever stride the caller
/// picked.
///
/// The producers do not allocate on that axis. They allocate on the CHUNK axis,
/// `num_chunks = 1 + (samples.len() - SEG_CHUNK_SAMPLES).div_ceil(step_samples)`,
/// which grows without bound as `step_samples` shrinks — and
/// [`WindowOptions::set_step_samples`] constrains that only to
/// `1..=SEG_CHUNK_SAMPLES` — and on the loaded segmenter's per-chunk FRAME
/// count, which [`crate::audio::speaker::segment::SegmentModel`] constrains only
/// to `shape[1] >= 1`. A ten-minute clip (9 600 000 samples, 38 400 000 bytes of
/// `f32`) at `step_samples = 2` derives:
///
/// | quantity | value |
/// |---|---|
/// | `num_chunks` | 4 720 001 |
/// | `segmentations` | 66 721 934 136 B |
/// | `raw_embeddings` | 14 499 843 072 B |
/// | tensors, total | 81 221 777 208 B — **75.64 GiB**, a 2 115x amplification |
/// | model calls | 9 440 002 |
/// | derived output frames | 35 557, against a cap of 4 194 304 — **0.85 % of it** |
///
/// and every guard that existed before this constant passed it: the output-frame
/// cap saw a ten-minute clip, and the placement scan found no rounding tie
/// because `2` is even. Those figures are not arithmetic in a doc comment —
/// `the_chunk_axis_cap_refuses_before_the_allocation_it_bounds` MEASURES the
/// 81 221 777 208 through the counting global allocator.
///
/// # The number, derived (round 11)
///
/// `MAX_EXTRACTION_CHUNKS * (594 * SEG_NUM_SLOTS * size_of::<f64>() +
/// SEG_NUM_SLOTS * EMBEDDING_DIM * size_of::<f32>())`, i.e. `70_770 * 17_328`,
/// where `594` is the crate's OWN one-chunk output grid —
/// `derived_output_frame_count(1, chunk_sliding_window(default),
/// frame_sliding_window())`, the number of
/// [`crate::audio::speaker::window::FRAME_STEP_S`] slots a single
/// `CHUNK_DURATION_S` chunk occupies.
///
/// Both factors are geometry this crate already commits to, and neither is a
/// property of any particular model:
///
/// - [`MAX_EXTRACTION_CHUNKS`] is the largest chunk grid a producer will run at
///   all, itself the first count [`MAX_OUTPUT_FRAMES`] refuses at the shipped
///   stride.
/// - `594` is the largest per-chunk frame count the aggregation can ADDRESS. A
///   chunk's frames are placed consecutively from
///   `window::aggregate_chunk_start_frame`, so frame `f` of the first chunk
///   lands on output frame `f`; past the one-chunk grid the
///   aggregation's own bounds test drops the frame
///   (`window::try_aggregate_output_frame_count`, `ofr >= num_output_frames =>
///   continue`). A segmenter emitting more frames than that per ten-second chunk
///   is emitting frames this crate's fixed output grid has nowhere to put.
///
/// `max_extraction_tensor_bytes_is_the_addressable_grids_own_footprint_derived_not_copied`
/// derives both factors through the crate's own functions rather than restating
/// them, the shape `plda_min_norm_is_diarics_own_floor_measured_not_copied` uses
/// for `diaric`'s floor.
///
/// # Why NOT `70_770 * 17_208`, round 10's number
///
/// `17_208` is one chunk's cost on community-1's 589-frame grid, so that ceiling
/// was the footprint of "the geometry the frame cap already refuses" AT 589 AND
/// NOWHERE ELSE. On a 590-frame grid a chunk costs `17_232`, the same ceiling
/// divides into 70 671 chunks, and the frame cap at the default stride still
/// admits 70 769 — so 1 130 880 001 samples (70 672 chunks, 1 217 819 904 bytes,
/// an output grid of 4 188 505 against a cap of 4 194 304) were refused by a
/// bound whose whole justification was that it refused nothing new. Acceptance
/// depended on an assumed frame count.
///
/// Deriving the ceiling at the ADDRESSABLE grid instead makes that justification
/// true for every frame count rather than one: `70_769 * 17_328 = 1 226 285 232`
/// is under this ceiling, so for ANY `num_frames_per_chunk` the output grid can
/// address, the largest chunk grid the frame cap admits at the default stride
/// fits (`the_byte_ceiling_admits_every_frame_grid_the_frame_cap_admits` sweeps
/// all 594 of them). The shipped 589-frame path is unchanged: its chunk
/// allowance was 70 770 under round 10's ceiling and is 70 770 under
/// [`MAX_EXTRACTION_CHUNKS`] now.
///
/// # Why the byte count, and not `num_chunks` or the model-call count
///
/// `num_chunks` does not stand for memory: the memory it implies is `num_chunks
/// * num_frames_per_chunk * ...`, and `num_frames_per_chunk` comes from the
/// loaded segmenter, so a chunk cap alone bounds bytes only for the frame count
/// it was calibrated against — a segmenter declaring 51 095 813 frames per chunk
/// reaches 1 226 302 584 bytes in a SINGLE chunk. So what is bounded here is the
/// byte count itself — the quantity that actually exhausts memory — with every
/// product `checked_mul`ed (`derived_extraction_tensor_bytes`).
///
/// The converse is round 11's finding and is why [`MAX_EXTRACTION_CHUNKS`]
/// exists beside this: the byte count does not stand for compute either. At the
/// loader's own floor of one frame per chunk a chunk costs 3 096 bytes, so a
/// byte ceiling near 1.2 GiB divides into ~393 000 chunks and ~786 000 model
/// calls. The two bounds are independent because the two quantities are: this
/// one is the only one that can see the frame axis, and that one is the only one
/// that can see a cheap chunk.
///
/// # Why not a `step_samples` floor instead
///
/// The obvious upstream cure is to refuse `step_samples = 2` as a configuration
/// rather than to bound what it costs, and there IS a principled threshold to
/// refuse it at. [`crate::audio::speaker::window::FRAME_STEP_S`] times
/// [`crate::audio::speaker::window::SAMPLE_RATE_HZ`] is exactly 270 samples, and
/// `window::aggregate_chunk_start_frame` places chunk `c` at
/// `round_ties_even(c * step_samples / 270)` — so below a 270-sample stride
/// consecutive chunks land on the SAME output-frame index, and the chunk grid is
/// finer than the grid its own output can address. The shipped
/// [`crate::audio::speaker::window::DEFAULT_STEP_SAMPLES`] is 16 000, 59.26 frame
/// steps, and [`crate::audio::speaker::source::argmax::ARGMAX_WINDOW_STRIDE_SAMPLES`]
/// — compiled into that graph, not a choice — is the same 16 000.
///
/// A floor was still not adopted, for three reasons, and
/// `a_step_samples_floor_at_one_frame_step_would_not_have_bounded_the_tensors`
/// pins the decisive one:
///
/// - It would not close the hole. At a stride of exactly 270,
///   [`MAX_OUTPUT_FRAMES`] still admits 4 193 711 chunks — 72 165 378 888 bytes,
///   67.21 GiB — from a 19.66 h clip. A floor caps the AMPLIFICATION (2 115x at
///   `step_samples = 2`, down to 15.9x) without capping the total, so it is a
///   third proxy on a branch already bitten twice by bounding proxies.
/// - It is not implied by anything upstream. dia guards `step_samples` at `0`
///   and at `> SEG_CHUNK_SAMPLES` and nowhere else (`owned.rs:374-387`), which
///   [`crate::audio::speaker::window::WindowOptions`] mirrors exactly; a floor
///   would be a refusal this crate invents, and a sub-frame stride still
///   produces a well-defined `Extraction` — its extra chunks are averaged into
///   the same output frame rather than discarded.
/// - It would refuse geometries the existing suite exercises:
///   `every_shipping_extract_geometry_places_its_chunks_identically` sweeps even
///   strides from `2` upward through
///   [`crate::audio::speaker::window::WindowOptions::with_step_samples`].
///
/// # NOT part of the fourteen-check assembly sequence
///
/// Deliberately, and this is a real asymmetry between the producers and
/// [`Extraction::try_from_parts`]. That constructor is HANDED the tensors, so
/// its check 3 requires `raw_embeddings.len()` to equal `num_chunks *
/// SEG_NUM_SLOTS * EMBEDDING_DIM` and `segmentations.len()` to equal
/// `num_chunks * num_frames_per_chunk * SEG_NUM_SLOTS`, exactly: a caller
/// declaring 4 720 001 chunks must already hold the 75.64 GiB of `Vec` to get
/// past it, and both products are themselves `checked_mul`ed, so an overflowing
/// `num_chunks` raises [`ExtractError::ExtractionGeometryOverflow`] before any
/// comparison and without allocating. The chunk axis is bounded there by the
/// caller's own inputs; adding this cap would only refuse memory already
/// spent.
pub const MAX_EXTRACTION_TENSOR_BYTES: usize = 1_226_302_560;

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
  #[cfg_attr(feature = "serde", serde(default = "default_segmenter_compute"))]
  segmenter: crate::ComputeUnits,
  #[cfg_attr(feature = "serde", serde(default = "default_embedder_compute"))]
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

  /// The chunk count, both timing grids, and every guard that reads nothing
  /// else — checks 4 and 6 (the output-frame grid and the resource bound it is
  /// held to), then the CHUNK-axis resource bound, then check 8 (chunk
  /// placement).
  ///
  /// This is the WHOLE of what [`Self::extract`] decides before it allocates a
  /// tensor or touches a model, and it is a separate function so that a test
  /// can run it. `extract` itself cannot be handed the inputs these guards
  /// exist for: the smallest clip the frame cap refuses is 1 132 448 001
  /// samples — 4.5 GB of `f32` and 70 770 pairs of CoreML calls — so a
  /// falsifier for its PLACEMENT has nowhere else to attach. The chunk-axis
  /// bound is reachable from a 923 KB clip, but never in fewer than 141 542
  /// model calls — and from the ten-minute clip its own falsifier uses, 9 440 002
  /// of them — so that falsifier attaches here too.
  ///
  /// `num_frames_per_chunk` is `seg.num_frames()`, which `extract` has already
  /// agreed with the embedder (check 5) before calling this. It is a parameter
  /// rather than a re-read because the byte bound needs it and this method must
  /// stay ahead of every model call: the frame count is the LAST input the
  /// tensor footprint needs, and `extract` has it before it allocates anything.
  ///
  /// The starts vector is deliberately not built here (`chunk_starts`,
  /// `owned.rs:447-451`): nothing in this function reads a chunk's start, and
  /// at a cap-tripping clip those `8 * num_chunks` bytes would be the first of
  /// the allocations the cap exists to prevent. `window::num_chunks` is the
  /// function `chunk_starts` itself counts with, so the two cannot disagree.
  ///
  /// Both grids are derived here rather than at `owned.rs:653-657`, after the
  /// chunk loop, so the guards can run ahead of inference. Nothing between here
  /// and step 9-11 reads them, so the move is a hoist only.
  ///
  /// # Errors
  /// - [`ExtractError::OutputFrameCountOverflow`] / [`ExtractError::OutputFrameCountTooLarge`]
  ///   — checks 4 and 6, through `checked_output_frame_count` (round 9).
  ///   Duplicated for the same reason the placement guard is: `assemble_checked`
  ///   runs both again at the end and THAT run is the guarantee, but the derived
  ///   grid reads nothing but `num_chunks` and the two windows — so a clip past
  ///   [`MAX_OUTPUT_FRAMES`] (19.6 h) is refused here instead of after
  ///   1 217 810 160 bytes of tensors, 70 770 pairs of CoreML calls and
  ///   404 771 544 bytes of count scratch have been spent reaching the same
  ///   verdict.
  /// - [`ExtractError::ExtractionChunkCountTooLarge`] — the CHUNK-axis COMPUTE
  ///   bound, through `checked_extraction_chunk_count` (round 11). The one that
  ///   holds the model-call count whatever the loaded segmenter declares: at the
  ///   loader's own floor of one frame per chunk a byte ceiling near 1.2 GiB
  ///   divides into 393 349 chunks, i.e. 786 698 model calls for 59.17 s of
  ///   audio.
  /// - [`ExtractError::ExtractionGeometryOverflow`] /
  ///   [`ExtractError::ExtractionTensorBytesTooLarge`] — the CHUNK-axis MEMORY
  ///   bound, through `checked_extraction_tensor_bytes` (round 10, re-derived in
  ///   round 11). NOT duplicated at assembly, and that asymmetry is deliberate:
  ///   see [`MAX_EXTRACTION_TENSOR_BYTES`] for why
  ///   [`Extraction::try_from_parts`]'s check 3 already bounds this axis by the
  ///   caller's own allocation. Both are ordered after the frame cap so a
  ///   geometry tripping several still names
  ///   [`ExtractError::OutputFrameCountTooLarge`], unchanged from round 9.
  /// - [`ExtractError::MisalignedChunkPlacement`] — check 8, round 3's cure.
  ///   Ordered AFTER both caps, matching the shared sequence's own order (check
  ///   6 before check 8) so this method and [`Extraction::try_from_parts`] name
  ///   the same error for a geometry that trips both — and, independently,
  ///   because the caps are `O(1)` where this scan is `O(num_chunks)`, on the
  ///   very axis the byte bound has just limited.
  /// - [`ExtractError::UncoveredLastChunk`] — check 14. The COVERAGE bound, and
  ///   the only guard here that the LOADED SEGMENTER's declared frame count can
  ///   trip: `SegmentModel` constrains it to `shape[1] >= 1` and nothing else,
  ///   while the output grid is fixed by the two windows, so a 595-frame model
  ///   on a one-chunk clip needs 595 frames of a 594-frame grid.
  ///   `diaric::reconstruct` refuses that on BOTH routes — but only after the
  ///   tensors, both models and (offline) AHC and VBx. Ordered AFTER the
  ///   placement scan, matching the shared sequence's own order (check 8 before
  ///   check 14) so a geometry tripping both names the same error here and at
  ///   [`Extraction::try_from_parts`].
  ///
  /// # Panics
  /// Panics if `self.options.window().step_samples()` is `0`, as
  /// [`crate::audio::speaker::window::chunk_starts`] does and for the same
  /// reason. [`Self::extract`] refuses that as [`ExtractError::ZeroStepSamples`]
  /// before it reaches here.
  fn checked_geometry(
    &self,
    samples_len: usize,
    num_frames_per_chunk: usize,
  ) -> Result<(usize, SlidingWindow, SlidingWindow), ExtractError> {
    let w = self.options.window();
    let num_chunks = crate::audio::speaker::window::num_chunks(samples_len, &w);
    let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&w); // owned.rs:653-655
    let frames_sw = crate::audio::speaker::window::frame_sliding_window(); // owned.rs:656-657

    let derived_output_frames = checked_output_frame_count(num_chunks, chunks_sw, frames_sw)?;
    // The DURATION axis is now bounded; these two are the CHUNK axis, which
    // `step_samples` scales independently of it and which sizes both tensors,
    // the starts vector, and the model-call count. COMPUTE first — it reads
    // `num_chunks` alone, so it holds whatever the loaded segmenter declares —
    // then MEMORY, the only one of the three that can see the frame axis. Both
    // ahead of the placement scan below, which is itself `O(num_chunks)`.
    checked_extraction_chunk_count(num_chunks)?;
    checked_extraction_tensor_bytes(num_chunks, num_frames_per_chunk)?;

    // No dia analog, and the guard `Extraction::try_from_parts` applies as its
    // check 8: the `count` built at step 9-11 is written on the AGGREGATION's
    // frame grid, while `diaric::reconstruct` — which both cluster backends
    // feed — places the same chunk's activations by `closest_frame`. Where the
    // two disagree the count marks frames the activations never reach and
    // suppresses the ones they do, and `diarize_online` re-derives its own
    // count through the same aggregation, so no choice of `count` repairs it.
    // See `window::first_misaligned_chunk` for which geometries are affected —
    // none of them reachable with the default `step_samples`.
    if let Some(m) =
      crate::audio::speaker::window::first_misaligned_chunk(num_chunks, chunks_sw, frames_sw)
    {
      return Err(ExtractError::MisalignedChunkPlacement(m));
    }

    // Check 14, and the one bound on this method's `num_frames_per_chunk`
    // argument that the loaded segmenter can actually trip: `SegmentModel`
    // constrains its declared frame count only to `shape[1] >= 1`, and a
    // 595-frame model on a one-chunk clip needs 595 output frames where the
    // geometry derives 594. Both backends refuse that extraction with
    // `OutputFrameCountTooSmall`, so running it here — after the placement
    // scan, matching the shared sequence's own order (check 8 before check
    // 14) so a geometry tripping both names the same error at both doors —
    // spends no inference on a clip nothing downstream could finish. The grid
    // it is compared against is the value `checked_output_frame_count` above
    // already returned, not a re-derivation of it.
    if let Some(u) = crate::audio::speaker::window::uncovered_last_chunk(
      num_chunks,
      num_frames_per_chunk,
      derived_output_frames,
      chunks_sw,
      frames_sw,
    ) {
      return Err(ExtractError::UncoveredLastChunk(u));
    }

    Ok((num_chunks, chunks_sw, frames_sw))
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
  ///   this crate cannot diarize honestly costs no model time. Since round 8
  ///   that early raise is a COST optimisation, not the guarantee — the same
  ///   check runs again at assembly, through the shared sequence below.
  /// - [`ExtractError::ExtractionChunkCountTooLarge`] if the chunk grid
  ///   `step_samples` and `samples.len()` derive is above
  ///   [`MAX_EXTRACTION_CHUNKS`], and
  ///   [`ExtractError::ExtractionTensorBytesTooLarge`] if that grid would need
  ///   more than [`MAX_EXTRACTION_TENSOR_BYTES`] of `segmentations` +
  ///   `raw_embeddings`, or [`ExtractError::ExtractionGeometryOverflow`] if that
  ///   footprint does not fit in `usize` at all. The axis
  ///   [`MAX_OUTPUT_FRAMES`] cannot see: the output grid scales with the clip's
  ///   DURATION while the tensors and the call count scale with `samples.len() /
  ///   step_samples`, so a ten-minute clip at `step_samples = 2` used to reach
  ///   the chunk loop holding 75.64 GiB and facing 9 440 002 model calls with
  ///   every existing guard satisfied. The two bounds are independent because
  ///   the two quantities are: `seg.num_frames()` scales the bytes but not the
  ///   calls, so at one frame per chunk the byte ceiling alone still admits
  ///   786 698 calls. Both raised BEFORE any inference and before either tensor,
  ///   from `num_chunks` and `seg.num_frames()` alone.
  /// - [`ExtractError::Infer`] (via `#[from]`) if either model's inference
  ///   fails (`owned.rs:477,600`).
  /// - [`ExtractError::OutputFrameCountOverflow`] if the derived
  ///   `num_output_frames` would not fit in `usize` (converted from
  ///   [`crate::audio::speaker::window`]'s `WindowError` by exhaustive match —
  ///   unreachable through `extract`'s own geometry, kept typed per this
  ///   crate's no-panic-on-untrusted-config posture; `owned.rs:663-673`).
  /// - Anything [`Extraction::try_from_parts`] raises. This method assembles
  ///   through `Extraction::assemble_checked`, which runs that constructor's
  ///   ENTIRE check sequence over the tensors this method just built — round
  ///   8's class fix: the unchecked assembly door is gone, so `extract` can no
  ///   longer emit an `Extraction` its own public constructor would refuse. All
  ///   of those refusals are unreachable through this method's own geometry
  ///   (see the matrix on `check_assembled_parts`) with ONE exception:
  ///   [`ExtractError::OutputFrameCountTooLarge`] is reachable, for a clip whose
  ///   derived grid exceeds [`MAX_OUTPUT_FRAMES`] — 19.6 hours at the
  ///   community-1 frame step, far past what either clustering backend can
  ///   finish. Since round 9 it is raised at step 7a instead: the grid depends
  ///   only on `samples.len()` and `step_samples`, so a late refusal cost 1.2 GB
  ///   of tensors, every model call, and 405 MB of count scratch to reach a
  ///   verdict the geometry had already fixed. Same error, same clips, only
  ///   sooner.
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
      return Err(ExtractError::StepSamplesExceedsWindow(
        StepSamplesExceedsWindow::new(w.step_samples(), SEG_CHUNK_SAMPLES),
      ));
    }
    if !crate::audio::speaker::window::check_onset(w.onset()) {
      return Err(ExtractError::OnsetOutOfRange(w.onset()));
    }

    // ── 5. Cross-model frame-count agreement (no dia analog) ──────────
    let num_frames = seg.num_frames();
    if num_frames != embed.num_mask_frames() {
      return Err(ExtractError::FrameCountMismatch(FrameCountMismatch::new(
        num_frames,
        embed.num_mask_frames(),
      )));
    }

    // ── 6-7b. Every guard that reads only geometry ────────────────────
    // `num_frames` is check 5's agreed frame count, and the last input the
    // tensor footprint needs — so the chunk-axis bound runs here, before the
    // two `vec![..]`s below size themselves from the very same products.
    let (num_chunks, chunks_sw, frames_sw) = self.checked_geometry(samples.len(), num_frames)?;

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
    // `start = c * step` (owned.rs:447-451), materialised here rather than at
    // step 6-7 so the geometry guards above allocate nothing.
    let starts = crate::audio::speaker::window::chunk_starts(samples.len(), &w);
    debug_assert_eq!(
      starts.len(),
      num_chunks,
      "`chunk_starts` counts with `window::num_chunks`, so the two must agree"
    );
    for (c, &start) in starts.iter().enumerate() {
      // a. Build the (possibly zero-padded) chunk window (owned.rs:469-475).
      fill_padded_chunk(&mut padded, samples, start);

      // b-d. Segment → multilabel → write this chunk's [f][s] slab
      // (owned.rs:477-498).
      let logits = seg.infer(&padded)?;
      let slab = crate::audio::speaker::segment::multilabel(&logits, num_frames);
      segmentations[chunk_segmentation_range(c, num_frames)].copy_from_slice(&slab);

      // e-g. Mask derivation, the batched embed call, and both drop paths —
      // through `embed_chunk_slots`, THE one implementation of the embed
      // stage, which `Self::extract_chunk_embeddings` also runs so the split
      // pipeline cannot drift from this loop (issue #127).
      embed_chunk_slots(
        embed,
        plda,
        &padded,
        &mut segmentations[chunk_segmentation_range(c, num_frames)],
        &mut raw_embeddings[chunk_embedding_range(c)],
        num_frames,
        onset,
      )?;
    }

    // ── 9-11. Count tensor + timing over the post-zeroing buffer ──────
    // `chunks_sw` / `frames_sw` were derived at step 6-7 so the placement
    // guard could run ahead of inference; they are the same two values
    // `owned.rs:653-657` builds here. `assemble_checked` derives `count` from
    // the post-zeroing buffer, runs the SAME fourteen checks
    // `Extraction::try_from_parts` runs, and assembles — this method no longer
    // has an unchecked door to reach for (round 8).
    Extraction::assemble_checked(
      raw_embeddings,
      segmentations,
      num_chunks,
      num_frames,
      w.onset(),
      chunks_sw,
      frames_sw,
    )
  }

  /// Runs the EMBED stage over ONE chunk — for a caller that owns the chunk
  /// loop itself.
  ///
  /// This is the open end of the `segmentation → embed → cluster` split
  /// [`ExtractionParts`] describes (issue #127). The CLUSTER end is
  /// [`Extraction::try_from_parts`]; the SEGMENTATION end is
  /// [`crate::audio::speaker::segment::SegmentModel::infer`] +
  /// [`crate::audio::speaker::segment::multilabel`] + the [`crate::audio::speaker::window`]
  /// module; this is the middle one, and it exists so a standalone embed node
  /// does not have to re-derive pyannote's `embedding_exclude_overlap` masking
  /// by hand. It runs steps (e)-(g) of [`Self::extract`]'s fused loop through
  /// the SAME private function that loop calls, so the two paths cannot
  /// disagree about a policy this crate parity-gates against dia.
  ///
  /// # What goes in
  ///
  /// - `embed` — the embedding model. NO [`crate::audio::speaker::segment::SegmentModel`]:
  ///   the point of the split is that the embed node never loads or re-runs a
  ///   segmenter, so this method takes the segmentation the caller already has.
  /// - `samples` — that chunk's audio window, EXACTLY [`SEG_CHUNK_SAMPLES`]
  ///   long: the identical buffer the caller handed
  ///   [`crate::audio::speaker::segment::SegmentModel::infer`], whose own
  ///   input-length contract is the same one and whose
  ///   [`crate::audio::speaker::error::InferError::InputLength`] this reuses.
  ///   A SHORTER slice is refused rather than padded here, because the padding
  ///   would be the wrong one: [`Self::extract`] zero-pads a final partial chunk
  ///   (dia's `owned.rs:469-475`) while
  ///   [`crate::audio::speaker::embed::EmbedModel::embed_chunk`] REPEAT-pads
  ///   whatever it is given, so accepting a short slice here would silently
  ///   embed different audio than the fused path does. The caller zero-pads its
  ///   own tail chunk — which it has already done, to call the segmenter.
  /// - `chunk_segmentations` — that chunk's `[f][s]` multilabel slab,
  ///   `embed.num_mask_frames() * SEG_NUM_SLOTS` values (a ONE-chunk
  ///   [`ExtractionParts::segmentations`]), MUTATED IN PLACE. Deriving the frame
  ///   count from the embedder rather than taking it as an argument is this
  ///   door's form of [`Self::extract`]'s check 5: with no segmenter present
  ///   there is no second declared frame count to compare, so the slab that
  ///   disagrees with the embedder's mask width is the one refused.
  ///
  /// Only [`Options::window`]'s `onset` is read — the compute and source
  /// options select models and a segmentation backend, neither of which this
  /// method touches. Configure the extractor exactly as for [`Self::extract`]
  /// and the two agree by construction.
  ///
  /// # What comes out
  ///
  /// The chunk's `[s][d]` raw-embedding block, `SEG_NUM_SLOTS * EMBEDDING_DIM`
  /// f32 — append it to the accumulating [`ExtractionParts::raw_embeddings`] —
  /// and, through `chunk_segmentations`, the POST-ZEROING slab to append to
  /// [`ExtractionParts::segmentations`]. Both drop paths are already applied:
  /// a slot with no active frame and a slot whose row cannot reach the
  /// clustering both leave an all-zero embedding row over an all-zero
  /// segmentation column, which is the pairing
  /// [`Extraction::try_from_parts`]' check 10 requires. Derive `count` from the
  /// accumulated (post-zeroing) `segmentations` with
  /// [`crate::audio::speaker::window::count_from_segmentations`], exactly as
  /// [`Self::extract`] does after ITS loop — never per chunk.
  ///
  /// # Errors
  /// - [`ExtractError::OnsetOutOfRange`] if the configured `onset` is not
  ///   finite in `(0.0, 1.0]` — [`Self::extract`]'s guard 4, for the same
  ///   serde-bypass reason, and raised before any inference.
  /// - [`ExtractError::Infer`] carrying
  ///   [`crate::audio::speaker::error::InferError::InputLength`] if
  ///   `samples.len() != SEG_CHUNK_SAMPLES`; or carrying whatever
  ///   [`crate::audio::speaker::embed::EmbedModel::embed_chunk`] raises
  ///   (`owned.rs:600`).
  /// - [`ExtractError::ExtractionLenMismatch`] naming
  ///   [`crate::audio::speaker::error::ExtractionPart::Segmentations`] if
  ///   `chunk_segmentations.len()` is not `embed.num_mask_frames() *
  ///   SEG_NUM_SLOTS`.
  /// - [`ExtractError::PldaTransformUnavailable`] if the shared transform the
  ///   per-row drop path validates against cannot be built — resolved BEFORE
  ///   the embedding call, as in [`Self::extract`].
  ///
  /// # Examples
  /// ```no_run
  /// # use coremlit::audio::speaker::{
  /// #   embed::{EMBEDDING_DIM, EmbedModel},
  /// #   extract::Extractor,
  /// #   segment::{SEG_CHUNK_SAMPLES, SEG_NUM_SLOTS, SegmentModel, multilabel},
  /// # };
  /// # fn one_chunk(
  /// #   seg: &SegmentModel,
  /// #   embed: &EmbedModel,
  /// #   padded: &[f32],
  /// # ) -> Result<(), Box<dyn std::error::Error>> {
  /// let extractor = Extractor::new();
  /// assert_eq!(padded.len(), SEG_CHUNK_SAMPLES);
  ///
  /// // The segmentation node's work; the embed node receives its output.
  /// let logits = seg.infer(padded)?;
  /// let mut slab = multilabel(&logits, seg.num_frames());
  ///
  /// // The embed node's work: no segmenter in sight, no masking policy here.
  /// let rows = extractor.extract_chunk_embeddings(embed, padded, &mut slab)?;
  /// assert_eq!(rows.len(), SEG_NUM_SLOTS * EMBEDDING_DIM);
  ///
  /// // `rows` and the now-zeroed `slab` are this chunk's contribution to the
  /// // cluster node's `ExtractionParts`.
  /// # Ok(())
  /// # }
  /// ```
  pub fn extract_chunk_embeddings(
    &self,
    embed: &EmbedModel,
    samples: &[f32],
    chunk_segmentations: &mut [f64],
  ) -> Result<Vec<f32>, ExtractError> {
    use crate::audio::speaker::error::{
      ExtractionLenMismatch, ExtractionPart, InferError, InputLength,
    };

    let w = self.options.window();
    if !crate::audio::speaker::window::check_onset(w.onset()) {
      return Err(ExtractError::OnsetOutOfRange(w.onset()));
    }
    if samples.len() != SEG_CHUNK_SAMPLES {
      return Err(ExtractError::Infer(InferError::InputLength(
        InputLength::new(samples.len(), SEG_CHUNK_SAMPLES),
      )));
    }

    let num_frames = embed.num_mask_frames();
    // `saturating_mul` rather than `*`: `num_frames` is a shape the loaded
    // model declares, so this cannot overflow from any real embedder — and a
    // saturated `usize::MAX` matches no slab a caller can allocate, which
    // refuses rather than panicking on a value that only a broken model could
    // produce.
    let expected = num_frames.saturating_mul(SEG_NUM_SLOTS);
    if chunk_segmentations.len() != expected {
      return Err(ExtractError::ExtractionLenMismatch(
        ExtractionLenMismatch::new(
          ExtractionPart::Segmentations,
          chunk_segmentations.len(),
          expected,
        ),
      ));
    }

    // Resolved before the embedding call for the reason `extract` resolves it
    // before its loop: an unavailable transform must refuse the call rather
    // than surface after the model has already run.
    let plda = shared_plda_transform()?;

    let mut chunk_embeddings = vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM];
    embed_chunk_slots(
      embed,
      plda,
      samples,
      chunk_segmentations,
      &mut chunk_embeddings,
      num_frames,
      f64::from(w.onset()),
    )?;
    Ok(chunk_embeddings)
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
/// The upstream half of that road is public too, and none of it re-implements
/// anything: the segmentation node runs
/// [`crate::audio::speaker::segment::SegmentModel::infer`] +
/// [`crate::audio::speaker::segment::multilabel`] on
/// [`crate::audio::speaker::window::chunk_starts`]' grid, and the embed node runs
/// [`Extractor::extract_chunk_embeddings`] per chunk, appending its answer to
/// `raw_embeddings` and its (post-zeroing) slab to `segmentations` — see the
/// module doc's "The split pipeline's three doors".
///
/// # Not parameters
/// - `num_speakers` is the fixed [`SEG_NUM_SLOTS`] (3) — the powerset
///   segmenter's slot count, not a caller choice
///   ([`Extraction::num_speakers`]).
/// - `num_output_frames` IS `count.len()`
///   (`diarization/src/offline/owned.rs:674`), derived by the constructor so the
///   two cannot disagree — the same property the module-private `from_parts`
///   has always had.
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
  /// the online one never reads ([`Extraction::try_from_parts`]'s check 12).
  /// Dropped `(chunk, slot)` rows are all-zero, which satisfies that. See
  /// [`Extraction::raw_embeddings`].
  pub raw_embeddings: Vec<f32>,
  /// Per-`(chunk, frame, speaker)` activity, flattened `[c][f][s]`. Must have
  /// length `num_chunks * num_frames_per_chunk * num_speakers`, and every cell
  /// must be exactly `0.0` or exactly `1.0` — a HARD multilabel decode, which is
  /// all either in-crate producer emits and the only domain on which the two
  /// backends read this tensor the same way
  /// ([`Extraction::try_from_parts`]'s check 9: offline sums these magnitudes,
  /// online counts nonzero frames). See [`Extraction::segmentations`].
  pub segmentations: Vec<f64>,
  /// Per-output-frame instantaneous speaker count, `[t]`. Its length becomes
  /// [`Extraction::num_output_frames`], and its VALUES must be exactly what
  /// `segmentations` derive through
  /// [`crate::audio::speaker::window::count_from_segmentations`] over
  /// `seg > 0.0` — [`Extraction::try_from_parts`]'s check 11, an equality in
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

/// Check 4's derivation: the output-frame grid that `num_chunks` and the two
/// timing windows imply. The value `count.len()` must equal (check 5), the one
/// [`MAX_OUTPUT_FRAMES`] bounds (check 6), and the one
/// [`Extraction::diarize_online`] re-derives on every call.
///
/// The ONE place that expression is written — `check_assembled_parts` runs it
/// as its check 4, and every producer runs it AHEAD of its own work through
/// [`checked_output_frame_count`]. Written out a second time it would be a
/// second expression that is algebraically equal and numerically different,
/// which is the class check 8 exists for.
///
/// `saturating_sub(1)` where this was written `- 1` inline: at
/// `check_assembled_parts`' call site check 1 has already refused `num_chunks
/// == 0`, and at every producer `window::num_chunks` returns at least `1`, so
/// the two forms agree everywhere this is reachable. At `num_chunks == 0` it
/// derives the one-chunk grid, which leaves `window::try_count_from_segmentations`'
/// own `assert!(num_chunks > 0)` to fire exactly as it did before this preflight
/// existed.
fn derived_output_frame_count(
  num_chunks: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Result<usize, ExtractError> {
  let last_chunk_end =
    chunks_sw.duration() + num_chunks.saturating_sub(1) as f64 * chunks_sw.step();
  crate::audio::speaker::window::try_num_output_frames(last_chunk_end, frames_sw.step()).map_err(
    |e| match e {
      crate::audio::speaker::window::WindowError::OutputFrameCountOverflow => {
        ExtractError::OutputFrameCountOverflow
      }
    },
  )
}

/// Check 6: the RESOURCE bound. The ONE comparison against
/// [`MAX_OUTPUT_FRAMES`], so the bound cannot drift between the assembly
/// sequence and the producers' preflight.
fn check_output_frame_cap(derived_output_frames: usize) -> Result<(), ExtractError> {
  if derived_output_frames > MAX_OUTPUT_FRAMES {
    return Err(ExtractError::OutputFrameCountTooLarge(
      derived_output_frames,
    ));
  }
  Ok(())
}

/// Checks 4 and 6 over GEOMETRY ALONE — the preflight every producer runs
/// before it allocates a tensor or calls a model, and that
/// `Extraction::assemble_checked` runs before it derives a `count`.
///
/// # Why a producer runs this at all (round 9)
///
/// Round 8 gave every producer the whole check sequence, but at the END: both
/// build their extraction tensors and complete inference, and
/// `assemble_checked` then derives `count` — so check 6, whose entire job is to
/// refuse a grid this crate is not willing to allocate for, ran after the
/// allocations. At the smallest clip it refuses (1 132 448 001 samples, 19.66 h
/// at 16 kHz, deriving 4 194 312 frames against a cap of 4 194 304) that meant
/// 1 217 810 160 bytes of `segmentations` + `raw_embeddings`, 70 770 pairs of
/// CoreML calls, and then 404 771 544 bytes of count scratch, all spent on an
/// input the geometry alone had already condemned.
///
/// Nothing about the grid depends on any of that work. Round 3 established the
/// same fact for check 8 — the output grid is a function of `samples.len()` and
/// `step_samples`, which is why the placement guard already runs pre-inference
/// — and checks 4 and 6 read strictly less: `num_chunks` and the two windows.
/// So the cap moves to where its inputs are known, and the late check stays
/// where it is, because that is the one [`Extraction::try_from_parts`] runs for
/// parts this crate did not compute.
///
/// Refuses NOTHING it did not refuse before. It is the same two functions over
/// the same three values a producer hands `assemble_checked` verbatim, so the
/// verdict is identical and only its position changed.
///
/// # Errors
/// [`crate::audio::speaker::error::ExtractError::OutputFrameCountOverflow`]
/// (check 4) or [`ExtractError::OutputFrameCountTooLarge`] (check 6).
pub(crate) fn checked_output_frame_count(
  num_chunks: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Result<usize, ExtractError> {
  let derived_output_frames = derived_output_frame_count(num_chunks, chunks_sw, frames_sw)?;
  check_output_frame_cap(derived_output_frames)?;
  Ok(derived_output_frames)
}

/// The exact byte footprint of the two tensors a producer is about to allocate
/// for `(num_chunks, num_frames_per_chunk)`: `segmentations`' `num_chunks *
/// num_frames_per_chunk * SEG_NUM_SLOTS` `f64`s plus `raw_embeddings`'
/// `num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM` `f32`s.
///
/// The ONE place that expression is written, for the reason
/// `derived_output_frame_count` is the one place ITS expression is written: both
/// producers size those two `vec![..]`s from these very products, so a second
/// spelling could be algebraically equal and numerically different.
///
/// Every product `checked_mul`ed, and `raw_embeddings` before `segmentations` —
/// `check_assembled_parts`' own check-3 order, so a geometry whose products
/// overflow names the same [`crate::audio::speaker::error::ExtractionPart`] in
/// this preflight as in the assembly sequence.
///
/// # Errors
/// [`ExtractError::ExtractionGeometryOverflow`] if either product, or their sum,
/// exceeds `usize`. That is the variant check 3 already raises for an
/// overflowing geometry, carrying the same `(part, num_chunks,
/// num_frames_per_chunk)` diagnosis; the only difference is that the products
/// here are in BYTES, so this arm can also fire for an element count that fits
/// `usize` while its byte size does not — a `Vec` of which is unallocatable
/// regardless (`Vec` caps capacity at `isize::MAX` bytes). The sum's own arm
/// names `Segmentations`, the dominant term.
fn derived_extraction_tensor_bytes(
  num_chunks: usize,
  num_frames_per_chunk: usize,
) -> Result<usize, ExtractError> {
  use crate::audio::speaker::error::{ExtractionGeometryOverflow, ExtractionPart};

  let overflow = |part| {
    ExtractError::ExtractionGeometryOverflow(ExtractionGeometryOverflow::new(
      part,
      num_chunks,
      num_frames_per_chunk,
    ))
  };
  let raw_embeddings = num_chunks
    .checked_mul(SEG_NUM_SLOTS)
    .and_then(|n| n.checked_mul(EMBEDDING_DIM))
    .and_then(|n| n.checked_mul(size_of::<f32>()))
    .ok_or_else(|| overflow(ExtractionPart::RawEmbeddings))?;
  let segmentations = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(SEG_NUM_SLOTS))
    .and_then(|n| n.checked_mul(size_of::<f64>()))
    .ok_or_else(|| overflow(ExtractionPart::Segmentations))?;
  raw_embeddings
    .checked_add(segmentations)
    .ok_or_else(|| overflow(ExtractionPart::Segmentations))
}

/// The COMPUTE bound: the preflight every producer runs on its chunk count
/// alone, before it allocates a tensor, scans a chunk placement, or calls a
/// model.
///
/// The ONE comparison against [`MAX_EXTRACTION_CHUNKS`], so the bound cannot
/// drift between the two producers — the same reason `check_output_frame_cap`
/// is one function.
///
/// Ordered AFTER [`checked_output_frame_count`] and BEFORE
/// [`checked_extraction_tensor_bytes`] at both call sites. After the frame cap,
/// so a geometry tripping both keeps naming
/// [`ExtractError::OutputFrameCountTooLarge`] exactly as it did before either
/// chunk-axis bound existed. Before the byte ceiling, because the two own
/// different regimes and this is the one that owns the stride axis: at every
/// per-chunk frame count the crate's output grid can address, this bound is the
/// one a small `step_samples` reaches, and a geometry that trips both is a
/// geometry with too many chunks rather than one with too large a model.
///
/// Reads `num_chunks` and nothing else — no model shape, no window — which is
/// precisely what makes it complementary to a byte ceiling the loaded
/// segmenter's frame count scales.
///
/// # Errors
/// [`ExtractError::ExtractionChunkCountTooLarge`] carrying the derived chunk
/// count.
pub(crate) fn checked_extraction_chunk_count(num_chunks: usize) -> Result<usize, ExtractError> {
  if num_chunks > MAX_EXTRACTION_CHUNKS {
    return Err(ExtractError::ExtractionChunkCountTooLarge(num_chunks));
  }
  Ok(num_chunks)
}

/// The MEMORY bound: the preflight every producer runs on geometry alone,
/// before it allocates a tensor, scans a chunk placement, or calls a model.
///
/// The ONE comparison against [`MAX_EXTRACTION_TENSOR_BYTES`], so the bound
/// cannot drift between the two producers — the same reason
/// `check_output_frame_cap` is one function.
///
/// Ordered AFTER [`checked_extraction_chunk_count`] at both call sites, itself
/// after [`checked_output_frame_count`]. This is the only one of the three that
/// reads `num_frames_per_chunk`, so it is the only one that can refuse a
/// segmenter whose declared grid is the thing that is too large — see
/// [`MAX_EXTRACTION_TENSOR_BYTES`] for the 51 095 813-frame single chunk that is
/// this bound's own smallest refusal.
///
/// # Errors
/// [`ExtractError::ExtractionGeometryOverflow`] from
/// `derived_extraction_tensor_bytes`, or
/// [`ExtractError::ExtractionTensorBytesTooLarge`] carrying the derived byte
/// total.
pub(crate) fn checked_extraction_tensor_bytes(
  num_chunks: usize,
  num_frames_per_chunk: usize,
) -> Result<usize, ExtractError> {
  let bytes = derived_extraction_tensor_bytes(num_chunks, num_frames_per_chunk)?;
  if bytes > MAX_EXTRACTION_TENSOR_BYTES {
    return Err(ExtractError::ExtractionTensorBytesTooLarge(bytes));
  }
  Ok(bytes)
}

/// The ONE implementation of [`Extraction::try_from_parts`]'s fourteen checks,
/// over BORROWED parts — run by that constructor and, through
/// `Extraction::assemble_checked`, by every in-crate
/// [`crate::audio::speaker::source::ModelSource`].
///
/// Every check, its number, its ordering rationale and the backend split it
/// closes is documented on [`Extraction::try_from_parts`]; this function is
/// that documentation's body, moved so there is exactly one of it.
///
/// # The check × producer matrix (round 8)
///
/// The round-8 finding is that a closure argument scoped to ONE construction
/// path is worse than none. There are three paths that assemble an
/// [`Extraction`] — [`Extractor::extract`], `ArgmaxSource::extract`, and
/// [`Extraction::try_from_parts`] — and before round 8 only the third ran this
/// sequence. The first ran a hand-copied check 8 (round 3's cure); the second
/// ran nothing, so a segmenter with the pinned F16 I/O shapes that returned
/// `0.1` per frame assembled an extraction whose stored `count` was all zero
/// (offline: silence) while the online route read every cell as active (a
/// 9.94 s speaker).
///
/// All three now run all fourteen, so the matrix has no holes and no
/// exemptions. What it still records is WHY each check cannot fire from the two
/// sources — a check that no producer can trip is a check whose cost is a
/// premium against a future producer, and that is a different statement from
/// "structurally impossible, therefore skipped":
///
/// | # | check | `Extractor::extract` | `ArgmaxSource::extract` |
/// |---|-------|----------------------|-------------------------|
/// | 1 | non-zero dims | `chunk_starts` returns >= 1 start; `SegmentModel`'s contract pins `shape[1] >= 1`; `count.len() = try_num_output_frames(..) >= 1` | same, with `num_frames_per_chunk` the `ARGMAX_FRAMES_PER_WINDOW` constant |
/// | 2 | usable windows | `chunk_sliding_window` is `(0.0, 10.0, step_samples/16000)` and `step_samples != 0` is guarded; `frame_sliding_window` is three constants | same, `step_samples` pinned to argmax's stride |
/// | 3 | length products | both buffers are `vec![_; num_chunks * .. ]` at those very dimensions | same |
/// | 4 | derived count fits `usize` | `try_num_output_frames` via the count derivation | same |
/// | 5 | `count.len()` == derived | `count` IS that derivation's output | same |
/// | 6 | derived <= `MAX_OUTPUT_FRAMES` | ENFORCED, and reachable: a clip past 19.6 h fails. Since round 9 it fails at `checked_output_frame_count`, before the tensors and the models — this run is the backstop | same |
/// | 7 | `frames_sw.step()` survives `f32` | `FRAME_STEP_S` is `0.016875` | same |
/// | 8 | both grids place every chunk alike | round 3's cure, still run pre-inference as well so a bad grid costs no model time | the stride is compiled into the graph and even, so no rounding tie exists (`the_fixed_argmax_grid_places_every_chunk_identically_under_both_mappings`) |
/// | 9 | segmentations hard-binary | `multilabel` writes `POWERSET_TABLE` literals and `zero_slot_column` writes `0.0` | **the round-8 hole**: `write_segmentations` copies `f64::from(speaker_ids[..])` verbatim, and `from_dir_with` accepts any model with the pinned F16 shapes |
/// | 10 | active slot's row reaches PLDA | the same `raw_embedding_reaches_plda`; a row that fails it has its column zeroed | same, in `place_embeddings` |
/// | 11 | `count` IS the derived count | `count` is the same aggregation under `seg >= onset`, which check 9 + `check_onset` make identical to `seg > 0.0` | same, and it was NOT identical while check 9 was open: at `0.1` with onset `0.5` the stored count was zero where `seg > 0.0` derives one |
/// | 12 | whole `raw_embeddings` finite | buffer pre-zeroed; a written row passed `from_wespeaker`'s own finiteness scan | same, plus `place_embeddings`' explicit per-row scan (`InferError::NonFiniteOutput`) |
/// | 13 | frame centers strictly increasing | `frame_sliding_window`'s fixed grid stays strictly increasing past `MAX_OUTPUT_FRAMES` | same |
/// | 14 | the grid reaches the last chunk's last frame | ENFORCED, and reachable: `SegmentModel` constrains its declared frame count only to `shape[1] >= 1`, so a 595-frame segmenter on a one-chunk clip needs 595 of 594 frames. Since this round it fails at `checked_geometry`, before the tensors and the models — this run is the backstop | the frame count is the compiled `ARGMAX_FRAMES_PER_WINDOW` (589), which leaves the last chunk >= 4 frames of headroom at every chunk count the caps admit (`the_pinned_argmax_grid_covers_its_last_chunk_at_every_chunk_count`) |
///
/// Row 11 is the one worth reading twice: it is not independently sound at
/// `ArgmaxSource`. It holds only BECAUSE check 9 holds, which is what makes
/// "each producer enforces the checks it can reach" the wrong shape of
/// argument and one shared sequence the right one.
///
/// Two bounds are deliberately absent from this table:
/// [`MAX_EXTRACTION_TENSOR_BYTES`] and [`MAX_EXTRACTION_CHUNKS`], the CHUNK-axis
/// resource bounds, are enforced in each producer's `checked_geometry` and NOT
/// here. Check 3 already holds this axis for caller-supplied parts — the two
/// length equalities are exact, so a declared `num_chunks` is bounded by the
/// buffers the caller had to allocate to reach them, and the products are
/// `checked_mul`ed so an unrepresentable one raises before any comparison — and
/// this constructor calls no model, so there is no compute for the second to
/// hold. See both constants' own docs.
#[allow(clippy::too_many_arguments)]
fn check_assembled_parts(
  raw_embeddings: &[f32],
  segmentations: &[f64],
  count: &[u8],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
) -> Result<(), ExtractError> {
  use crate::audio::speaker::error::{
    ExtractionGeometryOverflow, ExtractionLenMismatch, ExtractionPart, InvalidSlidingWindow,
  };

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
  // `num_chunks >= 1` (check 1) makes `derived_output_frame_count`'s
  // `saturating_sub(1)` an exact `- 1`; the windows are finite and positive
  // (check 2), so `last_chunk_end` is the only quantity left that can drive
  // the division out of range.
  //
  // Through the shared `derived_output_frame_count` — the same function every
  // producer's pre-inference preflight calls, so the value refused there and
  // the value validated here cannot drift.
  let derived_output_frames = derived_output_frame_count(num_chunks, chunks_sw, frames_sw)?;

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
  //
  // Through the shared `check_output_frame_cap`, so this and every producer's
  // preflight compare against the one constant. Reached here only for parts a
  // caller assembled: both in-crate producers have already run this exact
  // comparison, over this exact derivation, before they allocated a tensor —
  // see `checked_output_frame_count`. Kept regardless, because
  // `try_from_parts` has no earlier point to run it at.
  check_output_frame_cap(derived_output_frames)?;

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
  // The `count` this constructor validates (check 11) is written on the
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

  // ── 9. EVERY segmentations cell is exactly `0.0` or exactly `1.0` ──
  // The DOMAIN check, and it comes first among the three that read this
  // tensor because the other two only describe both backends INSIDE it.
  //
  // Checks 10 and 11 booleanize at `seg > 0.0`, and so does the whole online
  // route — its per-slot `activity` frame count and its distinct-cluster
  // count. `diaric`'s OFFLINE route sums the MAGNITUDES instead:
  // `filter_embeddings` accumulates `clean_frames += segmentations[..]` over
  // singly-active frames against `0.2 * num_frames_per_chunk`
  // (`diarization/src/offline/algo.rs:644-679`), and stage 7's inactive mask
  // sums the whole column and tests `sum_activity == 0.0`
  // (`diarization/src/pipeline/algo.rs:698-711`). On `{0.0, 1.0}` a magnitude
  // sum IS the active-frame count and `sum == 0.0` IS "no active frame", so
  // every reading collapses onto one boolean; off it they are different
  // functions of the same buffer, and the difference is a SPEAKER COUNT (see
  // `ExtractError::NonBinarySegmentation`, and
  // `a_fractional_segmentation_splits_the_two_backends`).
  //
  // Confining the input rather than teaching the check to model offline's sum:
  // there is nothing to model FOR. Neither in-crate producer can emit a
  // fractional cell — `Extractor::extract` writes `segment::multilabel`'s
  // powerset table (literal `0.0`/`1.0`) and zeroed columns, `ArgmaxSource`
  // writes the graph's hard-binary `speaker_ids` and zeroed columns — so soft
  // segmentation support was a capability with no producer and two
  // incompatible consumers.
  //
  // The equality also refuses NaN and ±inf, which the old "Finiteness of
  // `segmentations`" omission left to the backends. That omission's reasoning
  // still holds and is why this is a by-product rather than a second finding:
  // a non-finite cell is NOT a split. Every path ends in
  // `diaric::reconstruct`, which scans the whole tensor and raises
  // `NonFiniteField::Segmentations`
  // (`diarization/src/reconstruct/algo.rs:497-508`) — ONLINE hands it
  // `self.segmentations` directly, OFFLINE the same slice at its stage 5
  // (`diarization/src/offline/algo.rs:808`), and OFFLINE meets
  // `assign_embeddings`' own copy of that scan first
  // (`diarization/src/pipeline/algo.rs:456-460`). So the two refuse with
  // different typed variants — `Pipeline(NonFinite(Segmentations))` offline,
  // `Reconstruct(NonFinite(Segmentations))` online — and neither returns `Ok`.
  // What changes is only WHERE: the cell is now named at assembly instead of
  // after a backend was chosen.
  //
  // Stated over the WHOLE buffer, for check 12's reason: the property is "no
  // cell of this tensor is outside the domain", and a flat scan cannot drift
  // from the `[c][f][s]` indexing the way a hand-rolled walk can.
  //
  // Ordered AHEAD of the other two readers of this tensor. Against check 11
  // that is load-bearing, not a preference: on a fractional buffer that
  // check's derived count is computed under a predicate that does not
  // describe offline at all, so `CountNotSegmentationDerived` would name the
  // `count` for a defect that is in the `segmentations`. Against check 10 it
  // is cost — this scan is a twentieth of that one — and the same diagnostic
  // point, since `ActiveSlotWithoutEmbedding` reports a slot whose activity
  // was decided by the very predicate under question. Ordered after check 3,
  // which bounds the length this walks.
  //
  // Cost, on the same 10-minute extraction the other two scans are quoted
  // against (591 chunks x 589 frames x 3 slots = 8.0 MiB of `f64`, release,
  // this host): ~0.70 ms. That is ~3.6x check 12's finiteness scan
  // (~0.19 ms) — the buffer-size ratio and nothing more — and ~4% of check
  // 10's row chain (~16.6-20.9 ms for all 1 773 rows), so it lands between
  // the two and next to the cheap one.
  if let Some((i, &v)) = segmentations
    .iter()
    .enumerate()
    .find(|(_, v)| **v != 0.0 && **v != 1.0)
  {
    return Err(ExtractError::NonBinarySegmentation(
      crate::audio::speaker::error::NonBinarySegmentation::new(i, v),
    ));
  }

  // ── 10. An active slot must carry an embedding BOTH engines can use ──
  // The activity predicate is `seg > 0.0` — the same "any nonzero entry is
  // binary-active" rule `diarize_online` applies and dia's
  // `filter_embeddings` uses (`diarization/src/offline/algo.rs:656-660`).
  // Past check 9 that is exactly `seg == 1.0`.
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

  // ── 11. `count` must BE the count these segmentations derive ──────
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
  // derived grid past check 6. Check 12 follows it only because that one is
  // allocation-free and deliberately yields to check 10's diagnosis; nothing
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

  // ── 12. EVERY raw_embeddings value is finite, active or not ───────
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
  // Ordered AFTER check 10 on purpose, and it costs nothing to do so. Ahead of
  // it, this blanket scan would swallow every ACTIVE slot's non-finite row —
  // the round-1 falsifier's NaN included — and report a bare buffer offset
  // where `ActiveSlotWithoutEmbedding` names the `(chunk, slot)` whose column
  // claims speech, the more specific diagnosis and the more actionable one.
  // The scan is ~1% of check 10 on a realistic extraction (10 minutes of audio
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

  // ── 13. Every output frame must have its OWN center time ──────────
  // Check 2 asks whether `frames_sw`'s three fields are usable numbers; this
  // asks whether the GRID they generate is a usable timeline, which is a
  // different question and the one the spans are built from. Every span
  // endpoint either backend emits is a frame CENTER — `frames_sw.start + t *
  // frames_sw.step + frames_sw.duration / 2`, evaluated by `diaric`'s
  // `try_discrete_to_spans` (`diarization/src/reconstruct/rttm.rs:172,
  // 216-217,231-232`) — and that sum rounds. At `frames_sw.start = 1e9` the
  // `f64` ULP is `1.1920928955078125e-7`, so a `step` of `1e-8` adds nothing
  // (`1e9 + 1e-8 == 1e9`) and frames 0 and 1 land on one center. A one-frame
  // active run then closes at `start == end` and the backend returns `Ok`
  // with a span of DURATION ZERO. Both backends, identically — this is not a
  // split but a silently meaningless answer, which is the other failure this
  // constructor exists to refuse.
  //
  // `window::first_collapsed_frame_center` is the ONE definition, shared with
  // every producer through `Self::assemble_checked`, and it MIRRORS the span
  // conversion's own arithmetic rather than an algebraically equal
  // rearrangement — `start + (t * step + duration / 2)` is a different `f64`
  // from `(start + t * step) + duration / 2`, and it is exactly that kind of
  // re-association the check-8 class is made of.
  //
  // Ordered LAST, deliberately. It is pure geometry over a range check 6 has
  // already bounded, so its position costs at most one extra pass; putting it
  // anywhere earlier would renumber the sequence and move the precedence of
  // every check after it, and each of those precedences is pinned by a
  // falsifier that says why it is where it is.
  //
  // Cost, on the same 10-minute extraction the other scans are quoted against
  // (591 chunks, 35 557 output frames, release, this host): ~24 µs. That is
  // ~13% of check 12's finiteness scan (~0.18 ms), ~3% of check 9's domain scan
  // (~0.69 ms) and ~0.1% of check 10's row chain (~16.6 ms) — the output grid
  // is thirty times smaller than the segmentation buffer, so this is the
  // cheapest O(n) check in the sequence. (Only check 8's O(num_chunks)
  // placement scan, ~0.6 µs, is cheaper, and it walks 591 values to this one's
  // 35 557.)
  if let Some(c) =
    crate::audio::speaker::window::first_collapsed_frame_center(count.len(), frames_sw)
  {
    return Err(ExtractError::CollapsedFrameCenter(c));
  }

  // ── 14. The grid must REACH the last chunk's last frame ───────────
  // Checks 5 and 8 settle which grid this is and that both mappings agree
  // about where each chunk starts on it. Neither says the grid is LONG
  // ENOUGH. The derived length is `round(last_chunk_end / frames_sw.step()) +
  // 1`, and the requirement is `reconstruct_chunk_start_frame(last) +
  // num_frames_per_chunk` — different functions of the same windows, so a
  // `chunks_sw.duration()` spanning fewer frame-steps than
  // `num_frames_per_chunk` derives a grid the last chunk overruns. On the
  // shipped `(0.0, 10.0, 1.0)` / `(0.0, 0.0619375, 0.016875)` pair a
  // 594-frame segmenter over THREE chunks derives 712 frames while the last
  // chunk sits at 119 and needs 713.
  //
  // `diaric::reconstruct` skips every `out_f >= num_output_frames`, so those
  // cells are simply not written; it refuses the geometry rather than
  // truncating (`ShapeError::OutputFrameCountTooSmall`,
  // `diarization/src/reconstruct/algo.rs:478-495`) and BOTH routes reach it —
  // ONLINE directly, OFFLINE at stage 5 after AHC and VBx
  // (`diarization/src/offline/algo.rs:808`). So this refuses nothing new; it
  // moves a typed, late, post-clustering failure to assembly, which is this
  // constructor's whole contract.
  //
  // `window::uncovered_last_chunk` is the ONE definition, and it calls the
  // SAME `reconstruct_chunk_start_frame` check 8 runs for every chunk. An
  // earlier revision of the "deliberately NOT checked" list declined this
  // check because re-deriving the placement would be a second copy of
  // `closest_frame`'s float arithmetic; round 8 introduced that mirror and
  // anchored it, so the derivation is now a call rather than a copy.
  //
  // Ordered LAST, for check 13's reason and one more. It is `O(1)` geometry
  // over values check 4 has already derived, so its position costs nothing
  // measurable; putting it earlier would move the precedence of every check
  // after it, and each of those is pinned by a falsifier that says why it is
  // where it is. Both producers run it in their `checked_geometry` instead,
  // where it lands before any inference — this run is the backstop, and the
  // only run for parts a caller assembled.
  if let Some(u) = crate::audio::speaker::window::uncovered_last_chunk(
    num_chunks,
    num_frames_per_chunk,
    derived_output_frames,
    chunks_sw,
    frames_sw,
  ) {
    return Err(ExtractError::UncoveredLastChunk(u));
  }

  Ok(())
}

impl Extraction {
  /// The single ASSEMBLY site for an [`Extraction`] — every construction path
  /// in this crate lands here, and this is the only place the struct's fields
  /// are written.
  ///
  /// UNCHECKED, and PRIVATE TO THIS MODULE, which is the round-8 change. It
  /// used to be `pub(crate)`, and each [`crate::audio::speaker::source::ModelSource`]
  /// called it directly having satisfied — by its own local reasoning — the
  /// invariants [`Self::try_from_parts`] enforces. That reasoning is exactly
  /// what does not survive contact with a model: `ArgmaxSource` copied the
  /// segmenter's decoded IDs verbatim, so a graph that returned `0.1` per frame
  /// assembled an extraction whose stored `count` was zero (offline: silence)
  /// while the online route read every cell as active (a 9.9 s speaker), and
  /// nothing refused it. A comment saying "in-crate callers are self-consistent
  /// by construction" is not a mechanism; module privacy is.
  ///
  /// So the two doors are now:
  ///
  /// - `Self::assemble_checked` — every in-crate producer, which runs the
  ///   shared `check_assembled_parts` and then lands here.
  /// - [`Self::try_from_parts`] — every out-of-crate caller, which runs the
  ///   same shared check plus the two that read the caller's own `count`, and
  ///   then lands here.
  ///
  /// Nothing outside `extract` can reach this function at all (tests excepted,
  /// through [`Self::from_parts_unchecked`], which exists so a falsifier can
  /// still build the very witness a producer must not).
  ///
  /// `num_output_frames` is not a parameter: it IS `count.len()`
  /// (`owned.rs:674`), so deriving it here — at the one site every path reaches
  /// — makes the two impossible to disagree.
  fn from_parts(
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

  /// TEST-ONLY reach-through to the module-private [`Self::from_parts`].
  ///
  /// A falsifier for a constructor refusal has to exhibit the refused input
  /// reaching a backend, which means assembling one UNCHECKED — and several of
  /// those falsifiers live outside this module (`source::argmax::tests`). This
  /// is the one seam they get, `#[cfg(test)]` so no production path can take
  /// it, and named so that using it is a statement rather than an accident.
  #[cfg(test)]
  pub(crate) fn from_parts_unchecked(
    raw_embeddings: Vec<f32>,
    segmentations: Vec<f64>,
    count: Vec<u8>,
    num_chunks: usize,
    num_frames_per_chunk: usize,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
  ) -> Self {
    Self::from_parts(
      raw_embeddings,
      segmentations,
      count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    )
  }

  /// The door every in-crate [`crate::audio::speaker::source::ModelSource`]
  /// assembles through: derive `count` from the segmentations the source just
  /// wrote, run the shared invariant check, and assemble.
  ///
  /// # Why this exists (round 8)
  ///
  /// Round 3 closed a hole where [`Extractor::extract`] emitted a chunk grid
  /// its own public constructor refuses, by copying ONE of `try_from_parts`'s
  /// checks into that method. Round 8's finding is that the same hole was left
  /// open for every OTHER check at every OTHER producer — `ArgmaxSource` copies
  /// its segmenter's decoded IDs verbatim, so a model with the pinned F16 I/O
  /// shapes that returns `0.1` per frame produced an extraction the two
  /// backends read in opposite directions. The cure is not a third copy of a
  /// check: it is a single door with a single check behind it, so a check added
  /// to `try_from_parts` is a check every producer gains, and a new producer
  /// cannot be written that skips them.
  ///
  /// # What it enforces
  ///
  /// ALL FOURTEEN of [`Self::try_from_parts`]'s checks, in that constructor's
  /// own order, through the one `check_assembled_parts` both call. No
  /// exemptions: a producer is held to exactly the standard an out-of-crate
  /// caller is, which is the only version of this fix whose closure argument
  /// does not have to enumerate which cells it skips.
  ///
  /// Two of the fourteen are provably redundant here and are run anyway.
  /// Check 5 (`count.len()` equals the derived grid) compares
  /// `try_count_from_segmentations`'s output length with the
  /// `try_num_output_frames` call that produced it. Check 11 (`count` EQUALS
  /// what the segmentations derive under `seg > 0.0`) re-runs the overlap-add
  /// under `seg > 0.0` where `count` was built under `seg >= onset`; check 9
  /// confines every cell to `{0.0, 1.0}` and `check_onset` confines `onset` to
  /// `(0.0, 1.0]`, so the two predicates select the same cells. The price is
  /// one extra `num_chunks * num_frames_per_chunk` buffer and a second pass of
  /// the aggregation — measured at ~1.4 ms on a 10-minute extraction (591
  /// chunks, release, this host), against the seconds of CoreML inference that
  /// produced the tensors — and what it buys is that neither identity has to be
  /// re-argued the next time a producer changes.
  ///
  /// # The cap runs first (round 9)
  ///
  /// Checks 4 and 6 are applied to the GEOMETRY before `count` is derived at
  /// all, through the same `checked_output_frame_count` both producers run
  /// earlier still. Deriving the count is what allocates — three buffers sized
  /// by the very grid check 6 bounds, 404 771 544 bytes at the smallest grid it
  /// refuses — so a bound applied afterwards is a bound applied to work already
  /// done. The full sequence still runs below and still reaches checks 4 and 6
  /// in their numbered place; the preflight can only return what that sequence
  /// would have returned, because it is those two checks.
  ///
  /// # Errors
  /// Every error `check_assembled_parts` raises, plus
  /// [`crate::audio::speaker::error::ExtractError::OutputFrameCountOverflow`]
  /// from the count derivation itself.
  pub(crate) fn assemble_checked(
    raw_embeddings: Vec<f32>,
    segmentations: Vec<f64>,
    num_chunks: usize,
    num_frames_per_chunk: usize,
    onset: f32,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
  ) -> Result<Self, ExtractError> {
    // ── Checks 4 and 6, BEFORE the count buffers ──────────────────────
    // `try_count_from_segmentations` derives the very grid check 6 bounds, and
    // allocates three buffers sized by it on the way (a `num_chunks *
    // num_frames_per_chunk` chunk-count vector, the `num_output_frames`
    // aggregate/coverage pair, and the `count` itself: 404 771 544 bytes at the
    // smallest grid the cap refuses). Running the cap over the geometry first
    // means an over-cap grid costs none of them. Round 9.
    //
    // Not a second derivation: this is `check_assembled_parts`' own checks 4
    // and 6, the same functions over the same three values, so it can only
    // return what that sequence would have returned below. Check 5 is the one
    // it cannot preempt, and does not need to — a producer's `count` IS the
    // derivation's own output, so its length is equal by construction.
    checked_output_frame_count(num_chunks, chunks_sw, frames_sw)?;

    // Manual exhaustive match, deliberately not a `From` impl — see
    // `ExtractError::OutputFrameCountOverflow`'s doc. Unreachable through
    // either source's own geometry (num_chunks * step ≈ samples.len()), kept
    // typed regardless (owned.rs:663-673).
    let count = crate::audio::speaker::window::try_count_from_segmentations(
      &segmentations,
      num_chunks,
      num_frames_per_chunk,
      SEG_NUM_SLOTS,
      onset,
      chunks_sw,
      frames_sw,
    )
    .map_err(|e| match e {
      crate::audio::speaker::window::WindowError::OutputFrameCountOverflow => {
        ExtractError::OutputFrameCountOverflow
      }
    })?;
    check_assembled_parts(
      &raw_embeddings,
      &segmentations,
      &count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    )?;
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

  /// The PUBLIC construction site: validate an [`ExtractionParts`] and assemble
  /// the [`Extraction`] it describes.
  ///
  /// The door for parts this crate did not compute: mediagraph's cluster node
  /// accumulates the same seven values from TWO upstream stages across many
  /// messages (issue #110), so a dropped or misordered message reaches here as a
  /// geometry that does not describe its own tensors. Every check below exists
  /// so that failure surfaces HERE, naming the disagreeing part, instead of
  /// producing silently wrong clusters or panicking deep inside `diaric`.
  ///
  /// An earlier revision justified this constructor by contrast — the
  /// crate-private `from_parts` "trusts its in-crate callers, every
  /// `ModelSource` builds a self-consistent tensor set by construction". That
  /// contrast was the defect (round 8): `ArgmaxSource`'s tensor set is a
  /// MODEL's output, and a model is accepted on its I/O shapes. The checks are
  /// now the same at every path — see "Where these checks run" below.
  ///
  /// `num_output_frames` and `num_speakers` are not parameters — see
  /// [`ExtractionParts`]'s "Not parameters". Assembly itself is delegated to
  /// the module-private `from_parts`, so the `num_output_frames == count.len()`
  /// derivation still lives at exactly one place.
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
  ///    over buffers this one bounds. Both in-crate producers apply this same
  ///    bound to the same derivation before they allocate anything
  ///    (`checked_output_frame_count`); this is where it runs for parts a
  ///    caller assembled, which have no earlier point.
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
  /// 9. EVERY `segmentations` cell is exactly `0.0` or exactly `1.0` — the
  ///    DOMAIN on which the two backends read this tensor the same way.
  ///    Everything below booleanizes it at `seg > 0.0` (checks 10 and 11), and
  ///    so does the whole online route; `diaric`'s offline route instead SUMS
  ///    the magnitudes — `filter_embeddings`' `clean_frames += segmentations[..]`
  ///    against `0.2 * num_frames_per_chunk`
  ///    (`diarization/src/offline/algo.rs:644-679`), and stage 7's
  ///    `sum_activity == 0.0` mask (`diarization/src/pipeline/algo.rs:698-711`).
  ///    On `{0.0, 1.0}` those are the same function; off it they disagree about
  ///    how many speakers an extraction contains. Refuses nothing either
  ///    in-crate producer emits — both write only hard-binary decodes and zeroed
  ///    columns. Subsumes the finiteness of this tensor, which used to sit under
  ///    "NOT checked": a non-finite cell is not a backend SPLIT (both routes
  ///    reach `diaric::reconstruct`'s own scan and refuse), so it was left to
  ///    them; the domain equality now names the cell here for free. See
  ///    [`ExtractError::NonBinarySegmentation`].
  /// 10. Every `(chunk, slot)` whose segmentation column is active (`seg > 0.0`,
  ///     the activity rule both backends use) carries a raw-embedding row that
  ///     `raw_embedding_reaches_plda` accepts — which is to say a row BOTH
  ///     backends' row chains accept, because that predicate calls them:
  ///     [`diaric::embed::Embedding::normalize_from`] (what
  ///     [`Self::diarize_online`] runs, `f32`-narrowed norm, `1e-12` floor) must
  ///     return `Some`, AND `diaric::plda::RawEmbedding::from_wespeaker` (the
  ///     PLDA raw boundary [`Self::diarize_with`] reaches, `f64` norm,
  ///     [`PLDA_MIN_NORM`] floor) must return `Ok`, AND
  ///     [`diaric::plda::PldaTransform::project`] — the stage offline runs
  ///     immediately after that boundary, with its own `0.1` centered-norm
  ///     rejection — must return `Ok` too. None alone: a row only online accepts
  ///     makes it create a speaker where offline fails the extraction, a row
  ///     only offline accepts is silently dropped online, and a row both
  ///     ADMISSION functions take but projection refuses is the same split one
  ///     stage further in. See [`ExtractError::ActiveSlotWithoutEmbedding`] and
  ///     [`ExtractError::PldaTransformUnavailable`].
  /// 11. `count` EQUALS the count the supplied `segmentations` derive, through
  ///     the same overlap-add aggregation
  ///     [`crate::audio::speaker::window::count_from_segmentations`] runs over
  ///     `seg > 0.0`. Not a bound: offline consumes this field and online
  ///     derives its own, so a `count` above the derived one fabricates
  ///     speakers offline and a `count` below it makes offline silent where
  ///     online speaks. See [`ExtractError::CountNotSegmentationDerived`].
  /// 12. EVERY `raw_embeddings` value is finite — the whole buffer, not only
  ///     the rows check 10 reaches. An INACTIVE slot's row has no active column
  ///     to bring it to check 10, and the two backends read it in opposite ways:
  ///     offline's `diaric::pipeline::assign_embeddings` scans the WHOLE matrix
  ///     (train subset or not, active or not — stage 6 scores every row) and
  ///     fails the extraction with `NonFiniteField::Embeddings`
  ///     (`diarization/src/pipeline/algo.rs:443-455`), while
  ///     [`Self::diarize_online`] skips the inactive column before it copies the
  ///     row and returns `Ok`. Ordered after check 10 so an ACTIVE slot's
  ///     non-finite row keeps that check's more specific `(chunk, slot)`
  ///     diagnosis. Finiteness is the WHOLE of what that offline scan can find
  ///     in an `f32` buffer: its companion refusal `ShapeError::RowNormOverflow`
  ///     needs `Σ v²` to overflow `f64`, and `256 · f32::MAX² ≈ 3e79` cannot.
  ///     See [`ExtractError::NonFiniteRawEmbedding`].
  /// 13. EVERY output frame has its OWN center time: the sequence
  ///     `frames_sw.start + t * frames_sw.step + frames_sw.duration / 2`,
  ///     `t` in `0..count.len()`, is finite and STRICTLY increasing. Check 2
  ///     asks whether `frames_sw`'s three fields are usable numbers; this asks
  ///     whether the grid they generate is a usable TIMELINE, and the two are
  ///     different questions. Every span endpoint either backend emits is one
  ///     of those centers — `diaric::reconstruct`'s `try_discrete_to_spans`
  ///     computes them at `diarization/src/reconstruct/rttm.rs:172,216-217,
  ///     231-232` — and the sum rounds: at `start = 1e9` the `f64` ULP is
  ///     `1.19e-7`, so a `step` of `1e-8` adds nothing and adjacent frames
  ///     share a center. A one-frame active run then closes at `start == end`
  ///     and the call returns `Ok` with a span of DURATION ZERO. Computed
  ///     through the shared `window::first_collapsed_frame_center`, which
  ///     mirrors that source's own operation order rather than an
  ///     algebraically equal rearrangement. Ordered LAST: it is pure geometry
  ///     over a range check 6 already bounds, so its position costs one pass,
  ///     while moving it earlier would shift the precedence of every check
  ///     after it. See [`ExtractError::CollapsedFrameCenter`].
  ///
  /// 14. The derived grid REACHES the last chunk's last frame. Checks 5 and 8
  ///     say which grid this is and that both mappings place every chunk on it
  ///     identically; neither says it is long enough. The length is
  ///     `round(last_chunk_end / frames_sw.step()) + 1` while the requirement is
  ///     `reconstruct_chunk_start_frame(last) + num_frames_per_chunk`, and those
  ///     are different functions of the same windows — on the shipped grid one
  ///     chunk admits 594 frames per chunk and THREE admit only 593, because the
  ///     last of the three lands at output frame 119 of 712. Below that,
  ///     `diaric::reconstruct` refuses the extraction on BOTH routes
  ///     (`ShapeError::OutputFrameCountTooSmall`,
  ///     `diarization/src/reconstruct/algo.rs:478-495`) rather than truncating,
  ///     so this check refuses nothing new — it moves a late, post-clustering
  ///     failure to assembly. Computed through the shared
  ///     `window::uncovered_last_chunk`, which calls the very
  ///     `reconstruct_chunk_start_frame` check 8 runs for every chunk; both
  ///     producers run it once more before they touch a model. Ordered LAST,
  ///     for check 13's reason. See [`ExtractError::UncoveredLastChunk`].
  ///
  /// Checks 1, 2 and 4 are the PANIC-preventing ones: `window`'s
  /// `try_aggregate_output_frame_count` asserts the first two with bare
  /// `assert!`s and [`Self::diarize_online`] `.expect(..)`s the third, so
  /// without them a publicly-assembled `Extraction` could panic far from its
  /// cause. Check 3 is what keeps every `[c][s][d]` / `[c][f][s]` index inside
  /// its buffer. Checks 5, 7-8, 10-11 and 14 are the CROSS-PART ones: each is a
  /// pair of parts that are individually well-formed and jointly describe
  /// something the producing pipeline cannot have produced — check 14's pair
  /// being the declared per-chunk frame count and the grid the two windows
  /// derive. Checks 9, 12 and 13 are neither: check 12 holds ONE part to a
  /// standard only ONE consumer enforces, check 9 holds ONE part to the domain
  /// outside which the two consumers stop reading it the same way, and check 13
  /// holds ONE part to the sub-domain on which it is a timeline at all. The
  /// first two end in the same failure — the backends disagreeing about an
  /// identical `Extraction` — arrived at without a second part being involved;
  /// check 13's failure is not a disagreement but an answer BOTH backends give
  /// and neither can mean. Check 14's is a third shape again: both backends
  /// REFUSE, identically and typed, and what the check buys is only when and
  /// where.
  ///
  /// # Where these checks run
  ///
  /// At every construction path, which is the round-8 change. The sequence
  /// above is not implemented in this method: it is `check_assembled_parts`,
  /// and `Self::assemble_checked` — the door every in-crate
  /// [`crate::audio::speaker::source::ModelSource`] now assembles through —
  /// runs the identical call. Before round 8 this constructor was the only
  /// path that ran it: [`Extractor::extract`] carried a hand-copied check 8
  /// (round 3's cure) and `ArgmaxSource::extract` carried nothing, so a
  /// segmenter with the pinned F16 I/O shapes returning `0.1` per frame
  /// assembled, through the then-`pub(crate)` unchecked `from_parts`, exactly
  /// the `Extraction` this constructor refuses. `from_parts` is now private to
  /// this module. The per-check argument for each producer is the matrix on
  /// `check_assembled_parts`.
  ///
  /// # How each part is read, by this constructor and by each backend
  ///
  /// Every finding this constructor has ever closed has the same shape: the
  /// validator reads a value one way and a backend reads it another. So the
  /// question worth asking of each part is not "is it checked" but "does the
  /// reading MATCH, and on what domain". Where a reading matches only on a
  /// sub-domain, a check has to CONFINE the input to it — a validator cannot
  /// otherwise be sound for both consumers at once.
  ///
  /// Round 8 added the second half of that question, which round 7's version of
  /// this table did not ask: **at which construction paths does the confinement
  /// apply**. A confinement that holds at one entry point and not the others is
  /// worse than none, because it invites the reasoning that produced round 8's
  /// finding 1 — round 7 confined `segmentations` to `{0.0, 1.0}` HERE, and
  /// `ArgmaxSource::extract` went on emitting fractional cells through a
  /// separate, unchecked door. Each entry below therefore names its
  /// confinement AND the paths it applies at. Since round 8 that answer is the
  /// same for every entry — all three paths, through the one
  /// `check_assembled_parts` — and it is stated per entry anyway, because the
  /// day it stops being the same for every entry is the day this table has to
  /// say so.
  ///
  /// - **`raw_embeddings`** — VALIDATOR: finite over the whole buffer (check
  ///   12), and for every ACTIVE `(chunk, slot)` a row that
  ///   `raw_embedding_reaches_plda` accepts (check 10). ONLINE: reads only an
  ///   active slot's row, through `Embedding::normalize_from`. OFFLINE: widens
  ///   the WHOLE matrix, scans all of it for finiteness, cosine-scores every row
  ///   at stage 6, and runs `from_raw_array` + `project` on the train subset.
  ///   MATCH: exact — the check is the CONJUNCTION of the two chains, run over
  ///   the union of the rows they read. The one asymmetry is that check 10
  ///   examines active rows offline's `filter_embeddings` would never project;
  ///   it is stricter than offline there, never weaker, and is discussed below.
  ///   CONFINED BY checks 10 and 12, AT all three construction paths: this
  ///   constructor, `Extractor::extract` and `ArgmaxSource::extract`, through
  ///   `check_assembled_parts`. Both sources additionally drop a failing row at
  ///   the point they write it (zeroing the slot's column), so for them the
  ///   check confirms rather than discovers.
  /// - **`segmentations`** — VALIDATOR: booleanized at `seg > 0.0` (check 10's
  ///   activity scan, check 11's count derivation). ONLINE: booleanized twice
  ///   (per-slot active-frame count → `f32` speech duration; distinct-cluster
  ///   count). OFFLINE: MAGNITUDES — `filter_embeddings`' `clean_frames +=`
  ///   against `0.2 * num_frames_per_chunk`, stage 7's `sum_activity == 0.0`.
  ///   BOTH, through the shared `diaric::reconstruct`: a finiteness scan, then
  ///   `max` over the slots of a cluster (a magnitude again). MATCH: only on
  ///   `{0.0, 1.0}`, where every one of those readings is the same function.
  ///   CONFINED BY check 9 — the round-7 fix — AT all three construction paths
  ///   since round 8. Round 7 applied it at this constructor ALONE, and that is
  ///   the gap round 8's finding 1 walked through: `write_segmentations` copies
  ///   `f64::from(speaker_ids[..])` verbatim and `ArgmaxSource::from_dir_with`
  ///   accepts any model with the pinned F16 I/O shapes, so a segmenter
  ///   returning `0.1` per frame put fractional cells into an `Extraction` this
  ///   constructor would have refused. `Extractor::extract` was inside the
  ///   domain by construction (`POWERSET_TABLE` literals), which is a reason to
  ///   record, not a reason to skip.
  /// - **`count`** — VALIDATOR: length equals the derived grid (check 5) and
  ///   values equal the derived count (check 11). OFFLINE: consumes the values
  ///   verbatim, as an INJECTIVE per-cluster count. ONLINE: reads the LENGTH
  ///   only (as `num_output_frames`) and derives its own distinct-cluster count.
  ///   MATCH: exact, and it is an equality precisely because the two readings
  ///   can only coincide at one value. Note what is NOT a mismatch: the two
  ///   engines then reconstruct against different count vectors (local slots
  ///   offline, distinct clusters online) — that is the definition of the two
  ///   engines, not a disagreement about this field. CONFINED BY checks 5 and
  ///   11, AT all three paths. At the two sources `count` is
  ///   `window::try_count_from_segmentations`' own output, so both checks are
  ///   identities — but only ONCE check 9 holds: under a fractional cell the
  ///   source's `seg >= onset` aggregation and check 11's `seg > 0.0` derivation
  ///   are different functions, and at `0.1` with the default onset `0.5` they
  ///   differ by the whole answer (zero speakers against one).
  /// - **`num_chunks`** — VALIDATOR: non-zero (check 1), both length products
  ///   (check 3), the derived grid (checks 4-5), the placement comparison (check
  ///   8), and the loop bound of checks 9-12. ONLINE and OFFLINE: the same chunk
  ///   count, as a stride multiplier and a loop bound. MATCH: exact, one
  ///   reading. CONFINED BY checks 1, 3 and 8, AT all three paths; at both
  ///   sources it is `window::chunk_starts(..).len()`, which is `>= 1` and is
  ///   the very length their two buffers were allocated at.
  /// - **`num_frames_per_chunk`** — VALIDATOR: non-zero (check 1), the
  ///   `segmentations` length (check 3), and the `[c][f][s]` stride everywhere
  ///   below. ONLINE and OFFLINE: the same stride. OFFLINE additionally reads
  ///   its MAGNITUDE, as the denominator of `filter_embeddings`' `0.2` ratio.
  ///   MATCH: exact as a stride. The ratio is a POLICY offline applies to the
  ///   same booleans, not a second reading of the field — see the train-subset
  ///   entry below, which is where its consequence is recorded. CONFINED BY
  ///   checks 1 and 3, AT all three paths; at `Extractor::extract` it is
  ///   `SegmentModel::num_frames()` (contract-pinned `>= 1`), at
  ///   `ArgmaxSource::extract` the `ARGMAX_FRAMES_PER_WINDOW` constant.
  /// - **`chunks_sw`** — VALIDATOR: a usable grid (check 2); `duration` and
  ///   `step` derive the output-frame count (checks 4-6); and the two
  ///   chunk-to-frame mappings are compared chunk by chunk (check 8). ONLINE:
  ///   the same derivation, the same aggregation (which reads `step` only), then
  ///   `reconstruct` (which reads `start` too). OFFLINE: `reconstruct`, the
  ///   same. MATCH: exact, and check 8 is itself a confinement — a grid whose
  ///   two mappings differ is refused, so on the admitted set the aggregation
  ///   and the reconstruction agree by construction rather than by luck.
  ///   CONFINED BY checks 2, 4-6 and 8, AT all three paths. `Extractor::extract`
  ///   ALSO runs check 8 before it touches a model, so a grid it cannot diarize
  ///   honestly costs no inference — that pre-check is a cost optimisation now,
  ///   not the guarantee; the guarantee is the shared sequence at assembly.
  /// - **`frames_sw`** — VALIDATOR: a usable grid (check 2), a `step` that
  ///   survives narrowing to `f32` (check 7), and the same mapping comparison
  ///   (check 8). ONLINE: narrows `step` to `f32` to build the speech duration
  ///   the `min_speech_duration` gate reads, and passes the window to
  ///   `reconstruct`. OFFLINE: `reconstruct` only — it never narrows. MATCH:
  ///   only on the sub-domain where the narrowing is faithful, which is exactly
  ///   what check 7 CONFINES the input to. But `start` and `duration` are NOT
  ///   read only by check 8's comparison, which is what round 7's version of
  ///   this line got wrong: BOTH backends also read all three fields forward,
  ///   as the frame CENTERS every span endpoint is built from
  ///   (`diaric`'s `try_discrete_to_spans`,
  ///   `diarization/src/reconstruct/rttm.rs:216-217,231-232`). That reading has
  ///   its own sub-domain — the centers must stay distinct — and a window can be
  ///   inside check 2's and check 7's sub-domains and outside it: `start = 1e9,
  ///   step = 1e-8` collapses adjacent centers onto one `f64` and both backends
  ///   then return `Ok` with a span of duration zero. CONFINED BY checks 2, 7,
  ///   8 and 13, AT all three paths. At both sources it is
  ///   `window::frame_sliding_window()`'s fixed `(0.0, 0.0619375, 0.016875)`,
  ///   which is inside every one of those sub-domains past
  ///   [`MAX_OUTPUT_FRAMES`].
  ///
  /// What is left after that is not a divergent READING of any part. It is
  /// divergent POLICY over readings that agree — offline's train-subset
  /// selection and online's `min_speech_duration` gate, both below — which is a
  /// property of choosing between two clustering engines, not of the parts.
  ///
  /// # What is deliberately NOT checked
  ///
  /// - **`count[t] <= diaric::reconstruct::MAX_COUNT_PER_FRAME`.** Now IMPLIED
  ///   by check 11 and kept unchecked for that reason rather than by deferral:
  ///   check 11 makes `count` EQUAL the derived value, and that derivation is
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
  /// - **An INACTIVE slot carrying a usable embedding row** — the converse of
  ///   check 10, and deliberately allowed. *Verified against both:* ONLINE,
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
  ///   output-irrelevant to BOTH engines. (`extract`'s own output now goes
  ///   through this very constructor's checks, via
  ///   `Self::assemble_checked` — but a refusal here would still be the wrong
  ///   cure, because it would refuse a shape neither engine can act on.) Both
  ///   halves are pinned by
  ///   `an_inactive_slots_row_cannot_change_the_online_result`.
  ///
  ///   So the shape stays a consequence of the caller's own data ("this slot
  ///   has an embedding but no speech"), not of a part disagreeing with
  ///   another; `tiny_extraction`'s third slot is exactly it.
  /// - **`num_output_frames` covering the last chunk's last frame — WAS on this
  ///   list, and is now check 14.** Kept here as history, because the reason it
  ///   was omitted is the useful part: it was not "both backends refuse, so we
  ///   need not", which is true and would have applied to check 8 as well. It
  ///   was that re-deriving the requirement meant writing `closest_frame`'s
  ///   float arithmetic out a SECOND time, and a second spelling of a rounding
  ///   expression is exactly the drift check 8 exists to catch — so the check
  ///   would have introduced the class it was meant to guard.
  ///
  ///   That reasoning lapsed at round 8, and quietly. Round 8 added
  ///   `window::reconstruct_chunk_start_frame` — one mirror of that private
  ///   `closest_frame`, anchored by `window::tests` against `diaric`'s own
  ///   OBSERVABLE placement, read back out of the very
  ///   `OutputFrameCountTooSmall { required }` this check predicts — and check 8
  ///   already called it for every chunk. From that commit the derivation this
  ///   entry declined to make was already being made, on every path, and already
  ///   pinned; the requirement is that mirror plus `num_frames_per_chunk`, so
  ///   check 14 is a CALL and not a copy. The omission outlived its own premise
  ///   by three rounds, which is the general lesson: an entry on this list is
  ///   only as good as the code it was written against, and a "we would have to
  ///   duplicate X" justification expires the moment X gains a shared, anchored
  ///   definition.
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
  ///   check 10 examines exactly the rows online examines, and a superset of the
  ///   rows offline trains on. The asymmetry runs the safe way: a check that
  ///   examined FEWER rows than a backend reads is the failure mode, and it is
  ///   the one check 12 just closed. *Deliberate:* the alternative is a
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
  /// - [`ExtractError::NonBinarySegmentation`] — check 9.
  /// - [`ExtractError::ActiveSlotWithoutEmbedding`] — check 10.
  /// - [`ExtractError::PldaTransformUnavailable`] — check 10's transform could
  ///   not be built, so the row standard cannot be applied at all.
  /// - [`ExtractError::CountNotSegmentationDerived`] — check 11.
  /// - [`ExtractError::NonFiniteRawEmbedding`] — check 12.
  /// - [`ExtractError::CollapsedFrameCenter`] — check 13.
  /// - [`ExtractError::UncoveredLastChunk`] — check 14.
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
    let ExtractionParts {
      raw_embeddings,
      segmentations,
      count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    } = parts;

    // Every check, in the order the list above documents, through the ONE
    // function every in-crate producer also runs (`Self::assemble_checked`).
    // Written out here a second time it would be a second sequence to keep in
    // step — which is precisely how `ArgmaxSource` came to emit an extraction
    // this constructor refuses (round 8).
    check_assembled_parts(
      &raw_embeddings,
      &segmentations,
      &count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    )?;

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

/// The flat `raw_embeddings` sub-block for chunk `c` — ALL [`SEG_NUM_SLOTS`]
/// rows at once: `c * S * D .. (c + 1) * S * D`. The chunk-granular face of
/// [`embedding_range`], and the unit [`embed_chunk_slots`] writes, because the
/// chunk is the granularity at which the embed stage decides anything.
///
/// `chunk_embedding_range(c).start + slot_embedding_range(s).start ==
/// embedding_range(c, s).start`, pinned by
/// `chunk_and_slot_embedding_ranges_compose_to_embedding_range`.
fn chunk_embedding_range(c: usize) -> core::ops::Range<usize> {
  let stride = SEG_NUM_SLOTS * EMBEDDING_DIM;
  c * stride..(c + 1) * stride
}

/// Slot `s`'s row WITHIN one chunk's [`chunk_embedding_range`] block: `s * D ..
/// (s + 1) * D`. The chunk-relative half of [`embedding_range`]'s absolute
/// offset.
fn slot_embedding_range(s: usize) -> core::ops::Range<usize> {
  let base = s * EMBEDDING_DIM;
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

/// Steps (e)-(g) of [`Extractor::extract`]'s fused per-chunk loop — the EMBED
/// stage for ONE chunk — and THE single implementation of it.
///
/// Two callers, one body: [`Extractor::extract`] runs this once per chunk over
/// sub-slices of its own two tensors, and [`Extractor::extract_chunk_embeddings`]
/// runs it once over a caller's chunk. Neither restates the overlap-exclusion
/// policy, the drop paths, or the placeholder-mask batching, so the fused path
/// and the split (`segmentation → embed → cluster`) path cannot drift. That
/// property is the whole reason the split's embed end is a per-chunk DOOR rather
/// than a published [`derive_slot_plans`] a caller would have to loop correctly
/// itself.
///
/// - `padded` is the chunk's audio window, exactly [`SEG_CHUNK_SAMPLES`] long.
/// - `chunk_segs` is the chunk's `[f][s]` slab, `num_frames * SEG_NUM_SLOTS`
///   long, MUTATED in place: every dropped slot's column is zeroed, by both the
///   Skip drop (step f, `owned.rs:561-571`) and the PLDA-norm drop (step g,
///   `owned.rs:619-630`).
/// - `chunk_embeddings` is the chunk's `[s][d]` block, `SEG_NUM_SLOTS *
///   EMBEDDING_DIM` long. Only a SURVIVING slot's row is written, so the caller
///   hands a ZEROED block — dia's pre-zeroed, never-written rows
///   (`owned.rs:502-505`) are what a dropped slot must leave behind, and
///   [`Extraction::try_from_parts`]' check 10 reads exactly that.
/// - `plda` is [`shared_plda_transform`]'s value, hoisted out of the caller's
///   loop; `onset` is the already-validated threshold.
///
/// # Errors
/// [`ExtractError::Infer`] if the embedding call fails (`owned.rs:600`).
///
/// # Panics
/// Panics if `chunk_segs.len() != num_frames * SEG_NUM_SLOTS`
/// ([`derive_slot_plans`]' own contract) or if `chunk_embeddings` is shorter
/// than `SEG_NUM_SLOTS * EMBEDDING_DIM`. Both callers derive the two lengths
/// from the same `num_frames` before calling — `extract` from the frame count
/// its check 5 already agreed between the two models, the public door from a
/// length check against the embedder's own.
fn embed_chunk_slots(
  embed: &EmbedModel,
  plda: &diaric::plda::PldaTransform,
  padded: &[f32],
  chunk_segs: &mut [f64],
  chunk_embeddings: &mut [f32],
  num_frames: usize,
  onset: f64,
) -> Result<(), ExtractError> {
  // e. Per-slot embedding plans from the overlap-exclusion rule
  // (owned.rs:507-591).
  let plans = derive_slot_plans(chunk_segs, num_frames, onset);

  // f. Zero every Skip slot's segmentation column (owned.rs:561-571).
  for (s, plan) in plans.iter().enumerate() {
    if matches!(plan, SlotPlan::Skip) {
      zero_slot_column(chunk_segs, num_frames, s);
    }
  }

  // g. One batched embed call if any slot is planned; Skip slots borrow the
  // first planned slot's mask as a non-degenerate placeholder and their output
  // rows are discarded (module doc).
  let placeholder = plans.iter().find_map(|p| match p {
    SlotPlan::Embed(mask) => Some(mask.as_slice()),
    SlotPlan::Skip => None,
  });
  if let Some(placeholder) = placeholder {
    let masks: [&[bool]; EMBED_SLOTS] = core::array::from_fn(|s| match &plans[s] {
      SlotPlan::Embed(mask) => mask.as_slice(),
      SlotPlan::Skip => placeholder,
    });
    let rows = embed.embed_chunk(padded, &masks)?;
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
        zero_slot_column(chunk_segs, num_frames, s);
      } else {
        chunk_embeddings[slot_embedding_range(s)].copy_from_slice(&rows[s]); // owned.rs:631-632
      }
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests;
