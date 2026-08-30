//! Structured, per-domain error types for the `speakerkit` backends (design
//! spec §5). Foreign errors from `coremlit` are wrapped as typed `#[from]`
//! variants; [`ExtractError`] composes both domain errors at the top level.

/// Failure locating, loading, or validating a CoreML segmentation or
/// embedding model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] crate::LoadError),
  /// A loaded model's input or output feature does not match the
  /// shape/dtype contract this crate was built against (see
  /// `tests/model_io.rs` for the pinned ground truth).
  #[error("model contract mismatch on `{feature}`: expected {expected}, got {actual}")]
  ContractMismatch {
    /// Name of the input/output feature that mismatched.
    feature: &'static str,
    /// The contract this crate expects, rendered for display.
    expected: String,
    /// What the loaded model actually declares, rendered for display.
    actual: String,
  },
}

/// Failure running or interpreting a segmentation or embedding inference
/// call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InferError {
  /// The CoreML runtime failed to run the model.
  #[error("prediction failed: {0}")]
  Prediction(#[from] crate::PredictionError),
  /// A tensor failed to construct or view.
  #[error("tensor failed: {0}")]
  Tensor(#[from] crate::TensorError),
  /// An output tensor contained a NaN or infinite value — the exact `ort`
  /// CoreML-EP failure mode this crate exists to replace (spec §6, gate 2).
  #[error("output contains a non-finite value at index {index}")]
  NonFiniteOutput {
    /// Flat index of the offending element.
    index: usize,
  },
  /// The caller's input slice did not have the model's required length.
  #[error("input length mismatch: expected {expected}, got {got}")]
  InputLength {
    /// Elements the caller provided.
    got: usize,
    /// Elements the model requires.
    expected: usize,
  },
  /// A predict-time output tensor's shape diverged from the contract
  /// validated at construction. `crate::MultiArray::copy_into` alone
  /// only validates total element count, so an axes-swapped output (e.g.
  /// `[1, classes, frames]` instead of `[1, frames, classes]`) can carry
  /// the same element count as the expected shape and would otherwise pass
  /// silently, transposing two axes instead of erroring.
  #[error("output shape mismatch: expected {expected:?}, got {got:?}")]
  OutputShape {
    /// Shape the runtime tensor actually had.
    got: Vec<usize>,
    /// Shape the construction-time contract declares.
    expected: Vec<usize>,
  },
  /// The caller's input contained a NaN or infinite value before inference
  /// ran. Complements [`Self::NonFiniteOutput`]: an unchecked NaN sample
  /// can otherwise propagate silently into a finite-looking but garbage
  /// embedding that no downstream check would catch. Mirrors dia's
  /// analogous embed-side guard, `embed::Error::NonFiniteInput`
  /// (`diarization/src/embed/error.rs:107-109`) — a unit variant there.
  /// This variant adds the offending flat index, matching this crate's own
  /// [`Self::NonFiniteOutput`] shape: a deliberate enhancement over dia's,
  /// not a parity requirement (dia's own variant carries no index).
  #[error("input contains a non-finite value at index {index}")]
  NonFiniteInput {
    /// Flat index of the offending element.
    index: usize,
  },
  /// The caller's input was finite in `f32` but its magnitude exceeds `f16`'s
  /// finite range (`|x| > f16::MAX`, i.e. `65504`), so narrowing it to the
  /// argmax segmenter's `.float16` `waveform` input would round it to an f16
  /// infinity and reach CoreML as a non-finite value — the very thing
  /// [`Self::NonFiniteInput`] exists to prevent, one representability step in.
  /// Only the argmax source narrows host `f32` samples to `f16` before
  /// inference; the FluidAudio and dia-coreml paths feed `f32` unchanged, so
  /// this guard is scoped to that source's `extract` (the public contract
  /// places no amplitude bound on `samples`, `source/mod.rs`).
  #[error(
    "input value at index {index} is finite in f32 but overflows the model's f16 input \
     domain (|x| > f16::MAX)"
  )]
  F16OverflowInput {
    /// Flat index of the offending element.
    index: usize,
  },
  /// A per-frame speaker-activity mask had no active (`true`) frame at
  /// all. Every WeSpeaker call backed by an all-zero mask would receive
  /// all-zero pooling weights, which divides by zero inside statistics
  /// pooling and yields a NaN/Inf row — rejected here as a typed error
  /// instead. Mirrors dia's `embed::Error::EmptyOrInactiveMask`
  /// (`diarization/src/embed/error.rs:65-71`; the check itself lives at
  /// `diarization/src/embed/model.rs:646-649`).
  #[error("mask has no active (true) frame")]
  EmptyMask,
}

/// Which member of [`crate::audio::speaker::extract::ExtractionParts`] a
/// [`crate::audio::speaker::extract::Extraction::try_from_parts`] validation
/// failure names — one variant per field of that struct.
///
/// The point of naming the part is diagnostic reach: the mediagraph cluster node
/// assembles those seven fields from TWO upstream stages across many messages
/// (issue #110), so "which part disagreed" is the same question as "which
/// upstream stage dropped or reordered a message".
///
/// Closed (no `#[non_exhaustive]`) on purpose: the vocabulary IS
/// `ExtractionParts`'s field set, so a new variant can only appear alongside a
/// new part — already a breaking change to that struct — and a caller routing a
/// failure back to the stage that produced it wants an exhaustive `match`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
pub enum ExtractionPart {
  /// `raw_embeddings`: the flattened `[c][s][d]` pre-PLDA embedding tensor.
  #[display("raw_embeddings")]
  RawEmbeddings,
  /// `segmentations`: the flattened `[c][f][s]` activity tensor.
  #[display("segmentations")]
  Segmentations,
  /// `count`: the per-output-frame speaker count. Its LENGTH is
  /// `num_output_frames`, which is why an empty `count` is reported as a
  /// zero dimension.
  #[display("count")]
  Count,
  /// `num_chunks`: the number of sliding-window chunks.
  #[display("num_chunks")]
  NumChunks,
  /// `num_frames_per_chunk`: the segmentation model's per-chunk frame count.
  #[display("num_frames_per_chunk")]
  NumFramesPerChunk,
  /// `chunks_sw`: the outer (chunk-level) sliding window.
  #[display("chunks_sw")]
  ChunksSw,
  /// `frames_sw`: the inner (frame-level) sliding window.
  #[display("frames_sw")]
  FramesSw,
}

/// A flattened tensor whose length disagrees with the geometry declared beside
/// it in the same [`crate::audio::speaker::extract::ExtractionParts`].
///
/// Payload of [`ExtractError::ExtractionLenMismatch`]. Extracted into a named
/// struct rather than written as a struct-shaped variant, per this repository's
/// `rust-type-conventions` ("Variants are UNIT or NEWTYPE only").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtractionLenMismatch {
  part: ExtractionPart,
  got: usize,
  expected: usize,
}

impl ExtractionLenMismatch {
  /// The rejected tensor — [`ExtractionPart::RawEmbeddings`],
  /// [`ExtractionPart::Segmentations`], or [`ExtractionPart::Count`].
  #[inline(always)]
  pub const fn part(&self) -> ExtractionPart {
    self.part
  }
  /// The length the caller actually supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }
  /// The length the caller's own declared geometry requires:
  /// `num_chunks * num_speakers * EMBEDDING_DIM` for
  /// [`ExtractionPart::RawEmbeddings`], `num_chunks * num_frames_per_chunk *
  /// num_speakers` for [`ExtractionPart::Segmentations`], and — for
  /// [`ExtractionPart::Count`], whose length IS `num_output_frames` — the
  /// output-frame count the two sliding windows and `num_chunks` derive, the
  /// same one [`crate::audio::speaker::extract::Extraction::diarize_online`]
  /// re-derives and requires.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(part: ExtractionPart, got: usize, expected: usize) -> Self {
    Self {
      part,
      got,
      expected,
    }
  }
}

/// The element count a declared geometry requires overflows `usize`, so NO
/// slice length can satisfy it and no expected length can even be named.
///
/// Payload of [`ExtractError::ExtractionGeometryOverflow`]. Reported separately
/// from [`ExtractionLenMismatch`] because the two are different diagnoses: a
/// mismatch means "this tensor is the wrong size", an overflow means "these
/// dimensions are not a describable tensor at all". Mirrors `diaric`'s own split
/// between `ShapeError::RawEmbeddingsOverflow` /
/// `ShapeError::SegmentationsOverflow` and their `*LenMismatch` counterparts
/// (`diarization/src/offline/algo.rs:575-588`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtractionGeometryOverflow {
  part: ExtractionPart,
  num_chunks: usize,
  num_frames_per_chunk: usize,
}

impl ExtractionGeometryOverflow {
  /// Whose length product overflowed — [`ExtractionPart::RawEmbeddings`] or
  /// [`ExtractionPart::Segmentations`].
  #[inline(always)]
  pub const fn part(&self) -> ExtractionPart {
    self.part
  }
  /// The declared `num_chunks`. A factor of BOTH products.
  #[inline(always)]
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// The declared `num_frames_per_chunk`. A factor of the `segmentations`
  /// product only; reported for [`ExtractionPart::RawEmbeddings`] too because
  /// both dimensions come from the same message-assembly step and a caller
  /// debugging one wants to see the other.
  #[inline(always)]
  pub const fn num_frames_per_chunk(&self) -> usize {
    self.num_frames_per_chunk
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(
    part: ExtractionPart,
    num_chunks: usize,
    num_frames_per_chunk: usize,
  ) -> Self {
    Self {
      part,
      num_chunks,
      num_frames_per_chunk,
    }
  }
}

/// A supplied [`crate::audio::speaker::window::SlidingWindow`] is not a usable
/// timing grid.
///
/// Payload of [`ExtractError::InvalidSlidingWindow`]. Carries the whole window
/// rather than a "which field" tag: all three components are needed to see what
/// the assembling caller actually sent, and
/// [`crate::audio::speaker::window::SlidingWindow`] is `Copy`.
///
/// No `Eq`/`Hash`: `SlidingWindow` is three `f64`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidSlidingWindow {
  part: ExtractionPart,
  window: crate::audio::speaker::window::SlidingWindow,
}

impl InvalidSlidingWindow {
  /// Which window — [`ExtractionPart::ChunksSw`] or
  /// [`ExtractionPart::FramesSw`].
  #[inline(always)]
  pub const fn part(&self) -> ExtractionPart {
    self.part
  }
  /// The rejected window, verbatim.
  #[inline(always)]
  pub const fn window(&self) -> crate::audio::speaker::window::SlidingWindow {
    self.window
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(
    part: ExtractionPart,
    window: crate::audio::speaker::window::SlidingWindow,
  ) -> Self {
    Self { part, window }
  }
}

/// A `(chunk, slot)` whose `segmentations` column claims speech but whose
/// `raw_embeddings` row is not a usable embedding.
///
/// Payload of [`ExtractError::ActiveSlotWithoutEmbedding`]. The two indices are
/// what a caller needs to route the failure back to the upstream stage that
/// produced the row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActiveSlotWithoutEmbedding {
  chunk: usize,
  slot: usize,
}

impl ActiveSlotWithoutEmbedding {
  /// Index of the chunk carrying the offending slot.
  #[inline(always)]
  pub const fn chunk(&self) -> usize {
    self.chunk
  }
  /// Index of the speaker slot within that chunk, in `0..SEG_NUM_SLOTS`.
  #[inline(always)]
  pub const fn slot(&self) -> usize {
    self.slot
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(chunk: usize, slot: usize) -> Self {
    Self { chunk, slot }
  }
}

/// A `count[t]` that is not the value the `segmentations` supplied beside it
/// derive at output frame `t`.
///
/// Payload of [`ExtractError::CountNotSegmentationDerived`]. `expected` is the
/// overlap-add aggregation
/// [`crate::audio::speaker::window::count_from_segmentations`] runs, taken over
/// the activity predicate BOTH cluster backends apply to a segmentation column
/// (`seg > 0.0`, `diaric`'s "any nonzero entry is binary-active" convention) —
/// so it is the one count the two of them can agree on, and the one
/// [`crate::audio::speaker::extract::Extractor::extract`] itself produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CountNotSegmentationDerived {
  frame: usize,
  got: u8,
  expected: u8,
}

impl CountNotSegmentationDerived {
  /// The output frame `t` whose count disagrees. The FIRST such frame: the
  /// scan stops at the earliest disagreement.
  #[inline(always)]
  pub const fn frame(&self) -> usize {
    self.frame
  }
  /// The `count[t]` the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> u8 {
    self.got
  }
  /// The `count[t]` the caller's own `segmentations` and geometry derive at
  /// that frame — the only value accepted there.
  #[inline(always)]
  pub const fn expected(&self) -> u8 {
    self.expected
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(frame: usize, got: u8, expected: u8) -> Self {
    Self {
      frame,
      got,
      expected,
    }
  }
}

/// A chunk that the count aggregation and `diaric`'s reconstruction place at
/// DIFFERENT output frames.
///
/// Payload of [`ExtractError::MisalignedChunkPlacement`]. The two mappings are
/// [`crate::audio::speaker::window`]'s `aggregate_chunk_start_frame` (dia's
/// `count.rs` expression, which reads no window origin) and its
/// `reconstruct_chunk_start_frame` (the mirror of
/// `diaric::reconstruct`'s private `closest_frame`, which reads both). They are
/// algebraically equal whenever the two origins cancel, but not NUMERICALLY
/// equal: the reconstruction route adds `frames_sw.duration / 2` to the chunk
/// start and subtracts it again, and that round trip is not the identity in
/// binary floating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkPlacementMismatch {
  chunk: usize,
  aggregated: i64,
  reconstructed: i64,
}

impl ChunkPlacementMismatch {
  /// Index of the first chunk the two mappings disagree about.
  #[inline(always)]
  pub const fn chunk(&self) -> usize {
    self.chunk
  }
  /// The output frame the COUNT aggregation places that chunk's first frame at.
  #[inline(always)]
  pub const fn aggregated(&self) -> i64 {
    self.aggregated
  }
  /// The output frame `diaric::reconstruct` places the same frame at.
  #[inline(always)]
  pub const fn reconstructed(&self) -> i64 {
    self.reconstructed
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(chunk: usize, aggregated: i64, reconstructed: i64) -> Self {
    Self {
      chunk,
      aggregated,
      reconstructed,
    }
  }
}

/// Top-level extraction failure, composing model-lifecycle and inference
/// errors (spec §5) plus [`crate::audio::speaker::extract::Extractor::extract`]'s own
/// input-validation and geometry guards.
// No `Eq`: `OnsetOutOfRange` carries an `f32` payload, and `f32` is not
// `Eq` (mirrors dia's own `ShapeError::OnsetOutOfRange { onset: f32 }`,
// `diarization/src/offline/algo.rs:90-97`, which is likewise not `Eq`).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ExtractError {
  /// A model failed to load, or its contract mismatched.
  #[error("model error: {0}")]
  Model(#[from] ModelError),
  /// An inference call failed.
  #[error("infer error: {0}")]
  Infer(#[from] InferError),
  /// The caller passed an empty `samples` slice. Mirrors dia's own
  /// first-line guard, `ShapeError::EmptySamples`
  /// (`diarization/src/offline/owned.rs:369-371`): with no audio there is
  /// no chunk grid to build.
  #[error("samples is empty")]
  EmptySamples,
  /// The configured `step_samples` is `0`. Mirrors dia's
  /// `ShapeError::ZeroStepSamples`
  /// (`diarization/src/offline/owned.rs:374-376`): a zero step would hang
  /// the chunk planner's `div_ceil`. [`crate::audio::speaker::window::WindowOptions`]'s
  /// own builders already reject this, so reaching it means a
  /// serde-deserialized config bypassed the builder — defense-in-depth,
  /// exactly as dia re-checks it here despite `with_step_samples`'s panic.
  #[error("step_samples must be > 0")]
  ZeroStepSamples,
  /// The configured `step_samples` exceeds [`crate::audio::speaker::segment::SEG_CHUNK_SAMPLES`].
  /// Mirrors dia's `ShapeError::StepSamplesExceedsWindow`
  /// (`diarization/src/offline/owned.rs:377-387`, whose own comment gives
  /// the serde-bypass defense-in-depth rationale): with `step > window`,
  /// samples in `[window .. step)` per chunk are never segmented or
  /// embedded — silent data loss returning `Ok(_)` with missing speech.
  #[error("step_samples ({step}) must not exceed SEG_CHUNK_SAMPLES ({window})")]
  StepSamplesExceedsWindow {
    /// The rejected `step_samples`.
    step: u32,
    /// The chunk window length ([`crate::audio::speaker::segment::SEG_CHUNK_SAMPLES`]).
    window: usize,
  },
  /// The configured `onset` is not finite in `(0.0, 1.0]`. Mirrors dia's
  /// `ShapeError::OnsetOutOfRange`
  /// (`diarization/src/offline/owned.rs:388-393`) and
  /// [`crate::audio::speaker::window`]'s `check_onset` `(0.0, 1.0]` contract: the hard
  /// segmentation mask `seg >= onset` degenerates — `> 1.0`/NaN makes
  /// every frame inactive (empty diarization), `<= 0.0` makes every zero
  /// cell active (corrupted masks/counts).
  #[error("onset ({onset}) must be finite in (0.0, 1.0]")]
  OnsetOutOfRange {
    /// The rejected `onset`.
    onset: f32,
  },
  /// The configured `step_samples` is one the selected source cannot honor
  /// because its sliding-window stride is compiled INTO the model graph.
  ///
  /// Raised only by [`crate::audio::speaker::source::ArgmaxSource`]: argmax's segmenter
  /// slides its 21 windows internally at a fixed
  /// [`crate::audio::speaker::source::argmax::ARGMAX_WINDOW_STRIDE_SAMPLES`] (16 000 = 1 s,
  /// derived from the graph's own `[21, 1, 160000]` output shape), so there
  /// is no knob to vary. [`crate::audio::speaker::extract::Extractor`]'s host-side chunk
  /// planner has no such constraint and accepts any `step_samples` in
  /// `(0, SEG_CHUNK_SAMPLES]`.
  ///
  /// Rejected rather than ignored: silently overriding the caller's
  /// `step_samples` would return an `Extraction` whose `chunks_sw.step()`
  /// did not describe its own chunk grid, corrupting every downstream time
  /// offset `diaric` reconstructs from it.
  #[error(
    "step_samples ({step}) is not supported by this source: its window stride is fixed at \
     {required} by the model graph"
  )]
  UnsupportedStepSamples {
    /// The rejected `step_samples`.
    step: u32,
    /// The stride the source's graph requires.
    required: u32,
  },
  /// The segmentation model's per-chunk frame count disagrees with the
  /// embedding model's mask frame count. This guard has NO dia analog and
  /// cannot: dia shares one `FRAMES_PER_WINDOW` const across both stages
  /// (`diarization/src/offline/owned.rs:479,540`), so its two stages are
  /// frame-aligned by construction. This crate's two models declare their
  /// frame counts independently at load
  /// ([`crate::audio::speaker::segment::SegmentModel::num_frames`],
  /// [`crate::audio::speaker::embed::EmbedModel::num_mask_frames`]); a mismatch would
  /// silently repeat-pad time-misaligned masks (`embed_chunk` pads each
  /// mask to its OWN frame count), so it is rejected up front instead.
  #[error(
    "segmenter frame count ({segmenter}) does not match embedder mask frame count ({embedder})"
  )]
  FrameCountMismatch {
    /// The segmentation model's per-chunk frame count.
    segmenter: usize,
    /// The embedding model's mask frame count.
    embedder: usize,
  },
  /// A required dimension of the
  /// [`crate::audio::speaker::extract::ExtractionParts`] handed to
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`] is zero:
  /// [`ExtractionPart::NumChunks`], [`ExtractionPart::NumFramesPerChunk`], or
  /// [`ExtractionPart::Count`] (whose LENGTH is `num_output_frames`).
  ///
  /// Each of the three is separately rejected by `diaric`
  /// (`ShapeError::ZeroNumChunks` / `ZeroNumFramesPerChunk` /
  /// `ZeroNumOutputFrames`, `diarization/src/offline/algo.rs:542-550,602-606`),
  /// but a zero `num_chunks` or `num_frames_per_chunk` ALSO trips a bare
  /// `assert!` inside this crate's own
  /// `window::try_aggregate_output_frame_count` on the
  /// [`crate::audio::speaker::extract::Extraction::diarize_online`] route — a
  /// panic, not a typed error. Rejecting here is what keeps every public method
  /// on a publicly-constructed `Extraction` panic-free.
  #[error("extraction parts: {0} must be non-zero")]
  ZeroExtractionDimension(ExtractionPart),
  /// A flattened tensor's length disagrees with the geometry declared beside it
  /// in the same [`crate::audio::speaker::extract::ExtractionParts`]. See
  /// [`ExtractionLenMismatch`] for which tensor and by how much.
  #[error(
    "extraction parts: {} has length {} but the declared geometry requires {}",
    .0.part(),
    .0.got(),
    .0.expected()
  )]
  ExtractionLenMismatch(ExtractionLenMismatch),
  /// The element count a declared geometry requires overflows `usize`, so no
  /// slice can ever match it. Raised BEFORE the corresponding length equality,
  /// because an unchecked product could wrap to a small value that a short (or
  /// empty) slice would spuriously satisfy. See [`ExtractionGeometryOverflow`].
  #[error(
    "extraction parts: the length {} requires overflows usize (num_chunks {}, \
     num_frames_per_chunk {}, num_speakers {})",
    .0.part(),
    .0.num_chunks(),
    .0.num_frames_per_chunk(),
    crate::audio::speaker::segment::SEG_NUM_SLOTS
  )]
  ExtractionGeometryOverflow(ExtractionGeometryOverflow),
  /// A supplied [`crate::audio::speaker::window::SlidingWindow`] is not a usable
  /// timing grid: `start` must be finite and `duration`/`step` must both be
  /// finite and strictly positive.
  ///
  /// Not merely `diaric`'s contract (`TimingError::NonFiniteParameter` /
  /// `NonPositiveDurationOrStep`,
  /// `diarization/src/reconstruct/algo.rs:397-405`): this crate's own
  /// `window::try_aggregate_output_frame_count` asserts the duration/step half
  /// with bare `assert!`s, so a non-positive or non-finite window would PANIC
  /// inside [`crate::audio::speaker::extract::Extraction::diarize_online`]
  /// rather than surface a typed error. See [`InvalidSlidingWindow`].
  #[error(
    "extraction parts: {} is not a usable timing grid (start {}, duration {}, step {}) — \
     start must be finite and duration/step must be finite and > 0",
    .0.part(),
    .0.window().start(),
    .0.window().duration(),
    .0.window().step()
  )]
  InvalidSlidingWindow(InvalidSlidingWindow),
  /// The two mappings from a chunk index to an output frame — the one the
  /// `count` aggregation uses and the one `diaric::reconstruct` uses — place a
  /// chunk at DIFFERENT frames, so the count is written against activations
  /// that are not there.
  ///
  /// [`crate::audio::speaker::window::count_from_segmentations`] — the
  /// aggregation [`crate::audio::speaker::extract::Extraction::diarize_online`]
  /// runs to build its own count, and the one
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`] validates
  /// `count` against — places chunk `c` at `round(c * chunks_sw.step /
  /// frames_sw.step)`, reading NEITHER origin (dia's own
  /// `diarization/src/aggregate/count.rs` does the same, and dia's pipeline only
  /// ever passes `0.0`). `diaric::reconstruct`, which BOTH backends then feed,
  /// places the same chunk at `frames_sw.closest_frame(chunks_sw.start + c *
  /// chunks_sw.step + frames_sw.duration / 2)`, which reads both
  /// (`diarization/src/reconstruct/algo.rs:110,690`). Where the two differ, the
  /// surviving `count` marks zero-activation cells active and suppresses the
  /// real ones — speech silently shifted onto the wrong frames, returned as
  /// `Ok`.
  ///
  /// Testing the two origins for `0.0` is NOT this condition, in either
  /// direction. Too weak: the reconstruction route adds `frames_sw.duration / 2`
  /// to the chunk start and subtracts it again, and that round trip is not the
  /// identity in binary floating point, so a grid with both origins at `0.0` can
  /// still disagree by a whole frame. Too strong: equal, non-zero origins (both
  /// windows `start = 1.0`) cancel exactly and place every chunk identically.
  /// The condition checked is the one that matters — the mappings AGREE, chunk
  /// by chunk. See [`ChunkPlacementMismatch`] for the payload.
  ///
  /// Raised by BOTH construction paths, over the one shared
  /// `window::first_misaligned_chunk`:
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`] for parts a
  /// caller assembled, and
  /// [`crate::audio::speaker::extract::Extractor::extract`] for the grid its own
  /// `step_samples` and clip length derive — the latter before it runs a model,
  /// since a geometry this crate cannot diarize honestly is not worth inferring
  /// over. Through `extract` it is reachable only for an ODD `step_samples`: the
  /// aggregation's quotient is `c * step_samples / 270` exactly, so a rounding
  /// tie needs `c * step_samples` to be an odd multiple of `135`.
  /// [`crate::audio::speaker::window::DEFAULT_STEP_SAMPLES`] and argmax's fixed
  /// stride are both even, so neither can raise it.
  #[error(
    "chunk {} is placed at output frame {} by the count aggregation but at \
     frame {} by diaric's reconstruction — the two grids must agree, or the count selects \
     frames the activations never reach",
    .0.chunk(),
    .0.aggregated(),
    .0.reconstructed()
  )]
  MisalignedChunkPlacement(ChunkPlacementMismatch),
  /// `frames_sw.step()` is finite and strictly positive in `f64` but does not
  /// survive the narrowing to `f32` that
  /// [`crate::audio::speaker::extract::Extraction::diarize_online`] applies to
  /// it.
  ///
  /// That method builds the online engine's speech duration as
  /// `active_frame_count as f32 * (frames_sw.step() as f32)` — FluidAudio's
  /// `Float(activity) * slidingWindow.step`, kept in `f32` for
  /// bit-parity with the Swift oracle (`tests/parity_online_swift.rs`). A step
  /// below `f32`'s smallest subnormal narrows to `0.0`, so every slot is handed
  /// a zero speech duration and the `min_speech_duration` gate drops speakers
  /// whose DECLARED duration meets it; a step above `f32::MAX` narrows to
  /// `+inf`, which makes an inactive slot's duration `0.0 * inf = NaN`. Both
  /// are silent: `Ok`, with the wrong speakers.
  ///
  /// Rejected rather than repaired by widening the arithmetic, because the
  /// `f32` product is the parity contract; a step this crate cannot represent
  /// in it is not a grid the online engine can honour. See
  /// [`InvalidSlidingWindow`] for the payload.
  #[error(
    "extraction parts: {} declares a step ({:e}) that narrows to {:e} in f32 — the online speech \
     duration is an f32 product, so the step must stay finite and > 0 through that narrowing",
    .0.part(),
    .0.window().step(),
    .0.window().step() as f32
  )]
  FrameStepNotRepresentableInF32(InvalidSlidingWindow),
  /// A `(chunk, slot)` whose `segmentations` column claims speech pairs with a
  /// `raw_embeddings` row that at least one of the two backends cannot use.
  ///
  /// The test is not a threshold this crate names; it is a CALL to each stage
  /// the backends put a row through, and ALL must accept —
  /// `extract::raw_embedding_reaches_plda` requires
  /// [`diaric::embed::Embedding::normalize_from`] to return `Some`,
  /// `diaric::plda::RawEmbedding::from_wespeaker` to return `Ok`, and
  /// [`diaric::plda::PldaTransform::project`] to return `Ok`. Any one alone
  /// leaves the two backends disagreeing about the identical `Extraction`, in
  /// opposite directions:
  ///
  /// - ONLINE, `normalize_from` returning `None` is
  ///   [`crate::audio::speaker::extract::Extraction::diarize_online`]'s
  ///   DROPPED-SLOT sentinel — a dropped slot's row is all-zero precisely so
  ///   that it is rejected there and the slot stays UNMATCHED. A corrupt row
  ///   under an ACTIVE column takes that same path, so the engine reads "no
  ///   speaker here" and returns `Ok` with the speech missing.
  /// - OFFLINE, `RawEmbedding::from_wespeaker` refuses the row outright
  ///   (`Plda(NonFiniteInput)` / `Plda(DegenerateInput)`) at a `f64` norm floor
  ///   of [`crate::audio::speaker::extract::PLDA_MIN_NORM`] (`0.01`,
  ///   `diarization/src/plda/transform.rs:72,152-165`), ten orders of
  ///   magnitude above `normalize_from`'s `NORM_EPSILON` (`1e-12`).
  /// - OFFLINE AGAIN, one stage later: `project` re-rejects a row whose
  ///   CENTERED norm `‖row - mean1‖` is below `XVEC_CENTERED_MIN_NORM` (`0.1`,
  ///   `diarization/src/plda/transform.rs:315,436`), which the raw floor above
  ///   cannot see — the `f32` cast of `mean1` has raw norm `1.42`.
  ///
  /// So a row at `[0.005, 0.0, …]` is clustered into a speaker by ONLINE and
  /// fatal to OFFLINE; a row at `[f32::MAX, f32::MAX, 0.0, …]` clears PLDA's
  /// `f64` floor and normalizes to `None` for ONLINE — because `normalize_from`
  /// narrows the norm to `f32` and `4.8e38` does not fit; and `mean1` cast to
  /// `f32` clears BOTH admission functions and dies in the projection between
  /// them and the clustering. The conjunction is the standard exactly because
  /// each stage has corners the others do not.
  ///
  /// Every in-crate path already satisfies it:
  /// [`crate::audio::speaker::extract::Extractor::extract`] and the argmax
  /// source both DROP such a slot, zeroing its segmentation column at the same
  /// moment they leave its row zero — through the same
  /// `extract::raw_embedding_reaches_plda` this check applies, so the producers
  /// and the constructor cannot drift apart. Rejecting at assembly is what
  /// separates "this slot was deliberately dropped upstream" (column zeroed
  /// too) from "this slot's embedding is broken". See
  /// [`ActiveSlotWithoutEmbedding`].
  #[error(
    "chunk {} slot {} has an active segmentations column but its raw_embeddings \
     row cannot reach the clustering both backends run (non-finite, L2 norm below \
     PLDA's 0.01 floor, a norm the online engine's f32 narrowing turns into inf, \
     or a centered norm PLDA's projection refuses)",
    .0.chunk(),
    .0.slot()
  )]
  ActiveSlotWithoutEmbedding(ActiveSlotWithoutEmbedding),
  /// The one shared [`diaric::plda::PldaTransform`] that
  /// `extract::raw_embedding_reaches_plda` validates a row's PROJECTION against
  /// could not be built.
  ///
  /// `PldaTransform::new()` is declared fallible; it takes no arguments and
  /// decodes weight blobs `include_bytes!`d into the binary
  /// (`diarization/src/plda/loader.rs:17-36`), so today it has no failing path
  /// and `plda_transform_is_available` pins that. It is surfaced anyway, and as
  /// a TYPED refusal rather than a weaker check: without the transform the row
  /// standard cannot be applied, and the two alternatives are both silent —
  /// dropping the projection clause would admit rows that fail the whole
  /// offline extraction, and treating every row as unusable would make
  /// [`crate::audio::speaker::extract::Extractor::extract`] zero every
  /// segmentation column and return an empty diarization. A validator that
  /// quietly degrades when its own inputs are missing is the defect class this
  /// predicate exists to close.
  ///
  /// Unit-shaped deliberately: `diaric::plda::Error` is not `Clone`, and the
  /// cached [`std::sync::OnceLock`] cannot hand the same error out twice.
  #[error(
    "diaric's PLDA transform could not be built, so an active slot's raw \
     embedding cannot be validated against the projection the offline backend runs"
  )]
  PldaTransformUnavailable,
  /// A `count[t]` is not the value the `segmentations` supplied beside it
  /// derive at output frame `t`. BOTH directions are rejected, because both
  /// make the two backends disagree about the same `Extraction`.
  ///
  /// Above the derived value: `diaric`'s offline reconstruction treats `count[t]`
  /// as an INJECTIVE per-cluster count — it pads the cluster axis out to
  /// `max(count)` and marks exactly `count[t]` columns active by descending
  /// activation (`diarization/src/reconstruct/algo.rs:736-810`), with no lower
  /// bound on the activation it will select — so an inflated `count[t]` selects
  /// zero-activation padded columns and emits that many phantom speakers, as
  /// `Ok`. A range check does not catch it: `count = [3, 3]` is inside both
  /// `SEG_NUM_SLOTS` and `diaric::reconstruct::MAX_COUNT_PER_FRAME` and still
  /// fabricates two speakers on a grid with one active slot.
  ///
  /// Below it: the OFFLINE route consumes the supplied `count` while the ONLINE
  /// route ignores it and derives its own from the same segmentations
  /// ([`crate::audio::speaker::extract::Extraction::diarize_online`]'s own
  /// comment says why it must). An all-zero `count` over an active grid is
  /// therefore silence offline and a speaker online — the same `Extraction`,
  /// contradictory answers. Bounding one direction cannot prevent that; only
  /// equality can.
  ///
  /// The derived value is the caller's own data run through the overlap-add
  /// aggregation [`crate::audio::speaker::window::count_from_segmentations`]
  /// runs, over `seg > 0.0` — the activity predicate BOTH backends apply to a
  /// segmentation column, and the one
  /// [`crate::audio::speaker::extract::Extractor::extract`]'s own `count`
  /// satisfies (it aggregates `seg >= onset` over a hard `0.0`/`1.0`
  /// multilabel, on which the two predicates coincide for every `onset` in
  /// `(0.0, 1.0]`). See [`CountNotSegmentationDerived`].
  #[error(
    "extraction parts: count[{}] is {} but the supplied segmentations derive {} at that output \
     frame",
    .0.frame(),
    .0.got(),
    .0.expected()
  )]
  CountNotSegmentationDerived(CountNotSegmentationDerived),
  /// The output-frame grid the geometry derives is above
  /// [`crate::audio::speaker::extract::MAX_OUTPUT_FRAMES`].
  ///
  /// A RESOURCE bound, not a consistency invariant: the grid is internally
  /// consistent, it is simply larger than this crate is willing to build
  /// scratch buffers for. See
  /// [`crate::audio::speaker::extract::MAX_OUTPUT_FRAMES`] for the budget the
  /// number encodes and why a cap was chosen over fallible allocation.
  ///
  /// Carries the derived frame count that was refused; the ceiling is the public
  /// constant.
  #[error(
    "extraction parts: the declared geometry derives {} output frames, above the \
     MAX_OUTPUT_FRAMES cap ({})",
    .0,
    crate::audio::speaker::extract::MAX_OUTPUT_FRAMES
  )]
  OutputFrameCountTooLarge(usize),
  /// The derived `num_output_frames` would not fit in `usize`. Converted
  /// from [`crate::audio::speaker::window`]'s crate-private `WindowError` by an exhaustive
  /// manual match in [`crate::audio::speaker::extract::Extractor::extract`] (deliberately
  /// NOT a `From` impl — a `From` would put a crate-private type into a
  /// public trait impl and add a second conversion surface for a single
  /// call site; the exhaustive match forces revisiting this if
  /// `WindowError` ever grows variants). Unreachable through `extract`'s
  /// own geometry (`num_chunks * step_samples ≈ samples.len() <=
  /// isize::MAX/4`, so `num_output_frames` stays far below `usize::MAX`),
  /// but kept typed per this crate's no-panic-on-untrusted-config posture.
  /// Message text mirrors `WindowError::OutputFrameCountOverflow`'s
  /// display and dia's `ShapeError::OutputFrameCountOverflow`
  /// (`diarization/src/aggregate/count.rs:114-117`).
  ///
  /// REACHABLE, by contrast, through
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`], whose
  /// caller picks both sliding windows outright: a finite `chunks_sw.duration()
  /// = 1e300` over a finite `frames_sw.step() = 1e-300` divides to `+inf`. That
  /// constructor raises this variant for exactly the geometry
  /// [`crate::audio::speaker::extract::Extraction::diarize_online`] would later
  /// re-derive, which is what lets that method keep its `.expect(..)`.
  #[error(
    "num_output_frames overflows usize (chunk_duration / frame_step too large \
     to represent or saturated past usize::MAX)"
  )]
  OutputFrameCountOverflow,
}

#[cfg(test)]
mod tests;
