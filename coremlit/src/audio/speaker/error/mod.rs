//! Structured, per-domain error types for the `speakerkit` backends (design
//! spec §5). Foreign errors from `coremlit` are wrapped as typed `#[from]`
//! variants; [`ExtractError`] composes both domain errors at the top level.

/// A loaded model's input or output feature does not match the
/// shape/dtype contract this crate was built against (see
/// `tests/model_io.rs` for the pinned ground truth).
///
/// Payload of [`ModelError::ContractMismatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractMismatch {
  /// Name of the input/output feature that mismatched.
  feature: &'static str,
  /// The contract this crate expects, rendered for display.
  expected: String,
  /// What the loaded model actually declares, rendered for display.
  actual: String,
}

impl ContractMismatch {
  /// Construct from the mismatched feature, the expected contract, and what
  /// the loaded model actually declares.
  #[inline(always)]
  pub const fn new(feature: &'static str, expected: String, actual: String) -> Self {
    Self {
      feature,
      expected,
      actual,
    }
  }

  /// Name of the input/output feature that mismatched.
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }
  /// The contract this crate expects, rendered for display.
  #[inline(always)]
  pub fn expected(&self) -> &str {
    &self.expected
  }
  /// What the loaded model actually declares, rendered for display.
  #[inline(always)]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

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
  #[error(
    "model contract mismatch on `{}`: expected {}, got {}",
    .0.feature(),
    .0.expected(),
    .0.actual()
  )]
  ContractMismatch(ContractMismatch),
}

/// The caller's input slice did not have the model's required length.
///
/// Payload of [`InferError::InputLength`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputLength {
  /// Elements the caller provided.
  got: usize,
  /// Elements the model requires.
  expected: usize,
}

impl InputLength {
  /// Construct from the element count the caller provided and the count the
  /// model requires.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// Elements the caller provided.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// Elements the model requires.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// A predict-time output tensor's shape diverged from the contract
/// validated at construction. `crate::MultiArray::copy_into` alone
/// only validates total element count, so an axes-swapped output (e.g.
/// `[1, classes, frames]` instead of `[1, frames, classes]`) can carry
/// the same element count as the expected shape and would otherwise pass
/// silently, transposing two axes instead of erroring.
///
/// Payload of [`InferError::OutputShape`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputShape {
  /// Shape the runtime tensor actually had.
  got: Vec<usize>,
  /// Shape the construction-time contract declares.
  expected: Vec<usize>,
}

impl OutputShape {
  /// Construct from the shape the runtime tensor had and the shape the
  /// construction-time contract declares.
  #[inline(always)]
  pub const fn new(got: Vec<usize>, expected: Vec<usize>) -> Self {
    Self { got, expected }
  }

  /// Shape the runtime tensor actually had.
  #[inline(always)]
  pub fn got(&self) -> &[usize] {
    &self.got
  }

  /// Shape the construction-time contract declares.
  #[inline(always)]
  pub fn expected(&self) -> &[usize] {
    &self.expected
  }
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
  /// Carries the flat index of the offending element.
  #[error("output contains a non-finite value at index {0}")]
  NonFiniteOutput(usize),
  /// The caller's input slice did not have the model's required length.
  /// See [`InputLength`] for the two element counts.
  #[error("input length mismatch: expected {}, got {}", .0.expected(), .0.got())]
  InputLength(InputLength),
  /// A predict-time output tensor's shape diverged from the contract
  /// validated at construction. `crate::MultiArray::copy_into` alone
  /// only validates total element count, so an axes-swapped output (e.g.
  /// `[1, classes, frames]` instead of `[1, frames, classes]`) can carry
  /// the same element count as the expected shape and would otherwise pass
  /// silently, transposing two axes instead of erroring.
  /// See [`OutputShape`] for the two shapes.
  #[error("output shape mismatch: expected {:?}, got {:?}", .0.expected(), .0.got())]
  OutputShape(OutputShape),
  /// The caller's input contained a NaN or infinite value before inference
  /// ran. Complements [`Self::NonFiniteOutput`]: an unchecked NaN sample
  /// can otherwise propagate silently into a finite-looking but garbage
  /// embedding that no downstream check would catch. Mirrors dia's
  /// analogous embed-side guard, `embed::Error::NonFiniteInput`
  /// (`diarization/src/embed/error.rs:107-109`) — a unit variant there.
  /// This variant adds the offending flat index, matching this crate's own
  /// [`Self::NonFiniteOutput`] shape: a deliberate enhancement over dia's,
  /// not a parity requirement (dia's own variant carries no index).
  /// Carries the flat index of the offending element.
  #[error("input contains a non-finite value at index {0}")]
  NonFiniteInput(usize),
  /// The caller's input was finite in `f32` but its magnitude exceeds `f16`'s
  /// finite range (`|x| > f16::MAX`, i.e. `65504`), so narrowing it to the
  /// argmax segmenter's `.float16` `waveform` input would round it to an f16
  /// infinity and reach CoreML as a non-finite value — the very thing
  /// [`Self::NonFiniteInput`] exists to prevent, one representability step in.
  /// Only the argmax source narrows host `f32` samples to `f16` before
  /// inference; the FluidAudio and dia-coreml paths feed `f32` unchanged, so
  /// this guard is scoped to that source's `extract` (the public contract
  /// places no amplitude bound on `samples`, `source/mod.rs`).
  /// Carries the flat index of the offending element.
  #[error(
    "input value at index {0} is finite in f32 but overflows the model's f16 input \
     domain (|x| > f16::MAX)"
  )]
  F16OverflowInput(usize),
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

/// The size a declared geometry requires overflows `usize`, so NO slice can
/// satisfy it and no expected length can even be named.
///
/// Payload of [`ExtractError::ExtractionGeometryOverflow`]. Reported separately
/// from [`ExtractionLenMismatch`] because the two are different diagnoses: a
/// mismatch means "this tensor is the wrong size", an overflow means "these
/// dimensions are not a describable tensor at all". Mirrors `diaric`'s own split
/// between `ShapeError::RawEmbeddingsOverflow` /
/// `ShapeError::SegmentationsOverflow` and their `*LenMismatch` counterparts
/// (`diarization/src/offline/algo.rs:575-588`).
///
/// Raised from two places, over the same dimensions in two units.
/// [`ExtractError::ExtractionTensorBytesTooLarge`]'s preflight computes the same
/// products in BYTES, so it also covers an element count that fits `usize` while
/// its byte size does not — a `Vec` of which is unallocatable regardless. Both
/// check `raw_embeddings` before `segmentations`, so a geometry that overflows
/// both names the same [`ExtractionPart`] either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtractionGeometryOverflow {
  part: ExtractionPart,
  num_chunks: usize,
  num_frames_per_chunk: usize,
}

impl ExtractionGeometryOverflow {
  /// Whose product overflowed — [`ExtractionPart::RawEmbeddings`] or
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

  /// Crate-private: only the validating constructor and the producers'
  /// geometry preflight raise this.
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

/// A `segmentations` cell outside the hard-binary `{0.0, 1.0}` domain.
///
/// Payload of [`ExtractError::NonBinarySegmentation`]. Carries the FLAT
/// `[c][f][s]` index and the value found there. The index alone fixes the slot
/// (`index % SEG_NUM_SLOTS`, since the slot axis is innermost and its extent is
/// the constant [`crate::audio::speaker::segment::SEG_NUM_SLOTS`]); recovering
/// the chunk and frame needs the caller's own `num_frames_per_chunk`, which sits
/// in the same [`crate::audio::speaker::extract::ExtractionParts`] —
/// `chunk = index / (num_frames_per_chunk * SEG_NUM_SLOTS)`,
/// `frame = (index / SEG_NUM_SLOTS) % num_frames_per_chunk`.
///
/// No `Eq`/`Hash`: the value is an `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonBinarySegmentation {
  index: usize,
  value: f64,
}

impl NonBinarySegmentation {
  /// Flat index into the `[c][f][s]` `segmentations` buffer. The FIRST
  /// offending cell: the scan stops at the earliest one.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }
  /// The value found there — neither `0.0` nor `1.0`.
  #[inline(always)]
  pub const fn value(&self) -> f64 {
    self.value
  }
  /// The speaker slot the offending cell belongs to, in `0..SEG_NUM_SLOTS`.
  #[inline(always)]
  pub const fn slot(&self) -> usize {
    self.index % crate::audio::speaker::segment::SEG_NUM_SLOTS
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(index: usize, value: f64) -> Self {
    Self { index, value }
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

/// The first output frame whose CENTER is not a strictly larger, finite time
/// than its predecessor's.
///
/// Payload of [`ExtractError::CollapsedFrameCenter`]. Frame `t`'s center is
/// `frames_sw.start + t * frames_sw.step + frames_sw.duration / 2` — the
/// expression `diaric`'s span conversion evaluates for every span endpoint
/// (`diarization/src/reconstruct/rttm.rs:216-217,231-232`), mirrored by
/// [`crate::audio::speaker::window`]'s `frame_center`. A window whose three
/// fields are each finite and positive can still collapse it: at
/// `start = 1e9` the `f64` ULP is `1.19e-7`, so a `step` of `1e-8` adds
/// nothing at all and consecutive frames share one center. The active run then
/// closes at endpoints that are equal, and the backend returns `Ok` with a span
/// of duration zero.
///
/// Carries the offending frame, its center, and the previous frame's center.
/// For frame `0` — which has no predecessor and can only fail by being
/// non-finite — `previous` is `f64::NEG_INFINITY`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollapsedFrameCenter {
  frame: usize,
  center: f64,
  previous: f64,
}

impl CollapsedFrameCenter {
  /// Index of the first output frame whose center is not finite, or not
  /// strictly greater than the previous frame's.
  #[inline(always)]
  pub const fn frame(&self) -> usize {
    self.frame
  }
  /// That frame's center, in seconds.
  #[inline(always)]
  pub const fn center(&self) -> f64 {
    self.center
  }
  /// The previous frame's center, in seconds — `f64::NEG_INFINITY` when
  /// [`Self::frame`] is `0`.
  #[inline(always)]
  pub const fn previous(&self) -> f64 {
    self.previous
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(frame: usize, center: f64, previous: f64) -> Self {
    Self {
      frame,
      center,
      previous,
    }
  }
}

/// The derived output-frame grid is too short to hold the LAST chunk's frames,
/// so `diaric::reconstruct` would drop every cell past its end.
///
/// Payload of [`ExtractError::UncoveredLastChunk`]. `diaric::reconstruct` writes
/// chunk `c`'s frame `f` at output frame `closest_frame(chunks_sw.start + c *
/// chunks_sw.step + frames_sw.duration / 2) + f` and requires the grid to reach
/// the last chunk's last frame, raising its own
/// `ShapeError::OutputFrameCountTooSmall { got, required }`
/// (`diarization/src/reconstruct/algo.rs:478-495`) when it does not. This
/// carries the same two numbers, plus the placement they are derived from.
///
/// [`Self::required`] is [`Self::start_frame`] `+ num_frames_per_chunk`, where
/// the placement comes from [`crate::audio::speaker::window`]'s
/// `reconstruct_chunk_start_frame` — the mirror of that `closest_frame`, and
/// the SAME quantity `diaric`'s `required` is built from. [`Self::got`] is the
/// grid the geometry derives, which past
/// [`ExtractError::ExtractionLenMismatch`]'s check-5 equality IS `count.len()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UncoveredLastChunk {
  start_frame: i64,
  required: usize,
  got: usize,
}

impl UncoveredLastChunk {
  /// The output frame `diaric::reconstruct` places the LAST chunk's first frame
  /// at. Non-negative: the check does not fire below zero, matching `diaric`'s
  /// own `last_start_frame >= 0` guard.
  #[inline(always)]
  pub const fn start_frame(&self) -> i64 {
    self.start_frame
  }
  /// The grid length that placement demands: [`Self::start_frame`] `+
  /// num_frames_per_chunk`, saturating.
  #[inline(always)]
  pub const fn required(&self) -> usize {
    self.required
  }
  /// The grid length the geometry actually derives.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// Crate-private: only the validating constructor raises this.
  pub(crate) const fn new(start_frame: i64, required: usize, got: usize) -> Self {
    Self {
      start_frame,
      required,
      got,
    }
  }
}

/// The configured `step_samples` exceeds
/// [`crate::audio::speaker::segment::SEG_CHUNK_SAMPLES`], so samples in
/// `[window .. step)` per chunk would never be segmented or embedded.
///
/// Payload of [`ExtractError::StepSamplesExceedsWindow`].
#[derive(Debug, Clone, PartialEq)]
pub struct StepSamplesExceedsWindow {
  /// The rejected `step_samples`.
  step: u32,
  /// The chunk window length ([`crate::audio::speaker::segment::SEG_CHUNK_SAMPLES`]).
  window: usize,
}

impl StepSamplesExceedsWindow {
  /// Construct from the rejected `step_samples` and the chunk window length
  /// it exceeded.
  #[inline(always)]
  pub const fn new(step: u32, window: usize) -> Self {
    Self { step, window }
  }

  /// The rejected `step_samples`.
  #[inline(always)]
  pub const fn step(&self) -> u32 {
    self.step
  }

  /// The chunk window length ([`crate::audio::speaker::segment::SEG_CHUNK_SAMPLES`]).
  #[inline(always)]
  pub const fn window(&self) -> usize {
    self.window
  }
}

/// The configured `step_samples` is one the selected source cannot honor
/// because its sliding-window stride is compiled INTO the model graph.
///
/// Payload of [`ExtractError::UnsupportedStepSamples`].
#[derive(Debug, Clone, PartialEq)]
pub struct UnsupportedStepSamples {
  /// The rejected `step_samples`.
  step: u32,
  /// The stride the source's graph requires.
  required: u32,
}

impl UnsupportedStepSamples {
  /// Construct from the rejected `step_samples` and the stride the selected
  /// source's graph requires.
  #[inline(always)]
  pub const fn new(step: u32, required: u32) -> Self {
    Self { step, required }
  }

  /// The rejected `step_samples`.
  #[inline(always)]
  pub const fn step(&self) -> u32 {
    self.step
  }

  /// The stride the source's graph requires.
  #[inline(always)]
  pub const fn required(&self) -> u32 {
    self.required
  }
}

/// The segmentation model's per-chunk frame count disagrees with the
/// embedding model's mask frame count.
///
/// Payload of [`ExtractError::FrameCountMismatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct FrameCountMismatch {
  /// The segmentation model's per-chunk frame count.
  segmenter: usize,
  /// The embedding model's mask frame count.
  embedder: usize,
}

impl FrameCountMismatch {
  /// Construct from the segmentation model's per-chunk frame count and the
  /// embedding model's mask frame count.
  #[inline(always)]
  pub const fn new(segmenter: usize, embedder: usize) -> Self {
    Self {
      segmenter,
      embedder,
    }
  }

  /// The segmentation model's per-chunk frame count.
  #[inline(always)]
  pub const fn segmenter(&self) -> usize {
    self.segmenter
  }

  /// The embedding model's mask frame count.
  #[inline(always)]
  pub const fn embedder(&self) -> usize {
    self.embedder
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
  /// See [`StepSamplesExceedsWindow`] for the step and the window it exceeded.
  #[error(
    "step_samples ({}) must not exceed SEG_CHUNK_SAMPLES ({})",
    .0.step(),
    .0.window()
  )]
  StepSamplesExceedsWindow(StepSamplesExceedsWindow),
  /// The configured `onset` is not finite in `(0.0, 1.0]`. Mirrors dia's
  /// `ShapeError::OnsetOutOfRange`
  /// (`diarization/src/offline/owned.rs:388-393`) and
  /// [`crate::audio::speaker::window`]'s `check_onset` `(0.0, 1.0]` contract: the hard
  /// segmentation mask `seg >= onset` degenerates — `> 1.0`/NaN makes
  /// every frame inactive (empty diarization), `<= 0.0` makes every zero
  /// cell active (corrupted masks/counts).
  /// Carries the rejected `onset`.
  #[error("onset ({0}) must be finite in (0.0, 1.0]")]
  OnsetOutOfRange(f32),
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
  ///
  /// See [`UnsupportedStepSamples`] for the step and the stride required.
  #[error(
    "step_samples ({}) is not supported by this source: its window stride is fixed at \
     {} by the model graph",
    .0.step(),
    .0.required()
  )]
  UnsupportedStepSamples(UnsupportedStepSamples),
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
  /// See [`FrameCountMismatch`] for the two frame counts.
  #[error(
    "segmenter frame count ({}) does not match embedder mask frame count ({})",
    .0.segmenter(),
    .0.embedder()
  )]
  FrameCountMismatch(FrameCountMismatch),
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
  /// Raised by EVERY construction path, over the one shared
  /// `window::first_misaligned_chunk`:
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`] for parts a
  /// caller assembled, and every in-crate
  /// [`crate::audio::speaker::source::ModelSource`] for the grid its own
  /// `step_samples` and clip length derive.
  /// [`crate::audio::speaker::extract::Extractor::extract`] additionally runs it
  /// BEFORE it touches a model, since a geometry this crate cannot diarize
  /// honestly is not worth inferring over — a cost optimisation on top of the
  /// shared sequence, not the guarantee. Through `extract` it is reachable only for an ODD `step_samples`: the
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
  /// The output-frame grid the geometry derives has two frames at the same
  /// center time, or a center that is not finite.
  ///
  /// `frames_sw`'s three fields being finite and positive is NOT enough to make
  /// the grid a usable timeline. Every span either backend emits is a pair of
  /// frame CENTERS — `frames_sw.start + t * frames_sw.step +
  /// frames_sw.duration / 2`, evaluated by `diaric`'s span conversion at
  /// `diarization/src/reconstruct/rttm.rs:216-217,231-232` — and that sum
  /// rounds. Where `frames_sw.step` is small against the ULP of
  /// `frames_sw.start`, consecutive frames land on the identical `f64`:
  /// `start = 1e9, step = 1e-8` puts frames 0 and 1 both at exactly `1e9`
  /// (`1e9 + 1e-8 == 1e9`, the ULP there being `1.19e-7`). A one-frame active
  /// run then closes at `start == end` and the backend returns `Ok` with a span
  /// of DURATION ZERO — speech reported as an instant, which no consumer can
  /// act on and no other check sees.
  ///
  /// Raised by
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`] (whose
  /// caller picks `frames_sw` outright) and by every in-crate
  /// [`crate::audio::speaker::source::ModelSource`], which reach it through the
  /// same shared check. Unreachable from either source in practice: both use
  /// [`crate::audio::speaker::window::frame_sliding_window`]'s fixed
  /// `(0.0, 0.0619375, 0.016875)` grid, whose centers stay strictly increasing
  /// past [`crate::audio::speaker::extract::MAX_OUTPUT_FRAMES`].
  #[error(
    "extraction parts: output frame {}'s center ({:e}) must be finite and strictly later than \
     the previous frame's ({:e}) — this frames_sw collapses adjacent frame centers, so a span \
     closes at duration zero",
    .0.frame(),
    .0.center(),
    .0.previous()
  )]
  CollapsedFrameCenter(CollapsedFrameCenter),
  /// The derived output-frame grid does not reach the LAST chunk's last frame,
  /// so `diaric::reconstruct` silently drops every cell past its end — check 14.
  ///
  /// Checks 5 and 8 make the grid the derived one and place every chunk
  /// identically under both mappings, and neither says the grid is LONG ENOUGH:
  /// a chunk whose declared `chunks_sw.duration()` spans fewer
  /// `frames_sw.step()` slots than `num_frames_per_chunk` derives a grid
  /// shorter than the chunk it must hold. At the shipped grid that first bites
  /// at `num_frames_per_chunk = 595` for a one-chunk clip (594 output frames)
  /// and at `594` for a three-chunk clip (712 frames against a last chunk
  /// placed at 119).
  ///
  /// *Verified against both:* both routes end in the same
  /// `diaric::reconstruct`, which raises the typed
  /// `ShapeError::OutputFrameCountTooSmall` before allocating the grid
  /// (`diarization/src/reconstruct/algo.rs:478-495`). They differ in the WORK
  /// that precedes it, not in the outcome: ONLINE reaches that call directly,
  /// OFFLINE only at its stage 5, after AHC and VBx have already run
  /// (`diarization/src/offline/algo.rs:808`). Neither can return `Ok`, so this
  /// variant makes the refusal EARLY rather than newly-refusing anything —
  /// and, at both producers, before inference rather than after.
  ///
  /// Reachable from every construction path, and from both producers'
  /// pre-inference `checked_geometry`: `num_frames_per_chunk` is
  /// `SegmentModel`'s declared frame count, which that loader constrains only
  /// to `shape[1] >= 1`. Unreachable with the shipped 589-frame community-1
  /// segmenter, whose last chunk always leaves at least four frames of headroom
  /// on the default stride, and with
  /// [`crate::audio::speaker::source::ArgmaxSource`]'s compiled-in 589.
  #[error(
    "extraction geometry: the derived output grid is {} frames, but the last chunk starts at \
     frame {} and needs {} — diaric::reconstruct would drop every cell past the end",
    .0.got(),
    .0.start_frame(),
    .0.required()
  )]
  UncoveredLastChunk(UncoveredLastChunk),
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
  /// A `segmentations` cell is neither exactly `0.0` nor exactly `1.0`.
  ///
  /// The two backends do not read this tensor the same way once a cell leaves
  /// that domain, and the split is not in a corner — it changes the number of
  /// speakers:
  ///
  /// - Everything this crate's validator does with `segmentations` BOOLEANIZES
  ///   it at `seg > 0.0`: the active-slot scan behind
  ///   [`Self::ActiveSlotWithoutEmbedding`], and the count derivation behind
  ///   [`Self::CountNotSegmentationDerived`]. So does
  ///   [`crate::audio::speaker::extract::Extraction::diarize_online`], twice —
  ///   its per-slot `activity` frame count (which becomes the `f32` speech
  ///   duration the `min_speech_duration` gate reads) and its distinct-cluster
  ///   count.
  /// - `diaric`'s OFFLINE route instead SUMS the magnitudes. Its
  ///   `filter_embeddings` accumulates `clean_frames += segmentations[…]` over
  ///   singly-active frames and compares that sum against
  ///   `0.2 * num_frames_per_chunk` to pick the PLDA train subset
  ///   (`diarization/src/offline/algo.rs:644-679`); its stage-7 mask sums the
  ///   whole column and tests `sum_activity == 0.0`
  ///   (`diarization/src/pipeline/algo.rs:698-711`).
  ///
  /// On the hard-binary domain a magnitude sum IS the active-frame count and
  /// `sum == 0.0` IS "no active frame", so every one of those readings collapses
  /// onto the same boolean. Off it they diverge: four frames with a slot at
  /// `0.1` on two of them and a second slot at `0.1` on the other two sum to
  /// `0.2` each, below `0.2 * 4`, so offline trains on NEITHER and merges both
  /// into ONE speaker, while online sees two one-second slots at cosine
  /// distance 1 and emits TWO. Same [`crate::audio::speaker::extract::Extraction`],
  /// one span or two.
  ///
  /// Confining the input to `{0.0, 1.0}` is what removes the divergence, and it
  /// costs no capability that the SHIPPING models exercise.
  /// [`crate::audio::speaker::extract::Extractor::extract`] writes only
  /// `crate::audio::speaker::segment::multilabel`'s powerset table (whose rows
  /// are literal `0.0`/`1.0`) and zeroed columns — a compile-time constant, so
  /// for that source the domain is structural.
  /// [`crate::audio::speaker::source::ArgmaxSource`] writes the graph's
  /// `speaker_ids` VERBATIM, and there the hard-binary decode is a property of
  /// the shipped graph (pinned by the model-gated
  /// `argmax_decoded_output_value_semantics`) rather than of the code:
  /// `ArgmaxSource::from_dir_with` accepts a model on its I/O SHAPES. Round 8:
  /// that source therefore raises this variant itself, at its assembly door — a
  /// segmenter returning `0.1` per frame used to produce an extraction whose
  /// stored `count` was all zero (offline: silence) against 589 active frames
  /// read by the online route.
  ///
  /// Non-finite cells are refused here too, as a by-product of the same
  /// equality — but they are NOT a split, and this variant does not claim to
  /// close one. Both routes end in `diaric::reconstruct`, which scans the whole
  /// tensor and raises `NonFiniteField::Segmentations`
  /// (`diarization/src/reconstruct/algo.rs:497-508`); the offline route meets
  /// `diaric::pipeline::assign_embeddings`' own copy of that scan first
  /// (`diarization/src/pipeline/algo.rs:456-460`), so the two refuse with
  /// different typed variants and neither returns `Ok`. What this variant adds
  /// there is only the position: the cell is named at assembly rather than after
  /// a backend was chosen.
  ///
  /// See [`NonBinarySegmentation`] for the payload.
  #[error(
    "extraction parts: segmentations[{}] is {} (slot {}) — every cell must be exactly 0.0 or \
     1.0, because the offline backend sums these magnitudes where the online backend counts \
     nonzero frames",
    .0.index(),
    .0.value(),
    .0.slot()
  )]
  NonBinarySegmentation(NonBinarySegmentation),
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
  /// Unit-shaped because nothing else compiles: `diaric::plda::Error` derives
  /// only `Debug` + `Error` (`diarization/src/plda/error.rs:21`), while this
  /// enum derives `Clone` and `PartialEq`, so a newtype carrying it would break
  /// both. The cache makes the same point from the other side — the transform
  /// is built once behind a [`std::sync::OnceLock`], so a `diaric` error
  /// produced there could be moved out at most once, while this refusal has to
  /// be returnable on every later call. The REFUSAL therefore repeats; the
  /// CAUSE does not survive the first build, and a caller that needs it can
  /// call `diaric::plda::PldaTransform::new()` itself.
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
  /// A `raw_embeddings` value is NaN or `±inf`. Carries its flat index into
  /// that buffer.
  ///
  /// Raised by a scan of the WHOLE buffer, which is what separates it from
  /// [`Self::ActiveSlotWithoutEmbedding`]: that one holds the row of an ACTIVE
  /// `(chunk, slot)` to what both engines' row chains accept, and an INACTIVE
  /// slot's row never reaches it. Under an all-zero segmentation column the two
  /// backends then split, in the direction the row check cannot see:
  ///
  /// - OFFLINE, `diaric::pipeline::assign_embeddings` scans EVERY row of the
  ///   matrix — train subset or not, active or not, because its stage 6 cosine
  ///   scoring reads all of them — and returns `NonFiniteField::Embeddings`
  ///   (`diarization/src/pipeline/algo.rs:443-455`). The whole extraction fails.
  /// - ONLINE, [`crate::audio::speaker::extract::Extraction::diarize_online`]
  ///   skips an inactive column before it ever copies the row, so the value is
  ///   never read and the call returns `Ok`.
  ///
  /// So the identical [`crate::audio::speaker::extract::Extraction`] is fatal to
  /// one backend and silently fine for the other — the split this constructor
  /// exists to refuse, one buffer position away from the row check that already
  /// refuses its active-slot twin.
  ///
  /// Neither in-crate producer can emit one:
  /// [`crate::audio::speaker::extract::Extractor::extract`] and
  /// [`crate::audio::speaker::source::argmax::ArgmaxSource`] both start from an
  /// all-zero buffer (and `0.0` is finite) and write a row only when
  /// `extract::raw_embedding_reaches_plda` accepts it, which requires
  /// `diaric::plda::RawEmbedding::from_wespeaker` to return `Ok` and so cannot
  /// admit a non-finite value.
  ///
  /// Newtype, not struct-shaped: the flat index IS the whole diagnosis. It
  /// determines the `(chunk, slot)` pair — `chunk = i / (SEG_NUM_SLOTS *
  /// EMBEDDING_DIM)`, `slot = (i / EMBEDDING_DIM) % SEG_NUM_SLOTS`, both spelled
  /// out in the message — plus the lane within the row, which a `(chunk, slot)`
  /// payload would drop. Mirrors [`InferError::NonFiniteOutput`], which reports
  /// the same defect at the same granularity one stage upstream.
  #[error(
    "extraction parts: raw_embeddings[{}] is not finite (chunk {}, slot {}, dimension {}); the \
     offline backend rejects the whole matrix while the online backend never reads an inactive \
     slot's row",
    .0,
    .0 / (crate::audio::speaker::segment::SEG_NUM_SLOTS * crate::audio::speaker::embed::EMBEDDING_DIM),
    (.0 / crate::audio::speaker::embed::EMBEDDING_DIM) % crate::audio::speaker::segment::SEG_NUM_SLOTS,
    .0 % crate::audio::speaker::embed::EMBEDDING_DIM
  )]
  NonFiniteRawEmbedding(usize),
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
  /// The `segmentations` + `raw_embeddings` footprint the chunk grid derives is
  /// above [`crate::audio::speaker::extract::MAX_EXTRACTION_TENSOR_BYTES`].
  ///
  /// The sibling of [`Self::OutputFrameCountTooLarge`] on the OTHER axis, and
  /// the reason both exist: that one bounds the output grid, which is a function
  /// of the clip's duration, while the producers allocate on the chunk grid,
  /// which is `samples.len() / step_samples`. A ten-minute clip at
  /// `step_samples = 2` derives 4 720 001 chunks and 81 221 777 208 bytes of
  /// tensors while its output grid is 0.85 % of `MAX_OUTPUT_FRAMES` — so the
  /// frame cap passes it and this one refuses it.
  ///
  /// A RESOURCE bound, not a consistency invariant: the geometry is internally
  /// consistent and would produce a correct `Extraction` on a machine large
  /// enough. See
  /// [`crate::audio::speaker::extract::MAX_EXTRACTION_TENSOR_BYTES`] for how the
  /// ceiling is derived, why the byte count is bounded rather than the chunk
  /// count or the model-call count, and why
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`] does not
  /// raise this.
  ///
  /// Raised from GEOMETRY ALONE, before any tensor is allocated or any model is
  /// called. Carries the derived byte total that was refused; the ceiling is the
  /// public constant.
  #[error(
    "extraction geometry: the declared chunk grid derives {} bytes of extraction \
     tensors, above the MAX_EXTRACTION_TENSOR_BYTES cap ({})",
    .0,
    crate::audio::speaker::extract::MAX_EXTRACTION_TENSOR_BYTES
  )]
  ExtractionTensorBytesTooLarge(usize),
  /// The chunk grid the geometry derives is above
  /// [`crate::audio::speaker::extract::MAX_EXTRACTION_CHUNKS`].
  ///
  /// The COMPUTE sibling of [`Self::ExtractionTensorBytesTooLarge`], and the
  /// reason both exist: that one bounds the BYTES the two extraction tensors
  /// occupy, which the loaded segmenter's per-chunk frame count scales, and this
  /// one bounds the number of CHUNKS, which is what every producer's model-call
  /// count is proportional to. A segmenter emitting one frame per ten-second
  /// chunk costs 3 096 bytes per chunk, so the byte ceiling alone would admit
  /// 393 349 chunks — 786 698 CoreML calls for 59.17 s of audio, from 946 695
  /// samples at `step_samples = 2`. Cheap chunks are still chunks.
  ///
  /// A RESOURCE bound, not a consistency invariant: the geometry is internally
  /// consistent and would produce a correct `Extraction` given enough time. See
  /// [`crate::audio::speaker::extract::MAX_EXTRACTION_CHUNKS`] for how the
  /// ceiling is derived and why
  /// [`crate::audio::speaker::extract::Extraction::try_from_parts`] does not
  /// raise this.
  ///
  /// Raised from GEOMETRY ALONE, before any tensor is allocated or any model is
  /// called. Carries the derived chunk count that was refused; the ceiling is
  /// the public constant.
  #[error(
    "extraction geometry: the declared chunk grid is {} chunks, above the \
     MAX_EXTRACTION_CHUNKS cap ({})",
    .0,
    crate::audio::speaker::extract::MAX_EXTRACTION_CHUNKS
  )]
  ExtractionChunkCountTooLarge(usize),
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

/// The caller's raw embedding row did not have [`EMBEDDING_DIM`] elements.
///
/// Payload of [`CalibrateError::ProfileLength`].
///
/// [`EMBEDDING_DIM`]: crate::audio::speaker::embed::EMBEDDING_DIM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProfileLength {
  /// Elements the caller provided.
  got: usize,
  /// Elements a raw WeSpeaker row has.
  expected: usize,
}

impl ProfileLength {
  /// Construct from the provided and required element counts.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// Elements the caller provided.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// Elements a raw WeSpeaker row has.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

/// Two profiles prepared for different score sources were scored against each
/// other.
///
/// Payload of [`CalibrateError::ScoringMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScoringMismatch {
  /// The score source the side being normalized was prepared for.
  side: crate::audio::speaker::calibrate::Scoring,
  /// The score source the other profile was prepared for.
  other: crate::audio::speaker::calibrate::Scoring,
}

impl ScoringMismatch {
  /// Construct from the two disagreeing score sources.
  #[inline(always)]
  pub const fn new(
    side: crate::audio::speaker::calibrate::Scoring,
    other: crate::audio::speaker::calibrate::Scoring,
  ) -> Self {
    Self { side, other }
  }

  /// The score source the side being normalized was prepared for.
  #[inline(always)]
  pub const fn side(&self) -> crate::audio::speaker::calibrate::Scoring {
    self.side
  }

  /// The score source the other profile was prepared for.
  #[inline(always)]
  pub const fn other(&self) -> crate::audio::speaker::calibrate::Scoring {
    self.other
  }
}

/// A trial score and the two cohort statistics it was normalized against were
/// not all computed in the same score source.
///
/// Payload of [`CalibrateError::NormalizationMismatch`].
///
/// All three sources are carried, not merely the disagreeing pair. AS-Norm1
/// combines one score with two independently computed sides, so *which* of the
/// three is the odd one out is the whole diagnosis: a `PldaCosine` trial
/// against two `Cosine` sides is a caller who did not re-derive their
/// statistics, while one `Cosine` side among two `PldaCosine` values is a
/// stale cache entry. The pair-shaped [`ScoringMismatch`] cannot say which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormalizationMismatch {
  /// The score source the trial score was computed in.
  trial: crate::audio::speaker::calibrate::Scoring,
  /// The score source the enrolment side's statistics were computed in.
  enrolled: crate::audio::speaker::calibrate::Scoring,
  /// The score source the probe side's statistics were computed in.
  probe: crate::audio::speaker::calibrate::Scoring,
}

impl NormalizationMismatch {
  /// Construct from the trial score's source and the two sides'.
  #[inline(always)]
  pub const fn new(
    trial: crate::audio::speaker::calibrate::Scoring,
    enrolled: crate::audio::speaker::calibrate::Scoring,
    probe: crate::audio::speaker::calibrate::Scoring,
  ) -> Self {
    Self {
      trial,
      enrolled,
      probe,
    }
  }

  /// The score source the trial score was computed in.
  #[inline(always)]
  pub const fn trial(&self) -> crate::audio::speaker::calibrate::Scoring {
    self.trial
  }

  /// The score source the enrolment side's statistics were computed in.
  #[inline(always)]
  pub const fn enrolled(&self) -> crate::audio::speaker::calibrate::Scoring {
    self.enrolled
  }

  /// The score source the probe side's statistics were computed in.
  #[inline(always)]
  pub const fn probe(&self) -> crate::audio::speaker::calibrate::Scoring {
    self.probe
  }
}

/// Failure preparing a voice profile, scoring a trial, or deriving a side's
/// cohort statistics — [`crate::audio::speaker::calibrate`]'s error.
///
/// # Why this one is neither `Clone` nor `PartialEq`
///
/// Every other error in this module is both. This one wraps two `diaric`
/// errors — [`diaric::plda::Error`] and [`diaric::score_norm::Error`] — and
/// neither derives anything past `Debug` + `Error`. [`ExtractError`] met the
/// same wall and answered it by making
/// [`ExtractError::PldaTransformUnavailable`] UNIT-shaped, discarding the
/// cause to keep its own derives. That trade is right there and wrong here:
/// `PldaTransformUnavailable` has exactly one cause, while
/// `diaric::score_norm::Error` distinguishes a cohort that was too small from
/// one whose selected scores do not spread from an arithmetic refusal — the
/// three things a caller tuning [`AsNormOptions`] has to tell apart. Keeping
/// the payload and dropping the derives is the direction that keeps
/// information.
///
/// [`AsNormOptions`]: diaric::score_norm::AsNormOptions
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CalibrateError {
  /// The raw embedding row handed to
  /// [`Scoring::prepare`](crate::audio::speaker::calibrate::Scoring::prepare)
  /// was not [`EMBEDDING_DIM`] elements long.
  ///
  /// [`EMBEDDING_DIM`]: crate::audio::speaker::embed::EMBEDDING_DIM
  #[error(
    "voice profile: raw embedding row is {} elements, expected {}",
    .0.got(),
    .0.expected()
  )]
  ProfileLength(ProfileLength),

  /// The prepared vector has no usable direction, so a cosine against it would
  /// be noise rather than a similarity.
  ///
  /// Carries the score source that refused, because the two refuse for
  /// different reasons and neither number is this crate's to invent:
  ///
  /// - [`Scoring::Cosine`](crate::audio::speaker::calibrate::Scoring::Cosine)
  ///   refuses whatever [`diaric::embed::Embedding::normalize_from`] refuses —
  ///   a non-finite row, or an L2 norm under `diaric`'s own `NORM_EPSILON`.
  ///   This door CALLS that constructor rather than re-deriving its floor, the
  ///   same discipline [`crate::audio::speaker::extract::PLDA_MIN_NORM`]'s doc
  ///   describes.
  /// - [`Scoring::PldaCosine`](crate::audio::speaker::calibrate::Scoring::PldaCosine)
  ///   refuses a projected vector whose norm is zero or non-finite, or whose
  ///   normalization leaves the range. There is deliberately NO floor above
  ///   zero: `diaric` publishes none for the 128-d PLDA space, the WeSpeaker
  ///   `NORM_EPSILON` is calibrated for a different one, and a fabricated
  ///   constant would refuse real projections on a number nothing measured.
  #[error("voice profile: the prepared {0:?} vector has no usable direction")]
  DegenerateProfile(crate::audio::speaker::calibrate::Scoring),

  /// `diaric`'s PLDA projection refused the row.
  #[error("voice profile: plda: {0}")]
  Plda(#[from] diaric::plda::Error),

  /// The same refusal [`ExtractError::PldaTransformUnavailable`] carries, from
  /// the same process-wide cached transform: without it a
  /// [`Scoring::PldaCosine`](crate::audio::speaker::calibrate::Scoring::PldaCosine)
  /// profile cannot be projected at all.
  #[error(
    "diaric's PLDA transform could not be built, so a raw embedding cannot be \
     projected into the space `Scoring::PldaCosine` scores in"
  )]
  PldaTransformUnavailable,

  /// Two profiles prepared for different score sources were scored against
  /// each other.
  ///
  /// A tag comparison rather than a type error, and the trade is deliberate —
  /// see [`crate::audio::speaker::calibrate::VoiceProfile`]'s own docs. What
  /// matters is that it cannot be silent: an AS-Norm side is built from a
  /// whole cohort, so a mixed cohort would otherwise contribute scores from
  /// two different spaces to one mean.
  #[error(
    "voice profile: a {:?} profile cannot be scored against a {:?} one",
    .0.side(),
    .0.other()
  )]
  ScoringMismatch(ScoringMismatch),

  /// A trial score and the two sides handed to
  /// [`as_norm`](crate::audio::speaker::calibrate::as_norm) were not all
  /// computed in one score source.
  ///
  /// The refusal [`ScoringMismatch`] could not make, because the final
  /// combination step reads no profiles at all — it reads a number and two
  /// statistics. `Cosine` cohort scores of `[-1, 1]` have mean `0` and
  /// deviation `1`, so any finite `PldaCosine` trial score normalized against
  /// them comes back finite and plausible: one metric calibrated by another,
  /// with nothing out of range to notice.
  #[error(
    "AS-Norm: a {:?} trial score cannot be normalized by a {:?} enrolment side \
     and a {:?} probe side",
    .0.trial(),
    .0.enrolled(),
    .0.probe()
  )]
  NormalizationMismatch(NormalizationMismatch),

  /// `diaric`'s AS-Norm refused this side's cohort statistics.
  #[error("voice profile: {0}")]
  ScoreNorm(#[from] diaric::score_norm::Error),
}

#[cfg(test)]
mod tests;
