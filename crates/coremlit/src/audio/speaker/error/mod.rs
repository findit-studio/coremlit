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
  Load(#[from] coremlit::LoadError),
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
  Prediction(#[from] coremlit::PredictionError),
  /// A tensor failed to construct or view.
  #[error("tensor failed: {0}")]
  Tensor(#[from] coremlit::TensorError),
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
  /// validated at construction. `coremlit::MultiArray::copy_into` alone
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

/// Top-level extraction failure, composing model-lifecycle and inference
/// errors (spec §5) plus [`crate::extract::Extractor::extract`]'s own
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
  /// the chunk planner's `div_ceil`. [`crate::window::WindowOptions`]'s
  /// own builders already reject this, so reaching it means a
  /// serde-deserialized config bypassed the builder — defense-in-depth,
  /// exactly as dia re-checks it here despite `with_step_samples`'s panic.
  #[error("step_samples must be > 0")]
  ZeroStepSamples,
  /// The configured `step_samples` exceeds [`crate::segment::SEG_CHUNK_SAMPLES`].
  /// Mirrors dia's `ShapeError::StepSamplesExceedsWindow`
  /// (`diarization/src/offline/owned.rs:377-387`, whose own comment gives
  /// the serde-bypass defense-in-depth rationale): with `step > window`,
  /// samples in `[window .. step)` per chunk are never segmented or
  /// embedded — silent data loss returning `Ok(_)` with missing speech.
  #[error("step_samples ({step}) must not exceed SEG_CHUNK_SAMPLES ({window})")]
  StepSamplesExceedsWindow {
    /// The rejected `step_samples`.
    step: u32,
    /// The chunk window length ([`crate::segment::SEG_CHUNK_SAMPLES`]).
    window: usize,
  },
  /// The configured `onset` is not finite in `(0.0, 1.0]`. Mirrors dia's
  /// `ShapeError::OnsetOutOfRange`
  /// (`diarization/src/offline/owned.rs:388-393`) and
  /// [`crate::window`]'s `check_onset` `(0.0, 1.0]` contract: the hard
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
  /// Raised only by [`crate::source::ArgmaxSource`]: argmax's segmenter
  /// slides its 21 windows internally at a fixed
  /// [`crate::source::argmax::ARGMAX_WINDOW_STRIDE_SAMPLES`] (16 000 = 1 s,
  /// derived from the graph's own `[21, 1, 160000]` output shape), so there
  /// is no knob to vary. [`crate::extract::Extractor`]'s host-side chunk
  /// planner has no such constraint and accepts any `step_samples` in
  /// `(0, SEG_CHUNK_SAMPLES]`.
  ///
  /// Rejected rather than ignored: silently overriding the caller's
  /// `step_samples` would return an `Extraction` whose `chunks_sw.step()`
  /// did not describe its own chunk grid, corrupting every downstream time
  /// offset `dia` reconstructs from it.
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
  /// ([`crate::segment::SegmentModel::num_frames`],
  /// [`crate::embed::EmbedModel::num_mask_frames`]); a mismatch would
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
  /// The derived `num_output_frames` would not fit in `usize`. Converted
  /// from [`crate::window`]'s crate-private `WindowError` by an exhaustive
  /// manual match in [`crate::extract::Extractor::extract`] (deliberately
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
  #[error(
    "num_output_frames overflows usize (chunk_duration / frame_step too large \
     to represent or saturated past usize::MAX)"
  )]
  OutputFrameCountOverflow,
}

#[cfg(test)]
mod tests;
