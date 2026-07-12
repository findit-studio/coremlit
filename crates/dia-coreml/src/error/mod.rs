//! Structured, per-domain error types for the `dia-coreml` backends (design
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
/// errors (spec §5).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ExtractError {
  /// A model failed to load, or its contract mismatched.
  #[error("model error: {0}")]
  Model(#[from] ModelError),
  /// An inference call failed.
  #[error("infer error: {0}")]
  Infer(#[from] InferError),
}

#[cfg(test)]
mod tests;
