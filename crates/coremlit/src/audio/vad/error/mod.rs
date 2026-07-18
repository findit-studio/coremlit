//! Structured, per-domain error types for the vadkit model layer (design
//! spec §4). Foreign errors from [`coremlit`] are wrapped as typed `#[from]`
//! variants, mirroring `speakerkit::error`.

/// Failure locating, loading, or validating the CoreML VAD model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] coremlit::LoadError),
  /// A loaded model's input or output feature does not match the shape/dtype
  /// contract this crate was built against — the exact contract pinned from
  /// the artifact's `metadata.json` (see `tests/model_io.rs` for the ground
  /// truth and per-file SHA-256).
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

/// Failure running or interpreting one VAD inference call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InferError {
  /// The CoreML runtime failed to run the model.
  #[error("prediction failed: {0}")]
  Prediction(#[from] coremlit::PredictionError),
  /// A tensor failed to construct or view.
  #[error("tensor failed: {0}")]
  Tensor(#[from] coremlit::TensorError),
  /// The caller's chunk was longer than one model chunk
  /// ([`crate::CHUNK_SAMPLES`]). Short chunks are padded (FluidAudio
  /// repeat-last semantics, `VadManager.swift:173-182`); over-long ones are
  /// rejected rather than silently truncated, because a caller feeding more
  /// than one 256 ms window per call has a chunking bug this crate cannot
  /// paper over — the discarded tail would be dropped speech.
  #[error("chunk length {got} exceeds one model chunk ({max})")]
  ChunkTooLong {
    /// Samples the caller provided.
    got: usize,
    /// The one-chunk maximum ([`crate::CHUNK_SAMPLES`]).
    max: usize,
  },
  /// The caller's chunk contained a NaN or infinite sample before inference
  /// ran — the exact `ort` CoreML-EP corruption mode the CoreML backends
  /// exist to replace. A NaN sample would otherwise reach CoreML and can be
  /// absorbed into a finite-looking but garbage probability no downstream
  /// check would catch (mirrors `speakerkit::error::InferError::NonFiniteInput`).
  #[error("input contains a non-finite value at index {index}")]
  NonFiniteInput {
    /// Flat index of the offending sample within the assembled model window.
    index: usize,
  },
  /// A predict-time output tensor's shape diverged from the contract
  /// validated once at construction. The CoreML runtime is a trust boundary
  /// independent of its declared metadata, so every prediction's output
  /// shapes are re-checked (mirrors
  /// `speakerkit::error::InferError::OutputShape`).
  #[error("output `{feature}` shape mismatch: expected {expected:?}, got {got:?}")]
  OutputShape {
    /// The output feature whose runtime shape diverged.
    feature: &'static str,
    /// Shape the runtime tensor actually had.
    got: Vec<usize>,
    /// Shape the construction-time contract declares.
    expected: Vec<usize>,
  },
  /// The model's probability or a recurrent-state element came back NaN or
  /// infinite. The VAD graph's output is a noisy-OR of sigmoids (bounded in
  /// `[0, 1]`) and its LSTM state is finite by construction, so a non-finite
  /// value is the CoreML-EP corruption mode this crate exists to replace, not
  /// a valid result (mirrors `speakerkit::error::InferError::NonFiniteOutput`).
  #[error("output `{feature}` contains a non-finite value at index {index}")]
  NonFiniteOutput {
    /// The output feature that carried the non-finite value.
    feature: &'static str,
    /// Flat index of the offending element within that output.
    index: usize,
  },
}

#[cfg(test)]
mod tests;
