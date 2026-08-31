//! Structured, per-domain error types for the vadkit model layer (design
//! spec §4). Foreign errors from [`crate`] are wrapped as typed `#[from]`
//! variants, mirroring `coremlit::audio::speaker::error`.

/// A loaded model's input or output feature does not match the shape/dtype
/// contract this crate was built against — the exact contract pinned from
/// the artifact's `metadata.json` (see `tests/model_io.rs` for the ground
/// truth and per-file SHA-256).
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

/// Failure locating, loading, or validating the CoreML VAD model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
  /// The CoreML runtime failed to load the compiled model.
  #[error("failed to load model: {0}")]
  Load(#[from] crate::LoadError),
  /// A loaded model's input or output feature does not match the shape/dtype
  /// contract this crate was built against — the exact contract pinned from
  /// the artifact's `metadata.json` (see `tests/model_io.rs` for the ground
  /// truth and per-file SHA-256).
  #[error(
    "model contract mismatch on `{}`: expected {}, got {}",
    .0.feature(),
    .0.expected(),
    .0.actual()
  )]
  ContractMismatch(ContractMismatch),
}

/// The caller's chunk was longer than one model chunk
/// ([`crate::audio::vad::CHUNK_SAMPLES`]).
///
/// Payload of [`InferError::ChunkTooLong`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkTooLong {
  /// Samples the caller provided.
  got: usize,
  /// The one-chunk maximum ([`crate::audio::vad::CHUNK_SAMPLES`]).
  max: usize,
}

impl ChunkTooLong {
  /// Construct from the samples the caller provided and the one-chunk maximum.
  #[inline(always)]
  pub const fn new(got: usize, max: usize) -> Self {
    Self { got, max }
  }

  /// Samples the caller provided.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The one-chunk maximum ([`crate::audio::vad::CHUNK_SAMPLES`]).
  #[inline(always)]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// A predict-time output tensor's shape diverged from the contract validated
/// once at construction.
///
/// Payload of [`InferError::OutputShape`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputShape {
  /// The output feature whose runtime shape diverged.
  feature: &'static str,
  /// Shape the runtime tensor actually had.
  got: Vec<usize>,
  /// Shape the construction-time contract declares.
  expected: Vec<usize>,
}

impl OutputShape {
  /// Construct from the diverging feature, its runtime shape, and the shape
  /// the construction-time contract declares.
  #[inline(always)]
  pub const fn new(feature: &'static str, got: Vec<usize>, expected: Vec<usize>) -> Self {
    Self {
      feature,
      got,
      expected,
    }
  }

  /// The output feature whose runtime shape diverged.
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
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

/// The model's probability or a recurrent-state element came back NaN or
/// infinite.
///
/// Payload of [`InferError::NonFiniteOutput`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonFiniteOutput {
  /// The output feature that carried the non-finite value.
  feature: &'static str,
  /// Flat index of the offending element within that output.
  index: usize,
}

impl NonFiniteOutput {
  /// Construct from the output feature and the flat index of the offending
  /// element within it.
  #[inline(always)]
  pub const fn new(feature: &'static str, index: usize) -> Self {
    Self { feature, index }
  }

  /// The output feature that carried the non-finite value.
  #[inline(always)]
  pub const fn feature(&self) -> &'static str {
    self.feature
  }

  /// Flat index of the offending element within that output.
  #[inline(always)]
  pub const fn index(&self) -> usize {
    self.index
  }
}

/// Failure running or interpreting one VAD inference call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InferError {
  /// The CoreML runtime failed to run the model.
  #[error("prediction failed: {0}")]
  Prediction(#[from] crate::PredictionError),
  /// A tensor failed to construct or view.
  #[error("tensor failed: {0}")]
  Tensor(#[from] crate::TensorError),
  /// The caller's chunk was longer than one model chunk
  /// ([`crate::audio::vad::CHUNK_SAMPLES`]). Short chunks are padded (FluidAudio
  /// repeat-last semantics, `VadManager.swift:173-182`); over-long ones are
  /// rejected rather than silently truncated, because a caller feeding more
  /// than one 256 ms window per call has a chunking bug this crate cannot
  /// paper over — the discarded tail would be dropped speech.
  #[error("chunk length {} exceeds one model chunk ({})", .0.got(), .0.max())]
  ChunkTooLong(ChunkTooLong),
  /// The caller's chunk contained a NaN or infinite sample before inference
  /// ran — the exact `ort` CoreML-EP corruption mode the CoreML backends
  /// exist to replace. A NaN sample would otherwise reach CoreML and can be
  /// absorbed into a finite-looking but garbage probability no downstream
  /// check would catch (mirrors `coremlit::audio::speaker::error::InferError::NonFiniteInput`).
  ///
  /// Carries the flat index of the offending sample within the assembled
  /// model window.
  #[error("input contains a non-finite value at index {0}")]
  NonFiniteInput(usize),
  /// A predict-time output tensor's shape diverged from the contract
  /// validated once at construction. The CoreML runtime is a trust boundary
  /// independent of its declared metadata, so every prediction's output
  /// shapes are re-checked (mirrors
  /// `coremlit::audio::speaker::error::InferError::OutputShape`).
  #[error(
    "output `{}` shape mismatch: expected {:?}, got {:?}",
    .0.feature(),
    .0.expected(),
    .0.got()
  )]
  OutputShape(OutputShape),
  /// The model's probability or a recurrent-state element came back NaN or
  /// infinite. The VAD graph's output is a noisy-OR of sigmoids (bounded in
  /// `[0, 1]`) and its LSTM state is finite by construction, so a non-finite
  /// value is the CoreML-EP corruption mode this crate exists to replace, not
  /// a valid result (mirrors `coremlit::audio::speaker::error::InferError::NonFiniteOutput`).
  #[error(
    "output `{}` contains a non-finite value at index {}",
    .0.feature(),
    .0.index()
  )]
  NonFiniteOutput(NonFiniteOutput),
}

#[cfg(test)]
mod tests;
