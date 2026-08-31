//! Compute-unit selection for model loading.

use objc2_core_ml::MLComputeUnits;

/// Which hardware CoreML may schedule a model on.
///
/// Mirrors `MLComputeUnits`; WhisperKit defaults: mel = CPU+GPU,
/// encoder/decoder = CPU+ANE.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum ComputeUnits {
  /// CPU only.
  CpuOnly,
  /// CPU and GPU.
  CpuAndGpu,
  /// CPU and Apple Neural Engine.
  CpuAndNeuralEngine,
  /// Any available hardware.
  #[default]
  All,
}

impl ComputeUnits {
  /// Stable snake_case name of the variant.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::CpuOnly => "cpu_only",
      Self::CpuAndGpu => "cpu_and_gpu",
      Self::CpuAndNeuralEngine => "cpu_and_neural_engine",
      Self::All => "all",
    }
  }

  #[inline(always)]
  pub(crate) const fn to_raw(self) -> MLComputeUnits {
    match self {
      Self::CpuOnly => MLComputeUnits::CPUOnly,
      Self::CpuAndGpu => MLComputeUnits::CPUAndGPU,
      Self::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
      Self::All => MLComputeUnits::All,
    }
  }
}

/// Error parsing a [`ComputeUnits`] name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown compute units name")]
pub struct ParseComputeUnitsError(());

impl core::str::FromStr for ComputeUnits {
  type Err = ParseComputeUnitsError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "cpu_only" => Self::CpuOnly,
      "cpu_and_gpu" => Self::CpuAndGpu,
      "cpu_and_neural_engine" => Self::CpuAndNeuralEngine,
      "all" => Self::All,
      _ => return Err(ParseComputeUnitsError(())),
    })
  }
}

#[cfg(test)]
mod tests;
