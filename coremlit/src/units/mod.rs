//! Compute-unit selection for model loading.

use objc2_core_ml::MLComputeUnits;

/// Which hardware CoreML may schedule a model on.
///
/// Mirrors `MLComputeUnits`; WhisperKit defaults: mel = CPU+GPU,
/// encoder/decoder = CPU+ANE.
///
/// # Wire form
///
/// Under the `serde` feature this is a snake_case STRING, one per variant:
/// `"cpu_only"`, `"cpu_and_gpu"`, `"cpu_and_neural_engine"`, `"all"` — exactly
/// what [`Self::as_str`] returns and what [`FromStr`](core::str::FromStr)
/// accepts. A caller can put the knob straight into a config struct of their
/// own — `coreml_compute = "cpu_only"` — with no `serde(with)` bridge.
///
/// The impls are WRITTEN, not derived, and that is the whole point. A derive
/// would use serde's enum protocol (`serialize_unit_variant` /
/// `deserialize_enum`), which renders the same `"cpu_only"` in JSON and TOML
/// but a bare variant INDEX in a format that does not spell its own shape —
/// postcard writes one byte, `0`, where a string is nine. Every door field used
/// to reach the wire through a private `serde(with)` bridge that wrote that
/// STRING, so the two coexisting routes agreed in every self-describing format
/// and disagreed in every other one: a value written by one route could not be
/// read by the other.
///
/// There is one route now. [`Serialize`](serde::Serialize) is
/// `serialize_str(self.as_str())`, [`Deserialize`](serde::Deserialize) is a
/// `String` read back through that same `FromStr`, and the ten private bridge
/// modules that duplicated them are deleted. An unknown name therefore fails with
/// [`ParseComputeUnitsError`]'s own text ("unknown compute units name")
/// wherever it is read from — the wording `audio::whisper`'s
/// `compute_units_rejects_unknown_names` pins — instead of that text through a
/// bridge and serde's "unknown variant" through the derive.
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

/// The snake_case [`ComputeUnits::as_str`] name as a STRING — never a variant
/// index. See the type's "Wire form" for why this is written rather than
/// derived.
#[cfg(feature = "serde")]
impl serde::Serialize for ComputeUnits {
  #[inline]
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(self.as_str())
  }
}

/// A `String` read back through [`FromStr`](core::str::FromStr) — the exact
/// inverse of the [`Serialize`](serde::Serialize) above, so an unknown name is
/// refused with [`ParseComputeUnitsError`]'s text in every format rather than
/// with serde's "unknown variant" wording in some of them.
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ComputeUnits {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    let name = <String as serde::Deserialize>::deserialize(deserializer)?;
    name.parse::<Self>().map_err(serde::de::Error::custom)
  }
}

#[cfg(test)]
mod tests;
