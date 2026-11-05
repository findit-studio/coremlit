//! The audio part: the precision of the audio bundle.

/// Which audio package runs: FluidInference's fp16 conversion, or this
/// crate's own fp32 conversion of the same audio tower.
///
/// Both keep one I/O contract (`waveform` F32 `[1, 160160]`, `frame_mask` F16
/// `[1, 1000]` → `audio_tokens` F16 `[1, 250, 512]`); only the precision the
/// graph computes in differs. The text function that pools the tokens is fp16
/// either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum AudioPrecision {
  /// `EmbeddingGemma2Audio.mlpackage`, FluidInference's fp16 conversion. The
  /// default.
  Fp16,
  /// `EmbeddingGemma2AudioFp32.mlpackage`, converted by
  /// `coremlit/conversion/embeddinggemma2` under `compute_precision=FLOAT32`.
  Fp32,
}

impl AudioPrecision {
  /// Stable snake_case name of the precision: `"fp16"` or `"fp32"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Fp16 => "fp16",
      Self::Fp32 => "fp32",
    }
  }

  /// The package's name without its extension: `"EmbeddingGemma2Audio"` or
  /// `"EmbeddingGemma2AudioFp32"`.
  #[inline]
  pub const fn package_stem(&self) -> &'static str {
    match self {
      Self::Fp16 => "EmbeddingGemma2Audio",
      Self::Fp32 => "EmbeddingGemma2AudioFp32",
    }
  }
}

#[cfg(test)]
mod tests;
