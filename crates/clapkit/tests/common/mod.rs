//! Shared helpers for clapkit's model-gated integration tests.
//!
//! The CoreML artifacts (`clap_audio.mlmodelc` / `clap_text.mlmodelc`) are
//! gitignored dev-time downloads from `FinDIT-Studio/clapkit-coreml`
//! (revision `97d631f3814e1e46b798a8e88c9aa2e2202fdf67`). Model-gated tests are
//! `#[ignore]` by default and run only when the tree is present.

use std::path::{Path, PathBuf};

/// Directory containing the downloaded clapkit CoreML artifacts.
///
/// Overridable via `CLAPKIT_TEST_MODELS`; otherwise `<workspace>/Models/clapkit`
/// — gitignored, fetched dev-time (mirrors alignkit's `ALIGNKIT_TEST_MODELS` /
/// speakerkit's `SPEAKERKIT_TEST_MODELS` conventions).
#[allow(dead_code)]
pub fn models_dir() -> PathBuf {
  std::env::var_os("CLAPKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("clapkit")
    },
    PathBuf::from,
  )
}

/// Path to the compiled audio encoder, `clap_audio.mlmodelc`.
#[allow(dead_code)]
pub fn audio_model_path() -> PathBuf {
  models_dir().join("clap_audio.mlmodelc")
}

/// Path to the compiled text encoder, `clap_text.mlmodelc`.
#[allow(dead_code)]
pub fn text_model_path() -> PathBuf {
  models_dir().join("clap_text.mlmodelc")
}

/// Lowercase-hex SHA-256 of a file's contents. Backs the `model_io` provenance
/// pins.
#[allow(dead_code)]
pub fn sha256_hex(path: &Path) -> String {
  use sha2::{Digest, Sha256};
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// A deterministic 48 kHz mono window of `len` samples: a sum of a few fixed
/// sinusoids (no RNG dependency), for placement/agreement tests that only need a
/// stable, non-trivial input to compare across compute units.
#[allow(dead_code)]
pub fn deterministic_window(len: usize) -> Vec<f32> {
  const SR: f32 = 48_000.0;
  (0..len)
    .map(|i| {
      let t = i as f32 / SR;
      let two_pi = std::f32::consts::TAU;
      0.5 * (two_pi * 220.0 * t).sin()
        + 0.3 * (two_pi * 440.0 * t).sin()
        + 0.2 * (two_pi * 1760.0 * t).sin()
    })
    .collect()
}
