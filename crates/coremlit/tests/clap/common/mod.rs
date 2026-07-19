//! Shared helpers for clapkit's model-gated integration tests.
//!
//! The CoreML artifacts (fp16 `clap_{audio,text}.mlmodelc` + int8
//! `clap_{audio,text}_int8.mlmodelc`) are gitignored dev-time downloads from
//! `FinDIT-Studio/clapkit-coreml` (revision
//! `02a99c6a8be21da1e9a947499ea503a10c80c4f1`). Model-gated tests are `#[ignore]`
//! by default and run only when the tree is present.

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

/// Path to the compiled **int8** audio encoder, `clap_audio_int8.mlmodelc` —
/// the quantized variant shipped at `FinDIT-Studio/clapkit-coreml` revision
/// `02a99c6a` alongside the fp16 artifacts. Same I/O contract as the fp16
/// encoder (quantization compresses the weights, not the graph interface); its
/// bytes are pinned in `tests/clap/model_io.rs`.
#[allow(dead_code)]
pub fn audio_model_int8_path() -> PathBuf {
  models_dir().join("clap_audio_int8.mlmodelc")
}

/// Path to the compiled **int8** text encoder, `clap_text_int8.mlmodelc` (see
/// [`audio_model_int8_path`]); its bytes are pinned in
/// `tests/clap/text_model_io.rs`.
#[allow(dead_code)]
pub fn text_model_int8_path() -> PathBuf {
  models_dir().join("clap_text_int8.mlmodelc")
}

/// Directory holding textclap's Xenova ONNX graphs — the T4 parity oracle
/// (`tests/clap/parity_textclap.rs`). Contains the quantized (int8-class) graphs
/// `audio_model_quantized.onnx` / `text_model_quantized.onnx` that textclap
/// ships, and optionally the fp32 unquantized `audio_model.onnx` /
/// `text_model.onnx` used for the unquantized fp32 control.
///
/// Overridable via `CLAPKIT_TEXTCLAP_ONNX`; otherwise
/// `<workspace>/Models/textclap-onnx/onnx`. The Hugging Face Hub **preserves the
/// repository structure** under `--local-dir`, so `hf download
/// Xenova/clap-htsat-unfused --include "onnx/*model*.onnx" --local-dir
/// Models/textclap-onnx` lands the graphs under the `onnx/` subdirectory — the
/// default resolves straight to it (and the README's `CLAPKIT_TEXTCLAP_ONNX`
/// example points at the same `onnx/` path). Gitignored, fetched dev-time from
/// `Xenova/clap-htsat-unfused` revision `c28f2883…` (the exact revision textclap
/// pins in `models/MODELS.md`).
#[allow(dead_code)]
pub fn textclap_onnx_dir() -> PathBuf {
  std::env::var_os("CLAPKIT_TEXTCLAP_ONNX").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("textclap-onnx")
        .join("onnx")
    },
    PathBuf::from,
  )
}

/// Absolute path to a committed test fixture under `crates/coremlit/tests/clap/fixtures`.
#[allow(dead_code)]
pub fn fixture_path(relative: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("clap")
    .join("fixtures")
    .join(relative)
}

/// Decode a 48 kHz mono WAV fixture into `f32` samples in `[-1, 1]`. Asserts the
/// rate/channel contract so a mis-encoded fixture fails loudly rather than
/// feeding the wrong geometry into the encoders. Mirrors textclap's integration
/// reader so an identical `&[f32]` reaches both crates in the parity gate.
#[allow(dead_code)]
pub fn read_wav_48k_mono(path: &Path) -> Vec<f32> {
  let mut reader =
    hound::WavReader::open(path).unwrap_or_else(|e| panic!("open wav {path:?}: {e}"));
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 48_000, "fixture {path:?} must be 48 kHz");
  assert_eq!(spec.channels, 1, "fixture {path:?} must be mono");
  match spec.sample_format {
    hound::SampleFormat::Int => {
      let scale = 1.0 / (1_i64 << (spec.bits_per_sample - 1)) as f32;
      reader
        .samples::<i32>()
        .map(|s| s.expect("decode i32 sample") as f32 * scale)
        .collect()
    }
    hound::SampleFormat::Float => reader
      .samples::<f32>()
      .map(|s| s.expect("decode f32 sample"))
      .collect(),
  }
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
