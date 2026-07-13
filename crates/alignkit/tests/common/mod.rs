use std::path::{Path, PathBuf};

/// Directory containing the downloaded alignkit model artifacts.
///
/// Overridable via `ALIGNKIT_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/alignkit` — gitignored, fetched dev-time (mirrors
/// whisperkit's `WHISPERKIT_TEST_MODELS`/`Models/` and dia-coreml's
/// `DIA_COREML_TEST_MODELS`/`Models/dia-coreml` conventions, one directory
/// level down for this crate's own model set).
pub fn models_dir() -> PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("alignkit")
    },
    PathBuf::from,
  )
}

/// Path to the compiled forced-aligner artifact.
///
/// Compiled from the downloaded `base960h_aligner.mlpackage` via `xcrun
/// coremlcompiler compile` at model-acquisition time (`coremlit::Model::load`
/// only accepts a compiled `.mlmodelc`; see `tests/model_io.rs`'s module doc
/// for the full acquisition record: source, revision, licence, per-file
/// SHA-256).
pub fn model_path() -> PathBuf {
  models_dir().join("base960h_aligner.mlmodelc")
}

/// Path to the 60 s @ 16 kHz mono fixture used by the graph-truth test.
///
/// alignkit has no committed audio fixtures of its own. `ted_60.wav` in the
/// whisperkit crate's `tests/fixtures/audio/` is already exactly 960,000
/// samples (60.000000 s @ 16 kHz mono int16, `afinfo`-verified at write
/// time) — precisely the `[1, 960000]` window `base960h_aligner.mlmodelc`
/// requires, with no padding needed — so this crate borrows it by relative
/// path instead of committing a second copy of a ~1.9 MB binary fixture that
/// would then need to stay byte-identical to the original forever. Both
/// crates live in this workspace and move together.
///
/// `#[allow(dead_code)]`: only `tests/model_io.rs` uses it; the per-binary
/// `common` copy in `tests/align_chunk.rs` does not.
#[allow(dead_code)]
pub fn ted_60_wav_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../whisperkit/tests/fixtures/audio/ted_60.wav")
}

/// Path to the 11 s @ 16 kHz mono `jfk.wav` fixture (176,000 samples, well
/// inside the encoder's 960,000 window), borrowed from the whisperkit crate
/// exactly as [`ted_60_wav_path`] is. Its known transcript is
/// [`JFK_TRANSCRIPT`]; together they drive `tests/align_chunk.rs`'s
/// end-to-end alignment.
///
/// `#[allow(dead_code)]`: only `tests/align_chunk.rs` uses it.
#[allow(dead_code)]
pub fn jfk_wav_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../whisperkit/tests/fixtures/audio/jfk.wav")
}

/// The known transcript for [`jfk_wav_path`]'s audio (whisperkit's
/// `tests/fixtures/golden/jfk_tiny_golden.json`).
///
/// `#[allow(dead_code)]`: only `tests/align_chunk.rs` uses it.
#[allow(dead_code)]
pub const JFK_TRANSCRIPT: &str = "And so my fellow Americans ask not what your country can do for \
                                  you, ask what you can do for your country.";

/// SHA-256 of [`jfk_wav_path`]'s **decoded** buffer — the 176,000 f32
/// samples [`load_wav_mono_f32`] returns, hashed as little-endian bytes.
///
/// This is the input-identity pin for `tests/parity_words.rs`. That gate
/// compares alignkit's word timings against asry's ONNX aligner, and such a
/// comparison is worth exactly nothing if the two sides are not looking at
/// the same audio: the FIRST attempt at an alignkit-vs-asry comparison
/// (`.superpowers/sdd/alignkit-gate1-diagnostic.md`) reported an alarming
/// "86.6% divergence" that turned out to be a harness bug — one side got a
/// padded buffer, the other an unpadded one. The number was measuring the
/// harness, not the models.
///
/// The gate feeds one `Vec<f32>`, by reference, to both aligners, so
/// buffer identity holds by construction; this digest additionally pins the
/// FIXTURE, so a `jfk.wav` that is silently re-encoded, resampled, or
/// swapped out from under the cross-crate relative path fails loudly instead
/// of re-measuring parity on different audio.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub const JFK_SAMPLES_SHA256: &str =
  "ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0";

/// Lowercase-hex SHA-256 of a decoded sample buffer, over its little-endian
/// `f32` bytes. Backs [`JFK_SAMPLES_SHA256`].
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn sha256_samples_hex(samples: &[f32]) -> String {
  use sha2::{Digest, Sha256};
  let mut hasher = Sha256::new();
  for sample in samples {
    hasher.update(sample.to_le_bytes());
  }
  hasher
    .finalize()
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// Directory holding asry's ONNX wav2vec2 oracle — the `models/` directory
/// of the asry checkout alignkit already path-depends on
/// (`crates/alignkit/Cargo.toml`'s `asry = { path = "../../../asry" }`), so
/// the default resolves for anyone who can build this crate at all.
/// Overridable via `ALIGNKIT_ASRY_MODELS`.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn asry_models_dir() -> PathBuf {
  std::env::var_os("ALIGNKIT_ASRY_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../asry")
        .join("models")
    },
    PathBuf::from,
  )
}

/// asry's ONNX wav2vec2-base-960h export (`onnx-community/
/// wav2vec2-base-960h-ONNX`, fetched by asry's own `build.rs`). Raw
/// **logits**, 32-class head — the oracle's encoder.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn asry_onnx_model_path() -> PathBuf {
  asry_models_dir().join("wav2vec2-base-960h.onnx")
}

/// The 32-class HuggingFace tokenizer matching [`asry_onnx_model_path`].
/// **Not** alignkit's bundled 29-class chordai asset: each tokenizer belongs
/// to its own CTC head, and asry's `Aligner::from_paths` validates the width.
///
/// `#[allow(dead_code)]`: only `tests/parity_words.rs` uses it.
#[allow(dead_code)]
pub fn asry_tokenizer_path() -> PathBuf {
  asry_models_dir().join("wav2vec2-base-960h-tokenizer.json")
}

/// Reads a 16 kHz mono 16-bit PCM WAV into normalized f32 samples.
///
/// Mirrors whisperkit's `tests/common::load_wav_mono_f32`.
pub fn load_wav_mono_f32(path: &Path) -> Vec<f32> {
  let mut reader = hound::WavReader::open(path).expect("fixture wav opens");
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

/// Lowercase-hex SHA-256 digest of a file's contents.
///
/// Backs `tests/model_io.rs`'s provenance/integrity pin over the downloaded
/// model artifacts. `common` is a `mod`, not a separate crate, so each
/// `tests/*.rs` integration-test binary compiles its own copy; binaries
/// that don't happen to call this one (e.g. `tests/align_chunk.rs`)
/// would otherwise warn `dead_code` on it.
#[allow(dead_code)]
pub fn sha256_hex(path: &Path) -> String {
  use sha2::{Digest, Sha256};
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}
