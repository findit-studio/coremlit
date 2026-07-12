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
pub fn ted_60_wav_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../whisperkit/tests/fixtures/audio/ted_60.wav")
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
/// that don't happen to call this one (e.g. `tests/parity_emissions.rs`)
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
