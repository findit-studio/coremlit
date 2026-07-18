// Not every integration-test binary that includes `mod common;` uses every
// helper below (`model_io.rs` uses the model-path + sha helpers; the parity
// and state suites use the audio/fnv helpers). Each test file is its own
// crate, so an unused helper is dead code *in that crate* — allow it here so
// the shared module compiles clean under the workspace's `-D warnings` gate.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The compiled VAD artifact's directory name within [`models_dir`] — the
/// FluidInference `silero-vad-unified-256ms-v6.2.1` `.mlmodelc` (design spec
/// §5; the v6.2.1 artifact ships pre-compiled, so this loads directly with no
/// `coremlcompiler` step). Its HF revision + per-file SHA-256 are pinned in
/// `tests/model_io.rs` — the alignkit/speakerkit convention for adopted models,
/// NOT `MODELS_LOCK` (which a whisperkit hermetic gate holds to exactly the two
/// CI-downloaded whisper tables).
pub const ARTIFACT: &str = "silero-vad-unified-256ms-v6.2.1.mlmodelc";

/// Directory containing the downloaded vadkit model artifact.
///
/// Overridable via `VADKIT_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/vadkit` — gitignored, fetched dev-time (mirrors
/// `speakerkit`'s `SPEAKERKIT_TEST_MODELS`/`Models/speakerkit` and
/// `alignkit`'s `ALIGNKIT_TEST_MODELS`/`Models/alignkit`, one directory level
/// down for this crate's own model set). Fetch with:
///
/// ```text
/// hf download FluidInference/silero-vad-coreml \
///   --include "silero-vad-unified-256ms-v6.2.1*" \
///   --revision b419383c55c110e2c9271fa6ee0ea83d03c70d96 \
///   --local-dir Models/vadkit
/// ```
pub fn models_dir() -> PathBuf {
  std::env::var_os("VADKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("vadkit")
    },
    PathBuf::from,
  )
}

/// Path to the compiled VAD `.mlmodelc` artifact.
pub fn model_path() -> PathBuf {
  models_dir().join(ARTIFACT)
}

/// A committed parity fixture: a short 16 kHz mono clip borrowed by relative
/// path from the `speakerkit` crate's fixtures (dia's parity corpus), plus its
/// provenance. Borrowed rather than re-committed — both crates live in this
/// workspace and move together, exactly as `alignkit` borrows whisperkit's
/// `ted_60.wav`.
pub struct Fixture {
  /// Basename (no extension) of the WAV and its committed Swift golden.
  pub name: &'static str,
  /// Path within the `speakerkit` crate this clip is borrowed from.
  pub source: &'static str,
  /// SHA-256 of the borrowed WAV — pins the exact audio a swap/re-encode
  /// would silently change out from under the cross-crate relative path.
  pub sha256: &'static str,
  /// Why this clip is in the set (coverage rationale).
  pub note: &'static str,
}

/// The Swift-trace parity fixture set (spec §6 model-layer gate). Two real-
/// speech clips from dia's parity corpus: `02_pyannote_sample` is pyannote's
/// canonical multi-speaker demo (30.0 s → 118 chunks); `07_yuhewei_dongbei_
/// english` is a second real conversational clip (25.26 s → 99 chunks) that
/// exercises the short-final-chunk padding path. Together: 217 chunks across
/// 2 clips (the gate requires ≥ 40 chunks over ≥ 2 clips). Speaker counts are
/// deliberately NOT claimed here beyond "multi-speaker demo" — the fixture
/// names in dia's corpus do not reliably encode speaker counts, and VAD only
/// needs real speech with speech/non-speech transitions.
pub const FIXTURES: &[Fixture] = &[
  Fixture {
    name: "02_pyannote_sample",
    source: "crates/coremlit/tests/speaker/fixtures/audio/02_pyannote_sample.wav",
    sha256: "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e",
    note: "pyannote's canonical 30.0 s multi-speaker demo → 118 full 256 ms chunks",
  },
  Fixture {
    name: "07_yuhewei_dongbei_english",
    source: "crates/coremlit/tests/speaker/fixtures/audio/07_yuhewei_dongbei_english.wav",
    sha256: "096890ba8ffbaf10ca770c5373bf6c6664777f9421595c2cb7780af8cb2e46ff",
    note: "25.26 s clip → 98 full chunks + 1 short final chunk (exercises repeat-last padding)",
  },
];

/// Absolute path to a borrowed fixture WAV by basename.
pub fn fixture_wav_path(name: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/speaker/fixtures/audio")
    .join(format!("{name}.wav"))
}

/// Directory holding this crate's committed Swift-trace goldens.
pub fn golden_swift_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vad/fixtures/golden_swift")
}

/// Loads a 16 kHz mono WAV as `f32` samples — the single source of truth both
/// the Swift dumper (`tests/swift/.../DumpVadTraces.swift`'s `readPcm16Mono16k`)
/// and the Rust gate feed their models, so the two sides are input-identical by
/// construction; the [`fnv1a_f32`] recorded in each golden re-proves it at
/// replay time (the alignkit/speakerkit Gate-1 lesson: prove the inputs match).
///
/// 16-bit PCM is scaled by `1 / 32768`; float WAVs pass through.
///
/// # Panics
/// If the file is missing, not 16 kHz mono, or an unsupported bit depth.
pub fn load_wav_16k_mono(path: &Path) -> Vec<f32> {
  let mut reader =
    hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 16_000, "{}: not 16 kHz", path.display());
  assert_eq!(spec.channels, 1, "{}: not mono", path.display());
  match spec.sample_format {
    hound::SampleFormat::Int => {
      assert_eq!(
        spec.bits_per_sample,
        16,
        "{}: only 16-bit int PCM supported",
        path.display()
      );
      reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("read i16 sample")) / 32_768.0)
        .collect()
    }
    hound::SampleFormat::Float => reader
      .samples::<f32>()
      .map(|s| s.expect("read f32 sample"))
      .collect(),
  }
}

/// FNV-1a-64 over the little-endian bytes of `samples` — byte-for-byte the
/// same construction as `coremlit::audio::speaker::tests::common::fnv1a_f32` and the Swift
/// dumper's `fnv1aHex`, so the golden's recorded input hash proves both sides
/// fed the model element-identical audio.
pub fn fnv1a_f32(samples: &[f32]) -> u64 {
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  for &s in samples {
    for b in s.to_le_bytes() {
      h ^= u64::from(b);
      h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
  }
  h
}

/// Lowercase 16-hex-digit rendering of a [`fnv1a_f32`] hash for JSON storage.
pub fn fnv_hex(h: u64) -> String {
  format!("{h:016x}")
}

/// Lowercase-hex SHA-256 digest of a file's contents — the provenance/
/// integrity pin over the downloaded model artifacts (`tests/model_io.rs`).
pub fn sha256_hex(path: &Path) -> String {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}
