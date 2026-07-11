use std::path::PathBuf;

pub fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  )
}

pub fn tiny_dir() -> PathBuf {
  models_dir()
    .join("whisperkit-coreml")
    .join("openai_whisper-tiny")
}

// `tests/common/mod.rs` is compiled fresh into each integration-test
// binary that declares `mod common;`; not every binary uses every helper
// (only `pipeline.rs` and `parity_jfk.rs` need a tokenizer path so far),
// so an unused-in-THIS-binary helper is expected here, not a real
// dead-code bug.
#[allow(dead_code)]
pub fn tokenizer_dir() -> PathBuf {
  models_dir().join("tokenizers").join("whisper-tiny")
}

pub fn fixtures_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("fixtures")
}

/// Reads a 16 kHz mono 16-bit PCM WAV into normalized f32 samples.
///
/// All three committed fixtures (`jfk.wav`, `es_test_clip.wav`,
/// `ja_test_clip.wav`) are already 16 kHz mono 16-bit PCM as copied from
/// `argmax-oss-swift` (`afinfo`-verified at plan time: jfk 11.000s /
/// 176,000 samples, es_test_clip 7.664562s / 122,633 samples, ja_test_clip
/// 2.773s / 44,368 samples) — no `afconvert` resampling was needed for any
/// of them, though only `jfk.wav`'s sample count is asserted below.
pub fn load_wav_mono_f32(path: &std::path::Path) -> Vec<f32> {
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
