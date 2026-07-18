//! End-to-end throughput harness on the tiny model (model-gated).
//!
//! Prints per-run wall time, real-time factor (processing / audio duration),
//! speed factor (audio / processing), and tokens/sec — the metrics of Swift
//! WhisperKit's regression benches (`BENCHMARKS.md`; `TranscriptionTimings.
//! tokensPerSecond` / `realTimeFactor`), so results are directly comparable
//! to a Swift run on the same machine.
//!
//! Run: `cargo bench -p whisperkit --bench rtf`
//! Skips (exit 0) when the tiny model is not downloaded *or* incomplete
//! (see [`models_ready`]) — see the README's "Getting models" section.
//! `models_ready`'s own hermetic tests run separately, under `cargo test`,
//! via the `rtf_gate` test target in `Cargo.toml`: this file's `harness =
//! false` bench target keeps `cargo bench` running its real `main`
//! directly, but that also means no libtest runner is ever linked here to
//! call the `#[cfg(test)]` tests below (see the `tests` module's own
//! comment, and the `rtf_gate` stanza's).

use std::{
  path::{Path, PathBuf},
  time::Instant,
};

use whisperkit::{
  options::{DecodingOptions, Options},
  transcribe::WhisperKit,
};

const RUNS: usize = 5;

fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  )
}

fn load_wav_mono_f32(path: &Path) -> Vec<f32> {
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

/// Required on-disk artifacts for the tiny model: the three compiled
/// CoreML bundles under `model_dir`, and the tokenizer file under
/// `tokenizer_dir`. Directory existence alone is not proof of a complete
/// download — an interrupted `hf download` (see MODELS_LOCK / the
/// README's "Getting models") can leave both folders present while
/// missing individual files inside them, which used to reach
/// `WhisperKit::new().expect(...)` and panic instead of skipping.
fn models_ready(model_dir: &Path, tokenizer_dir: &Path) -> bool {
  const MODEL_BUNDLES: [&str; 3] = ["MelSpectrogram", "AudioEncoder", "TextDecoder"];
  MODEL_BUNDLES
    .iter()
    .all(|name| model_dir.join(format!("{name}.mlmodelc")).is_dir())
    && tokenizer_dir.join("tokenizer.json").is_file()
}

fn main() {
  let tiny = models_dir()
    .join("whisperkit-coreml")
    .join("openai_whisper-tiny");
  let tokenizer = models_dir().join("tokenizers").join("whisper-tiny");
  if !models_ready(&tiny, &tokenizer) {
    eprintln!(
      "rtf bench skipped: openai_whisper-tiny not found or incomplete under {} \
       (see README: Getting models)",
      models_dir().display()
    );
    return;
  }

  let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
  let audio = load_wav_mono_f32(&fixtures.join("audio/jfk.wav"));
  let audio_seconds = audio.len() as f64 / 16_000.0;

  // Belt-and-braces alongside `models_ready`: a model directory can still
  // fail to load for reasons the artifact check above doesn't enumerate
  // (a corrupt bundle, an unreadable file) — skip rather than panic here
  // too, instead of the `expect` this replaced.
  let kit = match WhisperKit::new(&Options::new(tiny, tokenizer)) {
    Ok(kit) => kit,
    Err(err) => {
      eprintln!(
        "rtf bench skipped: WhisperKit::new failed ({err}) — treating as an incomplete install"
      );
      return;
    }
  };
  let options = DecodingOptions::new();

  // Warmup run amortizes model specialization / ANE compilation.
  kit
    .transcribe(&audio, &options)
    .expect("warmup transcription");

  println!("jfk.wav ({audio_seconds:.1} s) x {RUNS} runs, openai_whisper-tiny:");
  let mut wall_rtfs = Vec::with_capacity(RUNS);
  for run in 1..=RUNS {
    let started = Instant::now();
    let result = kit.transcribe(&audio, &options).expect("transcription");
    let wall = started.elapsed().as_secs_f64();
    let rtf = wall / audio_seconds;
    let timings = result.timings();
    wall_rtfs.push(rtf);
    println!(
      "  run {run}: wall {wall:.3} s  rtf {rtf:.4}  speed {speed:.1}x  \
       tokens/s {tps:.1}  (internal rtf {internal:.4})",
      speed = 1.0 / rtf,
      tps = timings.tokens_per_second(),
      internal = timings.real_time_factor(),
    );
  }
  wall_rtfs.sort_by(f64::total_cmp);
  let median = wall_rtfs[RUNS / 2];
  println!("  median: rtf {median:.4}  speed {:.1}x", 1.0 / median);
}

// `cfg(test)` is true for BOTH targets that compile this file (Cargo sets
// it for bench/test-kind targets independent of `harness`), but this
// module's contents are only ever actually CALLED under the `rtf_gate`
// test target (`harness = true`, so libtest's synthesized runner invokes
// every `#[test]` fn). Under the `rtf` bench target itself (`harness =
// false`, so no runner is linked — see that stanza's Cargo.toml comment),
// this module still compiles but nothing calls into it, which makes
// `mlmodelc_dirs` and the `models_ready` import look dead from that
// build's own reachability graph even though both are exercised
// (`models_ready` genuinely, via `main`; the rest via `rtf_gate`).
#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod tests {
  use super::models_ready;

  fn mlmodelc_dirs(model_dir: &std::path::Path) {
    for name in ["MelSpectrogram", "AudioEncoder", "TextDecoder"] {
      std::fs::create_dir_all(model_dir.join(format!("{name}.mlmodelc"))).unwrap();
    }
  }

  #[test]
  fn empty_root_is_not_ready() {
    let model_dir = tempfile::tempdir().unwrap();
    let tokenizer_dir = tempfile::tempdir().unwrap();
    assert!(!models_ready(model_dir.path(), tokenizer_dir.path()));
  }

  #[test]
  fn model_dir_without_tokenizer_json_is_not_ready() {
    let model_dir = tempfile::tempdir().unwrap();
    let tokenizer_dir = tempfile::tempdir().unwrap();
    mlmodelc_dirs(model_dir.path());
    // tokenizer_dir exists but stays empty — the interrupted-download case.
    assert!(!models_ready(model_dir.path(), tokenizer_dir.path()));
  }

  #[test]
  fn tokenizer_json_without_model_dirs_is_not_ready() {
    let model_dir = tempfile::tempdir().unwrap();
    let tokenizer_dir = tempfile::tempdir().unwrap();
    std::fs::write(tokenizer_dir.path().join("tokenizer.json"), b"{}").unwrap();
    assert!(!models_ready(model_dir.path(), tokenizer_dir.path()));
  }

  #[test]
  fn fully_populated_root_is_ready() {
    let model_dir = tempfile::tempdir().unwrap();
    let tokenizer_dir = tempfile::tempdir().unwrap();
    mlmodelc_dirs(model_dir.path());
    std::fs::write(tokenizer_dir.path().join("tokenizer.json"), b"{}").unwrap();
    assert!(models_ready(model_dir.path(), tokenizer_dir.path()));
  }
}
