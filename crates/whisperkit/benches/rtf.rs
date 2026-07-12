//! End-to-end throughput harness on the tiny model (model-gated).
//!
//! Prints per-run wall time, real-time factor (processing / audio duration),
//! speed factor (audio / processing), and tokens/sec — the metrics of Swift
//! WhisperKit's regression benches (`BENCHMARKS.md`; `TranscriptionTimings.
//! tokensPerSecond` / `realTimeFactor`), so results are directly comparable
//! to a Swift run on the same machine.
//!
//! Run: `cargo bench -p whisperkit --bench rtf`
//! Skips (exit 0) when the tiny model is not downloaded — see the README's
//! "Getting models" section.

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

fn main() {
  let tiny = models_dir()
    .join("whisperkit-coreml")
    .join("openai_whisper-tiny");
  let tokenizer = models_dir().join("tokenizers").join("whisper-tiny");
  if !tiny.exists() || !tokenizer.exists() {
    eprintln!(
      "rtf bench skipped: openai_whisper-tiny not found under {} \
       (see README: Getting models)",
      models_dir().display()
    );
    return;
  }

  let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
  let audio = load_wav_mono_f32(&fixtures.join("audio/jfk.wav"));
  let audio_seconds = audio.len() as f64 / 16_000.0;

  let kit = WhisperKit::new(&Options::new(tiny, tokenizer)).expect("pipeline constructs");
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
