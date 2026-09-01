//! End-to-end throughput harness on the tiny model (model-gated).
//!
//! Prints per-run wall time, real-time factor (processing / audio duration),
//! speed factor (audio / processing), and tokens/sec — the metrics of Swift
//! WhisperKit's regression benches (`BENCHMARKS.md`; `TranscriptionTimings.
//! tokensPerSecond` / `realTimeFactor`), so results are directly comparable
//! to a Swift run on the same machine.
//!
//! Run: `cargo bench -p coremlit --features whisper --bench whisper_rtf`
//! Skips (exit 0) when the tiny model is not downloaded *or* incomplete (see
//! [`models_ready`]) — see the README's "Getting models" section.
//! [`models_ready`]'s own hermetic tests live in the sibling
//! `rtf_gate.rs` (the `whisper_rtf_gate` test target, run via `cargo test -p
//! coremlit --features whisper --test whisper_rtf_gate`) — a real, separate
//! file rather than this one compiled a second time under a `harness = true`
//! target, which used to cost a permanent `cargo`-level "file present in
//! multiple build targets" warning. The two share only [`models_ready`]
//! itself, via `rtf_models_ready.rs` (`#[path]`, the `workspace_root.rs`
//! convention).

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops — see its module doc.
#[path = "../../tests/support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

#[path = "rtf_models_ready.rs"]
mod rtf_models_ready;
use rtf_models_ready::models_ready;

use std::{
  path::{Path, PathBuf},
  time::Instant,
};

use coremlit::audio::whisper::{
  options::{DecodingOptions, Options},
  transcribe::WhisperKit,
};

const RUNS: usize = 5;

fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(workspace_root::models_root, PathBuf::from)
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
  if !models_ready(&tiny, &tokenizer) {
    eprintln!(
      "rtf bench skipped: openai_whisper-tiny not found or incomplete under {} \
       (see README: Getting models)",
      models_dir().display()
    );
    return;
  }

  let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/whisper/fixtures");
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
