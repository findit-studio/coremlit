//! EmbeddingGemma 2 audio throughput per window, per precision staged and per
//! placement, model-gated.
//!
//! `harness = false`, like the text bench: a load is a one-shot cost, which
//! criterion's resampling would hide. For each precision whose bundle the tree
//! stages, and each placement, it prints what `prewarm` reports (the load and
//! the first prediction), then the median and p90 of [`WARM_RUNS`] warm full
//! windows, the audio model alone and the window through to its vector, and
//! windows per second. The text part runs under its own default placement.
//!
//! Run with the staged tree in `EMBEDDINGGEMMA2_TEST_MODELS` (or
//! `Models/embeddinggemma2`):
//! `cargo bench -p coremlit --features embeddinggemma2 --bench embeddinggemma2_audio`

use std::{
  path::PathBuf,
  time::{Duration, Instant},
};

use coremlit::{
  ComputeUnits, LoadOptions,
  embeddings::embeddinggemma2::{
    AudioEncoder, AudioOptions, AudioPrecision, TextArtifacts, TextEmbedder, TextOptions,
    compile_artifacts,
  },
};

/// Warm windows per precision and placement.
const WARM_RUNS: usize = 20;

fn staged() -> Option<PathBuf> {
  let dir = std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../Models/embeddinggemma2"
      ))
    },
    PathBuf::from,
  );
  dir.is_dir().then_some(dir)
}

/// The median and p90 of `times`.
fn spread(mut times: Vec<Duration>) -> (Duration, Duration) {
  times.sort();
  (times[times.len() / 2], times[times.len() * 9 / 10])
}

fn main() {
  let Some(dir) = staged() else {
    eprintln!("SKIP: no staged EmbeddingGemma 2 tree (set EMBEDDINGGEMMA2_TEST_MODELS)");
    return;
  };
  compile_artifacts(&dir).expect("the staged tree compiles");
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  // A full window of a 440 Hz tone at half scale.
  let window: Vec<f32> = (0..AudioEncoder::WINDOW_SAMPLES)
    .map(|i| 0.5 * (i as f32 * 440.0 * core::f32::consts::TAU / 16_000.0).sin())
    .collect();
  for precision in [AudioPrecision::Fp16, AudioPrecision::Fp32] {
    for units in [
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuAndGpu,
      ComputeUnits::CpuAndNeuralEngine,
      ComputeUnits::All,
    ] {
      let options = AudioOptions::new()
        .with_precision(precision)
        .with_coreml(LoadOptions::new(units));
      let mut encoder = match AudioEncoder::new(&dir, &options) {
        Ok(encoder) => encoder,
        Err(e) => {
          eprintln!("SKIP {precision}: {e}");
          break;
        }
      };
      let warmed = encoder.prewarm().expect("warms").warmed()[0];
      let (mut tokens_only, mut through) = (Vec::new(), Vec::new());
      for _ in 0..WARM_RUNS {
        let started = Instant::now();
        let tokens = encoder.encode_window(&window).expect("encodes");
        tokens_only.push(started.elapsed());
        text.embed_audio_tokens(&tokens).expect("embeds");
        through.push(started.elapsed());
      }
      let (tokens_median, tokens_p90) = spread(tokens_only);
      let (median, p90) = spread(through);
      println!(
        "{:<5} {:<24} load {:>9.1?}  first {:>9.1?}  tokens median {:>8.2?} p90 {:>8.2?}  to vector median {:>8.2?} p90 {:>8.2?}  {:>6.2} windows/s",
        precision.as_str(),
        units.as_str(),
        warmed.load(),
        warmed.first_prediction(),
        tokens_median,
        tokens_p90,
        median,
        p90,
        1.0 / median.as_secs_f64()
      );
    }
  }
}
