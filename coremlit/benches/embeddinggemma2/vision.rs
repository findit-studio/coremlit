//! EmbeddingGemma 2 vision throughput per budget and per placement,
//! model-gated.
//!
//! `harness = false`, like the text and audio benches: a load is a one-shot
//! cost, which criterion's resampling would hide. For each placement and each
//! budget it prints what `prewarm` reports (the load and the first
//! prediction), then the median and p90 of [`WARM_RUNS`] warm 1280 × 720
//! frames: the preparation alone (CPU), the vision model alone, and the frame
//! through to its vector, and frames per second. The text part runs under its
//! own default placement.
//!
//! Run with the staged tree in `EMBEDDINGGEMMA2_TEST_MODELS` (or
//! `Models/embeddinggemma2`):
//! `cargo bench -p coremlit --features embeddinggemma2 --bench embeddinggemma2_vision`

use std::{
  path::PathBuf,
  time::{Duration, Instant},
};

use coremlit::{
  ComputeUnits, LoadOptions,
  embeddings::embeddinggemma2::{
    ImageEncoder, ImageOptions, PreparedImage, Rgb8Image, TextArtifacts, TextEmbedder, TextOptions,
    VisionArtifacts, VisionBudget, VisionBudgets, compile_artifacts,
  },
};

/// Warm frames per budget and placement.
const WARM_RUNS: usize = 20;

/// The frame's geometry: 720p, the common keyframe.
const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

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
  let vision = VisionArtifacts::open(dir.as_path()).expect("opens");
  // A frame of smooth gradients, so the resize has work to do.
  let pixels: Vec<u8> = (0..HEIGHT)
    .flat_map(|y| (0..WIDTH).flat_map(move |x| [(x / 5) as u8, (y / 3) as u8, ((x + y) / 8) as u8]))
    .collect();
  let image = Rgb8Image::new(&pixels, WIDTH, HEIGHT).expect("a frame");
  let mut prepared = PreparedImage::new();
  for units in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    for budget in VisionBudget::ALL {
      let options = ImageOptions::new()
        .with_coreml(LoadOptions::new(units))
        .with_budgets(VisionBudgets::only(budget))
        .with_default_budget(budget);
      let mut encoder = ImageEncoder::new(&vision, &options).expect("an instance");
      let warmed = encoder.prewarm().expect("warms").warmed()[0];
      let (mut prepare, mut model, mut through) = (Vec::new(), Vec::new(), Vec::new());
      for _ in 0..WARM_RUNS {
        let started = Instant::now();
        vision
          .prepare_into(image, None, &options, &mut prepared)
          .expect("prepares");
        let prepared_at = started.elapsed();
        prepare.push(prepared_at);
        let tokens = encoder.encode(&prepared).expect("encodes");
        model.push(started.elapsed() - prepared_at);
        text.embed_image_tokens(&tokens).expect("embeds");
        through.push(started.elapsed());
      }
      let (prepare_median, _) = spread(prepare);
      let (model_median, model_p90) = spread(model);
      let (median, p90) = spread(through);
      println!(
        "{:<24} {:<10} load {:>9.1?}  first {:>9.1?}  prepare median {:>8.2?}  model median {:>8.2?} p90 {:>8.2?}  to vector median {:>8.2?} p90 {:>8.2?}  {:>6.2} frames/s",
        units.as_str(),
        budget.as_str(),
        warmed.load(),
        warmed.first_prediction(),
        prepare_median,
        model_median,
        model_p90,
        median,
        p90,
        1.0 / median.as_secs_f64()
      );
    }
  }
}
