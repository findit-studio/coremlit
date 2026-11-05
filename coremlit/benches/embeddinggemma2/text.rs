//! EmbeddingGemma 2 text throughput, per function and packed, per placement,
//! model-gated.
//!
//! `harness = false`, like `benches/clap/encode.rs`: a load is a one-shot
//! cost, which criterion's resampling would hide. For each placement and each
//! function it prints the first-observed load and first prediction (what
//! `prewarm` reports), then the median and p90 of [`WARM_RUNS`] warm calls,
//! and texts per second. `pack_256` runs with eight short texts per call.
//!
//! True cold specialization is not forced: it is a one-time cost of a host
//! whose Core ML cache has never seen the model, and this harness touches no
//! shared cache. The first-observed column carries it on such a host.
//!
//! Run with the staged tree in `EMBEDDINGGEMMA2_TEST_MODELS` (or
//! `Models/embeddinggemma2`):
//! `cargo bench -p coremlit --features embeddinggemma2 --bench embeddinggemma2_text`

use std::{path::PathBuf, time::Instant};

use coremlit::{
  ComputeUnits, LoadOptions,
  embeddings::embeddinggemma2::{
    Batching, Prompt, TextArtifacts, TextEmbedder, TextFunction, TextFunctions, TextOptions,
    compile_artifacts,
  },
};

/// Warm calls per function and placement.
const WARM_RUNS: usize = 30;

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

/// A text of `words` words, which encodes to about `words + 2` ids.
fn text_of(words: usize) -> String {
  (0..words)
    .map(|i| ["river", "boat", "light", "stone", "wind"][i % 5])
    .collect::<Vec<_>>()
    .join(" ")
}

fn main() {
  let Some(dir) = staged() else {
    eprintln!("SKIP: no staged EmbeddingGemma 2 tree (set EMBEDDINGGEMMA2_TEST_MODELS)");
    return;
  };
  compile_artifacts(&dir).expect("the staged tree compiles");
  let artifacts = TextArtifacts::open(&dir).expect("opens");
  let cases = [
    (TextFunction::Embed32, 20),
    (TextFunction::Embed48, 40),
    (TextFunction::Embed64, 60),
    (TextFunction::Embed128, 120),
    (TextFunction::Embed256, 250),
    (TextFunction::Embed512, 500),
    (TextFunction::Pack256, 20),
  ];
  for units in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    for (function, words) in cases {
      let batching = if function == TextFunction::Pack256 {
        Batching::Packed
      } else {
        Batching::Single
      };
      let options = TextOptions::new()
        .with_coreml(LoadOptions::new(units))
        .with_batching(batching)
        .with_prewarm(TextFunctions::empty().with(function));
      let mut embedder = TextEmbedder::new(&artifacts, &options).expect("an instance");
      let report = embedder.prewarm().expect("warms");
      let warmed = report.warmed()[0];
      let text = text_of(words);
      let texts: Vec<&str> = if function == TextFunction::Pack256 {
        vec![text.as_str(); 8]
      } else {
        vec![text.as_str()]
      };
      let mut out = Vec::new();
      let mut times = Vec::with_capacity(WARM_RUNS);
      for _ in 0..WARM_RUNS {
        out.clear();
        let started = Instant::now();
        embedder
          .embed_batch(&texts, Prompt::Bare, &mut out)
          .expect("embeds");
        times.push(started.elapsed());
      }
      times.sort();
      let median = times[WARM_RUNS / 2];
      let p90 = times[WARM_RUNS * 9 / 10];
      println!(
        "{:<24} {:<9} load {:>9.1?}  first {:>9.1?}  median {:>8.2?}  p90 {:>8.2?}  {:>7.1} texts/s",
        units.as_str(),
        function.as_str(),
        warmed.load(),
        warmed.first_prediction(),
        median,
        p90,
        texts.len() as f64 / median.as_secs_f64()
      );
    }
  }
}
