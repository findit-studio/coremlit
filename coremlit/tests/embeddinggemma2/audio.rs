//! The audio part over the staged EmbeddingGemma 2 tree: the contract, the
//! precision, prewarm, parity against R1, the resampler's reach, placement,
//! footprint, run identity and throughput.
//!
//! Every gate here is `#[ignore]`d and model-gated. The tree is
//! `EMBEDDINGGEMMA2_TEST_MODELS`, else the workspace's `Models/embeddinggemma2`;
//! with neither present a gate says so and passes. Each gate calls
//! `compile_artifacts` first, idempotently.
//!
//! The measured bands go through `measured_band`: asserted on the host class
//! in [`CHARACTERIZED_ON`], computed and printed everywhere. The portable
//! floor (M5's 0.97) is asserted on every host.

#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

#[path = "../support/host_class.rs"]
#[allow(dead_code)]
mod host_class;

// `measured_band.rs` reads the host class through its parent.
use host_class::HostClass;

#[path = "../support/measured_band.rs"]
#[allow(dead_code)]
mod measured_band;

#[path = "child.rs"]
mod child;

use std::{
  path::PathBuf,
  time::{Duration, Instant},
};

use child::{CHILD, in_child, phys_footprint};
use coremlit::{
  ComputeUnits, LoadOptions,
  embeddings::embeddinggemma2::{
    ArtifactLabel, AudioEncoder, AudioOptions, AudioPrecision, AudioTokens, Embedding, Part,
    TextArtifacts, TextEmbedder, TextOptions, compile_artifacts, error::Error, run_identity,
  },
};
use measured_band::{BandGate, CharacterizedHost};
use serde_json::Value;

/// The host class the measured bands below were characterized on. There:
/// M5's mean cosine against R1 is 0.999151; M3's worst coordinate against R2
/// is 1.3e-8; M8's worst cosine against `CpuOnly` is 0.999515 (`CpuAndGpu`),
/// 0.999959 (`CpuAndNeuralEngine`) and 0.999977 (`All`); M11 found every arm
/// repeating bit for bit.
const CHARACTERIZED_ON: Option<CharacterizedHost> = Some(CharacterizedHost {
  os_build: "26A428",
  os_product_version: "27.0",
  chip: "Apple M1 Max",
  arch: "arm64",
});

/// The command that re-characterizes these bands on this host.
const RECHARACTERIZE: &str = "EMBEDDINGGEMMA2_TEST_MODELS=<tree> cargo test -p coremlit \
  --features embeddinggemma2 --test embeddinggemma2_audio -- --ignored --nocapture --test-threads=1";

/// M5's portable floor: the worst window's cosine against R1.
const R1_FLOOR: f64 = 0.97;

/// M5f's provisional floor for the fp32 bundle: the worst window's cosine
/// against R1, measured 0.999885 at its first run (C13, the local package;
/// mean 0.999968), pinned with a 8.85e-4 margin.
const R1_FP32_FLOOR: f64 = 0.999;

/// M5's mean, measured 0.999151 on the characterizing host, pinned with a
/// 6.5e-4 margin.
const R1_MEAN_LO: f32 = 0.9985;

/// M8's band: the worst cosine of any placement against `CpuOnly`, measured
/// 0.999515 at worst, pinned with a 5e-4 margin.
const PLACEMENT_LO: f32 = 0.999;

/// The four placements, `CpuOnly` first: M8's reference.
const ARMS: [ComputeUnits; 4] = [
  ComputeUnits::CpuOnly,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuAndNeuralEngine,
  ComputeUnits::All,
];

/// The staged tree, compiled, or `None` (with the reason printed).
fn staged() -> Option<PathBuf> {
  let dir = std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || workspace_root::models_root().join("embeddinggemma2"),
    PathBuf::from,
  );
  if !dir.is_dir() {
    eprintln!(
      "SKIP: no staged EmbeddingGemma 2 tree at {} (set EMBEDDINGGEMMA2_TEST_MODELS)",
      dir.display()
    );
    return None;
  }
  compile_artifacts(&dir).expect("the staged tree compiles");
  Some(dir)
}

fn golden(name: &str) -> Value {
  let path = workspace_root::workspace_root()
    .join("coremlit/tests/embeddinggemma2/fixtures/goldens")
    .join(name);
  serde_json::from_slice(&std::fs::read(&path).expect("the golden reads")).expect("parses")
}

/// A committed 16 kHz mono clip as the goldens read it: s16le over 32,768,
/// cut to the golden's sample count.
fn clip(entry: &Value) -> Vec<f32> {
  let path = workspace_root::workspace_root()
    .join("coremlit/tests")
    .join(entry["file"].as_str().expect("a file"));
  let mut reader = hound::WavReader::open(&path).expect("the clip opens");
  let spec = reader.spec();
  assert_eq!(
    (spec.sample_rate, spec.channels),
    (16_000, 1),
    "{}",
    path.display()
  );
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("a sample")) / 32_768.0)
    .collect();
  let count = entry["samples"].as_u64().expect("samples") as usize;
  samples[..count].to_vec()
}

fn options(units: ComputeUnits) -> AudioOptions {
  AudioOptions::new().with_coreml(LoadOptions::new(units))
}

fn vector(value: &Value) -> Vec<f64> {
  value
    .as_array()
    .expect("a vector")
    .iter()
    .map(|v| v.as_f64().expect("a number"))
    .collect()
}

fn cosine(a: &[f32], b: &[f64]) -> f64 {
  let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * y).sum();
  let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
  let nb: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
  dot / (na * nb)
}

fn bits(v: &Embedding) -> Vec<u32> {
  v.as_slice().iter().map(|x| x.to_bits()).collect()
}

/// Every window of `samples`, encoded and embedded one by one.
fn embed_clip(
  encoder: &mut AudioEncoder,
  text: &mut TextEmbedder,
  samples: &[f32],
) -> Vec<Embedding> {
  encoder
    .plan(samples.len())
    .expect("plans")
    .iter()
    .map(|span| {
      let tokens: AudioTokens = encoder
        .encode_window(&samples[span.start()..span.end()])
        .expect("encodes");
      text.embed_audio_tokens(&tokens).expect("embeds")
    })
    .collect()
}

/// M1 (audio) and M10's local half, and G14's prewarm law: the fp16 bundle
/// loads under the kit's contract, and `prewarm` reports the load and one
/// throwaway prediction. Where that prediction pays Core ML's device
/// specialization, the first window after it does not, and runs faster. The
/// comparison is a measurement, asserted on the characterizing host and
/// printed elsewhere, as the vision suite's is: a host whose `CpuAndGpu` runs
/// on the CPU may time the two alike.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_audio_bundle_keeps_its_contract_and_prewarms() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 audio prewarm",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let mut encoder = AudioEncoder::new(&dir, &AudioOptions::new()).expect("an instance");
  assert_eq!(encoder.precision(), AudioPrecision::Fp16);
  let report = encoder.prewarm().expect("warms");
  assert_eq!(report.warmed().len(), 1);
  let warmed = report.warmed()[0];
  assert_eq!(warmed.function(), None);
  let window = vec![0.25f32; AudioEncoder::WINDOW_SAMPLES];
  let started = Instant::now();
  let tokens = encoder.encode_window(&window).expect("encodes");
  let next = started.elapsed();
  eprintln!(
    "M10 (local, CpuAndGpu): audio fp16 load {:.1?}, first prediction {:.1?}, the next {next:.1?}",
    warmed.load(),
    warmed.first_prediction()
  );
  assert_eq!(tokens.valid().get(), 250);
  println!(
    "{}",
    gate.check_holds(
      &format!(
        "the first window after prewarm took {next:.1?}, prewarm's prediction {:.1?}",
        warmed.first_prediction()
      ),
      next < warmed.first_prediction(),
      "faster than prewarm's prediction",
      "the first window after prewarm was not faster than prewarm's prediction",
    )
  );
}

/// The fp32 encoder over the tree. The lock's overlay table stages the fp32
/// package, so a tree without it is a failure, never a skip: the shard would
/// otherwise run these laws over nothing.
fn fp32_encoder(dir: &std::path::Path) -> AudioEncoder {
  AudioEncoder::new(
    dir,
    &AudioOptions::new().with_precision(AudioPrecision::Fp32),
  )
  .unwrap_or_else(|refusal| {
    panic!(
      "the staged tree holds no loadable fp32 audio bundle (MODELS_LOCK's overlay table stages \
       EmbeddingGemma2AudioFp32.mlpackage): {refusal}"
    )
  })
}

/// M1 (audio, fp32): the fp32 bundle keeps the one contract both precisions
/// share, and warms. The shard stages it.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_fp32_audio_bundle_keeps_the_one_contract() {
  let Some(dir) = staged() else { return };
  let mut encoder = fp32_encoder(&dir);
  assert_eq!(encoder.precision(), AudioPrecision::Fp32);
  let report = encoder.prewarm().expect("warms");
  let tokens = encoder
    .encode_window(&vec![0.25f32; AudioEncoder::WINDOW_SAMPLES])
    .expect("encodes");
  assert_eq!(tokens.valid().get(), 250);
  eprintln!(
    "M1 audio fp32: load {:.1?}, first prediction {:.1?}",
    report.warmed()[0].load(),
    report.warmed()[0].first_prediction()
  );
}

/// M5f's measurement: every window of the six clips through `encoder`, against
/// R1 only. Returns the worst and mean cosine and the window that is worst.
fn m5f_measure(
  encoder: &mut AudioEncoder,
  text: &mut TextEmbedder,
  audio: &Value,
) -> (f64, f64, String) {
  let (mut worst, mut worst_at, mut sum, mut count) = (f64::INFINITY, String::new(), 0.0, 0usize);
  for entry in audio["clips"].as_array().expect("clips") {
    let id = entry["id"].as_str().expect("id");
    let samples = clip(entry);
    let windows = entry["windows"].as_array().expect("windows");
    let vectors = embed_clip(encoder, text, &samples);
    assert_eq!(vectors.len(), windows.len(), "{id}: the plan");
    for (window, vector) in windows.iter().zip(&vectors) {
      let c = cosine(vector.as_slice(), &self::vector(&window["vector"]));
      if c < worst {
        worst = c;
        worst_at = format!("{id} @{}", window["start"]);
      }
      sum += c;
      count += 1;
    }
  }
  (worst, sum / count as f64, worst_at)
}

/// M5f: every window of the six clips through the fp32 bundle, against R1
/// only (never against the fp16 bundle, G14), on EVERY `ComputeUnits` arm the
/// host can load. Each must clear the floor, so the law proves the CPU road
/// too: a host with no usable GPU (the CI runners) runs `CpuAndGpu` on it, and
/// the first fp32 package, whose beta-less `layer_norm` Core ML's CPU road
/// miscomputes, held 0.9999 on the GPU road and 0.86 on the CPU road. The
/// default arm must load. One line per arm.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_fp32_audio_vectors_hold_r1() {
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let audio = golden("audio.json");
  let default_units = AudioOptions::new().coreml().compute();
  let (mut failures, mut default_loaded) = (Vec::new(), false);
  for units in ARMS {
    let options = AudioOptions::new()
      .with_precision(AudioPrecision::Fp32)
      .with_coreml(LoadOptions::new(units));
    match AudioEncoder::new(&dir, &options) {
      Ok(mut encoder) => {
        let (worst, mean, worst_at) = m5f_measure(&mut encoder, &mut text, &audio);
        eprintln!("M5f {units:?}: worst {worst:.6} ({worst_at}), mean {mean:.6}");
        default_loaded |= units == default_units;
        if worst < R1_FP32_FLOOR {
          failures.push(format!(
            "{units:?}: worst {worst} ({worst_at}) under {R1_FP32_FLOOR}"
          ));
        }
      }
      Err(refusal) => eprintln!("M5f {units:?}: not loaded on this host: {refusal}"),
    }
  }
  assert!(
    default_loaded,
    "the default arm {default_units:?} did not load"
  );
  assert!(failures.is_empty(), "arms under the floor: {failures:?}");
}

/// L14 (a), the audio wiring: a pin equal to the stamp's label loads, another
/// is refused naming the part and both labels.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_pinned_audio_label_is_honoured() {
  let Some(dir) = staged() else { return };
  let matching = AudioOptions::new().with_expect_artifacts(ArtifactLabel::FLUIDINFERENCE);
  assert!(AudioEncoder::new(&dir, &matching).is_ok());
  let other: ArtifactLabel = "sha256:0123456789ab".parse().expect("a label");
  match AudioEncoder::new(&dir, &AudioOptions::new().with_expect_artifacts(other)) {
    Err(Error::ArtifactIdentity(refusal)) => {
      assert_eq!(refusal.part(), Part::Audio);
      assert_eq!(refusal.expected(), other);
      assert_eq!(refusal.actual(), ArtifactLabel::FLUIDINFERENCE);
    }
    other => panic!("a mismatching pin is refused, got {other:?}"),
  }
}

/// The precision law: `fp32` runs the fp32 bundle the stamp names, whose
/// label is its own digests' and never FluidInference's. `fp16` runs
/// FluidInference's. (A directory without the bundle is refused by name; the
/// compile tests drive that hermetically.)
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_precision_selects_its_bundle() {
  let Some(dir) = staged() else { return };
  let fp16 = AudioEncoder::new(&dir, &AudioOptions::new()).expect("an instance");
  assert!(
    fp16
      .derivation_terms()
      .to_string()
      .contains("fluidinference@3f978dd")
  );
  let terms = fp32_encoder(&dir).derivation_terms().to_string();
  eprintln!("precision: fp32 runs {terms}");
  assert!(terms.contains("sha256:"), "{terms}");
  assert!(!terms.contains("fluidinference"), "{terms}");
}

/// M5: every window of the six clips, fp16, against R1 (PyTorch fp32): the
/// worst cosine clears the portable floor; the mean holds its band on the
/// characterizing host.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_audio_vectors_hold_r1() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 audio vs R1",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let mut encoder = AudioEncoder::new(&dir, &AudioOptions::new()).expect("an instance");
  let audio = golden("audio.json");
  let (mut worst, mut sum, mut count) = (f64::INFINITY, 0.0, 0usize);
  for entry in audio["clips"].as_array().expect("clips") {
    let id = entry["id"].as_str().expect("id");
    let samples = clip(entry);
    let windows = entry["windows"].as_array().expect("windows");
    let spans = encoder.plan(samples.len()).expect("plans");
    assert_eq!(spans.len(), windows.len(), "{id}: the plan");
    let vectors = embed_clip(&mut encoder, &mut text, &samples);
    for ((span, window), vector) in spans.iter().zip(windows).zip(&vectors) {
      assert_eq!(
        span.start() as u64,
        window["start"].as_u64().expect("start")
      );
      assert_eq!(
        span.len() as u64,
        window["samples"].as_u64().expect("samples")
      );
      let c = cosine(vector.as_slice(), &self::vector(&window["vector"]));
      eprintln!("M5 {id} @{}: {c:.6}", span.start());
      worst = worst.min(c);
      sum += c;
      count += 1;
    }
  }
  let mean = sum / count as f64;
  eprintln!("M5: {count} windows, worst {worst:.6}, mean {mean:.6}");
  assert!(worst >= R1_FLOOR, "worst {worst} under {R1_FLOOR}");
  println!(
    "{}",
    gate.check_band("audio mean cosine vs R1", mean as f32, R1_MEAN_LO, 1.0)
  );
}

/// A5: how far a resampler moves an audio vector. `jfk.wav`'s first window,
/// taken to 48 kHz and back by rubato's sinc resampler, against the window as
/// committed. Printed. FFmpeg's `swresample`, the production resampler, is
/// mediaio's and not this crate's to measure.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_resamplers_reach_is_characterized() {
  use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::direct::SequentialSliceOfSlices};

  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let mut encoder = AudioEncoder::new(&dir, &AudioOptions::new()).expect("an instance");
  let audio = golden("audio.json");
  let entry = &audio["clips"][0];
  assert_eq!(entry["id"], "jfk");
  let samples = clip(entry);
  let window = &samples[..AudioEncoder::WINDOW_SAMPLES];
  let resample = |input: &[f32], from: usize, to: usize| -> Vec<f32> {
    let mut resampler = Fft::<f32>::new(from, to, 1_024, 1, FixedSync::Input).expect("a resampler");
    let channels = [input];
    let adapter = SequentialSliceOfSlices::new(&channels, 1, input.len()).expect("an adapter");
    resampler
      .process_all(&adapter, input.len(), None)
      .expect("resamples")
      .take_data()
  };
  let up = resample(window, 16_000, 48_000);
  let mut back = resample(&up, 48_000, 16_000);
  back.truncate(AudioEncoder::WINDOW_SAMPLES);
  let direct = text
    .embed_audio_tokens(&encoder.encode_window(window).expect("encodes"))
    .expect("embeds");
  let round = text
    .embed_audio_tokens(&encoder.encode_window(&back).expect("encodes"))
    .expect("embeds");
  eprintln!(
    "A5: rubato 16 kHz -> 48 kHz -> 16 kHz moves jfk's first window to cosine {:.6}",
    direct.cosine(&round)
  );
}

/// M3 (audio), the outputs half: every window of the R2 probe's clips, fp16
/// under the default placements, against the vector FluidUse returned:
/// within 1e-6 per coordinate on the characterizing host, once R2's vectors
/// are re-normalized as the kit's are.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_audio_vectors_are_the_swift_probes() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 audio vs R2",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let mut encoder = AudioEncoder::new(&dir, &AudioOptions::new()).expect("an instance");
  let probe = golden("swift_probe.json");
  let mut worst = 0f64;
  let mut windows = 0;
  for entry in probe["audio"].as_array().expect("audio") {
    let samples = clip(entry);
    for window in entry["windows"].as_array().expect("windows") {
      let start = window["start"].as_u64().expect("start") as usize;
      let len = window["samples"].as_u64().expect("samples") as usize;
      let tokens = encoder
        .encode_window(&samples[start..start + len])
        .expect("encodes");
      let ours = text.embed_audio_tokens(&tokens).expect("embeds");
      let theirs = self::vector(&window["vector"]);
      let norm = theirs.iter().map(|v| v * v).sum::<f64>().sqrt();
      let deviation = ours
        .as_slice()
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (f64::from(*a) - b / norm).abs())
        .fold(0f64, f64::max);
      worst = worst.max(deviation);
      windows += 1;
    }
  }
  eprintln!("M3 audio: {windows} windows, worst coordinate against R2 {worst:.3e}");
  println!(
    "{}",
    gate.check_ceiling("audio vs R2, worst coordinate", worst as f32, 1e-6)
  );
}

/// M8 (audio): one full window under every placement, its vector's cosine
/// against the same window under `CpuOnly`, through the text part's default
/// placement. Printed, and asserted on the characterizing host as a band.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_audio_bundle_is_characterized_per_placement() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 audio placement",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let audio = golden("audio.json");
  let samples = clip(&audio["clips"][0]);
  let window = &samples[..AudioEncoder::WINDOW_SAMPLES];
  let embed = |units: ComputeUnits, text: &mut TextEmbedder| {
    let mut encoder = AudioEncoder::new(&dir, &options(units)).expect("an instance");
    let tokens = encoder.encode_window(window).expect("encodes");
    text.embed_audio_tokens(&tokens).expect("embeds")
  };
  let reference = embed(ComputeUnits::CpuOnly, &mut text);
  for units in &ARMS[1..] {
    let cosine = embed(*units, &mut text).cosine(&reference);
    eprintln!("M8 audio {units:?}: cosine against CpuOnly {cosine:.6}");
    println!(
      "{}",
      gate.check_band(
        &format!("audio under {units:?} vs CpuOnly"),
        cosine,
        PLACEMENT_LO,
        1.0
      )
    );
  }
}

/// M9a (audio): the physical footprint of one audio instance after its first
/// window, then of a second. Measured in a child process, since the
/// footprint is the whole process's.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_audio_footprint_is_characterized() {
  const NAME: &str = "the_audio_footprint_is_characterized";
  let Some(dir) = staged() else { return };
  if std::env::var_os(CHILD).is_some() {
    let start = phys_footprint();
    let mut first = AudioEncoder::new(&dir, &AudioOptions::new()).expect("an instance");
    first.prewarm().expect("warms");
    let with_one = phys_footprint();
    let mut second = AudioEncoder::new(&dir, &AudioOptions::new()).expect("a second");
    second.prewarm().expect("warms");
    let with_two = phys_footprint();
    println!("M9A {start} {with_one} {with_two}");
    return;
  }
  let line = in_child(NAME, "footprint", "M9A ");
  let readings: Vec<u64> = line
    .split(' ')
    .map(|n| n.parse().expect("a reading"))
    .collect();
  let mib = |bytes: u64| bytes as f64 / f64::from(1 << 20);
  eprintln!(
    "M9a audio: one instance +{:.1} MiB; a second +{:.1} MiB",
    mib(readings[1].saturating_sub(readings[0])),
    mib(readings[2].saturating_sub(readings[1]))
  );
}

/// One window's vector's bits, its tokens from a fresh audio instance under
/// `units`, embedded by `text`. The text part repeats bit for bit (M11,
/// text), so a moved bit here is the audio part's.
fn fresh_bits(
  dir: &std::path::Path,
  units: ComputeUnits,
  text: &mut TextEmbedder,
  window: &[f32],
) -> Vec<u32> {
  let mut encoder = AudioEncoder::new(dir, &options(units)).expect("an instance");
  window_bits(&mut encoder, text, window)
}

/// The bits of a window's vector: its tokens from `encoder`, embedded by
/// `text`.
fn window_bits(encoder: &mut AudioEncoder, text: &mut TextEmbedder, window: &[f32]) -> Vec<u32> {
  let tokens = encoder.encode_window(window).expect("encodes");
  bits(&text.embed_audio_tokens(&tokens).expect("embeds"))
}

/// M11 (audio): does one window repeat bit for bit, twice on one instance, on
/// two instances and in two processes, under every placement? Read through
/// the vector the text part makes of its tokens. Prints the table
/// `run_identity` holds; each `Some(true)` it holds is asserted here on the
/// characterizing host.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn audio_run_identity_is_measured_per_placement() {
  const NAME: &str = "audio_run_identity_is_measured_per_placement";
  let Some(dir) = staged() else { return };
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let audio = golden("audio.json");
  let samples = clip(&audio["clips"][0]);
  let window = &samples[..AudioEncoder::WINDOW_SAMPLES];
  if let Some(arm) = std::env::var_os(CHILD) {
    let index: usize = arm
      .to_str()
      .and_then(|a| a.parse().ok())
      .expect("an arm index");
    let digest = digest(&fresh_bits(&dir, ARMS[index], &mut text, window));
    println!("M11 {digest}");
    return;
  }
  let gate = BandGate::open(
    "embeddinggemma2 audio run identity",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  for (index, units) in ARMS.into_iter().enumerate() {
    let mut encoder = AudioEncoder::new(&dir, &options(units)).expect("an instance");
    let first = window_bits(&mut encoder, &mut text, window);
    let twice = window_bits(&mut encoder, &mut text, window) == first;
    let two_instances = fresh_bits(&dir, units, &mut text, window) == first;
    let two_processes = in_child(NAME, &index.to_string(), "M11 ") == digest(&first);
    let identical = twice && two_instances && two_processes;
    eprintln!(
      "M11 audio {units:?}: twice {twice}, two instances {two_instances}, two processes {two_processes}"
    );
    if run_identity(Part::Audio, units) == Some(true) {
      println!(
        "{}",
        gate.check_holds(
          &format!("audio run identity under {units:?}"),
          identical,
          "bit-identical",
          "a repeat moved the tokens",
        )
      );
    }
  }
}

/// A short digest of a vector's bits, for a child to print.
fn digest(bits: &[u32]) -> String {
  use sha2::{Digest, Sha256};
  let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
  Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// M12 (audio): full windows per second against the number of instances (1
/// to 4, one thread each) per placement. Printed.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn audio_throughput_is_characterized() {
  let Some(dir) = staged() else { return };
  let audio = golden("audio.json");
  let samples = clip(&audio["clips"][0]);
  let window = &samples[..AudioEncoder::WINDOW_SAMPLES];
  const PER_INSTANCE: usize = 6;
  for units in ARMS {
    for instances in 1..=4 {
      let mut encoders: Vec<AudioEncoder> = (0..instances)
        .map(|_| {
          let mut e = AudioEncoder::new(&dir, &options(units)).expect("an instance");
          e.prewarm().expect("warms");
          e
        })
        .collect();
      let started = Instant::now();
      std::thread::scope(|scope| {
        for encoder in &mut encoders {
          scope.spawn(move || {
            for _ in 0..PER_INSTANCE {
              encoder.encode_window(window).expect("encodes");
            }
          });
        }
      });
      let elapsed: Duration = started.elapsed();
      eprintln!(
        "M12 audio {units:?} x{instances}: {:.2} windows/s",
        (instances * PER_INSTANCE) as f64 / elapsed.as_secs_f64()
      );
    }
  }
}
