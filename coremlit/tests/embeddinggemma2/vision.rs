//! The vision part over the staged EmbeddingGemma 2 tree: each budget's
//! contract, prewarm, the pinned label, the unloaded budget, parity against R1
//! at each budget and against FluidUse's vectors, placement, footprint, run
//! identity and throughput.
//!
//! Every gate here is `#[ignore]`d and model-gated. The tree is
//! `EMBEDDINGGEMMA2_TEST_MODELS`, else the workspace's `Models/embeddinggemma2`;
//! with neither present a gate says so and passes. Each gate calls
//! `compile_artifacts` first, idempotently.
//!
//! The measured bands go through `measured_band`: asserted on the host class
//! in [`CHARACTERIZED_ON`], computed and printed everywhere. The portable
//! floor (M6's, per budget) is asserted on every host.

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
    ArtifactLabel, Embedding, ImageEncoder, ImageOptions, ImageTokens, Part, PreparedImage,
    Rgb8Image, TextArtifacts, TextEmbedder, TextOptions, VisionArtifacts, VisionBudget,
    VisionBudgets, compile_artifacts, error::Error, run_identity,
  },
};
use measured_band::{BandGate, CharacterizedHost};
use serde_json::Value;

/// The host class the measured bands below were characterized on. There:
/// M3's worst coordinate against R2 is 2.2e-8; M8's worst cosine against
/// `CpuOnly` is 0.999338 (`CpuAndGpu`), 0.999139 (`CpuAndNeuralEngine`) and
/// 0.999313 (`All`); M11 found every arm repeating bit for bit.
const CHARACTERIZED_ON: Option<CharacterizedHost> = Some(CharacterizedHost {
  os_build: "26A428",
  os_product_version: "27.0",
  chip: "Apple M1 Max",
  arch: "arm64",
});

/// The command that re-characterizes these bands on this host.
const RECHARACTERIZE: &str = "EMBEDDINGGEMMA2_TEST_MODELS=<tree> cargo test -p coremlit \
  --features embeddinggemma2 --test embeddinggemma2_vision -- --ignored --nocapture --test-threads=1";

/// M6's portable floor, per budget: the worst image's cosine against R1.
/// Measured at C15 (worst, mean over the seven images): 70 at 0.999878 and
/// 0.999943, 140 at 0.999895 and 0.999954, 280 at 0.999973 and 0.999982.
/// Each is pinned at 0.998, M2's portable floor: a host whose `CpuAndGpu`
/// runs on the CPU moves a vector by up to M8's placement spread, which the
/// margin covers.
const R1_FLOOR: [(VisionBudget, f64); 3] = [
  (VisionBudget::Tokens70, 0.998),
  (VisionBudget::Tokens140, 0.998),
  (VisionBudget::Tokens280, 0.998),
];

/// M8's band: the worst cosine of any placement against `CpuOnly`, measured
/// 0.999139 at worst, pinned with a 6.4e-4 margin.
const PLACEMENT_LO: f32 = 0.9985;

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

/// The synthetic images' rule, `_corpus.pattern_pixel`.
fn pattern(width: usize, height: usize) -> Vec<u8> {
  (0..height)
    .flat_map(|y| {
      (0..width).flat_map(move |x| {
        (0..3).map(move |c| ((x * 7 + y * 13 + c * 101 + ((x / 5) ^ (y / 3)) * 29) % 256) as u8)
      })
    })
    .collect()
}

/// A golden image's pixels: a committed PNG, or the pattern.
fn pixels(entry: &Value) -> (Vec<u8>, usize, usize) {
  match entry["source"]["file"].as_str() {
    Some(file) => {
      let path = workspace_root::workspace_root()
        .join("coremlit/tests")
        .join(file);
      let reader = std::fs::File::open(&path).expect("the image opens");
      let mut reader = png::Decoder::new(std::io::BufReader::new(reader))
        .read_info()
        .expect("a png");
      let mut buf = vec![0u8; reader.output_buffer_size().expect("a size")];
      let info = reader.next_frame(&mut buf).expect("decodes");
      let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
          .as_chunks::<4>()
          .0
          .iter()
          .flat_map(|p| [p[0], p[1], p[2]])
          .collect(),
        other => panic!("{other:?}"),
      };
      (rgb, info.width as usize, info.height as usize)
    }
    None => {
      let (w, h) = (
        entry["source"]["width"].as_u64().expect("width") as usize,
        entry["source"]["height"].as_u64().expect("height") as usize,
      );
      (pattern(w, h), w, h)
    }
  }
}

fn options(units: ComputeUnits, budgets: VisionBudgets) -> ImageOptions {
  ImageOptions::new()
    .with_coreml(LoadOptions::new(units))
    .with_budgets(budgets)
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

/// An image through `encoder` and `text`: prepared at `budget`, encoded,
/// embedded.
fn embed(
  vision: &VisionArtifacts,
  encoder: &mut ImageEncoder,
  text: &mut TextEmbedder,
  image: Rgb8Image<'_>,
  budget: VisionBudget,
) -> Embedding {
  let prepared: PreparedImage = vision
    .prepare(image, Some(budget), encoder.options())
    .expect("prepares");
  let tokens: ImageTokens = encoder.encode(&prepared).expect("encodes");
  text.embed_image_tokens(&tokens).expect("embeds")
}

/// M1 (vision) and M10's local half, and G14's prewarm law: every budget's
/// function loads under the kit's contract; `prewarm` reports each one's load
/// and a throwaway prediction, and the next image at a budget pays no
/// specialization: it runs faster than the prediction `prewarm` paid for.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn every_vision_budget_keeps_its_contract_and_prewarms() {
  let Some(dir) = staged() else { return };
  let vision = VisionArtifacts::open(&dir).expect("opens");
  assert_eq!(vision.artifact_label(), ArtifactLabel::FLUIDINFERENCE);
  let mut encoder = ImageEncoder::new(
    &vision,
    &options(ComputeUnits::CpuAndGpu, VisionBudgets::all()),
  )
  .expect("an instance");
  let report = encoder.prewarm().expect("warms");
  assert_eq!(report.warmed().len(), 3);
  let pixels = pattern(912, 672);
  let image = Rgb8Image::new(&pixels, 912, 672).expect("an image");
  for (warmed, budget) in report.warmed().iter().zip(VisionBudget::ALL) {
    let prepared = vision
      .prepare(image, Some(budget), encoder.options())
      .expect("prepares");
    let started = Instant::now();
    let tokens = encoder.encode(&prepared).expect("encodes");
    let next = started.elapsed();
    eprintln!(
      "M10 (local, CpuAndGpu): {budget}: load {:.1?}, first prediction {:.1?}, the next {next:.1?}",
      warmed.load(),
      warmed.first_prediction()
    );
    assert_eq!(usize::from(tokens.valid().get()), prepared.tokens());
    assert!(
      next < warmed.first_prediction(),
      "{budget}: the first image after prewarm took {next:?}, against prewarm's {:?}",
      warmed.first_prediction()
    );
  }
}

/// L14 (a), the vision wiring: a pin equal to the stamp's label loads, another
/// is refused naming the part and both labels.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn a_pinned_vision_label_is_honoured() {
  let Some(dir) = staged() else { return };
  let vision = VisionArtifacts::open(&dir).expect("opens");
  let matching = ImageOptions::new().with_expect_artifacts(ArtifactLabel::FLUIDINFERENCE);
  assert!(ImageEncoder::new(&vision, &matching).is_ok());
  let other: ArtifactLabel = "sha256:0123456789ab".parse().expect("a label");
  match ImageEncoder::new(&vision, &ImageOptions::new().with_expect_artifacts(other)) {
    Err(Error::ArtifactIdentity(refusal)) => {
      assert_eq!(refusal.part(), Part::Vision);
      assert_eq!(refusal.expected(), other);
      assert_eq!(refusal.actual(), ArtifactLabel::FLUIDINFERENCE);
    }
    other => panic!("a mismatching pin is refused, got {other:?}"),
  }
}

/// V5: a budget the instance did not load is refused by name, and never
/// loaded on the side: at `prepare` under the instance's options, and at
/// `encode` for an image prepared under wider ones.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn an_unloaded_budget_is_refused_by_name() {
  let Some(dir) = staged() else { return };
  let vision = VisionArtifacts::open(&dir).expect("opens");
  let only = ImageOptions::new();
  assert_eq!(only.budgets(), VisionBudgets::only(VisionBudget::Tokens140));
  let mut encoder = ImageEncoder::new(&vision, &only).expect("an instance");
  let pixels = pattern(912, 672);
  let image = Rgb8Image::new(&pixels, 912, 672).expect("an image");
  match vision.prepare(image, Some(VisionBudget::Tokens280), &only) {
    Err(Error::BudgetNotLoaded(refusal)) => {
      assert_eq!(refusal.budget(), VisionBudget::Tokens280);
      assert_eq!(refusal.loaded(), only.budgets());
    }
    other => panic!("expected BudgetNotLoaded at prepare, got {other:?}"),
  }
  let wide = ImageOptions::new().with_budgets(VisionBudgets::all());
  let prepared = vision
    .prepare(image, Some(VisionBudget::Tokens280), &wide)
    .expect("prepares");
  match encoder.encode(&prepared) {
    Err(Error::BudgetNotLoaded(refusal)) => {
      assert_eq!(refusal.budget(), VisionBudget::Tokens280);
      assert_eq!(refusal.loaded(), only.budgets());
    }
    other => panic!("expected BudgetNotLoaded at encode, got {other:?}"),
  }
  let prepared = vision.prepare(image, None, &only).expect("prepares");
  assert_eq!(prepared.budget(), Some(VisionBudget::Tokens140));
  encoder
    .encode(&prepared)
    .expect("the default budget encodes");
}

/// M6: the seven images at 70, 140 and 280, each budget against R1 on its
/// own: the worst cosine per budget clears its floor.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_vision_vectors_hold_r1_at_each_budget() {
  let Some(dir) = staged() else { return };
  let vision = VisionArtifacts::open(&dir).expect("opens");
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let mut encoder = ImageEncoder::new(
    &vision,
    &options(ComputeUnits::CpuAndGpu, VisionBudgets::all()),
  )
  .expect("an instance");
  let golden = golden("image.json");
  for (budget, floor) in R1_FLOOR {
    let (mut worst, mut sum, mut count) = (f64::INFINITY, 0.0, 0usize);
    for entry in golden["images"].as_array().expect("images") {
      let id = entry["id"].as_str().expect("id");
      let (pixels, width, height) = pixels(entry);
      let image = Rgb8Image::new(&pixels, width, height).expect("an image");
      let got = embed(&vision, &mut encoder, &mut text, image, budget);
      let want = vector(&entry["budgets"][budget.tokens().to_string()]["vector"]);
      let c = cosine(got.as_slice(), &want);
      eprintln!("M6 {budget} {id}: {c:.6}");
      worst = worst.min(c);
      sum += c;
      count += 1;
    }
    eprintln!(
      "M6 {budget}: {count} images, worst {worst:.6}, mean {:.6}",
      sum / count as f64
    );
    assert!(worst >= floor, "{budget}: worst {worst} under {floor}");
  }
}

/// M3 (vision), the outputs half: the probe's three synthetic images, against
/// the vectors FluidUse returned, within 1e-6 per coordinate on the
/// characterizing host once re-normalized.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_vision_vectors_are_the_swift_probes() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 vision vs R2",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let vision = VisionArtifacts::open(&dir).expect("opens");
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let mut encoder = ImageEncoder::new(
    &vision,
    &options(ComputeUnits::CpuAndGpu, VisionBudgets::all()),
  )
  .expect("an instance");
  let probe = golden("swift_probe.json");
  let mut worst = 0f64;
  for entry in probe["image"].as_array().expect("image") {
    let budget = VisionBudget::from_tokens(entry["budget"].as_u64().expect("budget") as usize)
      .expect("a budget");
    let (width, height) = (
      entry["width"].as_u64().expect("width") as usize,
      entry["height"].as_u64().expect("height") as usize,
    );
    let pixels = pattern(width, height);
    let image = Rgb8Image::new(&pixels, width, height).expect("an image");
    let ours = embed(&vision, &mut encoder, &mut text, image, budget);
    let theirs = vector(&entry["vector"]);
    let norm = theirs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let deviation = ours
      .as_slice()
      .iter()
      .zip(&theirs)
      .map(|(a, b)| (f64::from(*a) - b / norm).abs())
      .fold(0f64, f64::max);
    worst = worst.max(deviation);
  }
  eprintln!("M3 vision: worst coordinate against R2 {worst:.3e}");
  println!(
    "{}",
    gate.check_ceiling("vision vs R2, worst coordinate", worst as f32, 1e-6)
  );
}

/// M8 (vision): one image at each budget under every placement, its vector's
/// cosine against the same under `CpuOnly`, through the text part's default
/// placement. Printed, and asserted on the characterizing host as a band.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn every_vision_budget_is_characterized_per_placement() {
  let Some(dir) = staged() else { return };
  let gate = BandGate::open(
    "embeddinggemma2 vision placement",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  let vision = VisionArtifacts::open(&dir).expect("opens");
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let golden = golden("image.json");
  let (pixels, width, height) = pixels(&golden["images"][0]);
  let image = Rgb8Image::new(&pixels, width, height).expect("an image");
  let per_arm = |units: ComputeUnits, text: &mut TextEmbedder| -> Vec<Embedding> {
    let mut encoder =
      ImageEncoder::new(&vision, &options(units, VisionBudgets::all())).expect("an instance");
    VisionBudget::ALL
      .into_iter()
      .map(|budget| embed(&vision, &mut encoder, text, image, budget))
      .collect()
  };
  let reference = per_arm(ComputeUnits::CpuOnly, &mut text);
  for units in &ARMS[1..] {
    let worst = per_arm(*units, &mut text)
      .iter()
      .zip(&reference)
      .map(|(a, b)| a.cosine(b))
      .fold(f32::INFINITY, f32::min);
    eprintln!("M8 vision {units:?}: worst cosine against CpuOnly {worst:.6}");
    println!(
      "{}",
      gate.check_band(
        &format!("vision under {units:?} vs CpuOnly"),
        worst,
        PLACEMENT_LO,
        1.0
      )
    );
  }
}

/// M9a (vision): the physical footprint of one vision instance at its default
/// budget, warmed, then of a second. Measured in a child process.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_vision_footprint_is_characterized() {
  const NAME: &str = "the_vision_footprint_is_characterized";
  let Some(dir) = staged() else { return };
  if std::env::var_os(CHILD).is_some() {
    let vision = VisionArtifacts::open(&dir).expect("opens");
    let start = phys_footprint();
    let mut first = ImageEncoder::new(&vision, &ImageOptions::new()).expect("an instance");
    first.prewarm().expect("warms");
    let with_one = phys_footprint();
    let mut second = ImageEncoder::new(&vision, &ImageOptions::new()).expect("a second");
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
    "M9a vision: one instance +{:.1} MiB; a second +{:.1} MiB",
    mib(readings[1].saturating_sub(readings[0])),
    mib(readings[2].saturating_sub(readings[1]))
  );
}

/// An image's vector's bits at 140, its tokens from a fresh vision instance
/// under `units`, embedded by `text`.
fn fresh_bits(
  vision: &VisionArtifacts,
  units: ComputeUnits,
  text: &mut TextEmbedder,
  image: Rgb8Image<'_>,
) -> Vec<u32> {
  let mut encoder = ImageEncoder::new(
    vision,
    &options(units, VisionBudgets::only(VisionBudget::Tokens140)),
  )
  .expect("an instance");
  bits(&embed(
    vision,
    &mut encoder,
    text,
    image,
    VisionBudget::Tokens140,
  ))
}

/// M11 (vision): does one image repeat bit for bit, twice on one instance, on
/// two instances and in two processes, under every placement? Read through
/// the vector the text part makes of its tokens; prints the table
/// `run_identity` holds; each `Some(true)` it holds is asserted here on the
/// characterizing host.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn vision_run_identity_is_measured_per_placement() {
  const NAME: &str = "vision_run_identity_is_measured_per_placement";
  let Some(dir) = staged() else { return };
  let vision = VisionArtifacts::open(&dir).expect("opens");
  let artifacts = TextArtifacts::open(dir.as_path()).expect("opens");
  let mut text = TextEmbedder::new(&artifacts, &TextOptions::new()).expect("an instance");
  let golden = golden("image.json");
  let (pixels, width, height) = pixels(&golden["images"][0]);
  let image = Rgb8Image::new(&pixels, width, height).expect("an image");
  if let Some(arm) = std::env::var_os(CHILD) {
    let index: usize = arm
      .to_str()
      .and_then(|a| a.parse().ok())
      .expect("an arm index");
    println!(
      "M11 {:?}",
      fresh_bits(&vision, ARMS[index], &mut text, image)
    );
    return;
  }
  let gate = BandGate::open(
    "embeddinggemma2 vision run identity",
    CHARACTERIZED_ON,
    RECHARACTERIZE,
  );
  for (index, units) in ARMS.into_iter().enumerate() {
    let mut encoder = ImageEncoder::new(
      &vision,
      &options(units, VisionBudgets::only(VisionBudget::Tokens140)),
    )
    .expect("an instance");
    let first = bits(&embed(
      &vision,
      &mut encoder,
      &mut text,
      image,
      VisionBudget::Tokens140,
    ));
    let twice = bits(&embed(
      &vision,
      &mut encoder,
      &mut text,
      image,
      VisionBudget::Tokens140,
    )) == first;
    let two_instances = fresh_bits(&vision, units, &mut text, image) == first;
    let two_processes = in_child(NAME, &index.to_string(), "M11 ") == format!("{first:?}");
    let identical = twice && two_instances && two_processes;
    eprintln!(
      "M11 vision {units:?}: twice {twice}, two instances {two_instances}, two processes {two_processes}"
    );
    if run_identity(Part::Vision, units) == Some(true) {
      println!(
        "{}",
        gate.check_holds(
          &format!("vision run identity under {units:?}"),
          identical,
          "bit-identical",
          "a repeat moved the vector",
        )
      );
    }
  }
}

/// M12 (vision): images per second at 140, the image prepared once, against
/// the number of instances (1 to 4, one thread each) per placement. Printed.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn vision_throughput_is_characterized() {
  let Some(dir) = staged() else { return };
  let vision = VisionArtifacts::open(&dir).expect("opens");
  let pixels = pattern(624, 480);
  let image = Rgb8Image::new(&pixels, 624, 480).expect("an image");
  const PER_INSTANCE: usize = 8;
  for units in ARMS {
    for instances in 1..=4 {
      let mut encoders: Vec<ImageEncoder> = (0..instances)
        .map(|_| {
          let mut e = ImageEncoder::new(
            &vision,
            &options(units, VisionBudgets::only(VisionBudget::Tokens140)),
          )
          .expect("an instance");
          e.prewarm().expect("warms");
          e
        })
        .collect();
      // One prepared image per thread: a prepared image is `Send`, not `Sync`.
      let prepared: Vec<PreparedImage> = (0..instances)
        .map(|_| {
          vision
            .prepare(image, Some(VisionBudget::Tokens140), &ImageOptions::new())
            .expect("prepares")
        })
        .collect();
      let started = Instant::now();
      std::thread::scope(|scope| {
        for (encoder, prepared) in encoders.iter_mut().zip(prepared) {
          scope.spawn(move || {
            for _ in 0..PER_INSTANCE {
              encoder.encode(&prepared).expect("encodes");
            }
          });
        }
      });
      let elapsed: Duration = started.elapsed();
      eprintln!(
        "M12 vision {units:?} x{instances}: {:.2} images/s",
        (instances * PER_INSTANCE) as f64 / elapsed.as_secs_f64()
      );
    }
  }
}
