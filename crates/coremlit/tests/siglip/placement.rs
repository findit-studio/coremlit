//! Compute-placement characterization (measured, never marketed).
//!
//! # Status: Wave C (model-gated)
//!
//! `#[ignore]`d until the conversion is staged (`SIGLIP_TEST_MODELS`). Records, per
//! tower × `{CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All}`, the worst corpus cosine
//! vs the committed goldens. The measured picture on this machine:
//!
//! - **Vision**: `CpuAndGpu` holds the floor (≈ 0.99999, fp32 GPU accumulation);
//!   `CpuAndNeuralEngine` COLLAPSES (≈ 0.31–0.41 worst, systematic across all 6
//!   images — materially worse than the earlier probe's 0.998118), and `All`
//!   FOLLOWS the ANE (the planner dispatches the ~99%-ANE-preferred vision graph
//!   to it) — so `All` is unsafe for vision (D1: keep the `CpuAndGpu` default;
//!   revisit only with a measurement showing an ANE fix).
//! - **Text**: robust on every arm (≈ 0.9998–0.99999); its whole-graph ANECCompile
//!   fails and falls back gracefully, so `CpuAndGpu` ships without the ANE-dispatch
//!   cost.
//!
//! The ANE arm is CHARACTERIZED in a wide band, never floor-gated (spec §3's
//! "no vacuous gate" rule); a future re-conversion that changes the ANE band REDs
//! this gate on purpose, forcing a deliberate re-characterization.

mod common;

use coremlit::{
  ComputeUnits,
  embeddings::siglip::{
    ImageEmbedder, ImageEmbedderOptions, Rgb8Image, TextEmbedder, TextEmbedderOptions,
  },
};

const FLOOR: f32 = 0.99917;

/// Worst corpus-image cosine vs the goldens for a given compute unit.
fn vision_worst(unit: ComputeUnits) -> f32 {
  let (images, _texts) = common::golden_corpus();
  let e = ImageEmbedder::load(
    common::vision_model_path(),
    common::pos_embed_path(),
    ImageEmbedderOptions::new().with_compute(unit),
  )
  .expect("load vision");
  let mut worst = 1.0f32;
  for g in &images {
    let (rgb, w, h) =
      common::decode_png_rgb8(&common::fixture_path(&format!("goldens/{}", g.file)));
    let emb = e
      .embed(Rgb8Image::new(&rgb, w, h).expect("rgb"))
      .expect("embed");
    worst = worst.min(common::cosine_checked(emb.as_slice(), &g.embedding));
  }
  worst
}

/// Worst corpus-text cosine vs the goldens for a given compute unit.
fn text_worst(unit: ComputeUnits) -> f32 {
  let (_images, texts) = common::golden_corpus();
  let e = TextEmbedder::load(
    common::text_model_path(),
    TextEmbedderOptions::new().with_compute(unit),
  )
  .expect("load text");
  let mut worst = 1.0f32;
  for g in &texts {
    let emb = e.embed(&g.text).expect("embed");
    worst = worst.min(common::cosine_checked(emb.as_slice(), &g.embedding));
  }
  worst
}

/// Characterize every placement arm's agreement with the goldens, pinning the
/// measured bands.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn placement_arms_are_characterized() {
  let units = [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ];

  println!("== vision ==");
  let mut v = std::collections::BTreeMap::new();
  for u in units {
    let w = vision_worst(u);
    println!("  {u:20?} worst {w:.6}");
    v.insert(format!("{u:?}"), w);
  }
  println!("== text ==");
  let mut t = std::collections::BTreeMap::new();
  for u in units {
    let w = text_worst(u);
    println!("  {u:20?} worst {w:.6}");
    t.insert(format!("{u:?}"), w);
  }

  let vg = v["CpuAndGpu"];
  let va = v["CpuAndNeuralEngine"];
  let vall = v["All"];

  // Vision CpuAndGpu is the floor-holding ship arm.
  assert!(vg >= FLOOR, "vision CpuAndGpu {vg:.6} below floor {FLOOR}");
  // Vision ANE is the CHARACTERIZED degraded arm (measured 0.31–0.41): a wide band,
  // and clearly below the floor (non-vacuity — it is NOT GPU-identical).
  assert!(
    (0.20..=0.70).contains(&va),
    "vision ANE worst {va:.6} outside the characterized band [0.20, 0.70] — \
     re-characterize placement (an ANE fix would land here and may allow All)"
  );
  assert!(
    va < 0.9,
    "vision ANE must be materially below the floor (non-vacuity)"
  );
  // D1: All follows the ANE for vision (planner dispatch), so All is unsafe here.
  assert!(
    (vall - va).abs() < 0.05,
    "vision All ({vall:.6}) is expected to track the ANE ({va:.6}) — the planner \
     dispatches the vision graph to the ANE, which is why the default is CpuAndGpu"
  );

  // Text is robust on every arm.
  for (name, w) in &t {
    assert!(*w >= 0.999, "text arm {name} worst {w:.6} unexpectedly low");
  }
}
