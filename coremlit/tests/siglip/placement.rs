//! Compute-placement characterization (measured, never marketed).
//!
//! # Status: Wave C (model-gated)
//!
//! `#[ignore]`d until the conversion is staged (`SIGLIP_TEST_MODELS`). Records, per
//! tower × `{CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All}`, the worst corpus cosine
//! vs the committed goldens. The measured picture on the machine this was
//! characterized on:
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
//! # That picture is a description of ONE COMPUTER
//!
//! Every number in it — the collapse band, "materially below the floor", "All
//! follows the ANE" — is a statement about a particular Neural Engine
//! generation, its compiler and its firmware. CoreML contracts none of that to
//! reproduce across macOS builds or chips (#36). Asserting it elsewhere measures
//! the elsewhere, and CI proved it: the `macos-15-arm64` runner measured vision
//! ANE at 0.999664 — the ANE arm agreeing with the `CpuAndGpu` arm, exactly what
//! the non-vacuity check was written to rule out on the characterizing host —
//! and red a band nothing had broken.
//!
//! So the measured bands go through [`common::BandGate`]
//! (`tests/support/measured_band.rs`): asserted on the host class recorded in
//! [`CHARACTERIZED_ON`], computed and printed everywhere. The spec §3 floor and
//! the text-robustness floor are PORTABLE and stay unconditional, so this test
//! still gates on every host. A future re-conversion that changes the ANE band
//! REDs this gate on purpose on its characterization host, forcing a deliberate
//! re-characterization.

mod common;

use coremlit::{
  ComputeUnits,
  embeddings::siglip::{
    ImageEmbedder, ImageEmbedderOptions, Rgb8Image, TextEmbedder, TextEmbedderOptions,
  },
};

/// Spec §3's ship floor. PORTABLE — the vision `CpuAndGpu` arm is what this
/// crate ships, and it must clear this on every host.
const FLOOR: f32 = 0.99917;

/// Coarse "the text tower did not collapse on this arm" bound. PORTABLE, and
/// deliberately LOOSER than [`FLOOR`]: it is not a pinned measurement (the
/// measured values are ≈ 0.9998–0.99999, three orders of magnitude of headroom),
/// it is the shape of the claim "text is robust on every placement". A host
/// where an arm drops through this is a genuine finding worth a red light.
const TEXT_ROBUST_FLOOR: f32 = 0.999;

/// MEASURED band on the vision ANE arm's worst corpus cosine — the characterized
/// collapse, ≈ 0.31–0.41, pinned wide.
const VISION_ANE_LO: f32 = 0.20;
/// Upper edge of the vision-ANE collapse band. See [`VISION_ANE_LO`].
const VISION_ANE_HI: f32 = 0.70;
/// MEASURED non-vacuity ceiling on the vision ANE arm: on the characterizing
/// host the ANE is materially degraded, so the band cannot be satisfied by the
/// ANE arm silently being the `CpuAndGpu` arm. Implied by
/// `[VISION_ANE_LO, VISION_ANE_HI]` and kept beside it because it states the
/// INTENT the band exists to serve.
const VISION_ANE_NON_VACUITY: f32 = 0.9;
/// MEASURED tolerance for "vision `All` tracks the ANE": on the characterizing
/// host the planner dispatches the ~99%-ANE-preferred vision graph to the ANE,
/// so `All` inherits its numbers. Which arm a planner picks is a property of
/// that machine's compute set and compiler, so this rides the band gate too.
const ALL_TRACKS_ANE_TOL: f32 = 0.05;

/// The host class every MEASURED constant above was characterized on.
///
/// `None` — UNRECORDED, and deliberately left so. The 0.31–0.41 collapse was
/// measured on some machine and nothing in this repo records which; stamping a
/// plausible-looking host would fabricate provenance and hard-red every machine
/// that did not match the fiction. Until someone re-measures and records their
/// own host class, these bands are computed and printed on every host and
/// asserted on none — while [`FLOOR`] and [`TEXT_ROBUST_FLOOR`] keep gating
/// everywhere.
///
/// To arm them: run the command the band-gate banner prints, pin the printed
/// numbers, and set this to `Some(CharacterizedHost { .. })` describing the
/// machine that produced them.
const CHARACTERIZED_ON: Option<common::CharacterizedHost> = None;

/// This suite's source path, quoted into the re-characterization instructions.
const SOURCE_REL: &str = "coremlit/tests/siglip/placement.rs";

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

/// Characterize every placement arm's agreement with the goldens: assert the
/// portable floors on every host, and the pinned bands on the host class that
/// produced them.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn placement_arms_are_characterized() {
  // Opened BEFORE the measurement so the log leads with the verdict, the running
  // host class and — when the bands are not asserted — how to arm them.
  let gate = common::BandGate::open(
    "siglip placement",
    CHARACTERIZED_ON,
    common::recharacterize_command("siglip_placement", SOURCE_REL),
  );

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

  // ── Portable contracts: asserted on EVERY host, and asserted FIRST so a real
  // regression on the shipping arm is never reported behind a host-scoped band.
  //
  // Vision CpuAndGpu is the floor-holding ship arm.
  assert!(vg >= FLOOR, "vision CpuAndGpu {vg:.6} below floor {FLOOR}");
  // Text is robust on every arm.
  for (name, w) in &t {
    assert!(
      *w >= TEXT_ROBUST_FLOOR,
      "text arm {name} worst {w:.6} unexpectedly low (floor {TEXT_ROBUST_FLOOR})"
    );
  }

  // ── Measured bands: computed and printed on every host, asserted only on the
  // host class that produced them.
  //
  // Vision ANE is the CHARACTERIZED degraded arm (measured 0.31–0.41): a wide
  // band, and clearly below the floor.
  gate.check_band(
    "vision ANE worst — characterized collapse",
    va,
    VISION_ANE_LO,
    VISION_ANE_HI,
  );
  // Non-vacuity: the ANE arm must be materially degraded, i.e. NOT quietly the
  // same numbers as the CpuAndGpu arm. An ANE fix lands here and may allow All.
  gate.check_ceiling(
    "vision ANE worst — non-vacuity (must be materially below the floor, not GPU-identical)",
    va,
    VISION_ANE_NON_VACUITY,
  );
  // D1: All follows the ANE for vision (planner dispatch), so All is unsafe here.
  gate.check_ceiling(
    "vision |All − ANE| — All must track the ANE (planner dispatch, why the default is CpuAndGpu)",
    (vall - va).abs(),
    ALL_TRACKS_ANE_TOL,
  );
}
