//! Compute-placement **characterization** for the granite embedder — measured,
//! never marketed. Pins per-unit embedding agreement (cosine across placements
//! vs the `CpuOnly` reference) and documents the MEASURED placement reality.
//!
//! # Measured placement reality (T1)
//!
//! Unlike CLAP's audio (HTSAT) tower, the granite ModernBERT graph **does**
//! compile for the ANE — the T1 probe measured ~97.8% ANE residency and a fp16
//! cosine of 0.99996 vs a `CpuOnly` reference. This test characterizes that: it
//! never asserts residency, only that every placement agrees to fp16 tolerance.
//!
//! # What is pinned
//!
//! One deterministic input is embedded under every placement and the pairwise
//! cosine against the [`ComputeUnits::CpuOnly`] reference is held to a two-sided
//! band `[MIN, 1.0]`. Identical fp16 graph, different hardware ⇒ near-1
//! agreement; the lower bound is the measured worst case (measure-then-pin). A
//! drop below `MIN` means a placement changed the numerics materially — a
//! finding, not a threshold to loosen.

mod common;

use coremlit::{
  ComputeUnits,
  embeddings::granite::{Embedding, TextEmbedder, TextEmbedderOptions},
};

/// Lower bound on the cross-placement cosine over the full public matrix — `All`,
/// `CpuAndNeuralEngine`, `CpuAndGpu`, `CpuOnly` — each vs the `CpuOnly`
/// reference. MEASURED worst = 0.99998212 (2026-07-19; the CpuOnly-vs-ANE
/// fp16/fp32 pair); pinned at 0.9999 with a small fp16 margin. A drop below is a
/// finding.
const MIN_COSINE: f32 = 0.9999;

const UNITS: &[ComputeUnits] = &[
  ComputeUnits::All,
  ComputeUnits::CpuAndNeuralEngine,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuOnly,
];

fn assert_band(unit: ComputeUnits, cos: f32) {
  assert!(
    (MIN_COSINE..=1.0 + 1e-6).contains(&cos),
    "granite [{}] cosine vs CpuOnly = {cos:.8} outside [{MIN_COSINE}, 1.0] — a placement changed \
     the numerics materially (a finding, not a threshold to loosen)",
    unit.as_str()
  );
}

#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn placement_agreement_characterized() {
  const PROMPT: &str =
    "on-device text embeddings via CoreML on Apple silicon, multilingual and prompt-free";

  let embed = |unit: ComputeUnits| -> Embedding {
    TextEmbedder::load(
      common::model_path(),
      TextEmbedderOptions::new().with_compute(unit),
    )
    .unwrap_or_else(|e| panic!("load granite [{}]: {e}", unit.as_str()))
    .embed(PROMPT)
    .unwrap_or_else(|e| panic!("embed granite [{}]: {e}", unit.as_str()))
  };

  let reference = embed(ComputeUnits::CpuOnly);
  let mut worst = 1.0f32;
  for &unit in UNITS {
    let cos = embed(unit).cosine(&reference);
    worst = worst.min(cos);
    assert_band(unit, cos);
  }
  // Non-vacuity: the reference is a valid unit embedding.
  assert!((reference.cosine(&reference) - 1.0).abs() <= 1e-5);
  eprintln!("[placement] granite worst cross-unit cosine = {worst:.8}");
}
