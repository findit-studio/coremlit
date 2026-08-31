//! Placement characterization for the CED graph — per-compute-unit agreement
//! and warm latency, CHARACTERIZED, never marketed, per [`CedModel`] size.
//!
//! # Status: Wave-C (model-gated, per size)
//!
//! The per-size gates (`tiny::`/`mini::`/`small::`/`base::`) are `#[ignore]`d
//! until the staged conversion exists. Per unit in {CpuOnly, CpuAndGpu,
//! CpuAndNeuralEngine, All}: logit agreement vs the CpuOnly reference (cosine +
//! max|Δ|), a NaN scan, and a warm-latency line. Agreement is asserted only
//! against a WIDE sanity floor (measured, never marketed) and NaN-freedom is
//! hard; latency is printed for the `DEFAULT_COMPUTE` decision (the module ships
//! `All`, MEASURED here to hold the parity floor on every arm — CED, unlike
//! siglip, runs the whole transformer natively, so the ANE arm is not demoted).

mod common;

use std::time::Instant;

use coremlit::{
  ComputeUnits,
  audio::ced::{CedModel, Classifier, ClassifierOptions},
};

const UNITS: [(ComputeUnits, &str); 4] = [
  (ComputeUnits::CpuOnly, "CpuOnly"),
  (ComputeUnits::CpuAndGpu, "CpuAndGpu"),
  (ComputeUnits::CpuAndNeuralEngine, "CpuAndNeuralEngine"),
  (ComputeUnits::All, "All"),
];
/// Wide sanity floor: every arm must agree with the CpuOnly reference at least
/// this well (true fp16 on the ANE still clears it — measured ~0.99999).
const SANITY_COS: f32 = 0.99;

fn clip_wav(model: CedModel, file: &str) -> std::path::PathBuf {
  common::fixture_path(&format!("goldens/{}", model.as_str())).join(file)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
  a.iter()
    .zip(b)
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max)
}

fn characterize(model: CedModel) {
  let corpus = common::load_golden_corpus(model);
  assert!(!corpus.clips.is_empty(), "goldens corpus must not be empty");
  let wavs: Vec<Vec<f32>> = corpus
    .clips
    .iter()
    .map(|c| common::read_wav_16k_mono(&clip_wav(model, &c.file)))
    .collect();

  // CpuOnly reference logits.
  let cpu = Classifier::load(
    common::model_path(model),
    ClassifierOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .unwrap_or_else(|e| panic!("load {model} CpuOnly: {e}"));
  let refs: Vec<Vec<f32>> = wavs.iter().map(|w| cpu.raw_scores(w).unwrap()).collect();

  for (unit, name) in UNITS {
    let clf = Classifier::load(
      common::model_path(model),
      ClassifierOptions::new().with_compute(unit),
    )
    .unwrap_or_else(|e| panic!("load {model} {name}: {e}"));
    clf.prewarm().unwrap();
    let mut worst_cos = 1.0f32;
    let mut worst_maxabs = 0.0f32;
    let mut nan = false;
    let t0 = Instant::now();
    for (w, r) in wavs.iter().zip(&refs) {
      let got = clf.raw_scores(w).unwrap();
      nan |= got.iter().any(|v| !v.is_finite());
      worst_cos = worst_cos.min(common::cosine_checked(&got, r));
      worst_maxabs = worst_maxabs.max(max_abs(&got, r));
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / wavs.len() as f64;
    println!(
      "[placement] {model} {name:18} cos_vs_cpu={worst_cos:.8} max|Δ|={worst_maxabs:.6} nan={nan} warm~{ms:.2}ms/clip"
    );
    assert!(!nan, "{model} {name}: produced a non-finite logit");
    assert!(
      worst_cos >= SANITY_COS,
      "{model} {name}: cos {worst_cos:.8} < sanity {SANITY_COS}"
    );
  }
}

macro_rules! per_model_gates {
  ($($m:ident => $v:expr),+ $(,)?) => {$(
    mod $m {
      use super::CedModel;

      #[test]
      #[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave C"]
      fn per_unit_agreement_and_latency_are_characterized() {
        super::characterize($v);
      }
    }
  )+};
}

per_model_gates!(
  tiny => CedModel::Tiny,
  mini => CedModel::Mini,
  small => CedModel::Small,
  base => CedModel::Base,
);
