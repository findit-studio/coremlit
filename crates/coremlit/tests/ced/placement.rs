//! Placement characterization for the CED graph — per-compute-unit agreement
//! and warm latency, CHARACTERIZED, never asserted (measured, never marketed),
//! per [`CedModel`] size.
//!
//! # Status: Wave-C shell (model-gated, per size)
//!
//! The per-size gates (`tiny::`/`mini::`/`small::`/`base::`) are `#[ignore]`d
//! until the staged conversion exists. Wave C: per unit in {CpuOnly, CpuAndGpu,
//! CpuAndNeuralEngine, All} — logit agreement vs the CpuOnly reference (cosine +
//! max |Δ|), NaN scan, and a warm-latency table; the results drive the
//! `DEFAULT_COMPUTE` re-pin (the module ships `All` as PROVISIONAL) and land in
//! the module docs as the measured latency × placement table. After all four
//! sizes: one shared winner re-pins `DEFAULT_COMPUTE`; divergent winners exercise
//! the pre-declared `CedModel::default_compute()` seam.

mod common;

use coremlit::audio::ced::CedModel;

/// Wave C: `model`'s per-unit characterization matrix (see the module comment).
fn characterize(model: CedModel) {
  // Wave C: load per unit via ClassifierOptions::with_compute, sweep the golden
  // corpus, record agreement + latency, pin the characterized bands.
  let _ = common::model_path(model);
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
