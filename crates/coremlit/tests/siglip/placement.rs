//! Compute-placement characterization (measured, never marketed).
//!
//! # Status: Wave C shell (model-gated)
//!
//! `#[ignore]`d until the conversion is staged (`SIGLIP_TEST_MODELS`). Wave C
//! records, per tower × `{CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All}`, the
//! cross-unit cosine vs the `CpuAndGpu` reference and vs the goldens (§7 arm
//! table). The `All` arm closes D1: record which device the planner picks for
//! vision under `All`; if it is GPU-identical and floor-holding the owner may
//! flip [`DEFAULT_IMAGE_COMPUTE`] to `All` — with the measurement in hand.

mod common;

use coremlit::ComputeUnits;

/// Characterize each placement arm's agreement with the `CpuAndGpu` reference
/// (and the goldens). Wave C measures and records the bands.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS) — Wave C"]
fn placement_arms_are_characterized() {
  let _dir = common::model_root();
  for _unit in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    // Wave C: embed the corpus per unit, record cosine vs the CpuAndGpu reference
    //         and vs the goldens; note the `All`-arm vision dispatch device (D1).
  }
}
