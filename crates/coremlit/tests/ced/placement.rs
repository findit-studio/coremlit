//! Placement characterization for the CED graph — per-compute-unit agreement
//! and warm latency, CHARACTERIZED, never asserted (measured, never
//! marketed).
//!
//! # Status: Wave-C shell (model-gated)
//!
//! `#[ignore]`d until the staged conversion exists. Wave C: per unit in
//! {CpuOnly, CpuAndGpu, CpuAndNeuralEngine, All} — logit agreement vs the
//! CpuOnly reference (cosine + max |Δ|), NaN scan, and a warm-latency table;
//! the results drive the `DEFAULT_COMPUTE` re-pin (the module ships `All` as
//! PROVISIONAL) and land in the module docs as the measured
//! latency × placement table.

mod common;

/// Wave C: the per-unit characterization matrix (see the module comment).
#[test]
#[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave C"]
fn per_unit_agreement_and_latency_are_characterized() {
  // Wave C: load per unit via ClassifierOptions::with_compute, sweep the
  // golden corpus, record agreement + latency, pin the characterized bands.
  let _ = common::model_path();
}
