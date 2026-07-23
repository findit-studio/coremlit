//! The siglip parity SHIP GATE — CoreML embeddings vs the committed
//! transformers-fp32 goldens, per compute unit.
//!
//! # Status: Wave C shell (model-gated)
//!
//! `#[ignore]`d until the owner stages the conversion (`SIGLIP_TEST_MODELS`) and
//! the committed goldens (Wave B). Wave C measures on this machine and pins the
//! bands (§7): the `CpuAndGpu` arm is THE GATE (floor never below 0.99917;
//! expected pin ≈ 0.9998 — probe worst 0.999959 vision / 0.999998 text), with the
//! ANE / CpuOnly arms CHARACTERIZED (not floor-gated — the ANE misses the floor
//! by design). Non-vacuity mutations: zeroed `position_embeddings` (proves the
//! pos-emb lift is load-bearing), mask off-by-one, rotated output slice.

mod common;

/// The `CpuAndGpu` GATE: worst-corpus cosine vs the fp32 goldens holds the floor.
/// Wave C measures and pins the two-sided band.
#[test]
#[ignore = "requires staged siglip models + committed goldens — Wave C"]
fn cpu_and_gpu_arm_holds_the_parity_floor() {
  let (_images, _texts) = common::golden_corpus();
  // Wave C: embed each corpus image/text on CpuAndGpu, cosine vs golden
  //         (common::cosine_checked), assert worst >= pinned floor (>= 0.99917).
}

/// Non-vacuity: zeroing `position_embeddings` (the port's central novel step)
/// must collapse vision parity far below the floor. Wave C implements the
/// raw-pipeline mutation harness.
#[test]
#[ignore = "requires staged siglip models + committed goldens — Wave C"]
fn zeroed_position_embeddings_break_the_gate() {
  // Wave C: prove the pos-emb lift is load-bearing.
}
