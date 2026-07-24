//! The CED logits parity SHIP GATE — CoreML logits vs the committed CED ONNX
//! fp32 CPU goldens (generated owner-side; ort never enters this repo, not
//! even dev).
//!
//! # Status: Wave-B shell (model + goldens gated)
//!
//! `#[ignore]`d until the owner stages the conversion (`CED_TEST_MODELS`) and
//! commits `fixtures/goldens/corpus.json` (Wave B). Wave B measures on this
//! machine and pins the bands (spec §7): the fp32-CPU arm is PRIMARY
//! (measured-then-pinned two-sided band on max |Δlogit| + cosine + top-10
//! set/rank agreement); the default-compute fp16 arm is CHARACTERIZED
//! separately in its own measured band (bands never loosened — a shift in
//! either direction is a finding). Negative controls: a non-vacuity ceiling
//! (mismatched clip↔golden pairs score far apart) + mutation reds.

mod common;

/// PRIMARY: fp32-CPU-arm logits vs the committed oracle, per corpus clip.
/// Wave B: `Classifier::load(model_path, CpuOnly)`, `read_wav_16k_mono` +
/// `raw_scores` per entry (sub-window entries exercise the tail-padding
/// semantics), then max |Δlogit| + `common::cosine_checked` + top-10 set/rank
/// agreement, asserted against the measured-then-pinned two-sided band.
#[test]
#[ignore = "requires staged CED model + committed goldens (CED_TEST_MODELS) — Wave B"]
fn fp32_cpu_arm_holds_the_logit_parity_band() {
  let corpus = common::load_golden_corpus();
  assert!(!corpus.is_empty(), "goldens corpus must not be empty");
}

/// CHARACTERIZED: the default-compute fp16 arm in its own measured band —
/// never floor-gated against the fp32 band (measured, never marketed).
/// Wave B: same metric sweep under `ClassifierOptions::new()`, band pinned
/// from measurement.
#[test]
#[ignore = "requires staged CED model + committed goldens (CED_TEST_MODELS) — Wave B"]
fn default_compute_arm_is_characterized() {
  let corpus = common::load_golden_corpus();
  assert!(!corpus.is_empty(), "goldens corpus must not be empty");
}
