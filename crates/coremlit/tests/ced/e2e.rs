//! End-to-end gates on real audio — WAV in, pinned events out.
//!
//! # Status: Wave-C shell (model + fixtures gated)
//!
//! `#[ignore]`d until the staged conversion + committed fixtures exist.
//! Wave C pins (two-sided, from measurement): a real clip's top-k — expected
//! labels AND confidence bands; a long clip's window count + aggregate +
//! rank order; and the prewarm path.

mod common;

/// Wave C: real WAV → `classify` top-k: expected label set + per-label
/// confidence bands (two-sided).
#[test]
#[ignore = "requires staged CED model + fixtures (CED_TEST_MODELS) — Wave C"]
fn single_window_clip_yields_the_pinned_top_k() {
  let corpus = common::load_golden_corpus();
  assert!(!corpus.is_empty(), "goldens corpus must not be empty");
}

/// Wave C: multi-window WAV → `classify_windows` count == plan.spans count,
/// `classify_long` Mean vs Max ranked outputs pinned.
#[test]
#[ignore = "requires staged CED model + fixtures (CED_TEST_MODELS) — Wave C"]
fn long_clip_windows_aggregate_and_rank_as_pinned() {
  let corpus = common::load_golden_corpus();
  assert!(!corpus.is_empty(), "goldens corpus must not be empty");
}

/// Wave C: `Classifier::prewarm` succeeds and the next classify is warm.
#[test]
#[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave C"]
fn prewarm_smoke() {
  let _ = common::model_path();
}
