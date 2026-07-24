//! End-to-end gates on real audio — WAV in, pinned events out, per [`CedModel`]
//! size.
//!
//! # Status: Wave-C shell (model + fixtures gated, per size)
//!
//! The per-size gates (`tiny::`/`mini::`/`small::`/`base::`) are `#[ignore]`d
//! until the staged conversion + committed fixtures exist. Wave C pins
//! (two-sided, from measurement): a real clip's top-k — expected labels AND
//! confidence bands; a long clip's window count + aggregate + rank order; and
//! the prewarm path.

mod common;

use coremlit::audio::ced::CedModel;

/// Wave C: real WAV → `classify` top-k: expected label set + per-label
/// confidence bands (two-sided).
fn single_window_top_k(model: CedModel) {
  let corpus = common::load_golden_corpus(model);
  assert!(!corpus.clips.is_empty(), "goldens corpus must not be empty");
}

/// Wave C: multi-window WAV → `classify_windows` count == plan.spans count,
/// `classify_long` Mean vs Max ranked outputs pinned.
fn long_clip_rank(model: CedModel) {
  let corpus = common::load_golden_corpus(model);
  assert!(!corpus.clips.is_empty(), "goldens corpus must not be empty");
}

/// Wave C: `Classifier::prewarm` succeeds and the next classify is warm.
fn prewarm(model: CedModel) {
  let _ = common::model_path(model);
}

macro_rules! per_model_gates {
  ($($m:ident => $v:expr),+ $(,)?) => {$(
    mod $m {
      use super::CedModel;

      #[test]
      #[ignore = "requires staged CED model + fixtures (CED_TEST_MODELS) — Wave C"]
      fn single_window_clip_yields_the_pinned_top_k() {
        super::single_window_top_k($v);
      }

      #[test]
      #[ignore = "requires staged CED model + fixtures (CED_TEST_MODELS) — Wave C"]
      fn long_clip_windows_aggregate_and_rank_as_pinned() {
        super::long_clip_rank($v);
      }

      #[test]
      #[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave C"]
      fn prewarm_smoke() {
        super::prewarm($v);
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
