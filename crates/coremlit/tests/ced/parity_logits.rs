//! The CED logits parity SHIP GATE — CoreML logits vs the committed CED ONNX
//! fp32 CPU goldens (generated owner-side; ort never enters this repo, not even
//! dev), per [`CedModel`] size.
//!
//! # Status: Wave-B shell (model + goldens gated, per size)
//!
//! The per-size gates (`tiny::`/`mini::`/`small::`/`base::`) are `#[ignore]`d
//! until the owner stages that size's conversion (`CED_TEST_MODELS`) and commits
//! `fixtures/goldens/<size>/corpus.json` (Wave B). Wave B measures on this
//! machine and pins the bands PER SIZE (spec §7): the fp32-CPU arm is PRIMARY
//! (measured-then-pinned two-sided band on max |Δlogit| + cosine + top-10
//! set/rank agreement); the default-compute fp16 arm is CHARACTERIZED separately
//! in its own measured band. Bands are never shared and never loosened — a shift
//! in either direction on any size is a finding. Negative controls: a
//! non-vacuity ceiling (mismatched clip↔golden pairs score far apart) + mutation
//! reds.

mod common;

use coremlit::audio::ced::CedModel;

/// PRIMARY core: `model`'s fp32-CPU-arm logits vs its committed oracle, per
/// corpus clip. Wave B: `Classifier::load(model_path, CpuOnly)`,
/// `read_wav_16k_mono` + `raw_scores` per entry (sub-window entries exercise the
/// tail-padding semantics), then max |Δlogit| + `common::cosine_checked` +
/// top-10 set/rank agreement, asserted against the measured-then-pinned
/// two-sided band.
fn fp32_cpu_arm(model: CedModel) {
  let corpus = common::load_golden_corpus(model);
  assert!(!corpus.clips.is_empty(), "goldens corpus must not be empty");
}

/// CHARACTERIZED core: `model`'s default-compute fp16 arm in its own measured
/// band — never floor-gated against the fp32 band (measured, never marketed).
/// Wave B: same metric sweep under `ClassifierOptions::new()`, band pinned from
/// measurement.
fn default_compute_arm(model: CedModel) {
  let corpus = common::load_golden_corpus(model);
  assert!(!corpus.clips.is_empty(), "goldens corpus must not be empty");
}

macro_rules! per_model_gates {
  ($($m:ident => $v:expr),+ $(,)?) => {$(
    mod $m {
      use super::CedModel;

      #[test]
      #[ignore = "requires staged CED model + committed goldens (CED_TEST_MODELS) — Wave B"]
      fn fp32_cpu_arm_holds_the_logit_parity_band() {
        super::fp32_cpu_arm($v);
      }

      #[test]
      #[ignore = "requires staged CED model + committed goldens (CED_TEST_MODELS) — Wave B"]
      fn default_compute_arm_is_characterized() {
        super::default_compute_arm($v);
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
