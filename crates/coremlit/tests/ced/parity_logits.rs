//! The CED logits parity SHIP GATE — CoreML logits vs the committed CED PyTorch
//! fp32 CPU goldens (the exact pre-sigmoid of the unmodified
//! `CedForAudioClassification` forward — `model.safetensors`, `CedMelToLogits`,
//! `conversion/ced`; the shipped `model.onnx` is post-sigmoid so it cannot be
//! the pre-sigmoid oracle; ort never enters this repo, not even dev), per
//! [`CedModel`] size.
//!
//! # Status: Wave-B (model + goldens gated, per size)
//!
//! The per-size gates (`tiny::`/`mini::`/`small::`/`base::`) are `#[ignore]`d
//! until the owner stages that size's conversion (`CED_TEST_MODELS`) and commits
//! `fixtures/goldens/<size>/corpus.json` (Wave B). The bands below were MEASURED
//! on this machine (Apple silicon) against the shipped fp16 artifacts and pinned
//! with margin — never loosened; a shift in either direction is a finding.
//!
//! The **CpuOnly** arm is PRIMARY: the shipped graph is fp16, but CoreML computes
//! it in ~fp32 on the CPU, making this the deterministic REFERENCE arm vs the
//! PyTorch fp32 goldens (residual = the Rust f64 mel vs the torchaudio f32 mel +
//! the fp16 weight quantization) — reference, not tightest: the measured envelope
//! shows the **default-compute** (`All` ⇒ ANE/GPU) arm actually lands closer to
//! the goldens (worst cos 0.99999988 vs CpuOnly's 0.99999803), characterized in
//! its own band below. Negative control: a mismatched clip↔golden pair scores
//! far below the matched floor.

mod common;

use coremlit::{
  ComputeUnits,
  audio::ced::{CedModel, Classifier, ClassifierOptions, NUM_CLASSES},
};

use common::GoldenClip;

// --- MEASURED-then-pinned bands (Apple silicon, Wave B) -----------------------------------------
// Measured envelope over all four sizes × the 4-clip corpus:
//   CpuOnly  : cos ∈ [0.99999094, 0.99999803], max|Δlogit| ≤ 0.1573
//   default  : cos ∈ [0.99999988, 0.99999994], max|Δlogit| ≤ 0.0296
//   mismatch : cos(sine440 logits, silence golden) ≤ 0.9886
// max|Δlogit| is a single-outlier-logit metric under fp16 (top-1/top-10 stay exact and cosine
// stays ~1.0), so its ceiling carries ~2× margin; cosine is the tight, meaningful floor.
// CpuOnly (PRIMARY): the fp16 graph computed on the CPU vs the PyTorch fp32 goldens.
const CPU_COS_FLOOR: f32 = 0.9999;
const CPU_MAXABS_CEIL: f32 = 0.30;
// default-compute (All ⇒ ANE/GPU): true fp16, CHARACTERIZED in its own wider band.
const DEFAULT_COS_FLOOR: f32 = 0.999;
const DEFAULT_MAXABS_CEIL: f32 = 0.10;
// Negative control: a mismatched clip↔golden pair must fall below this (measured ≤ 0.9886).
const MISMATCH_COS_CEIL: f32 = 0.99;

/// Resolve a corpus clip's committed WAV (its `file` is relative to the per-size
/// `fixtures/goldens/<size>/` corpus dir).
fn clip_wav(model: CedModel, clip: &GoldenClip) -> std::path::PathBuf {
  common::fixture_path(&format!("goldens/{}", model.as_str())).join(&clip.file)
}

/// Indices of the top `k` scores, descending, ties broken by ascending index
/// (the soundevents contract).
fn top_k_idx(v: &[f32], k: usize) -> Vec<usize> {
  let mut idx: Vec<usize> = (0..v.len()).collect();
  idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
  idx.truncate(k);
  idx
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
  a.iter()
    .zip(b)
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max)
}

/// Shared arm core: score every corpus clip on `compute` and assert the pinned
/// band vs the committed goldens (cosine floor + max|Δlogit| ceiling + top-1
/// index + top-10 overlap), then the mismatched-pair negative control.
fn parity_arm(
  model: CedModel,
  compute: ComputeUnits,
  cos_floor: f32,
  maxabs_ceil: f32,
  label: &str,
) {
  let corpus = common::load_golden_corpus(model);
  assert!(!corpus.clips.is_empty(), "goldens corpus must not be empty");
  let clf = Classifier::load(
    common::model_path(model),
    ClassifierOptions::new().with_compute(compute),
  )
  .unwrap_or_else(|e| panic!("load {model} on {compute:?}: {e}"));

  let mut produced: Vec<Vec<f32>> = Vec::new();
  let mut worst_cos = 1.0f32;
  let mut worst_maxabs = 0.0f32;
  for clip in &corpus.clips {
    assert_eq!(
      clip.logits.len(),
      NUM_CLASSES,
      "golden {} must be [527]",
      clip.id
    );
    let wav = common::read_wav_16k_mono(&clip_wav(model, clip));
    assert_eq!(
      wav.len(),
      clip.n_samples,
      "{}: decoded sample count",
      clip.id
    );
    let got = clf
      .raw_scores(&wav)
      .unwrap_or_else(|e| panic!("raw_scores {}: {e}", clip.id));

    let cos = common::cosine_checked(&got, &clip.logits);
    let maxabs = max_abs(&got, &clip.logits);
    let (tg, tgold) = (top_k_idx(&got, 10), top_k_idx(&clip.logits, 10));
    let overlap = tg.iter().filter(|i| tgold.contains(i)).count();
    println!(
      "[{label}] {m} {id:14} cos={cos:.8} max|Δ|={maxabs:.6} top1={t1}(gold {g1}) overlap={overlap}/10",
      m = model,
      id = clip.id,
      t1 = tg[0],
      g1 = tgold[0]
    );
    assert_eq!(
      tg[0], tgold[0],
      "[{label}] {model}/{}: top-1 mismatch",
      clip.id
    );
    assert!(
      overlap >= 9,
      "[{label}] {model}/{}: top-10 overlap {overlap}/10",
      clip.id
    );

    worst_cos = worst_cos.min(cos);
    worst_maxabs = worst_maxabs.max(maxabs);
    produced.push(got);
  }
  println!("[{label}] {model} worst_cos={worst_cos:.8} worst_maxabs={worst_maxabs:.6}");
  assert!(
    worst_cos >= cos_floor,
    "[{label}] {model}: worst cosine {worst_cos:.8} < floor {cos_floor}"
  );
  assert!(
    worst_maxabs <= maxabs_ceil,
    "[{label}] {model}: worst max|Δlogit| {worst_maxabs:.6} > ceil {maxabs_ceil}"
  );

  // Negative control (non-vacuity): the two most distinct clips (a 440 Hz sine
  // vs 2 s of silence) must cross-score FAR below the matched floor, proving the
  // gate is not vacuously passing identical vectors. Both ids are REQUIRED in
  // every committed corpus — assert their presence rather than silently skipping
  // the check, which would let the gate go vacuous without anyone noticing.
  let find = |id: &str| {
    corpus
      .clips
      .iter()
      .position(|c| c.id == id)
      .unwrap_or_else(|| {
        panic!("[{label}] {model}: golden corpus missing required negative-control clip `{id}`")
      })
  };
  let (si, zi) = (find("sine440_10s"), find("silence_2s"));
  let cross = common::cosine_checked(&produced[si], &corpus.clips[zi].logits);
  println!("[{label}] {model} non-vacuity cos(sine440 logits, silence golden)={cross:.6}");
  assert!(
    cross < MISMATCH_COS_CEIL,
    "[{label}] {model}: mismatched pair cos {cross:.6} not < {MISMATCH_COS_CEIL} (vacuous gate?)"
  );
}

/// PRIMARY: the CpuOnly arm holds the tight logit-parity band vs the PyTorch fp32
/// goldens.
fn fp32_cpu_arm(model: CedModel) {
  parity_arm(
    model,
    ComputeUnits::CpuOnly,
    CPU_COS_FLOOR,
    CPU_MAXABS_CEIL,
    "cpu",
  );
}

/// CHARACTERIZED: the default-compute (fp16, ANE/GPU) arm in its own wider band.
fn default_compute_arm(model: CedModel) {
  parity_arm(
    model,
    ClassifierOptions::new().compute(),
    DEFAULT_COS_FLOOR,
    DEFAULT_MAXABS_CEIL,
    "default",
  );
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
