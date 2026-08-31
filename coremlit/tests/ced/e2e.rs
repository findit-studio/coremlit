//! End-to-end gates on real audio — WAV in, pinned events out, per [`CedModel`]
//! size.
//!
//! # Status: Wave-C (model + fixtures gated, per size)
//!
//! The per-size gates (`tiny::`/`mini::`/`small::`/`base::`) are `#[ignore]`d
//! until the staged conversion + committed fixtures exist. Expectations are
//! MEASURED on this machine and pinned: a 440 Hz sine window ranks **Sine wave**
//! (AudioSet class 501) top-1 with a two-sided confidence band; a 15 s
//! sine-then-noise clip plans exactly two windows and its Mean/Max aggregate
//! keeps a tone class on top; and the prewarm path succeeds.

mod common;

use coremlit::audio::ced::{CedModel, ChunkAggregation, Classifier, ClassifierOptions, WindowPlan};

const SINE_CLASS: usize = 501; // AudioSet "Sine wave"
const SINE_NAME: &str = "Sine wave";
// Two-sided confidence band for the 440 Hz sine's top-1 across all four sizes
// (measured [0.8888, 0.9267] on Apple silicon, fp16 default compute; margin for
// fp16/OS drift — a shift outside this is a finding).
const SINE_CONF_LO: f32 = 0.80;
const SINE_CONF_HI: f32 = 0.98;

fn clip_wav(model: CedModel, file: &str) -> std::path::PathBuf {
  common::fixture_path(&format!("goldens/{}", model.as_str())).join(file)
}

/// Wave C: a 440 Hz sine window ranks Sine wave top-1 with a pinned confidence
/// band (default compute).
fn single_window_top_k(model: CedModel) {
  let corpus = common::load_golden_corpus(model);
  assert!(!corpus.clips.is_empty(), "goldens corpus must not be empty");
  let clf = Classifier::from_file(common::model_path(model)).unwrap();
  let wav = common::read_wav_16k_mono(&clip_wav(model, "../../mel/sine440_10s.wav"));
  let top = clf.classify(&wav, 5).unwrap();
  println!(
    "[e2e] {model} sine440 top-5: {:?}",
    top
      .iter()
      .map(|p| (p.index(), p.name(), p.confidence()))
      .collect::<Vec<_>>()
  );
  assert_eq!(
    top[0].index(),
    SINE_CLASS,
    "{model}: sine top-1 not Sine wave"
  );
  assert_eq!(top[0].name(), SINE_NAME);
  let c = top[0].confidence();
  assert!(
    (SINE_CONF_LO..=SINE_CONF_HI).contains(&c),
    "{model}: sine confidence {c:.4} outside [{SINE_CONF_LO}, {SINE_CONF_HI}]"
  );
}

/// Wave C: the 15 s clip plans exactly two windows; `classify_windows` count ==
/// `plan.spans` count; `classify_long` Mean/Max keep a tone class on top.
fn long_clip_rank(model: CedModel) {
  let clf = Classifier::from_file(common::model_path(model)).unwrap();
  let wav = common::read_wav_16k_mono(&clip_wav(model, "../clips/long_15s.wav"));
  let plan = WindowPlan::new();
  let n_spans = plan.spans(wav.len()).unwrap().len();
  assert_eq!(
    n_spans, 2,
    "{model}: 15 s clip must plan two 10 s windows, got {n_spans}"
  );

  let windows = clf.classify_windows(&wav, &plan).unwrap();
  assert_eq!(
    windows.len(),
    n_spans,
    "{model}: per-window count != plan spans"
  );

  let mean = clf
    .classify_long(&wav, 5, &plan, ChunkAggregation::Mean)
    .unwrap();
  let max = clf
    .classify_long(&wav, 5, &plan, ChunkAggregation::Max)
    .unwrap();
  println!(
    "[e2e] {model} long Mean top1={}({}) Max top1={}({})",
    mean[0].name(),
    mean[0].confidence(),
    max[0].name(),
    max[0].confidence()
  );
  // The first 10 s window is a pure 440 Hz sine, so a tone class dominates both
  // aggregates (Max keeps window-1's peak; Mean averages the two windows).
  assert_eq!(
    mean[0].index(),
    SINE_CLASS,
    "{model}: long Mean top-1 not Sine wave"
  );
  assert_eq!(
    max[0].index(),
    SINE_CLASS,
    "{model}: long Max top-1 not Sine wave"
  );
  assert_eq!(mean.len(), 5);
  assert_eq!(max.len(), 5);
}

/// Wave C: `prewarm` succeeds and a subsequent classify runs warm.
fn prewarm(model: CedModel) {
  let clf = Classifier::load(common::model_path(model), ClassifierOptions::new()).unwrap();
  clf.prewarm().unwrap();
  let wav = common::read_wav_16k_mono(&clip_wav(model, "../../mel/silence_2s.wav"));
  let out = clf.classify(&wav, 1).unwrap();
  assert_eq!(
    out.len(),
    1,
    "{model}: warm classify returned no prediction"
  );
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
