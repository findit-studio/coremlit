//! End-to-end language identification against a committed reference clip.
//!
//! # The anchor
//!
//! `fixtures/audio/udhr_th_16k.wav` is upstream's own `udhr_th.wav` — a Thai
//! reading of the Universal Declaration of Human Rights, shipped with
//! `speechbrain/lang-id-voxlingua107-ecapa` as that model card's worked example
//! — resampled 44.1 kHz -> 16 kHz mono. At 207 952 samples it is 12.997 s, and
//! `1 + 207952 / 160` = 1 300 mel frames, comfortably inside the graph's
//! `10..=3001`.
//!
//! Running the whole Rust front end plus the CoreML graph over it must put Thai
//! (`th`, model column 94) first, at a log probability of about -0.0101 — i.e.
//! the model is ~99% confident. That single number exercises every convention
//! this door had to get right at once: get the window, the padding, the
//! triangles, the dB floor, the layout or the fused mean subtraction wrong and
//! the answer moves or changes language.
//!
//! The gates are `#[ignore]`d and need the artifact staged (`LID_TEST_MODELS`,
//! default `Models/lid`); the fixture checks below are hermetic.

mod common;

use coremlit::{
  ComputeUnits,
  audio::lid::{Identifier, IdentifierOptions, Language, NUM_LANGUAGES, frame_count},
};

/// SHA-256 of the committed reference clip. Pinned so a re-encode — which would
/// still decode to plausible audio and still be Thai — cannot silently move the
/// expected log probability below.
const CLIP_SHA256: &str = "bf3b3ec7a039eed14de04cccec5cff682943111c3df82c8027acc3f3160c125e";

/// Samples in the committed clip, and the mel frames they produce.
const CLIP_SAMPLES: usize = 207_952;
const CLIP_FRAMES: usize = 1_300;

/// Model column of Thai in this door's roster.
const THAI_INDEX: usize = 94;

/// The reference top-1 log probability, from the artifact author's own probe of
/// this exact clip on `.all` / `.cpuAndGpu` (bit-identical there).
///
/// This crate reproduces it to ~1e-7 on the default placement, but the gate
/// below allows 0.01 on purpose: the same clip reads -0.010544 on
/// `.cpuAndNeuralEngine` and -0.015625 on `.cpuOnly`, so a tolerance tight
/// enough to pin the GPU arm alone would turn a placement change into a
/// mysterious numeric failure instead of the placement question it is.
/// `default_placement_agrees_with_the_gpu_arm_bit_for_bit` is what watches the
/// placement itself.
const THAI_LOG_PROBABILITY: f32 = -0.010_064;

fn clip() -> Vec<f32> {
  common::read_wav_16k_mono(&common::fixture_path("audio/udhr_th_16k.wav"))
}

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The committed clip is the exact bytes the anchor was measured on, with the
/// geometry the anchor assumes.
#[test]
fn reference_clip_is_the_pinned_bytes_and_geometry() {
  let path = common::fixture_path("audio/udhr_th_16k.wav");
  assert_eq!(common::sha256_file(&path), CLIP_SHA256, "clip drift");

  let samples = clip();
  assert_eq!(samples.len(), CLIP_SAMPLES);
  assert_eq!(frame_count(samples.len()), CLIP_FRAMES);
  assert!(samples.iter().all(|s| s.is_finite()));
  assert!(
    samples.iter().any(|s| s.abs() > 0.05),
    "the clip must actually carry speech, not silence"
  );

  // 12.997 s of audio: inside the door's 30 s ceiling with room to spare.
  let seconds = samples.len() as f64 / 16_000.0;
  assert!((seconds - 12.997).abs() < 1e-3, "{seconds} s");
}

/// The column the anchor names really is Thai, so a roster reorder would red
/// here rather than turning the gate below into a silent tautology.
#[test]
fn the_anchor_column_is_thai() {
  let thai = Language::from_index(THAI_INDEX).expect("Thai must be in the roster");
  assert_eq!(thai.code(), "th");
  assert_eq!(thai.name(), "Thai");
}

// ── Model-gated ─────────────────────────────────────────────────────────────

/// The end-to-end anchor: Thai first, at the reference log probability.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn reference_clip_identifies_as_thai() {
  let identifier = Identifier::from_file(common::model_path()).expect("load identifier");
  let ranked = identifier.identify(&clip(), 3).expect("identify");

  assert_eq!(ranked.len(), 3);
  assert_eq!(ranked[0].index(), THAI_INDEX, "top-1 must be Thai");
  assert_eq!(ranked[0].code(), "th");
  assert!(
    (ranked[0].log_probability() - THAI_LOG_PROBABILITY).abs() < 0.01,
    "top-1 log probability {} is not the reference {THAI_LOG_PROBABILITY}",
    ranked[0].log_probability()
  );
  assert!(
    ranked[0].probability() > 0.95,
    "the reference clip is a confident call, got {}",
    ranked[0].probability()
  );

  // Descending, and the runner-up is far behind.
  assert!(ranked[0].log_probability() > ranked[1].log_probability());
  assert!(ranked[1].log_probability() > ranked[2].log_probability());
  assert!(
    ranked[0].log_probability() - ranked[1].log_probability() > 3.0,
    "the top call must be decisive"
  );
}

/// The raw row really is a normalized natural-log distribution: every value is
/// `<= 0` and finite, and the exponentials sum to 1. Nothing in Rust applies a
/// softmax, so this is a statement about the graph.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn raw_row_is_an_already_normalized_log_distribution() {
  let identifier = Identifier::from_file(common::model_path()).expect("load identifier");
  let row = identifier.log_probabilities(&clip()).expect("scores");

  assert_eq!(row.len(), NUM_LANGUAGES);
  assert!(row.iter().all(|v| v.is_finite() && *v <= 0.0));

  let mass: f64 = row.iter().map(|v| f64::from(*v).exp()).sum();
  assert!(
    (mass - 1.0).abs() < 1e-3,
    "exp of the row must sum to 1, got {mass}"
  );

  // `identify` ranks that same row: its top-1 is the row's argmax.
  let argmax = (0..NUM_LANGUAGES)
    .max_by(|&a, &b| row[a].total_cmp(&row[b]))
    .expect("non-empty row");
  assert_eq!(argmax, THAI_INDEX);
}

/// `.all` and `.cpuAndGpu` agree bit for bit on this clip — the measurement the
/// module's `DEFAULT_COMPUTE` note rests on. If this ever diverges, `All` has
/// started dispatching somewhere else (the ANE arm is the pathological one) and
/// the placement note needs revisiting.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn default_placement_agrees_with_the_gpu_arm_bit_for_bit() {
  let samples = clip();
  let load = |compute| {
    Identifier::load(
      common::model_path(),
      IdentifierOptions::new().with_compute(compute),
    )
    .expect("load identifier")
  };
  let all = load(ComputeUnits::All)
    .log_probabilities(&samples)
    .expect("scores under All");
  let gpu = load(ComputeUnits::CpuAndGpu)
    .log_probabilities(&samples)
    .expect("scores under CpuAndGpu");
  assert_eq!(all, gpu, "`All` is expected to dispatch to the GPU here");
}

/// `prewarm` runs the whole path on synthetic audio, so a broken model shows up
/// at prewarm time rather than on the first real request.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn prewarm_exercises_the_prediction_path() {
  Identifier::from_file(common::model_path())
    .expect("load identifier")
    .prewarm()
    .expect("prewarm");
}
