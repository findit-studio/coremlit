//! End-to-end long-audio pipeline (model-gated): a multi-minute clip through
//! [`WindowPlan`] → [`AudioEncoder::embed_windows`] → an
//! [`AggregatePolicy`](coremlit::embeddings::clap::aggregate::AggregatePolicy) → zero-shot
//! [`score`](coremlit::embeddings::clap::score::score), with the window count, the aggregate
//! embedding's self-consistency, and the top label pinned.
//!
//! The clip is the committed JFK speech fixture tiled to ~192 s (a real
//! multi-minute input without committing a multi-megabyte WAV); every sample is
//! speech, so the aggregate stays coherent and the top zero-shot label is the
//! speech anchor.

mod common;

use coremlit::embeddings::clap::{
  AudioEncoder, MeanRenormalized, ScoreMode, TextAnchor, TextEncoder, WindowPlan,
  aggregate::{CoverageWeightedMean, aggregate},
  score,
};

/// Target clip length: 192.5 s at 48 kHz = 9 240 000 samples. With the default
/// no-overlap plan that is 19 full 10 s windows + a 120 000-sample tail = 20
/// windows (the tail's coverage is 0.25).
const TOTAL_SAMPLES: usize = 9_240_000;
const EXPECTED_WINDOWS: usize = 20;

// MEASURED-then-pinned (2026-07-18). The aggregate is not bit-stable across
// CoreML placements, so the DECISION (top label) is pinned exactly and the top
// score two-sided with margin; the aggregate's cosine to a constituent window is
// pinned loosely (homogeneous speech ⇒ coherent aggregate).
const TOP_LABEL: &str = "This is a sound of a person speaking";
const TOP_SCORE_LO: f32 = 8.3;
const TOP_SCORE_HI: f32 = 9.3;
const AGG_SELF_COSINE_LO: f32 = 0.97;

// int8 tier (opt-in): same DECISION (top label unchanged), top logit pinned in an
// int8-specific two-sided band around the measured 8.880292 (measured LOCALLY
// 2026-07-19 on artifact `clapkit-coreml@02a99c6a`; matches issue #30's 8.885357).
// The window geometry and aggregate self-consistency are tier-agnostic and reuse
// the fp16 pins above.
const TOP_SCORE_INT8_LO: f32 = 8.4;
const TOP_SCORE_INT8_HI: f32 = 9.4;

// The canonical CLAP zero-shot prompt template ("This is a sound of {label}",
// used by LAION/HF's own zero-shot-audio-classification examples). With bare
// nouns the audio↔text cosines are tiny and speech/music rank ambiguously; the
// template makes the contrast obvious, as measured below.
const ANCHORS: &[&str] = &[
  "This is a sound of a person speaking",
  "This is a sound of music",
  "This is a sound of a dog barking",
  "This is a sound of rain falling",
];

fn tiled_speech_clip() -> Vec<f32> {
  let jfk = common::read_wav_48k_mono(&common::fixture_path("audio/speech_jfk_48k.wav"));
  let mut clip = Vec::with_capacity(TOTAL_SAMPLES + jfk.len());
  while clip.len() < TOTAL_SAMPLES {
    clip.extend_from_slice(&jfk);
  }
  clip.truncate(TOTAL_SAMPLES);
  clip
}

/// Cosine of two unit-norm embeddings (== dot product), **fail-closed**: a
/// length mismatch (`zip` would silently truncate) or any non-finite input/output
/// panics rather than returning a wrong-but-finite / NaN value that a downstream
/// comparison could pass blindly. Same guard as the parity reducer
/// (`tests/clap/parity_textclap.rs`); the inputs are L2-normalized embeddings so
/// cosine == the raw dot product — that contract is unchanged.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(
    a.len(),
    b.len(),
    "cosine operands differ in length: {} vs {}",
    a.len(),
    b.len()
  );
  assert!(
    a.iter().chain(b).all(|v| v.is_finite()),
    "cosine operand contains a non-finite value (NaN/inf) — the reducer must fail closed"
  );
  let c: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
  assert!(c.is_finite(), "cosine produced a non-finite value: {c}");
  c
}

/// Fail-closed proof (HERMETIC — no model, runs in the default `cargo test` under
/// the `clap` feature): a `NaN` operand must PANIC, not slip through the
/// self-consistency check. Removing the finite guard reds this test.
#[test]
#[should_panic(expected = "non-finite")]
fn cosine_rejects_non_finite_operand() {
  let a = [f32::NAN, 0.0, 0.0];
  let b = [1.0, 0.0, 0.0];
  let _ = cosine(&a, &b);
}

/// Fail-closed proof (HERMETIC): mismatched operand lengths must PANIC rather
/// than let `zip` truncate to the shorter side. Removing the length assertion
/// reds this test.
#[test]
#[should_panic(expected = "differ in length")]
fn cosine_rejects_length_mismatch() {
  let a = [1.0, 0.0, 0.0];
  let b = [1.0, 0.0];
  let _ = cosine(&a, &b);
}

#[test]
#[ignore = "requires clapkit models (CLAPKIT_TEST_MODELS)"]
fn multi_minute_pipeline_pins_windows_aggregate_and_top_label() {
  let audio = AudioEncoder::from_file(common::audio_model_path()).unwrap();
  let text = TextEncoder::from_file(common::text_model_path()).unwrap();
  run_pipeline_and_pin(&audio, &text, "fp16", TOP_SCORE_LO, TOP_SCORE_HI);
}

/// int8 tier (opt-in): the SAME multi-minute pipeline with the int8 encoders. The
/// DECISION (top zero-shot label) must be unchanged and the top logit lands in the
/// int8-specific band; window geometry / aggregate coherence are tier-agnostic.
#[test]
#[ignore = "requires clapkit int8 models (CLAPKIT_TEST_MODELS)"]
fn multi_minute_pipeline_int8_pins_top_label() {
  let audio = AudioEncoder::from_file(common::audio_model_int8_path()).unwrap();
  let text = TextEncoder::from_file(common::text_model_int8_path()).unwrap();
  run_pipeline_and_pin(&audio, &text, "int8", TOP_SCORE_INT8_LO, TOP_SCORE_INT8_HI);
}

/// The documented prewarm/reuse path (issue #30 perf pass): constructing an
/// encoder pays the load, [`AudioEncoder::prewarm`] / [`TextEncoder::prewarm`]
/// absorb the first-inference specialization, and the SAME encoder is then reused
/// for real requests. Pins that prewarm succeeds on both towers and leaves each
/// encoder fully usable (a real `embed*` after prewarm returns a valid unit
/// embedding).
#[test]
#[ignore = "requires clapkit models (CLAPKIT_TEST_MODELS)"]
fn prewarm_then_reuse_both_towers() {
  let audio = AudioEncoder::from_file(common::audio_model_path()).unwrap();
  let text = TextEncoder::from_file(common::text_model_path()).unwrap();

  audio.prewarm().expect("audio prewarm");
  text.prewarm().expect("text prewarm");

  // Reuse after prewarm: both encoders still produce valid unit-norm embeddings.
  let window = common::deterministic_window(coremlit::embeddings::clap::audio::TARGET_SAMPLES);
  let a = audio
    .embed_window(&window)
    .expect("audio embed after prewarm");
  let t = text
    .embed("a violin playing a slow melody")
    .expect("text embed after prewarm");
  assert!(
    (a.cosine(&a) - 1.0).abs() <= 1e-5,
    "audio embedding is unit-norm after prewarm"
  );
  assert!(
    (t.cosine(&t) - 1.0).abs() <= 1e-5,
    "text embedding is unit-norm after prewarm"
  );
}

/// Drive the full long-audio pipeline for one tier and pin the geometry, the
/// aggregate self-consistency, the top zero-shot label, and the top logit band.
fn run_pipeline_and_pin(
  audio: &AudioEncoder,
  text: &TextEncoder,
  tier: &str,
  top_score_lo: f32,
  top_score_hi: f32,
) {
  let clip = tiled_speech_clip();

  // 1. Plan → 2. per-window embeddings (always exposed to the caller).
  let plan = WindowPlan::new(); // no overlap, keep the padded tail
  let windows = audio.embed_windows(&clip, &plan).unwrap();
  println!(
    "[e2e/{tier}] {} windows over {} samples",
    windows.len(),
    clip.len()
  );
  assert_eq!(
    windows.len(),
    plan.spans(clip.len()).len(),
    "embed_windows count must match the plan geometry"
  );
  assert_eq!(
    windows.len(),
    EXPECTED_WINDOWS,
    "pinned window count drifted"
  );
  // The final window is a padded tail; interiors are full coverage.
  assert_eq!(windows[0].span().coverage(), 1.0);
  assert!((windows[EXPECTED_WINDOWS - 1].span().coverage() - 0.25).abs() < 1e-6);

  // 3. Aggregate (the shipped default policy) into one clip embedding.
  let clip_embedding = aggregate(&MeanRenormalized, &windows).unwrap();
  // Aggregate is unit-norm and — since every window is speech — stays close to
  // any single window.
  let norm_sq: f32 = clip_embedding.as_slice().iter().map(|x| x * x).sum();
  assert!((norm_sq - 1.0).abs() < 1e-5, "aggregate not unit-norm");
  let self_cos = cosine(clip_embedding.as_slice(), windows[0].value().as_slice());
  println!("[e2e/{tier}] aggregate↔window[0] cosine = {self_cos:.6}");
  assert!(
    self_cos >= AGG_SELF_COSINE_LO,
    "aggregate {self_cos:.6} diverged from its windows (below {AGG_SELF_COSINE_LO})"
  );
  // A second built-in produces a valid, near-identical aggregate on homogeneous
  // audio (exercises the coverage path end to end).
  let cov_embedding = aggregate(&CoverageWeightedMean, &windows).unwrap();
  assert!(cov_embedding.is_close_cosine(&clip_embedding, 1e-3));

  // 4. Zero-shot score the aggregate against the label anchors.
  let anchor_embeddings: Vec<_> = ANCHORS.iter().map(|p| text.embed(p).unwrap()).collect();
  let anchors: Vec<TextAnchor<'_>> = ANCHORS
    .iter()
    .zip(anchor_embeddings.iter())
    .map(|(label, emb)| TextAnchor::new(label, emb))
    .collect();
  let ranked = score(&clip_embedding, &anchors, ScoreMode::LogitScaled);
  for r in &ranked {
    println!("[e2e/{tier}] {:<30} logit = {:.6}", r.label(), r.score());
  }

  assert_eq!(ranked[0].label(), TOP_LABEL, "top zero-shot label drifted");
  let top = ranked[0].score();
  assert!(
    (top_score_lo..=top_score_hi).contains(&top),
    "top logit {top:.6} outside pinned band [{top_score_lo}, {top_score_hi}]"
  );
  // The decision is unambiguous: the speech anchor beats the runner-up.
  assert!(
    ranked[0].score() > ranked[1].score(),
    "speech anchor did not win outright"
  );
}
