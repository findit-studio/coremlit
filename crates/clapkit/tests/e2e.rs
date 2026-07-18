//! End-to-end long-audio pipeline (model-gated): a multi-minute clip through
//! [`WindowPlan`] → [`AudioEncoder::embed_windows`] → an
//! [`AggregatePolicy`](clapkit::aggregate::AggregatePolicy) → zero-shot
//! [`score`](clapkit::score::score), with the window count, the aggregate
//! embedding's self-consistency, and the top label pinned.
//!
//! The clip is the committed JFK speech fixture tiled to ~192 s (a real
//! multi-minute input without committing a multi-megabyte WAV); every sample is
//! speech, so the aggregate stays coherent and the top zero-shot label is the
//! speech anchor.

mod common;

use clapkit::{
  AudioEncoder, MeanRenormalized, ScoreMode, TextAnchor, TextEncoder, WindowPlan,
  aggregate::{AggregatePolicy, CoverageWeightedMean},
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
  a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "requires clapkit models (CLAPKIT_TEST_MODELS)"]
fn multi_minute_pipeline_pins_windows_aggregate_and_top_label() {
  let audio = AudioEncoder::from_file(common::audio_model_path()).unwrap();
  let text = TextEncoder::from_file(common::text_model_path()).unwrap();
  let clip = tiled_speech_clip();

  // 1. Plan → 2. per-window embeddings (always exposed to the caller).
  let plan = WindowPlan::new(); // no overlap, keep the padded tail
  let windows = audio.embed_windows(&clip, &plan).unwrap();
  println!(
    "[e2e] {} windows over {} samples",
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
  assert_eq!(windows[0].coverage(), 1.0);
  assert!((windows[EXPECTED_WINDOWS - 1].coverage() - 0.25).abs() < 1e-6);

  // 3. Aggregate (the shipped default policy) into one clip embedding.
  let clip_embedding = MeanRenormalized.aggregate(&windows).unwrap();
  // Aggregate is unit-norm and — since every window is speech — stays close to
  // any single window.
  let norm_sq: f32 = clip_embedding.as_slice().iter().map(|x| x * x).sum();
  assert!((norm_sq - 1.0).abs() < 1e-5, "aggregate not unit-norm");
  let self_cos = cosine(clip_embedding.as_slice(), windows[0].embedding().as_slice());
  println!("[e2e] aggregate↔window[0] cosine = {self_cos:.6}");
  assert!(
    self_cos >= AGG_SELF_COSINE_LO,
    "aggregate {self_cos:.6} diverged from its windows (below {AGG_SELF_COSINE_LO})"
  );
  // A second built-in produces a valid, near-identical aggregate on homogeneous
  // audio (exercises the coverage path end to end).
  let cov_embedding = CoverageWeightedMean.aggregate(&windows).unwrap();
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
    println!("[e2e] {:<30} logit = {:.6}", r.label(), r.score());
  }

  assert_eq!(ranked[0].label(), TOP_LABEL, "top zero-shot label drifted");
  let top = ranked[0].score();
  assert!(
    (TOP_SCORE_LO..=TOP_SCORE_HI).contains(&top),
    "top logit {top:.6} outside pinned band [{TOP_SCORE_LO}, {TOP_SCORE_HI}]"
  );
  // The decision is unambiguous: the speech anchor beats the runner-up.
  assert!(
    ranked[0].score() > ranked[1].score(),
    "speech anchor did not win outright"
  );
}
