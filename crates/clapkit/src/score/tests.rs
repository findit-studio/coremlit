use super::*;
use crate::{embedding::EMBEDDING_DIM, window::WindowSpan};

/// A unit-norm embedding from the given `(index, weight)` components.
fn emb(components: &[(usize, f32)]) -> Embedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  for &(i, w) in components {
    v[i] = w;
  }
  Embedding::from_slice_normalizing(&v).unwrap()
}

#[test]
fn logit_scales_are_pinned_from_the_learned_checkpoint() {
  // These f64 widenings are the LEARNED checkpoint parameters
  // `model.logit_scale_{a,t}.exp()` (via conversion/scripts/inspect_struct.py),
  // NOT the config. The audio scale trained away from init to 18.661177. The text
  // f64 literal is 14.285714149475098 — the learned value — and is distinct from
  // the config's `logit_scale_init_value` exp (14.285714285714285); the two only
  // coincide once rounded to the shipped f32 (14.285714). Pinning "from config"
  // would therefore select the wrong f64 source (and the wrong audio temperature).
  assert_eq!(LOGIT_SCALE_AUDIO as f64, 18.661176681518555);
  assert_eq!(LOGIT_SCALE_TEXT as f64, 14.285714149475098);
}

#[test]
fn ranks_labels_by_cosine_descending() {
  let audio = emb(&[(0, 2.0), (1, 1.0)]); // closest to axis 0, then axis 1
  let dog = emb(&[(0, 1.0)]);
  let cat = emb(&[(1, 1.0)]);
  let bird = emb(&[(2, 1.0)]);
  let anchors = [
    TextAnchor::new("cat", &cat),
    TextAnchor::new("dog", &dog),
    TextAnchor::new("bird", &bird),
  ];

  let ranked = score(&audio, &anchors, ScoreMode::Cosine);
  let labels: Vec<&str> = ranked.iter().map(|r| r.label()).collect();
  assert_eq!(labels, ["dog", "cat", "bird"]);
  // dog cosine = 2/√5 ≈ 0.8944272 (two-sided).
  assert!((ranked[0].score() - 0.894_427_2).abs() < 1e-5);
  assert!((ranked[1].score() - 0.447_213_6).abs() < 1e-5);
  assert!(ranked[2].score().abs() < 1e-6); // orthogonal
}

#[test]
fn logit_scaled_preserves_ranking_and_scales_score() {
  let audio = emb(&[(0, 2.0), (1, 1.0)]);
  let dog = emb(&[(0, 1.0)]);
  let cat = emb(&[(1, 1.0)]);
  let anchors = [TextAnchor::new("dog", &dog), TextAnchor::new("cat", &cat)];

  let cosine = score(&audio, &anchors, ScoreMode::Cosine);
  let logit = score(&audio, &anchors, ScoreMode::LogitScaled);
  // Same ranking (monotonic transform).
  assert_eq!(
    cosine.iter().map(|r| r.label()).collect::<Vec<_>>(),
    logit.iter().map(|r| r.label()).collect::<Vec<_>>()
  );
  // Score scaled by exactly LOGIT_SCALE_AUDIO.
  for (c, l) in cosine.iter().zip(logit.iter()) {
    assert!((l.score() - c.score() * LOGIT_SCALE_AUDIO).abs() < 1e-4);
  }
}

#[test]
fn ties_keep_input_order() {
  let audio = emb(&[(0, 1.0)]);
  let same = emb(&[(0, 1.0)]);
  let anchors = [
    TextAnchor::new("first", &same),
    TextAnchor::new("second", &same),
  ];
  let ranked = score(&audio, &anchors, ScoreMode::Cosine);
  // Identical scores ⇒ stable order preserves the caller's sequence.
  assert_eq!(ranked[0].label(), "first");
  assert_eq!(ranked[1].label(), "second");
}

#[test]
fn empty_anchors_yield_empty() {
  let audio = emb(&[(0, 1.0)]);
  assert!(score(&audio, &[], ScoreMode::Cosine).is_empty());
}

#[test]
fn per_window_scores_are_exposed() {
  let dog = emb(&[(0, 1.0)]);
  let cat = emb(&[(1, 1.0)]);
  let anchors = [TextAnchor::new("dog", &dog), TextAnchor::new("cat", &cat)];
  let windows = [
    WindowEmbedding::new(emb(&[(0, 1.0)]), WindowSpan::new(0, 480_000)),
    WindowEmbedding::new(emb(&[(1, 1.0)]), WindowSpan::new(480_000, 480_000)),
  ];
  let per_window = score_windows(&windows, &anchors, ScoreMode::Cosine);
  assert_eq!(per_window.len(), 2);
  // Window 0 is pure "dog", window 1 is pure "cat".
  assert_eq!(per_window[0][0].label(), "dog");
  assert_eq!(per_window[1][0].label(), "cat");
}

#[test]
fn labeled_score_owned_round_trip() {
  let audio = emb(&[(0, 1.0)]);
  let dog = emb(&[(0, 1.0)]);
  let anchors = [TextAnchor::new("dog", &dog)];
  let ranked = score(&audio, &anchors, ScoreMode::Cosine);
  let owned = ranked[0].to_owned();
  assert_eq!(owned.label(), "dog");
  assert_eq!(owned.score(), ranked[0].score());
  assert_eq!(owned.into_label(), "dog");
}
