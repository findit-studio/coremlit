use super::*;
use crate::embeddings::siglip::embedding::EMBEDDING_DIM;

/// A unit embedding with all mass on component `i`.
fn axis(i: usize) -> Embedding {
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[i] = 1.0;
  Embedding::from_slice_normalizing(&s).expect("unit")
}

/// A unit embedding blended between axes 0 and 1 (weight `w` on axis 0).
fn blend(w: f32) -> Embedding {
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[0] = w;
  s[1] = 1.0 - w;
  Embedding::from_slice_normalizing(&s).expect("unit")
}

#[test]
fn rank_orders_by_cosine_descending() {
  let query = axis(0);
  let near = blend(0.9); // close to axis 0
  let mid = blend(0.5);
  let far = axis(1); // orthogonal to the query
  let candidates = [
    Candidate::new("far", &far),
    Candidate::new("near", &near),
    Candidate::new("mid", &mid),
  ];
  let ranked = rank(&query, &candidates);
  assert_eq!(
    ranked.iter().map(Ranked::label).collect::<Vec<_>>(),
    vec!["near", "mid", "far"]
  );
  // Scores are descending and are the query↔candidate cosines.
  assert!(ranked[0].score() >= ranked[1].score());
  assert!(ranked[1].score() >= ranked[2].score());
  assert!((ranked[0].score() - query.cosine(&near)).abs() <= 1e-6);
}

#[test]
fn rank_is_stable_on_ties() {
  let query = axis(0);
  let a = blend(0.7);
  let b = blend(0.7); // identical score to `a`
  let candidates = [Candidate::new("a", &a), Candidate::new("b", &b)];
  let ranked = rank(&query, &candidates);
  // Equal scores keep input order (stable sort).
  assert_eq!(
    ranked.iter().map(Ranked::label).collect::<Vec<_>>(),
    vec!["a", "b"]
  );
}

#[test]
fn rank_empty_candidates_is_empty() {
  let query = axis(0);
  assert!(rank(&query, &[]).is_empty());
}

#[test]
fn rank_query_matches_its_identical_candidate_top1() {
  // Cross-modal shape: a "text" query ranks its matching "image" first.
  let image_a = blend(0.8);
  let image_b = blend(0.2);
  let query = blend(0.8); // identical to image_a
  let candidates = [
    Candidate::new("image_b", &image_b),
    Candidate::new("image_a", &image_a),
  ];
  let ranked = rank(&query, &candidates);
  assert_eq!(
    ranked[0].label(),
    "image_a",
    "identical embedding ranks top-1"
  );
  assert!((ranked[0].score() - 1.0).abs() <= 1e-5, "self-cosine ≈ 1");
}
