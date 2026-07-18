use super::*;

/// A unit vector: `e_0` (all mass on component 0).
fn e0() -> [f32; EMBEDDING_DIM] {
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[0] = 1.0;
  s
}

#[test]
fn from_slice_normalizing_produces_unit_norm() {
  let s: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (i as f32) + 1.0).collect();
  let e = Embedding::from_slice_normalizing(&s).unwrap();
  let norm_sq: f32 = e.as_slice().iter().map(|x| x * x).sum();
  assert!((norm_sq - 1.0).abs() <= NORM_BUDGET, "norm² = {norm_sq}");
}

#[test]
fn from_slice_normalizing_rejects_zero() {
  let s = [0.0f32; EMBEDDING_DIM];
  let err = Embedding::from_slice_normalizing(&s).unwrap_err();
  assert!(matches!(err, Error::EmbeddingZero), "got {err:?}");
}

#[test]
fn from_slice_normalizing_rejects_nan() {
  let mut s = e0();
  s[7] = f32::NAN;
  let err = Embedding::from_slice_normalizing(&s).unwrap_err();
  assert!(
    matches!(err, Error::NonFiniteEmbedding { component_index: 7 }),
    "got {err:?}"
  );
}

#[test]
fn from_slice_normalizing_rejects_inf() {
  let mut s = e0();
  s[3] = f32::INFINITY;
  let err = Embedding::from_slice_normalizing(&s).unwrap_err();
  assert!(
    matches!(err, Error::NonFiniteEmbedding { component_index: 3 }),
    "got {err:?}"
  );
}

#[test]
fn from_slice_normalizing_handles_overflow_magnitude() {
  // f32::MAX components would overflow an f32 norm accumulator; the f64 path
  // must still produce a finite unit vector.
  let s = [f32::MAX; EMBEDDING_DIM];
  let e = Embedding::from_slice_normalizing(&s).expect("f32::MAX normalizes via f64");
  let norm_sq: f32 = e.as_slice().iter().map(|x| x * x).sum();
  assert!((norm_sq - 1.0).abs() <= NORM_BUDGET, "norm² = {norm_sq}");
}

#[test]
fn from_slice_normalizing_handles_smallest_subnormal() {
  // Casting inv_norm to f32 before the per-component multiply would overflow to
  // +Inf here; the f64 multiply must keep it finite and unit-norm.
  let s = [f32::from_bits(1); EMBEDDING_DIM]; // ~1.4e-45
  let e = Embedding::from_slice_normalizing(&s).expect("subnormal magnitude normalizes");
  let norm_sq: f32 = e.as_slice().iter().map(|x| x * x).sum();
  assert!((norm_sq - 1.0).abs() <= NORM_BUDGET, "norm² = {norm_sq}");
}

#[test]
fn from_slice_normalizing_wrong_len() {
  let s = [0.0f32; EMBEDDING_DIM - 1];
  let err = Embedding::from_slice_normalizing(&s).unwrap_err();
  assert!(
    matches!(
      err,
      Error::EmbeddingDimMismatch {
        expected: EMBEDDING_DIM,
        got
      } if got == EMBEDDING_DIM - 1
    ),
    "got {err:?}"
  );
}

#[test]
fn try_from_unit_slice_accepts_at_budget_edge() {
  // norm² = 1 + 0.5·NORM_BUDGET must pass (≤ inclusive).
  let target_sq = 1.0 + 0.5 * NORM_BUDGET;
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[0] = target_sq.sqrt();
  Embedding::try_from_unit_slice(&s).expect("within budget");
}

#[test]
fn try_from_unit_slice_rejects_beyond_budget() {
  // norm² = 1 + 2·NORM_BUDGET must fail.
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[0] = (1.0 + 2.0 * NORM_BUDGET).sqrt();
  let err = Embedding::try_from_unit_slice(&s).unwrap_err();
  assert!(
    matches!(err, Error::EmbeddingNotUnitNorm { .. }),
    "got {err:?}"
  );
}

#[test]
fn dot_and_cosine_agree_for_unit_vectors() {
  let e = Embedding::from_slice_normalizing(&e0()).unwrap();
  assert_eq!(e.dot(&e), e.cosine(&e));
  assert!((e.cosine(&e) - 1.0).abs() <= 1e-6);
}

#[test]
fn orthogonal_unit_vectors_have_zero_cosine() {
  let a = Embedding::from_slice_normalizing(&e0()).unwrap();
  let mut y = [0.0f32; EMBEDDING_DIM];
  y[1] = 1.0;
  let b = Embedding::from_slice_normalizing(&y).unwrap();
  assert!(a.cosine(&b).abs() <= 1e-6);
}

#[test]
fn is_close_self_at_zero_tolerance() {
  let a = Embedding::from_slice_normalizing(&e0()).unwrap();
  assert!(a.is_close(&a, 0.0));
  assert!(a.is_close_cosine(&a, 0.0));
}

#[test]
fn is_close_cosine_separates_orthogonal() {
  let a = Embedding::from_slice_normalizing(&e0()).unwrap();
  let mut y = [0.0f32; EMBEDDING_DIM];
  y[1] = 1.0;
  let b = Embedding::from_slice_normalizing(&y).unwrap();
  // 1 − cos = 1 for orthogonal; must exceed a tiny tolerance.
  assert!(!a.is_close_cosine(&b, 1.0e-6));
  assert!(!a.is_close(&b, 1.0e-6));
}

#[test]
fn deref_and_as_ref_expose_the_slice() {
  let e = Embedding::from_slice_normalizing(&e0()).unwrap();
  assert_eq!(e.len(), EMBEDDING_DIM); // via Deref<[f32]>
  let r: &[f32] = e.as_ref();
  assert_eq!(r.len(), EMBEDDING_DIM);
  assert_eq!(e.dim(), EMBEDDING_DIM);
}
