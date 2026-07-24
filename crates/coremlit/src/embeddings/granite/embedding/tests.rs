use super::*;

/// A unit vector: `e_0` (all mass on component 0).
fn e0() -> [f32; EMBEDDING_DIM] {
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[0] = 1.0;
  s
}

/// f64-accumulated norm² of a stored embedding. The renormalization tests
/// measure norm in f64: the f32 accumulation error over 384 terms (up to
/// ~2e-5) would drown the ~1e-7 deviation the post-renormalization invariant
/// asserts, so an f32 measure could not tell a renormalized vector from a
/// raw-copied budget-edge one.
fn norm_sq_f64(e: &Embedding) -> f64 {
  e.as_slice()
    .iter()
    .map(|&x| f64::from(x) * f64::from(x))
    .sum()
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
fn check_finite_output_accepts_finite() {
  // A finite model-output row (need not be unit-norm — this gate runs BEFORE
  // normalization) passes.
  let s: Vec<f32> = (0..EMBEDDING_DIM).map(|i| (i as f32) - 100.0).collect();
  assert!(check_finite_output(&s).is_ok());
}

#[test]
fn check_finite_output_rejects_model_nan_as_output_not_embedding() {
  // A NaN the model produced is MODEL corruption (`NonFiniteOutput`), NOT
  // caller-supplied embedding data (`NonFiniteEmbedding`). This is the seam the
  // embedder calls before `from_slice_normalizing`, so the CoreML corruption
  // mode is classified correctly. Removing that call site (or this gate) makes
  // `NonFiniteOutput` unreachable again.
  let mut s = e0();
  s[5] = f32::NAN;
  let err = check_finite_output(&s).unwrap_err();
  assert!(
    matches!(err, Error::NonFiniteOutput { index: 5 }),
    "got {err:?}"
  );
}

#[test]
fn check_finite_output_rejects_inf() {
  let mut s = e0();
  s[9] = f32::NEG_INFINITY;
  let err = check_finite_output(&s).unwrap_err();
  assert!(
    matches!(err, Error::NonFiniteOutput { index: 9 }),
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
  // norm² = 1 + 0.5·NORM_BUDGET must pass (≤ inclusive). Acceptance is the
  // pinned behavior; the accepted vector is additionally stored renormalized.
  let target_sq = 1.0 + 0.5 * NORM_BUDGET;
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[0] = target_sq.sqrt();
  let e = Embedding::try_from_unit_slice(&s).expect("within budget");
  assert!(
    (norm_sq_f64(&e) - 1.0).abs() <= 1e-6,
    "accepted vector must be stored unit-norm (f64 norm² = {})",
    norm_sq_f64(&e)
  );
}

#[test]
fn restored_budget_edge_vector_is_renormalized() {
  // The audit replay: norm² ≈ 1.0000522 (dev 5.22e-5, inside NORM_BUDGET).
  // Copied raw, cos(x, x) = 1.0000522 > 1 and cos(x, −x) < −1; renormalized,
  // the stored vector is unit-norm and cosine stays in [−1, 1].
  let mut s = [0.0f32; EMBEDDING_DIM];
  s[0] = (1.0f32 + 0.522e-4).sqrt();
  let e = Embedding::try_from_unit_slice(&s).expect("within budget");

  assert!(
    (norm_sq_f64(&e) - 1.0).abs() <= 1e-6,
    "stored vector must be renormalized to unit norm (f64 norm² = {})",
    norm_sq_f64(&e)
  );
  let self_cos = e.cosine(&e);
  assert!(
    (1.0 - 1e-5..=1.0 + 1e-5).contains(&self_cos),
    "cos(x, x) must stay in [−1, 1] (got {self_cos})"
  );
  assert!(
    1.0 - self_cos >= -1e-5,
    "cosine distance must be non-negative (1 − cos = {})",
    1.0 - self_cos
  );

  let mut neg_s = [0.0f32; EMBEDDING_DIM];
  neg_s[0] = -s[0];
  let neg = Embedding::try_from_unit_slice(&neg_s).expect("−s is also within budget");
  let opp = e.cosine(&neg);
  assert!(
    (-1.0 - 1e-5..=-1.0 + 1e-5).contains(&opp),
    "cos(x, −x) must stay in [−1, 1] (got {opp})"
  );
}

#[test]
fn near_budget_vectors_restore_unit_norm_property() {
  // A deterministic grid straddling the budget edge, over three vector shapes.
  // The ACTUAL f32-accumulated deviation (the production gate's own metric)
  // decides expected accept/reject in-test, so f32 rounding at the boundary
  // cannot make the test brittle.
  let devs = [-1.0f32, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.95, 1.0];
  let uniform = (1.0f32 / EMBEDDING_DIM as f32).sqrt();
  for &d in &devs {
    let scale = (1.0f32 + d * NORM_BUDGET).sqrt();
    let shapes: [Vec<f32>; 3] = [
      {
        let mut v = vec![0.0f32; EMBEDDING_DIM];
        v[0] = scale; // single axis: e0 · scale
        v
      },
      (0..EMBEDDING_DIM).map(|_| uniform * scale).collect(), // uniform
      (0..EMBEDDING_DIM) // alternating-sign uniform
        .map(|i| {
          if i % 2 == 0 {
            uniform * scale
          } else {
            -uniform * scale
          }
        })
        .collect(),
    ];
    for s in &shapes {
      let norm_sq: f32 = s.iter().map(|x| x * x).sum();
      let expect_accept = (norm_sq - 1.0).abs() <= NORM_BUDGET;
      match Embedding::try_from_unit_slice(s) {
        Ok(e) => {
          assert!(
            expect_accept,
            "accepted a vector the gate should reject (norm² = {norm_sq})"
          );
          assert!(
            (norm_sq_f64(&e) - 1.0).abs() <= 1e-6,
            "accepted vector not stored unit-norm (f64 norm² = {})",
            norm_sq_f64(&e)
          );
          let c = e.cosine(&e);
          assert!(
            (1.0 - 1e-5..=1.0 + 1e-5).contains(&c),
            "cos(x, x) escaped [−1, 1]: {c}"
          );
          assert!(1.0 - c >= -1e-5, "negative cosine distance: {}", 1.0 - c);
          let neg: Vec<f32> = e.to_vec().iter().map(|v| -v).collect();
          let neg_e =
            Embedding::try_from_unit_slice(&neg).expect("a negated unit vector is unit-norm");
          let o = e.cosine(&neg_e);
          assert!(
            (-1.0 - 1e-5..=-1.0 + 1e-5).contains(&o),
            "cos(x, −x) escaped [−1, 1]: {o}"
          );
        }
        Err(Error::EmbeddingNotUnitNorm { .. }) => {
          assert!(
            !expect_accept,
            "rejected a vector the gate should accept (norm² = {norm_sq})"
          );
        }
        Err(other) => panic!("unexpected error: {other:?}"),
      }
    }
  }
}

#[test]
fn try_from_unit_slice_is_idempotent_on_its_output() {
  // Renormalizing an already-renormalized vector is a no-op to ULP scale.
  let uniform = (1.0f32 / EMBEDDING_DIM as f32).sqrt();
  let scale = (1.0f32 + 0.5 * NORM_BUDGET).sqrt();
  let s: Vec<f32> = (0..EMBEDDING_DIM).map(|_| uniform * scale).collect();
  let first = Embedding::try_from_unit_slice(&s).expect("within budget");
  let second =
    Embedding::try_from_unit_slice(&first.to_vec()).expect("renormalized output re-accepts");
  assert!(
    second.is_close(&first, 1e-6),
    "renormalizing an already-renormalized vector must be a no-op to ULP scale"
  );
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
fn try_from_unit_slice_wrong_len() {
  let s = [0.0f32; EMBEDDING_DIM + 1];
  let err = Embedding::try_from_unit_slice(&s).unwrap_err();
  assert!(
    matches!(
      err,
      Error::EmbeddingDimMismatch {
        expected: EMBEDDING_DIM,
        got
      } if got == EMBEDDING_DIM + 1
    ),
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

#[test]
fn to_vec_roundtrips_via_try_from_unit_slice() {
  let e = Embedding::from_slice_normalizing(&e0()).unwrap();
  let v = e.to_vec();
  assert_eq!(v.len(), EMBEDDING_DIM);
  // The unit vector round-trips back through the trusted-path constructor.
  let back = Embedding::try_from_unit_slice(&v).expect("unit vector round-trips");
  assert!(back.is_close(&e, 0.0));
}
