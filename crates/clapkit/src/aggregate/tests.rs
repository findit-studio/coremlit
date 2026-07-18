use super::*;
use crate::window::WindowSpan;

const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.70710677

/// A unit-norm window embedding pointing along axis `i`, with a span covering
/// `real_len` real samples (so `coverage == real_len / 480_000`).
fn axis(i: usize, real_len: usize) -> WindowEmbedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = 1.0;
  let e = Embedding::from_slice_normalizing(&v).unwrap();
  WindowEmbedding::new(e, WindowSpan::new(0, real_len))
}

fn assert_close(got: &Embedding, expected: &[(usize, f32)]) {
  let s = got.as_slice();
  // Unit-norm invariant holds for every aggregation result.
  let norm_sq: f32 = s.iter().map(|x| x * x).sum();
  assert!((norm_sq - 1.0).abs() < 1e-5, "not unit-norm: {norm_sq}");
  for &(i, want) in expected {
    assert!(
      (s[i] - want).abs() < 1e-5,
      "component {i}: got {}, want {want}",
      s[i]
    );
  }
}

#[test]
fn mean_of_two_orthogonal_windows_is_the_bisector() {
  let out = MeanRenormalized
    .aggregate(&[axis(0, 480_000), axis(1, 480_000)])
    .unwrap();
  assert_close(&out, &[(0, FRAC_1_SQRT_2), (1, FRAC_1_SQRT_2)]);
}

#[test]
fn mean_of_one_window_is_that_window() {
  let out = MeanRenormalized.aggregate(&[axis(3, 240_000)]).unwrap();
  assert_close(&out, &[(3, 1.0)]);
}

#[test]
fn ema_alpha_edges_pick_first_and_last() {
  let windows = [axis(0, 480_000), axis(1, 480_000)];
  // alpha = 0 keeps the first window; alpha = 1 keeps the last.
  let first = EmaRenormalized::new(0.0).aggregate(&windows).unwrap();
  assert_close(&first, &[(0, 1.0), (1, 0.0)]);
  let last = EmaRenormalized::new(1.0).aggregate(&windows).unwrap();
  assert_close(&last, &[(0, 0.0), (1, 1.0)]);
}

#[test]
fn ema_half_over_two_windows_is_the_bisector() {
  let out = EmaRenormalized::new(0.5)
    .aggregate(&[axis(0, 480_000), axis(1, 480_000)])
    .unwrap();
  assert_close(&out, &[(0, FRAC_1_SQRT_2), (1, FRAC_1_SQRT_2)]);
}

#[test]
fn ema_half_over_three_windows() {
  // ema = (0.25, 0.25, 0.5) before renormalization (‖·‖ = √0.375).
  let out = EmaRenormalized::new(0.5)
    .aggregate(&[axis(0, 480_000), axis(1, 480_000), axis(2, 480_000)])
    .unwrap();
  assert_close(
    &out,
    &[(0, 0.408_248_3), (1, 0.408_248_3), (2, 0.816_496_6)],
  );
}

#[test]
fn coverage_weighting_down_weights_a_padded_tail() {
  // Full window on axis 0 (coverage 1.0) + quarter-coverage tail on axis 1
  // (coverage 0.25): weighted mean = (0.8, 0.2), renormalized.
  let out = CoverageWeightedMean
    .aggregate(&[axis(0, 480_000), axis(1, 120_000)])
    .unwrap();
  assert_close(&out, &[(0, 0.970_142_5), (1, 0.242_535_63)]);
  // Contrast: an equal-weight mean would put both at 1/√2 ≈ 0.707 — the tail is
  // demonstrably down-weighted (0.24 < 0.71).
  assert!(out.as_slice()[1] < FRAC_1_SQRT_2);
}

#[test]
fn coverage_weighting_equals_mean_at_full_coverage() {
  let windows = [axis(0, 480_000), axis(1, 480_000)];
  let cov = CoverageWeightedMean.aggregate(&windows).unwrap();
  let mean = MeanRenormalized.aggregate(&windows).unwrap();
  assert!(cov.is_close(&mean, 1e-6));
}

#[test]
fn every_policy_rejects_empty_windows() {
  let policies: [&dyn AggregatePolicy; 3] = [
    &MeanRenormalized,
    &EmaRenormalized::new(0.5),
    &CoverageWeightedMean,
  ];
  for p in policies {
    assert!(matches!(p.aggregate(&[]), Err(Error::EmptyWindows)));
  }
}

#[test]
fn ema_rejects_out_of_range_alpha_at_aggregation() {
  let windows = [axis(0, 480_000)];
  for bad in [1.5f32, -0.1, f32::NAN, f32::INFINITY] {
    let err = EmaRenormalized::new(bad).aggregate(&windows).unwrap_err();
    assert!(
      matches!(
        err,
        Error::InvalidPolicyParameter {
          policy: "EmaRenormalized",
          param: "alpha",
          ..
        }
      ),
      "alpha {bad} should be rejected, got {err:?}"
    );
  }
}

#[test]
fn into_policy_dispatches_to_the_matching_built_in() {
  let windows = [axis(0, 480_000), axis(1, 120_000)];
  let cases = [
    (AggregatePolicyKind::MeanRenormalized, {
      MeanRenormalized.aggregate(&windows).unwrap()
    }),
    (AggregatePolicyKind::EmaRenormalized { alpha: 0.5 }, {
      EmaRenormalized::new(0.5).aggregate(&windows).unwrap()
    }),
    (AggregatePolicyKind::CoverageWeightedMean, {
      CoverageWeightedMean.aggregate(&windows).unwrap()
    }),
  ];
  for (kind, expected) in cases {
    let via_box = kind.into_policy().aggregate(&windows).unwrap();
    assert!(
      via_box.is_close(&expected, 1e-6),
      "{kind:?} box disagreed with the concrete policy"
    );
  }
}

#[cfg(feature = "serde")]
mod serde_tests {
  use super::*;

  #[test]
  fn kind_wire_spellings_are_pinned_and_round_trip() {
    // Wildcard-free match: a new variant must pin its JSON here (F3 tripwire).
    for &kind in AggregatePolicyKind::REPRESENTATIVES {
      let expected = match kind {
        AggregatePolicyKind::MeanRenormalized => r#""mean_renormalized""#.to_string(),
        AggregatePolicyKind::EmaRenormalized { alpha } => {
          format!(r#"{{"ema_renormalized":{{"alpha":{alpha}}}}}"#)
        }
        AggregatePolicyKind::CoverageWeightedMean => r#""coverage_weighted_mean""#.to_string(),
      };
      let json = serde_json::to_string(&kind).unwrap();
      assert_eq!(json, expected, "serde spelling for {kind:?} drifted");
      let back: AggregatePolicyKind = serde_json::from_str(&json).unwrap();
      assert_eq!(back, kind, "{kind:?} must round-trip from its own JSON");
    }
  }

  #[test]
  fn non_snake_case_spelling_is_rejected() {
    assert!(serde_json::from_str::<AggregatePolicyKind>(r#""MeanRenormalized""#).is_err());
    assert!(
      serde_json::from_str::<AggregatePolicyKind>(r#"{"EmaRenormalized":{"alpha":0.5}}"#).is_err()
    );
  }
}
