use super::*;
use crate::embeddings::clap::{
  embedding::EMBEDDING_DIM,
  error::WinditError,
  window::{Span, WINDOW_SAMPLES},
};

const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1/√2 ≈ 0.70710677

/// A unit-norm window embedding pointing along axis `i`, with a span covering
/// `real_len` real samples (so `coverage == real_len / 480_000`).
fn axis(i: usize, real_len: usize) -> WindowEmbedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = 1.0;
  let e = Embedding::from_slice_normalizing(&v).unwrap();
  WindowEmbedding::new(e, Span::new(0, real_len, WINDOW_SAMPLES))
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
  let out = aggregate(&MeanRenormalized, &[axis(0, 480_000), axis(1, 480_000)]).unwrap();
  assert_close(&out, &[(0, FRAC_1_SQRT_2), (1, FRAC_1_SQRT_2)]);
}

#[test]
fn mean_of_one_window_is_that_window() {
  let out = aggregate(&MeanRenormalized, &[axis(3, 240_000)]).unwrap();
  assert_close(&out, &[(3, 1.0)]);
}

#[test]
fn ema_alpha_edges_pick_first_and_last() {
  let windows = [axis(0, 480_000), axis(1, 480_000)];
  // alpha = 0 keeps the first window; alpha = 1 keeps the last.
  let first = aggregate(&EmaRenormalized::new(0.0), &windows).unwrap();
  assert_close(&first, &[(0, 1.0), (1, 0.0)]);
  let last = aggregate(&EmaRenormalized::new(1.0), &windows).unwrap();
  assert_close(&last, &[(0, 0.0), (1, 1.0)]);
}

#[test]
fn ema_half_over_two_windows_is_the_bisector() {
  let out = aggregate(
    &EmaRenormalized::new(0.5),
    &[axis(0, 480_000), axis(1, 480_000)],
  )
  .unwrap();
  assert_close(&out, &[(0, FRAC_1_SQRT_2), (1, FRAC_1_SQRT_2)]);
}

#[test]
fn ema_half_over_three_windows() {
  // ema = (0.25, 0.25, 0.5) before renormalization (‖·‖ = √0.375).
  let out = aggregate(
    &EmaRenormalized::new(0.5),
    &[axis(0, 480_000), axis(1, 480_000), axis(2, 480_000)],
  )
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
  let out = aggregate(&CoverageWeightedMean, &[axis(0, 480_000), axis(1, 120_000)]).unwrap();
  assert_close(&out, &[(0, 0.970_142_5), (1, 0.242_535_63)]);
  // Contrast: an equal-weight mean would put both at 1/√2 ≈ 0.707 — the tail is
  // demonstrably down-weighted (0.24 < 0.71).
  assert!(out.as_slice()[1] < FRAC_1_SQRT_2);
}

#[test]
fn coverage_weighting_equals_mean_at_full_coverage() {
  let windows = [axis(0, 480_000), axis(1, 480_000)];
  let cov = aggregate(&CoverageWeightedMean, &windows).unwrap();
  let mean = aggregate(&MeanRenormalized, &windows).unwrap();
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
    assert!(matches!(aggregate(p, &[]), Err(Error::EmptyWindows)));
  }
}

/// A pathological custom [`AggregatePolicy`]: always reports
/// `WinditError::Empty`, including for a NONEMPTY `embeddings` slice. No real
/// policy should do this; it exists only to drive the test below, which pins
/// that the wrapper's error mapping does not assume otherwise.
#[derive(Debug, Clone, Copy)]
struct ClaimsEmptyRegardless;

impl AggregatePolicy for ClaimsEmptyRegardless {
  fn aggregate_values(
    &self,
    _embeddings: &[&[f64]],
    _coverages: &[f64],
    _dim: usize,
  ) -> core::result::Result<Vec<f64>, WinditError> {
    Err(WinditError::Empty)
  }
}

#[test]
fn a_custom_policys_empty_claim_on_nonempty_input_reaches_windowing_not_emptywindows() {
  // The error variant alone cannot distinguish "the engine saw no windows" from
  // "the policy refused": this fixture is two windows, not zero, so a policy
  // reporting `Empty` here is reporting an aggregation failure, not "there were
  // no windows". A caller matching `EmptyWindows` to mean the latter must not be
  // misled by this call.
  let windows = [axis(0, 480_000), axis(1, 480_000)];
  assert!(
    !windows.is_empty(),
    "the fixture must be nonempty for this to prove anything"
  );

  let err = aggregate(&ClaimsEmptyRegardless, &windows).unwrap_err();

  assert!(
    matches!(err, Error::Windowing(WinditError::Empty)),
    "expected Windowing(Empty), got {err:?}"
  );
  assert!(
    !matches!(err, Error::EmptyWindows),
    "a nonempty input's aggregation failure must never surface as EmptyWindows \
     (that variant means zero windows were supplied, which is false here); got \
     {err:?}"
  );
}

#[test]
fn ema_rejects_out_of_range_alpha_at_aggregation() {
  let windows = [axis(0, 480_000)];
  for bad in [1.5f64, -0.1, f64::NAN, f64::INFINITY] {
    let err = aggregate(&EmaRenormalized::new(bad), &windows).unwrap_err();
    assert!(
      matches!(err, Error::Windowing(WinditError::AlphaOutOfRange)),
      "alpha {bad} should be rejected as Windowing(AlphaOutOfRange), got {err:?}"
    );
  }
}

#[test]
fn into_policy_dispatches_to_the_matching_built_in() {
  let windows = [axis(0, 480_000), axis(1, 120_000)];
  let cases = [
    (AggregatePolicyKind::MeanRenormalized, {
      aggregate(&MeanRenormalized, &windows).unwrap()
    }),
    (
      AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.5)),
      { aggregate(&EmaRenormalized::new(0.5), &windows).unwrap() },
    ),
    (AggregatePolicyKind::CoverageWeightedMean, {
      aggregate(&CoverageWeightedMean, &windows).unwrap()
    }),
  ];
  for (kind, expected) in cases {
    let via_box = aggregate(kind.into_policy().as_ref(), &windows).unwrap();
    assert!(
      via_box.is_close(&expected, 1e-6),
      "{kind:?} box disagreed with the concrete policy"
    );
  }
}

#[test]
fn a_configured_alpha_reaches_the_fold_at_f64_precision() {
  // The wire field is `f64`, so a configured `0.3` is the `f64` nearest 3/10 and
  // not the `f32` one widened into the fold. The two differ in the eighth
  // significant digit, which is under the 1e-6 `is_close` the dispatch test
  // above uses — so the decision is pinned here instead, bit-exactly, at the
  // `f64` compute domain the policy actually folds in.
  //
  // Reverting `EmaRenormalizedOptions::alpha` to `f32` fails this test by
  // failing to COMPILE: `into_policy` hands the accessor's value to
  // `EmaRenormalized::new`, which takes `C: Real`, and `Real` is sealed to
  // `f64`.
  let boxed = AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(0.3)).into_policy();
  let (a, b) = ([1.0f64, 0.0], [0.0f64, 1.0]);
  let embeddings: [&[f64]; 2] = [&a, &b];
  let coverages = [1.0f64, 0.25];

  let configured = boxed.aggregate_values(&embeddings, &coverages, 2).unwrap();
  let exact = EmaRenormalized::new(0.3f64)
    .aggregate_values(&embeddings, &coverages, 2)
    .unwrap();
  let widened_from_f32 = EmaRenormalized::new(f64::from(0.3f32))
    .aggregate_values(&embeddings, &coverages, 2)
    .unwrap();

  assert_eq!(
    configured, exact,
    "a configured alpha must reach the fold unrounded"
  );
  assert_ne!(
    configured, widened_from_f32,
    "an f32 wire field would have folded these weights instead — if this ever \
     passes, the eighth-digit move this release took on has been reverted"
  );
}

#[test]
fn alpha_that_rounds_into_range_as_f32_is_rejected_at_f64() {
  // The behavioural fact `EmaRenormalizedOptions::alpha`'s rustdoc now
  // documents: `1.00000001` parses as `f32` to exactly `1.0` (in range) but
  // stays `1.00000001` as `f64` (out of range). A legacy JSON config with this
  // literal used to aggregate and no longer does — pin both sides so the doc
  // and the behaviour cannot quietly drift apart again.
  let widened_from_legacy_f32 = f64::from(1.00000001f32);
  assert_eq!(
    widened_from_legacy_f32, 1.0,
    "premise: f32 must round this literal to exactly 1.0"
  );

  let windows = [axis(0, 480_000), axis(1, 480_000)];

  let via_f32 =
    AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(widened_from_legacy_f32))
      .into_policy();
  assert!(
    aggregate(via_f32.as_ref(), &windows).is_ok(),
    "an alpha an f32 field would have rounded to 1.0 must still aggregate"
  );

  let via_f64 =
    AggregatePolicyKind::EmaRenormalized(EmaRenormalizedOptions::new(1.00000001)).into_policy();
  let err = aggregate(via_f64.as_ref(), &windows).unwrap_err();
  assert!(
    matches!(err, Error::Windowing(WinditError::AlphaOutOfRange)),
    "the same literal at f64 precision must be rejected as AlphaOutOfRange, got {err:?}"
  );
}

#[cfg(feature = "serde")]
mod serde_tests {
  use super::*;

  #[test]
  fn kind_wire_spellings_are_pinned_and_round_trip() {
    // Wildcard-free `match`: a new variant fails to compile until its expected
    // JSON is written here. Executing that variant's round-trip additionally
    // needs a `REPRESENTATIVES` entry — the hand-kept roster this loop iterates.
    for &kind in AggregatePolicyKind::REPRESENTATIVES {
      let expected = match kind {
        AggregatePolicyKind::MeanRenormalized => r#""mean_renormalized""#.to_string(),
        AggregatePolicyKind::EmaRenormalized(ema) => {
          let alpha = ema.alpha();
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
