use super::*;
use crate::audio::ced::{NUM_CLASSES, WINDOW_SAMPLES, prediction::sigmoid, window::Span};

/// A full-coverage window at slot `i` carrying a uniform confidence vector
/// with `values[0]` and `values[1]` scripted.
fn window(i: usize, v0: f32, v1: f32) -> WindowConfidences {
  let mut values = vec![0.25f32; NUM_CLASSES];
  values[0] = v0;
  values[1] = v1;
  WindowConfidences::new(
    Confidences::new(values),
    Span::new(i * WINDOW_SAMPLES, WINDOW_SAMPLES, WINDOW_SAMPLES),
  )
}

#[test]
fn mean_is_the_equal_weight_arithmetic_mean() {
  let windows = [window(0, 0.2, 0.9), window(1, 0.6, 0.1)];
  let agg = aggregate_windows(ChunkAggregation::Mean, &windows).unwrap();
  assert!((agg.as_slice()[0] - 0.4).abs() < 1e-6);
  assert!((agg.as_slice()[1] - 0.5).abs() < 1e-6);
  assert!((agg.as_slice()[2] - 0.25).abs() < 1e-6);
}

#[test]
fn max_is_the_elementwise_peak() {
  let windows = [window(0, 0.2, 0.9), window(1, 0.6, 0.1)];
  let agg = aggregate_windows(ChunkAggregation::Max, &windows).unwrap();
  assert_eq!(agg.as_slice()[0], 0.6);
  assert_eq!(agg.as_slice()[1], 0.9);
  assert_eq!(agg.as_slice()[2], 0.25);
}

#[test]
fn mean_and_max_discriminate() {
  // The scripted vectors are chosen so Mean ≠ Max on class 0 — a policy mixup
  // cannot pass both pinned tests above.
  let windows = [window(0, 0.2, 0.9), window(1, 0.6, 0.1)];
  let mean = aggregate_windows(ChunkAggregation::Mean, &windows).unwrap();
  let max = aggregate_windows(ChunkAggregation::Max, &windows).unwrap();
  assert_ne!(mean.as_slice()[0], max.as_slice()[0]);
}

#[test]
fn single_window_is_the_bit_exact_identity() {
  // soundevents divides only when count > 1, so one window aggregates to
  // ITSELF — no rounding drift from a multiply-then-divide round trip.
  let w = window(0, 0.123_456_7, 0.9);
  for aggregation in [ChunkAggregation::Mean, ChunkAggregation::Max] {
    let agg = aggregate_windows(aggregation, std::slice::from_ref(&w)).unwrap();
    assert_eq!(agg, *w.value(), "{aggregation:?}");
  }
}

#[test]
fn empty_windows_is_a_typed_error() {
  let err = aggregate_windows(ChunkAggregation::Mean, &[]).unwrap_err();
  assert!(
    matches!(err, crate::audio::ced::Error::EmptyWindows),
    "got {err:?}"
  );
}

#[test]
fn aggregation_runs_in_confidence_space_not_logit_space() {
  // Mutation red (spec §8): sigmoid is nonlinear, so mean-of-confidences and
  // sigmoid-of-mean-logit are DIFFERENT numbers. Two windows from logits 0
  // and 2: confidence-space mean = (σ(0) + σ(2)) / 2 ≈ 0.690399; logit-space
  // would give σ(1) ≈ 0.731059. The pinned expectation is the confidence-space
  // value AND its distance from the logit-space value, so an implementation
  // that aggregates logits cannot pass.
  let mut l0 = vec![0.0f32; NUM_CLASSES];
  let mut l1 = vec![0.0f32; NUM_CLASSES];
  l0[0] = 0.0;
  l1[0] = 2.0;
  let windows = [
    WindowConfidences::new(
      Confidences::from_logits(&l0),
      Span::new(0, WINDOW_SAMPLES, WINDOW_SAMPLES),
    ),
    WindowConfidences::new(
      Confidences::from_logits(&l1),
      Span::new(WINDOW_SAMPLES, WINDOW_SAMPLES, WINDOW_SAMPLES),
    ),
  ];
  let agg = aggregate_windows(ChunkAggregation::Mean, &windows).unwrap();
  let confidence_space = (sigmoid(0.0) + sigmoid(2.0)) / 2.0;
  let logit_space = sigmoid(1.0);
  assert!((agg.as_slice()[0] - confidence_space).abs() < 1e-6);
  assert!(
    (agg.as_slice()[0] - logit_space).abs() > 0.02,
    "confidence-space and logit-space aggregation must be distinguishable"
  );
}

#[test]
fn accumulator_finish_without_pushes_is_empty_windows() {
  // The streaming fold's empty case matches `aggregate_windows(&[])`: a typed
  // EmptyWindows, never a panic on the empty `values`.
  for aggregation in [ChunkAggregation::Mean, ChunkAggregation::Max] {
    let err = Accumulator::new(aggregation).finish().unwrap_err();
    assert!(
      matches!(err, crate::audio::ced::Error::EmptyWindows),
      "{aggregation:?}: got {err:?}"
    );
  }
}

#[test]
fn accumulator_matches_aggregate_windows_bit_exactly() {
  // The streaming fold (what `classify_long` drives via push/finish) must equal
  // the batch `aggregate_windows` BIT for bit — the accumulator refactor is
  // pure single-sourcing, not a numeric change. Multi-window so Mean's divide
  // and Max's peak both engage.
  let windows = [
    window(0, 0.2, 0.9),
    window(1, 0.6, 0.1),
    window(2, 0.123_456_7, 0.000_001),
  ];
  for aggregation in [ChunkAggregation::Mean, ChunkAggregation::Max] {
    let batch = aggregate_windows(aggregation, &windows).unwrap();
    let mut acc = Accumulator::new(aggregation);
    for w in &windows {
      acc.push(w.value());
    }
    let streamed = acc.finish().unwrap();
    assert_eq!(streamed.as_slice(), batch.as_slice(), "{aggregation:?}");
  }
}

#[cfg(feature = "serde")]
mod serde_tests {
  use super::*;

  #[test]
  fn wire_spellings_are_pinned() {
    // Wildcard-free: a new variant fails to compile until its spelling is
    // pinned here (the clap AggregatePolicyKind golden pattern).
    for kind in [ChunkAggregation::Mean, ChunkAggregation::Max] {
      let expected = match kind {
        ChunkAggregation::Mean => "\"mean\"",
        ChunkAggregation::Max => "\"max\"",
      };
      assert_eq!(serde_json::to_string(&kind).unwrap(), expected);
      let back: ChunkAggregation = serde_json::from_str(expected).unwrap();
      assert_eq!(back, kind);
    }
  }

  #[test]
  fn default_is_mean() {
    assert_eq!(ChunkAggregation::default(), ChunkAggregation::Mean);
  }

  #[test]
  fn unknown_spelling_is_rejected() {
    assert!(serde_json::from_str::<ChunkAggregation>("\"median\"").is_err());
    assert!(serde_json::from_str::<ChunkAggregation>("\"Mean\"").is_err());
  }
}
