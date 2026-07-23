use super::*;
use crate::audio::ced::{NUM_CLASSES, WINDOW_SAMPLES, window::Span};

/// Deterministic scripted scores: a fixed LCG over the full class range, with
/// deliberate duplicates injected so the tie-break is exercised.
fn scripted_scores() -> Vec<f32> {
  let mut state = 0x2545F4914F6CDD1Du64;
  let mut out: Vec<f32> = (0..NUM_CLASSES)
    .map(|_| {
      state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      // Map to [-6, 6) — a realistic logit range, and strictly below the 6.5
      // tie value injected next, so the tie triple is guaranteed maximal.
      ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 12.0
    })
    .collect();
  // Ties: three classes share the maximum, two share another value.
  out[10] = 6.5;
  out[200] = 6.5;
  out[500] = 6.5;
  out[3] = -1.25;
  out[400] = -1.25;
  out
}

/// Reference ranking: full sort by (score desc via total_cmp, index asc) —
/// soundevents' RankedScore contract, spelled naively.
fn sorted_reference(scores: &[f32]) -> Vec<(usize, f32)> {
  let mut pairs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
  pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
  pairs
}

#[test]
fn sigmoid_matches_the_soundevents_form() {
  assert_eq!(sigmoid(0.0), 0.5);
  assert!((sigmoid(2.0) - 0.880_797).abs() < 1e-6);
  assert!((sigmoid(-2.0) - 0.119_203).abs() < 1e-6);
  // Monotonic on a coarse grid.
  let mut prev = f32::NEG_INFINITY;
  for i in -50..=50 {
    let v = sigmoid(i as f32 / 5.0);
    assert!(v > prev);
    prev = v;
  }
}

#[test]
fn top_k_matches_a_full_sort_reference() {
  let scores = scripted_scores();
  let reference = sorted_reference(&scores);
  for k in [1usize, 5, 50, NUM_CLASSES] {
    let preds = top_k_from_scores(scores.iter().copied().enumerate(), k, sigmoid).unwrap();
    assert_eq!(preds.len(), k.min(NUM_CLASSES));
    for (p, &(ref_index, ref_score)) in preds.iter().zip(reference.iter()) {
      assert_eq!(p.index(), ref_index, "k={k}");
      assert_eq!(p.confidence(), sigmoid(ref_score), "k={k}");
    }
  }
}

#[test]
fn ties_break_by_ascending_class_index() {
  // The three 6.5-scored classes are the maximum: they must come out first,
  // ordered 10, 200, 500 (soundevents' RankedScore contract).
  let scores = scripted_scores();
  let preds = top_k_from_scores(scores.iter().copied().enumerate(), 3, sigmoid).unwrap();
  let indices: Vec<usize> = preds.iter().map(|p| p.index()).collect();
  assert_eq!(indices, vec![10, 200, 500]);
}

#[test]
fn zero_k_is_empty_not_an_error() {
  let scores = scripted_scores();
  let preds = top_k_from_scores(scores.iter().copied().enumerate(), 0, sigmoid).unwrap();
  assert!(preds.is_empty());
}

#[test]
fn oversized_k_saturates_at_num_classes() {
  let scores = scripted_scores();
  let preds = top_k_from_scores(
    scores.iter().copied().enumerate(),
    NUM_CLASSES + 100,
    sigmoid,
  )
  .unwrap();
  assert_eq!(preds.len(), NUM_CLASSES);
}

#[test]
fn sigmoid_at_extraction_equals_sigmoid_then_rank() {
  // Monotonicity: ranking raw logits and mapping sigmoid at extraction must
  // give the SAME order and values as pre-mapping every score (the soundevents
  // trick that avoids a 527-element sort per call).
  let scores = scripted_scores();
  let at_extraction = top_k_from_scores(scores.iter().copied().enumerate(), 20, sigmoid).unwrap();
  let confidences: Vec<f32> = scores.iter().copied().map(sigmoid).collect();
  let pre_mapped = top_k_from_scores(confidences.iter().copied().enumerate(), 20, |c| c).unwrap();
  for (a, b) in at_extraction.iter().zip(pre_mapped.iter()) {
    assert_eq!(a.index(), b.index());
    assert_eq!(a.confidence(), b.confidence());
  }
}

#[test]
fn event_prediction_round_trips_known_rows() {
  // /m/09x0r "Speech" is class 0 in the released rated label set.
  let p = EventPrediction::from_confidence(0, 0.75).unwrap();
  assert_eq!(p.index(), 0);
  assert_eq!(p.id(), "/m/09x0r");
  assert_eq!(p.name(), "Speech");
  assert_eq!(p.confidence(), 0.75);
  assert_eq!(p.event().index(), 0);
  // The last valid row exists; one past it is the typed defensive error.
  assert!(EventPrediction::from_confidence(NUM_CLASSES - 1, 0.5).is_ok());
  let err = EventPrediction::from_confidence(NUM_CLASSES, 0.5).unwrap_err();
  assert!(
    matches!(err, crate::audio::ced::Error::UnknownClassIndex { index } if index == NUM_CLASSES),
    "got {err:?}"
  );
}

#[test]
fn confidences_hold_the_class_count_invariant() {
  let c = Confidences::new(vec![0.5; NUM_CLASSES]);
  assert_eq!(c.as_slice().len(), NUM_CLASSES);
}

#[test]
#[should_panic(expected = "NUM_CLASSES")]
fn confidences_reject_a_wrong_length_vector() {
  // Internal invariant (pub(crate) constructor): a wrong-length vector is a
  // module bug, not a caller error — assert, never a silent truncation.
  let _ = Confidences::new(vec![0.5; NUM_CLASSES - 1]);
}

#[test]
fn from_logits_maps_sigmoid_elementwise() {
  let mut logits = vec![0.0f32; NUM_CLASSES];
  logits[0] = 2.0;
  logits[526] = -2.0;
  let c = Confidences::from_logits(&logits);
  assert_eq!(c.as_slice()[0], sigmoid(2.0));
  assert_eq!(c.as_slice()[1], 0.5);
  assert_eq!(c.as_slice()[526], sigmoid(-2.0));
}

#[test]
fn confidences_top_k_ranks_in_confidence_space() {
  let mut values = vec![0.1f32; NUM_CLASSES];
  values[7] = 0.9;
  values[300] = 0.8;
  let c = Confidences::new(values);
  let preds = c.top_k(2).unwrap();
  assert_eq!(preds[0].index(), 7);
  assert_eq!(preds[0].confidence(), 0.9);
  assert_eq!(preds[1].index(), 300);
  assert!(c.top_k(0).unwrap().is_empty());
}

#[test]
fn window_confidences_pair_values_with_spans() {
  let span = Span::new(0, WINDOW_SAMPLES, WINDOW_SAMPLES);
  let w = WindowConfidences::new(Confidences::new(vec![0.5; NUM_CLASSES]), span);
  assert_eq!(w.span().start(), 0);
  assert_eq!(w.value().as_slice().len(), NUM_CLASSES);
}
