use super::*;

/// Extract `(start, real_len)` pairs so the pinned geometry reads as data.
fn offsets(plan: &WindowPlan, total: usize) -> Vec<(usize, usize)> {
  plan
    .spans(total)
    .iter()
    .map(|s| (s.start(), s.real_len()))
    .collect()
}

#[test]
fn window_samples_is_the_model_geometry() {
  assert_eq!(WINDOW_SAMPLES, 480_000);
  assert_eq!(
    WINDOW_SAMPLES,
    crate::embeddings::clap::audio::TARGET_SAMPLES
  );
  assert_eq!(DEFAULT_HOP_SAMPLES, 480_000);
  assert_eq!(DEFAULT_TAIL_MIN_SAMPLES, 120_000);
}

#[test]
fn default_plan_is_no_overlap_pad() {
  let plan = WindowPlan::new();
  assert_eq!(plan, WindowPlan::default());
  assert_eq!(plan.hop_samples(), 480_000);
  assert_eq!(plan.tail_policy(), TailPolicy::Pad);
  assert_eq!(TailPolicy::default(), TailPolicy::Pad);
}

#[test]
fn empty_clip_plans_no_windows() {
  assert!(WindowPlan::new().spans(0).is_empty());
}

#[test]
fn short_clip_is_one_window_regardless_of_hop() {
  // total <= window ⇒ exactly one span [0, total), whatever the hop — a smaller
  // hop must NOT re-embed the same content (textclap's single-chunk rule).
  for hop in [1u32, 120_000, 240_000, 480_000] {
    let plan = WindowPlan::new().with_hop_samples(hop);
    assert_eq!(offsets(&plan, 100), vec![(0, 100)], "hop {hop}");
    assert_eq!(offsets(&plan, 480_000), vec![(0, 480_000)], "hop {hop}");
  }
}

#[test]
fn short_clip_coverage_is_padding_aware() {
  let spans = WindowPlan::new().spans(240_000);
  assert_eq!(spans.len(), 1);
  assert_eq!(spans[0].coverage(), 0.5); // 240_000 / 480_000
  assert_eq!(WindowPlan::new().spans(480_000)[0].coverage(), 1.0);
}

#[test]
fn no_overlap_tiling_with_padded_tail() {
  // total = 1_000_000, hop = 480_000, Pad: two full windows + a 40 000 tail.
  let plan = WindowPlan::new();
  assert_eq!(
    offsets(&plan, 1_000_000),
    vec![(0, 480_000), (480_000, 480_000), (960_000, 40_000)]
  );
  // The exact 2× window boundary produces two full windows and NO empty tail.
  assert_eq!(
    offsets(&plan, 960_000),
    vec![(0, 480_000), (480_000, 480_000)]
  );
}

#[test]
fn drop_below_min_drops_the_short_tail() {
  let plan = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin {
    min_samples: 120_000,
  });
  // The 40 000-sample tail (< 120 000) is dropped; the two full windows remain.
  assert_eq!(
    offsets(&plan, 1_000_000),
    vec![(0, 480_000), (480_000, 480_000)]
  );
  // A tail at exactly the threshold is kept (inclusive `>=`).
  let plan2 = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin {
    min_samples: 40_000,
  });
  assert_eq!(
    offsets(&plan2, 1_000_000),
    vec![(0, 480_000), (480_000, 480_000), (960_000, 40_000)]
  );
}

#[test]
fn overlapping_hop_produces_full_windows_then_tails() {
  // total = 1_000_000, hop = 240_000, Pad.
  let plan = WindowPlan::new().with_hop_samples(240_000);
  assert_eq!(
    offsets(&plan, 1_000_000),
    vec![
      (0, 480_000),
      (240_000, 480_000),
      (480_000, 480_000),
      (720_000, 280_000),
      (960_000, 40_000),
    ]
  );
  // DropBelowMin keeps the 280 000 tail (>= 120 000), drops the 40 000 one.
  let dropped = plan.with_tail_policy(TailPolicy::DropBelowMin {
    min_samples: 120_000,
  });
  assert_eq!(
    offsets(&dropped, 1_000_000),
    vec![
      (0, 480_000),
      (240_000, 480_000),
      (480_000, 480_000),
      (720_000, 280_000),
    ]
  );
}

#[test]
fn window_just_over_boundary_keeps_a_one_sample_tail_under_pad() {
  assert_eq!(
    offsets(&WindowPlan::new(), 480_001),
    vec![(0, 480_000), (480_000, 1)]
  );
  // …and drops it under DropBelowMin (it is not the first span).
  let dropped = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin { min_samples: 2 });
  assert_eq!(offsets(&dropped, 480_001), vec![(0, 480_000)]);
}

#[test]
fn span_geometry_accessors() {
  let s = WindowSpan::new(720_000, 280_000);
  assert_eq!(s.start(), 720_000);
  assert_eq!(s.real_len(), 280_000);
  assert_eq!(s.end(), 1_000_000);
  assert!((s.coverage() - 280_000.0 / 480_000.0).abs() < 1e-7);
}

#[test]
fn window_embedding_pairs_embedding_with_span() {
  let mut raw = [0.0f32; crate::embeddings::clap::embedding::EMBEDDING_DIM];
  raw[0] = 1.0;
  let emb = Embedding::from_slice_normalizing(&raw).unwrap();
  let span = WindowSpan::new(0, 240_000);
  let we = WindowEmbedding::new(emb, span);
  assert_eq!(we.span(), span);
  assert_eq!(we.coverage(), 0.5);
  assert_eq!(we.embedding().as_slice()[0], 1.0);
}

#[test]
#[should_panic(expected = "hop_samples")]
fn zero_hop_setter_panics() {
  let _ = WindowPlan::new().with_hop_samples(0);
}

#[test]
#[should_panic(expected = "hop_samples")]
fn hop_past_window_setter_panics() {
  let _ = WindowPlan::new().with_hop_samples(480_001);
}

#[test]
#[should_panic(expected = "min_samples")]
fn zero_drop_min_setter_panics() {
  let _ = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin { min_samples: 0 });
}

#[cfg(feature = "serde")]
mod serde_tests {
  use super::*;

  #[test]
  fn round_trips_through_json() {
    for plan in [
      WindowPlan::new(),
      WindowPlan::new().with_hop_samples(240_000),
      WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin {
        min_samples: 120_000,
      }),
    ] {
      let json = serde_json::to_string(&plan).unwrap();
      let back: WindowPlan = serde_json::from_str(&json).unwrap();
      assert_eq!(back, plan, "round-trip drift via {json}");
    }
  }

  #[test]
  fn defaults_fill_for_a_partial_config() {
    let plan: WindowPlan = serde_json::from_str("{}").unwrap();
    assert_eq!(plan, WindowPlan::new());
    let hop_only: WindowPlan = serde_json::from_str(r#"{"hop_samples": 240000}"#).unwrap();
    assert_eq!(hop_only.hop_samples(), 240_000);
    assert_eq!(hop_only.tail_policy(), TailPolicy::Pad);
  }

  #[test]
  fn tail_policy_wire_spellings_are_pinned() {
    assert_eq!(serde_json::to_string(&TailPolicy::Pad).unwrap(), r#""pad""#);
    assert_eq!(
      serde_json::to_string(&TailPolicy::DropBelowMin {
        min_samples: 120_000
      })
      .unwrap(),
      r#"{"drop_below_min":{"min_samples":120000}}"#
    );
  }

  #[test]
  fn invalid_hop_fails_to_deserialize() {
    // A zero hop (would loop forever) and a hop past the window (would skip
    // audio) are rejected at the serde boundary, not silently accepted.
    assert!(serde_json::from_str::<WindowPlan>(r#"{"hop_samples": 0}"#).is_err());
    assert!(serde_json::from_str::<WindowPlan>(r#"{"hop_samples": 480001}"#).is_err());
  }

  #[test]
  fn invalid_tail_min_fails_to_deserialize() {
    assert!(
      serde_json::from_str::<WindowPlan>(r#"{"tail": {"drop_below_min": {"min_samples": 0}}}"#)
        .is_err()
    );
    assert!(
      serde_json::from_str::<WindowPlan>(
        r#"{"tail": {"drop_below_min": {"min_samples": 480001}}}"#
      )
      .is_err()
    );
  }
}
