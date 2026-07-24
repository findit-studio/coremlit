use super::*;

/// Flatten spans to `(start, end)` pairs for terse geometry assertions.
fn offsets(plan: &WindowPlan, total: usize) -> Vec<(usize, usize)> {
  plan
    .spans(total)
    .iter()
    .map(|s| (s.start(), s.end()))
    .collect()
}

/// The naive soundevents `chunk_slices` geometry: starts at 0, H, 2H, … while
/// `start < total`, each chunk `[start, min(start + W, total))`. The reference
/// CED's multi-tail continuation must match for `total > WINDOW_SAMPLES` under
/// `TailPolicy::Pad` (spec §2: "matching soundevents' chunk_slices overlapped
/// semantics"). For `total <= WINDOW_SAMPLES` the short-clip guard deliberately
/// deviates (exactly one span regardless of hop — spec-settled), so this
/// reference is only consulted for long clips.
fn chunk_slices_reference(total: usize, window: usize, hop: usize) -> Vec<(usize, usize)> {
  let mut out = Vec::new();
  let mut start = 0usize;
  while start < total {
    out.push((start, total.min(start + window)));
    start += hop;
  }
  out
}

#[test]
fn window_samples_is_the_model_geometry() {
  assert_eq!(WINDOW_SAMPLES, 160_000); // 10 s × 16 kHz
  assert_eq!(DEFAULT_HOP_SAMPLES as usize, WINDOW_SAMPLES);
}

#[test]
fn default_plan_is_no_overlap_pad() {
  let plan = WindowPlan::default();
  assert_eq!(plan, WindowPlan::new());
  assert_eq!(plan.hop_samples(), DEFAULT_HOP_SAMPLES);
  assert_eq!(plan.tail_policy(), TailPolicy::Pad);
}

#[test]
fn empty_clip_plans_no_windows() {
  assert!(WindowPlan::new().spans(0).is_empty());
}

#[test]
fn short_clip_is_one_window_regardless_of_hop() {
  // Clap contract 1: total <= window ⇒ exactly one span, even under a tiny hop
  // (where a literal soundevents chunk_slices would emit several sub-window
  // chunks — the recorded deviation, spec §2).
  let plan = WindowPlan::new().with_hop_samples(1_000);
  assert_eq!(offsets(&plan, 50_000), vec![(0, 50_000)]);
  assert_eq!(offsets(&plan, WINDOW_SAMPLES), vec![(0, WINDOW_SAMPLES)]);
}

#[test]
fn short_clip_survives_drop_below_min() {
  // A short clip's sole span is never dropped — there is nothing else to
  // represent it.
  let plan = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin {
    min_samples: 100_000,
  });
  assert_eq!(offsets(&plan, 50_000), vec![(0, 50_000)]);
}

#[test]
fn short_clip_coverage_is_padding_aware() {
  let spans = WindowPlan::new().spans(40_000);
  assert_eq!(spans.len(), 1);
  let coverage = spans[0].coverage();
  assert!(
    (coverage - 0.25).abs() < 1e-6,
    "40_000 / 160_000 = 0.25, got {coverage}"
  );
}

#[test]
fn no_overlap_tiling_with_padded_tail() {
  // 400_000 = 2 full windows + an 80_000-sample tail, kept under Pad.
  let plan = WindowPlan::new();
  assert_eq!(
    offsets(&plan, 400_000),
    vec![(0, 160_000), (160_000, 320_000), (320_000, 400_000)]
  );
}

#[test]
fn drop_below_min_drops_the_short_tail() {
  let plan = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin {
    min_samples: 100_000,
  });
  // The 80_000-sample tail is below the 100_000 threshold.
  assert_eq!(
    offsets(&plan, 400_000),
    vec![(0, 160_000), (160_000, 320_000)]
  );
  // A tail AT the threshold is kept.
  assert_eq!(
    offsets(&plan, 420_000),
    vec![(0, 160_000), (160_000, 320_000), (320_000, 420_000)]
  );
}

#[test]
fn overlapping_hop_produces_full_windows_then_tails() {
  // hop 80_000 over 400_000: full windows at 0/80k/160k/240k, then the
  // multi-tail continuation keeps striding past windit's first-tail stop
  // (clap contract 2), emitting the 80k tail at 320k.
  let plan = WindowPlan::new().with_hop_samples(80_000);
  assert_eq!(
    offsets(&plan, 400_000),
    vec![
      (0, 160_000),
      (80_000, 240_000),
      (160_000, 320_000),
      (240_000, 400_000),
      (320_000, 400_000),
    ]
  );
}

#[test]
fn spans_match_soundevents_chunk_slices_for_long_clips() {
  // The Pad-policy geometry for total > WINDOW_SAMPLES is exactly soundevents'
  // chunk_slices (start, end) sequence — including EVERY progressively shorter
  // tail an overlapped hop generates.
  for (total, hop) in [
    (400_000usize, 80_000u32),
    (500_000, 60_000),
    (160_001, 160_000),
    (1_000_000, 160_000),
    (330_000, 100_000),
  ] {
    let plan = WindowPlan::new().with_hop_samples(hop);
    assert_eq!(
      offsets(&plan, total),
      chunk_slices_reference(total, WINDOW_SAMPLES, hop as usize),
      "total={total} hop={hop}"
    );
  }
}

#[test]
fn window_just_over_boundary_keeps_a_one_sample_tail_under_pad() {
  assert_eq!(
    offsets(&WindowPlan::new(), WINDOW_SAMPLES + 1),
    vec![(0, WINDOW_SAMPLES), (WINDOW_SAMPLES, WINDOW_SAMPLES + 1)]
  );
}

#[test]
fn span_geometry_accessors() {
  let spans = WindowPlan::new().spans(400_000);
  let tail = spans[2];
  assert_eq!(tail.start(), 320_000);
  assert_eq!(tail.len(), 80_000);
  assert_eq!(tail.end(), 400_000);
  assert!((tail.coverage() - 0.5).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "hop_samples")]
fn zero_hop_setter_panics() {
  let _ = WindowPlan::new().with_hop_samples(0);
}

#[test]
#[should_panic(expected = "hop_samples")]
fn hop_past_window_setter_panics() {
  // hop > window would leave gaps of un-classified audio (the soundevents
  // sparse-skim mode is a recorded non-goal).
  let _ = WindowPlan::new().with_hop_samples(WINDOW_SAMPLES as u32 + 1);
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
    let plan =
      WindowPlan::new()
        .with_hop_samples(80_000)
        .with_tail_policy(TailPolicy::DropBelowMin {
          min_samples: 40_000,
        });
    let json = serde_json::to_string(&plan).unwrap();
    let back: WindowPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(back, plan);
  }

  #[test]
  fn defaults_fill_for_a_partial_config() {
    let plan: WindowPlan = serde_json::from_str("{}").unwrap();
    assert_eq!(plan, WindowPlan::new());
  }

  #[test]
  fn tail_policy_wire_spellings_are_pinned() {
    // Wildcard-free: a new variant fails to compile until its spelling is
    // pinned here (the ChunkAggregation golden pattern, `aggregate/tests.rs`).
    for kind in [
      TailPolicy::Pad,
      TailPolicy::DropBelowMin {
        min_samples: 40_000,
      },
    ] {
      let expected = match kind {
        TailPolicy::Pad => "\"pad\"".to_string(),
        TailPolicy::DropBelowMin { min_samples } => {
          format!("{{\"drop_below_min\":{{\"min_samples\":{min_samples}}}}}")
        }
      };
      assert_eq!(serde_json::to_string(&kind).unwrap(), expected);
      let back: TailPolicy = serde_json::from_str(&expected).unwrap();
      assert_eq!(back, kind);
    }
    assert_eq!(
      serde_json::to_string(&WindowPlan::new()).unwrap(),
      "{\"hop_samples\":160000,\"tail\":\"pad\"}"
    );
  }

  #[test]
  fn invalid_hop_fails_to_deserialize() {
    // The validated repr makes the checked setters unbypassable via serde:
    // hop 0 would loop forever, hop > window would skip audio.
    assert!(serde_json::from_str::<WindowPlan>("{\"hop_samples\":0}").is_err());
    assert!(serde_json::from_str::<WindowPlan>("{\"hop_samples\":160001}").is_err());
  }

  #[test]
  fn invalid_tail_min_fails_to_deserialize() {
    let json = "{\"tail\":{\"drop_below_min\":{\"min_samples\":0}}}";
    assert!(serde_json::from_str::<WindowPlan>(json).is_err());
    let json = "{\"tail\":{\"drop_below_min\":{\"min_samples\":160001}}}";
    assert!(serde_json::from_str::<WindowPlan>(json).is_err());
  }
}
