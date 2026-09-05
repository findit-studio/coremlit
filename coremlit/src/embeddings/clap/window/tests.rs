use super::*;

/// Extract `(start, len)` pairs so the pinned geometry reads as data.
fn offsets(plan: &WindowPlan, total: usize) -> Vec<(usize, usize)> {
  plan
    .spans(total)
    .unwrap()
    .iter()
    .map(|s| (s.start(), s.len()))
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
  assert!(WindowPlan::new().spans(0).unwrap().is_empty());
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
fn short_clip_survives_drop_below_min() {
  // Mismatch #1: windit's DropBelowMin drops a short clip's sole span; clap's
  // contract keeps it — a clip's only representation is never dropped.
  let plan =
    WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(200_000)));
  assert_eq!(
    plan
      .spans(100_000)
      .unwrap()
      .iter()
      .map(|s| (s.start(), s.len()))
      .collect::<Vec<_>>(),
    vec![(0, 100_000)]
  );
}

#[test]
fn short_clip_coverage_is_padding_aware() {
  let spans = WindowPlan::new().spans(240_000).unwrap();
  assert_eq!(spans.len(), 1);
  assert_eq!(spans[0].coverage(), 0.5); // 240_000 / 480_000
  assert_eq!(WindowPlan::new().spans(480_000).unwrap()[0].coverage(), 1.0);
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
  let plan =
    WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(120_000)));
  // The 40 000-sample tail (< 120 000) is dropped; the two full windows remain.
  assert_eq!(
    offsets(&plan, 1_000_000),
    vec![(0, 480_000), (480_000, 480_000)]
  );
  // A tail at exactly the threshold is kept (inclusive `>=`).
  let plan2 =
    WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(40_000)));
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
  let dropped = plan.with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(120_000)));
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
  let dropped = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(2)));
  assert_eq!(offsets(&dropped, 480_001), vec![(0, 480_000)]);
}

#[test]
fn span_geometry_accessors() {
  let s = Span::new(720_000, 280_000, WINDOW_SAMPLES);
  assert_eq!(s.start(), 720_000);
  assert_eq!(s.len(), 280_000);
  assert_eq!(s.end(), 1_000_000);
  assert_eq!(s.window(), WINDOW_SAMPLES);
  // Exact, not epsilon: `coverage()` resolves in `f64`, and for a window at or
  // under `2^53` it is the IEEE division of the two exact counts — bit for bit
  // the same expression as the right-hand side. The `1e-7` slack this replaces
  // was sized for the old `f32` return and is now unfalsifiable.
  assert_eq!(s.coverage(), 280_000.0 / 480_000.0);
}

#[test]
fn window_embedding_pairs_embedding_with_span() {
  let mut raw = [0.0f32; crate::embeddings::clap::embedding::EMBEDDING_DIM];
  raw[0] = 1.0;
  let emb = Embedding::from_slice_normalizing(&raw).unwrap();
  let span = Span::new(0, 240_000, WINDOW_SAMPLES);
  let we = WindowEmbedding::new(emb, span);
  assert_eq!(we.span(), span);
  assert_eq!(we.span().coverage(), 0.5);
  assert_eq!(we.value().as_slice()[0], 1.0);
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
  let _ = WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(0)));
}

#[test]
fn huge_total_is_rejected_typed_not_panic() {
  // THE DoS REGRESSION. Pre-fix, `spans` had no cap: with hop=1 (setter-accepted)
  // it asked windit to reserve ~`usize::MAX` spans, `try_reserve_exact` overflowed
  // capacity, windit returned `AllocFailed`, and clap's `.expect()` PANICKED. The
  // O(1) cap now refuses this exact input typed, in constant time, with NO
  // allocation and NO panic — `got` is the FULL planned count.
  let plan = WindowPlan::new().with_hop_samples(1);
  let err = plan.spans(usize::MAX).unwrap_err();
  assert!(
    matches!(
      err,
      Error::Windowing(WinditError::TooManyWindows { got: usize::MAX, max })
        if max == DEFAULT_MAX_WINDOWS as usize
    ),
    "expected TooManyWindows {{ got: usize::MAX, max: {} }}, got {err:?}",
    DEFAULT_MAX_WINDOWS
  );
}

#[test]
fn hop_one_over_long_clip_is_rejected_typed() {
  // The codex [high] regression, clap-cut: a serde-supplied hop of 1 over a 20 s
  // clip would plan 960 000 windows (~1.9 GiB of retained embeddings + 960 000
  // CoreML inferences). The O(1) cap refuses it typed BEFORE materializing
  // anything — this test completing at all (no OOM, no 960 000 pushes) is half
  // the point; the exact `got` pins the FULL-count semantics.
  let plan = WindowPlan::new().with_hop_samples(1);
  let total = 2 * WINDOW_SAMPLES; // 960_000 = 20 s at 48 kHz
  let err = plan.spans(total).unwrap_err();
  assert!(
    matches!(
      err,
      Error::Windowing(WinditError::TooManyWindows { got: 960_000, max })
        if max == DEFAULT_MAX_WINDOWS as usize
    ),
    "expected TooManyWindows {{ got: 960_000, max: {} }}, got {err:?}",
    DEFAULT_MAX_WINDOWS
  );
}

#[test]
fn cap_boundary_exact_count_passes_and_plus_one_fails() {
  // hop 240_000 over 1_000_000 is the pinned 5-span geometry
  // (`overlapping_hop_produces_full_windows_then_tails`). A cap of exactly the
  // planned count admits it unchanged; one below refuses with the full count.
  let expected = vec![
    (0, 480_000),
    (240_000, 480_000),
    (480_000, 480_000),
    (720_000, 280_000),
    (960_000, 40_000),
  ];
  let at_cap = WindowPlan::new()
    .with_hop_samples(240_000)
    .with_max_windows(5);
  assert_eq!(offsets(&at_cap, 1_000_000), expected);

  let under_cap = WindowPlan::new()
    .with_hop_samples(240_000)
    .with_max_windows(4);
  let err = under_cap.spans(1_000_000).unwrap_err();
  assert!(
    matches!(
      err,
      Error::Windowing(WinditError::TooManyWindows { got: 5, max: 4 })
    ),
    "got {err:?}"
  );
}

#[test]
fn planned_windows_matches_materialized_len() {
  // The O(1) formula MUST equal the real materialized length for every
  // admissible geometry — otherwise the cap check would guard the wrong count.
  // The cap is lifted (`u32::MAX`) so only geometry, never the rail, is tested.
  let pad = |hop: u32| {
    WindowPlan::new()
      .with_hop_samples(hop)
      .with_max_windows(u32::MAX)
  };
  let drop_min = |hop: u32, min: u32| {
    WindowPlan::new()
      .with_hop_samples(hop)
      .with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(min)))
      .with_max_windows(u32::MAX)
  };
  let cases: [(WindowPlan, usize); 18] = [
    // Pad geometry (the pinned grid) + boundary totals.
    (pad(480_000), 1_000_000),
    (pad(240_000), 1_000_000),
    (pad(120_000), 1_000_000),
    (pad(100_000), 1_000_000),
    (pad(480_000), 960_000),
    (pad(480_000), 480_001),
    (pad(480_000), 0),
    (pad(480_000), 100),
    (pad(480_000), WINDOW_SAMPLES - 1),
    (pad(480_000), WINDOW_SAMPLES),
    (pad(1), WINDOW_SAMPLES), // guard-1 immunity: 1 span, no materialization blowup
    // DropBelowMin geometry (the pinned 1_000_000 @ 120_000/40_000 cases + edges).
    (drop_min(480_000, 120_000), 1_000_000),
    (drop_min(480_000, 40_000), 1_000_000),
    (drop_min(240_000, 120_000), 1_000_000),
    (drop_min(240_000, 40_000), 1_000_000),
    (drop_min(480_000, 2), 480_001),
    (drop_min(480_000, 120_000), 100),
    (drop_min(480_000, 120_000), 0),
  ];
  for (plan, total) in cases {
    assert_eq!(
      plan.planned_windows(total),
      plan.spans(total).unwrap().len(),
      "planned_windows != materialized len for hop={} tail={:?} total={total}",
      plan.hop_samples(),
      plan.tail_policy(),
    );
  }
}

#[test]
fn short_clip_never_trips_cap() {
  // Guard-1 immunity: a short clip is one span regardless of hop/cap, so even
  // the tightest cap admits it (planned == 1 <= any valid cap).
  let plan = WindowPlan::new().with_max_windows(1).with_hop_samples(1);
  let spans = plan.spans(WINDOW_SAMPLES).unwrap();
  assert_eq!(spans.len(), 1);
  assert_eq!((spans[0].start(), spans[0].len()), (0, WINDOW_SAMPLES));
}

#[test]
#[should_panic(expected = "max_windows")]
fn zero_max_windows_setter_panics() {
  let _ = WindowPlan::new().with_max_windows(0);
}

#[cfg(feature = "serde")]
mod serde_tests {
  use super::*;

  #[test]
  fn round_trips_through_json() {
    for plan in [
      WindowPlan::new(),
      WindowPlan::new().with_hop_samples(240_000),
      WindowPlan::new().with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(120_000))),
      WindowPlan::new()
        .with_hop_samples(240_000)
        .with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(120_000)))
        .with_max_windows(50_000),
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
    // The omitted cap fills the default — it is default-on for every config.
    assert_eq!(plan.max_windows(), DEFAULT_MAX_WINDOWS);
    let hop_only: WindowPlan = serde_json::from_str(r#"{"hop_samples": 240000}"#).unwrap();
    assert_eq!(hop_only.hop_samples(), 240_000);
    assert_eq!(hop_only.tail_policy(), TailPolicy::Pad);
    assert_eq!(hop_only.max_windows(), DEFAULT_MAX_WINDOWS);
  }

  /// The [`TailPolicy`] document, pinned byte-exactly in BOTH directions, plus
  /// the whole-plan document that carries it.
  ///
  /// The form is windit 0.4's: adjacently tagged, `kind` naming the variant and
  /// `value` carrying the payload. The externally tagged `"pad"` /
  /// `{"drop_below_min":{…}}` spellings this replaced are asserted to be
  /// REFUSED, so the wire break is a fact this suite states rather than a
  /// comment somebody has to trust.
  #[test]
  fn tail_policy_wire_spellings_are_pinned() {
    for (policy, doc) in [
      (TailPolicy::Pad, r#"{"kind":"pad"}"#),
      (
        TailPolicy::DropBelowMin(DropBelowMin::new(120_000)),
        r#"{"kind":"drop_below_min","value":{"min_samples":120000}}"#,
      ),
    ] {
      assert_eq!(serde_json::to_string(&policy).unwrap(), doc);
      assert_eq!(serde_json::from_str::<TailPolicy>(doc).unwrap(), policy);
    }
    // The full-plan wire form pins the `max_windows` field's spelling and the
    // shape the tail reaches a config file in.
    assert_eq!(
      serde_json::to_string(&WindowPlan::new()).unwrap(),
      r#"{"hop_samples":480000,"tail":{"kind":"pad"},"max_windows":100000}"#
    );
    let doc = r#"{"hop_samples":240000,"tail":{"kind":"drop_below_min","value":{"min_samples":120000}},"max_windows":50000}"#;
    let plan = WindowPlan::new()
      .with_hop_samples(240_000)
      .with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(120_000)))
      .with_max_windows(50_000);
    assert_eq!(serde_json::to_string(&plan).unwrap(), doc);
    assert_eq!(serde_json::from_str::<WindowPlan>(doc).unwrap(), plan);
    // The retired externally tagged form is refused, both variants.
    assert!(serde_json::from_str::<TailPolicy>(r#""pad""#).is_err());
    assert!(
      serde_json::from_str::<TailPolicy>(r#"{"drop_below_min":{"min_samples":120000}}"#).is_err()
    );
  }

  /// Every [`TailPolicy`] variant and the plans that carry it survive a
  /// NON-self-describing format.
  ///
  /// The falsifier for the `is_human_readable` split. postcard carries no field
  /// names, so the adjacently tagged form cannot be read back at all: serde
  /// writes the tag as a struct FIELD and reads it through
  /// `deserialize_identifier`, which postcard refuses. Measured on the derived
  /// `tail_policy_serde::Document` mirror — the exact shape the branch replaced
  /// — EVERY variant serializes and then fails to deserialize, `Pad` (bytes
  /// `[0]`) and `DropBelowMin` (`[1, …]`) alike; the unit variant additionally
  /// has its missing content read through `deserialize_any`. Drop the
  /// `is_human_readable` branch and all four cases below red — both policies
  /// and both plans, the DEFAULT `WindowPlan` included — while every JSON pin
  /// above stays green.
  #[test]
  fn every_variant_round_trips_through_a_non_self_describing_format() {
    for policy in [
      TailPolicy::Pad,
      TailPolicy::DropBelowMin(DropBelowMin::new(120_000)),
    ] {
      let bytes = postcard::to_allocvec(&policy).unwrap();
      assert_eq!(
        postcard::from_bytes::<TailPolicy>(&bytes).unwrap(),
        policy,
        "postcard round-trip lost {policy:?} (bytes {bytes:?})"
      );
    }
    for plan in [
      WindowPlan::new(),
      WindowPlan::new()
        .with_hop_samples(240_000)
        .with_tail_policy(TailPolicy::DropBelowMin(DropBelowMin::new(120_000)))
        .with_max_windows(50_000),
    ] {
      let bytes = postcard::to_allocvec(&plan).unwrap();
      assert_eq!(
        postcard::from_bytes::<WindowPlan>(&bytes).unwrap(),
        plan,
        "postcard round-trip lost {plan:?} (bytes {bytes:?})"
      );
    }
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
      serde_json::from_str::<WindowPlan>(
        r#"{"tail": {"kind": "drop_below_min", "value": {"min_samples": 0}}}"#
      )
      .is_err()
    );
    assert!(
      serde_json::from_str::<WindowPlan>(
        r#"{"tail": {"kind": "drop_below_min", "value": {"min_samples": 480001}}}"#
      )
      .is_err()
    );
  }

  #[test]
  fn zero_max_windows_fails_to_deserialize() {
    // A zero cap can never embed any clip; the validated repr rejects it, just
    // as the setter panics on it.
    assert!(serde_json::from_str::<WindowPlan>(r#"{"max_windows": 0}"#).is_err());
  }

  /// A MISSPELLED key is REFUSED, not silently discarded.
  ///
  /// Every field defaults, so without `deny_unknown_fields` `{"max_window":1}`
  /// — the plural dropped — deserializes with the typo thrown away and
  /// `max_windows` filled from [`DEFAULT_MAX_WINDOWS`]: an operator capping the
  /// door at ONE window gets 100 000 instead, so a misspelled RESOURCE LIMIT
  /// silently becomes up to 100 000 CoreML predictions. A misspelled `hop` or
  /// `tail` key changes the embedded geometry the same silent way. Each
  /// rejection is paired with the spelling that must still parse, so this
  /// cannot pass by the whole struct having stopped deserializing.
  #[test]
  fn a_misspelled_key_is_refused_rather_than_silently_defaulted() {
    for (doc, key) in [
      (r#"{"max_window":1}"#, "max_window"),
      (r#"{"hop":240000}"#, "hop"),
      (r#"{"tail_policy":{"kind":"pad"}}"#, "tail_policy"),
      // Beside a well-spelled key, where a permissive impl is likeliest to let
      // it through.
      (r#"{"hop_samples":240000,"max_window":1}"#, "max_window"),
    ] {
      let err = match serde_json::from_str::<WindowPlan>(doc) {
        Ok(plan) => panic!(
          "{doc} must be refused; it deserialized to {plan:?} (max_windows {})",
          plan.max_windows()
        ),
        Err(e) => e.to_string(),
      };
      assert!(
        err.contains(key),
        "the refusal must name {key}, got {err:?}"
      );
    }
    // Positive control: the correct spellings still parse and land.
    let ok: WindowPlan = serde_json::from_str(r#"{"hop_samples":240000,"max_windows":1}"#).unwrap();
    assert_eq!(ok.hop_samples(), 240_000);
    assert_eq!(ok.max_windows(), 1);
  }

  /// The rejection MESSAGE, pinned byte-exactly. `TailPolicy::DropBelowMin`
  /// carries a payload struct that shares its variant's name, so interpolating
  /// the PAYLOAD renders `DropBelowMin { min_samples: 0 }` — what the
  /// struct-shaped variant rendered before it was newtyped. Interpolating the
  /// whole policy instead would double the name, and nothing else asserts this.
  #[test]
  fn invalid_tail_min_rejection_message_names_the_payload_once() {
    let err = WindowPlan::try_from(WindowPlanRepr {
      hop_samples: DEFAULT_HOP_SAMPLES,
      tail: TailPolicy::DropBelowMin(DropBelowMin::new(0)),
      max_windows: DEFAULT_MAX_WINDOWS,
    })
    .unwrap_err();
    assert_eq!(
      err,
      "tail DropBelowMin.min_samples must be > 0 and <= WINDOW_SAMPLES \
       (480000), got DropBelowMin { min_samples: 0 }"
    );
  }
}
