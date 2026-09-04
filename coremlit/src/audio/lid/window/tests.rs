use super::*;
use crate::audio::lid::frame_count;

/// A naive reference plan, written the obvious way — collect every full-length
/// window, then apply the tail rule — so the real [`WindowPlan::spans`] (which
/// delegates its head to windit and derives its tail arithmetically) is checked
/// against something that shares no code with it.
fn reference_spans(plan: &WindowPlan, total: usize) -> Vec<(usize, usize)> {
  let (window, hop) = (plan.window_samples() as usize, plan.hop_samples() as usize);
  if total == 0 {
    return Vec::new();
  }
  if total <= window {
    return vec![(0, total)];
  }
  let mut out = Vec::new();
  let mut start = 0;
  while start + window <= total {
    out.push((start, window));
    start += hop;
  }
  let last_full_start = start - hop;
  if last_full_start + window < total {
    match plan.tail_policy() {
      TailPolicy::Drop => {}
      TailPolicy::SlideBack => out.push((total - window, window)),
      TailPolicy::Partial => {
        let tail_start = last_full_start + hop;
        let len = total - tail_start;
        if len >= MIN_SAMPLES {
          out.push((tail_start, len));
        }
      }
    }
  }
  out
}

fn materialized(plan: &WindowPlan, total: usize) -> Vec<(usize, usize)> {
  plan
    .spans(total)
    .expect("plan must be admissible")
    .into_iter()
    .map(|s| (s.start(), s.len()))
    .collect()
}

// ── Defaults ────────────────────────────────────────────────────────────────

/// The shipped defaults, and the two facts the window length rests on: it is a
/// frame count the graph accepts, and it is exactly 10 s.
#[test]
fn defaults_are_the_documented_geometry() {
  assert_eq!(DEFAULT_WINDOW_SAMPLES, 160_000);
  assert_eq!(DEFAULT_HOP_SAMPLES, DEFAULT_WINDOW_SAMPLES);
  assert_eq!(DEFAULT_MAX_WINDOWS, 100_000);

  let plan = WindowPlan::new();
  assert_eq!(plan, WindowPlan::default());
  assert_eq!(plan.window_samples(), DEFAULT_WINDOW_SAMPLES);
  assert_eq!(plan.hop_samples(), DEFAULT_HOP_SAMPLES);
  assert_eq!(plan.tail_policy(), TailPolicy::SlideBack);
  assert_eq!(plan.max_windows(), DEFAULT_MAX_WINDOWS);

  assert_eq!(frame_count(DEFAULT_WINDOW_SAMPLES as usize), 1_001);
  assert_eq!(DEFAULT_WINDOW_SAMPLES as usize % 16_000, 0);
}

/// `TailPolicy::default()` is the variant `WindowPlan::new` picks — one source
/// of truth, so the serde `#[serde(default)]` path and the constructor cannot
/// drift apart.
#[test]
fn tail_policy_default_matches_the_plan_default() {
  assert_eq!(TailPolicy::default(), WindowPlan::new().tail_policy());
}

// ── Geometry ────────────────────────────────────────────────────────────────

/// A clip no longer than one window is exactly one span covering all of it —
/// whatever the hop and tail policy, including the policies that would
/// otherwise drop a short span. This is the guard that makes the long path
/// agree with the single-shot one.
#[test]
fn a_clip_within_one_window_is_a_single_full_coverage_span() {
  for tail in [TailPolicy::SlideBack, TailPolicy::Partial, TailPolicy::Drop] {
    let plan = WindowPlan::new()
      .with_geometry(160_000, 40_000)
      .with_tail_policy(tail);
    for total in [MIN_SAMPLES, 50_000, 159_999, 160_000] {
      let spans = plan.spans(total).expect("admissible");
      assert_eq!(spans.len(), 1, "{tail:?} at {total}");
      assert_eq!(spans[0].start(), 0);
      assert_eq!(spans[0].len(), total);
      assert_eq!(spans[0].window(), 160_000);
    }
  }
  assert!(WindowPlan::new().spans(0).expect("admissible").is_empty());
}

/// The three tail policies on one worked clip, so the difference between them
/// is visible as offsets rather than prose. 250 000 samples at a 100 000-sample
/// window and hop: two full windows, then 50 000 samples the head cannot reach.
#[test]
fn the_three_tail_policies_differ_only_in_the_final_span() {
  let base = WindowPlan::new().with_geometry(100_000, 100_000);
  let total = 250_000;

  assert_eq!(
    materialized(&base.with_tail_policy(TailPolicy::SlideBack), total),
    [(0, 100_000), (100_000, 100_000), (150_000, 100_000)],
    "SlideBack ends flush with the clip, full length, overlapping by 50 000"
  );
  assert_eq!(
    materialized(&base.with_tail_policy(TailPolicy::Partial), total),
    [(0, 100_000), (100_000, 100_000), (200_000, 50_000)],
    "Partial scores the leftover at its own length"
  );
  assert_eq!(
    materialized(&base.with_tail_policy(TailPolicy::Drop), total),
    [(0, 100_000), (100_000, 100_000)],
    "Drop discards the leftover"
  );
}

/// A clip that is an exact multiple of the hop has no uncovered tail, so all
/// three policies agree and none of them adds a redundant span.
#[test]
fn an_exact_fit_produces_no_tail_under_any_policy() {
  let base = WindowPlan::new().with_geometry(100_000, 50_000);
  for total in [100_000, 150_000, 200_000, 500_000] {
    let expected = materialized(&base.with_tail_policy(TailPolicy::Drop), total);
    for tail in [TailPolicy::SlideBack, TailPolicy::Partial] {
      assert_eq!(
        materialized(&base.with_tail_policy(tail), total),
        expected,
        "{tail:?} at {total} must not add a span over already-covered audio"
      );
    }
  }
}

/// A `Partial` tail below the graph's own minimum is dropped rather than
/// planned — the model would refuse it — and that is the only case where
/// `Partial` fails to cover the whole clip.
#[test]
fn a_partial_tail_shorter_than_the_graph_accepts_is_dropped() {
  let plan = WindowPlan::new()
    .with_geometry(100_000, 100_000)
    .with_tail_policy(TailPolicy::Partial);

  // One sample below MIN_SAMPLES of leftover: dropped.
  let short = materialized(&plan, 100_000 + MIN_SAMPLES - 1);
  assert_eq!(short, [(0, 100_000)]);

  // Exactly MIN_SAMPLES: kept, and it is the shortest clip the graph accepts.
  let kept = materialized(&plan, 100_000 + MIN_SAMPLES);
  assert_eq!(kept, [(0, 100_000), (100_000, MIN_SAMPLES)]);
  assert_eq!(frame_count(MIN_SAMPLES), crate::audio::lid::MIN_FRAMES);
}

/// Every planned span is a length the graph can actually score, at every
/// geometry — the property that lets `log_probabilities_windows` promise no
/// window is ever rejected mid-clip for its size.
#[test]
fn every_planned_span_is_a_scoreable_length() {
  for window in [
    MIN_SAMPLES as u32,
    48_000,
    DEFAULT_WINDOW_SAMPLES,
    MAX_SAMPLES as u32,
  ] {
    for hop_div in [1, 2, 3] {
      for tail in [TailPolicy::SlideBack, TailPolicy::Partial, TailPolicy::Drop] {
        let plan = WindowPlan::new()
          .with_geometry(window, window / hop_div)
          .with_tail_policy(tail);
        for total in [
          MIN_SAMPLES,
          window as usize,
          window as usize + 1,
          window as usize * 3 + 7,
          1_000_000,
        ] {
          for span in plan.spans(total).expect("admissible") {
            let frames = frame_count(span.len());
            assert!(
              (crate::audio::lid::MIN_FRAMES..=crate::audio::lid::MAX_FRAMES).contains(&frames),
              "{tail:?} window {window} hop {} total {total}: span {span:?} is {frames} frames",
              window / hop_div
            );
            assert!(span.end() <= total, "span {span:?} runs past {total}");
          }
        }
      }
    }
  }
}

/// Coverage, stated as the property each policy promises: `SlideBack` and
/// `Partial` leave nothing behind (bar a sub-`MIN_SAMPLES` sliver under
/// `Partial`), and `Drop` never discards as much as one hop.
#[test]
fn coverage_matches_what_each_policy_promises() {
  for hop_div in [1, 2, 3] {
    let window = 100_000u32;
    let hop = window / hop_div;
    for total in [100_001, 123_456, 250_000, 400_000, 999_999] {
      let covered_to = |tail| {
        let plan = WindowPlan::new()
          .with_geometry(window, hop)
          .with_tail_policy(tail);
        plan
          .spans(total)
          .expect("admissible")
          .iter()
          .map(Span::end)
          .max()
          .expect("non-empty")
      };
      assert_eq!(
        covered_to(TailPolicy::SlideBack),
        total,
        "SlideBack {total}"
      );
      let partial_gap = total - covered_to(TailPolicy::Partial);
      assert!(
        partial_gap < MIN_SAMPLES,
        "Partial left {partial_gap} at {total}"
      );
      let dropped = total - covered_to(TailPolicy::Drop);
      assert!(dropped < hop as usize, "Drop left {dropped} at {total}");
    }
  }
}

/// The O(1) count IS the materialized length, over a sweep that crosses every
/// boundary the branch structure has. A `debug_assert_eq!` in `spans` says the
/// same thing on every call; this makes it a first-class gate.
#[test]
fn planned_windows_matches_materialized_len() {
  for window in [2_000u32, 7_000, 100_000] {
    for hop in [1_000u32, 1_999, window / 2, window] {
      if hop == 0 || hop > window {
        continue;
      }
      for tail in [TailPolicy::SlideBack, TailPolicy::Partial, TailPolicy::Drop] {
        let plan = WindowPlan::new()
          .with_geometry(window, hop)
          .with_tail_policy(tail);
        for total in (0..window as usize * 4).step_by(313) {
          let spans = plan.spans(total).expect("admissible");
          assert_eq!(
            plan.planned_windows(total),
            spans.len(),
            "window {window} hop {hop} {tail:?} total {total}"
          );
          let reference: Vec<(usize, usize)> = spans.iter().map(|s| (s.start(), s.len())).collect();
          assert_eq!(
            reference,
            reference_spans(&plan, total),
            "window {window} hop {hop} {tail:?} total {total}"
          );
        }
      }
    }
  }
}

// ── Resource cap ────────────────────────────────────────────────────────────

/// The cap refuses an over-plan BEFORE materializing anything, and reports the
/// full would-be count rather than windit's abort-at-`max + 1`.
#[test]
fn the_cap_refuses_the_full_planned_count_before_materializing() {
  let plan = WindowPlan::new()
    .with_geometry(160_000, 1)
    .with_max_windows(10);
  let error = plan.spans(200_000).expect_err("must refuse");
  let Error::Windowing(WinditError::TooManyWindows { got, max }) = error else {
    panic!("expected TooManyWindows, got {error:?}");
  };
  assert_eq!(max, 10);
  assert_eq!(got, 40_001, "the FULL planned count, not max + 1");

  // Exactly at the cap is admitted, one over is not.
  let sized = WindowPlan::new().with_geometry(1_000_000_u32.min(MAX_SAMPLES as u32), 10_000);
  let at_cap = sized.with_max_windows(3);
  assert_eq!(
    at_cap
      .spans(MAX_SAMPLES + 20_000)
      .expect("admissible")
      .len(),
    3
  );
  assert!(
    sized
      .with_max_windows(2)
      .spans(MAX_SAMPLES + 20_000)
      .is_err()
  );
}

/// A clip within one window is one span, so it is admitted by the smallest
/// legal cap — the cap is a rail against hop-abuse, not a clip-length limit.
#[test]
fn the_smallest_cap_still_admits_a_single_window_clip() {
  let plan = WindowPlan::new().with_max_windows(1);
  assert_eq!(
    plan
      .spans(DEFAULT_WINDOW_SAMPLES as usize)
      .expect("admissible")
      .len(),
    1
  );
  assert!(plan.spans(DEFAULT_WINDOW_SAMPLES as usize + 1).is_err());
}

// ── Validation ──────────────────────────────────────────────────────────────

/// The geometry setter accepts exactly the pair the graph and the stride
/// permit, and round-trips.
#[test]
fn the_geometry_setter_round_trips_the_valid_pair() {
  let plan = WindowPlan::new().with_geometry(MIN_SAMPLES as u32, 1);
  assert_eq!(plan.window_samples(), MIN_SAMPLES as u32);
  assert_eq!(plan.hop_samples(), 1);

  let plan = WindowPlan::new().with_geometry(MAX_SAMPLES as u32, MAX_SAMPLES as u32);
  assert_eq!(plan.window_samples(), MAX_SAMPLES as u32);

  let mut mutated = WindowPlan::new();
  mutated
    .set_geometry(48_000, 16_000)
    .set_tail_policy(TailPolicy::Partial)
    .set_max_windows(7);
  assert_eq!(mutated.window_samples(), 48_000);
  assert_eq!(mutated.hop_samples(), 16_000);
  assert_eq!(mutated.tail_policy(), TailPolicy::Partial);
  assert_eq!(mutated.max_windows(), 7);
  assert_ne!(mutated, WindowPlan::new());
}

/// `with_geometry` is usable in a `const` context, so a service can pin its
/// plan at compile time.
#[test]
fn the_plan_is_const_constructible() {
  const PINNED: WindowPlan = WindowPlan::new()
    .with_geometry(48_000, 24_000)
    .with_tail_policy(TailPolicy::Drop)
    .with_max_windows(64);
  assert_eq!(PINNED.window_samples(), 48_000);
  assert_eq!(PINNED.hop_samples(), 24_000);
  assert_eq!(PINNED.tail_policy(), TailPolicy::Drop);
  assert_eq!(PINNED.max_windows(), 64);
}

#[test]
#[should_panic(expected = "window_samples must be in")]
fn a_window_the_graph_cannot_score_panics() {
  let _ = WindowPlan::new().with_geometry(MIN_SAMPLES as u32 - 1, 1);
}

#[test]
#[should_panic(expected = "window_samples must be in")]
fn a_window_past_the_graph_ceiling_panics() {
  let _ = WindowPlan::new().with_geometry(MAX_SAMPLES as u32 + 1, 1);
}

#[test]
#[should_panic(expected = "hop_samples must be")]
fn a_zero_hop_panics() {
  let _ = WindowPlan::new().with_geometry(160_000, 0);
}

#[test]
#[should_panic(expected = "hop_samples must be")]
fn a_hop_past_the_window_panics() {
  let _ = WindowPlan::new().with_geometry(160_000, 160_001);
}

#[test]
#[should_panic(expected = "max_windows must be > 0")]
fn a_zero_cap_panics() {
  let _ = WindowPlan::new().with_max_windows(0);
}

// ── serde ───────────────────────────────────────────────────────────────────

/// The wire form round-trips, every field defaults when absent, and the
/// deserializer enforces the SAME invariants the setters assert — so a config
/// file cannot build a plan the builders reject.
#[cfg(feature = "serde")]
#[test]
fn the_serde_path_defaults_and_validates_like_the_setters() {
  let plan = WindowPlan::new()
    .with_geometry(48_000, 16_000)
    .with_tail_policy(TailPolicy::Partial)
    .with_max_windows(9);
  let json = serde_json::to_string(&plan).expect("serialize");
  assert!(json.contains("\"partial\""), "snake_case spelling: {json}");
  assert_eq!(
    serde_json::from_str::<WindowPlan>(&json).expect("deserialize"),
    plan
  );

  assert_eq!(
    serde_json::from_str::<WindowPlan>("{}").expect("all fields default"),
    WindowPlan::new()
  );
  assert_eq!(
    serde_json::from_str::<WindowPlan>(r#"{"tail":"drop"}"#)
      .expect("partial")
      .tail_policy(),
    TailPolicy::Drop
  );

  for rejected in [
    r#"{"hop_samples":0}"#,
    r#"{"hop_samples":160001}"#,
    r#"{"window_samples":1439}"#,
    r#"{"window_samples":480160}"#,
    r#"{"window_samples":48000,"hop_samples":48001}"#,
    r#"{"max_windows":0}"#,
    r#"{"tail":"pad"}"#,
  ] {
    assert!(
      serde_json::from_str::<WindowPlan>(rejected).is_err(),
      "{rejected} must not deserialize"
    );
  }
}

/// The [`TailPolicy`] document, pinned byte-exactly in BOTH directions, plus
/// the whole-plan document that carries it.
///
/// **Deliberately NOT the adjacently tagged `{"kind": …}` form** that
/// `audio::ced` and `embeddings::clap` took on alongside windit 0.4. Every
/// variant here is unit-shaped (see the type's "All three variants are
/// unit-shaped on purpose"), so there is no payload for a `value` to carry and
/// the externally tagged snake_case STRING is already the whole document. The
/// two shapes are pinned side by side across the doors so the difference is a
/// recorded decision rather than drift nobody measured.
#[cfg(feature = "serde")]
#[test]
fn tail_policy_wire_spellings_are_pinned() {
  // Wildcard-free: a new variant fails to compile until its spelling is pinned
  // here.
  for policy in [TailPolicy::SlideBack, TailPolicy::Partial, TailPolicy::Drop] {
    let doc = match policy {
      TailPolicy::SlideBack => r#""slide_back""#,
      TailPolicy::Partial => r#""partial""#,
      TailPolicy::Drop => r#""drop""#,
    };
    assert_eq!(serde_json::to_string(&policy).unwrap(), doc);
    assert_eq!(serde_json::from_str::<TailPolicy>(doc).unwrap(), policy);
  }
  assert_eq!(
    serde_json::to_string(&WindowPlan::new()).unwrap(),
    r#"{"window_samples":160000,"hop_samples":160000,"tail":"slide_back","max_windows":100000}"#
  );
  let doc = r#"{"window_samples":48000,"hop_samples":16000,"tail":"partial","max_windows":9}"#;
  let plan = WindowPlan::new()
    .with_geometry(48_000, 16_000)
    .with_tail_policy(TailPolicy::Partial)
    .with_max_windows(9);
  assert_eq!(serde_json::to_string(&plan).unwrap(), doc);
  assert_eq!(serde_json::from_str::<WindowPlan>(doc).unwrap(), plan);
}

/// A MISSPELLED key is REFUSED, not silently discarded.
///
/// Every field defaults, so without `deny_unknown_fields` `{"max_window":1}` —
/// the plural dropped — deserializes with the typo thrown away and
/// `max_windows` filled from [`DEFAULT_MAX_WINDOWS`]: an operator capping the
/// door at ONE window gets 100 000 instead, so a misspelled RESOURCE LIMIT
/// silently becomes up to 100 000 CoreML predictions. A misspelled `window`,
/// `hop` or `tail` key changes the scored geometry the same silent way. Each
/// rejection is paired with the spelling that must still parse, so this cannot
/// pass by the whole struct having stopped deserializing.
#[cfg(feature = "serde")]
#[test]
fn a_misspelled_key_is_refused_rather_than_silently_defaulted() {
  for (doc, key) in [
    (r#"{"max_window":1}"#, "max_window"),
    (r#"{"window":48000}"#, "window"),
    (r#"{"hop":16000}"#, "hop"),
    (r#"{"tail_policy":"drop"}"#, "tail_policy"),
    // Beside a well-spelled key, where a permissive impl is likeliest to let it
    // through.
    (r#"{"hop_samples":16000,"max_window":1}"#, "max_window"),
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
  let ok: WindowPlan =
    serde_json::from_str(r#"{"window_samples":48000,"hop_samples":16000,"max_windows":1}"#)
      .unwrap();
  assert_eq!(ok.window_samples(), 48_000);
  assert_eq!(ok.hop_samples(), 16_000);
  assert_eq!(ok.max_windows(), 1);
}

/// Every [`TailPolicy`] variant and the plans that carry it survive a
/// NON-self-describing format.
///
/// The measurement that `deny_unknown_fields` cannot reach a format with no
/// field names: postcard writes a struct as a bare sequence, so there is no key
/// for the repr to reject and no default to fill, and every plan still reads
/// back. Unlike `audio::ced` and `embeddings::clap`, this door's [`TailPolicy`]
/// is unit-only and externally tagged, so it needs no `is_human_readable`
/// split — a variant index is the whole encoding.
#[cfg(feature = "serde")]
#[test]
fn every_variant_round_trips_through_a_non_self_describing_format() {
  for policy in [TailPolicy::SlideBack, TailPolicy::Partial, TailPolicy::Drop] {
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
      .with_geometry(48_000, 16_000)
      .with_tail_policy(TailPolicy::Partial)
      .with_max_windows(9),
  ] {
    let bytes = postcard::to_allocvec(&plan).unwrap();
    assert_eq!(
      postcard::from_bytes::<WindowPlan>(&bytes).unwrap(),
      plan,
      "postcard round-trip lost {plan:?} (bytes {bytes:?})"
    );
  }
}
