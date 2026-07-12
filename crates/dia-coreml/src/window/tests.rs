use super::*;

// ---------------------------------------------------------------------
// SlidingWindow: accessors, builders, dia visibility-decision conversion
// ---------------------------------------------------------------------

#[test]
fn sliding_window_accessors() {
  let sw = SlidingWindow::new(1.0, 2.0, 0.5);
  assert_eq!(sw.start(), 1.0);
  assert_eq!(sw.duration(), 2.0);
  assert_eq!(sw.step(), 0.5);
}

#[test]
fn sliding_window_builders_replace_fields() {
  let sw = SlidingWindow::new(0.0, 0.0, 0.0)
    .with_start(1.0)
    .with_duration(2.0)
    .with_step(3.0);
  assert_eq!(sw, SlidingWindow::new(1.0, 2.0, 3.0));
}

#[test]
fn sliding_window_is_copy() {
  let sw = SlidingWindow::new(1.0, 2.0, 3.0);
  let copy = sw;
  // Both usable after the "move" — proves Copy, not just Clone.
  assert_eq!(sw, copy);
}

#[cfg(feature = "dia")]
#[test]
fn sliding_window_round_trips_into_dia_and_back() {
  let ours = SlidingWindow::new(0.25, 4.0, 1.5);
  let theirs: dia::reconstruct::SlidingWindow = ours.into();
  assert_eq!(theirs.start(), 0.25);
  assert_eq!(theirs.duration(), 4.0);
  assert_eq!(theirs.step(), 1.5);
  let back: SlidingWindow = theirs.into();
  assert_eq!(back, ours);
}

// ---------------------------------------------------------------------
// WindowOptions: defaults, builders, validation, serde
// ---------------------------------------------------------------------

#[test]
fn options_new_matches_dia_defaults() {
  let o = WindowOptions::new();
  assert_eq!(o.step_samples(), 16_000);
  assert_eq!(o.onset(), 0.5);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(WindowOptions::default(), WindowOptions::new());
}

#[test]
fn options_with_step_samples_overrides() {
  let o = WindowOptions::new().with_step_samples(40_000);
  assert_eq!(o.step_samples(), 40_000);
}

#[test]
fn options_set_step_samples_in_place() {
  let mut o = WindowOptions::new();
  o.set_step_samples(80_000);
  assert_eq!(o.step_samples(), 80_000);
}

#[test]
fn options_with_onset_overrides() {
  let o = WindowOptions::new().with_onset(0.7);
  assert_eq!(o.onset(), 0.7);
}

#[test]
fn options_set_onset_in_place() {
  let mut o = WindowOptions::new();
  o.set_onset(0.3);
  assert_eq!(o.onset(), 0.3);
}

#[test]
#[should_panic(expected = "step_samples must be > 0")]
fn options_with_step_samples_zero_panics() {
  let _ = WindowOptions::new().with_step_samples(0);
}

#[test]
#[should_panic(expected = "step_samples must be > 0")]
fn options_set_step_samples_zero_panics() {
  let mut o = WindowOptions::new();
  o.set_step_samples(0);
}

#[test]
#[should_panic(expected = "step_samples must be <= SEG_CHUNK_SAMPLES")]
fn options_with_step_samples_above_chunk_panics() {
  let _ = WindowOptions::new().with_step_samples(SEG_CHUNK_SAMPLES as u32 + 1);
}

#[test]
fn options_with_step_samples_equal_to_chunk_ok() {
  let o = WindowOptions::new().with_step_samples(SEG_CHUNK_SAMPLES as u32);
  assert_eq!(o.step_samples(), SEG_CHUNK_SAMPLES as u32);
}

#[test]
#[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
fn options_with_onset_zero_panics() {
  let _ = WindowOptions::new().with_onset(0.0);
}

#[test]
#[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
fn options_with_onset_negative_panics() {
  let _ = WindowOptions::new().with_onset(-0.1);
}

#[test]
#[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
fn options_with_onset_above_one_panics() {
  let _ = WindowOptions::new().with_onset(1.01);
}

#[test]
#[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
fn options_with_onset_nan_panics() {
  let _ = WindowOptions::new().with_onset(f32::NAN);
}

#[test]
#[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
fn options_with_onset_infinity_panics() {
  let _ = WindowOptions::new().with_onset(f32::INFINITY);
}

#[test]
#[should_panic(expected = "onset must be finite in (0.0, 1.0]")]
fn options_with_onset_neg_infinity_panics() {
  let _ = WindowOptions::new().with_onset(f32::NEG_INFINITY);
}

#[test]
fn options_with_onset_one_ok() {
  let o = WindowOptions::new().with_onset(1.0);
  assert_eq!(o.onset(), 1.0);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_fields_default() {
  let o: WindowOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(o, WindowOptions::new());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips_explicit_values() {
  let o: WindowOptions = serde_json::from_str(r#"{"step_samples":40000,"onset":0.7}"#).unwrap();
  assert_eq!(o.step_samples(), 40_000);
  assert_eq!(o.onset(), 0.7);
  let json = serde_json::to_string(&o).unwrap();
  assert!(json.contains("40000"), "round-tripped json: {json}");
  assert!(json.contains("0.7"), "round-tripped json: {json}");
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_partial_fields_default_the_rest() {
  let o: WindowOptions = serde_json::from_str(r#"{"onset":0.3}"#).unwrap();
  assert_eq!(o.step_samples(), 16_000);
  assert_eq!(o.onset(), 0.3);
}

// ---------------------------------------------------------------------
// chunk_starts: hand-computed geometry edge cases (brief Step 1)
// ---------------------------------------------------------------------

#[test]
fn chunk_starts_zero_samples_yields_single_zero_padded_chunk() {
  // See the module doc's "total_samples == 0" note: the formula's own
  // literal answer is one fully-zero-padded chunk. dia's own pipeline
  // never reaches this arithmetic for empty audio (it rejects earlier,
  // `owned.rs:369-371`) — this is this crate's own, documented, total
  // answer for the input domain a Result-free function must cover.
  let starts = chunk_starts(0, &WindowOptions::new());
  assert_eq!(starts, vec![0]);
}

#[test]
fn chunk_starts_shorter_than_one_chunk_yields_single_start() {
  let starts = chunk_starts(50_000, &WindowOptions::new());
  assert_eq!(starts, vec![0]);
}

#[test]
fn chunk_starts_exactly_one_chunk_yields_single_start() {
  let starts = chunk_starts(SEG_CHUNK_SAMPLES, &WindowOptions::new());
  assert_eq!(starts, vec![0]);
}

#[test]
fn chunk_starts_chunk_plus_one_sample_yields_two_chunks() {
  let starts = chunk_starts(SEG_CHUNK_SAMPLES + 1, &WindowOptions::new());
  assert_eq!(starts, vec![0, 16_000]);
}

#[test]
fn chunk_starts_chunk_plus_step_yields_two_chunks_exact_fit() {
  let starts = chunk_starts(SEG_CHUNK_SAMPLES + 16_000, &WindowOptions::new());
  assert_eq!(starts, vec![0, 16_000]);
}

#[test]
fn chunk_starts_chunk_plus_step_plus_one_yields_three_chunks() {
  let starts = chunk_starts(SEG_CHUNK_SAMPLES + 16_000 + 1, &WindowOptions::new());
  assert_eq!(starts, vec![0, 16_000, 32_000]);
}

#[test]
fn chunk_starts_multi_chunk_regular_grid() {
  let starts = chunk_starts(SEG_CHUNK_SAMPLES + 3 * 16_000, &WindowOptions::new());
  assert_eq!(starts, vec![0, 16_000, 32_000, 48_000]);
}

#[test]
fn chunk_starts_custom_step() {
  let options = WindowOptions::new().with_step_samples(40_000);
  let starts = chunk_starts(200_000, &options);
  assert_eq!(starts, vec![0, 40_000]);
}

#[test]
fn chunk_starts_step_equal_to_chunk_no_overlap() {
  let options = WindowOptions::new().with_step_samples(SEG_CHUNK_SAMPLES as u32);
  let starts = chunk_starts(2 * SEG_CHUNK_SAMPLES, &options);
  assert_eq!(starts, vec![0, SEG_CHUNK_SAMPLES]);
}

#[test]
#[should_panic(expected = "step_samples must be > 0")]
fn chunk_starts_panics_on_zero_step_via_bypassed_options() {
  // WindowOptions's own builder rejects step_samples == 0 (see the
  // options panic tests above); construct the field directly here
  // (this test module is a child of `window`, so private fields are
  // visible) to simulate a serde-deserialized WindowOptions that
  // bypassed builder validation — the exact scenario chunk_starts' own
  // defense-in-depth assert exists for.
  let options = WindowOptions {
    step_samples: 0,
    onset: DEFAULT_ONSET,
  };
  let _ = chunk_starts(1_000_000, &options);
}

// ---------------------------------------------------------------------
// chunk_sliding_window / frame_sliding_window
// ---------------------------------------------------------------------

#[test]
fn chunk_sliding_window_matches_dia_defaults() {
  let sw = chunk_sliding_window(&WindowOptions::new());
  assert_eq!(sw.start(), 0.0);
  assert_eq!(sw.duration(), 10.0);
  assert_eq!(sw.step(), 1.0); // 16_000 / 16_000
}

#[test]
fn chunk_sliding_window_reflects_custom_step() {
  let options = WindowOptions::new().with_step_samples(40_000);
  let sw = chunk_sliding_window(&options);
  assert_eq!(sw.step(), 2.5); // 40_000 / 16_000
}

#[test]
fn frame_sliding_window_matches_dia_constants() {
  let sw = frame_sliding_window();
  assert_eq!(sw.start(), 0.0);
  assert_eq!(sw.duration(), 0.0619375);
  assert_eq!(sw.step(), 0.016875);
}

// ---------------------------------------------------------------------
// count_from_segmentations: hand-computed 3-chunk overlap (brief Step 1)
// plus the dia code-oracle cross-check (brief Step 1, "THE ORACLE IS
// CODE")
// ---------------------------------------------------------------------

/// Synthetic 3-chunk overlap scenario shared between the hand-computed
/// hermetic test and the `dia`-gated code-oracle test: `num_chunks=3`,
/// `num_frames_per_chunk=4`, `num_speakers=2`, `chunks_sw = (0.0, 4.0,
/// 2.0)`, `frames_sw = (0.0, 1.0, 1.0)`, `onset=0.5`.
///
/// Hand derivation (also independently re-derived by dia's own
/// `try_count_pyannote` in the `dia`-gated oracle test below):
///
/// `chunk_count[c][f]` (active-speaker count per (chunk, frame), `v >=
/// 0.5`):
/// - c=0: `[1, 2, 1, 0]` (f=2 has `s0 == 0.5`, exactly at onset —
///   exercises the inclusive `>=`)
/// - c=1: `[2, 1, 1, 2]`
/// - c=2: `[1, 2, 1, 0]`
///
/// `start_frame(c) = round_ties_even(c * 2.0 / 1.0)`: `0, 2, 4` (exact,
/// no rounding needed). `num_output_frames = round_ties_even((4.0 + 2 *
/// 2.0) / 1.0) + 1 = 9`.
///
/// Per-output-frame `(aggregated, overlapping_count)`:
/// `t=0: (1,1)`, `t=1: (2,1)`, `t=2: (1+2=3,2)`, `t=3: (0+1=1,2)`,
/// `t=4: (1+1=2,2)`, `t=5: (2+2=4,2)`, `t=6: (1,1)`, `t=7: (0,1)`,
/// `t=8: (0,0)` (no chunk covers frame 8 — c=2's frames only reach
/// `start_frame(2) + 3 = 7`).
///
/// `count[t] = round_ties_even(aggregated[t] / overlapping_count[t])`,
/// `0` where uncovered: `[1, 2, 2, 0, 1, 2, 1, 0, 0]`. `t=2` (`3/2 =
/// 1.5`) and `t=3` (`1/2 = 0.5`) both exercise round_ties_even's
/// banker's-rounding tie rule (1.5 -> 2, the nearest EVEN integer; 0.5
/// -> 0, likewise nearest even) — not "round half up", which would give
/// `[1, 2, 2, 1, 1, 2, 1, 0, 0]` instead (see the mutation-testing
/// evidence in the task report for a rounding-rule regression this
/// exact case catches).
fn three_chunk_overlap_segmentations() -> Vec<f64> {
  #[rustfmt::skip]
  let segs = vec![
    1.0, 0.0,  1.0, 1.0,  0.5, 0.0,  0.0, 0.0, // c=0: f0..f3
    1.0, 1.0,  0.4, 0.6,  1.0, 0.0,  1.0, 1.0, // c=1: f0..f3
    0.0, 1.0,  1.0, 1.0,  1.0, 0.0,  0.0, 0.0, // c=2: f0..f3
  ];
  segs
}

#[test]
fn count_from_segmentations_single_chunk_no_overlap() {
  // 1 chunk, 3 frames, 2 speakers — isolates the basic accumulate/divide
  // path (no overlap) before the multi-chunk scenario below.
  #[rustfmt::skip]
  let segmentations = vec![
    1.0, 0.0, // f0: 1 active
    1.0, 1.0, // f1: 2 active
    0.0, 0.0, // f2: 0 active
  ];
  let chunks_sw = SlidingWindow::new(0.0, 3.0, 1.0);
  let frames_sw = SlidingWindow::new(0.0, 1.0, 1.0);
  let got = count_from_segmentations(&segmentations, 1, 3, 2, 0.5, chunks_sw, frames_sw);
  // num_output_frames = round(3.0 / 1.0) + 1 = 4; output frame 3 is
  // never covered by the single chunk's 3 frames (0..=2).
  assert_eq!(got, vec![1, 2, 0, 0]);
}

#[test]
fn count_from_segmentations_hand_computed_3_chunk_overlap() {
  let segmentations = three_chunk_overlap_segmentations();
  let chunks_sw = SlidingWindow::new(0.0, 4.0, 2.0);
  let frames_sw = SlidingWindow::new(0.0, 1.0, 1.0);

  let got = count_from_segmentations(&segmentations, 3, 4, 2, 0.5, chunks_sw, frames_sw);

  assert_eq!(got, vec![1, 2, 2, 0, 1, 2, 1, 0, 0]);
}

#[cfg(feature = "dia")]
#[test]
fn count_from_segmentations_matches_dia_oracle_3_chunk_overlap() {
  // THE ORACLE IS CODE: dia's own `try_count_pyannote` is public
  // (`diarization::aggregate::try_count_pyannote`) and is called here
  // directly on the SAME synthetic segmentations this crate's hermetic
  // test hand-derives an expected vector for — an independent,
  // compiled-code cross-check of that hand math, not just a repeat of
  // it.
  let segmentations = three_chunk_overlap_segmentations();

  let ours = count_from_segmentations(
    &segmentations,
    3,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 4.0, 2.0),
    SlidingWindow::new(0.0, 1.0, 1.0),
  );

  let golden = dia::aggregate::try_count_pyannote(
    &segmentations,
    3,
    4,
    2,
    0.5_f64,
    dia::reconstruct::SlidingWindow::new(0.0, 4.0, 2.0),
    dia::reconstruct::SlidingWindow::new(0.0, 1.0, 1.0),
    &dia::spill::SpillOptions::default(),
  )
  .expect("dia try_count_pyannote on synthetic 3-chunk overlap");

  assert_eq!(ours.as_slice(), golden.count_slice());
  // Cross-check against the hand-derived vector too, so a divergence
  // between the hand math and dia's own oracle would fail HERE, not
  // just in the separate hermetic test.
  assert_eq!(ours, vec![1, 2, 2, 0, 1, 2, 1, 0, 0]);
}

// ---------------------------------------------------------------------
// count_from_segmentations: precondition panics ("Who validates dims")
// ---------------------------------------------------------------------

#[test]
#[should_panic(expected = "num_chunks must be at least 1")]
fn count_from_segmentations_panics_on_zero_num_chunks() {
  let _ = count_from_segmentations(
    &[],
    0,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 10.0, 1.0),
    frame_sliding_window(),
  );
}

#[test]
#[should_panic(expected = "num_frames_per_chunk must be at least 1")]
fn count_from_segmentations_panics_on_zero_num_frames_per_chunk() {
  let _ = count_from_segmentations(
    &[],
    3,
    0,
    2,
    0.5,
    SlidingWindow::new(0.0, 10.0, 1.0),
    frame_sliding_window(),
  );
}

#[test]
#[should_panic(expected = "num_speakers must be at least 1")]
fn count_from_segmentations_panics_on_zero_num_speakers() {
  let _ = count_from_segmentations(
    &[],
    3,
    4,
    0,
    0.5,
    SlidingWindow::new(0.0, 10.0, 1.0),
    frame_sliding_window(),
  );
}

#[test]
#[should_panic(expected = "chunks_sw.duration() must be a positive finite scalar")]
fn count_from_segmentations_panics_on_zero_chunk_duration() {
  let segs = vec![0.0; 3 * 4 * 2];
  let _ = count_from_segmentations(
    &segs,
    3,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 0.0, 1.0),
    frame_sliding_window(),
  );
}

#[test]
#[should_panic(expected = "chunks_sw.step() must be a positive finite scalar")]
fn count_from_segmentations_panics_on_negative_chunk_step() {
  let segs = vec![0.0; 3 * 4 * 2];
  let _ = count_from_segmentations(
    &segs,
    3,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 10.0, -1.0),
    frame_sliding_window(),
  );
}

#[test]
#[should_panic(expected = "frames_sw.duration() must be a positive finite scalar")]
fn count_from_segmentations_panics_on_non_finite_frame_duration() {
  let segs = vec![0.0; 3 * 4 * 2];
  let _ = count_from_segmentations(
    &segs,
    3,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 10.0, 1.0),
    SlidingWindow::new(0.0, f64::NAN, 0.016875),
  );
}

#[test]
#[should_panic(expected = "frames_sw.step() must be a positive finite scalar")]
fn count_from_segmentations_panics_on_non_finite_frame_step() {
  let segs = vec![0.0; 3 * 4 * 2];
  let _ = count_from_segmentations(
    &segs,
    3,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 10.0, 1.0),
    SlidingWindow::new(0.0, 0.0619375, f64::INFINITY),
  );
}

#[test]
#[should_panic(expected = "onset must be finite")]
fn count_from_segmentations_panics_on_non_finite_onset() {
  let segs = vec![0.0; 3 * 4 * 2];
  let _ = count_from_segmentations(
    &segs,
    3,
    4,
    2,
    f32::NAN,
    SlidingWindow::new(0.0, 10.0, 1.0),
    frame_sliding_window(),
  );
}

#[test]
#[should_panic(
  expected = "segmentations.len() must equal num_chunks * num_frames_per_chunk * num_speakers"
)]
fn count_from_segmentations_panics_on_length_mismatch() {
  let segs = vec![0.0; 23]; // one short of 3*4*2 = 24
  let _ = count_from_segmentations(
    &segs,
    3,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 10.0, 1.0),
    frame_sliding_window(),
  );
}

#[test]
#[should_panic(expected = "segmentations must not contain NaN/infinite values")]
fn count_from_segmentations_panics_on_non_finite_segmentation_value() {
  let mut segs = vec![0.0; 24];
  segs[5] = f64::NAN;
  let _ = count_from_segmentations(
    &segs,
    3,
    4,
    2,
    0.5,
    SlidingWindow::new(0.0, 10.0, 1.0),
    frame_sliding_window(),
  );
}
