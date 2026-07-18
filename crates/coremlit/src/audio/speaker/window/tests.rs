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

#[cfg(feature = "speaker")]
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

// L1: `Deserialize` must route through the SAME invariants the checked setters
// enforce (`serde(try_from = WindowOptionsRepr)`), not bypass them. The derived
// field-Deserialize let `{"step_samples":200000}` construct geometry that
// silently drops the samples in `[SEG_CHUNK_SAMPLES, step)` of every chunk.

#[cfg(feature = "serde")]
#[test]
fn options_serde_rejects_step_samples_exceeding_window() {
  // The brief's exact witness: 200000 > SEG_CHUNK_SAMPLES (160000). On 320000
  // samples this produced windows [0,160000)+[200000,360000), omitting
  // [160000,200000). Must now fail to deserialize.
  let r: Result<WindowOptions, _> = serde_json::from_str(r#"{"step_samples":200000,"onset":0.5}"#);
  assert!(
    r.is_err(),
    "step_samples 200000 > SEG_CHUNK_SAMPLES must fail to deserialize, got {r:?}"
  );
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_rejects_zero_step_samples() {
  let r: Result<WindowOptions, _> = serde_json::from_str(r#"{"step_samples":0}"#);
  assert!(
    r.is_err(),
    "step_samples 0 must fail to deserialize, got {r:?}"
  );
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_rejects_out_of_range_onset() {
  for bad in [r#"{"onset":0.0}"#, r#"{"onset":1.5}"#, r#"{"onset":-0.1}"#] {
    let r: Result<WindowOptions, _> = serde_json::from_str(bad);
    assert!(r.is_err(), "{bad} must fail to deserialize, got {r:?}");
  }
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_accepts_valid_boundary_values() {
  // The deserialize path must be no stricter than the builders: `step ==
  // SEG_CHUNK_SAMPLES` and `onset == 1.0` are both valid at the setter
  // (`options_with_step_samples_equal_to_chunk_ok`, `options_with_onset_one_ok`).
  let o: WindowOptions = serde_json::from_str(r#"{"step_samples":160000,"onset":1.0}"#).unwrap();
  assert_eq!(o.step_samples(), 160_000);
  assert_eq!(o.onset(), 1.0);
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
/// `[1, 2, 2, 1, 1, 2, 1, 0, 0]` instead.
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

#[cfg(feature = "speaker")]
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
// count_from_segmentations: dia oracle cross-check at this crate's
// DEFAULT production geometry (~10x nominal simultaneous overlap —
// CHUNK_DURATION_S / step_s = 10.0 / 1.0 = 10 — vs. the 2x-overlap
// fixture above)
// ---------------------------------------------------------------------

// Only consumed by the `dia`-gated oracle test below (unlike
// `three_chunk_overlap_segmentations` above, there is no non-`dia`
// hermetic test at this data volume — see that test's own doc for why),
// so this whole fixture is `dia`-gated too; otherwise it's unused
// (dead code) under the default/non-`dia` feature set.
#[cfg(feature = "speaker")]
const DEFAULT_10X_NUM_CHUNKS: usize = 15;
/// dia's real `FRAMES_PER_WINDOW` (`diarization/src/segment/options.rs:
/// 21`) — the actual pyannote segmentation model's per-chunk frame
/// count (also this crate's own introspected model contract, the
/// `crate::audio::speaker::segment` module doc's `[1, 589, 7]`). A smaller synthetic
/// frame count would NOT exercise genuine 10x overlap: consecutive
/// chunks' `start_frame(c) = round_ties_even(c * chunk_step /
/// frame_step)` values are ~59.26 frames apart (`1.0 / 0.016875`), so a
/// chunk's own frame span must be close to this real value for ~10
/// consecutive chunks' windows to overlap a common output frame at all.
#[cfg(feature = "speaker")]
const DEFAULT_10X_NUM_FRAMES_PER_CHUNK: usize = 589;
#[cfg(feature = "speaker")]
const DEFAULT_10X_NUM_SPEAKERS: usize = 2;

/// Synthetic segmentations at this crate's DEFAULT production geometry
/// (`chunk_sliding_window(&WindowOptions::new())` = `(0.0, 10.0, 1.0)`,
/// `frame_sliding_window()` = `(0.0, 0.0619375, 0.016875)`).
///
/// Base values follow `((c * 37 + f * 17 + s * 11) % 10) as f64 / 10.0`
/// — deterministic, spans `{0.0, 0.1, ..., 0.9}`, crossing `onset =
/// 0.5` in both directions broadly across the whole tensor (including
/// exact `0.5` hits from the formula itself, at cells where `(c * 37 +
/// f * 17 + s * 11) % 10 == 5`).
///
/// Ten cells are then deliberately overridden to force output frame `t
/// = 711` onto an EXACT `round_ties_even` rounding TIE at full 10x
/// overlap — a structurally different tie than the 2x-overlap fixture
/// above (that one ties at divisor 2; a divisor-10 tie is only possible
/// where the covering-chunk count is EVEN, and only exists at all in
/// the interior of a long-enough recording). Exact arithmetic
/// (`start_frame(c) = round_ties_even(c * 1.0 / 0.016875)` for `c` in
/// `0..15`): output frame `t = 711` is covered by EXACTLY chunks `[3, 4,
/// 5, 6, 7, 8, 9, 10, 11, 12]` (10 consecutive chunks — a covering
/// range is always contiguous, since `start_frame` is monotonic in
/// `c`), at local frame offsets `t - start_frame(c)`:
///
/// | c        | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  | 11 | 12 |
/// |----------|-----|-----|-----|-----|-----|-----|-----|-----|----|----|
/// | local_f  | 533 | 474 | 415 | 355 | 296 | 237 | 178 | 118 | 59 | 0  |
///
/// Chunks `3..=7` get speaker 0 forced ACTIVE (`1.0`) at their local
/// frame (speaker 1 forced `0.0`) — `chunk_count = 1` each. Chunks
/// `8..=12` get BOTH speakers forced `0.0` — `chunk_count = 0` each.
/// `aggregated[711] = 5 * 1 + 5 * 0 = 5`, `overlapping_count[711] =
/// 10`, ratio `0.5` exactly — `round_ties_even` rounds to the nearest
/// EVEN integer, `0`.
#[cfg(feature = "speaker")]
fn default_geometry_10x_overlap_segmentations() -> Vec<f64> {
  let mut segs =
    vec![
      0.0_f64;
      DEFAULT_10X_NUM_CHUNKS * DEFAULT_10X_NUM_FRAMES_PER_CHUNK * DEFAULT_10X_NUM_SPEAKERS
    ];
  for c in 0..DEFAULT_10X_NUM_CHUNKS {
    for f in 0..DEFAULT_10X_NUM_FRAMES_PER_CHUNK {
      for s in 0..DEFAULT_10X_NUM_SPEAKERS {
        let idx = (c * DEFAULT_10X_NUM_FRAMES_PER_CHUNK + f) * DEFAULT_10X_NUM_SPEAKERS + s;
        segs[idx] = ((c * 37 + f * 17 + s * 11) % 10) as f64 / 10.0;
      }
    }
  }

  let overrides: [(usize, usize, bool); 10] = [
    (3, 533, true),
    (4, 474, true),
    (5, 415, true),
    (6, 355, true),
    (7, 296, true),
    (8, 237, false),
    (9, 178, false),
    (10, 118, false),
    (11, 59, false),
    (12, 0, false),
  ];
  for (c, f, active) in overrides {
    let base = (c * DEFAULT_10X_NUM_FRAMES_PER_CHUNK + f) * DEFAULT_10X_NUM_SPEAKERS;
    segs[base] = if active { 1.0 } else { 0.0 }; // speaker 0
    segs[base + 1] = 0.0; // speaker 1: always inactive at these cells
  }

  segs
}

#[cfg(feature = "speaker")]
#[test]
fn count_from_segmentations_matches_dia_oracle_default_geometry_10x_overlap() {
  let segmentations = default_geometry_10x_overlap_segmentations();
  let chunks_sw = chunk_sliding_window(&WindowOptions::new());
  let frames_sw = frame_sliding_window();

  let ours = count_from_segmentations(
    &segmentations,
    DEFAULT_10X_NUM_CHUNKS,
    DEFAULT_10X_NUM_FRAMES_PER_CHUNK,
    DEFAULT_10X_NUM_SPEAKERS,
    0.5,
    chunks_sw,
    frames_sw,
  );

  let golden = dia::aggregate::try_count_pyannote(
    &segmentations,
    DEFAULT_10X_NUM_CHUNKS,
    DEFAULT_10X_NUM_FRAMES_PER_CHUNK,
    DEFAULT_10X_NUM_SPEAKERS,
    0.5_f64,
    chunks_sw.into(),
    frames_sw.into(),
    &dia::spill::SpillOptions::default(),
  )
  .expect("dia try_count_pyannote on default-geometry 10x-overlap synthetic fixture");

  assert_eq!(ours.as_slice(), golden.count_slice());
  // Pin the deliberately engineered round_ties_even tie directly too —
  // makes the construction's intent independently checkable, not just
  // implied by the oracle match.
  assert_eq!(ours[711], 0);
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

// ---------------------------------------------------------------------
// try_num_output_frames / count_from_segmentations: output-frame-count
// overflow guard (review finding: adversarial-but-finite SlidingWindow
// values — both individually positive and finite, so no OTHER
// precondition assert above rejects them — previously drove
// `num_output_frames`'s division/round/cast/+1 sequence to a debug
// "attempt to add with overflow" panic, or a silent release wrap to 0,
// instead of a typed/diagnosable failure. See `try_num_output_frames`'s
// own doc for dia's cited guard this ports.)
// ---------------------------------------------------------------------

#[test]
fn try_num_output_frames_accepts_valid_geometry() {
  // Same last_chunk_end/frame_step as the hand-computed 3-chunk-overlap
  // fixture above: chunk_duration=4.0, chunk_step=2.0, num_chunks=3 ->
  // last_chunk_end = 4.0 + 2 * 2.0 = 8.0; frame_step=1.0 -> 8.0 + 1 = 9,
  // matching that fixture's documented num_output_frames = 9.
  assert_eq!(try_num_output_frames(8.0, 1.0), Ok(9));
}

#[test]
fn try_num_output_frames_rejects_infinite_division() {
  // Reviewer's exact adversarial construction: duration=1e300,
  // frame_step=1e-300 -> the division overflows f64 directly to +inf
  // (1e300 / 1e-300 = 1e600, outside f64's finite range). round_ties_even
  // of +inf is still +inf, caught by the `!frames_f.is_finite()` arm.
  assert_eq!(
    try_num_output_frames(1e300, 1e-300),
    Err(WindowError::OutputFrameCountOverflow)
  );
}

#[test]
fn try_num_output_frames_rejects_finite_but_saturating_division() {
  // A DIFFERENT adversarial route than the +inf case above: the
  // division stays FINITE (1e20 is far below f64::MAX ~1.8e308) but
  // exceeds usize::MAX (~1.8447e19 on 64-bit), so `as usize` would
  // saturate. Exercises the `frames_f >= usize::MAX as f64` arm
  // specifically, not the `!frames_f.is_finite()` arm above — dia's own
  // guard distinguishes the two, so this crate's port must too.
  assert_eq!(
    try_num_output_frames(1e20, 1.0),
    Err(WindowError::OutputFrameCountOverflow)
  );
}

#[test]
#[should_panic(expected = "num_output_frames must fit in usize")]
fn count_from_segmentations_panics_on_output_frame_count_overflow() {
  // End-to-end: the same +inf adversarial geometry, reached through the
  // public function. Both `chunks_sw` and `frames_sw` are constructible
  // via the public, unchecked `SlidingWindow::new`; no OTHER guard in
  // this function rejects them (duration/step are each individually
  // positive and finite — only their COMBINATION overflows
  // `num_output_frames`).
  let segs = vec![0.0; 1]; // num_chunks=1, num_frames_per_chunk=1, num_speakers=1
  let chunks_sw = SlidingWindow::new(0.0, 1e300, 1.0);
  let frames_sw = SlidingWindow::new(0.0, 0.0619375, 1e-300);
  let _ = count_from_segmentations(&segs, 1, 1, 1, 0.5, chunks_sw, frames_sw);
}
