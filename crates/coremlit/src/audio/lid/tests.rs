use super::*;

// ── Frame-count arithmetic ──────────────────────────────────────────────────

/// `frames = 1 + n_samples / 160`, integer division, at the values the rest of
/// the module derives its bounds from.
#[test]
fn frame_count_follows_the_centre_padded_hop_arithmetic() {
  assert_eq!(frame_count(0), 1);
  assert_eq!(frame_count(1), 1);
  assert_eq!(frame_count(159), 1);
  assert_eq!(frame_count(160), 2);
  assert_eq!(frame_count(161), 2);
  assert_eq!(frame_count(16_000), 101); // 1 s
  assert_eq!(frame_count(48_000), 301); // 3 s
  assert_eq!(frame_count(480_000), 3_001); // exactly 30 s

  // Integer division, so a whole hop's worth of trailing samples is free.
  for extra in 0..HOP {
    assert_eq!(
      frame_count(480_000 + extra),
      3_001,
      "{extra} extra samples must not add a frame"
    );
  }
  assert_eq!(frame_count(480_000 + HOP), 3_002);
}

/// The sample bounds are exactly the frame bounds, translated. Stated as a
/// round trip so a change to either constant that is not matched by the other
/// reds here rather than in the runtime.
#[test]
fn sample_bounds_round_trip_through_the_frame_bounds() {
  assert_eq!(MIN_SAMPLES, 1_440);
  assert_eq!(MAX_SAMPLES, 480_159);
  assert_eq!(frame_count(MIN_SAMPLES), MIN_FRAMES);
  assert_eq!(frame_count(MAX_SAMPLES), MAX_FRAMES);

  // One sample outside either bound lands one frame outside the range.
  assert_eq!(frame_count(MIN_SAMPLES - 1), MIN_FRAMES - 1);
  assert_eq!(frame_count(MAX_SAMPLES + 1), MAX_FRAMES + 1);

  // And they really are 0.09 s / ~30.01 s at 16 kHz.
  let rate = f64::from(SAMPLE_RATE_HZ);
  assert!((MIN_SAMPLES as f64 / rate - 0.09).abs() < 1e-9);
  assert!((MAX_SAMPLES as f64 / rate - 30.0099).abs() < 1e-4);
}

// ── The range guard, at both boundaries ─────────────────────────────────────

/// The guard accepts exactly `MIN_SAMPLES..=MAX_SAMPLES` and rejects the two
/// samples immediately outside — the boundary pair on each end, so an
/// off-by-one in either direction reds.
#[test]
fn the_range_guard_accepts_and_rejects_at_both_boundaries() {
  assert_eq!(
    validate_frame_range(MIN_SAMPLES).expect("accepted"),
    MIN_FRAMES
  );
  assert_eq!(
    validate_frame_range(MIN_SAMPLES + 1).expect("accepted"),
    MIN_FRAMES
  );
  assert_eq!(
    validate_frame_range(MAX_SAMPLES).expect("accepted"),
    MAX_FRAMES
  );
  assert_eq!(
    validate_frame_range(MAX_SAMPLES - 1).expect("accepted"),
    MAX_FRAMES
  );

  for rejected in [0, 1, MIN_SAMPLES - 1] {
    let error = validate_frame_range(rejected).expect_err("must reject");
    let Error::FrameCountOutOfRange(detail) = error else {
      panic!("expected FrameCountOutOfRange for {rejected} samples, got {error:?}");
    };
    assert_eq!(detail.samples(), rejected);
    assert_eq!(detail.frames(), frame_count(rejected));
    assert!(detail.is_too_short(), "{rejected} samples is a short clip");
  }

  for rejected in [MAX_SAMPLES + 1, MAX_SAMPLES + HOP, 10_000_000] {
    let error = validate_frame_range(rejected).expect_err("must reject");
    let Error::FrameCountOutOfRange(detail) = error else {
      panic!("expected FrameCountOutOfRange for {rejected} samples, got {error:?}");
    };
    assert_eq!(detail.samples(), rejected);
    assert!(!detail.is_too_short(), "{rejected} samples is a long clip");
  }
}

/// Empty audio is a frame-count rejection, not a separate variant: zero samples
/// is one frame, which is below [`MIN_FRAMES`]. One guard, one error, and the
/// message still names the bounds a caller has to satisfy.
#[test]
fn empty_audio_is_rejected_as_a_short_clip() {
  let error = validate_frame_range(0).expect_err("empty audio must be rejected");
  assert!(matches!(&error, Error::FrameCountOutOfRange(d) if d.is_too_short()));
  let rendered = error.to_string();
  assert!(rendered.contains("0 samples"), "{rendered}");
  assert!(rendered.contains("1440"), "{rendered}");
}

// ── Non-finite input ────────────────────────────────────────────────────────

/// NaN and both infinities are rejected, and the reported index is the FIRST
/// offending sample.
#[test]
fn non_finite_samples_are_reported_by_first_index() {
  assert!(check_finite_samples(&[0.0, 1.0, -1.0]).is_ok());
  assert!(check_finite_samples(&[]).is_ok());

  for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    let mut samples = vec![0.5f32; 16];
    samples[7] = bad;
    samples[11] = f32::NAN;
    assert!(matches!(
      check_finite_samples(&samples),
      Err(Error::NonFiniteInput(7))
    ));
  }
}

// ── Options ─────────────────────────────────────────────────────────────────

/// `new`, `Default` and the builder all agree, and the setters round-trip.
#[test]
fn options_share_one_default() {
  assert_eq!(IdentifierOptions::new(), IdentifierOptions::default());
  assert_eq!(IdentifierOptions::new().compute(), DEFAULT_COMPUTE);
  assert_eq!(DEFAULT_COMPUTE, ComputeUnits::All);

  let options = IdentifierOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(options.compute(), ComputeUnits::CpuAndGpu);

  let mut mutated = IdentifierOptions::new();
  mutated.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(mutated.compute(), ComputeUnits::CpuOnly);
  assert_ne!(mutated, IdentifierOptions::new());
}

/// `with_compute` is usable in a `const` context — the same guarantee ced's
/// options carry, so a caller can build a placement table at compile time.
#[test]
fn options_are_const_constructible() {
  const PINNED: IdentifierOptions = IdentifierOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(PINNED.compute(), ComputeUnits::CpuAndGpu);
}

#[cfg(feature = "serde")]
#[test]
fn options_round_trip_through_serde_by_compute_unit_name() {
  let options = IdentifierOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  let json = serde_json::to_string(&options).expect("serialize");
  assert!(
    json.contains(ComputeUnits::CpuAndNeuralEngine.as_str()),
    "the bridge must write the unit's own name: {json}"
  );
  assert_eq!(
    serde_json::from_str::<IdentifierOptions>(&json).expect("deserialize"),
    options
  );

  // The field defaults when absent, and an unknown name is a typed failure
  // rather than a silent fallback.
  assert_eq!(
    serde_json::from_str::<IdentifierOptions>("{}").expect("default"),
    IdentifierOptions::new()
  );
  assert!(serde_json::from_str::<IdentifierOptions>(r#"{"compute":"quantum"}"#).is_err());
}

// ── Contract-mismatch rendering ─────────────────────────────────────────────

/// `describe` renders what a mismatched feature actually declares, including
/// the case where the feature carries no multi-array constraint at all.
#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(
    describe(&[1, 301, 60], Some(DataType::F32)),
    "[1, 301, 60] float32"
  );
  assert_eq!(describe(&[], None), "[] none");
}

/// The declared tensor names are the graph's, spelled once.
#[test]
fn tensor_names_are_pinned() {
  assert_eq!(names::MEL_FEATURES, "mel_features");
  assert_eq!(names::LOG_PROBABILITIES, "log_probabilities");
}

// ── The long path's own guard ───────────────────────────────────────────────

/// The long path keeps the SHORT-clip floor and drops the ceiling — the whole
/// point of it. Stated at both ends so a copy-paste of `validate_frame_range`'s
/// upper bound would red here.
#[test]
fn the_long_guard_keeps_the_floor_and_drops_the_ceiling() {
  assert!(validate_long_input(&vec![0.0; MIN_SAMPLES]).is_ok());
  for accepted in [MAX_SAMPLES, MAX_SAMPLES + 1, 10 * MAX_SAMPLES] {
    assert!(
      validate_long_input(&vec![0.0; accepted]).is_ok(),
      "{accepted} samples must be accepted by the long path"
    );
  }

  for rejected in [0, 1, MIN_SAMPLES - 1] {
    let error = validate_long_input(&vec![0.0; rejected]).expect_err("must reject");
    let Error::FrameCountOutOfRange(detail) = error else {
      panic!("expected FrameCountOutOfRange for {rejected} samples, got {error:?}");
    };
    assert_eq!(detail.samples(), rejected);
    assert!(detail.is_too_short());
  }
}

/// The long guard scans the WHOLE clip before any window is sliced, so the
/// reported index is clip-absolute — a NaN deep inside a later window is not
/// renumbered relative to that window's start.
#[test]
fn the_long_guard_reports_a_clip_absolute_index() {
  let mut samples = vec![0.5f32; 3 * MAX_SAMPLES];
  let deep = 2 * MAX_SAMPLES + 12_345;
  samples[deep] = f32::NAN;
  assert!(matches!(
    validate_long_input(&samples),
    Err(Error::NonFiniteInput(index)) if index == deep
  ));
}

/// `prewarm` warms exactly the default plan's window, because it reads its
/// length from that constant. The tone it builds is unchanged from before the
/// long path existed (10 s at 16 kHz was already 160 000 samples), so this is a
/// single-source-of-truth statement, not a behaviour change.
#[test]
fn prewarm_covers_the_default_plans_window() {
  assert_eq!(
    DEFAULT_WINDOW_SAMPLES as usize,
    10 * SAMPLE_RATE_HZ as usize
  );
  assert_eq!(frame_count(DEFAULT_WINDOW_SAMPLES as usize), 1_001);
  assert_eq!(
    WindowPlan::new().window_samples(),
    DEFAULT_WINDOW_SAMPLES,
    "prewarm's clip length is the default plan's window"
  );
}
