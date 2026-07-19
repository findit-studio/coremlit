use super::*;

#[test]
fn options_default_equals_new() {
  assert_eq!(AudioEncoderOptions::default(), AudioEncoderOptions::new());
  assert_eq!(AudioEncoderOptions::new().compute(), DEFAULT_AUDIO_COMPUTE);
  assert_eq!(DEFAULT_AUDIO_COMPUTE, ComputeUnits::All);
}

#[test]
fn options_with_and_set_compute() {
  let opts = AudioEncoderOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);

  let mut opts = AudioEncoderOptions::new();
  opts.set_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndGpu);
}

#[test]
fn first_non_finite_finds_offenders() {
  assert_eq!(first_non_finite(&[0.0, 1.0, 2.0]), None);
  assert_eq!(first_non_finite(&[0.0, f32::NAN, 2.0]), Some(1));
  assert_eq!(first_non_finite(&[f32::INFINITY]), Some(0));
  assert_eq!(first_non_finite(&[1.0, 2.0, f32::NEG_INFINITY]), Some(2));
  // Subnormals and signed zeros are finite.
  assert_eq!(
    first_non_finite(&[0.0, -0.0, f32::MIN_POSITIVE / 2.0]),
    None
  );
}

#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(
    describe(&[1, 1, 1001, 64], Some(DataType::F32)),
    "[1, 1, 1001, 64] float32"
  );
  assert_eq!(describe(&[1, 512], None), "[1, 512] none");
}

/// `embed_window` accepts `1..=TARGET_SAMPLES` and rejects an over-length clip
/// with [`Error::AudioTooLong`] (naming `embed_windows`) instead of silently
/// head-truncating it. Gated at the `check_window_len` seam so it needs no model.
///
/// Mutation tripwire: relaxing the bound (`>` → `>=`, or `TARGET_SAMPLES` →
/// `TARGET_SAMPLES + 1`) makes the over-length case pass, and dropping the guard
/// re-admits the silent-truncation defect.
#[test]
fn check_window_len_rejects_over_length_only() {
  // The exact window and anything shorter are accepted.
  assert!(check_window_len(TARGET_SAMPLES).is_ok());
  assert!(check_window_len(TARGET_SAMPLES - 1).is_ok());
  assert!(check_window_len(1).is_ok());
  // One sample past the window is rejected, and the error carries len + limit and
  // points the caller at the long-audio path.
  let err = check_window_len(TARGET_SAMPLES + 1).unwrap_err();
  let msg = err.to_string();
  assert!(
    matches!(err, Error::AudioTooLong { len, max } if len == TARGET_SAMPLES + 1 && max == TARGET_SAMPLES),
    "expected AudioTooLong{{ len: {}, max: {TARGET_SAMPLES} }}, got {err:?}",
    TARGET_SAMPLES + 1
  );
  assert!(
    msg.contains("embed_windows"),
    "AudioTooLong should name the long-audio path: {msg}"
  );
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = AudioEncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("cpu_and_gpu"), "serialized as as_str: {json}");
  let back: AudioEncoderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}
