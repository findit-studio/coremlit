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

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = AudioEncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("cpu_and_gpu"), "serialized as as_str: {json}");
  let back: AudioEncoderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}
