use super::*;

// ── A5: Rgb8Image geometry validation ────────────────────────────────────────

#[test]
fn rgb8_image_accepts_valid_geometry_and_exposes_dims() {
  let data = vec![7u8; 4 * 3 * 3]; // 4×3 RGB
  let img = Rgb8Image::new(&data, 4, 3).expect("valid");
  assert_eq!(img.width(), 4);
  assert_eq!(img.height(), 3);
  assert_eq!(img.data().len(), 4 * 3 * 3);
  // The borrowed view round-trips the exact bytes.
  assert_eq!(img.data(), data.as_slice());
}

#[test]
fn rgb8_image_rejects_zero_width() {
  let data: Vec<u8> = Vec::new();
  match Rgb8Image::new(&data, 0, 3) {
    Err(Error::ImageDimensions {
      width: 0,
      height: 3,
    }) => {}
    other => panic!("expected ImageDimensions, got {other:?}"),
  }
}

#[test]
fn rgb8_image_rejects_zero_height() {
  let data: Vec<u8> = Vec::new();
  match Rgb8Image::new(&data, 4, 0) {
    Err(Error::ImageDimensions {
      width: 4,
      height: 0,
    }) => {}
    other => panic!("expected ImageDimensions, got {other:?}"),
  }
}

#[test]
fn rgb8_image_rejects_length_mismatch() {
  let data = vec![0u8; 4 * 3 * 3 - 1]; // one byte short
  match Rgb8Image::new(&data, 4, 3) {
    Err(Error::ImageDataLength { got, expected }) => {
      assert_eq!(got, 4 * 3 * 3 - 1);
      assert_eq!(expected, 4 * 3 * 3);
    }
    other => panic!("expected ImageDataLength, got {other:?}"),
  }
}

#[test]
fn rgb8_image_rejects_size_overflow() {
  // width·height·3 overflows usize; data length is irrelevant to the overflow.
  let data = [0u8; 1];
  match Rgb8Image::new(&data, usize::MAX, 2) {
    Err(Error::ImageDimensions { .. }) => {}
    other => panic!("expected ImageDimensions on overflow, got {other:?}"),
  }
}

// ── A4: options ──────────────────────────────────────────────────────────────

#[test]
fn options_default_equals_new_and_is_cpu_and_gpu() {
  assert_eq!(ImageEmbedderOptions::default(), ImageEmbedderOptions::new());
  assert_eq!(ImageEmbedderOptions::new().compute(), DEFAULT_IMAGE_COMPUTE);
  // D1: the floor-holding default is CpuAndGpu, NOT All.
  assert_eq!(DEFAULT_IMAGE_COMPUTE, ComputeUnits::CpuAndGpu);
}

#[test]
fn options_with_and_set_compute() {
  let opts = ImageEmbedderOptions::new().with_compute(ComputeUnits::All);
  assert_eq!(opts.compute(), ComputeUnits::All);
  let mut opts = ImageEmbedderOptions::new();
  opts.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(
    describe(&[1, 512, 768], Some(DataType::F32)),
    "[1, 512, 768] float32"
  );
  assert_eq!(describe(&[1, 512], None), "[1, 512] none");
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let opts = ImageEmbedderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  let json = serde_json::to_string(&opts).unwrap();
  assert!(json.contains("cpu_and_neural_engine"), "serialized: {json}");
  let back: ImageEmbedderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_defaults_missing_compute_to_the_module_default() {
  // A missing `compute` field defaults to DEFAULT_IMAGE_COMPUTE (serde default).
  let back: ImageEmbedderOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(back, ImageEmbedderOptions::new());
}
