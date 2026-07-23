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

// ── embed_preprocessed: PreprocessedImage validation ─────────────────────────

/// A well-formed padded bundle at budget `p` with `n_real` real patches: real
/// rows filled with `0.5`, pad rows zero, exact binary prefix mask.
fn bundle(p: usize, n_real: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let mut pixel_values = vec![0.0f32; p * PATCH_DIM];
  let mut position_embeddings = vec![0.0f32; p * EMBEDDING_DIM];
  let mut attention_mask = vec![0.0f32; p];
  pixel_values[..n_real * PATCH_DIM].fill(0.5);
  position_embeddings[..n_real * EMBEDDING_DIM].fill(0.5);
  attention_mask[..n_real].fill(1.0);
  (pixel_values, position_embeddings, attention_mask)
}

#[test]
fn preprocessed_image_accepts_well_formed_bundle() {
  let (px, pos, mask) = bundle(4, 3);
  let pre = PreprocessedImage::try_new(px, pos, mask, 4).expect("well-formed bundle");
  assert_eq!(pre.max_num_patches(), 4);
  assert_eq!(pre.pixel_values().len(), 4 * PATCH_DIM);
  assert_eq!(pre.position_embeddings().len(), 4 * EMBEDDING_DIM);
  assert_eq!(pre.attention_mask(), &[1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn preprocessed_image_accepts_full_budget_bundle() {
  // No pad rows — exercises the empty pad-scan edge.
  let (px, pos, mask) = bundle(4, 4);
  PreprocessedImage::try_new(px, pos, mask, 4).expect("full-budget bundle");
}

#[test]
fn preprocessed_image_accepts_negative_zero_mask_pad() {
  let (px, pos, mut mask) = bundle(4, 3);
  mask[3] = -0.0; // IEEE `-0.0 == 0.0`: documents the accepted edge.
  PreprocessedImage::try_new(px, pos, mask, 4).expect("negative-zero pad accepted");
}

#[test]
fn preprocessed_image_rejects_zero_budget() {
  match PreprocessedImage::try_new(vec![], vec![], vec![], 0) {
    Err(Error::PreprocessedPatchBudget { max_num_patches: 0 }) => {}
    other => panic!("expected PreprocessedPatchBudget, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_overflowing_budget() {
  // The budget guard runs before any multiplication, so this must not
  // panic/overflow in debug.
  match PreprocessedImage::try_new(vec![], vec![], vec![], usize::MAX) {
    Err(Error::PreprocessedPatchBudget { .. }) => {}
    other => panic!("expected PreprocessedPatchBudget, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_wrong_pixel_values_length() {
  let (mut px, pos, mask) = bundle(4, 3);
  px.pop();
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedLength {
      feature: "pixel_values",
      got,
      expected,
    }) => {
      assert_eq!(got, 4 * PATCH_DIM - 1);
      assert_eq!(expected, 4 * PATCH_DIM);
    }
    other => panic!("expected PreprocessedLength, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_wrong_position_embeddings_length() {
  let (px, mut pos, mask) = bundle(4, 3);
  pos.push(0.0);
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedLength {
      feature: "position_embeddings",
      got,
      expected,
    }) => {
      assert_eq!(got, 4 * EMBEDDING_DIM + 1);
      assert_eq!(expected, 4 * EMBEDDING_DIM);
    }
    other => panic!("expected PreprocessedLength, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_wrong_mask_length() {
  let (px, pos, _mask) = bundle(4, 3);
  let mask = vec![0.0f32; 5]; // length 5 at budget 4
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedLength {
      feature: "attention_mask",
      got: 5,
      expected: 4,
    }) => {}
    other => panic!("expected PreprocessedLength, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_non_finite_pixel_values() {
  let (mut px, pos, mask) = bundle(4, 3);
  px[10] = f32::NAN;
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedNonFinite {
      feature: "pixel_values",
      index: 10,
    }) => {}
    other => panic!("expected PreprocessedNonFinite, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_non_finite_position_embeddings() {
  let (px, mut pos, mask) = bundle(4, 3);
  pos[0] = f32::NEG_INFINITY;
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedNonFinite {
      feature: "position_embeddings",
      index: 0,
    }) => {}
    other => panic!("expected PreprocessedNonFinite, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_classifies_nan_mask_as_mask_value() {
  // A NaN mask entry is a MaskValue, not NonFinite: the mask never enters the
  // finiteness scan; its exact-binary domain check subsumes finiteness.
  let (px, pos, mut mask) = bundle(4, 3);
  mask[1] = f32::NAN;
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedMaskValue { index: 1, value }) => assert!(value.is_nan()),
    other => panic!("expected PreprocessedMaskValue, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_mask_value_outside_domain() {
  let (px, pos, mut mask) = bundle(4, 3);
  mask[1] = 0.5;
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedMaskValue { index: 1, value }) => assert_eq!(value, 0.5),
    other => panic!("expected PreprocessedMaskValue, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_mask_one_after_zero() {
  // Mask `1.0` after a `0.0` — the mask check precedes the pad-row check, so the
  // tensor content is irrelevant.
  let (px, pos, _mask) = bundle(4, 3);
  let mask = vec![1.0, 0.0, 1.0, 0.0];
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedMaskOrder { index: 2 }) => {}
    other => panic!("expected PreprocessedMaskOrder, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_all_pad_mask() {
  // All-zero tensors + all-zero mask at budget 4 (`bundle(4, 0)`).
  let (px, pos, mask) = bundle(4, 0);
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedMaskEmpty) => {}
    other => panic!("expected PreprocessedMaskEmpty, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_nonzero_pixel_pad_row() {
  let (mut px, pos, mask) = bundle(4, 3);
  px[3 * PATCH_DIM + 5] = 0.25; // a nonzero value inside the masked pad row
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedPadNonZero {
      feature: "pixel_values",
      index,
    }) => assert_eq!(index, 3 * PATCH_DIM + 5),
    other => panic!("expected PreprocessedPadNonZero, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_nonzero_position_embedding_pad_row() {
  let (px, mut pos, mask) = bundle(4, 3);
  pos[3 * EMBEDDING_DIM] = 1e-3; // first element of the masked pad row
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedPadNonZero {
      feature: "position_embeddings",
      index,
    }) => assert_eq!(index, 3 * EMBEDDING_DIM),
    other => panic!("expected PreprocessedPadNonZero, got {other:?}"),
  }
}

#[test]
fn check_patch_budget_accepts_equal_and_rejects_mismatch() {
  check_patch_budget(512, 512).expect("equal budgets accepted");
  match check_patch_budget(256, 512) {
    Err(Error::PatchBudgetMismatch {
      input: 256,
      model: 512,
    }) => {}
    other => panic!("expected PatchBudgetMismatch, got {other:?}"),
  }
}

#[test]
fn internal_pipeline_output_passes_public_validation() {
  use super::preprocess::{POS_EMBED_ELEMS, preprocess_image};
  // The internal NaFlex pipeline's outputs must satisfy the public validator —
  // the exact contract `embed`'s trusted `from_pipeline` path relies on.
  let v = preprocess_image(
    &[128u8; 8 * 8 * 3],
    8,
    8,
    &vec![0.0f32; POS_EMBED_ELEMS],
    512,
  )
  .expect("preprocess");
  let real = v.grid.0 * v.grid.1;
  let ones = v.attention_mask.iter().filter(|&&m| m == 1.0).count();
  assert_eq!(ones, real, "mask real-count equals the resolved grid");
  PreprocessedImage::try_new(v.pixel_values, v.position_embeddings, v.attention_mask, 512)
    .expect("pipeline output passes public validation");
}

#[test]
fn preprocessed_image_debug_is_compact() {
  let (px, pos, mask) = bundle(4, 3);
  let pre = PreprocessedImage::try_new(px, pos, mask, 4).expect("well-formed");
  let debug = format!("{pre:?}");
  assert!(debug.contains("max_num_patches"), "{debug}");
  assert!(debug.contains("num_real_patches: 3"), "{debug}");
  // Tensors are elided (`finish_non_exhaustive` renders `..`).
  assert!(!debug.contains("pixel_values"), "{debug}");
}
