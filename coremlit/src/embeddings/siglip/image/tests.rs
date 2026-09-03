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
    Err(Error::ImageDimensions(ref e)) if e.width() == 0 && e.height() == 3 => {}
    other => panic!("expected ImageDimensions, got {other:?}"),
  }
}

#[test]
fn rgb8_image_rejects_zero_height() {
  let data: Vec<u8> = Vec::new();
  match Rgb8Image::new(&data, 4, 0) {
    Err(Error::ImageDimensions(ref e)) if e.width() == 4 && e.height() == 0 => {}
    other => panic!("expected ImageDimensions, got {other:?}"),
  }
}

#[test]
fn rgb8_image_rejects_length_mismatch() {
  let data = vec![0u8; 4 * 3 * 3 - 1]; // one byte short
  match Rgb8Image::new(&data, 4, 3) {
    Err(Error::ImageDataLength(e)) => {
      assert_eq!(e.got(), 4 * 3 * 3 - 1);
      assert_eq!(e.expected(), 4 * 3 * 3);
    }
    other => panic!("expected ImageDataLength, got {other:?}"),
  }
}

#[test]
fn rgb8_image_rejects_size_overflow() {
  // width·height·3 overflows usize; data length is irrelevant to the overflow.
  let data = [0u8; 1];
  match Rgb8Image::new(&data, usize::MAX, 2) {
    Err(Error::ImageDimensions(_)) => {}
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
    Err(Error::PreprocessedPatchBudget(0)) => {}
    other => panic!("expected PreprocessedPatchBudget, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_overflowing_budget() {
  // The budget guard runs before any multiplication, so this must not
  // panic/overflow in debug.
  match PreprocessedImage::try_new(vec![], vec![], vec![], usize::MAX) {
    Err(Error::PreprocessedPatchBudget(_)) => {}
    other => panic!("expected PreprocessedPatchBudget, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_wrong_pixel_values_length() {
  let (mut px, pos, mask) = bundle(4, 3);
  px.pop();
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedLength(e)) if e.feature() == "pixel_values" => {
      assert_eq!(e.got(), 4 * PATCH_DIM - 1);
      assert_eq!(e.expected(), 4 * PATCH_DIM);
    }
    other => panic!("expected PreprocessedLength, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_wrong_position_embeddings_length() {
  let (px, mut pos, mask) = bundle(4, 3);
  pos.push(0.0);
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedLength(e)) if e.feature() == "position_embeddings" => {
      assert_eq!(e.got(), 4 * EMBEDDING_DIM + 1);
      assert_eq!(e.expected(), 4 * EMBEDDING_DIM);
    }
    other => panic!("expected PreprocessedLength, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_wrong_mask_length() {
  let (px, pos, _mask) = bundle(4, 3);
  let mask = vec![0.0f32; 5]; // length 5 at budget 4
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedLength(ref e))
      if e.feature() == "attention_mask" && e.got() == 5 && e.expected() == 4 => {}
    other => panic!("expected PreprocessedLength, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_non_finite_pixel_values() {
  let (mut px, pos, mask) = bundle(4, 3);
  px[10] = f32::NAN;
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedNonFinite(ref e))
      if e.feature() == "pixel_values" && e.index() == 10 => {}
    other => panic!("expected PreprocessedNonFinite, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_non_finite_position_embeddings() {
  let (px, mut pos, mask) = bundle(4, 3);
  pos[0] = f32::NEG_INFINITY;
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedNonFinite(ref e))
      if e.feature() == "position_embeddings" && e.index() == 0 => {}
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
    Err(Error::PreprocessedMaskValue(e)) if e.index() == 1 => assert!(e.value().is_nan()),
    other => panic!("expected PreprocessedMaskValue, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_mask_value_outside_domain() {
  let (px, pos, mut mask) = bundle(4, 3);
  mask[1] = 0.5;
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedMaskValue(e)) if e.index() == 1 => assert_eq!(e.value(), 0.5),
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
    Err(Error::PreprocessedMaskOrder(2)) => {}
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
    Err(Error::PreprocessedPadNonZero(e)) if e.feature() == "pixel_values" => {
      assert_eq!(e.index(), 3 * PATCH_DIM + 5)
    }
    other => panic!("expected PreprocessedPadNonZero, got {other:?}"),
  }
}

#[test]
fn preprocessed_image_rejects_nonzero_position_embedding_pad_row() {
  let (px, mut pos, mask) = bundle(4, 3);
  pos[3 * EMBEDDING_DIM] = 1e-3; // first element of the masked pad row
  match PreprocessedImage::try_new(px, pos, mask, 4) {
    Err(Error::PreprocessedPadNonZero(e)) if e.feature() == "position_embeddings" => {
      assert_eq!(e.index(), 3 * EMBEDDING_DIM)
    }
    other => panic!("expected PreprocessedPadNonZero, got {other:?}"),
  }
}

#[test]
fn check_patch_budget_accepts_equal_and_rejects_mismatch() {
  check_patch_budget(512, 512).expect("equal budgets accepted");
  match check_patch_budget(256, 512) {
    Err(Error::PatchBudgetMismatch(ref e)) if e.input() == 256 && e.model() == 512 => {}
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

// ── The door's own contract ────────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this door's `LoadContract` itself — its feature names, its
// element type, its geometry, its state clause, and the one axis it READS back
// rather than requires — against descriptions built with the same fixture
// machinery, so a mis-stated contract is caught here and a mis-implemented
// checker is caught there.

use crate::{
  AxisRange, ComputeUnits, FeatureInfo, Model, ModelDescription,
  embeddings::siglip::error::contract_violation, model::RawShapeConstraint,
};

/// The patch budget the staged 512-tier conversion pins
/// (`conversion/siglip/scripts/_siglip_common.py`: `PATCH_BUDGET = 512`). The
/// door never spells this number — it reads whatever the graph pins — so it
/// appears here only as the fixture's own choice, and
/// `the_contract_reads_back_whatever_budget_the_graph_pins` proves a different
/// one is equally acceptable.
const STAGED_PATCH_BUDGET: usize = 512;

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  multi_array(name, shape, dtype, false, 2, vec![shape.to_vec()], shape)
}

/// One multi-array feature, spelled out: the constraint's raw type code, its
/// enumerated shapes, and the axes its per-axis ranges pin.
fn multi_array(
  name: &str,
  shape: &[usize],
  dtype: DataType,
  optional: bool,
  raw_type: isize,
  enumerated: Vec<Vec<usize>>,
  pinned: &[usize],
) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    optional,
    Some(RawShapeConstraint::new(
      raw_type,
      enumerated,
      pinned.iter().map(|d| AxisRange::new(*d, 1)).collect(),
    )),
  )
}

/// A vision description at patch budget `p`: the three NaFlex inputs and the
/// one projection output, all fixed-shape f32, no state.
fn vision_description(p: usize) -> ModelDescription {
  ModelDescription::from_parts(
    vec![
      fixed(names::PIXEL_VALUES, &[1, p, PATCH_DIM], DataType::F32),
      fixed(
        names::POSITION_EMBEDDINGS,
        &[1, p, EMBEDDING_DIM],
        DataType::F32,
      ),
      fixed(names::ATTENTION_MASK, &[1, p], DataType::F32),
    ],
    vec![fixed(
      names::IMAGE_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into the siglip
/// error vocabulary — exactly what `ImageEmbedder::from_parts` does after
/// `Model::load`.
fn check(description: &ModelDescription) -> Result<()> {
  let declared = declared_patch_budget(description)?;
  crate::model::contract::check_load_contract(description, &image_contract(declared))
    .map_err(contract_violation)
}

/// The contract states exactly the geometry the staged conversion emits.
#[test]
fn the_contract_accepts_the_staged_geometry() {
  assert!(check(&vision_description(STAGED_PATCH_BUDGET)).is_ok());
}

/// **The `AnyFixed` clause, which is the whole reason this door's contract is
/// built at load rather than written down.** The patch budget is the conversion
/// tier's, not this crate's, so a 256-tier graph is as acceptable as a
/// 512-tier one — and both are read back rather than required.
#[test]
fn the_contract_reads_back_whatever_budget_the_graph_pins() {
  for p in [1usize, 64, 256, STAGED_PATCH_BUDGET, 1024] {
    let description = vision_description(p);
    assert!(check(&description).is_ok(), "budget {p}");
    assert_eq!(
      declared_patch_budget(&description).expect("declared"),
      p,
      "the budget read back must be the one the graph pins"
    );
  }
}

/// **The clause `AnyFixed` cannot make, and why the budget is read from ONE
/// feature.** A graph whose three inputs disagree about the budget passes every
/// per-feature clause a per-axis "one fixed size" rule could state, and then
/// fails every prediction — this door builds all three tensors at the budget
/// `pixel_values` declares. The contract states the other two as
/// `Exactly(p)`, so the disagreement is refused at load.
#[test]
fn the_contract_refuses_inputs_that_disagree_about_the_budget() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(
        names::PIXEL_VALUES,
        &[1, STAGED_PATCH_BUDGET, PATCH_DIM],
        DataType::F32,
      ),
      fixed(
        names::POSITION_EMBEDDINGS,
        &[1, 256, EMBEDDING_DIM],
        DataType::F32,
      ),
      fixed(
        names::ATTENTION_MASK,
        &[1, STAGED_PATCH_BUDGET],
        DataType::F32,
      ),
    ],
    vec![fixed(
      names::IMAGE_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::POSITION_EMBEDDINGS),
    "{err}"
  );
}

/// **The flexible-shape refusal**, and it bites hardest on exactly the axis
/// this door reads back: [`crate::FeatureInfo::shape`] reports the DEFAULT
/// shape of a `RangeDims` input, so a graph whose patch budget is a RANGE
/// declares one number and accepts others — and this door would allocate every
/// tensor at that default. `Dim::AnyFixed` requires the axis to admit exactly
/// one size, which under an all-`Exactly`/`AnyFixed` contract means the whole
/// feature must be `ShapeConstraint::Fixed`.
#[test]
fn the_contract_refuses_a_flexible_patch_budget() {
  let description = ModelDescription::from_parts(
    vec![
      multi_array(
        names::PIXEL_VALUES,
        &[1, STAGED_PATCH_BUDGET, PATCH_DIM],
        DataType::F32,
        false,
        3,
        Vec::new(),
        &[1, STAGED_PATCH_BUDGET, PATCH_DIM],
      ),
      fixed(
        names::POSITION_EMBEDDINGS,
        &[1, STAGED_PATCH_BUDGET, EMBEDDING_DIM],
        DataType::F32,
      ),
      fixed(
        names::ATTENTION_MASK,
        &[1, STAGED_PATCH_BUDGET],
        DataType::F32,
      ),
    ],
    vec![fixed(
      names::IMAGE_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::PIXEL_VALUES),
    "{err}"
  );
}

/// A budget of ZERO is refused before any contract is built: `AnyFixed` asks
/// only that the axis admit exactly one size, and zero is one size.
#[test]
fn a_zero_patch_budget_is_refused_before_a_contract_exists() {
  let description = vision_description(0);
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::PIXEL_VALUES),
    "{err}"
  );
}

/// The patch dimension is `3·16·16`, and the NaFlex mask is f32 rather than the
/// int32 a mask usually is — the §0 contract, and the two numbers a wrong
/// conversion would move.
#[test]
fn the_contract_refuses_a_wrong_patch_dim_or_mask_dtype() {
  let wrong_patch_dim = ModelDescription::from_parts(
    vec![
      fixed(
        names::PIXEL_VALUES,
        &[1, STAGED_PATCH_BUDGET, 3 * 14 * 14],
        DataType::F32,
      ),
      fixed(
        names::POSITION_EMBEDDINGS,
        &[1, STAGED_PATCH_BUDGET, EMBEDDING_DIM],
        DataType::F32,
      ),
      fixed(
        names::ATTENTION_MASK,
        &[1, STAGED_PATCH_BUDGET],
        DataType::F32,
      ),
    ],
    vec![fixed(
      names::IMAGE_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(matches!(
    check(&wrong_patch_dim),
    Err(Error::ContractMismatch(_))
  ));

  let mut int_mask = vision_description(STAGED_PATCH_BUDGET);
  int_mask = ModelDescription::from_parts(
    vec![
      fixed(
        names::PIXEL_VALUES,
        &[1, STAGED_PATCH_BUDGET, PATCH_DIM],
        DataType::F32,
      ),
      fixed(
        names::POSITION_EMBEDDINGS,
        &[1, STAGED_PATCH_BUDGET, EMBEDDING_DIM],
        DataType::F32,
      ),
      fixed(
        names::ATTENTION_MASK,
        &[1, STAGED_PATCH_BUDGET],
        DataType::I32,
      ),
    ],
    int_mask.outputs().to_vec(),
    Vec::new(),
  );
  let err = check(&int_mask).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::ATTENTION_MASK),
    "{err}"
  );
}

/// **A graph carrying this door's three inputs plus another REQUIRED one**
/// clears every per-feature clause and then fails on every prediction, because
/// [`ImageEmbedder::embed`] supplies exactly those three.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let mut inputs = vision_description(STAGED_PATCH_BUDGET).inputs().to_vec();
  inputs.push(fixed("spatial_shapes", &[1, 2], DataType::I32));
  let description = ModelDescription::from_parts(
    inputs,
    vec![fixed(
      names::IMAGE_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableInput(name)) if name == "spatial_shapes"),
    "{:?}",
    check(&description)
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one, so it cannot make this door's prediction fail.
#[test]
fn the_contract_accepts_an_extra_optional_input() {
  let mut inputs = vision_description(STAGED_PATCH_BUDGET).inputs().to_vec();
  inputs.push(multi_array(
    "spatial_shapes",
    &[1, 2],
    DataType::I32,
    true,
    2,
    vec![vec![1, 2]],
    &[1, 2],
  ));
  let description = ModelDescription::from_parts(
    inputs,
    vec![fixed(
      names::IMAGE_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
    )],
    Vec::new(),
  );
  assert!(check(&description).is_ok());
}

/// An output the door READS that the graph may leave out: every geometry
/// clause passes and the prediction is still free to omit it.
#[test]
fn the_contract_refuses_an_optional_features_output() {
  let description = ModelDescription::from_parts(
    vision_description(STAGED_PATCH_BUDGET).inputs().to_vec(),
    vec![multi_array(
      names::IMAGE_FEATURES,
      &[1, EMBEDDING_DIM],
      DataType::F32,
      true,
      2,
      vec![vec![1, EMBEDDING_DIM]],
      &[1, EMBEDDING_DIM],
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::IMAGE_FEATURES),
    "{err}"
  );
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful ML Program declaring
/// exactly this door's four features plus a state clears every per-feature
/// clause AND the input set — and only then meets
/// [`ImageEmbedder::embed`], which predicts through the STATELESS API.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let base = vision_description(STAGED_PATCH_BUDGET);
  let description = ModelDescription::from_parts(
    base.inputs().to_vec(),
    base.outputs().to_vec(),
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableState(name)) if name == "kv_cache")
  );
}

// ── The one gate here that loads a real artifact ───────────────────────────

/// **This door's `Checked::new` call site, pinned on a REAL model, in every
/// `cargo test`.**
///
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` is COMMITTED, so
/// unlike everything else in this repository that loads a model this needs no
/// staged artifact and carries no `#[ignore]` — which matters more here than
/// anywhere else in this crate, because the siglip `.mlmodelc` bundles are the
/// one kit `Models/` stages nothing of (only the tokenizer sidecar), so every
/// `tests/siglip/` gate that loads a model is `#[ignore]`d with no artifact to
/// run it against.
#[test]
fn the_image_contract_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; \
     looked for {}",
    bundle.display()
  );

  let model = Model::load(&bundle, ComputeUnits::CpuOnly).expect("the committed bundle loads");
  assert!(
    model.description().input(names::PIXEL_VALUES).is_none(),
    "silero declares no `pixel_values`, which is what makes it this gate's model"
  );

  // The budget cannot even be READ off this description, so the door refuses it
  // before a contract exists — the first of the two refusals `load` runs.
  let err = declared_patch_budget(model.description()).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m)
      if m.feature() == names::PIXEL_VALUES && m.actual() == "missing"),
    "{err}"
  );

  // And with a budget supplied anyway, `Checked::new` itself refuses it.
  let violation = Checked::new(model, &image_contract(STAGED_PATCH_BUDGET))
    .expect_err("silero does not satisfy the siglip vision contract");
  assert!(
    matches!(&violation, crate::model::contract::ContractViolation::Missing(m)
      if m.feature() == names::PIXEL_VALUES),
    "expected `pixel_values` missing, got {violation}"
  );
}
