//! The manifest holds what the conversion recipe measured.
//!
//! Hermetic — a manifest is a value, so these need no artifact. What they are
//! for is the direction a gated test cannot cover: `tests/face/model_io.rs`
//! reads the artifact and checks it against [`MODEL`], so a manifest edited to
//! agree with a re-converted artifact would leave that gate green while
//! silently changing what the door sends. These pin the value itself against
//! the recipe.

use super::{BUNDLE_NAME, MODEL, RECOMMENDED_COMPUTE, STAGED_PATH};
use crate::{
  ComputeUnits,
  embeddings::face::{ChannelOrder, Preprocessing, TensorLayout},
};

/// The contract `conversion/face/README.md` records, field by field.
#[test]
fn the_manifest_is_the_contract_the_recipe_converted() {
  assert_eq!(MODEL.input(), "data");
  assert_eq!(MODEL.output(), "embedding");
  assert_eq!(MODEL.dim(), 512);
  assert_eq!(MODEL.preprocessing(), Preprocessing::ARCFACE);
}

/// The preprocessing spelled out rather than compared to a named constant.
///
/// `Preprocessing::ARCFACE` is shared with any other ArcFace-family artifact,
/// so asserting equality against it says the two agree and not what either
/// one IS. The channel order in particular was decided by measurement — BGR
/// drops the worst same-person pair through InsightFace's own 0.28 line — and
/// a silent flip of that constant would move every cosine this kit produces
/// while every equality above still held.
#[test]
fn the_preprocessing_is_rgb_nchw_and_maps_bytes_onto_minus_one_to_one() {
  let p = MODEL.preprocessing();
  assert_eq!(p.order(), ChannelOrder::Rgb);
  assert_eq!(p.layout(), TensorLayout::Nchw);
  assert_eq!(p.scale(), 1.0 / 127.5);
  assert_eq!(p.bias(), [-1.0, -1.0, -1.0]);

  // `(x - 127.5) / 127.5` at both ends of the byte range — and NOT bit-exact
  // with it, which is worth stating rather than rounding away. `1/127.5` has
  // no exact `f32`, so the affine form the manifest carries maps 255 to
  // 1.0000001 where the divide form gives exactly 1. The gap is 1.2e-7 on a
  // value the network sees at ~1, six orders under fp16's own resolution
  // there, and it is the SAME arithmetic on both sides of every parity number
  // this kit reports: InsightFace's own `blobFromImages(1/127.5, …)` is the
  // affine form too, and so is the recipe's `preprocess`, which cut the
  // committed ONNX reference. Asserting equality with 1.0 would be asserting
  // a divide nobody performs.
  assert_eq!(0.0f32.mul_add(p.scale(), p.bias()[0]), -1.0);
  let top = 255.0f32.mul_add(p.scale(), p.bias()[0]);
  assert!(
    (f64::from(top) - 1.0).abs() < 1e-6,
    "byte 255 must land on the top of the [-1, 1] range, got {top}"
  );
}

/// The measured recommendation, and the fact that it is NOT the door's
/// default — the two answer different questions and a change that collapsed
/// them would go unnoticed otherwise.
#[test]
fn the_recommended_placement_is_the_measured_arm_and_not_the_doors_default() {
  assert_eq!(RECOMMENDED_COMPUTE, ComputeUnits::CpuAndNeuralEngine);
  assert_ne!(
    RECOMMENDED_COMPUTE,
    crate::embeddings::face::DEFAULT_FACE_COMPUTE,
    "the door's default is the planner's choice for ANY artifact; this is a measurement of one"
  );
}

/// The staged path names the bundle, so the two constants cannot drift apart.
#[test]
fn the_staged_path_ends_in_the_bundle_name() {
  assert_eq!(BUNDLE_NAME, "w600k_r50.mlmodelc");
  assert_eq!(STAGED_PATH, format!("Models/facekit/{BUNDLE_NAME}"));
}
