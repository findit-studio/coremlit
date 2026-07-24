//! Ground-truth introspection + provenance pins for the siglip TEXT artifact
//! `siglip2_text_64.mlmodelc`.
//!
//! # Status: Wave C shell (model-gated)
//!
//! `#[ignore]`d until the owner stages the conversion (`SIGLIP_TEST_MODELS`). The
//! contract (§0): `input_ids` int32 `[1, T]` → `text_features` f32 `[1, 768]`,
//! and — the SigLIP text specificity — the input SET is EXACTLY `{input_ids}`
//! (NO `attention_mask`). Wave C fills the exact-SHA manifest.

mod common;

use std::collections::BTreeSet;

use coremlit::{ComputeUnits, DataType, Model, embeddings::siglip::embedding::EMBEDDING_DIM};

/// Text `.mlmodelc` per-file SHA-256, EXACTLY enumerated, from `CHECKSUMS.sha256`
/// of the staged conversion (`conversion/siglip`). As with the vision bundle,
/// `model.mil` / `weights/weight.bin` are deterministic while `coremldata.bin` /
/// `metadata.json` carry a coremltools conversion-instance stamp, so a
/// re-conversion re-pins those two (the exact-set gate's deliberate re-stage
/// behavior).
const ARTIFACT_SHA256: &[(&str, &str)] = &[
  (
    "analytics/coremldata.bin",
    "2e8a886dd9ca5c9983d353876ddb61a99d5870617ae6eb262b2b143ec453ae96",
  ),
  (
    "coremldata.bin",
    "767113ccc24387c3445261f83bc5118fa805ea0fa164f05d31b93aa17031e119",
  ),
  (
    "metadata.json",
    "bc561b31047c1bc0b929af7246d6867cac6b79cd07d029eebb3922a394443aac",
  ),
  (
    "model.mil",
    "540433d3abe2e768c8f124eb3a2f514b3c47247ec3edc29ab52f35ca35c6fb69",
  ),
  (
    "weights/weight.bin",
    "8b781500cc6a596fa3a27b16b56e3d81e675e642ecd3542722d1f185aa0a6f67",
  ),
];

/// Text graph I/O contract: resolves `T` from `input_ids [1, T]` int32 and
/// asserts the input SET is EXACTLY `{input_ids}` (no `attention_mask`).
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn text_io_matches_spec_and_has_no_attention_mask() {
  let model = Model::load(common::text_model_path(), ComputeUnits::CpuOnly).unwrap();
  let d = model.description();

  let ids = d.input("input_ids").expect("input_ids input");
  assert_eq!(ids.shape()[0], 1, "input_ids batch");
  assert_eq!(ids.data_type(), Some(DataType::I32));
  let t = ids.shape()[1];
  assert!(t >= 1, "resolved window T must be positive");

  // The SigLIP text graph has a SINGLE input — no attention_mask.
  let input_names: BTreeSet<&str> = d.inputs().iter().map(|f| f.name()).collect();
  assert_eq!(
    input_names,
    BTreeSet::from(["input_ids"]),
    "text must declare EXACTLY {{input_ids}} — no attention_mask"
  );

  let out = d.output("text_features").expect("text_features output");
  assert_eq!(out.shape(), &[1, EMBEDDING_DIM]);
  assert_eq!(out.data_type(), Some(DataType::F32));
}

/// Exact-SHA manifest for the text bundle. Wave C fills `ARTIFACT_SHA256`.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn text_artifact_bytes_match_pinned_sha256() {
  common::assert_exact_sha_manifest(&common::text_model_path(), ARTIFACT_SHA256);
}
