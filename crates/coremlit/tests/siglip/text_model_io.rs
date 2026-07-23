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

/// Text `.mlmodelc` per-file SHA-256, EXACTLY enumerated. Filled from
/// `CHECKSUMS.sha256` at `common::SIGLIP_REVISION` (Wave C).
const ARTIFACT_SHA256: &[(&str, &str)] = &[
  // ("coremldata.bin", "<sha256 — Wave C>"),
  // ("model.mil", "<sha256 — Wave C>"),
  // ("weights/weight.bin", "<sha256 — Wave C>"),
];

/// Text graph I/O contract: resolves `T` from `input_ids [1, T]` int32 and
/// asserts the input SET is EXACTLY `{input_ids}` (no `attention_mask`).
#[test]
#[ignore = "requires staged siglip text model (SIGLIP_TEST_MODELS) — Wave C"]
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
#[ignore = "requires staged siglip text model (SIGLIP_TEST_MODELS) — Wave C"]
fn text_artifact_bytes_match_pinned_sha256() {
  common::assert_exact_sha_manifest(&common::text_model_path(), ARTIFACT_SHA256);
}
