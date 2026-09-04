//! Ground-truth introspection + provenance pins for the siglip TEXT artifact
//! `siglip2_text_64.mlmodelc`.
//!
//! # Status: complete, model-gated
//!
//! `#[ignore]`d until the artifacts are staged (`SIGLIP_TEST_MODELS`) — fetched
//! from the published bundle (`FinDIT-Studio/siglip2-naflex-coreml`) or re-derived
//! per the `conversion/siglip` runbook. The contract (§0): `input_ids` int32
//! `[1, T]` → `text_features` f32 `[1, 768]`, and — the SigLIP text specificity —
//! the input SET is EXACTLY `{input_ids}` (NO `attention_mask`). The exact-SHA
//! manifest is already filled and matches that bundle's `CHECKSUMS.sha256`.

mod common;

use std::collections::BTreeSet;

use coremlit::{
  ComputeUnits, DataType, Model,
  embeddings::siglip::{embedding::EMBEDDING_DIM, text::TextEmbedder},
};

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

  // The assertions above read the declaration; this one runs the DOOR over it.
  // `TextEmbedder::from_file` builds a `Checked` and reads the window back off
  // it, so a real artifact that satisfies every clause above and fails the
  // door's own `LoadContract` — an `input_ids` whose window is a RANGE rather
  // than a pin (which `shape()` above cannot distinguish), a declared state
  // buffer, an optional `text_features` — is caught here and only here. The
  // exact-input-SET assertion above is no longer this test's alone either: the
  // contract refuses any required input it does not name.
  let embedder = TextEmbedder::from_file(common::text_model_path())
    .expect("the staged artifact must satisfy this door's load contract");
  assert_eq!(
    embedder.max_tokens(),
    t,
    "the window read back off the checked model must be the one the graph pins"
  );
}

/// Exact-SHA manifest for the text bundle, read from
/// `MODELS_LOCK.d/siglip2-naflex@<revision>.sha256` — which IS the published
/// bundle's own `CHECKSUMS.sha256` at the pinned revision.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn text_artifact_bytes_match_pinned_sha256() {
  common::assert_exact_sha_manifest(
    &common::text_model_path(),
    &common::artifact_sha256("siglip2_text_64.mlmodelc"),
  );
}
