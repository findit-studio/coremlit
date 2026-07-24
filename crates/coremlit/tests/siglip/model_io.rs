//! Ground-truth introspection + provenance pins for the siglip VISION artifact
//! `siglip2_vision_512.mlmodelc` and the base position-grid sidecar.
//!
//! # Status: Wave C shell (model-gated)
//!
//! The I/O + exact-SHA gates below are `#[ignore]`d until the owner stages the
//! conversion under `Models/siglip2-naflex/` (`SIGLIP_TEST_MODELS`) per the port
//! plan's conversion runbook, at which point the pinned SHA manifest
//! (`ARTIFACT_SHA256`, the sidecar SHA) is filled from `CHECKSUMS.sha256` at the
//! recorded revision. The contract mirrors the plan's §0 table:
//! `pixel_values` f32 `[1, P, 768]` + `position_embeddings` f32 `[1, P, 768]` +
//! `attention_mask` f32 `[1, P]` → `image_features` f32 `[1, 768]`.
//!
//! The sidecar-filter non-vacuity test IS hermetic (no model) and runs now.

mod common;

use std::collections::BTreeSet;

use coremlit::{ComputeUnits, DataType, Model, embeddings::siglip::embedding::EMBEDDING_DIM};

/// Vision `.mlmodelc` per-file SHA-256, EXACTLY enumerated (the #30 pattern),
/// from `CHECKSUMS.sha256` of the staged conversion (`conversion/siglip`). The
/// graph bytes (`model.mil`, `weights/weight.bin`) are deterministic from the
/// pinned checkpoint + toolchain; `coremldata.bin` / `metadata.json` also carry a
/// coremltools conversion-instance stamp (a `conversion_date` and a non-repeatable
/// blob), so a re-conversion re-pins these two — the exact-set gate's intended
/// "deliberate re-stage" behavior. A later public artifact re-upload freezes them.
const ARTIFACT_SHA256: &[(&str, &str)] = &[
  (
    "analytics/coremldata.bin",
    "0abe1152b48156d8a41a116660d8848fafa7de70643aac864a6e842caed75d37",
  ),
  (
    "coremldata.bin",
    "96ce055e241ab4302d563965ee5f70f36200794b73331d548d264be62ddff2f1",
  ),
  (
    "metadata.json",
    "1c9c16e43fc421919ea91a9792e386f62a8ab6da08d88a50d69e1dca54be849c",
  ),
  (
    "model.mil",
    "427a2cfe2b6b9caef6f34ae93b9bfc4441541513621b12daaec679d8cd844747",
  ),
  (
    "weights/weight.bin",
    "31fc44e771553c5b28b7af6561b46650ce5e1e4711dfef9f471ed32d502077b6",
  ),
];

/// The base position-grid sidecar's SHA-256 (deterministic: the checkpoint's
/// `position_embedding.weight` reshaped 16×16×768, little-endian f32).
const SIDECAR_SHA256: &str = "3ba1ba032ad8d97e0a1afebf4513615fbfedb56f646c14dcdb83d3c228c12860";

/// Vision graph I/O contract (§0): resolves `P` from `pixel_values [1, P, 768]`
/// and cross-checks `position_embeddings`, `attention_mask`, and the exact input
/// SET against it. Wave C: extend with the exact-SHA manifest.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn vision_io_matches_spec() {
  let model = Model::load(common::vision_model_path(), ComputeUnits::CpuOnly).unwrap();
  let d = model.description();

  let pv = d.input("pixel_values").expect("pixel_values input");
  assert_eq!(pv.shape()[0], 1, "pixel_values batch");
  assert_eq!(pv.shape()[2], 768, "pixel_values patch_dim = 3·16·16");
  assert_eq!(pv.data_type(), Some(DataType::F32));
  let p = pv.shape()[1];
  assert!(p >= 1, "resolved patch budget P must be positive");

  let pe = d
    .input("position_embeddings")
    .expect("position_embeddings input");
  assert_eq!(pe.shape(), &[1, p, EMBEDDING_DIM]);
  assert_eq!(pe.data_type(), Some(DataType::F32));

  let mask = d.input("attention_mask").expect("attention_mask input");
  assert_eq!(mask.shape(), &[1, p]);
  // Note: the NaFlex mask input is f32 (not int32) — the §0 contract.
  assert_eq!(mask.data_type(), Some(DataType::F32));

  let input_names: BTreeSet<&str> = d.inputs().iter().map(|f| f.name()).collect();
  assert_eq!(
    input_names,
    BTreeSet::from(["pixel_values", "position_embeddings", "attention_mask"]),
    "vision must declare exactly these three inputs"
  );

  let out = d.output("image_features").expect("image_features output");
  assert_eq!(out.shape(), &[1, EMBEDDING_DIM]);
  assert_eq!(out.data_type(), Some(DataType::F32));
}

/// Exact-SHA manifest for the vision bundle + the pos-emb sidecar. Wave C fills
/// `ARTIFACT_SHA256` and the sidecar SHA from `CHECKSUMS.sha256`.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn vision_artifact_bytes_match_pinned_sha256() {
  common::assert_exact_sha_manifest(&common::vision_model_path(), ARTIFACT_SHA256);
  assert_eq!(
    common::sha256_file(&common::pos_embed_path()),
    SIDECAR_SHA256,
    "pos-emb sidecar bytes drifted from the pinned 16×16×768 grid"
  );
}

/// Hermetic non-vacuity proof for [`common::collect_files_rel`]'s sidecar filter
/// (no staged model needed): AppleDouble `._*` / `.DS_Store` are dropped at every
/// depth, while a real unpinned extra still surfaces to the exact-set gate.
#[test]
fn collect_files_rel_skips_sidecars_but_surfaces_real_extras() {
  let tmp = tempfile::tempdir().expect("temp dir");
  let bundle = tmp.path().join("siglip2_vision_512.mlmodelc");
  std::fs::create_dir_all(bundle.join("weights")).expect("mkdir weights");

  std::fs::write(bundle.join("model.mil"), b"mil").expect("write model.mil");
  std::fs::write(bundle.join("weights/weight.bin"), b"w").expect("write weight.bin");
  std::fs::write(bundle.join("._model.mil"), b"ad").expect("write ._model.mil");
  std::fs::write(bundle.join(".DS_Store"), b"ds").expect("write .DS_Store");
  std::fs::write(bundle.join("weights/._weight.bin"), b"ad").expect("write nested ._");
  std::fs::write(bundle.join("rogue.bin"), b"x").expect("write rogue.bin");

  let mut found = Vec::new();
  common::collect_files_rel(&bundle, "", &mut found);
  let discovered: BTreeSet<String> = found.into_iter().collect();

  assert_eq!(
    discovered,
    BTreeSet::from([
      "model.mil".to_string(),
      "rogue.bin".to_string(),
      "weights/weight.bin".to_string(),
    ]),
    "discovery must exclude `._*`/.DS_Store sidecars and keep every real file"
  );

  let pinned: BTreeSet<String> =
    BTreeSet::from(["model.mil".to_string(), "weights/weight.bin".to_string()]);
  let extras: Vec<String> = discovered.difference(&pinned).cloned().collect();
  assert_eq!(
    extras,
    vec!["rogue.bin".to_string()],
    "a real unpinned extra must still surface (not be blanket-suppressed)"
  );
}
