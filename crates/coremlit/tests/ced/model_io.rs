//! Ground-truth introspection + provenance pins for the CED CoreML artifact
//! `ced_tiny.mlmodelc`. Every shape/dtype claim comes from loading the real
//! `.mlmodelc` via `coremlit::Model::load` + `.description()`; every SHA comes
//! from the downloaded bytes.
//!
//! # Status: Wave-B/C shell (model-gated)
//!
//! `#[ignore]`d until the owner stages the conversion (`CED_TEST_MODELS`).
//! Wave B fills `ARTIFACT_SHA256` from the staged `CHECKSUMS.sha256` and
//! ratifies (or corrects) the believed contract; Wave C pins the HF revision
//! (`common::CED_REVISION`).
//!
//! # Believed contract (probe-pinned)
//!
//! `mel` f32 `[1, 64, 1001]` → `logits` f32 `[1, 527]` (PRE-sigmoid; the
//! module applies sigmoid in Rust). Names declared by the module (`names`
//! mod); the Wave-B export must emit exactly them.

mod common;

use std::collections::BTreeSet;

use coremlit::{ComputeUnits, DataType, Model, audio::ced::NUM_CLASSES};

/// The `.mlmodelc` bundle's per-file SHA-256, EXACTLY enumerated (the #30
/// pattern). Wave B fills this from the staged `CHECKSUMS.sha256`:
const ARTIFACT_SHA256: &[(&str, &str)] = &[
  // ("coremldata.bin", "<sha256 — Wave B>"),
  // ("model.mil", "<sha256 — Wave B>"),
  // ("weights/weight.bin", "<sha256 — Wave B>"),
];

/// Believed I/O contract, both directions: exact input/output name SETS and
/// per-feature shape/dtype. A divergence here in Wave B is the probe
/// correcting the believed constants — a constants + golden change by design.
#[test]
#[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave B"]
fn io_contract_matches_the_believed_spec() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).unwrap();
  let d = model.description();

  let mel = d.input("mel").expect("mel input");
  assert_eq!(
    mel.shape(),
    &[1, 64, 1001],
    "believed [1, n_mels, T] layout"
  );
  assert_eq!(mel.data_type(), Some(DataType::F32));

  let logits = d.output("logits").expect("logits output");
  assert_eq!(logits.shape(), &[1, NUM_CLASSES]);
  assert_eq!(logits.data_type(), Some(DataType::F32));

  let input_names: BTreeSet<&str> = d.inputs().iter().map(|f| f.name()).collect();
  assert_eq!(input_names, BTreeSet::from(["mel"]), "exactly one input");
  let output_names: BTreeSet<&str> = d.outputs().iter().map(|f| f.name()).collect();
  assert_eq!(
    output_names,
    BTreeSet::from(["logits"]),
    "exactly one output"
  );
}

/// Exact-SHA manifest over the staged bundle. Wave B fills `ARTIFACT_SHA256`.
#[test]
#[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave B"]
fn artifact_bytes_match_pinned_sha256() {
  assert!(
    !ARTIFACT_SHA256.is_empty(),
    "Wave B must pin the artifact manifest before this gate can pass"
  );
  common::assert_exact_sha_manifest(&common::model_path(), ARTIFACT_SHA256);
}

/// HERMETIC non-vacuity for the manifest walker: sidecars are skipped, real
/// extras still red (runs in the default suite — no model needed).
#[test]
fn collect_files_rel_skips_sidecars_but_surfaces_real_extras() {
  let dir = tempfile::tempdir().unwrap();
  std::fs::write(dir.path().join("model.mil"), b"mil").unwrap();
  std::fs::write(dir.path().join(".DS_Store"), b"junk").unwrap();
  std::fs::write(dir.path().join("._model.mil"), b"appledouble").unwrap();
  std::fs::create_dir(dir.path().join("weights")).unwrap();
  std::fs::write(dir.path().join("weights/weight.bin"), b"w").unwrap();

  let mut found = Vec::new();
  common::collect_files_rel(dir.path(), "", &mut found);
  found.sort();
  assert_eq!(found, vec!["model.mil", "weights/weight.bin"]);
}
