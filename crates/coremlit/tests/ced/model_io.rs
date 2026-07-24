//! Ground-truth introspection + provenance pins for the CED CoreML artifacts
//! `ced_<size>.mlmodelc`, per [`CedModel`] size. Every shape/dtype claim comes
//! from loading the real `.mlmodelc` via `coremlit::Model::load` +
//! `.description()`; every SHA comes from the downloaded bytes.
//!
//! # Status: Wave-B/C shell (model-gated, per size)
//!
//! The per-size gates (`tiny::`/`mini::`/`small::`/`base::`) are `#[ignore]`d
//! until the owner stages that size's conversion (`CED_TEST_MODELS`); each size
//! gates independently by test-name filter. Wave B fills the size's `*_SHA256`
//! table from its staged `CHECKSUMS.sha256` and ratifies (or corrects) the
//! believed contract; Wave C pins the HF revision (`common::ced_revision`). The
//! hermetic self-checks below run with no model.
//!
//! # Believed contract (probe-pinned, shared by all four sizes)
//!
//! `mel` f32 `[1, 64, 1001]` → `logits` f32 `[1, 527]` (PRE-sigmoid; the module
//! applies sigmoid in Rust). Names declared by the module (`names` mod); the
//! Wave-B export must emit exactly them.

mod common;

use std::collections::BTreeSet;

use coremlit::{
  ComputeUnits, DataType, Model,
  audio::ced::{CedModel, NUM_CLASSES},
};

// Per-size exact per-file SHA-256 of each `.mlmodelc` bundle (the #30 pattern).
// Each starts empty; Wave B fills the matching size's table from its staged
// `CHECKSUMS.sha256`:
const TINY_SHA256: &[(&str, &str)] = &[
  // ("coremldata.bin", "<sha256 — Wave B>"),
  // ("model.mil", "<sha256 — Wave B>"),
  // ("weights/weight.bin", "<sha256 — Wave B>"),
];
const MINI_SHA256: &[(&str, &str)] = &[];
const SMALL_SHA256: &[(&str, &str)] = &[];
const BASE_SHA256: &[(&str, &str)] = &[];

/// The exact per-file SHA-256 manifest for `model`'s bundle. Totality is
/// compiler-enforced by the closed [`CedModel`] enum, so a fifth size would
/// force a new table here.
const fn artifact_sha256(model: CedModel) -> &'static [(&'static str, &'static str)] {
  match model {
    CedModel::Tiny => TINY_SHA256,
    CedModel::Mini => MINI_SHA256,
    CedModel::Small => SMALL_SHA256,
    CedModel::Base => BASE_SHA256,
  }
}

/// Shared io-contract core: load the staged bundle and pin the believed I/O —
/// `mel [1, 64, 1001]` f32 → `logits [1, 527]` f32, exact input/output name
/// sets. A divergence in Wave B is the probe correcting the believed constants.
fn io_contract(model: CedModel) {
  let m = Model::load(common::model_path(model), ComputeUnits::CpuOnly).unwrap();
  let d = m.description();

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

/// Shared manifest core: the staged bundle must match `model`'s exact per-file
/// SHA-256 table (filled by Wave B from that size's `CHECKSUMS.sha256`).
fn artifact_manifest(model: CedModel) {
  assert!(
    !artifact_sha256(model).is_empty(),
    "Wave B must pin {model}'s artifact manifest before this gate can pass"
  );
  common::assert_exact_sha_manifest(&common::model_path(model), artifact_sha256(model));
}

macro_rules! per_model_gates {
  ($($m:ident => $v:expr),+ $(,)?) => {$(
    mod $m {
      use super::CedModel;

      #[test]
      #[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave B"]
      fn io_contract_matches_the_believed_spec() {
        super::io_contract($v);
      }

      #[test]
      #[ignore = "requires staged CED model (CED_TEST_MODELS) — Wave B"]
      fn artifact_bytes_match_pinned_sha256() {
        super::artifact_manifest($v);
      }
    }
  )+};
}

per_model_gates!(
  tiny => CedModel::Tiny,
  mini => CedModel::Mini,
  small => CedModel::Small,
  base => CedModel::Base,
);

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

/// HERMETIC: `common::model_path` composes the family root + hyphenated dir +
/// underscored bundle for every size (pure path math — no model, no network).
#[test]
fn model_path_composes_per_size_under_the_family_root() {
  for m in CedModel::ALL {
    assert_eq!(
      common::model_path(m),
      common::models_dir()
        .join(m.dir_name())
        .join(m.mlmodelc_name()),
    );
  }
}

/// HERMETIC: the golden loader's oracle cross-check fails closed on a corpus
/// whose header names a DIFFERENT size — the anti-cross-size-mix-up guard.
#[test]
#[should_panic(expected = "does not match")]
fn golden_corpus_rejects_a_cross_size_oracle() {
  let json = br#"{
    "oracle": {"repo": "mispeech/ced-tiny", "revision": "r", "file": "model.onnx", "sha256": "00"},
    "clips": []
  }"#;
  common::parse_golden_corpus(json, CedModel::Small);
}

/// HERMETIC: a corpus whose oracle header matches the requested size parses and
/// keeps its clips.
#[test]
fn golden_corpus_accepts_a_matching_oracle() {
  let json = br#"{
    "oracle": {"repo": "mispeech/ced-small", "revision": "r", "file": "model.onnx", "sha256": "00"},
    "clips": [{"id": "c0", "file": "clips/c0.wav", "n_samples": 160000, "logits": [0.0, 1.0]}]
  }"#;
  let corpus = common::parse_golden_corpus(json, CedModel::Small);
  assert_eq!(corpus.oracle.repo, "mispeech/ced-small");
  assert_eq!(corpus.clips.len(), 1);
}
