//! Ground-truth introspection + provenance pins for the granite CoreML artifact
//! `granite_97m_512.mlmodelc`. Every shape/dtype claim comes from loading the
//! real `.mlmodelc` via `coremlit::Model::load` + `.description()`; every SHA
//! comes from the downloaded bytes.
//!
//! # Artifact
//!
//! Source: [`FinDIT-Studio/embedkit-coreml`](https://huggingface.co/FinDIT-Studio/embedkit-coreml),
//! revision `81852f70`, converted by embedkit T1 from
//! [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
//! (ModernBERT encoder + CLS pooling, fp16). Gitignored, fetched dev-time under
//! `Models/embedkit-granite/` (`EMBEDKIT_TEST_MODELS`).
//!
//! # License (load-bearing)
//!
//! The upstream IBM granite model is **Apache-2.0**; the CoreML artifact is a
//! format conversion (no weights retrained or altered). Preserve the Apache-2.0
//! attribution when redistributing the weights (crate `NOTICE`, section 7).
//!
//! # Contract (matches metadata.json and the spec table)
//!
//! `input_ids` int32 `[1, 512]` + `attention_mask` int32 `[1, 512]` →
//! `embedding` fp32 `[1, 384]` (CLS-pooled projection, PRE-L2-norm; the module
//! normalizes in Rust). Fixed length 512 (the export sequence length).

mod common;

use std::collections::BTreeSet;

use coremlit::{
  ComputeUnits, DataType, Model,
  embeddings::granite::{
    Error, MAX_TOKENS, TextEmbedder, TextEmbedderOptions, embedding::EMBEDDING_DIM,
  },
};

/// The `.mlmodelc` bundle's per-file SHA-256, EXACTLY enumerated (the #30
/// pattern): the discovered file set is compared against these keys before any
/// hashing, so a file added to or removed from the bundle fails the gate rather
/// than slipping past a fixed named list. Paths are relative to the `.mlmodelc`
/// directory; hashes are from `CHECKSUMS.sha256` at revision `81852f70`.
const ARTIFACT_SHA256: &[(&str, &str)] = &[
  (
    "analytics/coremldata.bin",
    "ae37c06948edcc0f030369f8563d63b80a5bcd349ecb6d4219dcc7d3d3525fe9",
  ),
  (
    "coremldata.bin",
    "e8e470b2d49b73cf350eaa2c2f97fb39c99355c4bc507501675a8e53282cc337",
  ),
  (
    "metadata.json",
    "635299df02dfde6115bbcdb7a8a2cdbe26ecef6be35a393276c5381a32a8f893",
  ),
  (
    "model.mil",
    "b00d8da3bd408b23aa00b6935d35376f88d7d82c7c3f02c19b13375cbea42610",
  ),
  (
    "weights/weight.bin",
    "276bc93c49a4f37ffefdfb2e10f7d7e1ef57db9027c7ad0d3f2e4160f81a79be",
  ),
];

#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn granite_io_matches_spec() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  // Inputs, BOTH directions: each pinned name exists at the pinned shape/dtype…
  for name in ["input_ids", "attention_mask"] {
    let input = description
      .input(name)
      .unwrap_or_else(|| panic!("{name} input"));
    assert_eq!(input.shape(), &[1, MAX_TOKENS], "{name} shape");
    assert_eq!(input.data_type(), Some(DataType::I32), "{name} dtype");
  }
  // …and the model declares EXACTLY these two inputs (no unpinned extra).
  let input_names: BTreeSet<&str> = description.inputs().iter().map(|f| f.name()).collect();
  assert_eq!(
    input_names,
    BTreeSet::from(["input_ids", "attention_mask"]),
    "granite must declare exactly input_ids + attention_mask"
  );

  let output = description.output("embedding").expect("embedding output");
  assert_eq!(output.shape(), &[1, EMBEDDING_DIM]);
  assert_eq!(output.data_type(), Some(DataType::F32));
  let output_names: BTreeSet<&str> = description.outputs().iter().map(|f| f.name()).collect();
  assert_eq!(
    output_names,
    BTreeSet::from(["embedding"]),
    "granite must declare exactly the `embedding` output"
  );
}

#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn granite_artifact_bytes_match_pinned_sha256() {
  let bundle = common::model_path();

  // EXACT-enumerate (the #30 pattern): the discovered file set must EQUAL the
  // pinned key set — no unpinned extra (a file slipped into the bundle) and none
  // missing — BEFORE hashing. Hashing only the 5 hard-coded keys never notices a
  // 6th file.
  let pinned: BTreeSet<String> = ARTIFACT_SHA256
    .iter()
    .map(|(rel, _)| (*rel).to_string())
    .collect();
  let mut discovered: BTreeSet<String> = BTreeSet::new();
  common::collect_files_rel(&bundle, &bundle, &mut discovered);
  assert_eq!(
    discovered,
    pinned,
    "granite .mlmodelc tree (revision {}) does not match the pinned SHA-256 key set — \
     unpinned extras: {:?}, pinned-but-absent: {:?}. A file was added to or removed from the \
     bundle; re-introspect and re-pin.",
    common::EMBEDKIT_REVISION,
    discovered.difference(&pinned).collect::<Vec<_>>(),
    pinned.difference(&discovered).collect::<Vec<_>>(),
  );

  for (rel, expected) in ARTIFACT_SHA256 {
    let path = bundle.join(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
      panic!(
        "read {} (pinned at embedkit revision {}): {e}",
        path.display(),
        common::EMBEDKIT_REVISION
      )
    });
    let actual = common::sha256_hex(&bytes);
    assert_eq!(
      &actual,
      expected,
      "{}: sha256 drift against the pin at revision {} — the downloaded bytes changed; \
       re-run the introspection tests and re-pin",
      path.display(),
      common::EMBEDKIT_REVISION
    );
  }
}

/// A caller-supplied tokenizer that parses but does not match the Granite
/// contract is refused at construction, fail-closed — the audit's live repro (a
/// tiny WordLevel tokenizer via `from_memory`). The contract gate runs before
/// `Model::load`, so this proves the constructor's fail-closed order end-to-end.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn from_memory_rejects_foreign_tokenizer() {
  const TINY_WORDLEVEL: &[u8] = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"world":1},"unk_token":"<unk>"}}"#;
  let err = TextEmbedder::from_memory(
    common::model_path(),
    TINY_WORDLEVEL,
    TextEmbedderOptions::new(),
  )
  .expect_err("a foreign tokenizer must be refused at construction");
  assert!(
    matches!(err, Error::TokenizerContractMismatch { .. }),
    "expected TokenizerContractMismatch, got {err:?}"
  );
}

/// Hermetic non-vacuity proof for [`common::collect_files_rel`]'s sidecar
/// filter (no staged model needed). On exFAT/FAT/SMB volumes macOS
/// materializes AppleDouble `._*` and `.DS_Store` sidecars inside `.mlmodelc`
/// bundles; discovery must drop EXACTLY those, while every real file —
/// crucially including an unpinned real extra — still reaches the exact-set
/// gate in [`granite_artifact_bytes_match_pinned_sha256`]. This proves the
/// filter fixes the false-failure WITHOUT blanket-suppressing genuine extras.
#[test]
fn collect_files_rel_skips_sidecars_but_surfaces_real_extras() {
  let tmp = tempfile::tempdir().expect("create temp dir");
  let bundle = tmp.path().join("granite_97m_512.mlmodelc");
  std::fs::create_dir_all(bundle.join("weights")).expect("mkdir bundle weights/");

  // Two real, pinned-style artifacts (one nested).
  std::fs::write(bundle.join("model.mil"), b"mil").expect("write model.mil");
  std::fs::write(bundle.join("weights/weight.bin"), b"w").expect("write weight.bin");
  // OS-generated sidecars at two depths — every one must be skipped.
  std::fs::write(bundle.join("._model.mil"), b"ad").expect("write ._model.mil");
  std::fs::write(bundle.join(".DS_Store"), b"ds").expect("write .DS_Store");
  std::fs::write(bundle.join("weights/._weight.bin"), b"ad").expect("write nested ._");
  // A real, ordinary-named file that is NOT a sidecar and NOT pinned.
  std::fs::write(bundle.join("rogue.bin"), b"x").expect("write rogue.bin");

  // Enumerate relative to the bundle root, exactly as
  // `granite_artifact_bytes_match_pinned_sha256` does (`root == bundle`).
  let mut discovered: BTreeSet<String> = BTreeSet::new();
  common::collect_files_rel(&bundle, &bundle, &mut discovered);

  // Discovery keeps every real file and drops every sidecar (at both depths).
  assert_eq!(
    discovered,
    BTreeSet::from([
      "model.mil".to_string(),
      "rogue.bin".to_string(),
      "weights/weight.bin".to_string(),
    ]),
    "discovery must exclude `._*`/.DS_Store sidecars and keep every real file"
  );

  // The filter did NOT blanket-suppress extras: an ordinary unpinned file still
  // breaks the exact-set equality and shows up as the sole difference against a
  // pin listing only the two real artifacts.
  let pinned: BTreeSet<String> =
    BTreeSet::from(["model.mil".to_string(), "weights/weight.bin".to_string()]);
  assert_ne!(
    discovered, pinned,
    "a real unpinned extra must still break the exact-set equality"
  );
  let extras: Vec<String> = discovered.difference(&pinned).cloned().collect();
  assert_eq!(
    extras,
    vec!["rogue.bin".to_string()],
    "the surviving extra must be exactly the real unpinned file, not a sidecar"
  );
}
