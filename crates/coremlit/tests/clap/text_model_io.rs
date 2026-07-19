//! Ground-truth introspection + provenance pins for the CLAP **text** artifact
//! `clap_text.mlmodelc`. Every shape/dtype claim comes from loading the real
//! `.mlmodelc`; every SHA comes from the downloaded bytes. See
//! `tests/clap/model_io.rs` for the shared artifact/license/revision record.
//!
//! # Per-file SHA-256 (pinned; EVERY file in each `.mlmodelc`, from
//! `CHECKSUMS.sha256` at revision `02a99c6a8be21da1e9a947499ea503a10c80c4f1`)
//!
//! Both tiers pin all five compiled-model files, so drift in ANY committed
//! artifact byte reds. The fp16 bytes are unchanged from the original `97d631f3…`
//! publication.
//!
//! fp16 (`clap_text.mlmodelc`):
//!
//! | File | SHA-256 |
//! |---|---|
//! | `analytics/coremldata.bin` | `6ce0a63b5bf13fc8f60fb1b8956373a2317d0a7349a0b6899ce5030eb9f8aef6` |
//! | `coremldata.bin` | `edbf40e51518ad8f65a1d070f4c0e9b87fea644e6d69eb9eb55652e9fc697885` |
//! | `metadata.json` | `8592b665a1e5a2d9b9078917b3dc9aedb0eb3fa8b8b4ae4630ee78b953082a7c` |
//! | `model.mil` | `0ec4d567c8d26a1aac4e161b1bdea6f9cd36441ec9e1da51fa5d28d79c22b744` |
//! | `weights/weight.bin` | `7f4e15e9ccb0ffbc2341eec286e9d9934d3d3d8d6465dfddebed248bddc0e3dd` |
//!
//! int8 (`clap_text_int8.mlmodelc`; the 2×-smaller variant, same I/O contract):
//!
//! | File | SHA-256 |
//! |---|---|
//! | `analytics/coremldata.bin` | `b914a93d50ae8336aad0977e28c4c1f84dccc39f7845fcd1660c43d565ca6fb7` |
//! | `coremldata.bin` | `95bc733c2b0d3d2fa64edf548f77a50f344c020cce382a0b96c46477ad4a0b84` |
//! | `metadata.json` | `0ed552101f045f5864c9d7212bffe09b478a2c1e8c39ec1d2732f95d206ee1df` |
//! | `model.mil` | `0cc3ccdcf48e622a4701fa44e4c2096f0bead887461cb400e2b4af4e5641c2ee` |
//! | `weights/weight.bin` | `f181a595cefce402335499c32ea2f9727ef334afea9c592a2eabebb4172350a0` |
//!
//! # Contract (matches the spec table and T1's I/O record)
//!
//! `input_ids` int32 `[1, 512]` + `attention_mask` int32 `[1, 512]` →
//! `text_embeds` fp32 `[1, 512]` (projection, PRE-L2-norm; clapkit normalizes in
//! Rust). Fixed length 512 (the model max); shorter inputs are right-padded +
//! masked, reproducing the natural-length embedding exactly (T1). No delta found.
//! The int8 variant shares this contract exactly.

mod common;

use coremlit::{
  ComputeUnits, DataType, Model,
  embeddings::clap::{embedding::EMBEDDING_DIM, text::TEXT_MAX_TOKENS},
};

#[test]
#[ignore = "requires local clapkit models (CLAPKIT_TEST_MODELS)"]
fn clap_text_io_matches_spec() {
  let model = Model::load(common::text_model_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  for name in ["input_ids", "attention_mask"] {
    let input = description
      .input(name)
      .unwrap_or_else(|| panic!("{name} input"));
    assert_eq!(input.shape(), &[1, TEXT_MAX_TOKENS], "{name} shape");
    assert_eq!(input.data_type(), Some(DataType::I32), "{name} dtype");
  }

  let output = description
    .output("text_embeds")
    .expect("text_embeds output");
  assert_eq!(output.shape(), &[1, EMBEDDING_DIM]);
  assert_eq!(output.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local clapkit models (CLAPKIT_TEST_MODELS)"]
fn clap_text_artifacts_match_pinned_sha256() {
  let dir = common::text_model_path();
  // The EXACT pinned manifest: every file in the `.mlmodelc` (per
  // `CHECKSUMS.sha256` @ 02a99c6a). The helper enumerates + set-compares against
  // these keys before hashing, so an added/removed artifact reds too.
  let cases = [
    (
      "analytics/coremldata.bin",
      "6ce0a63b5bf13fc8f60fb1b8956373a2317d0a7349a0b6899ce5030eb9f8aef6",
    ),
    (
      "coremldata.bin",
      "edbf40e51518ad8f65a1d070f4c0e9b87fea644e6d69eb9eb55652e9fc697885",
    ),
    (
      "metadata.json",
      "8592b665a1e5a2d9b9078917b3dc9aedb0eb3fa8b8b4ae4630ee78b953082a7c",
    ),
    (
      "model.mil",
      "0ec4d567c8d26a1aac4e161b1bdea6f9cd36441ec9e1da51fa5d28d79c22b744",
    ),
    (
      "weights/weight.bin",
      "7f4e15e9ccb0ffbc2341eec286e9d9934d3d3d8d6465dfddebed248bddc0e3dd",
    ),
  ];
  common::assert_exact_sha_manifest(&dir, &cases);
}

#[test]
#[ignore = "requires local clapkit int8 models (CLAPKIT_TEST_MODELS)"]
fn clap_text_int8_io_matches_spec() {
  let model = Model::load(common::text_model_int8_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  for name in ["input_ids", "attention_mask"] {
    let input = description
      .input(name)
      .unwrap_or_else(|| panic!("{name} input"));
    assert_eq!(input.shape(), &[1, TEXT_MAX_TOKENS], "{name} shape");
    assert_eq!(input.data_type(), Some(DataType::I32), "{name} dtype");
  }

  let output = description
    .output("text_embeds")
    .expect("text_embeds output");
  assert_eq!(output.shape(), &[1, EMBEDDING_DIM]);
  assert_eq!(output.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local clapkit int8 models (CLAPKIT_TEST_MODELS)"]
fn clap_text_int8_artifacts_match_pinned_sha256() {
  let dir = common::text_model_int8_path();
  // The EXACT pinned manifest for the int8 `.mlmodelc` (per `CHECKSUMS.sha256` @
  // 02a99c6a); the helper enumerates + set-compares before hashing, so a
  // missing/added artifact reds too.
  let cases = [
    (
      "analytics/coremldata.bin",
      "b914a93d50ae8336aad0977e28c4c1f84dccc39f7845fcd1660c43d565ca6fb7",
    ),
    (
      "coremldata.bin",
      "95bc733c2b0d3d2fa64edf548f77a50f344c020cce382a0b96c46477ad4a0b84",
    ),
    (
      "metadata.json",
      "0ed552101f045f5864c9d7212bffe09b478a2c1e8c39ec1d2732f95d206ee1df",
    ),
    (
      "model.mil",
      "0cc3ccdcf48e622a4701fa44e4c2096f0bead887461cb400e2b4af4e5641c2ee",
    ),
    (
      "weights/weight.bin",
      "f181a595cefce402335499c32ea2f9727ef334afea9c592a2eabebb4172350a0",
    ),
  ];
  common::assert_exact_sha_manifest(&dir, &cases);
}
