//! Ground-truth introspection + provenance pins for the CLAP **text** artifact
//! `clap_text.mlmodelc`. Every shape/dtype claim comes from loading the real
//! `.mlmodelc`; every SHA comes from the downloaded bytes. See
//! `tests/clap/model_io.rs` for the shared artifact/license/revision record.
//!
//! # Per-file SHA-256 (pinned; `CHECKSUMS.sha256` on the HF revision)
//!
//! fp16 (revision `97d631f3814e1e46b798a8e88c9aa2e2202fdf67`):
//!
//! | File | SHA-256 |
//! |---|---|
//! | `clap_text.mlmodelc/model.mil` | `0ec4d567c8d26a1aac4e161b1bdea6f9cd36441ec9e1da51fa5d28d79c22b744` |
//! | `clap_text.mlmodelc/weights/weight.bin` | `7f4e15e9ccb0ffbc2341eec286e9d9934d3d3d8d6465dfddebed248bddc0e3dd` |
//!
//! int8 (the quantized variant shipped at revision `02a99c6a`; same I/O
//! contract, compressed weights):
//!
//! | File | SHA-256 |
//! |---|---|
//! | `clap_text_int8.mlmodelc/model.mil` | `0cc3ccdcf48e622a4701fa44e4c2096f0bead887461cb400e2b4af4e5641c2ee` |
//! | `clap_text_int8.mlmodelc/weights/weight.bin` | `f181a595cefce402335499c32ea2f9727ef334afea9c592a2eabebb4172350a0` |
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
  let cases = [
    (
      "model.mil",
      "0ec4d567c8d26a1aac4e161b1bdea6f9cd36441ec9e1da51fa5d28d79c22b744",
    ),
    (
      "weights/weight.bin",
      "7f4e15e9ccb0ffbc2341eec286e9d9934d3d3d8d6465dfddebed248bddc0e3dd",
    ),
  ];
  for (relative, expected) in cases {
    let actual = common::sha256_hex(&dir.join(relative));
    assert_eq!(actual, expected, "sha256 drift on text artifact {relative}");
  }
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
  let cases = [
    (
      "model.mil",
      "0cc3ccdcf48e622a4701fa44e4c2096f0bead887461cb400e2b4af4e5641c2ee",
    ),
    (
      "weights/weight.bin",
      "f181a595cefce402335499c32ea2f9727ef334afea9c592a2eabebb4172350a0",
    ),
  ];
  for (relative, expected) in cases {
    let actual = common::sha256_hex(&dir.join(relative));
    assert_eq!(
      actual, expected,
      "sha256 drift on text int8 artifact {relative}"
    );
  }
}
