//! Ground-truth introspection + provenance pins for the CLAP **audio** artifact
//! `clap_audio.mlmodelc`. Every shape/dtype claim comes from loading the real
//! `.mlmodelc` via `coremlit::Model::load` + `.description()`; every SHA comes
//! from the downloaded bytes. Text lives in `tests/text_model_io.rs`.
//!
//! # Artifact
//!
//! Source: [`FinDIT-Studio/clapkit-coreml`](https://huggingface.co/FinDIT-Studio/clapkit-coreml),
//! revision (commit SHA) `97d631f3814e1e46b798a8e88c9aa2e2202fdf67`, converted
//! by T1 from `laion/clap-htsat-unfused`
//! (`@ 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`, fp16). Gitignored, fetched
//! dev-time under `Models/clapkit/` (`CLAPKIT_TEST_MODELS`).
//!
//! # License (load-bearing; owner to reconcile before the branch PR)
//!
//! The spec, textclap's MODELS.md, and the clapkit HF README front-matter treat
//! the LAION weights as **CC-BY-4.0** (attribution to `laion/clap-htsat-unfused`
//! required; carried in the crate `NOTICE`). T1 flagged that the live
//! `laion/clap-htsat-unfused` HF card declares **apache-2.0**; attribution to the
//! source repo satisfies both, and reconciling the front-matter is an owner
//! decision recorded here, not this crate's to make.
//!
//! # Per-file SHA-256 (pinned; `CHECKSUMS.sha256` on the HF revision)
//!
//! | File | SHA-256 |
//! |---|---|
//! | `clap_audio.mlmodelc/model.mil` | `1ecf76edf7846153623485a98c0a3d047e660cc68ee10a6d21a39664d309db52` |
//! | `clap_audio.mlmodelc/weights/weight.bin` | `723fe6aab7c4af1c671a210a35c289c67763bc6a7532b9df155a0c3fc0c3c9d7` |
//!
//! # Contract (matches the spec table and T1's I/O record)
//!
//! `input_features` fp32 `[1, 1, 1001, 64]` (HF-mel spectrogram) →
//! `audio_embeds` fp32 `[1, 512]` (projection, PRE-L2-norm; clapkit normalizes
//! in Rust). No delta found.

mod common;

use clapkit::{
  audio::{N_MELS, T_FRAMES},
  embedding::EMBEDDING_DIM,
};
use coremlit::{ComputeUnits, DataType, Model};

#[test]
#[ignore = "requires local clapkit models (CLAPKIT_TEST_MODELS)"]
fn clap_audio_io_matches_spec() {
  let model = Model::load(common::audio_model_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let input = description
    .input("input_features")
    .expect("input_features input");
  assert_eq!(input.shape(), &[1, 1, T_FRAMES, N_MELS]);
  assert_eq!(input.data_type(), Some(DataType::F32));

  let output = description
    .output("audio_embeds")
    .expect("audio_embeds output");
  assert_eq!(output.shape(), &[1, EMBEDDING_DIM]);
  assert_eq!(output.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local clapkit models (CLAPKIT_TEST_MODELS)"]
fn clap_audio_artifacts_match_pinned_sha256() {
  let dir = common::audio_model_path();
  let cases = [
    (
      "model.mil",
      "1ecf76edf7846153623485a98c0a3d047e660cc68ee10a6d21a39664d309db52",
    ),
    (
      "weights/weight.bin",
      "723fe6aab7c4af1c671a210a35c289c67763bc6a7532b9df155a0c3fc0c3c9d7",
    ),
  ];
  for (relative, expected) in cases {
    let actual = common::sha256_hex(&dir.join(relative));
    assert_eq!(
      actual, expected,
      "sha256 drift on audio artifact {relative}"
    );
  }
}
