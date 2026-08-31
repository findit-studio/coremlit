//! Ground-truth introspection + provenance pins for the CLAP **audio** artifact
//! `clap_audio.mlmodelc`. Every shape/dtype claim comes from loading the real
//! `.mlmodelc` via `coremlit::Model::load` + `.description()`; every SHA comes
//! from the downloaded bytes. Text lives in `tests/clap/text_model_io.rs`.
//!
//! # Artifact
//!
//! Source: [`FinDIT-Studio/clapkit-coreml`](https://huggingface.co/FinDIT-Studio/clapkit-coreml),
//! revision (commit SHA) `02a99c6a8be21da1e9a947499ea503a10c80c4f1` — the current
//! revision, which ships BOTH tiers: the **fp16** tier (converted by T1 from
//! `laion/clap-htsat-unfused` `@ 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`,
//! byte-identical to the original fp16-only publication
//! `97d631f3814e1e46b798a8e88c9aa2e2202fdf67`) and the 2×-smaller **int8** tier.
//! Gitignored, fetched dev-time under `Models/clapkit/` (`CLAPKIT_TEST_MODELS`).
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
//! # Per-file SHA-256 (pinned; EVERY file in each `.mlmodelc`, from
//! `CHECKSUMS.sha256` at revision `02a99c6a8be21da1e9a947499ea503a10c80c4f1`)
//!
//! Both tiers pin all five compiled-model files, so drift in ANY committed
//! artifact byte (not just `model.mil` / `weight.bin`) reds. The fp16 bytes are
//! unchanged from the original `97d631f3…` publication.
//!
//! fp16 (`clap_audio.mlmodelc`):
//!
//! | File | SHA-256 |
//! |---|---|
//! | `analytics/coremldata.bin` | `29fbf161ab063c891080f328de0ee4ed80fdbaef4b5d36c4086c38da582aa7c4` |
//! | `coremldata.bin` | `652c4652d19b4a7e926468e14582442f047389145576827068f6ae47f97ebb3e` |
//! | `metadata.json` | `44ea8733243b41a005a6aa25144fbde165c5e3f80ede79889f2d039da6a65ec4` |
//! | `model.mil` | `1ecf76edf7846153623485a98c0a3d047e660cc68ee10a6d21a39664d309db52` |
//! | `weights/weight.bin` | `723fe6aab7c4af1c671a210a35c289c67763bc6a7532b9df155a0c3fc0c3c9d7` |
//!
//! int8 (`clap_audio_int8.mlmodelc`; the 2×-smaller variant, same I/O contract):
//!
//! | File | SHA-256 |
//! |---|---|
//! | `analytics/coremldata.bin` | `4455479a6d65004fe95d582246675cc9167c0b8fb6e0e673ad9db4009c0443de` |
//! | `coremldata.bin` | `4bee9628bdf8821a391b32a7d23967c3c6712ae72088a318266583f35743ac33` |
//! | `metadata.json` | `c5c466395cfb5a58ae9ed6f44e208492c59c6a1a245919928d08d87a8ffcf964` |
//! | `model.mil` | `1cdbbcc0911e9a9d427119e182dc3efa93d90b7159a614d372d397aff7861bb1` |
//! | `weights/weight.bin` | `b3a37ec5550dcdd6932b314b830275ebcba013748421e1a517760b9afeabafb8` |
//!
//! # Contract (matches the spec table and T1's I/O record)
//!
//! `input_features` fp32 `[1, 1, 1001, 64]` (HF-mel spectrogram) →
//! `audio_embeds` fp32 `[1, 512]` (projection, PRE-L2-norm; clapkit normalizes
//! in Rust). No delta found. The int8 variant shares this contract exactly.

mod common;

use std::collections::BTreeSet;

use coremlit::{
  ComputeUnits, DataType, Model,
  embeddings::clap::{
    audio::{N_MELS, T_FRAMES},
    embedding::EMBEDDING_DIM,
  },
};

/// Hermetic non-vacuity proof for [`common::collect_files_rel`]'s sidecar
/// filter (no staged model needed). On exFAT/FAT/SMB volumes macOS
/// materializes AppleDouble `._*` and `.DS_Store` sidecars inside `.mlmodelc`
/// bundles; discovery must drop EXACTLY those, while every real file —
/// crucially including an unpinned real extra — still reaches the exact-set
/// gate [`common::assert_exact_sha_manifest`] builds on. This proves the
/// filter fixes the false-failure WITHOUT blanket-suppressing genuine extras.
#[test]
fn collect_files_rel_skips_sidecars_but_surfaces_real_extras() {
  let tmp = tempfile::tempdir().expect("create temp dir");
  let bundle = tmp.path().join("clap_audio.mlmodelc");
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

  let mut found = Vec::new();
  common::collect_files_rel(&bundle, "", &mut found);
  let discovered: BTreeSet<String> = found.into_iter().collect();

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
  // breaks the exact-set equality `assert_exact_sha_manifest` performs, and
  // shows up as the sole difference against a set listing only the two real
  // artifacts.
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
  // The EXACT pinned manifest: every file in the `.mlmodelc` (per
  // `CHECKSUMS.sha256` @ 02a99c6a), not just model.mil + weight.bin. The helper
  // enumerates the bundle and set-compares against these keys before hashing, so
  // an ADDED or REMOVED artifact reds as well as any byte drift.
  let cases = [
    (
      "analytics/coremldata.bin",
      "29fbf161ab063c891080f328de0ee4ed80fdbaef4b5d36c4086c38da582aa7c4",
    ),
    (
      "coremldata.bin",
      "652c4652d19b4a7e926468e14582442f047389145576827068f6ae47f97ebb3e",
    ),
    (
      "metadata.json",
      "44ea8733243b41a005a6aa25144fbde165c5e3f80ede79889f2d039da6a65ec4",
    ),
    (
      "model.mil",
      "1ecf76edf7846153623485a98c0a3d047e660cc68ee10a6d21a39664d309db52",
    ),
    (
      "weights/weight.bin",
      "723fe6aab7c4af1c671a210a35c289c67763bc6a7532b9df155a0c3fc0c3c9d7",
    ),
  ];
  common::assert_exact_sha_manifest(&dir, &cases);
}

#[test]
#[ignore = "requires local clapkit int8 models (CLAPKIT_TEST_MODELS)"]
fn clap_audio_int8_io_matches_spec() {
  let model = Model::load(common::audio_model_int8_path(), ComputeUnits::CpuOnly).unwrap();
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
#[ignore = "requires local clapkit int8 models (CLAPKIT_TEST_MODELS)"]
fn clap_audio_int8_artifacts_match_pinned_sha256() {
  let dir = common::audio_model_int8_path();
  // The EXACT pinned manifest for the int8 `.mlmodelc` (per `CHECKSUMS.sha256` @
  // 02a99c6a); the helper enumerates + set-compares before hashing, so a
  // missing/added artifact reds too.
  let cases = [
    (
      "analytics/coremldata.bin",
      "4455479a6d65004fe95d582246675cc9167c0b8fb6e0e673ad9db4009c0443de",
    ),
    (
      "coremldata.bin",
      "4bee9628bdf8821a391b32a7d23967c3c6712ae72088a318266583f35743ac33",
    ),
    (
      "metadata.json",
      "c5c466395cfb5a58ae9ed6f44e208492c59c6a1a245919928d08d87a8ffcf964",
    ),
    (
      "model.mil",
      "1cdbbcc0911e9a9d427119e182dc3efa93d90b7159a614d372d397aff7861bb1",
    ),
    (
      "weights/weight.bin",
      "b3a37ec5550dcdd6932b314b830275ebcba013748421e1a517760b9afeabafb8",
    ),
  ];
  common::assert_exact_sha_manifest(&dir, &cases);
}
