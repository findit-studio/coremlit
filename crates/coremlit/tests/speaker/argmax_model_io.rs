//! Ground-truth introspection of argmax's `speakerkit-coreml` CoreML
//! artifacts (`argmaxinc/speakerkit-coreml` on HuggingFace) — the SECOND
//! `ModelSource` this crate targets (Task 1's `ModelSource` trait /
//! `FluidAudioSource`; Task 3's `ArgmaxSource` codes against the contracts
//! pinned here). Every shape/dtype assertion below comes from loading the
//! real `.mlmodelc` via `coremlit::Model::load` + `.description()` — the
//! plan brief's expected-contract list (`.superpowers/sdd/task-2-brief.md`,
//! itself sourced from HF metadata, not from running the models) is a
//! HYPOTHESIS; reality wins, and this file IS that reality check. See
//! "Spec-vs-reality deltas" below for every place introspection added
//! information the brief didn't pin down.
//!
//! # Acquisition (`Models/argmax-speakerkit/`, gitignored, fetched dev-time)
//!
//! ```text
//! hf download argmaxinc/speakerkit-coreml --local-dir Models/argmax-speakerkit
//! ```
//!
//! Public, un-gated, ~32 MB on disk (45 files). Every artifact ships
//! pre-compiled as `.mlmodelc` — `find Models/argmax-speakerkit -iname
//! '*.mlpackage'` returns nothing anywhere in the repo, so no `xcrun coremlc
//! compile` step was needed (unlike a repo that ships raw `.mlpackage`
//! sources).
//!
//! Revision (git commit SHA of the HF repo's `main` at acquisition time),
//! cross-checked two independent ways — the huggingface_hub local cache's
//! per-file `.metadata` sidecars
//! (`.cache/huggingface/download/**/*.metadata`, line 1 of each) and `git
//! ls-remote https://huggingface.co/argmaxinc/speakerkit-coreml`, which
//! reports it as both `HEAD` and `refs/heads/main` — both agree on:
//!
//! ```text
//! 86ec9c929b52208b6656eb6a6361ed0d822a1f78
//! ```
//!
//! (See [`ARGMAX_REVISION`]. Unlike this workspace's `MODELS_LOCK`, which
//! deliberately pins the moving `revision = "main"` for CI's cache key —
//! see that file's own header comment — this is a real, immutable commit
//! SHA captured at acquisition time, not re-verified live by any test here:
//! doing so would require network access, which these tests deliberately
//! don't have.)
//!
//! # Artifacts
//!
//! | Bundle | Category | Variant | Role |
//! |---|---|---|---|
//! | `speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc` | segmentation | W32A32 | **pinned** |
//! | `speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc` | segmentation | W8A16 | **pinned** |
//! | `speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc` | embedding frontend (fbank) | W16A16 | **pinned** |
//! | `speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedder.mlmodelc` | embedding backend | W16A16 | **pinned** |
//! | `speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedderPreprocessor.mlmodelc` | embedding frontend (fbank) | W8A16 | **pinned** |
//! | `speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc` | embedding backend | W8A16 | **pinned** |
//! | `speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc` | clustering input transform | W32A32 | recorded, out of scope |
//!
//! "W32A32"/"W16A16"/"W8A16" are argmax's own directory names (weight-bits ×
//! activation-bits). The brief calls the fp32-ish baseline variants
//! "W32A32"/"W16A16" and the quantized ones "W8A16"; see delta 1 below for
//! what that does and doesn't say about the actual `MLMultiArray` I/O
//! dtype. Clustering (PLDA/VBx) stays in `dia`, per the design spec's
//! non-goals — `PldaProjector` is introspected below purely for
//! completeness, mirroring `tests/model_io.rs`'s PLDA-recording precedent
//! (this task's own brief names no such artifact).
//!
//! # Licenses
//!
//! The repo-level HuggingFace frontmatter
//! (`Models/argmax-speakerkit/README.md`) declares no `license:` key at all
//! — undeclared, verified. Each category/variant directory ships its own
//! `README.txt` pointing to the ORIGINAL upstream weights' license instead
//! of a license for argmax's own conversion/compilation into CoreML:
//!
//! - segmenter (pyannote/segmentation-3.0): <https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE>
//! - embedder (wespeaker): <https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md#model-license>
//! - clusterer (VBx): <https://github.com/BUTSpeechFIT/VBx?tab=readme-ov-file#license>
//!
//! Recorded here for provenance only — no license text is added and no
//! legal conclusion is drawn; the README task (T6) documents the situation
//! for the crate as a whole.
//!
//! # Spec-vs-reality deltas
//!
//! 1. **Every pinned input/output dtype introspects as F16, not F32** —
//!    including on the "W32A32" segmenter/clusterer variants. The plan
//!    brief's expected-contract list (shapes only) didn't specify a dtype,
//!    so this isn't a shape mismatch, but it's load-bearing ground truth
//!    Task 3 needs for buffer allocation: NEITHER variant of ANY of these
//!    models exposes an f32 `MLMultiArray` boundary. "W32A32"/"W8A16"
//!    describe internal weight/activation storage precision (confirmed
//!    distinct per each bundle's own `metadata.json` `storagePrecision`
//!    field at acquisition time: `"Mixed (Float16, Float32)"` for W32A32
//!    vs. `"Mixed (Float16, Palettized (8 bits))"` for W8A16 on the
//!    segmenter), not the external `MLFeature` type, which is F16 on every
//!    input and output of every one of the seven artifacts.
//! 2. **Every shape in the plan brief's expected-contract list matches
//!    exactly** — `waveform [480000]`, `speaker_probs`/`speaker_ids
//!    [21,589,3]`, `speaker_activity [21,3]`, `overlapped_speaker_activity
//!    [21,589]`, `voice_activity [1767]`, `sliding_window_waveform
//!    [21,1,160000]`, `waveforms [1,480000]` → `preprocessor_output_1
//!    [1,2998,80]`, `preprocessor_output_1 [1,2998,80]` + `speaker_masks
//!    [1,64,1767]` → `speaker_embeddings [1,64,256]`. No shape delta to
//!    report.
//! 3. `SpeakerEmbedderPreprocessor.mlmodelc` is BYTE-IDENTICAL (sha256 of
//!    all 5 constituent files — `diff -rq` confirms too) between the
//!    `W16A16` and `W8A16` embedder directories: argmax doesn't quantize
//!    the fbank frontend differently per variant, only the embedder
//!    backend. Not called out in the plan brief; each variant is still
//!    introspected independently below anyway, mirroring
//!    `tests/model_io.rs`'s
//!    `wespeaker_v2_and_wespeaker_int8_are_byte_identical` precedent of
//!    re-verifying via `Model::load` rather than assuming from
//!    byte-identity alone.
//! 4. `PldaProjector.mlmodelc` (`speaker_clusterer`, out of scope): input
//!    `embeddings [1, 64, 256]`, output `plda_embeddings [1, 64, 128]` —
//!    recorded for completeness (mirroring `tests/model_io.rs`'s
//!    PLDA-recording precedent) with no expected contract to check it
//!    against, so this is recorded, not a delta.
//!    Note the name is `plda_embeddings`, not the FluidAudio-side PLDA
//!    artifact's `plda_features` (`tests/model_io.rs`'s
//!    `plda_io_recorded_out_of_scope`) — the two sources don't share a
//!    contract here either, consistent with them being unrelated
//!    conversions (pyannote-v4 vs. FluidAudio's own).
//! 5. None of the seven artifacts declare any shape flexibility
//!    (`hasShapeFlexibility: "0"` on every input/output in every
//!    `metadata.json`) — unlike some of `tests/model_io.rs`'s
//!    FluidAudio-side candidates, nothing here needs
//!    default-of-a-range-constraint handling.
//!
//! # Env / path convention
//!
//! `ARGMAX_TEST_MODELS` (default `Models/argmax-speakerkit`), resolved by
//! `common::argmax_models_dir()` — sibling to `SPEAKERKIT_TEST_MODELS`/
//! `common::models_dir()` (the FluidAudio-sourced artifacts `tests/model_io.rs`
//! pins). Two distinct env vars/defaults because Task 3 loads both
//! `ModelSource`s side by side and each needs its own independently
//! overridable path.

mod common;

use std::{
  collections::BTreeSet,
  path::{Path, PathBuf},
};

use coremlit::{ComputeUnits, DataType, Model};

/// HF commit SHA this file's shapes/dtypes/hashes were introspected
/// against (`argmaxinc/speakerkit-coreml`, `main` branch at acquisition
/// time). See the module doc's Acquisition section for how this was
/// cross-checked.
const ARGMAX_REVISION: &str = "86ec9c929b52208b6656eb6a6361ed0d822a1f78";

/// SHA-256 of every real file inside the seven pinned `.mlmodelc` bundles
/// (`model.mil`, `coremldata.bin`, `metadata.json`, `weights/weight.bin`,
/// `analytics/coremldata.bin` — five per bundle, 35 total), paths relative
/// to `common::argmax_models_dir()`. Deliberately scoped to the `.mlmodelc`
/// bundles themselves, not the repo's top-level
/// `README.md`/`.gitattributes`/per-directory `README.txt`: those aren't
/// model artifacts and can change independently of the pinned contracts
/// (e.g. a documentation edit upstream) without invalidating this pin.
///
/// Recorded via `shasum -a 256` over the freshly downloaded tree at
/// revision [`ARGMAX_REVISION`]; the two `weights/weight.bin` entries this
/// crate directly cares about (`SpeakerSegmenter` W32A32/W8A16) were
/// additionally cross-checked against the huggingface_hub cache's own etag
/// sidecar (`*.metadata` line 2, itself the LFS `sha256:` pointer content)
/// and matched exactly.
#[rustfmt::skip]
const ARTIFACT_SHA256: &[(&str, &str)] = &[
    ("speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc/analytics/coremldata.bin", "3e13c8f4df77ea27cbbcdd6d083c63f5e7b3f32566cc5bf223fab92d40b81b8b"),
    ("speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc/coremldata.bin", "6f6820ccf221d4cc7c107101d0ae4d716eb224e4fdffaf8f7ca36af70ae64c40"),
    ("speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc/metadata.json", "acc86b4a4f542d8e7eaf84fc290fe72bd8629b0ffcc16e5b1a9e13d777abd9d8"),
    ("speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc/model.mil", "209e641bf4d9c3868c9dc43ab8705094997917eb2df5f5d8c8fda09fed54b1d4"),
    ("speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc/weights/weight.bin", "a1dbbb651a0a67fcfe5334672f459df090fa960917a6ee3a5423245a7ab92ced"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedder.mlmodelc/analytics/coremldata.bin", "17d567af44a172e09251880ccdb8bca4431a2ebdeaf0167fb033dc5d03654c31"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedder.mlmodelc/coremldata.bin", "a45c627a63eb0a24cfbdb5baf7bca25b6755170841cc62c026f1522fedcdafb6"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedder.mlmodelc/metadata.json", "040de100400c44b869a822a0aca8b8b49da006a3f981d90d9766af9ba11e882b"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedder.mlmodelc/model.mil", "7148d95e8c1fc180ad00004a2b5bee244ed8bdadb87be76b5d64b6a598349fc9"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedder.mlmodelc/weights/weight.bin", "6dba18a57a81b1e872802ca4def29541bb7900ccff430d9b2040092cadd7d688"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc/analytics/coremldata.bin", "ce9bef9fb3125a5401300b5c5998c5d8f211094692cae780645d3e2757410f2c"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc/coremldata.bin", "b4ebd0b9ce5a84768672663aff426eb19f9648d4b9f74286f0e19fc753ad76ba"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc/metadata.json", "789f81c17dc04d469611d253684e534565fb4a008e54c722b925f1608bf87fce"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc/model.mil", "42e552ebd7efb12ea813eceb474018dd0f46168e84ad3a1c54945bfc47be7a82"),
    ("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc/weights/weight.bin", "5f2c284bd22f1f7ab76901c1c6e57f82d4ebbf057fa0b924aad057f124f77a89"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc/analytics/coremldata.bin", "ba8405dfc9b9348ade705e052888b4bdc7fb8d079ef3ff71108a5f692d0209f2"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc/coremldata.bin", "1597d6c037ac52436b5c2e1abc47e6c68483c19eeac75267dfb8795a78ec07c5"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc/metadata.json", "29ea3421161c8344f6ea95db9b472217638a869f686f2494d10e5d11f11f4cda"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc/model.mil", "5eee9f6aa380aef88fee604d75c5deaa23adc83c9480cb8f6dedc72803973e77"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc/weights/weight.bin", "a02861969f47cf3a67e3b0d276e54b3c8bc3a6e43d40d77d1cccbd57da0e5795"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedderPreprocessor.mlmodelc/analytics/coremldata.bin", "ce9bef9fb3125a5401300b5c5998c5d8f211094692cae780645d3e2757410f2c"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedderPreprocessor.mlmodelc/coremldata.bin", "b4ebd0b9ce5a84768672663aff426eb19f9648d4b9f74286f0e19fc753ad76ba"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedderPreprocessor.mlmodelc/metadata.json", "789f81c17dc04d469611d253684e534565fb4a008e54c722b925f1608bf87fce"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedderPreprocessor.mlmodelc/model.mil", "42e552ebd7efb12ea813eceb474018dd0f46168e84ad3a1c54945bfc47be7a82"),
    ("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedderPreprocessor.mlmodelc/weights/weight.bin", "5f2c284bd22f1f7ab76901c1c6e57f82d4ebbf057fa0b924aad057f124f77a89"),
    ("speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc/analytics/coremldata.bin", "42e2434809899abce6a3947f4f3b1365af7c8d3762e4c4bfc0df886f1dca8347"),
    ("speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc/coremldata.bin", "ed53832ecc7af1c0eb6bc1bc8a475c369d6103ea08f21a40302a57f06966c6c8"),
    ("speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc/metadata.json", "3af014cc496b87a42cf315f041a554f38e0b56d910a2bf8a40e0f1ec535ab257"),
    ("speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc/model.mil", "25c39c5c59ffe1a5d244389aeaa8c195636b86239f8213496385d21c0efc2c56"),
    ("speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc/weights/weight.bin", "1584619d1180ef89807b66c2c96605720365c02d4fbdcc9be02bbad91d188128"),
    ("speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc/analytics/coremldata.bin", "40637aa0cb2a073bc303c7ca9ee79da35fa81d2cad1ead180e93b134005b95de"),
    ("speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc/coremldata.bin", "6c356ed983b2a3332ce51299ca0f9747a35cb6c2a67b0ac24c69dbef3f989634"),
    ("speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc/metadata.json", "2fd6aaf6beb17b3758f5d0c5b2cf5feeacb0cc0c9267dbcd7536b247b1a5860e"),
    ("speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc/model.mil", "423c358915acab0d440c99f5162c17456936c2c02f7394b05ab226b9a34c122a"),
    ("speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc/weights/weight.bin", "75ff1725ef4e58dacf9176466ec274a8a13a6132c296d6b571fb78ddad5455c4"),
];

/// Joins a bundle-relative path onto [`common::argmax_models_dir`].
fn artifact_path(rel: &str) -> PathBuf {
  common::argmax_models_dir().join(rel)
}

/// Recursively collects every FILE under `dir` as a path relative to `root`,
/// inserting each into `out` with `/` separators to match [`ARTIFACT_SHA256`]'s
/// keys. Used to enumerate a `.mlmodelc` bundle's actual tree so it can be
/// checked against the pinned key set (no unpinned extras, none missing).
fn collect_files_rel(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
  for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
  {
    let entry = entry.expect("read dir entry");
    let path = entry.path();
    if entry.file_type().expect("file type").is_dir() {
      collect_files_rel(root, &path, out);
    } else {
      let rel = path
        .strip_prefix(root)
        .expect("walked path is under root")
        .to_str()
        .expect("utf-8 path")
        .to_string();
      out.insert(rel);
    }
  }
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn argmax_artifact_bytes_match_pinned_sha256() {
  assert_eq!(
    ARTIFACT_SHA256.len(),
    35,
    "expected 5 files (model.mil, coremldata.bin, metadata.json, \
     weights/weight.bin, analytics/coremldata.bin) across 7 .mlmodelc \
     bundles; table shape changed, update this assertion alongside it"
  );

  // BEFORE hashing: enumerate each of the seven .mlmodelc bundles' real file
  // trees and assert the discovered relative-path set EQUALS the pinned key set
  // exactly — no unpinned extra (a file slipped into a bundle) and none missing
  // (L3). Hashing only the 35 hard-coded keys never notices a 36th file. The
  // bundle roots are DERIVED from the pinned keys, so the walk is scoped to
  // exactly the seven real bundles: the sibling
  // `.cache/huggingface/download/**/*.mlmodelc/*.metadata` download-cache mirror
  // (not a model artifact, and not under any pinned bundle root) is never
  // entered.
  let pinned: BTreeSet<String> = ARTIFACT_SHA256
    .iter()
    .map(|(rel, _)| (*rel).to_string())
    .collect();
  let bundles: BTreeSet<&str> = ARTIFACT_SHA256
    .iter()
    .map(|(rel, _)| {
      let end = rel
        .find(".mlmodelc")
        .expect("every pinned key is inside a .mlmodelc bundle")
        + ".mlmodelc".len();
      &rel[..end]
    })
    .collect();
  assert_eq!(
    bundles.len(),
    7,
    "expected 7 distinct .mlmodelc bundles across the pinned table; its bundle span changed"
  );
  let root = common::argmax_models_dir();
  let mut discovered: BTreeSet<String> = BTreeSet::new();
  for bundle in &bundles {
    collect_files_rel(&root, &root.join(bundle), &mut discovered);
  }
  assert_eq!(
    discovered,
    pinned,
    "argmax .mlmodelc bundle trees (revision {ARGMAX_REVISION}) do not match the pinned \
     SHA-256 table's key set -- unpinned extras: {:?}, pinned-but-absent: {:?}. A file was \
     added to or removed from a bundle; re-introspect and re-pin.",
    discovered.difference(&pinned).collect::<Vec<_>>(),
    pinned.difference(&discovered).collect::<Vec<_>>(),
  );

  for (rel_path, expected) in ARTIFACT_SHA256 {
    let path = artifact_path(rel_path);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
      panic!(
        "read {} (pinned at argmax revision {ARGMAX_REVISION}): {e}",
        path.display()
      )
    });
    let actual = common::sha256_hex(&bytes);
    assert_eq!(
      &actual,
      expected,
      "{}: sha256 mismatch against the pin recorded at argmax revision \
       {ARGMAX_REVISION} (expected {expected}, got {actual}) -- the \
       downloaded bytes changed; re-run the introspection tests in this \
       file and re-pin both the hash and any contract that moved",
      path.display()
    );
  }
}

#[test]
fn sha256_hex_matches_known_vector() {
  // `common::sha256_hex` is now a thin wrapper over the upstream-tested
  // `sha2` crate (see its doc comment), so re-proving SHA-256 itself with a
  // full FIPS 180-4 / RFC 6234 known-answer-vector battery would be
  // redundant -- that's `sha2`'s own test suite's job. One vector is kept as
  // a hermetic wiring smoke test (digest + hex-encode aren't swapped or
  // mis-cased), consistent with this file's practice of not just asserting
  // but independently cross-checking (here, against the system
  // `shasum -a 256`, matching this constant's pre-existing value). This test
  // is intentionally NOT `#[ignore]`d: it needs no models, so it runs in the
  // normal `cargo test` gate, unlike `argmax_artifact_bytes_match_pinned_sha256`
  // above.
  assert_eq!(
    common::sha256_hex(b"abc"),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  );
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn speaker_segmenter_w32a32_io_matches_spec() {
  let path = artifact_path("speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let waveform = description.input("waveform").expect("waveform input");
  assert_eq!(waveform.shape(), &[480_000]);
  assert_eq!(waveform.data_type(), Some(DataType::F16));

  let speaker_probs = description
    .output("speaker_probs")
    .expect("speaker_probs output");
  assert_eq!(speaker_probs.shape(), &[21, 589, 3]);
  assert_eq!(speaker_probs.data_type(), Some(DataType::F16));

  let speaker_ids = description
    .output("speaker_ids")
    .expect("speaker_ids output");
  assert_eq!(speaker_ids.shape(), &[21, 589, 3]);
  assert_eq!(speaker_ids.data_type(), Some(DataType::F16));

  let speaker_activity = description
    .output("speaker_activity")
    .expect("speaker_activity output");
  assert_eq!(speaker_activity.shape(), &[21, 3]);
  assert_eq!(speaker_activity.data_type(), Some(DataType::F16));

  let overlapped = description
    .output("overlapped_speaker_activity")
    .expect("overlapped_speaker_activity output");
  assert_eq!(overlapped.shape(), &[21, 589]);
  assert_eq!(overlapped.data_type(), Some(DataType::F16));

  let voice_activity = description
    .output("voice_activity")
    .expect("voice_activity output");
  assert_eq!(voice_activity.shape(), &[1767]);
  assert_eq!(voice_activity.data_type(), Some(DataType::F16));

  let sliding_window = description
    .output("sliding_window_waveform")
    .expect("sliding_window_waveform output");
  assert_eq!(sliding_window.shape(), &[21, 1, 160_000]);
  assert_eq!(sliding_window.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn speaker_segmenter_w8a16_io_matches_spec() {
  // Same contract as the W32A32 variant (module doc delta 1: the W8A16
  // suffix describes internal weight/activation storage, not the external
  // MLFeature dtype, which is F16 on both).
  let path = artifact_path("speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let waveform = description.input("waveform").expect("waveform input");
  assert_eq!(waveform.shape(), &[480_000]);
  assert_eq!(waveform.data_type(), Some(DataType::F16));

  let speaker_probs = description
    .output("speaker_probs")
    .expect("speaker_probs output");
  assert_eq!(speaker_probs.shape(), &[21, 589, 3]);
  assert_eq!(speaker_probs.data_type(), Some(DataType::F16));

  let speaker_ids = description
    .output("speaker_ids")
    .expect("speaker_ids output");
  assert_eq!(speaker_ids.shape(), &[21, 589, 3]);
  assert_eq!(speaker_ids.data_type(), Some(DataType::F16));

  let speaker_activity = description
    .output("speaker_activity")
    .expect("speaker_activity output");
  assert_eq!(speaker_activity.shape(), &[21, 3]);
  assert_eq!(speaker_activity.data_type(), Some(DataType::F16));

  let overlapped = description
    .output("overlapped_speaker_activity")
    .expect("overlapped_speaker_activity output");
  assert_eq!(overlapped.shape(), &[21, 589]);
  assert_eq!(overlapped.data_type(), Some(DataType::F16));

  let voice_activity = description
    .output("voice_activity")
    .expect("voice_activity output");
  assert_eq!(voice_activity.shape(), &[1767]);
  assert_eq!(voice_activity.data_type(), Some(DataType::F16));

  let sliding_window = description
    .output("sliding_window_waveform")
    .expect("sliding_window_waveform output");
  assert_eq!(sliding_window.shape(), &[21, 1, 160_000]);
  assert_eq!(sliding_window.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn speaker_embedder_preprocessor_w16a16_io_matches_spec() {
  let path =
    artifact_path("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedderPreprocessor.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let waveforms = description.input("waveforms").expect("waveforms input");
  assert_eq!(waveforms.shape(), &[1, 480_000]);
  assert_eq!(waveforms.data_type(), Some(DataType::F16));

  let output = description
    .output("preprocessor_output_1")
    .expect("preprocessor_output_1 output");
  assert_eq!(output.shape(), &[1, 2998, 80]);
  assert_eq!(output.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn speaker_embedder_w16a16_io_matches_spec() {
  let path = artifact_path("speaker_embedder/pyannote-v3/W16A16/SpeakerEmbedder.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let preprocessor_output = description
    .input("preprocessor_output_1")
    .expect("preprocessor_output_1 input");
  assert_eq!(preprocessor_output.shape(), &[1, 2998, 80]);
  assert_eq!(preprocessor_output.data_type(), Some(DataType::F16));

  let masks = description
    .input("speaker_masks")
    .expect("speaker_masks input");
  assert_eq!(masks.shape(), &[1, 64, 1767]);
  assert_eq!(masks.data_type(), Some(DataType::F16));

  let embeddings = description
    .output("speaker_embeddings")
    .expect("speaker_embeddings output");
  assert_eq!(embeddings.shape(), &[1, 64, 256]);
  assert_eq!(embeddings.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn speaker_embedder_preprocessor_w8a16_io_matches_spec() {
  // Module doc delta 3: byte-identical to the W16A16 preprocessor above,
  // but re-verified independently via `Model::load` rather than assumed.
  let path =
    artifact_path("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedderPreprocessor.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let waveforms = description.input("waveforms").expect("waveforms input");
  assert_eq!(waveforms.shape(), &[1, 480_000]);
  assert_eq!(waveforms.data_type(), Some(DataType::F16));

  let output = description
    .output("preprocessor_output_1")
    .expect("preprocessor_output_1 output");
  assert_eq!(output.shape(), &[1, 2998, 80]);
  assert_eq!(output.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn speaker_embedder_w8a16_io_matches_spec() {
  // Same contract as the W16A16 variant (module doc delta 1).
  let path = artifact_path("speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let preprocessor_output = description
    .input("preprocessor_output_1")
    .expect("preprocessor_output_1 input");
  assert_eq!(preprocessor_output.shape(), &[1, 2998, 80]);
  assert_eq!(preprocessor_output.data_type(), Some(DataType::F16));

  let masks = description
    .input("speaker_masks")
    .expect("speaker_masks input");
  assert_eq!(masks.shape(), &[1, 64, 1767]);
  assert_eq!(masks.data_type(), Some(DataType::F16));

  let embeddings = description
    .output("speaker_embeddings")
    .expect("speaker_embeddings output");
  assert_eq!(embeddings.shape(), &[1, 64, 256]);
  assert_eq!(embeddings.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local argmax speakerkit models (ARGMAX_TEST_MODELS)"]
fn plda_projector_w32a32_io_recorded_out_of_scope() {
  // Out of scope per the design spec's non-goals (clustering/VBx/PLDA stay
  // in `dia`) -- introspected anyway for completeness, mirroring
  // `tests/model_io.rs`'s PLDA-recording precedent (this task's own brief
  // names no such artifact). Module doc delta 4: no expected contract was
  // given to check this against.
  let path = artifact_path("speaker_clusterer/pyannote-v4/W32A32/PldaProjector.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let embeddings = description.input("embeddings").expect("embeddings input");
  assert_eq!(embeddings.shape(), &[1, 64, 256]);
  assert_eq!(embeddings.data_type(), Some(DataType::F16));

  let plda_embeddings = description
    .output("plda_embeddings")
    .expect("plda_embeddings output");
  assert_eq!(plda_embeddings.shape(), &[1, 64, 128]);
  assert_eq!(plda_embeddings.data_type(), Some(DataType::F16));
}
