//! Ground-truth introspection of every candidate segmentation and embedding
//! artifact named in the design spec
//! (`docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md` §4, §9
//! open item). Every value below comes from loading the real `.mlmodelc` via
//! `coremlit::Model::load` + `.description()` — the spec's table is a
//! HYPOTHESIS; reality wins, and every place it differs is marked `SPEC
//! DELTA`. Feeds Task 2 (`SegmentModel`) and later tasks (`EmbedModel`,
//! `Extractor`).
//!
//! # Artifacts (`Models/speakerkit/`, gitignored, fetched dev-time)
//!
//! | File | Role | Targeted? |
//! |---|---|---|
//! | `pyannote_segmentation.mlmodelc` | segmentation | **yes** — see DECISION |
//! | `Segmentation.mlmodelc` | segmentation, alt conversion | no |
//! | `wespeaker_v2.mlmodelc` | embedding | **yes** — see DECISION |
//! | `wespeaker_int8.mlmodelc` | embedding, byte-identical to `wespeaker_v2` | no (same file) |
//! | `wespeaker.mlmodelc` | embedding, fp32, contract-equal | no |
//! | `FBank.mlmodelc` | embedding frontend, split-pipeline alt | no |
//! | `Embedding.mlmodelc` | embedding backend, split-pipeline alt | no |
//!
//! The repo also ships `PLDA.mlmodelc` / `PldaRho.mlmodelc`. Neither is a
//! candidate and neither is introspected here: clustering — and with it the
//! PLDA projection — stays in `diaric` (spec §3 non-goal), which projects in
//! f64 on the host. `coremlit::audio::speaker::extract`'s `into_offline_input`
//! records why the CoreML graphs are unusable even if that ever changed.
//!
//! ## Retired coverage: `plda_io_recorded_out_of_scope` (a deliberate loss)
//!
//! `PLDA.mlmodelc` once had a row in the table above and an `#[ignore]`d
//! introspection test, `plda_io_recorded_out_of_scope`. Both were DELETED, not
//! relocated, and deleting the test did cost coverage that nothing else
//! provides. "No coverage was lost" would be false, so here is the ledger:
//!
//! - **What it uniquely asserted:** that `PLDA.mlmodelc` loads at all
//!   (`Model::load`, `CpuOnly`), and that its description carries the input
//!   `embeddings [1, 256]` F32 and the output `plda_features [1, 128]` F32.
//! - **What still touches the artifact, and how far that reaches:** only
//!   `tests/fp16_guards.rs`. Its sweep requires the bundle to be present with a
//!   parseable `model.mil` and hard-fails if the pinned guard sites change or
//!   vanish — so deletion of the artifact and drift in its numerical guards are
//!   still caught. But it reads the MIL as TEXT: it never calls `Model::load`,
//!   so it says nothing about CoreML loadability, and it never reads the model
//!   description, so it says nothing about feature names, shapes or dtypes.
//! - **So the loadability and schema assertions are simply gone**, and nothing
//!   replaces them. A re-conversion that reshaped the I/O while leaving the
//!   guard sites alone would now pass unnoticed.
//! - **Why that is the intended trade rather than an oversight:** the artifact
//!   is on no runtime path — nothing in this crate calls `Model::load` on it,
//!   and nothing is planned to, because the projection is
//!   `diaric::plda::PldaTransform` in f64 on the host. The test pinned the I/O
//!   contract of a graph this crate will never consume, and being `#[ignore]`d
//!   it ran only when someone staged the models and asked for it by name.
//!   Retiring it with this record beats keeping a check whose presence implies
//!   the graph is still a live candidate; the one scenario in which the lost
//!   assertions would matter — the graph becoming a candidate again — is also
//!   the one scenario in which it would be re-introspected from scratch anyway.
//! - **If it ever does become one**, restore exactly the assertions listed
//!   above; they are spelled out here so the restoration needs no `git log`
//!   archaeology.
//!
//! # Segmentation provenance (the fp16-safe re-conversion, issue #15)
//!
//! `pyannote_segmentation.mlmodelc` — and ONLY that artifact — comes from
//! <https://huggingface.co/FinDIT-Studio/speakerkit-coreml>, revision (commit
//! SHA) `3db69988bf2de12bab250614d6ac2b03d35132a2`. Every one of its files is
//! byte-pinned by [`fp16_safe_segmentation_matches_pinned_sha256`]. Every other
//! artifact in `Models/speakerkit/` is still FluidInference's
//! `speaker-diarization-coreml` conversion.
//!
//! ```text
//! hf download FinDIT-Studio/speakerkit-coreml \
//!   --revision 3db69988bf2de12bab250614d6ac2b03d35132a2 \
//!   --include 'pyannote_segmentation.mlmodelc/*' --local-dir Models/speakerkit
//! ```
//!
//! It is a **re-conversion of the same upstream weights**, not a different
//! model. FluidInference's fp16 conversion ended `softmax` →
//! `log(epsilon = 0x0p+0)`; `0` is below fp16's smallest subnormal (`2^-24`),
//! so wherever the graph computes in fp16 the guard is inert and an underflowed
//! softmax reaches an unguarded `log(0)`. Measured on `09_mrbeast_dollar_date`,
//! 1033 chunks, the shipping `ComputeUnits::All` placement: minimum `segments`
//! value **−45440.0** against **−32.31** on `CpuOnly`. The re-conversion emits
//! the fused `reduce_log_sum_exp` → `sub` form — no `log` op at all, nothing to
//! saturate — and the same measurement gives **−31.80** on `All`.
//!
//! **What this swap does NOT do.** It is end-to-end inert on the whole
//! multi-speaker DER corpus: with only this artifact changed, every gated
//! number on clips 06 / 09 / 10 / 14 is bit-identical to the pre-swap
//! measurement, including clip 09's 5-of-8-speaker collapse
//! (`parity_shipping_der`'s known-defect pin, unchanged). The clip-09 defect is
//! a segmentation-conversion defect — the model cross-product attributes it
//! there, and the embedder is exonerated at cosine 1.000000 — but its mechanism
//! is the fp16 conversion's ordinary tail precision (1091 of 608 437 powerset
//! argmax frames flip against dia-ort's fp32 ONNX), not the vanished epsilon,
//! and the re-conversion is still fp16, so it does not move it. The swap
//! removes a real, silent, four-orders-of-magnitude corruption on the default
//! placement; it does not repair clip 09.
//!
//! The published re-conversions of `wespeaker`/`wespeaker_int8` are
//! deliberately NOT adopted: the int8 one is also a re-palettization, and it
//! moves clip 14's shipping ANE arm from 0.8178 % to 1.4860 % DER, past
//! `parity_shipping_der`'s ±1 pp bound (isolated by swapping one artifact at a
//! time; see issue #15). Their `fp16_guards` pins therefore stand.
//!
//! # Licenses (`Models/speakerkit/README.md`)
//!
//! The repo's HuggingFace frontmatter declares `license: cc-by-4.0` for the
//! model repo as a whole; the body clarifies "the SDK itself is Apache 2.0,
//! but the parent model from Pyannote is `cc-by-4.0`" ("SDK" = FluidAudio's
//! conversion tooling, not the weights this crate loads). The newer
//! "community-1" conversion set (`Segmentation`/`FBank`/`Embedding`/`PLDA`,
//! see DECISION below) additionally self-declares `"license": "CC-BY-4.0"`
//! inside its own `metadata.json`, confirming the same terms independently.
//! CC-BY-4.0 requires attribution; the README's Citations section gives the
//! required BibTeX: segmentation model (Plaquet & Bredin, "Powerset
//! multi-class cross entropy loss for neural speaker diarization",
//! INTERSPEECH 2023), speaker embedding model (Wang et al., "Wespeaker: A
//! research and production oriented speaker embedding learning toolkit",
//! ICASSP 2023), and speaker clustering / VBx (Landini et al., "Bayesian
//! HMM clustering of x-vector sequences (VBx) in speaker diarization",
//! Computer Speech & Language 2022) — the last is irrelevant to speakerkit
//! (clustering stays in `diaric`, spec §3) but ships in the same README and is
//! reproduced here for completeness.
//!
//! # DECISION
//!
//! - **Segmentation: `pyannote_segmentation.mlmodelc`.** The two candidates
//!   are NOT contract-equal — the plan brief's stated tiebreaker condition
//!   ("pick pyannote_segmentation if contract-equal") does not actually
//!   hold, see `segmentation_alt_io_recorded_not_targeted` below. It is
//!   chosen anyway because its single-chunk, fixed-shape `segments` output
//!   (per-frame powerset **log-probabilities** — the graph's tail is
//!   `reduce_log_sum_exp` → `sub`, see `crate::segment`'s module doc) matches both the
//!   spec's pinned contract (§4 table) and the `SegmentModel::infer`
//!   single-chunk API (§5) exactly, and it is FluidAudio's shipping name —
//!   the brief's fallback tiebreaker.
//! - **Embedding: `wespeaker_v2.mlmodelc`.** Verified byte-identical
//!   (`diff -rq`, sha256 of `model.mil` and `weights/weight.bin`, at plan
//!   time) to `wespeaker_int8.mlmodelc` — "v2" is an alias for the
//!   int8-palettized model, not a distinct fp32 architecture; see
//!   `wespeaker_v2_and_wespeaker_int8_are_byte_identical` below.
//!   `wespeaker.mlmodelc` is contract-equal but ships uncompressed fp32
//!   weights (27 MB vs 6.9 MB, `du -sh */weights`).
//!
//!   **The rationale recorded here until 2026-07-26 was false.** It read
//!   "the smaller int8 footprint better serves the issue's ANE uplift
//!   targets". There is no uplift, and `parity_shipping_der.rs`'s
//!   `shipping_embedder_cost_int8_vs_fp32` measures it directly. On 120 s of
//!   `10_mrbeast_clean_water`, warm, one config at a time: on the shipping
//!   `All` placement int8 EXTRACTS SLOWER than fp32 (4.92 s vs 4.41 s,
//!   24.4× vs 27.2× realtime); on `CpuOnly` it is faster (25.03 s vs
//!   28.89 s). No consistent speed advantage in either direction, and the
//!   sign is negative on the placement we ship. Palettization buys 21.5 MB of
//!   on-disk footprint (8.0 MB vs 29.4 MB, 3.7×) and nothing else. Reasoning
//!   from "smaller ⇒ faster on the ANE" without measuring is what produced
//!   the wrong claim.
//!
//!   The DECISION is still correct, for a different reason: int8 does not
//!   move the CLUSTERING decision. On `10_mrbeast_clean_water` — the clip
//!   that exposed the argmax source's spurious 8th speaker — the
//!   precision-isolated int8 arm clusters identically to fp32 (0.0000 %
//!   standard-collar DER, zero confusion) while carrying a *worse*
//!   embedding cosine (~0.90-0.92) than argmax's ~0.94, and no clip in the
//!   gated set has ever shown int8 and fp32 disagreeing on the speaker
//!   count. That is the noise-vs-warp distinction: quantization scatter is
//!   roughly isotropic and survives the frozen community-1 LDA+PLDA basis,
//!   whereas argmax's front-end change is a systematic rotation that does
//!   not. So int8 is chosen because it is free in accuracy terms and cheap
//!   in footprint — NOT because it is faster. The evidence lives in
//!   `parity_shipping_der.rs`; a parity gate (spec §6.2) separately confirms
//!   quantization doesn't reintroduce the NaN/Inf corruption dia already
//!   routes around `ort`'s CoreML EP for (spec §1).
//!
//!   `FBank.mlmodelc` + `Embedding.mlmodelc` (the split fbank-then-embed
//!   pipeline) are NOT targeted per spec §2.4: wespeaker_v2 computes fbank
//!   in-graph from raw waveform, so the split frontend is unnecessary.
//!
//! # Spec-vs-reality deltas
//!
//! 1. The segmentation candidates are NOT contract-equal (see DECISION).
//!    `Segmentation.mlmodelc` is part of a distinct, newer "community-1"
//!    conversion set (`coremltools` 9.0b1/`torch` 2.8.0, converted
//!    2025-10-13, minimum macOS 14) vs `pyannote_segmentation.mlmodelc`
//!    (`coremltools` 8.3.0/`torch` 2.6.0, minimum macOS 12): it batches
//!    1..=32 chunks per call (default shape `[32, 1, 160000]`) and its sole
//!    output is named `log_probs` (log-softmaxed, per its `metadata.json`)
//!    with a shape CoreML leaves unpinned — not `segments`, raw powerset
//!    logits, fixed `[1, 589, 7]`. Only `pyannote_segmentation`'s contract
//!    matches the spec's table, which introspection confirms exactly:
//!    `audio [1, 1, 160000]` f32 -> `segments [1, 589, 7]` f32.
//! 2. `wespeaker_v2.mlmodelc` (and its `wespeaker`/`wespeaker_int8`
//!    siblings) carry an undocumented second output, `constant`:
//!    fixed-shape (rank-0/scalar, NOT a symptom of input flexibility —
//!    `hasShapeFlexibility` is false in `metadata.json`) `Some(F32)`. Not
//!    in the spec; Task 2 ignores it and reads `embedding` only.
//! 3. `wespeaker_v2.mlmodelc` and `wespeaker_int8.mlmodelc` are the same
//!    file (see DECISION) — the spec's table names only `wespeaker_v2` and
//!    doesn't mention this duplication.
//! 4. Every output whose shape depends on a flexible input
//!    (`Segmentation`'s `log_probs`, `FBank`'s `fbank_features`,
//!    `Embedding`'s `embedding`) introspects to an EMPTY shape (`[]`) with
//!    the dtype still populated — `coremlit`'s `FeatureInfo` reports a real
//!    `multiArrayConstraint` (so `data_type()` resolves) but CoreML
//!    declares no static shape for it. None of these are targeted
//!    artifacts, so this doesn't block Task 2, but it is the shape a future
//!    flexible-batch design would need to handle explicitly (a predict-time
//!    concern, not a load-time one).
//! 5. Flexible-shape INPUTS (`Segmentation`'s `audio`, `FBank`'s `audio`,
//!    `Embedding`'s `fbank_features`/`weights`) introspect to their
//!    declared DEFAULT shape, not an empty/unconstrained one:
//!    `Segmentation`'s default is `[32, 1, 160000]` (the max of its 1..=32
//!    enumerated range), `FBank`'s default is `[1, 1, 160000]` (batch 1),
//!    `Embedding`'s are the low end of its range constraints (`[1, 1, 80,
//!    998]`, `[1, 589]`). None of the four targeted-artifact inputs are
//!    flexible, so this doesn't affect Task 2 either.

mod common;

use std::{collections::BTreeSet, path::Path};

use coremlit::{ComputeUnits, DataType, Model};

/// Recursively collects every FILE under `dir` as a `/`-separated path relative
/// to `root`. Used by `wespeaker_v2_and_wespeaker_int8_are_byte_identical` to
/// compare two `.mlmodelc` bundle trees file-for-file.
///
/// OS-generated sidecars are skipped: AppleDouble `._*` files and `.DS_Store`.
/// macOS materializes these inside bundles on non-native filesystems
/// (exFAT/FAT/SMB); CoreML's loader never reads them, so excluding them from
/// discovery cannot mask a functional artifact change — whereas NOT excluding
/// them would false-fail the byte-identity comparison below as a phantom
/// "unpinned extra" even though every real byte is untouched.
fn collect_files_rel(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
  for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
  {
    let entry = entry.expect("read dir entry");
    // Drop OS-generated sidecars (AppleDouble `._*`, `.DS_Store`) at every
    // depth, before the file/dir split — see the doc comment above.
    let name = entry.file_name();
    let name = name.to_string_lossy();
    if name.starts_with("._") || name == ".DS_Store" {
      continue;
    }
    let path = entry.path();
    if entry.file_type().expect("file type").is_dir() {
      collect_files_rel(root, &path, out);
    } else {
      out.insert(
        path
          .strip_prefix(root)
          .expect("walked path is under root")
          .to_str()
          .expect("utf-8 path")
          .to_string(),
      );
    }
  }
}

/// Hermetic non-vacuity proof for [`collect_files_rel`]'s sidecar filter (no
/// staged model needed). On exFAT/FAT/SMB volumes macOS materializes
/// AppleDouble `._*` and `.DS_Store` sidecars inside `.mlmodelc` bundles;
/// discovery must drop EXACTLY those, while every real file — crucially
/// including an unpinned real extra — still reaches the byte-identity gate in
/// [`wespeaker_v2_and_wespeaker_int8_are_byte_identical`]. This proves the
/// filter fixes the false-failure WITHOUT blanket-suppressing genuine extras.
#[test]
fn collect_files_rel_skips_sidecars_but_surfaces_real_extras() {
  let tmp = tempfile::tempdir().expect("create temp dir");
  let root = tmp.path();
  let bundle = root.join("Model.mlmodelc");
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

  let mut discovered: BTreeSet<String> = BTreeSet::new();
  collect_files_rel(root, &bundle, &mut discovered);

  // Discovery keeps every real file and drops every sidecar (at both depths).
  assert_eq!(
    discovered,
    BTreeSet::from([
      "Model.mlmodelc/model.mil".to_string(),
      "Model.mlmodelc/rogue.bin".to_string(),
      "Model.mlmodelc/weights/weight.bin".to_string(),
    ]),
    "discovery must exclude `._*`/.DS_Store sidecars and keep every real file"
  );

  // The filter did NOT blanket-suppress extras: an ordinary unpinned file still
  // breaks the exact-set equality and shows up as the sole difference against a
  // tree containing only the two real artifacts.
  let pinned: BTreeSet<String> = BTreeSet::from([
    "Model.mlmodelc/model.mil".to_string(),
    "Model.mlmodelc/weights/weight.bin".to_string(),
  ]);
  assert_ne!(
    discovered, pinned,
    "a real unpinned extra must still break the exact-set equality"
  );
  let extras: Vec<String> = discovered.difference(&pinned).cloned().collect();
  assert_eq!(
    extras,
    vec!["Model.mlmodelc/rogue.bin".to_string()],
    "the surviving extra must be exactly the real unpinned file, not a sidecar"
  );
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn pyannote_segmentation_io_matches_spec() {
  let model = Model::load(common::seg_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  // DECISION: this is the Task 2 segmentation target — see the module doc.
  let audio = description.input("audio").expect("audio input");
  assert_eq!(audio.shape(), &[1, 1, 160000]);
  assert_eq!(audio.data_type(), Some(DataType::F32));

  // Spec hypothesis: "589 frames, not the 592 the fps math suggests" (§4);
  // introspection CONFIRMS 589.
  //
  // This comment used to continue "raw powerset logits ... (not
  // log-probabilities)", reasoning from the output's NAME. That was
  // exactly backwards, and it is why nobody went looking for the `log` op
  // that is actually in the graph. `model.mil` ends `softmax` -> `log(eps
  // = 0x0p+0)` -> `cast` -> `-> (segments)`: `segments` carries
  // log-probabilities. A name is not a contract; the graph is. Shape and
  // dtype are all this introspection test can pin — the numerics are
  // pinned by `coremlit`'s `tests/fp16_guards.rs`.
  let segments = description.output("segments").expect("segments output");
  assert_eq!(segments.shape(), &[1, 589, 7]);
  assert_eq!(segments.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn segmentation_alt_io_recorded_not_targeted() {
  let path = common::models_dir().join("Segmentation.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  // SPEC DELTA (module doc item 1): batched, not single-chunk.
  let audio = description.input("audio").expect("audio input");
  assert_eq!(
    audio.shape(),
    &[32, 1, 160000],
    "default of the enumerated 1..=32 batch shape"
  );
  assert_eq!(audio.data_type(), Some(DataType::F32));

  // SPEC DELTA (module doc item 1): named `log_probs`, not `segments`, and
  // CoreML leaves its shape unpinned because it tracks the flexible input.
  let log_probs = description.output("log_probs").expect("log_probs output");
  assert!(
    log_probs.shape().is_empty(),
    "dynamic output shape tracking the flexible `audio` input"
  );
  assert_eq!(log_probs.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn wespeaker_v2_io_matches_spec() {
  let model = Model::load(common::embed_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  // DECISION: this is the Task 2 embedding target — see the module doc.
  let waveform = description.input("waveform").expect("waveform input");
  assert_eq!(waveform.shape(), &[3, 160000]);
  assert_eq!(waveform.data_type(), Some(DataType::F32));

  let mask = description.input("mask").expect("mask input");
  assert_eq!(mask.shape(), &[3, 589]);
  assert_eq!(mask.data_type(), Some(DataType::F32));

  let embedding = description.output("embedding").expect("embedding output");
  assert_eq!(embedding.shape(), &[3, 256]);
  assert_eq!(embedding.data_type(), Some(DataType::F32));

  // SPEC DELTA (module doc item 2): undocumented scalar second output.
  let constant = description.output("constant").expect("constant output");
  assert!(
    constant.shape().is_empty(),
    "fixed rank-0 output, not a symptom of input flexibility"
  );
  assert_eq!(constant.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn wespeaker_v2_and_wespeaker_int8_are_byte_identical() {
  // The module's embedding DECISION rests on this premise (item 3):
  // `wespeaker_v2.mlmodelc` is an ALIAS for the int8-palettized
  // `wespeaker_int8.mlmodelc`, "v2" naming the same artifact, not a distinct
  // fp32 architecture. This test is NAMED for that byte-identity but previously
  // loaded only ONE bundle and checked shapes — it opened neither the other
  // bundle nor any bytes (L4). It now enumerates BOTH bundle trees and
  // byte-compares them file-for-file: equal relative path sets, and equal bytes
  // for every file. A divergence breaks the DECISION's premise and is a finding
  // to surface, NOT a test to relax.
  let v2 = common::embed_path(); // wespeaker_v2.mlmodelc (the shipping int8 alias)
  let int8 = common::models_dir().join("wespeaker_int8.mlmodelc");

  let mut v2_tree = BTreeSet::new();
  let mut int8_tree = BTreeSet::new();
  collect_files_rel(&v2, &v2, &mut v2_tree);
  collect_files_rel(&int8, &int8, &mut int8_tree);
  assert!(
    !v2_tree.is_empty(),
    "wespeaker_v2.mlmodelc has no files — bundle missing or empty at {}",
    v2.display()
  );
  assert_eq!(
    v2_tree,
    int8_tree,
    "wespeaker_v2 / wespeaker_int8 bundle file trees differ — v2-only: {:?}, int8-only: {:?}",
    v2_tree.difference(&int8_tree).collect::<Vec<_>>(),
    int8_tree.difference(&v2_tree).collect::<Vec<_>>(),
  );

  for rel in &v2_tree {
    let a = std::fs::read(v2.join(rel)).unwrap_or_else(|e| panic!("read v2/{rel}: {e}"));
    let b = std::fs::read(int8.join(rel)).unwrap_or_else(|e| panic!("read int8/{rel}: {e}"));
    assert!(
      a == b,
      "wespeaker_v2 and wespeaker_int8 differ at `{rel}` ({} vs {} bytes; sha256 {} vs {}) — the \
       `v2 == int8` premise the embedding DECISION rests on (module doc) is broken; investigate \
       before relying on either as the shipping artifact",
      a.len(),
      b.len(),
      common::sha256_hex(&a),
      common::sha256_hex(&b),
    );
  }

  // Re-verify the shared I/O contract actually loads (the argmax-side precedent
  // of confirming via `Model::load`, not assuming it from byte-identity alone).
  let model = Model::load(&v2, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();
  assert_eq!(description.input("waveform").unwrap().shape(), &[3, 160000]);
  assert_eq!(description.input("mask").unwrap().shape(), &[3, 589]);
  assert_eq!(description.output("embedding").unwrap().shape(), &[3, 256]);
}

/// Byte-pins every file of the fp16-safe segmentation artifact against the
/// published `FinDIT-Studio/speakerkit-coreml` revision (module doc,
/// "Segmentation provenance").
///
/// This is the gate that makes the issue-#15 swap non-reversible by accident.
/// The whole difference between the two conversions lives inside `model.mil`:
/// the pre-swap FluidInference artifact has the identical filename, the
/// identical `audio [1,1,160000]` → `segments [1,589,7]` contract, and passes
/// every other test in this file — while restoring a `segments` minimum of
/// −45440 on the shipping `ComputeUnits::All` placement. Only these bytes carry
/// the repair, so only a byte pin can defend it.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn fp16_safe_segmentation_matches_pinned_sha256() {
  const FILES: &[(&str, &str)] = &[
    (
      "metadata.json",
      "2926b811344e40ab6ce5406354bf5aaac35a297d0e67e3c0d3f6dd766e9f5f8f",
    ),
    (
      "model.mil",
      "ded0d1ee11d77976b5c706ce667d0c8cb49977d3fe4367cccbd7b582bdb86dec",
    ),
    (
      "weights/weight.bin",
      "0266f4ad4d843ecf31ef9220ad6b80616b3ec64a4404b64f3ea0371554e236ec",
    ),
    (
      "coremldata.bin",
      "e75fc0ac9641de87e3514369455c8c8a65b00aae339817b20642f115d5d8861e",
    ),
    (
      "analytics/coremldata.bin",
      "f283c01fa863188733c33c6fddac4c5dca42ca7cb22580918e1bc55877897e69",
    ),
  ];

  let root = common::seg_path();
  for (relative, expected) in FILES {
    let path = root.join(relative);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
      common::sha256_hex(&bytes),
      *expected,
      "sha256 drift on pyannote_segmentation.mlmodelc/{relative}. These bytes ARE the issue-#15 \
       fp16-guard repair; the pre-repair FluidInference artifact is contract-identical and would \
       pass every other gate while restoring a −45440 `segments` minimum on ComputeUnits::All. \
       Re-download from FinDIT-Studio/speakerkit-coreml at the pinned revision, or re-baseline \
       this pin deliberately."
    );
  }
}

/// The shipped segmentation graph reaches its powerset log-probabilities through
/// the FUSED `reduce_log_sum_exp` → `sub` tail, and contains no `log` op that a
/// vanishing epsilon could leave unguarded. Reading the MIL is what the whole
/// issue-#15 investigation turned on — the docs asserted "raw powerset logits"
/// and nobody went looking for the `log` — so the structural claim is asserted,
/// not narrated. `tests/fp16_guards.rs` checks epsilon FLOORS across every
/// vendor; this checks the one structural property that makes this graph
/// epsilon-free in the first place.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn segmentation_graph_has_no_log_op() {
  let mil = common::seg_path().join("model.mil");
  let text =
    std::fs::read_to_string(&mil).unwrap_or_else(|e| panic!("read {}: {e}", mil.display()));
  assert!(
    text.contains("reduce_log_sum_exp"),
    "pyannote_segmentation/model.mil has no `reduce_log_sum_exp`: this is not the fused-tail \
     conversion the fp16 repair rests on"
  );
  assert!(
    !text.contains("= log("),
    "pyannote_segmentation/model.mil grew a `log` op back. The decomposed `softmax` → `log` tail \
     is what saturated to −45440 on the ANE; a re-conversion that reintroduces it must be \
     re-audited, not accepted."
  );
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn wespeaker_fp32_io_contract_equal_but_not_targeted() {
  // `wespeaker.mlmodelc` shares the identical I/O contract with
  // `wespeaker_v2`/`wespeaker_int8` (same names/shapes/dtypes below) but is
  // a DIFFERENT, non-palettized artifact: 27 MB of weights vs 6.9 MB
  // (`du -sh */weights`, recorded at plan time) — full float32 storage
  // precision instead of 8-bit palettized. Not targeted (see DECISION in
  // the module doc), but contract equality means Task 2 could fall back to
  // it without any shape-handling changes.
  let path = common::models_dir().join("wespeaker.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();
  assert_eq!(description.input("waveform").unwrap().shape(), &[3, 160000]);
  assert_eq!(description.input("mask").unwrap().shape(), &[3, 589]);
  assert_eq!(description.output("embedding").unwrap().shape(), &[3, 256]);
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn fbank_io_recorded_not_targeted() {
  // NOT targeted (spec §2.4 — `wespeaker_v2` computes fbank in-graph, so
  // this split-pipeline frontend is unused). Recorded because the plan
  // brief names it as a candidate embedding-pipeline artifact.
  let path = common::models_dir().join("FBank.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let audio = description.input("audio").expect("audio input");
  assert_eq!(
    audio.shape(),
    &[1, 1, 160000],
    "default of the enumerated 1..=32 batch shape"
  );
  assert_eq!(audio.data_type(), Some(DataType::F32));

  // SPEC DELTA (module doc item 4): dynamic output shape. `metadata.json`'s
  // shortDescription documents the resolved per-chunk shape as 80 x 998
  // (mel bins x frames) once a concrete batch is chosen.
  let features = description
    .output("fbank_features")
    .expect("fbank_features output");
  assert!(
    features.shape().is_empty(),
    "dynamic output shape tracking the flexible `audio` input"
  );
  assert_eq!(features.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embedding_split_io_recorded_not_targeted() {
  // NOT targeted — the split-pipeline second stage (fbank features +
  // per-frame weights -> embedding), superseded by wespeaker_v2's single
  // raw-waveform call (spec §2.4). Recorded per the plan brief.
  let path = common::models_dir().join("Embedding.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let features = description
    .input("fbank_features")
    .expect("fbank_features input");
  assert_eq!(
    features.shape(),
    &[1, 1, 80, 998],
    "default (low end) of a [1,32]x[1,1]x[80,80]x[998,998] range constraint"
  );
  assert_eq!(features.data_type(), Some(DataType::F32));

  // Named `weights` here, distinct from wespeaker_v2's `mask` — same
  // per-frame speaker-activity role (589 = the segmentation frame count),
  // but `metadata.json`'s shortDescription says this pipeline interpolates
  // it to 125 frames internally before pooling.
  let weights = description.input("weights").expect("weights input");
  assert_eq!(
    weights.shape(),
    &[1, 589],
    "default (low end) of a [1,32]x[589,589] range constraint"
  );
  assert_eq!(weights.data_type(), Some(DataType::F32));

  // SPEC DELTA (module doc item 4): dynamic output shape.
  let embedding = description.output("embedding").expect("embedding output");
  assert!(
    embedding.shape().is_empty(),
    "dynamic output shape tracking the flexible `fbank_features`/`weights` inputs"
  );
  assert_eq!(embedding.data_type(), Some(DataType::F32));
}
