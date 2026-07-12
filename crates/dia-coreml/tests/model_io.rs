//! Ground-truth introspection of every candidate segmentation and embedding
//! artifact named in the design spec
//! (`docs/superpowers/specs/2026-07-11-dia-coreml-backends-design.md` §4, §9
//! open item) plus the out-of-scope PLDA artifact the plan brief also names.
//! Every value below comes from loading the real `.mlmodelc` via
//! `coremlit::Model::load` + `.description()` — the spec's table is a
//! HYPOTHESIS; reality wins, and every place it differs is marked `SPEC
//! DELTA`. Feeds Task 2 (`SegmentModel`) and later tasks (`EmbedModel`,
//! `Extractor`).
//!
//! # Artifacts (`Models/dia-coreml/`, gitignored, fetched dev-time)
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
//! | `PLDA.mlmodelc` | clustering input transform | out of scope (spec §3 non-goal) |
//!
//! # Licenses (`Models/dia-coreml/README.md`)
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
//! Computer Speech & Language 2022) — the last is irrelevant to dia-coreml
//! (clustering stays in `dia`, spec §3) but ships in the same README and is
//! reproduced here for completeness.
//!
//! # DECISION
//!
//! - **Segmentation: `pyannote_segmentation.mlmodelc`.** The two candidates
//!   are NOT contract-equal — the plan brief's stated tiebreaker condition
//!   ("pick pyannote_segmentation if contract-equal") does not actually
//!   hold, see `segmentation_alt_io_recorded_not_targeted` below. It is
//!   chosen anyway because its single-chunk, fixed-shape `segments` output
//!   (raw powerset logits) matches both the spec's pinned contract (§4
//!   table) and the `SegmentModel::infer` single-chunk API (§5) exactly,
//!   and it is FluidAudio's shipping name — the brief's fallback
//!   tiebreaker.
//! - **Embedding: `wespeaker_v2.mlmodelc`.** Verified byte-identical
//!   (`diff -rq`, sha256 of `model.mil` and `weights/weight.bin`, at plan
//!   time) to `wespeaker_int8.mlmodelc` — "v2" is an alias for the
//!   int8-palettized model, not a distinct fp32 architecture; see
//!   `wespeaker_v2_and_wespeaker_int8_are_byte_identical` below.
//!   `wespeaker.mlmodelc` is contract-equal but ships uncompressed fp32
//!   weights (27 MB vs 6.9 MB, `du -sh */weights`); not targeted, since the
//!   smaller int8 footprint better serves the issue's ANE uplift targets —
//!   a parity gate (spec §6.2) confirms quantization doesn't reintroduce
//!   the NaN/Inf corruption dia already routes around `ort`'s CoreML EP
//!   for (spec §1). `FBank.mlmodelc` + `Embedding.mlmodelc` (the split
//!   fbank-then-embed pipeline) are NOT targeted per spec §2.4: wespeaker_v2
//!   computes fbank in-graph from raw waveform, so the split frontend is
//!   unnecessary.
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

use coremlit::{ComputeUnits, DataType, Model};

#[test]
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
fn pyannote_segmentation_io_matches_spec() {
  let model = Model::load(common::seg_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  // DECISION: this is the Task 2 segmentation target — see the module doc.
  let audio = description.input("audio").expect("audio input");
  assert_eq!(audio.shape(), &[1, 1, 160000]);
  assert_eq!(audio.data_type(), Some(DataType::F32));

  // Spec hypothesis: "589 frames, not the 592 the fps math suggests" (§4);
  // introspection CONFIRMS 589 — raw powerset logits, matching the spec's
  // `segments` name exactly (not log-probabilities; see the alt candidate
  // below).
  let segments = description.output("segments").expect("segments output");
  assert_eq!(segments.shape(), &[1, 589, 7]);
  assert_eq!(segments.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
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
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
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
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
fn wespeaker_v2_and_wespeaker_int8_are_byte_identical() {
  // SPEC DELTA (module doc item 3): `wespeaker_v2.mlmodelc` and
  // `wespeaker_int8.mlmodelc` are not merely contract-equal, they are the
  // SAME artifact (`diff -rq` and sha256 of `model.mil` +
  // `weights/weight.bin`, verified at plan time — see task-1-report.md).
  // This test pins the I/O contract only; byte-identity isn't re-verified
  // per run.
  let path = common::models_dir().join("wespeaker_int8.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();
  assert_eq!(description.input("waveform").unwrap().shape(), &[3, 160000]);
  assert_eq!(description.input("mask").unwrap().shape(), &[3, 589]);
  assert_eq!(description.output("embedding").unwrap().shape(), &[3, 256]);
}

#[test]
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
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
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
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
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
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

#[test]
#[ignore = "requires local dia-coreml models (DIA_COREML_TEST_MODELS)"]
fn plda_io_recorded_out_of_scope() {
  // Out of scope per spec §3 non-goals ("Any clustering/VBx/PLDA port") —
  // recorded because the plan brief names it as a candidate artifact to
  // introspect. Fixed-shape throughout; no flexibility declared at all.
  let path = common::models_dir().join("PLDA.mlmodelc");
  let model = Model::load(path, ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  let embeddings = description.input("embeddings").expect("embeddings input");
  assert_eq!(embeddings.shape(), &[1, 256]);
  assert_eq!(embeddings.data_type(), Some(DataType::F32));

  let plda_features = description
    .output("plda_features")
    .expect("plda_features output");
  assert_eq!(plda_features.shape(), &[1, 128]);
  assert_eq!(plda_features.data_type(), Some(DataType::F32));
}
