//! Ground-truth introspection + provenance pins for the ArcFace CoreML
//! artifact.
//!
//! Every shape/dtype claim in the model-gated half comes from loading the real
//! `.mlmodelc` through `coremlit::Model::load` + `.description()`, and every
//! SHA from the downloaded bytes. The hermetic half runs with no model at all
//! — including the one gate here that loads a real bundle and is NOT
//! `#[ignore]`d, because the bundle it loads is committed.
//!
//! **Research-only weights.** See `tests/face/arcface/mod.rs` for the terms and
//! for why the artifact repository is private.
//!
//! # Pinned contract
//!
//! ```text
//! data       f32  [1, 3, 112, 112]   fixed shape, never RangeDim
//! embedding  f32  [1, 512]           RAW — no L2 in the graph
//! ```

#[path = "arcface/mod.rs"]
mod common;

use coremlit::{
  ComputeUnits, DataType, Model, MultiArray, ShapeConstraint,
  embeddings::face::{
    FaceEmbedder, FaceEmbedderOptions, FaceModel, TEMPLATE_SIZE, arcface, error::Error,
  },
};

/// The batch, channel count and side the artifact pins, spelled here rather
/// than imported: the door reads them back from the graph, so a test that took
/// them from the door would be comparing the artifact against itself.
const INPUT_SHAPE: [usize; 4] = [1, 3, 112, 112];

/// The embedding width the manifest declares and the graph must produce.
const EMBEDDING_DIM: usize = 512;

/// A `[1, 3, 112, 112]` f32 input tensor of zeros, for probing the runtime.
fn face_tensor() -> MultiArray {
  let len: usize = INPUT_SHAPE.iter().product();
  MultiArray::from_slice(&INPUT_SHAPE, &vec![0.0f32; len]).expect("build face tensor")
}

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The provenance strings are pinned so a re-download from a different
/// revision cannot quietly reuse this file's SHA table, and so the two chains
/// stay distinct: the ARTIFACT repo's revision is not the upstream SOURCE's,
/// and neither is the converted member's own hash.
#[test]
fn provenance_is_pinned() {
  assert_eq!(common::HF_REPO, "FinDIT-Studio/facekit-coreml");
  assert_eq!(common::HF_REVISION.len(), 40);
  assert_eq!(common::SOURCE_PACK, "buffalo_l.zip");
  assert_eq!(common::SOURCE_PACK_SHA256.len(), 64);
  assert_eq!(common::SOURCE_MEMBER, "w600k_r50.onnx");
  assert_eq!(common::SOURCE_MEMBER_SHA256.len(), 64);
  assert_eq!(common::INSIGHTFACE_REVISION.len(), 40);
  assert_ne!(
    common::HF_REVISION,
    common::INSIGHTFACE_REVISION,
    "the artifact repo's revision and the upstream source's are different chains"
  );
  assert_ne!(
    common::SOURCE_PACK_SHA256,
    common::SOURCE_MEMBER_SHA256,
    "the pack's hash is not the converted member's"
  );
  assert_eq!(common::ARTIFACT_SHA256.len(), 5);
  for (path, sha) in common::ARTIFACT_SHA256 {
    assert_eq!(sha.len(), 64, "{path} needs a full SHA-256");
    assert!(
      sha
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
      "{path}: SHA-256 must be lowercase hex"
    );
  }
}

/// The library's manifest and this suite's pins describe one artifact.
///
/// Both are hand-written constants in different files, and the model-gated
/// gates below read the artifact through the library's. A drift between them
/// would move what the door sends while every assertion about the graph still
/// passed, so the two are reconciled here, hermetically.
#[test]
fn the_library_and_this_suite_name_one_bundle() {
  assert_eq!(arcface::BUNDLE_NAME, common::BUNDLE_NAME);
  assert!(
    arcface::STAGED_PATH.ends_with(arcface::BUNDLE_NAME),
    "the library's staged path must name the bundle: {}",
    arcface::STAGED_PATH
  );
  assert_eq!(arcface::MODEL.dim(), EMBEDDING_DIM);
  assert_eq!(arcface::MODEL.input(), "data");
  assert_eq!(arcface::MODEL.output(), "embedding");
  assert_eq!(TEMPLATE_SIZE, INPUT_SHAPE[2]);
  assert_eq!(TEMPLATE_SIZE, INPUT_SHAPE[3]);
}

/// **The manifest refuses a model that is not its artifact, on a bundle this
/// repository commits.**
///
/// NOT `#[ignore]`d, and that is the point of choosing silero:
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` is 1.1 MiB of
/// committed bytes staged by no download, so this runs in the modelless
/// `features` job under `--features commercial-face-arcface` and on any fresh
/// clone. Every other door in this crate carries the same pin.
///
/// What it establishes that a fixture cannot: that the refusal comes out of
/// [`FaceEmbedder::load`] over a description the CoreML runtime itself built,
/// and that the manifest doing the refusing is the SHIPPED one —
/// [`arcface::MODEL`], not a value assembled in the test. A mis-pointed
/// `model_path` is exactly this shape.
#[test]
fn the_arcface_manifest_refuses_the_vendored_silero_bundle() {
  let bundle = common::models_root()
    .join("vadkit")
    .join("silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; looked for {}",
    bundle.display()
  );
  let options = FaceEmbedderOptions::new().with_compute(ComputeUnits::CpuOnly);

  // The by-name clause: silero declares no `data`, and the message names what
  // it does declare.
  let error = FaceEmbedder::load(&bundle, arcface::MODEL, options)
    .expect_err("silero declares no `data` feature");
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == arcface::MODEL.input() && m.actual().contains("audio_input")),
    "{error}"
  );

  // And the geometry clause, reached by naming a feature silero DOES declare
  // with this artifact's width: `audio_input` is `[1, 4160]`, a rank no
  // contract of this door's can be built from. Same preprocessing, same width
  // — only the artifact is wrong, which is the mis-pointed-path case.
  let error = FaceEmbedder::load(
    &bundle,
    FaceModel::new("audio_input", "vad_output", EMBEDDING_DIM)
      .with_preprocessing(arcface::MODEL.preprocessing()),
    options,
  )
  .expect_err("silero's audio window is not a template face");
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == "audio_input" && m.actual() == "[1, 4160]"),
    "{error}"
  );
}

// ── Model-gated ─────────────────────────────────────────────────────────────

/// The declared I/O contract, read off the real model. Fixed shapes on both
/// sides: the conversion refuses a `RangeDim` input on purpose, because a
/// flexible input is off the Neural Engine for every shape but its default
/// (coremltools #2370 measured ANE residency going 78 % → 0 %).
///
/// `shape()` alone cannot say that. A flexible input reports its DEFAULT shape
/// through the snapshot, so a graph converted at `[1, 3, 112, 112]` and then
/// made flexible satisfies every assertion about the numbers — which is why
/// [`ShapeConstraint`] is asserted here as well, and why `FaceEmbedder::load`
/// refuses anything but [`ShapeConstraint::Fixed`].
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn model_declares_the_pinned_io_contract() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).expect("load model");
  let description = model.description();

  let input = description.input("data").expect("`data` input");
  assert_eq!(input.shape(), INPUT_SHAPE);
  assert_eq!(input.data_type(), Some(DataType::F32));
  assert_eq!(
    input.shape_constraint(),
    Some(ShapeConstraint::Fixed),
    "the `data` input must accept exactly one shape; a flexible one reports the same NUMBERS \
     through `shape()` and takes the graph off the accelerator"
  );

  let output = description.output("embedding").expect("`embedding` output");
  assert_eq!(output.shape(), [1, EMBEDDING_DIM]);
  assert_eq!(output.data_type(), Some(DataType::F32));
  assert_eq!(output.shape_constraint(), Some(ShapeConstraint::Fixed));

  assert_eq!(description.inputs().len(), 1, "exactly one input");
  assert_eq!(description.outputs().len(), 1, "exactly one output");
  assert!(
    description.states().is_empty(),
    "the ArcFace graph must declare NO `MLState` buffers: `FaceEmbedder::embed` predicts through \
     the stateless API, which CoreML does not allow for a stateful model. Declared: {:?}",
    description
      .states()
      .iter()
      .map(coremlit::FeatureInfo::name)
      .collect::<Vec<_>>()
  );
  assert!(
    !input.is_optional(),
    "the one input this door supplies is the one the graph requires"
  );

  // The COMPLETE input set, not just the feature the door sends. A graph
  // carrying `data` plus a second REQUIRED input satisfies every per-feature
  // assertion above and then fails on every prediction.
  let unsatisfiable: Vec<&str> = description
    .inputs()
    .iter()
    .filter(|f| f.name() != "data" && !f.is_optional())
    .map(coremlit::FeatureInfo::name)
    .collect();
  assert!(
    unsatisfiable.is_empty(),
    "the graph requires {unsatisfiable:?}, which this door never sends"
  );

  // The batch the door will chunk to, read back through the door itself.
  let embedder = FaceEmbedder::from_file(common::model_path(), arcface::MODEL).expect("load door");
  assert_eq!(embedder.batch_capacity(), 1, "the artifact pins batch 1");
  assert_eq!(embedder.dim(), EMBEDDING_DIM);
}

/// The staged bundle is byte-for-byte the pinned artifact — exact file set,
/// exact SHA-256 per file — **and the artifact repository's own
/// `CHECKSUMS.sha256` says the same thing**.
///
/// Two enumerations rather than one, because they fail in different
/// directions. This file's table is what the licence row keys on
/// (`tests/model_licences.rs` reads `ARTIFACT_SHA256` and refuses a row whose
/// hash it does not hold), and it is exact, so a file added to the bundle reds
/// here. `CHECKSUMS.sha256` is the publisher's own statement, staged beside the
/// bundle and verified by CI's `shasum -c` step before any gate runs;
/// reconciling the two means a re-publish that updated one and not the other
/// cannot pass.
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn artifact_matches_the_pinned_sha_manifest() {
  common::assert_exact_sha_manifest(&common::model_path(), common::ARTIFACT_SHA256);

  let checksums = common::models_dir().join("CHECKSUMS.sha256");
  let text = std::fs::read_to_string(&checksums)
    .unwrap_or_else(|e| panic!("read {checksums:?}: {e} — the kit stages its own manifest"));
  let published: std::collections::BTreeMap<String, String> = text
    .lines()
    .filter(|line| !line.trim().is_empty())
    .map(|line| {
      let (sha, path) = line
        .split_once("  ")
        .unwrap_or_else(|| panic!("{checksums:?}: not `<sha256>  <path>`: {line:?}"));
      // Kit-root-relative, `./`-prefixed — which is why CI's checksum step
      // runs from the kit root with no filter.
      let relative = path
        .trim()
        .strip_prefix(&format!("./{}/", common::BUNDLE_NAME))
        .unwrap_or_else(|| {
          panic!(
            "{checksums:?}: {path:?} is not under ./{}/",
            common::BUNDLE_NAME
          )
        })
        .to_owned();
      (relative, sha.to_owned())
    })
    .collect();
  let pinned: std::collections::BTreeMap<String, String> = common::ARTIFACT_SHA256
    .iter()
    .map(|(path, sha)| ((*path).to_owned(), (*sha).to_owned()))
    .collect();
  assert_eq!(
    published, pinned,
    "the staged CHECKSUMS.sha256 and this suite's pinned manifest describe different bundles"
  );
}

/// `FaceEmbedder::load` accepts the real artifact under every compute
/// placement, so the contract check is not accidentally tied to one of them —
/// and the recommended arm is one of the four rather than a special case.
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn embedder_loads_under_every_compute_placement() {
  for compute in [
    ComputeUnits::All,
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
  ] {
    let embedder = FaceEmbedder::load(
      common::model_path(),
      arcface::MODEL,
      FaceEmbedderOptions::new().with_compute(compute),
    )
    .unwrap_or_else(|e| panic!("load under {compute:?}: {e}"));
    assert_eq!(embedder.dim(), EMBEDDING_DIM);
  }
  assert!(
    [
      ComputeUnits::All,
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuAndGpu,
      ComputeUnits::CpuAndNeuralEngine,
    ]
    .contains(&arcface::RECOMMENDED_COMPUTE),
    "the recommended arm must be one of the four this gate loads under"
  );
}

/// **The graph's own output is RAW.** The conversion's whole "there is no L2
/// to strip" claim reduces to this: a graph that normalised would answer 1.0,
/// and the door's normalisation would then be a second one.
///
/// Taken through [`Model::predict_with`] rather than through the door, because
/// the door's [`FaceEmbedding`][emb] is unit-norm BY CONTRACT — asking it for
/// a norm would measure the door's own L2 and say nothing about the artifact.
/// The recipe measured `‖e‖` 17.01 – 24.91 over the fixture faces; the bound
/// here is far wider than that, because this exists to catch a graph that grew
/// a normalisation rather than to re-pin a measurement.
///
/// [emb]: coremlit::embeddings::face::FaceEmbedding
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn the_graphs_embeddings_are_raw_and_not_unit_norm() {
  let reference = common::load_reference();
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).expect("load model");
  let preprocessing = arcface::MODEL.preprocessing();

  for face in reference.faces.iter().take(3) {
    // RGB, NCHW, `byte * scale + bias` — the manifest's own arithmetic, laid
    // out by hand here so this gate does not depend on the door it checks.
    let mut planar = vec![0.0f32; INPUT_SHAPE.iter().product()];
    let side = TEMPLATE_SIZE;
    for y in 0..side {
      for x in 0..side {
        for c in 0..3 {
          planar[c * side * side + y * side + x] = f32::from(face.crop[(y * side + x) * 3 + c])
            .mul_add(preprocessing.scale(), preprocessing.bias()[c]);
        }
      }
    }
    let input = MultiArray::from_slice(&INPUT_SHAPE, &planar).expect("build tensor");
    let outputs = model
      .predict_with(&[("data", &input)])
      .unwrap_or_else(|e| panic!("{}: predict: {e}", face.id));
    let raw = outputs.get("embedding").expect("output present");
    assert_eq!(raw.shape(), [1, EMBEDDING_DIM]);
    let values: &[f32] = raw.as_slice().expect("read embedding");
    assert!(
      values.iter().all(|v| v.is_finite()),
      "{}: non-finite component",
      face.id
    );
    let norm = values
      .iter()
      .map(|v| f64::from(*v) * f64::from(*v))
      .sum::<f64>()
      .sqrt();
    eprintln!("[arcface] {}: raw ‖e‖ = {norm:.4}", face.id);
    assert!(
      norm > 2.0,
      "{} embedded to norm {norm:.4} — an L2 in the graph would read 1.0, and coremlit's \
       contract is that the DOOR normalises",
      face.id
    );
  }
}

/// The runtime takes the pinned shape and refuses its neighbours. A
/// fixed-shape graph should reject a transposed, off-by-one or re-batched
/// input outright, and this is what would catch a re-export that quietly went
/// flexible.
#[test]
#[ignore = "requires the staged arcface model (FACEKIT_TEST_MODELS)"]
fn runtime_accepts_exactly_the_pinned_shape() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).expect("load model");
  let outputs = model
    .predict_with(&[("data", &face_tensor())])
    .expect("the pinned shape must be accepted");
  assert_eq!(
    outputs.get("embedding").expect("output present").shape(),
    [1, EMBEDDING_DIM]
  );

  for shape in [
    vec![1, TEMPLATE_SIZE, TEMPLATE_SIZE, 3],
    vec![1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE + 1],
    vec![2, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
  ] {
    let len: usize = shape.iter().product();
    let input = MultiArray::from_slice(&shape, &vec![0.0f32; len]).expect("build tensor");
    assert!(
      model.predict_with(&[("data", &input)]).is_err(),
      "{shape:?} must be refused by the runtime"
    );
  }
}
