//! Ground-truth introspection + provenance pins for the identity-embedder
//! CoreML artifact.
//!
//! Every shape/dtype claim in the model-gated half comes from loading the real
//! `.mlmodelc` through `coremlit::Model::load` + `.description()`, and every SHA
//! from the downloaded bytes. The hermetic half runs with no model at all.
//!
//! The model gates are `#[ignore]`d and run only with the artifact staged
//! (`IDENTITY_TEST_MODELS`, default `Models/redimnet`). **They have never run in
//! CI**: the artifact repository is private and the workflow's `hf download`
//! carries no credentials, so the `identity` shard cannot stage this kit until
//! a Hugging Face read token reaches it. `tests/identity/common/mod.rs` says why
//! the repository is private; `.github/workflows/ci.yml`'s `identity` row says
//! what the shard needs.
//!
//! # Pinned contract
//!
//! ```text
//! mel        f32  [1, 72, 401]     fixed shape, never RangeDim
//! embedding  f32  [1, 192]         RAW — no L2 in the graph
//! ```

mod common;

use coremlit::{
  ComputeUnits, DataType, Model, MultiArray, ShapeConstraint,
  audio::identity::{
    EMBEDDING_DIM, Embedder, EmbedderOptions, Error, N_FRAMES, N_MELS, SAMPLE_RATE_HZ,
    WINDOW_SAMPLES,
  },
};

/// A `[1, 72, 401]` f32 input tensor of zeros, for probing the runtime.
fn mel_tensor() -> MultiArray {
  MultiArray::from_slice(&[1, N_MELS, N_FRAMES], &vec![0.0f32; N_MELS * N_FRAMES])
    .expect("build mel tensor")
}

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The provenance strings are pinned so a re-download from a different revision
/// cannot quietly reuse this file's SHA table, and so the two chains stay
/// distinct: the ARTIFACT repo's revision is not the upstream SOURCE's, and
/// neither is the weights asset's own hash.
#[test]
fn provenance_is_pinned() {
  assert_eq!(common::HF_REPO, "FinDIT-Studio/redimnetkit-coreml");
  assert_eq!(common::HF_REVISION.len(), 40);
  assert_eq!(common::SOURCE_ASSET, "b5-vox2-ft_lm.pt");
  assert_eq!(common::SOURCE_ASSET_SHA256.len(), 64);
  assert_eq!(common::SOURCE_CODE_REVISION.len(), 40);
  assert_ne!(
    common::HF_REVISION,
    common::SOURCE_CODE_REVISION,
    "the artifact repo's revision and the upstream model source's are different chains"
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

/// The geometry this file pins against the runtime below is the library's
/// published geometry, restated independently so a drift in either is visible.
#[test]
fn published_geometry_agrees_with_this_files_pins() {
  assert_eq!(SAMPLE_RATE_HZ, 16_000);
  assert_eq!(WINDOW_SAMPLES, 96_000);
  assert_eq!(WINDOW_SAMPLES, 6 * SAMPLE_RATE_HZ as usize);
  assert_eq!((N_MELS, N_FRAMES), (72, 401));
  assert_eq!(EMBEDDING_DIM, 192);
}

/// A clip that is not exactly one window is refused WITHOUT a model — the guard
/// runs before any tensor is built, so it is reachable here.
#[test]
fn a_wrong_length_clip_is_refused_before_any_model_is_needed() {
  let embedder_free_check = coremlit::audio::identity::WindowLength::new(48_000, WINDOW_SAMPLES);
  assert_eq!(embedder_free_check.got(), 48_000);
  assert_eq!(embedder_free_check.expected(), WINDOW_SAMPLES);
  let rendered = Error::WindowLength(embedder_free_check).to_string();
  assert!(
    rendered.contains("neither padded nor truncated"),
    "the refusal must say what did NOT happen to the caller's audio: {rendered}"
  );
}

// ── Model-gated ─────────────────────────────────────────────────────────────

/// The declared I/O contract, read off the real model. Fixed shapes on both
/// sides: the conversion refuses a `RangeDim` input on purpose, because a
/// flexible input takes the graph off the ANE.
///
/// `shape()` alone cannot say that. A `RangeDims` input reports its DEFAULT
/// shape through the snapshot, so a flexible graph converted at
/// `[1, 72, 401]` satisfies every assertion about the numbers — which is why
/// [`coremlit::FeatureInfo::shape_constraint`] is asserted here as well, and
/// why `Embedder::load` refuses anything but
/// [`coremlit::ShapeConstraint::Fixed`].
///
/// **This is also where a prediction about the artifact gets settled.** The
/// classification was measured on a DIFFERENT fixed-shape export (the staged
/// silero VAD bundle, `hasShapeFlexibility: "0"`, which reports raw
/// constraint type 2 with one enumerated shape and unit spans). The redimnet
/// bundle is converted the same way — `ct.TensorType(shape=(1, 72, 401))`,
/// no `RangeDim` — so it is EXPECTED to classify `Fixed`; nothing in this
/// repository has confirmed it, because the artifact has never been staged.
/// If that prediction is wrong, `Embedder::load` refuses the artifact and
/// this assertion is what says so, on the first run that stages it.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn model_declares_the_pinned_io_contract() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).expect("load model");
  let description = model.description();

  let input = description.input("mel").expect("`mel` input");
  assert_eq!(input.shape(), [1, N_MELS, N_FRAMES]);
  assert_eq!(input.data_type(), Some(DataType::F32));
  assert_eq!(
    input.shape_constraint(),
    Some(ShapeConstraint::Fixed),
    "the mel input must accept exactly one shape; a flexible one reports the same NUMBERS \
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
    "the identity graph must declare NO `MLState` buffers: `Embedder::embed` predicts through \
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
  // carrying `mel` plus a second REQUIRED input satisfies every per-feature
  // assertion above and then fails on every prediction, because `embed`
  // supplies `mel` and nothing else.
  let unsatisfiable: Vec<&str> = description
    .inputs()
    .iter()
    .filter(|f| f.name() != "mel" && !f.is_optional())
    .map(coremlit::FeatureInfo::name)
    .collect();
  assert!(
    unsatisfiable.is_empty(),
    "the graph requires {unsatisfiable:?}, which this door never sends"
  );
}

/// The staged bundle is byte-for-byte the pinned artifact — exact file set,
/// exact SHA-256 per file.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn artifact_matches_the_pinned_sha_manifest() {
  common::assert_exact_sha_manifest(&common::model_path(), common::ARTIFACT_SHA256);
}

/// The runtime takes the pinned shape and refuses its neighbours. A fixed-shape
/// graph should reject a transposed or off-by-one mel outright, and this is what
/// would catch a re-export that quietly went flexible.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn runtime_accepts_exactly_the_pinned_shape() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).expect("load model");

  let outputs = model
    .predict_with(&[("mel", &mel_tensor())])
    .expect("the pinned shape must be accepted");
  let embedding = outputs.get("embedding").expect("output present");
  assert_eq!(embedding.shape(), [1, EMBEDDING_DIM]);

  for shape in [
    vec![1, N_FRAMES, N_MELS],
    vec![1, N_MELS, N_FRAMES + 1],
    vec![2, N_MELS, N_FRAMES],
  ] {
    let len: usize = shape.iter().product();
    let input = MultiArray::from_slice(&shape, &vec![0.0f32; len]).expect("build tensor");
    assert!(
      model.predict_with(&[("mel", &input)]).is_err(),
      "{shape:?} must be refused by the runtime"
    );
  }
}

/// `Embedder::load` accepts the real artifact under every compute placement, so
/// the contract check is not accidentally tied to one of them — and the shipped
/// default is one of the four rather than a special case.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn embedder_loads_under_every_compute_placement() {
  for compute in [
    ComputeUnits::All,
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
  ] {
    Embedder::load(
      common::model_path(),
      EmbedderOptions::new().with_compute(compute),
    )
    .unwrap_or_else(|e| panic!("load under {compute:?}: {e}"));
  }
}

/// The door's own guard fires before the model does: a clip that is not exactly
/// one window comes back as [`Error::WindowLength`] rather than as a CoreML
/// prediction error, at both ends.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn embedder_guards_the_window_before_calling_the_model() {
  let embedder = Embedder::load(
    common::model_path(),
    EmbedderOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load embedder");

  let raw = embedder
    .embed(&vec![0.0f32; WINDOW_SAMPLES])
    .expect("exactly one window must be accepted");
  assert_eq!(raw.len(), EMBEDDING_DIM);

  for rejected in [WINDOW_SAMPLES - 1, WINDOW_SAMPLES + 1] {
    let error = embedder
      .embed(&vec![0.0f32; rejected])
      .expect_err("must be rejected");
    assert!(
      matches!(error, Error::WindowLength(_)),
      "{rejected} samples must be a typed window error, got {error:?}"
    );
  }
}

/// **The output is RAW.** The conversion's whole "there is no L2 to strip"
/// claim reduces to this: real embeddings have norms far from 1, so the caller's
/// normalization is doing work rather than repeating the graph's.
///
/// The recipe measured `‖e‖ ≈ 15.8 – 21.9` over its own corpus. The bound here
/// is deliberately much wider than that — this gate exists to catch a graph that
/// grew a normalization, which would read exactly 1.0, not to re-pin a number
/// measured on eight synthetic clips.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn embeddings_are_raw_and_not_unit_norm() {
  let embedder = Embedder::from_file(common::model_path()).expect("load embedder");
  for seed in [180.0f64, 240.0, 330.0] {
    let raw = embedder
      .embed(&common::synthetic_window(seed))
      .expect("embed");
    let norm = raw
      .iter()
      .map(|v| f64::from(*v) * f64::from(*v))
      .sum::<f64>()
      .sqrt();
    assert!(
      norm > 2.0,
      "a {seed} Hz window embedded to norm {norm:.4} — an L2 in the graph would read 1.0, \
       and coremlit's contract is that the CALLER normalizes"
    );
    assert!(raw.iter().all(|v| v.is_finite()), "non-finite component");
  }
}

/// Embedding is deterministic and input-dependent: the same window twice is the
/// same vector, and two different windows are not. A graph wired to a constant,
/// or one whose input is being ignored, passes every shape check above and fails
/// here.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn embedding_is_deterministic_and_depends_on_the_input() {
  let embedder = Embedder::from_file(common::model_path()).expect("load embedder");
  let a1 = embedder
    .embed(&common::synthetic_window(180.0))
    .expect("a1");
  let a2 = embedder
    .embed(&common::synthetic_window(180.0))
    .expect("a2");
  let b = embedder.embed(&common::synthetic_window(430.0)).expect("b");

  assert_eq!(a1, a2, "the same window must embed identically");
  let across = common::cosine(&a1, &b);
  assert!(
    across < 0.999,
    "two different windows must not embed to the same direction (cos {across:.6})"
  );
}

/// `prewarm` runs the whole path on synthetic audio, so a broken model surfaces
/// at prewarm time rather than on a caller's first real clip.
#[test]
#[ignore = "requires the staged identity model (IDENTITY_TEST_MODELS)"]
fn prewarm_exercises_the_prediction_path() {
  let embedder = Embedder::from_file(common::model_path()).expect("load embedder");
  embedder.prewarm().expect("prewarm");
}
