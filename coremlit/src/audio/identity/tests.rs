//! Door gates.
//!
//! Until a CI shard stages the artifact, nothing in this repository loads the
//! identity model. So everything the load path decides is factored into DATA
//! and free functions over shapes, dtypes and slices — [`identity_contract`],
//! [`validate_window_input`], [`check_finite_embedding`] — and those are
//! exercised here in full, with no model present.
//!
//! The contract CHECKER's own clauses live with the checker
//! (`crate::model::contract`); what is pinned here is this door's own numbers
//! and its error vocabulary. One test does load a real artifact:
//! `the_identity_contract_refuses_the_vendored_silero_bundle` runs on every
//! `cargo test`, because that bundle is committed. What is still NOT covered is
//! that the redimnet `.mlmodelc` declares the contract this door states; that
//! is `tests/identity/model_io.rs`'s job, and it is `#[ignore]`d until the
//! artifact is staged.

use super::*;

// ── Geometry ───────────────────────────────────────────────────────────────

/// The window is 6 s at the contract rate, and the mel geometry it implies is
/// the graph's declared input shape.
#[test]
fn geometry_is_the_converted_contract() {
  assert_eq!(SAMPLE_RATE_HZ, 16_000);
  assert_eq!(WINDOW_SAMPLES, 6 * SAMPLE_RATE_HZ as usize);
  assert_eq!(N_MELS, 72);
  assert_eq!(N_FRAMES, 401);
  assert_eq!(EMBEDDING_DIM, 192);
}

/// This door's embedding dimension is NOT the diarization embedder's, and the
/// two are not interchangeable — a caller who mixes them gets a length error
/// rather than a wrong answer, and this pins the two numbers apart so a future
/// edit cannot quietly unify them.
#[cfg(feature = "speaker")]
#[test]
fn identity_and_diarization_embedding_dims_are_different_numbers() {
  assert_ne!(EMBEDDING_DIM, crate::audio::speaker::embed::EMBEDDING_DIM);
  assert_eq!(
    EMBEDDING_DIM,
    crate::audio::speaker::calibrate::Scoring::IdentityCosine.row_len(),
    "the identity score source must take exactly this door's raw row"
  );
}

/// The feature names the graph declares. Owned by the conversion recipe, which
/// this repository also owns, so they are pinned on both sides.
#[test]
fn feature_names_are_the_converted_ones() {
  assert_eq!(names::MEL, "mel");
  assert_eq!(names::EMBEDDING, "embedding");
}

/// The placement default is the MEASURED `CpuAndGpu`, and deliberately not the
/// `All` the crate's other doors take — see [`DEFAULT_COMPUTE`] for the
/// four-arm table. A change here without a new sweep is the thing this asserts
/// against.
#[test]
fn default_compute_is_the_measured_cpu_and_gpu() {
  assert_eq!(DEFAULT_COMPUTE, ComputeUnits::CpuAndGpu);
  assert_ne!(DEFAULT_COMPUTE, ComputeUnits::All);
}

// ── EmbedderOptions (rust-options-pattern) ─────────────────────────────────

#[test]
fn options_default_equals_new() {
  assert_eq!(EmbedderOptions::default(), EmbedderOptions::new());
  assert_eq!(EmbedderOptions::new().compute(), DEFAULT_COMPUTE);
}

#[test]
fn options_with_and_set_compute() {
  let opts = EmbedderOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndNeuralEngine);
  let mut opts = EmbedderOptions::new();
  opts.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip_and_pinned_spelling() {
  let opts = EmbedderOptions::new().with_compute(ComputeUnits::All);
  let json = serde_json::to_string(&opts).unwrap();
  assert_eq!(json, "{\"compute\":\"all\"}");
  let back: EmbedderOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

#[cfg(feature = "serde")]
#[test]
fn options_missing_compute_defaults_to_the_measured_placement() {
  let opts: EmbedderOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(opts.compute(), DEFAULT_COMPUTE);
}

#[cfg(feature = "serde")]
#[test]
fn options_unknown_compute_spelling_is_rejected() {
  assert!(serde_json::from_str::<EmbedderOptions>("{\"compute\":\"gpu\"}").is_err());
}

// ── The door's own contract ────────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this door's `LoadContract` itself — its feature names, its
// element type, its geometry and its state clause — against descriptions built
// with the same fixture machinery, so a mis-stated contract is caught here and
// a mis-implemented checker is caught there.

use crate::{AxisRange, FeatureInfo, ModelDescription, model::RawShapeConstraint};

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  multi_array(name, shape, dtype, false, 2, vec![shape.to_vec()], shape)
}

/// One multi-array feature, spelled out: the constraint's raw type code, its
/// enumerated shapes, and the axes its per-axis ranges pin.
fn multi_array(
  name: &str,
  shape: &[usize],
  dtype: DataType,
  optional: bool,
  raw_type: isize,
  enumerated: Vec<Vec<usize>>,
  pinned: &[usize],
) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    optional,
    Some(RawShapeConstraint::new(
      raw_type,
      enumerated,
      pinned.iter().map(|d| AxisRange::new(*d, 1)).collect(),
    )),
  )
}

/// The published redimnet bundle's description, as the Swift probe read it back
/// off the artifact: `mel` raw 2, `[1, 72, 401]`, ranges `1+1, 72+1, 401+1`,
/// Float32; `embedding [1, 192]`; `states = []`.
fn redimnet_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into this
/// module's errors — exactly what `Embedder::load` does after `Model::load`.
fn check(description: &ModelDescription) -> Result<()> {
  crate::model::contract::check_load_contract(description, &identity_contract())
    .map_err(contract_violation)
}

/// The contract states exactly the geometry the conversion emits.
#[test]
fn the_contract_accepts_the_converted_geometry() {
  assert!(check(&redimnet_description()).is_ok());
}

/// The feature names are the converted ones, and a differently spelled graph is
/// refused BY NAME rather than matched positionally.
#[test]
fn the_contract_refuses_a_differently_spelled_feature() {
  let description = ModelDescription::from_parts(
    vec![fixed("audio", &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m)
      if m.feature() == names::MEL && m.actual() == "missing"),
    "{err}"
  );
}

/// **The flexible-shape refusal, which is why this door has a contract at
/// all.** [`crate::FeatureInfo::shape`] reports the DEFAULT shape of a
/// `RangeDims` input, so a flexible graph converted at `[1, 72, 401]` declares
/// this contract's exact numbers AND reports `(d, 1)` on every axis. Only the
/// whole-feature verdict separates the two — and a flexible input is what takes
/// the graph off the accelerator, which is the one reason the conversion recipe
/// pins a fixed shape.
#[test]
fn the_contract_refuses_a_flexible_mel_declaring_its_exact_numbers() {
  let description = ModelDescription::from_parts(
    vec![multi_array(
      names::MEL,
      &[1, N_MELS, N_FRAMES],
      DataType::F32,
      false,
      3,
      Vec::new(),
      &[1, N_MELS, N_FRAMES],
    )],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::MEL),
    "{err}"
  );
  assert!(err.to_string().contains("range"), "{err}");
}

/// An mlprogram converted at `compute_precision=FLOAT16` without an explicit
/// `dtype=np.float32` reports Float16 I/O — measured on all three mlprogram
/// probes in `model/tests.rs` — so this clause is the one that catches a
/// conversion-recipe regression at load, not a restatement of a constant.
#[test]
fn the_contract_refuses_a_right_shaped_fp16_graph() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F16)],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::MEL),
    "{err}"
  );
  assert!(err.to_string().contains("float16"), "{err}");
}

/// A transposed `mel` of the same total size — the mutation no element-count
/// check can see.
#[test]
fn the_contract_refuses_a_transposed_shape_of_the_same_size() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_FRAMES, N_MELS], DataType::F32)],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    Vec::new(),
  );
  assert!(matches!(
    check(&description),
    Err(Error::ContractMismatch(_))
  ));
}

/// The embedding width is this door's, not the diarization embedder's: a 256-d
/// graph is a different model in a different lane, and the two are not
/// interchangeable.
#[test]
fn the_contract_refuses_the_diarization_embedding_width() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![fixed(names::EMBEDDING, &[1, 256], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::EMBEDDING),
    "{err}"
  );
}

/// **A graph carrying `mel` plus another REQUIRED input** clears every
/// per-feature clause and then fails on every prediction, because
/// [`Embedder::embed`] supplies `mel` and nothing else.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32),
      fixed("speaker_mask", &[1, N_FRAMES], DataType::F32),
    ],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    Vec::new(),
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableInput(name)) if name == "speaker_mask")
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one, so it cannot make this door's prediction fail. Optionality is exactly
/// the distinction this needs, and a count of inputs cannot make it.
#[test]
fn the_contract_accepts_an_extra_optional_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32),
      multi_array(
        "mask",
        &[1, N_FRAMES],
        DataType::F32,
        true,
        2,
        vec![vec![1, N_FRAMES]],
        &[1, N_FRAMES],
      ),
    ],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    Vec::new(),
  );
  assert!(check(&description).is_ok());
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful ML Program declaring
/// exactly `mel` and `embedding` plus a `kv_cache` state clears every
/// per-feature clause AND the input set — and only then meets
/// [`Embedder::embed`], which predicts through the STATELESS API. CoreML
/// requires a stateful model to receive an `MLState` on every prediction, so
/// that either fails or silently throws the persistence away.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![fixed(names::EMBEDDING, &[1, EMBEDDING_DIM], DataType::F32)],
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableState(name)) if name == "kv_cache")
  );
}

// ── The one gate here that loads a real artifact ───────────────────────────

/// **The single crate-wide call site of `Checked::new`, pinned on a REAL model,
/// in every `cargo test`.**
///
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` is COMMITTED — 1.1
/// MiB, staged by no download — so unlike everything else in this repository
/// that loads a model, this needs no artifact and carries no `#[ignore]`.
/// Silero is a real, fixed-shape, six-feature CoreML graph that is simply not
/// this door's model, which is the exact shape of a mis-pointed `model_path`.
///
/// What it pins that the fixture tests cannot: that the check actually RUNS
/// where `load` puts it, over a description CoreML itself produced. Delete the
/// `check_load_contract` call inside `Checked::new` and every fixture test
/// still passes — they call the checker directly — while this one goes green on
/// a model with no `mel` at all. It is the falsifier for the wiring.
#[test]
fn the_identity_contract_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; \
     looked for {}",
    bundle.display()
  );

  let model = Model::load(&bundle, ComputeUnits::CpuOnly).expect("the committed bundle loads");
  // The graph really does declare features, and none of them is `mel` — so the
  // refusal below is the contract's, not an empty description's.
  assert!(
    !model.description().inputs().is_empty(),
    "silero declares inputs"
  );
  assert!(
    model.description().input(names::MEL).is_none(),
    "silero declares no `mel`, which is what makes it this gate's model"
  );

  // **The tightening, checked against a real artifact rather than a fixture.**
  // `Fixed` now also requires the sole enumerated shape to BE the declared one
  // and one `(d, 1)` range per axis. Had a shipped fixed-shape export not
  // satisfied those, every such artifact would now be refused — and this
  // bundle's own `metadata.json` records `hasShapeFlexibility: "0"` for all six
  // of its features, so every one of them must still read `Fixed` here. The
  // vadkit gate that asserts the same thing needs `VADKIT_TEST_MODELS` and is
  // `#[ignore]`d; this one is not.
  let description = model.description();
  for feature in description.inputs().iter().chain(description.outputs()) {
    assert_eq!(
      feature.shape_constraint(),
      Some(crate::ShapeConstraint::Fixed),
      "{}: `hasShapeFlexibility: \"0\"` must still reach the snapshot as `Fixed`",
      feature.name()
    );
  }

  let violation = Checked::new(model, &identity_contract())
    .expect_err("silero does not satisfy the identity contract");
  assert!(
    matches!(&violation, ContractViolation::Missing(m) if m.feature() == names::MEL),
    "expected `mel` missing, got {violation}"
  );
}

// ── Input and output guards ────────────────────────────────────────────────

/// Exactly one window, or a typed refusal carrying both counts. Not padded and
/// not truncated at any length.
#[test]
fn validate_window_input_requires_an_exact_window() {
  assert!(validate_window_input(&vec![0.0f32; WINDOW_SAMPLES]).is_ok());
  for len in [0usize, 1, WINDOW_SAMPLES - 1, WINDOW_SAMPLES + 1] {
    let err = validate_window_input(&vec![0.0f32; len]).unwrap_err();
    assert!(
      matches!(err, Error::WindowLength(w) if w.got() == len && w.expected() == WINDOW_SAMPLES),
      "len {len}: got {err:?}"
    );
  }
}

/// A NaN or ±∞ sample is refused with its index, before it can reach the mel —
/// where the per-bin mean would spread it across every frame of that bin.
#[test]
fn validate_window_input_reports_the_first_non_finite_sample() {
  let mut samples = vec![0.0f32; WINDOW_SAMPLES];
  samples[4_100] = f32::INFINITY;
  samples[41] = f32::NAN;
  assert!(matches!(
    validate_window_input(&samples),
    Err(Error::NonFiniteInput(41))
  ));
}

/// The length check runs BEFORE the finite scan, so a short clip full of NaNs
/// is reported as the length error it is — the caller's actual mistake.
#[test]
fn validate_window_input_checks_length_before_finiteness() {
  let short = vec![f32::NAN; 16];
  assert!(matches!(
    validate_window_input(&short),
    Err(Error::WindowLength(_))
  ));
}

/// A non-finite model output is caught before it reaches the caller's L2, where
/// one NaN component would make all 192 of them NaN.
#[test]
fn finite_embedding_check_reports_the_index() {
  let mut row = [0.0f32; EMBEDDING_DIM];
  assert!(check_finite_embedding(&row).is_ok());
  row[7] = f32::NEG_INFINITY;
  assert!(matches!(
    check_finite_embedding(&row),
    Err(Error::NonFiniteOutput(7))
  ));
}
