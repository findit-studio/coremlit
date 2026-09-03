use super::*;
use crate::DataType;
use soundevents_dataset::RatedSoundEvent;

#[test]
fn num_classes_matches_the_rated_dataset() {
  assert_eq!(NUM_CLASSES, 527);
  assert_eq!(RatedSoundEvent::events().len(), NUM_CLASSES);
  assert!(RatedSoundEvent::from_index(NUM_CLASSES - 1).is_some());
  assert!(RatedSoundEvent::from_index(NUM_CLASSES).is_none());
}

#[test]
fn window_is_ten_seconds_at_the_contract_rate() {
  assert_eq!(WINDOW_SAMPLES, 10 * SAMPLE_RATE_HZ as usize);
}

#[test]
fn default_compute_is_the_provisional_all() {
  assert_eq!(DEFAULT_COMPUTE, ComputeUnits::All);
}

// ── ClassifierOptions (rust-options-pattern, the granite shape) ────────────

#[test]
fn options_default_equals_new() {
  assert_eq!(ClassifierOptions::default(), ClassifierOptions::new());
  assert_eq!(ClassifierOptions::new().compute(), DEFAULT_COMPUTE);
}

#[test]
fn options_with_and_set_compute() {
  let opts = ClassifierOptions::new().with_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndNeuralEngine);
  let mut opts = ClassifierOptions::new();
  opts.set_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip_and_pinned_spelling() {
  let opts = ClassifierOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&opts).unwrap();
  assert_eq!(json, "{\"compute\":\"cpu_and_gpu\"}");
  let back: ClassifierOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(back, opts);
}

#[cfg(feature = "serde")]
#[test]
fn options_missing_compute_defaults_to_provisional_all() {
  let opts: ClassifierOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(opts.compute(), DEFAULT_COMPUTE);
}

#[cfg(feature = "serde")]
#[test]
fn options_unknown_compute_spelling_is_rejected() {
  assert!(serde_json::from_str::<ClassifierOptions>("{\"compute\":\"gpu\"}").is_err());
}

// ── Input validation + output guards (the hermetic classifier seams) ───────

#[test]
fn validate_rejects_empty_audio() {
  assert!(matches!(validate_window_input(&[]), Err(Error::EmptyAudio)));
}

#[test]
fn validate_rejects_overlong_audio_never_truncates() {
  let long = vec![0.0f32; WINDOW_SAMPLES + 1];
  assert!(matches!(
    validate_window_input(&long),
    Err(Error::AudioTooLong(e)) if e.len() == WINDOW_SAMPLES + 1 && e.max() == WINDOW_SAMPLES
  ));
}

#[test]
fn validate_reports_the_first_non_finite_sample() {
  let mut samples = vec![0.0f32; 100];
  samples[41] = f32::NAN;
  samples[43] = f32::INFINITY;
  assert!(matches!(
    validate_window_input(&samples),
    Err(Error::NonFiniteInput(41))
  ));
}

#[test]
fn classify_long_zero_k_guard_catches_non_finite_samples_beyond_one_window() {
  // classify_long's k == 0 arm must still reject a NaN/±∞ clip (previously it
  // returned Ok(vec![]) unconditionally once EmptyAudio was ruled out). The
  // guard must work on clips LONGER than WINDOW_SAMPLES — the whole point of
  // the long-clip path — so it calls check_finite_samples directly rather
  // than validate_window_input, which would reject on AudioTooLong first.
  let mut samples = vec![0.0f32; WINDOW_SAMPLES + 500];
  samples[WINDOW_SAMPLES + 300] = f32::NAN;
  assert!(matches!(
    check_finite_samples(&samples),
    Err(Error::NonFiniteInput(index)) if index == WINDOW_SAMPLES + 300
  ));
  assert!(check_finite_samples(&vec![0.0f32; WINDOW_SAMPLES + 500]).is_ok());
}

#[test]
fn validate_accepts_one_sample_and_a_full_window() {
  assert!(validate_window_input(&[0.5]).is_ok());
  assert!(validate_window_input(&vec![0.0f32; WINDOW_SAMPLES]).is_ok());
}

#[test]
fn finite_logit_check_reports_the_index() {
  let mut logits = vec![0.0f32; NUM_CLASSES];
  assert!(check_finite_logits(&logits).is_ok());
  logits[7] = f32::NEG_INFINITY;
  assert!(matches!(
    check_finite_logits(&logits),
    Err(Error::NonFiniteOutput(7))
  ));
}

// ── The door's own contract ────────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this door's `LoadContract` itself — its feature names, its
// element type, its geometry and its state clause — against descriptions built
// with the same fixture machinery, so a mis-stated contract is caught here and
// a mis-implemented checker is caught there.

use crate::{
  AxisRange, ComputeUnits, FeatureInfo, Model, ModelDescription, model::RawShapeConstraint,
};

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

/// The staged CED bundle's description: `mel [1, 64, 1001]` f32 in,
/// `logits [1, 527]` f32 out, no state — identical across all four sizes.
fn ced_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![fixed(names::LOGITS, &[1, NUM_CLASSES], DataType::F32)],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into this
/// module's errors — exactly what `Classifier::load` does after `Model::load`.
fn check(description: &ModelDescription) -> Result<()> {
  crate::model::contract::check_load_contract(description, &ced_contract())
    .map_err(contract_violation)
}

/// The contract states exactly the geometry the conversion emits.
#[test]
fn the_contract_accepts_the_converted_geometry() {
  assert!(check(&ced_description()).is_ok());
}

/// **The flexible-shape refusal.** [`crate::FeatureInfo::shape`] reports the
/// DEFAULT shape of a `RangeDims` input, so a flexible graph converted at
/// `[1, 64, 1001]` declares this contract's exact numbers AND reports `(d, 1)`
/// on every axis. Only the whole-feature verdict separates the two.
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
    vec![fixed(names::LOGITS, &[1, NUM_CLASSES], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::MEL),
    "{err}"
  );
}

/// An mlprogram converted at `compute_precision=FLOAT16` without an explicit
/// `dtype=np.float32` reports Float16 I/O, so this clause catches a
/// conversion-recipe regression at load rather than restating a constant.
#[test]
fn the_contract_refuses_a_right_shaped_fp16_graph() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F16)],
    vec![fixed(names::LOGITS, &[1, NUM_CLASSES], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::MEL),
    "{err}"
  );
}

/// A transposed `mel` of the same total size — the mutation no element-count
/// check can see.
#[test]
fn the_contract_refuses_a_transposed_shape_of_the_same_size() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_FRAMES, N_MELS], DataType::F32)],
    vec![fixed(names::LOGITS, &[1, NUM_CLASSES], DataType::F32)],
    Vec::new(),
  );
  assert!(matches!(
    check(&description),
    Err(Error::ContractMismatch(_))
  ));
}

/// The class count is the rated AudioSet set's, not another tagger's.
#[test]
fn the_contract_refuses_a_different_class_count() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![fixed(names::LOGITS, &[1, 521], DataType::F32)],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::LOGITS),
    "{err}"
  );
}

/// **A graph carrying `mel` plus another REQUIRED input** clears every
/// per-feature clause and then fails on every prediction, because
/// [`Classifier::raw_scores`] supplies `mel` and nothing else.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32),
      fixed("clip_mask", &[1, N_FRAMES], DataType::F32),
    ],
    vec![fixed(names::LOGITS, &[1, NUM_CLASSES], DataType::F32)],
    Vec::new(),
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableInput(name)) if name == "clip_mask"),
    "{:?}",
    check(&description)
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one, so it cannot make this door's prediction fail.
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
    vec![fixed(names::LOGITS, &[1, NUM_CLASSES], DataType::F32)],
    Vec::new(),
  );
  assert!(check(&description).is_ok());
}

/// An output the door READS that the graph may leave out: every geometry
/// clause passes and the prediction is still free to omit it.
#[test]
fn the_contract_refuses_an_optional_logits_output() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![multi_array(
      names::LOGITS,
      &[1, NUM_CLASSES],
      DataType::F32,
      true,
      2,
      vec![vec![1, NUM_CLASSES]],
      &[1, NUM_CLASSES],
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, Error::ContractMismatch(m) if m.feature() == names::LOGITS),
    "{err}"
  );
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful ML Program declaring
/// exactly `mel` and `logits` plus a state clears every per-feature clause AND
/// the input set — and only then meets [`Classifier::raw_scores`], which
/// predicts through the STATELESS API.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let description = ModelDescription::from_parts(
    vec![fixed(names::MEL, &[1, N_MELS, N_FRAMES], DataType::F32)],
    vec![fixed(names::LOGITS, &[1, NUM_CLASSES], DataType::F32)],
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  assert!(
    matches!(check(&description), Err(Error::UnsatisfiableState(name)) if name == "kv_cache")
  );
}

// ── The one gate here that loads a real artifact ───────────────────────────

/// **This door's `Checked::new` call site, pinned on a REAL model, in every
/// `cargo test`.**
///
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` is COMMITTED, so
/// unlike everything else in this repository that loads a model this needs no
/// staged artifact and carries no `#[ignore]`. Silero is a real, fixed-shape,
/// six-feature CoreML graph that is simply not this door's model — the exact
/// shape of a mis-pointed `model_path`.
#[test]
fn the_ced_contract_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; \
     looked for {}",
    bundle.display()
  );

  let model = Model::load(&bundle, ComputeUnits::CpuOnly).expect("the committed bundle loads");
  assert!(
    model.description().input(names::MEL).is_none(),
    "silero declares no `mel`, which is what makes it this gate's model"
  );

  let violation = crate::model::contract::Checked::new(model, &ced_contract())
    .expect_err("silero does not satisfy the CED contract");
  assert!(
    matches!(&violation, crate::model::contract::ContractViolation::Missing(m)
      if m.feature() == names::MEL),
    "expected `mel` missing, got {violation}"
  );
}
