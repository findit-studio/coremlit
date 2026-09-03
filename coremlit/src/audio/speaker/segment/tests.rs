use super::*;

// ---------------------------------------------------------------------
// multilabel: hermetic powerset-table + tie-handling tests (brief Step 1)
// ---------------------------------------------------------------------

/// One frame's logits with `class` dominant (10.0) and every other class
/// suppressed (-10.0) — unambiguous, not a tie.
fn row_for_class(class: usize) -> [f32; POWERSET_CLASSES] {
  let mut row = [-10.0f32; POWERSET_CLASSES];
  row[class] = 10.0;
  row
}

#[test]
fn multilabel_class_0_silence() {
  assert_eq!(multilabel(&row_for_class(0), 1), vec![0.0, 0.0, 0.0]);
}

#[test]
fn multilabel_class_1_speaker_a() {
  assert_eq!(multilabel(&row_for_class(1), 1), vec![1.0, 0.0, 0.0]);
}

#[test]
fn multilabel_class_2_speaker_b() {
  assert_eq!(multilabel(&row_for_class(2), 1), vec![0.0, 1.0, 0.0]);
}

#[test]
fn multilabel_class_3_speaker_c() {
  assert_eq!(multilabel(&row_for_class(3), 1), vec![0.0, 0.0, 1.0]);
}

#[test]
fn multilabel_class_4_speakers_a_and_b() {
  assert_eq!(multilabel(&row_for_class(4), 1), vec![1.0, 1.0, 0.0]);
}

#[test]
fn multilabel_class_5_speakers_a_and_c() {
  assert_eq!(multilabel(&row_for_class(5), 1), vec![1.0, 0.0, 1.0]);
}

#[test]
fn multilabel_class_6_speakers_b_and_c() {
  assert_eq!(multilabel(&row_for_class(6), 1), vec![0.0, 1.0, 1.0]);
}

/// dia's argmax loop seeds `max` from class 0 and only updates on strict
/// `>` (`diarization/src/segment/powerset.rs:69-76`), so an exact tie
/// across every class resolves to class 0 (silence) — the seeded value,
/// never displaced by an equal value.
#[test]
fn multilabel_tie_all_classes_breaks_to_silence() {
  let row = [0.0f32; POWERSET_CLASSES];
  assert_eq!(multilabel(&row, 1), vec![0.0, 0.0, 0.0]);
}

/// Same rule for a tie not involving class 0: classes 2 and 5 tied at the
/// maximum, class 2 (the lower index) wins because class 5's equal value
/// does not satisfy strict `>` against the running max.
#[test]
fn multilabel_tie_breaks_to_lowest_class_index() {
  let mut row = [-10.0f32; POWERSET_CLASSES];
  row[2] = 5.0;
  row[5] = 5.0;
  assert_eq!(multilabel(&row, 1), vec![0.0, 1.0, 0.0]); // class 2 = speaker B
}

/// Multi-frame buffers decode frame-major (`frame * SEG_NUM_SLOTS +
/// slot`), matching dia's `segmentations` layout for one chunk
/// (`diarization/src/offline/owned.rs:496`).
#[test]
fn multilabel_multi_frame_layout_is_frame_major() {
  let mut logits = Vec::new();
  logits.extend_from_slice(&row_for_class(1)); // frame 0: speaker A
  logits.extend_from_slice(&row_for_class(2)); // frame 1: speaker B
  let out = multilabel(&logits, 2);
  assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
}

/// The length contract is a hard assert in every build profile — dia's
/// inline decode panics on a short buffer via direct indexing
/// (`diarization/src/offline/owned.rs:482`); silently truncating would
/// misalign the downstream `segmentations` buffer instead.
#[test]
#[should_panic(expected = "logits.len() must equal num_frames * POWERSET_CLASSES")]
fn multilabel_panics_on_short_logits() {
  let _ = multilabel(&row_for_class(0), 2);
}

#[test]
#[should_panic(expected = "logits.len() must equal num_frames * POWERSET_CLASSES")]
fn multilabel_panics_on_long_logits() {
  let mut logits = Vec::new();
  logits.extend_from_slice(&row_for_class(0));
  logits.extend_from_slice(&row_for_class(1));
  let _ = multilabel(&logits, 1);
}

// ---------------------------------------------------------------------
// check_input_length / check_finite: hermetic coverage of the two
// `infer`-boundary checks the brief calls "the product story" — extracted
// so they're directly testable without a loaded model.
// ---------------------------------------------------------------------

#[test]
fn check_input_length_accepts_exact_length() {
  assert_eq!(check_input_length(SEG_CHUNK_SAMPLES), Ok(()));
}

#[test]
fn check_input_length_rejects_short_input() {
  assert_eq!(
    check_input_length(100),
    Err(InferError::InputLength(InputLength::new(
      100,
      SEG_CHUNK_SAMPLES
    )))
  );
}

#[test]
fn check_input_length_rejects_long_input() {
  let got = SEG_CHUNK_SAMPLES + 1;
  assert_eq!(
    check_input_length(got),
    Err(InferError::InputLength(InputLength::new(
      got,
      SEG_CHUNK_SAMPLES
    )))
  );
}

#[test]
fn check_output_shape_accepts_correct_shape() {
  assert_eq!(check_output_shape(&[1, 589, POWERSET_CLASSES], 589), Ok(()));
}

/// The exact corruption `check_output_shape` exists to catch: axes swapped
/// (`[1, POWERSET_CLASSES, num_frames]` instead of `[1, num_frames,
/// POWERSET_CLASSES]`) carries the identical element count as the correct
/// shape, so a total-element-count check alone (as `MultiArray::copy_into`
/// performs) would not detect it.
#[test]
fn check_output_shape_rejects_swapped_axes() {
  assert_eq!(
    check_output_shape(&[1, POWERSET_CLASSES, 589], 589),
    Err(InferError::OutputShape(OutputShape::new(
      vec![1, POWERSET_CLASSES, 589],
      vec![1, 589, POWERSET_CLASSES]
    )))
  );
}

#[test]
fn check_output_shape_rejects_wrong_rank() {
  assert_eq!(
    check_output_shape(&[589, POWERSET_CLASSES], 589),
    Err(InferError::OutputShape(OutputShape::new(
      vec![589, POWERSET_CLASSES],
      vec![1, 589, POWERSET_CLASSES]
    )))
  );
}

#[test]
fn check_output_shape_rejects_wrong_frame_count() {
  assert_eq!(
    check_output_shape(&[1, 590, POWERSET_CLASSES], 589),
    Err(InferError::OutputShape(OutputShape::new(
      vec![1, 590, POWERSET_CLASSES],
      vec![1, 589, POWERSET_CLASSES]
    )))
  );
}

#[test]
fn check_output_shape_rejects_wrong_batch_dim() {
  assert_eq!(
    check_output_shape(&[2, 589, POWERSET_CLASSES], 589),
    Err(InferError::OutputShape(OutputShape::new(
      vec![2, 589, POWERSET_CLASSES],
      vec![1, 589, POWERSET_CLASSES]
    )))
  );
}

#[test]
fn check_finite_accepts_all_finite() {
  assert_eq!(check_finite(&[0.0, 1.0, -1.0]), Ok(()));
}

#[test]
fn check_finite_rejects_nan_at_reported_index() {
  assert_eq!(
    check_finite(&[0.0, f32::NAN, 2.0]),
    Err(InferError::NonFiniteOutput(1))
  );
}

#[test]
fn check_finite_rejects_positive_infinity() {
  assert_eq!(
    check_finite(&[f32::INFINITY]),
    Err(InferError::NonFiniteOutput(0))
  );
}

#[test]
fn check_finite_rejects_negative_infinity() {
  assert_eq!(
    check_finite(&[0.0, 0.0, f32::NEG_INFINITY]),
    Err(InferError::NonFiniteOutput(2))
  );
}

#[test]
fn check_finite_reports_first_offending_index() {
  assert_eq!(
    check_finite(&[f32::NAN, f32::INFINITY]),
    Err(InferError::NonFiniteOutput(0))
  );
}

// M2: the input-side scan `infer` now runs BEFORE the CoreML call, so a NaN
// sample surfaces as `NonFiniteInput` instead of reaching the model. Mirrors
// the embed module's identical `check_finite_input` and dia's own input guard.

#[test]
fn check_finite_input_accepts_all_finite() {
  assert_eq!(check_finite_input(&[0.0, 1.0, -1.0]), Ok(()));
}

#[test]
fn check_finite_input_rejects_nan_at_reported_index() {
  assert_eq!(
    check_finite_input(&[0.0, f32::NAN, 2.0]),
    Err(InferError::NonFiniteInput(1))
  );
}

#[test]
fn check_finite_input_rejects_positive_infinity() {
  assert_eq!(
    check_finite_input(&[f32::INFINITY]),
    Err(InferError::NonFiniteInput(0))
  );
}

#[test]
fn check_finite_input_rejects_negative_infinity() {
  assert_eq!(
    check_finite_input(&[0.0, 0.0, f32::NEG_INFINITY]),
    Err(InferError::NonFiniteInput(2))
  );
}

// ---------------------------------------------------------------------
// SegmentModelOptions
// ---------------------------------------------------------------------

#[test]
fn options_new_defaults_to_all_compute() {
  assert_eq!(SegmentModelOptions::new().compute(), ComputeUnits::All);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(SegmentModelOptions::default(), SegmentModelOptions::new());
}

#[test]
fn options_with_compute_overrides() {
  let options = SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_set_compute_in_place() {
  let mut options = SegmentModelOptions::new();
  options.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(options.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_compute_defaults_to_all() {
  let options: SegmentModelOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(options.compute(), ComputeUnits::All);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips_explicit_compute() {
  let options: SegmentModelOptions = serde_json::from_str(r#"{"compute":"cpu_only"}"#).unwrap();
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
  let json = serde_json::to_string(&options).unwrap();
  assert!(json.contains("cpu_only"), "round-tripped json: {json}");
}

// ---------------------------------------------------------------------
// SegmentModel: model-gated (brief Step 2) — requires a local
// pyannote_segmentation.mlmodelc (SPEAKERKIT_TEST_MODELS or
// Models/speakerkit/, same convention as tests/model_io.rs's `common`
// module). Duplicated here in miniature because unit tests under `src/`
// cannot import the separate `tests/` integration-test crate.
// ---------------------------------------------------------------------

fn models_dir() -> std::path::PathBuf {
  std::env::var_os("SPEAKERKIT_TEST_MODELS").map_or_else(
    || crate::tests::models_root().join("speakerkit"),
    std::path::PathBuf::from,
  )
}

fn seg_path() -> std::path::PathBuf {
  models_dir().join("pyannote_segmentation.mlmodelc")
}

/// Loads the real segmentation model with `ComputeUnits::CpuOnly` —
/// matching `tests/model_io.rs`'s introspection convention (every load
/// there also uses `ComputeUnits::CpuOnly`): deterministic, no ANE
/// compile-latency variance across runs. `DEFAULT_SEGMENT_COMPUTE`
/// (`ComputeUnits::All`) stays the production default.
fn load_seg_model() -> SegmentModel {
  SegmentModel::from_file_with(
    seg_path(),
    SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load pyannote_segmentation.mlmodelc")
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn from_file_loads_and_reports_frame_count() {
  let model = load_seg_model();
  // Ground truth pinned by
  // `tests/model_io.rs::pyannote_segmentation_io_matches_spec`: 589 frames.
  assert_eq!(model.num_frames(), 589);
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn from_file_rejects_wrong_contract_model() {
  // wespeaker_v2.mlmodelc has no `audio` input at all (its inputs are
  // `waveform`/`mask`) — a real, locally-available model with a
  // definitely-mismatched contract, exercising `ContractMismatch` without
  // needing a second downloaded fixture.
  let path = models_dir().join("wespeaker_v2.mlmodelc");
  let err = SegmentModel::from_file(path).expect_err("wrong contract must be rejected");
  assert!(matches!(
    err,
    ModelError::ContractMismatch(m) if m.feature() == "audio"
  ));
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn infer_rejects_wrong_input_length() {
  let model = load_seg_model();
  let err = model
    .infer(&[0.0f32; 100])
    .expect_err("wrong length must be rejected");
  assert_eq!(
    err,
    InferError::InputLength(InputLength::new(100, SEG_CHUNK_SAMPLES))
  );
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn infer_produces_correctly_shaped_finite_logits() {
  let model = load_seg_model();
  let samples = vec![0.0f32; SEG_CHUNK_SAMPLES];
  let logits = model.infer(&samples).expect("infer on silence");
  assert_eq!(logits.len(), model.num_frames() * POWERSET_CLASSES);
  assert!(logits.iter().all(|v| v.is_finite()), "all logits finite");
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn infer_is_deterministic_across_repeated_calls() {
  let model = load_seg_model();
  // Small-amplitude non-zero signal, not pure silence, so this exercises
  // real signal-path compute rather than just a bias/floor.
  let samples: Vec<f32> = (0..SEG_CHUNK_SAMPLES)
    .map(|i| 0.01 * (i as f32 * 0.001).sin())
    .collect();
  let first = model.infer(&samples).expect("first infer");
  let second = model.infer(&samples).expect("second infer");
  assert_eq!(first, second, "repeated infer must be bit-identical");
}

// ── The door's own contract ────────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this door's `LoadContract` itself, against descriptions built
// with the same fixture machinery.

use crate::{AxisRange, FeatureInfo, ModelDescription, model::RawShapeConstraint};

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    false,
    Some(RawShapeConstraint::new(
      2,
      vec![shape.to_vec()],
      shape.iter().map(|d| AxisRange::new(*d, 1)).collect(),
    )),
  )
}

/// A `RangeDims` multi-array feature; `shape` is the DEFAULT.
fn ranged(name: &str, shape: &[usize], dtype: DataType, ranges: &[AxisRange]) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    false,
    Some(RawShapeConstraint::new(3, Vec::new(), ranges.to_vec())),
  )
}

/// The frame count the staged `pyannote_segmentation.mlmodelc` declares.
/// Spelled here rather than imported: the door reads this number off the
/// artifact and hardcodes it nowhere.
const MEASURED_FRAMES: usize = 589;

/// The staged artifact's description, as `Model::load` reads it back:
/// `audio [1, 1, 160_000]` f32 `Fixed` in, `segments [1, 589, 7]` f32 `Fixed`
/// out, no state and no extra feature in either direction.
fn pyannote_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(
      names::AUDIO,
      &[1, 1, SEG_CHUNK_SAMPLES],
      DataType::F32,
    )],
    vec![fixed(
      names::SEGMENTS,
      &[1, MEASURED_FRAMES, POWERSET_CLASSES],
      DataType::F32,
    )],
    Vec::new(),
  )
}

/// This door's contract, run against `description` and mapped into this
/// module's errors — exactly what `SegmentModel::from_file_with` does after
/// `Model::load`.
fn check(description: &ModelDescription) -> Result<(), ModelError> {
  crate::model::contract::check_load_contract(description, &segment_contract())
    .map_err(crate::audio::speaker::error::contract_violation)
}

#[test]
fn the_contract_accepts_the_staged_pyannote_description() {
  let description = pyannote_description();
  assert_eq!(check(&description), Ok(()));
  assert_eq!(
    description
      .output(names::SEGMENTS)
      .expect("segments")
      .shape()[1],
    MEASURED_FRAMES
  );
}

/// **FALSIFIER (red first) — issue #137's defect (ii).**
///
/// The check this contract replaced read `segments.shape()[1]` and required
/// only `>= 1`. `FeatureInfo::shape` reports the DEFAULT shape of a flexible
/// feature, so a `segments` head declared over 1..=4096 frames and converted at
/// 589 satisfied every clause it made, bound `F = 589`, and made every `infer`
/// allocate and range-check its output at a length the graph does not require.
#[test]
fn the_contract_refuses_a_flexible_segments_whose_default_is_the_artifacts_589() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::AUDIO,
      &[1, 1, SEG_CHUNK_SAMPLES],
      DataType::F32,
    )],
    vec![ranged(
      names::SEGMENTS,
      &[1, MEASURED_FRAMES, POWERSET_CLASSES],
      DataType::F32,
      &[
        AxisRange::new(1, 1),
        AxisRange::inclusive(1, 4096),
        AxisRange::new(POWERSET_CLASSES, 1),
      ],
    )],
    Vec::new(),
  );
  assert_eq!(
    description
      .output(names::SEGMENTS)
      .expect("segments")
      .shape()[1],
    MEASURED_FRAMES
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, ModelError::ContractMismatch(m)
      if m.feature() == names::SEGMENTS && m.actual() == "range" && m.expected() == "fixed"),
    "{err}"
  );
}

/// A zero-frame `segments` head is pinned, so it satisfies "exactly one size" —
/// and every `infer` would return an empty `Vec` with no error.
#[test]
fn the_contract_refuses_a_zero_frame_segments_head() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::AUDIO,
      &[1, 1, SEG_CHUNK_SAMPLES],
      DataType::F32,
    )],
    vec![fixed(
      names::SEGMENTS,
      &[1, 0, POWERSET_CLASSES],
      DataType::F32,
    )],
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, ModelError::ContractMismatch(m) if m.feature() == names::SEGMENTS),
    "{err}"
  );
}

/// **Defect (i).** A graph carrying `audio` plus another REQUIRED input clears
/// every per-feature clause and then fails every prediction.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed(names::AUDIO, &[1, 1, SEG_CHUNK_SAMPLES], DataType::F32),
      fixed("speaker_prior", &[1, POWERSET_CLASSES], DataType::F32),
    ],
    pyannote_description().outputs().to_vec(),
    Vec::new(),
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, ModelError::UnsatisfiableInput(name) if name == "speaker_prior"),
    "{err}"
  );
}

/// **State is not an input**, so a stateful graph declaring exactly `audio` and
/// `segments` clears every other clause — and then meets a door predicting
/// through the stateless API.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let base = pyannote_description();
  let description = ModelDescription::from_parts(
    base.inputs().to_vec(),
    base.outputs().to_vec(),
    vec![fixed("lstm_state", &[1, 128], DataType::F32)],
  );
  let err = check(&description).unwrap_err();
  assert!(
    matches!(&err, ModelError::UnsatisfiableState(name) if name == "lstm_state"),
    "{err}"
  );
}

/// **The wiring, pinned on a REAL model, in every `cargo test`** — the
/// committed silero bundle, which is a real fixed-shape CoreML graph and is
/// simply not this door's model. See the embed door's twin for what it covers
/// that the fixtures cannot.
#[test]
fn the_segment_contract_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; looked for {}",
    bundle.display()
  );
  let err = SegmentModel::from_file(&bundle).expect_err("silero is not this door's model");
  assert!(
    matches!(&err, ModelError::ContractMismatch(m)
      if m.feature() == names::AUDIO && m.actual() == "missing"),
    "{err}"
  );
}

/// `base` with one axis of one named feature made one larger — see the embed
/// door's twin for why the sweep below is one test rather than one per axis.
fn with_axis_bumped(base: &ModelDescription, feature: &str, axis: usize) -> ModelDescription {
  let bump = |declared: &FeatureInfo| -> FeatureInfo {
    if declared.name() != feature {
      return declared.clone();
    }
    let mut shape = declared.shape().to_vec();
    shape[axis] += 1;
    fixed(
      declared.name(),
      &shape,
      declared.data_type().expect("a multi-array feature"),
    )
  };
  ModelDescription::from_parts(
    base.inputs().iter().map(bump).collect(),
    base.outputs().iter().map(bump).collect(),
    base.states().to_vec(),
  )
}

/// **Every axis clause is load-bearing, and the free one is named.** Perturbs
/// every axis of every named feature and requires a refusal, except the frame
/// count this door reads back. Reds in both directions — see the embed door's
/// twin.
#[test]
fn every_axis_is_pinned_except_the_frame_count_the_door_reads_back() {
  /// The one axis this door READS: `segments`' frame count.
  const FREE: &[(&str, usize)] = &[(names::SEGMENTS, 1)];

  let base = pyannote_description();
  let mut perturbations = 0_usize;
  for declared in base.inputs().iter().chain(base.outputs()) {
    for axis in 0..declared.shape().len() {
      let perturbed = with_axis_bumped(&base, declared.name(), axis);
      let free = FREE.contains(&(declared.name(), axis));
      assert_eq!(
        check(&perturbed).is_ok(),
        free,
        "`{}` axis {axis}: the contract {} it",
        declared.name(),
        if free { "must accept" } else { "must refuse" }
      );
      perturbations += 1;
    }
  }
  // Non-vacuous: one rank-3 input and one rank-3 output.
  assert_eq!(perturbations, 6);
}

/// **Every named feature's element type is pinned**, which no check this door
/// replaced stated for more than the two it happened to look at. Each is
/// re-declared at a type the door does not write, and every one must be
/// refused.
#[test]
fn every_named_features_element_type_is_pinned() {
  let base = pyannote_description();

  let mut checked = 0_usize;
  for declared in base.inputs().iter().chain(base.outputs()) {
    let swap = |other: &FeatureInfo| -> FeatureInfo {
      if other.name() == declared.name() {
        fixed(other.name(), other.shape(), DataType::F16)
      } else {
        other.clone()
      }
    };
    let perturbed = ModelDescription::from_parts(
      base.inputs().iter().map(swap).collect(),
      base.outputs().iter().map(swap).collect(),
      base.states().to_vec(),
    );
    assert!(
      matches!(check(&perturbed), Err(ModelError::ContractMismatch(m))
        if m.feature() == declared.name()
          && m.expected() == "float32"
          && m.actual() == "float16"),
      "`{}` re-declared float16 must be refused",
      declared.name()
    );
    checked += 1;
  }
  assert_eq!(checked, 2);
}
