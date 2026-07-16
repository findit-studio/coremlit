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
    Err(InferError::InputLength {
      got: 100,
      expected: SEG_CHUNK_SAMPLES,
    })
  );
}

#[test]
fn check_input_length_rejects_long_input() {
  let got = SEG_CHUNK_SAMPLES + 1;
  assert_eq!(
    check_input_length(got),
    Err(InferError::InputLength {
      got,
      expected: SEG_CHUNK_SAMPLES,
    })
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
    Err(InferError::OutputShape {
      got: vec![1, POWERSET_CLASSES, 589],
      expected: vec![1, 589, POWERSET_CLASSES],
    })
  );
}

#[test]
fn check_output_shape_rejects_wrong_rank() {
  assert_eq!(
    check_output_shape(&[589, POWERSET_CLASSES], 589),
    Err(InferError::OutputShape {
      got: vec![589, POWERSET_CLASSES],
      expected: vec![1, 589, POWERSET_CLASSES],
    })
  );
}

#[test]
fn check_output_shape_rejects_wrong_frame_count() {
  assert_eq!(
    check_output_shape(&[1, 590, POWERSET_CLASSES], 589),
    Err(InferError::OutputShape {
      got: vec![1, 590, POWERSET_CLASSES],
      expected: vec![1, 589, POWERSET_CLASSES],
    })
  );
}

#[test]
fn check_output_shape_rejects_wrong_batch_dim() {
  assert_eq!(
    check_output_shape(&[2, 589, POWERSET_CLASSES], 589),
    Err(InferError::OutputShape {
      got: vec![2, 589, POWERSET_CLASSES],
      expected: vec![1, 589, POWERSET_CLASSES],
    })
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
    Err(InferError::NonFiniteOutput { index: 1 })
  );
}

#[test]
fn check_finite_rejects_positive_infinity() {
  assert_eq!(
    check_finite(&[f32::INFINITY]),
    Err(InferError::NonFiniteOutput { index: 0 })
  );
}

#[test]
fn check_finite_rejects_negative_infinity() {
  assert_eq!(
    check_finite(&[0.0, 0.0, f32::NEG_INFINITY]),
    Err(InferError::NonFiniteOutput { index: 2 })
  );
}

#[test]
fn check_finite_reports_first_offending_index() {
  assert_eq!(
    check_finite(&[f32::NAN, f32::INFINITY]),
    Err(InferError::NonFiniteOutput { index: 0 })
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
    Err(InferError::NonFiniteInput { index: 1 })
  );
}

#[test]
fn check_finite_input_rejects_positive_infinity() {
  assert_eq!(
    check_finite_input(&[f32::INFINITY]),
    Err(InferError::NonFiniteInput { index: 0 })
  );
}

#[test]
fn check_finite_input_rejects_negative_infinity() {
  assert_eq!(
    check_finite_input(&[0.0, 0.0, f32::NEG_INFINITY]),
    Err(InferError::NonFiniteInput { index: 2 })
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
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("speakerkit")
    },
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
    ModelError::ContractMismatch {
      feature: "audio",
      ..
    }
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
    InferError::InputLength {
      got: 100,
      expected: SEG_CHUNK_SAMPLES,
    }
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
