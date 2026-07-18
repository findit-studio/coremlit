use super::*;

// ---------------------------------------------------------------------
// repeat_pad_f32: hermetic repeat-padding math (brief Step 1) — exact
// hand-computed expected rows for short inputs, plus an empirical
// cross-check against a literal transliteration of FluidAudio's Swift
// doubling-copy loop.
// ---------------------------------------------------------------------

#[test]
fn repeat_pad_f32_empty_source_returns_zeros() {
  assert_eq!(repeat_pad_f32(&[], 5), vec![0.0; 5]);
}

#[test]
fn repeat_pad_f32_empty_source_and_zero_target_returns_empty() {
  assert_eq!(repeat_pad_f32(&[], 0), Vec::<f32>::new());
}

#[test]
fn repeat_pad_f32_exact_length_is_identity() {
  assert_eq!(
    repeat_pad_f32(&[1.0, 2.0, 3.0, 4.0], 4),
    vec![1.0, 2.0, 3.0, 4.0]
  );
}

/// Hand-computed: source `[1,2,3]` repeat-tiled to 10 elements. Traced by
/// hand against FluidAudio's doubling-copy loop (module doc's
/// `repeat_pad_f32` proof): fill=[1,2,3], double to
/// [1,2,3,1,2,3,?,?,?,?] (copy 3), then copy min(6,4)=4 from the start to
/// reach [1,2,3,1,2,3,1,2,3,1].
#[test]
fn repeat_pad_f32_short_source_tiles_periodically() {
  assert_eq!(
    repeat_pad_f32(&[1.0, 2.0, 3.0], 10),
    vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
  );
}

#[test]
fn repeat_pad_f32_single_element_source_fills_uniformly() {
  assert_eq!(repeat_pad_f32(&[7.0], 4), vec![7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn repeat_pad_f32_longer_source_truncates() {
  assert_eq!(
    repeat_pad_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], 3),
    vec![1.0, 2.0, 3.0]
  );
}

#[test]
fn repeat_pad_f32_zero_target_returns_empty() {
  assert_eq!(repeat_pad_f32(&[1.0, 2.0], 0), Vec::<f32>::new());
}

/// Literal Rust transliteration of FluidAudio's Swift doubling-copy loop
/// (`fillWaveformBuffer`/`fillMaskBufferOptimized`,
/// `EmbeddingExtractor.swift#L117-199` at the pinned SHA — module doc) —
/// used ONLY to empirically cross-check that `repeat_pad_f32`'s closed-form
/// periodic-tile output is bit-identical to the doubling-copy algorithm's
/// output, per this task's brief instruction to verify the loop-pad
/// behavior empirically, not just by proof:
///
/// ```swift
/// while sampleCount < requiredCount {
///     let copyCount = min(sampleCount, requiredCount - sampleCount)
///     vDSP_mmov(ptr, ptr.advanced(by: sampleCount), vDSP_Length(copyCount), ...)
///     sampleCount += copyCount
/// }
/// ```
fn doubling_copy_simulation(source: &[f32], target_len: usize) -> Vec<f32> {
  let mut buf = vec![0.0f32; target_len];
  let mut sample_count = source.len().min(target_len);
  buf[..sample_count].copy_from_slice(&source[..sample_count]);
  if sample_count == 0 {
    return buf;
  }
  while sample_count < target_len {
    let copy_count = sample_count.min(target_len - sample_count);
    let (filled, rest) = buf.split_at_mut(sample_count);
    rest[..copy_count].copy_from_slice(&filled[..copy_count]);
    sample_count += copy_count;
  }
  buf
}

#[test]
fn doubling_copy_simulation_matches_repeat_pad_f32_non_power_of_two_lengths() {
  // Several non-power-of-2 (source, target_len) pairs — the case the
  // module doc's equivalence proof is least obviously true for at a
  // glance (the proof only needs "multiple of source.len()", not "power
  // of two", but empirical cross-checking beats trusting the proof alone).
  let cases: &[(&[f32], usize)] = &[
    (&[1.0, 2.0, 3.0], 10),
    (&[1.0, 2.0, 3.0, 4.0, 5.0], 13),
    (&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 20),
    (&[9.0], 7),
    (&[1.0, 2.0], 2),
    (&[1.0, 2.0, 3.0, 4.0, 5.0], 3),
    (&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1),
  ];
  for &(source, target_len) in cases {
    assert_eq!(
      repeat_pad_f32(source, target_len),
      doubling_copy_simulation(source, target_len),
      "mismatch for source={source:?}, target_len={target_len}"
    );
  }
}

#[test]
fn doubling_copy_simulation_matches_repeat_pad_f32_on_empty_source() {
  assert_eq!(repeat_pad_f32(&[], 6), doubling_copy_simulation(&[], 6));
}

// ---------------------------------------------------------------------
// mask_row_f32: hermetic mask f32 conversion (brief Step 1).
// ---------------------------------------------------------------------

#[test]
fn mask_row_f32_converts_true_and_false() {
  assert_eq!(
    mask_row_f32(&[true, false, true, true, false]),
    vec![1.0, 0.0, 1.0, 1.0, 0.0]
  );
}

#[test]
fn mask_row_f32_empty_mask_is_empty() {
  assert_eq!(mask_row_f32(&[]), Vec::<f32>::new());
}

#[test]
fn mask_row_f32_all_false() {
  assert_eq!(mask_row_f32(&[false, false]), vec![0.0, 0.0]);
}

#[test]
fn mask_row_f32_all_true() {
  assert_eq!(mask_row_f32(&[true, true, true]), vec![1.0, 1.0, 1.0]);
}

/// Composition of `mask_row_f32` + `repeat_pad_f32` — the exact pipeline
/// `build_masks` runs per slot: a short boolean mask, converted then
/// repeat-padded. Hand-computed: `[true,false]` -> `[1.0,0.0]`, tiled to 5
/// (period 2) -> `[1.0,0.0,1.0,0.0,1.0]`.
#[test]
fn mask_row_padding_short_mask_tiles_after_conversion() {
  let converted = mask_row_f32(&[true, false]);
  assert_eq!(repeat_pad_f32(&converted, 5), vec![1.0, 0.0, 1.0, 0.0, 1.0]);
}

// ---------------------------------------------------------------------
// build_waveform / build_masks: hermetic batched-buffer assembly.
// ---------------------------------------------------------------------

#[test]
fn build_waveform_repeats_the_same_row_in_every_slot() {
  let out = build_waveform(&[1.0, 2.0]);
  assert_eq!(
    out.len(),
    EMBED_SLOTS * crate::audio::speaker::segment::SEG_CHUNK_SAMPLES
  );
  for slot in out
    .as_chunks::<{ crate::audio::speaker::segment::SEG_CHUNK_SAMPLES }>()
    .0
  {
    assert_eq!(slot[0], 1.0);
    assert_eq!(slot[1], 2.0);
    assert_eq!(slot[2], 1.0); // period-2 tiling continues into slot's tail
  }
}

#[test]
fn build_masks_pads_each_slot_independently() {
  let mask_a = [true, false, true];
  let mask_b: [bool; 0] = [];
  let mask_c = [true];
  let masks: [&[bool]; EMBED_SLOTS] = [&mask_a, &mask_b, &mask_c];
  let out = build_masks(&masks, 4);
  assert_eq!(out.len(), EMBED_SLOTS * 4);
  assert_eq!(&out[0..4], &[1.0, 0.0, 1.0, 1.0]); // period-3 tiling
  assert_eq!(&out[4..8], &[0.0, 0.0, 0.0, 0.0]); // empty -> zero-fill
  assert_eq!(&out[8..12], &[1.0, 1.0, 1.0, 1.0]); // period-1 tiling
}

// ---------------------------------------------------------------------
// check_mask_active: hermetic dia-parity mask-validity check.
// ---------------------------------------------------------------------

#[test]
fn check_mask_active_accepts_one_active_frame() {
  assert_eq!(check_mask_active(&[false, true, false]), Ok(()));
}

#[test]
fn check_mask_active_accepts_all_active() {
  assert_eq!(check_mask_active(&[true, true]), Ok(()));
}

#[test]
fn check_mask_active_rejects_all_false() {
  assert_eq!(
    check_mask_active(&[false, false, false]),
    Err(InferError::EmptyMask)
  );
}

#[test]
fn check_mask_active_rejects_empty_mask() {
  assert_eq!(check_mask_active(&[]), Err(InferError::EmptyMask));
}

// ---------------------------------------------------------------------
// check_finite_input / check_finite_output: hermetic NonFinite scans.
// ---------------------------------------------------------------------

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

#[test]
fn check_finite_output_accepts_all_finite() {
  assert_eq!(check_finite_output(&[0.0, 1.0, -1.0]), Ok(()));
}

#[test]
fn check_finite_output_rejects_nan_at_reported_index() {
  assert_eq!(
    check_finite_output(&[0.0, f32::NAN, 2.0]),
    Err(InferError::NonFiniteOutput { index: 1 })
  );
}

#[test]
fn check_finite_output_reports_first_offending_index() {
  assert_eq!(
    check_finite_output(&[f32::NAN, f32::INFINITY]),
    Err(InferError::NonFiniteOutput { index: 0 })
  );
}

// ---------------------------------------------------------------------
// check_output_shape: hermetic output-shape validation (T2 commit
// fcbce74's precedent — review-queue item 3).
// ---------------------------------------------------------------------

#[test]
fn check_output_shape_accepts_correct_shape() {
  assert_eq!(check_output_shape(&[EMBED_SLOTS, EMBEDDING_DIM]), Ok(()));
}

/// The exact corruption this guard exists to catch: axes swapped
/// (`[EMBEDDING_DIM, EMBED_SLOTS]` instead of `[EMBED_SLOTS,
/// EMBEDDING_DIM]`) carries the identical element count as the correct
/// shape, so a total-element-count check alone (as `MultiArray::copy_into`
/// performs) would not detect it.
#[test]
fn check_output_shape_rejects_swapped_axes() {
  assert_eq!(
    check_output_shape(&[EMBEDDING_DIM, EMBED_SLOTS]),
    Err(InferError::OutputShape {
      got: vec![EMBEDDING_DIM, EMBED_SLOTS],
      expected: vec![EMBED_SLOTS, EMBEDDING_DIM],
    })
  );
}

#[test]
fn check_output_shape_rejects_wrong_rank() {
  assert_eq!(
    check_output_shape(&[EMBED_SLOTS * EMBEDDING_DIM]),
    Err(InferError::OutputShape {
      got: vec![EMBED_SLOTS * EMBEDDING_DIM],
      expected: vec![EMBED_SLOTS, EMBEDDING_DIM],
    })
  );
}

#[test]
fn check_output_shape_rejects_wrong_slot_count() {
  assert_eq!(
    check_output_shape(&[2, EMBEDDING_DIM]),
    Err(InferError::OutputShape {
      got: vec![2, EMBEDDING_DIM],
      expected: vec![EMBED_SLOTS, EMBEDDING_DIM],
    })
  );
}

#[test]
fn check_output_shape_rejects_wrong_embedding_dim() {
  assert_eq!(
    check_output_shape(&[EMBED_SLOTS, EMBEDDING_DIM - 1]),
    Err(InferError::OutputShape {
      got: vec![EMBED_SLOTS, EMBEDDING_DIM - 1],
      expected: vec![EMBED_SLOTS, EMBEDDING_DIM],
    })
  );
}

// ---------------------------------------------------------------------
// EmbedModelOptions
// ---------------------------------------------------------------------

#[test]
fn options_new_defaults_to_all_compute() {
  assert_eq!(EmbedModelOptions::new().compute(), ComputeUnits::All);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(EmbedModelOptions::default(), EmbedModelOptions::new());
}

#[test]
fn options_with_compute_overrides() {
  let options = EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_set_compute_in_place() {
  let mut options = EmbedModelOptions::new();
  options.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(options.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_compute_defaults_to_all() {
  let options: EmbedModelOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(options.compute(), ComputeUnits::All);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips_explicit_compute() {
  let options: EmbedModelOptions = serde_json::from_str(r#"{"compute":"cpu_only"}"#).unwrap();
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
  let json = serde_json::to_string(&options).unwrap();
  assert!(json.contains("cpu_only"), "round-tripped json: {json}");
}

// ---------------------------------------------------------------------
// EmbedModel: model-gated (brief Step 2) — requires local
// wespeaker_v2.mlmodelc AND wespeaker.mlmodelc (SPEAKERKIT_TEST_MODELS or
// Models/speakerkit/, same convention as tests/model_io.rs's `common`
// module and crate::audio::speaker::segment::tests). Duplicated here in miniature because
// unit tests under `src/` cannot import the separate `tests/`
// integration-test crate.
//
// The full suite below runs against BOTH artifacts, per this task's
// brief. Determinism is checked WITHIN one artifact only per call — int8
// (`wespeaker_v2`) and fp32 (`wespeaker`) legitimately diverge
// numerically, so no test compares results ACROSS artifacts.
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

fn embed_v2_path() -> std::path::PathBuf {
  models_dir().join("wespeaker_v2.mlmodelc")
}

fn embed_fp32_path() -> std::path::PathBuf {
  models_dir().join("wespeaker.mlmodelc")
}

/// Loads a real embedding model from `path` with `ComputeUnits::CpuOnly` —
/// matching `tests/model_io.rs`'s and
/// `crate::audio::speaker::segment::tests::load_seg_model`'s convention: deterministic, no
/// ANE compile-latency variance across runs. `DEFAULT_EMBED_COMPUTE`
/// (`ComputeUnits::All`) stays the production default. Parameterized over
/// `path` so the SAME loader drives both `wespeaker_v2.mlmodelc` (int8,
/// T1's targeted artifact) and `wespeaker.mlmodelc` (fp32, contract-equal
/// per T1) — this task's brief requires the full model-gated suite run
/// against both.
fn load_embed_model(path: std::path::PathBuf) -> EmbedModel {
  EmbedModel::from_file_with(
    path,
    EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load embedding model")
}

/// Non-silent, deterministic synthetic signal for tests that need a real
/// (non-degenerate) embedding — a two-tone sine mix, not noise, so it is
/// exactly reproducible across runs and platforms.
fn synthetic_samples(len: usize) -> Vec<f32> {
  (0..len)
    .map(|i| {
      let t = i as f32 / 16_000.0; // 16 kHz, dia's SAMPLE_RATE_HZ (diarization/src/embed/options.rs:34)
      0.2 * (2.0 * core::f32::consts::PI * 220.0 * t).sin()
        + 0.1 * (2.0 * core::f32::consts::PI * 440.0 * t).sin()
    })
    .collect()
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn from_file_loads_and_reports_mask_frame_count_v2() {
  let model = load_embed_model(embed_v2_path());
  // Ground truth pinned by
  // `tests/model_io.rs::wespeaker_v2_io_matches_spec`: 589 frames.
  assert_eq!(model.num_mask_frames(), 589);
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn from_file_loads_and_reports_mask_frame_count_fp32() {
  let model = load_embed_model(embed_fp32_path());
  // Ground truth pinned by
  // `tests/model_io.rs::wespeaker_fp32_io_contract_equal_but_not_targeted`.
  assert_eq!(model.num_mask_frames(), 589);
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn from_file_rejects_wrong_contract_model() {
  // pyannote_segmentation.mlmodelc has no `waveform`/`mask` inputs at all
  // (its input is `audio`) — a real, locally-available model with a
  // definitely-mismatched contract, mirroring
  // `crate::audio::speaker::segment::tests::from_file_rejects_wrong_contract_model`'s
  // reciprocal use of `wespeaker_v2.mlmodelc`.
  let path = models_dir().join("pyannote_segmentation.mlmodelc");
  let err = EmbedModel::from_file(path).expect_err("wrong contract must be rejected");
  assert!(matches!(
    err,
    ModelError::ContractMismatch {
      feature: "waveform",
      ..
    }
  ));
}

fn embed_chunk_produces_correctly_shaped_finite_embeddings(path: std::path::PathBuf) {
  let model = load_embed_model(path);
  let samples = synthetic_samples(crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  let mask = vec![true; model.num_mask_frames()];
  let masks: [&[bool]; EMBED_SLOTS] = [&mask, &mask, &mask];
  let out = model
    .embed_chunk(&samples, &masks)
    .expect("embed_chunk on real audio");
  for row in out.iter() {
    assert_eq!(row.len(), EMBEDDING_DIM);
    assert!(
      row.iter().all(|v| v.is_finite()),
      "all embedding values finite"
    );
  }
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_produces_correctly_shaped_finite_embeddings_v2() {
  embed_chunk_produces_correctly_shaped_finite_embeddings(embed_v2_path());
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_produces_correctly_shaped_finite_embeddings_fp32() {
  embed_chunk_produces_correctly_shaped_finite_embeddings(embed_fp32_path());
}

fn embed_chunk_is_deterministic_across_repeated_calls(path: std::path::PathBuf) {
  let model = load_embed_model(path);
  let samples = synthetic_samples(crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  let mask = vec![true; model.num_mask_frames()];
  let masks: [&[bool]; EMBED_SLOTS] = [&mask, &mask, &mask];
  let first = model
    .embed_chunk(&samples, &masks)
    .expect("first embed_chunk");
  let second = model
    .embed_chunk(&samples, &masks)
    .expect("second embed_chunk");
  assert_eq!(
    first, second,
    "repeated embed_chunk must be bit-identical WITHIN one artifact"
  );
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_is_deterministic_across_repeated_calls_v2() {
  embed_chunk_is_deterministic_across_repeated_calls(embed_v2_path());
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_is_deterministic_across_repeated_calls_fp32() {
  embed_chunk_is_deterministic_across_repeated_calls(embed_fp32_path());
}

fn embed_chunk_with_frame_mask_is_raw_not_unit_norm(path: std::path::PathBuf) {
  let model = load_embed_model(path);
  let samples = synthetic_samples(crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  let mask = vec![true; model.num_mask_frames()];
  let embedding = model
    .embed_chunk_with_frame_mask(&samples, &mask)
    .expect("embed_chunk_with_frame_mask on real audio");
  let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
  assert!(
    (norm - 1.0).abs() > 1e-3,
    "raw WeSpeaker output must not be unit-normalized (dia normalizes downstream, not here); got norm {norm}"
  );
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_with_frame_mask_is_raw_not_unit_norm_v2() {
  embed_chunk_with_frame_mask_is_raw_not_unit_norm(embed_v2_path());
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_with_frame_mask_is_raw_not_unit_norm_fp32() {
  embed_chunk_with_frame_mask_is_raw_not_unit_norm(embed_fp32_path());
}

/// [`EmbedModel::embed_chunk_with_frame_mask`] must equal
/// [`EmbedModel::embed_chunk`]'s slot 0 when called with the same samples
/// and the same mask in slot 0 (empty masks in slots 1-2) — pins the
/// veneer relationship the module doc describes, against the real model,
/// not just by code inspection.
fn embed_chunk_with_frame_mask_matches_batched_slot_zero(path: std::path::PathBuf) {
  let model = load_embed_model(path);
  let samples = synthetic_samples(crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  // Partial (not all-active) mask — exercises a non-trivial pooling
  // weight, not just the degenerate all-ones case.
  let mut mask = vec![false; model.num_mask_frames()];
  for m in mask.iter_mut().step_by(3) {
    *m = true;
  }
  let veneer = model
    .embed_chunk_with_frame_mask(&samples, &mask)
    .expect("embed_chunk_with_frame_mask");
  let empty: &[bool] = &[];
  let masks: [&[bool]; EMBED_SLOTS] = [&mask, empty, empty];
  let batched = model.embed_chunk(&samples, &masks).expect("embed_chunk");
  assert_eq!(
    veneer, batched[0],
    "embed_chunk_with_frame_mask must equal embed_chunk's slot 0 for the same mask"
  );
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_with_frame_mask_matches_batched_slot_zero_v2() {
  embed_chunk_with_frame_mask_matches_batched_slot_zero(embed_v2_path());
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_with_frame_mask_matches_batched_slot_zero_fp32() {
  embed_chunk_with_frame_mask_matches_batched_slot_zero(embed_fp32_path());
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_with_frame_mask_rejects_all_false_mask() {
  let model = load_embed_model(embed_v2_path());
  let samples = synthetic_samples(crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  let mask = vec![false; model.num_mask_frames()];
  let err = model
    .embed_chunk_with_frame_mask(&samples, &mask)
    .expect_err("all-false mask must be rejected");
  assert_eq!(err, InferError::EmptyMask);
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_rejects_non_finite_samples() {
  let model = load_embed_model(embed_v2_path());
  let mut samples = synthetic_samples(crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  samples[1234] = f32::NAN;
  let mask = vec![true; model.num_mask_frames()];
  let masks: [&[bool]; EMBED_SLOTS] = [&mask, &mask, &mask];
  let err = model
    .embed_chunk(&samples, &masks)
    .expect_err("NaN samples must be rejected");
  assert_eq!(err, InferError::NonFiniteInput { index: 1234 });
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn embed_chunk_handles_short_padded_input() {
  // A chunk shorter than SEG_CHUNK_SAMPLES exercises the repeat-pad path
  // against the real model end to end (not just the hermetic
  // `repeat_pad_f32` unit tests).
  let model = load_embed_model(embed_v2_path());
  let samples = synthetic_samples(40_000); // 2.5s, well under the 10s chunk
  let mask = vec![true; model.num_mask_frames()];
  let masks: [&[bool]; EMBED_SLOTS] = [&mask, &mask, &mask];
  let out = model
    .embed_chunk(&samples, &masks)
    .expect("embed_chunk on a short, repeat-padded chunk");
  for row in out.iter() {
    assert!(row.iter().all(|v| v.is_finite()));
  }
}
