use super::*;

#[test]
fn geometry_constants_match_fluidaudio() {
  // FluidAudio `VadManager`: chunkSize 4096, contextSize 64, modelInputSize
  // 4160, stateSize 128 (`VadManager.swift:22-25`).
  assert_eq!(CHUNK_SAMPLES, 4096);
  assert_eq!(CONTEXT_SAMPLES, 64);
  assert_eq!(MODEL_INPUT_SAMPLES, 4160);
  assert_eq!(MODEL_INPUT_SAMPLES, CONTEXT_SAMPLES + CHUNK_SAMPLES);
  assert_eq!(STATE_SIZE, 128);
}

// ── prepare_chunk (FluidAudio processChunk padding) ────────────────────────

#[test]
fn prepare_chunk_exact_length_passes_through() {
  let chunk: Vec<f32> = (0..CHUNK_SAMPLES).map(|i| i as f32).collect();
  let out = prepare_chunk(&chunk).expect("exact length");
  assert_eq!(&out[..], &chunk[..]);
}

#[test]
fn prepare_chunk_pads_short_by_repeating_last_sample() {
  // Repeat-LAST, not zeros (`VadManager.swift:174-178`).
  let chunk = [0.5f32, -0.25, 0.75];
  let out = prepare_chunk(&chunk).expect("short chunk");
  assert_eq!(&out[..3], &chunk[..]);
  // Every padded slot is the last real sample, 0.75 — a zero-padding mutation
  // (`padded[chunk.len()..]` left at 0.0) turns this red.
  assert!(
    out[3..].iter().all(|&v| v == 0.75),
    "short chunk must pad by repeating the last sample"
  );
}

#[test]
fn prepare_chunk_empty_pads_with_zero() {
  // `chunk.last ?? 0.0`.
  let out = prepare_chunk(&[]).expect("empty chunk");
  assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn prepare_chunk_rejects_over_long() {
  let chunk = vec![0.0f32; CHUNK_SAMPLES + 1];
  let err = prepare_chunk(&chunk).expect_err("over-long must reject, not truncate");
  assert_eq!(
    err,
    InferError::ChunkTooLong {
      got: CHUNK_SAMPLES + 1,
      max: CHUNK_SAMPLES,
    }
  );
}

// ── assemble_window (context then chunk) ───────────────────────────────────

#[test]
fn assemble_window_places_context_then_chunk() {
  let mut context = [0.0f32; CONTEXT_SAMPLES];
  for (i, c) in context.iter_mut().enumerate() {
    *c = i as f32;
  }
  let mut chunk = [0.0f32; CHUNK_SAMPLES];
  for (i, c) in chunk.iter_mut().enumerate() {
    *c = 1000.0 + i as f32;
  }
  let window = assemble_window(&context, &chunk);

  // Context occupies [0..64], chunk occupies [64..4160]. A swap of the copy
  // order (chunk first, context last) turns both asserts red.
  assert_eq!(&window[..CONTEXT_SAMPLES], &context[..]);
  assert_eq!(&window[CONTEXT_SAMPLES..], &chunk[..]);
  assert_eq!(window[CONTEXT_SAMPLES - 1], 63.0, "last context sample");
  assert_eq!(window[CONTEXT_SAMPLES], 1000.0, "first chunk sample");
}

// ── next_context (last 64 of the padded chunk) ─────────────────────────────

#[test]
fn next_context_is_the_last_64_samples() {
  let mut chunk = [0.0f32; CHUNK_SAMPLES];
  for (i, c) in chunk.iter_mut().enumerate() {
    *c = i as f32;
  }
  let context = next_context(&chunk);
  // Exactly chunk[4032..4096]. A one-sample offset skew (the exact stitching
  // bug the model-gated `misaligned_context_changes_the_probability` also
  // catches) turns this red.
  assert_eq!(
    context[0],
    (CHUNK_SAMPLES - CONTEXT_SAMPLES) as f32,
    "4032.0"
  );
  assert_eq!(
    context[CONTEXT_SAMPLES - 1],
    (CHUNK_SAMPLES - 1) as f32,
    "4095.0"
  );
  for (i, &v) in context.iter().enumerate() {
    assert_eq!(v, (CHUNK_SAMPLES - CONTEXT_SAMPLES + i) as f32);
  }
}

#[test]
fn stitching_carries_the_previous_chunks_tail_forward() {
  // The end-to-end stitching invariant, hermetically: the context that the
  // next window's [0..64] would carry is exactly the previous (padded)
  // chunk's last 64 samples.
  let mut chunk0 = [0.0f32; CHUNK_SAMPLES];
  for (i, c) in chunk0.iter_mut().enumerate() {
    *c = (i as f32) * 0.001;
  }
  let carried = next_context(&chunk0);
  let chunk1 = [7.0f32; CHUNK_SAMPLES];
  let window1 = assemble_window(&carried, &chunk1);
  assert_eq!(
    &window1[..CONTEXT_SAMPLES],
    &chunk0[CHUNK_SAMPLES - CONTEXT_SAMPLES..]
  );
  assert_eq!(&window1[CONTEXT_SAMPLES..], &chunk1[..]);
}

// ── finite input scan ──────────────────────────────────────────────────────

#[test]
fn check_finite_input_accepts_finite() {
  assert!(check_finite_input(&[0.0, 1.0, -1.0, 0.5]).is_ok());
}

#[test]
fn check_finite_input_rejects_nan_at_its_index() {
  let mut window = [0.0f32; 8];
  window[5] = f32::NAN;
  assert_eq!(
    check_finite_input(&window),
    Err(InferError::NonFiniteInput { index: 5 })
  );
}

#[test]
fn check_finite_input_rejects_infinity() {
  let mut window = [0.0f32; 8];
  window[2] = f32::INFINITY;
  assert_eq!(
    check_finite_input(&window),
    Err(InferError::NonFiniteInput { index: 2 })
  );
}

// ── output shape re-check ──────────────────────────────────────────────────

#[test]
fn check_output_shape_accepts_the_contract() {
  assert!(check_output_shape(&[1, 1, 1], names::VAD_OUTPUT, &[1, 1, 1]).is_ok());
  assert!(check_output_shape(&[1, STATE_SIZE], names::NEW_HIDDEN_STATE, &[1, STATE_SIZE]).is_ok());
}

#[test]
fn check_output_shape_rejects_a_divergent_shape() {
  let err =
    check_output_shape(&[1, 1], names::VAD_OUTPUT, &[1, 1, 1]).expect_err("wrong rank must fail");
  assert_eq!(
    err,
    InferError::OutputShape {
      feature: names::VAD_OUTPUT,
      got: vec![1, 1],
      expected: vec![1, 1, 1],
    }
  );
}

// ── VadState ───────────────────────────────────────────────────────────────

#[test]
fn vad_state_initial_is_all_zero() {
  let s = VadState::initial();
  assert!(s.hidden().iter().all(|&v| v == 0.0));
  assert!(s.cell().iter().all(|&v| v == 0.0));
  assert!(s.context().iter().all(|&v| v == 0.0));
  assert_eq!(s, VadState::default());
}

#[test]
fn vad_state_from_parts_round_trips_through_accessors() {
  let hidden = [1.0f32; STATE_SIZE];
  let cell = [2.0f32; STATE_SIZE];
  let context = [3.0f32; CONTEXT_SAMPLES];
  let s = VadState::from_parts(hidden, cell, context);
  assert_eq!(s.hidden(), &hidden);
  assert_eq!(s.cell(), &cell);
  assert_eq!(s.context(), &context);
}

// ── VadModelOptions ────────────────────────────────────────────────────────

#[test]
fn vad_model_options_default_is_all() {
  assert_eq!(VadModelOptions::new().compute(), ComputeUnits::All);
  assert_eq!(VadModelOptions::default(), VadModelOptions::new());
  assert_eq!(DEFAULT_VAD_COMPUTE, ComputeUnits::All);
}

#[test]
fn vad_model_options_with_and_set_compute() {
  let opts = VadModelOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(opts.compute(), ComputeUnits::CpuOnly);
  let mut opts = VadModelOptions::new();
  opts.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(opts.compute(), ComputeUnits::CpuAndNeuralEngine);
}
