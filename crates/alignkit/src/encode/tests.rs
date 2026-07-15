use super::*;

// ---------------------------------------------------------------------
// truncated_frame_count: hermetic coverage of the truncation/clamp math.
// The comments on each case below call out which mutation of
// `truncated_frame_count` (module doc's "Truncation formula" section)
// the test would catch, per the task's mutation-evidence requirement.
// ---------------------------------------------------------------------

#[test]
fn truncated_frame_count_zero_samples_is_zero() {
  assert_eq!(truncated_frame_count(0, 2999), 0);
}

#[test]
fn truncated_frame_count_exact_multiple_of_hop() {
  // 320 samples == exactly one hop: ceil(320/320) == 1, not 2. Catches a
  // mutant that used a plain `/` (floor) with a stray `+1`, or a
  // `div_ceil` misuse that rounds exact multiples up.
  assert_eq!(truncated_frame_count(HOP_SAMPLES, 2999), 1);
}

#[test]
fn truncated_frame_count_rounds_up_on_remainder() {
  // 321 samples is one hop plus one leftover sample: ceil(321/320) == 2.
  // Catches a mutant that used floor division (`/`) instead of
  // `div_ceil`, which would wrongly return 1.
  assert_eq!(truncated_frame_count(HOP_SAMPLES + 1, 2999), 2);
}

#[test]
fn truncated_frame_count_short_clip_not_clamped() {
  // 48,000 samples (3 s) against the real model's 2,999-frame ceiling:
  // nominal ceil(48_000/320) == 150, well under 2,999, so the clamp must
  // be a no-op here. Catches a mutant that flips `.min` to `.max` (which
  // would wrongly clamp UP to 2,999 even though nominal is already
  // smaller) — `.max` only produces the same answer as `.min` when
  // nominal >= available_frames, which is not this case.
  assert_eq!(truncated_frame_count(48_000, 2_999), 150);
}

#[test]
fn truncated_frame_count_at_clamp_boundary_is_not_clamped() {
  // 2_999 * 320 == 959_680 samples: nominal ceil(959_680/320) == 2_999
  // exactly, equal to (not exceeding) available_frames, so clamping is a
  // no-op at this exact boundary.
  assert_eq!(truncated_frame_count(2_999 * HOP_SAMPLES, 2_999), 2_999);
}

#[test]
fn truncated_frame_count_one_sample_past_clamp_boundary_engages_clamp() {
  // One more sample than the boundary above: nominal ceil(959_681/320)
  // == 3_000, one past available_frames — the clamp must now engage and
  // cap the result at 2_999.
  assert_eq!(truncated_frame_count(2_999 * HOP_SAMPLES + 1, 2_999), 2_999);
}

#[test]
fn truncated_frame_count_full_window_clamps_to_available_frames() {
  // THE boundary the module doc calls out by name: a full,
  // zero-padding-free ENCODER_WINDOW_SAMPLES (960,000, i.e. exactly the
  // `ted_60.wav` fixture's own case) has nominal ceil(960_000/320) ==
  // 3_000 — one MORE than `base960h_aligner.mlmodelc`'s actual 2,999
  // frames (`tests/model_io.rs::base960h_aligner_io_matches_spec`).
  // Without the `.min(available_frames)` clamp, `Encoder::emissions`
  // would compute `real_frames = 3_000` and then either panic slicing
  // `raw` (2,999 * VOCAB_SIZE elements) at a 3,000-frame boundary, or
  // (if `truncate` silently no-ops past `raw.len()`, which it does)
  // silently under-report by handing `LogProbsTV::new` a `t` that no
  // longer matches `raw.len() / VOCAB_SIZE`, tripping the `.expect(...)`
  // in `Encoder::emissions` instead. This is the exact regression this
  // test pins: removing the clamp (or flipping `.min` to `.max`) makes
  // this assertion fail with `3_000 != 2_999`.
  assert_eq!(truncated_frame_count(ENCODER_WINDOW_SAMPLES, 2_999), 2_999);
}

#[test]
fn truncated_frame_count_never_exceeds_available_frames_near_full_window() {
  // Property-style sweep just below/at/above the full window, cross-
  // checking the invariant `result <= available_frames` the clamp
  // exists to guarantee.
  let available_frames = 2_999;
  for real_samples in [
    ENCODER_WINDOW_SAMPLES - 1,
    ENCODER_WINDOW_SAMPLES,
    available_frames * HOP_SAMPLES,
    available_frames * HOP_SAMPLES + 1,
  ] {
    let t = truncated_frame_count(real_samples, available_frames);
    assert!(
      t <= available_frames,
      "truncated_frame_count({real_samples}, {available_frames}) = {t} exceeds available_frames"
    );
  }
}

// ---------------------------------------------------------------------
// EncoderInput: the F1 capability. Hermetic, and that is the whole point —
// a wrong real-sample length is unrepresentable at CONSTRUCTION, before any
// Encoder or model exists, so the mismatch the free `real_samples: usize`
// argument used to allow cannot reach a prediction.
// ---------------------------------------------------------------------

#[test]
fn encoder_input_from_samples_binds_real_length_to_the_slice() {
  // A 176,000-sample chunk fed as raw audio: `real_samples` IS the slice's own
  // length, 176,000. The F1 defect declared 175,360 (two hops short) for this
  // same buffer to get 548 frames where 550 belong; there is now no
  // `real_samples` argument to declare it into.
  let chunk = vec![0.0f32; 176_000];
  let input = EncoderInput::from_samples(&chunk).expect("176k <= window");
  assert_eq!(input.real_samples, 176_000);
  assert_eq!(input.encoder_input.len(), 176_000);
  assert_eq!(truncated_frame_count(input.real_samples, 2_999), 550);
  // The buggy answer is now unreachable: 175_360 gives 548, but nothing can
  // bind 175_360 to this 176,000-sample buffer.
  assert_eq!(truncated_frame_count(175_360, 2_999), 548);
  assert_ne!(
    truncated_frame_count(input.real_samples, 2_999),
    truncated_frame_count(175_360, 2_999)
  );
}

#[test]
fn encoder_input_gate_binds_real_length_independent_of_the_padded_buffer() {
  // The pipeline geometry: 200 real samples that asry silence-masks and zero-pads
  // to the 400-sample receptive field. The gate every constructor funnels through
  // records the real length as the UNPADDED count (200), never the padded buffer's
  // length (400). `from_prepared` reads exactly this (buffer, real_samples) pair
  // off an unforgeable `PreparedChunk`; here we drive the gate directly so the
  // geometry is pinned with no model and no seam (the end-to-end `from_prepared`
  // door, on a real chunk, is `tests/prepared_composition.rs`). The F1 defect
  // recorded `encoder_input.len()` (400) as the real count, yielding 2 frames
  // where 1 belongs.
  let real_len = 200usize;
  let padded_buffer = vec![0.0f32; 400];
  let input = EncoderInput::new(&padded_buffer, real_len).expect("valid geometry");
  assert_eq!(input.real_samples, 200); // NOT 400
  assert_eq!(input.encoder_input.len(), 400);
  assert_eq!(truncated_frame_count(input.real_samples, 2_999), 1); // ceil(200/320)
  // The buffer-length count (2) is the F1 bug — exactly what `from_samples` on the
  // padded buffer would produce, which is why the padded buffer must never take
  // that door.
  assert_eq!(truncated_frame_count(400, 2_999), 2); // the buggy count
  assert_ne!(
    truncated_frame_count(input.real_samples, 2_999),
    truncated_frame_count(padded_buffer.len(), 2_999)
  );
}

#[test]
fn encoder_input_rejects_a_buffer_longer_than_the_window_before_any_prediction() {
  // Invalid geometry is caught at construction, with no Encoder and no model in
  // sight — so it can never reach a prediction. (Formerly this check lived
  // inside `emissions_raw`, one predict away.)
  let too_long = vec![0.0f32; ENCODER_WINDOW_SAMPLES + 1];
  let err = EncoderInput::from_samples(&too_long).unwrap_err();
  assert!(matches!(
    err,
    AlignError::InputTooLong { got, max }
      if got == ENCODER_WINDOW_SAMPLES + 1 && max == ENCODER_WINDOW_SAMPLES
  ));
}

#[test]
fn encoder_input_accepts_a_buffer_exactly_the_window() {
  // The exact-window boundary is valid — it is the `ted_60.wav` case, where
  // `emissions_raw` borrows the buffer rather than padding it.
  let full = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
  let input = EncoderInput::from_samples(&full).expect("exactly the window is fine");
  assert_eq!(input.real_samples, ENCODER_WINDOW_SAMPLES);
  assert_eq!(input.encoder_input.len(), ENCODER_WINDOW_SAMPLES);
}

// ---------------------------------------------------------------------
// check_waveform_contract / check_emissions_contract: hermetic coverage
// without a loaded model (see their doc comments for why this crate
// tests the validation logic directly rather than model-gating against a
// second, deliberately-wrong local model fixture — `Models/alignkit/`
// holds exactly one model).
// ---------------------------------------------------------------------

#[test]
fn check_waveform_contract_accepts_correct_shape_and_dtype() {
  assert_eq!(
    check_waveform_contract(&[1, ENCODER_WINDOW_SAMPLES], Some(DataType::F32)),
    Ok(())
  );
}

#[test]
fn check_waveform_contract_rejects_wrong_shape() {
  let err = check_waveform_contract(&[1, 480_000], Some(DataType::F32)).unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "waveform",
      ..
    }
  ));
}

#[test]
fn check_waveform_contract_rejects_wrong_dtype() {
  let err = check_waveform_contract(&[1, ENCODER_WINDOW_SAMPLES], Some(DataType::F16)).unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "waveform",
      ..
    }
  ));
}

#[test]
fn check_waveform_contract_rejects_missing_dtype() {
  let err = check_waveform_contract(&[1, ENCODER_WINDOW_SAMPLES], None).unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_accepts_correct_shape_and_returns_frame_count() {
  assert_eq!(
    check_emissions_contract(&[1, 2_999, crate::vocab::VOCAB_SIZE], Some(DataType::F32)),
    Ok(2_999)
  );
}

#[test]
fn check_emissions_contract_rejects_wrong_rank() {
  let err =
    check_emissions_contract(&[2_999, crate::vocab::VOCAB_SIZE], Some(DataType::F32)).unwrap_err();
  assert!(matches!(
    err,
    AlignerError::ContractMismatch {
      feature: "emissions",
      ..
    }
  ));
}

#[test]
fn check_emissions_contract_rejects_wrong_batch_dim() {
  let err = check_emissions_contract(&[2, 2_999, crate::vocab::VOCAB_SIZE], Some(DataType::F32))
    .unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_rejects_zero_frames() {
  // A zero-frame model would "load fine" and make every `emissions()`
  // call silently return an empty result — reject at construction.
  let err =
    check_emissions_contract(&[1, 0, crate::vocab::VOCAB_SIZE], Some(DataType::F32)).unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_rejects_wrong_vocab_dim() {
  let err = check_emissions_contract(&[1, 2_999, 32], Some(DataType::F32)).unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

#[test]
fn check_emissions_contract_rejects_wrong_dtype() {
  let err = check_emissions_contract(&[1, 2_999, crate::vocab::VOCAB_SIZE], Some(DataType::F64))
    .unwrap_err();
  assert!(matches!(err, AlignerError::ContractMismatch { .. }));
}

// ---------------------------------------------------------------------
// check_log_prob_floor: hermetic coverage of the fp16 `log(0)` sentinel
// guard. The model-gated half (`emissions_reject_an_ane_corrupted_matrix`)
// proves the real ANE artifact trips it; these prove the predicate itself,
// including the two boundaries a mutant would move.
// ---------------------------------------------------------------------

#[test]
fn check_log_prob_floor_accepts_real_log_probs() {
  // The measured legitimate range on this model: max exactly 0.0, min -30.81
  // (`CpuOnly`) / -30.02 (`CpuAndGpu`). Nothing here is anywhere near the floor.
  let data = [0.0, -0.06, -19.0, -21.75, -30.02, -30.81];
  assert!(check_log_prob_floor(&data, ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_log_prob_floor_accepts_an_empty_matrix() {
  // `real_samples == 0` truncates to zero frames; the guard must not invent a
  // failure out of an empty scan (min would be +inf).
  assert!(check_log_prob_floor(&[], ComputeUnits::CpuOnly).is_ok());
}

#[test]
fn check_log_prob_floor_rejects_the_fp16_log_zero_sentinel() {
  // One corrupt cell in an otherwise clean matrix is still a corrupt matrix:
  // the ANE run corrupts 16.7% of cells, but a single one is enough to move a
  // trellis path. Catches a mutant that thresholds on a FRACTION of cells.
  let data = [0.0, -1.5, -45_440.0, -20.0];
  let Err(err) = check_log_prob_floor(&data, ComputeUnits::All) else {
    panic!("the -45440 fp16 log(0) sentinel must be rejected");
  };
  let AlignError::CorruptEmissions {
    compute,
    min,
    cells,
    total,
  } = err
  else {
    panic!("expected AlignError::CorruptEmissions, got {err:?}");
  };
  assert_eq!(compute, ComputeUnits::All);
  assert_eq!(min, -45_440.0);
  assert_eq!(cells, 1);
  assert_eq!(total, 4);
}

#[test]
fn check_log_prob_floor_is_a_strict_lower_bound_at_the_floor_itself() {
  // The floor is INCLUSIVE (`< LOG_PROB_FLOOR` fails, `== LOG_PROB_FLOOR`
  // passes). Pins the comparison's direction and strictness together: a mutant
  // flipping `<` to `<=` fails the first assertion, one flipping it to `>`
  // fails the second.
  assert!(check_log_prob_floor(&[LOG_PROB_FLOOR], ComputeUnits::CpuOnly).is_ok());
  assert!(
    check_log_prob_floor(&[LOG_PROB_FLOOR - 1.0], ComputeUnits::CpuOnly).is_err(),
    "one ulp-plus below the floor is already outside the log-prob domain"
  );
}

#[test]
fn check_log_prob_floor_leaves_non_finite_values_to_from_log_probs() {
  // Deliberate division of labour, documented on `check_log_prob_floor`: the
  // floor guard is the LOWER bound only. `NaN` compares false against
  // everything and passes here; `Emissions::from_log_probs`' finite ∧ <= 0 scan
  // (which runs on the very next line of `Encoder::emissions`) is what rejects
  // it. Neither scan is redundant with the other, and this pins that seam so a
  // later "simplification" cannot silently drop one of them.
  assert!(check_log_prob_floor(&[f32::NAN], ComputeUnits::CpuOnly).is_ok());
  assert!(check_log_prob_floor(&[f32::INFINITY], ComputeUnits::CpuOnly).is_ok());
  // -inf is genuinely below the floor and IS the guard's business.
  assert!(check_log_prob_floor(&[f32::NEG_INFINITY], ComputeUnits::CpuOnly).is_err());
}

// LOG_PROB_FLOOR's separation property (strictly between the -30.81 legitimate
// minimum and the -45440 sentinel) is asserted in `mod.rs` at COMPILE time, not
// here: both operands are constants, so a runtime test of it is dead weight that
// only fires after a build already succeeded.

// ---------------------------------------------------------------------
// EncoderOptions
// ---------------------------------------------------------------------

#[test]
fn options_new_defaults_to_cpu_only_compute() {
  // Not a perf preference: the ANE placements corrupt this model's emissions.
  // See `DEFAULT_ENCODER_COMPUTE` and
  // `emissions_have_no_fp16_log_zero_sentinel`.
  assert_eq!(EncoderOptions::new().compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(EncoderOptions::new().compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(EncoderOptions::default(), EncoderOptions::new());
}

#[test]
fn options_with_compute_overrides() {
  // A NON-default placement, or this would also pass against a `with_compute`
  // that silently ignored its argument.
  let options = EncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(options.compute(), ComputeUnits::CpuAndGpu);
}

#[test]
fn options_set_compute_in_place() {
  let mut options = EncoderOptions::new();
  options.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(options.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_compute_defaults_to_cpu_only() {
  let options: EncoderOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(options.compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips_explicit_compute() {
  // Round-trip a non-default placement: deserializing `cpu_only` would now be
  // indistinguishable from the field defaulting.
  let options: EncoderOptions = serde_json::from_str(r#"{"compute":"cpu_and_gpu"}"#).unwrap();
  assert_eq!(options.compute(), ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&options).unwrap();
  assert!(json.contains("cpu_and_gpu"), "round-tripped json: {json}");
}

// ---------------------------------------------------------------------
// Encoder: model-gated (requires a local base960h_aligner.mlmodelc,
// ALIGNKIT_TEST_MODELS or Models/alignkit/, same convention as
// tests/model_io.rs's `common` module and tests/common/mod.rs).
// Duplicated here in miniature because unit tests under `src/` cannot
// import the separate `tests/` integration-test crate (mirrors
// dia-coreml::segment::tests's identical duplication and rationale).
//
// These load the encoder on DEFAULT_ENCODER_COMPUTE — never a hardcoded
// placement — so they validate the SHIPPING configuration for free. A gate
// pinned to a compute unit proves only that compute unit; pinning CpuOnly
// here is exactly how the `All`-path emission corruption survived review.
//
// Mostly synthetic signals, but not exclusively: the fp16 `log(0)` sentinel
// only appears on inputs whose probabilities fall under the fp16 floor, which
// silence and a low-amplitude sine never do — see
// `emissions_have_no_fp16_log_zero_sentinel`, which needs real speech.
// ---------------------------------------------------------------------

fn models_dir() -> std::path::PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("alignkit")
    },
    std::path::PathBuf::from,
  )
}

fn encoder_path() -> std::path::PathBuf {
  models_dir().join("base960h_aligner.mlmodelc")
}

/// Loads the real encoder model on [`DEFAULT_ENCODER_COMPUTE`] — the shipping
/// placement, via the same `EncoderOptions::new()` door production code takes.
/// Deliberately NOT a hardcoded `ComputeUnits::_`: every model-gated test
/// below is then a test OF the default.
fn load_encoder() -> Encoder {
  Encoder::from_file(encoder_path())
    .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS to the model directory)")
}

/// `EncoderInput::from_samples` for the model-gated tests below, whose fixtures
/// are always within the window. The fallible construction is F1's geometry
/// gate; its rejection path is proven hermetically by
/// `encoder_input_rejects_a_buffer_longer_than_the_window_before_any_prediction`
/// (no model needed), so there is no longer a model-gated too-long test — the
/// too-long buffer never reaches `emissions_raw` at all.
fn window_input(samples: &[f32]) -> EncoderInput<'_> {
  EncoderInput::from_samples(samples).expect("model-gated fixtures are <= ENCODER_WINDOW_SAMPLES")
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn from_file_loads_and_reports_frame_count() {
  let encoder = load_encoder();
  // Ground truth pinned by
  // `tests/model_io.rs::base960h_aligner_io_matches_spec`: 2,999 frames.
  assert_eq!(encoder.frames(), 2_999);
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_on_full_window_produces_correctly_shaped_finite_log_probs() {
  let encoder = load_encoder();
  let samples = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
  let raw = encoder
    .emissions_raw(window_input(&samples))
    .expect("emissions on silence");
  assert_eq!(raw.frames, encoder.frames());
  assert_eq!(raw.data.len(), raw.frames * crate::vocab::VOCAB_SIZE);
  assert!(
    raw.data.iter().all(|v| v.is_finite()),
    "all log-probs finite"
  );
  // Log-probabilities are bounded above by log(1) == 0. This is also the
  // exact domain `Emissions::from_log_probs` enforces, so a pass here is a
  // canary that `Encoder::emissions` (the wrapped door) will not trip the
  // value-domain scan on this input.
  assert!(
    raw.data.iter().all(|&v| v <= 0.0),
    "log-probs must satisfy log(p) <= 0"
  );
}

/// **THE C1 REGRESSION ORACLE.** No emission cell may be an fp16 `log(0)`
/// saturation sentinel.
///
/// `base960h_aligner.mlmodelc` ends in an fp16 `softmax` followed by an fp16
/// `log` whose `epsilon = 0x1p-149` guard is far below fp16's smallest
/// subnormal and therefore inert (see [`DEFAULT_ENCODER_COMPUTE`]). On an ANE
/// placement every softmax output under the fp16 floor underflows to 0 and
/// `log(0)` saturates to ≈ `-45440`, silently replacing ordinary log-probs of
/// `-19.0` … `-21.75` and shifting real word timings by hundreds of ms.
///
/// The encoder is built from [`DEFAULT_ENCODER_COMPUTE`] — NEVER a hardcoded
/// placement — so this is a test of the shipping default. Flipping that
/// constant to `ComputeUnits::All` makes it fail (measured `min = -45440`,
/// 2,667 of 15,950 cells past the threshold); on `CpuOnly` it passes
/// (`min = -30.81`).
///
/// It must run on REAL SPEECH. This bug is invisible to synthetic input:
/// measured on the same model, 960,000 samples of silence bottom out at
/// `min = -8.55` and a low-amplitude sine at `-9.07` — both far ABOVE the fp16
/// floor (`log(2⁻²⁴) ≈ -16.6`), so nothing underflows and an `All` run of
/// either passes clean. Only real speech drives per-class probabilities down to
/// `e^-30.8 ≈ 4e-14`, deep under the floor. Hence the cross-crate `jfk.wav`
/// borrow.
///
/// [`LOG_PROB_FLOOR`] is not a tolerance to be relaxed: it separates two
/// populations three orders of magnitude apart (worst legitimate log-prob
/// measured anywhere on this model ≈ `-30.8`; the sentinel ≈ `-45440`).
/// Anything in between is already a broken emission matrix.
///
/// This measures the RAW tensor. `emissions_reject_an_ane_corrupted_matrix`
/// pins the same fact at the public door, where it is now an error rather than
/// a measurement.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_have_no_fp16_log_zero_sentinel() {
  let encoder = load_encoder();
  let samples = load_jfk_wav();
  let raw = encoder
    .emissions_raw(window_input(&samples))
    .expect("emissions on jfk.wav");

  let min = raw.data.iter().copied().fold(f32::INFINITY, f32::min);
  let sentinels = raw.data.iter().filter(|v| **v < LOG_PROB_FLOOR).count();
  assert_eq!(
    sentinels,
    0,
    "{sentinels} of {} emission cells are below {LOG_PROB_FLOOR} (min = {min}) — the fp16 \
     `log(0)` sentinel. The encoder is on {:?}; an ANE placement corrupts this model's emissions \
     and cannot be used. See DEFAULT_ENCODER_COMPUTE.",
    raw.data.len(),
    DEFAULT_ENCODER_COMPUTE,
  );
}

/// **THE SILENT-CORRUPTION REGRESSION.** An ANE-corrupted emission matrix must
/// be REJECTED by the public door, not returned as a plausible `Ok`.
///
/// [`EncoderOptions::with_compute`] is public and accepts `ComputeUnits::All`.
/// Before [`LOG_PROB_FLOOR`] existed, this exact call returned **`Ok`**: the
/// `-45440` sentinel is finite and `<= 0`, so it satisfies every check
/// [`Emissions::from_log_probs`] runs, and the caller got word timings that were
/// wrong by up to 881 ms with no diagnostic anywhere. Measured on the real
/// model, pre-guard: `Aligner::align_chunk(jfk, …)` → `Ok`, with `ask` at
/// 7533.7 ms instead of 8415.3 ms.
///
/// REAL SPEECH is load-bearing, and a synthetic input cannot replace it: on the
/// corrupt path 960,000 samples of digital silence bottom out at `-8.55` and a
/// low-amplitude sine at `-9.07`, both ABOVE the fp16 floor
/// (`log(2⁻²⁴) ≈ -16.6`), so nothing underflows and this test would pass
/// **against the corrupt model**. Only real speech drives a class posterior
/// under the floor. Hence the cross-crate `jfk.wav` borrow.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_reject_an_ane_corrupted_matrix() {
  let encoder = Encoder::from_file_with(
    encoder_path(),
    EncoderOptions::new().with_compute(ComputeUnits::All),
  )
  .expect("load base960h_aligner.mlmodelc on ComputeUnits::All");
  let samples = load_jfk_wav();

  let Err(err) = encoder.emissions(window_input(&samples)) else {
    panic!(
      "an ANE-corrupted emission matrix was accepted. `Emissions::from_log_probs` cannot catch \
       this — -45440 is finite and <= 0 — so the caller now has plausible, silently wrong word \
       timings. LOG_PROB_FLOOR is the only thing standing here."
    );
  };
  let AlignError::CorruptEmissions {
    compute,
    min,
    cells,
    total,
  } = err
  else {
    panic!("expected AlignError::CorruptEmissions, got {err:?}");
  };
  // The measured ANE signature, pinned: 2,667 of 15,950 cells (16.7%),
  // min = -45440. Asserted as bounds rather than as equalities — the exact
  // count is a property of one OS/ANE firmware pair, but the ORDER of the
  // corruption is the fact worth pinning.
  assert_eq!(compute, ComputeUnits::All);
  assert_eq!(total, 550 * crate::vocab::VOCAB_SIZE);
  assert!(
    cells > 0 && cells <= total,
    "corrupt cells: {cells}/{total}"
  );
  assert!(
    min < LOG_PROB_FLOOR,
    "reported min {min} must be past the floor it tripped"
  );
  // Self-diagnosing: the message must NAME the placement, or the caller is
  // left to rediscover a 450×-slower, 16.7%-corrupt configuration by hand.
  let rendered = AlignError::CorruptEmissions {
    compute,
    min,
    cells,
    total,
  }
  .to_string();
  assert!(
    rendered.contains("All"),
    "error must name the placement: {rendered}"
  );
  println!("rejected with: {rendered}");
}

/// The guard keys on the emission VALUES, never on the placement — so a
/// non-default but numerically-clean placement must still be accepted.
///
/// `CpuAndGpu` is that placement: measured `min = -30.02`, zero cells past
/// [`LOG_PROB_FLOOR`] on the same real speech the ANE corrupts. A guard that
/// rejected "any non-default compute" would fail here, and would also forbid a
/// future re-converted artifact that runs correctly on the ANE. This test is
/// what keeps the fix a value-domain check instead of a placement ban.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_accept_the_cpu_and_gpu_placement() {
  let encoder = Encoder::from_file_with(
    encoder_path(),
    EncoderOptions::new().with_compute(ComputeUnits::CpuAndGpu),
  )
  .expect("load base960h_aligner.mlmodelc on ComputeUnits::CpuAndGpu");
  let samples = load_jfk_wav();

  let emissions = encoder
    .emissions(window_input(&samples))
    .expect("CpuAndGpu emissions are clean log-probs and must pass the floor guard");
  assert_eq!(emissions.frames(), 550);
  assert_eq!(emissions.vocab().get(), crate::vocab::VOCAB_SIZE);
}

/// The shipping default on the same real speech, through the SAME guarded door
/// the ANE test fails at — the third leg of the placement-agnostic proof
/// (`CpuOnly` Ok, `CpuAndGpu` Ok, `All` Err).
///
/// `emissions_wraps_into_validated_emissions` covers the door on silence, which
/// (as `emissions_reject_an_ane_corrupted_matrix` explains) never reaches the
/// failure regime at all — so it cannot stand in for this.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_accept_the_default_placement_on_real_speech() {
  let encoder = load_encoder();
  let samples = load_jfk_wav();
  let emissions = encoder
    .emissions(window_input(&samples))
    .unwrap_or_else(|e| panic!("the SHIPPING placement must produce clean log-probs: {e}"));
  assert_eq!(emissions.frames(), 550);
  assert_eq!(emissions.vocab().get(), crate::vocab::VOCAB_SIZE);
}

/// Decodes the 11 s `jfk.wav` fixture (16 kHz mono int16) to f32 samples.
///
/// Borrowed from the whisperkit crate by relative path rather than committing
/// a second copy — the same borrow `tests/common/mod.rs` makes, and it FAILS
/// LOUDLY (never skips) if that path ever moves.
fn load_jfk_wav() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("../whisperkit/tests/fixtures/audio/jfk.wav");
  let mut reader = hound::WavReader::open(&path)
    .unwrap_or_else(|e| panic!("open the jfk.wav fixture at {path:?}: {e}"));
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_wraps_into_validated_emissions() {
  // The wrapped door: proves `Emissions::from_log_probs`' O(T·V) value scan
  // passes on the real model's output (the fp16 log-prob ceiling holds), and
  // that the shape handshake (`frames`/`vocab`) survives the wrap.
  let encoder = load_encoder();
  let samples = vec![0.0f32; 48_000];
  let emissions = encoder
    .emissions(window_input(&samples))
    .expect("emissions wraps into a validated Emissions");
  assert_eq!(emissions.frames(), 150);
  assert_eq!(emissions.vocab().get(), crate::vocab::VOCAB_SIZE);
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_on_short_input_truncates_to_hermetic_formula() {
  let encoder = load_encoder();
  // 3 s @ 16 kHz: well under the model's 2,999-frame ceiling, so the
  // real model's output must match the pure hermetic formula exactly
  // (cross-validates `truncated_frame_count` against the live model,
  // not just itself).
  let samples = vec![0.0f32; 48_000];
  let raw = encoder
    .emissions_raw(window_input(&samples))
    .expect("emissions on short input");
  assert_eq!(raw.frames, truncated_frame_count(48_000, encoder.frames()));
  assert_eq!(raw.frames, 150);
  assert_eq!(raw.data.len(), 150 * crate::vocab::VOCAB_SIZE);
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_is_deterministic_across_repeated_calls() {
  let encoder = load_encoder();
  // Small-amplitude non-zero signal, not pure silence, so this exercises
  // real signal-path compute rather than just a bias/floor.
  let samples: Vec<f32> = (0..ENCODER_WINDOW_SAMPLES)
    .map(|i| 0.01 * (i as f32 * 0.001).sin())
    .collect();
  let first = encoder
    .emissions_raw(window_input(&samples))
    .expect("first emissions call");
  let second = encoder
    .emissions_raw(window_input(&samples))
    .expect("second emissions call");
  assert_eq!(first.frames, second.frames);
  assert_eq!(
    first.data, second.data,
    "repeated emissions_raw() must be bit-identical"
  );
}
