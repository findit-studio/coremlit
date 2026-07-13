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
// EncoderOptions
// ---------------------------------------------------------------------

#[test]
fn options_new_defaults_to_all_compute() {
  assert_eq!(EncoderOptions::new().compute(), ComputeUnits::All);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(EncoderOptions::default(), EncoderOptions::new());
}

#[test]
fn options_with_compute_overrides() {
  let options = EncoderOptions::new().with_compute(ComputeUnits::CpuOnly);
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_set_compute_in_place() {
  let mut options = EncoderOptions::new();
  options.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(options.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_compute_defaults_to_all() {
  let options: EncoderOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(options.compute(), ComputeUnits::All);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips_explicit_compute() {
  let options: EncoderOptions = serde_json::from_str(r#"{"compute":"cpu_only"}"#).unwrap();
  assert_eq!(options.compute(), ComputeUnits::CpuOnly);
  let json = serde_json::to_string(&options).unwrap();
  assert!(json.contains("cpu_only"), "round-tripped json: {json}");
}

// ---------------------------------------------------------------------
// Encoder: model-gated (requires a local base960h_aligner.mlmodelc,
// ALIGNKIT_TEST_MODELS or Models/alignkit/, same convention as
// tests/model_io.rs's `common` module and tests/common/mod.rs).
// Duplicated here in miniature because unit tests under `src/` cannot
// import the separate `tests/` integration-test crate (mirrors
// dia-coreml::segment::tests's identical duplication and rationale).
// Real-audio, cross-tool parity testing against asry's ONNX path lives
// in `tests/parity_emissions.rs` (Gate 1), not here — these tests use
// only synthetic signals, matching dia-coreml's own src/-level
// convention.
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

/// Loads the real encoder model with `ComputeUnits::CpuOnly` — matching
/// `tests/model_io.rs`'s introspection convention: deterministic, no ANE
/// compile-latency variance across runs. [`DEFAULT_ENCODER_COMPUTE`]
/// (`ComputeUnits::All`) stays the production default.
fn load_encoder() -> Encoder {
  Encoder::from_file_with(
    encoder_path(),
    EncoderOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load base960h_aligner.mlmodelc")
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
fn emissions_rejects_input_too_long() {
  let encoder = load_encoder();
  let samples = vec![0.0f32; ENCODER_WINDOW_SAMPLES + 1];
  let Err(err) = encoder.emissions_raw(&samples, samples.len()) else {
    panic!("longer-than-window input must be rejected");
  };
  assert!(matches!(
    err,
    AlignError::InputTooLong {
      got,
      max
    } if got == ENCODER_WINDOW_SAMPLES + 1 && max == ENCODER_WINDOW_SAMPLES
  ));
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_on_full_window_produces_correctly_shaped_finite_log_probs() {
  let encoder = load_encoder();
  let samples = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
  let raw = encoder
    .emissions_raw(&samples, samples.len())
    .expect("emissions on silence");
  assert_eq!(raw.frames(), encoder.frames());
  assert_eq!(raw.vocab(), crate::vocab::VOCAB_SIZE);
  assert!(
    raw.data().iter().all(|v| v.is_finite()),
    "all log-probs finite"
  );
  // Log-probabilities are bounded above by log(1) == 0. This is also the
  // exact domain `Emissions::from_log_probs` enforces, so a pass here is a
  // canary that `Encoder::emissions` (the wrapped door) will not trip the
  // value-domain scan on this input.
  assert!(
    raw.data().iter().all(|&v| v <= 0.0),
    "log-probs must satisfy log(p) <= 0"
  );
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
    .emissions(&samples, samples.len())
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
    .emissions_raw(&samples, samples.len())
    .expect("emissions on short input");
  assert_eq!(
    raw.frames(),
    truncated_frame_count(48_000, encoder.frames())
  );
  assert_eq!(raw.frames(), 150);
  assert_eq!(raw.vocab(), crate::vocab::VOCAB_SIZE);
  assert_eq!(raw.data().len(), 150 * crate::vocab::VOCAB_SIZE);
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
    .emissions_raw(&samples, samples.len())
    .expect("first emissions call");
  let second = encoder
    .emissions_raw(&samples, samples.len())
    .expect("second emissions call");
  assert_eq!(first.frames(), second.frames());
  assert_eq!(first.vocab(), second.vocab());
  assert_eq!(
    first.data(),
    second.data(),
    "repeated emissions_raw() must be bit-identical"
  );
}
