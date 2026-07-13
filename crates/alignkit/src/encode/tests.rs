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
/// The `-100.0` threshold is not a tolerance to be relaxed: it separates two
/// populations three orders of magnitude apart (worst legitimate log-prob
/// measured anywhere on this model ≈ `-30.8`; the sentinel ≈ `-45440`).
/// Anything in between is already a broken emission matrix.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_have_no_fp16_log_zero_sentinel() {
  const FLOOR: f32 = -100.0;

  let encoder = load_encoder();
  let samples = load_jfk_wav();
  let raw = encoder
    .emissions_raw(&samples, samples.len())
    .expect("emissions on jfk.wav");

  let min = raw.data.iter().copied().fold(f32::INFINITY, f32::min);
  let sentinels = raw.data.iter().filter(|v| **v < FLOOR).count();
  assert_eq!(
    sentinels,
    0,
    "{sentinels} of {} emission cells are below {FLOOR} (min = {min}) — the fp16 `log(0)` \
     sentinel. The encoder is on {:?}; an ANE placement corrupts this model's emissions and \
     cannot be used. See DEFAULT_ENCODER_COMPUTE.",
    raw.data.len(),
    DEFAULT_ENCODER_COMPUTE,
  );
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
    .emissions_raw(&samples, samples.len())
    .expect("first emissions call");
  let second = encoder
    .emissions_raw(&samples, samples.len())
    .expect("second emissions call");
  assert_eq!(first.frames, second.frames);
  assert_eq!(
    first.data, second.data,
    "repeated emissions_raw() must be bit-identical"
  );
}
