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
    Err(Error::AudioTooLong { len, max }) if len == WINDOW_SAMPLES + 1 && max == WINDOW_SAMPLES
  ));
}

#[test]
fn validate_reports_the_first_non_finite_sample() {
  let mut samples = vec![0.0f32; 100];
  samples[41] = f32::NAN;
  samples[43] = f32::INFINITY;
  assert!(matches!(
    validate_window_input(&samples),
    Err(Error::NonFiniteInput { index: 41 })
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
    Err(Error::NonFiniteInput { index }) if index == WINDOW_SAMPLES + 300
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
    Err(Error::NonFiniteOutput { index: 7 })
  ));
}

#[test]
fn describe_renders_shape_and_dtype() {
  assert_eq!(
    describe(&[1, 64, 1001], Some(DataType::F32)),
    "[1, 64, 1001] float32"
  );
  assert_eq!(describe(&[1, 527], None), "[1, 527] none");
}
