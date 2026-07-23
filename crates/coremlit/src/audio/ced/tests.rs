use super::*;
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
