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
