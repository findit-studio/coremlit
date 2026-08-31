use super::*;

#[test]
fn pad_or_trim_exact() {
  assert_eq!(pad_or_trim(&[1.0, 2.0], 4), vec![1.0, 2.0, 0.0, 0.0]);
  assert_eq!(pad_or_trim(&[1.0, 2.0, 3.0], 2), vec![1.0, 2.0]);
  assert_eq!(pad_or_trim(&[1.0], 1), vec![1.0]);
}

#[test]
fn signal_energy_is_rms() {
  // RMS of a constant signal is its magnitude.
  assert!((signal_energy(&[0.5; 100]) - 0.5).abs() < 1e-6);
  assert_eq!(signal_energy(&[]), 0.0);
}

#[test]
fn relative_energy_normalizes_to_unit_range() {
  // Full-scale (RMS 1.0) is 0 dB -> 1.0 against any reference; at the
  // reference it is 0.0. Ports AudioProcessor.swift:724-741.
  assert!((relative_energy(1.0, 1e-3) - 1.0).abs() < 1e-6);
  assert!(relative_energy(1e-3, 1e-3).abs() < 1e-6);
  let mid = relative_energy(0.1, 1e-3);
  assert!(mid > 0.0 && mid < 1.0);
  // Below-reference energies clamp to 0, silence does not go negative.
  assert_eq!(relative_energy(1e-9, 1e-3), 0.0);
}

#[test]
fn voice_activity_in_chunks_thresholds_rms() {
  // Two silent 4-sample chunks then one loud chunk.
  let mut samples = vec![0.0f32; 8];
  samples.extend([0.5f32; 4]);
  assert_eq!(
    voice_activity_in_chunks(&samples, 4, 0, 0.022),
    vec![false, false, true]
  );
  // Overlap pulls the loud tail into the preceding chunk.
  assert_eq!(
    voice_activity_in_chunks(&samples, 4, 4, 0.022),
    vec![false, true, true]
  );
}

#[test]
fn voice_detection_checks_the_oldest_prefix_of_the_recent_window() {
  // AudioProcessor.swift:636-655: consider = seconds/0.1 entries; within
  // them, only the OLDEST max(10, n-10) are checked.
  let quiet_then_loud: Vec<f32> = std::iter::repeat_n(0.0, 10)
    .chain(std::iter::repeat_n(0.9, 10))
    .collect();
  // 20 considered, check prefix max(10, 10) = 10 -> all quiet -> NOT detected.
  assert!(!is_voice_detected(&quiet_then_loud, 2.0, 0.3));

  let loud_then_quiet: Vec<f32> = std::iter::repeat_n(0.9, 10)
    .chain(std::iter::repeat_n(0.0, 10))
    .collect();
  assert!(is_voice_detected(&loud_then_quiet, 2.0, 0.3));

  // Short history: prefix(10) of 5 entries = all 5.
  assert!(is_voice_detected(&[0.9; 5], 0.5, 0.3));
  assert!(!is_voice_detected(&[], 1.0, 0.3));
}
