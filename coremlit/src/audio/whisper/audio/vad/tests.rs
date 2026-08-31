use super::*;

#[test]
fn energy_thresholding_flags_loud_frames() {
  // EnergyVad default frame = 0.1 s = 1600 samples @ 16 kHz:
  // two silent frames then one loud frame.
  let mut samples = vec![0.0f32; 3200];
  samples.extend(std::iter::repeat_n(0.5, 1600));
  let vad = EnergyVad::new();
  let activity = vad.voice_activity(&samples);
  assert_eq!(activity.len(), 3);
  assert!(!activity[0]);
  assert!(*activity.last().unwrap());
}

#[test]
fn partial_trailing_frame_is_still_scored() {
  // 1.5 frames of loud audio: the trailing half frame forms its own chunk.
  let samples = vec![0.5f32; 2400];
  let vad = EnergyVad::new();
  assert_eq!(vad.voice_activity(&samples), vec![true, true]);
}

#[test]
fn longest_silence_between_speech() {
  let frames = [true, false, false, false, false, true];
  let vad = EnergyVad::new();
  let (start, end) = vad.find_longest_silence(&frames).unwrap();
  assert_eq!((start, end), (1, 5));
  assert!(vad.find_longest_silence(&[true, true]).is_none());
}

#[test]
fn active_frame_ranges_merges_consecutive_frames() {
  let frames = [false, true, true, false, true];
  let vad = EnergyVad::new();
  assert_eq!(vad.active_frame_ranges(&frames), vec![(1, 3), (4, 5)]);
  assert!(vad.active_frame_ranges(&[false, false]).is_empty());
}

#[test]
fn active_chunks_returns_clamped_sample_ranges() {
  // One silent frame then 1.5 loud frames: Swift returns sample ranges
  // sliceable from the waveform, final end clamped to its length.
  let mut samples = vec![0.0f32; 1600];
  samples.extend(vec![0.5f32; 2400]);
  let vad = EnergyVad::new();
  assert_eq!(vad.active_chunks(&samples), vec![(1600, 4000)]);
}

#[test]
fn frame_conversions_use_frame_length() {
  let vad = EnergyVad::new();
  assert_eq!(vad.frame_length_samples(), 1600);
  assert_eq!(vad.frame_to_sample(3), 4800);
  assert!((vad.frame_to_seconds(10) - 1.0).abs() < 1e-6);
}

#[test]
fn options_pattern_constructors() {
  assert_eq!(EnergyVad::default(), EnergyVad::new());
  assert_eq!(
    EnergyVad::new().energy_threshold(),
    DEFAULT_ENERGY_THRESHOLD
  );
  let vad = EnergyVad::new().with_energy_threshold(0.5);
  assert_eq!(vad.energy_threshold(), 0.5);
  let mut vad = EnergyVad::new();
  vad.set_frame_overlap_samples(160);
  assert_eq!(vad.frame_overlap_samples(), 160);
}
