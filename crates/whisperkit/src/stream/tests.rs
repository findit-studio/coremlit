use super::*;
use crate::{
  options::DecodingOptions,
  result::{TranscriptionProgress, TranscriptionSegment, TranscriptionTimings},
};

#[test]
fn state_change_callback_type_is_constructible_and_send() {
  // `(old, new)` per assignment, mirroring Swift's `didSet`
  // (AudioStreamTranscriber.swift:27-31). `Sync` on the callback should
  // make a shared reference to it `Send` (`&F: Send` iff `F: Sync`) —
  // verified here, not just asserted in the type's doc comment.
  let old = StreamState::new();
  let mut newer = StreamState::new();
  newer.set_current_fallbacks(1);

  let cb: StateChangeCallback<'_> = &|prev, next| {
    assert_eq!(prev.current_fallbacks(), 0);
    assert_eq!(next.current_fallbacks(), 1);
  };
  cb(&old, &newer);

  fn assert_send<T: Send>(_: &T) {}
  assert_send(&cb);
}

#[test]
fn stream_options_defaults_match_swift_init() {
  // AudioStreamTranscriber.swift:51-54.
  let options = AudioStreamOptions::new();
  assert_eq!(options.required_segments_for_confirmation(), 2);
  assert_eq!(options.silence_threshold(), 0.3);
  assert_eq!(options.compression_check_window(), 60);
  assert!(options.use_vad());
  assert_eq!(AudioStreamOptions::default(), AudioStreamOptions::new());
  let options = options
    .with_silence_threshold(0.5)
    .with_required_segments_for_confirmation(3);
  assert_eq!(options.silence_threshold(), 0.5);
  assert_eq!(options.required_segments_for_confirmation(), 3);
}

#[test]
fn stream_update_vocabulary() {
  assert_eq!(StreamUpdate::AwaitingVoice.as_str(), "awaiting_voice");
  assert_eq!(StreamUpdate::Transcribed.to_string(), "transcribed");
  assert!(StreamUpdate::AwaitingAudio.is_awaiting_audio());
}

fn progress_with(tokens: Vec<u32>, avg_logprob: Option<f32>) -> TranscriptionProgress {
  let mut progress = TranscriptionProgress::new(TranscriptionTimings::new(), String::new(), tokens);
  if let Some(avg) = avg_logprob {
    progress.set_avg_logprob(avg);
  }
  progress
}

#[test]
fn should_stop_early_matches_swift_decision_table() {
  let options = DecodingOptions::new();
  // 61 identical tokens (> window 60): ratio >> 2.4 -> stop.
  assert_eq!(
    should_stop_early(&progress_with(vec![42; 61], None), &options, 60),
    Some(false)
  );
  // Below the window, repetitive or not: no compression verdict.
  assert_eq!(
    should_stop_early(&progress_with(vec![42; 60], None), &options, 60),
    None
  );
  // Bad average logprob -> stop.
  assert_eq!(
    should_stop_early(&progress_with(vec![1, 2, 3], Some(-2.0)), &options, 60),
    Some(false)
  );
  // Clean -> keep decoding.
  assert_eq!(
    should_stop_early(&progress_with(vec![1, 2, 3], Some(-0.1)), &options, 60),
    None
  );
  // Faithful quirk (AudioStreamTranscriber.swift:217): a DISABLED compression
  // threshold compares against 0.0, so any long token run trips the stop.
  let disabled = DecodingOptions::new().maybe_compression_ratio_threshold(None);
  let varied: Vec<u32> = (0..61).collect();
  assert_eq!(
    should_stop_early(&progress_with(varied, None), &disabled, 60),
    Some(false)
  );
}

#[test]
fn energy_tracker_frames_and_first_frame_zero() {
  // AudioProcessor.swift:906-921: one entry per 1600-sample frame; the
  // first frame has no reference history and records 0 (NaN-clamp parity).
  let mut tracker = EnergyTracker::default();
  let mut buffer = vec![0.001f32; 2 * ENERGY_FRAME_SAMPLES];
  tracker.absorb(&buffer);
  assert_eq!(tracker.relative_energies().len(), 2);
  assert_eq!(tracker.relative_energies()[0], 0.0);
  // Loud frames against the quiet reference read near 1.
  buffer.extend(std::iter::repeat_n(0.5, ENERGY_FRAME_SAMPLES));
  tracker.absorb(&buffer);
  let energies = tracker.relative_energies();
  assert_eq!(energies.len(), 3);
  assert!(
    energies[2] > 0.5,
    "loud-after-quiet is high relative energy, got {}",
    energies[2]
  );
  // Partial frames wait for completion.
  buffer.extend(std::iter::repeat_n(0.5, 10));
  tracker.absorb(&buffer);
  assert_eq!(tracker.relative_energies().len(), 3);
}

#[test]
fn stream_state_defaults_and_pub_crate_mutation() {
  // AudioStreamTranscriber.swift:7-17 (State), minus `isRecording` (mic
  // lifecycle, dropped — see module doc). `set_*` is `pub(crate)`: only
  // this crate's own state machine (Plan 4 T8) mutates a session's state
  // in practice, but the vocabulary earns its own coverage here, ahead of
  // that consumer.
  let mut state = StreamState::new();
  assert_eq!(state, StreamState::default());
  assert_eq!(state.current_fallbacks(), 0);
  assert_eq!(state.last_buffer_size(), 0);
  assert_eq!(state.last_confirmed_segment_end_seconds(), 0.0);
  assert!(state.buffer_energy_slice().is_empty());
  assert!(state.current_text().is_empty());
  assert!(state.confirmed_segments_slice().is_empty());
  assert!(state.unconfirmed_segments_slice().is_empty());
  assert!(state.unconfirmed_text_slice().is_empty());

  state.set_current_fallbacks(2);
  state.set_last_buffer_size(1_600);
  state.set_last_confirmed_segment_end_seconds(3.5);
  state.set_buffer_energy(vec![0.1, 0.2]);
  state.set_current_text("hello");
  let segment = TranscriptionSegment::new().with_text("hi");
  state.set_confirmed_segments(vec![segment.clone()]);
  state.set_unconfirmed_segments(vec![segment]);
  state.set_unconfirmed_text(vec!["stale".to_string()]);

  assert_eq!(state.current_fallbacks(), 2);
  assert_eq!(state.last_buffer_size(), 1_600);
  assert_eq!(state.last_confirmed_segment_end_seconds(), 3.5);
  assert_eq!(state.buffer_energy_slice().to_vec(), vec![0.1, 0.2]);
  assert_eq!(state.current_text(), "hello");
  assert_eq!(state.confirmed_segments_slice().len(), 1);
  assert_eq!(state.confirmed_segments_slice()[0].text(), "hi");
  assert_eq!(state.unconfirmed_segments_slice().len(), 1);
  assert_eq!(state.unconfirmed_segments_slice()[0].text(), "hi");
  assert_eq!(
    state.unconfirmed_text_slice().to_vec(),
    vec!["stale".to_string()]
  );
}

#[cfg(feature = "serde")]
#[test]
fn stream_options_partial_config_falls_back_to_defaults() {
  // Options-pattern serde pairing (review finding): every field carries a
  // fn-default, so a partial document inherits new()'s values.
  let partial: AudioStreamOptions = serde_json::from_str(r#"{"use_vad":false}"#).unwrap();
  assert!(!partial.use_vad());
  assert_eq!(
    partial.required_segments_for_confirmation(),
    DEFAULT_REQUIRED_SEGMENTS_FOR_CONFIRMATION
  );
  assert_eq!(partial.silence_threshold(), DEFAULT_SILENCE_THRESHOLD);
  assert_eq!(
    partial.compression_check_window(),
    DEFAULT_COMPRESSION_CHECK_WINDOW
  );
  let round: AudioStreamOptions =
    serde_json::from_str(&serde_json::to_string(&AudioStreamOptions::new()).unwrap()).unwrap();
  assert_eq!(round, AudioStreamOptions::new());
}
