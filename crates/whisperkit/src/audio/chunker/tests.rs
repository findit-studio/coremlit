use super::*;
use crate::{
  audio::vad::EnergyVad,
  constants::WINDOW_SAMPLES,
  result::{TranscriptionResult, TranscriptionSegment},
};

#[test]
fn seek_clips_from_timestamps() {
  assert_eq!(prepare_seek_clips(&[], 1000).unwrap(), vec![(0, 1000)]);
  assert_eq!(
    prepare_seek_clips(&[1.0, 2.0], 100_000).unwrap(),
    vec![(16_000, 32_000)]
  );
  assert_eq!(
    prepare_seek_clips(&[1.0], 100_000).unwrap(),
    vec![(16_000, 100_000)]
  );
  assert_eq!(
    prepare_seek_clips(&[0.0, 1.0, 2.0, 3.0], 100_000).unwrap(),
    vec![(0, 16_000), (32_000, 48_000)]
  );
}

#[test]
fn seek_clips_reject_negative_and_non_finite_timestamps() {
  use crate::error::AudioError;
  // A usize cast would saturate -0.5 to sample 0 and silently transcribe
  // from the start; Swift's signed indices make the loop guard skip such
  // clips instead. Loud rejection is the documented divergence.
  assert!(matches!(
    prepare_seek_clips(&[-0.5, 1.0], 100_000),
    Err(AudioError::InvalidClipRange { .. })
  ));
  assert!(matches!(
    prepare_seek_clips(&[f32::NAN], 100_000),
    Err(AudioError::InvalidClipRange { .. })
  ));
}

#[test]
fn short_audio_yields_single_chunk() {
  let vad = EnergyVad::new();
  let samples = vec![0.1f32; 100_000];
  let chunks = VadChunker::new().chunk_all(&vad, &samples, WINDOW_SAMPLES, &[(0, samples.len())]);
  assert_eq!(chunks.len(), 1);
  assert_eq!(chunks[0].seek_offset(), 0);
  assert_eq!(chunks[0].samples_slice().len(), 100_000);
}

#[test]
fn long_audio_splits_at_silence_midpoint() {
  // 35 s: speech 0-20 s, silence 20-25 s, speech 25-35 s -> split inside
  // the silence.
  let mut samples = vec![0.5f32; 20 * 16_000];
  samples.extend(vec![0.0f32; 5 * 16_000]);
  samples.extend(vec![0.5f32; 10 * 16_000]);
  let vad = EnergyVad::new();
  let chunks = VadChunker::new().chunk_all(&vad, &samples, WINDOW_SAMPLES, &[(0, samples.len())]);
  assert_eq!(chunks.len(), 2);
  let split = chunks[1].seek_offset();
  assert!(
    (20 * 16_000..25 * 16_000).contains(&split),
    "split at {split} not inside the silence"
  );
  assert_eq!(
    chunks
      .iter()
      .map(|c| c.samples_slice().len())
      .sum::<usize>(),
    samples.len()
  );
}

#[test]
fn tiny_clip_inside_window_padding_produces_no_chunks() {
  // Ports the Swift loop guard `startIndex < seekClipEnd - windowPadding`:
  // a clip shorter than the padding yields nothing (and must not
  // underflow), matching Swift's signed arithmetic.
  let vad = EnergyVad::new();
  let samples = vec![0.5f32; 600_000];
  let chunks = VadChunker::new().chunk_all(&vad, &samples, WINDOW_SAMPLES, &[(0, 1_000)]);
  assert!(chunks.is_empty());
}

#[test]
fn apply_seek_offsets_shifts_segments_and_words() {
  use crate::result::WordTiming;
  let mut segment = TranscriptionSegment::new();
  segment.set_seek(100).set_start(1.0).set_end(2.0);
  segment.set_words(vec![WordTiming::new("hi", vec![1], 1.0, 1.5, 0.9)]);
  let mut segments = vec![segment];
  apply_seek_offsets(&mut segments, 32_000); // 2.0 s at 16 kHz
  assert_eq!(segments[0].seek(), 32_100);
  assert_eq!(segments[0].start(), 3.0);
  assert_eq!(segments[0].end(), 4.0);
  assert_eq!(segments[0].words_slice()[0].start(), 3.0);
  assert_eq!(segments[0].words_slice()[0].end(), 3.5);
}

#[test]
fn apply_result_seek_offset_stamps_seek_time_and_shifts_nested_times() {
  use crate::result::WordTiming;
  let mut segment = TranscriptionSegment::new();
  segment.set_start(1.0).set_end(2.0);
  segment.set_words(vec![WordTiming::new("hi", vec![1], 1.0, 1.5, 0.9)]);
  let mut result = TranscriptionResult::new(
    "hi",
    vec![segment],
    "en",
    crate::result::TranscriptionTimings::new(),
  );
  apply_result_seek_offset(&mut result, 32_000);
  assert_eq!(result.seek_time(), Some(2.0));
  let segment = &result.segments_slice()[0];
  assert_eq!(segment.start(), 3.0);
  assert_eq!(segment.words_slice()[0].end(), 3.5);
}
