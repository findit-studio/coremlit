//! VAD-based chunking of long audio into windows the encoder can take.
//!
//! Ports `WhisperKit/Core/Audio/AudioChunker.swift` (`VADAudioChunker`) and
//! `DecodingOptions.prepareSeekClips`
//! (`Utilities/Extensions+Internal.swift:112-130`).

use crate::audio::whisper::{
  audio::vad::VoiceActivityDetector,
  constants::SAMPLE_RATE,
  error::AudioError,
  result::{TranscriptionResult, TranscriptionSegment},
};

#[cfg(test)]
mod tests;

/// Default samples to stop early before a clip's end, preventing
/// end-of-clip hallucinations (`AudioChunker.swift:47`, 1 s at 16 kHz).
pub const DEFAULT_WINDOW_PADDING: usize = 16_000;

/// One chunk of a longer input: its samples and where they began.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
  seek_offset: usize,
  samples: Vec<f32>,
}

impl AudioChunk {
  /// Builds a chunk from its absolute start sample and samples.
  pub fn new(seek_offset: usize, samples: impl Into<Vec<f32>>) -> Self {
    Self {
      seek_offset,
      samples: samples.into(),
    }
  }

  /// Absolute sample index this chunk starts at in the original input.
  #[inline(always)]
  pub const fn seek_offset(&self) -> usize {
    self.seek_offset
  }

  /// The chunk's samples.
  #[inline(always)]
  pub const fn samples_slice(&self) -> &[f32] {
    self.samples.as_slice()
  }
}

/// Turns `clip_timestamps` seconds into `(start, end)` sample ranges.
///
/// Ports `DecodingOptions.prepareSeekClips`
/// (`Extensions+Internal.swift:112-130`): empty input becomes one
/// full-range clip; an odd final timestamp runs to `content_frames`; pairs
/// are consumed in order. Ordering and upper bounds are NOT validated,
/// like Swift.
///
/// # Errors
/// [`AudioError::InvalidClipRange`] for a negative or non-finite
/// timestamp. This is a deliberate, documented divergence: Swift keeps the
/// signed sample index and lets the chunk loop's bounds guard silently
/// skip such clips, but a `usize` cast would saturate `-0.5` to `0` and
/// silently transcribe from the start instead — rejecting loudly is the
/// only faithful-or-better option.
pub fn prepare_seek_clips(
  clip_timestamps: &[f32],
  content_frames: usize,
) -> Result<Vec<(usize, usize)>, AudioError> {
  if let Some(&bad) = clip_timestamps.iter().find(|t| !t.is_finite() || **t < 0.0) {
    return Err(AudioError::InvalidClipRange {
      start: bad,
      end: bad,
    });
  }
  let mut seek_points: Vec<usize> = clip_timestamps
    .iter()
    .map(|seconds| (seconds * SAMPLE_RATE as f32).round() as usize)
    .collect();
  if seek_points.is_empty() {
    seek_points.push(0);
  }
  if seek_points.len() % 2 == 1 {
    seek_points.push(content_frames);
  }
  Ok(
    seek_points
      .chunks(2)
      .map(|pair| (pair[0], pair[1]))
      .collect(),
  )
}

/// Shifts segments (seek, times, and word times) by an absolute chunk
/// offset, re-anchoring per-chunk results into the original timeline.
///
/// Ports `TranscriptionUtilities.updateSegmentTimings`
/// (`TranscriptionUtilities.swift:55-69`) applied per segment as the
/// chunker's `updateSeekOffsetsForResults` does (`AudioChunker.swift:14-39`).
/// Re-anchors a whole per-chunk result: stamps [`TranscriptionResult`]'s
/// `seek_time` and shifts every segment and word.
///
/// Ports the result-level half of `updateSeekOffsetsForResults`
/// (`AudioChunker.swift:14-39`, `seekTime` assignment at `:29`);
/// [`apply_seek_offsets`] is its per-segment core.
pub fn apply_result_seek_offset(result: &mut TranscriptionResult, seek_offset: usize) {
  let seek_seconds = seek_offset as f32 / crate::audio::whisper::constants::SAMPLE_RATE as f32;
  result.set_seek_time(seek_seconds);
  apply_seek_offsets(result.segments_slice_mut(), seek_offset);
}

/// Shifts segments (seek, times, and word times) by an absolute chunk
/// offset — the per-segment core of [`apply_result_seek_offset`].
///
/// Ports `TranscriptionUtilities.updateSegmentTimings`
/// (`TranscriptionUtilities.swift:55-69`).
pub fn apply_seek_offsets(segments: &mut [TranscriptionSegment], seek_offset: usize) {
  let seek_seconds = seek_offset as f32 / SAMPLE_RATE as f32;
  for segment in segments {
    segment.set_seek(segment.seek() + seek_offset);
    segment.set_start(segment.start() + seek_seconds);
    segment.set_end(segment.end() + seek_seconds);
    for word in segment.words_slice_mut() {
      word.set_start(word.start() + seek_seconds);
      word.set_end(word.end() + seek_seconds);
    }
  }
}

/// VAD-driven chunker: splits overlong stretches at the middle of the
/// longest silence in each candidate window's second half.
///
/// Ports `VADAudioChunker` (`AudioChunker.swift:43-108`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadChunker {
  window_padding: usize,
}

impl Default for VadChunker {
  fn default() -> Self {
    Self::new()
  }
}

impl VadChunker {
  /// A chunker with the default 1 s window padding.
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      window_padding: DEFAULT_WINDOW_PADDING,
    }
  }

  /// The early-stop padding in samples.
  #[inline(always)]
  pub const fn window_padding(&self) -> usize {
    self.window_padding
  }

  /// Sets the early-stop padding.
  #[inline(always)]
  pub const fn set_window_padding(&mut self, samples: usize) -> &mut Self {
    self.window_padding = samples;
    self
  }

  /// Consuming form of [`Self::set_window_padding`].
  #[must_use]
  #[inline(always)]
  pub const fn with_window_padding(mut self, samples: usize) -> Self {
    self.window_padding = samples;
    self
  }

  /// Splits `samples` into chunks of at most `max_len`, breaking at
  /// silence midpoints within the given seek clips.
  ///
  /// Ports `VADAudioChunker.chunkAll` (`AudioChunker.swift:66-108`): input
  /// no longer than `max_len` returns whole as one chunk regardless of
  /// clips; otherwise each clip is walked window by window, and a clip
  /// shorter than the window padding yields nothing (Swift's signed
  /// `startIndex < seekClipEnd - windowPadding` loop guard).
  pub fn chunk_all<V>(
    &self,
    vad: &V,
    samples: &[f32],
    max_len: usize,
    clip_ranges: &[(usize, usize)],
  ) -> Vec<AudioChunk>
  where
    V: VoiceActivityDetector + ?Sized,
  {
    if samples.len() <= max_len {
      return vec![AudioChunk::new(0, samples)];
    }
    let mut chunks = Vec::new();
    for &(clip_start, clip_end) in clip_ranges {
      let mut start = clip_start;
      // Swift compares signed values; usize subtraction would underflow
      // when the clip is shorter than the padding.
      while (start as i64) < clip_end as i64 - self.window_padding as i64 {
        if start >= samples.len() {
          break;
        }
        let mut end = clip_end;
        if start + max_len < end {
          end = self.split_on_middle_of_longest_silence(
            vad,
            samples,
            start,
            samples.len().min(start + max_len),
          );
        }
        if end <= start {
          break;
        }
        chunks.push(AudioChunk::new(
          start,
          &samples[start..end.min(samples.len())],
        ));
        start = end;
      }
    }
    chunks
  }

  /// Ports `splitOnMiddleOfLongestSilence` (`AudioChunker.swift:54-65`):
  /// only the window's second half is searched, aiming for the longest
  /// possible chunk; without silence the full window is kept.
  fn split_on_middle_of_longest_silence<V>(
    &self,
    vad: &V,
    samples: &[f32],
    start: usize,
    end: usize,
  ) -> usize
  where
    V: VoiceActivityDetector + ?Sized,
  {
    let mid = start + (end - start) / 2;
    let activity = vad.voice_activity(&samples[mid..end]);
    match vad.find_longest_silence(&activity) {
      Some((silence_start, silence_end)) => {
        let silence_mid = silence_start + (silence_end - silence_start) / 2;
        mid + vad.frame_to_sample(silence_mid)
      }
      None => end,
    }
  }
}
