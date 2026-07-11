//! Voice-activity detection: the pluggable trait and the energy detector.
//!
//! Ports `WhisperKit/Core/Audio/VoiceActivityDetector.swift` (the base
//! class's shared chunk/silence utilities become provided trait methods)
//! and `EnergyVAD.swift`.

use crate::constants::SAMPLE_RATE;

#[cfg(test)]
mod tests;

/// Default VAD frame length: 0.1 s at 16 kHz (`EnergyVAD.swift:18`).
pub const DEFAULT_FRAME_LENGTH_SAMPLES: usize = (SAMPLE_RATE as usize) / 10;
/// Default VAD frame overlap: none (`EnergyVAD.swift:19`).
pub const DEFAULT_FRAME_OVERLAP_SAMPLES: usize = 0;
/// Default energy threshold (`EnergyVAD.swift:20`).
pub const DEFAULT_ENERGY_THRESHOLD: f32 = 0.02;

/// Frame-level voice-activity detection over 16 kHz mono PCM.
///
/// Implementors supply per-frame activity; the provided methods port the
/// frame/segment utilities every detector shares
/// (`VoiceActivityDetector.swift`).
pub trait VoiceActivityDetector {
  /// One activity flag per [`Self::frame_length_samples`]-sized frame
  /// (final partial frame included).
  fn voice_activity(&self, samples: &[f32]) -> Vec<bool>;

  /// Samples per analysis frame.
  fn frame_length_samples(&self) -> usize;

  /// Merges consecutive active frames into half-open `(start, end)` frame
  /// ranges (`VoiceActivityDetector.swift:52-77` in frame terms).
  fn active_chunks(&self, frames: &[bool]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut current_start: Option<usize> = None;
    for (index, &active) in frames.iter().enumerate() {
      match (active, current_start) {
        (true, None) => current_start = Some(index),
        (false, Some(start)) => {
          result.push((start, index));
          current_start = None;
        }
        _ => {}
      }
    }
    if let Some(start) = current_start {
      result.push((start, frames.len()));
    }
    result
  }

  /// First sample index of a frame.
  fn frame_to_sample(&self, frame: usize) -> usize {
    frame * self.frame_length_samples()
  }

  /// Start time of a frame in seconds.
  fn frame_to_seconds(&self, frame: usize) -> f32 {
    self.frame_to_sample(frame) as f32 / SAMPLE_RATE as f32
  }

  /// Longest run of inactive frames, as a half-open `(start, end)` frame
  /// range (`VoiceActivityDetector.swift:95-128`). `None` when no frame is
  /// inactive.
  fn find_longest_silence(&self, frames: &[bool]) -> Option<(usize, usize)> {
    let mut longest: Option<(usize, usize)> = None;
    let mut index = 0;
    while index < frames.len() {
      if frames[index] {
        index += 1;
        continue;
      }
      let start = index;
      while index < frames.len() && !frames[index] {
        index += 1;
      }
      if longest.is_none_or(|(s, e)| index - start > e - s) {
        longest = Some((start, index));
      }
    }
    longest
  }
}

/// Energy-threshold voice-activity detector (`EnergyVAD.swift`).
///
/// Frames whose RMS energy exceeds the threshold are active. Follows the
/// options pattern: [`Self::new`] is the default configuration and
/// `Default` delegates to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyVad {
  frame_length_samples: usize,
  frame_overlap_samples: usize,
  energy_threshold: f32,
}

impl Default for EnergyVad {
  fn default() -> Self {
    Self::new()
  }
}

impl EnergyVad {
  /// The default detector: 0.1 s frames, no overlap, threshold 0.02.
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      frame_length_samples: DEFAULT_FRAME_LENGTH_SAMPLES,
      frame_overlap_samples: DEFAULT_FRAME_OVERLAP_SAMPLES,
      energy_threshold: DEFAULT_ENERGY_THRESHOLD,
    }
  }

  /// The energy threshold.
  #[inline(always)]
  pub const fn energy_threshold(&self) -> f32 {
    self.energy_threshold
  }

  /// Sets the energy threshold.
  #[inline(always)]
  pub const fn set_energy_threshold(&mut self, threshold: f32) -> &mut Self {
    self.energy_threshold = threshold;
    self
  }

  /// Consuming form of [`Self::set_energy_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_energy_threshold(mut self, threshold: f32) -> Self {
    self.energy_threshold = threshold;
    self
  }

  /// Samples each frame extends into its successor.
  #[inline(always)]
  pub const fn frame_overlap_samples(&self) -> usize {
    self.frame_overlap_samples
  }

  /// Sets the frame overlap.
  #[inline(always)]
  pub const fn set_frame_overlap_samples(&mut self, samples: usize) -> &mut Self {
    self.frame_overlap_samples = samples;
    self
  }

  /// Consuming form of [`Self::set_frame_overlap_samples`].
  #[must_use]
  #[inline(always)]
  pub const fn with_frame_overlap_samples(mut self, samples: usize) -> Self {
    self.frame_overlap_samples = samples;
    self
  }

  /// Sets the frame length.
  #[inline(always)]
  pub const fn set_frame_length_samples(&mut self, samples: usize) -> &mut Self {
    self.frame_length_samples = samples;
    self
  }

  /// Consuming form of [`Self::set_frame_length_samples`].
  #[must_use]
  #[inline(always)]
  pub const fn with_frame_length_samples(mut self, samples: usize) -> Self {
    self.frame_length_samples = samples;
    self
  }
}

impl VoiceActivityDetector for EnergyVad {
  fn voice_activity(&self, samples: &[f32]) -> Vec<bool> {
    super::voice_activity_in_chunks(
      samples,
      self.frame_length_samples,
      self.frame_overlap_samples,
      self.energy_threshold,
    )
  }

  fn frame_length_samples(&self) -> usize {
    self.frame_length_samples
  }
}
