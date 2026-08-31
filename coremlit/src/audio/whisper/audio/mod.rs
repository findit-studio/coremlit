//! Audio DSP over 16 kHz mono PCM — the crate's sans-I/O boundary.
//!
//! Decoding, resampling, and capture are the caller's domain; everything
//! here operates on `&[f32]` samples already at
//! [`SAMPLE_RATE`](crate::audio::whisper::constants::SAMPLE_RATE). Ports the pure-math
//! statics of `WhisperKit/Core/Audio/AudioProcessor.swift`.

pub mod chunker;
pub mod vad;

#[cfg(test)]
mod tests;

/// Pads `samples` with trailing zeros, or truncates, to exactly `len`.
///
/// Ports `AudioProcessing.padOrTrimAudio`'s core semantics
/// (`AudioProcessor.swift`): Whisper windows are always exactly
/// [`WINDOW_SAMPLES`](crate::audio::whisper::constants::WINDOW_SAMPLES) long.
pub fn pad_or_trim(samples: &[f32], len: usize) -> Vec<f32> {
  let mut out = Vec::with_capacity(len);
  let copy = samples.len().min(len);
  out.extend_from_slice(&samples[..copy]);
  out.resize(len, 0.0);
  out
}

/// Root-mean-square energy of a chunk.
///
/// Ports `AudioProcessor.calculateAverageEnergy`
/// (`AudioProcessor.swift:698-702`, `vDSP_rmsqv`). Empty input is `0.0`.
pub fn signal_energy(chunk: &[f32]) -> f32 {
  if chunk.is_empty() {
    return 0.0;
  }
  let sum_squares: f32 = chunk.iter().map(|s| s * s).sum();
  (sum_squares / chunk.len() as f32).sqrt()
}

/// Normalizes a chunk's RMS energy to `0..=1` against a reference floor.
///
/// Ports `AudioProcessor.calculateRelativeEnergy`
/// (`AudioProcessor.swift:724-741`): both energies convert to dB
/// (`20·log10`), and the signal's position between the reference floor and
/// full scale (0 dB — samples are `-1..=1`, so RMS never exceeds 1) becomes
/// the normalized value, clamped to `0..=1`. The reference is floored at
/// `1e-8` exactly as Swift does; Swift's `nil` reference default (`1e-3`,
/// "measured empirically in a silent room") is the caller's concern here —
/// pass it explicitly.
pub fn relative_energy(chunk_energy: f32, reference: f32) -> f32 {
  let reference_energy = reference.max(1e-8);
  let db_energy = 20.0 * chunk_energy.log10();
  let ref_energy = 20.0 * reference_energy.log10();
  let normalized = (db_energy - ref_energy) / (0.0 - ref_energy);
  normalized.clamp(0.0, 1.0)
}

/// Per-chunk voice-activity flags: RMS energy over `threshold`.
///
/// Ports `AudioProcessor.calculateVoiceActivityInChunks`
/// (`AudioProcessor.swift:674-693`): the signal is cut into
/// `chunk_len`-sample chunks (the final partial chunk is scored too), each
/// chunk extended by `overlap` samples into its successor to catch audio
/// starting exactly at a boundary, and each chunk's RMS is compared
/// strictly against `threshold` (Swift's default threshold is `0.022`; its
/// doc comment saying `0.05` is stale against its own code).
pub fn voice_activity_in_chunks(
  samples: &[f32],
  chunk_len: usize,
  overlap: usize,
  threshold: f32,
) -> Vec<bool> {
  if chunk_len == 0 || samples.is_empty() {
    return Vec::new();
  }
  let chunk_count = samples.len().div_ceil(chunk_len);
  (0..chunk_count)
    .map(|index| {
      let start = index * chunk_len;
      let end = (start + chunk_len + overlap).min(samples.len());
      signal_energy(&samples[start..end]) > threshold
    })
    .collect()
}

/// Whether the most recently pushed audio contains voice, per the
/// streaming [`relative_energy`] history recorded so far.
///
/// Ports `AudioProcessor.isVoiceDetected` (`AudioProcessor.swift:636-655`):
/// `relative_energy` holds one entry per completed 0.1 s frame (oldest
/// first — `crate::audio::whisper::stream`'s `EnergyTracker` is what builds this history
/// in practice). `next_buffer_seconds` (the duration of audio just pushed)
/// converts to a count of recent entries to `consider` (`/ 0.1`, one entry
/// per frame); Swift's `max(0, Int(...))` clamp is redundant once ported —
/// `as usize` on a negative float already saturates to `0` (and on NaN,
/// where Swift's `Int(Float)` would trap outright, this port is strictly
/// more permissive: `as usize` yields `0`). Within that considered suffix,
/// only the **oldest** `max(10, considered - 10)` entries are checked
/// against `silence_threshold`: the newest ~1 s (10 entries) is
/// deliberately excluded as possible trailing silence that just hasn't
/// been confirmed as the end of speech yet. A `consider`/suffix shorter
/// than 10 entries (little history yet) checks everything available,
/// exactly like Swift's `ArraySlice.prefix(n)` on a shorter slice.
pub fn is_voice_detected(
  relative_energy: &[f32],
  next_buffer_seconds: f32,
  silence_threshold: f32,
) -> bool {
  let consider = (next_buffer_seconds / 0.1) as usize;
  let suffix = &relative_energy[relative_energy.len().saturating_sub(consider)..];
  let check_len = suffix.len().saturating_sub(10).max(10).min(suffix.len());
  suffix[..check_len]
    .iter()
    .any(|&energy| energy > silence_threshold)
}
