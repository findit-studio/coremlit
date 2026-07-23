//! Native CoreML **CED-tiny** AudioSet sound-event tagging — coremlit's first
//! multi-label classifier: 16 kHz mono waveform in, ranked AudioSet predictions
//! out (527 rated classes: name + `/m/…` id + class index + sigmoid
//! confidence), long clips via windowed chunking + Mean/Max aggregation.
//!
//! CED (Consistent Ensemble Distillation, arXiv 2308.11957; upstream
//! RicherMans/CED, `mispeech/ced-tiny`) is a distilled AudioSet transformer.
//! The mel front-end runs in Rust (the private `mel` submodule) and the
//! mel→logits transformer runs natively on Apple silicon as one fp16
//! `.mlmodelc` — an in-graph STFT/mel is the exact fragility class behind the
//! ORT CoreML EP zeroed-logits bug this feature closes. NO `ort` anywhere.
//!
//! Design spec: `docs/superpowers/specs/2026-07-23-ced-native-ane-design.md`.
//!
//! macOS only (built on [`crate`]).

use crate::ComputeUnits;

#[cfg(test)]
mod tests;

/// The sample rate this module's contract is defined at: callers decode and
/// resample to **16 kHz mono f32** before calling (sans-I/O — the workspace
/// convention; CED natively matches it).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// The fixed inference-window length in samples: 160 000 = 10 s at 16 kHz,
/// CED's training window. The CoreML export is fixed-shape, so this is model
/// geometry, not a knob (soundevents exposes `window_samples` only because its
/// ONNX graph is dynamic-length — recorded non-goal).
pub const WINDOW_SAMPLES: usize = 160_000;

/// Number of AudioSet classes the model scores: the 527 released rated classes.
/// Compile-time-pinned to `RatedSoundEvent::events().len()` below, so the
/// dataset crate and this module can never drift apart silently.
pub const NUM_CLASSES: usize = 527;

const _: () = assert!(
  soundevents_dataset::RatedSoundEvent::events().len() == NUM_CLASSES,
  "soundevents-dataset's rated label set must have exactly NUM_CLASSES entries"
);

/// Default compute placement: [`ComputeUnits::All`].
///
/// **PROVISIONAL** — placement is measured, never marketed, and the CED
/// conversion has not been measured yet: the Wave-C placement pass
/// (`tests/ced/placement.rs`) re-pins this to the measured winner (the spec
/// anticipates `CpuAndGpu`; ANE-capable ≠ floor-holding — the siglip lesson)
/// and this doc then carries the measured latency × placement table.
pub const DEFAULT_COMPUTE: ComputeUnits = ComputeUnits::All;
