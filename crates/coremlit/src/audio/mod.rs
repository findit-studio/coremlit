//! On-device audio-understanding pipelines.
//!
//! A namespace of feature-gated modules, each a self-contained CoreML pipeline
//! built over the always-compiled runtime core (`Model` / `MultiArray` /
//! `Features`). Every module is a former standalone kit crate, collapsed here
//! per the mono-crate restructure; enabling a module's feature is the only way
//! it compiles, and `default = []` pulls none of them.
//!
//! - `whisper` — Whisper speech-to-text (feature `whisper`).
//! - `align` — wav2vec2 forced word-level alignment (feature `align`;
//!   `align-oracle` adds the asry ONNX parity oracle).
//! - `speaker` — native CoreML segmentation/embedding backends for the `dia`
//!   diarization pipeline (feature `speaker`; `speaker-oracle` adds the dia-ort
//!   DER oracle).
//! - `vad` — Silero voice-activity detection (feature `vad`; `vad-bundled`
//!   adds the silero ONNX cross-backend oracle).
//! - `ced` — CED-tiny AudioSet sound-event tagging (feature `ced`).
//!
//! See the crate README's layering map for module authority and the dependency
//! arrows to the `silero`, `asry`, and `dia` seams.

#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "align")]
pub mod align;

#[cfg(feature = "speaker")]
pub mod speaker;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "ced")]
pub mod ced;
