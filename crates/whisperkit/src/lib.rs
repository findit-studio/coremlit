//! Rust port of WhisperKit (speech-to-text on CoreML).
//!
//! Pure-Rust port of Argmax's WhisperKit pipeline on top of the `coremlit`
//! CoreML runtime layer. Unlike `coremlit`'s flat re-exports, this crate
//! exposes its modules publicly per the spec's module map (constants,
//! errors, options, tokenizer, model, audio, decode, transcribe, ...);
//! later plans fill the map in task by task.
//!
//! macOS only. Swift source of truth: `argmax-oss-swift`.
//!
//! Note on scope: [`model`] ships the model-lifecycle *vocabulary*
//! (`ModelState`, `ModelVariant`, folder/glob detection, `ModelInfo`,
//! `SupportConfig`) and the `ModelLoader` seam, but not `ModelManager` —
//! Swift's coalesced load/unload/prewarm orchestrator. That belongs with
//! the backend that actually loads models (`backend`, Plan 3), so it is
//! deferred there rather than living here ahead of anything to drive it.

pub mod audio;
pub mod constants;
pub mod error;
pub mod model;
pub mod options;
pub mod result;
pub mod text;
pub mod tokenizer;
