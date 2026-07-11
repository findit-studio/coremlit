//! Rust port of WhisperKit (speech-to-text on CoreML).
//!
//! Pure-Rust port of Argmax's WhisperKit pipeline on top of the `coremlit`
//! CoreML runtime layer. Unlike `coremlit`'s flat re-exports, this crate
//! exposes its modules publicly per the spec's module map (constants,
//! errors, options, tokenizer, model, audio, decode, transcribe, ...);
//! later plans fill the map in task by task.
//!
//! macOS only. Swift source of truth: `argmax-oss-swift`.

pub mod constants;
pub mod error;
pub mod options;
