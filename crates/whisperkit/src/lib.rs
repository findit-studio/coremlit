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
//! # Module map
//!
//! - [`constants`] — sample rate, window sizes, the language table.
//! - [`options`] — [`DecodingOptions`](options::DecodingOptions) (per-run
//!   knobs), [`Options`](options::Options) (construction config),
//!   [`ComputeOptions`](options::ComputeOptions).
//! - [`error`] — per-domain error types composing into
//!   [`TranscribeError`](error::TranscribeError).
//! - [`result`] — transcription value types and the temperature-fallback
//!   decision ([`needs_fallback`](result::needs_fallback)).
//! - [`tokenizer`] — the Whisper tokenizer facade and special tokens.
//! - [`model`] — model-lifecycle vocabulary (states, variants, folder
//!   detection, device support).
//! - [`audio`] — sans-I/O DSP over 16 kHz mono PCM: pad/trim, energy,
//!   VAD, long-form chunking.
//! - [`backend`] — [`InferenceBackend`](backend::InferenceBackend) trait
//!   (mel/encode/decode-step seam), [`ModelDims`](backend::ModelDims), and
//!   the scripted [`MockBackend`](backend::mock::MockBackend) test double
//!   used for hermetic pipeline tests.
//! - [`decode`] — autoregressive decoding: the per-window loop
//!   ([`decode::decode_text`]) and one-shot language detection
//!   ([`decode::detect_language`]) driven against an
//!   [`InferenceBackend`](backend::InferenceBackend), the prefill-prompt
//!   assembly that feeds both ([`decode::prefill_tokens`]), the per-step
//!   [`LogitsFilter`](decode::filter::LogitsFilter) chain the loop runs
//!   against each step's raw logits, and the
//!   [`GreedyTokenSampler`](decode::sampler::GreedyTokenSampler) that
//!   picks the next token from what the chain leaves unmasked.
//! - [`log`] — leveled logging with a replacing callback.
//!
//! # Example
//!
//! ```
//! use whisperkit::options::{ChunkingStrategy, DecodingOptions};
//! use whisperkit::result::{DecodingResult, FallbackReason, needs_fallback};
//!
//! let options = DecodingOptions::new()
//!   .with_temperature(0.2)
//!   .with_chunking_strategy(ChunkingStrategy::Vad);
//! assert_eq!(options.temperature(), 0.2);
//!
//! // A window whose compression ratio crosses the threshold asks for a
//! // retry at the next temperature; the first-token flag comes from the
//! // decode loop (see `result::needs_fallback`).
//! let repetitive = DecodingResult::new()
//!   .with_avg_logprob(-0.4)
//!   .with_no_speech_prob(0.1)
//!   .with_compression_ratio(3.4);
//! assert_eq!(
//!   needs_fallback(false, &repetitive, &options),
//!   Some(FallbackReason::CompressionRatioThreshold),
//! );
//! ```
//!
//! Note on scope: [`model`] ships the model-lifecycle *vocabulary*
//! (`ModelState`, `ModelVariant`, folder/glob detection, `ModelInfo`,
//! `SupportConfig`) and the `ModelLoader` seam, but not `ModelManager` —
//! Swift's coalesced load/unload/prewarm orchestrator. That belongs with
//! the backend that actually loads models (`backend`, Plan 3), so it is
//! deferred there rather than living here ahead of anything to drive it.

pub mod audio;
pub mod backend;
pub mod constants;
pub mod decode;
pub mod error;
pub mod log;
pub mod model;
pub mod options;
pub mod result;
pub mod text;
pub mod tokenizer;
