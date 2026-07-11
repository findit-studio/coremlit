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
//!   detection, device support) and
//!   [`ModelManager`](model::manager::ModelManager), the coalesced
//!   load/prewarm/unload orchestrator over
//!   [`LoadedModels`](model::manager::LoadedModels).
//! - [`audio`] — sans-I/O DSP over 16 kHz mono PCM: pad/trim, energy,
//!   VAD, long-form chunking.
//! - [`backend`] — [`InferenceBackend`](backend::InferenceBackend) trait
//!   (mel/encode/decode-step seam), [`ModelDims`](backend::ModelDims),
//!   [`AlignmentView`](backend::AlignmentView) (the borrowed
//!   cross-attention alignment slice word-timestamp code reads), the real
//!   [`CoreMlBackend`](backend::coreml::CoreMlBackend) driving the three
//!   loaded CoreML models, and the scripted
//!   [`MockBackend`](backend::mock::MockBackend) test double used for
//!   hermetic pipeline tests.
//! - [`text`] — zlib compression-ratio repetition signal
//!   ([`text::compression_ratio_of_tokens`]) and Whisper string
//!   normalization/trimming ([`text::normalized`],
//!   [`text::trim_special_token_chars`]), consumed by the decode loop's
//!   fallback checks and language post-processing.
//! - [`decode`] — autoregressive decoding: the per-window loop
//!   ([`decode::decode_text`]) and one-shot language detection
//!   ([`decode::detect_language`]) driven against an
//!   [`InferenceBackend`](backend::InferenceBackend), the prefill-prompt
//!   assembly that feeds both ([`decode::prefill_tokens`]), the per-step
//!   [`LogitsFilter`](decode::filter::LogitsFilter) chain the loop runs
//!   against each step's raw logits, and the
//!   [`GreedyTokenSampler`](decode::sampler::GreedyTokenSampler) that
//!   picks the next token from what the chain leaves unmasked.
//! - [`segment`] — how a decoded window becomes
//!   [`TranscriptionSegment`](result::TranscriptionSegment)s and the next
//!   seek offset
//!   ([`find_seek_point_and_segments`](segment::find_seek_point_and_segments)),
//!   plus the word-timestamp math
//!   [`transcribe::TranscribeTask::run`] wires into the pipeline behind
//!   [`DecodingOptions::word_timestamps`](options::DecodingOptions::word_timestamps):
//!   [`dynamic_time_warping`](segment::dynamic_time_warping) over a
//!   decoded-token x audio-frame alignment matrix,
//!   [`find_alignment`](segment::find_alignment),
//!   [`merge_punctuations`](segment::merge_punctuations), the word-duration
//!   heuristics
//!   ([`calculate_word_duration_constraints`](segment::calculate_word_duration_constraints),
//!   [`truncate_long_words_at_sentence_boundaries`](segment::truncate_long_words_at_sentence_boundaries)),
//!   and the orchestrating
//!   [`add_word_timestamps`](segment::add_word_timestamps).
//! - [`transcribe`] — [`WhisperKit`](transcribe::WhisperKit), the public
//!   pipeline entry point (`transcribe`/`transcribe_all`/
//!   `detect_language`), composing
//!   [`TranscribeTask`](transcribe::TranscribeTask) — the seek/window loop
//!   and temperature-fallback ladder that drives the decode stack over a
//!   full audio buffer — and, for VAD-chunked audio, folding per-chunk
//!   results together via
//!   [`merge_transcription_results`](result::merge_transcription_results).
//! - [`stream`] — push-based streaming vocabulary (spec §5.3 `stream`
//!   row): [`StreamState`](stream::StreamState) (Swift's
//!   `AudioStreamTranscriber.State`),
//!   [`AudioStreamOptions`](stream::AudioStreamOptions),
//!   [`StreamUpdate`](stream::StreamUpdate), and the early-stop gate
//!   [`should_stop_early`](stream::should_stop_early) (Swift's static
//!   `shouldStopEarly`) that a later push-based driver
//!   (`AudioStreamTranscriber`, Plan 4 T8) is built from.
//! - [`log`] — leveled logging with a replacing callback.
//!
//! # Examples
//!
//! ```no_run
//! use whisperkit::options::{DecodingOptions, Options};
//! use whisperkit::transcribe::WhisperKit;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let options = Options::new(
//!   "Models/whisperkit-coreml/openai_whisper-tiny",
//!   "Models/tokenizers/whisper-tiny",
//! );
//! let kit = WhisperKit::new(&options)?;
//! let audio: Vec<f32> = vec![0.0; 16_000]; // 1 s of 16 kHz mono PCM
//! let result = kit.transcribe(&audio, &DecodingOptions::new())?;
//! println!("{}", result.text());
//! # Ok(())
//! # }
//! ```
//!
//! Fallback-ladder decisions are pure functions over
//! [`DecodingOptions`](options::DecodingOptions)/
//! [`DecodingResult`](result::DecodingResult), independently testable
//! without a loaded model — a window whose compression ratio crosses the
//! threshold asks for a retry at the next temperature:
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
//! // The first-token flag comes from the decode loop (see
//! // `result::needs_fallback`).
//! let repetitive = DecodingResult::new()
//!   .with_avg_logprob(-0.4)
//!   .with_no_speech_prob(0.1)
//!   .with_compression_ratio(3.4);
//! assert_eq!(
//!   needs_fallback(false, &repetitive, &options),
//!   Some(FallbackReason::CompressionRatioThreshold),
//! );
//! ```

pub mod audio;
pub mod backend;
pub mod constants;
pub mod decode;
pub mod error;
pub mod log;
pub mod model;
pub mod options;
pub mod result;
pub mod segment;
pub mod stream;
pub mod text;
pub mod tokenizer;
pub mod transcribe;
