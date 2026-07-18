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
//!   decision ([`needs_fallback`](result::needs_fallback)), plus the
//!   [`writer`](result::writer) submodule's transcript writers:
//!   [`SrtWriter`](result::writer::SrtWriter) and
//!   [`VttWriter`](result::writer::VttWriter) always, and — behind the
//!   `serde` feature — `JsonWriter`, all behind the shared
//!   [`ResultWriter`](result::writer::ResultWriter) trait.
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
//!   [`merge_transcription_results_with_options`](result::merge_transcription_results_with_options),
//!   which is equally the entry point for folding a `transcribe_all` batch
//!   by hand ([`merge_transcription_results`](result::merge_transcription_results)
//!   is its options-blind, exactly-Swift sibling).
//! - [`stream`] — push-based streaming vocabulary (spec §5.3 `stream`
//!   row): [`StreamState`](stream::StreamState) (Swift's
//!   `AudioStreamTranscriber.State`),
//!   [`AudioStreamOptions`](stream::AudioStreamOptions),
//!   [`StreamUpdate`](stream::StreamUpdate), and the early-stop gate
//!   [`should_stop_early`](stream::should_stop_early) (Swift's static
//!   `shouldStopEarly`), driving the push-based state machine
//!   [`AudioStreamTranscriber`](stream::AudioStreamTranscriber): push new
//!   samples through
//!   [`push_samples`](stream::AudioStreamTranscriber::push_samples) as
//!   they arrive and read [`StreamState`](stream::StreamState) back for
//!   the session's live transcript. [`stream::agreement`] adds the
//!   LocalAgreement-2 confirmation engine
//!   [`LocalAgreement`](stream::agreement::LocalAgreement) and its
//!   simulated-stream driver
//!   [`LocalAgreementTranscriber`](stream::agreement::LocalAgreementTranscriber).
//!   [`WhisperKit`](transcribe::WhisperKit) builds either streamer from an
//!   already-constructed pipeline
//!   ([`audio_stream_transcriber`](transcribe::WhisperKit::audio_stream_transcriber)/
//!   [`local_agreement_transcriber`](transcribe::WhisperKit::local_agreement_transcriber)).
//! - [`task_facts`] — [`TaskFacts`](task_facts::TaskFacts), the single record
//!   of the decode-time facts a run *controls* (RNG draw, observed language,
//!   early-stop truncation, worker coordinates, allocated id span), with the
//!   one associative [merge law](task_facts::TaskFacts::merge) every merge
//!   entry point calls. Carried on every
//!   [`TranscriptionResult`](result::TranscriptionResult) and embedded in
//!   [`Provenance`](provenance::Provenance).
//! - [`log`] — leveled logging with a replacing callback.
//!
//! # Reproducibility and provenance
//!
//! The same audio through the same model can produce different text,
//! tokens, and segments when decode options drift — coremlit issue #9's
//! round-1–4 validation found exact Rust/Swift parity under one pinned
//! configuration and observable divergence when single knobs moved (e.g.
//! VAD-chunked with prefill was parity-pass; the same run without prefill
//! was not). Consumers that index, snapshot, or regression-test
//! transcripts should therefore set the decode policy **explicitly**
//! rather than relying on defaults, and record the full configuration
//! alongside every stored transcript or benchmark artifact:
//!
//! - model folder identity and revision (e.g. the Hugging Face repo id +
//!   revision the `.mlmodelc` folder came from — this crate loads local
//!   folders and does not know their provenance;
//!   [`Options::model_folder`](options::Options::model_folder) is only
//!   the path)
//! - tokenizer identity and revision (same caveat;
//!   [`Options::tokenizer_folder`](options::Options::tokenizer_folder))
//! - compute units, per stage
//!   ([`ComputeOptions`](options::ComputeOptions))
//! - chunking strategy
//!   ([`DecodingOptions::chunking_strategy`](options::DecodingOptions::chunking_strategy))
//! - language, when known
//!   ([`DecodingOptions::language`](options::DecodingOptions::language) —
//!   set it explicitly instead of leaving auto-detection to pick one)
//! - prefill
//!   ([`DecodingOptions::use_prefill_prompt`](options::DecodingOptions::use_prefill_prompt))
//! - special-token skipping
//!   ([`DecodingOptions::skip_special_tokens`](options::DecodingOptions::skip_special_tokens))
//! - word timestamps
//!   ([`DecodingOptions::word_timestamps`](options::DecodingOptions::word_timestamps))
//! - VAD strategy (the detector driving VAD chunking —
//!   [`EnergyVad`](audio::vad::EnergyVad) by default, swappable via
//!   [`WhisperKit::set_vad_detector`](transcribe::WhisperKit::set_vad_detector);
//!   Silero VAD is available opt-in behind the `vadkit` feature as
//!   `silero_vad::SileroVad`)
//!
//! [`Provenance`](provenance::Provenance) assembles that record for you
//! (coremlit issue #14). Of the items above, the **library-known** ones are
//! captured by
//! [`Provenance::from_options`](provenance::Provenance::from_options) —
//! which **embeds the entire resolved
//! [`DecodingOptions`](options::DecodingOptions)**
//! ([`Provenance::decoding`](provenance::Provenance::decoding)) rather than
//! copying a chosen handful of its knobs into flat fields of its own,
//! alongside the [`ComputeOptions`](options::ComputeOptions) and the
//! *effective* decode temperature.
//!
//! "Wholesale" is load-bearing and not a figure of speech. A curated
//! projection is a list somebody must remember to extend, and this one had
//! already drifted to 11 of 30 knobs — silently dropping, among others,
//! `drop_blank_audio` and `word_grouping`, both of which visibly change the
//! transcript, so two runs differing only in them left byte-identical
//! records. Embedding the struct makes the record complete **by
//! construction**: an option added tomorrow is recorded with no change to
//! the provenance code, and `provenance`'s mutation table — whose coverage
//! is derived from [`DecodingOptions`](options::DecodingOptions)' own field
//! set, not hand-written — fails the suite if one ever is not.
//!
//! The remaining three are **consumer-supplied**: settable `Option` fields,
//! left `None` and never guessed, because this crate genuinely cannot
//! observe them.
//!
//! - Model identity and tokenizer identity — it loads bare local folders,
//!   so it never sees a repo id or a revision.
//! - The **VAD detector**
//!   ([`Provenance::vad_detector`](provenance::Provenance::vad_detector)).
//!   This one is not an oversight: the pipeline holds the detector as a
//!   `Box<dyn `[`VoiceActivityDetector`](audio::vad::VoiceActivityDetector)`>`
//!   ([`WhisperKit::vad_detector`](transcribe::WhisperKit::vad_detector)),
//!   and a trait object carries no identity to read — nor does it live in
//!   either options struct, so
//!   [`from_options`](provenance::Provenance::from_options) could not
//!   reach it even if it had a name. The record therefore says *whether*
//!   VAD chunking ran
//!   ([`chunking_strategy`](options::DecodingOptions::chunking_strategy)),
//!   never *which* detector drove it. Since a swapped detector moves the
//!   chunk boundaries — and the boundaries move the transcript — a
//!   consumer that calls
//!   [`set_vad_detector`](transcribe::WhisperKit::set_vad_detector) should
//!   name it here, or two runs that differ only in detector leave
//!   byte-identical records with no trace of what made their text differ.
//!
//! Under the `serde` feature the whole record serializes (unset
//! consumer-supplied fields omitted, so they can never masquerade as a
//! known `null`), as do [`DecodingOptions`](options::DecodingOptions),
//! [`ComputeOptions`](options::ComputeOptions), and
//! [`Options`](options::Options) individually.
//!
//! Provenance-adjacent behaviors to plan for:
//!
//! - **Compute units affect output.** The same model/audio/options can
//!   yield different transcripts across `cpuOnly`/`cpuAndGPU`/
//!   `cpuAndNeuralEngine`/`all` — CoreML backend numeric drift, not a bug
//!   in this port (Rust and Swift match each other when the unit
//!   matches). Never compare outputs across compute units as if
//!   equivalent: fix one unit for regression baselines, or keep a
//!   separate baseline per unit.
//! - **Silence decodes to a marker, and this crate drops it by default.**
//!   Confirmed under the validated configuration coremlit issue #9 pinned
//!   (tiny model, matched VAD/prefill options, 5 s of digital silence):
//!   the window decodes to exactly
//!   [`BLANK_AUDIO_MARKER`](constants::BLANK_AUDIO_MARKER) — see that
//!   constant's doc for the general, model/config-dependent shape of this
//!   behavior. Rather than leave every product layer to post-filter that,
//!   [`DecodingOptions::drop_blank_audio`](options::DecodingOptions::drop_blank_audio)
//!   defaults `true` and removes such segments after decoding, so pure
//!   silence yields an empty result (coremlit issue #14). This is the one
//!   default in this crate that deliberately diverges from Swift, which
//!   emits the marker; set the option `false` for exact Swift parity.
//! - **Sampling above temperature 0 is non-reproducible by default, on
//!   both runtimes — set [`DecodingOptions::seed`](options::DecodingOptions::seed) to make it
//!   reproducible on this side.** Whenever the fallback ladder retries at
//!   `temperature > 0`, upstream Swift draws from an unseeded RNG
//!   (`Float.random`, `TokenSampler.swift:169`), and with
//!   [`DecodingOptions::seed`](options::DecodingOptions::seed) left `None` (the default) this port's
//!   sampler matches that non-determinism
//!   ([`GreedyTokenSampler::new`](decode::sampler::GreedyTokenSampler::new)
//!   seeds from the OS via `StdRng::from_os_rng`) — two identical unseeded
//!   runs that fall back can legitimately differ, by design, exactly like
//!   Swift, and this is the byte-unchanged default path. Setting
//!   [`DecodingOptions::seed`](options::DecodingOptions::seed) to `Some(_)` makes the full
//!   [`WhisperKit::transcribe`](transcribe::WhisperKit::transcribe)/
//!   [`transcribe_all`](transcribe::WhisperKit::transcribe_all) pipeline
//!   **bit-reproducible across runs, with the fallback ladder still fully
//!   enabled**: [`transcribe::TranscribeTask`] derives an independent
//!   per-(worker, window, attempt) sub-seed from the base seed
//!   ([`decode::sampler::derive_attempt_seed`] — see its doc for the exact
//!   mixing function and why one seed can't just be reused verbatim
//!   everywhere), so the same audio/options/seed always samples the same
//!   tokens end to end, while different windows and different fallback
//!   attempts still draw distinct, uncorrelated streams. A seed makes
//!   **this port's own** output reproducible; it does not, and cannot,
//!   make that output match Swift's draws — Swift has no seed knob at
//!   all, and its sampling stays unseeded regardless of anything set on
//!   this side. Direct callers of [`decode::decode_text`]/
//!   [`decode::detect_language`] (this crate's own tests included) sit one
//!   layer below the pipeline and supply their own sampler; they can call
//!   [`GreedyTokenSampler::with_seed`](decode::sampler::GreedyTokenSampler::with_seed)
//!   directly (optionally through
//!   [`decode::sampler::derive_attempt_seed`] too, to replicate the
//!   pipeline's own seed schedule) for the same reproducibility at that
//!   layer. Either way, seeded or not, record the effective temperature
//!   (the initial value plus any fallback increments actually taken) in
//!   provenance — already carried per window by
//!   [`TranscriptionSegment::temperature`](result::TranscriptionSegment::temperature)/
//!   [`DecodingResult::temperature`](result::DecodingResult::temperature) —
//!   since it marks exactly which outputs carry sampling randomness.
//! - **Swift CLI comparison pitfall: `--language` forces prefill on.**
//!   The upstream Swift CLI resolves
//!   `usePrefillPrompt: arguments.usePrefillPrompt || arguments.language
//!   != nil || task == .translate` (`TranscribeCLIUtils.swift:56`), so
//!   passing `--language` to it silently turns prefill ON regardless of
//!   the flag under test. A parity or benchmark harness that sets
//!   `--language` while intending a no-prefill configuration is
//!   comparing mismatched configurations — pin the language through
//!   [`DecodingOptions::language`](options::DecodingOptions::language)
//!   on this side and double-check what the Swift side actually resolved
//!   before attributing any divergence to the ports.
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
//!
//! # Streaming example
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
//! let mut streamer = kit.audio_stream_transcriber(DecodingOptions::new());
//! loop {
//!   let samples: Vec<f32> = vec![0.0; 16_000]; // 1 s of 16 kHz mono from the caller's source
//!   let update = streamer.push_samples(&samples)?;
//!   if update.is_transcribed() {
//!     for segment in streamer.state().confirmed_segments_slice() {
//!       println!("confirmed: {}", segment.text());
//!     }
//!     break;
//!   }
//! }
//! # Ok(())
//! # }
//! ```

pub mod audio;
pub mod backend;
pub mod constants;
pub mod decode;
pub mod error;
pub mod log;
pub mod model;
pub mod options;
pub mod provenance;
pub mod result;
pub mod segment;
/// Silero VAD (`vadkit`-backed) as an opt-in [`VoiceActivityDetector`], behind
/// the `vadkit` feature — the energy VAD stays the default.
///
/// [`VoiceActivityDetector`]: audio::vad::VoiceActivityDetector
#[cfg(feature = "vadkit")]
pub mod silero_vad;
pub mod stream;
pub mod task_facts;
pub mod text;
pub mod tokenizer;
pub mod transcribe;
