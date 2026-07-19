//! Transcription value types + the fallback-decision port (spec §6.3).
//! Ports `Models.swift` `WordTiming`/`TranscriptionSegment`/
//! `TranscriptionResultStruct`/`TranscriptionTimings`/`DecodingResult`/
//! `DecodingFallback`.
//!
//! Every type here is a plain owned value struct: `Clone` + `PartialEq`
//! (never `Eq` — each one carries `f32`/`f64` fields), no locks. Swift's
//! `TranscriptionResult` is a reference type guarded by a per-property
//! lock (`TranscriptionPropertyLock`); this port drops that entirely (spec
//! §6.3) — mutation (e.g. chunk seek-offset re-application) happens on
//! owned values before a result is ever returned, so ordinary Rust
//! ownership is enough, matching Swift's own lock-free
//! `TranscriptionResultStruct` sibling (`Models.swift:543-563`) rather
//! than the locked `TranscriptionResult` class. `serde` (optional
//! feature) never emits `null` for the Swift-parity fields; `Vec`/`String`
//! fields that carry meaningful "not present" semantics are
//! empty-means-absent (`skip_serializing_if` + `default`, golden §10). The
//! sole exception is the Rust-only [`TaskFacts`] a
//! [`TranscriptionResult`] carries, whose explicit-unknown reproducibility
//! facts serialize as a deliberate `null` (see that type) so a dropped key
//! is distinguishable from an unknown value.
//!
//! [`needs_fallback`]'s decision order and comparisons are copied verbatim
//! from Swift's `DecodingFallback.init?` (`Models.swift:357-381`) — see
//! its doc comment for the exact citations, including a correction to
//! this task's own brief (the "silence" short-circuit does not consult
//! `avg_logprob`, contrary to the brief's exploration).

use crate::audio::whisper::{
  constants::DEFAULT_LANGUAGE_CODE,
  options::DecodingOptions,
  task_facts::{SpanKnowledge, TaskFacts, TaskFactsAccumulator},
};

pub mod writer;

// ---------------------------------------------------------------------
// WordTiming
// ---------------------------------------------------------------------

/// A single word's decoded text, timing, and DTW alignment confidence
/// (Swift `WordTiming`, `Models.swift:622-641`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WordTiming {
  word: String,
  tokens: Vec<u32>,
  start: f32,
  end: f32,
  probability: f32,
}

impl WordTiming {
  /// Builds a word timing from its already-known fields. DTW alignment
  /// always has all five in hand at once, and Swift's `init` has no
  /// default parameters either (`Models.swift:634`) — so there is no
  /// partial or zero-value construction path.
  pub fn new(
    word: impl Into<String>,
    tokens: impl Into<Vec<u32>>,
    start: f32,
    end: f32,
    probability: f32,
  ) -> Self {
    Self {
      word: word.into(),
      tokens: tokens.into(),
      start,
      end,
      probability,
    }
  }

  /// The word's decoded text.
  #[inline(always)]
  pub fn word(&self) -> &str {
    self.word.as_str()
  }

  /// Token ids that decode to this word.
  #[inline(always)]
  pub const fn tokens_slice(&self) -> &[u32] {
    self.tokens.as_slice()
  }

  /// Start time, in seconds.
  #[inline(always)]
  pub const fn start(&self) -> f32 {
    self.start
  }

  /// End time, in seconds.
  #[inline(always)]
  pub const fn end(&self) -> f32 {
    self.end
  }

  /// Sets the start time, in seconds.
  #[inline(always)]
  pub const fn set_start(&mut self, start: f32) -> &mut Self {
    self.start = start;
    self
  }

  /// Sets the end time, in seconds.
  #[inline(always)]
  pub const fn set_end(&mut self, end: f32) -> &mut Self {
    self.end = end;
    self
  }

  /// DTW alignment confidence.
  #[inline(always)]
  pub const fn probability(&self) -> f32 {
    self.probability
  }

  /// `end - start` (Swift `WordTiming.duration`, `Models.swift:630-632`).
  #[inline(always)]
  pub const fn duration(&self) -> f32 {
    self.end - self.start
  }
}

// ---------------------------------------------------------------------
// TranscriptionSegment
// ---------------------------------------------------------------------

/// Default [`TranscriptionSegment::temperature`] (Swift
/// `Models.swift:601` — NOT `f32::default()`).
pub const DEFAULT_SEGMENT_TEMPERATURE: f32 = 1.0;
/// Default [`TranscriptionSegment::compression_ratio`] (Swift
/// `Models.swift:603` — NOT `f32::default()`).
pub const DEFAULT_SEGMENT_COMPRESSION_RATIO: f32 = 1.0;

#[cfg(feature = "serde")]
fn default_segment_temperature() -> f32 {
  DEFAULT_SEGMENT_TEMPERATURE
}
#[cfg(feature = "serde")]
fn default_segment_compression_ratio() -> f32 {
  DEFAULT_SEGMENT_COMPRESSION_RATIO
}

/// One transcribed segment of a window (Swift `TranscriptionSegment`,
/// `Models.swift:574-620`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptionSegment {
  /// Segment index within the transcription.
  #[cfg_attr(feature = "serde", serde(default))]
  id: usize,
  /// Seek position, in samples, this segment started decoding from.
  #[cfg_attr(feature = "serde", serde(default))]
  seek: usize,
  /// Start time, in seconds.
  #[cfg_attr(feature = "serde", serde(default))]
  start: f32,
  /// End time, in seconds.
  #[cfg_attr(feature = "serde", serde(default))]
  end: f32,
  /// Decoded text.
  #[cfg_attr(feature = "serde", serde(default))]
  text: String,
  /// Sampled token ids.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  tokens: Vec<u32>,
  /// Per-step `(token id, log probability)`.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  token_log_probs: Vec<(u32, f32)>,
  /// Sampling temperature this segment was decoded at.
  ///
  /// Bridged through the finite-float `serde` helper (codex round 3, F6) — the
  /// one segment float that is: it is what `provenance`'s
  /// `unanimous_temperature` reads for the effective temperature it records, so
  /// a non-finite value silently changing across a round
  /// trip would corrupt that record. The descriptive telemetry floats beside
  /// it (`avg_logprob`, `compression_ratio`, `no_speech_prob`) are left as-is:
  /// `compression_ratio` legitimately reaches `f32::INFINITY` on empty text.
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_segment_temperature",
      with = "crate::audio::whisper::options::finite_f32"
    )
  )]
  temperature: f32,
  /// Average sampled-token log probability.
  #[cfg_attr(feature = "serde", serde(default))]
  avg_logprob: f32,
  /// Compression ratio of `text` (repetition signal).
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_segment_compression_ratio")
  )]
  compression_ratio: f32,
  /// Probability this segment contains no speech.
  #[cfg_attr(feature = "serde", serde(default))]
  no_speech_prob: f32,
  /// Word-level timings from DTW alignment. Empty means word timestamps
  /// were not computed (golden empty-means-absent).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  words: Vec<WordTiming>,
}

impl Default for TranscriptionSegment {
  fn default() -> Self {
    Self::new()
  }
}

impl TranscriptionSegment {
  /// A segment matching Swift's all-default `init`
  /// (`Models.swift:593-606`).
  pub const fn new() -> Self {
    Self {
      id: 0,
      seek: 0,
      start: 0.0,
      end: 0.0,
      text: String::new(),
      tokens: Vec::new(),
      token_log_probs: Vec::new(),
      temperature: DEFAULT_SEGMENT_TEMPERATURE,
      avg_logprob: 0.0,
      compression_ratio: DEFAULT_SEGMENT_COMPRESSION_RATIO,
      no_speech_prob: 0.0,
      words: Vec::new(),
    }
  }

  // -- id -------------------------------------------------------------
  /// Segment index within the transcription.
  #[inline(always)]
  pub const fn id(&self) -> usize {
    self.id
  }
  /// Builder form of [`Self::set_id`].
  #[must_use]
  #[inline(always)]
  pub const fn with_id(mut self, id: usize) -> Self {
    self.set_id(id);
    self
  }
  /// Sets [`Self::id`] in place.
  #[inline(always)]
  pub const fn set_id(&mut self, id: usize) -> &mut Self {
    self.id = id;
    self
  }

  // -- seek -------------------------------------------------------------
  /// Seek position, in samples, this segment started decoding from.
  #[inline(always)]
  pub const fn seek(&self) -> usize {
    self.seek
  }
  /// Builder form of [`Self::set_seek`].
  #[must_use]
  #[inline(always)]
  pub const fn with_seek(mut self, seek: usize) -> Self {
    self.set_seek(seek);
    self
  }
  /// Sets [`Self::seek`] in place.
  #[inline(always)]
  pub const fn set_seek(&mut self, seek: usize) -> &mut Self {
    self.seek = seek;
    self
  }

  // -- start -------------------------------------------------------------
  /// Start time, in seconds.
  #[inline(always)]
  pub const fn start(&self) -> f32 {
    self.start
  }
  /// Builder form of [`Self::set_start`].
  #[must_use]
  #[inline(always)]
  pub const fn with_start(mut self, start: f32) -> Self {
    self.set_start(start);
    self
  }
  /// Sets [`Self::start`] in place.
  #[inline(always)]
  pub const fn set_start(&mut self, start: f32) -> &mut Self {
    self.start = start;
    self
  }

  // -- end -------------------------------------------------------------
  /// End time, in seconds.
  #[inline(always)]
  pub const fn end(&self) -> f32 {
    self.end
  }
  /// Builder form of [`Self::set_end`].
  #[must_use]
  #[inline(always)]
  pub const fn with_end(mut self, end: f32) -> Self {
    self.set_end(end);
    self
  }
  /// Sets [`Self::end`] in place.
  #[inline(always)]
  pub const fn set_end(&mut self, end: f32) -> &mut Self {
    self.end = end;
    self
  }

  // -- text -------------------------------------------------------------
  /// Decoded text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    self.text.as_str()
  }
  /// Builder form of [`Self::set_text`].
  #[must_use]
  #[inline(always)]
  pub fn with_text(mut self, text: impl Into<String>) -> Self {
    self.set_text(text);
    self
  }
  /// Sets [`Self::text`] in place.
  #[inline(always)]
  pub fn set_text(&mut self, text: impl Into<String>) -> &mut Self {
    self.text = text.into();
    self
  }

  // -- tokens (Vec<u32>) -------------------------------------------------
  /// Sampled token ids.
  #[inline(always)]
  pub const fn tokens_slice(&self) -> &[u32] {
    self.tokens.as_slice()
  }
  /// Builder form of [`Self::set_tokens`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokens(mut self, tokens: impl Into<Vec<u32>>) -> Self {
    self.set_tokens(tokens);
    self
  }
  /// Sets [`Self::tokens_slice`] in place.
  #[inline(always)]
  pub fn set_tokens(&mut self, tokens: impl Into<Vec<u32>>) -> &mut Self {
    self.tokens = tokens.into();
    self
  }

  // -- token_log_probs (Vec<(u32,f32)>) -----------------------------------
  /// Per-step `(token id, log probability)`.
  #[inline(always)]
  pub const fn token_log_probs_slice(&self) -> &[(u32, f32)] {
    self.token_log_probs.as_slice()
  }
  /// Builder form of [`Self::set_token_log_probs`].
  #[must_use]
  #[inline(always)]
  pub fn with_token_log_probs(mut self, token_log_probs: impl Into<Vec<(u32, f32)>>) -> Self {
    self.set_token_log_probs(token_log_probs);
    self
  }
  /// Sets [`Self::token_log_probs_slice`] in place.
  #[inline(always)]
  pub fn set_token_log_probs(&mut self, token_log_probs: impl Into<Vec<(u32, f32)>>) -> &mut Self {
    self.token_log_probs = token_log_probs.into();
    self
  }

  // -- temperature -------------------------------------------------------
  /// Sampling temperature this segment was decoded at.
  #[inline(always)]
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }
  /// Builder form of [`Self::set_temperature`].
  #[must_use]
  #[inline(always)]
  pub const fn with_temperature(mut self, temperature: f32) -> Self {
    self.set_temperature(temperature);
    self
  }
  /// Sets [`Self::temperature`] in place.
  #[inline(always)]
  pub const fn set_temperature(&mut self, temperature: f32) -> &mut Self {
    self.temperature = temperature;
    self
  }

  // -- avg_logprob -------------------------------------------------------
  /// Average sampled-token log probability.
  #[inline(always)]
  pub const fn avg_logprob(&self) -> f32 {
    self.avg_logprob
  }
  /// Builder form of [`Self::set_avg_logprob`].
  #[must_use]
  #[inline(always)]
  pub const fn with_avg_logprob(mut self, avg_logprob: f32) -> Self {
    self.set_avg_logprob(avg_logprob);
    self
  }
  /// Sets [`Self::avg_logprob`] in place.
  #[inline(always)]
  pub const fn set_avg_logprob(&mut self, avg_logprob: f32) -> &mut Self {
    self.avg_logprob = avg_logprob;
    self
  }

  // -- compression_ratio ---------------------------------------------------
  /// Compression ratio of [`Self::text`] (repetition signal).
  #[inline(always)]
  pub const fn compression_ratio(&self) -> f32 {
    self.compression_ratio
  }
  /// Builder form of [`Self::set_compression_ratio`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compression_ratio(mut self, compression_ratio: f32) -> Self {
    self.set_compression_ratio(compression_ratio);
    self
  }
  /// Sets [`Self::compression_ratio`] in place.
  #[inline(always)]
  pub const fn set_compression_ratio(&mut self, compression_ratio: f32) -> &mut Self {
    self.compression_ratio = compression_ratio;
    self
  }

  // -- no_speech_prob -------------------------------------------------------
  /// Probability this segment contains no speech.
  #[inline(always)]
  pub const fn no_speech_prob(&self) -> f32 {
    self.no_speech_prob
  }
  /// Builder form of [`Self::set_no_speech_prob`].
  #[must_use]
  #[inline(always)]
  pub const fn with_no_speech_prob(mut self, no_speech_prob: f32) -> Self {
    self.set_no_speech_prob(no_speech_prob);
    self
  }
  /// Sets [`Self::no_speech_prob`] in place.
  #[inline(always)]
  pub const fn set_no_speech_prob(&mut self, no_speech_prob: f32) -> &mut Self {
    self.no_speech_prob = no_speech_prob;
    self
  }

  // -- words (Vec<WordTiming>) ---------------------------------------------
  /// Word-level timings from DTW alignment. Empty means word timestamps
  /// were not computed.
  #[inline(always)]
  pub const fn words_slice(&self) -> &[WordTiming] {
    self.words.as_slice()
  }

  /// Mutable view of the per-word timings (fixed length; use
  /// [`Self::set_words`] to replace the collection). Word times carry no
  /// cross-field invariant, so in-place mutation is safe to expose — the
  /// chunker's seek re-anchoring shifts them directly.
  #[inline(always)]
  pub const fn words_slice_mut(&mut self) -> &mut [WordTiming] {
    self.words.as_mut_slice()
  }
  /// Builder form of [`Self::set_words`].
  #[must_use]
  #[inline(always)]
  pub fn with_words(mut self, words: impl Into<Vec<WordTiming>>) -> Self {
    self.set_words(words);
    self
  }
  /// Sets [`Self::words_slice`] in place.
  #[inline(always)]
  pub fn set_words(&mut self, words: impl Into<Vec<WordTiming>>) -> &mut Self {
    self.words = words.into();
    self
  }

  /// `end - start` (Swift `TranscriptionSegment.duration`,
  /// `Models.swift:588-591`).
  #[inline(always)]
  pub const fn duration(&self) -> f32 {
    self.end - self.start
  }
}

// ---------------------------------------------------------------------
// TranscriptionTimings
// ---------------------------------------------------------------------

/// Default [`TranscriptionTimings::pipeline_start`]/
/// [`TranscriptionTimings::first_token_time`]: a "not yet reached"
/// sentinel (Swift `Double.greatestFiniteMagnitude`, `Models.swift:810-
/// 811`).
pub const DEFAULT_PIPELINE_TIME_SENTINEL: f64 = f64::MAX;
/// Default [`TranscriptionTimings::input_audio_seconds`] floor (Swift
/// `0.001`, `Models.swift:812` — never a bare zero denominator).
pub const DEFAULT_INPUT_AUDIO_SECONDS: f64 = 0.001;

#[cfg(feature = "serde")]
fn default_pipeline_time_sentinel() -> f64 {
  DEFAULT_PIPELINE_TIME_SENTINEL
}
#[cfg(feature = "serde")]
fn default_input_audio_seconds() -> f64 {
  DEFAULT_INPUT_AUDIO_SECONDS
}

/// Pipeline timing accumulator (Swift `TranscriptionTimings`,
/// `Models.swift:730-844`): every stage duration/count Swift collects,
/// plus three projections computed from them
/// ([`Self::tokens_per_second`]/[`Self::real_time_factor`]/
/// [`Self::speed_factor`]). Every setter mutates in place and returns
/// `&mut Self` (no `with_*` builders — later pipeline stages accumulate
/// into a live `TranscriptionTimings` across many windows rather than
/// rebuilding one from scratch).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptionTimings {
  /// Absolute time the pipeline started; the sentinel means "not yet
  /// started".
  #[cfg_attr(feature = "serde", serde(default = "default_pipeline_time_sentinel"))]
  pipeline_start: f64,
  /// Absolute time the first token was produced; the sentinel means "not
  /// yet reached".
  #[cfg_attr(feature = "serde", serde(default = "default_pipeline_time_sentinel"))]
  first_token_time: f64,
  /// Length of the input audio, in seconds.
  #[cfg_attr(feature = "serde", serde(default = "default_input_audio_seconds"))]
  input_audio_seconds: f64,
  /// Total time loading all models.
  #[cfg_attr(feature = "serde", serde(default))]
  model_loading: f64,
  /// Time spent in the prewarm load/unload pass.
  #[cfg_attr(feature = "serde", serde(default))]
  prewarm_load_time: f64,
  /// Time spent loading the audio encoder model.
  #[cfg_attr(feature = "serde", serde(default))]
  encoder_load_time: f64,
  /// Time spent loading the text decoder model.
  #[cfg_attr(feature = "serde", serde(default))]
  decoder_load_time: f64,
  /// Time spent specializing the encoder for the Neural Engine.
  #[cfg_attr(feature = "serde", serde(default))]
  encoder_specialization_time: f64,
  /// Time spent specializing the decoder for the Neural Engine.
  #[cfg_attr(feature = "serde", serde(default))]
  decoder_specialization_time: f64,
  /// Time spent loading the tokenizer.
  #[cfg_attr(feature = "serde", serde(default))]
  tokenizer_load_time: f64,
  /// Time spent loading audio input.
  #[cfg_attr(feature = "serde", serde(default))]
  audio_loading: f64,
  /// Time spent padding/trimming decode windows to length — the per-window
  /// [`audio::pad_or_trim`](crate::audio::whisper::audio::pad_or_trim) in the
  /// seek loop, matching what Swift's `audioProcessing` bucket wraps
  /// (`TranscribeTask.swift`: slice + pad/trim + window preprocess). It does
  /// NOT include VAD chunking (`chunk_all`), which is untimed here and in
  /// Swift, nor mel extraction (tracked separately by [`Self::logmels`]).
  #[cfg_attr(feature = "serde", serde(default))]
  audio_processing: f64,
  /// Time spent computing mel-spectrogram features.
  #[cfg_attr(feature = "serde", serde(default))]
  logmels: f64,
  /// Time spent running the audio encoder.
  #[cfg_attr(feature = "serde", serde(default))]
  encoding: f64,
  /// Time spent initializing decoder state/matrices.
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_init: f64,
  /// Total time spent in the decode loop, across all windows.
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_loop: f64,
  /// Time spent on decoder model inference (predicting logits).
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_predictions: f64,
  /// Time spent applying logits filters.
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_filtering: f64,
  /// Time spent sampling the next token.
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_sampling: f64,
  /// Time spent evaluating and retrying temperature fallbacks.
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_fallback: f64,
  /// Time spent on window-level bookkeeping (includes word-timestamp
  /// time).
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_windowing: f64,
  /// Time spent updating the KV cache.
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_kv_caching: f64,
  /// Time spent computing word-level timestamps (DTW alignment).
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_word_timestamps: f64,
  /// Time spent in the decode loop outside of model inference.
  #[cfg_attr(feature = "serde", serde(default))]
  decoding_non_prediction: f64,
  /// Number of audio-processing passes run (divisor for
  /// [`Self::audio_processing`]).
  #[cfg_attr(feature = "serde", serde(default))]
  total_audio_processing_runs: f64,
  /// Number of mel-spectrogram passes run (divisor for [`Self::logmels`]).
  #[cfg_attr(feature = "serde", serde(default))]
  total_logmel_runs: f64,
  /// Number of encoder passes run (divisor for [`Self::encoding`]).
  #[cfg_attr(feature = "serde", serde(default))]
  total_encoding_runs: f64,
  /// Number of decode-loop steps (tokens sampled) across all windows.
  #[cfg_attr(feature = "serde", serde(default))]
  total_decoding_loops: f64,
  /// Number of KV-cache updates run (divisor for
  /// [`Self::decoding_kv_caching`]).
  #[cfg_attr(feature = "serde", serde(default))]
  total_kv_update_runs: f64,
  /// Number of word-timestamp alignment passes run.
  #[cfg_attr(feature = "serde", serde(default))]
  total_timestamp_alignment_runs: f64,
  /// Zero-based attempt index of the most recent temperature fallback — a
  /// faithful port of Swift's `totalDecodingFallbacks = Double(i)`
  /// (`TranscribeTask.swift`), *assigned* (not accumulated) on each
  /// fallback, so despite the name it is NOT a fallback count. The first
  /// fallback writes `0.0`, which is also this field's initial value, so
  /// `0.0` cannot distinguish "never fell back" from "one fallback at
  /// attempt 0"; a true count would be `attempt + 1`. A multi-window task
  /// overwrites per window, keeping only the last window's index, and
  /// merged results sum these per-result values. For an unambiguous "a
  /// fallback occurred" signal, use
  /// [`TaskFacts::drew_from_rng`](crate::audio::whisper::task_facts::TaskFacts::drew_from_rng).
  #[cfg_attr(feature = "serde", serde(default))]
  total_decoding_fallbacks: f64,
  /// Number of 30 s windows decoded.
  #[cfg_attr(feature = "serde", serde(default))]
  total_decoding_windows: f64,
  /// Total end-to-end pipeline duration.
  #[cfg_attr(feature = "serde", serde(default))]
  full_pipeline: f64,
}

impl Default for TranscriptionTimings {
  fn default() -> Self {
    Self::new()
  }
}

impl TranscriptionTimings {
  /// A fresh timings accumulator (Swift `TranscriptionTimings.init`,
  /// `Models.swift:778-843`): every duration/count starts at zero except
  /// the two "not yet reached" sentinels and the audio-seconds floor.
  pub const fn new() -> Self {
    Self {
      pipeline_start: DEFAULT_PIPELINE_TIME_SENTINEL,
      first_token_time: DEFAULT_PIPELINE_TIME_SENTINEL,
      input_audio_seconds: DEFAULT_INPUT_AUDIO_SECONDS,
      model_loading: 0.0,
      prewarm_load_time: 0.0,
      encoder_load_time: 0.0,
      decoder_load_time: 0.0,
      encoder_specialization_time: 0.0,
      decoder_specialization_time: 0.0,
      tokenizer_load_time: 0.0,
      audio_loading: 0.0,
      audio_processing: 0.0,
      logmels: 0.0,
      encoding: 0.0,
      decoding_init: 0.0,
      decoding_loop: 0.0,
      decoding_predictions: 0.0,
      decoding_filtering: 0.0,
      decoding_sampling: 0.0,
      decoding_fallback: 0.0,
      decoding_windowing: 0.0,
      decoding_kv_caching: 0.0,
      decoding_word_timestamps: 0.0,
      decoding_non_prediction: 0.0,
      total_audio_processing_runs: 0.0,
      total_logmel_runs: 0.0,
      total_encoding_runs: 0.0,
      total_decoding_loops: 0.0,
      total_kv_update_runs: 0.0,
      total_timestamp_alignment_runs: 0.0,
      total_decoding_fallbacks: 0.0,
      total_decoding_windows: 0.0,
      full_pipeline: 0.0,
    }
  }

  // -- pipeline_start ------------------------------------------------------
  /// Absolute time the pipeline started; the sentinel means "not yet started".
  #[inline(always)]
  pub const fn pipeline_start(&self) -> f64 {
    self.pipeline_start
  }
  /// Sets [`Self::pipeline_start`] in place.
  #[inline(always)]
  pub const fn set_pipeline_start(&mut self, pipeline_start: f64) -> &mut Self {
    self.pipeline_start = pipeline_start;
    self
  }

  // -- first_token_time ----------------------------------------------------
  /// Absolute time the first token was produced; the sentinel means "not yet reached".
  #[inline(always)]
  pub const fn first_token_time(&self) -> f64 {
    self.first_token_time
  }
  /// Sets [`Self::first_token_time`] in place.
  #[inline(always)]
  pub const fn set_first_token_time(&mut self, first_token_time: f64) -> &mut Self {
    self.first_token_time = first_token_time;
    self
  }

  // -- input_audio_seconds -------------------------------------------------
  /// Length of the input audio, in seconds.
  #[inline(always)]
  pub const fn input_audio_seconds(&self) -> f64 {
    self.input_audio_seconds
  }
  /// Sets [`Self::input_audio_seconds`] in place.
  #[inline(always)]
  pub const fn set_input_audio_seconds(&mut self, input_audio_seconds: f64) -> &mut Self {
    self.input_audio_seconds = input_audio_seconds;
    self
  }

  // -- model_loading -------------------------------------------------------
  /// Total time loading all models.
  #[inline(always)]
  pub const fn model_loading(&self) -> f64 {
    self.model_loading
  }
  /// Sets [`Self::model_loading`] in place.
  #[inline(always)]
  pub const fn set_model_loading(&mut self, model_loading: f64) -> &mut Self {
    self.model_loading = model_loading;
    self
  }

  // -- prewarm_load_time ---------------------------------------------------
  /// Time spent in the prewarm load/unload pass.
  #[inline(always)]
  pub const fn prewarm_load_time(&self) -> f64 {
    self.prewarm_load_time
  }
  /// Sets [`Self::prewarm_load_time`] in place.
  #[inline(always)]
  pub const fn set_prewarm_load_time(&mut self, prewarm_load_time: f64) -> &mut Self {
    self.prewarm_load_time = prewarm_load_time;
    self
  }

  // -- encoder_load_time ---------------------------------------------------
  /// Time spent loading the audio encoder model.
  #[inline(always)]
  pub const fn encoder_load_time(&self) -> f64 {
    self.encoder_load_time
  }
  /// Sets [`Self::encoder_load_time`] in place.
  #[inline(always)]
  pub const fn set_encoder_load_time(&mut self, encoder_load_time: f64) -> &mut Self {
    self.encoder_load_time = encoder_load_time;
    self
  }

  // -- decoder_load_time ---------------------------------------------------
  /// Time spent loading the text decoder model.
  #[inline(always)]
  pub const fn decoder_load_time(&self) -> f64 {
    self.decoder_load_time
  }
  /// Sets [`Self::decoder_load_time`] in place.
  #[inline(always)]
  pub const fn set_decoder_load_time(&mut self, decoder_load_time: f64) -> &mut Self {
    self.decoder_load_time = decoder_load_time;
    self
  }

  // -- encoder_specialization_time -----------------------------------------
  /// Time spent specializing the encoder for the Neural Engine.
  #[inline(always)]
  pub const fn encoder_specialization_time(&self) -> f64 {
    self.encoder_specialization_time
  }
  /// Sets [`Self::encoder_specialization_time`] in place.
  #[inline(always)]
  pub const fn set_encoder_specialization_time(
    &mut self,
    encoder_specialization_time: f64,
  ) -> &mut Self {
    self.encoder_specialization_time = encoder_specialization_time;
    self
  }

  // -- decoder_specialization_time -----------------------------------------
  /// Time spent specializing the decoder for the Neural Engine.
  #[inline(always)]
  pub const fn decoder_specialization_time(&self) -> f64 {
    self.decoder_specialization_time
  }
  /// Sets [`Self::decoder_specialization_time`] in place.
  #[inline(always)]
  pub const fn set_decoder_specialization_time(
    &mut self,
    decoder_specialization_time: f64,
  ) -> &mut Self {
    self.decoder_specialization_time = decoder_specialization_time;
    self
  }

  // -- tokenizer_load_time -------------------------------------------------
  /// Time spent loading the tokenizer.
  #[inline(always)]
  pub const fn tokenizer_load_time(&self) -> f64 {
    self.tokenizer_load_time
  }
  /// Sets [`Self::tokenizer_load_time`] in place.
  #[inline(always)]
  pub const fn set_tokenizer_load_time(&mut self, tokenizer_load_time: f64) -> &mut Self {
    self.tokenizer_load_time = tokenizer_load_time;
    self
  }

  // -- audio_loading -------------------------------------------------------
  /// Time spent loading audio input.
  #[inline(always)]
  pub const fn audio_loading(&self) -> f64 {
    self.audio_loading
  }
  /// Sets [`Self::audio_loading`] in place.
  #[inline(always)]
  pub const fn set_audio_loading(&mut self, audio_loading: f64) -> &mut Self {
    self.audio_loading = audio_loading;
    self
  }

  // -- audio_processing ----------------------------------------------------
  /// Time spent padding/trimming decode windows to length — the per-window
  /// [`audio::pad_or_trim`](crate::audio::whisper::audio::pad_or_trim) in the
  /// seek loop, matching what Swift's `audioProcessing` bucket wraps
  /// (`TranscribeTask.swift`: slice + pad/trim + window preprocess). It does
  /// NOT include VAD chunking (`chunk_all`), which is untimed here and in
  /// Swift, nor mel extraction (tracked separately by [`Self::logmels`]).
  #[inline(always)]
  pub const fn audio_processing(&self) -> f64 {
    self.audio_processing
  }
  /// Sets [`Self::audio_processing`] in place.
  #[inline(always)]
  pub const fn set_audio_processing(&mut self, audio_processing: f64) -> &mut Self {
    self.audio_processing = audio_processing;
    self
  }

  // -- logmels -------------------------------------------------------------
  /// Time spent computing mel-spectrogram features.
  #[inline(always)]
  pub const fn logmels(&self) -> f64 {
    self.logmels
  }
  /// Sets [`Self::logmels`] in place.
  #[inline(always)]
  pub const fn set_logmels(&mut self, logmels: f64) -> &mut Self {
    self.logmels = logmels;
    self
  }

  // -- encoding ------------------------------------------------------------
  /// Time spent running the audio encoder.
  #[inline(always)]
  pub const fn encoding(&self) -> f64 {
    self.encoding
  }
  /// Sets [`Self::encoding`] in place.
  #[inline(always)]
  pub const fn set_encoding(&mut self, encoding: f64) -> &mut Self {
    self.encoding = encoding;
    self
  }

  // -- decoding_init -------------------------------------------------------
  /// Time spent initializing decoder state/matrices.
  #[inline(always)]
  pub const fn decoding_init(&self) -> f64 {
    self.decoding_init
  }
  /// Sets [`Self::decoding_init`] in place.
  #[inline(always)]
  pub const fn set_decoding_init(&mut self, decoding_init: f64) -> &mut Self {
    self.decoding_init = decoding_init;
    self
  }

  // -- decoding_loop -------------------------------------------------------
  /// Total time spent in the decode loop, across all windows.
  #[inline(always)]
  pub const fn decoding_loop(&self) -> f64 {
    self.decoding_loop
  }
  /// Sets [`Self::decoding_loop`] in place.
  #[inline(always)]
  pub const fn set_decoding_loop(&mut self, decoding_loop: f64) -> &mut Self {
    self.decoding_loop = decoding_loop;
    self
  }

  // -- decoding_predictions ------------------------------------------------
  /// Time spent on decoder model inference (predicting logits).
  #[inline(always)]
  pub const fn decoding_predictions(&self) -> f64 {
    self.decoding_predictions
  }
  /// Sets [`Self::decoding_predictions`] in place.
  #[inline(always)]
  pub const fn set_decoding_predictions(&mut self, decoding_predictions: f64) -> &mut Self {
    self.decoding_predictions = decoding_predictions;
    self
  }

  // -- decoding_filtering --------------------------------------------------
  /// Time spent applying logits filters.
  #[inline(always)]
  pub const fn decoding_filtering(&self) -> f64 {
    self.decoding_filtering
  }
  /// Sets [`Self::decoding_filtering`] in place.
  #[inline(always)]
  pub const fn set_decoding_filtering(&mut self, decoding_filtering: f64) -> &mut Self {
    self.decoding_filtering = decoding_filtering;
    self
  }

  // -- decoding_sampling ---------------------------------------------------
  /// Time spent sampling the next token.
  #[inline(always)]
  pub const fn decoding_sampling(&self) -> f64 {
    self.decoding_sampling
  }
  /// Sets [`Self::decoding_sampling`] in place.
  #[inline(always)]
  pub const fn set_decoding_sampling(&mut self, decoding_sampling: f64) -> &mut Self {
    self.decoding_sampling = decoding_sampling;
    self
  }

  // -- decoding_fallback ---------------------------------------------------
  /// Time spent evaluating and retrying temperature fallbacks.
  #[inline(always)]
  pub const fn decoding_fallback(&self) -> f64 {
    self.decoding_fallback
  }
  /// Sets [`Self::decoding_fallback`] in place.
  #[inline(always)]
  pub const fn set_decoding_fallback(&mut self, decoding_fallback: f64) -> &mut Self {
    self.decoding_fallback = decoding_fallback;
    self
  }

  // -- decoding_windowing --------------------------------------------------
  /// Time spent on window-level bookkeeping (includes word-timestamp time).
  #[inline(always)]
  pub const fn decoding_windowing(&self) -> f64 {
    self.decoding_windowing
  }
  /// Sets [`Self::decoding_windowing`] in place.
  #[inline(always)]
  pub const fn set_decoding_windowing(&mut self, decoding_windowing: f64) -> &mut Self {
    self.decoding_windowing = decoding_windowing;
    self
  }

  // -- decoding_kv_caching -------------------------------------------------
  /// Time spent updating the KV cache.
  #[inline(always)]
  pub const fn decoding_kv_caching(&self) -> f64 {
    self.decoding_kv_caching
  }
  /// Sets [`Self::decoding_kv_caching`] in place.
  #[inline(always)]
  pub const fn set_decoding_kv_caching(&mut self, decoding_kv_caching: f64) -> &mut Self {
    self.decoding_kv_caching = decoding_kv_caching;
    self
  }

  // -- decoding_word_timestamps --------------------------------------------
  /// Time spent computing word-level timestamps (DTW alignment).
  #[inline(always)]
  pub const fn decoding_word_timestamps(&self) -> f64 {
    self.decoding_word_timestamps
  }
  /// Sets [`Self::decoding_word_timestamps`] in place.
  #[inline(always)]
  pub const fn set_decoding_word_timestamps(&mut self, decoding_word_timestamps: f64) -> &mut Self {
    self.decoding_word_timestamps = decoding_word_timestamps;
    self
  }

  // -- decoding_non_prediction ---------------------------------------------
  /// Time spent in the decode loop outside of model inference.
  #[inline(always)]
  pub const fn decoding_non_prediction(&self) -> f64 {
    self.decoding_non_prediction
  }
  /// Sets [`Self::decoding_non_prediction`] in place.
  #[inline(always)]
  pub const fn set_decoding_non_prediction(&mut self, decoding_non_prediction: f64) -> &mut Self {
    self.decoding_non_prediction = decoding_non_prediction;
    self
  }

  // -- total_audio_processing_runs -----------------------------------------
  /// Number of audio-processing passes run.
  #[inline(always)]
  pub const fn total_audio_processing_runs(&self) -> f64 {
    self.total_audio_processing_runs
  }
  /// Sets [`Self::total_audio_processing_runs`] in place.
  #[inline(always)]
  pub const fn set_total_audio_processing_runs(
    &mut self,
    total_audio_processing_runs: f64,
  ) -> &mut Self {
    self.total_audio_processing_runs = total_audio_processing_runs;
    self
  }

  // -- total_logmel_runs ---------------------------------------------------
  /// Number of mel-spectrogram passes run.
  #[inline(always)]
  pub const fn total_logmel_runs(&self) -> f64 {
    self.total_logmel_runs
  }
  /// Sets [`Self::total_logmel_runs`] in place.
  #[inline(always)]
  pub const fn set_total_logmel_runs(&mut self, total_logmel_runs: f64) -> &mut Self {
    self.total_logmel_runs = total_logmel_runs;
    self
  }

  // -- total_encoding_runs -------------------------------------------------
  /// Number of encoder passes run.
  #[inline(always)]
  pub const fn total_encoding_runs(&self) -> f64 {
    self.total_encoding_runs
  }
  /// Sets [`Self::total_encoding_runs`] in place.
  #[inline(always)]
  pub const fn set_total_encoding_runs(&mut self, total_encoding_runs: f64) -> &mut Self {
    self.total_encoding_runs = total_encoding_runs;
    self
  }

  // -- total_decoding_loops ------------------------------------------------
  /// Number of decode-loop steps (tokens sampled) across all windows.
  #[inline(always)]
  pub const fn total_decoding_loops(&self) -> f64 {
    self.total_decoding_loops
  }
  /// Sets [`Self::total_decoding_loops`] in place.
  #[inline(always)]
  pub const fn set_total_decoding_loops(&mut self, total_decoding_loops: f64) -> &mut Self {
    self.total_decoding_loops = total_decoding_loops;
    self
  }

  // -- total_kv_update_runs ------------------------------------------------
  /// Number of KV-cache updates run.
  #[inline(always)]
  pub const fn total_kv_update_runs(&self) -> f64 {
    self.total_kv_update_runs
  }
  /// Sets [`Self::total_kv_update_runs`] in place.
  #[inline(always)]
  pub const fn set_total_kv_update_runs(&mut self, total_kv_update_runs: f64) -> &mut Self {
    self.total_kv_update_runs = total_kv_update_runs;
    self
  }

  // -- total_timestamp_alignment_runs --------------------------------------
  /// Number of word-timestamp alignment passes run.
  #[inline(always)]
  pub const fn total_timestamp_alignment_runs(&self) -> f64 {
    self.total_timestamp_alignment_runs
  }
  /// Sets [`Self::total_timestamp_alignment_runs`] in place.
  #[inline(always)]
  pub const fn set_total_timestamp_alignment_runs(
    &mut self,
    total_timestamp_alignment_runs: f64,
  ) -> &mut Self {
    self.total_timestamp_alignment_runs = total_timestamp_alignment_runs;
    self
  }

  // -- total_decoding_fallbacks --------------------------------------------
  /// Zero-based attempt index of the most recent temperature fallback — a
  /// faithful port of Swift's `totalDecodingFallbacks = Double(i)`
  /// (`TranscribeTask.swift`), *assigned* (not accumulated) on each
  /// fallback, so despite the name it is NOT a fallback count. The first
  /// fallback writes `0.0`, which is also this field's initial value, so
  /// `0.0` cannot distinguish "never fell back" from "one fallback at
  /// attempt 0"; a true count would be `attempt + 1`. A multi-window task
  /// overwrites per window, keeping only the last window's index, and
  /// merged results sum these per-result values. For an unambiguous "a
  /// fallback occurred" signal, use
  /// [`TaskFacts::drew_from_rng`](crate::audio::whisper::task_facts::TaskFacts::drew_from_rng).
  #[inline(always)]
  pub const fn total_decoding_fallbacks(&self) -> f64 {
    self.total_decoding_fallbacks
  }
  /// Sets [`Self::total_decoding_fallbacks`] in place.
  #[inline(always)]
  pub const fn set_total_decoding_fallbacks(&mut self, total_decoding_fallbacks: f64) -> &mut Self {
    self.total_decoding_fallbacks = total_decoding_fallbacks;
    self
  }

  // -- total_decoding_windows ----------------------------------------------
  /// Number of 30 s windows decoded.
  #[inline(always)]
  pub const fn total_decoding_windows(&self) -> f64 {
    self.total_decoding_windows
  }
  /// Sets [`Self::total_decoding_windows`] in place.
  #[inline(always)]
  pub const fn set_total_decoding_windows(&mut self, total_decoding_windows: f64) -> &mut Self {
    self.total_decoding_windows = total_decoding_windows;
    self
  }

  // -- full_pipeline -------------------------------------------------------
  /// Total end-to-end pipeline duration.
  #[inline(always)]
  pub const fn full_pipeline(&self) -> f64 {
    self.full_pipeline
  }
  /// Sets [`Self::full_pipeline`] in place.
  #[inline(always)]
  pub const fn set_full_pipeline(&mut self, full_pipeline: f64) -> &mut Self {
    self.full_pipeline = full_pipeline;
    self
  }

  /// Sampled tokens per second (Swift `tokensPerSecond`,
  /// `Models.swift:766-768`): `total_decoding_loops / full_pipeline`,
  /// guarded to `0.0` when `full_pipeline` is `0.0` (Swift's unguarded
  /// division would yield `NaN`/`inf`).
  #[inline(always)]
  pub const fn tokens_per_second(&self) -> f64 {
    if self.full_pipeline == 0.0 {
      0.0
    } else {
      self.total_decoding_loops / self.full_pipeline
    }
  }

  /// Wall-clock seconds per second of input audio (Swift
  /// `realTimeFactor`, `Models.swift:770-772`): `full_pipeline /
  /// input_audio_seconds`, guarded to `0.0` when `input_audio_seconds` is
  /// `0.0`.
  #[inline(always)]
  pub const fn real_time_factor(&self) -> f64 {
    if self.input_audio_seconds == 0.0 {
      0.0
    } else {
      self.full_pipeline / self.input_audio_seconds
    }
  }

  /// Inverse of [`Self::real_time_factor`] (Swift `speedFactor`,
  /// `Models.swift:774-776`): `input_audio_seconds / full_pipeline`,
  /// guarded to `0.0` when `full_pipeline` is `0.0`.
  #[inline(always)]
  pub const fn speed_factor(&self) -> f64 {
    if self.full_pipeline == 0.0 {
      0.0
    } else {
      self.input_audio_seconds / self.full_pipeline
    }
  }
}

// ---------------------------------------------------------------------
// TranscriptionResult
// ---------------------------------------------------------------------

/// Final transcription output (Swift `TranscriptionResultStruct`,
/// `Models.swift:543-563` — the value-type sibling of Swift's lock-guarded
/// `TranscriptionResult` reference type; see this module's doc comment
/// for why this port uses it instead).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptionResult {
  /// Full transcribed text (all segments concatenated).
  #[cfg_attr(feature = "serde", serde(default))]
  text: String,
  /// Transcribed segments, in order.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  segments: Vec<TranscriptionSegment>,
  /// Detected or configured spoken language (ISO code). Empty means
  /// undetermined (golden empty-means-absent).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "String::is_empty")
  )]
  language: String,
  /// Aggregated pipeline timings.
  #[cfg_attr(feature = "serde", serde(default))]
  timings: TranscriptionTimings,
  /// Seek position, in seconds, transcription stopped at. `None` when
  /// transcription ran to the end of the input.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  seek_time: Option<f32>,
  /// The decode-time facts this transcription **run controlled** — whether it
  /// drew from the token sampler, the language it genuinely observed, whether a
  /// progress callback truncated it, whether it swallowed a child error, the
  /// worker coordinates its RNG streams rode, and the segment-id span its decode
  /// allocated — as one carried record (coremlit issue #14, codex round 6).
  /// Replaces the six separate fields these facts used to scatter across, whose
  /// per-fact plumbing lost or fabricated three of them at aggregation
  /// boundaries; see
  /// [`TaskFacts`](crate::audio::whisper::task_facts::TaskFacts) for the history and the one
  /// merge law [`merge_transcription_results_with_options`] now folds them by.
  ///
  /// [`Provenance::for_result`](crate::audio::whisper::provenance::Provenance::for_result) reads
  /// this record whole; [`Self::new`] starts it
  /// [`TaskFacts::unknown`](crate::audio::whisper::task_facts::TaskFacts::unknown) (a hand-built
  /// result drew nothing, witnessed no language, was not truncated, and rode an
  /// **unknown** worker coordinate — never a fabricated `0`), and the pipeline
  /// sets the observed value via [`Self::with_task_facts`].
  ///
  /// Required on deserialize: the reproducibility facts inside it must never
  /// silently default to their optimistic values (see that type's serde
  /// contract).
  task_facts: TaskFacts,
}

impl TranscriptionResult {
  /// Builds a result from its four required fields (Swift
  /// `TranscriptionResultStruct.init`, `Models.swift:550-562`, has no
  /// defaults for these either); [`Self::seek_time`] starts `None` and
  /// [`Self::task_facts`] starts [`TaskFacts::unknown`].
  ///
  /// A result assembled by hand therefore records an **unknown** sampling draw,
  /// **no** observed language, an **unknown** callback truncation, an **unknown**
  /// swallowed child error, an **unknown** worker coordinate (never a fabricated
  /// `0`, R6-F2), and an untracked id span — explicit unknown throughout, never
  /// the optimistic `Some(false)` a "nothing happened" default would forge (F1).
  /// That is the right default for a caller inventing a transcript, and it is not
  /// a hole in the reproducibility guarantee:
  /// [`Provenance::for_result`](crate::audio::whisper::provenance::Provenance::for_result) TRUSTS
  /// this carried record whole rather than scanning the surviving segments' own
  /// temperatures (inferring the draw from the survivors was the bug it
  /// replaced), so an unknown draw is treated CONSERVATIVELY as non-reproducible,
  /// never optimistically waved through. What only the carried
  /// [`TaskFacts::drew_from_rng`](crate::audio::whisper::task_facts::TaskFacts::drew_from_rng)
  /// flag can carry is a sampled window whose segments are *gone* — and only
  /// the decode path can know about those, setting the observed facts via
  /// [`Self::with_task_facts`].
  pub fn new(
    text: impl Into<String>,
    segments: impl Into<Vec<TranscriptionSegment>>,
    language: impl Into<String>,
    timings: TranscriptionTimings,
  ) -> Self {
    Self {
      text: text.into(),
      segments: segments.into(),
      language: language.into(),
      timings,
      seek_time: None,
      // A hand-built result controlled no decode, so it cannot OBSERVE whether a
      // draw, a callback truncation, or a swallowed child error occurred: it
      // records explicit UNKNOWN for each (never the optimistic Some(false)),
      // witnessed no language, rode an UNKNOWN worker coordinate (never a
      // fabricated 0), and tracked no id span. The pipeline sets the observed
      // record via `with_task_facts`.
      task_facts: TaskFacts::unknown(),
    }
  }

  // -- text -------------------------------------------------------------
  /// Full transcribed text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    self.text.as_str()
  }
  /// Builder form of [`Self::set_text`].
  #[must_use]
  #[inline(always)]
  pub fn with_text(mut self, text: impl Into<String>) -> Self {
    self.set_text(text);
    self
  }
  /// Sets [`Self::text`] in place.
  #[inline(always)]
  pub fn set_text(&mut self, text: impl Into<String>) -> &mut Self {
    self.text = text.into();
    self
  }

  // -- segments (Vec<TranscriptionSegment>) --------------------------------
  /// Transcribed segments, in order.
  #[inline(always)]
  pub const fn segments_slice(&self) -> &[TranscriptionSegment] {
    self.segments.as_slice()
  }

  /// Mutable view of the segments (fixed length; use
  /// [`Self::set_segments`] to replace the collection). Segment timings
  /// carry no cross-field invariant with the rest of the result, so
  /// in-place mutation is safe to expose — chunk re-anchoring shifts them
  /// directly.
  #[inline(always)]
  pub const fn segments_slice_mut(&mut self) -> &mut [TranscriptionSegment] {
    self.segments.as_mut_slice()
  }
  /// Builder form of [`Self::set_segments`].
  #[must_use]
  #[inline(always)]
  pub fn with_segments(mut self, segments: impl Into<Vec<TranscriptionSegment>>) -> Self {
    self.set_segments(segments);
    self
  }
  /// Sets [`Self::segments_slice`] in place.
  #[inline(always)]
  pub fn set_segments(&mut self, segments: impl Into<Vec<TranscriptionSegment>>) -> &mut Self {
    self.segments = segments.into();
    self
  }

  // -- language -----------------------------------------------------------
  /// Detected or configured spoken language (ISO code); empty means
  /// undetermined.
  #[inline(always)]
  pub fn language(&self) -> &str {
    self.language.as_str()
  }
  /// Builder form of [`Self::set_language`].
  #[must_use]
  #[inline(always)]
  pub fn with_language(mut self, language: impl Into<String>) -> Self {
    self.set_language(language);
    self
  }
  /// Sets [`Self::language`] in place.
  #[inline(always)]
  pub fn set_language(&mut self, language: impl Into<String>) -> &mut Self {
    self.language = language.into();
    self
  }

  // -- task_facts ----------------------------------------------------------
  /// The decode-time facts this transcription run **controlled** — the RNG
  /// draw, the observed language, the early-stop truncation, the worker
  /// coordinates, and the allocated id span — as one record.
  /// [`Provenance::for_result`](crate::audio::whisper::provenance::Provenance::for_result)
  /// reads it whole. See the field's doc.
  #[inline(always)]
  pub const fn task_facts(&self) -> &TaskFacts {
    &self.task_facts
  }
  /// Mutable access to [`Self::task_facts`] — the pipeline uses it to
  /// [`merge`](crate::audio::whisper::task_facts::TaskFacts::merge) a shared sink's recovered
  /// facts (a dropped-because-errored VAD chunk's draw/observation/early stop)
  /// into a merged transcript in place.
  #[inline(always)]
  pub const fn task_facts_mut(&mut self) -> &mut TaskFacts {
    &mut self.task_facts
  }
  /// Builder form of [`Self::set_task_facts`].
  #[must_use]
  #[inline(always)]
  pub fn with_task_facts(mut self, task_facts: TaskFacts) -> Self {
    self.set_task_facts(task_facts);
    self
  }
  /// Assigns [`Self::task_facts`] directly — the pipeline passes the record it
  /// accumulated across the decode.
  #[inline(always)]
  pub fn set_task_facts(&mut self, task_facts: TaskFacts) -> &mut Self {
    self.task_facts = task_facts;
    self
  }

  // -- timings --------------------------------------------------------------
  /// Aggregated pipeline timings.
  #[inline(always)]
  pub const fn timings(&self) -> &TranscriptionTimings {
    &self.timings
  }
  /// Builder form of [`Self::set_timings`].
  #[must_use]
  #[inline(always)]
  pub fn with_timings(mut self, timings: TranscriptionTimings) -> Self {
    self.set_timings(timings);
    self
  }
  /// Sets [`Self::timings`] in place.
  #[inline(always)]
  pub fn set_timings(&mut self, timings: TranscriptionTimings) -> &mut Self {
    self.timings = timings;
    self
  }

  // -- seek_time (Option<f32>) ----------------------------------------------
  /// Seek position, in seconds, transcription stopped at. `None` when
  /// transcription ran to the end of the input.
  #[inline(always)]
  pub const fn seek_time(&self) -> Option<f32> {
    self.seek_time
  }
  /// Builder form of [`Self::set_seek_time`].
  #[must_use]
  #[inline(always)]
  pub const fn with_seek_time(mut self, seek_time: f32) -> Self {
    self.set_seek_time(seek_time);
    self
  }
  /// Sets [`Self::seek_time`] to `Some(seek_time)`.
  #[inline(always)]
  pub const fn set_seek_time(&mut self, seek_time: f32) -> &mut Self {
    self.seek_time = Some(seek_time);
    self
  }
  /// Builder form of [`Self::update_seek_time`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_seek_time(mut self, seek_time: Option<f32>) -> Self {
    self.update_seek_time(seek_time);
    self
  }
  /// Assigns [`Self::seek_time`] directly.
  #[inline(always)]
  pub const fn update_seek_time(&mut self, seek_time: Option<f32>) -> &mut Self {
    self.seek_time = seek_time;
    self
  }
  /// Sets [`Self::seek_time`] to `None`.
  #[inline(always)]
  pub const fn clear_seek_time(&mut self) -> &mut Self {
    self.seek_time = None;
    self
  }

  // -- all_words ----------------------------------------------------------
  /// Every word timing across every segment, flattened in segment order —
  /// ports the `TranscriptionResult.allWords` extension (`Models.swift:
  /// 566-570`, `segments.compactMap { $0.words }.flatMap { $0 }`). This
  /// port's [`TranscriptionSegment::words_slice`] is never optional
  /// (empty-means-absent, this module's own doc comment), so a segment
  /// with no words simply contributes zero elements here — no
  /// `compactMap`-equivalent filter is needed to reproduce Swift's
  /// nil-dropping.
  pub fn all_words(&self) -> Vec<WordTiming> {
    self
      .segments
      .iter()
      .flat_map(TranscriptionSegment::words_slice)
      .cloned()
      .collect()
  }
}

// ---------------------------------------------------------------------
// DecodingResult
// ---------------------------------------------------------------------

/// Per-window decode output (Swift `DecodingResult`,
/// `Models.swift:383-439`). Three Swift fields have no place here:
/// `cache` (KV-cache tensors are a backend-layer concern, not a result
/// value type), `timings` (window timings roll up into
/// [`TranscriptionResult::timings`] instead of nesting per window), and
/// `fallback` (the fallback decision is a pure function of this type,
/// [`needs_fallback`], rather than a stored, mutually-recursive field).
///
/// One field goes the other way, beyond Swift's own set:
/// [`Self::first_token_log_prob`]. Swift computes the first sampled
/// token's log probability only transiently, inside the decode loop
/// (`TextDecoder.swift:662-667`), to build a local `isFirstTokenLogProbTooLow`
/// bool that never leaves the function. This port's decode loop
/// ([`crate::audio::whisper::decode::decode_text`]) has no such back door — its only
/// output is this struct — so the raw value is stored here instead,
/// letting a later fallback-ladder caller recompute the threshold
/// comparison itself and pass it to [`needs_fallback`] (that function's
/// own doc comment's "assumption (b)").
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodingResult {
  /// Detected or configured spoken language (ISO code). Empty means
  /// undetermined (golden empty-means-absent).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "String::is_empty")
  )]
  language: String,
  /// Per-language detection probabilities. Empty means language
  /// detection did not run.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  language_probs: Vec<(String, f32)>,
  /// The language the model actually PREDICTED (an ISO code), or `None` when it
  /// predicted none — a genuine in-loop language detection. Distinct from the
  /// Swift-faithful display [`Self::language`], which reports the FIRST language
  /// token in the WHOLE sequence (the forced prefill `<|en|>` included): this is
  /// the first `<|lang|>` token the model emitted at or after the forced prompt,
  /// so the two disagree exactly when a forced `<|en|>` prefill is followed by a
  /// differently-predicted language.
  ///
  /// `None` for a configured language (an input, not a detection), for the
  /// [`DEFAULT_LANGUAGE_CODE`] display fallback, and when the predicted region
  /// holds no language token at all (a zero-iteration decode forces `<|en|>`
  /// into the prompt but predicts nothing). The pipeline promotes THIS predicted
  /// code into the result's
  /// [`TaskFacts::observed_language`](crate::audio::whisper::task_facts::TaskFacts::observed_language);
  /// recording the display
  /// [`Self::language`] there instead would misreport a forced `<|en|>` as the
  /// detection when a different language was predicted after it (coremlit issue
  /// #14, codex round 5) — the reason this carries the predicted STRING and not
  /// a mere "observed" boolean. Not a Swift field — Swift's `DecodingResult` has
  /// no detection-provenance concept, the same Rust-only extension rationale as
  /// the sibling [`Self::first_token_log_prob`] (see this struct's doc).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  observed_language: Option<String>,
  /// Sampled token ids for this window.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  tokens: Vec<u32>,
  /// Per-step `(token id, log probability)`, in decode order; the first
  /// entry is the first sampled token ([`needs_fallback`] reads it).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  token_log_probs: Vec<(u32, f32)>,
  /// Decoded text for this window.
  #[cfg_attr(feature = "serde", serde(default))]
  text: String,
  /// Average sampled-token log probability.
  #[cfg_attr(feature = "serde", serde(default))]
  avg_logprob: f32,
  /// Probability this window contains no speech.
  #[cfg_attr(feature = "serde", serde(default))]
  no_speech_prob: f32,
  /// Sampling temperature this window was decoded at.
  #[cfg_attr(feature = "serde", serde(default))]
  temperature: f32,
  /// Compression ratio of `text` (repetition signal).
  #[cfg_attr(feature = "serde", serde(default))]
  compression_ratio: f32,
  /// The first sampled token's raw log probability (not a Swift field —
  /// see this struct's doc comment).
  #[cfg_attr(feature = "serde", serde(default))]
  first_token_log_prob: f32,
  /// Whether a progress callback requested an early stop that TRUNCATED this
  /// window's decode (`Some(false)` past the prefill steps) — a caller CONTROL
  /// action, distinct from every ordinary termination (EOT, the token-context
  /// cap, a too-low first-token log-prob). The pipeline carries this out to the
  /// result's [`TaskFacts::early_stopped`](crate::audio::whisper::task_facts::TaskFacts::early_stopped),
  /// which
  /// [`Provenance::is_reproducible`](crate::audio::whisper::provenance::Provenance::is_reproducible)
  /// reads: a transcript an unrecorded callback truncated cannot be reproduced
  /// from the recorded options and seed alone (coremlit issue #14, codex round
  /// 5). Not a Swift field — the same Rust-only extension rationale as the
  /// sibling [`Self::first_token_log_prob`].
  #[cfg_attr(feature = "serde", serde(default))]
  early_stopped: bool,
}

impl Default for DecodingResult {
  fn default() -> Self {
    Self::new()
  }
}

impl DecodingResult {
  /// A zero-value result (Swift `DecodingResult.emptyResults`,
  /// `Models.swift:397-410`).
  pub const fn new() -> Self {
    Self {
      language: String::new(),
      language_probs: Vec::new(),
      observed_language: None,
      tokens: Vec::new(),
      token_log_probs: Vec::new(),
      text: String::new(),
      avg_logprob: 0.0,
      no_speech_prob: 0.0,
      temperature: 0.0,
      compression_ratio: 0.0,
      first_token_log_prob: 0.0,
      early_stopped: false,
    }
  }

  // -- language ---------------------------------------------------------
  /// Detected or configured spoken language (ISO code); empty means
  /// undetermined.
  #[inline(always)]
  pub fn language(&self) -> &str {
    self.language.as_str()
  }
  /// Builder form of [`Self::set_language`].
  #[must_use]
  #[inline(always)]
  pub fn with_language(mut self, language: impl Into<String>) -> Self {
    self.set_language(language);
    self
  }
  /// Sets [`Self::language`] in place.
  #[inline(always)]
  pub fn set_language(&mut self, language: impl Into<String>) -> &mut Self {
    self.language = language.into();
    self
  }

  // -- language_probs (Vec<(String,f32)>) ----------------------------------
  /// Per-language detection probabilities. Empty means language
  /// detection did not run.
  #[inline(always)]
  pub const fn language_probs_slice(&self) -> &[(String, f32)] {
    self.language_probs.as_slice()
  }
  /// Builder form of [`Self::set_language_probs`].
  #[must_use]
  #[inline(always)]
  pub fn with_language_probs(mut self, language_probs: impl Into<Vec<(String, f32)>>) -> Self {
    self.set_language_probs(language_probs);
    self
  }
  /// Sets [`Self::language_probs_slice`] in place.
  #[inline(always)]
  pub fn set_language_probs(&mut self, language_probs: impl Into<Vec<(String, f32)>>) -> &mut Self {
    self.language_probs = language_probs.into();
    self
  }

  // -- observed_language (Option<String>) ----------------------------------
  /// The language the model actually PREDICTED (an ISO code), or `None`. A
  /// genuine in-loop detection, SEPARATE from the Swift-faithful display
  /// [`Self::language`]; see the field doc for how the pipeline promotes this
  /// into the result's
  /// [`TaskFacts::observed_language`](crate::audio::whisper::task_facts::TaskFacts::observed_language),
  /// and why it is not the display language.
  #[inline(always)]
  pub fn observed_language(&self) -> Option<&str> {
    self.observed_language.as_deref()
  }
  /// Builder form of [`Self::update_observed_language`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_observed_language(mut self, observed_language: Option<String>) -> Self {
    self.update_observed_language(observed_language);
    self
  }
  /// Assigns [`Self::observed_language`] directly — the finalized decode passes
  /// the predicted language code, or `None` when nothing was predicted.
  #[inline(always)]
  pub fn update_observed_language(&mut self, observed_language: Option<String>) -> &mut Self {
    self.observed_language = observed_language;
    self
  }

  // -- tokens (Vec<u32>) ---------------------------------------------------
  /// Sampled token ids for this window.
  #[inline(always)]
  pub const fn tokens_slice(&self) -> &[u32] {
    self.tokens.as_slice()
  }
  /// Builder form of [`Self::set_tokens`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokens(mut self, tokens: impl Into<Vec<u32>>) -> Self {
    self.set_tokens(tokens);
    self
  }
  /// Sets [`Self::tokens_slice`] in place.
  #[inline(always)]
  pub fn set_tokens(&mut self, tokens: impl Into<Vec<u32>>) -> &mut Self {
    self.tokens = tokens.into();
    self
  }

  // -- token_log_probs (Vec<(u32,f32)>) -------------------------------------
  /// Per-step `(token id, log probability)`; the first entry is the
  /// first sampled token.
  #[inline(always)]
  pub const fn token_log_probs_slice(&self) -> &[(u32, f32)] {
    self.token_log_probs.as_slice()
  }
  /// Builder form of [`Self::set_token_log_probs`].
  #[must_use]
  #[inline(always)]
  pub fn with_token_log_probs(mut self, token_log_probs: impl Into<Vec<(u32, f32)>>) -> Self {
    self.set_token_log_probs(token_log_probs);
    self
  }
  /// Sets [`Self::token_log_probs_slice`] in place.
  #[inline(always)]
  pub fn set_token_log_probs(&mut self, token_log_probs: impl Into<Vec<(u32, f32)>>) -> &mut Self {
    self.token_log_probs = token_log_probs.into();
    self
  }

  // -- text -----------------------------------------------------------------
  /// Decoded text for this window.
  #[inline(always)]
  pub fn text(&self) -> &str {
    self.text.as_str()
  }
  /// Builder form of [`Self::set_text`].
  #[must_use]
  #[inline(always)]
  pub fn with_text(mut self, text: impl Into<String>) -> Self {
    self.set_text(text);
    self
  }
  /// Sets [`Self::text`] in place.
  #[inline(always)]
  pub fn set_text(&mut self, text: impl Into<String>) -> &mut Self {
    self.text = text.into();
    self
  }

  // -- avg_logprob ------------------------------------------------------------
  /// Average sampled-token log probability.
  #[inline(always)]
  pub const fn avg_logprob(&self) -> f32 {
    self.avg_logprob
  }
  /// Builder form of [`Self::set_avg_logprob`].
  #[must_use]
  #[inline(always)]
  pub const fn with_avg_logprob(mut self, avg_logprob: f32) -> Self {
    self.set_avg_logprob(avg_logprob);
    self
  }
  /// Sets [`Self::avg_logprob`] in place.
  #[inline(always)]
  pub const fn set_avg_logprob(&mut self, avg_logprob: f32) -> &mut Self {
    self.avg_logprob = avg_logprob;
    self
  }

  // -- no_speech_prob -----------------------------------------------------------
  /// Probability this window contains no speech.
  #[inline(always)]
  pub const fn no_speech_prob(&self) -> f32 {
    self.no_speech_prob
  }
  /// Builder form of [`Self::set_no_speech_prob`].
  #[must_use]
  #[inline(always)]
  pub const fn with_no_speech_prob(mut self, no_speech_prob: f32) -> Self {
    self.set_no_speech_prob(no_speech_prob);
    self
  }
  /// Sets [`Self::no_speech_prob`] in place.
  #[inline(always)]
  pub const fn set_no_speech_prob(&mut self, no_speech_prob: f32) -> &mut Self {
    self.no_speech_prob = no_speech_prob;
    self
  }

  // -- temperature ----------------------------------------------------------
  /// Sampling temperature this window was decoded at.
  #[inline(always)]
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }
  /// Builder form of [`Self::set_temperature`].
  #[must_use]
  #[inline(always)]
  pub const fn with_temperature(mut self, temperature: f32) -> Self {
    self.set_temperature(temperature);
    self
  }
  /// Sets [`Self::temperature`] in place.
  #[inline(always)]
  pub const fn set_temperature(&mut self, temperature: f32) -> &mut Self {
    self.temperature = temperature;
    self
  }

  // -- compression_ratio ------------------------------------------------------
  /// Compression ratio of [`Self::text`] (repetition signal).
  #[inline(always)]
  pub const fn compression_ratio(&self) -> f32 {
    self.compression_ratio
  }
  /// Builder form of [`Self::set_compression_ratio`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compression_ratio(mut self, compression_ratio: f32) -> Self {
    self.set_compression_ratio(compression_ratio);
    self
  }
  /// Sets [`Self::compression_ratio`] in place.
  #[inline(always)]
  pub const fn set_compression_ratio(&mut self, compression_ratio: f32) -> &mut Self {
    self.compression_ratio = compression_ratio;
    self
  }

  // -- first_token_log_prob ------------------------------------------------
  /// The first sampled token's raw log probability (not a Swift field —
  /// see this struct's doc comment).
  #[inline(always)]
  pub const fn first_token_log_prob(&self) -> f32 {
    self.first_token_log_prob
  }
  /// Builder form of [`Self::set_first_token_log_prob`].
  #[must_use]
  #[inline(always)]
  pub const fn with_first_token_log_prob(mut self, first_token_log_prob: f32) -> Self {
    self.set_first_token_log_prob(first_token_log_prob);
    self
  }
  /// Sets [`Self::first_token_log_prob`] in place.
  #[inline(always)]
  pub const fn set_first_token_log_prob(&mut self, first_token_log_prob: f32) -> &mut Self {
    self.first_token_log_prob = first_token_log_prob;
    self
  }

  // -- early_stopped (bool) ------------------------------------------------
  /// Whether a progress callback truncated this window's decode with an early
  /// stop. See the field doc; the pipeline carries this to the result's
  /// [`TaskFacts::early_stopped`](crate::audio::whisper::task_facts::TaskFacts::early_stopped).
  #[inline(always)]
  pub const fn early_stopped(&self) -> bool {
    self.early_stopped
  }
  /// Builder form of [`Self::update_early_stopped`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_early_stopped(mut self, early_stopped: bool) -> Self {
    self.update_early_stopped(early_stopped);
    self
  }
  /// Assigns [`Self::early_stopped`] directly — the decode loop passes whether
  /// its early-stop latch fired.
  #[inline(always)]
  pub const fn update_early_stopped(&mut self, early_stopped: bool) -> &mut Self {
    self.early_stopped = early_stopped;
    self
  }
}

// ---------------------------------------------------------------------
// TranscriptionProgress
// ---------------------------------------------------------------------

/// Live per-step decode progress (Swift `TranscriptionProgress`,
/// `Models.swift:643-661`), delivered to a
/// [`TranscriptionProgressCallback`](crate::audio::whisper::decode::TranscriptionProgressCallback)
/// after every non-completed decode step, prefill steps included.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscriptionProgress {
  /// Timings accumulated so far this run.
  #[cfg_attr(feature = "serde", serde(default))]
  timings: TranscriptionTimings,
  /// Decoded text so far.
  #[cfg_attr(feature = "serde", serde(default))]
  text: String,
  /// Sampled token ids so far (prompt included).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  tokens: Vec<u32>,
  /// Sampling temperature, once known.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  temperature: Option<f32>,
  /// Average sampled-token log probability so far, once known.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  avg_logprob: Option<f32>,
  /// Compression ratio of [`Self::text`] so far, once known.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  compression_ratio: Option<f32>,
  /// Which decode window this progress update belongs to.
  #[cfg_attr(feature = "serde", serde(default))]
  window_id: usize,
}

impl TranscriptionProgress {
  /// Builds a progress update from its three always-known fields (Swift
  /// `TranscriptionProgress.init`, `Models.swift:652-660`, defaults every
  /// other parameter to `nil`/`0`); the optional trio starts `None` and
  /// [`Self::window_id`] starts `0`.
  pub fn new(
    timings: TranscriptionTimings,
    text: impl Into<String>,
    tokens: impl Into<Vec<u32>>,
  ) -> Self {
    Self {
      timings,
      text: text.into(),
      tokens: tokens.into(),
      temperature: None,
      avg_logprob: None,
      compression_ratio: None,
      window_id: 0,
    }
  }

  // -- timings --------------------------------------------------------------
  /// Timings accumulated so far this run.
  #[inline(always)]
  pub const fn timings(&self) -> &TranscriptionTimings {
    &self.timings
  }
  /// Builder form of [`Self::set_timings`].
  #[must_use]
  #[inline(always)]
  pub fn with_timings(mut self, timings: TranscriptionTimings) -> Self {
    self.set_timings(timings);
    self
  }
  /// Sets [`Self::timings`] in place.
  #[inline(always)]
  pub fn set_timings(&mut self, timings: TranscriptionTimings) -> &mut Self {
    self.timings = timings;
    self
  }

  // -- text -------------------------------------------------------------
  /// Decoded text so far.
  #[inline(always)]
  pub fn text(&self) -> &str {
    self.text.as_str()
  }
  /// Builder form of [`Self::set_text`].
  #[must_use]
  #[inline(always)]
  pub fn with_text(mut self, text: impl Into<String>) -> Self {
    self.set_text(text);
    self
  }
  /// Sets [`Self::text`] in place.
  #[inline(always)]
  pub fn set_text(&mut self, text: impl Into<String>) -> &mut Self {
    self.text = text.into();
    self
  }

  // -- tokens (Vec<u32>) -------------------------------------------------
  /// Sampled token ids so far (prompt included).
  #[inline(always)]
  pub const fn tokens_slice(&self) -> &[u32] {
    self.tokens.as_slice()
  }
  /// Builder form of [`Self::set_tokens`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokens(mut self, tokens: impl Into<Vec<u32>>) -> Self {
    self.set_tokens(tokens);
    self
  }
  /// Sets [`Self::tokens_slice`] in place.
  #[inline(always)]
  pub fn set_tokens(&mut self, tokens: impl Into<Vec<u32>>) -> &mut Self {
    self.tokens = tokens.into();
    self
  }

  // -- temperature (Option<f32>) -------------------------------------------
  /// Sampling temperature, once known.
  #[inline(always)]
  pub const fn temperature(&self) -> Option<f32> {
    self.temperature
  }
  /// Builder form of [`Self::set_temperature`].
  #[must_use]
  #[inline(always)]
  pub const fn with_temperature(mut self, temperature: f32) -> Self {
    self.set_temperature(temperature);
    self
  }
  /// Sets [`Self::temperature`] to `Some(temperature)`.
  #[inline(always)]
  pub const fn set_temperature(&mut self, temperature: f32) -> &mut Self {
    self.temperature = Some(temperature);
    self
  }
  /// Builder form of [`Self::update_temperature`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_temperature(mut self, temperature: Option<f32>) -> Self {
    self.update_temperature(temperature);
    self
  }
  /// Assigns [`Self::temperature`] directly.
  #[inline(always)]
  pub const fn update_temperature(&mut self, temperature: Option<f32>) -> &mut Self {
    self.temperature = temperature;
    self
  }
  /// Sets [`Self::temperature`] to `None`.
  #[inline(always)]
  pub const fn clear_temperature(&mut self) -> &mut Self {
    self.temperature = None;
    self
  }

  // -- avg_logprob (Option<f32>) -------------------------------------------
  /// Average sampled-token log probability so far, once known.
  #[inline(always)]
  pub const fn avg_logprob(&self) -> Option<f32> {
    self.avg_logprob
  }
  /// Builder form of [`Self::set_avg_logprob`].
  #[must_use]
  #[inline(always)]
  pub const fn with_avg_logprob(mut self, avg_logprob: f32) -> Self {
    self.set_avg_logprob(avg_logprob);
    self
  }
  /// Sets [`Self::avg_logprob`] to `Some(avg_logprob)`.
  #[inline(always)]
  pub const fn set_avg_logprob(&mut self, avg_logprob: f32) -> &mut Self {
    self.avg_logprob = Some(avg_logprob);
    self
  }
  /// Builder form of [`Self::update_avg_logprob`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_avg_logprob(mut self, avg_logprob: Option<f32>) -> Self {
    self.update_avg_logprob(avg_logprob);
    self
  }
  /// Assigns [`Self::avg_logprob`] directly.
  #[inline(always)]
  pub const fn update_avg_logprob(&mut self, avg_logprob: Option<f32>) -> &mut Self {
    self.avg_logprob = avg_logprob;
    self
  }
  /// Sets [`Self::avg_logprob`] to `None`.
  #[inline(always)]
  pub const fn clear_avg_logprob(&mut self) -> &mut Self {
    self.avg_logprob = None;
    self
  }

  // -- compression_ratio (Option<f32>) -------------------------------------
  /// Compression ratio of [`Self::text`] so far, once known.
  #[inline(always)]
  pub const fn compression_ratio(&self) -> Option<f32> {
    self.compression_ratio
  }
  /// Builder form of [`Self::set_compression_ratio`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compression_ratio(mut self, compression_ratio: f32) -> Self {
    self.set_compression_ratio(compression_ratio);
    self
  }
  /// Sets [`Self::compression_ratio`] to `Some(compression_ratio)`.
  #[inline(always)]
  pub const fn set_compression_ratio(&mut self, compression_ratio: f32) -> &mut Self {
    self.compression_ratio = Some(compression_ratio);
    self
  }
  /// Builder form of [`Self::update_compression_ratio`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_compression_ratio(mut self, compression_ratio: Option<f32>) -> Self {
    self.update_compression_ratio(compression_ratio);
    self
  }
  /// Assigns [`Self::compression_ratio`] directly.
  #[inline(always)]
  pub const fn update_compression_ratio(&mut self, compression_ratio: Option<f32>) -> &mut Self {
    self.compression_ratio = compression_ratio;
    self
  }
  /// Sets [`Self::compression_ratio`] to `None`.
  #[inline(always)]
  pub const fn clear_compression_ratio(&mut self) -> &mut Self {
    self.compression_ratio = None;
    self
  }

  // -- window_id ------------------------------------------------------------
  /// Which decode window this progress update belongs to.
  #[inline(always)]
  pub const fn window_id(&self) -> usize {
    self.window_id
  }
  /// Builder form of [`Self::set_window_id`].
  #[must_use]
  #[inline(always)]
  pub const fn with_window_id(mut self, window_id: usize) -> Self {
    self.set_window_id(window_id);
    self
  }
  /// Sets [`Self::window_id`] in place.
  #[inline(always)]
  pub const fn set_window_id(&mut self, window_id: usize) -> &mut Self {
    self.window_id = window_id;
    self
  }
}

// ---------------------------------------------------------------------
// FallbackReason / needs_fallback
// ---------------------------------------------------------------------

/// Reason a decoding window must be retried at the next (higher)
/// temperature — the `needsFallback: true` outcomes of Swift's
/// `DecodingFallback.init?` (`Models.swift:357-381`). Swift's fourth
/// `fallbackReason`, `"silence"` (`Models.swift:370`), carries
/// `needsFallback: false` — it never requires a retry, so
/// [`needs_fallback`] folds it into `None` rather than modeling it as a
/// variant here (see that function's doc comment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum FallbackReason {
  /// The first sampled token's log probability fell below
  /// `DecodingOptions::first_token_logprob_threshold` (Swift
  /// `Models.swift:367`).
  FirstTokenLogProbThreshold,
  /// The window's text compression ratio exceeded
  /// `DecodingOptions::compression_ratio_threshold` — too repetitive
  /// (Swift `Models.swift:373`).
  CompressionRatioThreshold,
  /// The window's average sampled-token log probability fell below
  /// `DecodingOptions::logprob_threshold` (Swift `Models.swift:376`).
  LogProbThreshold,
}

impl FallbackReason {
  /// Stable name matching Swift's `DecodingFallback.fallbackReason`
  /// string exactly (`Models.swift:367,373,376`).
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::FirstTokenLogProbThreshold => "firstTokenLogProbThreshold",
      Self::CompressionRatioThreshold => "compressionRatioThreshold",
      Self::LogProbThreshold => "logProbThreshold",
    }
  }
}

/// Ports Swift's `DecodingFallback.init?(options:isFirstTokenLogProbTooLow:
/// noSpeechProb:compressionRatio:avgLogProb:)`
/// (`argmax-oss-swift/Sources/WhisperKit/Core/Models.swift:357-381`).
///
/// **The caller (the decode loop) computes `first_token_log_prob_too_low`**
/// by comparing the first sampled token's log probability against
/// `options.first_token_logprob_threshold()` *inside* the decode loop
/// (`TextDecoder.swift:662-667`). This value is not recoverable from
/// `decoding.token_log_probs_slice()` because the prefill phase does not
/// append the first token's log probability to that vector — the first
/// entry stored there is actually a placeholder or undefined for prefill
/// results. Therefore, the caller must compute this flag from the
/// loop-local first sampled token's log probability and thread it through
/// to this function as a parameter (mirroring Swift's signature).
///
/// **Decision order matters** (this is the Swift source's own comment,
/// `Models.swift:365`) and every comparison is strict, exactly matching
/// the source:
///
/// 1. `first_token_log_prob_too_low` is `true` -> `Some(FirstTokenLogProbThreshold)`.
/// 2. else, if `no_speech_prob` `>` threshold -> **silence**, returns
///    `None` unconditionally. This step does *not* also consult
///    `avg_logprob` — an earlier exploration of this port assumed it
///    did; `Models.swift:368-370` shows it does not, and the source
///    wins.
/// 3. else, if `compression_ratio` `>` threshold -> `Some(CompressionRatioThreshold)`.
/// 4. else, if `avg_logprob` `<` threshold -> `Some(LogProbThreshold)`.
/// 5. else `None` — a clean result, or every configured threshold was
///    `None` (disabled).
///
/// Any threshold left `None` in `options` disables its own check and
/// falls through to the next step, matching Swift's `if let threshold =
/// options.xThreshold` optional-binding guards.
pub fn needs_fallback(
  first_token_log_prob_too_low: bool,
  decoding: &DecodingResult,
  options: &DecodingOptions,
) -> Option<FallbackReason> {
  if first_token_log_prob_too_low {
    return Some(FallbackReason::FirstTokenLogProbThreshold);
  }
  if let Some(threshold) = options.no_speech_threshold()
    && decoding.no_speech_prob() > threshold
  {
    return None;
  }
  if let Some(threshold) = options.compression_ratio_threshold()
    && decoding.compression_ratio() > threshold
  {
    return Some(FallbackReason::CompressionRatioThreshold);
  }
  if let Some(threshold) = options.logprob_threshold()
    && decoding.avg_logprob() < threshold
  {
    return Some(FallbackReason::LogProbThreshold);
  }
  None
}

// ---------------------------------------------------------------------
// format_segments
// ---------------------------------------------------------------------

/// Renders `segments` as display lines, one per segment — ports
/// `TranscriptionUtilities.formatSegments`
/// (`Utilities/TranscriptionUtilities.swift:16-27`), timestamp formatting
/// included (`Logging.formatTimestamp`, `Utilities/Logging.swift:50-52`,
/// `String(format: "%.2f", _)`, i.e. Rust's `{:.2}`).
///
/// When `with_timestamps` is `true`, each line is `"[{start:.2} -->
/// {end:.2}] {text}"`; when `false`, each line is `text` verbatim. Note
/// the bracketed timestamp carries its own trailing space (Swift's own
/// string interpolation builds `"[...] "`), so a segment `text` that
/// itself starts with a leading space renders with two spaces before the
/// first word — faithful to Swift, not a bug.
pub fn format_segments(segments: &[TranscriptionSegment], with_timestamps: bool) -> Vec<String> {
  segments
    .iter()
    .map(|segment| {
      if with_timestamps {
        format!(
          "[{:.2} --> {:.2}] {}",
          segment.start(),
          segment.end(),
          segment.text()
        )
      } else {
        segment.text().to_string()
      }
    })
    .collect()
}

// ---------------------------------------------------------------------
// merge_transcription_results
// ---------------------------------------------------------------------

/// Sums `f(result.timings())` over `results` — the "work time"/count merge
/// rule (Swift `.reduce(0, +)`).
fn sum_timing(results: &[TranscriptionResult], f: impl Fn(&TranscriptionTimings) -> f64) -> f64 {
  results.iter().map(|r| f(r.timings())).sum()
}

/// Maximum of `f(result.timings())` over `results`, `0.0` if `results` is
/// empty — the "load time" merge rule (Swift `.max() ?? 0`).
fn max_timing(results: &[TranscriptionResult], f: impl Fn(&TranscriptionTimings) -> f64) -> f64 {
  results
    .iter()
    .map(|r| f(r.timings()))
    .max_by(f64::total_cmp)
    .unwrap_or(0.0)
}

/// Minimum of `f(result.timings())` over `results`, `0.0` if `results` is
/// empty — the "earliest pipeline mark" merge rule (Swift `.min() ?? 0`).
fn min_timing(results: &[TranscriptionResult], f: impl Fn(&TranscriptionTimings) -> f64) -> f64 {
  results
    .iter()
    .map(|r| f(r.timings()))
    .min_by(f64::total_cmp)
    .unwrap_or(0.0)
}

/// Merges several per-chunk/per-window [`TranscriptionResult`]s into one.
/// Ports `TranscriptionUtilities.mergeTranscriptionResults`
/// (`TranscriptionUtilities.swift:76-160`), minus the streaming-only
/// `confirmedWords` parameter (Plan 4 territory): this port always takes
/// Swift's plain-text-join `else` branch (:82-84), and `results` is a
/// plain slice rather than Swift's `[TranscriptionResult?]` — there is no
/// per-element "missing result" case to `compactMap` away here, so every
/// entry of `results` participates (Swift's `validResults`).
///
/// - [`TranscriptionResult::text`][]: every result's text, joined with `"
///   "` (:82-84; empty for empty `results`, matching `[].joined(separator:
///   " ") == ""`). **An empty-text result is joined as a bare separator,
///   not skipped** — faithfully, because Swift's `validResults`
///   `compactMap`s away only *nil* elements (`:80`), never empty-text ones,
///   so `["a", "", "b"].joined(separator: " ")` is `"a  b"` there too. A
///   zero-segment, empty-text result is reachable on its own — any audio
///   shorter than [`DecodingOptions::window_clip_time`](crate::audio::whisper::options::DecodingOptions::window_clip_time)
///   runs no window at all and returns one — and this port keeps the
///   quirk rather than "fixing" it, exactly like the segment re-`id`
///   below. This function is therefore the merge for
///   [`DecodingOptions::drop_blank_audio`](crate::audio::whisper::options::DecodingOptions::drop_blank_audio)
///   `== false` — exact Swift — by definition;
///   [`merge_transcription_results_with_options`] is the entry point that
///   skips the empties instead, for the callers whose own options made an
///   emptied result routine. Both share one implementation.
/// - [`TranscriptionResult::segments_slice`][]: every result's segments,
///   concatenated in order, each re-`id`'d to `result_index +
///   segment_index` (:89-94) — a faithful bug-for-bug port: upstream
///   renumbers segments this way, not sequentially across the whole
///   merged list (verified against source, not "fixed"). Swift's
///   `previousSeek`/local `seekTime` bookkeeping in this same loop
///   (:90-99) is not ported: it is dead code in the source itself — every
///   value it computes is either overwritten next iteration or never read
///   again, and the returned `TranscriptionResult(...)` call at the end
///   never consumes it either. `results` are expected to already carry
///   correct per-segment/per-result seek anchoring from
///   [`crate::audio::whisper::audio::chunker::apply_result_seek_offset`] before reaching
///   this function, exactly as Swift's `updateSeekOffsetsForResults`
///   re-anchors chunk results before its own call into this merge.
/// - [`TranscriptionResult::language`][]: the first result's language, or
///   [`DEFAULT_LANGUAGE_CODE`] if `results` is empty (:104).
/// - [`TranscriptionResult::timings`][]: [`TranscriptionTimings::model_loading`]/
///   [`prewarm_load_time`](TranscriptionTimings::prewarm_load_time)/
///   [`encoder_load_time`](TranscriptionTimings::encoder_load_time)/
///   [`decoder_load_time`](TranscriptionTimings::decoder_load_time)/
///   [`tokenizer_load_time`](TranscriptionTimings::tokenizer_load_time)
///   take the max across results; every work-time/count field sums
///   (:106-152); [`pipeline_start`](TranscriptionTimings::pipeline_start)/
///   [`first_token_time`](TranscriptionTimings::first_token_time) take the
///   min; [`input_audio_seconds`](TranscriptionTimings::input_audio_seconds)
///   sums; [`full_pipeline`](TranscriptionTimings::full_pipeline) is
///   `user_pipeline_duration.min(system_pipeline_duration)`, where
///   `user_pipeline_duration` is the wall-clock span from the earliest
///   `pipeline_start` to the latest `pipeline_start + full_pipeline`
///   across results, and `system_pipeline_duration` is the sum of every
///   result's own `full_pipeline`. **Documented deviation — the
///   all-sentinel case is special-cased**, because this port's inputs
///   differ from Swift's: Swift's `TranscribeTask.run` stamps a real
///   `CFAbsoluteTimeGetCurrent()` into `pipelineStart`
///   (`TranscribeTask.swift:65`), so its merge always sees finite starts;
///   neither this crate's [`TranscribeTask::run`](crate::audio::whisper::transcribe::TranscribeTask::run)
///   nor [`crate::audio::whisper::decode::detect_language`] ever populates
///   `pipeline_start`/`first_token_time` (no absolute wall clock exists
///   in this sync port to stamp them with — see `crate::audio::whisper::decode`'s module
///   doc), so every real result's `pipeline_start` is still
///   [`DEFAULT_PIPELINE_TIME_SENTINEL`] (`f64::MAX`), where the verbatim
///   Swift subtraction silently degenerates to `0.0` and would zero out
///   the merged `full_pipeline` entirely (see the in-body comment for the
///   exact float arithmetic). When no result carries a real
///   `pipeline_start`, `user_pipeline_duration` is therefore treated as
///   unbounded, collapsing the formula to `system_pipeline_duration` —
///   the sum, the honest value for sync sequential composition. Results
///   that do carry real stamps still merge through Swift's formula
///   verbatim.
///   [`encoder_specialization_time`](TranscriptionTimings::encoder_specialization_time)/
///   [`decoder_specialization_time`](TranscriptionTimings::decoder_specialization_time)
///   are **not** carried into the merged timings — Swift's own
///   `TranscriptionTimings(...)` call inside `mergeTranscriptionResults`
///   omits both parameters, so they take that initializer's `0` default
///   regardless of what the source results held; ported faithfully
///   (left at [`TranscriptionTimings::new`]'s own `0.0` default) rather
///   than folded into the max-for-load-times group they would naively
///   belong to.
pub fn merge_transcription_results(results: &[TranscriptionResult]) -> TranscriptionResult {
  // `false` — Swift's own join: every text participates, an empty one as a
  // bare separator. See this function's doc for why that is not "fixed".
  merge_results(results, false)
}

/// Merges `results` exactly as [`merge_transcription_results`] does, but
/// takes the [`DecodingOptions`] they were decoded under and applies the
/// one merge rule those options govern: when
/// [`DecodingOptions::drop_blank_audio`] is set (**the default**), a result
/// whose text is empty contributes **nothing to the text join** — not the
/// bare `" "` separator Swift's join gives it.
///
/// This is the entry point for folding a
/// [`WhisperKit::transcribe_all`](crate::audio::whisper::transcribe::WhisperKit::transcribe_all)
/// batch, and the one
/// [`WhisperKit::transcribe`](crate::audio::whisper::transcribe::WhisperKit::transcribe)'s
/// VAD branch uses for its own chunk results. Hand it the same `options`
/// the results were decoded with and the merged text cannot contradict
/// them.
///
/// # Why the option has to reach the merge at all
///
/// [`DecodingOptions::drop_blank_audio`] makes an **empty result routine**:
/// a wholly-silent VAD chunk — the chunker is *contiguous*, so silence is
/// cut around, never skipped — decodes to nothing but
/// [`BLANK_AUDIO_MARKER`](crate::audio::whisper::constants::BLANK_AUDIO_MARKER), the filter
/// removes that one segment, and the chunk is left with no text at all.
/// Joined Swift's way, every such chunk lands in the transcript as a bare
/// separator: a doubled space between two speech runs, a leading or
/// trailing one at the clip's edges. [`merge_transcription_results`] cannot
/// simply filter them out, because an empty-text result is **not** unique
/// to the drop — any audio shorter than
/// [`DecodingOptions::window_clip_time`] runs no window and returns one,
/// which predates this option entirely — and Swift joins *those* as bare
/// separators too, so filtering there would silently change the
/// `drop_blank_audio == false` path, whose whole purpose is to be
/// byte-for-byte Swift. The rule therefore travels with the option that
/// created the need for it, and this is where the two meet.
///
/// # What is skipped is *empty text*, not *blank audio*
///
/// The merge cannot see **why** a result came back empty, and deliberately
/// does not ask. With `drop_blank_audio` set, an empty result from
/// **short audio** is skipped from the join exactly like an emptied blank
/// chunk. That is the intended reading of the option — *blank-dropping
/// means empty chunks do not pollute the text* — not an accidental
/// over-reach: a caller who asked not to see `[BLANK_AUDIO]` has no more
/// use for a bare separator standing in for a sub-second clip than for one
/// standing in for silence. Callers who want Swift's join for every input,
/// empties included, call [`merge_transcription_results`] — or clear the
/// option, which makes this function that one exactly.
///
/// # Every result is still merged
///
/// Only the *join* skips empties. Segment concatenation and every timing
/// reduction run over **all** of `results` either way, so the merged
/// segments' text and words, and every timing field — the summed
/// [`input_audio_seconds`](TranscriptionTimings::input_audio_seconds) and
/// [`audio_processing`](TranscriptionTimings::audio_processing), the
/// [`real_time_factor`](TranscriptionTimings::real_time_factor) derived
/// from them, all of it — are byte-identical to
/// [`merge_transcription_results`]'s on the same input, whichever way the
/// option is set. Dropping an emptied result from the merge *input* instead
/// would take its metrics out with it, quietly corrupting the sums (and the
/// RTF) to fix a spacing bug.
///
/// The option is **not** confined to [`TranscriptionResult::text`], though.
/// It also selects the segment **id mapping**: a running injective base when
/// dropping, Swift's `result_index + segment_index` when not (see the section
/// below). So the merged segments' ids — but nothing else about them — can
/// differ between the two settings even when every text is non-empty. (The
/// [`confirmed_words`](merge_transcription_results_with_words) door notes the
/// same, and depends on it.)
///
/// # The segment id mapping depends on the option, too
///
/// With the drop ON (the default), each chunk's survivors are re-`id`'d onto a
/// running base that advances by the chunk's decoded id span — injective across
/// chunks, each chunk's local gaps preserved. With it OFF, the ids are exactly
/// Swift's `result_index + segment_index`, which **duplicates** ids across a
/// multi-segment chunk (pinned parity). Two chunks with local ids `[0, 2]` and
/// `[0, 1]` therefore merge to `[0, 2, 3, 4]` dropped but `[0, 1, 1, 2]` not:
///
/// ```
/// use coremlit::audio::whisper::options::DecodingOptions;
/// use coremlit::audio::whisper::result::{
///   TranscriptionResult, TranscriptionSegment, TranscriptionTimings,
///   merge_transcription_results_with_options,
/// };
///
/// // Two chunks whose survivors sit at local ids [0, 2] and [0, 1].
/// let chunk = |ids: &[usize]| {
///   let segments: Vec<TranscriptionSegment> = ids
///     .iter()
///     .map(|&id| {
///       let mut s = TranscriptionSegment::new();
///       s.set_id(id).set_text(" w");
///       s
///     })
///     .collect();
///   TranscriptionResult::new(" w w", segments, "en", TranscriptionTimings::new())
/// };
/// let results = [chunk(&[0, 2]), chunk(&[0, 1])];
/// let ids =
///   |r: &TranscriptionResult| r.segments_slice().iter().map(|s| s.id()).collect::<Vec<_>>();
///
/// // Dropping ON (the default): a running injective base -> [0, 2, 3, 4].
/// let dropped = merge_transcription_results_with_options(&results, &DecodingOptions::new());
/// assert_eq!(ids(&dropped), [0, 2, 3, 4]);
///
/// // Dropping OFF: Swift's result_index + segment_index -> [0, 1, 1, 2] (ids collide).
/// let swift = merge_transcription_results_with_options(
///   &results,
///   &DecodingOptions::new().maybe_drop_blank_audio(false),
/// );
/// assert_eq!(ids(&swift), [0, 1, 1, 2]);
/// ```
///
/// # Panics
///
/// When [`DecodingOptions::drop_blank_audio`] is set, each chunk's segment ids
/// are re-mapped onto a running base advancing by the chunk's id span, so the
/// merged ids stay injective while preserving each chunk's local gaps. That
/// arithmetic is checked: a hand-built segment id near [`usize::MAX`] panics
/// deliberately rather than wrapping into a colliding id. Pipeline-produced
/// ids are small decode ordinals and never approach this.
pub fn merge_transcription_results_with_options(
  results: &[TranscriptionResult],
  options: &DecodingOptions,
) -> TranscriptionResult {
  merge_results(results, options.drop_blank_audio())
}

/// Each child's **effective** span knowledge for the drop-ON merge: its carried
/// [`TaskFacts::decoded_span`] **floored by its own survivors' extent** — `max
/// local id + 1`, or `0` when none survived. This is the round-9 F2 trust rule
/// lifted from a bare number to a [`SpanKnowledge`](crate::audio::whisper::task_facts::SpanKnowledge):
/// a carried bound legitimately EXCEEDS the extent when a filter dropped ordinals
/// after allocating them (the whole reason it is carried), but a carried bound
/// BELOW the extent under-counts — a hand-built or deserialized inconsistency —
/// so the extent is the trusted floor, never blindly the carried value.
///
/// - [`Exact(n)`](crate::audio::whisper::task_facts::SpanKnowledge::Exact) stays `Exact(n)` when
///   the extent does not exceed `n`; when the survivors OUT-count `n` the exact
///   claim is contradicted, so it degrades to `AtLeast(extent)` — a bound, no
///   longer a fabricated exact total.
/// - [`AtLeast(k)`](crate::audio::whisper::task_facts::SpanKnowledge::AtLeast) floors to
///   `AtLeast(max(k, extent))`; a [wholly-unknown](crate::audio::whisper::task_facts::SpanKnowledge::wholly_unknown)
///   `AtLeast(0)` therefore relies on the extent outright.
///
/// This ONE value drives BOTH the drop-ON id-base advance AND the drop-ON stored
/// fold (codex round 13, M1): the base advances by its
/// [lower bound](crate::audio::whisper::task_facts::SpanKnowledge::lower_bound), and the merged
/// result STORES the fold of these same effective spans — so every staging
/// materializes the same floors at the same points and the stored fact after any
/// partial merge carries them. Folding the RAW carried spans instead left the
/// store staging-dependent: a left-staged `merge([merge([A, B]), T])` stored a
/// bound below the floors the one-shot ids had already committed to, so a further
/// re-merge (its survivor extent too small to recover the lost floor) renumbered a
/// trailing chunk onto an id the one-shot merge left free.
///
/// **`None` when the survivors' extent overflows `usize`** (a hand-built segment
/// id at [`usize::MAX`]): the extent is then unrepresentable. At the drop-ON
/// id-base advance a `None` is the documented DELIBERATE panic — the span there
/// drives an injective id mapping that cannot proceed past `usize::MAX` without a
/// colliding wraparound (see [`merge_transcription_results_with_options`]'s own
/// `# Panics`). The drop-OFF fold never calls this at all: its ids are Swift's
/// `result_index + segment_index` and never consult the span, so it folds the RAW
/// carried spans and never touches the overflowing extent (codex round 8, F4).
fn effective_span_knowledge(result: &TranscriptionResult) -> Option<SpanKnowledge> {
  // The survivors' own extent — `max local id + 1`, `0` when none survived, and
  // `None` (propagated out of this function) when a `usize::MAX` survivor
  // overflows it (genuinely unrepresentable).
  let survivor_extent = match result
    .segments_slice()
    .iter()
    .map(TranscriptionSegment::id)
    .max()
  {
    None => 0,
    Some(max_local_id) => max_local_id.checked_add(1)?,
  };
  // Floor the carried lower bound at the survivor extent (F2, codex round 9): the
  // extent is a floor the carried bound may exceed but must never fall below. An
  // exact count the extent does not out-count stays exact; otherwise the value
  // degrades to an at-least bound AT the floor rather than a contradicted exact.
  let carried = result.task_facts().decoded_span();
  let floored = carried.lower_bound().max(survivor_extent);
  Some(if carried.is_exact() && floored == carried.lower_bound() {
    SpanKnowledge::Exact(floored)
  } else {
    SpanKnowledge::AtLeast(floored)
  })
}

/// The single merge implementation behind [`merge_transcription_results`]
/// and [`merge_transcription_results_with_options`].
///
/// `skip_empty_texts` governs the **text join and the segment id mapping**:
/// every result participates in the segment concatenation and in every timing
/// reduction regardless of it, so the two entry points can differ only in
/// [`TranscriptionResult::text`] and in the segments' **ids** (a running
/// injective base when dropping, Swift's `result_index + segment_index` when
/// not — see [`merge_transcription_results_with_options`]'s own doc). Keeping
/// that promise structural — one body, one `filter`, reached through both
/// doors — is the point of the split: the alternative (a second join written
/// out at the call site that happens to know the option) is exactly how the
/// two drifted apart before.
fn merge_results(results: &[TranscriptionResult], skip_empty_texts: bool) -> TranscriptionResult {
  let text = results
    .iter()
    .map(TranscriptionResult::text)
    .filter(|text| !(skip_empty_texts && text.is_empty()))
    .collect::<Vec<_>>()
    .join(" ");

  let mut segments = Vec::new();
  // Running id base for the drop-ON (`skip_empty_texts`) mapping. Each chunk's
  // survivors are re-`id`'d to `id_base + segment.id()` (its **decode
  // ordinal**, its position within its own chunk), and `id_base` then advances
  // by that chunk's decoded id SPAN. This is INJECTIVE across chunks (no two
  // chunks' id windows overlap) while preserving each chunk's own local gaps: a
  // blank dropped mid-chunk leaves `[.., 2]` where segment 1 was, and that hole
  // survives inside the chunk's window as the audit trail `drop_blank_audio`
  // promises. The earlier `result_index + segment.id()` COLLIDED the moment a
  // chunk had more than one segment — `[0,1] + [0,1]` renumbered to `[0,1,1,2]`,
  // and a blank-dropped `[0,2] + [0,1]` to `[0,2,1,2]` — because `result_index`
  // advances by 1 per chunk regardless of how many ids the chunk actually
  // spans. `id_base` advances by the real span instead. With dropping OFF the
  // false path stays EXACTLY Swift's `resultIndex + segmentIndex`, byte-for-byte
  // (its duplicate ids are pinned parity), and `id_base` is left untouched.
  //
  // The span is the chunk's DECODED ordinal count (`TaskFacts::decoded_span`),
  // carried on the result, NOT the surviving segments' `max local id + 1`: a
  // chunk whose segments were ALL dropped (a blank-only VAD chunk) survives with
  // zero segments yet still consumed ordinals, and inferring the span from the
  // survivors would collapse it to 0 — indistinguishable from a genuinely
  // zero-window chunk — so the NEXT chunk's survivors would renumber down onto
  // this one's window (coremlit issue #14, codex round 5). A hand-built or
  // deserialized result carries no span; the merge then falls back to the
  // survivors' own extent, its pre-existing behavior.
  let mut id_base = 0usize;
  for (result_index, result) in results.iter().enumerate() {
    for (segment_index, segment) in result.segments_slice().iter().enumerate() {
      let id = if skip_empty_texts {
        // Checked: a hand-built `usize::MAX` id is adversarial input, so a
        // deliberate documented panic beats a silent wraparound collision (see
        // this function's `# Panics`).
        id_base.checked_add(segment.id()).expect(
          "drop_blank_audio segment-id mapping overflowed usize (a segment id near usize::MAX)",
        )
      } else {
        result_index + segment_index
      };
      segments.push(segment.clone().with_id(id));
    }
    if skip_empty_texts {
      // Advance past this chunk's DECODED id window by exactly the effective
      // span's lower bound — the carried span floored at the survivors' own extent
      // ([`effective_span_knowledge`]). That floor keeps a re-merge from
      // under-counting the survivors it is renumbering, and the STORED fold below
      // now folds this SAME effective span, so a staged re-merge stores the very
      // floor it advanced by (codex round 13, M1). Here — and ONLY here, where the
      // span drives an injective id mapping — a `None` span (a `usize::MAX`
      // survivor whose extent overflowed) OR an overflowing base is the documented
      // adversarial-input panic; the drop-OFF fold, which never uses the span for
      // ids, keeps the checked `None` instead (codex round 8, F4).
      id_base = effective_span_knowledge(result)
        .and_then(|span| id_base.checked_add(span.lower_bound()))
        .expect("drop_blank_audio segment-id base overflowed usize (a segment id near usize::MAX)");
    }
  }

  let language = results.first().map_or_else(
    || DEFAULT_LANGUAGE_CODE.to_string(),
    |first| first.language().to_string(),
  );

  let earliest_pipeline_start = min_timing(results, TranscriptionTimings::pipeline_start);
  let latest_pipeline_end = results
    .iter()
    .map(|r| r.timings().pipeline_start() + r.timings().full_pipeline())
    .max_by(f64::total_cmp)
    .unwrap_or(0.0);
  // With NO real pipeline_start stamped anywhere (every value still the
  // f64::MAX sentinel — what every result this sync port produces looks
  // like), the wall-clock span is unknowable and the subtraction below
  // would silently compute 0.0, NOT the intended sum: `f64::MAX +
  // full_pipeline` ABSORBS back to exactly f64::MAX (the ULP at that
  // magnitude is ~2e292, far above any real duration; it does not
  // overflow to infinity), so `latest - earliest` is `f64::MAX - f64::MAX
  // == 0.0` and the min() would zero out full_pipeline. An unknowable
  // user duration is unbounded, not zero — INFINITY hands min() to the
  // summed work time. A mixed batch (some real stamps) needs no guard:
  // `earliest` is then finite and `latest` is ~f64::MAX, so the huge span
  // already loses the min() to the sum on its own.
  let user_pipeline_duration = if earliest_pipeline_start == DEFAULT_PIPELINE_TIME_SENTINEL {
    f64::INFINITY
  } else {
    latest_pipeline_end - earliest_pipeline_start
  };
  let system_pipeline_duration = sum_timing(results, TranscriptionTimings::full_pipeline);

  let mut timings = TranscriptionTimings::new();
  timings
    .set_model_loading(max_timing(results, TranscriptionTimings::model_loading))
    .set_prewarm_load_time(max_timing(results, TranscriptionTimings::prewarm_load_time))
    .set_encoder_load_time(max_timing(results, TranscriptionTimings::encoder_load_time))
    .set_decoder_load_time(max_timing(results, TranscriptionTimings::decoder_load_time))
    .set_tokenizer_load_time(max_timing(
      results,
      TranscriptionTimings::tokenizer_load_time,
    ))
    .set_audio_loading(sum_timing(results, TranscriptionTimings::audio_loading))
    .set_audio_processing(sum_timing(results, TranscriptionTimings::audio_processing))
    .set_logmels(sum_timing(results, TranscriptionTimings::logmels))
    .set_encoding(sum_timing(results, TranscriptionTimings::encoding))
    .set_decoding_init(sum_timing(results, TranscriptionTimings::decoding_init))
    .set_decoding_loop(sum_timing(results, TranscriptionTimings::decoding_loop))
    .set_decoding_predictions(sum_timing(
      results,
      TranscriptionTimings::decoding_predictions,
    ))
    .set_decoding_filtering(sum_timing(
      results,
      TranscriptionTimings::decoding_filtering,
    ))
    .set_decoding_sampling(sum_timing(results, TranscriptionTimings::decoding_sampling))
    .set_decoding_fallback(sum_timing(results, TranscriptionTimings::decoding_fallback))
    .set_decoding_windowing(sum_timing(
      results,
      TranscriptionTimings::decoding_windowing,
    ))
    .set_decoding_kv_caching(sum_timing(
      results,
      TranscriptionTimings::decoding_kv_caching,
    ))
    .set_decoding_word_timestamps(sum_timing(
      results,
      TranscriptionTimings::decoding_word_timestamps,
    ))
    .set_decoding_non_prediction(sum_timing(
      results,
      TranscriptionTimings::decoding_non_prediction,
    ))
    .set_total_audio_processing_runs(sum_timing(
      results,
      TranscriptionTimings::total_audio_processing_runs,
    ))
    .set_total_logmel_runs(sum_timing(results, TranscriptionTimings::total_logmel_runs))
    .set_total_encoding_runs(sum_timing(
      results,
      TranscriptionTimings::total_encoding_runs,
    ))
    .set_total_decoding_loops(sum_timing(
      results,
      TranscriptionTimings::total_decoding_loops,
    ))
    .set_total_kv_update_runs(sum_timing(
      results,
      TranscriptionTimings::total_kv_update_runs,
    ))
    .set_total_timestamp_alignment_runs(sum_timing(
      results,
      TranscriptionTimings::total_timestamp_alignment_runs,
    ))
    .set_total_decoding_fallbacks(sum_timing(
      results,
      TranscriptionTimings::total_decoding_fallbacks,
    ))
    .set_total_decoding_windows(sum_timing(
      results,
      TranscriptionTimings::total_decoding_windows,
    ))
    .set_input_audio_seconds(sum_timing(
      results,
      TranscriptionTimings::input_audio_seconds,
    ))
    .set_full_pipeline(user_pipeline_duration.min(system_pipeline_duration))
    .set_pipeline_start(earliest_pipeline_start)
    .set_first_token_time(min_timing(results, TranscriptionTimings::first_token_time));

  // The task facts, folded left-to-right through the ONE merge law
  // ([`TaskFacts::merge`]) rather than the four scattered `.any()`/`.find_map()`/
  // `.first()` reductions this replaced — the consolidation that closes R6-F1/F2/F3
  // (coremlit issue #14, codex round 6). Over the children in order it:
  //
  // - **OR**s the RNG-draw and early-stop facts: a chunk the blank-audio drop
  //   emptied contributes NO segments to the merge, so its accepted temperature
  //   is invisible in the merged segment list — the fact has to travel with the
  //   result, never be read back off the output it no longer appears in. A
  //   callback truncated the merge if it truncated ANY chunk (R6-F1's merge side).
  // - keeps the **first** genuine language observation (a later `Some` cannot
  //   overwrite an earlier one), deliberately NOT agreeing with the merged
  //   DISPLAY `language` (the first result's, keeping its Swift-compat `"en"`
  //   fallback).
  // - **concatenates** the worker schedules in order, so `[0, 2]` stays distinct
  //   from `[0, 1]` instead of collapsing to the first child's coordinate (R6-F2).
  // - folds each child's **span** under [`SpanKnowledge::merge`]. WHICH span it
  //   folds depends on the id mapping this merge performs (codex round 13, M1):
  //   * **drop-ON** folds each child's EFFECTIVE span — its carried span floored
  //     at its own survivor extent ([`effective_span_knowledge`]), the SAME value
  //     the id-base advanced by — so the stored fact carries the floors the ids
  //     already committed to, and a staged re-merge numbers identically to a
  //     one-shot one (R6-F3). Folding the RAW carried spans here instead stored a
  //     bound BELOW those floors, and a further re-merge whose survivor extent
  //     could not recover them renumbered a trailing chunk onto a consumed id.
  //   * **drop-OFF** folds each child's RAW carried span: its ids are Swift's
  //     `result_index + segment_index` and never consult the span, so the fold
  //     must not touch the (possibly `usize::MAX`-overflowing) survivor extent
  //     (codex round 8, F4).
  //   Either way the sum is checked-add over exacts (an overflow degrades to a
  //   saturated `AtLeast(usize::MAX)`, never a wrapped or fabricated exact) and
  //   saturating once any child is a mere `AtLeast`, associative by construction.
  //   Unlike the pre-round-12 absorbing-`None` fold, an unknown child no longer
  //   erases a known sibling's ordinals: `AtLeast(0) + Exact(1) = AtLeast(1)`, so
  //   the known lower bound survives to drive a staged re-merge's id base.
  //
  // Folded through [`TaskFactsAccumulator`], NOT seeded at `TaskFacts::unknown()`:
  // under the Kleene OR (codex round 8, F2) `unknown()` is no longer the merge
  // identity — seeding there would null a first child's observed-clean
  // `Some(false)` to `None` — so the accumulator takes the first contributor
  // verbatim and folds the rest, and yields `unknown()` only for an empty
  // `results`.
  let mut task_facts = TaskFactsAccumulator::new();
  for result in results {
    if skip_empty_texts {
      // Drop-ON: fold the EFFECTIVE span — the floored value this child's id-base
      // advance used above — so the stored fact materializes the same floors the
      // ids did and a staged re-merge numbers identically (codex round 13, M1).
      // The `usize::MAX`-survivor overflow cannot reach here: the id loop above
      // already panicked on it, so the saturated fallback only keeps this a total
      // function and never actually materializes on the drop-ON path.
      let effective =
        effective_span_knowledge(result).unwrap_or(SpanKnowledge::AtLeast(usize::MAX));
      task_facts.merge(&result.task_facts().clone().with_decoded_span(effective));
    } else {
      // Drop-OFF: fold the RAW carried span; the Swift `result_index +
      // segment_index` ids never consult it, so the fold must not touch the
      // (possibly overflowing) survivor extent (codex round 8, F4).
      task_facts.merge(result.task_facts());
    }
  }
  let task_facts = task_facts.into_facts();

  TranscriptionResult::new(text, segments, language, timings).with_task_facts(task_facts)
}

// ---------------------------------------------------------------------
// merge_transcription_results_with_words
// ---------------------------------------------------------------------

/// Merges `results` under `options`, then overrides the merged text with
/// `confirmed_words` — ports the `confirmedWords:` branch of
/// `TranscriptionUtilities.mergeTranscriptionResults`
/// (`Utilities/TranscriptionUtilities.swift:76-82`): `words.map { $0.word
/// }.joined()`, i.e. every confirmed word's text concatenated with **no**
/// separator (word strings carry their own leading spaces, e.g. `" And"`).
/// Everything else — segments, language, every timing field — is
/// byte-identical to [`merge_transcription_results_with_options`]'s own
/// output.
///
/// # Why this needs the options after all
///
/// The `confirmed_words` override discards the merged **text join**, so it is
/// tempting to think [`DecodingOptions::drop_blank_audio`] is unobservable
/// here and to delegate to the plain, options-blind merge. It is not:
/// dropping now governs the **segment id mapping** too, not just the text
/// (see [`merge_transcription_results_with_options`]). Delegating to the plain
/// (drop-OFF) merge collapsed a survivor id gap `[0, 2]` back to a dense
/// `[0, 1]`, and [`crate::audio::whisper::stream::agreement::LocalAgreement::finalize`] — the
/// default streaming path — inherited that loss at finalization. Threading the
/// same `options` the results were decoded under keeps the merged **segments**
/// honoring the drop even when the merged text does not come from them.
pub fn merge_transcription_results_with_words(
  results: &[TranscriptionResult],
  confirmed_words: &[WordTiming],
  options: &DecodingOptions,
) -> TranscriptionResult {
  let mut merged = merge_results(results, options.drop_blank_audio());
  let text: String = confirmed_words.iter().map(WordTiming::word).collect();
  merged.set_text(text);
  merged
}

#[cfg(test)]
mod tests;
