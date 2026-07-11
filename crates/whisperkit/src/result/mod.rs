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
//! feature) never emits `null`; `Vec`/`String` fields that carry
//! meaningful "not present" semantics are empty-means-absent
//! (`skip_serializing_if` + `default`, golden §10).
//!
//! [`needs_fallback`]'s decision order and comparisons are copied verbatim
//! from Swift's `DecodingFallback.init?` (`Models.swift:357-381`) — see
//! its doc comment for the exact citations, including a correction to
//! this task's own brief (the "silence" short-circuit does not consult
//! `avg_logprob`, contrary to the brief's exploration).

use crate::options::DecodingOptions;

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
  #[cfg_attr(feature = "serde", serde(default = "default_segment_temperature"))]
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
  /// Time spent on audio pre-processing (pad/trim, energy, VAD).
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
  /// Number of temperature-fallback retries performed.
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
  /// Time spent on audio pre-processing (pad/trim, energy, VAD).
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
  /// Number of temperature-fallback retries performed.
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
}

impl TranscriptionResult {
  /// Builds a result from its four required fields (Swift
  /// `TranscriptionResultStruct.init`, `Models.swift:550-562`, has no
  /// defaults for these either); [`Self::seek_time`] starts `None`.
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
      tokens: Vec::new(),
      token_log_probs: Vec::new(),
      text: String::new(),
      avg_logprob: 0.0,
      no_speech_prob: 0.0,
      temperature: 0.0,
      compression_ratio: 0.0,
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

#[cfg(test)]
mod tests;
