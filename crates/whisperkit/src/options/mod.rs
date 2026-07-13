//! Configuration types for the WhisperKit pipeline (spec §5.3, §6.2): the
//! 27-knob decode-time surface Swift exposes, plus one Rust-only
//! reproducibility addition ([`DecodingOptions`]), per-stage compute unit
//! selection ([`ComputeOptions`]), and construction config ([`Options`]).
//! Ports `Configurations.swift` `DecodingOptions`/`WhisperKitConfig` and
//! `Models.swift` `DecodingTask`/`ChunkingStrategy`/`ModelComputeOptions`.
//!
//! Reference implementation of rust-options-pattern for this workspace:
//! `DEFAULT_*` consts are the single source of truth; `new()` is `const`
//! and returns the defaults; `Default` delegates to `new()`; every knob has
//! a projected `#[inline(always)]` accessor plus `with_*`/`set_*`, and
//! `Option`/`bool` knobs get the full `set_`/`with_`/`update_`/`maybe_`/
//! `clear_` vocabulary. `serde` (optional feature) never emits `null` and
//! every field falls back to its true default on a partial config —
//! including fields whose default isn't the field type's own `Default`
//! (`NonZeroUsize` has none at all; several thresholds default `Some(_)`;
//! `ComputeOptions`'s per-stage defaults aren't `ComputeUnits::default()`),
//! which is exactly why those fields use `serde(default = "fn")` rather
//! than the bare form.

use std::{
  num::NonZeroUsize,
  path::{Path, PathBuf},
};

use coremlit::ComputeUnits;

use crate::constants::MAX_TOKEN_CONTEXT;

// ---------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------

/// Which Whisper decode task to run (Swift `DecodingTask`).
#[derive(
  Debug, Default, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Task {
  /// Transcribe speech in its spoken language.
  #[default]
  Transcribe,
  /// Translate speech to English.
  Translate,
}

impl Task {
  /// Stable snake_case name of the variant.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Transcribe => "transcribe",
      Self::Translate => "translate",
    }
  }
}

/// Error parsing a [`Task`] name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown task name")]
pub struct ParseTaskError(());

impl core::str::FromStr for Task {
  type Err = ParseTaskError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "transcribe" => Self::Transcribe,
      "translate" => Self::Translate,
      _ => return Err(ParseTaskError(())),
    })
  }
}

// ---------------------------------------------------------------------
// ChunkingStrategy
// ---------------------------------------------------------------------

/// How long-form audio is split into chunks before transcription (Swift
/// `ChunkingStrategy`). Swift's `.none` case is renamed `Disabled` here — a
/// variant literally named `None` reads badly next to `Option::None` — but
/// keeps the wire name `"none"` for parity (spec §6.1).
#[derive(
  Debug, Default, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ChunkingStrategy {
  /// No chunking: the whole input is windowed and decoded sequentially.
  #[default]
  #[cfg_attr(feature = "serde", serde(rename = "none"))]
  Disabled,
  /// Split at VAD (voice activity detection) silence boundaries.
  Vad,
}

impl ChunkingStrategy {
  /// Stable wire name of the variant (`"none"`/`"vad"`).
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Disabled => "none",
      Self::Vad => "vad",
    }
  }
}

/// Error parsing a [`ChunkingStrategy`] name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown chunking strategy name")]
pub struct ParseChunkingStrategyError(());

impl core::str::FromStr for ChunkingStrategy {
  type Err = ParseChunkingStrategyError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "none" => Self::Disabled,
      "vad" => Self::Vad,
      _ => return Err(ParseChunkingStrategyError(())),
    })
  }
}

// ---------------------------------------------------------------------
// DecodingOptions
// ---------------------------------------------------------------------

/// Default [`DecodingOptions::temperature`] (greedy/argmax decoding).
pub const DEFAULT_TEMPERATURE: f32 = 0.0;
/// Default [`DecodingOptions::temperature_increment_on_fallback`].
pub const DEFAULT_TEMPERATURE_INCREMENT_ON_FALLBACK: f32 = 0.2;
/// Default [`DecodingOptions::temperature_fallback_count`].
pub const DEFAULT_TEMPERATURE_FALLBACK_COUNT: usize = 5;
/// Default [`DecodingOptions::sample_length`] (Whisper's `448 / 2` token context).
pub const DEFAULT_SAMPLE_LENGTH: usize = MAX_TOKEN_CONTEXT;
/// Default [`DecodingOptions::top_k`].
pub const DEFAULT_TOP_K: usize = 5;
/// Default [`DecodingOptions::window_clip_time`], in seconds.
pub const DEFAULT_WINDOW_CLIP_TIME: f32 = 1.0;
/// Default [`DecodingOptions::compression_ratio_threshold`].
pub const DEFAULT_COMPRESSION_RATIO_THRESHOLD: f32 = 2.4;
/// Default [`DecodingOptions::logprob_threshold`].
pub const DEFAULT_LOGPROB_THRESHOLD: f32 = -1.0;
/// Default [`DecodingOptions::first_token_logprob_threshold`].
pub const DEFAULT_FIRST_TOKEN_LOGPROB_THRESHOLD: f32 = -1.5;
/// Default [`DecodingOptions::no_speech_threshold`].
pub const DEFAULT_NO_SPEECH_THRESHOLD: f32 = 0.6;
/// Default [`DecodingOptions::use_prefill_prompt`].
pub const DEFAULT_USE_PREFILL_PROMPT: bool = true;
/// Default [`DecodingOptions::concurrent_worker_count`] (Swift's macOS default).
pub const DEFAULT_CONCURRENT_WORKER_COUNT: NonZeroUsize = NonZeroUsize::new(16).unwrap();

#[cfg(feature = "serde")]
fn default_temperature_increment_on_fallback() -> f32 {
  DEFAULT_TEMPERATURE_INCREMENT_ON_FALLBACK
}
#[cfg(feature = "serde")]
fn default_temperature_fallback_count() -> usize {
  DEFAULT_TEMPERATURE_FALLBACK_COUNT
}
#[cfg(feature = "serde")]
fn default_sample_length() -> usize {
  DEFAULT_SAMPLE_LENGTH
}
#[cfg(feature = "serde")]
fn default_top_k() -> usize {
  DEFAULT_TOP_K
}
#[cfg(feature = "serde")]
fn default_window_clip_time() -> f32 {
  DEFAULT_WINDOW_CLIP_TIME
}
#[cfg(feature = "serde")]
fn default_concurrent_worker_count() -> NonZeroUsize {
  DEFAULT_CONCURRENT_WORKER_COUNT
}
#[cfg(feature = "serde")]
fn default_compression_ratio_threshold() -> Option<f32> {
  Some(DEFAULT_COMPRESSION_RATIO_THRESHOLD)
}
#[cfg(feature = "serde")]
fn default_logprob_threshold() -> Option<f32> {
  Some(DEFAULT_LOGPROB_THRESHOLD)
}
#[cfg(feature = "serde")]
fn default_first_token_logprob_threshold() -> Option<f32> {
  Some(DEFAULT_FIRST_TOKEN_LOGPROB_THRESHOLD)
}
#[cfg(feature = "serde")]
fn default_no_speech_threshold() -> Option<f32> {
  Some(DEFAULT_NO_SPEECH_THRESHOLD)
}
// `bool::default()` is `false`; `use_prefill_prompt` defaults `true`
// (Swift `usePrefillPrompt = true`), so it needs a fn-default too.
#[cfg(feature = "serde")]
fn default_use_prefill_prompt() -> bool {
  DEFAULT_USE_PREFILL_PROMPT
}

/// Decode-time configuration: Swift's 27-knob `DecodingOptions` surface
/// (spec §6.2), plus one Rust-only addition — [`Self::seed`], for
/// reproducible temperature-fallback sampling (coremlit issue #9; see the
/// crate root's "Reproducibility and provenance" docs). `new()`/`Default`
/// apply Swift's defaults verbatim for every ported knob; `seed` defaults
/// unset (`None`), matching today's OS-seeded behavior exactly.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecodingOptions {
  /// Which decode task to run.
  #[cfg_attr(feature = "serde", serde(default))]
  task: Task,
  /// Spoken language, as an ISO code (see [`crate::constants::languages`]).
  /// Empty means auto-detect (golden empty-means-absent).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "String::is_empty")
  )]
  language: String,
  /// Sampling temperature; `0.0` is greedy (argmax) decoding.
  #[cfg_attr(feature = "serde", serde(default))]
  temperature: f32,
  /// Amount added to `temperature` on each fallback retry.
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_temperature_increment_on_fallback")
  )]
  temperature_increment_on_fallback: f32,
  /// Maximum number of temperature-fallback retries.
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_temperature_fallback_count")
  )]
  temperature_fallback_count: usize,
  /// Maximum number of tokens to sample per window.
  #[cfg_attr(feature = "serde", serde(default = "default_sample_length"))]
  sample_length: usize,
  /// Candidate count for non-zero-temperature sampling.
  #[cfg_attr(feature = "serde", serde(default = "default_top_k"))]
  top_k: usize,
  /// Base seed for reproducible temperature-fallback sampling. `None`
  /// (the default) leaves sampling OS-seeded. See [`Self::seed`] for the
  /// full contract.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  seed: Option<u64>,
  /// Force the prefill tokens from `task`/`language` rather than let the
  /// model choose them.
  #[cfg_attr(feature = "serde", serde(default = "default_use_prefill_prompt"))]
  use_prefill_prompt: bool,
  /// Detect the spoken language instead of using `language`. Tri-state:
  /// `None` means the caller never chose, and the value resolves against
  /// `use_prefill_prompt` at read time — see [`Self::detect_language`]
  /// (the getter is the single resolution point) for the Swift rule this
  /// ports (`Configurations.swift:222`).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  detect_language: Option<bool>,
  /// Omit special tokens (e.g. `<|endoftext|>`) from decoded text.
  #[cfg_attr(feature = "serde", serde(default))]
  skip_special_tokens: bool,
  /// Omit timestamp tokens from decoding entirely.
  #[cfg_attr(feature = "serde", serde(default))]
  without_timestamps: bool,
  /// Compute word-level timestamps via DTW alignment.
  #[cfg_attr(feature = "serde", serde(default))]
  word_timestamps: bool,
  /// Reject an initial timestamp token above this many seconds into the
  /// window. `None` disables the check (golden `Option<Copy>` exception).
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  max_initial_timestamp: Option<f32>,
  /// Cap the seek position, in samples, for any single window. `None`
  /// disables the cap.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  max_window_seek: Option<usize>,
  /// Explicit `(start, end)`-pair timestamps, in seconds, to split the
  /// audio into segments before transcription. Empty means the whole input
  /// is one segment.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  clip_timestamps: Vec<f32>,
  /// Seconds clipped from the end of each window, to reduce hallucinated
  /// trailing text.
  #[cfg_attr(feature = "serde", serde(default = "default_window_clip_time"))]
  window_clip_time: f32,
  /// Token ids prepended to the prefill tokens as a conditioning prompt.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  prompt_tokens: Vec<u32>,
  /// Token ids appended to the prefill tokens as a forced prefix.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  prefix_tokens: Vec<u32>,
  /// Suppress the blank (space) token during decoding.
  #[cfg_attr(feature = "serde", serde(default))]
  suppress_blank: bool,
  /// Additional token ids to suppress during decoding.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Vec::is_empty")
  )]
  suppress_tokens: Vec<u32>,
  /// Treat decoding as failed if the output text's compression ratio
  /// exceeds this value (too repetitive). `None` disables the check.
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_compression_ratio_threshold",
      skip_serializing_if = "Option::is_none"
    )
  )]
  compression_ratio_threshold: Option<f32>,
  /// Treat decoding as failed if the average sampled-token log
  /// probability falls below this value. `None` disables the check.
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_logprob_threshold",
      skip_serializing_if = "Option::is_none"
    )
  )]
  logprob_threshold: Option<f32>,
  /// Treat decoding as failed if the first sampled token's log
  /// probability falls below this value. `None` disables the check.
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_first_token_logprob_threshold",
      skip_serializing_if = "Option::is_none"
    )
  )]
  first_token_logprob_threshold: Option<f32>,
  /// Treat a window as silent when the no-speech probability strictly
  /// exceeds this value. `None` disables the check. (Silence short-circuits
  /// on this comparison ALONE — Swift's own doc comment claiming the
  /// average log probability is also consulted is stale against its code;
  /// see `result::needs_fallback`, `Models.swift:368-370`.)
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_no_speech_threshold",
      skip_serializing_if = "Option::is_none"
    )
  )]
  no_speech_threshold: Option<f32>,
  /// Worker threads for batch transcription (Swift's macOS default: 16).
  #[cfg_attr(feature = "serde", serde(default = "default_concurrent_worker_count"))]
  concurrent_worker_count: NonZeroUsize,
  /// How long-form audio is split into chunks before transcription.
  #[cfg_attr(feature = "serde", serde(default))]
  chunking_strategy: ChunkingStrategy,
  /// Emit verbose per-step decode logging.
  #[cfg_attr(feature = "serde", serde(default))]
  verbose: bool,
}

impl Default for DecodingOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl DecodingOptions {
  /// Decode options matching Swift's defaults (spec §6.2).
  pub const fn new() -> Self {
    Self {
      task: Task::Transcribe,
      language: String::new(),
      temperature: DEFAULT_TEMPERATURE,
      temperature_increment_on_fallback: DEFAULT_TEMPERATURE_INCREMENT_ON_FALLBACK,
      temperature_fallback_count: DEFAULT_TEMPERATURE_FALLBACK_COUNT,
      sample_length: DEFAULT_SAMPLE_LENGTH,
      top_k: DEFAULT_TOP_K,
      seed: None,
      use_prefill_prompt: DEFAULT_USE_PREFILL_PROMPT,
      detect_language: None,
      skip_special_tokens: false,
      without_timestamps: false,
      word_timestamps: false,
      max_initial_timestamp: None,
      max_window_seek: None,
      clip_timestamps: Vec::new(),
      window_clip_time: DEFAULT_WINDOW_CLIP_TIME,
      prompt_tokens: Vec::new(),
      prefix_tokens: Vec::new(),
      suppress_blank: false,
      suppress_tokens: Vec::new(),
      compression_ratio_threshold: Some(DEFAULT_COMPRESSION_RATIO_THRESHOLD),
      logprob_threshold: Some(DEFAULT_LOGPROB_THRESHOLD),
      first_token_logprob_threshold: Some(DEFAULT_FIRST_TOKEN_LOGPROB_THRESHOLD),
      no_speech_threshold: Some(DEFAULT_NO_SPEECH_THRESHOLD),
      concurrent_worker_count: DEFAULT_CONCURRENT_WORKER_COUNT,
      chunking_strategy: ChunkingStrategy::Disabled,
      verbose: false,
    }
  }

  // -- task ---------------------------------------------------------
  /// The configured decode task.
  #[inline(always)]
  pub const fn task(&self) -> Task {
    self.task
  }
  /// Builder form of [`Self::set_task`].
  #[must_use]
  #[inline(always)]
  pub const fn with_task(mut self, task: Task) -> Self {
    self.set_task(task);
    self
  }
  /// Sets [`Self::task`] in place.
  #[inline(always)]
  pub const fn set_task(&mut self, task: Task) -> &mut Self {
    self.task = task;
    self
  }

  // -- language -------------------------------------------------------
  /// Spoken language (ISO code); empty means auto-detect.
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

  // -- temperature ------------------------------------------------------
  /// Sampling temperature; `0.0` is greedy (argmax) decoding.
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

  // -- temperature_increment_on_fallback ---------------------------------
  /// Amount added to `temperature` on each fallback retry.
  #[inline(always)]
  pub const fn temperature_increment_on_fallback(&self) -> f32 {
    self.temperature_increment_on_fallback
  }
  /// Builder form of [`Self::set_temperature_increment_on_fallback`].
  #[must_use]
  #[inline(always)]
  pub const fn with_temperature_increment_on_fallback(
    mut self,
    temperature_increment_on_fallback: f32,
  ) -> Self {
    self.set_temperature_increment_on_fallback(temperature_increment_on_fallback);
    self
  }
  /// Sets [`Self::temperature_increment_on_fallback`] in place.
  #[inline(always)]
  pub const fn set_temperature_increment_on_fallback(
    &mut self,
    temperature_increment_on_fallback: f32,
  ) -> &mut Self {
    self.temperature_increment_on_fallback = temperature_increment_on_fallback;
    self
  }

  // -- temperature_fallback_count -----------------------------------------
  /// Maximum number of temperature-fallback retries.
  #[inline(always)]
  pub const fn temperature_fallback_count(&self) -> usize {
    self.temperature_fallback_count
  }
  /// Builder form of [`Self::set_temperature_fallback_count`].
  #[must_use]
  #[inline(always)]
  pub const fn with_temperature_fallback_count(
    mut self,
    temperature_fallback_count: usize,
  ) -> Self {
    self.set_temperature_fallback_count(temperature_fallback_count);
    self
  }
  /// Sets [`Self::temperature_fallback_count`] in place.
  #[inline(always)]
  pub const fn set_temperature_fallback_count(
    &mut self,
    temperature_fallback_count: usize,
  ) -> &mut Self {
    self.temperature_fallback_count = temperature_fallback_count;
    self
  }

  // -- sample_length --------------------------------------------------
  /// Maximum number of tokens to sample per window.
  #[inline(always)]
  pub const fn sample_length(&self) -> usize {
    self.sample_length
  }
  /// Builder form of [`Self::set_sample_length`].
  #[must_use]
  #[inline(always)]
  pub const fn with_sample_length(mut self, sample_length: usize) -> Self {
    self.set_sample_length(sample_length);
    self
  }
  /// Sets [`Self::sample_length`] in place.
  #[inline(always)]
  pub const fn set_sample_length(&mut self, sample_length: usize) -> &mut Self {
    self.sample_length = sample_length;
    self
  }

  // -- top_k ------------------------------------------------------------
  /// Candidate count for non-zero-temperature sampling.
  #[inline(always)]
  pub const fn top_k(&self) -> usize {
    self.top_k
  }
  /// Builder form of [`Self::set_top_k`].
  #[must_use]
  #[inline(always)]
  pub const fn with_top_k(mut self, top_k: usize) -> Self {
    self.set_top_k(top_k);
    self
  }
  /// Sets [`Self::top_k`] in place.
  #[inline(always)]
  pub const fn set_top_k(&mut self, top_k: usize) -> &mut Self {
    self.top_k = top_k;
    self
  }

  // -- seed (Option<u64>) ---------------------------------------------------
  /// Base seed for reproducible temperature-fallback sampling.
  ///
  /// `None` (the default) leaves every attempt's
  /// [`GreedyTokenSampler`](crate::decode::sampler::GreedyTokenSampler)
  /// OS-seeded, matching Swift's own unseeded `Float.random`
  /// (`TokenSampler.swift:169`) — nondeterministic at `temperature > 0`,
  /// by design, on both runtimes; this is the byte-unchanged default
  /// path.
  ///
  /// `Some(seed)` makes the whole transcription reproducible instead:
  /// [`crate::transcribe::TranscribeTask`]'s fallback ladder derives an
  /// independent per-(worker, window, attempt) sub-seed from it via
  /// [`derive_attempt_seed`](crate::decode::sampler::derive_attempt_seed)
  /// rather than reusing `seed` verbatim everywhere (see that function's
  /// doc for why, and for the exact mixing function) — so re-running the
  /// same audio through the same options and `seed` always samples the
  /// identical tokens, with the fallback ladder still fully enabled. A
  /// seed makes this port's *own* output reproducible; it cannot make
  /// that output match Swift's, which has no seed knob of its own and
  /// always draws unseeded — record the effective temperature in
  /// provenance either way
  /// ([`TranscriptionSegment::temperature`](crate::result::TranscriptionSegment::temperature)).
  #[inline(always)]
  pub const fn seed(&self) -> Option<u64> {
    self.seed
  }
  /// Builder form of [`Self::set_seed`].
  #[must_use]
  #[inline(always)]
  pub const fn with_seed(mut self, seed: u64) -> Self {
    self.set_seed(seed);
    self
  }
  /// Sets [`Self::seed`] to `Some(seed)`.
  #[inline(always)]
  pub const fn set_seed(&mut self, seed: u64) -> &mut Self {
    self.seed = Some(seed);
    self
  }
  /// Builder form of [`Self::update_seed`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_seed(mut self, seed: Option<u64>) -> Self {
    self.update_seed(seed);
    self
  }
  /// Assigns [`Self::seed`] directly.
  #[inline(always)]
  pub const fn update_seed(&mut self, seed: Option<u64>) -> &mut Self {
    self.seed = seed;
    self
  }
  /// Sets [`Self::seed`] to `None`.
  #[inline(always)]
  pub const fn clear_seed(&mut self) -> &mut Self {
    self.seed = None;
    self
  }

  // -- use_prefill_prompt (bool) ------------------------------------------
  /// Force the prefill tokens from `task`/`language` rather than let the
  /// model choose them.
  #[inline(always)]
  pub const fn use_prefill_prompt(&self) -> bool {
    self.use_prefill_prompt
  }
  /// Builder form of [`Self::set_use_prefill_prompt`].
  #[must_use]
  #[inline(always)]
  pub const fn with_use_prefill_prompt(mut self) -> Self {
    self.set_use_prefill_prompt();
    self
  }
  /// Sets [`Self::use_prefill_prompt`] to `true`.
  #[inline(always)]
  pub const fn set_use_prefill_prompt(&mut self) -> &mut Self {
    self.use_prefill_prompt = true;
    self
  }
  /// Builder form of [`Self::update_use_prefill_prompt`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_use_prefill_prompt(mut self, use_prefill_prompt: bool) -> Self {
    self.update_use_prefill_prompt(use_prefill_prompt);
    self
  }
  /// Assigns [`Self::use_prefill_prompt`] directly.
  #[inline(always)]
  pub const fn update_use_prefill_prompt(&mut self, use_prefill_prompt: bool) -> &mut Self {
    self.use_prefill_prompt = use_prefill_prompt;
    self
  }
  /// Sets [`Self::use_prefill_prompt`] to `false`.
  #[inline(always)]
  pub const fn clear_use_prefill_prompt(&mut self) -> &mut Self {
    self.use_prefill_prompt = false;
    self
  }

  // -- detect_language (tri-state bool) ------------------------------------
  /// Detect the spoken language instead of using `language`, resolved:
  /// when the caller never set it, it defaults to `!use_prefill_prompt` —
  /// detection kicks in by default exactly when prefill is off. Ports
  /// Swift's constructor resolution `detectLanguage ?? !usePrefillPrompt`
  /// (`Configurations.swift:222`, "If prefill is false, detect language
  /// by default"). An explicit [`Self::set_detect_language`]/
  /// [`Self::clear_detect_language`]/[`Self::update_detect_language`]
  /// always wins over the coupling, in either direction, regardless of
  /// mutation order.
  ///
  /// **Construction is Swift-identical.** Build with the chained
  /// `with_*`/`maybe_*` form and this getter returns exactly what
  /// Swift's memberwise initializer would have stored for the same
  /// `(detect_language, use_prefill_prompt)` pair: e.g.
  /// `DecodingOptions::new().maybe_use_prefill_prompt(false)` resolves
  /// `true` here, same as Swift's `DecodingOptions(usePrefillPrompt:
  /// false)`, because both sides apply the identical `?? !usePrefillPrompt`
  /// formula to the same final inputs.
  ///
  /// **Pinned deviation: in-place mutation of `use_prefill_prompt` after
  /// construction, while `detect_language` is still unset.** Swift's
  /// `detectLanguage` is a plain stored `Bool` — resolved once inside
  /// `init`, then frozen; reassigning the `var usePrefillPrompt`
  /// property afterward never touches it again. This getter has no such
  /// freeze point and re-resolves on every call against whatever
  /// `use_prefill_prompt` currently holds. Concretely:
  /// `DecodingOptions::new()` (prefill ON, the default) followed by the
  /// in-place mutator [`Self::clear_use_prefill_prompt`] resolves `true`
  /// here — but Swift's equivalent, `var o = DecodingOptions()`
  /// (`detectLanguage` already frozen `false`) followed by
  /// `o.usePrefillPrompt = false`, leaves `detectLanguage` at the
  /// `false` its initializer committed to. At a nonzero temperature
  /// this changes whether a language-detection probe runs at all, which
  /// consumes the attempt's sampler RNG draw and can shift both the
  /// sampled tokens that follow and the word-level language split.
  ///
  /// This is a deliberate, accepted deviation, not a defect to fix:
  /// eagerly freezing the resolution — inside `new()` or every mutator
  /// that touches `use_prefill_prompt` — would have to materialize
  /// `Some(false)` immediately, destroying the `None` ("caller never
  /// chose") tri-state this field exists to represent, and with it
  /// `serde`'s absent-on-unset round trip
  /// (`skip_serializing_if = "Option::is_none"`) and the explicit-wins
  /// guarantee above — both depend on telling "never set" apart from
  /// "set to whatever the coupling currently says." A caller who wants
  /// Swift's frozen-at-construction behavior after mutating
  /// `use_prefill_prompt` in place sets `detect_language` explicitly
  /// first (one call to [`Self::set_detect_language`]/
  /// [`Self::clear_detect_language`]/[`Self::update_detect_language`]
  /// before the mutation) — an explicit choice always wins over the
  /// coupling.
  ///
  /// This getter is the single resolution point: every pipeline consumer
  /// reads the coupled value through it.
  #[inline(always)]
  pub const fn detect_language(&self) -> bool {
    match self.detect_language {
      Some(explicit) => explicit,
      None => !self.use_prefill_prompt,
    }
  }
  /// Builder form of [`Self::set_detect_language`].
  #[must_use]
  #[inline(always)]
  pub const fn with_detect_language(mut self) -> Self {
    self.set_detect_language();
    self
  }
  /// Sets [`Self::detect_language`] explicitly to `true` (an explicit
  /// value always beats the `!use_prefill_prompt` default coupling).
  #[inline(always)]
  pub const fn set_detect_language(&mut self) -> &mut Self {
    self.detect_language = Some(true);
    self
  }
  /// Builder form of [`Self::update_detect_language`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_detect_language(mut self, detect_language: bool) -> Self {
    self.update_detect_language(detect_language);
    self
  }
  /// Assigns [`Self::detect_language`] explicitly (an explicit value
  /// always beats the `!use_prefill_prompt` default coupling).
  #[inline(always)]
  pub const fn update_detect_language(&mut self, detect_language: bool) -> &mut Self {
    self.detect_language = Some(detect_language);
    self
  }
  /// Sets [`Self::detect_language`] explicitly to `false` (an explicit
  /// value always beats the `!use_prefill_prompt` default coupling — this
  /// is how a no-prefill caller opts back out of detection).
  #[inline(always)]
  pub const fn clear_detect_language(&mut self) -> &mut Self {
    self.detect_language = Some(false);
    self
  }

  // -- skip_special_tokens (bool) -----------------------------------------
  /// Omit special tokens (e.g. `<|endoftext|>`) from decoded text.
  #[inline(always)]
  pub const fn skip_special_tokens(&self) -> bool {
    self.skip_special_tokens
  }
  /// Builder form of [`Self::set_skip_special_tokens`].
  #[must_use]
  #[inline(always)]
  pub const fn with_skip_special_tokens(mut self) -> Self {
    self.set_skip_special_tokens();
    self
  }
  /// Sets [`Self::skip_special_tokens`] to `true`.
  #[inline(always)]
  pub const fn set_skip_special_tokens(&mut self) -> &mut Self {
    self.skip_special_tokens = true;
    self
  }
  /// Builder form of [`Self::update_skip_special_tokens`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_skip_special_tokens(mut self, skip_special_tokens: bool) -> Self {
    self.update_skip_special_tokens(skip_special_tokens);
    self
  }
  /// Assigns [`Self::skip_special_tokens`] directly.
  #[inline(always)]
  pub const fn update_skip_special_tokens(&mut self, skip_special_tokens: bool) -> &mut Self {
    self.skip_special_tokens = skip_special_tokens;
    self
  }
  /// Sets [`Self::skip_special_tokens`] to `false`.
  #[inline(always)]
  pub const fn clear_skip_special_tokens(&mut self) -> &mut Self {
    self.skip_special_tokens = false;
    self
  }

  // -- without_timestamps (bool) ------------------------------------------
  /// Omit timestamp tokens from decoding entirely.
  #[inline(always)]
  pub const fn without_timestamps(&self) -> bool {
    self.without_timestamps
  }
  /// Builder form of [`Self::set_without_timestamps`].
  #[must_use]
  #[inline(always)]
  pub const fn with_without_timestamps(mut self) -> Self {
    self.set_without_timestamps();
    self
  }
  /// Sets [`Self::without_timestamps`] to `true`.
  #[inline(always)]
  pub const fn set_without_timestamps(&mut self) -> &mut Self {
    self.without_timestamps = true;
    self
  }
  /// Builder form of [`Self::update_without_timestamps`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_without_timestamps(mut self, without_timestamps: bool) -> Self {
    self.update_without_timestamps(without_timestamps);
    self
  }
  /// Assigns [`Self::without_timestamps`] directly.
  #[inline(always)]
  pub const fn update_without_timestamps(&mut self, without_timestamps: bool) -> &mut Self {
    self.without_timestamps = without_timestamps;
    self
  }
  /// Sets [`Self::without_timestamps`] to `false`.
  #[inline(always)]
  pub const fn clear_without_timestamps(&mut self) -> &mut Self {
    self.without_timestamps = false;
    self
  }

  // -- word_timestamps (bool) ----------------------------------------------
  /// Compute word-level timestamps via DTW alignment.
  ///
  /// When set, [`crate::transcribe::TranscribeTask::run`]'s window loop
  /// runs [`crate::segment::add_word_timestamps`] against each window's
  /// alignment-weight snapshot and writes the result onto that window's
  /// segments (Swift's `addWordTimestamps`, `TranscribeTask.swift:
  /// 196-233). `false` (the default) leaves every segment's `words` empty
  /// and skips that (relatively expensive) DTW alignment pass entirely.
  #[inline(always)]
  pub const fn word_timestamps(&self) -> bool {
    self.word_timestamps
  }
  /// Builder form of [`Self::set_word_timestamps`].
  #[must_use]
  #[inline(always)]
  pub const fn with_word_timestamps(mut self) -> Self {
    self.set_word_timestamps();
    self
  }
  /// Sets [`Self::word_timestamps`] to `true`.
  #[inline(always)]
  pub const fn set_word_timestamps(&mut self) -> &mut Self {
    self.word_timestamps = true;
    self
  }
  /// Builder form of [`Self::update_word_timestamps`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_word_timestamps(mut self, word_timestamps: bool) -> Self {
    self.update_word_timestamps(word_timestamps);
    self
  }
  /// Assigns [`Self::word_timestamps`] directly.
  #[inline(always)]
  pub const fn update_word_timestamps(&mut self, word_timestamps: bool) -> &mut Self {
    self.word_timestamps = word_timestamps;
    self
  }
  /// Sets [`Self::word_timestamps`] to `false`.
  #[inline(always)]
  pub const fn clear_word_timestamps(&mut self) -> &mut Self {
    self.word_timestamps = false;
    self
  }

  // -- max_initial_timestamp (Option<f32>) --------------------------------
  /// Reject an initial timestamp token above this many seconds into the
  /// window. `None` disables the check.
  #[inline(always)]
  pub const fn max_initial_timestamp(&self) -> Option<f32> {
    self.max_initial_timestamp
  }
  /// Builder form of [`Self::set_max_initial_timestamp`].
  #[must_use]
  #[inline(always)]
  pub const fn with_max_initial_timestamp(mut self, max_initial_timestamp: f32) -> Self {
    self.set_max_initial_timestamp(max_initial_timestamp);
    self
  }
  /// Sets [`Self::max_initial_timestamp`] to `Some(max_initial_timestamp)`.
  #[inline(always)]
  pub const fn set_max_initial_timestamp(&mut self, max_initial_timestamp: f32) -> &mut Self {
    self.max_initial_timestamp = Some(max_initial_timestamp);
    self
  }
  /// Builder form of [`Self::update_max_initial_timestamp`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_max_initial_timestamp(mut self, max_initial_timestamp: Option<f32>) -> Self {
    self.update_max_initial_timestamp(max_initial_timestamp);
    self
  }
  /// Assigns [`Self::max_initial_timestamp`] directly.
  #[inline(always)]
  pub const fn update_max_initial_timestamp(
    &mut self,
    max_initial_timestamp: Option<f32>,
  ) -> &mut Self {
    self.max_initial_timestamp = max_initial_timestamp;
    self
  }
  /// Sets [`Self::max_initial_timestamp`] to `None`.
  #[inline(always)]
  pub const fn clear_max_initial_timestamp(&mut self) -> &mut Self {
    self.max_initial_timestamp = None;
    self
  }

  // -- max_window_seek (Option<usize>) -------------------------------------
  /// Cap the seek position, in samples, for any single window. `None`
  /// disables the cap. The pipeline floors a configured cap at one
  /// sample of forward progress per window — `Some(0)` would otherwise
  /// pin the seek loop to the same window forever.
  #[inline(always)]
  pub const fn max_window_seek(&self) -> Option<usize> {
    self.max_window_seek
  }
  /// Builder form of [`Self::set_max_window_seek`].
  #[must_use]
  #[inline(always)]
  pub const fn with_max_window_seek(mut self, max_window_seek: usize) -> Self {
    self.set_max_window_seek(max_window_seek);
    self
  }
  /// Sets [`Self::max_window_seek`] to `Some(max_window_seek)`.
  #[inline(always)]
  pub const fn set_max_window_seek(&mut self, max_window_seek: usize) -> &mut Self {
    self.max_window_seek = Some(max_window_seek);
    self
  }
  /// Builder form of [`Self::update_max_window_seek`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_max_window_seek(mut self, max_window_seek: Option<usize>) -> Self {
    self.update_max_window_seek(max_window_seek);
    self
  }
  /// Assigns [`Self::max_window_seek`] directly.
  #[inline(always)]
  pub const fn update_max_window_seek(&mut self, max_window_seek: Option<usize>) -> &mut Self {
    self.max_window_seek = max_window_seek;
    self
  }
  /// Sets [`Self::max_window_seek`] to `None`.
  #[inline(always)]
  pub const fn clear_max_window_seek(&mut self) -> &mut Self {
    self.max_window_seek = None;
    self
  }

  // -- clip_timestamps (Vec<f32>) -----------------------------------------
  /// Explicit `(start, end)`-pair timestamps, in seconds, to split the
  /// audio into segments before transcription. Empty means one segment.
  #[inline(always)]
  pub const fn clip_timestamps_slice(&self) -> &[f32] {
    self.clip_timestamps.as_slice()
  }
  /// Builder form of [`Self::set_clip_timestamps`].
  #[must_use]
  #[inline(always)]
  pub fn with_clip_timestamps(mut self, clip_timestamps: impl Into<Vec<f32>>) -> Self {
    self.set_clip_timestamps(clip_timestamps);
    self
  }
  /// Sets [`Self::clip_timestamps_slice`] in place.
  #[inline(always)]
  pub fn set_clip_timestamps(&mut self, clip_timestamps: impl Into<Vec<f32>>) -> &mut Self {
    self.clip_timestamps = clip_timestamps.into();
    self
  }

  // -- window_clip_time -----------------------------------------------
  /// Seconds clipped from the end of each window, to reduce hallucinated
  /// trailing text.
  #[inline(always)]
  pub const fn window_clip_time(&self) -> f32 {
    self.window_clip_time
  }
  /// Builder form of [`Self::set_window_clip_time`].
  #[must_use]
  #[inline(always)]
  pub const fn with_window_clip_time(mut self, window_clip_time: f32) -> Self {
    self.set_window_clip_time(window_clip_time);
    self
  }
  /// Sets [`Self::window_clip_time`] in place.
  #[inline(always)]
  pub const fn set_window_clip_time(&mut self, window_clip_time: f32) -> &mut Self {
    self.window_clip_time = window_clip_time;
    self
  }

  // -- prompt_tokens (Vec<u32>) ---------------------------------------
  /// Token ids prepended to the prefill tokens as a conditioning prompt.
  #[inline(always)]
  pub const fn prompt_tokens_slice(&self) -> &[u32] {
    self.prompt_tokens.as_slice()
  }
  /// Builder form of [`Self::set_prompt_tokens`].
  #[must_use]
  #[inline(always)]
  pub fn with_prompt_tokens(mut self, prompt_tokens: impl Into<Vec<u32>>) -> Self {
    self.set_prompt_tokens(prompt_tokens);
    self
  }
  /// Sets [`Self::prompt_tokens_slice`] in place.
  #[inline(always)]
  pub fn set_prompt_tokens(&mut self, prompt_tokens: impl Into<Vec<u32>>) -> &mut Self {
    self.prompt_tokens = prompt_tokens.into();
    self
  }

  // -- prefix_tokens (Vec<u32>) ---------------------------------------
  /// Token ids appended to the prefill tokens as a forced prefix.
  #[inline(always)]
  pub const fn prefix_tokens_slice(&self) -> &[u32] {
    self.prefix_tokens.as_slice()
  }
  /// Builder form of [`Self::set_prefix_tokens`].
  #[must_use]
  #[inline(always)]
  pub fn with_prefix_tokens(mut self, prefix_tokens: impl Into<Vec<u32>>) -> Self {
    self.set_prefix_tokens(prefix_tokens);
    self
  }
  /// Sets [`Self::prefix_tokens_slice`] in place.
  #[inline(always)]
  pub fn set_prefix_tokens(&mut self, prefix_tokens: impl Into<Vec<u32>>) -> &mut Self {
    self.prefix_tokens = prefix_tokens.into();
    self
  }

  // -- suppress_blank (bool) -----------------------------------------------
  /// Suppress the blank (space) token during decoding.
  #[inline(always)]
  pub const fn suppress_blank(&self) -> bool {
    self.suppress_blank
  }
  /// Builder form of [`Self::set_suppress_blank`].
  #[must_use]
  #[inline(always)]
  pub const fn with_suppress_blank(mut self) -> Self {
    self.set_suppress_blank();
    self
  }
  /// Sets [`Self::suppress_blank`] to `true`.
  #[inline(always)]
  pub const fn set_suppress_blank(&mut self) -> &mut Self {
    self.suppress_blank = true;
    self
  }
  /// Builder form of [`Self::update_suppress_blank`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_suppress_blank(mut self, suppress_blank: bool) -> Self {
    self.update_suppress_blank(suppress_blank);
    self
  }
  /// Assigns [`Self::suppress_blank`] directly.
  #[inline(always)]
  pub const fn update_suppress_blank(&mut self, suppress_blank: bool) -> &mut Self {
    self.suppress_blank = suppress_blank;
    self
  }
  /// Sets [`Self::suppress_blank`] to `false`.
  #[inline(always)]
  pub const fn clear_suppress_blank(&mut self) -> &mut Self {
    self.suppress_blank = false;
    self
  }

  // -- suppress_tokens (Vec<u32>) --------------------------------------
  /// Additional token ids to suppress during decoding.
  #[inline(always)]
  pub const fn suppress_tokens_slice(&self) -> &[u32] {
    self.suppress_tokens.as_slice()
  }
  /// Builder form of [`Self::set_suppress_tokens`].
  #[must_use]
  #[inline(always)]
  pub fn with_suppress_tokens(mut self, suppress_tokens: impl Into<Vec<u32>>) -> Self {
    self.set_suppress_tokens(suppress_tokens);
    self
  }
  /// Sets [`Self::suppress_tokens_slice`] in place.
  #[inline(always)]
  pub fn set_suppress_tokens(&mut self, suppress_tokens: impl Into<Vec<u32>>) -> &mut Self {
    self.suppress_tokens = suppress_tokens.into();
    self
  }

  // -- compression_ratio_threshold (Option<f32>) ---------------------------
  /// Treat decoding as failed if the output text's compression ratio
  /// exceeds this value (too repetitive). `None` disables the check.
  #[inline(always)]
  pub const fn compression_ratio_threshold(&self) -> Option<f32> {
    self.compression_ratio_threshold
  }
  /// Builder form of [`Self::set_compression_ratio_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compression_ratio_threshold(
    mut self,
    compression_ratio_threshold: f32,
  ) -> Self {
    self.set_compression_ratio_threshold(compression_ratio_threshold);
    self
  }
  /// Sets [`Self::compression_ratio_threshold`] to `Some(compression_ratio_threshold)`.
  #[inline(always)]
  pub const fn set_compression_ratio_threshold(
    &mut self,
    compression_ratio_threshold: f32,
  ) -> &mut Self {
    self.compression_ratio_threshold = Some(compression_ratio_threshold);
    self
  }
  /// Builder form of [`Self::update_compression_ratio_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_compression_ratio_threshold(
    mut self,
    compression_ratio_threshold: Option<f32>,
  ) -> Self {
    self.update_compression_ratio_threshold(compression_ratio_threshold);
    self
  }
  /// Assigns [`Self::compression_ratio_threshold`] directly.
  #[inline(always)]
  pub const fn update_compression_ratio_threshold(
    &mut self,
    compression_ratio_threshold: Option<f32>,
  ) -> &mut Self {
    self.compression_ratio_threshold = compression_ratio_threshold;
    self
  }
  /// Sets [`Self::compression_ratio_threshold`] to `None`.
  #[inline(always)]
  pub const fn clear_compression_ratio_threshold(&mut self) -> &mut Self {
    self.compression_ratio_threshold = None;
    self
  }

  // -- logprob_threshold (Option<f32>) -------------------------------------
  /// Treat decoding as failed if the average sampled-token log
  /// probability falls below this value. `None` disables the check.
  #[inline(always)]
  pub const fn logprob_threshold(&self) -> Option<f32> {
    self.logprob_threshold
  }
  /// Builder form of [`Self::set_logprob_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_logprob_threshold(mut self, logprob_threshold: f32) -> Self {
    self.set_logprob_threshold(logprob_threshold);
    self
  }
  /// Sets [`Self::logprob_threshold`] to `Some(logprob_threshold)`.
  #[inline(always)]
  pub const fn set_logprob_threshold(&mut self, logprob_threshold: f32) -> &mut Self {
    self.logprob_threshold = Some(logprob_threshold);
    self
  }
  /// Builder form of [`Self::update_logprob_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_logprob_threshold(mut self, logprob_threshold: Option<f32>) -> Self {
    self.update_logprob_threshold(logprob_threshold);
    self
  }
  /// Assigns [`Self::logprob_threshold`] directly.
  #[inline(always)]
  pub const fn update_logprob_threshold(&mut self, logprob_threshold: Option<f32>) -> &mut Self {
    self.logprob_threshold = logprob_threshold;
    self
  }
  /// Sets [`Self::logprob_threshold`] to `None`.
  #[inline(always)]
  pub const fn clear_logprob_threshold(&mut self) -> &mut Self {
    self.logprob_threshold = None;
    self
  }

  // -- first_token_logprob_threshold (Option<f32>) -------------------------
  /// Treat decoding as failed if the first sampled token's log
  /// probability falls below this value. `None` disables the check.
  #[inline(always)]
  pub const fn first_token_logprob_threshold(&self) -> Option<f32> {
    self.first_token_logprob_threshold
  }
  /// Builder form of [`Self::set_first_token_logprob_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_first_token_logprob_threshold(
    mut self,
    first_token_logprob_threshold: f32,
  ) -> Self {
    self.set_first_token_logprob_threshold(first_token_logprob_threshold);
    self
  }
  /// Sets [`Self::first_token_logprob_threshold`] to
  /// `Some(first_token_logprob_threshold)`.
  #[inline(always)]
  pub const fn set_first_token_logprob_threshold(
    &mut self,
    first_token_logprob_threshold: f32,
  ) -> &mut Self {
    self.first_token_logprob_threshold = Some(first_token_logprob_threshold);
    self
  }
  /// Builder form of [`Self::update_first_token_logprob_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_first_token_logprob_threshold(
    mut self,
    first_token_logprob_threshold: Option<f32>,
  ) -> Self {
    self.update_first_token_logprob_threshold(first_token_logprob_threshold);
    self
  }
  /// Assigns [`Self::first_token_logprob_threshold`] directly.
  #[inline(always)]
  pub const fn update_first_token_logprob_threshold(
    &mut self,
    first_token_logprob_threshold: Option<f32>,
  ) -> &mut Self {
    self.first_token_logprob_threshold = first_token_logprob_threshold;
    self
  }
  /// Sets [`Self::first_token_logprob_threshold`] to `None`.
  #[inline(always)]
  pub const fn clear_first_token_logprob_threshold(&mut self) -> &mut Self {
    self.first_token_logprob_threshold = None;
    self
  }

  // -- no_speech_threshold (Option<f32>) -----------------------------------
  /// Treat a window as silent when the no-speech probability strictly
  /// exceeds this value. `None` disables the check. (Silence short-circuits
  /// on this comparison ALONE — Swift's own doc comment claiming the
  /// average log probability is also consulted is stale against its code;
  /// see `result::needs_fallback`, `Models.swift:368-370`.)
  #[inline(always)]
  pub const fn no_speech_threshold(&self) -> Option<f32> {
    self.no_speech_threshold
  }
  /// Builder form of [`Self::set_no_speech_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn with_no_speech_threshold(mut self, no_speech_threshold: f32) -> Self {
    self.set_no_speech_threshold(no_speech_threshold);
    self
  }
  /// Sets [`Self::no_speech_threshold`] to `Some(no_speech_threshold)`.
  #[inline(always)]
  pub const fn set_no_speech_threshold(&mut self, no_speech_threshold: f32) -> &mut Self {
    self.no_speech_threshold = Some(no_speech_threshold);
    self
  }
  /// Builder form of [`Self::update_no_speech_threshold`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_no_speech_threshold(mut self, no_speech_threshold: Option<f32>) -> Self {
    self.update_no_speech_threshold(no_speech_threshold);
    self
  }
  /// Assigns [`Self::no_speech_threshold`] directly.
  #[inline(always)]
  pub const fn update_no_speech_threshold(
    &mut self,
    no_speech_threshold: Option<f32>,
  ) -> &mut Self {
    self.no_speech_threshold = no_speech_threshold;
    self
  }
  /// Sets [`Self::no_speech_threshold`] to `None`.
  #[inline(always)]
  pub const fn clear_no_speech_threshold(&mut self) -> &mut Self {
    self.no_speech_threshold = None;
    self
  }

  // -- concurrent_worker_count ------------------------------------------
  /// Worker threads for batch transcription (Swift's macOS default: 16).
  #[inline(always)]
  pub const fn concurrent_worker_count(&self) -> NonZeroUsize {
    self.concurrent_worker_count
  }
  /// Builder form of [`Self::set_concurrent_worker_count`].
  #[must_use]
  #[inline(always)]
  pub const fn with_concurrent_worker_count(
    mut self,
    concurrent_worker_count: NonZeroUsize,
  ) -> Self {
    self.set_concurrent_worker_count(concurrent_worker_count);
    self
  }
  /// Sets [`Self::concurrent_worker_count`] in place.
  #[inline(always)]
  pub const fn set_concurrent_worker_count(
    &mut self,
    concurrent_worker_count: NonZeroUsize,
  ) -> &mut Self {
    self.concurrent_worker_count = concurrent_worker_count;
    self
  }

  // -- chunking_strategy ------------------------------------------------
  /// How long-form audio is split into chunks before transcription.
  #[inline(always)]
  pub const fn chunking_strategy(&self) -> ChunkingStrategy {
    self.chunking_strategy
  }
  /// Builder form of [`Self::set_chunking_strategy`].
  #[must_use]
  #[inline(always)]
  pub const fn with_chunking_strategy(mut self, chunking_strategy: ChunkingStrategy) -> Self {
    self.set_chunking_strategy(chunking_strategy);
    self
  }
  /// Sets [`Self::chunking_strategy`] in place.
  #[inline(always)]
  pub const fn set_chunking_strategy(&mut self, chunking_strategy: ChunkingStrategy) -> &mut Self {
    self.chunking_strategy = chunking_strategy;
    self
  }

  // -- verbose (bool) -------------------------------------------------------
  /// Emit verbose per-step decode logging.
  #[inline(always)]
  pub const fn verbose(&self) -> bool {
    self.verbose
  }
  /// Builder form of [`Self::set_verbose`].
  #[must_use]
  #[inline(always)]
  pub const fn with_verbose(mut self) -> Self {
    self.set_verbose();
    self
  }
  /// Sets [`Self::verbose`] to `true`.
  #[inline(always)]
  pub const fn set_verbose(&mut self) -> &mut Self {
    self.verbose = true;
    self
  }
  /// Builder form of [`Self::update_verbose`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_verbose(mut self, verbose: bool) -> Self {
    self.update_verbose(verbose);
    self
  }
  /// Assigns [`Self::verbose`] directly.
  #[inline(always)]
  pub const fn update_verbose(&mut self, verbose: bool) -> &mut Self {
    self.verbose = verbose;
    self
  }
  /// Sets [`Self::verbose`] to `false`.
  #[inline(always)]
  pub const fn clear_verbose(&mut self) -> &mut Self {
    self.verbose = false;
    self
  }
}

// ---------------------------------------------------------------------
// ComputeOptions
// ---------------------------------------------------------------------

/// Default [`ComputeOptions::mel`] (Swift `ModelComputeOptions.melCompute`).
pub const DEFAULT_MEL_COMPUTE_UNITS: ComputeUnits = ComputeUnits::CpuAndGpu;
/// Default [`ComputeOptions::encoder`] (Swift
/// `ModelComputeOptions.audioEncoderCompute`, macOS 14+/iOS 17+ path).
pub const DEFAULT_ENCODER_COMPUTE_UNITS: ComputeUnits = ComputeUnits::CpuAndNeuralEngine;
/// Default [`ComputeOptions::decoder`] (Swift
/// `ModelComputeOptions.textDecoderCompute`).
pub const DEFAULT_DECODER_COMPUTE_UNITS: ComputeUnits = ComputeUnits::CpuAndNeuralEngine;

// `ComputeUnits::default()` is `All` (coremlit's own general-purpose
// default), which does NOT match WhisperKit's per-stage defaults above —
// so every field here needs a fn-default, never the bare form.
#[cfg(feature = "serde")]
fn default_mel_compute_units() -> ComputeUnits {
  DEFAULT_MEL_COMPUTE_UNITS
}
#[cfg(feature = "serde")]
fn default_encoder_compute_units() -> ComputeUnits {
  DEFAULT_ENCODER_COMPUTE_UNITS
}
#[cfg(feature = "serde")]
fn default_decoder_compute_units() -> ComputeUnits {
  DEFAULT_DECODER_COMPUTE_UNITS
}

// `coremlit` has no `serde` feature of its own (it depends on no
// serialization crate at all), so `ComputeUnits` isn't `Serialize`/
// `Deserialize`. Bridge it as a string through its existing `as_str`/
// `FromStr` — the same open-vocabulary shape rust-options-pattern uses for
// enum config fields the derive can't reach directly.
#[cfg(feature = "serde")]
mod compute_units_serde {
  use core::str::FromStr;

  use coremlit::ComputeUnits;
  use serde::{Deserialize, Deserializer, Serializer};

  pub(super) fn serialize<S: Serializer>(
    value: &ComputeUnits,
    serializer: S,
  ) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(value.as_str())
  }

  pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
  ) -> Result<ComputeUnits, D::Error> {
    let name = String::deserialize(deserializer)?;
    ComputeUnits::from_str(&name).map_err(serde::de::Error::custom)
  }
}

/// Per-stage CoreML compute-unit selection (Swift `ModelComputeOptions`).
/// Defaults mirror Swift: mel = CPU+GPU, encoder/decoder = CPU+ANE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeOptions {
  /// Compute units for the mel-spectrogram feature extractor.
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_mel_compute_units", with = "compute_units_serde")
  )]
  mel: ComputeUnits,
  /// Compute units for the audio encoder.
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_encoder_compute_units",
      with = "compute_units_serde"
    )
  )]
  encoder: ComputeUnits,
  /// Compute units for the text decoder.
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_decoder_compute_units",
      with = "compute_units_serde"
    )
  )]
  decoder: ComputeUnits,
}

impl Default for ComputeOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ComputeOptions {
  /// Compute options matching Swift's `ModelComputeOptions` defaults.
  pub const fn new() -> Self {
    Self {
      mel: DEFAULT_MEL_COMPUTE_UNITS,
      encoder: DEFAULT_ENCODER_COMPUTE_UNITS,
      decoder: DEFAULT_DECODER_COMPUTE_UNITS,
    }
  }

  /// Compute units for the mel-spectrogram feature extractor.
  #[inline(always)]
  pub const fn mel(&self) -> ComputeUnits {
    self.mel
  }
  /// Builder form of [`Self::set_mel`].
  #[must_use]
  #[inline(always)]
  pub const fn with_mel(mut self, mel: ComputeUnits) -> Self {
    self.set_mel(mel);
    self
  }
  /// Sets [`Self::mel`] in place.
  #[inline(always)]
  pub const fn set_mel(&mut self, mel: ComputeUnits) -> &mut Self {
    self.mel = mel;
    self
  }

  /// Compute units for the audio encoder.
  #[inline(always)]
  pub const fn encoder(&self) -> ComputeUnits {
    self.encoder
  }
  /// Builder form of [`Self::set_encoder`].
  #[must_use]
  #[inline(always)]
  pub const fn with_encoder(mut self, encoder: ComputeUnits) -> Self {
    self.set_encoder(encoder);
    self
  }
  /// Sets [`Self::encoder`] in place.
  #[inline(always)]
  pub const fn set_encoder(&mut self, encoder: ComputeUnits) -> &mut Self {
    self.encoder = encoder;
    self
  }

  /// Compute units for the text decoder.
  #[inline(always)]
  pub const fn decoder(&self) -> ComputeUnits {
    self.decoder
  }
  /// Builder form of [`Self::set_decoder`].
  #[must_use]
  #[inline(always)]
  pub const fn with_decoder(mut self, decoder: ComputeUnits) -> Self {
    self.set_decoder(decoder);
    self
  }
  /// Sets [`Self::decoder`] in place.
  #[inline(always)]
  pub const fn set_decoder(&mut self, decoder: ComputeUnits) -> &mut Self {
    self.decoder = decoder;
    self
  }
}

// ---------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------

/// Default [`Options::prewarm`]. Swift's `WhisperKitConfig.prewarm` is
/// `Bool? = nil`, resolved at call sites by `if let prewarm = config.prewarm,
/// prewarm` — nil skips prewarming — so that resolved value is `false`.
pub const DEFAULT_PREWARM: bool = false;
/// Default [`Options::load`]. Swift's `WhisperKitConfig.load` is
/// `Bool? = nil`, resolved by `config.load ?? (config.modelFolder != nil)`.
/// [`Options::new`] always requires a model folder, so that resolves to
/// `true`.
pub const DEFAULT_LOAD: bool = true;

#[cfg(feature = "serde")]
fn default_prewarm() -> bool {
  DEFAULT_PREWARM
}
// `bool::default()` is `false`, which does not match this field's `true`
// default — see `DEFAULT_LOAD`'s doc for the Swift resolution this ports.
#[cfg(feature = "serde")]
fn default_load() -> bool {
  DEFAULT_LOAD
}

/// Construction config for a WhisperKit pipeline: where to load the model
/// and tokenizer from, per-stage compute units, and load-time lifecycle
/// flags (Swift `WhisperKitConfig`, spec §5.3). Model auto-download is
/// deferred (spec §4.7); folders are always local.
///
/// No [`Default`]/zero-arg `new()`: there is no honest default model or
/// tokenizer folder, so construction always requires both (golden §1 — no
/// honest default means no `Default`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Options {
  /// Local folder containing the compiled (`.mlmodelc`) Whisper models.
  model_folder: PathBuf,
  /// Local folder containing the BPE tokenizer files.
  tokenizer_folder: PathBuf,
  /// Per-stage CoreML compute-unit selection.
  #[cfg_attr(feature = "serde", serde(default))]
  compute: ComputeOptions,
  /// Sequentially load-then-unload each model once before real use, to
  /// force ANE specialization up front instead of at first inference.
  #[cfg_attr(feature = "serde", serde(default = "default_prewarm"))]
  prewarm: bool,
  /// Load the models at construction time.
  #[cfg_attr(feature = "serde", serde(default = "default_load"))]
  load: bool,
}

impl Options {
  /// Construction config for the given model/tokenizer folders, with every
  /// other knob at its default. Not `const`: owned-path construction takes
  /// `impl Into<PathBuf>`.
  pub fn new(model_folder: impl Into<PathBuf>, tokenizer_folder: impl Into<PathBuf>) -> Self {
    Self {
      model_folder: model_folder.into(),
      tokenizer_folder: tokenizer_folder.into(),
      compute: ComputeOptions::new(),
      prewarm: DEFAULT_PREWARM,
      load: DEFAULT_LOAD,
    }
  }

  // -- model_folder -----------------------------------------------------
  /// Local folder containing the compiled (`.mlmodelc`) Whisper models.
  #[inline(always)]
  pub fn model_folder(&self) -> &Path {
    self.model_folder.as_path()
  }
  /// Builder form of [`Self::set_model_folder`].
  #[must_use]
  #[inline(always)]
  pub fn with_model_folder(mut self, model_folder: impl Into<PathBuf>) -> Self {
    self.set_model_folder(model_folder);
    self
  }
  /// Sets [`Self::model_folder`] in place.
  #[inline(always)]
  pub fn set_model_folder(&mut self, model_folder: impl Into<PathBuf>) -> &mut Self {
    self.model_folder = model_folder.into();
    self
  }

  // -- tokenizer_folder ---------------------------------------------------
  /// Local folder containing the BPE tokenizer files.
  #[inline(always)]
  pub fn tokenizer_folder(&self) -> &Path {
    self.tokenizer_folder.as_path()
  }
  /// Builder form of [`Self::set_tokenizer_folder`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokenizer_folder(mut self, tokenizer_folder: impl Into<PathBuf>) -> Self {
    self.set_tokenizer_folder(tokenizer_folder);
    self
  }
  /// Sets [`Self::tokenizer_folder`] in place.
  #[inline(always)]
  pub fn set_tokenizer_folder(&mut self, tokenizer_folder: impl Into<PathBuf>) -> &mut Self {
    self.tokenizer_folder = tokenizer_folder.into();
    self
  }

  // -- compute ------------------------------------------------------------
  /// Per-stage CoreML compute-unit selection.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeOptions {
    self.compute
  }
  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compute(mut self, compute: ComputeOptions) -> Self {
    self.set_compute(compute);
    self
  }
  /// Sets [`Self::compute`] in place.
  #[inline(always)]
  pub const fn set_compute(&mut self, compute: ComputeOptions) -> &mut Self {
    self.compute = compute;
    self
  }

  // -- prewarm (bool) -------------------------------------------------------
  /// Sequentially load-then-unload each model once before real use, to
  /// force ANE specialization up front instead of at first inference.
  #[inline(always)]
  pub const fn prewarm(&self) -> bool {
    self.prewarm
  }
  /// Builder form of [`Self::set_prewarm`].
  #[must_use]
  #[inline(always)]
  pub const fn with_prewarm(mut self) -> Self {
    self.set_prewarm();
    self
  }
  /// Sets [`Self::prewarm`] to `true`.
  #[inline(always)]
  pub const fn set_prewarm(&mut self) -> &mut Self {
    self.prewarm = true;
    self
  }
  /// Builder form of [`Self::update_prewarm`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_prewarm(mut self, prewarm: bool) -> Self {
    self.update_prewarm(prewarm);
    self
  }
  /// Assigns [`Self::prewarm`] directly.
  #[inline(always)]
  pub const fn update_prewarm(&mut self, prewarm: bool) -> &mut Self {
    self.prewarm = prewarm;
    self
  }
  /// Sets [`Self::prewarm`] to `false`.
  #[inline(always)]
  pub const fn clear_prewarm(&mut self) -> &mut Self {
    self.prewarm = false;
    self
  }

  // -- load (bool) ----------------------------------------------------------
  /// Load the models at construction time.
  #[inline(always)]
  pub const fn load(&self) -> bool {
    self.load
  }
  /// Builder form of [`Self::set_load`].
  #[must_use]
  #[inline(always)]
  pub const fn with_load(mut self) -> Self {
    self.set_load();
    self
  }
  /// Sets [`Self::load`] to `true`.
  #[inline(always)]
  pub const fn set_load(&mut self) -> &mut Self {
    self.load = true;
    self
  }
  /// Builder form of [`Self::update_load`].
  #[must_use]
  #[inline(always)]
  pub const fn maybe_load(mut self, load: bool) -> Self {
    self.update_load(load);
    self
  }
  /// Assigns [`Self::load`] directly.
  #[inline(always)]
  pub const fn update_load(&mut self, load: bool) -> &mut Self {
    self.load = load;
    self
  }
  /// Sets [`Self::load`] to `false`.
  #[inline(always)]
  pub const fn clear_load(&mut self) -> &mut Self {
    self.load = false;
    self
  }
}

#[cfg(test)]
mod tests;
