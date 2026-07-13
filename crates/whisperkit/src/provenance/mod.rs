//! Structured transcription provenance (coremlit issue #14, following
//! issue #9's "record what produced this transcript" recommendation):
//! [`Provenance`] bundles, in one serde-serializable record, the decode
//! facts that determine — or merely describe — a transcript, so a consumer
//! can persist them alongside the text instead of hand-assembling the
//! bundle from [`DecodingOptions`]/[`ComputeOptions`]/
//! [`TranscriptionSegment`] at every call site.
//!
//! # What this type can and cannot know
//!
//! This module is deliberately honest about the boundary, because a
//! provenance record that quietly invents a field is worse than one that
//! admits the gap:
//!
//! - **Library-known** — everything reachable from the resolved
//!   [`DecodingOptions`] (task, language and its resolved
//!   `detect_language` coupling, prefill, skip-special, word-timestamps,
//!   chunking/VAD strategy, the whole temperature-fallback ladder, the
//!   seed), the [`ComputeOptions`] the pipeline was built with, and the
//!   *effective* decode temperature a segment actually landed on.
//!   [`Provenance::from_options`] fills every one of these in for you.
//! - **Consumer-supplied** — the model and tokenizer identity
//!   ([`Provenance::model_id`]/[`Provenance::model_revision`],
//!   [`Provenance::tokenizer_id`]/[`Provenance::tokenizer_revision`]).
//!   These are *load-time* facts: this crate loads models and tokenizers
//!   from plain local folders ([`crate::options::Options`] holds nothing
//!   but two [`std::path::Path`]s; model auto-download is deferred, spec
//!   §4.7), so nothing in the pipeline ever sees a Hub repo id or a git
//!   revision. They start `None` and stay `None` unless the caller — who
//!   *does* know which artifact it put in those folders — sets them. This
//!   crate will not fabricate a revision it cannot observe.
//!
//! # Why these fields are worth recording
//!
//! - **The compute unit changes the output.** CoreML's CPU, GPU, and
//!   Neural Engine paths do not produce bit-identical floating-point
//!   results, so a transcript is only reproducible against the same
//!   [`ComputeOptions`] — this crate's own golden baselines are pinned
//!   per-unit for exactly that reason. Recording the units is what makes a
//!   later mismatch diagnosable instead of mysterious.
//! - **A non-zero temperature is not reproducible without a seed.** The
//!   temperature-fallback ladder samples stochastically once it climbs off
//!   `0.0`, and [`DecodingOptions::seed`] is unset by default (OS-seeded,
//!   matching Swift's own unseeded draw). [`Provenance::effective_temperature`]
//!   is therefore the field that tells you whether a given segment was
//!   greedy/deterministic (`0.0`) or sampled, and
//!   [`Provenance::seed`] tells you whether that sampling can be replayed
//!   at all. Record both, or a re-run that disagrees is uninvestigable.

use coremlit::ComputeUnits;

use crate::{
  options::{ChunkingStrategy, ComputeOptions, DecodingOptions, Task},
  result::TranscriptionSegment,
};

#[cfg(test)]
mod tests;

/// A serde-serializable record of what produced a transcript: the resolved
/// decode configuration, the compute units it ran on, the effective decode
/// temperature, and — when the caller supplies them — the model and
/// tokenizer identity.
///
/// Build it with [`Self::from_options`] (or [`Self::for_segment`], which
/// reads the effective temperature straight off a decoded segment), then
/// attach the identity the library cannot know:
///
/// ```
/// use whisperkit::{
///   options::{ComputeOptions, DecodingOptions},
///   provenance::Provenance,
/// };
///
/// let decoding = DecodingOptions::new().with_language("en");
/// let compute = ComputeOptions::new();
///
/// // `0.0` here is the effective temperature the segment decoded at —
/// // read it off `TranscriptionSegment::temperature()` in real use.
/// let provenance = Provenance::from_options(&decoding, &compute, 0.0)
///   .with_model_id("openai_whisper-tiny")
///   .with_model_revision("a1b2c3d");
///
/// assert_eq!(provenance.language(), "en");
/// assert_eq!(provenance.model_id(), Some("openai_whisper-tiny"));
/// // Never fabricated: the tokenizer identity was not supplied.
/// assert_eq!(provenance.tokenizer_revision(), None);
/// ```
///
/// The library-known fields are captured facts, so they are read-only
/// (there are no setters for them — reconstruct from the options instead);
/// only the four identity fields are settable, and each serializes as
/// **absent** while unset rather than as `null`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Provenance {
  // -- library-known: the resolved decode configuration -----------------
  /// The decode task (transcribe vs. translate) — it changes the output
  /// text outright, so a record that omitted it could not tell a
  /// translation from a transcription.
  task: Task,
  /// The configured spoken language (ISO code); empty means the decoder
  /// was left to auto-detect. This is the *configured* value: the language
  /// actually detected for a given run is reported by
  /// [`TranscriptionResult::language`](crate::result::TranscriptionResult::language),
  /// which is where a consumer should read the outcome rather than the
  /// input.
  language: String,
  /// Whether language detection ran, already **resolved** through
  /// [`DecodingOptions::detect_language`]'s coupling to
  /// `use_prefill_prompt` — the value the pipeline actually acted on, not
  /// the raw tri-state. It is recorded because a detection probe consumes
  /// a sampler draw, so at a non-zero temperature it can shift the tokens
  /// that follow.
  detect_language: bool,
  /// Whether the prefill tokens were forced from `task`/`language`.
  use_prefill_prompt: bool,
  /// Whether special tokens were omitted from decoded segment text.
  skip_special_tokens: bool,
  /// Whether word-level DTW timestamps were computed.
  word_timestamps: bool,
  /// The chunking/VAD strategy long-form audio was split with.
  chunking_strategy: ChunkingStrategy,
  /// The **base** decode temperature the fallback ladder started from.
  /// See [`Self::effective_temperature`] for the one a segment landed on.
  temperature: f32,
  /// The temperature added per fallback retry.
  temperature_increment_on_fallback: f32,
  /// The maximum number of fallback retries.
  temperature_fallback_count: usize,
  /// The base seed for reproducible sampling, or `None` when sampling was
  /// left OS-seeded (the default). See this module's docs: without a seed,
  /// any segment whose [`Self::effective_temperature`] climbed above `0.0`
  /// is not reproducible.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  seed: Option<u64>,

  // -- library-known: compute + outcome ---------------------------------
  /// The per-stage CoreML compute units the pipeline ran on. Recorded
  /// because they change the output (see this module's docs).
  compute: ComputeOptions,
  /// The temperature the decode **actually landed on** — the fallback
  /// ladder's accepted attempt, read off
  /// [`TranscriptionSegment::temperature`]. Equal to [`Self::temperature`]
  /// when no fallback was needed; higher when the ladder climbed. `0.0`
  /// means greedy/argmax and therefore deterministic.
  effective_temperature: f32,

  // -- consumer-supplied: load-time identity ----------------------------
  /// The model's identity (e.g. a Hub repo id), if the caller supplied it.
  /// Never inferred — see this module's docs.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  model_id: Option<String>,
  /// The model's revision (e.g. a git sha), if the caller supplied it.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  model_revision: Option<String>,
  /// The tokenizer's identity, if the caller supplied it.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  tokenizer_id: Option<String>,
  /// The tokenizer's revision, if the caller supplied it.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  tokenizer_revision: Option<String>,
}

impl Provenance {
  /// Captures every library-known fact from `decoding`/`compute` plus the
  /// `effective_temperature` a decode landed on, leaving the model and
  /// tokenizer identity unset (`None`) for the caller to fill in — this
  /// crate loads from bare local folders and genuinely does not know it
  /// (see the module docs).
  ///
  /// `effective_temperature` is per-*segment*, not per-result: the
  /// fallback ladder runs per window, so two segments of one transcript
  /// can legitimately land on different temperatures. Pass
  /// [`TranscriptionSegment::temperature`] for the segment being recorded
  /// — or use [`Self::for_segment`], which does exactly that.
  pub fn from_options(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    effective_temperature: f32,
  ) -> Self {
    Self {
      task: decoding.task(),
      language: decoding.language().to_string(),
      // The resolved getter, not the raw tri-state: this records what the
      // pipeline acted on (see the field's doc).
      detect_language: decoding.detect_language(),
      use_prefill_prompt: decoding.use_prefill_prompt(),
      skip_special_tokens: decoding.skip_special_tokens(),
      word_timestamps: decoding.word_timestamps(),
      chunking_strategy: decoding.chunking_strategy(),
      temperature: decoding.temperature(),
      temperature_increment_on_fallback: decoding.temperature_increment_on_fallback(),
      temperature_fallback_count: decoding.temperature_fallback_count(),
      seed: decoding.seed(),
      compute: *compute,
      effective_temperature,
      model_id: None,
      model_revision: None,
      tokenizer_id: None,
      tokenizer_revision: None,
    }
  }

  /// [`Self::from_options`] with the effective temperature read straight
  /// off a decoded `segment` — the ergonomic form when recording
  /// provenance for a transcript you already have.
  ///
  /// Provenance is per-segment for the reason spelled out on
  /// [`Self::from_options`]: only the segment knows which rung of the
  /// fallback ladder its decode was accepted at.
  pub fn for_segment(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    segment: &TranscriptionSegment,
  ) -> Self {
    Self::from_options(decoding, compute, segment.temperature())
  }

  // -- task ---------------------------------------------------------------
  /// The decode task the transcript was produced with.
  #[inline(always)]
  pub const fn task(&self) -> Task {
    self.task
  }

  // -- language -----------------------------------------------------------
  /// The configured spoken language (ISO code); empty means auto-detect.
  #[inline(always)]
  pub fn language(&self) -> &str {
    self.language.as_str()
  }

  // -- detect_language ----------------------------------------------------
  /// Whether language detection ran (already resolved — see the field doc).
  #[inline(always)]
  pub const fn detect_language(&self) -> bool {
    self.detect_language
  }

  // -- use_prefill_prompt -------------------------------------------------
  /// Whether the prefill tokens were forced from `task`/`language`.
  #[inline(always)]
  pub const fn use_prefill_prompt(&self) -> bool {
    self.use_prefill_prompt
  }

  // -- skip_special_tokens ------------------------------------------------
  /// Whether special tokens were omitted from decoded segment text.
  #[inline(always)]
  pub const fn skip_special_tokens(&self) -> bool {
    self.skip_special_tokens
  }

  // -- word_timestamps ----------------------------------------------------
  /// Whether word-level DTW timestamps were computed.
  #[inline(always)]
  pub const fn word_timestamps(&self) -> bool {
    self.word_timestamps
  }

  // -- chunking_strategy --------------------------------------------------
  /// The chunking/VAD strategy long-form audio was split with.
  #[inline(always)]
  pub const fn chunking_strategy(&self) -> ChunkingStrategy {
    self.chunking_strategy
  }

  // -- temperature --------------------------------------------------------
  /// The base decode temperature the fallback ladder started from.
  #[inline(always)]
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }

  // -- temperature_increment_on_fallback ----------------------------------
  /// The temperature added per fallback retry.
  #[inline(always)]
  pub const fn temperature_increment_on_fallback(&self) -> f32 {
    self.temperature_increment_on_fallback
  }

  // -- temperature_fallback_count -----------------------------------------
  /// The maximum number of fallback retries.
  #[inline(always)]
  pub const fn temperature_fallback_count(&self) -> usize {
    self.temperature_fallback_count
  }

  // -- seed ---------------------------------------------------------------
  /// The base seed for reproducible sampling, or `None` when sampling was
  /// OS-seeded. Without one, a segment whose
  /// [`Self::effective_temperature`] exceeds `0.0` cannot be replayed.
  #[inline(always)]
  pub const fn seed(&self) -> Option<u64> {
    self.seed
  }

  // -- compute ------------------------------------------------------------
  /// The per-stage CoreML compute units the pipeline ran on.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeOptions {
    self.compute
  }

  /// The audio encoder's compute units — the stage whose unit most visibly
  /// moves the output, and the one a baseline is usually pinned against.
  #[inline(always)]
  pub const fn encoder_compute_units(&self) -> ComputeUnits {
    self.compute.encoder()
  }

  // -- effective_temperature ----------------------------------------------
  /// The temperature the decode actually landed on. `0.0` means greedy and
  /// therefore deterministic; anything higher was sampled, and is only
  /// reproducible if [`Self::seed`] is set.
  #[inline(always)]
  pub const fn effective_temperature(&self) -> f32 {
    self.effective_temperature
  }

  /// Whether this transcript can be reproduced byte-for-byte by re-running
  /// the same audio through the same options: true when the decode was
  /// greedy (an effective temperature of `0.0` never draws from the
  /// sampler) or when a [`Self::seed`] makes the draws replayable.
  ///
  /// A seed makes *this port's* output reproducible; it cannot make that
  /// output match Swift's, which has no seed knob and always draws
  /// unseeded (see [`DecodingOptions::seed`]).
  #[inline(always)]
  pub const fn is_reproducible(&self) -> bool {
    self.effective_temperature == 0.0 || self.seed.is_some()
  }

  // -- model_id (Option<String>) ------------------------------------------
  /// The model's identity, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn model_id(&self) -> Option<&str> {
    self.model_id.as_deref()
  }
  /// Builder form of [`Self::set_model_id`].
  #[must_use]
  #[inline(always)]
  pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
    self.set_model_id(model_id);
    self
  }
  /// Sets [`Self::model_id`] to `Some(model_id)`.
  #[inline(always)]
  pub fn set_model_id(&mut self, model_id: impl Into<String>) -> &mut Self {
    self.model_id = Some(model_id.into());
    self
  }
  /// Builder form of [`Self::update_model_id`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_model_id(mut self, model_id: Option<String>) -> Self {
    self.update_model_id(model_id);
    self
  }
  /// Assigns [`Self::model_id`] directly.
  #[inline(always)]
  pub fn update_model_id(&mut self, model_id: Option<String>) -> &mut Self {
    self.model_id = model_id;
    self
  }
  /// Sets [`Self::model_id`] to `None`.
  #[inline(always)]
  pub fn clear_model_id(&mut self) -> &mut Self {
    self.model_id = None;
    self
  }

  // -- model_revision (Option<String>) ------------------------------------
  /// The model's revision, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn model_revision(&self) -> Option<&str> {
    self.model_revision.as_deref()
  }
  /// Builder form of [`Self::set_model_revision`].
  #[must_use]
  #[inline(always)]
  pub fn with_model_revision(mut self, model_revision: impl Into<String>) -> Self {
    self.set_model_revision(model_revision);
    self
  }
  /// Sets [`Self::model_revision`] to `Some(model_revision)`.
  #[inline(always)]
  pub fn set_model_revision(&mut self, model_revision: impl Into<String>) -> &mut Self {
    self.model_revision = Some(model_revision.into());
    self
  }
  /// Builder form of [`Self::update_model_revision`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_model_revision(mut self, model_revision: Option<String>) -> Self {
    self.update_model_revision(model_revision);
    self
  }
  /// Assigns [`Self::model_revision`] directly.
  #[inline(always)]
  pub fn update_model_revision(&mut self, model_revision: Option<String>) -> &mut Self {
    self.model_revision = model_revision;
    self
  }
  /// Sets [`Self::model_revision`] to `None`.
  #[inline(always)]
  pub fn clear_model_revision(&mut self) -> &mut Self {
    self.model_revision = None;
    self
  }

  // -- tokenizer_id (Option<String>) --------------------------------------
  /// The tokenizer's identity, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn tokenizer_id(&self) -> Option<&str> {
    self.tokenizer_id.as_deref()
  }
  /// Builder form of [`Self::set_tokenizer_id`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokenizer_id(mut self, tokenizer_id: impl Into<String>) -> Self {
    self.set_tokenizer_id(tokenizer_id);
    self
  }
  /// Sets [`Self::tokenizer_id`] to `Some(tokenizer_id)`.
  #[inline(always)]
  pub fn set_tokenizer_id(&mut self, tokenizer_id: impl Into<String>) -> &mut Self {
    self.tokenizer_id = Some(tokenizer_id.into());
    self
  }
  /// Builder form of [`Self::update_tokenizer_id`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_tokenizer_id(mut self, tokenizer_id: Option<String>) -> Self {
    self.update_tokenizer_id(tokenizer_id);
    self
  }
  /// Assigns [`Self::tokenizer_id`] directly.
  #[inline(always)]
  pub fn update_tokenizer_id(&mut self, tokenizer_id: Option<String>) -> &mut Self {
    self.tokenizer_id = tokenizer_id;
    self
  }
  /// Sets [`Self::tokenizer_id`] to `None`.
  #[inline(always)]
  pub fn clear_tokenizer_id(&mut self) -> &mut Self {
    self.tokenizer_id = None;
    self
  }

  // -- tokenizer_revision (Option<String>) --------------------------------
  /// The tokenizer's revision, if the caller supplied it. Never inferred.
  #[inline(always)]
  pub fn tokenizer_revision(&self) -> Option<&str> {
    self.tokenizer_revision.as_deref()
  }
  /// Builder form of [`Self::set_tokenizer_revision`].
  #[must_use]
  #[inline(always)]
  pub fn with_tokenizer_revision(mut self, tokenizer_revision: impl Into<String>) -> Self {
    self.set_tokenizer_revision(tokenizer_revision);
    self
  }
  /// Sets [`Self::tokenizer_revision`] to `Some(tokenizer_revision)`.
  #[inline(always)]
  pub fn set_tokenizer_revision(&mut self, tokenizer_revision: impl Into<String>) -> &mut Self {
    self.tokenizer_revision = Some(tokenizer_revision.into());
    self
  }
  /// Builder form of [`Self::update_tokenizer_revision`].
  #[must_use]
  #[inline(always)]
  pub fn maybe_tokenizer_revision(mut self, tokenizer_revision: Option<String>) -> Self {
    self.update_tokenizer_revision(tokenizer_revision);
    self
  }
  /// Assigns [`Self::tokenizer_revision`] directly.
  #[inline(always)]
  pub fn update_tokenizer_revision(&mut self, tokenizer_revision: Option<String>) -> &mut Self {
    self.tokenizer_revision = tokenizer_revision;
    self
  }
  /// Sets [`Self::tokenizer_revision`] to `None`.
  #[inline(always)]
  pub fn clear_tokenizer_revision(&mut self) -> &mut Self {
    self.tokenizer_revision = None;
    self
  }
}
