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
//!   seed), the [`ComputeOptions`] the pipeline was built with, and — from
//!   the transcript itself — the language the decode actually **detected**
//!   and the *effective* temperature it actually landed on.
//!   [`Provenance::from_options`] fills in everything the options alone
//!   settle; [`Provenance::for_result`] adds the two outcome facts, and is
//!   the constructor to reach for when you have a transcript in hand.
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
//!   is therefore the field that tells you whether the decode was
//!   greedy/deterministic (`Some(0.0)`) or sampled, and
//!   [`Provenance::seed`] tells you whether that sampling can be replayed
//!   at all. Record both, or a re-run that disagrees is uninvestigable.
//! - **Auto-detect makes the *configured* language useless as a record.**
//!   It is `""` whenever the decoder is left to detect (the default
//!   pairing), so a record built from options alone names no language at
//!   all. [`Provenance::detected_language`] is the field that carries what
//!   was actually spoken, and only [`Provenance::for_result`] — which is
//!   handed the transcript — can fill it in.

use coremlit::ComputeUnits;

use crate::{
  options::{ChunkingStrategy, ComputeOptions, DecodingOptions, Task},
  result::{TranscriptionResult, TranscriptionSegment},
};

#[cfg(test)]
mod tests;

/// Deserializes a **required but nullable** `Option` field.
///
/// Serde's derive special-cases a *missing* `Option` field to `None` (its
/// `missing_field` helper feeds the type a deserializer that answers
/// `deserialize_option` with `visit_none`), so a bare `Option<T>` field is
/// silently optional even with no `serde(default)` on it. That is exactly
/// the silent defaulting this type refuses: an absent
/// [`Provenance::effective_temperature`] would read back as "the fallback
/// ladder split the segments" when all it really means is "whoever wrote
/// this record dropped the field". Naming a `deserialize_with` sends the
/// derive down its required-field path instead — the key must be present,
/// and `null` then carries its real meaning and nothing else.
#[cfg(feature = "serde")]
fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
  D: serde::Deserializer<'de>,
  T: serde::Deserialize<'de>,
{
  <Option<T> as serde::Deserialize<'de>>::deserialize(deserializer)
}

/// The one temperature every segment of a result agrees on, or `None` when
/// they do not — backing [`Provenance::effective_temperature`] for
/// [`Provenance::for_result`].
///
/// `None` for an empty slice too: the temperature-fallback ladder runs
/// per *window*, so a result is only describable by a single effective
/// temperature when every segment in it actually landed on the same rung,
/// and a result with no segments has nothing to land. (Both cases are
/// ordinary now — [`DecodingOptions::drop_blank_audio`] empties a silent
/// chunk outright.) Claiming a number here for either would be a
/// fabrication; `None` says the honest thing.
fn unanimous_temperature(segments: &[TranscriptionSegment]) -> Option<f32> {
  let first = segments.first()?.temperature();
  segments
    .iter()
    .all(|segment| segment.temperature() == first)
    .then_some(first)
}

/// A serde-serializable record of what produced a transcript: the resolved
/// decode configuration, the compute units it ran on, the language it
/// detected and the effective temperature it landed on, and — when the
/// caller supplies them — the model and tokenizer identity.
///
/// Build it with [`Self::for_result`] when you have the transcript (the
/// form that records the detected language and the effective temperature);
/// with [`Self::for_segment`] to record one segment's own rung of the
/// fallback ladder; or with [`Self::from_options`] from the configuration
/// alone. Then attach the identity the library cannot know:
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
  /// The language the transcript was actually decoded in
  /// ([`TranscriptionResult::language`]) — the **outcome**, where
  /// [`Self::language`] is the **input**. This is the one that matters
  /// under auto-detect: `language` is then empty (and it is empty on every
  /// run with [`DecodingOptions::use_prefill_prompt`] cleared, the default
  /// pairing for detection), so the configured value says nothing at all
  /// about what was spoken, and only this field does.
  ///
  /// `None` **iff the record was built without a result** — by
  /// [`Self::from_options`] or [`Self::for_segment`], neither of which is
  /// handed a [`TranscriptionResult`] and so neither of which can observe
  /// the detection outcome. Never inferred from the configured language.
  /// [`Self::for_result`] always fills it in.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  detected_language: Option<String>,
  /// The temperature the decode **actually landed on** — the fallback
  /// ladder's accepted attempt, read off
  /// [`TranscriptionSegment::temperature`]. Equal to [`Self::temperature`]
  /// when no fallback was needed; higher when the ladder climbed.
  /// `Some(0.0)` means greedy/argmax and therefore deterministic; the
  /// overwhelmingly common no-fallback case.
  ///
  /// `None` means **no single temperature describes this transcript**. The
  /// ladder runs per *window*, so two segments of one result can
  /// legitimately land on different rungs; when they do, any single number
  /// here would be a lie about at least one of them. A result with no
  /// segments at all (silence, once
  /// [`DecodingOptions::drop_blank_audio`] has emptied it) is `None` for
  /// the same reason: nothing landed anywhere.
  /// [`Self::from_options`]/[`Self::for_segment`] are always `Some` — they
  /// are handed the one temperature they record.
  #[cfg_attr(feature = "serde", serde(deserialize_with = "required_option"))]
  effective_temperature: Option<f32>,

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
  /// The shared capture: every library-known fact off `decoding`/`compute`,
  /// with the two outcome fields — which only a result or a segment can
  /// supply — passed in, and the identity left `None`.
  fn capture(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    detected_language: Option<String>,
    effective_temperature: Option<f32>,
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
      detected_language,
      effective_temperature,
      model_id: None,
      model_revision: None,
      tokenizer_id: None,
      tokenizer_revision: None,
    }
  }

  /// Captures every library-known fact from `decoding`/`compute` plus the
  /// `effective_temperature` a decode landed on, leaving the model and
  /// tokenizer identity unset (`None`) for the caller to fill in — this
  /// crate loads from bare local folders and genuinely does not know it
  /// (see the module docs).
  ///
  /// [`Self::detected_language`] is left `None`: options alone cannot know
  /// what language was detected. Reach for [`Self::for_result`] when you
  /// have the transcript — that is the constructor that records it.
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
    Self::capture(decoding, compute, None, Some(effective_temperature))
  }

  /// [`Self::from_options`] with the effective temperature read straight
  /// off a decoded `segment` — the ergonomic form when recording
  /// provenance for one segment of a transcript you already have.
  ///
  /// Per-segment for the reason spelled out on [`Self::from_options`]:
  /// only the segment knows which rung of the fallback ladder its decode
  /// was accepted at. For the whole transcript, use [`Self::for_result`].
  pub fn for_segment(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    segment: &TranscriptionSegment,
  ) -> Self {
    Self::from_options(decoding, compute, segment.temperature())
  }

  /// The **result-level** capture: [`Self::from_options`]'s facts, plus the
  /// two a whole transcript — and only a whole transcript — can settle.
  ///
  /// - [`Self::detected_language`] becomes `Some(result.language())`: the
  ///   language the decode **actually ran in**. This is the fact worth
  ///   recording, and it is one [`Self::for_segment`] structurally cannot
  ///   reach (it is handed a segment, not the result that carries the
  ///   detection outcome). Under the default auto-detect the *configured*
  ///   [`Self::language`] is just `""`, so without this a record of the
  ///   common case names no language at all.
  /// - [`Self::effective_temperature`] becomes `Some(t)` **iff every
  ///   segment landed on the same `t`** — the overwhelmingly common
  ///   no-fallback case, which yields `Some(0.0)` — and `None` when the
  ///   per-window fallback ladder split them, or when the result has no
  ///   segments at all to agree (silence, after
  ///   [`DecodingOptions::drop_blank_audio`] empties it). A result-level
  ///   `f32` would have had to invent a number for both.
  ///
  /// The model/tokenizer identity is still the caller's to supply — a
  /// result cannot know it either.
  ///
  /// ```
  /// use whisperkit::{
  ///   options::{ComputeOptions, DecodingOptions},
  ///   provenance::Provenance,
  ///   result::{TranscriptionResult, TranscriptionTimings},
  /// };
  ///
  /// // Auto-detect: the CONFIGURED language is empty ...
  /// let decoding = DecodingOptions::new();
  /// let compute = ComputeOptions::new();
  /// let result = TranscriptionResult::new(
  ///   "Hello world.",
  ///   Vec::new(),
  ///   "en",
  ///   TranscriptionTimings::new(),
  /// );
  ///
  /// let provenance = Provenance::for_result(&decoding, &compute, &result);
  /// assert_eq!(provenance.language(), "");
  /// // ... and the DETECTED one is the fact actually worth persisting.
  /// assert_eq!(provenance.detected_language(), Some("en"));
  /// // No segments landed anywhere, so no single temperature describes it.
  /// assert_eq!(provenance.effective_temperature(), None);
  /// ```
  pub fn for_result(
    decoding: &DecodingOptions,
    compute: &ComputeOptions,
    result: &TranscriptionResult,
  ) -> Self {
    Self::capture(
      decoding,
      compute,
      Some(result.language().to_string()),
      unanimous_temperature(result.segments_slice()),
    )
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

  // -- detected_language --------------------------------------------------
  /// The language the transcript was actually decoded in — the outcome,
  /// where [`Self::language`] is the configured input. `None` iff this
  /// record was built without a result ([`Self::from_options`] /
  /// [`Self::for_segment`]); never inferred. See the field's doc.
  #[inline(always)]
  pub fn detected_language(&self) -> Option<&str> {
    self.detected_language.as_deref()
  }

  // -- effective_temperature ----------------------------------------------
  /// The temperature the decode actually landed on. `Some(0.0)` means
  /// greedy and therefore deterministic; anything higher was sampled, and
  /// is only reproducible if [`Self::seed`] is set. `None` means no single
  /// temperature describes the transcript — the per-window fallback ladder
  /// split its segments, or it has no segments. See the field's doc.
  #[inline(always)]
  pub const fn effective_temperature(&self) -> Option<f32> {
    self.effective_temperature
  }

  /// Whether this transcript can be reproduced byte-for-byte by re-running
  /// the same audio through the same options: true when the decode was
  /// greedy (an effective temperature of `0.0` never draws from the
  /// sampler) or when a [`Self::seed`] makes the draws replayable.
  ///
  /// A `None` [`Self::effective_temperature`] is treated as **not**
  /// self-evidently reproducible, and so needs a seed: the ladder having
  /// split the segments means at least one of them climbed off `0.0` and
  /// sampled (the rungs only ever ascend), and a segment-less result
  /// carries no evidence either way. Conservative on purpose — this
  /// predicate must never claim reproducibility it cannot back.
  ///
  /// A seed makes *this port's* output reproducible; it cannot make that
  /// output match Swift's, which has no seed knob and always draws
  /// unseeded (see [`DecodingOptions::seed`]).
  #[inline(always)]
  pub const fn is_reproducible(&self) -> bool {
    match self.effective_temperature {
      Some(temperature) => temperature == 0.0 || self.seed.is_some(),
      None => self.seed.is_some(),
    }
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
