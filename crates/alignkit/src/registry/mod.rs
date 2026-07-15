//! Multi-language aligner registry: [`AlignmentSet`] keyed by
//! [`AlignerKey`], built with [`AlignmentSetBuilder`].
//!
//! Semantics mirror asry's own registry
//! (`asry/src/runner/aligner/{key.rs,set.rs,builder.rs}`) exactly — the same
//! `Lang → Any → fallback` strict lookup, the same
//! failure-never-falls-through-to-`Any` rule, the same OOV language patch —
//! with **one deliberate divergence**: this registry stores a plain
//! [`Aligner`], not a `Mutex<Aligner>`. asry needs the mutex because its ORT
//! `Aligner::align` is `&mut self`; alignkit's
//! [`Aligner::align_chunk`](crate::aligner::Aligner::align_chunk) is `&self`
//! (the CoreML `Model` predicts without `&mut`), so there is nothing to lock.
//!
//! # Scope of that win
//!
//! `&self` alignment means many chunks can be aligned through one registry
//! without interior mutability — but **not** that an [`AlignmentSet`] can be
//! shared across threads today. It is `!Sync`: an [`Aligner`] holds asry's
//! `EmissionsAligner`, which owns a
//! [`DynTextNormalizer`](asry::emissions::DynTextNormalizer) =
//! `Box<dyn TextNormalizer>`, and asry declares
//! `TextNormalizer: Send` with no `Sync` bound — so the box is not `Sync`,
//! and neither is anything holding it. An `Arc<AlignmentSet>` fanned out to
//! worker threads will not compile until asry widens that bound to
//! `Send + Sync`. Until then the mutex-free design buys single-threaded reuse
//! and one less lock in the hot path, not free cross-thread sharing.

use core::sync::atomic::AtomicBool;
use std::collections::HashMap;

use asry::{
  AlignmentResult, Lang, TimeRange,
  emissions::{OovEvent, OutputClock, ResolvedOov},
};

use crate::{aligner::Aligner, error::AlignError};

/// Identifies an aligner in the [`AlignmentSet`] registry.
///
/// Lookup order (see [`AlignmentSet::lookup`]):
/// 1. [`AlignerKey::Lang`]`(L)` — the explicit aligner for a language.
/// 2. [`AlignerKey::Any`] — the multilingual fallback (registry miss only).
/// 3. The configured [`AlignmentFallback`].
///
/// **A registered aligner's *failure* does NOT fall through to `Any`.** If
/// `Lang(L)` is registered but its alignment errors, that error surfaces;
/// the `Any` aligner is not consulted (mirrors asry's strict-lookup
/// contract, `asry/src/runner/aligner/key.rs`).
///
/// `#[non_exhaustive]`: a registry key is a vocabulary this crate expects to
/// grow (a dialect- or model-keyed variant is the obvious next one), and a
/// caller matching on it should be forced to say what it does with a key it has
/// never heard of rather than fail to compile — or, worse, quietly mis-route.
/// Constructing [`AlignerKey::Lang`] / [`AlignerKey::Any`] is unaffected.
#[derive(Clone, PartialEq, Eq, Hash, Debug, derive_more::IsVariant)]
#[non_exhaustive]
pub enum AlignerKey {
  /// The explicit aligner for a specific language.
  Lang(Lang),
  /// The multilingual fallback aligner; consulted only on a registry miss
  /// for the requested language.
  Any,
}

/// Policy for a requested language with no registered aligner (and no
/// [`AlignerKey::Any`] fallback registered either).
///
/// A vocabulary enum with the workspace's full contract (mirrors
/// `whisperkit::log::LogLevel`): [`as_str`](Self::as_str) + a derived
/// [`Display`](core::fmt::Display), a total
/// [`FromStr`](core::str::FromStr) whose error
/// ([`ParseAlignmentFallbackError`]) is opaque, `snake_case` serde under the
/// `serde` feature, and `#[non_exhaustive]`. It is a *policy* — the kind of
/// value that arrives from a config file, a CLI flag or an env var — so it has
/// to survive a round trip through text, which it previously could not do in
/// either direction even though [`crate::AlignerOptions`] is itself
/// serde-gated.
#[derive(
  Copy, Clone, PartialEq, Eq, Debug, Default, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
#[cfg_attr(
  feature = "serde",
  derive(serde::Serialize, serde::Deserialize),
  serde(rename_all = "snake_case")
)]
#[non_exhaustive]
pub enum AlignmentFallback {
  /// Skip alignment for the chunk: the caller keeps the ASR text with empty
  /// per-word timings. The default — alignment availability never blocks the
  /// pipeline.
  #[default]
  SkipChunk,
  /// Treat a registry miss as a hard error the caller must handle — useful
  /// when a missing language should be a loud signal, not a silent skip.
  Error,
}

impl AlignmentFallback {
  /// Stable `snake_case` name of the policy — the same spelling the `serde`
  /// feature reads and writes, and the one
  /// [`FromStr`](core::str::FromStr) parses.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::SkipChunk => "skip_chunk",
      Self::Error => "error",
    }
  }
}

/// Error parsing an [`AlignmentFallback`] name.
///
/// Opaque (`(())`): the rejected input is the caller's own string and carrying
/// it back adds nothing they do not already hold, while the empty payload keeps
/// the type free to grow a real one later without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown alignment fallback policy name")]
pub struct ParseAlignmentFallbackError(());

impl core::str::FromStr for AlignmentFallback {
  type Err = ParseAlignmentFallbackError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "skip_chunk" => Self::SkipChunk,
      "error" => Self::Error,
      _ => return Err(ParseAlignmentFallbackError(())),
    })
  }
}

/// The result of an [`AlignmentSet::lookup`].
pub enum AlignmentLookup<'a> {
  /// Hit on [`AlignerKey::Lang`]`(L)`. A failure of this aligner does NOT
  /// fall through to [`AlignerKey::Any`].
  Hit {
    /// The matched key (always [`AlignerKey::Lang`]).
    matched: AlignerKey,
    /// The language-specific aligner.
    aligner: &'a Aligner,
  },
  /// Miss on `Lang(L)`, hit on [`AlignerKey::Any`] — the multilingual
  /// fallback is used.
  AnyFallback {
    /// The multilingual fallback aligner.
    aligner: &'a Aligner,
  },
  /// Miss on both `Lang(L)` and `Any`. The configured [`AlignmentFallback`]
  /// decides what the caller does.
  Miss {
    /// The configured miss policy.
    fallback: AlignmentFallback,
  },
}

/// A registry of [`Aligner`]s keyed by [`AlignerKey`].
///
/// Fields are private; construct via [`AlignmentSetBuilder`]. Lookup is
/// `&self`, and — unlike asry's `Mutex`-wrapped pool — the stored aligners
/// align through `&self` too, so the whole set is usable behind a shared
/// reference with no interior mutability. It is **not** `Sync`, so that
/// shared reference cannot cross threads; see the module doc's "Scope of that
/// win".
pub struct AlignmentSet {
  aligners: HashMap<AlignerKey, Aligner>,
  fallback: AlignmentFallback,
}

impl AlignmentSet {
  /// The configured registry-miss policy.
  #[must_use]
  pub const fn fallback(&self) -> AlignmentFallback {
    self.fallback
  }

  /// Number of registered aligners (including [`AlignerKey::Any`] if it was
  /// registered).
  #[must_use]
  pub fn len(&self) -> usize {
    self.aligners.len()
  }

  /// Whether the registry has zero aligners.
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.aligners.is_empty()
  }

  /// Look up an aligner for `language`, applying the strict `Lang → Any →
  /// fallback` order.
  #[must_use]
  pub fn lookup<'a>(&'a self, language: &Lang) -> AlignmentLookup<'a> {
    let lang_key = AlignerKey::Lang(language.clone());
    if let Some(aligner) = self.aligners.get(&lang_key) {
      return AlignmentLookup::Hit {
        matched: lang_key,
        aligner,
      };
    }
    if let Some(aligner) = self.aligners.get(&AlignerKey::Any) {
      return AlignmentLookup::AnyFallback { aligner };
    }
    AlignmentLookup::Miss {
      fallback: self.fallback,
    }
  }

  /// Detect out-of-vocabulary characters in `text` against the aligner
  /// registered for `language` (or the [`AlignerKey::Any`] aligner), with
  /// every event's language patched back to the *requested* `language`.
  ///
  /// Returns `Ok(empty)` on a registry miss (the caller then skips the chunk
  /// or surfaces the miss itself, so an empty decisions vec is the right
  /// shape).
  ///
  /// The language patch mirrors asry's `AlignmentSet::detect_oov`
  /// (`asry/src/runner/aligner/set.rs`):
  /// [`Aligner::detect_oov`](crate::aligner::Aligner::detect_oov) stamps
  /// each event with the matched aligner's OWN construction language, so an
  /// `Any` fallback (e.g. an English aligner serving another language) would
  /// otherwise route per-language OOV policy on the wrong key.
  ///
  /// # Errors
  /// [`AlignError::Alignment`] on a normalizer / tokenizer-engine failure
  /// from the matched aligner.
  pub fn detect_oov(&self, text: &str, language: &Lang) -> Result<Vec<OovEvent>, AlignError> {
    let aligner = match self.lookup(language) {
      AlignmentLookup::Hit { aligner, .. } | AlignmentLookup::AnyFallback { aligner } => aligner,
      AlignmentLookup::Miss { .. } => return Ok(Vec::new()),
    };
    let mut events = aligner.detect_oov(text)?;
    for event in &mut events {
      event.set_language(language.clone());
    }
    Ok(events)
  }

  /// Align one chunk end-to-end through the aligner registered for `language`,
  /// applying the strict `Lang → Any → fallback` lookup and — crucially —
  /// crossing the caller's requested-language OOV decisions safely into the
  /// aligner that actually runs.
  ///
  /// This is the registry-owned counterpart to
  /// [`Aligner::align_chunk`](crate::aligner::Aligner::align_chunk): call it
  /// with the SAME `language` you passed to [`Self::detect_oov`] and the same
  /// caller-resolved `oov_decisions` (in that order); the remaining arguments
  /// are [`Aligner::align_chunk`](crate::aligner::Aligner::align_chunk)'s,
  /// forwarded unchanged.
  ///
  /// # Why a registry-level align is needed at all
  ///
  /// [`Self::detect_oov`] stamps every OOV event with the *requested* language
  /// (so per-language OOV policy keys on it), but the bound aligner's
  /// `EmissionsAligner::prepare` validates decisions against the aligner's OWN
  /// construction language. When those differ — an English [`AlignerKey::Any`]
  /// aligner serving a Chinese request — handing that aligner the
  /// requested-language decisions directly fails with a hard decision-language
  /// error, so `Any`-fallback alignment breaks the moment any OOV decision is
  /// present. This method reconciles the two: it validates the decisions carry
  /// `language`, then re-stamps them to the bound aligner's language before
  /// aligning. The decision CONTENT (wildcard / fail-closed, chosen by the
  /// caller's per-`language` policy) is positional and unchanged; only the
  /// language tag is crossed, and asry's `ResolvedOov` positional identity
  /// ignores it. There is deliberately **no** caller-controlled
  /// expected-language knob — that is exactly the guard bypass this reconciles.
  ///
  /// An [`AlignerKey::Lang`]`(L)` hit needs no crossing (the decisions already
  /// carry `L`), but the same requested-language validation still runs here,
  /// before dispatch, so a mis-stamped decision is the typed
  /// [`AlignError::DecisionLanguage`] at the identical precedence to the `Any`
  /// route — not a generic alignment error (or, on oversized audio, an
  /// input-length error) from deep inside the bound aligner.
  ///
  /// # Registry miss
  ///
  /// On a miss (no `Lang(language)`, no `Any`) the configured
  /// [`AlignmentFallback`] decides: [`AlignmentFallback::SkipChunk`] returns an
  /// empty [`AlignmentResult`] (the ASR text survives, only per-word timings are
  /// dropped); [`AlignmentFallback::Error`] returns
  /// [`AlignError::LanguageUnsupported`].
  ///
  /// # Errors
  /// [`AlignError::DecisionLanguage`] if an `oov_decisions` entry does not carry
  /// `language`. [`AlignError::LanguageUnsupported`] on a miss under
  /// [`AlignmentFallback::Error`]. Otherwise any error
  /// [`Aligner::align_chunk`](crate::aligner::Aligner::align_chunk) itself
  /// returns.
  // Mirrors `Aligner::align_chunk`'s argument surface (already at the 7-arg
  // limit) plus the registry's `language` lookup key, so a caller uses the exact
  // call shape they already know rather than an opaque params struct. Same
  // rationale as whisperkit's Swift-mirroring signatures.
  #[allow(clippy::too_many_arguments)]
  pub fn align_chunk(
    &self,
    language: &Lang,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    clock: OutputClock,
    abort_flag: &AtomicBool,
    oov_decisions: &[ResolvedOov],
  ) -> Result<AlignmentResult, AlignError> {
    match self.lookup(language) {
      AlignmentLookup::Hit { aligner, .. } => {
        // Requested language == the aligner's own language (the builder asserts
        // it for AlignerKey::Lang), so a correctly-resolved decision already
        // carries the tag the aligner's `prepare` expects. Validate that HERE,
        // before dispatch, so a MIS-stamped decision surfaces as the same typed
        // DecisionLanguage error at the same precedence as the Any-fallback route
        // (which validates in `cross_decisions_into`) — not the undifferentiated
        // Alignment asry's `prepare` would raise, nor the InputTooLong the
        // encoder could raise first on oversized audio, both of which made the
        // error route-dependent (F2). The bound aligner's own guard still
        // re-checks underneath; this is the classifier in front of it.
        validate_decisions_language(oov_decisions, language)?;
        aligner.align_chunk(
          samples,
          sub_segments,
          text,
          clock,
          abort_flag,
          oov_decisions,
        )
      }
      AlignmentLookup::AnyFallback { aligner } => {
        // The Any aligner's language differs from the request. Validate the
        // decisions were resolved for the REQUESTED language (so we are not
        // masking a wrong-policy payload), then cross them into the aligner's
        // own language — the only tag its `prepare` will accept.
        let crossed = cross_decisions_into(oov_decisions, language, aligner.language_ref())?;
        aligner.align_chunk(samples, sub_segments, text, clock, abort_flag, &crossed)
      }
      AlignmentLookup::Miss { fallback } => match fallback {
        AlignmentFallback::SkipChunk => Ok(AlignmentResult::new(Vec::new())),
        AlignmentFallback::Error => Err(AlignError::LanguageUnsupported {
          language: language.clone(),
        }),
      },
    }
  }
}

/// Cross caller-resolved OOV decisions from the `requested` language into
/// `aligner_language`, for an [`AlignerKey::Any`] fallback.
///
/// Validates every decision carries `requested` (else
/// [`AlignError::DecisionLanguage`]), then returns a copy re-stamped to
/// `aligner_language`. Only the language tag changes; the
/// [`OovDecision`](asry::emissions::OovDecision) is positional and preserved,
/// and asry's [`ResolvedOov`] positional identity deliberately ignores language
/// — so the re-stamped decisions apply exactly the caller's
/// per-`requested`-language policy at the same positions while satisfying the
/// bound aligner's own-language guard.
///
/// When `requested == aligner_language` (an `Any` aligner serving its own
/// language) it is a validated clone.
fn cross_decisions_into(
  decisions: &[ResolvedOov],
  requested: &Lang,
  aligner_language: &Lang,
) -> Result<Vec<ResolvedOov>, AlignError> {
  // The same requested-language validation the exact-hit path runs — factored
  // out so both routes reject a mis-stamped decision with the identical typed
  // error at the identical precedence (before any re-stamp or dispatch).
  validate_decisions_language(decisions, requested)?;
  let mut crossed = Vec::with_capacity(decisions.len());
  for resolved in decisions {
    let mut event = resolved.event().clone();
    event.set_language(aligner_language.clone());
    crossed.push(ResolvedOov::new(event, resolved.decision()));
  }
  Ok(crossed)
}

/// Validate that every OOV decision carries the `requested` language, returning
/// [`AlignError::DecisionLanguage`] — naming the first offending index and the
/// language it was found stamped with — if any does not.
///
/// Shared by BOTH align routes ([`AlignmentSet::align_chunk`]'s exact-`Lang` hit
/// and, via [`cross_decisions_into`], the `Any` fallback) so a wrong-language
/// decision is rejected with the same typed error at the same precedence — ahead
/// of any dispatch — regardless of which aligner the lookup selected. The
/// exact-hit path used to skip this and forward the decisions straight to the
/// bound aligner, whose `prepare` DOES reject them, but only as an
/// undifferentiated [`AlignError::Alignment`] (asry's `Tokenization`); worse, on
/// oversized audio the encoder's [`AlignError::InputTooLong`] could surface
/// first, so the SAME wrong input produced different errors on the two routes
/// (F2).
fn validate_decisions_language(
  decisions: &[ResolvedOov],
  requested: &Lang,
) -> Result<(), AlignError> {
  for (index, resolved) in decisions.iter().enumerate() {
    if resolved.event().language() != requested {
      return Err(AlignError::DecisionLanguage {
        index,
        requested: requested.clone(),
        found: resolved.event().language().clone(),
      });
    }
  }
  Ok(())
}

/// Builder for [`AlignmentSet`]. Mirrors the crate's `with_`/`set_` builder
/// style.
pub struct AlignmentSetBuilder {
  aligners: HashMap<AlignerKey, Aligner>,
  fallback: AlignmentFallback,
}

impl AlignmentSetBuilder {
  /// An empty builder. Fallback defaults to [`AlignmentFallback::SkipChunk`].
  #[must_use]
  pub fn new() -> Self {
    Self {
      aligners: HashMap::new(),
      fallback: AlignmentFallback::SkipChunk,
    }
  }

  /// Builder form of [`Self::set_fallback`].
  #[must_use]
  pub const fn with_fallback(mut self, fallback: AlignmentFallback) -> Self {
    self.fallback = fallback;
    self
  }

  /// Set the registry-miss policy in place.
  pub const fn set_fallback(&mut self, fallback: AlignmentFallback) {
    self.fallback = fallback;
  }

  /// Register `aligner` under `key`, replacing any prior registration for
  /// the same key (last call wins).
  ///
  /// # Panics
  /// If `key` is [`AlignerKey::Lang`]`(L)` and `aligner.language_ref() != L`.
  /// A swapped registration would silently route another language's chunks
  /// through the wrong normalizer / tokenizer / model, producing
  /// plausible-but-corrupt timings; fail fast at build time instead.
  /// [`AlignerKey::Any`] accepts any aligner — it is the explicit
  /// multilingual escape hatch. (Mirrors asry's
  /// `AlignmentSetBuilder::register`.)
  #[must_use]
  pub fn register(mut self, key: AlignerKey, aligner: Aligner) -> Self {
    if let AlignerKey::Lang(ref key_lang) = key {
      assert_eq!(
        aligner.language_ref(),
        key_lang,
        "AlignerKey::Lang({key_lang:?}) cannot accept an aligner built for {actual:?}; \
 register it under AlignerKey::Lang({actual:?}) or AlignerKey::Any, or rebuild the \
 aligner for the desired language",
        actual = aligner.language_ref(),
      );
    }
    self.aligners.insert(key, aligner);
    self
  }

  /// Number of currently-registered aligners.
  #[must_use]
  pub fn len(&self) -> usize {
    self.aligners.len()
  }

  /// Whether the builder has zero registered aligners.
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.aligners.is_empty()
  }

  /// Finalise into an [`AlignmentSet`].
  #[must_use]
  pub fn build(self) -> AlignmentSet {
    AlignmentSet {
      aligners: self.aligners,
      fallback: self.fallback,
    }
  }
}

impl Default for AlignmentSetBuilder {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests;
