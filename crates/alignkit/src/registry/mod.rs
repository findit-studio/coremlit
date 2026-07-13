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

use std::collections::HashMap;

use asry::{Lang, emissions::OovEvent};

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
