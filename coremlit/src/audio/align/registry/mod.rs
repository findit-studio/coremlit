//! Multi-language aligner registry: [`AlignmentSet`] keyed by
//! [`AlignerKey`], built with [`AlignmentSetBuilder`].
//!
//! Semantics mirror asry's own registry
//! (`asry/src/runner/aligner/{key.rs,set.rs,builder.rs}`) — the same
//! `Lang → Any → fallback` strict lookup, the same
//! failure-never-falls-through-to-`Any` rule, decisions bound to the request
//! they were decided for — with **two deliberate divergences**:
//!
//! - This registry stores a plain [`Aligner`], not a `Mutex<Aligner>`. asry
//!   needs the mutex because its ORT `Aligner::align` is `&mut self`; alignkit's
//!   [`Aligner::align_chunk`](crate::audio::align::aligner::Aligner::align_chunk)
//!   is `&self` (the CoreML `Model` predicts without `&mut`), so there is
//!   nothing to lock.
//! - It aligns a caller's text, not a pool job. asry's pool re-stamps an
//!   [`AlignerKey::Any`] fallback's events with the job's requested language and
//!   reports a unit no aligner reads as one event the caller's policy decides;
//!   both are asry's to make, from its job, and its direct front end makes the
//!   first only for a unit job (`detect_oov_unit_as_fallback`), which only asry
//!   builds. A detection of a caller's own text carries the language of the
//!   aligner that read it, so here a [`SetDetection`] hands its policy every
//!   event as a [`SetOovEvent`], under the REQUESTED language, while the bound
//!   aligner's detection stays as asry made it, bound to the text and to that
//!   aligner. A miss is no detection at all, answered by the
//!   [`AlignmentFallback`].
//!
//! # Scope of that win
//!
//! `&self` alignment means many chunks can be aligned through one registry
//! without interior mutability — but **not** that an [`AlignmentSet`] can be
//! shared across threads. It is `!Sync` for **two independent reasons**, either
//! of which alone is fatal to an `Arc<AlignmentSet>` fanned out to workers:
//!
//! 1. **The CoreML model.** Each [`Aligner`] owns an
//!    `Encoder` → [`crate::Model`], which is
//!    deliberately [`Send`] but
//!    **not** [`Sync`]: Apple documents "use an `MLModel` instance on one thread
//!    or one dispatch queue at a time" (`crate::Model`'s `# Concurrency`), so
//!    concurrent `&Model` access from multiple threads is outside contract. This
//!    blocker is intrinsic to the model — no bound widening removes it.
//! 2. **The text normalizer.** That same [`Aligner`]'s asry `EmissionsAligner`
//!    owns a [`DynTextNormalizer`](asry::emissions::DynTextNormalizer) =
//!    `Box<dyn TextNormalizer>`, and asry declares `TextNormalizer: Send` with no
//!    `Sync` bound, so the box is not `Sync` either.
//!
//! The earlier claim that widening asry's bound to `Send + Sync` would enable
//! sharing was wrong: it removes reason 2 but not reason 1, so
//! `assert_sync::<AlignmentSet>()` still fails on the model. The real cross-thread
//! route is therefore **one model per worker** — a separate [`AlignmentSet`] per
//! thread, each `Model` independently loaded — or serializing all access to one
//! set behind an external `Mutex`. Until then the mutex-free design buys
//! single-threaded reuse and one less lock in the hot path, not free cross-thread
//! sharing.

use core::{
  num::NonZeroU64,
  sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::collections::HashMap;

use asry::{
  Lang, TimeRange,
  emissions::{
    OovDecision, OovDetection, OovEvent, OovKind, OovResolution, OutputClock, UnalignedCause,
    UnitAlignment, default_oov_policy,
  },
};

use crate::audio::align::{
  aligner::Aligner,
  error::{AlignError, DecisionLanguage, ForeignResolution, MisroutedResolution},
};

/// Identifies an aligner in the [`AlignmentSet`] registry.
///
/// Lookup order (see [`AlignmentSet::resolve`]):
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

/// Error parsing an [`AlignmentFallback`] name.
///
/// Opaque (`(())`): the rejected input is the caller's own string and carrying
/// it back adds nothing they do not already hold, while the empty payload keeps
/// the type free to grow a real one later without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown alignment fallback policy name")]
pub struct ParseAlignmentFallbackError(());

/// Defines [`AlignmentFallback`] and everything that MUST stay in lockstep with
/// its variant list — from ONE table (F2). Each `Variant => "spelling"` row
/// generates, for that variant: the enum arm, its stable wire spelling (shared
/// by [`AlignmentFallback::as_str`], the derived [`Display`](core::fmt::Display)
/// and, under `serde`, its `rename`), the total [`FromStr`](core::str::FromStr)
/// arm, and its entry in the exhaustive `AlignmentFallback::ALL` roster the
/// round-trip tests iterate.
///
/// The macro grammar REQUIRES the `=> "spelling"`, so a variant with no wire
/// form and no parser mapping cannot even be written — it fails to compile. That
/// closes the gap two independent structures left open: a hand-listed roster and
/// a wildcard-armed `FromStr` did not constrain each other, so a variant could
/// be added while BOTH the roster entry and the parser arm were forgotten —
/// leaving `Display`/serde emitting a name `from_str` then rejected, every
/// text-form test still green because the roster they iterate never grew.
macro_rules! define_alignment_fallback {
  (
    $(#[$enum_meta:meta])*
    $vis:vis enum $Name:ident {
      $(
        $(#[$variant_meta:meta])*
        $Variant:ident => $spelling:literal
      ),+ $(,)?
    }
  ) => {
    $(#[$enum_meta])*
    $vis enum $Name {
      $(
        $(#[$variant_meta])*
        #[cfg_attr(feature = "serde", serde(rename = $spelling))]
        $Variant,
      )+
    }

    impl $Name {
      /// Stable `snake_case` name of the policy — the same spelling the `serde`
      /// feature reads and writes, [`Display`](core::fmt::Display) prints, and
      /// [`FromStr`](core::str::FromStr) parses. All are generated from the same
      /// table row as this variant, so they cannot drift apart.
      #[inline(always)]
      pub const fn as_str(&self) -> &'static str {
        match self {
          $( Self::$Variant => $spelling, )+
        }
      }

      /// Every variant, in declaration order — the single generated roster the
      /// round-trip / spelling / serde tests iterate. Generated from the same
      /// table as the variants themselves, so a variant that is not in this
      /// slice cannot exist (F2): the roster can never fall behind the enum.
      #[cfg(test)]
      pub(crate) const ALL: &'static [Self] = &[$( Self::$Variant, )+];
    }

    impl core::str::FromStr for $Name {
      type Err = ParseAlignmentFallbackError;

      fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
          $( $spelling => Self::$Variant, )+
          _ => return Err(ParseAlignmentFallbackError(())),
        })
      }
    }
  };
}

define_alignment_fallback! {
  /// Policy for a requested language with no registered aligner (and no
  /// [`AlignerKey::Any`] fallback registered either).
  ///
  /// A vocabulary enum with the workspace's full contract (mirrors
  /// `coremlit::audio::whisper::log::LogLevel`): [`as_str`](Self::as_str) + a derived
  /// [`Display`](core::fmt::Display), a total
  /// [`FromStr`](core::str::FromStr) whose error
  /// ([`ParseAlignmentFallbackError`]) is opaque, `snake_case` serde under the
  /// `serde` feature, and `#[non_exhaustive]`. It is a *policy* — the kind of
  /// value that arrives from a config file, a CLI flag or an env var — so it has
  /// to survive a round trip through text, which it previously could not do in
  /// either direction even though [`crate::audio::align::AlignerOptions`] is itself
  /// serde-gated.
  ///
  /// The enum, its wire spellings, its parser and its test roster are all
  /// generated from one table by `define_alignment_fallback!`, so a new policy
  /// is a single `Variant => "spelling"` row and cannot be half-added.
  #[derive(
    Copy, Clone, PartialEq, Eq, Debug, Default, derive_more::Display, derive_more::IsVariant,
  )]
  #[display("{}", self.as_str())]
  #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
  #[non_exhaustive]
  pub enum AlignmentFallback {
    /// Skip alignment for the chunk: the caller keeps the ASR text with empty
    /// per-word timings. The default — alignment availability never blocks the
    /// pipeline.
    #[default]
    SkipChunk => "skip_chunk",
    /// Treat a registry miss as a hard error the caller must handle — useful
    /// when a missing language should be a loud signal, not a silent skip.
    Error => "error",
  }
}

/// Internal result of [`AlignmentSet::lookup`]: the aligner-carrying resolution
/// the guarded methods dispatch on. Deliberately **not** public — an
/// `AnyFallback`'s raw `&Aligner` is exactly the cross-language escape hatch
/// [`AlignmentHandle`] exists to close (F1). The only public resolver is
/// [`AlignmentSet::resolve`], which hands back a handle, never an aligner.
enum AlignmentLookup<'a> {
  /// Hit on [`AlignerKey::Lang`]`(L)`. A failure of this aligner does NOT
  /// fall through to [`AlignerKey::Any`]. The matched key is always
  /// `Lang(requested)`, so it carries no information beyond the handle's own
  /// [`language`](AlignmentHandle::language) and is not stored.
  ///
  /// Carries the language-specific aligner.
  Hit(&'a Aligner),
  /// Miss on `Lang(L)`, hit on [`AlignerKey::Any`] — the multilingual
  /// fallback is used.
  ///
  /// Carries the multilingual fallback aligner.
  AnyFallback(&'a Aligner),
  /// Miss on both `Lang(L)` and `Any`. The configured [`AlignmentFallback`]
  /// decides what the caller does.
  ///
  /// Carries the configured miss policy.
  Miss(AlignmentFallback),
}

impl AlignmentLookup<'_> {
  /// This lookup as [`AlignmentBinding`] data: the aligner it bound named by
  /// its language, never handed out.
  fn binding(&self) -> AlignmentBinding {
    match self {
      Self::Hit(_) => AlignmentBinding::Exact,
      Self::AnyFallback(aligner) => AlignmentBinding::AnyFallback(aligner.language_ref().clone()),
      Self::Miss(fallback) => AlignmentBinding::Miss(*fallback),
    }
  }
}

/// The identity of one [`AlignmentSet`]: what a [`SetDetection`] and a
/// [`SetResolution`] carry from the set that made them, and what
/// [`AlignmentSet::align_chunk`] checks before anything else.
///
/// Minted from a process-wide counter when [`AlignmentSetBuilder::build`] makes
/// a set, so it names the INSTANCE, never its contents: two sets built from
/// equal aligners and an equal policy are two sets, as two asry aligners are
/// (asry binds a detection to the aligner instance that read the text, by a
/// counter of the same kind). A fingerprint of a set's contents would equate
/// those two and still need a nonce to tell them apart; the nonce alone is the
/// identity. Its `Display` is `alignment set #n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SetId(NonZeroU64);

impl SetId {
  /// Mint the next process-unique identity.
  fn next() -> Self {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let raw = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Unreachable: exhausting this needs 2^64 sets built in one process.
    // Typed rather than wrapped, so the impossible case cannot hand out one
    // identity twice.
    Self(NonZeroU64::new(raw).expect("SetId counter overflowed u64"))
  }
}

/// `alignment set #n`.
impl core::fmt::Display for SetId {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "alignment set #{}", self.0)
  }
}

/// How [`AlignmentSet::resolve`] matched a request — the hit-vs-fallback-vs-miss
/// resolution as DATA, never a raw `&Aligner`.
///
/// Returned by [`AlignmentHandle::binding`] for a caller that needs to know
/// which aligner the registry bound (which language, hit or fallback); the
/// aligner itself stays behind the handle's guarded
/// [`detect_oov`](AlignmentHandle::detect_oov) /
/// [`align_chunk`](AlignmentHandle::align_chunk), because a raw cross-language
/// `&Aligner` is the escape hatch F1 closes.
///
/// Deliberately **not** `#[non_exhaustive]` — unlike the input vocabularies
/// [`AlignerKey`] and [`AlignmentFallback`], this is a CLOSED trichotomy: the
/// strict `Lang → Any → fallback` lookup resolves in exactly these three ways,
/// and that stays true however many key KINDS [`AlignerKey`] later grows (a new
/// key still resolves as an exact hit, the `Any` fallback, or a miss). Leaving
/// it exhaustive lets a caller `match` it without a `_` arm and compare it with
/// `==`, which is the ergonomics a result type wants.
#[derive(Clone, PartialEq, Eq, Debug, derive_more::IsVariant)]
pub enum AlignmentBinding {
  /// Exact [`AlignerKey::Lang`]`(L)` hit: the requested language has its own
  /// registered aligner. The bound language IS the request
  /// ([`AlignmentHandle::language`]).
  Exact,
  /// Miss on `Lang(L)`, served by the [`AlignerKey::Any`] fallback, whose OWN
  /// construction language this variant carries. What makes this a fallback is
  /// that the lookup found no exact [`AlignerKey::Lang`]`(L)` and fell through to
  /// [`AlignerKey::Any`] — **not** that the languages differ. They usually do (an
  /// `Any` aligner serving another language), but they need not: with an English
  /// aligner registered under [`AlignerKey::Any`] alone, an English request finds
  /// no `Lang(En)` key and resolves to `AnyFallback(En)`,
  /// matching the request (`registry::tests::any_fallback_can_match_the_requested_language`).
  /// Either way policy keys on the REQUESTED language, not on this one.
  ///
  /// Carries the `Any` aligner's own construction language — MAY equal the request
  /// (see above); it is the aligner the fallback bound, not a guarantee of
  /// difference.
  AnyFallback(Lang),
  /// Miss on both `Lang(L)` and `Any`: the configured [`AlignmentFallback`]
  /// decides what [`AlignmentHandle::align_chunk`] does.
  ///
  /// Carries the configured miss policy.
  Miss(AlignmentFallback),
}

/// A registry bound to one requested language — the guarded, request-scoped view
/// over an [`AlignmentSet`], returned by [`AlignmentSet::resolve`].
///
/// [`detect_oov`](Self::detect_oov) and [`align_chunk`](Self::align_chunk)
/// delegate to [`AlignmentSet::detect_oov`] / [`AlignmentSet::align_chunk`] under
/// the bound language, so a detection is bound to the REQUESTED language and
/// alignment refuses decisions made for another request — the same guarantees
/// those set methods give.
///
/// The handle deliberately exposes **no** raw `&Aligner`. Handing back the
/// aligner of an `Any` match — an English aligner serving a Chinese request, say
/// — would let a caller detect and align through it with no requested language
/// at all, so decisions a per-language policy made for one request could be
/// applied under another: the exact guard bypass the registry exists to make
/// unrepresentable (F1). To learn which aligner was bound, read
/// [`Self::binding`] — that is data, not an escape hatch.
pub struct AlignmentHandle<'a> {
  set: &'a AlignmentSet,
  language: Lang,
}

/// A registry of [`Aligner`]s keyed by [`AlignerKey`].
///
/// Fields are private; construct via [`AlignmentSetBuilder`]. Lookup is
/// `&self`, and — unlike asry's `Mutex`-wrapped pool — the stored aligners
/// align through `&self` too, so the whole set is usable behind a shared
/// reference with no interior mutability. It is **not** `Sync`, so that
/// shared reference cannot cross threads; see the module doc's "Scope of that
/// win".
///
/// A set cannot change once built, and it has an identity ([`Self::id`]):
/// the detections it makes and the resolutions decided from them answer it
/// alone ([`Self::align_chunk`]).
pub struct AlignmentSet {
  aligners: HashMap<AlignerKey, Aligner>,
  fallback: AlignmentFallback,
  /// This set's identity, minted when it was built.
  id: SetId,
}

impl AlignmentSet {
  /// This set's identity: the one its detections and resolutions carry.
  #[must_use]
  pub const fn id(&self) -> SetId {
    self.id
  }

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

  /// Bind this registry to a requested `language`, returning an
  /// [`AlignmentHandle`] whose [`detect_oov`](AlignmentHandle::detect_oov) and
  /// [`align_chunk`](AlignmentHandle::align_chunk) dispatch through the SAME
  /// guarded paths as [`Self::detect_oov`] / [`Self::align_chunk`]: a detection
  /// bound to the REQUESTED `language`, decisions made for another request
  /// refused, typed errors throughout.
  ///
  /// This is the **only** public resolver, and it never yields a raw
  /// `&Aligner`. The raw aligner of an [`AlignerKey::Any`] match would let a
  /// caller align decisions made for one request under another, past the
  /// requested-language binding [`AlignError::DecisionLanguage`] enforces — the
  /// guard bypass F1 closes. Ask the returned handle
  /// [`what it bound`](AlignmentHandle::binding) if you need the hit-vs-fallback
  /// metadata; that comes back as data, not as the aligner.
  ///
  /// The raw aligner-resolving primitive and its `AnyFallback` `&Aligner` are
  /// private, so the leak is unrepresentable through the public API — including
  /// from an external crate:
  ///
  /// ```compile_fail
  /// use coremlit::audio::align::{AlignmentSetBuilder, Lang};
  /// let set = AlignmentSetBuilder::new().build();
  /// // `lookup` (and its `AlignmentLookup`, whose `AnyFallback` leaked a
  /// // cross-language `&Aligner`) are private: this does NOT compile. `resolve`
  /// // is the guarded replacement.
  /// let _leak = set.lookup(&Lang::En);
  /// ```
  #[must_use]
  pub fn resolve<'a>(&'a self, language: &Lang) -> AlignmentHandle<'a> {
    AlignmentHandle {
      set: self,
      language: language.clone(),
    }
  }

  /// How a request for `language` resolves — exact hit, [`AlignerKey::Any`]
  /// fallback (carrying the bound aligner's own language), or miss — as
  /// [`AlignmentBinding`] data.
  fn binding(&self, language: &Lang) -> AlignmentBinding {
    self.lookup(language).binding()
  }

  /// Look up an aligner for `language`, applying the strict `Lang → Any →
  /// fallback` order. Internal aligner-carrying primitive that
  /// [`Self::resolve`], [`Self::detect_oov`] and [`Self::align_chunk`] dispatch
  /// on; not public — see [`AlignmentLookup`] for why an `Any` match's raw
  /// `&Aligner` must not escape.
  #[must_use]
  fn lookup<'a>(&'a self, language: &Lang) -> AlignmentLookup<'a> {
    let lang_key = AlignerKey::Lang(language.clone());
    if let Some(aligner) = self.aligners.get(&lang_key) {
      return AlignmentLookup::Hit(aligner);
    }
    if let Some(aligner) = self.aligners.get(&AlignerKey::Any) {
      return AlignmentLookup::AnyFallback(aligner);
    }
    AlignmentLookup::Miss(self.fallback)
  }

  /// Detect out-of-vocabulary characters in `text` with the aligner registered
  /// for `language` (or the [`AlignerKey::Any`] aligner), as data — no policy
  /// decision is made.
  ///
  /// Returns a [`SetDetection`] bound to this set ([`Self::id`]) and to the
  /// requested `language`: the bound aligner's detection, or none on a registry
  /// miss, which reads no text and so is never reported as a text found spelled
  /// whole. Decide it with a policy,
  /// which sees every event under `language` ([`SetOovEvent::language`]), then
  /// hand the [`SetResolution`] to [`Self::align_chunk`] with the same
  /// `language` and the same text.
  ///
  /// # Errors
  /// As [`Aligner::detect_oov`](crate::audio::align::aligner::Aligner::detect_oov),
  /// from the matched aligner.
  pub fn detect_oov(&self, text: &str, language: &Lang) -> Result<SetDetection, AlignError> {
    let detection = match self.lookup(language) {
      AlignmentLookup::Hit(aligner) | AlignmentLookup::AnyFallback(aligner) => {
        Some(aligner.detect_oov(text)?)
      }
      AlignmentLookup::Miss(_) => None,
    };
    Ok(SetDetection {
      made_by: self.id,
      language: language.clone(),
      detection,
    })
  }

  /// Align one chunk end-to-end through the aligner registered for `language`,
  /// applying the strict `Lang → Any → fallback` lookup, with the decisions the
  /// caller made for that request.
  ///
  /// This is the registry-owned counterpart to
  /// [`Aligner::align_chunk`](crate::audio::align::aligner::Aligner::align_chunk): call it
  /// on the SAME set, with the SAME `language` and text you passed to
  /// [`Self::detect_oov`] and that detection's [`SetResolution`]; the remaining
  /// arguments are
  /// [`Aligner::align_chunk`](crate::audio::align::aligner::Aligner::align_chunk)'s,
  /// forwarded unchanged.
  ///
  /// # The decisions are this set's, and the request's
  ///
  /// A `resolution` answers the set that made it, on every route: one another
  /// set made is [`AlignError::ForeignResolution`], checked first, before the
  /// route is read. Without it a set with no aligner for the language would
  /// answer another set's decisions with its miss policy — a `FailClosed`
  /// decision skipped as `Unaligned(Skipped)`, or the binding mistake reported
  /// as an unsupported language.
  ///
  /// asry binds a resolution to the text and to the aligner that detected it,
  /// and refuses any other. What it cannot see is the request: an
  /// [`AlignerKey::Any`] aligner serves every language without one of its own,
  /// so a resolution decided by one language's policy would be accepted for
  /// another. The registry binds that. A `resolution` decided for another
  /// language than `language` is [`AlignError::DecisionLanguage`], checked before
  /// any dispatch, so the same wrong input is the same typed error on every route
  /// and whatever the audio — never an [`AlignError::InputTooLong`] the
  /// aligner's length check raises first, nor a miss policy's answer.
  ///
  /// # Registry miss
  ///
  /// On a miss (no `Lang(language)`, no `Any`) the configured
  /// [`AlignmentFallback`] decides: [`AlignmentFallback::SkipChunk`] returns
  /// `Unaligned(Skipped)` (the ASR text survives, only per-word timings are
  /// dropped); [`AlignmentFallback::Error`] returns
  /// [`AlignError::LanguageUnsupported`].
  ///
  /// # Errors
  /// [`AlignError::ForeignResolution`] if another set made `resolution`, on
  /// every route; [`AlignError::DecisionLanguage`] if it was decided for another
  /// language, on every route; [`AlignError::MisroutedResolution`] if its shape
  /// is not the route's — an aligner's decisions on a miss, or none where an
  /// aligner reads the text (a set cannot change once built, so its own
  /// resolution for this language always matches).
  /// [`AlignError::LanguageUnsupported`] on a miss under
  /// [`AlignmentFallback::Error`]. Otherwise any error
  /// [`Aligner::align_chunk`](crate::audio::align::aligner::Aligner::align_chunk) itself
  /// returns — an [`AlignError::Refused`] naming the refused positions where the
  /// bound aligner's detection found them, each under the requested `language`
  /// ([`RefusedOov::language`](crate::audio::align::error::RefusedOov::language)),
  /// whichever aligner refused.
  ///
  /// With the `tracing` feature: one `alignkit.registry.align_chunk` span at
  /// `DEBUG` per call, carrying the `requested_language` and the `route` the
  /// lookup took ([`AlignmentBinding`]: an [`AlignerKey::Any`] fallback names
  /// its own language there), with the bound aligner's `alignkit.align_chunk`
  /// span, which names that aligner's `aligner_language`, nested inside it.
  // Mirrors `Aligner::align_chunk`'s argument surface (already at the 7-arg
  // limit) plus the registry's `language` lookup key, so a caller uses the exact
  // call shape they already know rather than an opaque params struct. Same
  // rationale as whisperkit's Swift-mirroring signatures.
  #[allow(clippy::too_many_arguments)]
  #[cfg_attr(
    feature = "tracing",
    tracing::instrument(
      name = "alignkit.registry.align_chunk",
      level = "debug",
      skip_all,
      fields(
        requested_language = ?language,
        route = ?self.binding(language),
      ),
    )
  )]
  pub fn align_chunk(
    &self,
    language: &Lang,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    clock: OutputClock,
    abort_flag: &AtomicBool,
    resolution: SetResolution,
  ) -> Result<UnitAlignment, AlignError> {
    // The binding first, before the route is read: a resolution answers the
    // set that made it on every route, a miss included.
    if resolution.made_by != self.id {
      return Err(AlignError::ForeignResolution(ForeignResolution::new(
        resolution.made_by,
        self.id,
      )));
    }
    if resolution.language != *language {
      return Err(AlignError::DecisionLanguage(DecisionLanguage::new(
        language.clone(),
        resolution.language,
      )));
    }
    // Then the shape, symmetric on every route: an aligner's decisions where an
    // aligner reads the text, none on a miss.
    match (self.lookup(language), resolution.resolution) {
      (AlignmentLookup::Hit(aligner) | AlignmentLookup::AnyFallback(aligner), Some(resolution)) => {
        aligner
          .align_chunk(samples, sub_segments, text, clock, abort_flag, resolution)
          .map_err(|error| for_request(error, language))
      }
      (AlignmentLookup::Miss(fallback), None) => match fallback {
        AlignmentFallback::SkipChunk => Ok(UnitAlignment::Unaligned(UnalignedCause::Skipped)),
        AlignmentFallback::Error => Err(AlignError::LanguageUnsupported(language.clone())),
      },
      (lookup, decided) => Err(AlignError::MisroutedResolution(MisroutedResolution::new(
        lookup.binding(),
        decided.is_some(),
      ))),
    }
  }
}

/// What an [`AlignmentSet`] read in one text for one requested language: the
/// OOV detection of the aligner that language resolves to, or none when no
/// aligner reads it.
///
/// The one way to decide what the registry detected: [`Self::decide`] makes the
/// [`SetResolution`] [`AlignmentSet::align_chunk`] takes, bound to the requested
/// language. The bound aligner's detection is asry's
/// [`OovDetection`], bound to the text and to that
/// aligner, so its decisions apply there alone.
///
/// # Every event under the requested language
///
/// asry stamps a detection's events with the language of the aligner that read
/// the text: an [`AlignerKey::Any`] fallback's own construction language, when
/// the request fell through to it. A policy keyed on that stamp would judge a
/// Korean request by an English fallback's rules — wildcard where Korean fails
/// closed — and the wildcard would answer a plausible, wrong timing. asry
/// stamps the requested language itself only on its unit road
/// (`EmissionsAligner::detect_oov_unit_as_fallback`), for a unit job only asry
/// builds, so it offers no road for a caller's own text. The registry therefore
/// shows its policy, and every reader of its events, each event as a
/// [`SetOovEvent`]: the position asry detected, under [`Self::language`]. The
/// detection itself is untouched, so the resolution stays bound to the text and
/// to the aligner that read it; its `Debug` shows the requested language and
/// the events under it, never asry's stamped detection.
#[must_use = "a detection does nothing until it is decided"]
pub struct SetDetection {
  /// The set that made this detection.
  made_by: SetId,
  /// The requested language.
  language: Lang,
  /// The bound aligner's detection; `None` on a registry miss.
  detection: Option<OovDetection>,
}

impl SetDetection {
  /// The set that made this detection: the one its resolution answers.
  #[must_use]
  pub const fn made_by(&self) -> SetId {
    self.made_by
  }

  /// The requested language: the language every event is judged under.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// The events the bound aligner found, under the requested language, in the
  /// order its tokenizer meets them, or `None` when no aligner read the text (a
  /// registry miss). An empty list is a text read and found spelled whole; a
  /// miss is never reported so.
  #[must_use]
  pub fn events(&self) -> Option<Vec<SetOovEvent<'_>>> {
    self.detection.as_ref().map(|detection| {
      detection
        .events()
        .iter()
        .map(|event| SetOovEvent::new(event, &self.language))
        .collect()
    })
  }

  /// Decide every event with `policy`, in order, into the resolution
  /// [`AlignmentSet::align_chunk`] takes for the requested language. `policy`
  /// sees each event under the requested language, whichever aligner read the
  /// text. A miss has no event to decide: the registry's [`AlignmentFallback`]
  /// answers it.
  pub fn decide(self, mut policy: impl FnMut(&SetOovEvent<'_>) -> OovDecision) -> SetResolution {
    let language = self.language;
    let resolution = self
      .detection
      .map(|detection| detection.decide(|event| policy(&SetOovEvent::new(event, &language))));
    SetResolution {
      made_by: self.made_by,
      language,
      resolution,
    }
  }
}

/// The requested language and the events under it ([`SetDetection::events`]):
/// asry's detection stays behind, since its events carry the stamp of the
/// aligner that read the text.
impl core::fmt::Debug for SetDetection {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("SetDetection")
      .field("made_by", &self.made_by)
      .field("language", &self.language)
      .field("events", &self.events())
      .finish()
  }
}

/// One OOV event of a [`SetDetection`] or a [`SetResolution`], as the registry
/// shows it: the position the bound aligner's detection found, under the
/// REQUESTED language.
///
/// It mirrors asry's [`OovEvent`] reader for reader, with one difference:
/// [`Self::language`] is the request's, never the language asry stamped the
/// event with, which for an [`AlignerKey::Any`] fallback is the fallback's own
/// (see [`SetDetection`]'s "Every event under the requested language"). The
/// stamped event is not reachable from it, so no reader of this view can judge
/// the event under the fallback's language. Its equality is the position and the
/// language shown.
#[derive(Clone, Copy)]
pub struct SetOovEvent<'a> {
  /// The event as asry detected it.
  event: &'a OovEvent,
  /// The requested language.
  language: &'a Lang,
}

impl<'a> SetOovEvent<'a> {
  /// `event`, under the requested `language`.
  const fn new(event: &'a OovEvent, language: &'a Lang) -> Self {
    Self { event, language }
  }

  /// What kind of wildcard-generating position this is.
  #[must_use]
  pub const fn kind(&self) -> &'a OovKind {
    self.event.kind()
  }

  /// Zero-based char index in the chunk's normalized text.
  #[must_use]
  pub const fn char_index(&self) -> usize {
    self.event.char_index()
  }

  /// Zero-based word index (separator-counted).
  #[must_use]
  pub const fn word_index(&self) -> usize {
    self.event.word_index()
  }

  /// The offending character when the kind is `Symbol` or `InternalPunct`;
  /// `None` for `BoundaryPunct`, whose mark the normalizer removed, and for
  /// `NotInspected`.
  #[must_use]
  pub fn char(&self) -> Option<char> {
    self.event.char()
  }

  /// The requested language: the language a policy judges this event under.
  #[must_use]
  pub const fn language(&self) -> &'a Lang {
    self.language
  }

  /// asry's [`default_oov_policy`] decision for this event. That policy decides
  /// by an event's kind and character alone (asry 0.3's `core/oov.rs`), so the
  /// language does not enter it: a per-language policy over this view can fall
  /// back to it, as asry's documentation suggests for its own events.
  #[must_use]
  pub fn default_decision(&self) -> OovDecision {
    default_oov_policy(self.event)
  }
}

impl PartialEq for SetOovEvent<'_> {
  fn eq(&self, other: &Self) -> bool {
    self.event.matches_position(other.event) && self.language == other.language
  }
}

impl Eq for SetOovEvent<'_> {}

/// The position and the language shown, never the language asry stamped.
impl core::fmt::Debug for SetOovEvent<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("SetOovEvent")
      .field("kind", self.kind())
      .field("char_index", &self.char_index())
      .field("word_index", &self.word_index())
      .field("language", self.language)
      .finish()
  }
}

/// One decided event of a [`SetResolution`]: the event under the requested
/// language, and the decision the policy made for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetResolvedOov<'a> {
  /// The event, under the requested language.
  event: SetOovEvent<'a>,
  /// The policy's decision.
  decision: OovDecision,
}

impl<'a> SetResolvedOov<'a> {
  /// The event the decision was made for, under the requested language.
  #[must_use]
  pub const fn event(&self) -> SetOovEvent<'a> {
    self.event
  }

  /// The policy's decision for this position.
  #[must_use]
  pub const fn decision(&self) -> OovDecision {
    self.decision
  }
}

/// A decided [`SetDetection`]: the only form in which OOV decisions reach
/// [`AlignmentSet::align_chunk`], bound to the language they were decided for.
///
/// It cannot be cloned, and alignment consumes it, so its decisions apply once.
/// It keeps asry's resolution, which alone carries the binding to the text and
/// the aligner that read it; its `Debug` shows the requested language and the
/// decided events under it, never that resolution's stamped events.
#[must_use = "a resolution does nothing until alignment applies it"]
pub struct SetResolution {
  /// The set that made the detection these decisions were made from.
  made_by: SetId,
  /// The requested language the decisions were made for.
  language: Lang,
  /// The bound aligner's resolution; `None` on a registry miss.
  resolution: Option<OovResolution>,
}

impl SetResolution {
  /// The set these decisions answer: the one whose detection they were made
  /// from.
  #[must_use]
  pub const fn made_by(&self) -> SetId {
    self.made_by
  }

  /// The requested language the decisions were made for.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// Every event paired with its decision, under the requested language, in the
  /// order the tokenizer meets them, or `None` for a registry miss.
  #[must_use]
  pub fn resolved(&self) -> Option<Vec<SetResolvedOov<'_>>> {
    self.resolution.as_ref().map(|resolution| {
      resolution
        .resolved()
        .iter()
        .map(|resolved| SetResolvedOov {
          event: SetOovEvent::new(resolved.event(), &self.language),
          decision: resolved.decision(),
        })
        .collect()
    })
  }
}

/// The requested language and the decided events under it
/// ([`SetResolution::resolved`]): asry's resolution stays behind, since its
/// events carry the stamp of the aligner that read the text.
impl core::fmt::Debug for SetResolution {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("SetResolution")
      .field("made_by", &self.made_by)
      .field("language", &self.language)
      .field("resolved", &self.resolved())
      .finish()
  }
}

/// `error` as the registry returns it for a request in `language`: a refusal's
/// positions under the requested language, as every event the registry shows
/// is. The aligner that refused names them under its own, which on an
/// [`AlignerKey::Any`] fallback is the fallback's.
fn for_request(error: AlignError, language: &Lang) -> AlignError {
  match error {
    AlignError::Refused(refusal) => AlignError::Refused(refusal.under(language)),
    other => other,
  }
}

impl AlignmentHandle<'_> {
  /// The requested language this handle is bound to. Every policy decision — the
  /// language each OOV event is shown under ([`SetOovEvent::language`]), the
  /// language decisions are validated against — keys on THIS, never on a
  /// fallback aligner's own construction language.
  #[must_use]
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// How the registry resolved this request — exact hit, [`AlignerKey::Any`]
  /// fallback (carrying the bound aligner's own language), or miss (carrying the
  /// policy) — as [`AlignmentBinding`] DATA. It never yields the aligner itself;
  /// that stays behind [`Self::detect_oov`] / [`Self::align_chunk`] (F1).
  #[must_use]
  pub fn binding(&self) -> AlignmentBinding {
    self.set.binding(&self.language)
  }

  /// Detect OOV characters in `text` with the bound aligner — the guarded
  /// [`AlignmentSet::detect_oov`] bound to this handle's language, so the
  /// detection names the request, never an [`AlignerKey::Any`] fallback's own
  /// language.
  ///
  /// # Errors
  /// As [`AlignmentSet::detect_oov`].
  pub fn detect_oov(&self, text: &str) -> Result<SetDetection, AlignError> {
    self.set.detect_oov(text, &self.language)
  }

  /// Align one chunk end-to-end through the bound language — the guarded
  /// [`AlignmentSet::align_chunk`]. The requested-language binding of the
  /// decisions and the miss policy apply exactly as they do there; only the
  /// `language` lookup key is supplied for you.
  ///
  /// # Errors
  /// As [`AlignmentSet::align_chunk`].
  // Mirrors `AlignmentSet::align_chunk`'s argument surface minus the `language`
  // this handle already carries — the same 7-arg shape a caller of the set
  // method already knows.
  #[allow(clippy::too_many_arguments)]
  pub fn align_chunk(
    &self,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    clock: OutputClock,
    abort_flag: &AtomicBool,
    resolution: SetResolution,
  ) -> Result<UnitAlignment, AlignError> {
    self.set.align_chunk(
      &self.language,
      samples,
      sub_segments,
      text,
      clock,
      abort_flag,
      resolution,
    )
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

  /// Finalise into an [`AlignmentSet`] with a fresh identity ([`SetId`]).
  #[must_use]
  pub fn build(self) -> AlignmentSet {
    AlignmentSet {
      aligners: self.aligners,
      fallback: self.fallback,
      id: SetId::next(),
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
