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
//! shared across threads. It is `!Sync` for **two independent reasons**, either
//! of which alone is fatal to an `Arc<AlignmentSet>` fanned out to workers:
//!
//! 1. **The CoreML model.** Each [`Aligner`] owns an
//!    [`Encoder`](crate::encode::Encoder) → [`coremlit::Model`], which is
//!    deliberately [`Send`] but
//!    **not** [`Sync`]: Apple documents "use an `MLModel` instance on one thread
//!    or one dispatch queue at a time" (`coremlit::Model`'s `# Concurrency`), so
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

use core::sync::atomic::AtomicBool;
use std::collections::HashMap;

use asry::{
  AlignmentResult, Lang, TimeRange,
  emissions::{OovEvent, OutputClock, ResolvedOov},
};

use crate::{aligner::Aligner, error::AlignError};

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
  /// `whisperkit::log::LogLevel`): [`as_str`](Self::as_str) + a derived
  /// [`Display`](core::fmt::Display), a total
  /// [`FromStr`](core::str::FromStr) whose error
  /// ([`ParseAlignmentFallbackError`]) is opaque, `snake_case` serde under the
  /// `serde` feature, and `#[non_exhaustive]`. It is a *policy* — the kind of
  /// value that arrives from a config file, a CLI flag or an env var — so it has
  /// to survive a round trip through text, which it previously could not do in
  /// either direction even though [`crate::AlignerOptions`] is itself
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
  Hit {
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
  /// construction language is `aligner_language` — different from the request
  /// (that difference is what makes it a fallback). Policy still keys on the
  /// REQUESTED language, not on this one.
  AnyFallback {
    /// The `Any` aligner's own construction language.
    aligner_language: Lang,
  },
  /// Miss on both `Lang(L)` and `Any`: the configured [`AlignmentFallback`]
  /// decides what [`AlignmentHandle::align_chunk`] does.
  Miss {
    /// The configured miss policy.
    fallback: AlignmentFallback,
  },
}

/// A registry bound to one requested language — the guarded, request-scoped view
/// over an [`AlignmentSet`], returned by [`AlignmentSet::resolve`].
///
/// [`detect_oov`](Self::detect_oov) and [`align_chunk`](Self::align_chunk)
/// delegate to [`AlignmentSet::detect_oov`] / [`AlignmentSet::align_chunk`] under
/// the bound language, so OOV events and decision-language policy always key on
/// the REQUESTED language and an [`AlignerKey::Any`] fallback's decisions are
/// re-stamped on the crossing — the same guarantees those set methods give.
///
/// The handle deliberately exposes **no** raw `&Aligner`. Handing back the
/// aligner of an `Any` match — an English aligner serving a Chinese request, say
/// — would let a caller call `detect_oov` through it and stamp events with the
/// aligner's OWN language, or `align_chunk` through it and hit the
/// undifferentiated decision-language error the typed
/// [`AlignError::DecisionLanguage`] replaced: the exact guard bypass the registry
/// exists to make unrepresentable (F1). To learn which aligner was bound, read
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

  /// Bind this registry to a requested `language`, returning an
  /// [`AlignmentHandle`] whose [`detect_oov`](AlignmentHandle::detect_oov) and
  /// [`align_chunk`](AlignmentHandle::align_chunk) dispatch through the SAME
  /// guarded paths as [`Self::detect_oov`] / [`Self::align_chunk`]: OOV events
  /// and decision-language policy keyed on the REQUESTED `language`, an `Any`
  /// fallback's decisions re-stamped on the crossing, typed errors throughout.
  ///
  /// This is the **only** public resolver, and it never yields a raw
  /// `&Aligner`. An [`AlignerKey::Any`] aligner serving another language would
  /// otherwise stamp OOV events with ITS construction language and reproduce the
  /// generic decision-language error the typed [`AlignError::DecisionLanguage`]
  /// replaced — the guard bypass F1 closes. Ask the returned handle
  /// [`what it bound`](AlignmentHandle::binding) if you need the hit-vs-fallback
  /// metadata; that comes back as data, not as the aligner.
  ///
  /// The raw aligner-resolving primitive and its `AnyFallback` `&Aligner` are
  /// private, so the leak is unrepresentable through the public API — including
  /// from an external crate:
  ///
  /// ```compile_fail
  /// use alignkit::{AlignmentSetBuilder, Lang};
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

  /// Look up an aligner for `language`, applying the strict `Lang → Any →
  /// fallback` order. Internal aligner-carrying primitive that
  /// [`Self::resolve`], [`Self::detect_oov`] and [`Self::align_chunk`] dispatch
  /// on; not public — see [`AlignmentLookup`] for why an `Any` match's raw
  /// `&Aligner` must not escape.
  #[must_use]
  fn lookup<'a>(&'a self, language: &Lang) -> AlignmentLookup<'a> {
    let lang_key = AlignerKey::Lang(language.clone());
    if let Some(aligner) = self.aligners.get(&lang_key) {
      return AlignmentLookup::Hit { aligner };
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
      AlignmentLookup::Hit { aligner } | AlignmentLookup::AnyFallback { aligner } => aligner,
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
      AlignmentLookup::Hit { aligner } => {
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

impl AlignmentHandle<'_> {
  /// The requested language this handle is bound to. Every policy decision — the
  /// language OOV events are stamped with, the language decisions are validated
  /// against — keys on THIS, never on a fallback aligner's own construction
  /// language.
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
    match self.set.lookup(&self.language) {
      AlignmentLookup::Hit { .. } => AlignmentBinding::Exact,
      AlignmentLookup::AnyFallback { aligner } => AlignmentBinding::AnyFallback {
        aligner_language: aligner.language_ref().clone(),
      },
      AlignmentLookup::Miss { fallback } => AlignmentBinding::Miss { fallback },
    }
  }

  /// Detect OOV characters in `text`, every event stamped the REQUESTED
  /// language — the guarded [`AlignmentSet::detect_oov`] bound to this handle's
  /// language, so an [`AlignerKey::Any`] fallback's events are patched back to
  /// the request rather than left on the aligner's own language.
  ///
  /// # Errors
  /// As [`AlignmentSet::detect_oov`].
  pub fn detect_oov(&self, text: &str) -> Result<Vec<OovEvent>, AlignError> {
    self.set.detect_oov(text, &self.language)
  }

  /// Align one chunk end-to-end through the bound language — the guarded
  /// [`AlignmentSet::align_chunk`]. The requested-language decision validation,
  /// the `Any`-fallback decision crossing, and the miss policy all apply exactly
  /// as they do there; only the `language` lookup key is supplied for you.
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
    oov_decisions: &[ResolvedOov],
  ) -> Result<AlignmentResult, AlignError> {
    self.set.align_chunk(
      &self.language,
      samples,
      sub_segments,
      text,
      clock,
      abort_flag,
      oov_decisions,
    )
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
