use super::*;

use asry::{
  emissions::{EnglishNormalizer, OovDecision, OovKind, default_oov_decisions},
  time::ANALYSIS_TIMEBASE,
};

// ---------------------------------------------------------------------
// Hermetic: registry key / fallback / miss semantics need no aligner.
// ---------------------------------------------------------------------

#[test]
fn aligner_key_distinguishes_lang_from_any() {
  assert_ne!(AlignerKey::Lang(Lang::En), AlignerKey::Any);
  assert_eq!(AlignerKey::Lang(Lang::En), AlignerKey::Lang(Lang::En));
  assert_ne!(AlignerKey::Lang(Lang::En), AlignerKey::Lang(Lang::Zh));
}

#[test]
fn aligner_key_hashes_consistently() {
  use std::collections::HashSet;
  let mut set = HashSet::new();
  set.insert(AlignerKey::Lang(Lang::En));
  set.insert(AlignerKey::Any);
  set.insert(AlignerKey::Lang(Lang::En)); // duplicate
  assert_eq!(set.len(), 2);
}

#[test]
fn fallback_default_is_skip_chunk() {
  assert_eq!(AlignmentFallback::default(), AlignmentFallback::SkipChunk);
}

// ---------------------------------------------------------------------
// The enum contract (mirrors `whisperkit::log::LogLevel`): as_str, a Display
// derived FROM as_str, a TOTAL FromStr, an opaque parse error, snake_case
// serde, and IsVariant. A fallback policy arrives from a config file or a CLI
// flag, so it has to survive a round trip through text — which it could not do
// in either direction before.
// ---------------------------------------------------------------------

#[test]
fn fallback_round_trips_through_its_own_text_form() {
  // THE contract, as one property: for every variant, `as_str` → `from_str` is
  // the identity. Written as an exhaustive slice rather than a variant-by-
  // variant assertion so that adding a variant without a `from_str` arm fails
  // here instead of silently becoming unparseable.
  for fallback in [AlignmentFallback::SkipChunk, AlignmentFallback::Error] {
    let text = fallback.as_str();
    assert_eq!(
      text.parse::<AlignmentFallback>(),
      Ok(fallback),
      "`{text}` must parse back to the variant that produced it"
    );
    // Display is DERIVED from as_str (`#[display("{}", self.as_str())]`), so
    // the two can never drift apart — this pins that they are wired that way.
    assert_eq!(fallback.to_string(), text);
  }
}

#[test]
fn fallback_from_str_names_the_snake_case_spelling() {
  // The exact spellings, pinned: they are the serde wire form and the CLI/env
  // form, so a rename is a breaking change and must be a deliberate one.
  assert_eq!(
    "skip_chunk".parse::<AlignmentFallback>(),
    Ok(AlignmentFallback::SkipChunk)
  );
  assert_eq!(
    "error".parse::<AlignmentFallback>(),
    Ok(AlignmentFallback::Error)
  );
}

#[test]
fn fallback_from_str_is_total_and_rejects_everything_else() {
  // Total: every input has an answer, and an unknown one is an Err rather than
  // a panic or a silent default. A policy that quietly defaulted to SkipChunk
  // on a typo'd config value is exactly the failure this crate exists not to
  // have.
  for unknown in [
    "",
    "SkipChunk",
    "skip-chunk",
    "Error",
    "skip_chunk ",
    "fail",
  ] {
    assert!(
      unknown.parse::<AlignmentFallback>().is_err(),
      "`{unknown}` must not parse"
    );
  }
}

#[test]
fn fallback_is_variant_predicates() {
  assert!(AlignmentFallback::SkipChunk.is_skip_chunk());
  assert!(!AlignmentFallback::SkipChunk.is_error());
  assert!(AlignmentFallback::Error.is_error());
}

#[test]
fn aligner_key_is_variant_predicates() {
  assert!(AlignerKey::Lang(Lang::En).is_lang());
  assert!(!AlignerKey::Lang(Lang::En).is_any());
  assert!(AlignerKey::Any.is_any());
}

#[cfg(feature = "serde")]
#[test]
fn fallback_serde_uses_the_same_snake_case_spelling() {
  // One spelling across `as_str`, `FromStr` and serde, or the text form is not
  // a round trip at all — a config file that serializes `skip_chunk` and a CLI
  // that parses `SkipChunk` is two vocabularies wearing one type.
  let json = serde_json::to_string(&AlignmentFallback::SkipChunk).unwrap();
  assert_eq!(json, r#""skip_chunk""#);
  assert_eq!(
    json,
    format!("\"{}\"", AlignmentFallback::SkipChunk.as_str())
  );

  let back: AlignmentFallback = serde_json::from_str(r#""error""#).unwrap();
  assert_eq!(back, AlignmentFallback::Error);
  assert!(serde_json::from_str::<AlignmentFallback>(r#""SkipChunk""#).is_err());
}

#[test]
fn empty_set_misses_with_default_fallback() {
  let set = AlignmentSetBuilder::new().build();
  assert!(set.is_empty());
  assert_eq!(set.len(), 0);
  match set.lookup(&Lang::En) {
    AlignmentLookup::Miss { fallback } => assert_eq!(fallback, AlignmentFallback::SkipChunk),
    _ => panic!("expected Miss"),
  }
}

#[test]
fn empty_set_misses_with_error_fallback() {
  let set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::Error)
    .build();
  assert_eq!(set.fallback(), AlignmentFallback::Error);
  match set.lookup(&Lang::Zh) {
    AlignmentLookup::Miss { fallback } => assert_eq!(fallback, AlignmentFallback::Error),
    _ => panic!("expected Miss"),
  }
}

#[test]
fn builder_set_fallback_in_place() {
  let mut builder = AlignmentSetBuilder::new();
  assert!(builder.is_empty());
  builder.set_fallback(AlignmentFallback::Error);
  assert_eq!(builder.build().fallback(), AlignmentFallback::Error);
}

#[test]
fn empty_set_detect_oov_on_miss_is_empty() {
  let set = AlignmentSetBuilder::new().build();
  assert!(set.detect_oov("anything", &Lang::En).unwrap().is_empty());
}

// ---------------------------------------------------------------------
// F2: registry-owned alignment orchestration. The language crossing and the
// miss policy are hermetic (no aligner, no model); the end-to-end proof that
// alignment reaches encoding without a decision-language error is model-gated
// below.
// ---------------------------------------------------------------------

#[test]
fn cross_decisions_restamps_language_and_preserves_the_decision() {
  // Decisions the caller resolved for the REQUESTED language (Zh) — a wildcard
  // and a fail-closed, so this proves the DECISION content survives the crossing,
  // not merely that it does not error.
  let decisions = vec![
    ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('4'), 3, 1, Lang::Zh),
      OovDecision::Wildcard,
    ),
    ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('&'), 7, 2, Lang::Zh),
      OovDecision::FailClosed,
    ),
  ];
  // Cross into the bound Any-fallback aligner's OWN language (En).
  let crossed = cross_decisions_into(&decisions, &Lang::Zh, &Lang::En).expect("valid crossing");
  assert_eq!(crossed.len(), 2);
  for (crossed, original) in crossed.iter().zip(&decisions) {
    // The language tag is crossed to the aligner's...
    assert_eq!(crossed.event().language(), &Lang::En);
    // ...but the positional identity and the caller's decision are untouched, so
    // asry applies exactly the per-Zh policy at the same position.
    assert!(crossed.event().matches_position(original.event()));
    assert_eq!(crossed.decision(), original.decision());
  }
  assert_eq!(crossed[0].decision(), OovDecision::Wildcard);
  assert_eq!(crossed[1].decision(), OovDecision::FailClosed);
}

#[test]
fn cross_decisions_rejects_a_decision_not_carrying_the_requested_language() {
  // A decision stamped En handed to a Zh request: crossing it would silently
  // apply En policy under a Zh request, so it is rejected BEFORE any re-stamp —
  // the check that keeps the crossing from becoming a wrong-policy path.
  let decisions = vec![
    ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('4'), 0, 0, Lang::Zh),
      OovDecision::Wildcard,
    ),
    ResolvedOov::new(
      OovEvent::new(OovKind::Symbol('&'), 1, 0, Lang::En),
      OovDecision::FailClosed,
    ),
  ];
  let err = cross_decisions_into(&decisions, &Lang::Zh, &Lang::En).unwrap_err();
  assert!(matches!(
    err,
    AlignError::DecisionLanguage { index, ref requested, ref found }
      if index == 1 && *requested == Lang::Zh && *found == Lang::En
  ));
}

#[test]
fn cross_decisions_same_language_is_a_validated_clone() {
  let decisions = vec![ResolvedOov::new(
    OovEvent::new(OovKind::Symbol('4'), 0, 0, Lang::En),
    OovDecision::Wildcard,
  )];
  let crossed = cross_decisions_into(&decisions, &Lang::En, &Lang::En).expect("no-op crossing");
  assert_eq!(crossed, decisions);
}

#[test]
fn align_chunk_miss_skip_chunk_returns_empty_words() {
  // Empty registry, default SkipChunk policy: a miss is not an error, it drops
  // the timings and keeps going. No aligner is touched, so this is hermetic.
  let set = AlignmentSetBuilder::new().build();
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let result = set
    .align_chunk(&Lang::Zh, &[], &[], "anything", clock, &abort, &[])
    .expect("a SkipChunk miss is not an error");
  assert!(result.words().is_empty());
}

#[test]
fn align_chunk_miss_error_returns_language_unsupported() {
  let set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::Error)
    .build();
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let err = set
    .align_chunk(&Lang::Zh, &[], &[], "anything", clock, &abort, &[])
    .unwrap_err();
  assert!(matches!(
    err,
    AlignError::LanguageUnsupported { ref language } if *language == Lang::Zh
  ));
}

// ---------------------------------------------------------------------
// Model-gated: populated lookup / register / detect_oov need a real
// Aligner, which loads the CoreML model (ALIGNKIT_TEST_MODELS). Same
// convention as src/encode/tests.rs (a separate `tests/` integration
// crate is unreachable from these src-level unit tests).
// ---------------------------------------------------------------------

fn models_dir() -> std::path::PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("alignkit")
    },
    std::path::PathBuf::from,
  )
}

/// Loads the real model as an En aligner on the SHIPPING compute placement
/// (`AlignerOptions::new()` → `DEFAULT_ENCODER_COMPUTE`), never a hardcoded
/// `ComputeUnits::_` — so these tests exercise the default rather than a
/// configuration no user runs.
fn en_aligner() -> Aligner {
  Aligner::from_paths(
    Lang::En,
    &models_dir().join("base960h_aligner.mlmodelc"),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("load base960h_aligner.mlmodelc as an En aligner (set ALIGNKIT_TEST_MODELS)")
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn lookup_hits_registered_language() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Lang(Lang::En), en_aligner())
    .build();
  assert_eq!(set.len(), 1);
  match set.lookup(&Lang::En) {
    AlignmentLookup::Hit { matched, .. } => assert_eq!(matched, AlignerKey::Lang(Lang::En)),
    _ => panic!("expected Hit"),
  }
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn lookup_misses_unregistered_language_without_any() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Lang(Lang::En), en_aligner())
    .build();
  assert!(matches!(
    set.lookup(&Lang::Zh),
    AlignmentLookup::Miss {
      fallback: AlignmentFallback::SkipChunk
    }
  ));
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn strict_lookup_prefers_lang_over_any() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Lang(Lang::En), en_aligner())
    .register(AlignerKey::Any, en_aligner())
    .build();
  // A registered language hits its own aligner, never the Any fallback.
  assert!(matches!(set.lookup(&Lang::En), AlignmentLookup::Hit { .. }));
  // An unregistered language falls through to Any.
  assert!(matches!(
    set.lookup(&Lang::Zh),
    AlignmentLookup::AnyFallback { .. }
  ));
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
#[should_panic(expected = "cannot accept an aligner built for")]
fn register_panics_on_language_mismatch() {
  let _ = AlignmentSetBuilder::new().register(AlignerKey::Lang(Lang::Zh), en_aligner());
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn detect_oov_patches_language_on_any_fallback() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Any, en_aligner())
    .build();
  // An `Any`-registered En aligner serving a Zh request: every event must
  // carry the REQUESTED language (Zh), not the aligner's construction
  // language (En) — otherwise per-language OOV policy keys on the wrong one.
  let events = set
    .detect_oov("hello, world", &Lang::Zh)
    .expect("detect_oov on the Any-fallback aligner");
  assert!(!events.is_empty(), "the comma should yield an OOV event");
  assert!(events.iter().all(|event| event.language() == &Lang::Zh));
}

/// **The F2 regression, end-to-end.** An English aligner registered as the
/// multilingual [`AlignerKey::Any`] fallback, a Chinese request, real speech,
/// and a real punctuation OOV (the jfk transcript's commas).
///
/// Before the registry-owned crossing this hard-failed: [`AlignmentSet::detect_oov`]
/// stamps the events Zh so per-language policy keys on the request, but the En
/// aligner's `EmissionsAligner::prepare` validates decisions against its OWN En,
/// so the Zh-stamped decisions were rejected with a decision-language
/// `Tokenization` error the moment any OOV was present — `Any`-fallback
/// alignment was unusable with OOV decisions.
///
/// Now `align_chunk` validates the decisions carry the requested Zh and
/// re-stamps them to the aligner's En before aligning, so alignment reaches
/// encoding and produces words. Stripping the re-stamp in `cross_decisions_into`
/// (passing the decisions through unchanged) turns the `.expect` below back into
/// `Err(Alignment(Tokenization))` — the mutation proof.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn any_fallback_aligns_a_cross_language_request_with_punctuation_oov() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Any, en_aligner())
    .build();
  let samples = load_jfk_wav();
  let text = JFK_TRANSCRIPT;

  // Policy selection observes the REQUESTED language (Zh), not the fallback
  // aligner's En.
  let events = set
    .detect_oov(text, &Lang::Zh)
    .expect("detect_oov on the Any fallback");
  assert!(
    !events.is_empty(),
    "the jfk transcript's commas must yield OOV events"
  );
  assert!(
    events.iter().all(|e| e.language() == &Lang::Zh),
    "policy selection must observe the requested language"
  );
  let decisions = default_oov_decisions(&events);

  // Alignment reaches encoding WITHOUT a decision-language error and produces
  // words — the requested-language OOV policy is preserved through the crossing.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let result = set
    .align_chunk(
      &Lang::Zh,
      &samples,
      &whole_chunk_is_speech(&samples),
      text,
      clock,
      &abort,
      &decisions,
    )
    .expect("Any-fallback alignment must not fail on the decision language");
  assert!(
    !result.words().is_empty(),
    "the English Any aligner must align English speech to English words"
  );
}

/// **The F2 regression: the exact-hit path validates the decision language too.**
/// An exact [`AlignerKey::Lang`]`(Lang::En)` hit handed a decision stamped
/// [`Lang::Zh`]. The request IS En, so a Zh decision is a wrong-language payload
/// that must surface as the typed [`AlignError::DecisionLanguage`] — at the SAME
/// precedence the `Any` route gives it, BEFORE any dispatch — on this route too.
///
/// Two calls pin both ways the un-validated exact-hit path used to leak:
/// - **in-window** audio: without the fix the decisions reach the bound aligner
///   and asry's `prepare` rejects the Zh tag as an undifferentiated
///   [`AlignError::Alignment`] (its `Tokenization`) — the finding's headline;
/// - **oversized** audio: without the fix
///   [`Aligner::align_chunk`](crate::aligner::Aligner::align_chunk)'s own length
///   check raises [`AlignError::InputTooLong`] before `prepare` even runs, so the
///   SAME wrong input produced a DIFFERENT error depending on the audio length.
///
/// With the fix both are the identical typed [`AlignError::DecisionLanguage`],
/// because [`AlignmentSet::align_chunk`] validates the decision language ahead of
/// dispatching to either. Deleting the `validate_decisions_language` call on the
/// Hit path restores the two route-dependent errors above — the mutation proof.
///
/// This MUST go through [`AlignmentSet::align_chunk`]: the crossing tests call
/// `cross_decisions_into` directly (only the `Any` path) and the e2e test above
/// supplies valid decisions, so neither exercises the exact-hit validator.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn exact_hit_validates_decision_language_before_dispatch() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Lang(Lang::En), en_aligner())
    .build();
  // A Zh-stamped decision under an En request: the exact En hit's decisions
  // should carry En, so this is the wrong-language payload the validator catches.
  let decisions = vec![ResolvedOov::new(
    OovEvent::new(OovKind::Symbol('4'), 0, 0, Lang::Zh),
    OovDecision::Wildcard,
  )];
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);

  let assert_decision_language = |samples: &[f32], case: &str| {
    let err = set
      .align_chunk(&Lang::En, samples, &[], "test", clock, &abort, &decisions)
      .expect_err("a Zh decision under an En request must be rejected");
    assert!(
      matches!(
        err,
        AlignError::DecisionLanguage { index, ref requested, ref found }
          if index == 0 && *requested == Lang::En && *found == Lang::Zh
      ),
      "{case}: exact En hit + Zh decision must be the typed DecisionLanguage, got {err:?}"
    );
  };

  // In-window (1 s): the mutation would surface asry's generic Alignment here.
  let in_window = vec![0.0f32; 16_000];
  assert_decision_language(&in_window, "in-window");

  // Oversized (window + 1): the mutation would surface InputTooLong here, since
  // `Aligner::align_chunk`'s length check runs before `prepare` — so this pins
  // that the validator precedes even that earliest error.
  let oversized = vec![0.0f32; crate::encode::ENCODER_WINDOW_SAMPLES + 1];
  assert_decision_language(&oversized, "oversized");
}

/// The known transcript for `jfk.wav`, with the commas that make the F2 test's
/// punctuation OOV real (duplicated from `tests/common`, as the other src-level
/// unit tests duplicate their fixtures — a `tests/` module is unreachable here).
const JFK_TRANSCRIPT: &str = "And so my fellow Americans ask not what your country can do for you, \
                              ask what you can do for your country.";

/// "No VAD" — one span over the whole chunk in the 1/16000 analysis timebase
/// (empty would mean "all silence" and drop every word; see
/// [`crate::aligner::Aligner::align_chunk`]).
fn whole_chunk_is_speech(samples: &[f32]) -> [TimeRange; 1] {
  [TimeRange::new(0, samples.len() as i64, ANALYSIS_TIMEBASE)]
}

/// The 11 s `jfk.wav` fixture, borrowed from the whisperkit crate by relative
/// path (as the other src-level tests do) and failing LOUDLY if it ever moves.
fn load_jfk_wav() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("../whisperkit/tests/fixtures/audio/jfk.wav");
  let mut reader = hound::WavReader::open(&path)
    .unwrap_or_else(|e| panic!("open the jfk.wav fixture at {path:?}: {e}"));
  assert_eq!(reader.spec().sample_rate, 16_000, "fixture must be 16 kHz");
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}
