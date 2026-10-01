use super::*;

use asry::{
  emissions::{EnglishNormalizer, default_oov_policy, fail_closed_all_policy},
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
// The enum contract (mirrors `coremlit::audio::whisper::log::LogLevel`): as_str, a Display
// derived FROM as_str, a TOTAL FromStr, an opaque parse error, snake_case
// serde, and IsVariant. A fallback policy arrives from a config file or a CLI
// flag, so it has to survive a round trip through text — which it could not do
// in either direction before.
// ---------------------------------------------------------------------

// The roster the round-trip / spelling / serde tests below iterate is
// `AlignmentFallback::ALL`, GENERATED from the same one-row-per-variant table as
// the enum, its spellings and its parser (`define_alignment_fallback!`, F2). It
// can no longer fall behind the enum by construction — a variant that is not in
// `ALL` cannot exist — so the hand-listed roster and the compiler-checked
// exhaustive match this file used to carry (two structures that did not
// constrain each other, the finding) are gone: the macro grammar now rejects a
// variant with no spelling/parser at COMPILE time, which no test could.

#[test]
fn fallback_round_trips_through_its_own_text_form() {
  // THE contract, as one property over the exhaustive roster: for every variant,
  // `as_str` → `from_str` is the identity. Iterating ALL_FALLBACKS (kept complete
  // by `fallback_roster_is_exhaustive`) rather than a local slice means a new
  // variant added without a `from_str` arm fails here instead of silently becoming
  // unparseable.
  for &fallback in AlignmentFallback::ALL {
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
  // The exact spellings, pinned per variant: they are the serde wire form and the
  // CLI/env form, so a rename is a breaking change and must be a deliberate one.
  // The wildcard-free `match` is the F3 tripwire — a new variant must pin its own
  // spelling here — and the loop over ALL_FALLBACKS checks every variant both
  // spells to and parses from that literal.
  for &fallback in AlignmentFallback::ALL {
    let spelling = match fallback {
      AlignmentFallback::SkipChunk => "skip_chunk",
      AlignmentFallback::Error => "error",
    };
    assert_eq!(
      fallback.as_str(),
      spelling,
      "as_str spelling for {fallback:?} drifted from its pinned wire form"
    );
    assert_eq!(
      spelling.parse::<AlignmentFallback>(),
      Ok(fallback),
      "`{spelling}` must parse back to {fallback:?}"
    );
  }
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
  // One spelling across `as_str`, `FromStr` and serde for EVERY variant, or the
  // text form is not a round trip at all — a config file that serializes
  // `skip_chunk` and a CLI that parses `SkipChunk` is two vocabularies wearing one
  // type. The wildcard-free `match` is the F3 tripwire (a new variant must pin its
  // JSON here); the loop checks each variant serializes to that literal, matches
  // `as_str`, and deserializes back.
  for &fallback in AlignmentFallback::ALL {
    let expected_json = match fallback {
      AlignmentFallback::SkipChunk => r#""skip_chunk""#,
      AlignmentFallback::Error => r#""error""#,
    };
    let json = serde_json::to_string(&fallback).unwrap();
    assert_eq!(
      json, expected_json,
      "serde spelling for {fallback:?} drifted"
    );
    assert_eq!(json, format!("\"{}\"", fallback.as_str()));
    let back: AlignmentFallback = serde_json::from_str(&json).unwrap();
    assert_eq!(
      back, fallback,
      "{fallback:?} must deserialize back from its own JSON"
    );
  }
  // A non-snake_case spelling must NOT deserialize — the rename guard, unchanged.
  assert!(serde_json::from_str::<AlignmentFallback>(r#""SkipChunk""#).is_err());
}

#[test]
fn empty_set_misses_with_default_fallback() {
  let set = AlignmentSetBuilder::new().build();
  assert!(set.is_empty());
  assert_eq!(set.len(), 0);
  match set.lookup(&Lang::En) {
    AlignmentLookup::Miss(fallback) => assert_eq!(fallback, AlignmentFallback::SkipChunk),
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
    AlignmentLookup::Miss(fallback) => assert_eq!(fallback, AlignmentFallback::Error),
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

/// **A miss reads no text, so it is never reported as a text found spelled
/// whole.** The detection names the requested language and holds no events at
/// all — `None`, not an empty list — and deciding it decides nothing.
#[test]
fn empty_set_detect_oov_on_miss_reads_nothing() {
  let set = AlignmentSetBuilder::new().build();
  let detection = set.detect_oov("anything", &Lang::En).unwrap();
  assert_eq!(detection.language(), &Lang::En);
  assert!(detection.events().is_none());
  let resolution = detection.decide(fail_closed_all_policy);
  assert_eq!(resolution.language(), &Lang::En);
  assert!(resolution.resolved().is_none());
}

// ---------------------------------------------------------------------
// F2: registry-owned alignment orchestration. The request binding and the
// miss policy are hermetic (no aligner, no model); the end-to-end proof that a
// cross-language request reaches encoding is model-gated below.
// ---------------------------------------------------------------------

/// **A resolution decided for another request is refused, on every route,
/// before the route decides anything.** On a miss the fallback would otherwise
/// answer for a language the decisions were not made for; the model-gated
/// `a_resolution_decided_for_another_request_is_refused_before_dispatch` pins the
/// aligner routes.
#[test]
fn a_resolution_decided_for_another_language_is_refused_on_a_miss_too() {
  for fallback in [AlignmentFallback::SkipChunk, AlignmentFallback::Error] {
    let set = AlignmentSetBuilder::new().with_fallback(fallback).build();
    let resolution = set
      .detect_oov("anything", &Lang::Zh)
      .expect("detect_oov")
      .decide(default_oov_policy);
    let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
    let abort = AtomicBool::new(false);
    let err = set
      .align_chunk(&Lang::En, &[], &[], "anything", clock, &abort, resolution)
      .expect_err("decisions made for Zh do not answer an En request");
    assert!(
      matches!(
        err,
        AlignError::DecisionLanguage(ref e)
          if *e.requested() == Lang::En && *e.found() == Lang::Zh
      ),
      "{fallback}: {err:?}"
    );
  }
}

/// A registry miss's resolution: the request's language, no detection.
fn missed(set: &AlignmentSet, language: &Lang) -> SetResolution {
  set
    .detect_oov("anything", language)
    .expect("a miss detects nothing, and fails at nothing")
    .decide(default_oov_policy)
}

#[test]
fn align_chunk_miss_skip_chunk_returns_empty_words() {
  // Empty registry, default SkipChunk policy: a miss is not an error, it drops
  // the timings and keeps going, and says so. No aligner is touched, so this is
  // hermetic.
  let set = AlignmentSetBuilder::new().build();
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let alignment = set
    .align_chunk(
      &Lang::Zh,
      &[],
      &[],
      "anything",
      clock,
      &abort,
      missed(&set, &Lang::Zh),
    )
    .expect("a SkipChunk miss is not an error");
  assert!(alignment.words().is_empty());
  assert!(matches!(alignment.cause(), Some(UnalignedCause::Skipped)));
}

#[test]
fn align_chunk_miss_error_returns_language_unsupported() {
  let set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::Error)
    .build();
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let err = set
    .align_chunk(
      &Lang::Zh,
      &[],
      &[],
      "anything",
      clock,
      &abort,
      missed(&set, &Lang::Zh),
    )
    .unwrap_err();
  assert!(matches!(
    err,
    AlignError::LanguageUnsupported(ref language) if *language == Lang::Zh
  ));
}

#[test]
fn resolve_binds_the_requested_language_and_reports_the_miss() {
  // Empty registry: `resolve` binds the REQUESTED language, and `binding()`
  // reports the miss policy as data — no aligner involved, so hermetic.
  let set = AlignmentSetBuilder::new().build();
  let handle = set.resolve(&Lang::Zh);
  assert_eq!(handle.language(), &Lang::Zh);
  assert_eq!(
    handle.binding(),
    AlignmentBinding::Miss(AlignmentFallback::SkipChunk)
  );
}

#[test]
fn handle_align_chunk_is_the_guarded_set_align_chunk() {
  // The handle's `align_chunk` is `AlignmentSet::align_chunk` bound to the
  // requested language: a SkipChunk miss returns empty words, exactly as the set
  // method does, with no aligner touched (hermetic).
  let set = AlignmentSetBuilder::new().build();
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let handle = set.resolve(&Lang::Zh);
  let resolution = handle
    .detect_oov("anything")
    .expect("detect_oov")
    .decide(default_oov_policy);
  let alignment = handle
    .align_chunk(&[], &[], "anything", clock, &abort, resolution)
    .expect("a SkipChunk miss is not an error");
  assert!(alignment.words().is_empty());
  assert!(matches!(alignment.cause(), Some(UnalignedCause::Skipped)));
}

// ---------------------------------------------------------------------
// Model-gated: populated lookup / register / detect_oov need a real
// Aligner, which loads the CoreML model (ALIGNKIT_TEST_MODELS). Same
// convention as src/encode/tests.rs (a separate `tests/` integration
// crate is unreachable from these src-level unit tests).
// ---------------------------------------------------------------------

fn models_dir() -> std::path::PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || crate::tests::models_root().join("alignkit"),
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
  // Internal lookup hits the language's own aligner (never the Any fallback).
  assert!(matches!(set.lookup(&Lang::En), AlignmentLookup::Hit(_)));
  // The public handle reports the same resolution as DATA: an exact bind.
  assert_eq!(set.resolve(&Lang::En).binding(), AlignmentBinding::Exact);
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn lookup_misses_unregistered_language_without_any() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Lang(Lang::En), en_aligner())
    .build();
  assert!(matches!(
    set.lookup(&Lang::Zh),
    AlignmentLookup::Miss(AlignmentFallback::SkipChunk)
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
  assert!(matches!(set.lookup(&Lang::En), AlignmentLookup::Hit(_)));
  // An unregistered language falls through to Any.
  assert!(matches!(
    set.lookup(&Lang::Zh),
    AlignmentLookup::AnyFallback(_)
  ));
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn any_fallback_can_match_the_requested_language() {
  // The AnyFallback binding does NOT require the fallback aligner's language to
  // DIFFER from the request — the state the variant doc once excluded. Register
  // ONLY an English aligner under AlignerKey::Any (no AlignerKey::Lang(En)); an
  // English request finds no exact Lang(En) key and falls through to Any, whose
  // own construction language IS En. What made it a fallback is the missing exact
  // key, not a language difference.
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Any, en_aligner())
    .build();
  assert!(matches!(
    set.lookup(&Lang::En),
    AlignmentLookup::AnyFallback(_)
  ));
  // The public binding carries the Any aligner's own language — here EQUAL to the
  // requested language, exactly the case the doc now cites.
  assert_eq!(
    set.resolve(&Lang::En).binding(),
    AlignmentBinding::AnyFallback(Lang::En)
  );
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
#[should_panic(expected = "cannot accept an aligner built for")]
fn register_panics_on_language_mismatch() {
  let _ = AlignmentSetBuilder::new().register(AlignerKey::Lang(Lang::Zh), en_aligner());
}

/// An `Any`-registered En aligner serving a Zh request: the detection names the
/// REQUESTED language (Zh), the key a per-language policy decides on, while its
/// events carry the language of the aligner that read the text (En) — asry
/// stamps them, and nothing re-stamps an event.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn any_fallback_detection_names_the_requested_language() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Any, en_aligner())
    .build();
  let detection = set
    .detect_oov("hello AT&T", &Lang::Zh)
    .expect("detect_oov on the Any-fallback aligner");
  assert_eq!(detection.language(), &Lang::Zh);
  let events = detection.events().expect("the Any aligner read the text");
  assert!(!events.is_empty(), "the `&` is an OOV event");
  assert!(events.iter().all(|event| event.language() == &Lang::En));
}

/// **The F2 regression, end-to-end.** An English aligner registered as the
/// multilingual [`AlignerKey::Any`] fallback, a Chinese request, real speech,
/// and a real OOV decision (the `é` of `Américans`, which the bundled table
/// cannot spell and the default policy wildcards).
///
/// Under asry 0.2 this needed the registry to re-stamp the caller's
/// requested-language decisions into the Any aligner's own language, or its
/// `prepare` refused them. asry binds decisions to the aligner that detected
/// them now, and the registry binds them to the request, so a cross-language
/// request reaches encoding and produces words with the decisions its own
/// detection made.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn any_fallback_aligns_a_cross_language_request_with_an_oov_decision() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Any, en_aligner())
    .build();
  let samples = load_jfk_wav();
  let text = JFK_TRANSCRIPT.replace("Americans", "Américans");

  let detection = set
    .detect_oov(&text, &Lang::Zh)
    .expect("detect_oov on the Any fallback");
  assert_eq!(detection.language(), &Lang::Zh);
  assert!(
    detection
      .events()
      .is_some_and(|events| events.iter().any(|event| event.char() == Some('é'))),
    "the `é` of `Américans` is an OOV event"
  );
  let resolution = detection.decide(default_oov_policy);

  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let alignment = set
    .align_chunk(
      &Lang::Zh,
      &samples,
      &whole_chunk_is_speech(&samples),
      &text,
      clock,
      &abort,
      resolution,
    )
    .expect("Any-fallback alignment must not fail on the decisions");
  assert!(
    !alignment.words().is_empty(),
    "the English Any aligner must align English speech to English words"
  );
}

/// **The F2 regression: decisions answer the request they were decided for,
/// checked before any dispatch.** asry binds a resolution to the text and the
/// aligner that detected it, so what it cannot see is the request. Three calls
/// pin the ways an unbound registry would leak:
///
/// - **one `Any` aligner, two requests**: Zh and Fr both fall through to the
///   same `Any` aligner, which detected the decisions in the same text, so asry
///   would ACCEPT the Zh policy's decisions under the Fr request and align
///   silently — the leak only the registry's binding closes;
/// - **in-window** audio under the exact En hit, with decisions the `Any`
///   aligner made for Zh: asry would refuse them as another aligner's, an
///   undifferentiated [`AlignError::Alignment`] (its `Tokenization`);
/// - **oversized** audio, same decisions:
///   [`Aligner::align_chunk`](crate::audio::align::aligner::Aligner::align_chunk)'s own length
///   check would raise [`AlignError::InputTooLong`] before `prepare` even runs,
///   so the SAME wrong input would produce a DIFFERENT error depending on the
///   audio length.
///
/// With the binding all three are the identical typed
/// [`AlignError::DecisionLanguage`], because [`AlignmentSet::align_chunk`] checks
/// the resolution's language ahead of dispatching anywhere. Deleting that check
/// turns the first call into a successful alignment and the other two into the
/// route-dependent errors above — the mutation proof.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn a_resolution_decided_for_another_request_is_refused_before_dispatch() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Lang(Lang::En), en_aligner())
    .register(AlignerKey::Any, en_aligner())
    .build();
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);

  let assert_decision_language =
    |requested: Lang, decided_for: Lang, samples: &[f32], case: &str| {
      let resolution = set
        .detect_oov("test", &decided_for)
        .expect("detect_oov")
        .decide(default_oov_policy);
      let err = set
        .align_chunk(&requested, samples, &[], "test", clock, &abort, resolution)
        .expect_err("decisions made for another request must be refused");
      assert!(
        matches!(
          err,
          AlignError::DecisionLanguage(ref e)
            if *e.requested() == requested && *e.found() == decided_for
        ),
        "{case}: must be the typed DecisionLanguage, got {err:?}"
      );
    };

  let in_window = vec![0.0f32; 16_000];
  assert_decision_language(
    Lang::Fr,
    Lang::Zh,
    &in_window,
    "one Any aligner, two requests",
  );
  assert_decision_language(Lang::En, Lang::Zh, &in_window, "in-window");
  let oversized = vec![0.0f32; crate::audio::align::encode::ENCODER_WINDOW_SAMPLES + 1];
  assert_decision_language(Lang::En, Lang::Zh, &oversized, "oversized");
}

/// **A resolution another registry decided on a miss is refused where the
/// language has an aligner.** A registry's lookup is a function of the language
/// alone, so on one registry a miss's resolution never meets an aligner; it can
/// only be another registry's. It carries no decisions the aligner detected,
/// and is refused by name — as asry refuses decisions another aligner
/// detected — rather than aligned with none.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn another_registrys_miss_resolution_is_refused_on_a_hit() {
  let empty = AlignmentSetBuilder::new().build();
  let resolution = empty
    .detect_oov("test", &Lang::En)
    .expect("a miss detects nothing")
    .decide(default_oov_policy);
  assert!(resolution.resolved().is_none());

  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Lang(Lang::En), en_aligner())
    .build();
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let err = set
    .align_chunk(
      &Lang::En,
      &[0.0f32; 16_000],
      &[],
      "test",
      clock,
      &abort,
      resolution,
    )
    .expect_err("no aligner here detected these decisions");
  assert!(
    matches!(err, AlignError::Alignment(EmissionsError::Tokenization(_))),
    "{err:?}"
  );
}

/// The known transcript for `jfk.wav`, with the commas that make the F2 test's
/// punctuation OOV real (duplicated from `tests/common`, as the other src-level
/// unit tests duplicate their fixtures — a `tests/` module is unreachable here).
const JFK_TRANSCRIPT: &str = "And so my fellow Americans ask not what your country can do for you, \
                              ask what you can do for your country.";

/// "No VAD" — one explicit span over the whole chunk in the 1/16000 analysis
/// timebase, i.e. all speech. Passing empty `sub_segments` is also "no VAD": the
/// shipping API maps empty to
/// [`SpeechSpans::all_speech`](asry::emissions::SpeechSpans::all_speech), NOT to
/// "all silence" (see [`crate::audio::align::aligner::Aligner::align_chunk`]); this helper just
/// states the whole-chunk span explicitly.
fn whole_chunk_is_speech(samples: &[f32]) -> [TimeRange; 1] {
  [TimeRange::new(0, samples.len() as i64, ANALYSIS_TIMEBASE)]
}

/// The 11 s `jfk.wav` fixture, borrowed from the whisperkit crate by relative
/// path (as the other src-level tests do) and failing LOUDLY if it ever moves.
fn load_jfk_wav() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/whisper/fixtures/audio/jfk.wav");
  let mut reader = hound::WavReader::open(&path)
    .unwrap_or_else(|e| panic!("open the jfk.wav fixture at {path:?}: {e}"));
  assert_eq!(reader.spec().sample_rate, 16_000, "fixture must be 16 kHz");
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}
