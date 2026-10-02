//! **F1 regression** — the registry's PUBLIC cross-language surface is the
//! request-bound [`AlignmentHandle`], never a raw `&Aligner`.
//!
//! Round 2 gave the registry `AlignmentSet::align_chunk` to bind an
//! `AlignerKey::Any` fallback's decisions to the request, but `lookup()` still
//! handed back the raw `AnyFallback` aligner: extracting it let a caller detect
//! and align through it past that binding — the guard bypass round 2 closed,
//! reopened. `lookup` / `AlignmentLookup` are now private, and
//! `AlignmentSet::resolve` returns a handle whose `detect_oov` / `align_chunk`
//! delegate through the guarded paths, keyed on the REQUESTED language.
//!
//! The compile-fail proof that no public path yields a raw `&Aligner` from an
//! `Any` match is the `compile_fail` doctest on `AlignmentSet::resolve` (run by
//! `cargo test --doc`): re-exposing `lookup` makes it compile, failing that
//! doctest. This file is the behavioural half — the language-dependent policy an
//! external caller decides THROUGH the handle keys on the request, not on the
//! fallback aligner's own language.

mod common;

use core::sync::atomic::AtomicBool;

use coremlit::audio::align::{
  ANALYSIS_TIMEBASE, AlignError, Aligner, AlignerKey, AlignmentBinding, AlignmentFallback,
  AlignmentSetBuilder, EnglishNormalizer, Lang, OovDecision, OutputClock, TimeRange,
};

// ---------------------------------------------------------------------
// Hermetic: the handle's public shape needs no model.
// ---------------------------------------------------------------------

#[test]
fn resolve_exposes_binding_as_data_never_an_aligner() {
  // An external caller resolving Zh against an empty registry: the ONLY things
  // the public handle exposes are the requested language and the binding DATA.
  // There is no accessor that returns the bound `&Aligner` — the whole point of
  // F1. (That the raw `lookup`/`AlignmentLookup` path is gone is proved at
  // COMPILE time by the `compile_fail` doctest on `AlignmentSet::resolve`.)
  let set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::Error)
    .build();
  let handle = set.resolve(&Lang::Zh);
  assert_eq!(handle.language(), &Lang::Zh);
  assert_eq!(
    handle.binding(),
    AlignmentBinding::Miss(AlignmentFallback::Error)
  );
}

// ---------------------------------------------------------------------
// Model-gated: the cross-language policy end-to-end through the handle.
// ---------------------------------------------------------------------

/// Loads the real CoreML model as an En aligner on the SHIPPING compute default
/// (`Aligner::from_paths` → `AlignerOptions::new()`), not a hardcoded placement.
fn en_aligner() -> Aligner {
  Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("load base960h_aligner.mlmodelc as an En aligner (set ALIGNKIT_TEST_MODELS)")
}

/// "No VAD" — one explicit span over the whole chunk in the 1/16000 analysis
/// timebase, i.e. all speech. Passing empty `sub_segments` is also "no VAD": the
/// shipping API maps empty to `SpeechSpans::all_speech`, NOT to "all silence".
fn whole_chunk_is_speech(samples: &[f32]) -> [TimeRange; 1] {
  [TimeRange::new(0, samples.len() as i64, ANALYSIS_TIMEBASE)]
}

/// **The F1 regression, end-to-end through the public bound API.** An English
/// aligner registered as the multilingual [`AlignerKey::Any`] fallback, a Chinese
/// request, real speech, and a real OOV event (the `é` of `Américans`, which the
/// bundled table cannot spell) — driven entirely through
/// `AlignmentSet::resolve(...)` → [`AlignmentHandle`], the guarded surface an
/// external caller actually has now that the raw `&Aligner` is unreachable.
///
/// The language-dependent policy is judged under the REQUESTED language (Zh):
/// the binding reports the bound aligner's own language as DATA (En), and the
/// detection names the request and shows every event under it, though asry
/// stamped them with the aligner's En. The policy below keys on the language
/// each event is shown under — wildcard under Zh, refuse under any other — so
/// it reaches encoding and produces words; shown the stamped En, it would refuse
/// the `é` and the alignment would fail.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn any_fallback_handle_keys_policy_on_the_requested_language() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Any, en_aligner())
    .build();
  let handle = set.resolve(&Lang::Zh);

  // The request is bound to Zh; the binding reports it is served by the Any
  // fallback whose OWN construction language is En — metadata as DATA, not the
  // aligner itself.
  assert_eq!(handle.language(), &Lang::Zh);
  assert_eq!(handle.binding(), AlignmentBinding::AnyFallback(Lang::En));

  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  let text = common::JFK_TRANSCRIPT.replacen("Americans", "Américans", 1);

  // detect_oov THROUGH the handle names the REQUESTED language (Zh) and shows
  // every event under it.
  let detection = handle.detect_oov(&text).expect("handle detect_oov");
  assert_eq!(detection.language(), &Lang::Zh);
  let events = detection.events().expect("the Any aligner read the text");
  assert!(
    events.iter().any(|event| event.char() == Some('é')),
    "the `é` of `Américans` is an OOV event: {events:?}"
  );
  assert!(events.iter().all(|event| event.language() == &Lang::Zh));

  // A per-language policy, keyed on the language each event is shown under.
  let resolution = detection.decide(|event| {
    if event.language() == &Lang::Zh {
      OovDecision::Wildcard
    } else {
      OovDecision::FailClosed
    }
  });

  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let alignment = handle
    .align_chunk(
      &samples,
      &whole_chunk_is_speech(&samples),
      &text,
      clock,
      &abort,
      resolution,
    )
    .expect("Any-fallback alignment through the handle must not fail on the decisions");
  assert!(
    !alignment.words().is_empty(),
    "the English Any aligner must align English speech to English words"
  );
}

/// **A refusal through the `Any` fallback names exactly the positions the
/// caller decided**, where the bound aligner's detection found them and under
/// the requested language: the refusal is the caller's own decisions, never a
/// re-detected copy of them.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn any_fallback_refusal_names_the_events_the_caller_decided() {
  let set = AlignmentSetBuilder::new()
    .register(AlignerKey::Any, en_aligner())
    .build();
  let handle = set.resolve(&Lang::Zh);
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  let text = "ask not what your country can do for you, AT&T b4d";

  let detection = handle.detect_oov(text).expect("handle detect_oov");
  let decided: Vec<_> = detection
    .events()
    .expect("the Any aligner read the text")
    .iter()
    .map(|event| (event.kind().clone(), event.char_index(), event.word_index()))
    .collect();
  assert!(
    decided.len() >= 2,
    "the `&` and the `4` are both events: {decided:?}"
  );
  let resolution = detection.decide(|_| OovDecision::FailClosed);

  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let err = handle
    .align_chunk(&samples, &[], text, clock, &abort, resolution)
    .expect_err("a policy that fails closed on every event refuses the chunk");
  let AlignError::Refused(refusal) = err else {
    panic!("the refusal must be named, got {err:?}");
  };
  let refused: Vec<_> = refusal
    .events()
    .iter()
    .map(|event| (event.kind().clone(), event.char_index(), event.word_index()))
    .collect();
  assert_eq!(
    refused, decided,
    "the refusal names exactly the positions the caller decided"
  );
  assert!(
    refusal
      .events()
      .iter()
      .all(|event| event.language() == &Lang::Zh),
    "the refusal names every position under the requested language"
  );
}
