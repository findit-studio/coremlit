//! **F1 regression** — the registry's PUBLIC cross-language surface is the
//! request-bound [`AlignmentHandle`], never a raw `&Aligner`.
//!
//! Round 2 gave the registry `AlignmentSet::align_chunk` to cross an
//! `AlignerKey::Any` fallback's decisions safely, but `lookup()` still handed
//! back the raw `AnyFallback` aligner: extracting it and calling `detect_oov`
//! through it stamped events with the aligner's OWN language (English), and
//! `align_chunk` through it reproduced the generic decision-language error the
//! typed `AlignError::DecisionLanguage` replaced — the guard bypass round 2
//! closed, reopened. `lookup` / `AlignmentLookup` are now private, and
//! `AlignmentSet::resolve` returns a handle whose `detect_oov` / `align_chunk`
//! delegate through the guarded paths, keyed on the REQUESTED language.
//!
//! The compile-fail proof that no public path yields a raw `&Aligner` from an
//! `Any` match is the `compile_fail` doctest on `AlignmentSet::resolve` (run by
//! `cargo test --doc`): re-exposing `lookup` makes it compile, failing that
//! doctest. This file is the behavioural half — the language-dependent policy an
//! external caller observes THROUGH the handle keys on the request, not on the
//! fallback aligner's own language.

mod common;

use core::sync::atomic::AtomicBool;

use coremlit::audio::align::{
  ANALYSIS_TIMEBASE, Aligner, AlignerKey, AlignmentBinding, AlignmentFallback, AlignmentSetBuilder,
  EnglishNormalizer, Lang, OutputClock, TimeRange, default_oov_decisions,
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
    AlignmentBinding::Miss {
      fallback: AlignmentFallback::Error,
    }
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
/// request, real speech, and a real punctuation OOV (the jfk transcript's commas)
/// — driven entirely through `AlignmentSet::resolve(...)` → [`AlignmentHandle`],
/// the guarded surface an external caller actually has now that the raw
/// `&Aligner` is unreachable.
///
/// The language-dependent policy MUST observe the REQUESTED language (Zh): the
/// binding reports the bound aligner's own language as DATA (En), while
/// `detect_oov` stamps every event Zh and `align_chunk` crosses the Zh-resolved
/// decisions into the En aligner and produces words. Were the handle to delegate
/// to the raw aligner instead of the guarded set methods, `detect_oov` would
/// stamp En and the all-Zh assertion below would fail — the mutation proof for
/// the bound-API half of F1.
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
  assert_eq!(
    handle.binding(),
    AlignmentBinding::AnyFallback {
      aligner_language: Lang::En,
    }
  );

  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  let text = common::JFK_TRANSCRIPT;

  // detect_oov THROUGH the handle stamps the REQUESTED language (Zh), never the
  // fallback aligner's En — the language-dependent policy keys on the request.
  let events = handle.detect_oov(text).expect("handle detect_oov");
  assert!(
    !events.is_empty(),
    "the jfk transcript's commas must yield OOV events"
  );
  assert!(
    events.iter().all(|e| e.language() == &Lang::Zh),
    "policy selection must observe the requested language, not the Any aligner's En"
  );
  let decisions = default_oov_decisions(&events);

  // align_chunk THROUGH the handle crosses the Zh-resolved decisions into the En
  // aligner and reaches encoding WITHOUT a decision-language error, producing
  // words — the guarded path, not the reopened raw one.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock");
  let abort = AtomicBool::new(false);
  let result = handle
    .align_chunk(
      &samples,
      &whole_chunk_is_speech(&samples),
      text,
      clock,
      &abort,
      &decisions,
    )
    .expect("Any-fallback alignment through the handle must not fail on the decision language");
  assert!(
    !result.words().is_empty(),
    "the English Any aligner must align English speech to English words"
  );
}
