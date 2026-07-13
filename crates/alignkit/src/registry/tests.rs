use super::*;

use asry::emissions::EnglishNormalizer;

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
