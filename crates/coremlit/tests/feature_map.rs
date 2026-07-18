//! Golden feature-map test — pins the mono-crate flat feature contract.
//!
//! The restructure collapsed five crates into feature-gated modules and
//! renamed each per-crate feature to a flat one (`FEATURE_MAP.md`). This test
//! parses the crate `Cargo.toml` and fails if the declared feature set, the
//! composite compositions, or the doc drift from that pinned rename table — so
//! a renamed, dropped, or re-composed feature cannot land silently. Hermetic:
//! it reads the manifest text, needs no models and no feature to be enabled.

use std::{collections::BTreeSet, path::Path};

fn manifest() -> String {
  std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
    .expect("read crate Cargo.toml")
}

/// The top-level keys of the `[features]` table (feature *names*). Robust to
/// multi-line array values: a feature key sits at column 0, while an array's
/// continuation lines are indented.
fn feature_keys(manifest: &str) -> BTreeSet<String> {
  let mut keys = BTreeSet::new();
  let mut in_features = false;
  for line in manifest.lines() {
    if line.starts_with('[') {
      in_features = line.trim() == "[features]";
      continue;
    }
    if !in_features || line.starts_with(char::is_whitespace) {
      continue;
    }
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with('#') {
      continue;
    }
    if let Some((key, _)) = line.split_once('=') {
      let key = key.trim();
      if !key.is_empty() && !key.contains(char::is_whitespace) {
        keys.insert(key.to_string());
      }
    }
  }
  keys
}

#[test]
fn feature_set_matches_the_rename_table() {
  let keys = feature_keys(&manifest());
  let expected: BTreeSet<String> = [
    "default",
    "serde",
    "tracing",
    "whisper",
    "nl-recognizer",
    "align",
    "align-oracle",
    "speaker",
    "speaker-oracle",
    "vad",
    "vad-bundled",
  ]
  .iter()
  .map(|s| (*s).to_string())
  .collect();
  assert_eq!(
    keys, expected,
    "Cargo.toml [features] drifted from the pinned flat feature set (FEATURE_MAP.md rename table)"
  );
}

#[test]
fn old_per_crate_feature_names_are_gone() {
  let keys = feature_keys(&manifest());
  for old in ["dia", "dia-oracle", "parity-oracle", "vadkit", "bundled"] {
    assert!(
      !keys.contains(old),
      "old per-crate feature `{old}` is still declared — the rename table maps it away"
    );
  }
}

#[test]
fn composite_features_compose_as_pinned() {
  let m = manifest();
  for (decl, needle) in [
    ("nl-recognizer = [", "\"whisper\""),
    ("align-oracle = [", "\"align\""),
    ("align-oracle = [", "\"asry/alignment\""),
    ("speaker-oracle = [", "\"speaker\""),
    ("speaker-oracle = [", "\"dia/ort\""),
    ("vad-bundled = [", "\"vad\""),
    ("vad-bundled = [", "\"silero/bundled\""),
  ] {
    let line = m
      .lines()
      .find(|l| l.trim_start().starts_with(decl))
      .unwrap_or_else(|| panic!("no `{decl}` feature line in Cargo.toml"));
    assert!(
      line.contains(needle),
      "`{decl}…` must contain {needle} (FEATURE_MAP.md composition)"
    );
  }
}

#[test]
fn feature_map_doc_lists_every_feature() {
  let doc = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("FEATURE_MAP.md"))
    .expect("read FEATURE_MAP.md");
  for feat in feature_keys(&manifest()) {
    if feat == "default" {
      continue;
    }
    assert!(
      doc.contains(&format!("`{feat}`")),
      "FEATURE_MAP.md must document the `{feat}` feature (doc/manifest drift)"
    );
  }
}
