//! Golden feature-map test — pins the mono-crate flat feature contract.
//!
//! The restructure collapsed five crates into feature-gated modules and renamed
//! each per-crate feature to a flat one (`FEATURE_MAP.md`). This test PINS that
//! contract against its three sources of truth, so a renamed, dropped,
//! re-composed, or cross-kit-leaking feature — or a silently dropped CI combo —
//! cannot land:
//!
//!   1. `Cargo.toml` `[features]` — the exact feature-name set AND the exact
//!      dependency set of every feature (a leak like `whisper` pulling `vad`
//!      changes a set and reds).
//!   2. `FEATURE_MAP.md`'s rename table — parsed (not substring-scanned) so a
//!      removed/altered bare-crate row reds even if the token survives elsewhere
//!      in the doc.
//!   3. `.github/workflows/ci.yml` — the curated `--features` combo matrix, so
//!      dropping `whisper,vad` or an all-on combo reds.
//!
//! Hermetic: pure file reads (via `CARGO_MANIFEST_DIR`), no models, no cargo
//! invocation, no feature needs enabling.

use std::{collections::BTreeSet, path::Path};

/// Read a file addressed relative to the crate manifest directory.
fn read_rel(rel: &str) -> String {
  std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
    .unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

fn manifest() -> String {
  read_rel("Cargo.toml")
}

/// `.github/workflows/ci.yml` lives two levels above the crate dir
/// (`crates/coremlit` → repo root).
fn ci_yml() -> String {
  let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml");
  std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The intended flat feature graph — the single in-test source that both the
/// name-set and the per-feature dependency-set assertions are driven from. Any
/// drift in `Cargo.toml` (a renamed feature, a dropped dep, or a CROSS-KIT LEAK
/// such as adding `"vad"` to `whisper`) changes a set here and reds.
fn expected_features() -> Vec<(&'static str, Vec<&'static str>)> {
  vec![
    ("default", vec![]),
    ("serde", vec!["dep:serde"]),
    ("tracing", vec!["dep:tracing"]),
    (
      "whisper",
      vec![
        "dep:libc",
        "dep:mach2",
        "dep:rand",
        "dep:serde_json",
        "dep:tokenizers",
        "dep:unicode_categories",
      ],
    ),
    (
      "nl-recognizer",
      vec!["whisper", "dep:objc2-natural-language"],
    ),
    ("align", vec!["dep:asry"]),
    ("align-oracle", vec!["align", "asry/alignment"]),
    ("speaker", vec!["dep:dia"]),
    (
      "speaker-oracle",
      vec!["speaker", "dia/ort", "dia/bundled-segmentation"],
    ),
    ("vad", vec!["dep:silero"]),
    ("vad-bundled", vec!["vad", "silero/bundled"]),
  ]
}

/// The former per-crate kits and the flat module-feature each bare crate maps
/// to. Drives the rename-table row check, so a REMOVED bare-crate row reds.
const BARE_CRATE_MAP: &[(&str, &str)] = &[
  ("whisperkit", "whisper"),
  ("alignkit", "align"),
  ("speakerkit", "speaker"),
  ("vadkit", "vad"),
];

/// The curated CI feature combos the restructure committed to. Substring-pinned
/// against `ci.yml` (each token is quoted, so a single feature never matches
/// inside a longer combo), so REMOVING any combo reds. The empty `""` (none)
/// combo is not listed — it is not substring-checkable.
const REQUIRED_CI_COMBOS: &[&str] = &[
  "\"whisper\"",
  "\"align\"",
  "\"speaker\"",
  "\"vad\"",
  "\"whisper,vad\"",
  "\"align-oracle\"",
  "\"speaker-oracle\"",
  "\"vad-bundled\"",
  "\"whisper,align,speaker,vad,serde,tracing,nl-recognizer\"",
  "\"whisper,align-oracle,speaker-oracle,vad-bundled,serde,tracing,nl-recognizer\"",
];

/// The text of the `[features]` table (its lines, blank/comment lines included).
fn features_block(manifest: &str) -> String {
  let mut out = String::new();
  let mut in_features = false;
  for line in manifest.lines() {
    if line.starts_with('[') {
      in_features = line.trim() == "[features]";
      continue;
    }
    if in_features {
      out.push_str(line);
      out.push('\n');
    }
  }
  out
}

/// The feature *names* declared in the `[features]` block. A feature key sits at
/// column 0; an array's continuation lines are indented (and so skipped).
fn feature_names(block: &str) -> BTreeSet<String> {
  let mut names = BTreeSet::new();
  for line in block.lines() {
    if line.starts_with(char::is_whitespace) {
      continue;
    }
    let trimmed = line.trim_start();
    if trimmed.is_empty() || trimmed.starts_with('#') {
      continue;
    }
    if let Some((key, _)) = line.split_once('=') {
      let key = key.trim();
      if !key.is_empty() && !key.contains(char::is_whitespace) {
        names.insert(key.to_string());
      }
    }
  }
  names
}

/// The dependency set of one feature — the quoted entries of its `[..]` value,
/// robust to a value spread over multiple (indented) lines.
fn feature_deps(block: &str, feature: &str) -> BTreeSet<String> {
  let mut collecting = false;
  let mut buf = String::new();
  for line in block.lines() {
    if collecting {
      buf.push('\n');
      buf.push_str(line);
      if line.contains(']') {
        break;
      }
      continue;
    }
    if line.starts_with(char::is_whitespace) {
      continue;
    }
    let Some((key, rest)) = line.split_once('=') else {
      continue;
    };
    if key.trim() != feature {
      continue;
    }
    collecting = true;
    buf.push_str(rest);
    if rest.contains(']') {
      break;
    }
  }
  // Quoted contents are the odd-indexed pieces of a split on '"'.
  buf
    .split('"')
    .skip(1)
    .step_by(2)
    .map(str::to_string)
    .collect()
}

/// Parse ONLY the "## Rename table" section of `FEATURE_MAP.md` into rows of
/// trimmed cells. Scoped to that section, so the separate curated-CI-combo table
/// lower in the doc cannot satisfy a rename-row assertion, and a bare token
/// elsewhere in the prose cannot stand in for a removed row.
fn rename_table_rows(doc: &str) -> Vec<Vec<String>> {
  let mut rows = Vec::new();
  let mut in_table = false;
  for line in doc.lines() {
    if let Some(heading) = line.strip_prefix("## ") {
      in_table = heading.contains("Rename table");
      continue;
    }
    if !in_table {
      continue;
    }
    let line = line.trim();
    if !line.starts_with('|') {
      continue;
    }
    let cells: Vec<String> = line
      .trim_matches('|')
      .split('|')
      .map(|c| c.trim().to_string())
      .collect();
    // Skip the header row and the `|---|---|` separator row.
    if cells.iter().any(|c| c == "Old crate") {
      continue;
    }
    if cells
      .iter()
      .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-'))
    {
      continue;
    }
    rows.push(cells);
  }
  rows
}

fn unbacktick(cell: &str) -> &str {
  cell.trim_matches('`')
}

/// `Cargo.toml` `[features]` names match the pinned flat set exactly — no
/// renamed, added, or dropped feature.
#[test]
fn feature_names_match_the_pinned_set() {
  let actual = feature_names(&features_block(&manifest()));
  let expected: BTreeSet<String> = expected_features()
    .iter()
    .map(|(name, _)| (*name).to_string())
    .collect();
  assert_eq!(
    actual, expected,
    "Cargo.toml [features] names drifted from the pinned flat feature set (FEATURE_MAP.md)"
  );
}

/// Every former per-crate feature name is gone (renamed away by the table).
#[test]
fn old_per_crate_feature_names_are_gone() {
  let names = feature_names(&features_block(&manifest()));
  for old in ["dia", "dia-oracle", "parity-oracle", "vadkit", "bundled"] {
    assert!(
      !names.contains(old),
      "old per-crate feature `{old}` is still declared — the rename table maps it away"
    );
  }
}

/// Each feature's dependency set matches its pinned set exactly. Exact-set
/// equality catches cross-kit LEAKAGE (e.g. adding `"vad"` to `whisper` adds an
/// entry the pinned `whisper` set does not have) as well as a dropped
/// composition edge (e.g. `nl-recognizer` losing `whisper`).
#[test]
fn feature_deps_are_pinned_with_no_cross_kit_leakage() {
  let block = features_block(&manifest());
  for (name, deps) in expected_features() {
    let actual = feature_deps(&block, name);
    let expected: BTreeSet<String> = deps.iter().map(|d| (*d).to_string()).collect();
    assert_eq!(
      actual, expected,
      "feature `{name}` dependency set drifted (cross-kit leakage or a dropped/added dep)"
    );
  }
}

/// The `FEATURE_MAP.md` rename table maps every former kit's BARE crate to its
/// module feature. Parsed structurally (crate cell + `(crate)` cell + feature
/// cell), so removing or altering a bare-crate row reds even when the feature
/// token still appears elsewhere in the doc (the flat-feature list, a `dia`-style
/// feature row, or the prose).
#[test]
fn rename_table_pins_every_bare_crate_row() {
  let rows = rename_table_rows(&read_rel("FEATURE_MAP.md"));
  assert!(
    rows.len() >= BARE_CRATE_MAP.len(),
    "rename-table parse found only {} row(s) — the parser or the table shape broke",
    rows.len()
  );
  for (kit, feature) in BARE_CRATE_MAP {
    let found = rows
      .iter()
      .any(|r| r.len() >= 3 && r[0] == *kit && r[1] == "(crate)" && unbacktick(&r[2]) == *feature);
    assert!(
      found,
      "FEATURE_MAP.md rename table must map bare crate `{kit}` | (crate) | `{feature}` \
       (a removed or altered bare-crate row)"
    );
  }
}

/// `ci.yml` still runs every curated feature combo. Each required `--features`
/// value is substring-pinned, so removing `whisper,vad`, an oracle combo, or an
/// all-on combo reds.
#[test]
fn ci_pins_the_curated_feature_combos() {
  let ci = ci_yml();
  for combo in REQUIRED_CI_COMBOS {
    assert!(
      ci.contains(combo),
      "ci.yml feature matrix must include the {combo} combo (a removed CI combination)"
    );
  }
}
