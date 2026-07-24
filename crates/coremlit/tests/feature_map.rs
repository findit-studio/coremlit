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
//!   3. `.github/workflows/ci.yml` — the curated `--features` combo matrix,
//!      parsed structurally and compared as an exact set, so dropping OR
//!      commenting out any curated combo (including the bare-core `""`) reds.
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
    ("speaker", vec!["dep:diaric"]),
    (
      "speaker-oracle",
      vec!["speaker", "dep:dia", "dia/ort", "dia/bundled-segmentation"],
    ),
    ("vad", vec!["dep:silero"]),
    ("vad-bundled", vec!["vad", "silero/bundled"]),
    ("clap", vec!["dep:rustfft", "dep:tokenizers", "dep:windit"]),
    ("clap-oracle", vec!["clap", "dep:textclap"]),
    (
      "granite",
      vec!["dep:tokenizers", "dep:windit", "windit/text"],
    ),
    ("siglip", vec!["dep:tokenizers", "dep:colconv"]),
  ]
}

/// The former per-crate kits and the flat module-feature each bare crate maps
/// to. Drives the rename-table row check, so a REMOVED bare-crate row reds.
const BARE_CRATE_MAP: &[(&str, &str)] = &[
  ("whisperkit", "whisper"),
  ("alignkit", "align"),
  ("speakerkit", "speaker"),
  ("vadkit", "vad"),
  ("clapkit", "clap"),
];

/// The curated CI feature combos the mono-crate restructure committed to — the
/// EXACT intended set of `jobs.features.strategy.matrix.features` entries in
/// `.github/workflows/ci.yml`, as raw (unquoted) combo strings. The empty
/// string is a real member: the bare-core `default = []` run (ci.yml `- ""`).
/// `ci_feature_combos` parses the ACTIVE matrix and the test asserts exact set
/// equality against this, so removing OR commenting out any entry (the bare-core
/// `""` included) drops it from the parsed set and reds.
const INTENDED_CI_COMBOS: &[&str] = &[
  "", // bare core / none (`default = []`)
  "whisper",
  "align",
  "speaker",
  "vad",
  "whisper,vad",
  "align-oracle",
  "speaker-oracle",
  "vad-bundled",
  "clap",
  "clap-oracle",
  "granite",
  "siglip",
  "whisper,align,speaker,vad,clap,granite,siglip,serde,tracing,nl-recognizer",
  "whisper,align-oracle,speaker-oracle,vad-bundled,clap-oracle,granite,siglip,serde,tracing,nl-recognizer",
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

/// Parse the ACTIVE `jobs.features.strategy.matrix.features` list from a ci.yml
/// text into the set of `--features` combo strings it runs.
///
/// Structural, not substring: it enters the list only at the `features:` key
/// that FOLLOWS `matrix:` (so the `features` JOB name and the `cargo build
/// --features` step cannot be mistaken for it), collects each `- "..."` item's
/// inner value (the empty `- ""` is the empty-string member), SKIPS any line
/// whose first non-space char is `#` (a commented-out `# - "..."` entry does NOT
/// count as present, and does not end the list), and stops at the first dedent
/// to the key's column or left of it.
fn ci_feature_combos(yaml: &str) -> BTreeSet<String> {
  let mut combos = BTreeSet::new();
  let mut seen_matrix = false;
  let mut key_indent: Option<usize> = None;
  for line in yaml.lines() {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    let Some(ki) = key_indent else {
      if trimmed == "matrix:" {
        seen_matrix = true;
      } else if seen_matrix && trimmed == "features:" {
        key_indent = Some(indent);
      }
      continue;
    };
    // Comments and blanks are transparent: a commented-out entry is skipped
    // (not counted) WITHOUT terminating the list.
    if trimmed.is_empty() || trimmed.starts_with('#') {
      continue;
    }
    // A dedent to the key's column (or left of it) ends the list.
    if indent <= ki {
      break;
    }
    if let Some(inner) = trimmed.strip_prefix("- ").and_then(quoted_inner) {
      combos.insert(inner);
    }
  }
  combos
}

/// The text between the first pair of `"` in `s` (`""` → the empty string).
fn quoted_inner(s: &str) -> Option<String> {
  let start = s.find('"')?;
  let rest = &s[start + 1..];
  let end = rest.find('"')?;
  Some(rest[..end].to_string())
}

/// The intended CI combos as an owned set.
fn intended_ci_combos() -> BTreeSet<String> {
  INTENDED_CI_COMBOS
    .iter()
    .map(|s| (*s).to_string())
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

/// Assert two combo sets are equal, naming the symmetric difference so a drift
/// reports exactly what moved (a `""` member prints as an empty string).
fn assert_combo_sets_eq(actual: &BTreeSet<String>, expected: &BTreeSet<String>, what: &str) {
  let missing: Vec<&String> = expected.difference(actual).collect();
  let unexpected: Vec<&String> = actual.difference(expected).collect();
  assert!(
    missing.is_empty() && unexpected.is_empty(),
    "{what} drifted from the pinned curated set — missing (pinned but not in the \
     active matrix): {missing:?}; unexpected (in the active matrix but not pinned): {unexpected:?}"
  );
}

/// `ci.yml`'s ACTIVE feature matrix is EXACTLY the curated combo set. The parsed
/// `matrix.features` set is compared for exact equality with `INTENDED_CI_COMBOS`
/// (not substring containment), so removing a combo, commenting one out, or
/// adding an unexpected one all red — the bare-core `""` included.
#[test]
fn ci_pins_the_curated_feature_combos() {
  assert_combo_sets_eq(
    &ci_feature_combos(&ci_yml()),
    &intended_ci_combos(),
    "ci.yml feature matrix",
  );
}

/// A well-formed matrix snippet whose active `features:` list is exactly the
/// intended set — the fixture the mutation cases below perturb. Its surrounding
/// keys (the `features` JOB name above `matrix:`, the `steps:` dedent below)
/// prove the parser enters at the right `features:` and stops at the dedent,
/// rather than latching onto the job name.
const DOCTORED_MATRIX: &str = r#"
  features:
    runs-on: macos-15
    strategy:
      fail-fast: false
      matrix:
        features:
          - ""
          - "whisper"
          - "align"
          - "speaker"
          - "vad"
          - "whisper,vad"
          - "align-oracle"
          - "speaker-oracle"
          - "vad-bundled"
          - "clap"
          - "clap-oracle"
          - "granite"
          - "siglip"
          - "whisper,align,speaker,vad,clap,granite,siglip,serde,tracing,nl-recognizer"
          - "whisper,align-oracle,speaker-oracle,vad-bundled,clap-oracle,granite,siglip,serde,tracing,nl-recognizer"
    steps:
      - uses: actions/checkout@v7
"#;

/// The parser reads the intended set from the well-formed fixture — a guard on
/// the mutation cases below (each perturbs this same fixture).
#[test]
fn ci_combo_parser_reads_the_wellformed_matrix() {
  assert_combo_sets_eq(
    &ci_feature_combos(DOCTORED_MATRIX),
    &intended_ci_combos(),
    "well-formed doctored matrix",
  );
}

/// Deleting the bare-core `- ""` entry drops it from the parsed set — the
/// set-equality check must red (the R2 gap: the bare-core run was unpinned).
#[test]
fn ci_combo_check_reds_when_bare_core_is_deleted() {
  let doctored = DOCTORED_MATRIX.replace("          - \"\"\n", "");
  assert_ne!(
    ci_feature_combos(&doctored),
    intended_ci_combos(),
    "deleting the bare-core `- \"\"` entry must make the parsed set differ from the pinned set"
  );
}

/// Commenting out the bare-core `- ""` entry (`# - ""`) must red: comment lines
/// are skipped (not counted), so the parsed set loses `""` — this is what the
/// old substring `.contains()` check silently ACCEPTED.
#[test]
fn ci_combo_check_reds_when_bare_core_is_commented_out() {
  let doctored = DOCTORED_MATRIX.replace("          - \"\"\n", "          # - \"\"\n");
  assert_ne!(
    ci_feature_combos(&doctored),
    intended_ci_combos(),
    "commenting out the bare-core `- \"\"` entry must make the parsed set differ from the pinned set"
  );
}

/// Dropping any other curated combo (`whisper,vad`) must red as well.
#[test]
fn ci_combo_check_reds_when_a_combo_is_dropped() {
  let doctored = DOCTORED_MATRIX.replace("          - \"whisper,vad\"\n", "");
  assert_ne!(
    ci_feature_combos(&doctored),
    intended_ci_combos(),
    "dropping the `whisper,vad` combo must make the parsed set differ from the pinned set"
  );
}
