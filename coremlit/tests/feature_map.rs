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
//!      changes a set and reds). BOTH manifests: this crate's and the sibling
//!      `coremlit-parity`'s, which owns the three third-party parity oracles.
//!   2. `FEATURE_MAP.md`'s rename table — parsed (not substring-scanned) so a
//!      removed/altered bare-crate row reds even if the token survives elsewhere
//!      in the doc.
//!   3. `.github/workflows/ci.yml` — the curated `--features` combo matrices,
//!      parsed structurally PER JOB and compared as exact sets, so dropping OR
//!      commenting out any curated combo (including the bare-core `""`) reds.
//!      Two jobs carry one: `features` (this crate) and `parity` (the oracles).
//!
//! The oracle features (`speaker-oracle`, `clap-oracle`, `vad-bundled`) are NOT
//! this crate's any more — `dia` and `textclap` are unpublished git sources that
//! `cargo publish` rejects, so they and their nine parity binaries moved to the
//! never-published `coremlit-parity` package. `align-oracle` did NOT move: it
//! only turns on a feature of `asry`, a dependency this crate keeps either way.
//!
//! Hermetic: pure file reads (via `CARGO_MANIFEST_DIR`), no models, no cargo
//! invocation, no feature needs enabling.

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops — see its module doc.
#[path = "support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use std::{collections::BTreeSet, path::Path};

/// Read a file addressed relative to the crate manifest directory.
fn read_rel(rel: &str) -> String {
  std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
    .unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

fn manifest() -> String {
  read_rel("Cargo.toml")
}

/// The sibling `coremlit-parity` manifest — same workspace, one directory over.
fn parity_manifest() -> String {
  read_rel("../coremlit-parity/Cargo.toml")
}

/// `.github/workflows/ci.yml`, at the workspace root — found, not counted.
fn ci_yml() -> String {
  let path = workspace_root::workspace_root().join(".github/workflows/ci.yml");
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
    ("vad", vec!["dep:zuoer"]),
    ("clap", vec!["dep:rustfft", "dep:tokenizers", "dep:windit"]),
    (
      "granite",
      vec!["dep:tokenizers", "dep:windit", "windit/text", "dep:sha2"],
    ),
    (
      "ced",
      vec!["dep:rustfft", "dep:soundevents-dataset", "dep:windit"],
    ),
    ("lid", vec!["dep:rustfft", "dep:windit"]),
    ("siglip", vec!["dep:tokenizers", "dep:pixon", "dep:sha2"]),
  ]
}

/// The intended feature graph of the sibling `coremlit-parity` package — the
/// oracle half of the contract, pinned with the same exact-set discipline. Each
/// oracle rides its OWN feature (so one row's `ort` build is not forced on the
/// others) and turns on exactly the `coremlit` module feature it measures; a
/// leak like `clap-oracle` pulling `coremlit/speaker` changes a set and reds.
fn expected_parity_features() -> Vec<(&'static str, Vec<&'static str>)> {
  vec![
    ("default", vec![]),
    (
      "speaker-oracle",
      vec![
        "coremlit/speaker",
        "dep:dia",
        "dep:diaric",
        "dia/ort",
        "dia/bundled-segmentation",
      ],
    ),
    ("clap-oracle", vec!["coremlit/clap", "dep:textclap"]),
    (
      "vad-bundled",
      vec!["coremlit/vad", "dep:silero", "silero/bundled"],
    ),
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
  "clap",
  "granite",
  "siglip",
  "ced",
  "lid",
  "whisper,align,speaker,vad,clap,granite,siglip,ced,lid,serde,tracing,nl-recognizer",
  "whisper,align-oracle,speaker,vad,clap,granite,siglip,ced,lid,serde,tracing,nl-recognizer",
];

/// The curated combos of ci.yml's `parity` job — one row per third-party oracle
/// plus an all-on row, run against `coremlit-parity`. Same exact-set discipline
/// as [`INTENDED_CI_COMBOS`]: deleting or commenting out a row reds, so an
/// oracle cannot quietly stop being built once it no longer rides this crate's
/// own feature matrix.
const INTENDED_PARITY_CI_COMBOS: &[&str] = &[
  "speaker-oracle",
  "clap-oracle",
  "vad-bundled",
  "speaker-oracle,clap-oracle,vad-bundled",
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

/// The column `jobs.<name>:` keys sit at in `.github/workflows/ci.yml`.
const JOB_INDENT: usize = 2;

/// Parse the ACTIVE `jobs.<job>.strategy.matrix.features` list from a ci.yml
/// text into the set of `--features` combo strings that job runs.
///
/// Structural, not substring, and SCOPED TO ONE JOB: it enters at the
/// `  <job>:` key, leaves at the next key in that column (so a later job's
/// matrix cannot be folded into this one's set — the split into `features` +
/// `parity` made that a live confusion), and within the job takes the
/// `features:` key that FOLLOWS `matrix:` (so the `features` JOB name and the
/// `cargo build --features` step cannot be mistaken for it). It collects each
/// `- "..."` item's inner value (the empty `- ""` is the empty-string member),
/// SKIPS any line whose first non-space char is `#` (a commented-out
/// `# - "..."` entry does NOT count as present, and does not end the list), and
/// stops at the first dedent to the key's column or left of it.
fn ci_feature_combos(yaml: &str, job: &str) -> BTreeSet<String> {
  let job_key = format!("{job}:");
  let mut combos = BTreeSet::new();
  let mut in_job = false;
  let mut seen_matrix = false;
  let mut key_indent: Option<usize> = None;
  for line in yaml.lines() {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    // Comments and blanks are transparent everywhere: a commented-out entry is
    // skipped (not counted) WITHOUT terminating the list or the job.
    if trimmed.is_empty() || trimmed.starts_with('#') {
      continue;
    }
    if !in_job {
      in_job = indent == JOB_INDENT && trimmed == job_key;
      continue;
    }
    // A dedent to the job's own column (or left of it) ends the job.
    if indent <= JOB_INDENT {
      break;
    }
    let Some(ki) = key_indent else {
      if trimmed == "matrix:" {
        seen_matrix = true;
      } else if seen_matrix && trimmed == "features:" {
        key_indent = Some(indent);
      }
      continue;
    };
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

/// A pinned combo list as an owned set.
fn owned(combos: &[&str]) -> BTreeSet<String> {
  combos.iter().map(|s| (*s).to_string()).collect()
}

/// The intended `features`-job combos as an owned set.
fn intended_ci_combos() -> BTreeSet<String> {
  owned(INTENDED_CI_COMBOS)
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

/// The oracle features are GONE from this crate — they belong to
/// `coremlit-parity` now. A re-added `speaker-oracle` here would drag the
/// unpublished `dia` git source back into the publishable manifest, which is the
/// exact thing this split exists to prevent, so it is pinned as an absence.
#[test]
fn oracle_features_are_not_this_crates() {
  let names = feature_names(&features_block(&manifest()));
  for moved in ["speaker-oracle", "clap-oracle", "vad-bundled"] {
    assert!(
      !names.contains(moved),
      "`{moved}` is declared on coremlit again — it belongs to coremlit-parity, whose \
       oracle deps (`dia`, `textclap`) are unpublished git sources cargo publish rejects"
    );
  }
  assert!(
    names.contains("align-oracle"),
    "`align-oracle` must STAY on coremlit: it only enables a feature of `asry`, a dependency \
     this crate has either way, so moving it would relocate code without removing a git dep"
  );
}

/// `coremlit-parity`'s `[features]` names match its pinned set exactly.
#[test]
fn parity_feature_names_match_the_pinned_set() {
  let actual = feature_names(&features_block(&parity_manifest()));
  let expected: BTreeSet<String> = expected_parity_features()
    .iter()
    .map(|(name, _)| (*name).to_string())
    .collect();
  assert_eq!(
    actual, expected,
    "coremlit-parity [features] names drifted from the pinned oracle feature set"
  );
}

/// Each `coremlit-parity` feature's dependency set matches its pinned set —
/// same exact-set discipline, so an oracle that quietly stops enabling its
/// `coremlit` module feature (or starts enabling another kit's) reds.
#[test]
fn parity_feature_deps_are_pinned_with_no_cross_oracle_leakage() {
  let block = features_block(&parity_manifest());
  for (name, deps) in expected_parity_features() {
    let actual = feature_deps(&block, name);
    let expected: BTreeSet<String> = deps.iter().map(|d| (*d).to_string()).collect();
    assert_eq!(
      actual, expected,
      "coremlit-parity feature `{name}` dependency set drifted"
    );
  }
}

/// `coremlit-parity` must never be publishable: it exists to hold the two
/// unpublished git oracles, so `publish = false` is load-bearing, not tidiness.
#[test]
fn parity_crate_is_never_published() {
  assert!(
    parity_manifest()
      .lines()
      .any(|l| l.trim() == "publish = false"),
    "coremlit-parity must declare `publish = false` — it depends on unpublished git sources"
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
    &ci_feature_combos(&ci_yml(), "features"),
    &intended_ci_combos(),
    "ci.yml `features` job matrix",
  );
}

/// The same exact-set pin for ci.yml's `parity` job — the only place the three
/// third-party oracles are still built in CI now that they are not on this
/// crate's feature matrix. Dropping a row here would silently stop building an
/// oracle; the job scoping is what keeps this set distinct from the one above.
#[test]
fn ci_pins_the_curated_parity_combos() {
  assert_combo_sets_eq(
    &ci_feature_combos(&ci_yml(), "parity"),
    &owned(INTENDED_PARITY_CI_COMBOS),
    "ci.yml `parity` job matrix",
  );
}

/// A well-formed TWO-JOB snippet whose active `features:` lists are exactly the
/// intended sets — the fixture the mutation cases below perturb. Its surrounding
/// keys (a preceding job carrying a `cargo build --features` step, the `features`
/// JOB name above `matrix:`, the `steps:` dedent below, and a SECOND job with its
/// own `matrix.features`) prove the parser enters at the right `features:`, stops
/// at the dedent, and does not fold one job's rows into the other's.
const DOCTORED_MATRIX: &str = r#"
jobs:
  check:
    runs-on: macos-15
    steps:
      - run: cargo build --features whisper --examples
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
          - "clap"
          - "granite"
          - "siglip"
          - "ced"
          - "lid"
          - "whisper,align,speaker,vad,clap,granite,siglip,ced,lid,serde,tracing,nl-recognizer"
          - "whisper,align-oracle,speaker,vad,clap,granite,siglip,ced,lid,serde,tracing,nl-recognizer"
    steps:
      - uses: actions/checkout@v7
  parity:
    runs-on: macos-15
    strategy:
      fail-fast: false
      matrix:
        features:
          - "speaker-oracle"
          - "clap-oracle"
          - "vad-bundled"
          - "speaker-oracle,clap-oracle,vad-bundled"
    steps:
      - uses: actions/checkout@v7
"#;

/// The parser reads the intended set from the well-formed fixture — a guard on
/// the mutation cases below (each perturbs this same fixture).
#[test]
fn ci_combo_parser_reads_the_wellformed_matrix() {
  assert_combo_sets_eq(
    &ci_feature_combos(DOCTORED_MATRIX, "features"),
    &intended_ci_combos(),
    "well-formed doctored matrix",
  );
}

/// Job scoping is real, not incidental: the same fixture yields the PARITY set
/// for the `parity` job, and neither job's rows appear in the other's set. A
/// parser that merged every `matrix.features` in the file would fail both.
#[test]
fn ci_combo_parser_scopes_each_job_separately() {
  let features = ci_feature_combos(DOCTORED_MATRIX, "features");
  let parity = ci_feature_combos(DOCTORED_MATRIX, "parity");
  assert_combo_sets_eq(
    &parity,
    &owned(INTENDED_PARITY_CI_COMBOS),
    "well-formed doctored parity matrix",
  );
  assert!(
    features.is_disjoint(&parity),
    "the two jobs' parsed combo sets bled into each other: {features:?} vs {parity:?}"
  );
}

/// An unknown job name parses to the EMPTY set rather than silently falling
/// back to the first matrix in the file — otherwise a renamed job would keep
/// passing its pin against another job's rows.
#[test]
fn ci_combo_parser_returns_empty_for_an_absent_job() {
  assert!(
    ci_feature_combos(DOCTORED_MATRIX, "no-such-job").is_empty(),
    "an absent job must parse to no combos at all"
  );
}

/// Deleting the bare-core `- ""` entry drops it from the parsed set — the
/// set-equality check must red (the R2 gap: the bare-core run was unpinned).
#[test]
fn ci_combo_check_reds_when_bare_core_is_deleted() {
  let doctored = DOCTORED_MATRIX.replace("          - \"\"\n", "");
  assert_ne!(
    ci_feature_combos(&doctored, "features"),
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
    ci_feature_combos(&doctored, "features"),
    intended_ci_combos(),
    "commenting out the bare-core `- \"\"` entry must make the parsed set differ from the pinned set"
  );
}

/// Dropping any other curated combo (`whisper,vad`) must red as well.
#[test]
fn ci_combo_check_reds_when_a_combo_is_dropped() {
  let doctored = DOCTORED_MATRIX.replace("          - \"whisper,vad\"\n", "");
  assert_ne!(
    ci_feature_combos(&doctored, "features"),
    intended_ci_combos(),
    "dropping the `whisper,vad` combo must make the parsed set differ from the pinned set"
  );
}
