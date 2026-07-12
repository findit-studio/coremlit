//! `MODELS_LOCK` governs what CI's `model-tests` job downloads (see the
//! lock file's own header comment and `.github/workflows/ci.yml`'s
//! "Download models (cache miss)" step). These checks are hermetic — no
//! network, no models — and guard the two ways that contract can silently
//! rot: the lock stops parsing, or the workflow stops actually reading it.
//!
//! No TOML crate: this is a deliberately tiny hand-rolled reader over the
//! lock's fixed two-table shape (`["repo/name"]` headers, single-line
//! `key = "value"` fields), mirroring the sed/awk parsing `ci.yml` itself
//! performs at CI time — not a general TOML parser.

use std::{fs, path::PathBuf};

struct LockTable {
  name: String,
  fields: Vec<(String, String)>,
}

fn workspace_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn field<'a>(table: &'a LockTable, key: &str) -> Option<&'a str> {
  table
    .fields
    .iter()
    .find(|(k, _)| k == key)
    .map(|(_, v)| v.as_str())
}

/// Parses `["repo/name"]` table headers and, within a table, `key =
/// "value"` fields — in order. Top-level keys before the first table
/// header (`cache-epoch`, an unquoted integer) are cache-key metadata, not
/// part of any table, and are intentionally skipped: this parser only
/// needs the per-table selector/revision fields ci.yml's download step
/// also reads. Panics (via `expect`/`assert`, this is test-only code) on
/// any in-table line that isn't a recognized `key = "value"` field — a
/// real parser failure, not a soft mismatch, since a lock file CI depends
/// on should never silently parse into nothing.
fn parse_lock(contents: &str) -> Vec<LockTable> {
  let mut tables: Vec<LockTable> = Vec::new();
  for line in contents.lines() {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
      continue;
    }
    if let Some(name) = line.strip_prefix("[\"").and_then(|s| s.strip_suffix("\"]")) {
      tables.push(LockTable {
        name: name.to_string(),
        fields: Vec::new(),
      });
      continue;
    }
    let Some(table) = tables.last_mut() else {
      continue; // pre-table key (`cache-epoch`), not this parser's concern
    };
    let (key, value) = line
      .split_once('=')
      .unwrap_or_else(|| panic!("MODELS_LOCK: not a table header or `key = value`: {line:?}"));
    let key = key.trim().to_string();
    let value = value.trim();
    let value = value
      .strip_prefix('"')
      .and_then(|v| v.strip_suffix('"'))
      .unwrap_or_else(|| {
        panic!("MODELS_LOCK: value for {key:?} is not a quoted string: {value:?}")
      });
    table.fields.push((key, value.to_string()));
  }
  tables
}

#[test]
fn lock_parses_and_every_table_has_a_selector_and_a_revision() {
  let contents =
    fs::read_to_string(workspace_root().join("MODELS_LOCK")).expect("MODELS_LOCK reads");
  let tables = parse_lock(&contents);

  assert_eq!(
    tables.len(),
    2,
    "MODELS_LOCK: expected exactly two tables, found {}",
    tables.len()
  );
  for table in &tables {
    let has_selector = field(table, "include").is_some() || field(table, "files").is_some();
    assert!(
      has_selector,
      "MODELS_LOCK: table {:?} has neither `include` nor `files`",
      table.name
    );
    assert!(
      field(table, "revision").is_some(),
      "MODELS_LOCK: table {:?} has no `revision`",
      table.name
    );
  }
}

#[test]
fn ci_workflow_derives_downloads_from_the_lock_instead_of_hardcoding_them() {
  let lock_contents =
    fs::read_to_string(workspace_root().join("MODELS_LOCK")).expect("MODELS_LOCK reads");
  let tables = parse_lock(&lock_contents);
  let ci_contents = fs::read_to_string(workspace_root().join(".github/workflows/ci.yml"))
    .expect(".github/workflows/ci.yml reads");

  let repo_names: Vec<&str> = tables.iter().map(|t| t.name.as_str()).collect();
  assert_eq!(
    repo_names,
    vec!["argmaxinc/whisperkit-coreml", "openai/whisper-tiny"],
    "MODELS_LOCK's table names changed — update this test alongside it"
  );

  // The literal repo strings belong to MODELS_LOCK alone. If ci.yml also
  // spells one out, the workflow is hardcoding what the lock is supposed
  // to govern, and editing MODELS_LOCK silently stops affecting what CI
  // downloads (the exact failure mode this test exists to catch).
  for repo in &repo_names {
    assert!(
      !ci_contents.contains(repo),
      "ci.yml hardcodes locked repo {repo:?}; it must be derived from parsing \
       MODELS_LOCK at runtime instead"
    );
  }

  // The download step must actually read MODELS_LOCK and drive `hf
  // download` from what it parsed out of it, revision included.
  assert!(
    ci_contents.contains("MODELS_LOCK"),
    "ci.yml's model-tests job never references MODELS_LOCK"
  );
  assert!(
    ci_contents.contains("hf download \"$model_repo\""),
    "download step doesn't invoke hf with a lock-derived $model_repo"
  );
  assert!(
    ci_contents.contains("hf download \"$tokenizer_repo\""),
    "download step doesn't invoke hf with a lock-derived $tokenizer_repo"
  );
  assert!(
    ci_contents.contains("--revision \"$model_revision\""),
    "download step doesn't pass a lock-derived --revision for the model repo"
  );
  assert!(
    ci_contents.contains("--revision \"$tokenizer_revision\""),
    "download step doesn't pass a lock-derived --revision for the tokenizer repo"
  );
}
