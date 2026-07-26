//! `MODELS_LOCK` governs what CI's `model-tests` job downloads (see the
//! lock file's own header comment and `.github/workflows/ci.yml`'s
//! "Download models (cache miss)" step). These checks are hermetic — no
//! network, no models — and guard the two ways that contract can silently
//! rot: the lock stops parsing, or the workflow stops actually reading it.
//!
//! No TOML crate: this is a deliberately tiny hand-rolled reader over the
//! lock's fixed three-table shape (`["repo/name"]` headers, single-line
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

/// This is a repository-infrastructure check: the workspace's MODELS_LOCK
/// and ci.yml are deliberately NOT packaged with the crate (verified via
/// `cargo package --list`), so a `cargo test` run from the published
/// tarball must SKIP rather than fail `NotFound`.
fn repo_files() -> Option<(PathBuf, PathBuf)> {
  let root = workspace_root();
  let lock = root.join("MODELS_LOCK");
  let workflow = root.join(".github/workflows/ci.yml");
  if lock.is_file() && workflow.is_file() {
    Some((lock, workflow))
  } else {
    eprintln!("models_lock checks skipped: not in the repository workspace");
    None
  }
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
  let Some((lock_path, _)) = repo_files() else {
    return;
  };
  let contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&contents);

  assert_eq!(
    tables.len(),
    3,
    "MODELS_LOCK: expected exactly three tables, found {}",
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
  let Some((lock_path, workflow_path)) = repo_files() else {
    return;
  };
  let lock_contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&lock_contents);
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");

  // ORDER matters as much as membership: ci.yml's parser selects a table by
  // INDEX, so a reordered lock silently re-points every `hf download` at the
  // wrong repo's selector and revision. `assert_eq!` on the Vec pins both.
  let repo_names: Vec<&str> = tables.iter().map(|t| t.name.as_str()).collect();
  assert_eq!(
    repo_names,
    vec![
      "argmaxinc/whisperkit-coreml",
      "openai/whisper-tiny",
      "FinDIT-Studio/embedkit-coreml"
    ],
    "MODELS_LOCK's table names or their order changed — update this test alongside it"
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
    ci_contents.contains("hf download \"$granite_repo\""),
    "download step doesn't invoke hf with a lock-derived $granite_repo"
  );
  assert!(
    ci_contents.contains("--revision \"$model_revision\""),
    "download step doesn't pass a lock-derived --revision for the model repo"
  );
  assert!(
    ci_contents.contains("--revision \"$tokenizer_revision\""),
    "download step doesn't pass a lock-derived --revision for the tokenizer repo"
  );
  assert!(
    ci_contents.contains("--revision \"$granite_revision\""),
    "download step doesn't pass a lock-derived --revision for the granite repo"
  );

  // MODELS_LOCK's header states that the granite bytes are checked against the
  // `CHECKSUMS.sha256` the artifact ships. A lock file that documents an
  // integrity check ci.yml no longer performs is a lock file lying about what
  // CI does — the same rot this test exists to catch one level down.
  assert!(
    ci_contents.contains("shasum -a 256 -c CHECKSUMS.sha256"),
    "ci.yml no longer verifies the granite bundle against its shipped CHECKSUMS.sha256, which \
     MODELS_LOCK's header claims it does"
  );
}
