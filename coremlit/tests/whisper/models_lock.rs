//! `MODELS_LOCK` governs what CI's `model-tests` shards download (see the
//! lock file's own header comment and `.github/workflows/ci.yml`). These
//! checks are hermetic — no network, no models — and guard the four ways that
//! contract can silently rot: the lock stops parsing, the workflow stops
//! actually reading it, the workflow keeps downloading the artifacts but stops
//! RUNNING the gates that read them, or a whole kit/vendor quietly drops out of
//! the shard matrix.
//!
//! That last one is new with sharding, and it is the one no single CI job can
//! see. A shard only knows its own kit: it can refuse a kit that names no
//! table, but it cannot notice a TABLE no shard consumes, or a KNOWN_DEFECTS
//! vendor whose fp16 pins are now swept by nobody. Those are cross-shard facts,
//! and this file is where they are pinned.
//!
//! No TOML crate, and no YAML crate: these are deliberately tiny hand-rolled
//! readers over the lock's fixed `["repo/name"]` + `key = "value"` shape and
//! over ci.yml's fixed matrix indentation, mirroring the sed/awk parsing
//! `ci.yml` itself performs at CI time. The point is to read what those two
//! files literally say, not to model their grammars.

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops — see its module doc.
#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use std::{
  collections::{BTreeMap, BTreeSet},
  fs,
  path::{Path, PathBuf},
};

struct LockTable {
  name: String,
  fields: Vec<(String, String)>,
}

/// [`KNOWN_DEFECTS`](../fp16_guards.rs) / `LOAD_BEARING_NORMS` vendors that NO
/// shard stages, and why.
///
/// This is a real, named coverage gap, not a formality: three of the sweep's
/// nine pinned defects live under these two vendors, and no `model-tests`
/// shard verifies them because MODELS_LOCK has no table that fetches them. They
/// stay local/dev-only. Listing them here is what makes the gap reviewable —
/// `ci_fp16_sweep_shards_cover_every_pinned_vendor` fails on any OTHER pinned
/// vendor going unswept, so a future defect pin cannot join this set silently.
const UNSTAGED_DEFECT_VENDORS: &[(&str, &str)] = &[
  (
    "alignkit",
    "the chordai wav2vec2 aligner is not staged by any shard: `audio::align`'s model gates \
     (ALIGNKIT_TEST_MODELS) are local/dev gates and MODELS_LOCK has no table for it",
  ),
  (
    "argmax-speakerkit",
    "argmaxinc/speakerkit-coreml declares NO license on Hugging Face, so \
     undeclared-means-all-rights-reserved applies to those converted graphs and this repository \
     deliberately does not fetch them in CI at all (NOTICE section 4). Adding a shard for them \
     would reverse a licensing decision, not just widen coverage.",
  ),
];

/// Vendor directories a shard stages that hold NO `.mlmodelc`, and so give the
/// fp16 graph sweep nothing to audit.
///
/// The sweep's vendor manifest is a list of directories that must EXIST, so
/// naming a graph-less tree in it would only assert that a download happened —
/// which the shard's own `probe` and checksum steps already do, more
/// specifically. Every OTHER staged vendor must appear in its shard's manifest.
const GRAPHLESS_VENDORS: &[&str] = &["tokenizers"];

/// The kits permitted to declare `checksum-dir: none`, each with the reason
/// none of its artifact repos ships a `shasum -c`-readable `CHECKSUMS.sha256`
/// and what covers those bytes instead.
///
/// `none` is a DECLARED absence, and the entry here IS the declaration: for
/// every other kit it would mean a verification was silently dropped. A
/// reasoned registry rather than one name, because the property is per-REPO
/// and the reasons genuinely differ — whisper's upstreams publish no digests at
/// all, while lid's publishes them in a format `shasum -c` cannot read. Naming
/// the substitute coverage is the load-bearing half: an exemption whose reason
/// nobody can restate is one nobody can retire.
///
/// Bidirectional, and that is the staleness check
/// ([`ci_shards_every_kit_in_the_lock`]): a kit listed here that does NOT
/// declare `checksum-dir: none` fails, so an exemption cannot outlive the
/// upstream gap it describes. The day one of these repos grows a checksum file
/// and its shard starts verifying against it, this entry has to go with it.
const CHECKSUMLESS_KITS: &[(&str, &str)] = &[
  (
    "whisper",
    "neither argmaxinc/whisperkit-coreml nor openai/whisper-tiny publishes digests in any form, \
     and both tables are still on `revision = \"main\"` (MODELS_LOCK's LOUD FOLLOW-UP), so there \
     is nothing to verify against that would mean anything. Their bytes are covered by the \
     whisper gates' own model_io pins and by the fp16_guards graph sweep.",
  ),
  (
    "lid",
    "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML ships NO CHECKSUMS.sha256: its per-file \
     digests live in `artifact_manifest.json`, which `shasum -c` cannot read. Pointing \
     `checksum-file` at it would fail on a correct tree. The bytes are covered instead by \
     `ARTIFACT_SHA256` in tests/lid/common/mod.rs — an EXACT file set, so a missing or an added \
     file reds too — verified by `artifact_matches_the_pinned_sha_manifest`, which this shard \
     runs, and by the fp16_guards graph sweep.",
  ),
];

/// This is a repository-infrastructure check: the workspace's MODELS_LOCK
/// and ci.yml are deliberately NOT packaged with the crate (verified via
/// `cargo package --list`), so a `cargo test` run from the published
/// tarball must SKIP rather than fail `NotFound`.
fn repo_files() -> Option<(PathBuf, PathBuf)> {
  // `try_…`, not the asserting form: outside a workspace there is nothing to
  // find and this check is meant to SKIP, which is the same answer it gives
  // when the lock and the workflow are simply not packaged.
  let root = workspace_root::try_workspace_root()?;
  let lock = root.join("MODELS_LOCK");
  let workflow = root.join(".github/workflows/ci.yml");
  if lock.is_file() && workflow.is_file() {
    Some((lock, workflow))
  } else {
    eprintln!("models_lock checks skipped: not in the repository workspace");
    None
  }
}

/// The step condition every check in a `model-tests` shard must carry.
///
/// `!cancelled()` is what makes a check independent of the ones before it;
/// `steps.download.outcome != 'failure'` keeps the ONE genuine dependency —
/// nothing below can run without the artifacts. `outcome` is `skipped`, not
/// `failure`, when the download is itself skipped on a cache hit, so the
/// common path stays unaffected.
const GATE_GUARD: &str = "if: ${{ !cancelled() && steps.download.outcome != 'failure' }}";

/// The `model-tests` job's steps, in order, each as its own raw YAML text.
///
/// Text-based like `parse_lock`, and for the same reason: the point is to read
/// what ci.yml literally says, not to model YAML. A step begins at exactly six
/// columns of indent followed by `- name:`/`- uses:`/`- run:`, which no line
/// inside a `run: |` block (indented ten) can imitate, and the job ends at the
/// next key indented two columns.
fn model_tests_steps(ci: &str) -> Vec<String> {
  let job = model_tests_job(ci);
  let mut steps: Vec<String> = Vec::new();
  for line in job.lines() {
    let body = line.trim_start();
    if !body.is_empty() && line.len() - body.len() == 2 {
      break;
    }
    if ["- name:", "- uses:", "- run:"].iter().any(|start| {
      line
        .strip_prefix("      ")
        .is_some_and(|l| l.starts_with(start))
    }) {
      steps.push(String::new());
    }
    if let Some(step) = steps.last_mut() {
      step.push_str(line);
      step.push('\n');
    }
  }
  steps
}

fn model_tests_job(ci: &str) -> &str {
  ci.split_once("\n  model-tests:\n")
    .expect("ci.yml has no `model-tests` job")
    .1
}

/// One `strategy.matrix.include` row of the `model-tests` job, as ordered
/// `key -> value` pairs.
///
/// Same text-based contract as [`parse_lock`]: rows begin at exactly ten
/// columns of indent followed by `- kit: `, their keys sit at twelve, and a
/// `key: |` block scalar's body sits deeper still. Comment lines are skipped at
/// every level. A row that does not start with `kit` is a parse failure, not a
/// soft mismatch — the kit is what selects the row's MODELS_LOCK tables, so a
/// row without one is unreadable by the workflow too.
fn matrix_rows(ci: &str) -> Vec<BTreeMap<String, String>> {
  let matrix = model_tests_job(ci)
    .split_once("\n        include:\n")
    .expect("ci.yml's model-tests job has no `strategy.matrix.include`")
    .1;

  let mut rows: Vec<BTreeMap<String, String>> = Vec::new();
  let mut block: Option<(String, Vec<String>)> = None;
  for line in matrix.lines() {
    let body = line.trim_start();
    let indent = line.len() - body.len();

    // A block scalar's body is anything indented deeper than its key; the
    // first line that is not closes it.
    if block.is_some() {
      if body.is_empty() || indent > 12 {
        let (_, lines) = block.as_mut().expect("block is Some");
        lines.push(body.to_string());
        continue;
      }
      let (key, lines) = block.take().expect("block is Some");
      rows
        .last_mut()
        .expect("a block scalar belongs to a row")
        .insert(key, lines.join("\n"));
    }

    // The matrix ends at the job's next key (`env:`, indented four).
    if !body.is_empty() && indent < 10 {
      break;
    }
    if body.is_empty() || body.starts_with('#') {
      continue;
    }
    if indent == 10 {
      let kit = body
        .strip_prefix("- kit: ")
        .unwrap_or_else(|| panic!("ci.yml matrix row does not begin with `- kit: `: {line:?}"));
      let mut row = BTreeMap::new();
      row.insert("kit".to_string(), kit.trim().to_string());
      rows.push(row);
      continue;
    }
    assert_eq!(
      indent, 12,
      "ci.yml matrix line has an unexpected indent; this reader needs 10 for a row and 12 for its \
       keys: {line:?}"
    );
    let (key, value) = body
      .split_once(": ")
      .or_else(|| body.strip_suffix(':').map(|k| (k, "")))
      .unwrap_or_else(|| panic!("ci.yml matrix line is not a `key: value`: {line:?}"));
    let row = rows
      .last_mut()
      .unwrap_or_else(|| panic!("ci.yml matrix key outside any row: {line:?}"));
    if value.trim() == "|" {
      block = Some((key.to_string(), Vec::new()));
      continue;
    }
    row.insert(key.to_string(), unquote(value.trim()).to_string());
  }
  if let Some((key, lines)) = block.take() {
    rows
      .last_mut()
      .expect("a block scalar belongs to a row")
      .insert(key, lines.join("\n"));
  }
  assert!(
    !rows.is_empty(),
    "ci.yml's model-tests matrix parsed to no rows — either the job lost its matrix, or this \
     reader stopped matching its layout"
  );
  rows
}

/// Strips the single quotes YAML needs around a scalar that starts with a regex
/// metacharacter. Single-quoted YAML performs no backslash escaping, so the
/// inner text is the literal value.
fn unquote(value: &str) -> &str {
  value
    .strip_prefix('\'')
    .and_then(|v| v.strip_suffix('\''))
    .unwrap_or(value)
}

fn row_field<'a>(row: &'a BTreeMap<String, String>, key: &str) -> &'a str {
  row.get(key).map_or("", String::as_str)
}

fn require_field<'a>(row: &'a BTreeMap<String, String>, key: &str) -> &'a str {
  let kit = row_field(row, "kit");
  let value = row_field(row, key);
  assert!(
    !value.is_empty(),
    "ci.yml's {kit:?} matrix row has no {key:?}; every shard must declare one"
  );
  value
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
/// needs the per-table fields ci.yml's download step also reads. Panics
/// (via `expect`/`assert`, this is test-only code) on any in-table line
/// that isn't a recognized `key = "value"` field — a real parser failure,
/// not a soft mismatch, since a lock file CI depends on should never
/// silently parse into nothing.
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

/// The vendor directory a `local-dir` stages into: `Models/<vendor>/...`.
fn vendor_of(local_dir: &str) -> &str {
  local_dir
    .strip_prefix("Models/")
    .unwrap_or_else(|| {
      panic!("MODELS_LOCK `local-dir` {local_dir:?} does not start with `Models/`")
    })
    .split('/')
    .next()
    .expect("split always yields one element")
}

/// The vendor prefix of every [`KNOWN_DEFECTS`] pin in `tests/fp16_guards.rs`,
/// read from that file's source text.
///
/// The sweep lives in a different test binary, so its roster cannot be imported
/// — but it CAN be read, which is the same thing this file already does to hold
/// ci.yml's overlay pins against `tests/speaker/model_io.rs`.
fn fp16_pinned_vendors() -> BTreeSet<String> {
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fp16_guards.rs");
  let source = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  // BOTH of the sweep's registers. `KNOWN_DEFECTS` pins guards whose constant
  // vanishes in fp16; `LOAD_BEARING_NORMS` pins artifacts where a SURVIVING
  // epsilon is the only thing guarding a channel whose stored variance is zero.
  // Either kind of pin is verified nowhere if no shard sweeps its vendor, so
  // the coverage claim below has to reason about the union.
  let vendors: BTreeSet<String> = [
    "const KNOWN_DEFECTS: &[KnownDefect] = &[",
    "const LOAD_BEARING_NORMS: &[LoadBearingNorm] = &[",
  ]
  .into_iter()
  .flat_map(|decl| pin_vendors_in(&source, decl))
  .collect();
  assert!(
    !vendors.is_empty(),
    "no `path:` entries parsed out of tests/fp16_guards.rs's pin registers — the sweep's roster \
     moved or this reader stopped matching it, and every coverage claim below would be vacuous"
  );
  vendors
}

/// The vendor prefixes of every `path: "…",` entry inside the `decl` register of
/// `source`. Text-based like the rest of this file's readers, and it PANICS on a
/// register it cannot find: a rename that made this silently return nothing
/// would turn every coverage assertion below vacuous.
fn pin_vendors_in(source: &str, decl: &str) -> BTreeSet<String> {
  let block = source
    .split_once(decl)
    .unwrap_or_else(|| panic!("tests/fp16_guards.rs no longer declares `{decl}`"))
    .1
    .split_once("\n];")
    .unwrap_or_else(|| panic!("tests/fp16_guards.rs's `{decl}` list is unterminated"))
    .0;
  let vendors: BTreeSet<String> = block
    .lines()
    .filter_map(|line| line.trim().strip_prefix("path: \""))
    .map(|rest| {
      let path = rest
        .strip_suffix("\",")
        .unwrap_or_else(|| panic!("`{decl}` path entry is not `path: \"...\",`: {rest:?}"));
      path
        .split('/')
        .next()
        .expect("split always yields one element")
        .to_string()
    })
    .collect();
  assert!(
    !vendors.is_empty(),
    "no `path:` entries parsed out of `{decl}` in tests/fp16_guards.rs"
  );
  vendors
}

#[test]
fn lock_parses_and_every_table_is_complete() {
  let Some((lock_path, _)) = repo_files() else {
    return;
  };
  let contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&contents);

  assert!(
    !tables.is_empty(),
    "MODELS_LOCK parsed to no tables — the lock lost its `[\"repo/name\"]` shape, or this reader \
     stopped matching it"
  );
  for table in &tables {
    let has_selector = field(table, "include").is_some() || field(table, "files").is_some();
    assert!(
      has_selector,
      "MODELS_LOCK: table {:?} has neither `include` nor `files`",
      table.name
    );
    for key in ["kit", "revision", "local-dir"] {
      assert!(
        field(table, key).is_some_and(|v| !v.is_empty()),
        "MODELS_LOCK: table {:?} has no {key:?}. Since ci.yml selects tables by `kit` rather than \
         by index, a table missing one is unreachable: no shard would ever download it, and the \
         lock would document a download CI never performs.",
        table.name
      );
    }
  }
}

/// ci.yml must DERIVE its downloads from the lock, not restate them.
///
/// The failure this catches is a workflow that keeps MODELS_LOCK around as a
/// cache key while hardcoding the repos it actually fetches — at which point
/// editing the lock silently stops affecting what CI downloads.
#[test]
fn ci_workflow_derives_downloads_from_the_lock_instead_of_hardcoding_them() {
  let Some((lock_path, workflow_path)) = repo_files() else {
    return;
  };
  let lock_contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&lock_contents);
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");

  // The literal repo strings belong to MODELS_LOCK alone.
  for table in &tables {
    assert!(
      !ci_contents.contains(&table.name),
      "ci.yml hardcodes locked repo {:?}; it must be derived from parsing MODELS_LOCK at runtime \
       instead",
      table.name
    );
  }

  assert!(
    ci_contents.contains("MODELS_LOCK"),
    "ci.yml's model-tests job never references MODELS_LOCK"
  );
  // One generic `hf download` per selected table, every argument lock-derived.
  // Hardcoding any of these four is how a lock edit stops mattering.
  for needle in [
    "hf download \"$repo\"",
    "--revision \"$revision\"",
    "--local-dir \"$localdir\"",
    "-v want=\"$KIT\"",
  ] {
    assert!(
      ci_contents.contains(needle),
      "ci.yml's download step no longer builds its `hf download` from the lock ({needle:?} is \
       absent), so editing MODELS_LOCK would stop changing what CI fetches"
    );
  }

  // The `table_count -ne 7` tripwire that guarded the old index-selected parser
  // is gone with the indices. Its replacement must still be here: every table
  // has to carry a `kit`, or it is unreachable. The count comparison is what
  // catches BOTH an added table and a table that lost its tag — the
  // empty-extraction guard cannot, since it only inspects tables that WERE
  // selected.
  assert!(
    ci_contents.contains("grep -cE '^kit[[:space:]]*=' \"$lock\""),
    "ci.yml's download step no longer counts MODELS_LOCK's `kit` fields against its table count, \
     so a table added (or a tag deleted) would be silently unreachable — the failure the old \
     `table_count` pin existed to prevent"
  );
  assert!(
    ci_contents.contains("[ \"$table_count\" -ne \"$kit_count\" ]"),
    "ci.yml's download step counts kits but no longer compares them to the table count"
  );
  // ...and a shard whose own kit selects nothing must fail rather than gate a
  // bare checkout. Checked in the download step AND in the overlay step, which
  // is the one that runs on a cache hit too.
  assert!(
    ci_contents
      .matches("MODELS_LOCK defines no table with kit")
      .count()
      >= 2,
    "ci.yml must refuse a shard whose kit matches no MODELS_LOCK table in BOTH the download step \
     and a step that runs on cache hits — otherwise a matrix row with no table would silently \
     gate an empty tree whenever the cache was warm"
  );
}

/// The shard matrix and the lock's kits must cover each other EXACTLY.
///
/// A kit in the matrix with no table downloads nothing and gates a bare
/// checkout. A table whose kit no shard consumes is a download CI never
/// performs — the same lie the old `table_count` tripwire caught, in the shape
/// the kit scheme allows. A shard can detect the first from the inside; only a
/// hermetic check that sees BOTH files can detect the second.
///
/// This also pins each row's completeness, because a shard is only as good as
/// its data: an empty `probe` gates without checking the artifact arrived, an
/// empty `gates` plan gates nothing, and a `checksum-dir` that went missing
/// drops a verification silently.
#[test]
fn ci_shards_every_kit_in_the_lock() {
  let Some((lock_path, workflow_path)) = repo_files() else {
    return;
  };
  let lock_contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&lock_contents);
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");
  let rows = matrix_rows(&ci_contents);

  let lock_kits: BTreeSet<&str> = tables
    .iter()
    .map(|t| field(t, "kit").expect("checked by lock_parses_and_every_table_is_complete"))
    .collect();
  let shard_kits: BTreeSet<&str> = rows.iter().map(|r| row_field(r, "kit")).collect();
  assert_eq!(
    shard_kits.len(),
    rows.len(),
    "ci.yml's model-tests matrix has two shards for the same kit; they would race for one \
     Models/ tree and one cache key"
  );
  assert_eq!(
    lock_kits, shard_kits,
    "MODELS_LOCK's kits and ci.yml's model-tests shards have drifted apart. A kit in the lock \
     with no shard is a download nothing performs; a shard with no table downloads nothing and \
     then gates a bare checkout. Add the missing table or the missing matrix row."
  );

  let checksumless: BTreeMap<&str, &str> = CHECKSUMLESS_KITS.iter().copied().collect();
  assert_eq!(
    checksumless.len(),
    CHECKSUMLESS_KITS.len(),
    "CHECKSUMLESS_KITS lists a kit twice; the second reason would be unreachable"
  );
  let mut declared_checksumless: BTreeSet<&str> = BTreeSet::new();

  for row in &rows {
    let kit = row_field(row, "kit");
    let local_dirs: Vec<&str> = tables
      .iter()
      .filter(|t| field(t, "kit") == Some(kit))
      .map(|t| field(t, "local-dir").expect("checked above"))
      .collect();

    // Every path the shard names must live inside a directory the lock actually
    // stages, or the cache would save an empty tree and the probe would guard a
    // path nothing writes.
    let mut paths: Vec<&str> = Vec::new();
    for key in ["cache", "probe"] {
      paths.extend(require_field(row, key).split_whitespace());
    }
    let checksum_dir = require_field(row, "checksum-dir");
    if checksum_dir == "none" {
      let reason = checksumless.get(kit).unwrap_or_else(|| {
        panic!(
          "ci.yml's {kit:?} shard declares `checksum-dir: none`, but CHECKSUMLESS_KITS does not \
           record that kit's repos as shipping no shasum-readable CHECKSUMS.sha256 (it records \
           {:?}). Either a verification was dropped, or the absence is real — in which case add \
           {kit:?} here WITH the reason and the coverage that stands in for it.",
          checksumless.keys().collect::<Vec<_>>()
        )
      });
      assert!(
        !reason.trim().is_empty(),
        "CHECKSUMLESS_KITS records {kit:?} with an empty reason; the reason is the declaration"
      );
      declared_checksumless.insert(kit);
    } else {
      paths.push(checksum_dir);
      assert!(
        !require_field(row, "checksum-file").is_empty(),
        "ci.yml's {kit:?} shard names a checksum directory but no checksum file"
      );
    }
    for path in paths {
      assert!(
        local_dirs
          .iter()
          .any(|dir| path == *dir || path.starts_with(&format!("{dir}/"))),
        "ci.yml's {kit:?} shard names {path:?}, which is not under any of that kit's MODELS_LOCK \
         `local-dir` destinations ({local_dirs:?}) — the shard would cache, probe or verify a \
         tree its own download never writes"
      );
    }

    // A filtered checksum verification must pin its line count, or a filter
    // that rots into matching nothing "verifies" an empty set.
    let filter = row_field(row, "checksum-filter");
    let lines = row_field(row, "checksum-lines");
    assert_eq!(
      filter.is_empty(),
      lines.is_empty(),
      "ci.yml's {kit:?} shard has a checksum-filter without a checksum-lines pin (or the \
       reverse); the count is what makes the filter non-vacuous"
    );
    if !lines.is_empty() {
      assert!(
        lines.parse::<u32>().is_ok_and(|n| n > 0),
        "ci.yml's {kit:?} shard pins checksum-lines at {lines:?}, which is not a positive count"
      );
    }

    // The gate plan must select at least one target under at least one feature
    // set, or the shard downloads artifacts and gates nothing.
    let gates = require_field(row, "gates");
    let groups: Vec<&str> = gates.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
      !groups.is_empty(),
      "ci.yml's {kit:?} shard has an empty gate plan"
    );
    for group in groups {
      let mut parts = group.splitn(3, '|');
      let features = parts.next().unwrap_or_default();
      let selectors = parts.next().unwrap_or_default();
      assert!(
        selectors.split_whitespace().next().is_some(),
        "ci.yml's {kit:?} shard has a gate group with no test selector: {group:?} (features \
         {features:?})"
      );
    }
  }

  // The staleness half of the checksum exemption. An entry in CHECKSUMLESS_KITS
  // is a claim about an upstream repo, and upstream repos change: the day one
  // starts publishing a `shasum -c`-readable CHECKSUMS.sha256 and its shard
  // begins verifying against it, the exemption is a lie that would let the NEXT
  // dropped verification hide behind it. So the registry must name only kits
  // that actually declare `none`, today — and only kits that still have a
  // shard, since a deleted shard's exemption is the same stale claim.
  for (kit, _) in CHECKSUMLESS_KITS {
    assert!(
      shard_kits.contains(kit),
      "CHECKSUMLESS_KITS names {kit:?}, which has no model-tests shard in ci.yml. The exemption \
       describes a shard that no longer exists — drop the entry."
    );
    assert!(
      declared_checksumless.contains(kit),
      "CHECKSUMLESS_KITS records {kit:?} as staging no repo with a shasum-readable \
       CHECKSUMS.sha256, but its shard now names a checksum-dir. If the repo grew one, that is \
       good news — delete this entry so the exemption cannot outlive its reason and shelter the \
       next verification somebody drops."
    );
  }
}

/// MODELS_LOCK stages ONE CED size (`ced-tiny`: 10.64 MB of the artifact repo's
/// 234 MB), but `tests/ced/*` declares every gate four times — one `#[ignore]`d
/// module per size — and an ignored-ONLY run selects ALL FOUR. So the `ced`
/// shard's gate plan must filter to the staged size, and the two must agree.
///
/// Both drift directions are silent without this pin. Widen the lock's
/// `include` to another size and CI downloads bytes no gate reads; change the
/// gate filter and CI runs gates against a bundle the lock never staged.
#[test]
fn ci_runs_the_ced_gates_for_exactly_the_size_the_lock_stages() {
  let Some((lock_path, workflow_path)) = repo_files() else {
    return;
  };
  let lock_contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&lock_contents);
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");

  let ced = tables
    .iter()
    .find(|t| field(t, "kit") == Some("ced"))
    .expect("MODELS_LOCK has no `ced` kit table");
  let include = field(ced, "include").expect("the CED table has an `include` selector");

  // `ced-<size>/*` — exactly one size, no glob over the family. A selector this
  // parse rejects (`ced-*/*`, `*`, a bare file) is one whose staged size cannot
  // be named, so it cannot be checked against the gate filter either.
  let size = include
    .strip_prefix("ced-")
    .and_then(|rest| rest.strip_suffix("/*"))
    .unwrap_or_else(|| {
      panic!(
        "MODELS_LOCK's CED `include` is {include:?}; this pin needs the single-size \
         `ced-<size>/*` shape so the staged size can be matched against the shard's gate filter"
      )
    });
  assert!(
    !size.is_empty() && size.chars().all(|c| c.is_ascii_lowercase()),
    "MODELS_LOCK's CED `include` names {size:?}, which is not a CedModel size name"
  );
  assert_eq!(
    field(ced, "local-dir"),
    Some("Models/ced"),
    "MODELS_LOCK doesn't stage CED into Models/ced, the family root tests/ced/common/mod.rs \
     resolves without CED_TEST_MODELS"
  );

  let row = matrix_rows(&ci_contents)
    .into_iter()
    .find(|r| row_field(r, "kit") == "ced")
    .expect("ci.yml has no `ced` shard");
  let bundle = format!("Models/ced/ced-{size}/ced_{size}.mlmodelc");
  for key in ["probe", "checksum-dir"] {
    assert!(
      row_field(&row, key).split_whitespace().any(|p| p == bundle),
      "MODELS_LOCK stages CED size {size:?}, but the ced shard's {key:?} does not name its bundle \
       {bundle:?} — the absent-artifact guard and the checksum verification would be pointed at a \
       different size than the one downloaded"
    );
  }

  // The gate plan's filter must carry the size. The gate runner takes its
  // anti-vacuum `--list --ignored` count THROUGH THE SAME FILTER as the run, so
  // pinning the filter pins both: an unfiltered plan would select all four
  // sizes and fail on the three MODELS_LOCK never staged, and a filter that no
  // longer matches would list zero and trip the vacuum guard.
  let filters: Vec<&str> = row_field(&row, "gates")
    .lines()
    .filter(|l| !l.trim().is_empty())
    .map(|group| group.splitn(3, '|').nth(2).unwrap_or_default())
    .collect();
  assert_eq!(
    filters,
    vec![format!("{size}::")],
    "the ced shard's gate plan must filter every group on {size:?}, the one size MODELS_LOCK \
     stages"
  );
  assert!(
    ci_contents.contains("-- --list --ignored ${filter:+\"$filter\"}")
      && ci_contents.contains("-- --ignored ${filter:+\"$filter\"}"),
    "ci.yml's gate runner no longer applies the plan's filter to BOTH the anti-vacuum `--list` \
     count and the run, so a deleted or renamed `{size}` module could still count the other \
     sizes' gates and then run none"
  );
}

/// Every `#[ignore]`d test under `dir`, as `(full libtest path, ignore reason)`.
///
/// Text-based like every other reader in this file, and for the same reason:
/// the alternative is asking libtest for the list, which means building the
/// `speaker` feature and — for the gates this is used to check — having the
/// models the check exists to run without.
///
/// The shape it needs is the one `src/` uses throughout: `#[ignore = "..."]` on
/// ONE line (asserted, so a reason that grows a line continuation fails here
/// rather than silently dropping a gate), the `fn <name>` somewhere below it in
/// the same attribute block, and the module path taken from the file's own path
/// under `src/` — the sibling-`tests.rs` layout.
fn ignored_gates(src_root: &Path, dir: &Path) -> Vec<(String, String)> {
  let mut gates: Vec<(String, String)> = Vec::new();
  let mut stack = vec![dir.to_path_buf()];
  while let Some(current) = stack.pop() {
    let entries = fs::read_dir(&current)
      .unwrap_or_else(|e| panic!("{} reads: {e}", current.display()))
      .map(|e| e.expect("a directory entry reads").path());
    for path in entries {
      if path.is_dir() {
        stack.push(path);
        continue;
      }
      if path.extension().is_none_or(|ext| ext != "rs") {
        continue;
      }
      let module = path
        .strip_prefix(src_root)
        .expect("scanned under src/")
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .filter(|c| c != "mod")
        .collect::<Vec<_>>()
        .join("::");
      let contents = fs::read_to_string(&path).expect("a source file reads");
      let lines: Vec<&str> = contents.lines().collect();
      for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.trim_start().strip_prefix("#[ignore") else {
          continue;
        };
        assert!(
          rest.trim_end().ends_with(']'),
          "{}:{}: this reader needs the whole `#[ignore ...]` attribute on one line",
          path.display(),
          i + 1
        );
        let reason = match rest.split_once('"').and_then(|(_, r)| r.rsplit_once('"')) {
          Some((reason, _)) => reason.to_string(),
          None => String::new(),
        };
        let name = lines[i + 1..]
          .iter()
          .find_map(|l| l.trim_start().strip_prefix("fn "))
          .and_then(|l| l.split(['(', '<']).next())
          .unwrap_or_else(|| {
            panic!(
              "{}:{}: an `#[ignore]` attribute with no `fn` after it",
              path.display(),
              i + 1
            )
          });
        gates.push((format!("{module}::{name}"), reason));
      }
    }
  }
  gates.sort();
  gates
}

/// The `speaker` shard runs the library's own model gates, and the one thing it
/// must NOT run is a gate that reads a tree no runner is allowed to fetch.
///
/// 230 of this repository's model gates are `#[ignore]`d unit tests inside the
/// pipeline modules rather than `tests/` binaries (#61), and
/// `--features speaker --lib` lists 42 of them. ELEVEN read
/// `ARGMAX_TEST_MODELS` — the `argmax-speakerkit` tree
/// [`UNSTAGED_DEFECT_VENDORS`] records as deliberately unfetched, for a
/// LICENSING reason rather than a cost one — so the shard's gate plan skips
/// exactly those and runs the other 31.
///
/// Both drift directions are pinned, and they fail in opposite ways. A new
/// argmax-tree gate the filter does not name turns the shard red on a missing
/// bundle: loud, but only after a 108 MB download. A filter that grows to match
/// a SPEAKERKIT-only gate drops it from CI with no signal at all — the gate
/// runner's anti-vacuum `--list` count is taken through the same filter as the
/// run, so both numbers fall together and neither reaches zero.
#[test]
fn ci_speaker_lib_gates_skip_exactly_the_unstaged_argmax_tree() {
  let Some((_, workflow_path)) = repo_files() else {
    return;
  };
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");
  let row = matrix_rows(&ci_contents)
    .into_iter()
    .find(|r| row_field(r, "kit") == "speaker")
    .expect("ci.yml has no `speaker` shard");

  // The `@lib` group must exist at all. Without it the shard downloads 108 MB,
  // runs three `tests/` targets, and leaves the larger half of the kit's gates
  // unrun — the gap this pin was added with.
  let lib_group = require_field(&row, "gates")
    .lines()
    .find(|group| {
      group
        .split('|')
        .nth(1)
        .is_some_and(|selectors| selectors.split_whitespace().any(|s| s == "@lib"))
    })
    .map(str::to_string)
    .expect(
      "ci.yml's speaker shard has no `@lib` gate group. The library's own speaker model gates \
       read exactly the two graphs this shard already stages, so dropping the group means the \
       shard downloads for them and then runs none of them.",
    );
  let filter = lib_group.splitn(3, '|').nth(2).unwrap_or_default();

  // The gate runner hands `filter` to libtest as ONE word, so an exclusion is
  // spelled `--skip=<substring>`. Any other token here (a positional include
  // filter, a bare `--skip`) would change which gates run WITHOUT changing the
  // skip set this check models, so it is refused rather than ignored.
  let skips: Vec<&str> = filter
    .split_whitespace()
    .map(|token| {
      token.strip_prefix("--skip=").unwrap_or_else(|| {
        panic!(
          "ci.yml's speaker `@lib` group has the filter {filter:?}; this pin models libtest's \
           `--skip=<substring>` exclusions only, and a token of another shape would change which \
           gates run without changing what is checked here"
        )
      })
    })
    .collect();
  assert!(
    !skips.is_empty(),
    "ci.yml's speaker `@lib` group has no filter, so it would run the eleven gates that load \
     Models/argmax-speakerkit — a tree no runner fetches (see UNSTAGED_DEFECT_VENDORS)"
  );

  // This binary only ever compiles into the `coremlit` package, so its own
  // manifest dir IS the crate root — no hop, nothing to miscount.
  let src_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
  let gates = ignored_gates(&src_root, &src_root.join("audio/speaker"));
  assert!(
    gates.len() > 30,
    "only {} `#[ignore]`d gates found under src/audio/speaker; this reader has stopped matching \
     the source layout, and every assertion below would pass vacuously",
    gates.len()
  );

  // `argmax` in the ignore REASON is what marks a gate as needing the unstaged
  // tree; both wordings say it ("requires local argmax speakerkit models
  // (ARGMAX_TEST_MODELS)" and the two-root "requires local argmax + speakerkit
  // models (both env vars)"). The reason is the right marker rather than the
  // module path, because `source::tests` holds one gate of each kind.
  let mut needs_argmax = 0usize;
  for (name, reason) in &gates {
    let skipped = skips.iter().any(|s| name.contains(s));
    if reason.contains("argmax") {
      needs_argmax += 1;
      assert!(
        skipped,
        "the in-lib gate {name:?} says it needs the argmax tree ({reason:?}), but the speaker \
         shard's `@lib` filter {filter:?} does not skip it. CI deliberately does not fetch that \
         tree (UNSTAGED_DEFECT_VENDORS), so the shard would fail on a missing bundle. Name the \
         gate so an existing `--skip=` substring matches it, or add one."
      );
    } else {
      assert!(
        !skipped,
        "the speaker shard's `@lib` filter {filter:?} skips {name:?}, which needs only the \
         speakerkit tree this shard stages ({reason:?}). A skip that widens onto a runnable gate \
         drops it from CI silently: the anti-vacuum `--list` count is taken through this same \
         filter, so it falls with the run instead of reaching zero."
      );
    }
  }
  // Non-vacuity for the branch above: gates were found, but if none of them
  // declared an argmax dependency the completeness half checked nothing at all.
  assert!(
    needs_argmax > 0,
    "no in-lib speaker gate names argmax in its `#[ignore]` reason, so the filter this shard \
     carries is excluding gates for a dependency nothing declares any more"
  );
  for skip in &skips {
    assert!(
      gates
        .iter()
        .any(|(name, reason)| name.contains(skip) && reason.contains("argmax")),
      "the speaker shard's `@lib` filter skips {skip:?}, which matches no in-lib gate that needs \
       the argmax tree — a stale exclusion can only be hiding a gate CI could run"
    );
  }
}

/// The two `speaker` tables are the one place in MODELS_LOCK where table order
/// decides which BYTES ship, not merely which parser variable gets which value.
///
/// Both download into `Models/speakerkit/` and both publish
/// `pyannote_segmentation.mlmodelc` and `wespeaker.mlmodelc`, so the last one
/// downloaded wins those two filenames. The base layer's copies are
/// FluidInference's PRE-REPAIR conversions: contract-identical — same feature
/// names, shapes and dtypes — so they load, run, and pass every structural gate
/// while an inert `log(epsilon = 0)` saturates `segments` to -45440 on the
/// default `ComputeUnits::All` placement (issue #15). Only bytes tell them
/// apart.
///
/// So this pins three things that must move together: the lock's order WITHIN
/// the kit, the shard's `overlay-pins`, and the fact that ci.yml requires those
/// pins of any layered kit rather than only of this one. That last part is what
/// turns a wrong order into a named failure instead of anonymous "sha256 drift"
/// reported a full build later by `speaker_model_io` — and the pins are a
/// deliberate SECOND copy of that gate's, so this test holds the two together.
/// A THIRD copy — `tests/speaker/common/mod.rs`'s `skipped_for_stale_overlay`,
/// the parity gates' own runtime hash-vs-LOCK check — is held to the same two
/// below alongside them.
#[test]
fn ci_stages_the_speakerkit_overlay_last_and_proves_it_won() {
  let Some((lock_path, workflow_path)) = repo_files() else {
    return;
  };
  let lock_contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&lock_contents);
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");

  let index_of = |name: &str| {
    tables
      .iter()
      .position(|t| t.name == name)
      .unwrap_or_else(|| panic!("MODELS_LOCK has no {name:?} table"))
  };
  let base = index_of("FluidInference/speaker-diarization-coreml");
  let overlay = index_of("FinDIT-Studio/speakerkit-coreml");
  for (index, name) in [(base, "base"), (overlay, "overlay")] {
    assert_eq!(
      field(&tables[index], "kit"),
      Some("speaker"),
      "MODELS_LOCK's speakerkit {name} table is not `kit = \"speaker\"`, so the speaker shard \
       would not download it at all"
    );
    assert_eq!(
      field(&tables[index], "local-dir"),
      Some("Models/speakerkit"),
      "MODELS_LOCK's speakerkit {name} table no longer stages into Models/speakerkit, so the two \
       layers no longer collide — which silently retires the ordering invariant below"
    );
  }
  assert!(
    base < overlay,
    "MODELS_LOCK stages the speakerkit OVERLAY (table {}) before the BASE layer (table {}), so \
     the base layer's PRE-REPAIR pyannote_segmentation/wespeaker graphs would overwrite the \
     fp16-guard-repaired ones the pipeline ships. The shard downloads a kit's tables in LOCK \
     ORDER, so this file's order is what picks the winner.",
    overlay + 1,
    base + 1
  );

  // The overlay selector must not be widened to a bare `*.mlmodelc/*`: that
  // would also stage the overlay repo's `wespeaker_int8` re-conversion over the
  // base layer's bytes, breaking both the `wespeaker_v2 == wespeaker_int8`
  // byte-identity gate and that bundle's fp16_guards pin (the re-palettization
  // regresses clip 14's int8 ANE arm from 0.8178 % to 1.4860 % DER).
  let overlay_include =
    field(&tables[overlay], "include").expect("the speakerkit overlay table has an `include`");
  for bundle in ["pyannote_segmentation.mlmodelc/*", "wespeaker.mlmodelc/*"] {
    assert!(
      overlay_include.split(' ').any(|p| p == bundle),
      "MODELS_LOCK's speakerkit overlay `include` ({overlay_include:?}) does not name {bundle:?}, \
       so that shipping graph would stay the base layer's pre-repair conversion"
    );
  }
  assert!(
    !overlay_include.split(' ').any(|p| p == "*.mlmodelc/*"),
    "MODELS_LOCK's speakerkit overlay `include` ({overlay_include:?}) globs every bundle, which \
     stages that repo's NOT-adopted wespeaker_int8 re-conversion over the base layer's bytes"
  );

  // ci.yml must demand overlay pins of ANY kit whose tables share a local-dir,
  // not just of this one — otherwise a second layered kit could be added with
  // nothing proving which of its layers landed.
  assert!(
    ci_contents.contains("sort | uniq -d")
      && ci_contents.contains("but this shard declares no overlay-pins"),
    "ci.yml's overlay step no longer derives \"this kit stages layers\" from MODELS_LOCK's \
     duplicate `local-dir`s, so a layered kit could be added with no proof of which layer won"
  );

  // The pins themselves, which must be the same bytes `tests/speaker/model_io.rs`
  // pins. A re-baseline there that forgets ci.yml (or the reverse) leaves CI
  // asserting bytes no gate accepts.
  let row = matrix_rows(&ci_contents)
    .into_iter()
    .find(|r| row_field(r, "kit") == "speaker")
    .expect("ci.yml has no `speaker` shard");
  let pins = require_field(&row, "overlay-pins");
  let model_io = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/speaker/model_io.rs");
  let model_io =
    fs::read_to_string(&model_io).unwrap_or_else(|e| panic!("read {}: {e}", model_io.display()));
  // A THIRD copy of the same two hashes lives in `tests/speaker/common/mod.rs`'s
  // `skipped_for_stale_overlay` — the parity gates' own hash-vs-LOCK check, independent of both
  // ci.yml's overlay step and model_io.rs's byte-pin tests. Held to the same pin below so a
  // re-baseline of one copy cannot silently leave another checking retired bytes.
  let common_mod = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/speaker/common/mod.rs");
  let common_mod = fs::read_to_string(&common_mod)
    .unwrap_or_else(|e| panic!("read {}: {e}", common_mod.display()));
  for (bundle, gate) in [
    (
      "pyannote_segmentation.mlmodelc",
      "fp16_safe_segmentation_matches_pinned_sha256",
    ),
    (
      "wespeaker.mlmodelc",
      "fp16_safe_wespeaker_fp32_matches_pinned_sha256",
    ),
  ] {
    let prefix = format!("{bundle}:");
    let hash = pins
      .split_whitespace()
      .find_map(|pin| pin.strip_prefix(&prefix))
      .unwrap_or_else(|| {
        panic!(
          "the speaker shard's overlay-pins carry no `{bundle}:<sha256>` entry, so a wrong \
           download order would only surface later as anonymous hash drift"
        )
      });
    assert!(
      hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()),
      "the speaker shard's `{bundle}` overlay pin is not a 64-hex-digit sha256: {hash:?}"
    );
    assert!(
      model_io.contains(gate),
      "tests/speaker/model_io.rs no longer defines {gate:?}, the gate ci.yml's overlay pin for \
       {bundle:?} is a second copy of"
    );
    assert!(
      model_io.contains(hash),
      "ci.yml pins {bundle}/model.mil at sha256 {hash}, which appears nowhere in \
       tests/speaker/model_io.rs — the overlay check and the {gate} gate have drifted apart, so \
       CI would demand bytes that gate rejects (or accept bytes it would reject). Re-baseline \
       both together."
    );
    assert!(
      common_mod.contains(hash),
      "ci.yml pins {bundle}/model.mil at sha256 {hash}, which appears nowhere in \
       tests/speaker/common/mod.rs — `skipped_for_stale_overlay`'s OVERLAY_MODEL_MIL_PINS has \
       drifted from the overlay check and the {gate} gate, so a stale-overlay skip could fire (or \
       fail to fire) on bytes the other two disagree with. Re-baseline all three together."
    );
  }
}

/// The fp16 graph sweep runs once per shard over a NARROWED vendor manifest, so
/// its real coverage is the UNION across shards — and a union with a hole in it
/// is exactly the silent loss sharding is supposed to prevent.
///
/// `COREMLIT_FP16_SWEEP_VENDORS` is fail-closed per shard (a named vendor
/// directory that is missing fails the sweep) but says nothing about vendors no
/// shard names at all. Unset, the sweep demands every `KNOWN_DEFECTS` vendor;
/// narrowed seven ways, it demands whatever the seven rows happen to list. This
/// pins both directions of that:
///
/// - every pinned-defect vendor is swept by SOME shard, or is declared in
///   [`UNSTAGED_DEFECT_VENDORS`] with a reason;
/// - every vendor a shard STAGES is swept by that same shard, or is declared
///   graph-less — so a kit cannot download `.mlmodelc` bundles the sweep never
///   reads.
#[test]
fn ci_fp16_sweep_shards_cover_every_pinned_vendor() {
  let Some((lock_path, workflow_path)) = repo_files() else {
    return;
  };
  let lock_contents = fs::read_to_string(lock_path).expect("MODELS_LOCK reads");
  let tables = parse_lock(&lock_contents);
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");
  let rows = matrix_rows(&ci_contents);

  assert!(
    ci_contents.contains("COREMLIT_FP16_SWEEP_VENDORS: ${{ matrix.fp16-vendors }}"),
    "ci.yml's fp16 sweep step no longer takes its vendor manifest from the matrix row, so the \
     per-shard coverage this test reasons about is not what CI runs"
  );

  let mut swept: BTreeSet<&str> = BTreeSet::new();
  for row in &rows {
    let kit = row_field(row, "kit");
    let manifest: BTreeSet<&str> = require_field(row, "fp16-vendors").split(',').collect();
    assert!(
      manifest.contains("vadkit"),
      "ci.yml's {kit:?} shard does not name `vadkit` in its fp16 sweep manifest. The VAD graph is \
       COMMITTED, so it costs no download, and naming it in every shard is what makes deleting \
       the vendored model a hard failure rather than a silent drop of the sweep's clean control."
    );
    let staged: BTreeSet<&str> = tables
      .iter()
      .filter(|t| field(t, "kit") == Some(kit))
      .map(|t| vendor_of(field(t, "local-dir").expect("checked by lock_parses...")))
      .collect();
    for vendor in &staged {
      assert!(
        manifest.contains(vendor) || GRAPHLESS_VENDORS.contains(vendor),
        "ci.yml's {kit:?} shard stages Models/{vendor}/ but does not sweep it \
         (fp16-vendors = {:?}). Either add it, or — if that tree holds no `.mlmodelc` for the \
         sweep to audit — add it to GRAPHLESS_VENDORS here with that reason.",
        require_field(row, "fp16-vendors")
      );
    }
    for vendor in &manifest {
      assert!(
        *vendor == "vadkit" || staged.contains(vendor),
        "ci.yml's {kit:?} shard sweeps Models/{vendor}/, which that kit's MODELS_LOCK tables \
         never stage ({staged:?}). The sweep's manifest is fail-closed on a missing directory, so \
         this shard would fail every run."
      );
    }
    swept.extend(manifest);
  }

  let unstaged: BTreeMap<&str, &str> = UNSTAGED_DEFECT_VENDORS.iter().copied().collect();
  let pinned = fp16_pinned_vendors();
  for vendor in &pinned {
    let vendor = vendor.as_str();
    if swept.contains(vendor) {
      continue;
    }
    assert!(
      unstaged.contains_key(vendor),
      "tests/fp16_guards.rs pins fp16 guard findings under Models/{vendor}/ (KNOWN_DEFECTS or \
       LOAD_BEARING_NORMS), but NO model-tests shard sweeps that vendor — those pins are verified \
       nowhere in CI. Stage it in a shard, or record it in UNSTAGED_DEFECT_VENDORS here with the \
       reason it cannot be."
    );
  }
  // ...and the escape list must not outlive its reason: a vendor recorded as
  // unstaged that a shard now DOES sweep, or that no longer carries a pin at
  // all, is a stale exemption hiding what is actually covered.
  for (vendor, _) in UNSTAGED_DEFECT_VENDORS {
    assert!(
      !swept.contains(vendor),
      "UNSTAGED_DEFECT_VENDORS records Models/{vendor}/ as swept by no shard, but a shard now \
       sweeps it — drop the exemption"
    );
    assert!(
      pinned.contains(*vendor),
      "UNSTAGED_DEFECT_VENDORS records Models/{vendor}/, which carries no fp16_guards pin in \
       tests/fp16_guards.rs any more — drop the entry"
    );
  }
}

/// GitHub's default step condition is `success()`, so one red step marks every
/// step after it `skipped` — and `skipped` is silent. `model-tests` ran that
/// way for four weeks: a stale assertion in the whisper suite went red the day
/// it merged, and the four gate steps below it (Whisper+VAD, granite, SigLIP,
/// CED) never executed on CI at all — each was added after the step above it
/// was already permanently red.
///
/// Sharding by kit removes the shared fate BETWEEN families; within a shard the
/// checks are still independent — a bad checksum says nothing about whether the
/// fp16 sweep passes — so each carries [`GATE_GUARD`] and reports its own
/// verdict. A failed DOWNLOAD is the one genuine dependency and still
/// short-circuits them all.
///
/// ci.yml's own `Gate ledger` step catches a check that did not run, but only on
/// a run where something else already failed; in a green run a step added
/// without the guard is invisible until the day it matters. This catches that
/// at authoring time, in the modelless `features` job.
#[test]
fn ci_model_tests_gates_cannot_be_silently_skipped() {
  let Some((_, workflow_path)) = repo_files() else {
    return;
  };
  let ci_contents = fs::read_to_string(workflow_path).expect(".github/workflows/ci.yml reads");
  let steps = model_tests_steps(&ci_contents);

  let download = steps
    .iter()
    .position(|step| step.contains("id: download"))
    .expect("ci.yml's model-tests job has no step with `id: download`");
  let (gates, ledger) = steps[download + 1..]
    .split_last()
    .map(|(last, rest)| (rest, last))
    .expect("ci.yml's model-tests job has no steps after the download");

  assert!(
    ledger.contains("name: Gate ledger") && ledger.contains("if: ${{ !cancelled() }}"),
    "ci.yml's model-tests job must END with the `Gate ledger` step, guarded by `!cancelled()` \
     alone so it reports even when the download died and took every check with it; its last step \
     is instead:\n{ledger}"
  );

  // Vacuum guard: a parse that produced no steps would pass every loop below.
  // Every shard runs this exact list — that uniformity is what lets one ledger
  // and one set of pins cover all seven.
  for gate in [
    "name: Verify staged overlay ordering",
    "name: Verify staged artifact checksums",
    "name: fp16 graph sweep",
    "name: fp16 sweep inventory",
    "name: Model gates",
  ] {
    assert!(
      gates.iter().any(|step| step.contains(gate)),
      "ci.yml's model-tests job has no `{gate}` step after the download — either it was removed, \
       or this test stopped parsing the job (it found {} step(s))",
      gates.len()
    );
  }

  for step in gates {
    assert!(
      step.contains(GATE_GUARD),
      "this model-tests step does not carry `{GATE_GUARD}`, so a failure in any step before it \
       marks it `skipped` and the shard reports nothing about the check that never ran:\n{step}"
    );
    let id = step
      .lines()
      .find_map(|line| line.trim().strip_prefix("id: "))
      .unwrap_or_else(|| {
        panic!(
          "this model-tests step has no `id:`, so the `Gate ledger` step cannot \
           report whether it ran:\n{step}"
        )
      });
    let entry = format!("=${{{{ steps.{id}.outcome }}}}");
    assert!(
      ledger.contains(&entry),
      "the `Gate ledger` step never reads step {id:?} ({entry:?}), so that check could be skipped \
       without the shard saying so"
    );
  }
}
