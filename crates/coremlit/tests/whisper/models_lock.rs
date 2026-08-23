//! `MODELS_LOCK` governs what CI's `model-tests` job downloads (see the
//! lock file's own header comment and `.github/workflows/ci.yml`'s
//! "Download models (cache miss)" step). These checks are hermetic — no
//! network, no models — and guard the three ways that contract can silently
//! rot: the lock stops parsing, the workflow stops actually reading it, or
//! the workflow keeps downloading the artifacts but stops RUNNING the gates
//! that read them.
//!
//! No TOML crate: this is a deliberately tiny hand-rolled reader over the
//! lock's fixed seven-table shape (`["repo/name"]` headers, single-line
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

/// The step condition every gate in ci.yml's `model-tests` job must carry.
///
/// `!cancelled()` is what makes a gate independent of the ones before it;
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
  let job = ci
    .split_once("\n  model-tests:\n")
    .expect("ci.yml has no `model-tests` job")
    .1;
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
    7,
    "MODELS_LOCK: expected exactly seven tables, found {}",
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
      "FinDIT-Studio/embedkit-coreml",
      "FinDIT-Studio/siglip2-naflex-coreml",
      "FinDIT-Studio/cedkit-coreml",
      "FluidInference/speaker-diarization-coreml",
      "FinDIT-Studio/speakerkit-coreml"
    ],
    "MODELS_LOCK's table names or their order changed — update this test alongside it. The last \
     two are not interchangeable: both download into Models/speakerkit/ and both publish \
     `pyannote_segmentation.mlmodelc` and `wespeaker.mlmodelc`, so the LAST one wins those two \
     filenames and it must be the fp16-guard-repaired overlay. See \
     `ci_stages_the_speakerkit_overlay_last_and_proves_it_won`"
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
    ci_contents.contains("hf download \"$siglip_repo\""),
    "download step doesn't invoke hf with a lock-derived $siglip_repo"
  );
  assert!(
    ci_contents.contains("hf download \"$ced_repo\""),
    "download step doesn't invoke hf with a lock-derived $ced_repo"
  );
  assert!(
    ci_contents.contains("hf download \"$speakerkit_base_repo\""),
    "download step doesn't invoke hf with a lock-derived $speakerkit_base_repo"
  );
  assert!(
    ci_contents.contains("hf download \"$speakerkit_overlay_repo\""),
    "download step doesn't invoke hf with a lock-derived $speakerkit_overlay_repo"
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
  assert!(
    ci_contents.contains("--revision \"$siglip_revision\""),
    "download step doesn't pass a lock-derived --revision for the siglip repo"
  );
  assert!(
    ci_contents.contains("--revision \"$ced_revision\""),
    "download step doesn't pass a lock-derived --revision for the ced repo"
  );
  assert!(
    ci_contents.contains("--revision \"$speakerkit_base_revision\""),
    "download step doesn't pass a lock-derived --revision for the speakerkit base repo"
  );
  assert!(
    ci_contents.contains("--revision \"$speakerkit_overlay_revision\""),
    "download step doesn't pass a lock-derived --revision for the speakerkit overlay repo"
  );

  // ci.yml selects tables by INDEX, so an appended table it does not extract is
  // silently ignored: MODELS_LOCK would document a download CI never performs,
  // and ci.yml's empty-extraction guard — which only inspects variables that
  // WERE extracted — cannot catch it. The workflow therefore pins the table
  // count it can parse; assert that pin exists and agrees with the lock, so the
  // two cannot drift apart in either direction.
  assert!(
    ci_contents.contains(&format!("\"$table_count\" -ne {} ]", tables.len())),
    "ci.yml's download step doesn't pin MODELS_LOCK's table count at {}, so a table appended to \
     the lock would be silently ignored by its index-selected parser",
    tables.len()
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

  // Same claim for CED, and NOT the same string: CED's checksum file lists
  // paths relative to the `.mlmodelc` root rather than to the directory holding
  // it, so its step reads `../CHECKSUMS.sha256` from inside the bundle. A
  // copy-paste of granite's `-c CHECKSUMS.sha256` would report five missing
  // files — this pins the CED-shaped invocation specifically.
  assert!(
    ci_contents.contains("shasum -a 256 -c ../CHECKSUMS.sha256"),
    "ci.yml no longer verifies the CED bundle against its shipped CHECKSUMS.sha256 (read from \
     inside the .mlmodelc, whose root its paths are relative to), which MODELS_LOCK's header \
     claims it does"
  );

  // And a third shape for the speakerkit overlay, which cannot reuse either of
  // the two above: its checksum file lists its OWN name with the sha256 of an
  // empty file, plus the `wespeaker_int8`/`.mlpackage` paths MODELS_LOCK's
  // selector deliberately does not stage, so `shasum -c` over the whole file
  // fails on a correct tree. ci.yml verifies the filtered subset from stdin.
  assert!(
    ci_contents.contains("shasum -a 256 -c \"$filtered\""),
    "ci.yml no longer verifies the staged speakerkit overlay files against the CHECKSUMS.sha256 \
     that artifact ships, which MODELS_LOCK's header claims it does"
  );
}

/// MODELS_LOCK stages ONE CED size (`ced-tiny`: 10.64 MB of the artifact repo's
/// 234 MB), but `tests/ced/*` declares every gate four times — one `#[ignore]`d
/// module per size — and an ignored-ONLY run selects ALL FOUR. So ci.yml's CED
/// gate step must filter to the staged size, and the two must agree.
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
    .find(|t| t.name == "FinDIT-Studio/cedkit-coreml")
    .expect("MODELS_LOCK has no CED table");
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
         `ced-<size>/*` shape so the staged size can be matched against ci.yml's gate filter"
      )
    });
  assert!(
    !size.is_empty() && size.chars().all(|c| c.is_ascii_lowercase()),
    "MODELS_LOCK's CED `include` names {size:?}, which is not a CedModel size name"
  );

  // The download lands the size directory under the family root the tests
  // resolve by default.
  assert!(
    ci_contents.contains("--local-dir Models/ced"),
    "ci.yml doesn't download CED into Models/ced, the family root tests/ced/common/mod.rs \
     resolves without CED_TEST_MODELS"
  );

  let bundle = format!("Models/ced/ced-{size}/ced_{size}.mlmodelc");
  assert!(
    ci_contents.contains(&bundle),
    "MODELS_LOCK stages CED size {size:?}, but ci.yml never names its bundle {bundle:?} — the \
     absent-artifact guard and the checksum verification would be pointed at a different size \
     than the one downloaded"
  );

  // The run filter and the anti-vacuum count must BOTH carry the size. Counting
  // an unfiltered list would let a renamed `tiny` module still report the other
  // sizes' gates and then run zero.
  let run_filter = format!("-- --ignored {size}::");
  assert!(
    ci_contents.contains(&run_filter),
    "ci.yml's CED gate step doesn't run the {size:?} gates ({run_filter:?}); an unfiltered \
     `-- --ignored` selects all four sizes and would fail on the three MODELS_LOCK never staged"
  );
  let list_filter = format!("-- --list --ignored {size}::");
  assert!(
    ci_contents.contains(&list_filter),
    "ci.yml's CED anti-vacuum count doesn't go through the same {size:?} filter as the run \
     ({list_filter:?}), so a deleted or renamed `{size}` module could still count the other \
     sizes' gates and then run none"
  );
}

/// The two speakerkit tables are the one place in MODELS_LOCK where table order
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
/// So this pins three things that must move together: the lock's order, the
/// order of the two `hf download` commands ci.yml derives from it, and the
/// early overlay check that re-hashes both graphs. That last one is what turns
/// a wrong order into a named failure instead of anonymous "sha256 drift"
/// reported a full build later by `speaker_model_io` — and its expected hashes
/// are a deliberate SECOND copy of that gate's pins, so this test also holds
/// the two copies together.
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
  assert!(
    base < overlay,
    "MODELS_LOCK stages the speakerkit OVERLAY (table {}) before the BASE layer (table {}), so \
     the base layer's PRE-REPAIR pyannote_segmentation/wespeaker graphs would overwrite the \
     fp16-guard-repaired ones the pipeline ships",
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

  // ci.yml emits the two downloads in lock order; pin that it still does, since
  // the lock order above is only load-bearing if the workflow follows it.
  let base_cmd = ci_contents
    .find("hf download \"$speakerkit_base_repo\"")
    .expect("ci.yml never downloads the speakerkit base layer");
  let overlay_cmd = ci_contents
    .find("hf download \"$speakerkit_overlay_repo\"")
    .expect("ci.yml never downloads the speakerkit overlay");
  assert!(
    base_cmd < overlay_cmd,
    "ci.yml downloads the speakerkit overlay BEFORE the base layer, so the base layer's \
     pre-repair graphs overwrite the shipping ones no matter what MODELS_LOCK's table order says"
  );

  // The early overlay check, and its two expected hashes — which must be the
  // same bytes `tests/speaker/model_io.rs` pins. A re-baseline there that
  // forgets ci.yml (or the reverse) leaves CI asserting bytes no gate accepts.
  let model_io = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/speaker/model_io.rs");
  let model_io =
    fs::read_to_string(&model_io).unwrap_or_else(|e| panic!("read {}: {e}", model_io.display()));
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
    let needle = format!("\"{bundle}:");
    let rest = ci_contents
      .split_once(&needle)
      .unwrap_or_else(|| {
        panic!(
          "ci.yml's speakerkit overlay check carries no `{bundle}:<sha256>` pin, so a wrong \
           download order would only surface later as anonymous hash drift"
        )
      })
      .1;
    let hash = rest.get(..64).unwrap_or_default();
    assert!(
      hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()),
      "ci.yml's `{bundle}` overlay pin is not a 64-hex-digit sha256: {hash:?}"
    );
    assert!(
      model_io.contains(gate),
      "tests/speaker/model_io.rs no longer defines {gate:?}, the gate ci.yml's overlay pin for \
       {bundle:?} is a second copy of"
    );
    assert!(
      model_io.contains(hash),
      "ci.yml pins {bundle}/model.mil at sha256 {hash}, which appears nowhere in \
       tests/speaker/model_io.rs — the early overlay check and the {gate} gate have drifted \
       apart, so CI would demand bytes that gate rejects (or accept bytes it would reject). \
       Re-baseline both together."
    );
  }
}

/// GitHub's default step condition is `success()`, so one red step marks every
/// step after it `skipped` — and `skipped` is silent. `model-tests` ran that
/// way for four weeks: a stale assertion in the whisper suite went red the day
/// it merged, and the four gate steps below it (Whisper+VAD, granite, SigLIP,
/// CED) have between them never executed on CI at all — each was added after
/// the step above it was already permanently red.
///
/// These gates are independent — a corrupt granite bundle says nothing about
/// whether the CED graph loads — so each carries [`GATE_GUARD`] and reports its
/// own verdict. A failed DOWNLOAD is the one genuine dependency and still
/// short-circuits them all.
///
/// ci.yml's own `Gate ledger` step catches a gate that did not run, but only on
/// a run where something else already failed; in a green run a gate step added
/// without the guard is invisible until the day it matters. This catches that
/// at authoring time, in the modelless `check` job.
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
     alone so it reports even when the download died and took every gate with it; its last step \
     is instead:\n{ledger}"
  );

  // Vacuum guard: a parse that produced no steps would pass every loop below.
  // These five are the gate steps the four-week outage darkened.
  for gate in [
    "name: Whisper model gates",
    "name: Whisper + Silero-VAD composition gates",
    "name: Granite model gates",
    "name: SigLIP tokenizer gates",
    "name: CED model gates (tiny)",
    "name: Speaker model gates",
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
       marks it `skipped` and the job reports nothing about the gate that never ran:\n{step}"
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
      "the `Gate ledger` step never reads step {id:?} ({entry:?}), so that gate could be skipped \
       without the job saying so"
    );
  }
}
