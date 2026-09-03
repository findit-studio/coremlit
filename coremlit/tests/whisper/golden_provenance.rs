//! Provenance of the whisper goldens: the host-class predicate's falsifiers,
//! the append-or-replace tool that decides what a golden may claim, and the
//! structural guards that keep a golden producible only by the external Swift
//! oracle.
//!
//! Hermetic — no model, no network, no CoreML. These run in CI's `features` job
//! on every PR, which is the point: the model-gated parity tests they protect
//! run only where `openai_whisper-tiny` is staged, and the rules below have to
//! hold everywhere.
//!
//! Three jobs, in the order they matter:
//!
//! 1. **The gate works.** `check_host_class` must diagnose a host outside the
//!    golden's recorded set, pass one inside it — in ANY position — and tolerate
//!    an unstamped legacy golden with the ambiguity note. Those are the cases
//!    the parity suites depend on, driven here with synthetic values so no
//!    hardware is involved.
//! 2. **The set grows only by measurement.** A golden records the host classes
//!    its payload was REPRODUCED on. `merge_golden_hosts.sh` is the only thing
//!    that writes that set, and it appends a class only when the fresh payload
//!    is byte-identical to the committed one; when the payload moves it replaces
//!    the set outright, because the old classes reproduced the old numbers and
//!    have said nothing about the new ones. Both branches are driven below.
//! 3. **The gate cannot be dissolved.** A golden re-baselined against
//!    coremlit's own output would make every whisper parity test assert that
//!    coremlit agrees with coremlit. The crate no longer contains any code that
//!    can write one — the `UPDATE_GOLDEN` writer arms that used to sit in
//!    `parity_es.rs` and `parity_jfk.rs` were deleted, not merely discouraged —
//!    and the guards below fail the suite if a path back appears.

mod common;

use common::{
  HostClass, HostVerdict, RecordedHost, WHISPER_REGEN_SCRIPT, check_host_class, legacy_failure_note,
};

/// Every committed golden the whisper suites read.
const GOLDENS: [&str; 3] = [
  "es_tiny_golden.json",
  "jfk_tiny_golden.json",
  "jfk_tiny_words_golden.json",
];

/// The regeneration script, and the tool that decides what its output may claim.
const REGEN_SCRIPT: &str = "coremlit/tests/whisper/swift/regen_goldens.sh";
const MERGE_TOOL: &str = "coremlit/tests/whisper/swift/merge_golden_hosts.sh";

/// The repository root, for the non-Rust artifacts guarded below.
fn repo_root() -> std::path::PathBuf {
  common::workspace_root()
}

fn read_repo_file(rel: &str) -> String {
  let path = repo_root().join(rel);
  std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

/// A synthetic host-class for the pure-predicate tests — no sysctl, so these
/// assert the same thing on every machine.
fn synthetic_host() -> HostClass {
  HostClass {
    os_build: "24F74".to_string(),
    os_product_version: "15.5".to_string(),
    chip: "Apple M1".to_string(),
    arch: "arm64".to_string(),
  }
}

/// The two host classes GitHub's `macos-15` pool served simultaneously through
/// the rollover that motivated the set — same virtual chip, different OS build —
/// with the `whisperkit-cli` build each image shipped. The committed goldens
/// record exactly this pair; these copies are synthetic so the tests below stay
/// hermetic and keep asserting the rule if the fixtures are regenerated.
fn rolling_pool() -> [RecordedHost; 2] {
  let class = |build: &str, version: &str| HostClass {
    os_build: build.to_string(),
    os_product_version: version.to_string(),
    chip: "Apple M1 (Virtual)".to_string(),
    arch: "arm64".to_string(),
  };
  [
    RecordedHost {
      class: class("24G720", "15.7.7"),
      source: Some("whisperkit-cli @ argmax-oss-swift (v1.0.0)".to_string()),
    },
    RecordedHost {
      class: class("24G830", "15.7.9"),
      source: Some("whisperkit-cli @ argmax-oss-swift (v1.1.0)".to_string()),
    },
  ]
}

/// One `generationHosts` entry as it appears in a golden.
fn host_json(build: &str, version: &str, source: &str) -> serde_json::Value {
  serde_json::json!({
    "osBuild": build,
    "osProductVersion": version,
    "chip": "Apple M1 (Virtual)",
    "arch": "arm64",
    "source": source,
  })
}

// ── 1. The gate works ───────────────────────────────────────────────────────

/// FALSIFIER: a golden whose recorded set does NOT contain the running host must
/// be diagnosed, never compared.
///
/// This is the case CI hits after GitHub rotates the macos-15 image to a build
/// nobody has regenerated on. It has to produce an unmistakable instruction,
/// because the alternative — what the three whisper gates did from late July
/// until this test existed — is a raw token divergence that reads like a port
/// defect and gets ignored as flake.
#[test]
fn foreign_host_class_is_diagnosed_with_the_regeneration_command() {
  let golden_host = synthetic_host();
  // One pair differs only in osBuild (the monthly runner-image rotation); the
  // other only in chip (a different Apple Silicon generation).
  let rotated_image = HostClass {
    os_build: "24G84".to_string(),
    ..golden_host.clone()
  };
  let other_silicon = HostClass {
    chip: "Apple M4 Pro".to_string(),
    ..golden_host.clone()
  };
  let recorded = [RecordedHost::from(golden_host.clone())];

  for running in [&rotated_image, &other_silicon] {
    let diagnosis = check_host_class(
      "es_tiny_golden.json",
      &recorded,
      running,
      WHISPER_REGEN_SCRIPT,
    )
    .expect_err("a host outside the recorded set must be diagnosed, not silently compared");

    assert!(
      diagnosis.contains("DIFFERENT host-class"),
      "diagnosis must name the cause: {diagnosis}"
    );
    assert!(
      diagnosis.contains(golden_host.to_string().as_str()),
      "diagnosis must name the golden's host: {diagnosis}"
    );
    assert!(
      diagnosis.contains(running.to_string().as_str()),
      "diagnosis must name the running host: {diagnosis}"
    );
    assert!(
      diagnosis.contains("NOT evidence of a port defect"),
      "diagnosis must not let the port take the blame: {diagnosis}"
    );
    assert!(
      diagnosis.contains("regen_goldens.sh"),
      "diagnosis must name the exact regeneration command: {diagnosis}"
    );
    assert!(
      diagnosis.contains("regen-whisper-goldens.yml"),
      "diagnosis must offer the runner-host-class route too: {diagnosis}"
    );
    assert!(
      diagnosis.contains("Do NOT widen the parity tolerances"),
      "diagnosis must refuse the tempting fix: {diagnosis}"
    );
  }
}

/// FALSIFIER: an IDENTICAL host-class compares strictly — the gate must not
/// have become a blanket excuse. On a matching host a divergence is the port's,
/// and `Match` is what says so (it yields an empty note, so nothing softens the
/// failure).
#[test]
fn identical_host_class_compares_strictly() {
  let host = synthetic_host();
  assert_eq!(
    check_host_class(
      "jfk_tiny_golden.json",
      &[RecordedHost::from(host.clone())],
      &host,
      WHISPER_REGEN_SCRIPT
    ),
    Ok(HostVerdict::Match)
  );
}

/// FALSIFIER for the rolling-pool rule: a golden recording TWO host classes
/// matches on EITHER of them, in either position — and matches through the JSON
/// the gates actually read, not only through a hand-built `Vec`.
///
/// This is the red the single-host compare could not turn green. GitHub's
/// `macos-15` pool served build 24G720 (macOS 15.7.7) and 24G830 (15.7.9) at the
/// same time; the whisper goldens' tokens, segments and word timestamps came out
/// byte-identical on both, yet a golden naming one build refused every job that
/// landed on the other — a host difference reported as a port failure, which is
/// the exact misattribution this file exists to prevent.
///
/// Both positions are asserted deliberately. "Compare against the first recorded
/// class only" is the mutation this test exists to catch, and every other test
/// in this file matches at index 0 — so this is the only one that would notice.
#[test]
fn any_host_class_in_the_recorded_set_matches() {
  let pool = rolling_pool();
  for expected in &pool {
    assert_eq!(
      check_host_class(
        "es_tiny_golden.json",
        &pool,
        &expected.class,
        WHISPER_REGEN_SCRIPT
      ),
      Ok(HostVerdict::Match),
      "a golden reproduced on {} must match a machine of that class, whatever its position \
       in the recorded set",
      expected.class
    );
  }

  // The same set, read out of a golden — the path `common::golden_host_note`
  // takes. A predicate that matches while the parser drops the key is no gate.
  let golden = serde_json::json!({
    "source": "whisperkit-cli @ argmax-oss-swift (v1.0.0)",
    "generationHosts": [
      host_json("24G720", "15.7.7", "whisperkit-cli @ argmax-oss-swift (v1.0.0)"),
      host_json("24G830", "15.7.9", "whisperkit-cli @ argmax-oss-swift (v1.1.0)"),
    ],
    "tokens": [50258, 50259],
  });
  let parsed = RecordedHost::all_from_golden("es_tiny_golden.json", &golden)
    .expect("a two-host set must parse");
  assert_eq!(
    parsed,
    pool.to_vec(),
    "the parser must read every recorded host, with the oracle observed on each"
  );
  for expected in &pool {
    assert_eq!(
      check_host_class(
        "es_tiny_golden.json",
        &parsed,
        &expected.class,
        WHISPER_REGEN_SCRIPT
      ),
      Ok(HostVerdict::Match)
    );
  }
}

/// FALSIFIER: widening the claim to a SET must not widen it to "any host". A
/// machine outside the set is still refused, and the refusal names every class
/// the golden does record, so the reader can see at a glance whether their
/// machine is one regeneration away or a different chip entirely.
#[test]
fn a_host_outside_the_recorded_set_is_refused_naming_every_class() {
  let pool = rolling_pool();
  let outsider = HostClass {
    os_build: "25F74".to_string(),
    os_product_version: "26.5".to_string(),
    chip: "Apple M1 Max".to_string(),
    arch: "arm64".to_string(),
  };

  let diagnosis = check_host_class(
    "jfk_tiny_words_golden.json",
    &pool,
    &outsider,
    WHISPER_REGEN_SCRIPT,
  )
  .expect_err("a machine matching NO recorded class must still be refused");

  assert!(
    diagnosis.contains("NOT been verified on this host-class"),
    "the refusal must say what is actually unknown: {diagnosis}"
  );
  for recorded in &pool {
    assert!(
      diagnosis.contains(recorded.class.to_string().as_str()),
      "the refusal must list every recorded class, missing {}: {diagnosis}",
      recorded.class
    );
  }
  assert!(
    diagnosis.contains(outsider.to_string().as_str()),
    "the refusal must name the running host: {diagnosis}"
  );
  assert!(
    diagnosis.contains("NOT evidence of a port defect"),
    "the refusal must not let the port take the blame: {diagnosis}"
  );
  assert!(
    diagnosis.contains("regen_goldens.sh"),
    "the refusal must name the fix: {diagnosis}"
  );
}

/// FALSIFIER: an UNSTAMPED (legacy) golden is tolerated, with the ambiguity
/// note — it does not become an error, and it does not become a free pass.
#[test]
fn unstamped_legacy_golden_is_tolerated_with_the_ambiguity_note() {
  let running = synthetic_host();
  assert_eq!(
    check_host_class("jfk_tiny_golden.json", &[], &running, WHISPER_REGEN_SCRIPT),
    Ok(HostVerdict::LegacyUnknown)
  );

  let note = legacy_failure_note(WHISPER_REGEN_SCRIPT);
  assert!(
    note.contains("AMBIGUOUS"),
    "the note's job is to refuse a confident attribution: {note}"
  );
  assert!(
    note.contains("generationHost"),
    "the note must name the missing field: {note}"
  );
  assert!(
    note.contains("regen_goldens.sh"),
    "the note must name the fix: {note}"
  );
  assert!(
    note.contains("Do NOT widen the tolerance"),
    "the note must refuse the tempting fix: {note}"
  );
}

/// `RecordedHost::all_from_golden` is strict about a host record that IS present
/// and tolerant of one that is not — the asymmetry the legacy path rests on —
/// and it reads the legacy single `generationHost` as a one-element set, so a
/// golden stamped before the set existed keeps gating exactly as it did.
#[test]
fn generation_host_parse_is_strict_and_legacy_tolerant() {
  assert_eq!(
    RecordedHost::all_from_golden("wf", &serde_json::json!({})).expect("absent must be Ok"),
    Vec::new(),
    "an unstamped golden must keep parsing on every host"
  );

  // The legacy single-host stamp, read as a set of one. Its oracle label lives
  // at the document level in that schema, so the entry carries none.
  let legacy = serde_json::json!({
    "generationHost": {
      "osBuild": "24G84",
      "osProductVersion": "15.6",
      "chip": "Apple M2",
      "arch": "arm64"
    }
  });
  assert_eq!(
    RecordedHost::all_from_golden("wf", &legacy).expect("well-formed legacy must be Ok"),
    vec![RecordedHost::from(HostClass {
      os_build: "24G84".to_string(),
      os_product_version: "15.6".to_string(),
      chip: "Apple M2".to_string(),
      arch: "arm64".to_string(),
    })]
  );

  // The set, with the oracle observed on each host.
  let set = serde_json::json!({
    "generationHosts": [
      host_json("24G720", "15.7.7", "whisperkit-cli @ argmax-oss-swift (v1.0.0)"),
      host_json("24G830", "15.7.9", "whisperkit-cli @ argmax-oss-swift (v1.1.0)"),
    ]
  });
  assert_eq!(
    RecordedHost::all_from_golden("wf", &set).expect("well-formed set must be Ok"),
    rolling_pool().to_vec()
  );

  // A half-written stamp is an error, not a silent legacy fallback: silently
  // downgrading it would turn a typo into a permanently un-attributable gate.
  let missing_arch = serde_json::json!({
    "generationHost": { "osBuild": "24G84", "osProductVersion": "15.6", "chip": "Apple M2" }
  });
  let err = RecordedHost::all_from_golden("wf", &missing_arch).unwrap_err();
  assert!(err.contains("generationHost.arch"), "{err}");

  let not_an_object = serde_json::json!({ "generationHost": "24G84" });
  let err = RecordedHost::all_from_golden("wf", &not_an_object).unwrap_err();
  assert!(err.contains("generationHost"), "{err}");

  // The same strictness inside the set, and the offending index is named.
  let mut bad_entry = set.clone();
  bad_entry["generationHosts"][1]
    .as_object_mut()
    .expect("entry is an object")
    .remove("chip");
  let err = RecordedHost::all_from_golden("wf", &bad_entry).unwrap_err();
  assert!(err.contains("generationHosts[1].chip"), "{err}");

  let bad_source = serde_json::json!({
    "generationHosts": [{
      "osBuild": "24G720", "osProductVersion": "15.7.7",
      "chip": "Apple M1 (Virtual)", "arch": "arm64", "source": 1
    }]
  });
  let err = RecordedHost::all_from_golden("wf", &bad_source).unwrap_err();
  assert!(err.contains("generationHosts[0].source"), "{err}");

  // An empty set claims a payload nothing reproduced — that is a malformation,
  // not a legacy golden. Legacy is the ABSENCE of the key.
  let empty = serde_json::json!({ "generationHosts": [] });
  let err = RecordedHost::all_from_golden("wf", &empty).unwrap_err();
  assert!(err.contains("generationHosts"), "{err}");

  let not_an_array = serde_json::json!({ "generationHosts": { "osBuild": "24G720" } });
  let err = RecordedHost::all_from_golden("wf", &not_an_array).unwrap_err();
  assert!(err.contains("generationHosts"), "{err}");

  // Both keys at once: two records of one fact, free to disagree. Refused,
  // rather than picking a winner silently.
  let mut both = set.clone();
  both["generationHost"] = serde_json::json!({
    "osBuild": "24G84", "osProductVersion": "15.6", "chip": "Apple M2", "arch": "arm64"
  });
  let err = RecordedHost::all_from_golden("wf", &both).unwrap_err();
  assert!(err.contains("BOTH"), "{err}");
}

/// Every committed golden records the host classes its payload was reproduced
/// on, each with the oracle observed there, and the document-level `source`
/// agrees with the first of them.
///
/// The unstamped state is no longer tolerated HERE — the committed goldens are
/// stamped, and `merge_golden_hosts.sh` stamps every regeneration — while
/// `check_host_class` keeps tolerating it for the speaker and vad goldens that
/// still predate provenance.
#[test]
fn every_committed_golden_records_the_host_classes_it_was_reproduced_on() {
  for golden in GOLDENS {
    let value = common::load_golden_json(golden);
    let hosts = RecordedHost::all_from_golden(golden, &value)
      .unwrap_or_else(|e| panic!("{golden}: the recorded host set must parse: {e}"));

    assert!(
      !hosts.is_empty(),
      "{golden}: records no host class. A committed whisper golden states where its numbers \
       were reproduced; regenerate it via {WHISPER_REGEN_SCRIPT}"
    );

    let classes: Vec<&HostClass> = hosts.iter().map(|h| &h.class).collect();
    for (i, class) in classes.iter().enumerate() {
      assert!(
        !classes[..i].contains(class),
        "{golden}: records {class} twice — the set is a set, and a duplicate means the \
         append path stopped checking"
      );
    }

    for host in &hosts {
      let source = host.source.as_deref().unwrap_or_else(|| {
        panic!(
          "{golden}: recorded host {} carries no `source` — the set records WHICH oracle \
           reproduced the payload on each image, and that is the field that differs \
           between two images whose numbers agree",
          host.class
        )
      });
      assert!(
        source.contains("whisperkit-cli"),
        "{golden}: recorded host {} names oracle {source:?}, not the external Swift CLI",
        host.class
      );
    }

    let document_source = value
      .get("source")
      .and_then(serde_json::Value::as_str)
      .unwrap_or_else(|| panic!("{golden}: missing a `source` field"));
    assert_eq!(
      Some(document_source),
      hosts[0].source.as_deref(),
      "{golden}: the document-level `source` must be the oracle of the FIRST recorded host — \
       the run whose payload the later hosts reproduced. A drift between the two means one \
       of them is stale."
    );

    println!(
      "[host] {golden}: reproduced on {} host class(es): {}",
      hosts.len(),
      hosts
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" | ")
    );
  }
}

/// The running host reads back as four non-empty fields. The one test here that
/// touches the machine, kept because a silently-empty sysctl read would make
/// every stamped golden mismatch for a reason nobody would look for.
#[test]
fn running_host_class_is_well_formed() {
  let h = HostClass::running();
  for (field, value) in [
    ("os_build", &h.os_build),
    ("os_product_version", &h.os_product_version),
    ("chip", &h.chip),
    ("arch", &h.arch),
  ] {
    assert!(
      !value.is_empty(),
      "running host-class field `{field}` is empty"
    );
  }
  assert!(
    h.arch == "arm64" || h.arch == "x86_64",
    "unexpected arch spelling `{}` — the dumpers record arm64/x86_64",
    h.arch
  );
  println!("[host] running: {h}");
}

// ── 2. The set grows only by measurement ────────────────────────────────────
//
// `merge_golden_hosts.sh` is what stands between "this payload was reproduced
// on these hosts" and a set that grows by assumption. It is driven here as a
// black box — two synthetic goldens in, one merged golden out — because that is
// exactly how `regen_goldens.sh` and the CI job use it.
//
// These are the only tests in this file that write anything, and the reason
// this file is the one exempted from `no_whisper_test_can_write_a_golden`
// below. What they write are SYNTHETIC INPUTS under the test target's own
// `CARGO_TARGET_TMPDIR`, asserted in `scratch_dir`; the tool itself opens no
// file for writing at all, so no committed golden is reachable from here.
//
// They need `jq`, which the tool needs, which `regen_goldens.sh` already needs
// — and which the macos runner image ships. A missing `jq` fails them loudly
// rather than skipping, because a silently-skipped decision test is how the
// decision stops being tested.

/// A private scratch directory for one merge case's synthetic inputs, asserted
/// to live under this test target's `CARGO_TARGET_TMPDIR` and nowhere near
/// `fixtures/golden/`.
fn scratch_dir(case: &str) -> std::path::PathBuf {
  let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
  let dir = root.join("merge_golden_hosts").join(case);
  assert!(
    dir.starts_with(&root),
    "merge-tool scratch {dir:?} escaped {root:?}"
  );
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {dir:?}: {e}"));
  dir
}

/// Runs `merge_golden_hosts.sh` over two synthetic goldens; returns the JSON it
/// wrote to stdout and everything it said on stderr (where its decision goes).
fn run_merge(
  case: &str,
  committed: &serde_json::Value,
  fresh: &serde_json::Value,
) -> (serde_json::Value, String) {
  let dir = scratch_dir(case);
  let committed_path = dir.join("committed.json");
  let fresh_path = dir.join("fresh.json");
  for (path, value) in [(&committed_path, committed), (&fresh_path, fresh)] {
    let text = serde_json::to_string_pretty(value).expect("synthetic golden serializes");
    std::fs::write(path, text).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
  }

  let tool = repo_root().join(MERGE_TOOL);
  let output = std::process::Command::new(&tool)
    .arg(&committed_path)
    .arg(&fresh_path)
    .output()
    .unwrap_or_else(|e| panic!("cannot run {tool:?}: {e} — is it executable?"));
  let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
  assert!(
    output.status.success(),
    "{MERGE_TOOL} exited {} for case `{case}`:\n{stderr}\n(it needs jq: brew install jq)",
    output.status
  );
  let merged = serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
    panic!("{MERGE_TOOL} wrote invalid JSON for case `{case}`: {e}\nstderr:\n{stderr}")
  });
  (merged, stderr)
}

/// A golden's payload: everything the parity gates actually compare, i.e. the
/// document minus the provenance keys the merge decision is allowed to move.
fn payload_of(golden: &serde_json::Value) -> serde_json::Value {
  let mut stripped = golden.clone();
  let map = stripped.as_object_mut().expect("a golden is an object");
  for key in ["generationHost", "generationHosts", "source"] {
    map.remove(key);
  }
  stripped
}

/// A synthetic golden in the current schema: one payload, one recorded host.
fn golden_with(hosts: serde_json::Value, source: &str, tokens: [u32; 3]) -> serde_json::Value {
  serde_json::json!({
    "model": "openai_whisper-tiny",
    "source": source,
    "generationHosts": hosts,
    "text": "And so my fellow Americans",
    "language": "en",
    "tokens": tokens,
  })
}

const V1_0: &str = "whisperkit-cli @ argmax-oss-swift (v1.0.0)";
const V1_1: &str = "whisperkit-cli @ argmax-oss-swift (v1.1.0)";

/// APPEND: a byte-identical payload on a new host class earns that class a place
/// in the set, and changes nothing else.
///
/// This is the rollover case the whole schema exists for. Note what the tool
/// must NOT do: the fresh run's `source` differs (v1.0.0 → v1.1.0, because the
/// newer image ships a newer CLI), and if that counted as a payload change the
/// set would be replaced on every image bump and the goldens would be
/// single-host again by another route.
#[test]
fn merge_tool_appends_a_host_whose_payload_is_byte_identical() {
  let committed = golden_with(
    serde_json::json!([host_json("24G720", "15.7.7", V1_0)]),
    V1_0,
    [400, 370, 452],
  );
  let fresh = golden_with(
    serde_json::json!([host_json("24G830", "15.7.9", V1_1)]),
    V1_1,
    [400, 370, 452],
  );

  let (merged, stderr) = run_merge("append", &committed, &fresh);
  let hosts = RecordedHost::all_from_golden("merged", &merged).expect("merged set parses");

  assert_eq!(hosts.len(), 2, "the set must grow by exactly one: {stderr}");
  assert_eq!(
    hosts[0].class.os_build, "24G720",
    "committed host stays first"
  );
  assert_eq!(hosts[1].class.os_build, "24G830", "new host is appended");
  assert_eq!(hosts[0].source.as_deref(), Some(V1_0));
  assert_eq!(
    hosts[1].source.as_deref(),
    Some(V1_1),
    "each host records the oracle observed THERE, which is the field that differs"
  );
  assert_eq!(
    payload_of(&merged),
    payload_of(&committed),
    "an append must not touch a single compared value"
  );
  assert_eq!(
    merged["source"], committed["source"],
    "the document-level source stays the first host's — the run the others reproduced"
  );
  assert!(
    stderr.contains("APPEND"),
    "the decision must be visible in the log: {stderr}"
  );
}

/// REPLACE: a payload that MOVED drops every previously recorded class.
///
/// The old classes reproduced the old numbers; about these numbers they have
/// said nothing, and carrying them over would be a claim nobody measured — a
/// golden asserting parity on hosts that never produced it. This is also the
/// falsifier for deleting the payload comparison from the tool: without it every
/// run appends, and a changed oracle output would silently inherit the old set's
/// endorsement.
#[test]
fn merge_tool_replaces_the_whole_set_when_the_payload_moved() {
  let committed = golden_with(
    serde_json::json!([
      host_json("24G720", "15.7.7", V1_0),
      host_json("24G830", "15.7.9", V1_1),
    ]),
    V1_0,
    [400, 370, 452],
  );
  let fresh = golden_with(
    serde_json::json!([host_json("24H100", "15.8.0", V1_1)]),
    V1_1,
    [400, 370, 999],
  );

  let (merged, stderr) = run_merge("replace", &committed, &fresh);
  let hosts = RecordedHost::all_from_golden("merged", &merged).expect("merged set parses");

  assert_eq!(
    hosts.len(),
    1,
    "a changed payload must leave EXACTLY the host that produced it: {stderr}"
  );
  assert_eq!(hosts[0].class.os_build, "24H100");
  assert_eq!(hosts[0].source.as_deref(), Some(V1_1));
  assert_eq!(
    payload_of(&merged),
    payload_of(&fresh),
    "a replace takes the fresh payload whole"
  );
  assert_eq!(merged["source"], fresh["source"]);
  assert!(
    stderr.contains("REPLACE"),
    "a replaced set must be announced loudly, not slipped into a diff: {stderr}"
  );
  assert!(
    stderr.contains("READ THE DIFF"),
    "changed oracle output is news and must say so: {stderr}"
  );
}

/// IDEMPOTENT: re-running on a class the golden already records adds nothing.
///
/// Without this the set would grow by one duplicate every time the regeneration
/// workflow is dispatched onto the same image, and "the classes this payload was
/// reproduced on" would slowly become "the number of times someone pressed the
/// button".
#[test]
fn merge_tool_does_not_record_a_host_class_twice() {
  let committed = golden_with(
    serde_json::json!([
      host_json("24G720", "15.7.7", V1_0),
      host_json("24G830", "15.7.9", V1_1),
    ]),
    V1_0,
    [400, 370, 452],
  );
  // The same class as the committed second entry, re-run under a newer CLI.
  let fresh = golden_with(
    serde_json::json!([host_json(
      "24G830",
      "15.7.9",
      "whisperkit-cli @ argmax-oss-swift (v1.2.0)"
    )]),
    "whisperkit-cli @ argmax-oss-swift (v1.2.0)",
    [400, 370, 452],
  );

  let (merged, stderr) = run_merge("idempotent", &committed, &fresh);
  let hosts = RecordedHost::all_from_golden("merged", &merged).expect("merged set parses");

  assert_eq!(
    hosts.len(),
    2,
    "an already-recorded class must not repeat: {stderr}"
  );
  assert_eq!(
    RecordedHost::all_from_golden("committed", &committed).expect("committed set parses"),
    hosts,
    "a re-run on a recorded class changes nothing at all — the source on file is the label \
     of a run that DID produce this payload, and rewriting provenance is not this tool's job"
  );
  assert_eq!(payload_of(&merged), payload_of(&committed));
  assert!(
    stderr.contains("UNCHANGED"),
    "the no-op must say so rather than looking like an append: {stderr}"
  );
}

/// The legacy single-host stamp is PROMOTED into the set, carrying the oracle
/// label that lived at the document level.
///
/// This is the path that produced the committed fixtures: both the golden on
/// disk and the artifact the regeneration workflow uploaded were written in the
/// old schema, and the set they now carry was assembled by this tool from those
/// two documents rather than typed by hand.
#[test]
fn merge_tool_promotes_a_legacy_single_host_stamp_into_the_set() {
  let legacy = |build: &str, version: &str, source: &str| {
    serde_json::json!({
      "model": "openai_whisper-tiny",
      "source": source,
      "generationHost": {
        "osBuild": build,
        "osProductVersion": version,
        "chip": "Apple M1 (Virtual)",
        "arch": "arm64",
      },
      "text": "And so my fellow Americans",
      "language": "en",
      "tokens": [400, 370, 452],
    })
  };

  let committed = legacy("24G720", "15.7.7", V1_0);
  let fresh = legacy("24G830", "15.7.9", V1_1);
  let (merged, stderr) = run_merge("legacy", &committed, &fresh);
  let hosts = RecordedHost::all_from_golden("merged", &merged).expect("merged set parses");

  assert_eq!(hosts.len(), 2, "{stderr}");
  assert_eq!(hosts[0].class.os_build, "24G720");
  assert_eq!(hosts[1].class.os_build, "24G830");
  assert_eq!(
    hosts[0].source.as_deref(),
    Some(V1_0),
    "the document-level oracle label of a legacy stamp belongs to ITS host, and must survive \
     the promotion rather than being dropped"
  );
  assert_eq!(hosts[1].source.as_deref(), Some(V1_1));
  assert!(
    merged.get("generationHost").is_none(),
    "the legacy key must be REPLACED by the set, not left beside it to disagree"
  );
  assert_eq!(payload_of(&merged), payload_of(&committed));
}

// ── 3. The gate cannot be dissolved ─────────────────────────────────────────

/// STRUCTURAL: no committed golden may claim coremlit as its producer — at the
/// document level or on any recorded host.
///
/// The `source` field is the golden's own statement of where its numbers came
/// from. `rust-coreml (self-golden)` was a real, documented fallback in the
/// original pipeline plan — it was never taken (all three goldens name the
/// Swift oracle), and this makes taking it later a test failure rather than a
/// judgement call.
#[test]
fn no_committed_golden_claims_coremlit_as_its_oracle() {
  for golden in GOLDENS {
    let value = common::load_golden_json(golden);
    let document_source = value
      .get("source")
      .and_then(serde_json::Value::as_str)
      .unwrap_or_else(|| panic!("{golden}: missing a `source` field — provenance is not optional"))
      .to_string();
    let hosts = RecordedHost::all_from_golden(golden, &value)
      .unwrap_or_else(|e| panic!("{golden}: the recorded host set must parse: {e}"));

    // The document's own claim, plus every per-host claim: promoting the label
    // into the set must not create a place where an unexamined `source` can sit.
    let claims = std::iter::once(document_source)
      .chain(hosts.iter().filter_map(|h| h.source.clone()))
      .collect::<Vec<_>>();
    assert!(
      claims.len() > 1,
      "{golden}: no per-host `source` was checked — a guard over nothing passes vacuously"
    );

    for source in claims {
      let external = source.contains("whisperkit-cli") || source.contains("Swift WhisperKit");
      assert!(
        external,
        "{golden}: `source` is {source:?}, which does not name the external Swift oracle. A \
         golden's entire value is that something other than the code under test produced it."
      );

      // The two markers of the documented (never-taken) self-golden fallback.
      // Deliberately NOT a bare "coremlit": the words golden's `source` cites a
      // repository PATH, and a guard that cannot tell a citation from a claim
      // gets deleted the first time it cries wolf.
      for banned in ["self-golden", "rust-coreml"] {
        assert!(
          !source.to_lowercase().contains(banned),
          "{golden}: `source` is {source:?} — this golden claims to have been produced by the \
           crate it is supposed to be an independent check on. Regenerate it from whisperkit-cli \
           via {WHISPER_REGEN_SCRIPT}"
        );
      }
    }
  }
}

/// STRUCTURAL: neither regeneration script can emit coremlit's own output.
///
/// They are shell scripts whose only source of numbers is the `--report` JSON
/// `whisperkit-cli` writes, and the committed golden they merge it with. The way
/// that could stop being true is someone adding a "if the CLI is missing, fall
/// back to the Rust path" arm — which would need Rust's build tool. So the
/// absence of that tool's name is the tripwire, and both scripts' headers say
/// so, so nobody adds one by accident.
#[test]
fn regen_script_cannot_emit_coremlits_own_output() {
  let script = read_repo_file(REGEN_SCRIPT);

  assert!(
    script.contains("whisperkit-cli transcribe"),
    "{REGEN_SCRIPT} must drive the external Swift oracle"
  );
  // Byte-assembled so this test's own source does not trip the grep it runs.
  let build_tool = concat!("car", "go");
  for (name, body) in [
    (REGEN_SCRIPT, &script),
    (MERGE_TOOL, &read_repo_file(MERGE_TOOL)),
  ] {
    assert!(
      !body.to_lowercase().contains(build_tool),
      "{name} mentions Rust's build tool. The goldens must come from whisperkit-cli and \
       nothing else: a regeneration path that can invoke this crate can re-baseline a golden \
       against the very output it is meant to check, and every whisper parity gate then \
       asserts only that coremlit agrees with coremlit."
    );
  }
  assert!(
    script.contains("generationHosts"),
    "{REGEN_SCRIPT} must stamp the host it ran on, or the goldens it writes are unattributable"
  );
  // Read off the runner, never hardcoded — otherwise the stamp becomes a lie
  // the first time the image rotates.
  for key in [
    "kern.osversion",
    "kern.osproductversion",
    "machdep.cpu.brand_string",
  ] {
    assert!(
      script.contains(key),
      "{REGEN_SCRIPT} must read `{key}` from the running machine to build the host stamp"
    );
  }
  // And it must route its output through the decision tool rather than
  // overwriting the committed golden: an overwrite is what threw away the other
  // hosts' evidence and made a rolling runner pool red half the jobs.
  assert!(
    script.contains("merge_golden_hosts.sh"),
    "{REGEN_SCRIPT} must hand its fresh output to {MERGE_TOOL}, which decides by measurement \
     whether this host JOINS the golden's recorded set or REPLACES it"
  );
}

/// STRUCTURAL: the merge tool decides by MEASURING the payload, and cannot aim
/// at a committed golden by itself.
#[test]
fn merge_tool_decides_by_measurement_and_writes_no_file() {
  let tool = read_repo_file(MERGE_TOOL);

  assert!(
    tool.contains("del(.generationHost, .generationHosts, .source)"),
    "{MERGE_TOOL} must compare the payload with the provenance keys stripped — that \
     comparison IS the append-or-replace decision, and without it every run appends"
  );
  assert!(
    !tool.contains("fixtures/golden"),
    "{MERGE_TOOL} must take both goldens as ARGUMENTS and write to stdout. A script that \
     names the committed fixtures can write into them; the CI job's proof that it left the \
     checkout alone rests on it not being able to."
  );
}

/// STRUCTURAL: the regeneration workflow cannot commit what it produces.
///
/// A job that regenerated and committed on every push would make the goldens
/// decorative — whatever the oracle emitted that day would silently become the
/// expectation. Four independent mechanisms stop that, and this pins all four,
/// because any one of them alone could be edited away.
#[test]
fn regen_workflow_cannot_commit_what_it_produces() {
  const WORKFLOW: &str = ".github/workflows/regen-whisper-goldens.yml";
  let workflow = read_repo_file(WORKFLOW);

  // (1) manual only.
  assert!(
    workflow.contains("workflow_dispatch:"),
    "{WORKFLOW} must be manually dispatched"
  );
  for trigger in ["\n  push:", "\n  pull_request:"] {
    assert!(
      !workflow.contains(trigger),
      "{WORKFLOW} must not run on `{}` — regenerating goldens on every code change is what \
       turns a parity gate into a self-portrait",
      trigger.trim()
    );
  }

  // (2) a token that cannot write.
  assert!(
    workflow.contains("permissions:") && workflow.contains("contents: read"),
    "{WORKFLOW} must hold a read-only `contents` permission so it cannot push a golden"
  );

  // (3) writes outside the checkout, and (4) proves it afterwards.
  assert!(
    workflow.contains("WHISPER_GOLDEN_OUT"),
    "{WORKFLOW} must redirect the script's output away from the committed fixtures"
  );
  assert!(
    workflow.contains("git status --porcelain"),
    "{WORKFLOW} must assert it left the checkout unmodified"
  );

  // And the output must leave as something a human picks up deliberately.
  assert!(
    workflow.contains("upload-artifact"),
    "{WORKFLOW} must hand the goldens to a human as an artifact to review and commit"
  );

  // The two branches a human has to be able to tell apart in the artifact.
  for branch in ["APPEND", "REPLACE"] {
    assert!(
      workflow.contains(branch),
      "{WORKFLOW}'s header must document the {branch} branch of the merge decision — the \
       artifact's meaning depends on which one ran"
    );
  }
}

/// STRUCTURAL: no whisper test writes into the golden directory.
///
/// The `UPDATE_GOLDEN` arms in `parity_es.rs` and `parity_jfk.rs` did exactly
/// that — they serialized this crate's own `TranscriptionResult` over the
/// committed golden. They are gone. This is what keeps them gone, including in
/// the tempting form (`UPDATE_GOLDEN=1 cargo test ...` to clear a red gate on a
/// foreign host, which is precisely the situation the host-class diagnosis
/// exists to talk you out of).
#[test]
fn no_whisper_test_can_write_a_golden() {
  // This file is the one legitimate place in the tree to SAY these names — it
  // is the guard, and its doc comments explain what it forbids and why. It is
  // also the only one that writes anything: the merge-tool cases above put
  // SYNTHETIC inputs under `CARGO_TARGET_TMPDIR` (see `scratch_dir`, which
  // asserts it) and read the tool's decision off its stdout. The exemption is
  // therefore taken by exact filename rather than by a wildcard, so the hole is
  // one auditable line rather than a category.
  const GUARD: &str = "golden_provenance.rs";

  let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/whisper");
  let mut checked = 0usize;
  let mut skipped = 0usize;
  let mut offenders = Vec::new();

  for entry in std::fs::read_dir(&dir).expect("tests/whisper is readable") {
    let path = entry.expect("readable dir entry").path();
    if path.extension().is_none_or(|e| e != "rs") {
      continue;
    }
    if path.file_name().is_some_and(|n| n == GUARD) {
      skipped += 1;
      continue;
    }
    let body = std::fs::read_to_string(&path).expect("readable test source");
    checked += 1;
    // `fs::write` is how a golden gets overwritten; `UPDATE_GOLDEN` is the env
    // switch that used to guard it in `parity_es.rs` and `parity_jfk.rs`.
    // Either one, in this tree, is the re-baselining path coming back — most
    // temptingly as `UPDATE_GOLDEN=1 ...` to clear a red gate on a foreign
    // host, which is exactly the move the host-class diagnosis exists to talk
    // you out of.
    for pattern in [concat!("fs:", ":write"), concat!("UPDATE_", "GOLDEN")] {
      if body.contains(pattern) {
        offenders.push(format!("{}: {pattern}", path.display()));
      }
    }
  }

  assert_eq!(
    skipped, 1,
    "the exemption must cover exactly this guard file"
  );
  assert!(
    checked >= 5,
    "expected to scan the whisper suite, saw {checked} files — a scan of nothing passes \
     vacuously"
  );
  assert!(
    offenders.is_empty(),
    "these whisper tests can write a golden from this crate's own output, which would make \
     the parity gates assert that coremlit agrees with coremlit:\n  {}\n\nRegeneration is \
     legitimate, but only from the external Swift oracle: {WHISPER_REGEN_SCRIPT}",
    offenders.join("\n  ")
  );
}
