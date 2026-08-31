//! Provenance of the whisper goldens: the host-class predicate's falsifiers,
//! and the structural guards that keep a golden producible only by the external
//! Swift oracle.
//!
//! Hermetic — no model, no network, no CoreML. These run in CI's `features` job
//! on every PR, which is the point: the model-gated parity tests they protect
//! run only where `openai_whisper-tiny` is staged, and the rules below have to
//! hold everywhere.
//!
//! Two jobs, in the order they matter:
//!
//! 1. **The gate works.** `check_host_class` must diagnose a foreign host, pass
//!    a matching one, and tolerate an unstamped legacy golden with the
//!    ambiguity note — the three cases the parity suites depend on, driven here
//!    with synthetic values so no hardware is involved.
//! 2. **The gate cannot be dissolved.** A golden re-baselined against
//!    coremlit's own output would make every whisper parity test assert that
//!    coremlit agrees with coremlit. The crate no longer contains any code that
//!    can write one — the `UPDATE_GOLDEN` writer arms that used to sit in
//!    `parity_es.rs` and `parity_jfk.rs` were deleted, not merely discouraged —
//!    and the guards below fail the suite if a path back appears.

mod common;

use common::{HostClass, HostVerdict, WHISPER_REGEN_SCRIPT, check_host_class, legacy_failure_note};

/// Every committed golden the whisper suites read.
const GOLDENS: [&str; 3] = [
  "es_tiny_golden.json",
  "jfk_tiny_golden.json",
  "jfk_tiny_words_golden.json",
];

/// The repository root, for the two non-Rust artifacts guarded below.
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

// ── 1. The gate works ───────────────────────────────────────────────────────

/// FALSIFIER: a golden stamped with a DIFFERENT host-class must be diagnosed,
/// never compared.
///
/// This is the case CI hits after GitHub rotates the macos-15 image. It has to
/// produce an unmistakable instruction, because the alternative — what the
/// three whisper gates did from late July until this test existed — is a raw
/// token divergence that reads like a port defect and gets ignored as flake.
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

  for running in [&rotated_image, &other_silicon] {
    let diagnosis = check_host_class(
      "es_tiny_golden.json",
      Some(&golden_host),
      running,
      WHISPER_REGEN_SCRIPT,
    )
    .expect_err("a differing host-class must be diagnosed, not silently compared");

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
      Some(&host),
      &host,
      WHISPER_REGEN_SCRIPT
    ),
    Ok(HostVerdict::Match)
  );
}

/// FALSIFIER: an UNSTAMPED (legacy) golden is tolerated, with the ambiguity
/// note — it does not become an error, and it does not become a free pass.
///
/// This is the state the committed goldens are in until the runner job
/// regenerates them, so it is the path the three whisper gates take today: the
/// tight bounds still run, and a failure carries the note that says why it
/// cannot be pinned on the port yet.
#[test]
fn unstamped_legacy_golden_is_tolerated_with_the_ambiguity_note() {
  let running = synthetic_host();
  assert_eq!(
    check_host_class("jfk_tiny_golden.json", None, &running, WHISPER_REGEN_SCRIPT),
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

/// `HostClass::from_golden` is strict about a `generationHost` that IS present
/// and tolerant of one that is not — the asymmetry the legacy path rests on.
#[test]
fn generation_host_parse_is_strict_and_legacy_tolerant() {
  assert_eq!(
    HostClass::from_golden("wf", &serde_json::json!({})).expect("absent must be Ok"),
    None,
    "an unstamped golden must keep parsing on every host"
  );

  let stamped = serde_json::json!({
    "generationHost": {
      "osBuild": "24G84",
      "osProductVersion": "15.6",
      "chip": "Apple M2",
      "arch": "arm64"
    }
  });
  assert_eq!(
    HostClass::from_golden("wf", &stamped).expect("well-formed must be Ok"),
    Some(HostClass {
      os_build: "24G84".to_string(),
      os_product_version: "15.6".to_string(),
      chip: "Apple M2".to_string(),
      arch: "arm64".to_string(),
    })
  );

  // A half-written stamp is an error, not a silent legacy fallback: silently
  // downgrading it would turn a typo into a permanently un-attributable gate.
  let missing_arch = serde_json::json!({
    "generationHost": { "osBuild": "24G84", "osProductVersion": "15.6", "chip": "Apple M2" }
  });
  let err = HostClass::from_golden("wf", &missing_arch).unwrap_err();
  assert!(err.contains("generationHost.arch"), "{err}");

  let not_an_object = serde_json::json!({ "generationHost": "24G84" });
  let err = HostClass::from_golden("wf", &not_an_object).unwrap_err();
  assert!(err.contains("generationHost"), "{err}");
}

/// Every committed golden parses on THIS host, whatever its stamp state — the
/// loader must never be the thing that fails. Stamped or not is deliberately
/// not asserted: the committed goldens are unstamped until a runner regenerates
/// them, and pinning either state here would make this test the thing that
/// blocks that.
#[test]
fn every_committed_golden_parses_its_generation_host() {
  for golden in GOLDENS {
    let value = common::load_golden_json(golden);
    let parsed = HostClass::from_golden(golden, &value)
      .unwrap_or_else(|e| panic!("{golden}: generationHost must parse or be absent: {e}"));
    match parsed {
      Some(host) => println!("[host] {golden}: stamped {host}"),
      None => println!("[host] {golden}: unstamped (legacy)"),
    }
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

// ── 2. The gate cannot be dissolved ─────────────────────────────────────────

/// STRUCTURAL: no committed golden may claim coremlit as its producer.
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
    let source = value
      .get("source")
      .and_then(serde_json::Value::as_str)
      .unwrap_or_else(|| panic!("{golden}: missing a `source` field — provenance is not optional"));

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

/// STRUCTURAL: the regeneration script cannot emit coremlit's own output.
///
/// It is a shell script whose only source of numbers is the `--report` JSON
/// `whisperkit-cli` writes. The way it could stop being that is someone adding
/// a "if the CLI is missing, fall back to the Rust path" arm — which would need
/// Rust's build tool. So the absence of that tool's name is the tripwire, and
/// the script's header says so, so nobody adds one by accident.
#[test]
fn regen_script_cannot_emit_coremlits_own_output() {
  const SCRIPT: &str = "coremlit/tests/whisper/swift/regen_goldens.sh";
  let script = read_repo_file(SCRIPT);

  assert!(
    script.contains("whisperkit-cli transcribe"),
    "{SCRIPT} must drive the external Swift oracle"
  );
  // Byte-assembled so this test's own source does not trip the grep it runs.
  let build_tool = concat!("car", "go");
  assert!(
    !script.to_lowercase().contains(build_tool),
    "{SCRIPT} mentions Rust's build tool. The goldens must come from whisperkit-cli and \
     nothing else: a regeneration path that can invoke this crate can re-baseline a golden \
     against the very output it is meant to check, and every whisper parity gate then \
     asserts only that coremlit agrees with coremlit."
  );
  assert!(
    script.contains("generationHost"),
    "{SCRIPT} must stamp the host it ran on, or the goldens it writes are unattributable"
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
      "{SCRIPT} must read `{key}` from the running machine to build generationHost"
    );
  }
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
  // therefore the single exemption, taken by exact filename rather than by a
  // wildcard so the hole is one auditable line rather than a category.
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
