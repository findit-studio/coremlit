//! Hermetic proof that the siglip measured-band host gate behaves, on all three
//! verdicts, without owning three machines.
//!
//! # Status: complete, hermetic
//!
//! No models, no network, no `SIGLIP_TEST_MODELS` — these drive
//! `tests/support/measured_band.rs` with synthetic host classes, so they run in
//! CI's `features` job on every push. That matters: the band gate is the thing
//! standing between a host-specific measurement and a false red, so its own
//! behaviour must not itself be model-gated.
//!
//! What is pinned here:
//!
//! 1. **The predicate is three-way** — unrecorded, matching, foreign.
//! 2. **An unarmed band cannot red.** An out-of-band value under
//!    `Unrecorded`/`Foreign` returns instead of panicking. This is the exact
//!    defect PR #89's CI run hit: vision ANE 0.999664 against a `[0.20, 0.70]`
//!    band characterized on another computer.
//! 3. **An armed band still reds.** On the recorded host the assertion is
//!    unchanged, and its message carries the attribution.
//! 4. **Nothing skips silently.** Every verdict prints a banner naming the
//!    reason, the running host and the re-characterization command, and every
//!    measurement prints its number whether or not it is asserted.
//!
//! The band VALUES themselves live in `parity_embed.rs` / `placement.rs`; this
//! file never asserts a cosine.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{BandGate, BandVerdict, CharacterizedHost, HostClass, band_verdict};

/// A synthetic running host — deliberately not this machine's, so nothing here
/// depends on where it runs.
fn running_host() -> HostClass {
  HostClass {
    os_build: "24F74".to_string(),
    os_product_version: "15.5".to_string(),
    chip: "Apple M9 Ultra".to_string(),
    arch: "arm64".to_string(),
  }
}

/// The recorded constant that MATCHES [`running_host`].
const SAME_HOST: CharacterizedHost = CharacterizedHost {
  os_build: "24F74",
  os_product_version: "15.5",
  chip: "Apple M9 Ultra",
  arch: "arm64",
};

/// A recorded constant for a DIFFERENT host class (a different chip generation —
/// the axis the Neural Engine's fp16 arithmetic actually varies along).
const OTHER_HOST: CharacterizedHost = CharacterizedHost {
  os_build: "24F74",
  os_product_version: "15.5",
  chip: "Apple M1",
  arch: "arm64",
};

fn gate(recorded: Option<CharacterizedHost>) -> BandGate {
  BandGate::open_with(
    "hermetic band gate",
    recorded,
    running_host(),
    common::recharacterize_command(
      "siglip_placement",
      "crates/coremlit/tests/siglip/placement.rs",
    ),
  )
}

/// The three-way contract, on the pure predicate: absent → `Unrecorded`,
/// present-and-equal → `Armed`, present-and-different → `Foreign`.
#[test]
fn band_verdict_is_three_way() {
  let running = running_host();
  assert_eq!(band_verdict(None, &running), BandVerdict::Unrecorded);
  assert_eq!(band_verdict(Some(&SAME_HOST), &running), BandVerdict::Armed);
  assert_eq!(
    band_verdict(Some(&OTHER_HOST), &running),
    BandVerdict::Foreign
  );

  // Only `Armed` asserts. This is the whole safety property in one line.
  assert!(BandVerdict::Armed.asserts());
  assert!(!BandVerdict::Unrecorded.asserts());
  assert!(!BandVerdict::Foreign.asserts());
}

/// Every field of the host class participates in the match — a band recorded on
/// a different macOS build, product version, chip or arch is `Foreign`, not
/// `Armed`.
#[test]
fn every_host_class_field_is_load_bearing() {
  let running = running_host();
  assert!(SAME_HOST.matches(&running));
  for differing in [
    CharacterizedHost {
      os_build: "24G90",
      ..SAME_HOST
    },
    CharacterizedHost {
      os_product_version: "15.6",
      ..SAME_HOST
    },
    CharacterizedHost {
      chip: "Apple M1",
      ..SAME_HOST
    },
    CharacterizedHost {
      arch: "x86_64",
      ..SAME_HOST
    },
  ] {
    assert!(
      !differing.matches(&running),
      "{differing} must not match {running}"
    );
    assert_eq!(
      band_verdict(Some(&differing), &running),
      BandVerdict::Foreign
    );
  }
}

/// THE regression this whole mechanism exists for: an UNRECORDED band must not
/// red on a wildly out-of-band value. Replays PR #89's exact CI numbers.
#[test]
fn an_unrecorded_band_reports_and_cannot_red() {
  let gate = gate(None);
  assert_eq!(gate.verdict(), BandVerdict::Unrecorded);
  assert!(!gate.armed());

  // parity_embed: the runner's 0.99966413 against the pinned 0.9999 (at f32
  // precision, which is what the gate compares).
  let floor_line = gate.check_floor("vision CpuAndGpu parity", 0.999_664_1, 0.9999);
  // placement: the runner's 0.999664 against the characterized [0.20, 0.70].
  let band_line = gate.check_band("vision ANE worst", 0.999_664, 0.20, 0.70);
  // placement's non-vacuity companion, false on the runner for the same reason.
  let ceiling_line = gate.check_ceiling("vision ANE non-vacuity", 0.999_664, 0.9);

  for line in [&floor_line, &band_line, &ceiling_line] {
    assert!(
      line.contains("OUTSIDE"),
      "the line must say the value did not fit, even unarmed: {line}"
    );
    assert!(
      line.contains("BAND NOT ASSERTED"),
      "an unasserted band must say so on its own line: {line}"
    );
    assert!(
      line.contains("no characterization host recorded"),
      "and must say WHY: {line}"
    );
    assert!(
      line.contains("0.9996"),
      "the measurement must be printed so the band can be characterized from this log: {line}"
    );
  }
}

/// The unrecorded banner is loud enough that reading a log tells you the band
/// was not checked, why, what still gates, and how to arm it.
#[test]
fn the_unrecorded_banner_explains_itself() {
  let banner = gate(None).banner();
  for needle in [
    "NOT ASSERTED",
    "no characterization host is recorded in source",
    "CHARACTERIZED_ON = None",
    // The running host class, so a CI log carries what to record.
    "Apple M9 Ultra",
    // The portable floors are NOT what is being skipped.
    "still gated",
    // The exact command that arms it.
    "cargo test -p coremlit --features siglip --test siglip_placement",
    "crates/coremlit/tests/siglip/placement.rs",
  ] {
    assert!(
      banner.contains(needle),
      "banner is missing {needle:?}:\n{banner}"
    );
  }
}

/// A FOREIGN band reports both host classes and does not red — the same
/// no-false-red property as unrecorded, with a diagnosis that names the machine
/// the numbers describe.
#[test]
fn a_foreign_band_reports_both_hosts_and_cannot_red() {
  let gate = gate(Some(OTHER_HOST));
  assert_eq!(gate.verdict(), BandVerdict::Foreign);
  assert!(!gate.armed());

  let line = gate.check_band("vision ANE worst", 0.999_664, 0.20, 0.70);
  assert!(line.contains("OUTSIDE"), "{line}");
  assert!(line.contains("BAND NOT ASSERTED"), "{line}");
  assert!(
    line.contains("characterized on a different host class"),
    "{line}"
  );

  let banner = gate.banner();
  for needle in [
    "NOT ASSERTED",
    "DIFFERENT host class",
    // Both hosts, so the reader can see which machine the band describes.
    "Apple M1",
    "Apple M9 Ultra",
    "still gated",
    // The anti-widening rule, stated where someone would reach for it.
    "never widen one band to span",
    "cargo test -p coremlit --features siglip --test siglip_placement",
  ] {
    assert!(
      banner.contains(needle),
      "banner is missing {needle:?}:\n{banner}"
    );
  }
}

/// An ARMED band keeps its full strictness: an in-band value passes and says
/// ASSERTED, and an out-of-band value PANICS with the failure plus the
/// attribution. Losing this would turn the fix into a deletion of the gate.
#[test]
fn an_armed_band_still_reds() {
  let gate = gate(Some(SAME_HOST));
  assert_eq!(gate.verdict(), BandVerdict::Armed);
  assert!(gate.armed());

  // In-band: passes, and the line records that it was really asserted.
  let ok = gate.check_band("vision ANE worst", 0.31, 0.20, 0.70);
  assert!(ok.contains("[ASSERTED]"), "{ok}");
  assert!(!ok.contains("BAND NOT ASSERTED"), "{ok}");
  assert!(ok.contains("0.31"), "{ok}");

  // Out-of-band, all three shapes: each must panic, with the band it broke, the
  // attribution, and a refusal of the wrong fix.
  let expect_red = |expected: &str, body: &dyn Fn()| {
    let err = catch_unwind(AssertUnwindSafe(body))
      .expect_err("an armed band MUST still fail on an out-of-band value");
    let message = err
      .downcast_ref::<String>()
      .map_or_else(|| "<non-string panic>".to_string(), Clone::clone);
    assert!(
      message.contains(expected),
      "armed failure must name the band it broke: wanted {expected:?} in\n{message}"
    );
    assert!(
      message.contains("this IS the host class the band was characterized on"),
      "armed failure must say the failure is attributable:\n{message}"
    );
    assert!(
      message.contains("do NOT widen the band"),
      "armed failure must forbid the wrong fix:\n{message}"
    );
  };

  // The exact CI numbers from PR #89, which must red HERE and only here.
  expect_red("outside the characterized band [0.2, 0.7]", &|| {
    gate.check_band("vision ANE worst", 0.999_664, 0.20, 0.70);
  });
  expect_red("below the measured floor 0.9999", &|| {
    gate.check_floor("vision CpuAndGpu parity", 0.999_664_1, 0.9999);
  });
  expect_red("not below the measured ceiling 0.9", &|| {
    gate.check_ceiling("vision ANE non-vacuity", 0.999_664, 0.9);
  });
}

/// The armed banner says the opposite of the other two: bands ARE asserted here,
/// so a band failure is a finding.
#[test]
fn the_armed_banner_says_bands_are_enforced() {
  let banner = gate(Some(SAME_HOST)).banner();
  assert!(banner.contains("ARE ASSERTED"), "{banner}");
  assert!(!banner.contains("NOT ASSERTED"), "{banner}");
  assert!(banner.contains("Apple M9 Ultra"), "{banner}");
  assert!(banner.contains("do not widen"), "{banner}");
}

/// An in-band value prints its number under EVERY verdict — the "always
/// measured, always printed" half of the contract, which is what lets a runner
/// be characterized later from its own logs.
#[test]
fn every_verdict_prints_the_measurement() {
  for recorded in [None, Some(SAME_HOST), Some(OTHER_HOST)] {
    let gate = gate(recorded);
    let line = gate.check_band("vision ANE worst", 0.314_159, 0.20, 0.70);
    assert!(
      line.contains("0.3141"),
      "every verdict must print the measured value: {line}"
    );
    assert!(line.contains("[band]"), "{line}");
    assert!(line.contains("ok"), "an in-band value reads as ok: {line}");
  }
}

/// This machine's host class is readable, well-formed, and PRINTED — so anyone
/// running `cargo test -p coremlit --features siglip` on a machine they want to
/// characterize can copy the four fields straight into a `CHARACTERIZED_ON`.
///
/// Mirrors the speaker/vad/whisper smoke tests of the same name; it is the
/// reason "record the host you measured on" is a one-minute task rather than a
/// reason to guess.
#[test]
fn running_host_class_is_readable_for_recording() {
  let running = HostClass::running();
  println!("[host] this machine's host class: {running}");
  println!(
    "[host] to record it:\n  const CHARACTERIZED_ON: Option<common::CharacterizedHost> = \
     Some(common::CharacterizedHost {{\n    os_build: {:?},\n    os_product_version: {:?},\n    \
     chip: {:?},\n    arch: {:?},\n  }});",
    running.os_build, running.os_product_version, running.chip, running.arch
  );

  for (field, value) in [
    ("os_build", &running.os_build),
    ("os_product_version", &running.os_product_version),
    ("chip", &running.chip),
    ("arch", &running.arch),
  ] {
    assert!(!value.is_empty(), "host class field {field} is empty");
  }
  assert!(
    running.arch == "arm64" || running.arch == "x86_64",
    "unexpected process arch {:?}",
    running.arch
  );

  // A host class read from this machine matches itself and nothing else — the
  // round trip that makes recording meaningful.
  // `String::leak` because `CharacterizedHost` holds `&'static str` (so a real
  // recording can be a `const`). Four short strings, once, in a test process.
  let recorded = CharacterizedHost {
    os_build: running.os_build.clone().leak(),
    os_product_version: running.os_product_version.clone().leak(),
    chip: running.chip.clone().leak(),
    arch: running.arch.clone().leak(),
  };
  assert_eq!(band_verdict(Some(&recorded), &running), BandVerdict::Armed);
}

// ── The fix cannot be silently reverted ─────────────────────────────────────
//
// The defect being fixed is one line long: a host-measured number asserted with
// a bare `assert!`. Nothing above stops someone re-adding one, so this reads the
// two suite sources — the `models_lock.rs` / `feature_map.rs` idiom, which parse
// their sources rather than model them — and pins the routing:
//
//   * every MEASURED constant reaches the test ONLY as an argument to a
//     `gate.check_*` call, and
//   * every PORTABLE floor reaches it ONLY as a bare `assert!`,
//
// so a reversion in either direction reds here, hermetically, without a model.

/// Reads one suite source and returns it with comments and string literals
/// removed, so the routing scan cannot be fooled by prose or by a label that
/// happens to contain a constant's name or a `;`.
fn scannable_source(rel: &str) -> String {
  let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("..")
    .join("..")
    .join(rel);
  let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  let mut out = String::with_capacity(text.len());
  for line in text.lines() {
    let code = line.split("//").next().unwrap_or("");
    let mut in_string = false;
    let mut escaped = false;
    for ch in code.chars() {
      match ch {
        _ if escaped => escaped = false,
        '\\' if in_string => escaped = true,
        '"' => in_string = !in_string,
        _ if in_string => {}
        _ => out.push(ch),
      }
    }
    out.push('\n');
  }
  out
}

/// Splits stripped source into statements, dropping `const` declarations (where
/// every constant necessarily names itself).
fn statements(source: &str) -> Vec<String> {
  source
    .split(';')
    .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
    .filter(|s| !s.is_empty() && !s.contains("const "))
    .collect()
}

/// Every MEASURED constant is routed through the band gate, and every PORTABLE
/// floor is asserted bare — in both suites, checked against their real source.
#[test]
fn measured_constants_are_routed_through_the_gate() {
  let cases: [(&str, &[&str], &[&str]); 2] = [
    (
      "crates/coremlit/tests/siglip/parity_embed.rs",
      &["MEASURED_FLOOR"],
      &["PARITY_FLOOR"],
    ),
    (
      "crates/coremlit/tests/siglip/placement.rs",
      &[
        "VISION_ANE_LO",
        "VISION_ANE_HI",
        "VISION_ANE_NON_VACUITY",
        "ALL_TRACKS_ANE_TOL",
      ],
      &["FLOOR", "TEXT_ROBUST_FLOOR"],
    ),
  ];

  for (rel, measured, portable) in cases {
    let source = scannable_source(rel);
    assert!(
      source.contains("CHARACTERIZED_ON"),
      "{rel} no longer records a characterization host"
    );
    assert!(
      source.contains("BandGate::open"),
      "{rel} no longer opens a band gate"
    );

    for statement in statements(&source) {
      for name in measured {
        if statement.contains(*name) {
          assert!(
            statement.contains("gate.check_"),
            "{rel}: MEASURED constant {name} is asserted outside the band gate — that is the \
             defect this mechanism exists to prevent (a number measured on one machine, \
             asserted as if it were portable). Route it through gate.check_floor / \
             check_ceiling / check_band.\n  offending statement: {statement}"
          );
        }
      }
      for name in portable {
        // `TEXT_ROBUST_FLOOR` contains `FLOOR`, so match on a word boundary.
        let mentions = statement
          .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
          .any(|word| word == *name);
        if mentions {
          assert!(
            !statement.contains("gate.check_"),
            "{rel}: PORTABLE floor {name} was moved into the band gate — a spec contract must \
             gate on EVERY host, including a runner that never characterized anything. Assert \
             it bare.\n  offending statement: {statement}"
          );
        }
      }
    }
  }
}
