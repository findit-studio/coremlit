//! Host-class provenance for the committed-Swift-golden parity gates — the
//! SINGLE copy, shared by the speaker, vad and whisper suites.
//!
//! CoreML floating point is not contracted to be identical across macOS builds
//! or chip generations (#36): neither the `CpuOnly` kernels, which ship as part
//! of the OS binary set, nor the Neural Engine's fp16 arithmetic, whose last
//! bit can differ between Apple Silicon generations. A parity gate that
//! compares against an external Swift oracle therefore has to know WHICH host
//! that oracle ran on, or a host difference gets reported as a port defect.
//!
//! So every golden carries an optional `generationHost` block, and the gates
//! enforce their tight bounds against it via [`check_host_class`]:
//!
//! | recorded `generationHost` | verdict | what the gate does |
//! | --- | --- | --- |
//! | absent | [`HostVerdict::LegacyUnknown`] | tight bounds still enforced; [`legacy_failure_note`] appended to any failure, because a FAILURE is ambiguous |
//! | present, equal | [`HostVerdict::Match`] | strict comparison; a failure is cleanly the port's |
//! | present, different | `Err(diagnosis)` | panic BEFORE any CoreML number, naming the regeneration command and stating this is not a port defect |
//!
//! This file was promoted out of `tests/{speaker,vad}/common/mod.rs`, which
//! held byte-identical 188-line copies of it. It is pulled in by `#[path]`, the
//! `tests/support/coremlit_dir.rs` convention, so the three suites cannot
//! drift; each suite keeps its own hermetic tests driving this one copy, and
//! each passes its own regeneration command.

/// The host-class identity that determines CoreML float reproducibility on
/// either compute path: macOS build (the OS binary set every CPU kernel — and
/// every ANE compiler/firmware revision — ships in), chip (kernel dispatch
/// varies by microarchitecture, and the ANE's fp16 units differ by
/// generation), process arch (Rosetta), plus the human-readable product
/// version (fully determined by the build; carried for diagnostics).
/// Deliberately NOT included: Xcode / Swift toolchain (compiles the dumper,
/// not the runtime kernels — inputs are FNV-pinned and model bytes
/// SHA-pinned), `hw.model` (same chip + build ⇒ same floats), RAM/core
/// counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostClass {
  pub os_build: String,
  pub os_product_version: String,
  pub chip: String,
  pub arch: String,
}

impl HostClass {
  /// The RUNNING host's class. Reads the sysctl keys the Swift dumpers read
  /// (`kern.osversion`, `kern.osproductversion`, `machdep.cpu.brand_string`) by
  /// shelling out to `/usr/sbin/sysctl`, and normalizes the process arch to the
  /// `arm64`/`x86_64` spelling the dumpers record. Called only from model-gated
  /// tests and the `running_host_class_is_well_formed` smoke test — hermetic
  /// predicate tests use synthetic values.
  ///
  /// Production code (`src/audio/whisper/model/mod.rs::device_identifier`)
  /// deliberately uses `libc::sysctlbyname`; this test-side reader deliberately
  /// shells out instead — spawn cost is irrelevant in a model-gated test and it
  /// keeps the test tree free of `unsafe`.
  ///
  /// # Panics
  /// With a `host-class gate:` message if a sysctl read fails or is empty
  /// (model-gated dev machines only — sysctl always exists on macOS).
  pub fn running() -> Self {
    HostClass {
      os_build: sysctl_string("kern.osversion"),
      os_product_version: sysctl_string("kern.osproductversion"),
      chip: sysctl_string("machdep.cpu.brand_string"),
      // The aarch64 -> arm64 normalization is load-bearing: without it every
      // Apple-Silicon run would mismatch every golden (the dumpers record
      // `arm64` via compile-time `#if arch(arm64)`, and compile-time arch IS
      // the process arch that governs the in-process `CpuOnly` kernels).
      arch: match std::env::consts::ARCH {
        "aarch64" => "arm64".to_string(),
        other => other.to_string(),
      },
    }
  }

  /// Reads the OPTIONAL `generationHost` object from a golden's JSON. Absent or
  /// `null` → `Ok(None)` (a legacy golden, pre-host-provenance). Present → must
  /// be an object whose `osBuild`/`osProductVersion`/`chip`/`arch` are all
  /// non-empty strings; unknown extra keys inside the object are tolerated.
  ///
  /// FORM only — the host MATCH is [`check_host_class`]'s job, never the
  /// parser's: the hermetic loader tests and the committed goldens must parse on
  /// EVERY host (this one included).
  ///
  /// # Errors
  /// If `generationHost` is present but not an object, or is missing any of the
  /// four fields as a non-empty string. Every message contains `generationHost`.
  pub fn from_golden(name: &str, v: &serde_json::Value) -> Result<Option<Self>, String> {
    let host = match v.get("generationHost") {
      None | Some(serde_json::Value::Null) => return Ok(None),
      Some(serde_json::Value::Object(map)) => map,
      Some(other) => {
        return Err(format!(
          "{name}: `generationHost` is {other} — expected an object with osBuild / \
           osProductVersion / chip / arch string fields"
        ));
      }
    };
    let field = |key: &str| -> Result<String, String> {
      host
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{name}: `generationHost.{key}` is missing, not a string, or empty"))
    };
    Ok(Some(HostClass {
      os_build: field("osBuild")?,
      os_product_version: field("osProductVersion")?,
      chip: field("chip")?,
      arch: field("arch")?,
    }))
  }
}

impl std::fmt::Display for HostClass {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "macOS {} (build {}), {}, {}",
      self.os_product_version, self.os_build, self.chip, self.arch
    )
  }
}

/// Reads one sysctl value as a trimmed string via `/usr/sbin/sysctl -n`
/// (absolute path — PATH-independent).
///
/// # Panics
/// With a `host-class gate:` message on spawn failure, a non-zero exit, or
/// empty output.
fn sysctl_string(key: &str) -> String {
  let output = std::process::Command::new("/usr/sbin/sysctl")
    .args(["-n", key])
    .output()
    .unwrap_or_else(|e| panic!("host-class gate: cannot spawn /usr/sbin/sysctl for `{key}`: {e}"));
  assert!(
    output.status.success(),
    "host-class gate: `/usr/sbin/sysctl -n {key}` exited {}",
    output.status
  );
  let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
  assert!(
    !value.is_empty(),
    "host-class gate: `/usr/sbin/sysctl -n {key}` produced empty output"
  );
  value
}

/// Verdict of the host-class gate for one golden.
#[derive(Debug, PartialEq, Eq)]
pub enum HostVerdict {
  /// `generationHost` recorded and equal to the running host-class: the tight
  /// parity bounds apply and a failure is cleanly attributable to the port.
  Match,
  /// The golden predates host provenance (no `generationHost`): the tight
  /// bounds still apply (a PASS is sound evidence on any host — a port bug
  /// exactly cancelling host drift under these bounds is not a real risk),
  /// but a FAILURE is ambiguous between a port defect and host-CoreML drift —
  /// append [`legacy_failure_note`] to fidelity failure messages.
  LegacyUnknown,
}

/// THE host-class match predicate + mismatch diagnosis. Pure — no I/O — so the
/// hermetic tests drive it with synthetic host-class values.
///
/// # Errors
/// The full actionable diagnosis for a recorded-but-different host; the caller
/// panics with it BEFORE any CoreML number is produced.
pub fn check_host_class(
  fixture: &str,
  recorded: Option<&HostClass>,
  running: &HostClass,
  regen_script: &str,
) -> Result<HostVerdict, String> {
  match recorded {
    None => Ok(HostVerdict::LegacyUnknown),
    Some(r) if r == running => Ok(HostVerdict::Match),
    Some(r) => Err(format!(
      "{fixture}: committed golden was generated on a DIFFERENT host-class.\n  \
       golden host : {r}\n  this host   : {running}\n\
       CoreML floating point is not contracted portable across macOS builds or chips —\n\
       neither the `CpuOnly` kernels that ship with the OS nor the Neural Engine's fp16\n\
       arithmetic — so the tight parity bound would misattribute host float drift to the\n\
       port. This failure is NOT evidence of a port defect. To test the port on this\n\
       machine, regenerate a same-host oracle and re-run:\n  {regen_script}\n\
       Do NOT widen the parity tolerances instead — the tight bounds are what catch real\n\
       stitching/index-mapping regressions on a matching host."
    )),
  }
}

/// The ambiguity note appended to a fidelity failure when the golden predates
/// host-class provenance (a [`HostVerdict::LegacyUnknown`] golden): the failure
/// cannot be cleanly attributed to the port versus host-CoreML drift, and the
/// fix is a same-host regeneration, never a widened tolerance.
pub fn legacy_failure_note(regen_script: &str) -> String {
  format!(
    "\nNOTE: this golden predates host-class provenance (no `generationHost` field), so this\n\
     failure is AMBIGUOUS between a port defect and host-CoreML float drift (neither the\n\
     `CpuOnly` kernels nor the Neural Engine's fp16 arithmetic are contracted portable across\n\
     macOS builds/chips). Regenerate a same-host oracle via {regen_script} to disambiguate —\n\
     regeneration also stamps `generationHost`. Do NOT widen the tolerance."
  )
}
