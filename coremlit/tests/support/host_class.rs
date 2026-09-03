//! Host-class provenance for the committed-Swift-golden parity gates — the
//! SINGLE copy, shared by the speaker, vad and whisper suites.
//!
//! CoreML floating point is not contracted to be identical across macOS builds
//! or chip generations (#36): neither the `CpuOnly` kernels, which ship as part
//! of the OS binary set, nor the Neural Engine's fp16 arithmetic, whose last
//! bit can differ between Apple Silicon generations. A parity gate that
//! compares against an external Swift oracle therefore has to know WHICH hosts
//! that oracle ran on, or a host difference gets reported as a port defect.
//!
//! So every golden carries an optional record of the host classes its payload
//! was reproduced on, and the gates enforce their tight bounds against it via
//! [`check_host_class`]:
//!
//! | recorded hosts | verdict | what the gate does |
//! | --- | --- | --- |
//! | none | [`HostVerdict::LegacyUnknown`] | tight bounds still enforced; [`legacy_failure_note`] appended to any failure, because a FAILURE is ambiguous |
//! | one of them equals the running host | [`HostVerdict::Match`] | strict comparison; a failure is cleanly the port's |
//! | none of them equals it | `Err(diagnosis)` | panic BEFORE any CoreML number, listing every recorded class, naming the regeneration command and stating this is not a port defect |
//!
//! # Why a SET of classes, and why that does not soften the gate
//!
//! GitHub's hosted `macos-15` pool is not one image. Through a rolling upgrade
//! some runners answer `kern.osversion` with `24G720` (macOS 15.7.7) and others
//! with `24G830` (15.7.9), and a job lands on whichever it lands on. A golden
//! that names ONE class reds roughly half the jobs for the duration of the
//! rollover while saying nothing at all about the port — the exact
//! misattribution this file exists to prevent, arriving from the other side.
//!
//! Measurement settles it rather than a tolerance: on both of those images the
//! three whisper goldens' tokens, segments and word timestamps came out
//! byte-identical, so the claim the goldens can honestly make is not "this
//! payload came from `24G720`" but "this payload was reproduced on `24G720`
//! AND on `24G830`" — and a run on either is a run on a verified host.
//!
//! A class earns its place in the set only by having reproduced the committed
//! payload exactly; `coremlit/tests/whisper/swift/merge_golden_hosts.sh` is
//! where that is decided, and it REPLACES the whole set the moment the payload
//! moves. So `Match` still means what it always meant — the oracle produced
//! exactly these numbers on a machine of this class — and a host nobody has
//! measured is still refused.
//!
//! Only the whisper goldens carry a set today; the speaker and vad goldens
//! still carry the legacy single `generationHost`, which
//! [`RecordedHost::all_from_golden`] reads as a set of one, so their gates
//! behave exactly as they did.
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
/// counts, and the ORACLE's own version — see [`RecordedHost::source`].
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

/// One entry of a golden's recorded host set: the [`HostClass`] identity that
/// decides reproducibility, plus the oracle provenance observed THERE.
///
/// `source` sits outside [`HostClass`], and so outside the match, on purpose.
/// The oracle's own version label legitimately differs between two images whose
/// numbers agree exactly — the `macos-15` rollover shipped `whisperkit-cli`
/// v1.0.0 on build `24G720` and v1.1.0 on `24G830` while every token, segment
/// and word timestamp stayed identical — so folding it into the identity would
/// refuse a host the payload demonstrably reproduces on. Recording it per host
/// keeps the fact rather than averaging it away: the set says which oracle
/// build reproduced the payload on which image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedHost {
  /// The four-field identity compared by [`check_host_class`].
  pub class: HostClass,
  /// The `source` label the oracle reported on this host, when the golden
  /// records one. `None` for a legacy golden that carries only a document-level
  /// `source`, or none at all.
  pub source: Option<String>,
}

impl From<HostClass> for RecordedHost {
  fn from(class: HostClass) -> Self {
    RecordedHost {
      class,
      source: None,
    }
  }
}

impl std::fmt::Display for RecordedHost {
  /// The [`HostClass`] rendering, with the oracle observed there appended when
  /// the golden records one. The class rendering is a PREFIX of this one, so a
  /// diagnosis that names a recorded host still contains that host's plain
  /// class rendering.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.class)?;
    match &self.source {
      Some(source) => write!(f, " — oracle: {source}"),
      None => Ok(()),
    }
  }
}

impl RecordedHost {
  /// Reads the OPTIONAL recorded host set from a golden's JSON.
  ///
  /// Three accepted shapes, in precedence order:
  ///
  /// - `generationHosts`: a non-empty array of host objects, each of which may
  ///   carry its own `source`. This is what
  ///   `coremlit/tests/whisper/swift/merge_golden_hosts.sh` writes.
  /// - `generationHost`: the legacy SINGLE host object, read as a one-element
  ///   set so every golden stamped before the set existed keeps working
  ///   unchanged. Its `source`, if any, lives at the document level and is not
  ///   pulled in here — [`RecordedHost::source`] stays `None`.
  /// - neither (absent or `null`): an empty set, i.e. a legacy golden that
  ///   predates host provenance entirely.
  ///
  /// Carrying BOTH keys is an error: one golden gets one record of where it was
  /// produced, and a stale single-host key silently disagreeing with the set is
  /// exactly the ambiguity this whole file exists to remove.
  ///
  /// FORM only — the host MATCH is [`check_host_class`]'s job, never the
  /// parser's: the hermetic loader tests and the committed goldens must parse on
  /// EVERY host (this one included).
  ///
  /// # Errors
  /// If either key is present but malformed: not an object (or, for the set, not
  /// a non-empty array of objects), missing any of the four host fields as a
  /// non-empty string, or carrying a `source` that is not a non-empty string.
  /// Every message names the offending key path, so it always contains
  /// `generationHost`.
  pub fn all_from_golden(name: &str, v: &serde_json::Value) -> Result<Vec<Self>, String> {
    let present = |key: &str| v.get(key).filter(|x| !x.is_null());
    match (present("generationHosts"), present("generationHost")) {
      (Some(_), Some(_)) => Err(format!(
        "{name}: carries BOTH `generationHosts` and `generationHost`. A golden records where \
         it was produced exactly once — keep the `generationHosts` set and drop the legacy \
         single-host key, or the two can disagree with nothing to say which is true"
      )),
      (Some(set), None) => {
        let entries = set.as_array().ok_or_else(|| {
          format!(
            "{name}: `generationHosts` is {set} — expected a non-empty array of objects with \
             osBuild / osProductVersion / chip / arch string fields"
          )
        })?;
        if entries.is_empty() {
          return Err(format!(
            "{name}: `generationHosts` is empty. A golden either records the host classes its \
             payload was reproduced on, or omits the key entirely (legacy, unattributable) — \
             an empty set claims a payload nothing reproduced"
          ));
        }
        entries
          .iter()
          .enumerate()
          .map(|(i, entry)| Self::from_object(name, &format!("generationHosts[{i}]"), entry))
          .collect()
      }
      (None, Some(one)) => Ok(vec![Self::from_object(name, "generationHost", one)?]),
      (None, None) => Ok(Vec::new()),
    }
  }

  /// Parses ONE host object at the JSON path `label` inside golden `name`. The
  /// four identity fields are required and must be non-empty strings; `source`
  /// is optional but must be a non-empty string when present; any other key is
  /// tolerated and ignored.
  fn from_object(name: &str, label: &str, v: &serde_json::Value) -> Result<Self, String> {
    let host = match v {
      serde_json::Value::Object(map) => map,
      other => {
        return Err(format!(
          "{name}: `{label}` is {other} — expected an object with osBuild / osProductVersion / \
           chip / arch string fields"
        ));
      }
    };
    let field = |key: &str| -> Result<String, String> {
      host
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{name}: `{label}.{key}` is missing, not a string, or empty"))
    };
    let source = match host.get("source") {
      None | Some(serde_json::Value::Null) => None,
      Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s.clone()),
      Some(other) => {
        return Err(format!(
          "{name}: `{label}.source` is {other} — expected a non-empty string naming the oracle \
           that produced this payload on this host"
        ));
      }
    };
    Ok(RecordedHost {
      class: HostClass {
        os_build: field("osBuild")?,
        os_product_version: field("osProductVersion")?,
        chip: field("chip")?,
        arch: field("arch")?,
      },
      source,
    })
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
  /// The running host-class is ONE OF the classes the golden records: the tight
  /// parity bounds apply and a failure is cleanly attributable to the port.
  /// Every recorded class reproduced this exact payload, so which one the
  /// machine happens to be does not change what a divergence means.
  Match,
  /// The golden predates host provenance (no recorded class at all): the tight
  /// bounds still apply (a PASS is sound evidence on any host — a port bug
  /// exactly cancelling host drift under these bounds is not a real risk),
  /// but a FAILURE is ambiguous between a port defect and host-CoreML drift —
  /// append [`legacy_failure_note`] to fidelity failure messages.
  LegacyUnknown,
}

/// THE host-class match predicate + mismatch diagnosis. Pure — no I/O — so the
/// hermetic tests drive it with synthetic host-class values.
///
/// `recorded` is the golden's whole set, as [`RecordedHost::all_from_golden`]
/// read it: empty for a legacy golden, one element for a single-host stamp, more
/// for a payload measured identical on several classes. ANY member matching the
/// running host is a [`HostVerdict::Match`] — every member reproduced this same
/// payload, which is the only reason it is in the set.
///
/// # Errors
/// The full actionable diagnosis when the running host is not among the recorded
/// classes; the caller panics with it BEFORE any CoreML number is produced.
pub fn check_host_class(
  fixture: &str,
  recorded: &[RecordedHost],
  running: &HostClass,
  regen_script: &str,
) -> Result<HostVerdict, String> {
  if recorded.is_empty() {
    return Ok(HostVerdict::LegacyUnknown);
  }
  if recorded.iter().any(|host| &host.class == running) {
    return Ok(HostVerdict::Match);
  }
  let listed = recorded
    .iter()
    .map(|host| format!("\n    - {host}"))
    .collect::<String>();
  let count = recorded.len();
  Err(format!(
    "{fixture}: committed golden has NOT been verified on this host-class — this machine is a \
     DIFFERENT host-class from every one the golden was reproduced on.\n  \
     golden hosts ({count} recorded):{listed}\n  \
     this host   : {running}\n\
     CoreML floating point is not contracted portable across macOS builds or chips —\n\
     neither the `CpuOnly` kernels that ship with the OS nor the Neural Engine's fp16\n\
     arithmetic — so the tight parity bound would misattribute host float drift to the\n\
     port. This failure is NOT evidence of a port defect. To test the port on this\n\
     machine, regenerate a same-host oracle and re-run:\n  {regen_script}\n\
     A regeneration whose payload matches the committed one ADDS this machine's class to the\n\
     set above rather than replacing it, so a host that reproduces the goldens joins them.\n\
     Do NOT widen the parity tolerances instead — the tight bounds are what catch real\n\
     stitching/index-mapping regressions on a matching host."
  ))
}

/// The ambiguity note appended to a fidelity failure when the golden predates
/// host-class provenance (a [`HostVerdict::LegacyUnknown`] golden): the failure
/// cannot be cleanly attributed to the port versus host-CoreML drift, and the
/// fix is a same-host regeneration, never a widened tolerance.
pub fn legacy_failure_note(regen_script: &str) -> String {
  format!(
    "\nNOTE: this golden predates host-class provenance (no `generationHosts` field), so this\n\
     failure is AMBIGUOUS between a port defect and host-CoreML float drift (neither the\n\
     `CpuOnly` kernels nor the Neural Engine's fp16 arithmetic are contracted portable across\n\
     macOS builds/chips). Regenerate a same-host oracle via {regen_script} to disambiguate —\n\
     regeneration also stamps the host class it ran on. Do NOT widen the tolerance."
  )
}
