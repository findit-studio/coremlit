// Not every integration-test binary that includes `mod common;` uses every
// helper below (`model_io.rs` uses the model-path + sha helpers; the parity
// and state suites use the audio/fnv helpers). Each test file is its own
// crate, so an unused helper is dead code *in that crate* — allow it here so
// the shared module compiles clean under the workspace's `-D warnings` gate.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The compiled VAD artifact's directory name within [`models_dir`] — the
/// FluidInference `silero-vad-unified-256ms-v6.2.1` `.mlmodelc` (design spec
/// §5; the v6.2.1 artifact ships pre-compiled, so this loads directly with no
/// `coremlcompiler` step). Its HF revision + per-file SHA-256 are pinned in
/// `tests/model_io.rs` — the alignkit/speakerkit convention for adopted models,
/// NOT `MODELS_LOCK` (which a whisperkit hermetic gate holds to exactly the two
/// CI-downloaded whisper tables).
pub const ARTIFACT: &str = "silero-vad-unified-256ms-v6.2.1.mlmodelc";

/// Directory containing the downloaded vadkit model artifact.
///
/// Overridable via `VADKIT_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/vadkit` — gitignored, fetched dev-time (mirrors
/// `speakerkit`'s `SPEAKERKIT_TEST_MODELS`/`Models/speakerkit` and
/// `alignkit`'s `ALIGNKIT_TEST_MODELS`/`Models/alignkit`, one directory level
/// down for this crate's own model set). Fetch with:
///
/// ```text
/// hf download FluidInference/silero-vad-coreml \
///   --include "silero-vad-unified-256ms-v6.2.1*" \
///   --revision b419383c55c110e2c9271fa6ee0ea83d03c70d96 \
///   --local-dir Models/vadkit
/// ```
pub fn models_dir() -> PathBuf {
  std::env::var_os("VADKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("vadkit")
    },
    PathBuf::from,
  )
}

/// Path to the compiled VAD `.mlmodelc` artifact.
pub fn model_path() -> PathBuf {
  models_dir().join(ARTIFACT)
}

/// A committed parity fixture: a short 16 kHz mono clip borrowed by relative
/// path from the `speakerkit` crate's fixtures (dia's parity corpus), plus its
/// provenance. Borrowed rather than re-committed — both crates live in this
/// workspace and move together, exactly as `alignkit` borrows whisperkit's
/// `ted_60.wav`.
pub struct Fixture {
  /// Basename (no extension) of the WAV and its committed Swift golden.
  pub name: &'static str,
  /// Path within the `speakerkit` crate this clip is borrowed from.
  pub source: &'static str,
  /// SHA-256 of the borrowed WAV — pins the exact audio a swap/re-encode
  /// would silently change out from under the cross-crate relative path.
  pub sha256: &'static str,
  /// Why this clip is in the set (coverage rationale).
  pub note: &'static str,
}

/// The Swift-trace parity fixture set (spec §6 model-layer gate). Two real-
/// speech clips from dia's parity corpus: `02_pyannote_sample` is pyannote's
/// canonical multi-speaker demo (30.0 s → 118 chunks); `07_yuhewei_dongbei_
/// english` is a second real conversational clip (25.26 s → 99 chunks) that
/// exercises the short-final-chunk padding path. Together: 217 chunks across
/// 2 clips (the gate requires ≥ 40 chunks over ≥ 2 clips). Speaker counts are
/// deliberately NOT claimed here beyond "multi-speaker demo" — the fixture
/// names in dia's corpus do not reliably encode speaker counts, and VAD only
/// needs real speech with speech/non-speech transitions.
pub const FIXTURES: &[Fixture] = &[
  Fixture {
    name: "02_pyannote_sample",
    source: "crates/coremlit/tests/speaker/fixtures/audio/02_pyannote_sample.wav",
    sha256: "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e",
    note: "pyannote's canonical 30.0 s multi-speaker demo → 118 full 256 ms chunks",
  },
  Fixture {
    name: "07_yuhewei_dongbei_english",
    source: "crates/coremlit/tests/speaker/fixtures/audio/07_yuhewei_dongbei_english.wav",
    sha256: "096890ba8ffbaf10ca770c5373bf6c6664777f9421595c2cb7780af8cb2e46ff",
    note: "25.26 s clip → 98 full chunks + 1 short final chunk (exercises repeat-last padding)",
  },
];

/// Absolute path to a borrowed fixture WAV by basename.
pub fn fixture_wav_path(name: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/speaker/fixtures/audio")
    .join(format!("{name}.wav"))
}

/// Directory holding this crate's committed Swift-trace goldens.
pub fn golden_swift_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vad/fixtures/golden_swift")
}

/// Loads a 16 kHz mono WAV as `f32` samples — the single source of truth both
/// the Swift dumper (`tests/vad/swift/.../DumpVadTraces.swift`'s `readPcm16Mono16k`)
/// and the Rust gate feed their models, so the two sides are input-identical by
/// construction; the [`fnv1a_f32`] recorded in each golden re-proves it at
/// replay time (the alignkit/speakerkit Gate-1 lesson: prove the inputs match).
///
/// 16-bit PCM is scaled by `1 / 32768`; float WAVs pass through.
///
/// # Panics
/// If the file is missing, not 16 kHz mono, or an unsupported bit depth.
pub fn load_wav_16k_mono(path: &Path) -> Vec<f32> {
  let mut reader =
    hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 16_000, "{}: not 16 kHz", path.display());
  assert_eq!(spec.channels, 1, "{}: not mono", path.display());
  match spec.sample_format {
    hound::SampleFormat::Int => {
      assert_eq!(
        spec.bits_per_sample,
        16,
        "{}: only 16-bit int PCM supported",
        path.display()
      );
      reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("read i16 sample")) / 32_768.0)
        .collect()
    }
    hound::SampleFormat::Float => reader
      .samples::<f32>()
      .map(|s| s.expect("read f32 sample"))
      .collect(),
  }
}

/// FNV-1a-64 over the little-endian bytes of `samples` — byte-for-byte the
/// same construction as `coremlit::audio::speaker::tests::common::fnv1a_f32` and the Swift
/// dumper's `fnv1aHex`, so the golden's recorded input hash proves both sides
/// fed the model element-identical audio.
pub fn fnv1a_f32(samples: &[f32]) -> u64 {
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  for &s in samples {
    for b in s.to_le_bytes() {
      h ^= u64::from(b);
      h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
  }
  h
}

/// Lowercase 16-hex-digit rendering of a [`fnv1a_f32`] hash for JSON storage.
pub fn fnv_hex(h: u64) -> String {
  format!("{h:016x}")
}

/// Lowercase-hex SHA-256 digest of a file's contents — the provenance/
/// integrity pin over the downloaded model artifacts (`tests/model_io.rs`).
pub fn sha256_hex(path: &Path) -> String {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

// ── Host-class provenance for the committed-Swift-golden parity gate ────────
//
// CoreML `CpuOnly` kernels ship with the OS and are not contracted to produce
// identical floats across macOS builds or chip generations (#36), so the parity
// gate stamps each golden with the host-class it was generated on and enforces
// its tight bound only against a matching host (see `check_host_class`). This
// block is duplicated verbatim in the vad and speaker `common/mod.rs` — the
// repo's self-contained-`common` convention, the same one that already
// duplicates `fnv1a_f32`/`load_wav_16k_mono`; each suite's hermetic tests drive
// its own copy, which is what guards the two copies against drift.

/// The host-class identity that determines CoreML `CpuOnly` float
/// reproducibility: macOS build (the OS binary set every CPU kernel ships
/// in), chip (kernel dispatch varies by microarchitecture), process arch
/// (Rosetta), plus the human-readable product version (fully determined by
/// the build; carried for diagnostics). Deliberately NOT included: Xcode /
/// Swift toolchain (compiles the dumper, not the runtime kernels — inputs
/// are FNV-pinned and model bytes SHA-pinned), `hw.model` (same chip + build
/// ⇒ same CPU floats), RAM/core counts.
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
       CoreML CpuOnly floating point is not contracted portable across macOS builds or chips,\n\
       so the tight parity bound would misattribute host float drift to the port. This failure\n\
       is NOT evidence of a port defect. To test the port on this machine, regenerate a\n\
       same-host oracle and re-run:\n  {regen_script}\n\
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
     failure is AMBIGUOUS between a port defect and host-CoreML CpuOnly float drift (CpuOnly\n\
     floats are not contracted portable across macOS builds/chips). Regenerate a same-host\n\
     oracle via {regen_script} to disambiguate — regeneration also stamps `generationHost`.\n\
     Do NOT widen the tolerance."
  )
}
