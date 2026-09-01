//! Shared helpers for the language-identification tests.
//!
//! Two data sources, kept distinct:
//!
//! - **Committed fixtures** (`tests/lid/fixtures/`) — the 16 kHz mono WAV the
//!   end-to-end anchor runs on, read hermetically; no model, no network.
//! - **CoreML artifact** ([`BUNDLE_NAME`]) — a gitignored dev-time download
//!   staged under `Models/lid/`, overridable via `LID_TEST_MODELS`.
//!   Model-gated tests are `#[ignore]` by default and run only when the owner
//!   stages the artifact.

// The workspace-root anchor every `models_dir()` below resolves against, and
// the sibling-checkout anchor the oracle gates read. FOUND by searching upward
// for the `[workspace]` manifest, never counted in `../` hops — see its module
// doc for why a count is the wrong shape here. Re-exported so the binaries
// that pull this `common` in share the one resolver.
#[path = "../../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;
#[allow(unused_imports)]
pub use workspace_root::{checkout_parent, models_root, workspace_root};

use std::path::{Path, PathBuf};

/// Hugging Face repo the artifact comes from (Apache-2.0).
#[allow(dead_code)]
pub const HF_REPO: &str = "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML";

/// Pinned artifact-repo revision the SHA-256 table below was taken from.
#[allow(dead_code)]
pub const HF_REVISION: &str = "2aa4d715a79e410d5f9aa32bd7a4fc9225bf9eb0";

/// Upstream PyTorch model the artifact is an export of.
#[allow(dead_code)]
pub const SOURCE_MODEL: &str = "speechbrain/lang-id-voxlingua107-ecapa";

/// Upstream source revision recorded in the graph's own creator metadata.
#[allow(dead_code)]
pub const SOURCE_REVISION: &str = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9";

/// The compiled bundle's directory name inside the models root. The library
/// never spells this — [`coremlit::audio::lid::Identifier::from_file`] takes
/// whatever path the caller staged — so the artifact's own name lives here,
/// with the rest of its provenance.
#[allow(dead_code)]
pub const BUNDLE_NAME: &str = "SpeechBrainECAPAVoxLingua107.mlmodelc";

/// Exact per-file SHA-256 of the staged `.mlmodelc` bundle. The set is exact:
/// a missing OR an added file reds, not just a changed one.
#[allow(dead_code)]
pub const ARTIFACT_SHA256: &[(&str, &str)] = &[
  (
    "analytics/coremldata.bin",
    "9e092f41490e5313e38cb6bcc10ce9fe1ed0bdf5ce33c7c3143dfdc47141c8b7",
  ),
  (
    "coremldata.bin",
    "546fd351966c937770a8eb64d86764cd87a65d1f58c74627bb47584eeb20413e",
  ),
  (
    "model.mil",
    "32a6a3fcb77aae3b32123514e83e0c16e4427e986dc2d23cfdde2a4dba1b81c2",
  ),
  (
    "weights/weight.bin",
    "81fbb61f6706c50e924a2ee2a4fc04e6408276df948117a1c6ac7675c23aac67",
  ),
];

/// Directory holding the downloaded CoreML artifact tree.
///
/// Overridable via `LID_TEST_MODELS`; otherwise `<workspace>/Models/lid` —
/// gitignored, fetched dev-time (the `CED_TEST_MODELS` / `EMBEDKIT_TEST_MODELS`
/// convention).
#[allow(dead_code)]
pub fn models_dir() -> PathBuf {
  std::env::var_os("LID_TEST_MODELS")
    .map_or_else(|| workspace_root::models_root().join("lid"), PathBuf::from)
}

/// Path to the compiled graph under [`models_dir`].
#[allow(dead_code)]
pub fn model_path() -> PathBuf {
  models_dir().join(BUNDLE_NAME)
}

/// Absolute path to a committed fixture under `coremlit/tests/lid/fixtures`.
#[allow(dead_code)]
pub fn fixture_path(relative: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("lid")
    .join("fixtures")
    .join(relative)
}

/// Lowercase-hex SHA-256 of a byte slice.
#[allow(dead_code)]
pub fn sha256_hex(bytes: &[u8]) -> String {
  use core::fmt::Write;

  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .fold(String::new(), |mut acc, b| {
      let _ = write!(acc, "{b:02x}");
      acc
    })
}

/// Lowercase-hex SHA-256 of a file's contents.
#[allow(dead_code)]
pub fn sha256_file(path: &Path) -> String {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  sha256_hex(&bytes)
}

/// Read a committed WAV into normalized mono f32, asserting its 16 kHz mono
/// header first so a mis-encoded fixture — which would invalidate the stated
/// provenance yet still decode to numbers — fails loudly instead of quietly
/// feeding the wrong geometry into a gate.
#[allow(dead_code)]
pub fn read_wav_16k_mono(path: &Path) -> Vec<f32> {
  let mut reader =
    hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
  let spec = reader.spec();
  assert_eq!(
    spec.sample_rate,
    16_000,
    "{}: must be 16 kHz",
    path.display()
  );
  assert_eq!(spec.channels, 1, "{}: must be mono", path.display());
  match spec.sample_format {
    hound::SampleFormat::Int => {
      let scale = 1.0 / (1_i64 << (spec.bits_per_sample - 1)) as f32;
      reader
        .samples::<i32>()
        .map(|s| s.expect("decode sample") as f32 * scale)
        .collect()
    }
    hound::SampleFormat::Float => reader
      .samples::<f32>()
      .map(|s| s.expect("decode sample"))
      .collect(),
  }
}

/// Recursively collect the forward-slash relative path of every FILE under
/// `dir`. OS-generated sidecars (AppleDouble `._*`, `.DS_Store`) are skipped:
/// CoreML's loader never reads them, so excluding them cannot mask a functional
/// change, while keeping them would false-fail the exact-set gate.
#[allow(dead_code)]
pub fn collect_files_rel(dir: &Path, prefix: &str, out: &mut Vec<String>) {
  let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"));
  for entry in entries {
    let entry = entry.unwrap_or_else(|e| panic!("dir entry under {dir:?}: {e}"));
    let name = entry.file_name().to_string_lossy().into_owned();
    if name.starts_with("._") || name == ".DS_Store" {
      continue;
    }
    let rel = if prefix.is_empty() {
      name
    } else {
      format!("{prefix}/{name}")
    };
    let file_type = entry
      .file_type()
      .unwrap_or_else(|e| panic!("file_type {:?}: {e}", entry.path()));
    if file_type.is_dir() {
      collect_files_rel(&entry.path(), &rel, out);
    } else {
      out.push(rel);
    }
  }
}

/// Assert the bundle at `dir` matches the EXACT pinned manifest: the discovered
/// file set must EQUAL the pinned key set (so a missing OR an added artifact
/// both red), and each file's SHA-256 must equal its pinned value.
#[allow(dead_code)]
pub fn assert_exact_sha_manifest(dir: &Path, cases: &[(&str, &str)]) {
  use std::collections::BTreeSet;

  let mut found = Vec::new();
  collect_files_rel(dir, "", &mut found);
  let on_disk: BTreeSet<String> = found.into_iter().collect();
  let pinned: BTreeSet<String> = cases.iter().map(|(rel, _)| (*rel).to_owned()).collect();

  if on_disk != pinned {
    let missing: Vec<&String> = pinned.difference(&on_disk).collect();
    let extra: Vec<&String> = on_disk.difference(&pinned).collect();
    panic!(
      "artifact manifest mismatch under {dir:?}:\n  \
       missing (pinned but not on disk): {missing:?}\n  \
       extra (on disk but not pinned): {extra:?}"
    );
  }

  for (relative, expected) in cases {
    assert_eq!(
      &sha256_file(&dir.join(relative)),
      expected,
      "sha256 drift on artifact {relative} under {dir:?}"
    );
  }
}

// ── Host capability: predicting AT the graph's own default shape ────────────

/// Mel frames in the shape the shipped artifact names as its own default.
///
/// `model.mil`'s `FlexibleShapeInformation` carries both halves:
/// `DefaultShapes {"mel_features", [1, 301, 60]}` beside
/// `RangeDims [[1, 1], [10, 3001], [60, 60]]`. So 301 is not just "a short
/// window" — it is the ONE length CoreML specializes the graph for at load
/// time, and it is reachable from ordinary use: a caller who asks
/// [`coremlit::audio::lid::WindowPlan`] for 3 s windows lands exactly on it.
#[allow(dead_code)]
pub const GRAPH_DEFAULT_SHAPE_FRAMES: usize = 301;

/// 16 kHz samples that produce [`GRAPH_DEFAULT_SHAPE_FRAMES`] mel frames.
#[allow(dead_code)]
pub const GRAPH_DEFAULT_SHAPE_SAMPLES: usize = 48_000;

/// `Some(reason)` when this host's CoreML cannot predict at the shape above.
///
/// # Why this probe exists
///
/// On the GitHub `macos-15` runner, a prediction at EXACTLY 301 frames under
/// the door's default placement comes back as
/// `Prediction(Native(.. "Unable to compute the prediction using ML Program"))`,
/// deterministically, while 101, 900, 1 001 and 1 300 frames all answer
/// normally on the same loaded model. The refusal is not about length — it is
/// non-monotone in length — and it is not this crate's doing:
///
///   - the failing call is a bare
///     [`Identifier::log_probabilities`](coremlit::audio::lid::Identifier::log_probabilities)
///     on a sub-slice, with no window plan anywhere in the path;
///   - `model_io`'s `runtime_accepts_exactly_the_pinned_frame_range` feeds the
///     runtime a `[1, 301, 60]` tensor on that same runner and it is ACCEPTED,
///     so the shape is supported and the tensor this crate builds is valid;
///   - the identical call on the identical artifact bytes answers on
///     macOS 26.5 / M1 Max with a mass deviation of 6.8e-8.
///
/// That leaves the host's CoreML refusing its own default-shape specialization,
/// which no amount of Rust can fix and which a gate must therefore state rather
/// than absorb.
///
/// # Why it cannot hide a real break
///
/// The probe runs the DEFAULT-length window first and requires it to answer. A
/// model that is actually broken here fails that call, and the probe panics
/// instead of excusing anything. Only a `Prediction` refusal at the default
/// shape, on a host that is otherwise predicting fine, is reported as a host
/// limitation; every other error is re-raised.
#[allow(dead_code)]
fn default_shape_refusal() -> Option<&'static String> {
  use std::sync::OnceLock;

  use coremlit::audio::lid::{DEFAULT_WINDOW_SAMPLES, Error, Identifier};

  static PROBE: OnceLock<Option<String>> = OnceLock::new();
  PROBE
    .get_or_init(|| {
      // The two constants must describe the SAME input, or the refusal message
      // below names a length the probe did not actually run.
      assert_eq!(
        coremlit::audio::lid::frame_count(GRAPH_DEFAULT_SHAPE_SAMPLES),
        GRAPH_DEFAULT_SHAPE_FRAMES,
        "the probe's sample count and frame count disagree"
      );
      let identifier = Identifier::from_file(model_path())
        .unwrap_or_else(|e| panic!("host-capability probe: load identifier: {e}"));
      // The real fixture, so the probe runs the very call the gates run.
      let samples = read_wav_16k_mono(&fixture_path("audio/udhr_th_16k.wav"));

      // A host that cannot answer the SHIPPED default window is broken, not
      // limited: fail here rather than excusing anything below.
      identifier
        .log_probabilities(&samples[..DEFAULT_WINDOW_SAMPLES as usize])
        .unwrap_or_else(|e| {
          panic!(
            "host-capability probe: this host cannot predict at the door's own \
             default window ({DEFAULT_WINDOW_SAMPLES} samples): {e} — that is a \
             broken model or a broken host, not the narrow default-shape \
             refusal this probe excuses, so it must red"
          )
        });

      match identifier.log_probabilities(&samples[..GRAPH_DEFAULT_SHAPE_SAMPLES]) {
        Ok(_) => None,
        Err(e @ Error::Prediction(_)) => Some(format!(
          "this host's CoreML refuses to predict at the graph's own \
           DefaultShapes [1, {GRAPH_DEFAULT_SHAPE_FRAMES}, 60] \
           ({GRAPH_DEFAULT_SHAPE_SAMPLES} samples, 3 s) under the door's \
           default placement, while answering normally at the default \
           {DEFAULT_WINDOW_SAMPLES}-sample window: {e}"
        )),
        // Anything else is a real defect and must not be excused.
        Err(e) => panic!(
          "host-capability probe: predicting at the graph's default shape failed \
           with {e} — only a CoreML `Prediction` refusal is a host limitation, \
           so this reds"
        ),
      }
    })
    .as_ref()
}

/// Reports and returns `true` when this gate cannot run on this host.
///
/// The line goes to the INHERITED stderr descriptor rather than through
/// `println!`, for the reason `model_gate_report` spells out next door: libtest
/// discards a passing test's output unless the reader remembered `--nocapture`,
/// and a skip nobody can see is the silent pass this is meant to prevent.
#[allow(dead_code)]
pub fn skipped_for_the_default_shape_refusal(gate: &str) -> bool {
  use std::io::Write;

  let Some(reason) = default_shape_refusal() else {
    return false;
  };
  let line = format!("model-gates | SKIPPED {gate}: {reason}\n");
  // SAFETY: fd 2 is open for the whole life of the process (libtest redirects
  // the Rust-level handles, never the descriptor), it is only written to here,
  // and `ManuallyDrop` keeps the `File` from closing a descriptor it does not
  // own.
  let mut fd2 = std::mem::ManuallyDrop::new(unsafe {
    <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(2)
  });
  let _ = fd2.write_all(line.as_bytes());
  true
}

// ── Model-gate visibility (#61) ─────────────────────────────────────────────
//
// NOT `#[ignore]`d, deliberately: an ignored-only run never selects it, so the
// anti-vacuum counts are unchanged, while a plain modelless run gains a line
// saying how many gates were skipped and where their models would have been.
#[path = "../../support/model_gate_report.rs"]
mod model_gate_report;

/// Reports how many of this binary's tests are `#[ignore]`d model gates that did
/// not run, and whether the models root they read is on disk.
#[test]
fn model_gate_report() {
  model_gate_report::report(&[("LID_TEST_MODELS", models_dir())]);
}
