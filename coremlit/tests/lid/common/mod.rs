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
