//! Shared helpers for the CED classifier tests, parameterized over the four
//! [`CedModel`] sizes (tiny/mini/small/base).
//!
//! Two data sources, kept distinct:
//!
//! - **Committed fixtures** (`tests/ced/fixtures/`) — staged by Wave B: the
//!   16 kHz mono WAV clips are committed ONCE and shared across sizes
//!   (`fixtures/mel/`, `fixtures/goldens/clips/`); only the per-size logits
//!   differ, as `fixtures/goldens/<size>/corpus.json` — a [`GoldenCorpus`]
//!   whose [`OracleProvenance`] header names the `mispeech/ced-<size>` ONNX
//!   fp32 CPU oracle the logits came from (generated owner-side — ort never
//!   enters this repo, not even dev). [`load_golden_corpus`] cross-checks that
//!   header against [`CedModel::hf_repo`], so a cross-size golden mix-up (four
//!   I/O-identical models make it otherwise invisible) fails closed. Read
//!   hermetically; no model, no network.
//! - **CoreML artifact** (`ced_<size>.mlmodelc`) — a gitignored dev-time
//!   download staged under `Models/ced/ced-<size>/` (the `Models/ced` family
//!   root is overridable via `CED_TEST_MODELS`). Model-gated tests are
//!   `#[ignore]` by default and run per size only when the owner stages that
//!   size's Wave-B conversion.

use std::path::{Path, PathBuf};

use coremlit::audio::ced::CedModel;

// Per-size placeholders for the HF revision (commit SHA) each CED artifact and
// its SHA-256 pins are recorded at — filled independently by the conversion
// runbook's upload step (Wave C), one size at a time.
const TINY_REVISION: &str = "<pending conversion upload (Wave C)>";
const MINI_REVISION: &str = "<pending conversion upload (Wave C)>";
const SMALL_REVISION: &str = "<pending conversion upload (Wave C)>";
const BASE_REVISION: &str = "<pending conversion upload (Wave C)>";

/// The HF revision the given size's artifact and per-file SHA-256 pins are
/// recorded at (a placeholder until that size's Wave-C upload). Totality is
/// compiler-enforced by the closed [`CedModel`] enum.
#[allow(dead_code)]
pub const fn ced_revision(model: CedModel) -> &'static str {
  match model {
    CedModel::Tiny => TINY_REVISION,
    CedModel::Mini => MINI_REVISION,
    CedModel::Small => SMALL_REVISION,
    CedModel::Base => BASE_REVISION,
  }
}

/// Directory containing the downloaded CED CoreML artifact tree.
///
/// Overridable via `CED_TEST_MODELS`; otherwise `<workspace>/Models/ced` —
/// gitignored, fetched dev-time (mirrors the `EMBEDKIT_TEST_MODELS` /
/// `SIGLIP_TEST_MODELS` conventions).
#[allow(dead_code)]
pub fn models_dir() -> PathBuf {
  std::env::var_os("CED_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("ced")
    },
    PathBuf::from,
  )
}

/// The given size's bundle directory (`.../ced-<size>`) under the family root,
/// holding the `.mlmodelc` and `CHECKSUMS.sha256`.
#[allow(dead_code)]
pub fn model_root(model: CedModel) -> PathBuf {
  models_dir().join(model.dir_name())
}

/// Path to the given size's compiled mel→logits graph
/// (`.../ced-<size>/ced_<size>.mlmodelc`) — [`CedModel::mlmodelc_path`] rooted
/// at [`models_dir`], so Tiny resolves to the byte-identical Wave-A path.
#[allow(dead_code)]
pub fn model_path(model: CedModel) -> PathBuf {
  model.mlmodelc_path(models_dir())
}

/// Absolute path to a committed fixture under `crates/coremlit/tests/ced/fixtures`.
#[allow(dead_code)]
pub fn fixture_path(relative: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("ced")
    .join("fixtures")
    .join(relative)
}

/// One committed golden clip: a 16 kHz mono WAV fixture and the CED ONNX fp32
/// CPU oracle's `[527]` pre-sigmoid logits for it. The corpus must include at
/// least one sub-window clip (tail-padding semantics) and one multi-window
/// clip (aggregation e2e) — spec §7.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
pub struct GoldenClip {
  /// Stable per-entry id.
  pub id: String,
  /// Corpus-relative WAV path: `../../mel/<file>` for a clip shared via
  /// `fixtures/mel/`, or `../clips/<file>` for one shared via
  /// `fixtures/goldens/clips/`. 16 kHz mono.
  pub file: String,
  /// Decoded sample count (a cheap decode cross-check).
  pub n_samples: usize,
  /// The oracle's `[527]` fp32 PRE-sigmoid logits.
  pub logits: Vec<f32>,
}

/// Provenance header of a per-size [`GoldenCorpus`]: the exact ONNX fp32 CPU
/// oracle its logits were generated from. [`load_golden_corpus`] asserts
/// [`Self::repo`] equals the size's [`CedModel::hf_repo`] — the fail-closed
/// guard against a cross-size golden mix-up, which four I/O-identical models
/// make otherwise undetectable until parity mysteriously fails (precedent:
/// vad's goldens embed and assert their generator revision).
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
pub struct OracleProvenance {
  /// The oracle's Hugging Face repo id (must equal `model.hf_repo()`).
  pub repo: String,
  /// The oracle's pinned HF revision (commit SHA).
  pub revision: String,
  /// The oracle file within the repo (e.g. `model.onnx`).
  pub file: String,
  /// Lowercase-hex SHA-256 of the oracle file.
  pub sha256: String,
}

/// A committed per-size golden corpus: its [`OracleProvenance`] header plus the
/// oracle's pre-sigmoid logits for each shared clip.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
pub struct GoldenCorpus {
  /// Which `mispeech/ced-<size>` ONNX oracle produced `clips`' logits.
  pub oracle: OracleProvenance,
  /// The golden clips (≥ 1 sub-window + ≥ 1 full-window — spec §7).
  pub clips: Vec<GoldenClip>,
}

/// Deserialize a [`GoldenCorpus`] from raw JSON bytes and FAIL CLOSED unless its
/// oracle header matches `model` — the hermetic core of [`load_golden_corpus`],
/// factored out so the cross-size guard is testable without the filesystem.
///
/// # Panics
/// If `bytes` is not a valid [`GoldenCorpus`], or if `oracle.repo` is not
/// `model.`[`hf_repo`](CedModel::hf_repo)`()`.
#[allow(dead_code)]
pub fn parse_golden_corpus(bytes: &[u8], model: CedModel) -> GoldenCorpus {
  let corpus: GoldenCorpus =
    serde_json::from_slice(bytes).unwrap_or_else(|e| panic!("parse golden corpus: {e}"));
  assert_eq!(
    corpus.oracle.repo,
    model.hf_repo(),
    "golden corpus oracle repo `{}` does not match {} (`{}`) — cross-size mix-up",
    corpus.oracle.repo,
    model,
    model.hf_repo()
  );
  corpus
}

/// Load the given size's committed golden corpus
/// (`fixtures/goldens/<size>/corpus.json`) and cross-check its oracle header
/// against [`CedModel::hf_repo`]. Hermetic — reads the in-tree fixture, never
/// `Models/`. (Staged per size by Wave B; until then the fixture is absent and
/// only the `#[ignore]`d gates that call this are affected.)
#[allow(dead_code)]
pub fn load_golden_corpus(model: CedModel) -> GoldenCorpus {
  let path = fixture_path(&format!("goldens/{}/corpus.json", model.as_str()));
  let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  parse_golden_corpus(&bytes, model)
}

/// Read a committed WAV into normalized mono f32, asserting its 16 kHz mono
/// header first so a mis-encoded fixture (wrong rate/channels — which would
/// invalidate the stated provenance yet still decode to numbers) fails loudly
/// instead of quietly feeding the wrong geometry into a gate.
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
        .map(|s| s.unwrap() as f32 * scale)
        .collect()
    }
    hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
  }
}

/// Lowercase-hex SHA-256 of a byte slice.
#[allow(dead_code)]
pub fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// Lowercase-hex SHA-256 of a file's contents.
#[allow(dead_code)]
pub fn sha256_file(path: &Path) -> String {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  sha256_hex(&bytes)
}

/// Cosine of two equal-length finite vectors — FAIL-CLOSED (the #30 lesson):
/// panics on a length mismatch or any non-finite component rather than
/// returning a silently-wrong scalar.
#[allow(dead_code)]
pub fn cosine_checked(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(a.len(), b.len(), "cosine operands differ in length");
  assert!(!a.is_empty(), "cosine of empty vectors is undefined");
  let mut dot = 0.0f64;
  let mut na = 0.0f64;
  let mut nb = 0.0f64;
  for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
    assert!(x.is_finite(), "left operand non-finite at {i}");
    assert!(y.is_finite(), "right operand non-finite at {i}");
    dot += (x as f64) * (y as f64);
    na += (x as f64) * (x as f64);
    nb += (y as f64) * (y as f64);
  }
  assert!(na > 0.0 && nb > 0.0, "cosine of a zero vector is undefined");
  (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Recursively collects the forward-slash relative path of every FILE under
/// `dir` into `out`. Pure path enumeration — hermetically testable without
/// real model bytes. OS-generated sidecars (AppleDouble `._*`, `.DS_Store`)
/// are skipped: CoreML's loader never reads them, so excluding them cannot
/// mask a functional change, whereas keeping them would false-fail the
/// exact-set gate.
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

/// Assert that the compiled-model bundle at `dir` matches an EXACT pinned
/// manifest (`(relative_path, sha256)` list): the discovered file set must
/// EQUAL the pinned key set (so a missing OR an added artifact both red), and
/// each file's SHA-256 must equal its pinned value (the clap #30 exact-SHA
/// remediation standard).
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
       extra (on disk but not pinned): {extra:?}\n  \
       if the bundle changed intentionally, update the pinned `cases` list AND \
       the doc SHA table"
    );
  }

  for (relative, expected) in cases {
    let actual = sha256_file(&dir.join(relative));
    assert_eq!(
      &actual, expected,
      "sha256 drift on artifact {relative} under {dir:?}"
    );
  }
}
