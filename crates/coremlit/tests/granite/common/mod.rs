//! Shared helpers for the granite embedding tests.
//!
//! Two data sources, kept distinct:
//!
//! - **Committed goldens** (`tests/granite/fixtures/goldens/corpus.json`) — the
//!   in-tree transformers-fp32 oracle (token-ids AND unit-normalized embedding
//!   goldens). Read hermetically; no model, no network. This is the parity
//!   ground truth (the embedkit "no ort anywhere" rule), committed the way the
//!   speaker/vad Swift goldens are.
//! - **CoreML artifact** (`granite_97m_512.mlmodelc`) — a gitignored dev-time
//!   download from `FinDIT-Studio/embedkit-coreml` (revision `81852f70`), under
//!   `Models/embedkit-granite/` (overridable via `EMBEDKIT_TEST_MODELS`).
//!   Model-gated tests are `#[ignore]` by default and run only when present.

use std::path::{Path, PathBuf};

/// The HF revision (commit SHA) of `FinDIT-Studio/embedkit-coreml` the model
/// artifact and its per-file SHA-256 pins are recorded at.
#[allow(dead_code)]
pub const EMBEDKIT_REVISION: &str = "81852f70";

/// Directory containing the downloaded granite CoreML artifact tree.
///
/// Overridable via `EMBEDKIT_TEST_MODELS`; otherwise
/// `<workspace>/Models/embedkit-granite` — gitignored, fetched dev-time
/// (mirrors the clapkit `CLAPKIT_TEST_MODELS` / speakerkit `SPEAKERKIT_TEST_MODELS`
/// conventions).
#[allow(dead_code)]
pub fn models_dir() -> PathBuf {
  std::env::var_os("EMBEDKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("embedkit-granite")
    },
    PathBuf::from,
  )
}

/// The granite model bundle's parent directory
/// (`.../granite-97m-multilingual-r2`), which holds the `.mlmodelc`,
/// `.mlpackage`, and `CHECKSUMS.sha256`.
#[allow(dead_code)]
pub fn model_root() -> PathBuf {
  models_dir().join("granite-97m-multilingual-r2")
}

/// Path to the compiled granite text encoder, `granite_97m_512.mlmodelc`.
#[allow(dead_code)]
pub fn model_path() -> PathBuf {
  model_root().join("granite_97m_512.mlmodelc")
}

/// Absolute path to a committed fixture under `crates/coremlit/tests/granite/fixtures`.
#[allow(dead_code)]
pub fn fixture_path(relative: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("granite")
    .join("fixtures")
    .join(relative)
}

/// One committed golden corpus entry: the raw text, its exact token-ids (granite
/// tokenizer, truncated at 512, special tokens included), and the
/// transformers-fp32 **unit-normalized** 384-d embedding.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
pub struct GoldenEntry {
  /// Stable per-entry id (`en_q`, `zh`, `near512`, …).
  pub id: String,
  /// The raw input string (prompt-free — no task prefix).
  pub text: String,
  /// The golden token-id sequence (truncated at 512, specials included).
  pub token_ids: Vec<u32>,
  /// The golden token count (`== token_ids.len()`).
  pub n_tokens: usize,
  /// The transformers-fp32 unit-normalized 384-d embedding.
  pub embedding: Vec<f32>,
}

#[derive(Debug, serde::Deserialize)]
struct Corpus {
  entries: Vec<GoldenEntry>,
}

/// Load the committed golden corpus (16 entries). Hermetic — reads the in-tree
/// fixture, never `Models/`.
#[allow(dead_code)]
pub fn golden_corpus() -> Vec<GoldenEntry> {
  let path = fixture_path("goldens/corpus.json");
  let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  let corpus: Corpus =
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
  assert_eq!(
    corpus.entries.len(),
    16,
    "the committed granite golden corpus must have 16 entries"
  );
  corpus.entries
}

/// Lowercase-hex SHA-256 of a byte slice. Backs the `model_io` provenance pins
/// (the #30 enumerate-then-hash manifest pattern hashes the bytes it read).
#[allow(dead_code)]
pub fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// Cosine of two equal-length finite vectors — FAIL-CLOSED (the #30 lesson):
/// panics on a length mismatch or any non-finite component rather than returning
/// a silently-wrong scalar. `Embedding` is fixed-dim, but a golden vector is a
/// `Vec`, so its length and finiteness are checked here.
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

/// Recursively collects every real FILE under `dir` as a path relative to
/// `root`, with `/` separators, into `out`. Used to enumerate a `.mlmodelc`
/// bundle's actual tree so it can be set-compared against a pinned key manifest
/// (no unpinned extras, none missing) BEFORE hashing — the #30 exact-enumerate
/// pattern.
///
/// OS-generated sidecars are skipped: AppleDouble `._*` files and `.DS_Store`.
/// macOS materializes these inside bundles on non-native filesystems
/// (exFAT/FAT/SMB); CoreML's loader never reads them, so excluding them from
/// discovery cannot mask a functional artifact change — whereas NOT excluding
/// them would false-fail the exact-set gate as a phantom "unpinned extra" even
/// though every pinned byte is untouched.
#[allow(dead_code)]
pub fn collect_files_rel(root: &Path, dir: &Path, out: &mut std::collections::BTreeSet<String>) {
  for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
  {
    let entry = entry.expect("read dir entry");
    // Drop OS-generated sidecars (AppleDouble `._*`, `.DS_Store`) at every
    // depth, before the file/dir split — see the doc comment above.
    let name = entry.file_name();
    let name = name.to_string_lossy();
    if name.starts_with("._") || name == ".DS_Store" {
      continue;
    }
    let path = entry.path();
    if entry.file_type().expect("file type").is_dir() {
      collect_files_rel(root, &path, out);
    } else {
      let rel = path
        .strip_prefix(root)
        .expect("walked path is under root")
        .to_str()
        .expect("utf-8 path")
        .replace('\\', "/");
      out.insert(rel);
    }
  }
}
