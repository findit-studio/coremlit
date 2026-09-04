//! Shared helpers and provenance pins for the identity-embedder tests.
//!
//! Two data sources, kept distinct:
//!
//! - **Committed fixtures** (`tests/identity/fixtures/`) — the mel goldens and
//!   their 16 kHz mono WAVs, cut from the conversion recipe's own oracle. Those
//!   are consumed by the library's in-source front-end gates
//!   (`src/audio/identity/mel/tests.rs`), which need no model and no network;
//!   this module does not read them.
//! - **CoreML artifact** ([`BUNDLE_NAME`]) — a gitignored dev-time download
//!   staged under `Models/redimnet/`, overridable via `IDENTITY_TEST_MODELS`.
//!   Model-gated tests are `#[ignore]` by default and run only when the
//!   artifact is staged.
//!
//! # The artifact repository is PRIVATE, and that is deliberate
//!
//! `IDRnD/redimnet` ships MIT over *"the Software"* — its model source — and
//! extends nothing in writing to the released `.pt` weights. Publishing a
//! conversion of them openly would make this the first FinDIT-Studio artifact
//! redistributing weights under no upstream grant, so the artifact repository
//! is private: CI fetching from our own private repository is USE, which is the
//! line `NOTICE` already draws ("CI DOWNLOADS; IT DOES NOT REDISTRIBUTE").
//!
//! The practical consequence is recorded here because it is where someone will
//! hit it: a tokenless `hf download` cannot stage this kit. See the `identity`
//! shard in `.github/workflows/ci.yml`.

// The workspace-root anchor `models_dir()` resolves against. FOUND by searching
// upward for the `[workspace]` manifest, never counted in `../` hops — see its
// module doc for why a count is the wrong shape here.
#[path = "../../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

// The committed per-table file manifest every digest below is read from. See
// its module doc for the grammar and for why the gates read a data file rather
// than each holding their own copy.
#[path = "../../support/models_lock_manifest.rs"]
#[allow(dead_code)]
mod models_lock_manifest;
#[allow(unused_imports)]
pub use workspace_root::{checkout_parent, models_root, workspace_root};

use std::path::{Path, PathBuf};

/// Hugging Face repository the artifact comes from. **Private** — see the
/// module doc.
#[allow(dead_code)]
pub const HF_REPO: &str = "FinDIT-Studio/redimnetkit-coreml";

/// Pinned artifact-repo revision the SHA-256 table below was taken from.
#[allow(dead_code)]
pub const HF_REVISION: &str = "80c2d0a40b0bacc738db2d8607470515afd9d405";

/// The upstream release asset the artifact was converted from, and its lock.
///
/// The release TAG is literally named `latest` and is mutable, so the tag is
/// not the pin — this SHA-256 is, and every stage of
/// `conversion/redimnet/run_redimnet.sh` verifies it.
#[allow(dead_code)]
pub const SOURCE_ASSET: &str = "b5-vox2-ft_lm.pt";

/// SHA-256 of [`SOURCE_ASSET`].
#[allow(dead_code)]
pub const SOURCE_ASSET_SHA256: &str =
  "8b0c11bbf5a3a8bb39e5c072c4192d0b694d8c447cf126d4cd3c7346a04b39c8";

/// Pinned revision of the upstream MODEL SOURCE (`IDRnD/redimnet`).
///
/// A checkpoint is only half the provenance: `ReDimNetWrap` is *reconstructed*
/// from the archive's own `model_config`, and the reconstructing code decides
/// what the weights compute — including the `MelBanks` the Rust front end
/// reproduces.
#[allow(dead_code)]
pub const SOURCE_CODE_REVISION: &str = "ce039a624cb99fe127702ceb94c6080090e5032f";

/// The compiled bundle's directory name inside the models root. The library
/// never spells this — [`coremlit::audio::identity::Embedder::from_file`] takes
/// whatever path the caller staged — so the artifact's own name lives here,
/// with the rest of its provenance.
#[allow(dead_code)]
pub const BUNDLE_NAME: &str = "redimnet_b5.mlmodelc";

/// The `MODELS_LOCK` `local-dir` this kit stages into, with `Models/` removed.
///
/// Half of the key into `MODELS_LOCK.d/`; [`HF_REVISION`] is the other half.
/// Both are read from the lock, so a revision bump that does not regenerate the
/// manifest lands as a missing file rather than as a stale digest.
#[allow(dead_code)]
pub const VENDOR_DIR: &str = "redimnet";

/// Exact per-file SHA-256 of the staged `.mlmodelc` bundle as published at
/// [`HF_REVISION`], read from the ONE place this repository records it:
/// `MODELS_LOCK.d/redimnet@<revision>.sha256`, which is upstream's own
/// `CHECKSUMS.sha256` at that revision with its paths made table-relative.
///
/// The set is exact — a missing OR an added file reds, not just a changed one.
/// It used to be a `const ARTIFACT_SHA256` here, with a second copy of the same
/// digests in `tests/model_licences.rs` tied to this one by a scanner over Rust
/// source (coremlit #139 retired both).
#[allow(dead_code)]
pub fn artifact_sha256() -> Vec<(String, String)> {
  models_lock_manifest::bundle_manifest(
    &workspace_root::workspace_root(),
    VENDOR_DIR,
    HF_REVISION,
    BUNDLE_NAME,
  )
}

/// Directory holding the downloaded CoreML artifact tree.
///
/// Overridable via `IDENTITY_TEST_MODELS`; otherwise
/// `<workspace>/Models/redimnet` — gitignored, fetched dev-time (the
/// `LID_TEST_MODELS` / `CED_TEST_MODELS` convention). The directory is named
/// for the BACKEND while the kit, the feature and the module are named for the
/// LANE, which is the same split `MODELS_LOCK` records: a second identity
/// backend would be a second directory under one `identity` gate.
#[allow(dead_code)]
pub fn models_dir() -> PathBuf {
  std::env::var_os("IDENTITY_TEST_MODELS").map_or_else(
    || workspace_root::models_root().join("redimnet"),
    PathBuf::from,
  )
}

/// Path to the compiled graph under [`models_dir`].
#[allow(dead_code)]
pub fn model_path() -> PathBuf {
  models_dir().join(BUNDLE_NAME)
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
pub fn assert_exact_sha_manifest(dir: &Path, cases: &[(String, String)]) {
  use std::collections::BTreeSet;

  let mut found = Vec::new();
  collect_files_rel(dir, "", &mut found);
  let on_disk: BTreeSet<String> = found.into_iter().collect();
  let pinned: BTreeSet<String> = cases.iter().map(|(rel, _)| rel.clone()).collect();

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

/// A deterministic, non-degenerate 6 s window: a two-tone signal with a slow
/// amplitude envelope, so a graph that answers with a constant regardless of
/// input is visibly wrong. Generated rather than committed — the committed
/// clips exist to pin the MEL, and a model gate needs only something that is
/// not silence.
#[allow(dead_code)]
pub fn synthetic_window(seed_hz: f64) -> Vec<f32> {
  let sr = f64::from(coremlit::audio::identity::SAMPLE_RATE_HZ);
  (0..coremlit::audio::identity::WINDOW_SAMPLES)
    .map(|i| {
      let t = i as f64 / sr;
      let env = 0.55 + 0.35 * (core::f64::consts::TAU * 0.9 * t).sin();
      let v = 0.6 * (core::f64::consts::TAU * seed_hz * t).sin()
        + 0.3 * (core::f64::consts::TAU * seed_hz * 2.5 * t).sin();
      (env * v * 0.5) as f32
    })
    .collect()
}

/// Cosine similarity between two raw embeddings, normalizing both — the door
/// emits RAW vectors on purpose, so every comparison here does the L2 the
/// scoring layer would.
#[allow(dead_code)]
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
  let dot: f64 = a
    .iter()
    .zip(b.iter())
    .map(|(x, y)| f64::from(*x) * f64::from(*y))
    .sum();
  let na = a
    .iter()
    .map(|x| f64::from(*x) * f64::from(*x))
    .sum::<f64>()
    .sqrt();
  let nb = b
    .iter()
    .map(|x| f64::from(*x) * f64::from(*x))
    .sum::<f64>()
    .sqrt();
  dot / (na * nb)
}
