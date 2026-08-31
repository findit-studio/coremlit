//! Emits `cfg(models_present)` when a DOWNLOADED model tree is on disk, so
//! `tests/fp16_guards.rs`'s graph sweep is **unignored and runs** exactly when
//! there is a fetched tree to sweep, and reports `ignored` — never a green
//! `ok` over zero models — when there is not.
//!
//! **Downloaded** is the load-bearing word, and it is why this is not simply
//! `Models/.is_dir()` any more. The VAD artifact
//! (`Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc/`) is COMMITTED —
//! `.gitignore` un-ignores exactly that path — so `Models/` now exists in every
//! fresh clone and in CI's modelless `check`/`features` jobs. Keying the cfg on
//! its bare existence would un-ignore the sweep there, where the sweep's
//! fail-closed vendor manifest (every `KNOWN_DEFECTS` vendor: `alignkit`,
//! `speakerkit`, `argmax-speakerkit`) demands vendors a clone legitimately does
//! not have — turning a fresh `cargo test` red. So the predicate is "a vendor
//! directory other than the committed ones", i.e. somebody actually fetched
//! models.
//!
//! The committed VAD graph is not thereby left unswept. Once the cfg is on the
//! sweep walks `Models/` WHOLE, so CI's model job — which downloads whisper,
//! granite and ced-tiny on every PR — audits the vendored VAD graph too, and
//! its one fp16 guard site is additionally pinned hermetically by the parser
//! test `accepts_vadkits_stft_sqrt_guard`, which runs everywhere with no models
//! at all.
//!
//! Downloaded model families added to `MODELS_LOCK` need no change here, and
//! must not be added to `VENDORED`: `Models/ced/` is gitignored like
//! whisper's and granite's trees, so its presence DOES mean somebody fetched
//! models — exactly what this predicate is asking. Only a path `.gitignore`
//! un-ignores belongs in that list.
//!
//! `rerun-if-changed` on `Models/` re-runs this script when that tree changes,
//! so the cfg flips on as soon as the models are downloaded — a stale `ignored`
//! cannot outlive their arrival.

use std::{
  env, fs,
  path::{Path, PathBuf},
};

/// Vendor directories COMMITTED under `Models/`. Their presence is guaranteed
/// by the checkout and so says nothing about whether anyone fetched models;
/// they must not flip `models_present` on. Adding a vendored model here without
/// adding it to this list fails in the SAFE direction — the sweep switches on
/// for fresh clones and goes red loudly, rather than going quietly dark.
const VENDORED: &[&str] = &["vadkit"];

fn main() {
  println!("cargo::rustc-check-cfg=cfg(models_present)");
  println!("cargo::rerun-if-changed=build.rs");

  let Some(root) = workspace_root() else {
    // Not in a workspace: a build from the PUBLISHED tarball, which packages
    // no `Models/` either. Nothing to find, nothing to watch, no cfg.
    return;
  };
  let models = root.join("Models");
  println!("cargo::rerun-if-changed={}", models.display());

  // Loud where loudness is possible. This repository is the only workspace
  // carrying `MODELS_LOCK`, and it COMMITS `Models/vadkit` (`.gitignore`
  // un-ignores exactly that path), so inside it the vendored artifact is
  // always on disk. If it is not, either the root found above is not this
  // repository's — the failure a `../` hop count used to produce in silence —
  // or the checkout is damaged. Outside this repository (the tarball, or
  // coremlit vendored into someone else's workspace) there is nothing to
  // assert, and the probe below simply finds no downloaded tree.
  assert!(
    !root.join("MODELS_LOCK").is_file() || models.join("vadkit").is_dir(),
    "{} declares the workspace and carries MODELS_LOCK, so it is coremlit's own repository, but \
     the committed VAD artifact is not at {}",
    root.display(),
    models.join("vadkit").display()
  );

  if downloaded_tree_present(&models) {
    println!("cargo::rustc-cfg=models_present");
  }
}

/// The workspace root: the nearest ancestor of this package whose `Cargo.toml`
/// declares a `[workspace]` table. `None` outside a workspace.
///
/// FOUND, not counted. A `../` hop count encodes the package's DEPTH in the
/// repository, so moving the package re-points it — and it fails silently,
/// because `..` always resolves: a wrong count lands on a real directory that
/// merely holds no models, and the whole fp16 graph sweep goes quietly dark
/// instead of red. Searching for Cargo's own definition of the root is
/// depth-independent and carries its own witness.
///
/// The test tree shares this idiom through `tests/support/workspace_root.rs`.
/// It is duplicated here rather than `#[path]`-included because a build script
/// must still compile from the published tarball, where which `tests/` files
/// ship is the packaging manifest's business and not this script's.
fn workspace_root() -> Option<PathBuf> {
  let manifest =
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
  manifest
    .ancestors()
    .find(|dir| declares_a_workspace(&dir.join("Cargo.toml")))
    .map(Path::to_path_buf)
}

/// Whether `manifest` exists and holds a `[workspace]` table header.
///
/// Matched as a whole trimmed line, so the `[workspace.package]` and
/// `[workspace.dependencies]` tables — which a `starts_with` would accept —
/// cannot stand in for the one that actually defines the workspace.
fn declares_a_workspace(manifest: &Path) -> bool {
  fs::read_to_string(manifest)
    .is_ok_and(|text| text.lines().any(|line| line.trim() == "[workspace]"))
}

/// Whether `models` holds any vendor directory that is not committed to the
/// repository. Dot-directories are skipped for the same reason the sweep's walk
/// skips them: `hf download` leaves a `.cache/huggingface/` bookkeeping tree
/// that is not a model.
fn downloaded_tree_present(models: &Path) -> bool {
  let Ok(entries) = fs::read_dir(models) else {
    return false;
  };
  entries.flatten().any(|entry| {
    let name = entry.file_name();
    let name = name.to_string_lossy();
    !name.starts_with('.') && !VENDORED.contains(&name.as_ref()) && entry.path().is_dir()
  })
}
