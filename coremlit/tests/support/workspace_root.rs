//! Resolves the **workspace root** by SEARCHING for it, so nothing has to
//! count `../` hops from wherever it happens to sit.
//!
//! Every kit's `models_dir()`, every repository-file reader, and every
//! sibling-checkout resolver needs one of two anchors: the workspace root, or
//! the directory that HOLDS this checkout. Both used to be spelled as a hop
//! count off `env!("CARGO_MANIFEST_DIR")` — `…/../..` and `…/../../..` — once
//! per call site, in more than seventy places.
//!
//! A hop count is the wrong shape for this. It encodes the package's DEPTH in
//! the repository, so moving a package re-points every one of them at once;
//! and it fails SILENTLY, because `..` always resolves. A wrong count lands on
//! a real directory that merely holds no `Models/`, so a model gate reports
//! "not on disk" and a sibling resolver takes its not-found fallback — both of
//! which are also what a legitimately absent tree looks like. Nothing goes
//! red; a gate just stops gating.
//!
//! So the anchor is FOUND, not computed: walk up from the compiling package's
//! own directory to the nearest ancestor whose `Cargo.toml` declares a
//! `[workspace]` table. That is the workspace root by Cargo's own definition,
//! it is independent of how deep the package sits, and — because the answer
//! carries its own witness — it either finds a directory that really is the
//! root or [`workspace_root`] panics naming what it searched for.
//!
//! `env!("CARGO_MANIFEST_DIR")` expands against the crate being COMPILED, so
//! this resolves correctly from both packages — the same property the
//! neighbouring `coremlit_dir.rs` relies on for `coremlit`-relative fixtures.
//! Included with `#[path]`, the `tests/support/` convention, so there is one
//! copy and the count it replaces has nowhere to hide.

use std::path::{Path, PathBuf};

/// The nearest ancestor of `start` (itself included) whose `Cargo.toml`
/// declares a `[workspace]` table, or `None` when no ancestor does.
///
/// `None` is not an error here: a `cargo test` run from the PUBLISHED tarball
/// has no workspace manifest above it, and the repository-infrastructure
/// checks that notice deliberately skip rather than fail. Call sites that
/// genuinely require the repository use [`workspace_root`], which panics.
pub fn find_workspace_root(start: &Path) -> Option<PathBuf> {
  start
    .ancestors()
    .find(|dir| declares_a_workspace(&dir.join("Cargo.toml")))
    .map(Path::to_path_buf)
}

/// Whether `manifest` exists and holds a `[workspace]` table header.
///
/// Matched as a whole trimmed line so `[workspace.package]` and
/// `[workspace.dependencies]` — which every member manifest's ancestor also
/// carries, and which a `starts_with` would accept — cannot stand in for the
/// table that actually defines the workspace.
fn declares_a_workspace(manifest: &Path) -> bool {
  let Ok(text) = std::fs::read_to_string(manifest) else {
    return false;
  };
  text.lines().any(|line| line.trim() == "[workspace]")
}

/// The package directory of whichever crate is being compiled.
fn manifest_dir() -> &'static Path {
  Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The workspace root, or `None` outside a workspace (the published tarball).
pub fn try_workspace_root() -> Option<PathBuf> {
  find_workspace_root(manifest_dir())
}

/// The workspace root.
///
/// # Panics
/// If no ancestor of the compiling package declares `[workspace]`. That is the
/// point: a resolver that cannot name its anchor must say so, not hand back a
/// directory that merely exists.
pub fn workspace_root() -> PathBuf {
  try_workspace_root().unwrap_or_else(|| {
    panic!(
      "no ancestor of {} has a Cargo.toml declaring `[workspace]`; the workspace root could not \
       be found",
      manifest_dir().display()
    )
  })
}

/// `<workspace>/Models` — the tree every kit's `models_dir()` falls back to
/// when its `*_TEST_MODELS` override is unset.
///
/// Never panics. Outside a workspace the published tarball packages neither
/// `Models/` nor the workspace manifest, and the model gates that read this
/// are `#[ignore]`d there; the ordinary-run `model_gate_report` is meant to
/// say "not on disk", which a nonexistent path under the package directory
/// reports exactly as well as a nonexistent path under a missing root.
pub fn models_root() -> PathBuf {
  try_workspace_root()
    .unwrap_or_else(|| manifest_dir().to_path_buf())
    .join("Models")
}

/// The directory that HOLDS this checkout — where the sibling checkouts the
/// oracle gates read (`diarization`, `asry`, `argmax-oss-swift`, `FluidAudio`)
/// are cloned next to it.
///
/// # Panics
/// If the workspace root cannot be found, or is the filesystem root and so has
/// no parent. Callers layer their own not-found fallback on TOP of this: a
/// missing sibling checkout is ordinary and skips, whereas not knowing where
/// to look is the miscount this module exists to prevent.
pub fn checkout_parent() -> PathBuf {
  let root = workspace_root();
  root
    .parent()
    .unwrap_or_else(|| panic!("workspace root {} has no parent directory", root.display()))
    .to_path_buf()
}
