//! Locates paths expressed relative to the **`coremlit` crate root** from
//! either workspace member that compiles the shared test-support modules.
//!
//! `crates/coremlit/tests/{speaker,clap,vad}/common/mod.rs` are shared: the
//! `coremlit` package's own test binaries declare them with a plain
//! `mod common;`, while the `coremlit-parity` package's oracle binaries pull
//! the very same files in with `#[path = "../../../coremlit/tests/…"]` (one
//! copy, so the two sides cannot drift). `env!("CARGO_MANIFEST_DIR")` expands
//! against the crate being COMPILED, not the file's location, so from
//! `coremlit-parity` it names `crates/coremlit-parity` and every committed
//! fixture under `crates/coremlit/tests/**/fixtures/` would resolve to a
//! nonexistent path.
//!
//! [`coremlit_dir`] closes that gap with a compile-time-decided hop to the
//! sibling package directory. Deliberately NOT a filesystem probe: the
//! resolved path is the manifest dir itself when compiled into `coremlit`, so
//! the 13 test binaries that stay behind see byte-identical paths (and
//! byte-identical panic messages) before and after the split.
//!
//! Note that `Models/` and sibling-checkout resolvers do NOT need this: both
//! packages sit at `crates/<name>`, so `CARGO_MANIFEST_DIR/../..` is the
//! workspace root either way and `…/../../..` the checkout's parent.

use std::path::{Path, PathBuf};

/// The `crates/coremlit` directory, whichever package is being compiled.
pub fn coremlit_dir() -> PathBuf {
  let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
  if env!("CARGO_PKG_NAME") == "coremlit" {
    manifest.to_path_buf()
  } else {
    manifest.join("../coremlit")
  }
}

/// Joins `rel` onto [`coremlit_dir`].
pub fn coremlit_path(rel: impl AsRef<Path>) -> PathBuf {
  coremlit_dir().join(rel)
}
