//! Emits `cfg(models_present)` when the workspace's `Models/` tree is on
//! disk, so `tests/fp16_guards.rs`'s graph sweep is **unignored and runs**
//! exactly when there is something to sweep, and reports `ignored` — never
//! a green `ok` over zero models — when there is not.
//!
//! `Models/` is gitignored (`.gitignore`: `Models/`) and CI's ordinary
//! `check` job never downloads it, so the sweep cannot be an unconditional
//! `#[test]`: it would fail every fresh clone. It equally must not be an
//! unconditional `#[ignore]`, or it would stay dark on the developer
//! machines that DO have the models. A build-time cfg is the only thing
//! that yields libtest's third status, `ignored`, on exactly the runs that
//! have nothing to check.
//!
//! `rerun-if-changed` on a path that does not exist re-runs this script on
//! every build, so the cfg flips on as soon as the models are downloaded —
//! a stale `ignored` cannot outlive the `Models/` directory's arrival.

use std::{env, path::PathBuf};

fn main() {
  println!("cargo::rustc-check-cfg=cfg(models_present)");
  println!("cargo::rerun-if-changed=build.rs");

  let models =
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
      .join("../../Models");
  println!("cargo::rerun-if-changed={}", models.display());

  if models.is_dir() {
    println!("cargo::rustc-cfg=models_present");
  }
}
