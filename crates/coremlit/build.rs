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
//! sweep walks `Models/` WHOLE, so CI's model job — which downloads whisper and
//! granite on every PR — audits the vendored VAD graph too, and its one fp16
//! guard site is additionally pinned hermetically by the parser test
//! `accepts_vadkits_stft_sqrt_guard`, which runs everywhere with no models at
//! all.
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

  let models =
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
      .join("../../Models");
  println!("cargo::rerun-if-changed={}", models.display());

  if downloaded_tree_present(&models) {
    println!("cargo::rustc-cfg=models_present");
  }
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
