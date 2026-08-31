// The workspace-root anchor every `models_dir()` below resolves against, and
// the sibling-checkout anchor the oracle gates read. FOUND by searching upward
// for the `[workspace]` manifest, never counted in `../` hops — see its module
// doc for why a count is the wrong shape here. Re-exported so the binaries
// that pull this `common` in share the one resolver.
#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;
#[allow(unused_imports)]
pub use workspace_root::{checkout_parent, models_root, workspace_root};

use std::path::PathBuf;

pub fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(workspace_root::models_root, PathBuf::from)
}

pub fn tiny_dir() -> PathBuf {
  models_dir()
    .join("whisperkit-coreml")
    .join("openai_whisper-tiny")
}

// ── Model-gate visibility (#61) ─────────────────────────────────────────────
//
// NOT `#[ignore]`d, deliberately. This is the ordinary-run half of the gate
// accounting: an ignored-ONLY run (`-- --ignored`, what every CI gate uses)
// never selects it, and it never appears in an ignored-only `--list`, so the
// anti-vacuum counts those gates take are unchanged. What it adds is the case
// no gate covers — a plain, modelless run — where the skipped gates otherwise
// say nothing but `ignored`. Mechanism, and what it does and does not refuse,
// in the shared module.
#[path = "../support/model_gate_report.rs"]
mod model_gate_report;

/// Reports how many of this binary's tests are `#[ignore]`d whisperkit model gates
/// that did not run, and whether the models root they read is on disk.
#[test]
fn model_gate_report() {
  model_gate_report::report(&[("WHISPERKIT_TEST_MODELS", models_dir())]);
}
