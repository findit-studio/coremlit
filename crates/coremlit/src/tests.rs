//! The in-lib half of the model-gate report (#61).
//!
//! Most of this crate's model gates are NOT integration tests: they are
//! `#[ignore]`d unit tests inside the pipeline modules — more of them than
//! every `tests/` binary holds put together — and CI reaches them through
//! three `model-tests` shards: whisper's two `@all` groups, which build and run
//! the lib target alongside every integration one, and the granite and speaker
//! shards' `@lib` ones. The `align` gates reach no shard at all, because
//! alignkit has no MODELS_LOCK table; ci.yml's matrix carries the per-kit
//! ledger, counts included. They are skipped by the same
//! silence, so they get the same report; see
//! `crates/coremlit/tests/support/model_gate_report.rs` for the mechanism.
//!
//! That module lives under `tests/` because every other caller is a test
//! binary, and this one `#[path]`-hops to it rather than keeping a second copy
//! — the `tests/support/coremlit_dir.rs` convention, one level further. The hop
//! is `#[cfg(test)]`, so a published crate never compiles it.

use std::path::PathBuf;

#[path = "../tests/support/model_gate_report.rs"]
mod model_gate_report;

/// `<workspace>/Models/<sub>` unless `var` overrides it — the fallback every
/// in-lib `models_dir()` in this crate resolves. `sub` is empty for whisper,
/// whose gates read the `Models/` root itself.
fn root(var: &str, sub: &str) -> PathBuf {
  std::env::var_os(var).map_or_else(
    || {
      let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models");
      if sub.is_empty() {
        models
      } else {
        models.join(sub)
      }
    },
    PathBuf::from,
  )
}

/// Reports how many of the library's own unit tests are `#[ignore]`d model
/// gates that did not run, and whether the models roots they read are on disk.
///
/// The roots are named per feature because the gates are: a bare
/// `default = []` build compiles no pipeline, so it has no model gates and
/// claims no roots. `cfg!` rather than `#[cfg]` on the elements, so the vector
/// is used mutably on every combination — including that empty one.
///
/// Today's in-lib gates span exactly these four kits (whisper, align, speaker,
/// granite); the `clap`/`siglip`/`ced`/`vad` gates are all in `tests/`, where
/// their own `common/mod.rs` names their root. A new in-lib gate under one of
/// those would still be COUNTED — the count is libtest's, not this list's — it
/// would simply have no root named beside it until a line is added here.
#[test]
fn model_gate_report() {
  let mut roots: Vec<(&str, PathBuf)> = Vec::new();
  if cfg!(feature = "whisper") {
    roots.push(("WHISPERKIT_TEST_MODELS", root("WHISPERKIT_TEST_MODELS", "")));
  }
  if cfg!(feature = "align") {
    roots.push((
      "ALIGNKIT_TEST_MODELS",
      root("ALIGNKIT_TEST_MODELS", "alignkit"),
    ));
  }
  if cfg!(feature = "speaker") {
    roots.push((
      "SPEAKERKIT_TEST_MODELS",
      root("SPEAKERKIT_TEST_MODELS", "speakerkit"),
    ));
    roots.push((
      "ARGMAX_TEST_MODELS",
      root("ARGMAX_TEST_MODELS", "argmax-speakerkit"),
    ));
  }
  if cfg!(feature = "granite") {
    roots.push((
      "EMBEDKIT_TEST_MODELS",
      root("EMBEDKIT_TEST_MODELS", "embedkit-granite"),
    ));
  }
  model_gate_report::report(&roots);
}
