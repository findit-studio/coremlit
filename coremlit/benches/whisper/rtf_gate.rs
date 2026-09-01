//! Hermetic unit tests for [`models_ready`] (`benches/whisper/rtf.rs`'s
//! model-completeness check) — the `whisper_rtf_gate` test target.
//!
//! A real, separate file from the `whisper_rtf` bench itself. Before this
//! split, both targets compiled the SAME `benches/whisper/rtf.rs` source
//! under two different Cargo target kinds: deliberate (this target, with its
//! default libtest harness, was the only way to run these tests at all,
//! since the bench's `harness = false` links no runner to call them), but it
//! cost a permanent `cargo`-level "file `benches/whisper/rtf.rs` found to be
//! present in multiple build targets" warning on every build, for a saving
//! of moving seven lines. The two now share only [`models_ready`] itself, via
//! `rtf_models_ready.rs` (`#[path]`, the `tests/support/workspace_root.rs`
//! convention) — real, separate files, no drifting duplicate, and the
//! warning is gone rather than merely explained.
//!
//! Run: `cargo test -p coremlit --features whisper --test whisper_rtf_gate`

#[path = "rtf_models_ready.rs"]
mod rtf_models_ready;
use rtf_models_ready::models_ready;

fn mlmodelc_dirs(model_dir: &std::path::Path) {
  for name in ["MelSpectrogram", "AudioEncoder", "TextDecoder"] {
    std::fs::create_dir_all(model_dir.join(format!("{name}.mlmodelc"))).unwrap();
  }
}

#[test]
fn empty_root_is_not_ready() {
  let model_dir = tempfile::tempdir().unwrap();
  let tokenizer_dir = tempfile::tempdir().unwrap();
  assert!(!models_ready(model_dir.path(), tokenizer_dir.path()));
}

#[test]
fn model_dir_without_tokenizer_json_is_not_ready() {
  let model_dir = tempfile::tempdir().unwrap();
  let tokenizer_dir = tempfile::tempdir().unwrap();
  mlmodelc_dirs(model_dir.path());
  // tokenizer_dir exists but stays empty — the interrupted-download case.
  assert!(!models_ready(model_dir.path(), tokenizer_dir.path()));
}

#[test]
fn tokenizer_json_without_model_dirs_is_not_ready() {
  let model_dir = tempfile::tempdir().unwrap();
  let tokenizer_dir = tempfile::tempdir().unwrap();
  std::fs::write(tokenizer_dir.path().join("tokenizer.json"), b"{}").unwrap();
  assert!(!models_ready(model_dir.path(), tokenizer_dir.path()));
}

#[test]
fn fully_populated_root_is_ready() {
  let model_dir = tempfile::tempdir().unwrap();
  let tokenizer_dir = tempfile::tempdir().unwrap();
  mlmodelc_dirs(model_dir.path());
  std::fs::write(tokenizer_dir.path().join("tokenizer.json"), b"{}").unwrap();
  assert!(models_ready(model_dir.path(), tokenizer_dir.path()));
}
