//! Shared between `benches/whisper/rtf.rs` (the `whisper_rtf` bench, `harness
//! = false`) and `benches/whisper/rtf_gate.rs` (the `whisper_rtf_gate` test,
//! default harness) via `#[path]` — the `tests/support/workspace_root.rs`
//! convention for a small piece two standalone binaries both need.
//!
//! Before this split, both targets compiled the SAME `benches/whisper/rtf.rs`
//! source under two different Cargo target kinds (bench and test), which was
//! deliberate — the gate target was the only way to run [`models_ready`]'s
//! hermetic tests, since a `harness = false` bench links no libtest runner to
//! call them — but it cost a permanent `cargo`-level "file
//! `benches/whisper/rtf.rs` found to be present in multiple build targets"
//! warning on every build, for a saving of moving seven lines. Splitting the
//! one thing the gate target actually needs into its own file keeps both
//! entry points real, separate files with no drifting duplicate, and the
//! warning is gone rather than merely explained.

use std::path::Path;

/// Required on-disk artifacts for the tiny model: the three compiled CoreML
/// bundles under `model_dir`, and the tokenizer file under `tokenizer_dir`.
/// Directory existence alone is not proof of a complete download — an
/// interrupted `hf download` (see MODELS_LOCK / the README's "Getting
/// models") can leave both folders present while missing individual files
/// inside them, which used to reach `WhisperKit::new().expect(...)` and
/// panic instead of skipping.
pub fn models_ready(model_dir: &Path, tokenizer_dir: &Path) -> bool {
  const MODEL_BUNDLES: [&str; 3] = ["MelSpectrogram", "AudioEncoder", "TextDecoder"];
  MODEL_BUNDLES
    .iter()
    .all(|name| model_dir.join(format!("{name}.mlmodelc")).is_dir())
    && tokenizer_dir.join("tokenizer.json").is_file()
}
