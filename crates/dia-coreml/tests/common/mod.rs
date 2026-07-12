use std::path::PathBuf;

/// Directory containing the downloaded dia-coreml model artifacts.
///
/// Overridable via `DIA_COREML_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/dia-coreml` — gitignored, fetched dev-time per the
/// design spec §4 (mirrors whisperkit's `WHISPERKIT_TEST_MODELS`/`Models/`
/// convention, one directory level down for this crate's own model set).
pub fn models_dir() -> PathBuf {
  std::env::var_os("DIA_COREML_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("dia-coreml")
    },
    PathBuf::from,
  )
}

/// Path to the decided segmentation artifact.
///
/// See `tests/model_io.rs`'s `// DECISION:` comment for the introspection
/// that picked `pyannote_segmentation.mlmodelc` over `Segmentation.mlmodelc`.
pub fn seg_path() -> PathBuf {
  models_dir().join("pyannote_segmentation.mlmodelc")
}

/// Path to the decided embedding artifact: the raw-waveform, in-graph-fbank
/// WeSpeaker v2 model (spec §2.4 — no separate fbank stage needed).
///
/// See `tests/model_io.rs`'s `// DECISION:` comment for why this is
/// `wespeaker_v2.mlmodelc` and not `wespeaker.mlmodelc`/`wespeaker_int8.mlmodelc`.
pub fn embed_path() -> PathBuf {
  models_dir().join("wespeaker_v2.mlmodelc")
}
