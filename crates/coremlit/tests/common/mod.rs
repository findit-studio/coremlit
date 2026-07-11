use std::path::PathBuf;

pub fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  )
}

pub fn tiny_dir() -> PathBuf {
  models_dir()
    .join("whisperkit-coreml")
    .join("openai_whisper-tiny")
}
