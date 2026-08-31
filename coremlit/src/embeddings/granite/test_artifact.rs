//! Test-only access to the granite `tokenizer.json` the SHIPPED artifact
//! carries.
//!
//! The crate embeds no granite tokenizer: the artifact is ~24 MB and
//! [`TextEmbedder::load`](super::TextEmbedder::load) reads it from the directory
//! containing the `.mlmodelc`. So the in-lib tests that need REAL granite
//! tokenization read the same staged file the integration tests do, and are
//! `#[ignore]`d on it. Shared by `granite::tests` and
//! `granite::token_index::tests`, which cannot reach the integration-test
//! helpers in `tests/granite/common`.

use std::{path::PathBuf, sync::OnceLock};

/// SHA-256 of the granite `tokenizer.json` at the source-model revision that cut
/// the committed goldens — the literal both the runtime pin
/// (`super::contract::TOKENIZER_SHA256_HEX`) and the staged artifact's bytes are
/// tied to, in two separate tests so the hermetic half still runs with no
/// artifact staged.
pub(super) const GOLDEN_SOURCE_TOKENIZER_SHA256: &str =
  "4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f";

/// The granite `tokenizer.json` sidecar as the shipped artifact stages it —
/// beside the `.mlmodelc` under `Models/embedkit-granite/` (overridable with
/// `EMBEDKIT_TEST_MODELS`), which is exactly where `TextEmbedder::load` reads it.
/// Read once and shared.
///
/// # Panics
/// If the staged artifact tree has no `tokenizer.json`. Every caller is
/// `#[ignore]`d on that artifact, so this panics only when a gate was asked for
/// explicitly against an incomplete tree.
pub(super) fn tokenizer_bytes() -> &'static [u8] {
  static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
  BYTES.get_or_init(|| {
    let path = tokenizer_path();
    std::fs::read(&path).unwrap_or_else(|e| {
      panic!(
        "read the staged granite tokenizer {}: {e} — stage the artifact tree (or point \
         EMBEDKIT_TEST_MODELS at one) before running this gate",
        path.display()
      )
    })
  })
}

/// Where [`tokenizer_bytes`] looks: the artifact root the granite integration
/// tests resolve (`tests/granite/common::model_root`), reproduced here because an
/// in-lib test module cannot use the integration-test helpers.
pub(super) fn tokenizer_path() -> PathBuf {
  std::env::var_os("EMBEDKIT_TEST_MODELS")
    .map_or_else(
      || crate::tests::models_root().join("embedkit-granite"),
      PathBuf::from,
    )
    .join("granite-97m-multilingual-r2")
    .join(super::TOKENIZER_FILE_NAME)
}
