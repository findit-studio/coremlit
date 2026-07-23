//! Hermetic tokenizer-identity gate — the proof the bundled SigLIP Gemma
//! tokenizer is byte-correct and reproduces the committed padded token windows,
//! with NO model and NO network.
//!
//! # Status: Wave B shell (goldens-gated)
//!
//! `#[ignore]`d until the golden-generation step stages the source-revision
//! tokenizer bytes (`src/embeddings/siglip/assets/tokenizer.json`, currently a
//! placeholder) and the committed corpus (`fixtures/goldens/corpus.json`). Wave B
//! then pins the tokenizer SHA-256 and asserts every text entry's built PADDED
//! window (D6 — the full `[T]` window, side/id load-bearing) equals its golden
//! `token_ids_padded` byte-for-byte, plus a non-vacuity perturbation and a
//! >64-token truncated entry.

mod common;

use coremlit::embeddings::siglip::BUNDLED_TOKENIZER;

/// The bundled tokenizer is the exact source-revision artifact that cut the
/// goldens (SHA-256) and is the real multi-megabyte Gemma tokenizer. Wave B pins
/// the real SHA; today `BUNDLED_TOKENIZER` is a placeholder.
#[test]
#[ignore = "requires source-revision Gemma tokenizer + committed goldens — Wave B"]
fn bundled_tokenizer_matches_pinned_sha256_and_is_real() {
  let _sha = common::sha256_hex(BUNDLED_TOKENIZER);
  // Wave B: assert_eq!(_sha, "<source-revision tokenizer.json SHA-256>");
  //         assert!(BUNDLED_TOKENIZER.len() > 1_000_000, "real Gemma tokenizer");
}

/// Every committed text entry's built PADDED window equals its golden
/// `token_ids_padded` exactly (D6). Wave B implements against the staged corpus.
#[test]
#[ignore = "requires source-revision Gemma tokenizer + committed goldens — Wave B"]
fn every_corpus_text_builds_its_golden_padded_window() {
  let (_images, _texts) = common::golden_corpus();
  // Wave B: for each text, TextEmbedder-configured tokenizer + build_window ==
  //         entry.token_ids_padded exactly; include a >64-token (truncated) and
  //         a MixedCase entry; one-token perturbation must differ (non-vacuity).
}
