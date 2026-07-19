//! Hermetic tokenizer-identity gate — the proof the bundled granite tokenizer
//! is byte-correct, with NO model and NO network. Runs in the default
//! `cargo test --features granite` (not `#[ignore]`d).
//!
//! For every entry of the committed golden corpus
//! (`tests/granite/fixtures/goldens/corpus.json`), the bundled tokenizer —
//! configured EXACTLY as [`coremlit::embeddings::granite::TextEmbedder`] configures
//! it (`LongestFirst` truncation at 512, right direction, special tokens on) —
//! must reproduce the committed `token_ids` sequence EXACTLY. The goldens store
//! the truncated-at-512, special-token-bracketed, UNPADDED ids
//! (`token_ids.len() == n_tokens`), so this compares the real token sequence the
//! embedder feeds the graph (before the fixed-window pad) against the oracle.
//!
//! A mismatch means the bundled tokenizer or its version is wrong: report it and
//! FIX the tokenizer — never adjust the goldens.

mod common;

use coremlit::embeddings::granite::BUNDLED_TOKENIZER;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// The bundled tokenizer, configured identically to the embedder's runtime seam
/// (`configure_truncation`): `LongestFirst` at 512, stride 0, right direction.
fn configured_tokenizer() -> Tokenizer {
  let mut tok = Tokenizer::from_bytes(BUNDLED_TOKENIZER).expect("load bundled granite tokenizer");
  tok
    .with_truncation(Some(TruncationParams {
      max_length: 512,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .expect("configure truncation");
  tok
}

/// The bundled tokenizer is the exact artifact that cut the goldens (SHA-256)
/// and is the real multi-megabyte tokenizer, not a stub. Byte-identity is the
/// foundation of token-id identity.
#[test]
fn bundled_tokenizer_matches_pinned_sha256_and_is_real() {
  let sha = common::sha256_hex(BUNDLED_TOKENIZER);
  assert_eq!(
    sha, "4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f",
    "coremlit::embeddings::granite::BUNDLED_TOKENIZER diverged from the granite tokenizer \
     (ibm-granite/granite-embedding-97m-multilingual-r2 @ 835ad140) that produced the goldens"
  );
  assert!(
    BUNDLED_TOKENIZER.len() > 1_000_000,
    "bundled granite tokenizer is implausibly small ({} bytes)",
    BUNDLED_TOKENIZER.len()
  );
}

/// EXACT token-id equality over the full committed corpus (English, CJK, RTL
/// Arabic, emoji, code, a URL/number string, a >512-token entry, mixed scripts).
/// This is the hermetic identity gate: if any sequence differs, the bundled
/// tokenizer/version is wrong — report and fix the tokenizer, do NOT touch the
/// goldens.
#[test]
fn every_corpus_entry_tokenizes_to_its_golden_ids() {
  let tok = configured_tokenizer();
  let corpus = common::golden_corpus();
  let mut mismatches = Vec::new();
  for e in &corpus {
    // Sanity on the golden itself: the committed ids are unpadded (len == n_tokens).
    assert_eq!(
      e.token_ids.len(),
      e.n_tokens,
      "golden `{}` is internally inconsistent (token_ids.len() != n_tokens)",
      e.id
    );
    let got = tok
      .encode(e.text.as_str(), true)
      .unwrap_or_else(|err| panic!("encode `{}`: {err}", e.id))
      .get_ids()
      .to_vec();
    if got != e.token_ids {
      let n = got.len().min(e.token_ids.len());
      let first = (0..n).find(|&i| got[i] != e.token_ids[i]).unwrap_or(n);
      mismatches.push(format!(
        "`{}`: got {} ids, golden {} ids, first divergence @ {first}",
        e.id,
        got.len(),
        e.token_ids.len()
      ));
    }
  }
  assert!(
    mismatches.is_empty(),
    "bundled granite tokenizer does not reproduce the committed token-id goldens \
     (the tokenizer/version is wrong — fix the tokenizer, do NOT adjust the goldens):\n  {}",
    mismatches.join("\n  ")
  );

  // Non-vacuity: at least one corpus entry actually exercised >512 truncation
  // (the `near512` entry is naturally 675 tokens, truncated to exactly 512).
  assert!(
    corpus.iter().any(|e| e.n_tokens == 512),
    "the corpus must include a truncated-at-512 entry to exercise the truncation path"
  );
}

/// A deliberately WRONG expectation must NOT match — proves the identity check
/// above is non-vacuous (it is comparing real ids, not trivially passing). A
/// one-token perturbation of a golden sequence must differ from the tokenizer's
/// real output.
#[test]
fn identity_check_is_non_vacuous() {
  let tok = configured_tokenizer();
  let corpus = common::golden_corpus();
  let e = &corpus[0];
  let real = tok
    .encode(e.text.as_str(), true)
    .expect("encode")
    .get_ids()
    .to_vec();
  let mut perturbed = e.token_ids.clone();
  perturbed[1] = perturbed[1].wrapping_add(1);
  assert_ne!(
    real, perturbed,
    "a one-token perturbation must differ from the real tokenization"
  );
  assert_eq!(
    real, e.token_ids,
    "and the real tokenization equals the golden"
  );
}
