//! Hermetic tokenizer-identity gate — the proof the bundled SigLIP Gemma
//! tokenizer is byte-correct and reproduces the committed padded token windows,
//! with NO model and NO network.
//!
//! # Status: Wave B (goldens-gated, hermetic)
//!
//! The bundled `src/embeddings/siglip/assets/tokenizer.json` is the
//! source-revision Gemma artifact, and the committed corpus
//! (`fixtures/goldens/corpus.json`) carries every text's golden padded window.
//! These gates pin the tokenizer SHA-256 and assert every text entry's built
//! PADDED window (D6 — the full `[T]` window, side/id load-bearing) equals its
//! golden `token_ids_padded` byte-for-byte, plus a non-vacuity perturbation and
//! the >64-token truncated entry.

mod common;

use coremlit::embeddings::siglip::{BUNDLED_TOKENIZER, text::configured_tokenizer_from_bytes};

/// The committed goldens' text window `T` (the shipped 64-token tier).
const WINDOW: usize = 64;

/// Build the fixed `[WINDOW]` padded `input_ids` for `text` exactly as the module
/// does: the configured tokenizer (composed `Lowercase` + `LongestFirst`
/// truncation at `WINDOW`, own padding disabled) encodes with special tokens, then
/// the real ids are right-padded with `<pad>` — the `build_window` Right contract.
fn build_padded_window(tok: &tokenizers::Tokenizer, text: &str) -> Vec<i32> {
  let encoding = tok.encode(text, true).expect("encode");
  let ids = encoding.get_ids();
  assert!(
    ids.len() <= WINDOW,
    "truncation must cap ids at the window (got {})",
    ids.len()
  );
  let pad_id = tok
    .token_to_id("<pad>")
    .and_then(|id| i32::try_from(id).ok())
    .unwrap_or(0);
  let mut window = vec![pad_id; WINDOW];
  for (i, &id) in ids.iter().enumerate() {
    window[i] = i32::try_from(id).expect("gemma id fits i32");
  }
  window
}

/// The bundled tokenizer is the exact source-revision Gemma artifact that cut the
/// goldens (SHA-256) and is the real multi-megabyte tokenizer — not the Wave-A
/// placeholder.
#[test]
fn bundled_tokenizer_matches_pinned_sha256_and_is_real() {
  let sha = common::sha256_hex(BUNDLED_TOKENIZER);
  assert_eq!(
    sha, "58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0",
    "bundled tokenizer.json is not the pinned google/siglip2-base-patch16-naflex \
     revision artifact"
  );
  assert_eq!(
    BUNDLED_TOKENIZER.len(),
    34_356_304,
    "unexpected tokenizer byte length"
  );
  assert!(
    BUNDLED_TOKENIZER.len() > 1_000_000,
    "the real Gemma tokenizer is tens of MB, never the small placeholder"
  );
}

/// Every committed text entry's built PADDED window equals its golden
/// `token_ids_padded` exactly (D6): the six captions, the MixedCase twin (whose
/// window must equal its lowercase caption's), and the >64-token entry (sticky-EOS
/// truncation — filled window, `<eos>` at the last slot, no pad). A one-token
/// perturbation of an input must change the window (non-vacuity).
#[test]
fn every_corpus_text_builds_its_golden_padded_window() {
  let (_images, texts) = common::golden_corpus();
  assert!(!texts.is_empty(), "committed corpus has no texts");
  let tok = configured_tokenizer_from_bytes(BUNDLED_TOKENIZER, WINDOW)
    .expect("configure bundled tokenizer");

  let mut saw_mixedcase = false;
  let mut saw_truncated = false;
  for entry in &texts {
    assert_eq!(
      entry.token_ids_padded.len(),
      WINDOW,
      "golden window must be {WINDOW}"
    );
    let built = build_padded_window(&tok, &entry.text);
    assert_eq!(
      built, entry.token_ids_padded,
      "built window for {:?} diverges from its golden token_ids_padded",
      entry.id
    );

    // n_real matches the non-<pad>(0) count of the golden window.
    let n_real = entry
      .token_ids_padded
      .iter()
      .take_while(|&&id| id != 0)
      .count();
    assert_eq!(n_real, entry.n_real, "n_real mismatch for {:?}", entry.id);

    if entry.id == "mixedcase_cat" {
      saw_mixedcase = true;
      let twin = texts
        .iter()
        .find(|t| t.id == "cap_cat")
        .expect("lowercase twin cap_cat");
      assert_eq!(
        entry.token_ids_padded, twin.token_ids_padded,
        "MixedCase twin window must equal its lowercase caption window"
      );
    }
    if entry.id == "long_truncated" {
      saw_truncated = true;
      assert_eq!(entry.n_real, WINDOW, "the long entry must fill the window");
      assert_eq!(
        entry.token_ids_padded[WINDOW - 1],
        1,
        "sticky-EOS: the last window slot must be <eos> (id 1)"
      );
      assert!(
        !entry.token_ids_padded.contains(&0),
        "a filled truncated window has no <pad>"
      );
    }
  }
  assert!(
    saw_mixedcase,
    "corpus must include the MixedCase twin entry"
  );
  assert!(
    saw_truncated,
    "corpus must include the >64-token truncated entry"
  );

  // Non-vacuity: a one-token perturbation of a caption changes its window.
  let base = &texts[0].text;
  let perturbed = format!("{base} umbrella");
  assert_ne!(
    build_padded_window(&tok, base),
    build_padded_window(&tok, &perturbed),
    "a token perturbation must change the built window (non-vacuity)"
  );
}
