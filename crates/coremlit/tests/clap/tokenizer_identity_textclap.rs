//! LIVE textclap tokenizer cross-check — the `clap-oracle` half of the
//! token-id identity gate.
//!
//! Built ONLY under `--features clap-oracle` (see the `[[test]]`
//! `required-features` in `Cargo.toml`), because linking the sibling `textclap`
//! crate pulls its non-optional `ort` dependency. clapkit calls only textclap's
//! `BUNDLED_TOKENIZER` const here (no `ort` runtime), so the gate needs no
//! `libonnxruntime` at run time.
//!
//! Byte-identity of the two bundled tokenizers is the definitive "vs textclap"
//! proof: identical `tokenizer.json` bytes + the identical `LongestFirst@512`
//! truncation config (`src/text.rs::configure_truncation`, mirroring textclap's
//! `force_max_length_truncation`) produce identical token ids by construction.
//! The hermetic `tests/clap/tokenizer_identity.rs` + `src/text/tests.rs` gates carry
//! the same guarantee via a SHA pin and pinned golden ids; this is the live
//! belt-and-suspenders confirmation.

#[test]
fn bundled_tokenizers_are_byte_identical() {
  assert_eq!(
    coremlit::embeddings::clap::BUNDLED_TOKENIZER,
    textclap::BUNDLED_TOKENIZER,
    "clapkit and textclap bundle different tokenizer.json bytes — token-id identity is broken at \
     the artifact level"
  );
}
