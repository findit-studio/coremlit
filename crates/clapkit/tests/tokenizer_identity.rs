//! Public-surface tokenizer-identity contract (hermetic; always runs).
//!
//! The token-id EXACT-equality gate is a three-part structure:
//!
//! 1. **This file** pins the PUBLIC [`clapkit::BUNDLED_TOKENIZER`] const's
//!    SHA-256 to the identical Xenova artifact textclap pins
//!    (`textclap/models/MODELS.sha256`) — byte-identity of the tokenizer is the
//!    foundation of token-id identity.
//! 2. `src/text/tests.rs` exercises clapkit's ACTUAL configured tokenizer seam
//!    over a pinned corpus (English, CJK, emoji, >512-token truncation) and pins
//!    the exact token-id sequences.
//! 3. `tests/tokenizer_identity_textclap.rs` (feature `parity-oracle`) is the
//!    LIVE cross-check: it links the sibling textclap crate and asserts the two
//!    crates' bundled tokenizer bytes are byte-for-byte identical.

/// The pinned Xenova `tokenizer.json` SHA-256 (revision `c28f2883…`), identical
/// to textclap's pin.
const PINNED_TOKENIZER_SHA256: &str =
  "dc239041d98de27ffc3975473a1a23e3db4c937b23c138c38bbc66588bd247e5";

#[test]
fn bundled_tokenizer_matches_pinned_sha256() {
  use sha2::{Digest, Sha256};
  let sha: String = Sha256::digest(clapkit::BUNDLED_TOKENIZER)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
  assert_eq!(
    sha, PINNED_TOKENIZER_SHA256,
    "clapkit::BUNDLED_TOKENIZER diverged from the pinned Xenova tokenizer textclap uses"
  );
  // Non-vacuity: the bundled artifact is the real ~2 MB tokenizer, not a stub.
  assert!(
    clapkit::BUNDLED_TOKENIZER.len() > 1_000_000,
    "bundled tokenizer is implausibly small ({} bytes)",
    clapkit::BUNDLED_TOKENIZER.len()
  );
}
