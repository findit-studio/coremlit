//! The committed mini tokenizer's image, compiled and opened, for the kit's
//! hermetic laws.

use std::io::Write;

use sha2::{Digest, Sha256};

use super::{super::compile::compile_image, Counts, Identity, TokenizerImage};

/// The committed mini `tokenizer.json`.
pub(crate) const MINI: &[u8] =
  include_bytes!("../../../../../../tests/embeddinggemma2/fixtures/mini_tokenizer.json");

/// The mini fixture's identity: its source digest and its counts.
pub(crate) fn mini_identity() -> Identity {
  Identity {
    source: Sha256::digest(MINI).into(),
    counts: Counts {
      vocab: 286,
      merges: 9,
      added: 8,
      scalars: 13,
      bridges: 1,
    },
  }
}

/// The mini image, compiled to a temporary file and opened. The file lives as
/// long as the returned handle.
pub(crate) fn mini() -> (TokenizerImage, tempfile::NamedTempFile) {
  let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
  file
    .write_all(&compile_image(MINI).expect("the mini fixture compiles"))
    .expect("write");
  file.flush().expect("flush");
  let image = TokenizerImage::open(file.path(), &mini_identity()).expect("opens");
  (image, file)
}
