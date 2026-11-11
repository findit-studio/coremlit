//! The artifacts a part runs: which part, and the label that names them.

use sha2::{Digest, Sha256};

use super::{alloc, error::AllocationSite};

pub(crate) mod compile;
pub(crate) mod naming;
pub(crate) mod stamp;
pub(crate) mod tree;

/// One of the kit's three separable parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, derive_more::Display)]
#[display("{}", self.as_str())]
pub enum Part {
  /// The text part: the seven text functions, the token table, the tokenizer
  /// and `config.json`. Every modality's vector comes out of it.
  Text,
  /// The audio part: one audio bundle, fp16 or fp32.
  Audio,
  /// The vision part: the vision bundle and the position tables.
  Vision,
}

impl Part {
  /// Stable snake_case name of the part: `"text"`, `"audio"` or `"vision"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Text => "text",
      Self::Audio => "audio",
      Self::Vision => "vision",
    }
  }
}

/// Which artifacts a part runs: a published revision, or the digest of its
/// files.
///
/// Three spellings, the [`Display`](core::fmt::Display) and
/// [`FromStr`](core::str::FromStr) forms and the serde string alike:
/// - `fluidinference@3f978dd`: FluidInference's conversion at that revision,
///   when the part's files are exactly the pinned upstream set
///   ([`Self::FLUIDINFERENCE`]);
/// - `findit-studio@<7 hex>`: this crate's own conversion at that revision;
/// - `sha256:<12 hex>`: any other set of files, named by the first 12 hex of
///   the SHA-256 over its manifest lines ([`Self::from_digests`]).
///
/// The label is not an option: it is read from the part's stamp, and it enters
/// the part's derivation terms. A deployment may PIN the label it expects
/// (`expect_artifacts`), and a part whose stamp yields another one is refused
/// by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactLabel(Repr);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Repr {
  /// A publisher's revision, its first 7 lowercase hex digits.
  Revision(Revision),
  /// The first 12 lowercase hex digits of a manifest digest.
  Digest([u8; 12]),
}

/// A publisher and the first 7 lowercase hex digits of its revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Revision {
  publisher: Publisher,
  rev: [u8; 7],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Publisher {
  FluidInference,
  FinditStudio,
}

impl Publisher {
  const fn as_str(self) -> &'static str {
    match self {
      Self::FluidInference => "fluidinference",
      Self::FinditStudio => "findit-studio",
    }
  }
}

impl ArtifactLabel {
  /// `fluidinference@3f978dd`: FluidInference's `embeddinggemma-2-coreml` at
  /// revision `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`, the text, vision and
  /// fp16 audio artifacts this kit is pinned to.
  pub const FLUIDINFERENCE: Self = Self(Repr::Revision(Revision {
    publisher: Publisher::FluidInference,
    rev: *b"3f978dd",
  }));

  /// The `sha256:` label of a set of files, each given as its path relative to
  /// the part's directory and the SHA-256 of its bytes.
  ///
  /// Each file is one line `<64 lowercase hex><two spaces><path>\n`, the
  /// lines sorted bytewise by path: the order and shape of a
  /// `MODELS_LOCK.d/` manifest. The label is `sha256:` and the first 12 hex of
  /// the SHA-256 over those lines, so a reader recomputes it with `sort` and
  /// `shasum -a 256`.
  ///
  /// # Errors
  /// [`Error::Allocation`](super::Error::Allocation) at
  /// [`AllocationSite::StampDigests`] if the list the lines are sorted in
  /// cannot be reserved.
  pub fn from_digests<'p, I>(files: I) -> super::error::Result<Self>
  where
    I: IntoIterator<Item = (&'p str, [u8; 32])>,
  {
    let files = files.into_iter();
    let mut lines: Vec<(&'p str, [u8; 32])> = Vec::new();
    alloc::reserve(
      &mut lines,
      files.size_hint().0,
      AllocationSite::StampDigests,
    )?;
    for line in files {
      alloc::reserve(&mut lines, 1, AllocationSite::StampDigests)?;
      lines.push(line);
    }
    lines.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut manifest = Manifest::new();
    for (path, digest) in &lines {
      manifest.line(path, digest);
    }
    Ok(Self::from_manifest(&manifest.finish()))
  }

  /// The `sha256:` label of a manifest digest: its first 12 hex.
  pub(crate) fn from_manifest(digest: &[u8; 32]) -> Self {
    let mut full = [0u8; 64];
    hex_into(digest, &mut full);
    let mut prefix = [0u8; 12];
    prefix.copy_from_slice(&full[..12]);
    Self(Repr::Digest(prefix))
  }

  /// Whether this is a publisher's revision label rather than a `sha256:`
  /// digest label.
  #[inline]
  pub const fn is_revision(&self) -> bool {
    matches!(self.0, Repr::Revision(_))
  }
}

/// The SHA-256 over manifest-shaped lines `<sha256>  <path>\n`, the lines
/// given sorted bytewise by path (a stamp's maps hold them so): the digest a
/// `sha256:` label and a bundle's name take their first 12 hex from. Streams;
/// holds no list.
pub(crate) fn manifest_sha256<'p>(lines: impl IntoIterator<Item = (&'p str, &'p str)>) -> [u8; 32] {
  let mut hasher = Sha256::new();
  let mut previous: Option<&str> = None;
  for (path, sha) in lines {
    debug_assert!(
      previous.is_none_or(|p| p.as_bytes() <= path.as_bytes()),
      "manifest lines arrive sorted bytewise by path"
    );
    previous = Some(path);
    hasher.update(sha.as_bytes());
    hasher.update(b"  ");
    hasher.update(path.as_bytes());
    hasher.update(b"\n");
  }
  hasher.finalize().into()
}

/// A manifest digest built line by line, each line's digest given as bytes
/// and hashed as lowercase hex: the streaming form of [`manifest_sha256`].
pub(crate) struct Manifest(Sha256);

impl Manifest {
  pub(crate) fn new() -> Self {
    Self(Sha256::new())
  }

  /// The line `<64 lowercase hex>  <path>\n`; lines come sorted bytewise by
  /// path.
  pub(crate) fn line(&mut self, path: &str, digest: &[u8; 32]) {
    let mut text = [0u8; 64];
    hex_into(digest, &mut text);
    self.0.update(text);
    self.0.update(b"  ");
    self.0.update(path.as_bytes());
    self.0.update(b"\n");
  }

  pub(crate) fn finish(self) -> [u8; 32] {
    self.0.finalize().into()
  }
}

/// Lowercase hex of a 32-byte digest, into `out`.
pub(crate) fn hex_into(digest: &[u8; 32], out: &mut [u8; 64]) {
  const DIGITS: &[u8; 16] = b"0123456789abcdef";
  for (byte, pair) in digest.iter().zip(out.as_chunks_mut::<2>().0) {
    *pair = [
      DIGITS[usize::from(byte >> 4)],
      DIGITS[usize::from(byte & 0x0f)],
    ];
  }
}

/// Lowercase hex of a 32-byte digest.
pub(crate) fn hex(digest: &[u8; 32]) -> String {
  let mut text = [0u8; 64];
  hex_into(digest, &mut text);
  text.iter().map(|&digit| char::from(digit)).collect()
}

/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a major bump.**
///
/// `fluidinference@3f978dd`, `findit-studio@<7 hex>` or `sha256:<12 hex>`.
impl core::fmt::Display for ArtifactLabel {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let (head, tail): (&str, &[u8]) = match &self.0 {
      Repr::Revision(Revision { publisher, rev }) => (publisher.as_str(), rev),
      Repr::Digest(prefix) => ("sha256", prefix),
    };
    let separator = if self.is_revision() { '@' } else { ':' };
    write!(f, "{head}{separator}")?;
    // Every byte of `tail` is an ASCII hex digit, by construction.
    tail
      .iter()
      .try_for_each(|&digit| core::fmt::Write::write_char(f, char::from(digit)))
  }
}

/// The text was not one of [`ArtifactLabel`]'s three spellings.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
  "`{text}` is not an artifact label: expected `fluidinference@<7 hex>`, `findit-studio@<7 hex>` or `sha256:<12 hex>`, in lowercase"
)]
pub struct ParseArtifactLabelError {
  text: String,
}

impl ParseArtifactLabelError {
  /// The text that failed to parse.
  #[inline]
  pub fn text(&self) -> &str {
    &self.text
  }
}

impl core::str::FromStr for ArtifactLabel {
  type Err = ParseArtifactLabelError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    let refuse = || ParseArtifactLabelError { text: s.to_owned() };
    let lower_hex = |tail: &str| {
      tail
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    let revision = |publisher: Publisher, tail: &str| -> Result<Self, ParseArtifactLabelError> {
      let rev: [u8; 7] = tail.as_bytes().try_into().map_err(|_| refuse())?;
      if lower_hex(tail) {
        Ok(Self(Repr::Revision(Revision { publisher, rev })))
      } else {
        Err(refuse())
      }
    };
    if let Some(tail) = s.strip_prefix("fluidinference@") {
      revision(Publisher::FluidInference, tail)
    } else if let Some(tail) = s.strip_prefix("findit-studio@") {
      revision(Publisher::FinditStudio, tail)
    } else if let Some(tail) = s.strip_prefix("sha256:") {
      let prefix: [u8; 12] = tail.as_bytes().try_into().map_err(|_| refuse())?;
      if lower_hex(tail) {
        Ok(Self(Repr::Digest(prefix)))
      } else {
        Err(refuse())
      }
    } else {
      Err(refuse())
    }
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for ArtifactLabel {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(self)
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ArtifactLabel {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    let text = <String as serde::Deserialize>::deserialize(deserializer)?;
    text.parse().map_err(serde::de::Error::custom)
  }
}

#[cfg(test)]
mod tests;
