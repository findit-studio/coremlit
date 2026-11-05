//! The tokenizer image, format 1: Gemma's BPE tables, compiled from
//! `tokenizer.json` at staging into one little-endian file that the tokenizer
//! reads through a shared read-only map.
//!
//! # Layout
//!
//! A 4,096-byte header, then eight sections, each 8-byte aligned:
//!
//! | offset | bytes | field |
//! |---:|---:|---|
//! | 0 | 8 | magic `b"CMLGBPE\0"` |
//! | 8 | 4 | format version, [`TOKENIZER_IMAGE_VERSION`] |
//! | 12 | 4 | header size, 4,096 |
//! | 16 | 32 | the SHA-256 of the `tokenizer.json` compiled |
//! | 48 | 4 × 5 | counts: vocabulary, merges, added tokens, single-scalar tokens, bridging merges |
//! | 68 | 16 × 8 | section table: `(offset u64, length u64)` per section |
//! | 196 | 32 | the SHA-256 of bytes `[4096, file length)` |
//! | 228 | — | zeros to 4,096 |
//!
//! | # | section | layout |
//! |---:|---|---|
//! | 0 | token arena | every token's UTF-8 bytes, in id order |
//! | 1 | token offsets | `u32 × (vocabulary + 1)` into the arena |
//! | 2 | scalar index | open addressing, a power of two of `(u32 scalar, u32 id)` slots, empty = `u32::MAX`; Fibonacci hashing, linear probing |
//! | 3 | merge offsets | CSR over the left id: `u32 × (vocabulary + 1)` into the entries |
//! | 4 | merge entries | `(right u32, rank u32, merged u32)`, sorted by right within each left |
//! | 5 | byte-fallback ids | `u32 × 256`, the ids of `<0x00>` … `<0xFF>` |
//! | 6 | added tokens | their ids, longest literal first, ties by id |
//! | 7 | bridging merges | `(left u32, right u32)`: merges whose left token ends in a non-`▁` scalar and whose right token begins with `▁` |
//!
//! Every read is bounds-checked safe Rust over the mapped bytes, so a
//! malformed image can be refused or misread, and never read outside its map.
//!
//! # Versioning
//!
//! [`TOKENIZER_IMAGE_VERSION`] changes with any change to the layout, the
//! hash, a section's meaning or the compiler's checks; an image of another
//! version is refused by name, and `compile_artifacts` builds a new one under
//! its new name. The image is derived and local, so a version bump costs one
//! rebuild.

// The tokenizer reads this image as it lands; until then the tests are its
// only reader.
#![cfg_attr(not(test), allow(dead_code))]

use std::{path::Path, sync::Arc};

use super::super::{
  error::{Artifact, Error, ImageField, Result, TokenizerImageDefect},
  map::{Access, ReadOnlyMap},
};

/// The image format this build reads and writes.
pub(crate) const TOKENIZER_IMAGE_VERSION: u32 = 1;

/// The image's magic bytes.
pub(crate) const MAGIC: [u8; 8] = *b"CMLGBPE\0";

/// The header's size, and the offset of the first section.
pub(crate) const HEADER_SIZE: usize = 4_096;

/// The header fields' offsets.
pub(crate) mod header {
  pub(crate) const VERSION: usize = 8;
  pub(crate) const HEADER_SIZE: usize = 12;
  pub(crate) const SOURCE_DIGEST: usize = 16;
  pub(crate) const COUNTS: usize = 48;
  pub(crate) const SECTIONS: usize = 68;
  pub(crate) const IMAGE_DIGEST: usize = 196;
  pub(crate) const END: usize = 228;
}

/// The image's eight sections, in table order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Section {
  Arena = 0,
  TokenOffsets = 1,
  ScalarIndex = 2,
  MergeOffsets = 3,
  MergeEntries = 4,
  ByteFallback = 5,
  AddedTokens = 6,
  Bridges = 7,
}

/// The five counts the header carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counts {
  pub(crate) vocab: u32,
  pub(crate) merges: u32,
  pub(crate) added: u32,
  pub(crate) scalars: u32,
  pub(crate) bridges: u32,
}

impl Counts {
  pub(crate) const fn as_array(&self) -> [u32; 5] {
    [
      self.vocab,
      self.merges,
      self.added,
      self.scalars,
      self.bridges,
    ]
  }
}

/// What an open requires the image to be compiled from: the `tokenizer.json`
/// digest, which is its identity, and the counts that follow from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Identity {
  pub(crate) source: [u8; 32],
  pub(crate) counts: Counts,
}

/// `google/embeddinggemma-2`'s `tokenizer.json` (byte-identical in
/// FluidInference's tree): SHA-256 `4d777ef5…28be4`, 262,144 tokens, 514,906
/// merges, 24 added tokens, 19,227 single-scalar tokens, one bridging merge.
pub(crate) const UPSTREAM: Identity = Identity {
  source: hex32("4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
  counts: Counts {
    vocab: 262_144,
    merges: 514_906,
    added: 24,
    scalars: 19_227,
    bridges: 1,
  },
};

/// 64 lowercase hex digits as 32 bytes, at compile time; anything else does
/// not compile.
pub(crate) const fn hex32(text: &str) -> [u8; 32] {
  const fn digit(c: u8) -> u8 {
    match c {
      b'0'..=b'9' => c - b'0',
      b'a'..=b'f' => c - b'a' + 10,
      _ => panic!("not a lowercase hex digit"),
    }
  }
  let bytes = text.as_bytes();
  assert!(bytes.len() == 64, "not 64 hex digits");
  let mut out = [0u8; 32];
  let mut i = 0;
  while i < 32 {
    out[i] = digit(bytes[2 * i]) << 4 | digit(bytes[2 * i + 1]);
    i += 1;
  }
  out
}

/// The Fibonacci multiplier the scalar index hashes with.
pub(crate) const FIBONACCI: u64 = 0x9E37_79B9_7F4A_7C15;

/// The slot a scalar's probe starts at, in an index of `2^bits` slots.
#[inline]
pub(crate) const fn scalar_slot(scalar: u32, bits: u32) -> usize {
  ((scalar as u64).wrapping_mul(FIBONACCI) >> (64 - bits)) as usize
}

/// A tokenizer image, mapped and checked.
#[derive(Debug, Clone)]
pub(crate) struct TokenizerImage {
  map: Arc<ReadOnlyMap>,
  sections: [(usize, usize); 8],
  counts: Counts,
  scalar_bits: u32,
}

fn defect(field: ImageField, expected: impl Into<String>, actual: impl Into<String>) -> Error {
  Error::TokenizerImage(TokenizerImageDefect::new(
    field,
    expected.into(),
    actual.into(),
  ))
}

fn hex(bytes: &[u8]) -> String {
  bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The little-endian u32 at byte `at`, if the bytes are there.
#[inline]
fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
  let four = bytes.get(at..at.checked_add(4)?)?;
  Some(u32::from_le_bytes([four[0], four[1], four[2], four[3]]))
}

#[inline]
fn u64_at(bytes: &[u8], at: usize) -> Option<u64> {
  let eight = bytes.get(at..at.checked_add(8)?)?;
  let mut word = [0u8; 8];
  word.copy_from_slice(eight);
  Some(u64::from_le_bytes(word))
}

impl TokenizerImage {
  /// Maps the image at `path` and checks, in order: the magic, the format
  /// version, the header size, every section inside the file and aligned,
  /// the five counts, and the source digest against `identity`.
  ///
  /// There is no whole-image hash here: it would touch all of a lazily mapped
  /// file. `compile_artifacts` checks the image digest whenever it keeps an
  /// existing image.
  ///
  /// # Errors
  /// [`Error::Read`]; [`Error::TokenizerImage`] naming the first field that
  /// is wrong.
  pub(crate) fn open(path: &Path, identity: &Identity) -> Result<Self> {
    Self::check(
      ReadOnlyMap::open(Artifact::TokenizerImage, path, Access::Random)?,
      identity,
    )
  }

  fn check(map: Arc<ReadOnlyMap>, identity: &Identity) -> Result<Self> {
    let bytes = map.bytes();
    if bytes.len() < HEADER_SIZE {
      return Err(defect(
        ImageField::Length,
        format!("at least the {HEADER_SIZE}-byte header"),
        format!("{} bytes", bytes.len()),
      ));
    }
    if bytes[..8] != MAGIC {
      return Err(defect(ImageField::Magic, hex(&MAGIC), hex(&bytes[..8])));
    }
    let version = u32_at(bytes, header::VERSION).unwrap_or(0);
    if version != TOKENIZER_IMAGE_VERSION {
      return Err(defect(
        ImageField::FormatVersion,
        TOKENIZER_IMAGE_VERSION.to_string(),
        version.to_string(),
      ));
    }
    let header_size = u32_at(bytes, header::HEADER_SIZE).unwrap_or(0);
    if header_size as usize != HEADER_SIZE {
      return Err(defect(
        ImageField::HeaderSize,
        HEADER_SIZE.to_string(),
        header_size.to_string(),
      ));
    }
    let mut sections = [(0usize, 0usize); 8];
    for (index, section) in sections.iter_mut().enumerate() {
      let at = header::SECTIONS + index * 16;
      let (offset, length) = (
        u64_at(bytes, at).unwrap_or(u64::MAX),
        u64_at(bytes, at + 8).unwrap_or(u64::MAX),
      );
      let fits = offset >= HEADER_SIZE as u64
        && offset % 8 == 0
        && offset
          .checked_add(length)
          .is_some_and(|end| end <= bytes.len() as u64);
      if !fits {
        return Err(defect(
          ImageField::Section,
          format!(
            "section {index} 8-byte aligned inside the {}-byte file",
            bytes.len()
          ),
          format!("offset {offset}, length {length}"),
        ));
      }
      *section = (offset as usize, length as usize);
    }
    let counts = Counts {
      vocab: u32_at(bytes, header::COUNTS).unwrap_or(0),
      merges: u32_at(bytes, header::COUNTS + 4).unwrap_or(0),
      added: u32_at(bytes, header::COUNTS + 8).unwrap_or(0),
      scalars: u32_at(bytes, header::COUNTS + 12).unwrap_or(0),
      bridges: u32_at(bytes, header::COUNTS + 16).unwrap_or(0),
    };
    if counts != identity.counts {
      return Err(defect(
        ImageField::Count,
        format!("{:?}", identity.counts.as_array()),
        format!("{:?}", counts.as_array()),
      ));
    }
    let source = &bytes[header::SOURCE_DIGEST..header::SOURCE_DIGEST + 32];
    if source != identity.source {
      return Err(defect(
        ImageField::SourceDigest,
        hex(&identity.source),
        hex(source),
      ));
    }
    let image = Self {
      sections,
      counts,
      scalar_bits: 0,
      map,
    };
    image.check_section_lengths()
  }

  /// Each section's length must be what the counts make it, and the scalar
  /// index a power of two of slots.
  fn check_section_lengths(mut self) -> Result<Self> {
    let vocab = self.counts.vocab as usize;
    let expect = |section: Section, want: usize, got: usize| -> Result<()> {
      if want == got {
        Ok(())
      } else {
        Err(defect(
          ImageField::Section,
          format!("section {} of {want} bytes", section as usize),
          format!("{got} bytes"),
        ))
      }
    };
    let len = |section: Section| self.sections[section as usize].1;
    expect(
      Section::TokenOffsets,
      (vocab + 1) * 4,
      len(Section::TokenOffsets),
    )?;
    expect(
      Section::MergeOffsets,
      (vocab + 1) * 4,
      len(Section::MergeOffsets),
    )?;
    expect(
      Section::MergeEntries,
      self.counts.merges as usize * 12,
      len(Section::MergeEntries),
    )?;
    expect(Section::ByteFallback, 256 * 4, len(Section::ByteFallback))?;
    expect(
      Section::AddedTokens,
      self.counts.added as usize * 4,
      len(Section::AddedTokens),
    )?;
    expect(
      Section::Bridges,
      self.counts.bridges as usize * 8,
      len(Section::Bridges),
    )?;
    let slots = len(Section::ScalarIndex) / 8;
    if len(Section::ScalarIndex) % 8 != 0
      || !slots.is_power_of_two()
      || slots <= self.counts.scalars as usize
    {
      return Err(defect(
        ImageField::Section,
        format!(
          "a scalar index of a power of two of slots, more than {}",
          self.counts.scalars
        ),
        format!("{} bytes", len(Section::ScalarIndex)),
      ));
    }
    self.scalar_bits = slots.trailing_zeros();
    Ok(self)
  }

  #[inline]
  fn section(&self, section: Section) -> &[u8] {
    let (offset, length) = self.sections[section as usize];
    &self.map.bytes()[offset..offset + length]
  }

  /// The image's counts.
  #[inline]
  pub(crate) const fn counts(&self) -> Counts {
    self.counts
  }

  /// Token `id`'s UTF-8 bytes, or `None` past the vocabulary or for a
  /// malformed offset.
  pub(crate) fn token(&self, id: u32) -> Option<&[u8]> {
    let offsets = self.section(Section::TokenOffsets);
    let start = u32_at(offsets, id as usize * 4)? as usize;
    let end = u32_at(offsets, id as usize * 4 + 4)? as usize;
    self.section(Section::Arena).get(start..end)
  }

  /// The id of the one-scalar token `scalar`, if the vocabulary has one.
  pub(crate) fn scalar_id(&self, scalar: char) -> Option<u32> {
    let index = self.section(Section::ScalarIndex);
    let slots = 1usize << self.scalar_bits;
    let want = scalar as u32;
    let mut slot = scalar_slot(want, self.scalar_bits);
    for _ in 0..slots {
      let key = u32_at(index, slot * 8)?;
      if key == u32::MAX {
        return None;
      }
      if key == want {
        return u32_at(index, slot * 8 + 4);
      }
      slot = (slot + 1) & (slots - 1);
    }
    None
  }

  /// The merge of `left` then `right`: its rank (0 is applied first) and the
  /// merged id, if the merges hold that pair.
  pub(crate) fn merge(&self, left: u32, right: u32) -> Option<(u32, u32)> {
    let offsets = self.section(Section::MergeOffsets);
    let start = u32_at(offsets, left as usize * 4)? as usize;
    let end = u32_at(offsets, left as usize * 4 + 4)? as usize;
    let entries = self
      .section(Section::MergeEntries)
      .get(start * 12..end * 12)?;
    let (mut lo, mut hi) = (0usize, entries.len() / 12);
    while lo < hi {
      let mid = lo + (hi - lo) / 2;
      let candidate = u32_at(entries, mid * 12)?;
      match candidate.cmp(&right) {
        core::cmp::Ordering::Less => lo = mid + 1,
        core::cmp::Ordering::Greater => hi = mid,
        core::cmp::Ordering::Equal => {
          return Some((
            u32_at(entries, mid * 12 + 4)?,
            u32_at(entries, mid * 12 + 8)?,
          ));
        }
      }
    }
    None
  }

  /// The id of the byte-fallback token `<0xNN>` for `byte`.
  #[inline]
  pub(crate) fn byte_id(&self, byte: u8) -> Option<u32> {
    u32_at(self.section(Section::ByteFallback), usize::from(byte) * 4)
  }

  /// The added tokens' ids, longest literal first.
  pub(crate) fn added(&self) -> impl Iterator<Item = u32> + '_ {
    self
      .section(Section::AddedTokens)
      .chunks_exact(4)
      .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
  }

  /// The bridging merges, as `(left, right)` ids.
  pub(crate) fn bridges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
    self.section(Section::Bridges).chunks_exact(8).map(|w| {
      (
        u32::from_le_bytes([w[0], w[1], w[2], w[3]]),
        u32::from_le_bytes([w[4], w[5], w[6], w[7]]),
      )
    })
  }
}

#[cfg(test)]
mod tests;
