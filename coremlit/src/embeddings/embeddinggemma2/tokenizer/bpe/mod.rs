//! The tokenizer's runtime: Hugging Face's Gemma BPE, exactly, over a
//! tokenizer image.
//!
//! 1. **Segmentation.** Under [`SplitSpecialTokens::Recognize`], the added
//!    literals are split out of the text first, leftmost-longest: at each
//!    `<`, the literals are tried longest first. Every literal begins with
//!    `<`, so this is `tokenizers`' Aho–Corasick `LeftmostLongest` for this
//!    set. Under [`SplitSpecialTokens::Split`] the whole text is one segment.
//! 2. **Normalization.** Each space becomes `▁`, keeping its one byte's span.
//!    Nothing else is normalized.
//! 3. **Pieces.** A segment is split before every `▁` that follows a non-`▁`
//!    scalar, except at a bridging site: where the left side ends with a
//!    bridging merge's left token and the right side begins with its right
//!    token. The compiler checked that BPE crosses no other such boundary, so
//!    BPE per piece is BPE over the whole segment.
//! 4. **BPE per piece.** A scalar is its one-scalar token, or its UTF-8
//!    bytes' fallback tokens, each spanning the whole character. The merge
//!    loop is `Word::merge_all`'s: the lowest rank first, the leftmost on ties.
//! 5. **Framing.** `<bos>` … `<eos>`, whose spans are empty.
//!
//! The prompt and the text are read as one string, so a literal or a merge
//! that spans the seam is found as it would be in the concatenation. Spans are
//! byte ranges of that joint string.

use core::{cmp::Ordering, ops::Range};
use std::collections::BinaryHeap;

use super::{
  super::{
    config::{BOS_ID, EOS_ID},
    error::{Allocation, AllocationSite, Error, Result},
    prompt::Prompt,
  },
  SplitSpecialTokens,
  image::TokenizerImage,
};

/// `▁`, the character every space becomes.
const SPACE_MARK: char = '\u{2581}';

/// A token id the tokenizer minted: always one of the image's ids, so always
/// a row of the token table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenId(u32);

impl TokenId {
  /// The id.
  #[inline]
  pub const fn get(self) -> u32 {
    self.0
  }

  /// An id the kit's compiled contract names (`<bos>`, a media face's start
  /// and end tokens), each a row of the table by that contract, or a test's.
  #[inline]
  pub(crate) const fn from_vocab(id: u32) -> Self {
    Self(id)
  }
}

/// One scalar of the joint string: the character and its byte range.
#[derive(Debug, Clone, Copy)]
struct Scalar {
  ch: char,
  start: u32,
  end: u32,
}

/// One symbol of a piece being merged, in a doubly linked list.
#[derive(Debug, Clone, Copy)]
struct Symbol {
  id: u32,
  prev: i32,
  next: i32,
  start: u32,
  end: u32,
  live: bool,
}

/// A candidate merge: the pair at `pos` and its successor, by `rank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
  rank: u32,
  pos: u32,
  merged: u32,
}

/// The lowest rank first, then the leftmost: `BinaryHeap` is a max-heap.
impl Ord for Candidate {
  fn cmp(&self, other: &Self) -> Ordering {
    other
      .rank
      .cmp(&self.rank)
      .then_with(|| other.pos.cmp(&self.pos))
  }
}

impl PartialOrd for Candidate {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

/// The tokenizer's reusable buffers, owned by the caller or the instance and
/// grown as the longest text so far needs, fallibly.
#[derive(Debug, Default)]
pub struct TokenizerScratch {
  scalars: Vec<Scalar>,
  symbols: Vec<Symbol>,
  heap: BinaryHeap<Candidate>,
}

impl TokenizerScratch {
  /// Empty buffers; they grow on first use.
  pub fn new() -> Self {
    Self::default()
  }
}

/// Grows `buffer` to hold `additional` more, or names the site.
fn reserve<T>(buffer: &mut Vec<T>, additional: usize, site: AllocationSite) -> Result<()> {
  buffer.try_reserve(additional).map_err(|_| {
    Error::Allocation(Allocation::new(
      site,
      additional.saturating_mul(core::mem::size_of::<T>()),
    ))
  })
}

/// What one encoding produced, written into the caller's buffers.
pub(crate) struct Encoded<'o> {
  pub(crate) ids: &'o mut Vec<TokenId>,
  pub(crate) spans: Option<&'o mut Vec<Range<usize>>>,
}

impl Encoded<'_> {
  fn push(&mut self, id: u32, span: Range<usize>) -> Result<()> {
    reserve(self.ids, 1, AllocationSite::TokenIds)?;
    self.ids.push(TokenId(id));
    if let Some(spans) = self.spans.as_deref_mut() {
      reserve(spans, 1, AllocationSite::TokenOffsets)?;
      spans.push(span);
    }
    Ok(())
  }
}

impl TokenizerImage {
  /// Encodes `prompt ⧺ text` as one string, framed by `<bos>` and `<eos>`,
  /// appending to `out`. Spans are byte ranges of the joint string; `<bos>`
  /// and `<eos>` have empty spans at 0.
  ///
  /// # Errors
  /// [`Error::Allocation`] if a buffer cannot grow.
  pub(crate) fn encode_joint(
    &self,
    prompt: &Prompt<'_>,
    text: &str,
    split: SplitSpecialTokens,
    scratch: &mut TokenizerScratch,
    mut out: Encoded<'_>,
  ) -> Result<()> {
    let pieces = prompt.pieces();
    let parts = [pieces[0], pieces[1], pieces[2], text];
    let total: usize = parts.iter().map(|p| p.chars().count()).sum();
    scratch.scalars.clear();
    reserve(
      &mut scratch.scalars,
      total,
      AllocationSite::TokenizerScratch,
    )?;
    let mut at = 0u32;
    for part in parts {
      for ch in part.chars() {
        let end = at + ch.len_utf8() as u32;
        scratch.scalars.push(Scalar { ch, start: at, end });
        at = end;
      }
    }
    out.push(BOS_ID, 0..0)?;
    let scalars = core::mem::take(&mut scratch.scalars);
    let result = self.encode_scalars(&scalars, split, scratch, &mut out);
    scratch.scalars = scalars;
    result?;
    out.push(EOS_ID, 0..0)
  }

  fn encode_scalars(
    &self,
    scalars: &[Scalar],
    split: SplitSpecialTokens,
    scratch: &mut TokenizerScratch,
    out: &mut Encoded<'_>,
  ) -> Result<()> {
    let mut segment = 0;
    let mut i = 0;
    while i < scalars.len() {
      if split == SplitSpecialTokens::Recognize
        && scalars[i].ch == '<'
        && let Some((id, len)) = self.special_at(&scalars[i..])
      {
        self.encode_segment(&scalars[segment..i], scratch, out)?;
        out.push(
          id,
          scalars[i].start as usize..scalars[i + len - 1].end as usize,
        )?;
        i += len;
        segment = i;
        continue;
      }
      i += 1;
    }
    self.encode_segment(&scalars[segment..], scratch, out)
  }

  /// The added literal at the start of `scalars`, longest first: its id and
  /// its length in scalars.
  fn special_at(&self, scalars: &[Scalar]) -> Option<(u32, usize)> {
    'literals: for id in self.added() {
      let literal = core::str::from_utf8(self.token(id)?).ok()?;
      let mut len = 0;
      for ch in literal.chars() {
        match scalars.get(len) {
          Some(scalar) if scalar.ch == ch => len += 1,
          _ => continue 'literals,
        }
      }
      return Some((id, len));
    }
    None
  }

  /// One text segment: normalized, cut into pieces, each piece BPE'd.
  fn encode_segment(
    &self,
    scalars: &[Scalar],
    scratch: &mut TokenizerScratch,
    out: &mut Encoded<'_>,
  ) -> Result<()> {
    let marked = |s: &Scalar| if s.ch == ' ' { SPACE_MARK } else { s.ch };
    let mut start = 0;
    for i in 1..scalars.len() {
      let boundary = marked(&scalars[i]) == SPACE_MARK && marked(&scalars[i - 1]) != SPACE_MARK;
      if boundary && !self.bridged(&scalars[..i], &scalars[i..]) {
        self.encode_piece(&scalars[start..i], scratch, out)?;
        start = i;
      }
    }
    if start < scalars.len() {
      self.encode_piece(&scalars[start..], scratch, out)?;
    }
    Ok(())
  }

  /// Whether a bridging merge can join the two sides of a word boundary: the
  /// left side ends with its left token and the right side begins with its
  /// right token.
  fn bridged(&self, left: &[Scalar], right: &[Scalar]) -> bool {
    let marked = |s: &Scalar| if s.ch == ' ' { SPACE_MARK } else { s.ch };
    self.bridges().any(|(l, r)| {
      let (Some(l), Some(r)) = (self.token(l), self.token(r)) else {
        return false;
      };
      let (Ok(l), Ok(r)) = (core::str::from_utf8(l), core::str::from_utf8(r)) else {
        return false;
      };
      let l_len = l.chars().count();
      left.len() >= l_len
        && left[left.len() - l_len..].iter().map(marked).eq(l.chars())
        && right
          .iter()
          .map(marked)
          .take(r.chars().count())
          .eq(r.chars())
    })
  }

  /// One piece: its scalars' initial symbols, then the merge loop, then the
  /// surviving symbols in order.
  fn encode_piece(
    &self,
    scalars: &[Scalar],
    scratch: &mut TokenizerScratch,
    out: &mut Encoded<'_>,
  ) -> Result<()> {
    if scalars.is_empty() {
      return Ok(());
    }
    let symbols = &mut scratch.symbols;
    symbols.clear();
    reserve(symbols, scalars.len() * 4, AllocationSite::TokenizerScratch)?;
    for scalar in scalars {
      let ch = if scalar.ch == ' ' {
        SPACE_MARK
      } else {
        scalar.ch
      };
      let push = |id: u32, symbols: &mut Vec<Symbol>| {
        let n = symbols.len() as i32;
        symbols.push(Symbol {
          id,
          prev: n - 1,
          next: n + 1,
          start: scalar.start,
          end: scalar.end,
          live: true,
        });
      };
      match self.scalar_id(ch) {
        Some(id) => push(id, symbols),
        None => {
          let mut utf8 = [0u8; 4];
          for byte in ch.encode_utf8(&mut utf8).bytes() {
            push(self.byte_id(byte).unwrap_or(0), symbols);
          }
        }
      }
    }
    if let Some(last) = symbols.last_mut() {
      last.next = -1;
    }
    let heap = &mut scratch.heap;
    heap.clear();
    for pos in 0..symbols.len().saturating_sub(1) {
      if let Some((rank, merged)) = self.merge(symbols[pos].id, symbols[pos + 1].id) {
        heap.push(Candidate {
          rank,
          pos: pos as u32,
          merged,
        });
      }
    }
    while let Some(top) = heap.pop() {
      let pos = top.pos as usize;
      if !symbols[pos].live || symbols[pos].next < 0 {
        continue;
      }
      let next = symbols[pos].next as usize;
      match self.merge(symbols[pos].id, symbols[next].id) {
        Some((_, merged)) if merged == top.merged => {}
        _ => continue,
      }
      symbols[pos].id = top.merged;
      symbols[pos].end = symbols[next].end;
      symbols[pos].next = symbols[next].next;
      symbols[next].live = false;
      if symbols[pos].next >= 0 {
        let after = symbols[pos].next as usize;
        symbols[after].prev = pos as i32;
        if let Some((rank, merged)) = self.merge(symbols[pos].id, symbols[after].id) {
          heap.push(Candidate {
            rank,
            pos: pos as u32,
            merged,
          });
        }
      }
      if symbols[pos].prev >= 0 {
        let before = symbols[pos].prev as usize;
        if let Some((rank, merged)) = self.merge(symbols[before].id, symbols[pos].id) {
          heap.push(Candidate {
            rank,
            pos: before as u32,
            merged,
          });
        }
      }
    }
    let mut cursor = 0i32;
    while cursor >= 0 && (cursor as usize) < symbols.len() {
      let symbol = symbols[cursor as usize];
      if symbol.live {
        out.push(symbol.id, symbol.start as usize..symbol.end as usize)?;
      }
      cursor = symbol.next;
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests;
