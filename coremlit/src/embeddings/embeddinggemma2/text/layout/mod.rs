//! What the text functions are fed: the function a sequence takes, the packs a
//! batch splits into, and the bytes of each input.
//!
//! A sequence is either a text's ids or a media face's tokens framed by its
//! start and end tokens. Its rows are gathered from the token table (ids) or
//! copied as the media model wrote them (tokens, not scaled); rows past a
//! sequence are the pad row, row 0 gathered like any other.

use half::f16;

use super::{
  super::{
    alloc,
    config::{BOS_ID, EOS_ID, HIDDEN_SIZE, PACK_SLOTS},
    error::AllocationSite,
    options::Batching,
    table::{GatherRefusal, ShapeRefusal, TokenTable},
    tokenizer::bpe::TokenId,
  },
  TextFunction,
};

/// The rows of `pack_256`'s inputs.
pub(crate) const PACK_ROWS: usize = 256;

/// The bias outside a sequence's own block: Swift's `-10_000`, which `exp`
/// takes to zero in fp16.
pub(crate) const MASKED: f16 = f16::from_f32_const(-10_000.0);

/// The smallest `embed_S` that holds `n` ids; `embed_512` for any `n` past
/// 256. `n` is at most 512 by construction.
#[inline]
pub(crate) const fn unpacked(n: usize) -> TextFunction {
  if n <= 32 {
    TextFunction::Embed32
  } else if n <= 48 {
    TextFunction::Embed48
  } else if n <= 64 {
    TextFunction::Embed64
  } else if n <= 128 {
    TextFunction::Embed128
  } else if n <= 256 {
    TextFunction::Embed256
  } else {
    TextFunction::Embed512
  }
}

/// The function one sequence of `n` ids takes under `batching`: `pack_256`
/// for any `n` up to 256 under [`Batching::Packed`] (a batch of one
/// included), the smallest `embed_S` otherwise.
#[inline]
pub(crate) const fn choose(n: usize, batching: Batching) -> TextFunction {
  match batching {
    Batching::Packed if n <= PACK_ROWS => TextFunction::Pack256,
    _ => unpacked(n),
  }
}

/// One `pack_256` call: the batch indexes of its sequences, in slot order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Pack {
  members: Vec<usize>,
  rows: usize,
}

impl Pack {
  /// The batch indexes, slot 0 first.
  #[inline]
  pub(crate) fn members(&self) -> &[usize] {
    &self.members
  }

  /// The rows the members fill.
  #[cfg(test)]
  pub(crate) const fn rows(&self) -> usize {
    self.rows
  }
}

/// Splits a batch whose sequences have `lengths` into `pack_256` calls,
/// first-fit in input order: each sequence of at most 256 rows joins the
/// first pack with a free slot and room for its rows, or opens a new one. A
/// pack holds at most eight sequences and 256 rows. The indexes of longer
/// sequences come back apart, in input order, for `embed_512`.
///
/// Every list is reserved before it is filled: at most one pack per sequence,
/// at most every sequence apart, at most eight members a pack. So no push
/// grows a list.
///
/// # Errors
/// [`Error::Allocation`](super::super::error::Error::Allocation) naming the
/// pack layout if a list cannot be held.
pub(crate) fn plan_packs(
  lengths: &[usize],
) -> super::super::error::Result<(Vec<Pack>, Vec<usize>)> {
  let mut packs: Vec<Pack> = alloc::with_capacity(lengths.len(), AllocationSite::PackLayout)?;
  let mut alone = alloc::with_capacity(lengths.len(), AllocationSite::PackLayout)?;
  for (index, &length) in lengths.iter().enumerate() {
    if length > PACK_ROWS {
      alone.push(index);
      continue;
    }
    match packs
      .iter_mut()
      .find(|pack| pack.members.len() < PACK_SLOTS && pack.rows + length <= PACK_ROWS)
    {
      Some(pack) => {
        pack.members.push(index);
        pack.rows += length;
      }
      None => {
        let mut members = alloc::with_capacity(PACK_SLOTS, AllocationSite::PackLayout)?;
        members.push(index);
        packs.push(Pack {
          members,
          rows: length,
        });
      }
    }
  }
  Ok((packs, alone))
}

/// One sequence a text function embeds.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Sequence<'a> {
  /// A text's ids, `<bos>` and `<eos>` included.
  Text(&'a [TokenId]),
  /// A media face's tokens.
  Media(MediaSequence<'a>),
}

/// A media face's tokens: `<bos>`, `open`, the model's rows (whole rows of
/// 512, contiguous), `close`, `<eos>`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MediaSequence<'a> {
  /// The face's start token.
  pub(crate) open: TokenId,
  /// The face's rows.
  pub(crate) rows: &'a [f16],
  /// The face's end token.
  pub(crate) close: TokenId,
}

impl Sequence<'_> {
  /// The rows the sequence fills.
  #[inline]
  pub(crate) const fn len(&self) -> usize {
    match self {
      Self::Text(ids) => ids.len(),
      Self::Media(media) => media.rows.len() / HIDDEN_SIZE + 4,
    }
  }

  /// Writes the sequence's rows into the first [`Self::len`] rows of `out`.
  fn write(&self, table: &TokenTable, out: &mut [f16]) -> Result<(), GatherRefusal> {
    match self {
      Self::Text(ids) => {
        for (id, slot) in ids.iter().zip(out.as_chunks_mut::<HIDDEN_SIZE>().0) {
          table.gather_id(*id, slot)?;
        }
      }
      Self::Media(MediaSequence { open, rows, close }) => {
        let tokens = rows.len() / HIDDEN_SIZE;
        let (head, rest) = out.split_at_mut(2 * HIDDEN_SIZE);
        let (bos, start) = head.split_at_mut(HIDDEN_SIZE);
        table.gather_id(TokenId::from_vocab(BOS_ID), bos)?;
        table.gather_id(*open, start)?;
        let (body, tail) = rest.split_at_mut(tokens * HIDDEN_SIZE);
        body.copy_from_slice(&rows[..tokens * HIDDEN_SIZE]);
        let (end, eos) = tail.split_at_mut(HIDDEN_SIZE);
        table.gather_id(*close, end)?;
        table.gather_id(TokenId::from_vocab(EOS_ID), &mut eos[..HIDDEN_SIZE])?;
      }
    }
    Ok(())
  }
}

/// Fills an `embed_S` call: the sequence's rows, the pad row after them, and
/// the mask (1 over the sequence, 0 after it). `inputs_embeds` holds `S` rows
/// and `mask` `S` cells, with `sequence.len() <= S`.
pub(crate) fn fill_unpacked(
  table: &TokenTable,
  sequence: &Sequence<'_>,
  inputs_embeds: &mut [f16],
  mask: &mut [f16],
) -> Result<(), GatherRefusal> {
  let n = sequence.len();
  if inputs_embeds.len() != mask.len() * HIDDEN_SIZE || n > mask.len() {
    return Err(GatherRefusal::Shape(ShapeRefusal {
      ids: n,
      cells: inputs_embeds.len(),
    }));
  }
  match sequence {
    Sequence::Text(ids) => table.gather_sequence(ids, inputs_embeds)?,
    Sequence::Media(_) => {
      sequence.write(table, inputs_embeds)?;
      table.gather_pad(&mut inputs_embeds[n * HIDDEN_SIZE..])?;
    }
  }
  mask[..n].fill(f16::ONE);
  mask[n..].fill(f16::ZERO);
  Ok(())
}

/// `pack_256`'s four inputs, as the caller's buffers.
pub(crate) struct PackedInputs<'b> {
  /// `[1, 1, 256, 256]`.
  pub(crate) attention_bias: &'b mut [f16],
  /// `[1, 256, 512]`.
  pub(crate) inputs_embeds: &'b mut [f16],
  /// `[8, 256]`.
  pub(crate) pool: &'b mut [f16],
  /// `[256, 1]`.
  pub(crate) positions: &'b mut [f16],
}

/// Fills a `pack_256` call with `sequences` in slot order (at most eight, at
/// most 256 rows in all).
///
/// - `inputs_embeds`: each sequence's rows in order, then the pad row.
/// - `attention_bias`: [`MASKED`] everywhere but each sequence's own `n × n`
///   block and each padding row's own diagonal, which are 0, so a padding
///   row's softmax stays finite.
/// - `positions`: a row's index within its sequence, 0 for padding.
/// - `pool`: `f16(f32(1) / f32(n))` over slot `s`'s rows, 0 elsewhere; Swift's
///   formula.
pub(crate) fn fill_packed(
  table: &TokenTable,
  sequences: &[Sequence<'_>],
  inputs: PackedInputs<'_>,
) -> Result<(), GatherRefusal> {
  let PackedInputs {
    attention_bias,
    inputs_embeds,
    pool,
    positions,
  } = inputs;
  let total: usize = sequences.iter().map(Sequence::len).sum();
  if sequences.len() > PACK_SLOTS
    || total > PACK_ROWS
    || attention_bias.len() != PACK_ROWS * PACK_ROWS
    || inputs_embeds.len() != PACK_ROWS * HIDDEN_SIZE
    || pool.len() != PACK_SLOTS * PACK_ROWS
    || positions.len() != PACK_ROWS
  {
    return Err(GatherRefusal::Shape(ShapeRefusal {
      ids: total,
      cells: inputs_embeds.len(),
    }));
  }
  attention_bias.fill(MASKED);
  pool.fill(f16::ZERO);
  positions.fill(f16::ZERO);
  let mut offset = 0;
  for (slot, sequence) in sequences.iter().enumerate() {
    let n = sequence.len();
    sequence.write(table, &mut inputs_embeds[offset * HIDDEN_SIZE..])?;
    let weight = f16::from_f32(1.0 / n as f32);
    for row in offset..offset + n {
      attention_bias[row * PACK_ROWS + offset..row * PACK_ROWS + offset + n].fill(f16::ZERO);
      positions[row] = f16::from_f32((row - offset) as f32);
      pool[slot * PACK_ROWS + row] = weight;
    }
    offset += n;
  }
  table.gather_pad(&mut inputs_embeds[offset * HIDDEN_SIZE..])?;
  for row in offset..PACK_ROWS {
    attention_bias[row * PACK_ROWS + row] = f16::ZERO;
  }
  Ok(())
}

#[cfg(test)]
mod tests;
