use half::f16;

use super::*;
use crate::embeddings::embeddinggemma2::{
  config::{BOA_ID, EOA_ID},
  table::{TokenTable, fixture, gather_cell},
};

/// `raw` as the token ids a tokenizer would mint.
fn ids(raw: &[u32]) -> Vec<TokenId> {
  raw.iter().copied().map(TokenId::from_vocab).collect()
}

/// The column the laws probe: one no sentinel cell of rows 0 and 2 uses.
const PROBE: usize = 5;

/// Row `id`'s probe cell, holding the bf16 bits `value`.
fn first_cell(id: u32, value: u16) -> (usize, usize, u16) {
  (id as usize, PROBE, value)
}

/// Row `id` gathered, whole.
fn gathered(table: &TokenTable, id: u32) -> [f16; HIDDEN_SIZE] {
  let mut row = [f16::ZERO; HIDDEN_SIZE];
  table
    .gather_id(TokenId::from_vocab(id), &mut row)
    .expect("a row");
  row
}

/// T4: under `Single` a sequence of `n` ids takes the smallest `embed_S` with
/// `S >= n`; under `Packed` every `n` up to 256 takes `pack_256`, a batch of
/// one included, and longer ones `embed_512`.
#[test]
fn a_sequence_takes_the_smallest_function_that_holds_it() {
  for n in 1..=512 {
    let expected = [32, 48, 64, 128, 256, 512]
      .into_iter()
      .find(|s| *s >= n)
      .expect("n <= 512");
    let single = choose(n, Batching::Single);
    assert_eq!(single.tokens(), expected, "Single, n = {n}");
    assert_ne!(single, TextFunction::Pack256);
    let packed = choose(n, Batching::Packed);
    if n <= 256 {
      assert_eq!(packed, TextFunction::Pack256, "Packed, n = {n}");
    } else {
      assert_eq!(packed, TextFunction::Embed512, "Packed, n = {n}");
    }
  }
}

/// T5's layout: an `embed_S` input is the sequence's rows, then the pad row
/// (row 0, gathered and scaled like any row) in every row after it, and the
/// mask is 1 over the sequence and 0 after it.
#[test]
fn an_embed_input_is_the_rows_then_the_pad_row_and_its_mask() {
  let (_file, table) = fixture::token_table(&[
    first_cell(0, 0x3f80),
    first_cell(1, 0x4000),
    first_cell(2, 0x4040),
    first_cell(7, 0x4080),
  ]);
  let sequence_ids = ids(&[2, 7, 1]);
  let mut inputs = vec![f16::from_f32(9.0); 32 * HIDDEN_SIZE];
  let mut mask = vec![f16::from_f32(9.0); 32];
  fill_unpacked(
    &table,
    &Sequence::Text(&sequence_ids),
    &mut inputs,
    &mut mask,
  )
  .expect("fits");
  for (row, id) in [2u32, 7, 1].into_iter().enumerate() {
    let mut expected = [f16::ZERO; HIDDEN_SIZE];
    table
      .gather_id(TokenId::from_vocab(id), &mut expected)
      .expect("a row");
    assert_eq!(
      inputs[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE],
      expected,
      "row {row} is id {id}'s"
    );
  }
  let pad = gathered(&table, 0);
  assert_eq!(
    pad[PROBE].to_bits(),
    gather_cell(0x3f80).to_bits(),
    "the pad row is row 0, scaled"
  );
  for row in 3..32 {
    assert_eq!(
      inputs[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE],
      pad,
      "row {row} is the pad row"
    );
  }
  assert_eq!(&mask[..3], [f16::ONE; 3]);
  assert!(mask[3..].iter().all(|v| v.to_bits() == 0));
}

/// A media sequence is `<bos>`, the face's start token, the model's rows as
/// it wrote them (not scaled), the end token, `<eos>`; four rows longer than
/// its tokens.
#[test]
fn a_media_sequence_frames_its_rows_unscaled() {
  let (_file, table) = fixture::token_table(&[
    first_cell(BOS_ID, 0x4040),
    first_cell(EOS_ID, 0x4000),
    first_cell(BOA_ID, 0x40a0),
    first_cell(EOA_ID, 0x40c0),
  ]);
  let mut rows = vec![f16::ZERO; 2 * HIDDEN_SIZE];
  rows[PROBE] = f16::from_f32(0.5);
  rows[HIDDEN_SIZE + PROBE] = f16::from_f32(-0.25);
  let sequence = Sequence::Media {
    open: TokenId::from_vocab(BOA_ID),
    rows: &rows,
    close: TokenId::from_vocab(EOA_ID),
  };
  assert_eq!(sequence.len(), 6);
  let mut inputs = vec![f16::from_f32(9.0); 32 * HIDDEN_SIZE];
  let mut mask = vec![f16::ZERO; 32];
  fill_unpacked(&table, &sequence, &mut inputs, &mut mask).expect("fits");
  let firsts: Vec<u16> = (0..6)
    .map(|row| inputs[row * HIDDEN_SIZE + PROBE].to_bits())
    .collect();
  assert_eq!(
    firsts,
    [
      gather_cell(0x4040).to_bits(),
      gather_cell(0x40a0).to_bits(),
      f16::from_f32(0.5).to_bits(),
      f16::from_f32(-0.25).to_bits(),
      gather_cell(0x40c0).to_bits(),
      gather_cell(0x4000).to_bits(),
    ],
    "<bos> <|audio> the two rows unscaled <audio|> <eos>"
  );
  assert_eq!(&mask[..6], [f16::ONE; 6]);
  assert!(mask[6..].iter().all(|v| v.to_bits() == 0));
}

/// T6: a `pack_256` input holds each sequence's rows in slot order, then the
/// pad row; the bias is 0 inside each sequence's block and on each padding
/// row's own diagonal and -10,000 elsewhere; positions count within a
/// sequence and are 0 for padding; the pool is `f16(f32(1) / f32(n))` over
/// slot `s`'s rows.
#[test]
fn a_pack_input_is_its_blocks_positions_and_pool() {
  let (_file, table) = fixture::token_table(&[first_cell(0, 0x3f80), first_cell(7, 0x4080)]);
  let first = ids(&[2, 7, 1]);
  let second = ids(&[2, 1]);
  let mut bias = vec![f16::ONE; PACK_ROWS * PACK_ROWS];
  let mut inputs = vec![f16::ONE; PACK_ROWS * HIDDEN_SIZE];
  let mut pool = vec![f16::ONE; PACK_SLOTS * PACK_ROWS];
  let mut positions = vec![f16::ONE; PACK_ROWS];
  fill_packed(
    &table,
    &[Sequence::Text(&first), Sequence::Text(&second)],
    PackedInputs {
      attention_bias: &mut bias,
      inputs_embeds: &mut inputs,
      pool: &mut pool,
      positions: &mut positions,
    },
  )
  .expect("fits");

  let block = |row: usize| match row {
    0..=2 => Some(0..3),
    3..=4 => Some(3..5),
    _ => None,
  };
  for row in 0..PACK_ROWS {
    for column in 0..PACK_ROWS {
      let open = match block(row) {
        Some(range) => range.contains(&column),
        None => column == row,
      };
      let want = if open { f16::ZERO } else { MASKED };
      assert_eq!(
        bias[row * PACK_ROWS + column].to_bits(),
        want.to_bits(),
        "bias at ({row}, {column})"
      );
    }
  }
  let expected_positions: Vec<f32> = [0.0, 1.0, 2.0, 0.0, 1.0]
    .into_iter()
    .chain(core::iter::repeat_n(0.0, PACK_ROWS - 5))
    .collect();
  assert_eq!(
    positions.iter().map(|p| p.to_f32()).collect::<Vec<_>>(),
    expected_positions
  );
  let third = f16::from_f32(1.0f32 / 3.0f32);
  let half = f16::from_f32(1.0f32 / 2.0f32);
  for slot in 0..PACK_SLOTS {
    for row in 0..PACK_ROWS {
      let want = match (slot, row) {
        (0, 0..=2) => third,
        (1, 3..=4) => half,
        _ => f16::ZERO,
      };
      assert_eq!(
        pool[slot * PACK_ROWS + row].to_bits(),
        want.to_bits(),
        "pool at ({slot}, {row})"
      );
    }
  }
  assert_eq!(
    inputs[HIDDEN_SIZE + PROBE].to_bits(),
    gather_cell(0x4080).to_bits(),
    "row 1 is id 7"
  );
  let pad = gathered(&table, 0);
  for row in 5..PACK_ROWS {
    assert_eq!(
      inputs[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE],
      pad,
      "row {row} is the pad row"
    );
  }
}

/// T6b: a batch splits into packs first-fit in input order, at most eight
/// sequences and 256 rows per pack; a sequence past 256 rows is left for
/// `embed_512`; a batch of one is one pack, at slot 0. The audio and image
/// batches below are packed by the same rule, each on its own: a pack's
/// members come from one batch, so a pack call never mixes faces.
#[test]
fn a_batch_packs_first_fit_in_input_order() {
  let (packs, alone) = plan_packs(&[100, 200, 50, 300, 106, 1]);
  let members: Vec<&[usize]> = packs.iter().map(Pack::members).collect();
  assert_eq!(members, [&[0, 2, 4][..], &[1, 5][..]]);
  assert_eq!(packs.iter().map(Pack::rows).collect::<Vec<_>>(), [256, 201]);
  assert_eq!(alone, [3]);

  let (packs, alone) = plan_packs(&[3; 9]);
  let members: Vec<&[usize]> = packs.iter().map(Pack::members).collect();
  assert_eq!(
    members,
    [&[0, 1, 2, 3, 4, 5, 6, 7][..], &[8][..]],
    "eight sequences fill a pack"
  );
  assert!(alone.is_empty());

  let (packs, _) = plan_packs(&[17]);
  assert_eq!(packs.len(), 1, "a batch of one is packed");
  assert_eq!(packs[0].members(), [0], "alone, at slot 0");

  // Audio windows: a full window is 254 rows and fills a pack alone; two
  // partial ones share one.
  let (packs, _) = plan_packs(&[254, 91, 87, 254]);
  let members: Vec<&[usize]> = packs.iter().map(Pack::members).collect();
  assert_eq!(members, [&[0][..], &[1, 2][..], &[3][..]]);

  // Images at budget 140: 124 rows each, two to a pack; at 280, 284 rows,
  // none packs.
  let (packs, alone) = plan_packs(&[144, 144, 144, 284]);
  let members: Vec<&[usize]> = packs.iter().map(Pack::members).collect();
  assert_eq!(members, [&[0][..], &[1][..], &[2][..]]);
  assert_eq!(alone, [3]);
  let (packs, _) = plan_packs(&[124, 124, 124]);
  let members: Vec<&[usize]> = packs.iter().map(Pack::members).collect();
  assert_eq!(members, [&[0, 1][..], &[2][..]]);
}
