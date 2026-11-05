//! The two tables FluidInference lifted out of the graphs: the bf16 token
//! table every text function's input is gathered from, and the f16 patch
//! position tables the vision inputs are summed from.
//!
//! Each is one [`ReadOnlyMap`], opened once and shared by every instance
//! through a clone, and checked at open: its length against the model's
//! shape, and the sentinel cells its stamp recorded against the bytes now
//! mapped.

// The text and vision parts open these tables as they land; until then the
// tests are their only reader.
#![cfg_attr(not(test), allow(dead_code))]

use std::{path::Path, sync::Arc};

use half::{f16, slice::HalfFloatSliceExt};

use super::{
  config::{EMBED_SCALE, HIDDEN_SIZE, POSITION_ROWS, POSITION_TABLES, POSITION_WIDTH, VOCAB_SIZE},
  error::{
    Artifact, Error, ReadFailure, Result, StampDefect, StampField, TableCell, TableIdentity,
    TableShape,
  },
  map::{Access, ReadOnlyMap},
};

/// The token table's bytes: 262,144 rows of 512 bf16.
pub(crate) const TOKEN_TABLE_BYTES: u64 = (VOCAB_SIZE * HIDDEN_SIZE * 2) as u64;

/// The position tables' bytes: 2 tables of 1,024 rows of 768 f16.
pub(crate) const POSITION_TABLE_BYTES: u64 =
  (POSITION_TABLES * POSITION_ROWS * POSITION_WIDTH * 2) as u64;

/// The token-table cells a stamp records and an open compares. Each is a
/// different page of the 256 MiB file except the first six, which share row
/// 0 and row 2.
pub(crate) const TOKEN_SENTINEL_CELLS: [TableCell; 9] = [
  TableCell::token(0, 0),
  TableCell::token(0, 1),
  TableCell::token(0, 2),
  TableCell::token(0, 3),
  TableCell::token(2, 0),
  TableCell::token(2, 1),
  TableCell::token(506, 160),
  TableCell::token(236_743, 0),
  TableCell::token(262_143, 511),
];

/// Those cells' bits in FluidInference's table @ `3f978dd`: row 0's first
/// four, `<bos>`'s first two, the table's largest magnitude (−0.5234375), `▁`'s
/// first, and the last cell. The staged stamp must record exactly these.
pub(crate) const UPSTREAM_TOKEN_SENTINELS: [u16; 9] = [
  0xbbe5, 0xbca4, 0xbd26, 0xbc2d, 0x3dd2, 0xbd47, 0xbf06, 0x3d0c, 0x3aa5,
];

/// The position-table cells a stamp records: the last column of row 0 in
/// each table. Column 0 holds ±0 in both and would be a weak sentinel.
pub(crate) const POSITION_SENTINEL_CELLS: [TableCell; 2] = [
  TableCell::position(0, 0, 767),
  TableCell::position(1, 0, 767),
];

/// Those cells' bits in FluidInference's tables @ `3f978dd`.
pub(crate) const UPSTREAM_POSITION_SENTINELS: [u16; 2] = [0x2d68, 0xb040];

/// Checks `path`'s length against the model's `bytes` before anything is
/// mapped, checks that `sentinels` names exactly the kit's `cells`, maps the
/// file, and compares each recorded value against the mapped bytes. `offset`
/// places one of `cells` in the file.
fn open_checked(
  artifact: Artifact,
  path: &Path,
  bytes: u64,
  access: Access,
  cells: &[TableCell],
  sentinels: &[(TableCell, u16)],
  offset: impl Fn(TableCell) -> usize,
) -> Result<Arc<ReadOnlyMap>> {
  let len = std::fs::metadata(path)
    .map_err(|e| Error::Read(ReadFailure::new(artifact, path.to_path_buf(), e)))?
    .len();
  if len != bytes {
    return Err(Error::TableShape(TableShape::new(artifact, bytes, len)));
  }
  if !sentinels.iter().map(|(cell, _)| cell).eq(cells) {
    return Err(Error::Stamp(StampDefect::new(
      StampField::Files,
      format!(
        "the sentinels of {artifact} must be exactly the cells [{}], and the stamp recorded [{}]",
        Cells(cells.iter().copied()),
        Cells(sentinels.iter().map(|(cell, _)| *cell)),
      ),
    )));
  }
  let map = ReadOnlyMap::open(artifact, path, access)?;
  let mapped = map.bytes().len() as u64;
  if mapped != bytes {
    return Err(Error::TableShape(TableShape::new(artifact, bytes, mapped)));
  }
  for &(cell, expected) in sentinels {
    let at = offset(cell);
    let actual = u16::from_le_bytes([map.bytes()[at], map.bytes()[at + 1]]);
    if actual != expected {
      return Err(Error::TableIdentity(TableIdentity::new(
        artifact, cell, expected, actual,
      )));
    }
  }
  Ok(map)
}

/// `row 0, column 0; row 0, column 1; …`.
struct Cells<I>(I);

impl<I: Iterator<Item = TableCell> + Clone> core::fmt::Display for Cells<I> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    for (index, cell) in self.0.clone().enumerate() {
      if index > 0 {
        f.write_str("; ")?;
      }
      write!(f, "{cell}")?;
    }
    Ok(())
  }
}

/// Why [`TokenTable::gather_sequence`] could not lay out a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GatherRefusal {
  /// The table has no row for this id.
  NoRow(u32),
  /// `out` is not whole rows, or holds fewer rows than there are ids.
  Shape {
    /// The ids to gather.
    ids: usize,
    /// The f16 cells `out` holds.
    cells: usize,
  },
}

/// The scalar reference of the gather: one bf16 cell, widened to f32 by
/// shifting its bits up 16, scaled by f32(√512), and rounded to f16 (round to
/// nearest even, subnormals kept). The Swift reference computes exactly this.
#[inline]
pub(crate) fn gather_cell(bf16: u16) -> f16 {
  f16::from_f32(f32::from_bits(u32::from(bf16) << 16) * EMBED_SCALE)
}

/// The scalar path: [`gather_cell`] over one row's little-endian bf16 bytes.
/// `row` holds `out.len()` cells.
pub(crate) fn gather_row_scalar(row: &[u8], out: &mut [f16]) {
  debug_assert_eq!(row.len(), out.len() * 2, "a row of out.len() cells");
  for (cell, slot) in row.chunks_exact(2).zip(out.iter_mut()) {
    *slot = gather_cell(u16::from_le_bytes([cell[0], cell[1]]));
  }
}

/// The vector path, the one the kit runs: widen and scale a row into f32,
/// then round it to f16 with `half`'s slice conversion (`fcvtn`, round to
/// nearest even). Bit for bit [`gather_row_scalar`], which law T3 holds over
/// every bf16 pattern. `row` holds `out.len()` ≤ 512 cells.
pub(crate) fn gather_row(row: &[u8], out: &mut [f16]) {
  debug_assert_eq!(row.len(), out.len() * 2, "a row of out.len() cells");
  let mut wide = [0f32; HIDDEN_SIZE];
  let wide = &mut wide[..out.len()];
  for (cell, slot) in row.chunks_exact(2).zip(wide.iter_mut()) {
    *slot = f32::from_bits(u32::from(u16::from_le_bytes([cell[0], cell[1]])) << 16) * EMBED_SCALE;
  }
  out.convert_from_f32_slice(wide);
}

/// `embeddings.bf16`, mapped once and shared: 262,144 rows of 512 bf16.
///
/// A clone shares the one mapping (an `Arc` count), so N instances over one
/// table map it once.
#[derive(Debug, Clone)]
pub(crate) struct TokenTable {
  map: Arc<ReadOnlyMap>,
}

impl TokenTable {
  /// Refuses a file that is not the model's 268,435,456 bytes before mapping
  /// it, maps it for random reads (`MADV_RANDOM`), and compares the value the
  /// stamp recorded for each of [`TOKEN_SENTINEL_CELLS`].
  ///
  /// # Errors
  /// [`Error::Read`]; [`Error::TableShape`] for another length;
  /// [`Error::Stamp`] when `sentinels` is not exactly the kit's cells, in
  /// order (none, fewer, more or other); [`Error::TableIdentity`] naming the
  /// first cell whose bits differ.
  pub(crate) fn open(path: &Path, sentinels: &[(TableCell, u16)]) -> Result<Self> {
    let map = open_checked(
      Artifact::TokenTable,
      path,
      TOKEN_TABLE_BYTES,
      Access::Random,
      &TOKEN_SENTINEL_CELLS,
      sentinels,
      |cell| (cell.row() as usize * HIDDEN_SIZE + cell.column() as usize) * 2,
    )?;
    Ok(Self { map })
  }

  /// Row `id`'s 1,024 bytes, or `None` past the table.
  #[inline]
  pub(crate) fn row(&self, id: u32) -> Option<&[u8]> {
    let start = (id as usize).checked_mul(HIDDEN_SIZE * 2)?;
    self
      .map
      .bytes()
      .get(start..start.checked_add(HIDDEN_SIZE * 2)?)
  }

  /// Gathers `ids` into the first `ids.len()` rows of `out` (512 f16 each)
  /// and the pad row (row 0) into every row after them. `out` must be whole
  /// rows, at least one per id.
  ///
  /// # Errors
  /// [`GatherRefusal::Shape`] for a ragged or short `out`, before anything
  /// is written; [`GatherRefusal::NoRow`] for an id past the table, which the
  /// tokenizer's ids never are.
  pub(crate) fn gather_sequence(
    &self,
    ids: &[u32],
    out: &mut [f16],
  ) -> core::result::Result<(), GatherRefusal> {
    if out.len() % HIDDEN_SIZE != 0 || ids.len() > out.len() / HIDDEN_SIZE {
      return Err(GatherRefusal::Shape {
        ids: ids.len(),
        cells: out.len(),
      });
    }
    let pad = self.row(0).ok_or(GatherRefusal::NoRow(0))?;
    let mut rows = out.chunks_exact_mut(HIDDEN_SIZE);
    for (&id, slot) in ids.iter().zip(&mut rows) {
      gather_row(self.row(id).ok_or(GatherRefusal::NoRow(id))?, slot);
    }
    for slot in rows {
      gather_row(pad, slot);
    }
    Ok(())
  }

  /// Whether `other` reads the same mapping, not merely the same file.
  #[inline]
  pub(crate) fn same_mapping(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.map, &other.map)
  }
}

/// `position_embeddings.f16`, mapped once and shared: table 0 is the patch
/// column (x) embedding and table 1 the patch row (y) embedding, 1,024 rows of
/// 768 f16 each.
#[derive(Debug, Clone)]
pub(crate) struct PositionTable {
  map: Arc<ReadOnlyMap>,
}

impl PositionTable {
  /// Maps the tables at `path`, refuses a file that is not the model's
  /// 3,145,728 bytes, and compares each sentinel the stamp recorded.
  ///
  /// # Errors
  /// As [`TokenTable::open`], for [`Artifact::PositionTable`].
  pub(crate) fn open(path: &Path, sentinels: &[(TableCell, u16)]) -> Result<Self> {
    let map = open_checked(
      Artifact::PositionTable,
      path,
      POSITION_TABLE_BYTES,
      Access::Default,
      &POSITION_SENTINEL_CELLS,
      sentinels,
      |cell| {
        let table = usize::from(cell.table().unwrap_or(0));
        ((table * POSITION_ROWS + cell.row() as usize) * POSITION_WIDTH + cell.column() as usize)
          * 2
      },
    )?;
    Ok(Self { map })
  }

  /// Row `index` of table `table` (0 for x, 1 for y): 1,536 bytes of
  /// little-endian f16, or `None` past the tables.
  #[inline]
  pub(crate) fn row(&self, table: usize, index: usize) -> Option<&[u8]> {
    if table >= POSITION_TABLES || index >= POSITION_ROWS {
      return None;
    }
    let start = (table * POSITION_ROWS + index) * POSITION_WIDTH * 2;
    self.map.bytes().get(start..start + POSITION_WIDTH * 2)
  }

  /// Whether `other` reads the same mapping.
  #[inline]
  pub(crate) fn same_mapping(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.map, &other.map)
  }
}

// The tables are shared across instances and threads: N instances hold
// clones of one map.
const _: fn() = || {
  fn shared<T: Send + Sync>() {}
  shared::<TokenTable>();
  shared::<PositionTable>();
  shared::<Arc<ReadOnlyMap>>();
};

#[cfg(test)]
mod tests;
