//! A sparse token table for the kit's hermetic laws: the model's length, the
//! kit's sentinel cells holding distinct values, and the caller's cells; zero
//! everywhere else.

use std::io::{Seek, SeekFrom, Write};

use super::{
  super::{config::HIDDEN_SIZE, error::TableCell},
  POSITION_SENTINEL_CELLS, POSITION_TABLE_BYTES, PositionTable, TOKEN_SENTINEL_CELLS,
  TOKEN_TABLE_BYTES, TokenTable,
};

/// Distinct non-zero values for the nine sentinel cells.
const SENTINEL_VALUES: [u16; 9] = [
  0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x0888, 0x0999,
];

/// The sentinel list a stamp would record for the fixture.
pub(crate) fn sentinels() -> Vec<(TableCell, u16)> {
  TOKEN_SENTINEL_CELLS
    .iter()
    .copied()
    .zip(SENTINEL_VALUES)
    .collect()
}

/// A temporary token table holding `cells` (row, column, bf16 bits) beside
/// the sentinels, opened. The file lives as long as the returned handle.
pub(crate) fn token_table(cells: &[(usize, usize, u16)]) -> (tempfile::NamedTempFile, TokenTable) {
  let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
  file
    .as_file()
    .set_len(TOKEN_TABLE_BYTES)
    .expect("a sparse file");
  let sentinel_cells = TOKEN_SENTINEL_CELLS
    .iter()
    .zip(SENTINEL_VALUES)
    .map(|(cell, value)| (cell.row() as usize, cell.column() as usize, value));
  for (row, column, value) in sentinel_cells.chain(cells.iter().copied()) {
    let at = ((row * HIDDEN_SIZE + column) * 2) as u64;
    file.seek(SeekFrom::Start(at)).expect("seek");
    file.write_all(&value.to_le_bytes()).expect("write");
  }
  file.flush().expect("flush");
  let table = TokenTable::open(file.path(), &sentinels()).expect("the fixture opens");
  (file, table)
}

/// A temporary position table, zero but for its two sentinel cells, opened.
/// The file lives as long as the returned handle.
pub(crate) fn position_table() -> (tempfile::NamedTempFile, PositionTable) {
  const VALUES: [u16; 2] = [0x2d68, 0xb040];
  let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
  file
    .as_file()
    .set_len(POSITION_TABLE_BYTES)
    .expect("a sparse file");
  for (cell, value) in POSITION_SENTINEL_CELLS.iter().zip(VALUES) {
    let table = usize::from(cell.table().unwrap_or(0));
    let at = (((table * 1_024 + cell.row() as usize) * 768 + cell.column() as usize) * 2) as u64;
    file.seek(SeekFrom::Start(at)).expect("seek");
    file.write_all(&value.to_le_bytes()).expect("write");
  }
  file.flush().expect("flush");
  let sentinels: Vec<(TableCell, u16)> = POSITION_SENTINEL_CELLS
    .iter()
    .copied()
    .zip(VALUES)
    .collect();
  let table = PositionTable::open(file.path(), &sentinels).expect("the fixture opens");
  (file, table)
}
