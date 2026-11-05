use std::{
  io::{Seek, SeekFrom, Write},
  path::PathBuf,
};

use super::*;

/// The committed gather golden: 65,536 little-endian f16, the gather of every
/// bf16 pattern in pattern order, computed by numpy.
const GATHER_GOLDEN: &[u8] =
  include_bytes!("../../../../tests/embeddinggemma2/fixtures/gather_f16.bin");

/// A sparse temporary file of `len` bytes holding `cells` (byte offset, u16)
/// and zeros everywhere else.
fn file_with(len: u64, cells: &[(usize, u16)]) -> tempfile::NamedTempFile {
  let mut file = tempfile::NamedTempFile::new().expect("a temporary file");
  file.as_file().set_len(len).expect("a sparse file");
  for &(at, value) in cells {
    file.seek(SeekFrom::Start(at as u64)).expect("seek");
    file.write_all(&value.to_le_bytes()).expect("write");
  }
  file.flush().expect("flush");
  file
}

fn token_offset(row: usize, column: usize) -> usize {
  (row * HIDDEN_SIZE + column) * 2
}

fn position_offset(table: usize, row: usize, column: usize) -> usize {
  ((table * POSITION_ROWS + row) * POSITION_WIDTH + column) * 2
}

/// Values for the two position sentinel cells, and the list recording them.
const POSITION_VALUES: [u16; 2] = [0x2d68, 0xb040];

fn recorded_position_sentinels() -> Vec<(TableCell, u16)> {
  POSITION_SENTINEL_CELLS
    .iter()
    .copied()
    .zip(POSITION_VALUES)
    .collect()
}

fn position_table_file(extra: &[(usize, u16)]) -> tempfile::NamedTempFile {
  let mut cells: Vec<(usize, u16)> = POSITION_SENTINEL_CELLS
    .iter()
    .zip(POSITION_VALUES)
    .map(|(cell, value)| {
      let table = usize::from(cell.table().unwrap());
      (
        position_offset(table, cell.row() as usize, cell.column() as usize),
        value,
      )
    })
    .collect();
  cells.extend_from_slice(extra);
  file_with(POSITION_TABLE_BYTES, &cells)
}

/// Distinct non-zero values for the nine token sentinel cells.
const SENTINEL_VALUES: [u16; 9] = [
  0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x0888, 0x0999,
];

fn recorded_token_sentinels() -> Vec<(TableCell, u16)> {
  TOKEN_SENTINEL_CELLS
    .iter()
    .copied()
    .zip(SENTINEL_VALUES)
    .collect()
}

fn token_table_file(extra: &[(usize, u16)]) -> tempfile::NamedTempFile {
  let mut cells: Vec<(usize, u16)> = TOKEN_SENTINEL_CELLS
    .iter()
    .zip(SENTINEL_VALUES)
    .map(|(cell, value)| {
      (
        token_offset(cell.row() as usize, cell.column() as usize),
        value,
      )
    })
    .collect();
  cells.extend_from_slice(extra);
  file_with(TOKEN_TABLE_BYTES, &cells)
}

/// T3: the vector path, the scalar path and the committed numpy golden agree
/// bit for bit on every finite bf16 pattern (±0, the subnormal results, those
/// that round to zero and the 29,846 that overflow to ±∞), and every NaN
/// pattern maps to a NaN.
#[test]
fn the_gather_is_the_golden_on_every_bf16_pattern() {
  let patterns: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
  let golden: Vec<u16> = GATHER_GOLDEN
    .chunks_exact(2)
    .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
    .collect();
  assert_eq!(golden.len(), 65_536);
  let mut vector = vec![f16::ZERO; 65_536];
  let mut scalar = vec![f16::ZERO; 65_536];
  for ((row, v), s) in patterns
    .chunks_exact(HIDDEN_SIZE * 2)
    .zip(vector.chunks_exact_mut(HIDDEN_SIZE))
    .zip(scalar.chunks_exact_mut(HIDDEN_SIZE))
  {
    gather_row(row, v);
    gather_row_scalar(row, s);
  }
  let (mut finite, mut nan, mut overflow) = (0, 0, 0);
  for (pattern, ((v, s), g)) in vector.iter().zip(&scalar).zip(&golden).enumerate() {
    let input = f32::from_bits((pattern as u32) << 16);
    if input.is_nan() {
      nan += 1;
      assert!(
        v.is_nan() && s.is_nan() && f16::from_bits(*g).is_nan(),
        "pattern {pattern:#06x}"
      );
      continue;
    }
    assert_eq!(v.to_bits(), *g, "the vector path at pattern {pattern:#06x}");
    assert_eq!(s.to_bits(), *g, "the scalar path at pattern {pattern:#06x}");
    if input.is_finite() {
      finite += 1;
      overflow += usize::from(v.is_infinite());
    }
  }
  assert_eq!((finite, nan, overflow), (65_280, 254, 29_846));
}

/// A table whose bytes hold the recorded sentinels opens; one whose recorded
/// cell differs is refused, naming the cell and both values.
#[test]
fn a_sentinel_the_file_does_not_hold_is_refused_naming_the_cell() {
  let file = token_table_file(&[]);
  TokenTable::open(file.path(), &recorded_token_sentinels()).expect("the recorded sentinels hold");

  let mut wrong = recorded_token_sentinels();
  wrong[6].1 ^= 0x8000;
  match TokenTable::open(file.path(), &wrong) {
    Err(Error::TableIdentity(identity)) => {
      assert_eq!(identity.artifact(), Artifact::TokenTable);
      assert_eq!(identity.cell(), TableCell::token(506, 160));
      assert_eq!(
        (identity.expected(), identity.actual()),
        (0x7777 ^ 0x8000, 0x7777)
      );
    }
    other => panic!("expected TableIdentity, got {other:?}"),
  }

  let outside = [(TableCell::token(262_144, 0), 0)];
  assert!(matches!(
    TokenTable::open(file.path(), &outside),
    Err(Error::Stamp(_))
  ));
}

/// The stamp's sentinel list must be exactly the kit's cells, in order: an
/// empty, partial, longer or other list is refused by name before anything
/// is mapped, for both tables.
#[test]
fn a_sentinel_list_other_than_the_kits_cells_is_refused() {
  let file = token_table_file(&[]);
  let full = recorded_token_sentinels();
  let mut other = full.clone();
  other[8].0 = TableCell::token(262_143, 510);
  let mut longer = full.clone();
  longer.push((TableCell::token(1, 1), 0));
  for (what, list) in [
    ("none", Vec::new()),
    ("fewer", full[..8].to_vec()),
    ("more", longer),
    ("other", other),
    ("reordered", full.iter().rev().copied().collect()),
  ] {
    match TokenTable::open(file.path(), &list) {
      Err(Error::Stamp(defect)) => {
        assert_eq!(defect.field(), StampField::Files, "{what}");
        assert!(
          defect.detail().contains("must be exactly the cells"),
          "{what}: {}",
          defect.detail()
        );
      }
      other => panic!("{what}: expected a stamp defect, got {other:?}"),
    }
  }
  let positions = position_table_file(&[]);
  assert!(matches!(
    PositionTable::open(positions.path(), &[]),
    Err(Error::Stamp(_))
  ));
  PositionTable::open(positions.path(), &recorded_position_sentinels())
    .expect("exactly the cells opens");
}

/// A table of another length is refused by name, with both lengths.
#[test]
fn a_table_of_another_length_is_refused() {
  let short = file_with(TOKEN_TABLE_BYTES - 2, &[]);
  match TokenTable::open(short.path(), &[]) {
    Err(Error::TableShape(shape)) => {
      assert_eq!(
        (shape.expected(), shape.actual()),
        (268_435_456, 268_435_454)
      );
    }
    other => panic!("expected TableShape, got {other:?}"),
  }
  let positions = file_with(POSITION_TABLE_BYTES + 2, &[]);
  match PositionTable::open(positions.path(), &[]) {
    Err(Error::TableShape(shape)) => assert_eq!(shape.artifact(), Artifact::PositionTable),
    other => panic!("expected TableShape, got {other:?}"),
  }
}

/// Every clone of a table reads the one mapping; a second open maps again.
#[test]
fn every_clone_reads_one_mapping() {
  let file = token_table_file(&[]);
  let table = TokenTable::open(file.path(), &recorded_token_sentinels()).expect("opens");
  let (a, b) = (table.clone(), table.clone());
  assert!(table.same_mapping(&a) && a.same_mapping(&b));
  assert_eq!(a.row(7).unwrap().as_ptr(), table.row(7).unwrap().as_ptr());
  assert!(table.row(VOCAB_SIZE as u32 - 1).is_some() && table.row(VOCAB_SIZE as u32).is_none());
  let again = TokenTable::open(file.path(), &recorded_token_sentinels()).expect("opens");
  assert!(!again.same_mapping(&table));

  let positions = position_table_file(&[(position_offset(0, 3, 767), 0x2d68)]);
  let p = PositionTable::open(positions.path(), &recorded_position_sentinels()).expect("opens");
  assert!(p.same_mapping(&p.clone()));
  let row = p.row(0, 3).expect("row 3 of table 0");
  assert_eq!(
    (row.len(), u16::from_le_bytes([row[1534], row[1535]])),
    (1536, 0x2d68)
  );
  assert!(p.row(1, 1023).is_some() && p.row(1, 1024).is_none() && p.row(2, 0).is_none());
}

/// T5's hermetic half: a sequence is its own rows, then the pad row (row 0,
/// gathered like any row) in every row after it; an id past the table is
/// named.
#[test]
fn a_sequence_is_its_rows_then_the_pad_row() {
  let file = token_table_file(&[
    (token_offset(0, 5), 0x3f80),
    (token_offset(7, 0), 0x4000),
    (token_offset(9, 511), 0xc040),
  ]);
  let table = TokenTable::open(file.path(), &recorded_token_sentinels()).expect("opens");
  let mut out = vec![f16::from_f32(9.0); 4 * HIDDEN_SIZE];
  table
    .gather_sequence(&[7, 9], &mut out)
    .expect("ids in range");
  let mut expected = vec![f16::ZERO; 4 * HIDDEN_SIZE];
  for (slot, id) in expected.chunks_exact_mut(HIDDEN_SIZE).zip([7u32, 9, 0, 0]) {
    gather_row_scalar(table.row(id).unwrap(), slot);
  }
  let first_difference = out
    .iter()
    .zip(&expected)
    .position(|(got, want)| got.to_bits() != want.to_bits());
  assert_eq!(
    first_difference, None,
    "the first element that is not its row's"
  );
  assert_eq!(
    out[5 + 2 * HIDDEN_SIZE],
    gather_cell(0x3f80),
    "the pad row is row 0, scaled"
  );
  assert_ne!(out[5 + 2 * HIDDEN_SIZE], f16::ZERO);
  assert_eq!(
    table.gather_sequence(&[262_144], &mut out),
    Err(GatherRefusal::NoRow(262_144))
  );
  assert_eq!(
    table.gather_sequence(&[1, 2, 3, 4, 5], &mut out),
    Err(GatherRefusal::Shape {
      ids: 5,
      cells: 4 * HIDDEN_SIZE
    }),
    "more ids than rows"
  );
  let mut ragged = vec![f16::ZERO; 2 * HIDDEN_SIZE + 1];
  assert_eq!(
    table.gather_sequence(&[1], &mut ragged),
    Err(GatherRefusal::Shape {
      ids: 1,
      cells: 2 * HIDDEN_SIZE + 1
    }),
    "a ragged buffer"
  );
}

/// The staged model directory: `EMBEDDINGGEMMA2_TEST_MODELS`, else the
/// workspace's `Models/embeddinggemma2`.
fn staged_dir() -> PathBuf {
  std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../Models/embeddinggemma2"
      ))
    },
    PathBuf::from,
  )
}

fn upstream_token_sentinels() -> Vec<(TableCell, u16)> {
  TOKEN_SENTINEL_CELLS
    .iter()
    .copied()
    .zip(UPSTREAM_TOKEN_SENTINELS)
    .collect()
}

/// T5's model-gated half: over the staged table, every corpus text's
/// `inputs_embeds` for its unpacked function is byte for byte the one numpy
/// built for the goldens. The staged tables also carry §1.5's sentinels.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_gather_is_the_goldens_inputs_embeds() {
  use sha2::{Digest, Sha256};

  let dir = staged_dir();
  let table = TokenTable::open(&dir.join("embeddings.bf16"), &upstream_token_sentinels())
    .expect("the staged token table carries the upstream sentinels");
  let position_sentinels: Vec<(TableCell, u16)> = POSITION_SENTINEL_CELLS
    .iter()
    .copied()
    .zip(UPSTREAM_POSITION_SENTINELS)
    .collect();
  PositionTable::open(&dir.join("position_embeddings.f16"), &position_sentinels)
    .expect("the staged position tables carry the upstream sentinels");

  let goldens = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/embeddinggemma2/fixtures/goldens/text.json"
  );
  let doc: serde_json::Value =
    serde_json::from_str(&std::fs::read_to_string(goldens).expect("the text goldens"))
      .expect("JSON");
  let mut checked = 0;
  for item in doc["items"].as_array().expect("items") {
    let Some(digest) = item["inputs_embeds_sha256"].as_str() else {
      continue;
    };
    let ids: Vec<u32> = item["recognize"]["ids"]
      .as_array()
      .expect("ids")
      .iter()
      .map(|id| id.as_u64().expect("an id") as u32)
      .collect();
    let rows: usize = item["function"]
      .as_str()
      .and_then(|f| f.strip_prefix("embed_"))
      .expect("an embed_S function")
      .parse()
      .expect("S");
    let mut out = vec![f16::ZERO; rows * HIDDEN_SIZE];
    table.gather_sequence(&ids, &mut out).expect("ids in range");
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    let got: String = Sha256::digest(&bytes)
      .iter()
      .map(|b| format!("{b:02x}"))
      .collect();
    assert_eq!(got, digest, "{}", item["id"]);
    checked += 1;
  }
  assert!(checked >= 70, "{checked} items checked");
  eprintln!("the staged gather reproduced {checked} goldens' inputs_embeds byte for byte");
}

/// This process's physical footprint (`phys_footprint`), in bytes.
fn phys_footprint() -> u64 {
  // SAFETY: `task_vm_info` is plain-old-data; the all-zero bit pattern is a
  // valid value for every field.
  let mut info: mach2::task_info::task_vm_info = unsafe { core::mem::zeroed() };
  let mut count = (core::mem::size_of::<mach2::task_info::task_vm_info>()
    / core::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
  // SAFETY: `mach_task_self()` names the current task; `info` is a zeroed,
  // properly sized and aligned TASK_VM_INFO out-struct and `count` carries its
  // capacity in words per the in-out contract, so the kernel writes only
  // within `info`. The result code is checked before `info` is read.
  let result = unsafe {
    libc::task_info(
      mach2::traps::mach_task_self(),
      mach2::task_info::TASK_VM_INFO,
      (&raw mut info).cast(),
      &mut count,
    )
  };
  assert_eq!(result, libc::KERN_SUCCESS, "task_info(TASK_VM_INFO)");
  info.phys_footprint
}

/// Set in the child process M9b measures in.
const M9B_CHILD: &str = "EMBEDDINGGEMMA2_M9B_CHILD";

/// M9b: gathering 10,000 random rows of the staged 256 MiB table leaves the
/// physical footprint within noise, because mapped pages are clean and
/// file-backed; and two clones read one mapping.
///
/// `phys_footprint` is the whole process's, and the test harness runs other
/// tests beside this one, so the measurement runs in a child process that
/// runs this test alone and prints its two readings.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_table_stays_clean_and_mapped_once() {
  const NAME: &str =
    "embeddings::embeddinggemma2::table::tests::the_staged_table_stays_clean_and_mapped_once";
  if std::env::var_os(M9B_CHILD).is_some() {
    let table = TokenTable::open(
      &staged_dir().join("embeddings.bf16"),
      &upstream_token_sentinels(),
    )
    .expect("the staged token table");
    let second = table.clone();
    assert!(second.same_mapping(&table));
    let mut out = vec![f16::ZERO; HIDDEN_SIZE];
    let before = phys_footprint();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..10_000 {
      state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
      let id = ((state >> 33) % VOCAB_SIZE as u64) as u32;
      gather_row(second.row(id).expect("in range"), &mut out);
    }
    println!("M9B {before} {}", phys_footprint());
    return;
  }
  let child = std::process::Command::new(std::env::current_exe().expect("the test binary"))
    .args([
      "--exact",
      NAME,
      "--ignored",
      "--nocapture",
      "--test-threads=1",
    ])
    .env(M9B_CHILD, "1")
    .output()
    .expect("the child runs");
  let stdout = String::from_utf8_lossy(&child.stdout);
  assert!(
    child.status.success(),
    "the child failed:\n{stdout}\n{}",
    String::from_utf8_lossy(&child.stderr)
  );
  let readings: Vec<u64> = stdout
    .lines()
    .find_map(|line| line.split_once("M9B ").map(|(_, readings)| readings))
    .expect("the child's readings")
    .split(' ')
    .map(|n| n.parse().expect("a reading"))
    .collect();
  let grew = readings[1].saturating_sub(readings[0]);
  eprintln!(
    "M9b: phys_footprint {} -> {} bytes (+{grew}) over 10,000 random rows",
    readings[0], readings[1]
  );
  assert!(
    grew < 1 << 20,
    "the footprint grew {grew} bytes: mapped rows are not clean"
  );
}
