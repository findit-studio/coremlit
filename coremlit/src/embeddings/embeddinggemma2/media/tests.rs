use core::num::NonZeroU16;

use half::f16;

use super::*;

/// A `[1, rows, 512]` F16 array whose row `r` holds `r + 1` in every cell.
fn numbered(rows: usize) -> MultiArray {
  let data: Vec<f16> = (0..rows)
    .flat_map(|r| core::iter::repeat_n(f16::from_f32((r + 1) as f32), HIDDEN_SIZE))
    .collect();
  MultiArray::from_slice(&[1, rows, HIDDEN_SIZE], &data).expect("an array")
}

/// A media carrier hands over exactly its real rows, in order, as written.
#[test]
fn a_carrier_hands_over_its_real_rows() {
  let tokens = AudioTokens::new(numbered(AUDIO_ROWS), NonZeroU16::new(3).unwrap());
  let mut rows = Vec::new();
  tokens.rows_into(&mut rows).expect("copies");
  assert_eq!(rows.len(), 3 * HIDDEN_SIZE);
  for (row, chunk) in rows.chunks_exact(HIDDEN_SIZE).enumerate() {
    assert!(
      chunk.iter().all(|v| v.to_f32() == (row + 1) as f32),
      "row {row}"
    );
  }

  let image = ImageTokens::new(
    numbered(VisionBudget::Tokens70.tokens()),
    VisionBudget::Tokens70,
    NonZeroU16::new(70).unwrap(),
  );
  image.rows_into(&mut rows).expect("copies");
  assert_eq!(
    rows.len(),
    70 * HIDDEN_SIZE,
    "a buffer is refilled, not appended"
  );
  assert_eq!(rows[69 * HIDDEN_SIZE].to_f32(), 70.0);
  assert_eq!(image.budget(), VisionBudget::Tokens70);
}
