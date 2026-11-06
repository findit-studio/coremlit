//! What a media model hands the text part: its own `[1, L, 512]` F16 output,
//! kept as Core ML returned it, and how many of its `L` rows are real.
//!
//! A text function embeds the real rows as a sequence framed by the face's
//! start and end tokens. The rows are copied as the model wrote them,
//! honouring the output's strides, and are not scaled: they are already in
//! the text model's input space.

use core::num::NonZeroU16;

use half::f16;

use super::{
  config::HIDDEN_SIZE,
  error::{Allocation, AllocationSite, Error, Result},
  vision::VisionBudget,
};
use crate::MultiArray;

/// The rows of one audio window's output.
pub(crate) const AUDIO_ROWS: usize = 250;

/// One audio window's tokens: the audio model's `[1, 250, 512]` F16 output
/// and its real rows, `1..=250`; a partial window has fewer.
#[derive(Debug)]
pub struct AudioTokens {
  array: MultiArray,
  valid: NonZeroU16,
}

impl AudioTokens {
  /// Wraps the audio model's output. `valid` is at most 250 and `array` is
  /// `[1, 250, 512]` F16, as the audio contract checked.
  pub(crate) fn new(array: MultiArray, valid: NonZeroU16) -> Self {
    debug_assert_eq!(array.shape(), [1, AUDIO_ROWS, HIDDEN_SIZE]);
    debug_assert!(usize::from(valid.get()) <= AUDIO_ROWS);
    Self { array, valid }
  }

  /// The real rows, `1..=250`.
  #[inline]
  pub const fn valid(&self) -> NonZeroU16 {
    self.valid
  }

  /// Copies the real rows into `out`, contiguous, as the model wrote them.
  pub(crate) fn rows_into(&self, out: &mut Vec<f16>) -> Result<()> {
    rows_into(&self.array, self.valid, out)
  }
}

/// One image's tokens at its budget: the vision model's `[1, T, 512]` F16
/// output, `T` the budget's tokens, and its real rows, the image's patches
/// over nine.
#[derive(Debug)]
pub struct ImageTokens {
  array: MultiArray,
  budget: VisionBudget,
  valid: NonZeroU16,
}

impl ImageTokens {
  /// Wraps the vision model's output at `budget`. `valid` is at most the
  /// budget's tokens and `array` is `[1, T, 512]` F16, as the vision contract
  /// checked.
  #[cfg_attr(not(test), allow(dead_code))]
  pub(crate) fn new(array: MultiArray, budget: VisionBudget, valid: NonZeroU16) -> Self {
    debug_assert_eq!(array.shape(), [1, budget.tokens(), HIDDEN_SIZE]);
    debug_assert!(usize::from(valid.get()) <= budget.tokens());
    Self {
      array,
      budget,
      valid,
    }
  }

  /// The budget the image was encoded at.
  #[inline]
  pub const fn budget(&self) -> VisionBudget {
    self.budget
  }

  /// The real rows, `1..=T`.
  #[inline]
  pub const fn valid(&self) -> NonZeroU16 {
    self.valid
  }

  /// Copies the real rows into `out`, contiguous, as the model wrote them.
  pub(crate) fn rows_into(&self, out: &mut Vec<f16>) -> Result<()> {
    rows_into(&self.array, self.valid, out)
  }
}

/// The first `valid` rows of `array`, strides honoured, into `out`.
fn rows_into(array: &MultiArray, valid: NonZeroU16, out: &mut Vec<f16>) -> Result<()> {
  let count = array.count();
  out.clear();
  out.try_reserve_exact(count).map_err(|_| {
    Error::Allocation(Allocation::new(
      AllocationSite::GatherRows,
      count.saturating_mul(2),
    ))
  })?;
  out.resize(count, f16::ZERO);
  array.copy_into(out)?;
  out.truncate((usize::from(valid.get()) * HIDDEN_SIZE).min(count));
  Ok(())
}

#[cfg(test)]
mod tests;
