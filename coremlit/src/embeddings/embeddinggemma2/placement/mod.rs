//! What the kit has measured about its parts' repeatability, per placement.
//!
//! Both facts depend on the device program, the Neural Engine's, the GPU's or
//! the CPU's under one OS build, and `ComputeUnits` is an option. So each is a
//! table over the four arms, filled by a measurement on the characterized
//! host class and asserted there by a law scoped through `measured_band`. An
//! arm nobody measured answers the safe value: `false`, or `None`.

use super::{artifacts::Part, options::Batching};
use crate::ComputeUnits;

/// A value per [`ComputeUnits`] arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlacementTable<T> {
  cpu_only: T,
  cpu_and_gpu: T,
  cpu_and_neural_engine: T,
  all: T,
}

impl<T: Copy> PlacementTable<T> {
  /// `value` under every arm: what an unmeasured table answers.
  #[inline]
  pub const fn unmeasured(value: T) -> Self {
    Self {
      cpu_only: value,
      cpu_and_gpu: value,
      cpu_and_neural_engine: value,
      all: value,
    }
  }

  /// This table with `units` answering `value`.
  #[must_use]
  #[inline]
  pub const fn with(self, units: ComputeUnits, value: T) -> Self {
    let mut table = self;
    match units {
      ComputeUnits::CpuOnly => table.cpu_only = value,
      ComputeUnits::CpuAndGpu => table.cpu_and_gpu = value,
      ComputeUnits::CpuAndNeuralEngine => table.cpu_and_neural_engine = value,
      ComputeUnits::All => table.all = value,
    }
    table
  }

  /// The value under `units`.
  #[inline]
  pub const fn get(&self, units: ComputeUnits) -> T {
    match units {
      ComputeUnits::CpuOnly => self.cpu_only,
      ComputeUnits::CpuAndGpu => self.cpu_and_gpu,
      ComputeUnits::CpuAndNeuralEngine => self.cpu_and_neural_engine,
      ComputeUnits::All => self.all,
    }
  }
}

/// Whether `pack_256` gives a sequence the same vector, bit for bit, at every
/// slot and beside any companions, per placement: M4, on the characterized
/// host class. `false` under every arm nobody has measured.
///
/// M4 measured it `false` under all four arms on Apple M1 Max, macOS 27.0
/// (26A428): a sequence's packed vector moves with its slot and companions,
/// so a packed face is not batch-independent anywhere yet.
pub const PACKED_SLOT_INVARIANT: PlacementTable<bool> = PlacementTable::unmeasured(false);

/// M11's text table: does one text input repeat bit for bit, twice on one
/// instance, on two instances and in two processes. `None` where unmeasured.
///
/// Measured `Some(true)` under all four arms on Apple M1 Max, macOS 27.0
/// (26A428), and asserted there by `embeddinggemma2_text`'s M11 gate.
const TEXT_RUN_IDENTITY: PlacementTable<Option<bool>> = PlacementTable::unmeasured(None)
  .with(ComputeUnits::CpuOnly, Some(true))
  .with(ComputeUnits::CpuAndGpu, Some(true))
  .with(ComputeUnits::CpuAndNeuralEngine, Some(true))
  .with(ComputeUnits::All, Some(true));

/// M11's audio table, filled at C12. `None` where unmeasured.
const AUDIO_RUN_IDENTITY: PlacementTable<Option<bool>> = PlacementTable::unmeasured(None);

/// M11's vision table, filled at C15. `None` where unmeasured.
const VISION_RUN_IDENTITY: PlacementTable<Option<bool>> = PlacementTable::unmeasured(None);

/// Whether one call of `part` under `units` repeats bit for bit: twice on one
/// instance, on two instances and in two processes (M11). `None` where
/// unmeasured.
#[inline]
pub const fn run_identity(part: Part, units: ComputeUnits) -> Option<bool> {
  match part {
    Part::Text => TEXT_RUN_IDENTITY.get(units),
    Part::Audio => AUDIO_RUN_IDENTITY.get(units),
    Part::Vision => VISION_RUN_IDENTITY.get(units),
  }
}

impl Batching {
  /// Whether a sequence's vector is independent of the batch it rides in:
  /// always under [`Batching::Single`], and under [`Batching::Packed`] what
  /// [`PACKED_SLOT_INVARIANT`] measured for `units`, the TEXT part's
  /// placement, where every pack runs.
  #[inline]
  pub const fn batch_independent(self, units: ComputeUnits) -> bool {
    self.batch_independent_in(units, &PACKED_SLOT_INVARIANT)
  }

  /// [`Self::batch_independent`] over `table` in place of
  /// [`PACKED_SLOT_INVARIANT`]: the one rule, for a law to drive with a
  /// table whose arms differ.
  #[inline]
  pub(crate) const fn batch_independent_in(
    self,
    units: ComputeUnits,
    table: &PlacementTable<bool>,
  ) -> bool {
    match self {
      Self::Single => true,
      Self::Packed => table.get(units),
    }
  }
}

#[cfg(test)]
mod tests;
