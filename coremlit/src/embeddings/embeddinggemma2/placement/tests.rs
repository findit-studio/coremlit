use super::*;

const ARMS: [ComputeUnits; 4] = [
  ComputeUnits::CpuOnly,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuAndNeuralEngine,
  ComputeUnits::All,
];

/// A table answers per arm, and a measured arm moves only itself.
#[test]
fn a_placement_table_answers_per_arm() {
  let table = PlacementTable::unmeasured(false).with(ComputeUnits::CpuOnly, true);
  for units in ARMS {
    assert_eq!(
      table.get(units),
      units == ComputeUnits::CpuOnly,
      "{units:?}"
    );
  }
}

/// `Single` is batch-independent under every arm; `Packed` is what the
/// measured table says for the arm, and false where unmeasured.
#[test]
fn batch_independence_reads_the_packing_table() {
  for units in ARMS {
    assert!(Batching::Single.batch_independent(units));
    assert_eq!(
      Batching::Packed.batch_independent(units),
      PACKED_SLOT_INVARIANT.get(units)
    );
  }
}
