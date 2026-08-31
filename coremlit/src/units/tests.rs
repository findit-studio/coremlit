use super::*;

#[test]
fn as_str_round_trips_from_str() {
  for u in [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ] {
    assert_eq!(u.as_str().parse::<ComputeUnits>().unwrap(), u);
  }
}

#[test]
fn display_matches_as_str() {
  assert_eq!(
    ComputeUnits::CpuAndNeuralEngine.to_string(),
    "cpu_and_neural_engine"
  );
}

#[test]
fn unknown_name_is_opaque_error() {
  assert!("tpu".parse::<ComputeUnits>().is_err());
}

#[test]
fn default_is_all() {
  assert_eq!(ComputeUnits::default(), ComputeUnits::All);
}
