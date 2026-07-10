use super::*;
use crate::ComputeUnits;

#[test]
fn load_missing_path_is_not_found() {
  let err = Model::load("/nonexistent/Foo.mlmodelc", ComputeUnits::CpuOnly).unwrap_err();
  assert!(matches!(err, crate::LoadError::NotFound { .. }));
}
