use super::*;
use crate::ComputeUnits;

#[test]
fn model_is_send() {
  fn assert_send<T: Send>() {}
  assert_send::<Model>();
}

#[test]
fn load_missing_path_is_not_found() {
  let err = Model::load("/nonexistent/Foo.mlmodelc", ComputeUnits::CpuOnly).unwrap_err();
  assert!(matches!(err, crate::LoadError::NotFound { .. }));
}

#[test]
fn compile_missing_source_is_not_found() {
  let err = Model::compile("/nonexistent/foo.mlpackage").unwrap_err();
  assert!(matches!(err, crate::CompileError::NotFound { .. }));
}
