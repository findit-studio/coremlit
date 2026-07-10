use super::*;
use crate::DataType;

#[test]
fn ns_error_info_captures_domain_code_message() {
  use objc2_foundation::{NSError, NSString};
  // SAFETY: Creating a test NSError with a valid domain and code.
  let err =
    unsafe { NSError::errorWithDomain_code_userInfo(&NSString::from_str("TestDomain"), 42, None) };
  let info = NsErrorInfo::from_ns_error(&err);
  assert_eq!(info.domain(), "TestDomain");
  assert_eq!(info.code(), 42);
  assert!(!info.message().is_empty());
}

#[test]
fn tensor_error_displays_structured_fields() {
  let e = TensorError::DataTypeMismatch {
    expected: DataType::F16,
    actual: DataType::F32,
  };
  assert_eq!(
    e.to_string(),
    "data type mismatch: expected `float16`, got `float32`"
  );
}

#[test]
fn load_error_not_found_displays_path() {
  let e = LoadError::NotFound {
    path: "/tmp/missing.mlmodelc".into(),
  };
  assert!(e.to_string().contains("/tmp/missing.mlmodelc"));
}
