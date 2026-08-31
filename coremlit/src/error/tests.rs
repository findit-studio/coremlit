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
  let e = TensorError::DataTypeMismatch(DataTypeMismatch::new(DataType::F16, DataType::F32));
  assert_eq!(
    e.to_string(),
    "data type mismatch: expected `float16`, got `float32`"
  );
}

#[test]
fn load_error_not_found_displays_path() {
  let e = LoadError::NotFound("/tmp/missing.mlmodelc".into());
  assert!(e.to_string().contains("/tmp/missing.mlmodelc"));
}

/// Every `TensorError` message the variant sweep rewrote into accessor form.
/// Byte-exact because these render field values in a FIXED ORDER — the one
/// thing a `.0.a(), .0.b()` transcription can silently swap.
#[test]
fn tensor_error_rewritten_variants_display_their_payloads_in_order() {
  assert_eq!(
    TensorError::RankMismatch(RankMismatch::new(3, 2)).to_string(),
    "rank mismatch: expected 3 indices, got 2"
  );
  assert_eq!(
    TensorError::IndexOutOfBounds(IndexOutOfBounds::new(7, 4)).to_string(),
    "index 7 out of bounds for length 4"
  );
  // `new` takes (shape, strides); the message prints STRIDES first.
  assert_eq!(
    TensorError::NonContiguous(NonContiguous::new(vec![2, 3], vec![1, 1])).to_string(),
    "array layout is not contiguous (strides [1, 1] for shape [2, 3])"
  );
  assert_eq!(
    TensorError::UnsupportedDataType(DataType::F32).to_string(),
    "unsupported data type `float32` for array construction"
  );
  assert_eq!(
    TensorError::PixelBuffer(-6660).to_string(),
    "pixel buffer creation failed with CVReturn -6660"
  );
  assert_eq!(
    TensorError::ShapeOverflow(vec![usize::MAX, 2]).to_string(),
    "shape [18446744073709551615, 2] element count overflows usize"
  );
}

#[test]
fn prediction_error_not_multi_array_names_the_output() {
  assert_eq!(
    PredictionError::NotMultiArray("embedding".to_string()).to_string(),
    "prediction output `embedding` is not a multi-array"
  );
}
