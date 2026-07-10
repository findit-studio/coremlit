use super::*;

#[test]
fn raw_round_trip_is_lossless_including_unknown() {
  for dt in [
    DataType::F16,
    DataType::F32,
    DataType::F64,
    DataType::I32,
    DataType::Unknown(7),
  ] {
    assert_eq!(DataType::from_raw(dt.to_raw()), dt);
  }
}

#[test]
fn known_raw_values_match_coreml() {
  // MLMultiArrayDataType: Float64 = 0x10000|64, Float32 = 0x10000|32,
  // Float16 = 0x10000|16, Int32 = 0x20000|32.
  assert_eq!(DataType::F64.to_raw(), 0x10040);
  assert_eq!(DataType::F32.to_raw(), 0x10020);
  assert_eq!(DataType::F16.to_raw(), 0x10010);
  assert_eq!(DataType::I32.to_raw(), 0x20020);
}

#[test]
fn element_sizes() {
  assert_eq!(DataType::F16.size_of(), Some(2));
  assert_eq!(DataType::F32.size_of(), Some(4));
  assert_eq!(DataType::F64.size_of(), Some(8));
  assert_eq!(DataType::I32.size_of(), Some(4));
  assert_eq!(DataType::Unknown(7).size_of(), None);
}

#[test]
fn as_str_names() {
  assert_eq!(DataType::F16.as_str(), "float16");
  assert_eq!(DataType::Unknown(7).as_str(), "unknown");
}
