use half::f16;

use super::*;
use crate::{DataType, TensorError};

#[test]
fn zeros_has_shape_count_dtype_and_zero_content() {
  let arr = MultiArray::zeros(&[2, 3, 4], DataType::F32).unwrap();
  assert_eq!(arr.shape(), vec![2, 3, 4]);
  assert_eq!(arr.count(), 24);
  assert_eq!(arr.data_type(), DataType::F32);
  assert!(arr.as_slice::<f32>().unwrap().iter().all(|v| *v == 0.0));
}

#[test]
fn from_slice_round_trips_f32() {
  let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
  let arr = MultiArray::from_slice(&[2, 3], &data).unwrap();
  assert_eq!(arr.as_slice::<f32>().unwrap(), &data);
}

#[test]
fn from_slice_round_trips_f16_and_i32() {
  let h = [f16::from_f32(0.5), f16::from_f32(-1.0)];
  let arr = MultiArray::from_slice(&[2], &h).unwrap();
  assert_eq!(arr.as_slice::<f16>().unwrap(), &h);
  assert_eq!(arr.data_type(), DataType::F16);

  let ints = [7i32, -7];
  let arr = MultiArray::from_slice(&[1, 2], &ints).unwrap();
  assert_eq!(arr.as_slice::<i32>().unwrap(), &ints);
}

#[test]
fn wrong_view_type_is_dtype_mismatch() {
  let arr = MultiArray::zeros(&[4], DataType::F32).unwrap();
  let err = arr.as_slice::<i32>().unwrap_err();
  assert_eq!(
    err,
    TensorError::DataTypeMismatch {
      expected: DataType::I32,
      actual: DataType::F32
    }
  );
}

#[test]
fn from_slice_rejects_shape_element_mismatch() {
  let err = MultiArray::from_slice(&[2, 2], &[1.0f32]).unwrap_err();
  assert_eq!(
    err,
    TensorError::ShapeMismatch {
      expected: 4,
      actual: 1
    }
  );
}

#[test]
fn as_slice_mut_writes_are_visible() {
  let mut arr = MultiArray::zeros(&[3], DataType::F32).unwrap();
  arr.as_slice_mut::<f32>().unwrap()[1] = 9.5;
  assert_eq!(arr.as_slice::<f32>().unwrap()[1], 9.5);
}
