use super::*;
use crate::{DataType, MultiArray};

#[test]
fn insert_get_take_names() {
  let mut features = Features::new();
  features
    .insert("audio", MultiArray::zeros(&[4], DataType::F32).unwrap())
    .insert("mask", MultiArray::zeros(&[2], DataType::F32).unwrap());
  assert_eq!(features.len(), 2);
  assert_eq!(features.names().collect::<Vec<_>>(), vec!["audio", "mask"]);
  assert_eq!(features.get("audio").unwrap().count(), 4);
  assert_eq!(features.take("mask").unwrap().count(), 2);
  assert!(features.get("mask").is_none());
}

#[test]
fn provider_round_trip_preserves_names_shapes_and_data() {
  let mut features = Features::new();
  features.insert(
    "x",
    MultiArray::from_slice(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0]).unwrap(),
  );
  let provider = features.to_provider().unwrap();
  let back = Features::from_provider(ProtocolObject::from_ref(&*provider)).unwrap();
  let x = back.get("x").unwrap();
  assert_eq!(x.shape(), vec![2, 2]);
  assert_eq!(x.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}
