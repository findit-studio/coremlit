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
  // `features` (still in scope) and `back` alias the same underlying
  // MLMultiArray buffer; `from_raw`'s sole-ownership invariant tolerates
  // this only because every access below is read-only — a future edit must
  // not add mutation through either handle while the other is alive.
  // `known_regions` is empty (not seeded with `features`'s own regions, as
  // `Model::predict` seeds it with its live inputs), so this aliasing is
  // not detected/copied here — that's exercised separately below.
  let mut known_regions = Vec::new();
  let back =
    Features::from_provider(ProtocolObject::from_ref(&*provider), &mut known_regions).unwrap();
  let x = back.get("x").unwrap();
  assert_eq!(x.shape(), vec![2, 2]);
  assert_eq!(x.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn from_provider_deep_copies_one_array_shared_under_two_names() {
  // Hand-construct a provider with ONE MLMultiArray registered under TWO
  // feature names — the "two output names reference the same array" case
  // from the review — mirroring `to_provider`'s own construction.
  let shared = MultiArray::from_slice(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
  // SAFETY: featureValueWithMultiArray is a plain constructor send;
  // `shared.raw()` borrows a live MLMultiArray for the call's duration and
  // the returned MLFeatureValue retains it, so no dangling reference
  // results once this closure returns.
  let value: Retained<MLFeatureValue> =
    unsafe { MLFeatureValue::featureValueWithMultiArray(shared.raw()) };
  let value: Retained<AnyObject> = value.into();
  let keys = [NSString::from_str("a"), NSString::from_str("b")];
  let key_refs: Vec<&NSString> = keys.iter().map(AsRef::as_ref).collect();
  let dict = NSDictionary::from_retained_objects(&key_refs, &[value.clone(), value]);
  // SAFETY: as in `Features::to_provider` — `dict` maps NSString keys to
  // MLFeatureValue objects (erased to AnyObject); `alloc()` supplies a
  // fresh, unaliased receiver.
  let provider = unsafe {
    MLDictionaryFeatureProvider::initWithDictionary_error(
      MLDictionaryFeatureProvider::alloc(),
      &dict,
    )
  }
  .unwrap();

  let mut known_regions = Vec::new();
  let mut extracted =
    Features::from_provider(ProtocolObject::from_ref(&*provider), &mut known_regions).unwrap();

  let a = extracted.get("a").unwrap();
  let b = extracted.get("b").unwrap();
  assert_ne!(a.byte_range().0, b.byte_range().0);
  assert_eq!(a.as_slice::<f32>().unwrap(), b.as_slice::<f32>().unwrap());

  // Mutating one must not affect the other now that they own separate
  // buffers.
  let a_owned = extracted.take("a").unwrap();
  let mut b_owned = extracted.take("b").unwrap();
  b_owned.as_slice_mut::<f32>().unwrap()[0] = 99.0;
  assert_eq!(a_owned.as_slice::<f32>().unwrap()[0], 1.0);
}

#[test]
fn from_provider_deep_copies_output_that_aliases_a_seeded_input() {
  // The identity/zero-copy model case: an output feature literally is one
  // of the caller's own (still-live) input arrays.
  let input = MultiArray::from_slice(&[2], &[5.0f32, 6.0]).unwrap();
  let input_region = input.byte_range();

  // SAFETY: as in `Features::to_provider`.
  let value: Retained<MLFeatureValue> =
    unsafe { MLFeatureValue::featureValueWithMultiArray(input.raw()) };
  let value: Retained<AnyObject> = value.into();
  let key = NSString::from_str("y");
  let key_refs: Vec<&NSString> = vec![key.as_ref()];
  let dict = NSDictionary::from_retained_objects(&key_refs, &[value]);
  // SAFETY: as in `Features::to_provider`.
  let provider = unsafe {
    MLDictionaryFeatureProvider::initWithDictionary_error(
      MLDictionaryFeatureProvider::alloc(),
      &dict,
    )
  }
  .unwrap();

  // Simulates what `Model::predict` does before extracting: seed
  // `known_regions` with every live input's byte range.
  let mut known_regions = vec![input_region];
  let extracted =
    Features::from_provider(ProtocolObject::from_ref(&*provider), &mut known_regions).unwrap();
  let output = extracted.get("y").unwrap();
  assert_ne!(output.byte_range().0, input_region.0);
  assert_eq!(output.as_slice::<f32>().unwrap(), &[5.0, 6.0]);
}

#[test]
fn insert_replacing_moves_name_to_end() {
  let mut features = Features::new();
  features
    .insert("a", MultiArray::zeros(&[1], DataType::F32).unwrap())
    .insert("b", MultiArray::zeros(&[2], DataType::F32).unwrap());
  features.insert("a", MultiArray::zeros(&[9], DataType::F32).unwrap());
  assert_eq!(features.names().collect::<Vec<_>>(), vec!["b", "a"]);
  assert_eq!(features.get("a").unwrap().count(), 9);
}

#[test]
fn overlapping_offset_regions_are_detected() {
  // Regions are half-open [start, end): a view starting inside another
  // array's span must collide even without pointer equality.
  let base = MultiArray::from_slice(&[8], &[0.0f32; 8]).unwrap();
  let (start, end) = base.byte_range();
  assert_eq!(end - start, 8 * 4);
  let mut known = vec![(start, end)];
  let offset_view = (start + 8, start + 16); // bytes 8..16 inside base
  assert!(
    known
      .iter()
      .any(|&k| k.0 < offset_view.1 && offset_view.0 < k.1)
  );
  let adjacent = (end, end + 16); // begins exactly at end: no overlap
  assert!(!known.iter().any(|&k| k.0 < adjacent.1 && adjacent.0 < k.1));
  known.push(adjacent);
}
