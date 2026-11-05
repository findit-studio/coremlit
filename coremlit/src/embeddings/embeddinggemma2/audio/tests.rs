use super::*;

/// K2: the precisions' spellings and package stems.
#[test]
fn the_precision_spellings_are_pinned() {
  let spelled: Vec<String> = [AudioPrecision::Fp16, AudioPrecision::Fp32]
    .iter()
    .map(|p| format!("{p}={}:{}", p.as_str(), p.package_stem()))
    .collect();
  assert_eq!(
    spelled,
    [
      "fp16=fp16:EmbeddingGemma2Audio",
      "fp32=fp32:EmbeddingGemma2AudioFp32"
    ]
  );
}

/// K2: the serde form is the snake_case name.
#[cfg(feature = "serde")]
#[test]
fn the_precisions_round_trip_through_json_and_postcard() {
  for p in [AudioPrecision::Fp16, AudioPrecision::Fp32] {
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(json, format!("\"{}\"", p.as_str()));
    assert_eq!(serde_json::from_str::<AudioPrecision>(&json).unwrap(), p);
    let bytes = postcard::to_allocvec(&p).unwrap();
    assert_eq!(postcard::from_bytes::<AudioPrecision>(&bytes).unwrap(), p);
  }
}
