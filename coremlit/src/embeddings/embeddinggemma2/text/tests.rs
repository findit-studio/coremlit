use super::*;

/// K2: each function's name, rows and spellings, in package order.
#[test]
fn the_function_spellings_are_pinned() {
  let spelled: Vec<String> = TextFunction::ALL
    .iter()
    .map(|f| format!("{f}={}:{}", f.as_str(), f.tokens()))
    .collect();
  assert_eq!(
    spelled,
    [
      "embed_32=embed_32:32",
      "embed_48=embed_48:48",
      "embed_64=embed_64:64",
      "embed_128=embed_128:128",
      "embed_256=embed_256:256",
      "embed_512=embed_512:512",
      "pack_256=pack_256:256",
    ]
  );
}

/// The set holds what was added, lists it in package order, and prints it.
#[test]
fn the_function_set_is_a_set_in_package_order() {
  let set: TextFunctions = [
    TextFunction::Pack256,
    TextFunction::Embed32,
    TextFunction::Pack256,
  ]
  .into_iter()
  .collect();
  assert_eq!(set.len(), 2);
  assert!(set.contains(TextFunction::Embed32) && !set.contains(TextFunction::Embed48));
  assert_eq!(set.to_string(), "[embed_32,pack_256]");
  assert_eq!(format!("{set:?}"), "{Embed32, Pack256}");
  assert_eq!(TextFunctions::empty().to_string(), "[]");
  assert!(TextFunctions::empty().is_empty());
  assert_eq!(TextFunctions::all().len(), 7);
  assert_eq!(
    TextFunctions::all().iter().collect::<Vec<_>>(),
    TextFunction::ALL
  );
}

/// K2: the serde form is the function's name, the set is a list of names, and
/// a name listed twice is refused by name.
#[cfg(feature = "serde")]
#[test]
fn the_functions_round_trip_through_json_toml_and_postcard() {
  use std::collections::BTreeMap;

  for f in TextFunction::ALL {
    let json = serde_json::to_string(&f).unwrap();
    assert_eq!(json, format!("\"{}\"", f.as_str()));
    assert_eq!(serde_json::from_str::<TextFunction>(&json).unwrap(), f);
    let bytes = postcard::to_allocvec(&f).unwrap();
    assert_eq!(postcard::from_bytes::<TextFunction>(&bytes).unwrap(), f);
  }

  let set = TextFunctions::empty()
    .with(TextFunction::Embed512)
    .with(TextFunction::Embed64);
  let json = serde_json::to_string(&set).unwrap();
  assert_eq!(json, r#"["embed_64","embed_512"]"#);
  assert_eq!(serde_json::from_str::<TextFunctions>(&json).unwrap(), set);
  let toml_doc = toml::to_string(&BTreeMap::from([("prewarm", set)])).unwrap();
  assert_eq!(toml_doc, "prewarm = [\"embed_64\", \"embed_512\"]\n");
  let back: BTreeMap<String, TextFunctions> = toml::from_str(&toml_doc).unwrap();
  assert_eq!(back["prewarm"], set);
  let bytes = postcard::to_allocvec(&set).unwrap();
  assert_eq!(postcard::from_bytes::<TextFunctions>(&bytes).unwrap(), set);

  let twice = serde_json::from_str::<TextFunctions>(r#"["pack_256","pack_256"]"#).unwrap_err();
  assert!(
    twice.to_string().contains("`pack_256` is listed twice"),
    "{twice}"
  );
  let unknown = serde_json::from_str::<TextFunctions>(r#"["embed_1024"]"#).unwrap_err();
  assert!(unknown.to_string().contains("embed_1024"), "{unknown}");
}
