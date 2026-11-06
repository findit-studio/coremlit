use super::*;

/// K2: each budget's function name, token count, patches and `Display`.
#[test]
fn the_budget_spellings_are_pinned() {
  let spelled: Vec<String> = VisionBudget::ALL
    .iter()
    .map(|b| format!("{b}={}:{}:{}", b.as_str(), b.tokens(), b.patches()))
    .collect();
  assert_eq!(
    spelled,
    [
      "70=vision_70:70:630",
      "140=vision_140:140:1260",
      "280=vision_280:280:2520"
    ]
  );
  for b in VisionBudget::ALL {
    assert_eq!(VisionBudget::from_tokens(b.tokens()), Some(b));
  }
  assert_eq!(VisionBudget::from_tokens(560), None);
  assert_eq!(ResizeRoad::HfBicubic.to_string(), "hf_bicubic");
}

/// The set holds what was added, smallest first.
#[test]
fn the_budget_set_is_a_set_smallest_first() {
  let set: VisionBudgets = [VisionBudget::Tokens280, VisionBudget::Tokens70]
    .into_iter()
    .collect();
  assert_eq!(set.to_string(), "[70,280]");
  assert_eq!(set.len(), 2);
  assert!(!set.contains(VisionBudget::Tokens140));
  assert_eq!(
    VisionBudgets::only(VisionBudget::Tokens140).to_string(),
    "[140]"
  );
  assert_eq!(VisionBudgets::all().to_string(), "[70,140,280]");
  assert_eq!(VisionBudgets::empty().to_string(), "[]");
  assert_eq!(format!("{set:?}"), "{Tokens70, Tokens280}");
}

/// K2: a budget's serde form is its token count, the set a list of counts; an
/// unknown count and a count listed twice are refused by name.
#[cfg(feature = "serde")]
#[test]
fn the_budgets_round_trip_through_json_toml_and_postcard() {
  use std::collections::BTreeMap;

  for b in VisionBudget::ALL {
    let json = serde_json::to_string(&b).unwrap();
    assert_eq!(json, b.tokens().to_string());
    assert_eq!(serde_json::from_str::<VisionBudget>(&json).unwrap(), b);
    let bytes = postcard::to_allocvec(&b).unwrap();
    assert_eq!(postcard::from_bytes::<VisionBudget>(&bytes).unwrap(), b);
  }
  let unknown = serde_json::from_str::<VisionBudget>("560").unwrap_err();
  assert!(
    unknown.to_string().contains("unknown vision budget 560"),
    "{unknown}"
  );

  let set = VisionBudgets::only(VisionBudget::Tokens140).with(VisionBudget::Tokens280);
  let toml_doc = toml::to_string(&BTreeMap::from([("budgets", set)])).unwrap();
  assert_eq!(toml_doc, "budgets = [140, 280]\n");
  let back: BTreeMap<String, VisionBudgets> = toml::from_str(&toml_doc).unwrap();
  assert_eq!(back["budgets"], set);
  let bytes = postcard::to_allocvec(&set).unwrap();
  assert_eq!(postcard::from_bytes::<VisionBudgets>(&bytes).unwrap(), set);
  let twice = serde_json::from_str::<VisionBudgets>("[140, 140]").unwrap_err();
  assert!(twice.to_string().contains("140 is listed twice"), "{twice}");

  let road = serde_json::to_string(&ResizeRoad::HfBicubic).unwrap();
  assert_eq!(road, "\"hf_bicubic\"");
  assert_eq!(
    serde_json::from_str::<ResizeRoad>(&road).unwrap(),
    ResizeRoad::HfBicubic
  );
}
