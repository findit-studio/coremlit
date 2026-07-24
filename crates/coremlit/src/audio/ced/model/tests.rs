use std::{
  collections::BTreeSet,
  path::{Path, PathBuf},
};

use super::CedModel;

#[test]
fn as_str_and_display_are_the_snake_case_size_names() {
  assert_eq!(CedModel::Tiny.as_str(), "tiny");
  assert_eq!(CedModel::Mini.as_str(), "mini");
  assert_eq!(CedModel::Small.as_str(), "small");
  assert_eq!(CedModel::Base.as_str(), "base");
  for m in CedModel::ALL {
    assert_eq!(m.to_string(), m.as_str(), "Display must mirror as_str");
  }
}

#[test]
fn from_str_accepts_the_four_names_and_rejects_everything_else() {
  for m in CedModel::ALL {
    assert_eq!(m.as_str().parse::<CedModel>().unwrap(), m);
  }
  // Wrong case, the hyphenated dir spelling, the underscored stem, an
  // out-of-family size, and the empty string are all rejections.
  for bad in ["Tiny", "TINY", "ced-tiny", "ced_tiny", "large", ""] {
    assert!(bad.parse::<CedModel>().is_err(), "{bad:?} must not parse");
  }
}

#[test]
fn all_is_the_four_distinct_sizes() {
  assert_eq!(
    CedModel::ALL,
    [
      CedModel::Tiny,
      CedModel::Mini,
      CedModel::Small,
      CedModel::Base
    ]
  );
  let names: BTreeSet<&str> = CedModel::ALL.iter().map(|m| m.as_str()).collect();
  assert_eq!(names.len(), 4, "ALL entries must be distinct");
}

#[test]
fn metadata_strings_are_exact_per_size() {
  assert_eq!(CedModel::Tiny.hf_repo(), "mispeech/ced-tiny");
  assert_eq!(CedModel::Mini.hf_repo(), "mispeech/ced-mini");
  assert_eq!(CedModel::Small.hf_repo(), "mispeech/ced-small");
  assert_eq!(CedModel::Base.hf_repo(), "mispeech/ced-base");

  assert_eq!(CedModel::Tiny.dir_name(), "ced-tiny");
  assert_eq!(CedModel::Mini.dir_name(), "ced-mini");
  assert_eq!(CedModel::Small.dir_name(), "ced-small");
  assert_eq!(CedModel::Base.dir_name(), "ced-base");

  assert_eq!(CedModel::Tiny.mlmodelc_name(), "ced_tiny.mlmodelc");
  assert_eq!(CedModel::Mini.mlmodelc_name(), "ced_mini.mlmodelc");
  assert_eq!(CedModel::Small.mlmodelc_name(), "ced_small.mlmodelc");
  assert_eq!(CedModel::Base.mlmodelc_name(), "ced_base.mlmodelc");
}

#[test]
fn mlmodelc_path_composes_hyphen_dir_and_underscore_stem_under_root() {
  let root = Path::new("/models/ced");
  assert_eq!(
    CedModel::Small.mlmodelc_path(root),
    Path::new("/models/ced/ced-small/ced_small.mlmodelc"),
  );
  // Tiny keeps the exact Wave-A layout (`Models/ced/ced-tiny/ced_tiny.mlmodelc`).
  assert_eq!(
    CedModel::Tiny.mlmodelc_path("relative/root"),
    PathBuf::from("relative/root/ced-tiny/ced_tiny.mlmodelc"),
  );
}

#[cfg(feature = "serde")]
#[test]
fn serde_uses_the_pinned_snake_case_spellings() {
  for m in CedModel::ALL {
    let json = serde_json::to_string(&m).unwrap();
    assert_eq!(json, format!("\"{}\"", m.as_str()));
    let back: CedModel = serde_json::from_str(&json).unwrap();
    assert_eq!(back, m);
  }
  assert_eq!(serde_json::to_string(&CedModel::Base).unwrap(), "\"base\"");
}

#[cfg(feature = "serde")]
#[test]
fn serde_rejects_unknown_and_wrong_case_spellings() {
  assert!(serde_json::from_str::<CedModel>("\"Tiny\"").is_err());
  assert!(serde_json::from_str::<CedModel>("\"ced-tiny\"").is_err());
  assert!(serde_json::from_str::<CedModel>("\"large\"").is_err());
  assert!(serde_json::from_str::<CedModel>("\"\"").is_err());
}
