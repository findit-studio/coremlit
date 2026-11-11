use super::*;

/// The label of a manifest is its first 12 hex, computed over the lines
/// sorted bytewise by path. The golden is `shasum -a 256` over
/// `0…0  Z.json\n1…1  a/b.bin\n`, the order `LC_ALL=C sort -k2` gives, so it
/// is computed by a tool other than this crate.
#[test]
fn the_digest_label_is_the_sorted_manifests_sha256() {
  let label = ArtifactLabel::from_digests([("a/b.bin", [0x11; 32]), ("Z.json", [0x00; 32])])
    .expect("a label");
  assert_eq!(label.to_string(), "sha256:896a97a8cefa");
  assert!(!label.is_revision());

  let reordered = ArtifactLabel::from_digests([("Z.json", [0x00; 32]), ("a/b.bin", [0x11; 32])])
    .expect("a label");
  assert_eq!(label, reordered, "the input order does not move the label");

  let moved = ArtifactLabel::from_digests([("a/b.bin", [0x11; 32]), ("Z.json", [0x01; 32])])
    .expect("a label");
  assert_ne!(label, moved, "a changed digest moves the label");
}

/// Condition 4: the list a caller's lines are sorted in is reserved through
/// the door, so a list that cannot be held is refused by name.
#[test]
fn the_digest_label_refuses_its_list_by_name() {
  let _refuse = super::super::alloc::inject::refuse(AllocationSite::StampDigests, 0);
  match ArtifactLabel::from_digests([("a/b.bin", [0x11; 32]), ("Z.json", [0x00; 32])]) {
    Err(super::super::Error::Allocation(refusal)) => {
      assert_eq!(refusal.site(), AllocationSite::StampDigests);
    }
    other => panic!("expected the list refused at stamp_digests, got {other:?}"),
  }
}

/// The three spellings print and parse back; nothing else parses.
#[test]
fn the_label_spellings_round_trip_and_refuse_by_name() {
  assert_eq!(
    ArtifactLabel::FLUIDINFERENCE.to_string(),
    "fluidinference@3f978dd"
  );
  assert!(ArtifactLabel::FLUIDINFERENCE.is_revision());
  for text in [
    "fluidinference@3f978dd",
    "findit-studio@0a1b2c3",
    "sha256:896a97a8cefa",
  ] {
    let label: ArtifactLabel = text.parse().expect(text);
    assert_eq!(label.to_string(), text);
  }
  assert_eq!(
    "fluidinference@3f978dd".parse::<ArtifactLabel>().unwrap(),
    ArtifactLabel::FLUIDINFERENCE
  );
  for text in [
    "",
    "fluidinference@3F978DD",
    "fluidinference@3f978d",
    "fluidinference@3f978ddf",
    "findit-studio@zzzzzzz",
    "sha256:896a97a8cef",
    "sha256:896a97a8cefa0",
    "sha1:896a97a8cefa",
    "huggingface@3f978dd",
  ] {
    let refused = text.parse::<ArtifactLabel>().unwrap_err();
    assert_eq!(refused.text(), text);
    assert!(
      refused.to_string().contains("is not an artifact label"),
      "{refused}"
    );
  }
}

/// The parts' spellings.
#[test]
fn the_part_spellings_are_pinned() {
  let spelled: Vec<String> = [Part::Text, Part::Audio, Part::Vision]
    .iter()
    .map(|part| format!("{part}={}", part.as_str()))
    .collect();
  assert_eq!(spelled, ["text=text", "audio=audio", "vision=vision"]);
}

/// The serde form is the label's string, in every format, and a string that
/// is not a label is refused.
#[cfg(feature = "serde")]
#[test]
fn the_label_round_trips_through_json_toml_and_postcard() {
  use std::collections::BTreeMap;

  for label in [
    ArtifactLabel::FLUIDINFERENCE,
    "findit-studio@0a1b2c3".parse().unwrap(),
    ArtifactLabel::from_digests([("x", [7; 32])]).expect("a label"),
  ] {
    let json = serde_json::to_string(&label).unwrap();
    assert_eq!(json, format!("\"{label}\""));
    assert_eq!(serde_json::from_str::<ArtifactLabel>(&json).unwrap(), label);

    let toml_doc = toml::to_string(&BTreeMap::from([("expect_artifacts", label)])).unwrap();
    assert_eq!(toml_doc, format!("expect_artifacts = \"{label}\"\n"));
    let back: BTreeMap<String, ArtifactLabel> = toml::from_str(&toml_doc).unwrap();
    assert_eq!(back["expect_artifacts"], label);

    let bytes = postcard::to_allocvec(&label).unwrap();
    assert_eq!(
      postcard::from_bytes::<ArtifactLabel>(&bytes).unwrap(),
      label
    );
  }
  let refused = serde_json::from_str::<ArtifactLabel>("\"fluidinference@main\"").unwrap_err();
  assert!(
    refused.to_string().contains("is not an artifact label"),
    "{refused}"
  );
}
