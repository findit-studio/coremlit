use serde_json::Value;

use super::*;

/// The committed copy of FluidInference's `config.json` @ `3f978dd`
/// (2,845 bytes, SHA-256 `30668aba…9629b`).
const CONFIG: &str = include_str!("../../../../tests/embeddinggemma2/fixtures/config.json");

fn check_text(text: &str) -> Result<()> {
  check(Path::new("config.json"), text.as_bytes())
}

/// The field a refusal names, or a panic naming what came back instead.
fn refused_field(text: &str) -> &'static str {
  match check_text(text) {
    Err(Error::Config(mismatch)) => mismatch.field().as_str(),
    other => panic!("expected a config refusal, got {other:?}"),
  }
}

/// K1: the committed `config.json` equals the compiled contract, field by
/// field.
#[test]
fn the_committed_config_equals_the_compiled_contract() {
  check_text(CONFIG).expect("the committed config.json is the compiled contract");
}

/// K1's named refusal: an `embed_scale` one millionth off is refused naming
/// `embed_scale`, even though it rounds to the same f32.
#[test]
fn an_embed_scale_off_by_a_millionth_is_refused_by_name() {
  let planted = CONFIG.replace("\"embed_scale\": 22.627417", "\"embed_scale\": 22.627416");
  assert_ne!(
    planted, CONFIG,
    "the fixture spells embed_scale as expected"
  );
  assert_eq!(22.627_416_f32.to_bits(), EMBED_SCALE.to_bits());
  assert_eq!(refused_field(&planted), "embed_scale");
  match check_text(&planted) {
    Err(e) => assert_eq!(
      e.to_string(),
      "config.json disagrees with the kit's compiled contract at `embed_scale`: expected 22.627417, got 22.627416"
    ),
    Ok(()) => unreachable!(),
  }
}

/// Every kind of difference is refused by the dotted path of its field.
#[test]
fn each_difference_is_refused_by_its_dotted_path() {
  let cases: [(&str, &str, &str); 7] = [
    ("\"slots\": 8", "\"slots\": 9", "functions.pack_256.slots"),
    (
      "\"waveform_samples\": 160160",
      "\"waveform_samples\": 160000",
      "audio.waveform_samples",
    ),
    (
      "\"document\": \"title: none | text: \"",
      "\"document\": \"title: none | text:\"",
      "prompts.document",
    ),
    ("256,\n    128\n  ]", "256\n  ]", "matryoshka_dims"),
    (
      "\"hidden_size\": 512",
      "\"hidden_size\": \"512\"",
      "hidden_size",
    ),
    (
      "\"patches\": 1260",
      "\"patches\": 1260, \"stride\": 1",
      "vision.functions.vision_140",
    ),
    ("\"bos_token_id\": 2,", "", "bos_token_id"),
  ];
  let pretty =
    serde_json::to_string_pretty(&serde_json::from_str::<Value>(CONFIG).unwrap()).unwrap();
  for (from, to, field) in cases {
    assert!(pretty.contains(from), "the fixture holds `{from}`");
    assert_eq!(
      refused_field(&pretty.replacen(from, to, 1)),
      field,
      "{from} → {to}"
    );
  }
}

/// A missing key reads `absent`; an extra top-level key names the top level.
#[test]
fn a_missing_and_an_extra_key_are_refused() {
  let mut value: Value = serde_json::from_str(CONFIG).unwrap();
  value.as_object_mut().unwrap().remove("format");
  match check_text(&value.to_string()) {
    Err(Error::Config(m)) => {
      assert_eq!((m.field().as_str(), m.actual()), ("format", "absent"));
    }
    other => panic!("expected a config refusal, got {other:?}"),
  }

  let mut value: Value = serde_json::from_str(CONFIG).unwrap();
  value
    .as_object_mut()
    .unwrap()
    .insert("revision".into(), Value::from(2));
  match check_text(&value.to_string()) {
    Err(Error::Config(m)) => {
      assert_eq!(m.field().as_str(), "");
      assert_eq!(m.field().to_string(), "(top level)");
      assert!(
        m.actual().starts_with("unknown field `revision`"),
        "{}",
        m.actual()
      );
    }
    other => panic!("expected a config refusal, got {other:?}"),
  }
}

/// A file that is not JSON is a read failure of `config.json`, not a field:
/// malformed text as invalid data, a document cut short as an early end.
#[test]
fn a_file_that_is_not_json_is_a_read_failure() {
  for (text, kind) in [
    ("{\"base_model\": ]", std::io::ErrorKind::InvalidData),
    ("{\"base_model\": ", std::io::ErrorKind::UnexpectedEof),
  ] {
    match check_text(text) {
      Err(Error::Read(failure)) => {
        assert_eq!(failure.artifact(), Artifact::Config);
        assert_eq!(failure.io_error().kind(), kind, "{text}");
      }
      other => panic!("expected a read failure, got {other:?}"),
    }
  }
}

/// The scale the gather multiplies by is f32(√512), `0x41B504F3`.
#[test]
fn the_embed_scale_is_the_f32_square_root_of_512() {
  assert_eq!(EMBED_SCALE.to_bits(), 0x41B5_04F3);
  assert_eq!(EMBED_SCALE, (HIDDEN_SIZE as f32).sqrt());
  assert_eq!(EMBED_SCALE_LITERAL as f32, EMBED_SCALE);
}

/// The caps: a `config.json` one byte past its 1 MiB cap is refused by name,
/// at the top level, before a byte of it is read: it is unreadable here, so
/// any read would fail with another error.
#[test]
fn a_config_past_its_cap_is_refused_unread() {
  use std::os::unix::fs::PermissionsExt;

  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("config.json");
  std::fs::File::create(&path)
    .unwrap()
    .set_len(CONFIG_JSON_CAP + 1)
    .unwrap();
  std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
  let result = load(&path);
  std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
  match result {
    Err(Error::Config(mismatch)) => {
      assert_eq!(mismatch.field().as_str(), "");
      assert!(
        mismatch
          .actual()
          .contains(&format!("at least {} bytes", CONFIG_JSON_CAP + 1)),
        "{}",
        mismatch.actual()
      );
    }
    other => panic!("expected the cap refused, got {other:?}"),
  }
}

/// The read door reads into the buffer it reserved at the document's size
/// and never grows it. A `config.json` that grows between its size and its
/// read, still valid JSON, is refused at its cap: within the cap as a document
/// that grew while it was read, past the cap by its size.
#[test]
fn a_document_that_grows_while_read_is_refused_at_its_cap() {
  use std::io::Write as _;

  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("config.json");
  for (grown_by, past) in [(64usize, false), (CONFIG_JSON_CAP as usize, true)] {
    std::fs::write(&path, CONFIG).unwrap();
    let _hook = super::super::artifacts::tree::after_size::set(move |at| {
      std::fs::OpenOptions::new()
        .append(true)
        .open(at)
        .unwrap()
        .write_all(&vec![b' '; grown_by])
        .unwrap();
    });
    match load(&path) {
      Err(Error::Config(mismatch)) => {
        assert_eq!(mismatch.field().as_str(), "");
        let said = if past {
          "past its 1048576-byte cap".to_owned()
        } else {
          format!("grew past the {} bytes", CONFIG.len())
        };
        assert!(mismatch.actual().contains(&said), "{}", mismatch.actual());
      }
      other => panic!("expected the grown document refused, got {other:?}"),
    }
  }
}

/// An object of the contract is read from an object only: `token_table`
/// written as an array of its values in order is refused at `token_table`.
#[test]
fn an_object_written_as_an_array_is_refused_at_its_field() {
  let mut value: Value = serde_json::from_str(CONFIG).unwrap();
  value["token_table"] = serde_json::json!(["embeddings.bf16", "bfloat16", [262144, 512]]);
  match check_text(&value.to_string()) {
    Err(Error::Config(m)) => {
      assert_eq!(m.field().as_str(), "token_table");
      assert!(m.actual().contains("sequence"), "{}", m.actual());
    }
    other => panic!("expected a config refusal, got {other:?}"),
  }
}

/// A `null` is a value of another type, not absence: it is refused at its
/// field as what it is.
#[test]
fn a_null_is_refused_apart_from_absence() {
  let pretty =
    serde_json::to_string_pretty(&serde_json::from_str::<Value>(CONFIG).unwrap()).unwrap();
  let planted = pretty.replacen("\"hidden_size\": 512", "\"hidden_size\": null", 1);
  assert_ne!(planted, pretty, "the fixture holds hidden_size");
  match check_text(&planted) {
    Err(Error::Config(m)) => {
      assert_eq!(m.field().as_str(), "hidden_size");
      assert!(m.actual().contains("null"), "{}", m.actual());
    }
    other => panic!("expected a config refusal, got {other:?}"),
  }
}

/// A refusal the parse raises is named by the deepest field of the contract
/// on its path: a negative number and a value of another type each at its
/// own field; a number too large for its field at that field; an element of
/// an array of integers at the array.
#[test]
fn a_refusal_of_the_parse_is_named_by_its_field() {
  let cases: [(&str, &str, &str); 4] = [
    ("\"frames\": 1000", "\"frames\": -1", "audio.frames"),
    (
      "\"embed_scale\": 22.627417",
      "\"embed_scale\": \"22.627417\"",
      "embed_scale",
    ),
    (
      "\"slots\": 8",
      "\"slots\": 18446744073709551616",
      "functions.pack_256.slots",
    ),
    ("768,\n    512", "768,\n    \"512\"", "matryoshka_dims"),
  ];
  let pretty =
    serde_json::to_string_pretty(&serde_json::from_str::<Value>(CONFIG).unwrap()).unwrap();
  for (from, to, field) in cases {
    assert!(pretty.contains(from), "the fixture holds `{from}`");
    match check_text(&pretty.replacen(from, to, 1)) {
      Err(Error::Config(m)) => {
        assert_eq!(m.field().as_str(), field, "{from} → {to}");
        assert_ne!(m.actual(), "absent", "{from} → {to}");
      }
      other => panic!("expected a config refusal for {to}, got {other:?}"),
    }
  }
}
