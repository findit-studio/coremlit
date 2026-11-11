//! The committed R1 goldens of the EmbeddingGemma 2 kit and the R2 Swift
//! probe's dump: their provenance, their byte offsets, and their shapes.
//!
//! Hermetic: committed files only. The published package excludes
//! `tests/embeddinggemma2/fixtures/goldens/`, so a `cargo test` from the
//! tarball has no workspace root and no goldens; the laws that read them skip
//! there, each naming the file on stderr. Inside a workspace a missing golden
//! is a failure.

#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use serde_json::Value;
use sha2::{Digest, Sha256};

const GOLDENS: &str = "coremlit/tests/embeddinggemma2/fixtures/goldens";

/// The keys of the recipe's `REQUIRED_TOOLCHAIN`: every version a golden was
/// computed under.
const TOOLCHAIN: [&str; 10] = [
  "coremltools",
  "numpy",
  "pillow",
  "python",
  "safetensors",
  "sentence_transformers",
  "tokenizers",
  "torch",
  "torchvision",
  "transformers",
];

/// A committed golden, or `None` (said by name) outside a source tree.
fn golden(name: &str) -> Option<Value> {
  let Some(root) = workspace_root::try_workspace_root() else {
    eprintln!(
      "embeddinggemma2 goldens: skipped — {GOLDENS}/{name} is not in this source tree (a published tarball?)"
    );
    return None;
  };
  let path = root.join(GOLDENS).join(name);
  let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}")))
}

fn str_at<'v>(v: &'v Value, pointer: &str) -> &'v str {
  v.pointer(pointer)
    .and_then(Value::as_str)
    .unwrap_or_else(|| panic!("{pointer} is not a string"))
}

fn vector(v: &Value, what: &str) -> Vec<f64> {
  v.as_array()
    .unwrap_or_else(|| panic!("{what}: not an array"))
    .iter()
    .map(|x| x.as_f64().unwrap_or_else(|| panic!("{what}: not a number")))
    .collect()
}

fn assert_unit_768(v: &Value, what: &str) {
  let v = vector(v, what);
  assert_eq!(v.len(), 768, "{what}: width");
  let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
  assert!((norm - 1.0).abs() < 1e-5, "{what}: norm {norm}");
}

/// Every golden names the toolchain it was computed under, key for key, the
/// pinned source and staged revisions with the weights' digest, and the host.
#[test]
fn every_golden_pins_its_toolchain_sources_and_host() {
  for name in ["text.json", "audio.json", "image.json"] {
    let Some(doc) = golden(name) else { return };
    let toolchain = doc
      .pointer("/provenance/toolchain")
      .and_then(Value::as_object)
      .unwrap_or_else(|| panic!("{name}: no toolchain"));
    let mut keys: Vec<&str> = toolchain.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, TOOLCHAIN, "{name}: the toolchain keys");
    for (key, version) in toolchain {
      let version = version.as_str().unwrap_or_default();
      assert!(!version.is_empty(), "{name}: {key} has no version");
    }
    assert_eq!(
      str_at(&doc, "/provenance/source/revision"),
      "f9972ac22e696a105c482202646979e33e4b5a1e",
      "{name}"
    );
    assert_eq!(
      str_at(&doc, "/provenance/source/files/model.safetensors"),
      "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
      "{name}"
    );
    assert_eq!(
      str_at(&doc, "/provenance/staged/revision"),
      "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4",
      "{name}"
    );
    for field in ["machine", "cpu", "os"] {
      assert!(
        !str_at(&doc, &format!("/provenance/host/{field}")).is_empty(),
        "{name}: host {field}"
      );
    }
    let path = str_at(&doc, "/provenance/host/torchvision_uint8_antialias_path");
    assert!(matches!(path, "float" | "integer AVX2"), "{name}: {path}");
  }
}

/// The text goldens' offsets are UTF-8 BYTE offsets over `prefix + text`:
/// each span lies on character boundaries of the joint string, and the
/// content's last span ends at its byte length. Character offsets would end
/// short of it on every non-ASCII item.
#[test]
fn the_text_offsets_are_utf8_bytes_of_the_joint_string() {
  let Some(doc) = golden("text.json") else {
    return;
  };
  let items = doc["items"].as_array().expect("items");
  let mut non_ascii = 0;
  for item in items {
    let id = item["id"].as_str().expect("id");
    let joint = format!(
      "{}{}",
      item["prefix"].as_str().expect("prefix"),
      item["text"].as_str().expect("text")
    );
    if !joint.is_ascii() {
      non_ascii += 1;
    }
    for setting in ["recognize", "split"] {
      let offsets = item[setting]["offsets"].as_array().expect("offsets");
      let ids = item[setting]["ids"].as_array().expect("ids");
      assert_eq!(offsets.len(), ids.len(), "{id} {setting}");
      let mut end = 0;
      for span in offsets {
        let (a, b) = (
          span[0].as_u64().unwrap() as usize,
          span[1].as_u64().unwrap() as usize,
        );
        assert!(
          a <= b && b <= joint.len(),
          "{id} {setting}: {a}..{b} of {} bytes",
          joint.len()
        );
        assert!(
          joint.is_char_boundary(a) && joint.is_char_boundary(b),
          "{id} {setting}: {a}..{b} splits a character"
        );
        end = end.max(b);
      }
      assert_eq!(
        end,
        joint.len(),
        "{id} {setting}: the content ends at the joint string's byte length"
      );
    }
  }
  assert!(
    non_ascii >= 20,
    "the corpus carries non-ASCII items ({non_ascii})"
  );
}

/// The goldens have the kit's shapes: unit 768-d vectors; each text item's
/// unpacked function holds its ids; each audio window's token count is the
/// mask's closed form; each image budget's geometry is HF's.
#[test]
fn the_goldens_have_the_kits_shapes() {
  if let Some(text) = golden("text.json") {
    for item in text["items"].as_array().expect("items") {
      let id = item["id"].as_str().expect("id");
      let n = item["recognize"]["ids"].as_array().expect("ids").len();
      if n <= 512 {
        assert_unit_768(&item["vector"], id);
        let s = [32, 48, 64, 128, 256, 512]
          .into_iter()
          .find(|s| *s >= n)
          .unwrap();
        assert_eq!(
          item["function"].as_str(),
          Some(format!("embed_{s}").as_str()),
          "{id}"
        );
        assert_eq!(
          item["inputs_embeds_sha256"].as_str().map(str::len),
          Some(64),
          "{id}"
        );
      } else {
        assert!(
          item.get("vector").is_none(),
          "{id}: a long text carries no vector"
        );
      }
      if let Some(split) = item.get("vector_split") {
        assert_unit_768(split, id);
      }
    }
    let framing = &text["framing"];
    for (key, framed) in [
      ("document", 8),
      ("search_query", 9),
      ("classification", 8),
      ("bare", 2),
    ] {
      assert_eq!(framing[key].as_u64(), Some(framed), "the framing of {key}");
    }
  }
  if let Some(audio) = golden("audio.json") {
    for clip in audio["clips"].as_array().expect("clips") {
      let id = clip["id"].as_str().expect("id");
      for window in clip["windows"].as_array().expect("windows") {
        let samples = window["samples"].as_u64().unwrap() as usize;
        let frames = if 160 + samples < 321 {
          0
        } else {
          ((160 + samples - 321) / 160 + 1).min(1000)
        };
        assert_eq!(
          window["tokens"].as_u64(),
          Some(frames.div_ceil(4) as u64),
          "{id}"
        );
        assert_eq!(window["start"].as_u64().unwrap() % 160_000, 0, "{id}");
        assert_unit_768(&window["vector"], id);
      }
    }
  }
  if let Some(image) = golden("image.json") {
    for entry in image["images"].as_array().expect("images") {
      let id = entry["id"].as_str().expect("id");
      for budget in [70u64, 140, 280] {
        let at = &entry["budgets"][budget.to_string()];
        let (h, w) = (
          at["resized"][0].as_u64().unwrap(),
          at["resized"][1].as_u64().unwrap(),
        );
        assert!(h % 48 == 0 && w % 48 == 0, "{id} @ {budget}: {h} x {w}");
        let patches = at["patches"].as_u64().unwrap();
        assert_eq!(patches, (h / 16) * (w / 16), "{id} @ {budget}");
        assert!(patches <= 9 * budget, "{id} @ {budget}");
        assert_eq!(at["tokens"].as_u64(), Some(patches / 9), "{id} @ {budget}");
        assert_unit_768(&at["vector"], id);
        assert_eq!(
          at["tiles"].as_array().map(Vec::len),
          Some(2),
          "{id} @ {budget}"
        );
      }
    }
    for synthetic in image["synthetic"]["budgets"].as_array().expect("synthetic") {
      let t = synthetic["budget"].as_u64().unwrap();
      let n = 9 * t;
      let inputs = &synthetic["inputs"];
      for (input, shape) in [
        ("patches", vec![1, n, 768]),
        ("positions", vec![1, n, 2]),
        ("valid", vec![1, n]),
        ("pool", vec![t, n]),
        ("position_embeddings", vec![1, n, 768]),
      ] {
        let got: Vec<u64> = inputs[input]["shape"]
          .as_array()
          .unwrap()
          .iter()
          .map(|d| d.as_u64().unwrap())
          .collect();
        assert_eq!(got, shape, "{input} @ {t}");
      }
      assert!(
        inputs["grid"]["patches"].as_u64().unwrap() < n,
        "padding patches @ {t}"
      );
    }
    assert_eq!(
      image["position_table"]["equal_to_staged"].as_bool(),
      Some(true)
    );
    assert_eq!(
      image["position_table"]["token_table"]["equal_to_staged_bytes"].as_bool(),
      Some(true)
    );
  }
}

/// The gather golden is the file its provenance record describes, with the
/// design's counts.
#[test]
fn the_gather_golden_is_its_provenance_record() {
  let dir = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/embeddinggemma2/fixtures"
  );
  let bytes = std::fs::read(format!("{dir}/gather_f16.bin")).expect("the gather golden");
  let record: Value = serde_json::from_str(
    &std::fs::read_to_string(format!("{dir}/gather_f16.provenance.json")).expect("its record"),
  )
  .expect("JSON");
  assert_eq!(bytes.len(), 131_072);
  let digest: String = Sha256::digest(&bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
  assert_eq!(record["sha256"].as_str(), Some(digest.as_str()));
  assert_eq!(record["counts"]["finite_inputs"].as_u64(), Some(65_280));
  assert_eq!(
    record["counts"]["finite_inputs_overflowing_to_infinity"].as_u64(),
    Some(29_846)
  );
  assert_eq!(record["counts"]["nan_inputs"].as_u64(), Some(254));
  assert!(
    record["toolchain"]["numpy"]
      .as_str()
      .is_some_and(|v| !v.is_empty())
  );
}

/// R2, the Swift probe's dump (`swift_probe.json`): it pins the FluidUse
/// revision and host it ran on, refused nothing, and covers the committed
/// corpus item for item. Every text item of at most 512 tokens, every audio
/// window of every clip (the clip files the goldens hashed), and one synthetic
/// image per budget, each with the predictions FluidUse made in order (an
/// unpacked text function; the audio model then a text function; `vision_N`
/// then a text function) and a 768-d output of unit norm within fp16's reach.
#[test]
fn the_swift_probe_pins_fluiduse_and_covers_the_corpus() {
  let Some(probe) = golden("swift_probe.json") else {
    return;
  };
  assert_eq!(str_at(&probe, "/provenance/probe"), "R2");
  assert_eq!(
    str_at(&probe, "/provenance/fluiduse_revision"),
    "b51f7adfbade197f0a9c802e34ba8a98cd4ecbc9",
    "the FluidUse revision Package.swift pins"
  );
  for field in ["swift_version", "sw_vers", "cpu"] {
    assert!(
      !str_at(&probe, &format!("/provenance/{field}")).is_empty(),
      "the probe's host {field}"
    );
  }
  assert_eq!(
    probe["refused"].as_array().map(Vec::len),
    Some(0),
    "the probe refused an item"
  );
  let near_unit = |v: &Value, what: &str| {
    let v = vector(v, what);
    assert_eq!(v.len(), 768, "{what}: width");
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "{what}: norm {norm}");
  };
  let functions = |entry: &Value| -> Vec<String> {
    entry["captures"]
      .as_array()
      .expect("captures")
      .iter()
      .map(|c| c["function"].as_str().unwrap_or("(default)").to_owned())
      .collect()
  };
  let unpacked = |name: &str| {
    name
      .strip_prefix("embed_")
      .is_some_and(|s| ["32", "48", "64", "128", "256", "512"].contains(&s))
  };

  if let Some(text) = golden("text.json") {
    let expected: Vec<&str> = text["items"]
      .as_array()
      .expect("items")
      .iter()
      .filter(|item| item["recognize"]["ids"].as_array().expect("ids").len() <= 512)
      .map(|item| item["id"].as_str().expect("id"))
      .collect();
    let probed = probe["text"].as_array().expect("text");
    let ids: Vec<&str> = probed
      .iter()
      .map(|e| e["id"].as_str().expect("id"))
      .collect();
    assert_eq!(
      ids, expected,
      "one probe entry per text item of at most 512 tokens"
    );
    for entry in probed {
      let id = entry["id"].as_str().expect("id");
      assert!(
        !entry["ids"].as_array().expect("ids").is_empty(),
        "{id}: no ids"
      );
      let called = functions(entry);
      assert!(
        called.len() == 1 && unpacked(&called[0]),
        "{id}: {called:?}"
      );
      near_unit(&entry["vector"], id);
    }
  }
  if let Some(audio) = golden("audio.json") {
    let clips = audio["clips"].as_array().expect("clips");
    let probed = probe["audio"].as_array().expect("audio");
    assert_eq!(probed.len(), clips.len(), "one probe entry per clip");
    for (clip, entry) in clips.iter().zip(probed) {
      let id = clip["id"].as_str().expect("id");
      assert_eq!(entry["id"].as_str(), Some(id), "the clips in order");
      assert_eq!(
        entry["file_sha256"].as_str(),
        clip["file_sha256"].as_str(),
        "{id}: the clip the goldens hashed"
      );
      let windows = entry["windows"].as_array().expect("windows");
      assert_eq!(
        windows.len(),
        clip["windows"].as_array().expect("windows").len(),
        "{id}: every window"
      );
      for window in windows {
        let called = functions(window);
        assert!(
          called.len() == 2 && called[0] == "(default)" && unpacked(&called[1]),
          "{id}: {called:?}"
        );
        near_unit(&window["vector"], id);
      }
    }
  }
  let probed = probe["image"].as_array().expect("image");
  let budgets: Vec<u64> = probed
    .iter()
    .map(|e| e["budget"].as_u64().expect("budget"))
    .collect();
  assert_eq!(budgets, [70, 140, 280], "one synthetic image per budget");
  for entry in probed {
    let budget = entry["budget"].as_u64().expect("budget");
    let called = functions(entry);
    assert!(
      called.len() == 2 && called[0] == format!("vision_{budget}") && unpacked(&called[1]),
      "image @ {budget}: {called:?}"
    );
    near_unit(&entry["vector"], &format!("image @ {budget}"));
  }
}
