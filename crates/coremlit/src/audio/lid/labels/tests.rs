use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::*;

/// SHA-256 of the committed label asset. Byte-identical to the copy in the
/// artifact author's MLX export of the same model.
const LABELS_SHA256: &str = "f13f0331965a4a402f4308ed80de662f2d55167d77e163406712ba170b92eb35";

/// One asset entry, deserialized for the row-by-row comparison below.
/// `serde_json` is a dev-dependency, so the `lid` feature never pulls a JSON
/// parser into a consumer's graph — the asset is parsed HERE and nowhere else.
#[derive(serde::Deserialize)]
struct AssetEntry {
  code: String,
  id: usize,
  name: String,
  upstream_label: String,
}

fn asset_entries() -> Vec<AssetEntry> {
  serde_json::from_slice(labels_json_bytes()).expect("the embedded asset must parse")
}

/// Path to the committed asset on disk. `CARGO_MANIFEST_DIR` is a compile-time
/// constant naming this crate's own source tree, which is what test binaries
/// run against — the caution in [`labels_json_bytes`]'s docs is about using it
/// as a RUNTIME path, which this is not.
fn asset_path() -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR")).join("src/audio/lid/assets/voxlingua107_labels.json")
}

#[test]
fn embedded_bytes_match_the_committed_file_on_disk() {
  let on_disk = std::fs::read(asset_path()).expect("committed asset must be readable");
  assert_eq!(
    on_disk,
    labels_json_bytes(),
    "include_bytes! must reflect the committed asset exactly"
  );
}

#[test]
fn asset_length_and_digest_are_pinned() {
  assert_eq!(labels_json_bytes().len(), LABELS_JSON_LEN);
  assert_eq!(LABELS_JSON_LEN, 10_756);
  let digest = Sha256::digest(labels_json_bytes())
    .iter()
    .fold(String::new(), |mut acc, b| {
      use core::fmt::Write;
      let _ = write!(acc, "{b:02x}");
      acc
    });
  assert_eq!(digest, LABELS_SHA256);
}

/// The roster and the asset agree ROW BY ROW: same length, same order, same
/// codes, same names, same indices.
///
/// This is the check that makes the hand-written table safe. Without it the
/// table is an unverified transcription of 107 rows, and a single transposed
/// pair would mislabel two languages forever while every other test still
/// passed.
#[test]
fn every_table_row_matches_the_committed_asset() {
  let entries = asset_entries();
  assert_eq!(entries.len(), NUM_LANGUAGES);
  assert_eq!(languages().len(), NUM_LANGUAGES);

  for (entry, language) in entries.iter().zip(languages().iter()) {
    assert_eq!(language.index(), entry.id, "index drift at {}", entry.code);
    assert_eq!(
      language.code(),
      entry.code,
      "code drift at index {}",
      entry.id
    );
    assert_eq!(
      language.name(),
      entry.name,
      "name drift at index {}",
      entry.id
    );
  }
}

/// Every asset entry's `upstream_label` is exactly upstream's own
/// `label_encoder.txt` line, `"<code>: <name>"` — the mechanical form of the
/// claim that this roster reproduces upstream in index order. A roster
/// refreshed from a re-derived or "cleaned" list would break this before it
/// broke anything a user could see.
#[test]
fn asset_upstream_labels_reconstruct_the_label_encoder_lines() {
  for entry in asset_entries() {
    assert_eq!(
      entry.upstream_label,
      format!("{}: {}", entry.code, entry.name),
      "entry {} does not reconstruct its upstream label",
      entry.id
    );
  }
}

/// The roster is exactly `NUM_LANGUAGES` long and indices are dense and
/// ascending: `languages()[i].index() == i`.
#[test]
fn indices_are_dense_and_positional() {
  for (i, language) in languages().iter().enumerate() {
    assert_eq!(language.index(), i);
    assert_eq!(Language::from_index(i), Some(language));
  }
  assert_eq!(Language::from_index(NUM_LANGUAGES), None);
  assert_eq!(Language::from_index(usize::MAX), None);
}

/// Semantic pins on specific columns. These are the rows every other check
/// would still pass without: the ends of the roster, and the entry the
/// end-to-end anchor lands on.
#[test]
fn index_pins_are_stable() {
  let pins = [
    (0, "ab", "Abkhazian"),
    (52, "la", "Latin"),
    (55, "lo", "Lao"),
    (94, "th", "Thai"),
    (106, "zh", "Chinese"),
  ];
  for (index, code, name) in pins {
    let language = Language::from_index(index).expect("pinned index must resolve");
    assert_eq!(language.code(), code, "code at index {index}");
    assert_eq!(language.name(), name, "name at index {index}");
  }
}

/// The legacy ISO 639-1 codes are preserved EXACTLY as upstream spells them.
///
/// `iw` (Hebrew) and `jw` (Javanese) are the pre-1989 codes; the modern ones
/// are `he` and `jv`. Modernizing them here would shift nothing by itself, but
/// it would put this roster out of step with the label encoder the graph's
/// output columns were trained against — and the next person to re-derive the
/// table from a "clean" list would have no way to notice. The alias mapping is
/// a downstream concern, which is why the modern spellings deliberately do NOT
/// resolve here.
#[test]
fn legacy_hebrew_and_javanese_codes_are_preserved() {
  assert_eq!(Language::from_index(44).map(Language::code), Some("iw"));
  assert_eq!(Language::from_index(44).map(Language::name), Some("Hebrew"));
  assert_eq!(Language::from_index(46).map(Language::code), Some("jw"));
  assert_eq!(
    Language::from_index(46).map(Language::name),
    Some("Javanese")
  );

  // The modern spellings are absent, on purpose: folding them in here would
  // silently mask a roster that had been modernized.
  assert_eq!(Language::from_code("he"), None);
  assert_eq!(Language::from_code("jv"), None);
  assert_eq!(Language::from_code("iw").map(Language::index), Some(44));
  assert_eq!(Language::from_code("jw").map(Language::index), Some(46));

  // Indonesian is NOT in the same situation: upstream already uses the modern
  // `id`, so no alias question arises for it.
  assert_eq!(Language::from_code("in"), None);
  assert!(Language::from_code("id").is_some());
}

/// Codes are unique and code-sorted — the precondition
/// [`Language::from_code`]'s binary search depends on. An unsorted roster would
/// make that lookup return wrong answers rather than fail, so this is a
/// correctness pin, not a tidiness one.
#[test]
fn codes_are_unique_and_sorted() {
  let codes: Vec<&str> = languages().iter().map(Language::code).collect();
  let mut sorted = codes.clone();
  sorted.sort_unstable();
  assert_eq!(codes, sorted, "the roster must stay code-sorted");
  sorted.dedup();
  assert_eq!(sorted.len(), NUM_LANGUAGES, "codes must be unique");

  // The search agrees with a linear scan for every code, and rejects misses.
  for language in languages() {
    assert_eq!(Language::from_code(language.code()), Some(language));
  }
  for miss in ["", "z", "zzz", "TH", "th ", "xx"] {
    assert_eq!(Language::from_code(miss), None, "{miss:?} must not resolve");
  }
}

// --- Mutation checks -------------------------------------------------
//
// The comparisons above would pass vacuously if the asset were, say, empty and
// the parse short-circuited. These prove the parse and the row check actually
// discriminate a corrupted asset from a valid one. Both mutate the embedded
// bytes in memory, so they stay hermetic.

/// Structural corruption: truncate the asset mid-array. Still valid UTF-8, no
/// longer valid JSON, so the parse every check above depends on must reject it.
#[test]
fn truncated_asset_is_rejected_by_the_parser() {
  let bytes = labels_json_bytes();
  let truncated = &bytes[..bytes.len() / 2];
  assert!(
    core::str::from_utf8(truncated).is_ok(),
    "fixture assumption: the halfway point must land on an ASCII byte"
  );
  assert!(
    serde_json::from_slice::<Vec<AssetEntry>>(truncated).is_err(),
    "truncated JSON must not parse as a roster"
  );
}

/// Semantic corruption: swap two adjacent language names in the asset while
/// leaving the JSON well-formed and the entry count unchanged. This is exactly
/// the class of drift `every_table_row_matches_the_committed_asset` exists to
/// catch — the file still loads, the roster is still 107 long, and two
/// languages are now mislabelled.
#[test]
fn a_swapped_name_is_caught_by_the_row_comparison() {
  let text = core::str::from_utf8(labels_json_bytes()).expect("asset is UTF-8");
  let needle = "\"name\": \"Hebrew\"";
  assert!(
    text.contains(needle),
    "fixture assumption: `{needle}` must be present for this mutation to corrupt anything"
  );
  let mutated = text.replacen(needle, "\"name\": \"Hindi\"", 1);
  let entries: Vec<AssetEntry> =
    serde_json::from_str(&mutated).expect("still structurally valid JSON");

  assert_eq!(entries.len(), NUM_LANGUAGES, "the count is unaffected");
  let mismatches = entries
    .iter()
    .zip(languages().iter())
    .filter(|(entry, language)| language.name() != entry.name)
    .count();
  assert_eq!(
    mismatches, 1,
    "the row comparison must see exactly the one corrupted name"
  );
}
