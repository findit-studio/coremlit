use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

use super::*;

/// The chordai base960h CTC vocabulary, byte-for-byte from
/// `Models/alignkit/base960h_dict.json` (SHA-256
/// `ef41495ab958d4416ad2f81ea51a77d4a3c79cace96e92e978c443c7bfbdd2e5`, the
/// same file `tests/model_io.rs` pins), transcribed per this module's `#
/// Generator note`, id-ascending, one entry per dict key. Not re-read from
/// `Models/` at test time so these tests stay hermetic — no
/// `ALIGNKIT_TEST_MODELS` download required.
const DICT_ENTRIES: [(&str, u32); VOCAB_SIZE] = [
  ("-", 0),
  ("|", 1),
  ("E", 2),
  ("T", 3),
  ("A", 4),
  ("O", 5),
  ("N", 6),
  ("I", 7),
  ("H", 8),
  ("S", 9),
  ("R", 10),
  ("D", 11),
  ("L", 12),
  ("U", 13),
  ("M", 14),
  ("W", 15),
  ("C", 16),
  ("F", 17),
  ("G", 18),
  ("Y", 19),
  ("P", 20),
  ("B", 21),
  ("V", 22),
  ("K", 23),
  ("'", 24),
  ("X", 25),
  ("J", 26),
  ("Q", 27),
  ("Z", 28),
];

/// Path to the committed asset on disk. `CARGO_MANIFEST_DIR` is a
/// compile-time constant naming this crate's own source tree, which is
/// exactly what test binaries run against — unlike `tokenizer_json_bytes`'s
/// rustdoc caution about using it as a *runtime* asset path, that concern
/// doesn't apply here.
fn asset_path() -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("src/audio/align/assets/chordai_base960h_tokenizer.json")
}

/// Parses the embedded asset. Same call asry's `load_tokenizer_with_compat`
/// makes on its fast path (`asry/src/runner/aligner/aligner.rs:1203`):
/// `Tokenizer::from_bytes`, never `from_file`.
fn load_tokenizer() -> Tokenizer {
  Tokenizer::from_bytes(tokenizer_json_bytes()).expect("embedded asset must parse")
}

#[test]
fn vocab_size_is_29() {
  assert_eq!(VOCAB_SIZE, 29);
  assert_eq!(DICT_ENTRIES.len(), 29);
}

#[test]
fn blank_id_is_zero() {
  assert_eq!(BLANK_ID, 0);
}

#[test]
fn word_delimiter_is_pipe() {
  assert_eq!(WORD_DELIMITER, "|");
}

#[test]
fn embedded_bytes_match_the_committed_file_on_disk() {
  let disk_bytes = std::fs::read(asset_path()).expect("committed asset must be readable");
  assert_eq!(
    disk_bytes,
    tokenizer_json_bytes(),
    "include_bytes! must reflect the committed asset exactly"
  );
}

/// Mirrors asry's exact tokenizer-loading call shape end to end
/// (`asry/src/runner/aligner/aligner.rs:1198-1206`,
/// `load_tokenizer_with_compat`): read the path to bytes with
/// `std::fs::read`, then `Tokenizer::from_bytes`. The asset parses on the
/// first attempt — no compat-patch retry — because it already declares
/// `"model": {"type": "WordLevel", ...}` explicitly (see this module's `#
/// Generator note`).
#[test]
fn on_disk_asset_round_trips_through_asrys_loader_shape() {
  let bytes = std::fs::read(asset_path()).expect("read tokenizer asset");
  let tok = Tokenizer::from_bytes(&bytes).expect("Tokenizer::from_bytes parses the asset");
  assert_eq!(tok.get_vocab_size(true), VOCAB_SIZE);
}

/// Mirrors asry's `validate_vocab_dim`
/// (in `asry/src/runner/aligner/algorithm/encode.rs`): the tokenizer's
/// vocab size — with and without added tokens; this asset has none, so
/// both must agree — has to equal `VOCAB_SIZE` EXACTLY.
#[test]
fn tokenizer_vocab_size_matches_vocab_size_exactly() {
  let tok = load_tokenizer();
  assert_eq!(tok.get_vocab_size(true), VOCAB_SIZE);
  assert_eq!(tok.get_vocab_size(false), VOCAB_SIZE);
}

#[test]
fn blank_token_resolves_to_blank_id() {
  let tok = load_tokenizer();
  assert_eq!(tok.token_to_id("-"), Some(BLANK_ID));
}

/// Mirrors asry's `validate_word_delimiter_present`
/// (`asry/src/runner/aligner/aligner.rs:1125-1143`): `token_to_id("|")`
/// must resolve — asry looks the delimiter up dynamically rather than
/// assuming a fixed id, so this test checks the resolved id, not just
/// presence.
#[test]
fn word_delimiter_resolves_via_token_to_id() {
  let tok = load_tokenizer();
  assert_eq!(tok.token_to_id(WORD_DELIMITER), Some(1));
}

/// Round-trip: every dict entry resolves to its exact id through the
/// loaded tokenizer, and the loaded vocab has no extra entries beyond
/// those 29 — the property `Aligner::from_paths`-style construction
/// (design spec §6) depends on to align model output columns with vocab
/// tokens correctly.
#[test]
fn every_dict_entry_round_trips_through_token_to_id() {
  let tok = load_tokenizer();
  for (token, expected_id) in DICT_ENTRIES {
    assert_eq!(
      tok.token_to_id(token),
      Some(expected_id),
      "token {token:?} must resolve to id {expected_id}"
    );
  }
  assert_eq!(
    tok.get_vocab(true).len(),
    VOCAB_SIZE,
    "vocab must contain exactly these 29 entries, no more"
  );
}

// --- Mutation checks -------------------------------------------------
//
// The tests above would pass vacuously if the asset were, say, empty and
// every assertion happened to short-circuit past a parse failure before
// reaching a real check. These prove the loader and the checks above
// actually discriminate a corrupted asset from a valid one. Both mutate
// the embedded bytes in memory (`tokenizer_json_bytes()`), not a temp-dir
// copy — no filesystem needed, so these stay hermetic too.

/// Structural corruption: truncate the asset mid-object. Valid UTF-8 (the
/// cut lands on an ASCII byte) but no longer valid JSON, so
/// `Tokenizer::from_bytes` — the same call the loader shape above uses —
/// must reject it outright.
#[test]
fn truncated_asset_is_rejected_by_tokenizer_from_bytes() {
  let bytes = tokenizer_json_bytes();
  let truncated = &bytes[..bytes.len() / 2];
  assert!(
    std::str::from_utf8(truncated).is_ok(),
    "fixture assumption: the halfway point must still land on an ASCII byte"
  );
  assert!(
    Tokenizer::from_bytes(truncated).is_err(),
    "truncated JSON must not parse as a valid tokenizer"
  );
}

/// Semantic corruption #1: shift the `|` delimiter's id from 1 to 91 while
/// keeping the JSON otherwise well-formed (vocab count unchanged). This is
/// exactly the class of bug `validate_word_delimiter_present`-style checks
/// exist to catch: the file still "loads", but the delimiter resolves to
/// the wrong id, which would silently misalign every CTC frame around word
/// boundaries.
#[test]
fn corrupted_delimiter_id_is_caught_by_the_round_trip_check() {
  let text = std::str::from_utf8(tokenizer_json_bytes()).expect("asset is UTF-8");
  let needle = "\"|\": 1,";
  assert!(
    text.contains(needle),
    "fixture assumption: the delimiter's exact `{needle}` line must be present in the asset \
     for this mutation to actually corrupt it"
  );
  let mutated = text.replacen(needle, "\"|\": 91,", 1);
  let tok = Tokenizer::from_bytes(mutated.as_bytes()).expect("still structurally valid JSON");
  // Vocab count is unaffected by the id shift...
  assert_eq!(tok.get_vocab_size(true), VOCAB_SIZE);
  // ...but the id our checks require is no longer what's stored, so the
  // real round-trip assertion this test mirrors
  // (`word_delimiter_resolves_via_token_to_id`) would now fail:
  assert_ne!(tok.token_to_id(WORD_DELIMITER), Some(1));
  assert_eq!(tok.token_to_id(WORD_DELIMITER), Some(91));
}

/// Semantic corruption #2: delete one vocab entry (`"Q": 27,`) outright.
/// Still structurally valid JSON (the entry sits between two others, so
/// removing its whole line doesn't orphan a comma), but the tokenizer's
/// vocab size drops to 28 — exactly the "V != expected_v" case asry's
/// `validate_vocab_dim` exists to reject before it can corrupt an
/// alignment.
#[test]
fn corrupted_vocab_entry_removal_is_caught_by_vocab_size_check() {
  let text = std::str::from_utf8(tokenizer_json_bytes()).expect("asset is UTF-8");
  let needle = "\"Q\": 27,\n";
  assert!(
    text.contains(needle),
    "fixture assumption: the `{needle:?}` line must be present in the asset for this \
     mutation to actually corrupt it"
  );
  let mutated = text.replacen(needle, "", 1);
  let tok = Tokenizer::from_bytes(mutated.as_bytes()).expect("still structurally valid JSON");
  let corrupted_size = tok.get_vocab_size(true);
  assert_ne!(
    corrupted_size, VOCAB_SIZE,
    "removing an entry must change the observed vocab size"
  );
  assert_eq!(corrupted_size, VOCAB_SIZE - 1);
  assert_eq!(tok.token_to_id("Q"), None);
}
