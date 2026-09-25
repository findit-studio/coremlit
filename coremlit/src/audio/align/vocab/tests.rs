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

// --- Vocabulary: a model's own table, read at run time -----------------

/// SHA-256 of `Models/alignkit/base960h_dict.json`, the pin this module's
/// `# Generator note` records.
const STAGED_DICT_SHA256: &str = "ef41495ab958d4416ad2f81ea51a77d4a3c79cace96e92e978c443c7bfbdd2e5";

/// `base960h_dict.json`'s exact bytes, rebuilt from [`DICT_ENTRIES`] in the
/// file's own spelling — one line, `"token": id` pairs joined by `", "`, no
/// trailing newline — and proved byte-identical by its SHA-256.
fn staged_dict() -> Vec<u8> {
  let entries: Vec<String> = DICT_ENTRIES
    .iter()
    .map(|(token, id)| format!("\"{token}\": {id}"))
    .collect();
  let dict = format!("{{{}}}", entries.join(", ")).into_bytes();
  assert_eq!(
    sha256_hex(&dict),
    STAGED_DICT_SHA256,
    "the rebuilt table must be the staged file, byte for byte"
  );
  dict
}

fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|byte| format!("{byte:02x}"))
    .collect()
}

fn json(bytes: &[u8]) -> serde_json::Value {
  serde_json::from_slice(bytes).expect("JSON")
}

/// **The staged model's own table reads back as the bundled one.** Read
/// through [`Vocabulary::from_json`], `base960h_dict.json` has the bundled
/// table's 29 entries, and the tokenizer document written for it is the
/// committed asset, field for field: the run-time road and the committed asset
/// are the same generator rule set. The model-gated half aligns the staged
/// model through both and compares the words (`tests/align/align_chunk.rs`).
#[test]
fn the_staged_dict_reads_back_as_the_bundled_table() {
  let vocabulary = Vocabulary::from_json(&staged_dict()).expect("the staged table reads");
  assert_eq!(vocabulary.size().get(), VOCAB_SIZE);
  assert_eq!(
    json(vocabulary.tokenizer_json()),
    json(tokenizer_json_bytes()),
    "the written document must be the committed asset"
  );

  let tok = Tokenizer::from_bytes(vocabulary.tokenizer_json()).expect("the document parses");
  for (token, id) in DICT_ENTRIES {
    assert_eq!(tok.token_to_id(token), Some(id), "{token:?}");
  }
}

#[test]
fn bundled_is_the_committed_table() {
  let bundled = Vocabulary::bundled();
  assert_eq!(bundled.size().get(), VOCAB_SIZE);
  assert_eq!(bundled.tokenizer_json(), tokenizer_json_bytes());
}

/// **A table carries no blank, whatever its names.** A flat table does not say
/// which column the head scores as "no token", and names are no answer: a table
/// can hold `<blank>` at 0 and an ordinary `<pad>` at 1, a `-` that is the
/// hyphen, or no conventional name at all. Every one of these reads as the
/// plain table it is — each token at its own id, none singled out — and the
/// blank is left to the model's contract, which the aligner checks against the
/// table's ids at load (`aligner::tests`).
#[test]
fn a_table_carries_no_blank_whatever_its_names() {
  for (table, size) in [
    (
      r#"{"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5}"#,
      6,
    ),
    (r#"{"a": 0, "b": 1, "|": 2, "[UNK]": 3, "[PAD]": 4}"#, 5),
    (r#"{"<blank>": 0, "<pad>": 1, "a": 2}"#, 3),
    (r#"{"-": 0, "<pad>": 1, "a": 2}"#, 3),
    (r#"{"a": 0, "-": 1, "b": 2}"#, 3),
    (r#"{"a": 0, "b": 1}"#, 2),
  ] {
    let vocabulary = Vocabulary::from_json(table.as_bytes()).expect("the table reads");
    assert_eq!(vocabulary.size().get(), size, "{table}");
    let document = json(vocabulary.tokenizer_json());
    assert_eq!(
      document["model"]["vocab"],
      json(table.as_bytes()),
      "{table}: every token at its own id"
    );
    assert_eq!(
      document["added_tokens"],
      serde_json::json!([]),
      "{table}: no token is special"
    );
  }
}

/// **A table that does not name every id once is refused by name.** A CTC
/// head has one column per class and an id is the column its token is scored
/// in, so a table that skips an id leaves a column unnamed, and one that gives
/// two tokens the same id reads one column for both. Both are
/// `VocabularyError::MissingId`, naming the lowest id no token holds.
#[test]
fn a_table_that_skips_or_repeats_an_id_is_refused_by_name() {
  for (table, id) in [
    (r#"{"-": 0, "|": 2}"#, 1),
    (r#"{"-": 0, "a": 0}"#, 1),
    (r#"{"<pad>": 1, "a": 2}"#, 0),
  ] {
    let Err(VocabularyError::MissingId(missing)) = Vocabulary::from_json(table.as_bytes()) else {
      panic!("{table} must be refused as a missing id");
    };
    assert_eq!((missing.id(), missing.entries()), (id, 2), "{table}");
  }
}

/// **A token the object names twice is refused by name.** JSON leaves a
/// repeated key's meaning to the reader — one keeps the first id, another the
/// last — and a map built by insertion silently kept one:
/// `{"<pad>": 0, "A": 0, "A": 1}` collapsed into a valid two-entry table that
/// scored `A` from a column its own file never settled. The object is read
/// entry by entry now, so the repetition is seen and named, even when the ids
/// agree and even when the second spelling is an escape.
#[test]
fn a_token_named_twice_is_refused_by_name() {
  for (table, token) in [
    (r#"{"<pad>": 0, "A": 0, "A": 1}"#, "A"),
    (r#"{"-": 0, "a": 1, "a": 1}"#, "a"),
    (r#"{"<pad>": 0, "A": 1, "A": 2}"#, "A"),
  ] {
    let Err(VocabularyError::DuplicateToken(repeated)) = Vocabulary::from_json(table.as_bytes())
    else {
      panic!("{table} must be refused as a repeated token");
    };
    assert_eq!(repeated, token, "{table}");
  }
}

/// A table that names no token is no vocabulary: a CTC head has at least one
/// class, its blank.
#[test]
fn an_empty_table_is_refused_by_name() {
  assert!(matches!(
    Vocabulary::from_json(b"{}"),
    Err(VocabularyError::Empty)
  ));
}

/// Anything but a flat object of token → non-negative `u32` id is not a table
/// — the bundled `tokenizer.json` document included, whose `version` is a
/// string.
#[test]
fn bytes_that_are_not_a_token_table_are_refused_by_name() {
  let inputs: [&[u8]; 7] = [
    b"[1, 2]",
    br#"{"-": -1}"#,
    br#"{"-": "0"}"#,
    br#"{"-": 1.5}"#,
    br#"{"-": 4294967296}"#,
    b"not json",
    tokenizer_json_bytes(),
  ];
  for input in inputs {
    assert!(
      matches!(Vocabulary::from_json(input), Err(VocabularyError::Parse(_))),
      "{}",
      String::from_utf8_lossy(input)
    );
  }
}

/// [`Vocabulary::from_file`] reads the table that ships beside a model, and a
/// file it cannot read is refused naming that file, with the I/O failure as its
/// source.
#[test]
fn from_file_reads_the_table_beside_a_model() {
  let dir = tempfile::tempdir().expect("a temporary directory");
  let path = dir.path().join("base960h_dict.json");
  std::fs::write(&path, staged_dict()).expect("write the table");
  let vocabulary = Vocabulary::from_file(&path).expect("the table reads");
  assert_eq!(vocabulary.size().get(), VOCAB_SIZE);

  let missing = dir.path().join("absent_dict.json");
  let Err(VocabularyError::Read(read)) = Vocabulary::from_file(&missing) else {
    panic!("an absent file must be refused as a read failure");
  };
  assert_eq!(read.path(), missing);
  let source = std::error::Error::source(&read)
    .and_then(|source| source.downcast_ref::<std::io::Error>())
    .expect("the I/O failure is the source");
  assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn debug_names_the_size() {
  assert_eq!(
    format!("{:?}", Vocabulary::bundled()),
    "Vocabulary { size: 29, .. }"
  );
}
