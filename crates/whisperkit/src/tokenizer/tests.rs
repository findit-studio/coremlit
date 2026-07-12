use std::path::PathBuf;

use super::*;

fn tiny() -> WhisperTokenizer {
  let root = std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  );
  WhisperTokenizer::from_folder(root.join("tokenizers/whisper-tiny")).unwrap()
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn special_tokens_match_swift_defaults() {
  let t = tiny();
  let s = t.special_tokens();
  assert_eq!(s.start_of_transcript_token(), 50258);
  assert_eq!(s.end_token(), 50257);
  assert_eq!(s.transcribe_token(), 50359);
  assert_eq!(s.translate_token(), 50358);
  assert_eq!(s.no_timestamps_token(), 50363);
  assert_eq!(s.time_token_begin(), 50364);
  assert_eq!(s.no_speech_token(), 50362);
  assert_eq!(s.start_of_previous_token(), 50361);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn encode_decode_round_trip() {
  let t = tiny();
  let ids = t.encode(" Hello world").unwrap();
  assert!(!ids.is_empty());
  assert_eq!(t.decode(&ids, false).unwrap(), " Hello world");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn language_tokens_cover_the_table() {
  let t = tiny();
  assert!(t.all_language_tokens().len() >= 96); // tiny is multilingual: ~99 language tokens
  let en = t.token_to_id("<|en|>").unwrap();
  assert_eq!(t.language_for_token(en), Some("en"));
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn split_words_space_vs_unicode() {
  let t = tiny();
  let ids = t.encode(" Hello world").unwrap();
  let words = t.split_to_word_tokens(&ids, "en").unwrap();
  let texts: Vec<&str> = words.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(texts, vec![" Hello", " world"]);
  // unicode-split path: every CJK char its own word
  let zh = t.encode("你好世界").unwrap();
  let words = t.split_to_word_tokens(&zh, "zh").unwrap();
  assert!(words.len() >= 4 || words.iter().all(|(w, _)| !w.contains(' ')));
}

// ---------------------------------------------------------------------
// Additional coverage beyond the brief's four fixed tests.
// ---------------------------------------------------------------------

#[test]
fn from_folder_missing_file_reports_searched_path() {
  // Hermetic: `src/` always exists (it's this crate's own source root) but
  // never contains a `tokenizer.json`, so this needs no tokenizer fixture
  // and no filesystem mutation/cleanup.
  let folder = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
  let err = WhisperTokenizer::from_folder(&folder).unwrap_err();
  match err {
    TokenizerError::FileNotFound { searched } => {
      assert_eq!(searched, vec![folder.join("tokenizer.json")]);
    }
    other => panic!("expected FileNotFound, got {other:?}"),
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn special_tokens_remaining_fields_match_swift_defaults() {
  // The brief's own test covers 8 of the 11 fields; these are the rest.
  let t = tiny();
  let s = t.special_tokens();
  assert_eq!(s.special_token_begin(), 50257);
  assert_eq!(s.english_token(), 50259);
  assert_eq!(s.whitespace_token(), 220);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn token_to_id_id_to_token_round_trip() {
  let t = tiny();
  let id = t.token_to_id("<|en|>").unwrap();
  assert_eq!(id, t.special_tokens().english_token());
  assert_eq!(t.id_to_token(id).as_deref(), Some("<|en|>"));
  assert_eq!(t.token_to_id("<|this_token_does_not_exist|>"), None);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn all_language_tokens_are_deduplicated() {
  // `constants::languages()` has known duplicate codes pointing at the same
  // token (e.g. "burmese"/"myanmar" both -> "my"); this pins that the
  // probe -> dedup step in `from_folder` actually collapses them, matching
  // Swift's `Set<Int>` semantics (`Models.swift:1219-1223`).
  let t = tiny();
  let ids = t.all_language_tokens();
  let mut sorted = ids.to_vec();
  sorted.sort_unstable();
  sorted.dedup();
  assert_eq!(
    sorted.len(),
    ids.len(),
    "all_language_tokens must not contain duplicate ids"
  );

  let my_id = t.token_to_id("<|my|>").unwrap();
  assert_eq!(ids.iter().filter(|&&id| id == my_id).count(), 1);
  assert_eq!(t.language_for_token(my_id), Some("my"));
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn language_for_token_returns_none_for_non_language_id() {
  let t = tiny();
  assert_eq!(t.language_for_token(t.special_tokens().end_token()), None);
  let content_ids = t.encode("hello").unwrap();
  assert_eq!(t.language_for_token(content_ids[0]), None);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn decode_skip_special_strips_control_tokens_but_keeps_timestamps() {
  let t = tiny();
  let s = t.special_tokens();
  let content = t.encode(" hi").unwrap();
  let mut ids = vec![s.start_of_transcript_token()];
  ids.extend(&content);
  ids.push(s.end_token());

  let kept = t.decode(&ids, false).unwrap();
  assert!(kept.contains("<|startoftranscript|>"));
  assert!(kept.contains("<|endoftext|>"));

  let stripped = t.decode(&ids, true).unwrap();
  assert!(!stripped.contains("<|startoftranscript|>"));
  assert!(!stripped.contains("<|endoftext|>"));
  assert!(stripped.contains("hi"));

  // Timestamp tokens are not flagged `"special"` in the tokenizer.json
  // (verified against the fixture: every `<|0.00|>`..`<|30.00|>` entry has
  // `"special": false`), so `skip_special_tokens` leaves them in place —
  // only the control tokens above (`"special": true`) get stripped.
  let mut with_timestamp = vec![s.time_token_begin()];
  with_timestamp.extend(&content);
  let timestamp_stripped = t.decode(&with_timestamp, true).unwrap();
  assert!(timestamp_stripped.contains("<|0.00|>"));
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn split_to_word_tokens_empty_input_is_empty() {
  let t = tiny();
  assert_eq!(t.split_to_word_tokens(&[], "en").unwrap(), vec![]);
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn split_to_word_tokens_preserves_token_coverage_and_text() {
  // Structural invariants that must hold regardless of split strategy:
  // every input token is covered by exactly one word, in original order;
  // concatenating the words' text reconstructs the full decode; and no
  // word is ever left holding a dangling replacement character (the
  // subtle part of `split_tokens_on_unicode`: a BPE token that splits a
  // multi-byte character mid-sequence must never surface as its own word).
  let t = tiny();
  for (text, lang) in [(" The quick brown fox.", "en"), ("你好，世界！", "zh")] {
    let ids = t.encode(text).unwrap();
    let words = t.split_to_word_tokens(&ids, lang).unwrap();

    let covered: Vec<u32> = words
      .iter()
      .flat_map(|(_, toks)| toks.iter().copied())
      .collect();
    assert_eq!(covered, ids, "language {lang}");

    let joined: String = words.iter().map(|(w, _)| w.as_str()).collect();
    assert_eq!(joined, t.decode(&ids, false).unwrap(), "language {lang}");

    for (word, _) in &words {
      assert!(
        !word.contains('\u{FFFD}'),
        "word {word:?} for language {lang}"
      );
    }
  }
}
