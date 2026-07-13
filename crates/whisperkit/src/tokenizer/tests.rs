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
  let words = t
    .split_to_word_tokens(&ids, "en", WordGrouping::FineGrained)
    .unwrap();
  let texts: Vec<&str> = words.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(texts, vec![" Hello", " world"]);
  // unicode-split path: every CJK char its own word
  let zh = t.encode("你好世界").unwrap();
  let words = t
    .split_to_word_tokens(&zh, "zh", WordGrouping::FineGrained)
    .unwrap();
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
  assert_eq!(
    t.split_to_word_tokens(&[], "en", WordGrouping::FineGrained)
      .unwrap(),
    vec![]
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn cjk_languages_split_into_fine_grained_words() {
  // Pins a deliberate, chosen DIVERGENCE from Swift (coremlit issue #9,
  // "Chinese word timestamp grouping needs a product policy"): Swift's
  // `NLLanguageRecognizer` reports `zh-Hant` for Traditional Chinese text,
  // but Swift's own CJK allowlist in `splitToWordTokens` is exactly
  // `zh`/`ja`/`th`/`lo`/`my`/`yue` (`Models.swift:1293-1306`) --
  // `zh-Hant` misses that list and falls through to the space-based
  // splitter, which (Chinese has no spaces) groups a whole utterance into
  // one coarse phrase blob instead of timing each character. This crate
  // never reproduces that gap: the language code driving the split always
  // comes from the decoder's own `<|lang|>` prompt token (see
  // [`WhisperTokenizer::split_to_word_tokens`]'s doc), which is a bare
  // base code (`zh`, never `zh-Hant`) by construction, so it always lands
  // on the CJK arm below. The sample string and its expected
  // per-character split are copied verbatim from issue #9's own
  // Rust/Swift comparison run (its "Representative output" section). If a
  // future change ever "fixes" this by routing decoder language codes
  // through Swift's raw, un-normalized recognizer output, this test
  // catches the regression back to phrase-blob grouping.
  let t = tiny();
  let text = "你上學也不也不說普通話";
  let expected_words = vec![
    "你", "上", "學", "也", "不", "也", "不", "說", "普", "通", "話",
  ];
  assert_eq!(expected_words.len(), text.chars().count());

  for lang in ["zh", "ja", "yue"] {
    let ids = t.encode(text).unwrap();
    let words = t
      .split_to_word_tokens(&ids, lang, WordGrouping::FineGrained)
      .unwrap();
    let texts: Vec<&str> = words.iter().map(|(w, _)| w.as_str()).collect();
    assert_eq!(texts, expected_words, "language {lang}");
    assert_eq!(
      words.len(),
      text.chars().count(),
      "language {lang}: word count must equal char count"
    );
  }

  // Contrast: a non-CJK language code routes the exact same tokens to the
  // space-based splitter instead -- since the sample has no spaces, the
  // whole utterance collapses into a single coarse "word". This is the
  // failure mode the CJK arm above exists to avoid.
  let ids = t.encode(text).unwrap();
  let en_words = t
    .split_to_word_tokens(&ids, "en", WordGrouping::FineGrained)
    .unwrap();
  assert_eq!(
    en_words.len(),
    1,
    "non-CJK routing must not split per character"
  );
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
    let words = t
      .split_to_word_tokens(&ids, lang, WordGrouping::FineGrained)
      .unwrap();

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

// ---------------------------------------------------------------------
// WordGrouping (coremlit issue #14)
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_grouping_routes_cjk_fine_grained_by_default_and_phrase_on_opt_in() {
  // A ZH utterance with no spaces to split on -- the shape behind coremlit
  // issue #11's divergence (Rust's 85 fine-grained words against Swift's 24
  // phrase blobs on the real ZH clip), in miniature.
  let t = tiny();
  let zh = t.encode("我今天很高兴见到你").unwrap();

  // DEFAULT -- the #11-pinned behavior, unchanged: the Unicode splitter
  // carves the utterance into its Unicode-complete units. (Those units are
  // BPE-token-shaped, not strictly one-per-character -- "今天" is a single
  // token -- which is exactly what `split_tokens_on_unicode` produces and
  // what #11 pinned; the point is that they are FINE-GRAINED.)
  let fine = t
    .split_to_word_tokens(&zh, "zh", WordGrouping::FineGrained)
    .unwrap();
  let fine_texts: Vec<&str> = fine.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(
    fine_texts,
    vec!["我", "今天", "很", "高", "兴", "见", "到", "你"]
  );
  assert_eq!(
    crate::options::DecodingOptions::new().word_grouping(),
    WordGrouping::FineGrained,
    "and fine-grained is what a caller gets without asking"
  );

  // OPT-IN -- the space splitter finds no space anywhere in CJK, so the
  // whole utterance collapses into a single phrase blob with one start/end
  // time: Swift's `zh-Hant`-fallthrough grouping, reproduced deliberately
  // rather than stumbled into.
  let phrase = t
    .split_to_word_tokens(&zh, "zh", WordGrouping::Phrase)
    .unwrap();
  let phrase_texts: Vec<&str> = phrase.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(phrase_texts, vec!["我今天很高兴见到你"]);

  // MUTATION EVIDENCE: identical tokens, identical language code -- only the
  // grouping differs, and it alone moves 8 words to 1.
  assert!(
    fine.len() > phrase.len(),
    "fine-grained must out-split phrase on CJK: {fine_texts:?} vs {phrase_texts:?}"
  );
  // Neither mode loses text; they only disagree on where the boundaries are.
  assert_eq!(fine_texts.concat(), phrase_texts.concat());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_grouping_is_inert_for_whitespace_delimited_languages() {
  // The grouping re-routes the CJK arm ONLY: a space-delimited language
  // already takes the space splitter under both modes, so the two are
  // identical there. This is the structural reason the English/Spanish
  // goldens cannot move no matter what this knob is set to.
  let t = tiny();
  let ids = t.encode(" Hello world").unwrap();

  let fine = t
    .split_to_word_tokens(&ids, "en", WordGrouping::FineGrained)
    .unwrap();
  let phrase = t
    .split_to_word_tokens(&ids, "en", WordGrouping::Phrase)
    .unwrap();

  assert_eq!(fine, phrase, "non-CJK: both modes split on spaces");
  assert_eq!(
    fine.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>(),
    vec![" Hello", " world"]
  );
}
