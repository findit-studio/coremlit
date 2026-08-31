use std::path::PathBuf;

use super::*;

/// The staged whisper-tiny tokenizer folder.
///
/// Its CONTENTS beyond `tokenizer.json` are a staging detail, not a property
/// of this crate, and no test may assert on them: a local
/// `hf download openai/whisper-tiny tokenizer.json` leaves the folder holding
/// that file alone, while the three-file `files` selector MODELS_LOCK hands CI
/// (`tokenizer.json tokenizer_config.json config.json`) lands the checkpoint's
/// own `tokenizer_config.json` beside it. Both stagings resolve the cleanup
/// flag to `true` -- the first through Swift's `or:` default, the second by
/// reading the `"clean_up_tokenization_spaces": true` every OpenAI Whisper
/// checkpoint ships -- so a test that needs one shape must CONSTRUCT it.
fn tiny_folder() -> PathBuf {
  let root = std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  );
  root.join("tokenizers/whisper-tiny")
}

fn tiny() -> WhisperTokenizer {
  WhisperTokenizer::from_folder(tiny_folder()).unwrap()
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
    TokenizerError::FileNotFound(searched) => {
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
// WordGrouping (coremlit issue #14; parity corrected in codex round 1)
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn swift_parity_matches_swifts_pinned_japanese_word_tokens() {
  // Swift's OWN test, ported verbatim: `testSplitToWordTokensJapanese`
  // (`Tests/WhisperKitTests/UnitTests.swift:1360-1375`), token vector and
  // both expectations copied unchanged. Its assertion message reads "Words
  // did not match expected output in Unicode split", and its expectations
  // ARE the Unicode-split groups -- because Swift Unicode-splits Japanese.
  //
  //   こんにちは、世界！これはテストですよね？
  //
  // This is the test the old `Phrase` variant could not have passed. It
  // forced the space splitter for every CJK language, which on spaceless
  // Japanese collapses the whole utterance into one blob -- while claiming,
  // by name and in its docs, to be byte-comparable with Swift. Swift's
  // `NLLanguageRecognizer` returns the BARE code "ja", which its own CJK
  // check matches, so Swift takes the Unicode arm here. Only Chinese
  // (`zh-Hans`/`zh-Hant`, regional) misses that check.
  let t = tiny();
  let token_ids: Vec<u32> = vec![
    50364, 38088, 1231, 24486, 171, 120, 223, 25212, 22985, 40498, 4767, 30346, 171, 120, 253,
    50257,
  ];

  let expected_words = vec![
    "<|0.00|>",
    "こんにちは",
    "、",
    "世界",
    "！",
    "これは",
    "テ",
    "スト",
    "です",
    "よね",
    "？",
    "<|endoftext|>",
  ];
  let expected_word_tokens: Vec<Vec<u32>> = vec![
    vec![50364],
    vec![38088],
    vec![1231],
    vec![24486],
    vec![171, 120, 223],
    vec![25212],
    vec![22985],
    vec![40498],
    vec![4767],
    vec![30346],
    vec![171, 120, 253],
    vec![50257],
  ];

  let split = t
    .split_to_word_tokens(&token_ids, "ja", WordGrouping::SwiftParity)
    .unwrap();
  let words: Vec<&str> = split.iter().map(|(word, _)| word.as_str()).collect();
  let word_tokens: Vec<Vec<u32>> = split.iter().map(|(_, ids)| ids.clone()).collect();

  assert_eq!(words, expected_words, "Words did not match Swift's output.");
  assert_eq!(
    word_tokens, expected_word_tokens,
    "Word tokens did not match Swift's output."
  );
  assert_eq!(words.len(), 12, "Swift pins twelve groups");

  // The default grouping agrees with Swift here too -- for Japanese there is
  // nothing to trade off, because Swift is already fine-grained.
  assert_eq!(
    t.split_to_word_tokens(&token_ids, "ja", WordGrouping::FineGrained)
      .unwrap(),
    split,
    "ja is the SAME splitter under both modes: Swift Unicode-splits it, and \
     so does this port's default"
  );

  // The units are BPE-token-shaped, NOT one-per-Unicode-scalar: "こんにちは"
  // is five scalars in a single group, because it is a single BPE token.
  // (`FineGrained`'s doc used to claim one word per scalar; this is the
  // counter-example, straight out of Swift's own fixture.)
  assert_eq!("こんにちは".chars().count(), 5);
  assert_eq!(split[1].1.len(), 1, "one token, five scalars, one group");
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_grouping_splits_chinese_and_only_chinese() {
  // The whole of the divergence, in one test: `zh` is the ONLY language
  // (with `yue`) where the two groupings disagree, because it is the only one
  // whose `NLLanguage` raw value is regional (`zh-Hans`/`zh-Hant`) and so
  // misses Swift's bare-code CJK check.
  //
  // A ZH utterance with no spaces to split on -- the shape behind coremlit
  // issue #11's divergence (Rust's 85 fine-grained words against Swift's 24
  // blobs on the real ZH clip), in miniature.
  let t = tiny();
  let zh = t.encode("我今天很高兴见到你").unwrap();

  // OPT-IN -- the #11-pinned behavior: the Unicode splitter carves the
  // utterance into its Unicode-complete units. Those units are
  // BPE-token-shaped, not one-per-character ("今天" is a single token); the
  // guarantee is that they are FINE-GRAINED, never one-per-scalar.
  let fine = t
    .split_to_word_tokens(&zh, "zh", WordGrouping::FineGrained)
    .unwrap();
  let fine_texts: Vec<&str> = fine.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(
    fine_texts,
    vec!["我", "今天", "很", "高", "兴", "见", "到", "你"]
  );
  assert_eq!(
    crate::audio::whisper::options::DecodingOptions::new().word_grouping(),
    WordGrouping::SwiftParity,
    "and swift-parity is what a caller gets without asking -- #41; FineGrained is the opt-in"
  );

  // DEFAULT -- the space splitter finds no space anywhere in Chinese, so the
  // whole utterance collapses into a single blob with one start/end time:
  // Swift's `zh-Hant`-fallthrough grouping, reproduced deliberately rather
  // than stumbled into.
  let swift = t
    .split_to_word_tokens(&zh, "zh", WordGrouping::SwiftParity)
    .unwrap();
  let swift_texts: Vec<&str> = swift.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(swift_texts, vec!["我今天很高兴见到你"]);

  // MUTATION EVIDENCE: identical tokens, identical language code -- only the
  // grouping differs, and it alone moves 8 words to 1.
  assert!(
    fine.len() > swift.len(),
    "fine-grained must out-split Swift's grouping on Chinese: \
     {fine_texts:?} vs {swift_texts:?}"
  );
  // Neither mode loses text; they only disagree on where the boundaries are.
  assert_eq!(fine_texts.concat(), swift_texts.concat());

  // Cantonese rides with Chinese: `NLLanguage` has no Cantonese case, so
  // Swift's recognizer answers `zh-Hans`/`zh-Hant` for it too.
  assert_eq!(
    t.split_to_word_tokens(&zh, "yue", WordGrouping::SwiftParity)
      .unwrap(),
    swift
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn swift_parity_unicode_splits_every_cjk_language_except_chinese() {
  // The table on `WordGrouping`, executed. `ja`/`th`/`lo`/`my` have BARE
  // `NLLanguage` raw values, so Swift's own check matches them and Swift
  // Unicode-splits them -- meaning the two groupings must be IDENTICAL
  // there. Only `zh`/`yue` may differ.
  //
  // This is the assertion the old `Phrase` variant failed by construction:
  // it forced spaces for all six.
  let t = tiny();
  let ja = t.encode("こんにちは世界").unwrap();

  for language in ["ja", "th", "lo", "my"] {
    assert_eq!(
      t.split_to_word_tokens(&ja, language, WordGrouping::SwiftParity)
        .unwrap(),
      t.split_to_word_tokens(&ja, language, WordGrouping::FineGrained)
        .unwrap(),
      "Swift Unicode-splits `{language}`, so the two groupings must agree"
    );
  }

  for language in ["zh", "yue"] {
    assert_ne!(
      t.split_to_word_tokens(&ja, language, WordGrouping::SwiftParity)
        .unwrap(),
      t.split_to_word_tokens(&ja, language, WordGrouping::FineGrained)
        .unwrap(),
      "`{language}` is the accident: Swift space-splits it and this port does not"
    );
  }
}

// ---------------------------------------------------------------------
// clean_up_tokenization_spaces (coremlit issue #59)
// ---------------------------------------------------------------------

/// Swift's ten rules written out again, unguarded and in Swift's order — the
/// literal transcription of `Tokenizer.swift:439-448`. `clean_up_tokenization`
/// must equal this for every input; it only adds a short-circuit.
fn swift_clean_up_reference(text: &str) -> String {
  text
    .replace(" .", ".")
    .replace(" ?", "?")
    .replace(" !", "!")
    .replace(" ,", ",")
    .replace(" ' ", "'")
    .replace(" n't", "n't")
    .replace(" 'm", "'m")
    .replace(" 's", "'s")
    .replace(" 've", "'ve")
    .replace(" 're", "'re")
}

#[test]
fn clean_up_applies_swifts_ten_rules_and_only_those() {
  // One case per rule, in Swift's order (`Tokenizer.swift:439-448`), each
  // paired with a near-miss that must NOT be touched. The near-misses are the
  // point: this layer's failure mode is doing MORE than Swift, which would
  // trade issue #59's divergence for a new one.
  for (input, expected) in [
    (" a .", " a."),
    (" a ?", " a?"),
    (" a !", " a!"),
    (" a ,", " a,"),
    ("it ' s", "it's"),
    ("do n't", "don't"),
    ("I 'm", "I'm"),
    ("it 's", "it's"),
    ("we 've", "we've"),
    ("we 're", "we're"),
    // Near-misses: punctuation Swift does not list, and apostrophe forms
    // Swift does not list. HuggingFace does not list them either.
    (" a ;", " a ;"),
    (" a :", " a :"),
    (" a -", " a -"),
    (" a \"", " a \""),
    (" a …", " a …"),
    ("we 'll", "we 'll"),
    ("we 'd", "we 'd"),
    // Swift's `WordPieceDecoder` cleanup has a `" do not"` -> `" don't"`
    // rule (`Decoder.swift:107`). Whisper's decoder is `ByteLevel`, so that
    // rule is not in force here and this must pass through unchanged.
    ("I do not care", "I do not care"),
    // No space before the punctuation: nothing to clean.
    ("a. b? c! d,", "a. b? c! d,"),
    // Interior/leading whitespace that is not a plain space is untouched:
    // every Swift pattern starts with U+0020 specifically.
    ("a\t.", "a\t."),
    ("a\n.", "a\n."),
    ("a\u{a0}.", "a\u{a0}."),
    ("", ""),
    (" ", " "),
  ] {
    assert_eq!(
      clean_up_tokenization(input.to_owned()),
      expected,
      "input {input:?}"
    );
  }
}

#[test]
fn clean_up_is_swifts_chain_including_its_order() {
  // Swift chains the ten, so each rule sees the previous one's output and a
  // deletion can expose a later match. These are the cases that distinguish
  // "chain" from "ten independent passes over the original".
  //
  // The order witness, found by replaying every adjacent transposition of
  // Swift's ten over all short strings drawn from their own alphabet: `" ,"`
  // and `" ' "` are the one adjacent pair that does not commute. Swift runs
  // `" ,"` first, so it takes the comma's space and leaves `" ',"`; running
  // `" ' "` first would take both spaces and leave `"',"`. If these ten are
  // ever reordered, this is the assertion that notices.
  assert_eq!(clean_up_tokenization(" ' ,".to_owned()), " ',");
  // A token-spaced possessive still collapses to one apostrophe.
  assert_eq!(clean_up_tokenization(" ' s".to_owned()), "'s");
  // A match CREATED by an earlier rule, which only a chain can reach: the
  // original has no `" 's"` anywhere (its apostrophe and `s` are separated),
  // but `" ' "` collapsing to `"'"` puts one there for `" 's"` to take.
  assert_eq!(clean_up_tokenization("a  ' s".to_owned()), "a's");
  // Two independent rules, each firing once on its own region.
  assert_eq!(clean_up_tokenization("a . ,".to_owned()), "a.,");
  // Repeated matches of the same rule: each deletes only the space BEFORE
  // its punctuation, so the separating spaces after them survive.
  assert_eq!(clean_up_tokenization("a . b . c .".to_owned()), "a. b. c.");
  // Two spaces: `replace` is non-overlapping and left-to-right, so only the
  // pair adjacent to the `.` matches, leaving one space behind.
  assert_eq!(clean_up_tokenization("a  .".to_owned()), "a .");

  // And the whole table agrees with the unguarded transcription of Swift.
  for text in [
    " ...",
    " ' ",
    " n't",
    "a . ,",
    " ' s",
    "a  .",
    " . ? ! , 'm 's 've 're n't",
    "we ' re",
    " no trigger here",
    "こんにちは、世界！",
    "¿Qué ? ¡Sí !",
    "a\u{FFFD} .",
  ] {
    assert_eq!(
      clean_up_tokenization(text.to_owned()),
      swift_clean_up_reference(text),
      "guarded and unguarded must agree on {text:?}"
    );
  }
}

#[test]
fn clean_up_short_circuit_never_changes_the_answer() {
  // The short-circuit's soundness claim, exercised: over every string built
  // from an alphabet that contains each rule's first two bytes plus a few
  // decoys, the guarded implementation equals the unguarded transcription.
  // If the guard ever skipped a string a rule would have matched, or a
  // replacement ever manufactured a newly-matchable pair, this fails.
  let alphabet = [
    " ", ".", "?", "!", ",", "'", "n", "t", "m", "s", "v", "r", "e", "a",
  ];
  let mut checked = 0usize;
  for a in alphabet {
    for b in alphabet {
      for c in alphabet {
        for d in alphabet {
          let text = format!("{a}{b}{c}{d}");
          assert_eq!(
            clean_up_tokenization(text.clone()),
            swift_clean_up_reference(&text),
            "guard diverged on {text:?}"
          );
          checked += 1;
        }
      }
    }
  }
  assert_eq!(checked, alphabet.len().pow(4));
}

#[test]
fn config_boolean_coerces_like_swift() {
  // `Config.Data.boolean()` (`Config.swift:152-170`) as a table. Every row is
  // the answer the pinned oracle's own `Config` gives for that JSON scalar --
  // read back from it, not derived from what this port happens to return.
  for (json, expected) in [
    // `.boolean`: itself.
    ("true", Some(true)),
    ("false", Some(false)),
    // `.integer`: `val == 1`. `0`, `2` and `-1` are all a definite `false`,
    // not "no value" -- which is the whole reason a `0` disables the cleanup.
    ("1", Some(true)),
    ("0", Some(false)),
    ("2", Some(false)),
    ("-1", Some(false)),
    // An integer spelled with a fraction or an exponent is still `.integer`:
    // `Config`'s decoder tries `Int` before `Float` (`Config.swift:657-666`).
    ("1.0", Some(true)),
    ("1e0", Some(true)),
    ("0.0", Some(false)),
    ("-0.0", Some(false)),
    ("1.5e1", Some(false)), // = 15
    // `Int.max` and `Int.min` are integers, and are not 1.
    ("9223372036854775807", Some(false)),
    ("-9223372036854775808", Some(false)),
    // A real fraction, or a value past `Int64`, stays `.floating` -- and
    // `boolean()` has no `.floating` case, so it coerces to nothing.
    ("0.5", None),
    ("1e19", None),
    ("9223372036854775808", None), // 2^63, one past `Int.max`
    // The spellings at the edges of `Int`'s range, where this port and the
    // oracle do not agree everywhere, are tabulated in
    // `config_boolean_matches_swift_except_where_the_f64_lost_the_literal` instead --
    // every row of THIS table is a row the two answer alike.
    // `.string`: lowercased (`Config.swift:160`), then two fixed sets.
    (r#""true""#, Some(true)),
    (r#""t""#, Some(true)),
    (r#""1""#, Some(true)),
    (r#""false""#, Some(false)),
    (r#""f""#, Some(false)),
    (r#""0""#, Some(false)),
    (r#""True""#, Some(true)),
    (r#""TRUE""#, Some(true)),
    (r#""FALSE""#, Some(false)),
    (r#""T""#, Some(true)),
    // Every other string is nothing -- Swift matches the whole string and
    // does not trim it first.
    (r#""yes""#, None),
    (r#""maybe""#, None),
    (r#""""#, None),
    (r#"" true ""#, None),
    // `boolean()` has no case for these three either.
    ("null", None),
    ("[]", None),
    ("{}", None),
  ] {
    assert_eq!(
      config_boolean(&serde_json::from_str(json).unwrap()),
      expected,
      "value {json}"
    );
  }
}

#[test]
fn config_boolean_matches_swift_except_where_the_f64_lost_the_literal() {
  // `serde_json` keeps a number's spelling only while it fits `i64`/`u64`. A
  // fraction or an exponent leaves an `f64`, while the oracle's decoder reads
  // the digits, so a literal carrying more precision than an `f64` holds can
  // land differently on the two sides. This table carries BOTH columns -- what
  // the pinned oracle answers and what this port answers -- so the residue is
  // visible rather than asserted away.
  //
  // The oracle column was measured the same way the other tables were --
  // compiling its `Config.swift` and `BinaryDistinct.swift` against
  // `JSONDecoder` and reading back `Config.boolean()` -- not predicted from
  // this port. `9223372036854775296` and `9223372036854775807` are the two
  // magnitudes the oracle actually flips at near `Int`'s edge, and
  // `1844674407370955161` the one it flips at for a nonzero fraction; all
  // three were found by sweeping magnitudes, not by modelling the decoder.
  //
  // A row where the two columns differ is not automatically a gate that moves:
  // `or: true` can absorb the difference. The rows where the RESOLVED flag
  // really does differ are enumerated separately below, so that set stays
  // small, explicit and reviewable.
  const GATE_MOVES: &[&str] = &[
    "-9223372036854775807.0",
    "-9.223372036854775807e18",
    "-9223372036854775296.0",
    "1844674407370955162.5",
  ];

  for (json, swift, port) in [
    // Bare integers inside `i64`: `serde_json` holds these exactly, so the
    // `as_i64` arm answers and the two agree by construction.
    ("-9223372036854775807", Some(false), Some(false)),
    ("-9223372036854775808", Some(false), Some(false)), // `Int.min`
    // Bare integers past `i64` reach the float arm with `|f64| >= 2^63`. The
    // oracle's own `Int` decode fails on them too, so both decline.
    ("-9223372036854775809", None, None),
    ("-9223372036854776832", None, None), // -2^63 - 1024, still the `-2^63` f64
    ("-9223372036854776833", None, None), // one step further, a distinct f64
    // Fraction and exponent spellings landing on the `-2^63` f64 whose exact
    // magnitude is `2^63` or more. The oracle declines these as well, and this
    // is the larger half of that f64's cell.
    ("-9223372036854775808.0", None, None),
    ("-9223372036854775809.0", None, None),
    ("-9.223372036854775808e18", None, None),
    // THE RESIDUE. Same f64, exact magnitude below `2^63`, so the oracle
    // decodes `.integer` and DISABLES the cleanup while this port declines and
    // leaves the caller's `or: true` in place. The band runs from
    // `-(2^63 - 1)` to `-(2^63 - 512)`; both ends are pinned here.
    ("-9223372036854775807.0", Some(false), None),
    ("-9.223372036854775807e18", Some(false), None),
    ("-9223372036854775296.0", Some(false), None),
    // One integer past that band the value rounds to a different f64, which is
    // decisive again -- so the divergence really is bounded, not a cliff.
    ("-9223372036854775295.0", Some(false), Some(false)),
    ("-9223372036854774784.0", Some(false), Some(false)),
    // The positive edge has no such band: the oracle's cutoff is exactly the
    // first magnitude that rounds up to the f64 `2^63`, which is where the
    // port's exclusive upper bound already puts it.
    ("9223372036854775295.0", Some(false), Some(false)),
    ("9223372036854775296.0", None, None),
    ("9223372036854775807", Some(false), Some(false)), // `Int.max`, bare
    ("9223372036854775807.0", None, None),             // ...but not spelled `.0`
    ("9223372036854775808.0", None, None),
    // The SECOND divergence class, and it runs the opposite way. Past
    // `1844674407370955161` a nonzero fractional part stops decoding as
    // `.integer` in the oracle, while both spellings below still round to a
    // whole `f64` and are accepted here. The two rows straddle that flip.
    ("1844674407370955161.5", Some(false), Some(false)),
    ("1844674407370955162.5", None, Some(false)),
    // Nearby fractions that do agree, so the class above is pinned as a
    // boundary rather than as "fractions are unreliable": a fraction the `f64`
    // keeps is declined by both, and one it rounds away is taken by both.
    ("2251799813685248.5", None, None),
    ("9007199254740993.5", Some(false), Some(false)),
    ("4503599627370496.0001", Some(false), Some(false)),
    // A THIRD class, and the reason "they differ" and "the gate moves" have to
    // be asked separately. At 16 nines the `f64` has rounded all the way up to
    // `1.0` while the oracle still sees a fraction, so the two disagree -- yet
    // `Some(true)` and `or: true` are the same flag, and nothing moves. 15
    // nines is the control the `f64` still keeps apart from 1, and 19 is where
    // the oracle's own decode has rounded up too.
    ("0.999999999999999", None, None),
    ("0.9999999999999999", None, Some(true)),
    ("0.9999999999999999999", Some(true), Some(true)),
  ] {
    assert_eq!(
      config_boolean(&serde_json::from_str(json).unwrap()),
      port,
      "port's answer for {json}"
    );
    assert_eq!(
      swift.unwrap_or(true) != port.unwrap_or(true),
      GATE_MOVES.contains(&json),
      "whether the RESOLVED cleanup flag differs from Swift's for {json} \
       (swift={swift:?}, port={port:?})"
    );
  }
}

#[test]
fn clean_up_flag_follows_tokenizer_config_like_swift() {
  // Swift: `tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: true)`
  // (`Tokenizer.swift:407`). The member name is camelCase, and `Config`'s
  // dynamic-member subscript (`Config.swift:593-599`) tries it verbatim before
  // falling back to its `uncamelCase` form (`:601-621`), which for this name
  // is exactly `clean_up_tokenization_spaces`. `or: true` then applies only
  // where the chosen value coerces to nothing.
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join("tokenizer_config.json");

  // No file at all: this crate's folders need only `tokenizer.json`.
  assert!(clean_up_tokenization_spaces_from(dir.path()));

  for (body, expected) in [
    // The snake_case spelling every OpenAI checkpoint ships.
    (r#"{"clean_up_tokenization_spaces": true}"#, true),
    (r#"{"clean_up_tokenization_spaces": false}"#, false),
    // The camelCase spelling Swift actually looks for FIRST.
    (r#"{"cleanUpTokenizationSpaces": true}"#, true),
    (r#"{"cleanUpTokenizationSpaces": false}"#, false),
    // The value is coerced, not required to be a JSON boolean: `0` and
    // `"false"` DISABLE the cleanup in Swift. Pinning those two as `true` here
    // was this port's bug, so they are the regression rows.
    (r#"{"clean_up_tokenization_spaces": 0}"#, false),
    (r#"{"clean_up_tokenization_spaces": "false"}"#, false),
    (r#"{"cleanUpTokenizationSpaces": "0"}"#, false),
    (r#"{"clean_up_tokenization_spaces": 1}"#, true),
    (r#"{"clean_up_tokenization_spaces": "TRUE"}"#, true),
    // Present but uncoercible -> `or: true`, which is the default and not a
    // `false` in disguise.
    (r#"{"clean_up_tokenization_spaces": null}"#, true),
    (r#"{"clean_up_tokenization_spaces": "maybe"}"#, true),
    // Key absent -> `or: true`.
    (r#"{"model_max_length": 448}"#, true),
    ("{}", true),
    // Precedence: camelCase wins outright when both spellings are present...
    (
      r#"{"cleanUpTokenizationSpaces": false, "clean_up_tokenization_spaces": true}"#,
      false,
    ),
    (
      r#"{"cleanUpTokenizationSpaces": true, "clean_up_tokenization_spaces": false}"#,
      true,
    ),
    // ...and it wins on PRESENCE, not on coercibility: Swift's `??` chains the
    // two dictionary lookups, not their booleans, so an uncoercible camelCase
    // value falls through to `or: true` and never to the snake_case key.
    (
      r#"{"cleanUpTokenizationSpaces": "yes", "clean_up_tokenization_spaces": false}"#,
      true,
    ),
    (
      r#"{"cleanUpTokenizationSpaces": null, "clean_up_tokenization_spaces": false}"#,
      true,
    ),
    // Unparseable, and valid JSON that is not an object: same default.
    ("{ not json", true),
    ("[]", true),
    ("3", true),
    // The `Int`-range edge, at the call site that actually gates the cleanup.
    // The first three match the oracle; the fourth is the documented residue,
    // where Swift resolves `false` and this resolves `true` because the `.0`
    // spelling did not survive `serde_json`. See
    // `config_boolean_matches_swift_except_where_the_f64_lost_the_literal`.
    (
      r#"{"clean_up_tokenization_spaces": -9223372036854775808}"#,
      false,
    ),
    (
      r#"{"clean_up_tokenization_spaces": -9223372036854775808.0}"#,
      true,
    ),
    (
      r#"{"clean_up_tokenization_spaces": -9223372036854775809}"#,
      true,
    ),
    (
      r#"{"clean_up_tokenization_spaces": -9223372036854775807.0}"#,
      true,
    ),
  ] {
    std::fs::write(&path, body).unwrap();
    assert_eq!(
      clean_up_tokenization_spaces_from(dir.path()),
      expected,
      "config {body}"
    );
  }
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn decode_cleans_the_leading_space_off_the_ellipsis_token() {
  // coremlit issue #59's whole case, pinned at the token that produces it.
  // Token 1097 is byte-level BPE's `Ġ...`, i.e. a space plus an ellipsis of
  // three ASCII periods. The `tokenizers` backend decodes it literally; Swift
  // then strips the space via the `" ."` -> `"."` rule, and so must this.
  let t = tiny();
  assert_eq!(t.id_to_token(1097).as_deref(), Some("Ġ..."));

  // The backend still returns the uncleaned string -- this layer is what
  // changes the answer, not a different tokenizer configuration.
  assert_eq!(t.tokenizer.decode(&[1097], false).unwrap(), " ...");
  assert!(t.clean_up_tokenization_spaces);
  assert_eq!(t.decode(&[1097], false).unwrap(), "...");

  // And the cleanup that produces that answer is reached by Swift's `or:`
  // DEFAULT, not by a configured flag -- worth saying out loud, because a
  // folder carrying no `tokenizer_config.json` at all is a shape this crate
  // supports (`from_folder` requires only `tokenizer.json`) and the one where
  // the ellipsis fix could silently stop applying.
  //
  // That folder is CONSTRUCTED, not looked for. Asserting the staged folder
  // has no `tokenizer_config.json` is an assertion about whoever staged it,
  // not about this crate -- see `tiny_folder` -- and it is what made this test
  // pass on a dev box and fail on CI for a month.
  let bare = tempfile::tempdir().unwrap();
  std::fs::copy(
    tiny_folder().join("tokenizer.json"),
    bare.path().join("tokenizer.json"),
  )
  .unwrap();
  assert!(!bare.path().join("tokenizer_config.json").exists());
  let bare_tokenizer = WhisperTokenizer::from_folder(bare.path()).unwrap();
  assert!(bare_tokenizer.clean_up_tokenization_spaces);
  assert_eq!(bare_tokenizer.decode(&[1097], false).unwrap(), "...");

  // The staged folder then resolves `true` under EITHER shape, which is the
  // only thing about it this test is entitled to depend on.
  assert!(clean_up_tokenization_spaces_from(&tiny_folder()));
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn cleaned_ellipsis_joins_the_previous_word_but_a_period_still_starts_one() {
  // The consequence of the test above, in `splitTokensOnSpaces` terms
  // (`Models.swift:1255-1276`) -- this is the one grouping decision that
  // moved LuYu's word edit distance.
  let t = tiny();
  let hi = t.encode(" hi").unwrap();

  // `" ..."` -> `"..."`: not space-prefixed, and three scalars rather than
  // one, so neither the `withSpace` nor the `punctuation` arm fires and it
  // is appended to the preceding word.
  let mut with_ellipsis = hi.clone();
  with_ellipsis.push(1097);
  let words = t
    .split_to_word_tokens(&with_ellipsis, "en", WordGrouping::SwiftParity)
    .unwrap();
  let texts: Vec<&str> = words.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(texts, vec![" hi..."]);
  assert_eq!(words.last().unwrap().1, vec![*hi.last().unwrap(), 1097]);

  // Contrast: a lone `" ."` is cleaned to `"."` too, but one punctuation
  // scalar still starts its own word, so grouping there is unchanged.
  let period = t.token_to_id("Ġ.").unwrap();
  let mut with_period = hi.clone();
  with_period.push(period);
  assert_eq!(t.decode(&[period], false).unwrap(), ".");
  let words = t
    .split_to_word_tokens(&with_period, "en", WordGrouping::SwiftParity)
    .unwrap();
  let texts: Vec<&str> = words.iter().map(|(w, _)| w.as_str()).collect();
  assert_eq!(texts, vec![" hi", "."]);

  // Both groupings agree: this is a decode-level change, not a mode-level one.
  assert_eq!(
    t.split_to_word_tokens(&with_ellipsis, "en", WordGrouping::FineGrained)
      .unwrap(),
    t.split_to_word_tokens(&with_ellipsis, "en", WordGrouping::SwiftParity)
      .unwrap()
  );
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn cleanup_leaves_special_and_timestamp_token_decodes_alone() {
  // Blast-radius check: the cleanup runs on EVERY decode, including the ones
  // that carry `<|...|>` markers, and those must survive it intact -- segment
  // text and the word splitter both parse them.
  let t = tiny();
  let s = t.special_tokens();
  let ids = [
    s.start_of_transcript_token(),
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
    s.no_timestamps_token(),
    s.end_token(),
  ];
  let decoded = t.decode(&ids, false).unwrap();
  assert_eq!(
    decoded,
    "<|startoftranscript|><|en|><|transcribe|><|0.00|><|notimestamps|><|endoftext|>"
  );
  assert_eq!(decoded, t.tokenizer.decode(&ids, false).unwrap());
}

#[test]
#[ignore = "requires local tokenizer (WHISPERKIT_TEST_MODELS)"]
fn word_grouping_is_inert_for_whitespace_delimited_languages() {
  // A space-delimited language already takes the space splitter under both
  // modes, so the two are identical there. This is the structural reason the
  // English/Spanish goldens cannot move no matter what this knob is set to.
  let t = tiny();
  let ids = t.encode(" Hello world").unwrap();

  let fine = t
    .split_to_word_tokens(&ids, "en", WordGrouping::FineGrained)
    .unwrap();
  let swift = t
    .split_to_word_tokens(&ids, "en", WordGrouping::SwiftParity)
    .unwrap();

  assert_eq!(fine, swift, "non-CJK: both modes split on spaces");
  assert_eq!(
    fine.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>(),
    vec![" Hello", " world"]
  );
}
