use super::*;

// ---------------------------------------------------------------------
// compression_ratio_of_tokens / compression_ratio_of_text
// ---------------------------------------------------------------------

#[test]
fn compression_ratio_detects_repetition() {
  let unique: Vec<u32> = (0..200).collect();
  let repeated = vec![42u32; 200];
  assert!(compression_ratio_of_tokens(&repeated) > compression_ratio_of_tokens(&unique));
  assert!(compression_ratio_of_tokens(&repeated) > 2.4); // crosses the fallback threshold
  assert_eq!(compression_ratio_of_tokens(&[]), f32::INFINITY);
}

#[test]
fn compression_ratio_uses_i32_le_bytes() {
  // Byte-format parity: ratio == raw_len / zlib_len computed over i32-LE bytes.
  let tokens = [1u32, 2, 3, 4];
  let bytes: Vec<u8> = tokens
    .iter()
    .flat_map(|t| (*t as i32).to_le_bytes())
    .collect();
  let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
  std::io::Write::write_all(&mut enc, &bytes).unwrap();
  let compressed = enc.finish().unwrap();
  let expected = bytes.len() as f32 / compressed.len() as f32;
  assert_eq!(compression_ratio_of_tokens(&tokens), expected);
}

#[test]
fn compression_ratio_of_text_empty_is_infinite() {
  // TextUtilities.swift:34-36 — explicit empty-string guard, checked
  // before any UTF-8 encode/compress is even attempted.
  assert_eq!(compression_ratio_of_text(""), f32::INFINITY);
}

#[test]
fn compression_ratio_of_text_matches_hand_computed_zlib_ratio() {
  // Same ratio formula as compression_ratio_of_tokens, different byte
  // source (UTF-8 vs. i32-LE): sanity-checks the formula against a
  // hand-rolled zlib pass over the string's own UTF-8 bytes, at the same
  // Compression::default() level.
  let text = "the quick brown fox jumps over the lazy dog ".repeat(5);
  let bytes = text.as_bytes();
  let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
  std::io::Write::write_all(&mut enc, bytes).unwrap();
  let compressed = enc.finish().unwrap();
  let expected = bytes.len() as f32 / compressed.len() as f32;
  assert_eq!(compression_ratio_of_text(&text), expected);
}

#[test]
fn compression_ratio_of_text_detects_repetition() {
  let unique = "the quick brown fox jumps over the lazy dog";
  let repeated = "the the the the the the the the the the the the the the the the the the the the";
  assert!(compression_ratio_of_text(repeated) > compression_ratio_of_text(unique));
}

// ---------------------------------------------------------------------
// normalized
// ---------------------------------------------------------------------

#[test]
fn normalized_matches_swift_semantics() {
  // Extensions+Public.swift:24-41. NOTE (source-corrected per this task's
  // own mandate): this task's brief asserted
  // `normalized("multi-word_test") == "multi word test"`. Running the
  // live Swift extension standalone (see task report) shows the actual
  // output is `"multi wordtest"` — `_` is Unicode general category `Pc`
  // (Connector Punctuation), a member of Foundation's
  // `CharacterSet.punctuationCharacters`, and step 3 of `normalized`
  // *deletes* punctuation rather than replacing it with a space. Only the
  // literal ASCII `-` becomes a space, via the separate, earlier, literal
  // (non-regex) `replacingOccurrences(of: "-", with: " ")` call. Source
  // wins; the assertion below reflects the verified Swift behavior, not
  // the brief's sketch.
  assert_eq!(normalized("Hello, World!"), "hello world");
  assert_eq!(normalized("multi-word_test"), "multi wordtest");
  assert_eq!(normalized("  a   b  "), "a b");
}

#[test]
fn normalized_deletes_underscores_fusing_the_surrounding_word() {
  assert_eq!(normalized("under_score"), "underscore");
  assert_eq!(normalized("a_b_c"), "abc");
}

#[test]
fn normalized_only_the_literal_ascii_hyphen_becomes_a_space() {
  // Other Unicode dashes (general category Pd, same as the ASCII hyphen)
  // are still punctuation, so they get DELETED by step 3, not spaced —
  // only the exact ASCII `-` is special-cased to a space, by step 2,
  // which runs first and is a literal (non-regex) string replace.
  assert_eq!(normalized("em\u{2014}dash"), "emdash"); // em dash
  assert_eq!(normalized("en\u{2013}dash"), "endash"); // en dash
  assert_eq!(normalized("a-b"), "a b");
}

#[test]
fn normalized_preserves_unicode_letters() {
  assert_eq!(normalized("Café Résumé"), "café résumé");
}

#[test]
fn normalized_collapses_multi_hyphen_runs_and_drops_other_punctuation() {
  assert_eq!(normalized("a--b"), "a b"); // both hyphens -> spaces, then collapsed to one
  assert_eq!(normalized("100%"), "100"); // '%' is Po (Other Punctuation), deleted
}

#[test]
fn normalized_empty_and_blank_inputs() {
  assert_eq!(normalized(""), "");
  assert_eq!(normalized("   "), "");
  assert_eq!(normalized("!!!"), "");
}

// ---------------------------------------------------------------------
// trim_special_token_chars
// ---------------------------------------------------------------------

#[test]
fn trims_special_token_wrapping() {
  assert_eq!(trim_special_token_chars("<|endoftext|>"), "endoftext");
  assert_eq!(trim_special_token_chars("plain"), "plain");
}

#[test]
fn trim_special_token_chars_is_a_character_class_trim_not_a_fixed_affix() {
  // Core/Models.swift:1332 `Constants.specialTokenCharacters =
  // CharacterSet(charactersIn: "<|>")`; Extensions+Public.swift:43-45
  // trims every leading/trailing member of {<,|,>}, repeatedly, from both
  // ends — not a fixed `"<|"` prefix / `"|>"` suffix strip. Verified
  // against the live Swift extension (see task report): a naive
  // strip_prefix("<|")/strip_suffix("|>") implementation would return
  // `"<<|x"` for the first case below (no literal `"<|"` prefix to
  // strip) instead of the correct `"x"`.
  assert_eq!(trim_special_token_chars("<<|x|>"), "x");
  assert_eq!(trim_special_token_chars("<|a|><|b|>"), "a|><|b");
}

#[test]
fn normalized_deletes_apostrophes_and_curly_quotes() {
  // Po/Pi/Pf punctuation must be deleted like any other P* class — pins the
  // classifier choice against a future mechanism swap.
  assert_eq!(normalized("don't"), "dont");
  assert_eq!(
    normalized("\u{2018}quoted\u{2019} \u{201C}text\u{201D}"),
    "quoted text"
  );
}

// ---------------------------------------------------------------------
// find_longest_common_prefix / find_longest_different_suffix
// ---------------------------------------------------------------------

use crate::result::WordTiming;

fn word(text: &str, start: f32, end: f32) -> WordTiming {
  // NOTE (source-corrected per this task's own mandate): the brief's
  // literal snippet called `.into()` on `text` here. Against
  // `WordTiming::new`'s generic `impl Into<String>` parameter that is
  // ambiguous (E0283 — `&str` implements `Into<T>` for several `T`, e.g.
  // `String`/`Box<str>`/`Cow<str>`, and nothing pins which one an `impl
  // Into<String>`-bounded type parameter should be). `&str` already
  // satisfies `impl Into<String>` directly, so passing `text` itself
  // (already `&str`-typed) needs no `.into()` call at all.
  WordTiming::new(text, vec![1], start, end, 0.9)
}

#[test]
fn common_prefix_compares_normalized_and_returns_current_elements() {
  // TranscriptionUtilities.swift:34-37 — comparison is over String.normalized
  // (case/punctuation-insensitive) and the RESULT elements come from the
  // second (newer) array.
  let previous = [
    word(" Hey", 0.0, 0.2),
    word(" you", 0.2, 0.4),
    word(" there", 0.4, 0.6),
  ];
  let current = [
    word(" hey,", 0.1, 0.3),
    word(" You", 0.3, 0.5),
    word(" friend", 0.5, 0.7),
  ];
  let prefix = find_longest_common_prefix(&previous, &current);
  assert_eq!(prefix.len(), 2);
  assert_eq!(prefix[0].word(), " hey,");
  assert!((prefix[0].start() - 0.1).abs() < 1e-6, "newer timings kept");
  // Length-asymmetric inputs stop at the shorter zip.
  assert_eq!(
    find_longest_common_prefix(&previous[..1], &current).len(),
    1
  );
  assert!(find_longest_common_prefix(&[], &current).is_empty());
}

#[test]
fn different_suffix_is_current_past_the_common_prefix() {
  // TranscriptionUtilities.swift:44-48
  let previous = [word(" Hey", 0.0, 0.2), word(" you", 0.2, 0.4)];
  let current = [
    word(" hey", 0.0, 0.2),
    word(" you", 0.2, 0.4),
    word(" friend", 0.4, 0.7),
  ];
  let suffix = find_longest_different_suffix(&previous, &current);
  assert_eq!(suffix.len(), 1);
  assert_eq!(suffix[0].word(), " friend");
  // No agreement at all -> the whole current array is the suffix.
  let disjoint = [word(" But", 0.0, 0.2)];
  assert_eq!(find_longest_different_suffix(&disjoint, &current).len(), 3);
}
