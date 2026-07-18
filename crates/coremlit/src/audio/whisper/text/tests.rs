use super::*;

// ---------------------------------------------------------------------
// compression_ratio_of_tokens / compression_ratio_of_text
// ---------------------------------------------------------------------

// Recover the codec's compressed byte length from the public ratio
// (`ratio == raw_len / compressed_len`), so the GOLDEN tests below can pin
// Apple libcompression's ACTUAL output length — the true oracle — with a
// clear integer failure message. `.round()` absorbs f32 division noise
// (each captured length round-trips exactly; a wrong codec lands on a
// different integer). Raw length is the i32-LE byte count (4 bytes/token)
// for tokens and the UTF-8 byte count for text.
fn compressed_len_of_tokens(tokens: &[u32]) -> usize {
  (tokens.len() as f32 * 4.0 / compression_ratio_of_tokens(tokens)).round() as usize
}
fn compressed_len_of_text(text: &str) -> usize {
  (text.len() as f32 / compression_ratio_of_text(text)).round() as usize
}

#[test]
fn compression_ratio_detects_repetition() {
  let unique: Vec<u32> = (0..200).collect();
  let repeated = vec![42u32; 200];
  assert!(compression_ratio_of_tokens(&repeated) > compression_ratio_of_tokens(&unique));
  assert!(compression_ratio_of_tokens(&repeated) > 2.4); // crosses the fallback threshold
  // Exact Apple oracle for the repeated case (see the golden test): 800
  // raw i32-LE bytes compress to 13. flate2/miniz_oxide gave 27 — still
  // > 2.4, which is precisely why the `> 2.4` check alone cannot guard the
  // codec choice, but this length pin can.
  assert_eq!(compressed_len_of_tokens(&repeated), 13);
}

#[test]
fn compression_ratio_of_tokens_matches_apple_libcompression_golden() {
  // GOLDEN / DIFFERENTIAL (coremlit issue #9). The pinned lengths are
  // Apple libcompression's ACTUAL `.zlib` output — the exact
  // `NSData.compressed(using: .zlib)` bytes Swift WhisperKit's
  // `TextUtilities.compressionRatio` produces — captured with an
  // independent oracle (NOT this crate's own encoder) and verified to
  // inflate as raw DEFLATE / RFC 1951. Because they are
  // not self-referential, they FAIL if the codec regresses to
  // flate2/miniz_oxide (RFC-1950 zlib-wrapped, +6 bytes of wrapper, and a
  // weaker body); each case notes what flate2 would emit instead.
  //
  // Byte-level anchor: `[1212,318,257,1332,13]` x2 →
  // Apple emits 24 bytes beginning `db c3 c2 c0` (raw DEFLATE); flate2
  // would emit 30. This case simultaneously pins the i32-LE token encoding
  // (the input) and the Apple codec (the length).
  let doc_example: Vec<u32> = [1212, 318, 257, 1332, 13].repeat(2);
  assert_eq!(doc_example.len(), 10, "40 raw i32-LE bytes");
  assert_eq!(compressed_len_of_tokens(&doc_example), 24); // flate2: 30

  // The same 5-token phrase x8 → Apple plateaus at 24 bytes (its harder
  // compression on repetition); flate2 would emit 37. 160 raw bytes.
  let phrase_x8: Vec<u32> = [1212, 318, 257, 1332, 13].repeat(8);
  assert_eq!(compressed_len_of_tokens(&phrase_x8), 24); // flate2: 37
}

#[test]
fn compression_ratio_of_tokens_empty_is_zero_matching_swift() {
  // PARITY (coremlit issue #9). Swift's TOKENS overload
  // (Utilities/TextUtilities.swift:14-28) has NO empty guard: it compresses
  // an empty `Data()`, and Apple's libcompression turns a zero-length
  // buffer into 2 bytes (it does NOT throw — proven by the issue-9 objc2
  // probe), so Swift returns `0 / 2 == 0.0`, not infinity. This value is
  // pinned to Swift's proven result, not a re-run of our own encoder.
  //
  // Decision-level consequence: in `needs_fallback`, `0.0 > threshold`
  // (default 2.4) is false, so an empty word-token window does NOT force a
  // repetition fallback — matching Swift. INFINITY (the pre-fix value)
  // would have flipped that to a wrongful fallback. The end-to-end check
  // lives in
  // `result::tests::empty_word_tokens_do_not_trigger_compression_fallback`.
  assert_eq!(compression_ratio_of_tokens(&[]), 0.0);
}

#[test]
fn compression_ratio_of_text_empty_is_infinite() {
  // TextUtilities.swift:34-36 — explicit empty-string guard, checked
  // before any UTF-8 encode/compress is even attempted.
  assert_eq!(compression_ratio_of_text(""), f32::INFINITY);
}

#[test]
fn compression_ratio_of_text_matches_apple_libcompression_golden() {
  // GOLDEN (see the tokens golden above for provenance): Apple `.zlib`
  // UTF-8 output lengths, independently captured and raw-DEFLATE-verified;
  // they fail under a flate2 regression (flate2 lengths, noted, are
  // larger).
  let the_x20 = "the the the the the the the the the the the the the the the the the the the the";
  assert_eq!(the_x20.len(), 79);
  assert_eq!(compressed_len_of_text(the_x20), 9); // flate2: 28

  let fox_x5 = "the quick brown fox jumps over the lazy dog ".repeat(5);
  assert_eq!(fox_x5.len(), 220);
  assert_eq!(compressed_len_of_text(&fox_x5), 48); // flate2: 58
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
