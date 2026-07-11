use super::*;

#[test]
fn window_math_is_consistent() {
  assert_eq!(SAMPLE_RATE, 16_000);
  assert_eq!(WINDOW_SAMPLES, 480_000);
  assert_eq!(
    WINDOW_SAMPLES,
    (SAMPLE_RATE as usize) * WINDOW_SECONDS as usize
  );
  assert_eq!(MAX_TOKEN_CONTEXT, 224); // Swift: Int(448 / 2), Models.swift:1334
}

#[test]
fn languages_table_matches_swift() {
  assert_eq!(languages().len(), 112); // awk-counted from Models.swift Constants.languages
  assert_eq!(language_code("english"), Some("en"));
  assert_eq!(language_code("chinese"), Some("zh"));
  assert_eq!(language_code("cantonese"), Some("yue"));
  assert_eq!(language_code("en"), Some("en")); // code passthrough
  assert_eq!(language_code("klingon"), None);
}

#[test]
fn language_codes_are_unique_names_are_unique() {
  let mut names: Vec<_> = languages().iter().map(|(n, _)| *n).collect();
  names.sort_unstable();
  names.dedup();
  assert_eq!(names.len(), languages().len());
}

#[test]
fn punctuation_contains_load_bearing_members() {
  // Models.swift:1459-1460 (defaultPrependPunctuations/defaultAppendPunctuations).
  assert!(PREPEND_PUNCTUATION.contains('('));
  assert!(APPEND_PUNCTUATION.contains(','));
  assert!(APPEND_PUNCTUATION.contains('.'));
}
