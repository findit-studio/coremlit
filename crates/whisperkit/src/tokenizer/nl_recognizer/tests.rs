use super::*;

// No `WHISPERKIT_TEST_MODELS` gating needed here, unlike the tokenizer's
// own fixture-driven tests: `NLLanguageRecognizer` is an Apple system
// framework always present on macOS, not a downloaded model asset.

#[test]
fn redetects_english() {
  assert_eq!(
    redetect_language("This is a plain English sentence for language detection.").as_deref(),
    Some("en")
  );
}

#[test]
fn redetects_traditional_chinese_and_normalizes_to_zh() {
  // issue #9's own repro string: `NLLanguageRecognizer`'s raw output for
  // this text is `zh-Hant`, not `zh` (verified empirically against this
  // crate version) -- collapsing that down to `zh` is the entire point of
  // this wrapper existing instead of callers using the class directly.
  let detected = redetect_language("你上學也不也不說普通話全是東北話").unwrap();
  assert_eq!(detected, "zh");
}

#[test]
fn empty_input_is_undetermined() {
  assert_eq!(redetect_language(""), None);
}

#[test]
fn normalize_bcp47_strips_script_and_region_subtags() {
  assert_eq!(normalize_bcp47("zh-Hant"), "zh");
  assert_eq!(normalize_bcp47("zh-Hans"), "zh");
  assert_eq!(normalize_bcp47("zh-CN"), "zh");
  assert_eq!(normalize_bcp47("pt-BR"), "pt");
  assert_eq!(normalize_bcp47("en"), "en");
}

#[test]
fn normalize_bcp47_maps_cmn_to_zh() {
  assert_eq!(normalize_bcp47("cmn"), "zh");
  assert_eq!(normalize_bcp47("cmn-Hant"), "zh");
}
