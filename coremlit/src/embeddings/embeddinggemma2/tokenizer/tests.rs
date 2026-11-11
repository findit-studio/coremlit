use super::*;

/// The typed face of `split_special_tokens`: `false` recognizes, `true`
/// splits, and the spellings are pinned.
#[test]
fn the_split_behaviours_map_the_boolean_option() {
  assert_eq!(
    SplitSpecialTokens::from_split(false),
    SplitSpecialTokens::Recognize
  );
  assert_eq!(
    SplitSpecialTokens::from_split(true),
    SplitSpecialTokens::Split
  );
  for split in [false, true] {
    assert_eq!(SplitSpecialTokens::from_split(split).is_split(), split);
  }
  assert_eq!(SplitSpecialTokens::Recognize.to_string(), "recognize");
  assert_eq!(SplitSpecialTokens::Split.to_string(), "split");
}
