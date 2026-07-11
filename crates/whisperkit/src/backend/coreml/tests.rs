use super::*;

#[test]
fn names_match_recorded_ground_truth() {
  // Pins Task 1's introspected names as compile-visible constants.
  assert_eq!(names::LOGITS, "logits");
  assert_eq!(names::KEY_UPDATES, "key_cache_updates");
  assert_eq!(names::VALUE_UPDATES, "value_cache_updates");
  assert_eq!(names::ALIGNMENT, "alignment_heads_weights");
  assert_eq!(names::KV_UPDATE_MASK, "kv_cache_update_mask");
}
