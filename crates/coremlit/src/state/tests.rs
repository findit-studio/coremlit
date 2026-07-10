use super::*;

#[test]
fn state_is_send() {
  fn assert_send<T: Send>() {}
  assert_send::<State>();
}
