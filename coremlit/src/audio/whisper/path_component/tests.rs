use super::{
  PathComponentDefect::{CurrentDirectory, Empty, Nul, ParentDirectory, Separator},
  *,
};

#[test]
fn one_plain_component_is_accepted_and_every_other_spelling_names_its_defect() {
  // ACCEPTED: everything that joins to exactly one entry of the given
  // directory, however unusual it looks.
  for accepted in [
    "talk",
    "MelSpectrogram",
    "a.b",
    "a b",
    "文件",
    "...",
    ".hidden",
    "..a",
    "a..",
    "-",
    "a-b_c.d",
    "with:colon",
    "with*star",
    "a\tb",
    "a\nb",
  ] {
    assert_eq!(
      single_path_component(accepted),
      Ok(accepted),
      "refused a plain component {accepted:?}"
    );
  }

  // REFUSED: every spelling that changes WHICH directory the join resolves
  // into, each reported as the defect it actually trips.
  for (refused, defect) in [
    ("", Empty),
    ("/", Separator),
    ("/abs", Separator),
    ("a/b", Separator),
    ("a/", Separator),
    ("./a", Separator),
    ("../escape", Separator),
    ("..//..", Separator),
    ("sub/dir", Separator),
    ("\\", Separator),
    ("back\\slash", Separator),
    ("..\\escape", Separator),
    (".", CurrentDirectory),
    ("..", ParentDirectory),
    ("\0", Nul),
    ("nul\0byte", Nul),
  ] {
    assert_eq!(
      single_path_component(refused),
      Err(defect),
      "wrong verdict for {refused:?}"
    );
  }
}

#[test]
fn the_rule_is_only_about_which_directory_a_join_resolves_into() {
  // No length cap: a name too long for the filesystem is the filesystem's
  // `ENAMETOOLONG` to report, not this rule's.
  let long = "x".repeat(64 * 1024);
  assert_eq!(single_path_component(&long), Ok(long.as_str()));

  // No reserved-name list: these are ordinary names where this crate runs, and
  // none of them redirects a join.
  for name in ["CON", "NUL", "aux", "COM1", "$MFT", "lost+found"] {
    assert_eq!(single_path_component(name), Ok(name));
  }

  // No Unicode normalisation: the composed and decomposed spellings of `café`
  // are different strings and stay different names, which is exactly what
  // `std::fs` will do with them.
  let nfc = "caf\u{e9}";
  let nfd = "cafe\u{301}";
  assert_ne!(nfc, nfd);
  assert_eq!(single_path_component(nfc), Ok(nfc));
  assert_eq!(single_path_component(nfd), Ok(nfd));
}

#[test]
fn every_defect_renders_a_distinct_reason() {
  let defects = [Empty, Separator, Nul, CurrentDirectory, ParentDirectory];
  for (i, a) in defects.iter().enumerate() {
    assert!(!a.reason().is_empty());
    for b in &defects[i + 1..] {
      assert_ne!(a.reason(), b.reason(), "{a:?} and {b:?} render the same");
    }
  }
}
