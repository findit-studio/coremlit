use std::os::unix::fs::symlink;

use super::*;

/// Codex round 3, finding 5: a broken link on a path's way is named, not
/// taken for an absent file. A file looked up under a directory link that
/// now leads nowhere is refused as that link's `BrokenLink`, by the resolve
/// door and by the read door alike; a path with nothing at it and no broken
/// link on the way is still absent.
#[test]
fn a_dangling_ancestor_is_refused_as_a_broken_link() {
  let root = tempfile::tempdir().unwrap();
  let target = root.path().join("bundle");
  fs::create_dir(&target).unwrap();
  fs::write(target.join("coremldata.bin"), b"stamped").unwrap();
  let link = root
    .path()
    .join("model")
    .join("EmbeddingGemma2Audio.mlmodelc");
  fs::create_dir(root.path().join("model")).unwrap();
  symlink(&target, &link).unwrap();
  let file = link.join("coremldata.bin");
  assert!(
    resolve(&file)
      .expect("resolves")
      .is_some_and(|found| found.is_file())
  );
  fs::remove_dir_all(&target).unwrap();
  match resolve(&file) {
    Err(Error::Traversal(refused)) => {
      assert_eq!(refused.defect(), TraversalDefect::BrokenLink);
      assert_eq!(refused.path(), link, "the link on the way is named");
    }
    other => panic!("expected the dangling directory link refused, got {other:?}"),
  }
  match open(&file, None) {
    Err(Refusal::Named(Error::Traversal(refused))) => assert_eq!(refused.path(), link),
    other => panic!("expected the read door to name the link, got {other:?}"),
  }
  assert!(
    resolve(&root.path().join("model").join("absent.json"))
      .expect("nothing broken on the way")
      .is_none(),
    "a path with nothing at it is absent"
  );
}
