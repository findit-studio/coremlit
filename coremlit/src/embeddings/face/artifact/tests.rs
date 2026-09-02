//! Gates for the artifact digest: what changes it, and what must not.

use std::{fs, path::Path};

use super::*;

/// Writes a minimal `.mlmodelc`-shaped tree: the file names a real compiled
/// bundle has, so a gate that renames or edits one is talking about the same
/// geometry the door will meet.
fn stage(root: &Path) {
  fs::create_dir_all(root.join("weights")).expect("create weights");
  fs::create_dir_all(root.join("analytics")).expect("create analytics");
  fs::write(root.join("coremldata.bin"), b"header bytes").expect("write coremldata");
  fs::write(root.join("model.mil"), b"program(1.3) {}").expect("write model.mil");
  fs::write(root.join("metadata.json"), b"[{}]").expect("write metadata");
  fs::write(root.join("weights/weight.bin"), b"0123456789abcdef").expect("write weights");
  fs::write(root.join("analytics/coremldata.bin"), b"analytics").expect("write analytics");
}

fn digest_of(root: &Path) -> ArtifactDigest {
  digest_artifact(root).expect("a staged bundle hashes")
}

#[test]
fn a_byte_identical_copy_of_a_bundle_has_the_same_digest() {
  // The property the whole design rests on: identity is of the BYTES, not of
  // the load. `&self` inference means fan-out is one embedder per worker over
  // the same artifact, and two workers — or two machines — that read equal
  // bytes have to name one space, or the digest would refuse exactly the
  // cross-worker comparisons those workers exist to make.
  let temp = tempfile::tempdir().expect("tempdir");
  let (left, right) = (
    temp.path().join("a.mlmodelc"),
    temp.path().join("b.mlmodelc"),
  );
  stage(&left);
  stage(&right);
  assert_eq!(
    digest_of(&left),
    digest_of(&right),
    "two byte-identical bundles at different paths must be one artifact"
  );
  // And the digest is a function, not a nonce: hashing twice agrees.
  assert_eq!(digest_of(&left), digest_of(&left));
}

#[test]
fn one_changed_weight_byte_changes_the_digest() {
  // The failure the digest exists to catch, at its smallest: a fine-tune, a
  // requantisation, or a different checkpoint entirely, all of which leave the
  // schema — width, feature names, preprocessing — exactly where it was.
  let temp = tempfile::tempdir().expect("tempdir");
  let (left, right) = (
    temp.path().join("a.mlmodelc"),
    temp.path().join("b.mlmodelc"),
  );
  stage(&left);
  stage(&right);
  fs::write(right.join("weights/weight.bin"), b"0123456789abcdeF").expect("rewrite weights");
  assert_ne!(
    digest_of(&left),
    digest_of(&right),
    "one byte of one weight file is a different artifact"
  );
}

#[test]
fn moving_a_files_bytes_to_another_name_changes_the_digest() {
  // The PATH is part of each entry, and it has to be: a bundle whose
  // `model.mil` and `metadata.json` have swapped contents is a different
  // bundle, and CoreML would read it differently — but the multiset of file
  // hashes is identical, so a digest over the hashes alone cannot see it.
  let temp = tempfile::tempdir().expect("tempdir");
  let (left, right) = (
    temp.path().join("a.mlmodelc"),
    temp.path().join("b.mlmodelc"),
  );
  stage(&left);
  stage(&right);
  let mil = fs::read(right.join("model.mil")).expect("read");
  let metadata = fs::read(right.join("metadata.json")).expect("read");
  fs::write(right.join("model.mil"), &metadata).expect("swap");
  fs::write(right.join("metadata.json"), &mil).expect("swap");
  assert_ne!(
    digest_of(&left),
    digest_of(&right),
    "two files with swapped contents are a different artifact, and only the path in each entry \
     can say so"
  );

  // The same point through a plain rename, which also changes nothing about
  // the set of file hashes.
  let renamed = temp.path().join("c.mlmodelc");
  stage(&renamed);
  fs::rename(renamed.join("model.mil"), renamed.join("model.mil.bak")).expect("rename");
  assert_ne!(digest_of(&left), digest_of(&renamed));
}

#[test]
fn a_ds_store_beside_the_weights_does_not_change_the_digest() {
  // `.DS_Store` is written into any directory a Finder window has been opened
  // on, and Spotlight and AppleDouble files arrive the same way. None of them
  // are the model. A digest that moved when one appeared would refuse a
  // caller's stored embeddings for a reason having nothing to do with the
  // weights — and it would do it on one machine and not another.
  let temp = tempfile::tempdir().expect("tempdir");
  let (left, right) = (
    temp.path().join("a.mlmodelc"),
    temp.path().join("b.mlmodelc"),
  );
  stage(&left);
  stage(&right);
  fs::write(right.join(".DS_Store"), b"finder junk").expect("write .DS_Store");
  fs::write(right.join("weights/.DS_Store"), b"more junk").expect("write nested .DS_Store");
  fs::create_dir_all(right.join(".hidden/deep")).expect("create dot-directory");
  fs::write(right.join(".hidden/deep/blob"), b"not the model").expect("write in dot-directory");
  assert_eq!(
    digest_of(&left),
    digest_of(&right),
    "dotfiles and dot-directories are excluded at every level"
  );
}

#[test]
fn a_symlinked_file_hashes_as_the_bytes_it_resolves_to() {
  // Symlinks are FOLLOWED, so a bundle assembled out of links to a shared
  // store hashes as the bytes it actually reads.
  let temp = tempfile::tempdir().expect("tempdir");
  let (left, right) = (
    temp.path().join("a.mlmodelc"),
    temp.path().join("b.mlmodelc"),
  );
  stage(&left);
  stage(&right);
  let elsewhere = temp.path().join("shared-weight.bin");
  fs::write(&elsewhere, b"0123456789abcdef").expect("write shared");
  fs::remove_file(right.join("weights/weight.bin")).expect("remove");
  std::os::unix::fs::symlink(&elsewhere, right.join("weights/weight.bin")).expect("symlink");
  assert_eq!(
    digest_of(&left),
    digest_of(&right),
    "a link to identical bytes is the same artifact"
  );

  // A link that cannot be followed is an ERROR, not a skip: a bundle with a
  // dangling entry is not a bundle whose bytes are known, and the load must
  // fail rather than stamp an identity for something it did not read.
  fs::remove_file(&elsewhere).expect("break the link");
  let error = digest_artifact(&right).expect_err("a dangling link has no bytes");
  assert!(
    matches!(&error, Error::ArtifactDigest(payload) if payload.path().ends_with("weight.bin")),
    "the failure must name the entry that could not be read, got {error:?}"
  );
}

#[test]
fn an_empty_directory_is_invisible_and_a_regular_file_root_is_allowed() {
  let temp = tempfile::tempdir().expect("tempdir");
  let (left, right) = (
    temp.path().join("a.mlmodelc"),
    temp.path().join("b.mlmodelc"),
  );
  stage(&left);
  stage(&right);
  fs::create_dir_all(right.join("empty/also-empty")).expect("create empty dirs");
  assert_eq!(
    digest_of(&left),
    digest_of(&right),
    "a directory contributes only through the files under it"
  );

  // `Model::load` takes any path CoreML accepts and a compiled bundle is a
  // directory in practice, but hashing what was actually loaded must not
  // depend on that.
  let file = temp.path().join("solitary.bin");
  fs::write(&file, b"just bytes").expect("write");
  let same = temp.path().join("also-solitary.bin");
  fs::write(&same, b"just bytes").expect("write");
  assert_eq!(digest_of(&file), digest_of(&same));
  let different = temp.path().join("other.bin");
  fs::write(&different, b"other bytes").expect("write");
  assert_ne!(digest_of(&file), digest_of(&different));
}

#[test]
fn a_missing_artifact_is_reported_by_path() {
  let temp = tempfile::tempdir().expect("tempdir");
  let absent = temp.path().join("not-there.mlmodelc");
  let error = digest_artifact(&absent).expect_err("nothing to hash");
  assert!(
    matches!(&error, Error::ArtifactDigest(payload)
      if payload.path() == absent && payload.source().kind() == std::io::ErrorKind::NotFound),
    "expected a NotFound naming the artifact, got {error:?}"
  );
  assert!(
    error
      .to_string()
      .contains("failed to hash the model artifact"),
    "the message must say what failed, got {error}"
  );
}

/// The unprefixed serialisation — `path ‖ file-hash` per entry, concatenated —
/// so a gate can show two entry lists collide under it before asserting that
/// `fold_entries` separates them.
fn unprefixed(entries: &[(Vec<u8>, [u8; 32])]) -> Vec<u8> {
  entries
    .iter()
    .flat_map(|(path, hash)| path.iter().chain(hash.iter()).copied())
    .collect()
}

#[test]
fn the_length_prefix_is_what_makes_the_encoding_injective() {
  // Two DIFFERENT entry lists whose unprefixed serialisations are equal BYTE
  // FOR BYTE. Without the length prefix these two artifacts have one digest,
  // and a vector from one would be scored against a vector from the other:
  //
  //   left  = [("x", Hx), ("y", Hy)]   →  "x" ‖ Hx ‖ "y" ‖ Hy
  //   right = [("x"‖Hx‖"y", Hy)]       →  "x" ‖ Hx ‖ "y" ‖ Hy
  //
  // Fed to `fold_entries` directly rather than staged on disk, and that is not
  // a shortcut: the right-hand list needs a file NAME holding the raw bytes of
  // a SHA-256, and APFS refuses any name that is not valid UTF-8 (this gate
  // was first written against `tempfile` and got `EILSEQ`). So the collision
  // is unreachable through this platform's filesystem while remaining a
  // property of the encoding — and the encoding is the thing this crate
  // defines and another implementation would have to match.
  let hash_x = [0xABu8; 32];
  let hash_y = [0xCDu8; 32];
  let left = vec![(b"x".to_vec(), hash_x), (b"y".to_vec(), hash_y)];
  let mut absorbed = b"x".to_vec();
  absorbed.extend_from_slice(&hash_x);
  absorbed.extend_from_slice(b"y");
  let right = vec![(absorbed, hash_y)];

  assert_eq!(
    unprefixed(&left),
    unprefixed(&right),
    "the two lists must collide without the prefix, or this gate proves nothing"
  );
  assert_ne!(
    fold_entries(left),
    fold_entries(right),
    "two different artifacts must not share one identity"
  );
}

#[test]
fn the_digest_does_not_depend_on_the_order_entries_are_discovered_in() {
  // `read_dir` order is the filesystem's business, not the artifact's. Two
  // machines, or one machine after a defragment, must agree.
  let entries = vec![
    (b"weights/weight.bin".to_vec(), [1u8; 32]),
    (b"coremldata.bin".to_vec(), [2u8; 32]),
    (b"analytics/coremldata.bin".to_vec(), [3u8; 32]),
    (b"model.mil".to_vec(), [4u8; 32]),
  ];
  let mut shuffled = entries.clone();
  shuffled.reverse();
  assert_ne!(
    entries, shuffled,
    "the two orders must differ, or this gate proves nothing"
  );
  assert_eq!(
    fold_entries(entries),
    fold_entries(shuffled),
    "the digest must be a function of the SET of entries, not of the walk order"
  );
}

/// Copies a directory tree, so a gate can compare a real bundle against a copy
/// of it at a different path.
fn copy_tree(from: &Path, to: &Path) {
  fs::create_dir_all(to).expect("mkdir");
  for entry in fs::read_dir(from).expect("read_dir") {
    let entry = entry.expect("entry");
    let (source, destination) = (entry.path(), to.join(entry.file_name()));
    if fs::metadata(&source).expect("metadata").is_dir() {
      copy_tree(&source, &destination);
    } else {
      fs::copy(&source, &destination).expect("copy");
    }
  }
}

#[test]
fn a_real_compiled_bundle_hashes_the_same_at_a_second_path() {
  // Every gate above walks a tree this file wrote. This one walks a REAL
  // compiled `.mlmodelc` — the vendored silero VAD model, committed and
  // therefore present in every `cargo test`: six files across two directory
  // levels, including a `weights/weight.bin` and a `LICENSE`. It is not a face
  // artifact (this crate stages none, see the `face` module doc) but it is the
  // only real bundle on disk, and the property under test — a bundle copied
  // elsewhere is the same artifact — is exactly what makes the digest usable
  // as an identity across workers and machines.
  let bundle = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  if !bundle.is_dir() {
    // `Models/` is outside the published package; the gate is about the walk,
    // not about the file's presence.
    return;
  }
  let temp = tempfile::tempdir().expect("tempdir");
  let copy = temp.path().join("elsewhere.mlmodelc");
  copy_tree(&bundle, &copy);
  assert_eq!(
    digest_of(&bundle),
    digest_of(&copy),
    "a bundle copied to another path, on another filesystem, is one artifact"
  );

  // And one byte of the real weights is a different artifact.
  fs::write(copy.join("weights/weight.bin"), b"not the weights").expect("overwrite");
  assert_ne!(digest_of(&bundle), digest_of(&copy));
}
