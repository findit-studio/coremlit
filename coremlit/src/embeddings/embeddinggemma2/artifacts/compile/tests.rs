use std::os::unix::fs::PermissionsExt;

use super::{
  super::{
    super::{
      audio::AudioPrecision,
      error::StampField,
      map::{Access, ReadOnlyMap},
      table::TOKEN_TABLE_BYTES,
    },
    ArtifactLabel, Part,
  },
  *,
};

const MANIFEST: &str = env!("CARGO_MANIFEST_DIR");

fn copy_dir(from: &Path, to: &Path) {
  fs::create_dir_all(to).unwrap();
  for entry in fs::read_dir(from).unwrap() {
    let entry = entry.unwrap();
    let target = to.join(entry.file_name());
    if entry.file_type().unwrap().is_dir() {
      copy_dir(&entry.path(), &target);
    } else {
      fs::copy(entry.path(), target).unwrap();
    }
  }
}

/// A model directory over committed fixtures: the two-function package as
/// the text package, the mini `tokenizer.json`, the committed `config.json`,
/// and a sparse full-size token table.
fn fixture_dir() -> tempfile::TempDir {
  let dir = tempfile::tempdir().unwrap();
  copy_dir(
    &Path::new(MANIFEST).join("tests/fixtures/two_functions.mlpackage"),
    &dir.path().join("EmbeddingGemma2Text.mlpackage"),
  );
  fs::copy(
    Path::new(MANIFEST).join("tests/embeddinggemma2/fixtures/mini_tokenizer.json"),
    dir.path().join("tokenizer.json"),
  )
  .unwrap();
  fs::copy(
    Path::new(MANIFEST).join("tests/embeddinggemma2/fixtures/config.json"),
    dir.path().join("config.json"),
  )
  .unwrap();
  File::create(dir.path().join("embeddings.bf16"))
    .unwrap()
    .set_len(TOKEN_TABLE_BYTES)
    .unwrap();
  dir
}

/// A caller's compiled bundle, plainly named, under `dir`.
fn supply(dir: &Path, name: &str) {
  let compiled =
    Model::compile(Path::new(MANIFEST).join("tests/fixtures/identity.mlmodel")).unwrap();
  copy_dir(&compiled, &dir.join(name));
}

/// Every file under `dir`, by relative path, with its SHA-256.
fn snapshot(dir: &Path) -> BTreeMap<String, String> {
  files_under(dir, dir)
    .unwrap()
    .into_iter()
    .map(|rel| {
      let sha = hex(&sha256_file(&dir.join(&rel)).unwrap());
      (rel, sha)
    })
    .collect()
}

/// The mini tokenizer with one byte changed and its length kept.
fn variant_tokenizer(dir: &Path) -> Vec<u8> {
  let text = fs::read_to_string(dir.join("tokenizer.json")).unwrap();
  let variant = text.replacen("\"version\": \"1.0\"", "\"version\": \"1.1\"", 1);
  assert_ne!(variant, text);
  variant.into_bytes()
}

fn stamp(dir: &Path) -> Stamp {
  Stamp::read(dir).expect("a stamp")
}

/// CA1: packages and the tokenizer compile to content-addressed names the
/// stamp names; a second call keeps everything and creates nothing.
#[test]
fn compile_names_content_addressed_artifacts_and_keeps_them() {
  let dir = fixture_dir();
  let first = compile_artifacts(dir.path()).expect("compiles");
  let text = stamp(dir.path()).bundles["text"].clone();
  let lines = text
    .digests
    .iter()
    .map(|(rel, sha)| (rel.as_str(), sha.as_str()));
  let expected_bundle = format!(
    "EmbeddingGemma2Text.{}.mlmodelc",
    &hex(&manifest_sha256(lines))[..12]
  );
  let source: [u8; 32] =
    Sha256::digest(fs::read(dir.path().join("tokenizer.json")).unwrap()).into();
  let expected_image = image_name(&source);
  assert_eq!(text.name, expected_bundle);
  assert_eq!(
    first.compiled(),
    [expected_bundle.clone(), expected_image.clone()]
  );
  assert!(dir.path().join(&expected_bundle).is_dir() && dir.path().join(&expected_image).is_file());
  let before = snapshot(dir.path());
  let second = compile_artifacts(dir.path()).expect("current");
  assert!(second.compiled().is_empty(), "{second:?}");
  assert_eq!(
    snapshot(dir.path()),
    before,
    "a current directory gains no file"
  );
}

/// CA1: two calls racing on one directory compile it once.
#[test]
fn two_concurrent_calls_compile_once() {
  let dir = fixture_dir();
  let path = dir.path().to_path_buf();
  let racers: Vec<_> = (0..2)
    .map(|_| {
      let path = path.clone();
      std::thread::spawn(move || compile_artifacts(&path).expect("compiles"))
    })
    .collect();
  let compiled: usize = racers
    .into_iter()
    .map(|r| r.join().unwrap().compiled().len())
    .sum();
  assert_eq!(
    compiled, 2,
    "the bundle and the image, once between the two calls"
  );
}

/// CA1: nothing is moved, overwritten or deleted: across a regeneration every
/// file stays as it was, the new image joins them, and a supplied bundle
/// stays byte for byte.
#[test]
fn the_file_set_only_grows_and_a_supplied_bundle_is_untouched() {
  let dir = fixture_dir();
  supply(dir.path(), "EmbeddingGemma2Audio.mlmodelc");
  compile_artifacts(dir.path()).expect("compiles");
  assert_eq!(
    stamp(dir.path()).bundles["audio_fp16"].origin,
    Origin::Supplied
  );
  let before = snapshot(dir.path());
  fs::write(
    dir.path().join("tokenizer.json"),
    variant_tokenizer(dir.path()),
  )
  .unwrap();
  compile_artifacts(dir.path()).expect("recompiles");
  let after = snapshot(dir.path());
  for (rel, sha) in &before {
    if rel != "tokenizer.json" && rel != STAMP_NAME {
      assert_eq!(after.get(rel), Some(sha), "{rel} moved or changed");
    }
  }
  assert_eq!(
    after.len(),
    before.len() + 1,
    "the new image joins; the old one stays"
  );
}

/// CA1, mandatory (the shared maps' SAFETY point 1): a running map of the
/// tokenizer image keeps reading its own bytes across a concurrent
/// regeneration, which writes the new image under a new name, and across the
/// `prune` that then unlinks the old one.
#[test]
fn a_running_map_survives_a_concurrent_regeneration() {
  let dir = fixture_dir();
  compile_artifacts(dir.path()).expect("compiles");
  let old_name = stamp(dir.path()).tokenizer_image.unwrap().name;
  let map = ReadOnlyMap::open(
    Artifact::TokenizerImage,
    &dir.path().join(&old_name),
    Access::Default,
  )
  .unwrap();
  let mapped_before: [u8; 32] = Sha256::digest(map.bytes()).into();
  fs::write(
    dir.path().join("tokenizer.json"),
    variant_tokenizer(dir.path()),
  )
  .unwrap();
  compile_artifacts(dir.path()).expect("regenerates");
  let new_name = stamp(dir.path()).tokenizer_image.unwrap().name;
  assert_ne!(new_name, old_name, "a new image takes a new name");
  let mapped_after: [u8; 32] = Sha256::digest(map.bytes()).into();
  assert_eq!(
    mapped_after, mapped_before,
    "the running map reads the inode it mapped, unchanged"
  );
  assert!(
    dir.path().join(&old_name).is_file(),
    "the old image stays until prune"
  );
  let pruned = prune(dir.path()).expect("prunes");
  assert_eq!(pruned.removed(), [dir.path().join(&old_name)]);
  let mapped_after_prune: [u8; 32] = Sha256::digest(map.bytes()).into();
  assert_eq!(
    mapped_after_prune, mapped_before,
    "an unlinked inode under the map stays alive and unchanged"
  );
}

/// CA1 (G5): a time-preserving copy over a source is seen through its ctime
/// and inode, so its digest is not kept stale.
#[test]
fn a_time_preserving_copy_is_detected() {
  let dir = fixture_dir();
  compile_artifacts(dir.path()).expect("compiles");
  let target = dir.path().join("tokenizer.json");
  let variant_path = dir.path().join("variant.json.in");
  fs::write(&variant_path, variant_tokenizer(dir.path())).unwrap();
  let mtime = fs::metadata(&target).unwrap().modified().unwrap();
  File::options()
    .write(true)
    .open(&variant_path)
    .unwrap()
    .set_modified(mtime)
    .unwrap();
  let copied = std::process::Command::new("cp")
    .arg("-p")
    .arg(&variant_path)
    .arg(&target)
    .status()
    .unwrap();
  assert!(copied.success());
  fs::remove_file(&variant_path).unwrap();
  assert_eq!(
    fs::metadata(&target).unwrap().modified().unwrap(),
    mtime,
    "the copy kept the time"
  );
  compile_artifacts(dir.path()).expect("recompiles");
  let recorded = stamp(dir.path()).files["tokenizer.json"].sha256.clone();
  assert_eq!(
    recorded,
    hex(&sha256_file(&target).unwrap()),
    "the copy's digest is recorded"
  );
}

/// CA1 (G11): a stamped directory that is read-only passes, and no lock is
/// taken, because the currency pass comes first.
#[test]
fn a_read_only_current_directory_passes_without_the_lock() {
  let dir = fixture_dir();
  compile_artifacts(dir.path()).expect("compiles");
  fs::remove_file(dir.path().join(LOCK_NAME)).unwrap();
  fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();
  let result = compile_artifacts(dir.path());
  fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
  assert!(
    result
      .expect("a read-only current directory passes")
      .compiled()
      .is_empty()
  );
  assert!(!dir.path().join(LOCK_NAME).exists(), "no lock was created");
}

/// CA1: `prune` removes only derived names the stamp does not name: never a
/// supplied bundle (stamped, or placed since the last compile), a source, the
/// stamp or the lock.
#[test]
fn prune_removes_only_unnamed_derived_files() {
  let dir = fixture_dir();
  supply(dir.path(), "EmbeddingGemma2Audio.mlmodelc");
  compile_artifacts(dir.path()).expect("compiles");
  let old_image = stamp(dir.path()).tokenizer_image.unwrap().name;
  fs::write(
    dir.path().join("tokenizer.json"),
    variant_tokenizer(dir.path()),
  )
  .unwrap();
  compile_artifacts(dir.path()).expect("recompiles");
  let stale = dir
    .path()
    .join("EmbeddingGemma2Text.0123456789ab.mlmodelc.tmp-1-2");
  fs::write(&stale, b"half written").unwrap();
  supply(dir.path(), "EmbeddingGemma2Vision.mlmodelc");
  let report = prune(dir.path()).expect("prunes");
  let mut expected = vec![dir.path().join(&old_image), stale];
  expected.sort();
  assert_eq!(report.removed(), expected);
  for kept in [
    "EmbeddingGemma2Audio.mlmodelc",
    "EmbeddingGemma2Vision.mlmodelc",
    "EmbeddingGemma2Text.mlpackage",
    "tokenizer.json",
    STAMP_NAME,
    LOCK_NAME,
  ] {
    assert!(dir.path().join(kept).exists(), "{kept} survives prune");
  }
}

/// CA1: a package beside a caller's plainly named bundle of the same artifact
/// is refused, naming both; neither is chosen.
#[test]
fn a_package_beside_a_supplied_bundle_is_ambiguous() {
  let dir = fixture_dir();
  supply(dir.path(), "EmbeddingGemma2Text.mlmodelc");
  match compile_artifacts(dir.path()) {
    Err(Error::AmbiguousArtifact(ambiguous)) => {
      assert_eq!(ambiguous.artifact(), Artifact::TextBundle);
      assert_eq!(ambiguous.candidates().len(), 2);
    }
    other => panic!("expected AmbiguousArtifact, got {other:?}"),
  }
}

/// The digest bytes of a stamp's hex digest.
fn bytes_of(sha: &str) -> [u8; 32] {
  let mut out = [0u8; 32];
  for (i, pair) in sha.as_bytes().chunks_exact(2).enumerate() {
    out[i] = u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap();
  }
  out
}

/// The load side and L14 (a): a stamped part yields its label (a digest label
/// here: the fixture is not FluidInference's tree); a matching pin passes, a
/// mismatching one is refused naming the part and both labels, an absent one
/// passes.
#[test]
fn a_stamped_part_yields_its_label_and_honours_a_pin() {
  let dir = fixture_dir();
  compile_artifacts(dir.path()).expect("compiles");
  let recorded = stamp(dir.path());
  let view = recorded
    .part(dir.path(), Part::Text, AudioPrecision::Fp16)
    .expect("the text part");
  let mut lines: Vec<(String, [u8; 32])> = recorded.bundles["text"]
    .digests
    .iter()
    .map(|(rel, sha)| (rel.clone(), bytes_of(sha)))
    .collect();
  for rel in ["embeddings.bf16", "tokenizer.json", "config.json"] {
    lines.push((rel.to_owned(), bytes_of(&recorded.files[rel].sha256)));
  }
  let expected = ArtifactLabel::from_digests(lines.iter().map(|(rel, sha)| (rel.as_str(), *sha)));
  assert_eq!(view.label(), expected);
  assert!(!view.label().is_revision());
  assert_eq!(view.sentinels().len(), 9);
  assert_eq!(
    view.bundle(),
    dir.path().join(&recorded.bundles["text"].name)
  );
  assert!(view.bundle().is_dir());
  assert_eq!(
    view.image().map(|i| i.name.as_str()),
    recorded.tokenizer_image.as_ref().map(|i| i.name.as_str())
  );
  view.expect(None).expect("an absent pin passes");
  view
    .expect(Some(view.label()))
    .expect("a matching pin passes");
  match view.expect(Some(ArtifactLabel::FLUIDINFERENCE)) {
    Err(Error::ArtifactIdentity(identity)) => {
      assert_eq!(identity.part(), Part::Text);
      assert_eq!(identity.expected(), ArtifactLabel::FLUIDINFERENCE);
      assert_eq!(identity.actual(), view.label());
    }
    other => panic!("expected ArtifactIdentity, got {other:?}"),
  }
}

/// The load side: an artifact the stamp does not list is `Unstamped`, a
/// stamped file whose size moved is `StaleStamp`, a stamp of another format is
/// refused, and a directory with no stamp is `Unstamped`.
#[test]
fn a_missing_resized_or_foreign_stamp_entry_is_refused_by_name() {
  let dir = fixture_dir();
  compile_artifacts(dir.path()).expect("compiles");
  let recorded = stamp(dir.path());
  assert!(matches!(
    recorded.part(dir.path(), Part::Vision, AudioPrecision::Fp16),
    Err(Error::Unstamped(u)) if u.artifact() == Artifact::VisionBundle
  ));
  let mut config = fs::read(dir.path().join("config.json")).unwrap();
  config.push(b'\n');
  fs::write(dir.path().join("config.json"), &config).unwrap();
  assert!(matches!(
    recorded.part(dir.path(), Part::Text, AudioPrecision::Fp16),
    Err(Error::StaleStamp(s)) if s.artifact() == Artifact::Config && s.actual() == s.recorded() + 1
  ));
  let stamp_path = dir.path().join(STAMP_NAME);
  let text = fs::read_to_string(&stamp_path)
    .unwrap()
    .replacen("\"format\": 1", "\"format\": 2", 1);
  fs::write(&stamp_path, text).unwrap();
  assert!(
    matches!(Stamp::read(dir.path()), Err(Error::Stamp(d)) if d.field() == StampField::Format)
  );
  fs::remove_file(&stamp_path).unwrap();
  assert!(
    matches!(Stamp::read(dir.path()), Err(Error::Unstamped(u)) if u.artifact() == Artifact::Stamp)
  );
}
