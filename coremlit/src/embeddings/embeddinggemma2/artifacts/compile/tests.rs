use std::{
  collections::BTreeSet,
  os::unix::fs::{PermissionsExt, symlink},
};

use super::{
  super::{
    super::{
      audio::AudioPrecision,
      error::{SourceField, StampField, TraversalDefect},
      map::{Access, ReadOnlyMap},
      table::TOKEN_TABLE_BYTES,
    },
    ArtifactLabel, Part,
    stamp::STAMP_CAP,
    tree::MAX_TREE_ENTRIES,
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

/// Fills `dir` with committed fixtures: the two-function package as the text
/// package, the mini `tokenizer.json`, the committed `config.json`, and a
/// sparse full-size token table.
fn fill_fixture(dir: &Path) {
  copy_dir(
    &Path::new(MANIFEST).join("tests/fixtures/two_functions.mlpackage"),
    &dir.join("EmbeddingGemma2Text.mlpackage"),
  );
  fs::copy(
    Path::new(MANIFEST).join("tests/embeddinggemma2/fixtures/mini_tokenizer.json"),
    dir.join("tokenizer.json"),
  )
  .unwrap();
  fs::copy(
    Path::new(MANIFEST).join("tests/embeddinggemma2/fixtures/config.json"),
    dir.join("config.json"),
  )
  .unwrap();
  File::create(dir.join("embeddings.bf16"))
    .unwrap()
    .set_len(TOKEN_TABLE_BYTES)
    .unwrap();
}

/// A model directory over committed fixtures.
fn fixture_dir() -> tempfile::TempDir {
  let dir = tempfile::tempdir().unwrap();
  fill_fixture(dir.path());
  dir
}

/// The fixture as a snapshot in a cache: the model directory `model/`
/// beside the cache's `blobs/`, as Hugging Face lays a snapshot out.
fn cached_fixture() -> (tempfile::TempDir, PathBuf) {
  let root = tempfile::tempdir().unwrap();
  let dir = root.path().join("model");
  fs::create_dir(&dir).unwrap();
  fill_fixture(&dir);
  (root, dir)
}

/// Moves `dir`'s file `rel` into the cache's `blobs/`, beside `dir`, as
/// `name`, and leaves a relative link to it in its place: a snapshot's file,
/// a link outside the snapshot. Returns the blob.
fn link_into_blobs(dir: &Path, rel: &str, name: &str) -> PathBuf {
  let blobs = dir.parent().unwrap().join("blobs");
  fs::create_dir_all(&blobs).unwrap();
  let blob = blobs.join(name);
  fs::rename(dir.join(rel), &blob).unwrap();
  let up = "../".repeat(Path::new(rel).components().count());
  symlink(format!("{up}blobs/{name}"), dir.join(rel)).unwrap();
  blob
}

/// The top-level names in `dir`, the lock's aside.
fn names(dir: &Path) -> BTreeSet<String> {
  fs::read_dir(dir)
    .unwrap()
    .map(|entry| entry.unwrap().file_name().into_string().unwrap())
    .filter(|name| name != LOCK_NAME)
    .collect()
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
    .map(|(rel, _)| {
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

/// CA1 (Codex round 1, finding 1): a supplied bundle is a source. Replaced by
/// another valid bundle under the same name, it is refused at load until the
/// directory is compiled again (its stamped files' sizes moved), and
/// `compile_artifacts` re-stamps it: the part's label moves with the bytes, so
/// a pin of the old label is refused.
#[test]
fn a_replaced_supplied_bundle_is_restamped_and_relabelled() {
  let dir = fixture_dir();
  supply(dir.path(), "EmbeddingGemma2Audio.mlmodelc");
  compile_artifacts(dir.path()).expect("compiles");
  let audio = |dir: &Path| stamp(dir).part(dir, Part::Audio, AudioPrecision::Fp16);
  let old = audio(dir.path())
    .expect("the supplied audio bundle")
    .label();
  let bundle = dir.path().join("EmbeddingGemma2Audio.mlmodelc");
  fs::remove_dir_all(&bundle).unwrap();
  let other = Model::compile(Path::new(MANIFEST).join("tests/fixtures/two_functions.mlpackage"))
    .expect("the two-function fixture compiles");
  copy_dir(&other, &bundle);
  match audio(dir.path()) {
    Err(Error::StaleStamp(stale)) => {
      assert_eq!(
        stale.artifact(),
        Artifact::AudioBundle(AudioPrecision::Fp16)
      );
    }
    other => panic!("expected StaleStamp before the directory is compiled again, got {other:?}"),
  }
  let before = stamp(dir.path()).bundles["audio_fp16"].digests.clone();
  compile_artifacts(dir.path()).expect("re-stamps");
  let after = stamp(dir.path()).bundles["audio_fp16"].digests.clone();
  assert_ne!(after, before, "the stamp records the new bundle's files");
  let view = audio(dir.path()).expect("the replaced bundle, re-stamped");
  assert_ne!(view.label(), old, "the label moves with the bytes");
  match view.expect(Some(old)) {
    Err(Error::ArtifactIdentity(identity)) => {
      assert_eq!(identity.part(), Part::Audio);
      assert_eq!(identity.expected(), old);
      assert_eq!(identity.actual(), view.label());
    }
    other => panic!("expected the old pin refused, got {other:?}"),
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
  for (i, pair) in sha.as_bytes().as_chunks::<2>().0.iter().enumerate() {
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
  let expected = ArtifactLabel::from_digests(lines.iter().map(|(rel, sha)| (rel.as_str(), *sha)))
    .expect("a label");
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

/// The text part's label in `dir`, from its stamp.
fn text_label(dir: &Path) -> ArtifactLabel {
  stamp(dir)
    .part(dir, Part::Text, AudioPrecision::Fp16)
    .expect("the text part")
    .label()
}

/// Codex round 2, finding 1 (the rule): a package whose files are links into
/// a cache's blobs, as Hugging Face lays a snapshot out, is stamped by its
/// links' targets. A target replaced is a replaced file: the directory is not
/// current, the package is compiled again under a new name, and the text
/// part's label moves, so a pin of the old label is refused. The fixture
/// package has no weights file; its model file, which Core ML reads, plays
/// the weights' part.
#[test]
fn a_package_linked_into_a_cache_is_restamped_when_a_target_is_replaced() {
  let (_root, dir) = cached_fixture();
  let model = "EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/model.mlmodel";
  link_into_blobs(
    &dir,
    "EmbeddingGemma2Text.mlpackage/Manifest.json",
    "manifest",
  );
  let blob = link_into_blobs(&dir, model, "model");
  compile_artifacts(&dir).expect("compiles through the links");
  let recorded = stamp(&dir);
  assert_eq!(
    recorded.files.get(model).map(|entry| entry.currency),
    Some(Currency::of(&fs::metadata(&blob).unwrap())),
    "the stamp records the link's target"
  );
  let (old_label, old_bundle) = (text_label(&dir), recorded.bundles["text"].name.clone());
  // The cache replaces the blob: another model's bytes under the same name.
  fs::remove_file(&blob).unwrap();
  fs::copy(
    Path::new(MANIFEST).join("tests/fixtures/identity.mlmodel"),
    &blob,
  )
  .unwrap();
  let report = compile_artifacts(&dir).expect("re-stamps");
  let new_bundle = stamp(&dir).bundles["text"].name.clone();
  assert_ne!(new_bundle, old_bundle, "compiled again under a new name");
  assert!(report.compiled().contains(&new_bundle), "{report:?}");
  let view = stamp(&dir)
    .part(&dir, Part::Text, AudioPrecision::Fp16)
    .expect("the text part");
  assert_ne!(view.label(), old_label, "the label moves with the target");
  assert!(
    matches!(
      view.expect(Some(old_label)),
      Err(Error::ArtifactIdentity(_))
    ),
    "a pin of the old label is refused"
  );
}

/// Codex round 2, finding 1 (the rule), on the supplied-bundle road: a
/// supplied bundle whose files are links into a cache's blobs is stamped by
/// their targets. A target replaced is refused at load until the directory is
/// compiled again (its size moved), and `compile_artifacts` re-stamps it under
/// a new label.
#[test]
fn a_supplied_bundle_linked_into_a_cache_is_restamped_when_a_target_is_replaced() {
  let (_root, dir) = cached_fixture();
  supply(&dir, "EmbeddingGemma2Audio.mlmodelc");
  let bundle = dir.join("EmbeddingGemma2Audio.mlmodelc");
  let blobs: Vec<PathBuf> = files_under(&dir, &bundle)
    .unwrap()
    .iter()
    .enumerate()
    .map(|(i, (rel, _))| link_into_blobs(&dir, rel, &format!("audio-{i}")))
    .collect();
  compile_artifacts(&dir).expect("compiles through the links");
  let audio = |dir: &Path| stamp(dir).part(dir, Part::Audio, AudioPrecision::Fp16);
  let old = audio(&dir).expect("the supplied audio bundle").label();
  // The cache replaces one blob with other bytes.
  let mut bytes = fs::read(&blobs[0]).unwrap();
  bytes.extend_from_slice(b"replaced");
  fs::remove_file(&blobs[0]).unwrap();
  fs::write(&blobs[0], &bytes).unwrap();
  match audio(&dir) {
    Err(Error::StaleStamp(stale)) => {
      assert_eq!(
        stale.artifact(),
        Artifact::AudioBundle(AudioPrecision::Fp16)
      );
    }
    other => panic!("expected StaleStamp before the directory is compiled again, got {other:?}"),
  }
  compile_artifacts(&dir).expect("re-stamps");
  let view = audio(&dir).expect("the bundle, re-stamped");
  assert_ne!(view.label(), old, "the label moves with the target");
  assert!(
    matches!(view.expect(Some(old)), Err(Error::ArtifactIdentity(_))),
    "a pin of the old label is refused"
  );
}

/// The rule: a broken link inside a tree is refused by name, before anything
/// is staged.
#[test]
fn a_broken_link_in_a_tree_is_refused_by_name() {
  let dir = fixture_dir();
  let link = dir
    .path()
    .join("EmbeddingGemma2Text.mlpackage/Data/missing.bin");
  symlink("no-such-blob", &link).unwrap();
  let before = names(dir.path());
  match compile_artifacts(dir.path()) {
    Err(Error::Traversal(refusal)) => {
      assert_eq!(refusal.defect(), TraversalDefect::BrokenLink);
      assert_eq!(refusal.path(), link);
    }
    other => panic!("expected the broken link refused, got {other:?}"),
  }
  assert_eq!(names(dir.path()), before, "nothing is staged");
}

/// The rule: a loose file that is a broken link is refused by name, not
/// taken for an absent file.
#[test]
fn a_broken_loose_link_is_refused_by_name() {
  let dir = fixture_dir();
  let json = dir.path().join("tokenizer.json");
  fs::remove_file(&json).unwrap();
  symlink("blobs/missing", &json).unwrap();
  match compile_artifacts(dir.path()) {
    Err(Error::Traversal(refusal)) => {
      assert_eq!(refusal.defect(), TraversalDefect::BrokenLink);
      assert_eq!(refusal.path(), json);
    }
    other => panic!("expected the broken link refused, got {other:?}"),
  }
}

/// The rule: a link back to a directory that holds it is a directory cycle,
/// refused by name rather than walked.
#[test]
fn a_directory_cycle_is_refused_by_name() {
  let dir = fixture_dir();
  let link = dir.path().join("EmbeddingGemma2Text.mlpackage/Data/loop");
  symlink("..", &link).unwrap();
  match compile_artifacts(dir.path()) {
    Err(Error::Traversal(refusal)) => {
      assert_eq!(refusal.defect(), TraversalDefect::Cycle);
      assert_eq!(refusal.path(), link);
    }
    other => panic!("expected the cycle refused, got {other:?}"),
  }
}

/// The rule: `prune` never follows a link. A derived-named link, to a
/// directory or to a file, is left as it is, and so is its target.
#[test]
fn prune_leaves_a_link_and_its_target_untouched() {
  let (root, dir) = cached_fixture();
  compile_artifacts(&dir).expect("compiles");
  let target_dir = root.path().join("elsewhere.mlmodelc");
  fs::create_dir(&target_dir).unwrap();
  fs::write(target_dir.join("coremldata.bin"), b"kept").unwrap();
  let target_file = root.path().join("elsewhere.gemmabpe");
  fs::write(&target_file, b"kept").unwrap();
  let bundle_link = dir.join("EmbeddingGemma2Text.0123456789ab.mlmodelc");
  let image_link = dir.join("tokenizer.v1-0123456789ab.gemmabpe");
  symlink(&target_dir, &bundle_link).unwrap();
  symlink(&target_file, &image_link).unwrap();
  let report = prune(&dir).expect("prunes");
  assert!(report.removed().is_empty(), "{report:?}");
  for link in [&bundle_link, &image_link] {
    assert!(
      fs::symlink_metadata(link).is_ok_and(|m| m.file_type().is_symlink()),
      "{} stays",
      link.display()
    );
  }
  assert_eq!(
    fs::read(target_dir.join("coremldata.bin")).unwrap(),
    b"kept"
  );
  assert_eq!(fs::read(&target_file).unwrap(), b"kept");
}

/// The rule: nothing writes through a link. A lock name that is a link is
/// refused, and the link's target is not created.
#[test]
fn a_lock_that_is_a_link_is_refused_and_nothing_is_written_through_it() {
  let (root, dir) = cached_fixture();
  let target = root.path().join("elsewhere.lock");
  symlink(&target, dir.join(LOCK_NAME)).unwrap();
  match compile_artifacts(&dir) {
    Err(Error::Io(failure)) => assert_eq!(failure.operation(), IoOperation::Lock),
    other => panic!("expected the lock refused, got {other:?}"),
  }
  assert!(!target.exists(), "nothing is written through the link");
}

/// The image build's reservations, in order: the dense ids, the id map, the
/// tokens, the merges, the bridges, the joined text, the scalars, the index,
/// the added tokens, the eight sections and the image.
const IMAGE_BUILD_RESERVATIONS: usize = 18;

/// Condition 4 (Codex round 2, finding 2): every reservation of the tokenizer
/// image's build goes through the door. Each of its 18, refused in turn, is
/// answered by name at `image_build`; past the last, the directory compiles.
#[test]
fn each_image_build_reservation_is_refused_by_name() {
  let dir = fixture_dir();
  // Not read by the build; spared its hashing on every call.
  fs::remove_file(dir.path().join("embeddings.bf16")).unwrap();
  for skip in 0..IMAGE_BUILD_RESERVATIONS {
    let _refuse = alloc::inject::refuse(AllocationSite::ImageBuild, skip);
    match compile_artifacts(dir.path()) {
      Err(Error::Allocation(refusal)) => {
        assert_eq!(
          refusal.site(),
          AllocationSite::ImageBuild,
          "reservation {skip}"
        );
      }
      other => panic!("reservation {skip}: expected a refusal at image_build, got {other:?}"),
    }
  }
  let _past = alloc::inject::refuse(AllocationSite::ImageBuild, IMAGE_BUILD_RESERVATIONS);
  compile_artifacts(dir.path()).expect("past the last reservation, the directory compiles");
}

/// Condition 4 (Codex round 2, finding 2): a refused image build stages
/// nothing. The image is built in memory before any package is compiled, so
/// the directory holds what it held.
#[test]
fn a_refused_image_build_stages_nothing() {
  let dir = fixture_dir();
  let before = names(dir.path());
  let _refuse = alloc::inject::refuse(AllocationSite::ImageBuild, 0);
  match compile_artifacts(dir.path()) {
    Err(Error::Allocation(refusal)) => assert_eq!(refusal.site(), AllocationSite::ImageBuild),
    other => panic!("expected a refusal at image_build, got {other:?}"),
  }
  assert_eq!(names(dir.path()), before, "nothing is staged");
}

/// The caps: a `tokenizer.json` one byte past its 64 MiB cap is refused by
/// name before a byte of it is read. It is unreadable here, so any read would
/// fail with another error.
#[test]
fn a_tokenizer_json_past_its_cap_is_refused_unread() {
  let dir = fixture_dir();
  let json = dir.path().join("tokenizer.json");
  File::create(&json)
    .unwrap()
    .set_len(TOKENIZER_JSON_CAP + 1)
    .unwrap();
  fs::set_permissions(&json, fs::Permissions::from_mode(0o000)).unwrap();
  let result = compile_artifacts(dir.path());
  fs::set_permissions(&json, fs::Permissions::from_mode(0o644)).unwrap();
  match result {
    Err(Error::TokenizerSource(defect)) => {
      assert_eq!(defect.field(), SourceField::Document);
      assert!(
        defect
          .detail()
          .starts_with(&format!("{} bytes", TOKENIZER_JSON_CAP + 1)),
        "{}",
        defect.detail()
      );
    }
    other => panic!("expected the cap refused, got {other:?}"),
  }
}

/// The caps: a stamp one byte past its 16 MiB cap is refused by name before
/// a byte of it is read.
#[test]
fn a_stamp_past_its_cap_is_refused_unread() {
  let dir = tempfile::tempdir().unwrap();
  let path = dir.path().join(STAMP_NAME);
  File::create(&path).unwrap().set_len(STAMP_CAP + 1).unwrap();
  fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
  let result = Stamp::read(dir.path());
  fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
  match result {
    Err(Error::Stamp(defect)) => {
      assert_eq!(defect.field(), StampField::Document);
      assert!(
        defect
          .detail()
          .starts_with(&format!("{} bytes", STAMP_CAP + 1)),
        "{}",
        defect.detail()
      );
    }
    other => panic!("expected the cap refused, got {other:?}"),
  }
}

/// The caps: a walked tree holds at most 1024 files and directories. A
/// supplied bundle at the cap is stamped; one entry more is refused by name,
/// naming the tree, before a byte of it is read (the extra file is unreadable
/// here).
#[test]
fn a_tree_past_its_entry_cap_is_refused_unread() {
  let dir = fixture_dir();
  fs::remove_file(dir.path().join("embeddings.bf16")).unwrap();
  let bundle = dir.path().join("EmbeddingGemma2Audio.mlmodelc");
  fs::create_dir(&bundle).unwrap();
  for i in 0..MAX_TREE_ENTRIES {
    fs::write(bundle.join(format!("{i:04}.bin")), b"").unwrap();
  }
  compile_artifacts(dir.path()).expect("a tree at its cap is stamped");
  let extra = bundle.join("extra.bin");
  fs::write(&extra, b"unreadable").unwrap();
  fs::set_permissions(&extra, fs::Permissions::from_mode(0o000)).unwrap();
  let result = compile_artifacts(dir.path());
  fs::set_permissions(&extra, fs::Permissions::from_mode(0o644)).unwrap();
  match result {
    Err(Error::Traversal(refusal)) => {
      assert_eq!(refusal.defect(), TraversalDefect::TooManyEntries);
      assert_eq!(refusal.path(), bundle);
    }
    other => panic!("expected the tree refused, got {other:?}"),
  }
}
