//! M0: `compile_artifacts` over the staged EmbeddingGemma 2 tree.
//!
//! CI's compile step runs this target (the `embeddinggemma2` row's
//! `compile: embeddinggemma2|embeddinggemma2_compile`) after the checksum step
//! and BEFORE the fp16 sweep, so the sweep audits the compiled graphs on every
//! run, cold cache or warm. Locally it compiles a staged tree in place, which
//! is the only write `compile_artifacts` makes there.
//!
//! It also proves that what was compiled is exactly what the lock staged: the
//! stamp's source digests are the committed manifests' lines (the base's
//! thirteen and the overlay's package files), the
//! tokenizer image was built from upstream's `tokenizer.json`, the stamp names
//! one content-addressed bundle per package, and a second call keeps
//! everything and makes nothing.
//!
//! The tree is `EMBEDDINGGEMMA2_TEST_MODELS`, else the workspace's
//! `Models/embeddinggemma2`; with neither present the gate says so and passes.

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops.
#[path = "../support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

// The committed manifest reader, one grammar for every gate that asserts it.
#[path = "../support/models_lock_manifest.rs"]
#[allow(dead_code)]
mod models_lock_manifest;

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use coremlit::embeddings::embeddinggemma2::compile_artifacts;
use serde_json::Value;

/// The base table's pinned revision, which names its committed manifest.
const REVISION: &str = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4";

/// The overlay table's pinned revision (the fp32 audio package's upload), which
/// names its committed manifest.
const OVERLAY_REVISION: &str = "02b08df00754a4ebf380826732677ad4835c7b6c";

/// The base table's `local-dir` without `Models/`.
const VENDOR: &str = "embeddinggemma2";

/// Upstream's `tokenizer.json`, the one source the tokenizer image is built
/// from.
const TOKENIZER_SOURCE: &str = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4";

/// The stamp `compile_artifacts` writes beside the sources.
const STAMP: &str = "coremlit-embeddinggemma2.stamp.json";

/// Each package the two tables stage, by its stamp key and its stem.
const PACKAGES: [(&str, &str); 4] = [
  ("text", "EmbeddingGemma2Text"),
  ("audio_fp16", "EmbeddingGemma2Audio"),
  ("audio_fp32", "EmbeddingGemma2AudioFp32"),
  ("vision", "EmbeddingGemma2Vision"),
];

/// The staged tree, or `None` (with the reason printed) when there is none.
fn staged() -> Option<PathBuf> {
  let dir = std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || workspace_root::models_root().join("embeddinggemma2"),
    PathBuf::from,
  );
  if dir.is_dir() {
    Some(dir)
  } else {
    eprintln!(
      "SKIP: no staged EmbeddingGemma 2 tree at {} (set EMBEDDINGGEMMA2_TEST_MODELS)",
      dir.display()
    );
    None
  }
}

fn str_at<'v>(value: &'v Value, pointer: &str) -> &'v str {
  value
    .pointer(pointer)
    .and_then(Value::as_str)
    .unwrap_or_else(|| panic!("the stamp has no string at {pointer}"))
}

/// Whether `name` is `<stem>.<12 lowercase hex>.mlmodelc`.
fn content_addressed(name: &str, stem: &str) -> bool {
  name
    .strip_prefix(stem)
    .and_then(|rest| rest.strip_prefix('.'))
    .and_then(|rest| rest.strip_suffix(".mlmodelc"))
    .is_some_and(|id| {
      id.len() == 12
        && id
          .bytes()
          .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS); CI's compile step"]
fn the_staged_tree_compiles_to_exactly_what_the_lock_staged() {
  let Some(dir) = staged() else {
    return;
  };

  let started = Instant::now();
  let first = compile_artifacts(&dir).expect("the staged tree compiles");
  eprintln!(
    "M10: compile_artifacts took {:.1?}; compiled {:?}; kept {:?}",
    started.elapsed(),
    first.compiled(),
    first.kept()
  );

  let stamp: Value = serde_json::from_slice(
    &std::fs::read(dir.join(STAMP)).unwrap_or_else(|e| panic!("read the stamp: {e}")),
  )
  .unwrap_or_else(|e| panic!("parse the stamp: {e}"));

  let stamped: BTreeMap<String, String> = stamp
    .pointer("/files")
    .and_then(Value::as_object)
    .expect("the stamp records its files")
    .iter()
    .map(|(path, entry)| (path.clone(), str_at(entry, "/sha256").to_owned()))
    .collect();
  let root = workspace_root::workspace_root();
  let mut manifest = models_lock_manifest::table_manifest(&root, VENDOR, REVISION);
  // The overlay's manifest also lists its own `CHECKSUMS.sha256` (against the
  // empty digest, as speakerkit's does): a digest list, not a source file.
  manifest.extend(
    models_lock_manifest::table_manifest(&root, VENDOR, OVERLAY_REVISION)
      .into_iter()
      .filter(|(path, _)| path != "CHECKSUMS.sha256"),
  );
  assert_eq!(
    stamped, manifest,
    "the stamp's sources are not the committed manifests' lines: compile_artifacts recorded a \
     file the lock does not stage, or missed one it does"
  );

  assert_eq!(
    str_at(&stamp, "/tokenizer_image/source_sha256"),
    TOKENIZER_SOURCE,
    "the tokenizer image was built from another tokenizer.json"
  );

  let bundles = stamp
    .pointer("/bundles")
    .and_then(Value::as_object)
    .expect("the stamp names its bundles");
  assert_eq!(
    bundles.keys().map(String::as_str).collect::<Vec<_>>(),
    {
      let mut keys: Vec<&str> = PACKAGES.iter().map(|(key, _)| *key).collect();
      keys.sort_unstable();
      keys
    },
    "the stamp names a bundle per staged package, and nothing else"
  );
  let mut named = Vec::new();
  for (key, stem) in PACKAGES {
    let name = str_at(&stamp, &format!("/bundles/{key}/name"));
    assert!(
      content_addressed(name, stem),
      "{key}: {name} is not a content-addressed {stem} bundle"
    );
    assert_eq!(
      str_at(&stamp, &format!("/bundles/{key}/origin")),
      "compiled",
      "{key}: compiled from its package"
    );
    assert!(dir.join(name).is_dir(), "{key}: {name} exists");
    named.push(name.to_owned());
  }
  named.push(str_at(&stamp, "/tokenizer_image/name").to_owned());
  named.sort();

  let second = compile_artifacts(&dir).expect("the compiled tree is current");
  assert!(
    second.compiled().is_empty(),
    "a second call made {:?}",
    second.compiled()
  );
  let mut kept = second.kept().to_vec();
  kept.sort();
  assert_eq!(
    kept, named,
    "a second call keeps everything the stamp names"
  );
}
