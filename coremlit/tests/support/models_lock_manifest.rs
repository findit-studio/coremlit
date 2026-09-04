//! **The one file that says which bytes an artifact is.**
//!
//! `MODELS_LOCK.d/<vendor_dir>@<revision>.sha256` holds one committed file list
//! per `MODELS_LOCK` table — upstream's own `CHECKSUMS.sha256` where one ships,
//! a `shasum -a 256` over the staged tree where none does — and ci.yml checks
//! it against the bytes CI downloads. `tests/model_licences.rs` enumerates a
//! globbed table's contents from it (coremlit #139); the per-kit `model_io`
//! gates read the digests they assert from it, through this module.
//!
//! **Why the gates read it rather than holding their own copy.** They used to
//! hold a `const ARTIFACT_SHA256: &[(&str, &str)]` each, and the licence table
//! held a THIRD copy of the same hashes tied to those consts by `pins_at` — a
//! hand-rolled scanner over Rust source, anchored on `const NAME:`, reading
//! quoted runs. Three copies of one fact, kept in step by a reader that could
//! drop an entry on a stray quoted string. One copy, in the format the tool
//! that produced it writes, is the whole point.
//!
//! # The grammar
//!
//! ```text
//! <64 lowercase hex digits><two spaces><path relative to the table's local-dir>
//! ```
//!
//! and nothing else. The path names ONE file UNDER the table's `local-dir`:
//! at most one leading `./` is stripped — `shasum` writes one, and the copies
//! committed verbatim carry it — and every `/`-separated component that remains
//! must be non-empty and neither `.` nor `..`. Every other line shape PANICS: a
//! blank line, a comment, a short or upper-case digest, one space instead of
//! two, `shasum`'s ` *` binary marker, an absolute path, `../sibling`, `a/..`,
//! `.`, `a//b`, `a/./b`, a trailing `/`, and a repeated path — repeated now
//! even when the two digests agree. A tolerant reader here would let a gate
//! assert over a SUBSET of a bundle and report success, which is the one
//! failure an exact-manifest gate exists to prevent.
//!
//! Mirrored by `read_manifest` in `tests/model_licences.rs`, whose
//! `falsifiers::the_manifest_reader_refuses_every_line_that_is_not_the_grammar`
//! drives every refusal above; the two are the same grammar read for two
//! purposes, and `every_committed_manifest_belongs_to_a_staged_table` is what
//! keeps the directory itself honest.

use std::{
  collections::BTreeMap,
  path::{Path, PathBuf},
};

/// The directory holding one committed manifest per staged table.
#[allow(dead_code)]
pub const MANIFEST_DIR: &str = "MODELS_LOCK.d";

/// The committed manifest for one table, as table-relative path to SHA-256.
///
/// `vendor_dir` is the table's `local-dir` with `Models/` removed; `revision`
/// is the table's pinned revision. Both come from `MODELS_LOCK`, and a caller
/// that reads them from its own kit constants gets a red the day either moves
/// without the manifest being regenerated — which is the intent, since the file
/// name is the tie.
///
/// `root` is the workspace root. Passed in rather than resolved here because
/// every caller already includes `support/workspace_root.rs` and a second
/// `#[path]` include of one file inside one test binary is
/// `clippy::duplicate_mod`.
#[allow(dead_code)]
pub fn table_manifest(root: &Path, vendor_dir: &str, revision: &str) -> BTreeMap<String, String> {
  read(&manifest_path(root, vendor_dir, revision))
}

/// One bundle's files, keyed BUNDLE-relative — the shape a `model_io` gate
/// asserts an exact manifest in.
///
/// `bundle` is table-relative (`redimnet_b5.mlmodelc`,
/// `ced-tiny/ced_tiny.mlmodelc`). Empty is a panic rather than an empty
/// assertion: a gate whose expected set silently became empty passes over any
/// tree at all.
#[allow(dead_code)]
pub fn bundle_manifest(
  root: &Path,
  vendor_dir: &str,
  revision: &str,
  bundle: &str,
) -> Vec<(String, String)> {
  let prefix = format!("{bundle}/");
  let files: Vec<(String, String)> = table_manifest(root, vendor_dir, revision)
    .into_iter()
    .filter_map(|(path, sha)| {
      path
        .strip_prefix(&prefix)
        .map(|tail| (tail.to_string(), sha))
    })
    .collect();
  assert!(
    !files.is_empty(),
    "{MANIFEST_DIR}/{vendor_dir}@{revision}.sha256 lists no file under {bundle:?}. The bundle \
     name and the manifest have diverged, and an empty expected set would let this gate pass \
     against any tree."
  );
  files
}

/// One file's digest, by table-relative path.
#[allow(dead_code)]
pub fn file_digest(root: &Path, vendor_dir: &str, revision: &str, path: &str) -> String {
  table_manifest(root, vendor_dir, revision)
    .remove(path)
    .unwrap_or_else(|| {
      panic!("{MANIFEST_DIR}/{vendor_dir}@{revision}.sha256 lists no {path:?}");
    })
}

/// Where a table's manifest lives.
#[allow(dead_code)]
pub fn manifest_path(root: &Path, vendor_dir: &str, revision: &str) -> PathBuf {
  root
    .join(MANIFEST_DIR)
    .join(format!("{vendor_dir}@{revision}.sha256"))
}

/// One manifest path, canonicalised — or why it is not one file under the
/// table's root.
///
/// The SAME rule lives in `tests/model_licences.rs` (`table_relative_path`
/// there) and in `.github/actions/stage-models/stage.sh`'s per-manifest awk.
/// `falsifiers::the_manifest_readers_refuse_every_path_that_is_not_one_file_under_the_table_root`
/// in `tests/model_licences.rs` drives BOTH Rust copies over one case table, so
/// a shape only one of them refuses fails there rather than in production.
///
/// `..` is the case with teeth: stage.sh hashes a manifest path with the staged
/// `local-dir` as its working directory, so `../sibling` would verify a file
/// this table does not stage while reporting it as the table's own. `.`, a
/// trailing `/` and `a/.` name a DIRECTORY, which has no digest, and `a//b`,
/// `a/./b` and `././a` are extra spellings of one path — the way one file gets
/// listed twice and read as two.
fn table_relative_path(raw: &str) -> Result<&str, &'static str> {
  // At most ONE leading `./`: that is what `shasum` writes, and what the
  // verbatim upstream copies carry on every line.
  let path = raw.strip_prefix("./").unwrap_or(raw);
  if path.is_empty() {
    return Err("is empty");
  }
  for component in path.split('/') {
    match component {
      "" => {
        return Err(
          "has an empty path component — a leading `/`, a doubled `/`, or a trailing one",
        );
      }
      "." => return Err("has a `.` component, which names a directory rather than a file"),
      ".." => {
        return Err("has a `..` component, so it can resolve outside the table's `local-dir`");
      }
      _ => {}
    }
  }
  Ok(path)
}

/// The grammar, read strictly. See this module's doc for every refused shape.
///
/// `pub` so `tests/model_licences.rs` can drive it directly: this module's doc
/// claims that file's falsifiers exercise every refusal here, and the claim was
/// unenforceable while the reader was private — each copy of the grammar could
/// only be tested through its own callers, which is how two readers of one file
/// format drift.
#[allow(dead_code)]
pub fn read(path: &Path) -> BTreeMap<String, String> {
  let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
    panic!(
      "read {}: {e}. The committed manifest is where this gate's expected digests come from; a \
       missing one is not an empty expectation.",
      path.display()
    )
  });
  let mut manifest = BTreeMap::new();
  for (at, line) in text.lines().enumerate() {
    let line_no = at + 1;
    let (hex, rest) = line.split_at_checked(64).unwrap_or_else(|| {
      panic!(
        "{}:{line_no}: {line:?} is shorter than a SHA-256; every line is `<64 lowercase \
         hex><two spaces><path>` and nothing else",
        path.display()
      )
    });
    assert!(
      hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
      "{}:{line_no}: {hex:?} is not 64 lowercase hex digits",
      path.display()
    );
    let file = rest.strip_prefix("  ").unwrap_or_else(|| {
      panic!(
        "{}:{line_no}: expected two spaces after the digest, got {rest:?}",
        path.display()
      )
    });
    let canonical = table_relative_path(file).unwrap_or_else(|why| {
      panic!(
        "{}:{line_no}: {file:?} {why}. Every line names ONE file UNDER the table's `local-dir`.",
        path.display()
      )
    });
    if manifest
      .insert(canonical.to_string(), hex.to_string())
      .is_some()
    {
      panic!(
        "{}:{line_no}: {canonical:?} is listed twice (this line spells it {file:?})",
        path.display()
      );
    }
  }
  assert!(!manifest.is_empty(), "{} lists no file", path.display());
  manifest
}
