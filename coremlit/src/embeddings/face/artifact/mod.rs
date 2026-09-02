//! The identity of the BYTES a face embedder was loaded from.
//!
//! An [`crate::embeddings::face::EmbeddingSpace`] is a claim about which
//! function produced a vector, and the trained parameters are most of that
//! function. Everything else the manifest carries — the width, the
//! preprocessing, the feature names — is schema: two unrelated artifacts are
//! free to agree on all of it, and their cosine would then be returned rather
//! than refused. This module closes that by hashing the artifact directory
//! itself, so the space a vector carries names the weights it came out of.
//!
//! # Identity is of the BYTES, not of the load
//!
//! A token minted per `load` would be simpler and would be wrong here.
//! `&self` inference means fan-out is one [`crate::embeddings::face::FaceEmbedder`]
//! per worker over the same artifact, so the same space is legitimately
//! produced by more than one producer, and a per-load token would refuse
//! exactly the cross-worker comparisons those workers exist to make. A digest
//! of the bytes is equal across workers, across processes and across machines,
//! which is what the identity has to be.
//!
//! It is also not caller-forgeable in the way that matters: the caller chooses
//! which artifact to load, not what its digest is, and [`ArtifactDigest`] has
//! no public constructor.

use std::{fs, io::Read, os::unix::ffi::OsStrExt, path::Path};

use sha2::{Digest, Sha256};

use crate::embeddings::face::error::{DigestFailure, Error, Result};

/// How deep [`digest_artifact`] will walk before refusing.
///
/// Symlinks are followed, and a symlink that points at one of its own parents
/// is an unbounded walk. A real compiled bundle nests two or three levels, so
/// this is a backstop against a malformed tree rather than a limit anything
/// legitimate approaches.
const MAX_DEPTH: usize = 64;

/// Bytes read per `read` call while hashing one file. A compiled model's
/// `weights/weight.bin` can be hundreds of megabytes, so it is streamed rather
/// than read whole.
const READ_CHUNK: usize = 1 << 16;

/// The SHA-256 identity of one compiled model artifact's bytes.
///
/// Produced only by [`crate::embeddings::face::FaceEmbedder::load`], from the
/// path it loads. **There is no public constructor**: a caller picks the
/// artifact, and this value is then a fact about it rather than a claim about
/// it.
///
/// Two `FaceEmbedder`s on different machines that loaded byte-identical
/// bundles hold equal digests, which is what lets their embeddings be
/// compared. Two that loaded different bundles do not, and
/// [`crate::embeddings::face::FaceEmbedding::dot`] refuses across them.
///
/// ```compile_fail,E0599
/// use coremlit::embeddings::face::ArtifactDigest;
/// // There is no public constructor: a digest is a fact about the bytes
/// // `FaceEmbedder::load` read, not a value a caller gets to state.
/// let _ = ArtifactDigest::from_raw([0u8; 32]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactDigest([u8; 32]);

impl ArtifactDigest {
  /// The 32 digest bytes.
  #[inline(always)]
  pub const fn as_bytes(&self) -> &[u8; 32] {
    &self.0
  }

  /// Wrap 32 already-computed bytes — **test-only**.
  ///
  /// A unit gate needs to name two distinct spaces without staging two
  /// `.mlmodelc` directories, and `#[cfg(test)]` is what lets it do that
  /// without leaving a second producer in the library. In a non-test build
  /// [`digest_artifact`] is the ONLY way to obtain an `ArtifactDigest` outside
  /// this module, so a door that stamped a made-up identity onto an embedding
  /// does not compile rather than merely going untested.
  ///
  /// That matters here specifically: `FaceEmbedder::load` needs a real
  /// artifact and this crate stages no face model, so **nothing runnable
  /// drives that door** — replacing its `digest_artifact` call with a constant
  /// fails no test. It fails the build instead. The general form of that gap
  /// is issue #138 §4/§6 and the reason `Checked<Model>` exists; this is the
  /// one field of it that a private constructor can close on its own.
  #[cfg(test)]
  #[inline(always)]
  pub(crate) const fn from_raw(bytes: [u8; 32]) -> Self {
    Self(bytes)
  }
}

/// The [`ArtifactDigest`] of everything under `root`.
///
/// # The encoding, stated exactly
///
/// Let the ENTRIES be every regular file reachable from `root`, each written
/// as a pair `(relative, SHA-256(file contents))` where `relative` is the
/// file's path below `root` with its components joined by a single `/`
/// (`0x2F`) and no leading separator. Sort the entries by `relative`, compared
/// as raw bytes. The digest is then
///
/// ```text
/// SHA-256( for each entry in order:  u64_le(relative.len()) ‖ relative ‖ sha256(file) )
/// ```
///
/// The length prefix is what makes that encoding injective: without it
/// `("ab", h₁), ("c", h₂)` and `("a", h₁'), ("bc", h₂')` could serialise to the
/// same bytes, and two different trees would hash the same. Sorting is what
/// makes it deterministic — `read_dir` order is the filesystem's business, not
/// the artifact's.
///
/// Four rules about what counts, each of which a gate in `tests.rs` pins:
///
/// - **regular files only.** A directory contributes only through the files
///   under it, so an empty directory is invisible; anything that is neither a
///   directory nor a regular file (a socket, a device node) carries no
///   artifact bytes and is skipped.
/// - **dotfiles excluded, at every level.** `.DS_Store` is written into any
///   directory a Finder window has been opened on, and Spotlight and
///   AppleDouble files appear the same way. None of them are the model, and a
///   digest that moved when one appeared would refuse comparisons for a reason
///   having nothing to do with the weights.
/// - **symlinks followed.** `fs::metadata` rather than `symlink_metadata`, so
///   a bundle assembled out of links hashes as the bytes it resolves to. A
///   BROKEN link is an error rather than a skip: it cannot be followed, and a
///   bundle with one is not a bundle whose bytes are known.
/// - **`root` may be a regular file**, in which case there is exactly one
///   entry and its `relative` is empty. [`crate::Model::load`] accepts any
///   path CoreML accepts, and a compiled `.mlmodelc` is a directory in
///   practice, but hashing what was actually loaded must not depend on that.
///
/// # Errors
/// [`Error::ArtifactDigest`] naming the path that failed, for any I/O failure
/// while walking or reading, for a `root` that is neither a directory nor a
/// regular file, and for a tree nested past [`MAX_DEPTH`]. It fails closed:
/// there is no digest that stands for "some of the bytes".
pub(crate) fn digest_artifact(root: &Path) -> Result<ArtifactDigest> {
  let metadata = fs::metadata(root).map_err(|source| failure(root, source))?;
  let mut entries = Vec::new();
  if metadata.is_dir() {
    collect(root, Vec::new(), 0, &mut entries)?;
  } else if metadata.is_file() {
    entries.push((Vec::new(), file_digest(root)?));
  } else {
    return Err(failure(
      root,
      std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "not a directory or a regular file",
      ),
    ));
  }
  Ok(fold_entries(entries))
}

/// Sorts the `(relative path, file digest)` entries and folds them into the
/// artifact digest.
///
/// Split out of [`digest_artifact`] because both of its properties are
/// properties of THIS function and of nothing else, and one of them cannot be
/// tested through a filesystem at all:
///
/// - **the sort** is what makes the digest independent of `read_dir` order,
///   which is the filesystem's business rather than the artifact's;
/// - **the length prefix** is what makes the concatenation injective. Without
///   it the entry lists `[("x", Hx), ("y", Hy)]` and `[("x"‖Hx‖"y", Hy)]`
///   serialise to the same bytes, and two different artifacts get one
///   identity. That collision needs a file NAME holding the raw bytes of a
///   SHA-256, which APFS refuses (`EILSEQ`: a name must be valid UTF-8) — so
///   the gate feeds the two lists in here directly rather than staging them.
///   Unreachable through one filesystem is not the same as absent from the
///   encoding, and the encoding is what this crate defines.
fn fold_entries(mut entries: Vec<(Vec<u8>, [u8; 32])>) -> ArtifactDigest {
  entries.sort_by(|(left, _), (right, _)| left.cmp(right));
  let mut hasher = Sha256::new();
  for (relative, digest) in &entries {
    // The length prefix, before the bytes it measures.
    hasher.update((relative.len() as u64).to_le_bytes());
    hasher.update(relative);
    hasher.update(digest);
  }
  // The one place the library writes this field; see `ArtifactDigest::from_raw`
  // for why there is no non-test constructor beside it.
  ArtifactDigest(hasher.finalize().into())
}

/// Appends every regular file under `directory` to `entries`, with `prefix` as
/// its path below the artifact root.
fn collect(
  directory: &Path,
  prefix: Vec<u8>,
  depth: usize,
  entries: &mut Vec<(Vec<u8>, [u8; 32])>,
) -> Result<()> {
  if depth >= MAX_DEPTH {
    return Err(failure(
      directory,
      std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "artifact nesting exceeds the walk depth limit (a symlink cycle?)",
      ),
    ));
  }
  for entry in fs::read_dir(directory).map_err(|source| failure(directory, source))? {
    let entry = entry.map_err(|source| failure(directory, source))?;
    let name = entry.file_name();
    if name.as_bytes().first() == Some(&b'.') {
      continue;
    }
    let path = entry.path();
    // `metadata`, not `symlink_metadata`: links are FOLLOWED.
    let metadata = fs::metadata(&path).map_err(|source| failure(&path, source))?;
    let mut relative = prefix.clone();
    if !relative.is_empty() {
      relative.push(b'/');
    }
    relative.extend_from_slice(name.as_bytes());
    if metadata.is_dir() {
      collect(&path, relative, depth + 1, entries)?;
    } else if metadata.is_file() {
      entries.push((relative, file_digest(&path)?));
    }
  }
  Ok(())
}

/// SHA-256 of one file's contents, streamed.
fn file_digest(path: &Path) -> Result<[u8; 32]> {
  let mut file = fs::File::open(path).map_err(|source| failure(path, source))?;
  let mut hasher = Sha256::new();
  let mut buffer = vec![0u8; READ_CHUNK];
  loop {
    let read = file
      .read(&mut buffer)
      .map_err(|source| failure(path, source))?;
    if read == 0 {
      break;
    }
    hasher.update(&buffer[..read]);
  }
  Ok(hasher.finalize().into())
}

/// One I/O failure, named by the path it happened on.
fn failure(path: &Path, source: std::io::Error) -> Error {
  Error::ArtifactDigest(DigestFailure::new(path.to_path_buf(), source))
}

#[cfg(test)]
mod tests;
