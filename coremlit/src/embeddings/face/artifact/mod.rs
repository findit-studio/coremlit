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
//!
//! # The digest names the bytes at `load`, and quiescence is a PRECONDITION
//!
//! One walk, taken from the path handed to [`crate::Model::load`]. The value it
//! produces identifies the bytes **as read at `load`**, and under this crate's
//! threat model that is also the bytes every later prediction runs on:
//!
//! > **The artifact must not be modified in place while a
//! > [`crate::embeddings::face::FaceEmbedder`] holds it.** That is the same
//! > precondition CoreML itself has for a model it has mapped. A model is
//! > replaced by an atomic `rename` followed by loading a new embedder — the
//! > live mapping keeps the old inode's bytes, which is what every macOS
//! > updater relies on.
//!
//! A digest is not a defence against a hostile artifact or a hostile
//! filesystem, and neither is in this library's scope: whoever can rewrite the
//! bundle under a running process can already choose which bytes it loads. What
//! the digest is for is *confusion* — vectors from one set of weights scored
//! against vectors from another — and one walk at `load` catches that.

use std::{fs, io::Read, os::unix::ffi::OsStrExt, path::Path};

use sha2::{Digest, Sha256};

use crate::embeddings::face::error::{DigestFailure, Error, Result};

/// How deep [`digest_artifact`] will walk before refusing.
///
/// **A plain resource cap, and no longer a safety mechanism.** It used to be
/// the only thing standing between the walk and an unbounded one, because
/// directory symlinks were followed and a link pointing at one of its own
/// parents is a cycle. A depth cap is not a bound on a GRAPH: two links per
/// level over ~25 physical levels expand to ~33 million logical leaves while
/// staying far inside this number. Directory symlinks are refused now (see
/// [`digest_artifact`]), so the walk is linear in the PHYSICAL tree and this
/// is a backstop against a pathologically nested real directory rather than
/// against a DAG. A real compiled bundle nests two or three levels.
const MAX_DEPTH: usize = 64;

/// How many directory entries [`digest_artifact`] will visit before refusing.
///
/// The second of the two plain resource caps on a walk that is now linear in
/// the physical tree — [`MAX_DEPTH`] bounds it downwards, this one bounds its
/// total size. Every entry the walk meets counts, whether it turns out to be a
/// file, a directory or neither. A compiled bundle holds a handful of files,
/// so 4 096 is generous by three orders of magnitude.
///
/// It REFUSES rather than truncating, like every other failure here: a digest
/// standing for "the first 4 096 files" would be an identity for bytes nobody
/// has.
const MAX_ENTRIES: usize = 4096;

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
  /// That mattered more when nothing runnable drove the door at all. It still
  /// matters: the only artifact this repository stages rides
  /// `commercial-face-arcface`, so in every OTHER configuration — the default
  /// build, and `--features face` — replacing the digest `load` stamps with a
  /// constant fails no test, and fails the build instead. The general form of
  /// that gap is issue #138 §4/§6 and the reason `Checked<Model>` exists;
  /// this is the one field of it that a private constructor can close on its
  /// own.
  ///
  /// WHERE the door takes the digest — one walk of the same path it hands
  /// [`crate::Model::load`], after the contract has been checked — is a
  /// separate question a private constructor cannot answer. That one is pinned
  /// over the single real bundle this repository commits, by
  /// `the_face_door_refuses_the_vendored_silero_bundle` and
  /// `a_load_that_cannot_open_the_artifact_never_walks_it` in the `embed`
  /// module's tests.
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
/// - **every regular file, with NO exemption by name.** A dot-prefixed child
///   is hashed exactly like any other. The rule used to skip them so that a
///   `.DS_Store` would not move the digest, and that was an enumeration of
///   "what does not matter" with a case missing: a CoreML ML Program can name
///   an external `BLOBFILE` by path, and `@model_path/.weights/weight.bin` is
///   a legal one — so two bundles agreeing on every visible file and
///   differing in their hidden weights had ONE digest, and vectors from one
///   were scored against vectors from the other. Sparing `.weights` next would
///   be the next enumeration; no rule over NAMES separates the model from the
///   noise, because the filesystem does not record that distinction. **The consequence, stated rather than dodged:** a
///   bundle a Finder window has been opened on is a different artifact from
///   the same bundle on a worker that never browsed it, and their embeddings
///   do not compare until the `.DS_Store` is removed. That is the honest
///   answer — the artifact is a different set of bytes — and it is the rule
///   `MODELS_LOCK` already applies to bundle bytes everywhere else here.
/// - **FILE symlinks followed, DIRECTORY symlinks refused.** A link to a file
///   is hashed as the bytes it resolves to, which is what a bundle assembled
///   out of links needs — a Hugging Face cache snapshot is a directory of file
///   links into `blobs/`, and it must work. A link to a DIRECTORY is
///   [`Error::ArtifactDigest`] naming the link, because recursing through one
///   makes this a walk of a GRAPH rather than of a tree: two links per level
///   over ~25 physical levels expand to ~33 million logical leaves while
///   staying far inside [`MAX_DEPTH`], so the walk exhausts memory long before
///   it exhausts its depth. No recursion through a link means no DAG, which is
///   why both caps below are plain resource caps on a walk that is linear in
///   the PHYSICAL tree. A BROKEN file link is an error rather than a skip: it
///   cannot be followed, and a bundle with one is not a bundle whose bytes are
///   known.
/// - **`root` may be a regular file**, in which case there is exactly one
///   entry and its `relative` is empty. [`crate::Model::load`] accepts any
///   path CoreML accepts, and a compiled `.mlmodelc` is a directory in
///   practice, but hashing what was actually loaded must not depend on that.
///   `root` may also itself be a symlink to a directory: it is the path the
///   caller chose rather than something found inside the bundle, it is
///   resolved exactly once, and nothing recurses through it.
///
/// # Every allocation here is bounded by a constant
///
/// [`crate::embeddings::face::embed`]'s rule is that a length known only at
/// run time is reserved fallibly. Nothing on this walk has one. The entry list
/// grows by `push` and [`MAX_ENTRIES`] refuses the 4 097th before it is
/// reached; each `relative` path is bounded by [`MAX_DEPTH`] names the
/// filesystem has already capped; the read buffer is exactly [`READ_CHUNK`],
/// which is why a multi-hundred-megabyte `weight.bin` is streamed rather than
/// read whole; and the only [`std::path::PathBuf`] built is on a refusal, from
/// a path that already exists. So the walk allocates infallibly, and no number
/// out of the artifact can move what it asks for.
///
/// # Errors
/// [`Error::ArtifactDigest`] naming the path that failed, for any I/O failure
/// while walking or reading, for a `root` that is neither a directory nor a
/// regular file, for a symlink to a directory anywhere under `root`, and for a
/// tree that exceeds [`MAX_DEPTH`] or [`MAX_ENTRIES`]. It fails closed: there
/// is no digest that stands for "some of the bytes".
pub(crate) fn digest_artifact(root: &Path) -> Result<ArtifactDigest> {
  let metadata = fs::metadata(root).map_err(|source| failure(root, source))?;
  let mut entries = Vec::new();
  if metadata.is_dir() {
    collect(root, Vec::new(), 0, &mut 0, &mut entries)?;
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
///
/// `visited` counts every entry the whole walk has met — files, directories
/// and everything else — against [`MAX_ENTRIES`]. It is threaded rather than
/// derived from `entries.len()` because the cap is on the WALK, and a tree can
/// be arbitrarily large in directories that contribute no file at all.
fn collect(
  directory: &Path,
  prefix: Vec<u8>,
  depth: usize,
  visited: &mut usize,
  entries: &mut Vec<(Vec<u8>, [u8; 32])>,
) -> Result<()> {
  if depth >= MAX_DEPTH {
    return Err(failure(
      directory,
      std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "artifact nesting exceeds the walk depth limit",
      ),
    ));
  }
  for entry in fs::read_dir(directory).map_err(|source| failure(directory, source))? {
    let entry = entry.map_err(|source| failure(directory, source))?;
    *visited += 1;
    if *visited > MAX_ENTRIES {
      return Err(failure(
        directory,
        std::io::Error::new(
          std::io::ErrorKind::InvalidInput,
          "artifact holds more entries than the walk will visit",
        ),
      ));
    }
    let name = entry.file_name();
    let path = entry.path();
    // `metadata`, not `symlink_metadata`: a link to a FILE is followed, so a
    // bundle of links into a shared blob store hashes as the bytes it reads.
    let metadata = fs::metadata(&path).map_err(|source| failure(&path, source))?;
    let mut relative = prefix.clone();
    if !relative.is_empty() {
      relative.push(b'/');
    }
    relative.extend_from_slice(name.as_bytes());
    if metadata.is_dir() {
      // A link to a DIRECTORY is refused rather than walked. Recursing
      // through one turns the walk into a walk of a graph, which no depth cap
      // bounds; refusing keeps it linear in the physical tree. The extra
      // `symlink_metadata` is paid only on directories, so a bundle of file
      // links costs nothing for it.
      let link = fs::symlink_metadata(&path).map_err(|source| failure(&path, source))?;
      if link.file_type().is_symlink() {
        return Err(failure(
          &path,
          std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "a symlink to a directory is not walked",
          ),
        ));
      }
      collect(&path, relative, depth + 1, visited, entries)?;
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
