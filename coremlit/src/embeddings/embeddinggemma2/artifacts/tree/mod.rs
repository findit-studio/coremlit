//! A model directory's tree: the one traversal rule every walk of the kit
//! follows, and the one way the kit reads a document whole.
//!
//! **The rule.**
//! - **Reading follows every link, wherever it points.** The stamp's
//!   currency keys, its digests and the supplied-bundle size check each record
//!   the link's TARGET, as Core ML loads it. Hugging Face's cache is why: a
//!   snapshot's files are links into the cache's `blobs`, outside the
//!   snapshot, so a replaced blob is a replaced file.
//! - **A broken link and a directory cycle are refused by name**, as is a
//!   tree of more than [`MAX_TREE_ENTRIES`] files and directories, before any
//!   file's bytes are read. A socket, a FIFO or a device is not a model file
//!   and is passed over.
//! - **Nothing writes or deletes through a link.** `prune` never follows one:
//!   it removes derived names only, and leaves a derived-named link alone.
//!   Every other write takes a name no file holds.
//!
//! Every directory walk of the kit is [`list`], and every look at one path
//! is [`resolve`].

use std::{
  fs::{self, File, Metadata},
  io::Read,
  os::unix::fs::MetadataExt,
  path::{Path, PathBuf},
};

use super::super::{
  alloc,
  error::{
    AllocationSite, Error, IoFailure, IoOperation, Result, TraversalDefect, TraversalRefusal,
  },
};

/// The most files and directories one walked tree may hold. The kit's trees
/// hold three or four files each, so the cap bounds the stamp's maps and the
/// walk's lists by a constant without refusing a real tree.
pub(crate) const MAX_TREE_ENTRIES: usize = 1024;

/// An `operation` on `path` that failed.
pub(crate) fn io(operation: IoOperation, path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
  move |e| Error::Io(IoFailure::new(operation, path.to_path_buf(), e))
}

fn refusal(defect: TraversalDefect, path: &Path) -> Error {
  Error::Traversal(TraversalRefusal::new(defect, path.to_path_buf()))
}

/// What a path resolves to, its links followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
  Dir,
  File,
  /// A socket, a FIFO or a device.
  Other,
}

/// A path resolved by the rule: its kind and its TARGET's metadata.
#[derive(Debug)]
pub(crate) struct Resolved {
  pub(crate) kind: Kind,
  pub(crate) metadata: Metadata,
}

impl Resolved {
  pub(crate) fn is_dir(&self) -> bool {
    self.kind == Kind::Dir
  }

  pub(crate) fn is_file(&self) -> bool {
    self.kind == Kind::File
  }
}

/// What is at `path`, its links followed wherever they point; `None` when
/// nothing is there.
///
/// # Errors
/// [`Error::Traversal`] naming a broken link; [`Error::Io`] for another
/// failed stat.
pub(crate) fn resolve(path: &Path) -> Result<Option<Resolved>> {
  resolve_or(path, io(IoOperation::Stat, path))
}

/// [`resolve`], with another failed stat answered by `failed`.
///
/// # Errors
/// [`Error::Traversal`] naming a broken link; `failed`'s error otherwise.
pub(crate) fn resolve_or(
  path: &Path,
  failed: impl FnOnce(std::io::Error) -> Error,
) -> Result<Option<Resolved>> {
  match fs::metadata(path) {
    Ok(metadata) => {
      let kind = if metadata.is_dir() {
        Kind::Dir
      } else if metadata.is_file() {
        Kind::File
      } else {
        Kind::Other
      };
      Ok(Some(Resolved { kind, metadata }))
    }
    Err(error) => match fs::symlink_metadata(path) {
      Ok(entry) if entry.file_type().is_symlink() && leads_nowhere(&error) => {
        Err(refusal(TraversalDefect::BrokenLink, path))
      }
      Err(missing) if missing.kind() == std::io::ErrorKind::NotFound => Ok(None),
      _ => Err(failed(error)),
    },
  }
}

/// Whether a failure to follow a link means it leads to nothing: no target,
/// a target under a file, or a chain of links that never ends.
fn leads_nowhere(error: &std::io::Error) -> bool {
  matches!(
    error.kind(),
    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
  ) || error.raw_os_error() == Some(libc::ELOOP)
}

/// One entry of a listed tree: its path as joined under the tree's root, its
/// kind, and its TARGET's metadata.
#[derive(Debug)]
pub(crate) struct Entry {
  pub(crate) path: PathBuf,
  pub(crate) kind: Kind,
  pub(crate) metadata: Metadata,
}

/// The tree under the directory `root`: every directory and regular file in
/// it, by the rule, depth first, each directory before what it holds. Reads
/// no file's bytes.
///
/// # Errors
/// [`Error::Traversal`] for a broken link, a directory cycle, or a tree of
/// more than [`MAX_TREE_ENTRIES`] entries (naming `root`);
/// [`Error::Allocation`] at [`AllocationSite::StampDigests`] if a list
/// cannot grow; [`Error::Io`] for a failed listing or stat.
pub(crate) fn list(root: &Path) -> Result<Vec<Entry>> {
  let identity = |metadata: &Metadata| (metadata.dev(), metadata.ino());
  let top = match resolve(root)? {
    Some(top) if top.is_dir() => top,
    _ => {
      return Err(io(IoOperation::Read, root)(
        std::io::ErrorKind::NotADirectory.into(),
      ));
    }
  };
  let mut out: Vec<Entry> = Vec::new();
  let mut pending: Vec<(PathBuf, usize, (u64, u64))> = Vec::new();
  let mut ancestors: Vec<(u64, u64)> = Vec::new();
  alloc::reserve(&mut pending, 1, AllocationSite::StampDigests)?;
  pending.push((root.to_path_buf(), 0, identity(&top.metadata)));
  while let Some((dir, depth, id)) = pending.pop() {
    // Depth first: the chain above `dir` is the first `depth` ancestors.
    ancestors.truncate(depth);
    alloc::reserve(&mut ancestors, 1, AllocationSite::StampDigests)?;
    ancestors.push(id);
    for entry in fs::read_dir(&dir).map_err(io(IoOperation::Read, &dir))? {
      let path = entry.map_err(io(IoOperation::Read, &dir))?.path();
      let Some(found) = resolve(&path)? else {
        // Gone since the listing.
        continue;
      };
      if found.kind == Kind::Other {
        continue;
      }
      if out.len() == MAX_TREE_ENTRIES {
        return Err(refusal(TraversalDefect::TooManyEntries, root));
      }
      if found.is_dir() {
        let child = identity(&found.metadata);
        if ancestors.contains(&child) {
          return Err(refusal(TraversalDefect::Cycle, &path));
        }
        alloc::reserve(&mut pending, 1, AllocationSite::StampDigests)?;
        pending.push((path.clone(), depth + 1, child));
      }
      alloc::reserve(&mut out, 1, AllocationSite::StampDigests)?;
      out.push(Entry {
        path,
        kind: found.kind,
        metadata: found.metadata,
      });
    }
  }
  Ok(out)
}

/// The document at `path`, read whole: its links followed by the rule, its
/// size held to `cap` before a byte is read, and its buffer reserved through
/// the door at [`AllocationSite::ArtifactRead`].
///
/// # Errors
/// `over(size)` for a document past `cap`; [`Error::Traversal`] for a broken
/// link; [`Error::Allocation`] if the buffer cannot be reserved; `failed`
/// for a missing document or a failed read.
pub(crate) fn read_capped(
  path: &Path,
  cap: u64,
  over: impl FnOnce(u64) -> Error,
  failed: impl Fn(std::io::Error) -> Error,
) -> Result<Vec<u8>> {
  let size = match resolve_or(path, &failed)? {
    Some(found) => found.metadata.len(),
    None => return Err(failed(std::io::ErrorKind::NotFound.into())),
  };
  if size > cap {
    return Err(over(size));
  }
  // `size` is at most `cap`, which every caller sets far below `usize::MAX`.
  let mut bytes = alloc::with_capacity(size as usize, AllocationSite::ArtifactRead)?;
  File::open(path)
    .and_then(|file| file.take(cap.saturating_add(1)).read_to_end(&mut bytes))
    .map_err(&failed)?;
  // A document that grew past the cap since its stat is refused all the same.
  if bytes.len() as u64 > cap {
    return Err(over(bytes.len() as u64));
  }
  Ok(bytes)
}
