//! A model directory's tree and its two doors: the resolve door every look at
//! a path goes through, and the read door every artifact byte the kit reads
//! goes through.
//!
//! **The rule.**
//! - **Reading follows every link, wherever it points.** The stamp's
//!   currency keys, its digests and the supplied-bundle size check each record
//!   the link's TARGET, as Core ML loads it. Hugging Face's cache is why: a
//!   snapshot's files are links into the cache's `blobs`, outside the
//!   snapshot, so a replaced blob is a replaced file.
//! - **A broken link and a directory cycle are refused by name**, as is a
//!   tree of more than [`MAX_TREE_ENTRIES`] files and directories, before any
//!   file's bytes are read. A broken link anywhere on a path is named, the
//!   path's own last component or one of its directories. A socket, a FIFO or
//!   a device is not a model file and is passed over.
//! - **Nothing writes or deletes through a link.** `prune` never follows one:
//!   it removes derived names only, and leaves a derived-named link alone.
//!   Every other write takes a name no file holds.
//!
//! **The resolve door** is [`resolve`] and [`resolve_or`]; every directory
//! walk is [`list`], which looks at each entry through it, and every stamped
//! path the kit opens (a mapped table or image, a bundle Core ML loads) is
//! resolved through it first. Only a path with nothing at it, and no broken
//! link on the way, is absent.
//!
//! **The read door** is [`open`], [`read`] and [`stream`]. A document is
//! resolved through the rule, its size held to its cap before a byte is read,
//! and then either read whole into a buffer reserved through `alloc/` at
//! `min(size, cap) + 1` bytes, which the read never grows, or streamed through
//! a fixed block. A document that grows while it is read is refused at its
//! cap. Every refusal is named.

use std::{
  fs::{self, File, Metadata},
  io::{ErrorKind, Read},
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

/// The block a document is streamed through: 1 MiB, whatever its size.
const BLOCK: usize = 1 << 20;

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
  fn of(metadata: Metadata) -> Self {
    let kind = if metadata.is_dir() {
      Kind::Dir
    } else if metadata.is_file() {
      Kind::File
    } else {
      Kind::Other
    };
    Self { kind, metadata }
  }

  pub(crate) fn is_dir(&self) -> bool {
    self.kind == Kind::Dir
  }

  pub(crate) fn is_file(&self) -> bool {
    self.kind == Kind::File
  }
}

/// What stops a look at a path.
enum Stop {
  /// A broken link on the way: the path itself or one of its directories.
  Broken(PathBuf),
  /// The operating system's failure.
  Failed(std::io::Error),
}

/// Whether a failure to follow a link means it leads to nothing: no target,
/// a target under a file, or a chain of links that never ends.
fn leads_nowhere(error: &std::io::Error) -> bool {
  matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory)
    || error.raw_os_error() == Some(libc::ELOOP)
}

/// Whether `path` is itself a link that leads to nothing.
fn is_broken_link(path: &Path) -> bool {
  fs::symlink_metadata(path).is_ok_and(|entry| entry.file_type().is_symlink())
    && fs::metadata(path).is_err_and(|error| leads_nowhere(&error))
}

/// What is at `path`, by the rule: `None` only when nothing is there and no
/// link on the way is broken.
fn lookup(path: &Path) -> core::result::Result<Option<Resolved>, Stop> {
  match fs::metadata(path) {
    Ok(metadata) => Ok(Some(Resolved::of(metadata))),
    Err(error) => {
      if leads_nowhere(&error) {
        if let Some(link) = path.ancestors().find(|step| is_broken_link(step)) {
          return Err(Stop::Broken(link.to_path_buf()));
        }
        if error.kind() == ErrorKind::NotFound {
          return Ok(None);
        }
      }
      Err(Stop::Failed(error))
    }
  }
}

/// The resolve door: what is at `path`, its links followed wherever they
/// point; `None` when nothing is there.
///
/// # Errors
/// [`Error::Traversal`] naming a broken link on the path; [`Error::Io`] for
/// another failed stat.
pub(crate) fn resolve(path: &Path) -> Result<Option<Resolved>> {
  resolve_or(path, io(IoOperation::Stat, path))
}

/// [`resolve`], with another failed stat answered by `failed`.
///
/// # Errors
/// [`Error::Traversal`] naming a broken link on the path; `failed`'s error
/// otherwise.
pub(crate) fn resolve_or(
  path: &Path,
  failed: impl FnOnce(std::io::Error) -> Error,
) -> Result<Option<Resolved>> {
  lookup(path).map_err(|stop| match stop {
    Stop::Broken(link) => refusal(TraversalDefect::BrokenLink, &link),
    Stop::Failed(error) => failed(error),
  })
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
      return Err(io(IoOperation::Read, root)(ErrorKind::NotADirectory.into()));
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

/// A document the read door held past its cap: at its size, or grown while
/// it was read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PastCap {
  at_least: u64,
  held: u64,
  cap: u64,
}

impl PastCap {
  /// A document of `size` bytes against `cap`, at its size.
  pub(crate) const fn at_size(size: u64, cap: u64) -> Self {
    Self {
      at_least: size,
      held: size,
      cap,
    }
  }

  /// What the refusal says of the document named `what`: past its cap, or
  /// grown while it was read.
  pub(crate) fn sentence(&self, what: &str) -> String {
    if self.at_least > self.cap {
      format!(
        "{what} holds at least {} bytes, past its {}-byte cap",
        self.at_least, self.cap
      )
    } else {
      format!(
        "{what} grew past the {} bytes it held when its size was taken, while it was read",
        self.held
      )
    }
  }
}

/// Why the read door refused, for its caller to name in its own terms.
#[derive(Debug)]
pub(crate) enum Refusal {
  /// Nothing is at the path: the one refusal a caller may read as absence.
  Absent,
  /// The document is past its cap, at its size or while it was read.
  PastCap(PastCap),
  /// A refusal the kit already names: a broken link, a refused reservation.
  Named(Error),
  /// The operating system's failure to stat, open or read it.
  Failed(std::io::Error),
}

/// An artifact the read door opened.
#[derive(Debug)]
pub(crate) struct Opened {
  pub(crate) file: File,
  pub(crate) metadata: Metadata,
}

/// The test seam: a hook run between the read door's stat and its open, so a
/// law can change a document while it is read.
#[cfg(test)]
pub(crate) mod after_size {
  use std::{cell::RefCell, path::Path};

  type Hook = Box<dyn FnMut(&Path)>;

  std::thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
  }

  /// Runs `hook` after each document's size is taken, until the guard drops.
  #[must_use = "the seam clears when the guard drops"]
  pub(crate) fn set(hook: impl FnMut(&Path) + 'static) -> Guard {
    HOOK.with(|cell| *cell.borrow_mut() = Some(Box::new(hook)));
    Guard(())
  }

  /// Clears the seam when dropped.
  pub(crate) struct Guard(());

  impl Drop for Guard {
    fn drop(&mut self) {
      HOOK.with(|cell| *cell.borrow_mut() = None);
    }
  }

  pub(crate) fn run(path: &Path) {
    HOOK.with(|cell| {
      if let Some(hook) = cell.borrow_mut().as_mut() {
        hook(path);
      }
    });
  }
}

#[cfg(not(test))]
mod after_size {
  #[inline(always)]
  pub(super) fn run(_: &std::path::Path) {}
}

/// The read door's first step: `path` resolved by the rule, its target a
/// regular file whose size is held to `cap` before a byte is read, and
/// opened.
///
/// # Errors
/// [`Refusal::Absent`] when nothing is there; [`Refusal::PastCap`] for a file
/// past `cap`; [`Refusal::Named`] for a broken link; [`Refusal::Failed`] for
/// a failed stat or open, or a path that is not a regular file.
pub(crate) fn open(path: &Path, cap: Option<u64>) -> core::result::Result<Opened, Refusal> {
  let found = match lookup(path) {
    Ok(Some(found)) => found,
    Ok(None) => return Err(Refusal::Absent),
    Err(Stop::Broken(link)) => {
      return Err(Refusal::Named(refusal(TraversalDefect::BrokenLink, &link)));
    }
    Err(Stop::Failed(error)) => return Err(Refusal::Failed(error)),
  };
  if !found.is_file() {
    return Err(Refusal::Failed(std::io::Error::other("not a regular file")));
  }
  let size = found.metadata.len();
  if let Some(cap) = cap
    && size > cap
  {
    return Err(Refusal::PastCap(PastCap::at_size(size, cap)));
  }
  after_size::run(path);
  let file = File::open(path).map_err(Refusal::Failed)?;
  Ok(Opened {
    file,
    metadata: found.metadata,
  })
}

/// Reads past a read's end, without holding a byte, to say how far a
/// document now runs: up to `cap + 1` bytes from `from`.
fn count_on(file: &mut File, from: u64, cap: u64) -> core::result::Result<u64, Refusal> {
  let mut seen = from;
  let mut block = [0u8; 4096];
  while seen <= cap {
    match file.read(&mut block) {
      Ok(0) => break,
      Ok(read) => seen += read as u64,
      Err(error) if error.kind() == ErrorKind::Interrupted => {}
      Err(error) => return Err(Refusal::Failed(error)),
    }
  }
  Ok(seen)
}

/// The read door's whole road: the document at `path`, at most `cap` bytes,
/// read into a buffer reserved through `alloc/` at `site` for `min(size, cap)
/// + 1` bytes. The read fills that buffer and never grows it: a document that
/// fills it has grown since its size was taken, and is refused at its cap.
///
/// # Errors
/// As [`open`]; [`Refusal::Named`] carrying [`Error::Allocation`] if the
/// buffer cannot be reserved; [`Refusal::PastCap`] for a document that grows
/// while it is read; [`Refusal::Failed`] for a failed read.
pub(crate) fn read(
  path: &Path,
  cap: u64,
  site: AllocationSite,
) -> core::result::Result<Vec<u8>, Refusal> {
  let Opened { mut file, metadata } = open(path, Some(cap))?;
  let held = metadata.len();
  // `held` is at most `cap`, and every cap is far below `usize::MAX`.
  let room = held as usize + 1;
  let mut bytes = alloc::with_capacity::<u8>(room, site).map_err(Refusal::Named)?;
  bytes.resize(room, 0);
  let mut filled = 0;
  while filled < room {
    match file.read(&mut bytes[filled..]) {
      Ok(0) => break,
      Ok(read) => filled += read,
      Err(error) if error.kind() == ErrorKind::Interrupted => {}
      Err(error) => return Err(Refusal::Failed(error)),
    }
  }
  if filled == room {
    let at_least = count_on(&mut file, filled as u64, cap)?;
    return Err(Refusal::PastCap(PastCap {
      at_least,
      held,
      cap,
    }));
  }
  bytes.truncate(filled);
  Ok(bytes)
}

/// The read door's streamed road: the file at `path` fed to `each` through a
/// fixed [`BLOCK`], never held whole. A file read past its size, or past
/// `cap`, is refused before `each` sees a byte beyond it. Returns the
/// metadata its size was taken from.
///
/// # Errors
/// As [`open`]; [`Refusal::PastCap`] for a file that grows while it is read
/// (named against `cap`, or against no cap at all); [`Refusal::Failed`] for a
/// failed read.
pub(crate) fn stream(
  path: &Path,
  cap: Option<u64>,
  mut each: impl FnMut(&[u8]),
) -> core::result::Result<Metadata, Refusal> {
  let Opened { mut file, metadata } = open(path, cap)?;
  let held = metadata.len();
  let mut block = vec![0u8; BLOCK];
  let mut seen = 0u64;
  loop {
    let read = match file.read(&mut block) {
      Ok(0) => break,
      Ok(read) => read,
      Err(error) if error.kind() == ErrorKind::Interrupted => continue,
      Err(error) => return Err(Refusal::Failed(error)),
    };
    seen += read as u64;
    if seen > held {
      return Err(Refusal::PastCap(PastCap {
        at_least: seen,
        held,
        cap: cap.unwrap_or(u64::MAX),
      }));
    }
    each(&block[..read]);
  }
  Ok(metadata)
}

#[cfg(test)]
mod tests;
