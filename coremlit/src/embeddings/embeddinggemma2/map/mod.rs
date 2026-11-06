//! The one read-only memory map the kit's tables and tokenizer image share.
//!
//! The token table (256 MiB), the tokenizer image and the position tables are
//! each mapped ONCE per process, and every part instance holds a clone of the
//! same [`Arc<ReadOnlyMap>`]: N instances read one mapping, and the last to
//! drop unmaps it. Mapped pages are clean and file-backed: they become
//! resident only when touched, and the kernel may drop them under pressure.

use std::{fs::File, path::Path, sync::Arc};

use super::error::{Artifact, Error, ReadFailure, Result};

/// How a map will be read, passed to the kernel as `madvise` advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Access {
  /// Rows read in no order: `MADV_RANDOM`, so a fault reads one page and no
  /// read-ahead wastes page cache.
  Random,
  /// No advice: the kernel's default read-ahead. The position tables'
  /// mode.
  Default,
}

/// A whole file, mapped read-only.
///
/// The only way to its bytes is [`Self::bytes`], a `&[u8]` every reader indexes
/// with bounds-checked safe Rust.
#[derive(Debug)]
pub(crate) struct ReadOnlyMap {
  map: memmap2::Mmap,
}

impl ReadOnlyMap {
  /// Maps the file at `path`, read-only and whole, and advises the kernel of
  /// `access`. A caller that knows the file's length checks it first, before
  /// anything is mapped.
  ///
  /// # Errors
  /// [`Error::Read`] if the file cannot be opened or the map made, or the
  /// advice is refused.
  pub(crate) fn open(artifact: Artifact, path: &Path, access: Access) -> Result<Arc<Self>> {
    let read = |e| Error::Read(ReadFailure::new(artifact, path.to_path_buf(), e));
    let file = File::open(path).map_err(read)?;
    // SAFETY: `Mmap::map` is `unsafe` because the bytes behind the returned
    // slice may change while it lives, and the compiler is entitled to treat
    // memory behind a `&[u8]` as immutable. What keeps that from happening,
    // and what is left over, in four points:
    //
    // 1. The kit never modifies a mapped inode. Only `compile_artifacts` and
    //    `prune` change a model directory, and the lock file either creates
    //    is empty, never written and never mapped. `compile_artifacts` writes
    //    every derived file under a temporary name and renames it to a
    //    content-addressed name no file held, and it never writes in place;
    //    an existing name is kept or refused, never replaced, so a rebuilt
    //    tokenizer image takes a new name rather than replacing the old one.
    //    The only name it replaces is the stamp's, and the stamp is never
    //    mapped. It never opens a source (a table, `tokenizer.json`) for
    //    writing at all. So a map keeps reading the inode it mapped,
    //    unchanged, until the last clone of the map drops; a replaced name
    //    leaves that inode alive and untouched beneath it. `prune` only
    //    unlinks derived names, and an unlinked inode under a map stays alive
    //    and untouched exactly like a renamed-over one, so a `prune` racing a
    //    running service leaves its maps sound. Law CA1 holds a running map
    //    across a concurrent regeneration and the `prune` after it, and checks
    //    it.
    // 2. Every read of a mapped byte is bounds-checked safe Rust: slicing and
    //    `get` over `bytes()`, no `get_unchecked`, no `align_to`, no pointer
    //    arithmetic. That defends against a file whose content is wrong but
    //    STABLE, a wrong table or image already in place when mapped: the
    //    sentinels catch the cells the stamp recorded, and any other
    //    difference is a wrong value or a refusal, never an access outside
    //    the map. A file that changes WHILE mapped is undefined behaviour,
    //    however the reads are checked, and belongs wholly to the residue of
    //    points 3 and 4. Its practical shape is named here and not promised:
    //    for the tables, wrong values, since no index into them is derived
    //    from mapped bytes; for the tokenizer image, whose section offsets ARE
    //    mapped bytes, possibly an out-of-bounds read.
    // 3. The trust model excludes every other writer. What remains is a
    //    non-malicious outside tool overwriting a mapped artifact IN PLACE
    //    while a process holds the map (`cp` over the file, `rsync
    //    --inplace`, an editor). That is named as accepted and unrecoverable:
    //    a wrong value, a refusal or SIGBUS. It is not claimed to be free of
    //    undefined behaviour; it is outside what this crate defends.
    // 4. Truncating a mapped file faults the next read past the new end with
    //    SIGBUS, and so does a volume that disappears under a map: the same
    //    accepted residue as point 3.
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(read)?;
    if access == Access::Random {
      map.advise(memmap2::Advice::Random).map_err(read)?;
    }
    Ok(Arc::new(Self { map }))
  }

  /// The mapped file's bytes.
  #[inline]
  pub(crate) fn bytes(&self) -> &[u8] {
    &self.map
  }
}
