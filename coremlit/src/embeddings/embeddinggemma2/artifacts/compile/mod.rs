//! `compile_artifacts` and `prune`: the one writer into a model directory,
//! and the one explicit removal.
//!
//! **The writer's invariant, which the shared maps' SAFETY argument rests on.**
//! No byte of a source, or of a file this module made, is ever written in
//! place. Every artifact is written under a temporary name and renamed to a
//! content-addressed name no file held; an existing name is kept, never
//! replaced. The one name replaced is the stamp's, and the stamp is never
//! mapped. `prune` only unlinks derived names. So a process that maps a table
//! or an image keeps reading the inode it mapped, unchanged, until it drops the
//! map.
//!
//! **Every read follows the tree module's one rule** (see `artifacts::tree`):
//! each link is followed wherever it points and its TARGET is recorded, as
//! Core ML loads it, because Hugging Face's cache lays a snapshot out as
//! links into its `blobs`; a broken link, a directory cycle and a tree past
//! its cap are refused by name; nothing writes or deletes through a link.

use std::{
  collections::BTreeMap,
  fs::{self, File},
  io::{BufWriter, Read, Write},
  os::unix::fs::OpenOptionsExt,
  path::{Path, PathBuf},
  sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

use super::{
  super::{
    alloc,
    config::{HIDDEN_SIZE, POSITION_ROWS, POSITION_WIDTH},
    error::{
      AllocationSite, AmbiguousArtifact, Artifact, CompileFailure, Error, ImageField, IoFailure,
      IoOperation, Result, TableCell, TokenizerImageDefect,
    },
    table::{POSITION_SENTINEL_CELLS, TOKEN_SENTINEL_CELLS},
    tokenizer::{
      compile::{TOKENIZER_JSON_CAP, compile_image, image_name, past_cap},
      image::{HEADER_SIZE, TOKENIZER_IMAGE_VERSION, header},
    },
  },
  hex, manifest_sha256,
  naming::{ID_HEX, compiled_bundle, parse_compiled_bundle, source_package},
  stamp::{
    BundleEntry, Currency, FileEntry, ImageEntry, KIT, LOOSE, Origin, PACKAGES, STAMP_FORMAT,
    STAMP_NAME, SentinelEntry, Stamp,
  },
  tree::{self, Kind, Resolved, io, resolve},
};
use crate::Model;

/// The directory's lock, taken only when there is work.
const LOCK_NAME: &str = ".coremlit-embeddinggemma2.lock";

/// The block a file is hashed through: 1 MiB, whatever the file's size.
const BLOCK: usize = 1 << 20;

/// What `compile_artifacts` did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompileReport {
  compiled: Vec<String>,
  kept: Vec<String>,
}

impl CompileReport {
  /// The artifacts it made, by name.
  pub fn compiled(&self) -> &[String] {
    &self.compiled
  }

  /// The artifacts it found current and kept, by name.
  pub fn kept(&self) -> &[String] {
    &self.kept
  }
}

/// What `prune` removed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneReport {
  removed: Vec<PathBuf>,
}

impl PruneReport {
  /// The paths it removed.
  pub fn removed(&self) -> &[PathBuf] {
    &self.removed
  }
}

/// A name no file holds: the final name with a temporary suffix.
fn temporary(final_path: &Path) -> PathBuf {
  static NONCE: AtomicU64 = AtomicU64::new(0);
  let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
  let mut name = final_path.as_os_str().to_owned();
  name.push(format!(".tmp-{}-{nonce}", std::process::id()));
  PathBuf::from(name)
}

/// Takes `dir`'s lock. A lock name that is a link is refused: nothing writes
/// through a link.
fn lock(dir: &Path) -> Result<File> {
  let lock_path = dir.join(LOCK_NAME);
  let lock = File::options()
    .create(true)
    .truncate(false)
    .write(true)
    .custom_flags(libc::O_NOFOLLOW)
    .open(&lock_path)
    .map_err(io(IoOperation::Lock, &lock_path))?;
  lock.lock().map_err(io(IoOperation::Lock, &lock_path))?;
  Ok(lock)
}

/// Feeds `each` the bytes of the file at `path`, one [`BLOCK`] at a time.
fn stream(path: &Path, mut each: impl FnMut(&[u8])) -> Result<()> {
  let mut file = File::open(path).map_err(io(IoOperation::Read, path))?;
  let mut block = vec![0u8; BLOCK];
  loop {
    let read = file.read(&mut block).map_err(io(IoOperation::Read, path))?;
    if read == 0 {
      return Ok(());
    }
    each(&block[..read]);
  }
}

fn sha256_file(path: &Path) -> Result<[u8; 32]> {
  let mut hasher = Sha256::new();
  stream(path, |bytes| hasher.update(bytes))?;
  Ok(hasher.finalize().into())
}

/// Every regular file under the directory `root`, by the tree's rule, by path
/// relative to `base`, with its target's currency key, sorted bytewise by
/// path.
fn files_under(base: &Path, root: &Path) -> Result<Vec<(String, Currency)>> {
  let entries = tree::list(root)?;
  let mut out = alloc::with_capacity(entries.len(), AllocationSite::StampDigests)?;
  for entry in &entries {
    if entry.kind == Kind::File {
      let rel = entry
        .path
        .strip_prefix(base)
        .unwrap_or(&entry.path)
        .to_string_lossy()
        .into_owned();
      out.push((rel, Currency::of(&entry.metadata)));
    }
  }
  out.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
  Ok(out)
}

/// Whether `path` is a directory, its links followed.
fn is_dir(path: &Path) -> Result<bool> {
  Ok(resolve(path)?.is_some_and(|found| found.is_dir()))
}

/// The currency key of every source in `dir`: each package's files, each
/// plainly named bundle's files, and each loose file present, by relative
/// path, each link's target's. A supplied bundle is a source like a package:
/// replacing it, or a target of a link in it, moves its files' keys, so the
/// stamp that named the old one stops being current.
fn sources(dir: &Path) -> Result<BTreeMap<String, Currency>> {
  let mut out = BTreeMap::new();
  for package in &PACKAGES {
    for root in [
      dir.join(source_package(package.stem)),
      dir.join(format!("{}.mlmodelc", package.stem)),
    ] {
      if is_dir(&root)? {
        out.extend(files_under(dir, &root)?);
      }
    }
  }
  for (rel, _) in &LOOSE {
    if let Some(found) = resolve(&dir.join(rel))?.filter(Resolved::is_file) {
      out.insert((*rel).to_owned(), Currency::of(&found.metadata));
    }
  }
  Ok(out)
}

/// Whether the stamp names exactly the sources present, each with its current
/// key, and every artifact it names exists. Creates nothing. The sources
/// include a supplied bundle's files, so a bundle replaced since the stamp,
/// in place or by a new directory, is not current.
fn current(dir: &Path) -> Result<Option<Stamp>> {
  let Ok(stamp) = Stamp::read(dir) else {
    return Ok(None);
  };
  let present = sources(dir)?;
  let sources_unchanged = present.len() == stamp.files.len()
    && present
      .iter()
      .zip(&stamp.files)
      .all(|((rel, currency), (stamped, entry))| rel == stamped && *currency == entry.currency);
  let mut derived_exist = true;
  for bundle in stamp.bundles.values() {
    derived_exist &= is_dir(&dir.join(&bundle.name))?;
  }
  if let Some(image) = &stamp.tokenizer_image {
    derived_exist &= resolve(&dir.join(&image.name))?.is_some_and(|found| found.is_file());
  }
  let mut supplied_unchanged = true;
  for package in &PACKAGES {
    let plain = is_dir(&dir.join(format!("{}.mlmodelc", package.stem)))?;
    let stamped_supplied = stamp
      .bundles
      .get(package.key)
      .is_some_and(|b| b.origin == Origin::Supplied);
    supplied_unchanged &= plain == stamped_supplied;
  }
  Ok((sources_unchanged && derived_exist && supplied_unchanged).then_some(stamp))
}

/// Compiles and stamps the model directory `dir`: every package to a
/// content-addressed bundle, `tokenizer.json` to a content-addressed tokenizer
/// image, and each source's digest, currency key and sentinel cells into the
/// stamp. Idempotent: a current directory is left as it is, with no file
/// created and no lock taken, so a stamped read-only directory passes.
///
/// It deletes nothing, moves nothing and overwrites nothing: a new artifact
/// takes a new name, a superseded one stays until [`prune`], and only the
/// stamp's name is replaced. It follows every link in the directory,
/// wherever it points, and stamps each link's target: Hugging Face's cache
/// lays a snapshot out as links into its `blobs`.
///
/// # Errors
/// [`Error::AmbiguousArtifact`] when a package sits beside a caller's plainly
/// named bundle of the same artifact; [`Error::Compile`] when Core ML cannot
/// compile a package; [`Error::TokenizerSource`] for a `tokenizer.json`
/// outside the configuration the kit implements, or past its 64 MiB cap
/// (refused before a byte of it is read); [`Error::Traversal`] for a broken
/// link, a directory cycle or a tree of more than 1024 entries;
/// [`Error::Allocation`] if a list or the tokenizer image cannot be reserved,
/// with nothing staged; [`Error::Io`] for a failed file-system operation.
pub fn compile_artifacts(dir: impl AsRef<Path>) -> Result<CompileReport> {
  let dir = dir.as_ref();
  if let Some(stamp) = current(dir)? {
    return kept_report(&stamp);
  }
  let _lock = lock(dir)?;
  if let Some(stamp) = current(dir)? {
    return kept_report(&stamp);
  }
  build(dir)
}

fn kept_report(stamp: &Stamp) -> Result<CompileReport> {
  let mut kept = alloc::with_capacity(stamp.bundles.len() + 1, AllocationSite::StampDigests)?;
  kept.extend(stamp.bundles.values().map(|b| b.name.clone()));
  kept.extend(stamp.tokenizer_image.as_ref().map(|i| i.name.clone()));
  Ok(CompileReport {
    compiled: Vec::new(),
    kept,
  })
}

/// The tokenizer image `build` stamps: one in place whose own digest still
/// verifies, or one built in memory, written once the packages are staged.
enum Image {
  Kept(ImageEntry),
  Built(ImageEntry, Vec<u8>),
}

fn build(dir: &Path) -> Result<CompileReport> {
  let previous = Stamp::read(dir).ok();
  // `tokenizer.json` is held to its cap before a byte of it is read.
  let json_size = resolve(&dir.join("tokenizer.json"))?
    .filter(Resolved::is_file)
    .map(|found| found.metadata.len());
  if let Some(size) = json_size.filter(|&size| size > TOKENIZER_JSON_CAP) {
    return Err(past_cap(size));
  }
  let mut report = CompileReport::default();
  let mut files = BTreeMap::new();
  // The loose files and the tokenizer image first, the image built in
  // memory, so that a refusal of either stages nothing.
  for (rel, artifact) in &LOOSE {
    let path = dir.join(rel);
    let Some(found) = resolve(&path)?.filter(Resolved::is_file) else {
      continue;
    };
    let currency = Currency::of(&found.metadata);
    let reused = previous
      .as_ref()
      .and_then(|stamp| stamp.files.get(*rel))
      .filter(|entry| entry.currency == currency)
      .cloned();
    let entry = match reused {
      Some(entry) => entry,
      None => {
        let sha256 = hex(&sha256_file(&path)?);
        let sentinels = match artifact {
          Artifact::TokenTable => read_sentinels(&path, &TOKEN_SENTINEL_CELLS, |c| {
            (u64::from(c.row()) * HIDDEN_SIZE as u64 + u64::from(c.column())) * 2
          })?,
          Artifact::PositionTable => read_sentinels(&path, &POSITION_SENTINEL_CELLS, |c| {
            let row = u64::from(c.table().unwrap_or(0)) * POSITION_ROWS as u64 + u64::from(c.row());
            (row * POSITION_WIDTH as u64 + u64::from(c.column())) * 2
          })?,
          _ => Vec::new(),
        };
        FileEntry {
          sha256,
          currency,
          sentinels,
        }
      }
    };
    files.insert((*rel).to_owned(), entry);
  }
  let image = match files.get("tokenizer.json") {
    Some(source) => Some(tokenizer_image(dir, &source.sha256)?),
    None => None,
  };
  let mut bundles = BTreeMap::new();
  for package in &PACKAGES {
    let source = dir.join(source_package(package.stem));
    let plain = dir.join(format!("{}.mlmodelc", package.stem));
    match (is_dir(&source)?, is_dir(&plain)?) {
      (true, true) => {
        return Err(Error::AmbiguousArtifact(AmbiguousArtifact::new(
          package.artifact,
          vec![source, plain],
        )));
      }
      (true, false) => {
        let digests = digest_files(dir, &source, &mut files)?;
        let lines = digests
          .iter()
          .map(|(rel, sha)| (rel.as_str(), sha.as_str()));
        let name = compiled_bundle(package.stem, &hex(&manifest_sha256(lines))[..ID_HEX]);
        let final_path = dir.join(&name);
        if is_dir(&final_path)? {
          report.kept.push(name.clone());
        } else {
          compile_package(package.artifact, &source, &final_path)?;
          report.compiled.push(name.clone());
        }
        bundles.insert(
          package.key.to_owned(),
          BundleEntry {
            name,
            origin: Origin::Compiled,
            digests,
          },
        );
      }
      (false, true) => {
        let digests = digest_files(dir, &plain, &mut files)?;
        report.kept.push(format!("{}.mlmodelc", package.stem));
        bundles.insert(
          package.key.to_owned(),
          BundleEntry {
            name: format!("{}.mlmodelc", package.stem),
            origin: Origin::Supplied,
            digests,
          },
        );
      }
      (false, false) => {}
    }
  }
  let tokenizer_image = match image {
    Some(Image::Kept(entry)) => {
      report.kept.push(entry.name.clone());
      Some(entry)
    }
    Some(Image::Built(entry, bytes)) => {
      write_new(&dir.join(&entry.name), false, |out| out.write_all(&bytes))?;
      report.compiled.push(entry.name.clone());
      Some(entry)
    }
    None => None,
  };
  let stamp = Stamp {
    format: STAMP_FORMAT,
    kit: KIT.to_owned(),
    coremlit: env!("CARGO_PKG_VERSION").to_owned(),
    os_build: os_build(),
    files,
    bundles,
    tokenizer_image,
  };
  write_new(&dir.join(STAMP_NAME), true, |out| {
    serde_json::to_writer_pretty(out, &stamp).map_err(std::io::Error::other)
  })?;
  Ok(report)
}

/// Hashes every file under `root` into `files` (with its currency key) and
/// returns the digests by relative path.
fn digest_files(
  dir: &Path,
  root: &Path,
  files: &mut BTreeMap<String, FileEntry>,
) -> Result<BTreeMap<String, String>> {
  let mut digests = BTreeMap::new();
  for (rel, currency) in files_under(dir, root)? {
    let sha256 = hex(&sha256_file(&dir.join(&rel))?);
    digests.insert(rel.clone(), sha256.clone());
    files.insert(
      rel,
      FileEntry {
        sha256,
        currency,
        sentinels: Vec::new(),
      },
    );
  }
  Ok(digests)
}

fn read_sentinels(
  path: &Path,
  cells: &[TableCell],
  offset: impl Fn(TableCell) -> u64,
) -> Result<Vec<SentinelEntry>> {
  use std::io::{Seek, SeekFrom};
  let mut file = File::open(path).map_err(io(IoOperation::Read, path))?;
  cells
    .iter()
    .map(|&cell| {
      let mut pair = [0u8; 2];
      file
        .seek(SeekFrom::Start(offset(cell)))
        .map_err(io(IoOperation::Read, path))?;
      file
        .read_exact(&mut pair)
        .map_err(io(IoOperation::Read, path))?;
      Ok(SentinelEntry {
        table: cell.table(),
        row: cell.row(),
        column: cell.column(),
        bits: u16::from_le_bytes(pair),
      })
    })
    .collect()
}

/// Whether the image at `path` holds its own digest (the header's record of
/// the SHA-256 of what follows the header), and the SHA-256 of all its bytes:
/// streamed, so no buffer holds the image.
fn check_image(path: &Path) -> Result<(bool, [u8; 32])> {
  let mut head = [0u8; HEADER_SIZE];
  let mut held = 0;
  let (mut whole, mut body) = (Sha256::new(), Sha256::new());
  stream(path, |bytes| {
    whole.update(bytes);
    let into_head = (HEADER_SIZE - held).min(bytes.len());
    head[held..held + into_head].copy_from_slice(&bytes[..into_head]);
    held += into_head;
    body.update(&bytes[into_head..]);
  })?;
  let holds = held == HEADER_SIZE
    && head[header::IMAGE_DIGEST..header::IMAGE_DIGEST + 32] == body.finalize()[..];
  Ok((holds, whole.finalize().into()))
}

/// The tokenizer image of the `tokenizer.json` whose digest is `source`: kept
/// when it is in place and its own digest still verifies, built in memory
/// under its content-addressed name otherwise.
fn tokenizer_image(dir: &Path, source: &str) -> Result<Image> {
  let source_bytes: [u8; 32] = {
    let mut out = [0u8; 32];
    for (i, pair) in source
      .as_bytes()
      .as_chunks::<2>()
      .0
      .iter()
      .enumerate()
      .take(32)
    {
      out[i] = u8::from_str_radix(core::str::from_utf8(pair).unwrap_or("00"), 16).unwrap_or(0);
    }
    out
  };
  let name = image_name(&source_bytes);
  let final_path = dir.join(&name);
  if resolve(&final_path)?.is_some_and(|found| found.is_file()) {
    let (holds_its_digest, image_sha256) = check_image(&final_path)?;
    if holds_its_digest {
      return Ok(Image::Kept(ImageEntry {
        name,
        format: TOKENIZER_IMAGE_VERSION,
        source_sha256: source.to_owned(),
        image_sha256: hex(&image_sha256),
      }));
    }
    return Err(Error::TokenizerImage(TokenizerImageDefect::new(
      ImageField::ImageDigest,
      "the image's own digest".to_owned(),
      format!("`{name}` holds another; remove it, it is never replaced"),
    )));
  }
  let json_path = dir.join("tokenizer.json");
  let json = tree::read_capped(&json_path, TOKENIZER_JSON_CAP, past_cap, |e| {
    Error::Io(IoFailure::new(IoOperation::Read, json_path.clone(), e))
  })?;
  let image = compile_image(&json)?;
  let image_sha256 = hex(&Sha256::digest(&image).into());
  Ok(Image::Built(
    ImageEntry {
      name,
      format: TOKENIZER_IMAGE_VERSION,
      source_sha256: source.to_owned(),
      image_sha256,
    },
    image,
  ))
}

/// Writes through `write` to a temporary name, syncs it, and renames it to
/// `final_path`. Only the stamp (`replace`) may take a name a file holds, a
/// link's included; the rename replaces the name itself and never writes
/// through a link.
fn write_new(
  final_path: &Path,
  replace: bool,
  write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> Result<()> {
  if !replace && fs::symlink_metadata(final_path).is_ok() {
    return Err(Error::Io(IoFailure::new(
      IoOperation::Write,
      final_path.to_path_buf(),
      std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "an artifact's name is never replaced",
      ),
    )));
  }
  let tmp = temporary(final_path);
  let file = File::options()
    .write(true)
    .create_new(true)
    .open(&tmp)
    .map_err(io(IoOperation::Write, &tmp))?;
  let mut out = BufWriter::new(file);
  write(&mut out).map_err(io(IoOperation::Write, &tmp))?;
  let file = out.into_inner().map_err(|e| {
    Error::Io(IoFailure::new(
      IoOperation::Write,
      tmp.clone(),
      e.into_error(),
    ))
  })?;
  file.sync_all().map_err(io(IoOperation::Sync, &tmp))?;
  fs::rename(&tmp, final_path).map_err(io(IoOperation::Rename, final_path))
}

/// Compiles `source` with Core ML and moves the result into place: a copy of
/// Core ML's temporary bundle under a temporary name, synced, then renamed to
/// `final_path`, a name no file holds.
fn compile_package(artifact: Artifact, source: &Path, final_path: &Path) -> Result<()> {
  let compiled =
    Model::compile(source).map_err(|e| Error::Compile(CompileFailure::new(artifact, e)))?;
  let tmp = temporary(final_path);
  copy_tree(&compiled, &tmp)?;
  fs::rename(&tmp, final_path).map_err(io(IoOperation::Rename, final_path))?;
  let _ = fs::remove_dir_all(&compiled);
  Ok(())
}

/// Copies the tree under `from`, Core ML's compiled bundle, into the new
/// directory `to`, by the tree's rule: a link in it is copied as its target's
/// bytes.
fn copy_tree(from: &Path, to: &Path) -> Result<()> {
  let entries = tree::list(from)?;
  fs::create_dir(to).map_err(io(IoOperation::Write, to))?;
  for entry in &entries {
    let target = to.join(
      entry
        .path
        .strip_prefix(from)
        .expect("a listed path lies under its root"),
    );
    if entry.kind == Kind::Dir {
      fs::create_dir(&target).map_err(io(IoOperation::Write, &target))?;
    } else {
      fs::copy(&entry.path, &target).map_err(io(IoOperation::Copy, &entry.path))?;
      File::open(&target)
        .and_then(|f| f.sync_all())
        .map_err(io(IoOperation::Sync, &target))?;
    }
  }
  Ok(())
}

/// The OS build, for provenance only.
fn os_build() -> String {
  std::process::Command::new("sw_vers")
    .arg("-buildVersion")
    .output()
    .ok()
    .and_then(|out| String::from_utf8(out.stdout).ok())
    .map(|s| s.trim().to_owned())
    .unwrap_or_default()
}

/// Whether `name` is one of the kit's derived names: a content-addressed
/// bundle of one of the kit's packages, a tokenizer image, or a temporary file.
fn derived(name: &str) -> bool {
  let hex12 = |s: &str| {
    s.len() == ID_HEX
      && s
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
  };
  let bundle = parse_compiled_bundle(name)
    .is_some_and(|(stem, _)| PACKAGES.iter().any(|package| package.stem == stem));
  let image = name
    .strip_prefix("tokenizer.v")
    .and_then(|rest| rest.strip_suffix(".gemmabpe"))
    .and_then(|rest| rest.split_once('-'))
    .is_some_and(|(version, digest)| {
      !version.is_empty() && version.bytes().all(|b| b.is_ascii_digit()) && hex12(digest)
    });
  bundle || image || name.contains(".tmp-")
}

/// Removes the derived files of `dir` that its stamp no longer names: the one
/// explicit removal. Never a source, a plainly named caller's bundle, the
/// stamp or the lock. It follows no link: a derived-named link is left as it
/// is, and so is its target. Takes the directory's lock.
///
/// # Errors
/// [`Error::Unstamped`] when `dir` has no stamp; [`Error::Allocation`] if the
/// list of removed paths cannot grow; [`Error::Io`] for a failed removal.
pub fn prune(dir: impl AsRef<Path>) -> Result<PruneReport> {
  let dir = dir.as_ref();
  let _lock = lock(dir)?;
  let stamp = Stamp::read(dir)?;
  let named = |name: &str| {
    stamp.bundles.values().any(|b| b.name == name)
      || stamp
        .tokenizer_image
        .as_ref()
        .is_some_and(|image| image.name == name)
  };
  let mut report = PruneReport::default();
  for entry in fs::read_dir(dir).map_err(io(IoOperation::Read, dir))? {
    let entry = entry.map_err(io(IoOperation::Read, dir))?;
    let file_name = entry.file_name();
    // Every derived name is ASCII.
    let Some(name) = file_name.to_str() else {
      continue;
    };
    if !derived(name) || named(name) {
      continue;
    }
    let path = entry.path();
    let kind = fs::symlink_metadata(&path)
      .map_err(io(IoOperation::Stat, &path))?
      .file_type();
    if kind.is_symlink() {
      continue;
    }
    alloc::reserve(&mut report.removed, 1, AllocationSite::StampDigests)?;
    if kind.is_dir() {
      fs::remove_dir_all(&path).map_err(io(IoOperation::Remove, &path))?;
    } else {
      fs::remove_file(&path).map_err(io(IoOperation::Remove, &path))?;
    }
    report.removed.push(path);
  }
  report.removed.sort_unstable();
  Ok(report)
}

#[cfg(test)]
mod tests;
