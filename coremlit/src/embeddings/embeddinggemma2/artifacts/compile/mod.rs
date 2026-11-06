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

use std::{
  collections::BTreeMap,
  fs::{self, File},
  io::Write,
  path::{Path, PathBuf},
  sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

use super::{
  super::{
    config::{HIDDEN_SIZE, POSITION_ROWS, POSITION_WIDTH},
    error::{
      AmbiguousArtifact, Artifact, CompileFailure, Error, ImageField, IoFailure, IoOperation,
      Result, TableCell, TokenizerImageDefect,
    },
    table::{POSITION_SENTINEL_CELLS, TOKEN_SENTINEL_CELLS},
    tokenizer::{
      compile::{compile_image, image_name},
      image::{HEADER_SIZE, TOKENIZER_IMAGE_VERSION, header},
    },
  },
  hex, manifest_sha256,
  naming::{ID_HEX, compiled_bundle, parse_compiled_bundle, source_package},
  stamp::{
    BundleEntry, Currency, FileEntry, ImageEntry, KIT, LOOSE, Origin, PACKAGES, STAMP_FORMAT,
    STAMP_NAME, SentinelEntry, Stamp,
  },
};
use crate::Model;

/// The directory's lock, taken only when there is work.
const LOCK_NAME: &str = ".coremlit-embeddinggemma2.lock";

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

fn io(operation: IoOperation, path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
  move |e| Error::Io(IoFailure::new(operation, path.to_path_buf(), e))
}

/// A name no file holds: the final name with a temporary suffix.
fn temporary(final_path: &Path) -> PathBuf {
  static NONCE: AtomicU64 = AtomicU64::new(0);
  let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
  let mut name = final_path.as_os_str().to_owned();
  name.push(format!(".tmp-{}-{nonce}", std::process::id()));
  PathBuf::from(name)
}

fn sha256_file(path: &Path) -> Result<[u8; 32]> {
  let mut file = File::open(path).map_err(io(IoOperation::Read, path))?;
  let mut hasher = Sha256::new();
  let mut block = vec![0u8; 1 << 20];
  loop {
    let read = std::io::Read::read(&mut file, &mut block).map_err(io(IoOperation::Read, path))?;
    if read == 0 {
      break;
    }
    hasher.update(&block[..read]);
  }
  Ok(hasher.finalize().into())
}

/// Every regular file under `root`, by path relative to `base`, sorted.
fn files_under(base: &Path, root: &Path) -> Result<Vec<String>> {
  let mut out = Vec::new();
  let mut pending = vec![root.to_path_buf()];
  while let Some(dir) = pending.pop() {
    for entry in fs::read_dir(&dir).map_err(io(IoOperation::Read, &dir))? {
      let entry = entry.map_err(io(IoOperation::Read, &dir))?;
      let path = entry.path();
      let kind = entry.file_type().map_err(io(IoOperation::Stat, &path))?;
      if kind.is_dir() {
        pending.push(path);
      } else if kind.is_file() {
        let rel = path
          .strip_prefix(base)
          .unwrap_or(&path)
          .to_string_lossy()
          .into_owned();
        out.push(rel);
      }
    }
  }
  out.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
  Ok(out)
}

/// The currency key of every source in `dir`: each package's files and each
/// loose file present, by relative path.
fn sources(dir: &Path) -> Result<BTreeMap<String, Currency>> {
  let mut out = BTreeMap::new();
  let mut record = |rel: String| -> Result<()> {
    let path = dir.join(&rel);
    let metadata = fs::metadata(&path).map_err(io(IoOperation::Stat, &path))?;
    out.insert(rel, Currency::of(&metadata));
    Ok(())
  };
  for package in &PACKAGES {
    let root = dir.join(source_package(package.stem));
    if root.is_dir() {
      for rel in files_under(dir, &root)? {
        record(rel)?;
      }
    }
  }
  for (rel, _) in &LOOSE {
    if dir.join(rel).is_file() {
      record((*rel).to_owned())?;
    }
  }
  Ok(out)
}

/// Whether the stamp names exactly the sources present, each with its current
/// key, and every artifact it names exists. Creates nothing.
fn current(dir: &Path) -> Result<Option<Stamp>> {
  let Ok(stamp) = Stamp::read(dir) else {
    return Ok(None);
  };
  let present = sources(dir)?;
  let stamped: BTreeMap<String, Currency> = stamp
    .files
    .iter()
    .filter(|(rel, _)| {
      !stamp
        .bundles
        .values()
        .any(|b| b.origin == Origin::Supplied && b.digests.contains_key(*rel))
    })
    .map(|(rel, entry)| (rel.clone(), entry.currency))
    .collect();
  let derived_exist = stamp.bundles.values().all(|b| dir.join(&b.name).is_dir())
    && stamp
      .tokenizer_image
      .as_ref()
      .is_none_or(|image| dir.join(&image.name).is_file());
  let supplied_unchanged = PACKAGES.iter().all(|package| {
    let plain = dir.join(format!("{}.mlmodelc", package.stem)).is_dir();
    let stamped_supplied = stamp
      .bundles
      .get(package.key)
      .is_some_and(|b| b.origin == Origin::Supplied);
    plain == stamped_supplied
  });
  Ok((present == stamped && derived_exist && supplied_unchanged).then_some(stamp))
}

/// Compiles and stamps the model directory `dir`: every package to a
/// content-addressed bundle, `tokenizer.json` to a content-addressed tokenizer
/// image, and each source's digest, currency key and sentinel cells into the
/// stamp. Idempotent: a current directory is left as it is, with no file
/// created and no lock taken, so a stamped read-only directory passes.
///
/// It deletes nothing, moves nothing and overwrites nothing: a new artifact
/// takes a new name, a superseded one stays until [`prune`], and only the
/// stamp's name is replaced.
///
/// # Errors
/// [`Error::AmbiguousArtifact`] when a package sits beside a caller's plainly
/// named bundle of the same artifact; [`Error::Compile`] when Core ML cannot
/// compile a package; [`Error::TokenizerSource`] for a `tokenizer.json`
/// outside the configuration the kit implements; [`Error::Io`] for a failed
/// file-system operation.
pub fn compile_artifacts(dir: impl AsRef<Path>) -> Result<CompileReport> {
  let dir = dir.as_ref();
  if let Some(stamp) = current(dir)? {
    return Ok(kept_report(&stamp));
  }
  let lock_path = dir.join(LOCK_NAME);
  let lock = File::options()
    .create(true)
    .truncate(false)
    .write(true)
    .open(&lock_path)
    .map_err(io(IoOperation::Lock, &lock_path))?;
  lock.lock().map_err(io(IoOperation::Lock, &lock_path))?;
  if let Some(stamp) = current(dir)? {
    return Ok(kept_report(&stamp));
  }
  build(dir)
}

fn kept_report(stamp: &Stamp) -> CompileReport {
  let mut kept: Vec<String> = stamp.bundles.values().map(|b| b.name.clone()).collect();
  kept.extend(stamp.tokenizer_image.as_ref().map(|i| i.name.clone()));
  CompileReport {
    compiled: Vec::new(),
    kept,
  }
}

fn build(dir: &Path) -> Result<CompileReport> {
  let previous = Stamp::read(dir).ok();
  let mut report = CompileReport::default();
  let mut files = BTreeMap::new();
  let mut bundles = BTreeMap::new();
  for package in &PACKAGES {
    let source = dir.join(source_package(package.stem));
    let plain = dir.join(format!("{}.mlmodelc", package.stem));
    match (source.is_dir(), plain.is_dir()) {
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
        if final_path.is_dir() {
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
        let mut ignored = BTreeMap::new();
        let digests = digest_files(dir, &plain, &mut ignored)?;
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
  for (rel, artifact) in &LOOSE {
    let path = dir.join(rel);
    if !path.is_file() {
      continue;
    }
    let currency = Currency::of(&fs::metadata(&path).map_err(io(IoOperation::Stat, &path))?);
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
  let tokenizer_image = match files.get("tokenizer.json") {
    Some(source) => Some(tokenizer_image(dir, &source.sha256, &mut report)?),
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
  write_new(
    &dir.join(STAMP_NAME),
    &serde_json::to_vec_pretty(&stamp).unwrap_or_default(),
    true,
  )?;
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
  for rel in files_under(dir, root)? {
    let path = dir.join(&rel);
    let currency = Currency::of(&fs::metadata(&path).map_err(io(IoOperation::Stat, &path))?);
    let sha256 = hex(&sha256_file(&path)?);
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
  use std::io::{Read, Seek, SeekFrom};
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

/// The tokenizer image of the `tokenizer.json` whose digest is `source`: kept
/// when the stamp named it and its image digest still verifies, built under
/// its content-addressed name otherwise.
fn tokenizer_image(dir: &Path, source: &str, report: &mut CompileReport) -> Result<ImageEntry> {
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
  if final_path.is_file() {
    let bytes = fs::read(&final_path).map_err(io(IoOperation::Read, &final_path))?;
    let digest_ok = bytes.len() >= HEADER_SIZE
      && bytes[header::IMAGE_DIGEST..header::IMAGE_DIGEST + 32]
        == Sha256::digest(&bytes[HEADER_SIZE..])[..];
    if digest_ok {
      report.kept.push(name.clone());
      return Ok(ImageEntry {
        name,
        format: TOKENIZER_IMAGE_VERSION,
        source_sha256: source.to_owned(),
        image_sha256: hex(&Sha256::digest(&bytes).into()),
      });
    }
    return Err(Error::TokenizerImage(TokenizerImageDefect::new(
      ImageField::ImageDigest,
      "the image's own digest".to_owned(),
      format!("`{name}` holds another; remove it, it is never replaced"),
    )));
  }
  let json_path = dir.join("tokenizer.json");
  let json = fs::read(&json_path).map_err(io(IoOperation::Read, &json_path))?;
  let image = compile_image(&json)?;
  write_new(&final_path, &image, false)?;
  report.compiled.push(name.clone());
  Ok(ImageEntry {
    name,
    format: TOKENIZER_IMAGE_VERSION,
    source_sha256: source.to_owned(),
    image_sha256: hex(&Sha256::digest(&image).into()),
  })
}

/// Writes `bytes` to a temporary name, syncs it, and renames it to
/// `final_path`. Only the stamp (`replace`) may take a name a file holds.
fn write_new(final_path: &Path, bytes: &[u8], replace: bool) -> Result<()> {
  if !replace && final_path.exists() {
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
  let mut file = File::options()
    .write(true)
    .create_new(true)
    .open(&tmp)
    .map_err(io(IoOperation::Write, &tmp))?;
  file
    .write_all(bytes)
    .map_err(io(IoOperation::Write, &tmp))?;
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

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
  fs::create_dir(to).map_err(io(IoOperation::Write, to))?;
  for entry in fs::read_dir(from).map_err(io(IoOperation::Read, from))? {
    let entry = entry.map_err(io(IoOperation::Read, from))?;
    let (source, target) = (entry.path(), to.join(entry.file_name()));
    if entry
      .file_type()
      .map_err(io(IoOperation::Stat, &source))?
      .is_dir()
    {
      copy_tree(&source, &target)?;
    } else {
      fs::copy(&source, &target).map_err(io(IoOperation::Copy, &source))?;
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
/// stamp or the lock; it follows no symbolic link. Takes the directory's
/// lock.
///
/// # Errors
/// [`Error::Unstamped`] when `dir` has no stamp; [`Error::Io`] for a failed
/// removal.
pub fn prune(dir: impl AsRef<Path>) -> Result<PruneReport> {
  let dir = dir.as_ref();
  let lock_path = dir.join(LOCK_NAME);
  let lock = File::options()
    .create(true)
    .truncate(false)
    .write(true)
    .open(&lock_path)
    .map_err(io(IoOperation::Lock, &lock_path))?;
  lock.lock().map_err(io(IoOperation::Lock, &lock_path))?;
  let stamp = Stamp::read(dir)?;
  let named: Vec<&str> = stamp
    .bundles
    .values()
    .map(|b| b.name.as_str())
    .chain(stamp.tokenizer_image.as_ref().map(|i| i.name.as_str()))
    .collect();
  let mut report = PruneReport::default();
  for entry in fs::read_dir(dir).map_err(io(IoOperation::Read, dir))? {
    let entry = entry.map_err(io(IoOperation::Read, dir))?;
    let name = entry.file_name().to_string_lossy().into_owned();
    if !derived(&name) || named.contains(&name.as_str()) {
      continue;
    }
    let path = entry.path();
    let kind = fs::symlink_metadata(&path)
      .map_err(io(IoOperation::Stat, &path))?
      .file_type();
    if kind.is_symlink() {
      continue;
    }
    if kind.is_dir() {
      fs::remove_dir_all(&path).map_err(io(IoOperation::Remove, &path))?;
    } else {
      fs::remove_file(&path).map_err(io(IoOperation::Remove, &path))?;
    }
    report.removed.push(path);
  }
  report.removed.sort();
  Ok(report)
}

#[cfg(test)]
mod tests;
