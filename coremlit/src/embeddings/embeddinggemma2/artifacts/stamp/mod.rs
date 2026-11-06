//! The stamp: one file per model directory, written by `compile_artifacts`,
//! that names the directory's current artifacts and records each source's
//! digest, currency key and, for the two tables, sentinel cells.
//!
//! The stamp is the one name `compile_artifacts` replaces (a temporary file,
//! then a rename), and it is never mapped. Load reads only the stamp and the
//! sizes of the files a part reads as stamped (its loose files, and a supplied
//! bundle's own files): it never hashes the 1.4 GB tree.

use std::{
  collections::BTreeMap,
  os::unix::fs::MetadataExt,
  path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{
  super::{
    alloc,
    audio::AudioPrecision,
    error::{
      AllocationSite, Artifact, ArtifactIdentity, Error, ReadFailure, Result, StaleStamp,
      StampDefect, StampField, TableCell, Unstamped, excerpt,
    },
  },
  ArtifactLabel, Manifest, Part,
  tree::{read_capped, resolve_or},
};

/// The stamp's file name, inside the model directory.
pub(crate) const STAMP_NAME: &str = "coremlit-embeddinggemma2.stamp.json";

/// The stamp format this build reads and writes.
pub(crate) const STAMP_FORMAT: u32 = 1;

/// The kit the stamp is for.
pub(crate) const KIT: &str = "embeddinggemma2";

/// The largest stamp the kit reads: 16 MiB. A stamp of the kit's trees is
/// about 7 KB, and one at the cap of 1024 entries in every tree stays under
/// 10 MB, so the cap refuses no stamp the kit writes and bounds the parse.
pub(crate) const STAMP_CAP: u64 = 16 << 20;

/// What identifies a source file's content without reading it: if any of these
/// moves, the file is hashed again. `ctime` and the inode survive no
/// replacement, in place or by rename, so a time-preserving copy cannot keep a
/// stale digest current.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Currency {
  pub(crate) size: u64,
  pub(crate) mtime_ns: i64,
  pub(crate) ctime_ns: i64,
  pub(crate) inode: u64,
}

impl Currency {
  pub(crate) fn of(metadata: &std::fs::Metadata) -> Self {
    Self {
      size: metadata.len(),
      mtime_ns: metadata
        .mtime()
        .saturating_mul(1_000_000_000)
        .saturating_add(metadata.mtime_nsec()),
      ctime_ns: metadata
        .ctime()
        .saturating_mul(1_000_000_000)
        .saturating_add(metadata.ctime_nsec()),
      inode: metadata.ino(),
    }
  }
}

/// One recorded sentinel cell: its place and its bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SentinelEntry {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub(crate) table: Option<u8>,
  pub(crate) row: u32,
  pub(crate) column: u32,
  pub(crate) bits: u16,
}

impl SentinelEntry {
  pub(crate) const fn cell(&self) -> TableCell {
    match self.table {
      Some(table) => TableCell::position(table, self.row, self.column),
      None => TableCell::token(self.row, self.column),
    }
  }
}

/// One stamped source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileEntry {
  pub(crate) sha256: String,
  pub(crate) currency: Currency,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub(crate) sentinels: Vec<SentinelEntry>,
}

/// Where a bundle came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Origin {
  /// `compile_artifacts` compiled it from a package beside it.
  Compiled,
  /// The caller placed it, plainly named; it is never touched.
  Supplied,
}

/// The current bundle of one package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BundleEntry {
  /// The bundle's directory name, inside the model directory.
  pub(crate) name: String,
  pub(crate) origin: Origin,
  /// The digests of the files the bundle stands for, by path relative to the
  /// model directory: the package's files, or the supplied bundle's own.
  pub(crate) digests: BTreeMap<String, String>,
}

/// The current tokenizer image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageEntry {
  pub(crate) name: String,
  pub(crate) format: u32,
  pub(crate) source_sha256: String,
  pub(crate) image_sha256: String,
}

/// Format 1 of the stamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Stamp {
  pub(crate) format: u32,
  pub(crate) kit: String,
  pub(crate) coremlit: String,
  pub(crate) os_build: String,
  /// The loose files, every package file and every supplied bundle's file,
  /// by path relative to the model directory.
  pub(crate) files: BTreeMap<String, FileEntry>,
  /// The current bundle per package key (`text`, `audio_fp16`, `audio_fp32`,
  /// `vision`).
  pub(crate) bundles: BTreeMap<String, BundleEntry>,
  #[serde(default)]
  pub(crate) tokenizer_image: Option<ImageEntry>,
}

/// One of the kit's packages: its key in the stamp, its stem and its artifact.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Package {
  pub(crate) key: &'static str,
  pub(crate) stem: &'static str,
  pub(crate) artifact: Artifact,
}

/// The kit's four packages.
pub(crate) const PACKAGES: [Package; 4] = [
  Package {
    key: "text",
    stem: "EmbeddingGemma2Text",
    artifact: Artifact::TextBundle,
  },
  Package {
    key: "audio_fp16",
    stem: "EmbeddingGemma2Audio",
    artifact: Artifact::AudioBundle(AudioPrecision::Fp16),
  },
  Package {
    key: "audio_fp32",
    stem: "EmbeddingGemma2AudioFp32",
    artifact: Artifact::AudioBundle(AudioPrecision::Fp32),
  },
  Package {
    key: "vision",
    stem: "EmbeddingGemma2Vision",
    artifact: Artifact::VisionBundle,
  },
];

/// The kit's loose source files and their artifacts.
pub(crate) const LOOSE: [(&str, Artifact); 4] = [
  ("embeddings.bf16", Artifact::TokenTable),
  ("position_embeddings.f16", Artifact::PositionTable),
  ("tokenizer.json", Artifact::TokenizerSource),
  ("config.json", Artifact::Config),
];

/// FluidInference @ `3f978dd`'s digests of each part's source files: the part
/// whose stamped files are exactly these is labelled
/// [`ArtifactLabel::FLUIDINFERENCE`].
const UPSTREAM_TEXT: [(&str, &str); 6] = [
  (
    "EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "0df9aca02807cf490db6d4f1c68607a8c7f486ad23b376d1968fd3be5eb14677",
  ),
  (
    "EmbeddingGemma2Text.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "75b97ab036ca87eae60560c4034a63d604201498dc4b6f6971dd0b1e4558d61f",
  ),
  (
    "EmbeddingGemma2Text.mlpackage/Manifest.json",
    "83eace80b69b46f8415af620ba1b40d7c490d00f7c00ba9dedd2e29da6e9ac19",
  ),
  (
    "config.json",
    "30668aba89f54a200a3fa38913f21e2a2f18aba7215de67c1806542a75b9629b",
  ),
  (
    "embeddings.bf16",
    "56ebbdbfd706c827f3fae1d97aa39135c7ddc0f1e6d9b12f8565f84bbfbda877",
  ),
  (
    "tokenizer.json",
    "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
  ),
];
const UPSTREAM_AUDIO_FP16: [(&str, &str); 3] = [
  (
    "EmbeddingGemma2Audio.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "d40896e5c88d62b02d15cc975f27e254c28a5774a1702e09bf5483071623548d",
  ),
  (
    "EmbeddingGemma2Audio.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "41d7b552caabe376e377dcbe1c8c5f9879853bc4f384a161bb1f886f19e84b56",
  ),
  (
    "EmbeddingGemma2Audio.mlpackage/Manifest.json",
    "65c8e9811516e7916a3d742261c7d5a02d4e43c113b437330ce86f98b3da3514",
  ),
];
const UPSTREAM_VISION: [(&str, &str); 4] = [
  (
    "EmbeddingGemma2Vision.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "0085b5e04a13d000fe5969adeaa13d897f5a3cb7be60c76c8685c023747f89ef",
  ),
  (
    "EmbeddingGemma2Vision.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "5b5b6dd8d80aa438bf91ba143fbe90d56a2e25bfcf685a56b990aad71cf75232",
  ),
  (
    "EmbeddingGemma2Vision.mlpackage/Manifest.json",
    "376afe7b51608637312cd15a0a89b1d7863f9779502aaf5f12fa6b14c4272d8d",
  ),
  (
    "position_embeddings.f16",
    "17de9480acc23638c0634a97ed89e51ed664433256d7d6f109587dcd91c79124",
  ),
];

fn defect(field: StampField, detail: impl Into<String>) -> Error {
  Error::Stamp(StampDefect::new(field, detail.into()))
}

/// 64 lowercase hex digits as bytes.
fn digest_bytes(hex: &str) -> Option<[u8; 32]> {
  let bytes = hex.as_bytes();
  if bytes.len() != 64 {
    return None;
  }
  let mut out = [0u8; 32];
  for (i, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
    let text = core::str::from_utf8(pair).ok()?;
    out[i] = u8::from_str_radix(text, 16).ok()?;
  }
  Some(out)
}

impl Stamp {
  /// Reads `dir`'s stamp, at most [`STAMP_CAP`] bytes.
  ///
  /// # Errors
  /// [`Error::Unstamped`] naming the stamp when there is none;
  /// [`Error::Read`] when it cannot be read; [`Error::Traversal`] when it is
  /// a broken link; [`Error::Stamp`] when it is past its cap, refused before
  /// a byte is read, or not a format-1 stamp of this kit;
  /// [`Error::Allocation`] at [`AllocationSite::ArtifactRead`].
  pub(crate) fn read(dir: &Path) -> Result<Self> {
    let path = dir.join(STAMP_NAME);
    let bytes = read_capped(
      &path,
      STAMP_CAP,
      |size| {
        defect(
          StampField::Document,
          format!("{size} bytes is past the {STAMP_CAP}-byte cap of a stamp"),
        )
      },
      |e| {
        if e.kind() == std::io::ErrorKind::NotFound {
          Error::Unstamped(Unstamped::new(Artifact::Stamp, dir.to_path_buf()))
        } else {
          Error::Read(ReadFailure::new(Artifact::Stamp, path.clone(), e))
        }
      },
    )?;
    let stamp: Self =
      serde_json::from_slice(&bytes).map_err(|e| defect(StampField::Document, excerpt(e)))?;
    if stamp.format != STAMP_FORMAT {
      return Err(defect(
        StampField::Format,
        format!("format {} is not this build's {STAMP_FORMAT}", stamp.format),
      ));
    }
    if stamp.kit != KIT {
      return Err(defect(
        StampField::Kit,
        format!("`{}` is not `{KIT}`", excerpt(&stamp.kit)),
      ));
    }
    Ok(stamp)
  }

  /// A stamped file, which must still have its stamped size: its target's,
  /// when it is a link.
  fn file(&self, dir: &Path, rel: &str, artifact: Artifact) -> Result<&FileEntry> {
    let entry = self
      .files
      .get(rel)
      .ok_or_else(|| Error::Unstamped(Unstamped::new(artifact, dir.to_path_buf())))?;
    let path = dir.join(rel);
    let failed = |e| Error::Read(ReadFailure::new(artifact, path.clone(), e));
    let actual = match resolve_or(&path, &failed)? {
      Some(found) => found.metadata.len(),
      None => return Err(failed(std::io::ErrorKind::NotFound.into())),
    };
    if actual != entry.currency.size {
      return Err(Error::StaleStamp(StaleStamp::new(
        artifact,
        entry.currency.size,
        actual,
      )));
    }
    Ok(entry)
  }

  /// A package's current bundle.
  fn bundle(&self, dir: &Path, package: &Package) -> Result<&BundleEntry> {
    self
      .bundles
      .get(package.key)
      .ok_or_else(|| Error::Unstamped(Unstamped::new(package.artifact, dir.to_path_buf())))
  }

  /// What a part needs from `dir`, read from the stamp: its bundle, its loose
  /// files, its table's sentinels, and the label of its artifacts.
  ///
  /// A supplied bundle is read as stamped, so each of its stamped files must
  /// still have its stamped size, as a loose file must; a link's size is its
  /// target's. A compiled bundle is this module's own, under a
  /// content-addressed name that is never replaced, so its package's later
  /// edits do not move what it loads.
  ///
  /// # Errors
  /// [`Error::Unstamped`] for an artifact the stamp does not list;
  /// [`Error::StaleStamp`] for a stamped file whose size moved; [`Error::Read`]
  /// for a stamped file that is gone; [`Error::Traversal`] for one that is now
  /// a broken link; [`Error::Allocation`] at [`AllocationSite::StampDigests`]
  /// for the table's sentinels.
  pub(crate) fn part(&self, dir: &Path, part: Part, precision: AudioPrecision) -> Result<PartView> {
    let (package, loose, upstream): PartSources<'_> = match part {
      Part::Text => (
        &PACKAGES[0],
        &[LOOSE[0], LOOSE[2], LOOSE[3]],
        &UPSTREAM_TEXT,
      ),
      Part::Audio => match precision {
        AudioPrecision::Fp16 => (&PACKAGES[1], &[], &UPSTREAM_AUDIO_FP16),
        AudioPrecision::Fp32 => (&PACKAGES[2], &[], &[]),
      },
      Part::Vision => (&PACKAGES[3], &[LOOSE[1]], &UPSTREAM_VISION),
    };
    let bundle = self.bundle(dir, package)?;
    if bundle.origin == Origin::Supplied {
      for rel in bundle.digests.keys() {
        self.file(dir, rel, package.artifact)?;
      }
    }
    // The part's loose files, sorted bytewise by name beside the bundle's
    // digests: at most three, held on the stack.
    let mut held: [(&str, &str); 3] = [("", ""); 3];
    let mut sentinels = Vec::new();
    for (slot, &(rel, artifact)) in held.iter_mut().zip(loose) {
      let entry = self.file(dir, rel, artifact)?;
      *slot = (rel, entry.sha256.as_str());
      if !entry.sentinels.is_empty() {
        sentinels = alloc::collect(
          entry.sentinels.iter().map(|s| (s.cell(), s.bits)),
          AllocationSite::StampDigests,
        )?;
      }
    }
    let held = &mut held[..loose.len()];
    held.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let image = match part {
      Part::Text => Some(self.tokenizer_image.clone().ok_or_else(|| {
        Error::Unstamped(Unstamped::new(Artifact::TokenizerImage, dir.to_path_buf()))
      })?),
      _ => None,
    };
    // Every file the part's label covers, in manifest order, from the stamp
    // as it stands: nothing is copied.
    let files = || {
      merged(
        bundle
          .digests
          .iter()
          .map(|(rel, hex)| (rel.as_str(), hex.as_str())),
        held.iter().copied(),
      )
    };
    let upstream_files = !upstream.is_empty()
      && files().count() == upstream.len()
      && files().all(|line| upstream.contains(&line));
    let label = if upstream_files {
      ArtifactLabel::FLUIDINFERENCE
    } else {
      let mut manifest = Manifest::new();
      for (rel, hex) in files() {
        let digest = digest_bytes(hex)
          .ok_or_else(|| defect(StampField::Files, "a digest is not 64 lowercase hex digits"))?;
        manifest.line(rel, &digest);
      }
      ArtifactLabel::from_manifest(&manifest.finish())
    };
    Ok(PartView {
      part,
      bundle: dir.join(&bundle.name),
      sentinels,
      image,
      label,
    })
  }
}

/// Two lists of lines, each sorted bytewise by path, as one sorted list; on a
/// path in both, the second's line, as an insertion over the first would
/// keep it.
fn merged<'a>(
  first: impl Iterator<Item = (&'a str, &'a str)>,
  second: impl Iterator<Item = (&'a str, &'a str)>,
) -> impl Iterator<Item = (&'a str, &'a str)> {
  let (mut first, mut second) = (first.peekable(), second.peekable());
  core::iter::from_fn(move || match (first.peek(), second.peek()) {
    (Some(a), Some(b)) => match a.0.as_bytes().cmp(b.0.as_bytes()) {
      core::cmp::Ordering::Less => first.next(),
      core::cmp::Ordering::Greater => second.next(),
      core::cmp::Ordering::Equal => {
        first.next();
        second.next()
      }
    },
    (Some(_), None) => first.next(),
    (None, _) => second.next(),
  })
}

/// A part's package, its loose files, and upstream's digests of its files.
type PartSources<'a> = (
  &'a Package,
  &'a [(&'a str, Artifact)],
  &'a [(&'a str, &'a str)],
);

/// What one part opens, as its stamp names it.
#[derive(Debug, Clone)]
pub(crate) struct PartView {
  part: Part,
  bundle: PathBuf,
  sentinels: Vec<(TableCell, u16)>,
  image: Option<ImageEntry>,
  label: ArtifactLabel,
}

impl PartView {
  /// The current bundle's path.
  pub(crate) fn bundle(&self) -> &Path {
    &self.bundle
  }

  /// The sentinel cells the stamp recorded for the part's table, if it has one.
  pub(crate) fn sentinels(&self) -> &[(TableCell, u16)] {
    &self.sentinels
  }

  /// The current tokenizer image, for the text part.
  pub(crate) fn image(&self) -> Option<&ImageEntry> {
    self.image.as_ref()
  }

  /// The label of the part's artifacts.
  pub(crate) const fn label(&self) -> ArtifactLabel {
    self.label
  }

  /// L14 (a): a deployment's pinned label, when it has one, must be the
  /// stamp's.
  ///
  /// # Errors
  /// [`Error::ArtifactIdentity`] naming the part, the pinned label and the
  /// stamp's.
  pub(crate) fn expect(&self, expected: Option<ArtifactLabel>) -> Result<()> {
    match expected {
      Some(expected) if expected != self.label => Err(Error::ArtifactIdentity(
        ArtifactIdentity::new(self.part, expected, self.label),
      )),
      _ => Ok(()),
    }
  }
}
