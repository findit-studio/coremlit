//! Model lifecycle vocabulary: the [`ModelState`] machine, [`ModelVariant`]
//! detection, compiled-bundle folder detection, [`ModelInfo`] download-path
//! descriptors, and device→model [`SupportConfig`]. Ports
//! `ArgmaxCore/{ModelState,ModelUtilities,ModelDownloader,
//! FoundationExtensions}.swift` and `WhisperKit/{Core/Models,
//! Utilities/ModelUtilities}.swift` (see each item's doc for exact line
//! citations).
//!
//! [`manager`] hosts [`manager::ModelManager`]: Swift's coalesced
//! load/unload/prewarm orchestrator (`ArgmaxCore/ModelManager.swift`),
//! reshaped for synchronous, single-owner use over a live `crate::Model`
//! and driving the [`ModelState`] transitions below (see that module's doc
//! for the sync reshape). Everything else here ships the model-lifecycle
//! *vocabulary* that [`manager::ModelManager`] and the rest of the
//! pipeline use:
//! state names, variant detection, folder/glob utilities, the
//! support-config lookup, and the [`ModelLoader`] seam `ModelManager`
//! resolves bundle paths through — [`ModelState`] transitions themselves
//! are enforced only by [`manager::ModelManager`], not by anything else in
//! this module.
//!
//! Two implementation decisions worth recording:
//!
//! - **[`device_identifier`]'s `unsafe`**: ports Swift's two-call
//!   `sysctlbyname` protocol (`ArgmaxCore/FoundationExtensions.swift:
//!   122-137`) via `libc::sysctlbyname` directly, rather than shelling out
//!   to `sysctl(8)` through `std::process::Command` (a process spawn plus
//!   stdout parsing for one syscall's worth of information, and not
//!   meaningfully safer) or pulling in a dedicated `sysctl`-wrapper crate
//!   for a single call. The whole function is small, with two
//!   narrowly-scoped `unsafe` blocks that each carry a `SAFETY` comment —
//!   exactly the "10-line unsafe route, if `clippy::undocumented_unsafe_blocks`
//!   stays clean" this task's brief asked to prefer; verified clean.
//! - **[`SupportConfig::from_json`] over `serde_json::Value`, not derived
//!   `Deserialize`**: `serde` stays an *optional* feature of this crate
//!   (gating the `Serialize`/`Deserialize` derives on `options`/`result`),
//!   but parsing a support-config JSON document is unconditional
//!   day-one behavior, not an opt-in extra. Deriving `Deserialize` on
//!   [`SupportConfig`] would force the crate's own `serde` dependency (not
//!   just `serde_json`) to become unconditional too, which would make the
//!   crate's `serde` *feature* meaningless (already unconditionally on).
//!   Adding `serde_json` (not `serde`) as a small, unconditional dependency
//!   and hand-walking its `Value` tree keeps the `serde` feature exactly
//!   what it already means elsewhere in this crate — optional derive
//!   support — while still parsing JSON unconditionally. None of the types
//!   in this module derive `serde::{Serialize, Deserialize}`.

use std::path::{Path, PathBuf};

use crate::ComputeUnits;
use serde_json::Value;

use crate::audio::whisper::error::ModelError;

// ---------------------------------------------------------------------
// ModelState
// ---------------------------------------------------------------------

/// Lifecycle state of a loaded CoreML model pipeline (Swift `ModelState`,
/// `ArgmaxCore/ModelState.swift:19-50` — shared there by WhisperKit and
/// TTSKit; TTSKit is out of scope for this port, spec §3, so this lives in
/// `whisperkit::model` instead of a shared crate).
///
/// State machine (Swift's own diagram, `ModelState.swift:13-18`):
/// ```text
/// unloaded -> downloading -> downloaded -> loading -> loaded
/// unloaded -> prewarming  -> prewarmed
/// loaded   -> unloading   -> unloaded
/// ```
/// Transitions are enforced by `ModelManager` (deferred to Plan 3, see this
/// module's doc); this type is pure vocabulary — any state can be
/// constructed directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum ModelState {
  /// No model is loaded.
  Unloaded,
  /// A model is being downloaded.
  Downloading,
  /// A model finished downloading but is not loaded.
  Downloaded,
  /// A model is being loaded and unloaded once, to force ANE
  /// specialization ahead of real use.
  Prewarming,
  /// The prewarm pass finished; the model is unloaded again.
  Prewarmed,
  /// A model is being loaded into memory.
  Loading,
  /// A model is loaded and ready for inference.
  Loaded,
  /// A loaded model is being released.
  Unloading,
}

impl ModelState {
  /// Stable lowercase name of the variant — the crate's own `as_str`
  /// convention (matching [`crate::audio::whisper::options::Task`]/
  /// [`crate::audio::whisper::options::ChunkingStrategy`]), **not** Swift's `description`,
  /// which separately renames `.prewarming`/`.prewarmed` to
  /// `"Specializing"`/`"Specialized"` for UI display
  /// (`ModelState.swift:36-37`). This port has no UI layer, so `as_str`
  /// and [`Display`](std::fmt::Display) both just lowercase the variant
  /// name instead, like every other enum in this crate.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Unloaded => "unloaded",
      Self::Downloading => "downloading",
      Self::Downloaded => "downloaded",
      Self::Prewarming => "prewarming",
      Self::Prewarmed => "prewarmed",
      Self::Loading => "loading",
      Self::Loaded => "loaded",
      Self::Unloading => "unloading",
    }
  }

  /// Whether a loading or downloading operation is in progress (Swift
  /// `ModelState.isBusy`, `ModelState.swift:44-49`).
  #[inline(always)]
  pub const fn is_busy(&self) -> bool {
    matches!(
      self,
      Self::Downloading | Self::Prewarming | Self::Loading | Self::Unloading
    )
  }
}

/// Callback invoked when a model pipeline's [`ModelState`] changes (Swift
/// `ModelStateCallback`, `ArgmaxCore/ModelState.swift:53`: `@Sendable
/// (_ oldState: ModelState?, _ newState: ModelState) -> Void`).
pub type StateCallback = Box<dyn Fn(Option<ModelState>, ModelState) + Send>;

// ---------------------------------------------------------------------
// ModelVariant / detect_variant
// ---------------------------------------------------------------------

/// Model size/language variant (Swift `ModelVariant`,
/// `WhisperKit/Core/Models.swift:39-84`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum ModelVariant {
  /// `tiny` — multilingual.
  Tiny,
  /// `tiny.en` — English-only.
  TinyEn,
  /// `base` — multilingual.
  Base,
  /// `base.en` — English-only.
  BaseEn,
  /// `small` — multilingual.
  Small,
  /// `small.en` — English-only.
  SmallEn,
  /// `medium` — multilingual.
  Medium,
  /// `medium.en` — English-only.
  MediumEn,
  /// `large` — multilingual, v1. [`detect_variant`] never produces this
  /// variant: v1 and v2 share the same vocab/encoder dimensions, and
  /// Swift's table resolves that combination to [`Self::LargeV2`]
  /// (`ModelUtilities.swift:144-145`, comment `// same for v1`).
  Large,
  /// `large-v2` — multilingual.
  LargeV2,
  /// `large-v3` — multilingual, one extra vocabulary token.
  LargeV3,
}

impl ModelVariant {
  /// Stable name matching Swift's `description` exactly
  /// (`Models.swift:62-83`). The English-only suffix is a **dot**
  /// (`"tiny.en"`), while the large-version suffix is a **dash**
  /// (`"large-v2"`) — these are two different separators, easy to
  /// conflate by guessing rather than reading the source.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Tiny => "tiny",
      Self::TinyEn => "tiny.en",
      Self::Base => "base",
      Self::BaseEn => "base.en",
      Self::Small => "small",
      Self::SmallEn => "small.en",
      Self::Medium => "medium",
      Self::MediumEn => "medium.en",
      Self::Large => "large",
      Self::LargeV2 => "large-v2",
      Self::LargeV3 => "large-v3",
    }
  }

  /// Whether this variant understands languages other than English (Swift
  /// `ModelVariant.isMultilingual`, `Models.swift:53-59`).
  #[inline(always)]
  pub const fn is_multilingual(&self) -> bool {
    !matches!(
      self,
      Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn
    )
  }
}

/// Whether a model's vocabulary size indicates a multilingual model — every
/// size except English-only's `51864` (Swift
/// `ModelUtilities.isModelMultilingual(logitsDim:)`,
/// `WhisperKit/Utilities/ModelUtilities.swift:124-126`: `logitsDim !=
/// 51864`). Swift's parameter is `Int?` (`nil != 51864` is `true`, so a
/// missing dimension reads as multilingual there); every call site in this
/// port always has a concrete dimension in hand, so this takes a required
/// `usize` instead.
#[inline(always)]
pub const fn is_model_multilingual(logits_dim: usize) -> bool {
  logits_dim != 51_864
}

/// Resolves a [`ModelVariant`] from the decoder's vocabulary size
/// (`logits_dim`) and the encoder's embedding width (`encoder_dim`) — ports
/// the lookup table `ModelUtilities.detectVariant(logitsDim:encoderDim:)`
/// (`WhisperKit/Utilities/ModelUtilities.swift:128-173`).
///
/// Swift's version is a *total* function: an unrecognized `encoder_dim`
/// within a recognized vocabulary falls back to `.base`/`.baseEn`, and an
/// unrecognized `logits_dim` falls back to `.base` after logging an error
/// (lines 147, 161, 168-169) — there is no "unknown" outcome. This port
/// turns that silent mismatch into an honest [`None`] instead (this task's
/// pinned test requires it: `detect_variant(999, 384) == None`): any
/// `(logits_dim, encoder_dim)` pair not literally present in Swift's
/// `switch` cases returns `None` here rather than guessing `Base`/`BaseEn`.
/// The *recognized* cells match Swift exactly, including the one branch
/// that does not consult `encoder_dim` at all: `logits_dim == 51866`
/// ("Large v3 has 1 additional language token", line 165) unconditionally
/// resolves to [`ModelVariant::LargeV3`] regardless of `encoder_dim` —
/// Swift's own code never branches on it there, so neither does this port.
pub const fn detect_variant(logits_dim: usize, encoder_dim: usize) -> Option<ModelVariant> {
  match logits_dim {
    51_865 => match encoder_dim {
      384 => Some(ModelVariant::Tiny),
      512 => Some(ModelVariant::Base),
      768 => Some(ModelVariant::Small),
      1024 => Some(ModelVariant::Medium),
      1280 => Some(ModelVariant::LargeV2),
      _ => None,
    },
    51_864 => match encoder_dim {
      384 => Some(ModelVariant::TinyEn),
      512 => Some(ModelVariant::BaseEn),
      768 => Some(ModelVariant::SmallEn),
      1024 => Some(ModelVariant::MediumEn),
      _ => None,
    },
    51_866 => Some(ModelVariant::LargeV3),
    _ => None,
  }
}

// ---------------------------------------------------------------------
// Compiled-bundle folder detection
// ---------------------------------------------------------------------

/// Detects a named CoreML model bundle inside `folder`, preferring the
/// compiled `{name}.mlmodelc` bundle over an `{name}.mlpackage` source
/// bundle. Mirrors the two related Swift overloads `ModelUtilities.
/// detectModelURL(inFolder:named:)` / `detectModelURL(inFolder:named:
/// recursive:)` (`ArgmaxCore/ModelUtilities.swift:37-70`), unified here
/// behind the `recursive` flag since the non-recursive Swift overload IS
/// the `recursive == false` branch of the recursive one (line 38: `guard
/// recursive else { return detectModelURL(inFolder:named:) }`).
///
/// - `recursive == false`: checks `folder/{name}.mlmodelc` and
///   `folder/{name}.mlpackage/Data/com.apple.CoreML/model.mlmodel`
///   directly; compiled wins whenever both exist
///   (`ModelUtilities.swift:59-70`).
/// - `recursive == true`: checks the direct compiled path only — **no**
///   package fallback in this branch, verified against source: Swift's
///   recursive overload never consults `.mlpackage` at all
///   (`ModelUtilities.swift:39-44`) — then walks every subdirectory of
///   `folder` looking for an entry named `{name}.mlmodelc` (ports Swift's
///   `FileManager.enumerator` walk, `ModelUtilities.swift:46-52`, as a
///   plain recursive [`std::fs::read_dir`] walk — no new dependency).
///
/// Swift's versions are *total* functions that always return a URL, even a
/// nonexistent one, leaving existence-checking to the caller. This port
/// checks existence itself and returns [`ModelError::NotFound`] (listing
/// every concrete candidate path checked) instead of ever fabricating a
/// guessed path — the more idiomatic Rust shape, and exactly what every
/// caller in this crate needs ([`LocalModelLoader::resolve`] in
/// particular).
///
/// # Errors
/// [`ModelError::NotFound`] if no matching bundle exists.
pub fn detect_model_url(folder: &Path, name: &str, recursive: bool) -> Result<PathBuf, ModelError> {
  let compiled = folder.join(format!("{name}.mlmodelc"));

  if recursive {
    if compiled.exists() {
      return Ok(compiled);
    }
    let target = format!("{name}.mlmodelc");
    if let Some(found) = find_named_recursive(folder, &target) {
      return Ok(found);
    }
    return Err(ModelError::NotFound {
      searched: vec![compiled],
    });
  }

  if compiled.exists() {
    return Ok(compiled);
  }
  let package = folder
    .join(format!("{name}.mlpackage"))
    .join("Data/com.apple.CoreML/model.mlmodel");
  if package.exists() {
    return Ok(package);
  }
  Err(ModelError::NotFound {
    searched: vec![compiled, package],
  })
}

/// Depth-first, pre-order search under `folder` for an entry (file or
/// directory) named exactly `target_name` — ports the effect of Swift's
/// `FileManager.enumerator(at:includingPropertiesForKeys:)` walk
/// (`ModelUtilities.swift:46-52`). The exact visitation order is
/// unspecified on both sides (Apple documents no enumerator order this
/// port needs to match), so any correct traversal that finds an existing,
/// uniquely-named bundle somewhere in the tree is equivalent; unreadable
/// subdirectories are silently skipped rather than aborting the whole
/// search (matches the enumerator's own default error-handling, which
/// simply omits entries it cannot read).
fn find_named_recursive(folder: &Path, target_name: &str) -> Option<PathBuf> {
  // Iterative worklist, descending only into REAL directories
  // (`DirEntry::file_type` does not follow symlinks): a self-referential
  // directory symlink (`loop -> .`) must terminate the search, not
  // overflow the stack recursing through itself. `.mlmodelc` bundle roots
  // reached via a symlinked PATH still match by name before the descent
  // decision, so plain model-folder symlinks keep working.
  let mut worklist = vec![folder.to_path_buf()];
  while let Some(dir) = worklist.pop() {
    let Ok(entries) = std::fs::read_dir(&dir) else {
      continue;
    };
    for entry in entries.flatten() {
      let path = entry.path();
      if path.file_name().and_then(|n| n.to_str()) == Some(target_name) {
        return Some(path);
      }
      if entry.file_type().is_ok_and(|t| t.is_dir()) {
        worklist.push(path);
      }
    }
  }
  None
}

// ---------------------------------------------------------------------
// glob_match
// ---------------------------------------------------------------------

/// `fnmatch`-style glob match: `*` matches any run of characters
/// (including path separators — this is `fnmatch` with no `FNM_PATHNAME`
/// flag), `?` matches exactly one character, every other character must
/// match literally. No bracket character classes (`[...]`) and no
/// backslash escaping — every call site this ports
/// (`ArgmaxCore/FoundationExtensions.swift:113-118`'s
/// `[String].matching(glob:)`, `filter { fnmatch(glob, $0, 0) == 0 }`,
/// called from `WhisperKit/Core/WhisperKit.swift:237`,
/// `TTSKit/TTSKit.swift:272`, and `ArgmaxCore/External/Hub/HubApi.swift:
/// 270,621`) only ever passes plain name/version/variant patterns built
/// from `*` and literal path segments — e.g. [`ModelInfo::download_pattern`]'s
/// `"name/version/variant/*"` shape — never brackets or escapes, so this
/// port only implements the subset those call sites exercise.
///
/// Iterative two-pointer wildcard match (the classic "wildcard matching"
/// algorithm, restricted to `*`/`?`): `O(pattern.len() * name.len())` worst
/// case, no recursion, and no allocation beyond the two `Vec<char>` scratch
/// buffers (needed so indexing never splits a multi-byte UTF-8 character).
pub fn glob_match(pattern: &str, name: &str) -> bool {
  let pat: Vec<char> = pattern.chars().collect();
  let text: Vec<char> = name.chars().collect();
  let (mut pi, mut ti) = (0usize, 0usize);
  // (pattern index just past the last '*', text index it last matched up to)
  let mut star: Option<(usize, usize)> = None;

  while ti < text.len() {
    if pi < pat.len() && (pat[pi] == '?' || pat[pi] == text[ti]) {
      pi += 1;
      ti += 1;
    } else if pi < pat.len() && pat[pi] == '*' {
      star = Some((pi + 1, ti));
      pi += 1;
    } else if let Some((resume_pi, matched_ti)) = star {
      // Backtrack: let the last '*' consume one more character than before.
      ti = matched_ti + 1;
      pi = resume_pi;
      star = Some((resume_pi, ti));
    } else {
      return false;
    }
  }
  while pat.get(pi) == Some(&'*') {
    pi += 1;
  }
  pi == pat.len()
}

// ---------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------

/// Metadata identifying a model for download and local resolution (Swift
/// `ModelInfo`, `ArgmaxCore/ModelDownloader.swift:300-349`) — the
/// descriptor a future model downloader/kit builds folder paths and Hub
/// glob patterns from. `version`/`variant` are optional in Swift (`String?`,
/// not `String`), and that optionality is load-bearing for
/// [`Self::download_pattern`]'s `"*"` fallback below — ported faithfully
/// rather than the required-`String` shape a first skim of this task's
/// brief suggested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
  name: String,
  version: Option<String>,
  variant: Option<String>,
  compute: ComputeUnits,
}

impl ModelInfo {
  /// Builds a model descriptor, rejecting an empty `name` (Swift's `name`
  /// is a required, non-optional `String`, `ModelDownloader.swift:303`, but
  /// nothing there stops it from being empty — this port adds that check).
  ///
  /// # Errors
  /// [`ModelError::EmptyName`] if `name` is empty.
  pub fn try_new(
    name: impl Into<String>,
    version: Option<String>,
    variant: Option<String>,
    compute: ComputeUnits,
  ) -> Result<Self, ModelError> {
    let name = name.into();
    if name.is_empty() {
      return Err(ModelError::EmptyName);
    }
    Ok(Self {
      name,
      version,
      variant,
      compute,
    })
  }

  /// The model's name (e.g. `"speaker_segmenter"`).
  #[inline(always)]
  pub fn name(&self) -> &str {
    self.name.as_str()
  }

  /// The model's version, if pinned.
  #[inline(always)]
  pub fn version(&self) -> Option<&str> {
    self.version.as_deref()
  }

  /// The model's quantization/precision variant, if pinned.
  #[inline(always)]
  pub fn variant(&self) -> Option<&str> {
    self.variant.as_deref()
  }

  /// The compute units this model should load with.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }

  /// Glob pattern selecting every file belonging to this model within a
  /// repo — `"{name}/{version}/{variant}/*"`, with a missing
  /// `version`/`variant` becoming a literal `"*"` (Swift
  /// `ModelInfo.downloadPattern`, `ModelDownloader.swift:326-328`:
  /// `"\(name)/\(version ?? "*")/\(variant ?? "*")/*"`). Verified against
  /// `ModelDownloaderTests.testModelInfoDownloadPattern`
  /// (`Tests/ArgmaxCoreTests/ModelDownloaderTests.swift:199-205`): a fully
  /// specified `ModelInfo` produces `"speaker_segmenter/pyannote-v3/
  /// W8A16/*"`; a name-only one produces `"speaker_segmenter/*/*/*"`.
  pub fn download_pattern(&self) -> String {
    format!(
      "{}/{}/{}/*",
      self.name,
      self.version.as_deref().unwrap_or("*"),
      self.variant.as_deref().unwrap_or("*"),
    )
  }

  /// Finds the base folder by walking **up** from `path` until a directory
  /// named [`Self::name`] is found, returning that directory's parent —
  /// e.g. for `name = "speaker_segmenter"` and `path =
  /// ".../speaker_segmenter/pyannote-v3/W8A16"`, returns `"..."`. Returns
  /// `None` if no ancestor of `path` is named [`Self::name`].
  ///
  /// Ports `ModelInfo.findBaseFolder(in:)` (`ModelDownloader.swift:
  /// 338-348`) **exactly as written**, not as this task's brief described
  /// it (which read the algorithm as "walk `root` for the deepest
  /// directory matching the pattern prefix" — verified against source and
  /// against `ModelDownloaderTests.testModelInfoFindBaseFolder`,
  /// `ModelDownloaderTests.swift:220-227`, and that reading does not
  /// match: the real algorithm never touches [`Self::download_pattern`]'s
  /// glob at all, does no filesystem existence-checking, and walks
  /// *upward* through path components rather than searching *downward*
  /// through a tree). Swift's version operates on Foundation `URL`s with no
  /// I/O; this port operates on [`Path`] components identically — pure
  /// string manipulation, no filesystem access.
  pub fn find_base_folder(&self, path: &Path) -> Option<PathBuf> {
    let mut current = path.to_path_buf();
    loop {
      if current.components().count() <= 1 {
        return None;
      }
      if current.file_name().and_then(|n| n.to_str()) == Some(self.name.as_str()) {
        current.pop();
        return Some(current);
      }
      if !current.pop() {
        return None;
      }
    }
  }
}

// ---------------------------------------------------------------------
// ModelSupport / DeviceSupport / SupportConfig
// ---------------------------------------------------------------------

/// The model recommended for devices with no matching [`SupportConfig`]
/// entry (Swift hardcodes this in `ModelSupportConfig.init`'s
/// `defaultSupport` construction, `Models.swift:196`).
pub const DEFAULT_FALLBACK_MODEL_NAME: &str = "openai_whisper-base";

/// Which models a device tier supports (Swift `ModelSupport`,
/// `Models.swift:120-138`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSupport {
  default_model: String,
  supported: Vec<String>,
  disabled: Vec<String>,
}

impl ModelSupport {
  /// Builds a model-support record directly (Swift `ModelSupport.init`,
  /// `Models.swift:129-137`).
  pub fn new(
    default_model: impl Into<String>,
    supported: impl Into<Vec<String>>,
    disabled: impl Into<Vec<String>>,
  ) -> Self {
    Self {
      default_model: default_model.into(),
      supported: supported.into(),
      disabled: disabled.into(),
    }
  }

  /// The recommended model for this tier.
  #[inline(always)]
  pub fn default_model(&self) -> &str {
    self.default_model.as_str()
  }

  /// Every model name supported by this tier.
  #[inline(always)]
  pub const fn supported_slice(&self) -> &[String] {
    self.supported.as_slice()
  }

  /// Known model names NOT supported by this tier — computed by
  /// [`SupportConfig`] as `known_models - supported` (Swift
  /// `computeDisabledModels`, `Models.swift:227-232`); empty when built
  /// directly via [`Self::new`] rather than through a [`SupportConfig`].
  #[inline(always)]
  pub const fn disabled_slice(&self) -> &[String] {
    self.disabled.as_slice()
  }
}

/// A device tier's identifiers and [`ModelSupport`] (Swift `DeviceSupport`,
/// `Models.swift:141-154`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSupport {
  chips: Option<String>,
  identifiers: Vec<String>,
  models: ModelSupport,
}

impl DeviceSupport {
  /// Builds a device-tier record directly (Swift `DeviceSupport.init`,
  /// `Models.swift:150-154`).
  pub fn new(
    chips: Option<String>,
    identifiers: impl Into<Vec<String>>,
    models: ModelSupport,
  ) -> Self {
    Self {
      chips,
      identifiers: identifiers.into(),
      models,
    }
  }

  /// Chip family names, for annotation only (e.g. `"A16, A17 Pro, A18"`).
  #[inline(always)]
  pub fn chips(&self) -> Option<&str> {
    self.chips.as_deref()
  }

  /// Device identifier prefixes this tier covers (e.g. `["iPhone15",
  /// "iPhone16"]`).
  #[inline(always)]
  pub const fn identifiers_slice(&self) -> &[String] {
    self.identifiers.as_slice()
  }

  /// This tier's model support.
  #[inline(always)]
  pub const fn models(&self) -> &ModelSupport {
    &self.models
  }
}

/// Device→model support table: which Whisper models each device tier can
/// run, with longest-identifier-prefix lookup (Swift `ModelSupportConfig`,
/// `Models.swift:156-244`). This port carries only the local-resolution
/// half of Swift's type — `repoName`/`repoVersion` and the
/// remote/fallback-merge machinery (`mergeDeviceSupport`,
/// `Models.swift:239-256`) are HubApi-network concerns, out of scope here
/// (spec §3's network-stack exclusion) — [`Self::from_json`] parses a
/// config document directly, with no merge step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportConfig {
  device_supports: Vec<DeviceSupport>,
  known_models: Vec<String>,
  default_support: DeviceSupport,
}

impl SupportConfig {
  /// Builds a config from raw device-support entries, computing
  /// [`Self::known_models_slice`] (the order-preserving union of every
  /// tier's supported models — Swift's `orderedSet` over a flat-mapped
  /// array, `Models.swift:186`) and each tier's
  /// [`ModelSupport::disabled_slice`] (`known_models - own supported`,
  /// Swift `computeDisabledModels`, `Models.swift:227-232`, which uses
  /// `Set` subtraction; this collects in `known_models` order instead for
  /// a deterministic result — the same "`Vec` instead of `Set`" tradeoff
  /// [`crate::audio::whisper::tokenizer::WhisperTokenizer::from_folder`] already makes for
  /// `all_language_tokens`), plus [`Self::default_support`] (Swift's
  /// `defaultSupport`, `Models.swift:196-202`: no identifiers, default
  /// model [`DEFAULT_FALLBACK_MODEL_NAME`], every known model "supported").
  fn from_device_supports(device_supports: Vec<DeviceSupport>) -> Self {
    let mut known_models: Vec<String> = Vec::new();
    for ds in &device_supports {
      for model in ds.models().supported_slice() {
        if !known_models.contains(model) {
          known_models.push(model.clone());
        }
      }
    }

    let device_supports: Vec<DeviceSupport> = device_supports
      .into_iter()
      .map(|ds| {
        let disabled: Vec<String> = known_models
          .iter()
          .filter(|m| !ds.models().supported_slice().contains(m))
          .cloned()
          .collect();
        DeviceSupport::new(
          ds.chips().map(str::to_string),
          ds.identifiers_slice().to_vec(),
          ModelSupport::new(
            ds.models().default_model().to_string(),
            ds.models().supported_slice().to_vec(),
            disabled,
          ),
        )
      })
      .collect();

    let default_support = DeviceSupport::new(
      None,
      Vec::new(),
      ModelSupport::new(
        DEFAULT_FALLBACK_MODEL_NAME,
        known_models.clone(),
        Vec::new(),
      ),
    );

    Self {
      device_supports,
      known_models,
      default_support,
    }
  }

  /// Parses a support-config JSON document (the `whisperkit-coreml`
  /// `config.json` shape, e.g. `Tests/WhisperKitTests/Resources/
  /// config-v03.json` in the Swift repo): `{"name", "version",
  /// "device_support": [{"chips"?, "identifiers", "models": {"default",
  /// "supported"}}]}`. Hand-walks a [`serde_json::Value`] tree rather than
  /// deriving `Deserialize` — see this module's doc for why.
  ///
  /// # Errors
  /// [`ModelError::InvalidSupportConfig`] if `json` is not valid JSON, or
  /// is missing/mistypes `device_support`, `identifiers`, `models`,
  /// `default`, or `supported`.
  pub fn from_json(json: &str) -> Result<Self, ModelError> {
    let value: Value =
      serde_json::from_str(json).map_err(|e| ModelError::InvalidSupportConfig(e.to_string()))?;

    let entries = value
      .get("device_support")
      .and_then(Value::as_array)
      .ok_or_else(|| {
        ModelError::InvalidSupportConfig("missing `device_support` array".to_string())
      })?;

    let device_supports = entries
      .iter()
      .map(device_support_from_json)
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Self::from_device_supports(device_supports))
  }

  /// The bundled device→model support table Swift ships as
  /// `Constants.fallbackModelSupportConfig`
  /// (`WhisperKit/Core/Models.swift:1465-1662`), used whenever no remote
  /// `config.json` is available: six device tiers spanning everything from
  /// A12/S9-class devices to M2/M3/M4 Apple Silicon. Extracted
  /// mechanically (a small Python script over the Swift source; recorded
  /// in this task's report) rather than hand-transcribed, to avoid copy
  /// errors across the ~90 device identifiers and ~35 distinct model names
  /// involved.
  pub fn fallback() -> Self {
    Self::from_device_supports(fallback_device_supports())
  }

  /// Finds the [`ModelSupport`] for `device_identifier` by longest
  /// matching identifier **prefix** — `device_identifier` must start with
  /// a configured identifier, and the longest such identifier wins (so a
  /// specific `"Mac14,13"` entry outranks a generic `"Mac14"` entry for
  /// the query `"Mac14,13"`, while the generic entry still matches
  /// `"Mac14,2"`). Falls back to [`Self::default_support`]'s models when
  /// nothing matches. Ports `ModelSupportConfig.modelSupport(for:)`
  /// (`Models.swift:205-225`) exactly, including its own comment about why
  /// this must be prefix, not exact, matching (`iPad13,16` must match
  /// itself, not the shorter `iPad13,1`). Ties (multiple equal-length
  /// matching identifiers) keep the first one encountered in
  /// [`Self::device_supports_slice`] order, matching Swift's own
  /// array-order iteration (`deviceSupports: [DeviceSupport]`, not a
  /// `Set`) and its strict `>` update condition (`Models.swift:213`)
  /// exactly.
  pub fn support_for(&self, device_identifier: &str) -> ModelSupport {
    let mut best: Option<(&DeviceSupport, usize)> = None;
    for ds in &self.device_supports {
      for identifier in ds.identifiers_slice() {
        if !device_identifier.starts_with(identifier.as_str()) {
          continue;
        }
        let len = identifier.len();
        let better = match best {
          None => true,
          Some((_, best_len)) => len > best_len,
        };
        if better {
          best = Some((ds, len));
        }
      }
    }
    best.map_or_else(
      || self.default_support.models().clone(),
      |(ds, _)| ds.models().clone(),
    )
  }

  /// Every configured device tier, in the order [`Self::support_for`]
  /// scans them.
  #[inline(always)]
  pub const fn device_supports_slice(&self) -> &[DeviceSupport] {
    self.device_supports.as_slice()
  }

  /// Every model name known to this config (the union of every tier's
  /// supported models).
  #[inline(always)]
  pub const fn known_models_slice(&self) -> &[String] {
    self.known_models.as_slice()
  }

  /// The support used for a device identifier that matches no configured
  /// tier.
  #[inline(always)]
  pub const fn default_support(&self) -> &DeviceSupport {
    &self.default_support
  }
}

fn device_support_from_json(value: &Value) -> Result<DeviceSupport, ModelError> {
  let chips = value
    .get("chips")
    .and_then(Value::as_str)
    .map(str::to_string);
  let identifiers = string_array(value, "identifiers")?;
  let models_value = value.get("models").ok_or_else(|| {
    ModelError::InvalidSupportConfig("device support entry missing `models`".to_string())
  })?;
  let models = model_support_from_json(models_value)?;
  Ok(DeviceSupport::new(chips, identifiers, models))
}

fn model_support_from_json(value: &Value) -> Result<ModelSupport, ModelError> {
  let default_model = value
    .get("default")
    .and_then(Value::as_str)
    .ok_or_else(|| ModelError::InvalidSupportConfig("models entry missing `default`".to_string()))?
    .to_string();
  let supported = string_array(value, "supported")?;
  Ok(ModelSupport::new(default_model, supported, Vec::new()))
}

fn string_array(value: &Value, key: &str) -> Result<Vec<String>, ModelError> {
  value
    .get(key)
    .and_then(Value::as_array)
    .ok_or_else(|| ModelError::InvalidSupportConfig(format!("missing `{key}` array")))?
    .iter()
    .map(|entry| {
      entry
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| ModelError::InvalidSupportConfig(format!("`{key}` entry is not a string")))
    })
    .collect()
}

/// Builds a `Vec<String>` from string literals — a compact way to write
/// [`fallback_device_supports`]'s ~90 identifiers and ~35 model names
/// without a `.to_string()` on every one at every call site.
fn strs(items: &[&str]) -> Vec<String> {
  items.iter().map(|s| (*s).to_string()).collect()
}

// Extracted from `Sources/WhisperKit/Core/Models.swift`'s
// `Constants.fallbackModelSupportConfig` (lines 1465-1662) with a Python
// script (recorded in this task's report) that parses each `DeviceSupport(...)`
// call's `chips`/`identifiers`/`models: ModelSupport(default:supported:)`
// arguments out of the Swift source text and re-emits them as the Rust
// literals below — mechanical transcription, not hand-typed, precisely
// because a table this size is where hand-typing introduces copy errors.
fn fallback_device_supports() -> Vec<DeviceSupport> {
  vec![
    DeviceSupport::new(
      Some("A12, A13, S9, S10".to_string()),
      strs(&[
        "iPhone11", "iPhone12", "iPad12,1", "iPad12,2", "Watch7", "Watch8",
      ]),
      ModelSupport::new(
        "openai_whisper-tiny",
        strs(&[
          "openai_whisper-base",
          "openai_whisper-base.en",
          "openai_whisper-tiny",
          "openai_whisper-tiny.en",
        ]),
        Vec::new(),
      ),
    ),
    DeviceSupport::new(
      Some("A16, A17 Pro, A18".to_string()),
      strs(&[
        "iPhone15", "iPhone16", "iPhone17", "iPad15,7", "iPad15,8", "iPad16,1", "iPad16,2",
      ]),
      ModelSupport::new(
        "openai_whisper-base",
        strs(&[
          "openai_whisper-tiny",
          "openai_whisper-tiny.en",
          "openai_whisper-base",
          "openai_whisper-base.en",
          "openai_whisper-small",
          "openai_whisper-small.en",
          "openai_whisper-large-v2_949MB",
          "openai_whisper-large-v2_turbo_955MB",
          "openai_whisper-large-v3_947MB",
          "openai_whisper-large-v3_turbo_954MB",
          "distil-whisper_distil-large-v3_594MB",
          "distil-whisper_distil-large-v3_turbo_600MB",
          "openai_whisper-large-v3-v20240930_626MB",
          "openai_whisper-large-v3-v20240930_turbo_632MB",
        ]),
        Vec::new(),
      ),
    ),
    DeviceSupport::new(
      Some("M1".to_string()),
      strs(&[
        "MacBookPro17,1",
        "MacBookPro18,1",
        "MacBookPro18,2",
        "MacBookPro18,3",
        "MacBookPro18,4",
        "MacBookAir10,1",
        "Macmini9,1",
        "iMac21,1",
        "iMac21,2",
        "Mac13",
        "iPad13,4",
        "iPad13,5",
        "iPad13,6",
        "iPad13,7",
        "iPad13,8",
        "iPad13,9",
        "iPad13,10",
        "iPad13,11",
        "iPad13,16",
        "iPad13,17",
      ]),
      ModelSupport::new(
        "openai_whisper-large-v3-v20240930_626MB",
        strs(&[
          "openai_whisper-tiny",
          "openai_whisper-tiny.en",
          "openai_whisper-base",
          "openai_whisper-base.en",
          "openai_whisper-small",
          "openai_whisper-small.en",
          "openai_whisper-large-v2",
          "openai_whisper-large-v2_949MB",
          "openai_whisper-large-v3",
          "openai_whisper-large-v3_947MB",
          "distil-whisper_distil-large-v3",
          "distil-whisper_distil-large-v3_594MB",
          "openai_whisper-large-v3-v20240930_626MB",
        ]),
        Vec::new(),
      ),
    ),
    DeviceSupport::new(
      Some("M2, M3, M4".to_string()),
      strs(&[
        "Mac14",
        "Mac15",
        "Mac16",
        "iPad14,3",
        "iPad14,4",
        "iPad14,5",
        "iPad14,6",
        "iPad14,8",
        "iPad14,9",
        "iPad14,10",
        "iPad14,11",
        "iPad15",
        "iPad16",
      ]),
      ModelSupport::new(
        "openai_whisper-large-v3-v20240930",
        strs(&[
          "openai_whisper-tiny",
          "openai_whisper-tiny.en",
          "openai_whisper-base",
          "openai_whisper-base.en",
          "openai_whisper-small",
          "openai_whisper-small.en",
          "openai_whisper-large-v2",
          "openai_whisper-large-v2_949MB",
          "openai_whisper-large-v2_turbo",
          "openai_whisper-large-v2_turbo_955MB",
          "openai_whisper-large-v3",
          "openai_whisper-large-v3_947MB",
          "openai_whisper-large-v3_turbo",
          "openai_whisper-large-v3_turbo_954MB",
          "distil-whisper_distil-large-v3",
          "distil-whisper_distil-large-v3_594MB",
          "distil-whisper_distil-large-v3_turbo",
          "distil-whisper_distil-large-v3_turbo_600MB",
          "openai_whisper-large-v3-v20240930",
          "openai_whisper-large-v3-v20240930_turbo",
          "openai_whisper-large-v3-v20240930_626MB",
          "openai_whisper-large-v3-v20240930_turbo_632MB",
        ]),
        Vec::new(),
      ),
    ),
    DeviceSupport::new(
      Some("A14".to_string()),
      strs(&["iPhone13", "iPad13,1", "iPad13,2", "iPad13,18", "iPad13,19"]),
      ModelSupport::new(
        "openai_whisper-base",
        strs(&[
          "openai_whisper-tiny",
          "openai_whisper-tiny.en",
          "openai_whisper-base",
          "openai_whisper-base.en",
          "openai_whisper-small",
          "openai_whisper-small.en",
        ]),
        Vec::new(),
      ),
    ),
    DeviceSupport::new(
      Some("A15".to_string()),
      strs(&["iPhone14", "iPad14,1", "iPad14,2"]),
      ModelSupport::new(
        "openai_whisper-base",
        strs(&[
          "openai_whisper-tiny",
          "openai_whisper-tiny.en",
          "openai_whisper-base",
          "openai_whisper-base.en",
          "openai_whisper-small",
          "openai_whisper-small.en",
          "openai_whisper-large-v2_949MB",
          "openai_whisper-large-v2_turbo_955MB",
          "openai_whisper-large-v3_947MB",
          "openai_whisper-large-v3_turbo_954MB",
          "distil-whisper_distil-large-v3_594MB",
          "distil-whisper_distil-large-v3_turbo_600MB",
          "openai_whisper-large-v3-v20240930_626MB",
          "openai_whisper-large-v3-v20240930_turbo_632MB",
        ]),
        Vec::new(),
      ),
    ),
  ]
}

/// The device's hardware model identifier (e.g. `"Mac15,6"`), the string
/// [`SupportConfig::support_for`] expects — ports
/// `ProcessInfo.stringFromSysctl(named:)` applied to `hw.model`
/// (`ArgmaxCore/FoundationExtensions.swift:122-137`, the `hwModel` static),
/// using the exact two-call `sysctlbyname` protocol Swift itself uses:
/// query the required buffer size with a null buffer, then fill a buffer
/// of that size.
///
/// Falls back to `"unknown"` if either `sysctlbyname` call fails — Swift's
/// version has no failure path (a failed call there leaves `machineModel`
/// all zero bytes, silently decoding to `""`); this port makes that
/// "sysctl failed" case an explicit, documented sentinel instead of a
/// silent empty string.
pub fn device_identifier() -> String {
  let name = c"hw.model";
  let mut size: usize = 0;

  // SAFETY: `name` is `'static` and nul-terminated (a C string literal); a
  // null `oldp` with a valid `&mut size` is `sysctlbyname`'s documented
  // "report the required buffer size" mode, so no buffer is written
  // through — only `size` is, and `&mut usize` is always valid to write.
  let query = unsafe {
    libc::sysctlbyname(
      name.as_ptr(),
      std::ptr::null_mut(),
      &mut size,
      std::ptr::null_mut(),
      0,
    )
  };
  if query != 0 || size == 0 {
    return "unknown".to_string();
  }

  let mut buf = vec![0_u8; size];
  // SAFETY: `buf` is freshly allocated with exactly the `size` bytes the
  // query call above just reported, and `size` (passed back in by mutable
  // reference) tells `sysctlbyname` that same capacity — the matching
  // second half of the documented two-call protocol, so the write stays
  // within `buf`'s allocation.
  let fill = unsafe {
    libc::sysctlbyname(
      name.as_ptr(),
      buf.as_mut_ptr().cast(),
      &mut size,
      std::ptr::null_mut(),
      0,
    )
  };
  if fill != 0 {
    return "unknown".to_string();
  }

  let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
  String::from_utf8_lossy(&buf[..end]).into_owned()
}

// ---------------------------------------------------------------------
// ModelLoader / ResolvedModels / LocalModelLoader
// ---------------------------------------------------------------------

/// Seam for resolving the three CoreML model bundles a Whisper pipeline
/// loads, from *some* source — a local folder today ([`LocalModelLoader`]);
/// a Hub downloader can implement this trait later without changing
/// anything above it (spec §5.3's `ModelLoader` seam; shaped after
/// `MLModelLoading.swift` plus the local-resolve half of
/// `ModelDownloader.swift`). The paths [`ResolvedModels`] returns are
/// exactly what `crate::Model::load` takes — this module only resolves
/// paths, it never loads a model (that is Plan 3's `backend` module).
pub trait ModelLoader {
  /// Resolves the mel/encoder/decoder bundle paths from `folder`.
  ///
  /// # Errors
  /// Whatever the implementation uses to report a missing bundle —
  /// [`LocalModelLoader`] returns [`ModelError::NotFound`].
  fn resolve(&self, folder: &Path) -> Result<ResolvedModels, ModelError>;
}

/// The three CoreML bundle paths a Whisper pipeline loads, resolved by a
/// [`ModelLoader`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModels {
  mel: PathBuf,
  encoder: PathBuf,
  decoder: PathBuf,
}

impl ResolvedModels {
  /// Builds a resolved-bundle triple directly.
  pub fn new(
    mel: impl Into<PathBuf>,
    encoder: impl Into<PathBuf>,
    decoder: impl Into<PathBuf>,
  ) -> Self {
    Self {
      mel: mel.into(),
      encoder: encoder.into(),
      decoder: decoder.into(),
    }
  }

  /// The mel-spectrogram feature extractor bundle path.
  #[inline(always)]
  pub fn mel_ref(&self) -> &Path {
    self.mel.as_path()
  }

  /// The audio encoder bundle path.
  #[inline(always)]
  pub fn encoder_ref(&self) -> &Path {
    self.encoder.as_path()
  }

  /// The text decoder bundle path.
  #[inline(always)]
  pub fn decoder_ref(&self) -> &Path {
    self.decoder.as_path()
  }
}

/// [`ModelLoader`] that resolves the three bundles directly inside a given
/// local folder — no version/variant subdirectories, no download. The
/// everyday loader for models already on disk.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LocalModelLoader;

impl LocalModelLoader {
  /// Builds a local-folder loader.
  #[inline(always)]
  pub const fn new() -> Self {
    Self
  }
}

impl ModelLoader for LocalModelLoader {
  fn resolve(&self, folder: &Path) -> Result<ResolvedModels, ModelError> {
    let mel = detect_model_url(folder, "MelSpectrogram", false)?;
    let encoder = detect_model_url(folder, "AudioEncoder", false)?;
    let decoder = detect_model_url(folder, "TextDecoder", false)?;
    Ok(ResolvedModels::new(mel, encoder, decoder))
  }
}

// ---------------------------------------------------------------------
// ModelManager
// ---------------------------------------------------------------------

pub mod manager;

#[cfg(test)]
mod tests;
