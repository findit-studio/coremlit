//! The kit's error type, its `Result` alias, and the payloads its variants
//! carry.
//!
//! Every refusal names what it is about: the artifact, the function, the
//! field or the option. The vocabularies those names come from ([`Artifact`],
//! [`Function`], [`AllocationSite`], [`ConfigField`], …) live here beside the
//! variants that use them.

use std::path::{Path, PathBuf};

use crate::{
  CompileError, LoadError, PredictionError, TensorError,
  model::contract::{ContractViolation, Rendered},
};

use super::{
  artifacts::{ArtifactLabel, Part},
  audio::AudioPrecision,
  text::TextFunction,
  vision::{VisionBudget, VisionBudgets},
};

/// The long-text refusal payloads, shared with the crate's other text doors.
pub use crate::embeddings::long_text::{
  ContentlessInputOverBudget, InputTooLarge, WindowOverBudget,
};

/// The image geometry payloads, shared with the crate's other image doors.
pub use crate::embeddings::image::{ImageDataLength, ImageDimensions};

/// windit's own error, which [`Error::Windowing`] carries unchanged.
pub use windit::WinditError;

/// Convenience alias for `Result<T, `[`Error`]`>`.
pub type Result<T> = core::result::Result<T, Error>;

/// An artifact the kit reads, compiles or stamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Artifact {
  /// The text package or its compiled bundle, `EmbeddingGemma2Text`.
  TextBundle,
  /// The audio package or bundle of one precision: `EmbeddingGemma2Audio`
  /// (fp16) or `EmbeddingGemma2AudioFp32`.
  AudioBundle(AudioPrecision),
  /// The vision package or its compiled bundle, `EmbeddingGemma2Vision`.
  VisionBundle,
  /// The bf16 token table, `embeddings.bf16`.
  TokenTable,
  /// The f16 patch-position tables, `position_embeddings.f16`.
  PositionTable,
  /// The tokenizer image compiled from `tokenizer.json`,
  /// `tokenizer.v<format>-<12 hex>.gemmabpe`.
  TokenizerImage,
  /// The tokenizer's source, `tokenizer.json`.
  TokenizerSource,
  /// The model's `config.json`.
  Config,
  /// The directory's stamp, `coremlit-embeddinggemma2.stamp.json`.
  Stamp,
}

impl Artifact {
  /// The artifact's name as it appears in its directory: the package stem of
  /// a bundle, the file name of a loose file, the name pattern of the image.
  #[inline]
  pub const fn name(&self) -> &'static str {
    match self {
      Self::TextBundle => "EmbeddingGemma2Text",
      Self::AudioBundle(precision) => precision.package_stem(),
      Self::VisionBundle => "EmbeddingGemma2Vision",
      Self::TokenTable => "embeddings.bf16",
      Self::PositionTable => "position_embeddings.f16",
      Self::TokenizerImage => "tokenizer.v<format>-<12 hex>.gemmabpe",
      Self::TokenizerSource => "tokenizer.json",
      Self::Config => "config.json",
      Self::Stamp => "coremlit-embeddinggemma2.stamp.json",
    }
  }

  /// Whether the artifact is a model package (or its compiled bundle).
  #[inline]
  pub const fn is_bundle(&self) -> bool {
    matches!(
      self,
      Self::TextBundle | Self::AudioBundle(_) | Self::VisionBundle
    )
  }
}

/// A bundle prints `the EmbeddingGemma2Text bundle`; a file prints its name in
/// backticks, `` `embeddings.bf16` ``.
impl core::fmt::Display for Artifact {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    if self.is_bundle() {
      write!(f, "the {} bundle", self.name())
    } else {
      write!(f, "`{}`", self.name())
    }
  }
}

/// One function of a multifunction package: a text function or a vision
/// budget. The audio package has one function, and names none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Function {
  /// A function of the text package.
  Text(TextFunction),
  /// A function of the vision package.
  Vision(VisionBudget),
}

impl Function {
  /// The function's name in its package: `"embed_128"`, `"vision_140"`, ….
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Text(function) => function.as_str(),
      Self::Vision(budget) => budget.as_str(),
    }
  }
}

/// The function's name in its package.
impl core::fmt::Display for Function {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(self.as_str())
  }
}

/// ` (function `embed_128`)` for a named function, nothing for none.
struct InFunction(Option<Function>);

impl core::fmt::Display for InFunction {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self.0 {
      Some(function) => write!(f, " (function `{function}`)"),
      None => Ok(()),
    }
  }
}

/// A buffer the kit reserves fallibly, sized by its input or by an artifact.
/// Every such buffer on the kit's roads grows through one door, so a
/// reservation that cannot be made is answered by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum AllocationSite {
  /// The vectors a batch call returns, and the lists a batch keeps per
  /// member on the way: its sequences, its media rows, its result slots, and
  /// the tokens the facade's `embed_images` gathers.
  BatchOutput,
  /// The windows of a long text or an audio clip, and the lists kept per
  /// window.
  WindowList,
  /// A text's token ids, and a batch's list of them.
  TokenIds,
  /// A text's token offsets.
  TokenOffsets,
  /// The tokenizer's scalars, a piece's symbols and its merge heap.
  TokenizerScratch,
  /// The long-text measure's piece table, its joined prompt, and the image's
  /// bridging texts it reads.
  PieceTable,
  /// The layout of a batch's `pack_256` calls: the members' lengths, the
  /// packs and their members, and the sequences kept apart.
  PackLayout,
  /// The rows gathered or copied for one sequence: the token table's, or a
  /// media output's.
  GatherRows,
  /// An image's resized pixels and input planes.
  ImagePlanes,
  /// An audio clip's window plan.
  AudioPlan,
  /// A tokenizer image under construction.
  ImageBuild,
  /// The digest list a stamp records.
  StampDigests,
}

impl AllocationSite {
  /// Stable snake_case name of the site.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::BatchOutput => "batch_output",
      Self::WindowList => "window_list",
      Self::TokenIds => "token_ids",
      Self::TokenOffsets => "token_offsets",
      Self::TokenizerScratch => "tokenizer_scratch",
      Self::PieceTable => "piece_table",
      Self::PackLayout => "pack_layout",
      Self::GatherRows => "gather_rows",
      Self::ImagePlanes => "image_planes",
      Self::AudioPlan => "audio_plan",
      Self::ImageBuild => "image_build",
      Self::StampDigests => "stamp_digests",
    }
  }
}

/// A field of `config.json`'s contract, by its dotted path: `embed_scale`,
/// `audio.waveform_samples`, `functions.pack_256.slots`, … The top level is
/// the empty path, printed `(top level)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfigField(&'static str);

impl ConfigField {
  #[inline]
  pub(crate) const fn new(path: &'static str) -> Self {
    Self(path)
  }

  /// The field's dotted path; empty for the top level.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    self.0
  }
}

/// The dotted path, or `(top level)`.
impl core::fmt::Display for ConfigField {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    if self.0.is_empty() {
      f.write_str("(top level)")
    } else {
      f.write_str(self.0)
    }
  }
}

/// A part of a stamp that is malformed or of another format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum StampField {
  /// The document as a whole: not JSON, or not an object.
  Document,
  /// `format`: the stamp format.
  Format,
  /// `kit`: the kit the stamp is for.
  Kit,
  /// `coremlit`: the crate version that wrote it.
  CoremlitVersion,
  /// `os_build`: the OS build it was written on.
  OsBuild,
  /// `files`: the stamped loose files.
  Files,
  /// `bundles`: the current bundle of each package.
  Bundles,
  /// `tokenizer_image`: the current tokenizer image.
  TokenizerImage,
}

impl StampField {
  /// The field's key in the stamp, or `document` for the whole of it.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Document => "document",
      Self::Format => "format",
      Self::Kit => "kit",
      Self::CoremlitVersion => "coremlit",
      Self::OsBuild => "os_build",
      Self::Files => "files",
      Self::Bundles => "bundles",
      Self::TokenizerImage => "tokenizer_image",
    }
  }
}

/// A part of a tokenizer image that does not hold what the kit requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum ImageField {
  /// The file is shorter than its header or its sections.
  Length,
  /// The magic bytes.
  Magic,
  /// The format version.
  FormatVersion,
  /// The header size.
  HeaderSize,
  /// A section's offset, length or alignment.
  Section,
  /// One of the five counts.
  Count,
  /// The digest of the `tokenizer.json` the image was compiled from: its
  /// identity.
  SourceDigest,
  /// The digest of the image's sections.
  ImageDigest,
  /// One of the sentinel encodings checked at open.
  Sentinel,
}

impl ImageField {
  /// Stable snake_case name of the field.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Length => "length",
      Self::Magic => "magic",
      Self::FormatVersion => "format_version",
      Self::HeaderSize => "header_size",
      Self::Section => "section",
      Self::Count => "count",
      Self::SourceDigest => "source_digest",
      Self::ImageDigest => "image_digest",
      Self::Sentinel => "sentinel",
    }
  }
}

/// A part of `tokenizer.json` outside the one configuration the kit's
/// tokenizer implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum SourceField {
  /// The document as a whole: not JSON, or not the expected shape.
  Document,
  /// `normalizer`.
  Normalizer,
  /// `pre_tokenizer`.
  PreTokenizer,
  /// `model`: its type, flags or unknown token.
  Model,
  /// `model.vocab`.
  Vocabulary,
  /// `model.merges`.
  Merges,
  /// `post_processor`.
  PostProcessor,
  /// `added_tokens`.
  AddedTokens,
  /// The byte-fallback tokens `<0x00>` … `<0xFF>`.
  ByteFallback,
}

impl SourceField {
  /// Stable snake_case name of the field.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Document => "document",
      Self::Normalizer => "normalizer",
      Self::PreTokenizer => "pre_tokenizer",
      Self::Model => "model",
      Self::Vocabulary => "vocabulary",
      Self::Merges => "merges",
      Self::PostProcessor => "post_processor",
      Self::AddedTokens => "added_tokens",
      Self::ByteFallback => "byte_fallback",
    }
  }
}

/// A file-system operation [`Error::Io`] names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum IoOperation {
  /// Reading a file or a directory listing.
  Read,
  /// Creating or writing a file.
  Write,
  /// Copying a compiled bundle out of Core ML's temporary directory.
  Copy,
  /// Renaming a finished file to its name.
  Rename,
  /// Flushing a file to stable storage.
  Sync,
  /// Reading a file's metadata.
  Stat,
  /// Taking the directory's lock.
  Lock,
  /// Removing a file `prune` selected.
  Remove,
}

impl IoOperation {
  /// Stable snake_case name of the operation.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Read => "read",
      Self::Write => "write",
      Self::Copy => "copy",
      Self::Rename => "rename",
      Self::Sync => "sync",
      Self::Stat => "stat",
      Self::Lock => "lock",
      Self::Remove => "remove",
    }
  }
}

/// A cell of a table: a row and a column of the token table, or a table, a row
/// and a column of the position tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableCell {
  table: Option<u8>,
  row: u32,
  column: u32,
}

impl TableCell {
  /// A cell of the token table.
  #[inline]
  pub const fn token(row: u32, column: u32) -> Self {
    Self {
      table: None,
      row,
      column,
    }
  }

  /// A cell of one of the two position tables (0 for x, 1 for y).
  #[inline]
  pub const fn position(table: u8, row: u32, column: u32) -> Self {
    Self {
      table: Some(table),
      row,
      column,
    }
  }

  /// The position table the cell is in, or `None` for the token table.
  #[inline]
  pub const fn table(&self) -> Option<u8> {
    self.table
  }

  /// The cell's row.
  #[inline]
  pub const fn row(&self) -> u32 {
    self.row
  }

  /// The cell's column.
  #[inline]
  pub const fn column(&self) -> u32 {
    self.column
  }
}

/// `row 506, column 160`, or `table 1, row 0, column 767`.
impl core::fmt::Display for TableCell {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    if let Some(table) = self.table {
      write!(f, "table {table}, ")?;
    }
    write!(f, "row {}, column {}", self.row, self.column)
  }
}

/// Payload of [`Error::Load`]: Core ML refused a bundle or one of its
/// functions.
#[derive(Debug, thiserror::Error)]
#[error("Core ML refused to load {artifact}{}: {source}", InFunction(*.function))]
pub struct LoadRefusal {
  artifact: Artifact,
  function: Option<Function>,
  #[source]
  source: LoadError,
}

impl LoadRefusal {
  /// The refusal of `artifact`'s `function` (or its one function).
  #[inline]
  pub const fn new(artifact: Artifact, function: Option<Function>, source: LoadError) -> Self {
    Self {
      artifact,
      function,
      source,
    }
  }

  /// The bundle that was being loaded.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The function that was being loaded, or `None` for a one-function bundle.
  #[inline]
  pub const fn function(&self) -> Option<Function> {
    self.function
  }

  /// Core ML's refusal.
  #[inline]
  pub const fn load_error(&self) -> &LoadError {
    &self.source
  }
}

/// Payload of [`Error::Contract`]: a loaded function's I/O is not the kit's
/// contract. This is how a user's own conversion with other I/O is refused.
#[derive(Debug, thiserror::Error)]
#[error(
  "{artifact}{} does not keep the kit's contract at feature `{feature}`: expected {expected}, got {actual}",
  InFunction(*.function)
)]
pub struct ContractRefusal {
  artifact: Artifact,
  function: Option<Function>,
  feature: String,
  expected: String,
  actual: String,
}

impl ContractRefusal {
  /// `violation` of `artifact`'s `function` (or its one function), in the
  /// kit's vocabulary: the feature it names, what the kit sends or reads, and
  /// what the bundle has.
  pub(crate) fn of(
    artifact: Artifact,
    function: Option<Function>,
    violation: ContractViolation,
  ) -> Error {
    let (feature, expected, actual) = match violation.rendered() {
      Rendered::Feature(feature) => (
        feature.feature().to_owned(),
        feature.clone().expected(),
        feature.actual(),
      ),
      Rendered::UnsatisfiableInput(name) => (
        name,
        "no required input the kit does not send".to_owned(),
        "a required input".to_owned(),
      ),
      Rendered::UnsatisfiableState(name) => {
        (name, "no state".to_owned(), "a state buffer".to_owned())
      }
    };
    Error::Contract(Self::new(artifact, function, feature, expected, actual))
  }

  /// The refusal of `artifact`'s `function` at `feature`.
  #[inline]
  pub const fn new(
    artifact: Artifact,
    function: Option<Function>,
    feature: String,
    expected: String,
    actual: String,
  ) -> Self {
    Self {
      artifact,
      function,
      feature,
      expected,
      actual,
    }
  }

  /// The bundle whose function was checked.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The function that was checked, or `None` for a one-function bundle.
  #[inline]
  pub const fn function(&self) -> Option<Function> {
    self.function
  }

  /// The input or output feature the contract is about.
  #[inline]
  pub fn feature(&self) -> &str {
    &self.feature
  }

  /// What the contract states.
  #[inline]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// What the model declares.
  #[inline]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// Payload of [`Error::Config`]: `config.json` disagrees with the compiled
/// contract.
#[derive(Debug, thiserror::Error)]
#[error(
  "config.json disagrees with the kit's compiled contract at `{field}`: expected {expected}, got {actual}"
)]
pub struct ConfigMismatch {
  field: ConfigField,
  expected: String,
  actual: String,
}

impl ConfigMismatch {
  /// The mismatch at `field`.
  #[inline]
  pub const fn new(field: ConfigField, expected: String, actual: String) -> Self {
    Self {
      field,
      expected,
      actual,
    }
  }

  /// The field that disagrees.
  #[inline]
  pub const fn field(&self) -> ConfigField {
    self.field
  }

  /// The compiled contract's value, as JSON.
  #[inline]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// The file's value, as JSON, or `absent`.
  #[inline]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// Payload of [`Error::Read`]: an artifact the kit reads is missing,
/// unreadable or not parseable.
#[derive(Debug, thiserror::Error)]
#[error("could not read {artifact} at `{}`: {source}", .path.display())]
pub struct ReadFailure {
  artifact: Artifact,
  path: PathBuf,
  #[source]
  source: std::io::Error,
}

impl ReadFailure {
  /// The failure to read `artifact` at `path`. A parse failure is carried as
  /// the I/O error its parser converts it to:
  /// [`std::io::ErrorKind::InvalidData`] for malformed text,
  /// [`std::io::ErrorKind::UnexpectedEof`] for a document cut short.
  #[inline]
  pub const fn new(artifact: Artifact, path: PathBuf, source: std::io::Error) -> Self {
    Self {
      artifact,
      path,
      source,
    }
  }

  /// The artifact that was being read.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The path that was read.
  #[inline]
  pub fn path(&self) -> &Path {
    &self.path
  }

  /// The I/O or parse failure.
  #[inline]
  pub const fn io_error(&self) -> &std::io::Error {
    &self.source
  }
}

/// Payload of [`Error::Unstamped`]: the directory's stamp lists nothing for an
/// artifact a part needs.
#[derive(Debug, thiserror::Error)]
#[error(
  "the stamp in `{}` lists nothing for {artifact}: run `compile_artifacts` over the directory first",
  .dir.display()
)]
pub struct Unstamped {
  artifact: Artifact,
  dir: PathBuf,
}

impl Unstamped {
  /// `artifact` is missing from the stamp in `dir`.
  #[inline]
  pub const fn new(artifact: Artifact, dir: PathBuf) -> Self {
    Self { artifact, dir }
  }

  /// The artifact the stamp does not list.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The directory whose stamp was read.
  #[inline]
  pub fn dir(&self) -> &Path {
    &self.dir
  }
}

/// Payload of [`Error::StaleStamp`]: a stamped file's size changed since it
/// was stamped.
#[derive(Debug, thiserror::Error)]
#[error(
  "{artifact} has changed since it was stamped: the stamp recorded {recorded} bytes and the file has {actual}; run `compile_artifacts` over the directory again"
)]
pub struct StaleStamp {
  artifact: Artifact,
  recorded: u64,
  actual: u64,
}

impl StaleStamp {
  /// `artifact` was stamped at `recorded` bytes and is now `actual`.
  #[inline]
  pub const fn new(artifact: Artifact, recorded: u64, actual: u64) -> Self {
    Self {
      artifact,
      recorded,
      actual,
    }
  }

  /// The artifact whose size moved.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The size the stamp recorded.
  #[inline]
  pub const fn recorded(&self) -> u64 {
    self.recorded
  }

  /// The size the file has now.
  #[inline]
  pub const fn actual(&self) -> u64 {
    self.actual
  }
}

/// Payload of [`Error::Stamp`]: a malformed stamp, or one of another format.
#[derive(Debug, thiserror::Error)]
#[error("the stamp is malformed at `{field}`: {detail}")]
pub struct StampDefect {
  field: StampField,
  detail: String,
}

impl StampDefect {
  /// The defect at `field`.
  #[inline]
  pub const fn new(field: StampField, detail: String) -> Self {
    Self { field, detail }
  }

  /// The part of the stamp that is malformed.
  #[inline]
  pub const fn field(&self) -> StampField {
    self.field
  }

  /// What is wrong with it.
  #[inline]
  pub fn detail(&self) -> &str {
    &self.detail
  }
}

/// Payload of [`Error::TableShape`]: a table is not the model's size.
#[derive(Debug, thiserror::Error)]
#[error("{artifact} is {actual} bytes; the model's table is {expected}")]
pub struct TableShape {
  artifact: Artifact,
  expected: u64,
  actual: u64,
}

impl TableShape {
  /// `artifact` is `actual` bytes against the model's `expected`.
  #[inline]
  pub const fn new(artifact: Artifact, expected: u64, actual: u64) -> Self {
    Self {
      artifact,
      expected,
      actual,
    }
  }

  /// The table.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The model's table size, in bytes.
  #[inline]
  pub const fn expected(&self) -> u64 {
    self.expected
  }

  /// The file's size, in bytes.
  #[inline]
  pub const fn actual(&self) -> u64 {
    self.actual
  }
}

/// Payload of [`Error::TableIdentity`]: a sentinel cell differs from the one
/// the stamp recorded from the same file.
#[derive(Debug, thiserror::Error)]
#[error(
  "{artifact} differs from its stamp at {cell}: the stamp recorded {expected:#06x}, the file holds {actual:#06x}"
)]
pub struct TableIdentity {
  artifact: Artifact,
  cell: TableCell,
  expected: u16,
  actual: u16,
}

impl TableIdentity {
  /// `artifact`'s `cell` holds `actual` bits where the stamp recorded
  /// `expected`.
  #[inline]
  pub const fn new(artifact: Artifact, cell: TableCell, expected: u16, actual: u16) -> Self {
    Self {
      artifact,
      cell,
      expected,
      actual,
    }
  }

  /// The table.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The sentinel cell.
  #[inline]
  pub const fn cell(&self) -> TableCell {
    self.cell
  }

  /// The bits the stamp recorded.
  #[inline]
  pub const fn expected(&self) -> u16 {
    self.expected
  }

  /// The bits the file holds.
  #[inline]
  pub const fn actual(&self) -> u16 {
    self.actual
  }
}

/// Payload of [`Error::TokenizerImage`]: the tokenizer image is malformed, of
/// another format, or compiled from another `tokenizer.json`.
#[derive(Debug, thiserror::Error)]
#[error("the tokenizer image's {field} is wrong: expected {expected}, got {actual}")]
pub struct TokenizerImageDefect {
  field: ImageField,
  expected: String,
  actual: String,
}

impl TokenizerImageDefect {
  /// The defect at `field`.
  #[inline]
  pub const fn new(field: ImageField, expected: String, actual: String) -> Self {
    Self {
      field,
      expected,
      actual,
    }
  }

  /// The part of the image that is wrong.
  #[inline]
  pub const fn field(&self) -> ImageField {
    self.field
  }

  /// What the kit requires.
  #[inline]
  pub fn expected(&self) -> &str {
    &self.expected
  }

  /// What the image holds.
  #[inline]
  pub fn actual(&self) -> &str {
    &self.actual
  }
}

/// Payload of [`Error::TokenizerSource`]: `tokenizer.json` is outside the one
/// configuration the kit's tokenizer implements.
#[derive(Debug, thiserror::Error)]
#[error("tokenizer.json is outside the configuration the kit implements at `{field}`: {detail}")]
pub struct TokenizerSourceDefect {
  field: SourceField,
  detail: String,
}

impl TokenizerSourceDefect {
  /// The defect at `field`.
  #[inline]
  pub const fn new(field: SourceField, detail: String) -> Self {
    Self { field, detail }
  }

  /// The part of `tokenizer.json` that is outside the configuration.
  #[inline]
  pub const fn field(&self) -> SourceField {
    self.field
  }

  /// What it holds, and what the kit requires.
  #[inline]
  pub fn detail(&self) -> &str {
    &self.detail
  }
}

/// Payload of [`Error::Compile`]: Core ML could not compile a package.
#[derive(Debug, thiserror::Error)]
#[error("Core ML could not compile {artifact}: {source}")]
pub struct CompileFailure {
  artifact: Artifact,
  #[source]
  source: CompileError,
}

impl CompileFailure {
  /// Core ML's failure to compile `artifact`.
  #[inline]
  pub const fn new(artifact: Artifact, source: CompileError) -> Self {
    Self { artifact, source }
  }

  /// The package.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// Core ML's failure.
  #[inline]
  pub const fn compile_error(&self) -> &CompileError {
    &self.source
  }
}

/// Payload of [`Error::AmbiguousArtifact`]: one artifact has two sources in a
/// directory, so neither is chosen.
#[derive(Debug, thiserror::Error)]
#[error(
  "{artifact} has more than one source in the directory ({}): remove all but one; none is chosen",
  Candidates(.candidates)
)]
pub struct AmbiguousArtifact {
  artifact: Artifact,
  candidates: Vec<PathBuf>,
}

impl AmbiguousArtifact {
  /// `artifact`'s `candidates`.
  #[inline]
  pub const fn new(artifact: Artifact, candidates: Vec<PathBuf>) -> Self {
    Self {
      artifact,
      candidates,
    }
  }

  /// The artifact.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The paths that each could be its source.
  #[inline]
  pub fn candidates(&self) -> &[PathBuf] {
    &self.candidates
  }
}

/// `` `a`, `b` ``.
struct Candidates<'a>(&'a [PathBuf]);

impl core::fmt::Display for Candidates<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    for (index, path) in self.0.iter().enumerate() {
      if index > 0 {
        f.write_str(", ")?;
      }
      write!(f, "`{}`", path.display())?;
    }
    Ok(())
  }
}

/// Payload of [`Error::Io`]: a file-system operation of `compile_artifacts`
/// or `prune` failed.
#[derive(Debug, thiserror::Error)]
#[error("could not {operation} `{}`: {source}", .path.display())]
pub struct IoFailure {
  operation: IoOperation,
  path: PathBuf,
  #[source]
  source: std::io::Error,
}

impl IoFailure {
  /// `operation` on `path` failed with `source`.
  #[inline]
  pub const fn new(operation: IoOperation, path: PathBuf, source: std::io::Error) -> Self {
    Self {
      operation,
      path,
      source,
    }
  }

  /// The operation.
  #[inline]
  pub const fn operation(&self) -> IoOperation {
    self.operation
  }

  /// The path it was applied to.
  #[inline]
  pub fn path(&self) -> &Path {
    &self.path
  }

  /// The operating system's error.
  #[inline]
  pub const fn io_error(&self) -> &std::io::Error {
    &self.source
  }
}

/// Payload of [`Error::PromptFillsWindow`]: the prompt's framing leaves no id
/// of the window for the text.
#[derive(Debug, thiserror::Error)]
#[error(
  "the prompt's framing takes {framing} ids of the {window}-id window, leaving none for the text"
)]
pub struct PromptFillsWindow {
  window: usize,
  framing: usize,
}

impl PromptFillsWindow {
  /// A `window` of total ids against a prompt framing of `framing` ids.
  #[inline]
  pub const fn new(window: usize, framing: usize) -> Self {
    Self { window, framing }
  }

  /// The window, in total ids.
  #[inline]
  pub const fn window(&self) -> usize {
    self.window
  }

  /// The prompt's framing: `<bos>`, `<eos>` and the prompt's own ids.
  #[inline]
  pub const fn framing(&self) -> usize {
    self.framing
  }
}

/// Payload of [`Error::Resize`]: pixon's resampler refused the resize.
#[derive(Debug, thiserror::Error)]
#[error("the image resize failed: {source}")]
pub struct ResizeFailure {
  #[source]
  source: pixon::sinker::MixedSinkerError,
}

impl ResizeFailure {
  /// pixon's refusal.
  #[inline]
  pub const fn new(source: pixon::sinker::MixedSinkerError) -> Self {
    Self { source }
  }

  /// pixon's error.
  #[inline]
  pub const fn pixon_error(&self) -> &pixon::sinker::MixedSinkerError {
    &self.source
  }
}

/// Payload of [`Error::AudioTooShort`]: the window completes no mel frame.
#[derive(Debug, thiserror::Error)]
#[error("an audio window of {samples} samples completes no mel frame: it needs at least {min}")]
pub struct AudioTooShort {
  samples: usize,
  min: usize,
}

impl AudioTooShort {
  /// A window of `samples` against the minimum `min`.
  #[inline]
  pub const fn new(samples: usize, min: usize) -> Self {
    Self { samples, min }
  }

  /// The window's samples.
  #[inline]
  pub const fn samples(&self) -> usize {
    self.samples
  }

  /// The fewest samples that complete a mel frame.
  #[inline]
  pub const fn min(&self) -> usize {
    self.min
  }
}

/// Payload of [`Error::AudioWindowTooLong`]: more samples than one window
/// holds.
#[derive(Debug, thiserror::Error)]
#[error("an audio window holds at most {max} samples, got {samples}")]
pub struct AudioWindowTooLong {
  samples: usize,
  max: usize,
}

impl AudioWindowTooLong {
  /// A window of `samples` against the maximum `max`.
  #[inline]
  pub const fn new(samples: usize, max: usize) -> Self {
    Self { samples, max }
  }

  /// The window's samples.
  #[inline]
  pub const fn samples(&self) -> usize {
    self.samples
  }

  /// The most samples one window holds.
  #[inline]
  pub const fn max(&self) -> usize {
    self.max
  }
}

/// Payload of [`Error::BudgetNotLoaded`]: a budget outside the instance's
/// loaded set.
#[derive(Debug, thiserror::Error)]
#[error(
  "vision budget {budget} is not among the loaded budgets {loaded}; a budget is never loaded on the side"
)]
pub struct BudgetNotLoaded {
  budget: VisionBudget,
  loaded: VisionBudgets,
}

impl BudgetNotLoaded {
  /// `budget` was asked for; `loaded` are the instance's budgets.
  #[inline]
  pub const fn new(budget: VisionBudget, loaded: VisionBudgets) -> Self {
    Self { budget, loaded }
  }

  /// The budget asked for.
  #[inline]
  pub const fn budget(&self) -> VisionBudget {
    self.budget
  }

  /// The budgets the instance loads.
  #[inline]
  pub const fn loaded(&self) -> VisionBudgets {
    self.loaded
  }
}

/// Payload of [`Error::NonFiniteOutput`]: a model fault, not caller data.
#[derive(Debug, thiserror::Error)]
#[error(
  "{artifact}{} produced a non-finite value at index {index} of a row it was asked for",
  InFunction(*.function)
)]
pub struct NonFiniteOutput {
  artifact: Artifact,
  function: Option<Function>,
  index: usize,
}

impl NonFiniteOutput {
  /// `artifact`'s `function` produced a non-finite value at `index`.
  #[inline]
  pub const fn new(artifact: Artifact, function: Option<Function>, index: usize) -> Self {
    Self {
      artifact,
      function,
      index,
    }
  }

  /// The bundle that ran.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The function that ran, or `None` for a one-function bundle.
  #[inline]
  pub const fn function(&self) -> Option<Function> {
    self.function
  }

  /// The index of the value within the row.
  #[inline]
  pub const fn index(&self) -> usize {
    self.index
  }
}

/// Payload of [`Error::Prediction`]: Core ML's prediction failed.
#[derive(Debug, thiserror::Error)]
#[error("Core ML's prediction failed in {artifact}{}: {source}", InFunction(*.function))]
pub struct PredictionFailure {
  artifact: Artifact,
  function: Option<Function>,
  #[source]
  source: PredictionError,
}

impl PredictionFailure {
  /// `artifact`'s `function` failed with `source`.
  #[inline]
  pub const fn new(
    artifact: Artifact,
    function: Option<Function>,
    source: PredictionError,
  ) -> Self {
    Self {
      artifact,
      function,
      source,
    }
  }

  /// The bundle that ran.
  #[inline]
  pub const fn artifact(&self) -> Artifact {
    self.artifact
  }

  /// The function that ran, or `None` for a one-function bundle.
  #[inline]
  pub const fn function(&self) -> Option<Function> {
    self.function
  }

  /// Core ML's failure.
  #[inline]
  pub const fn prediction_error(&self) -> &PredictionError {
    &self.source
  }
}

/// Payload of [`Error::Allocation`]: a fallible reservation was refused.
#[derive(Debug, thiserror::Error)]
#[error("could not reserve {bytes} bytes for {site}")]
pub struct Allocation {
  site: AllocationSite,
  bytes: usize,
}

impl Allocation {
  /// The reservation of `bytes` at `site` was refused.
  #[inline]
  pub const fn new(site: AllocationSite, bytes: usize) -> Self {
    Self { site, bytes }
  }

  /// The buffer.
  #[inline]
  pub const fn site(&self) -> AllocationSite {
    self.site
  }

  /// The bytes asked for.
  #[inline]
  pub const fn bytes(&self) -> usize {
    self.bytes
  }
}

/// Payload of [`Error::ArtifactIdentity`]: a part's artifacts are not the ones
/// the deployment pinned.
#[derive(Debug, thiserror::Error)]
#[error("the {part} part's artifacts are {actual}, but the deployment expects {expected}")]
pub struct ArtifactIdentity {
  part: Part,
  expected: ArtifactLabel,
  actual: ArtifactLabel,
}

impl ArtifactIdentity {
  /// `part`'s stamp yields `actual` where the deployment pinned `expected`.
  #[inline]
  pub const fn new(part: Part, expected: ArtifactLabel, actual: ArtifactLabel) -> Self {
    Self {
      part,
      expected,
      actual,
    }
  }

  /// The part.
  #[inline]
  pub const fn part(&self) -> Part {
    self.part
  }

  /// The label the deployment pinned (`expect_artifacts`).
  #[inline]
  pub const fn expected(&self) -> ArtifactLabel {
    self.expected
  }

  /// The label the part's stamp yields.
  #[inline]
  pub const fn actual(&self) -> ArtifactLabel {
    self.actual
  }
}

/// The kit's error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// The OS predates macOS 15: the packages are Core ML specification 9 and
  /// are loaded one function at a time.
  #[error(
    "EmbeddingGemma 2 needs macOS 15 or later: its packages are Core ML specification 9 and load one function at a time"
  )]
  UnsupportedOs,

  /// Core ML refused a bundle or a function; [`LoadError::FunctionsUnsupported`]
  /// and [`LoadError::OptionUnsupported`] included.
  #[error(transparent)]
  Load(LoadRefusal),

  /// A loaded function's I/O is not the kit's contract.
  #[error(transparent)]
  Contract(ContractRefusal),

  /// `config.json` disagrees with the compiled contract.
  #[error(transparent)]
  Config(ConfigMismatch),

  /// An artifact is missing, unreadable or not parseable.
  #[error(transparent)]
  Read(ReadFailure),

  /// The directory's stamp lists nothing for an artifact a part needs.
  #[error(transparent)]
  Unstamped(Unstamped),

  /// A stamped file's size changed since it was stamped.
  #[error(transparent)]
  StaleStamp(StaleStamp),

  /// A malformed stamp, or one of another format.
  #[error(transparent)]
  Stamp(StampDefect),

  /// A table is not the model's size.
  #[error(transparent)]
  TableShape(TableShape),

  /// A table's sentinel cell differs from the stamp's.
  #[error(transparent)]
  TableIdentity(TableIdentity),

  /// The tokenizer image is malformed, of another format, or compiled from
  /// another `tokenizer.json`.
  #[error(transparent)]
  TokenizerImage(TokenizerImageDefect),

  /// `tokenizer.json` is outside the one configuration the kit implements.
  #[error(transparent)]
  TokenizerSource(TokenizerSourceDefect),

  /// Core ML could not compile a package.
  #[error(transparent)]
  Compile(CompileFailure),

  /// One artifact has two sources in a directory: a package beside a caller's
  /// plainly named bundle. Neither is chosen.
  #[error(transparent)]
  AmbiguousArtifact(AmbiguousArtifact),

  /// A file-system operation of `compile_artifacts` or `prune` failed.
  #[error(transparent)]
  Io(IoFailure),

  /// The text is empty.
  #[error("text input is empty")]
  EmptyText,

  /// The text exceeds the configured input byte limit.
  #[error(
    "text input is {} bytes, exceeding the configured {}-byte limit",
    .0.got(),
    .0.max()
  )]
  InputTooLarge(InputTooLarge),

  /// The configured window exceeds the model's 512 ids.
  #[error(
    "embed_long window budget {} exceeds the model's fixed {}-token input window",
    .0.window(),
    .0.max()
  )]
  WindowOverBudget(WindowOverBudget),

  /// The prompt's framing leaves no id of the window for the text.
  #[error(transparent)]
  PromptFillsWindow(PromptFillsWindow),

  /// A contentless run that must be embedded whole measures past 512 ids.
  #[error(
    "contentless text at bytes {}..{} measures {} tokens, exceeding the model's fixed {}-token window",
    .0.start(),
    .0.end(),
    .0.tokens(),
    .0.max()
  )]
  ContentlessInputOverBudget(ContentlessInputOverBudget),

  /// Chunking or aggregation refused: windit's own typed error, unchanged.
  #[error("windowed-sequence processing failed: {0}")]
  Windowing(#[from] WinditError),

  /// The audio window completes no mel frame.
  #[error(transparent)]
  AudioTooShort(AudioTooShort),

  /// More samples than one audio window holds.
  #[error(transparent)]
  AudioWindowTooLong(AudioWindowTooLong),

  /// A zero side, a side over the axis cap, or a target whose sides both
  /// round to zero.
  #[error(
    "invalid image dimensions: {}×{} (a zero side, a side over the per-axis maximum, or both target sides rounding to zero)",
    .0.width(),
    .0.height()
  )]
  ImageDimensions(ImageDimensions),

  /// The image buffer is not `width · height · 3` bytes.
  #[error(
    "image data length mismatch: expected {} bytes (w·h·3), got {}",
    .0.expected(),
    .0.got()
  )]
  ImageBuffer(ImageDataLength),

  /// pixon refused the resize.
  #[error(transparent)]
  Resize(ResizeFailure),

  /// A budget outside the instance's loaded set.
  #[error(transparent)]
  BudgetNotLoaded(BudgetNotLoaded),

  /// A facade call into a lane whose options are absent: the facade loaded no
  /// part for it. Raised before any work; the parts themselves never raise
  /// it.
  #[error("the {0} lane is off: the options enable no {0} part, so the facade loaded none")]
  PartDisabled(Part),

  /// NaN or ±∞ in an output row the kit reads: a model fault.
  #[error(transparent)]
  NonFiniteOutput(NonFiniteOutput),

  /// Core ML's prediction failed.
  #[error(transparent)]
  Prediction(PredictionFailure),

  /// A `MultiArray` allocation or view failed.
  #[error("tensor failed: {0}")]
  Tensor(#[from] TensorError),

  /// A fallible reservation was refused.
  #[error(transparent)]
  Allocation(Allocation),

  /// A part's artifacts are not the ones the deployment pinned
  /// (`expect_artifacts`).
  #[error(transparent)]
  ArtifactIdentity(ArtifactIdentity),
}

impl From<crate::embeddings::long_text::Error> for Error {
  fn from(e: crate::embeddings::long_text::Error) -> Self {
    use crate::embeddings::long_text::Error as LongText;
    match e {
      LongText::Windowing(e) => Self::Windowing(e),
      LongText::WindowOverBudget(e) => Self::WindowOverBudget(e),
      LongText::InputTooLarge(e) => Self::InputTooLarge(e),
      LongText::ContentlessInputOverBudget(e) => Self::ContentlessInputOverBudget(e),
    }
  }
}

#[cfg(test)]
mod tests;
