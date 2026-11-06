//! The options of each part and of the facade, their defaults, and the
//! derivation terms a part spells from them.
//!
//! Every choice the model and Core ML allow is an option with an opinionated
//! default. The declaration states the default, the deployment document may
//! override it, and an option that changes vectors enters the part's
//! [`DerivationTerms`].
//!
//! # Wire form
//!
//! Under the `serde` feature each options type is a document whose keys all
//! default, so `{}` is [`TextOptions::new`]. Unknown keys are refused at every
//! level, so a misspelled `split_special_token` is an error naming the key
//! rather than a silent default. `deny_unknown_fields` and `flatten` do not
//! compose, so every composed value (`coreml`, `long_text`, `fold`) is nested
//! under its own key. A `coreml` table that is present follows
//! [`LoadOptions`]' own form, in which `compute` is required; an absent one is
//! the part's default.

use std::{
  num::NonZeroU32,
  path::{Path, PathBuf},
};

use crate::{
  ComputeUnits, LoadOptions,
  embeddings::{aggregate::AggregatePolicyKind, long_text::LongTextOptions},
};

use super::{
  artifacts::ArtifactLabel,
  audio::AudioPrecision,
  config::MAX_TOKENS,
  error::{BudgetNotLoaded, Error, Result, WindowOverBudget},
  placement::{PACKED_SLOT_INVARIANT, PlacementTable},
  text::TextFunctions,
  tokenizer::SplitSpecialTokens,
  vision::{ResizeRoad, VisionBudget, VisionBudgets},
};

/// How a batch of sequences is embedded by the text functions.
///
/// One option per face (text, audio, image), each settable by declaration,
/// document or per call, and each entering the face's derivation terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Batching {
  /// Sequences of at most 256 ids go through `pack_256`, first-fit in input
  /// order, at most eight per call; a batch of one too, alone at slot 0.
  /// Longer sequences take `embed_512`. A pack call never mixes faces.
  Packed,
  /// Every sequence takes the smallest `embed_S` that holds it, alone.
  Single,
}

impl Batching {
  /// Stable snake_case name: `"packed"` or `"single"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Packed => "packed",
      Self::Single => "single",
    }
  }
}

/// The text part's Core ML choices: the Neural Engine, Core ML's defaults
/// otherwise.
pub const DEFAULT_TEXT_COREML: LoadOptions = LoadOptions::new(ComputeUnits::CpuAndNeuralEngine);
/// The audio part's Core ML choices: the GPU, Core ML's defaults otherwise.
/// The Neural Engine computes in fp16 only, which would defeat the fp32
/// precision.
pub const DEFAULT_AUDIO_COREML: LoadOptions = LoadOptions::new(ComputeUnits::CpuAndGpu);
/// The vision part's Core ML choices: the GPU, Core ML's defaults otherwise.
pub const DEFAULT_VISION_COREML: LoadOptions = LoadOptions::new(ComputeUnits::CpuAndGpu);
/// Text packs by default.
pub const DEFAULT_TEXT_BATCHING: Batching = Batching::Packed;
/// Audio windows are embedded one by one by default.
pub const DEFAULT_AUDIO_BATCHING: Batching = Batching::Single;
/// Images are embedded one by one by default.
pub const DEFAULT_VISION_BATCHING: Batching = Batching::Single;
/// Special literals are recognized as their special ids by default, as
/// Hugging Face and sentence-transformers read them.
pub const DEFAULT_SPLIT_SPECIAL_TOKENS: bool = false;
/// Long-text windows fold by their coverage.
pub const DEFAULT_FOLD: AggregatePolicyKind = AggregatePolicyKind::CoverageWeightedMean;
/// The fp16 audio package runs by default.
pub const DEFAULT_AUDIO_PRECISION: AudioPrecision = AudioPrecision::Fp16;
/// Audio windows follow one another without overlap: a hop of one window,
/// 160,000 samples (10 s at 16 kHz).
pub const DEFAULT_HOP_SAMPLES: NonZeroU32 = NonZeroU32::new(160_000).unwrap();
/// A final audio window shorter than 8,000 samples (0.5 s) is dropped, unless
/// it is the only window.
pub const DEFAULT_MIN_TAIL_SAMPLES: u32 = 8_000;
/// The vision budget a call without its own uses: 140 tokens.
pub const DEFAULT_BUDGET: VisionBudget = VisionBudget::Tokens140;
/// The vision budgets an image instance loads: 140 alone.
pub const DEFAULT_BUDGETS: VisionBudgets = VisionBudgets::only(VisionBudget::Tokens140);
/// The resize road: Hugging Face's bicubic.
pub const DEFAULT_RESIZE: ResizeRoad = ResizeRoad::HfBicubic;

#[cfg(feature = "serde")]
mod serde_defaults {
  use super::*;

  pub(super) fn text_coreml() -> LoadOptions {
    DEFAULT_TEXT_COREML
  }
  pub(super) fn audio_coreml() -> LoadOptions {
    DEFAULT_AUDIO_COREML
  }
  pub(super) fn vision_coreml() -> LoadOptions {
    DEFAULT_VISION_COREML
  }
  pub(super) fn text_batching() -> Batching {
    DEFAULT_TEXT_BATCHING
  }
  pub(super) fn audio_batching() -> Batching {
    DEFAULT_AUDIO_BATCHING
  }
  pub(super) fn vision_batching() -> Batching {
    DEFAULT_VISION_BATCHING
  }
  pub(super) fn split_special_tokens() -> bool {
    DEFAULT_SPLIT_SPECIAL_TOKENS
  }
  pub(super) fn fold() -> AggregatePolicyKind {
    DEFAULT_FOLD
  }
  pub(super) fn audio_precision() -> AudioPrecision {
    DEFAULT_AUDIO_PRECISION
  }
  pub(super) fn hop_samples() -> NonZeroU32 {
    DEFAULT_HOP_SAMPLES
  }
  pub(super) fn min_tail_samples() -> u32 {
    DEFAULT_MIN_TAIL_SAMPLES
  }
  pub(super) fn budgets() -> VisionBudgets {
    DEFAULT_BUDGETS
  }
  pub(super) fn budget() -> VisionBudget {
    DEFAULT_BUDGET
  }
  pub(super) fn resize() -> ResizeRoad {
    DEFAULT_RESIZE
  }
}

/// `none`, or the path in quotes.
struct DirTerm<'a>(Option<&'a Path>);

impl core::fmt::Display for DirTerm<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self.0 {
      Some(dir) => write!(f, "{dir:?}"),
      None => f.write_str("none"),
    }
  }
}

/// `none`, or the value's own `Display`.
struct OptionTerm<T>(Option<T>);

impl<T: core::fmt::Display> core::fmt::Display for OptionTerm<T> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match &self.0 {
      Some(value) => write!(f, "{value}"),
      None => f.write_str("none"),
    }
  }
}

/// Options of the text part (rust-options-pattern).
///
/// Presence is semantic: an absent `dir` is the facade's root, an absent
/// `prewarm` is every function the configuration can reach, and an absent
/// `expect_artifacts` pins nothing.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct TextOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  dir: Option<PathBuf>,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::text_coreml"))]
  coreml: LoadOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  prewarm: Option<TextFunctions>,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::text_batching"))]
  batching: Batching,
  #[cfg_attr(
    feature = "serde",
    serde(default = "serde_defaults::split_special_tokens")
  )]
  split_special_tokens: bool,
  #[cfg_attr(feature = "serde", serde(default))]
  long_text: LongTextOptions,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::fold"))]
  fold: AggregatePolicyKind,
  #[cfg_attr(feature = "serde", serde(default))]
  expect_artifacts: Option<ArtifactLabel>,
}

impl Default for TextOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl TextOptions {
  /// The defaults: the facade's root, [`DEFAULT_TEXT_COREML`], every
  /// reachable function warmed, [`DEFAULT_TEXT_BATCHING`], special literals
  /// recognized, a 512-id window with no overlap, [`DEFAULT_FOLD`], and no
  /// pinned artifact label.
  pub const fn new() -> Self {
    Self {
      dir: None,
      coreml: DEFAULT_TEXT_COREML,
      prewarm: None,
      batching: DEFAULT_TEXT_BATCHING,
      split_special_tokens: DEFAULT_SPLIT_SPECIAL_TOKENS,
      long_text: LongTextOptions::new(),
      fold: DEFAULT_FOLD,
      expect_artifacts: None,
    }
  }

  /// The part's directory, or `None` for the facade's root.
  #[inline]
  pub fn dir(&self) -> Option<&Path> {
    self.dir.as_deref()
  }

  /// Builder form of [`Self::set_dir`].
  #[must_use]
  pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
    self.set_dir(Some(dir.into()));
    self
  }

  /// Sets [`Self::dir`]; `None` is the facade's root.
  pub fn set_dir(&mut self, dir: Option<PathBuf>) -> &mut Self {
    self.dir = dir;
    self
  }

  /// The Core ML choices the text functions load with.
  #[inline]
  pub const fn coreml(&self) -> LoadOptions {
    self.coreml
  }

  /// Builder form of [`Self::set_coreml`].
  #[must_use]
  pub fn with_coreml(mut self, coreml: LoadOptions) -> Self {
    self.set_coreml(coreml);
    self
  }

  /// Sets [`Self::coreml`].
  pub fn set_coreml(&mut self, coreml: LoadOptions) -> &mut Self {
    self.coreml = coreml;
    self
  }

  /// The functions `prewarm` warms, or `None` for every function this
  /// configuration can reach: all six `embed_S`, and `pack_256` when packing.
  #[inline]
  pub const fn prewarm(&self) -> Option<TextFunctions> {
    self.prewarm
  }

  /// Builder form of [`Self::set_prewarm`].
  #[must_use]
  pub fn with_prewarm(mut self, functions: TextFunctions) -> Self {
    self.set_prewarm(Some(functions));
    self
  }

  /// Sets [`Self::prewarm`]; `None` warms every reachable function.
  pub fn set_prewarm(&mut self, functions: Option<TextFunctions>) -> &mut Self {
    self.prewarm = functions;
    self
  }

  /// How a text batch is embedded.
  #[inline]
  pub const fn batching(&self) -> Batching {
    self.batching
  }

  /// Whether a text's vector is independent of the batch it rides in, under
  /// this instance's batching and placement: [`Batching::batch_independent`]
  /// at [`Self::coreml`]'s compute units.
  #[inline]
  pub const fn batch_independent(&self) -> bool {
    self.batching.batch_independent(self.coreml.compute())
  }

  /// Builder form of [`Self::set_batching`].
  #[must_use]
  pub fn with_batching(mut self, batching: Batching) -> Self {
    self.set_batching(batching);
    self
  }

  /// Sets [`Self::batching`].
  pub fn set_batching(&mut self, batching: Batching) -> &mut Self {
    self.batching = batching;
    self
  }

  /// Hugging Face's `split_special_tokens`: whether a special literal in the
  /// text is split as plain text (`true`) or read as its special id
  /// (`false`). [`Self::split`] is its typed face.
  #[inline]
  pub const fn split_special_tokens(&self) -> bool {
    self.split_special_tokens
  }

  /// [`Self::split_special_tokens`] as the behaviour it names.
  #[inline]
  pub const fn split(&self) -> SplitSpecialTokens {
    SplitSpecialTokens::from_split(self.split_special_tokens)
  }

  /// Builder form of [`Self::set_split_special_tokens`].
  #[must_use]
  pub fn with_split_special_tokens(mut self, split: bool) -> Self {
    self.set_split_special_tokens(split);
    self
  }

  /// Sets [`Self::split_special_tokens`].
  pub fn set_split_special_tokens(&mut self, split: bool) -> &mut Self {
    self.split_special_tokens = split;
    self
  }

  /// The long-text geometry and input bound. The window counts TOTAL ids,
  /// `<bos>`, the prompt and `<eos>` included, and may not exceed
  /// [`MAX_TOKENS`].
  #[inline]
  pub const fn long_text(&self) -> LongTextOptions {
    self.long_text
  }

  /// Builder form of [`Self::set_long_text`].
  #[must_use]
  pub fn with_long_text(mut self, long_text: LongTextOptions) -> Self {
    self.set_long_text(long_text);
    self
  }

  /// Sets [`Self::long_text`].
  pub fn set_long_text(&mut self, long_text: LongTextOptions) -> &mut Self {
    self.long_text = long_text;
    self
  }

  /// How a long text's windows fold into one vector.
  #[inline]
  pub const fn fold(&self) -> AggregatePolicyKind {
    self.fold
  }

  /// Builder form of [`Self::set_fold`].
  #[must_use]
  pub fn with_fold(mut self, fold: AggregatePolicyKind) -> Self {
    self.set_fold(fold);
    self
  }

  /// Sets [`Self::fold`].
  pub fn set_fold(&mut self, fold: AggregatePolicyKind) -> &mut Self {
    self.fold = fold;
    self
  }

  /// The artifact label the deployment expects the text part's stamp to
  /// yield, or `None` to pin nothing. A part whose stamp yields another label
  /// is refused as [`Error::ArtifactIdentity`]. It enters no derivation term:
  /// the label itself already does.
  #[inline]
  pub const fn expect_artifacts(&self) -> Option<ArtifactLabel> {
    self.expect_artifacts
  }

  /// Builder form of [`Self::set_expect_artifacts`].
  #[must_use]
  pub fn with_expect_artifacts(mut self, label: ArtifactLabel) -> Self {
    self.set_expect_artifacts(Some(label));
    self
  }

  /// Sets [`Self::expect_artifacts`]; `None` pins nothing.
  pub fn set_expect_artifacts(&mut self, label: Option<ArtifactLabel>) -> &mut Self {
    self.expect_artifacts = label;
    self
  }

  /// Checks what can be checked before anything loads: the window within
  /// [`MAX_TOKENS`], and windit's own geometry rules.
  ///
  /// Whether the window leaves room for text under a prompt depends on the
  /// prompt, which is the call's, so that is checked per call.
  ///
  /// # Errors
  /// [`Error::WindowOverBudget`] for a window over 512 ids; [`Error::Windowing`]
  /// for a geometry windit refuses (a zero window, an overlap at or above the
  /// window).
  pub fn validate(&self) -> Result<()> {
    let geometry = self.long_text.window_options();
    if geometry.window() > MAX_TOKENS {
      return Err(Error::WindowOverBudget(WindowOverBudget::new(
        geometry.window(),
        MAX_TOKENS,
      )));
    }
    geometry.validate()?;
    Ok(())
  }
}

/// `key=value` pairs in declaration order, joined by `,`; nested options in
/// parentheses, an absent value as `none`, an absent `prewarm` as `reachable`.
impl core::fmt::Display for TextOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "dir={},coreml=({}),prewarm=",
      DirTerm(self.dir()),
      self.coreml
    )?;
    match self.prewarm {
      Some(functions) => write!(f, "{functions}")?,
      None => f.write_str("reachable")?,
    }
    write!(
      f,
      ",batching={},split_special_tokens={},long_text=({}),fold={},expect_artifacts={}",
      self.batching,
      self.split_special_tokens,
      self.long_text,
      self.fold,
      OptionTerm(self.expect_artifacts)
    )
  }
}

/// Options of the audio part (rust-options-pattern). Their presence in an
/// [`EmbedderOptions`] enables the audio lane.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct AudioOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  dir: Option<PathBuf>,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::audio_precision"))]
  precision: AudioPrecision,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::audio_coreml"))]
  coreml: LoadOptions,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::audio_batching"))]
  batching: Batching,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::hop_samples"))]
  hop_samples: NonZeroU32,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::min_tail_samples"))]
  min_tail_samples: u32,
  #[cfg_attr(feature = "serde", serde(default))]
  max_windows: Option<NonZeroU32>,
  #[cfg_attr(feature = "serde", serde(default))]
  expect_artifacts: Option<ArtifactLabel>,
}

impl Default for AudioOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl AudioOptions {
  /// The defaults: the facade's root, [`DEFAULT_AUDIO_PRECISION`],
  /// [`DEFAULT_AUDIO_COREML`], [`DEFAULT_AUDIO_BATCHING`], a hop of one
  /// window, the reference's 0.5 s tail rule, no window cap, and no pinned
  /// artifact label.
  pub const fn new() -> Self {
    Self {
      dir: None,
      precision: DEFAULT_AUDIO_PRECISION,
      coreml: DEFAULT_AUDIO_COREML,
      batching: DEFAULT_AUDIO_BATCHING,
      hop_samples: DEFAULT_HOP_SAMPLES,
      min_tail_samples: DEFAULT_MIN_TAIL_SAMPLES,
      max_windows: None,
      expect_artifacts: None,
    }
  }

  /// The part's directory, or `None` for the facade's root.
  #[inline]
  pub fn dir(&self) -> Option<&Path> {
    self.dir.as_deref()
  }

  /// Builder form of [`Self::set_dir`].
  #[must_use]
  pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
    self.set_dir(Some(dir.into()));
    self
  }

  /// Sets [`Self::dir`]; `None` is the facade's root.
  pub fn set_dir(&mut self, dir: Option<PathBuf>) -> &mut Self {
    self.dir = dir;
    self
  }

  /// Which audio package runs.
  #[inline]
  pub const fn precision(&self) -> AudioPrecision {
    self.precision
  }

  /// Builder form of [`Self::set_precision`].
  #[must_use]
  pub fn with_precision(mut self, precision: AudioPrecision) -> Self {
    self.set_precision(precision);
    self
  }

  /// Sets [`Self::precision`].
  pub fn set_precision(&mut self, precision: AudioPrecision) -> &mut Self {
    self.precision = precision;
    self
  }

  /// The Core ML choices the audio bundle loads with.
  #[inline]
  pub const fn coreml(&self) -> LoadOptions {
    self.coreml
  }

  /// Builder form of [`Self::set_coreml`].
  #[must_use]
  pub fn with_coreml(mut self, coreml: LoadOptions) -> Self {
    self.set_coreml(coreml);
    self
  }

  /// Sets [`Self::coreml`].
  pub fn set_coreml(&mut self, coreml: LoadOptions) -> &mut Self {
    self.coreml = coreml;
    self
  }

  /// How a clip's windows are embedded by the text functions.
  #[inline]
  pub const fn batching(&self) -> Batching {
    self.batching
  }

  /// Builder form of [`Self::set_batching`].
  #[must_use]
  pub fn with_batching(mut self, batching: Batching) -> Self {
    self.set_batching(batching);
    self
  }

  /// Sets [`Self::batching`].
  pub fn set_batching(&mut self, batching: Batching) -> &mut Self {
    self.batching = batching;
    self
  }

  /// The samples from one window's start to the next's.
  #[inline]
  pub const fn hop_samples(&self) -> NonZeroU32 {
    self.hop_samples
  }

  /// Builder form of [`Self::set_hop_samples`].
  #[must_use]
  pub fn with_hop_samples(mut self, hop: NonZeroU32) -> Self {
    self.set_hop_samples(hop);
    self
  }

  /// Sets [`Self::hop_samples`].
  pub fn set_hop_samples(&mut self, hop: NonZeroU32) -> &mut Self {
    self.hop_samples = hop;
    self
  }

  /// The shortest final window kept when it is not the only one.
  #[inline]
  pub const fn min_tail_samples(&self) -> u32 {
    self.min_tail_samples
  }

  /// Builder form of [`Self::set_min_tail_samples`].
  #[must_use]
  pub fn with_min_tail_samples(mut self, samples: u32) -> Self {
    self.set_min_tail_samples(samples);
    self
  }

  /// Sets [`Self::min_tail_samples`].
  pub fn set_min_tail_samples(&mut self, samples: u32) -> &mut Self {
    self.min_tail_samples = samples;
    self
  }

  /// The most windows one clip may plan, or `None` for no cap.
  #[inline]
  pub const fn max_windows(&self) -> Option<NonZeroU32> {
    self.max_windows
  }

  /// Builder form of [`Self::set_max_windows`].
  #[must_use]
  pub fn with_max_windows(mut self, max: NonZeroU32) -> Self {
    self.set_max_windows(Some(max));
    self
  }

  /// Sets [`Self::max_windows`]; `None` is no cap.
  pub fn set_max_windows(&mut self, max: Option<NonZeroU32>) -> &mut Self {
    self.max_windows = max;
    self
  }

  /// The artifact label the deployment expects the audio part's stamp to
  /// yield, or `None` to pin nothing. It enters no derivation term.
  #[inline]
  pub const fn expect_artifacts(&self) -> Option<ArtifactLabel> {
    self.expect_artifacts
  }

  /// Builder form of [`Self::set_expect_artifacts`].
  #[must_use]
  pub fn with_expect_artifacts(mut self, label: ArtifactLabel) -> Self {
    self.set_expect_artifacts(Some(label));
    self
  }

  /// Sets [`Self::expect_artifacts`]; `None` pins nothing.
  pub fn set_expect_artifacts(&mut self, label: Option<ArtifactLabel>) -> &mut Self {
    self.expect_artifacts = label;
    self
  }
}

/// `key=value` pairs in declaration order, joined by `,`; nested options in
/// parentheses, an absent value as `none`.
impl core::fmt::Display for AudioOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "dir={},precision={},coreml=({}),batching={},hop_samples={},min_tail_samples={},max_windows={},expect_artifacts={}",
      DirTerm(self.dir()),
      self.precision,
      self.coreml,
      self.batching,
      self.hop_samples,
      self.min_tail_samples,
      OptionTerm(self.max_windows),
      OptionTerm(self.expect_artifacts)
    )
  }
}

/// Options of the vision part (rust-options-pattern). Their presence in an
/// [`EmbedderOptions`] enables the vision lane.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ImageOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  dir: Option<PathBuf>,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::vision_coreml"))]
  coreml: LoadOptions,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::vision_batching"))]
  batching: Batching,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::budgets"))]
  budgets: VisionBudgets,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::budget"))]
  default_budget: VisionBudget,
  #[cfg_attr(feature = "serde", serde(default = "serde_defaults::resize"))]
  resize: ResizeRoad,
  #[cfg_attr(feature = "serde", serde(default))]
  expect_artifacts: Option<ArtifactLabel>,
}

impl Default for ImageOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ImageOptions {
  /// The defaults: the facade's root, [`DEFAULT_VISION_COREML`],
  /// [`DEFAULT_VISION_BATCHING`], the 140-token budget alone and as the
  /// default, [`DEFAULT_RESIZE`], and no pinned artifact label.
  pub const fn new() -> Self {
    Self {
      dir: None,
      coreml: DEFAULT_VISION_COREML,
      batching: DEFAULT_VISION_BATCHING,
      budgets: DEFAULT_BUDGETS,
      default_budget: DEFAULT_BUDGET,
      resize: DEFAULT_RESIZE,
      expect_artifacts: None,
    }
  }

  /// The part's directory, or `None` for the facade's root.
  #[inline]
  pub fn dir(&self) -> Option<&Path> {
    self.dir.as_deref()
  }

  /// Builder form of [`Self::set_dir`].
  #[must_use]
  pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
    self.set_dir(Some(dir.into()));
    self
  }

  /// Sets [`Self::dir`]; `None` is the facade's root.
  pub fn set_dir(&mut self, dir: Option<PathBuf>) -> &mut Self {
    self.dir = dir;
    self
  }

  /// The Core ML choices the vision functions load with.
  #[inline]
  pub const fn coreml(&self) -> LoadOptions {
    self.coreml
  }

  /// Builder form of [`Self::set_coreml`].
  #[must_use]
  pub fn with_coreml(mut self, coreml: LoadOptions) -> Self {
    self.set_coreml(coreml);
    self
  }

  /// Sets [`Self::coreml`].
  pub fn set_coreml(&mut self, coreml: LoadOptions) -> &mut Self {
    self.coreml = coreml;
    self
  }

  /// How a batch of images is embedded by the text functions.
  #[inline]
  pub const fn batching(&self) -> Batching {
    self.batching
  }

  /// Builder form of [`Self::set_batching`].
  #[must_use]
  pub fn with_batching(mut self, batching: Batching) -> Self {
    self.set_batching(batching);
    self
  }

  /// Sets [`Self::batching`].
  pub fn set_batching(&mut self, batching: Batching) -> &mut Self {
    self.batching = batching;
    self
  }

  /// The budgets whose `vision_T` functions an image instance loads.
  #[inline]
  pub const fn budgets(&self) -> VisionBudgets {
    self.budgets
  }

  /// Builder form of [`Self::set_budgets`].
  #[must_use]
  pub fn with_budgets(mut self, budgets: VisionBudgets) -> Self {
    self.set_budgets(budgets);
    self
  }

  /// Sets [`Self::budgets`].
  pub fn set_budgets(&mut self, budgets: VisionBudgets) -> &mut Self {
    self.budgets = budgets;
    self
  }

  /// The budget a call without its own uses. It must be one of
  /// [`Self::budgets`].
  #[inline]
  pub const fn default_budget(&self) -> VisionBudget {
    self.default_budget
  }

  /// Builder form of [`Self::set_default_budget`].
  #[must_use]
  pub fn with_default_budget(mut self, budget: VisionBudget) -> Self {
    self.set_default_budget(budget);
    self
  }

  /// Sets [`Self::default_budget`].
  pub fn set_default_budget(&mut self, budget: VisionBudget) -> &mut Self {
    self.default_budget = budget;
    self
  }

  /// How an image is resized to its budget's geometry.
  #[inline]
  pub const fn resize(&self) -> ResizeRoad {
    self.resize
  }

  /// Builder form of [`Self::set_resize`].
  #[must_use]
  pub fn with_resize(mut self, resize: ResizeRoad) -> Self {
    self.set_resize(resize);
    self
  }

  /// Sets [`Self::resize`].
  pub fn set_resize(&mut self, resize: ResizeRoad) -> &mut Self {
    self.resize = resize;
    self
  }

  /// The artifact label the deployment expects the vision part's stamp to
  /// yield, or `None` to pin nothing. It enters no derivation term.
  #[inline]
  pub const fn expect_artifacts(&self) -> Option<ArtifactLabel> {
    self.expect_artifacts
  }

  /// Builder form of [`Self::set_expect_artifacts`].
  #[must_use]
  pub fn with_expect_artifacts(mut self, label: ArtifactLabel) -> Self {
    self.set_expect_artifacts(Some(label));
    self
  }

  /// Sets [`Self::expect_artifacts`]; `None` pins nothing.
  pub fn set_expect_artifacts(&mut self, label: Option<ArtifactLabel>) -> &mut Self {
    self.expect_artifacts = label;
    self
  }

  /// Checks what can be checked before anything loads: the default budget is
  /// one of the loaded budgets.
  ///
  /// # Errors
  /// [`Error::BudgetNotLoaded`] naming the default budget and the loaded set.
  pub fn validate(&self) -> Result<()> {
    if self.budgets.contains(self.default_budget) {
      Ok(())
    } else {
      Err(Error::BudgetNotLoaded(BudgetNotLoaded::new(
        self.default_budget,
        self.budgets,
      )))
    }
  }
}

/// `key=value` pairs in declaration order, joined by `,`; nested options in
/// parentheses, an absent value as `none`.
impl core::fmt::Display for ImageOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "dir={},coreml=({}),batching={},budgets={},default_budget={},resize={},expect_artifacts={}",
      DirTerm(self.dir()),
      self.coreml,
      self.batching,
      self.budgets,
      self.default_budget,
      self.resize,
      OptionTerm(self.expect_artifacts)
    )
  }
}

/// Options of the facade: the text part, and the audio and vision parts whose
/// presence enables their lanes. An absent lane loads nothing.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct EmbedderOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  text: TextOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  audio: Option<AudioOptions>,
  #[cfg_attr(feature = "serde", serde(default))]
  vision: Option<ImageOptions>,
}

impl Default for EmbedderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl EmbedderOptions {
  /// The text part's defaults, and both media lanes off.
  pub const fn new() -> Self {
    Self {
      text: TextOptions::new(),
      audio: None,
      vision: None,
    }
  }

  /// The text part's options.
  #[inline]
  pub const fn text(&self) -> &TextOptions {
    &self.text
  }

  /// Builder form of [`Self::set_text`].
  #[must_use]
  pub fn with_text(mut self, text: TextOptions) -> Self {
    self.set_text(text);
    self
  }

  /// Sets [`Self::text`].
  pub fn set_text(&mut self, text: TextOptions) -> &mut Self {
    self.text = text;
    self
  }

  /// The audio part's options, or `None` when the audio lane is off.
  #[inline]
  pub const fn audio(&self) -> Option<&AudioOptions> {
    self.audio.as_ref()
  }

  /// Builder form of [`Self::set_audio`]: turns the audio lane on.
  #[must_use]
  pub fn with_audio(mut self, audio: AudioOptions) -> Self {
    self.set_audio(Some(audio));
    self
  }

  /// Sets [`Self::audio`]; `None` turns the audio lane off.
  pub fn set_audio(&mut self, audio: Option<AudioOptions>) -> &mut Self {
    self.audio = audio;
    self
  }

  /// The vision part's options, or `None` when the vision lane is off.
  #[inline]
  pub const fn vision(&self) -> Option<&ImageOptions> {
    self.vision.as_ref()
  }

  /// Whether an audio window's vector is independent of the batch it rides
  /// in. A media face packs in the TEXT model's `pack_256`, so the answer is
  /// the audio face's batching at the text part's compute units: `true` under
  /// [`Batching::Single`], [`PACKED_SLOT_INVARIANT`] at the text placement
  /// under [`Batching::Packed`]. With the lane off, the default batching
  /// answers.
  ///
  #[inline]
  pub const fn audio_batch_independent(&self) -> bool {
    self.audio_batch_independent_in(&PACKED_SLOT_INVARIANT)
  }

  /// [`Self::audio_batch_independent`] over `table`.
  #[inline]
  pub(crate) const fn audio_batch_independent_in(&self, table: &PlacementTable<bool>) -> bool {
    let batching = match &self.audio {
      Some(audio) => audio.batching,
      None => DEFAULT_AUDIO_BATCHING,
    };
    batching.batch_independent_in(self.text.coreml.compute(), table)
  }

  /// Whether an image's vector is independent of the batch it rides in: the
  /// image face's batching at the TEXT part's compute units, as
  /// [`Self::audio_batch_independent`] for audio. With the lane off, the
  /// default batching answers.
  #[inline]
  pub const fn image_batch_independent(&self) -> bool {
    self.image_batch_independent_in(&PACKED_SLOT_INVARIANT)
  }

  /// [`Self::image_batch_independent`] over `table`.
  #[inline]
  pub(crate) const fn image_batch_independent_in(&self, table: &PlacementTable<bool>) -> bool {
    let batching = match &self.vision {
      Some(vision) => vision.batching,
      None => DEFAULT_VISION_BATCHING,
    };
    batching.batch_independent_in(self.text.coreml.compute(), table)
  }

  /// Builder form of [`Self::set_vision`]: turns the vision lane on.
  #[must_use]
  pub fn with_vision(mut self, vision: ImageOptions) -> Self {
    self.set_vision(Some(vision));
    self
  }

  /// Sets [`Self::vision`]; `None` turns the vision lane off.
  pub fn set_vision(&mut self, vision: Option<ImageOptions>) -> &mut Self {
    self.vision = vision;
    self
  }

  /// Checks every enabled part's options, text first.
  ///
  /// # Errors
  /// As [`TextOptions::validate`] and [`ImageOptions::validate`].
  pub fn validate(&self) -> Result<()> {
    self.text.validate()?;
    if let Some(vision) = &self.vision {
      vision.validate()?;
    }
    Ok(())
  }
}

/// `text=(…),audio=(…),vision=(…)`, each part's own spelling in parentheses,
/// an absent lane as `none`.
impl core::fmt::Display for EmbedderOptions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "text=({}),audio=", self.text)?;
    match &self.audio {
      Some(audio) => write!(f, "({audio})")?,
      None => f.write_str("none")?,
    }
    f.write_str(",vision=")?;
    match &self.vision {
      Some(vision) => write!(f, "({vision})"),
      None => f.write_str("none"),
    }
  }
}

/// What a part spells for the derivation version: exactly the options that
/// change its vectors, and the label of the artifacts it runs.
///
/// **This spelling is persisted by downstream derivation fingerprints — change
/// it only with a major bump.**
///
/// Never printed: the bounds (`max_windows`, `max_input_bytes`), and the
/// options that change no vector (`dir`, `prewarm`, `budgets`,
/// `expect_artifacts`). So changing one of them mints no run.
///
/// - text: `embeddinggemma2/text(artifacts=<label>,coreml=(<LoadOptions>),
///   batching=<batching>,split_special_tokens=<bool>,
///   window=(window=<ids>,hop=<ids>,tail=<TailPolicy>),fold=<fold>)`. The
///   geometry is its own projection of `long_text`, never `LongTextOptions`'
///   `Display`, which also prints the bounds.
/// - audio: `embeddinggemma2/audio(artifacts=<label>,precision=<precision>,
///   coreml=(<LoadOptions>),batching=<batching>,hop_samples=<n>,
///   min_tail_samples=<n>)`.
/// - vision: `embeddinggemma2/vision(artifacts=<label>,coreml=(<LoadOptions>),
///   batching=<batching>,default_budget=<tokens>,resize=<road>)`.
///
/// The per-call values a table embeds under (the prompt, the budget, the
/// width, the batching used) are the caller's to append, each spelled by its
/// own vocabulary's `Display`.
#[derive(Debug, Clone, Copy)]
pub struct DerivationTerms<'a>(Terms<'a>);

#[derive(Debug, Clone, Copy)]
enum Terms<'a> {
  Text(PartTerms<'a, TextOptions>),
  Audio(PartTerms<'a, AudioOptions>),
  // The vision part (C15) spells this.
  #[cfg_attr(not(test), allow(dead_code))]
  Vision(PartTerms<'a, ImageOptions>),
}

/// One part's terms: the label of the artifacts it runs, and its options.
#[derive(Debug)]
struct PartTerms<'a, O> {
  artifacts: ArtifactLabel,
  options: &'a O,
}

// By hand: a derive would ask `O: Copy`, and the options are only borrowed.
impl<O> Clone for PartTerms<'_, O> {
  fn clone(&self) -> Self {
    *self
  }
}

impl<O> Copy for PartTerms<'_, O> {}

impl<'a> DerivationTerms<'a> {
  /// The terms of a text part running `artifacts` under `options`.
  pub(crate) const fn text(artifacts: ArtifactLabel, options: &'a TextOptions) -> Self {
    Self(Terms::Text(PartTerms { artifacts, options }))
  }

  /// The terms of an audio part running `artifacts` under `options`.
  pub(crate) const fn audio(artifacts: ArtifactLabel, options: &'a AudioOptions) -> Self {
    Self(Terms::Audio(PartTerms { artifacts, options }))
  }

  /// The terms of a vision part running `artifacts` under `options`.
  #[cfg_attr(not(test), allow(dead_code))]
  pub(crate) const fn vision(artifacts: ArtifactLabel, options: &'a ImageOptions) -> Self {
    Self(Terms::Vision(PartTerms { artifacts, options }))
  }
}

impl core::fmt::Display for DerivationTerms<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self.0 {
      Terms::Text(PartTerms { artifacts, options }) => {
        let geometry = options.long_text.window_options();
        write!(
          f,
          "embeddinggemma2/text(artifacts={artifacts},coreml=({}),batching={},split_special_tokens={},window=(window={},hop={},tail={}),fold={})",
          options.coreml,
          options.batching,
          options.split_special_tokens,
          geometry.window(),
          geometry.hop(),
          geometry.tail(),
          options.fold
        )
      }
      Terms::Audio(PartTerms { artifacts, options }) => write!(
        f,
        "embeddinggemma2/audio(artifacts={artifacts},precision={},coreml=({}),batching={},hop_samples={},min_tail_samples={})",
        options.precision,
        options.coreml,
        options.batching,
        options.hop_samples,
        options.min_tail_samples
      ),
      Terms::Vision(PartTerms { artifacts, options }) => write!(
        f,
        "embeddinggemma2/vision(artifacts={artifacts},coreml=({}),batching={},default_budget={},resize={})",
        options.coreml, options.batching, options.default_budget, options.resize
      ),
    }
  }
}

#[cfg(test)]
mod tests;
