//! EmbeddingGemma 2 on Core ML: one 768-d embedding space for text, audio
//! windows and images.
//!
//! The model is `google/embeddinggemma-2`, run through FluidInference's Core ML
//! conversion (`FluidInference/embeddinggemma-2-coreml` @ `3f978dd`): three
//! packages and two lifted tables.
//!
//! - **Text** (`EmbeddingGemma2Text.mlpackage`, the Neural Engine by
//!   default): seven functions over one set of weights, `embed_32` …
//!   `embed_512` and `pack_256`. The token table (`embeddings.bf16`) is read
//!   outside the graph, one row per id.
//! - **Audio** (`EmbeddingGemma2Audio.mlpackage`, fp16, or this crate's own
//!   fp32 conversion; the GPU by default): 16 kHz mono windows of 10 s to 250
//!   audio tokens.
//! - **Vision** (`EmbeddingGemma2Vision.mlpackage`, the GPU by default): an
//!   image's patches to 70, 140 or 280 image tokens. The patch-position tables
//!   (`position_embeddings.f16`) are read outside the graph.
//!
//! Audio and image tokens are embedded by the TEXT functions, as a sequence
//! framed by their start and end tokens, so every modality lands in the one
//! space and a text query ranks audio windows, images and texts alike.
//!
//! # Matryoshka
//!
//! [`Embedding`] is 768-d; [`Embedding::truncate`] keeps the first 512, 256 or
//! 128 coordinates, re-normalized. The width is the type's, so vectors of two
//! widths cannot be compared by accident.
//!
//! # Options
//!
//! [`TextOptions`], [`AudioOptions`] and [`ImageOptions`] carry every choice
//! the model and Core ML allow, each with an opinionated default, and
//! [`EmbedderOptions`] composes them. A part spells the choices that change
//! its vectors, and the label of the artifacts it runs, as its
//! [`DerivationTerms`].
//!
//! # The facade and the parts
//!
//! [`Embedder`] loads one instance of each enabled part over a stamped model
//! directory: the text part always, the audio and vision parts when
//! [`EmbedderOptions`] carries their options. A lane whose options are absent
//! loads nothing.
//!
//! ```no_run
//! use coremlit::embeddings::embeddinggemma2::{
//!   AudioOptions, Embedder, EmbedderOptions, ImageOptions, Prompt, Rgb8Image, compile_artifacts,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = "Models/embeddinggemma2";
//! // Once per install: compile the packages, build the tokenizer image, stamp.
//! compile_artifacts(dir)?;
//! let options = EmbedderOptions::new()
//!   .with_audio(AudioOptions::new())
//!   .with_vision(ImageOptions::new());
//! let mut embedder = Embedder::load(dir, &options)?;
//! embedder.prewarm()?;
//!
//! let query = embedder.embed_text("a photo of a dog", Prompt::SearchQuery)?;
//! let pixels = vec![0u8; 640 * 480 * 3];
//! let image = embedder.embed_image(Rgb8Image::new(&pixels, 640, 480)?, None)?;
//! println!("the image: {:.3}", query.cosine(&image));
//! let clip = vec![0.0f32; 16_000 * 25];
//! for window in embedder.embed_audio(&clip)? {
//!   let score = query.cosine(window.embedding());
//!   println!("window {}: {score:.3}", window.ordinal());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! The facade runs one instance of each part, one stage after another. The
//! parts are separable, and each is instantiable many times over one mapping
//! of the shared tables: [`TextArtifacts`] and [`VisionArtifacts`] are opened
//! once and each instance holds a clone ([`TextEmbedder`], [`ImageEncoder`];
//! [`AudioEncoder`] maps nothing). Every instance is `Send`, so the GPU
//! stages (audio, vision) and the Neural Engine stage (text) can run on their
//! own threads, the tokens crossing between them as owned values.
//!
//! # The model directory
//!
//! [`compile_artifacts`] stamps a model directory, and [`prune`] removes what
//! its stamp no longer names. Every walk of the directory follows one rule:
//!
//! - **Reading follows every link, wherever it points.** The stamp, its
//!   digests and the supplied-bundle size check each record the link's
//!   TARGET, as Core ML loads it. Hugging Face's cache is why: a snapshot's
//!   files are links into the cache's `blobs`, outside the snapshot, so a
//!   replaced blob is a replaced file, stamped again under a new label.
//! - **A broken link and a directory cycle are refused by name**
//!   ([`Error::Traversal`]), as is a tree of more than 1024 files and
//!   directories.
//! - **Nothing writes or deletes through a link.** [`prune`] never follows
//!   one: it removes derived names only, and leaves a derived-named link
//!   alone.
//!
//! Two doors hold the rule. **The resolve door** answers every look at a
//! path, every walk's entries, and every stamped path before the kit opens
//! it: a mapped table or tokenizer image, a bundle Core ML loads. A broken
//! link anywhere on a path is named, its last component or one of its
//! directories; only a path with nothing at it is absent. **The read door**
//! takes every artifact byte the kit reads, `config.json`, `tokenizer.json`,
//! the stamp and every file the build hashes: it resolves the path, holds the
//! file to its cap before a byte is read, reserves `min(size, cap) + 1` bytes
//! through the kit's fallible reservations, reads into them without ever
//! growing them, and names every refusal. A document that grows while it is
//! read is refused at its cap. A stamp that cannot be read is refused by name,
//! never rebuilt over; only an absent one is rebuilt.
//!
//! # Allocation
//!
//! **Inputs.** Every buffer an input sizes is reserved fallibly, on every
//! serving path: a reservation that cannot be made is [`Error::Allocation`]
//! naming its [`AllocationSite`], never an abort. An input is what a call
//! embeds: its texts, clips, images and a batch's members. pixon and windit
//! reserve fallibly themselves, answered as [`Error::Resize`] and
//! [`Error::Windowing`].
//!
//! **Artifacts.** An artifact is a model directory's files; the directory's
//! path and the options are configuration. The artifacts are the deployer's:
//! the kit implements one revision of them, which the lock rows pin by
//! digest, and a file crafted to stress the parser is outside what the kit
//! defends against. What the kit promises of any artifact:
//!
//! - **Each document is held to its cap** before a byte of it is read:
//!   `tokenizer.json` at most 64 MiB (the one the kit implements is
//!   32.2 MB), `config.json` at most 1 MiB (it is 2.8 KB), the stamp at most
//!   16 MiB (about 7 KB). It is read through the read door into a buffer
//!   reserved fallibly and never grown.
//! - **The parse is serde's own**, into typed structs; no
//!   `serde_json::Value` is built for an artifact. What serde allocates while
//!   it parses is infallible, and bounded by the capped document's structure:
//!   no multiple of the cap is promised. `tokenizer.json`'s vocabulary, merges
//!   and added tokens grow through the door at `source_parse`.
//! - **A value the kit does not read is skipped, not validated**: the keys
//!   `tokenizer.json` carries beyond those the kit reads (its decoder, the
//!   post-processor's pair template) are passed over by serde.
//! - **A walked tree holds at most 1024 files and directories**; the kit's
//!   trees hold three or four files each. The walk's own lists are reserved
//!   fallibly. What is built per entry is bounded by that cap and is
//!   infallible: each entry's path (at most `PATH_MAX`), and the maps the
//!   stamp is assembled in (`sources`' currency keys, `digest_files`' digests
//!   and file entries), whose `BTreeMap` has no fallible insert; at most 1024
//!   entries a tree, over the kit's fixed set of trees.
//! - **Every other table, list and image the kit builds from an artifact** is
//!   reserved fallibly, at its own site.
//!
//! Beyond these, each with its bound:
//!
//! - **A refusal's quoted text**: at most 64 bytes of an artifact's own, all
//!   the refusal's quotations together.
//! - **Core ML's arrays**: the contract's fixed shapes, allocated by Core ML;
//!   a failure is [`Error::Tensor`].
//! - **Constant buffers**: the 1 MiB block a file is hashed through, the
//!   8 KiB writer of the stamp, a 64-byte hex digest.
//!
//! macOS 15 or later (the packages are Core ML specification 9).

use core::num::NonZeroU16;
use std::path::{Path, PathBuf};

use windit::plan::Span;

use self::{
  alloc::with_capacity,
  error::{AllocationSite, Result},
};

pub mod error;

mod alloc;
mod artifacts;
mod audio;
mod config;
mod contract;
mod embedding;
mod long_text;
mod map;
mod media;
mod options;
mod placement;
mod prewarm;
mod prompt;
mod table;
mod text;
mod tokenizer;
mod vision;

pub use artifacts::{
  ArtifactLabel, ParseArtifactLabelError, Part,
  compile::{CompileReport, PruneReport, compile_artifacts, prune},
};
pub use audio::{AudioEncoder, AudioPrecision};
pub use config::MAX_TOKENS;
pub use embedding::{EMBEDDING_DIM, Embedding, Matryoshka, Width};
pub use error::Error;
pub use long_text::WindowEmbedding;
pub use media::{AudioTokens, ImageTokens};
pub use options::{
  AudioOptions, Batching, DEFAULT_AUDIO_BATCHING, DEFAULT_AUDIO_COREML, DEFAULT_AUDIO_PRECISION,
  DEFAULT_BUDGET, DEFAULT_BUDGETS, DEFAULT_FOLD, DEFAULT_HOP_SAMPLES, DEFAULT_MIN_TAIL_SAMPLES,
  DEFAULT_RESIZE, DEFAULT_SPLIT_SPECIAL_TOKENS, DEFAULT_TEXT_BATCHING, DEFAULT_TEXT_COREML,
  DEFAULT_VISION_BATCHING, DEFAULT_VISION_COREML, DerivationTerms, EmbedderOptions, ImageOptions,
  TextOptions,
};
pub use placement::{PACKED_SLOT_INVARIANT, PlacementTable, run_identity};
pub use prewarm::{PrewarmReport, Warmed};
pub use prompt::Prompt;
pub use text::{Framing, TextArtifacts, TextEmbedder, TextFunction, TextFunctions};
pub use tokenizer::{
  SplitSpecialTokens,
  bpe::{TokenId, TokenizerScratch},
};
pub use vision::{
  ImageEncoder, PreparedImage, ResizeRoad, VisionArtifacts, VisionBudget, VisionBudgets,
};

/// The long-text geometry and input bound, shared with the crate's other text
/// doors; its window counts total ids and may not exceed [`MAX_TOKENS`].
pub use crate::embeddings::long_text::LongTextOptions;

/// How a long text's windows fold into one vector, shared with the crate's
/// other doors that fold windows.
pub use crate::embeddings::aggregate::{AggregatePolicyKind, EmaRenormalizedOptions};

/// The borrowed RGB8 image view the vision part takes, shared with the crate's
/// other image doors, and the refusal its constructor returns.
pub use crate::embeddings::image::{ImageGeometryError, Rgb8Image};

/// windit's window geometry and tail policy, carried by [`LongTextOptions`].
pub use windit::{TailPolicy, plan::WindowOptions};

/// One audio window of a clip and its embedding: what
/// [`Embedder::embed_audio`] returns, in plan order.
///
/// - [`Self::ordinal`]: its position in the clip's plan, `0..n`.
/// - [`Self::samples`]: its span over the clip's samples. A partial last
///   window says so through a span shorter than 160,000 samples.
/// - [`Self::tokens`]: its real audio tokens, `1..=250`; fewer than 250 for a
///   partial window.
/// - [`Self::embedding`]: the unit-norm vector the text part makes of them.
///
/// The span is windit's, so a consumer that wants one vector per clip folds
/// the windows through `windit::aggregate::aggregate` under any
/// [`AggregatePolicyKind`]. No `PartialEq` and no serde, for [`Embedding`]'s
/// reasons.
#[derive(Debug, Clone)]
pub struct AudioWindowEmbedding {
  ordinal: usize,
  samples: Span,
  tokens: NonZeroU16,
  embedding: Embedding,
}

impl AudioWindowEmbedding {
  /// The window's position in the clip's plan.
  #[inline]
  pub const fn ordinal(&self) -> usize {
    self.ordinal
  }

  /// The window's span over the clip's samples.
  #[inline]
  pub const fn samples(&self) -> Span {
    self.samples
  }

  /// The window's real audio tokens.
  #[inline]
  pub const fn tokens(&self) -> NonZeroU16 {
    self.tokens
  }

  /// The unit-norm embedding of the window.
  #[inline]
  pub const fn embedding(&self) -> &Embedding {
    &self.embedding
  }

  /// The embedding, moved out of the window.
  #[must_use]
  #[inline]
  pub fn into_embedding(self) -> Embedding {
    self.embedding
  }
}

/// The vision part of an [`Embedder`]: the artifacts, which prepare images on
/// any thread, and the instance that encodes them.
#[derive(Debug)]
pub struct VisionPart {
  artifacts: VisionArtifacts,
  encoder: ImageEncoder,
  /// The facade's own planes, refilled image after image.
  prepared: PreparedImage,
}

impl VisionPart {
  /// The artifacts, to prepare images with on any thread.
  #[inline]
  pub const fn artifacts(&self) -> &VisionArtifacts {
    &self.artifacts
  }

  /// The instance.
  #[inline]
  pub fn encoder_mut(&mut self) -> &mut ImageEncoder {
    &mut self.encoder
  }

  /// The artifacts and the instance, moved out.
  #[must_use]
  pub fn into_inner(self) -> (VisionArtifacts, ImageEncoder) {
    (self.artifacts, self.encoder)
  }
}

/// The parts an [`Embedder`] loaded, handed out by [`Embedder::into_parts`]
/// so that a service can move each to its own thread: the text instance, and
/// the audio and vision parts whose lanes are on.
#[derive(Debug)]
pub struct Parts {
  text: TextEmbedder,
  audio: Option<AudioEncoder>,
  vision: Option<VisionPart>,
}

impl Parts {
  /// The text instance.
  #[inline]
  pub fn text_mut(&mut self) -> &mut TextEmbedder {
    &mut self.text
  }

  /// The audio instance, when the audio lane is on.
  #[inline]
  pub fn audio_mut(&mut self) -> Option<&mut AudioEncoder> {
    self.audio.as_mut()
  }

  /// The vision part, when the vision lane is on.
  #[inline]
  pub fn vision_mut(&mut self) -> Option<&mut VisionPart> {
    self.vision.as_mut()
  }

  /// The three, moved out.
  #[must_use]
  pub fn into_inner(self) -> (TextEmbedder, Option<AudioEncoder>, Option<VisionPart>) {
    (self.text, self.audio, self.vision)
  }
}

/// One instance of each enabled part: the text part always, the audio and
/// vision parts when [`EmbedderOptions`] carries their options. A lane whose
/// options are absent loads nothing, and a call into it is refused by name,
/// [`Error::PartDisabled`], before any work.
///
/// The facade is the plain form: `&mut self`, one instance per part, and the
/// stages run one after another on the calling thread. A service that wants
/// several instances of a part, or the GPU and Neural Engine stages overlapped
/// on two threads, builds the parts over shared artifacts itself
/// ([`TextArtifacts`], [`VisionArtifacts`]), or takes this facade's parts with
/// [`Self::into_parts`].
#[derive(Debug)]
pub struct Embedder {
  parts: Parts,
}

impl Embedder {
  /// Loads the enabled parts. Each part reads the directory its options name
  /// ([`TextOptions::dir`], [`AudioOptions::dir`], [`ImageOptions::dir`]),
  /// else `root`. Never writes: each directory must already be stamped by
  /// [`compile_artifacts`].
  ///
  /// The text functions load at their first use or at [`Self::prewarm`]; the
  /// audio bundle and the configured vision budgets load here.
  ///
  /// # Errors
  /// As [`EmbedderOptions::validate`], [`TextArtifacts::open`],
  /// [`TextEmbedder::new`], [`AudioEncoder::new`], [`VisionArtifacts::open`]
  /// and [`ImageEncoder::new`].
  pub fn load(root: impl AsRef<Path>, options: &EmbedderOptions) -> Result<Self> {
    let root = root.as_ref();
    let dir = |own: Option<&Path>| -> PathBuf { own.unwrap_or(root).to_path_buf() };
    options.validate()?;
    let text_options = options.text();
    let artifacts = TextArtifacts::open(dir(text_options.dir()))?;
    let text = TextEmbedder::new(&artifacts, text_options)?;
    let audio = match options.audio() {
      Some(audio) => Some(AudioEncoder::new(dir(audio.dir()), audio)?),
      None => None,
    };
    let vision = match options.vision() {
      Some(vision) => {
        let artifacts = VisionArtifacts::open(dir(vision.dir()))?;
        let encoder = ImageEncoder::new(&artifacts, vision)?;
        Some(VisionPart {
          artifacts,
          encoder,
          prepared: PreparedImage::new(),
        })
      }
      None => None,
    };
    Ok(Self {
      parts: Parts {
        text,
        audio,
        vision,
      },
    })
  }

  /// Prewarms every enabled part, in one report and in part order: the text
  /// part's [`TextEmbedder::prewarm_set`], the audio bundle (its record names
  /// no function), then each loaded vision budget.
  ///
  /// # Errors
  /// As [`TextEmbedder::prewarm`], [`AudioEncoder::prewarm`] and
  /// [`ImageEncoder::prewarm`].
  pub fn prewarm(&mut self) -> Result<PrewarmReport> {
    let mut report = self.parts.text.prewarm()?;
    if let Some(audio) = self.parts.audio.as_mut() {
      report.extend(audio.prewarm()?);
    }
    if let Some(vision) = self.parts.vision.as_mut() {
      report.extend(vision.encoder.prewarm()?);
    }
    Ok(report)
  }

  /// Embeds `text` under `prompt`: [`TextEmbedder::embed`].
  ///
  /// # Errors
  /// As [`TextEmbedder::embed`].
  pub fn embed_text(&mut self, text: &str, prompt: Prompt<'_>) -> Result<Embedding> {
    self.parts.text.embed(text, prompt)
  }

  /// Embeds a text of any length as one vector: [`TextEmbedder::embed_long`].
  ///
  /// # Errors
  /// As [`TextEmbedder::embed_long`].
  pub fn embed_text_long(&mut self, text: &str, prompt: Prompt<'_>) -> Result<Embedding> {
    self.parts.text.embed_long(text, prompt)
  }

  /// Embeds a text of any length window by window:
  /// [`TextEmbedder::embed_windows`].
  ///
  /// # Errors
  /// As [`TextEmbedder::embed_windows`].
  pub fn embed_text_windows(
    &mut self,
    text: &str,
    prompt: Prompt<'_>,
  ) -> Result<Vec<WindowEmbedding>> {
    self.parts.text.embed_windows(text, prompt)
  }

  /// Embeds one clip, 16 kHz mono: its windows as the audio options plan
  /// them, each encoded, then all embedded as one batch under the audio
  /// face's `batching`, in plan order.
  ///
  /// Every window's tokens are held until that batch: about 256 KB per 10 s
  /// window, so about 92 MB for an hour. A service that embeds long clips
  /// drives the parts instead, [`AudioEncoder::encode_window`] per window and
  /// [`TextEmbedder::embed_audio_tokens_batch_with`] per batch of its own
  /// size.
  ///
  /// # Errors
  /// [`Error::PartDisabled`] when the audio lane is off, before any work;
  /// [`Error::Allocation`] if a window list cannot grow; otherwise as
  /// [`AudioEncoder::plan`], [`AudioEncoder::encode_window`] and
  /// [`TextEmbedder::embed_audio_tokens_batch_with`].
  pub fn embed_audio(&mut self, samples: &[f32]) -> Result<Vec<AudioWindowEmbedding>> {
    let audio = self
      .parts
      .audio
      .as_mut()
      .ok_or(Error::PartDisabled(Part::Audio))?;
    let spans = audio.plan(samples.len())?;
    let mut tokens: Vec<AudioTokens> = with_capacity(spans.len(), AllocationSite::WindowList)?;
    for span in &spans {
      tokens.push(audio.encode_window(&samples[span.start()..span.end()])?);
    }
    let batching = audio.options().batching();
    let mut rows: Vec<&AudioTokens> = with_capacity(tokens.len(), AllocationSite::WindowList)?;
    rows.extend(&tokens);
    let mut vectors = Vec::new();
    self
      .parts
      .text
      .embed_audio_tokens_batch_with(&rows, batching, &mut vectors)?;
    let mut windows: Vec<AudioWindowEmbedding> =
      with_capacity(spans.len(), AllocationSite::WindowList)?;
    for (ordinal, ((samples, tokens), embedding)) in
      spans.into_iter().zip(&tokens).zip(vectors).enumerate()
    {
      windows.push(AudioWindowEmbedding {
        ordinal,
        samples,
        tokens: tokens.valid(),
        embedding,
      });
    }
    Ok(windows)
  }

  /// Embeds one image at `budget`, or at the vision options' default budget
  /// when `None`: prepared into the facade's own planes, encoded, then
  /// embedded under the vision face's `batching`, as [`Self::embed_images`]
  /// embeds a batch of one: alone at slot 0 of `pack_256` when it packs, so
  /// both give the same vector, the one the derivation terms describe.
  ///
  /// # Errors
  /// [`Error::PartDisabled`] when the vision lane is off, before any work;
  /// otherwise as [`VisionArtifacts::prepare_into`], [`ImageEncoder::encode`]
  /// and [`TextEmbedder::embed_image_tokens_batch_with`].
  pub fn embed_image(
    &mut self,
    image: Rgb8Image<'_>,
    budget: Option<VisionBudget>,
  ) -> Result<Embedding> {
    let vision = self
      .parts
      .vision
      .as_mut()
      .ok_or(Error::PartDisabled(Part::Vision))?;
    let tokens = vision.encode(image, budget)?;
    let batching = vision.encoder.options().batching();
    let mut out = Vec::new();
    self
      .parts
      .text
      .embed_image_tokens_batch_with(&[&tokens], batching, &mut out)?;
    Ok(out.pop().expect("one image, one vector"))
  }

  /// Embeds several images at `budget` (`None`: the default budget): each
  /// prepared and encoded in turn, then all embedded as one batch under the
  /// vision face's `batching`, appended to `out` in input order. On an error,
  /// `out` keeps what it held.
  ///
  /// # Errors
  /// [`Error::PartDisabled`] when the vision lane is off, before any work;
  /// [`Error::Allocation`] if the token list cannot grow; otherwise as
  /// [`Self::embed_image`] and [`TextEmbedder::embed_image_tokens_batch_with`].
  pub fn embed_images(
    &mut self,
    images: &[Rgb8Image<'_>],
    budget: Option<VisionBudget>,
    out: &mut Vec<Embedding>,
  ) -> Result<()> {
    let vision = self
      .parts
      .vision
      .as_mut()
      .ok_or(Error::PartDisabled(Part::Vision))?;
    let mut tokens: Vec<ImageTokens> = with_capacity(images.len(), AllocationSite::BatchOutput)?;
    for image in images {
      tokens.push(vision.encode(*image, budget)?);
    }
    let batching = vision.encoder.options().batching();
    let mut rows: Vec<&ImageTokens> = with_capacity(tokens.len(), AllocationSite::BatchOutput)?;
    rows.extend(&tokens);
    self
      .parts
      .text
      .embed_image_tokens_batch_with(&rows, batching, out)
  }

  /// What `part`'s vectors derive from: the label of the artifacts it runs
  /// and every option of it that changes a vector. `None` when `part`'s lane
  /// is off; the text part is always on.
  ///
  /// An audio window or an image is embedded by the text part, so a media
  /// table's derivation version records its own part's terms and the text
  /// part's, then the per-call terms it embeds under. The join is the
  /// caller's; for a keyframe table, for example:
  ///
  /// ```no_run
  /// use coremlit::embeddings::embeddinggemma2::{Embedder, Part};
  ///
  /// fn keyframe_version(embedder: &Embedder) -> Option<String> {
  ///   let vision = embedder.derivation_terms(Part::Vision)?;
  ///   let text = embedder.derivation_terms(Part::Text)?;
  ///   Some(format!("{vision}+{text}+budget=140+width=768+batching=single"))
  /// }
  /// ```
  pub fn derivation_terms(&self, part: Part) -> Option<DerivationTerms<'_>> {
    match part {
      Part::Text => Some(self.parts.text.derivation_terms()),
      Part::Audio => self
        .parts
        .audio
        .as_ref()
        .map(AudioEncoder::derivation_terms),
      Part::Vision => self
        .parts
        .vision
        .as_ref()
        .map(|vision| vision.encoder.derivation_terms()),
    }
  }

  /// The parts, handed out.
  #[must_use]
  pub fn into_parts(self) -> Parts {
    self.parts
  }
}

impl VisionPart {
  /// Prepares `image` into the facade's planes at `budget` and encodes it.
  fn encode(&mut self, image: Rgb8Image<'_>, budget: Option<VisionBudget>) -> Result<ImageTokens> {
    self
      .artifacts
      .prepare_into(image, budget, self.encoder.options(), &mut self.prepared)?;
    self.encoder.encode(&self.prepared)
  }
}

#[cfg(test)]
mod tests;
