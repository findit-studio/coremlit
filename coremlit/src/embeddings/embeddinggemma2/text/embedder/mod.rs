//! The text part over its artifacts: [`TextArtifacts`], opened once per
//! process and shared, and [`TextEmbedder`], one instance's functions.

use std::{path::Path, sync::Arc, time::Instant};

use half::f16;
use windit::plan::Span;

use super::{
  super::{
    alloc::{self, reserve},
    artifacts::{
      ArtifactLabel, Part,
      stamp::{PartView, Stamp},
    },
    audio::AudioPrecision,
    config::{self, BOA_ID, BOI_ID, EOA_ID, EOI_ID, EOS_ID, HIDDEN_SIZE, MAX_TOKENS, PACK_SLOTS},
    embedding::{EMBEDDING_DIM, Embedding},
    error::{
      AllocationSite, Artifact, ContractRefusal, Error, Function, LoadRefusal, NonFiniteOutput,
      PredictionFailure, Result,
    },
    long_text::{WindowEmbedding, fold_windows, plan as plan_windows},
    media::{AudioTokens, ImageTokens},
    options::{Batching, DerivationTerms, TextOptions},
    prewarm::{PrewarmReport, Warmed},
    prompt::Prompt,
    table::TokenTable,
    tokenizer::{
      SplitSpecialTokens, UPSTREAM_SENTINELS,
      bpe::{Encoded, TokenId, TokenizerScratch},
      image::{TokenizerImage, UPSTREAM},
    },
  },
  TextFunction, TextFunctions,
  layout::{
    MediaSequence, PACK_ROWS, PackedInputs, Sequence, choose, fill_packed, fill_unpacked,
    plan_packs, unpacked,
  },
};
use crate::{
  DataType, FunctionName, Model, MultiArray,
  embeddings::{aggregate::AggregatePolicyKind, long_text::LongTextOptions},
  model::contract::{Checked, Dim, FeatureContract, LoadContract, StateContract},
};

/// The ids a prompt's framing takes in a window: `<bos>`, `<eos>` and the
/// prompt's own tokens. The trailing space of the eight prompts and of a
/// titled document is not counted, because it merges into the first content
/// token. A custom prefix can merge with the content, so its framing is the
/// conservative count: the prefix encoded alone, plus 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Framing {
  ids: usize,
  conservative: bool,
}

impl Framing {
  /// The ids the framing takes.
  #[inline]
  pub const fn ids(&self) -> usize {
    self.ids
  }

  /// Whether [`Self::ids`] is the conservative count of a custom prefix
  /// rather than the exact framing.
  #[inline]
  pub const fn is_conservative(&self) -> bool {
    self.conservative
  }
}

/// The framing of the eight named prompts, in [`Prompt::NAMED`] order, then of
/// [`Prompt::Bare`], computed from `image`: each prompt without its trailing
/// space, encoded and framed.
pub(crate) fn framing_table(
  image: &TokenizerImage,
  scratch: &mut TokenizerScratch,
) -> Result<[usize; 9]> {
  let mut ids = Vec::new();
  let mut table = [0; 9];
  let texts = Prompt::NAMED.map(|(_, text)| text.strip_suffix(' ').unwrap_or(text));
  for (slot, text) in texts.iter().chain(&[""]).enumerate() {
    ids.clear();
    image.encode_joint(
      &Prompt::Bare,
      text,
      SplitSpecialTokens::Recognize,
      scratch,
      Encoded {
        ids: &mut ids,
        spans: None,
      },
    )?;
    table[slot] = ids.len();
  }
  Ok(table)
}

/// The ids `prompt`'s framing takes in a window under `split`, from `image`
/// and the framing `table` computed from it: the table's count for the eight
/// prompts and `Bare`, 7 + tokens(`" " + title`) for a titled document, and
/// for a custom prefix the conservative count, the prefix encoded alone with
/// `<bos>` and `<eos>`.
pub(crate) fn framing_of(
  image: &TokenizerImage,
  table: &[usize; 9],
  prompt: Prompt<'_>,
  split: SplitSpecialTokens,
  scratch: &mut TokenizerScratch,
) -> Result<Framing> {
  let named = Prompt::NAMED
    .iter()
    .position(|(key, _)| prompt.config_key() == Some(*key));
  if let Some(slot) = named {
    return Ok(Framing {
      ids: table[slot],
      conservative: false,
    });
  }
  let mut ids = Vec::new();
  let mut encode = |prompt: Prompt<'_>, text: &str| -> Result<usize> {
    ids.clear();
    image.encode_joint(
      &prompt,
      text,
      split,
      scratch,
      Encoded {
        ids: &mut ids,
        spans: None,
      },
    )?;
    Ok(ids.len())
  };
  Ok(match prompt {
    Prompt::Document(Some(title)) => Framing {
      ids: 7 + encode(Prompt::Custom(" "), title)? - 2,
      conservative: false,
    },
    Prompt::Custom(prefix) => Framing {
      ids: encode(Prompt::Bare, prefix)?,
      conservative: true,
    },
    _ => Framing {
      ids: table[8],
      conservative: false,
    },
  })
}

/// What every text instance shares, behind one `Arc`.
#[derive(Debug)]
struct Shared {
  view: PartView,
  table: TokenTable,
  image: TokenizerImage,
  framing: [usize; 9],
}

/// The text part's read-only state: the token table and the tokenizer image,
/// each mapped once, the checked `config.json`, and what the directory's
/// stamp says of them. A clone is a reference count, so N instances share one
/// mapping of each.
#[derive(Debug, Clone)]
pub struct TextArtifacts {
  shared: Arc<Shared>,
}

impl TextArtifacts {
  /// Opens the text part of the stamped model directory `dir`: its stamp,
  /// `config.json` against the compiled contract, the token table against the
  /// sentinels the stamp recorded, and the tokenizer image the stamp names
  /// against the upstream tokenizer's identity and five encodings. Loads no
  /// model and writes nothing.
  ///
  /// # Errors
  /// [`Error::Unstamped`], [`Error::StaleStamp`] or [`Error::Stamp`] from the
  /// stamp; [`Error::Read`] or [`Error::Config`] for `config.json` (past its
  /// 1 MiB cap included); [`Error::Traversal`] for a stamped file that is now
  /// a broken link; [`Error::Allocation`] if a document's buffer cannot be
  /// reserved; [`Error::TableShape`] or [`Error::TableIdentity`] for the
  /// table; [`Error::TokenizerImage`] for the image.
  pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
    let dir = dir.as_ref();
    let stamp = Stamp::read(dir)?;
    let view = stamp.part(dir, Part::Text, AudioPrecision::Fp16)?;
    config::load(&dir.join("config.json"))?;
    let table = TokenTable::open(&dir.join("embeddings.bf16"), view.sentinels())?;
    let image_name = &view
      .image()
      .expect("the stamp's text part names its tokenizer image")
      .name;
    let image = TokenizerImage::open(&dir.join(image_name), &UPSTREAM)?;
    let mut scratch = TokenizerScratch::new();
    image.check_sentinels(&UPSTREAM_SENTINELS, &mut scratch)?;
    let framing = framing_table(&image, &mut scratch)?;
    Ok(Self {
      shared: Arc::new(Shared {
        view,
        table,
        image,
        framing,
      }),
    })
  }

  /// Encodes `prompt ⧺ text` as one string into `ids`, framed by `<bos>` and
  /// `<eos>`, on any thread with the caller's scratch. Nothing is truncated.
  ///
  /// # Errors
  /// [`Error::Allocation`] if a buffer cannot grow.
  pub fn token_ids(
    &self,
    text: &str,
    prompt: Prompt<'_>,
    split: SplitSpecialTokens,
    scratch: &mut TokenizerScratch,
    ids: &mut Vec<TokenId>,
  ) -> Result<()> {
    ids.clear();
    self
      .shared
      .image
      .encode_joint(&prompt, text, split, scratch, Encoded { ids, spans: None })
  }

  /// The label of the artifacts this part runs, from the stamp:
  /// `fluidinference@3f978dd` for the upstream files, `sha256:<12 hex>`
  /// otherwise.
  #[inline]
  pub fn artifact_label(&self) -> ArtifactLabel {
    self.shared.view.label()
  }

  /// The ids `prompt`'s framing takes in a window under `split`, from the
  /// loaded image. `split` reads a title's and a custom prefix's text as it
  /// reads the content's; the eight prompts hold no special literal.
  ///
  /// # Errors
  /// [`Error::Allocation`] if a buffer cannot grow while a title or a custom
  /// prefix is encoded.
  pub fn framing(&self, prompt: Prompt<'_>, split: SplitSpecialTokens) -> Result<Framing> {
    framing_of(
      &self.shared.image,
      &self.shared.framing,
      prompt,
      split,
      &mut TokenizerScratch::new(),
    )
  }

  /// The loaded tokenizer image.
  #[cfg(test)]
  pub(crate) fn image(&self) -> &TokenizerImage {
    &self.shared.image
  }

  /// The mapped token table.
  #[cfg(test)]
  pub(crate) fn table(&self) -> &TokenTable {
    &self.shared.table
  }

  /// Whether `other` shares this one's mappings.
  #[cfg(test)]
  pub(crate) fn same_mappings(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.shared, &other.shared)
  }
}

/// A function's contract: what the kit sends and reads.
fn contract(function: TextFunction) -> LoadContract {
  let rows = function.tokens();
  let f16 = |name, dims: &[usize]| {
    FeatureContract::new(
      name,
      DataType::F16,
      dims.iter().map(|&d| Dim::Exactly(d)).collect(),
    )
  };
  let (inputs, batch) = match function {
    TextFunction::Pack256 => (
      vec![
        f16("attention_bias", &[1, 1, PACK_ROWS, PACK_ROWS]),
        f16("inputs_embeds", &[1, PACK_ROWS, HIDDEN_SIZE]),
        f16("pool", &[PACK_SLOTS, PACK_ROWS]),
        f16("positions", &[PACK_ROWS, 1]),
      ],
      PACK_SLOTS,
    ),
    _ => (
      vec![
        f16("attention_mask", &[1, rows]),
        f16("inputs_embeds", &[1, rows, HIDDEN_SIZE]),
      ],
      1,
    ),
  };
  LoadContract::new(
    inputs,
    vec![FeatureContract::new(
      "embedding",
      DataType::F32,
      vec![Dim::Exactly(batch), Dim::Exactly(EMBEDDING_DIM)],
    )],
    StateContract::None,
  )
}

/// One loaded function and its inputs, allocated once and refilled.
#[derive(Debug)]
struct Loaded {
  model: Checked,
  inputs: Inputs,
}

/// A loaded function's inputs, by its kind.
#[derive(Debug)]
enum Inputs {
  Embed(EmbedInputs),
  Pack(PackInputs),
}

/// An `embed_S` function's inputs.
#[derive(Debug)]
struct EmbedInputs {
  mask: MultiArray,
  inputs_embeds: MultiArray,
}

/// `pack_256`'s inputs.
#[derive(Debug)]
struct PackInputs {
  attention_bias: MultiArray,
  inputs_embeds: MultiArray,
  pool: MultiArray,
  positions: MultiArray,
}

impl Inputs {
  fn new(function: TextFunction) -> Result<Self> {
    let zeros = |shape: &[usize]| MultiArray::zeros(shape, DataType::F16);
    Ok(match function {
      TextFunction::Pack256 => Self::Pack(PackInputs {
        attention_bias: zeros(&[1, 1, PACK_ROWS, PACK_ROWS])?,
        inputs_embeds: zeros(&[1, PACK_ROWS, HIDDEN_SIZE])?,
        pool: zeros(&[PACK_SLOTS, PACK_ROWS])?,
        positions: zeros(&[PACK_ROWS, 1])?,
      }),
      _ => Self::Embed(EmbedInputs {
        mask: zeros(&[1, function.tokens()])?,
        inputs_embeds: zeros(&[1, function.tokens(), HIDDEN_SIZE])?,
      }),
    })
  }
}

/// One text instance: its own text functions, each loaded on first use or by
/// [`Self::prewarm`], over a clone of the shared [`TextArtifacts`].
///
/// `Send` and not `Sync`: inference takes `&mut self`, which is also what
/// makes its input buffers and scratch its own.
#[derive(Debug)]
pub struct TextEmbedder {
  artifacts: TextArtifacts,
  options: TextOptions,
  functions: [Option<Loaded>; 7],
  scratch: TokenizerScratch,
  ids: Vec<TokenId>,
  media: Vec<f16>,
}

/// The slot of `function` in an instance's function array.
const fn slot(function: TextFunction) -> usize {
  function as usize
}

impl TextEmbedder {
  /// An instance over `artifacts` under `options`. Loads nothing: each
  /// function loads at its first use or at [`Self::prewarm`].
  ///
  /// # Errors
  /// [`Error::ArtifactIdentity`] when `options` pins an artifact label the
  /// directory's stamp does not give.
  pub fn new(artifacts: &TextArtifacts, options: &TextOptions) -> Result<Self> {
    artifacts.shared.view.expect(options.expect_artifacts())?;
    Ok(Self {
      artifacts: artifacts.clone(),
      options: options.clone(),
      functions: Default::default(),
      scratch: TokenizerScratch::new(),
      ids: Vec::new(),
      media: Vec::new(),
    })
  }

  /// The options the instance runs under.
  #[inline]
  pub fn options(&self) -> &TextOptions {
    &self.options
  }

  /// What the instance's vectors derive from: its artifacts' label and every
  /// option that changes a vector.
  #[inline]
  pub fn derivation_terms(&self) -> DerivationTerms<'_> {
    DerivationTerms::text(self.artifacts.artifact_label(), &self.options)
  }

  /// The functions loaded so far.
  #[cfg(test)]
  pub(crate) fn loaded(&self) -> TextFunctions {
    TextFunction::ALL
      .into_iter()
      .filter(|&function| self.functions[slot(function)].is_some())
      .collect()
  }

  /// The functions [`Self::prewarm`] warms: the options' set, else every
  /// function the options can reach (all six `embed_S`, and `pack_256` while
  /// text packs).
  pub fn prewarm_set(&self) -> TextFunctions {
    self.options.prewarm().unwrap_or_else(|| {
      let set: TextFunctions = TextFunction::ALL
        .into_iter()
        .filter(|f| *f != TextFunction::Pack256)
        .collect();
      match self.options.batching() {
        Batching::Packed => set.with(TextFunction::Pack256),
        Batching::Single => set,
      }
    })
  }

  /// Loads each function of [`Self::prewarm_set`] and runs one throwaway
  /// prediction on it, so the first request pays neither the load nor Core
  /// ML's device specialization. Reports each function's two durations.
  ///
  /// # Errors
  /// [`Error::Load`], [`Error::Contract`] or [`Error::Prediction`] naming the
  /// function.
  pub fn prewarm(&mut self) -> Result<PrewarmReport> {
    let mut report = PrewarmReport::default();
    let bos_eos = [
      TokenId::from_vocab(config::BOS_ID),
      TokenId::from_vocab(EOS_ID),
    ];
    for function in self.prewarm_set().iter() {
      let started = Instant::now();
      self.load(function)?;
      let load = started.elapsed();
      let started = Instant::now();
      let sequence = [Sequence::Text(&bos_eos)];
      self.predict(function, &sequence, &mut |_, _| Ok(()))?;
      report.push(Warmed::new(
        Some(Function::Text(function)),
        load,
        started.elapsed(),
      ));
    }
    Ok(report)
  }

  /// Embeds `text` under `prompt`: the joint encoding of `prompt ⧺ text`,
  /// truncated past 512 ids to its first 511 and `<eos>`, through the
  /// function the options' batching chooses (`pack_256` alone at slot 0 when
  /// text packs).
  ///
  /// An empty text embeds as `<bos>`, the prompt and `<eos>`, as the
  /// reference does: R1's corpus holds FluidUse's empty case, and M2 holds
  /// the kit to it. The long-text doors refuse it instead.
  ///
  /// # Errors
  /// [`Error::Allocation`]; [`Error::Load`], [`Error::Contract`] or
  /// [`Error::Prediction`] for the function; [`Error::NonFiniteOutput`] for a
  /// non-finite or zero output row.
  pub fn embed(&mut self, text: &str, prompt: Prompt<'_>) -> Result<Embedding> {
    let mut out = Vec::new();
    self.embed_batch(&[text], prompt, &mut out)?;
    Ok(out.pop().expect("one text, one vector"))
  }

  /// Embeds `texts` under `prompt` and the options' batching, appending to
  /// `out` in input order.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_batch(
    &mut self,
    texts: &[&str],
    prompt: Prompt<'_>,
    out: &mut Vec<Embedding>,
  ) -> Result<()> {
    self.embed_batch_with(texts, prompt, self.options.batching(), out)
  }

  /// [`Self::embed_batch`] under `batching` for this call: under
  /// [`Batching::Packed`], sequences of at most 256 ids are packed first-fit
  /// in input order and longer ones take `embed_512`; under
  /// [`Batching::Single`], each takes the smallest `embed_S` that holds it.
  /// Outputs come back in input order either way.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_batch_with(
    &mut self,
    texts: &[&str],
    prompt: Prompt<'_>,
    batching: Batching,
    out: &mut Vec<Embedding>,
  ) -> Result<()> {
    let split = self.options.split();
    let mut encoded: Vec<Vec<TokenId>> =
      alloc::with_capacity(texts.len(), AllocationSite::TokenIds)?;
    for text in texts {
      self
        .artifacts
        .token_ids(text, prompt, split, &mut self.scratch, &mut self.ids)?;
      if self.ids.len() > MAX_TOKENS {
        self.ids.truncate(MAX_TOKENS - 1);
        self.ids.push(TokenId::from_vocab(EOS_ID));
      }
      encoded.push(core::mem::take(&mut self.ids));
    }
    let sequences = alloc::collect(
      encoded.iter().map(|ids| Sequence::Text(ids)),
      AllocationSite::BatchOutput,
    )?;
    self.embed_sequences(&sequences, batching, out)
  }

  /// Embeds a text of any length under `prompt` as one vector: its windows
  /// ([`Self::embed_windows`]) folded under the options' `fold`, the
  /// coverage-weighted mean by default. A text that fits one window is that
  /// window's vector, embedded from exactly [`Self::embed`]'s ids.
  ///
  /// A document-level summary, for clustering, near-duplicates or a coarse
  /// first stage. For retrieval, score the windows instead: a match keeps its
  /// span.
  ///
  /// An empty text has no window to plan, so it is refused as
  /// [`Error::EmptyText`].
  ///
  /// # Errors
  /// As [`Self::embed_long_with`].
  pub fn embed_long(&mut self, text: &str, prompt: Prompt<'_>) -> Result<Embedding> {
    let (geometry, fold) = (self.options.long_text(), self.options.fold());
    self.embed_long_with(text, prompt, &geometry, &fold)
  }

  /// Embeds a text of any length window by window under `prompt` and the
  /// options' geometry: each window with its byte range and its vector.
  ///
  /// An empty text has no window to plan, so it is refused as
  /// [`Error::EmptyText`].
  ///
  /// # Errors
  /// As [`Self::embed_windows_with`].
  pub fn embed_windows(&mut self, text: &str, prompt: Prompt<'_>) -> Result<Vec<WindowEmbedding>> {
    let geometry = self.options.long_text();
    self.embed_windows_with(text, prompt, &geometry)
  }

  /// [`Self::embed_long`] under this call's `geometry` and `fold`: the
  /// windows of [`Self::embed_windows_with`], folded by windit under `fold`.
  /// One window is returned as it is.
  ///
  /// # Errors
  /// As [`Self::embed_windows_with`]; [`Error::Windowing`] if the windows
  /// cancel exactly.
  pub fn embed_long_with(
    &mut self,
    text: &str,
    prompt: Prompt<'_>,
    geometry: &LongTextOptions,
    fold: &AggregatePolicyKind,
  ) -> Result<Embedding> {
    let windows = self.embed_windows_with(text, prompt, geometry)?;
    fold_windows(windows, fold)
  }

  /// [`Self::embed_windows`] under this call's `geometry`.
  ///
  /// The windows are granite's: windit's content-aware chunker cuts on
  /// paragraphs, then sentences, then words, measuring each candidate as the
  /// joint encoding of `prompt ⧺ candidate`; the separators between chunks are
  /// reattached, so with no overlap the windows partition the text; a text
  /// with no content is one window; `max_windows` binds the final count.
  ///
  /// - **The prompt is in every window**, a custom prefix included: a window's
  ///   ids are the joint encoding of `prompt ⧺ chunk`.
  /// - **`window` counts a window's total ids**: `<bos>`, the prompt, the
  ///   content and `<eos>`. The content one holds is what the prompt leaves:
  ///   504 under `document`, 503 under `search_query`, 510 bare, at the
  ///   default 512.
  /// - **Each window is embedded as [`Self::embed_batch`] embeds a text**:
  ///   under the options' batching, so a window of at most 256 ids is packed
  ///   when text packs.
  /// - **Its span** is `Span::new(offset, ids, 512)`: its coverage is its ids
  ///   over the model's 512.
  ///
  /// # An empty text
  /// It has no window to plan, so it is refused as [`Error::EmptyText`].
  /// mediagraph's text node skips an empty text before it reaches the kit (a
  /// `NoWords` skip, mediagraph#581); this refusal stays the kit's own
  /// answer. [`Self::embed`] embeds it, as the reference does.
  ///
  /// # One character past the model's 512
  /// windit keeps a character that alone outgrows the window. Beside a prompt
  /// within a few ids of 512, it can outgrow the model too, and the window is
  /// refused as [`Error::PromptFillsWindow`] at the model's 512, never
  /// truncated.
  ///
  /// # Resource bounds
  /// `max_input_bytes` is checked before any tokenizer work, the bound to set
  /// for untrusted text. Past it, the text is encoded once whole, and each
  /// candidate's measure then re-encodes only its two edges, stopping as soon
  /// as the count passes the window. A text without spaces has no cuts, so
  /// windit's character walk over it re-encodes up to a window per step.
  ///
  /// # Errors
  /// In order: [`Error::InputTooLarge`]; [`Error::WindowOverBudget`] for a
  /// window over 512; [`Error::Windowing`] for a geometry windit refuses;
  /// [`Error::PromptFillsWindow`] when the prompt's framing leaves no id of the
  /// window for the text, or one character beside it does not fit the
  /// model's 512; [`Error::EmptyText`] for `""`;
  /// [`Error::ContentlessInputOverBudget`] for a run without content past
  /// 512 ids; [`Error::Windowing`] for `max_windows`; then any error of
  /// [`Self::embed`].
  pub fn embed_windows_with(
    &mut self,
    text: &str,
    prompt: Prompt<'_>,
    geometry: &LongTextOptions,
  ) -> Result<Vec<WindowEmbedding>> {
    let split = self.options.split();
    let framing = self.artifacts.framing(prompt, split)?;
    let planned = plan_windows(
      &self.artifacts.shared.image,
      text,
      prompt,
      split,
      geometry,
      framing,
      &mut self.scratch,
    )?;
    let sequences = alloc::collect(
      planned.iter().map(|window| Sequence::Text(&window.ids)),
      AllocationSite::WindowList,
    )?;
    let mut vectors = Vec::new();
    self.embed_sequences(&sequences, self.options.batching(), &mut vectors)?;
    let mut windows = alloc::with_capacity(planned.len(), AllocationSite::WindowList)?;
    let mut offset = 0;
    for (ordinal, (window, embedding)) in planned.into_iter().zip(vectors).enumerate() {
      let span = Span::try_new(offset, window.ids.len(), MAX_TOKENS)?;
      offset += window.ids.len();
      windows.push(WindowEmbedding::new(ordinal, window.range, span, embedding));
    }
    Ok(windows)
  }

  /// Embeds one audio window's tokens: `<bos> <|audio>` the real rows
  /// `<audio|> <eos>`, through the smallest `embed_S` that holds them.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_audio_tokens(&mut self, tokens: &AudioTokens) -> Result<Embedding> {
    let mut out = Vec::new();
    self.embed_audio_tokens_batch_with(&[tokens], Batching::Single, &mut out)?;
    Ok(out.pop().expect("one window, one vector"))
  }

  /// Embeds one image's tokens: `<bos> <|image>` the real rows `<image|>
  /// <eos>`, through the smallest `embed_S` that holds them.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_image_tokens(&mut self, tokens: &ImageTokens) -> Result<Embedding> {
    let mut out = Vec::new();
    self.embed_image_tokens_batch_with(&[tokens], Batching::Single, &mut out)?;
    Ok(out.pop().expect("one image, one vector"))
  }

  /// The door the audio face's packing option acts through: `tokens` under
  /// `batching` (the face's option, or a per-call override), packed with each
  /// other only, outputs appended in input order.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_audio_tokens_batch_with(
    &mut self,
    tokens: &[&AudioTokens],
    batching: Batching,
    out: &mut Vec<Embedding>,
  ) -> Result<()> {
    let mut rows: Vec<Vec<f16>> = alloc::with_capacity(tokens.len(), AllocationSite::BatchOutput)?;
    for window in tokens {
      window.rows_into(&mut self.media)?;
      rows.push(core::mem::take(&mut self.media));
    }
    let sequences = alloc::collect(
      rows.iter().map(|rows| {
        Sequence::Media(MediaSequence {
          open: TokenId::from_vocab(BOA_ID),
          rows,
          close: TokenId::from_vocab(EOA_ID),
        })
      }),
      AllocationSite::BatchOutput,
    )?;
    self.embed_sequences(&sequences, batching, out)
  }

  /// The door the image face's packing option acts through: `tokens` under
  /// `batching` (the face's option, or a per-call override), packed with each
  /// other only, outputs appended in input order.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_image_tokens_batch_with(
    &mut self,
    tokens: &[&ImageTokens],
    batching: Batching,
    out: &mut Vec<Embedding>,
  ) -> Result<()> {
    let mut rows: Vec<Vec<f16>> = alloc::with_capacity(tokens.len(), AllocationSite::BatchOutput)?;
    for image in tokens {
      image.rows_into(&mut self.media)?;
      rows.push(core::mem::take(&mut self.media));
    }
    let sequences = alloc::collect(
      rows.iter().map(|rows| {
        Sequence::Media(MediaSequence {
          open: TokenId::from_vocab(BOI_ID),
          rows,
          close: TokenId::from_vocab(EOI_ID),
        })
      }),
      AllocationSite::BatchOutput,
    )?;
    self.embed_sequences(&sequences, batching, out)
  }

  /// Embeds `sequences` of one face under `batching`, appending to `out` in
  /// input order.
  fn embed_sequences(
    &mut self,
    sequences: &[Sequence<'_>],
    batching: Batching,
    out: &mut Vec<Embedding>,
  ) -> Result<()> {
    reserve(out, sequences.len(), AllocationSite::BatchOutput)?;
    let mut vectors: Vec<Option<Embedding>> = alloc::collect(
      (0..sequences.len()).map(|_| None),
      AllocationSite::BatchOutput,
    )?;
    match batching {
      Batching::Single => {
        for function in TextFunction::ALL {
          let members = (0..sequences.len())
            .filter(|&i| choose(sequences[i].len(), Batching::Single) == function);
          for index in members {
            self.predict(function, &sequences[index..=index], &mut |_, vector| {
              vectors[index] = Some(vector);
              Ok(())
            })?;
          }
        }
      }
      Batching::Packed => {
        let lengths = alloc::collect(
          sequences.iter().map(Sequence::len),
          AllocationSite::PackLayout,
        )?;
        let (packs, alone) = plan_packs(&lengths)?;
        for pack in &packs {
          let members = alloc::collect(
            pack.members().iter().map(|&i| sequences[i]),
            AllocationSite::PackLayout,
          )?;
          self.predict(TextFunction::Pack256, &members, &mut |slot, vector| {
            vectors[pack.members()[slot]] = Some(vector);
            Ok(())
          })?;
        }
        for index in alone {
          let function = unpacked(sequences[index].len());
          self.predict(function, &sequences[index..=index], &mut |_, vector| {
            vectors[index] = Some(vector);
            Ok(())
          })?;
        }
      }
    }
    out.extend(
      vectors
        .into_iter()
        .map(|v| v.expect("every sequence is embedded once")),
    );
    Ok(())
  }

  /// Loads `function` if it is not loaded.
  fn load(&mut self, function: TextFunction) -> Result<&mut Loaded> {
    let index = slot(function);
    if self.functions[index].is_none() {
      let name = FunctionName::new(function.as_str()).expect("a function name is not empty");
      let model = Model::load_function_with(
        self.artifacts.shared.view.bundle(),
        name,
        &self.options.coreml(),
      )
      .map_err(|e| {
        Error::Load(LoadRefusal::new(
          Artifact::TextBundle,
          Some(Function::Text(function)),
          e,
        ))
      })?;
      let model = Checked::new(model, &contract(function)).map_err(|violation| {
        ContractRefusal::of(
          Artifact::TextBundle,
          Some(Function::Text(function)),
          violation,
        )
      })?;
      self.functions[index] = Some(Loaded {
        model,
        inputs: Inputs::new(function)?,
      });
    }
    Ok(self.functions[index].as_mut().expect("loaded just above"))
  }

  /// Runs `function` over `sequences` (one, or up to eight for `pack_256`)
  /// and hands each output row, checked and re-normalized, to `each` with its
  /// slot.
  fn predict(
    &mut self,
    function: TextFunction,
    sequences: &[Sequence<'_>],
    each: &mut dyn FnMut(usize, Embedding) -> Result<()>,
  ) -> Result<()> {
    let table = self.artifacts.shared.table.clone();
    let loaded = self.load(function)?;
    let features = match &mut loaded.inputs {
      Inputs::Embed(EmbedInputs {
        mask,
        inputs_embeds,
      }) => {
        fill_unpacked(
          &table,
          &sequences[0],
          inputs_embeds.as_slice_mut::<f16>()?,
          mask.as_slice_mut::<f16>()?,
        )?;
        loaded.model.predict_with(&[
          ("attention_mask", &*mask),
          ("inputs_embeds", &*inputs_embeds),
        ])
      }
      Inputs::Pack(PackInputs {
        attention_bias,
        inputs_embeds,
        pool,
        positions,
      }) => {
        fill_packed(
          &table,
          sequences,
          PackedInputs {
            attention_bias: attention_bias.as_slice_mut::<f16>()?,
            inputs_embeds: inputs_embeds.as_slice_mut::<f16>()?,
            pool: pool.as_slice_mut::<f16>()?,
            positions: positions.as_slice_mut::<f16>()?,
          },
        )?;
        loaded.model.predict_with(&[
          ("attention_bias", &*attention_bias),
          ("inputs_embeds", &*inputs_embeds),
          ("pool", &*pool),
          ("positions", &*positions),
        ])
      }
    };
    let failure = |e| {
      Error::Prediction(PredictionFailure::new(
        Artifact::TextBundle,
        Some(Function::Text(function)),
        e,
      ))
    };
    let mut features = features.map_err(failure)?;
    let output = features.take("embedding").ok_or_else(|| {
      failure(crate::PredictionError::MissingOutput(
        "embedding".to_owned(),
      ))
    })?;
    // The contract's output is `[1, 768]` or `[8, 768]`: at most a pack's
    // rows, held on the stack.
    let batch = output.shape()[0];
    debug_assert!(
      batch <= PACK_SLOTS,
      "a text function returns at most {PACK_SLOTS} rows"
    );
    let mut held = [0f32; PACK_SLOTS * EMBEDDING_DIM];
    let rows = &mut held[..batch.min(PACK_SLOTS) * EMBEDDING_DIM];
    output.copy_into::<f32>(rows)?;
    for (slot, row) in rows
      .as_chunks::<EMBEDDING_DIM>()
      .0
      .iter()
      .take(sequences.len())
      .enumerate()
    {
      let vector = Embedding::from_output(row).ok_or_else(|| {
        let index = row.iter().position(|v| !v.is_finite()).unwrap_or(0);
        Error::NonFiniteOutput(NonFiniteOutput::new(
          Artifact::TextBundle,
          Some(Function::Text(function)),
          slot * EMBEDDING_DIM + index,
        ))
      })?;
      each(slot, vector)?;
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests;
