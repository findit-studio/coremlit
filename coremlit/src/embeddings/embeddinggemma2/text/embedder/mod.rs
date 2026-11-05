//! The text part over its artifacts: [`TextArtifacts`], opened once per
//! process and shared, and [`TextEmbedder`], one instance's functions.

use std::{fs, path::Path, sync::Arc, time::Instant};

use half::f16;

use super::{
  super::{
    artifacts::{
      ArtifactLabel, Part,
      stamp::{PartView, Stamp},
    },
    audio::AudioPrecision,
    config::{self, BOA_ID, BOI_ID, EOA_ID, EOI_ID, EOS_ID, HIDDEN_SIZE, MAX_TOKENS, PACK_SLOTS},
    embedding::{EMBEDDING_DIM, Embedding},
    error::{
      Allocation, AllocationSite, Artifact, ContractRefusal, Error, Function, LoadRefusal,
      NonFiniteOutput, PredictionFailure, ReadFailure, Result,
    },
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
    PACK_ROWS, PackedInputs, Sequence, choose, fill_packed, fill_unpacked, plan_packs, unpacked,
  },
};
use crate::{
  DataType, FunctionName, Model, MultiArray,
  model::contract::{
    Checked, ContractViolation, Dim, FeatureContract, LoadContract, Rendered, StateContract,
  },
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
  /// stamp; [`Error::Read`] or [`Error::Config`] for `config.json`;
  /// [`Error::TableShape`] or [`Error::TableIdentity`] for the table;
  /// [`Error::TokenizerImage`] for the image.
  pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
    let dir = dir.as_ref();
    let stamp = Stamp::read(dir)?;
    let view = stamp.part(dir, Part::Text, AudioPrecision::Fp16)?;
    let config_path = dir.join("config.json");
    let json = fs::read(&config_path)
      .map_err(|e| Error::Read(ReadFailure::new(Artifact::Config, config_path.clone(), e)))?;
    config::check(&config_path, &json)?;
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

  /// The ids `prompt`'s framing takes in a window, from the loaded image.
  ///
  /// # Errors
  /// [`Error::Allocation`] if a buffer cannot grow while a title or a custom
  /// prefix is encoded.
  pub fn framing(&self, prompt: Prompt<'_>) -> Result<Framing> {
    let named = Prompt::NAMED
      .iter()
      .position(|(key, _)| prompt.config_key() == Some(*key));
    if let Some(slot) = named {
      return Ok(Framing {
        ids: self.shared.framing[slot],
        conservative: false,
      });
    }
    let mut scratch = TokenizerScratch::new();
    let mut ids = Vec::new();
    match prompt {
      Prompt::Document { title: Some(title) } => {
        self.token_ids(
          title,
          Prompt::Custom(" "),
          SplitSpecialTokens::Recognize,
          &mut scratch,
          &mut ids,
        )?;
        Ok(Framing {
          ids: 7 + ids.len() - 2,
          conservative: false,
        })
      }
      Prompt::Custom(prefix) => {
        self.token_ids(
          prefix,
          Prompt::Bare,
          SplitSpecialTokens::Recognize,
          &mut scratch,
          &mut ids,
        )?;
        Ok(Framing {
          ids: ids.len(),
          conservative: true,
        })
      }
      _ => Ok(Framing {
        ids: self.shared.framing[8],
        conservative: false,
      }),
    }
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

/// A contract violation of `function`, in the kit's vocabulary.
fn contract_refusal(function: TextFunction, violation: ContractViolation) -> Error {
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
  Error::Contract(ContractRefusal::new(
    Artifact::TextBundle,
    Some(Function::Text(function)),
    feature,
    expected,
    actual,
  ))
}

/// One loaded function and its inputs, allocated once and refilled.
#[derive(Debug)]
struct Loaded {
  model: Checked,
  inputs: Inputs,
}

#[derive(Debug)]
enum Inputs {
  Embed {
    mask: MultiArray,
    inputs_embeds: MultiArray,
  },
  Pack {
    attention_bias: MultiArray,
    inputs_embeds: MultiArray,
    pool: MultiArray,
    positions: MultiArray,
  },
}

impl Inputs {
  fn new(function: TextFunction) -> Result<Self> {
    let zeros = |shape: &[usize]| MultiArray::zeros(shape, DataType::F16);
    Ok(match function {
      TextFunction::Pack256 => Self::Pack {
        attention_bias: zeros(&[1, 1, PACK_ROWS, PACK_ROWS])?,
        inputs_embeds: zeros(&[1, PACK_ROWS, HIDDEN_SIZE])?,
        pool: zeros(&[PACK_SLOTS, PACK_ROWS])?,
        positions: zeros(&[PACK_ROWS, 1])?,
      },
      _ => Self::Embed {
        mask: zeros(&[1, function.tokens()])?,
        inputs_embeds: zeros(&[1, function.tokens(), HIDDEN_SIZE])?,
      },
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
  /// # Errors
  /// [`Error::Allocation`]; [`Error::Load`], [`Error::Contract`] or
  /// [`Error::Prediction`] for the function; [`Error::NonFiniteOutput`] for a
  /// non-finite or zero output row.
  pub fn embed(&mut self, text: &str, prompt: Prompt<'_>) -> Result<Embedding> {
    let mut out = Vec::with_capacity(1);
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
    let split = if self.options.split_special_tokens() {
      SplitSpecialTokens::Split
    } else {
      SplitSpecialTokens::Recognize
    };
    let mut encoded: Vec<Vec<TokenId>> = Vec::new();
    encoded.try_reserve_exact(texts.len()).map_err(|_| {
      Error::Allocation(Allocation::new(
        AllocationSite::TokenIds,
        texts.len() * core::mem::size_of::<Vec<TokenId>>(),
      ))
    })?;
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
    let sequences: Vec<Sequence<'_>> = encoded.iter().map(|ids| Sequence::Text(ids)).collect();
    self.embed_sequences(&sequences, batching, out)
  }

  /// Embeds one audio window's tokens: `<bos> <|audio>` the real rows
  /// `<audio|> <eos>`, through the smallest `embed_S` that holds them.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_audio_tokens(&mut self, tokens: &AudioTokens) -> Result<Embedding> {
    let mut out = Vec::with_capacity(1);
    self.embed_audio_tokens_batch_with(&[tokens], Batching::Single, &mut out)?;
    Ok(out.pop().expect("one window, one vector"))
  }

  /// Embeds one image's tokens: `<bos> <|image>` the real rows `<image|>
  /// <eos>`, through the smallest `embed_S` that holds them.
  ///
  /// # Errors
  /// As [`Self::embed`].
  pub fn embed_image_tokens(&mut self, tokens: &ImageTokens) -> Result<Embedding> {
    let mut out = Vec::with_capacity(1);
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
    let mut rows: Vec<Vec<f16>> = Vec::with_capacity(tokens.len());
    for window in tokens {
      window.rows_into(&mut self.media)?;
      rows.push(core::mem::take(&mut self.media));
    }
    let sequences: Vec<Sequence<'_>> = rows
      .iter()
      .map(|rows| Sequence::Media {
        open: TokenId::from_vocab(BOA_ID),
        rows,
        close: TokenId::from_vocab(EOA_ID),
      })
      .collect();
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
    let mut rows: Vec<Vec<f16>> = Vec::with_capacity(tokens.len());
    for image in tokens {
      image.rows_into(&mut self.media)?;
      rows.push(core::mem::take(&mut self.media));
    }
    let sequences: Vec<Sequence<'_>> = rows
      .iter()
      .map(|rows| Sequence::Media {
        open: TokenId::from_vocab(BOI_ID),
        rows,
        close: TokenId::from_vocab(EOI_ID),
      })
      .collect();
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
    out.try_reserve(sequences.len()).map_err(|_| {
      Error::Allocation(Allocation::new(
        AllocationSite::BatchOutput,
        sequences.len() * core::mem::size_of::<Embedding>(),
      ))
    })?;
    let mut vectors: Vec<Option<Embedding>> = (0..sequences.len()).map(|_| None).collect();
    match batching {
      Batching::Single => {
        for function in TextFunction::ALL {
          let members: Vec<usize> = (0..sequences.len())
            .filter(|&i| choose(sequences[i].len(), Batching::Single) == function)
            .collect();
          for index in members {
            self.predict(function, &sequences[index..=index], &mut |_, vector| {
              vectors[index] = Some(vector);
              Ok(())
            })?;
          }
        }
      }
      Batching::Packed => {
        let lengths: Vec<usize> = sequences.iter().map(Sequence::len).collect();
        let (packs, alone) = plan_packs(&lengths);
        for pack in &packs {
          let members: Vec<Sequence<'_>> = pack.members().iter().map(|&i| sequences[i]).collect();
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
      let model =
        Checked::new(model, &contract(function)).map_err(|v| contract_refusal(function, v))?;
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
      Inputs::Embed {
        mask,
        inputs_embeds,
      } => {
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
      Inputs::Pack {
        attention_bias,
        inputs_embeds,
        pool,
        positions,
      } => {
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
    let batch = output.shape()[0];
    let mut rows = vec![0f32; batch * EMBEDDING_DIM];
    output.copy_into::<f32>(&mut rows)?;
    for (slot, row) in rows
      .chunks_exact(EMBEDDING_DIM)
      .take(sequences.len())
      .enumerate()
    {
      let row: &[f32; EMBEDDING_DIM] = row.try_into().expect("a whole row");
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
