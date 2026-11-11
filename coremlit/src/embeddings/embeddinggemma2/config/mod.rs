//! `config.json`, FluidInference's description of the three packages, as a
//! compiled contract.
//!
//! Every field of the file is a constant here, and the file is checked against
//! all of them at load: a value that differs, a key that is missing and a key
//! the contract does not know are each refused by the field's dotted path. A
//! future revision of the artifacts that changes any of them is therefore a
//! code change, re-measured, and never absorbed silently.
//!
//! The document is read by serde's own derived readers into typed structs
//! that declare exactly the contract's keys, every number read by serde's own
//! impl, so a build with serde_json's `arbitrary_precision` reads it the same;
//! the values are then compared with the contract in its order. A refusal the
//! parse itself raises (a value of another type, a key the contract does not
//! know) is named by the deepest field of the contract on its path.

use std::path::Path;

use serde_json::error::Category;
use serde_path_to_error::Segment;

use super::{
  artifacts::tree::{self, PastCap, Refusal},
  contract::{
    self, Actual,
    Expect::{self, Float, Object, Str, Strs, Uint, Uints},
    Step, Wanted, typed_object,
  },
  error::{
    AllocationSite, Artifact, ConfigField, ConfigMismatch, Error, ReadFailure, Result, excerpt,
  },
  prompt::Prompt,
};

/// The largest `config.json` the kit reads: 1 MiB. The one it implements is
/// 2.8 KB; the cap holds what the read can ask for.
pub(crate) const CONFIG_JSON_CAP: u64 = 1 << 20;

/// The most ids one text function takes, `<bos>` and `<eos>` included: the
/// model's ceiling, and the largest long-text window.
pub const MAX_TOKENS: usize = 512;

/// The width of a token-table row and of an `inputs_embeds` row.
pub(crate) const HIDDEN_SIZE: usize = 512;

/// The rows of the token table: the tokenizer's vocabulary.
pub(crate) const VOCAB_SIZE: usize = 262_144;

/// The factor every gathered token row is scaled by, f32(√512):
/// `0x41B504F3`.
pub(crate) const EMBED_SCALE: f32 = 22.627_417;

/// `config.json`'s `embed_scale` literal, compared exactly as the f64 it
/// parses to. Its f32 rounding is [`EMBED_SCALE`].
const EMBED_SCALE_LITERAL: f64 = 22.627_417;

/// `<pad>`, `<eos>` and `<bos>`.
pub(crate) const PAD_ID: u32 = 0;
pub(crate) const EOS_ID: u32 = 1;
pub(crate) const BOS_ID: u32 = 2;

/// `<|image>`, `<|audio>`, `<image|>` and `<audio|>`, which frame a media
/// token sequence.
pub(crate) const BOI_ID: u32 = 255_999;
pub(crate) const BOA_ID: u32 = 256_000;
pub(crate) const EOI_ID: u32 = 258_882;
pub(crate) const EOA_ID: u32 = 258_883;

/// The sequences `pack_256` embeds in one call.
pub(crate) const PACK_SLOTS: usize = 8;

/// The audio window, its padded waveform, and what one window yields.
pub(crate) const SAMPLE_RATE_HZ: u32 = 16_000;
pub(crate) const WINDOW_SECONDS: u32 = 10;
pub(crate) const WAVEFORM_SAMPLES: usize = 160_160;
pub(crate) const LEFT_PAD_SAMPLES: usize = 160;
pub(crate) const MEL_FRAMES: usize = 1_000;
pub(crate) const AUDIO_TOKENS_PER_WINDOW: usize = 250;
pub(crate) const AUDIO_TOKEN_MS: u32 = 40;

/// The vision patch, the pooling kernel, and the position tables' shape.
pub(crate) const PATCH_SIZE: usize = 16;
pub(crate) const POOLING_KERNEL: usize = 3;
pub(crate) const POSITION_TABLES: usize = 2;
pub(crate) const POSITION_ROWS: usize = 1_024;
pub(crate) const POSITION_WIDTH: usize = 768;

/// One text function's entry: its rows.
macro_rules! embed {
  ($key:literal, $tokens:literal) => {
    (
      $key,
      concat!("functions.", $key),
      Object(&[(
        "tokens",
        concat!("functions.", $key, ".tokens"),
        Uint($tokens),
      )]),
    )
  };
}

/// One vision function's entry: its tokens and patches.
macro_rules! vision {
  ($key:literal, $tokens:literal, $patches:literal) => {
    (
      $key,
      concat!("vision.functions.", $key),
      Object(&[
        (
          "tokens",
          concat!("vision.functions.", $key, ".tokens"),
          Uint($tokens),
        ),
        (
          "patches",
          concat!("vision.functions.", $key, ".patches"),
          Uint($patches),
        ),
      ]),
    )
  };
}

/// One prompt's entry, from [`Prompt::NAMED`].
macro_rules! prompt {
  ($index:literal, $key:literal) => {
    (
      Prompt::NAMED[$index].0,
      concat!("prompts.", $key),
      Str(Prompt::NAMED[$index].1),
    )
  };
}

/// The whole of `config.json` at FluidInference @ `3f978dd`.
const CONTRACT: Expect = Object(&[
  ("base_model", "base_model", Str("google/embeddinggemma-2")),
  ("format", "format", Str("coreml")),
  ("modality", "modality", Strs(&["text", "audio", "image"])),
  ("precision", "precision", Str("fp16")),
  ("hidden_size", "hidden_size", Uint(HIDDEN_SIZE as u64)),
  ("embedding_dim", "embedding_dim", Uint(768)),
  (
    "matryoshka_dims",
    "matryoshka_dims",
    Uints(&[768, 512, 256, 128]),
  ),
  ("embed_scale", "embed_scale", Float(EMBED_SCALE_LITERAL)),
  (
    "token_table",
    "token_table",
    Object(&[
      ("file", "token_table.file", Str("embeddings.bf16")),
      ("dtype", "token_table.dtype", Str("bfloat16")),
      (
        "shape",
        "token_table.shape",
        Uints(&[VOCAB_SIZE as u64, HIDDEN_SIZE as u64]),
      ),
    ]),
  ),
  (
    "functions",
    "functions",
    Object(&[
      embed!("embed_32", 32),
      embed!("embed_48", 48),
      embed!("embed_64", 64),
      embed!("embed_128", 128),
      embed!("embed_256", 256),
      embed!("embed_512", 512),
      (
        "pack_256",
        "functions.pack_256",
        Object(&[
          ("tokens", "functions.pack_256.tokens", Uint(256)),
          ("slots", "functions.pack_256.slots", Uint(PACK_SLOTS as u64)),
        ]),
      ),
    ]),
  ),
  ("bos_token_id", "bos_token_id", Uint(BOS_ID as u64)),
  ("eos_token_id", "eos_token_id", Uint(EOS_ID as u64)),
  ("pad_token_id", "pad_token_id", Uint(PAD_ID as u64)),
  (
    "prompts",
    "prompts",
    Object(&[
      prompt!(0, "document"),
      prompt!(1, "search_query"),
      prompt!(2, "question_answering"),
      prompt!(3, "fact_checking"),
      prompt!(4, "code_retrieval"),
      prompt!(5, "classification"),
      prompt!(6, "clustering"),
      prompt!(7, "sentence_similarity"),
    ]),
  ),
  (
    "audio",
    "audio",
    Object(&[
      (
        "package",
        "audio.package",
        Str("EmbeddingGemma2Audio.mlpackage"),
      ),
      ("compute_units", "audio.compute_units", Str("cpuAndGPU")),
      ("precision", "audio.precision", Str("fp16")),
      (
        "sample_rate",
        "audio.sample_rate",
        Uint(SAMPLE_RATE_HZ as u64),
      ),
      (
        "window_seconds",
        "audio.window_seconds",
        Uint(WINDOW_SECONDS as u64),
      ),
      (
        "waveform_samples",
        "audio.waveform_samples",
        Uint(WAVEFORM_SAMPLES as u64),
      ),
      (
        "left_pad_samples",
        "audio.left_pad_samples",
        Uint(LEFT_PAD_SAMPLES as u64),
      ),
      ("frames", "audio.frames", Uint(MEL_FRAMES as u64)),
      (
        "frame_valid_if",
        "audio.frame_valid_if",
        Str("i * 160 + 321 <= 160 + audio_samples"),
      ),
      (
        "tokens_per_window",
        "audio.tokens_per_window",
        Uint(AUDIO_TOKENS_PER_WINDOW as u64),
      ),
      ("token_ms", "audio.token_ms", Uint(AUDIO_TOKEN_MS as u64)),
      ("boa_token_id", "audio.boa_token_id", Uint(BOA_ID as u64)),
      ("eoa_token_id", "audio.eoa_token_id", Uint(EOA_ID as u64)),
      (
        "text_sequence",
        "audio.text_sequence",
        Str("<bos> <|audio> [audio tokens] <audio|> <eos> -> embed_256"),
      ),
    ]),
  ),
  (
    "vision",
    "vision",
    Object(&[
      (
        "package",
        "vision.package",
        Str("EmbeddingGemma2Vision.mlpackage"),
      ),
      ("compute_units", "vision.compute_units", Str("cpuAndGPU")),
      ("precision", "vision.precision", Str("fp16")),
      (
        "functions",
        "vision.functions",
        Object(&[
          vision!("vision_70", 70, 630),
          vision!("vision_140", 140, 1260),
          vision!("vision_280", 280, 2520),
        ]),
      ),
      ("patch_size", "vision.patch_size", Uint(PATCH_SIZE as u64)),
      (
        "pooling_kernel",
        "vision.pooling_kernel",
        Uint(POOLING_KERNEL as u64),
      ),
      (
        "resize",
        "vision.resize",
        Str(
          "aspect-preserving, bicubic + antialias, sides multiple of 48 px, at most 9 x tokens patches",
        ),
      ),
      (
        "pixel_scale",
        "vision.pixel_scale",
        Str("RGB / 255 (the model applies 2x - 1)"),
      ),
      (
        "patch_layout",
        "vision.patch_layout",
        Str("row-major patches, each [16 rows][16 cols][RGB]"),
      ),
      (
        "position_table",
        "vision.position_table",
        Object(&[
          (
            "file",
            "vision.position_table.file",
            Str("position_embeddings.f16"),
          ),
          ("dtype", "vision.position_table.dtype", Str("float16")),
          (
            "shape",
            "vision.position_table.shape",
            Uints(&[
              POSITION_TABLES as u64,
              POSITION_ROWS as u64,
              POSITION_WIDTH as u64,
            ]),
          ),
          (
            "use",
            "vision.position_table.use",
            Str("table[0][x] + table[1][y] per patch, zero for padding"),
          ),
        ]),
      ),
      ("boi_token_id", "vision.boi_token_id", Uint(BOI_ID as u64)),
      ("eoi_token_id", "vision.eoi_token_id", Uint(EOI_ID as u64)),
      (
        "text_sequence",
        "vision.text_sequence",
        Str("<bos> <|image> [image tokens] <image|> <eos> -> smallest embed_S that fits"),
      ),
    ]),
  ),
]);

typed_object! {
  /// `config.json` as serde reads it: the contract's keys and no other.
  struct Document {
    "base_model" => base_model: String,
    "format" => format: String,
    "modality" => modality: Vec<String>,
    "precision" => precision: String,
    "hidden_size" => hidden_size: u64,
    "embedding_dim" => embedding_dim: u64,
    "matryoshka_dims" => matryoshka_dims: Vec<u64>,
    "embed_scale" => embed_scale: f64,
    "token_table" => token_table: TokenTable,
    "functions" => functions: Functions,
    "bos_token_id" => bos_token_id: u64,
    "eos_token_id" => eos_token_id: u64,
    "pad_token_id" => pad_token_id: u64,
    "prompts" => prompts: Prompts,
    "audio" => audio: Audio,
    "vision" => vision: Vision,
  }
}

typed_object! {
  struct TokenTable {
    "file" => file: String,
    "dtype" => dtype: String,
    "shape" => shape: Vec<u64>,
  }
}

typed_object! {
  struct Functions {
    "embed_32" => embed_32: Embed,
    "embed_48" => embed_48: Embed,
    "embed_64" => embed_64: Embed,
    "embed_128" => embed_128: Embed,
    "embed_256" => embed_256: Embed,
    "embed_512" => embed_512: Embed,
    "pack_256" => pack_256: Pack,
  }
}

typed_object! {
  struct Embed {
    "tokens" => tokens: u64,
  }
}

typed_object! {
  struct Pack {
    "tokens" => tokens: u64,
    "slots" => slots: u64,
  }
}

typed_object! {
  struct Prompts {
    "document" => document: String,
    "search_query" => search_query: String,
    "question_answering" => question_answering: String,
    "fact_checking" => fact_checking: String,
    "code_retrieval" => code_retrieval: String,
    "classification" => classification: String,
    "clustering" => clustering: String,
    "sentence_similarity" => sentence_similarity: String,
  }
}

typed_object! {
  struct Audio {
    "package" => package: String,
    "compute_units" => compute_units: String,
    "precision" => precision: String,
    "sample_rate" => sample_rate: u64,
    "window_seconds" => window_seconds: u64,
    "waveform_samples" => waveform_samples: u64,
    "left_pad_samples" => left_pad_samples: u64,
    "frames" => frames: u64,
    "frame_valid_if" => frame_valid_if: String,
    "tokens_per_window" => tokens_per_window: u64,
    "token_ms" => token_ms: u64,
    "boa_token_id" => boa_token_id: u64,
    "eoa_token_id" => eoa_token_id: u64,
    "text_sequence" => text_sequence: String,
  }
}

typed_object! {
  struct Vision {
    "package" => package: String,
    "compute_units" => compute_units: String,
    "precision" => precision: String,
    "functions" => functions: VisionFunctions,
    "patch_size" => patch_size: u64,
    "pooling_kernel" => pooling_kernel: u64,
    "resize" => resize: String,
    "pixel_scale" => pixel_scale: String,
    "patch_layout" => patch_layout: String,
    "position_table" => position_table: PositionTable,
    "boi_token_id" => boi_token_id: u64,
    "eoi_token_id" => eoi_token_id: u64,
    "text_sequence" => text_sequence: String,
  }
}

typed_object! {
  struct VisionFunctions {
    "vision_70" => vision_70: VisionFunction,
    "vision_140" => vision_140: VisionFunction,
    "vision_280" => vision_280: VisionFunction,
  }
}

typed_object! {
  struct VisionFunction {
    "tokens" => tokens: u64,
    "patches" => patches: u64,
  }
}

typed_object! {
  struct PositionTable {
    "file" => file: String,
    "dtype" => dtype: String,
    "shape" => shape: Vec<u64>,
    "use" => use_: String,
  }
}

/// The refusal of a `config.json` past its cap, at its size or while it was
/// read.
pub(crate) fn past_cap(past: PastCap) -> Error {
  mismatch(
    "",
    format!("a document of at most {CONFIG_JSON_CAP} bytes, still while it is read"),
    past.sentence("`config.json`"),
  )
}

/// Reads the `config.json` at `path` through the read door, at most
/// [`CONFIG_JSON_CAP`] bytes, and checks it against the compiled contract.
///
/// # Errors
/// [`Error::Config`] at the top level for a document past its cap, refused
/// before a byte is read, or one that grows while it is read;
/// [`Error::Traversal`] for a broken link on its path; [`Error::Allocation`]
/// if its buffer cannot be reserved; [`Error::Read`] if it is missing or
/// cannot be read; then as [`check`].
pub(crate) fn load(path: &Path) -> Result<()> {
  let read = |e| Error::Read(ReadFailure::new(Artifact::Config, path.to_path_buf(), e));
  let json = tree::read(path, CONFIG_JSON_CAP, AllocationSite::ArtifactRead).map_err(
    |refusal| match refusal {
      Refusal::Absent => read(std::io::ErrorKind::NotFound.into()),
      Refusal::PastCap(past) => past_cap(past),
      Refusal::Named(error) => error,
      Refusal::Failed(e) => read(e),
    },
  )?;
  check(path, &json)
}

/// Checks the `config.json` read from `path` against the compiled contract:
/// parsed into [`Document`], then compared with the contract in its order.
///
/// # Errors
/// [`Error::Read`] if `json` is not JSON; [`Error::Config`] naming the first
/// field, in the contract's order, whose value differs or that is missing;
/// [`Error::Config`] naming the deepest field on the path of a refusal the
/// parse raises: a value of another type, or a key the contract does not
/// know, named by the object that carries it.
pub(crate) fn check(path: &Path, json: &[u8]) -> Result<()> {
  let read = |e: serde_json::Error| {
    Error::Read(ReadFailure::new(
      Artifact::Config,
      path.to_path_buf(),
      std::io::Error::from(e),
    ))
  };
  let mut document = serde_json::Deserializer::from_slice(json);
  let parsed: Document = serde_path_to_error::deserialize(&mut document).map_err(|refused| {
    let (field, wanted) = contract::field_at(
      &CONTRACT,
      refused.path().iter().map_while(|segment| match segment {
        Segment::Map { key } => Some(Step::Key(key)),
        Segment::Seq { index } => Some(Step::Index(*index)),
        Segment::Enum { .. } | Segment::Unknown => None,
      }),
    );
    let refused = refused.into_inner();
    match refused.classify() {
      Category::Data => mismatch(field, Wanted(wanted).to_string(), quoted(&refused)),
      Category::Io | Category::Syntax | Category::Eof => read(refused),
    }
  })?;
  document.end().map_err(read)?;
  match contract::compare(&CONTRACT, "", Actual::Object(&parsed)) {
    Some(difference) => Err(mismatch(
      difference.path,
      difference.expected,
      difference.actual,
    )),
    None => Ok(()),
  }
}

/// serde's refusal, at most 64 bytes of it (it can quote the document), then
/// where in the document it stands.
fn quoted(refused: &serde_json::Error) -> String {
  let text = refused.to_string();
  let at = format!(" at line {} column {}", refused.line(), refused.column());
  match text.strip_suffix(&at) {
    Some(message) => format!("{}{at}", excerpt(message)),
    None => excerpt(&text),
  }
}

fn mismatch(path: &'static str, expected: String, actual: String) -> Error {
  Error::Config(ConfigMismatch::new(
    ConfigField::new(path),
    expected,
    actual,
  ))
}

#[cfg(test)]
mod tests;
