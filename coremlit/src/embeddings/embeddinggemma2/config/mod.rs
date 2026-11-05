//! `config.json`, FluidInference's description of the three packages, as a
//! compiled contract.
//!
//! Every field of the file is a constant here, and the file is checked against
//! all of them at load: a value that differs, a key that is missing and a key
//! the contract does not know are each refused by the field's dotted path. A
//! future revision of the artifacts that changes any of them is therefore a
//! code change, re-measured, and never absorbed silently.

// The text, audio and vision parts read these constants and run the check
// as they land; until then the tests are their only reader.
#![cfg_attr(not(test), allow(dead_code))]

use std::path::Path;

use serde_json::Value;

use super::{
  error::{Artifact, ConfigField, ConfigMismatch, Error, ReadFailure, Result},
  prompt::Prompt,
};

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

/// What one field of the contract must hold.
enum Expect {
  /// A non-negative integer.
  Uint(u64),
  /// A number, compared exactly as the f64 it parses to.
  Float(f64),
  /// A string, byte for byte.
  Str(&'static str),
  /// An array of non-negative integers.
  Uints(&'static [u64]),
  /// An array of strings.
  Strs(&'static [&'static str]),
  /// An object with exactly these keys: `(key, dotted path, value)`.
  Object(&'static [(&'static str, &'static str, Expect)]),
}

use Expect::{Float, Object, Str, Strs, Uint, Uints};

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

/// Checks the `config.json` read from `path` against the compiled contract.
///
/// # Errors
/// [`Error::Read`] if `json` is not JSON; [`Error::Config`] naming the first
/// field, in the contract's order, whose value differs or that is missing, or
/// naming an object that carries a key the contract does not know.
pub(crate) fn check(path: &Path, json: &[u8]) -> Result<()> {
  let value: Value = serde_json::from_slice(json).map_err(|e| {
    Error::Read(ReadFailure::new(
      Artifact::Config,
      path.to_path_buf(),
      std::io::Error::from(e),
    ))
  })?;
  check_value(&CONTRACT, "", &value)
}

fn check_value(expect: &Expect, path: &'static str, actual: &Value) -> Result<()> {
  let holds = match (expect, actual) {
    (Uint(n), _) => actual.as_u64() == Some(*n),
    (Float(x), Value::Number(number)) => number.as_f64() == Some(*x),
    (Float(_), _) => false,
    (Str(s), _) => actual.as_str() == Some(*s),
    (Uints(ns), Value::Array(items)) => {
      items.len() == ns.len()
        && items
          .iter()
          .zip(*ns)
          .all(|(item, n)| item.as_u64() == Some(*n))
    }
    (Strs(ss), Value::Array(items)) => {
      items.len() == ss.len()
        && items
          .iter()
          .zip(*ss)
          .all(|(item, s)| item.as_str() == Some(*s))
    }
    (Uints(_) | Strs(_), _) => false,
    (Object(entries), Value::Object(object)) => {
      for (key, entry_path, entry) in *entries {
        match object.get(*key) {
          Some(value) => check_value(entry, entry_path, value)?,
          None => return Err(mismatch(entry_path, render(entry), "absent".to_owned())),
        }
      }
      if let Some(extra) = object
        .keys()
        .find(|key| !entries.iter().any(|(known, _, _)| *known == key.as_str()))
      {
        let known: Vec<&str> = entries.iter().map(|(key, _, _)| *key).collect();
        return Err(mismatch(
          path,
          format!("exactly the keys {known:?}"),
          format!("also the key {extra:?}"),
        ));
      }
      true
    }
    (Object(_), _) => false,
  };
  if holds {
    Ok(())
  } else {
    Err(mismatch(path, render(expect), actual.to_string()))
  }
}

fn mismatch(path: &'static str, expected: String, actual: String) -> Error {
  Error::Config(ConfigMismatch::new(
    ConfigField::new(path),
    expected,
    actual,
  ))
}

/// The contract's value at a field, as JSON.
fn render(expect: &Expect) -> String {
  match expect {
    Uint(n) => n.to_string(),
    Float(x) => Value::from(*x).to_string(),
    Str(s) => Value::from(*s).to_string(),
    Uints(ns) => Value::from(ns.to_vec()).to_string(),
    Strs(ss) => Value::from(ss.to_vec()).to_string(),
    Object(entries) => {
      let object: serde_json::Map<String, Value> = entries
        .iter()
        .map(|(key, _, entry)| {
          let value = serde_json::from_str(&render(entry)).unwrap_or(Value::Null);
          ((*key).to_owned(), value)
        })
        .collect();
      Value::Object(object).to_string()
    }
  }
}

#[cfg(test)]
mod tests;
