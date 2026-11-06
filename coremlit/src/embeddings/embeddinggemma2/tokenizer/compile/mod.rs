//! The tokenizer image's compiler: `tokenizer.json` in, format-1 image bytes
//! out.
//!
//! A pure function. It refuses by name anything outside the one
//! configuration the kit's tokenizer implements, validates the vocabulary and
//! the merges, re-derives from the merges the word-boundary facts the
//! runtime's piece split rests on, and lays the tables out (see the image
//! module for the format).

use std::{borrow::Cow, collections::HashMap};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::super::{
  alloc,
  error::{AllocationSite, Error, Result, SourceField, TokenizerSourceDefect},
};
use super::image::{
  Counts, HEADER_SIZE, MAGIC, Section, TOKENIZER_IMAGE_VERSION, header, scalar_slot,
};

/// `▁`, the character every space becomes.
const SPACE_MARK: char = '\u{2581}';

/// A string borrowed from the document when it carries no escape.
#[derive(Deserialize)]
struct Text<'a>(#[serde(borrow)] Cow<'a, str>);

#[derive(Deserialize)]
struct AddedToken<'a> {
  id: u32,
  #[serde(borrow)]
  content: Text<'a>,
  single_word: bool,
  lstrip: bool,
  rstrip: bool,
  normalized: bool,
  special: bool,
}

#[derive(Deserialize)]
struct Model<'a> {
  #[serde(rename = "type")]
  kind: String,
  dropout: Option<f64>,
  unk_token: Option<String>,
  continuing_subword_prefix: Option<String>,
  end_of_word_suffix: Option<String>,
  fuse_unk: bool,
  byte_fallback: bool,
  ignore_merges: bool,
  #[serde(borrow, deserialize_with = "vocab_in_order")]
  vocab: Vec<(Text<'a>, u32)>,
  #[serde(borrow)]
  merges: Vec<(Text<'a>, Text<'a>)>,
}

#[derive(Deserialize)]
struct Document<'a> {
  #[serde(borrow)]
  added_tokens: Vec<AddedToken<'a>>,
  normalizer: Value,
  pre_tokenizer: Value,
  post_processor: Value,
  truncation: Value,
  padding: Value,
  #[serde(borrow)]
  model: Model<'a>,
}

/// The vocabulary object `{token: id}`, kept as a list in document order.
fn vocab_in_order<'de, D>(deserializer: D) -> core::result::Result<Vec<(Text<'de>, u32)>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  struct Entries;

  impl<'de> serde::de::Visitor<'de> for Entries {
    type Value = Vec<(Text<'de>, u32)>;

    fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      f.write_str("a map from token to id")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(
      self,
      mut map: A,
    ) -> core::result::Result<Self::Value, A::Error> {
      let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
      while let Some(entry) = map.next_entry::<Text<'de>, u32>()? {
        entries.push(entry);
      }
      Ok(entries)
    }
  }

  deserializer.deserialize_map(Entries)
}

fn refuse(field: SourceField, detail: impl Into<String>) -> Error {
  Error::TokenizerSource(TokenizerSourceDefect::new(field, detail.into()))
}

fn require(field: SourceField, got: &Value, want: Value) -> Result<()> {
  if *got == want {
    Ok(())
  } else {
    Err(refuse(field, format!("expected {want}, got {got}")))
  }
}

/// The checks of the one configuration the runtime implements.
fn check_configuration(doc: &Document<'_>) -> Result<()> {
  require(SourceField::Document, &doc.truncation, Value::Null)?;
  require(SourceField::Document, &doc.padding, Value::Null)?;
  require(
    SourceField::Normalizer,
    &doc.normalizer,
    json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}),
  )?;
  require(
    SourceField::PreTokenizer,
    &doc.pre_tokenizer,
    json!({"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false}),
  )?;
  require(
    SourceField::PostProcessor,
    doc.post_processor.get("type").unwrap_or(&Value::Null),
    json!("TemplateProcessing"),
  )?;
  require(
    SourceField::PostProcessor,
    doc.post_processor.get("single").unwrap_or(&Value::Null),
    json!([
      {"SpecialToken": {"id": "<bos>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "<eos>", "type_id": 0}}
    ]),
  )?;
  let model = &doc.model;
  let flags_hold = model.kind == "BPE"
    && model.dropout.is_none()
    && model.unk_token.as_deref() == Some("<unk>")
    && model.continuing_subword_prefix.is_none()
    && model.end_of_word_suffix.is_none()
    && model.fuse_unk
    && model.byte_fallback
    && !model.ignore_merges;
  if !flags_hold {
    return Err(refuse(
      SourceField::Model,
      "expected BPE with byte_fallback and fuse_unk on, ignore_merges off, no dropout, no affixes, and `<unk>` as the unknown token",
    ));
  }
  for token in &doc.added_tokens {
    let literal = &token.content.0;
    if !token.special || token.normalized || token.lstrip || token.rstrip || token.single_word {
      return Err(refuse(
        SourceField::AddedTokens,
        format!(
          "`{literal}` is not a plain special token (special, not normalized, no strip, not single-word)"
        ),
      ));
    }
    if !literal.starts_with('<') {
      return Err(refuse(
        SourceField::AddedTokens,
        format!("`{literal}` does not begin with `<`, which the runtime's special scan relies on"),
      ));
    }
  }
  Ok(())
}

/// The format-1 image of `tokenizer.json`'s bytes.
///
/// # Errors
/// [`Error::TokenizerSource`] naming the part of the document outside the
/// configuration the kit implements, or a vocabulary or merge it cannot hold;
/// [`Error::Allocation`] if the image cannot be reserved.
pub(crate) fn compile_image(json: &[u8]) -> Result<Vec<u8>> {
  let doc: Document<'_> =
    serde_json::from_slice(json).map_err(|e| refuse(SourceField::Document, e.to_string()))?;
  check_configuration(&doc)?;

  // Dense ids, each token once.
  let vocab_len = doc.model.vocab.len();
  let mut tokens: Vec<Option<&str>> = vec![None; vocab_len];
  let mut ids: HashMap<&str, u32> = HashMap::with_capacity(vocab_len);
  for (text, id) in &doc.model.vocab {
    let slot = tokens.get_mut(*id as usize).ok_or_else(|| {
      refuse(
        SourceField::Vocabulary,
        format!("id {id} is outside the {vocab_len} ids"),
      )
    })?;
    if slot.is_some() || ids.insert(&text.0, *id).is_some() {
      return Err(refuse(
        SourceField::Vocabulary,
        format!("`{}` or id {id} appears twice", text.0),
      ));
    }
    *slot = Some(&text.0);
  }
  let tokens: Vec<&str> = tokens
    .into_iter()
    .enumerate()
    .map(|(id, token)| {
      token.ok_or_else(|| refuse(SourceField::Vocabulary, format!("id {id} has no token")))
    })
    .collect::<Result<_>>()?;
  for token in &doc.added_tokens {
    if ids.get(token.content.0.as_ref()) != Some(&token.id) {
      return Err(refuse(
        SourceField::AddedTokens,
        format!(
          "`{}` is not the vocabulary's id {}",
          token.content.0, token.id
        ),
      ));
    }
  }

  let mut byte_ids = [0u32; 256];
  for (byte, slot) in byte_ids.iter_mut().enumerate() {
    let literal = format!("<0x{byte:02X}>");
    *slot = *ids
      .get(literal.as_str())
      .ok_or_else(|| refuse(SourceField::ByteFallback, format!("`{literal}` is missing")))?;
  }

  // Merges: every side and every concatenation a token; the word-boundary
  // facts re-derived; the bridging merges recorded.
  let mut entries: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(doc.model.merges.len());
  let mut bridges: Vec<(u32, u32)> = Vec::new();
  let mut joined = String::new();
  for (rank, (left, right)) in doc.model.merges.iter().enumerate() {
    let (left, right) = (left.0.as_ref(), right.0.as_ref());
    let id = |text: &str| {
      ids.get(text).copied().ok_or_else(|| {
        refuse(
          SourceField::Merges,
          format!("merge {rank}: `{text}` is not a token"),
        )
      })
    };
    joined.clear();
    joined.push_str(left);
    joined.push_str(right);
    let (l, r, m) = (id(left)?, id(right)?, id(&joined)?);
    let left_ends_in_mark = left.ends_with(SPACE_MARK);
    if left_ends_in_mark && !left.chars().all(|c| c == SPACE_MARK) {
      return Err(refuse(
        SourceField::Merges,
        format!(
          "merge {rank}: `{left}` ends in `▁` after another scalar, which crosses a word boundary the piece split relies on"
        ),
      ));
    }
    if !left_ends_in_mark && right.starts_with(SPACE_MARK) {
      bridges.push((l, r));
    }
    entries.push((l, r, rank as u32, m));
  }
  entries.sort_unstable_by_key(|&(l, r, _, _)| (l, r));
  if let Some(pair) = entries
    .windows(2)
    .find(|w| (w[0].0, w[0].1) == (w[1].0, w[1].1))
  {
    return Err(refuse(
      SourceField::Merges,
      format!(
        "the pair ({}, {}) is merged twice",
        tokens[pair[0].0 as usize], tokens[pair[0].1 as usize]
      ),
    ));
  }

  // Single-scalar tokens into an open-addressing index.
  let scalars: Vec<(u32, u32)> = tokens
    .iter()
    .enumerate()
    .filter_map(|(id, text)| {
      let mut chars = text.chars();
      match (chars.next(), chars.next()) {
        (Some(c), None) => Some((c as u32, id as u32)),
        _ => None,
      }
    })
    .collect();
  let slots = (scalars.len() + scalars.len() / 2 + 1)
    .next_power_of_two()
    .max(16);
  let bits = slots.trailing_zeros();
  let mut index = vec![(u32::MAX, u32::MAX); slots];
  for &(scalar, id) in &scalars {
    let mut slot = scalar_slot(scalar, bits);
    while index[slot].0 != u32::MAX {
      slot = (slot + 1) & (slots - 1);
    }
    index[slot] = (scalar, id);
  }

  let mut added: Vec<(usize, u32)> = doc
    .added_tokens
    .iter()
    .map(|t| (t.content.0.len(), t.id))
    .collect();
  added.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

  // The sections, in table order.
  let mut sections: [Vec<u8>; 8] = Default::default();
  let mut offsets = Vec::with_capacity((vocab_len + 1) * 4);
  let mut arena = Vec::new();
  for text in &tokens {
    offsets.extend_from_slice(&(arena.len() as u32).to_le_bytes());
    arena.extend_from_slice(text.as_bytes());
  }
  offsets.extend_from_slice(&(arena.len() as u32).to_le_bytes());
  sections[Section::Arena as usize] = arena;
  sections[Section::TokenOffsets as usize] = offsets;
  sections[Section::ScalarIndex as usize] = index
    .iter()
    .flat_map(|&(s, id)| [s.to_le_bytes(), id.to_le_bytes()])
    .flatten()
    .collect();
  let mut merge_offsets = Vec::with_capacity((vocab_len + 1) * 4);
  let mut cursor = 0usize;
  for left in 0..=vocab_len {
    while cursor < entries.len() && (entries[cursor].0 as usize) < left {
      cursor += 1;
    }
    merge_offsets.extend_from_slice(&(cursor as u32).to_le_bytes());
  }
  sections[Section::MergeOffsets as usize] = merge_offsets;
  sections[Section::MergeEntries as usize] = entries
    .iter()
    .flat_map(|&(_, r, rank, m)| [r.to_le_bytes(), rank.to_le_bytes(), m.to_le_bytes()])
    .flatten()
    .collect();
  sections[Section::ByteFallback as usize] =
    byte_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
  sections[Section::AddedTokens as usize] =
    added.iter().flat_map(|(_, id)| id.to_le_bytes()).collect();
  sections[Section::Bridges as usize] = bridges
    .iter()
    .flat_map(|&(l, r)| [l.to_le_bytes(), r.to_le_bytes()])
    .flatten()
    .collect();

  let counts = Counts {
    vocab: vocab_len as u32,
    merges: entries.len() as u32,
    added: added.len() as u32,
    scalars: scalars.len() as u32,
    bridges: bridges.len() as u32,
  };

  // Lay out: header, then each section 8-byte aligned.
  let mut table = [(0u64, 0u64); 8];
  let mut end = HEADER_SIZE;
  for (slot, section) in table.iter_mut().zip(&sections) {
    *slot = (end as u64, section.len() as u64);
    end = (end + section.len()).next_multiple_of(8);
  }
  let mut image = alloc::with_capacity(end, AllocationSite::ImageBuild)?;
  image.resize(HEADER_SIZE, 0);
  for (section, &(offset, _)) in sections.iter().zip(&table) {
    image.resize(offset as usize, 0);
    image.extend_from_slice(section);
  }
  image.resize(end, 0);

  image[..8].copy_from_slice(&MAGIC);
  image[header::VERSION..header::VERSION + 4]
    .copy_from_slice(&TOKENIZER_IMAGE_VERSION.to_le_bytes());
  image[header::HEADER_SIZE..header::HEADER_SIZE + 4]
    .copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
  image[header::SOURCE_DIGEST..header::SOURCE_DIGEST + 32].copy_from_slice(&Sha256::digest(json));
  for (i, count) in counts.as_array().iter().enumerate() {
    let at = header::COUNTS + i * 4;
    image[at..at + 4].copy_from_slice(&count.to_le_bytes());
  }
  for (i, (offset, length)) in table.iter().enumerate() {
    let at = header::SECTIONS + i * 16;
    image[at..at + 8].copy_from_slice(&offset.to_le_bytes());
    image[at + 8..at + 16].copy_from_slice(&length.to_le_bytes());
  }
  let digest = Sha256::digest(&image[HEADER_SIZE..]);
  image[header::IMAGE_DIGEST..header::IMAGE_DIGEST + 32].copy_from_slice(&digest);
  Ok(image)
}

/// The content-addressed name of the image compiled from a `tokenizer.json`
/// whose SHA-256 is `source`: `tokenizer.v<format>-<first 12 hex>.gemmabpe`.
pub(crate) fn image_name(source: &[u8; 32]) -> String {
  let hex: String = source[..6].iter().map(|b| format!("{b:02x}")).collect();
  format!("tokenizer.v{TOKENIZER_IMAGE_VERSION}-{hex}.gemmabpe")
}
