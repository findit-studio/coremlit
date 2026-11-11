//! The tokenizer image's compiler: `tokenizer.json` in, format-1 image bytes
//! out.
//!
//! A pure function. It refuses by name anything outside the one
//! configuration the kit's tokenizer implements, validates the vocabulary and
//! the merges, re-derives from the merges the word-boundary facts the
//! runtime's piece split rests on, and lays the tables out (see the image
//! module for the format).
//!
//! The parse is typed throughout and builds no `serde_json::Value`: the five
//! configuration objects are walked against their contracts as they parse
//! (`contract::Walk`), and the vocabulary, the merges and the added tokens
//! grow through the door at [`AllocationSite::SourceParse`]. What the parse
//! allocates beyond those lists is the document's escaped strings, unescaped
//! once each, at most the document's own size.

use core::{cell::Cell, marker::PhantomData};
use std::{borrow::Cow, collections::HashMap};

use serde::{
  Deserialize,
  de::{self, DeserializeSeed, Deserializer, SeqAccess, Visitor},
};
use sha2::{Digest, Sha256};

use super::{
  super::{
    alloc,
    artifacts::tree::PastCap,
    contract::{
      Difference,
      Expect::{self, Bool, Elements, Null, Object, Open, Str, Uint},
      Walk,
    },
    error::{
      Allocation, AllocationSite, Error, Result, SourceField, TokenizerSourceDefect, excerpt,
    },
  },
  image::{Counts, HEADER_SIZE, MAGIC, Section, TOKENIZER_IMAGE_VERSION, header, scalar_slot},
};

/// `▁`, the character every space becomes.
const SPACE_MARK: char = '\u{2581}';

/// The largest `tokenizer.json` the kit reads: 64 MiB. The one it implements
/// is 32.2 MB, pinned by digest in the lock row, and a new revision needs a
/// new row anyway; the cap holds what the read can ask for.
pub(crate) const TOKENIZER_JSON_CAP: u64 = 64 << 20;

/// The refusal of a `tokenizer.json` past its cap, at its size or while it
/// was read.
pub(crate) fn past_cap(past: PastCap) -> Error {
  refuse(SourceField::Document, past.sentence("`tokenizer.json`"))
}

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
  #[serde(rename = "type", borrow)]
  kind: Text<'a>,
  dropout: Option<f64>,
  #[serde(borrow)]
  unk_token: Option<Text<'a>>,
  #[serde(borrow)]
  continuing_subword_prefix: Option<Text<'a>>,
  #[serde(borrow)]
  end_of_word_suffix: Option<Text<'a>>,
  fuse_unk: bool,
  byte_fallback: bool,
  ignore_merges: bool,
  #[serde(borrow, deserialize_with = "vocab_in_order")]
  vocab: Vec<(Text<'a>, u32)>,
  #[serde(borrow, deserialize_with = "list")]
  merges: Vec<(Text<'a>, Text<'a>)>,
}

#[derive(Deserialize)]
struct Document<'a> {
  #[serde(borrow, deserialize_with = "list")]
  added_tokens: Vec<AddedToken<'a>>,
  #[serde(deserialize_with = "normalizer")]
  normalizer: Option<Difference>,
  #[serde(deserialize_with = "pre_tokenizer")]
  pre_tokenizer: Option<Difference>,
  #[serde(deserialize_with = "post_processor")]
  post_processor: Option<Difference>,
  #[serde(deserialize_with = "truncation")]
  truncation: Option<Difference>,
  #[serde(deserialize_with = "padding")]
  padding: Option<Difference>,
  #[serde(borrow)]
  model: Model<'a>,
}

/// The normalizer the runtime implements: every space to `▁`.
const NORMALIZER: Expect = Object(&[
  ("type", "normalizer.type", Str("Replace")),
  (
    "pattern",
    "normalizer.pattern",
    Object(&[("String", "normalizer.pattern.String", Str(" "))]),
  ),
  ("content", "normalizer.content", Str("\u{2581}")),
]);

/// The pre-tokenizer the runtime implements: a split on spaces, each kept
/// with the piece before it.
const PRE_TOKENIZER: Expect = Object(&[
  ("type", "pre_tokenizer.type", Str("Split")),
  (
    "pattern",
    "pre_tokenizer.pattern",
    Object(&[("String", "pre_tokenizer.pattern.String", Str(" "))]),
  ),
  (
    "behavior",
    "pre_tokenizer.behavior",
    Str("MergedWithPrevious"),
  ),
  ("invert", "pre_tokenizer.invert", Bool(false)),
]);

/// One special token of the single-sequence template, at `$index`.
macro_rules! piece {
  ($index:literal, $kind:literal, $id:literal) => {
    (
      concat!("post_processor.single[", $index, "]"),
      Object(&[(
        $kind,
        concat!("post_processor.single[", $index, "].", $kind),
        Object(&[
          (
            "id",
            concat!("post_processor.single[", $index, "].", $kind, ".id"),
            Str($id),
          ),
          (
            "type_id",
            concat!("post_processor.single[", $index, "].", $kind, ".type_id"),
            Uint(0),
          ),
        ]),
      )]),
    )
  };
}

/// The post-processor the runtime implements: `<bos>`, the text, `<eos>`. Its
/// other keys (the pair template, the special-token table) are read past.
const POST_PROCESSOR: Expect = Open(&[
  ("type", "post_processor.type", Str("TemplateProcessing")),
  (
    "single",
    "post_processor.single",
    Elements(&[
      piece!("0", "SpecialToken", "<bos>"),
      piece!("1", "Sequence", "A"),
      piece!("2", "SpecialToken", "<eos>"),
    ]),
  ),
]);

fn normalizer<'de, D: Deserializer<'de>>(
  d: D,
) -> core::result::Result<Option<Difference>, D::Error> {
  Walk {
    expect: &NORMALIZER,
    path: "normalizer",
  }
  .deserialize(d)
}

fn pre_tokenizer<'de, D: Deserializer<'de>>(
  d: D,
) -> core::result::Result<Option<Difference>, D::Error> {
  Walk {
    expect: &PRE_TOKENIZER,
    path: "pre_tokenizer",
  }
  .deserialize(d)
}

fn post_processor<'de, D: Deserializer<'de>>(
  d: D,
) -> core::result::Result<Option<Difference>, D::Error> {
  Walk {
    expect: &POST_PROCESSOR,
    path: "post_processor",
  }
  .deserialize(d)
}

fn truncation<'de, D: Deserializer<'de>>(
  d: D,
) -> core::result::Result<Option<Difference>, D::Error> {
  Walk {
    expect: &Null,
    path: "truncation",
  }
  .deserialize(d)
}

fn padding<'de, D: Deserializer<'de>>(d: D) -> core::result::Result<Option<Difference>, D::Error> {
  Walk {
    expect: &Null,
    path: "padding",
  }
  .deserialize(d)
}

std::thread_local! {
  /// A parse list's refused reservation, carried out of serde's error on this
  /// thread so it is answered as `Error::Allocation` by name.
  static REFUSED: Cell<Option<(AllocationSite, usize)>> = const { Cell::new(None) };
}

/// Appends `item`, growing `list` through the door at
/// [`AllocationSite::SourceParse`]: doubling, each growth one reservation.
fn push<T, E: de::Error>(list: &mut Vec<T>, item: T) -> core::result::Result<(), E> {
  if list.len() == list.capacity()
    && let Err(error) = alloc::reserve(list, list.len().max(64), AllocationSite::SourceParse)
  {
    if let Error::Allocation(refused) = &error {
      REFUSED.with(|cell| cell.set(Some((refused.site(), refused.bytes()))));
    }
    return Err(E::custom(error));
  }
  list.push(item);
  Ok(())
}

/// A JSON array read into a list grown through the door.
struct List<T>(PhantomData<T>);

impl<'de, T: Deserialize<'de>> Visitor<'de> for List<T> {
  type Value = Vec<T>;

  fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str("a list")
  }

  fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> core::result::Result<Vec<T>, A::Error> {
    let mut list = Vec::new();
    while let Some(item) = seq.next_element()? {
      push(&mut list, item)?;
    }
    Ok(list)
  }
}

fn list<'de, D, T>(deserializer: D) -> core::result::Result<Vec<T>, D::Error>
where
  D: Deserializer<'de>,
  T: Deserialize<'de>,
{
  deserializer.deserialize_seq(List(PhantomData))
}

/// The vocabulary object `{token: id}`, kept as a list in document order.
fn vocab_in_order<'de, D>(deserializer: D) -> core::result::Result<Vec<(Text<'de>, u32)>, D::Error>
where
  D: Deserializer<'de>,
{
  struct Entries;

  impl<'de> Visitor<'de> for Entries {
    type Value = Vec<(Text<'de>, u32)>;

    fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      f.write_str("a map from token to id")
    }

    fn visit_map<A: de::MapAccess<'de>>(
      self,
      mut map: A,
    ) -> core::result::Result<Self::Value, A::Error> {
      let mut entries = Vec::new();
      while let Some(entry) = map.next_entry::<Text<'de>, u32>()? {
        push(&mut entries, entry)?;
      }
      Ok(entries)
    }
  }

  deserializer.deserialize_map(Entries)
}

fn refuse(field: SourceField, detail: impl Into<String>) -> Error {
  Error::TokenizerSource(TokenizerSourceDefect::new(field, detail.into()))
}

/// The checks of the one configuration the runtime implements.
fn check_configuration(doc: &Document<'_>) -> Result<()> {
  for (field, difference) in [
    (SourceField::Document, &doc.truncation),
    (SourceField::Document, &doc.padding),
    (SourceField::Normalizer, &doc.normalizer),
    (SourceField::PreTokenizer, &doc.pre_tokenizer),
    (SourceField::PostProcessor, &doc.post_processor),
  ] {
    if let Some(difference) = difference {
      return Err(refuse(field, difference.to_string()));
    }
  }
  let model = &doc.model;
  fn text<'t>(text: &'t Option<Text<'_>>) -> Option<&'t str> {
    text.as_ref().map(|t| t.0.as_ref())
  }
  let flags_hold = model.kind.0 == "BPE"
    && model.dropout.is_none()
    && text(&model.unk_token) == Some("<unk>")
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
          "`{}` is not a plain special token (special, not normalized, no strip, not single-word)",
          excerpt(literal)
        ),
      ));
    }
    if !literal.starts_with('<') {
      return Err(refuse(
        SourceField::AddedTokens,
        format!(
          "`{}` does not begin with `<`, which the runtime's special scan relies on",
          excerpt(literal)
        ),
      ));
    }
  }
  Ok(())
}

/// The format-1 image of `tokenizer.json`'s bytes.
///
/// The parse is typed: its lists grow through the door at
/// [`AllocationSite::SourceParse`], and serde_json holds no more than the
/// document's escaped strings. Every table and list after it, and the image,
/// is reserved through the door at [`AllocationSite::ImageBuild`].
///
/// # Errors
/// [`Error::TokenizerSource`] naming the part of the document outside the
/// configuration the kit implements, or a vocabulary or merge it cannot hold;
/// [`Error::Allocation`] at [`AllocationSite::SourceParse`] or
/// [`AllocationSite::ImageBuild`] if a list, a table or the image cannot be
/// reserved.
pub(crate) fn compile_image(json: &[u8]) -> Result<Vec<u8>> {
  REFUSED.with(|cell| cell.set(None));
  let doc: Document<'_> =
    serde_json::from_slice(json).map_err(|e| match REFUSED.with(Cell::take) {
      Some((site, bytes)) => Error::Allocation(Allocation::new(site, bytes)),
      None => refuse(SourceField::Document, excerpt(e)),
    })?;
  check_configuration(&doc)?;

  // Dense ids, each token once.
  let vocab_len = doc.model.vocab.len();
  let mut slots: Vec<Option<&str>> = alloc::with_capacity(vocab_len, AllocationSite::ImageBuild)?;
  slots.resize(vocab_len, None);
  let mut ids: HashMap<&str, u32> = HashMap::new();
  alloc::reserve_map(&mut ids, vocab_len, AllocationSite::ImageBuild)?;
  for (text, id) in &doc.model.vocab {
    let slot = slots.get_mut(*id as usize).ok_or_else(|| {
      refuse(
        SourceField::Vocabulary,
        format!("id {id} is outside the {vocab_len} ids"),
      )
    })?;
    if slot.is_some() || ids.insert(&text.0, *id).is_some() {
      return Err(refuse(
        SourceField::Vocabulary,
        format!("`{}` or id {id} appears twice", excerpt(&text.0)),
      ));
    }
    *slot = Some(&text.0);
  }
  let mut tokens: Vec<&str> = alloc::with_capacity(vocab_len, AllocationSite::ImageBuild)?;
  for (id, token) in slots.into_iter().enumerate() {
    tokens
      .push(token.ok_or_else(|| refuse(SourceField::Vocabulary, format!("id {id} has no token")))?);
  }
  for token in &doc.added_tokens {
    if ids.get(token.content.0.as_ref()) != Some(&token.id) {
      return Err(refuse(
        SourceField::AddedTokens,
        format!(
          "`{}` is not the vocabulary's id {}",
          excerpt(&token.content.0),
          token.id
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
  // Each list is reserved once, for the most the merges can put in it.
  let merges = &doc.model.merges;
  let mut entries: Vec<(u32, u32, u32, u32)> =
    alloc::with_capacity(merges.len(), AllocationSite::ImageBuild)?;
  let bridging =
    |left: &str, right: &str| !left.ends_with(SPACE_MARK) && right.starts_with(SPACE_MARK);
  let mut bridges: Vec<(u32, u32)> = alloc::with_capacity(
    merges
      .iter()
      .filter(|(left, right)| bridging(&left.0, &right.0))
      .count(),
    AllocationSite::ImageBuild,
  )?;
  let mut joined = String::new();
  alloc::reserve_str(
    &mut joined,
    merges
      .iter()
      .map(|(left, right)| left.0.len() + right.0.len())
      .max()
      .unwrap_or(0),
    AllocationSite::ImageBuild,
  )?;
  for (rank, (left, right)) in merges.iter().enumerate() {
    let (left, right) = (left.0.as_ref(), right.0.as_ref());
    let id = |text: &str| {
      ids.get(text).copied().ok_or_else(|| {
        refuse(
          SourceField::Merges,
          format!("merge {rank}: `{}` is not a token", excerpt(text)),
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
          "merge {rank}: `{}` ends in `▁` after another scalar, which crosses a word boundary the piece split relies on",
          excerpt(left)
        ),
      ));
    }
    if bridging(left, right) {
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
        excerpt(tokens[pair[0].0 as usize]),
        excerpt(tokens[pair[0].1 as usize])
      ),
    ));
  }

  // Single-scalar tokens into an open-addressing index.
  let single = |text: &str| {
    let mut chars = text.chars();
    matches!((chars.next(), chars.next()), (Some(_), None))
  };
  let mut scalars: Vec<(u32, u32)> = alloc::with_capacity(
    tokens.iter().filter(|text| single(text)).count(),
    AllocationSite::ImageBuild,
  )?;
  scalars.extend(
    tokens
      .iter()
      .enumerate()
      .filter(|(_, text)| single(text))
      .filter_map(|(id, text)| Some((text.chars().next()? as u32, id as u32))),
  );
  let slots = (scalars.len() + scalars.len() / 2 + 1)
    .next_power_of_two()
    .max(16);
  let bits = slots.trailing_zeros();
  let mut index: Vec<(u32, u32)> = alloc::with_capacity(slots, AllocationSite::ImageBuild)?;
  index.resize(slots, (u32::MAX, u32::MAX));
  for &(scalar, id) in &scalars {
    let mut slot = scalar_slot(scalar, bits);
    while index[slot].0 != u32::MAX {
      slot = (slot + 1) & (slots - 1);
    }
    index[slot] = (scalar, id);
  }

  let mut added: Vec<(usize, u32)> = alloc::collect(
    doc.added_tokens.iter().map(|t| (t.content.0.len(), t.id)),
    AllocationSite::ImageBuild,
  )?;
  added.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

  // The sections, in table order, each reserved exactly before it is filled.
  let section = |len: usize| alloc::with_capacity::<u8>(len, AllocationSite::ImageBuild);
  let mut sections: [Vec<u8>; 8] = Default::default();
  let mut offsets = section((vocab_len + 1) * 4)?;
  let mut arena = section(tokens.iter().map(|text| text.len()).sum())?;
  for text in &tokens {
    offsets.extend_from_slice(&(arena.len() as u32).to_le_bytes());
    arena.extend_from_slice(text.as_bytes());
  }
  offsets.extend_from_slice(&(arena.len() as u32).to_le_bytes());
  sections[Section::Arena as usize] = arena;
  sections[Section::TokenOffsets as usize] = offsets;
  let mut scalar_index = section(index.len() * 8)?;
  scalar_index.extend(
    index
      .iter()
      .flat_map(|&(s, id)| [s.to_le_bytes(), id.to_le_bytes()])
      .flatten(),
  );
  sections[Section::ScalarIndex as usize] = scalar_index;
  let mut merge_offsets = section((vocab_len + 1) * 4)?;
  let mut cursor = 0usize;
  for left in 0..=vocab_len {
    while cursor < entries.len() && (entries[cursor].0 as usize) < left {
      cursor += 1;
    }
    merge_offsets.extend_from_slice(&(cursor as u32).to_le_bytes());
  }
  sections[Section::MergeOffsets as usize] = merge_offsets;
  let mut merge_entries = section(entries.len() * 12)?;
  merge_entries.extend(
    entries
      .iter()
      .flat_map(|&(_, r, rank, m)| [r.to_le_bytes(), rank.to_le_bytes(), m.to_le_bytes()])
      .flatten(),
  );
  sections[Section::MergeEntries as usize] = merge_entries;
  let mut byte_fallback = section(byte_ids.len() * 4)?;
  byte_fallback.extend(byte_ids.iter().flat_map(|id| id.to_le_bytes()));
  sections[Section::ByteFallback as usize] = byte_fallback;
  let mut added_section = section(added.len() * 4)?;
  added_section.extend(added.iter().flat_map(|(_, id)| id.to_le_bytes()));
  sections[Section::AddedTokens as usize] = added_section;
  let mut bridge_section = section(bridges.len() * 8)?;
  bridge_section.extend(
    bridges
      .iter()
      .flat_map(|&(l, r)| [l.to_le_bytes(), r.to_le_bytes()])
      .flatten(),
  );
  sections[Section::Bridges as usize] = bridge_section;

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
