//! Long texts: the windows a text is embedded in when it is longer than one,
//! the exact measure the shared chunker cuts them with, and the fold that
//! makes one vector of them.
//!
//! The prompt is in every window. A window's ids are the joint encoding of
//! `prompt ⧺ chunk`, `<bos>` and `<eos>` included, and the chunker measures a
//! candidate range exactly that way, so a window counts its TOTAL ids.
//!
//! # The measure
//!
//! One encoding of the whole text records its piece table: the text's cuts,
//! and its own tokens before each. A range's measure is then
//!
//! - the prompt's tokens before its last piece, counted once per text;
//! - the prompt's last piece joined with the range's first edge, re-encoded,
//!   so the prompt meets the content as it does in the window: the trailing
//!   `▁` of the eight prompts merges into the first word, a custom prefix's
//!   `:` meets content beginning `</`, and a literal spanning the seam is its
//!   special id;
//! - the table's tokens between the range's first and last cut;
//! - the range's last edge, re-encoded.
//!
//! A cut is a word boundary — a space or `▁` after another scalar — where no
//! bridging merge's left text ends and right text begins. The encoding of any
//! range around a cut splits there, and the tokens on either side do not
//! depend on the other, provided two facts of the image hold:
//!
//! - no added literal holds a space or `▁`, so no special token covers a cut;
//! - no bridging merge's text holds a word boundary inside it, so no bridging
//!   decision reads across a cut.
//!
//! Both are read from the image for each text. An image without them has no
//! cuts, and every range is re-encoded whole: slower, and just as exact.
//!
//! A cut joins a range's interior only when the bridging decision at it reads
//! nothing outside the range, so the prompt cannot move it. The prompt's own
//! last piece starts at its last word boundary that no bridging merge's left
//! text ends at, so no continuation can bridge it either.

use core::{cell::RefCell, ops::Range};

use windit::{
  plan::Span,
  split::{Chunk, MeasureText},
  windowed::Windowed,
};

use super::{
  config::MAX_TOKENS,
  embedding::Embedding,
  error::{Allocation, AllocationSite, Error, PromptFillsWindow, Result},
  prompt::Prompt,
  text::Framing,
  tokenizer::{
    SplitSpecialTokens,
    bpe::{Encoded, SPACE_MARK, TokenId, TokenizerScratch},
    image::TokenizerImage,
  },
};
use crate::embeddings::{
  aggregate::AggregatePolicyKind,
  long_text::{self, LongTextOptions, RangeMeasure},
};

/// A scalar as the piece split reads it: a space is `▁`.
#[inline]
fn marked(ch: char) -> char {
  if ch == ' ' { SPACE_MARK } else { ch }
}

/// Grows `buffer` by one, or names the site.
fn reserve_one<T>(buffer: &mut Vec<T>, site: AllocationSite) -> Result<()> {
  buffer.try_reserve(1).map_err(|_| {
    Error::Allocation(Allocation::new(
      site,
      (buffer.len() + 1).saturating_mul(core::mem::size_of::<T>()),
    ))
  })
}

/// What an image lets the measure cut at: its bridging merges' texts, and
/// whether a word boundary can be a cut at all.
#[derive(Debug)]
struct Seams {
  /// Each bridging merge's left and right texts, as the piece split reads
  /// them.
  bridges: Vec<(Vec<char>, Vec<char>)>,
  /// The longest bridging left text, in scalars: how far left of a word
  /// boundary its bridging decision reads.
  reach: usize,
  /// No added literal holds a space or `▁`, and no bridging text holds a
  /// word boundary inside it.
  cuts: bool,
}

impl Seams {
  fn of(image: &TokenizerImage) -> Self {
    let text = |id| {
      image
        .token(id)
        .and_then(|bytes| core::str::from_utf8(bytes).ok())
    };
    let mut cuts = image
      .added()
      .all(|id| text(id).is_some_and(|literal| !literal.chars().any(|c| marked(c) == SPACE_MARK)));
    let mut bridges = Vec::new();
    for (left, right) in image.bridges() {
      match (text(left), text(right)) {
        (Some(left), Some(right)) => {
          let left: Vec<char> = left.chars().map(marked).collect();
          let right: Vec<char> = right.chars().map(marked).collect();
          cuts &= !inner_boundary(&left) && !inner_boundary(&right);
          bridges.push((left, right));
        }
        _ => cuts = false,
      }
    }
    let reach = bridges
      .iter()
      .map(|(left, _)| left.len())
      .max()
      .unwrap_or(0);
    Self {
      bridges,
      reach,
      cuts,
    }
  }

  /// Whether some bridging merge's left text ends at byte `at` of `text`.
  fn left_meets(&self, text: &str, at: usize) -> bool {
    self
      .bridges
      .iter()
      .any(|(left, _)| ends_with(&text[..at], left))
  }

  /// Whether some bridging merge's texts meet at byte `at` of `text`: its left
  /// text ends there and its right text begins there.
  fn bridged(&self, text: &str, at: usize) -> bool {
    self.bridges.iter().any(|(left, right)| {
      ends_with(&text[..at], left)
        && text[at..]
          .chars()
          .map(marked)
          .take(right.len())
          .eq(right.iter().copied())
    })
  }
}

/// Whether `text`, as the piece split reads it, ends with `token`.
fn ends_with(text: &str, token: &[char]) -> bool {
  text
    .chars()
    .rev()
    .map(marked)
    .take(token.len())
    .eq(token.iter().rev().copied())
}

/// Whether `token` holds a word boundary inside it: a `▁` after another
/// scalar.
fn inner_boundary(token: &[char]) -> bool {
  token
    .windows(2)
    .any(|pair| pair[1] == SPACE_MARK && pair[0] != SPACE_MARK)
}

/// Whether byte `at` of `text` is a word boundary: a space or `▁` after
/// another scalar.
fn word_boundary(previous: char, ch: char) -> bool {
  marked(ch) == SPACE_MARK && marked(previous) != SPACE_MARK
}

/// The byte the prompt's last piece starts at: its last word boundary that no
/// bridging merge's left text ends at, so that no content can bridge it; or 0,
/// the whole prompt.
fn join_point(prompt: &str, seams: &Seams) -> usize {
  if !seams.cuts {
    return 0;
  }
  let mut join = 0;
  let mut previous = None;
  for (at, ch) in prompt.char_indices() {
    if let Some(previous) = previous
      && word_boundary(previous, ch)
      && !seams.left_meets(prompt, at)
    {
      join = at;
    }
    previous = Some(ch);
  }
  join
}

/// A prompt as the measure joins it to a range: its text, the byte its last
/// piece starts at, and its tokens before that byte.
#[derive(Debug)]
struct PromptJoin {
  text: String,
  join: usize,
  head: usize,
}

impl PromptJoin {
  fn new(
    image: &TokenizerImage,
    prompt: &Prompt<'_>,
    split: SplitSpecialTokens,
    seams: &Seams,
    scratch: &mut TokenizerScratch,
  ) -> Result<Self> {
    let text = prompt.pieces().concat();
    let join = join_point(&text, seams);
    let mut head = 0;
    if join > 0 {
      let (mut ids, mut spans) = (Vec::new(), Vec::new());
      image.encode_joint(
        &Prompt::Bare,
        &text,
        split,
        scratch,
        Encoded {
          ids: &mut ids,
          spans: Some(&mut spans),
        },
      )?;
      head = spans[1..spans.len() - 1]
        .iter()
        .filter(|span| span.start < join)
        .count();
    }
    Ok(Self { text, join, head })
  }

  /// The prompt from its last piece on.
  #[inline]
  fn tail(&self) -> &str {
    &self.text[self.join..]
  }
}

/// A text's cuts, ascending, and its own tokens before each.
#[derive(Debug, Default)]
struct PieceTable {
  cuts: Vec<usize>,
  before: Vec<usize>,
}

impl PieceTable {
  /// One encoding of `text`, and a walk over its word boundaries.
  fn build(
    image: &TokenizerImage,
    text: &str,
    split: SplitSpecialTokens,
    seams: &Seams,
    scratch: &mut TokenizerScratch,
  ) -> Result<Self> {
    let mut table = Self::default();
    if !seams.cuts {
      return Ok(table);
    }
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    image.encode_joint(
      &Prompt::Bare,
      text,
      split,
      scratch,
      Encoded {
        ids: &mut ids,
        spans: Some(&mut spans),
      },
    )?;
    let content = &spans[1..spans.len() - 1];
    let mut token = 0;
    let mut previous = None;
    for (at, ch) in text.char_indices() {
      if let Some(previous) = previous
        && word_boundary(previous, ch)
        && !seams.bridged(text, at)
      {
        while token < content.len() && content[token].start < at {
          token += 1;
        }
        reserve_one(&mut table.cuts, AllocationSite::PieceTable)?;
        reserve_one(&mut table.before, AllocationSite::PieceTable)?;
        table.cuts.push(at);
        table.before.push(token);
      }
      previous = Some(ch);
    }
    Ok(table)
  }
}

/// The buffers a measure encodes its edges into.
#[derive(Debug, Default)]
struct Work {
  scratch: TokenizerScratch,
  ids: Vec<TokenId>,
}

/// The EmbeddingGemma 2 measure of one text under one prompt: the ids of the
/// joint encoding of `prompt ⧺ text[a..b]`, for any range, without encoding
/// the range whole.
///
/// windit's chunker measures through the [`MeasureText`] face, whose
/// `measure_within` stops as soon as the count is known to pass its limit;
/// the repair measures through [`RangeMeasure`].
#[derive(Debug)]
pub(crate) struct JointMeasure<'a> {
  image: &'a TokenizerImage,
  text: &'a str,
  split: SplitSpecialTokens,
  seams: Seams,
  prompt: PromptJoin,
  table: PieceTable,
  /// The longest token in bytes: no token covers more of a text, so a range
  /// of `n` bytes takes at least `n / longest` tokens.
  longest: usize,
  work: RefCell<Work>,
}

impl<'a> JointMeasure<'a> {
  /// The measure of `text` under `prompt` and `split`: one encoding of the
  /// text, and one of the prompt.
  ///
  /// # Errors
  /// [`Error::Allocation`] if a buffer cannot grow.
  pub(crate) fn new(
    image: &'a TokenizerImage,
    text: &'a str,
    prompt: Prompt<'_>,
    split: SplitSpecialTokens,
  ) -> Result<Self> {
    let seams = Seams::of(image);
    let mut work = Work::default();
    let prompt = PromptJoin::new(image, &prompt, split, &seams, &mut work.scratch)?;
    let table = PieceTable::build(image, text, split, &seams, &mut work.scratch)?;
    Ok(Self {
      image,
      text,
      split,
      seams,
      prompt,
      table,
      longest: image.longest_token().max(1),
      work: RefCell::new(work),
    })
  }

  /// The ids of `prompt ⧺ text[start..end]`, or `None` as soon as they are
  /// known to exceed `limit`.
  fn within(&self, start: usize, end: usize, limit: usize) -> Result<Option<usize>> {
    let head = self.prompt.head;
    let tail = self.prompt.tail();
    let floor = |bytes: usize| bytes.div_ceil(self.longest);
    let last = self.table.cuts.partition_point(|&cut| cut <= end);
    let total = match self.first_cut(start) {
      Some(first) if first < last => {
        let (c1, c2) = (self.table.cuts[first], self.table.cuts[last - 1]);
        let middle = self.table.before[last - 1] - self.table.before[first];
        let right_floor = floor(end - c2);
        let left_floor = 2 + floor(tail.len() + (c1 - start));
        if head
          .saturating_add(left_floor)
          .saturating_add(middle)
          .saturating_add(right_floor)
          > limit
        {
          return Ok(None);
        }
        let left = self.count(Prompt::Custom(tail), &self.text[start..c1])?;
        if head
          .saturating_add(left)
          .saturating_add(middle)
          .saturating_add(right_floor)
          > limit
        {
          return Ok(None);
        }
        let right = self.count(Prompt::Bare, &self.text[c2..end])? - 2;
        head + left + middle + right
      }
      _ => {
        if head.saturating_add(2 + floor(tail.len() + (end - start))) > limit {
          return Ok(None);
        }
        head + self.count(Prompt::Custom(tail), &self.text[start..end])?
      }
    };
    Ok((total <= limit).then_some(total))
  }

  /// The first cut after `start` whose bridging decision reads nothing before
  /// `start`.
  fn first_cut(&self, start: usize) -> Option<usize> {
    let cuts = &self.table.cuts;
    let reach = self.seams.reach;
    (cuts.partition_point(|&cut| cut <= start)..cuts.len()).find(|&index| {
      reach == 0
        || self.text[..cuts[index]]
          .char_indices()
          .rev()
          .nth(reach - 1)
          .is_some_and(|(at, _)| at >= start)
    })
  }

  /// The ids of `prompt ⧺ text`, `<bos>` and `<eos>` included.
  fn count(&self, prompt: Prompt<'_>, text: &str) -> Result<usize> {
    let mut work = self.work.borrow_mut();
    let Work { scratch, ids } = &mut *work;
    ids.clear();
    self.image.encode_joint(
      &prompt,
      text,
      self.split,
      scratch,
      Encoded { ids, spans: None },
    )?;
    Ok(ids.len())
  }

  /// The byte range `s` covers in the measured text, if it is a slice of it.
  ///
  /// Recovered by pointer offset: live allocations are disjoint, so an offset
  /// inside the text can only mean `s` borrows from it. Arithmetic on
  /// addresses, never a dereference.
  fn range_of(&self, s: &str) -> Option<(usize, usize)> {
    let start = (s.as_ptr() as usize).checked_sub(self.text.as_ptr() as usize)?;
    let end = start.checked_add(s.len())?;
    self.text.get(start..end).map(|_| (start, end))
  }
}

impl MeasureText for JointMeasure<'_> {
  fn measure(&self, s: &str) -> usize {
    self.measure_within(s, usize::MAX).unwrap_or(usize::MAX)
  }

  /// An encoding that fails does not fit: windit descends to a smaller
  /// range, and a failure that persists surfaces from the window's own
  /// encoding.
  fn measure_within(&self, s: &str, limit: usize) -> Option<usize> {
    let measured = match self.range_of(s) {
      Some((start, end)) => self.within(start, end, limit),
      None => self
        .count(Prompt::Custom(&self.prompt.text), s)
        .map(|ids| (ids <= limit).then_some(ids)),
    };
    measured.unwrap_or(None)
  }
}

impl RangeMeasure for JointMeasure<'_> {
  type Error = Error;

  fn measure_range(&self, start: usize, end: usize) -> Result<usize> {
    Ok(self.within(start, end, usize::MAX)?.unwrap_or(usize::MAX))
  }
}

/// One window of a long text before it is embedded: its byte range in the
/// caller's text, and its ids, the joint encoding of `prompt ⧺ chunk`.
#[derive(Debug)]
pub(crate) struct Planned {
  pub(crate) range: Range<usize>,
  pub(crate) ids: Vec<TokenId>,
}

/// Plans the windows of `text` under `prompt`, `split` and `geometry`,
/// refusing, in order: an input over `max_input_bytes`; a window over the
/// model's 512 ids; a geometry windit refuses; a window the prompt's
/// `framing` fills; an empty text.
///
/// Each window's ids are the joint encoding of `prompt ⧺ chunk`, the prompt
/// in every window.
///
/// # Errors
/// [`Error::InputTooLarge`], [`Error::WindowOverBudget`],
/// [`Error::Windowing`], [`Error::PromptFillsWindow`] (also when one
/// character beside the prompt does not fit the model's 512 ids),
/// [`Error::EmptyText`], [`Error::ContentlessInputOverBudget`] and
/// [`Error::Allocation`].
pub(crate) fn plan(
  image: &TokenizerImage,
  text: &str,
  prompt: Prompt<'_>,
  split: SplitSpecialTokens,
  geometry: &LongTextOptions,
  framing: Framing,
  scratch: &mut TokenizerScratch,
) -> Result<Vec<Planned>> {
  long_text::validate_long_input(text, geometry, MAX_TOKENS)?;
  let window = geometry.window_options();
  window.validate()?;
  if window.window() <= framing.ids() {
    return Err(Error::PromptFillsWindow(PromptFillsWindow::new(
      window.window(),
      framing.ids(),
    )));
  }
  if text.is_empty() {
    return Err(Error::EmptyText);
  }
  let measure = JointMeasure::new(image, text, prompt, split)?;
  let chunks: Vec<Chunk> = long_text::chunk_long(&measure, text, &window, MAX_TOKENS)?;
  let mut windows = Vec::new();
  windows.try_reserve_exact(chunks.len()).map_err(|_| {
    Error::Allocation(Allocation::new(
      AllocationSite::WindowList,
      chunks.len().saturating_mul(core::mem::size_of::<Planned>()),
    ))
  })?;
  for chunk in chunks {
    let chunk_text = chunk
      .as_str(text)
      .expect("the chunker cuts on char boundaries");
    let mut ids = Vec::new();
    image.encode_joint(
      &prompt,
      chunk_text,
      split,
      scratch,
      Encoded {
        ids: &mut ids,
        spans: None,
      },
    )?;
    // windit keeps one character that alone outgrows the window. Beside a
    // prompt within a few ids of the model's ceiling, it can outgrow that too.
    if ids.len() > MAX_TOKENS {
      return Err(Error::PromptFillsWindow(PromptFillsWindow::new(
        MAX_TOKENS,
        framing.ids(),
      )));
    }
    windows.push(Planned {
      range: chunk.start()..chunk.end(),
      ids,
    });
  }
  Ok(windows)
}

/// One window of a long text and the embedding of exactly that window: what
/// [`TextEmbedder::embed_windows`](super::TextEmbedder::embed_windows)
/// returns.
///
/// - [`Self::ordinal`]: its position in planning order, `0..n`, which tells
///   apart two windows of the same bytes.
/// - [`Self::byte_range`]: its half-open range in the caller's text,
///   `char`-aligned, exactly the chunk the planner cut.
/// - [`Self::token_span`]: its place in the stream of the windows' ids. Its
///   length is the window's TOTAL ids, `<bos>`, the prompt and `<eos>`
///   included, and its window is the model's 512, so its coverage is the
///   weight [`TextEmbedder::embed_long`](super::TextEmbedder::embed_long)
///   folds with.
/// - [`Self::embedding`]: the unit-norm vector of the prompt and that window's
///   bytes.
///
/// No `PartialEq` and no serde, for [`Embedding`]'s reasons: a model's floats
/// are not bit-stable across runs and hosts, and a stored vector's format is
/// the consumer's choice.
#[derive(Debug, Clone)]
pub struct WindowEmbedding {
  ordinal: usize,
  byte_range: Range<usize>,
  token_span: Span,
  embedding: Embedding,
}

impl WindowEmbedding {
  /// A window: its ordinal, its byte range and its ids' span.
  pub(crate) const fn new(
    ordinal: usize,
    byte_range: Range<usize>,
    token_span: Span,
    embedding: Embedding,
  ) -> Self {
    Self {
      ordinal,
      byte_range,
      token_span,
      embedding,
    }
  }

  /// The window's position in planning order, `0..n` over one call's
  /// windows.
  #[inline]
  pub const fn ordinal(&self) -> usize {
    self.ordinal
  }

  /// The window's first byte in the caller's text, a `char` boundary.
  #[inline]
  pub const fn byte_start(&self) -> usize {
    self.byte_range.start
  }

  /// One past the window's last byte in the caller's text, a `char`
  /// boundary.
  #[inline]
  pub const fn byte_end(&self) -> usize {
    self.byte_range.end
  }

  /// The window's half-open byte range in the caller's text, to slice it with
  /// (`&text[w.byte_range()]`).
  #[inline]
  pub fn byte_range(&self) -> Range<usize> {
    self.byte_range.clone()
  }

  /// The window's place in the stream of the windows' ids: the ids before
  /// it, its own, and the model's window of 512. Ids, not bytes: use
  /// [`Self::byte_range`] to find the window in the text.
  #[inline]
  pub const fn token_span(&self) -> Span {
    self.token_span
  }

  /// The window's ids, the prompt's included: `token_span().len()`, at most
  /// 512.
  #[inline]
  pub const fn token_count(&self) -> usize {
    self.token_span.len()
  }

  /// The unit-norm embedding of the prompt and the window's bytes.
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

/// Folds `windows` into one vector under `fold`. One window is returned as it
/// is: under the default fold that is bit for bit what folding it gives.
///
/// # Errors
/// [`Error::Windowing`] if windit refuses the fold: no window, or windows that
/// cancel exactly.
pub(crate) fn fold_windows(
  mut windows: Vec<WindowEmbedding>,
  fold: &AggregatePolicyKind,
) -> Result<Embedding> {
  if windows.len() == 1 {
    return Ok(windows.pop().expect("one window").into_embedding());
  }
  let windowed: Vec<Windowed<Embedding>> = windows
    .into_iter()
    .map(|window| Windowed::new(window.embedding, window.token_span))
    .collect();
  Ok(windit::aggregate::aggregate(
    fold.into_policy().as_ref(),
    &windowed,
  )?)
}

#[cfg(test)]
mod tests;
