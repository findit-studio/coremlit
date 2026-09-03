//! The structural check on a caller-supplied tokenizer's post-processor that
//! the tokenizers crate's own deserializer skips.
//!
//! # The boundary
//!
//! `tokenizers::processors::template::TemplateProcessing` is built two ways.
//! `TemplateProcessingBuilder::build` runs a `validate` step; deserializing a
//! `tokenizer.json` does **not** — the file goes through
//! `From<TemplateProcessingDeserializer>`, which only recomputes the added-token
//! counts. So a template that the builder would refuse still *parses*, and the
//! refusal is deferred to the first `encode`, where it is a **panic** inside the
//! dependency rather than an error:
//!
//! * a `SpecialToken` piece naming an id absent from `special_tokens` indexes
//!   `self.special_tokens.0[id]` (a `HashMap` index) — "no entry found for key";
//! * a `Sequence` piece with id `B` in the SINGLE template indexes
//!   `encodings[1]` when a single sequence was encoded — "index out of bounds";
//! * a single template with no `$A` at all silently drops the text and encodes
//!   every input as its special tokens alone — the same degenerate answer the
//!   `>= window` special-token-overhead rule refuses, reported here instead.
//!
//! # Which post-processor runs at which count
//!
//! Those rules are about ONE template, and which template a `TemplateProcessing`
//! applies is not fixed: `process_encodings` selects it by how many encodings it
//! was handed — `2 => pair`, `1 => single`, and every other count is a `todo!()`,
//! i.e. another panic ("not yet implemented"). That count is not always one.
//! `apply_template` emits ONE ENCODING PER PIECE of the template it applied — a
//! `Sequence` piece clones the encoding it names, a `SpecialToken` piece builds
//! one from its ids — and the merge back into a single encoding happens at the
//! TOKENIZER level, after the whole post-processor has run. A post-processor
//! `Sequence` meanwhile threads each member's output into the next. So a
//! three-piece template inside a `Sequence` hands three encodings to whatever
//! follows it.
//!
//! The guard therefore does not judge each template independently: it SIMULATES
//! the encoding count along the chain, for the single-sequence encode the doors
//! here perform. The count starts at 1; a `Sequence` threads it through its
//! members in order, nested ones included; a `TemplateProcessing` hands on its
//! selected template's piece count; and every kind that ADDS TOKENS —
//! `TemplateProcessing`, `RobertaProcessing`, `BertProcessing` — must be reached
//! at EXACTLY ONE encoding or it is refused. `ByteLevel` adds nothing at any
//! count and only trims offsets, so it passes the count through untouched. Any
//! FINAL count is fine — that one the tokenizer merges.
//!
//! # Why exactly one, and why `$A` exactly once
//!
//! Both halves are about the door's TRUNCATION, which the tokenizer applies to
//! the RAW encoding — before the post-processor — at
//! `max_length - added_tokens(false)`. That subtrahend is the ONLY overhead
//! number in the system: `with_truncation` reads it, `post_process` reads it
//! again on every `encode`, and the doors' special-token-overhead guard reads
//! the same one. For a post-processor `Sequence` it is the SUM of the members'
//! `added_single`, whichever mode each member actually runs in.
//!
//! * **Exactly one encoding.** A `TemplateProcessing` reached at two applies its
//!   `pair` template and adds `added_pair`, which that reading does not report;
//!   `RobertaProcessing` and `BertProcessing` reached at `n` wrap EVERY encoding
//!   (`cls … sep` around the first, `sep … sep` around each of the rest) and add
//!   `2n` while reporting a flat 2. Either way the truncation is sized on an
//!   overhead the chain does not add. Measured on 0.23.1:
//!   `Sequence[Template("[CLS] $A [SEP]"), RobertaProcessing]` declares 4 and
//!   returns 12 ids at an 8-token window. Counts of 0, 3 and up have no template
//!   at all — that is the `todo!()`. Refusing every count but one costs nothing
//!   real: no door here encodes a pair, so a chain that manufactures a second
//!   encoding is not a shape any of them needs.
//! * **`$A` exactly once.** Each placement emits its own copy of the text, and
//!   `count_added` scores a `Sequence` piece as zero however often it appears,
//!   so `Template("$A $A")` advertises NO overhead and returns twice the
//!   truncated length — measured, a 3-token text comes back as 6 ids at a
//!   4-token window, which the door then refuses with a typed `TokenCount` for
//!   ordinary text although construction succeeded. Zero placements is the other
//!   end of the same rule: the text dropped entirely.
//!
//! Consequently the count a template hands on is its piece count, and a second
//! token-adding post-processor downstream is admitted only when the first was
//! the identity `$A` — one piece, one encoding. Nothing legitimate is refused by
//! that: no single-sequence tokenizer shape in common use chains two templating
//! processors. RoBERTa and BERT carry one `TemplateProcessing`, or their
//! dedicated `RobertaProcessing` / `BertProcessing`; GPT-2 and the other
//! byte-level BPEs a bare `ByteLevel`; the SentencePiece families (Gemma, T5)
//! one `TemplateProcessing`. Where a `Sequence` appears at all it pairs a
//! `ByteLevel` with a single template, which this admits.
//!
//! [`check_post_processor`] is the tokenizers builder's skipped `validate`,
//! applied to the template each `TemplateProcessing` in the chain would really
//! select, plus the placement and count rules that `validate` has no reason to
//! carry — the builder owns no truncation window. What it proves is bounded and
//! exact: the single-sequence encode reaches no `todo!()` and no undeclared-key
//! index inside the dependency's post-processing, the text is placed EXACTLY
//! ONCE, and `added_tokens(false)` is EXACTLY the overhead the chain adds. Those
//! three are what make the doors' `>= window` overhead guard sound as written —
//! a raw encoding truncated to `max_length - added` post-processes to at most
//! `max_length`. Everything else about a `tokenizer.json`'s internal consistency
//! remains the tokenizers crate's contract; no door here re-implements its
//! deserializer. A post-processor kind this module does not recognize is the one
//! place that reasoning stops: it carries no template to judge and no overhead
//! this module can know, so it passes and is credited with preserving the count,
//! as every kind that exists today does.
//!
//! A door runs it BEFORE reading `added_tokens(false)` off the post-processor.
//! Not because that reading is unsafe: `count_added` scores an undeclared
//! `SpecialToken` id as **zero**, so the special-token-overhead guard is simply
//! BLIND to the templates this module refuses — it cannot substitute for this
//! check, and this check catches them whichever order the two run in. The order
//! is a diagnostic choice. A count derived from a malformed template is not a
//! fact about the tokenizer, so a file that breaks both rules is reported by its
//! structural defect rather than by a number the caller would try to shrink.
//!
//! # Why serialization
//!
//! `TemplateProcessing` exposes no accessor for `special_tokens`, and its
//! `get_single` renders a `Debug` string. Its serde representation is the
//! `tokenizer.json` shape itself, so the structure is read back out through
//! `serde_json::to_value` — the same JSON the file supplied.

use tokenizers::Tokenizer;

/// Which structural rule of a post-processor was broken.
///
/// Payload of each door's `Error::PostProcessorTemplate` variant
/// ([`crate::embeddings::clap::error::Error::PostProcessorTemplate`] /
/// [`crate::embeddings::siglip::error::Error::PostProcessorTemplate`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostProcessorTemplate {
  /// A `SpecialToken` piece names an id the template's `special_tokens` map does
  /// not declare. The tokenizers crate indexes that map directly while applying
  /// the template, so the first `encode` panics with "no entry found for key" —
  /// and the added-token count it reports for the template is short by this
  /// piece, because `count_added` scores a missing key as zero.
  ///
  /// Carries the undeclared id.
  UndeclaredSpecialToken(String),
  /// The SINGLE template places the PAIR sequence (`$B`). Applying it to one
  /// encoded sequence indexes `encodings[1]`, so the first `encode` panics with
  /// "index out of bounds". Every door here encodes single sequences.
  PairSequenceInSingleTemplate,
  /// The SINGLE template never places the input sequence (`$A`), so the text is
  /// dropped and every input encodes to the template's special tokens alone —
  /// a silently wrong answer rather than a reported failure, exactly like a
  /// post-processor whose overhead fills the whole window.
  NoInputSequenceInSingleTemplate,
  /// The SINGLE template places the input sequence (`$A`) more than once, so the
  /// text is emitted once per placement and the post-processed encoding is that
  /// many times the length of the truncated one.
  ///
  /// Nothing bounds that. Truncation runs BEFORE the post-processor and is sized
  /// `max_length - added_tokens(false)`, which counts only `SpecialToken` pieces
  /// — a repeated `$A` reads as no overhead at all. So ordinary text longer than
  /// `max_length / n` leaves the window and the door's fixed-window construction
  /// refuses it with a typed `TokenCount`, although construction succeeded.
  ///
  /// Carries the number of placements.
  RepeatedInputSequence(usize),
  /// A post-processor that adds tokens would be reached at a number of encodings
  /// other than ONE, and only the single-sequence case is admitted here.
  ///
  /// At 0, or at 3 and above, a `TemplateProcessing` has no template at all —
  /// that arm of `process_encodings` is a `todo!()`, so the first `encode`
  /// panics with "not yet implemented". At 2 it has one, the PAIR template, and
  /// applies it happily; what breaks there is the arithmetic around it. The
  /// truncation, and the doors' overhead guard, are both sized on
  /// `added_tokens(false)` — `added_single` — while a template running in pair
  /// mode adds `added_pair`, and `RobertaProcessing` / `BertProcessing` reached
  /// at `n` encodings add `2n` against a reported 2. The window would then be
  /// sized on an overhead the chain does not add.
  ///
  /// Reachable whenever an earlier member of the same `Sequence` applied a
  /// template of more than one piece, since each piece emits its own encoding.
  ///
  /// Carries the count the post-processor would have received.
  UnsupportedEncodingCount(usize),
  /// The post-processor could not be serialized for inspection, so no rule
  /// above could be judged. Unreachable with the tokenizers crate's derived
  /// `Serialize` implementations (plain structs, `Vec`s and string-keyed maps
  /// have no fallible branch); it exists so this guard fails CLOSED rather than
  /// open if a future version gains one, since its whole input is caller-chosen.
  Unreadable,
}

impl core::fmt::Display for PostProcessorTemplate {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::UndeclaredSpecialToken(id) => write!(
        f,
        "its template uses the special token `{id}`, which its `special_tokens` map does not declare"
      ),
      Self::PairSequenceInSingleTemplate => {
        f.write_str("its single template places the pair sequence `$B`")
      }
      Self::NoInputSequenceInSingleTemplate => {
        f.write_str("its single template never places the input sequence `$A`")
      }
      Self::RepeatedInputSequence(placements) => write!(
        f,
        "its single template places the input sequence `$A` {placements} times, so it encodes that many copies of the text"
      ),
      Self::UnsupportedEncodingCount(count) => write!(
        f,
        "its post-processor chain hands {count} encodings to a post-processor that adds tokens, and only the single-sequence case (exactly one) is admitted here"
      ),
      Self::Unreadable => f.write_str("its post-processor could not be read for inspection"),
    }
  }
}

/// Applies the tokenizers builder's skipped `validate` to `tokenizer`'s
/// post-processor, plus the count and placement rules the door's truncation
/// needs — see the module docs for the exact boundary.
///
/// A tokenizer with no post-processor passes. `ByteLevel` adds no tokens and
/// maps encodings 1:1, so it passes at any count, and so does a kind added after
/// this was written. `RobertaProcessing` and `BertProcessing` map encodings 1:1
/// but WRAP each of them, so they pass only at a count of one. A `Sequence`
/// post-processor is walked in order, because it applies each of its members to
/// the previous one's output.
///
/// # Errors
/// [`PostProcessorTemplate`] naming the first rule the chain breaks, for the
/// SINGLE-sequence encode every door here performs.
pub(crate) fn check_post_processor(tokenizer: &Tokenizer) -> Result<(), PostProcessorTemplate> {
  let Some(post) = tokenizer.get_post_processor() else {
    return Ok(());
  };
  let value = serde_json::to_value(post).map_err(|_| PostProcessorTemplate::Unreadable)?;
  // A door encodes ONE sequence, so the post-processor is entered with exactly
  // one encoding; whatever it hands back is merged and never re-processed.
  check_value(&value, 1).map(|_| ())
}

/// One serialized post-processor, given the number of encodings reaching it;
/// answers with the number it hands on.
///
/// Dispatch is on the `type` tag: a `Sequence` threads the count through its
/// members in order (nested `Sequence`s included), a `TemplateProcessing` is
/// judged, `RobertaProcessing` and `BertProcessing` are admitted only at one,
/// and every other kind preserves the count.
fn check_value(value: &serde_json::Value, count: usize) -> Result<usize, PostProcessorTemplate> {
  match value.get("type").and_then(serde_json::Value::as_str) {
    Some("TemplateProcessing") => check_template(value, count),
    Some("Sequence") => value
      .get("processors")
      .and_then(serde_json::Value::as_array)
      .map_or(Ok(count), |processors| {
        processors
          .iter()
          .try_fold(count, |count, processor| check_value(processor, count))
      }),
    // These two map their input encodings 1:1 but WRAP each one — `cls … sep`
    // around the first, `sep … sep` around every other — while `added_tokens`
    // reports a flat 2. Their real overhead equals the reported one only at a
    // count of one, which is the only count the truncation could be sized for.
    Some("RobertaProcessing" | "BertProcessing") if count != 1 => {
      Err(PostProcessorTemplate::UnsupportedEncodingCount(count))
    }
    // `ByteLevel` carries no template and adds no tokens at any count — it only
    // trims offsets — so it is transparent to both rules. An unknown kind is a
    // tokenizers version this module predates, and refusing it would break a
    // legitimate tokenizer over a rule that may not even apply to it.
    _ => Ok(count),
  }
}

/// The rules, over the SINGLE template — the only one a `TemplateProcessing` in
/// an admitted chain ever applies.
///
/// The count is refused first: the dependency selects `pair` at two and panics
/// at anything but one or two, and neither is a mode whose overhead
/// `added_tokens(false)` reports, so no rule below is worth judging for a
/// template that would not run in single mode. `$B` stays refused because
/// applying it to one encoding indexes `encodings[1]`, and `$A` must appear
/// exactly once because every placement emits its own copy of the text.
///
/// The `single` template being absent or not an array leaves no pieces, so it
/// falls out of the no-input rule below rather than needing a shape check of its
/// own — which keeps the guard closed on a shape this module does not expect.
fn check_template(
  template: &serde_json::Value,
  count: usize,
) -> Result<usize, PostProcessorTemplate> {
  const EMPTY: &[serde_json::Value] = &[];
  if count != 1 {
    return Err(PostProcessorTemplate::UnsupportedEncodingCount(count));
  }
  let pieces = template
    .get("single")
    .and_then(serde_json::Value::as_array)
    .map_or(EMPTY, Vec::as_slice);
  let declared = template
    .get("special_tokens")
    .and_then(serde_json::Value::as_object);

  let mut placements = 0usize;
  for piece in pieces {
    if let Some(special) = piece.get("SpecialToken") {
      // An id that is absent, or not a string, is not a key of the map either;
      // both reach `special_tokens[id]` in the dependency, so both are refused.
      let id = special
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
      if !declared.is_some_and(|map| map.contains_key(id)) {
        return Err(PostProcessorTemplate::UndeclaredSpecialToken(
          id.to_string(),
        ));
      }
    } else if let Some(sequence) = piece.get("Sequence") {
      match sequence.get("id").and_then(serde_json::Value::as_str) {
        Some("A") => placements += 1,
        Some("B") => return Err(PostProcessorTemplate::PairSequenceInSingleTemplate),
        // Neither `A` nor `B`: not an input placement this module can credit,
        // so it does not satisfy the rule below.
        _ => {}
      }
    }
  }
  match placements {
    0 => Err(PostProcessorTemplate::NoInputSequenceInSingleTemplate),
    // Every piece emits its own encoding: a `Sequence` piece clones the one it
    // names, and a `SpecialToken` piece builds one from its ids — the latter
    // only when `add_special_tokens` is set, which every door here passes.
    1 => Ok(pieces.len()),
    repeated => Err(PostProcessorTemplate::RepeatedInputSequence(repeated)),
  }
}

#[cfg(test)]
mod tests;
