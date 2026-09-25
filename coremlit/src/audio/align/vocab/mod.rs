//! Bridge between the chordai base960h CTC vocabulary and the
//! `tokenizers`-crate schema asry's seam builder needs (design spec §3.1/§6,
//! `docs/superpowers/specs/2026-07-11-alignkit-forced-alignment-design.md`).
//!
//! `chordai/wav2vec2-base960h-aligner-coreml` ships a raw `{token: id}` CTC
//! dict (`Models/alignkit/base960h_dict.json`), not a HuggingFace
//! `tokenizer.json` — both asry's own `Aligner::from_paths` and alignkit's
//! [`crate::audio::align::aligner::Aligner::from_paths`] (via
//! [`asry::emissions::EmissionsAligner::builder`]) need the latter. This
//! module owns the derived, committed asset that fills that gap
//! (`assets/chordai_base960h_tokenizer.json`) plus the vocabulary constants
//! the seam validates a loaded tokenizer against — and [`Vocabulary`], which
//! applies the same rule set at run time to the table ANY model ships beside
//! it, so an aligner spells with its model's own alphabet.
//!
//! # Generator note (reproducibility record)
//!
//! `assets/chordai_base960h_tokenizer.json` is mechanically derived from
//! `Models/alignkit/base960h_dict.json` (SHA-256
//! `ef41495ab958d4416ad2f81ea51a77d4a3c79cace96e92e978c443c7bfbdd2e5`, the
//! same file `tests/model_io.rs` pins) by this rule set — re-running it
//! reproduces the asset byte-for-byte:
//!
//! 1. Parse the dict file as a flat JSON object `{token: id}` (29 entries).
//! 2. Copy every `(token, id)` pair unmodified into `model.vocab`. Key
//!    order is not semantically meaningful — the `tokenizers` crate's
//!    `WordLevel` model deserializes `vocab` into a hash map — so the
//!    asset simply preserves the dict's own id-ascending order for
//!    reviewability.
//! 3. Set `model.type = "WordLevel"` and `model.unk_token = "<unk>"`.
//!    `unk_token` is a REQUIRED key for `tokenizers` 0.23's `WordLevel`
//!    deserializer (its visitor's `missing_fields` check covers `vocab`
//!    and `unk_token`), but it is never validated against `vocab` at parse
//!    time — `WordLevelBuilder::build` stores whatever string it's given
//!    unchecked, and `Model::get_vocab_size` counts only `vocab`'s own
//!    entries. `"<unk>"` is deliberately NOT one of the 29 vocab entries
//!    (this CTC alphabet has no unknown-token concept), and doesn't need
//!    to be for the file to parse or for `VOCAB_SIZE` to stay exactly 29.
//! 4. Set every other top-level field to its schema default: `version =
//!    "1.0"` (the only value `tokenizers` 0.23 accepts), `truncation` /
//!    `padding` / `normalizer` / `pre_tokenizer` / `post_processor` /
//!    `decoder` = `null`, `added_tokens = []`. Neither this crate's vocab
//!    bridge nor asry's own runtime tokenization
//!    (`asry/src/runner/aligner/algorithm/tokenize.rs`) ever calls
//!    `Tokenizer::encode` — both go through `token_to_id` /
//!    `get_vocab_size` directly — so these pipeline fields are inert for
//!    this asset's purpose.
//!
//! Step 4's claim about asry holds from asry 0.2 (asry#21), the version this
//! crate requires. asry 0.1 classified each character by running it alone
//! through `Tokenizer::encode`, and a `WordLevel` model whose declared
//! `unk_token` is absent from its vocabulary — step 3's shape, on purpose —
//! answers that with `MissingUnkToken` for every character outside the 29: the
//! whole chunk failed before any OOV policy could decide the character. asry
//! 0.2 looks the (ASCII-uppercased) character up with `Tokenizer::token_to_id`
//! instead, so a character with no entry is an `OovKind::Symbol` event, and the
//! absent unknown token costs nothing.
//!
//! # Where the asset is consumed
//!
//! Parsing a vocabulary into a live tokenizer, and reporting a parse / blank /
//! delimiter failure, both happen inside asry's seam builder when an
//! [`crate::audio::align::aligner::Aligner`] hands it a [`Vocabulary`]'s
//! tokenizer document — [`tokenizer_json_bytes`] for the bundled table; that
//! failure surfaces as [`crate::audio::align::error::AlignerError::Seam`]. This
//! module constructs no `Tokenizer` itself (it needs no `tokenizers` dependency
//! outside tests): [`Vocabulary::from_json`] only validates a table's ids and
//! finds its blank, then writes the document by the rule set above.

use core::num::NonZeroUsize;
use std::{
  borrow::Cow,
  collections::{BTreeMap, btree_map::Entry},
  path::Path,
};

use crate::audio::align::error::{MissingId, VocabularyError, VocabularyRead};

/// Number of entries in the chordai base960h CTC vocabulary, including the
/// blank and word-delimiter tokens.
///
/// Derived from `Models/alignkit/base960h_dict.json` (see this module's
/// `# Generator note`). asry's `validate_vocab_dim` requires the CTC head's
/// output width `V` to equal the tokenizer's vocab size EXACTLY;
/// `base960h_aligner.mlmodelc` declares `emissions` `[1, 2999, 29]`
/// (`tests/model_io.rs`), so this is the width of the model the bundled table
/// belongs to. The encoder reads a model's width rather than assuming this
/// one; [`Vocabulary::size`] is what an aligner pairs it with.
pub const VOCAB_SIZE: usize = 29;

/// CTC blank-token id in the chordai base960h vocabulary.
///
/// The dict maps the literal token `"-"` to id `0` — chordai's own CTC
/// blank convention. This is distinct from the `<pad>` / `[PAD]` /
/// `<blank>` special-token probe asry's `detect_blank_token_id` performs by
/// default: this vocabulary has no `<pad>`-style entry at all, only the bare
/// `"-"` at id `0`. [`crate::audio::align::aligner::Aligner::from_paths`] therefore passes
/// this constant to the seam builder's `.blank_token_id(..)` explicitly
/// (the default auto-detect would fail construction).
pub const BLANK_ID: u32 = 0;

/// wav2vec2 inter-word delimiter token.
///
/// asry resolves the delimiter dynamically via `tokenizer.token_to_id("|")`
/// (`asry/src/runner/aligner/aligner.rs:1132`, in
/// `validate_word_delimiter_present`) rather than assuming a fixed id; this
/// constant is the TOKEN STRING that lookup uses, not its id — id `1` in
/// this vocabulary (`tests::word_delimiter_resolves_via_token_to_id`).
pub const WORD_DELIMITER: &str = "|";

/// Bytes of the committed tokenizer asset
/// (`assets/chordai_base960h_tokenizer.json`), in the `tokenizers`-crate
/// schema asry's loader accepts on its fast path. Unlike the model
/// artifacts under the gitignored `Models/` store, this asset is
/// deliberately committed: it is a small authored text file this crate
/// owns, not a downloaded artifact. Its schema is an explicit
/// `"model": {"type": "WordLevel", ...}` object never needs the
/// `load_tokenizer_with_compat` compat-patch shim
/// (`asry/src/runner/aligner/aligner.rs:1198`) that exists only for
/// upstream exports missing that discriminator.
///
/// # Why bytes, not a path
///
/// `include_bytes!` embeds the asset in the compiled artifact at build
/// time. A path helper built on `env!("CARGO_MANIFEST_DIR")` would only
/// resolve on the machine and source tree that built the crate — it reads
/// back correctly today only by accident of running in-tree, and breaks
/// the moment the crate is used as an installed/packaged dependency
/// elsewhere. Bytes also match asry's loader one step further downstream
/// than a path would: `load_tokenizer_with_compat` immediately turns
/// whatever path it's given into bytes (`std::fs::read`) before ever
/// calling `Tokenizer::from_bytes` — never `Tokenizer::from_file`, despite
/// that function's own error-message text saying so. These bytes are exactly
/// what [`crate::audio::align::aligner::Aligner::from_paths`] hands to
/// [`asry::emissions::EmissionsAligner::builder`] with no filesystem
/// round-trip. A vocabulary that ships beside a model is the caller's file,
/// like the model itself, and is read through [`Vocabulary::from_file`].
pub const fn tokenizer_json_bytes() -> &'static [u8] {
  include_bytes!("../assets/chordai_base960h_tokenizer.json")
}

/// [`VOCAB_SIZE`] as the [`NonZeroUsize`] a [`Vocabulary`] carries. The
/// conversion is infallible: `VOCAB_SIZE` is the nonzero constant `29`.
const BUNDLED_SIZE: NonZeroUsize = match NonZeroUsize::new(VOCAB_SIZE) {
  Some(size) => size,
  None => unreachable!(),
};

/// The names a CTC blank goes by wherever it sits, in the order
/// [`Vocabulary::from_json`] tries them: the three asry's own seam builder
/// probes (`<pad>`, `[PAD]`, `<blank>`, the HuggingFace conventions).
const NAMED_BLANKS: [&str; 3] = ["<pad>", "[PAD]", "<blank>"];

/// The chordai and torchaudio blank, `-` — a blank only at id 0, where both
/// conventions put it (`base960h_dict.json`: `"-": 0`, [`BLANK_ID`]). Anywhere
/// else a `-` is an ordinary character, the hyphen, and naming it the blank
/// would read every hyphen's column as silence.
const INDEXED_BLANK: (&str, u32) = ("-", 0);

/// A `{token: id}` JSON object read entry by entry, so a token the object
/// names twice reaches [`Vocabulary::from_json`] twice. A map's insert would
/// keep one of the two ids and say nothing.
struct Entries(Vec<(String, u32)>);

impl<'de> serde::Deserialize<'de> for Entries {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    struct Visit;
    impl<'de> serde::de::Visitor<'de> for Visit {
      type Value = Entries;
      fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("a JSON object mapping each token to its id")
      }
      fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Entries, A::Error> {
        let mut entries = Vec::new();
        while let Some(entry) = map.next_entry::<String, u32>()? {
          entries.push(entry);
        }
        Ok(Entries(entries))
      }
    }
    deserializer.deserialize_map(Visit)
  }
}

/// The unknown token a written tokenizer document declares (the generator
/// note's step 3).
const UNKNOWN_TOKEN: &str = "<unk>";

/// A CTC vocabulary: the table an aligner spells with, one entry per class of
/// its model's CTC head.
///
/// [`Self::bundled`] is the 29-class English table every
/// [`Aligner::from_paths`](crate::audio::align::aligner::Aligner::from_paths)
/// binds. A model that ships its own table beside it — a flat `{token: id}`
/// JSON object, as chordai's `base960h_dict.json` and HuggingFace's
/// `vocab.json` are — is read with [`Self::from_file`] (or [`Self::from_json`])
/// and paired with that model by
/// [`Aligner::from_paths_with_vocabulary`](crate::audio::align::aligner::Aligner::from_paths_with_vocabulary),
/// which refuses the pair at load unless the table has exactly one entry per
/// class of the model's head. This is how an aligner comes to spell a language
/// other than English: the model supplies the alphabet, not this crate.
#[derive(Clone)]
pub struct Vocabulary {
  /// The table as the `tokenizers`-crate document asry's seam builder parses.
  tokenizer_json: Cow<'static, [u8]>,
  /// Number of entries: the CTC head width this table names.
  size: NonZeroUsize,
  /// Id of the CTC blank.
  blank_id: u32,
}

impl Vocabulary {
  /// The bundled 29-class English table (chordai base960h): the document
  /// [`tokenizer_json_bytes`], blank [`BLANK_ID`].
  #[must_use]
  pub const fn bundled() -> Self {
    Self {
      tokenizer_json: Cow::Borrowed(tokenizer_json_bytes()),
      size: BUNDLED_SIZE,
      blank_id: BLANK_ID,
    }
  }

  /// Read a flat `{token: id}` JSON table — the shape of the vocabulary a CTC
  /// model ships beside it.
  ///
  /// The object must name each token once, and every id in `0..n` exactly once
  /// (`n` its entry count): a CTC head has one column per class, and each id
  /// is the column its token is scored in. The blank is the entry named
  /// `<pad>`, `[PAD]` or `<blank>` — the names asry's own seam builder probes
  /// for, in that order — or, failing those, a `-` at id 0, the chordai and
  /// torchaudio convention `base960h_dict.json` follows. The tokenizer document
  /// asry parses is then written by this module's generator rule set, the one
  /// the bundled asset was derived by: read through here, the staged
  /// `base960h_dict.json` yields the bundled table.
  ///
  /// # Errors
  /// [`VocabularyError::Parse`] if `json` is not a JSON object mapping each
  /// token to a non-negative integer id that fits a `u32`;
  /// [`VocabularyError::DuplicateToken`] if the object names a token twice;
  /// [`VocabularyError::NoBlank`] if no entry is a blank (an empty table
  /// included); [`VocabularyError::MissingId`] if an id in `0..n` names no
  /// token.
  pub fn from_json(json: &[u8]) -> Result<Self, VocabularyError> {
    let Entries(entries) =
      serde_json::from_slice(json).map_err(|error| VocabularyError::Parse(error.to_string()))?;
    let mut table = BTreeMap::new();
    for (token, id) in entries {
      match table.entry(token) {
        Entry::Occupied(repeated) => {
          return Err(VocabularyError::DuplicateToken(repeated.key().clone()));
        }
        Entry::Vacant(slot) => {
          slot.insert(id);
        }
      }
    }
    let size = NonZeroUsize::new(table.len()).ok_or(VocabularyError::NoBlank)?;
    let (indexed_name, indexed_id) = INDEXED_BLANK;
    let blank_id = NAMED_BLANKS
      .iter()
      .find_map(|name| table.get(*name).copied())
      .or_else(|| {
        table
          .get(indexed_name)
          .copied()
          .filter(|&id| id == indexed_id)
      })
      .ok_or(VocabularyError::NoBlank)?;

    // `n` ids, each in `0..n` at most once, is exactly "each id in `0..n`
    // once": a duplicate or an id past the end leaves one below it unnamed.
    let mut named = vec![false; size.get()];
    for &id in table.values() {
      if let Some(slot) = usize::try_from(id).ok().and_then(|id| named.get_mut(id)) {
        *slot = true;
      }
    }
    if let Some(id) = named.iter().position(|named| !named) {
      return Err(VocabularyError::MissingId(MissingId::new(id, size.get())));
    }

    let document = serde_json::json!({
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": table,
        "unk_token": UNKNOWN_TOKEN,
      },
    });
    Ok(Self {
      tokenizer_json: Cow::Owned(document.to_string().into_bytes()),
      size,
      blank_id,
    })
  }

  /// Read the `{token: id}` table in the file at `path` — the vocabulary that
  /// ships beside a model, such as `base960h_dict.json` beside
  /// `base960h_aligner.mlmodelc`.
  ///
  /// # Errors
  /// [`VocabularyError::Read`] if the file cannot be read; otherwise as
  /// [`Self::from_json`].
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self, VocabularyError> {
    let path = path.as_ref();
    let json = std::fs::read(path)
      .map_err(|source| VocabularyError::Read(VocabularyRead::new(path.to_path_buf(), source)))?;
    Self::from_json(&json)
  }

  /// Number of entries: the width of the CTC head a model paired with this
  /// table must have.
  #[must_use]
  pub const fn size(&self) -> NonZeroUsize {
    self.size
  }

  /// Id of the CTC blank.
  #[must_use]
  pub const fn blank_id(&self) -> u32 {
    self.blank_id
  }

  /// The tokenizer document asry's seam builder parses.
  pub(crate) fn tokenizer_json(&self) -> &[u8] {
    &self.tokenizer_json
  }
}

/// The table's two facts — its size and its blank — rather than the tokenizer
/// document's bytes.
impl core::fmt::Debug for Vocabulary {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Vocabulary")
      .field("size", &self.size)
      .field("blank_id", &self.blank_id)
      .finish_non_exhaustive()
  }
}

#[cfg(test)]
mod tests;
