//! Whisper BPE tokenizer facade: encode/decode, Whisper's special-token
//! table, per-language token ids, and the word-level split heuristics used
//! to align decoder token output to words. Ports `Models.swift`
//! `WhisperTokenizerWrapper`/`SpecialTokens`
//! (argmax-oss-swift `Sources/WhisperKit/Core/Models.swift:1111-1322`).
//!
//! Hub-based auto-download and the `TokenizerWrapper`/`AutoTokenizerWrapper`
//! multi-source search Swift builds on top of (`Utilities/
//! ModelUtilities.swift:16-71`) are out of scope here, matching this
//! crate's existing "folders are always local" scoping (see
//! `options::Options`'s doc): [`WhisperTokenizer::from_folder`] only ever
//! looks for `tokenizer.json` directly inside the given folder.

use std::path::Path;

use unicode_categories::UnicodeCategories;

use crate::{constants, error::TokenizerError, options::WordGrouping};

#[cfg(feature = "nl-recognizer")]
pub mod nl_recognizer;

// ---------------------------------------------------------------------
// SpecialTokens
// ---------------------------------------------------------------------

// Swift's hardcoded fallbacks, used whenever a probe below misses the
// loaded vocabulary (`Models.swift:1311-1321`).
const DEFAULT_WHITESPACE_TOKEN: u32 = 220;
const DEFAULT_SPECIAL_TOKEN_BEGIN: u32 = 50_257;
const DEFAULT_END_TOKEN: u32 = 50_257;
const DEFAULT_START_OF_PREVIOUS_TOKEN: u32 = 50_361;
const DEFAULT_START_OF_TRANSCRIPT_TOKEN: u32 = 50_258;
const DEFAULT_ENGLISH_TOKEN: u32 = 50_259;
const DEFAULT_TRANSCRIBE_TOKEN: u32 = 50_359;
const DEFAULT_TRANSLATE_TOKEN: u32 = 50_358;
const DEFAULT_NO_SPEECH_TOKEN: u32 = 50_362;
const DEFAULT_NO_TIMESTAMPS_TOKEN: u32 = 50_363;
const DEFAULT_TIME_TOKEN_BEGIN: u32 = 50_364;

/// Whisper's fixed special-token ids, resolved from the loaded tokenizer's
/// vocabulary with Swift's hardcoded defaults as fallback for any probe
/// that misses (Swift `SpecialTokens`, `Models.swift:1111-1149`; probed in
/// `WhisperTokenizerWrapper.init`, `Models.swift:1202-1215`, with defaults
/// from `Models.swift:1311-1321`).
///
/// There is no public constructor: every field is derived from a loaded
/// tokenizer's vocabulary (see [`WhisperTokenizer::from_folder`]), never
/// hand-configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
  end_token: u32,
  english_token: u32,
  no_speech_token: u32,
  no_timestamps_token: u32,
  special_token_begin: u32,
  start_of_previous_token: u32,
  start_of_transcript_token: u32,
  time_token_begin: u32,
  transcribe_token: u32,
  translate_token: u32,
  whitespace_token: u32,
}

impl SpecialTokens {
  /// Probes `tokenizer`'s vocabulary for each special token's literal
  /// string, falling back to Swift's hardcoded default id when the string
  /// is not itself a vocabulary entry (`Models.swift:1203-1214`).
  ///
  /// On a real Whisper tokenizer this fallback is not just a theoretical
  /// edge case: `no_speech_token` probes the literal string `"<|nospeech|>"`,
  /// but OpenAI's actual Whisper vocab spells that token `<|nocaptions|>`
  /// (verified against the downloaded `whisper-tiny` fixture — id 50362 is
  /// `<|nocaptions|>`, `<|nospeech|>` is absent), so this field is always
  /// resolved via [`DEFAULT_NO_SPEECH_TOKEN`] in practice — which happens
  /// to equal the real id anyway, so behavior still matches Swift exactly.
  /// Likewise `whitespace_token` probes the literal one-character string
  /// `" "`, which byte-level BPE vocabularies never contain as a literal
  /// key (they store it as `"Ġ"`, U+0120), so it also always resolves via
  /// [`DEFAULT_WHITESPACE_TOKEN`].
  fn probe(tokenizer: &tokenizers::Tokenizer) -> Self {
    let end_token = tokenizer
      .token_to_id("<|endoftext|>")
      .unwrap_or(DEFAULT_END_TOKEN);
    let english_token = tokenizer
      .token_to_id("<|en|>")
      .unwrap_or(DEFAULT_ENGLISH_TOKEN);
    let no_speech_token = tokenizer
      .token_to_id("<|nospeech|>")
      .unwrap_or(DEFAULT_NO_SPEECH_TOKEN);
    let no_timestamps_token = tokenizer
      .token_to_id("<|notimestamps|>")
      .unwrap_or(DEFAULT_NO_TIMESTAMPS_TOKEN);
    let special_token_begin = tokenizer
      .token_to_id("<|endoftext|>")
      .unwrap_or(DEFAULT_SPECIAL_TOKEN_BEGIN);
    let start_of_previous_token = tokenizer
      .token_to_id("<|startofprev|>")
      .unwrap_or(DEFAULT_START_OF_PREVIOUS_TOKEN);
    let start_of_transcript_token = tokenizer
      .token_to_id("<|startoftranscript|>")
      .unwrap_or(DEFAULT_START_OF_TRANSCRIPT_TOKEN);
    let time_token_begin = tokenizer
      .token_to_id("<|0.00|>")
      .unwrap_or(DEFAULT_TIME_TOKEN_BEGIN);
    let transcribe_token = tokenizer
      .token_to_id("<|transcribe|>")
      .unwrap_or(DEFAULT_TRANSCRIBE_TOKEN);
    let translate_token = tokenizer
      .token_to_id("<|translate|>")
      .unwrap_or(DEFAULT_TRANSLATE_TOKEN);
    let whitespace_token = tokenizer
      .token_to_id(" ")
      .unwrap_or(DEFAULT_WHITESPACE_TOKEN);

    Self {
      end_token,
      english_token,
      no_speech_token,
      no_timestamps_token,
      special_token_begin,
      start_of_previous_token,
      start_of_transcript_token,
      time_token_begin,
      transcribe_token,
      translate_token,
      whitespace_token,
    }
  }

  /// Builds the fixed multilingual-GPT-2 fallback ids directly, with no
  /// loaded tokenizer at all — the same defaults the vocabulary probe
  /// falls back to per-field when a probe misses
  /// (`Models.swift:1203-1214`, values from `Models.swift:1311-1321`).
  ///
  /// Exists so decode-chain code and its tests can build a plausible
  /// [`SpecialTokens`] table hermetically, without a `tokenizer.json`
  /// fixture on disk.
  #[inline(always)]
  pub const fn whisper_defaults() -> Self {
    Self {
      end_token: DEFAULT_END_TOKEN,
      english_token: DEFAULT_ENGLISH_TOKEN,
      no_speech_token: DEFAULT_NO_SPEECH_TOKEN,
      no_timestamps_token: DEFAULT_NO_TIMESTAMPS_TOKEN,
      special_token_begin: DEFAULT_SPECIAL_TOKEN_BEGIN,
      start_of_previous_token: DEFAULT_START_OF_PREVIOUS_TOKEN,
      start_of_transcript_token: DEFAULT_START_OF_TRANSCRIPT_TOKEN,
      time_token_begin: DEFAULT_TIME_TOKEN_BEGIN,
      transcribe_token: DEFAULT_TRANSCRIBE_TOKEN,
      translate_token: DEFAULT_TRANSLATE_TOKEN,
      whitespace_token: DEFAULT_WHITESPACE_TOKEN,
    }
  }

  /// `<|endoftext|>`'s id — Whisper's decoder EOS token.
  #[inline(always)]
  pub const fn end_token(&self) -> u32 {
    self.end_token
  }

  /// `<|en|>`'s id.
  #[inline(always)]
  pub const fn english_token(&self) -> u32 {
    self.english_token
  }

  /// The no-speech-probability probe token's id (see the vocabulary probe's doc
  /// for why this resolves via the default fallback on a real Whisper
  /// vocab rather than an actual `"<|nospeech|>"` vocabulary hit).
  #[inline(always)]
  pub const fn no_speech_token(&self) -> u32 {
    self.no_speech_token
  }

  /// `<|notimestamps|>`'s id.
  #[inline(always)]
  pub const fn no_timestamps_token(&self) -> u32 {
    self.no_timestamps_token
  }

  /// First id in the special/added-token range: every id at or above this
  /// is a special, language, or timestamp token, never plain vocabulary.
  #[inline(always)]
  pub const fn special_token_begin(&self) -> u32 {
    self.special_token_begin
  }

  /// `<|startofprev|>`'s id.
  #[inline(always)]
  pub const fn start_of_previous_token(&self) -> u32 {
    self.start_of_previous_token
  }

  /// `<|startoftranscript|>`'s id.
  #[inline(always)]
  pub const fn start_of_transcript_token(&self) -> u32 {
    self.start_of_transcript_token
  }

  /// `<|0.00|>`'s id: the first of Whisper's 1501 timestamp tokens
  /// (`<|0.00|>` through `<|30.00|>` in 0.02 s steps).
  #[inline(always)]
  pub const fn time_token_begin(&self) -> u32 {
    self.time_token_begin
  }

  /// `<|transcribe|>`'s id.
  #[inline(always)]
  pub const fn transcribe_token(&self) -> u32 {
    self.transcribe_token
  }

  /// `<|translate|>`'s id.
  #[inline(always)]
  pub const fn translate_token(&self) -> u32 {
    self.translate_token
  }

  /// A single space character's id (GPT-2 byte-level BPE's `Ġ`, U+0120 —
  /// id 220 in every Whisper vocab; see the vocabulary probe's doc).
  #[inline(always)]
  pub const fn whitespace_token(&self) -> u32 {
    self.whitespace_token
  }
}

// ---------------------------------------------------------------------
// WhisperTokenizer
// ---------------------------------------------------------------------

/// Whether `s`, after trimming Swift's `.whitespaces` character class
/// (Unicode general category `Zs` plus U+0009 CHARACTER TABULATION — this
/// is narrower than `.whitespacesAndNewlines`; it does not include
/// newlines), is exactly one Unicode scalar in general category `P*`
/// (punctuation).
///
/// Ports the inline check in `WhisperTokenizerWrapper.splitTokensOnSpaces`
/// (`Models.swift:1263-1266`): `UnicodeScalar(String)` only succeeds when
/// the string holds exactly one scalar, so multi-scalar trimmed content
/// (including the empty string) is never punctuation here, matching
/// Swift's `if let strippedSubword = UnicodeScalar(...)` guard exactly.
/// Swift's `Character`/Rust's `char` are both Unicode-scalar-grained (not
/// grapheme-cluster-grained), so this is a direct, unambiguous port.
fn is_single_punctuation_scalar(s: &str) -> bool {
  let trimmed = s.trim_matches(|c: char| c.is_separator_space() || c == '\u{0009}');
  let mut chars = trimmed.chars();
  match (chars.next(), chars.next()) {
    (Some(c), None) => c.is_punctuation(),
    _ => false,
  }
}

/// Whisper BPE tokenizer facade: raw encode/decode, the resolved
/// special-token table, per-language token ids, and the word-split
/// heuristics used to align decoder token output to words. Ports Swift's
/// `WhisperTokenizerWrapper` (`Models.swift:1165-1307`).
#[derive(Debug)]
pub struct WhisperTokenizer {
  tokenizer: tokenizers::Tokenizer,
  special_tokens: SpecialTokens,
  // `language_table` is the source of truth, probed once at load
  // (`Models.swift:1219-1223`); `language_ids` is a cached view of its
  // first components, kept because `all_language_tokens` must return a
  // `&[u32]` slice, which cannot be borrowed out of a `Vec<(u32, &str)>`
  // without allocating on every call.
  language_table: Vec<(u32, &'static str)>,
  language_ids: Vec<u32>,
}

impl WhisperTokenizer {
  /// Loads the BPE tokenizer from `folder/tokenizer.json` and derives the
  /// special-token table and per-language token ids from its vocabulary.
  ///
  /// Language ids are probed in [`constants::languages`] table order, one
  /// probe per distinct code, deduplicated by id and kept only if greater
  /// than [`SpecialTokens::special_token_begin`] — the same probe, filter,
  /// and dedup Swift's `allLanguageTokens` applies (`Models.swift:
  /// 1219-1223`), except Swift collects into a hash-ordered `Set<Int>`
  /// where this collects into an order-preserving `Vec<u32>` (a strictly
  /// more reproducible, equally correct, superset of that Set's content).
  ///
  /// # Errors
  /// [`TokenizerError::FileNotFound`] if `folder` has no `tokenizer.json`;
  /// [`TokenizerError::Backend`] if the file exists but fails to parse.
  pub fn from_folder(folder: impl AsRef<Path>) -> Result<Self, TokenizerError> {
    let path = folder.as_ref().join("tokenizer.json");
    if !path.is_file() {
      return Err(TokenizerError::FileNotFound {
        searched: vec![path],
      });
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&path)?;
    let special_tokens = SpecialTokens::probe(&tokenizer);

    let mut language_table: Vec<(u32, &'static str)> = Vec::new();
    for &(_, code) in constants::languages() {
      let Some(id) = tokenizer.token_to_id(&format!("<|{code}|>")) else {
        continue;
      };
      if id > special_tokens.special_token_begin
        && !language_table.iter().any(|&(existing, _)| existing == id)
      {
        language_table.push((id, code));
      }
    }
    let language_ids: Vec<u32> = language_table.iter().map(|&(id, _)| id).collect();

    Ok(Self {
      tokenizer,
      special_tokens,
      language_table,
      language_ids,
    })
  }

  /// Encodes `text` into token ids, without inserting Whisper's decoder
  /// prompt template (`<|startoftranscript|>`, `<|notimestamps|>`, ...,
  /// `<|endoftext|>`). This is a raw content encode.
  ///
  /// Swift's `WhisperTokenizerWrapper.encode(text:)` (`Models.swift:
  /// 1171-1173`) calls the tokenizer's single-argument `encode(text:)`,
  /// which defaults `addSpecialTokens: true` (`Tokenizer.swift:500-502`)
  /// and so *does* apply this tokenizer.json's `TemplateProcessing`
  /// post-processor. WhisperKit's own call sites then immediately strip
  /// the template back out, e.g. `tokenizer.encode(text:
  /// prefixText).filter { $0 < tokenizer.specialTokens.specialTokenBegin }`
  /// (`Tests/WhisperKitTests/UnitTests.swift:1710`). Encoding here with
  /// `add_special_tokens: false` produces the identical content ids in one
  /// fewer pass: `TemplateProcessing` only wraps the already-tokenized
  /// content sequence and does not change how that sequence itself is
  /// tokenized, so add-then-filter and never-add are equivalent for the
  /// ids this method returns.
  ///
  /// # Errors
  /// [`TokenizerError::Backend`] if the tokenizer backend fails to encode
  /// `text`.
  pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
    Ok(self.tokenizer.encode(text, false)?.get_ids().to_vec())
  }

  /// Decodes `ids` back to text. `skip_special` mirrors Swift's
  /// `skipSpecialTokens` (`Tokenizer.swift:504-525`): when `true`, ids in
  /// the tokenizer's special-token set are dropped before joining; when
  /// `false` (Swift's `decode(tokens:)` default, `Tokenizer.swift:
  /// 304-306`), every id that resolves to a vocabulary entry is included,
  /// literal special-token strings and all.
  ///
  /// Ids absent from the vocabulary entirely are silently dropped rather
  /// than causing an error, both here (`tokenizers` 0.23.1's
  /// `Tokenizer::decode`, `tokenizer/mod.rs:901-919`, `filter_map`s ids
  /// that neither the added-token table nor the base model resolve) and in
  /// Swift (`Tokenizer.swift:510-521`'s `compactMap`). See
  /// [`Self::split_to_word_tokens`]'s doc for why this module relies on
  /// that shared behavior instead of pre-filtering.
  ///
  /// # Errors
  /// [`TokenizerError::Backend`] if the tokenizer backend fails to decode
  /// `ids`.
  pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
    Ok(self.tokenizer.decode(ids, skip_special)?)
  }

  /// Looks up a token string's id, if the vocabulary has it.
  #[inline(always)]
  pub fn token_to_id(&self, token: &str) -> Option<u32> {
    self.tokenizer.token_to_id(token)
  }

  /// Looks up a token id's string, if the vocabulary has it.
  #[inline(always)]
  pub fn id_to_token(&self, id: u32) -> Option<String> {
    self.tokenizer.id_to_token(id)
  }

  /// The resolved special-token table.
  #[inline(always)]
  pub const fn special_tokens(&self) -> &SpecialTokens {
    &self.special_tokens
  }

  /// Every resolved `<|lang|>` token id, deduplicated. Ports Swift's
  /// `allLanguageTokens: Set<Int>` (`Models.swift:1219-1223`) — see
  /// [`Self::from_folder`]'s doc for the ordering deviation.
  #[inline(always)]
  pub fn all_language_tokens(&self) -> &[u32] {
    self.language_ids.as_slice()
  }

  /// The ISO language code for a language token id, if `id` is one of
  /// [`Self::all_language_tokens`].
  pub fn language_for_token(&self, id: u32) -> Option<&'static str> {
    self
      .language_table
      .iter()
      .find(|&&(tid, _)| tid == id)
      .map(|&(_, code)| code)
  }

  /// Decodes `tokens` into words and each word's contributing subtokens,
  /// choosing the split strategy by `language_code` and `grouping`:
  /// Unicode-boundary splitting (every complete Unicode scalar its own
  /// unit, merged only enough to repair BPE tokens that split a multi-byte
  /// character) for `zh`/`ja`/`th`/`lo`/`my`/`yue` — languages without
  /// reliable whitespace-delimited words — and space/punctuation-boundary
  /// splitting otherwise. Ports Swift's `splitToWordTokens(tokenIds:)`
  /// (`Models.swift:1293-1306`); `language_code` replaces Swift's
  /// `NLLanguageRecognizer.dominantLanguage` detection (spec §5.3) — the
  /// caller supplies it directly (e.g. from the decoded `<|lang|>` prompt
  /// token) instead of re-detecting it from the decoded text.
  ///
  /// # Choosing the grouping
  /// `grouping` is the second half of that decision, made explicit
  /// (coremlit issue #14).
  ///
  /// [`WordGrouping::FineGrained`] — the default, and this port's
  /// long-standing behavior — takes the Unicode arm for all six of the
  /// languages above.
  ///
  /// [`WordGrouping::SwiftParity`] reproduces **Swift's own** arm selection,
  /// which is not "spaces for all CJK": Swift matches its
  /// `NLLanguageRecognizer` result against the same six names, and
  /// `NLLanguage`'s raw values are bare for Japanese/Thai/Lao/Burmese
  /// (`ja`/`th`/`lo`/`my` — they match, and Swift Unicode-splits them) but
  /// regional for Chinese (`zh-Hans`/`zh-Hant` — they do not, so Chinese
  /// alone falls through to the space splitter; Cantonese has no
  /// `NLLanguage` case and is recognized as Chinese, so it goes the same
  /// way). This variant therefore space-splits `zh`/`yue` and Unicode-splits
  /// the rest, matching Swift's pinned Japanese expectation
  /// (`UnitTests.swift:1360-1375`).
  ///
  /// The two groupings consequently differ **only for `zh` and `yue`**. For
  /// every other `language_code` — CJK or not — they are the same splitter.
  ///
  /// # Overriding or pre-normalizing `language_code`
  /// `language_code` is an ordinary argument, not something this method
  /// derives itself — this crate's pipeline callers pass the decoder's own
  /// `<|lang|>` prompt token by default, and that stays the one source of
  /// truth (see the paragraph above). A caller that instead wants Swift's
  /// original text-based re-detection — e.g. for code-switched audio,
  /// where the decoder's single per-window language token can be a poor
  /// fit — can compute its own replacement code and pass that here. The
  /// optional `nl-recognizer` feature (off by default) ships exactly that
  /// as `tokenizer::nl_recognizer::redetect_language`, a thin wrapper over
  /// `NLLanguageRecognizer` that additionally normalizes its raw BCP-47
  /// result to a bare base code (`zh-Hant`/`zh-Hans`/`zh-*` all become
  /// `zh`) before returning — the exact normalization step Swift's own
  /// call site skips (`Models.swift:1301`), which is why a `zh-Hant`
  /// transcript falls through to space-based splitting there instead of
  /// landing on the CJK arm above (coremlit issue #9). See that
  /// function's doc for the full trade-off: a text-based second opinion
  /// can help, but it is still a second, independently-fallible signal,
  /// which is exactly why this crate does not call it automatically.
  ///
  /// `tokens` is **not** filtered before splitting, even though Swift's
  /// language-detection preamble filters its own (separate, ephemeral)
  /// decode to `id < specialTokenBegin` before feeding it to
  /// `NLLanguageRecognizer` (`Models.swift:1294`): that filtered string is
  /// used only to pick a language and is not itself split. The actual
  /// split functions always receive the full, unfiltered `tokenIds`
  /// (`Models.swift:1302` and `:1304`), special/timestamp ids included, and
  /// this port matches that exactly. This is safe because decoding an id
  /// absent from the vocabulary — the only way an out-of-range id could
  /// misbehave — silently drops it instead of erroring or panicking (see
  /// [`Self::decode`]'s doc), and because a real Whisper tokenizer's id
  /// space has no gaps in the first place (verified against the
  /// `whisper-tiny` fixture: base vocab ids `0..=50257` plus 1608
  /// contiguous added-token ids `50257..=51864` cover every id a decoder
  /// can produce). No pre-filtering is implemented, matching Swift.
  ///
  /// # Errors
  /// [`TokenizerError::Backend`] if the tokenizer backend fails to decode
  /// `tokens`.
  pub fn split_to_word_tokens(
    &self,
    tokens: &[u32],
    language_code: &str,
    grouping: WordGrouping,
  ) -> Result<Vec<(String, Vec<u32>)>, TokenizerError> {
    let unicode_split = match grouping {
      // Every non-whitespace-delimited language, fine-grained. This port's
      // default and its issue-#11 pin.
      WordGrouping::FineGrained => {
        matches!(language_code, "zh" | "ja" | "th" | "lo" | "my" | "yue")
      }
      // Swift's arm selection, expressed against the BARE base codes this
      // function is actually handed.
      //
      // Swift matches `NLLanguageRecognizer.dominantLanguage?.rawValue`
      // against `["zh", "ja", "th", "lo", "my", "yue"]` (`Models.swift:1299`)
      // -- but `NLLanguage`'s raw values are `ja`/`th`/`lo`/`my` (bare, so
      // they MATCH and Swift Unicode-splits them) and `zh-Hans`/`zh-Hant`
      // (regional, so they do NOT, and Chinese alone falls through to the
      // space splitter). Cantonese has no `NLLanguage` case at all and is
      // recognized as Chinese, so `yue` behaves the same way.
      //
      // Hence: `zh`/`yue` -> spaces, everything else per the list. Forcing
      // spaces for ALL CJK -- what this variant used to do -- would diverge
      // from Swift for Japanese, whose twelve Unicode-split groups Swift
      // pins in its own test suite (`UnitTests.swift:1360-1375`), under the
      // very name that promises parity with it.
      WordGrouping::SwiftParity => matches!(language_code, "ja" | "th" | "lo" | "my"),
    };

    if unicode_split {
      self.split_tokens_on_unicode(tokens)
    } else {
      self.split_tokens_on_spaces(tokens)
    }
  }

  /// Groups `tokens` into the fewest Unicode-scalar-complete units:
  /// accumulates tokens and re-decodes the running prefix after each one,
  /// committing it as a word as soon as its decode is either free of
  /// U+FFFD REPLACEMENT CHARACTER, or contains one that the *full* decode
  /// of all of `tokens` also has at that same position (i.e. a genuine
  /// replacement character in the source text, not an artifact of a
  /// multi-byte character split across a BPE token boundary).
  ///
  /// Ports `splitTokensOnUnicode` (`Models.swift:1226-1253`) exactly,
  /// including its actual mechanism rather than its vestigial one: Swift
  /// accumulates an `unicodeOffset` variable (`Models.swift:1233`,
  /// `:1248`) that is never read — the real gate is
  /// `decoded.range(of: replacementString)` sliced back into `decodedFull`
  /// at that *same* `String.Index` range (`Models.swift:1239-1242`), which
  /// only gives a meaningful answer because `decoded` (a lossy UTF-8 decode
  /// of a byte prefix of what produces `decodedFull`) is byte-identical to
  /// `decodedFull` up to the first incomplete multi-byte sequence. This
  /// port computes the same thing directly as a UTF-8 byte offset: find
  /// U+FFFD's byte offset in `decoded`, then check whether `decodedFull`
  /// has U+FFFD starting at that same byte offset. Unlike Swift's
  /// same-range reuse across two different strings (Apple does not
  /// document this as safe in general), this uses `str::get` so a
  /// hypothetical broken prefix invariant returns `false` instead of
  /// panicking — never observed to matter on real BPE output, since decode
  /// is a byte-prefix-preserving operation by construction, but strictly
  /// safer than the Swift original for the same result on every reachable
  /// input.
  fn split_tokens_on_unicode(
    &self,
    tokens: &[u32],
  ) -> Result<Vec<(String, Vec<u32>)>, TokenizerError> {
    let decoded_full = self.decode(tokens, false)?;
    let mut words: Vec<(String, Vec<u32>)> = Vec::new();
    let mut current_tokens: Vec<u32> = Vec::new();

    for &token in tokens {
      current_tokens.push(token);
      let decoded = self.decode(&current_tokens, false)?;

      let has_unicode_in_full_string = decoded.find('\u{FFFD}').is_some_and(|offset| {
        decoded_full
          .get(offset..)
          .and_then(|rest| rest.chars().next())
          == Some('\u{FFFD}')
      });

      if !decoded.contains('\u{FFFD}') || has_unicode_in_full_string {
        words.push((decoded, std::mem::take(&mut current_tokens)));
      }
    }

    Ok(words)
  }

  /// Merges [`Self::split_tokens_on_unicode`]'s Unicode-complete units into
  /// space/punctuation-delimited words: a unit starts a new word if its
  /// first token is a special/timestamp id (`>= special_token_begin`), it
  /// decodes with a leading space, it is exactly one punctuation scalar
  /// ([`is_single_punctuation_scalar`]), or no word has started yet;
  /// otherwise it is appended (text and tokens both) onto the previous
  /// word. Ports `splitTokensOnSpaces` (`Models.swift:1255-1277`) exactly.
  fn split_tokens_on_spaces(
    &self,
    tokens: &[u32],
  ) -> Result<Vec<(String, Vec<u32>)>, TokenizerError> {
    let subwords = self.split_tokens_on_unicode(tokens)?;
    let mut words: Vec<(String, Vec<u32>)> = Vec::new();

    for (subword, subword_tokens) in subwords {
      let is_special = subword_tokens
        .first()
        .is_some_and(|&id| id >= self.special_tokens.special_token_begin);
      let starts_with_space = subword.starts_with(' ');
      let is_punctuation = is_single_punctuation_scalar(&subword);

      if is_special || starts_with_space || is_punctuation || words.is_empty() {
        words.push((subword, subword_tokens));
      } else {
        let last = words.len() - 1;
        words[last].0.push_str(&subword);
        words[last].1.extend(subword_tokens);
      }
    }

    Ok(words)
  }
}

#[cfg(test)]
mod tests;
