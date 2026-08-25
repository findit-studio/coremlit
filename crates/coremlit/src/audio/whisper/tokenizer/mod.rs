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
//! looks directly inside the given folder, at `tokenizer.json` (required)
//! and `tokenizer_config.json` (optional — read for one flag, see
//! `clean_up_tokenization_spaces_from`).

use std::path::Path;

use unicode_categories::UnicodeCategories;

use crate::audio::whisper::{constants, error::TokenizerError, options::WordGrouping};

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

/// The LOWEST `special_token_begin` any Whisper vocabulary reports — the
/// conservative test for "a decoder might treat this id as special" available
/// to code that holds no tokenizer.
///
/// [`SpecialTokens::special_token_begin`] is probed per vocabulary
/// (`<|endoftext|>`'s id), so it is a property of the loaded artifact rather
/// than a constant: multilingual Whisper puts it at
/// `DEFAULT_SPECIAL_TOKEN_BEGIN` (`50257`, the value every tokenizer under
/// `Models/tokenizers/` probes to), and English-only Whisper — the `51864`-vocab
/// variants [`crate::audio::whisper::model::detect_variant`] recognizes as
/// `tiny.en`/`base.en`/`small.en`/`medium.en` — reuses GPT-2's table, where
/// `<|endoftext|>` is `50256` and every special id shifts down by one. This is
/// the minimum of the two, so `id < MIN_SPECIAL_TOKEN_BEGIN` implies `id <
/// special_token_begin` for EITHER family.
///
/// It is deliberately a floor rather than "the" threshold. Over-estimating a
/// vocabulary's special range only makes a caller treat one ORDINARY
/// multilingual id — `50256`, which that vocabulary maps to the empty string —
/// as though a filter might drop it; under-estimating it lets a genuinely
/// filtered id through. [`crate::audio::whisper::stream::agreement`]'s holdback
/// is the caller this exists for: it must decide, with no tokenizer in hand,
/// whether [`crate::audio::whisper::decode::prefill_tokens`] will carry a word's
/// tokens into the initial prompt whole.
///
/// **It is a DEFAULT, not an invariant, and nothing here enforces it.**
/// [`WhisperTokenizer::from_folder`] loads any parseable `tokenizer.json` and
/// takes [`SpecialTokens::special_token_begin`] from whatever `<|endoftext|>`
/// that artifact happens to map to — including something lower, which would make
/// this an OVER-estimate of nothing and an UNDER-estimate of that vocabulary's
/// special range. The loader deliberately does not reject such an artifact: this
/// bound is one module's premise, the rest of the crate decodes such a
/// vocabulary correctly, and refusing the load would take the whole pipeline down
/// over a streaming-only concern. The engine takes the real value instead — see
/// [`crate::audio::whisper::stream::agreement::LocalAgreement::special_token_begin`],
/// which defaults to this and which
/// [`crate::audio::whisper::stream::agreement::LocalAgreementTranscriber::new`]
/// sets from the loaded vocabulary.
pub const MIN_SPECIAL_TOKEN_BEGIN: u32 = 50_256;

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

/// Swift's `Config.boolean()` (`Config.swift:373-375`, delegating to the
/// `Config.Data.boolean()` coercion at `:152-170`) over a `serde_json` value —
/// the reason a tokenizer config's `true`, `"True"` and `1` are one answer:
///
/// - a JSON boolean is itself;
/// - a JSON **number** is `true` iff it is `1`, because Swift coerces through
///   `Int` (`val == 1`) — so every other integer, `0` and `2` included, is
///   `false`, not "no value";
/// - a JSON **string** is matched case-insensitively against `"true"`/`"t"`/
///   `"1"` and `"false"`/`"f"`/`"0"`; anything else yields no value, and Swift
///   does not trim, so `" true "` is nothing;
/// - `null`, arrays and objects yield no value.
///
/// "No value" is emphatically not `false`: the caller supplies Swift's `or:`
/// default for that case, which for the cleanup flag is `true`.
///
/// The number rule reaches floats too — not because floats are coerced
/// (`Config.Data.boolean()` has no `.floating` case at all) but because
/// `Config`'s decoder tries `Int` *before* `Float` (`Config.swift:657-666`), so
/// a JSON number an `Int` can hold generally becomes `.integer` however it was
/// spelled. `1.0` and `1e0` are therefore `true`, and `0.0`, `-0.0` and `1.5e1`
/// are `false`, while `0.5` (a real fraction) and `1e19` (past `Int64`) stay
/// `.floating` and yield no value. That ordering was read off the pinned
/// oracle's own `JSONDecoder` behavior rather than inferred from the enum,
/// which shows only the second half of it.
///
/// # Where this stops matching Swift
///
/// "However it was spelled" is *not* exact, and the gap is stated here rather
/// than papered over. `serde_json` keeps a number's spelling only while it fits
/// `i64`/`u64`; give it a fraction or an exponent and all that survives is an
/// `f64`, whereas the oracle's decoder reads the digits. The two agree on every
/// literal an `f64` carries faithfully — which is every value a real
/// `tokenizer_config.json` holds, `true` and `false` included — and can part
/// company on one carrying more precision than an `f64` has. What follows is
/// measured against the pinned oracle rather than predicted, and pinned by
/// `config_boolean_matches_swift_except_where_the_f64_lost_the_literal`.
///
/// **One `f64`, two oracle answers.** Three spellings arrive here as the
/// identical `f64` `-2^63`:
///
/// | JSON scalar              | oracle      | oracle's `boolean()` | this fn      |
/// |--------------------------|-------------|----------------------|--------------|
/// | `-9223372036854775808`   | `.integer`  | `Some(false)`        | `Some(false)`|
/// | `-9223372036854775808.0` | `.floating` | `None`               | `None`       |
/// | `-9223372036854775809`   | `.floating` | `None`               | `None`       |
/// | `-9223372036854775807.0` | `.integer`  | `Some(false)`        | **`None`**   |
///
/// Rows two to four are one `f64` and two oracle answers, so no rule reading
/// that `f64` can serve both — the choice is which side to be right on. This
/// arm decides only where the `f64` settles the question by itself and declines
/// at `-2^63` exactly, which makes the first three rows exact, row one included
/// because `serde_json` still holds its bare spelling as an `i64`. Row four is
/// the residue: a fraction- or exponent-spelled literal whose exact value lies
/// in `-9223372036854775807 ..= -9223372036854775296`, the 512 integers that
/// round to `-2^63` from above. The oracle calls those `.integer`, so cleanup
/// ends up *disabled* there and *enabled* here. The positive edge needs no such
/// concession: the oracle's own cutoff is `9223372036854775296`, the first
/// value that rounds up to the `f64` `2^63`, exactly where the upper bound
/// below already sits.
///
/// **Fractions past the precision an `f64` carries.** The same one-eyed view
/// shows up wherever a literal's digits outrun its `f64`, and the differences
/// do not all run the same way. Two measured representatives, both pinned:
/// `1844674407370955162.5` is `.floating` to the oracle (its integer path
/// gives up just past `1844674407370955161.5`, which is `.integer`) but rounds
/// to a whole `f64` and is taken here, so cleanup ends up *enabled* in Swift
/// and *disabled* here — the reverse of the table above. And
/// `0.9999999999999999` rounds all the way up to the `f64` `1.0`, so this
/// answers `Some(true)` where the oracle, still seeing a fraction, answers
/// nothing — which happens to resolve to the same flag, `or: true` being
/// `true`.
///
/// So the honest boundary is not a range but a property: this matches the
/// oracle wherever the `f64` is faithful to the literal. Beyond that, the list
/// of representatives above is measured, not exhaustive. In every case
/// measured, a divergence trades a definite answer for a default rather than
/// inverting `true` and `false`. Closing any of it would mean keeping the
/// lexeme — `serde_json`'s `arbitrary_precision`, which changes `Number`'s
/// representation for every crate in the graph — and then reproducing an
/// undocumented `JSONDecoder` numeric path exactly. For a config key whose
/// real-world values are `true` and `false`, that trade is not worth taking.
fn config_boolean(value: &serde_json::Value) -> Option<bool> {
  match value {
    serde_json::Value::Bool(b) => Some(*b),
    serde_json::Value::Number(n) => match n.as_i64() {
      Some(int) => Some(int == 1),
      // Written with a fraction or an exponent (or past `i64`): the spelling is
      // gone and only an `f64` is left, so ask the question the `f64` can still
      // answer — is it an integer strictly inside `Int`'s range? Inside, that
      // agrees with Swift's `Int`-first decode and `== 1` decides; outside, the
      // decode's `Int` attempt fails there too and `.floating` coerces to
      // nothing. `2^63` is excluded because it is past `Int.max`; `-2^63` is
      // excluded because it is the one `f64` two different Swift answers share,
      // so declining is the only way to be right about the rest of its cell.
      // See this function's doc for exactly what that costs and what else the
      // lost spelling costs.
      None => {
        const INT_MIN: f64 = -9_223_372_036_854_775_808.0; // -2^63
        const PAST_INT_MAX: f64 = 9_223_372_036_854_775_808.0; // 2^63

        let float = n.as_f64()?;
        (float > INT_MIN && float < PAST_INT_MAX && float.fract() == 0.0).then_some(float == 1.0)
      }
    },
    // Swift lowercases (`Config.swift:160`) before matching; `str::to_lowercase`
    // is the same locale-independent full Unicode mapping as `lowercased()`.
    serde_json::Value::String(s) => match s.to_lowercase().as_str() {
      "true" | "t" | "1" => Some(true),
      "false" | "f" | "0" => Some(false),
      _ => None,
    },
    serde_json::Value::Null | serde_json::Value::Array(_) | serde_json::Value::Object(_) => None,
  }
}

/// Resolves Swift's `cleanUpTokenizationSpaces` flag (`Tokenizer.swift:407`)
/// for a tokenizer folder: `tokenizer_config.json`'s
/// **`cleanUpTokenizationSpaces`, else `clean_up_tokenization_spaces`**, run
/// through [`config_boolean`], defaulting to `true` only when neither key
/// yields a value.
///
/// Swift spells that `tokenizerConfig.cleanUpTokenizationSpaces.boolean(or:
/// true)`, which looks like one snake_case lookup of a strict boolean and is
/// neither. `Config`'s dynamic-member subscript (`Config.swift:593-599`) tries
/// the member name *verbatim* first and only then its `uncamelCase` transform
/// (`:601-621`, which turns `cleanUpTokenizationSpaces` into exactly
/// `clean_up_tokenization_spaces`), so a camelCase key wins outright — and wins
/// on **presence**, not on coercibility, because `??` chains the two dictionary
/// lookups rather than their booleans. `{"cleanUpTokenizationSpaces": "yes",
/// "clean_up_tokenization_spaces": false}` is thus `true` in Swift, never
/// `false`, and a key present with `null` shadows the other spelling just as
/// firmly. Both lookups are byte-exact (`BinaryDistinctString`, the whole point
/// of that type), matching `serde_json`'s map keys with no normalization gap.
///
/// A root that is not a JSON object misses both keys and defaults, matching the
/// `dictionary()`-returned-nil arm of that same subscript (`Config.swift:598`).
///
/// This extends the same leniency one step further, to a missing or unparseable
/// file. Swift cannot reach that case (its `AutoTokenizer` fails the whole load
/// if `tokenizer_config.json` is not readable JSON), but this crate's
/// [`WhisperTokenizer::from_folder`] requires only `tokenizer.json` — a
/// contract predating this flag, and not worth narrowing over one boolean
/// whose default is what every OpenAI Whisper checkpoint ships anyway
/// (verified: `whisper-tiny` and `whisper-small` both set it to `true`
/// explicitly). A corrupt sibling file therefore degrades to Swift's own
/// default rather than failing a `tokenizer.json` that is perfectly good.
fn clean_up_tokenization_spaces_from(folder: &Path) -> bool {
  let Some(config) = std::fs::read(folder.join("tokenizer_config.json"))
    .ok()
    .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
  else {
    return true;
  };
  config
    .get("cleanUpTokenizationSpaces")
    .or_else(|| config.get("clean_up_tokenization_spaces"))
    .and_then(config_boolean)
    .unwrap_or(true)
}

/// Applies Swift's `Tokenizer.cleanUp` (`Tokenizer.swift:434-449`) — the
/// HuggingFace `clean_up_tokenization_spaces` layer, which the `tokenizers`
/// crate has no equivalent of — to an already-joined decode.
///
/// Swift's rule set is exactly ten literal (non-regex), non-overlapping,
/// left-to-right replacements, run as a chain so each one sees the previous
/// one's output. The chain below is those ten transcribed from
/// `Tokenizer.swift:439-448` in Swift's order, and is the only copy of the
/// list — `swift_clean_up_reference` in this module's tests transcribes it a
/// second time, unguarded, so the two cannot drift silently apart.
///
/// That order is load-bearing and is Swift's, not a re-derivation. The
/// witness is `" ' ,"`: `" ,"` runs first and consumes the comma's space,
/// leaving `" ',"`, where running `" ' "` first would have consumed both
/// spaces and left `"',"`. More generally, each rule deletes a space and so
/// brings its neighbours together, which can expose a match for a later one.
///
/// These ten are also exactly HuggingFace's own
/// `PreTrainedTokenizerBase.clean_up_tokenization` set, so nothing had to be
/// dropped to match Swift. Nothing is added either — in particular, Swift's
/// *other* cleanup, `WordPieceDecoder.cleanUpTokenization`
/// (`Decoder.swift:104-108`), is deliberately NOT ported: it is a different
/// rule set (regex-driven, and with an extra `" do not"` -> `" don't"`
/// rule), it is gated on its own `cleanup` config flag, and it belongs to
/// the WordPiece decoder, which a Whisper `tokenizer.json` never
/// instantiates — its `decoder` is `ByteLevel` (verified on the
/// `whisper-tiny` and `whisper-small` fixtures). Applying it here would
/// close this divergence by opening a new one.
///
/// `str::replace` matches Swift's `String.replacingOccurrences(of:with:)`
/// with no options: literal, non-overlapping, left-to-right. Every search
/// string here is pure ASCII, so Swift's canonical-equivalence-aware string
/// comparison cannot differ from Rust's byte comparison on them.
///
/// The leading scan is a pure short-circuit, not a behavioral rule: every one
/// of the ten patterns starts with a space followed by one of `.?!,'n`, so a
/// text with no such pair cannot match the first rule — and, being therefore
/// unchanged, cannot match the second, and so on down the chain. Replacements
/// only ever *delete* the leading space of a match and never insert one, so
/// the chain cannot manufacture a pair that the original text lacked. When
/// the scan finds nothing, the backend's `String` is handed back untouched
/// rather than rebuilt ten times — which matters because
/// [`WhisperTokenizer::split_tokens_on_unicode`] decodes once per token.
fn clean_up_tokenization(text: String) -> String {
  let can_match = text
    .as_bytes()
    .windows(2)
    .any(|pair| pair[0] == b' ' && matches!(pair[1], b'.' | b'?' | b'!' | b',' | b'\'' | b'n'));
  if !can_match {
    return text;
  }
  text
    .replace(" .", ".")
    .replace(" ?", "?")
    .replace(" !", "!")
    .replace(" ,", ",")
    .replace(" ' ", "'")
    .replace(" n't", "n't")
    .replace(" 'm", "'m")
    .replace(" 's", "'s")
    .replace(" 've", "'ve")
    .replace(" 're", "'re")
}

/// Whisper BPE tokenizer facade: raw encode/decode, the resolved
/// special-token table, per-language token ids, and the word-split
/// heuristics used to align decoder token output to words. Ports Swift's
/// `WhisperTokenizerWrapper` (`Models.swift:1165-1307`).
#[derive(Debug)]
pub struct WhisperTokenizer {
  tokenizer: tokenizers::Tokenizer,
  special_tokens: SpecialTokens,
  // Swift's `Tokenizer.cleanUpTokenizationSpaces` (`Tokenizer.swift:356`,
  // resolved at `:407`), read once at load: see
  // `clean_up_tokenization_spaces_from` and [`Self::decode`].
  clean_up_tokenization_spaces: bool,
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
    let folder = folder.as_ref();
    let path = folder.join("tokenizer.json");
    if !path.is_file() {
      return Err(TokenizerError::FileNotFound {
        searched: vec![path],
      });
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&path)?;
    let special_tokens = SpecialTokens::probe(&tokenizer);
    let clean_up_tokenization_spaces = clean_up_tokenization_spaces_from(folder);

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
      clean_up_tokenization_spaces,
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
  /// # The tokenization-space cleanup
  /// The joined decode is then passed through Swift's `cleanUp`
  /// (`Tokenizer.swift:434-449`) when this tokenizer folder enables it —
  /// see `clean_up_tokenization` for the ten rules and
  /// `clean_up_tokenization_spaces_from` for the flag. That is a real
  /// output change, not a formality: `decode(&[1097])`, the `Ġ...` token,
  /// is `" ..."` from the backend and `"..."` after cleanup, and the lost
  /// leading space flips the `starts_with_space` arm of the space-based
  /// word splitter behind [`Self::split_to_word_tokens`] (coremlit issue
  /// #59), merging that unit into the preceding word.
  ///
  /// This is the single place the layer lives, because it is the single
  /// place Swift puts it: `Tokenizer.decode` applies `cleanUp` to the
  /// *joined* string (`Tokenizer.swift:524`), and every Swift caller —
  /// `WhisperTokenizerWrapper.decode` (`Models.swift:1175-1176`),
  /// `splitTokensOnUnicode`'s full and per-prefix decodes
  /// (`Models.swift:1227`, `:1237`), and `splitToWordTokens`'s
  /// language-detection decode (`:1294`) — reaches it through that one
  /// method, exactly as every caller in this crate reaches it through this
  /// one. Two consequences of the *joined* placement are load-bearing and
  /// are why this is not instead a `tokenizers`-crate `Decoder`: that
  /// crate's decoders run per token and only then join, so a `" ."` split
  /// across two tokens would escape cleanup where Swift cleans it; and
  /// `tokenizers::Tokenizer` is a concrete `TokenizerImpl<..,
  /// DecoderWrapper>` whose `from_file` deserializes into fixed wrapper
  /// enums, with no seam for a custom decoder at all.
  ///
  /// # Errors
  /// [`TokenizerError::Backend`] if the tokenizer backend fails to decode
  /// `ids`.
  pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
    let decoded = self.tokenizer.decode(ids, skip_special)?;
    Ok(if self.clean_up_tokenization_spaces {
      clean_up_tokenization(decoded)
    } else {
      decoded
    })
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
  /// [`WordGrouping::FineGrained`] — the product-quality opt-in (coremlit
  /// issue #11), and this port's long-standing behavior — takes the Unicode
  /// arm for all six of the languages above.
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
      // issue-#11 opt-in (no longer the default after #41).
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
  ///
  /// Note that [`Self::decode`]'s tokenization-space cleanup is the one
  /// thing that can perturb that byte-prefix property, since it deletes
  /// spaces from `decoded` and `decoded_full` independently (a prefix
  /// ending in `" n"` is untouched where the full text's `" n't"` is not).
  /// This is not a deviation to correct: Swift routes both of its decodes
  /// through the same `cleanUp` (`Models.swift:1227`, `:1237` ->
  /// `Tokenizer.swift:524`), so its offsets are perturbed identically. It
  /// is only reachable at all when `decoded` holds a U+FFFD, i.e. mid
  /// multi-byte character, and the cleanup's patterns are pure ASCII —
  /// and where Swift would read a shifted range, this reads `None`.
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
  ///
  /// The leading-space test reads a *cleaned* decode, which is what makes
  /// [`Self::decode`]'s cleanup visible in word grouping rather than only
  /// in text: `" ..."` becomes `"..."`, which is neither space-prefixed nor
  /// a single punctuation scalar, so it joins the preceding word instead of
  /// starting a new one. Swift groups it the same way, for the same reason
  /// (coremlit issue #59). A single `" ."` is unaffected — it loses its
  /// space too, but `"."` is one punctuation scalar and still starts a word.
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
