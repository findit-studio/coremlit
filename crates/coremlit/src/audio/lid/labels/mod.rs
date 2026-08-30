//! The 107-language roster this door's graph scores, and the committed asset it
//! is pinned against.
//!
//! # Provenance
//!
//! `assets/voxlingua107_labels.json` (SHA-256
//! `f13f0331965a4a402f4308ed80de662f2d55167d77e163406712ba170b92eb35`,
//! [`LABELS_JSON_LEN`] bytes) is the label roster of
//! `aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML` @
//! `2aa4d715a79e410d5f9aa32bd7a4fc9225bf9eb0` (Apache-2.0), byte-identical to
//! the copy in that author's MLX export. Its entries reproduce upstream
//! `speechbrain/lang-id-voxlingua107-ecapa`'s `label_encoder.txt` **in index
//! order with zero mismatches** — each entry's `upstream_label` field is that
//! file's own `"<code>: <name>"` line, kept so the correspondence is
//! mechanically checkable rather than asserted in prose.
//!
//! Like `audio::align`'s tokenizer asset — and unlike the gitignored `Models/`
//! artifacts — this one is deliberately COMMITTED: it is a small text file the
//! roster below is derived from, and a downstream consumer that wants the raw
//! upstream spelling can read [`labels_json_bytes`] instead of re-deriving it.
//!
//! # Why a hand-written table AND the asset
//!
//! [`languages`] is a `const` table, not a parse of the asset: parsing JSON at
//! run time would put `serde_json` in the `lid` feature's dependency graph to
//! read 107 rows that never change. The asset stays the source of truth and the
//! sibling `tests.rs` proves the two agree — the embedded bytes equal the
//! committed file, and every table row equals the corresponding asset entry —
//! so the table cannot drift from its provenance unnoticed.
//!
//! # Legacy codes: `iw` and `jw` are correct here
//!
//! Index 44 is `iw` (Hebrew) and index 46 is `jw` (Javanese), the pre-1989
//! ISO 639-1 codes. Modern code is `he` and `jv`. **Upstream keeps the old
//! spellings so that no index ever shifts**, and this roster preserves them
//! byte for byte for the same reason: rewriting `iw` to `he` here would make
//! the roster disagree with the graph's own output column ordering the moment
//! anyone re-derived it from a "cleaned" list.
//!
//! Alias folding is therefore a DOWNSTREAM job. A consumer matching this
//! door's [`Language::code`] against BCP-47 / modern ISO 639-1 tags must map
//! `iw -> he` and `jw -> jv` itself (`in -> id` is not needed: upstream already
//! spells Indonesian `id`). The sibling tests pin all three facts so a future
//! roster refresh that silently modernises the codes reds.

use crate::audio::lid::NUM_LANGUAGES;

#[cfg(test)]
mod tests;

/// Byte length of the committed label asset — a cheap, readable pin that a
/// truncated or re-serialized asset trips before any parse is attempted.
pub const LABELS_JSON_LEN: usize = 10_756;

/// One row of the language roster: the model output column, its language code,
/// and its English name.
///
/// Rows are `&'static` and interned in the [`languages`] table; a
/// [`LanguageScore`](crate::audio::lid::LanguageScore) borrows one rather than
/// copying the strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Language {
  index: usize,
  code: &'static str,
  name: &'static str,
}

impl Language {
  const fn new(index: usize, code: &'static str, name: &'static str) -> Self {
    Self { index, code, name }
  }

  /// The model output column this language occupies — the index into the
  /// `[1, NUM_LANGUAGES]` log-probability row, and into [`languages`].
  #[inline]
  pub const fn index(&self) -> usize {
    self.index
  }

  /// The language code as upstream spells it, e.g. `"th"`.
  ///
  /// Upstream's spelling, NOT a normalized BCP-47 tag: see this module's
  /// `# Legacy codes` section for `iw` (Hebrew) and `jw` (Javanese).
  #[inline]
  pub const fn code(&self) -> &'static str {
    self.code
  }

  /// The English language name, e.g. `"Thai"`.
  #[inline]
  pub const fn name(&self) -> &'static str {
    self.name
  }

  /// The roster row at model output column `index`, or `None` when `index` is
  /// out of range.
  #[inline]
  #[must_use]
  pub const fn from_index(index: usize) -> Option<&'static Self> {
    // `<[T]>::get` is not const yet, so index explicitly behind the bound.
    if index < NUM_LANGUAGES {
      Some(&LANGUAGES[index])
    } else {
      None
    }
  }

  /// The roster row whose [`Self::code`] is exactly `code`, or `None`.
  ///
  /// Exact, case-sensitive match on upstream's own spelling — it does NOT fold
  /// `he`/`jv` onto their legacy `iw`/`jw` rows (this module's
  /// `# Legacy codes`), because a door that silently accepted both spellings
  /// would hide the very drift the roster pins exist to catch.
  ///
  /// Binary search: the roster is code-sorted, which the sibling tests pin.
  #[must_use]
  pub fn from_code(code: &str) -> Option<&'static Self> {
    LANGUAGES
      .binary_search_by(|entry| entry.code.cmp(code))
      .ok()
      .map(|index| &LANGUAGES[index])
  }
}

/// The whole roster, in model output-column order (so `languages()[i].index()
/// == i`). Also code-sorted, which is what lets [`Language::from_code`] binary
/// search — the two orders coincide because upstream's label encoder was built
/// from a code-sorted list.
#[inline]
#[must_use]
pub const fn languages() -> &'static [Language; NUM_LANGUAGES] {
  &LANGUAGES
}

/// Bytes of the committed label asset (`assets/voxlingua107_labels.json`), as
/// `include_bytes!` embeds them at build time.
///
/// Bytes, not a path: a path built on `env!("CARGO_MANIFEST_DIR")` resolves
/// only on the machine and source tree that compiled the crate, which reads
/// back correctly in-tree purely by accident and breaks as soon as the crate is
/// consumed as an installed dependency. This is the same choice
/// `audio::align`'s tokenizer asset makes, for the same reason.
///
/// The roster this module exposes is [`languages`]; these bytes are the
/// provenance record it is pinned against, and carry each row's raw upstream
/// `label_encoder.txt` line in the `upstream_label` field.
#[must_use]
pub const fn labels_json_bytes() -> &'static [u8] {
  include_bytes!("../assets/voxlingua107_labels.json")
}

/// The roster, in model output-column order. Derived mechanically from
/// `assets/voxlingua107_labels.json`; the sibling `tests.rs` proves the two
/// still agree row by row.
static LANGUAGES: [Language; NUM_LANGUAGES] = [
  Language::new(0, "ab", "Abkhazian"),
  Language::new(1, "af", "Afrikaans"),
  Language::new(2, "am", "Amharic"),
  Language::new(3, "ar", "Arabic"),
  Language::new(4, "as", "Assamese"),
  Language::new(5, "az", "Azerbaijani"),
  Language::new(6, "ba", "Bashkir"),
  Language::new(7, "be", "Belarusian"),
  Language::new(8, "bg", "Bulgarian"),
  Language::new(9, "bn", "Bengali"),
  Language::new(10, "bo", "Tibetan"),
  Language::new(11, "br", "Breton"),
  Language::new(12, "bs", "Bosnian"),
  Language::new(13, "ca", "Catalan"),
  Language::new(14, "ceb", "Cebuano"),
  Language::new(15, "cs", "Czech"),
  Language::new(16, "cy", "Welsh"),
  Language::new(17, "da", "Danish"),
  Language::new(18, "de", "German"),
  Language::new(19, "el", "Greek"),
  Language::new(20, "en", "English"),
  Language::new(21, "eo", "Esperanto"),
  Language::new(22, "es", "Spanish"),
  Language::new(23, "et", "Estonian"),
  Language::new(24, "eu", "Basque"),
  Language::new(25, "fa", "Persian"),
  Language::new(26, "fi", "Finnish"),
  Language::new(27, "fo", "Faroese"),
  Language::new(28, "fr", "French"),
  Language::new(29, "gl", "Galician"),
  Language::new(30, "gn", "Guarani"),
  Language::new(31, "gu", "Gujarati"),
  Language::new(32, "gv", "Manx"),
  Language::new(33, "ha", "Hausa"),
  Language::new(34, "haw", "Hawaiian"),
  Language::new(35, "hi", "Hindi"),
  Language::new(36, "hr", "Croatian"),
  Language::new(37, "ht", "Haitian"),
  Language::new(38, "hu", "Hungarian"),
  Language::new(39, "hy", "Armenian"),
  Language::new(40, "ia", "Interlingua"),
  Language::new(41, "id", "Indonesian"),
  Language::new(42, "is", "Icelandic"),
  Language::new(43, "it", "Italian"),
  Language::new(44, "iw", "Hebrew"),
  Language::new(45, "ja", "Japanese"),
  Language::new(46, "jw", "Javanese"),
  Language::new(47, "ka", "Georgian"),
  Language::new(48, "kk", "Kazakh"),
  Language::new(49, "km", "Central Khmer"),
  Language::new(50, "kn", "Kannada"),
  Language::new(51, "ko", "Korean"),
  Language::new(52, "la", "Latin"),
  Language::new(53, "lb", "Luxembourgish"),
  Language::new(54, "ln", "Lingala"),
  Language::new(55, "lo", "Lao"),
  Language::new(56, "lt", "Lithuanian"),
  Language::new(57, "lv", "Latvian"),
  Language::new(58, "mg", "Malagasy"),
  Language::new(59, "mi", "Maori"),
  Language::new(60, "mk", "Macedonian"),
  Language::new(61, "ml", "Malayalam"),
  Language::new(62, "mn", "Mongolian"),
  Language::new(63, "mr", "Marathi"),
  Language::new(64, "ms", "Malay"),
  Language::new(65, "mt", "Maltese"),
  Language::new(66, "my", "Burmese"),
  Language::new(67, "ne", "Nepali"),
  Language::new(68, "nl", "Dutch"),
  Language::new(69, "nn", "Norwegian Nynorsk"),
  Language::new(70, "no", "Norwegian"),
  Language::new(71, "oc", "Occitan"),
  Language::new(72, "pa", "Panjabi"),
  Language::new(73, "pl", "Polish"),
  Language::new(74, "ps", "Pushto"),
  Language::new(75, "pt", "Portuguese"),
  Language::new(76, "ro", "Romanian"),
  Language::new(77, "ru", "Russian"),
  Language::new(78, "sa", "Sanskrit"),
  Language::new(79, "sco", "Scots"),
  Language::new(80, "sd", "Sindhi"),
  Language::new(81, "si", "Sinhala"),
  Language::new(82, "sk", "Slovak"),
  Language::new(83, "sl", "Slovenian"),
  Language::new(84, "sn", "Shona"),
  Language::new(85, "so", "Somali"),
  Language::new(86, "sq", "Albanian"),
  Language::new(87, "sr", "Serbian"),
  Language::new(88, "su", "Sundanese"),
  Language::new(89, "sv", "Swedish"),
  Language::new(90, "sw", "Swahili"),
  Language::new(91, "ta", "Tamil"),
  Language::new(92, "te", "Telugu"),
  Language::new(93, "tg", "Tajik"),
  Language::new(94, "th", "Thai"),
  Language::new(95, "tk", "Turkmen"),
  Language::new(96, "tl", "Tagalog"),
  Language::new(97, "tr", "Turkish"),
  Language::new(98, "tt", "Tatar"),
  Language::new(99, "uk", "Ukrainian"),
  Language::new(100, "ur", "Urdu"),
  Language::new(101, "uz", "Uzbek"),
  Language::new(102, "vi", "Vietnamese"),
  Language::new(103, "war", "Waray"),
  Language::new(104, "yi", "Yiddish"),
  Language::new(105, "yo", "Yoruba"),
  Language::new(106, "zh", "Chinese"),
];
