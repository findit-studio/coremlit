//! Whisper pipeline constants. Ports `Models.swift` `Constants`
//! (argmax-oss-swift `Sources/WhisperKit/Core/Models.swift:1330-1460`).

/// Input sample rate the models are trained on.
pub const SAMPLE_RATE: u32 = 16_000;
/// Samples per 30 s encoder window.
pub const WINDOW_SAMPLES: usize = 480_000;
/// Seconds per encoder window.
pub const WINDOW_SECONDS: f32 = 30.0;
/// Maximum decoder token context (Swift: `448 / 2`).
pub const MAX_TOKEN_CONTEXT: usize = 224;
/// Seconds spanned by one timestamp-token step (`<|0.00|>` to `<|0.02|>`).
/// Ports `WhisperKit.secondsPerTimeToken` (`Core/WhisperKit.swift:40`).
pub const SECONDS_PER_TIME_TOKEN: f32 = 0.02;
/// Fallback language when detection is off and none is set.
pub const DEFAULT_LANGUAGE_CODE: &str = "en";
/// Punctuation merged onto the FOLLOWING word by word-timestamp merging
/// (Swift `defaultPrependPunctuations`, Models.swift:1459).
pub const PREPEND_PUNCTUATION: &str = "\"'“¡¿([{-";
/// Punctuation merged onto the PRECEDING word (Swift
/// `defaultAppendPunctuations`, Models.swift:1460).
pub const APPEND_PUNCTUATION: &str = "\"'.。,，!！?？:：”)]}、";

/// Upstream-compatible marker some Whisper models emit as decoded text for
/// silent or near-silent audio. This is a training-data artifact baked
/// into the model's learned output, not a tokenizer special token or a
/// control code this crate inserts: the decoder literally samples the BPE
/// tokens that spell this string out, the same as it would any other
/// text, and this crate decodes and reports it faithfully rather than
/// intercepting it.
///
/// Matches Swift's own observed output for the same input bit-for-bit
/// (coremlit issue #9, "Silence output should be filtered or explicitly
/// modeled": both runtimes produced exactly `[BLANK_AUDIO]`, one segment,
/// for 5 s of silence under matched VAD/prefill settings). Because this is
/// upstream-compatible model behavior rather than a bug, this crate does
/// not filter it out of
/// [`TranscriptionResult`](crate::result::TranscriptionResult)/
/// [`TranscriptionSegment`](crate::result::TranscriptionSegment) text —
/// product layers that don't want `[BLANK_AUDIO]` polluting search or
/// timeline results must filter or model it themselves.
///
/// **Result-level equality is the validated contract.** The pinned
/// silence golden asserts
/// [`TranscriptionResult::text`](crate::result::TranscriptionResult::text)
/// exactly equal to this constant. Segment text is a different shape:
/// under that same validated configuration (default
/// `skip_special_tokens == false`),
/// [`TranscriptionSegment::text`](crate::result::TranscriptionSegment::text)
/// is the undecorated per-segment decode and still carries its
/// special/timestamp tokens
/// (`<|startoftranscript|><|en|><|transcribe|><|0.00|> [BLANK_AUDIO]<|10.00|><|endoftext|>`
/// for that golden) — comparing it against this constant with equality
/// will not match there. Setting `skip_special_tokens` filters those
/// tokens out of segment text too (see
/// [`segment::find_seek_point_and_segments`](crate::segment::find_seek_point_and_segments)),
/// which narrows the gap, but that combination isn't itself pinned by a
/// golden here: treat a segment-level check as `contains`, not
/// equality, or filter/model on
/// [`TranscriptionResult::text`](crate::result::TranscriptionResult::text)
/// instead.
pub const BLANK_AUDIO_MARKER: &str = "[BLANK_AUDIO]";

/// Whisper language table: `(english_name, iso_code)`, 112 entries.
///
/// Extracted verbatim from `Models.swift` `Constants.languages`.
pub fn languages() -> &'static [(&'static str, &'static str)] {
  LANGUAGES
}

/// Resolves an English language name or ISO code to the ISO code.
pub fn language_code(name_or_code: &str) -> Option<&'static str> {
  LANGUAGES
    .iter()
    .find_map(|(name, code)| (*name == name_or_code || *code == name_or_code).then_some(*code))
}

// Extracted from Models.swift with:
//   awk '/static let languages/,/^    \]/' \
//     <argmax-oss-swift>/Sources/WhisperKit/Core/Models.swift \
//   | grep '":' | sed -E 's/^[[:space:]]*"([^"]+)": "([^"]+)",?/    ("\1", "\2"),/'
static LANGUAGES: &[(&str, &str)] = &[
  ("english", "en"),
  ("chinese", "zh"),
  ("german", "de"),
  ("spanish", "es"),
  ("russian", "ru"),
  ("korean", "ko"),
  ("french", "fr"),
  ("japanese", "ja"),
  ("portuguese", "pt"),
  ("turkish", "tr"),
  ("polish", "pl"),
  ("catalan", "ca"),
  ("dutch", "nl"),
  ("arabic", "ar"),
  ("swedish", "sv"),
  ("italian", "it"),
  ("indonesian", "id"),
  ("hindi", "hi"),
  ("finnish", "fi"),
  ("vietnamese", "vi"),
  ("hebrew", "he"),
  ("ukrainian", "uk"),
  ("greek", "el"),
  ("malay", "ms"),
  ("czech", "cs"),
  ("romanian", "ro"),
  ("danish", "da"),
  ("hungarian", "hu"),
  ("tamil", "ta"),
  ("norwegian", "no"),
  ("thai", "th"),
  ("urdu", "ur"),
  ("croatian", "hr"),
  ("bulgarian", "bg"),
  ("lithuanian", "lt"),
  ("latin", "la"),
  ("maori", "mi"),
  ("malayalam", "ml"),
  ("welsh", "cy"),
  ("slovak", "sk"),
  ("telugu", "te"),
  ("persian", "fa"),
  ("latvian", "lv"),
  ("bengali", "bn"),
  ("serbian", "sr"),
  ("azerbaijani", "az"),
  ("slovenian", "sl"),
  ("kannada", "kn"),
  ("estonian", "et"),
  ("macedonian", "mk"),
  ("breton", "br"),
  ("basque", "eu"),
  ("icelandic", "is"),
  ("armenian", "hy"),
  ("nepali", "ne"),
  ("mongolian", "mn"),
  ("bosnian", "bs"),
  ("kazakh", "kk"),
  ("albanian", "sq"),
  ("swahili", "sw"),
  ("galician", "gl"),
  ("marathi", "mr"),
  ("punjabi", "pa"),
  ("sinhala", "si"),
  ("khmer", "km"),
  ("shona", "sn"),
  ("yoruba", "yo"),
  ("somali", "so"),
  ("afrikaans", "af"),
  ("occitan", "oc"),
  ("georgian", "ka"),
  ("belarusian", "be"),
  ("tajik", "tg"),
  ("sindhi", "sd"),
  ("gujarati", "gu"),
  ("amharic", "am"),
  ("yiddish", "yi"),
  ("lao", "lo"),
  ("uzbek", "uz"),
  ("faroese", "fo"),
  ("haitian creole", "ht"),
  ("pashto", "ps"),
  ("turkmen", "tk"),
  ("nynorsk", "nn"),
  ("maltese", "mt"),
  ("sanskrit", "sa"),
  ("luxembourgish", "lb"),
  ("myanmar", "my"),
  ("tibetan", "bo"),
  ("tagalog", "tl"),
  ("malagasy", "mg"),
  ("assamese", "as"),
  ("tatar", "tt"),
  ("hawaiian", "haw"),
  ("lingala", "ln"),
  ("hausa", "ha"),
  ("bashkir", "ba"),
  ("javanese", "jw"),
  ("sundanese", "su"),
  ("cantonese", "yue"),
  ("burmese", "my"),
  ("valencian", "ca"),
  ("flemish", "nl"),
  ("haitian", "ht"),
  ("letzeburgesch", "lb"),
  ("pushto", "ps"),
  ("panjabi", "pa"),
  ("moldavian", "ro"),
  ("moldovan", "ro"),
  ("sinhalese", "si"),
  ("castilian", "es"),
  ("mandarin", "zh"),
];

#[cfg(test)]
mod tests;
