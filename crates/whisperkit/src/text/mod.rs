//! Text utilities: zlib compression-ratio repetition signal, string
//! normalization, and the streaming word-timing prefix/suffix helpers.
//! Ports `Utilities/TextUtilities.swift`
//! `TextUtilities.compressionRatio(of:)` (both overloads),
//! `Utilities/Extensions+Public.swift` `String.normalized`/
//! `String.trimmingSpecialTokenCharacters()`, and two functions from
//! `Utilities/TranscriptionUtilities.swift`: `findLongestCommonPrefix`/
//! `findLongestDifferentSuffix`.
//!
//! `Array.batched` (`Core/WhisperKit.swift:739`, concurrent-worker audio
//! batching) is unrelated to the above and stays out of scope: Plan 3 uses
//! `slice::chunks` directly wherever batching is needed, and this sync,
//! single-threaded port has no concurrent-worker fan-out to batch for.

use objc2::rc::autoreleasepool;
use objc2_foundation::{NSData, NSDataCompressionAlgorithm};
use unicode_categories::UnicodeCategories;

use crate::result::WordTiming;

/// zlib-compresses `bytes` with Apple's libcompression — the exact
/// `NSData.compressed(using: .zlib)` API Swift WhisperKit's
/// `TextUtilities.compressionRatio` calls
/// (`Utilities/TextUtilities.swift:14-28,33-53`) — and returns the
/// compressed byte length. `None` mirrors Swift's `catch { return
/// .infinity }`: on a genuine OS compression error the caller substitutes
/// [`f32::INFINITY`] rather than `unwrap`-panicking.
///
/// # Why Apple's compressor, not `flate2`/`miniz_oxide`
/// This length feeds the temperature-fallback repetition signal
/// ([`compression_ratio_of_tokens`] -> `decode::finalize_decoding_result`),
/// whose fallback *decision* must match Swift byte-for-byte (coremlit
/// issue #9). Apple's `.zlib` algorithm emits **raw DEFLATE (RFC 1951)** —
/// no zlib wrapper (its output begins e.g. `db c3 …`, not `78 …`) — and
/// compresses markedly harder than `miniz_oxide` on repetitive input.
/// `flate2::write::ZlibEncoder` emits RFC 1950 (a 2-byte header + 4-byte
/// Adler-32 trailer, +6 bytes) and saturates at a weaker ratio; both gaps
/// push our ratio *below* Swift's, flipping the fallback decision at
/// realistic thresholds. No `flate2` compression level reproduces Apple's
/// lengths, so this crate calls the identical Foundation API to get a
/// ratio equal to Swift's by construction. This is a safe `objc2` call —
/// `objc2` owns the FFI.
///
/// The [`autoreleasepool`] matches
/// [`crate::tokenizer::nl_recognizer::redetect_language`]'s rationale: any
/// Objective-C method may autorelease internally, so a pool must sit on the
/// stack or temporaries leak on a thread with no Cocoa run-loop pool above
/// it. Empty input needs no special-casing: Apple's libcompression
/// compresses a zero-length buffer to a small non-empty result (2 bytes,
/// per the issue-9 objc2 probe), never throwing, so this returns `Some(2)`
/// for `&[]`. `None` is reserved for a genuine OS compression error — the
/// only case Swift's `catch` actually handles.
fn zlib_compressed_len(bytes: &[u8]) -> Option<usize> {
  autoreleasepool(|_| {
    NSData::with_bytes(bytes)
      .compressedDataUsingAlgorithm_error(NSDataCompressionAlgorithm::Zlib)
      .ok()
      .map(|compressed| compressed.len())
  })
}

/// Compression ratio (`raw_bytes / compressed_bytes`) of `tokens`, encoded
/// as **little-endian `i32`** before zlib compression — ports Swift
/// `TextUtilities.compressionRatio(of textTokens: [Int])`
/// (`Utilities/TextUtilities.swift:14-28`): `Int32($0)` per token packed
/// into a `Data` buffer (platform-native byte order, little-endian on
/// every Apple target), then `(data as NSData).compressed(using: .zlib)`.
///
/// An empty `tokens` slice returns **`0.0`**, matching Swift's tokens
/// overload exactly: that overload has **no empty guard**
/// (`Utilities/TextUtilities.swift:14-28`), so it compresses an empty
/// `Data()`, which Apple's libcompression turns into 2 bytes (not an
/// error) — giving `0 / 2 == 0.0`. (Contrast [`compression_ratio_of_text`],
/// which Swift *does* guard.) Only a genuine OS compression error yields
/// [`f32::INFINITY`] here, mirroring Swift's `catch { return .infinity }` —
/// and Swift's tokens overload would land in that identical `catch` on the
/// same error, so the error path matches on both sides too.
///
/// The compression itself goes through the private `zlib_compressed_len`,
/// which calls the same Apple `NSData.compressed(using: .zlib)` API as
/// Swift — see that function for why the codec, not just the byte
/// encoding, must match Swift (coremlit issue #9). The `i32`-LE token
/// encoding below is unchanged: it already matched Apple; only the
/// compressor differed.
pub fn compression_ratio_of_tokens(tokens: &[u32]) -> f32 {
  let bytes: Vec<u8> = tokens
    .iter()
    .flat_map(|t| (*t as i32).to_le_bytes())
    .collect();
  zlib_compressed_len(&bytes).map_or(f32::INFINITY, |len| bytes.len() as f32 / len as f32)
}

/// Compression ratio (`raw_bytes / compressed_bytes`) of `text`'s UTF-8
/// bytes — ports Swift `TextUtilities.compressionRatio(of text: String)`
/// (`Utilities/TextUtilities.swift:33-53`). Returns [`f32::INFINITY`] for
/// empty text via Swift's explicit `if text.isEmpty { return .infinity }`
/// guard (lines 34-36), or on a genuine compression error (Swift's
/// `catch`, lines 49-51). Swift also guards a fallible `text.data(using:
/// .utf8)` (lines 39-42); that path is unreachable here since a Rust
/// `&str` is always valid UTF-8.
///
/// **Empty-input asymmetry (deliberate, matches Swift).** Unlike
/// [`compression_ratio_of_tokens`] — whose Swift overload has *no* empty
/// guard and so returns `0.0` for empty input — this text overload *does*
/// guard empty and returns infinity. The two Swift overloads diverge here
/// on purpose (`:34-36` guards text; `:14-28` does not guard tokens); keep
/// this guard.
///
/// Compresses via the private `zlib_compressed_len` (Apple's `.zlib` = raw
/// DEFLATE), identical to Swift; see [`compression_ratio_of_tokens`].
pub fn compression_ratio_of_text(text: &str) -> f32 {
  if text.is_empty() {
    return f32::INFINITY;
  }
  let bytes = text.as_bytes();
  zlib_compressed_len(bytes).map_or(f32::INFINITY, |len| bytes.len() as f32 / len as f32)
}

/// Normalizes `text` for repetition/equality comparisons — ports Swift
/// `String.normalized` (`Utilities/Extensions+Public.swift:24-41`)
/// **exactly**, verified empirically by running the live Swift extension
/// standalone (see this task's report), not just by reading it: lowercase
/// the whole string, then replace the literal ASCII `-` character with a
/// space (a plain, non-regex substring replace — no other dash variant is
/// touched by this step), then **delete** (not replace) every character
/// in Unicode general category `P` (Punctuation: `Pc`/`Pd`/`Pe`/`Pf`/`Pi`/
/// `Po`/`Ps`) — this is the step that removes `_` (category `Pc`) and
/// every non-ASCII dash/quote/CJK punctuation mark, matching Foundation's
/// `CharacterSet.punctuationCharacters` — then collapse runs of the
/// literal space character down to one space, then trim
/// [`char::is_whitespace`] from both ends (this matches Foundation's
/// `.whitespacesAndNewlines`: both sets are exactly U+0009-U+000D,
/// U+0020, U+0085, U+00A0, U+1680, U+2000-U+200A, U+2028, U+2029, U+202F,
/// U+205F, U+3000).
///
/// **Deviation from this task's own brief** (source-corrected, per this
/// task's explicit mandate to verify against source): the brief's
/// semantics sketch ("dashes/underscores to spaces") is wrong for
/// underscores. Swift's dash step is a literal, non-regex
/// `replacingOccurrences(of: "-", with: " ")` that matches only the ASCII
/// hyphen; `_` is punctuation category `Pc` and gets *deleted* by the
/// punctuation step, not turned into a space. Confirmed by running the
/// actual Swift extension standalone: `"multi-word_test".normalized ==
/// "multi wordtest"`, not `"multi word test"` as the brief's own given
/// test asserted — see the task report for the probe script and full
/// output; this module's tests reflect the verified behavior.
pub fn normalized(text: &str) -> String {
  let lowercased = text.to_lowercase();
  let no_dashes = lowercased.replace('-', " ");
  let no_punctuation: String = no_dashes.chars().filter(|c| !c.is_punctuation()).collect();

  let mut collapsed = String::with_capacity(no_punctuation.len());
  let mut last_was_space = false;
  for c in no_punctuation.chars() {
    if c == ' ' {
      if !last_was_space {
        collapsed.push(' ');
      }
      last_was_space = true;
    } else {
      collapsed.push(c);
      last_was_space = false;
    }
  }

  collapsed.trim().to_string()
}

/// Strips leading/trailing `<`/`|`/`>` characters, repeatedly, from both
/// ends — ports Swift `String.trimmingSpecialTokenCharacters()`
/// (`Utilities/Extensions+Public.swift:43-45`), which trims
/// `Constants.specialTokenCharacters` (`Core/Models.swift:1332`,
/// `CharacterSet(charactersIn: "<|>")`). This is a **character-class**
/// trim, not a fixed `"<|"`/`"|>"` substring strip: e.g. `"<<|x|>"` trims
/// to `"x"` (every wrapping character is a member of the set), not
/// `"<<|x"` (which a literal-prefix reading of this function's own
/// summary would wrongly produce, since `"<<|x|>"` does not start with
/// the literal two-character substring `"<|"`).
pub fn trim_special_token_chars(text: &str) -> &str {
  text.trim_matches(['<', '|', '>'])
}

/// Longest run of word-by-word agreement between two decode passes over
/// the same audio span, comparing [`normalized`] text — ports
/// `TranscriptionUtilities.findLongestCommonPrefix`
/// (`Utilities/TranscriptionUtilities.swift:34-37`): `zip(words1,
/// words2).prefix(while: { $0.word.normalized == $1.word.normalized })`,
/// mapped to `$0.1`, i.e. the **returned elements come from `current`**,
/// the newer pass, not `previous`. A borrowed prefix of `current` replaces
/// Swift's array copy — identical contents, zero allocation. Stops at the
/// shorter of the two inputs, exactly like `zip`.
pub fn find_longest_common_prefix<'a>(
  previous: &[WordTiming],
  current: &'a [WordTiming],
) -> &'a [WordTiming] {
  let agreed = previous
    .iter()
    .zip(current)
    .take_while(|(a, b)| normalized(a.word()) == normalized(b.word()))
    .count();
  &current[..agreed]
}

/// `current` past its agreement with `previous` — ports
/// `TranscriptionUtilities.findLongestDifferentSuffix`
/// (`Utilities/TranscriptionUtilities.swift:44-48`): `words2[commonPrefix.
/// count...]`. When `previous` and `current` share no common prefix at
/// all, the whole of `current` is returned.
pub fn find_longest_different_suffix<'a>(
  previous: &[WordTiming],
  current: &'a [WordTiming],
) -> &'a [WordTiming] {
  &current[find_longest_common_prefix(previous, current).len()..]
}

#[cfg(test)]
mod tests;
