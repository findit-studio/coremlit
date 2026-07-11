//! Text utilities: zlib compression-ratio repetition signal + string
//! normalization. Ports `Utilities/TextUtilities.swift`
//! `TextUtilities.compressionRatio(of:)` (both overloads) and
//! `Utilities/Extensions+Public.swift` `String.normalized`/
//! `String.trimmingSpecialTokenCharacters()`.
//!
//! Streaming-decode helpers (longest-common-prefix word-timing merge,
//! `batched`) are explicitly out of scope here — deferred to Plan 4
//! (streaming) per this task's brief; Plan 3 uses `slice::chunks` directly
//! wherever batching is needed instead.

use std::io::Write;

use flate2::{Compression, write::ZlibEncoder};
use unicode_categories::UnicodeCategories;

/// zlib-compresses `bytes` at [`Compression::default`], returning the
/// compressed byte length. `Err` only if the in-memory `Vec<u8>` sink's
/// `Write` impl fails, which does not happen in practice — kept
/// `Result`-typed so callers can fall back to [`f32::INFINITY`] the same
/// way Swift's `catch` does, rather than `unwrap`-panicking on a
/// theoretical error.
fn zlib_compressed_len(bytes: &[u8]) -> std::io::Result<usize> {
  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  encoder.write_all(bytes)?;
  Ok(encoder.finish()?.len())
}

/// Compression ratio (`raw_bytes / compressed_bytes`) of `tokens`, encoded
/// as **little-endian `i32`** before zlib compression — ports Swift
/// `TextUtilities.compressionRatio(of textTokens: [Int])`
/// (`Utilities/TextUtilities.swift:14-28`): `Int32($0)` per token packed
/// into a `Data` buffer (platform-native byte order, little-endian on
/// every Apple target), then `(data as NSData).compressed(using: .zlib)`.
///
/// Returns [`f32::INFINITY`] for an empty `tokens` slice (Swift's
/// zero-length `Data` compression attempt fails there, landing in its
/// `catch` block) or on a genuine compression error (mirrors Swift's
/// `catch { return .infinity }`).
///
/// Uses [`Compression::default`] (zlib level 6). Apple's
/// `NSData.compressed(using: .zlib)` compression level is fixed by the OS
/// and undocumented; if the golden-parity harness (Plan 3) shows
/// threshold-crossing differences against real WhisperKit output, revisit
/// the level there — not here.
pub fn compression_ratio_of_tokens(tokens: &[u32]) -> f32 {
  if tokens.is_empty() {
    return f32::INFINITY;
  }
  let bytes: Vec<u8> = tokens
    .iter()
    .flat_map(|t| (*t as i32).to_le_bytes())
    .collect();
  zlib_compressed_len(&bytes).map_or(f32::INFINITY, |len| bytes.len() as f32 / len as f32)
}

/// Compression ratio (`raw_bytes / compressed_bytes`) of `text`'s UTF-8
/// bytes — ports Swift `TextUtilities.compressionRatio(of text: String)`
/// (`Utilities/TextUtilities.swift:33-53`). Returns [`f32::INFINITY`] for
/// empty text (Swift's explicit `if text.isEmpty` guard, lines 34-36) or
/// on a genuine compression error (Swift's `catch`, lines 49-51). Swift
/// also guards a fallible `text.data(using: .utf8)` (lines 39-42); that
/// path is unreachable here since a Rust `&str` is always valid UTF-8.
///
/// See [`compression_ratio_of_tokens`] for the zlib-compression-level
/// caveat.
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

#[cfg(test)]
mod tests;
