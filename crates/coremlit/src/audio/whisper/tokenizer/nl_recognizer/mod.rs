//! Optional transcript-based language redetection via Apple's
//! `NLLanguageRecognizer`, gated behind the `nl-recognizer` feature (off
//! by default). See [`redetect_language`]'s doc for the trade-off this
//! exists to offer callers, and
//! [`super::WhisperTokenizer::split_to_word_tokens`]'s doc for why the
//! pipeline itself never calls this automatically (coremlit issue #9).

use objc2::rc::autoreleasepool;
use objc2_foundation::NSString;
use objc2_natural_language::NLLanguageRecognizer;

#[cfg(test)]
mod tests;

/// Redetects `text`'s dominant language from its own decoded content via
/// `NLLanguageRecognizer` — the same class Swift's WhisperKit uses
/// internally (`Models.swift:1297-1299`) — normalizing the raw BCP-47
/// result to a Whisper base language code before returning it.
///
/// # Why this exists
/// [`super::WhisperTokenizer::split_to_word_tokens`] always splits words
/// using the language the Whisper *decoder* already committed to (its
/// `<|lang|>` prompt token), not a second opinion from the decoded text —
/// that stays the pipeline's one source of truth (see that method's doc).
/// That default is right for single-language audio, but code-switched
/// audio (a clip whose dominant spoken language drifts mid-utterance) can
/// benefit from re-checking the language against what actually got
/// transcribed. This function is that opt-in second opinion: a caller who
/// wants it computes its own replacement language code and passes that
/// into `split_to_word_tokens` instead of the decoder's token. The
/// library does not wire this in by default, for two reasons: a
/// text-based re-detection can be wrong in ways the decoder's own
/// language token isn't (short strings, mixed-script strings, and proper
/// nouns are all documented `NLLanguageRecognizer` failure modes), and
/// doing it unconditionally would silently reintroduce a second,
/// competing language signal into every call.
///
/// # The normalization Swift's own call site lacks
/// `NLLanguageRecognizer.dominantLanguage` returns a full BCP-47 tag, not
/// a bare base code: for Traditional Chinese text it returns `zh-Hant`,
/// not `zh` (verified empirically against this exact crate version).
/// Swift's WhisperKit passes that raw tag straight into its own CJK
/// allowlist check (`Models.swift:1301`), which only matches the bare
/// codes `zh`/`ja`/`th`/`lo`/`my`/`yue` — so `zh-Hant` misses the list and
/// Swift falls through to space-based (phrase-blob) splitting for
/// Traditional Chinese (coremlit issue #9's root-caused finding). This
/// function normalizes before returning specifically so it cannot
/// reproduce that gap: every result is reduced to its primary BCP-47
/// subtag (`zh-Hant`/`zh-Hans`/`zh-*` all become `zh`), with ISO 639-2's
/// `cmn` additionally mapped to `zh` (Whisper's vocabulary has no separate
/// `cmn` token; `NLLanguageRecognizer` has not been observed to emit it,
/// but the mapping costs nothing).
///
/// Returns `None` when `NLLanguageRecognizer` cannot determine a dominant
/// language (notably: empty or extremely short input).
///
/// # Thread safety
/// This function establishes its own autorelease pool
/// ([`autoreleasepool`]) around the complete operation — including the
/// final owned-`String` conversion — so it is safe to call from any
/// thread, including a bare `std::thread` with no surrounding Cocoa/
/// AppKit run-loop pool. The pool is required, not optional: objc2's own
/// documented contract is that any Objective-C method may autorelease
/// internally (`objc2` 0.6.4 `src/rc/autorelease.rs:316`,
/// [`autoreleasepool`]'s own doc), and the final `NSString` -> `String`
/// conversion below specifically goes through `objc2-foundation`'s
/// `autoreleasepool_leaking` (`objc2-foundation` 0.3.2 `src/util.rs:46`),
/// which — per its own SAFETY comment (`objc2`
/// `src/rc/autorelease.rs:524-533`) — *assumes* a real pool exists
/// further up the call stack rather than establishing one itself. Without
/// the pool this function pushes, every temporary autoreleased while
/// redetecting a language would leak instead of draining on a thread
/// with no pool above it.
pub fn redetect_language(text: &str) -> Option<String> {
  autoreleasepool(|_| {
    // SAFETY: fresh, unshared recognizer instance; `new` has no
    // preconditions beyond ordinary Objective-C allocation.
    let recognizer = unsafe { NLLanguageRecognizer::new() };
    let ns_text = NSString::from_str(text);
    // SAFETY: `recognizer` and `ns_text` are both live for the call, and
    // the recognizer is not shared across threads.
    unsafe { recognizer.processString(&ns_text) };
    // SAFETY: accessor send on the same live recognizer.
    let language = unsafe { recognizer.dominantLanguage() }?;
    Some(normalize_bcp47(&language.to_string()))
  })
}

/// Reduces a BCP-47 language tag to the primary subtag Whisper's
/// `<|lang|>` vocabulary and
/// [`super::WhisperTokenizer::split_to_word_tokens`] expect: text before
/// the first `-`, lowercased (`zh-Hant` -> `zh`, `zh-Hans` -> `zh`,
/// `pt-BR` -> `pt`, a bare `en` -> `en` unchanged), with ISO 639-2's `cmn`
/// (Mandarin) additionally mapped to `zh` since the primary-subtag rule
/// alone does not reduce it.
fn normalize_bcp47(tag: &str) -> String {
  let primary = tag.split('-').next().unwrap_or(tag).to_ascii_lowercase();
  if primary == "cmn" {
    "zh".to_string()
  } else {
    primary
  }
}
