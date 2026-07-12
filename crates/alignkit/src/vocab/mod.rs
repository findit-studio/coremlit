//! Bridge between the chordai base960h CTC vocabulary and the
//! `tokenizers`-crate schema `Aligner::from_paths`-style construction will
//! need (design spec §3.1/§6,
//! `docs/superpowers/specs/2026-07-11-alignkit-forced-alignment-design.md`).
//!
//! `chordai/wav2vec2-base960h-aligner-coreml` ships a raw `{token: id}` CTC
//! dict (`Models/alignkit/base960h_dict.json`), not a HuggingFace
//! `tokenizer.json` — asry's `Aligner::from_paths` (and, later, alignkit's
//! own) both need the latter. This module owns the derived, committed
//! asset that fills that gap (`assets/chordai_base960h_tokenizer.json`)
//! plus the vocabulary constants later tasks validate a loaded tokenizer
//! against.
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
//! # Deferred
//!
//! This module exposes the asset and the constants derived from it — it
//! does not depend on the `tokenizers` crate outside of tests (a
//! dev-dependency only; see `Cargo.toml`) and does not itself construct a
//! `Tokenizer`. Parsing the asset into a live tokenizer, and the
//! `AlignerError` tokenizer-parse/vocab-mismatch variants that report on
//! that (design spec §8, flagged in [`crate::error`]'s module doc), land
//! with the concrete tokenizer type in a later task.

/// Number of entries in the chordai base960h CTC vocabulary, including the
/// blank and word-delimiter tokens.
///
/// Derived from `Models/alignkit/base960h_dict.json` (see this module's
/// `# Generator note`). asry's `validate_vocab_dim` requires the CTC head's
/// output width `V` to equal the tokenizer's vocab size EXACTLY;
/// `base960h_aligner.mlmodelc`'s pinned `emissions` contract is `[1, 2999,
/// 29]` (`tests/model_io.rs`), so this constant is also that model's
/// expected `V`.
pub const VOCAB_SIZE: usize = 29;

/// CTC blank-token id in the chordai base960h vocabulary.
///
/// The dict maps the literal token `"-"` to id `0` — chordai's own CTC
/// blank convention. This is distinct from the `<pad>` / `[PAD]` /
/// `<blank>` special-token probe asry's `detect_blank_token_id`
/// (`asry/src/runner/aligner/aligner.rs:1090`) performs by default: this
/// vocabulary has no `<pad>`-style entry at all, only the bare `"-"` at id
/// `0`. alignkit's own `Aligner` construction (a later task) uses this
/// constant directly rather than that dynamic probe.
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
/// that function's own error-message text saying so
/// (`asry/src/runner/aligner/aligner.rs:1203`, `:1216`). A future
/// `Aligner::from_paths`-style constructor that wants this bundled default
/// can feed these bytes straight into that same `Tokenizer::from_bytes`
/// call with no filesystem round-trip; the reverse direction (materialize
/// a temp file from bytes, for an API that insists on `impl AsRef<Path>`)
/// is trivial on demand, while the forward direction (a baked-in path
/// surviving repackaging) is not achievable at all.
pub const fn tokenizer_json_bytes() -> &'static [u8] {
  include_bytes!("../../assets/chordai_base960h_tokenizer.json")
}

#[cfg(test)]
mod tests;
