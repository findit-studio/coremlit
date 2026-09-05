//! Native CoreML **granite** text embeddings — a general sentence-embedding
//! surface whose first model is IBM's
//! `granite-embedding-97m-multilingual-r2` (a ModernBERT encoder with CLS
//! pooling projecting to a 384-dim space).
//!
//! A `&str` in, a unit-norm 384-dim [`Embedding`] out ([`TextEmbedder::embed`]):
//! the artifact's granite tokenizer around the fp16 CoreML ModernBERT graph, with
//! L2 normalization applied in Rust.
//!
//! Design spec: `docs/superpowers/specs/2026-07-18-embedkit-design.md`
//! (Amendment 3: granite confirmed, `coremlit::embeddings::granite`, prompt-free,
//! committed-golden oracle).
//!
//! macOS only (built on [`crate`]).
//!
//! # Prompt-free (raw strings)
//!
//! granite-embedding r2 retrieval is **prompt-free**: its
//! `config_sentence_transformers.json` query/document prompts are empty. Feed
//! **raw strings** — no task prefixes. (This is the model's documented retrieval
//! contract; it differs from instruction-tuned embedders.)
//!
//! # Model artifacts
//!
//! The CoreML graph is distributed on the Hugging Face Hub at
//! [`FinDIT-Studio/embedkit-coreml`](https://huggingface.co/FinDIT-Studio/embedkit-coreml),
//! revision `a61241cb`, converted from
//! [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
//! (**Apache-2.0**; see the crate `NOTICE`). It is a gitignored dev-time
//! download under `Models/embedkit-granite/`; its per-file SHA-256 and I/O
//! contract are pinned by `tests/granite/model_io.rs`.
//!
//! # Rust front-end around an fp16 CoreML graph
//!
//! The graph emits the **pre-normalization** CLS embedding (`hidden_states[:,
//! 0]` after the final LayerNorm); this module applies the final L2
//! normalization in Rust (keeping the fp16 rsqrt-guard class out of the graph,
//! the workspace convention). The graph takes tokenized `input_ids` /
//! `attention_mask` (`[1, 512]` int32), produced from the granite
//! `tokenizer.json` that ships in the artifact directory (see
//! [`TOKENIZER_FILE_NAME`]).
//!
//! # Committed-golden oracle (no ort)
//!
//! Parity is scored against **committed transformers-fp32 fixtures**
//! (`tests/granite/fixtures/goldens/corpus.json`), never a live ONNX crate — the
//! embedkit "no ort anywhere, not even dev" rule. The hermetic
//! `tests/granite/tokenizer_identity.rs` proves the artifact tokenizer is
//! byte-correct (token-ids match the goldens exactly, no model load needed);
//! `tests/granite/parity_embed.rs` scores the CoreML embeddings against the
//! fp32 goldens by cosine (model-gated).
//!
//! # Compute placement (measured, never marketed)
//!
//! Placement is characterized, not asserted (`tests/granite/placement.rs`).
//! Unlike CLAP's audio tower, the granite ModernBERT graph **does** compile for
//! the ANE: T1's ANECCompile accepted it and the `CPU_AND_NE` compile plan
//! reported 482/493 ops (97.8%) preferring the ANE — a planner/compile-eligibility
//! report, not measured runtime residency. fp16 under `CpuAndNeuralEngine` scored
//! worst cosine 0.99996 vs the fp32 reference build. [`crate::ComputeUnits::All`]
//! (the default) lets CoreML schedule it — on T1's Mac the planner chose the GPU
//! for this small graph; `CpuAndNeuralEngine` targets the ANE. The module
//! characterizes the placement rather than claiming it.

pub mod embedding;
pub mod error;

mod token_index;

pub use embedding::Embedding;
pub use error::{
  ArtifactTokenizerRead, ContentlessInputOverBudget, ContractMismatch, EmbeddingDimMismatch, Error,
  InputTooLarge, OutputShape, SpecialTokenOverhead, TokenCount, TokenizerContractMismatch,
  WindowOverBudget,
};

/// windit's window geometry, re-exported as one of the two windit types in
/// granite's public surface: the per-chunk token budget, overlap, tail policy,
/// and window cap. Carried by [`LongTextOptions`] (alongside granite's own
/// `max_input_bytes` bound), the options [`TextEmbedder::embed_long_with`]
/// accepts.
pub use windit::plan::WindowOptions;

/// The tail policy carried by [`WindowOptions`], re-exported so a caller can
/// name it — set it, read it back, round-trip a persisted geometry — without a
/// direct `windit` dependency of their own.
///
/// From windit's crate root, which is where 0.4 lifted it precisely because it
/// was otherwise unnameable through a re-exporting consumer.
///
/// On granite's own path it moves a boundary and never drops text: see
/// [`LongTextOptions::tail_policy`] for what each variant does — the same
/// geometry, and the same tail behavior, governs both `embed_long` and
/// `embed_windows`.
pub use windit::TailPolicy;

use std::{ops::Range, path::Path, sync::OnceLock};

use crate::{ComputeUnits, DataType, Model, MultiArray};
use tokenizers::{
  PostProcessor, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

use crate::embeddings::granite::{
  embedding::{EMBEDDING_DIM, check_finite_output},
  error::Result,
  token_index::{IndexMeasure, LazyTable, MergeTable, TokenIndex},
};

/// File name of the granite `tokenizer.json` sidecar inside the model artifact
/// directory — the file [`TextEmbedder::load`] / [`TextEmbedder::from_file`]
/// read from the directory *containing* the `.mlmodelc`.
///
/// The tokenizer is the one from the source model repo
/// [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2),
/// revision `835ad14087e140460703cf0fae09f97d469d65c2` (SHA-256
/// `4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f`) — the
/// exact tokenizer that produced the committed token-id goldens. It is
/// distributed with the CoreML graph at
/// [`FinDIT-Studio/embedkit-coreml`](https://huggingface.co/FinDIT-Studio/embedkit-coreml)
/// rather than compiled into this crate, and the bytes read from disk are held
/// to the same fail-closed identity pin a caller-supplied tokenizer is
/// ([`TextEmbedder::from_memory`]).
pub const TOKENIZER_FILE_NAME: &str = "tokenizer.json";

/// Declared feature names on the granite `.mlmodelc` (pinned by
/// `tests/granite/model_io.rs`).
mod names {
  pub const INPUT_IDS: &str = "input_ids";
  pub const ATTENTION_MASK: &str = "attention_mask";
  pub const EMBEDDING: &str = "embedding";
}

/// The Granite tokenizer/model contract, verified against the artifact tokenizer and
/// the committed goldens: the total vocabulary INCLUDING added tokens, the
/// highest id the model's embedding table can gather, the special tokens
/// [`TextEmbedder::token_ids`] brackets every sequence with, and one pinned
/// sentinel encoding. [`validate_tokenizer_contract`] checks every constructor's
/// tokenizer against these, fail-closed.
mod contract {
  /// Total vocabulary size (base + added tokens) `get_vocab_size(true)` reports.
  pub const VOCAB_SIZE: usize = 180_000;
  /// Highest token id the model's embedding table can gather; an id past this
  /// indexes outside the table and gathers zeros.
  pub const MAX_TOKEN_ID: u32 = 179_999;
  /// CLS / start-of-text special, pooled at position 0.
  pub const CLS_TOKEN: &str = "<|startoftext|>";
  pub const CLS_ID: u32 = 179_934;
  /// Padding special (also the fixed-window pad id).
  pub const PAD_TOKEN: &str = "<|endoftext|>";
  pub const PAD_ID: u32 = 179_935;
  /// End-of-sequence special.
  pub const EOS_TOKEN: &str = "<|return|>";
  pub const EOS_ID: u32 = 179_938;
  /// Pinned sentinel: `SENTINEL_TEXT` encodes to `SENTINEL_IDS` (special tokens
  /// included) — the same pin `token_ids_match_pinned_golden_subset` asserts.
  pub const SENTINEL_TEXT: &str = "hello world";
  pub const SENTINEL_IDS: [u32; 4] = [CLS_ID, 24_313, 2_318, EOS_ID];
  /// SHA-256 (lowercase hex) of the pinned granite `tokenizer.json` artifact —
  /// the byte identity EVERY tokenizer must match, whether it was read from the
  /// model artifact directory (`TextEmbedder::load`) or supplied by the caller
  /// (`TextEmbedder::from_memory` / `from_files`); the
  /// `validate_tokenizer_identity` backstop. Tied to the golden-source SHA
  /// literal by `tokenizer_sha_pin_matches_golden_source_literal`, and to the
  /// artifact bytes by `artifact_tokenizer_sha_matches_golden_source_pin`.
  pub const TOKENIZER_SHA256_HEX: &str =
    "4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f";
}

/// Fixed token-sequence length the ModernBERT graph was converted at (the
/// export sequence length, `[1, 512]`). Shorter inputs are right-padded to this
/// length with the mask zeroed on the pad positions; longer inputs are truncated
/// at this length. RoPE makes any fixed length sound, and CLS pooling reads
/// position 0 (never a pad), so the pad token value never reaches the output.
pub const MAX_TOKENS: usize = 512;

/// Special tokens the pinned tokenizer's post-processor adds to EVERY sequence:
/// `2`, the `<|startoftext|> A <|return|>` single-sequence template of the
/// artifact `tokenizer.json` ([`TOKENIZER_FILE_NAME`]).
///
/// Every encoding on this door's paths runs `encode(s, add_special_tokens =
/// true)` — [`TextEmbedder::token_ids`], the chunker's measurement, and the
/// sentinel gate alike — so this overhead is charged against [`MAX_TOKENS`]
/// unconditionally, leaving [`CONTENT_TOKENS_PER_WINDOW`] for the caller's own
/// text.
///
/// Not a knob: the tokenizer is pinned by SHA-256, so this is a fact about the
/// one artifact this door accepts, re-measured from it rather than asserted —
/// `special_token_overhead_matches_the_pinned_template` (model-gated) reads it
/// three independent ways: the post-processor's own `added_tokens(false)`, the
/// length of `encode("", true)`, and the difference an `add_special_tokens`
/// makes to one encoding.
pub const SPECIAL_TOKENS_PER_WINDOW: usize = 2;

/// Raw-content token budget of one window: [`MAX_TOKENS`] −
/// [`SPECIAL_TOKENS_PER_WINDOW`] = `510`.
///
/// The most tokens of the CALLER'S OWN text that fit one prediction — and the
/// tokenizer's own effective text window, since this module configures
/// truncation at [`MAX_TOKENS`] and `tokenizers` truncates at `max_length −
/// post_processor.added_tokens(false)`.
///
/// [`LongTextOptions::window_options`] is stated in TOTAL tokens, not in these:
/// `WindowOptions::window()` is compared against a measure that counts the
/// specials too (chunk measurement and per-chunk embedding run the same
/// `encode(s, true)`), so a `window()` of `w` admits
/// `w − `[`SPECIAL_TOKENS_PER_WINDOW`] content tokens and the default
/// `window() == MAX_TOKENS` admits exactly this many. Set `window()` as a total;
/// read the content budget off this constant.
pub const CONTENT_TOKENS_PER_WINDOW: usize = MAX_TOKENS - SPECIAL_TOKENS_PER_WINDOW;

/// Default [`TextEmbedderOptions::compute`]: [`ComputeUnits::All`]. The granite
/// ModernBERT graph is ANE-capable (T1's `CPU_AND_NE` compile plan: 97.8% of ops
/// ANE-preferred); `All` lets CoreML schedule it — T1 saw the planner pick the
/// GPU on Macs for this small graph; `CpuAndNeuralEngine` targets the ANE.
/// Placement is characterized, not asserted (`tests/granite/placement.rs`).
pub const DEFAULT_COMPUTE: ComputeUnits = ComputeUnits::All;

#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_COMPUTE
}

/// Construction options for [`TextEmbedder`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextEmbedderOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_compute"))]
  compute: ComputeUnits,
}

impl Default for TextEmbedderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl TextEmbedderOptions {
  /// Options matching the module default: [`DEFAULT_COMPUTE`].
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the graph on.
  #[inline]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }

  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }

  /// Sets [`Self::compute`] in place.
  #[inline]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// Options for [`TextEmbedder::embed_long_with`] and
/// [`TextEmbedder::embed_windows_with`] (rust-options-pattern) — both share this
/// one type: windit's chunk geometry ([`WindowOptions`]) plus granite's
/// pre-tokenization input bound.
///
/// The geometry is reachable whole ([`Self::window_options`]) and, for the tail
/// policy, one field at a time ([`Self::tail_policy`]). Neither requires naming
/// `windit`: [`WindowOptions`] and [`TailPolicy`] are both re-exported by this
/// module.
///
/// # Wire form
///
/// Serializable under the `serde` feature, which is what coremlit's `serde`
/// turning on `windit?/serde` buys — this type composes windit's
/// [`WindowOptions`], so it could not be derived while that crate carried no
/// impls. The geometry is windit's own document, nested under
/// `window_options`:
///
/// ```json
/// {"window_options":{"window":512,"hop":512,"tail":{"kind":"keep_with_coverage"},
///  "max_windows":null},"max_input_bytes":null}
/// ```
///
/// Both fields carry defaults, so a partial config fills the rest from
/// [`Self::new`] — `{}` deserializes to exactly it, the same convention
/// [`TextEmbedderOptions`] and the doors' `WindowPlan`s follow.
///
/// UNKNOWN KEYS ARE REFUSED, at this level and (windit's own rule) inside
/// `window_options`. Defaulted fields and a tolerated stray key compose into a
/// silent hole: `{"max_input_byte":4096}` — the plural dropped — would
/// otherwise deserialize to `max_input_bytes: None`, and
/// [`TextEmbedder::embed_long_with`] (or, sharing this same options type,
/// [`TextEmbedder::embed_windows_with`]) would then run with NO size gate on
/// its input, which is the one bound a caller sets for untrusted text. The
/// misspelling is a hard error naming the key instead.
///
/// That refusal makes this type UNFLATTENABLE: serde's `deny_unknown_fields`
/// and `flatten` do not compose (a flattened field sees the outer struct's
/// other keys and rejects them), so a config type composing these options must
/// NEST them under a key of its own — `long_text = { … }` — not
/// `#[serde(flatten)]` them into itself.
///
/// # Binary formats
///
/// The document above is a human-readable one; this type ALSO round-trips
/// through a non-self-describing format (postcard and friends), under every
/// tail policy. That is windit's contract, not coremlit's — the nested geometry
/// is windit's own type — and it holds only from windit 0.5: 0.4 tagged
/// `TailPolicy` adjacently through the derive, which writes the tag as a struct
/// FIELD and reads it back through `deserialize_identifier`, so a format
/// carrying no field names refused EVERY variant and a `LongTextOptions`
/// serialized to bytes it could not read back. windit 0.5 asks for the adjacent
/// shape itself (a `deserialize_struct` visitor serving both `visit_map` and
/// `visit_seq`), which leaves the document above byte-identical while making the
/// compact form readable. `long_text_options_round_trip_through_postcard` pins
/// it here, variant by variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LongTextOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_window_options"))]
  window_options: WindowOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  max_input_bytes: Option<usize>,
}

/// The [`LongTextOptions::window_options`] serde default: the same full-window
/// geometry [`LongTextOptions::new`] builds, so an omitted key and an omitted
/// options value agree.
#[cfg(feature = "serde")]
fn default_window_options() -> WindowOptions {
  WindowOptions::new(MAX_TOKENS)
}

impl Default for LongTextOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl From<WindowOptions> for LongTextOptions {
  /// Geometry-only options: the given windit geometry, no input byte limit.
  fn from(window_options: WindowOptions) -> Self {
    Self {
      window_options,
      max_input_bytes: None,
    }
  }
}

impl LongTextOptions {
  /// Options matching [`TextEmbedder::embed_long`] and
  /// [`TextEmbedder::embed_windows`] alike: a full-window geometry
  /// (`WindowOptions::new(MAX_TOKENS)`) and no input byte limit.
  pub const fn new() -> Self {
    Self {
      window_options: WindowOptions::new(MAX_TOKENS),
      max_input_bytes: None,
    }
  }

  /// The windit chunk geometry (per-chunk token budget, overlap, window cap).
  #[inline]
  pub const fn window_options(&self) -> WindowOptions {
    self.window_options
  }

  /// Builder form of [`Self::set_window_options`].
  #[must_use]
  #[inline]
  pub const fn with_window_options(mut self, window_options: WindowOptions) -> Self {
    self.set_window_options(window_options);
    self
  }

  /// Sets [`Self::window_options`] in place.
  #[inline]
  pub const fn set_window_options(&mut self, window_options: WindowOptions) -> &mut Self {
    self.window_options = window_options;
    self
  }

  /// The [`TailPolicy`] carried by [`Self::window_options`] — what windit does
  /// with a final chunk that does not fill a whole window.
  ///
  /// **It moves a boundary on `embed_long` and `embed_windows` alike; it never
  /// drops text.** The two cases differ, and both are windit 0.5's:
  ///
  /// - [`TailPolicy::PadFull`] is a NAMED GAP in windit's `ContentAware`
  ///   chunker — there is nothing to pad a byte range with, so it keeps the tail
  ///   indistinguishably from [`TailPolicy::KeepWithCoverage`], the default.
  ///   Setting it changes nothing here.
  /// - [`TailPolicy::DropBelowMin`] IS honoured by `ContentAware` from windit
  ///   0.5 (0.4 read every other geometry field and skipped `tail`): windit
  ///   discards a final chunk whose measure is below the minimum unless it fills
  ///   a whole window. What reaches a caller — of `embed_long`, or of a window
  ///   from `embed_windows` — is not that drop, though: the
  ///   separator-reattachment repair behind both exists to cover every byte
  ///   windit leaves out, and the discarded tail is exactly such a gap. It
  ///   comes back as its own chunk, because a dropped tail is by construction
  ///   the content that would not fit its predecessor, so the repair can never
  ///   fuse it left. **Measured across a window × minimum grid: the chunk
  ///   COUNT is unchanged and coverage is unchanged; what moves is the last
  ///   boundary** —
  ///   the separator run between the final two chunks, absorbed rightwards into
  ///   the tail instead of leftwards into its predecessor (e.g. window 4, a
  ///   `\n\n` break: `[(0,8),(8,17),(17,20)]` becomes
  ///   `[(0,8),(8,15),(15,20)]`). The embedding of the last chunk therefore
  ///   changes; the number of CoreML predictions does not. For
  ///   [`TextEmbedder::embed_windows`], that changed last chunk is a moved
  ///   [`WindowEmbedding::byte_range`] on the last window — geometry a consumer
  ///   who persists spans alongside an index should expect to change under
  ///   `DropBelowMin`.
  ///
  /// One shape has no chunks to move: when the whole input is a single chunk
  /// below the minimum, windit returns NO chunks at all (its own documented
  /// consequence). [`TextEmbedder::embed_long`] and
  /// [`TextEmbedder::embed_windows`] still embed it — the non-empty fallback
  /// emits the whole input as one chunk, the same escape whitespace-only text
  /// already took — so this knob cannot make a non-empty text embed to
  /// nothing.
  #[inline]
  pub const fn tail_policy(&self) -> TailPolicy {
    *self.window_options.tail()
  }

  /// Builder form of [`Self::set_tail_policy`].
  #[must_use]
  #[inline]
  pub const fn with_tail_policy(mut self, tail_policy: TailPolicy) -> Self {
    self.set_tail_policy(tail_policy);
    self
  }

  /// Sets [`Self::tail_policy`] on the carried [`Self::window_options`] in
  /// place, leaving its window, hop and cap alone.
  ///
  /// `DropBelowMin` moves `embed_long`'s and `embed_windows`'s last chunk
  /// boundary (a moved `byte_range` on the last window), `PadFull` does
  /// nothing, and neither drops text — see [`Self::tail_policy`].
  #[inline]
  pub const fn set_tail_policy(&mut self, tail_policy: TailPolicy) -> &mut Self {
    self.window_options = self.window_options.with_tail(tail_policy);
    self
  }

  /// The maximum accepted input length in UTF-8 bytes, if any. `None` (the
  /// default) means unbounded. Enforced before any tokenizer or chunker work —
  /// the limit callers embedding UNTRUSTED text should set.
  #[inline]
  pub const fn max_input_bytes(&self) -> Option<usize> {
    self.max_input_bytes
  }

  /// Builder form of [`Self::set_max_input_bytes`].
  #[must_use]
  #[inline]
  pub const fn with_max_input_bytes(mut self, max_input_bytes: usize) -> Self {
    self.set_max_input_bytes(max_input_bytes);
    self
  }

  /// Sets [`Self::max_input_bytes`] in place (to `Some(max_input_bytes)`).
  #[inline]
  pub const fn set_max_input_bytes(&mut self, max_input_bytes: usize) -> &mut Self {
    self.max_input_bytes = Some(max_input_bytes);
    self
  }
}

/// One planned window of a long text and the embedding of exactly that window —
/// the element type of [`TextEmbedder::embed_windows`].
///
/// It carries what a consumer needs to score windows independently and attach
/// its own provenance to a hit, without re-deriving the geometry:
///
/// * [`ordinal`](Self::ordinal) — position in planning order, `0..n`. The
///   OCCURRENCE IDENTITY: two windows whose text is byte-identical (a repeated
///   paragraph) embed to the same vector, so nothing else distinguishes them.
/// * [`byte_start`](Self::byte_start) / [`byte_end`](Self::byte_end) — the
///   window's half-open range in UTF-8 bytes of the `text` that was passed in,
///   `char`-aligned, exactly the chunk the planner cut. Slice the caller's own
///   string with it ([`byte_range`](Self::byte_range)).
/// * [`token_span`](Self::token_span) — the window's placement in TOKENS:
///   `start()` is the running sum of the preceding windows' token counts (a
///   position in the concatenated window token stream, not an index into `text`
///   — under a non-zero overlap the repeated tokens are counted twice, which is
///   the double weighting the overlap expresses), `len()` is this window's own
///   token count including the [`SPECIAL_TOKENS_PER_WINDOW`] specials
///   ([`token_count`](Self::token_count)), and `window()` is [`MAX_TOKENS`]. Its
///   `coverage()` is the weight [`TextEmbedder::embed_long`] aggregates with.
/// * [`embedding`](Self::embedding) — the unit-norm 384-d [`Embedding`] of that
///   window's bytes alone, from one CoreML prediction.
///
/// This is not windit's `Windowed` (aliased there as `WindowEmbedding`, hence
/// the name collision): that is a value plus a `Span`, and this adds the anchor
/// into the caller's own text and the occurrence identity.
/// [`TextEmbedder::embed_long`] builds windit's pairing from the embedding and
/// the token span, and never sees the other two.
///
/// # No `PartialEq`
///
/// Deliberately, and for [`Embedding`]'s own reason: an ML model's f32 outputs
/// are not bit-stable across runs, threads, or OSes, so `==` on a value carrying
/// one is a trap. Compare the geometry field by field, and the vectors with
/// [`Embedding::is_close`] / [`Embedding::is_close_cosine`]:
///
/// ```compile_fail,E0369
/// # fn f(a: &coremlit::embeddings::granite::WindowEmbedding,
/// #      b: &coremlit::embeddings::granite::WindowEmbedding) -> bool {
/// a == b
/// # }
/// ```
///
/// For the same reason this type carries no `serde` impls: neither [`Embedding`]
/// nor windit's `Span` has any, and a persisted window vector is a
/// storage-format decision (fp16? quantized? which index?) that belongs to the
/// consumer, not to this door.
#[derive(Clone, Debug)]
pub struct WindowEmbedding {
  ordinal: usize,
  byte_range: Range<usize>,
  token_span: windit::plan::Span,
  embedding: Embedding,
}

impl WindowEmbedding {
  /// The window's position in planning order, `0..n` over the windows one call
  /// returned — its occurrence identity.
  ///
  /// Two windows can hold the same bytes and the same vector (a document that
  /// repeats a paragraph); the ordinal, with the byte range, is what keeps them
  /// distinct.
  ///
  /// Deterministic for the same `text` and [`LongTextOptions`] replanned
  /// against the same model artifact: chunking does not depend on whether the
  /// separatorless fast lane's merge table was built (that is a performance
  /// detail, pinned equal to the slow twin by the fast-vs-slow differential
  /// gates), so the same inputs always plan to the same ordinals. A change to
  /// either `text` or the options invalidates them — see
  /// [`TextEmbedder::embed_windows`]'s note on what to persist alongside a
  /// window. It is a planning POSITION, not a content hash: it carries no
  /// information about what the window contains, and two different texts can
  /// plan to the same ordinal sequence.
  #[inline]
  pub const fn ordinal(&self) -> usize {
    self.ordinal
  }

  /// First byte of the window in the caller's `text`, a `char` boundary.
  #[inline]
  pub const fn byte_start(&self) -> usize {
    self.byte_range.start
  }

  /// One past the window's last byte in the caller's `text`, a `char` boundary.
  #[inline]
  pub const fn byte_end(&self) -> usize {
    self.byte_range.end
  }

  /// The window's half-open byte range in the caller's `text` — the form to
  /// slice that same string with (`&text[w.byte_range()]`), which is why this
  /// returns the range by value rather than the pair.
  #[inline]
  pub fn byte_range(&self) -> Range<usize> {
    self.byte_range.clone()
  }

  /// The window's placement in the concatenated window token stream: `start()`
  /// tokens before it, `len()` tokens of its own, padded to a `window()` of
  /// [`MAX_TOKENS`] — the MODEL's fixed window, always, regardless of a smaller
  /// [`WindowOptions::window`] the caller configured for chunking. So
  /// `coverage()` (`len() / window()`) is relative to [`MAX_TOKENS`], not to a
  /// smaller configured chunk budget: a chunk that completely fills a
  /// 128-token [`WindowOptions`] still reports a coverage around `128 / 512`,
  /// not `1.0`. Its `coverage()` is the weight [`TextEmbedder::embed_long`]
  /// gives this window; that weighting is scale-invariant (windit's
  /// `CoverageWeightedMean` divides through by the largest weight in the fold),
  /// so using [`MAX_TOKENS`] rather than the configured `window()` as the
  /// denominator does not change `embed_long`'s answer.
  ///
  /// Token positions, NOT byte offsets into `text` — use
  /// [`byte_range`](Self::byte_range) to locate the window in the source.
  #[inline]
  pub const fn token_span(&self) -> windit::plan::Span {
    self.token_span
  }

  /// The window's own token count, granite's [`SPECIAL_TOKENS_PER_WINDOW`]
  /// specials included — `token_span().len()`, and at most [`MAX_TOKENS`].
  #[inline]
  pub const fn token_count(&self) -> usize {
    self.token_span.len()
  }

  /// The unit-norm embedding of this window's bytes alone.
  #[inline]
  pub const fn embedding(&self) -> &Embedding {
    &self.embedding
  }

  /// The embedding, moved out of the window — the only way to own it without a
  /// clone.
  #[must_use]
  #[inline]
  pub fn into_embedding(self) -> Embedding {
    self.embedding
  }
}

/// granite text embedder: a `&str` in, a unit-norm 384-d [`Embedding`] out.
///
/// Tokenizes with the granite tokenizer (truncation `LongestFirst` at
/// [`MAX_TOKENS`] and the tokenizer's own padding disabled, matching the goldens'
/// convention so token ids are identical), right-pads to the fixed `[1, 512]`
/// window with an attention mask, runs the fp16 CoreML ModernBERT graph, and
/// L2-normalizes the pre-normalization CLS projection.
#[derive(Debug)]
pub struct TextEmbedder {
  model: Model,
  tokenizer: Tokenizer,
  /// Right-padding token id for the fixed-length window. The pad positions are
  /// masked to 0, so their embedding is never read, and CLS pooling reads
  /// position 0 (never a pad); this only needs to be a valid vocabulary index.
  /// Resolved from the tokenizer's pad token `<|endoftext|>` at load, else `0`
  /// (a guaranteed-valid vocabulary index).
  pad_id: i32,
  /// Lazily built clone of `tokenizer` with truncation DISABLED — the tokenizer
  /// [`Self::embed_long`] measures chunk lengths with. The embed path's
  /// `tokenizer` truncates at [`MAX_TOKENS`], so its id counts saturate at 512
  /// and would tell the content-aware chunker that EVERY document fits one
  /// window; measurement must see the true, untruncated count. Lazy so
  /// embed-only callers pay nothing, and shared across every `embed_long` call.
  measure_tokenizer: OnceLock<Tokenizer>,
  /// The merge table behind the separatorless fast lane (#72), built from
  /// the stored tokenizer on the lane's first engagement — the first chunking
  /// probe longer than any token into a qualifying pre-token (a letter run the
  /// Split regex glues whole), so an embedder that never meets separatorless
  /// text never pays the build (about half a second, ~74
  /// MB transient, ~10 MB retained; see `token_index::bpe_mirror`) — and
  /// `None` when the tokenizer is not the configuration the lane is pinned
  /// to, in which case chunking measures as before.
  merge_table: OnceLock<Option<MergeTable>>,
}

impl TextEmbedder {
  /// Loads the granite `.mlmodelc` from `model_path` with the artifact's own
  /// [`TOKENIZER_FILE_NAME`] sidecar and custom `options` — the primary
  /// constructor. Pins the model's I/O contract against the metadata at load.
  ///
  /// The tokenizer is read from the model artifact's ROOT — the directory
  /// *containing* `model_path`, where the published bundle places
  /// `tokenizer.json` beside the `.mlmodelc` — and its bytes are held to the
  /// same fail-closed identity pin a caller-supplied tokenizer is. Callers who
  /// stage the tokenizer somewhere else use [`Self::from_files`] /
  /// [`Self::from_memory`].
  ///
  /// # Errors
  /// [`Error::ArtifactTokenizerRead`] if the sidecar is missing or unreadable;
  /// otherwise as [`Self::from_files`].
  pub fn load(model_path: impl AsRef<Path>, options: TextEmbedderOptions) -> Result<Self> {
    let model_path = model_path.as_ref();
    let tokenizer_path = artifact_tokenizer_path(model_path);
    let bytes = std::fs::read(&tokenizer_path).map_err(|source| {
      Error::ArtifactTokenizerRead(ArtifactTokenizerRead::new(tokenizer_path.clone(), source))
    })?;
    // Hash the RAW file bytes from the SAME read the parse sees (no second read,
    // no TOCTOU) and judge them BEFORE parsing — see `validate_tokenizer_identity`
    // for why the identity gate precedes `Tokenizer::from_bytes`.
    let sha256_hex = sha256_hex(&bytes);
    validate_tokenizer_identity(&TokenizerProvenance::Artifact(Artifact::new(
      tokenizer_path,
      sha256_hex,
    )))?;
    let tokenizer = Tokenizer::from_bytes(&bytes).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  /// Loads the granite `.mlmodelc` from `model_path` using the artifact's
  /// [`TOKENIZER_FILE_NAME`] sidecar and [`TextEmbedderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, TextEmbedderOptions::new())
  }

  /// Loads the model and a `tokenizer.json` from separate file paths.
  ///
  /// # Errors
  /// [`Error::TokenizerContractMismatch`] if the bytes are not byte-identical
  /// (SHA-256) to the pinned granite `tokenizer.json`, which is judged BEFORE
  /// they are parsed (`validate_tokenizer_identity`), or if the parsed tokenizer
  /// then fails the Granite tokenizer/model contract
  /// (`validate_tokenizer_contract`) — granite is a fixed model with exactly one
  /// correct tokenizer artifact; supply the pinned bytes (the artifact's own
  /// [`TOKENIZER_FILE_NAME`]). [`Error::TokenizerLoad`] if the pinned bytes are
  /// unreadable/invalid; [`Error::TokenizerConfig`] if truncation cannot be
  /// configured; [`Error::Load`] if CoreML rejects the model /
  /// [`Error::ContractMismatch`] if its I/O contract mismatches.
  pub fn from_files(
    model_path: impl AsRef<Path>,
    tokenizer_json_path: impl AsRef<Path>,
    options: TextEmbedderOptions,
  ) -> Result<Self> {
    // Read the file ONCE and delegate to `from_memory`, so the identity hash and
    // the parse see the SAME bytes (no re-read, no TOCTOU). An unreadable file
    // keeps today's `Error::TokenizerLoad` identity — `tokenizers::Error` is a
    // boxed `dyn Error`, so `io::Error` converts with `.into()`.
    let bytes =
      std::fs::read(tokenizer_json_path.as_ref()).map_err(|e| Error::TokenizerLoad(e.into()))?;
    Self::from_memory(model_path, &bytes, options)
  }

  /// Loads the model from a path and the tokenizer from caller-supplied bytes.
  ///
  /// # Errors
  /// As [`Self::from_files`].
  pub fn from_memory(
    model_path: impl AsRef<Path>,
    tokenizer_json_bytes: &[u8],
    options: TextEmbedderOptions,
  ) -> Result<Self> {
    // Hash the RAW supplied bytes (never a re-serialization of the parsed
    // `Tokenizer`, which would not reproduce the artifact's formatting/ordering)
    // and judge them BEFORE parsing — see `validate_tokenizer_identity`.
    let sha256_hex = sha256_hex(tokenizer_json_bytes);
    validate_tokenizer_identity(&TokenizerProvenance::Supplied(sha256_hex))?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_json_bytes).map_err(Error::TokenizerLoad)?;
    Self::from_parts(model_path, tokenizer, options)
  }

  fn from_parts(
    model_path: impl AsRef<Path>,
    mut tokenizer: Tokenizer,
    options: TextEmbedderOptions,
  ) -> Result<Self> {
    configure_tokenizer(&mut tokenizer)?;
    // The behavioral half of the tokenizer gate, fail-closed and BEFORE the
    // expensive `Model::load`. Every public constructor passes through here, and
    // each has already judged its RAW bytes against the byte-identity pin, so
    // this stage now re-derives from the pinned artifact what the pin already
    // guarantees. It is kept, and runs on every construction, because it is the
    // check that would still catch a tokenizer the pin ever stopped covering —
    // and because it is what the hermetic `configured_tokenizer_from_bytes` seam
    // exercises.
    validate_tokenizer_contract(&tokenizer)?;
    // The pad positions are attention-masked to 0 and CLS pooling reads position
    // 0 (never a pad), so the pad token value is immaterial to the output; a
    // valid vocabulary index is all that is required. `validate_tokenizer_contract`
    // above proved `<|endoftext|>` resolves to `contract::PAD_ID` — and
    // `contract::MAX_TOKEN_ID` (179_999) is far below `i32::MAX`, so the whole
    // vocabulary converts — which makes the `unwrap_or(0)` fallback unreachable
    // defensive code, kept for its guarantee of a valid index.
    let pad_id = tokenizer
      .token_to_id("<|endoftext|>")
      .and_then(|id| i32::try_from(id).ok())
      .unwrap_or(0);

    let model = Model::load(model_path, options.compute())?;
    let description = model.description();

    let ids_expected = format!("[1, {MAX_TOKENS}] int32");
    for name in [names::INPUT_IDS, names::ATTENTION_MASK] {
      let input = description.input(name).ok_or_else(|| {
        Error::ContractMismatch(ContractMismatch::new(
          name,
          ids_expected.clone(),
          "missing".to_string(),
        ))
      })?;
      if input.shape() != [1, MAX_TOKENS] || input.data_type() != Some(DataType::I32) {
        return Err(Error::ContractMismatch(ContractMismatch::new(
          name,
          ids_expected.clone(),
          describe(input.shape(), input.data_type()),
        )));
      }
    }

    let output_expected = format!("[1, {EMBEDDING_DIM}] float32");
    let output = description.output(names::EMBEDDING).ok_or_else(|| {
      Error::ContractMismatch(ContractMismatch::new(
        names::EMBEDDING,
        output_expected.clone(),
        "missing".to_string(),
      ))
    })?;
    if output.shape() != [1, EMBEDDING_DIM] || output.data_type() != Some(DataType::F32) {
      return Err(Error::ContractMismatch(ContractMismatch::new(
        names::EMBEDDING,
        output_expected,
        describe(output.shape(), output.data_type()),
      )));
    }

    Ok(Self {
      model,
      tokenizer,
      pad_id,
      measure_tokenizer: OnceLock::new(),
      merge_table: OnceLock::new(),
    })
  }

  /// The real token-id sequence for `text` (post-truncation at [`MAX_TOKENS`],
  /// pre-padding, granite special tokens included) — the sequence that is
  /// identity-comparable to the committed goldens
  /// (`tests/granite/tokenizer_identity.rs`).
  ///
  /// Tokenization runs over the whole of `text` before any truncation, so the
  /// cost is linear in the input's length and the input budget is the caller's
  /// (#118).
  ///
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure.
  pub fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
    if text.is_empty() {
      return Err(Error::EmptyText);
    }
    let encoding = self.tokenizer.encode(text, true).map_err(Error::Tokenize)?;
    Ok(encoding.get_ids().to_vec())
  }

  /// Embeds one text into a unit-norm [`Embedding`]. Prompt-free: feed the raw
  /// string.
  ///
  /// Tokenization runs over the whole of `text` before any truncation, so the
  /// cost is linear in the input's length and the input budget is the caller's
  /// (#118).
  ///
  /// # Errors
  /// [`Error::EmptyText`] if `text` is empty; [`Error::Tokenize`] on a tokenizer
  /// failure; [`Error::TokenCount`] if the tokenized input exceeds [`MAX_TOKENS`]
  /// or [`Error::TokenIdRange`] if a token id is out of `int32` range (both
  /// defensive — the tokenizer config makes neither reachable in practice);
  /// [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML
  /// failure; [`Error::OutputShape`] if the predicted `embedding` shape diverges
  /// from `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] if the model
  /// output has a NaN/infinite component — model corruption, classified apart
  /// from a caller's own non-finite embedding data
  /// ([`Error::NonFiniteEmbedding`]); [`Error::EmbeddingZero`] if the (finite)
  /// projection has zero magnitude.
  pub fn embed(&self, text: &str) -> Result<Embedding> {
    let ids = self.token_ids(text)?;
    self.embed_tokenized(&ids)
  }

  /// Everything after tokenization: right-pads `ids` to the fixed `[1, 512]`
  /// window, runs the CoreML graph, checks the output is finite, and
  /// L2-normalizes it. [`Self::embed`] is [`Self::token_ids`] composed with this;
  /// [`Self::embed_long`] runs it once per content-aware chunk.
  ///
  /// # Errors
  /// As the tensor / prediction / output tail of [`Self::embed`]:
  /// [`Error::TokenCount`] if `ids` exceeds [`MAX_TOKENS`] or
  /// [`Error::TokenIdRange`] if a token id is out of `int32` range (both
  /// defensive); [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML
  /// failure; [`Error::OutputShape`] on a shape divergence;
  /// [`Error::NonFiniteOutput`] on a NaN/infinite model output;
  /// [`Error::EmbeddingZero`] if the projection has zero magnitude.
  fn embed_tokenized(&self, ids: &[u32]) -> Result<Embedding> {
    // Right-pad to the fixed [1, 512] window; real tokens masked 1, pads 0. The
    // tokenizer config guarantees `ids` is real and within the window, but
    // `build_window` still guards it with a typed error instead of a panic.
    let (input_ids, attention_mask) = build_window(ids, self.pad_id)?;

    let ids_tensor = MultiArray::from_slice(&[1, MAX_TOKENS], &input_ids)?;
    let mask_tensor = MultiArray::from_slice(&[1, MAX_TOKENS], &attention_mask)?;
    let mut outputs = self.model.predict_with(&[
      (names::INPUT_IDS, &ids_tensor),
      (names::ATTENTION_MASK, &mask_tensor),
    ])?;
    let embeds = outputs
      .take(names::EMBEDDING)
      .ok_or_else(|| crate::PredictionError::MissingOutput(names::EMBEDDING.to_string()))?;
    if embeds.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape(OutputShape::new(
        embeds.shape().to_vec(),
        vec![1, EMBEDDING_DIM],
      )));
    }

    let mut row = [0.0f32; EMBEDDING_DIM];
    embeds.copy_into::<f32>(&mut row)?;
    // Classify a NaN/∞ the CoreML runtime produced as model-output corruption
    // (`NonFiniteOutput`) before it reaches `from_slice_normalizing`, which would
    // otherwise mislabel it as caller-supplied embedding data
    // (`NonFiniteEmbedding`).
    check_finite_output(&row)?;
    Embedding::from_slice_normalizing(&row)
  }

  /// A clone of the stored tokenizer with truncation DISABLED, built once and
  /// cached in [`Self::measure_tokenizer`]. This is the tokenizer
  /// [`Self::embed_long`] measures chunk lengths with: the embed path's
  /// tokenizer truncates at [`MAX_TOKENS`], so its counts saturate at 512 and
  /// would report every long document as fitting a single window.
  ///
  /// # Errors
  /// [`Error::TokenizerConfig`] if truncation cannot be reconfigured.
  fn measuring_tokenizer(&self) -> Result<&Tokenizer> {
    if let Some(t) = self.measure_tokenizer.get() {
      return Ok(t);
    }
    // Padding is already disabled on the stored tokenizer (construction), and the
    // clone inherits that; only truncation is lifted.
    let mut t = self.tokenizer.clone();
    t.with_truncation(None).map_err(Error::TokenizerConfig)?;
    // Racing initializers build identical values; the loser's clone is dropped.
    let _ = self.measure_tokenizer.set(t);
    Ok(
      self
        .measure_tokenizer
        .get()
        .expect("measure_tokenizer was set just above, on this thread or another"),
    )
  }

  /// Embeds arbitrarily long text: splits it into content-aware chunks of at
  /// most [`MAX_TOKENS`] tokens (respecting paragraph, sentence, and word
  /// boundaries), embeds each chunk with one CoreML prediction, and combines the
  /// per-chunk embeddings by a coverage-weighted spherical mean into one
  /// unit-norm [`Embedding`]. The chunks jointly cover every byte of `text` —
  /// separator bytes the content-aware splitter leaves at chunk boundaries
  /// (paragraph breaks; inter-word punctuation under its oversized-sentence
  /// fallback) are reattached to an adjacent chunk before embedding — so the
  /// aggregate represents the caller's whole text, as `embed` does within one
  /// window. Prompt-free, like [`Self::embed`], and equivalent to
  /// `embed_long_with(text, &LongTextOptions::new())`.
  ///
  /// Text that fits a single window returns exactly [`Self::embed`]'s embedding.
  ///
  /// Tokenization runs over the whole of `text` before any chunking, so the cost
  /// is linear in the input's length and the input budget is the caller's
  /// ([`LongTextOptions::max_input_bytes`], on [`Self::embed_long_with`], is the
  /// bound to set for untrusted input) (#118).
  ///
  /// # Not a retrieval representation for sparse evidence
  ///
  /// This is a DOCUMENT-LEVEL representation — one vector standing for the whole
  /// text — and averaging is what makes it one: a passage that answers a query
  /// is weighted by its share of the document, so a long document whose evidence
  /// sits in one window has that evidence diluted by every other window. #44
  /// measured it on a 16-document adversarial corpus (English, Chinese, mixed
  /// script, emoji; 513–8,192 true tokens; one relevant marker at the start,
  /// middle, or end): `embed_long` scored Recall@1 37.5% / MRR 0.5195 / nDCG
  /// 0.6274 — BELOW plain fixed-512 truncation (50.0%) — and 0/6 on the
  /// end-marker cases, while taking the max over the SAME windows scored 100%.
  /// (A purpose-built adversarial corpus, not a public benchmark: it rejects the
  /// production claim for that workload; the 100% is not a model-quality score.)
  ///
  /// For retrieval, embed the windows and score them: [`Self::embed_windows`]
  /// returns the same windows this call averages, each with its span in `text`,
  /// so a consumer can take a max, a top-k, or a fusion over them and keep the
  /// evidence span. Use `embed_long` for what it is — a whole-document summary
  /// vector (clustering, near-duplicate detection, a coarse first stage).
  ///
  /// # Errors
  /// As [`Self::embed_long_with`].
  pub fn embed_long(&self, text: &str) -> Result<Embedding> {
    self.embed_long_with(text, &LongTextOptions::new())
  }

  /// [`Self::embed_long`] with caller-controlled chunk geometry and an optional
  /// input-size bound ([`LongTextOptions`]). In the geometry
  /// ([`LongTextOptions::window_options`]): `window()` is the per-chunk token
  /// budget (must be `1..=`[`MAX_TOKENS`]), the overlap sets the repeated-token
  /// budget between consecutive chunks, and `max_windows()` caps the final chunk
  /// count — separator reattachment and the whole-input fallback chunk for
  /// contentless text included — which is exactly the number of CoreML
  /// predictions the call may dispatch. A cap of `0` therefore rejects every
  /// nonempty text, while `""` still fails [`Error::EmptyText`]. `tail()`
  /// ([`LongTextOptions::tail_policy`]) moves the last chunk boundary under
  /// `DropBelowMin` and does nothing under the other two — separator
  /// reattachment covers the tail windit drops, so no policy changes the chunk
  /// count or loses a byte.
  ///
  /// The per-chunk token budget counts granite's specials
  /// ([`SPECIAL_TOKENS_PER_WINDOW`] — `<|startoftext|>` and `<|return|>`, not a
  /// BERT `[CLS]`/`[SEP]` pair), because both the length measurement and each
  /// chunk's embedding run `encode(s, add_special_tokens = true)` —
  /// self-consistent by construction, so the effective content budget is
  /// `window − `[`SPECIAL_TOKENS_PER_WINDOW`], and at the default full window
  /// exactly [`CONTENT_TOKENS_PER_WINDOW`].
  ///
  /// With `overlap == 0` the chunks partition `text` (the first starts at byte 0,
  /// each begins where the previous ends, the last ends at `text.len()`); a
  /// non-zero overlap additionally repeats trailing regions. Reattached
  /// separators are re-measured against the budget; a pure-separator run neither
  /// neighbor can absorb becomes a chunk of its own and may exceed `window` up to
  /// [`MAX_TOKENS`] — the same tolerance as windit's lone oversized `char` — but a
  /// run measuring past [`MAX_TOKENS`] is refused with
  /// [`Error::ContentlessInputOverBudget`] rather than silently truncated. Such
  /// insertions count against `max_windows()`: the cap is enforced on the
  /// repaired chunk list, never silently exceeded.
  ///
  /// # Resource bounds
  /// Three independent limits, in the order the reject path applies them:
  /// * [`LongTextOptions::max_input_bytes`] — an input-size gate in UTF-8 bytes,
  ///   enforced BEFORE any tokenizer or chunker work; the only bound whose reject
  ///   cost is O(1) in the input size (`None` by default — the bound to set when
  ///   embedding untrusted text).
  /// * `window()` / `overlap()` — the per-chunk token geometry above.
  /// * `max_windows()` — a prediction-count cap: it bounds the CoreML predictions
  ///   dispatched and windit's chunk packing (which is cap-lazy), but NOT the
  ///   chunker's measurement cost — even a `max_windows()` of `0` tokenizes the
  ///   whole input once to build the single-pass token index (which then answers
  ///   every candidate-range measure without re-encoding) unless `max_input_bytes`
  ///   is set.
  ///
  /// # Errors
  /// [`Error::InputTooLarge`] if `text` exceeds `max_input_bytes`;
  /// [`Error::WindowOverBudget`] if `window()` exceeds [`MAX_TOKENS`] (every chunk
  /// would be silently truncated); [`Error::EmptyText`] for `""` (as
  /// [`Self::embed`]); [`Error::ContentlessInputOverBudget`] if a contentless run
  /// that must be embedded whole measures past [`MAX_TOKENS`]; [`Error::Tokenize`]
  /// on a tokenizer failure; [`Error::Windowing`] carrying a
  /// [`WinditError`](crate::embeddings::granite::error::WinditError) from chunking
  /// (e.g. `ZeroWindow`, `OverlapGeWindow`, `TooManyWindows` — the `max_windows`
  /// cap binds the final chunk count — post-reattachment, contentless nonempty
  /// text counting as one whole-input chunk — `got` reporting that full count) or
  /// aggregation (e.g. `NonFinite` when the per-chunk embeddings cancel exactly);
  /// plus any per-chunk tensor / prediction / output error (the same set
  /// [`Self::embed`] can raise).
  pub fn embed_long_with(&self, text: &str, opts: &LongTextOptions) -> Result<Embedding> {
    let mut windows = self.embed_windows_with(text, opts)?;
    // A single window is returned AS IT IS rather than routed through the
    // one-window aggregation, to skip a fold over `windit::aggregate` and the
    // second `Vec<Windowed<_>>` allocation just below — an optimization, not a
    // behavior change: `single_window_is_the_bit_exact_identity` pins
    // aggregating exactly one window under `CoverageWeightedMean` as the
    // BIT-EXACT identity of its input (`assert_eq!` on every component). For one
    // window the weight is `coverage / largest` = 1 exactly, the Neumaier fold
    // over one term is exact, and windit's `l2_renorm` composed with
    // `Embedding::from_unnormalized`'s `from_slice_normalizing` round-trips an
    // already-unit vector exactly — so aggregating would have answered
    // identically, not merely closely. After gap reattachment a single chunk
    // always spans `[0, text.len())`, so this also runs the same `token_ids` ∘
    // `embed_tokenized` path `embed` does on the same bytes;
    // `single_window_text_matches_embed` asserts that within a tolerance, since
    // separate CoreML predictions are not bit-stable with each other.
    if windows.len() == 1 {
      return Ok(
        windows
          .pop()
          .expect("a one-element vector has a last element")
          .into_embedding(),
      );
    }
    // The coverage-weighted spherical mean over the SAME windows
    // `embed_windows_with` just returned: `Span::coverage()` is `len / window`,
    // so a window carrying more real tokens weighs more. windit's pairing is
    // rebuilt here rather than carried through, because the byte anchor and the
    // ordinal — the two things this door adds — are of no interest to the
    // aggregator. This allocates a second `Vec` of `Windowed` (~1.6 KB/window: a
    // 384-`f32` `Embedding` plus a `Span`), because the two types' layouts
    // differ.
    let windowed: Vec<_> = windows
      .into_iter()
      .map(|w| windit::windowed::Windowed::new(w.embedding, w.token_span))
      .collect();
    Ok(windit::aggregate::aggregate(
      &windit::aggregate::CoverageWeightedMean,
      &windowed,
    )?)
  }

  /// Embeds arbitrarily long text ONE WINDOW AT A TIME: the same windows
  /// [`Self::embed_long`] plans and averages, each returned with its own
  /// unit-norm [`Embedding`] and its span in `text`. Equivalent to
  /// `embed_windows_with(text, &LongTextOptions::new())`.
  ///
  /// This is the retrieval path. `embed_long` collapses these vectors into one
  /// document representation, which dilutes localized evidence (#44, measured —
  /// see [`Self::embed_long`]); scoring the windows keeps it, and each window
  /// carries the byte range that produced it, so a hit points at its own
  /// evidence in the caller's text.
  ///
  /// Text that fits a single window returns exactly one [`WindowEmbedding`],
  /// spanning `[0, text.len())`, that runs the same `token_ids` ∘
  /// `embed_tokenized` path as [`Self::embed`] on the same bytes; equality is
  /// asserted with a tolerance in the tests (separate CoreML predictions are
  /// not bit-stable with each other).
  ///
  /// # The contract a consumer pins
  ///
  /// Query and indexed windows must be embedded by the same model, tokenizer,
  /// and pooling, or their cosines mean nothing. Everything that defines this
  /// space is FIXED and enforced, not configured:
  ///
  /// * **Model** — `granite-embedding-97m-multilingual-r2` as converted at
  ///   `FinDIT-Studio/embedkit-coreml` revision `a61241cb` (the revision
  ///   `MODELS_LOCK` pins and `tests/granite/model_io.rs` checks per file).
  /// * **Tokenizer** — the artifact's own [`TOKENIZER_FILE_NAME`], source
  ///   revision `835ad14087e140460703cf0fae09f97d469d65c2`, SHA-256
  ///   `4f2842…1582f`. Every constructor hashes the RAW bytes and REFUSES any
  ///   other tokenizer, so query-side and index-side agreement is a
  ///   construction-time guarantee rather than something a consumer must check.
  /// * **Pooling** — prompt-free CLS: the graph emits `hidden_states[:, 0]`
  ///   after the final LayerNorm, matching the checkpoint's own
  ///   `Transformer → Pooling(cls) → Normalize` module chain (asserted against
  ///   the pinned `1_Pooling/config.json` by `conversion/granite`). Feed RAW
  ///   strings — granite r2's query/document prompts are empty; a task prefix
  ///   would be embedded as text.
  /// * **Normalization** — every [`Embedding`] is L2-normalized to unit norm in
  ///   Rust (the graph emits the pre-norm vector), so a dot product IS the
  ///   cosine.
  /// * **Dimension** — [`EMBEDDING_DIM`], 384.
  /// * **Window budget** — [`MAX_TOKENS`] tokens per prediction, of which
  ///   [`SPECIAL_TOKENS_PER_WINDOW`] are the template's, leaving
  ///   [`CONTENT_TOKENS_PER_WINDOW`] for the caller's text.
  ///
  /// The geometry is the one variable, and it is the caller's: pin the
  /// [`LongTextOptions`] used at index time and reuse it, since a different
  /// window or overlap cuts different windows from the same document. That
  /// pinned value is exactly what a consumer must persist beside each window's
  /// spans to make later use of them well-defined — [`LongTextOptions`]
  /// (`LongTextOptions::new()` for this call) is `Copy + PartialEq + Eq`, and
  /// serde-capable under the `serde` feature, so storing it beside a persisted
  /// index is cheap and its identity is checkable at read time.
  ///
  /// # Errors
  /// As [`Self::embed_windows_with`] (see there for exactly how its error set
  /// relates to [`Self::embed_long_with`]'s). In particular `max_windows()`
  /// bounds the returned `Vec` (it is the prediction cap), so this cannot
  /// return more windows than that: it refuses with `TooManyWindows` rather
  /// than truncating.
  pub fn embed_windows(&self, text: &str) -> Result<Vec<WindowEmbedding>> {
    self.embed_windows_with(text, &LongTextOptions::new())
  }

  /// [`Self::embed_windows`] with caller-controlled chunk geometry and an
  /// optional input-size bound — the window-level twin of
  /// [`Self::embed_long_with`], which the latter is now defined in terms of:
  /// `embed_long_with` is the coverage-weighted spherical mean of these
  /// embeddings over these spans (a lone window is returned unaggregated: it
  /// runs the same `token_ids` ∘ `embed_tokenized` path as [`Self::embed`] on
  /// the same bytes; equality is asserted with a tolerance in the tests).
  ///
  /// The geometry, the resource bounds, and the whole-text coverage guarantee
  /// are [`Self::embed_long_with`]'s, unchanged: the windows jointly cover every
  /// byte of `text` (with `overlap == 0` they partition it — the first starts at
  /// byte 0, each starts where the previous ended, the last ends at
  /// `text.len()`), each range is `char`-aligned, and each window re-tokenizes
  /// to at most [`MAX_TOKENS`] ids.
  ///
  /// # Errors
  /// As [`Self::embed_long_with`], minus the aggregation failures it alone can
  /// raise (`NonFinite` from windit when the per-window embeddings cancel
  /// exactly): the per-window vectors are handed back before any combination.
  pub fn embed_windows_with(
    &self,
    text: &str,
    opts: &LongTextOptions,
  ) -> Result<Vec<WindowEmbedding>> {
    validate_long_input(text, opts)?;
    let wopts = opts.window_options();
    let mut chunks = chunk_long(
      self.measuring_tokenizer()?,
      LazyTable::new(&self.merge_table, &self.tokenizer),
      text,
      &wopts,
    )?;
    // Only `""` chunks to nothing — `chunk_long` already gives contentless
    // NONEMPTY text a single whole-input chunk — and the window `""` would have
    // had is that same whole-input one. Synthesizing it keeps ONE window body for
    // every input rather than a branch: the loop's own `token_ids` refuses the
    // empty string with `EmptyText`, which is `embed`'s empty-input contract
    // raised by the very call `embed` itself makes. No window is fabricated —
    // the refusal lands before any `Span`, which could not hold a zero-token
    // one.
    if chunks.is_empty() {
      chunks.push(windit::split::Chunk::new(0, text.len()));
    }
    let mut windows = Vec::with_capacity(chunks.len());
    // Cumulative token offset. Aggregation reads coverage, not position, so for
    // `embed_long` this is informational; for a consumer it is where the window
    // sits in the concatenated window token stream. Under overlap the offsets
    // overstate positions (overlapped tokens counted twice), which is exactly
    // the double-weighting overlap is meant to express.
    let mut offset = 0usize;
    for (ordinal, chunk) in chunks.iter().enumerate() {
      let s = chunk.as_str(text).expect(
        "chunk_long yields char-aligned boundaries (windit cuts, or 0/len from gap repair / the whole-input fallback)",
      );
      let ids = self.token_ids(s)?;
      let embedding = self.embed_tokenized(&ids)?;
      // `Span::new` needs `0 < ids.len() <= MAX_TOKENS`. `embed_tokenized` just
      // proved the upper bound (`build_window`'s typed guard). The lower one is
      // the post-processor's: it brackets EVERY sequence it encodes with
      // `<|startoftext|>`/`<|return|>`, so a returned `ids` is at least
      // `SPECIAL_TOKENS_PER_WINDOW` long — and the one input that would encode
      // to nothing, `""`, never reaches here, `token_ids` having refused it.
      let token_span = windit::plan::Span::new(offset, ids.len(), MAX_TOKENS);
      offset += ids.len();
      windows.push(WindowEmbedding {
        ordinal,
        byte_range: chunk.start()..chunk.end(),
        token_span,
        embedding,
      });
    }
    Ok(windows)
  }
}

/// Overrides the loaded tokenizer's truncation and padding policy to this
/// module's fixed-window contract, so the contract holds for ANY tokenizer
/// (artifact-read or caller-supplied) regardless of what it carried:
///
/// * **Truncation** `LongestFirst` at [`MAX_TOKENS`], stride 0, right direction —
///   the convention the committed goldens were tokenized under (fixed 512, right
///   truncation), so this module's token ids match them exactly. The export
///   sequence length is a hard model constraint, not a knob.
/// * **Padding disabled** (`with_padding(None)`) — this module does its own
///   fixed-window right-padding in [`build_window`] and masks the pad positions.
///   Leaving an inherited padding policy in place would let pad ids reach
///   [`TextEmbedder::token_ids`] marked as real tokens (corrupt mask), push the
///   CLS token off position 0 under left-padding (wrong CLS pooling), or overflow
///   the window under fixed-padding beyond 512.
///
/// The tokenizer is a caller input on the `from_files` path, so this is also
/// where its special-token overhead is checked against the window (see
/// [`SpecialTokenOverhead`]) — before [`validate_tokenizer_contract`]'s sentinel
/// encode, because the overhead breaks the CONFIGURATION, not the encoding.
///
/// # Errors
/// [`Error::SpecialTokenOverhead`] if the tokenizer's post-processor adds at
/// least [`MAX_TOKENS`] special tokens, leaving no room for text;
/// [`Error::TokenizerConfig`] if the tokenizer rejects the truncation policy.
fn configure_tokenizer(tokenizer: &mut Tokenizer) -> Result<()> {
  // `Tokenizer::with_truncation` computes the effective text window as
  // `max_length - post_processor.added_tokens(false)` with an UNCHECKED usize
  // subtraction, and `encode(_, true)` — which this module always calls —
  // repeats it. Read the same number off the public `PostProcessor` trait and
  // refuse the tokenizer while the arithmetic is still ours. `>=` rather than
  // the dependency's `>`: the equal case subtracts cleanly to a ZERO-token text
  // window, whose every encoding is the special tokens alone.
  let added = tokenizer
    .get_post_processor()
    .map_or(0, |post| post.added_tokens(false));
  if added >= MAX_TOKENS {
    return Err(Error::SpecialTokenOverhead(SpecialTokenOverhead::new(
      added, MAX_TOKENS,
    )));
  }
  tokenizer
    .with_truncation(Some(TruncationParams {
      max_length: MAX_TOKENS,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
      direction: TruncationDirection::Right,
    }))
    .map_err(Error::TokenizerConfig)?;
  tokenizer.with_padding(None);
  Ok(())
}

/// Validates a tokenizer against the Granite model contract, fail-closed: a
/// parseable-but-foreign tokenizer would otherwise produce finite, unit-norm,
/// semantically meaningless embeddings, or emit ids past the model's embedding
/// table that gather to zeros and surface only as a misleading
/// [`Error::EmbeddingZero`]. Run by every constructor on the CONFIGURED tokenizer
/// (after [`configure_tokenizer`]), so the sentinel check proves the exact
/// production [`TextEmbedder::token_ids`] behavior.
///
/// Checks, first failure wins: the three special-token ids, the total vocabulary
/// size, the maximum token id (the out-of-vocabulary gate), then the pinned
/// sentinel encoding. `validate_tokenizer_identity` has already refused anything
/// that is not the pinned artifact, byte for byte, so these behavioral checks now
/// re-derive from those bytes what the pin already guarantees: they are the
/// backstop that would still hold if the pin ever stopped covering a path, and
/// the gate the hermetic `configured_tokenizer_from_bytes` seam runs against.
///
/// # Errors
/// [`Error::TokenizerContractMismatch`] naming the first failed check;
/// [`Error::Tokenize`] if the sentinel encode itself fails.
fn validate_tokenizer_contract(tokenizer: &Tokenizer) -> Result<()> {
  for (check, token, expected_id) in [
    (
      "special token <|startoftext|>",
      contract::CLS_TOKEN,
      contract::CLS_ID,
    ),
    (
      "special token <|endoftext|>",
      contract::PAD_TOKEN,
      contract::PAD_ID,
    ),
    (
      "special token <|return|>",
      contract::EOS_TOKEN,
      contract::EOS_ID,
    ),
  ] {
    let actual = tokenizer.token_to_id(token);
    if actual != Some(expected_id) {
      return Err(Error::TokenizerContractMismatch(
        TokenizerContractMismatch::new(
          check,
          expected_id.to_string(),
          actual.map_or_else(|| "missing".to_string(), |id| id.to_string()),
        ),
      ));
    }
  }

  let vocab_size = tokenizer.get_vocab_size(true);
  if vocab_size != contract::VOCAB_SIZE {
    return Err(Error::TokenizerContractMismatch(
      TokenizerContractMismatch::new(
        "vocab size",
        contract::VOCAB_SIZE.to_string(),
        vocab_size.to_string(),
      ),
    ));
  }

  // The out-of-vocabulary gate: an id past the model's embedding table gathers
  // zeros. The count check above does not imply this bound — added tokens carry
  // explicit, possibly non-contiguous ids. `get_vocab(true)` allocates a ~180k
  // entry map, one-time at construction and trivial next to the model load.
  let max_id = tokenizer.get_vocab(true).values().copied().max();
  if !matches!(max_id, Some(id) if id <= contract::MAX_TOKEN_ID) {
    return Err(Error::TokenizerContractMismatch(
      TokenizerContractMismatch::new(
        "max token id",
        format!("<= {}", contract::MAX_TOKEN_ID),
        max_id.map_or_else(|| "empty vocab".to_string(), |id| id.to_string()),
      ),
    ));
  }

  let sentinel = tokenizer
    .encode(contract::SENTINEL_TEXT, true)
    .map_err(Error::Tokenize)?;
  if sentinel.get_ids() != contract::SENTINEL_IDS.as_slice() {
    return Err(Error::TokenizerContractMismatch(
      TokenizerContractMismatch::new(
        "sentinel encoding",
        format!("{:?}", contract::SENTINEL_IDS),
        format!("{:?}", sentinel.get_ids()),
      ),
    ));
  }

  Ok(())
}

/// Lowercase-hex SHA-256 of `bytes` — the tokenizer-identity digest. Mirrors the
/// dev-time provenance folds (`granite/tests.rs`, `tests/granite/common`).
fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// Read from the model artifact's own [`TOKENIZER_FILE_NAME`] sidecar
/// ([`TextEmbedder::load`] / [`TextEmbedder::from_file`]): the lowercase-hex
/// SHA-256 of the file's RAW bytes, plus the path they came from — a wrong or
/// truncated staged artifact must say WHICH file failed.
///
/// Payload of [`TokenizerProvenance::Artifact`].
struct Artifact {
  /// The sidecar path the bytes were read from.
  path: std::path::PathBuf,
  /// Lowercase-hex SHA-256 of the file's RAW bytes.
  sha256_hex: String,
}

impl Artifact {
  /// Construct from the sidecar path the bytes were read from and their
  /// lowercase-hex SHA-256.
  #[inline(always)]
  const fn new(path: std::path::PathBuf, sha256_hex: String) -> Self {
    Self { path, sha256_hex }
  }

  /// The sidecar path the bytes were read from.
  #[inline(always)]
  fn path(&self) -> &Path {
    &self.path
  }

  /// Lowercase-hex SHA-256 of the file's RAW bytes.
  #[inline(always)]
  fn sha256_hex(&self) -> &str {
    &self.sha256_hex
  }
}

/// Where a constructor's tokenizer bytes came from — carried so the
/// byte-identity backstop ([`validate_tokenizer_identity`]) can name the source
/// in its diagnostic. Both variants are checked against the same pin: nothing is
/// compiled in any more, so no tokenizer is identity-by-construction.
enum TokenizerProvenance {
  /// Read from the model artifact's own [`TOKENIZER_FILE_NAME`] sidecar
  /// ([`TextEmbedder::load`] / [`TextEmbedder::from_file`]): the lowercase-hex
  /// SHA-256 of the file's RAW bytes, plus the path they came from — a wrong or
  /// truncated staged artifact must say WHICH file failed.
  Artifact(Artifact),
  /// Caller-supplied bytes (`from_files` / `from_memory`): the lowercase-hex
  /// SHA-256 of the RAW input bytes, exactly as supplied (never a re-serialization
  /// of the parsed tokenizer).
  Supplied(String),
}

/// The artifact-root path [`TextEmbedder::load`] reads its tokenizer from: the
/// directory CONTAINING `model_path`, joined with [`TOKENIZER_FILE_NAME`].
///
/// The published bundle lays the artifact out with `tokenizer.json` beside the
/// `.mlmodelc` (its `CHECKSUMS.sha256` lists `./granite_97m_512.mlmodelc/...`
/// with the siblings at `./`). A `model_path` with no parent component yields
/// the bare file name, i.e. the current directory — the same place `.mlmodelc`
/// itself would resolve from.
fn artifact_tokenizer_path(model_path: &Path) -> std::path::PathBuf {
  model_path
    .parent()
    .unwrap_or_else(|| Path::new(""))
    .join(TOKENIZER_FILE_NAME)
}

/// Byte-identity backstop for EVERY tokenizer this module loads, fail-closed.
/// granite is a FIXED model, so exactly one tokenizer artifact is correct — the
/// pinned `tokenizer.json`, SHA-256 [`contract::TOKENIZER_SHA256_HEX`]. The
/// behavioral contract ([`validate_tokenizer_contract`]) runs FIRST (named,
/// actionable diagnostics for accidentally foreign tokenizers); this then catches
/// what no behavioral spot-check can — corruption or version skew outside the
/// sentinel's coverage (swapped vocab entries, divergent merges, normalizer
/// drift) that would silently produce wrong embeddings. A re-serialized but
/// behaviorally identical tokenizer is rejected BY DESIGN: supply the pinned
/// artifact bytes instead.
///
/// The artifact sidecar goes through the same gate as caller-supplied bytes:
/// once the tokenizer is a file on disk rather than compiled-in, "identity by
/// construction" no longer holds, and an unverified sidecar would be strictly
/// weaker than the embedded bytes it replaced.
///
/// # Why this runs before `Tokenizer::from_bytes`
///
/// Every constructor judges its RAW bytes here FIRST, so foreign bytes are never
/// handed to the tokenizers parser at all. That ordering is what closes the
/// dependency's own hazard: a `TemplateProcessing` that the tokenizers builder
/// would reject still DESERIALIZES (the crate's deserializer skips its builder's
/// `validate`), and applying it panics — indexing a `special_tokens` map that
/// does not carry the id, or `encodings[1]` for a single sequence. Because this
/// door pins one exact artifact, refusing before the parse costs nothing and
/// leaves no reachable path to that panic. The doors that accept an UNPINNED
/// tokenizer cannot rely on ordering and check the template's structure
/// explicitly instead (`embeddings::tokenizer_guard`).
///
/// The cost is a diagnostic one, and it is deliberate: an accidentally foreign
/// tokenizer is now named by its digest rather than by the first behavioral
/// difference [`validate_tokenizer_contract`] would have found. The remedy is the
/// same either way — supply the pinned artifact bytes.
///
/// # Errors
/// [`Error::TokenizerContractMismatch`] with `check = "tokenizer identity
/// (sha-256)"` (caller-supplied) or `"artifact tokenizer identity (sha-256)"`
/// (the sidecar) if the digest differs from the pin.
fn validate_tokenizer_identity(provenance: &TokenizerProvenance) -> Result<()> {
  let (check, sha256_hex, from) = match provenance {
    TokenizerProvenance::Artifact(artifact) => (
      "artifact tokenizer identity (sha-256)",
      artifact.sha256_hex(),
      Some(artifact.path()),
    ),
    TokenizerProvenance::Supplied(sha256_hex) => {
      ("tokenizer identity (sha-256)", sha256_hex.as_str(), None)
    }
  };
  if sha256_hex == contract::TOKENIZER_SHA256_HEX {
    return Ok(());
  }
  Err(Error::TokenizerContractMismatch(
    TokenizerContractMismatch::new(
      check,
      contract::TOKENIZER_SHA256_HEX.to_string(),
      from.map_or_else(
        || sha256_hex.to_string(),
        |path| format!("{sha256_hex} (read from {})", path.display()),
      ),
    ),
  ))
}

/// Builds the fixed `[1, `[`MAX_TOKENS`]`]` `input_ids` / `attention_mask` window
/// from the real token `ids`: the real tokens occupy the prefix (mask `1`) and
/// the remainder is right-padded with `pad_id` (mask `0`). CLS therefore stays at
/// position 0 and no pad position is ever attended.
///
/// [`configure_tokenizer`] forces truncation at [`MAX_TOKENS`] and disables the
/// tokenizer's own padding, so `ids` is already real and within the window; this
/// still returns a typed [`Error`] rather than panicking should that contract
/// ever be violated (an over-long or out-of-range id must not become an
/// out-of-bounds write or a wrapping cast). The id conversion in particular is a
/// backstop rather than a live guard: [`validate_tokenizer_contract`] refuses any
/// vocabulary whose maximum id exceeds [`contract::MAX_TOKEN_ID`] (179_999), far
/// inside `i32::MAX`, so no tokenizer that reaches here can produce one.
///
/// # Errors
/// [`Error::TokenCount`] if `ids` exceeds [`MAX_TOKENS`]; [`Error::TokenIdRange`]
/// if a token id does not fit the model's `int32` `input_ids` tensor.
fn build_window(ids: &[u32], pad_id: i32) -> Result<([i32; MAX_TOKENS], [i32; MAX_TOKENS])> {
  if ids.len() > MAX_TOKENS {
    return Err(Error::TokenCount(TokenCount::new(ids.len(), MAX_TOKENS)));
  }
  let mut input_ids = [pad_id; MAX_TOKENS];
  let mut attention_mask = [0i32; MAX_TOKENS];
  for (i, &id) in ids.iter().enumerate() {
    input_ids[i] = i32::try_from(id).map_err(|_| Error::TokenIdRange(id))?;
    attention_mask[i] = 1;
  }
  Ok((input_ids, attention_mask))
}

/// Rejects an oversized or mis-budgeted [`TextEmbedder::embed_long_with`] call
/// before any tokenizer or chunker work. Checked in order: the input byte limit
/// ([`Error::InputTooLarge`]), then the per-chunk budget
/// ([`Error::WindowOverBudget`]) — an over-budget window would let
/// [`TextEmbedder::token_ids`] silently truncate every chunk. Reads only
/// `text.len()` and the options — O(1), no tokenizer access — so the reject
/// path's cost is independent of the input size by construction. Factored out so
/// the check is hermetically testable. `window == 0` and `overlap >= window` are
/// left to windit's own validation (surfacing as [`Error::Windowing`]).
///
/// # Errors
/// [`Error::InputTooLarge`] if `text` exceeds
/// [`LongTextOptions::max_input_bytes`]; [`Error::WindowOverBudget`] if the
/// window exceeds [`MAX_TOKENS`].
fn validate_long_input(text: &str, opts: &LongTextOptions) -> Result<()> {
  if let Some(max) = opts.max_input_bytes()
    && text.len() > max
  {
    return Err(Error::InputTooLarge(InputTooLarge::new(text.len(), max)));
  }
  let window = opts.window_options().window();
  if window > MAX_TOKENS {
    return Err(Error::WindowOverBudget(WindowOverBudget::new(
      window, MAX_TOKENS,
    )));
  }
  Ok(())
}

/// The pure text-splitting stage of [`TextEmbedder::embed_long`]: token-budgeted,
/// boundary-aware byte ranges over `text`, measured with `measure_tok` (the
/// truncation-disabled tokenizer). Model-free, so the chunk geometry is
/// hermetically testable.
///
/// The chunks jointly cover `text`: windit's `ContentAware` extracts tokenized
/// content only, leaving separator bytes (paragraph breaks, whitespace-only
/// interiors, and word-fallback inter-word punctuation) uncovered at chunk
/// boundaries, so [`attach_gaps`] reattaches every such gap — re-measuring the
/// repaired substring against the window — before the chunks are returned. With
/// `overlap == 0` the chunks partition `text` (the first starts at byte 0, each
/// begins where the previous ends, the last ends at `text.len()`); a non-zero
/// overlap covers `text` while preserving its repeats. A
/// [`TailPolicy::DropBelowMin`] tail is a THIRD source of uncovered bytes, from
/// windit 0.5 (see [`LongTextOptions::tail_policy`]): the repair covers it like
/// any other gap, so the policy shifts the last boundary and never drops text.
/// Nonempty text always yields at least one chunk: text that chunks to nothing
/// — no tokenizable content at all (whitespace-only), or a lone chunk the tail
/// policy dropped — becomes a single whole-input chunk, the cost of the
/// whole-input `embed` fallback it is embedded by. Only `""` yields no
/// chunks.
///
/// Measurement and per-chunk embedding run the SAME tokenization
/// (`encode(s, add_special_tokens = true)`) on the SAME substring, so a chunk
/// measured at `<= window <= MAX_TOKENS` re-tokenizes to exactly the counted ids
/// and [`build_window`] never truncates or rejects it. Every chunk returned has
/// an untruncated measure `<= MAX_TOKENS`, with a single exception: windit's
/// lone oversized `char` (one `char` encodes to at most a handful of ids, far
/// below [`MAX_TOKENS`], so it can exceed a small `window` but never the model
/// window). Both contentless escapes — a pure-separator gap [`attach_gaps`]
/// would emit as its own chunk, and the whole-input fallback chunk for text with
/// no tokenizable content — are now MEASURED and refused past [`MAX_TOKENS`]
/// with [`Error::ContentlessInputOverBudget`] rather than silently truncated by
/// the embed path. The production tokenizer's truncation therefore never engages
/// on the `embed_long` path.
///
/// # Errors
/// [`Error::Windowing`] carrying whatever windit's `ContentAware::chunk` rejects
/// (a zero window, an overlap at or above the window, or a `max_windows`
/// overrun), or `TooManyWindows` raised here when gap reattachment or the
/// whole-input fallback chunk grows the final list past `opts.max_windows()`
/// — the cap binds the FINAL chunk count, exactly the per-chunk predictions
/// [`TextEmbedder::embed_long_with`] dispatches, `got` reporting that full
/// count (windit's own raise aborts at `max + 1`; this one reports the whole
/// overrun); [`Error::ContentlessInputOverBudget`] if a contentless run that
/// must be embedded whole (the whole input, or a pure-separator gap
/// [`attach_gaps`] emits) measures past [`MAX_TOKENS`] — measured at synthesis,
/// BEFORE the `max_windows` re-check; [`Error::Tokenize`] if the measuring
/// tokenizer fails to encode such a run.
fn chunk_long(
  measure_tok: &Tokenizer,
  table: LazyTable<'_>,
  text: &str,
  opts: &WindowOptions,
) -> Result<Vec<windit::split::Chunk>> {
  // Tokenize the whole input ONCE into a `TokenIndex`, then answer every range
  // measure from it: `index.measure_range(a, b)` returns exactly
  // `encode(&text[a..b], true).len()` — the count the old per-call
  // `encode(substring)` closure returned — but without re-encoding the growing
  // pack prefix that made chunking re-encode ~11× the input. An encode failure
  // during the build surfaces as `Error::Tokenize`, the same variant the
  // per-chunk `token_ids` in `embed_long_with` would raise one call later; the
  // `input_too_large` / window gate in `validate_long_input` has already run, so
  // this build is the first (and only whole-input) tokenization, exactly the cost
  // the old descent's first whole-input measure carried.
  let index = TokenIndex::build(measure_tok, text)?;
  // windit measures through this adapter — a real `MeasureText` impl, not a
  // blanket closure. Probes strictly inside a single letter-run pre-token
  // longer than any token (the separatorless regime of #72) engage the fast
  // lane — `table` is built on the first such probe, once per embedder — and
  // are answered from that pre-token's recorded merge process instead of a
  // whole-range re-encode; every other probe, and every probe when the
  // tokenizer cannot be mirrored, measures as before. It recovers each
  // subslice's byte range by pointer offset and
  // folds an encode error to `usize::MAX` ("does not fit"), so windit descends to
  // a smaller range and a persistent failure resurfaces from the per-chunk
  // `token_ids` later — the same behaviour the old infallible `measure` closure
  // had. The granite-side repair below instead calls `measure_range` directly
  // (fallible), surfacing an encode failure as `Error::Tokenize` at synthesis
  // rather than a bogus `ContentlessInputOverBudget { tokens: usize::MAX }` — the
  // same split the old `measure` / `measure_checked` pair drew, deliberately not
  // unified.
  let measurer = IndexMeasure::new(text, &index, measure_tok, table, opts.window());
  let measure_checked =
    |a: usize, b: usize| -> Result<usize> { index.measure_range(measure_tok, text, a, b) };
  let chunks = windit::split::ContentAware::new(&measurer)
    .chunk(text, opts)
    .map_err(Error::from)?;
  let mut repaired = attach_gaps(text, chunks, &measure_checked, opts.window())?;
  // Nonempty text can chunk to nothing two ways — no tokenizable content at all
  // (whitespace-only), or, from windit 0.5, a lone chunk below a
  // `TailPolicy::DropBelowMin` minimum, which windit discards as the tail it is
  // — yet `embed_long_with` still embeds it: the whole input through `embed`,
  // one CoreML prediction. So the tail policy cannot make a non-empty text
  // embed to nothing. Measure it first: a run measuring past
  // MAX_TOKENS would be silently right-truncated by the embed path (dropping
  // its suffix tokens), so refuse it with `ContentlessInputOverBudget`;
  // otherwise represent the cost as a single whole-input chunk so the cap below
  // bounds every prediction the result can dispatch. Only `""` stays chunkless
  // (it fails `EmptyText` before any prediction). The measure runs BEFORE the
  // `max_windows` re-check, so contentless over-budget input under
  // `max_windows == 0` yields `ContentlessInputOverBudget`, not `TooManyWindows`.
  if repaired.is_empty() && !text.is_empty() {
    let tokens = measure_checked(0, text.len())?;
    if tokens > MAX_TOKENS {
      return Err(Error::ContentlessInputOverBudget(
        ContentlessInputOverBudget::new(0, text.len(), tokens, MAX_TOKENS),
      ));
    }
    repaired.push(windit::split::Chunk::new(0, text.len()));
  }
  // windit enforced `max_windows` on ITS output; each own-chunk the repair
  // inserts — and the whole-input fallback chunk above — grows the count past
  // that check, and every chunk costs one CoreML prediction, so the cap
  // re-binds on the final list: it is exactly the number of predictions
  // `embed_long_with` may dispatch. Fail-closed: coverage and the cap cannot
  // both hold here, and silently exceeding the caller's work bound (or
  // silently dropping bytes) would be worse than a typed refusal. `got` is
  // the full final count, not windit's abort count.
  if let Some(max) = opts.max_windows()
    && repaired.len() > max
  {
    return Err(Error::Windowing(windit::WinditError::TooManyWindows {
      got: repaired.len(),
      max,
    }));
  }
  Ok(repaired)
}

/// Reattaches the byte gaps windit leaves between chunks, so [`chunk_long`]'s
/// output covers every byte of `text`. windit's `ContentAware` extracts
/// tokenized content only: paragraph separators (`\n\n` runs), whitespace-only
/// paragraph interiors, and — under its oversized-sentence word fallback — the
/// whitespace and punctuation between words are excluded, so a gap opens wherever
/// such bytes fall on a chunk boundary (including a leading gap before the first
/// chunk and a trailing gap after the last). A tail windit discarded under
/// [`TailPolicy::DropBelowMin`] is a trailing gap of the same kind, and is
/// reattached the same way — always as its own chunk, since the discarded tail
/// is by construction content its predecessor had no room for, so step 1 below
/// cannot take it.
///
/// A single left-to-right sweep closes every positive gap by re-measuring the
/// exact candidate substring against `window` (BPE is not additive — the repaired
/// range is re-measured, never assumed to gain a fixed token count), trying in
/// order:
///
/// 1. append the gap to the left neighbor if the extended range still fits —
///    left-first because terminal punctuation and paragraph breaks belong to the
///    preceding content, and it keeps every chunk starting where content starts;
/// 2. otherwise prepend it to the right neighbor if that range fits;
/// 3. otherwise emit the gap as its own chunk (pure separator bytes), reachable
///    only when both neighbors are already packed to exactly `window`.
///
/// With `overlap == 0` the result partitions `text`: the first chunk starts at
/// byte 0, each starts where the previous ends, the last ends at `text.len()`,
/// and the chunks concatenate back to `text`. With `overlap > 0` the pre-existing
/// overlaps are negative gaps, left untouched, so coverage is completed without
/// disturbing the repeats. The sweep never fuses two input chunks — each maps to
/// exactly one output chunk — so the output count is the input count plus one
/// per own-chunk emitted; [`chunk_long`] re-enforces `max_windows` on that
/// final count.
///
/// Every accepted attachment re-measures within `window`. A pure-separator
/// own-chunk (emitted when neither neighbor can absorb the gap) may still exceed
/// `window` up to [`MAX_TOKENS`] — the same tolerance as windit's lone
/// oversized-`char` escape — but its run is MEASURED, and a gap measuring past
/// [`MAX_TOKENS`] is refused with [`Error::ContentlessInputOverBudget`] rather
/// than left for the embed path to silently truncate. Every constructed boundary
/// is a windit cut or `0`/`text.len()`, all on `char` boundaries, so `Chunk::new`
/// never straddles a `char` and `as_str` never returns `None`.
///
/// # Errors
/// [`Error::ContentlessInputOverBudget`] if a pure-separator gap emitted as its
/// own chunk measures more than [`MAX_TOKENS`] tokens; [`Error::Tokenize`] if
/// the measuring tokenizer fails to encode a candidate substring (surfaced here,
/// one call earlier than the per-chunk `token_ids` would).
fn attach_gaps(
  text: &str,
  chunks: Vec<windit::split::Chunk>,
  measure: &dyn Fn(usize, usize) -> Result<usize>,
  window: usize,
) -> Result<Vec<windit::split::Chunk>> {
  use windit::split::Chunk;
  let Some(&first) = chunks.first() else {
    return Ok(chunks);
  };
  let mut out = Vec::with_capacity(chunks.len());
  let mut cur = first;
  // Leading gap: extend the first chunk left to byte 0, else emit the gap alone
  // (measured and refused past MAX_TOKENS, never left for the embed path to
  // silently truncate). `measure(a, b)` is the exact untruncated count of
  // `text[a..b]`, answered from the `TokenIndex`.
  if cur.start() > 0 {
    if measure(0, cur.end())? <= window {
      cur = Chunk::new(0, cur.end());
    } else {
      out.push(own_chunk(0, cur.start(), measure)?);
    }
  }
  for mut next in chunks.into_iter().skip(1) {
    let (gap_start, gap_end) = (cur.end(), next.start());
    if gap_start < gap_end {
      if measure(cur.start(), gap_end)? <= window {
        cur = Chunk::new(cur.start(), gap_end);
      } else if measure(gap_start, next.end())? <= window {
        next = Chunk::new(gap_start, next.end());
      } else {
        out.push(cur);
        out.push(own_chunk(gap_start, gap_end, measure)?);
        cur = next;
        continue;
      }
    }
    out.push(cur);
    cur = next;
  }
  // Trailing gap: extend the last chunk to `text.len()`, else emit the gap alone.
  if cur.end() < text.len() {
    if measure(cur.start(), text.len())? <= window {
      cur = Chunk::new(cur.start(), text.len());
    } else {
      let tail = own_chunk(cur.end(), text.len(), measure)?;
      out.push(cur);
      cur = tail;
    }
  }
  out.push(cur);
  Ok(out)
}

/// Builds the pure-separator own-chunk spanning the `start..end` byte range,
/// measuring its run first: a gap measuring past [`MAX_TOKENS`] would be silently
/// right-truncated by the embed path (dropping its suffix tokens), so it is
/// refused with [`Error::ContentlessInputOverBudget`] instead. The `(window,
/// MAX_TOKENS]` tolerance is kept — the same shape as windit's lone oversized
/// `char`.
///
/// # Errors
/// [`Error::ContentlessInputOverBudget`] if the run exceeds [`MAX_TOKENS`];
/// [`Error::Tokenize`] if the measuring tokenizer fails to encode it.
fn own_chunk(
  start: usize,
  end: usize,
  measure: &dyn Fn(usize, usize) -> Result<usize>,
) -> Result<windit::split::Chunk> {
  let tokens = measure(start, end)?;
  if tokens > MAX_TOKENS {
    return Err(Error::ContentlessInputOverBudget(
      ContentlessInputOverBudget::new(start, end, tokens, MAX_TOKENS),
    ));
  }
  Ok(windit::split::Chunk::new(start, end))
}

/// Test-only seam: the module's actual tokenizer configuration, without loading
/// a CoreML model — so `tests` can exercise the real tokenization path
/// hermetically (the tokenizer-identity gate).
#[cfg(test)]
pub(crate) fn configured_tokenizer_from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
  let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(Error::TokenizerLoad)?;
  configure_tokenizer(&mut tokenizer)?;
  Ok(tokenizer)
}

/// Test-only seam: the module's MEASURING tokenizer — the production
/// configuration ([`configured_tokenizer_from_bytes`]) with truncation then
/// DISABLED — without loading a CoreML model, so `tests` can exercise the real
/// `chunk_long` measurement path (and pin the truncation hazard) hermetically.
#[cfg(test)]
pub(crate) fn measuring_tokenizer_from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
  let mut tokenizer = configured_tokenizer_from_bytes(bytes)?;
  tokenizer
    .with_truncation(None)
    .map_err(Error::TokenizerConfig)?;
  Ok(tokenizer)
}

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

#[cfg(test)]
mod test_artifact;

#[cfg(test)]
mod tests;
