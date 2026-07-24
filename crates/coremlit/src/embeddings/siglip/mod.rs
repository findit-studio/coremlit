//! Native CoreML **SigLIP 2** (`siglip2-base-patch16-naflex`) inference: a NaFlex
//! vision encoder and a Gemma-tokenized text encoder that project into a shared
//! 768-dim joint embedding space, plus zero-shot cross-modal ranking
//! ([`rank`]).
//!
//! A decoded [`Rgb8Image`] in, a unit-norm 768-d [`Embedding`] out
//! ([`ImageEmbedder::embed`]); a `&str` in, the same [`Embedding`] out
//! ([`TextEmbedder::embed`]). Both towers share ONE [`Embedding`] type (a single
//! joint space), so an image and a caption are directly comparable by cosine.
//!
//! Design spec: `docs/superpowers/specs/2026-07-19-siglip-design.md` (GREENLIT,
//! dual-placement, end-user-decides).
//!
//! macOS only (built on [`crate`]).
//!
//! # NaFlex: native aspect-ratio patching (no windowing)
//!
//! Unlike a fixed-resolution ViT, NaFlex resizes each image to an
//! aspect-preserving grid that fills a fixed **patch budget** `P` (the shipped
//! tier is 512), so no tiling/windowing is needed. The host-side preprocessing
//! (the private `image::preprocess` port) is pure Rust: an aspect-preserving
//! budget solver, an antialiased-bilinear resize, rescale+normalize, patchify
//! into `[1, P, 768]` `pixel_values` + `[1, P]` `attention_mask`, and — the
//! port's central step — the **position-embedding lift**: the base `16×16×768`
//! grid is resized per image and passed as the `[1, P, 768]` `position_embeddings`
//! input (the in-graph resize does not convert to a single static CoreML graph,
//! so it is lifted host-side). `P` is resolved from the loaded model's
//! `pixel_values` contract at load ([`ImageEmbedder::max_num_patches`]), never a
//! code constant, so a 256/1024 tier is a drop-in artifact.
//!
//! Callers who reproduce the exact NaFlex pipeline offline can bypass the
//! in-crate preprocessing via [`ImageEmbedder::embed_preprocessed`]
//! ([`PreprocessedImage`]); [`ImageEmbedder::preprocess`] is the pipeline's
//! public producer, and [`ImageEmbedder::embed`] remains the safe default.
//!
//! # Text: single-input, full-window
//!
//! The SigLIP text graph takes **only** `input_ids` (`[1, T]`) — no attention
//! mask (canonical SigLIP attends every position) — and pools the final
//! position. Text is lowercased before tokenization (SigLIP2 convention;
//! checkpoint `do_lower_case: true`; mirrors transformers `Siglip2Tokenizer`).
//! The module builds a fixed `[1, T]` padded window whose pad id and side are
//! semantically load-bearing and are pinned by the committed goldens. `T` is
//! resolved from the loaded model at load ([`TextEmbedder::max_tokens`]).
//!
//! # Model artifacts
//!
//! The CoreML graphs (one fp16 artifact per tower) and the base position-grid
//! sidecar (`pos_embed_16x16x768.f32le.bin`) are distributed on the Hugging Face
//! Hub at `FinDIT-Studio/siglip2-naflex-coreml`, under the `512`-tier path
//! prefix, converted from
//! [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex)
//! (**Apache-2.0**; see the crate `NOTICE`). They are gitignored dev-time
//! downloads under `Models/siglip2-naflex/` (overridable via
//! `SIGLIP_TEST_MODELS`); their immutable revision, per-file SHA-256, and I/O
//! contract are pinned by `tests/siglip/model_io.rs` / `tests/siglip/text_model_io.rs`
//! once the owner stages the conversion (the conversion runbook in the port
//! plan).
//!
//! # Rust front-end around fp16 CoreML graphs
//!
//! Each graph emits the **pre-normalization** joint embedding; this module
//! applies the final L2 normalization in Rust (keeping the fp16 rsqrt-guard class
//! out of the graphs, the workspace convention).
//!
//! # Committed-golden oracle (no ort)
//!
//! Parity is scored against **committed transformers-fp32 fixtures**
//! (`tests/siglip/fixtures/goldens/`), never a live ONNX crate — the granite "no
//! ort anywhere, not even dev" rule. There is no `siglip-oracle` feature.
//!
//! # Compute placement (measured, never marketed)
//!
//! Placement is characterized, not asserted (`tests/siglip/placement.rs`). The
//! per-tower defaults are **measure-then-pin** [`crate::ComputeUnits::CpuAndGpu`]
//! (see [`DEFAULT_IMAGE_COMPUTE`] / [`DEFAULT_TEXT_COMPUTE`]): the vision graph is
//! ~99% ANE-preferred yet its fp16-on-ANE parity is below the committed floor, so
//! the floor-holding GPU path is the default; the text graph's ANE compile fails,
//! so it runs on the GPU regardless. Both stay overridable per tower via
//! `with_compute` / `set_compute`; the GPU parity is granite-class (vision
//! 0.999959, text 0.999998).
//!
//! # Construct once, reuse, prewarm
//!
//! Construct each embedder once and **reuse** it: it loads its CoreML model at
//! construction and runs `&self` inference (no per-call load). Call
//! [`ImageEmbedder::prewarm`] / [`TextEmbedder::prewarm`] once after construction
//! and before serving to absorb the first-inference graph specialization, so the
//! first real request is warm.

pub mod embedding;
pub mod error;
pub mod image;
pub mod text;

#[cfg(feature = "serde")]
mod compute_units_serde;

pub use embedding::Embedding;
pub use error::Error;
pub use image::{
  DEFAULT_IMAGE_COMPUTE, ImageEmbedder, ImageEmbedderOptions, PreprocessedImage, Rgb8Image,
};
pub use text::{DEFAULT_TEXT_COMPUTE, TextEmbedder, TextEmbedderOptions};

/// Bytes of the bundled SigLIP 2 Gemma `tokenizer.json` compiled into the crate.
///
/// These are the exact `tokenizer.json` bytes of the source model repo
/// [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex)
/// at revision `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` (SHA-256
/// `58a1696e…b1b0`), the revision that produces the committed token-id goldens —
/// proven byte-correct by `tests/siglip/tokenizer_identity.rs`. Exposed for
/// callers who construct [`TextEmbedder`] via [`TextEmbedder::from_memory`]; the
/// [`TextEmbedder::load`] / [`TextEmbedder::from_file`] constructors use it
/// directly.
///
/// The `include_bytes!` embeds ~34 MB into the rlib/binary — the Wave-A design
/// accepted this cost (`BUNDLED_TOKENIZER` is the API). A build-time placeholder
/// guard remains as a regression backstop: [`TextEmbedder::load`] /
/// [`TextEmbedder::from_memory`] fail closed ([`Error::TokenizerPlaceholder`]) if
/// the placeholder is ever re-committed, so a stripped tokenizer can never
/// silently produce meaningless embeddings.
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("assets/tokenizer.json");

/// A candidate paired with its precomputed [`Embedding`] — the input unit to
/// [`rank`]. Borrowing keeps ranking allocation-free and lets the label flow
/// straight into the returned [`Ranked`].
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
  label: &'a str,
  embedding: &'a Embedding,
}

impl<'a> Candidate<'a> {
  /// Pair `label` with its precomputed embedding (an image's or a text's — both
  /// towers share the joint space).
  pub const fn new(label: &'a str, embedding: &'a Embedding) -> Self {
    Self { label, embedding }
  }

  /// The candidate label.
  #[inline]
  pub const fn label(&self) -> &'a str {
    self.label
  }

  /// The candidate's precomputed embedding.
  #[inline]
  pub const fn embedding(&self) -> &'a Embedding {
    self.embedding
  }
}

/// One ranked candidate, borrowing its label from the [`Candidate`] it came
/// from, scored by cosine against the query.
#[derive(Debug, Clone, Copy)]
pub struct Ranked<'a> {
  label: &'a str,
  score: f32,
}

impl<'a> Ranked<'a> {
  /// The ranked label.
  #[inline]
  pub const fn label(&self) -> &'a str {
    self.label
  }

  /// The cosine score against the query, in roughly `[-1, 1]`.
  #[inline]
  pub const fn score(&self) -> f32 {
    self.score
  }
}

/// Rank `candidates` against a `query` [`Embedding`] by cosine, descending.
///
/// Cross-modal: the `query` can be an image and the `candidates` texts (zero-shot
/// classification / retrieval), or vice versa — both towers share the joint
/// space. The score is the raw cosine (v1 ships cosine/rank only; the checkpoint's
/// `logit_scale`/`logit_bias` sigmoid scoring is recorded in the artifact
/// metadata for a future `score()`). Ties keep input order (the sort is stable);
/// an empty `candidates` yields an empty vec.
#[must_use]
pub fn rank<'a>(query: &Embedding, candidates: &[Candidate<'a>]) -> Vec<Ranked<'a>> {
  let mut out: Vec<Ranked<'a>> = candidates
    .iter()
    .map(|c| Ranked {
      label: c.label(),
      score: query.cosine(c.embedding()),
    })
    .collect();
  // Descending by score; `sort_by` is stable, so ties keep input order.
  out.sort_by(|x, y| {
    y.score
      .partial_cmp(&x.score)
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  out
}

#[cfg(test)]
mod tests;
