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
//! sidecar (`pos_embed_16x16x768.f32le.bin`) are **derived from the official**
//! [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex)
//! checkpoint (**Apache-2.0**; see the crate `NOTICE`) by the recipes in
//! `conversion/siglip/` — never consumed from a third-party artifact repo. They
//! are staged gitignored under `Models/siglip2-naflex/` (overridable via
//! `SIGLIP_TEST_MODELS`); the source revision and I/O contract are pinned by
//! `tests/siglip/model_io.rs` / `tests/siglip/text_model_io.rs`, and the per-file
//! SHA-256 by the committed manifest `MODELS_LOCK.d/siglip2-naflex@<revision>.sha256`
//! those tests read.
//!
//! The converted bundle is published at
//! [`FinDIT-Studio/siglip2-naflex-coreml`](https://huggingface.co/FinDIT-Studio/siglip2-naflex-coreml)
//! (revision `90d4dd21df57f167e73b3cd94cdf305edef8ddf1` — the graphs carry the
//! Neural Engine rewrite of issue #51: an explicit attention-pooling head and an
//! elementwise tanh-GELU, weights unchanged), so those artifacts can be fetched
//! instead of re-converted. That repo is the OUTPUT of the recipes above, not an
//! upstream this crate trusts: the committed manifest is the authority either
//! way, and it IS that revision's `CHECKSUMS.sha256` file-for-file.
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
//! ~99% ANE-preferred and, since the issue #51 rewrite, holds the committed floor
//! there too — but on the characterizing host the ANE arm is the slower one, so
//! the GPU path stays the default; the text graph's ANE compile fails, so it runs
//! on the GPU regardless. Both stay overridable per tower via
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

pub use embedding::Embedding;
pub use error::{
  ArtifactTokenizerIdentity, ArtifactTokenizerRead, ContractMismatch, EmbeddingDimMismatch, Error,
  ImageDataLength, ImageDimensions, OutputShape, PatchBudgetMismatch, PatchCount, PosEmbedLength,
  PreprocessedLength, PreprocessedMaskValue, PreprocessedNonFinite, PreprocessedPadNonZero,
  SpecialTokenOverhead, TokenCount,
};
pub use image::{
  DEFAULT_IMAGE_COMPUTE, ImageEmbedder, ImageEmbedderOptions, PreprocessedImage, Rgb8Image,
};
pub use text::{DEFAULT_TEXT_COMPUTE, TextEmbedder, TextEmbedderOptions};

/// File name of the SigLIP 2 Gemma `tokenizer.json` sidecar inside the model
/// artifact directory — the file [`TextEmbedder::load`] /
/// [`TextEmbedder::from_file`] read from the directory *containing* the text
/// `.mlmodelc`.
///
/// The tokenizer is the exact `tokenizer.json` of the source model repo
/// [`google/siglip2-base-patch16-naflex`](https://huggingface.co/google/siglip2-base-patch16-naflex)
/// at revision `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` (SHA-256
/// `58a1696e…b1b0`), the revision that produces the committed token-id goldens.
/// It is ~34 MB, so it is distributed with the CoreML graphs at
/// [`FinDIT-Studio/siglip2-naflex-coreml`](https://huggingface.co/FinDIT-Studio/siglip2-naflex-coreml)
/// rather than compiled into this crate.
///
/// The bytes read from disk are NOT trusted: [`TextEmbedder::load`] fails closed
/// on the build-time placeholder sentinel AND on any file whose SHA-256 is not
/// the pinned artifact's, so a wrong, truncated, or stale sidecar can never
/// silently produce meaningless embeddings. Callers who stage the tokenizer
/// elsewhere use [`TextEmbedder::from_files`] / [`TextEmbedder::from_memory`],
/// which remain the caller-supplies-bytes escape hatches.
pub const TOKENIZER_FILE_NAME: &str = "tokenizer.json";

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
