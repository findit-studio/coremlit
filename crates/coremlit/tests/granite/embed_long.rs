//! `embed_long` on the granite CoreML graph (model-gated): long-document
//! aggregation, the single-window equivalence to `embed`, and the empty /
//! over-budget error contracts.
//!
//! The content-aware chunk GEOMETRY is proven model-free in the in-lib granite
//! suite (`src/embeddings/granite/tests.rs`); this file gates the CoreML
//! per-chunk embedding + windit aggregation path on the downloaded artifact.
//! Model-gated tests are `#[ignore]` by default and run only with the granite
//! model staged under `Models/embedkit-granite/` (or `EMBEDKIT_TEST_MODELS`).

mod common;

use coremlit::embeddings::granite::{Error, MAX_TOKENS, TextEmbedder, WindowOptions};

fn embedder() -> TextEmbedder {
  TextEmbedder::from_file(common::model_path()).unwrap_or_else(|e| panic!("load granite: {e}"))
}

/// A deterministic multi-paragraph document comfortably over several 512-token
/// windows, so `embed_long` exercises the true multi-chunk aggregation path.
fn long_document() -> String {
  (0..32)
    .map(|p| {
      (0..40)
        .map(|w| format!("paragraph{p}word{w}"))
        .collect::<Vec<_>>()
        .join(" ")
    })
    .collect::<Vec<_>>()
    .join("\n\n")
}

/// A document spanning multiple windows aggregates to one finite unit-norm
/// embedding (the coverage-weighted spherical mean through windit).
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn long_document_aggregates_to_one_unit_norm_vector() {
  let emb = embedder();
  let doc = long_document();
  let out = emb
    .embed_long(&doc)
    .expect("embed_long a multi-window document");
  let norm_sq: f32 = out.as_slice().iter().map(|x| x * x).sum();
  assert!(
    (norm_sq - 1.0).abs() < 1e-5,
    "aggregate is not unit-norm: norm² = {norm_sq}"
  );
  assert!(
    out.as_slice().iter().all(|v| v.is_finite()),
    "aggregate has a non-finite component"
  );
}

/// A text that fits one window returns `embed`'s embedding: the single-window
/// short-circuit runs the SAME `token_ids` ∘ `embed_tokenized` path on the same
/// bytes. Assert closeness, not bit-equality — model f32 outputs are not
/// bit-stable (why `Embedding: !PartialEq`).
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn single_window_text_matches_embed() {
  let emb = embedder();
  let text = "a compact sentence that fits comfortably inside one window";
  let via_long = emb.embed_long(text).expect("embed_long a short text");
  let via_embed = emb.embed(text).expect("embed the same text");
  assert!(
    via_long.is_close(&via_embed, 1e-5),
    "single-window embed_long must match embed"
  );
}

/// Empty text errors exactly as `embed` does (the 0-chunk delegate keeps the
/// empty-text contract identical).
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn empty_text_errors_like_embed() {
  let emb = embedder();
  assert!(matches!(emb.embed_long(""), Err(Error::EmptyText)));
}

/// A per-chunk budget above the model's fixed window is rejected before any
/// prediction runs (`Error::WindowOverBudget`), through the public
/// `embed_long_with` entry.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn over_budget_window_rejected_before_any_prediction() {
  let emb = embedder();
  let err = emb
    .embed_long_with("any text", &WindowOptions::new(MAX_TOKENS + 1))
    .unwrap_err();
  assert!(
    matches!(err, Error::WindowOverBudget { window, max } if window == MAX_TOKENS + 1 && max == MAX_TOKENS),
    "expected WindowOverBudget, got {err:?}"
  );
}

/// A `max_windows` of 0 can never be satisfied by nonempty text — even
/// whitespace-only text, whose content-aware chunking yields no content,
/// still costs one whole-input prediction — so `embed_long_with` refuses it
/// before any prediction, reporting the one-window cost against the cap.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn whitespace_at_cap_zero_rejected_before_any_prediction() {
  use coremlit::embeddings::granite::error::WinditError;

  let emb = embedder();
  let err = emb
    .embed_long_with("   ", &WindowOptions::new(MAX_TOKENS).with_max_windows(0))
    .unwrap_err();
  assert!(
    matches!(
      err,
      Error::Windowing(WinditError::TooManyWindows { got: 1, max: 0 })
    ),
    "expected Windowing(TooManyWindows {{ got: 1, max: 0 }}), got {err:?}"
  );
}

/// At cap 1 the same whitespace-only text embeds through the single
/// whole-input fallback chunk — the identical `token_ids` ∘ `embed_tokenized`
/// call `embed` makes on the same bytes, so the embeddings match.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn whitespace_at_cap_one_matches_embed() {
  let emb = embedder();
  let via_long = emb
    .embed_long_with("   ", &WindowOptions::new(MAX_TOKENS).with_max_windows(1))
    .expect("cap 1 admits the one whole-input prediction");
  let via_embed = emb.embed("   ").expect("embed whitespace");
  assert!(
    via_long.is_close(&via_embed, 1e-5),
    "whole-input fallback must match embed"
  );
}
