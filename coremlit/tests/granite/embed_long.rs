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

use coremlit::embeddings::granite::{
  Error, LongTextOptions, MAX_TOKENS, TailPolicy, TextEmbedder, WindowOptions,
};

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
    .embed_long_with(
      "any text",
      &LongTextOptions::from(WindowOptions::new(MAX_TOKENS + 1)),
    )
    .unwrap_err();
  assert!(
    matches!(err, Error::WindowOverBudget(ref b) if b.window() == MAX_TOKENS + 1 && b.max() == MAX_TOKENS),
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
    .embed_long_with(
      "   ",
      &LongTextOptions::from(WindowOptions::new(MAX_TOKENS).with_max_windows(0)),
    )
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
    .embed_long_with(
      "   ",
      &LongTextOptions::from(WindowOptions::new(MAX_TOKENS).with_max_windows(1)),
    )
    .expect("cap 1 admits the one whole-input prediction");
  let via_embed = emb.embed("   ").expect("embed whitespace");
  assert!(
    via_long.is_close(&via_embed, 1e-5),
    "whole-input fallback must match embed"
  );
}

/// A whitespace-only input too long to embed as one window is refused with
/// `Error::ContentlessInputOverBudget` — the public end-to-end proof of the
/// measured-fallback fix (the chunk geometry is proven model-free in the in-lib
/// granite suite; this is the CoreML-path contract).
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn contentless_over_budget_input_is_refused() {
  let emb = embedder();
  let err = emb.embed_long(&" ".repeat(100_000)).unwrap_err();
  assert!(
    matches!(err, Error::ContentlessInputOverBudget(_)),
    "expected ContentlessInputOverBudget, got {err:?}"
  );
}

/// An oversized input is refused by the byte gate before any tokenizer or
/// chunker work (`Error::InputTooLarge`). Error-identity only — no timing
/// assertions (flaky in CI); the reject-cost characterization belongs in the PR
/// notes, per the issue's gate.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn oversized_input_rejected_with_input_too_large() {
  let emb = embedder();
  let big = "x".repeat(8 * 1024 * 1024);
  let err = emb
    .embed_long_with(&big, &LongTextOptions::new().with_max_input_bytes(1 << 20))
    .unwrap_err();
  assert!(
    matches!(err, Error::InputTooLarge(ref l) if l.got() == big.len() && l.max() == (1 << 20)),
    "expected InputTooLarge, got {err:?}"
  );
}

/// A [`TailPolicy::DropBelowMin`] geometry embeds on the real graph, and the
/// text it names as droppable is still embedded.
///
/// windit 0.5's `ContentAware` honours the minimum where 0.4 ignored it, so this
/// is the first release in which the knob reaches CoreML at all. The chunk-level
/// consequence is pinned model-free
/// (`a_drop_below_min_tail_moves_the_last_boundary_and_keeps_every_byte`): the
/// tail comes back as its own chunk, so the boundary moves and the count does
/// not. What only the graph can say is that the moved boundary still aggregates
/// — every chunk re-tokenizes inside the window and the coverage-weighted mean
/// stays finite and unit-norm.
///
/// The second half is the shape windit documents as "a non-empty input can now
/// yield no chunks at all": a lone chunk under the minimum. `chunk_long`'s
/// whole-input fallback catches it, so the text embeds as `embed` would rather
/// than vanishing — the failure this would otherwise be. That windit really
/// returns nothing here is asserted in the hermetic twin; on the graph both
/// routes would produce the same vector, so what this half can say is the part
/// that matters to a caller: an answer comes back, and it is `embed`'s.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn a_drop_below_min_geometry_embeds_and_never_drops_the_text() {
  let emb = embedder();
  let doc = long_document();
  // A small per-chunk budget, so the document really has a ragged final chunk
  // for the minimum to bite on.
  let geometry = WindowOptions::new(64);
  for opts in [
    LongTextOptions::from(geometry),
    LongTextOptions::from(geometry).with_tail_policy(TailPolicy::DropBelowMin(48)),
  ] {
    let out = emb
      .embed_long_with(&doc, &opts)
      .unwrap_or_else(|e| panic!("embed_long under {opts:?}: {e}"));
    let norm_sq: f32 = out.as_slice().iter().map(|x| x * x).sum();
    assert!(
      (norm_sq - 1.0).abs() < 1e-5,
      "aggregate under {opts:?} is not unit-norm: norm² = {norm_sq}"
    );
    assert!(
      out.as_slice().iter().all(|v| v.is_finite()),
      "aggregate under {opts:?} has a non-finite component"
    );
  }

  // A text whose ONLY chunk is below the minimum: windit yields nothing, the
  // whole-input fallback embeds it, and that is `embed`'s own answer.
  let short = "a compact sentence that fits comfortably inside one window";
  let via_long = emb
    .embed_long_with(
      short,
      &LongTextOptions::from(WindowOptions::new(MAX_TOKENS))
        .with_tail_policy(TailPolicy::DropBelowMin(MAX_TOKENS)),
    )
    .expect("a lone below-minimum chunk must still embed");
  let via_embed = emb.embed(short).expect("embed the same text");
  assert!(
    via_long.is_close(&via_embed, 1e-5),
    "the whole-input fallback must match embed"
  );
}
