//! `embed_windows` on the granite CoreML graph (model-gated): the window-level
//! retrieval path of #160 — one embedding per planned window, its declared byte
//! and token spans, its occurrence identity, and the identity that makes it the
//! same pass `embed_long` runs.
//!
//! The chunk GEOMETRY is proven model-free in the in-lib granite suite
//! (`src/embeddings/granite/tests.rs`), and `embed_long`'s own contracts in
//! `tests/granite/embed_long.rs`. What only the graph can say is here: that the
//! windows this door hands back are exactly the ones `embed_long` averages, and
//! that its refusals are `embed_long`'s refusals. Model-gated tests are
//! `#[ignore]` by default and run only with the granite model staged under
//! `Models/embedkit-granite/` (or `EMBEDKIT_TEST_MODELS`).

mod common;

use coremlit::embeddings::granite::{
  Error, LongTextOptions, MAX_TOKENS, SPECIAL_TOKENS_PER_WINDOW, TextEmbedder, WindowEmbedding,
  WindowOptions,
};

fn embedder() -> TextEmbedder {
  TextEmbedder::from_file(common::model_path()).unwrap_or_else(|e| panic!("load granite: {e}"))
}

/// A deterministic multi-paragraph document comfortably over several 512-token
/// windows, so the window path exercises true multi-chunk planning.
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

/// One paragraph large enough to fill most of a window on its own, so a document
/// built by repeating it plans one paragraph per window.
fn wide_paragraph() -> String {
  (0..150)
    .map(|w| format!("lexeme{w}"))
    .collect::<Vec<_>>()
    .join(" ")
}

/// An INDEPENDENT coverage-weighted spherical mean, written from the declared
/// contract rather than called through windit: `Σ coverage_i · v_i`, L2-normalized,
/// accumulated in `f64`. `coverage` is `token_count / window`, read off the span
/// this door published — so if the published spans stopped describing the windows
/// `embed_long` weights, this reimplementation would stop reproducing it.
///
/// windit lifts its weights by `1 / max_j c_j` and sums with compensation; both
/// are uniform positive scalings or rounding, which the final normalization
/// removes, so the two agree to well inside [`TOL`].
fn coverage_weighted_mean(windows: &[WindowEmbedding]) -> Vec<f32> {
  assert!(!windows.is_empty(), "no windows to aggregate");
  let dim = windows[0].embedding().as_slice().len();
  let mut acc = vec![0.0f64; dim];
  for w in windows {
    #[expect(
      clippy::cast_precision_loss,
      reason = "token counts are <= 512; exact in f64"
    )]
    let coverage = w.token_count() as f64 / w.token_span().window() as f64;
    for (a, v) in acc.iter_mut().zip(w.embedding().as_slice()) {
      *a += coverage * f64::from(*v);
    }
  }
  let norm = acc.iter().map(|x| x * x).sum::<f64>().sqrt();
  assert!(norm > 0.0, "the window vectors cancelled exactly");
  #[expect(clippy::cast_possible_truncation, reason = "the door's own f32 output")]
  acc.iter().map(|x| (x / norm) as f32).collect()
}

/// Comparison tolerance for two f32 vectors that describe the same computation.
/// Model f32 outputs are not bit-stable (why `Embedding: !PartialEq`), and the
/// aggregate above is an independent f64 fold of them.
const TOL: f32 = 1e-5;

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(a.len(), b.len(), "dimension mismatch");
  a.iter()
    .zip(b)
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max)
}

/// The load-bearing identity of #160: `embed_long_with` IS the coverage-weighted
/// mean of `embed_windows_with`'s embeddings over their published spans — for a
/// many-window document, a text that fits one window, and contentless text, under
/// both the default geometry and a small one that makes the coverages differ
/// sharply between windows.
///
/// This is what makes the published span trustworthy: the weight a consumer can
/// compute from `token_span()` is the weight `embed_long` actually applied. It is
/// load-bearing against uniform weighting — replacing `coverage` above with `1.0`
/// takes the multi-window deltas to ~1e-2, three orders past `TOL`.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn embed_long_is_the_coverage_weighted_mean_of_the_windows() {
  let emb = embedder();
  let doc = long_document();
  let cases: [(&str, &str); 3] = [
    ("multi-window", doc.as_str()),
    (
      "single-window",
      "a compact sentence that fits comfortably inside one window",
    ),
    ("contentless", "   "),
  ];
  for (label, text) in cases {
    for (geometry, opts) in [
      ("default", LongTextOptions::new()),
      ("window 64", LongTextOptions::from(WindowOptions::new(64))),
    ] {
      let windows = emb
        .embed_windows_with(text, &opts)
        .unwrap_or_else(|e| panic!("{label}/{geometry}: embed_windows_with: {e}"));
      let pooled = emb
        .embed_long_with(text, &opts)
        .unwrap_or_else(|e| panic!("{label}/{geometry}: embed_long_with: {e}"));
      let rebuilt = coverage_weighted_mean(&windows);
      let delta = max_abs_delta(pooled.as_slice(), &rebuilt);
      assert!(
        delta <= TOL,
        "{label}/{geometry}: embed_long is not the coverage-weighted mean of its \
         {} windows (max |Δ| = {delta:e})",
        windows.len()
      );
    }
  }
}

/// The windows tile the caller's text in planning order: ordinals `0..n`, the
/// first starting at byte 0, each starting where the previous ended, the last
/// ending at `text.len()`, and every boundary on a `char` boundary — so
/// `&text[w.byte_range()]` is the exact substring that produced the embedding.
///
/// (`overlap == 0` here, which is what makes the tiling a partition; a non-zero
/// overlap covers the same bytes while repeating some.)
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn window_byte_ranges_tile_the_text_in_planning_order() {
  let emb = embedder();
  // Mixed script and multi-byte content, so a mis-aligned cut would be visible.
  let doc = format!(
    "{}\n\n{}\n\n{}",
    long_document(),
    "你好世界模型推理文本嵌入检索".repeat(200),
    "naïve café — résumé ✅🇯🇵 ".repeat(200)
  );
  for (label, opts) in [
    ("default", LongTextOptions::new()),
    ("window 64", LongTextOptions::from(WindowOptions::new(64))),
  ] {
    let windows = emb
      .embed_windows_with(&doc, &opts)
      .unwrap_or_else(|e| panic!("{label}: embed_windows_with: {e}"));
    assert!(
      windows.len() > 1,
      "{label}: the fixture must plan more than one window"
    );
    let mut cursor = 0usize;
    for (i, w) in windows.iter().enumerate() {
      assert_eq!(w.ordinal(), i, "{label}: ordinals are the planning order");
      assert_eq!(
        w.byte_start(),
        cursor,
        "{label}: window {i} does not start where window {} ended",
        i.wrapping_sub(1)
      );
      assert!(
        w.byte_end() > w.byte_start(),
        "{label}: window {i} covers no bytes"
      );
      assert!(
        doc.is_char_boundary(w.byte_start()) && doc.is_char_boundary(w.byte_end()),
        "{label}: window {i} is not char-aligned"
      );
      assert!(
        doc.get(w.byte_range()).is_some(),
        "{label}: window {i} does not slice its own text"
      );
      assert_eq!(w.byte_range(), w.byte_start()..w.byte_end());
      cursor = w.byte_end();
    }
    assert_eq!(
      cursor,
      doc.len(),
      "{label}: the windows must cover the text"
    );
  }
}

/// The token spans are the windows' placement in the concatenated window token
/// stream: `start()` is the running sum of the preceding windows' counts,
/// `len()` is this window's own count (the specials included, so never below
/// `SPECIAL_TOKENS_PER_WINDOW`) and never past `MAX_TOKENS`, and `window()` is
/// `MAX_TOKENS` — the denominator of the coverage `embed_long` weights with.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn token_span_offsets_are_the_running_sum_of_the_token_counts() {
  let emb = embedder();
  let doc = long_document();
  let windows = emb.embed_windows(&doc).expect("embed_windows");
  assert!(
    windows.len() > 1,
    "the fixture must plan more than one window"
  );
  let mut offset = 0usize;
  for w in &windows {
    assert_eq!(
      w.token_span().start(),
      offset,
      "window {} does not start at the running token sum",
      w.ordinal()
    );
    assert_eq!(w.token_count(), w.token_span().len(), "len() is the count");
    assert_eq!(w.token_span().window(), MAX_TOKENS, "the padded window");
    assert!(
      (SPECIAL_TOKENS_PER_WINDOW..=MAX_TOKENS).contains(&w.token_count()),
      "window {} has {} tokens",
      w.ordinal(),
      w.token_count()
    );
    offset += w.token_count();
  }
}

/// A document that repeats one paragraph plans byte-identical windows, and they
/// stay DISTINCT: same text and (to within model f32 stability) the same vector,
/// but different ordinals and different byte ranges. That is the occurrence
/// identity a consumer attaches its own provenance to — nothing else separates
/// the two.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn repeated_text_plans_distinct_windows_with_equal_embeddings() {
  let emb = embedder();
  let para = wide_paragraph();
  let doc = [para.as_str(); 4].join("\n\n");
  let windows = emb.embed_windows(&doc).expect("embed_windows");
  let mut identical = 0usize;
  for (i, a) in windows.iter().enumerate() {
    for b in &windows[i + 1..] {
      if doc[a.byte_range()] != doc[b.byte_range()] {
        continue;
      }
      identical += 1;
      assert_ne!(a.ordinal(), b.ordinal(), "identical text, same ordinal");
      assert_ne!(a.byte_range(), b.byte_range(), "identical text, same range");
      assert_eq!(
        a.token_count(),
        b.token_count(),
        "identical text, same count"
      );
      let delta = max_abs_delta(a.embedding().as_slice(), b.embedding().as_slice());
      assert!(
        delta <= TOL,
        "identical window text embedded differently (max |Δ| = {delta:e})"
      );
    }
  }
  assert!(
    identical > 0,
    "the fixture planned no two byte-identical windows: {:?}",
    windows
      .iter()
      .map(|w| doc[w.byte_range()].len())
      .collect::<Vec<_>>()
  );
}

/// Contentless nonempty text is ONE window spanning the whole input, embedding
/// to exactly what `embed` returns for it — the whole-input fallback chunk, the
/// same `token_ids` ∘ `embed_tokenized` call on the same bytes.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn contentless_text_is_one_whole_input_window_matching_embed() {
  let emb = embedder();
  for text in ["   ", "\n\n\t \u{a0}"] {
    let windows = emb
      .embed_windows(text)
      .unwrap_or_else(|e| panic!("embed_windows {text:?}: {e}"));
    assert_eq!(windows.len(), 1, "{text:?}: one whole-input window");
    let w = &windows[0];
    assert_eq!(w.ordinal(), 0);
    assert_eq!(w.byte_range(), 0..text.len(), "{text:?}: the whole input");
    let direct = emb
      .embed(text)
      .unwrap_or_else(|e| panic!("embed {text:?}: {e}"));
    let delta = max_abs_delta(w.embedding().as_slice(), direct.as_slice());
    assert!(
      delta <= TOL,
      "{text:?}: the fallback window is not embed's own answer (max |Δ| = {delta:e})"
    );
  }
}

/// A text that fits one window is one window over `[0, text.len())`, and its
/// embedding is `embed`'s — the equality `embed_long`'s single-window
/// short-circuit rests on.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn single_window_text_is_one_window_matching_embed() {
  let emb = embedder();
  let text = "a compact sentence that fits comfortably inside one window";
  let windows = emb.embed_windows(text).expect("embed_windows");
  assert_eq!(windows.len(), 1);
  assert_eq!(windows[0].byte_range(), 0..text.len());
  let direct = emb.embed(text).expect("embed");
  assert!(
    max_abs_delta(windows[0].embedding().as_slice(), direct.as_slice()) <= TOL,
    "the single window must be embed's own answer"
  );
  let pooled = emb.embed_long(text).expect("embed_long");
  assert!(
    max_abs_delta(pooled.as_slice(), direct.as_slice()) <= TOL,
    "embed_long must still be embed on a single-window text"
  );
}

/// `embed_windows` refuses exactly what `embed_long` refuses, with the same
/// error and in the same order — they share one planning-and-prediction pass, so
/// a caller can switch between them without re-learning the failure modes.
///
/// The cases walk the whole refusal set the two share: the empty string, the
/// input-byte gate, the over-budget window, the prediction cap (twice — windit's
/// own raise and the repaired-list re-check), and a contentless run past
/// `MAX_TOKENS`.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn refusals_are_identical_to_embed_long() {
  let emb = embedder();
  let doc = long_document();
  let big = "x".repeat(2 * 1024 * 1024);
  let spaces = " ".repeat(100_000);
  let cases: [(&str, &str, LongTextOptions); 6] = [
    ("empty", "", LongTextOptions::new()),
    (
      "input too large",
      big.as_str(),
      LongTextOptions::new().with_max_input_bytes(1 << 10),
    ),
    (
      "window over budget",
      "any text",
      LongTextOptions::from(WindowOptions::new(MAX_TOKENS + 1)),
    ),
    (
      "cap zero",
      "   ",
      LongTextOptions::from(WindowOptions::new(MAX_TOKENS).with_max_windows(0)),
    ),
    (
      "cap below the plan",
      doc.as_str(),
      LongTextOptions::from(WindowOptions::new(64).with_max_windows(2)),
    ),
    (
      "contentless over budget",
      spaces.as_str(),
      LongTextOptions::new(),
    ),
  ];
  for (label, text, opts) in cases {
    let long = emb
      .embed_long_with(text, &opts)
      .err()
      .unwrap_or_else(|| panic!("{label}: embed_long_with was expected to refuse"));
    let windows = emb
      .embed_windows_with(text, &opts)
      .err()
      .unwrap_or_else(|| panic!("{label}: embed_windows_with was expected to refuse"));
    assert_eq!(
      format!("{windows:?}"),
      format!("{long:?}"),
      "{label}: the two doors refuse differently"
    );
  }
  // The empty string is `embed`'s own contract, named rather than only compared.
  assert!(matches!(emb.embed_windows(""), Err(Error::EmptyText)));
}

/// `max_windows` is a bound on the returned windows, not a truncation: under a
/// cap the plan fits, every window comes back; past it, the call refuses.
#[test]
#[ignore = "requires local granite model (EMBEDKIT_TEST_MODELS)"]
fn max_windows_bounds_the_returned_windows() {
  let emb = embedder();
  let doc = long_document();
  let geometry = WindowOptions::new(128);
  let planned = emb
    .embed_windows_with(&doc, &LongTextOptions::from(geometry))
    .expect("uncapped")
    .len();
  assert!(planned > 2, "the fixture must plan several windows");
  let at_cap = emb
    .embed_windows_with(
      &doc,
      &LongTextOptions::from(geometry.with_max_windows(planned)),
    )
    .expect("a cap at the plan admits it");
  assert_eq!(at_cap.len(), planned, "an exact cap returns every window");
  let err = emb
    .embed_windows_with(
      &doc,
      &LongTextOptions::from(geometry.with_max_windows(planned - 1)),
    )
    .unwrap_err();
  assert!(
    matches!(
      err,
      Error::Windowing(coremlit::embeddings::granite::error::WinditError::TooManyWindows { .. })
    ),
    "expected TooManyWindows, got {err:?}"
  );
}
