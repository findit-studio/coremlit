use super::*;
use crate::embeddings::clap::{
  aggregate::{EmaRenormalized, aggregate},
  embedding::{EMBEDDING_DIM, Embedding},
  error::WinditError,
  window::{Span, WINDOW_SAMPLES},
};

/// A unit-norm window embedding along `±axis i`, spanning `real_len` real
/// samples from `start` (so `coverage == real_len / 480_000`).
fn axis(i: usize, sign: f32, start: usize, real_len: usize) -> WindowEmbedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = sign;
  let e = Embedding::from_slice_normalizing(&v).unwrap();
  WindowEmbedding::new(e, Span::new(start, real_len, WINDOW_SAMPLES))
}

/// Four windows on four orthogonal axes at a 240 000-sample hop, the last a
/// ragged 120 000-sample tail — the geometry `WindowPlan` produces for a 22.5 s
/// clip.
fn stream() -> Vec<WindowEmbedding> {
  (0..4)
    .map(|i| {
      axis(
        i,
        1.0,
        i * 240_000,
        if i == 3 { 120_000 } else { WINDOW_SAMPLES },
      )
    })
    .collect()
}

#[test]
fn window_i_is_the_ema_aggregate_of_the_prefix_through_i() {
  // The contract that makes `VectorEma` the STREAMING SIBLING of
  // `EmaRenormalized` rather than merely a filter with a similar name: at the
  // same alpha, output window `i` carries the direction the aggregate folds over
  // `[0..=i]`. Not bit-exact — the aggregate materializes each weight and folds
  // with Neumaier compensation where this carries a two-term recurrence — so
  // this is a tolerance, and it is checked at every prefix, not just the last.
  let windows = stream();
  for alpha in [0.25_f64, 0.5, 0.7] {
    let smoothed = smooth(&VectorEma::new(alpha), &windows).unwrap();
    assert_eq!(smoothed.len(), windows.len());
    for i in 0..windows.len() {
      let folded = aggregate(&EmaRenormalized::new(alpha), &windows[..=i]).unwrap();
      assert!(
        smoothed[i].value().is_close(&folded, 1e-6),
        "alpha {alpha}, window {i}: the streamed direction left the prefix aggregate"
      );
    }
  }
}

#[test]
fn smoothing_preserves_every_span() {
  // Span-preserving is the whole reason this tier exists: the output stream must
  // stay aligned with the input windows, ragged tail included.
  let windows = stream();
  let smoothed = smooth(&VectorEma::new(0.4), &windows).unwrap();
  let got: Vec<Span> = smoothed.iter().map(|w| w.span()).collect();
  let want: Vec<Span> = windows.iter().map(|w| w.span()).collect();
  assert_eq!(got, want);
  assert_eq!(got[3].coverage(), 0.25, "the ragged tail kept its coverage");
}

#[test]
fn alpha_one_passes_through_and_alpha_zero_holds_the_seed() {
  let windows = stream();

  let passed = smooth(&VectorEma::new(1.0), &windows).unwrap();
  for (out, inp) in passed.iter().zip(windows.iter()) {
    assert!(
      out.value().is_close(inp.value(), 1e-6),
      "alpha = 1 must emit each window unchanged"
    );
  }

  let held = smooth(&VectorEma::new(0.0), &windows).unwrap();
  for out in &held {
    assert!(
      out.value().is_close(windows[0].value(), 1e-6),
      "alpha = 0 must hold the seed direction"
    );
  }
}

#[test]
fn streaming_and_batch_agree() {
  // The batch convenience is a fresh smoother driven over the slice; a caller
  // that owns its windows drives one itself and sheds the per-window clone. The
  // two paths must return the same stream, or the documented escape hatch is a
  // different filter.
  let windows = stream();
  let batch = smooth(&VectorEma::new(0.6), &windows).unwrap();

  let mut smoother = VectorEma::new(0.6).smoother();
  let mut streamed = Vec::new();
  for w in windows {
    streamed.push(smoother.push(w).unwrap());
  }

  assert_eq!(batch.len(), streamed.len());
  for (b, s) in batch.iter().zip(streamed.iter()) {
    assert_eq!(b.span(), s.span());
    assert!(b.value().is_close(s.value(), 0.0), "batch != streaming");
  }
}

#[test]
fn a_fresh_call_restarts_the_average() {
  // `smooth` is a batch convenience, not an incremental decoder: chunk-by-chunk
  // calls are NOT one whole-stream call. Pinned so the doc claim cannot rot.
  let windows = stream();
  let whole = smooth(&VectorEma::new(0.5), &windows).unwrap();
  let second_half = smooth(&VectorEma::new(0.5), &windows[2..]).unwrap();
  assert!(
    !whole[2].value().is_close(second_half[0].value(), 1e-6),
    "a fresh call must NOT carry the running average across it"
  );
}

#[test]
fn identity_is_the_no_rewrite_baseline() {
  let windows = stream();
  let out = smooth(&Identity, &windows).unwrap();
  for (o, i) in out.iter().zip(windows.iter()) {
    assert_eq!(o.span(), i.span());
    assert!(o.value().is_close(i.value(), 0.0));
  }
}

#[test]
fn empty_input_smooths_to_an_empty_stream() {
  // Deliberately unlike `aggregate`, which refuses an empty slice with
  // `EmptyWindows`: a fold of nothing has no direction, a rewrite of nothing is
  // nothing.
  let out = smooth(&VectorEma::new(0.5), &[]).unwrap();
  assert!(out.is_empty());
}

#[test]
fn a_cancelled_accumulator_reaches_clap_error_taxonomy() {
  // The one windit refusal clap's fixed-width, always-unit `Embedding` can
  // actually provoke: at alpha = 0.5 the second window exactly cancels the seed,
  // so the determinacy gate has no direction to report. It must arrive as
  // clap's `Windowing`, not as a bare windit error and not as a fabricated
  // direction.
  let windows = [
    axis(0, 1.0, 0, WINDOW_SAMPLES),
    axis(0, -1.0, 240_000, WINDOW_SAMPLES),
  ];
  let err = smooth(&VectorEma::new(0.5), &windows).unwrap_err();
  assert!(
    matches!(err, Error::Windowing(WinditError::NonFinite)),
    "expected Windowing(NonFinite) from the determinacy gate, got {err:?}"
  );
}

#[test]
fn the_smoother_is_reachable_from_the_flat_clap_surface() {
  // #98's consumer imports through `embeddings::clap`, not through `windit`.
  // This is the re-export itself under test: it names the vector smoother and
  // both traits at the flat path the aggregation half already uses.
  use crate::embeddings::clap::{SmoothPolicy as _, Smoother as _, VectorEma, smooth};

  let windows = stream();
  let out = smooth(&VectorEma::new(0.5), &windows).unwrap();
  assert_eq!(out.len(), windows.len());

  let mut s = VectorEma::new(0.5).smoother();
  let first = s.push(windows[0].clone()).unwrap();
  assert!(first.value().is_close(out[0].value(), 0.0));
  s.reset();
  let reseeded = s.push(windows[1].clone()).unwrap();
  assert!(
    reseeded.value().is_close(windows[1].value(), 1e-6),
    "reset must return the smoother to its unseeded state"
  );
}
