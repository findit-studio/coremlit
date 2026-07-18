//! Zero-shot scoring: rank text labels against an audio [`Embedding`] by
//! audio↔text cosine, optionally scaled by the CLAP logit scale.
//!
//! The encoders are separate, so scoring is sans-model: the caller embeds the
//! audio (one window, or an aggregate — see
//! [`AggregatePolicy`](crate::aggregate::AggregatePolicy)) and each candidate
//! label (via [`TextEncoder::embed`](crate::TextEncoder::embed)), then passes the
//! precomputed embeddings here as [`TextAnchor`]s. [`score_windows`] applies the
//! same over a slice of [`WindowEmbedding`]s so per-window scores are exposed for
//! caller-side smoothing or voting.

use crate::{embedding::Embedding, window::WindowEmbedding};

#[cfg(test)]
mod tests;

/// CLAP **audio-side** logit scale, `logit_scale_a.exp()`, pinned from the
/// `laion/clap-htsat-unfused` config (`@8fa0f1c6…`,
/// `logit_scale_a.exp() == 18.661177` as f32).
///
/// Zero-shot audio classification scales `audio·text` by this before a softmax
/// over labels (HF `ClapModel.logits_per_audio`). Because it is a positive
/// constant it does not change the *ranking* — only the magnitude, and any
/// downstream softmax's temperature.
pub const LOGIT_SCALE_AUDIO: f32 = 18.661177;

/// CLAP **text-side** logit scale, `logit_scale_t.exp() == 14.285714` (f32) —
/// the counterpart used for `logits_per_text`. clapkit scores audio against text
/// labels, so [`LOGIT_SCALE_AUDIO`] is the one [`ScoreMode::LogitScaled`] applies;
/// this is provided for parity with the model config.
pub const LOGIT_SCALE_TEXT: f32 = 14.285714;

/// How a zero-shot score is derived from the audio↔text cosine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ScoreMode {
  /// Raw cosine similarity in roughly `[-1, 1]` — directly textclap-comparable
  /// (its `classify` returns the same dot product).
  Cosine,
  /// Cosine × [`LOGIT_SCALE_AUDIO`]: the CLAP logit, ready for a softmax over
  /// labels. Monotonic in cosine, so the ranking is identical to
  /// [`Self::Cosine`]; only the score magnitude differs.
  LogitScaled,
}

impl ScoreMode {
  /// Apply this mode to a raw cosine.
  #[inline]
  fn apply(self, cosine: f32) -> f32 {
    match self {
      Self::Cosine => cosine,
      Self::LogitScaled => cosine * LOGIT_SCALE_AUDIO,
    }
  }
}

/// A candidate label paired with its precomputed text [`Embedding`] — the input
/// unit to [`score()`].
///
/// Borrowing keeps scoring allocation-free and lets the label flow straight into
/// the returned [`LabeledScore`].
#[derive(Debug, Clone, Copy)]
pub struct TextAnchor<'a> {
  label: &'a str,
  embedding: &'a Embedding,
}

impl<'a> TextAnchor<'a> {
  /// Pair `label` with its precomputed text embedding.
  pub const fn new(label: &'a str, embedding: &'a Embedding) -> Self {
    Self { label, embedding }
  }

  /// The candidate label.
  #[inline]
  pub const fn label(&self) -> &'a str {
    self.label
  }

  /// The label's precomputed text embedding.
  #[inline]
  pub const fn embedding(&self) -> &'a Embedding {
    self.embedding
  }
}

/// One scored label, borrowing its text from the [`TextAnchor`] it came from.
#[derive(Debug, Clone, Copy)]
pub struct LabeledScore<'a> {
  label: &'a str,
  score: f32,
}

impl<'a> LabeledScore<'a> {
  /// The scored label.
  #[inline]
  pub const fn label(&self) -> &'a str {
    self.label
  }

  /// The score, in the units of the [`ScoreMode`] used.
  #[inline]
  pub const fn score(&self) -> f32 {
    self.score
  }

  /// Copy into an owned [`LabeledScoreOwned`] for storage or cross-thread send.
  pub fn to_owned(&self) -> LabeledScoreOwned {
    LabeledScoreOwned {
      label: self.label.to_string(),
      score: self.score,
    }
  }
}

/// Owned counterpart of [`LabeledScore`] — owns its label string.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledScoreOwned {
  label: String,
  score: f32,
}

impl LabeledScoreOwned {
  /// The scored label.
  #[inline]
  pub fn label(&self) -> &str {
    &self.label
  }

  /// The score.
  #[inline]
  pub const fn score(&self) -> f32 {
    self.score
  }

  /// Consume self, returning the owned label.
  #[inline]
  pub fn into_label(self) -> String {
    self.label
  }
}

/// Score `audio` against each [`TextAnchor`], returning results sorted
/// descending by score.
///
/// Ties keep input order (the sort is stable), so equal-scoring labels stay in
/// the order the caller supplied them. An empty `anchors` yields an empty vec.
/// The score's units follow `mode` (raw cosine or CLAP logit); the ordering is
/// identical either way (see [`ScoreMode::LogitScaled`]).
#[must_use]
pub fn score<'a>(
  audio: &Embedding,
  anchors: &[TextAnchor<'a>],
  mode: ScoreMode,
) -> Vec<LabeledScore<'a>> {
  let mut out: Vec<LabeledScore<'a>> = anchors
    .iter()
    .map(|a| LabeledScore {
      label: a.label(),
      score: mode.apply(audio.cosine(a.embedding())),
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

/// Per-window zero-shot scores — [`score()`] applied to each
/// [`WindowEmbedding`]'s embedding, one ranked `Vec` per window.
///
/// This is the exposed per-window score surface: callers can smooth or vote
/// across windows without a second aggregation seam (the deliberate cut in the
/// spec amendment).
#[must_use]
pub fn score_windows<'a>(
  windows: &[WindowEmbedding],
  anchors: &[TextAnchor<'a>],
  mode: ScoreMode,
) -> Vec<Vec<LabeledScore<'a>>> {
  windows
    .iter()
    .map(|w| score(w.embedding(), anchors, mode))
    .collect()
}
