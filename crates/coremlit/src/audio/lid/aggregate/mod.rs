//! Per-window score aggregation for long clips: [`ScorePooling`] +
//! [`aggregate_windows`], folding a clip's per-window log-probability rows into
//! one clip-level [`LogProbabilities`].
//!
//! # A third domain, and why neither neighbour's answer transfers
//!
//! `windit`'s aggregation engine is not used: its built-ins are renormalizing
//! unit-vector policies, which is the embedding domain, not this one.
//! `audio::ced`'s Mean/Max is not copied either: CED emits 527 INDEPENDENT
//! sigmoids, so "mean" there has exactly one reading. This graph's last op is a
//! log-softmax over [`NUM_LANGUAGES`] MUTUALLY EXCLUSIVE classes, a row that
//! already sums to 1 under `exp` — so "mean" is ambiguous, and the two readings
//! are different operations that give different answers:
//!
//! - the **linear opinion pool** ([`ScorePooling::MeanProbability`]) averages
//!   the distributions, and reports what fraction of the clip each language
//!   accounts for;
//! - the **logarithmic opinion pool** ([`ScorePooling::MeanLogProbability`])
//!   averages the log-probabilities — a renormalized geometric mean — and
//!   treats the windows as independent evidence about ONE language, so a
//!   language any window rejects confidently is rejected overall.
//!
//! Both are standard; they answer different questions. Which one this door
//! defaults to was measured, not assumed — see the module docs' "Clips longer
//! than 30 s" section for the table and the two oracles behind it.
//!
//! # Duration weighting is always on
//!
//! Every window contributes in proportion to the REAL audio it saw
//! ([`Span::len`], equivalently [`Span::coverage`] — the window length is fixed
//! within a plan, so the two differ by a constant that cancels). Under
//! [`TailPolicy::SlideBack`] and [`TailPolicy::Drop`] every span is exactly one
//! window long, so the weights are equal and this is precisely the unweighted
//! mean; it only bites under [`TailPolicy::Partial`], where it stops a 0.1 s
//! tail from outvoting a 10 s window. There is no unweighted knob because
//! there is no case where the equal-weight answer is the better one.
//!
//! # Precision
//!
//! The fold runs in **f64** and narrows once at the end. `audio::ced` pins f32
//! accumulation because its aggregation values are golden-pinned upstream;
//! nothing upstream pins these, so the fold uses the wider type — which matters
//! here in a way it does not there, since [`ScorePooling::MeanProbability`]
//! sums `exp` of numbers as low as −25 alongside numbers near 1.
//!
//! A **single** window is returned bit-for-bit unchanged, without folding or
//! renormalizing. That is what makes [`Identifier::identify_long`] on a clip
//! that already fits one window agree exactly — not approximately — with
//! [`Identifier::identify`].
//!
//! [`NUM_LANGUAGES`]: crate::audio::lid::NUM_LANGUAGES
//! [`Span::len`]: crate::audio::lid::Span::len
//! [`Span::coverage`]: crate::audio::lid::Span::coverage
//! [`TailPolicy::SlideBack`]: crate::audio::lid::TailPolicy::SlideBack
//! [`TailPolicy::Drop`]: crate::audio::lid::TailPolicy::Drop
//! [`TailPolicy::Partial`]: crate::audio::lid::TailPolicy::Partial
//! [`Identifier::identify`]: crate::audio::lid::Identifier::identify
//! [`Identifier::identify_long`]: crate::audio::lid::Identifier::identify_long

use core::cmp::Ordering;

use crate::audio::lid::{
  NUM_LANGUAGES,
  error::{Error, Result},
  prediction::{LogProbabilities, WindowLogProbabilities},
};

#[cfg(test)]
mod tests;

/// How a long clip's per-window log-probability rows combine into one
/// clip-level row.
///
/// Every variant returns a row that is still a natural-log distribution (`exp`
/// over it sums to 1), so the result is interchangeable with a single-window
/// row everywhere downstream; the two that are not distributions by
/// construction ([`Self::Max`]) are renormalized to make them so, which cannot
/// change the ranking.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ScorePooling {
  /// Duration-weighted mean **in log space**, renormalized — the logarithmic
  /// opinion pool, equivalently a weighted geometric mean of the per-window
  /// distributions. The default.
  ///
  /// Treats the windows as independent evidence about one language, so the
  /// result is sharper than any single window and a language that any window
  /// rejects confidently stays rejected. That is the right reading when the
  /// question is "what language is this span", which is what a per-speech-span
  /// language node asks; it is the WRONG reading when the clip genuinely
  /// contains more than one language, where it reports neither cleanly (see
  /// [`Self::MeanProbability`], and prefer per-window scores over any
  /// aggregate for code-switching).
  #[default]
  MeanLogProbability,
  /// Duration-weighted mean **in probability space** — the linear opinion
  /// pool, a mixture of the per-window distributions.
  ///
  /// Reads as "what fraction of this clip's duration was each language",
  /// so on a genuinely mixed clip it degrades gracefully into a mixture
  /// instead of a sharpened single answer. It is correspondingly less decisive
  /// on a single-language clip, because one uncertain window drags the whole
  /// mixture toward its own uncertainty.
  MeanProbability,
  /// The highest log-probability each language reached in ANY window,
  /// renormalized back into a distribution.
  ///
  /// "Was this language ever spoken here", not "what language is this" — the
  /// most sensitive to a short passage of a second language, and by the same
  /// token the most sensitive to a single bad window. Duration weighting does
  /// not apply to a maximum, so this is the one variant a window's length does
  /// not influence.
  Max,
  /// Each window casts one vote for its own top language, weighted by that
  /// window's duration; the result is the vote share, in log space.
  ///
  /// Discards ALL magnitude information — a window that is 51 % sure and one
  /// that is 99.9 % sure count the same — which is exactly what makes it
  /// robust to one wildly wrong window and blunt everywhere else. It is also
  /// the only pooling that produces `f32::NEG_INFINITY`, for every language no
  /// window chose (a vote share of exactly zero, whose log is exactly `-∞`);
  /// [`LanguageScore::probability`] maps that to `0.0`.
  ///
  /// [`LanguageScore::probability`]: crate::audio::lid::LanguageScore::probability
  Vote,
}

/// Streaming fold shared by [`aggregate_windows`] and
/// [`Identifier::identify_long`], one window at a time — `identify_long` folds
/// each window's row in and never materializes the per-window vectors, so a
/// long clip retains O([`NUM_LANGUAGES`]) state rather than one 107-float row
/// per window.
///
/// Bit-identical to the batch fold by construction: both drive this same type
/// with the same op sequence over the same window order.
///
/// [`NUM_LANGUAGES`]: crate::audio::lid::NUM_LANGUAGES
/// [`Identifier::identify_long`]: crate::audio::lid::Identifier::identify_long
#[derive(Debug, Clone)]
pub(crate) struct Accumulator {
  pooling: ScorePooling,
  /// The f64 fold state — empty until the first [`Self::push`], then exactly
  /// `NUM_LANGUAGES` long.
  acc: Vec<f64>,
  /// The first window's row, verbatim. Returned unchanged when it is the only
  /// one, which is what makes a one-window fold the bit-exact identity.
  first: Vec<f32>,
  weight_sum: f64,
  count: usize,
}

impl Accumulator {
  /// An empty fold under `pooling`. [`Self::finish`] on it is
  /// [`Error::EmptyWindows`] until at least one window is pushed.
  pub(crate) fn new(pooling: ScorePooling) -> Self {
    Self {
      pooling,
      acc: Vec::new(),
      first: Vec::new(),
      weight_sum: 0.0,
      count: 0,
    }
  }

  /// Fold one window's row in, weighted by `weight_samples` — the real audio
  /// that window covered ([`Span::len`](crate::audio::lid::Span::len)).
  /// [`ScorePooling::Max`] ignores the weight; every other policy is
  /// proportional to it.
  pub(crate) fn push(&mut self, window: &LogProbabilities, weight_samples: usize) {
    let row = window.as_slice();
    let weight = weight_samples as f64;
    if self.count == 0 {
      self.first = row.to_vec();
      let seed = match self.pooling {
        ScorePooling::Max => f64::NEG_INFINITY,
        _ => 0.0,
      };
      self.acc = vec![seed; NUM_LANGUAGES];
    }
    match self.pooling {
      ScorePooling::MeanLogProbability => {
        for (a, &v) in self.acc.iter_mut().zip(row) {
          *a += weight * f64::from(v);
        }
      }
      ScorePooling::MeanProbability => {
        for (a, &v) in self.acc.iter_mut().zip(row) {
          *a += weight * f64::from(v).exp();
        }
      }
      ScorePooling::Max => {
        for (a, &v) in self.acc.iter_mut().zip(row) {
          *a = a.max(f64::from(v));
        }
      }
      ScorePooling::Vote => self.acc[argmax(row)] += weight,
    }
    self.weight_sum += weight;
    self.count += 1;
  }

  /// Finish the fold into one clip-level [`LogProbabilities`].
  ///
  /// A single pushed window is returned verbatim — no divide, no
  /// renormalization, no narrowing — so the long path and the single-shot path
  /// agree bit for bit on a clip that fits one window.
  ///
  /// # Errors
  /// [`Error::EmptyWindows`] if no window was pushed.
  pub(crate) fn finish(self) -> Result<LogProbabilities> {
    if self.count == 0 {
      return Err(Error::EmptyWindows);
    }
    if self.count == 1 {
      return Ok(LogProbabilities::new(self.first));
    }
    let mut acc = self.acc;
    match self.pooling {
      // Mean of the logs, then shift the whole row so it is a distribution
      // again. The shift is constant across languages, so it cannot reorder
      // them; it exists so `probability()` and any downstream `exp` still read
      // as probabilities.
      ScorePooling::MeanLogProbability => {
        for a in &mut acc {
          *a /= self.weight_sum;
        }
        renormalize(&mut acc);
      }
      // The weighted sum of per-window probabilities is itself `weight_sum`
      // (each row sums to 1), so dividing by it yields a distribution directly
      // and no renormalization is needed.
      ScorePooling::MeanProbability | ScorePooling::Vote => {
        for a in &mut acc {
          *a = (*a / self.weight_sum).ln();
        }
      }
      ScorePooling::Max => renormalize(&mut acc),
    }
    // Every arm above leaves values `<= 0`: `renormalize` subtracts a
    // log-sum-exp that is at least the row maximum, and the two `ln` arms
    // divide by a sum that bounds every element. Narrowing to f32 rounds to
    // nearest and so cannot turn a non-positive value positive.
    Ok(LogProbabilities::new(
      acc.into_iter().map(|v| v as f32).collect(),
    ))
  }
}

/// Shift `values` down by their log-sum-exp so `exp` over the row sums to 1.
///
/// Runs the standard max-subtraction form, which is what keeps
/// [`ScorePooling::Max`] finite when the row's maximum is far from zero. The
/// row maximum is never `-∞` in practice (a model row is a log-softmax and its
/// argmax is finite), and an all-`-∞` row is left alone rather than turned into
/// NaN by `-∞ − (−∞)`.
fn renormalize(values: &mut [f64]) {
  let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  if !max.is_finite() {
    return;
  }
  let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
  let log_sum_exp = max + sum.ln();
  for v in values {
    *v -= log_sum_exp;
  }
}

/// Model column of the row's largest value, ties broken by ascending column —
/// the crate's ranking tie-break, so a window's vote agrees with what
/// `top_k(1)` on that same window would have returned.
fn argmax(row: &[f32]) -> usize {
  let mut best = 0;
  for (index, value) in row.iter().enumerate().skip(1) {
    if value.total_cmp(&row[best]) == Ordering::Greater {
      best = index;
    }
  }
  best
}

/// Combine per-window rows into one clip-level [`LogProbabilities`] under
/// `pooling`, weighting each window by the real audio its [`Span`] covered.
///
/// This is the batch form of what [`Identifier::identify_long`] streams, driven
/// by the same private accumulator, so the two agree bit for bit. Reach for it when
/// the per-window rows are already in hand — from
/// [`Identifier::log_probabilities_windows`], or hand-built via
/// [`LogProbabilities::try_from_slice`] with no model at all.
///
/// # Errors
/// [`Error::EmptyWindows`] if `windows` is empty. (Unreachable through
/// `identify_long` — a clip long enough to reach the model always plans at
/// least one span.)
///
/// # Examples
/// ```
/// use coremlit::audio::lid::{
///   LogProbabilities, NUM_LANGUAGES, ScorePooling, Span, WindowLogProbabilities,
///   aggregate_windows,
/// };
///
/// // Two equal-length windows: one sure of column 94, one sure of column 0.
/// let window = 160_000;
/// let row = |hot: usize| {
///   let mut values = vec![-20.0f32; NUM_LANGUAGES];
///   values[hot] = -0.001;
///   LogProbabilities::try_from_slice(&values).expect("valid row")
/// };
/// let windows = vec![
///   WindowLogProbabilities::new(row(94), Span::new(0, window, window)),
///   WindowLogProbabilities::new(row(0), Span::new(window, window, window)),
/// ];
///
/// // A mixture splits the mass; a vote splits it too, but exactly in half and
/// // with every other language at log 0 probability.
/// let mixed = aggregate_windows(ScorePooling::MeanProbability, &windows)?;
/// assert!((mixed.as_slice()[94].exp() - 0.5).abs() < 1e-3);
/// let voted = aggregate_windows(ScorePooling::Vote, &windows)?;
/// assert_eq!(voted.as_slice()[94], 0.5f32.ln());
/// assert_eq!(voted.as_slice()[1], f32::NEG_INFINITY);
/// # Ok::<(), coremlit::audio::lid::Error>(())
/// ```
///
/// [`Span`]: crate::audio::lid::Span
/// [`Identifier::identify_long`]: crate::audio::lid::Identifier::identify_long
/// [`Identifier::log_probabilities_windows`]: crate::audio::lid::Identifier::log_probabilities_windows
pub fn aggregate_windows(
  pooling: ScorePooling,
  windows: &[WindowLogProbabilities],
) -> Result<LogProbabilities> {
  let mut acc = Accumulator::new(pooling);
  for window in windows {
    acc.push(window.value(), window.span().len());
  }
  acc.finish()
}
