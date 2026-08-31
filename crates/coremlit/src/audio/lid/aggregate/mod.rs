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
//! Width alone does not save that sum, though: `exp` underflows to exactly zero
//! below about −744.4 (the log of `f64`'s smallest subnormal), so
//! [`ScorePooling::MeanProbability`] never exponentiates a log-probability on
//! its own. It runs an online **log-sum-exp** — each language's weighted sum is
//! held relative to the largest log-probability that language has been given,
//! and the shift is added back at the end. A finite score therefore keeps a
//! finite pooled value, and its rank, however far below the row maximum it
//! sits; a literal `ln(Σ w·exp(x))` re-emits the whole tail as `-∞` and ranks
//! it by the tie-break, from inputs that were every one of them finite. The two
//! forms agree to f32 on anything this model produces (its measured tail is
//! −37.27) — they part company only where the literal one has already lost the
//! answer.
//!
//! A **single** window is returned bit-for-bit unchanged, without folding or
//! renormalizing. That is what makes [`Identifier::identify_long`] on a clip
//! that already fits one window agree exactly — not approximately — with
//! [`Identifier::identify`].
//!
//! # Totality: a distribution comes back, or an error does
//!
//! [`LogProbabilities`] accepts `-∞`. It has to: that is the exact log of a
//! zero probability, and [`ScorePooling::Vote`] emits it by construction for
//! every language no window chose. So a pooling can be handed rows it cannot
//! fold into a distribution at all. Two windows each certain of a DIFFERENT
//! language, `-∞` everywhere else, are the case: the logarithmic pool is a
//! geometric mean, so a language any window scores at zero is zero in the pool,
//! and between them the two windows zero out every language. The arithmetic is
//! right and the result is not a distribution — its exponentials sum to zero,
//! so its "top" languages are whichever ones the tie-break surfaces, each at
//! probability zero.
//!
//! This module states the remedy as a POSTCONDITION rather than a special case
//! for the one pooling that provoked it: every row [`aggregate_windows`]
//! returns has positive probability mass, and a fold that leaves none is
//! [`Error::ZeroMassAggregate`] instead. It is checked once, on the narrowed
//! row the caller actually receives, so it covers the single-window identity
//! path as well and cannot be missed by a pooling added later. It costs one
//! `exp` (`exp` is monotonic, so the row's total is positive exactly when its
//! maximum's is). It is unreachable from the model:
//! [`Identifier::log_probabilities`] rejects a non-finite score, so no model
//! row carries a `-∞` to propagate.
//!
//! [`NUM_LANGUAGES`]: crate::audio::lid::NUM_LANGUAGES
//! [`Span::len`]: crate::audio::lid::Span::len
//! [`Span::coverage`]: crate::audio::lid::Span::coverage
//! [`TailPolicy::SlideBack`]: crate::audio::lid::TailPolicy::SlideBack
//! [`TailPolicy::Drop`]: crate::audio::lid::TailPolicy::Drop
//! [`TailPolicy::Partial`]: crate::audio::lid::TailPolicy::Partial
//! [`Identifier::identify`]: crate::audio::lid::Identifier::identify
//! [`Identifier::identify_long`]: crate::audio::lid::Identifier::identify_long
//! [`Identifier::log_probabilities`]: crate::audio::lid::Identifier::log_probabilities

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
/// row everywhere downstream; [`Self::Max`], the one variant that is not a
/// distribution by construction, is renormalized to make it so, which cannot
/// change the ranking.
///
/// Where a pooling's honest answer is that EVERY language has probability zero
/// — the logarithmic pool over windows with disjoint supports — there is no
/// distribution to return, and [`aggregate_windows`] refuses with
/// [`Error::ZeroMassAggregate`] rather than hand back a row that ranks
/// arbitrarily. See the module docs' "Totality" section.
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
  ///
  /// Computed as a log-sum-exp, not as a literal sum of `exp`, so a language
  /// far below the row maximum keeps a finite pooled score and its rank rather
  /// than underflowing to `-∞` (module docs, "Precision").
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
  /// [`ScorePooling::MeanProbability`]'s log-sum-exp shift: `shift[j]` is the
  /// largest log-probability any window has given language `j`, and `acc[j]`
  /// the weighted probability sum taken RELATIVE to it. Empty under every
  /// other pooling — none of them exponentiates during the fold, so none of
  /// them needs a shift.
  shift: Vec<f64>,
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
      shift: Vec::new(),
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
      if self.pooling == ScorePooling::MeanProbability {
        self.shift = vec![f64::NEG_INFINITY; NUM_LANGUAGES];
      }
    }
    match self.pooling {
      ScorePooling::MeanLogProbability => {
        for (a, &v) in self.acc.iter_mut().zip(row) {
          *a += weight * f64::from(v);
        }
      }
      // Online weighted log-sum-exp, one running shift per language, so no
      // log-probability is ever exponentiated on its own: `exp` flushes
      // anything below ln(f64::MIN_POSITIVE) ≈ −744.4 to exactly zero, and a
      // literal `Σ w·exp(x)` then re-logs the whole tail as `-∞` — losing an
      // ordering the input row still carried, from finite values throughout.
      ScorePooling::MeanProbability => {
        for ((sum, shift), &v) in self.acc.iter_mut().zip(&mut self.shift).zip(row) {
          let value = f64::from(v);
          if value > *shift {
            // Re-express what is already summed against the new, larger shift.
            // While the shift is still `-∞` the sum is 0, so this is `0 · 0`.
            *sum = *sum * (*shift - value).exp() + weight;
            *shift = value;
          } else if value > f64::NEG_INFINITY {
            // A non-positive difference: `exp` lands in (0, 1] and underflows
            // only where the term genuinely rounds away against the shift.
            // Skipping an exact `-∞`, which contributes nothing anyway, is
            // what keeps `(-∞) − (-∞)` = NaN unreachable.
            *sum += weight * (value - *shift).exp();
          }
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
  /// [`Error::EmptyWindows`] if no window was pushed;
  /// [`Error::ZeroMassAggregate`] if the fold left no probability mass at all
  /// (module docs, "Totality").
  pub(crate) fn finish(self) -> Result<LogProbabilities> {
    if self.count == 0 {
      return Err(Error::EmptyWindows);
    }
    let Self {
      pooling,
      mut acc,
      shift,
      first,
      weight_sum,
      count,
    } = self;
    let values = if count == 1 {
      first
    } else {
      match pooling {
        // Mean of the logs, then shift the whole row so it is a distribution
        // again. The shift is constant across languages, so it cannot reorder
        // them; it exists so `probability()` and any downstream `exp` still
        // read as probabilities.
        ScorePooling::MeanLogProbability => {
          for a in &mut acc {
            *a /= weight_sum;
          }
          renormalize(&mut acc);
        }
        // Close the log-sum-exp the fold has been running: the mixture's
        // log-probability is `shift + ln(Σ w·exp(x − shift) / weight_sum)`.
        // A mixture of distributions is a distribution already, so there is
        // nothing to renormalize.
        ScorePooling::MeanProbability => {
          for (a, &s) in acc.iter_mut().zip(&shift) {
            *a = s + (*a / weight_sum).ln();
          }
        }
        // The votes cast sum to `weight_sum` by construction, so dividing by
        // it yields the vote share — a distribution — directly.
        ScorePooling::Vote => {
          for a in &mut acc {
            *a = (*a / weight_sum).ln();
          }
        }
        ScorePooling::Max => renormalize(&mut acc),
      }
      acc.into_iter().map(|v| v as f32).collect()
    };
    // Every arm above leaves values `<= 0`: `renormalize` subtracts a
    // log-sum-exp that is at least the row maximum; `Vote` divides by a sum
    // that bounds every element; and `MeanProbability` adds a shift that is
    // itself `<= 0` to the log of a ratio whose numerator each term bounds by
    // its own weight. Narrowing to f32 rounds to nearest and so cannot turn a
    // non-positive value positive.
    //
    // What none of that establishes is that anything is LEFT — a pooling can
    // be handed rows whose honest pool is zero everywhere. Stated here, once,
    // as a postcondition on the row the caller receives rather than as a guard
    // inside the one pooling that provokes it: it therefore also covers the
    // single-window identity path above, and a pooling added later inherits it
    // without having to remember to.
    if !has_probability_mass(&values) {
      return Err(Error::ZeroMassAggregate(pooling));
    }
    Ok(LogProbabilities::new(values))
  }
}

/// Shift `values` down by their log-sum-exp so `exp` over the row sums to 1.
///
/// Runs the standard max-subtraction form, which is what keeps
/// [`ScorePooling::Max`] finite when the row's maximum is far from zero. The
/// row maximum is never `-∞` in practice (a model row is a log-softmax and its
/// argmax is finite), and an all-`-∞` row is left alone rather than turned into
/// NaN by `-∞ − (−∞)`. It stays infallible and total for that reason: the row
/// it declines to touch is not a distribution, and refusing it is
/// [`Accumulator::finish`]'s postcondition, which catches it for every pooling
/// rather than only for the two that renormalize.
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

/// Whether `row` carries any probability mass — whether `exp` over it sums to
/// something greater than zero.
///
/// One `exp` wide, and exact: `exp` is monotonic, so the row's total is
/// positive exactly when its LARGEST entry exponentiates to a positive number.
/// `-∞` in every column is the case this exists for; a row of finite but
/// enormous negatives (an f32 column may hold −3.4e38) is the same degenerate
/// answer and the same test rejects it.
fn has_probability_mass(row: &[f32]) -> bool {
  let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  f64::from(max).exp() > 0.0
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
/// [`Error::ZeroMassAggregate`] if the pooled row assigns probability zero to
/// every language, which is not a distribution and cannot be ranked — see the
/// module docs' "Totality" section for when a pooling honestly answers that.
/// Also unreachable through `identify_long`: a model row is all-finite, so no
/// `-∞` enters the fold.
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
