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
//! # A row's own scale is not evidence
//!
//! A row's RATIOS are what it says about the languages; its overall scale says
//! nothing. Straight off the graph a row is a log-softmax and `exp` over it
//! should sum to 1 — and it does not, quite: fp16 arithmetic leaves it up to
//! as much as 7.7e-3 away from 1 on [`ComputeUnits::CpuOnly`] and 1.5e-4 with
//! the ANE — on either side of it, the deviation being signed. That gap is a
//! fact about how the row was computed, not about what was spoken.
//!
//! Folded raw it becomes a per-window WEIGHT, because three of the four
//! poolings fold VALUES. Two equal 160 000-sample windows, each one-hot and so
//! each perfectly certain — one on column 0 at `ln(0.99235)`, one on column 1
//! at exactly `0.0` — used to pool to p(0) = 0.498080 against p(1) = 0.501920
//! under [`ScorePooling::MeanProbability`], and to the same split under
//! [`ScorePooling::Max`]: the clip went to whichever window's row had rounded
//! better. The pooled row's mass was 1.0000000086, four orders of magnitude
//! inside `MAX_MASS_DEVIATION`, so the postcondition could not see it.
//!
//! How much it was worth on real audio depends entirely on the compute unit,
//! which is why the door's measurement tables (`audio::lid`'s own module docs)
//! did not move when this was fixed. Over `MAX_MASS_DEVIATION`'s 192-fold
//! sweep, normalizing at the door
//! changed the pooled row by at most 3.8e-6 nats under `ComputeUnits::All` and
//! `CpuAndGpu`, and 2.6e-4 on the ANE — but by 1.0e-2 on
//! [`ComputeUnits::CpuOnly`], whose rows carry the 7.7e-3 gap. No fold in
//! that sweep changed its top-1 language, and [`ScorePooling::MeanLogProbability`]
//! and [`ScorePooling::Vote`] came back bit-identical on every one of the 192.
//!
//! Every row is therefore made a distribution AT THE DOOR, in `push`, under the
//! same shift the fold's exit uses — one `DistributionShift`, called from both
//! ends, rather than the arithmetic restated per caller. Rescaling one window's
//! row by a constant now moves the pooled row by no more than the f32 narrowing
//! at the exit. Measured over two overlapping-support rows with one rescaled by
//! `ln(0.99235)`, the largest per-column gap that rescale opens:
//!
//! | pooling                              | folding raw rows | folding normalized rows |
//! |--------------------------------------|------------------|-------------------------|
//! | [`ScorePooling::MeanLogProbability`] | 0.0              | 0.0                     |
//! | [`ScorePooling::MeanProbability`]    | 1.28e-3          | 0.0                     |
//! | [`ScorePooling::Max`]                | 4.04e-3          | 5.96e-8                 |
//! | [`ScorePooling::Vote`]               | 0.0              | 0.0                     |
//!
//! The logarithmic pool was already immune, and for a reason rather than by
//! luck: a constant added to one row adds a constant to the mean of the logs,
//! and the closing renormalization takes exactly that constant back off.
//!
//! [`ScorePooling::Vote`] is the one pooling still folding the RAW row, and is
//! exactly invariant either way. Its ballot is an `argmax` — a comparison, not
//! arithmetic — so a shift cannot reorder it, and shifting first could only
//! round two distinct values onto one and hand the outcome to the ranking
//! tie-break. `the_fold_is_invariant_to_a_rows_own_scale` holds the table.
//!
//! # Totality: a distribution comes back, or an error does
//!
//! [`LogProbabilities`] accepts `-∞`. It has to: that is the exact log of a
//! zero probability, and [`ScorePooling::Vote`] emits it for every language no
//! window chose. So a pooling can be handed rows it cannot fold into a
//! distribution at all. The module answers that with one PRECONDITION on every
//! window and one POSTCONDITION on every fold, both stated here rather than
//! inside whichever pooling last provoked them — so a pooling added later
//! inherits both without having to remember to, and so does a future edit to
//! one of these four.
//!
//! **Precondition: a window must be normalizable.** The fold asks one thing of
//! every row it is handed — that `DistributionShift` can turn it into a
//! distribution — and that is exactly that it has a **finite maximum**, a bound
//! that is finite AND that the whole row actually sits under. Three rows fail
//! it, and none is one any pooling could have folded:
//!
//! - `-∞` in every column says no language is possible. It is not evidence
//!   about which language was spoken, and each pooling would mishandle it in
//!   its own way: the logarithmic pool zeroes the whole clip out; the linear
//!   pool skips all of its terms (that is what keeps `(-∞) − (-∞)` unreachable)
//!   while still counting its duration in the denominator, so every other
//!   window comes out diluted; and [`ScorePooling::Vote`] casts its ballot for
//!   whatever column the ranking tie-break surfaces, handing a share of the
//!   clip to a language nothing chose.
//! - `+∞` anywhere is not a log-probability row at all: `exp` over it sums to
//!   `∞`, and no constant makes that a distribution. It used to be let through
//!   here and left to the postcondition, which caught it under only two of the
//!   four poolings ([`Error::NotADistribution`], carrying a mass of `∞`).
//!   [`ScorePooling::Vote`] returned a clean-looking distribution putting the
//!   whole clip on whichever column held the `∞`; [`ScorePooling::MeanProbability`]
//!   returned a row of 107 NaNs, which the postcondition cannot see because
//!   every comparison against NaN is false; and a LONE `+∞` window took the
//!   identity path back to the caller verbatim, as a [`LogProbabilities`]
//!   holding a positive value.
//! - A NaN anywhere is under no bound at all, `+∞` included, so there is no
//!   shift and no ranking. It survived the `+∞` round because that round's
//!   guard folded the row with `f32::max`, whose documented `maxNum` semantics
//!   are to IGNORE a NaN operand: `[-1, NaN, -1, …]` reported a maximum of `-1`
//!   and was let through. Past it the four poolings lost it in three different
//!   directions, and the postcondition could see only one of them.
//!   [`ScorePooling::MeanLogProbability`] spread the NaN over all 107 columns —
//!   one NaN anywhere in `DistributionShift`'s sum poisons the whole shift —
//!   for a mass of NaN. [`ScorePooling::Max`] and [`ScorePooling::MeanProbability`]
//!   DROPPED the window instead, `f64::max` ignoring the NaN and the
//!   log-sum-exp skipping it exactly as it skips a `-∞`: a NaN window beside a
//!   real one answered from the real one alone at mass 1, and a clip of nothing
//!   but NaN came back as [`Error::ZeroMassAggregate`], a refusal naming the
//!   wrong reason. [`ScorePooling::Vote`] handed the window's whole ballot to
//!   the NaN's column, also at mass 1, because `total_cmp` ranks a NaN above
//!   every real value. And a LONE NaN window took the identity path back to the
//!   caller verbatim under all four — a [`LogProbabilities`] holding a NaN,
//!   which this type's invariant forbids, whose `top_k` reports the NaN's
//!   language first, and which the postcondition does not apply to at all.
//!
//! [`aggregate_windows`] refuses all three with [`Error::UnnormalizableWindow`],
//! naming the window, before any of that. It costs one pass and no `exp`.
//!
//! **What the precondition must NOT decide is a row's SCALE.**
//! `[-800, -801, …, -906]` and `[0, -1, …, -106]` differ by exactly 800 in
//! every column, so no probability ratio differs between them and they
//! normalize to the identical distribution. An earlier form of this guard —
//! `exp(max) > 0.0`, i.e. "the row's total is positive" — refused the first and
//! folded the second, because `exp` underflows f64 to exactly zero below
//! −744.44. That was this module's own "a row's own scale is not evidence"
//! leak, taken out of the fold and left standing in the door in front of it.
//! `the_door_is_invariant_to_a_rows_own_scale` holds the property now; what
//! makes it true is that `DistributionShift` forms `(v − max)` before anything
//! else, which is well-conditioned for any finite maximum.
//!
//! **Postcondition: what comes back sums to 1.** Two windows each certain of a
//! DIFFERENT language, `-∞` everywhere else, both pass the precondition and
//! still have no pool: the logarithmic pool is a geometric mean, so a language
//! any window scores at zero is zero in the pool, and between them the two
//! windows zero out every language. The arithmetic is right and the result is
//! not a distribution — its exponentials sum to zero, so its "top" languages
//! are whichever ones the tie-break surfaces, each at probability zero. That is
//! [`Error::ZeroMassAggregate`], and it is the one deviation from 1 that is an
//! honest answer rather than a defect. Any OTHER deviation is a defect, and is
//! [`Error::NotADistribution`] carrying the mass the fold actually left — the
//! tolerance and the measurements behind it are on `MAX_MASS_DEVIATION`.
//!
//! It is written as the predicate that ACCEPTS — `|mass − 1| <=
//! MAX_MASS_DEVIATION`, returned from — rather than as the `>` that refuses,
//! because those two are the same test only over an ORDERED domain and f64 is
//! not one. Every ordered comparison against a NaN is false, so the refusing
//! form reads a NaN mass as "not outside the tolerance" and hands the caller
//! 107 NaNs; the accepting form reads it as "not inside", and it lands in the
//! refusal. A postcondition stated once so that whatever is added later
//! inherits it has to be TOTAL, or what it is inherited by is a hole.
//!
//! The postcondition applies to a FOLDED row only. A lone window is returned
//! verbatim, and holding a row this module did not compute to "sums to 1" would
//! break the [`Identifier::identify_long`] == [`Identifier::identify`] promise
//! on real audio rather than only in principle: a model row's own mass is off
//! by up to 7.7e-3 on [`ComputeUnits::CpuOnly`] and 1.5e-4 on the ANE. The
//! precondition covers that path instead, which is the property a row this
//! module did not compute can be held to — it normalizes, so it ranks. A lone
//! window written at a very low scale is therefore returned verbatim with an
//! f64 mass of exactly zero, and that is the right answer: it is what
//! [`Identifier::identify`] returns for the same row.
//!
//! Both are unreachable from the model. [`Identifier::log_probabilities`]
//! rejects a non-finite score, so no window `identify_long` folds can have a
//! non-finite maximum, and a log-softmax row's largest entry is at least
//! `ln(1/107)`.
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
//! [`ComputeUnits::CpuOnly`]: crate::ComputeUnits::CpuOnly

use core::cmp::Ordering;

use crate::audio::lid::{
  NUM_LANGUAGES,
  error::{Error, NotADistribution, Result},
  prediction::{LogProbabilities, WindowLogProbabilities},
};

#[cfg(test)]
mod tests;

/// How a long clip's per-window log-probability rows combine into one
/// clip-level row.
///
/// Every variant returns a row that is still a natural-log distribution (`exp`
/// over it sums to 1), so the result is interchangeable with a single-window
/// row everywhere downstream. Only [`Self::Vote`] is a distribution by
/// construction, from the shares it divides; the other three close with a
/// renormalization, which is a constant shift and so cannot change the ranking.
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
  /// than underflowing to `-∞` (module docs, "Precision"), and renormalized at
  /// the close. A mixture of distributions is a distribution already, and the
  /// rows folded here are made distributions before they are folded, so the
  /// closing renormalization is very nearly a no-op; what it buys is that the
  /// answer is one whether or not the mixture arithmetic left it one.
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
  /// largest NORMALIZED log-probability any window has given language `j`
  /// (normalized because that is what [`Self::push`] folds), and `acc[j]` the
  /// weighted probability sum taken RELATIVE to it. Empty under every other
  /// pooling — none of them exponentiates during the fold, so none of them
  /// needs a shift.
  shift: Vec<f64>,
  /// The first window's row, verbatim — RAW, not the normalized row
  /// [`Self::push`] folds. Returned unchanged when it is the only one, which is
  /// what makes a one-window fold the bit-exact identity.
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
  ///
  /// The row is made a distribution BEFORE it is folded, so a row's own overall
  /// scale — which is fp noise, not evidence — cannot act as a second weight
  /// beside `weight_samples` (module docs, "A row's own scale is not
  /// evidence"). [`ScorePooling::Vote`] is the exception, and the first row is
  /// kept raw for [`Self::finish`]'s identity path; both are argued at their
  /// sites below.
  ///
  /// # Errors
  /// [`Error::UnnormalizableWindow`], carrying the window's position, if the
  /// row has no finite maximum — `-∞` throughout, which rules every language
  /// out; `+∞` anywhere, which is not a log-probability row at all; or a NaN
  /// anywhere, which has no order against any bound. The module docs'
  /// "Totality" section for why those are refused rather than folded, and why a
  /// row's absolute SCALE is not among the things refused.
  pub(crate) fn push(&mut self, window: &LogProbabilities, weight_samples: usize) -> Result<()> {
    let row = window.as_slice();
    // The fold's precondition, stated once for every pooling rather than in
    // the one that provokes it, and stating exactly what the fold needs: that
    // the row can be made a distribution, which is that its maximum is finite.
    // Not that its maximum is LARGE — the normalization below subtracts the
    // row's own maximum first, so a row written at any finite scale folds to
    // the same thing, and a guard that refused low ones would be the module's
    // own "a row's own scale is not evidence" defect wearing a different hat.
    //
    // What the three refused rows would have done: an all-`-inf` row is not
    // evidence about any language, and each pooling mishandles it differently
    // (the linear pool's terms are all skipped — that is what keeps
    // `(-inf) - (-inf)` unreachable — while its weight still lands in the
    // denominator, so the pool comes out diluted; a vote would be cast for
    // whatever column the ranking tie-break surfaces). A row holding `+inf`
    // is not a log-probability row at all. A row holding a NaN has no order
    // against anything, so each pooling loses it in its own direction: the
    // logarithmic pool spreads it over all 107 columns, `Max` and the linear
    // pool DROP the window (`f64::max` ignores a NaN and the log-sum-exp skips
    // it like a `-inf`), and `Vote` hands the window's whole ballot to the NaN's
    // column, because `total_cmp` ranks a NaN above every real value.
    //
    // The exit cannot be trusted with any of the three, and the reason is the
    // same each time: it reads a MASS, and two of these three leave a mass of
    // exactly 1. `Vote` does so on all three; `Max` and the linear pool do on a
    // NaN beside a real window, having quietly answered from the real one. Only
    // `MeanProbability` on a `+inf` and the logarithmic pool on a NaN leave a
    // mass the postcondition can see. And a LONE bad window takes the identity
    // path straight back to the caller, where the postcondition does not apply
    // at all.
    if !has_a_finite_maximum(row) {
      return Err(Error::UnnormalizableWindow(self.count));
    }
    let weight = weight_samples as f64;
    if self.count == 0 {
      // The RAW row, before the normalization below: `finish` hands it straight
      // back when it turns out to be the only one, and that verbatim return is
      // the whole of the `identify_long` == `identify` promise. It is the one
      // place in this fold where a row's own scale is the right answer, because
      // the answer is the caller's or the model's rather than one this module
      // computed.
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
    // The row is made a distribution HERE, at the door, before any pooling
    // folds it — under the same shift, from the same helper, that normalizes
    // the row the fold produces. Three of the four poolings fold VALUES, so
    // folding a raw row folds its own overall scale along with the ratios that
    // are the evidence, and that scale is fp noise (module docs, "A row's own
    // scale is not evidence").
    //
    // `Vote` is exempt, and not by oversight. Its ballot is an `argmax`, and an
    // argmax is a comparison: subtracting one constant from every column cannot
    // reorder them, so normalizing first is measurably a no-op (0.0 deviation)
    // and can only do harm — two distinct values can round onto one shifted
    // value, creating a tie the tie-break then decides. So `Vote` reads `row`
    // and the other three read `normalized`.
    let normalized = match self.pooling {
      ScorePooling::Vote => Vec::new(),
      _ => as_distribution(row),
    };
    match self.pooling {
      ScorePooling::MeanLogProbability => {
        for (a, &v) in self.acc.iter_mut().zip(&normalized) {
          *a += weight * v;
        }
      }
      // Online weighted log-sum-exp, one running shift per language, so no
      // log-probability is ever exponentiated on its own: `exp` flushes
      // anything below ln(f64::MIN_POSITIVE) ≈ −744.4 to exactly zero, and a
      // literal `Σ w·exp(x)` then re-logs the whole tail as `-∞` — losing an
      // ordering the input row still carried, from finite values throughout.
      ScorePooling::MeanProbability => {
        for ((sum, shift), &value) in self.acc.iter_mut().zip(&mut self.shift).zip(&normalized) {
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
        for (a, &v) in self.acc.iter_mut().zip(&normalized) {
          *a = a.max(v);
        }
      }
      // The RAW row, per the exemption above.
      ScorePooling::Vote => self.acc[argmax(row)] += weight,
    }
    self.weight_sum += weight;
    self.count += 1;
    Ok(())
  }

  /// Finish the fold into one clip-level [`LogProbabilities`].
  ///
  /// A single pushed window is returned verbatim — no divide, no
  /// renormalization, no narrowing — so the long path and the single-shot path
  /// agree bit for bit on a clip that fits one window.
  ///
  /// # Errors
  /// [`Error::EmptyWindows`] if no window was pushed;
  /// [`Error::ZeroMassAggregate`] if the fold left no probability mass at all,
  /// and [`Error::NotADistribution`] if it left anything other than 1 — a NaN
  /// mass among them, which is the shape a fold that is not arithmetic at all
  /// comes back as (module docs, "Totality").
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
    if count == 1 {
      // The identity path. Nothing was folded, so the fold's postcondition has
      // nothing to say here: the row is the caller's, or the model's, and
      // returning it unchanged is precisely the promise that makes
      // `identify_long` a drop-in for `identify`. Holding it to "sums to 1"
      // would break that promise on real audio, not only in principle — a
      // model row's own mass is off by up to 7.7e-3 on `CpuOnly` and 1.5e-4 on
      // the ANE (measured; see the module docs' "Totality" section), so the
      // long path would start refusing clips the short path answers. What the
      // row IS held to is `push`'s precondition, which is the only property a
      // row this code did not compute can be held to: it normalizes, so it
      // ranks. A row written low enough that its f64 mass underflows to zero
      // comes back through here unchanged, and that is the answer `identify`
      // gives for the same row.
      return Ok(LogProbabilities::new(first));
    }
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
      //
      // `s + ln(...)` is the same large-shift-plus-small-constant shape the
      // normalizer below is written to avoid, and here it is harmless — but
      // only for a reason worth writing down rather than re-deriving. `s` is
      // one language's own largest log-probability, not a shift shared with
      // the value it is added to, so there is no rewrite that expresses the
      // sum near zero: the answer genuinely IS of magnitude `s`. What makes
      // it benign is that absorption needs `|s|` above about 1e16, where the
      // constant falls under the f64 ULP — and `s` is a NORMALIZED value, one
      // `push` already took each row's own maximum off, so it is at most 0 and
      // its magnitude measures how far below its row's maximum the column sat,
      // not how far from zero the row was written. A column reaching 1e16 is
      // therefore 1e16 nats below its own row's maximum. Its probability is
      // exactly zero either way, and no f64 could hold the difference the
      // constant would have made.
      //
      // Then renormalize, exactly as `Max` does. A mixture of distributions
      // IS a distribution, and `push` folds nothing else, so the shift is
      // near zero and the row barely moves; what it buys is that the row is a
      // distribution whether or not this pooling's own f64 arithmetic left it
      // one. The model's mass gap — 7.7e-3 on `CpuOnly` — is no longer
      // among the things it has to absorb: it comes off each row at the door,
      // which is also where it stops acting as a per-window weight. A
      // constant shift cannot reorder the row, so no ranking this pooling
      // reported before changes.
      ScorePooling::MeanProbability => {
        for (a, &s) in acc.iter_mut().zip(&shift) {
          *a = s + (*a / weight_sum).ln();
        }
        renormalize(&mut acc);
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
    let values: Vec<f32> = acc.into_iter().map(|v| v as f32).collect();
    // Every arm above leaves values `<= 0`: `renormalize` subtracts a
    // log-sum-exp that is at least the row maximum, and `Vote` divides by a
    // sum that bounds every element. Narrowing to f32 rounds to nearest and so
    // cannot turn a non-positive value positive.
    //
    // What none of that establishes is that the row is a DISTRIBUTION. Stated
    // here, once, as a postcondition on the row the caller receives rather
    // than as a guard inside whichever pooling last got it wrong: a pooling
    // added later inherits it without having to remember to, and so does a
    // future edit to one of these four.
    let mass = probability_mass(&values);
    if mass <= 0.0 {
      // Not an arithmetic slip: a logarithmic pool over windows with disjoint
      // supports honestly leaves nothing. Its own variant, and its own
      // explanation.
      return Err(Error::ZeroMassAggregate(pooling));
    }
    // Written as the predicate that ACCEPTS, and returned from, rather than as
    // a `> MAX_MASS_DEVIATION` that refuses — which is the same test only over
    // an ordered domain. Every ordered comparison against a NaN is false, so a
    // NaN mass reads as "not outside the tolerance" under the refusing form and
    // is handed to the caller as a row of NaN; under this one it is simply not
    // inside the tolerance, and falls into the refusal below. A guard a future
    // pooling can walk straight through is not the "stated once, inherited by
    // whatever is added later" this postcondition is for, so it is one TOTAL
    // predicate rather than a pair of partial ones with an `is_nan` beside them.
    if (mass - 1.0).abs() <= MAX_MASS_DEVIATION {
      return Ok(LogProbabilities::new(values));
    }
    // `NotADistribution` already carries the mass, and a NaN in that payload is
    // the honest report: the fold left something that is not a number, and the
    // variant's own `Display` renders it as `sum to NaN, not 1`. No case here
    // needs a variant of its own.
    Err(NotADistribution::new(pooling, mass).into())
  }
}

/// The one constant that turns a row of natural-log values into a distribution:
/// subtract it from every value and `exp` over the row sums to 1.
///
/// **One definition, called from both ends of the fold.** The row a pooling
/// FOLDS is normalized by [`Accumulator::push`] and the row a pooling PRODUCES
/// is normalized by [`renormalize`], and both take the shift from here rather
/// than spelling it out for themselves. Two ends that each spell it out are two
/// places to get it wrong and two places to fix it, and this module has been
/// both: the exit's arithmetic had to be corrected once for the fusion the next
/// paragraph describes, while the entrance — which had no normalization at all
/// — went on folding raw rows for another two review rounds. Whatever the next
/// change to "the shift that makes a row a distribution" is, there is now one
/// place to make it.
///
/// Held in TWO parts — the row's maximum and the log of the shifted sum — which
/// [`Self::apply`] subtracts one after the other and which nothing may fold
/// into one. Forming `max + log_sum` and subtracting that loses `log_sum`
/// entirely whenever `max` is large enough that the sum's log falls below its
/// ULP: at −5e19 the f64 ULP is 8192, so `max + ln(2)` IS `max`, every leading
/// column comes back as exactly `0.0`, and the row's mass is its number of
/// leading columns instead of 1. Subtracting the maximum first lands each value
/// near zero, where the small constant is representable. The two forms agree
/// everywhere the fused one has not already lost the constant.
#[derive(Debug, Clone, Copy)]
struct DistributionShift {
  max: f64,
  log_sum: f64,
}

impl DistributionShift {
  /// The shift for the row `values` yields.
  ///
  /// Total, and deliberately so: a row whose maximum is not finite has no shift
  /// that makes it a distribution — an all-`-∞` row is the case that reaches
  /// here, and `-∞ − (−∞)` is NaN — so the shift is the IDENTITY and the row
  /// comes back untouched rather than poisoned. Refusing such a row belongs to
  /// [`Accumulator::push`]'s precondition at the entrance and
  /// [`Accumulator::finish`]'s postcondition at the exit, which cover every
  /// pooling between them; doing it here as well would only give the shift a
  /// second place to disagree with them.
  ///
  /// **Which end still reaches this branch, since the precondition became
  /// "the maximum is finite".** [`as_distribution`], the ENTRANCE, no longer
  /// can: `push` has already refused every row it would fire on. [`renormalize`],
  /// the EXIT, still does and is why the branch stays —
  /// [`ScorePooling::MeanLogProbability`] over windows with disjoint supports
  /// folds to `-∞` in every column, and this branch is what makes that a clean
  /// [`Error::ZeroMassAggregate`] instead of a row of NaN
  /// (`a_zero_mass_logarithmic_pool_is_refused_rather_than_returned` walks the
  /// whole path; `renormalize_does_not_turn_an_impossible_row_into_nan` pins
  /// the branch on its own).
  fn of(values: impl Iterator<Item = f64> + Clone) -> Self {
    let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
      return Self {
        max: 0.0,
        log_sum: 0.0,
      };
    }
    let sum: f64 = values.map(|v| (v - max).exp()).sum();
    Self {
      max,
      log_sum: sum.ln(),
    }
  }

  /// `value`, shifted — the two subtractions, in the order that keeps the
  /// second one representable.
  fn apply(self, value: f64) -> f64 {
    (value - self.max) - self.log_sum
  }
}

/// Shift `values` down by their log-sum-exp so `exp` over the row sums to 1 —
/// the fold's EXIT normalizer, closing [`ScorePooling::MeanLogProbability`],
/// [`ScorePooling::MeanProbability`] and [`ScorePooling::Max`].
///
/// The max-subtraction form is what keeps [`ScorePooling::Max`] finite when the
/// row's maximum is far from zero; [`DistributionShift`] carries the arithmetic
/// and the reason its two subtractions must stay apart. The row maximum is
/// never `-∞` in practice (a model row is a log-softmax and its argmax is
/// finite), and an all-`-∞` row is left alone rather than turned into NaN. It
/// stays infallible and total for that reason: the row it declines to touch is
/// not a distribution, and refusing it is [`Accumulator::finish`]'s
/// postcondition, which catches it for every pooling rather than only for the
/// ones that renormalize.
fn renormalize(values: &mut [f64]) {
  let shift = DistributionShift::of(values.iter().copied());
  for v in values {
    *v = shift.apply(*v);
  }
}

/// `row` as a distribution, in f64 — the fold's ENTRANCE normalizer, and the
/// values [`Accumulator::push`] actually folds.
///
/// A row's own total mass is fp noise: a model row comes back as much as 7.7e-3
/// away from 1 on [`ComputeUnits::CpuOnly`] and 1.5e-4 with the ANE, on either
/// side of it, and that gap says nothing about any language, only about the
/// graph's fp16 arithmetic. Folded raw it acts as a per-window WEIGHT — see the module docs'
/// "A row's own scale is not evidence" section for the two windows it flipped.
///
/// [`ComputeUnits::CpuOnly`]: crate::ComputeUnits::CpuOnly
fn as_distribution(row: &[f32]) -> Vec<f64> {
  let shift = DistributionShift::of(row.iter().copied().map(f64::from));
  row.iter().map(|&v| shift.apply(f64::from(v))).collect()
}

/// Whether `row` has a finite maximum — the fold's precondition, and the exact
/// condition under which [`DistributionShift`] can make the row a distribution.
///
/// The two rows it refuses are the two [`DistributionShift::of`] has no shift
/// for: `-∞` in every column, and `+∞` anywhere. `aggregate`'s module docs
/// ("Totality") carry what each of them would have done to each pooling.
///
/// **It deliberately says nothing about a row's absolute SCALE**, and an
/// earlier form of it did. The guard used to be `exp(max) > 0.0`, which is
/// "the row's total is positive" — and `exp` underflows f64 to exactly zero
/// below `ln(f64::MIN_POSITIVE)` ≈ −744.44, so `[-800, -801, …, -906]` was
/// refused while `[0, -1, …, -106]` was folded. Those two rows carry the
/// identical evidence: every column sits the same distance below its own row's
/// maximum, and both normalize to the identical distribution. Refusing one of
/// them was the module's "a row's own scale is not evidence" leak still
/// standing in the door after it had been taken out of the fold.
///
/// The old test was defensible while nothing could normalize a row of enormous
/// negatives, and that stopped being true when [`DistributionShift`] gained one
/// anchored definition: it forms `(v − max)` before anything else, which is
/// well-conditioned for ANY finite maximum however far from zero it sits.
///
/// **Why the maximum is checked to be an upper bound as well as finite.**
/// [`f32::max`] IGNORES a NaN operand — that is its documented `maxNum`
/// semantics — so folding a row with it returns the maximum of the row's
/// non-NaN values, which for `[-1, NaN, -1, …]` is a perfectly finite `-1`.
/// The row is not normalizable: `DistributionShift::of` sums `exp(v − max)`
/// over EVERY value, so the NaN poisons the sum, its log, and every column the
/// shift is then applied to. Naming the fold's result `max` and then asking
/// whether the row actually sits `<=` it is what makes the predicate total: a
/// NaN compares false against every bound, which is the same reason the row has
/// no maximum. The alternative spelling — a separate `is_nan` scan beside the
/// finiteness test — is the pair of partial predicates this door already had
/// one of.
fn has_a_finite_maximum(row: &[f32]) -> bool {
  let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  max.is_finite() && row.iter().all(|&value| value <= max)
}

/// How far a FOLDED row's probability mass may sit from 1 before
/// [`Accumulator::finish`] refuses it.
///
/// Measured, not chosen. Every row that ENTERS the fold is normalized at the
/// door and every row that LEAVES it is a distribution by construction
/// ([`ScorePooling::Vote`], from the shares it divides) or by a closing
/// renormalization (the other three), so the only deviation that should survive
/// is the narrowing to f32 at the exit. Two numbers bound that:
///
/// - **Derived.** A row that sums to 1 in f64 has mass error at most
///   `Σ p·2⁻²⁴·|ln p|`, maximized by the uniform row at
///   `2⁻²⁴·ln(107) = 2.8e-7`.
/// - **Observed.** Over the committed Thai clip repeated to 39 s and 52 s, that
///   clip spliced with English, and English alone, at three geometries
///   (10 s/10 s, 5 s/2.5 s, 3 s/3 s), on all four compute units and all four
///   poolings — 192 folds over 2 to 20 windows each — the largest deviation
///   any fold produced was **3.9e-8** (`Max`, on the ANE). It was 5.7e-8 while
///   the fold still took raw rows; normalizing each row before folding it
///   removed the input deficit the exit shift had been absorbing.
///
/// That sweep is a committed model gate rather than a remembered run:
/// `every_fold_in_the_published_sweep_lands_far_inside_the_mass_tolerance`, in
/// `tests/lid/long_clip.rs`, runs all 192 folds and prints the per-compute-unit
/// table, so both numbers above are re-derivable from this tree.
///
/// `1e-5` is 36× the derived bound and 258× the largest observed, and still
/// four orders of magnitude below either defect this postcondition was written
/// for (mass 2 and mass 0.5). What it is NOT derived from is the model: the
/// same sweep's raw per-window rows are off by up to **7.7e-3**, five orders
/// looser, because a fold's mass is something the poolings establish rather
/// than inherit. That is also why it is applied ONLY to a folded row — the
/// single-window identity path returns a row this module did not compute, and
/// refusing that would break the `identify_long` == `identify` promise on real
/// audio.
const MAX_MASS_DEVIATION: f64 = 1e-5;

/// The row's total probability mass: `exp` summed over it, in f64.
///
/// Wider than the row it reads, on purpose — the narrowing to f32 is the thing
/// being measured, so the measurement must not be narrowed too.
///
/// It is `NUM_LANGUAGES`-bounded and NaN-free for every row the four poolings
/// currently produce, because each of them emits values that are `<= 0` and
/// non-NaN, so every term lands in `(0, 1]`. That is a fact about those four
/// and NOT a guarantee this function makes: `exp` of a NaN is a NaN and the sum
/// carries it, so a fifth pooling that left one would be reported here as a NaN
/// mass. [`Accumulator::finish`] is written to refuse that rather than to
/// assume it cannot happen.
///
/// [`NUM_LANGUAGES`]: crate::audio::lid::NUM_LANGUAGES
fn probability_mass(row: &[f32]) -> f64 {
  row.iter().map(|v| f64::from(*v).exp()).sum()
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
/// [`Error::UnnormalizableWindow`], naming the window, if one of them has no
/// finite maximum — `-∞` throughout (which is not evidence about any language),
/// `+∞` anywhere (which is not a log-probability row), or a NaN anywhere (which
/// sits under no bound at all). Every pooling refuses those alike. A row's
/// absolute SCALE is not among the things refused: a row whose largest value is
/// `-800` folds exactly as one shifted up to `0` does.
///
/// [`Error::ZeroMassAggregate`] if the pooled row assigns probability zero to
/// every language, which is not a distribution and cannot be ranked — see the
/// module docs' "Totality" section for when a pooling honestly answers that.
///
/// [`Error::NotADistribution`] if the pooled row's mass is neither zero nor 1,
/// which is a defect in this crate rather than a property of `windows`.
///
/// All three are unreachable through `identify_long`: a model row is a
/// log-softmax, so it is all-finite and its mass is positive.
///
/// # Examples
/// ```
/// use coremlit::audio::lid::{
///   LogProbabilities, NUM_LANGUAGES, ScorePooling, Span, WindowLogProbabilities,
///   aggregate_windows,
/// };
///
/// // Two equal-length windows: one sure of column 94, one sure of column 0.
/// // Each row is a real distribution: 99.9 % on one language, the remaining
/// // 0.1 % spread over the other 106.
/// let window = 160_000;
/// let row = |hot: usize| {
///   let rest = (0.001f64 / (NUM_LANGUAGES - 1) as f64).ln() as f32;
///   let mut values = vec![rest; NUM_LANGUAGES];
///   values[hot] = 0.999f64.ln() as f32;
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
    acc.push(window.value(), window.span().len())?;
  }
  acc.finish()
}
