//! Long-clip windowing and pooling, against the real graph.
//!
//! # The two oracles, reproduced
//!
//! The policy `audio::lid` ships was chosen on two oracles built from the model
//! itself, because there is no labelled long-form corpus for it (the module
//! docs carry the full tables and the honest list of what that leaves
//! unverified). Both are re-run here, narrowed to the committed Thai fixture so
//! they are reproducible from this repository alone:
//!
//! 1. **Self-consistency** — on a clip that fits ONE prediction, the model's
//!    single-shot answer is ground truth by definition. Window it, aggregate,
//!    and compare. The whole-clip identity (`identify_long` == `identify` on a
//!    clip that fits one window) is the degenerate case of the same oracle, and
//!    it is asserted bit for bit.
//! 2. **Concatenation** — repeating `udhr_th_16k.wav` gives a clip whose
//!    language is known by construction and whose length is past the graph's
//!    30 s ceiling. Splicing another language in gives the case where the
//!    poolings genuinely disagree, which is asserted rather than described.
//!
//! The gate that justifies the tail policy is here too: the same audio scored
//! honestly and zero-padded up to a full window does not agree, and for a short
//! enough tail it does not even agree on the LANGUAGE. That is why
//! [`TailPolicy`] has no `Pad` variant.
//!
//! The gates are `#[ignore]`d and need the artifact staged (`LID_TEST_MODELS`,
//! default `Models/lid`).

mod common;

use coremlit::audio::lid::{
  DEFAULT_WINDOW_SAMPLES, Error, Identifier, LogProbabilities, NUM_LANGUAGES, ScorePooling, Span,
  TailPolicy, WindowPlan, aggregate_windows, languages,
};

/// Model column of Thai, the committed clip's language (pinned independently by
/// `e2e.rs`).
const THAI_INDEX: usize = 94;

/// The single-shot reference for the whole 13 s clip, from `e2e.rs`.
const THAI_LOG_PROBABILITY: f32 = -0.010_064;

const POOLINGS: [(&str, ScorePooling); 4] = [
  ("mean-log", ScorePooling::MeanLogProbability),
  ("mean-prob", ScorePooling::MeanProbability),
  ("max", ScorePooling::Max),
  ("vote", ScorePooling::Vote),
];

fn clip() -> Vec<f32> {
  common::read_wav_16k_mono(&common::fixture_path("audio/udhr_th_16k.wav"))
}

fn identifier() -> Identifier {
  Identifier::from_file(common::model_path()).expect("load identifier")
}

fn repeated(times: usize) -> Vec<f32> {
  let one = clip();
  std::iter::repeat_n(one, times).flatten().collect()
}

fn argmax(row: &[f32]) -> usize {
  let mut best = 0;
  for (index, value) in row.iter().enumerate().skip(1) {
    if value.total_cmp(&row[best]) == core::cmp::Ordering::Greater {
      best = index;
    }
  }
  best
}

fn top_three(row: &[f32]) -> [usize; 3] {
  let mut order: Vec<usize> = (0..row.len()).collect();
  order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
  [order[0], order[1], order[2]]
}

fn code(index: usize) -> &'static str {
  languages()[index].code()
}

// ── Hermetic ────────────────────────────────────────────────────────────────

/// The committed clip is longer than one default window and shorter than three,
/// so the concatenations below really do exercise the multi-window path and the
/// slid-back tail. Stated here so a re-cut fixture reds on the ASSUMPTION
/// rather than silently turning the model gates into single-window runs.
#[test]
fn the_fixture_has_the_geometry_these_gates_assume() {
  let samples = clip();
  assert_eq!(samples.len(), 207_952);
  assert!(samples.len() > DEFAULT_WINDOW_SAMPLES as usize);
  assert!(samples.len() < 3 * DEFAULT_WINDOW_SAMPLES as usize);

  // Three copies is past the graph's own ceiling, which is the point.
  let three = 3 * samples.len();
  assert!(three > coremlit::audio::lid::MAX_SAMPLES);
  assert!((three as f64 / 16_000.0 - 38.99).abs() < 0.01);
}

/// A 39 s clip plans four full-length windows under the default policy, the
/// last slid back to end flush with the clip — no padding, no ragged span, one
/// graph shape. Pure geometry: no model.
#[test]
fn the_default_plan_slides_the_tail_back_to_full_length() {
  let total = 3 * clip().len();
  let window = DEFAULT_WINDOW_SAMPLES as usize;
  let spans = WindowPlan::new().spans(total).expect("plan");

  assert_eq!(spans.len(), 4);
  assert!(
    spans.iter().all(|s| s.len() == window),
    "every span must be a full window: {spans:?}"
  );
  assert_eq!(spans[3], Span::new(total - window, window, window));
  assert_eq!(spans[3].end(), total, "the plan ends flush with the clip");
  assert!(
    spans[3].start() < spans[2].end(),
    "the slid-back window overlaps its predecessor rather than running short"
  );
}

// ── Oracle 1: self-consistency ──────────────────────────────────────────────

/// The degenerate case of the self-consistency oracle, and the contract that
/// makes `identify_long` a drop-in: on a clip that FITS one window it returns
/// bit-for-bit what `identify` returns, under every pooling. Not "close" —
/// equal, because a one-window fold is the identity.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn the_long_path_is_bit_identical_to_identify_on_a_clip_that_fits() {
  let identifier = identifier();
  let fits = &clip()[..DEFAULT_WINDOW_SAMPLES as usize];
  let single_shot = identifier.identify(fits, 5).expect("identify");
  assert_eq!(single_shot[0].index(), THAI_INDEX);

  for (label, pooling) in POOLINGS {
    let long = identifier
      .identify_long(fits, 5, &WindowPlan::new(), pooling)
      .expect("identify_long");
    assert_eq!(long.len(), single_shot.len(), "{label}");
    for (got, want) in long.iter().zip(&single_shot) {
      assert_eq!(got.index(), want.index(), "{label}");
      assert_eq!(
        got.log_probability(),
        want.log_probability(),
        "{label}: the one-window fold must be the bit-exact identity"
      );
    }
  }

  // And the per-window seam agrees: one span, the whole clip, full coverage.
  let windows = identifier
    .log_probabilities_windows(fits, &WindowPlan::new())
    .expect("windows");
  assert_eq!(windows.len(), 1);
  assert_eq!(windows[0].span().len(), fits.len());
  assert_eq!(
    windows[0].value().as_slice(),
    identifier.log_probabilities(fits).expect("row").as_slice()
  );
}

/// Self-consistency proper: window the 13 s clip at 3 s — five windows, each
/// with far less context than the whole — and check every pooling still
/// reproduces the single-shot ranking, with the default reproducing the
/// single-shot VALUE most closely.
///
/// The ordering assertion is the measurement the default rests on, narrowed to
/// what one clip can support: the logarithmic pool must not be beaten at the
/// top-1 log-probability by the two candidates the module docs reject on that
/// metric.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn windowed_scores_reproduce_the_single_shot_ranking() {
  let identifier = identifier();
  let samples = clip();
  let truth = identifier.log_probabilities(&samples).expect("single shot");
  assert_eq!(argmax(&truth), THAI_INDEX);

  let plan = WindowPlan::new().with_geometry(48_000, 48_000);
  let windows = identifier
    .log_probabilities_windows(&samples, &plan)
    .expect("windows");
  assert!(windows.len() >= 4, "got {} windows", windows.len());

  let mut deltas = Vec::new();
  for (label, pooling) in POOLINGS {
    let got = aggregate_windows(pooling, &windows).expect("aggregate");
    let row = got.as_slice();
    assert_eq!(
      argmax(row),
      THAI_INDEX,
      "{label} must reproduce the single-shot top-1"
    );
    let delta = f64::from(row[THAI_INDEX] - truth[THAI_INDEX]).abs();
    println!(
      "  {label:>10}  top-1 {} {:>9.5}  |Δ vs single-shot| {:.5}  top-3 {:?}",
      code(argmax(row)),
      row[argmax(row)],
      delta,
      top_three(row).map(code)
    );
    deltas.push((label, delta));
  }

  let mean_log = deltas[0].1;
  for (label, delta) in &deltas[1..] {
    if *label == "vote" {
      // A vote reports a window SHARE, not a probability; on a clip every
      // window agrees about, that share is exactly 1 and the comparison is
      // meaningless. Its rejection is a top-3 argument, made below.
      continue;
    }
    assert!(
      mean_log <= *delta + 1e-6,
      "the default ({mean_log:.5}) must not be beaten by {label} ({delta:.5})"
    );
  }

  // The vote's actual defect: everything below the winners is a zero share, so
  // its ranking below the top is not a ranking at all.
  let voted = aggregate_windows(ScorePooling::Vote, &windows).expect("aggregate");
  let below_top = top_three(voted.as_slice())[1];
  assert_eq!(
    voted.as_slice()[below_top],
    f32::NEG_INFINITY,
    "a vote leaves every unchosen language at exactly zero probability"
  );
}

// ── Oracle 2: concatenation ─────────────────────────────────────────────────

/// A clip whose language is known by construction, past the graph's ceiling:
/// three copies of the Thai reference, 39 s. Every pooling must still say Thai,
/// and the default must still say it about as loudly as the single shot did.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn a_concatenated_clip_past_the_ceiling_keeps_its_language() {
  let identifier = identifier();
  let long = repeated(3);

  // `identify`'s own contract is unchanged: this clip is still a typed refusal
  // there. The long path is additive, not a relaxation.
  let refused = identifier
    .identify(&long, 1)
    .expect_err("must still refuse");
  assert!(
    matches!(&refused, Error::FrameCountOutOfRange(d) if !d.is_too_short()),
    "got {refused:?}"
  );

  for (label, pooling) in POOLINGS {
    let ranked = identifier
      .identify_long(&long, 3, &WindowPlan::new(), pooling)
      .expect("identify_long");
    println!(
      "  {label:>10}  {} {:>9.5} | {} {:>9.5} | {} {:>9.5}",
      ranked[0].code(),
      ranked[0].log_probability(),
      ranked[1].code(),
      ranked[1].log_probability(),
      ranked[2].code(),
      ranked[2].log_probability(),
    );
    assert_eq!(ranked[0].index(), THAI_INDEX, "{label}");
    assert!(
      ranked[0].probability() > 0.95,
      "{label}: a clip that is Thai three times over is a confident call, got {}",
      ranked[0].probability()
    );
  }

  // The default lands within 0.01 nats of the whole-clip single-shot reference
  // — the aggregate is usable wherever a single-window score is.
  let default = identifier
    .identify_long(&long, 1, &WindowPlan::new(), ScorePooling::default())
    .expect("identify_long");
  assert!(
    (default[0].log_probability() - THAI_LOG_PROBABILITY).abs() < 0.01,
    "default pooling gave {} against the single-shot reference {THAI_LOG_PROBABILITY}",
    default[0].log_probability()
  );
}

/// Where the poolings genuinely diverge, asserted rather than described: a 37 s
/// clip that is 70 % Thai and 30 % English.
///
/// The default (logarithmic pool) sharpens to Thai and drops English out of the
/// top three entirely; the linear pool reports the MIXTURE, with English second
/// at roughly its share of the clip. Both are correct answers to different
/// questions, and this is the gate that keeps the docs' claim honest.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn a_spliced_minority_language_is_kept_by_one_pooling_and_erased_by_another() {
  let identifier = identifier();
  let english = common::read_wav_16k_mono(
    &std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("tests/whisper/fixtures/audio/jfk.wav"),
  );
  let mixed: Vec<f32> = repeated(2).into_iter().chain(english).collect();
  assert!(mixed.len() > coremlit::audio::lid::MAX_SAMPLES);

  let english_index = languages()
    .iter()
    .position(|l| l.code() == "en")
    .expect("English is in the roster");

  let windows = identifier
    .log_probabilities_windows(&mixed, &WindowPlan::new())
    .expect("windows");
  for w in &windows {
    let row = w.value().as_slice();
    println!(
      "  window {:>7}..{:<7}  {} {:.4}",
      w.span().start(),
      w.span().end(),
      code(argmax(row)),
      row[argmax(row)]
    );
  }

  let logarithmic =
    aggregate_windows(ScorePooling::MeanLogProbability, &windows).expect("aggregate");
  let linear = aggregate_windows(ScorePooling::MeanProbability, &windows).expect("aggregate");
  println!(
    "  mean-log  top3 {:?}\n  mean-prob top3 {:?}",
    top_three(logarithmic.as_slice()).map(code),
    top_three(linear.as_slice()).map(code)
  );

  assert_eq!(argmax(logarithmic.as_slice()), THAI_INDEX);
  assert_eq!(argmax(linear.as_slice()), THAI_INDEX);
  assert!(
    !top_three(logarithmic.as_slice()).contains(&english_index),
    "the logarithmic pool sharpens to one language and drops the minority"
  );
  assert_eq!(
    top_three(linear.as_slice())[1],
    english_index,
    "the linear pool reports the mixture, English second"
  );
  let english_share = f64::from(linear.as_slice()[english_index]).exp();
  assert!(
    (0.05..0.45).contains(&english_share),
    "English should read as roughly its share of the clip, got {english_share:.3}"
  );
}

// ── The tail: why there is no `Pad` ─────────────────────────────────────────

/// The measurement [`TailPolicy`]'s shape rests on: the same audio scored
/// honestly and zero-padded up to a full window disagrees by many nats, and for
/// a short tail it disagrees about the LANGUAGE. Sliding a full-length window
/// back over the clip end instead pays none of it.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn padding_a_short_tail_moves_the_answer_and_sliding_back_does_not() {
  let identifier = identifier();
  let samples = clip();
  let window = DEFAULT_WINDOW_SAMPLES as usize;

  let slid_back = identifier
    .log_probabilities(&samples[..window])
    .expect("slid-back window");
  assert_eq!(argmax(&slid_back), THAI_INDEX);

  let mut moved_the_language = 0;
  for tail in [16_000usize, 48_000, 96_000] {
    let honest = identifier
      .log_probabilities(&samples[..tail])
      .expect("honest tail");
    let mut padded_input = samples[..tail].to_vec();
    padded_input.resize(window, 0.0);
    let padded = identifier
      .log_probabilities(&padded_input)
      .expect("padded tail");

    let worst = honest
      .iter()
      .zip(&padded)
      .map(|(h, p)| f64::from(*p - *h).abs())
      .fold(0.0f64, f64::max);
    println!(
      "  tail {:>6} ({:>4.1} s)  honest {} {:>8.4}  padded {} {:>8.4}  worst shift {:>6.3} nats",
      tail,
      tail as f64 / 16_000.0,
      code(argmax(&honest)),
      honest[argmax(&honest)],
      code(argmax(&padded)),
      padded[argmax(&padded)],
      worst
    );
    assert!(
      worst > 2.0,
      "padding a {tail}-sample tail shifted the row by only {worst:.3} nats — \
       if this ever becomes small, the `Pad`-is-excluded argument needs redoing"
    );
    if argmax(&padded) != argmax(&honest) {
      moved_the_language += 1;
    }
  }
  assert!(
    moved_the_language > 0,
    "padding must be shown to change the reported language on at least one tail"
  );
}

// ── No ceiling ──────────────────────────────────────────────────────────────

/// The point of the whole change: a clip far past 30 s is answered rather than
/// refused, the per-window seam is available alongside it, and the aggregate of
/// the per-window rows is exactly what the streaming path returned.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn a_two_minute_clip_is_answered_not_refused() {
  let identifier = identifier();
  let long = repeated(10); // ~130 s
  assert!(long.len() > 4 * coremlit::audio::lid::MAX_SAMPLES);

  let plan = WindowPlan::new();
  let windows = identifier
    .log_probabilities_windows(&long, &plan)
    .expect("windows");
  assert_eq!(windows.len(), plan.spans(long.len()).expect("plan").len());
  assert!(
    windows
      .iter()
      .all(|w| argmax(w.value().as_slice()) == THAI_INDEX),
    "every window of a Thai clip should be Thai"
  );

  let streamed = identifier
    .identify_long(&long, 3, &plan, ScorePooling::default())
    .expect("identify_long");
  let batched = aggregate_windows(ScorePooling::default(), &windows)
    .expect("aggregate")
    .top_k(3)
    .expect("rank");
  assert_eq!(streamed.len(), 3);
  assert_eq!(streamed[0].index(), THAI_INDEX);
  for (a, b) in streamed.iter().zip(&batched) {
    assert_eq!(a.index(), b.index());
    assert_eq!(
      a.log_probability(),
      b.log_probability(),
      "the streaming and batch folds must agree bit for bit"
    );
  }

  // A hand-built row round-trips through the public seam, so a consumer can
  // unit-test its own aggregation without a model.
  let rebuilt = LogProbabilities::try_from_slice(windows[0].value().as_slice()).expect("rebuild");
  assert_eq!(rebuilt.as_slice(), windows[0].value().as_slice());
  assert_eq!(rebuilt.as_slice().len(), NUM_LANGUAGES);
}

/// The three tail policies on one real long clip: all of them answer Thai, and
/// they differ exactly where the geometry says they should — `SlideBack` and
/// `Drop` produce only full windows, `Partial` produces one short one.
#[test]
#[ignore = "requires the staged LID model (LID_TEST_MODELS)"]
fn every_tail_policy_answers_the_concatenated_clip() {
  let identifier = identifier();
  let long = repeated(3);
  let window = DEFAULT_WINDOW_SAMPLES as usize;

  for tail in [TailPolicy::SlideBack, TailPolicy::Partial, TailPolicy::Drop] {
    let plan = WindowPlan::new().with_tail_policy(tail);
    let windows = identifier
      .log_probabilities_windows(&long, &plan)
      .expect("windows");
    let short = windows.iter().filter(|w| w.span().len() < window).count();
    assert_eq!(
      short,
      usize::from(tail == TailPolicy::Partial),
      "{tail:?} produced {short} sub-window spans"
    );

    let ranked = identifier
      .identify_long(&long, 1, &plan, ScorePooling::default())
      .expect("identify_long");
    println!(
      "  {tail:?}: {} windows, {} {:.5}",
      windows.len(),
      ranked[0].code(),
      ranked[0].log_probability()
    );
    assert_eq!(ranked[0].index(), THAI_INDEX, "{tail:?}");
  }
}
