//! Native CoreML **spoken-language identification** — 16 kHz mono waveform in,
//! ranked languages out ([`NUM_LANGUAGES`] of them: code + English name +
//! model column + natural-log probability), with clips past the graph's 30 s
//! ceiling handled by a measured windowing + pooling policy.
//!
//! The mel front end runs in Rust (the private `mel` submodule) and the
//! mel→log-probabilities network runs natively on Apple silicon as one
//! `.mlmodelc`. NO `ort` anywhere.
//!
//! # A backend-neutral door
//!
//! Nothing in this module's public surface names the model behind it. The
//! types ([`Identifier`], [`LanguageScore`], [`Language`]), the geometry
//! constants, the feature name, the env var and the models directory are all
//! spelled for the *task*, not the network — so a second or third LID backend
//! can land behind the same seam without renaming a caller's code. The
//! provenance that IS backend-specific — the artifact, its revision, its label
//! roster's origin — lives in the `labels` submodule's docs and in
//! `tests/lid/`, where it belongs.
//!
//! Today that backend is `aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML`
//! @ `2aa4d715a79e410d5f9aa32bd7a4fc9225bf9eb0` (Apache-2.0), an export of
//! `speechbrain/lang-id-voxlingua107-ecapa`.
//!
//! # Model artifacts
//!
//! No model is bundled (a `.mlmodelc` is a directory artifact). It is staged as
//! a gitignored dev-time download under `Models/lid/`, overridable with the
//! `LID_TEST_MODELS` environment variable; its per-file SHA-256 and I/O
//! contract are pinned by `tests/lid/model_io.rs`.
//!
//! # The contract
//!
//! ```text
//! input   mel_features       f32  [1, frames, 60]   frames in 10..=3001, TIME-major
//! output  log_probabilities  f32  [1, 107]          natural log, already softmaxed
//! ```
//!
//! `frames = 1 + n_samples / 160` (integer division), so the accepted envelope
//! is [`MIN_SAMPLES`]..=[`MAX_SAMPLES`] — **0.09 s to 30.01 s** at 16 kHz. The
//! runtime rejects anything outside that; this module rejects it FIRST, as
//! [`Error::FrameCountOutOfRange`], so a caller never has to string-match
//! CoreML's own axis-indexed complaint. That envelope bounds ONE prediction;
//! [`Identifier::identify_long`] windows a clip of any length over it (see
//! "Clips longer than 30 s").
//!
//! Both tensors are fp32 at the boundary, but the graph casts to **fp16**
//! immediately and computes in it throughout. That is why the placements do not
//! all agree to the last digit — the reference clip's top score reads -0.010064
//! on the GPU arm and -0.015625 on `CpuOnly` — and why a parity gate against
//! this door wants a tolerance rather than an equality.
//!
//! ## Clips longer than 30 s
//!
//! [`Identifier::identify_long`] windows them, under a [`WindowPlan`] and a
//! [`ScorePooling`]. [`Identifier::identify`] is untouched: same ceiling, same
//! contract, same numbers. The long path is additive, and on a clip that fits
//! one window it returns **bit-identically** what `identify` returns, so there
//! is no boundary to straddle.
//!
//! There is still no upstream-authored windowing policy for this model. The two
//! things that had to be invented — the geometry, and how per-window
//! log-probability vectors combine — were therefore MEASURED rather than
//! assumed. What follows is what was measured, on what, and what it leaves
//! unverified.
//!
//! ### Two oracles, because there is no labelled long-form corpus
//!
//! 1. **Self-consistency.** On a clip that FITS one prediction, the model's own
//!    single-shot answer is ground truth by definition. Window that same clip,
//!    aggregate, and compare. The policy that best reproduces the single-shot
//!    ranking is the defensible default. Run over sixteen clips — the committed
//!    Thai reference, English, Spanish, Japanese and Chinese speech, two 30 s
//!    TED segments, noise, a tone, and reversed / attenuated / noise-mixed
//!    variants of the Thai clip, which is where the model is uncertain and the
//!    policies actually diverge.
//! 2. **Concatenation.** Repeat the committed 13 s Thai clip to 39 s and 52 s:
//!    the answer is `th` by construction. Then splice English, Spanish or noise
//!    into it and watch each policy degrade.
//!
//! ### Aggregation: all four candidates, including the rejected ones
//!
//! Oracle 1 at the default geometry (10 s window, 10 s hop,
//! [`TailPolicy::SlideBack`]; 11 clips long enough to window, 26 windows).
//! "top-3 set" is the overlap between the aggregate's top three and the
//! single-shot top three; the last two columns are mean absolute error against
//! the single-shot row, in nats. Read [`Vote`]'s row-error column with care —
//! it is averaged over only the languages that received a vote, because the
//! rest are exactly `-∞`, so it is not comparable with the other three:
//!
//! | pooling                                  | top-1 | top-3 set | Δ at top-1 | MAE, whole row |
//! |------------------------------------------|-------|-----------|------------|----------------|
//! | [`MeanLogProbability`] (**the default**) | 10/11 | **78.8 %**| **0.138**  | **0.743**      |
//! | [`MeanProbability`]                      | 10/11 | 78.8 %    | 0.194      | 1.382          |
//! | [`Max`]                                  | 10/11 | 78.8 %    | 0.270      | 1.753          |
//! | [`Vote`]                                 | 10/11 | 36.4 %    | ∞          | 0.501          |
//!
//! At a finer geometry (5 s window, 2.5 s hop; 15 clips, 87 windows) they
//! separate further — this is the row that decides it:
//!
//! | pooling                | top-1 | top-3 set | Δ at top-1 | MAE, whole row |
//! |------------------------|-------|-----------|------------|----------------|
//! | [`MeanLogProbability`] | 14/15 | **84.4 %**| **0.117**  | **1.285**      |
//! | [`MeanProbability`]    | 14/15 | 75.6 %    | 0.259      | 3.045          |
//! | [`Max`]                | 12/15 | 73.3 %    | 0.499      | 3.757          |
//! | [`Vote`]               | 14/15 | 40.0 %    | ∞          | 0.979          |
//!
//! Reading it:
//!
//! - **Mean in log space wins both ranking metrics at every geometry tried**
//!   (3 s, 5 s and 10 s windows, overlapped and not) and the row error among
//!   the three policies whose rows are comparable. Its clip-level number is
//!   also close enough to the single-shot one to be used interchangeably — on
//!   the Thai reference it reads −0.010 against a single-shot −0.0101, on a
//!   30 s TED segment −0.001 against −0.0004.
//! - **Mean in probability space** costs roughly double the row error and, at
//!   the finer geometry, nine points of top-3 agreement. It is kept because it
//!   answers a different and sometimes better question — see the mixed-clip
//!   table below.
//! - **Per-class max** is the only candidate that loses top-1 agreement outright
//!   (12/15). One over-confident window sets a language's clip-level score, and
//!   nothing damps it.
//! - **The vote is rejected for the default on its own numbers**, not on taste.
//!   Its top-1 agreement is competitive — but its top-3 agreement is 36–40 %,
//!   because everything below the winners is `-∞` and the ranking below the top
//!   is arbitrary. The ∞ in the Δ column is literal: on at least one clip the
//!   single-shot top-1 language won ZERO windows, so its aggregate probability
//!   is exactly zero. A caller who wants a majority-of-windows answer can still
//!   ask for it.
//!
//! The single top-1 miss shared by all four at the default geometry is a
//! non-speech AudioSet clip whose single-shot "truth" is itself only
//! −1.75 nats (17 %) — the oracle has nothing to say there.
//!
//! ### The same four on a MIXED clip, which is where they really diverge
//!
//! Oracle 2, `th + th + English` (37.0 s, 70 % Thai), per-window argmaxes
//! `[th th th en]`:
//!
//! | pooling                | 1st          | 2nd            | 3rd        |
//! |------------------------|--------------|----------------|------------|
//! | [`MeanLogProbability`] | `th` −0.0099 | `lo` −4.63     | `la` −11.0 |
//! | [`MeanProbability`]    | `th` −0.3032 | **`en` −1.54** | `lo` −4.46 |
//! | [`Max`]                | `th` −0.7134 | `en` −0.8693   | `lo` −3.95 |
//! | [`Vote`]               | `th` −0.2877 | `en` −1.3863   | (−∞)       |
//!
//! The default **erases the minority language**: English is not in its top
//! three at all. That is correct behaviour for the question it answers — "what
//! language is this span" — and wrong for "what languages are in this clip".
//! `MeanProbability` reports 73.8 % / 21.4 %, tracking the actual 70/30 split;
//! `Max` reads it as almost a coin flip. **For a genuinely multilingual clip,
//! read [`Identifier::log_probabilities_windows`] per window rather than any
//! aggregate** — and note that a stretch shorter than one window may never win
//! a window at all (spliced English between two Thai halves loses every 10 s
//! window, and loses every 30 s window even when it is a third of the clip).
//!
//! ### Duration weighting
//!
//! Every window contributes in proportion to the audio it actually saw, always.
//! Under [`TailPolicy::SlideBack`] and [`TailPolicy::Drop`] every span is one
//! full window, so this is exactly the equal-weight mean — measured identical,
//! bit for bit. It only bites under [`TailPolicy::Partial`], and there it is
//! measurably better: 78.8 % vs 72.7 % top-3 agreement, 0.128 vs 0.216 at
//! top-1, 1.08 vs 1.97 row MAE. There is no equal-weight knob because there is
//! no case where equal weights were better.
//!
//! ### The tail: four treatments, one clip set
//!
//! Eight clips that all leave a real tail at the default geometry, mean-in-log
//! throughout. "shapes" is the number of DISTINCT mel frame counts the graph is
//! asked to specialize:
//!
//! | tail treatment              | top-1 | top-3 set | Δ at top-1 | MAE, row  | shapes |
//! |-----------------------------|-------|-----------|------------|-----------|--------|
//! | [`TailPolicy::SlideBack`]   | 7/8   | **79.2 %**| 0.187      | **0.568** | **1**  |
//! | [`TailPolicy::Partial`]     | 7/8   | 79.2 %    | **0.173**  | 1.034     | 6      |
//! | [`TailPolicy::Drop`]        | 7/8   | 75.0 %    | 0.226      | 0.792     | 1      |
//! | zero-pad the tail (**not shipped**) | 7/8 | 75.0 % | 0.218 | 1.854     | 1      |
//!
//! **Padding is not among the shipped policies, and this is why.** Scoring the
//! first `n` seconds of the reference clip honestly, then again zero-padded up
//! to the 10 s window:
//!
//! | real audio | honest      | zero-padded to 10 s | worst shift | slid back to 10 s |
//! |------------|-------------|---------------------|-------------|-------------------|
//! | 0.5 s      | `tl` −0.372 | `sq` −2.140         | 9.1 nats    | `th` −0.0054      |
//! | 1.0 s      | `th` −0.051 | `as` −1.567         | 19.1 nats   | `th` −0.0054      |
//! | 3.0 s      | `th` −0.010 | `lo` −0.210         | 16.0 nats   | `th` −0.0054      |
//! | 6.0 s      | `th` −0.015 | `th` −0.177         | 6.8 nats    | `th` −0.0054      |
//! | 9.0 s      | `th` −0.013 | `th` −0.013         | 2.5 nats    | `th` −0.0054      |
//!
//! Padding a tail of 3 s or less **changes the language**. The fused in-graph
//! mean subtraction reduces over the time axis, so it sees the zeros; this is
//! the same effect the performance note below records for bucketing, at the
//! magnitude a short tail provokes.
//!
//! So the default is [`TailPolicy::SlideBack`]: a full-length, unpadded final
//! window that ends flush with the clip, at the cost of re-reading the audio it
//! overlaps. It keeps the whole plan to ONE graph shape, covers every sample,
//! and has the lowest row error of the four. [`TailPolicy::Partial`] is a
//! fraction better at the top-1 value and worse everywhere else, and it asks
//! the graph to specialize a new shape per distinct tail length — it also
//! produces genuinely noisier windows: on four repeats of the Thai clip its 2 s
//! tail scores as Lao, which drags [`ScorePooling::Max`] from a −0.039 call on
//! Thai to −0.629, against Lao at −0.762 — one bad window short of flipping the
//! whole clip.
//! [`TailPolicy::Drop`] is there for callers who would rather see nothing than
//! a re-read window; it discards up to one hop from the end.
//!
//! ### What none of this verifies
//!
//! Stated plainly, because the policy is chosen on these two oracles and
//! nothing else:
//!
//! - **There is no labelled long-form benchmark here.** Oracle 1 measures
//!   agreement with the model's own single-shot answer, which is self-consistency,
//!   not accuracy: if the model is wrong about a clip, the aggregation that
//!   reproduces that wrong answer scores best. Oracle 2's ground truth is real
//!   but constructed by repetition, so it tests robustness to windowing, not
//!   generalization to unseen speech.
//! - **The clip set is small and narrow** — sixteen clips over about five
//!   languages, from this repository's existing fixtures. Nothing here says how
//!   the policy behaves over the model's other hundred-odd languages, over
//!   telephone-band audio, or over speakers unlike these.
//! - **No long-form conversational or code-switched corpus was used.** The
//!   mixed-clip numbers come from splices, which have hard boundaries real
//!   code-switching does not.
//! - **Nothing here validates the window length against accuracy on long
//!   speech.** [`DEFAULT_WINDOW_SAMPLES`] is 10 s because self-consistency
//!   improves monotonically with window length up to it (81 % at 3 s, 87 % at
//!   5 s, 91 % at 10 s) while code-switch resolution gets worse above it, and
//!   because it is the one frame count [`Identifier::prewarm`] already warms.
//!   A different corpus could move it.
//!
//! [`MeanLogProbability`]: ScorePooling::MeanLogProbability
//! [`MeanProbability`]: ScorePooling::MeanProbability
//! [`Max`]: ScorePooling::Max
//! [`Vote`]: ScorePooling::Vote
//!
//! ```no_run
//! use coremlit::audio::lid::{Error, Identifier, ScorePooling, WindowPlan};
//!
//! # let speech_span_16k: Vec<f32> = Vec::new();
//! let identifier = Identifier::from_file("Models/lid/lid.mlmodelc")?;
//! identifier.prewarm()?; // warms exactly the default plan's window length
//!
//! let plan = WindowPlan::new();
//! for score in identifier.identify_long(&speech_span_16k, 3, &plan, ScorePooling::default())? {
//!   println!("{:>3} {:<12} {:.4}", score.code(), score.name(), score.probability());
//! }
//! # Ok::<(), Error>(())
//! ```
//!
//! # Reading the scores
//!
//! The graph's last op is a log-softmax, so the returned values are natural-log
//! probabilities that already sum to 1 under `exp`; no softmax runs in Rust.
//! [`Identifier::identify`] returns the top `k` of them, descending. See
//! [`LanguageScore`] for why the element is a struct rather than a tuple, and
//! why comparisons want the log form.
//!
//! ```no_run
//! use coremlit::audio::lid::{Error, Identifier};
//!
//! # let samples_16k: Vec<f32> = Vec::new();
//! let identifier = Identifier::from_file("Models/lid/lid.mlmodelc")?;
//! for score in identifier.identify(&samples_16k, 3)? {
//!   println!("{:>3} {:<12} {:.4}", score.code(), score.name(), score.probability());
//! }
//! # Ok::<(), Error>(())
//! ```
//!
//! # Compute placement
//!
//! [`DEFAULT_COMPUTE`] is [`ComputeUnits::All`], the house default, and the
//! measurements say it is the right one here — but read [`DEFAULT_COMPUTE`]'s
//! own docs before assuming it is right for a reason. It is not: the ANE arm of
//! this graph is pathological, and `All` currently happens to avoid it.
//!
//! # Performance notes (measured)
//!
//! - **Every unseen frame count costs a one-off specialization**, roughly
//!   55–97 ms, against a steady state of 9–23 ms. A service that sees many
//!   distinct clip lengths pays it repeatedly.
//! - **Padding to bucket lengths does not fix that for free.** The fused
//!   in-graph mean subtraction reduces over the time axis, so it sees the
//!   padding: bucketing shifts tail log-probabilities by up to 3 nats. Bucket
//!   only if you have measured that the shift does not matter for your
//!   decision — and note that 3 nats is what BUCKETING costs, not a bound on
//!   the effect: padding a 1 s clip out to 10 s moves the row by 19 nats and
//!   changes the reported language ("Clips longer than 30 s", the tail table).
//!   It is why [`TailPolicy`] has no padding variant.
//! - [`Identifier::prewarm`] pays the first prediction's graph specialization
//!   once, off the first real request — for ONE frame count.
//!
//! Fan-out is one [`Identifier`] per worker ([`crate::Model`] is `Send` but
//! deliberately not `Sync`).
//!
//! macOS only (built on [`crate`]).

use std::path::Path;

use crate::{ComputeUnits, DataType, Model, MultiArray};

pub mod aggregate;
pub mod error;
pub mod labels;
pub mod prediction;
pub mod window;

mod mel;

#[cfg(feature = "serde")]
mod compute_units_serde;

pub use aggregate::{ScorePooling, aggregate_windows};
pub use error::{
  ContractMismatch, Error, FrameCountOutOfRange, InvalidLogProbability, OutputShape, Result,
  WinditError,
};
pub use labels::{LABELS_JSON_LEN, Language, labels_json_bytes, languages};
pub use prediction::{LanguageScore, LogProbabilities, WindowLogProbabilities};
pub use window::{
  DEFAULT_HOP_SAMPLES, DEFAULT_MAX_WINDOWS, DEFAULT_WINDOW_SAMPLES, Span, TailPolicy, WindowPlan,
};

use crate::audio::lid::mel::{HOP, MelExtractor, N_MELS};

#[cfg(test)]
mod tests;

/// The sample rate this module's contract is defined at: callers decode and
/// resample to **16 kHz mono f32** before calling (sans-I/O — the workspace
/// convention; the model natively matches it).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Number of languages the model scores — the width of its output row and the
/// length of [`languages`].
pub const NUM_LANGUAGES: usize = 107;

// The roster's length is enforced by its TYPE — `labels`'s table is a
// `[Language; NUM_LANGUAGES]`, so a row added or dropped is a build error, not
// a runtime surprise. A `const` assert restating it here would be true by
// construction and would prove nothing; what the sibling `labels/tests.rs`
// proves instead is the claim that is NOT free: that those rows still match the
// committed asset, entry for entry.

/// Fewest mel frames the graph accepts. Below this the CoreML runtime rejects
/// the input outright; [`Identifier`] rejects it first, as
/// [`Error::FrameCountOutOfRange`].
pub const MIN_FRAMES: usize = 10;

/// Most mel frames the graph accepts (its `RangeDims` upper bound).
pub const MAX_FRAMES: usize = 3_001;

/// Fewest 16 kHz samples that reach [`MIN_FRAMES`]: `(MIN_FRAMES - 1) · 160`
/// = 1 440, or 0.09 s. One sample fewer produces 9 frames and is rejected.
pub const MIN_SAMPLES: usize = (MIN_FRAMES - 1) * HOP;

/// Most 16 kHz samples that still fit [`MAX_FRAMES`]:
/// `MAX_FRAMES · 160 - 1` = 480 159, or ~30.0099 s. Integer division means the
/// last 159 samples of that window are free — 480 000 (exactly 30 s) and
/// 480 159 both give 3 001 frames.
pub const MAX_SAMPLES: usize = MAX_FRAMES * HOP - 1;

/// Number of mel frames a clip of `n_samples` 16 kHz samples produces:
/// `1 + n_samples / 160`, integer division.
///
/// The graph's time axis is exactly this, so it is also the check
/// [`Identifier`] runs before calling the model. Total, and `const` — it
/// answers "will this clip be accepted?" without constructing anything:
///
/// ```
/// use coremlit::audio::lid::{MAX_FRAMES, MAX_SAMPLES, MIN_FRAMES, MIN_SAMPLES, frame_count};
///
/// assert_eq!(frame_count(MIN_SAMPLES), MIN_FRAMES);
/// assert_eq!(frame_count(MIN_SAMPLES - 1), MIN_FRAMES - 1); // rejected
/// assert_eq!(frame_count(MAX_SAMPLES), MAX_FRAMES);
/// assert_eq!(frame_count(MAX_SAMPLES + 1), MAX_FRAMES + 1); // rejected
///
/// // 16 000 samples is one second of audio.
/// assert_eq!(frame_count(16_000), 101);
/// ```
#[inline]
#[must_use]
pub const fn frame_count(n_samples: usize) -> usize {
  1 + n_samples / HOP
}

/// Default compute placement: [`ComputeUnits::All`].
///
/// MEASURED — and correct here for a reason worth writing down, because the
/// reason is luck rather than design:
///
/// | `computeUnits`         | load          | 13 s clip  | 3 s clip |
/// |------------------------|---------------|------------|----------|
/// | `All`                  | 113 ms        | 13.9 ms    | 4.8 ms   |
/// | `CpuAndGpu`            | 50–100 ms     | 13.8 ms    | 7.2 ms   |
/// | `CpuOnly`              | 21 ms         | 24.7 ms    | 6.7 ms   |
/// | `CpuAndNeuralEngine`   | **2 440 ms**  | **145 ms** | **36 ms**|
///
/// `All` is **bit-identical** to `CpuAndGpu` on both clips: it dispatches to
/// the GPU and never touches the ANE. The `CpuAndNeuralEngine` arm is
/// pathological — twenty times the load time, ten times the inference time —
/// and additionally emits `BNNS Graph Shape Deduction: Unsupported kernel id
/// 512` on stderr, i.e. the ANE compiler is falling back mid-graph rather than
/// running it.
///
/// So the house default is right today only because CoreML's own placement
/// heuristic declines the ANE for this graph. That is an OS-version-dependent
/// decision, not a property of the model, and a future macOS that chose
/// differently would make `All` ten times slower with no code change here.
/// This is a KNOWN RISK, recorded so it is a regression rather than a mystery:
/// if this door's latency ever jumps by an order of magnitude, check the
/// placement before anything else, and pin
/// [`IdentifierOptions::with_compute`]`(`[`ComputeUnits::CpuAndGpu`]`)`.
pub const DEFAULT_COMPUTE: ComputeUnits = ComputeUnits::All;

/// Declared feature names on the `.mlmodelc` (pinned by
/// `tests/lid/model_io.rs`).
mod names {
  pub const MEL_FEATURES: &str = "mel_features";
  pub const LOG_PROBABILITIES: &str = "log_probabilities";
}

#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_COMPUTE
}

/// Construction options for the [`Identifier`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IdentifierOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_compute",
      with = "crate::audio::lid::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for IdentifierOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl IdentifierOptions {
  /// Options matching the module default: [`DEFAULT_COMPUTE`].
  #[must_use]
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the graph on.
  #[inline]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }

  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }

  /// Sets [`Self::compute`] in place.
  #[inline]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// Spoken-language identifier: 16 kHz mono `&[f32]` in, ranked
/// [`LanguageScore`]s out.
///
/// The front end is a Rust log-mel port (the private `mel` submodule); the
/// CoreML graph maps `[1, frames, 60]` log-mel features to `[1, `
/// [`NUM_LANGUAGES`] `]` natural-log probabilities, already normalized.
///
/// `&self` inference (no mutable scratch): the FFT plan, window and filterbank
/// are built once at load and per-call buffers are local, so fan-out means one
/// [`Identifier`] per worker over a `Send` [`crate::Model`] (which is
/// deliberately `!Sync`).
#[derive(Debug)]
pub struct Identifier {
  model: Model,
  mel: MelExtractor,
}

impl Identifier {
  /// Loads the `.mlmodelc` from `model_path` with custom `options` — the
  /// primary constructor. Pins the model's I/O contract against its metadata at
  /// load.
  ///
  /// The input's time axis is flexible, so it cannot be pinned to one value
  /// here: [`crate::FeatureInfo::shape`] reports the graph's DEFAULT shape for
  /// a `RangeDims` input (`[1, 301, 60]` for this artifact), not its bounds,
  /// which CoreML does not expose through that snapshot. What is checked is
  /// everything that is fixed — rank 3, unit batch, 60 mel columns, f32 — and
  /// the frame bounds themselves are pinned by `tests/lid/model_io.rs` against
  /// the live runtime.
  ///
  /// No model is bundled: the `.mlmodelc` is a directory artifact, staged
  /// gitignored under `Models/lid/`.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model; [`Error::ContractMismatch`]
  /// if its I/O contract mismatches.
  pub fn load(model_path: impl AsRef<Path>, options: IdentifierOptions) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let description = model.description();

    let input_expected = format!("[1, {MIN_FRAMES}..={MAX_FRAMES}, {N_MELS}] float32");
    let input = description.input(names::MEL_FEATURES).ok_or_else(|| {
      ContractMismatch::new(
        names::MEL_FEATURES,
        input_expected.clone(),
        "missing".to_owned(),
      )
    })?;
    let shape = input.shape();
    let time_axis_plausible = shape.len() == 3
      && shape[0] == 1
      && shape[2] == N_MELS
      && (MIN_FRAMES..=MAX_FRAMES).contains(&shape[1]);
    if !time_axis_plausible || input.data_type() != Some(DataType::F32) {
      return Err(
        ContractMismatch::new(
          names::MEL_FEATURES,
          input_expected,
          describe(shape, input.data_type()),
        )
        .into(),
      );
    }

    let output_expected = format!("[1, {NUM_LANGUAGES}] float32");
    let output = description
      .output(names::LOG_PROBABILITIES)
      .ok_or_else(|| {
        ContractMismatch::new(
          names::LOG_PROBABILITIES,
          output_expected.clone(),
          "missing".to_owned(),
        )
      })?;
    if output.shape() != [1, NUM_LANGUAGES] || output.data_type() != Some(DataType::F32) {
      return Err(
        ContractMismatch::new(
          names::LOG_PROBABILITIES,
          output_expected,
          describe(output.shape(), output.data_type()),
        )
        .into(),
      );
    }

    Ok(Self {
      model,
      mel: MelExtractor::new(),
    })
  }

  /// Loads the `.mlmodelc` with [`IdentifierOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, IdentifierOptions::new())
  }

  /// The full `[`[`NUM_LANGUAGES`]`]` row of **natural-log probabilities**, in
  /// model column order (index `i` is `languages()[i]`) — the parity seam and
  /// the power-user escape.
  ///
  /// Already log-softmaxed by the graph: the values are `<= 0` and `exp` over
  /// the row sums to 1. Nothing is applied on top of them here.
  ///
  /// `samples_16k` is 16 kHz mono and must produce [`MIN_FRAMES`]..=
  /// [`MAX_FRAMES`] frames ([`MIN_SAMPLES`]..=[`MAX_SAMPLES`] samples); it is
  /// never padded or truncated to fit.
  ///
  /// # Errors
  /// [`Error::FrameCountOutOfRange`] if the clip is too short or too long
  /// (empty audio lands here too — zero samples is one frame, below
  /// [`MIN_FRAMES`]); [`Error::NonFiniteInput`] if any sample is NaN or
  /// infinite (it would silently poison the mel); [`Error::Tensor`] /
  /// [`Error::Prediction`] on a tensor or CoreML failure; [`Error::OutputShape`]
  /// if the predicted row's shape diverges from `[1, `[`NUM_LANGUAGES`]`]`;
  /// [`Error::NonFiniteOutput`] if the model emits a NaN or infinite score
  /// (model corruption — never reaches ranking).
  pub fn log_probabilities(&self, samples_16k: &[f32]) -> Result<Vec<f32>> {
    let frames = validate_frame_range(samples_16k.len())?;

    let mut features = vec![0.0f32; frames * N_MELS];
    self.mel.extract_into(samples_16k, &mut features)?;

    // Time-major mel [frames, 60] maps directly onto the row-major
    // `mel_features [1, frames, 60]` contract.
    let input = MultiArray::from_slice(&[1, frames, N_MELS], &features)?;
    let mut outputs = self.model.predict_with(&[(names::MEL_FEATURES, &input)])?;
    let scores = outputs.take(names::LOG_PROBABILITIES).ok_or_else(|| {
      crate::PredictionError::MissingOutput {
        name: names::LOG_PROBABILITIES.to_owned(),
      }
    })?;
    if scores.shape() != [1, NUM_LANGUAGES] {
      return Err(OutputShape::new(scores.shape().to_vec(), vec![1, NUM_LANGUAGES]).into());
    }

    let mut row = vec![0.0f32; NUM_LANGUAGES];
    scores.copy_into::<f32>(&mut row)?;
    if let Some(index) = row.iter().position(|value| !value.is_finite()) {
      return Err(Error::NonFiniteOutput(index));
    }
    Ok(row)
  }

  /// The top `k` languages for `samples_16k`, **descending** by log
  /// probability, ties broken by ascending model column.
  ///
  /// `k == 0` returns an empty vec without running the model (it still applies
  /// the same input validation, so a bad clip is still reported); `k` above
  /// [`NUM_LANGUAGES`] saturates.
  ///
  /// # Errors
  /// As [`Self::log_probabilities`]; [`Error::UnknownLanguageIndex`] is
  /// defensive-only.
  pub fn identify(&self, samples_16k: &[f32], k: usize) -> Result<Vec<LanguageScore>> {
    if k == 0 {
      validate_frame_range(samples_16k.len())?;
      check_finite_samples(samples_16k)?;
      return Ok(Vec::new());
    }
    let scores = self.log_probabilities(samples_16k)?;
    prediction::top_k_from_scores(scores.into_iter().enumerate(), k)
  }

  /// The long-clip primitive: one log-probability row per planned window,
  /// paired with the [`Span`] it was scored over — ALWAYS exposed, so
  /// code-switch detection ("where did it change language") is a caller-side
  /// read of `windows[i].value().as_slice()` against `windows[i].span()`, with
  /// no second API.
  ///
  /// Slices `samples_16k` at the plan's offsets and runs one
  /// [`Self::log_probabilities`] per span. Runs sequentially: [`crate::Model`]
  /// is `!Sync`, so windows share one identifier on one thread.
  ///
  /// Every span a [`WindowPlan`] produces is a length the graph accepts, so no
  /// window is ever rejected mid-clip for its size. Under the default
  /// [`TailPolicy::SlideBack`] every span is exactly one window long, so the
  /// whole clip costs ONE graph specialization; see
  /// [`DEFAULT_WINDOW_SAMPLES`] for why the default is the length
  /// [`Self::prewarm`] warms.
  ///
  /// # Errors
  /// [`Error::FrameCountOutOfRange`] if the whole clip is shorter than
  /// [`MIN_SAMPLES`] (there is no upper bound here — that is the point);
  /// [`Error::NonFiniteInput`] if any sample is NaN or infinite, carrying its
  /// index **in the clip** (the whole clip is scanned once up front, so the
  /// index is never window-relative); [`Error::Windowing`] if the plan exceeds
  /// [`WindowPlan::max_windows`] or a buffer cannot be allocated; otherwise any
  /// per-window [`Self::log_probabilities`] error.
  pub fn log_probabilities_windows(
    &self,
    samples_16k: &[f32],
    plan: &WindowPlan,
  ) -> Result<Vec<WindowLogProbabilities>> {
    validate_long_input(samples_16k)?;
    let spans = plan.spans(samples_16k.len())?;
    // Fallible reservation: the cap already bounds `spans.len()`, but the
    // result vector is still caller-geometry-sized, so reserve it checked
    // rather than risk an infallible `with_capacity` abort under memory
    // pressure.
    let mut out = Vec::new();
    out.try_reserve_exact(spans.len()).map_err(|_| {
      Error::Windowing(WinditError::AllocFailed {
        elements: spans.len(),
      })
    })?;
    for span in spans {
      let row = self.log_probabilities(&samples_16k[span.start()..span.end()])?;
      out.push(WindowLogProbabilities::new(
        LogProbabilities::new(row),
        span,
      ));
    }
    Ok(out)
  }

  /// The composed long-clip answer: scores each planned window and folds the
  /// per-window rows into one clip-level row under `pooling`, then returns its
  /// top `k` languages — the long-clip counterpart of [`Self::identify`], with
  /// no 30 s ceiling.
  ///
  /// The fold streams through an O([`NUM_LANGUAGES`]) accumulator, so a clip of
  /// any length retains one row rather than one per window; use
  /// [`Self::log_probabilities_windows`] when per-window access is wanted.
  ///
  /// A clip that already fits one window returns **exactly** what
  /// [`Self::identify`] returns for it, bit for bit: the plan is a single span
  /// and a one-window fold is the identity, whatever the pooling. So this is a
  /// drop-in for `identify` rather than a separate regime with a boundary to
  /// straddle.
  ///
  /// `k == 0` returns an empty vec without running the model OR any windowing,
  /// so the [`WindowPlan::max_windows`] cap does not apply to it; it still
  /// applies the same clip-level validation the scoring path would.
  ///
  /// # Errors
  /// As [`Self::log_probabilities_windows`]; [`Error::UnknownLanguageIndex`] is
  /// defensive-only. ([`Error::EmptyWindows`] is unreachable — a clip that
  /// passes validation always plans at least one span — and so is
  /// [`Error::ZeroMassAggregate`]: [`Self::log_probabilities`] rejects a
  /// non-finite score, so every row this folds is all-finite and no pooling can
  /// zero the whole clip out.)
  ///
  /// [`NUM_LANGUAGES`]: NUM_LANGUAGES
  pub fn identify_long(
    &self,
    samples_16k: &[f32],
    k: usize,
    plan: &WindowPlan,
    pooling: ScorePooling,
  ) -> Result<Vec<LanguageScore>> {
    validate_long_input(samples_16k)?;
    if k == 0 {
      // Matches `identify`: no model, no windowing, and therefore no cap — but
      // a clip that is too short or non-finite is still refused rather than
      // waved through as an empty result.
      return Ok(Vec::new());
    }
    let spans = plan.spans(samples_16k.len())?;
    let mut acc = aggregate::Accumulator::new(pooling);
    for span in spans {
      let row = self.log_probabilities(&samples_16k[span.start()..span.end()])?;
      acc.push(&LogProbabilities::new(row), span.len());
    }
    acc.finish()?.top_k(k)
  }

  /// Runs one throwaway inference on a fixed synthetic clip to fully specialize
  /// the prediction path, so the first user-facing request is warm.
  /// Construction pays the model load; what it does NOT pay is the first
  /// prediction's own graph specialization.
  ///
  /// **This warms ONE frame count**, and it is deliberately
  /// [`DEFAULT_WINDOW_SAMPLES`] long — 1 001 frames, 10 s — so one `prewarm`
  /// covers every window of a default [`WindowPlan`], however long the clip.
  /// The length is read from that constant rather than restated, so the two
  /// cannot drift apart. The specialization is per frame count (module docs,
  /// "Performance notes"), so a service that will see one OTHER clip length
  /// should prewarm at that length instead, by calling
  /// [`Self::log_probabilities`] on a throwaway buffer of the right size — and
  /// a plan using [`TailPolicy::Partial`] pays one more specialization per
  /// distinct tail length, which no prewarm can anticipate.
  ///
  /// # Errors
  /// As [`Self::log_probabilities`]; a failure here surfaces a broken model at
  /// prewarm time rather than on the first request.
  pub fn prewarm(&self) -> Result<()> {
    let rate = SAMPLE_RATE_HZ as f32;
    let signal: Vec<f32> = (0..DEFAULT_WINDOW_SAMPLES as usize)
      .map(|i| 0.5 * (core::f32::consts::TAU * 440.0 * (i as f32 / rate)).sin())
      .collect();
    self.log_probabilities(&signal)?;
    Ok(())
  }
}

/// Reject a clip whose mel frame count the graph would refuse, returning that
/// frame count when it is in range.
///
/// A free fn so the guard is hermetically testable without a model, and so the
/// rejection happens strictly before [`Model::predict_with`] — the CoreML
/// runtime's own message names an internal axis index and would have to be
/// string-matched to act on.
fn validate_frame_range(n_samples: usize) -> Result<usize> {
  let frames = frame_count(n_samples);
  if !(MIN_FRAMES..=MAX_FRAMES).contains(&frames) {
    return Err(FrameCountOutOfRange::for_samples(n_samples).into());
  }
  Ok(frames)
}

/// Reject a clip the LONG path must not see: shorter than [`MIN_SAMPLES`] (no
/// window could be scored, and windowing cannot rescue a clip that is simply
/// too short), or carrying a NaN/±∞ sample.
///
/// There is deliberately no upper bound: lifting it is what the long path is
/// for. The finite scan runs over the WHOLE clip once, before any window is
/// sliced, so [`Error::NonFiniteInput`] carries a clip-absolute index rather
/// than one relative to whichever window happened to contain it.
fn validate_long_input(samples: &[f32]) -> Result<()> {
  if samples.len() < MIN_SAMPLES {
    return Err(FrameCountOutOfRange::for_samples(samples.len()).into());
  }
  check_finite_samples(samples)
}

/// Reject a NaN/±∞ sample ([`Error::NonFiniteInput`]) — it would poison the mel
/// frames it touches and, through the whole-utterance `top_db` floor, every
/// other frame as well.
fn check_finite_samples(samples: &[f32]) -> Result<()> {
  match samples.iter().position(|value| !value.is_finite()) {
    Some(index) => Err(Error::NonFiniteInput(index)),
    None => Ok(()),
  }
}

/// Human-readable `shape dtype` rendering for [`ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}
