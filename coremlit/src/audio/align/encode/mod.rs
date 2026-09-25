//! CoreML wrapper over a fixed-window CTC acoustic encoder: `waveform [1, W]`
//! f32 in, `emissions [1, T, V]` f32 out. The encoder is an
//! [`Aligner`](crate::audio::align::Aligner)'s, and only an aligner's: see
//! "The encoder is the aligner's" below. The staged
//! `base960h_aligner.mlmodelc` (design spec §3 Candidate A) is `W = 960_000`
//! (60 s @ 16 kHz), `T = 2999`, `V = 29`, 20 ms/frame (stride 320 samples @
//! 16 kHz) — ground truth pinned by
//! `tests/model_io.rs::base960h_aligner_io_matches_spec`.
//!
//! `W`, `T` and `V` are READ at load (`Encoder::window_samples`,
//! `Encoder::frames`, `Encoder::vocab_size`), never pinned: a model
//! converted at another window, or one that spells another alphabet, is as
//! correct through this door as the staged one. What no declaration says is
//! the model's [`AcousticContract`], which the caller states and this door
//! checks: `T` must be the frames the contract's front-end geometry makes of
//! `W`, else [`AlignerError::FrameCountMismatch`] at load. Pairing `V` with a
//! vocabulary is [`crate::audio::align::aligner::Aligner`]'s, which refuses a
//! table of another size at load.
//!
//! # The encoder is the aligner's
//!
//! An encoder runs one model under one [`AcousticContract`], and asry's seam
//! has to tokenize with the same contract's blank, stride and tokenization:
//! its `prepare` pads and silence-masks a chunk for THAT seam, and its `finish`
//! reads the emissions' columns by THAT seam's blank. Nothing in asry's
//! `PreparedChunk` or `Emissions` names the contract they belong to, so a
//! prepared chunk of one seam and the emissions of another model's encoder
//! would compose without an error and align against the wrong columns.
//!
//! So the composition is not public. `Encoder` and `EncoderInput` are this
//! crate's, and the one road from audio to words is
//! [`Aligner::align_chunk`](crate::audio::align::Aligner::align_chunk), whose
//! aligner owns the seam, the encoder and the contract they were both built
//! from. No public value can cross from one aligner to another:
//!
//! ```compile_fail,E0603
//! use coremlit::audio::align::encode::Encoder;
//! ```
//!
//! ```compile_fail,E0603
//! use coremlit::audio::align::encode::EncoderInput;
//! ```
//!
//! # Fixed-window bridging
//!
//! `Encoder::emissions` hides the model's fixed window (60 s on the staged
//! model) behind a variable-length `&[f32]` contract, mirroring asry's own
//! encoder call shape (`(1, T) -> (1, T', V)`, `asry/src/runner/aligner/algorithm/
//! encode.rs`) as closely as a fixed-window CoreML graph allows:
//!
//! - **Longer than the window** (`Encoder::window_samples`): rejected with
//!   [`AlignError::InputTooLong`] rather than silently truncated, before any
//!   prediction. The caller is responsible for chunking audio to at most the
//!   window before calling — see "60 s clamp vs asry's `MAX_CHUNK_SIZE`" below
//!   for why the staged model's ceiling is far tighter than asry's own
//!   per-chunk cap.
//! - **Shorter**: zero-padded up to exactly the window, never rejected —
//!   unlike `dia-coreml`'s `SegmentModel::infer`, which
//!   rejects rather than pads short input
//!   (`crates/dia-coreml/src/segment/mod.rs`'s "dia contract match"
//!   section). wav2vec2 is not causal, so whether padding perturbs
//!   in-range emissions was an open empirical question — the B5 word-timing
//!   parity gate (`tests/parity_words.rs`) has since MEASURED it: fed a full
//!   window the encoder is frame-exact against asry's ONNX reference, and
//!   zero-padding a short clip leaves the median boundary frame-identical
//!   (p90 40.1 ms on the 11 s `jfk.wav`). See the crate root's "How far you
//!   can trust the timings" table for both clips.
//! - **`emissions` frames past the real (non-padded) audio**: truncated
//!   away, to the frames the contract's geometry makes of the real audio —
//!   see `Encoder::emissions`'s doc for the exact formula and why it must be
//!   clamped to the model's actual frame count.
//!
//! # 60 s clamp vs asry's `MAX_CHUNK_SIZE`
//!
//! asry's own chunk-size ceiling is `pub const MAX_CHUNK_SIZE: Duration =
//! Duration::from_secs(600);` (`asry/src/core/transcriber.rs:137`) — 10
//! minutes. The staged model's window, [`ENCODER_WINDOW_SAMPLES`], is 60 s,
//! ten times tighter. This is a deliberate, DOCUMENTED divergence, not a
//! parity target: asry's 600 s cap bounds per-chunk RAM for its own
//! (non-fixed-window) ONNX wav2vec2 path, which allocates proportionally to
//! whatever length it is given. `base960h_aligner.mlmodelc` allocates a
//! *fixed* 960,000-sample input / `[1, 2999, 29]` output tensor pair
//! regardless of how much of it is real audio, so there is no equivalent "let
//! it grow" option on this side — the model's own fixed graph is the ceiling,
//! not a tunable, and a model converted at another window has that window as
//! its ceiling. A caller must chunk audio to at most the encoder's window
//! (`Encoder::window_samples`) before calling `Encoder::emissions`; that
//! chunking responsibility is explicitly out of scope here (design spec §7's
//! data flow already assumes per-chunk audio, not a whole-file stream).
//!
//! # The log-prob door: `from_log_probs`, not `from_logits`
//!
//! `Encoder::emissions` wraps the raw `emissions` tensor into an
//! [`Emissions`] through [`Emissions::from_log_probs`] — the log-prob door —
//! with **no softmax or log-softmax applied**. The model's own graph already
//! ends in one (`Models/alignkit/base960h_aligner.mlmodelc/model.mil`, final
//! ops — this is graph truth, not an inference from measured values):
//!
//! ```text
//! linear_73_cast_fp16       = linear(...)                          // CTC head → logits
//! var_849_softmax_cast_fp16 = softmax(axis = -1, x = linear_73_cast_fp16)
//! var_849_cast_fp16         = log(epsilon = 0x1p-149, x = var_849_softmax_cast_fp16)
//! emissions                 = cast(dtype = fp32, x = var_849_cast_fp16)
//! ```
//!
//! The reason to prefer this door is NOT that re-applying a log-softmax
//! would corrupt the values. It would not: **log-softmax is exactly
//! idempotent.** For `y = log_softmax(x)`, `lse(y) = ln Σ exp(x_j − lse(x)) =
//! ln 1 = 0`, so `log_softmax(y) = y`. Routing genuine log-probs through
//! [`Emissions::from_logits`] (asry's raw-logit door, which applies
//! `log_softmax_with_finite_guard`) would be a numerical no-op.
//!
//! The real reason is that this door **refuses to paper over a model-artifact
//! swap**. [`Emissions::from_logits`] would apply its own
//! `log_softmax_with_finite_guard` and **re-normalize whatever it is handed** —
//! genuine log-probs (a no-op, per above) *or* raw logits — into a plausible
//! log-prob domain, then align on the result forever. Taking `from_log_probs`
//! consumes the tensor **as-is**, so a future model revision that ships a
//! raw-logit CTC head — entirely plausible, since that is the *standard*
//! wav2vec2 export, and asry's own ONNX model does exactly that
//! (`asry/src/runner/aligner/algorithm/encode.rs` takes the `from_logits` door)
//! — is caught rather than absorbed.
//!
//! Caught by what, exactly, is the subtle part, and the earlier revisions of
//! this doc got it wrong. `from_log_probs`'s own `O(T·V)` scan (every element
//! finite ∧ `<= 0`) is **necessary but not sufficient**: it rejects a raw-logit
//! head only when some logit is *positive*. Logits are defined only up to an
//! additive per-frame constant, so a raw-logit row shifted wholly into, say,
//! `[-20, -10]` — or the degenerate all-zeros row, `exp(0) = 1` on every class —
//! is finite and `<= 0` on every cell and sails straight through that scan while
//! being nothing like a probability distribution. What actually pins the tensor
//! to the log-probability domain is the **per-frame logsumexp guard**
//! (`check_log_prob_normalization`, [`log_prob_sum_tolerance`]): a genuine CTC
//! log-prob row satisfies `logsumexp(row) = ln Σ exp(log p_j) = ln Σ p_j =
//! ln 1 = 0` by construction, while an un-normalized row is off by whole units
//! (the all-zeros row by `ln 29 ≈ 3.37`, a `[-20, -10]` shifted row by `>= 6.6`).
//! That guard and the `<= 0` scan are together what make "these really are
//! log-probs" a *checked* contract rather than a hope — for any artifact loaded
//! through the public API, not merely the one reviewed here. See "The
//! normalization guard" below. The staged artifact's contract adds a third,
//! its sentinel band (below).
//!
//! For the same reason the raw tensor is passed through **unclamped**. The
//! graph's `softmax` output is in `[0, 1]` by construction, so its `log` is
//! `<= 0` *guaranteed by the graph* (measured max is exactly `0.0` on every
//! compute placement). An unbounded `.min(0.0)` would be actively dangerous —
//! it is exactly what would mask a raw-logit head's *positive maxima* from the
//! `<= 0` scan above (the logsumexp guard would still catch a shifted-negative
//! head, but suppressing any model-swap signal is the wrong direction). If a
//! clamp is ever needed here, it must be **bounded** to a pinned slack, never
//! open-ended.
//!
//! # The sentinel band: one artifact's measurement, in its contract
//!
//! [`Emissions::from_log_probs`]'s scan bounds the emissions from **above**
//! (`<= 0`) and rules out non-finite values. It does not bound them from
//! **below**, and it cannot: `-45440` is finite and negative, so the staged
//! model's ANE-corrupted matrix — every softmax output under the fp16 floor
//! underflowed to `0`, every `log(0)` saturated to that sentinel
//! ([`DEFAULT_ENCODER_COMPUTE`]) — sails straight through it and aligns to
//! plausible, silently wrong timings.
//!
//! That gap was reachable from this crate's own public API
//! ([`crate::audio::align::AlignerOptions::with_compute`]),
//! and it was the *measured* defect, not the hypothetical one the paragraph
//! above guards against. So when the model's contract carries a
//! [`SentinelBand`], `Encoder::emissions` scans for it: any cell in the band
//! is [`AlignError::CorruptEmissions`], a typed error that NAMES the compute
//! placement the encoder was loaded with. Loud, and self-diagnosing.
//!
//! Only [`AcousticContract::BASE960H`] carries a band. No law bounds a
//! log-probability from below, so no finite threshold tells a sentinel from a
//! valid value for every model: `[0, -40000]` is a normalized row of some
//! model, and it lies inside the staged model's band. A band is therefore a
//! measurement of one artifact, and a contract made with
//! [`AcousticContract::new`] has none.
//!
//! # The normalization guard: per-frame logsumexp
//!
//! A sentinel band and `from_log_probs`'s `<= 0` scan bound each *cell*;
//! neither checks that a frame's `V` log-probs describe a *distribution*.
//! `check_log_prob_normalization` does, and it is what makes the "The log-prob
//! door" section's model-swap claim actually true. For every truncated frame it
//! recomputes `logsumexp` over the vocab axis (in `f64`, so the bound reflects
//! the model's own fp16 deviation, not this crate's summation error) and rejects
//! the matrix with [`AlignError::UnnormalizedEmissions`] — naming the worst frame
//! and its `logsumexp` — the moment any frame's `|logsumexp|` exceeds
//! [`log_prob_sum_tolerance`] of the head's width. A genuine CTC log-prob frame
//! sums to 1 in probability space, so its `logsumexp` is `0`; a raw-logit frame
//! (even one shifted wholly `<= 0`), or an all-zeros frame, is off by whole
//! units. This is
//! the check that closes the bypass the `<= 0` scan leaves open, for *any*
//! artifact a caller loads through the public API — not only the reviewed one,
//! whose normalization `tests/model_io.rs::emissions_are_log_probs_not_raw_logits`
//! also pins. The identity is CTC's (a frame of log-probabilities sums to one);
//! the allowance is fp16 arithmetic's, scaled by the head's width
//! ([`log_prob_sum_tolerance`]), so it holds for a head of any width.
//!
//! Cost: one `f64` exp/sum/log over `[<= 2999, 29]` per window on the staged
//! model — ~87k operations against a 0.74 s CoreML predict. Not measurable
//! ([`log_prob_sum_tolerance`]'s "Cost").

use core::num::NonZeroUsize;
use std::{borrow::Cow, path::Path};

use crate::{
  ComputeUnits, DataType, Model, ModelDescription, MultiArray,
  model::contract::{
    Checked, ContractViolation, Dim, FeatureContract, LoadContract, Rendered, StateContract,
  },
};
use asry::emissions::{Emissions, PreparedChunk};

use crate::audio::align::{
  acoustic::{
    ASRY_PREPARE_PAD_SAMPLES, AcousticContract, AcousticGeometry, OutputKind, SentinelBand,
  },
  error::{
    AlignError, AlignerError, ContractMismatch, CorruptEmissions, FrameCountMismatch, InputTooLong,
    OutputShape, UnnormalizedEmissions, UnprovableNormalization,
  },
};

/// The staged `base960h_aligner.mlmodelc`'s input window: 960,000 samples,
/// 60 s @ 16 kHz. Pinned by `tests/model_io.rs::base960h_aligner_io_matches_spec`
/// (`waveform [1, 960_000]`). See the module doc's "Fixed-window bridging" and
/// "60 s clamp vs asry's `MAX_CHUNK_SIZE`" sections.
///
/// A fact of that artifact, not of this door: an aligner's encoder reads its
/// model's own window at load
/// ([`Aligner::window_samples`](crate::audio::align::Aligner::window_samples)),
/// and this constant is what the staged model declares there.
pub const ENCODER_WINDOW_SAMPLES: usize = 960_000;

/// wav2vec2's frame stride: 20 ms @ 16 kHz, [`AcousticGeometry::WAV2VEC2`]'s
/// stride and so the staged model's, and asry's own `hop_samples` default
/// (`asry/src/runner/aligner/aligner.rs`'s `Aligner::from_paths` doc:
/// "`hop_samples` defaults to 320").
///
/// A fact of that geometry, not of this door: an aligner's encoder truncates by
/// the stride of its model's [`AcousticContract`], and the aligner hands the
/// seam that same stride. This constant is what [`AcousticContract::BASE960H`]
/// states.
pub const HOP_SAMPLES: usize = 320;

/// The one output frame count `base960h_aligner.mlmodelc` declares for its
/// [`ENCODER_WINDOW_SAMPLES`] window: **2999**, the wav2vec2 feature
/// extractor's output length for one full window, `floor((960_000 - 400) /
/// 320) + 1`. The load no longer requires it (a model's frame count is read
/// and checked against its contract's geometry), but the staged artifact's
/// tests and the compile-time tie below do.
const EXPECTED_OUTPUT_FRAMES: usize = 2_999;

/// Ties the staged artifact's three numbers to [`AcousticContract::BASE960H`]
/// at **compile time**: its geometry must make [`EXPECTED_OUTPUT_FRAMES`] of
/// [`ENCODER_WINDOW_SAMPLES`], and its stride must be [`HOP_SAMPLES`] — the
/// check `Encoder::load` runs at load, here run on the constants, so
/// re-spelling any of them without the others is a BUILD failure, not a model
/// that fails to load.
const _: () = {
  let geometry = AcousticContract::BASE960H.geometry();
  assert!(
    geometry.frames(ENCODER_WINDOW_SAMPLES) == EXPECTED_OUTPUT_FRAMES,
    "the staged contract's geometry must make the staged model's 2999 frames of its window"
  );
  assert!(
    geometry.stride().get() as usize == HOP_SAMPLES,
    "HOP_SAMPLES must be the staged contract's stride"
  );
};

/// The feature names this door sends and reads: the staged
/// `base960h_aligner.mlmodelc`'s (pinned by
/// `tests/model_io.rs::base960h_aligner_io_matches_spec`). They are the door's
/// interface, checked at load rather than trusted: a model under other names
/// is refused there ([`AlignerError::ContractMismatch`] for a missing
/// `waveform` or `emissions`, [`AlignerError::UnsatisfiableInput`] for a
/// required input the door never sends).
mod names {
  pub const WAVEFORM: &str = "waveform";
  pub const EMISSIONS: &str = "emissions";
}

/// The default compute placement of an aligner's encoder
/// ([`AlignerOptions::compute`](crate::audio::align::AlignerOptions::compute)).
///
/// **`CpuOnly` is a correctness requirement of this model, not a performance
/// preference.** Do NOT "optimise" it back to `ComputeUnits::All`: on the ANE
/// this model produces a corrupted emission matrix.
///
/// # Why
///
/// `base960h_aligner.mlmodelc` does not end in a fused, numerically-stable
/// `log_softmax`. Its graph decomposes the CTC tail into an fp16 `softmax`
/// followed by a separate fp16 `log`
/// (`Models/alignkit/base960h_aligner.mlmodelc/model.mil`, final ops):
///
/// ```text
/// var_849_softmax_cast_fp16 = softmax(axis = -1, x = linear_73_cast_fp16)
/// var_849_cast_fp16         = log(epsilon = 0x1p-149, x = var_849_softmax_cast_fp16)
/// ```
///
/// That `log`'s anti-`log(0)` guard is `epsilon = 0x1p-149` (2⁻¹⁴⁹) — far
/// below fp16's smallest subnormal (2⁻²⁴ ≈ `5.96e-8`) — so inside an fp16
/// `log` it rounds to zero and the guard is **inert**. On the ANE any softmax
/// output beneath the fp16 floor therefore underflows to 0, and `log(0)`
/// saturates to ≈ `-45440`: a sentinel standing where an ordinary log-prob
/// (`-19.0` … `-21.75`) belongs.
///
/// Measured on `jfk.wav` (549 frames × 29 = 15,921 cells). `load` is a cold
/// first load — the ANE compilation is cached afterwards, so a warm `All`
/// load is fast and hides nothing:
///
/// | compute | load (cold) | predict | `min(emissions)` | sentinel cells |
/// |---|---|---|---|---|
/// | `CpuOnly` | 0.68 s | **0.74 s** | **-30.81** | **0** |
/// | `All` | 308 s | 2.15 s | `-45440` | 2,667 (16.7%) |
/// | `CpuAndNeuralEngine` | 508 s | 2.32 s | `-45440` | 2,667 (16.7%) |
/// | `CpuAndGpu` | 0.37 s | 3.55 s | -30.02 | 0 |
///
/// The corruption is bit-identical run to run — systematic, not
/// nondeterminism — and it reaches the output: on the `All` path 8 of the 22
/// jfk words shift in time (`ask` starts 881.6 ms early — 7533.7 ms against the
/// correct 8415.3 ms) and all 22 differ in timing and/or score. Those
/// word-shift figures are a pre-truncation-fix measurement, whose exact ms
/// shifted with the fix; unlike the post-fix 549-frame table above they show
/// only the ANE corruption's *direction* and *magnitude*, not timings against
/// the current frame geometry. There is no trade-off to weigh, because the ANE
/// placement is ~450× slower to load, ~3× slower to predict, **and** wrong;
/// `CpuOnly` additionally has the best predict time of any numerically-correct
/// placement.
///
/// Running this model on the ANE would require **re-converting the artifact**
/// with a fused (or fp32) `log_softmax` tail. That is a model fix, not a code
/// fix — nothing in this crate can recover the underflowed cells.
///
/// Pinned by `tests::emissions_have_no_fp16_log_zero_sentinel`, which builds
/// its encoder from this constant (never a hardcoded placement) and fails on
/// `All`.
///
/// This is the *default*, not a lock:
/// [`AlignerOptions::with_compute`](crate::audio::align::AlignerOptions::with_compute)
/// still accepts any placement. What stops an ANE override from silently
/// corrupting a caller's timings is the staged contract's
/// [`SentinelBand::Fp16Saturation`] — a value-domain guard in the encoder, not
/// a ban on the placement.
///
/// For a model loaded with its own contract `CpuOnly` is a default, not an
/// assumption about the model: every graph runs on the CPU, and whether another
/// placement computes a given graph correctly is that model's to show (the
/// staged one's ANE placements do not).
pub const DEFAULT_ENCODER_COMPUTE: ComputeUnits = ComputeUnits::CpuOnly;

/// fp16's unit roundoff, `2^-11`: the largest relative error of one rounding to
/// the nearest fp16. fp16 is the least precise float format CoreML computes a
/// softmax in.
const FP16_UNIT_ROUNDOFF: f64 = 1.0 / 2048.0;

/// Largest per-frame `|logsumexp|` an aligner's encoder accepts from a head of
/// `vocab_size` classes as normalized log-probabilities:
/// **`2·(V + 1)·2^-11`**, `0.0293` at the staged model's 29 classes. A frame
/// whose `logsumexp` over the vocab axis exceeds it in magnitude is not a
/// probability distribution — a genuine CTC log-prob frame satisfies
/// `logsumexp = ln Σ exp(log p_j) = ln Σ p_j = ln 1 = 0` by construction — and
/// the encoder rejects the whole matrix with
/// [`AlignError::UnnormalizedEmissions`] rather than align on it.
///
/// # Why a normalization check, on top of the `<= 0` scan
///
/// It is the half of the log-prob contract [`Emissions::from_log_probs`]'s
/// `finite ∧ <= 0` scan cannot cover — the model-swap guard the module doc's
/// "The log-prob door" advertises but that scan alone does not deliver. That
/// scan rejects a raw-logit CTC head only when some logit is *positive*; logits
/// are defined only up to an additive per-frame constant, so a raw-logit frame
/// shifted wholly into `[-20, -10]` — or the degenerate all-zeros frame,
/// `exp(0) = 1` on every class — is finite and `<= 0` on every cell yet carries
/// no distribution. The `logsumexp` identity is the property that tells the two
/// apart. See the module doc's "The normalization guard: per-frame logsumexp".
///
/// # Why this allowance, for every width
///
/// The identity is a law of CTC log-probabilities; the allowance is a law of the
/// arithmetic that computes them. A log-softmax computed in fp16 normalizes a
/// frame only to within its roundings, each at most fp16's unit roundoff
/// `u = 2^-11` relative: summing `V` positive terms rounds up to `V − 1`
/// times, and the exponential, the division and the logarithm add a few more,
/// the last weighted by the frame's entropy (at most `ln V`). So a genuine frame's `|logsumexp|`
/// stays within about `(V + 2 + 2 ln V)·u`, which `2·(V + 1)·u` covers at every
/// width (`V ≥ 2 ln V` for every `V`). A precision finer than fp16 (the CPU's
/// fp32, `u = 2^-24`) stays far inside it.
///
/// It grows with the head because the rounding does: a fixed allowance measured
/// on one 29-class head would refuse the genuine emissions of an fp16 head of a
/// few hundred classes or more, the size of a character-level vocabulary for a
/// script with many characters. What it must refuse stays whole units away for
/// the heads a CTC aligner uses: an all-zeros frame of `V` classes is off by
/// `ln V` (3.37 at 29 classes, 8.5 at 5,000), and a raw-logit head, whose
/// logits are never normalized, is off by whole units on most frames. Past
/// about 9,000 classes the allowance passes `ln V`, so for a head that wide the
/// degenerate all-zeros frame is no longer caught, and the guard rests on a
/// raw-logit head's typical frames alone.
///
/// Measured on the staged model (f64 accumulation, the real runtime path), both
/// gate clips and both numerically-clean gate placements:
///
/// | placement | clip | worst `|logsumexp|` |
/// |---|---|---|
/// | `CpuOnly` (the default) | `ted_60.wav` (full 960 k window) | **5.2485e-3** |
/// | `CpuOnly` | `jfk.wav` | 4.7453e-3 |
/// | `CpuAndGpu` | `ted_60.wav` | 2.578e-7 |
/// | `CpuAndGpu` | `jfk.wav` | 2.406e-7 |
///
/// Its 29-class allowance, `0.0293`, sits 5.6× above the worst of them and more
/// than two orders of magnitude below the smallest deviation it must reject
/// there: an all-zeros frame's `ln 29 ≈ 3.367` (115×) and a `[-20, -10]` shifted
/// raw-logit frame's `|logsumexp| >= 6.6` (>225×). Nothing the model produces
/// lands between `5.2e-3` and `3.37`.
///
/// It is deliberately *looser* than `tests/model_io.rs`'s `1e-2` logsumexp
/// tolerance. That is a **tripwire** on the one reviewed artifact — tight, to
/// catch drift in a known quantity; this is a **fence** for any artifact a caller
/// loads through the public API — lenient enough not to false-reject an
/// unknown-but-legitimate one, still two orders below any un-normalized tensor.
///
/// # Cost
///
/// One `f64` exp/sum/log pass over `<= 2,999 × 29 = 86,971` cells on the staged
/// model, against a **0.74 s** CoreML predict. Not measurable.
///
/// Pinned by `tests::check_log_prob_normalization_*` (hermetic: a `[-20, -10]`
/// shifted-logit matrix and an all-zeros frame rejected, real log-probs accepted,
/// the allowance's growth with the width) and
/// `tests::emissions_pass_the_normalization_guard_on_real_speech` (the live
/// model, both clips, both numerically-clean placements).
#[must_use]
pub const fn log_prob_sum_tolerance(vocab_size: NonZeroUsize) -> f64 {
  2.0 * (vocab_size.get() as f64 + 1.0) * FP16_UNIT_ROUNDOFF
}

/// [`log_prob_sum_tolerance`]'s separation property on the staged model's 29
/// classes, asserted at **compile time**: it must sit strictly above the worst
/// legitimate per-frame `|logsumexp|` that model produces (`5.2485e-3`,
/// `CpuOnly` `ted_60`) and at least an order of magnitude below the smallest
/// un-normalized deviation the guard must reject there (an all-zeros frame's
/// `ln 29 ≈ 3.367`). Re-deriving the allowance into either danger zone is then a
/// BUILD failure, not a test failure — below the jitter it false-rejects real
/// audio, and up toward `ln 29` it stops separating a shifted raw-logit head from
/// a real log-prob one, the exact bypass this guard exists to close.
const _: () = {
  let staged = log_prob_sum_tolerance(NonZeroUsize::new(29).unwrap());
  assert!(
    staged > 5.248_517e-3,
    "the allowance would reject the staged model's own measured fp16 logsumexp jitter (worst \
     |logsumexp| 5.2485e-3, CpuOnly ted_60)"
  );
  assert!(
    // 0.34 ≈ ln(29)/10: an order of magnitude below an all-zeros frame's own
    // ln(29) ≈ 3.367 deviation, so the allowance cannot drift up toward the
    // reject region.
    staged < 0.34,
    "the allowance would drift within one order of magnitude of an all-zeros \
     (unnormalized) 29-class frame's logsumexp (ln 29 ≈ 3.367)"
  );
};

/// The smallest `|logsumexp|` of a frame the normalization check must refuse:
/// **`ln 2`**, a frame whose probabilities sum to 2 or more, or to 1/2 or less.
/// Such a frame is no distribution by any reading; a raw-logit frame is
/// typically off by whole units.
pub const UNNORMALIZED_LOGSUMEXP: f64 = core::f64::consts::LN_2;

/// The widest CTC head, **353** classes, whose log-probabilities the
/// normalization check can tell from an unnormalized frame's: the widest `V`
/// for which twice [`log_prob_sum_tolerance`] of `V`, the most fp16 rounding
/// can move a genuine frame, stays within [`UNNORMALIZED_LOGSUMEXP`].
///
/// Past it the allowance a genuine frame needs reaches within a factor of two
/// of a frame off by `ln 2`, and at 5,000 classes it admits a frame of `-4` in
/// every column (a probability mass of about 92). So a contract stating
/// [`OutputKind::LogProbabilities`] for a wider head is refused at load
/// ([`AlignerError::UnprovableNormalization`]); such a head is stated as
/// [`OutputKind::Logits`], which asry normalizes, and normalizing a log-softmax
/// output again changes nothing.
pub const MAX_LOG_PROB_WIDTH: usize =
  (UNNORMALIZED_LOGSUMEXP / (4.0 * FP16_UNIT_ROUNDOFF)) as usize - 1;

/// [`MAX_LOG_PROB_WIDTH`] is exactly the widest separated head, asserted at
/// **compile time**: twice its allowance is within [`UNNORMALIZED_LOGSUMEXP`],
/// one class more is not.
const _: () = {
  let widest = NonZeroUsize::new(MAX_LOG_PROB_WIDTH).unwrap();
  let wider = NonZeroUsize::new(MAX_LOG_PROB_WIDTH + 1).unwrap();
  assert!(
    2.0 * log_prob_sum_tolerance(widest) <= UNNORMALIZED_LOGSUMEXP,
    "MAX_LOG_PROB_WIDTH must be separated from an unnormalized frame"
  );
  assert!(
    2.0 * log_prob_sum_tolerance(wider) > UNNORMALIZED_LOGSUMEXP,
    "MAX_LOG_PROB_WIDTH must be the widest separated head"
  );
};

/// The load-time check of the contract's output kind against the head's
/// width: a head stated as [`OutputKind::LogProbabilities`] must be one the
/// normalization check can separate ([`MAX_LOG_PROB_WIDTH`]).
///
/// # Errors
/// [`AlignerError::UnprovableNormalization`], naming the width and the widest.
fn check_output_width(output: OutputKind, vocab_size: NonZeroUsize) -> Result<(), AlignerError> {
  match output {
    OutputKind::LogProbabilities if vocab_size.get() > MAX_LOG_PROB_WIDTH => {
      Err(AlignerError::UnprovableNormalization(
        UnprovableNormalization::new(vocab_size, MAX_LOG_PROB_WIDTH),
      ))
    }
    OutputKind::LogProbabilities | OutputKind::Logits => Ok(()),
  }
}

/// The load contract this door states: `waveform` `[1, W]` f32 in, with `W`
/// at least the 400 samples asry pads a short chunk to, `emissions` `[1, T, V]`
/// f32 out, no state, every axis one fixed size.
///
/// Data rather than a sequence of checks, and the ONLY check
/// [`Encoder::load`] makes of the graph itself beyond calling [`Model::load`]. The six free functions this replaced — a presence
/// resolver, a shape-and-dtype check and an `expected …` renderer per feature —
/// were each a call the constructor could forget to make, and deleting any of
/// them failed no runnable test, because `Models/alignkit/` holds exactly one
/// artifact and every gate that loads it is `#[ignore]`d. A [`Checked`] field
/// turns that mutation into a compile error.
///
/// **"Fixed in every dimension" is a check rather than a sentence.** A contract
/// whose every axis is [`Dim::Exactly`] or [`Dim::AnyFixed`] requires both
/// features to be [`crate::ShapeConstraint::Fixed`], so a `RangeDims` export
/// declaring fixed-looking numbers — which [`crate::FeatureInfo::shape`] reports
/// identically — is refused. Measured on the staged `base960h_aligner.mlmodelc`:
/// `waveform` `[1, 960000]` Float32 with spans `1+1, 960000+1`, `emissions`
/// `[1, 2999, 29]` Float32 with spans `1+1, 2999+1, 29+1`, both `Fixed`,
/// `states` empty.
///
/// The window `W` is [`Dim::AtLeast`] the 400 samples asry's `prepare` pads a
/// short chunk to: every chunk that reaches the encoder is at least that long,
/// so a smaller window would refuse every one of them. The frame count `T` and
/// the head width `V` are [`Dim::AnyFixed`]. All three are READ back after the
/// check ([`Declared`]). Every step of
/// this door sizes itself by what it read: the zero-padding to `W`, the copy of
/// `[1, T, V]`, the truncation, the per-frame normalization and the wrap. What
/// the numbers must AGREE with is checked by the door that can see the other
/// side of each pairing, the way [`Dim::AnyFixed`] asks:
///
/// - `T` against `W`: the frames the model's [`AcousticContract`] geometry
///   makes of `W` must be `T` ([`check_frame_count`], at load). One `T` fits
///   several geometries, so the geometry is the caller's statement and this is
///   the check the declaration allows.
/// - `V` against the vocabulary the seam tokenizes with: a pairing only the
///   aligner can see. [`crate::audio::align::aligner::Aligner`] refuses a table
///   of another size at load, and asry re-checks the emissions' width against
///   its tokenizer on every chunk.
fn align_contract() -> LoadContract {
  LoadContract::new(
    vec![FeatureContract::new(
      names::WAVEFORM,
      DataType::F32,
      vec![
        Dim::Exactly(1),
        Dim::AtLeast(ASRY_PREPARE_PAD_SAMPLES as usize),
      ],
    )],
    vec![FeatureContract::new(
      names::EMISSIONS,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::AnyFixed, Dim::AnyFixed],
    )],
    StateContract::None,
  )
}

/// What a checked model declares: its input window, its frames per window and
/// its CTC head width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Declared {
  /// `W`: the samples of `waveform`'s last axis.
  window: NonZeroUsize,
  /// `T`: `emissions`' frame axis.
  frames: NonZeroUsize,
  /// `V`: `emissions`' class axis, one column per vocabulary entry.
  vocab_size: NonZeroUsize,
}

/// Reads [`Declared`] off the checked `description`.
///
/// Read AFTER the check (see [`Dim::AnyFixed`]): [`align_contract`] names
/// `waveform` at rank 2 and `emissions` at rank 3, every read axis one
/// non-zero fixed size, so a description that passed it has each axis and none
/// is zero.
fn declared(description: &ModelDescription) -> Declared {
  let axis = |feature: Option<&crate::FeatureInfo>, axis: usize| {
    feature
      .and_then(|feature| feature.shape().get(axis).copied())
      .and_then(NonZeroUsize::new)
      .expect("the contract names this axis one non-zero fixed size, and the check passed")
  };
  let emissions = description.output(names::EMISSIONS);
  Declared {
    window: axis(description.input(names::WAVEFORM), 1),
    frames: axis(emissions, 1),
    vocab_size: axis(emissions, 2),
  }
}

/// The load-time pairing of the contract's geometry with what the model
/// declares: the geometry must make exactly the declared `frames` of the
/// declared `window`.
///
/// The declaration fixes the two ends and not the front end between them, and
/// several geometries fit one pair: a 400-sample and a 640-sample receptive
/// field both make 2999 frames of a 960,000-sample window at a 320-sample
/// stride. So this cannot tell such geometries apart. What it does is refuse
/// every geometry that does NOT fit, which would truncate every chunk to the
/// wrong number of frames.
///
/// # Errors
/// [`AlignerError::FrameCountMismatch`], carrying the geometry and both counts.
fn check_frame_count(
  geometry: AcousticGeometry,
  window: NonZeroUsize,
  frames: NonZeroUsize,
) -> Result<(), AlignerError> {
  if geometry.frames(window.get()) == frames.get() {
    Ok(())
  } else {
    Err(AlignerError::FrameCountMismatch(FrameCountMismatch::new(
      geometry,
      window.get(),
      frames.get(),
    )))
  }
}

/// Map a [`ContractViolation`] into this module's error vocabulary.
///
/// The two "unsatisfiable" clauses keep their own variants — they are about
/// what the door cannot SUPPLY, not about a feature's declared shape — and the
/// per-feature clauses all land in [`AlignerError::ContractMismatch`], which
/// already carries a feature name and a rendered expected/actual pair. An
/// output the model declares OPTIONAL is one of those: it is a fact about the
/// named feature's declaration, so "expected a required output, got optional"
/// is the shape that pair was made for.
///
/// `ContractViolation::rendered` performs that reduction, so a clause added to
/// the checker later lands in the `Feature` arm rather than breaking this
/// function and its five siblings at once.
fn contract_violation(violation: ContractViolation) -> AlignerError {
  match violation.rendered() {
    Rendered::UnsatisfiableInput(name) => AlignerError::UnsatisfiableInput(name),
    Rendered::UnsatisfiableState(name) => AlignerError::UnsatisfiableState(name),
    Rendered::Feature(feature) => AlignerError::ContractMismatch(ContractMismatch::new(
      feature.feature(),
      feature.clone().expected(),
      feature.actual(),
    )),
  }
}

/// Copies a prediction's `emissions` tensor out as `frames × vocab_size`
/// row-major cells — only once the tensor is shown to BE
/// `[1, frames, vocab_size]`, the shape the load contract declared and every
/// later read of the buffer assumes.
///
/// The load contract established what the graph DECLARES; this is the tensor a
/// prediction RETURNED. `MultiArray::copy_into` validates only the element
/// count, and a tensor with that count and other axes (`[1, V, frames]`, or
/// `[frames, V, 1]`) would be copied without complaint and then read as frames
/// of `V` classes that are nothing of the kind. So the shape check and the copy
/// are one function, the only one [`Encoder::emissions_raw`] reads the tensor
/// through: there is no copy without the check.
///
/// # Errors
/// [`AlignError::OutputShape`], carrying both shapes, before any cell is
/// copied; [`AlignError::Tensor`] if the copy itself fails.
fn read_emissions(
  tensor: &MultiArray,
  frames: usize,
  vocab_size: NonZeroUsize,
) -> Result<Vec<f32>, AlignError> {
  let expected = [1, frames, vocab_size.get()];
  if tensor.shape() != expected {
    return Err(AlignError::OutputShape(OutputShape::new(
      tensor.shape().to_vec(),
      expected.to_vec(),
    )));
  }
  let mut data = vec![0.0f32; frames * vocab_size.get()];
  tensor.copy_into::<f32>(&mut data)?;
  Ok(data)
}

/// Rejects an emission matrix holding a cell in `band`, the values the model's
/// contract says it emits IN PLACE OF a log-probability (a saturated fp16
/// `log(0)`, for the staged model). With no band — every contract but the
/// staged artifact's — there is nothing to refuse: no finite value is outside
/// the log-probability domain for every model. Hermetic (no loaded model), and
/// a PREDICT-time guard: the load contract established what the graph
/// declares, which says nothing about the numbers a prediction comes back with.
///
/// `compute` is carried into the error so the failure NAMES the placement that
/// produced it — the diagnosis, not just the symptom.
///
/// Deliberately only the band: the upper bound (`<= 0`) and finiteness are
/// [`Emissions::from_log_probs`]'s scan, which [`Encoder::emissions`] runs
/// immediately after this guard — in [`ValueDomainChecked::into_emissions`], on
/// the very tensor this guard just cleared. A `NaN` therefore passes *here*
/// (`NaN <= x` is false) and is caught *there*; neither scan is redundant with
/// the other.
fn check_sentinel_band(
  data: &[f32],
  band: Option<SentinelBand>,
  compute: ComputeUnits,
) -> Result<(), AlignError> {
  let Some(band) = band else {
    return Ok(());
  };
  let mut min = f32::INFINITY;
  let mut cells = 0usize;
  for &value in data {
    if value < min {
      min = value;
    }
    if band.holds(value) {
      cells += 1;
    }
  }
  if cells > 0 {
    return Err(AlignError::CorruptEmissions(CorruptEmissions::new(
      compute,
      band,
      min,
      cells,
      data.len(),
    )));
  }
  Ok(())
}

/// Rejects an emission matrix whose frames are not **normalized**
/// log-probabilities: a genuine CTC log-prob frame satisfies
/// `logsumexp(frame) = ln Σ exp(log p_j) = ln Σ p_j = ln 1 = 0` by construction,
/// so a frame whose `|logsumexp|` over the vocab axis exceeds the allowance
/// [`log_prob_sum_tolerance`] gives its width carries raw logits — or another
/// un-normalized distribution — not log-probabilities. Reports the single worst
/// frame (largest `|logsumexp|`) in [`AlignError::UnnormalizedEmissions`], with
/// `compute` for the placement, so the failure is self-diagnosing. Hermetic (no
/// loaded model), like [`check_sentinel_band`].
///
/// This is the half of the contract [`Emissions::from_log_probs`]'s
/// `finite ∧ <= 0` scan cannot cover: a raw-logit frame shifted wholly into
/// `[-20, -10]`, or the all-zeros frame, is finite and `<= 0` on every cell yet
/// is no distribution at all — the model-swap the module doc's "The log-prob
/// door" warns of. See the module doc's "The normalization guard".
///
/// `logsumexp` is accumulated in `f64` so the bound reflects the MODEL's
/// deviation rather than this scan's own summation error, matching how the
/// staged model's jitter was measured. A frame with a non-finite maximum
/// (all `-inf`, or a `+inf`/`NaN` cell) is skipped here and left to
/// [`Emissions::from_log_probs`]'s finite scan, which [`Encoder::emissions`] runs
/// immediately after this guard (in [`ValueDomainChecked::into_emissions`]) —
/// exactly the division of labour [`check_sentinel_band`] keeps with `NaN`;
/// recomputing `logsumexp` over it would only manufacture a `NaN` bound. An empty
/// matrix (`real_samples == 0` → zero frames) has no frame to check and is
/// accepted.
///
/// A frame is `vocab_size` cells: the model's CTC head width, read at load
/// ([`Encoder::vocab_size`]), which also sizes the allowance.
fn check_log_prob_normalization(
  data: &[f32],
  vocab_size: NonZeroUsize,
  compute: ComputeUnits,
) -> Result<(), AlignError> {
  debug_assert!(
    data.len().is_multiple_of(vocab_size.get()),
    "emissions buffer is frames × vocab_size by construction"
  );
  let tolerance = log_prob_sum_tolerance(vocab_size);
  let mut worst_row = 0usize;
  let mut worst_abs = 0.0f64;
  let mut worst_lse = 0.0f64;
  for (row, frame) in data.chunks_exact(vocab_size.get()).enumerate() {
    let max = frame.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
      continue;
    }
    let max = f64::from(max);
    let sum: f64 = frame.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    let lse = max + sum.ln();
    if lse.abs() > worst_abs {
      worst_abs = lse.abs();
      worst_lse = lse;
      worst_row = row;
    }
  }
  if worst_abs > tolerance {
    return Err(AlignError::UnnormalizedEmissions(
      UnnormalizedEmissions::new(compute, worst_row, worst_lse, tolerance),
    ));
  }
  Ok(())
}

/// The value-domain guard sequence run over a raw log-prob tensor before it is
/// wrapped: [`check_sentinel_band`] then [`check_log_prob_normalization`], in
/// that order, over the same buffer. It takes the buffer **by value and hands it
/// back on success**, so the minter cannot check one buffer and seal another:
/// [`RawEmissions::check_value_domain`] moves its tensor through here and can seal
/// only what this returns. Clearing `&[]` (or any second buffer) and then wrapping
/// the real tensor no longer type-checks — there is no borrowed slice to swap for
/// an empty one, and after the move no second handle to the original.
///
/// On the production door this is the exact pair [`RawEmissions::check_value_domain`]
/// runs to mint a [`ValueDomainChecked`] — the capability [`Encoder::emissions`]
/// must hold before [`ValueDomainChecked::into_emissions`] will wrap the tensor
/// through [`Emissions::from_log_probs`]. The guard is therefore not merely called
/// *near* the wrap; the wrap is unreachable without it, so swapping in the weaker
/// [`check_sentinel_band`] alone at the call site stops type-checking (a bare
/// `()` mints no token). This mirrors the [`EncoderInput`] capability one screen
/// down: just as that type makes a buffer paired with the wrong real-sample count
/// unrepresentable, this token makes "wrap a tensor the guard never cleared"
/// unrepresentable.
///
/// The hermetic suite drives this sequence through the minter itself
/// (`tests::raw_emissions_check_value_domain_binds_the_guard_and_the_minted_buffer`),
/// not through this helper in isolation: the normalization half has no real-model
/// fixture — no loadable model emits un-normalized raw logits — so binding that
/// predicate to the door means handing the door's own minter a hand-built
/// shifted-raw-logit tensor and asserting both the rejection and, on the accepted
/// frame, that the sealed buffer is the one the guard validated. The band half is
/// additionally exercised at the full public door by the model-gated
/// `tests::emissions_reject_an_ane_corrupted_matrix`.
///
/// Band before normalization, so the staged model's ANE-corrupted matrix is
/// reported as [`AlignError::CorruptEmissions`] rather than merely
/// un-normalized.
///
/// # Errors
/// [`AlignError::CorruptEmissions`] from [`check_sentinel_band`] (a cell in the
/// contract's band); [`AlignError::UnnormalizedEmissions`] from
/// [`check_log_prob_normalization`] (a frame's `|logsumexp|` past
/// [`log_prob_sum_tolerance`] of its width).
fn check_emission_value_domain(
  data: Vec<f32>,
  vocab_size: NonZeroUsize,
  band: Option<SentinelBand>,
  compute: ComputeUnits,
) -> Result<Vec<f32>, AlignError> {
  check_sentinel_band(&data, band, compute)?;
  check_log_prob_normalization(&data, vocab_size, compute)?;
  Ok(data)
}

/// A capability proof that a specific raw log-prob tensor cleared the FULL
/// value-domain guard — [`check_sentinel_band`] THEN
/// [`check_log_prob_normalization`] — and which OWNS that exact tensor.
/// Non-`Copy`, module-private, minted only by [`RawEmissions::check_value_domain`]
/// on success and consumed only by [`Self::into_emissions`].
///
/// This is the [`EncoderInput`] capability pattern (same file, same intent)
/// applied to the value-domain guard rather than to input geometry. `EncoderInput`
/// makes a buffer paired with the wrong real-sample count unrepresentable; this
/// token makes "wrap a tensor the guard never cleared" unrepresentable. The
/// mechanism is the same — carry the checked thing INSIDE the capability so it
/// cannot be swapped after the fact:
///
/// - Calling only [`check_sentinel_band`], or skipping the guard, yields `()`
///   and no token, so [`Self::into_emissions`] — the sole production route from a
///   raw tensor to [`Emissions`] — has nothing to consume and [`Encoder::emissions`]
///   no longer compiles.
/// - Clearing `&[]` (or any other buffer) and then wrapping the real tensor does
///   not type-check: [`check_emission_value_domain`] takes the buffer by value and
///   returns THAT buffer, and [`RawEmissions::check_value_domain`] seals only what
///   it returns — there is no borrowed slice to validate and discard, and after the
///   move no second handle to the original. The token owns the very bytes the guard
///   validated, so the only tensor `into_emissions` can wrap is the one that passed.
struct ValueDomainChecked {
  /// Truncated frame count `T`, carried through from the guarded [`RawEmissions`].
  frames: usize,
  /// The CTC head width `V`, carried through from the guarded [`RawEmissions`].
  vocab_size: NonZeroUsize,
  /// The row-major `frames × vocab_size` log-probabilities that cleared the
  /// guard — the exact tensor [`Self::into_emissions`] wraps, never a second one.
  data: Vec<f32>,
}

impl ValueDomainChecked {
  /// Wraps the guard-cleared tensor through [`Emissions::from_log_probs`] (the
  /// log-prob door), consuming the capability. The only production path from a
  /// value-domain-checked tensor to [`Emissions`].
  ///
  /// # Errors
  /// [`AlignError::Alignment`] if [`Emissions::from_log_probs`]'s own finite ∧
  /// `<= 0` scan rejects the tensor — the domain half the value-domain guard
  /// deliberately leaves to it (see [`check_sentinel_band`]).
  fn into_emissions(self) -> Result<Emissions, AlignError> {
    Ok(Emissions::from_log_probs(
      self.frames,
      self.vocab_size,
      self.data,
    )?)
  }
}

/// A provenance-bound encoder input: the buffer [`Encoder::emissions`] runs the
/// model on, bound at construction to the count of REAL (pre-pad) audio samples
/// that determines the truncated frame count `T`.
///
/// # Why this type exists — the recurring bug class, closed at the type level
///
/// [`Encoder::emissions`] needs two lengths that are NOT the same number: the
/// buffer it feeds the fixed-window CoreML graph (asry's silence-masked,
/// receptive-field-padded `encoder_input`, or a standalone caller's raw
/// samples), and the count of real audio the chunk represents — which drives
/// `truncated_frame_count`, and through it every word's timing. When those
/// arrived as two independent arguments (`encoder_input: &[f32]` and a free
/// `real_samples: usize`) nothing tied them together:
///
/// - A 176,000-sample buffer with `real_samples = 175_360` (two hops short)
///   silently produced 547 frames where 549 belong, moving the tail by two
///   frames with **no error** — asry's own `chunk_extent ± 2·hop` stride check
///   is too loose to catch a two-hop lie.
/// - Naturally passing `encoder_input.len()` as the real count on a padded chunk
///   (200 real samples zero-padded to 400) recorded the padded extent as real.
///   With the corrected conv-geometry truncation that *particular* slip is now
///   benign for a sub-receptive-field chunk — 200 real samples and their
///   400-sample pad both yield the single receptive-field frame — but the
///   binding still has to hold for the general case above, where a real count
///   short of a full-window slice genuinely moves the count.
///
/// This type makes that mismatch **unrepresentable**. `real_samples` is never a
/// free integer supplied alongside the buffer; it is always a slice length,
/// captured at construction from the audio itself:
///
/// - [`from_samples`](Self::from_samples) — the standalone / raw door. The
///   buffer IS the real audio, so both lengths are one slice's `.len()` and
///   cannot disagree.
/// - [`from_prepared`](Self::from_prepared) — the composition door. Reads BOTH
///   the padded buffer and the true pre-pad real length off one
///   [`PreparedChunk`] — the capability token
///   only asry's `prepare` can mint — so the two are drawn from the same
///   authoritative object and cannot be paired wrong. It is the door
///   [`Aligner::align_chunk`](crate::audio::align::aligner::Aligner::align_chunk) takes AND the
///   one an external `prepare` → `Encoder` → `finish` composer takes.
///
/// There is deliberately **no** public `(buffer, count)` constructor: a free
/// `real_samples: usize` supplied alongside a buffer is exactly the forgeable
/// pair this type exists to delete. `from_prepared` is safe to expose precisely
/// because it takes neither a loose integer nor a loose buffer — it reads both
/// off the unforgeable [`PreparedChunk`], whose
/// [`real_samples`](asry::emissions::PreparedChunk::real_samples) is asry's own
/// pre-pad `samples.len()`, not a number the caller gets to choose.
///
/// The window ceiling is not this type's to check: the window is the model's,
/// read at load ([`Encoder::window_samples`]), so [`Encoder::emissions`] refuses
/// a buffer longer than its own window, before any prediction runs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EncoderInput<'a> {
  /// The buffer the model runs on (raw samples, or asry's masked+padded
  /// buffer). Zero-padded up to the full window inside [`Encoder::emissions`].
  encoder_input: &'a [f32],
  /// The count of REAL (pre-pad) audio samples — a slice length captured at
  /// construction, never a caller-supplied integer.
  real_samples: usize,
}

impl<'a> EncoderInput<'a> {
  /// A raw-audio encoder input for **un-prepared** samples: `samples` is both the
  /// buffer the model runs on AND the real audio it represents, so
  /// `real_samples == samples.len()` and a mismatch between them is impossible by
  /// construction.
  ///
  /// This door is for genuinely raw audio only. Do **not** hand it a
  /// [`PreparedChunk`]'s
  /// [`encoder_input`](asry::emissions::PreparedChunk::encoder_input): that buffer
  /// is receptive-field-padded, so its `.len()` is the PADDED count and the
  /// honest length to record is the chunk's own pre-pad `real_samples`. A
  /// prepared chunk must use [`from_prepared`](Self::from_prepared), which reads
  /// that true pre-pad length off the chunk itself. (Under the conv-geometry
  /// truncation the padded and real lengths now agree on the FRAME COUNT for a
  /// sub-receptive-field chunk — 200 real and its 400-sample pad both yield one
  /// frame — but `from_prepared` is still the correct, self-documenting door,
  /// and the only one that stays right for the general case.)
  ///
  /// `samples` shorter than the encoder's window is zero-padded up to it inside
  /// [`Encoder::emissions`]; longer is refused there
  /// ([`AlignError::InputTooLong`]), before any prediction.
  #[cfg(test)]
  pub(crate) fn from_samples(samples: &'a [f32]) -> Self {
    // real == buffer: one slice, so `real_samples` cannot disagree with the
    // buffer length — the raw path's whole safety argument.
    Self::new(samples, samples.len())
  }

  /// The composition door: build straight from asry's [`PreparedChunk`], reading
  /// BOTH the silence-masked, receptive-field-padded
  /// [`encoder_input`](asry::emissions::PreparedChunk::encoder_input) buffer AND
  /// the true pre-pad real length
  /// ([`real_samples`](asry::emissions::PreparedChunk::real_samples)) off the one
  /// object — so the length that drives truncation is asry's own `samples.len()`,
  /// never a count the caller pairs with the buffer by hand.
  ///
  /// This is the door for a caller who drives the supported seam directly —
  /// `EmissionsAligner::prepare` → this [`Encoder`] → `EmissionsAligner::finish` —
  /// and it is what [`Aligner::align_chunk`](crate::audio::align::aligner::Aligner::align_chunk)
  /// uses internally too. Exposing it is safe *because* the [`PreparedChunk`] is
  /// unforgeable (only asry's `prepare` mints one) and carries both lengths
  /// together: there is no way to hand this door a padded buffer with a mismatched
  /// real count. Reaching for [`from_samples`](Self::from_samples) on
  /// `prepared.encoder_input()` instead — treating the padded samples as real —
  /// records the padded length; the corrected conv-geometry truncation makes
  /// that harmless for a sub-receptive-field chunk (padded and real yield the
  /// same single frame), but this door is the one that stays honest without
  /// relying on that coincidence.
  ///
  /// asry's own per-chunk cap is far looser than the staged model's 60 s window
  /// (see the module doc's "60 s clamp" section), so a chunk asry accepted can
  /// still be too long for an encoder; [`Encoder::emissions`] refuses it
  /// ([`AlignError::InputTooLong`]), before any prediction.
  pub(crate) fn from_prepared(prepared: &'a PreparedChunk<'_>) -> Self {
    Self::new(prepared.encoder_input(), prepared.real_samples())
  }

  /// The one constructor both doors funnel through. Debug-asserts the real
  /// length does not exceed the buffer — an internal invariant both doors
  /// satisfy by construction (`from_samples` by equality,
  /// [`from_prepared`](Self::from_prepared) because asry only ever pads the real
  /// audio UP).
  fn new(encoder_input: &'a [f32], real_samples: usize) -> Self {
    debug_assert!(
      real_samples <= encoder_input.len(),
      "real_samples ({real_samples}) exceeds the encoder buffer ({} samples): the real audio \
       cannot be longer than the (already silence-masked, padded) buffer it was built into",
      encoder_input.len(),
    );
    Self {
      encoder_input,
      real_samples,
    }
  }
}

/// Refuses an encoder buffer longer than the model's `window`, before any
/// prediction: the graph takes exactly `window` samples, and truncating the
/// buffer to fit would silently drop audio the caller meant to align.
///
/// # Errors
/// [`AlignError::InputTooLong`], carrying both lengths.
fn check_window(samples: usize, window: NonZeroUsize) -> Result<(), AlignError> {
  if samples > window.get() {
    Err(AlignError::InputTooLong(InputTooLong::new(
      samples,
      window.get(),
    )))
  } else {
    Ok(())
  }
}

/// CoreML wrapper over a fixed-window CTC acoustic encoder: one
/// [`Self::window_samples`]-sample window in, per-frame CTC log-probabilities
/// out — see the module doc for the padding/truncation contract that bridges
/// the fixed window to asry's variable-length encoder shape.
#[derive(Debug)]
pub(crate) struct Encoder {
  /// A [`Checked`], never a bare [`Model`]: [`align_contract`] is the only
  /// contract this door states and [`Checked::new`] is the only way one is
  /// built, so removing the check from [`Self::from_file_with_contract`] does
  /// not compile.
  model: Checked,
  /// What the checked model declares: its window, its frames per window and
  /// its head width.
  declared: Declared,
  /// The model's contract: the geometry that truncates its emissions, checked
  /// against [`Self::declared`] at load, and the sentinel band, if any, the
  /// value-domain guard refuses.
  contract: AcousticContract,
  /// The placement this encoder was loaded on, kept so
  /// [`AlignError::CorruptEmissions`] can name it. The corruption a sentinel
  /// band catches is a property of the model artifact, but the placement is
  /// what a caller can actually change, so it is the one fact the error most
  /// needs to carry.
  compute: ComputeUnits,
}

impl Encoder {
  /// Loads the model at `path`, whose [`AcousticContract`] is `contract`, on
  /// the `compute` placement.
  ///
  /// The model is checked against this door's load contract (`align_contract`)
  /// and held as a crate-internal `Checked` wrapper whose only constructor runs
  /// that check:
  ///
  /// ```text
  /// input   waveform   f32  [1, W]       1 exactly, W any one non-zero fixed size
  /// output  emissions  f32  [1, T, V]    1 exactly, T and V any one non-zero fixed size
  /// state   none
  /// ```
  ///
  /// **This is where the module's "fixed in every dimension" becomes a check.**
  /// `crate::FeatureInfo::shape` reports the same numbers for a `RangeDims`
  /// graph converted at them, so a variable-window export would otherwise load
  /// into a door whose whole padding/truncation bridge assumes one window. A
  /// contract of `Exactly` and `AnyFixed` axes requires both features to be
  /// [`crate::ShapeConstraint::Fixed`], which is the only thing that separates
  /// the two.
  ///
  /// `W`, `T` and `V` are then read back ([`Self::window_samples`],
  /// [`Self::frames`], [`Self::vocab_size`]), and `T` is checked against the
  /// contract: its geometry must make exactly `T` frames of `W`. A model is
  /// thereby held to the front end its caller states, and a statement the
  /// declaration contradicts is refused here, by name, before any chunk.
  ///
  /// The encoder uses the contract's geometry and band; the blank is the
  /// seam's. A caller composing `prepare` → this encoder → `finish` itself
  /// builds asry's seam with the same contract's blank and stride;
  /// [`crate::audio::align::aligner::Aligner`] does so by construction, and it is
  /// the only caller: the composition is not public (see the module doc).
  ///
  /// The ground truth stays pinned by
  /// `tests/model_io.rs::base960h_aligner_io_matches_spec`, which loads the
  /// staged artifact THROUGH this door.
  ///
  /// # Errors
  /// [`AlignerError::Load`] if CoreML rejects the model;
  /// [`AlignerError::ContractMismatch`] if a named feature's type or geometry
  /// mismatches; [`AlignerError::UnsatisfiableInput`] if it requires an input
  /// this door never sends; [`AlignerError::UnsatisfiableState`] if it declares
  /// a state buffer; [`AlignerError::FrameCountMismatch`] if the contract's
  /// geometry does not make the declared frame count of the declared window;
  /// [`AlignerError::UnprovableNormalization`] if the contract states
  /// log-probabilities for a head wider than [`MAX_LOG_PROB_WIDTH`].
  ///
  /// With the `tracing` feature: an `alignkit.encoder.load` span at `INFO`.
  /// The CoreML load is where the wall-clock hides — 0.68 s cold on the
  /// `CpuOnly` default, and **308 s** the first time a caller sets an ANE
  /// placement (see [`DEFAULT_ENCODER_COMPUTE`]) — so the span carries the
  /// placement, which is the field that explains the number.
  #[cfg_attr(
    feature = "tracing",
    tracing::instrument(
      name = "alignkit.encoder.load",
      level = "info",
      skip_all,
      fields(path = ?path.as_ref(), compute = ?compute),
    )
  )]
  pub(crate) fn load(
    path: impl AsRef<Path>,
    contract: &AcousticContract,
    compute: ComputeUnits,
  ) -> Result<Self, AlignerError> {
    let model = Model::load(path, compute)?;
    let model = Checked::new(model, &align_contract()).map_err(contract_violation)?;
    let declared = declared(model.description());
    check_frame_count(contract.geometry(), declared.window, declared.frames)?;
    check_output_width(contract.output(), declared.vocab_size)?;

    Ok(Self {
      model,
      declared,
      contract: *contract,
      compute,
    })
  }

  /// The model's input window `W`, in samples: **960,000** (60 s @ 16 kHz) for
  /// `base960h_aligner.mlmodelc` ([`ENCODER_WINDOW_SAMPLES`]).
  ///
  /// READ at load from the declared `waveform` shape, after the contract
  /// established it as one non-zero fixed size. A chunk is at most this long;
  /// [`Self::emissions`] zero-pads a shorter one up to it.
  #[inline(always)]
  pub(crate) const fn window_samples(&self) -> usize {
    self.declared.window.get()
  }

  /// The model's output frame count `T` for one full (unpadded) window:
  /// **2999** for `base960h_aligner.mlmodelc` (pinned by
  /// `tests/model_io.rs::base960h_aligner_io_matches_spec`).
  ///
  /// READ at load from the declared `emissions` shape and checked there against
  /// the contract's geometry, which must make exactly this many frames of
  /// [`Self::window_samples`].
  #[inline(always)]
  pub(crate) const fn frames(&self) -> usize {
    self.declared.frames.get()
  }

  /// The model's CTC head width `V`: how many classes every emission frame
  /// scores, one per entry of the vocabulary the model was trained on — **29**
  /// for `base960h_aligner.mlmodelc`.
  ///
  /// READ at load from the declared `emissions` shape, after the contract
  /// established it as one non-zero fixed size; never pinned, so a model that
  /// spells another alphabet loads through this door as well. The vocabulary a
  /// caller tokenizes with must have exactly this many entries:
  /// [`Aligner`](crate::audio::align::aligner::Aligner) checks that at load
  /// ([`AlignerError::VocabularyMismatch`]), and asry re-checks it on every chunk.
  #[inline(always)]
  pub(crate) const fn vocab_size(&self) -> NonZeroUsize {
    self.declared.vocab_size
  }

  /// The contract this encoder was loaded with.
  #[inline(always)]
  pub(crate) const fn contract(&self) -> &AcousticContract {
    &self.contract
  }

  /// [`Self::emissions`] without the [`Emissions`] value-domain scan or
  /// wrapping: the truncated log-probabilities as a plain [`RawEmissions`]
  /// carrier. See [`Self::emissions`] for the [`EncoderInput`] contract, the
  /// truncation formula, and the errors — this is the same method minus the
  /// final wrap.
  ///
  /// Crate-private, and staying that way until something needs otherwise:
  /// [`Emissions`] deliberately exposes no per-cell reads, and the only
  /// in-crate caller that legitimately wants the values back is the numeric
  /// regression coverage in `tests.rs` (which is precisely how the fp16
  /// `log(0)` sentinel behind [`DEFAULT_ENCODER_COMPUTE`] is pinned).
  ///
  /// # Errors
  /// As [`Self::emissions`], minus the three value-domain rejections that method
  /// adds on top of the raw tensor: [`AlignError::CorruptEmissions`] and
  /// [`AlignError::UnnormalizedEmissions`] (the band and normalization guards this
  /// method deliberately skips — it hands back an ANE-corrupted or shifted-raw-logit
  /// tensor as `Ok`, which is its whole unguarded purpose) and
  /// [`AlignError::Alignment`] (skipping the wrap is exactly skipping the
  /// [`Emissions::from_log_probs`] scan that raises it). What remains —
  /// [`AlignError::InputTooLong`], [`AlignError::Tensor`],
  /// [`AlignError::Prediction`] and [`AlignError::OutputShape`] — arises here
  /// exactly as in [`Self::emissions`], which runs the identical window check,
  /// the identical predict and the identical shape check.
  pub(crate) fn emissions_raw(&self, input: EncoderInput<'_>) -> Result<RawEmissions, AlignError> {
    let EncoderInput {
      encoder_input,
      real_samples,
    } = input;
    let window = self.declared.window;
    // Before any prediction; the pad branch's `buf[..encoder_input.len()]` copy
    // relies on it, and the borrow branch on the exact-window equality.
    check_window(encoder_input.len(), window)?;
    let window = window.get();

    let waveform: Cow<'_, [f32]> = if encoder_input.len() == window {
      Cow::Borrowed(encoder_input)
    } else {
      let mut buf = vec![0.0f32; window];
      buf[..encoder_input.len()].copy_from_slice(encoder_input);
      Cow::Owned(buf)
    };

    let array = MultiArray::from_slice(&[1, window], waveform.as_ref())?;
    let mut outputs = self.model.predict_with(&[(names::WAVEFORM, &array)])?;
    let emissions = outputs
      .take(names::EMISSIONS)
      .ok_or_else(|| crate::PredictionError::MissingOutput(names::EMISSIONS.to_string()))?;

    let vocab_size = self.declared.vocab_size;
    let mut data = read_emissions(&emissions, self.frames(), vocab_size)?;

    let frames = truncated_frame_count(self.contract.geometry(), real_samples, self.frames());
    // `frames <= self.frames()` always (see `truncated_frame_count`'s clamp),
    // so `frames * V <= data.len() == self.frames() * V` and `truncate` below
    // always shrinks to exactly that length (never a no-op past `data.len()`,
    // which would leave `data` longer than `frames * V`).
    data.truncate(frames * vocab_size.get());

    Ok(RawEmissions {
      frames,
      vocab_size,
      data,
    })
  }

  /// Runs the encoder on `input` and wraps the truncated per-frame
  /// CTC log-probabilities into an [`Emissions`] — the sole log-prob currency
  /// [`asry::emissions::EmissionsAligner::finish`] accepts — with `T` the
  /// frames the contract's geometry makes of the real audio (clamped to
  /// [`Self::frames`], see below) and `V = `[`Self::vocab_size`].
  ///
  /// The wrap goes through [`Emissions::from_log_probs`], the log-prob door:
  /// **no softmax or log-softmax is applied**, and the raw tensor is passed
  /// through unclamped. See the module doc's "The log-prob door" section for
  /// why that door — and not [`Emissions::from_logits`] — is the correct one,
  /// which is a subtler argument than it looks.
  ///
  /// That door's scan bounds the emissions from above and rules out non-finite
  /// values; it does not check that each frame is a normalized distribution,
  /// and it cannot tell a finite sentinel from a log-probability. Guards run
  /// first. When the contract carries a [`SentinelBand`] (the staged model's),
  /// a cell in it is [`AlignError::CorruptEmissions`] here rather than a
  /// plausible but silently wrong alignment (the pre-truncation-fix ANE
  /// measurement put `ask` 881.6 ms early — see [`DEFAULT_ENCODER_COMPUTE`]).
  /// `check_log_prob_normalization` then checks each frame's `logsumexp` is
  /// `≈ 0` within [`log_prob_sum_tolerance`] of the head's width: a raw-logit
  /// model swap (shifted wholly `<= 0`, so past the `<= 0` scan) is
  /// [`AlignError::UnnormalizedEmissions`] rather than silently re-normalized
  /// garbage. Unlike the crate-private `emissions_raw`, which hands back the
  /// tensor unchecked, **this is the guarded door** — and the only one
  /// [`crate::audio::align::aligner::Aligner`] uses.
  ///
  /// `input` is an [`EncoderInput`]: the buffer the model runs on, bound to the
  /// count of real (pre-pad) audio samples that drives the truncation. A
  /// standalone caller builds one from raw audio with
  /// [`EncoderInput::from_samples`] (buffer == real audio); a `prepare` → encode
  /// → `finish` composer (including this crate's own
  /// [`Aligner`](crate::audio::align::aligner::Aligner)) builds it from asry's already-masked,
  /// receptive-field-padded [`PreparedChunk`] with
  /// [`EncoderInput::from_prepared`], which reads the padded buffer and the true
  /// pre-pad `real_samples` off the one chunk, so this method never re-implements
  /// the mask. Either way the two lengths are captured together from the audio and
  /// cannot disagree — that
  /// binding is the whole reason [`EncoderInput`] exists rather than a
  /// `(&[f32], usize)` pair (see its doc). A buffer shorter than the window
  /// ([`Self::window_samples`]) is zero-padded up to it before prediction; the
  /// real-sample count feeds the truncation formula alone and is never
  /// re-scanned, so frames computed from the padded tail are truncated away and
  /// the result reflects only the real audio.
  ///
  /// # Truncation formula
  ///
  /// Piecewise in the real (pre-pad) sample count `L`, with the contract
  /// geometry's receptive field `R` and stride `S` (the staged model's are 400
  /// and 320, [`AcousticGeometry::WAV2VEC2`]):
  ///
  /// ```text
  /// L == 0        →  T = 0
  /// 1 <= L < R    →  T = 1   (padded up to the receptive field)
  /// L >= R        →  T = floor((L − R) / S) + 1
  /// ```
  ///
  /// The lower two branches are the conv front end's OWN output-length
  /// arithmetic — for the staged model the frame count asry's variable-length
  /// ONNX encoder produces for the same audio, which is exactly what this crate
  /// must reproduce to align identically (`tests/parity_words.rs`). `R` is the
  /// strided conv stack's receptive field: the first output frame needs a full
  /// `R`-sample window, not one stride, and each further frame needs one more
  /// stride. A chunk shorter than the receptive field is padded up to it (the
  /// encoder's own zero-padding supplies the rest of the window, and asry pads
  /// a sub-400 chunk too) and yields exactly one frame — the middle branch. The
  /// closed form `floor((L.max(R) − R) / S) + 1` folds that middle branch into
  /// the third via the `.max(R)` and is exact for every `L >= 1`; it is **not**
  /// exact at zero, where it would floor UP to one phantom frame, so `L == 0 →
  /// 0` is a separate branch. That zero is alignkit's own empty-audio POLICY —
  /// no real audio, no real frames — not a reproduction of asry's encoder
  /// geometry: asry short-circuits a TRIVIAL chunk (no alignable text) before
  /// the encoder, but empty audio carrying alignable text is non-trivial, so
  /// asry pads it to 400 and its encoder returns ONE frame there, where this
  /// branch deliberately keeps zero.
  ///
  /// The geometry is the contract's because no declaration fixes it: a
  /// 640-sample receptive field makes the same 2999 frames of a 960,000-sample
  /// window at a 320-sample stride as a 400-sample one, but truncates 720 real
  /// samples to ONE frame where the 400-sample field keeps two. Taking the
  /// staged model's 400 for such a model would keep a frame computed from
  /// padding.
  ///
  /// On the staged geometry it is **not** `ceil(L / 320)`. That earlier formula
  /// agrees with the conv geometry only up to one hop — the two are identical on
  /// `0..=HOP_SAMPLES` (both **0** at empty, both **1** across `1..=320`) — and
  /// over-counts by one or two frames at every length ABOVE `HOP_SAMPLES`,
  /// inventing phantom frames out of the receptive-field slack: `ceil(321/320) =
  /// 2` and `ceil(641/320) = 3` where the conv stack yields **1** (neither 321
  /// nor 641 real samples fills a second 400-wide window), `ceil(48_000/320) =
  /// 150` where it yields **149**. Those phantom frames are pure padding-derived
  /// structure — a 641-sample chunk carrying three distinct tokens then returned
  /// a plausible alignment across three frames that do not exist, where the
  /// reference correctly returns `NoAlignmentPath` (one real frame cannot carry
  /// three tokens); asry's `chunk_extent ± 2·hop` stride check (`3×320 = 960`
  /// inside `641 ± 640`) is too loose to catch it. `tests/prepared_composition.rs`
  /// and `tests/align_chunk.rs` pin that end-to-end.
  ///
  /// Clamped to [`Self::frames`] as a defensive invariant only. At `L` equal to
  /// the window the formula evaluates to the declared frame count — the load
  /// checked exactly that ([`AlignerError::FrameCountMismatch`] otherwise); on
  /// the staged model `floor((960_000 − 400) / 320) + 1 = 2_999`, the
  /// `ted_60.wav` case — and it is monotone in `L`, so it cannot exceed that for
  /// any in-window `L`. Unlike the old `ceil` formula, which reached 3,000 and
  /// genuinely NEEDED the clamp, the `.min` never engages on a valid input. It
  /// stays because `emissions_raw`'s `data.truncate(frames * V)` relies on
  /// `frames <= Self::frames`.
  ///
  /// The contract's stride is the ONE stride of an aligner: the one the encoder
  /// truncates by, which — via the frame count `T` it yields — fixes asry's
  /// effective `n_samples / (T - 1)` grid (~20 ms on the staged model;
  /// `tests/parity_words.rs`) where the word boundaries land, and the SAME
  /// stride [`crate::audio::align::aligner::Aligner`] hands its seam. It is a
  /// fact of the model's graph, stated once in its contract, and deliberately
  /// not an option — a seam-only stride would declare a hop the encoder never
  /// truncated by (at `T >= 2` it would not even move the boundaries, which
  /// follow the encoder-driven grid).
  ///
  /// # The shape a prediction returns, checked every time
  ///
  /// `crate::MultiArray::copy_into` validates only the predict-time
  /// `emissions` tensor's *total element count*, and an axes-swapped runtime
  /// output carrying the identical count (`[1, V, frames]` instead of
  /// `[1, frames, V]`) passes that. So the tensor's own shape is required to be
  /// exactly `[1, Self::frames, Self::vocab_size]` before any cell is copied —
  /// the way `dia-coreml::SegmentModel::infer` re-validates its output shape on
  /// every call (`crates/dia-coreml/src/segment/mod.rs`'s
  /// `check_output_shape`). The load contract cannot stand in for it: it
  /// judges what the graph declares, and a door that reads any model's head
  /// cannot take one fixed artifact's word for what a prediction returns.
  ///
  /// # Errors
  /// [`AlignError::InputTooLong`] if the buffer is longer than the window, before
  /// any prediction. [`AlignError::Tensor`] if building the input tensor or
  /// reading the output tensor fails. [`AlignError::Prediction`] on a CoreML
  /// prediction failure, including a prediction whose runtime output set omits
  /// `emissions` entirely. [`AlignError::OutputShape`] if the returned tensor is
  /// not exactly `[1, frames, V]`. [`AlignError::CorruptEmissions`] if any cell
  /// is in the contract's [`SentinelBand`] — for the staged model a saturated
  /// fp16 `log(0)`, which it produces on an ANE placement.
  /// [`AlignError::UnnormalizedEmissions`] if a frame's `logsumexp` exceeds
  /// [`log_prob_sum_tolerance`] of the head's width — a raw-logit model swap the
  /// `<= 0` scan misses. [`AlignError::Alignment`] (an
  /// `asry::emissions::EmissionsError`) if the model output leaves the
  /// log-probability domain the other way: `from_log_probs` runs an `O(T·V)`
  /// finite ∧ `<= 0` scan, so a non-finite or positive value is a real error
  /// path here — not the panic the pre-seam `LogProbsTV::new` let this crate
  /// assume away.
  ///
  /// With the `tracing` feature: an `alignkit.encoder.emissions` span at
  /// `DEBUG`, nested inside `alignkit.align_chunk` when the [`Aligner`] drives
  /// it. This is the CoreML predict — the dominant cost of a chunk (0.74 s on
  /// the `CpuOnly` default) — so it is the span that tells a caller whether a
  /// slow alignment is the model or the trellis.
  ///
  /// [`Aligner`]: crate::audio::align::aligner::Aligner
  #[cfg_attr(
    feature = "tracing",
    tracing::instrument(
      name = "alignkit.encoder.emissions",
      level = "debug",
      skip_all,
      fields(
        encoder_input = input.encoder_input.len(),
        real_samples = input.real_samples,
        compute = ?self.compute,
      ),
    )
  )]
  pub(crate) fn emissions(&self, input: EncoderInput<'_>) -> Result<Emissions, AlignError> {
    let raw = self.emissions_raw(input)?;
    match self.contract.output() {
      // Log-probabilities reach `Emissions` ONLY through the value-domain guard:
      // the guard mints a `ValueDomainChecked` capability that owns the cleared
      // tensor, and only that capability's `into_emissions` wraps it. Swapping the
      // guard for the weaker `check_sentinel_band` here mints no token and stops
      // compiling — the call-site binding `EncoderInput` gives input geometry,
      // given the guard.
      OutputKind::LogProbabilities => raw
        .check_value_domain(self.contract.sentinel_band(), self.compute)?
        .into_emissions(),
      // Logits are normalized by asry's log-softmax: there is no domain to check
      // them against, and nothing to trust.
      OutputKind::Logits => raw.into_logit_emissions(),
    }
  }
}

/// The **raw** truncated per-frame CTC log-probabilities from
/// [`Encoder::emissions_raw`]: `frames × vocab_size` row-major, exactly the
/// tensor [`Encoder::emissions`] hands to [`Emissions::from_log_probs`].
///
/// Crate-private, like the method that produces it: the public currency is
/// [`Emissions`], which intentionally exposes no per-cell reads (its opaque
/// design deletes the row-major aliasing footgun asry documents). This is a
/// plain internal carrier, not an API — it holds no invariant beyond
/// `data.len() == frames * vocab_size`, and in particular it is NOT a
/// validated log-prob tensor (that is [`Emissions`], reached only through the
/// two guarded constructors).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawEmissions {
  /// Truncated frame count `T`: real-audio frames only, padded-tail frames
  /// already dropped.
  pub(crate) frames: usize,
  /// The CTC head width `V` ([`Encoder::vocab_size`]): the cells per frame.
  pub(crate) vocab_size: NonZeroUsize,
  /// The row-major `frames × vocab_size` log-probabilities.
  pub(crate) data: Vec<f32>,
}

impl RawEmissions {
  /// Wraps this tensor as raw logits through [`Emissions::from_logits`], which
  /// normalizes every frame with a log-softmax: the road of a contract stating
  /// [`OutputKind::Logits`].
  ///
  /// # Errors
  /// [`AlignError::Alignment`] if a logit is non-finite.
  fn into_logit_emissions(self) -> Result<Emissions, AlignError> {
    Ok(Emissions::from_logits(
      self.frames,
      self.vocab_size,
      self.data,
    )?)
  }

  /// Moves this tensor through the full value-domain guard
  /// ([`check_emission_value_domain`], which takes the buffer by value and returns
  /// it) and, on success, seals the RETURNED buffer into a [`ValueDomainChecked`] —
  /// the sole route from a raw tensor to [`Emissions`] on the production door
  /// ([`Encoder::emissions`]). Because the buffer is moved into the guard and the
  /// token is built only from what the guard hands back, the bytes the guard
  /// validated are exactly the bytes [`ValueDomainChecked::into_emissions`] later
  /// wraps: the door cannot clear one buffer and wrap another.
  ///
  /// # Errors
  /// As [`check_emission_value_domain`]: [`AlignError::CorruptEmissions`] (a cell
  /// in `band`) or [`AlignError::UnnormalizedEmissions`] (a frame past
  /// [`log_prob_sum_tolerance`] of its width).
  fn check_value_domain(
    self,
    band: Option<SentinelBand>,
    compute: ComputeUnits,
  ) -> Result<ValueDomainChecked, AlignError> {
    let RawEmissions {
      frames,
      vocab_size,
      data,
    } = self;
    let data = check_emission_value_domain(data, vocab_size, band, compute)?;
    Ok(ValueDomainChecked {
      frames,
      vocab_size,
      data,
    })
  }
}

/// See [`Encoder::emissions`]'s "Truncation formula" doc section.
fn truncated_frame_count(
  geometry: AcousticGeometry,
  real_samples: usize,
  available_frames: usize,
) -> usize {
  if real_samples == 0 {
    // No real audio → no real frames: alignkit's empty-audio policy, not asry's
    // encoder geometry. asry short-circuits a TRIVIAL chunk (no alignable text)
    // before the encoder, but empty audio carrying alignable text is non-trivial —
    // asry pads it to 400 and its encoder returns one frame; this branch keeps
    // zero, and `emissions_raw` truncates to an empty tensor. The conv formula
    // below would otherwise floor UP to 1 here.
    return 0;
  }
  // The conv front end's own output-length arithmetic,
  // `floor((L - receptive_field) / stride) + 1`, where `L` is the real audio
  // padded up to at least the receptive field (`real_samples.max(receptive_field)`;
  // the encoder zero-pads the rest of the window). On the staged geometry this
  // is verified bit-identical to the exact nested per-layer conv composition —
  // and to asry's ONNX model's own output shape — for every `real_samples` in
  // `[1, ENCODER_WINDOW_SAMPLES]`, so alignkit truncates to the exact frame
  // count asry's variable-length encoder would produce for the same audio.
  // `.min(available_frames)` is a defensive invariant only: the load checked
  // that the formula yields exactly `available_frames` at the full window, and
  // it is monotone, so the `.min` never engages on an in-window input — but
  // `emissions_raw`'s `data.truncate` relies on `frames <= available_frames`.
  let receptive_field = geometry.receptive_field().get() as usize;
  geometry
    .frames(real_samples.max(receptive_field))
    .min(available_frames)
}

#[cfg(test)]
mod tests;
