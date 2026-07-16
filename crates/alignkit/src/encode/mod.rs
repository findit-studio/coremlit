//! CoreML wrapper over `base960h_aligner.mlmodelc` (design spec §3
//! Candidate A): the fixed-window wav2vec2 CTC acoustic encoder,
//! `waveform [1, 960_000]` f32 in, `emissions [1, 2999, 29]` f32 out,
//! 20 ms/frame (stride 320 samples @ 16 kHz) — ground truth pinned by
//! `tests/model_io.rs::base960h_aligner_io_matches_spec`.
//!
//! # Fixed-window bridging
//!
//! [`Encoder::emissions`] hides the model's fixed 60 s window behind a
//! variable-length `&[f32]` contract, mirroring asry's own encoder call
//! shape (`(1, T) -> (1, T', V)`, `asry/src/runner/aligner/algorithm/
//! encode.rs`) as closely as a fixed-window CoreML graph allows:
//!
//! - **Longer than [`ENCODER_WINDOW_SAMPLES`]**: rejected with
//!   [`AlignError::InputTooLong`] rather than silently truncated. The
//!   caller (a future `Aligner`, spec §6/§7) is responsible for chunking
//!   audio to at most [`ENCODER_WINDOW_SAMPLES`] before calling — see
//!   "60 s clamp vs asry's `MAX_CHUNK_SIZE`" below for why this crate's
//!   ceiling is far tighter than asry's own per-chunk cap.
//! - **Shorter**: zero-padded up to exactly [`ENCODER_WINDOW_SAMPLES`],
//!   never rejected — unlike `dia-coreml`'s `SegmentModel::infer`, which
//!   rejects rather than pads short input
//!   (`crates/dia-coreml/src/segment/mod.rs`'s "dia contract match"
//!   section). wav2vec2 is not causal, so whether padding perturbs
//!   in-range emissions remains an open empirical question, not an
//!   assumption — it is for the B5 word-timing parity gate to measure, per
//!   design spec §3's Candidate A note.
//! - **`emissions` frames past the real (non-padded) audio**: truncated
//!   away — see [`Encoder::emissions`]'s doc for the exact formula and why
//!   it must be clamped to the model's actual frame count.
//!
//! # 60 s clamp vs asry's `MAX_CHUNK_SIZE`
//!
//! asry's own chunk-size ceiling is `pub const MAX_CHUNK_SIZE: Duration =
//! Duration::from_secs(600);` (`asry/src/core/transcriber.rs:137`) — 10
//! minutes. [`ENCODER_WINDOW_SAMPLES`] is 60 s, ten times tighter. This is
//! a deliberate, DOCUMENTED divergence, not a parity target: asry's 600 s
//! cap bounds per-chunk RAM for its own (non-fixed-window) ONNX wav2vec2
//! path, which allocates proportionally to whatever length it is given.
//! `base960h_aligner.mlmodelc` allocates a *fixed* 960,000-sample
//! input / `[1, 2999, 29]` output tensor pair regardless of how much of it
//! is real audio, so there is no equivalent "let it grow" option on this
//! side — the model's own fixed graph is the ceiling, not a tunable. A
//! caller (the future `Aligner`, spec §6/§7) must chunk audio to at most
//! [`ENCODER_WINDOW_SAMPLES`] before calling [`Encoder::emissions`]; that
//! chunking responsibility is explicitly out of scope here (design spec
//! §7's data flow already assumes per-chunk audio, not a whole-file
//! stream).
//!
//! # The log-prob door: `from_log_probs`, not `from_logits`
//!
//! [`Encoder::emissions`] wraps the raw `emissions` tensor into an
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
//! The real reason is that `from_log_probs` doubles as a **contract check on
//! the model artifact**. Its `O(T·V)` value-domain scan (every element finite
//! ∧ `<= 0`) is an assertion that the tensor really is log-probs. Should a
//! future model revision ship a raw-logit CTC head — entirely plausible,
//! since that is the *standard* wav2vec2 export, and asry's own ONNX model
//! does exactly that (`asry/src/runner/aligner/algorithm/encode.rs` takes the
//! `from_logits` door) — the scan sees positive maxima and fails **loudly**
//! with [`AlignError::Alignment`] (`EmissionsError::Value`). `from_logits`
//! would instead **silently re-normalize** the garbage into a plausible
//! log-prob domain and align on it forever. That is precisely the bug class
//! this seam exists to kill: the scan is the guard, so the door that runs it
//! is the door to take.
//!
//! For the same reason the raw tensor is passed through **unclamped**. The
//! graph's `softmax` output is in `[0, 1]` by construction, so its `log` is
//! `<= 0` *guaranteed by the graph* (measured max is exactly `0.0` on every
//! compute placement). An unbounded `.min(0.0)` would be actively dangerous —
//! it is exactly what would mask a raw-logit model from the scan above. If a
//! clamp is ever needed here, it must be **bounded** to a pinned slack, never
//! open-ended.
//!
//! # The floor: [`LOG_PROB_FLOOR`], the door's other half
//!
//! [`Emissions::from_log_probs`]'s scan bounds the emissions from **above**
//! (`<= 0`) and rules out non-finite values. It does not bound them from
//! **below**, and it cannot: `-45440` is finite and negative, so an
//! ANE-corrupted matrix — every softmax output under the fp16 floor underflowed
//! to `0`, every `log(0)` saturated to that sentinel
//! ([`DEFAULT_ENCODER_COMPUTE`]) — sails straight through it and aligns to
//! plausible, silently wrong timings.
//!
//! That gap was reachable from this crate's own public API
//! ([`EncoderOptions::with_compute`], [`crate::AlignerOptions::with_compute`]),
//! and it was the *measured* defect, not the hypothetical one the paragraph
//! above guards against. So [`Encoder::emissions`] scans the other side too:
//! any cell below [`LOG_PROB_FLOOR`] is [`AlignError::CorruptEmissions`], a
//! typed error that NAMES the compute placement the encoder was loaded with.
//! Loud, and self-diagnosing.

use core::num::NonZeroUsize;
use std::{borrow::Cow, path::Path};

use asry::emissions::{Emissions, PreparedChunk};
use coremlit::{ComputeUnits, DataType, Model, MultiArray};

use crate::error::{AlignError, AlignerError};

/// Fixed sample count of the encoder's input window (60 s @ 16 kHz).
/// Pinned by `tests/model_io.rs::base960h_aligner_io_matches_spec`
/// (`waveform [1, 960_000]`). See the module doc's "Fixed-window bridging"
/// and "60 s clamp vs asry's `MAX_CHUNK_SIZE`" sections.
pub const ENCODER_WINDOW_SAMPLES: usize = 960_000;

/// Frame stride: 20 ms @ 16 kHz, matching wav2vec2-base's convention and
/// asry's own `hop_samples` default (`asry/src/runner/aligner/
/// aligner.rs`'s `Aligner::from_paths` doc: "`hop_samples` defaults to
/// 320"). Pinned by the design spec §3/§7 and the model card's "20
/// ms/frame" claim; unlike [`ENCODER_WINDOW_SAMPLES`], this is not itself
/// one of the model's declared tensor dimensions, so it is not
/// introspectable from `coremlit::ModelDescription` — see
/// [`Encoder::emissions`]'s doc for how it combines with the model's
/// actual (introspected) frame count.
pub const HOP_SAMPLES: usize = 320;

/// wav2vec2-base's feature-extractor **receptive field**: 400 samples (25 ms
/// @ 16 kHz). It is the composition of the CNN front-end's seven
/// kernel/stride layers (kernels `[10, 3, 3, 3, 3, 2, 2]`, strides
/// `[5, 2, 2, 2, 2, 2, 2]`), and it is the SAME 400 asry zero-pads a
/// sub-receptive-field chunk up to before the conv stack
/// (`asry/src/runner/aligner/core.rs`'s `prepare`, its `< 400` arm). Below it
/// the conv stack produces no complete output frame at all, so a chunk shorter
/// than this is padded up to exactly one frame; at or above it the output
/// length is `floor((L - RECEPTIVE_FIELD_SAMPLES) / HOP_SAMPLES) + 1`. See
/// `truncated_frame_count` and [`Encoder::emissions`]'s "Truncation formula".
const RECEPTIVE_FIELD_SAMPLES: usize = 400;

/// [`crate::vocab::VOCAB_SIZE`] as a [`NonZeroUsize`], for the
/// [`Emissions::from_log_probs`] `v` argument. The conversion is
/// infallible: `VOCAB_SIZE` is the nonzero constant `29`.
const VOCAB_SIZE_NZ: NonZeroUsize = match NonZeroUsize::new(crate::vocab::VOCAB_SIZE) {
  Some(v) => v,
  None => unreachable!(),
};

/// Declared feature names on `base960h_aligner.mlmodelc`
/// (pinned by `tests/model_io.rs::base960h_aligner_io_matches_spec`).
mod names {
  pub const WAVEFORM: &str = "waveform";
  pub const EMISSIONS: &str = "emissions";
}

/// Default [`EncoderOptions::compute`].
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
/// jfk words shift in time (`ask` starts 881.6 ms late) and all 22 differ in
/// timing and/or score. There is no trade-off to weigh, because the ANE
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
/// This is the *default*, not a lock: [`EncoderOptions::with_compute`] still
/// accepts any placement. What stops an ANE override from silently corrupting
/// a caller's timings is [`LOG_PROB_FLOOR`] — a value-domain guard in
/// [`Encoder::emissions`], not a ban on the knob.
pub const DEFAULT_ENCODER_COMPUTE: ComputeUnits = ComputeUnits::CpuOnly;

/// Lower bound of the log-probability domain [`Encoder::emissions`] will
/// accept: **`-100.0`**. A cell strictly below it is not a log-probability at
/// all — it is the fp16 `log(0)` saturation sentinel
/// ([`DEFAULT_ENCODER_COMPUTE`] has the mechanism) — and
/// [`Encoder::emissions`] rejects the whole matrix with
/// [`AlignError::CorruptEmissions`] rather than align on it.
///
/// # Why a bound on the VALUE, never on the placement
///
/// The corruption is a property of the *artifact*, not of the ANE: a
/// re-converted `base960h_aligner` with a fused (or fp32) `log_softmax` tail
/// would be correct on the ANE, and a placement-keyed guard ("reject `All`")
/// would forbid it forever while still failing to describe what is actually
/// wrong. A value-domain guard is placement-agnostic in both directions — it
/// fails the corrupt artifact wherever it runs, and passes any artifact whose
/// emissions really are log-probabilities, including on
/// [`ComputeUnits::CpuAndGpu`], a legitimate non-default placement this crate
/// measures clean (see below).
///
/// # Why `-100`
///
/// It separates two populations three orders of magnitude apart. Measured on
/// `jfk.wav` (549 frames × 29 = 15,921 cells), `min(emissions)` per placement:
///
/// | compute | `min(emissions)` | cells below `-100` |
/// |---|---|---|
/// | `CpuOnly` (the default) | **-30.81** | 0 |
/// | `CpuAndGpu` | **-30.02** | 0 |
/// | `All` / `CpuAndNeuralEngine` (ANE) | **-45440** | **2,667 of 15,921 (16.7%)** |
///
/// `-100` sits ~3.2× below the worst legitimate value ever measured on this
/// model and ~454× above the sentinel; nothing this model produces lands in
/// between. It is not a tolerance to be tuned: `exp(-100) ≈ 3.7e-44` is a
/// posterior no 29-class CTC head assigns to anything (it is beneath fp32's
/// smallest *normal*, `1.2e-38`), so a matrix that reaches it is already
/// broken — while a *correct* fp16 tail cannot underflow below
/// `log(2⁻²⁴) ≈ -16.6` in the first place.
///
/// # Cost
///
/// One extra pass of `<= 2,999 × 29 = 86,971` float comparisons against a
/// **0.74 s** CoreML predict, and [`Emissions::from_log_probs`] already walks
/// every element immediately afterwards. It is not measurable.
///
/// Pinned by `tests::emissions_reject_an_ane_corrupted_matrix` (an `All`
/// encoder on real speech must return `Err`) and
/// `tests::emissions_accept_the_cpu_and_gpu_placement` (a non-default but
/// numerically-clean placement must still return `Ok` — the guard keys on the
/// values, not the hardware).
pub const LOG_PROB_FLOOR: f32 = -100.0;

/// [`LOG_PROB_FLOOR`]'s separation property, asserted at **compile time**: it
/// must sit strictly between the worst legitimate log-probability this model
/// produces on any placement (`-30.81`, `CpuOnly`) and the fp16 `log(0)`
/// sentinel (`-45440`). Tuning the constant into either population is then a
/// BUILD failure, not a test failure — which is the right severity, because a
/// floor inside the legitimate population rejects correct audio and a floor
/// below the sentinel silently disarms the guard that exists to catch it.
const _: () = {
  assert!(
    LOG_PROB_FLOOR < -30.81,
    "LOG_PROB_FLOOR would reject this model's own legitimate log-probs (measured min -30.81)"
  );
  assert!(
    LOG_PROB_FLOOR > -45_440.0,
    "LOG_PROB_FLOOR would no longer reject the fp16 log(0) sentinel (measured -45440)"
  );
};

#[cfg(feature = "serde")]
fn default_encoder_compute() -> ComputeUnits {
  DEFAULT_ENCODER_COMPUTE
}

/// Construction options for [`Encoder`] (rust-options-pattern), mirroring
/// `dia-coreml::segment::SegmentModelOptions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncoderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_encoder_compute",
      with = "crate::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for EncoderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl EncoderOptions {
  /// Options matching the crate's default: [`DEFAULT_ENCODER_COMPUTE`]
  /// (`ComputeUnits::CpuOnly` — see that constant for why the ANE placements
  /// are not merely slower but numerically wrong on this model).
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_ENCODER_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the encoder model on. Defaults to
  /// [`DEFAULT_ENCODER_COMPUTE`] (`ComputeUnits::CpuOnly`), which is a
  /// **correctness** requirement of this model artifact, not a performance
  /// preference — an ANE placement corrupts its emissions.
  ///
  /// Setting one is not silent: [`Encoder::emissions`] then fails with
  /// [`AlignError::CorruptEmissions`], which names the placement. The guard is
  /// on the emission VALUES ([`LOG_PROB_FLOOR`]), not on the placement, so a
  /// numerically-clean non-default placement (`CpuAndGpu`) still works.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }
  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }
  /// Sets [`Self::compute`] in place.
  #[inline(always)]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// Human-readable `shape dtype` rendering for
/// [`AlignerError::ContractMismatch`]'s `actual`/`expected` fields.
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

/// Validates a loaded model's `waveform` input against the pinned
/// contract, in isolation — hermetically testable without a loaded model
/// (unlike `dia-coreml::SegmentModel`'s analogous checks, this crate has
/// no second local model fixture with a mismatched contract to
/// model-gate against; `Models/alignkit/` holds exactly one model, so the
/// validation logic itself is what gets tested, not a specific wrong
/// artifact).
fn check_waveform_contract(shape: &[usize], dtype: Option<DataType>) -> Result<(), AlignerError> {
  if shape != [1, ENCODER_WINDOW_SAMPLES] || dtype != Some(DataType::F32) {
    return Err(AlignerError::ContractMismatch {
      feature: names::WAVEFORM,
      expected: format!("[1, {ENCODER_WINDOW_SAMPLES}] float32"),
      actual: describe(shape, dtype),
    });
  }
  Ok(())
}

/// Validates a loaded model's `emissions` output against the pinned
/// contract, in isolation (see [`check_waveform_contract`]'s doc for why
/// this is hermetic rather than model-gated). Returns the frame count
/// (`shape[1]`) on success — read dynamically rather than hardcoded, see
/// [`Encoder::frames`].
///
/// `shape[1] >= 1`: a zero-frame model would "load fine" and then make
/// every [`Encoder::emissions`] call return an empty result with no
/// error — reject the degenerate contract at construction instead
/// (mirrors `dia-coreml::SegmentModel::from_file_with`'s identical
/// guard).
fn check_emissions_contract(
  shape: &[usize],
  dtype: Option<DataType>,
) -> Result<usize, AlignerError> {
  let shape_ok =
    shape.len() == 3 && shape[0] == 1 && shape[1] >= 1 && shape[2] == crate::vocab::VOCAB_SIZE;
  if !shape_ok || dtype != Some(DataType::F32) {
    return Err(AlignerError::ContractMismatch {
      feature: names::EMISSIONS,
      expected: format!("[1, >=1, {}] float32", crate::vocab::VOCAB_SIZE),
      actual: describe(shape, dtype),
    });
  }
  Ok(shape[1])
}

/// Rejects an emission matrix that has left the log-probability domain from
/// BELOW: any cell under [`LOG_PROB_FLOOR`] is an fp16 `log(0)` saturation
/// sentinel, not a log-probability. Hermetic (no loaded model), like
/// [`check_waveform_contract`] and [`check_emissions_contract`].
///
/// `compute` is carried into the error so the failure NAMES the placement that
/// produced it — the diagnosis, not just the symptom.
///
/// Deliberately only the lower bound: the upper bound (`<= 0`) and finiteness
/// are [`Emissions::from_log_probs`]'s scan, which runs on the very next line
/// of [`Encoder::emissions`]. A `NaN` therefore passes *here* (`NaN < x` is
/// false) and is caught *there*; neither scan is redundant with the other.
fn check_log_prob_floor(data: &[f32], compute: ComputeUnits) -> Result<(), AlignError> {
  let mut min = f32::INFINITY;
  let mut cells = 0usize;
  for &value in data {
    if value < min {
      min = value;
    }
    if value < LOG_PROB_FLOOR {
      cells += 1;
    }
  }
  if cells > 0 {
    return Err(AlignError::CorruptEmissions {
      compute,
      min,
      cells,
      total: data.len(),
    });
  }
  Ok(())
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
///   [`Aligner::align_chunk`](crate::aligner::Aligner::align_chunk) takes AND the
///   one an external `prepare` → `Encoder` → `finish` composer takes.
///
/// There is deliberately **no** public `(buffer, count)` constructor: a free
/// `real_samples: usize` supplied alongside a buffer is exactly the forgeable
/// pair this type exists to delete. `from_prepared` is safe to expose precisely
/// because it takes neither a loose integer nor a loose buffer — it reads both
/// off the unforgeable [`PreparedChunk`], whose
/// [`real_samples`](asry::emissions::PreparedChunk::real_samples) is asry's own
/// pre-pad `samples.len()`, not a number the caller gets to choose. The fixed
/// window ceiling `encoder_input.len() <= `[`ENCODER_WINDOW_SAMPLES`] is checked
/// here, at construction — so invalid geometry is rejected BEFORE any prediction
/// runs, and by the time [`Encoder::emissions`] holds an `EncoderInput` there is
/// no wrong length left to pass it.
#[derive(Debug, Clone, Copy)]
pub struct EncoderInput<'a> {
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
  /// `samples` shorter than [`ENCODER_WINDOW_SAMPLES`] is zero-padded up to the
  /// full window inside [`Encoder::emissions`]; longer is rejected here.
  ///
  /// # Errors
  /// [`AlignError::InputTooLong`] if `samples.len() > `[`ENCODER_WINDOW_SAMPLES`]
  /// — rejected at construction, before any prediction.
  pub fn from_samples(samples: &'a [f32]) -> Result<Self, AlignError> {
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
  /// and it is what [`Aligner::align_chunk`](crate::aligner::Aligner::align_chunk)
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
  /// # Errors
  /// [`AlignError::InputTooLong`] if the prepared buffer exceeds
  /// [`ENCODER_WINDOW_SAMPLES`]. asry's own per-chunk cap is far looser than this
  /// crate's fixed 60 s window (see the module doc's "60 s clamp" section), so a
  /// chunk asry accepted can still be too long for this encoder; it is rejected
  /// here, at construction, before any prediction.
  pub fn from_prepared(prepared: &'a PreparedChunk<'_>) -> Result<Self, AlignError> {
    Self::new(prepared.encoder_input(), prepared.real_samples())
  }

  /// The single geometry gate both constructors funnel through, run at
  /// construction so [`Encoder::emissions`] never repeats it. Rejects a buffer
  /// larger than the fixed window; debug-asserts the real length does not exceed
  /// the buffer — an internal invariant both doors satisfy by construction
  /// (`from_samples` by equality, [`from_prepared`](Self::from_prepared) because
  /// asry only ever pads the real audio UP).
  fn new(encoder_input: &'a [f32], real_samples: usize) -> Result<Self, AlignError> {
    if encoder_input.len() > ENCODER_WINDOW_SAMPLES {
      return Err(AlignError::InputTooLong {
        got: encoder_input.len(),
        max: ENCODER_WINDOW_SAMPLES,
      });
    }
    debug_assert!(
      real_samples <= encoder_input.len(),
      "real_samples ({real_samples}) exceeds the encoder buffer ({} samples): the real audio \
       cannot be longer than the (already silence-masked, padded) buffer it was built into",
      encoder_input.len(),
    );
    Ok(Self {
      encoder_input,
      real_samples,
    })
  }
}

/// CoreML wrapper over `base960h_aligner.mlmodelc`: one
/// [`ENCODER_WINDOW_SAMPLES`]-sample fixed window in, per-frame CTC
/// log-probabilities out — see the module doc for the padding/truncation
/// contract that bridges this fixed window to asry's variable-length
/// encoder shape.
#[derive(Debug)]
pub struct Encoder {
  model: Model,
  frames: usize,
  /// The placement this encoder was loaded on, kept so
  /// [`AlignError::CorruptEmissions`] can name it. The corruption
  /// [`LOG_PROB_FLOOR`] catches is a property of the model artifact, but the
  /// placement is what a caller can actually change, so it is the one fact the
  /// error most needs to carry.
  compute: ComputeUnits,
}

impl Encoder {
  /// Loads the model with [`EncoderOptions::new`] ([`DEFAULT_ENCODER_COMPUTE`]).
  ///
  /// # Errors
  /// As [`Self::from_file_with`].
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self, AlignerError> {
    Self::from_file_with(path, EncoderOptions::new())
  }

  /// Loads the model with custom options, introspecting and validating its
  /// I/O contract against the ground truth pinned by
  /// `tests/model_io.rs::base960h_aligner_io_matches_spec`.
  ///
  /// # Errors
  /// [`AlignerError::Load`] if CoreML rejects the model.
  /// [`AlignerError::ContractMismatch`] if the loaded model's `waveform`
  /// input isn't `[1, ENCODER_WINDOW_SAMPLES]` f32, or its `emissions`
  /// output isn't rank 3 with `shape[0] == 1`, `shape[1] >= 1`, and
  /// `shape[2] == crate::vocab::VOCAB_SIZE` f32. The frame count
  /// (`shape[1]`) is read dynamically, not hardcoded — mirrors
  /// `dia-coreml::SegmentModel`'s `num_frames` field
  /// (`crates/dia-coreml/src/segment/mod.rs`) — see [`Self::frames`].
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
      fields(path = ?path.as_ref(), compute = ?options.compute()),
    )
  )]
  pub fn from_file_with(
    path: impl AsRef<Path>,
    options: EncoderOptions,
  ) -> Result<Self, AlignerError> {
    let model = Model::load(path, options.compute())?;
    let description = model.description();

    let waveform =
      description
        .input(names::WAVEFORM)
        .ok_or_else(|| AlignerError::ContractMismatch {
          feature: names::WAVEFORM,
          expected: format!("[1, {ENCODER_WINDOW_SAMPLES}] float32"),
          actual: "missing".to_string(),
        })?;
    check_waveform_contract(waveform.shape(), waveform.data_type())?;

    let emissions =
      description
        .output(names::EMISSIONS)
        .ok_or_else(|| AlignerError::ContractMismatch {
          feature: names::EMISSIONS,
          expected: format!("[1, >=1, {}] float32", crate::vocab::VOCAB_SIZE),
          actual: "missing".to_string(),
        })?;
    let frames = check_emissions_contract(emissions.shape(), emissions.data_type())?;

    Ok(Self {
      model,
      frames,
      compute: options.compute(),
    })
  }

  /// Output frame count for one full (unpadded) window — the introspected
  /// `emissions` shape's middle dimension (2999 for
  /// `base960h_aligner.mlmodelc`, pinned by
  /// `tests/model_io.rs::base960h_aligner_io_matches_spec`; read
  /// dynamically at construction, not hardcoded).
  #[inline(always)]
  pub const fn frames(&self) -> usize {
    self.frames
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
  /// As [`Self::emissions`], minus [`AlignError::Alignment`] — skipping the
  /// wrap is exactly skipping the check that can raise it. [`AlignError::InputTooLong`]
  /// cannot arise here: [`EncoderInput`] already validated the window ceiling at
  /// construction (see that type's doc).
  pub(crate) fn emissions_raw(&self, input: EncoderInput<'_>) -> Result<RawEmissions, AlignError> {
    let EncoderInput {
      encoder_input,
      real_samples,
    } = input;
    // Guaranteed by `EncoderInput::new` at construction — the pad branch's
    // `buf[..encoder_input.len()]` copy relies on it, and the borrow branch on
    // the exact-window equality.
    debug_assert!(encoder_input.len() <= ENCODER_WINDOW_SAMPLES);

    let waveform: Cow<'_, [f32]> = if encoder_input.len() == ENCODER_WINDOW_SAMPLES {
      Cow::Borrowed(encoder_input)
    } else {
      let mut buf = vec![0.0f32; ENCODER_WINDOW_SAMPLES];
      buf[..encoder_input.len()].copy_from_slice(encoder_input);
      Cow::Owned(buf)
    };

    let array = MultiArray::from_slice(&[1, ENCODER_WINDOW_SAMPLES], waveform.as_ref())?;
    let mut outputs = self.model.predict_with(&[(names::WAVEFORM, &array)])?;
    let emissions =
      outputs
        .take(names::EMISSIONS)
        .ok_or_else(|| coremlit::PredictionError::MissingOutput {
          name: names::EMISSIONS.to_string(),
        })?;

    let mut data = vec![0.0f32; self.frames * crate::vocab::VOCAB_SIZE];
    emissions.copy_into::<f32>(&mut data)?;

    let frames = truncated_frame_count(real_samples, self.frames);
    // `frames <= self.frames` always (see `truncated_frame_count`'s clamp),
    // so `frames * VOCAB_SIZE <= data.len() == self.frames * VOCAB_SIZE` and
    // `truncate` below always shrinks to exactly that length (never a no-op
    // past `data.len()`, which would leave `data` longer than
    // `frames * VOCAB_SIZE`).
    data.truncate(frames * crate::vocab::VOCAB_SIZE);

    Ok(RawEmissions { frames, data })
  }

  /// Runs the encoder on `input` and wraps the truncated per-frame
  /// CTC log-probabilities into an [`Emissions`] — the sole log-prob currency
  /// [`asry::emissions::EmissionsAligner::finish`] accepts — with
  /// `T = truncated_frame_count(real_samples)` (clamped to [`Self::frames`],
  /// see below) and `V = `[`crate::vocab::VOCAB_SIZE`].
  ///
  /// The wrap goes through [`Emissions::from_log_probs`], the log-prob door:
  /// **no softmax or log-softmax is applied**, and the raw tensor is passed
  /// through unclamped. See the module doc's "The log-prob door" section for
  /// why that door — and not [`Emissions::from_logits`] — is the correct one,
  /// which is a subtler argument than it looks.
  ///
  /// That door's scan bounds the emissions from above and rules out non-finite
  /// values; it does not bound them from below. [`LOG_PROB_FLOOR`] does, and
  /// this method checks it first: an ANE-corrupted matrix (finite, negative,
  /// and utterly wrong) is [`AlignError::CorruptEmissions`] here rather than a
  /// plausible alignment 881 ms off. Unlike the crate-private `emissions_raw`,
  /// which hands back the tensor unchecked, **this is the guarded door** — and
  /// the only one [`crate::aligner::Aligner`] uses.
  ///
  /// `input` is an [`EncoderInput`]: the buffer the model runs on, bound to the
  /// count of real (pre-pad) audio samples that drives the truncation. A
  /// standalone caller builds one from raw audio with
  /// [`EncoderInput::from_samples`] (buffer == real audio); a `prepare` → encode
  /// → `finish` composer (including this crate's own
  /// [`Aligner`](crate::aligner::Aligner)) builds it from asry's already-masked,
  /// receptive-field-padded [`PreparedChunk`] with
  /// [`EncoderInput::from_prepared`], which reads the padded buffer and the true
  /// pre-pad `real_samples` off the one chunk, so this method never re-implements
  /// the mask. Either way the two lengths are captured together from the audio and
  /// cannot disagree — that
  /// binding is the whole reason [`EncoderInput`] exists rather than a
  /// `(&[f32], usize)` pair (see its doc). A buffer shorter than
  /// [`ENCODER_WINDOW_SAMPLES`] is zero-padded up to the full window before
  /// prediction; the real-sample count feeds the truncation formula alone and is
  /// never re-scanned, so frames computed from the padded tail are truncated away
  /// and the result reflects only the real audio.
  ///
  /// # Truncation formula
  ///
  /// `T = floor((real_samples.max(400) − 400) / HOP_SAMPLES) + 1`, the
  /// wav2vec2 feature extractor's OWN output-length arithmetic — the frame
  /// count asry's variable-length ONNX encoder produces for the same audio,
  /// which is exactly what this crate must reproduce to align identically
  /// (`tests/parity_words.rs`). The `400` is the seven-layer strided conv
  /// stack's receptive field (`RECEPTIVE_FIELD_SAMPLES`): the first output
  /// frame needs a full 400-sample window, not one [`HOP_SAMPLES`] stride, and
  /// each further frame needs one more stride. A chunk shorter than the
  /// receptive field is padded up to it (asry's own `< 400` pad, mirrored by
  /// the `.max(400)`) and yields exactly one frame.
  ///
  /// It is **not** `ceil(real_samples / HOP_SAMPLES)`. That earlier formula
  /// over-counted at every length by inventing a phantom frame out of the
  /// receptive-field slack: `ceil(641/320) = 3` where the conv stack yields
  /// **1** (641 real samples do not fill a second 400-wide window),
  /// `ceil(48_000/320) = 150` where it yields **149**. Those phantom frames are
  /// pure padding-derived structure — a 641-sample chunk carrying three
  /// distinct tokens then returned a plausible alignment across three frames
  /// that do not exist, where the reference correctly returns `NoAlignmentPath`
  /// (one real frame cannot carry three tokens); asry's `chunk_extent ± 2·hop`
  /// stride check (`3×320 = 960` inside `641 ± 640`) is too loose to catch it.
  /// `tests/prepared_composition.rs` and `tests/align_chunk.rs` pin that
  /// end-to-end.
  ///
  /// Clamped to [`Self::frames`] as a defensive invariant only. At
  /// `real_samples == ENCODER_WINDOW_SAMPLES` (960,000 — the `ted_60.wav`
  /// case) the formula already evaluates to `floor((960_000 − 400) / 320) + 1
  /// = 2_999`, `base960h_aligner.mlmodelc`'s actual count
  /// (`tests/model_io.rs::base960h_aligner_io_matches_spec`), and it cannot
  /// exceed that for any in-window `real_samples` — so unlike the old `ceil`
  /// formula, which reached 3,000 and genuinely NEEDED the clamp, the `.min`
  /// never engages on a valid input. It stays because `emissions_raw`'s
  /// `data.truncate(frames * VOCAB_SIZE)` relies on `frames <= Self::frames`.
  ///
  /// [`HOP_SAMPLES`] is the ONE stride in this crate: the same constant times
  /// the words in [`crate::aligner::Aligner`]'s seam. It is fixed by the
  /// model's graph and is deliberately not configurable — a caller-settable
  /// stride that truncated at 320 while timing at 319 would skew every word
  /// silently.
  ///
  /// # Known gap
  ///
  /// `coremlit::MultiArray::copy_into` validates only the predict-time
  /// `emissions` tensor's *total element count* against
  /// `Self::frames * crate::vocab::VOCAB_SIZE` (established once at
  /// construction) — an axes-swapped runtime output carrying the identical
  /// element count (e.g. `[1, VOCAB_SIZE, frames]` instead of `[1, frames,
  /// VOCAB_SIZE]`) is not independently re-validated per call the way
  /// `dia-coreml::SegmentModel::infer` re-validates its own output shape
  /// on every call (`crates/dia-coreml/src/segment/mod.rs`'s
  /// `check_output_shape`). Accepted here rather than ported: unlike
  /// `dia-coreml`'s `SegmentModel` (which validates a whole family of
  /// possible models sharing one contract), this crate's `Encoder`
  /// contract is pinned to one SHA-tracked model revision
  /// (`tests/model_io.rs`'s module doc), so an axis swap surfacing between
  /// two predictions of an already-loaded, already-contract-validated
  /// `Model` would be a CoreML runtime regression, not a data-dependent
  /// outcome this crate's own inputs can trigger.
  ///
  /// # Errors
  /// Not [`AlignError::InputTooLong`]: [`EncoderInput`] validated the window
  /// ceiling at construction, before this method is ever reachable.
  /// [`AlignError::Tensor`] if building the input tensor or reading the
  /// output tensor fails. [`AlignError::Prediction`] on a CoreML prediction
  /// failure, including a prediction whose runtime output set omits
  /// `emissions` entirely. [`AlignError::CorruptEmissions`] if any cell is
  /// below [`LOG_PROB_FLOOR`] — the fp16 `log(0)` sentinel an ANE placement
  /// produces on this model artifact. [`AlignError::Alignment`] (an
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
  /// [`Aligner`]: crate::aligner::Aligner
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
  pub fn emissions(&self, input: EncoderInput<'_>) -> Result<Emissions, AlignError> {
    let RawEmissions { frames, data } = self.emissions_raw(input)?;
    check_log_prob_floor(&data, self.compute)?;
    Ok(Emissions::from_log_probs(frames, VOCAB_SIZE_NZ, data)?)
  }
}

/// The **raw** truncated per-frame CTC log-probabilities from
/// [`Encoder::emissions_raw`]: `frames × VOCAB_SIZE` row-major, exactly the
/// tensor [`Encoder::emissions`] hands to [`Emissions::from_log_probs`].
///
/// Crate-private, like the method that produces it: the public currency is
/// [`Emissions`], which intentionally exposes no per-cell reads (its opaque
/// design deletes the row-major aliasing footgun asry documents). This is a
/// plain internal carrier, not an API — it holds no invariant beyond
/// `data.len() == frames * VOCAB_SIZE`, and in particular it is NOT a
/// validated log-prob tensor (that is [`Emissions`], reached only through the
/// two guarded constructors).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawEmissions {
  /// Truncated frame count `T`: real-audio frames only, padded-tail frames
  /// already dropped.
  pub(crate) frames: usize,
  /// The row-major `frames × VOCAB_SIZE` log-probabilities.
  pub(crate) data: Vec<f32>,
}

/// See [`Encoder::emissions`]'s "Truncation formula" doc section.
fn truncated_frame_count(real_samples: usize, available_frames: usize) -> usize {
  if real_samples == 0 {
    // No real audio → no real frames. asry short-circuits a trivial chunk
    // before the encoder ever runs, and `emissions_raw` truncates to an empty
    // tensor; the conv formula below would otherwise floor UP to 1 here.
    return 0;
  }
  // The wav2vec2 feature extractor's own output-length arithmetic:
  // `floor((L - RECEPTIVE_FIELD_SAMPLES) / HOP_SAMPLES) + 1`, where `L` is the
  // real audio padded up to at least the receptive field (asry pads a sub-400
  // chunk to exactly 400 before the conv stack — mirrored here by flooring the
  // numerator at 0 with `saturating_sub`, i.e. `real_samples.max(400)`). This
  // is verified bit-identical to the exact nested per-layer conv composition —
  // and to asry's ONNX model's own output shape — for every `real_samples` in
  // `[1, ENCODER_WINDOW_SAMPLES]`, so alignkit truncates to the exact frame
  // count asry's variable-length encoder would produce for the same audio.
  // `.min(available_frames)` is a defensive invariant only: the formula already
  // yields exactly `available_frames` at the full window and never exceeds it
  // for any in-window input, so unlike the old `ceil` (which reached 3,000 and
  // NEEDED the clamp) the `.min` never engages on a valid input — but
  // `emissions_raw`'s `data.truncate` relies on `frames <= available_frames`.
  (real_samples.saturating_sub(RECEPTIVE_FIELD_SAMPLES) / HOP_SAMPLES + 1).min(available_frames)
}

#[cfg(test)]
mod tests;
