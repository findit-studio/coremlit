//! Native CoreML **speaker-identity embedding** — one fixed 6 s window of
//! 16 kHz mono audio in, one raw [`EMBEDDING_DIM`]-dimensional vector out.
//!
//! The mel front end runs in Rust (the private `mel` submodule) and the
//! mel→embedding network runs natively on Apple silicon as one fp16
//! `.mlmodelc`. NO `ort` anywhere.
//!
//! # A backend-neutral door
//!
//! Nothing in this module's public surface names the model behind it. The types
//! ([`Embedder`], [`EmbedderOptions`], [`Error`]), the geometry constants and
//! the feature names are spelled for the *task*, so a second identity backend
//! can land behind the same seam without renaming a caller's code — the same
//! shape `audio::lid` takes. The provenance that IS backend-specific lives in
//! `conversion/redimnet/`, in `MODELS_LOCK`, and in `tests/identity/`.
//!
//! Today that backend is ReDimNet-B5 (`IDRnD/redimnet`, arXiv 2407.18223),
//! converted from the official `b5-vox2-ft_lm.pt` release asset by
//! `conversion/redimnet`.
//!
//! # This door is NOT the diarization embedder
//!
//! `audio::speaker::embed` also produces speaker embeddings, and the two are
//! not interchangeable. That one is a batch-3, mask-taking, 256-d WeSpeaker
//! graph shaped by pyannote's diarization slots, and it is pinned by a DER gate
//! that goes red on any change to it. This one embeds **one** window, takes no
//! mask, and emits 192 dimensions. Changing the diarization lane's embedder is
//! a diarization-lane decision with its own oracle; this door is additive and
//! touches none of it.
//!
//! # The contract
//!
//! ```text
//! input   mel        f32  [1, 72, 401]
//! output  embedding  f32  [1, 192]
//! ```
//!
//! Fixed shape, never `RangeDim` — a flexible input takes the graph off the
//! ANE, and [`Embedder::load`] ENFORCES that rather than restating it: the
//! declared shape is the same either way, so the door states its contract over
//! the model's shape CONSTRAINT and refuses anything that accepts more than one
//! shape. Batch is 1: this lane embeds one window, and the diarization
//! embedder's batch-3 shape is a slot artifact that does not apply here.
//!
//! **`Fixed` here is a measurement, not a prediction.** The published bundle
//! was read directly with a Swift probe against
//! `MLMultiArrayShapeConstraint`: `mel` reports raw type 2 with the one
//! enumerated shape `[1, 72, 401]`, per-axis ranges `1+1, 72+1, 401+1` and
//! **Float32**; `embedding` reports `[1, 192]`; `stateDescriptionsByName` is
//! empty. So every clause the contract below states was checked against the
//! artifact's own metadata before it was written, rather than predicted from
//! the conversion recipe and left for a CI run that has never happened.
//!
//! ## The output is RAW, and the caller normalizes
//!
//! Measured `‖e‖ ≈ 15.8 – 21.9` across the conversion corpus, nowhere near 1.
//! That is not an oversight to be corrected here — it is this crate's rule
//! (`audio::speaker::embed`: *"L2 normalization is a HIGHER-level concern"*),
//! and the checkpoint already complies: its tail is `ASTP → BatchNorm1d(4608) →
//! Linear(4608, 192)`, with no `emb_bn` and no classifier head, so **there is
//! no L2 in the graph to strip**. The conversion recipe asserts that
//! structurally and refuses an asset that ever grows one.
//!
//! So [`Embedder::embed`] returns the raw vector, and scoring normalizes. Under
//! the `speaker` feature that is
//! `audio::speaker::calibrate::Scoring::IdentityCosine`, whose `prepare` takes
//! exactly [`EMBEDDING_DIM`] raw `f32`s and does the L2 once per profile rather
//! than once per trial.
//!
//! # The window is exactly 6 s, and short clips are refused
//!
//! [`WINDOW_SAMPLES`] is 96 000 samples. It is a decision with evidence behind
//! it, not a default: the shipped asset is the `-ft_lm` fine-tune, and the
//! paper's §3.2 says that stage expanded training utterances to 6 seconds
//! (pre-training used 2 s; the published EER figures score full utterances,
//! ~8 s on VoxCeleb1-test). 6 s is the regime these weights were optimized in.
//! In shipped precision it costs 20.4 ms warm on `CpuAndGpu` — ~294× real time.
//!
//! A clip that is not exactly one window is an [`Error::WindowLength`], never
//! padded and never truncated. That is stricter than the crate's other audio
//! doors and the reason is specific to this front end: its last stage subtracts
//! each mel bin's mean over all 401 frames, so appended silence shifts every
//! real frame's value rather than just the padded ones. `WindowLength`'s own
//! docs carry the argument. Enrolment naturally averages several windows
//! anyway, which is the caller-side shape this pushes toward.
//!
//! # Compute placement (measured, never marketed)
//!
//! [`DEFAULT_COMPUTE`] ships as [`crate::ComputeUnits::CpuAndGpu`] — see its
//! docs for the four-arm table and why the usual `All` is wrong here.
//!
//! # Model artifacts
//!
//! No model is bundled (a `.mlmodelc` is a directory artifact). It is staged as
//! a gitignored dev-time download under `Models/redimnet/`; the `MODELS_LOCK`
//! table names the artifact repository and its revision, and
//! `tests/identity/model_io.rs` pins the per-file SHA-256 and the I/O contract.
//!
//! **No CI run has ever staged this artifact, and that is not a formality.**
//! The artifact repository is PRIVATE — `IDRnD/redimnet`'s MIT grant is written
//! over "the Software" and extends to the released weights nowhere in writing,
//! so publishing a conversion of them openly would be redistribution under no
//! grant — and the workflow's `hf download` carries no credentials. Until a
//! Hugging Face read token reaches the `identity` shard, every model-gated test
//! in `tests/identity/` stays `#[ignore]`d in CI.
//!
//! What HAS run, by hand, against the published bytes (whose per-file SHA-256
//! `tests/identity/model_io.rs` pins and checks): the whole of
//! `tests/identity/`, all four compute placements, all green — the contract
//! check, the exact-shape probe, the raw-norm and determinism gates, and the
//! end-to-end comparison of this door's output against the PyTorch fp32
//! reference (`tests/identity/parity_embed.rs`), whose worst cosine on the
//! shipping placement was 0.99998543. Everything else — the load path's own
//! logic, and the mel front end against goldens cut from the conversion
//! recipe's oracle — is exercised hermetically on every `cargo test`, with no
//! model and no network.
//!
//! Read the placement and window tables above as the conversion recipe's
//! measurements, which is what they are.
//!
//! # Performance: construct once, reuse, prewarm
//!
//! Construction pays model load/specialization; [`Embedder::prewarm`] runs one
//! throwaway inference to absorb first-prediction specialization before
//! serving. Fan-out is one [`Embedder`] per worker ([`crate::Model`] is `Send`
//! but deliberately not `Sync`).
//!
//! macOS only (built on [`crate`]).

use std::path::Path;

use crate::{
  ComputeUnits, DataType, Model, MultiArray,
  model::contract::{
    Checked, ContractViolation, Dim, FeatureContract, LoadContract, Rendered, StateContract,
  },
};

pub mod error;

mod mel;

pub use error::{ContractMismatch, Error, OutputShape, WindowLength};

use crate::audio::identity::{error::Result, mel::MelExtractor};

#[cfg(test)]
mod tests;

/// The sample rate this module's contract is defined at: callers decode and
/// resample to **16 kHz mono f32** before calling (sans-I/O — the workspace
/// convention).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// The fixed inference window in samples: 96 000 = **6 s** at
/// [`SAMPLE_RATE_HZ`]. The export is fixed-shape, so this is model geometry
/// rather than a knob, and the module docs carry the evidence for the 6 s.
pub const WINDOW_SAMPLES: usize = 96_000;

/// Mel-frequency bin count — the graph's input height.
pub const N_MELS: usize = 72;

/// Mel time-frame count for the fixed window: `1 + WINDOW_SAMPLES / hop` at the
/// front end's 240-sample hop with `center=True`, i.e. 401 — the graph's input
/// width. Derived from the hop rather than written down a second time.
pub const N_FRAMES: usize = 1 + WINDOW_SAMPLES / mel::HOP;

/// Dimensionality of the raw embedding the graph emits.
///
/// `audio::speaker::embed::EMBEDDING_DIM` is a DIFFERENT number (256) for a
/// different model in a different lane; the module docs say why the two doors
/// are not interchangeable.
pub const EMBEDDING_DIM: usize = 192;

/// Default compute placement: [`ComputeUnits::CpuAndGpu`].
///
/// MEASURED, and deliberately **not** the `All` this crate's other doors
/// default to. The conversion recipe's four-arm sweep
/// (`conversion/redimnet/scripts/sweep_placement.py`, arms in separate
/// processes so a compiled-program cache cannot fake a cold load, warm latency
/// = median of 30 runs, reproduced twice):
///
/// | arm | load | first predict | warm predict | worst cos vs fp32 CPU | `BNNS Graph Shape Deduction` |
/// |---|---|---|---|---|---|
/// | `All` | 199 ms | 240.5 ms | 79.4 ms | 0.999329 | none |
/// | `CpuAndGpu` | 164 ms | 183.1 ms | **20.4 ms** | **0.999901** | none |
/// | `CpuOnly` | 104 ms | 87.0 ms | 80.1 ms | 0.998635 | none |
/// | `CpuAndNeuralEngine` | 156 ms | 74.6 ms | 73.9 ms | 0.999304 | none |
///
/// Every arm loads, predicts, stays finite and clears the ≥ 0.99 floor, and no
/// arm emits a `BNNS Graph Shape Deduction` line — so this is a choice between
/// four working placements, not a defect being routed around.
///
/// `All` is the wrong one of the four. It tracks the **ANE** arm on both
/// timing (79.4 vs 73.9 ms) and numerics (0.999329, next to
/// `CpuAndNeuralEngine`'s 0.999304 and distinctly not `CpuAndGpu`'s 0.999901):
/// CoreML's heuristic sends this graph to the Neural Engine, where it is
/// **3.9× slower** than the GPU. That is the mirror image of `audio::lid`,
/// where `All` is right only because the heuristic declines the ANE — the same
/// lesson with the opposite sign, and as OS-version-dependent there as here.
///
/// A caller who wants the ANE anyway can ask for it; nothing here forbids a
/// placement, it only stops picking the slow one by default.
pub const DEFAULT_COMPUTE: ComputeUnits = ComputeUnits::CpuAndGpu;

/// Declared feature names on the identity `.mlmodelc` (pinned by
/// `tests/identity/model_io.rs`; emitted by `conversion/redimnet`, which this
/// repository owns).
mod names {
  pub const MEL: &str = "mel";
  pub const EMBEDDING: &str = "embedding";
}

#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_COMPUTE
}

/// Construction options for the identity [`Embedder`] (rust-options-pattern): a
/// single `compute` knob with one source of truth shared by
/// `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbedderOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_compute"))]
  compute: ComputeUnits,
}

impl Default for EmbedderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl EmbedderOptions {
  /// Options matching the module default: [`DEFAULT_COMPUTE`].
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

/// Speaker-identity embedder: one [`WINDOW_SAMPLES`]-sample 16 kHz mono window
/// in, one raw `[`[`EMBEDDING_DIM`]`]` vector out.
///
/// The front end is a Rust log-mel port (the private `mel` submodule); the fp16
/// CoreML graph maps that `[1, 72, 401]` mel to a `[1, 192]` **un-normalized**
/// embedding, and L2 normalization is the scoring layer's job.
///
/// `&self` inference (no mutable scratch): the FFT plan and filterbank are
/// built once at load and per-call buffers are local, so fan-out means one
/// [`Embedder`] per worker over a `Send` [`crate::Model`] (`crate::Model` is
/// deliberately `!Sync`).
///
/// ```no_run
/// use coremlit::audio::identity::{Embedder, WINDOW_SAMPLES};
///
/// # let window: Vec<f32> = vec![0.0; WINDOW_SAMPLES];
/// let embedder = Embedder::from_file("Models/redimnet/redimnet_b5.mlmodelc")?;
/// let raw = embedder.embed(&window)?;
/// assert_eq!(raw.len(), coremlit::audio::identity::EMBEDDING_DIM);
/// # Ok::<(), coremlit::audio::identity::Error>(())
/// ```
#[derive(Debug)]
pub struct Embedder {
  /// A [`Checked`], never a bare [`Model`]: the load contract below is the
  /// only way one is built, so removing the check from [`Self::load`] does not
  /// compile.
  model: Checked,
  mel: MelExtractor,
}

impl Embedder {
  /// Loads the identity `.mlmodelc` from `model_path` with custom `options` —
  /// the primary constructor.
  ///
  /// The model is checked against this door's load contract and held as a
  /// crate-internal `Checked` wrapper whose only constructor runs that check.
  /// The contract IS this door's statement of what it needs, so there is
  /// nothing here to restate:
  ///
  /// ```text
  /// input   mel        f32  [1, 72, 401]   every axis Exactly
  /// output  embedding  f32  [1, 192]       every axis Exactly
  /// state   none
  /// ```
  ///
  /// What an all-`Exactly` contract buys beyond the numbers is the reason it
  /// is stated that way. `crate::FeatureInfo::shape` reports the DEFAULT shape
  /// of a flexible input, so a `RangeDims` graph converted at `[1, 72, 401]`
  /// declares this contract's exact numbers — and a flexible input is what
  /// takes the graph off the accelerator, which is the one reason the
  /// conversion recipe pins a fixed shape. An all-`Exactly` contract therefore
  /// requires the whole feature to be `crate::ShapeConstraint::Fixed`, which is
  /// the only thing that separates the two.
  ///
  /// The contract is also COMPLETE over the members of
  /// [`crate::ModelDescription`] that can make a conformant prediction fail,
  /// not just over the features this door sends: a graph carrying `mel` plus
  /// another REQUIRED input clears every per-feature clause and then fails on
  /// every prediction, and a STATE buffer is not an input at all — it lives in
  /// its own dictionary, so a stateful ML Program declaring exactly `mel` and
  /// `embedding` plus a state clears the input set too, and then meets
  /// [`Self::embed`], which predicts through the stateless API CoreML does not
  /// let a stateful model be called with.
  ///
  /// No model is bundled: the `.mlmodelc` is a directory artifact, staged
  /// gitignored under `Models/redimnet/` from the `MODELS_LOCK` table.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model; [`Error::ContractMismatch`]
  /// if a named feature's type or geometry mismatches;
  /// [`Error::UnsatisfiableInput`] if it requires an input this door never
  /// sends; [`Error::UnsatisfiableState`] if it declares a state buffer.
  pub fn load(model_path: impl AsRef<Path>, options: EmbedderOptions) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let model = Checked::new(model, &identity_contract()).map_err(contract_violation)?;
    Ok(Self {
      model,
      mel: MelExtractor::new(),
    })
  }

  /// Loads the identity `.mlmodelc` with [`EmbedderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self> {
    Self::load(model_path, EmbedderOptions::new())
  }

  /// Embeds one window: the **raw, un-normalized** `[`[`EMBEDDING_DIM`]`]`
  /// vector.
  ///
  /// `samples_16k` is 16 kHz mono and must be exactly [`WINDOW_SAMPLES`] long
  /// — neither padded nor truncated; [`WindowLength`] says why this door
  /// cannot offer either.
  ///
  /// # Errors
  /// [`Error::WindowLength`] if `samples_16k` is not exactly one window;
  /// [`Error::NonFiniteInput`] if any sample is NaN/infinite (it would silently
  /// poison the mel); [`Error::Tensor`] / [`Error::Prediction`] on a tensor or
  /// CoreML failure; [`Error::OutputShape`] if the predicted `embedding` shape
  /// diverges from `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] if
  /// the model output has a NaN/infinite component — caught here rather than
  /// left for the caller's L2, where one NaN makes the whole vector NaN.
  pub fn embed(&self, samples_16k: &[f32]) -> Result<[f32; EMBEDDING_DIM]> {
    validate_window_input(samples_16k)?;

    let mut features = vec![0.0f32; N_MELS * N_FRAMES];
    self.mel.extract_into(samples_16k, &mut features)?;

    // Freq-major mel [72, 401] maps directly onto the row-major `mel
    // [1, 72, 401]` contract.
    let input = MultiArray::from_slice(&[1, N_MELS, N_FRAMES], &features)?;
    let mut outputs = self.model.predict_with(&[(names::MEL, &input)])?;
    let embedding = outputs
      .take(names::EMBEDDING)
      .ok_or_else(|| crate::PredictionError::MissingOutput(names::EMBEDDING.to_string()))?;
    if embedding.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape(OutputShape::new(
        embedding.shape().to_vec(),
        vec![1, EMBEDDING_DIM],
      )));
    }

    let mut row = [0.0f32; EMBEDDING_DIM];
    embedding.copy_into::<f32>(&mut row)?;
    check_finite_embedding(&row)?;
    Ok(row)
  }

  /// Runs one throwaway [`Self::embed`] on a fixed synthetic window to fully
  /// specialize the prediction path, so the first user-facing request is warm.
  /// Construction pays the model load / device specialization; what it does NOT
  /// pay is the first prediction's own graph specialization — the measured gap
  /// on the shipping placement is 183.1 ms for the first predict against
  /// 20.4 ms warm, so this moves an order of magnitude off the first real clip.
  ///
  /// The warm-up runs a fixed 440 Hz tone over the whole window, so it reads no
  /// caller audio.
  ///
  /// # Errors
  /// As [`Self::embed`]; a failure here surfaces a broken model at prewarm time
  /// rather than on the first request.
  pub fn prewarm(&self) -> Result<()> {
    let sr = f64::from(SAMPLE_RATE_HZ);
    let window: Vec<f32> = (0..WINDOW_SAMPLES)
      .map(|i| (0.5 * (std::f64::consts::TAU * 440.0 * (i as f64) / sr).sin()) as f32)
      .collect();
    self.embed(&window)?;
    Ok(())
  }
}

/// The load contract this door states: `mel` `[1, 72, 401]` f32 in,
/// `embedding` `[1, 192]` f32 out, no state.
///
/// Data rather than a sequence of checks, and the ONLY thing [`Embedder::load`]
/// does beyond calling [`Model::load`]. The three free functions this replaced
/// — one per feature, one over the input set, one over the state set — were
/// each a call `load` could forget to make, and deleting any of them failed no
/// runnable test. A [`Checked`] field turns that mutation into a compile error;
/// what remains here is the door's own numbers.
///
/// Built rather than `const` because a [`LoadContract`] owns its axes: a door
/// whose geometry comes from a manifest read at load builds its contract at
/// load. This one's is fixed, so it is the same value every call.
fn identity_contract() -> LoadContract {
  LoadContract::new(
    vec![FeatureContract::new(
      names::MEL,
      DataType::F32,
      vec![
        Dim::Exactly(1),
        Dim::Exactly(N_MELS),
        Dim::Exactly(N_FRAMES),
      ],
    )],
    vec![FeatureContract::new(
      names::EMBEDDING,
      DataType::F32,
      vec![Dim::Exactly(1), Dim::Exactly(EMBEDDING_DIM)],
    )],
    StateContract::None,
  )
}

/// Map a [`ContractViolation`] into this module's error vocabulary.
///
/// The two "unsatisfiable" clauses keep their own variants — they are about
/// what the door cannot SUPPLY, not about a named feature's declared shape —
/// and every per-feature clause lands in [`Error::ContractMismatch`], which already
/// carries a feature name and a rendered expected/actual pair.
///
/// `ContractViolation::rendered` performs that reduction, so a clause added to
/// the checker later lands in the `Feature` arm rather than breaking this
/// function and its five siblings at once.
fn contract_violation(violation: ContractViolation) -> Error {
  match violation.rendered() {
    Rendered::UnsatisfiableInput(name) => Error::UnsatisfiableInput(name),
    Rendered::UnsatisfiableState(name) => Error::UnsatisfiableState(name),
    Rendered::Feature(feature) => Error::ContractMismatch(ContractMismatch::new(
      feature.feature(),
      feature.clone().expected(),
      feature.actual(),
    )),
  }
}

/// Reject a window the pipeline must not see: one that is not exactly
/// [`WINDOW_SAMPLES`] long, or that carries a NaN/±∞ sample (which would
/// silently poison the mel, and through the per-bin mean, every frame of it).
/// Free fn so the guards are hermetically testable without a model.
fn validate_window_input(samples: &[f32]) -> Result<()> {
  if samples.len() != WINDOW_SAMPLES {
    return Err(Error::WindowLength(WindowLength::new(
      samples.len(),
      WINDOW_SAMPLES,
    )));
  }
  if let Some(index) = samples.iter().position(|v| !v.is_finite()) {
    return Err(Error::NonFiniteInput(index));
  }
  Ok(())
}

/// Classify a NaN/∞ the CoreML runtime produced as model-output corruption
/// ([`Error::NonFiniteOutput`]) before the raw vector reaches a caller's L2,
/// where a single non-finite component makes every one of the
/// [`EMBEDDING_DIM`] outputs NaN.
fn check_finite_embedding(embedding: &[f32]) -> Result<()> {
  if let Some(index) = embedding.iter().position(|v| !v.is_finite()) {
    return Err(Error::NonFiniteOutput(index));
  }
  Ok(())
}
