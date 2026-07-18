//! Ground-truth introspection of the FluidInference unified Silero VAD
//! artifact (design spec §4/§5). Every claim below comes from loading the real
//! `.mlmodelc` via `coremlit::Model::load` + `.description()`, or from actually
//! running it — the artifact's own `metadata.json` is a HYPOTHESIS re-verified
//! here, not trusted blind, and it wins over the plan wherever they differ.
//!
//! # Artifact (`Models/vadkit/`, gitignored, fetched dev-time)
//!
//! Source: <https://huggingface.co/FluidInference/silero-vad-coreml>, revision
//! (commit SHA) `b419383c55c110e2c9271fa6ee0ea83d03c70d96` — pinned at download
//! time (`hf api`/the HF API `sha` field). Artifact
//! `silero-vad-unified-256ms-v6.2.1.mlmodelc`.
//!
//! | File | Role |
//! |---|---|
//! | `silero-vad-unified-256ms-v6.2.1.mlmodelc/metadata.json` | I/O contract (downloaded) |
//! | `silero-vad-unified-256ms-v6.2.1.mlmodelc/model.mil` | model graph (downloaded) |
//! | `silero-vad-unified-256ms-v6.2.1.mlmodelc/weights/weight.bin` | weights (downloaded) |
//! | `silero-vad-unified-256ms-v6.2.1.mlmodelc/coremldata.bin` | compiled model data (downloaded) |
//! | `silero-vad-unified-256ms-v6.2.1.mlmodelc/analytics/coremldata.bin` | analytics blob (downloaded) |
//!
//! Unlike alignkit, the targeted `.mlmodelc` is itself a DOWNLOADED artifact
//! (v6.2.1 ships pre-compiled, with no `.mlpackage`), so every one of its files
//! is byte-pinned below by SHA-256 — there is no local `coremlcompiler` output
//! whose bytes could legitimately drift.
//!
//! # License
//!
//! HuggingFace `cardData.license` = `mit`. MIT end to end: upstream Silero VAD
//! is MIT, and FluidInference's CoreML conversion is MIT. MIT requires
//! preserving the notice, not a specific attribution string; this record (repo
//! id, revision, license) is that preservation, and the crate README/NOTICE
//! (T6) carries the human-readable attribution.
//!
//! # Per-file SHA-256 (downloaded artifacts)
//!
//! | File | SHA-256 |
//! |---|---|
//! | `metadata.json` | `2740be542c611e1ba358e1849b4e265c65cdf0b17192767e1e5de86a31ac94d6` |
//! | `model.mil` | `c6a9d1bf22d413265da0a07a1d14151c3ea2fad296b3aa5859275b33ef1c3270` |
//! | `weights/weight.bin` | `53ecc8b5081146140ab654c89109cf001f2183abddd7a2411c5081feeffff063` |
//! | `coremldata.bin` | `7db35a4fd995222a7fb0129713473b15d1462572ab4a2e5e4d56bcaad9e40f41` |
//! | `analytics/coremldata.bin` | `8067594eb3126ab8318af507f0c00cabfed40d5fedb8a0ee5075dd02e903d909` |
//!
//! # DECISION
//!
//! - **Target: `silero-vad-unified-256ms-v6.2.1.mlmodelc`** — the exact version
//!   FluidAudio pins (spec §5). The repo also ships `-v6.0.0`, an un-suffixed
//!   `-256ms` sibling, `silero_vad*` and a 4-bit variant; only v6.2.1 is
//!   targeted, downloaded, and pinned.
//!
//! # Spec-vs-reality
//!
//! 1. The plan expected "4160 in, one probability out" (spec §4, from
//!    `VadManager.swift:21-26`). Introspection CONFIRMS `audio_input [1, 4160]`
//!    f32 and a probability output, and reveals the artifact ALSO declares the
//!    explicit LSTM state I/O `VadManager` drives:
//!    `hidden_state`/`cell_state [1, 128]` f32 in → `new_hidden_state`/
//!    `new_cell_state [1, 128]` f32 out. `stateSchema` is EMPTY — this is NOT a
//!    CoreML `MLState` model; the recurrent state is ordinary feature I/O.
//! 2. `vad_output` is a rank-3 `[1, 1, 1]` f32 tensor, not a bare scalar `[1]`.
//! 3. v6.2.1 ships ONLY as a compiled `.mlmodelc` — there is no `.mlpackage`
//!    for this version (the plan brief said "mlpackage + mlmodelc"), so no
//!    `coremlcompiler` step is needed or possible.
//! 4. Spec §5 / plan T2 called for the revision to be "revision-pinned in
//!    `MODELS_LOCK`". Reality: `MODELS_LOCK` is held by a whisperkit hermetic
//!    gate (`crates/coremlit/tests/whisper/models_lock.rs`) to EXACTLY the two
//!    whisper tables CI actually downloads, and the established convention for
//!    an adopted, CI-untested model (alignkit, speakerkit) is to pin its
//!    revision + per-file SHA-256 in the crate's own `model_io.rs` — which is
//!    where this record lives. Following that gated convention, not the plan's
//!    letter, keeps the workspace green and consistent with the sibling kits.

mod common;

use coremlit::{
  ComputeUnits, DataType, Model,
  audio::vad::{CHUNK_SAMPLES, MODEL_INPUT_SAMPLES, STATE_SIZE, VadModel, VadModelOptions},
};

/// The model-layer contract, EXACT in both directions (design spec §4). Every
/// input AND output feature's name, shape and dtype is pinned against the live
/// model's introspected `description()` — the ground truth `VadModel::load_with`
/// validates at construction (mutating any pin here, or the matching check in
/// `crate::model::check_feature`, turns one side red).
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn silero_vad_unified_io_matches_metadata() {
  let model = Model::load(common::model_path(), ComputeUnits::CpuOnly).unwrap();
  let description = model.description();

  // Inputs.
  let audio = description.input("audio_input").expect("audio_input");
  assert_eq!(audio.shape(), &[1, MODEL_INPUT_SAMPLES]); // [1, 4160]
  assert_eq!(audio.data_type(), Some(DataType::F32));

  let hidden = description.input("hidden_state").expect("hidden_state");
  assert_eq!(hidden.shape(), &[1, STATE_SIZE]); // [1, 128]
  assert_eq!(hidden.data_type(), Some(DataType::F32));

  let cell = description.input("cell_state").expect("cell_state");
  assert_eq!(cell.shape(), &[1, STATE_SIZE]);
  assert_eq!(cell.data_type(), Some(DataType::F32));

  // Outputs.
  let vad = description.output("vad_output").expect("vad_output");
  assert_eq!(vad.shape(), &[1, 1, 1]);
  assert_eq!(vad.data_type(), Some(DataType::F32));

  let new_hidden = description
    .output("new_hidden_state")
    .expect("new_hidden_state");
  assert_eq!(new_hidden.shape(), &[1, STATE_SIZE]);
  assert_eq!(new_hidden.data_type(), Some(DataType::F32));

  let new_cell = description
    .output("new_cell_state")
    .expect("new_cell_state");
  assert_eq!(new_cell.shape(), &[1, STATE_SIZE]);
  assert_eq!(new_cell.data_type(), Some(DataType::F32));

  // No surprise state schema: the recurrent state is the explicit I/O above,
  // not a CoreML MLState buffer (which `supports_state()` would report).
  assert_eq!(description.inputs().len(), 3, "exactly 3 declared inputs");
  assert_eq!(description.outputs().len(), 3, "exactly 3 declared outputs");

  // The construction-time contract validator accepts the real model.
  VadModel::load_with(
    common::model_path(),
    VadModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("the pinned contract must accept the real artifact");
}

/// Byte-pins every downloaded file of the artifact. A drift, corruption, or
/// silent re-download of DIFFERENT bytes fails here (the alignkit
/// `source_artifacts_match_pinned_sha256` precedent).
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn source_artifacts_match_pinned_sha256() {
  let dir = common::model_path();
  let cases = [
    (
      "metadata.json",
      "2740be542c611e1ba358e1849b4e265c65cdf0b17192767e1e5de86a31ac94d6",
    ),
    (
      "model.mil",
      "c6a9d1bf22d413265da0a07a1d14151c3ea2fad296b3aa5859275b33ef1c3270",
    ),
    (
      "weights/weight.bin",
      "53ecc8b5081146140ab654c89109cf001f2183abddd7a2411c5081feeffff063",
    ),
    (
      "coremldata.bin",
      "7db35a4fd995222a7fb0129713473b15d1462572ab4a2e5e4d56bcaad9e40f41",
    ),
    (
      "analytics/coremldata.bin",
      "8067594eb3126ab8318af507f0c00cabfed40d5fedb8a0ee5075dd02e903d909",
    ),
  ];
  for (relative, expected) in cases {
    let actual = common::sha256_hex(&dir.join(relative));
    assert_eq!(actual, expected, "sha256 drift on artifact {relative}");
  }
}

/// **Compute-placement characterization** (design spec §4 "ANE honesty" / §6).
/// Records what is actually measurable through this runtime, and asserts ONLY
/// that — never "runs on the ANE". `coremlit` exposes no `MLComputePlan`
/// placement introspection (only `ComputeUnits` SELECTION and I/O
/// `description()`), so the honest characterization is cross-placement
/// NUMERICAL agreement: run the same first chunk under every `ComputeUnits`
/// and record the resulting probability.
///
/// Measured (this test's own run, `02_pyannote_sample` chunk 0 from the
/// initial state, on the machine this branch was cut on):
/// - `cpu_only`             : 0.083007812
/// - `cpu_and_gpu`          : 0.083007812
/// - `cpu_and_neural_engine`: 0.083007812
/// - `all`                  : 0.083007812
/// - worst cross-placement |Δ| vs `cpu_only`: 0.000e0 (bit-identical)
///
/// All four placements agreed to the bit here — CoreML happened to schedule
/// this small graph identically. That is a MEASUREMENT, not a guarantee: the
/// ANE/GPU paths compute in fp16 (the graph is `Mixed(Float16, Float32)`) and
/// the fp32-capable `cpu_only` reference could diverge by up to one fp16 step
/// (~1e-3 near these values) on other hardware/OS, which is what the pinned
/// bound below allows for. Every placement must still return a finite
/// probability in `[0, 1]` (the noisy-OR output's range). `cpu_only` is the
/// deterministic reference the Swift trace gate (`parity_swift.rs`) and the
/// state gates pin against. This test asserts what it measures — agreement and
/// range — and never that any op runs on the ANE (`coremlit` cannot observe
/// that).
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn compute_placement_is_characterized_not_asserted_ane() {
  let samples = common::load_wav_16k_mono(&common::fixture_wav_path("02_pyannote_sample"));
  let chunk = &samples[..CHUNK_SAMPLES];

  let placements = [
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuAndGpu,
    ComputeUnits::CpuAndNeuralEngine,
    ComputeUnits::All,
  ];

  let mut cpu_only = f32::NAN;
  let mut worst_delta = 0.0f64;
  for units in placements {
    let mut model = VadModel::load_with(
      common::model_path(),
      VadModelOptions::new().with_compute(units),
    )
    .unwrap_or_else(|e| panic!("load under {units}: {e}"));
    let p = model
      .predict_chunk(chunk)
      .unwrap_or_else(|e| panic!("predict under {units}: {e}"));
    assert!(
      p.is_finite() && (0.0..=1.0).contains(&p),
      "{units}: probability {p} outside the noisy-OR output range [0, 1]"
    );
    println!("[placement] {units}: {p:.9}");
    if units == ComputeUnits::CpuOnly {
      cpu_only = p;
    } else {
      worst_delta = worst_delta.max((f64::from(p) - f64::from(cpu_only)).abs());
    }
  }
  println!("[placement] worst |Δ| vs cpu_only: {worst_delta:.3e}");

  // Assert only what is measured: every placement agrees with the fp32 cpu
  // reference within the fp16 headroom recorded above. Pinned two-sided after
  // measurement (mutation: a real misplacement/corruption blows past it).
  assert!(
    worst_delta <= PLACEMENT_AGREEMENT_TOL,
    "cross-placement |Δ| {worst_delta:.3e} exceeds the characterized fp16 headroom \
     {PLACEMENT_AGREEMENT_TOL:.0e}"
  );
}

/// Worst tolerated cross-placement probability delta vs the `cpu_only`
/// reference — the fp16 headroom the ANE/GPU paths incur. **Measured worst:
/// `0.000e0`** (bit-identical across all four placements,
/// `compute_placement_is_characterized_not_asserted_ane`). Pinned at `1e-2` —
/// ~10x the fp16 output granularity (~1e-3 near these values) so genuine
/// cross-hardware fp16 drift does not flake, yet a full order of magnitude
/// below the O(1e-1) probability swing a real fp16 misplacement/corruption
/// produces (cf. the alignkit/speakerkit fp16 collapses), so it cannot mask one.
const PLACEMENT_AGREEMENT_TOL: f64 = 1e-2;
