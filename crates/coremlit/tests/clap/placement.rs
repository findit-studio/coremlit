//! Compute-placement **characterization** for both encoders — measured, never
//! marketed (spec §4). It pins per-unit embedding agreement (cosine across
//! placements), documents the MEASURED placement reality, and never asserts ANE
//! residency for the audio tower.
//!
//! # Measured placement reality (T1)
//!
//! - **Audio (HTSAT Swin).** `ANECCompile()` **fails**; CoreML falls back to
//!   GPU/CPU (fp16-clean there). Loading with any [`ComputeUnits`] that names the
//!   ANE therefore still runs on GPU/CPU. This test never claims the audio graph
//!   runs on the ANE.
//! - **Text (RoBERTa).** Compiles for the ANE.
//!
//! # What is pinned
//!
//! For each encoder, one deterministic input is embedded under every placement
//! and the pairwise cosine against the [`ComputeUnits::CpuOnly`] reference is
//! held to a two-sided band `[MIN, 1.0]`. Identical fp16 graph, different
//! hardware ⇒ near-1 agreement; the lower bound is the measured worst case
//! (measure-then-pin). A drop below `MIN` means a placement changed the numerics
//! materially — a finding, not a threshold to loosen.

mod common;

use coremlit::{
  ComputeUnits,
  embeddings::clap::{
    AudioEncoder, AudioEncoderOptions, Embedding, TextEncoder, TextEncoderOptions,
  },
};

/// Lower bound on the audio tower's cross-placement cosine over the FULL public
/// matrix — `All`, `CpuAndNeuralEngine`, `CpuAndGpu`, `CpuOnly` — each vs the
/// `CpuOnly` reference. The audio graph runs on GPU/CPU on every unit (ANECCompile
/// fails for HTSAT, so even the ANE-naming selection falls back), so all four
/// agree to fp16 tolerance. MEASURED worst = 0.99998260; pinned at 0.9999 with a
/// small fp16 margin. A drop below is a finding, not a threshold to loosen.
const AUDIO_MIN_COSINE: f32 = 0.9999;
/// Lower bound on the text tower's cross-placement cosine. MEASURED worst =
/// 0.99994725 (the CpuOnly-vs-ANE fp16/fp32 pair — the text graph DOES compile
/// for the ANE); pinned at 0.9999.
const TEXT_MIN_COSINE: f32 = 0.9999;

/// Lower bound on the **int8** audio tower's cross-placement cosine over the same
/// public matrix. Same characterization as fp16 (HTSAT still falls back off the
/// ANE — `ANECCompile()` fails at int8 too, confirmed at run time — so all units
/// agree to compression tolerance). MEASURED worst = 0.99998385 (locally
/// 2026-07-19 on `clapkit-coreml@02a99c6a`; matches issue #30's 0.99998385);
/// pinned at 0.9999. A drop below is a finding, not a threshold to loosen.
const AUDIO_INT8_MIN_COSINE: f32 = 0.9999;
/// Lower bound on the **int8** text tower's cross-placement cosine. MEASURED
/// worst = 0.99993771 (locally 2026-07-19 on `clapkit-coreml@02a99c6a`; matches
/// issue #30's 0.99993682); pinned at 0.9999.
const TEXT_INT8_MIN_COSINE: f32 = 0.9999;

const AUDIO_UNITS: &[ComputeUnits] = &[
  ComputeUnits::All,
  // The public ANE-naming variant MUST be exercised: the doc claims every ANE
  // selection falls back to GPU/CPU (ANECCompile fails for HTSAT), so this is the
  // characterized proof of that fallback, not an omitted case.
  ComputeUnits::CpuAndNeuralEngine,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuOnly,
];
const TEXT_UNITS: &[ComputeUnits] = &[
  ComputeUnits::All,
  ComputeUnits::CpuAndNeuralEngine,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuOnly,
];

fn assert_band(label: &str, unit: ComputeUnits, cos: f32, min: f32) {
  assert!(
    (min..=1.0 + 1e-6).contains(&cos),
    "{label} [{}] cosine vs CpuOnly = {cos:.8} outside [{min}, 1.0] — a placement changed the \
     numerics materially (a finding, not a threshold to loosen)",
    unit.as_str()
  );
}

#[test]
#[ignore = "requires local clapkit models (CLAPKIT_TEST_MODELS)"]
fn audio_placement_agreement_characterized() {
  let samples = common::deterministic_window(coremlit::embeddings::clap::audio::TARGET_SAMPLES);

  let embed = |unit: ComputeUnits| -> Embedding {
    AudioEncoder::from_file_with(
      common::audio_model_path(),
      AudioEncoderOptions::new().with_compute(unit),
    )
    .unwrap_or_else(|e| panic!("load audio [{}]: {e}", unit.as_str()))
    .embed_window(&samples)
    .unwrap_or_else(|e| panic!("embed audio [{}]: {e}", unit.as_str()))
  };

  let reference = embed(ComputeUnits::CpuOnly);
  let mut worst = 1.0f32;
  for &unit in AUDIO_UNITS {
    let cos = embed(unit).cosine(&reference);
    worst = worst.min(cos);
    assert_band("audio", unit, cos, AUDIO_MIN_COSINE);
  }
  // Non-vacuity: the reference is a valid unit embedding.
  assert!((reference.cosine(&reference) - 1.0).abs() <= 1e-5);
  eprintln!("[placement] audio worst cross-unit cosine = {worst:.8}");
}

#[test]
#[ignore = "requires local clapkit models (CLAPKIT_TEST_MODELS)"]
fn text_placement_agreement_characterized() {
  const PROMPT: &str = "a violin playing a slow melody in a concert hall";

  let embed = |unit: ComputeUnits| -> Embedding {
    TextEncoder::from_bundled_tokenizer(
      common::text_model_path(),
      TextEncoderOptions::new().with_compute(unit),
    )
    .unwrap_or_else(|e| panic!("load text [{}]: {e}", unit.as_str()))
    .embed(PROMPT)
    .unwrap_or_else(|e| panic!("embed text [{}]: {e}", unit.as_str()))
  };

  let reference = embed(ComputeUnits::CpuOnly);
  let mut worst = 1.0f32;
  for &unit in TEXT_UNITS {
    let cos = embed(unit).cosine(&reference);
    worst = worst.min(cos);
    assert_band("text", unit, cos, TEXT_MIN_COSINE);
  }
  assert!((reference.cosine(&reference) - 1.0).abs() <= 1e-5);
  eprintln!("[placement] text worst cross-unit cosine = {worst:.8}");
}

// ── int8 tier (opt-in): the SAME placement characterization on the int8 encoders.
//    The audio int8 graph still falls back off the ANE (same as fp16); the text
//    int8 graph compiles for the ANE. Never asserts ANE residency for audio. ──

#[test]
#[ignore = "requires local clapkit int8 models (CLAPKIT_TEST_MODELS)"]
fn audio_int8_placement_agreement_characterized() {
  let samples = common::deterministic_window(coremlit::embeddings::clap::audio::TARGET_SAMPLES);

  let embed = |unit: ComputeUnits| -> Embedding {
    AudioEncoder::from_file_with(
      common::audio_model_int8_path(),
      AudioEncoderOptions::new().with_compute(unit),
    )
    .unwrap_or_else(|e| panic!("load audio int8 [{}]: {e}", unit.as_str()))
    .embed_window(&samples)
    .unwrap_or_else(|e| panic!("embed audio int8 [{}]: {e}", unit.as_str()))
  };

  let reference = embed(ComputeUnits::CpuOnly);
  let mut worst = 1.0f32;
  for &unit in AUDIO_UNITS {
    let cos = embed(unit).cosine(&reference);
    worst = worst.min(cos);
    assert_band("audio int8", unit, cos, AUDIO_INT8_MIN_COSINE);
  }
  assert!((reference.cosine(&reference) - 1.0).abs() <= 1e-5);
  eprintln!("[placement] audio int8 worst cross-unit cosine = {worst:.8}");
}

#[test]
#[ignore = "requires local clapkit int8 models (CLAPKIT_TEST_MODELS)"]
fn text_int8_placement_agreement_characterized() {
  const PROMPT: &str = "a violin playing a slow melody in a concert hall";

  let embed = |unit: ComputeUnits| -> Embedding {
    TextEncoder::from_bundled_tokenizer(
      common::text_model_int8_path(),
      TextEncoderOptions::new().with_compute(unit),
    )
    .unwrap_or_else(|e| panic!("load text int8 [{}]: {e}", unit.as_str()))
    .embed(PROMPT)
    .unwrap_or_else(|e| panic!("embed text int8 [{}]: {e}", unit.as_str()))
  };

  let reference = embed(ComputeUnits::CpuOnly);
  let mut worst = 1.0f32;
  for &unit in TEXT_UNITS {
    let cos = embed(unit).cosine(&reference);
    worst = worst.min(cos);
    assert_band("text int8", unit, cos, TEXT_INT8_MIN_COSINE);
  }
  assert!((reference.cosine(&reference) - 1.0).abs() <= 1e-5);
  eprintln!("[placement] text int8 worst cross-unit cosine = {worst:.8}");
}
