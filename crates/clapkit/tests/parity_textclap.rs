//! Model-level parity vs **textclap** — the characterized comparison (spec §4).
//!
//! clapkit's fp16 CoreML encoders are scored, per-window, against textclap
//! running the **Xenova ONNX** graphs it ships. Built ONLY under
//! `--features parity-oracle` (linking the sibling `textclap` crate and its
//! `ort` runtime; see the `[[test]]` `required-features` in `Cargo.toml`), and
//! model-gated (`#[ignore]`): it needs both the clapkit CoreML models
//! (`CLAPKIT_TEST_MODELS`) and the textclap ONNX (`CLAPKIT_TEXTCLAP_ONNX`;
//! `Xenova/clap-htsat-unfused@c28f2883…`).
//!
//! # HONESTY CLAUSE — this is a CHARACTERIZED comparison, not bit parity
//!
//! textclap ships the **quantized** (int8-class) Xenova graphs
//! (`audio_model_quantized.onnx` 33 MB / `text_model_quantized.onnx` 121 MB),
//! while clapkit's conversion is fp16 from the fp32 source. The primary gates
//! below therefore pin the audio/text cosine two-sided at MEASURED values, not at
//! 1.0. A **same-precision control** runs the identical comparison against
//! Xenova's **unquantized fp32** graphs (`audio_model.onnx` / `text_model.onnx`,
//! which Xenova also ships): its cosine is higher, and the gap between the two
//! bands is the **quantization contribution** — measured, not asserted away. The
//! control is `#[ignore]`d, so it never runs in a default `cargo test`; but once
//! FORCED (`--ignored`, the parity suite the README documents) it is fail-closed —
//! an absent fp32 oracle is a hard failure, never a green in-`#[test]` skip.
//!
//! Both crates receive the **identical** `&[f32]` (audio) or `&str` (text); the
//! mel front-end and tokenizer are the same ported/pinned artifacts (T2/T3
//! gates), so any residual gap is the encoder graph (precision + lowering), which
//! is exactly what this measures.

mod common;

use clapkit::{AudioEncoder, TextEncoder};
use textclap::{
  AudioEncoder as TcAudioEncoder, Options as TcOptions, TextEncoder as TcTextEncoder,
};

// ── MEASURED-then-pinned two-sided bands (measured 2026-07-18; worst cosine over
//    the windows/prompts below). Cosine of two unit-norm embeddings == their dot
//    product. CoreML `All`-vs-placement drift is ~1e-4 (T2/T3 placement gate), so
//    each band clears the measured worst by ≥10× that. Tightening OR loosening
//    fires the gate — a shift in either direction is a finding, re-measure.
//
//    The quantized bands' UPPER bound is meaningfully < 1.0: a test bug that
//    loaded clapkit's own graph (or an unquantized ONNX) on both sides would jump
//    the cosine to ~1.0 and blow it. The fp32-control bands sit at ~1.0 by design
//    — the control's point is that de-quantizing the oracle removes almost the
//    entire gap, so essentially all of it is textclap's quantization, not
//    clapkit's fp16 (audio: fp32 worst 0.99999756 vs quant 0.99804741; text: fp32
//    worst 0.99994940 vs quant 0.96725219). ──
const AUDIO_QUANT_LO: f32 = 0.9965; // measured worst 0.99804741
const AUDIO_QUANT_HI: f32 = 0.9990;
const TEXT_QUANT_LO: f32 = 0.9620; // measured worst 0.96725219 (CJK)
const TEXT_QUANT_HI: f32 = 0.9720;
const AUDIO_FP32_LO: f32 = 0.9998; // measured worst 0.99999756
const AUDIO_FP32_HI: f32 = 1.0 + 1e-4;
const TEXT_FP32_LO: f32 = 0.9998; // measured worst 0.99994940
const TEXT_FP32_HI: f32 = 1.0 + 1e-4;

/// Cosine of two unit-norm embeddings (== dot product).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
  a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The audio windows fed to both crates: the whole JFK clip and two halves — all
/// ≤ 480 000 samples, so textclap's `embed` (which rejects > one window) accepts
/// them and both crates `repeatpad` the short clips to the fixed window.
fn audio_windows(samples: &[f32]) -> Vec<(&'static str, &[f32])> {
  let mid = samples.len() / 2;
  vec![
    ("full", samples),
    ("first_half", &samples[..mid]),
    ("second_half", &samples[mid..]),
  ]
}

/// The text prompts fed to both crates (English + a rain/music contrast + CJK).
const PROMPTS: &[&str] = &[
  "a dog barking",
  "the sound of heavy rain",
  "upbeat electronic dance music",
  "a person speaking",
  "一只猫在喵喵叫",
];

fn jfk_samples() -> Vec<f32> {
  common::read_wav_48k_mono(&common::fixture_path("audio/speech_jfk_48k.wav"))
}

fn textclap_onnx(file: &str) -> std::path::PathBuf {
  common::textclap_onnx_dir().join(file)
}

/// Run the audio per-window comparison against `onnx_file`, returning the worst
/// (min) cosine over the windows. Prints each so the measurement is visible.
fn audio_parity_worst(onnx_file: &str, tier: &str) -> f32 {
  let clap = AudioEncoder::from_file(common::audio_model_path()).expect("load clapkit audio");
  let mut tc = TcAudioEncoder::from_file(textclap_onnx(onnx_file), TcOptions::new())
    .expect("load textclap audio ONNX");
  let samples = jfk_samples();
  let mut worst = 1.0f32;
  for (name, w) in audio_windows(&samples) {
    let a = clap.embed_window(w).expect("clapkit audio embed");
    let b = tc.embed(w).expect("textclap audio embed");
    let cos = cosine(a.as_slice(), b.as_slice());
    println!("[audio/{tier}] window {name:<12} cosine = {cos:.8}");
    worst = worst.min(cos);
  }
  println!("[audio/{tier}] WORST cosine = {worst:.8}");
  worst
}

/// Run the text comparison against `onnx_file`, returning the worst cosine.
fn text_parity_worst(onnx_file: &str, tier: &str) -> f32 {
  let clap = TextEncoder::from_file(common::text_model_path()).expect("load clapkit text");
  let mut tc = TcTextEncoder::from_onnx_file(textclap_onnx(onnx_file), TcOptions::new())
    .expect("load textclap text ONNX");
  let mut worst = 1.0f32;
  for p in PROMPTS {
    let a = clap.embed(p).expect("clapkit text embed");
    let b = tc.embed(p).expect("textclap text embed");
    let cos = cosine(a.as_slice(), b.as_slice());
    println!("[text/{tier}] {p:<28} cosine = {cos:.8}");
    worst = worst.min(cos);
  }
  println!("[text/{tier}] WORST cosine = {worst:.8}");
  worst
}

#[test]
#[ignore = "requires clapkit models (CLAPKIT_TEST_MODELS) + textclap quantized ONNX (CLAPKIT_TEXTCLAP_ONNX)"]
fn audio_per_window_parity_vs_textclap_quantized() {
  let worst = audio_parity_worst("audio_model_quantized.onnx", "quant");
  assert!(
    (AUDIO_QUANT_LO..=AUDIO_QUANT_HI).contains(&worst),
    "audio quantized-parity worst cosine {worst:.8} outside pinned band [{AUDIO_QUANT_LO}, {AUDIO_QUANT_HI}] \
     — a shift in EITHER direction is a finding (re-measure, do not just widen)"
  );
}

#[test]
#[ignore = "requires clapkit models (CLAPKIT_TEST_MODELS) + textclap quantized ONNX (CLAPKIT_TEXTCLAP_ONNX)"]
fn text_parity_vs_textclap_quantized() {
  let worst = text_parity_worst("text_model_quantized.onnx", "quant");
  assert!(
    (TEXT_QUANT_LO..=TEXT_QUANT_HI).contains(&worst),
    "text quantized-parity worst cosine {worst:.8} outside pinned band [{TEXT_QUANT_LO}, {TEXT_QUANT_HI}]"
  );
}

#[test]
#[ignore = "requires clapkit models + textclap UNQUANTIZED fp32 ONNX (same-precision control)"]
fn audio_parity_vs_textclap_fp32_control() {
  assert!(
    textclap_onnx("audio_model.onnx").exists(),
    "fp32-control oracle `audio_model.onnx` absent under CLAPKIT_TEXTCLAP_ONNX ({:?}). The README \
     marks the fp32 control load-bearing, so a FORCED run must not green-skip it — fetch the \
     unquantized Xenova ONNX (README 'Test models') or don't force this #[ignore]d test.",
    common::textclap_onnx_dir()
  );
  let fp32 = audio_parity_worst("audio_model.onnx", "fp32");
  let quant = audio_parity_worst("audio_model_quantized.onnx", "quant");
  // The de-quantized oracle must agree at least as well as the quantized one:
  // removing quantization can only reduce the gap. The delta is the measured
  // quantization contribution.
  println!(
    "[audio] quantization contribution ≈ {:.8} (fp32 {fp32:.8} − quant {quant:.8})",
    fp32 - quant
  );
  assert!(
    (AUDIO_FP32_LO..=AUDIO_FP32_HI).contains(&fp32),
    "audio fp32-control worst cosine {fp32:.8} outside pinned band [{AUDIO_FP32_LO}, {AUDIO_FP32_HI}]"
  );
  assert!(
    fp32 + 1e-4 >= quant,
    "fp32 control ({fp32:.8}) should agree at least as well as quantized ({quant:.8})"
  );
}

#[test]
#[ignore = "requires clapkit models + textclap UNQUANTIZED fp32 ONNX (same-precision control)"]
fn text_parity_vs_textclap_fp32_control() {
  assert!(
    textclap_onnx("text_model.onnx").exists(),
    "fp32-control oracle `text_model.onnx` absent under CLAPKIT_TEXTCLAP_ONNX ({:?}). The README \
     marks the fp32 control load-bearing, so a FORCED run must not green-skip it — fetch the \
     unquantized Xenova ONNX (README 'Test models') or don't force this #[ignore]d test.",
    common::textclap_onnx_dir()
  );
  let fp32 = text_parity_worst("text_model.onnx", "fp32");
  let quant = text_parity_worst("text_model_quantized.onnx", "quant");
  println!(
    "[text] quantization contribution ≈ {:.8} (fp32 {fp32:.8} − quant {quant:.8})",
    fp32 - quant
  );
  assert!(
    (TEXT_FP32_LO..=TEXT_FP32_HI).contains(&fp32),
    "text fp32-control worst cosine {fp32:.8} outside pinned band [{TEXT_FP32_LO}, {TEXT_FP32_HI}]"
  );
  assert!(
    fp32 + 1e-4 >= quant,
    "fp32 control ({fp32:.8}) should agree at least as well as quantized ({quant:.8})"
  );
}

/// Negative control: an audio embedding and a text embedding are NOT trivially
/// aligned, so the ~parity cosines above are a real agreement, not an artifact of
/// everything landing near 1.0. A geometry/modality mix-up in the harness would
/// trip this. Hermetic w.r.t. textclap (uses clapkit's own two encoders), still
/// model-gated.
#[test]
#[ignore = "requires clapkit models (CLAPKIT_TEST_MODELS)"]
fn cross_modal_cosine_is_far_below_the_parity_floor() {
  let audio = AudioEncoder::from_file(common::audio_model_path()).unwrap();
  let text = TextEncoder::from_file(common::text_model_path()).unwrap();
  let a = audio.embed_window(&jfk_samples()).unwrap();
  let t = text.embed("a jet engine roaring at takeoff").unwrap();
  let cross = cosine(a.as_slice(), t.as_slice());
  println!("[negative-control] audio↔unrelated-text cosine = {cross:.8}");
  assert!(
    cross < 0.90,
    "cross-modal cosine {cross:.8} is implausibly high — the parity metric is not discriminating"
  );
}
