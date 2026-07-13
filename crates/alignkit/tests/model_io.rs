//! Ground-truth introspection of `chordai/wav2vec2-base960h-aligner-coreml`
//! (design spec §3 Candidate A) — the CTC acoustic encoder alignkit's
//! forced-aligner wraps. Every claim below comes from loading the real
//! `.mlmodelc` via `coremlit::Model::load` + `.description()`, or from
//! actually running it (`Model::predict`); the model card's own claims are a
//! HYPOTHESIS re-verified here, not trusted blind.
//!
//! # Artifact (`Models/alignkit/`, gitignored, fetched dev-time)
//!
//! Source: <https://huggingface.co/chordai/wav2vec2-base960h-aligner-coreml>,
//! revision (commit SHA) `a7b796f23585b48af9f21977412953680291f27d` — pinned
//! at download time; `hf api`'s `sha` field and `git ls-remote HEAD` agree.
//!
//! | File | Role |
//! |---|---|
//! | `base960h_aligner.mlpackage/Data/com.apple.CoreML/model.mlmodel` | model graph (downloaded) |
//! | `base960h_aligner.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | weights, fp16 (downloaded) |
//! | `base960h_aligner.mlpackage/Manifest.json` | mlpackage manifest (downloaded) |
//! | `base960h_dict.json` | 29-entry CTC vocab, consumed by Task B2 (downloaded) |
//! | `base960h_aligner.mlmodelc/` | **targeted** — compiled from the `.mlpackage` row above via `xcrun coremlcompiler compile`; not itself downloaded (`coremlit::Model::load` only accepts a compiled `.mlmodelc`; see `tests/common::model_path`) |
//!
//! # License
//!
//! HuggingFace `cardData.license` = `apache-2.0` (also tagged
//! `license:apache-2.0`). The repo's own README states the weights are
//! converted from `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` ("Facebook/
//! Meta wav2vec2-base fine-tuned on LibriSpeech 960h; Apache-2.0 lineage").
//! Apache-2.0 requires preserving notices, not a specific attribution
//! string; this record (repo id, revision, license) is that preservation.
//!
//! # Per-file SHA-256 (downloaded artifacts only)
//!
//! `base960h_aligner.mlmodelc` is a local `coremlcompiler` output, not a
//! downloaded artifact, so it is deliberately NOT pinned here — a different
//! Xcode/`coremlcompiler` version could legitimately re-emit different
//! compiled bytes for the same source `.mlpackage`; see
//! `source_artifacts_match_pinned_sha256` for what this test suite actually
//! checks against drift/corruption.
//!
//! | File | SHA-256 |
//! |---|---|
//! | `base960h_aligner.mlpackage/Data/com.apple.CoreML/model.mlmodel` | `25e58f76ec1de033c7ae52d20e5bc8a468657b1a7800e2340f2e5b962da8dfbb` |
//! | `base960h_aligner.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `de51193fe73fb3aad085f9c794f08bfde1b939fc12f92e0834edcd4cb712e642` |
//! | `base960h_aligner.mlpackage/Manifest.json` | `58650570fbd6fe8e011f9134847da2fc7b5f1e867305e70354aa342c5b6aef93` |
//! | `base960h_dict.json` | `ef41495ab958d4416ad2f81ea51a77d4a3c79cace96e92e978c443c7bfbdd2e5` |
//!
//! # DECISION
//!
//! - **Target: `base960h_aligner.mlmodelc`** (spec §3 Candidate A — the only
//!   candidate this task downloads; Candidate B, an in-house
//!   coremltools conversion, is the documented STOP fallback if Candidate A
//!   later fails a parity gate, spec §3/§10, not evaluated here).
//! - **Representation:** see `emissions_are_log_probs_not_raw_logits`
//!   below — the graph-truth investigation this task exists to pin.
//!
//! # Spec-vs-reality
//!
//! Confirmed exactly against the design spec §3's stated contract and the
//! model card: `waveform [1, 960000]` f32 -> `emissions [1, 2999, 29]` f32,
//! 20 ms/frame (stride 320 samples @ 16 kHz). No deltas found.

mod common;

use alignkit::encode::DEFAULT_ENCODER_COMPUTE;
use coremlit::{DataType, Features, Model, MultiArray};

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn base960h_aligner_io_matches_spec() {
  let model = Model::load(common::model_path(), DEFAULT_ENCODER_COMPUTE).unwrap();
  let description = model.description();

  // DECISION: this is the Task B1 encoder target — see the module doc.
  let waveform = description.input("waveform").expect("waveform input");
  assert_eq!(waveform.shape(), &[1, 960_000]);
  assert_eq!(waveform.data_type(), Some(DataType::F32));

  let emissions = description.output("emissions").expect("emissions output");
  assert_eq!(emissions.shape(), &[1, 2999, 29]);
  assert_eq!(emissions.data_type(), Some(DataType::F32));
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn source_artifacts_match_pinned_sha256() {
  let dir = common::models_dir();
  let cases = [
    (
      "base960h_aligner.mlpackage/Data/com.apple.CoreML/model.mlmodel",
      "25e58f76ec1de033c7ae52d20e5bc8a468657b1a7800e2340f2e5b962da8dfbb",
    ),
    (
      "base960h_aligner.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
      "de51193fe73fb3aad085f9c794f08bfde1b939fc12f92e0834edcd4cb712e642",
    ),
    (
      "base960h_aligner.mlpackage/Manifest.json",
      "58650570fbd6fe8e011f9134847da2fc7b5f1e867305e70354aa342c5b6aef93",
    ),
    (
      "base960h_dict.json",
      "ef41495ab958d4416ad2f81ea51a77d4a3c79cace96e92e978c443c7bfbdd2e5",
    ),
  ];
  for (relative, expected) in cases {
    let actual = common::sha256_hex(&dir.join(relative));
    assert_eq!(actual, expected, "sha256 drift on artifact {relative}");
  }
}

/// Numerically-stable per-frame logsumexp over one frame's vocab logits (or
/// log-probs — that's exactly the question this module answers), accumulated
/// in `f64` so the measurement reflects the MODEL's behavior rather than
/// this test's own summation error.
fn logsumexp(frame: &[f32]) -> f64 {
  let max = frame.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  assert!(max.is_finite(), "frame has no finite logit/log-prob");
  let max64 = f64::from(max);
  let sum: f64 = frame.iter().map(|&x| (f64::from(x) - max64).exp()).sum();
  max64 + sum.ln()
}

/// Runs `waveform` (must be exactly 960,000 samples) through the live model
/// and returns the flat `[2999 * 29]` emissions row-major buffer.
fn run_emissions(model: &Model, waveform: &[f32]) -> Vec<f32> {
  assert_eq!(waveform.len(), 960_000);
  let input = MultiArray::from_slice(&[1, 960_000], waveform).expect("waveform tensor");
  let outputs = model
    .predict(&Features::new().with("waveform", input))
    .expect("prediction succeeds");
  let emissions = outputs.get("emissions").expect("emissions output");
  let mut buf = vec![0f32; 2999 * 29];
  emissions.copy_into(&mut buf).expect("emissions copy_into");
  buf
}

/// **THE GRAPH-TRUTH TEST** (design spec §7 data flow, evaluation item 2 —
/// the blocking input to Task B3's encoder wrapper design). Determines
/// empirically whether `emissions` is raw CTC logits or already
/// log-softmaxed log-probabilities, via the one property that tells them
/// apart: a proper log-probability distribution sums to 1 in probability
/// space, so `logsumexp` over the 29-entry vocab axis is `ln(1) = 0`; raw
/// logits carry no such constraint, and `logsumexp` instead tracks the
/// logits' own (here, tens-of-units) dynamic range.
///
/// Measured (`ComputeUnits::CpuOnly`, this test's own run):
/// - Real audio (`ted_60.wav`, the full 60 s / 960,000-sample window —
///   already exactly the model's window, no padding needed): per-frame
///   `|logsumexp|` across all 2999 frames has max ≈ `5.2485e-3`, mean ≈
///   `1.0512e-3`.
/// - All-zeros input (second sample, corroborating on a degenerate input):
///   max ≈ `3.4290e-3`, mean ≈ `2.9445e-3`.
/// - The raw `emissions` values on the real-audio sample range up to
///   exactly `0.0` (`[-28.4375, 0.0]`) — the hard ceiling `log(p) <= 0`
///   admits for any probability `p <= 1`, which only a log-probability
///   tensor can hit exactly; raw logits have no such ceiling.
///
/// VERDICT: **log-probs.** Both signals agree with each other and with the
/// model card's own claim ("output | emissions float32 [1, 2999, 29] —
/// log-probs", `Models/alignkit/README.md`) — re-verified here rather than
/// trusted blind, per this module's opening paragraph.
///
/// Tolerance: `1e-2`, deliberately not the naively-hypothesized `1e-3` —
/// the measured max (`5.2485e-3`) already exceeds `1e-3`, and the model
/// card states `Precision: FLOAT16` (~3 decimal digits), which plausibly
/// explains `1e-3`-magnitude deviations accumulating over a 29-term
/// exp/sum/log per frame. `1e-2` keeps roughly 2x headroom above the
/// largest measured value while staying two-plus orders of magnitude below
/// the raw emission dynamic range (up to `28.4`) a genuinely-unnormalized
/// (raw-logit) tensor would be expected to produce — it cannot accidentally
/// mask a real logits-vs-log-probs mismatch.
///
/// DECISION CONSEQUENCE: because emissions are already log-probs, Task B3's
/// encoder wrapper must NOT re-apply softmax/log-softmax over this output —
/// it wires `asry::LogProbsTV` directly from the raw `emissions` tensor.
/// (Had the verdict instead been raw logits, the consequence would have
/// been the opposite: apply asry's `log_softmax_with_finite_guard` first.)
///
/// SUPERSEDED AS EVIDENCE, retained as a check: the model's `.mil` graph
/// settles the question directly — its final ops are `softmax` → `log` →
/// `cast(fp32)` (quoted in `alignkit::encode`'s module doc). The verdict below
/// is inferred from measured values; the graph states it. The two agree.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn emissions_are_log_probs_not_raw_logits() {
  const TOLERANCE: f64 = 1e-2;
  let model = Model::load(common::model_path(), DEFAULT_ENCODER_COMPUTE).unwrap();

  let waveform = common::load_wav_mono_f32(&common::ted_60_wav_path());
  let emissions = run_emissions(&model, &waveform);
  assert_logsumexp_near_zero(&emissions, TOLERANCE, "real audio (ted_60.wav)");

  // Corroborating signal: a log-probability can never exceed ln(1) = 0; raw
  // logits have no such ceiling.
  //
  // The `1e-3` slack is VESTIGIAL, not a live tension. The graph applies `log`
  // to a `softmax` output, which is in [0, 1] by construction, so `<= 0` is
  // guaranteed — and the measured max is exactly `0.0` on all four compute
  // placements, not merely close to it. The slack is kept only so this assert
  // reads as a ceiling check rather than an exact-equality trap; do not mistake
  // it for evidence that the model can emit slightly-positive log-probs.
  //
  // Do NOT "fix" a future positive max by clamping in `Encoder::emissions`: a
  // positive max is the signal that the model has been swapped for a
  // raw-logit CTC head, which `Emissions::from_log_probs` must be allowed to
  // reject loudly. See `alignkit::encode`'s module doc.
  let max_value = emissions.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  assert!(
    max_value <= 1e-3,
    "emissions exceed the log(p)<=0 ceiling by more than fp16 rounding slack: max={max_value}"
  );

  // Second sample (degenerate all-zeros input), per the graph-truth
  // investigation's own design: the identity should hold regardless of
  // what audio actually produced the emissions.
  let zeros = vec![0f32; 960_000];
  let emissions0 = run_emissions(&model, &zeros);
  assert_logsumexp_near_zero(&emissions0, TOLERANCE, "all-zeros input");
}

/// Asserts every frame's `logsumexp` over the 29-entry vocab axis is within
/// `tolerance` of zero. See `emissions_are_log_probs_not_raw_logits`'s doc
/// comment for what this checks and the measured values that picked
/// `tolerance`.
fn assert_logsumexp_near_zero(emissions: &[f32], tolerance: f64, sample_label: &str) {
  assert_eq!(emissions.len(), 2999 * 29);
  let (frames, remainder) = emissions.as_chunks::<29>();
  assert!(
    remainder.is_empty(),
    "2999 * 29 divides evenly; no partial frame"
  );
  for (t, frame) in frames.iter().enumerate() {
    let lse = logsumexp(frame);
    assert!(
      lse.abs() <= tolerance,
      "{sample_label}: frame {t} logsumexp={lse} exceeds tolerance {tolerance} — not log-probs?"
    );
  }
}
