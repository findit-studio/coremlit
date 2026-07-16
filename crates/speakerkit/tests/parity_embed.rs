//! Gate 2 (spec §6.2): speakerkit's CoreML `EmbedModel` vs dia's own `ort`
//! WeSpeaker ResNet34-LM inference, per `(chunk, slot)`.
//!
//! # Precision-matched pairing (the ONLY meaningful Gate-2 comparison)
//!
//! dia-ort runs the **fp32** `wespeaker_resnet34_lm.onnx` (26.7 MB float32).
//! The conversion-fidelity gate therefore compares it against the **fp32**
//! CoreML sibling `wespeaker.mlmodelc` ([`common::embed_fp32_path`]) — NOT the
//! int8-palettized shipping artifact `wespeaker_v2.mlmodelc`. T3 measured
//! ~0.90-0.92 cosine int8-vs-fp32 (pure quantization cost); comparing int8
//! CoreML against fp32 ort would fold that quantization loss into the gate and
//! produce a FALSE fail. The int8 path is measured separately below as
//! informational context, never gated at 0.9999.
//!
//! # Cross-fbank caveat (spec §2.4)
//!
//! Even at matched fp32 precision this is inherently a CROSS-FBANK comparison:
//! dia-ort computes the mel-fbank in Rust (`compute_full_fbank`, a
//! torchaudio-kaldi port) and feeds `fbank + weights` to the ONNX graph, while
//! CoreML's `wespeaker.mlmodelc` takes the RAW waveform and computes fbank
//! IN-GRAPH (FluidAudio's conversion). Both sides receive the identical
//! `(waveform, mask)` — the mask is replayed verbatim from the golden — so any
//! residual is the two fbank front-ends (plus ort-CPU vs CoreML-CPU resnet
//! arithmetic). If the fp32 cosine falls below [`EMBED_COS_TOL`], that is the
//! documented cross-fbank divergence: STOP and report (the fallback — convert
//! dia's own fbank-input ONNX to CoreML — is a DECISION, not an inline pivot),
//! never loosen the bound.
//!
//! `#[ignore]` (needs the gitignored `Models/speakerkit/` artifacts); run via
//! `cargo test -p speakerkit -- --ignored`.

mod common;

use coremlit::ComputeUnits;
use speakerkit::embed::{EmbedModel, EmbedModelOptions};

/// Gate-2 cosine floor: min per-`(chunk, slot)` cosine between fp32 CoreML and
/// fp32 dia-ort raw embeddings. Starting point 0.9999 (spec §6.2), settled
/// empirically (see `.superpowers/sdd/task-6-report.md`) — never loosened.
const EMBED_COS_TOL: f64 = 0.9999;

/// Sanity floor for the informational int8 measurement: quantization degrades
/// cosine (T3: ~0.90) but an int8 embedding must still be recognizably the
/// same vector. A value below this signals a harness/model error, not mere
/// quantization — so this leg asserts only the loose floor, never [`EMBED_COS_TOL`].
const INT8_SANITY_FLOOR: f64 = 0.5;

/// Replays every golden `(chunk, slot)` embedding through a CoreML embedder and
/// returns `(worst_cosine, best_cosine, count)`. `label` tags the per-slot log
/// lines. Fails hard on any non-finite CoreML output (spec §6.2 NaN/Inf
/// instant-fail) or an input-hash mismatch (input-match proof).
fn measure(model: &EmbedModel, label: &str) -> (f64, f64, usize) {
  let mut worst = 1.0f64;
  let mut best = -1.0f64;
  let mut n = 0usize;

  for fixture in common::FIXTURES {
    let golden = common::load_golden(fixture.name);
    let samples = common::load_wav_16k_mono(&common::audio_path(fixture.name));
    let chunks = common::chunk_and_pad(&samples);
    assert_eq!(
      chunks.len(),
      golden.num_chunks,
      "{}: chunk count",
      fixture.name
    );

    for (c, chunk) in chunks.iter().enumerate() {
      let gc = &golden.chunks[c];
      // INPUT-MATCH PROOF: identical audio to dia-ort (see parity_seg.rs). The
      // per-slot mask below is replayed VERBATIM from the golden, so the mask
      // input is byte-identical to dia-ort's by construction.
      assert_eq!(
        common::fnv1a_f32(chunk),
        gc.input_fnv1a,
        "{} chunk {c}: INPUT MISMATCH — CoreML and dia-ort were fed different audio",
        fixture.name
      );

      for slot in &gc.slots {
        let coreml = model
          .embed_chunk_with_frame_mask(chunk, &slot.mask)
          .expect("coreml embedding infer");
        assert!(
          coreml.iter().all(|v| v.is_finite()),
          "{label}: {} chunk {c} slot {} — CoreML embedding non-finite (instant fail)",
          fixture.name,
          slot.slot
        );
        let cos = common::cosine(&coreml, &slot.embedding);
        worst = worst.min(cos);
        best = best.max(cos);
        n += 1;
        println!(
          "[{label}] {} chunk {c} slot {}: cosine={cos:.8}",
          fixture.name, slot.slot
        );
      }
    }
  }
  // A gate that compared nothing is not a passing gate: with no `(chunk,
  // slot)` pair folded in, `worst` stays at its `1.0` seed and the caller's
  // `worst >= EMBED_COS_TOL` assert reports PERFECT parity over an empty set
  // (M1 — e.g. an empty fixture/golden set, or every slot skipped). Fail loud
  // instead of silently verifying nothing.
  assert!(
    n > 0,
    "{label}: no (chunk, slot) embeddings compared — the gate verified nothing"
  );
  (worst, best, n)
}

/// GATE 2 — fp32 CoreML (`wespeaker.mlmodelc`) vs fp32 dia-ort.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS) + committed goldens"]
fn embedding_parity_fp32_vs_dia_ort() {
  let model = EmbedModel::from_file_with(
    common::embed_fp32_path(),
    EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load wespeaker.mlmodelc (fp32)");

  let (worst, best, n) = measure(&model, "fp32");
  println!(
    "EMBED GATE 2 (fp32-CoreML vs fp32-ort): worst cosine={worst:.8}, best={best:.8} \
     over {n} (chunk,slot) (tol {EMBED_COS_TOL})"
  );
  assert!(
    worst >= EMBED_COS_TOL,
    "fp32 embedding cosine {worst:.8} < {EMBED_COS_TOL} — cross-fbank DIVERGENCE \
     (spec §2.4): STOP and report, do not loosen the bound"
  );
}

/// INFORMATIONAL — int8 CoreML (`wespeaker_v2.mlmodelc`, the shipping artifact)
/// vs fp32 dia-ort. Characterizes int8 quantization cost (T3: ~0.90-0.92); NOT
/// a conversion-fidelity gate, asserts only [`INT8_SANITY_FLOOR`].
#[test]
#[ignore = "informational (int8 quantization cost), requires local models + goldens"]
fn embedding_int8_quantization_cost_informational() {
  let model = EmbedModel::from_file_with(
    common::embed_path(),
    EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load wespeaker_v2.mlmodelc (int8)");

  let (worst, best, n) = measure(&model, "int8");
  println!(
    "EMBED int8-CoreML vs fp32-ort (INFORMATIONAL, not a gate): cosine range \
     [{worst:.6}, {best:.6}] over {n} (chunk,slot)"
  );
  assert!(
    worst > INT8_SANITY_FLOOR,
    "int8 cosine {worst:.6} below sanity floor {INT8_SANITY_FLOOR} — likely a \
     harness/model error, not mere quantization"
  );
}

// ---------------------------------------------------------------------
// Hermetic guards on `common::cosine` (M1): the metric must REJECT the
// degenerate inputs that would otherwise pass a garbage embedding off as
// perfect parity (an all-zero row → zero norm → 0/0 == NaN → discarded by
// `worst.min(NaN)`). Not `#[ignore]`d — no models needed — so they run in
// the ordinary `cargo test` gate.
// ---------------------------------------------------------------------

#[test]
#[should_panic(expected = "cosine: vector `a` has zero norm")]
fn cosine_rejects_zero_norm_first_vector() {
  let _ = common::cosine(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
}

#[test]
#[should_panic(expected = "cosine: vector `b` has zero norm")]
fn cosine_rejects_zero_norm_second_vector() {
  let _ = common::cosine(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]);
}

#[test]
#[should_panic(expected = "cosine: vector `a` contains a non-finite element")]
fn cosine_rejects_non_finite_first_vector() {
  let _ = common::cosine(&[1.0, f32::NAN, 3.0], &[1.0, 2.0, 3.0]);
}

#[test]
#[should_panic(expected = "cosine: vector `b` contains a non-finite element")]
fn cosine_rejects_non_finite_second_vector() {
  let _ = common::cosine(&[1.0, 2.0, 3.0], &[1.0, f32::INFINITY, 3.0]);
}

#[test]
fn cosine_accepts_ordinary_finite_nonzero_vectors() {
  // Identical vectors → cosine exactly 1.0; proves the new guards do not
  // reject legitimate inputs.
  let v = [0.5f32, -0.25, 0.75];
  assert!((common::cosine(&v, &v) - 1.0).abs() < 1e-12);
}
