//! Gate 1 (spec §6.1): speakerkit's CoreML `SegmentModel` vs dia's own `ort`
//! `pyannote/segmentation-3.0` inference, per chunk.
//!
//! Both sides run the SAME model (pyannote/segmentation-3.0) on the SAME raw
//! 16 kHz waveform — dia-ort's raw powerset logits are the committed golden
//! (`tests/fixtures/golden/*.json`, produced by `tests/generate_goldens.rs`);
//! CoreML re-runs the conversion (`pyannote_segmentation.mlmodelc`). Neither
//! side softmaxes (dia's `SegmentModel::infer` returns raw logits;
//! `speakerkit::segment::SegmentModel::infer` likewise), so the comparison is
//! raw-logit-vs-raw-logit in the identical frame-major `[frame * 7 + class]`
//! layout.
//!
//! Two measurements:
//! - **Logit fidelity**: max per-element |CoreML − ort| across every chunk,
//!   gated at [`SEG_MAX_ABS_TOL`].
//! - **Multilabel exactness**: per-frame powerset-argmax disagreements
//!   ("flips"); the hard 0/1 speaker mask feeds dia's clustering unchanged, so
//!   the budget is ZERO.
//!
//! `#[ignore]` (needs the gitignored `Models/speakerkit/` artifacts, like
//! `tests/model_io.rs`); run with local models via
//! `cargo test -p speakerkit -- --ignored`.

mod common;

use coremlit::ComputeUnits;
use speakerkit::segment::{POWERSET_CLASSES, SegmentModel, SegmentModelOptions};

/// Gate-1 logit tolerance: max absolute per-element divergence between CoreML
/// and dia-ort raw powerset logits. Starting point 1e-3 (spec §6.1),
/// settled empirically (see `.superpowers/sdd/task-6-report.md`) — never
/// loosened to pass.
const SEG_MAX_ABS_TOL: f64 = 1e-3;

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS) + committed goldens"]
fn segmentation_parity_vs_dia_ort() {
  // CpuOnly for run-to-run determinism (no ANE compile-latency variance) and
  // an apples-to-apples match with dia-ort's CPU EP — the same convention
  // `tests/model_io.rs` uses. Production dispatch is `ComputeUnits::All`.
  let model = SegmentModel::from_file_with(
    common::seg_path(),
    SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load pyannote_segmentation.mlmodelc");

  let mut worst_max_abs = 0.0f64;
  let mut worst_softmax_abs = 0.0f64;
  let mut total_flips = 0usize;
  let mut total_frames = 0usize;
  // (fixture, chunk, frame, ort_argmax, coreml_argmax) for every disagreement.
  let mut flip_sites: Vec<(&str, usize, usize, usize, usize)> = Vec::new();

  for fixture in common::FIXTURES {
    let golden = common::load_golden(fixture.name);
    let samples = common::load_wav_16k_mono(&common::audio_path(fixture.name));
    let chunks = common::chunk_and_pad(&samples);
    assert_eq!(
      chunks.len(),
      golden.num_chunks,
      "{}: chunk count vs golden",
      fixture.name
    );

    for (c, chunk) in chunks.iter().enumerate() {
      let gc = &golden.chunks[c];

      // ── INPUT-MATCH PROOF ──────────────────────────────────────────────
      // The samples handed to CoreML `predict` here are element-identical to
      // those `tests/generate_goldens.rs` handed to `ort::Session::run` (both
      // via `common::chunk_and_pad` over the same committed WAV). The FNV-1a
      // recorded in the golden re-proves it; a divergence below on mismatched
      // inputs would be a harness bug, not a model finding.
      assert_eq!(
        chunk.len(),
        gc.input_len,
        "{} chunk {c}: input length vs golden",
        fixture.name
      );
      assert_eq!(
        common::fnv1a_f32(chunk),
        gc.input_fnv1a,
        "{} chunk {c}: INPUT MISMATCH — CoreML and dia-ort were fed different audio",
        fixture.name
      );

      let coreml = model.infer(chunk).expect("coreml segmentation infer");
      assert_eq!(coreml.len(), gc.seg_logits.len(), "logit length");
      assert!(
        coreml.iter().all(|v| v.is_finite()),
        "{} chunk {c}: CoreML produced a non-finite logit",
        fixture.name
      );

      let max_abs = common::max_abs_diff(&coreml, &gc.seg_logits);

      let mut flips = 0usize;
      let mut softmax_abs = 0.0f64;
      for f in 0..golden.num_frames {
        let lo = f * POWERSET_CLASSES;
        let hi = lo + POWERSET_CLASSES;
        let (cr, or) = (&coreml[lo..hi], &gc.seg_logits[lo..hi]);
        // Softmax max-abs: the divergence's actual downstream impact (dia
        // softmaxes these logits before onset/argmax), always <= raw max-abs.
        let (sc, so) = (common::softmax_row(cr), common::softmax_row(or));
        softmax_abs = softmax_abs.max(common::max_abs_diff(&sc, &so));
        let (ac, ao) = (common::powerset_argmax(cr), common::powerset_argmax(or));
        if ac != ao {
          flips += 1;
          flip_sites.push((fixture.name, c, f, ao, ac));
        }
      }

      worst_max_abs = worst_max_abs.max(max_abs);
      worst_softmax_abs = worst_softmax_abs.max(softmax_abs);
      total_flips += flips;
      total_frames += golden.num_frames;
      println!(
        "[{}] chunk {c}: logit_max_abs={max_abs:.3e}  softmax_max_abs={softmax_abs:.3e}  \
         flips={flips}/{}",
        fixture.name, golden.num_frames
      );
    }
  }

  println!(
    "SEG GATE 1: worst logit_max_abs={worst_max_abs:.6e} (tol {SEG_MAX_ABS_TOL:.0e}); \
     worst softmax_max_abs={worst_softmax_abs:.6e}; multilabel flips={total_flips}/{total_frames}"
  );
  for (fx, c, f, ao, ac) in &flip_sites {
    println!("  FLIP: {fx} chunk {c} frame {f}: ort_argmax={ao} coreml_argmax={ac}");
  }
  assert!(
    worst_max_abs <= SEG_MAX_ABS_TOL,
    "seg logit max-abs {worst_max_abs:.6e} exceeds tolerance {SEG_MAX_ABS_TOL:.0e} \
     — report DIVERGENCE, do not loosen (spec §6.1)"
  );
  assert_eq!(
    total_flips, 0,
    "multilabel flips {total_flips} (budget is 0) — report DIVERGENCE (spec §6.1)"
  );
}
