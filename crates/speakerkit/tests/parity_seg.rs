//! Gate 1 (spec §5, **re-framed 2026-07-13**): speakerkit's CoreML
//! `SegmentModel` vs dia's own `ort` `pyannote/segmentation-3.0` inference,
//! per chunk.
//!
//! Both sides run the SAME model (pyannote/segmentation-3.0) on the SAME raw
//! 16 kHz waveform — dia-ort's raw powerset logits are the committed golden
//! (`tests/fixtures/golden/*.json`, produced by `tests/generate_goldens.rs`);
//! CoreML re-runs the conversion (`pyannote_segmentation.mlmodelc`). Neither
//! side softmaxes (both return raw logits), so the raw comparison is
//! logit-vs-logit in the identical frame-major `[frame * 7 + class]` layout.
//!
//! # The re-frame: the DECISION metric gates; the raw stats are REPORTED
//!
//! This suite once asserted a raw `1e-3` max-abs tolerance on the segmentation
//! logits and a zero-flip budget. It was **RED by design** (Task 6): the
//! measured raw max-abs is **0.221** (≈200× over) and there is **1 multilabel
//! flip / 3534**. Neither is a defect — they are the physics of comparing two
//! INDEPENDENT conversions of one net (FluidAudio's `.mlmodelc`, built with a
//! different coremltools/torch toolchain, vs pyannote's ONNX export). A
//! raw-value tolerance is simply **not achievable** across two conversions, and
//! Task 6 proved (input-match verified, FNV-checked) that the divergence is a
//! genuine cross-conversion difference, not a harness bug.
//!
//! Spec §5 mandates the re-scope, and this suite now implements it: **the pass
//! criterion is the DECISION-level metric** — the per-frame hard multilabel
//! speaker-set agreement, which is exactly the tensor that feeds `dia`'s
//! clustering ([`multilabel`], the downstream decision) — **and the raw-logit /
//! softmax max-abs are REPORTED, not asserted.** This is a re-scoping, not a
//! loosening: the raw stats are still printed every run, so a genuine numeric
//! regression stays visible; what changed is that the *gate* is now the
//! decision that actually matters rather than an unachievable proxy for it.
//!
//! ## The threshold is a CONTROLLER DECISION (flagged, not silently set)
//!
//! [`SEG_DECISION_AGREEMENT_MIN`] is set to **0.999** (≥ 99.9% of frames must
//! decode to the identical 3-slot speaker set). Measured today: **99.9717%**
//! (1 flip / 3534), so it passes with two flips of headroom. That single flip
//! is a documented benign near-tie — `07_yuhewei` chunk 0 frame 137, where
//! dia-ort's logits land on silence by 0.0028 and CoreML's land on speaker C by
//! 0.0020 (both see the frame as ~50/50). A "0 flips" gate would red on that
//! physics; "≥ 99.9%" tolerates one more such near-tie without masking a real
//! regression. The **value** is a controller decision that may be revised (a
//! tighter 99.95% ⇒ ≤ 1 flip, or an absolute "≤ 3 flips" budget, are equally
//! defensible on this 3534-frame corpus) — see the task report.
//!
//! ## What still makes this a GATE (mutation-proven falsifiable)
//!
//! A gate that cannot fail is not a gate. This one fails on a genuine
//! segmentation regression: if the CoreML model's decode diverges from dia-ort
//! by more than 0.1% of frames — a bad conversion, a wrong compute unit, an
//! axes-swapped output — the agreement drops below the floor. That was verified
//! by mutation (a one-sided perturbation of the CoreML logits craters the
//! agreement to ~0%; recorded in the task report). Note a SYMMETRIC mutation —
//! e.g. editing the shared `POWERSET_TABLE` [`multilabel`] uses — canNOT
//! falsify THIS suite, because both sides decode through it and the change
//! cancels; that same table-edit DOES falsify the argmax accuracy suite
//! (`parity_argmax_accuracy`), whose argmax side decodes in-graph and shares no
//! table with the dia oracle. Each gate is falsified by the perturbation its
//! own structure admits.
//!
//! `#[ignore]` (needs the gitignored `Models/speakerkit/` artifacts, like
//! `tests/model_io.rs`); run with local models via
//! `cargo test -p speakerkit -- --ignored`.

mod common;

use coremlit::ComputeUnits;
use speakerkit::segment::{
  POWERSET_CLASSES, SEG_NUM_SLOTS, SegmentModel, SegmentModelOptions, multilabel,
};

/// Re-framed Gate-1 pass criterion (spec §5): the minimum fraction of frames
/// whose CoreML hard multilabel speaker set equals dia-ort's. Set to **0.999**
/// (measured 99.9717%, 1 flip / 3534). This is the DECISION metric that feeds
/// `dia`'s clustering, not the raw-logit proxy the earlier `1e-3` max-abs
/// asserted (that stat is now REPORTED below, never gated — it is unachievable
/// across two independent conversions, Task 6). The exact value is a
/// **controller decision** (see the module doc's "The threshold is a CONTROLLER
/// DECISION"); it is never loosened to hide a regression, only revised
/// deliberately.
const SEG_DECISION_AGREEMENT_MIN: f64 = 0.999;

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

  // REPORTED, not asserted: the raw-logit and softmax max-abs across two
  // independent conversions (spec §5 re-frame).
  let mut worst_max_abs = 0.0f64;
  let mut worst_softmax_abs = 0.0f64;
  // GATED: the decision-level multilabel disagreements.
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

      // REPORTED: raw-logit max-abs across the two conversions (spec §5 —
      // never gated; unachievable across independent conversions).
      let max_abs = common::max_abs_diff(&coreml, &gc.seg_logits);

      // GATED: decode BOTH sides to the hard multilabel speaker set — the exact
      // tensor that feeds `dia`'s clustering — and count per-frame set
      // disagreements. Routing through `multilabel` (POWERSET_TABLE) makes this
      // the literal downstream DECISION, identical in definition to the argmax
      // accuracy suite's metric.
      let coreml_set = multilabel(&coreml, golden.num_frames);
      let ort_set = multilabel(&gc.seg_logits, golden.num_frames);

      let mut flips = 0usize;
      let mut softmax_abs = 0.0f64;
      for f in 0..golden.num_frames {
        let lo = f * POWERSET_CLASSES;
        let hi = lo + POWERSET_CLASSES;
        // REPORTED: softmax max-abs — the divergence's downstream impact (dia
        // softmaxes these logits before onset/argmax), always <= raw max-abs.
        let (sc, so) = (
          common::softmax_row(&coreml[lo..hi]),
          common::softmax_row(&gc.seg_logits[lo..hi]),
        );
        softmax_abs = softmax_abs.max(common::max_abs_diff(&sc, &so));

        // The DECISION: does the hard 3-slot speaker set agree this frame?
        let s = f * SEG_NUM_SLOTS;
        if coreml_set[s..s + SEG_NUM_SLOTS] != ort_set[s..s + SEG_NUM_SLOTS] {
          flips += 1;
          // Diagnostic label: the powerset class each side argmaxed to (a
          // set-flip is a class-flip, POWERSET_TABLE being injective).
          let ac = common::powerset_argmax(&coreml[lo..hi]);
          let ao = common::powerset_argmax(&gc.seg_logits[lo..hi]);
          flip_sites.push((fixture.name, c, f, ao, ac));
        }
      }

      worst_max_abs = worst_max_abs.max(max_abs);
      worst_softmax_abs = worst_softmax_abs.max(softmax_abs);
      total_flips += flips;
      total_frames += golden.num_frames;
      println!(
        "[{}] chunk {c}: logit_max_abs={max_abs:.3e} (REPORTED)  softmax_max_abs={softmax_abs:.3e} \
         (REPORTED)  decision_flips={flips}/{}",
        fixture.name, golden.num_frames
      );
    }
  }

  let agreement = (total_frames - total_flips) as f64 / total_frames as f64;
  println!(
    "SEG GATE 1 (re-framed, spec §5): DECISION agreement {}/{} = {:.4}% (min {:.4}%); \
     multilabel flips={total_flips}/{total_frames} | REPORTED raw stats: worst logit_max_abs={:.6e} \
     (NOT gated — cross-conversion physics, Task 6), worst softmax_max_abs={:.6e}",
    total_frames - total_flips,
    total_frames,
    agreement * 100.0,
    SEG_DECISION_AGREEMENT_MIN * 100.0,
    worst_max_abs,
    worst_softmax_abs,
  );
  for (fx, c, f, ao, ac) in &flip_sites {
    println!("  FLIP: {fx} chunk {c} frame {f}: ort_argmax_class={ao} coreml_argmax_class={ac}");
  }

  // THE GATE: the decision-level agreement (raw max-abs is REPORTED above, not
  // asserted — spec §5). Fails on a genuine seg regression (mutation-proven,
  // module doc); the threshold value is a controller decision, never loosened
  // to hide one.
  assert!(
    agreement >= SEG_DECISION_AGREEMENT_MIN,
    "seg multilabel DECISION agreement {:.4}% ({total_flips} flips / {total_frames} frames) fell \
     below the re-framed floor {:.4}% — a genuine segmentation regression (spec §5). Do NOT loosen \
     the threshold; investigate the CoreML decode.",
    agreement * 100.0,
    SEG_DECISION_AGREEMENT_MIN * 100.0
  );
}
