//! Gate 1 (spec §5, **re-framed 2026-07-13**): speakerkit's CoreML
//! `SegmentModel` vs dia's own `ort` `pyannote/segmentation-3.0` inference,
//! per chunk.
//!
//! Both sides run the SAME model (pyannote/segmentation-3.0) on the SAME raw
//! 16 kHz waveform — dia-ort's powerset output is the committed golden
//! (`tests/fixtures/golden/*.json`, produced by `tests/generate_goldens.rs`);
//! CoreML re-runs the conversion (`pyannote_segmentation.mlmodelc`).
//!
//! **Both sides emit `log(softmax(·))`, not raw logits** — this file, the
//! golden's `seg_logits` field name, and `generate_goldens.rs` all said
//! "raw logits, neither side softmaxes" until the graphs were read.
//! `pyannote_segmentation.mlmodelc/model.mil` ends `softmax` → `log` →
//! `-> (segments)` (quoted in `crate::segment`'s module doc), and the
//! committed ORT golden is log-probabilities on its own arithmetic: all
//! 4123 values are `<= 0` and every 7-class row satisfies
//! `sum(exp(row)) == 1.000000`. The comparison below is therefore still
//! apples-to-apples — log-probs vs log-probs, identical frame-major
//! `[frame * 7 + class]` layout — and its conclusions stand. Only the
//! description of WHAT is being compared was wrong.
//!
//! The naming is left as-is (`seg_logits` in the golden, `logits` locals):
//! renaming the committed fixture field would churn every golden for no
//! behavioral gain. The values are log-probabilities; read them as such.
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

use coremlit::{
  ComputeUnits,
  audio::speaker::segment::{
    POWERSET_CLASSES, SEG_NUM_SLOTS, SegmentModel, SegmentModelOptions, multilabel,
  },
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

/// dia's EXACT audio-in segmentation decode of one chunk's powerset
/// log-probabilities: per frame, `softmax_row` THEN hard argmax THEN the
/// powerset→speaker table — replicating `diarization/src/offline/owned.rs:
/// 479-497` (`softmax_row(&row)` then `powerset_to_speakers_hard(&probs)`).
/// Returns the same `[num_frames * SEG_NUM_SLOTS]` hard 0/1 mask layout, in the
/// same frame-major order, as speakerkit's shipping [`multilabel`].
///
/// speakerkit's `multilabel` argmaxes the log-probs DIRECTLY, without the
/// `softmax_row`, which `coremlit::audio::speaker::segment`'s module doc proves is
/// order-for-order dia's decode *in exact arithmetic* — `log(softmax(z))` is
/// `z` shifted by a per-row constant, which preserves both the argmax and its
/// exact ties. Over f32 the shortcut and dia's real path can still diverge on a
/// near-tie: `softmax_row`'s `exp`/divide can round two log-probs that differ
/// by one ULP to the SAME probability, turning a strict `>` into a tie that the
/// lowest-index rule then resolves the other way. This function is dia's real
/// f32 path, used for the ORT side of the gate so the oracle is decoded exactly
/// as dia's pipeline decodes it. `near_tie_softmax_can_flip_the_argmax`
/// exhibits such a divergence on a crafted row; `golden_direct_and_dia_decode_
/// agree` asserts none occurs on any committed golden row (today's baseline).
///
/// The table is dia's `TABLE` (`diarization/src/segment/powerset.rs:77-85`),
/// byte-identical to speakerkit's private `segment::POWERSET_TABLE` (silence, A,
/// B, C, A+B, A+C, B+C); replicated here because that constant is not public and
/// this file is not `dia`-gated (it must decode from the committed golden alone,
/// no `dia`/`ort` dependency).
fn dia_exact_multilabel(logits: &[f32], num_frames: usize) -> Vec<f64> {
  const TABLE: [[f64; SEG_NUM_SLOTS]; POWERSET_CLASSES] = [
    [0.0, 0.0, 0.0], // silence
    [1.0, 0.0, 0.0], // A
    [0.0, 1.0, 0.0], // B
    [0.0, 0.0, 1.0], // C
    [1.0, 1.0, 0.0], // A+B
    [1.0, 0.0, 1.0], // A+C
    [0.0, 1.0, 1.0], // B+C
  ];
  assert_eq!(
    logits.len(),
    num_frames * POWERSET_CLASSES,
    "logits.len() must equal num_frames * POWERSET_CLASSES"
  );
  let mut out = Vec::with_capacity(num_frames * SEG_NUM_SLOTS);
  for row in logits.as_chunks::<POWERSET_CLASSES>().0 {
    let probs = common::softmax_row(row);
    out.extend_from_slice(&TABLE[common::powerset_argmax(&probs)]);
  }
  out
}

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
      // disagreements. The two sides use the two PRODUCTION decodes, not one
      // shared shortcut: the CoreML side runs speakerkit's shipping `multilabel`
      // (direct argmax of the log-probs — its own module doc proves that is
      // order-for-order dia's decode in exact arithmetic), and the ORT side runs
      // dia's EXACT audio-in sequence, `softmax_row` THEN argmax
      // (`diarization/src/offline/owned.rs:479-497`), via
      // [`dia_exact_multilabel`]. Decoding the oracle the way dia's pipeline
      // actually decodes it — rather than through speakerkit's own
      // no-softmax shortcut — is the honest end-to-end parity; over reals the
      // two decodes coincide, over f32 they can differ on a near-tie
      // (`golden_direct_and_dia_decode_agree` pins that they do not on any
      // committed golden row).
      let coreml_set = multilabel(&coreml, golden.num_frames);
      let ort_set = dia_exact_multilabel(&gc.seg_logits, golden.num_frames);

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
          // set-flip is a class-flip, the powerset table being injective). Each
          // side's class matches its own decode above — CoreML direct, ORT
          // through dia's softmax (`so`, already computed for softmax_abs).
          let ac = common::powerset_argmax(&coreml[lo..hi]);
          let ao = common::powerset_argmax(&so);
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
  // NOTE: `agreement` pools both fixtures into one ratio, so a ≤3-flip burst
  // confined to a single clip still clears this floor — acceptable now; a
  // per-fixture floor could be added later if localized sensitivity is
  // wanted.
  assert!(
    agreement >= SEG_DECISION_AGREEMENT_MIN,
    "seg multilabel DECISION agreement {:.4}% ({total_flips} flips / {total_frames} frames) fell \
     below the re-framed floor {:.4}% — a genuine segmentation regression (spec §5). Do NOT loosen \
     the threshold; investigate the CoreML decode.",
    agreement * 100.0,
    SEG_DECISION_AGREEMENT_MIN * 100.0
  );
}

/// The whole reason the ORT side of the gate decodes through dia's exact
/// `softmax_row`-then-argmax sequence rather than speakerkit's direct-argmax
/// shortcut: over f32 the two DIVERGE on a near-tie. Here two powerset logits
/// one ULP apart are the row's top pair; direct argmax picks the strictly-larger
/// one (the higher index), but `softmax_row`'s `exp` rounds them to the SAME f32
/// probability, so the lowest-index tie rule picks the lower index instead.
///
/// Hermetic and always-run. MUTATION PROOF: widening the pair from one ULP to a
/// real gap (e.g. `b = a + 1.0`) stops the softmax collapse — `probs[1] ==
/// probs[2]` and the final `assert_ne!` both go red — so this test genuinely
/// depends on the near-tie, it is not vacuously true.
#[test]
fn near_tie_softmax_can_flip_the_argmax() {
  // `a` and the next representable f32 above it. At this magnitude one ULP is
  // ~2^-27, well under the ~2^-25 gap at which `exp` stops rounding to 1.0, so
  // after subtracting the max the two collapse to the same probability.
  let a = 0.1_f32;
  let b = f32::from_bits(a.to_bits() + 1);
  assert!(b > a, "b must be the next f32 above a");

  // Index 2 (`b`) is the strict RAW max; index 1 (`a`) is one ULP below it.
  // Every other class sits far below both.
  let row = [-10.0_f32, a, b, -10.0, -10.0, -10.0, -10.0];

  // speakerkit's shipping decode argmaxes the values DIRECTLY: `b` wins (2).
  let direct = common::powerset_argmax(&row);
  assert_eq!(
    direct, 2,
    "direct argmax must pick the strictly-largest logit"
  );

  // dia's decode softmaxes FIRST: `exp` collapses `a` and `b` onto the same f32
  // probability, and the lowest-index tie rule then picks index 1.
  let probs = common::softmax_row(&row);
  assert_eq!(
    probs[1], probs[2],
    "softmax must round the one-ULP-apart logits to the SAME probability (the collapse)"
  );
  let dia = common::powerset_argmax(&probs);
  assert_eq!(dia, 1, "softmax-then-argmax must take the lowest-index tie");

  assert_ne!(
    direct, dia,
    "direct argmax and dia's softmax-then-argmax must DIVERGE on this row — the exact \
     f32 hazard the ORT side is decoded through dia's real sequence to avoid"
  );
}

/// F3 baseline, made EXPLICIT and hermetic. On EVERY committed golden row,
/// speakerkit's shipping direct-argmax decode and dia's exact
/// softmax-then-argmax decode agree — so switching the gate's ORT side from
/// `multilabel` to [`dia_exact_multilabel`] does not move the measured flip
/// count on the committed oracle (no near-tie collapse bites a real frame
/// today). If a future re-cut golden ever disagrees, that is a genuine finding
/// AND a deliberate golden decision, not a silent pass: this fails loudly and
/// names every diverging frame. Reads only the committed
/// `tests/fixtures/golden/*.json` — no models, no `dia`/`ort`.
#[test]
fn golden_direct_and_dia_decode_agree() {
  let mut divergences: Vec<String> = Vec::new();
  let mut total_rows = 0usize;
  for fixture in common::FIXTURES {
    let golden = common::load_golden(fixture.name);
    for (c, chunk) in golden.chunks.iter().enumerate() {
      assert_eq!(
        chunk.seg_logits.len(),
        golden.num_frames * POWERSET_CLASSES,
        "{} chunk {c}: golden seg_logits length vs num_frames",
        fixture.name
      );
      for (f, row) in chunk
        .seg_logits
        .as_chunks::<POWERSET_CLASSES>()
        .0
        .iter()
        .enumerate()
      {
        total_rows += 1;
        let direct = common::powerset_argmax(row);
        let dia = common::powerset_argmax(&common::softmax_row(row));
        if direct != dia {
          divergences.push(format!(
            "{} chunk {c} frame {f}: direct_argmax={direct} dia_softmax_argmax={dia}",
            fixture.name
          ));
        }
      }
    }
  }
  assert!(
    total_rows > 0,
    "read zero golden rows — the committed oracle vanished, the check would be vacuous"
  );
  assert!(
    divergences.is_empty(),
    "{} of {total_rows} committed golden rows decode DIFFERENTLY under speakerkit's direct argmax \
     vs dia's softmax-then-argmax — a near-tie collapse now bites a real frame. This is a finding \
     AND a deliberate golden decision, not a silent pass; investigate before re-baselining:\n  {}",
    divergences.len(),
    divergences.join("\n  ")
  );
}

/// codex r7 F4: every committed golden's `seg_logits` really IS a powerset
/// log-softmax — each element `≤ 0`, each row's `Σ exp = 1` — enforced against the
/// oracle ON DISK, not just asserted in the generator's prose. A regeneration
/// that emitted RAW logits (positive values, unnormalized rows) with the argmax
/// ordering preserved would still decode the same speakers and pass the parity
/// suites, yet break the invariant the whole `softmax → log` comparison rests on.
/// This is the committed-side half of [`common::check_seg_log_probs`] (the
/// generator runs the other half before serializing). Reads only the committed
/// `tests/fixtures/golden/*.json` — no models, no `dia`/`ort`.
#[test]
fn committed_golden_seg_rows_are_log_probs() {
  let mut total_rows = 0usize;
  for fixture in common::FIXTURES {
    let golden = common::load_golden(fixture.name);
    for (c, chunk) in golden.chunks.iter().enumerate() {
      common::check_seg_log_probs(&chunk.seg_logits, golden.num_frames)
        .unwrap_or_else(|e| panic!("{} chunk {c}: {e}", fixture.name));
      total_rows += golden.num_frames;
    }
  }
  assert!(
    total_rows > 0,
    "read zero golden rows — the committed oracle vanished, the check would be vacuous"
  );
}
