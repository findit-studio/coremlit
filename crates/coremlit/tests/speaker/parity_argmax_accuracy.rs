//! Tier-2 **algorithmic accuracy** for [`ArgmaxSource`] versus the **fp32
//! dia-ort oracle** (design spec §5's tier 2), the same oracle and the same
//! metrics Task 6 held the FluidAudio source to (`tests/parity_seg.rs` +
//! `tests/parity_embed.rs`). It answers *"how close is argmax's decode to
//! fp32 pyannote/dia — and does argmax's 8-bit quantization cost accuracy?"*
//!
//! It is deliberately NOT the argmax fidelity gate — that is
//! `tests/parity_argmax_swift.rs` (our `Extraction` vs argmax's OWN Swift,
//! byte-tight, spec §5 tier 1). This suite is the OTHER tier: argmax's decode
//! vs an INDEPENDENT reference net (dia-ort's fp32 `segmentation-3.0.onnx` +
//! `wespeaker_resnet34_lm.onnx`), which no vendor's conversion can match
//! bit-for-bit. So every number here is CHARACTERIZATION, reported; the hard
//! asserts are the harness invariants (input match, finiteness, grid) plus
//! loose sanity tripwires documented at their constants.
//!
//! # The structural difference this suite must NOT paper over (spec §5)
//!
//! argmax decodes **in-graph** with its OWN semantics; dia decodes
//! **host-side**. A perfect match is therefore *not expected*, and three
//! concrete asymmetries shape every number below — each reported, none hidden:
//!
//! 1. **Segmentation is decision-level ONLY.** argmax's segmenter emits no
//!    per-frame scores — it returns the already-decoded hard `speaker_ids`
//!    (spec §3). So unlike FluidAudio's parity_seg (whose model emits a raw
//!    powerset log-prob tensor — `log(softmax)`, not raw logits — and reports a
//!    raw max-abs over it), the ONLY segmentation metric that exists for argmax
//!    is the decision: does argmax's hard 0/1 speaker mask agree with dia's
//!    [`multilabel`] decode of the fp32 log-probs? A raw-tensor max-abs simply
//!    has no argmax-side operand, so none is reported for segmentation (it IS
//!    reported for embeddings).
//! 2. **argmax's activity gate is stricter than dia's.** argmax drops a slot
//!    with `<= 2` active frames in a window (`ArgmaxSource` zeroes its column);
//!    dia keeps any slot with `>= 1` active frame. So a slot dia decodes active
//!    in 1-2 frames of a window shows up as a handful of decision
//!    disagreements. The count of such gate-dropped-but-dia-active columns is
//!    reported separately, because it explains most disagreement and is a
//!    known semantic difference, not a conversion error.
//! 3. **Embeddings are cross-mask AND cross-fbank AND cross-conversion — but
//!    FBANK dominates, not the mask (spec §5.4).** FluidAudio's parity_embed
//!    replayed dia's mask VERBATIM, so its 0.99999989 cosine isolates the
//!    WeSpeaker conversion alone. Here argmax pools over its OWN in-graph
//!    mask (`speaker_ids * (1 - overlapped)`) — which agrees with dia's
//!    decision ~99.98% of the time (the same decode item 1 measures) — and
//!    an 80-mel spectrogram from a SEPARATE `SpeakerEmbedderPreprocessor`
//!    model: argmax's OWN fbank conversion, NOT computed in-graph, against
//!    dia's kaldi fbank + exclude-overlap mask. That separate fbank is the
//!    dominant divergence (FluidAudio's *in-graph* fbank matched kaldi to
//!    1e-7 in T6; argmax's preprocessor does not); quantization is likewise
//!    near-free (both tiers land at ~0.94 cosine). A lower cosine than
//!    FluidAudio's is EXPECTED and is not a defect — it folds in every one of
//!    those differences. Reported, never asserted tight.
//!
//! # The chunk correspondence (proven, not assumed)
//!
//! dia's golden is generated on NON-overlapping 10 s chunks (`step = window`,
//! `common::chunk_and_pad`). argmax runs its own grid — 30 s chunks, 21
//! in-graph windows at a 1 s stride — and `ArgmaxSource` un-flattens each
//! window to one [`Extraction`] chunk whose index is the global window index
//! `c = k*21 + w` (the port's grid theorem, gated by
//! `parity_argmax_swift::argmax_execution_fidelity_vs_swift`). Because dia's
//! chunk `j` spans `[j*160000, +160000)` and argmax's Extraction chunk `c`
//! spans `[c*16000, +160000)`, the two describe the SAME 160 000-sample window
//! exactly when `c*16000 == j*160000`, i.e.
//!
//! ```text
//! c = j * (160000 / 16000) = j * 10        (= DIA_CHUNK_TO_ARGMAX * j)
//! ```
//!
//! valid whenever `c < extraction.num_chunks()`. Where it is not — dia's
//! zero-padded tail chunk whose argmax window `bounded()` discards
//! (`07_yuhewei` chunk 2, `c = 20 >= 17`) — that dia chunk has no argmax
//! operand and is reported as uncompared, never silently counted as agreement.
//!
//! # Input match is PROVEN before any number is trusted (the alignkit lesson)
//!
//! Both sides are fed the same committed WAV. For every compared dia chunk `j`
//! this suite re-proves `fnv1a(chunk_and_pad(samples)[j]) ==
//! golden.chunks[j].input_fnv1a` — the identical FNV the fp32 oracle recorded
//! (`tests/generate_goldens.rs`), so dia's operand is byte-verified. argmax is
//! handed the SAME `samples`, and the grid theorem above makes its window
//! `10*j` the same 160 000 samples by construction (`k=0` here, so the padded
//! 30 s buffer's `[j*160000, +160000)` slice IS `chunk_and_pad(samples)[j]`).
//!
//! # Compute unit: `CpuOnly`, the deterministic control (spec §5.3)
//!
//! All three argmax models run `CpuOnly` — the same control
//! `parity_argmax_swift`'s fidelity gate uses, and *forced* anyway because
//! argmax's fbank preprocessor hardcodes `.cpuOnly`
//! (`SpeakerPreEmbedderModel.swift:14`). dia-ort's golden is its CPU EP. So the
//! accuracy numbers are a like-for-like CPU-vs-CPU comparison, free of the ANE
//! scheduling jitter `parity_argmax_swift`'s placement study measures. The
//! shipping default is `All`; this suite deliberately does not use it (that is
//! the placement study's job, not the accuracy oracle's).
//!
//! `#[ignore]` (needs the gitignored `Models/argmax-speakerkit/` artifacts and
//! the committed fp32-dia goldens); run via
//! `ARGMAX_TEST_MODELS=… cargo test -p speakerkit -- --ignored`.

mod common;

use std::collections::BTreeSet;

use coremlit::{
  ComputeUnits,
  audio::speaker::{
    embed::EMBEDDING_DIM,
    segment::{SEG_NUM_SLOTS, multilabel},
    source::{ArgmaxComputeOptions, ArgmaxOptions, ArgmaxSource, ArgmaxVariant, ModelSource},
  },
};

/// dia's non-overlapping chunk `j` is argmax Extraction chunk `c = 10*j`:
/// `160000 / 16000`, i.e. dia's 10 s chunk stride over argmax's 1 s window
/// stride (module doc). A `const` so the one load-bearing index identity is
/// named once, not sprinkled as a literal `10`.
const DIA_CHUNK_TO_ARGMAX: usize = 160_000 / 16_000;

/// Frames per segmentation window (pyannote-3.0), shared by both sides.
const FRAMES_PER_WINDOW: usize = 589;

/// SANITY floor on segmentation decision agreement (fraction of `(frame,
/// slot)` cells where argmax's hard mask equals dia's [`multilabel`] decode),
/// per variant. This is a TRIPWIRE, not the fidelity gate: it catches a wrong
/// model, a broken index mapping, or a powerset misalignment (all of which
/// crater agreement toward 0), while leaving the actual measured value to the
/// report. Set well below the measured Baseline/W8A16 agreement (see the task
/// report) with wide margin, since the point of the number is to be *reported*,
/// not gated. `parity_seg` (Part B) is where the decision metric is a real gate.
const SEG_AGREEMENT_SANITY_FLOOR: f64 = 0.98;

/// SANITY floor on per-`(chunk, slot)` embedding cosine vs the fp32 dia-ort
/// oracle. Deliberately loose — argmax's embedder is a genuinely different
/// conversion pooling over its OWN mask (in-graph, from the segmenter) and
/// fbank (a SEPARATE `SpeakerEmbedderPreprocessor` model, NOT in-graph —
/// module doc's structural-difference item 3, where FBANK dominates the gap,
/// not the mask). The MEASURED worst is only **0.83** (07, chunk 1, slot 1),
/// FAR below the mask-matched FluidAudio path's 0.99999989 (`parity_embed`).
/// That gap is a reported FINDING, not a bug — so this is emphatically NOT a
/// precision gate; it is a gross-mismap backstop. Slot CORRESPONDENCE is
/// already guaranteed by the 99.98% segmentation agreement above (argmax
/// slot `s`'s active frames match dia slot `s`'s), so a swapped-speaker
/// embedding is caught there; this only trips on a catastrophe
/// (orthogonal/negative cosine). Set below the measured worst with margin —
/// do not raise it toward the measured value and mistake it for a fidelity
/// bound.
const EMBED_COS_SANITY_FLOOR: f64 = 0.70;

/// Max tolerated DROP in mean embedding cosine from the Baseline tier (W32A32
/// segmenter, but W16A16 — fp16, NOT fp32 — embedder) to the 8-bit W8A16
/// tier — the "does quantization cost accuracy" tripwire. Quantization is
/// EXPECTED to cost a little; a large regression (e.g. a broken W8A16
/// artifact) is a finding. Measured delta is reported; this only fails if
/// W8A16 falls materially below Baseline.
const MAX_QUANT_COS_DROP: f64 = 0.02;

/// One variant's measured accuracy against the fp32 dia-ort oracle.
#[derive(Debug, Clone)]
struct Accuracy {
  /// `(frame, slot)` decision cells compared (`compared_frames * 3`).
  seg_cells: usize,
  /// Cells where argmax's hard mask equals dia's [`multilabel`] decode.
  seg_cell_agree: usize,
  /// Frames compared (over the chunks with an argmax operand).
  frames: usize,
  /// Frames where the FULL 3-slot speaker set agrees — the metric comparable
  /// to FluidAudio's "1 flip / 3534" (`parity_seg`).
  frame_agree: usize,
  /// `(chunk, slot)` columns argmax's stricter activity gate dropped (all-zero)
  /// that dia decodes active — the documented semantic difference (§2) that
  /// explains most cell disagreement.
  gate_dropped_active: usize,
  /// dia chunks with no argmax operand (`c >= num_chunks`) — reported, not
  /// counted as agreement or disagreement.
  chunks_uncompared: usize,
  /// Worst (min) per-`(chunk, slot)` embedding cosine over slots BOTH embedded.
  worst_cos: f64,
  /// Best (max) per-`(chunk, slot)` embedding cosine.
  best_cos: f64,
  /// Mean per-`(chunk, slot)` embedding cosine.
  mean_cos: f64,
  /// Worst per-element |argmax − dia| embedding difference (the raw-value
  /// statistic the brief asks be reported alongside cosine).
  worst_max_abs: f64,
  /// IDENTITIES of the `(fixture, chunk, slot)` pairs BOTH sides embedded (the
  /// cosine denominator). A SET, not a count: the quantization-cost verdict
  /// compares this across the two tiers, and equal counts over DISJOINT
  /// identities (Baseline embeds `{A,B}`, W8A16 embeds `{A,C}`) would make the
  /// mean-cosine delta meaningless while a bare `compared_slots` count assert
  /// passed (L2).
  compared_slot_ids: BTreeSet<(&'static str, usize, usize)>,
  /// Slots dia embedded that argmax did not (activity gate / norm / bounded).
  only_dia: usize,
  /// Slots argmax embedded that dia did not.
  only_argmax: usize,
}

impl Accuracy {
  fn seg_agreement(&self) -> f64 {
    self.seg_cell_agree as f64 / self.seg_cells as f64
  }
  fn frame_agreement(&self) -> f64 {
    self.frame_agree as f64 / self.frames as f64
  }
  fn frame_flips(&self) -> usize {
    self.frames - self.frame_agree
  }
  /// `(fixture, chunk, slot)` pairs BOTH sides embedded — the cosine
  /// denominator, i.e. `compared_slot_ids.len()`.
  fn compared_slots(&self) -> usize {
    self.compared_slot_ids.len()
  }
}

/// Whether a flat embedding row is entirely zero — argmax's "not embedded"
/// marker (a dropped `(chunk, slot)` has an all-zero `raw_embeddings` row,
/// [`Extraction`]'s coupling invariant).
fn row_is_zero(row: &[f32]) -> bool {
  row.iter().all(|&v| v == 0.0)
}

/// Measures one [`ArgmaxVariant`] against the fp32 dia-ort golden over both
/// committed fixtures, on `CpuOnly`. Asserts only the harness invariants
/// (input match, finiteness, grid) — the accuracy NUMBERS are returned for the
/// caller to report and hold to the sanity tripwires.
fn measure(variant: ArgmaxVariant) -> Accuracy {
  let compute = ArgmaxComputeOptions::new()
    .with_segmenter(ComputeUnits::CpuOnly)
    .with_preprocessor(ComputeUnits::CpuOnly)
    .with_embedder(ComputeUnits::CpuOnly);
  let source = ArgmaxSource::from_dir_with(
    common::argmax_models_dir(),
    ArgmaxOptions::new()
      .with_variant(variant)
      .with_compute(compute),
  )
  .expect("load argmax models");

  let (mut seg_cells, mut seg_cell_agree) = (0usize, 0usize);
  let (mut frames, mut frame_agree) = (0usize, 0usize);
  let (mut gate_dropped_active, mut chunks_uncompared) = (0usize, 0usize);
  let (mut worst_cos, mut best_cos, mut cos_sum) = (1.0f64, -1.0f64, 0.0f64);
  let mut worst_max_abs = 0.0f64;
  let mut compared_slot_ids: BTreeSet<(&'static str, usize, usize)> = BTreeSet::new();
  let (mut only_dia, mut only_argmax) = (0usize, 0usize);

  for fixture in common::FIXTURES {
    let golden = common::load_golden(fixture.name);
    let samples = common::load_wav_16k_mono(&common::audio_path(fixture.name));
    let chunks = common::chunk_and_pad(&samples);
    assert_eq!(
      chunks.len(),
      golden.num_chunks,
      "{}: chunk_and_pad vs golden chunk count",
      fixture.name
    );

    let extraction = source.extract(&samples).expect("argmax extract");
    assert_eq!(
      extraction.num_frames_per_chunk(),
      FRAMES_PER_WINDOW,
      "{}: argmax frames/chunk",
      fixture.name
    );
    assert_eq!(
      golden.num_frames, FRAMES_PER_WINDOW,
      "{}: golden frames/chunk",
      fixture.name
    );
    let segmentations = extraction.segmentations();
    let embeddings = extraction.raw_embeddings();

    // `chunks.len() == golden.num_chunks` (asserted above), so enumerating the
    // chunks IS iterating dia's chunk grid `0..num_chunks`; `j` indexes the
    // parallel `golden.chunks` and drives the `c = 10*j` argmax map.
    for (j, chunk) in chunks.iter().enumerate() {
      let gc = &golden.chunks[j];

      // ── INPUT-MATCH PROOF (dia side, byte-verified against the oracle) ──
      // The argmax side is fed the same `samples`; the grid theorem (module
      // doc) makes its window 10*j the same 160 000-sample slice.
      assert_eq!(
        chunk.len(),
        gc.input_len,
        "{} chunk {j}: input length vs golden",
        fixture.name
      );
      assert_eq!(
        common::fnv1a_f32(chunk),
        gc.input_fnv1a,
        "{} chunk {j}: INPUT MISMATCH — dia golden and this run disagree on the audio",
        fixture.name
      );

      let c = DIA_CHUNK_TO_ARGMAX * j;
      if c >= extraction.num_chunks() {
        // dia's zero-padded tail chunk; argmax's bounded() dropped the window
        // (module doc). No operand — reported, not scored.
        chunks_uncompared += 1;
        println!(
          "[{} {variant:?}] dia chunk {j} -> argmax c={c} >= num_chunks={} : NO argmax operand \
           (bounded() dropped the mostly-padding window) — uncompared",
          fixture.name,
          extraction.num_chunks()
        );
        continue;
      }

      // ── Segmentation: argmax's hard mask vs dia's multilabel decode ──
      let dia_set = multilabel(&gc.seg_logits, FRAMES_PER_WINDOW);
      let amx = &segmentations[c * FRAMES_PER_WINDOW * SEG_NUM_SLOTS..]
        [..FRAMES_PER_WINDOW * SEG_NUM_SLOTS];
      assert!(
        amx.iter().all(|v| v.is_finite()),
        "{} chunk {j}: argmax segmentation non-finite",
        fixture.name
      );

      let mut chunk_cell_agree = 0usize;
      let mut chunk_frame_agree = 0usize;
      for f in 0..FRAMES_PER_WINDOW {
        let mut all_agree = true;
        for s in 0..SEG_NUM_SLOTS {
          let idx = f * SEG_NUM_SLOTS + s;
          if amx[idx] == dia_set[idx] {
            chunk_cell_agree += 1;
          } else {
            all_agree = false;
          }
        }
        if all_agree {
          chunk_frame_agree += 1;
        }
      }
      seg_cells += FRAMES_PER_WINDOW * SEG_NUM_SLOTS;
      seg_cell_agree += chunk_cell_agree;
      frames += FRAMES_PER_WINDOW;
      frame_agree += chunk_frame_agree;

      // Decompose: columns argmax's activity gate dropped that dia has active.
      let mut chunk_gate_dropped = 0usize;
      for s in 0..SEG_NUM_SLOTS {
        let amx_active = (0..FRAMES_PER_WINDOW).any(|f| amx[f * SEG_NUM_SLOTS + s] != 0.0);
        let dia_active = (0..FRAMES_PER_WINDOW).any(|f| dia_set[f * SEG_NUM_SLOTS + s] != 0.0);
        if dia_active && !amx_active {
          chunk_gate_dropped += 1;
        }
      }
      gate_dropped_active += chunk_gate_dropped;

      // ── Embeddings: argmax's raw row vs dia's golden embedding ──
      // dia's golden lists only the slots IT embedded; argmax may embed a
      // different subset (§2). Compare the intersection; count the symmetric
      // differences.
      let dia_slots: Vec<usize> = gc.slots.iter().map(|s| s.slot).collect();
      for slot in &gc.slots {
        let s = slot.slot;
        let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
        let amx_row = &embeddings[base..base + EMBEDDING_DIM];
        if row_is_zero(amx_row) {
          only_dia += 1;
          continue;
        }
        assert!(
          amx_row.iter().all(|v| v.is_finite()),
          "{} chunk {j} slot {s}: argmax embedding non-finite",
          fixture.name
        );
        let cos = common::cosine(amx_row, &slot.embedding);
        let max_abs = common::max_abs_diff(amx_row, &slot.embedding);
        worst_cos = worst_cos.min(cos);
        best_cos = best_cos.max(cos);
        cos_sum += cos;
        worst_max_abs = worst_max_abs.max(max_abs);
        // Record the IDENTITY, not just a count, so the two tiers' compared
        // sets can be checked for equality (L2), not merely equal size.
        compared_slot_ids.insert((fixture.name, c, s));
        println!(
          "[{} {variant:?}] chunk {j} (c={c}) slot {s}: cosine={cos:.8} max|diff|={max_abs:.4e}",
          fixture.name
        );
      }
      // Slots argmax embedded that dia did not.
      for s in 0..SEG_NUM_SLOTS {
        if dia_slots.contains(&s) {
          continue;
        }
        let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
        if !row_is_zero(&embeddings[base..base + EMBEDDING_DIM]) {
          only_argmax += 1;
        }
      }
    }
  }

  let mean_cos = if compared_slot_ids.is_empty() {
    0.0
  } else {
    cos_sum / compared_slot_ids.len() as f64
  };
  Accuracy {
    seg_cells,
    seg_cell_agree,
    frames,
    frame_agree,
    gate_dropped_active,
    chunks_uncompared,
    worst_cos,
    best_cos,
    mean_cos,
    worst_max_abs,
    compared_slot_ids,
    only_dia,
    only_argmax,
  }
}

/// Prints one variant's accuracy line and holds it to the sanity tripwires.
fn report(label: &str, a: &Accuracy) {
  println!(
    "\n=== ARGMAX {label} vs fp32 dia-ort (CpuOnly) ===\n  \
     SEG decision: cell agreement {}/{} = {:.4}% | full-set frame agreement {}/{} = {:.4}% \
     (flips {}) | activity-gate-dropped-but-dia-active columns: {} | uncompared dia chunks: {}\n  \
     EMBED cosine: worst={:.8} best={:.8} mean={:.8} | worst max|diff|={:.4e} | compared {} slots \
     (only-dia {}, only-argmax {})",
    a.seg_cell_agree,
    a.seg_cells,
    a.seg_agreement() * 100.0,
    a.frame_agree,
    a.frames,
    a.frame_agreement() * 100.0,
    a.frame_flips(),
    a.gate_dropped_active,
    a.chunks_uncompared,
    a.worst_cos,
    a.best_cos,
    a.mean_cos,
    a.worst_max_abs,
    a.compared_slots(),
    a.only_dia,
    a.only_argmax,
  );
  assert!(
    a.seg_agreement() >= SEG_AGREEMENT_SANITY_FLOOR,
    "{label}: segmentation decision agreement {:.4}% below sanity floor {:.2}% — a wrong model, a \
     broken index mapping, or a powerset misalignment, NOT mere cross-conversion physics. \
     Investigate; do not lower this floor.",
    a.seg_agreement() * 100.0,
    SEG_AGREEMENT_SANITY_FLOOR * 100.0
  );
  assert!(
    a.compared_slots() > 0,
    "{label}: no (chunk, slot) embedded by BOTH sides — the harness compared nothing"
  );
  assert!(
    a.worst_cos >= EMBED_COS_SANITY_FLOOR,
    "{label}: embedding cosine {:.8} below sanity floor {} — cross-mask/fbank/conversion cannot \
     explain a vector this far off; investigate a slot mismap. Not a bound to loosen.",
    a.worst_cos,
    EMBED_COS_SANITY_FLOOR
  );
}

/// PART A — argmax's two quantization tiers vs the fp32 dia-ort oracle, and the
/// quantization-cost delta between them. Characterization (spec §5 tier 2, spec
/// §9 "argmax accuracy unmeasured"); the hard asserts are the harness
/// invariants and the documented sanity tripwires.
#[test]
#[ignore = "needs Models/argmax-speakerkit (ARGMAX_TEST_MODELS) + committed fp32-dia goldens"]
fn argmax_accuracy_vs_fp32_dia_ort() {
  // The Baseline tier: W32A32 (fp32) segmenter / W16A16 (fp16, NOT fp32)
  // embedder — argmax's un-palettized tier. Comparing the fp16 embedder
  // against dia's fp32 WeSpeaker folds a small precision term into the
  // embedding-cosine gap below; the gap is not purely fbank + conversion.
  let baseline = measure(ArgmaxVariant::Baseline);
  report("Baseline (W32A32 seg / W16A16 embed)", &baseline);

  // The 8-bit-palettized tier.
  let w8a16 = measure(ArgmaxVariant::W8A16);
  report("W8A16", &w8a16);

  // ── The quantization-cost verdict: W8A16 vs the Baseline tier ──
  let seg_delta = baseline.seg_agreement() - w8a16.seg_agreement();
  let cos_delta = baseline.mean_cos - w8a16.mean_cos;
  println!(
    "\n=== QUANTIZATION COST (W8A16 vs Baseline, both vs fp32 dia-ort) ===\n  \
     seg cell-agreement: {:.4}% -> {:.4}% (Δ {:+.4} pp) | full-set frame flips: {} -> {}\n  \
     embed mean cosine: {:.8} -> {:.8} (Δ {:+.8}) | worst cosine: {:.8} -> {:.8} | worst max|diff|: \
     {:.4e} -> {:.4e}",
    baseline.seg_agreement() * 100.0,
    w8a16.seg_agreement() * 100.0,
    -seg_delta * 100.0,
    baseline.frame_flips(),
    w8a16.frame_flips(),
    baseline.mean_cos,
    w8a16.mean_cos,
    -cos_delta,
    baseline.worst_cos,
    w8a16.worst_cos,
    baseline.worst_max_abs,
    w8a16.worst_max_abs,
  );

  // Both tiers must compare the SAME operands, or the delta is meaningless.
  // The segmentation cells are all-or-nothing per compared chunk, and the
  // compared-chunk set is fixed by the shared geometry (identical num_chunks),
  // so an equal cell COUNT is an equal cell set here.
  assert_eq!(
    baseline.seg_cells, w8a16.seg_cells,
    "the two tiers compared different segmentation cell counts — the delta is not apples-to-apples"
  );
  // Embedding slots are NOT geometry-fixed: whether a slot is embedded depends
  // on each tier's own activity/mask decode, which quantization can flip. So
  // assert SET EQUALITY of the compared `(fixture, chunk, slot)` identities,
  // not just their count — Baseline `{A,B}` vs W8A16 `{A,C}` has equal size but
  // a meaningless mean-cosine delta (L2). A `symmetric_difference` diff names
  // exactly which slots diverged if this ever fires.
  assert_eq!(
    baseline.compared_slot_ids,
    w8a16.compared_slot_ids,
    "the two tiers embedded DIFFERENT (fixture, chunk, slot) sets — the delta is not \
     apples-to-apples. Symmetric difference: {:?}",
    baseline
      .compared_slot_ids
      .symmetric_difference(&w8a16.compared_slot_ids)
      .collect::<Vec<_>>()
  );
  assert!(
    cos_delta <= MAX_QUANT_COS_DROP,
    "W8A16 mean cosine is {cos_delta:.8} BELOW Baseline — beyond the {MAX_QUANT_COS_DROP} \
     quantization-cost tripwire. A drop this large points at a broken W8A16 artifact, not ordinary \
     8-bit palettization. Investigate before accepting."
  );
}
