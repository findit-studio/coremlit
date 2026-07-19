use super::*;
use crate::ComputeUnits;

// =====================================================================
// Hermetic: index/range helpers
// =====================================================================

#[test]
fn chunk_segmentation_range_hand_values() {
  // stride = F * SEG_NUM_SLOTS = 4 * 3 = 12.
  assert_eq!(chunk_segmentation_range(0, 4), 0..12);
  assert_eq!(chunk_segmentation_range(2, 4), 24..36);
}

#[test]
fn embedding_range_hand_values() {
  // base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM.
  // (0, 0): (0*3+0)*256 = 0     -> 0..256
  assert_eq!(embedding_range(0, 0), 0..256);
  // (1, 2): (1*3+2)*256 = 5*256 = 1280 -> 1280..1536
  assert_eq!(embedding_range(1, 2), 1280..1536);
}

// =====================================================================
// Hermetic: fill_padded_chunk (owned.rs:469-475 exact shape)
// =====================================================================

#[test]
fn fill_padded_chunk_middle_chunk_full_copy() {
  // A chunk fully inside `samples`: n == SEG_CHUNK_SAMPLES, no zero tail.
  // samples[i] = (i + 1) as f32; start = 5; len = SEG_CHUNK_SAMPLES + 10.
  // end = min(5 + 160_000, 160_010) = 160_005; lo = 5; n = 160_000.
  let samples: Vec<f32> = (0..SEG_CHUNK_SAMPLES + 10)
    .map(|i| (i + 1) as f32)
    .collect();
  let mut padded = vec![0.0f32; SEG_CHUNK_SAMPLES];
  fill_padded_chunk(&mut padded, &samples, 5);
  assert_eq!(padded.len(), SEG_CHUNK_SAMPLES);
  assert_eq!(padded[0], 6.0); // samples[5]
  assert_eq!(
    padded[SEG_CHUNK_SAMPLES - 1],
    (SEG_CHUNK_SAMPLES + 5) as f32
  ); // samples[160_004]
}

#[test]
fn fill_padded_chunk_final_chunk_partial_with_zero_tail() {
  // Final chunk running past the buffer: samples[i] = (i + 1); start = 10;
  // len = SEG_CHUNK_SAMPLES + 5. end = min(160_010, 160_005) = 160_005;
  // lo = 10; n = 159_995. padded[159_995..] stays zero.
  let samples: Vec<f32> = (0..SEG_CHUNK_SAMPLES + 5).map(|i| (i + 1) as f32).collect();
  let mut padded = vec![0.0f32; SEG_CHUNK_SAMPLES];
  fill_padded_chunk(&mut padded, &samples, 10);
  assert_eq!(padded[0], 11.0); // samples[10]
  assert_eq!(padded[159_994], (SEG_CHUNK_SAMPLES + 5) as f32); // samples[160_004]
  assert!(
    padded[159_995..].iter().all(|v| *v == 0.0),
    "out-of-range tail must be zero"
  );
}

#[test]
fn fill_padded_chunk_start_beyond_samples_is_all_zero() {
  // start >= len: lo = len, end = len, n = 0 — no copy, no panic.
  let samples = vec![1.0f32, 2.0, 3.0];
  let mut padded = vec![0.0f32; SEG_CHUNK_SAMPLES];
  fill_padded_chunk(&mut padded, &samples, 2_000);
  assert!(padded.iter().all(|v| *v == 0.0));
}

#[test]
fn fill_padded_chunk_samples_shorter_than_window() {
  // Degenerate: whole (short) clip copied at the head, rest zero.
  let samples: Vec<f32> = (0..500).map(|i| (i + 1) as f32).collect();
  let mut padded = vec![0.0f32; SEG_CHUNK_SAMPLES];
  fill_padded_chunk(&mut padded, &samples, 0);
  assert_eq!(padded[0], 1.0);
  assert_eq!(padded[499], 500.0);
  assert!(padded[500..].iter().all(|v| *v == 0.0));
}

// =====================================================================
// Hermetic: zero_slot_column
// =====================================================================

#[test]
fn zero_slot_column_zeroes_only_the_named_column() {
  // F = 3, S = 3 slab, frame-major [f*3 + s]:
  //   f0: [1,2,3]  f1: [4,5,6]  f2: [7,8,9]
  let mut slab = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
  zero_slot_column(&mut slab, 3, 1);
  assert_eq!(slab, vec![1.0, 0.0, 3.0, 4.0, 0.0, 6.0, 7.0, 0.0, 9.0]);
}

// =====================================================================
// Hermetic: derive_slot_plans — THE critical port (owned.rs:507-591).
//
// Every scenario feeds HAND logits THROUGH `crate::audio::speaker::segment::multilabel`
// (the brief mandates hand logits through multilabel, not hand-written
// slabs) and asserts the FULL [SlotPlan; 3] array. Class table
// (segment/mod.rs:412-420): 0=silence, 1=A, 2=B, 3=C, 4=A+B, 5=A+C,
// 6=B+C. F = 6, onset = 0.5 throughout. Each test's doc comment carries
// the frame-by-frame derivation — the in-test table IS the proof.
// =====================================================================

/// One-hot logits (dominant class = 5.0, rest 0.0) for a frame sequence —
/// fed through `multilabel` so the slab is built exactly as `extract`
/// builds it, not hand-written.
fn logits_for_classes(classes: &[usize]) -> Vec<f32> {
  let mut out =
    Vec::with_capacity(classes.len() * crate::audio::speaker::segment::POWERSET_CLASSES);
  for &c in classes {
    let mut row = [0.0f32; crate::audio::speaker::segment::POWERSET_CLASSES];
    row[c] = 5.0;
    out.extend_from_slice(&row);
  }
  out
}

/// `classes` → one chunk's `[f][s]` multilabel slab.
fn classes_to_slab(classes: &[usize]) -> Vec<f64> {
  crate::audio::speaker::segment::multilabel(&logits_for_classes(classes), classes.len())
}

/// `SlotPlan::Embed` from a fixed 6-frame mask literal.
fn embed6(mask: [bool; 6]) -> SlotPlan {
  SlotPlan::Embed(mask.to_vec())
}

/// s1: classes `[1,1,1,2,2,0]` — no overlap anywhere.
/// | f | class | s0 s1 s2 | active# | clean(<2) |
/// |---|-------|----------|---------|-----------|
/// | 0 | 1 (A) | 1 0 0    | 1       | T |
/// | 1 | 1 (A) | 1 0 0    | 1       | T |
/// | 2 | 1 (A) | 1 0 0    | 1       | T |
/// | 3 | 2 (B) | 0 1 0    | 1       | T |
/// | 4 | 2 (B) | 0 1 0    | 1       | T |
/// | 5 | 0 (-) | 0 0 0    | 0       | T |
/// slot0 active {0,1,2}, clean-active 3 > 2 → clean mask (t,t,t,f,f,f).
/// slot1 active {3,4}, clean-active 2 ≤ 2 → fallback raw (f,f,f,t,t,f).
/// slot2 no active → Skip.
#[test]
fn derive_slot_plans_s1_no_overlap() {
  let slab = classes_to_slab(&[1, 1, 1, 2, 2, 0]);
  assert_eq!(
    derive_slot_plans(&slab, 6, 0.5),
    [
      embed6([true, true, true, false, false, false]),
      embed6([false, false, false, true, true, false]),
      SlotPlan::Skip,
    ]
  );
}

/// s2: classes `[4,4,4,4,4,4]` — A+B every frame (full overlap).
/// Every frame active#=2 → clean=F everywhere. slot0/slot1 clean-active=0
/// ≤ 2 → fallback to raw mask (all true). Breaking the fallback would
/// leave all-false masks, so this test fails under M3 (remove fallback).
/// slot2 no active → Skip.
#[test]
fn derive_slot_plans_s2_full_overlap_falls_back() {
  let slab = classes_to_slab(&[4, 4, 4, 4, 4, 4]);
  assert_eq!(
    derive_slot_plans(&slab, 6, 0.5),
    [
      embed6([true, true, true, true, true, true]),
      embed6([true, true, true, true, true, true]),
      SlotPlan::Skip,
    ]
  );
}

/// s3: classes `[1,1,4,4,0,0]` — the `<=` fallback edge (exactly 2 clean).
/// | f | class | s0 s1 s2 | active# | clean(<2) |
/// |---|-------|----------|---------|-----------|
/// | 0 | 1 (A) | 1 0 0    | 1       | T |
/// | 1 | 1 (A) | 1 0 0    | 1       | T |
/// | 2 | 4 (AB)| 1 1 0    | 2       | F |
/// | 3 | 4 (AB)| 1 1 0    | 2       | F |
/// | 4 | 0 (-) | 0 0 0    | 0       | T |
/// | 5 | 0 (-) | 0 0 0    | 0       | T |
/// slot0 active {0,1,2,3}, clean-active {0,1}=2 ≤ 2 → FALLBACK →
///   (t,t,t,t,f,f). (Mutating `<=` to `<` drops the fallback → clean mask
///   (t,t,f,f,f,f) → this test fails: catches M2.)
/// slot1 active {2,3}, clean-active {}=0 → fallback (f,f,t,t,f,f).
/// slot2 Skip.
#[test]
fn derive_slot_plans_s3_exactly_two_clean_frames_falls_back() {
  let slab = classes_to_slab(&[1, 1, 4, 4, 0, 0]);
  assert_eq!(
    derive_slot_plans(&slab, 6, 0.5),
    [
      embed6([true, true, true, true, false, false]),
      embed6([false, false, true, true, false, false]),
      SlotPlan::Skip,
    ]
  );
}

/// s4: classes `[1,1,4,4,1,0]` — 3 clean frames, uses the CLEAN mask.
/// | f | class | s0 s1 s2 | active# | clean(<2) |
/// |---|-------|----------|---------|-----------|
/// | 0 | 1 (A) | 1 0 0    | 1       | T |
/// | 1 | 1 (A) | 1 0 0    | 1       | T |
/// | 2 | 4 (AB)| 1 1 0    | 2       | F |
/// | 3 | 4 (AB)| 1 1 0    | 2       | F |
/// | 4 | 1 (A) | 1 0 0    | 1       | T |
/// | 5 | 0 (-) | 0 0 0    | 0       | T |
/// slot0 active {0,1,2,3,4}, clean-active {0,1,4}=3 > 2 → CLEAN mask
///   (t,t,f,f,t,f) — DIFFERENT from the raw active mask (t,t,t,t,t,f), so
///   this pins that the exclusion actually excludes. (Mutating clean-def
///   `< 2` to `<= 2` marks f2,f3 clean → (t,t,t,t,t,f) → fails: catches M1.)
/// slot1 active {2,3}, clean-active {}=0 → fallback (f,f,t,t,f,f).
/// slot2 Skip.
#[test]
fn derive_slot_plans_s4_three_clean_frames_uses_clean_mask() {
  let slab = classes_to_slab(&[1, 1, 4, 4, 1, 0]);
  assert_eq!(
    derive_slot_plans(&slab, 6, 0.5),
    [
      embed6([true, true, false, false, true, false]),
      embed6([false, false, true, true, false, false]),
      SlotPlan::Skip,
    ]
  );
}

/// s5: classes `[1,1,1,4,4,0]` — fallback is PER-SLOT, not whole-chunk.
/// | f | class | s0 s1 s2 | active# | clean(<2) |
/// |---|-------|----------|---------|-----------|
/// | 0 | 1 (A) | 1 0 0    | 1       | T |
/// | 1 | 1 (A) | 1 0 0    | 1       | T |
/// | 2 | 1 (A) | 1 0 0    | 1       | T |
/// | 3 | 4 (AB)| 1 1 0    | 2       | F |
/// | 4 | 4 (AB)| 1 1 0    | 2       | F |
/// | 5 | 0 (-) | 0 0 0    | 0       | T |
/// slot0 active {0,1,2,3,4}, clean-active {0,1,2}=3 > 2 → CLEAN branch
///   (t,t,t,f,f,f), WHILE slot1 active {3,4}, clean-active {}=0 → FALLBACK
///   (f,f,f,t,t,f). One slot takes the clean branch and another falls
///   back in the SAME chunk — impossible under a whole-chunk fallback.
/// slot2 Skip.
#[test]
fn derive_slot_plans_s5_fallback_is_per_slot_not_whole_chunk() {
  let slab = classes_to_slab(&[1, 1, 1, 4, 4, 0]);
  assert_eq!(
    derive_slot_plans(&slab, 6, 0.5),
    [
      embed6([true, true, true, false, false, false]),
      embed6([false, false, false, true, true, false]),
      SlotPlan::Skip,
    ]
  );
}

/// s6: classes `[1,1,0,0,0,0]` — single speaker, slot0 through the
/// fallback branch (clean_count 2 ≤ 2), same values either way.
/// slot0 (t,t,f,f,f,f); slot1/slot2 Skip.
#[test]
fn derive_slot_plans_s6_single_speaker() {
  let slab = classes_to_slab(&[1, 1, 0, 0, 0, 0]);
  assert_eq!(
    derive_slot_plans(&slab, 6, 0.5),
    [
      embed6([true, true, false, false, false, false]),
      SlotPlan::Skip,
      SlotPlan::Skip,
    ]
  );
}

/// s7: classes `[0,0,0,0,0,0]` — all silence, every slot Skip.
#[test]
fn derive_slot_plans_s7_empty_chunk_all_skip() {
  let slab = classes_to_slab(&[0, 0, 0, 0, 0, 0]);
  assert_eq!(
    derive_slot_plans(&slab, 6, 0.5),
    [SlotPlan::Skip, SlotPlan::Skip, SlotPlan::Skip]
  );
}

#[test]
#[should_panic(expected = "chunk_segs.len() must equal num_frames * SEG_NUM_SLOTS")]
fn derive_slot_plans_panics_on_length_mismatch() {
  // len 5 != 2 * 3 = 6.
  let _ = derive_slot_plans(&[0.0f64; 5], 2, 0.5);
}

// =====================================================================
// Hermetic: geometry pipeline — concatenate per-chunk multilabel slabs
// via chunk_segmentation_range, then count via the window fns, at a small
// synthetic geometry with a hand-derived expected count.
// =====================================================================

/// 3 chunks, F = 4, S = SEG_NUM_SLOTS = 3, onset 0.5, chunks_sw = (0,4,2),
/// frames_sw = (0,1,1). Classes → active-speaker count per (chunk, frame):
/// - c0 `[1,4,0,2]` → [1,2,0,1]
/// - c1 `[1,1,6,0]` → [1,1,2,0]
/// - c2 `[0,5,3,1]` → [0,2,1,1]
///
/// start_frame(c) = round_ties_even(c*2/1) = 0, 2, 4.
/// num_output_frames = round_ties_even((4 + 2*2)/1) + 1 = 9.
/// Aggregate (sum ÷ covering count), round_ties_even, 0 where uncovered:
/// - t0 (1,1)→1  t1 (2,1)→2  t2 (0+1,2)=0.5→0  t3 (1+1,2)=1→1
/// - t4 (2+0,2)=1→1  t5 (0+2,2)=1→1  t6 (1,1)→1  t7 (1,1)→1  t8 (0,0)→0
///
/// Result: count = [1, 2, 0, 1, 1, 1, 1, 1, 0]. t2 exercises
/// round_ties_even's 0.5 → 0 tie.
#[test]
fn geometry_pipeline_three_chunks_hand_derived_count() {
  let num_chunks = 3;
  let num_frames = 4;
  let mut segmentations = vec![0.0f64; num_chunks * num_frames * SEG_NUM_SLOTS];
  let chunk_classes = [[1, 4, 0, 2], [1, 1, 6, 0], [0, 5, 3, 1]];
  for (c, classes) in chunk_classes.iter().enumerate() {
    let slab = classes_to_slab(classes);
    segmentations[chunk_segmentation_range(c, num_frames)].copy_from_slice(&slab);
  }

  let count = crate::audio::speaker::window::count_from_segmentations(
    &segmentations,
    num_chunks,
    num_frames,
    SEG_NUM_SLOTS,
    0.5,
    SlidingWindow::new(0.0, 4.0, 2.0),
    SlidingWindow::new(0.0, 1.0, 1.0),
  );
  assert_eq!(count, vec![1, 2, 0, 1, 1, 1, 1, 1, 0]);
  assert_eq!(count.len(), 9); // num_output_frames
}

// =====================================================================
// Hermetic: ComputeOptions / Options (rust-options-pattern)
// =====================================================================

#[test]
fn compute_options_new_matches_default() {
  assert_eq!(ComputeOptions::new(), ComputeOptions::default());
}

#[test]
fn compute_options_defaults_match_crate_consts() {
  let o = ComputeOptions::new();
  assert_eq!(
    o.segmenter(),
    crate::audio::speaker::segment::DEFAULT_SEGMENT_COMPUTE
  );
  assert_eq!(
    o.embedder(),
    crate::audio::speaker::embed::DEFAULT_EMBED_COMPUTE
  );
  // Both are ComputeUnits::All today; pin that too.
  assert_eq!(o.segmenter(), ComputeUnits::All);
  assert_eq!(o.embedder(), ComputeUnits::All);
}

#[test]
fn compute_options_builders_and_setters() {
  let o = ComputeOptions::new()
    .with_segmenter(ComputeUnits::CpuOnly)
    .with_embedder(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(o.segmenter(), ComputeUnits::CpuOnly);
  assert_eq!(o.embedder(), ComputeUnits::CpuAndNeuralEngine);

  let mut m = ComputeOptions::new();
  m.set_segmenter(ComputeUnits::CpuAndGpu);
  m.set_embedder(ComputeUnits::CpuOnly);
  assert_eq!(m.segmenter(), ComputeUnits::CpuAndGpu);
  assert_eq!(m.embedder(), ComputeUnits::CpuOnly);
}

#[test]
fn options_new_matches_default() {
  assert_eq!(Options::new(), Options::default());
}

#[test]
fn options_defaults_delegate_to_components() {
  let o = Options::new();
  assert_eq!(o.window(), WindowOptions::new());
  assert_eq!(o.compute(), ComputeOptions::new());
  assert_eq!(o.source(), Source::default());
  // Pin the concrete default too, matching the sibling `ComputeUnits::All`
  // pin just below.
  assert_eq!(o.source(), Source::FluidAudio);
}

#[test]
fn options_builders_and_setters() {
  let window = WindowOptions::new().with_onset(0.25);
  let compute = ComputeOptions::new().with_segmenter(ComputeUnits::CpuOnly);
  let source = Source::Argmax;
  let o = Options::new()
    .with_window(window)
    .with_compute(compute)
    .with_source(source);
  assert_eq!(o.window(), window);
  assert_eq!(o.compute(), compute);
  assert_eq!(o.source(), source);

  let mut m = Options::new();
  m.set_window(window);
  m.set_compute(compute);
  m.set_source(source);
  assert_eq!(m.window(), window);
  assert_eq!(m.compute(), compute);
  assert_eq!(m.source(), source);
}

// =====================================================================
// Hermetic: Extractor surface
// =====================================================================

#[test]
fn extractor_new_matches_default_and_holds_default_options() {
  assert_eq!(Extractor::new(), Extractor::default());
  assert_eq!(*Extractor::new().options_ref(), Options::new());
}

#[test]
fn extractor_with_options_round_trips() {
  let options = Options::new().with_window(WindowOptions::new().with_step_samples(40_000));
  let extractor = Extractor::with_options(options);
  assert_eq!(*extractor.options_ref(), options);
}

// =====================================================================
// Hermetic: serde (mirrors window/tests.rs:153-177 style)
// =====================================================================

#[cfg(feature = "serde")]
#[test]
fn options_serde_empty_object_is_full_defaults() {
  let o: Options = serde_json::from_str("{}").unwrap();
  assert_eq!(o, Options::new());
  assert_eq!(o.source(), Source::FluidAudio);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_partial_window_keeps_step_default() {
  // Only window.onset is given: window.step_samples defaults (via
  // WindowOptions' own per-field default), and compute/source default
  // whole.
  let o: Options = serde_json::from_str(r#"{"window":{"onset":0.25}}"#).unwrap();
  assert_eq!(o.window().onset(), 0.25);
  assert_eq!(o.window().step_samples(), 16_000);
  assert_eq!(o.compute(), ComputeOptions::new());
  assert_eq!(o.source(), Source::default());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_partial_compute_defaults_other_unit() {
  // Only compute.segmenter is given: compute.embedder defaults (via
  // ComputeOptions' own per-field default), window/source default whole.
  let o: Options = serde_json::from_str(r#"{"compute":{"segmenter":"cpu_only"}}"#).unwrap();
  assert_eq!(o.compute().segmenter(), ComputeUnits::CpuOnly);
  assert_eq!(o.compute().embedder(), ComputeUnits::All);
  assert_eq!(o.window(), WindowOptions::new());
  assert_eq!(o.source(), Source::default());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_partial_source_defaults_others() {
  // Only source is given: window/compute default whole. Mirrors the two
  // sibling partial-input tests just above, for the new field.
  let o: Options = serde_json::from_str(r#"{"source":"argmax"}"#).unwrap();
  assert_eq!(o.source(), Source::Argmax);
  assert_eq!(o.window(), WindowOptions::new());
  assert_eq!(o.compute(), ComputeOptions::new());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips() {
  let o = Options::new()
    .with_window(
      WindowOptions::new()
        .with_step_samples(40_000)
        .with_onset(0.7),
    )
    .with_compute(ComputeOptions::new().with_segmenter(ComputeUnits::CpuOnly))
    .with_source(Source::Argmax);
  let json = serde_json::to_string(&o).unwrap();
  let back: Options = serde_json::from_str(&json).unwrap();
  assert_eq!(back, o);
}

// =====================================================================
// into_offline_input — the compile/borrow proof AND the field round-trip.
// plda is hermetic (compile-time-embedded weights, transform.rs:341-379),
// so this needs no model. `diaric` rides the `speaker` feature that gates
// the whole module, so this runs in the ordinary unit suite (no oracle).
// =====================================================================

#[test]
fn into_offline_input_round_trips_against_real_dia() {
  // A small, self-consistent Extraction (num_chunks=1, F=2, count len ==
  // num_output_frames=4). Fields are private, but this child test module
  // sees them.
  let e = Extraction {
    raw_embeddings: (0..(SEG_NUM_SLOTS * EMBEDDING_DIM))
      .map(|i| i as f32 * 0.25 - 3.0)
      .collect(),
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    count: vec![1, 2, 1, 0],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    num_output_frames: 4,
    chunks_sw: crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new()),
    frames_sw: crate::audio::speaker::window::frame_sliding_window(),
  };

  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let input = e.into_offline_input(&plda);

  assert_eq!(input.raw_embeddings(), e.raw_embeddings());
  assert_eq!(input.num_chunks(), e.num_chunks());
  assert_eq!(input.num_speakers(), 3);
  assert_eq!(input.num_speakers(), e.num_speakers());
  assert_eq!(input.segmentations(), e.segmentations());
  assert_eq!(input.num_frames_per_chunk(), e.num_frames_per_chunk());
  assert_eq!(input.count(), e.count());
  assert_eq!(input.num_output_frames(), e.num_output_frames());

  // SlidingWindow fields, compared through the public accessors on both
  // sides (diaric's OfflineInput returns diaric's SlidingWindow by value).
  let cs = input.chunks_sw();
  assert_eq!(cs.start(), e.chunks_sw().start());
  assert_eq!(cs.duration(), e.chunks_sw().duration());
  assert_eq!(cs.step(), e.chunks_sw().step());
  let fs = input.frames_sw();
  assert_eq!(fs.start(), e.frames_sw().start());
  assert_eq!(fs.duration(), e.frames_sw().duration());
  assert_eq!(fs.step(), e.frames_sw().step());

  // The borrowed plda is the very same one we passed in.
  assert!(std::ptr::eq(input.plda(), &plda));
}

// =====================================================================
// diarize() — the public runtime clustering entry point. Hermetic proof
// that it is ONE code path with the manual `into_offline_input →
// diarize_offline` plumbing the parity harness used to inline (the alignkit
// canonical-wiring lesson): SAME Extraction, SAME PLDA ⇒ byte-identical
// Result. plda is hermetic (compile-time-embedded weights), so this needs
// no model and runs ort-free in the ordinary unit suite. The model-gated
// ≥3-speaker regime — where the clustering decision is non-trivial — is
// proven in `tests/parity_diarize_wiring.rs`.
// =====================================================================

#[test]
fn diarize_matches_manual_into_offline_input_pipeline() {
  // The same small, self-consistent Extraction as the round-trip test above.
  let e = Extraction {
    raw_embeddings: (0..(SEG_NUM_SLOTS * EMBEDDING_DIM))
      .map(|i| i as f32 * 0.25 - 3.0)
      .collect(),
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    count: vec![1, 2, 1, 0],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    num_output_frames: 4,
    chunks_sw: crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new()),
    frames_sw: crate::audio::speaker::window::frame_sliding_window(),
  };
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");

  // Subject: the public runtime method.
  let via_public = e.diarize(&plda);
  // Reference: the pre-refactor plumbing, reconstructed through the still-
  // public `into_offline_input` bridge.
  let via_manual = diaric::offline::diarize_offline(&e.into_offline_input(&plda));

  // The two must agree on their WHOLE Result — succeed identically, or refuse
  // identically. `OfflineOutput` is not `PartialEq`, so compare the observable
  // span geometry on success and the typed error's rendering on failure. A
  // mutation to `diarize`'s wiring (dropped PLDA, swapped option, wrong tensor)
  // breaks exactly one arm and this assertion fires.
  match (via_public, via_manual) {
    (Ok(pub_out), Ok(man_out)) => {
      let spans = |o: &diaric::offline::OfflineOutput| -> Vec<(f64, f64, usize)> {
        o.spans_slice()
          .iter()
          .map(|s| (s.start(), s.end(), s.cluster()))
          .collect()
      };
      assert_eq!(
        spans(&pub_out),
        spans(&man_out),
        "diarize() spans diverged from into_offline_input → diarize_offline"
      );
    }
    (Err(pub_err), Err(man_err)) => {
      assert_eq!(
        format!("{pub_err:?}"),
        format!("{man_err:?}"),
        "diarize() and the manual plumbing refused differently"
      );
    }
    (pub_res, man_res) => panic!(
      "diarize() ({}) diverged from manual into_offline_input → diarize_offline ({})",
      if pub_res.is_ok() { "Ok" } else { "Err" },
      if man_res.is_ok() { "Ok" } else { "Err" },
    ),
  }
}

// =====================================================================
// diarize_with — the ClusterBackend wiring (T2). Hermetic: no models,
// ort-free. Proves a NON-default backend's OfflineOptions actually flow through
// diarize_with (they are not silently ignored in favour of the default). The
// DEFAULT path is already covered by
// `diarize_matches_manual_into_offline_input_pipeline` above — diarize ==
// diarize_with(default) == the bare bridge — and the knob→dia-field mapping by
// `cluster::tests::apply_to_maps_each_knob_to_its_dia_field`.
// =====================================================================

/// A small, self-consistent [`Extraction`] (num_chunks=1, F=2, count len ==
/// num_output_frames=4) — the same shape the round-trip / diarize tests above
/// build inline. Private fields are visible to this child module.
fn tiny_extraction() -> Extraction {
  Extraction {
    raw_embeddings: (0..(SEG_NUM_SLOTS * EMBEDDING_DIM))
      .map(|i| i as f32 * 0.25 - 3.0)
      .collect(),
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    count: vec![1, 2, 1, 0],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    num_output_frames: 4,
    chunks_sw: crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new()),
    frames_sw: crate::audio::speaker::window::frame_sliding_window(),
  }
}

#[test]
fn diarize_with_offline_routes_the_backend_options() {
  // A NON-default Offline backend must produce exactly
  // diarize_offline(opts.apply_to(into_offline_input)) — i.e. diarize_with
  // threads the variant's OfflineOptions, not ClusterBackend::default()'s. A
  // mutation that ignored `backend` (always using the default) would break this
  // (the non-default knobs would not reach diaric).
  let e = tiny_extraction();
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let opts = crate::audio::speaker::cluster::OfflineOptions::new()
    .with_threshold(0.55)
    .with_fa(0.09)
    .with_fb(0.71)
    .with_max_iters(33)
    .with_min_duration_off(1.25);

  // Subject: the public runtime method with a selected non-default backend.
  let via_public = e.diarize_with(&plda, ClusterBackend::Offline(opts));
  // Reference: the same OfflineOptions applied by hand over the bare bridge.
  let via_manual = diaric::offline::diarize_offline(&opts.apply_to(e.into_offline_input(&plda)));

  // OfflineOutput is not PartialEq: compare span geometry on success, the typed
  // error's rendering on failure — same shape as the diarize test above.
  match (via_public, via_manual) {
    (Ok(pub_out), Ok(man_out)) => {
      let spans = |o: &diaric::offline::OfflineOutput| -> Vec<(f64, f64, usize)> {
        o.spans_slice()
          .iter()
          .map(|s| (s.start(), s.end(), s.cluster()))
          .collect()
      };
      assert_eq!(
        spans(&pub_out),
        spans(&man_out),
        "diarize_with routed a different OfflineInput than apply_to"
      );
    }
    (Err(pub_err), Err(man_err)) => {
      assert_eq!(
        format!("{pub_err:?}"),
        format!("{man_err:?}"),
        "diarize_with and the apply_to path refused differently"
      );
    }
    (p, m) => panic!(
      "diarize_with ({}) diverged from the apply_to path ({})",
      if p.is_ok() { "Ok" } else { "Err" },
      if m.is_ok() { "Ok" } else { "Err" },
    ),
  }
}

// =====================================================================
// diarize_online — the ONLINE engine wiring (T5). Hermetic: no models, ort-free,
// NO plda. Proves the full online plumbing (feed order → per-slot labelling →
// the SAME reconstruction the offline path uses): a purpose-built 2-chunk
// extraction with orthogonal one-hot-block embeddings makes every assignment
// predictable, so the exact hard_clusters can be pinned. The clusterer's own
// decision logic is separately gated by dia's mutation-proven unit tests and the
// Swift-trace oracle (`tests/parity_online_swift.rs`).
// =====================================================================

/// A 2-chunk extraction whose six slots are orthogonal one-hot 64-dim blocks
/// (near-antipodal in cosine space) except the zeroed `(chunk0, slot2)`:
///
/// | slot        | block   | outcome (min_speech_duration = 0) |
/// |-------------|---------|-----------------------------------|
/// | c0 s0       | 0 (A)   | New speaker 1 → cluster 0          |
/// | c0 s1       | 1 (B)   | New speaker 2 → cluster 1          |
/// | c0 s2       | (zero)  | dropped (normalize_from None) → -2 |
/// | c1 s0       | 0 (A)   | Existing speaker 1 → cluster 0     |
/// | c1 s1       | 0 (A)   | Existing speaker 1 → cluster 0     |
/// | c1 s2       | 2 (C)   | New speaker 3 → cluster 2          |
///
/// So `hard_clusters == [[0, 1, -2], [0, 0, 2]]`, `num_clusters == 3`. Timing is
/// the community-1 default (chunks_sw step 1 s, frames_sw step 0.016875 s): with
/// F = 4, chunk 1 lands at output frames 59..63, so `num_output_frames = 63`.
fn online_extraction() -> Extraction {
  const F: usize = 4;
  let seg_idx = |c: usize, f: usize, s: usize| (c * F + f) * SEG_NUM_SLOTS + s;
  let mut segmentations = vec![0.0f64; 2 * F * SEG_NUM_SLOTS];
  // Activity per surviving slot (nonzero frames → the online speech duration);
  // the exact counts do not matter with min_speech_duration = 0, only that the
  // dropped slot's column stays zero.
  for f in 0..2 {
    segmentations[seg_idx(0, f, 0)] = 1.0; // c0 s0
  }
  for f in 2..4 {
    segmentations[seg_idx(0, f, 1)] = 1.0; // c0 s1
  }
  // c0 s2: no active frame (dropped)
  for f in 0..4 {
    segmentations[seg_idx(1, f, 0)] = 1.0; // c1 s0
  }
  for f in 0..2 {
    segmentations[seg_idx(1, f, 1)] = 1.0; // c1 s1
  }
  for f in 2..4 {
    segmentations[seg_idx(1, f, 2)] = 1.0; // c1 s2
  }

  let mut raw_embeddings = vec![0.0f32; 2 * SEG_NUM_SLOTS * EMBEDDING_DIM];
  let mut set_block = |c: usize, s: usize, block: usize| {
    let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
    raw_embeddings[(base + block * 64)..(base + (block + 1) * 64)].fill(1.0);
  };
  set_block(0, 0, 0); // A
  set_block(0, 1, 1); // B
  // c0 s2 left zero → dropped by Embedding::normalize_from
  set_block(1, 0, 0); // A (reuse)
  set_block(1, 1, 0); // A (reuse)
  set_block(1, 2, 2); // C

  // count[t]: 2 active clusters over each chunk's frames, 0 elsewhere. Valid
  // (<= MAX_COUNT_PER_FRAME) and length == num_output_frames. NOTE: `diarize_online`
  // no longer consumes this field (it derives its own clustered-segmentation count);
  // it is retained as a valid `Extraction::count` (the offline path's contract).
  let mut count = vec![0u8; 63];
  count[0..4].fill(2);
  count[59..63].fill(2);

  // Chunk window sized to this fixture's 63-frame output grid: duration =
  // (F-1)·frame_step. Same rationale as `online_extraction_default_gate`'s chunk
  // window — `reconstruct` ignores chunk DURATION (start/step, unchanged here, place
  // the two chunks at output frames 0 and 59), but `diarize_online`'s own
  // `try_count_from_segmentations` derives `num_output_frames` from it, so the
  // nominal 10 s duration would make the derived count 653-long and mismatch this
  // 63-frame grid.
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new())
    .with_duration((F as f64 - 1.0) * crate::audio::speaker::window::FRAME_STEP_S);

  Extraction {
    raw_embeddings,
    segmentations,
    count,
    num_chunks: 2,
    num_frames_per_chunk: F,
    num_output_frames: 63,
    chunks_sw,
    frames_sw: crate::audio::speaker::window::frame_sliding_window(),
  }
}

#[test]
fn diarize_online_labels_slots_and_reconstructs_spans() {
  let e = online_extraction();
  // min_speech_duration = 0 isolates the plumbing from the duration gate: every
  // slot with a real embedding forms or joins a speaker; the drop path here is
  // exactly the zero-embedding slot.
  let opts = OnlineOptions::new().with_min_speech_duration(0.0);

  let out = e
    .diarize_online(opts)
    .expect("online reconstruction succeeds on a valid extraction");

  // The engine's per-slot assignment, mapped to 0-based cluster ids, with the
  // dropped (chunk0, slot2) as UNMATCHED (-2). This is THE wiring assertion: a
  // wrong feed order, a mis-mapped id, or a skipped/duplicated slot breaks it.
  assert_eq!(
    out.hard_clusters_slice(),
    &[[0, 1, -2], [0, 0, 2]],
    "online per-slot labels (chunk order, slot order) diverged"
  );
  assert_eq!(out.num_clusters(), 3);

  // The SAME reconstruction the offline path uses actually ran: it produced
  // spans, and every span names one of the three online clusters.
  let spans = out.spans_slice();
  assert!(!spans.is_empty(), "reconstruction produced no spans");
  assert!(
    spans.iter().all(|s| s.cluster() < 3),
    "a span named a cluster outside the online roster: {:?}",
    spans.iter().map(|s| s.cluster()).collect::<Vec<_>>()
  );
}

#[test]
fn diarize_with_online_routes_to_diarize_online_ignoring_plda() {
  // diarize_with(_, Online(opts)) MUST equal diarize_online(opts): same engine,
  // same labels, and the plda is unused (a mutation routing Online through the
  // offline PLDA path, or forwarding plda into a different engine, would diverge
  // — offline clustering of these embeddings is not the online greedy result).
  let e = online_extraction();
  let opts = OnlineOptions::new().with_min_speech_duration(0.0);
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");

  let via_online = e.diarize_online(opts).expect("diarize_online ok");
  let via_with = e
    .diarize_with(&plda, ClusterBackend::Online(opts))
    .expect("diarize_with(Online) ok");

  assert_eq!(
    via_online.hard_clusters_slice(),
    via_with.hard_clusters_slice(),
    "diarize_with(Online) routed to a different labelling than diarize_online"
  );
  assert_eq!(via_online.num_clusters(), via_with.num_clusters());
  let spans = |o: &diaric::offline::OfflineOutput| -> Vec<(f64, f64, usize)> {
    o.spans_slice()
      .iter()
      .map(|s| (s.start(), s.end(), s.cluster()))
      .collect()
  };
  assert_eq!(spans(&via_online), spans(&via_with));
}

#[test]
fn diarize_online_default_options_drops_subsecond_slots() {
  // With the DEFAULT min_speech_duration (1.0 s) and community-1 timing, every
  // slot here is far under a second of activity (≤ 4 frames × 0.016875 s ≈
  // 0.068 s) and none matches an existing speaker first, so all are dropped:
  // hard_clusters is all-UNMATCHED and reconstruction yields an empty diarization.
  // This exercises the default duration gate the plumbing test above bypasses.
  let e = online_extraction();
  let out = e
    .diarize_online(OnlineOptions::default())
    .expect("online reconstruction succeeds even with all slots dropped");
  assert_eq!(
    out.hard_clusters_slice(),
    &[[-2, -2, -2], [-2, -2, -2]],
    "default min_speech_duration should drop every sub-second slot"
  );
  assert!(
    out.spans_slice().is_empty(),
    "all-dropped extraction must produce no spans"
  );
}

/// A 1-chunk extraction that exercises the DEFAULT online duration gate
/// (`min_speech_duration = 1.0 s`) with BOTH above- and sub-threshold activity, so
/// the production duration bridge (`speech_duration = active_frame_count ×
/// frames_sw.step`, `extract/mod.rs`) is LOAD-BEARING. With `F = 64` frames and the
/// community-1 frame step `0.016875 s`, a fully-active slot is `64 × 0.016875 =
/// 1.08 s ≥ 1.0` (above the gate), while a 20-frame slot is `0.3375 s < 1.0` (below
/// it). Each surviving slot's embedding is an orthogonal one-hot 64-dim block
/// (near-antipodal in cosine space), so a sub-threshold slot sits far from every
/// existing centroid and therefore reaches the duration gate (Dropped) rather than
/// matching an existing speaker:
///
/// | slot  | block | active frames | duration | outcome (default gate)              |
/// |-------|-------|---------------|----------|-------------------------------------|
/// | c0 s0 | 0 (A) | 64            | 1.08 s   | New speaker 1 → cluster 0           |
/// | c0 s1 | 1 (B) | 20            | 0.3375 s | Dropped (< 1.0 s, orthogonal) → -2  |
/// | c0 s2 | 2 (C) | 64            | 1.08 s   | New speaker 2 → cluster 1           |
///
/// So `hard_clusters == [[0, -2, 1]]`, `num_clusters == 2`. Timing is community-1
/// (frames_sw step `0.016875 s`); chunk 0 lands at output frame 0, so
/// `num_output_frames = F = 64` (the tight fit reconstruct requires). Under the
/// BROKEN bridge (`speech_duration = 0.0`) every slot is `0 < 1.0`, no speaker is
/// ever seeded, and every candidate drops → `[[-2, -2, -2]]` with an empty
/// diarization.
fn online_extraction_default_gate() -> Extraction {
  const F: usize = 64;
  const ABOVE: usize = 64; // active frames; 1.08 s ≥ the 1.0 s gate
  const BELOW: usize = 20; // active frames; 0.3375 s < the 1.0 s gate
  let seg_idx = |c: usize, f: usize, s: usize| (c * F + f) * SEG_NUM_SLOTS + s;
  let mut segmentations = vec![0.0f64; F * SEG_NUM_SLOTS];
  for f in 0..ABOVE {
    segmentations[seg_idx(0, f, 0)] = 1.0; // s0: above threshold
  }
  for f in 0..BELOW {
    segmentations[seg_idx(0, f, 1)] = 1.0; // s1: below threshold
  }
  for f in 0..ABOVE {
    segmentations[seg_idx(0, f, 2)] = 1.0; // s2: above threshold
  }

  let mut raw_embeddings = vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM];
  let mut set_block = |s: usize, block: usize| {
    let base = s * EMBEDDING_DIM;
    raw_embeddings[(base + block * 64)..(base + (block + 1) * 64)].fill(1.0);
  };
  set_block(0, 0); // A
  set_block(1, 1); // B (orthogonal to A)
  set_block(2, 2); // C (orthogonal to A and B)

  // Chunk window sized to this fixture's F-frame output grid: duration =
  // (F-1)·frame_step. The community-1 `chunk_sliding_window` nominally spans 10 s
  // (~594 output frames), but this fixture emits only F output frames, so its chunk
  // duration must match for the per-output-frame count to be self-consistent.
  // `reconstruct` ignores chunk DURATION (only start/step place chunks), so this
  // leaves chunk placement and every span below unchanged; but
  // `try_count_from_segmentations` derives `num_output_frames` FROM the duration, so
  // a 10 s duration would make the count 594-long and mismatch this grid.
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new())
    .with_duration((F as f64 - 1.0) * crate::audio::speaker::window::FRAME_STEP_S);
  let frames_sw = crate::audio::speaker::window::frame_sliding_window();

  // HONEST, segmentation-derived count (dia's `count_from_segmentations`): three
  // active slots (s0,s1,s2) for frames `0..BELOW` and two (s0,s2) for `BELOW..F`,
  // i.e. `[3; 20] ++ [2; 44]`. It counts the DROPPED slot s1 as a speaker while s1
  // is active (frames 0..20). This is the count the production pipeline would hand
  // `diarize_online`; the fix REQUIRES `diarize_online` to IGNORE it and derive its
  // OWN clustered-segmentation count (2 speakers), emitting NO phantom third. Under
  // the OLD code (which fed `self.count` straight to reconstruct) the 3 inflated
  // `num_clusters` to 3 and produced a zero-activation phantom span — exactly the
  // bug this fixture now proves.
  let count = crate::audio::speaker::window::try_count_from_segmentations(
    &segmentations,
    1,
    F,
    SEG_NUM_SLOTS,
    0.5,
    chunks_sw,
    frames_sw,
  )
  .expect("fixture chunk/frame geometry yields exactly F output frames");

  Extraction {
    raw_embeddings,
    segmentations,
    count,
    num_chunks: 1,
    num_frames_per_chunk: F,
    num_output_frames: F,
    chunks_sw,
    frames_sw,
  }
}

#[test]
fn diarize_online_default_gate_keeps_above_threshold_drops_below() {
  // End-to-end proof that the production duration bridge is exercised (codex M2b):
  // with the DEFAULT gate (1.0 s), the two 64-frame slots (1.08 s) MUST seed
  // speakers and the 20-frame orthogonal slot (0.3375 s) MUST drop. The fence's
  // production mutation `speech_duration = 0.0` makes every candidate sub-threshold
  // — no speaker is ever seeded — collapsing hard_clusters to all-UNMATCHED and the
  // diarization to empty, which fails every assertion here. (The sibling
  // `..._default_options_drops_subsecond_slots` test above, all sub-threshold, stays
  // green under that mutation — this test is what turns it red.)
  let e = online_extraction_default_gate();
  let out = e
    .diarize_online(OnlineOptions::default())
    .expect("online reconstruction succeeds on the default-gate fixture");

  // Exact per-slot labels: above-threshold slots seed clusters 0 and 1 (feed order
  // c0 s0 then s2); the sub-threshold orthogonal slot is dropped.
  assert_eq!(
    out.hard_clusters_slice(),
    &[[0, -2, 1]],
    "default-gate labels: above-threshold slots create speakers, the sub-second slot drops"
  );
  assert_eq!(out.num_clusters(), 2, "two above-threshold speakers");

  // Exact span geometry: both surviving clusters are active over the whole output
  // grid, so each yields ONE span. `try_discrete_to_spans` closes an
  // active-through-end region at `start = start + i_start·step + duration/2` and
  // `end = start + (N-1)·step + duration/2` (i_start = 0 here). Recomputing via the
  // SAME formula off `frames_sw` keeps the assertion bit-exact without magic floats.
  let fs = e.frames_sw();
  let center_offset = fs.duration() / 2.0;
  let n = e.num_output_frames() as f64;
  let span_start = fs.start() + center_offset; // i_start = 0
  let span_end = fs.start() + (n - 1.0) * fs.step() + center_offset;
  let span_dur = span_end - span_start;
  let got: Vec<(usize, f64, f64)> = out
    .spans_slice()
    .iter()
    .map(|s| (s.cluster(), s.start(), s.duration()))
    .collect();
  assert_eq!(
    got,
    vec![(0, span_start, span_dur), (1, span_start, span_dur)],
    "default-gate spans: exactly clusters 0 and 1, each spanning the full output grid"
  );
}

// =====================================================================
// diarize_online — HIGH-CHURN allocation fence (codex R5). The M1 online-count
// fix used to build a dense `num_chunks × num_frames_per_chunk ×
// num_clusters_from_hard` f64 buffer; `num_clusters_from_hard` scales with the
// TOTAL distinct global-cluster count, so a long/permissive recording drove an
// unchecked ~GiB allocation BEFORE diaric's cluster/grid caps could fire — a
// reachable process-OOM. The fix computes the per-(chunk,frame) DISTINCT-cluster
// count directly (O(chunks×frames×slots), no cluster axis), so these prove (a)
// many clusters reconstruct correctly with NO cluster-proportional allocation,
// and (b) an over-cap grid is diaric's TYPED reconstruct error, never an OOM.
// =====================================================================

/// A high-churn online extraction that seeds `num_clusters` DISTINCT global
/// speakers: each active `(chunk, slot)` carries a mutually-near-antipodal
/// one-hot embedding (`+e_g` for `g < EMBEDDING_DIM`, `-e_{g-EMBEDDING_DIM}`
/// after), so every pairwise cosine distance is `>= 1.0` — comfortably past the
/// `0.65` `speaker_threshold` — and the greedy online clusterer spawns a NEW
/// speaker for every one (`Assignment::New`), never matching an existing
/// centroid. Feed order is chunk-major then slot-major (`g = c*SEG_NUM_SLOTS +
/// s`), so slot `g` seeds speaker `g+1` → 0-based label `g`; any slot past
/// `num_clusters` (the tail of the last chunk) is left zero and is dropped by
/// `Embedding::normalize_from` (UNMATCHED). `{±e_i}` gives at most
/// `2 * EMBEDDING_DIM` (512) distinct far vectors.
///
/// Each contributing slot is active across all `num_frames_per_chunk` frames, so
/// it forms a real cluster with a span and the distinct-cluster count sees it.
/// `min_speech_duration` must be `0.0` at the call site to keep the duration
/// gate out of the picture.
fn many_cluster_online_extraction(num_clusters: usize, num_frames_per_chunk: usize) -> Extraction {
  assert!(
    num_clusters <= 2 * EMBEDDING_DIM,
    "`{{±e_i}}` yields at most 2*EMBEDDING_DIM ({}) distinct far vectors",
    2 * EMBEDDING_DIM
  );
  let num_chunks = num_clusters.div_ceil(SEG_NUM_SLOTS);
  let f = num_frames_per_chunk;
  let mut segmentations = vec![0.0f64; num_chunks * f * SEG_NUM_SLOTS];
  let mut raw_embeddings = vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM];
  for g in 0..num_clusters {
    let c = g / SEG_NUM_SLOTS;
    let s = g % SEG_NUM_SLOTS;
    // Far vector #g: a signed one-hot. Distinct positions → cosine similarity 0
    // (distance 1.0); `+e_i` vs `-e_i` → similarity -1 (distance 2.0). Either way
    // >= speaker_threshold, so #g is a NEW speaker w.r.t. every earlier centroid.
    let pos = g % EMBEDDING_DIM;
    let sign = if g < EMBEDDING_DIM { 1.0f32 } else { -1.0f32 };
    raw_embeddings[(c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM + pos] = sign;
    for ff in 0..f {
      segmentations[(c * f + ff) * SEG_NUM_SLOTS + s] = 1.0;
    }
  }
  // Chunk window sized to this fixture's F-frame chunks (same rationale as the
  // other online fixtures: reconstruct ignores chunk DURATION, but the count
  // helpers derive num_output_frames from it).
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new())
    .with_duration((f as f64 - 1.0) * crate::audio::speaker::window::FRAME_STEP_S);
  let frames_sw = crate::audio::speaker::window::frame_sliding_window();
  // A valid offline `count` (diarize_online no longer consumes it, but Extraction
  // owns the offline contract); its length IS num_output_frames.
  let count = crate::audio::speaker::window::count_from_segmentations(
    &segmentations,
    num_chunks,
    f,
    SEG_NUM_SLOTS,
    0.5,
    chunks_sw,
    frames_sw,
  );
  Extraction::from_parts(
    raw_embeddings,
    segmentations,
    count,
    num_chunks,
    f,
    chunks_sw,
    frames_sw,
  )
}

#[test]
fn diarize_online_many_clusters_use_no_cluster_axis_allocation() {
  // 380 distinct global speakers over ceil(380/3) = 127 chunks (well past the
  // 3-slot local ceiling, deep into the total-cluster regime the finding is about;
  // 380 < 2*EMBEDDING_DIM = 512, the {±e_i} maximum). num_clusters_from_hard = 380,
  // so the DELETED dense buffer was `num_chunks × num_frames_per_chunk × 380` f64,
  // scaling with the TOTAL cluster count. At the PRODUCTION per-chunk frame count
  // (589) that is 127 × 589 × 380 = 2.84e7 cells ≈ 227 MB for this many clusters —
  // the hundreds-of-MB process-OOM the finding cites (and the `..._over_cap_grid_...`
  // test below drives the same shape past a GiB). F is kept tiny here purely so the
  // debug-build reconstruct stays fast; the allocation being fenced is independent
  // of F. The NEW code allocates only a num_chunks × F chunk_count (127 × 4 = 508
  // f64 ≈ 4 KB, NO cluster axis), then reuses the shared output-frame aggregator —
  // so this test completing IS the allocation proof: no cluster-proportional buffer
  // is ever materialized. (diaric's own reconstruct grid is checked/capped/
  // spill-backed, unlike the deleted speakerkit buffer.)
  const NUM_CLUSTERS: usize = 380;
  const F: usize = 4;
  let e = many_cluster_online_extraction(NUM_CLUSTERS, F);
  assert_eq!(e.num_chunks(), 127, "ceil(380/3) chunks");

  let out = e
    .diarize_online(OnlineOptions::new().with_min_speech_duration(0.0))
    .expect("high-churn online reconstruction succeeds with no cluster-axis allocation");

  assert_eq!(
    out.num_clusters(),
    NUM_CLUSTERS,
    "every distinct far embedding seeds its own global speaker"
  );
  // hard_clusters: chunk c slot s → label c*3+s for the first 380 slots; the 381st
  // slot (chunk 126, slot 2) is past NUM_CLUSTERS → the dropped tail (UNMATCHED),
  // which exercises the distinct-count's `k < 0` skip amid many clusters.
  let hc = out.hard_clusters_slice();
  assert_eq!(hc.len(), 127);
  assert_eq!(hc[0], [0, 1, 2], "first chunk seeds labels 0,1,2");
  assert_eq!(
    hc[126],
    [378, 379, -2],
    "last chunk: two labels + the dropped tail"
  );

  let spans = out.spans_slice();
  assert!(
    !spans.is_empty(),
    "reconstruction produced spans for the many clusters"
  );
  assert!(
    spans.iter().all(|s| s.cluster() < NUM_CLUSTERS),
    "every span names a cluster inside the 380-speaker roster"
  );
}

#[test]
fn diarize_online_over_cap_grid_is_a_typed_reconstruct_error_not_an_oom() {
  // The finding's OOM SHAPE made safe. 380 clusters × 127 chunks × F = 8300 frames
  // gives a clustered-grid cell count of 127 × 8300 × 380 = 4.006e8 cells, just past
  // diaric's MAX_RECONSTRUCT_GRID_CELLS (4e8). The OLD speakerkit code allocated
  // exactly this `num_chunks × num_frames_per_chunk × num_clusters_from_hard` f64
  // buffer (4.006e8 × 8 B ≈ 3.2 GiB) INSIDE speakerkit, BEFORE diaric's guard could
  // fire — the reachable process-OOM/abort. The NEW code allocates only a
  // num_chunks × F chunk_count (no cluster axis), so diaric's typed cell-count cap
  // (`ShapeError::OutputGridTooLarge`, reconstruct/algo.rs's cs_size guard) rejects
  // cleanly. Its SIBLING cluster-id cap (`ShapeError::HardClustersIdAboveMax`,
  // reconstruct/algo.rs — a hard-cluster id above MAX_CLUSTER_ID = 1023) is the
  // analogous typed rejection for the >1023-speaker case (not economical to seed
  // here: {±e_i} caps at 512 distinct far vectors). Either way the fix's guarantee
  // is the same: a typed `Reconstruct` error, never an OOM.
  const NUM_CLUSTERS: usize = 380;
  const F: usize = 8300; // 127 * 8300 * 380 = 4.006e8 > 4e8
  let e = many_cluster_online_extraction(NUM_CLUSTERS, F);

  let err = e
    .diarize_online(OnlineOptions::new().with_min_speech_duration(0.0))
    .expect_err("an over-cap clustered grid must be a typed reconstruct error, not an OOM/panic");

  assert!(
    matches!(
      err,
      diaric::offline::Error::Reconstruct(diaric::reconstruct::Error::Shape(
        diaric::reconstruct::ShapeError::OutputGridTooLarge { .. }
      ))
    ),
    "expected Reconstruct(Shape(OutputGridTooLarge)), got {err:?}"
  );
}

/// An online extraction whose FIRST `num_speakers` slots — in feed order
/// (chunk-major then slot-major) — each carry an identical all-ones (hence
/// normalizable) embedding active across all `F` frames, with every remaining
/// tail slot left zero (a zero embedding row `normalize_from` rejects, so the
/// slot stays UNMATCHED and spawns no speaker). Under `speaker_threshold = 0`
/// (cosine `distance >= 0` is never `< 0`, so the greedy match never fires) and
/// `min_speech_duration = 0` (every `duration >= 0` clears the spawn gate),
/// EVERY active slot spawns a brand-new global speaker regardless of similarity
/// — the ONLY shape that can drive the online path to an arbitrary global count.
/// (`{±e_i}` distinct far vectors cap at `2 * EMBEDDING_DIM = 512`, so
/// `many_cluster_online_extraction` cannot reach the id ceiling.) Feed order
/// makes active slot `g = c*SEG_NUM_SLOTS + s` seed global speaker `g + 1` →
/// 0-based label `g`, so the labels are exactly `0..num_speakers` over
/// `num_chunks = ceil(num_speakers / SEG_NUM_SLOTS)` chunks; a partial final
/// chunk's trailing slots are the dropped remainder. `F` is tiny so reconstruct
/// stays cheap.
///
/// `nan_cell = Some((c, ff, s))` overwrites one `segmentations` cell with
/// `f64::NAN` AFTER the offline `count` is computed (`count_from_segmentations`
/// itself panics on a non-finite cell). A NaN is not `> 0.0`, so it merely
/// drops that one frame from slot `s`'s activity — the slot still spawns its
/// New — but it is the poison `reconstruct` rejects as `NonFinite(Segmentations)`
/// BEFORE it checks the cluster-id cap, which is what separates an early in-loop
/// cap from a late reconstruct rejection.
fn all_new_online_extraction(
  num_speakers: usize,
  nan_cell: Option<(usize, usize, usize)>,
) -> Extraction {
  const F: usize = 4;
  let num_chunks = num_speakers.div_ceil(SEG_NUM_SLOTS);
  let mut segmentations = vec![0.0f64; num_chunks * F * SEG_NUM_SLOTS];
  let mut raw_embeddings = vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM];
  for g in 0..num_speakers {
    let c = g / SEG_NUM_SLOTS;
    let s = g % SEG_NUM_SLOTS;
    // All-ones row: nonzero → `normalize_from` keeps it. Rows are identical,
    // but `speaker_threshold = 0` still makes each active slot a New.
    let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
    raw_embeddings[base..base + EMBEDDING_DIM].fill(1.0);
    for ff in 0..F {
      segmentations[(c * F + ff) * SEG_NUM_SLOTS + s] = 1.0;
    }
  }
  // Same geometry rationale as the other online fixtures: reconstruct ignores
  // chunk DURATION, but the count helper derives num_output_frames from it.
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new())
    .with_duration((F as f64 - 1.0) * crate::audio::speaker::window::FRAME_STEP_S);
  let frames_sw = crate::audio::speaker::window::frame_sliding_window();
  // Offline count from the CLEAN segmentations — `count_from_segmentations`
  // panics on a non-finite cell — THEN plant the sentinel, so only `reconstruct`
  // (reached via `diarize_online`) ever validates the NaN.
  let count = crate::audio::speaker::window::count_from_segmentations(
    &segmentations,
    num_chunks,
    F,
    SEG_NUM_SLOTS,
    0.5,
    chunks_sw,
    frames_sw,
  );
  if let Some((c, ff, s)) = nan_cell {
    segmentations[(c * F + ff) * SEG_NUM_SLOTS + s] = f64::NAN;
  }
  Extraction::from_parts(
    raw_embeddings,
    segmentations,
    count,
    num_chunks,
    F,
    chunks_sw,
    frames_sw,
  )
}

#[test]
fn diarize_online_early_cap_not_late_reconstruction_rejection() {
  // The finding's sibling cap, seeded economically AND strengthened to catch
  // guard REMOVAL (not merely re-observe the error the old uncapped code also
  // returned). `speaker_threshold = 0` and `min_speech_duration = 0` are BOTH
  // accepted by OnlineOptions' validation (finiteness / finite-non-negative),
  // yet together they make the online clusterer spawn a NEW speaker for EVERY
  // active slot. Once 1024 speakers exist (labels 0..=1023), the 1025th's
  // 0-based label 1024 would exceed diaric's `MAX_CLUSTER_ID` (1023); the guard
  // returns the typed `HardClustersIdAboveMax` the moment that 1025th speaker
  // would be labelled, from INSIDE the assign loop — before building the count
  // or running `reconstruct`.
  //
  // The NaN sentinel is what distinguishes an EARLY in-loop cap from the LATE
  // reconstruct rejection the old uncapped code produced. `reconstruct`
  // validates segmentation finiteness (`NonFinite(Segmentations)`) BEFORE the
  // cluster-id cap (reconstruct/algo.rs: finiteness scan, then the id-range
  // check). The NaN sits in chunk 350 — AFTER chunk 341 slot 1, the feed-order
  // slot g = 1024 where the 1025th New is created — so it is reached ONLY if the
  // loop fails to stop at the cap:
  //   • WITH the guard: `diarize_online` returns at that 1025th New, never
  //     builds the count and never calls `reconstruct`, so the NaN is never
  //     validated → `HardClustersIdAboveMax`.
  //   • WITHOUT the guard: the loop runs all 1200 slots and hands the
  //     NaN-bearing segmentations to `reconstruct`, which rejects the NaN FIRST
  //     → `NonFinite(Segmentations)`, a DIFFERENT variant → this assertion reds.
  // (Mutation-verified while authoring: deleting the early return flips the
  // observed error to `NonFinite(Segmentations)` and this test fails.)
  const NUM_SPEAKERS: usize = 1200; // 400 chunks × 3 slots, all New
  let ceiling = diaric::reconstruct::MAX_CLUSTER_ID as usize + 1;
  assert!(
    NUM_SPEAKERS > ceiling,
    "fixture ({NUM_SPEAKERS} speakers) must exceed the {ceiling}-speaker ceiling to reach the cap"
  );
  // Poison one cell in chunk 350 (> chunk 341, where the 1025th New is created).
  let e = all_new_online_extraction(NUM_SPEAKERS, Some((350, 0, 0)));

  let opts = OnlineOptions::default()
    .with_speaker_threshold(0.0)
    .with_min_speech_duration(0.0);
  let err = e
    .diarize_online(opts)
    .expect_err("past MAX_CLUSTER_ID the online loop must return the typed cap error early");

  assert!(
    matches!(
      err,
      diaric::offline::Error::Reconstruct(diaric::reconstruct::Error::Shape(
        diaric::reconstruct::ShapeError::HardClustersIdAboveMax
      ))
    ),
    "expected an EARLY Reconstruct(Shape(HardClustersIdAboveMax)) from the assign-loop cap \
     (removing the guard surfaces NonFinite(Segmentations) from the planted NaN instead), got {err:?}"
  );
}

#[test]
fn diarize_online_accepts_exactly_max_cluster_id_plus_one_speakers() {
  // Boundary companion to the over-ceiling cap above: EXACTLY
  // `MAX_CLUSTER_ID + 1 = 1024` New speakers (labels 0..=1023) must SUCCEED. The
  // guard fires on `id - 1 > MAX_CLUSTER_ID`, and the 1024th New's label 1023 is
  // NOT `> 1023`, so no speaker is ever rejected. This test reds under a
  // `>` → `>=` mutation of the guard: `>=` would reject that 1024th speaker with
  // `HardClustersIdAboveMax`, and this `Ok` would fail.
  // (Mutation-verified while authoring: `>=` turns this into that error.)
  //
  // `reconstruct` must accept the 1024-wide grid: with F = 4 and the default 1 s
  // chunk step the grid is `num_output_frames × 1024` ≈ 2.07e7 cells, far under
  // diaric's `MAX_RECONSTRUCT_GRID_CELLS` (4e8); and `try_discrete_to_spans`
  // caps at `num_clusters > MAX_CLUSTER_ID + 1`, so exactly 1024 passes. The
  // tail slots of the partial final chunk (342 = ceil(1024/3)) are dropped.
  let ceiling = diaric::reconstruct::MAX_CLUSTER_ID as usize + 1;
  assert_eq!(
    ceiling, 1024,
    "diaric's reconstruction ceiling is MAX_CLUSTER_ID + 1"
  );
  let e = all_new_online_extraction(ceiling, None);

  let out = e
    .diarize_online(
      OnlineOptions::default()
        .with_speaker_threshold(0.0)
        .with_min_speech_duration(0.0),
    )
    .expect("exactly MAX_CLUSTER_ID + 1 speakers sit ON the ceiling and must reconstruct");

  assert_eq!(
    out.num_clusters(),
    ceiling,
    "every one of the 1024 all-New slots keeps its own cluster (labels 0..=1023)"
  );
}

// =====================================================================
// Model-gated (all #[ignore]): requires local speakerkit models
// (SPEAKERKIT_TEST_MODELS or Models/speakerkit/) plus the cross-crate
// ted_60.wav fixture. Loader/path helpers duplicated in miniature because
// unit tests under `src/` cannot import the separate `tests/`
// integration-test crate — same reason as crate::audio::speaker::embed::tests and
// crate::audio::speaker::segment::tests.
// =====================================================================

fn models_dir() -> std::path::PathBuf {
  std::env::var_os("SPEAKERKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("speakerkit")
    },
    std::path::PathBuf::from,
  )
}

fn load_seg_model() -> SegmentModel {
  // CpuOnly for determinism (no ANE compile-latency variance across runs),
  // matching crate::audio::speaker::segment::tests::load_seg_model. DEFAULT_SEGMENT_COMPUTE
  // (All) stays the production default.
  SegmentModel::from_file_with(
    models_dir().join("pyannote_segmentation.mlmodelc"),
    crate::audio::speaker::segment::SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load pyannote_segmentation.mlmodelc")
}

fn load_embed_model() -> EmbedModel {
  EmbedModel::from_file_with(
    models_dir().join("wespeaker_v2.mlmodelc"),
    crate::audio::speaker::embed::EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load wespeaker_v2.mlmodelc")
}

/// Reads the cross-crate `ted_60.wav` fixture (16 kHz mono 16-bit PCM,
/// 960_000 samples / 60 s), i16 → f32 / 32768.0 — the same loader shape as
/// `crates/coremlit/tests/whisper/common/mod.rs:45-55`. Reused across crates
/// because it is the one committed multi-speaker clip long enough to
/// exercise the 30 s chunk grid.
fn load_ted_60() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/whisper/fixtures/audio/ted_60.wav");
  let mut reader = hound::WavReader::open(&path).expect("ted_60.wav opens");
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn extract_ted30_invariants() {
  let seg = load_seg_model();
  let embed = load_embed_model();
  let all = load_ted_60();
  assert_eq!(all.len(), 960_000, "ted_60.wav is 60 s at 16 kHz");
  let samples = &all[..480_000]; // first 30 s

  let extraction = Extractor::new()
    .extract(&seg, &embed, samples)
    .expect("extract on 30 s of ted_60");

  let f = seg.num_frames();
  // num_chunks = (480_000 - 160_000).div_ceil(16_000) + 1 = 20 + 1 = 21.
  assert_eq!(extraction.num_chunks(), 21);
  assert_eq!(extraction.num_frames_per_chunk(), f);
  assert_eq!(extraction.num_speakers(), 3);
  assert_eq!(extraction.raw_embeddings().len(), 21 * 3 * EMBEDDING_DIM);
  assert_eq!(extraction.segmentations().len(), 21 * f * 3);
  assert_eq!(extraction.count().len(), extraction.num_output_frames());
  // num_output_frames = round_ties_even((10 + 20*1)/0.016875) + 1
  //                   = round_ties_even(30 / 0.016875) + 1 = 1778 + 1.
  assert_eq!(extraction.num_output_frames(), 1779);

  assert!(
    extraction.count().iter().all(|c| *c <= 3),
    "count never exceeds SEG_NUM_SLOTS = 3"
  );
  assert!(
    extraction.raw_embeddings().iter().all(|v| v.is_finite()),
    "every raw embedding value is finite"
  );
  assert!(
    extraction
      .segmentations()
      .iter()
      .all(|v| *v == 0.0 || *v == 1.0),
    "hard multilabel: every segmentation value is exactly 0.0 or 1.0"
  );
  assert!(
    (0..extraction.num_chunks() * 3).any(|i| extraction.raw_embeddings()
      [i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM]
      .iter()
      .any(|v| *v != 0.0)),
    "at least one embedding row is non-zero (real speech survives the drop paths)"
  );

  // Drop-path invariant: for every (c, s), the embedding row is all-zero
  // IFF the segmentation column is all-zero. Skip and norm-drop both zero
  // the column and leave the row zero (owned.rs:561-571, 619-630); every
  // surviving active slot writes a non-zero row over a non-zero column.
  for c in 0..extraction.num_chunks() {
    for s in 0..3 {
      let row = &extraction.raw_embeddings()[embedding_range(c, s)];
      let row_zero = row.iter().all(|v| *v == 0.0);
      let col_zero =
        (0..f).all(|frame| extraction.segmentations()[(c * f + frame) * SEG_NUM_SLOTS + s] == 0.0);
      assert_eq!(
        row_zero, col_zero,
        "chunk {c} slot {s}: embedding-row-zero must match segmentation-column-zero"
      );
    }
  }
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn extract_empty_samples_errors() {
  let seg = load_seg_model();
  let embed = load_embed_model();
  assert_eq!(
    Extractor::new().extract(&seg, &embed, &[]),
    Err(ExtractError::EmptySamples)
  );
}

// serde-bypass preflight: serde deserialization assigns fields directly,
// bypassing WindowOptions' builder panics (dia's own serde-bypass
// rationale, owned.rs:377-378). These reach `extract`'s own
// defense-in-depth guards, which run BEFORE any inference. Model-gated
// only because `extract`'s signature requires loaded models; they run
// under `cargo test -p coremlit --features speaker,serde -- --ignored`.

#[cfg(feature = "serde")]
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn extract_serde_bypassed_zero_step_samples_errors() {
  let seg = load_seg_model();
  let embed = load_embed_model();
  let options: Options = serde_json::from_str(r#"{"window":{"step_samples":0}}"#).unwrap();
  assert_eq!(
    Extractor::with_options(options).extract(&seg, &embed, &[0.0f32; 10]),
    Err(ExtractError::ZeroStepSamples)
  );
}

#[cfg(feature = "serde")]
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn extract_serde_bypassed_step_samples_exceeds_window_errors() {
  let seg = load_seg_model();
  let embed = load_embed_model();
  let options: Options = serde_json::from_str(r#"{"window":{"step_samples":200000}}"#).unwrap();
  assert_eq!(
    Extractor::with_options(options).extract(&seg, &embed, &[0.0f32; 10]),
    Err(ExtractError::StepSamplesExceedsWindow {
      step: 200_000,
      window: SEG_CHUNK_SAMPLES,
    })
  );
}

#[cfg(feature = "serde")]
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn extract_serde_bypassed_onset_out_of_range_errors() {
  let seg = load_seg_model();
  let embed = load_embed_model();
  let options: Options = serde_json::from_str(r#"{"window":{"onset":0.0}}"#).unwrap();
  assert_eq!(
    Extractor::with_options(options).extract(&seg, &embed, &[0.0f32; 10]),
    Err(ExtractError::OnsetOutOfRange { onset: 0.0 })
  );
}
