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
  // The shared small, self-consistent Extraction (num_chunks=1, F=2, count len
  // == num_output_frames=4). One definition, so the geometry these assertions
  // read cannot drift from the one `try_from_parts` validates below.
  let e = tiny_extraction();

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
  // Shared, not re-inlined: see that test's note.
  let e = tiny_extraction();
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
/// num_output_frames=4) — the fixture the round-trip / diarize tests above and
/// the `try_from_parts` suite below all build from. Private fields are visible
/// to this child module.
///
/// "Self-consistent" is load-bearing, not decoration: [`Extraction::try_from_parts`]
/// enforces every cross-part invariant these fields carry, so this fixture must
/// satisfy them all or the round-trip through the public constructor fails.
/// - The chunk window is shortened to three frame-steps so the geometry derives
///   exactly `count.len()` output frames. At the nominal 10 s chunk duration it
///   would derive 594.
/// - `count` is `[1, 1, 0, 0]`: the single chunk covers output frames 0 and 1
///   with one active slot each, and frames 2-3 are covered by no chunk at all.
fn tiny_extraction() -> Extraction {
  Extraction {
    raw_embeddings: (0..(SEG_NUM_SLOTS * EMBEDDING_DIM))
      .map(|i| i as f32 * 0.25 - 3.0)
      .collect(),
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    count: vec![1, 1, 0, 0],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    num_output_frames: 4,
    chunks_sw: crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new())
      .with_duration(3.0 * crate::audio::speaker::window::FRAME_STEP_S),
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

  // count[t]: the number of active slots this fixture's own segmentations put at
  // each output frame — ONE per frame over chunk 0 (output frames 0-3, one active
  // slot each) and TWO over chunk 1 (output frames 59-62, two active slots each),
  // 0 on the uncovered frames between. Length == num_output_frames. NOTE:
  // `diarize_online` no longer consumes this field (it derives its own
  // clustered-segmentation count); it is retained as a valid `Extraction::count`
  // (the offline path's contract, and what `try_from_parts` validates).
  let mut count = vec![0u8; 63];
  count[0..4].fill(1);
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

// =====================================================================
// try_from_parts — the PUBLIC construction site (issue #110). Hermetic:
// no models, ort-free. Two things are proven here.
//
// 1. THE ROUND TRIP, which is the property the issue exists for: an
//    `Extraction` taken apart through the public accessors and rebuilt
//    through the public constructor produces a byte-identical
//    `OfflineOutput` from `diarize_with` — on BOTH backends.
// 2. EVERY enforced invariant, each with a falsifier that violates that
//    one and satisfies the rest, asserting the specific typed error.
//
// The fixtures reuse `tiny_extraction()` / `online_extraction()` above, so
// the round trip is pinned on extractions the existing wiring tests
// already exercise.
// =====================================================================

use crate::audio::speaker::error::ExtractionPart;

/// The [`ExtractionParts`] of [`tiny_extraction`], self-consistent:
/// `num_chunks = 1`, `num_frames_per_chunk = 2`, so `raw_embeddings` is
/// `1 * 3 * 256 = 768` long and `segmentations` is `1 * 2 * 3 = 6`; the chunk
/// window derives exactly `count.len() == 4` output frames and `count` stays
/// within what those segmentations support. Every negative test below starts
/// from this and breaks exactly one thing.
///
/// Decomposed from [`tiny_extraction`] rather than re-listed, so the fixture and
/// its parts cannot drift apart — a `valid_parts` that stopped describing a real
/// `Extraction` would make every negative test below vacuous.
fn valid_parts() -> ExtractionParts {
  let e = tiny_extraction();
  ExtractionParts {
    raw_embeddings: e.raw_embeddings().to_vec(),
    segmentations: e.segmentations().to_vec(),
    count: e.count().to_vec(),
    num_chunks: e.num_chunks(),
    num_frames_per_chunk: e.num_frames_per_chunk(),
    chunks_sw: e.chunks_sw(),
    frames_sw: e.frames_sw(),
  }
}

/// Decomposes `e` through its PUBLIC accessors ONLY (no field access, even
/// though this child module can see the fields) and rebuilds it through the
/// PUBLIC constructor — exactly what mediagraph's cluster node does with parts
/// it accumulated from upstream messages.
fn rebuild_through_public_api(e: &Extraction) -> Result<Extraction, ExtractError> {
  Extraction::try_from_parts(ExtractionParts {
    raw_embeddings: e.raw_embeddings().to_vec(),
    segmentations: e.segmentations().to_vec(),
    count: e.count().to_vec(),
    num_chunks: e.num_chunks(),
    num_frames_per_chunk: e.num_frames_per_chunk(),
    chunks_sw: e.chunks_sw(),
    frames_sw: e.frames_sw(),
  })
}

/// The tuple [`output_fingerprint`] returns: spans, per-chunk hard assignment,
/// cluster count, frame-level discrete grid.
type OutputFingerprint = (
  Vec<(f64, f64, usize)>,
  Vec<diaric::pipeline::ChunkAssignment>,
  usize,
  Vec<f32>,
);

/// Every observable field of a [`diaric::offline::OfflineOutput`], flattened for
/// exact comparison (`OfflineOutput` is not `PartialEq`). Covers the span
/// geometry, the per-chunk hard assignment, the cluster count, AND the
/// frame-level discrete grid — so a divergence anywhere in the reconstruction,
/// not just in the spans, breaks the round-trip assertion.
fn output_fingerprint(o: &diaric::offline::OfflineOutput) -> OutputFingerprint {
  (
    o.spans_slice()
      .iter()
      .map(|s| (s.start(), s.end(), s.cluster()))
      .collect(),
    o.hard_clusters_slice().to_vec(),
    o.num_clusters(),
    o.discrete_diarization_slice().to_vec(),
  )
}

#[test]
fn try_from_parts_round_trips_an_extraction_through_the_public_api() {
  // THE issue-#110 property, offline backend: accessors out, constructor in,
  // identical clustering. A dropped or transposed part would break either the
  // `PartialEq` or the fingerprint.
  let original = tiny_extraction();
  let rebuilt = rebuild_through_public_api(&original).expect("a real Extraction's own parts");

  // Same value, including the two DERIVED members (num_output_frames from
  // count.len(), num_speakers from SEG_NUM_SLOTS) that are not parts.
  assert_eq!(
    rebuilt, original,
    "rebuilt Extraction diverged from the original"
  );
  assert_eq!(rebuilt.num_output_frames(), original.count().len());
  assert_eq!(rebuilt.num_speakers(), SEG_NUM_SLOTS);

  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let backend = ClusterBackend::default();
  let from_original = original.diarize_with(&plda, backend);
  let from_rebuilt = rebuilt.diarize_with(&plda, backend);

  match (from_original, from_rebuilt) {
    (Ok(a), Ok(b)) => assert_eq!(
      output_fingerprint(&a),
      output_fingerprint(&b),
      "rebuilt Extraction produced a different OfflineOutput"
    ),
    (Err(a), Err(b)) => assert_eq!(
      format!("{a:?}"),
      format!("{b:?}"),
      "rebuilt Extraction refused differently"
    ),
    (a, b) => panic!(
      "rebuilt Extraction diverged: original {} vs rebuilt {}",
      if a.is_ok() { "Ok" } else { "Err" },
      if b.is_ok() { "Ok" } else { "Err" },
    ),
  }
}

#[test]
fn try_from_parts_round_trips_the_online_backend_too() {
  // The same round trip on the ONLINE route, whose output depends on the raw
  // embeddings, the segmentation activity counts AND the frame timing — a part
  // that survived the offline comparison by accident cannot survive both.
  let original = online_extraction();
  let rebuilt = rebuild_through_public_api(&original).expect("a real Extraction's own parts");
  assert_eq!(rebuilt, original);

  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let backend = ClusterBackend::Online(OnlineOptions::new().with_min_speech_duration(0.0));
  let a = original
    .diarize_with(&plda, backend)
    .expect("online reconstruction succeeds on a valid extraction");
  let b = rebuilt
    .diarize_with(&plda, backend)
    .expect("online reconstruction succeeds on the rebuilt extraction");
  assert_eq!(
    output_fingerprint(&a),
    output_fingerprint(&b),
    "rebuilt Extraction produced a different online OfflineOutput"
  );
}

#[test]
fn try_from_parts_accepts_self_consistent_parts_and_derives_num_output_frames() {
  // `num_output_frames` is NOT a part: it IS count.len(). Pinned with a count
  // length that matches nothing else in the geometry (7 != 1, 2, 6, 768), so a
  // constructor that derived it from any other dimension would fail here. The
  // chunk window is stretched to six frame-steps so the geometry DERIVES those
  // seven frames; only the two the single chunk covers carry a speaker.
  let mut parts = valid_parts();
  parts.chunks_sw = parts
    .chunks_sw
    .with_duration(6.0 * crate::audio::speaker::window::FRAME_STEP_S);
  parts.count = vec![1, 1, 0, 0, 0, 0, 0];
  let e = Extraction::try_from_parts(parts).expect("self-consistent parts");
  assert_eq!(e.num_output_frames(), 7);
  assert_eq!(e.count().len(), 7);
  assert_eq!(e.num_speakers(), SEG_NUM_SLOTS);
  assert_eq!(e.num_chunks(), 1);
  assert_eq!(e.num_frames_per_chunk(), 2);
}

// ── Falsifiers: one per enforced invariant ──────────────────────────────

#[test]
fn try_from_parts_rejects_zero_num_chunks() {
  // num_chunks = 0 makes both expected lengths 0, so the empty tensors keep
  // every OTHER invariant satisfied; only the zero dimension is violated.
  // Unchecked, `window::try_aggregate_output_frame_count`'s `assert!(num_chunks
  // > 0)` would PANIC inside `diarize_online`.
  let parts = ExtractionParts {
    raw_embeddings: Vec::new(),
    segmentations: Vec::new(),
    num_chunks: 0,
    ..valid_parts()
  };
  assert_eq!(
    Extraction::try_from_parts(parts).unwrap_err(),
    ExtractError::ZeroExtractionDimension(ExtractionPart::NumChunks)
  );
}

#[test]
fn try_from_parts_rejects_zero_num_frames_per_chunk() {
  // num_frames_per_chunk = 0 makes the expected segmentations length 0, so the
  // empty segmentations satisfy that invariant; raw_embeddings stays exactly
  // 1 * 3 * 256. Only the zero dimension is violated.
  let parts = ExtractionParts {
    segmentations: Vec::new(),
    num_frames_per_chunk: 0,
    ..valid_parts()
  };
  assert_eq!(
    Extraction::try_from_parts(parts).unwrap_err(),
    ExtractError::ZeroExtractionDimension(ExtractionPart::NumFramesPerChunk)
  );
}

#[test]
fn try_from_parts_rejects_empty_count() {
  // count.len() IS num_output_frames; zero of them makes `diarize_online`'s
  // `discrete.len() / num_output_frames` a divide-by-zero and is `diaric`'s own
  // ZeroNumOutputFrames. No length invariant involves `count`, so every other
  // one still holds.
  let parts = ExtractionParts {
    count: Vec::new(),
    ..valid_parts()
  };
  assert_eq!(
    Extraction::try_from_parts(parts).unwrap_err(),
    ExtractError::ZeroExtractionDimension(ExtractionPart::Count)
  );
}

#[test]
fn try_from_parts_rejects_non_positive_chunks_sw_step() {
  // step = 0 trips `try_aggregate_output_frame_count`'s bare
  // `assert!(chunk_step > 0.0)` — a panic, not a typed error — on the online
  // route. Everything else in `valid_parts` is untouched.
  let base = valid_parts();
  let parts = ExtractionParts {
    chunks_sw: base.chunks_sw.with_step(0.0),
    ..base
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::InvalidSlidingWindow(w) = err else {
    panic!("expected InvalidSlidingWindow, got {err:?}")
  };
  assert_eq!(w.part(), ExtractionPart::ChunksSw);
  assert_eq!(w.window().step(), 0.0);
}

#[test]
fn try_from_parts_rejects_non_finite_frames_sw_duration() {
  // The frames window is the other half of the same invariant, and `duration`
  // the other half of the same predicate: NaN trips
  // `assert!(frame_duration.is_finite() && frame_duration > 0.0)`.
  let base = valid_parts();
  let parts = ExtractionParts {
    frames_sw: base.frames_sw.with_duration(f64::NAN),
    ..base
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::InvalidSlidingWindow(w) = err else {
    panic!("expected InvalidSlidingWindow, got {err:?}")
  };
  assert_eq!(w.part(), ExtractionPart::FramesSw);
  assert!(w.window().duration().is_nan());
}

// The window predicate is a five-clause conjunction, and NaN satisfies neither
// half of a `is_finite() && > 0.0` pair — so the NaN and zero cases above leave
// three clauses that no test can falsify on its own. These three do: each picks
// the one value that passes every OTHER clause, so deleting its clause makes the
// constructor return `Ok` and only this test goes red.

#[test]
fn try_from_parts_rejects_infinite_frames_sw_duration() {
  // Sole falsifier of `w.duration().is_finite()`. `+inf > 0.0` is TRUE, so the
  // positivity clause lets this through; `frames_sw.duration` is also not read
  // by `try_num_output_frames`, so check 4 lets it through too.
  let base = valid_parts();
  let parts = ExtractionParts {
    frames_sw: base.frames_sw.with_duration(f64::INFINITY),
    ..base
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::InvalidSlidingWindow(w) = err else {
    panic!("expected InvalidSlidingWindow, got {err:?}")
  };
  assert_eq!(w.part(), ExtractionPart::FramesSw);
  assert_eq!(w.window().duration(), f64::INFINITY);
}

#[test]
fn try_from_parts_rejects_zero_frames_sw_duration() {
  // Sole falsifier of `w.duration() > 0.0` — the "field never assigned" value.
  // `0.0.is_finite()` is TRUE, so the finiteness clause lets this through, and
  // a zero-duration frame window still trips
  // `try_aggregate_output_frame_count`'s
  // `assert!(frame_duration.is_finite() && frame_duration > 0.0)`.
  let base = valid_parts();
  let parts = ExtractionParts {
    frames_sw: base.frames_sw.with_duration(0.0),
    ..base
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::InvalidSlidingWindow(w) = err else {
    panic!("expected InvalidSlidingWindow, got {err:?}")
  };
  assert_eq!(w.part(), ExtractionPart::FramesSw);
  assert_eq!(w.window().duration(), 0.0);
}

#[test]
fn try_from_parts_rejects_infinite_frames_sw_step() {
  // Sole falsifier of `w.step().is_finite()`. The step case above uses `0.0`,
  // which IS finite, so it exercises only the positivity clause. `+inf > 0.0` is
  // TRUE, and `last_chunk_end / +inf` rounds to `0`, so check 4 returns `Ok(1)`
  // — without the finiteness clause this window would be ACCEPTED.
  let base = valid_parts();
  let parts = ExtractionParts {
    frames_sw: base.frames_sw.with_step(f64::INFINITY),
    ..base
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::InvalidSlidingWindow(w) = err else {
    panic!("expected InvalidSlidingWindow, got {err:?}")
  };
  assert_eq!(w.part(), ExtractionPart::FramesSw);
  assert_eq!(w.window().step(), f64::INFINITY);
}

#[test]
fn try_from_parts_rejects_non_finite_sliding_window_start() {
  // `start` is the third component of the timing grid. It is not read by
  // `try_aggregate_output_frame_count`, but `diaric`'s reconstruct requires it
  // finite (TimingError::NonFiniteParameter), so an infinite start is rejected
  // here rather than several stages later.
  let base = valid_parts();
  let parts = ExtractionParts {
    chunks_sw: base.chunks_sw.with_start(f64::INFINITY),
    ..base
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::InvalidSlidingWindow(w) = err else {
    panic!("expected InvalidSlidingWindow, got {err:?}")
  };
  assert_eq!(w.part(), ExtractionPart::ChunksSw);
  assert!(w.window().start().is_infinite());
}

#[test]
fn try_from_parts_rejects_raw_embeddings_geometry_overflow() {
  // num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM = 2^60 * 768 overflows usize.
  // Every invariant checked BEFORE this one is satisfied (non-zero dims, valid
  // windows). The segmentations length (2^60 * 1 * 3) is unsatisfiable by any
  // allocatable Vec — inherent, because both products share `num_chunks`, which
  // is exactly why the embeddings product is checked first.
  let parts = ExtractionParts {
    raw_embeddings: Vec::new(),
    segmentations: Vec::new(),
    num_chunks: 1usize << 60,
    num_frames_per_chunk: 1,
    ..valid_parts()
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::ExtractionGeometryOverflow(g) = err else {
    panic!("expected ExtractionGeometryOverflow, got {err:?}")
  };
  assert_eq!(g.part(), ExtractionPart::RawEmbeddings);
  assert_eq!(g.num_chunks(), 1usize << 60);
}

#[test]
fn try_from_parts_rejects_segmentations_geometry_overflow() {
  // 2^32 * 2^32 * 3 wraps to 0 in unchecked usize arithmetic, so an EMPTY
  // segmentations vector would satisfy a naive equality check — the precise
  // reason the products are computed with `checked_mul` before any comparison.
  // The embeddings product (2^32 * 768) does NOT overflow, so the earlier check
  // passes and this one is reached.
  let parts = ExtractionParts {
    raw_embeddings: Vec::new(),
    segmentations: Vec::new(),
    num_chunks: 1usize << 32,
    num_frames_per_chunk: 1usize << 32,
    ..valid_parts()
  };
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::ExtractionGeometryOverflow(g) = err else {
    panic!("expected ExtractionGeometryOverflow, got {err:?}")
  };
  assert_eq!(g.part(), ExtractionPart::Segmentations);
  assert_eq!(g.num_chunks(), 1usize << 32);
  assert_eq!(g.num_frames_per_chunk(), 1usize << 32);
}

#[test]
fn try_from_parts_rejects_raw_embeddings_len_mismatch() {
  // One element short of 1 * 3 * 256. Everything else — including the
  // segmentations length — is exactly right.
  let mut parts = valid_parts();
  parts.raw_embeddings.pop();
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::ExtractionLenMismatch(m) = err else {
    panic!("expected ExtractionLenMismatch, got {err:?}")
  };
  assert_eq!(m.part(), ExtractionPart::RawEmbeddings);
  assert_eq!(
    (m.got(), m.expected()),
    (767, SEG_NUM_SLOTS * EMBEDDING_DIM)
  );
  // The message must name the part and both numbers: a caller debugging a
  // message-assembly bug needs to know WHICH tensor and BY HOW MUCH.
  let rendered = err.to_string();
  assert!(rendered.contains("raw_embeddings"), "{rendered}");
  assert!(rendered.contains("767"), "{rendered}");
  assert!(rendered.contains("768"), "{rendered}");
}

#[test]
fn try_from_parts_rejects_segmentations_len_mismatch() {
  // One element short of 1 * 2 * 3, with the embeddings length exactly right —
  // the two tensors are reported separately so a caller knows which upstream
  // stage to look at.
  let mut parts = valid_parts();
  parts.segmentations.pop();
  let err = Extraction::try_from_parts(parts).unwrap_err();
  let ExtractError::ExtractionLenMismatch(m) = err else {
    panic!("expected ExtractionLenMismatch, got {err:?}")
  };
  assert_eq!(m.part(), ExtractionPart::Segmentations);
  assert_eq!((m.got(), m.expected()), (5, 6));
  let rendered = err.to_string();
  assert!(rendered.contains("segmentations"), "{rendered}");
  assert!(rendered.contains('5'), "{rendered}");
  assert!(rendered.contains('6'), "{rendered}");
}

#[test]
fn try_from_parts_rejects_geometry_whose_output_frame_count_overflows() {
  // Both windows are finite and strictly positive — they pass check 2 — yet
  // last_chunk_end / frames_sw.step() = 1e300 / 1e-300 divides to +inf. This is
  // the geometry `diarize_online` would feed to `try_num_output_frames` and
  // then `.expect(..)`: without this check the panic happens there, far from
  // the assembly bug that caused it.
  let parts = ExtractionParts {
    chunks_sw: SlidingWindow::new(0.0, 1e300, 1.0),
    frames_sw: SlidingWindow::new(0.0, 0.1, 1e-300),
    ..valid_parts()
  };
  assert_eq!(
    Extraction::try_from_parts(parts).unwrap_err(),
    ExtractError::OutputFrameCountOverflow
  );
}

#[test]
fn try_from_parts_guarantee_makes_diarize_online_panic_free() {
  // The end-to-end statement of what checks 1, 2 and 4 buy: an Extraction that
  // came through `try_from_parts` reaches `diarize_online` — the method with
  // the bare asserts and the `.expect(..)` — without panicking. Any Err here is
  // a typed `diaric` refusal, which is the contract; a panic is not.
  let e = Extraction::try_from_parts(valid_parts()).expect("self-consistent parts");
  let _ = e.diarize_online(OnlineOptions::new());
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let _ = e.diarize(&plda);
}

#[test]
fn diarize_online_refuses_an_oversized_derived_grid_before_allocating_it() {
  // TWO doors, both shut, on the one OOM vector a PUBLIC constructor opens.
  // These windows are finite and strictly positive and the derived output-frame
  // count (1e15 + 1) fits `usize`, yet they describe a grid `diarize_online`
  // would otherwise materialise as TWO `f64` buffers of 8 PB each. An
  // allocation that large is a process abort, not a catchable failure.
  //
  // Same convention as `diarize_online_over_cap_grid_is_a_typed_reconstruct_
  // error_not_an_oom` above: THIS TEST COMPLETING IS THE ALLOCATION PROOF.
  let chunks_sw = SlidingWindow::new(0.0, 1e13, 1.0);
  let frames_sw = SlidingWindow::new(0.0, 0.06, 0.01);

  // Door 1: the public constructor. `count.len()` is not the grid this geometry
  // derives, which is now a refusal rather than an accepted `Extraction`.
  let err = refused(ExtractionParts {
    chunks_sw,
    frames_sw,
    ..valid_parts()
  });
  let ExtractError::ExtractionLenMismatch(m) = err else {
    panic!("expected an ExtractionLenMismatch on count, got {err:?}")
  };
  assert_eq!(m.part(), ExtractionPart::Count);
  assert_eq!(m.got(), 4);

  // Door 2: `diarize_online`'s own guard, which stays load-bearing because
  // `Extraction`'s fields are crate-private but its in-crate constructors are
  // UNCHECKED (`from_parts`). Built here the way a source builds one, so the
  // guard is exercised on exactly the geometry door 1 now rejects.
  let base = tiny_extraction();
  let e = Extraction {
    chunks_sw,
    frames_sw,
    ..base
  };
  assert_eq!(e.num_output_frames(), 4);
  let err = e
    .diarize_online(OnlineOptions::new())
    .expect_err("a derived grid that does not match num_output_frames must be refused");
  assert!(
    matches!(
      err,
      diaric::offline::Error::Reconstruct(diaric::reconstruct::Error::Shape(
        diaric::reconstruct::ShapeError::CountLenMismatch
      ))
    ),
    "expected Reconstruct(Shape(CountLenMismatch)), got {err:?}"
  );
}

#[test]
fn try_from_parts_cannot_detect_mutually_inconsistent_parts() {
  // CHARACTERIZATION of the constructor's boundary, not an endorsement of it.
  //
  // Every check in `try_from_parts` is a SHAPE check. Parts that are each
  // individually well-formed, and whose declared geometry describes them all
  // correctly, are accepted even when they came from DIFFERENT tracks — which is
  // precisely the mediagraph failure mode the constructor cannot see: two
  // upstream stages, each internally consistent, joined on the wrong key.
  //
  // Here `segmentations`/`count`/geometry come from `online_extraction()` and
  // `raw_embeddings` from a second track with the identical geometry. Nothing
  // rejects it, and the clustering silently differs from the real track's — three
  // online clusters become one. Shapes carry no provenance, so no check inside
  // this constructor could distinguish the two; detecting it needs a track or
  // message identity carried alongside the parts, upstream of here.
  let a = online_extraction();

  // Second track, same geometry: every (chunk, slot) embedding in block 0, so
  // every surviving slot is the SAME speaker.
  let mut other_track_embeddings = vec![0.0f32; 2 * SEG_NUM_SLOTS * EMBEDDING_DIM];
  for c in 0..2 {
    for s in 0..SEG_NUM_SLOTS {
      let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
      other_track_embeddings[base..base + 64].fill(1.0);
    }
  }
  assert_eq!(
    other_track_embeddings.len(),
    a.raw_embeddings().len(),
    "the two tracks must be shape-identical for this to be about provenance"
  );

  let crossed = Extraction::try_from_parts(ExtractionParts {
    raw_embeddings: other_track_embeddings,
    segmentations: a.segmentations().to_vec(),
    count: a.count().to_vec(),
    num_chunks: a.num_chunks(),
    num_frames_per_chunk: a.num_frames_per_chunk(),
    chunks_sw: a.chunks_sw(),
    frames_sw: a.frames_sw(),
  })
  .expect("shape-valid parts from two tracks are ACCEPTED — the gap this test pins");

  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let online = ClusterBackend::Online(OnlineOptions::new().with_min_speech_duration(0.0));
  let real = output_fingerprint(&a.diarize_with(&plda, online).expect("real track diarizes"));
  let mixed = output_fingerprint(&crossed.diarize_with(&plda, online).expect("mixed diarizes"));

  // The mix-up is CONSEQUENTIAL, not cosmetic: it changes the answer.
  assert_eq!(real.2, 3, "the real track has three online clusters");
  assert_eq!(mixed.2, 1, "the mixed one collapses to a single speaker");
  assert_ne!(
    real, mixed,
    "a silently-accepted cross-track mix-up must at least be observable here"
  );
}

/// An [`Extraction`] whose `count` and windows are built the way
/// [`Extractor::extract`] builds them: real [`chunk_sliding_window`] /
/// [`frame_sliding_window`] grids and a `count` from the very
/// `window::try_count_from_segmentations` call `extract()` makes at its step
/// 9-11. The closest thing to a model-produced `Extraction` that is reachable
/// without staged models — `extract()` itself is behind a model gate.
///
/// [`chunk_sliding_window`]: crate::audio::speaker::window::chunk_sliding_window
/// [`frame_sliding_window`]: crate::audio::speaker::window::frame_sliding_window
fn extract_shaped_extraction(num_chunks: usize, num_frames_per_chunk: usize) -> Extraction {
  let w = WindowOptions::new();
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&w);
  let frames_sw = crate::audio::speaker::window::frame_sliding_window();

  let mut segmentations = vec![0.0f64; num_chunks * num_frames_per_chunk * SEG_NUM_SLOTS];
  for c in 0..num_chunks {
    for f in 0..num_frames_per_chunk {
      // One active slot per chunk, rotating, so the aggregation is non-trivial.
      segmentations[(c * num_frames_per_chunk + f) * SEG_NUM_SLOTS + (c % SEG_NUM_SLOTS)] = 1.0;
    }
  }
  // EXACTLY the call `extract()` makes at step 9-11.
  let count = crate::audio::speaker::window::try_count_from_segmentations(
    &segmentations,
    num_chunks,
    num_frames_per_chunk,
    SEG_NUM_SLOTS,
    w.onset(),
    chunks_sw,
    frames_sw,
  )
  .expect("this geometry's output-frame count fits usize");

  let raw_embeddings: Vec<f32> = (0..(num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM))
    .map(|i| ((i % 64) as f32).mul_add(0.01, 0.5))
    .collect();

  Extraction::try_from_parts(ExtractionParts {
    raw_embeddings,
    segmentations,
    count,
    num_chunks,
    num_frames_per_chunk,
    chunks_sw,
    frames_sw,
  })
  .unwrap_or_else(|err| panic!("extract()-shaped parts must be accepted: {err}"))
}

#[test]
fn try_from_parts_round_trips_an_extract_shaped_extraction() {
  // The round trip on the most production-like `Extraction` reachable WITHOUT
  // models: real sliding-window grids and a `count` from `extract()`'s own
  // derivation, rather than the hand-chosen geometry of `tiny_extraction` /
  // `online_extraction`. `Extractor::extract` itself needs staged CoreML models,
  // so this is the strongest form available here.
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  for (num_chunks, num_frames_per_chunk) in [(1usize, 4usize), (2, 8), (5, 3)] {
    let original = extract_shaped_extraction(num_chunks, num_frames_per_chunk);
    let rebuilt =
      rebuild_through_public_api(&original).expect("an extract()-shaped Extraction's own parts");
    assert_eq!(rebuilt, original, "({num_chunks}, {num_frames_per_chunk})");

    for backend in [
      ClusterBackend::default(),
      ClusterBackend::Online(OnlineOptions::new().with_min_speech_duration(0.0)),
    ] {
      match (
        original.diarize_with(&plda, backend),
        rebuilt.diarize_with(&plda, backend),
      ) {
        (Ok(a), Ok(b)) => assert_eq!(
          output_fingerprint(&a),
          output_fingerprint(&b),
          "({num_chunks}, {num_frames_per_chunk}) diverged on {backend:?}"
        ),
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
        (a, b) => panic!(
          "({num_chunks}, {num_frames_per_chunk}) diverged on {backend:?}: original {} vs \
           rebuilt {}",
          if a.is_ok() { "Ok" } else { "Err" },
          if b.is_ok() { "Ok" } else { "Err" },
        ),
      }
    }
  }
}

#[test]
fn diarize_online_never_refuses_a_count_derived_the_way_extract_derives_it() {
  // The derived-grid guard `diarize_online` gained alongside `try_from_parts`
  // must be a NO-OP for every in-crate construction path. `Extractor::extract`
  // and the argmax source both build `count` with
  // `window::try_count_from_segmentations`, whose returned length IS
  // `try_num_output_frames(last_chunk_end, frames_sw.step())` — the identical
  // quantity the guard re-derives. Only a comment asserted that agreement; this
  // pins it, hermetically, so a future change to either derivation cannot start
  // refusing extractions that `extract()` produces.
  for (num_chunks, num_frames_per_chunk) in [(1usize, 4usize), (2, 8), (5, 3)] {
    let e = extract_shaped_extraction(num_chunks, num_frames_per_chunk);

    let outcome = e.diarize_online(OnlineOptions::new());
    assert!(
      !matches!(
        outcome,
        Err(diaric::offline::Error::Reconstruct(
          diaric::reconstruct::Error::Shape(diaric::reconstruct::ShapeError::CountLenMismatch)
        ))
      ),
      "the derived-grid guard refused an extract()-derived count at \
       ({num_chunks} chunks, {num_frames_per_chunk} frames/chunk)"
    );
  }
}

// =====================================================================
// try_from_parts — the CROSS-PART invariants both backends consume
// (adversarial review of #110). Every test below is the reviewer's own
// trigger: each one is accepted by a constructor that checks only shapes,
// and each one then makes ONE of the two backends silently wrong.
// =====================================================================

/// A `raw_embeddings` buffer for one chunk whose slot `slot` carries a
/// usable (large, finite) embedding and whose other slots are all-zero.
fn one_usable_slot_row(slot: usize) -> Vec<f32> {
  let mut raw = vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM];
  let base = slot * EMBEDDING_DIM;
  raw[base..base + 64].fill(1.0);
  raw
}

/// Unit-scale timing: one 1-second chunk on a 1-second frame grid, so the
/// derived output-frame count is exactly `2` and every geometry below can be
/// read off by hand.
fn unit_sw() -> SlidingWindow {
  SlidingWindow::new(0.0, 1.0, 1.0)
}

/// `try_from_parts` must REFUSE `parts`; returns the error it raised.
///
/// Not `unwrap_err()`: on the failing (accepted) side that prints the whole
/// `Extraction`, whose 768-value embedding buffer buries the one fact the
/// falsifier is reporting. This names the accepted geometry instead.
#[track_caller]
fn refused(parts: ExtractionParts) -> ExtractError {
  match Extraction::try_from_parts(parts) {
    Err(e) => e,
    Ok(e) => panic!(
      "try_from_parts ACCEPTED these parts: num_chunks={}, num_frames_per_chunk={}, \
       num_output_frames={}, chunks_sw={:?}, frames_sw={:?}",
      e.num_chunks(),
      e.num_frames_per_chunk(),
      e.num_output_frames(),
      e.chunks_sw(),
      e.frames_sw()
    ),
  }
}

#[test]
fn try_from_parts_rejects_an_active_slot_whose_embedding_row_is_unusable() {
  // Finding 1. An ACTIVE segmentation column paired with a row that
  // `diaric::embed::Embedding::normalize_from` refuses. `None` from that
  // function is `diarize_online`'s DROPPED-SLOT sentinel (see its own comment
  // at the `normalize_from` call), so a NaN row is read as "no speaker here":
  // the slot stays UNMATCHED, the online engine returns `Ok` with NO speech,
  // and nothing anywhere says the embedding was corrupt.
  let mut raw = vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM];
  raw[0] = f32::NAN;
  let parts = ExtractionParts {
    raw_embeddings: raw,
    segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    count: vec![1, 1],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  };
  let err = refused(parts);
  let ExtractError::ActiveSlotWithoutEmbedding(a) = err else {
    panic!("expected ActiveSlotWithoutEmbedding, got {err:?}")
  };
  assert_eq!((a.chunk(), a.slot()), (0, 0));

  // The same rejection for the OTHER shape of the same defect: an all-zero
  // row under an active column. `normalize_from` refuses it for zero norm,
  // which is the very sentinel the online route reads as "dropped".
  let parts = ExtractionParts {
    raw_embeddings: vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM],
    segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    count: vec![1, 1],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  };
  let err = refused(parts);
  assert!(
    matches!(err, ExtractError::ActiveSlotWithoutEmbedding(a) if (a.chunk(), a.slot()) == (0, 0)),
    "expected ActiveSlotWithoutEmbedding(0, 0), got {err:?}"
  );
}

#[test]
fn try_from_parts_rejects_a_count_the_segmentations_cannot_support() {
  // Finding 2. One chunk, two frames, ONE active slot — so at most one
  // speaker can be simultaneously active anywhere on this grid — with
  // `count = [4, 4]`. Offline reconstruction pads its single hard cluster out
  // to four columns and top-K marks all four active: three phantom speakers,
  // `Ok`, no diagnostic. `count[t] <= MAX_COUNT_PER_FRAME` does not see it
  // (4 <= 64), and neither does a `SEG_NUM_SLOTS` range check (`[3, 3]`
  // fabricates two and is inside the slot bound).
  for (claimed, supported) in [(4u8, 1u8), (3, 1)] {
    let parts = ExtractionParts {
      raw_embeddings: one_usable_slot_row(0),
      segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      count: vec![claimed, claimed],
      num_chunks: 1,
      num_frames_per_chunk: 2,
      chunks_sw: unit_sw(),
      frames_sw: unit_sw(),
    };
    let err = refused(parts);
    let ExtractError::CountNotSegmentationDerived(c) = err else {
      panic!("expected CountNotSegmentationDerived for count {claimed}, got {err:?}")
    };
    assert_eq!((c.frame(), c.got(), c.expected()), (0, claimed, supported));
  }
}

#[test]
fn try_from_parts_rejects_a_count_length_the_geometry_does_not_derive() {
  // Finding 3. The geometry derives TWO output frames; the parts declare
  // something else. The two backends then DISAGREE about the same `Extraction`:
  // online refuses it with `CountLenMismatch` while offline accepts it. The
  // derived value is already computed here — it is what the overflow guard
  // returns — and the defect was throwing it away with `?`.
  //
  // A SHORT count first, because it is the half NO other check can reach: the
  // support scan walks `count` against the derived grid pairwise, so a `count`
  // that stops early simply ends the scan, and every other invariant holds.
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0],
    count: vec![1],
    num_chunks: 1,
    num_frames_per_chunk: 1,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  };
  let err = refused(parts);
  let ExtractError::ExtractionLenMismatch(m) = err else {
    panic!("expected ExtractionLenMismatch, got {err:?}")
  };
  assert_eq!(m.part(), ExtractionPart::Count);
  assert_eq!((m.got(), m.expected()), (1, 2));

  // And the reviewer's own trigger: TEN declared where two are derived. Offline
  // accepts it and emits speech out to 9.5 s — eight frames past the end of the
  // only chunk.
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0],
    count: vec![1; 10],
    num_chunks: 1,
    num_frames_per_chunk: 1,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  };
  let err = refused(parts);
  assert!(
    matches!(err, ExtractError::ExtractionLenMismatch(m)
      if m.part() == ExtractionPart::Count && (m.got(), m.expected()) == (10, 2)),
    "expected an ExtractionLenMismatch(Count, 10, 2), got {err:?}"
  );
}

#[test]
fn try_from_parts_rejects_an_uncancelled_sliding_window_origin() {
  // Finding 4. `window`'s count aggregation places chunk `c` at
  // `round(c * chunk_step / frame_step)`, ignoring BOTH origins;
  // `diaric::reconstruct` places it at `closest_frame(chunks_sw.start +
  // c * chunk_step + frames_duration / 2)`, which honours both. With
  // `chunks_sw.start = -1` the two disagree by a frame: the online count is
  // written at frames 0 and 1 while reconstruct clips the chunk's first frame
  // and aggregates nothing into frame 1 — whose surviving `count` then marks a
  // zero-activation cell active. Phantom speech, `Ok`, no diagnostic.
  //
  // Round 2 replaced the `start != 0.0` proxy with the condition it was
  // standing in for, so this geometry is now named by the frames the two
  // mappings actually chose.
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    count: vec![1, 1],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: SlidingWindow::new(-1.0, 1.0, 1.0),
    frames_sw: unit_sw(),
  };
  let err = refused(parts);
  let ExtractError::MisalignedChunkPlacement(m) = err else {
    panic!("expected MisalignedChunkPlacement, got {err:?}")
  };
  assert_eq!((m.chunk(), m.aggregated(), m.reconstructed()), (0, 0, -1));

  // The frames window's origin is the other half of the same difference. It has
  // to be a full step to move the mapping: `frames_sw.start = 0.5` on a unit
  // grid normalizes to exactly `-0.5`, and `round_ties_even` leaves that at
  // frame 0 — which is why the old `start != 0.0` proxy was refusing an aligned
  // geometry there rather than a shifted one.
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    count: vec![1, 1],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: unit_sw(),
    frames_sw: SlidingWindow::new(1.0, 1.0, 1.0),
  };
  let err = refused(parts);
  assert!(
    matches!(err, ExtractError::MisalignedChunkPlacement(m)
      if (m.chunk(), m.aggregated(), m.reconstructed()) == (0, 0, -1)),
    "expected MisalignedChunkPlacement(0, 0, -1), got {err:?}"
  );
}

#[test]
fn try_from_parts_rejects_a_frame_step_that_does_not_survive_f32_narrowing() {
  // Finding 5. `diarize_online` narrows `frames_sw.step()` to `f32` to build
  // the speech duration the online gate reads. A step of `7e-46` is finite and
  // strictly positive in `f64` — it passes every timing check — but narrows to
  // exactly `0.0f32`, so a slot active for three frames is handed a speech
  // duration of `0.0` where its own geometry declares `2.1e-45`. With
  // `min_speech_duration = 1.4e-45` (the smallest positive `f32`) the declared
  // duration MEETS the gate and the narrowed one does not: the speaker is
  // dropped and the method returns `Ok` with an empty diarization.
  //
  // `1e-300` (the reviewer's own value) narrows the same way; it is included
  // because it is the value the report names, not because its outcome differs.
  let step = 7.0e-46_f64;
  assert_eq!(step as f32, 0.0, "the premise: this step narrows to zero");
  assert_eq!(1e-300_f64 as f32, 0.0);
  for step in [step, 1e-300_f64] {
    let parts = ExtractionParts {
      raw_embeddings: one_usable_slot_row(0),
      segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      count: vec![1, 1, 1],
      num_chunks: 1,
      num_frames_per_chunk: 3,
      chunks_sw: SlidingWindow::new(0.0, 2.0 * step, step),
      frames_sw: SlidingWindow::new(0.0, 2.0 * step, step),
    };
    let err = refused(parts);
    let ExtractError::FrameStepNotRepresentableInF32(w) = err else {
      panic!("expected FrameStepNotRepresentableInF32 for step {step:e}, got {err:?}")
    };
    assert_eq!(w.part(), ExtractionPart::FramesSw);
    assert_eq!(w.window().step(), step);
  }

  // The other end of the same narrowing: a step above `f32::MAX` becomes
  // `+inf`, which turns the online speech duration into `+inf` for an active
  // slot and `0.0 * inf = NaN` for an inactive one.
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    count: vec![1, 1],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: SlidingWindow::new(0.0, 1e300, 1e300),
    frames_sw: SlidingWindow::new(0.0, 1e300, 1e300),
  };
  let err = refused(parts);
  assert!(
    matches!(err, ExtractError::FrameStepNotRepresentableInF32(w) if w.window().step() == 1e300),
    "expected FrameStepNotRepresentableInF32 for a step above f32::MAX, got {err:?}"
  );
}

#[test]
fn try_from_parts_rejects_an_output_frame_grid_above_the_allocation_cap() {
  // Finding 6, the resource bound. A one-chunk, one-frame extraction whose
  // chunk duration spans fifty million frame steps, with a `count` that
  // MATCHES the derived grid — so the consistency checks are all satisfied and
  // only a bound can refuse it. `diarize_online` would then build two
  // `f64` buffers of that length: measured at 726 MB peak RSS from a 50 MB
  // input before this check existed.
  //
  // THIS TEST COMPLETING WITHOUT THAT ALLOCATION IS THE PROOF: the refusal is
  // O(1) and happens before any grid-sized buffer is touched.
  //
  // `count` is all-zero on purpose: a zero count is supported at every frame, so
  // no consistency check can refuse these parts. The cap is the only thing that
  // does, which is what makes this a falsifier for the cap itself.
  let n = MAX_OUTPUT_FRAMES + 1;
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0],
    count: vec![0; n],
    num_chunks: 1,
    num_frames_per_chunk: 1,
    chunks_sw: SlidingWindow::new(0.0, (n - 1) as f64, 1.0),
    frames_sw: SlidingWindow::new(0.0, 1.0, 1.0),
  };
  let err = refused(parts);
  assert_eq!(err, ExtractError::OutputFrameCountTooLarge(n));

  // One frame BELOW the cap is accepted, so the cap is a boundary and not a
  // blanket refusal of large grids. `count` is the derived one — the single
  // chunk covers output frame 0 with one active slot and nothing else — since
  // check 10 is an equality.
  let n = MAX_OUTPUT_FRAMES;
  let mut count = vec![0u8; n];
  count[0] = 1;
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0],
    count,
    num_chunks: 1,
    num_frames_per_chunk: 1,
    chunks_sw: SlidingWindow::new(0.0, (n - 1) as f64, 1.0),
    frames_sw: SlidingWindow::new(0.0, 1.0, 1.0),
  };
  Extraction::try_from_parts(parts).expect("exactly at the cap is accepted");
}

#[test]
fn try_from_parts_refuses_the_soft_active_slot_rounds_1_and_2_argued_over() {
  // This test has now flipped verdict TWICE, and the second flip retires the
  // premise both earlier versions shared.
  //
  // Round 1 accepted a `count` derived at `onset = 0.5` under a `0.3` column,
  // reasoning that `try_from_parts` takes no `onset` and so cannot know which
  // threshold a producer binarized with. Round 2 kept the parts and tightened
  // the count: NEITHER backend reads an onset — `diarize_online`'s activity scan
  // and dia's `filter_embeddings` both use `seg > 0.0` — so the equality is
  // taken over that shared predicate. Round 2 recorded the choice explicitly,
  // noting that codex's "hard-binary only" alternative would have forbidden a
  // soft column, and preserved it.
  //
  // The premise underneath both was that a soft cell is legitimate input whose
  // only open question is which threshold `count` was derived at. It is not.
  // The BACKENDS — not just the count derivation — read a soft cell
  // incompatibly: offline sums the magnitudes (`filter_embeddings`'
  // `clean_frames`, stage 7's `sum_activity == 0.0`) where everything else
  // booleanizes, and that difference is a speaker count
  // (`a_fractional_segmentation_splits_the_two_backends`). And the capability
  // being preserved had no producer: `Extractor::extract` decodes through
  // `segment::multilabel`'s powerset table and `ArgmaxSource` through the
  // graph's `speaker_ids`, both exactly `{0.0, 1.0}`
  // (`no_producer_can_emit_a_segmentation_cell_the_domain_check_refuses`).
  // Round 2's preservation protected nothing real and admitted a split.
  //
  // So what this test pins now: the exact parts rounds 1 and 2 argued over are
  // REFUSED at the cell, before either round's count question can be asked; the
  // count equality they settled is unchanged on the domain that survives; and
  // the onset question itself is moot there.
  let mut raw = one_usable_slot_row(0);
  raw[EMBEDDING_DIM..EMBEDDING_DIM + 64].fill(1.0); // slot 1: usable too
  let soft = ExtractionParts {
    raw_embeddings: raw,
    // Slot 1's column carries `0.3`: nonzero, below the default `onset` of 0.5.
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    count: vec![1, 1, 0, 0],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new())
      .with_duration(3.0 * crate::audio::speaker::window::FRAME_STEP_S),
    frames_sw: crate::audio::speaker::window::frame_sliding_window(),
  };

  // Round 1 accepted these parts with `count = [1, 0, 0, 0]`; round 2 accepted
  // them with `[1, 1, 0, 0]`. Round 7 accepts neither: the refusal is the CELL.
  for count in [vec![1u8, 1, 0, 0], vec![1, 0, 0, 0]] {
    let err = refused(ExtractionParts {
      count,
      ..soft.clone()
    });
    let ExtractError::NonBinarySegmentation(n) = err else {
      panic!("expected NonBinarySegmentation, got {err:?}")
    };
    // `[c][f][s]`: frame 1, slot 1 = 1 * SEG_NUM_SLOTS + 1.
    assert_eq!(
      (n.index(), n.value(), n.slot()),
      (SEG_NUM_SLOTS + 1, 0.3, 1)
    );
  }

  // Round 2's count equality is untouched on the domain that survives. The same
  // geometry with the column hardened to `1.0` keeps round 2's `[1, 1, 0, 0]`
  // and still names round 1's `[1, 0, 0, 0]` at the frame the column occupies.
  let hard = ExtractionParts {
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ..soft.clone()
  };
  let e = Extraction::try_from_parts(hard.clone()).expect("the hard-binary twin is accepted");
  assert_eq!(e.num_output_frames(), 4);
  let err = refused(ExtractionParts {
    count: vec![1, 0, 0, 0],
    ..hard.clone()
  });
  assert!(
    matches!(err, ExtractError::CountNotSegmentationDerived(c)
      if (c.frame(), c.got(), c.expected()) == (1, 0, 1)),
    "expected CountNotSegmentationDerived(1, 0, 1), got {err:?}"
  );

  // And the onset question rounds 1 and 2 argued over cannot arise on that
  // domain: `seg >= onset` and `seg > 0.0` select the same cells for EVERY
  // onset the type admits, so there is no longer an onset-derived count that
  // differs from the derived one. This is what makes the round-2 wording
  // ("the checks are written against the WEAKEST predicate") a distinction
  // without a difference rather than a load-bearing choice.
  let derived_at = |onset: f32| {
    crate::audio::speaker::window::try_count_from_segmentations(
      &hard.segmentations,
      hard.num_chunks,
      hard.num_frames_per_chunk,
      SEG_NUM_SLOTS,
      onset,
      hard.chunks_sw,
      hard.frames_sw,
    )
    .expect("count")
  };
  for onset in [f32::MIN_POSITIVE, 0.01, 0.3, 0.5, 0.9, 1.0] {
    assert!(
      crate::audio::speaker::window::check_onset(onset),
      "onset={onset} must be VALID for this to prove the question is moot"
    );
    assert_eq!(
      derived_at(onset),
      vec![1u8, 1, 0, 0],
      "onset={onset} must derive the same count as `seg > 0.0` on a hard buffer"
    );
  }

  // The round-1 defect this test also carried survives the hardening: an ACTIVE
  // column over an all-zero row is still the row refusal, not the domain one.
  let broken = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0), // slot 1 back to all-zero
    ..hard
  };
  let err = refused(broken);
  assert!(
    matches!(err, ExtractError::ActiveSlotWithoutEmbedding(a) if (a.chunk(), a.slot()) == (0, 1)),
    "expected ActiveSlotWithoutEmbedding(0, 1), got {err:?}"
  );
}

// =====================================================================
// ROUND 2. Both findings are ONE structural defect: two code paths compute the
// same quantity separately and validation BOUNDED their disagreement instead of
// eliminating it. A bound leaves the unbounded direction open, which is how the
// same defect surfaced twice. Both are now equalities over ONE shared
// calculation.
// =====================================================================

#[test]
fn try_from_parts_requires_the_count_the_segmentations_derive_not_merely_one_they_support() {
  // Finding A. The one-sided `count[t] > supported[t]` check let UNDER-counts
  // through. One standard 589-frame chunk, slot 0 active throughout, a valid
  // slot-0 embedding, the standard sliding windows, and an ALL-ZERO 594-entry
  // `count` was ACCEPTED — and then offline trusted the supplied zeros and
  // emitted no speech (`spans == []`) while online ignored `count`, derived 1
  // from the active clustered slot, and emitted the speaker
  // (`spans == [(0.03096875, 9.97034375, 0)]`). Same `Extraction`,
  // contradictory results. The ONLINE route derives its own count and can never
  // be made to read this field, so the only reachable cure is to remove the
  // caller's freedom in it: `count` must BE the derived count.
  const F: usize = 589;
  let mut segmentations = vec![0.0f64; F * SEG_NUM_SLOTS];
  for f in 0..F {
    segmentations[f * SEG_NUM_SLOTS] = 1.0; // slot 0 active in every frame
  }
  let parts = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations,
    count: vec![0u8; 594],
    num_chunks: 1,
    num_frames_per_chunk: F,
    chunks_sw: crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new()),
    frames_sw: crate::audio::speaker::window::frame_sliding_window(),
  };
  let err = refused(parts.clone());
  let ExtractError::CountNotSegmentationDerived(c) = err else {
    panic!("expected CountNotSegmentationDerived for the all-zero count, got {err:?}")
  };
  assert_eq!((c.frame(), c.got(), c.expected()), (0, 0, 1));

  // And the derived count IS accepted, so this is an equality and not a blanket
  // refusal: the single chunk covers output frames 0..589 with one active slot
  // each, and frames 589..594 are covered by no chunk.
  let mut derived = vec![0u8; 594];
  derived[..F].fill(1);
  let e = Extraction::try_from_parts(ExtractionParts {
    count: derived,
    ..parts
  })
  .expect("the derived count is the one accepted");
  assert_eq!(e.num_output_frames(), 594);

  // With it, the two backends now AGREE that there is speech here — the
  // divergence the under-count created is gone.
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let offline = e
    .diarize_with(
      &plda,
      ClusterBackend::Offline(crate::audio::speaker::cluster::OfflineOptions::new()),
    )
    .expect("offline accepts it");
  let online = e
    .diarize_online(OnlineOptions::new().with_min_speech_duration(0.0))
    .expect("online accepts it");
  assert!(
    !offline.spans_slice().is_empty(),
    "offline must now emit the speaker its segmentations declare"
  );
  assert!(
    !online.spans_slice().is_empty(),
    "online emits the speaker too"
  );
}

#[test]
fn try_from_parts_rejects_misaligned_chunk_placement_even_at_a_zero_origin() {
  // Finding B. The old guard required both window origins to be `0.0`, which
  // does NOT imply aligned frame placement: the reconstruction route adds
  // `frames_sw.duration / 2` to the chunk start and subtracts it again, and
  // that round trip is not the identity in binary floating point.
  //
  // With chunk duration/step `0.04218750000000001` on the community-1 frame
  // grid, chunk 1 maps to frame 3 aggregating (`2.5000000000000004`) and frame
  // 2 reconstructing (`2.5`). These parts passed every check and reconstructed
  // to `[1, 0, 0, 1, 0, 0]`: the real frame-2 activation suppressed, an
  // uncovered frame 3 selected — silently shifted speech.
  let d = 0.04218750000000001_f64;
  let mut raw = vec![0.0f32; 2 * SEG_NUM_SLOTS * EMBEDDING_DIM];
  raw[0..64].fill(1.0); // chunk 0, slot 0
  raw[(SEG_NUM_SLOTS * EMBEDDING_DIM)..(SEG_NUM_SLOTS * EMBEDDING_DIM + 64)].fill(1.0); // chunk 1
  let parts = ExtractionParts {
    raw_embeddings: raw,
    segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    count: vec![1, 0, 0, 1, 0, 0],
    num_chunks: 2,
    num_frames_per_chunk: 1,
    chunks_sw: SlidingWindow::new(0.0, d, d),
    frames_sw: SlidingWindow::new(0.0, 0.0619375, 0.016875),
  };
  let err = refused(parts);
  let ExtractError::MisalignedChunkPlacement(m) = err else {
    panic!("expected MisalignedChunkPlacement at a zero origin, got {err:?}")
  };
  assert_eq!(
    (m.chunk(), m.aggregated(), m.reconstructed()),
    (1, 3, 2),
    "the two mappings disagree about chunk 1"
  );
}

#[test]
fn try_from_parts_accepts_equal_non_zero_origins_that_place_every_chunk_identically() {
  // The other half of finding B: the old zero-origin guard was also
  // OVER-restrictive. Both windows at `(start = 1, duration = 1, step = 1)`
  // cancel exactly — the aggregation places chunk 0 at frame 0 and so does the
  // reconstruction — so there is nothing to refuse. A guard that tested the
  // origins for `0.0` rejected this geometry anyway.
  let sw = SlidingWindow::new(1.0, 1.0, 1.0);
  let e = Extraction::try_from_parts(ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    count: vec![1, 1],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: sw,
    frames_sw: sw,
  })
  .expect("equal, cancelling origins place every chunk identically");
  assert_eq!(e.num_output_frames(), 2);
  assert_eq!(e.chunks_sw().start(), 1.0);
}

// =====================================================================
// The two grids must agree BEFORE an Extraction exists — adversarial
// review round 3, finding 1. `try_from_parts`'s check 8 was correct and
// `extract()` was the defect it exposed: `extract()` assembles through the
// crate-private, UNCHECKED `from_parts`, so it could emit exactly the
// `Extraction` its own public constructor refuses.
// =====================================================================

/// The chunk grid `Extractor::extract` derives from a 160 001-sample clip at
/// `step_samples = 31_995` — through the same three `window` calls `extract`
/// itself makes, so this IS that method's geometry and not a re-derivation.
fn misaligned_extract_geometry() -> (usize, SlidingWindow, SlidingWindow) {
  let w = WindowOptions::new().with_step_samples(31_995);
  let num_chunks = crate::audio::speaker::window::chunk_starts(160_001, &w).len();
  (
    num_chunks,
    crate::audio::speaker::window::chunk_sliding_window(&w),
    crate::audio::speaker::window::frame_sliding_window(),
  )
}

#[test]
fn extract_derives_a_geometry_whose_two_frame_mappings_disagree() {
  // The trigger, reproduced end to end at the geometry layer. `step_samples =
  // 31_995` is odd and `31_995 = 135 * 237` with `237` odd, so chunk 1's
  // aggregation quotient `1 * 31_995 / 270` is exactly `118.5` — a rounding
  // tie. The aggregation computes that `118.5` and banker's-rounds DOWN to
  // 118; the reconstruction's `(chunk_start + duration/2) - duration/2` round
  // trip computes `118.50000000000001` and rounds UP to 119.
  let (num_chunks, chunks_sw, frames_sw) = misaligned_extract_geometry();
  assert_eq!(num_chunks, 2, "160_001 samples over a 31_995 step");
  assert_eq!(chunks_sw.step(), 1.9996875);

  assert_eq!(
    crate::audio::speaker::window::aggregate_chunk_start_frame(
      1,
      chunks_sw.step(),
      frames_sw.step()
    ),
    118,
    "the count aggregation places chunk 1 at frame 118"
  );
  assert_eq!(
    crate::audio::speaker::window::reconstruct_chunk_start_frame(1, chunks_sw, frames_sw),
    119,
    "diaric's reconstruction places the same chunk at frame 119"
  );

  let m = crate::audio::speaker::window::first_misaligned_chunk(num_chunks, chunks_sw, frames_sw)
    .expect("the shared guard must see the same disagreement");
  assert_eq!(
    (m.chunk(), m.aggregated(), m.reconstructed()),
    (1, 118, 119)
  );
}

#[test]
fn a_misaligned_geometry_shifts_the_emitted_span_by_a_whole_frame() {
  // The OBSERVABLE consequence, and the reason refusing is the fix rather than
  // documenting: with the crate-private unchecked assembly — which is what
  // `extract()` used to reach unguarded — the emitted span lands on the frame
  // the COUNT names, not the frame the reconstruction put the activation on.
  //
  // Speaker A holds chunk 0's first 100 frames; two more slots are active in
  // chunk 1's FIRST frame only. `extract()`'s own aggregation puts that frame's
  // count at output frame 118, while `diaric::reconstruct` puts its activations
  // at 119.
  let (num_chunks, chunks_sw, frames_sw) = misaligned_extract_geometry();
  let nf = 589; // the real segmenter's per-chunk frame count
  let mut segmentations = vec![0.0f64; num_chunks * nf * SEG_NUM_SLOTS];
  for f in 0..100 {
    segmentations[f * SEG_NUM_SLOTS] = 1.0;
  }
  segmentations[nf * SEG_NUM_SLOTS + 1] = 1.0;
  segmentations[nf * SEG_NUM_SLOTS + 2] = 1.0;

  // EXACTLY the call `extract()` makes at its step 9-11.
  let count = crate::audio::speaker::window::try_count_from_segmentations(
    &segmentations,
    num_chunks,
    nf,
    SEG_NUM_SLOTS,
    WindowOptions::new().onset(),
    chunks_sw,
    frames_sw,
  )
  .expect("this geometry's output-frame count fits usize");
  assert_eq!(
    (count[118], count[119]),
    (1, 0),
    "the count marks frame 118 and leaves 119 empty"
  );

  // The same overlap-add, placing each chunk where `diaric::reconstruct` does.
  let aligned = {
    let mut agg = vec![0.0f64; count.len()];
    let mut cov = vec![0.0f64; count.len()];
    for c in 0..num_chunks {
      let start =
        crate::audio::speaker::window::reconstruct_chunk_start_frame(c, chunks_sw, frames_sw);
      for f in 0..nf {
        let Ok(t) = usize::try_from(start + f as i64) else {
          continue;
        };
        if t >= count.len() {
          continue;
        }
        agg[t] += segmentations[((c * nf + f) * SEG_NUM_SLOTS)..][..SEG_NUM_SLOTS]
          .iter()
          .filter(|v| **v > 0.0)
          .count() as f64;
        cov[t] += 1.0;
      }
    }
    (0..count.len())
      .map(|t| {
        if cov[t] > 0.0 {
          (agg[t] / cov[t]).round_ties_even() as u8
        } else {
          0
        }
      })
      .collect::<Vec<u8>>()
  };
  assert_eq!(
    (aligned[118], aligned[119]),
    (0, 1),
    "placed the way the activations are, the same frame's count belongs at 119"
  );

  let mut raw_embeddings = vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM];
  raw_embeddings[..64].fill(1.0);
  let c1 = SEG_NUM_SLOTS * EMBEDDING_DIM;
  raw_embeddings[c1 + EMBEDDING_DIM..c1 + EMBEDDING_DIM + 64].fill(-1.0);
  for k in 0..64 {
    raw_embeddings[c1 + 2 * EMBEDDING_DIM + 2 * k] = 1.0;
  }

  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let span_start = |count: Vec<u8>| -> f64 {
    Extraction::from_parts(
      raw_embeddings.clone(),
      segmentations.clone(),
      count,
      num_chunks,
      nf,
      chunks_sw,
      frames_sw,
    )
    .diarize_with(&plda, ClusterBackend::default())
    .expect("both counts diarize")
    .spans_slice()
    .last()
    .expect("a trailing span for chunk 1's lone active frame")
    .start()
  };
  let shifted = span_start(count.clone());
  let honest = span_start(aligned);
  assert!(
    (honest - shifted - frames_sw.step()).abs() < 1e-12,
    "the count's placement moves the emitted span a whole frame step earlier: \
     {shifted} vs {honest}"
  );

  // Which is why BOTH public paths must refuse this geometry outright.
  let err = refused(ExtractionParts {
    raw_embeddings,
    segmentations,
    count,
    num_chunks,
    num_frames_per_chunk: nf,
    chunks_sw,
    frames_sw,
  });
  assert!(
    matches!(err, ExtractError::MisalignedChunkPlacement(m)
      if (m.chunk(), m.aggregated(), m.reconstructed()) == (1, 118, 119)),
    "expected MisalignedChunkPlacement(1, 118, 119), got {err:?}"
  );
}

#[test]
fn every_shipping_extract_geometry_places_its_chunks_identically() {
  // The other half: the guard must not refuse a geometry the crate actually
  // ships. A tie needs `c * step_samples` to be an odd multiple of 135, which
  // an EVEN `step_samples` can never be — and the default (16 000) and argmax's
  // stride are both even. Swept over chunk counts far past any real clip.
  let w = WindowOptions::new();
  assert_eq!(w.step_samples() % 2, 0, "the default step is even");
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&w);
  let frames_sw = crate::audio::speaker::window::frame_sliding_window();
  assert_eq!(
    crate::audio::speaker::window::first_misaligned_chunk(100_000, chunks_sw, frames_sw),
    None,
    "the default geometry must survive 100 000 chunks (~27.7 h of audio)"
  );

  // And every EVEN step across the supported range, over a chunk count far
  // past the first tie any ODD step reaches (the smallest tying chunk index is
  // `135 / gcd(step_samples, 135)`, at most 135).
  for step in (2..=SEG_CHUNK_SAMPLES as u32).step_by(1_998) {
    assert_eq!(step % 2, 0, "the sweep must stay on even steps");
    let w = WindowOptions::new().with_step_samples(step);
    let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&w);
    assert_eq!(
      crate::audio::speaker::window::first_misaligned_chunk(4_096, chunks_sw, frames_sw),
      None,
      "even step_samples={step} must never tie"
    );
  }

  // The complement, so the sweep above is a real discrimination and not a
  // vacuous pass: the reviewer's own odd step DOES tie, at chunk 1.
  let odd = crate::audio::speaker::window::chunk_sliding_window(
    &WindowOptions::new().with_step_samples(31_995),
  );
  assert!(
    crate::audio::speaker::window::first_misaligned_chunk(2, odd, frames_sw).is_some(),
    "step_samples=31_995 must still be caught"
  );
}

#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn extract_refuses_a_geometry_whose_two_frame_mappings_disagree() {
  // The wiring, on the real method. `extract` assembles through the
  // crate-private `from_parts`, which validates nothing, so WITHOUT its own
  // guard this call returns `Ok` with the `Extraction`
  // `Extraction::try_from_parts` rejects — the shifted span the hermetic test
  // above measures. The guard runs before any inference, so this refusal costs
  // no model time.
  let seg = load_seg_model();
  let embed = load_embed_model();
  let options = Options::new().with_window(WindowOptions::new().with_step_samples(31_995));
  // Not `expect_err`: on the failing (accepted) side that renders the whole
  // 712-frame `Extraction` and buries the one fact this falsifier reports.
  match Extractor::with_options(options).extract(&seg, &embed, &vec![0.0f32; 160_001]) {
    Err(ExtractError::MisalignedChunkPlacement(m)) => {
      assert_eq!(
        (m.chunk(), m.aggregated(), m.reconstructed()),
        (1, 118, 119)
      );
    }
    Err(other) => panic!("expected MisalignedChunkPlacement(1, 118, 119), got {other:?}"),
    Ok(e) => panic!(
      "extract ACCEPTED a misaligned geometry: {} chunks on chunks_sw={:?}, which the count \
       aggregation places at frame {} and diaric's reconstruction at frame {}",
      e.num_chunks(),
      e.chunks_sw(),
      crate::audio::speaker::window::aggregate_chunk_start_frame(
        1,
        e.chunks_sw().step(),
        e.frames_sw().step()
      ),
      crate::audio::speaker::window::reconstruct_chunk_start_frame(1, e.chunks_sw(), e.frames_sw()),
    ),
  }

  // The shipping default over the identical clip stays accepted.
  Extractor::new()
    .extract(&seg, &embed, &vec![0.0f32; 160_001])
    .expect("the default geometry places every chunk identically");
}

// =====================================================================
// An active slot's row must clear PLDA's floor, not just the online
// engine's — adversarial review round 3, finding 2.
// =====================================================================

#[test]
fn try_from_parts_rejects_an_active_row_below_plda_norm_that_only_online_tolerates() {
  // The trigger: one default-geometry chunk, slot 0 active in all 589 frames,
  // and a slot-0 row of `[0.005, 0.0, …]` — norm 0.005. That is nine orders of
  // magnitude ABOVE `Embedding::normalize_from`'s `1e-12` floor (so check 9's
  // old predicate accepted it) and BELOW PLDA's 0.01 (so the offline backend
  // refuses the whole extraction). The two backends therefore disagree about
  // the identical `Extraction`, which is exactly what this constructor exists
  // to prevent.
  let w = WindowOptions::new();
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&w);
  let frames_sw = crate::audio::speaker::window::frame_sliding_window();
  let nf = 589;
  let mut segmentations = vec![0.0f64; nf * SEG_NUM_SLOTS];
  for f in 0..nf {
    segmentations[f * SEG_NUM_SLOTS] = 1.0;
  }
  let count = crate::audio::speaker::window::try_count_from_segmentations(
    &segmentations,
    1,
    nf,
    SEG_NUM_SLOTS,
    w.onset(),
    chunks_sw,
    frames_sw,
  )
  .expect("this geometry's output-frame count fits usize");

  let mut raw_embeddings = vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM];
  raw_embeddings[0] = 0.005;

  // Both halves of the divergence, proved against the engines themselves.
  let mut row = [0.0f32; EMBEDDING_DIM];
  row.copy_from_slice(&raw_embeddings[..EMBEDDING_DIM]);
  assert!(
    diaric::embed::Embedding::normalize_from(row).is_some(),
    "the online engine's own test ACCEPTS this row — matching it is the defect"
  );
  assert!(
    matches!(
      diaric::plda::RawEmbedding::from_wespeaker(row),
      Err(diaric::plda::Error::DegenerateInput)
    ),
    "PLDA's raw boundary refuses it at the 0.01 floor"
  );

  let parts = ExtractionParts {
    raw_embeddings,
    segmentations,
    count,
    num_chunks: 1,
    num_frames_per_chunk: nf,
    chunks_sw,
    frames_sw,
  };

  // Assembled unchecked, the two backends split: offline fails outright, online
  // normalizes the row and manufactures a ~9.94 s speaker.
  let unchecked = Extraction::from_parts(
    parts.raw_embeddings.clone(),
    parts.segmentations.clone(),
    parts.count.clone(),
    parts.num_chunks,
    parts.num_frames_per_chunk,
    parts.chunks_sw,
    parts.frames_sw,
  );
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  assert!(
    matches!(
      unchecked.diarize_with(&plda, ClusterBackend::default()),
      Err(diaric::offline::Error::Plda(
        diaric::plda::Error::DegenerateInput
      ))
    ),
    "offline must fail on this row"
  );
  let online = unchecked
    .diarize_online(OnlineOptions::new())
    .expect("online accepts it");
  assert_eq!(
    online.spans_slice().len(),
    1,
    "online manufactures a speaker from the same row"
  );

  let err = refused(parts);
  assert!(
    matches!(err, ExtractError::ActiveSlotWithoutEmbedding(a) if (a.chunk(), a.slot()) == (0, 0)),
    "expected ActiveSlotWithoutEmbedding(0, 0), got {err:?}"
  );
}

#[test]
fn the_active_row_floor_is_the_one_both_in_crate_producers_drop_at() {
  // The threshold is not this constructor's own: `Extractor::extract` and the
  // argmax source both DROP a slot whose row falls below it, and `diaric`
  // re-applies the identical number at the PLDA boundary. All three read
  // `raw_embedding_reaches_plda`, so this pins the value they share rather than
  // adding a fourth copy of it.
  assert_eq!(PLDA_MIN_NORM, 0.01);

  let plda = shared_plda_transform().expect("diaric's PLDA weights are embedded");

  // Agreement with PLDA's own admission test, straddling the floor. Equality
  // over the whole sweep is the property: a floor that drifts in either
  // direction shows up as a row one side keeps and the other refuses.
  //
  // The predicate's third clause (`PldaTransform::project`) is inert across
  // this band, and the equality itself is what proves it rather than a comment
  // claiming so: a row `project` refused would make `mine` false where
  // `from_wespeaker` says true, i.e. a disagreement. (The reason it is inert:
  // a single-component row of magnitude ~0.01 sits ~1.42 from `mean1`,
  // fourteen times PLDA's `0.1` centered floor.) So this sweep still isolates
  // the RAW floor exactly as it did before the projection clause existed.
  let mut disagreements = 0;
  for micro in 9_000..11_000u32 {
    let v = f64::from(micro) / 1_000_000.0;
    let mut row = [0.0f32; EMBEDDING_DIM];
    row[0] = v as f32;
    let mine = raw_embedding_reaches_plda(plda, &row);
    let admitted = diaric::plda::RawEmbedding::from_wespeaker(row).is_ok();
    if mine != admitted {
      disagreements += 1;
    }
  }
  assert_eq!(
    disagreements, 0,
    "the crate's row predicate and PLDA's own boundary must admit the same rows"
  );

  // And the floor really is 0.01, not the online engine's 1e-12: a row between
  // the two is refused here while `normalize_from` still accepts it.
  let mut between = [0.0f32; EMBEDDING_DIM];
  between[0] = 0.005;
  assert!(!raw_embedding_reaches_plda(plda, &between));
  assert!(diaric::embed::Embedding::normalize_from(between).is_some());

  let mut above = [0.0f32; EMBEDDING_DIM];
  above[0] = 0.02;
  assert!(raw_embedding_reaches_plda(plda, &above));

  // Non-finite is refused by the same predicate, matching `from_raw_array`'s
  // own leading finiteness scan — a `+inf` row has an INFINITE norm, which a
  // bare `norm >= floor` comparison would have admitted.
  let mut infinite = above;
  infinite[1] = f32::INFINITY;
  assert!(!raw_embedding_reaches_plda(plda, &infinite));
  assert!(diaric::plda::RawEmbedding::from_wespeaker(infinite).is_err());
  let mut nan = above;
  nan[1] = f32::NAN;
  assert!(!raw_embedding_reaches_plda(plda, &nan));
  assert!(diaric::plda::RawEmbedding::from_wespeaker(nan).is_err());
}

// =====================================================================
// An active row must clear BOTH backends' own admission tests, not a
// hand-written approximation of their intersection — adversarial review
// round 4.
// =====================================================================

#[test]
fn try_from_parts_rejects_an_active_row_whose_norm_overflows_f32_for_the_online_engine() {
  // The trigger: one default-geometry chunk, slot 0 active in 60 frames, and a
  // slot-0 row of `[f32::MAX, f32::MAX, 0.0, …]`. Every element is finite and
  // the f64 L2 norm is ~4.81e38 — forty orders of magnitude ABOVE
  // `PLDA_MIN_NORM`, so a `f64` norm floor accepts it and so does PLDA's own
  // raw boundary. `Embedding::normalize_from` narrows that norm to f32 FIRST,
  // where 4.81e38 is `+inf`, and returns `None` — `diarize_online`'s
  // dropped-slot sentinel. The active slot yields no speaker at all.
  let w = WindowOptions::new();
  let chunks_sw = crate::audio::speaker::window::chunk_sliding_window(&w);
  let frames_sw = crate::audio::speaker::window::frame_sliding_window();
  let nf = 60;
  let mut segmentations = vec![0.0f64; nf * SEG_NUM_SLOTS];
  for f in 0..nf {
    segmentations[f * SEG_NUM_SLOTS] = 1.0;
  }
  let count = crate::audio::speaker::window::try_count_from_segmentations(
    &segmentations,
    1,
    nf,
    SEG_NUM_SLOTS,
    w.onset(),
    chunks_sw,
    frames_sw,
  )
  .expect("this geometry's output-frame count fits usize");

  let mut raw_embeddings = vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM];
  raw_embeddings[0] = f32::MAX;
  raw_embeddings[1] = f32::MAX;

  // The split, proved against the two engines themselves and against the
  // arithmetic that produced four rounds of approximations.
  let mut row = [0.0f32; EMBEDDING_DIM];
  row.copy_from_slice(&raw_embeddings[..EMBEDDING_DIM]);
  let f64_norm: f64 = row
    .iter()
    .map(|v| f64::from(*v) * f64::from(*v))
    .sum::<f64>()
    .sqrt();
  assert!(
    f64_norm > PLDA_MIN_NORM && f64_norm.is_finite(),
    "the f64 norm ({f64_norm:e}) is finite and far above the floor — a f64 \
     comparison accepts this row"
  );
  assert!(
    !(f64_norm as f32).is_finite(),
    "and the SAME norm is +inf once narrowed to f32, which is what \
     `normalize_from` compares"
  );
  assert!(
    diaric::plda::RawEmbedding::from_wespeaker(row).is_ok(),
    "PLDA's raw boundary ACCEPTS this row — matching it alone is the defect"
  );
  assert!(
    diaric::embed::Embedding::normalize_from(row).is_none(),
    "the online engine's own test REFUSES it, and `None` is its dropped-slot \
     sentinel"
  );

  let parts = ExtractionParts {
    raw_embeddings,
    segmentations,
    count,
    num_chunks: 1,
    num_frames_per_chunk: nf,
    chunks_sw,
    frames_sw,
  };

  // Assembled unchecked, `diarize_online` returns Ok with the speech GONE:
  // 60 active frames, no span.
  let unchecked = Extraction::from_parts(
    parts.raw_embeddings.clone(),
    parts.segmentations.clone(),
    parts.count.clone(),
    parts.num_chunks,
    parts.num_frames_per_chunk,
    parts.chunks_sw,
    parts.frames_sw,
  );
  let online = unchecked
    .diarize_online(OnlineOptions::new())
    .expect("online returns Ok");
  assert_eq!(
    online.spans_slice().len(),
    0,
    "online silently drops the active slot — the failure this constructor exists \
     to make impossible"
  );

  let err = refused(parts);
  assert!(
    matches!(err, ExtractError::ActiveSlotWithoutEmbedding(a) if (a.chunk(), a.slot()) == (0, 0)),
    "expected ActiveSlotWithoutEmbedding(0, 0), got {err:?}"
  );
}

#[test]
fn the_row_predicate_is_the_two_backend_functions_not_a_description_of_them() {
  // Equality with the CONJUNCTION over rows that straddle every corner four
  // rounds of approximations were caught on. This is the property the fix
  // makes structural: the predicate cannot disagree with a backend, because it
  // is the two backends' own calls.
  let mut probes: Vec<[f32; EMBEDDING_DIM]> = Vec::new();
  let mut push = |first: f32, second: f32| {
    let mut row = [0.0f32; EMBEDDING_DIM];
    row[0] = first;
    row[1] = second;
    probes.push(row);
  };
  // Zero, the dropped-slot row every producer writes.
  push(0.0, 0.0);
  // Subnormal, and the smallest normal.
  push(f32::from_bits(1), 0.0);
  push(f32::MIN_POSITIVE, 0.0);
  // Below / at / above the online engine's `NORM_EPSILON` (1e-12).
  push(1e-13, 0.0);
  push(1e-12, 0.0);
  push(1e-11, 0.0);
  // Between the two floors — the round-3 corner.
  push(0.005, 0.0);
  // Straddling PLDA's floor.
  push(0.009_999, 0.0);
  push(0.01, 0.0);
  push(0.010_001, 0.0);
  // In distribution.
  push(2.07, 0.0);
  // Large but f32-representable norm, and norms that overflow f32 — the
  // round-4 corner, approached from both sides of the narrowing.
  push(1e19, 1e19);
  push(f32::MAX, 0.0);
  push(f32::MAX, f32::MAX);
  push(f32::MAX, f32::MAX / 2.0);
  // Non-finite, in either position, and signed.
  push(f32::INFINITY, 0.0);
  push(f32::NEG_INFINITY, 0.0);
  push(f32::NAN, 0.0);
  push(2.07, f32::NAN);
  push(-2.07, 0.0);
  // The round-6 corner: `mean1` cast to f32. Both ADMISSION functions take it
  // (raw norm 1.42) and the PROJECTION that follows does not, so it is what
  // makes the third clause below non-vacuous.
  probes.push(diaric_mean1_as_f32());

  let plda = shared_plda_transform().expect("diaric's PLDA weights are embedded");
  // The three stages, called separately, to compare against the predicate that
  // composes them. `projects` runs on the row's OWN `RawEmbedding`, so it is
  // `false` for a row the raw boundary already refused.
  let projects = |row: &[f32; EMBEDDING_DIM]| {
    diaric::plda::RawEmbedding::from_wespeaker(*row).is_ok_and(|raw| plda.project(&raw).is_ok())
  };

  for row in &probes {
    let online = diaric::embed::Embedding::normalize_from(*row).is_some();
    let offline = diaric::plda::RawEmbedding::from_wespeaker(*row).is_ok();
    let projected = projects(row);
    assert_eq!(
      raw_embedding_reaches_plda(plda, row),
      online && offline && projected,
      "row [{}, {}, 0, …] — online accepts {online}, offline accepts {offline}, \
       projection accepts {projected}",
      row[0],
      row[1]
    );
  }

  // Both single-sided corners are actually present in the probe set, so the
  // equality above is not vacuous: neither backend's test alone is this
  // predicate.
  assert!(
    probes.iter().any(|r| {
      diaric::embed::Embedding::normalize_from(*r).is_some()
        && diaric::plda::RawEmbedding::from_wespeaker(*r).is_err()
    }),
    "the probe set must contain a row ONLY the online engine accepts"
  );
  assert!(
    probes.iter().any(|r| {
      diaric::embed::Embedding::normalize_from(*r).is_none()
        && diaric::plda::RawEmbedding::from_wespeaker(*r).is_ok()
    }),
    "the probe set must contain a row ONLY the offline boundary accepts"
  );
  assert!(
    probes.iter().any(|r| {
      diaric::embed::Embedding::normalize_from(*r).is_some()
        && diaric::plda::RawEmbedding::from_wespeaker(*r).is_ok()
        && !projects(r)
    }),
    "the probe set must contain a row BOTH admission functions accept and the \
     PROJECTION refuses — otherwise the third clause is vacuous"
  );

  // A wrong-length row is refused rather than panicked on. Unreachable in
  // crate — every call site slices exactly `EMBEDDING_DIM` — but the array
  // conversion is what makes it so, and this pins which way it fails.
  assert!(!raw_embedding_reaches_plda(
    plda,
    &[1.0f32; EMBEDDING_DIM - 1]
  ));
  assert!(!raw_embedding_reaches_plda(
    plda,
    &[1.0f32; EMBEDDING_DIM + 1]
  ));
}

#[test]
fn plda_min_norm_is_diarics_own_floor_measured_not_copied() {
  // `PLDA_MIN_NORM` is published but no longer READ by anything in-crate, so
  // nothing would notice it drifting away from the number `diaric` enforces.
  // Measure that number instead of restating it: binary-search
  // `RawEmbedding::from_wespeaker` for the smallest single-component row it
  // accepts, and require the constant to name that boundary.
  //
  // A single-component row's f64 norm is exactly `|v|`, so the search is over
  // the norm itself.
  let admits = |v: f32| {
    let mut row = [0.0f32; EMBEDDING_DIM];
    row[0] = v;
    diaric::plda::RawEmbedding::from_wespeaker(row).is_ok()
  };
  let (mut lo, mut hi) = (0.0f32, 1.0f32);
  assert!(!admits(lo) && admits(hi), "the floor lies inside (0, 1]");
  for _ in 0..200 {
    let mid = f32::from_bits(lo.to_bits().midpoint(hi.to_bits()));
    if mid == lo || mid == hi {
      break;
    }
    if admits(mid) { hi = mid } else { lo = mid }
  }
  assert!(
    f64::from(lo) < PLDA_MIN_NORM && f64::from(hi) >= PLDA_MIN_NORM,
    "diaric's measured raw-embedding floor is in ({lo:e}, {hi:e}] but \
     PLDA_MIN_NORM says {PLDA_MIN_NORM:e} — the published constant no longer \
     names the number diaric enforces"
  );
  // The two f32 neighbours really do straddle it, so the bracket is tight.
  assert_eq!(hi.to_bits() - lo.to_bits(), 1, "search did not converge");
}

// =====================================================================
// The ONE transform the row predicate projects against — adversarial
// review round 6.
// =====================================================================

#[test]
fn plda_transform_is_available() {
  // `shared_plda_transform`'s and `ExtractError::PldaTransformUnavailable`'s
  // docs both claim `PldaTransform::new()` cannot fail today. Pin it: the
  // constructor takes no arguments and decodes `include_bytes!`d blobs, so
  // "cannot fail" is a property of the shipped dependency, and if a future
  // `diaric` adds a fallible step this is where it surfaces rather than as
  // every active slot suddenly being refused.
  let mine = shared_plda_transform().expect("diaric's PLDA weights are compile-time embedded");

  // Cached, not rebuilt: the same `&'static` every call. This is what makes
  // resolving it once per `extract` / `try_from_parts` — rather than once per
  // row — a choice about WHERE the ~0.15 ms is paid and not WHETHER.
  assert!(
    std::ptr::eq(
      mine,
      shared_plda_transform().expect("the cached transform is still there")
    ),
    "shared_plda_transform must hand out one process-wide transform"
  );

  // And it is the SAME transform a caller can hand `diarize_with`, which is
  // the claim that makes validating against a cached one meaningful:
  // `PldaTransform::new()` is diaric's only public constructor and takes no
  // arguments, so a caller's transform cannot differ from this one. Proved
  // through behaviour rather than asserted — `PldaTransform` is opaque.
  let theirs = diaric::plda::PldaTransform::new().expect("a caller's own transform");
  assert_eq!(
    mine.phi(),
    theirs.phi(),
    "the eigenvalue diagonal must match"
  );
  let mut row = [0.0f32; EMBEDDING_DIM];
  row[0] = 2.07; // in-distribution raw norm
  row[1] = -0.5;
  let raw = diaric::plda::RawEmbedding::from_wespeaker(row).expect("an in-distribution row");
  assert_eq!(
    mine
      .project(&raw)
      .expect("the cached transform projects it"),
    theirs
      .project(&raw)
      .expect("the caller's transform projects it"),
    "the cached transform must project exactly as a caller's own does"
  );
}

// =====================================================================
// An INACTIVE slot must not be able to move the online clusterer's
// state — adversarial review round 6, finding 1.
// =====================================================================

/// The round-6 witness geometry: three unit chunks of ONE frame each, slot 0
/// active in chunks 0 and 2 and INACTIVE in chunk 1, and `slot0_rows` supplying
/// chunk `c`'s slot-0 raw row. Every other row is the all-zero row a dropped
/// slot carries.
fn three_chunk_slot0_parts(slot0_rows: [[f32; EMBEDDING_DIM]; 3]) -> ExtractionParts {
  let mut raw_embeddings = vec![0.0f32; 3 * SEG_NUM_SLOTS * EMBEDDING_DIM];
  for (c, row) in slot0_rows.iter().enumerate() {
    raw_embeddings[embedding_range(c, 0)].copy_from_slice(row);
  }
  ExtractionParts {
    raw_embeddings,
    // [f][s] per chunk, one frame per chunk: chunk 1's whole column is zero.
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    // Chunk `c` lands at output frame `c`; frame 3 is covered by no chunk.
    count: vec![1, 0, 1, 0],
    num_chunks: 3,
    num_frames_per_chunk: 1,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  }
}

/// A unit row at `deg` degrees in the `(0, 1)` plane.
fn planar_row(deg: f64) -> [f32; EMBEDDING_DIM] {
  let mut row = [0.0f32; EMBEDDING_DIM];
  row[0] = deg.to_radians().cos() as f32;
  row[1] = deg.to_radians().sin() as f32;
  row
}

#[test]
fn an_inactive_slots_row_cannot_change_the_online_result() {
  // Finding 1 (round 6). `try_from_parts` deliberately admits an INACTIVE slot
  // carrying a usable row — but `diarize_online` used to NORMALIZE and ASSIGN
  // that row before it computed the zero activity that would have gated it.
  // `OnlineClusterer::assign` matches the nearest centroid and UPDATES it
  // BEFORE the `min_speech_duration` gate is consulted at all (its step 3 runs
  // `update_existing`; the duration gate is step 4, reached only when nothing
  // matched), so a zero-duration row still moves a centroid.
  //
  // A: 0 deg seeds speaker 1. B: 55 deg, under the inactive column, is 0.426
  //    from A — inside the default 0.65 threshold, so it matches and drags the
  //    centroid toward itself.
  // C: -50 deg, 0.357 from the UNPOLLUTED A, is ~0.829 from the polluted
  //    centroid — outside 0.65, so it spawns a SECOND speaker.
  //
  // The property, stated without reference to the mechanism: zeroing an
  // inactive slot's row — the all-zero row both in-crate producers write into
  // every dropped slot — must not change what `diarize_online` returns.
  let a = planar_row(0.0);
  let b = planar_row(55.0);
  let c = planar_row(-50.0);
  let zero = [0.0f32; EMBEDDING_DIM];

  let with_row = Extraction::try_from_parts(three_chunk_slot0_parts([a, b, c]))
    .expect("an inactive slot carrying a usable row is admitted by construction");
  let without_row = Extraction::try_from_parts(three_chunk_slot0_parts([a, zero, c]))
    .expect("the same parts with the dropped slot's row zeroed");

  let clusters = |e: &Extraction| -> Vec<(u64, u64, usize)> {
    e.diarize_online(OnlineOptions::new())
      .expect("this geometry clusters")
      .spans_slice()
      .iter()
      .map(|s| (s.start().to_bits(), s.end().to_bits(), s.cluster()))
      .collect()
  };
  assert_eq!(
    clusters(&with_row),
    clusters(&without_row),
    "an INACTIVE slot's raw-embedding row changed the online clustering"
  );

  // The offline route already ignores that row: `filter_embeddings` needs
  // `clean_frames >= 0.2 * num_frames_per_chunk` and an all-zero column sums to
  // zero, so chunk 1 slot 0 never reaches PLDA. Asserted, not assumed — it is
  // what makes the online skip SUFFICIENT and a constructor refusal
  // unnecessary.
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let spans = |e: &Extraction| -> Vec<(u64, u64, usize)> {
    e.diarize_with(&plda, ClusterBackend::default())
      .expect("this geometry clusters offline")
      .spans_slice()
      .iter()
      .map(|s| (s.start().to_bits(), s.end().to_bits(), s.cluster()))
      .collect()
  };
  assert_eq!(
    spans(&with_row),
    spans(&without_row),
    "the OFFLINE route was supposed to be blind to an inactive slot's row"
  );
}

// =====================================================================
// An active row must clear the WHOLE offline chain, projection
// included — adversarial review round 6, finding 2.
// =====================================================================

/// `diaric`'s shipped `models/plda/mean1.bin`, decoded little-endian `f64` and
/// cast elementwise to `f32` — the centering mean `PldaTransform::xvec_transform`
/// subtracts, quantized to the precision a WeSpeaker row is carried in.
///
/// Read from the dependency's own source tree rather than committed here: the
/// weights are pyannote's CC-BY-4.0 community-1 PLDA (see this repository's
/// `NOTICE`, section 4), and `NOTICE` records that the repository redistributes
/// exactly ONE model. Reading the bytes `diaric` already `include_bytes!`es
/// into this very test binary is use, not redistribution.
///
/// `mean1` is private in `diaric` with no accessor and is NOT recoverable from
/// its public API: every public read of it goes through `xvec_transform`, which
/// exposes only `normalize(lda.T @ n - mean2)` for the *unit direction*
/// `n = (x - mean1)/‖x - mean1‖` — a map through a 128x256 matrix that discards
/// 128 dimensions and normalizes away the scale. So the shipped bytes are the
/// only source, and `cargo metadata` is how a test finds them wherever Cargo
/// put the crate (registry, vendor directory, or path override).
///
/// # Panics
/// Loudly, if the dependency's blob cannot be located or decoded. A witness
/// that quietly stopped being a witness is the defect this test exists for.
fn diaric_mean1_as_f32() -> [f32; EMBEDDING_DIM] {
  let cargo = option_env!("CARGO").unwrap_or("cargo");
  let out = std::process::Command::new(cargo)
    .args([
      "metadata",
      "--format-version",
      "1",
      "--all-features",
      "--manifest-path",
      concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
    ])
    .output()
    .expect("`cargo metadata` must run: this test reads diaric's shipped PLDA weights");
  assert!(
    out.status.success(),
    "cargo metadata failed: {}",
    String::from_utf8_lossy(&out.stderr)
  );
  let meta: serde_json::Value =
    serde_json::from_slice(&out.stdout).expect("cargo metadata emits JSON");
  let manifest = meta["packages"]
    .as_array()
    .expect("metadata.packages is an array")
    .iter()
    .find(|p| p["name"] == "diaric")
    .and_then(|p| p["manifest_path"].as_str())
    .expect("diaric is an --all-features dependency of this crate");
  let blob = std::path::Path::new(manifest)
    .parent()
    .expect("a manifest path has a parent")
    .join("models/plda/mean1.bin");
  let bytes =
    std::fs::read(&blob).unwrap_or_else(|e| panic!("diaric ships {}: {e}", blob.display()));
  assert_eq!(
    bytes.len(),
    EMBEDDING_DIM * 8,
    "mean1.bin is {EMBEDDING_DIM} little-endian f64 values"
  );
  let mut row = [0.0f32; EMBEDDING_DIM];
  for (i, slot) in row.iter_mut().enumerate() {
    let mut le = [0u8; 8];
    le.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
    *slot = f64::from_le_bytes(le) as f32;
  }
  row
}

#[test]
fn try_from_parts_rejects_an_active_row_plda_projection_refuses() {
  // Finding 2 (round 6). `RawEmbedding::from_wespeaker` is only the offline
  // route's RAW-INPUT boundary; `PldaTransform::project` runs after it and
  // rejects again, on `‖row - mean1‖ < XVEC_CENTERED_MIN_NORM` (0.1). The
  // f32 cast of `mean1` itself sits 3.5e-8 from the centre of that ball while
  // its RAW norm is 1.42 — forty times PLDA's 0.01 raw floor and far above the
  // online engine's 1e-12 — so both admission functions accept it and the
  // projection that follows does not.
  let row = diaric_mean1_as_f32();
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");

  // The witness's three properties, proved against the engines themselves so
  // that a `diaric` weight change turns it into a loud failure rather than a
  // quietly vacuous test.
  assert!(
    diaric::embed::Embedding::normalize_from(row).is_some(),
    "the ONLINE engine accepts this row"
  );
  let raw = diaric::plda::RawEmbedding::from_wespeaker(row)
    .expect("PLDA's RAW boundary accepts this row (norm ~1.42)");
  assert!(
    matches!(
      plda.project(&raw),
      Err(diaric::plda::Error::DegenerateInput)
    ),
    "the PROJECTION that follows must refuse it — otherwise this is not a witness"
  );

  let mut raw_embeddings = vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM];
  raw_embeddings[..EMBEDDING_DIM].copy_from_slice(&row);
  let parts = ExtractionParts {
    raw_embeddings,
    segmentations: vec![1.0, 0.0, 0.0],
    count: vec![1, 0],
    num_chunks: 1,
    num_frames_per_chunk: 1,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  };

  // Assembled unchecked, the two backends split: offline fails the WHOLE
  // extraction, online manufactures a speaker.
  let unchecked = Extraction::from_parts(
    parts.raw_embeddings.clone(),
    parts.segmentations.clone(),
    parts.count.clone(),
    parts.num_chunks,
    parts.num_frames_per_chunk,
    parts.chunks_sw,
    parts.frames_sw,
  );
  assert!(
    matches!(
      unchecked.diarize_with(&plda, ClusterBackend::default()),
      Err(diaric::offline::Error::Plda(
        diaric::plda::Error::DegenerateInput
      ))
    ),
    "offline must fail on this row"
  );
  assert_eq!(
    unchecked
      .diarize_online(OnlineOptions::new())
      .expect("online accepts it")
      .spans_slice()
      .len(),
    1,
    "online manufactures a speaker from the same row"
  );

  let err = refused(parts);
  assert!(
    matches!(err, ExtractError::ActiveSlotWithoutEmbedding(a) if (a.chunk(), a.slot()) == (0, 0)),
    "expected ActiveSlotWithoutEmbedding(0, 0), got {err:?}"
  );
}

// =====================================================================
// Every raw_embeddings value must be finite, including under an
// INACTIVE column — the residual round 6 named and left open.
// =====================================================================

/// One unit chunk of one frame: slot 0 ACTIVE with a usable row, slots 1 and 2
/// INACTIVE with all-zero columns — the shape both in-crate producers write into
/// a dropped slot. `poison` places one value at `(slot, dimension)`.
fn one_chunk_with_poisoned_slot(slot: usize, dimension: usize, value: f32) -> ExtractionParts {
  let mut raw_embeddings = one_usable_slot_row(0);
  raw_embeddings[embedding_range(0, slot)][dimension] = value;
  ExtractionParts {
    raw_embeddings,
    // [f][s], one frame: only slot 0 is active.
    segmentations: vec![1.0, 0.0, 0.0],
    count: vec![1, 0],
    num_chunks: 1,
    num_frames_per_chunk: 1,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  }
}

#[test]
fn try_from_parts_rejects_a_non_finite_row_under_an_inactive_column() {
  // The gap round 6 enumerated and left open. Check 9 tests the row of every
  // ACTIVE slot; an INACTIVE slot has no active column to bring its row there,
  // so a NaN one buffer position away was accepted — and the two backends then
  // read it in opposite directions:
  //
  //   OFFLINE: `assign_embeddings` scans the WHOLE embedding matrix — train
  //   subset or not, active or not, because stage 6 cosine-scores every row —
  //   and fails the extraction with `NonFiniteField::Embeddings`.
  //   ONLINE: `diarize_online` skips an inactive column before it copies the
  //   row, so the value is never read and the call returns `Ok`.
  //
  // Fatal to one engine, invisible to the other, for the identical
  // `Extraction` — exactly the class this constructor exists to refuse.
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");

  for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    let parts = one_chunk_with_poisoned_slot(2, 0, bad);

    // The split, proved against the engines themselves rather than described.
    // Assembled through the crate-private `from_parts`, which is what the
    // constructor guards against.
    let unchecked = Extraction::from_parts(
      parts.raw_embeddings.clone(),
      parts.segmentations.clone(),
      parts.count.clone(),
      parts.num_chunks,
      parts.num_frames_per_chunk,
      parts.chunks_sw,
      parts.frames_sw,
    );
    assert!(
      matches!(
        unchecked.diarize_with(&plda, ClusterBackend::default()),
        Err(diaric::offline::Error::Pipeline(
          diaric::pipeline::Error::NonFinite(diaric::pipeline::error::NonFiniteField::Embeddings)
        ))
      ),
      "offline must fail the whole extraction on {bad:?} in an inactive slot's row, got {:?}",
      unchecked.diarize_with(&plda, ClusterBackend::default())
    );
    assert_eq!(
      unchecked
        .diarize_online(OnlineOptions::new())
        .expect("online never reads an inactive slot's row, so it returns Ok")
        .spans_slice()
        .len(),
      1,
      "online must be blind to the same value the offline route dies on"
    );

    // The constructor must refuse it, naming the offending buffer position.
    let err = refused(parts);
    let expected = embedding_range(0, 2).start;
    assert!(
      matches!(err, ExtractError::NonFiniteRawEmbedding(i) if i == expected),
      "expected NonFiniteRawEmbedding({expected}) for {bad:?}, got {err:?}"
    );
  }

  // Non-vacuity in the other direction: the SAME geometry with a finite value
  // in that position is accepted. The check refuses the non-finiteness, not the
  // inactive slot carrying a row — which stays deliberately allowed (see
  // `an_inactive_slots_row_cannot_change_the_online_result`).
  for ok in [0.0f32, -0.0, 7.5, f32::MAX, f32::MIN_POSITIVE] {
    Extraction::try_from_parts(one_chunk_with_poisoned_slot(2, 0, ok)).unwrap_or_else(|e| {
      panic!("a finite {ok:?} in an inactive slot's row must be accepted: {e:?}")
    });
  }
}

#[test]
fn the_finiteness_check_covers_the_whole_buffer_not_only_active_rows() {
  // The property is over the BUFFER: no `(chunk, slot, dimension)` position may
  // hold a non-finite value. Swept exhaustively over the slot axis and over the
  // first/last dimension of each row, so a check that walked only some rows —
  // or that stopped at a row boundary — fails here.
  //
  // Slot 0 is ACTIVE, so its refusal is check 9's `ActiveSlotWithoutEmbedding`
  // (`normalize_from` rejects a non-finite row): the more specific diagnosis,
  // which is why check 11 is ordered behind it. Slots 1 and 2 are inactive and
  // land on check 11. Both are refusals; neither position is accepted.
  for slot in 0..SEG_NUM_SLOTS {
    for dimension in [0, 1, EMBEDDING_DIM - 1] {
      for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = refused(one_chunk_with_poisoned_slot(slot, dimension, bad));
        let flat = embedding_range(0, slot).start + dimension;
        if slot == 0 {
          assert!(
            matches!(err, ExtractError::ActiveSlotWithoutEmbedding(a)
              if (a.chunk(), a.slot()) == (0, 0)),
            "an ACTIVE slot's non-finite row must keep check 9's diagnosis \
             (slot {slot}, dimension {dimension}, {bad:?}), got {err:?}"
          );
        } else {
          assert!(
            matches!(err, ExtractError::NonFiniteRawEmbedding(i) if i == flat),
            "expected NonFiniteRawEmbedding({flat}) for slot {slot}, dimension \
             {dimension}, {bad:?}, got {err:?}"
          );
        }
      }
    }
  }
}

#[test]
fn no_producer_can_emit_a_buffer_the_finiteness_check_refuses() {
  // Check 11 must refuse nothing either in-crate producer emits.
  // `Extractor::extract` and `ArgmaxSource::extract` both allocate an all-zero
  // `raw_embeddings` and write a row ONLY when `raw_embedding_reaches_plda`
  // accepts it, zeroing the slot's segmentation column otherwise. Two facts
  // make check 11 unreachable from either, and both are asserted here rather
  // than assumed:
  //
  //   a. an UNWRITTEN row stays all-zero, and `0.0` is finite;
  //   b. a WRITTEN row passed `raw_embedding_reaches_plda`, whose
  //      `RawEmbedding::from_wespeaker` clause carries its own finiteness scan.
  assert!(0.0f32.is_finite(), "an unwritten row is all-zero");

  let plda = shared_plda_transform().expect("hermetic PLDA weights load");
  let usable = planar_row(30.0);
  assert!(
    raw_embedding_reaches_plda(plda, &usable),
    "non-vacuity: the unpoisoned row must be one a producer WOULD write"
  );
  for dimension in [0, 1, EMBEDDING_DIM - 1] {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      let mut row = usable;
      row[dimension] = bad;
      assert!(
        !raw_embedding_reaches_plda(plda, &row),
        "a producer must never write a row holding {bad:?} at dimension {dimension}"
      );
    }
  }

  // And the producers' own emitted shape — an all-zero row under an all-zero
  // column — is accepted, which is what `one_usable_slot_row`'s untouched slots
  // and `valid_parts` already are.
  Extraction::try_from_parts(valid_parts()).expect("the reference parts stay accepted");
  Extraction::try_from_parts(ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    segmentations: vec![1.0, 0.0, 0.0],
    count: vec![1, 0],
    num_chunks: 1,
    num_frames_per_chunk: 1,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  })
  .expect("two all-zero rows under two all-zero columns is the producers' own output shape");
}

// =====================================================================
// The OFFLINE half of "an INACTIVE slot may carry a usable row": dia
// scores every row at stage 6, so blindness needs stage 7's mask and
// the assignment's row-shift invariance, not just the PLDA train gate.
// =====================================================================

/// A row with `1.0` at dimension `i` and zeros elsewhere.
fn axis_row(i: usize, v: f32) -> [f32; EMBEDDING_DIM] {
  let mut r = [0.0f32; EMBEDDING_DIM];
  r[i] = v;
  r
}

/// Six unit chunks of one frame, slot 0 active in every one and carrying
/// `SLOT0_ROWS[c]` — three well-separated pairs, so `diaric` reaches three
/// alive clusters and its Hungarian step has spare columns to hand the INACTIVE
/// slots real labels. `probe` is written into `(chunk, slot)`, and into both
/// inactive slots of a second chunk with opposite signs, so two constant rows
/// also compete for the leftover columns.
fn six_chunk_parts(probe: [f32; EMBEDDING_DIM], chunk: usize, slot: usize) -> ExtractionParts {
  let slot0: [[f32; EMBEDDING_DIM]; 6] = [
    planar_row(0.0),
    planar_row(4.0),
    planar_row(88.0),
    planar_row(92.0),
    axis_row(2, 1.0),
    {
      let mut r = axis_row(2, 1.0);
      r[3] = 0.07;
      r
    },
  ];
  let n = slot0.len();
  let mut raw_embeddings = vec![0.0f32; n * SEG_NUM_SLOTS * EMBEDDING_DIM];
  for (c, row) in slot0.iter().enumerate() {
    raw_embeddings[embedding_range(c, 0)].copy_from_slice(row);
  }
  raw_embeddings[embedding_range(chunk, slot)].copy_from_slice(&probe);
  let other = (chunk + 2) % n;
  raw_embeddings[embedding_range(other, 1)].copy_from_slice(&probe);
  let mut anti = probe;
  for v in anti.iter_mut() {
    *v = -*v;
  }
  raw_embeddings[embedding_range(other, 2)].copy_from_slice(&anti);

  // [f][s], one frame per chunk: only slot 0 is ever active.
  let mut segmentations = vec![0.0f64; n * SEG_NUM_SLOTS];
  for c in 0..n {
    segmentations[c * SEG_NUM_SLOTS] = 1.0;
  }
  let mut count = vec![1u8; n];
  count.push(0); // the trailing output frame no chunk covers

  ExtractionParts {
    raw_embeddings,
    segmentations,
    count,
    num_chunks: n,
    num_frames_per_chunk: 1,
    chunks_sw: unit_sw(),
    frames_sw: unit_sw(),
  }
}

#[test]
fn an_inactive_slots_row_cannot_change_the_offline_result() {
  // The claim the "deliberately NOT checked" list makes about OFFLINE. The
  // earlier justification stopped at `filter_embeddings`: an inactive slot never
  // enters the PLDA TRAIN subset. That is true and insufficient — dia's stage 6
  // cosine-scores EVERY row against every centroid, train subset or not. What
  // makes offline blind is stage 7 (it OVERWRITES an inactive slot's whole soft
  // row with `soft.min() - 1.0`) plus the fact that the residue — the row's
  // contribution to that `soft.min()` — is a per-row constant shift, which a
  // linear assignment problem is invariant under.
  //
  // Property, stated without the mechanism: an INACTIVE slot's raw-embedding row
  // must not change ANY observable of `diarize_with` — spans, per-chunk hard
  // assignment, cluster count, or the frame-level grid.
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let fingerprint = |parts: ExtractionParts| -> OutputFingerprint {
    output_fingerprint(
      &Extraction::try_from_parts(parts)
        .expect("the probe geometry is self-consistent")
        .diarize_with(&plda, ClusterBackend::default())
        .expect("this geometry clusters offline"),
    )
  };

  let zero = [0.0f32; EMBEDDING_DIM];
  let base = fingerprint(six_chunk_parts(zero, 0, 1));

  // Non-vacuity: the geometry must actually exercise the path. Three alive
  // clusters, and the INACTIVE slots must really draw labels — if dia left them
  // UNMATCHED there would be nothing for a probe to perturb.
  assert_eq!(base.2, 3, "the probe geometry must reach three clusters");
  assert!(
    base.1.iter().all(|row| row[1] >= 0 && row[2] >= 0),
    "the inactive slots must draw real cluster labels, got {:?}",
    base.1
  );

  let probes = [
    planar_row(180.0),
    planar_row(270.0),
    planar_row(45.0),
    axis_row(2, -1.0),
    axis_row(7, 1.0),
    axis_row(7, -3.5),
  ];
  for slot in [1usize, 2] {
    for chunk in [0usize, 3, 5] {
      let at = fingerprint(six_chunk_parts(zero, chunk, slot));
      assert_eq!(
        at, base,
        "moving the all-zero probe to (chunk {chunk}, slot {slot}) changed the offline output"
      );
      for (i, probe) in probes.iter().enumerate() {
        assert_eq!(
          fingerprint(six_chunk_parts(*probe, chunk, slot)),
          base,
          "probe {i} in INACTIVE (chunk {chunk}, slot {slot}) changed the offline output"
        );
      }
    }
  }
}

// =====================================================================
// ROUND 7. The SEGMENTATION DOMAIN. Every earlier round was "the validator
// reads a value one way, a backend reads it another"; this one is that at the
// level of the tensor's value set rather than a row, a count or a grid. The
// constructor booleanizes `segmentations` at `seg > 0.0` everywhere it touches
// them, and so does the whole online route — but `diaric`'s offline route SUMS
// the magnitudes (`filter_embeddings`' `clean_frames`, stage 7's
// `sum_activity == 0.0`). The two are the same function on `{0.0, 1.0}` and
// different functions off it, so the cure is to confine the input.
// =====================================================================

/// The round-7 trigger, as parts: ONE chunk of FOUR frames, slot 0 carrying
/// `v` on frames 0-1 and slot 1 carrying `v` on frames 2-3, with two usable
/// ORTHOGONAL embedding rows. `chunks_sw = (0, 3, 1)` over `frames_sw =
/// (0, 1, 1)` derives exactly four output frames, and every frame has exactly
/// one active slot, so `count` is `[1, 1, 1, 1]` for any `v > 0.0`.
///
/// At `v = 1.0` the two backends agree (two speakers). The whole geometry is
/// held fixed and only `v` moves, so a divergence is attributable to the VALUE
/// and to nothing else.
fn split_by_magnitude_parts(v: f64) -> ExtractionParts {
  let mut raw_embeddings = one_usable_slot_row(0);
  // Slot 1: orthogonal to slot 0's `[1.0; 64]` prefix, so the two rows sit at
  // cosine distance 1 — as far apart as the online engine can see.
  raw_embeddings[embedding_range(0, 1)][64..128].fill(1.0);
  let mut segmentations = vec![0.0f64; 4 * SEG_NUM_SLOTS];
  for f in 0..2 {
    segmentations[f * SEG_NUM_SLOTS] = v; // frames 0-1, slot 0
  }
  for f in 2..4 {
    segmentations[f * SEG_NUM_SLOTS + 1] = v; // frames 2-3, slot 1
  }
  ExtractionParts {
    raw_embeddings,
    segmentations,
    count: vec![1, 1, 1, 1],
    num_chunks: 1,
    num_frames_per_chunk: 4,
    chunks_sw: SlidingWindow::new(0.0, 3.0, 1.0),
    frames_sw: SlidingWindow::new(0.0, 1.0, 1.0),
  }
}

/// One backend's observable answer: the cluster count and the spans it emits.
type BackendAnswer = (usize, Vec<(f64, f64, usize)>);

/// Both backends' [`BackendAnswer`] for `e`, offline first.
fn both_backends(e: &Extraction) -> (BackendAnswer, BackendAnswer) {
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let read = |o: &diaric::offline::OfflineOutput| {
    (
      o.num_clusters(),
      o.spans_slice()
        .iter()
        .map(|s| (s.start(), s.end(), s.cluster()))
        .collect::<Vec<_>>(),
    )
  };
  (
    read(
      &e.diarize_with(&plda, ClusterBackend::default())
        .expect("this geometry clusters offline"),
    ),
    read(
      &e.diarize_online(OnlineOptions::new())
        .expect("this geometry clusters online"),
    ),
  )
}

#[test]
fn a_fractional_segmentation_splits_the_two_backends() {
  // Finding (round 7). `try_from_parts` BOOLEANIZES every positive
  // segmentation cell — check 10's activity scan and check 11's count
  // derivation both test `seg > 0.0` — but the offline backend sums the
  // ORIGINAL magnitudes: `filter_embeddings` accumulates
  // `clean_frames += segmentations[..]` over singly-active frames and compares
  // it with `MIN_ACTIVE_RATIO * num_frames_per_chunk`
  // (`diarization/src/offline/algo.rs:644-679`). At `v = 0.1` over four frames
  // each slot's clean sum is `0.2`, below `0.2 * 4 = 0.8`, so NEITHER slot
  // enters the PLDA train subset and offline collapses both into ONE speaker;
  // the online engine sees two one-second slots at cosine distance 1 and emits
  // TWO. One continuous speaker offline, two regions online, from parts that
  // every check before this one accepts.
  //
  // The cure is the DOMAIN, not a second model of offline's sum: no in-crate
  // producer emits a fractional cell (see
  // `no_producer_can_emit_a_segmentation_cell_the_domain_check_refuses`), so
  // soft segmentation support was a capability with no producer and two
  // incompatible readers.
  let parts = split_by_magnitude_parts(0.1);

  // Assembled UNCHECKED — the split itself, independent of what the
  // constructor decides to do about it.
  let unchecked = Extraction::from_parts(
    parts.raw_embeddings.clone(),
    parts.segmentations.clone(),
    parts.count.clone(),
    parts.num_chunks,
    parts.num_frames_per_chunk,
    parts.chunks_sw,
    parts.frames_sw,
  );
  let (offline, online) = both_backends(&unchecked);
  assert_eq!(
    offline,
    (1, vec![(0.5, 3.5, 0)]),
    "offline must read the 0.1 cells as magnitudes and merge the two slots"
  );
  assert_eq!(
    online,
    (2, vec![(0.5, 2.5, 0), (2.5, 3.5, 1)]),
    "online must read the same cells as booleans and split the two slots"
  );
  assert_ne!(offline, online, "the two backends must actually disagree");

  // The IDENTICAL geometry with the cells at `1.0` agrees, which is what makes
  // this a domain defect rather than a geometry one.
  let hard = Extraction::try_from_parts(split_by_magnitude_parts(1.0))
    .expect("the hard-binary twin of the same geometry is accepted");
  let (offline_hard, online_hard) = both_backends(&hard);
  assert_eq!(
    offline_hard, online_hard,
    "on the hard-binary domain the two backends must agree"
  );
  assert_eq!(offline_hard.0, 2, "and both must find the two speakers");

  // So the constructor must refuse the fractional twin, naming the first cell.
  let err = refused(parts);
  let ExtractError::NonBinarySegmentation(n) = err else {
    panic!("expected NonBinarySegmentation, got {err:?}")
  };
  assert_eq!((n.index(), n.value(), n.slot()), (0, 0.1, 0));
}

#[test]
fn the_segmentation_domain_check_is_an_equality_over_the_whole_buffer() {
  // The scan is `!= 0.0 && != 1.0` over every cell, so it is neither a range
  // test nor a floor: values INSIDE `(0.0, 1.0)`, ABOVE `1.0`, BELOW `0.0` and
  // non-finite are all refused, wherever they sit and whether or not the slot
  // they belong to is otherwise active.
  //
  // `-0.0` is the one value that must be ACCEPTED despite not being written
  // `0.0`: IEEE-754 equality makes `-0.0 == 0.0`, both backends' `> 0.0` and
  // `sum == 0.0` readings agree on it, and it is what a caller gets from
  // negating a zeroed column.
  let base = split_by_magnitude_parts(1.0);
  Extraction::try_from_parts(base.clone()).expect("the all-hard buffer is accepted");

  for (cell, bad) in [
    (0usize, 0.5f64),
    (1, 0.5),  // an INACTIVE slot's cell: the scan is not activity-gated
    (2, -1.0), // below zero
    (5, 1.5),  // above one
    (7, f64::NAN),
    (10, f64::INFINITY),
    (11, f64::NEG_INFINITY),
    (11, f64::MIN_POSITIVE),
  ] {
    let mut parts = base.clone();
    parts.segmentations[cell] = bad;
    // A cell change can also break `count`; the domain check runs FIRST, so the
    // diagnosis names the cell rather than the frame.
    let err = refused(parts);
    let ExtractError::NonBinarySegmentation(n) = err else {
      panic!("expected NonBinarySegmentation for {bad} at cell {cell}, got {err:?}")
    };
    assert_eq!(n.index(), cell, "the FIRST offending cell, for {bad}");
    assert_eq!(n.slot(), cell % SEG_NUM_SLOTS);
    assert_eq!(n.value().to_bits(), bad.to_bits(), "the value, verbatim");
  }

  // `-0.0` is accepted: it compares EQUAL to `0.0`, so the domain equality
  // admits it and both backends' `> 0.0` and `sum == 0.0` readings agree on it.
  let mut negative_zero = base;
  negative_zero.segmentations[1] = -0.0;
  Extraction::try_from_parts(negative_zero).expect("-0.0 == 0.0 and both backends read it so");
}

#[test]
fn no_producer_can_emit_a_segmentation_cell_the_domain_check_refuses() {
  // The premise the fix rests on, asserted rather than assumed: check 9
  // requires no more of a caller than this crate requires of itself.
  //
  // `Extractor::extract` writes exactly two things into `segmentations` —
  // `segment::multilabel`'s output at `chunk_segmentation_range(c, F)`, and
  // `zero_slot_column`'s zeros — so the reachable value set is
  // `POWERSET_TABLE ∪ {0.0}`. The table is walked here through the PUBLIC
  // `multilabel` over every powerset class, which is what makes this a check on
  // the producer rather than a restatement of the table.
  for class in 0..crate::audio::speaker::segment::POWERSET_CLASSES {
    let mut logits = vec![0.0f32; crate::audio::speaker::segment::POWERSET_CLASSES];
    logits[class] = 1.0;
    for v in crate::audio::speaker::segment::multilabel(&logits, 1) {
      assert!(
        v == 0.0 || v == 1.0,
        "multilabel class {class} emitted {v}, which check 9 would refuse"
      );
    }
  }

  // `ArgmaxSource` writes `f64::from(speaker_ids[..])` and `zero_slot_column`'s
  // zeros. `speaker_ids` is the segmenter graph's own hard powerset decode,
  // value set exactly `{0.0, 1.0}` — probed against the real model and pinned
  // by the model-gated `argmax_decoded_output_value_semantics` in
  // `source::argmax::tests`, which is where a graph change would surface. The
  // widening itself is exact for those two values, in both f16→f32 (the read)
  // and f32→f64 (the write), which is the step this test can prove hermetically.
  //
  // Round 8: for THAT source this paragraph is no longer the guarantee, only
  // the reason the guarantee never fires. `from_dir_with` accepts a model on
  // its I/O SHAPES, so "the decode is hard-binary" is a property of the shipped
  // graph and a model-gated test is not a runtime guard; the source now runs
  // check 9 itself, at its assembly door
  // (`a_fractional_speaker_id_splits_the_backends_and_is_refused_at_assembly`).
  // `Extractor::extract`'s half above stays structural — `POWERSET_TABLE` is a
  // compile-time constant, not a model output — and it runs the check too.
  for v in [0.0f32, 1.0] {
    let widened = f64::from(crate::f16::from_f32(v).to_f32());
    assert!(
      widened == 0.0 || widened == 1.0,
      "the f16 speaker_ids value {v} must widen to exactly {v}"
    );
  }
}

/// The `frames_sw` sub-domain round 7's table missed: three finite, positive
/// fields that still do not generate a timeline.
///
/// Round 8, finding 2. Check 2 asks whether `frames_sw`'s fields are usable
/// NUMBERS; both backends additionally read them FORWARD, as the frame centers
/// every span endpoint is built from. Those two readings have different
/// sub-domains, and this geometry is inside the first and outside the second.
fn collapsing_frame_grid_parts() -> ExtractionParts {
  ExtractionParts {
    raw_embeddings: one_usable_slot_row(0),
    // One chunk, two frames; only frame 0 of slot 0 is active, so the run
    // closes at frame 1 and the span is (center(0), center(1)).
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    count: vec![1, 0, 0, 0],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    // duration + (num_chunks - 1) * step = 3e-8, over a frame step of 1e-8:
    // four output frames, which is what `count` above declares.
    chunks_sw: SlidingWindow::new(1e9, 3e-8, 1e-8),
    frames_sw: SlidingWindow::new(1e9, 1e-8, 1e-8),
  }
}

#[test]
fn try_from_parts_rejects_a_frames_sw_that_collapses_adjacent_frame_centers() {
  // The arithmetic, stated rather than described: at 1e9 the f64 ULP is
  // 1.19e-7, an order of magnitude above the 1e-8 step, so the step adds
  // literally nothing.
  assert_eq!(
    f64::from_bits(1e9f64.to_bits() + 1) - 1e9,
    1.1920928955078125e-7
  );
  assert_eq!(1e9f64 + 1e-8, 1e9);

  let parts = collapsing_frame_grid_parts();

  // Every earlier check ACCEPTS these parts — the geometry is self-consistent,
  // the windows are finite and positive, the grids agree about chunk 0, the
  // segmentations are hard-binary, the row reaches PLDA, and `count` IS what
  // those segmentations derive. Proved by the twin below, which differs only in
  // the two ORIGINS and is accepted.
  let separated = ExtractionParts {
    chunks_sw: SlidingWindow::new(0.0, 3e-8, 1e-8),
    frames_sw: SlidingWindow::new(0.0, 1e-8, 1e-8),
    ..parts.clone()
  };
  let ok = Extraction::try_from_parts(separated).expect("the same grid at origin 0.0 is accepted");
  assert_eq!(ok.num_output_frames(), 4);

  // Assembled UNCHECKED, the span conversion closes the one-frame run at
  // endpoints that are the identical f64, and offline returns `Ok` with a span
  // of DURATION ZERO — speech reported as an instant. (Online returns `Ok` with
  // no span at all: its `min_speech_duration` gate drops a slot whose single
  // 1e-8 s frame cannot meet the default 1.0 s. Both answers are useless, and
  // the constructor accepted the parts that produced them.)
  let unchecked = Extraction::from_parts(
    parts.raw_embeddings.clone(),
    parts.segmentations.clone(),
    parts.count.clone(),
    parts.num_chunks,
    parts.num_frames_per_chunk,
    parts.chunks_sw,
    parts.frames_sw,
  );
  let plda = diaric::plda::PldaTransform::new().expect("hermetic PLDA weights load");
  let offline = unchecked
    .diarize_with(&plda, ClusterBackend::default())
    .expect("offline returns Ok");
  assert_eq!(
    offline
      .spans_slice()
      .iter()
      .map(|s| (s.start(), s.duration()))
      .collect::<Vec<_>>(),
    vec![(1e9, 0.0)],
    "the active run closes at identical endpoints: a zero-duration span"
  );
  assert_eq!(
    unchecked
      .diarize_online(OnlineOptions::new())
      .expect("online returns Ok")
      .spans_slice()
      .len(),
    0
  );
  // The SAME extraction on the separated grid emits a span with a real
  // duration, so the zero is the grid's doing and not the fixture's.
  assert_eq!(
    ok.diarize_with(&plda, ClusterBackend::default())
      .expect("offline returns Ok")
      .spans_slice()
      .iter()
      .map(|s| (s.start(), s.duration()))
      .collect::<Vec<_>>(),
    vec![(5e-9, 1.0000000000000002e-8)],
    "a real duration — the 2-ULP tail is the same rounding, now visible instead \
     of annihilating"
  );

  // And the constructor refuses it, naming the first frame that repeats a
  // center. This is what goes red if check 13 is reverted.
  let err = refused(parts);
  let ExtractError::CollapsedFrameCenter(c) = err else {
    panic!("expected CollapsedFrameCenter, got {err:?}")
  };
  assert_eq!(
    (c.frame(), c.center(), c.previous()),
    (1, 1e9, 1e9),
    "frame 1 lands on frame 0's center"
  );
}

#[test]
fn no_producer_can_emit_a_frame_grid_the_center_check_refuses() {
  // The premise the fix rests on for the two in-crate sources, asserted rather
  // than assumed: both take their `frames_sw` from
  // `window::frame_sliding_window()`, whose three constants generate a strictly
  // increasing center sequence over the WHOLE admitted range — so check 13
  // requires no more of a caller than this crate requires of itself.
  assert_eq!(
    crate::audio::speaker::window::first_collapsed_frame_center(
      MAX_OUTPUT_FRAMES,
      crate::audio::speaker::window::frame_sliding_window()
    ),
    None
  );
  // The MARGIN, so a future frame grid that narrows it fails loudly here rather
  // than at whatever grid size first collapses. At the last admitted center
  // (70 778.89 s, i.e. 19.6 h) the f64 ULP is 1.46e-11, so the 0.016875 s step
  // clears it by a factor of ~1.16e9.
  let last = crate::audio::speaker::window::frame_center(
    MAX_OUTPUT_FRAMES - 1,
    crate::audio::speaker::window::frame_sliding_window(),
  );
  let ulp = f64::from_bits(last.to_bits() + 1) - last;
  assert!(
    crate::audio::speaker::window::FRAME_STEP_S / ulp > 1e9,
    "step {} against ULP {ulp:e} at the last center {last}",
    crate::audio::speaker::window::FRAME_STEP_S
  );
}

/// The class fix, pinned: the door every in-crate producer assembles through
/// reaches the SAME verdict as the public constructor, on every input.
///
/// Round 8's finding is that a check confined to one construction path is worse
/// than none. `Extraction::from_parts` is now private to the `extract` module,
/// so the only ways in are `try_from_parts` and `assemble_checked` — and this
/// test is what says the second is not the weaker of the two. It is a
/// PROPERTY test over the shape of the two doors, not a restatement of any one
/// check: it would go red for a producer door that skipped a check, ran them in
/// a different order, or mapped an error differently.
#[test]
fn assemble_checked_reaches_the_same_verdict_as_try_from_parts() {
  let onset = WindowOptions::new().onset();
  let good_row = one_usable_slot_row(0);
  let unit = unit_sw();

  // Each case is what a PRODUCER hands the door: a label, two tensors and a
  // geometry. `count` is derived from them exactly as `assemble_checked`
  // derives it, so the `ExtractionParts` built below is the same input in the
  // other door's shape.
  type DoorCase = (
    &'static str,
    Vec<f32>,
    Vec<f64>,
    usize,
    usize,
    SlidingWindow,
    SlidingWindow,
  );
  let cases: [DoorCase; 7] = [
    (
      "accepted",
      good_row.clone(),
      vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      1,
      2,
      unit,
      unit,
    ),
    (
      "check 9: a fractional cell",
      good_row.clone(),
      vec![0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
      1,
      2,
      unit,
      unit,
    ),
    (
      "check 10: an active slot with an all-zero row",
      vec![0.0f32; SEG_NUM_SLOTS * EMBEDDING_DIM],
      vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      1,
      2,
      unit,
      unit,
    ),
    (
      "check 12: a non-finite row under an INACTIVE column",
      {
        let mut r = good_row.clone();
        r[2 * EMBEDDING_DIM] = f32::NAN;
        r
      },
      vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      1,
      2,
      unit,
      unit,
    ),
    (
      "check 13: a frames_sw that collapses adjacent centers",
      good_row.clone(),
      vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      1,
      2,
      SlidingWindow::new(1e9, 3e-8, 1e-8),
      SlidingWindow::new(1e9, 1e-8, 1e-8),
    ),
    (
      "check 8: the two grids place chunk 1 differently",
      {
        let mut r = vec![0.0f32; 2 * SEG_NUM_SLOTS * EMBEDDING_DIM];
        r[..64].fill(1.0);
        r
      },
      vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      2,
      1,
      SlidingWindow::new(0.0, 0.04218750000000001, 0.04218750000000001),
      crate::audio::speaker::window::frame_sliding_window(),
    ),
    (
      "check 7: a frames_sw step that vanishes in f32",
      good_row,
      vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      1,
      2,
      SlidingWindow::new(0.0, 1e-300, 1e-300),
      SlidingWindow::new(0.0, 1.0, 1e-300),
    ),
  ];

  // Each case must exercise a DIFFERENT check, or this is seven copies of one
  // agreement. The expected variant is read off the case's own label.
  let expected_variant = |name: &str| -> Option<&'static str> {
    match name
      .split(':')
      .next()
      .expect("split yields at least one part")
    {
      "accepted" => None,
      "check 7" => Some("FrameStepNotRepresentableInF32"),
      "check 8" => Some("MisalignedChunkPlacement"),
      "check 9" => Some("NonBinarySegmentation"),
      "check 10" => Some("ActiveSlotWithoutEmbedding"),
      "check 12" => Some("NonFiniteRawEmbedding"),
      "check 13" => Some("CollapsedFrameCenter"),
      other => panic!("unmapped case label `{other}`"),
    }
  };

  for (
    name,
    raw_embeddings,
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    chunks_sw,
    frames_sw,
  ) in cases
  {
    let count = crate::audio::speaker::window::try_count_from_segmentations(
      &segmentations,
      num_chunks,
      num_frames_per_chunk,
      SEG_NUM_SLOTS,
      onset,
      chunks_sw,
      frames_sw,
    )
    .expect("every case's geometry derives a count that fits usize");
    let public = Extraction::try_from_parts(ExtractionParts {
      raw_embeddings: raw_embeddings.clone(),
      segmentations: segmentations.clone(),
      count,
      num_chunks,
      num_frames_per_chunk,
      chunks_sw,
      frames_sw,
    });
    let producer = Extraction::assemble_checked(
      raw_embeddings,
      segmentations,
      num_chunks,
      num_frames_per_chunk,
      onset,
      chunks_sw,
      frames_sw,
    );
    assert_eq!(
      public.as_ref().err(),
      producer.as_ref().err(),
      "the two doors disagreed about `{name}`"
    );
    assert_eq!(
      public.ok(),
      producer.clone().ok(),
      "the two doors assembled different Extractions for `{name}`"
    );
    match (expected_variant(name), producer) {
      (None, Ok(_)) => {}
      (Some(variant), Err(e)) => assert!(
        format!("{e:?}").starts_with(variant),
        "`{name}` must be refused by {variant}, got {e:?}"
      ),
      (expected, got) => panic!("`{name}`: expected {expected:?}, got {got:?}"),
    }
  }
}

/// Round 9's falsifier: the output-frame cap must refuse a geometry BEFORE the
/// work it exists to bound, not after.
///
/// # The defect, measured
///
/// Round 8 gave every producer the whole thirteen-check sequence, but at the
/// END of its work. Both producers build their extraction tensors and run every
/// CoreML call, `assemble_checked` then derives `count`, and only then does
/// check 6 — whose entire job is to refuse a grid this crate will not allocate
/// for — get to speak. So the cap refused the right inputs after paying for
/// them.
///
/// The two allocations below are the price, and this test MEASURES them rather
/// than asserting them (`crate::tests::alloc_probe`, a counting global
/// allocator with per-thread counters):
///
/// - `1 217 810 160` bytes of `raw_embeddings` + `segmentations`, the tensors
///   a producer must build before it can call this door at all;
/// - `404 771 544` bytes inside `assemble_checked` itself — the chunk-count
///   vector, the aggregate/coverage pair and the `count` — every one of them
///   sized by the very grid the cap was about to refuse. That figure is what
///   this test printed before the preflight existed; it is `0` now.
///
/// # The input
///
/// `1 132 448 001` samples: the SMALLEST clip either producer refuses, and one
/// sample fewer is asserted below to be accepted, so this is a boundary and not
/// a large round number. At the default `step_samples` that is 70 770 chunks,
/// a last chunk ending at 70 779.0 s, and `round(70 779 / 0.016875) + 1 =
/// 4 194 312` output frames against a cap of `4 194 304` — 19.66 hours of
/// audio, far past what either clustering backend could finish.
///
/// # What is substituted for the producer, and why
///
/// `Extractor::extract` needs a loaded `SegmentModel`, and could not be RUN on
/// this input even with one: the clip is 4.5 GB of `f32` samples and 70 770
/// pairs of CoreML calls, which is not a unit test on any host. So the producer
/// is exercised in the two halves that bracket its models, and nothing between
/// them is skipped:
///
/// - `Extractor::checked_geometry`, which IS `extract`'s pre-inference
///   sequence — not a re-composition of it. `extract` calls this and nothing
///   else before it allocates a tensor or touches a model, so a cap moved back
///   behind either would leave this assertion unsatisfied.
/// - `Extraction::assemble_checked`, the assembly door, called with the exact
///   tensors and geometry `extract` would hand it at this input. That is the
///   call that used to allocate 404 771 544 bytes before refusing, and it is
///   the half the allocation measurement lives in.
///
/// `ArgmaxSource`'s own earliest point is asserted the same way, one module over
/// (that module's `the_frame_cap_refuses_argmax_geometry_before_its_input_scans`);
/// it is earlier than this one, because that source's stride is fixed at load.
///
/// What neither covers is the POSITION of the `checked_geometry` call inside
/// each producer's body: that is the diff, plus the model-gated end-to-end
/// gates, which report as `ignored` wherever the speaker models are not staged.
#[test]
fn the_frame_cap_refuses_before_the_allocation_it_bounds() {
  use crate::{audio::speaker::window, tests::alloc_probe};

  /// The smallest clip whose derived output grid exceeds `MAX_OUTPUT_FRAMES`.
  const SMALLEST_OVER_CAP_SAMPLES: usize = 1_132_448_001;
  /// The derived grid at that clip. Eight frames past the cap.
  const OVER_CAP_FRAMES: usize = 4_194_312;
  /// community-1's per-chunk frame count — `SegmentModel::num_frames()` for
  /// every segmenter this crate loads, and `ARGMAX_FRAMES_PER_WINDOW` for the
  /// other producer, asserted below rather than assumed.
  const FRAMES_PER_CHUNK: usize = 589;

  assert_eq!(
    FRAMES_PER_CHUNK,
    crate::audio::speaker::source::argmax::ARGMAX_FRAMES_PER_WINDOW
  );
  const { assert!(OVER_CAP_FRAMES > MAX_OUTPUT_FRAMES) };

  // ── `Extractor`'s own pre-inference sequence, called directly ──────
  // Not a re-composition of the functions `extract` calls — `extract` calls
  // THIS, and calls nothing else before it allocates or infers. So a cap moved
  // back behind the tensors would show up here.
  let extractor = Extractor::new();
  assert_eq!(
    extractor.checked_geometry(SMALLEST_OVER_CAP_SAMPLES, FRAMES_PER_CHUNK),
    Err(ExtractError::OutputFrameCountTooLarge(OVER_CAP_FRAMES)),
    "the cap is reachable from `samples.len()` and the options alone — no \
     tensor, no model"
  );

  // A BOUNDARY, not a blanket refusal of long clips: one sample fewer is one
  // chunk fewer, derives 4 194 253 frames, and is accepted. So this is the
  // smallest input that now fails early, and the fix refuses nothing that was
  // not already refused.
  let w = WindowOptions::new();
  let (num_chunks, chunks_sw, frames_sw) = extractor
    .checked_geometry(SMALLEST_OVER_CAP_SAMPLES - 1, FRAMES_PER_CHUNK)
    .expect("one sample fewer derives a grid inside the cap");
  assert_eq!(
    (
      num_chunks,
      checked_output_frame_count(num_chunks, chunks_sw, frames_sw)
    ),
    (70_769, Ok(4_194_253))
  );

  // The refused geometry itself, for the allocation half below. `num_chunks`
  // and both windows are what `checked_geometry` would have returned had the
  // cap not refused them, from the same three functions.
  let num_chunks = window::num_chunks(SMALLEST_OVER_CAP_SAMPLES, &w);
  assert_eq!(num_chunks, 70_770);
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();

  // ── What a producer had already built to reach the old refusal ─────
  // `Extractor::extract`'s two output buffers at this geometry, sized by its
  // own expressions. Zeroed and never read, so they cost address space and not
  // resident pages — the counting allocator is what sees them at all.
  let (tensors, built) = alloc_probe::measure(|| {
    (
      vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM],
      vec![0.0f64; num_chunks * FRAMES_PER_CHUNK * SEG_NUM_SLOTS],
    )
  });
  assert_eq!(
    (built.total, built.peak),
    (1_217_810_160, 1_217_810_160),
    "the two extraction tensors a producer builds before assembly"
  );

  // ── ...and what the assembly door itself spent before refusing ─────
  let (err, scratch) = alloc_probe::measure(|| {
    Extraction::assemble_checked(
      tensors.0,
      tensors.1,
      num_chunks,
      FRAMES_PER_CHUNK,
      w.onset(),
      chunks_sw,
      frames_sw,
    )
    .expect_err("a 19.66 h clip derives a grid past MAX_OUTPUT_FRAMES")
  });
  assert_eq!(err, ExtractError::OutputFrameCountTooLarge(OVER_CAP_FRAMES));
  assert_eq!(
    (scratch.total, scratch.peak),
    (0, 0),
    "the door allocated before refusing a geometry it could refuse from \
     `num_chunks` and the two windows alone (404 771 544 bytes before round 9)"
  );
}

/// The other half of "only sooner": the preflight must reach the SAME verdict
/// the late check reaches, on grids at and either side of the boundary.
///
/// A cap enforced in two places is a cap that can drift. This is the pinning
/// for the claim that it has not: for a swept range of geometries the
/// geometry-only preflight and the full thirteen-check sequence agree, both on
/// which are refused and on the exact `usize` the refusal names.
#[test]
fn the_geometry_preflight_and_the_assembled_check_never_disagree() {
  let onset = WindowOptions::new().onset();
  let unit = unit_sw();

  // `(num_chunks, chunks_sw, frames_sw)` triples spanning the cap: the unit
  // grid at one and two chunks, then the boundary trio — one frame below the
  // cap, exactly at it, and one past it — and finally a geometry that overflows
  // `usize`, so check 4's arm is swept as well as check 6's.
  let second = SlidingWindow::new(0.0, 1.0, 1.0);
  let cases: [(usize, SlidingWindow, SlidingWindow); 6] = [
    (1, unit, unit),
    (2, unit, unit),
    (
      1,
      SlidingWindow::new(0.0, (MAX_OUTPUT_FRAMES - 2) as f64, 1.0),
      second,
    ),
    (
      1,
      SlidingWindow::new(0.0, (MAX_OUTPUT_FRAMES - 1) as f64, 1.0),
      second,
    ),
    (
      1,
      SlidingWindow::new(0.0, MAX_OUTPUT_FRAMES as f64, 1.0),
      second,
    ),
    (
      1,
      SlidingWindow::new(0.0, 1e300, 1e300),
      SlidingWindow::new(0.0, 1e-300, 1e-300),
    ),
  ];

  for (num_chunks, chunks_sw, frames_sw) in cases {
    let preflight = checked_output_frame_count(num_chunks, chunks_sw, frames_sw);
    // The same geometry carried through the full sequence, over tensors that
    // satisfy every OTHER check — all-zero, so no slot is active and no cell is
    // outside `{0.0, 1.0}` — so whatever the two disagree about can only be
    // checks 4 and 6.
    let assembled = Extraction::assemble_checked(
      vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM],
      vec![0.0f64; num_chunks * SEG_NUM_SLOTS],
      num_chunks,
      1,
      onset,
      chunks_sw,
      frames_sw,
    );
    match (preflight, assembled) {
      (Err(pre), Err(post)) => assert_eq!(
        pre, post,
        "num_chunks={num_chunks} chunks_sw={chunks_sw:?}: the preflight and the \
         sequence named different errors"
      ),
      (Ok(n), Ok(e)) => assert_eq!(
        n,
        e.num_output_frames(),
        "num_chunks={num_chunks} chunks_sw={chunks_sw:?}: the preflight derived \
         a different grid than the one assembled"
      ),
      (pre, post) => panic!(
        "num_chunks={num_chunks} chunks_sw={chunks_sw:?}: preflight {pre:?} \
         disagrees with the sequence {post:?}"
      ),
    }
  }
}

/// Round 10: the cap round 9 hoisted bounds the OUTPUT axis, and the producers
/// allocate on the CHUNK axis. Nothing bounded that one.
///
/// # The defect, measured
///
/// `num_output_frames` is a function of the clip's DURATION —
/// `round(last_chunk_end / FRAME_STEP_S) + 1`, where `last_chunk_end =
/// CHUNK_DURATION_S + (num_chunks - 1) * step_samples / SAMPLE_RATE_HZ`. The two
/// `step_samples` cancel. The tensors are a function of `num_chunks =
/// samples.len() / step_samples`, which they do not.
///
/// So at a small `step_samples` the two diverge without limit. Ten minutes of
/// audio at `step_samples = 2` — a value `WindowOptions` accepts, since it
/// guards only against `0` — derives 4 720 001 chunks and 81 221 777 208 bytes
/// of `segmentations` + `raw_embeddings`, from 38 400 000 bytes of input. Its
/// output grid is 35 557 frames: 0.85 % of `MAX_OUTPUT_FRAMES`. Every guard that
/// existed passed it, the stride being even, so the placement scan found no tie
/// either — and `extract` then asked the allocator for 75.64 GiB and faced
/// 9 440 002 model calls.
///
/// The 81 221 777 208 below is MEASURED through `crate::tests::alloc_probe`, the
/// counting global allocator round 9 added, by building the two `vec![..]`s from
/// `Extractor::extract`'s own expressions at this geometry — not asserted from
/// the arithmetic that would be about to be fixed. It is `0` after the refusal,
/// because `checked_geometry` returns before the first `vec!`.
///
/// # What is substituted for the producer, and why
///
/// `Extractor::checked_geometry`, which IS `extract`'s pre-inference sequence —
/// `extract` calls it and nothing else before it allocates a tensor or touches a
/// model. `extract` itself cannot be run on this input: 9 440 002 CoreML calls
/// is not a unit test on any host, which is the same reason round 9's falsifier
/// attaches here.
#[test]
fn the_chunk_axis_cap_refuses_before_the_allocation_it_bounds() {
  use crate::{audio::speaker::window, tests::alloc_probe};

  /// A `step_samples` `WindowOptions` accepts today: it guards `0` and
  /// `> SEG_CHUNK_SAMPLES`, nothing between.
  const STEP: u32 = 2;
  /// Ten minutes at 16 kHz — 38 400 000 bytes of `f32`.
  const SAMPLES: usize = 9_600_000;
  /// community-1's per-chunk frame count, which `extract` reads from the loaded
  /// segmenter and hands `checked_geometry`.
  const FRAMES_PER_CHUNK: usize = 589;
  /// `1 + (9_600_000 - 160_000).div_ceil(2)`.
  const NUM_CHUNKS: usize = 4_720_001;
  /// `NUM_CHUNKS * (589 * 3 * 8 + 3 * 256 * 4)` — 75.64 GiB.
  const TENSOR_BYTES: usize = 81_221_777_208;

  let w = WindowOptions::new().with_step_samples(STEP);
  let extractor = Extractor::with_options(Options::new().with_window(w));
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();
  let num_chunks = window::num_chunks(SAMPLES, &w);
  assert_eq!(num_chunks, NUM_CHUNKS);

  // ── Every guard that existed before this one accepts the geometry ──
  assert_eq!(
    checked_output_frame_count(num_chunks, chunks_sw, frames_sw),
    Ok(35_557),
    "the output-frame cap sees a ten-minute clip and passes it"
  );
  // ...at well under 1 % of MAX_OUTPUT_FRAMES.
  const { assert!(35_557 * 100 / MAX_OUTPUT_FRAMES == 0) };
  assert_eq!(
    window::first_misaligned_chunk(num_chunks, chunks_sw, frames_sw),
    None,
    "and the even stride clears the placement scan"
  );

  // ── The caps, at the producer's own earliest point ─────────────────
  // Since round 11 the COMPUTE bound speaks first for this shape: 4 720 001
  // chunks is 9 440 002 model calls, and that is true of a geometry whatever
  // the loaded segmenter declares. The memory bound refuses the same geometry
  // on its own terms — 75.64 GiB — and both are reachable from three `usize`s.
  assert_eq!(
    extractor.checked_geometry(SAMPLES, FRAMES_PER_CHUNK),
    Err(ExtractError::ExtractionChunkCountTooLarge(NUM_CHUNKS)),
    "the chunk-axis bounds are reachable from `samples.len()`, `step_samples` \
     and the segmenter's frame count alone — no tensor, no model"
  );
  assert_eq!(
    checked_extraction_tensor_bytes(num_chunks, FRAMES_PER_CHUNK),
    Err(ExtractError::ExtractionTensorBytesTooLarge(TENSOR_BYTES)),
    "and the memory bound refuses it independently"
  );

  // ── What that refusal costs, and what it saves ─────────────────────
  // `Extractor::extract`'s two output buffers at this geometry, sized by its own
  // expressions. Zeroed and never read, so they cost address space rather than
  // resident pages — which is exactly why a counting allocator is what sees
  // them, and why the process survives long enough to report the number.
  let (tensors, built) = alloc_probe::measure(|| {
    (
      vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM],
      vec![0.0f64; num_chunks * FRAMES_PER_CHUNK * SEG_NUM_SLOTS],
    )
  });
  assert_eq!(
    (built.total, built.peak),
    (TENSOR_BYTES, TENSOR_BYTES),
    "the two extraction tensors this geometry used to reach the chunk loop with"
  );
  assert_eq!(
    (
      tensors.0.len() * size_of::<f32>(),
      tensors.1.len() * size_of::<f64>()
    ),
    (14_499_843_072, 66_721_934_136),
    "raw_embeddings and segmentations, the two terms the bound adds"
  );
  drop(tensors);

  // ...and the same door, on the same geometry, now allocating nothing.
  let (err, spent) = alloc_probe::measure(|| {
    extractor
      .checked_geometry(SAMPLES, FRAMES_PER_CHUNK)
      .expect_err("a 4 720 001-chunk grid is past both chunk-axis bounds")
  });
  assert_eq!(err, ExtractError::ExtractionChunkCountTooLarge(NUM_CHUNKS));
  assert_eq!(
    (spent.total, spent.peak),
    (0, 0),
    "the preflight reaches the verdict from three `usize`s"
  );
}

/// The boundary, and the ordering: the SMALLEST configuration the chunk-axis
/// COMPUTE cap refuses, and the proof it is applied ahead of the
/// `O(num_chunks)` placement scan.
///
/// `step_samples = 1` over 230 770 samples — 14.42 seconds, 923 080 bytes of
/// `f32` — is the smallest input any legal configuration can turn into 70 771
/// chunks, one past what `MAX_EXTRACTION_CHUNKS` admits. It is also misaligned
/// (`first_misaligned_chunk` fires at chunk 135), which is what makes it an
/// ordering falsifier: if the placement scan ran first, this call would name
/// `MisalignedChunkPlacement`.
///
/// One sample fewer is 70 770 chunks — exactly `MAX_EXTRACTION_CHUNKS`,
/// accepted — and the placement scan then gets to speak, unchanged from round 3.
/// So the bound is a boundary rather than a blanket refusal of small strides,
/// and it did not swallow the guard behind it.
///
/// Round 11 moved this boundary from the byte ceiling to the chunk ceiling
/// WITHOUT moving the boundary: round 10's ceiling was `70_770 * 17_208`, the
/// footprint of exactly this count on the shipped 589-frame grid, so the same
/// 230 770 samples were refused and the same 230 769 accepted. What changed is
/// which bound says so, and that it now says so for every frame count rather
/// than for 589 — pinned below.
#[test]
fn the_chunk_axis_cap_is_a_boundary_and_precedes_the_placement_scan() {
  use crate::audio::speaker::window;

  const FRAMES_PER_CHUNK: usize = 589;
  /// The smallest clip any legal `step_samples` can turn into 70 771 chunks:
  /// `SEG_CHUNK_SAMPLES + 70_770 * 1`.
  const SMALLEST_REFUSED_SAMPLES: usize = 230_770;

  let w = WindowOptions::new().with_step_samples(1);
  let extractor = Extractor::with_options(Options::new().with_window(w));
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();

  let num_chunks = window::num_chunks(SMALLEST_REFUSED_SAMPLES, &w);
  assert_eq!(num_chunks, 70_771);
  assert!(
    checked_output_frame_count(num_chunks, chunks_sw, frames_sw).is_ok(),
    "a 14.42 s clip is nowhere near MAX_OUTPUT_FRAMES"
  );
  assert_eq!(
    window::first_misaligned_chunk(num_chunks, chunks_sw, frames_sw)
      .map(|m| m.chunk())
      .expect("an odd stride ties, so the placement scan has something to say"),
    135
  );
  assert_eq!(
    extractor.checked_geometry(SMALLEST_REFUSED_SAMPLES, FRAMES_PER_CHUNK),
    Err(ExtractError::ExtractionChunkCountTooLarge(70_771)),
    "the chunk bound runs BEFORE the placement scan, which is O(num_chunks) \
     over the very axis it limits"
  );

  // One sample fewer: 70 770 chunks, exactly the ceiling, released to the guard
  // behind it.
  assert_eq!(window::num_chunks(SMALLEST_REFUSED_SAMPLES - 1, &w), 70_770);
  assert_eq!(
    checked_extraction_chunk_count(70_770),
    Ok(MAX_EXTRACTION_CHUNKS),
    "the largest grid admitted sits exactly ON the ceiling"
  );
  assert!(
    matches!(
      extractor.checked_geometry(SMALLEST_REFUSED_SAMPLES - 1, FRAMES_PER_CHUNK),
      Err(ExtractError::MisalignedChunkPlacement(m)) if m.chunk() == 135
    ),
    "one sample fewer passes both chunk-axis bounds and reaches round 3's guard"
  );

  // The boundary is where round 10 put it for the SHIPPED grid — and, unlike
  // round 10's, it is the same boundary at every frame count the output grid
  // can address. Asserted through `checked_geometry`, the seam that actually
  // reads the segmenter's frame count: 70 770 chunks must reach the placement
  // scan behind the bound, and 70 771 must not, for all of them.
  assert_eq!(
    derived_extraction_tensor_bytes(70_770, FRAMES_PER_CHUNK),
    Ok(1_217_810_160),
    "round 10's ceiling was exactly this geometry's footprint at 589 frames"
  );
  for frames in [1usize, 588, 589, 590, 594] {
    assert!(
      matches!(
        extractor.checked_geometry(SMALLEST_REFUSED_SAMPLES - 1, frames),
        Err(ExtractError::MisalignedChunkPlacement(m)) if m.chunk() == 135
      ),
      "frames_per_chunk={frames}: 70 770 chunks must reach round 3's guard"
    );
    assert_eq!(
      extractor.checked_geometry(SMALLEST_REFUSED_SAMPLES, frames),
      Err(ExtractError::ExtractionChunkCountTooLarge(70_771)),
      "frames_per_chunk={frames}: 70 771 must not"
    );
  }
}

/// `MAX_EXTRACTION_CHUNKS` is the frame cap's own chunk allowance, derived here
/// rather than copied into the constant and hoped about.
///
/// The number is the first chunk count `MAX_OUTPUT_FRAMES` refuses at
/// `DEFAULT_STEP_SAMPLES`. This searches for it through
/// `checked_output_frame_count` — the same function the preflight runs — so a
/// change to either cap turns the derivation into a failure instead of a silent
/// drift, the shape `plda_min_norm_is_diarics_own_floor_measured_not_copied`
/// uses for `diaric`'s floor.
///
/// It then pins the two properties that make the bound safe and useful: at the
/// shipped stride it refuses nothing the frame cap admits, and it holds the
/// model-call count for EVERY segmenter the loader accepts — which is the half
/// a byte ceiling cannot do.
#[test]
fn max_extraction_chunks_is_the_frame_caps_own_allowance_derived_not_copied() {
  use crate::audio::speaker::window;

  let w = WindowOptions::new();
  assert_eq!(w.step_samples(), window::DEFAULT_STEP_SAMPLES);
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();

  // The first `num_chunks` the OUTPUT cap refuses at the shipped stride. The
  // derived grid rises with `num_chunks`, so a binary search is exact.
  let (mut lo, mut hi) = (1usize, 1usize << 24);
  assert!(
    checked_output_frame_count(hi, chunks_sw, frames_sw).is_err(),
    "the search needs a refused upper bound"
  );
  while lo < hi {
    let mid = lo + (hi - lo) / 2;
    if checked_output_frame_count(mid, chunks_sw, frames_sw).is_err() {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  let first_refused = lo;
  assert_eq!(first_refused, 70_770);
  assert_eq!(
    MAX_EXTRACTION_CHUNKS, first_refused,
    "the chunk ceiling IS the allowance MAX_OUTPUT_FRAMES already gives at the \
     shipped stride, admitted inclusively"
  );

  // The safety property: at the default stride the frame cap always refuses
  // first, so this bound cannot newly refuse anything the shipped configuration
  // reaches. `first_refused - 1` is the largest grid that cap admits.
  assert_eq!(
    checked_extraction_chunk_count(first_refused - 1),
    Ok(70_769)
  );
  assert_eq!(
    checked_geometry_first_refusal_at_default_stride(),
    ExtractError::OutputFrameCountTooLarge(4_194_312),
    "at the shipped stride the FRAME cap is still the one that speaks"
  );

  // The compute this ceiling buys, per producer. Both call counts are
  // proportional to `num_chunks` with a fixed constant, which is why one
  // chunk-axis constant serves both.
  assert_eq!(2 * MAX_EXTRACTION_CHUNKS, 141_540, "Extractor::extract");
  assert_eq!(
    3 * MAX_EXTRACTION_CHUNKS
      .div_ceil(crate::audio::speaker::source::argmax::ARGMAX_WINDOWS_PER_CHUNK),
    10_110,
    "ArgmaxSource::extract"
  );

  // And the half a byte ceiling cannot hold: the allowance is the same 70 770
  // whatever the loaded segmenter declares — including the loader's own floor
  // of one frame per ten-second chunk, where the memory ceiling is 1 656 bytes
  // away from admitting 393 349 of them. Through `checked_geometry`, so the
  // frame count reaches the seam rather than being asserted around it.
  let fine =
    Extractor::with_options(Options::new().with_window(WindowOptions::new().with_step_samples(1)));
  for frames in [1usize, 2, 589, 594, 51_095_812] {
    assert_eq!(
      fine.checked_geometry(SEG_CHUNK_SAMPLES + MAX_EXTRACTION_CHUNKS, frames),
      Err(ExtractError::ExtractionChunkCountTooLarge(70_771)),
      "frames_per_chunk={frames} must not buy a single extra chunk"
    );
  }
  assert!(
    checked_extraction_tensor_bytes(393_349, 1).is_ok(),
    "the memory ceiling admits 393 349 one-frame chunks — 786 698 model calls"
  );
  assert!(
    checked_extraction_chunk_count(393_349).is_err(),
    "...and the compute ceiling is what refuses them"
  );
}

/// The helper the derivation above uses to show the FRAME cap still speaks
/// first at the shipped stride: the smallest clip whose derived output grid
/// exceeds `MAX_OUTPUT_FRAMES`, run through the producer's own seam.
fn checked_geometry_first_refusal_at_default_stride() -> ExtractError {
  const SMALLEST_OVER_CAP_SAMPLES: usize = 1_132_448_001;
  Extractor::with_options(Options::new())
    .checked_geometry(SMALLEST_OVER_CAP_SAMPLES, 589)
    .expect_err("a 19.66 h clip derives a grid past MAX_OUTPUT_FRAMES")
}

/// `MAX_EXTRACTION_TENSOR_BYTES` is the ADDRESSABLE grid's own footprint,
/// derived here rather than copied into the constant and hoped about.
///
/// The number is `MAX_EXTRACTION_CHUNKS * per_chunk_bytes(594)`, where `594` is
/// the crate's own one-chunk output grid — the number of `FRAME_STEP_S` slots a
/// single `CHUNK_DURATION_S` chunk occupies, and therefore the largest per-chunk
/// frame count the aggregation can address at all. Both factors are derived
/// through the crate's own functions here, so a change to either turns the
/// derivation into a failure instead of a silent drift.
///
/// Round 10 derived it at `589` instead — community-1's frame count — and that
/// is the defect this round closes: a constant calibrated at one model's grid
/// makes acceptance depend on the loaded model's grid. This pins the general
/// property that replaces it, and the fact that at the shipped grid the ceiling
/// cannot speak at all.
#[test]
fn max_extraction_tensor_bytes_is_the_addressable_grids_own_footprint_derived_not_copied() {
  use crate::audio::speaker::window;

  let w = WindowOptions::new();
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();

  // The addressable grid, from the very function the preflight derives output
  // grids with: one chunk's own output-frame span.
  let addressable = derived_output_frame_count(1, chunks_sw, frames_sw)
    .expect("the one-chunk output grid is well defined");
  assert_eq!(addressable, 594);
  assert_eq!(
    (window::CHUNK_DURATION_S / window::FRAME_STEP_S).ceil() as usize,
    593,
    "594 is that span plus the endpoint frame, as try_num_output_frames counts"
  );

  assert_eq!(
    derived_extraction_tensor_bytes(MAX_EXTRACTION_CHUNKS, addressable),
    Ok(MAX_EXTRACTION_TENSOR_BYTES),
    "the ceiling IS the footprint of the largest grid both other bounds admit"
  );
  assert_eq!(MAX_EXTRACTION_TENSOR_BYTES, 70_770 * 17_328);

  // The general property round 10's `70_770 * 17_208` did not have: for EVERY
  // addressable frame count, the largest chunk grid the frame cap admits at the
  // shipped stride fits under the ceiling. Round 10 fails this at 590.
  let largest_admitted = MAX_EXTRACTION_CHUNKS - 1;
  for frames in 1..=addressable {
    assert!(
      checked_extraction_tensor_bytes(largest_admitted, frames).is_ok(),
      "frames_per_chunk={frames}: the frame cap's own largest grid must fit"
    );
  }
  let widest_admitted = derived_extraction_tensor_bytes(largest_admitted, addressable)
    .expect("the frame cap's largest grid at the addressable width");
  assert_eq!(widest_admitted, 1_226_285_232);
  assert!(
    widest_admitted < MAX_EXTRACTION_TENSOR_BYTES,
    "the widest geometry the other two bounds admit must fit under this one"
  );
  assert_eq!(
    derived_extraction_tensor_bytes(70_672, 590),
    Ok(1_217_819_904),
    "round 10 refused this; the frame cap admits it"
  );
  assert!(checked_extraction_tensor_bytes(70_672, 590).is_ok());

  // At the SHIPPED grid this ceiling cannot speak: the chunk bound refuses
  // every count it would have refused, and more.
  let shipped = derived_extraction_tensor_bytes(MAX_EXTRACTION_CHUNKS, 589)
    .expect("the shipped grid at the chunk ceiling");
  assert_eq!(shipped, 1_217_810_160);
  assert!(
    shipped < MAX_EXTRACTION_TENSOR_BYTES,
    "round 10's ceiling was this exact figure, so the shipped grid's largest \
     admitted geometry sat ON it rather than under it"
  );

  // What it DOES hold — the axis the chunk bound cannot see. One chunk, from a
  // segmenter whose declared grid is the thing that is too large.
  let per_chunk = |f: usize| {
    f * SEG_NUM_SLOTS * size_of::<f64>() + SEG_NUM_SLOTS * EMBEDDING_DIM * size_of::<f32>()
  };
  let widest = (MAX_EXTRACTION_TENSOR_BYTES - per_chunk(0)) / (SEG_NUM_SLOTS * size_of::<f64>());
  assert_eq!(widest, 51_095_812);
  assert_eq!(
    checked_extraction_tensor_bytes(1, widest),
    Ok(MAX_EXTRACTION_TENSOR_BYTES),
    "the widest single chunk sits exactly ON the ceiling"
  );
  assert_eq!(
    checked_extraction_tensor_bytes(1, widest + 1),
    Err(ExtractError::ExtractionTensorBytesTooLarge(1_226_302_584)),
    "one frame more is this bound's own smallest refusal"
  );
  assert!(
    checked_extraction_chunk_count(1).is_ok(),
    "...and the chunk bound has nothing to say about a single chunk"
  );
}

/// The discrimination: across the whole legal `step_samples` range the
/// chunk-axis bounds refuse a geometry exactly when its chunk grid passes
/// 70 770, and never for any other reason.
///
/// A cap that refused a band of strides outright, or that varied with something
/// other than the chunk count, would pass the boundary tests above and fail this
/// one. Swept on the reviewer's own ten-minute clip, where the threshold falls
/// at `step_samples = 134`.
///
/// Swept at three per-chunk frame counts, which is the round-11 half: the
/// verdict must be the SAME at 1, 589 and 594 frames. Under round 10's ceiling
/// the 594-frame column turns at a different stride from the 589-frame one, and
/// the 1-frame column does not turn inside the legal range at all.
#[test]
fn the_chunk_axis_cap_tracks_the_chunk_grid_across_every_legal_stride() {
  use crate::audio::speaker::window;

  const SAMPLES: usize = 9_600_000;

  let refuses = |num_chunks: usize, frames: usize| {
    checked_extraction_chunk_count(num_chunks).is_err()
      || checked_extraction_tensor_bytes(num_chunks, frames).is_err()
  };

  for frames in [1usize, 589, 594] {
    let mut boundary = None;
    for step in (1u32..=SEG_CHUNK_SAMPLES as u32).step_by(7) {
      let w = WindowOptions::new().with_step_samples(step);
      let num_chunks = window::num_chunks(SAMPLES, &w);
      let refused = refuses(num_chunks, frames);
      assert_eq!(
        refused,
        num_chunks > MAX_EXTRACTION_CHUNKS,
        "frames={frames} step_samples={step}: {num_chunks} chunks, refused={refused}"
      );
      if !refused && boundary.is_none() {
        boundary = Some(step);
      }
    }
    // The sweep must actually cross the threshold, or it proves nothing.
    assert_eq!(
      boundary,
      Some(134),
      "frames={frames}: the ten-minute clip turns at step 134"
    );

    // Both endpoints exactly, off the sweep's stride-of-7 grid.
    for (step, expected) in [(133u32, true), (134, false)] {
      let w = WindowOptions::new().with_step_samples(step);
      let num_chunks = window::num_chunks(SAMPLES, &w);
      assert_eq!(
        refuses(num_chunks, frames),
        expected,
        "frames={frames} step_samples={step} -> {num_chunks} chunks"
      );
    }
  }
}

/// The overflow arms, which no clip can reach but `try_from_parts`' own check 3
/// can: a geometry whose tensor footprint does not fit in `usize` reports the
/// same `(part, num_chunks, num_frames_per_chunk)` diagnosis check 3 reports,
/// and reports `raw_embeddings` first, so the two never name different parts for
/// a geometry that overflows both.
#[test]
fn derived_extraction_tensor_bytes_overflow_arms_match_check_threes_diagnosis() {
  use crate::audio::speaker::error::{ExtractionGeometryOverflow, ExtractionPart};

  // `num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM * 4` overflows first, and it
  // does not read `num_frames_per_chunk` at all, so a `1` there still names
  // RawEmbeddings.
  assert_eq!(
    derived_extraction_tensor_bytes(usize::MAX, 1),
    Err(ExtractError::ExtractionGeometryOverflow(
      ExtractionGeometryOverflow::new(ExtractionPart::RawEmbeddings, usize::MAX, 1)
    ))
  );

  // A `num_chunks` small enough for the embeddings product but not for the
  // segmentations one: `n * 3 * 256 * 4` fits while `n * f * 3 * 8` does not.
  let n = usize::MAX / 8_192;
  assert!(
    derived_extraction_tensor_bytes(n, 1).is_ok(),
    "the embeddings product must still fit, or this case tests the wrong arm"
  );
  let m = ExtractionGeometryOverflow::new(ExtractionPart::Segmentations, n, usize::MAX);
  assert_eq!(
    derived_extraction_tensor_bytes(n, usize::MAX),
    Err(ExtractError::ExtractionGeometryOverflow(m))
  );
  assert_eq!(
    (m.part(), m.num_chunks(), m.num_frames_per_chunk()),
    (ExtractionPart::Segmentations, n, usize::MAX)
  );

  // Both products fit and their SUM does not. Reported against
  // `Segmentations`, the dominant term.
  let n = usize::MAX / 4_096;
  let frames = 86usize;
  let raw = n * SEG_NUM_SLOTS * EMBEDDING_DIM * size_of::<f32>();
  let seg = n * frames * SEG_NUM_SLOTS * size_of::<f64>();
  assert!(
    raw.checked_add(seg).is_none(),
    "this case needs two products that fit and a sum that does not"
  );
  assert_eq!(
    derived_extraction_tensor_bytes(n, frames),
    Err(ExtractError::ExtractionGeometryOverflow(
      ExtractionGeometryOverflow::new(ExtractionPart::Segmentations, n, frames)
    ))
  );

  // The BYTE products are what is checked, so an element count that fits while
  // its byte size does not is refused here even though check 3 would accept the
  // length. Such a `Vec` is unallocatable regardless (`isize::MAX` bytes).
  let elems = usize::MAX / (SEG_NUM_SLOTS * EMBEDDING_DIM);
  assert!(
    elems
      .checked_mul(SEG_NUM_SLOTS)
      .and_then(|v| v.checked_mul(EMBEDDING_DIM))
      .is_some(),
    "check 3's element product fits"
  );
  assert!(
    derived_extraction_tensor_bytes(elems, 1).is_err(),
    "...while its byte size does not"
  );
}

/// The alternative cure, measured and rejected: a `step_samples` floor at one
/// output-frame step would have left the tensors unbounded.
///
/// There is a real threshold to put such a floor at — `FRAME_STEP_S *
/// SAMPLE_RATE_HZ` is exactly 270 samples, the point below which
/// `aggregate_chunk_start_frame` maps consecutive chunks onto the same output
/// frame. This pins why that was not the fix: AT that floor, with every existing
/// guard in force, `MAX_OUTPUT_FRAMES` still admits a chunk grid worth 67.21 GiB
/// of extraction tensors. A floor caps the amplification and not the total, so
/// it is a bound on a proxy — which is the shape of defect this branch keeps
/// finding.
///
/// Kept so a later round cannot replace the byte ceiling with the floor and
/// believe the axis is closed.
#[test]
fn a_step_samples_floor_at_one_frame_step_would_not_have_bounded_the_tensors() {
  use crate::audio::speaker::window;

  const FRAMES_PER_CHUNK: usize = 589;

  // The threshold itself, exactly representable and exactly 270.
  let one_frame_step_in_samples = window::FRAME_STEP_S * f64::from(window::SAMPLE_RATE_HZ);
  assert_eq!(one_frame_step_in_samples, 270.0);
  let floor = 270u32;
  assert!(
    window::DEFAULT_STEP_SAMPLES > floor,
    "the shipped stride is 59.26 frame steps, far above any such floor"
  );

  // The largest chunk grid MAX_OUTPUT_FRAMES admits AT that floor.
  let w = WindowOptions::new().with_step_samples(floor);
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();
  let (mut lo, mut hi) = (1usize, 1usize << 25);
  assert!(checked_output_frame_count(hi, chunks_sw, frames_sw).is_err());
  while lo < hi {
    let mid = lo + (hi - lo) / 2;
    if checked_output_frame_count(mid, chunks_sw, frames_sw).is_err() {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  let largest_admitted = lo - 1;
  assert_eq!(largest_admitted, 4_193_711);

  // ...and what a producer would then allocate for it.
  assert_eq!(
    derived_extraction_tensor_bytes(largest_admitted, FRAMES_PER_CHUNK),
    Ok(72_165_378_888),
    "67.21 GiB, still reachable with a 270-sample floor in force"
  );
  assert_eq!(
    checked_extraction_tensor_bytes(largest_admitted, FRAMES_PER_CHUNK),
    Err(ExtractError::ExtractionTensorBytesTooLarge(72_165_378_888)),
    "only the byte ceiling refuses it"
  );
}

/// Round 11, half A: the BYTE ceiling does not bound COMPUTE.
///
/// Round 10 closed the chunk axis by bounding the bytes the chunk grid implies,
/// and argued that no second constant was needed because "memory and compute do
/// not diverge". They do. The byte total is `num_chunks *
/// (num_frames_per_chunk * SEG_NUM_SLOTS * 8 + SEG_NUM_SLOTS * EMBEDDING_DIM *
/// 4)`, and the frame count comes from the LOADED SEGMENTER —
/// `SegmentModel::from_dir_with` accepts any declared `segments` shape with
/// `shape[1] >= 1` (`segment/mod.rs`, "a zero-frame model would load fine").
///
/// At one frame per chunk a chunk costs 3 096 bytes, so the byte ceiling divides
/// into 393 349 of them. 946 695 samples — 59.17 seconds — at `step_samples =
/// 2` derives exactly that grid: 1 217 808 504 bytes, 1 656 UNDER round 10's
/// ceiling, an output grid of 3 507 frames against a cap of 4 194 304, and an
/// even stride so the placement scan finds no tie. `Extractor::extract` then
/// issues one segmentation call and one batched embedding call per chunk:
/// **786 698 model invocations for 59.17 seconds of audio**.
///
/// Round 10's own doc named that figure and waved it through as "the degenerate
/// limit of a segmenter emitting one frame per ten seconds". A degenerate
/// segmenter is a segmenter the loader accepts, so it is a reachable
/// denial-of-service surface, not a limit.
///
/// Asserted through `Extractor::checked_geometry`, which IS `extract`'s
/// pre-inference sequence, for the reason rounds 9 and 10 attach there: 786 698
/// CoreML calls is not a unit test on any host.
#[test]
fn the_byte_ceiling_alone_does_not_bound_the_model_call_count() {
  use crate::audio::speaker::window;

  /// A segmenter emitting ONE frame per ten-second chunk — the loader's own
  /// floor (`shape[1] >= 1`).
  const FRAMES_PER_CHUNK: usize = 1;
  /// A `step_samples` `WindowOptions` accepts: it guards `0` and
  /// `> SEG_CHUNK_SAMPLES`, nothing between.
  const STEP: u32 = 2;
  /// 59.17 s at 16 kHz — 3 786 780 bytes of `f32`.
  const SAMPLES: usize = 946_695;
  /// `1 + (946_695 - 160_000).div_ceil(2)`.
  const NUM_CHUNKS: usize = 393_349;
  /// `NUM_CHUNKS * (1 * 3 * 8 + 3 * 256 * 4)` = `NUM_CHUNKS * 3_096`.
  const TENSOR_BYTES: usize = 1_217_808_504;
  /// One `SegmentModel::infer` plus one batched `EmbedModel` call per chunk.
  const MODEL_CALLS: usize = 2 * NUM_CHUNKS;

  let w = WindowOptions::new().with_step_samples(STEP);
  let extractor = Extractor::with_options(Options::new().with_window(w));
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();
  let num_chunks = window::num_chunks(SAMPLES, &w);
  assert_eq!(num_chunks, NUM_CHUNKS);

  // ── Every OTHER guard accepts this geometry, round 10's included ────
  assert_eq!(
    checked_output_frame_count(num_chunks, chunks_sw, frames_sw),
    Ok(3_507),
    "the output-frame cap sees a 59.17 s clip and passes it"
  );
  assert_eq!(
    window::first_misaligned_chunk(num_chunks, chunks_sw, frames_sw),
    None,
    "and the even stride clears the placement scan"
  );
  assert_eq!(
    derived_extraction_tensor_bytes(num_chunks, FRAMES_PER_CHUNK),
    Ok(TENSOR_BYTES),
    "1 217 808 504 bytes — under round 10's 1 217 810 160 ceiling by 1 656"
  );
  assert_eq!(MODEL_CALLS, 786_698);

  // ── The compute bound, at the producer's own earliest point ────────
  assert_eq!(
    extractor.checked_geometry(SAMPLES, FRAMES_PER_CHUNK),
    Err(ExtractError::ExtractionChunkCountTooLarge(NUM_CHUNKS)),
    "786 698 model invocations for 59.17 s of audio must be refused from \
     `samples.len()`, `step_samples` and the segmenter's frame count alone"
  );
}

/// Round 11, half B: acceptance must not depend on the loaded model's frame
/// count.
///
/// Round 10 derived its ceiling as `70_770 * 17_208`, where 17 208 is one
/// chunk's cost ON COMMUNITY-1'S 589-FRAME GRID, and justified it as "the tensor
/// footprint of the very geometry the existing cap already declines" — so
/// nothing the frame cap admits is newly refused. That coincidence holds at 589
/// and nowhere else. On a 590-frame grid a chunk costs 17 232 bytes, so the same
/// ceiling divides into 70 671 chunks while `MAX_OUTPUT_FRAMES` at the default
/// stride still admits 70 769.
///
/// 1 130 880 001 samples at the shipped stride derive 70 672 chunks and
/// 1 217 819 904 bytes — 9 744 past round 10's ceiling — with an output grid of
/// 4 188 505 frames, 5 799 BELOW `MAX_OUTPUT_FRAMES`. `main` has no byte cap and
/// accepts it; round 10 refuses it. That half is a regression this branch
/// introduced, and it is the half a model-relative derivation fixes.
///
/// The property, not the one clip: for every per-chunk frame count the crate's
/// own output grid can address, the largest chunk grid the frame cap admits at
/// the default stride must stay inside the byte ceiling.
#[test]
fn the_byte_ceiling_admits_every_frame_grid_the_frame_cap_admits() {
  use crate::audio::speaker::window;

  const FRAMES_PER_CHUNK: usize = 590;
  /// The clip the finding names: 70 672 chunks at the shipped stride.
  const SAMPLES: usize = 1_130_880_001;

  let w = WindowOptions::new();
  assert_eq!(w.step_samples(), window::DEFAULT_STEP_SAMPLES);
  let extractor = Extractor::with_options(Options::new().with_window(w));
  let chunks_sw = window::chunk_sliding_window(&w);
  let frames_sw = window::frame_sliding_window();

  let num_chunks = window::num_chunks(SAMPLES, &w);
  assert_eq!(num_chunks, 70_672);
  assert_eq!(
    checked_output_frame_count(num_chunks, chunks_sw, frames_sw),
    Ok(4_188_505),
    "the output grid stays below MAX_OUTPUT_FRAMES"
  );
  const { assert!(4_188_505 < MAX_OUTPUT_FRAMES) };
  assert_eq!(
    derived_extraction_tensor_bytes(num_chunks, FRAMES_PER_CHUNK),
    Ok(1_217_819_904)
  );
  assert_eq!(
    extractor.checked_geometry(SAMPLES, FRAMES_PER_CHUNK),
    Ok((num_chunks, chunks_sw, frames_sw)),
    "a 590-frame segmenter at the SHIPPED stride must not be refused where the \
     frame cap accepts"
  );

  // The property across the whole addressable frame range: one chunk's frames
  // are placed consecutively from `aggregate_chunk_start_frame`, so the finest
  // grid this crate's own output axis can address is the one-chunk output grid.
  let addressable = derived_output_frame_count(1, chunks_sw, frames_sw)
    .expect("the one-chunk output grid is well defined");
  assert_eq!(addressable, 594);
  let largest_admitted = window::num_chunks(1_132_448_001 - 1, &w);
  assert_eq!(largest_admitted, 70_769);
  for frames in 1..=addressable {
    assert!(
      checked_extraction_tensor_bytes(largest_admitted, frames).is_ok(),
      "frames_per_chunk={frames}: the frame cap's own largest grid must fit"
    );
  }
}
