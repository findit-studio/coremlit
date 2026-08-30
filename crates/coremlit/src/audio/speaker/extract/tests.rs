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
fn try_from_parts_accepts_a_soft_active_slot_but_requires_the_count_to_include_it() {
  // Soft (non-binary) segmentation values are still accepted; what changed in
  // round 2 is that `count` must AGREE with them.
  //
  // Round 1 read this the other way: it accepted a `count` derived at
  // `onset = 0.5` while the check used `seg > 0.0`, on the reasoning that
  // `try_from_parts` cannot know which threshold a producer binarized with. But
  // NEITHER backend reads an onset — `diarize_online`'s activity scan and dia's
  // `filter_embeddings` both use `seg > 0.0` — so an onset-derived `count` under
  // a sub-onset column is the finding-A divergence again with a soft value in
  // place of a fabricated one: offline, silence at that frame; online, a
  // speaker. `seg > 0.0` is the ONE predicate the two share, so it is the one
  // the equality is taken over. `extract()`'s own `count` still satisfies it:
  // it aggregates `seg >= onset` over a hard `0.0`/`1.0` multilabel, on which
  // the two predicates coincide for every `onset` in `(0.0, 1.0]`.
  //
  // Slot 1's column carries `0.3`: nonzero, below the default `onset` of 0.5,
  // and ACTIVE to both engines. `count` must therefore be `[1, 1, 0, 0]`.
  let mut raw = one_usable_slot_row(0);
  raw[EMBEDDING_DIM..EMBEDDING_DIM + 64].fill(1.0); // slot 1: usable too
  let parts = ExtractionParts {
    raw_embeddings: raw,
    segmentations: vec![1.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    count: vec![1, 1, 0, 0],
    num_chunks: 1,
    num_frames_per_chunk: 2,
    chunks_sw: crate::audio::speaker::window::chunk_sliding_window(&WindowOptions::new())
      .with_duration(3.0 * crate::audio::speaker::window::FRAME_STEP_S),
    frames_sw: crate::audio::speaker::window::frame_sliding_window(),
  };
  let e = Extraction::try_from_parts(parts.clone()).expect("a soft column is not a defect");
  assert_eq!(e.num_output_frames(), 4);

  // The onset-derived count round 1 accepted is now named, at the frame the
  // sub-onset column occupies.
  let under = ExtractionParts {
    count: vec![1, 0, 0, 0],
    ..parts.clone()
  };
  let err = refused(under);
  assert!(
    matches!(err, ExtractError::CountNotSegmentationDerived(c)
      if (c.frame(), c.got(), c.expected()) == (1, 0, 1)),
    "expected CountNotSegmentationDerived(1, 0, 1), got {err:?}"
  );

  // And the sub-onset column IS an active column, so a zero row under it is
  // still the finding-1 defect — the online engine would read that slot as
  // "no speaker" while its segmentation says otherwise.
  let broken = ExtractionParts {
    raw_embeddings: one_usable_slot_row(0), // slot 1 back to all-zero
    ..parts
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

  // Agreement with PLDA's own admission test, straddling the floor. Equality
  // over the whole sweep is the property: a floor that drifts in either
  // direction shows up as a row one side keeps and the other refuses.
  let mut disagreements = 0;
  for micro in 9_000..11_000u32 {
    let v = f64::from(micro) / 1_000_000.0;
    let mut row = [0.0f32; EMBEDDING_DIM];
    row[0] = v as f32;
    let mine = raw_embedding_reaches_plda(&row);
    let plda = diaric::plda::RawEmbedding::from_wespeaker(row).is_ok();
    if mine != plda {
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
  assert!(!raw_embedding_reaches_plda(&between));
  assert!(diaric::embed::Embedding::normalize_from(between).is_some());

  let mut above = [0.0f32; EMBEDDING_DIM];
  above[0] = 0.02;
  assert!(raw_embedding_reaches_plda(&above));

  // Non-finite is refused by the same predicate, matching `from_raw_array`'s
  // own leading finiteness scan — a `+inf` row has an INFINITE norm, which a
  // bare `norm >= floor` comparison would have admitted.
  let mut infinite = above;
  infinite[1] = f32::INFINITY;
  assert!(!raw_embedding_reaches_plda(&infinite));
  assert!(diaric::plda::RawEmbedding::from_wespeaker(infinite).is_err());
  let mut nan = above;
  nan[1] = f32::NAN;
  assert!(!raw_embedding_reaches_plda(&nan));
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

  for row in &probes {
    let online = diaric::embed::Embedding::normalize_from(*row).is_some();
    let offline = diaric::plda::RawEmbedding::from_wespeaker(*row).is_ok();
    assert_eq!(
      raw_embedding_reaches_plda(row),
      online && offline,
      "row [{}, {}, 0, …] — online accepts {online}, offline accepts {offline}",
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

  // A wrong-length row is refused rather than panicked on. Unreachable in
  // crate — every call site slices exactly `EMBEDDING_DIM` — but the array
  // conversion is what makes it so, and this pins which way it fails.
  assert!(!raw_embedding_reaches_plda(&[1.0f32; EMBEDDING_DIM - 1]));
  assert!(!raw_embedding_reaches_plda(&[1.0f32; EMBEDDING_DIM + 1]));
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
