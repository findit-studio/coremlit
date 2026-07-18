//! **Cross-crate OFFLINE equivalence — the executable gate under the dual-crate
//! DER oracle, for the COMPLETE shared offline path (not just PLDA).**
//!
//! `plda_cross_crate_equivalence.rs` proves dia's and diaric's PLDA projections
//! are bit-identical. But projection is only ONE stage of the shared offline DER
//! path: after it, each crate independently runs the community-1 filter
//! (`MIN_ACTIVE_RATIO`), AHC, VBx, the Hungarian re-assignment, reconstruction,
//! and span conversion (`offline/algo.rs` in each crate). The DER oracle carries
//! TWO separately-pinned copies of that whole chain — dia's on the REFERENCE
//! (dia-ort) side and diaric's on the MEASURED (speakerkit) side — so "the two
//! clusterers cannot diverge" is an assumption the PLDA gate does NOT cover.
//!
//! This gate turns that assumption into an EXECUTABLE invariant. It builds
//! scripted typed offline inputs, feeds each to BOTH `diarize_offline`
//! implementations, and asserts their WHOLE observable output is identical:
//! typed outcome, per-chunk hard clusters, the discrete diarization grid
//! (bit-exact `f32`), the cluster count, and the RTTM spans (bit-exact `f64`
//! start/duration + cluster). Two fixtures feed the gate:
//!
//! 1. [`offline_cross_crate_equivalence`] — the baseline: single-active
//!    segmentations rich enough that the community-1 filter keeps many training
//!    pairs, AHC merges exact ties, and VBx actually iterates.
//! 2. [`offline_cross_crate_equivalence_activity_boundary`] — reaches the two
//!    community-1 activity-filter branches the baseline never exercises: the
//!    `MIN_ACTIVE_RATIO = 0.2` clean-frame threshold (an EDGE pair whose 5/20
//!    clean frames sit in the (0.2, 0.3) band — retained at 0.2, dropped by a
//!    0.3 re-pin — and which forms its OWN cluster, so its retention is visible
//!    in the compared output) and the `active_count == 1` single-active gate (a
//!    pair whose activity is mostly in genuinely overlapping frames, so those
//!    frames are rejected from its clean-frame tally).
//!
//! A future re-pin of either crate that changes an AHC tie, the
//! `MIN_ACTIVE_RATIO` boundary, the `active_count == 1` overlap rejection, a VBx
//! step, the reassignment, or reconstruction — while leaving PLDA untouched —
//! moves one side and fails here. The `MIN_ACTIVE_RATIO` and overlap coverage is
//! fixture 2's; a local shim of one crate's ratio to 0.3 turns the boundary gate
//! red (mutation-verified during development).
//!
//! # Why an INPUT-perturbation mutation proves drift-detection
//! The fence's own drift scenario (re-pin one crate to a diverging revision) is
//! not reproducible inside a single `cargo test` build: both crates are fixed at
//! their pinned revs. So the companion test
//! [`offline_cross_crate_gate_detects_a_one_sided_divergence`] instead proves the
//! gate's SENSITIVITY directly — it feeds the two crates deliberately DIFFERENT
//! inputs (one training slot's identity swapped on ONE side, exactly the "perturb
//! one side" mutation) and asserts the comparison REPORTS the divergence. That
//! establishes the property the re-pin fence needs: when the two shared clusterers
//! see different data (the observable symptom of a diverged post-PLDA stage), this
//! gate goes red rather than green. The equivalence assertion above is therefore
//! non-vacuous.
//!
//! Hermetic + ORDINARY (never `#[ignore]`d): PLDA weights are compile-time
//! embedded in both crates (`PldaTransform::new()` needs no `Models/`), the
//! inputs are scripted, and the offline path is ort-free — so this runs in the
//! `dia-oracle` ordinary suite with no model tree and no fixtures, same as the
//! PLDA gate.
#![cfg(feature = "dia-oracle")]

use speakerkit::window::{
  SlidingWindow, WindowOptions, chunk_sliding_window, frame_sliding_window,
};

// ── Scripted-input geometry ──────────────────────────────────────────
// Small but rich enough to reach the failure regime: 8 chunks × 3 slots, one
// active slot per chunk cycling slot = c % 3 and identity = c % 3, so three
// distinct WeSpeaker identities recur with EXACT duplicates (AHC ties) across 8
// surviving training pairs (VBx then iterates over the ≥3-cluster init).
const NUM_CHUNKS: usize = 8;
const NUM_SPEAKERS: usize = 3; // == diaric/dia MAX_SPEAKER_SLOTS
const FRAMES: usize = 10; // MIN_ACTIVE_RATIO * 10 = 2 clean frames to survive
const ACTIVE_FRAMES: usize = 5; // clean frames per active slot (>= 2)
const EMBED_DIM: usize = 256; // == {diaric,dia}::plda::EMBEDDING_DIMENSION
const ONSET: f32 = 0.5;

/// `splitmix64` — the same tiny deterministic PRNG the PLDA gate uses, so the
/// scripted embeddings are identical across runs and byte-identical between the
/// two crates (same bytes in ⇒ same projection ⇒ any output divergence is the
/// post-PLDA math, never the input).
fn splitmix64(state: &mut u64) -> u64 {
  *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
  let mut z = *state;
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^ (z >> 31)
}

/// A deterministic raw WeSpeaker-shaped embedding: 256 `f32` in `[-0.4, 0.4)`,
/// clearing both crates' norm floors so PLDA returns `Ok` (same distribution the
/// PLDA gate's `synth_raw` uses).
fn synth_raw(seed: u64) -> [f32; EMBED_DIM] {
  let mut state = seed;
  let mut arr = [0.0f32; EMBED_DIM];
  for slot in &mut arr {
    let unit = (splitmix64(&mut state) >> 40) as f32 / 16_777_216.0_f32;
    *slot = (unit - 0.5) * 0.8;
  }
  arr
}

/// The three distinct identity prototypes (fixed seeds), each recurring as an
/// exact duplicate across chunks so AHC sees distance-0 ties.
fn prototype(identity: usize) -> [f32; EMBED_DIM] {
  synth_raw(0xA1CE_0000_u64.wrapping_add(identity as u64))
}

/// The owned scripted tensors, shared verbatim by both crates' `OfflineInput`.
struct OfflineFixture {
  raw_embeddings: Vec<f32>,
  segmentations: Vec<f64>,
  count: Vec<u8>,
  num_output_frames: usize,
  /// Chunk/frame geometry carried on the fixture so both the baseline and the
  /// activity-boundary fixture (which have DIFFERENT shapes) reuse the same
  /// `{diaric,dia}_outcome` runners. `num_speakers` is always `NUM_SPEAKERS`
  /// (== `MAX_SPEAKER_SLOTS`), so it stays a const.
  num_chunks: usize,
  num_frames: usize,
}

/// Build the scripted offline input. `identity_of(c)` selects chunk `c`'s active
/// identity — the sole knob the sensitivity test perturbs on one side.
fn build_fixture(identity_of: impl Fn(usize) -> usize) -> OfflineFixture {
  let mut raw_embeddings = vec![0.0f32; NUM_CHUNKS * NUM_SPEAKERS * EMBED_DIM];
  let mut segmentations = vec![0.0f64; NUM_CHUNKS * FRAMES * NUM_SPEAKERS];

  for c in 0..NUM_CHUNKS {
    let s = c % NUM_SPEAKERS; // active slot for this chunk
    // Activate slot `s` over the first ACTIVE_FRAMES frames (single-active, so
    // every one is a clean frame): survives the MIN_ACTIVE_RATIO filter.
    for f in 0..ACTIVE_FRAMES {
      segmentations[(c * FRAMES + f) * NUM_SPEAKERS + s] = 1.0;
    }
    // The active slot's raw embedding is this chunk's identity prototype; the
    // other (inactive, filtered-out) slots stay zero.
    let proto = prototype(identity_of(c));
    let base = (c * NUM_SPEAKERS + s) * EMBED_DIM;
    raw_embeddings[base..base + EMBED_DIM].copy_from_slice(&proto);
  }

  // Derive count + num_output_frames from the SAME community-1 geometry the
  // runtime uses, guaranteeing a reconstruct-valid tensor set.
  let w = WindowOptions::new();
  let count = speakerkit::window::count_from_segmentations(
    &segmentations,
    NUM_CHUNKS,
    FRAMES,
    NUM_SPEAKERS,
    ONSET,
    chunk_sliding_window(&w),
    frame_sliding_window(),
  );
  let num_output_frames = count.len();
  OfflineFixture {
    raw_embeddings,
    segmentations,
    count,
    num_output_frames,
    num_chunks: NUM_CHUNKS,
    num_frames: FRAMES,
  }
}

/// The full observable output of one `diarize_offline`, lifted out of each
/// crate's own `OfflineOutput` type into a crate-neutral, exactly-comparable
/// form: the typed outcome, the per-chunk hard clusters, the bit-exact discrete
/// grid, the cluster count, and the bit-exact spans. `Err` is captured as its
/// `Debug` rendering (the shared error variants render identically across crates).
#[derive(Debug, PartialEq)]
enum Outcome {
  Ok {
    hard_clusters: Vec<[i32; NUM_SPEAKERS]>,
    num_clusters: usize,
    grid_bits: Vec<u32>,
    spans: Vec<(usize, u64, u64)>, // (cluster, start.to_bits, duration.to_bits)
  },
  Err(String),
}

fn diaric_outcome(fx: &OfflineFixture) -> Outcome {
  diaric_outcome_cs(fx, chunk_sliding_window(&WindowOptions::new()))
}

/// As [`diaric_outcome`], but with an EXPLICIT chunk sliding window. The baseline
/// and boundary fixtures pass the default community-1 window (its ~1 s step lands
/// consecutive chunks ~59 output frames apart, so no output cell is multiply
/// covered); the overlap-add fixture passes a small-step window so chunks OVERLAP
/// in output-frame space. The frame window is always the fixed community-1 grid.
fn diaric_outcome_cs(fx: &OfflineFixture, cs: SlidingWindow) -> Outcome {
  let plda = diaric::plda::PldaTransform::new().expect("diaric hermetic PLDA");
  let input = diaric::offline::OfflineInput::new(
    &fx.raw_embeddings,
    fx.num_chunks,
    NUM_SPEAKERS,
    &fx.segmentations,
    fx.num_frames,
    &fx.count,
    fx.num_output_frames,
    cs.into(),
    frame_sliding_window().into(),
    &plda,
  );
  match diaric::offline::diarize_offline(&input) {
    Ok(o) => Outcome::Ok {
      hard_clusters: o.hard_clusters_slice().to_vec(),
      num_clusters: o.num_clusters(),
      grid_bits: o
        .discrete_diarization_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect(),
      spans: o
        .spans_slice()
        .iter()
        .map(|s| (s.cluster(), s.start().to_bits(), s.duration().to_bits()))
        .collect(),
    },
    Err(e) => Outcome::Err(format!("{e:?}")),
  }
}

fn dia_outcome(fx: &OfflineFixture) -> Outcome {
  dia_outcome_cs(fx, chunk_sliding_window(&WindowOptions::new()))
}

/// As [`dia_outcome`], but with an EXPLICIT chunk sliding window — the dia-side
/// mirror of [`diaric_outcome_cs`], used identically by the overlap-add fixture.
fn dia_outcome_cs(fx: &OfflineFixture, cs: SlidingWindow) -> Outcome {
  let plda = dia::plda::PldaTransform::new().expect("dia hermetic PLDA");
  let fs = frame_sliding_window();
  let input = dia::offline::OfflineInput::new(
    &fx.raw_embeddings,
    fx.num_chunks,
    NUM_SPEAKERS,
    &fx.segmentations,
    fx.num_frames,
    &fx.count,
    fx.num_output_frames,
    dia::reconstruct::SlidingWindow::new(cs.start(), cs.duration(), cs.step()),
    dia::reconstruct::SlidingWindow::new(fs.start(), fs.duration(), fs.step()),
    &plda,
  );
  match dia::offline::diarize_offline(&input) {
    Ok(o) => Outcome::Ok {
      hard_clusters: o.hard_clusters_slice().to_vec(),
      num_clusters: o.num_clusters(),
      grid_bits: o
        .discrete_diarization_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect(),
      spans: o
        .spans_slice()
        .iter()
        .map(|s| (s.cluster(), s.start().to_bits(), s.duration().to_bits()))
        .collect(),
    },
    Err(e) => Outcome::Err(format!("{e:?}")),
  }
}

#[test]
fn offline_cross_crate_equivalence() {
  // Same identity on both sides: chunk c → identity c % 3.
  let fx = build_fixture(|c| c % NUM_SPEAKERS);
  let diaric = diaric_outcome(&fx);
  let dia = dia_outcome(&fx);

  assert_eq!(
    diaric, dia,
    "diaric and dia diverged on the shared offline path (filter / AHC / VBx / \
     reassignment / reconstruct / spans) for identical inputs"
  );

  // Non-vacuity: the input actually reached the failure regime — AHC + VBx
  // produced a non-trivial multi-cluster partition with real spans, so a
  // post-PLDA divergence would MANIFEST in the compared output above.
  match &diaric {
    Outcome::Ok {
      hard_clusters,
      num_clusters,
      spans,
      ..
    } => {
      assert!(
        *num_clusters >= 2,
        "fixture must exercise multi-cluster AHC/VBx (got {num_clusters} clusters)"
      );
      let distinct: std::collections::BTreeSet<i32> = hard_clusters
        .iter()
        .flatten()
        .copied()
        .filter(|&k| k >= 0)
        .collect();
      assert!(
        distinct.len() >= 2,
        "fixture must assign >= 2 distinct speakers (got {distinct:?})"
      );
      assert!(!spans.is_empty(), "fixture must produce spans");
      println!(
        "[offline-cross-crate] diaric == dia: {num_clusters} clusters, {} distinct speakers, \
         {} spans, {} grid cells — all bit-exact",
        distinct.len(),
        spans.len(),
        match &diaric {
          Outcome::Ok { grid_bits, .. } => grid_bits.len(),
          Outcome::Err(_) => 0,
        }
      );
    }
    Outcome::Err(e) => panic!("expected a clustered result on the scripted fixture, got Err: {e}"),
  }
}

#[test]
fn offline_cross_crate_gate_detects_a_one_sided_divergence() {
  // SENSITIVITY / non-vacuity guard (the "perturb one side" mutation made
  // permanent): feed diaric the baseline identities and dia a version with ONE
  // chunk's identity swapped, so the two clusterers see different data — the
  // observable symptom of a diverged post-PLDA stage. The comparison MUST report
  // the divergence; if it did not, the equivalence assertion above would be
  // vacuous.
  let baseline = build_fixture(|c| c % NUM_SPEAKERS);
  // Swap chunk 0's identity from 0 to 1: a single training slot moves to another
  // speaker, which flips the hard-cluster assignment / grid / spans.
  let perturbed = build_fixture(|c| if c == 0 { 1 } else { c % NUM_SPEAKERS });

  let diaric = diaric_outcome(&baseline);
  let dia = dia_outcome(&perturbed);
  assert_ne!(
    diaric, dia,
    "the cross-crate gate FAILED to detect a one-sided input perturbation — the equivalence \
     comparison is vacuous and would not catch a real post-PLDA drift"
  );
}

// ── Activity-boundary fixture ─────────────────────────────────────────
// A SECOND scripted input built to REACH the community-1 activity filter's two
// decision branches, which the single-active baseline above never touches: the
// `MIN_ACTIVE_RATIO = 0.2` clean-frame threshold and the `active_count == 1`
// single-active gate. 7 chunks × 3 slots × 20 frames, so the retention bars are
// `0.2 * 20 = 4` and `0.3 * 20 = 6` clean frames.
//
// Only TWO identities carry training weight, and they are chosen to be the pair
// that clusters SEPARATELY under the crates' AHC threshold (0.6): identity 0 is
// the base (>> bar, always retained) and identity 1 is the EDGE (in the flip
// band). Each recurs across THREE chunks as exact duplicates, so both stay
// VBx-alive and each forms its own cluster — the edge identity's whole cluster
// therefore appears at 0.2 and VANISHES under a 0.3 re-pin, which is the
// observable change the mutation flips.
const B_NUM_CHUNKS: usize = 7;
const B_FRAMES: usize = 20;
const B_BASE_ID: usize = 0; // base identity: >> bar clean frames, always retained
const B_EDGE_ID: usize = 1; // edge identity: 5/20 clean frames, in the (0.2, 0.3) band
// Base pairs: >> 6 clean frames each, so identity 0 survives ANY ratio in
// (0, 0.6] and is NOT what the 0.2-vs-0.3 flip is about.
const B_BASE_CLEAN: usize = 12;
// Edge pairs: 5/20 = 0.25 clean frames — RETAINED at 0.2 (5 >= 4), DROPPED by a
// 0.3 re-pin (5 < 6). Mirrors the finding's 25/100 example.
const B_EDGE_CLEAN: usize = 5;
// OVERLAP chunk (last): slot 0 (identity 2) active over [0, B_OV_ACTIVE); slot 1
// (identity 3) active over [0, B_OV_OVERLAP). The first B_OV_OVERLAP frames are
// DOUBLE-active (active_count == 2 → NOT single-active), so only the tail
// [B_OV_OVERLAP, B_OV_ACTIVE) is clean for slot 0: clean = 8 - 6 = 2 < 4. Slot 0
// is thus dropped BECAUSE of the single-active gate even though its raw activity
// (8 frames) clears the bar — making the `active_count == 1` gate LOAD-BEARING.
const B_OV_ACTIVE: usize = 8;
const B_OV_OVERLAP: usize = 6;
const B_OV_CHUNK: usize = B_NUM_CHUNKS - 1; // 6

/// Write a `1.0` activation for slot `s` of chunk `c` over frames `[f0, f1)`.
fn set_active(seg: &mut [f64], num_frames: usize, c: usize, s: usize, f0: usize, f1: usize) {
  for f in f0..f1 {
    seg[(c * num_frames + f) * NUM_SPEAKERS + s] = 1.0;
  }
}

/// Write identity `identity`'s prototype embedding into slot `s` of chunk `c`.
fn set_proto(raw: &mut [f32], c: usize, s: usize, identity: usize) {
  let base = (c * NUM_SPEAKERS + s) * EMBED_DIM;
  raw[base..base + EMBED_DIM].copy_from_slice(&prototype(identity));
}

/// Build the activity-boundary fixture (geometry documented in the const block
/// above). Reuses the same `count_from_segmentations` derivation as
/// `build_fixture`, so the tensor set is reconstruct-valid despite the genuine
/// overlap in the last chunk.
fn build_boundary_fixture() -> OfflineFixture {
  let mut raw_embeddings = vec![0.0f32; B_NUM_CHUNKS * NUM_SPEAKERS * EMBED_DIM];
  let mut segmentations = vec![0.0f64; B_NUM_CHUNKS * B_FRAMES * NUM_SPEAKERS];

  // Base cluster (identity 0): chunks 0,1,2, slot 0, exact duplicates, >> bar.
  for c in 0..3 {
    set_active(&mut segmentations, B_FRAMES, c, 0, 0, B_BASE_CLEAN);
    set_proto(&mut raw_embeddings, c, 0, B_BASE_ID);
  }
  // EDGE cluster (identity 1): chunks 3,4,5, slot 1, exact duplicates, each with
  // exactly B_EDGE_CLEAN single-active clean frames (the (0.2, 0.3) flip band).
  for c in 3..6 {
    set_active(&mut segmentations, B_FRAMES, c, 1, 0, B_EDGE_CLEAN);
    set_proto(&mut raw_embeddings, c, 1, B_EDGE_ID);
  }
  // OVERLAP chunk: slot 0 (identity 2) over [0, 8), slot 1 (identity 3) over
  // [0, 6) — the first 6 frames are double-active, so slot 0's clean tally is
  // only 2 and it is dropped by the single-active gate.
  set_active(&mut segmentations, B_FRAMES, B_OV_CHUNK, 0, 0, B_OV_ACTIVE);
  set_proto(&mut raw_embeddings, B_OV_CHUNK, 0, 2);
  set_active(&mut segmentations, B_FRAMES, B_OV_CHUNK, 1, 0, B_OV_OVERLAP);
  set_proto(&mut raw_embeddings, B_OV_CHUNK, 1, 3);

  let w = WindowOptions::new();
  let count = speakerkit::window::count_from_segmentations(
    &segmentations,
    B_NUM_CHUNKS,
    B_FRAMES,
    NUM_SPEAKERS,
    ONSET,
    chunk_sliding_window(&w),
    frame_sliding_window(),
  );
  let num_output_frames = count.len();
  OfflineFixture {
    raw_embeddings,
    segmentations,
    count,
    num_output_frames,
    num_chunks: B_NUM_CHUNKS,
    num_frames: B_FRAMES,
  }
}

/// Test-local, byte-faithful re-derivation of the community-1 activity filter
/// (`{diaric,dia}::offline::algo` Stage 1): the `active_count == 1` single-active
/// gate, then the `clean_frames >= ratio * num_frames` retention rule. It exists
/// ONLY so the fixture's frame geometry can be pinned to the exact decision
/// zones the two crates use — `ratio` is a parameter here, but the crates
/// hard-code `MIN_ACTIVE_RATIO = 0.2`. This is a MIRROR for asserting the
/// fixture, NOT the tested path: `diarize_offline` runs each crate's OWN copy.
fn surviving_train_pairs(
  seg: &[f64],
  num_chunks: usize,
  num_frames: usize,
  ratio: f64,
) -> std::collections::BTreeSet<(usize, usize)> {
  let min_clean = ratio * num_frames as f64;
  let mut out = std::collections::BTreeSet::new();
  for c in 0..num_chunks {
    let mut single_active = vec![false; num_frames];
    for (f, sa) in single_active.iter_mut().enumerate() {
      let active = (0..NUM_SPEAKERS)
        .filter(|&s| seg[(c * num_frames + f) * NUM_SPEAKERS + s] > 0.0)
        .count();
      *sa = active == 1;
    }
    for s in 0..NUM_SPEAKERS {
      let mut clean = 0.0f64;
      for (f, &sa) in single_active.iter().enumerate() {
        if sa {
          clean += seg[(c * num_frames + f) * NUM_SPEAKERS + s];
        }
      }
      if clean >= min_clean {
        out.insert((c, s));
      }
    }
  }
  out
}

/// The clean-frame tally the filter WOULD compute if the `active_count == 1`
/// gate were removed (naive `sum > 0` over frames): shows the overlap pair only
/// falls below the bar BECAUSE of the single-active gate.
fn naive_active_frames(seg: &[f64], num_frames: usize, c: usize, s: usize) -> f64 {
  (0..num_frames)
    .map(|f| seg[(c * num_frames + f) * NUM_SPEAKERS + s])
    .sum()
}

#[test]
fn offline_cross_crate_equivalence_activity_boundary() {
  let fx = build_boundary_fixture();

  // ── Fixture-geometry pins: both filter branches are genuinely reached ──
  let edge_pairs = [(3usize, 1usize), (4, 1), (5, 1)]; // identity 1, flip band
  let base_pairs = [(0usize, 0usize), (1, 0), (2, 0)]; // identity 0, always retained
  let overlap = (B_OV_CHUNK, 0usize); // identity 2, overlap-contaminated
  let at_02 = surviving_train_pairs(&fx.segmentations, B_NUM_CHUNKS, B_FRAMES, 0.2);
  let at_03 = surviving_train_pairs(&fx.segmentations, B_NUM_CHUNKS, B_FRAMES, 0.3);
  let bar_02 = 0.2 * B_FRAMES as f64;

  // (a) MIN_ACTIVE_RATIO boundary: every edge pair is RETAINED at 0.2 and
  //     DROPPED at 0.3 — the exact re-pin the gate must catch.
  for &e in &edge_pairs {
    assert!(
      at_02.contains(&e),
      "edge pair {e:?} must survive at 0.2 (clean {B_EDGE_CLEAN} >= {bar_02})"
    );
    assert!(
      !at_03.contains(&e),
      "edge pair {e:?} must be dropped at 0.3 (clean {B_EDGE_CLEAN} < {})",
      0.3 * B_FRAMES as f64
    );
  }
  for &b in &base_pairs {
    assert!(
      at_02.contains(&b) && at_03.contains(&b),
      "base pair {b:?} must survive BOTH ratios (it is not the flip)"
    );
  }

  // (b) active_count == 1 overlap rejection: the overlap pair has enough RAW
  //     active frames to clear the bar, but the single-active gate rejects its
  //     overlapping frames, so its clean tally falls below the bar → dropped.
  let raw = naive_active_frames(&fx.segmentations, B_FRAMES, overlap.0, overlap.1);
  assert!(
    raw >= bar_02,
    "overlap pair must clear the bar WITHOUT the gate ({raw} >= {bar_02})"
  );
  assert!(
    !at_02.contains(&overlap),
    "overlap pair must be dropped BY the active_count == 1 gate (clean < {bar_02})"
  );
  let double_active = (0..B_FRAMES).any(|f| {
    (0..NUM_SPEAKERS)
      .filter(|&s| fx.segmentations[(overlap.0 * B_FRAMES + f) * NUM_SPEAKERS + s] > 0.0)
      .count()
      >= 2
  });
  assert!(
    double_active,
    "the overlap chunk must contain genuinely double-active frames for the gate to reject"
  );

  // ── The gate: both crates' WHOLE offline output is bit-identical ──
  let diaric = diaric_outcome(&fx);
  let dia = dia_outcome(&fx);
  assert_eq!(
    diaric, dia,
    "diaric and dia diverged on the activity-boundary fixture (filter branches / AHC / VBx / \
     reassignment / reconstruct / spans) for identical inputs"
  );

  // ── Non-vacuity: the compared output OBSERVABLY depends on the edge cluster ──
  // The edge identity forms its OWN cluster, distinct from the base cluster, and
  // all three edge chunks agree on it. That cluster exists ONLY because the edge
  // pairs survived the 0.2 filter; a 0.3 re-pin drops all of them, the edge
  // cluster vanishes, and this side's hard clusters / grid / spans move — which
  // is exactly what the equivalence assertion above would then report as a
  // divergence. (Non-train pairs are reassigned to the nearest surviving
  // cluster, not left UNMATCHED, so the CLUSTER IDENTITY — not an UNMATCHED
  // sentinel — is the observable that flips.)
  match &diaric {
    Outcome::Ok {
      hard_clusters,
      num_clusters,
      spans,
      ..
    } => {
      let base_cluster = hard_clusters[base_pairs[0].0][base_pairs[0].1];
      let edge_cluster = hard_clusters[edge_pairs[0].0][edge_pairs[0].1];
      assert!(
        base_cluster >= 0 && edge_cluster >= 0,
        "base ({base_cluster}) and edge ({edge_cluster}) active slots must be clustered"
      );
      assert_ne!(
        base_cluster, edge_cluster,
        "edge identity must form its OWN cluster, distinct from base — otherwise a 0.3 \
         re-pin (which drops the whole edge cluster) would not move the compared output"
      );
      for &(c, s) in &edge_pairs {
        assert_eq!(
          hard_clusters[c][s], edge_cluster,
          "all edge chunks must share the edge cluster (chunk {c} slot {s})"
        );
      }
      let distinct: std::collections::BTreeSet<i32> = hard_clusters
        .iter()
        .flatten()
        .copied()
        .filter(|&k| k >= 0)
        .collect();
      assert_eq!(
        distinct.len(),
        2,
        "exactly the base + edge clusters should be assigned (got {distinct:?})"
      );
      assert!(*num_clusters >= 2, "grid must carry both clusters");
      assert!(!spans.is_empty(), "fixture must produce spans");
      println!(
        "[offline-cross-crate boundary] diaric == dia: {num_clusters} clusters, \
         {} distinct speakers, {} spans; edge identity retained at 0.2 as its own \
         cluster ({edge_cluster}) vs base ({base_cluster}), overlap pair rejected \
         by the active_count gate",
        distinct.len(),
        spans.len(),
      );
    }
    Outcome::Err(e) => panic!("expected a clustered result on the boundary fixture, got Err: {e}"),
  }
}

// ── Overlap-add fixture ───────────────────────────────────────────────
// A THIRD scripted input that fences the ONE reconstruction stage the two fixtures
// above never exercise: diaric's overlap-add accumulation. Both crates' reconstruct
// aggregates each chunk's clustered activation into the output grid with
// `aggregated[out_f * num_clusters + k] += clustered[..]` (diaric
// reconstruct/algo.rs:710; dia's mirror). The baseline (10 frames) and boundary (20
// frames) fixtures use the default community-1 chunk window, whose ~1 s step lands
// consecutive chunks ~59 output frames apart — so NO output cell is ever covered by
// two chunks, and a one-sided regression of the `+= v` accumulation to `= v`
// (last-chunk-wins) would leave BOTH of them bit-exact green. Real 589-frame chunks
// overlap ~10 deep and feed DER through exactly this accumulation, so that gap is
// load-bearing.
//
// This fixture makes the chunk step `OV_STEP_FRAMES` OUTPUT FRAMES (< `OV_FRAMES`),
// so consecutive chunks OVERLAP in output-frame space. Identity A (chunks 0,1,
// activation `OV_A_ACT = 0.6`) recurs so the two A chunks CLUSTER TOGETHER and their
// activations are SUMMED where they overlap; identity B (chunks 2,3, activation
// `OV_B_ACT = 1.0`) is the competing cluster. At the A-A-B triple-overlap frame
// (`chunk_start_frame(2)`), `count[t] == 1`, so reconstruct's top-K keeps ONE
// cluster: under the correct `+=`, A's SUMMED activation `0.6 + 0.6 = 1.2` beats B's
// single `1.0`, so A is selected; under a `= v` regression A collapses to its single
// `0.6 < 1.0` and B would win instead — flipping that cell's grid/spans and breaking
// the bit-exact `diaric == dia` equality. The pinned crates cannot be mutated to show
// the red inside one build, so this test asserts (1) `diaric == dia` bit-exact,
// (2) the overlap is STRUCTURALLY real (some output frame covered by >= 2 chunks,
// derived from the window geometry), and (3) the summed vote's outcome directly
// (A active, B inactive at the overlap cell) — the observable a `+= → =` mutation
// would flip.
const OV_NUM_CHUNKS: usize = 4;
const OV_FRAMES: usize = 10;
const OV_STEP_FRAMES: usize = 4; // chunk step in OUTPUT FRAMES; < OV_FRAMES ⇒ overlap
const OV_A_ID: usize = 0; // identity A: chunks 0,1 (cluster together, overlap-summed)
const OV_B_ID: usize = 1; // identity B: chunks 2,3 (the competing cluster)
const OV_A_ACT: f64 = 0.6; // A's per-chunk activation: single 0.6 < B, but 0.6+0.6 > B
const OV_B_ACT: f64 = 1.0; // B's per-chunk activation

/// Chunk sliding window whose STEP is `OV_STEP_FRAMES` output frames (not the ~1 s
/// the baseline uses), so consecutive chunks overlap by `OV_FRAMES - OV_STEP_FRAMES`
/// output frames. Duration spans exactly `OV_FRAMES` frames so
/// `count_from_segmentations` derives a tight, reconstruct-valid `num_output_frames`
/// (reconstruct ignores chunk duration — only start/step place chunks).
fn overlap_chunks_sw() -> SlidingWindow {
  let fs = frame_sliding_window();
  SlidingWindow::new(
    0.0,
    (OV_FRAMES as f64 - 1.0) * fs.step(),
    OV_STEP_FRAMES as f64 * fs.step(),
  )
}

/// The output frame chunk `c` starts at, computed the SAME way reconstruct and
/// `count_from_segmentations` place chunks: `round(chunk_start / frame_step)` (the
/// `center_offset` cancels `frames_sw.duration / 2` in `closest_frame`).
fn chunk_start_frame(c: usize, cs: SlidingWindow, fs: SlidingWindow) -> usize {
  ((c as f64 * cs.step()) / fs.step()).round_ties_even() as usize
}

/// Per-output-frame covering-chunk count, from the same window geometry
/// reconstruct's overlap-add uses — the structural proof that the fixture reaches
/// the multiply-covered regime (rather than a hard-coded magic number).
fn coverage(
  num_chunks: usize,
  num_frames: usize,
  num_output_frames: usize,
  cs: SlidingWindow,
  fs: SlidingWindow,
) -> Vec<usize> {
  let mut cov = vec![0usize; num_output_frames];
  for c in 0..num_chunks {
    let sf = chunk_start_frame(c, cs, fs);
    for f in 0..num_frames {
      let t = sf + f;
      if t < num_output_frames {
        cov[t] += 1;
      }
    }
  }
  cov
}

/// Build the overlap-add fixture (geometry documented in the const block above).
/// Reuses the shared `count_from_segmentations` derivation, so the tensor set is
/// reconstruct-valid despite the deliberate output-frame overlap.
fn build_overlap_fixture() -> OfflineFixture {
  let mut raw_embeddings = vec![0.0f32; OV_NUM_CHUNKS * NUM_SPEAKERS * EMBED_DIM];
  let mut segmentations = vec![0.0f64; OV_NUM_CHUNKS * OV_FRAMES * NUM_SPEAKERS];

  // Slot 0 is single-active every frame (so every frame is clean); chunks 0,1 carry
  // identity A at 0.6, chunks 2,3 carry identity B at 1.0.
  let plan = [
    (OV_A_ID, OV_A_ACT),
    (OV_A_ID, OV_A_ACT),
    (OV_B_ID, OV_B_ACT),
    (OV_B_ID, OV_B_ACT),
  ];
  for (c, &(identity, act)) in plan.iter().enumerate() {
    for f in 0..OV_FRAMES {
      segmentations[(c * OV_FRAMES + f) * NUM_SPEAKERS] = act; // slot 0
    }
    let base = (c * NUM_SPEAKERS) * EMBED_DIM;
    raw_embeddings[base..base + EMBED_DIM].copy_from_slice(&prototype(identity));
  }

  let cs = overlap_chunks_sw();
  let fs = frame_sliding_window();
  let count = speakerkit::window::count_from_segmentations(
    &segmentations,
    OV_NUM_CHUNKS,
    OV_FRAMES,
    NUM_SPEAKERS,
    ONSET,
    cs,
    fs,
  );
  let num_output_frames = count.len();
  OfflineFixture {
    raw_embeddings,
    segmentations,
    count,
    num_output_frames,
    num_chunks: OV_NUM_CHUNKS,
    num_frames: OV_FRAMES,
  }
}

/// Fences diaric's reconstruction OVERLAP-ADD (`aggregated[idx] += v`,
/// reconstruct/algo.rs:710). See the const block above for why the two existing
/// fixtures cannot: their chunks never share an output cell, so a `+= → =`
/// (last-chunk-wins) regression stays bit-exact green on both. Here chunks overlap
/// in output-frame space and cluster A's activation is summed across two overlapping
/// chunks, so that mutation would flip the overlap cell's top-K selection and break
/// the `diaric == dia` equality this test asserts.
#[test]
fn offline_cross_crate_equivalence_overlap_add() {
  let fx = build_overlap_fixture();
  let cs = overlap_chunks_sw();
  let fs = frame_sliding_window();

  // ── (2) Non-vacuity for overlap: the fixture genuinely multiply-covers output
  //        frames (derived from the window geometry, not hard-coded). ──
  let cov = coverage(OV_NUM_CHUNKS, OV_FRAMES, fx.num_output_frames, cs, fs);
  let multi = cov.iter().filter(|&&n| n >= 2).count();
  let max_cov = cov.iter().copied().max().unwrap_or(0);
  assert!(
    max_cov >= 2,
    "overlap fixture must cover at least one output frame with >= 2 chunks (the \
     overlap-add regime); got max coverage {max_cov}"
  );
  println!(
    "[offline-cross-crate overlap] {multi} of {} output frames are covered by >= 2 chunks \
     (max coverage {max_cov})",
    fx.num_output_frames
  );

  // ── (1) The gate: both crates' WHOLE offline output is bit-identical ──
  let diaric = diaric_outcome_cs(&fx, cs);
  let dia = dia_outcome_cs(&fx, cs);
  assert_eq!(
    diaric, dia,
    "diaric and dia diverged on the overlap-add fixture (reconstruct accumulation / \
     top-K / spans) for identical inputs"
  );

  // ── (3) Non-vacuity + the overlap-add outcome itself ──
  match &diaric {
    Outcome::Ok {
      hard_clusters,
      num_clusters,
      grid_bits,
      spans,
    } => {
      let distinct: std::collections::BTreeSet<i32> = hard_clusters
        .iter()
        .flatten()
        .copied()
        .filter(|&k| k >= 0)
        .collect();
      assert!(
        *num_clusters >= 2,
        "overlap fixture must reach multi-cluster reconstruction (got {num_clusters})"
      );
      assert!(
        distinct.len() >= 2,
        "overlap fixture must assign >= 2 distinct speakers (got {distinct:?})"
      );
      assert!(!spans.is_empty(), "overlap fixture must produce spans");

      // The summed-vote outcome: A (chunks 0,1) and B (chunks 2,3) form distinct
      // clusters, and at the A-A-B triple-overlap frame A's SUMMED activation
      // (0.6 + 0.6 = 1.2) beats B's single (1.0) under the correct `+=`, so A is the
      // ONE cluster the top-K keeps (count[t] == 1). A `+= → =` regression collapses
      // A to 0.6 < 1.0 and would select B instead — this is the observable it flips.
      let a_cluster = hard_clusters[0][0];
      let b_cluster = hard_clusters[2][0];
      assert!(
        a_cluster >= 0 && b_cluster >= 0 && a_cluster != b_cluster,
        "identities A ({a_cluster}) and B ({b_cluster}) must form two distinct clusters"
      );
      let t = chunk_start_frame(2, cs, fs); // first frame of chunk 2 = the A-A-B triple cell
      assert!(
        cov[t] >= 2,
        "the checked overlap frame {t} must be multiply covered (got {})",
        cov[t]
      );
      let nc = *num_clusters;
      let cell = |k: i32| grid_bits[t * nc + k as usize];
      assert_eq!(
        cell(a_cluster),
        1.0f32.to_bits(),
        "overlap-add: cluster A must be ACTIVE at the overlap cell (summed 1.2 > 1.0)"
      );
      assert_eq!(
        cell(b_cluster),
        0.0f32.to_bits(),
        "overlap-add: cluster B must be INACTIVE at the overlap cell (last-wins would flip this)"
      );
      println!(
        "[offline-cross-crate overlap] diaric == dia: {num_clusters} clusters, \
         {} distinct speakers, {} spans; overlap frame {t} selects summed-vote cluster A \
         ({a_cluster}) over B ({b_cluster})",
        distinct.len(),
        spans.len(),
      );
    }
    Outcome::Err(e) => panic!("expected a clustered result on the overlap fixture, got Err: {e}"),
  }
}
