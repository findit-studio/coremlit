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
//! This gate turns that assumption into an EXECUTABLE invariant. It builds ONE
//! set of typed offline inputs (scripted raw embeddings + segmentations rich
//! enough that the community-1 filter keeps many training pairs, AHC merges exact
//! ties, and VBx actually iterates), feeds it to BOTH `diarize_offline`
//! implementations, and asserts their WHOLE observable output is identical:
//! typed outcome, per-chunk hard clusters, the discrete diarization grid
//! (bit-exact `f32`), the cluster count, and the RTTM spans (bit-exact `f64`
//! start/duration + cluster). A future re-pin of either crate that changes an
//! AHC tie, `MIN_ACTIVE_RATIO`, a VBx step, the reassignment, or reconstruction
//! — while leaving PLDA untouched — moves one side and fails here.
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

use speakerkit::window::{WindowOptions, chunk_sliding_window, frame_sliding_window};

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
  let plda = diaric::plda::PldaTransform::new().expect("diaric hermetic PLDA");
  let w = WindowOptions::new();
  let input = diaric::offline::OfflineInput::new(
    &fx.raw_embeddings,
    NUM_CHUNKS,
    NUM_SPEAKERS,
    &fx.segmentations,
    FRAMES,
    &fx.count,
    fx.num_output_frames,
    chunk_sliding_window(&w).into(),
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
  let plda = dia::plda::PldaTransform::new().expect("dia hermetic PLDA");
  let w = WindowOptions::new();
  let cs = chunk_sliding_window(&w);
  let fs = frame_sliding_window();
  let input = dia::offline::OfflineInput::new(
    &fx.raw_embeddings,
    NUM_CHUNKS,
    NUM_SPEAKERS,
    &fx.segmentations,
    FRAMES,
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
