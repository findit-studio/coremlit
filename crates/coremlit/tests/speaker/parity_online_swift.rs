//! Online-clusterer **parity with FluidAudio's Swift `SpeakerManager`**, driven
//! through speakerkit's [`OnlineOptions`] wiring.
//!
//! This gates the CLUSTERER, not the segmentation/embedding models: it feeds a
//! deterministic SYNTHETIC embedding sequence (a documented 64-bit LCG, NOT
//! model output) to both sides and asserts per-step agreement. So — unlike
//! `parity_argmax_swift.rs`, which needs the gitignored `Models/` artifacts and
//! is therefore `#[ignore]`d — this suite is fully **hermetic**: it regenerates
//! the sequence in Rust and compares against a COMMITTED Swift trace
//! (`tests/speaker/fixtures/golden_online_swift/trace.json`), so it runs by default in
//! `cargo test` with no Swift toolchain and no models. That is also what lets
//! the campaign's order-dependence mutation ("perturb the feed order → this
//! fails") be observed in the ordinary suite.
//!
//! # The Swift oracle
//! `tests/speaker/swift/online_oracle/` is a COMMITTED SwiftPM executable that
//! `import`s the local FluidAudio checkout, drives `SpeakerManager` DIRECTLY
//! with the same LCG sequence, and dumps per-step decisions (assigned id / new
//! / dropped + the updated centroid) as JSON. Regenerate the committed trace
//! from that directory with
//! `swift run online_oracle > ../../fixtures/golden_online_swift/trace.json` (its
//! README carries the FluidAudio path-dep note and the `FLUIDAUDIO_SRC`
//! override).
//!
//! The oracle emits the Swift-attested `generator` block (constants incl.
//! `durationBase`/`durationSpan`) AND both input hashes, computed in Swift with
//! an FNV-1a-64 byte-identical to `common::fnv1a_f32`: `inputFnv1a` over the raw
//! embeddings and `durationsFnv1a` over the `[f32]` duration sequence. Both are
//! Swift attestations of what the oracle actually fed `SpeakerManager` — NOT
//! Rust self-hashes — so this harness proves cross-language input identity for
//! BOTH clusterer inputs (finding M3).
//!
//! # The inputs are PROVEN identical, not assumed (the alignkit lesson)
//! Before any decision is compared, BOTH clusterer inputs are proven identical to
//! the Swift dumper's:
//! - the Rust-regenerated **embedding** sequence is FNV-1a-64 hashed (byte-
//!   identical to the Swift hash of the vectors it fed `SpeakerManager`) and
//!   asserted equal to the trace's recorded `inputFnv1a`;
//! - the **duration** sequence — the SECOND input to `assign`, which gates
//!   New-vs-Dropped — is proven two ways: the trace's Swift-emitted `generator`
//!   block (`durationBase`/`durationSpan`/`seed`/…) is asserted against the Rust
//!   generator constants, and the regenerated durations are FNV-1a-64 hashed
//!   against the trace's `durationsFnv1a`, which the oracle now computes IN
//!   SWIFT (the same byte-identical FNV as `inputFnv1a`) over the very durations
//!   it fed `SpeakerManager` — a genuine cross-language attestation, not a Rust
//!   self-hash. The embedding hash alone does NOT cover durations:
//!   `DUR_BASE`/`DUR_SPAN` are affine constants applied to the LCG draw's
//!   OUTPUT, so a change shifts every duration while leaving the LCG progression
//!   (and thus the embedding hash) untouched — the hole that let a fully-broken
//!   duration bridge still pass 48/48.
//!
//! Any mismatch fails as a HARNESS bug (a drifted LCG or duration constant),
//! never as a clusterer finding.
//!
//! # Tie-freeness (the T4 caveat)
//! Swift's nearest-match iterates a `Dictionary` in nondeterministic order and
//! breaks an exact distance tie arbitrarily; dia pins tie → lowest id. The
//! synthetic sequence is built from 8 ORTHOGONAL one-hot prototype blocks with
//! tiny noise, so every embedding is either ~0.002 from exactly one existing
//! centroid or ~1.0 from all of them. This suite asserts a large
//! nearest/second-nearest gap on every assignment-to-existing step, so the two
//! sides can only ever select the SAME nearest speaker — the oracle compares
//! DEFINED behaviour, not a coin flip on a tie.
//!
//! # What is asserted
//! - **Assignment kind + id: EXACT.** Discrete outcomes, robust to float noise
//!   given the gap above; a divergence is a real semantics bug.
//! - **Centroid: within [`CENTROID_TOL`].** The one place a tolerance is
//!   justified: Swift's `vDSP` SIMD accumulation and dia's scalar loop sum the
//!   composite mean/EMA update in different orders, so the running centroids
//!   differ by float rounding — bounded, measured, never a decision flip.

mod common;

use coremlit::audio::speaker::OnlineOptions;
use diaric::{
  cluster::online::{Assignment, OnlineClusterer},
  embed::Embedding,
};

// ── Generator constants (mirror tests/speaker/swift/online_oracle's main.swift) ──
const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const STEPS: usize = 48;
const PROTOTYPES: usize = 8;
const BLOCK: usize = 32;
const DIM: usize = 256; // == diaric::embed::EMBEDDING_DIM
const NOISE_SCALE: f32 = 0.02;
const DUR_BASE: f32 = 0.3;
const DUR_SPAN: f32 = 2.0;

/// Minimum nearest/second-nearest cosine-distance gap required whenever an
/// assignment-to-existing happens, so Swift's dict-order tie-break and dia's
/// lowest-id pin cannot diverge. The orthogonal-block design keeps the real gap
/// near `0.98`; `0.1` is a wide safety margin.
const TIE_GAP: f32 = 0.1;

/// Max per-element absolute centroid difference tolerated between dia's running
/// centroid and Swift's, per non-drop step. Justified by vDSP-vs-scalar
/// accumulation order only (see the module doc). **Measured worst: `5.96e-8`**
/// (single-ulp level — dia's scalar composite update tracks Swift's `vDSP` to
/// the last bit); bounded at `1e-4` (~1700×) so a real update-math divergence
/// (a wrong EMA α, a dropped recalc — order `1e-2`+) fails loudly while the
/// frozen-vs-deterministic float rounding never does. The worst is printed each
/// run so any drift toward the bound is visible.
const CENTROID_TOL: f64 = 1e-4;

/// The 64-bit LCG (Knuth MMIX constants). Byte-identical to the Swift `Lcg`.
struct Lcg {
  state: u64,
}
impl Lcg {
  fn new(seed: u64) -> Self {
    Self { state: seed }
  }
  fn next_u64(&mut self) -> u64 {
    self.state = self
      .state
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    self.state
  }
  /// f32 in `[0, 1)` from the top 24 bits, scaled by an exact `2^-24`.
  fn next_unit(&mut self) -> f32 {
    let bits = (self.next_u64() >> 40) as u32;
    bits as f32 * (1.0 / 16777216.0)
  }
  /// A prototype index in `0..count` from the high bits (see the Swift note on
  /// why an LCG's low bits are unusable here). `next_unit() * count` is exact
  /// and truncates identically in both languages.
  fn next_index(&mut self, count: usize) -> usize {
    ((self.next_unit() * count as f32) as usize).min(count - 1)
  }
}

/// Regenerate the `(raw embedding, speech duration)` sequence. Byte-identical to
/// the Swift dumper's generation (same LCG, same per-step draw order: 1 index
/// draw, 1 duration unit, then `DIM` noise units).
fn synthetic_sequence() -> Vec<([f32; DIM], f32)> {
  let mut lcg = Lcg::new(SEED);
  let mut seq = Vec::with_capacity(STEPS);
  for _ in 0..STEPS {
    let j = lcg.next_index(PROTOTYPES);
    let duration = DUR_BASE + lcg.next_unit() * DUR_SPAN;
    let mut raw = [0.0f32; DIM];
    for (d, slot) in raw.iter_mut().enumerate() {
      let base = if d >= j * BLOCK && d < (j + 1) * BLOCK {
        1.0
      } else {
        0.0
      };
      let noise = (lcg.next_unit() * 2.0 - 1.0) * NOISE_SCALE;
      *slot = base + noise;
    }
    seq.push((raw, duration));
  }
  seq
}

fn dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
  a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One recorded Swift decision.
struct SwiftStep {
  kind: String,
  id: Option<i64>,
  centroid: Option<Vec<f32>>,
}

/// The committed trace's `generator` block: the EXACT constants the Swift oracle
/// drew its `(embedding, duration)` sequence from — Swift-emitted provenance.
/// Asserting the Rust generator constants against these proves the two sides
/// generated identical inputs, including the affine duration constants
/// (`durationBase`/`durationSpan`) the embedding hash cannot observe.
struct TraceGenerator {
  seed: u64,
  steps: usize,
  prototypes: usize,
  block: usize,
  dim: usize,
  noise_scale: f32,
  duration_base: f32,
  duration_span: f32,
}

/// The committed Swift trace.
struct SwiftTrace {
  input_fnv1a: String,
  /// FNV-1a-64 (hex) of the exact `[f32]` speech-duration sequence — the second
  /// clusterer input. Swift-EMITTED by the oracle (the same byte-identical FNV
  /// that reproduces `input_fnv1a`, hashed over the durations it fed
  /// `SpeakerManager`), so hashing the Rust-regenerated durations against it is
  /// a genuine cross-language check, not a Rust self-hash (findings M2a + M3).
  durations_fnv1a: String,
  generator: TraceGenerator,
  speaker_threshold: f32,
  embedding_threshold: f32,
  min_speech_duration: f32,
  new_count: usize,
  existing_count: usize,
  dropped_count: usize,
  steps: Vec<SwiftStep>,
}

fn load_trace() -> SwiftTrace {
  let path = common::fixtures_dir()
    .join("golden_online_swift")
    .join("trace.json");
  let bytes = std::fs::read(&path).unwrap_or_else(|e| {
    panic!(
      "read online-swift trace {}: {e}\n  regenerate: swift run online_oracle > {}",
      path.display(),
      path.display()
    )
  });
  let v: serde_json::Value = serde_json::from_slice(&bytes).expect("parse trace json");
  let f32_at = |val: &serde_json::Value| -> f32 { val.as_f64().expect("f64") as f32 };
  let gen_v = &v["generator"];
  let seed_hex = gen_v["seed"].as_str().expect("generator.seed hex string");
  let generator = TraceGenerator {
    seed: u64::from_str_radix(seed_hex.trim_start_matches("0x"), 16).expect("parse generator.seed"),
    steps: gen_v["steps"].as_u64().expect("generator.steps") as usize,
    prototypes: gen_v["prototypes"].as_u64().expect("generator.prototypes") as usize,
    block: gen_v["blockSize"].as_u64().expect("generator.blockSize") as usize,
    dim: gen_v["embeddingDim"]
      .as_u64()
      .expect("generator.embeddingDim") as usize,
    noise_scale: f32_at(&gen_v["noiseScale"]),
    duration_base: f32_at(&gen_v["durationBase"]),
    duration_span: f32_at(&gen_v["durationSpan"]),
  };
  let opts = &v["options"];
  let steps = v["steps"]
    .as_array()
    .expect("steps array")
    .iter()
    .map(|s| SwiftStep {
      kind: s["kind"].as_str().expect("kind").to_string(),
      id: s["id"].as_i64(),
      centroid: s["centroid"].as_array().map(|a| {
        a.iter()
          .map(|x| x.as_f64().expect("centroid f64") as f32)
          .collect()
      }),
    })
    .collect();
  SwiftTrace {
    input_fnv1a: v["inputFnv1a"].as_str().expect("inputFnv1a").to_string(),
    durations_fnv1a: v["durationsFnv1a"]
      .as_str()
      .expect("durationsFnv1a")
      .to_string(),
    generator,
    speaker_threshold: f32_at(&opts["speakerThreshold"]),
    embedding_threshold: f32_at(&opts["embeddingThreshold"]),
    min_speech_duration: f32_at(&opts["minSpeechDuration"]),
    new_count: v["newCount"].as_u64().expect("newCount") as usize,
    existing_count: v["existingCount"].as_u64().expect("existingCount") as usize,
    dropped_count: v["droppedCount"].as_u64().expect("droppedCount") as usize,
    steps,
  }
}

#[test]
fn online_clusterer_matches_fluidaudio_swift_trace() {
  assert_eq!(DIM, diaric::embed::EMBEDDING_DIM, "DIM must equal dia's");
  let trace = load_trace();
  let seq = synthetic_sequence();
  assert_eq!(
    seq.len(),
    trace.steps.len(),
    "step count differs from the trace"
  );

  // ── Input-match proof, BEFORE any decision is read (the alignkit lesson) ──
  let flat: Vec<f32> = seq.iter().flat_map(|(r, _)| r.iter().copied()).collect();
  assert_eq!(
    common::fnv_hex(common::fnv1a_f32(&flat)),
    trace.input_fnv1a,
    "the regenerated embedding sequence hashes differently from the Swift dumper's — the two \
     sides are NOT driving the clusterer with identical inputs; fix the LCG/generator before \
     reading any parity number"
  );

  // ── Duration input-match proof (codex M2a), also BEFORE any decision ──
  // Durations are the SECOND input to `assign` and gate New-vs-Dropped. The
  // embedding hash above does NOT cover them: DUR_BASE/DUR_SPAN are affine
  // constants applied to the LCG draw's OUTPUT, so changing DUR_BASE shifts every
  // duration while leaving the LCG progression (and the embedding hash) untouched
  // — exactly why a broken production duration bridge could still pass 48/48.
  // Two independent proofs close it:
  //   (1) the committed `generator` block is Swift-emitted provenance; assert
  //       every Rust generator constant against it, so a DUR_BASE/DUR_SPAN (or
  //       seed/steps/…) drift on the Rust side fails against the oracle's own
  //       recorded values;
  //   (2) hash the regenerated duration values against the trace's
  //       `durationsFnv1a` — which the Swift oracle emits with the same
  //       byte-identical FNV over the durations it actually fed `SpeakerManager`
  //       — mirroring the embedding hash. A true cross-language check (not a
  //       Rust self-hash): it ALSO catches a Swift-side change to the duration
  //       value/FORMULA that leaves the LCG stream and the constants intact.
  let g = &trace.generator;
  assert_eq!(
    SEED, g.seed,
    "SEED differs from the committed generator.seed"
  );
  assert_eq!(
    STEPS, g.steps,
    "STEPS differs from the committed generator.steps"
  );
  assert_eq!(
    PROTOTYPES, g.prototypes,
    "PROTOTYPES differs from generator.prototypes"
  );
  assert_eq!(BLOCK, g.block, "BLOCK differs from generator.blockSize");
  assert_eq!(DIM, g.dim, "DIM differs from generator.embeddingDim");
  assert_eq!(
    NOISE_SCALE, g.noise_scale,
    "NOISE_SCALE differs from generator.noiseScale"
  );
  assert_eq!(
    DUR_BASE, g.duration_base,
    "DUR_BASE differs from the committed generator.durationBase"
  );
  assert_eq!(
    DUR_SPAN, g.duration_span,
    "DUR_SPAN differs from the committed generator.durationSpan"
  );

  let durations: Vec<f32> = seq.iter().map(|(_, d)| *d).collect();
  assert_eq!(
    common::fnv_hex(common::fnv1a_f32(&durations)),
    trace.durations_fnv1a,
    "the regenerated speech-duration sequence hashes differently from the committed trace — the \
     two sides are NOT feeding `assign` identical durations (a drifted DUR_BASE/DUR_SPAN or the \
     duration formula); fix the generator before reading any parity number"
  );

  // ── The wiring under test: speakerkit's OnlineOptions::default() → dia. The
  // trace MUST have been generated at exactly these thresholds. ──
  let opts = OnlineOptions::default();
  assert_eq!(opts.speaker_threshold(), trace.speaker_threshold);
  assert_eq!(opts.embedding_threshold(), trace.embedding_threshold);
  assert_eq!(opts.min_speech_duration(), trace.min_speech_duration);

  let mut clusterer = OnlineClusterer::try_new(opts.to_dia_options())
    .expect("OnlineOptions map to a validated clusterer");
  let speaker_threshold = opts.speaker_threshold();
  let mut worst_centroid_diff = 0.0f64;
  let (mut n_new, mut n_existing, mut n_dropped) = (0usize, 0usize, 0usize);

  for (i, ((raw, duration), swift)) in seq.iter().zip(&trace.steps).enumerate() {
    let embedding = Embedding::normalize_from(*raw).expect("synthetic embedding is nonzero");

    // Tie-freeness guard: if the nearest existing centroid is within
    // speaker_threshold (i.e. an assignment-to-existing is about to happen),
    // the nearest and second-nearest must be well separated so Swift's dict
    // iteration order and dia's ascending-id scan select the SAME speaker.
    let mut dists: Vec<f32> = clusterer
      .speaker_ids()
      .map(|id| 1.0 - dot(embedding.as_array(), clusterer.centroid(id).unwrap()))
      .collect();
    dists.sort_by(|a, b| a.total_cmp(b));
    if let Some(&nearest) = dists.first()
      && nearest < speaker_threshold
    {
      let second = dists.get(1).copied().unwrap_or(f32::INFINITY);
      assert!(
        second - nearest > TIE_GAP,
        "step {i}: nearest/second gap {} <= {TIE_GAP} — the sequence has an ambiguous tie the \
         oracle cannot arbitrate (Swift's dict order vs dia's lowest-id pin could differ)",
        second - nearest
      );
    }

    let assignment = clusterer.assign(&embedding, *duration);

    // Assignment kind + id: EXACT.
    match (assignment, swift.kind.as_str()) {
      (Assignment::New(id), "new") => {
        n_new += 1;
        assert_eq!(
          Some(id as i64),
          swift.id,
          "step {i}: new speaker id differs"
        );
      }
      (Assignment::Existing(id), "existing") => {
        n_existing += 1;
        assert_eq!(
          Some(id as i64),
          swift.id,
          "step {i}: existing speaker id differs"
        );
      }
      (Assignment::Dropped, "dropped") => {
        n_dropped += 1;
        assert!(
          swift.id.is_none(),
          "step {i}: dia dropped, swift kept an id"
        );
      }
      (a, k) => panic!("step {i}: dia produced {a:?} but Swift recorded \"{k}\""),
    }

    // Centroid: within tolerance for the two non-drop kinds.
    if let Some(id) = assignment.speaker_id() {
      let dia_centroid = clusterer
        .centroid(id)
        .expect("assigned speaker has a centroid");
      let swift_centroid = swift
        .centroid
        .as_ref()
        .unwrap_or_else(|| panic!("step {i}: Swift has no centroid for a non-drop step"));
      assert_eq!(swift_centroid.len(), DIM, "step {i}: centroid dim");
      let diff = common::max_abs_diff(dia_centroid, swift_centroid);
      worst_centroid_diff = worst_centroid_diff.max(diff);
      assert!(
        diff <= CENTROID_TOL,
        "step {i} (speaker {id}): centroid max|diff| {diff:.3e} exceeds {CENTROID_TOL:e}. This is \
         the composite mean/EMA update diverging beyond vDSP-vs-scalar rounding — a FINDING, not a \
         bound to raise."
      );
    }
  }

  // The recorded distribution is itself part of the fixture's value: it proves
  // the sequence exercises all three outcomes (not a trivially-all-existing run).
  assert_eq!(n_new, trace.new_count, "new count");
  assert_eq!(n_existing, trace.existing_count, "existing count");
  assert_eq!(n_dropped, trace.dropped_count, "dropped count");
  assert!(
    n_new >= 2 && n_existing >= 2 && n_dropped >= 1,
    "the fixture must exercise New, Existing and Dropped (got new={n_new} existing={n_existing} \
     dropped={n_dropped})"
  );

  println!(
    "[online-oracle] {} steps matched Swift SpeakerManager exactly (new={n_new} existing={n_existing} \
     dropped={n_dropped}); worst centroid |diff| = {worst_centroid_diff:.3e} (tol {CENTROID_TOL:e})",
    seq.len()
  );
}

#[test]
fn online_oracle_sequence_is_deterministic() {
  // The engine, driven through the same wiring on the same sequence twice, is
  // total-deterministic — the defining property the order-dependence relies on.
  let seq = synthetic_sequence();
  let run = || -> Vec<Assignment> {
    let mut c = OnlineClusterer::try_new(OnlineOptions::default().to_dia_options())
      .expect("default OnlineOptions are valid");
    seq
      .iter()
      .map(|(raw, d)| c.assign(&Embedding::normalize_from(*raw).unwrap(), *d))
      .collect()
  };
  assert_eq!(
    run(),
    run(),
    "the same sequence must assign identically twice"
  );
}
