//! **Cross-crate PLDA equivalence — the executable gate under the dual-PLDA DER
//! oracle.**
//!
//! After the `diaric` extraction the DER oracle carries TWO separately-typed
//! community-1 PLDA projections: dia's (`diarization`) on the REFERENCE (dia-ort)
//! side and diaric's on the MEASURED (speakerkit) side. The old oracle shared
//! ONE `PldaTransform` instance, so "the two clustering runs cannot diverge on
//! the projection" was true by construction; with the split it is an assumption.
//! This gate turns that assumption into an EXECUTABLE, hermetic (no shipping
//! `Models/`) invariant: it loads BOTH crates' in-crate PLDA — each ships the
//! weights via `include_bytes!`, so `PldaTransform::new()` needs no external
//! model — transforms a fixed fixture set of raw WeSpeaker vectors through each,
//! and asserts the outputs are BIT-EXACT (`f64::to_bits`, so a sign bit or a
//! NaN payload would fail).
//!
//! What it proves, and the static grounding it hardens:
//!
//! * `dia::plda` and `diaric::plda` are the SAME code — this crate re-confirmed
//!   it at the byte level: `plda/{mod,transform,error}.rs` are identical and
//!   all eight `models/plda/*.bin` weight blobs share a sha256 (`loader.rs`
//!   differs only in one doc word). The diaric import fence had already checked
//!   every moved path blob-identical to its `diarization` source by git blob id.
//! * dia `b6a6f9a` → `d75b8f9` (diarization #19) was additive-only, in
//!   `cluster/online`; it did not touch the offline PLDA surface, so both pins
//!   expose identical `PldaTransform` construction and `xvec_transform` /
//!   `plda_transform` math.
//!
//! Those are facts about the *current* pins. This gate re-derives them at runtime
//! on every build and fails loudly — a real finding, not a flake — if a future
//! re-pin of either crate ever moves the projection.
//!
//! Hermetic + ORDINARY (never `#[ignore]`d): it depends only on the
//! compile-time-embedded weights, so it runs in the `dia-oracle` ordinary suite
//! with no `Models/` tree and no fixtures.
#![cfg(feature = "speaker-oracle")]

/// Number of raw fixture vectors projected through both crates.
const VECTORS: usize = 64;

/// `splitmix64` — a tiny deterministic PRNG. It makes the fixture vectors
/// identical across runs AND identical between the two crates (same bytes in ⇒
/// same `f32` ⇒ any output divergence is the PLDA projection, never the input).
fn splitmix64(state: &mut u64) -> u64 {
  *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
  let mut z = *state;
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^ (z >> 31)
}

/// A deterministic raw WeSpeaker-shaped embedding: 256 `f32` in `[-0.4, 0.4)`.
/// The resulting L2 norm (~2-4) sits inside the captured raw distribution
/// `[0.536, 6.97]` and clears both `RawEmbedding`'s norm floor (0.01) and
/// `xvec_transform`'s centered-norm guard (0.1), so `project` returns `Ok` on
/// both crates and the comparison is over real transform output, not a rejection.
fn synth_raw(seed: u64) -> [f32; 256] {
  let mut state = seed;
  let mut arr = [0.0f32; 256];
  for slot in &mut arr {
    // Top 24 bits → an exact `f32` integer in `[0, 2^24)`, / 2^24 → `[0, 1)`.
    let unit = (splitmix64(&mut state) >> 40) as f32 / 16_777_216.0_f32;
    *slot = (unit - 0.5) * 0.8;
  }
  arr
}

/// Assert dia's and diaric's community-1 PLDA are bit-identical on the exposed
/// eigen artifact (`phi`) and on `project` (`xvec_transform` → `plda_transform`)
/// over the fixture set — the runtime form of "same code + same blobs ⇒ same
/// numbers".
#[test]
fn plda_cross_crate_equivalence() {
  // The dimensional constants must agree (and be the documented 256→128).
  assert_eq!(
    dia::plda::EMBEDDING_DIMENSION,
    diaric::plda::EMBEDDING_DIMENSION,
    "EMBEDDING_DIMENSION differs across crates"
  );
  assert_eq!(
    dia::plda::PLDA_DIMENSION,
    diaric::plda::PLDA_DIMENSION,
    "PLDA_DIMENSION differs across crates"
  );
  assert_eq!(dia::plda::EMBEDDING_DIMENSION, 256);
  assert_eq!(dia::plda::PLDA_DIMENSION, 128);

  let dia_pt = dia::plda::PldaTransform::new().expect("dia community-1 PldaTransform");
  let dc_pt = diaric::plda::PldaTransform::new().expect("diaric community-1 PldaTransform");

  // Count every scalar bit-equality asserted, so the gate is non-vacuous and its
  // coverage is itself pinned (128 phi eigenvalues + VECTORS × 128 projection).
  let mut checks: u64 = 0;

  // ── The exposed eigen artifact: phi, the descending eigenvalue diagonal VBx
  //    consumes as the across-class covariance. ──
  let dia_phi = dia_pt.phi();
  let dc_phi = dc_pt.phi();
  assert_eq!(
    dia_phi.len(),
    dc_phi.len(),
    "phi length differs: dia {} vs diaric {}",
    dia_phi.len(),
    dc_phi.len()
  );
  assert_eq!(dia_phi.len(), 128, "phi is the 128-d eigenvalue diagonal");
  for (i, (a, b)) in dia_phi.iter().zip(dc_phi.iter()).enumerate() {
    assert_eq!(
      a.to_bits(),
      b.to_bits(),
      "phi[{i}] diverges: dia {a} vs diaric {b}"
    );
    checks += 1;
  }

  // ── The transform output: project() over a fixed fixture set. project's
  //    [f64; 128] is the externally-observable transform result
  //    (PostXvecEmbedding's accessor is test-only pub(crate)), and it exercises
  //    every stored factor — mean1/mean2/lda/sqrt-dims in xvec_transform, then
  //    plda_mu and the pinned descending eigenvector matrix in plda_transform. ──
  for v in 0..VECTORS {
    let arr = synth_raw(0x5EED_0000_u64.wrapping_add(v as u64));

    // Both crates must agree on admissibility, then on the projection itself.
    let (dia_raw, dc_raw) = match (
      dia::plda::RawEmbedding::from_wespeaker(arr),
      diaric::plda::RawEmbedding::from_wespeaker(arr),
    ) {
      (Ok(dr), Ok(cr)) => (dr, cr),
      (a, b) => {
        panic!("vector {v}: RawEmbedding::from_wespeaker diverged: dia {a:?} vs diaric {b:?}")
      }
    };
    match (dia_pt.project(&dia_raw), dc_pt.project(&dc_raw)) {
      (Ok(a), Ok(b)) => {
        assert_eq!(a.len(), 128);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
          assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "vector {v} projection[{i}] diverges: dia {x} vs diaric {y}"
          );
          checks += 1;
        }
      }
      (a, b) => panic!("vector {v}: project() diverged: dia {a:?} vs diaric {b:?}"),
    }
  }

  // Non-vacuity + a pinned coverage count: 128 phi + VECTORS × 128 projection.
  let expected = 128 + VECTORS as u64 * 128;
  assert_eq!(
    checks, expected,
    "expected {expected} bit-exact f64 comparisons (128 phi + {VECTORS}×128 projection), ran {checks}"
  );
  println!(
    "PLDA cross-crate equivalence: {checks} bit-exact f64 comparisons (dia vs diaric) all equal \
     — 128 phi eigenvalues + {VECTORS} vectors × 128 projection dims"
  );
}
