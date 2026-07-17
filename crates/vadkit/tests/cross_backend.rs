//! Cross-backend characterization gate (design spec §6). Measures the
//! **behavioral agreement** of two independent VAD stacks on the parity-corpus
//! clips, at the segment level:
//!
//! - **silero-ONNX** (the reference): the PUBLISHED `silero` crate at its
//!   default features, running the bundled Silero VAD ONNX graph through ONNX
//!   Runtime — 512-sample (32 ms) frames at 16 kHz — with silero's own
//!   `detect_speech` segmenter (hysteresis, `min_speech`/`min_silence`,
//!   `speech_pad`). A dev-dependency; `ort` never enters vadkit's runtime graph
//!   (`cargo tree -p vadkit -e no-dev -i ort` finds nothing — the Cargo.toml
//!   note).
//! - **vadkit-CoreML** (under test): `vadkit::VadModel` (the FluidInference
//!   unified 256 ms artifact, `silero-vad-unified-256ms-v6.2.1`) — 4096-sample
//!   (256 ms) frames — post-processed by the MINIMAL test-only harness below.
//!
//! # This is behavioral agreement, NOT bit parity — by construction
//!
//! The two stacks disagree at the bit level on purpose, along two axes:
//!
//! 1. **Different model versions.** silero ships an older bundled ONNX Silero
//!    graph; vadkit runs FluidInference's converted **v6.2.1** CoreML artifact.
//!    Different weights → different per-frame probabilities.
//! 2. **Different geometries.** silero emits one probability per **512** samples
//!    (32 ms); vadkit one per **4096** samples (256 ms) — an 8× coarser grid.
//!    vadkit's boundaries are quantized to 256 ms and it can never split a pause
//!    shorter than one 256 ms frame, where silero (32 ms grid, 100 ms
//!    `min_silence`) can — so vadkit legitimately produces FEWER, coarser
//!    segments (measured: 1 vs silero's 4–5). That is the documented geometry
//!    exception (spec §6 "segment count equality OR documented exceptions"),
//!    not a defect.
//!
//! So the gate does not — and must not — assert equal probabilities or identical
//! boundaries. It pins, two-sided from values MEASURED against the real models
//! (recorded in the constants below and this crate's T4 report), the aggregate
//! agreement a healthy pairing produces. Two complementary families of metric
//! cover the two mutations the campaign requires to turn this red:
//!
//! - **The grid metric catches a threshold swap.** Downsample silero to
//!   vadkit's 256 ms grid (a frame is speech if it holds ANY silero-speech) and
//!   count frames whose speech label differs from vadkit's `probability ≥
//!   threshold` call. MEASURED **0 on both clips** — the two independent models
//!   make the identical per-frame decision everywhere at the 0.5 threshold.
//!   Swapping the vadkit threshold 0.5 → 0.9 flips 2–3 confidently-but-not-
//!   overwhelmingly-speech boundary frames → `grid_disagree` 2–3 > the pinned 1.
//!   (These clips are 75–95 % speech and the VAD is very confident, so the
//!   threshold barely moves the aggregate masks — only the grid metric is
//!   sharp enough to catch it.)
//! - **The mask/span metrics catch a geometry lie.** Sample-level overlap
//!   (0.956–0.973), speech IoU (0.956–0.965), duration ratio (1.036–1.046) and
//!   the outer speech-envelope boundary deltas (≤ 0.194 s) all collapse when the
//!   harness places 4096-sample frames on silero's 512-sample stride
//!   (`frame_samples` 4096 → 512): the timeline compresses 8×, so overlap →
//!   0.15, IoU → 0.0–0.13, duration ratio → 0.13, and the envelope end-delta
//!   blows out to 21–26 s. (The grid metric is frame-decision based and so is
//!   deliberately blind to this — the two families do not overlap.)
//!
//! Model-gated (`#[ignore]`): needs `Models/vadkit` (`VADKIT_TEST_MODELS`) for
//! the CoreML side; the silero side uses its bundled model, no download.

mod common;

use coremlit::ComputeUnits;
use vadkit::{CHUNK_SAMPLES, VadModel, VadModelOptions};

/// 16 kHz — the corpus sample rate both stacks consume (asserted per clip).
const SAMPLE_RATE: u64 = 16_000;

/// The two committed parity fixtures (the same clips T3's Swift-trace gate uses;
/// `common::FIXTURES`): `02_pyannote_sample` (pyannote's multi-speaker demo,
/// 118 frames) and `07_yuhewei_dongbei_english` (a second conversational clip,
/// 99 frames, whose short final chunk exercises the padding path).
const GATE_FIXTURES: &[&str] = &["02_pyannote_sample", "07_yuhewei_dongbei_english"];

// ── The MINIMAL, TEST-ONLY vadkit post-processing harness ───────────────────
//
// INTERIM until T5. vadkit authors ZERO detection logic (spec §2-§3): the real
// segmenter is silero's, wired over the CoreML backend by T5's re-export layer,
// which does not exist yet. This harness exists ONLY so T4 can turn vadkit's
// per-frame probabilities into segments to compare. It is deliberately cruder
// than silero's segmenter — a fixed threshold plus min-duration merging, no
// hysteresis, no speech padding, no force-split — precisely so the gate
// measures the MODELS' agreement, not two copies of the same post-processor.
// It is NOT a public API and MUST NOT be promoted to one.

/// A half-open speech interval on the 16 kHz sample timeline, `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segment {
  start: u64,
  end: u64,
}

/// Configuration for [`vadkit_segments`] — the interim harness's knobs. Named so
/// the healthy config and the two mutants each differ from it by ONE field.
#[derive(Debug, Clone, Copy)]
struct HarnessConfig {
  /// Fixed speech threshold: a frame is speech iff `probability >= threshold`.
  threshold: f32,
  /// Interior silence gaps (in frames) no longer than this between two speech
  /// runs are bridged (filled to speech) — the "min-silence" half of the
  /// min-duration merge, approximating silero's dip-tolerant hysteresis on the
  /// coarse 256 ms grid.
  bridge_silence_frames: usize,
  /// Speech runs shorter than this (in frames, after bridging) are dropped — the
  /// "min-speech" half of the merge.
  min_speech_frames: usize,
  /// Samples per vadkit frame, used to place frame `i` at `[i·f, (i+1)·f)`. The
  /// TRUE geometry is [`CHUNK_SAMPLES`] (4096). The geometry-lie mutation sets
  /// this to silero's 512 — the single load-bearing timestamp constant.
  frame_samples: u64,
}

/// The healthy vadkit harness config the gate is pinned around. `threshold` 0.5
/// anchors on silero's `start_threshold`; `bridge_silence_frames` 1 lets a
/// single 256 ms probability dip stay inside a segment (silero, on its 32 ms
/// grid with 100 ms `min_silence` and 0.35 end-hysteresis, likewise rides out
/// sub-frame dips); `min_speech_frames` 1 keeps segments ≥ 256 ms (≈ silero's
/// 250 ms `min_speech`); `frame_samples` is the real 4096-sample geometry.
const HEALTHY: HarnessConfig = HarnessConfig {
  threshold: 0.5,
  bridge_silence_frames: 1,
  min_speech_frames: 1,
  frame_samples: CHUNK_SAMPLES as u64,
};

/// Mutation 1 — threshold swap 0.5 → 0.9 (spec §6 "swaps thresholds"). Every
/// other knob matches [`HEALTHY`]. At 0.9, vadkit drops 2–3 boundary frames,
/// so `grid_disagree` climbs from 0 past the pinned bound.
const MUT_THRESHOLD: HarnessConfig = HarnessConfig {
  threshold: 0.9,
  ..HEALTHY
};

/// Mutation 2 — geometry lie: place vadkit's 4096-sample frames on silero's
/// 512-sample stride (spec §6 "geometry"). Every other knob matches [`HEALTHY`].
/// This compresses vadkit's timeline 8×, collapsing overlap/IoU/duration and
/// blowing out the envelope boundary deltas.
const MUT_GEOMETRY: HarnessConfig = HarnessConfig {
  frame_samples: 512,
  ..HEALTHY
};

/// Turns vadkit's per-frame probabilities into segments per `cfg`: binarize at
/// the threshold, bridge short interior silence gaps, emit runs ≥
/// `min_speech_frames`, mapping frame `i` to `[i·frame_samples,
/// (i+1)·frame_samples)` clamped to `total_samples`.
fn vadkit_segments(probs: &[f32], cfg: HarnessConfig, total_samples: u64) -> Vec<Segment> {
  let mut speech: Vec<bool> = probs.iter().map(|&p| p >= cfg.threshold).collect();

  if cfg.bridge_silence_frames > 0
    && let (Some(first), Some(last)) = (
      speech.iter().position(|&s| s),
      speech.iter().rposition(|&s| s),
    )
  {
    let mut i = first;
    while i <= last {
      if speech[i] {
        i += 1;
        continue;
      }
      let gap_start = i;
      while i <= last && !speech[i] {
        i += 1;
      }
      if i - gap_start <= cfg.bridge_silence_frames {
        speech[gap_start..i].fill(true);
      }
    }
  }

  let mut segments = Vec::new();
  let mut i = 0;
  while i < speech.len() {
    if !speech[i] {
      i += 1;
      continue;
    }
    let run_start = i;
    while i < speech.len() && speech[i] {
      i += 1;
    }
    if i - run_start < cfg.min_speech_frames {
      continue;
    }
    let start = (run_start as u64) * cfg.frame_samples;
    let end = ((i as u64) * cfg.frame_samples).min(total_samples);
    if start < total_samples && end > start {
      segments.push(Segment { start, end });
    }
  }
  segments
}

/// Rasterizes segments to a per-sample speech mask on `[0, n)`.
fn speech_mask(segments: &[Segment], n: usize) -> Vec<bool> {
  let mut mask = vec![false; n];
  for seg in segments {
    let a = (seg.start as usize).min(n);
    let b = (seg.end as usize).min(n);
    mask[a..b].fill(true);
  }
  mask
}

// ── Pinned two-sided tolerances (MEASURED against the real models, then pinned
//    with documented margin; see the module docs and the T4 report) ───────────

/// Max tolerated `grid_disagree` frames per clip. **Measured 0** on both clips —
/// silero downsampled to vadkit's 256 ms grid and vadkit's `≥ 0.5` call agree on
/// every frame. Pinned at 1 (a single frame of headroom for future toolchain
/// drift, in the spirit of T3's `TRACE_TOL` headroom over a measured 0). The
/// threshold-swap mutation reaches 2 (02) / 3 (07), clear of the bound; the
/// geometry lie leaves this metric at 0 by construction (frame decisions do not
/// depend on the segment stride) — it is caught by the mask/span bounds instead.
const GRID_DISAGREE_MAX: usize = 1;

/// Sample-level speech/non-speech overlap ratio band. **Measured 0.973 (02) /
/// 0.956 (07)**; floor 0.90 leaves ≈ 0.05 margin. The geometry lie sends it to
/// 0.15–0.17, far below the floor. (Upper bound 1.0 is the natural ceiling —
/// perfect agreement — no healthy or mutant run approaches it.)
const OVERLAP_MIN: f64 = 0.90;
const OVERLAP_MAX: f64 = 1.0;

/// Speech-region IoU (Jaccard) floor. **Measured 0.965 (02) / 0.956 (07)**;
/// floor 0.85. The geometry lie sends it to 0.0 (02, disjoint) / 0.13 (07).
const IOU_MIN: f64 = 0.85;
const IOU_MAX: f64 = 1.0;

/// Total-speech duration ratio (vadkit ÷ silero) band. **Measured 1.036 (02) /
/// 1.046 (07)** — vadkit calls slightly more of the clip speech than silero.
/// Band [0.85, 1.20] straddles it two-sided; the geometry lie sends it to 0.13.
const DUR_RATIO_MIN: f64 = 0.85;
const DUR_RATIO_MAX: f64 = 1.20;

/// Max tolerated outer speech-envelope boundary delta (start and end), in
/// samples. **Measured worst 0.194 s** (07 end; 3 104 samples). Pinned at
/// 6 400 samples (0.40 s ≈ 1.5 frames), covering one 256 ms frame of
/// quantization plus silero's 30 ms pad plus margin. The geometry lie blows the
/// envelope end-delta out to 21–26 s.
const SPAN_DELTA_MAX_SAMPLES: u64 = 6_400;

// ── Metrics ─────────────────────────────────────────────────────────────────

/// The per-clip agreement between the two stacks. Every field is a MEASURED
/// characterization number; [`within_gate`] pins the load-bearing ones.
#[derive(Debug, Clone, Copy)]
struct Agreement {
  /// Frames where silero (downsampled to the 256 ms grid, any-speech rule) and
  /// vadkit (`probability ≥ threshold`) disagree — the threshold detector.
  grid_disagree: usize,
  /// Fraction of the timeline where both stacks agree (both speech or both
  /// silence) — the speech/non-speech overlap ratio.
  overlap_ratio: f64,
  /// Speech-region Jaccard: `|A∩B| / |A∪B|` over speech samples.
  speech_iou: f64,
  /// Total vadkit speech ÷ total silero speech.
  dur_ratio: f64,
  /// Outer speech-envelope boundary deltas (vadkit's `[min start, max end]` vs
  /// silero's), in samples.
  span_start_delta: u64,
  span_end_delta: u64,
  n_silero: usize,
  n_vadkit: usize,
}

/// Downsamples silero's sample-level speech mask to vadkit's 256 ms frame grid:
/// frame `i` is speech if ANY of its `CHUNK_SAMPLES` samples is silero-speech —
/// the natural correspondence ("a 256 ms frame holding speech is a speech
/// frame") under which the two independent models agree perfectly at threshold
/// 0.5. (A majority rule instead disagrees on boundary frames and, perversely,
/// AGREES MORE under the threshold mutation, so it cannot detect it — measured
/// and discarded.)
fn silero_grid(silero_mask: &[bool], n_frames: usize) -> Vec<bool> {
  (0..n_frames)
    .map(|i| {
      let a = i * CHUNK_SAMPLES;
      let b = (a + CHUNK_SAMPLES).min(silero_mask.len());
      silero_mask[a..b].iter().any(|&s| s)
    })
    .collect()
}

/// Characterizes the agreement between the silero reference segments and the
/// vadkit stack (raw `probs` post-processed by `cfg`) over a `total`-sample
/// timeline.
fn characterize(silero: &[Segment], probs: &[f32], cfg: HarnessConfig, total: usize) -> Agreement {
  let sil_mask = speech_mask(silero, total);

  // Grid metric: silero (any-speech, 256 ms) vs vadkit (threshold), per frame.
  let sil_grid = silero_grid(&sil_mask, probs.len());
  let grid_disagree = (0..probs.len())
    .filter(|&i| (probs[i] >= cfg.threshold) != sil_grid[i])
    .count();

  // Mask metrics from the harness segments.
  let vadkit = vadkit_segments(probs, cfg, total as u64);
  let vk_mask = speech_mask(&vadkit, total);
  let mut agree = 0u64;
  let mut inter = 0u64;
  let mut union = 0u64;
  let mut sil_speech = 0u64;
  let mut vk_speech = 0u64;
  for i in 0..total {
    let (a, b) = (sil_mask[i], vk_mask[i]);
    agree += u64::from(a == b);
    inter += u64::from(a && b);
    union += u64::from(a || b);
    sil_speech += u64::from(a);
    vk_speech += u64::from(b);
  }

  // Outer speech-envelope span deltas.
  let span = |segs: &[Segment]| -> (u64, u64) {
    (
      segs.iter().map(|s| s.start).min().unwrap_or(0),
      segs.iter().map(|s| s.end).max().unwrap_or(0),
    )
  };
  let (sil_lo, sil_hi) = span(silero);
  let (vk_lo, vk_hi) = span(&vadkit);

  Agreement {
    grid_disagree,
    overlap_ratio: agree as f64 / total as f64,
    speech_iou: if union == 0 {
      1.0
    } else {
      inter as f64 / union as f64
    },
    dur_ratio: if sil_speech == 0 {
      f64::INFINITY
    } else {
      vk_speech as f64 / sil_speech as f64
    },
    span_start_delta: vk_lo.abs_diff(sil_lo),
    span_end_delta: vk_hi.abs_diff(sil_hi),
    n_silero: silero.len(),
    n_vadkit: vadkit.len(),
  }
}

/// Checks an [`Agreement`] against every pinned bound, returning the list of
/// violations (empty ⇒ the clip is within the gate). Shared by the healthy gate
/// and the mutation-red proofs so the mutations are proven red against the SAME
/// bounds the gate enforces.
fn within_gate(a: &Agreement) -> Vec<String> {
  let mut v = Vec::new();
  if a.grid_disagree > GRID_DISAGREE_MAX {
    v.push(format!(
      "grid_disagree {} > {GRID_DISAGREE_MAX}",
      a.grid_disagree
    ));
  }
  if !(OVERLAP_MIN..=OVERLAP_MAX).contains(&a.overlap_ratio) {
    v.push(format!(
      "overlap_ratio {:.4} outside [{OVERLAP_MIN}, {OVERLAP_MAX}]",
      a.overlap_ratio
    ));
  }
  if !(IOU_MIN..=IOU_MAX).contains(&a.speech_iou) {
    v.push(format!(
      "speech_iou {:.4} outside [{IOU_MIN}, {IOU_MAX}]",
      a.speech_iou
    ));
  }
  if !(DUR_RATIO_MIN..=DUR_RATIO_MAX).contains(&a.dur_ratio) {
    v.push(format!(
      "dur_ratio {:.4} outside [{DUR_RATIO_MIN}, {DUR_RATIO_MAX}]",
      a.dur_ratio
    ));
  }
  if a.span_start_delta > SPAN_DELTA_MAX_SAMPLES {
    v.push(format!(
      "span_start_delta {} samples > {SPAN_DELTA_MAX_SAMPLES}",
      a.span_start_delta
    ));
  }
  if a.span_end_delta > SPAN_DELTA_MAX_SAMPLES {
    v.push(format!(
      "span_end_delta {} samples > {SPAN_DELTA_MAX_SAMPLES}",
      a.span_end_delta
    ));
  }
  // Documented geometry relationship: vadkit is coarser, so it never resolves
  // MORE segments than silero, and always finds at least one on a speech clip.
  if a.n_vadkit < 1 || a.n_vadkit > a.n_silero {
    v.push(format!(
      "n_vadkit {} outside [1, n_silero={}]",
      a.n_vadkit, a.n_silero
    ));
  }
  v
}

/// One-line render of an [`Agreement`] for the recorded characterization output.
fn render(clip: &str, tag: &str, a: &Agreement) -> String {
  format!(
    "[xback] {clip:26} {tag:9} grid_disagree={} overlap={:.4} iou={:.4} \
     dur_ratio={:.4} span_d=({:.3}s,{:.3}s) n_silero={} n_vadkit={}",
    a.grid_disagree,
    a.overlap_ratio,
    a.speech_iou,
    a.dur_ratio,
    a.span_start_delta as f64 / SAMPLE_RATE as f64,
    a.span_end_delta as f64 / SAMPLE_RATE as f64,
    a.n_silero,
    a.n_vadkit,
  )
}

// ── The two stacks ──────────────────────────────────────────────────────────

/// Runs the silero-ONNX reference stack: bundled model, default `SpeechOptions`
/// (0.5 start / 0.35 end threshold, 250 ms min-speech, 100 ms min-silence,
/// 30 ms pad, 512-sample frames), one-shot `detect_speech`. Returns segments on
/// the 16 kHz timeline.
fn silero_segments(samples: &[f32]) -> Vec<Segment> {
  let mut session = silero::Session::bundled().expect("load bundled silero ONNX model");
  let options = silero::SpeechOptions::default();
  assert_eq!(
    options.sample_rate(),
    silero::SampleRate::Rate16k,
    "silero reference must run at 16 kHz"
  );
  silero::detect_speech(&mut session, samples, options)
    .expect("silero detect_speech")
    .into_iter()
    .map(|s| Segment {
      start: s.start_sample(),
      end: s.end_sample(),
    })
    .collect()
}

/// Runs vadkit's CoreML model layer over the 4096-stride chunking on `cpu_only`
/// (deterministic; matches the trace oracle's placement) and returns one speech
/// probability per 256 ms frame.
fn vadkit_probs(samples: &[f32]) -> Vec<f32> {
  let mut model = VadModel::load_with(
    common::model_path(),
    VadModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load vadkit CoreML model");
  samples
    .chunks(CHUNK_SAMPLES)
    .map(|chunk| model.predict_chunk(chunk).expect("vadkit predict_chunk"))
    .collect()
}

/// Loads a fixture clip, proving its bytes match the SHA-256 pinned in
/// `common::FIXTURES` (the same audio T3's gate saw), and its geometry.
fn load_fixture(name: &str) -> Vec<f32> {
  let fixture = common::FIXTURES
    .iter()
    .find(|f| f.name == name)
    .unwrap_or_else(|| panic!("no fixture entry for {name}"));
  let path = common::fixture_wav_path(name);
  assert_eq!(
    common::sha256_hex(&path),
    fixture.sha256,
    "{name}: fixture audio SHA-256 changed"
  );
  common::load_wav_16k_mono(&path)
}

// ── The gate ────────────────────────────────────────────────────────────────

/// **THE CROSS-BACKEND GATE** (model-gated). For each parity clip: run the
/// silero-ONNX reference and the vadkit-CoreML stack, characterize their
/// agreement, and require every pinned bound to hold — recording the measured
/// numbers. The two `mutation_*` tests below prove the bounds turn red under a
/// threshold swap and a geometry lie.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn cross_backend_agreement_holds() {
  for &clip in GATE_FIXTURES {
    let samples = load_fixture(clip);
    let total = samples.len();
    assert!(total > CHUNK_SAMPLES, "{clip}: clip shorter than one frame");

    let silero = silero_segments(&samples);
    assert!(!silero.is_empty(), "{clip}: silero found no speech");
    let probs = vadkit_probs(&samples);

    let agreement = characterize(&silero, &probs, HEALTHY, total);
    println!("{}", render(clip, "HEALTHY", &agreement));

    let violations = within_gate(&agreement);
    assert!(
      violations.is_empty(),
      "{clip}: healthy cross-backend agreement violates the gate: {violations:?}"
    );
  }
}

/// Mutation 1 (recorded red): swapping the vadkit threshold 0.5 → 0.9 drives
/// `grid_disagree` past its bound — the same gate [`cross_backend_agreement_holds`]
/// enforces now reports a violation. Proves the gate is sensitive to the
/// threshold; the healthy run above proves it is not merely always-red.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn mutation_threshold_swap_breaks_gate() {
  for &clip in GATE_FIXTURES {
    let samples = load_fixture(clip);
    let total = samples.len();
    let silero = silero_segments(&samples);
    let probs = vadkit_probs(&samples);

    let agreement = characterize(&silero, &probs, MUT_THRESHOLD, total);
    println!("{}", render(clip, "mut:thr", &agreement));

    let violations = within_gate(&agreement);
    assert!(
      !violations.is_empty(),
      "{clip}: threshold 0.5->0.9 mutation must break the gate but did not"
    );
    assert!(
      violations.iter().any(|m| m.starts_with("grid_disagree")),
      "{clip}: threshold mutation must trip the grid metric; violations were {violations:?}"
    );
  }
}

/// Mutation 2 (recorded red): lying about the geometry — placing vadkit's
/// 4096-sample frames on silero's 512-sample stride — collapses the sample-level
/// agreement and blows out the envelope boundary deltas, tripping the mask/span
/// bounds. The grid metric (frame-decision based) is deliberately untouched, so
/// this proves a DIFFERENT family of bound from mutation 1.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn mutation_geometry_lie_breaks_gate() {
  for &clip in GATE_FIXTURES {
    let samples = load_fixture(clip);
    let total = samples.len();
    let silero = silero_segments(&samples);
    let probs = vadkit_probs(&samples);

    let agreement = characterize(&silero, &probs, MUT_GEOMETRY, total);
    println!("{}", render(clip, "mut:geom", &agreement));

    let violations = within_gate(&agreement);
    assert!(
      violations.iter().any(|m| m.starts_with("overlap_ratio")),
      "{clip}: geometry lie must trip the sample-overlap bound; violations were {violations:?}"
    );
    assert!(
      violations.iter().any(|m| m.starts_with("span_end_delta")),
      "{clip}: geometry lie must trip the envelope boundary bound; violations were {violations:?}"
    );
  }
}
