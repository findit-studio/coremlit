//! Cross-backend characterization gate (design spec §6). Measures the
//! **behavioral agreement** of two independent VAD stacks on the parity-corpus
//! clips, at the segment level:
//!
//! - **silero-ONNX** (the reference): the `silero` crate at its default
//!   features, running the bundled Silero VAD ONNX graph through ONNX Runtime —
//!   512-sample (32 ms) frames at 16 kHz — with silero's own `detect_speech`
//!   segmenter (hysteresis, `min_speech`/`min_silence`, `speech_pad`). A
//!   dev-dependency; `ort` never enters vadkit's runtime graph
//!   (`cargo tree -p coremlit --features vad -e no-dev -i ort` finds nothing — the Cargo.toml
//!   note).
//! - **vadkit-CoreML** (under test): the FluidInference unified 256 ms artifact
//!   (`silero-vad-unified-256ms-v6.2.1`) — 4096-sample (256 ms) frames — turned
//!   into segments by **silero's own segmenter through vadkit's re-export**
//!   ([`coremlit::audio::vad::detect_speech`], which forwards to `silero::detect_speech_with`
//!   over [`coremlit::audio::vad::CoreMlBackend`]). This REPLACES T4's interim test-local
//!   thresholding harness (T5): the vadkit side now runs the exact same
//!   detection logic as the reference, differing only in model and geometry —
//!   which is what makes the agreement a clean model-vs-model measurement.
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
//! (recorded in the constants below and this crate's T5 report), the aggregate
//! agreement a healthy pairing produces. Two complementary families of metric
//! cover the two mutations the campaign requires to turn this red:
//!
//! - **The grid metric catches a threshold swap.** Downsample silero to
//!   vadkit's 256 ms grid (a frame is speech if it holds ANY silero-speech) and
//!   count frames whose speech label differs from vadkit's `probability ≥
//!   threshold` call. MEASURED **0 on both clips** — the two independent models
//!   make the identical per-frame decision everywhere at the 0.5 threshold.
//!   Swapping that characterization threshold 0.5 → 0.9 flips 2–3 confidently-
//!   but-not-overwhelmingly-speech boundary frames → `grid_disagree` 2–3 > the
//!   pinned 1. (These clips are 75–95 % speech and the VAD is very confident, so
//!   the threshold barely moves the aggregate masks — only the grid metric is
//!   sharp enough to catch it. The grid metric reads vadkit's RAW per-256 ms
//!   probabilities, which is model inference, not segment-assembly logic — it
//!   stays here after T5's harness removal.)
//! - **The mask/span metrics catch a geometry lie.** Sample-level overlap, speech
//!   IoU, duration ratio and the outer speech-envelope boundary deltas all
//!   collapse when silero's real [`SpeechSegmenter`] is driven over vadkit's
//!   256 ms probabilities at silero's 512-sample stride
//!   ([`SpeechSegmenter::set_frame_samples`] 4096 → 512): the timeline
//!   compresses 8×, so overlap and IoU crater and the envelope end-delta blows
//!   out by tens of seconds. (The grid metric is frame-decision based and so is
//!   deliberately blind to this — the two families do not overlap.)
//!
//! Model-gated (`#[ignore]`): needs `Models/vadkit` (`VADKIT_TEST_MODELS`) for
//! the CoreML side; the silero side uses its bundled model, no download.

mod common;

use coremlit::{
  ComputeUnits,
  audio::vad::{CHUNK_SAMPLES, CoreMlBackend, VadModel, VadModelOptions, detect_speech},
};
use silero::{SpeechOptions, SpeechSegmenter};

/// 16 kHz — the corpus sample rate both stacks consume (asserted per clip).
const SAMPLE_RATE: u64 = 16_000;

/// The two committed parity fixtures (the same clips T3's Swift-trace gate uses;
/// `common::FIXTURES`): `02_pyannote_sample` (pyannote's multi-speaker demo,
/// 118 frames) and `07_yuhewei_dongbei_english` (a second conversational clip,
/// 99 frames, whose short final chunk exercises the padding path).
const GATE_FIXTURES: &[&str] = &["02_pyannote_sample", "07_yuhewei_dongbei_english"];

/// The grid-metric characterization threshold a HEALTHY run uses: vadkit calls a
/// 256 ms frame speech when its probability ≥ this. Anchored on silero's
/// `start_threshold` default. This is a TEST-SIDE measurement threshold over the
/// models' raw probabilities, not vadkit's detection threshold (that lives in
/// silero's segmenter, driven at its own default through the re-export).
const GRID_THRESHOLD_HEALTHY: f32 = 0.5;
/// Mutation 1 — the threshold-swap knob (spec §6 "swaps thresholds"): raise the
/// grid characterization threshold 0.5 → 0.9. At 0.9, vadkit drops 2–3 boundary
/// frames its raw probability no longer clears, so `grid_disagree` climbs from 0
/// past the pinned bound.
const GRID_THRESHOLD_MUTANT: f32 = 0.9;

/// Mutation 2 — the geometry lie (spec §6 "geometry"): drive silero's real
/// segmenter over vadkit's 256 ms probabilities at silero's 512-sample stride
/// instead of the true [`CHUNK_SAMPLES`] (4096). Compresses vadkit's timeline
/// 8×, collapsing overlap/IoU and blowing out the envelope boundary deltas.
const MUTANT_FRAME_SAMPLES: usize = 512;

// ── A half-open speech interval on the 16 kHz sample timeline, `[start, end)` ─

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segment {
  start: u64,
  end: u64,
}

impl Segment {
  fn of(seg: silero::SpeechSegment) -> Self {
    Self {
      start: seg.start_sample(),
      end: seg.end_sample(),
    }
  }
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
//    with documented margin; see the module docs and the T5 report) ───────────
//
// The bounds are UNCHANGED from T4 (the task forbids re-pinning): the T5
// re-export runs nearly the same segmentation math the interim harness
// approximated (silero's segmenter with its 100 ms `min_silence` closes on the
// second 256 ms low frame, exactly the harness's 1-frame bridge; 250 ms
// `min_speech` is under one 256 ms frame, exactly the harness's 1-frame
// minimum), so the healthy values land inside the same bands with room to
// spare. The "measured (re-export)" numbers below are the T5 re-measurement.

/// Max tolerated `grid_disagree` frames per clip. **Measured 0** on both clips —
/// silero downsampled to vadkit's 256 ms grid and vadkit's `≥ 0.5` call agree on
/// every frame. Pinned at 1 (a single frame of headroom for future toolchain
/// drift). The threshold-swap mutation reaches 2 (02) / 3 (07); the geometry lie
/// leaves this metric at 0 by construction (frame decisions do not depend on the
/// segment stride) — it is caught by the mask/span bounds instead.
const GRID_DISAGREE_MAX: usize = 1;

/// Sample-level speech/non-speech overlap ratio band. **Measured (re-export)
/// 0.972 (02) / 0.949 (07)**; floor 0.90 leaves ≈ 0.05 margin. The geometry lie
/// sends it to 0.15 (02) / 0.17 (07), far below the floor. (Upper bound 1.0 is
/// the natural ceiling — no healthy or mutant run approaches it.)
const OVERLAP_MIN: f64 = 0.90;
const OVERLAP_MAX: f64 = 1.0;

/// Speech-region IoU (Jaccard) floor. **Measured (re-export) 0.964 (02) / 0.949
/// (07)**; floor 0.85. The geometry lie sends it to 0.00 (02, disjoint) / 0.13
/// (07).
const IOU_MIN: f64 = 0.85;
const IOU_MAX: f64 = 1.0;

/// Total-speech duration ratio (vadkit ÷ silero) band. **Measured (re-export)
/// 1.038 (02) / 1.054 (07)** — vadkit calls slightly more of the clip speech
/// than silero (its coarse grid + 30 ms pad + 0.35 end-hysteresis bridge/extend
/// silero's short pauses). Band [0.85, 1.20] straddles it two-sided; the
/// geometry lie sends it to 0.13.
const DUR_RATIO_MIN: f64 = 0.85;
const DUR_RATIO_MAX: f64 = 1.20;

/// Max tolerated outer speech-envelope boundary delta (start and end), in
/// samples. **Measured (re-export) worst 0.369 s** (07 end; 5 904 samples — the
/// margin to the bound is tighter than T4's harness measured, because silero's
/// real segmenter holds 07's trailing segment ≈ one grid step later than the
/// hard-0.5 harness did, via its 0.35 end-hysteresis and 30 ms pad). Pinned at
/// 6 400 samples (0.40 s ≈ 1.5 frames), covering one 256 ms frame of
/// quantization plus silero's pad plus margin. The geometry lie blows the
/// envelope end-delta out to 21–26 s. This value is deterministic (`cpu_only`,
/// SHA-pinned artifact, pinned silero rev); a dependency bump is the correct
/// trigger to re-measure it.
const SPAN_DELTA_MAX_SAMPLES: u64 = 6_400;

// ── Metrics ─────────────────────────────────────────────────────────────────

/// The per-clip agreement between the two stacks. Every field is a MEASURED
/// characterization number; [`within_gate`] pins the load-bearing ones.
#[derive(Debug, Clone, Copy)]
struct Agreement {
  /// Frames where silero (downsampled to the 256 ms grid, any-speech rule) and
  /// vadkit (`probability ≥ grid_threshold`) disagree — the threshold detector.
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
/// 0.5.
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
/// vadkit segments over a `total`-sample timeline. `probs` are vadkit's raw
/// per-256 ms probabilities (for the grid metric, thresholded at
/// `grid_threshold`); `vadkit` are the segments the vadkit stack produced (the
/// real re-export for a healthy run, or silero's segmenter at a lied stride for
/// the geometry mutation).
fn characterize(
  silero: &[Segment],
  vadkit: &[Segment],
  probs: &[f32],
  grid_threshold: f32,
  total: usize,
) -> Agreement {
  let sil_mask = speech_mask(silero, total);

  // Grid metric: silero (any-speech, 256 ms) vs vadkit (threshold), per frame.
  let sil_grid = silero_grid(&sil_mask, probs.len());
  let grid_disagree = (0..probs.len())
    .filter(|&i| (probs[i] >= grid_threshold) != sil_grid[i])
    .count();

  // Mask metrics from the vadkit segments.
  let vk_mask = speech_mask(vadkit, total);
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

  // Outer speech-envelope span deltas. Ends are clamped to the audio length
  // (as T4's harness clamped every segment end to `total_samples`): silero's
  // `detect_speech_with` zero-pads a trailing PARTIAL frame and closes the
  // segment at the padded FRAME boundary (`n_frames * frame_samples`), which
  // can overhang the true sample count by up to one frame (e.g. 07's 99·4096 =
  // 405 504 past its 404 160 samples). That overhang is a timeline artifact,
  // not real speech past the end of the audio, so the honest in-audio envelope
  // — the same one the interim harness measured — clamps it away. It is a
  // no-op for `02` (no partial frame) and for the compressed geometry-lie
  // segments (all well within `total`), so it changes neither the healthy `02`
  // value nor the mutation's blow-out.
  let clamp = total as u64;
  let span = |segs: &[Segment]| -> (u64, u64) {
    (
      segs.iter().map(|s| s.start.min(clamp)).min().unwrap_or(0),
      segs.iter().map(|s| s.end.min(clamp)).max().unwrap_or(0),
    )
  };
  let (sil_lo, sil_hi) = span(silero);
  let (vk_lo, vk_hi) = span(vadkit);

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
  let options = SpeechOptions::default();
  assert_eq!(
    options.sample_rate(),
    silero::SampleRate::Rate16k,
    "silero reference must run at 16 kHz"
  );
  silero::detect_speech(&mut session, samples, options)
    .expect("silero detect_speech")
    .into_iter()
    .map(Segment::of)
    .collect()
}

/// Runs vadkit's per-256 ms CoreML probabilities on `cpu_only` (deterministic;
/// matches the trace oracle's placement). One probability per 256 ms frame — the
/// raw model output the grid metric and the geometry-lie mutation drive, kept
/// separate from segment assembly (which is silero's, below).
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

/// **The real re-export path** (T5): vadkit's segments as produced by silero's
/// own segmenter over the CoreML backend — [`detect_speech`] forwarding to
/// `silero::detect_speech_with`. Default `SpeechOptions`, `cpu_only`
/// (deterministic). This is the healthy vadkit stack the gate measures — the
/// same detection logic as the reference, only a different model and geometry.
fn vadkit_detect_segments(samples: &[f32]) -> Vec<Segment> {
  let mut backend = CoreMlBackend::load_with(
    common::model_path(),
    VadModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load vadkit CoreML backend");
  detect_speech(&mut backend, samples, SpeechOptions::default())
    .expect("vadkit detect_speech")
    .into_iter()
    .map(Segment::of)
    .collect()
}

/// Drives silero's REAL [`SpeechSegmenter`] over vadkit's per-256 ms `probs` at a
/// chosen `frame_samples` stride, collecting the segments — the same segmenter
/// [`detect_speech`] uses internally, exposed here so the geometry-lie mutation
/// can feed it silero's 512-sample stride instead of the true 4096 while the
/// probabilities stay vadkit's real 256 ms outputs. Authors no detection logic:
/// `SpeechSegmenter` is silero's.
fn silero_segmenter_segments(probs: &[f32], frame_samples: usize) -> Vec<Segment> {
  let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
  segmenter.set_frame_samples(frame_samples);
  let mut segments = Vec::new();
  for &probability in probs {
    if let Some(segment) = segmenter.push_probability(probability) {
      segments.push(Segment::of(segment));
    }
  }
  if let Some(segment) = segmenter.finish() {
    segments.push(Segment::of(segment));
  }
  segments
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
/// silero-ONNX reference and the vadkit-CoreML stack (through the real
/// re-export), characterize their agreement, and require every pinned bound to
/// hold — recording the measured numbers. The two `mutation_*` tests below prove
/// the bounds turn red under a threshold swap and a geometry lie.
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
    let vadkit = vadkit_detect_segments(&samples);

    let agreement = characterize(&silero, &vadkit, &probs, GRID_THRESHOLD_HEALTHY, total);
    println!("{}", render(clip, "HEALTHY", &agreement));

    let violations = within_gate(&agreement);
    assert!(
      violations.is_empty(),
      "{clip}: healthy cross-backend agreement violates the gate: {violations:?}"
    );
  }
}

/// Mutation 1 (recorded red): raising the grid characterization threshold
/// 0.5 → 0.9 drives `grid_disagree` past its bound — the same gate
/// [`cross_backend_agreement_holds`] enforces now reports a violation. The
/// vadkit SEGMENTS are the healthy re-export (unchanged), so only the threshold
/// family moves: proves the gate is sensitive to the threshold without being
/// merely always-red.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn mutation_threshold_swap_breaks_gate() {
  for &clip in GATE_FIXTURES {
    let samples = load_fixture(clip);
    let total = samples.len();
    let silero = silero_segments(&samples);
    let probs = vadkit_probs(&samples);
    let vadkit = vadkit_detect_segments(&samples);

    let agreement = characterize(&silero, &vadkit, &probs, GRID_THRESHOLD_MUTANT, total);
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

/// Mutation 2 (recorded red): lying about the geometry — driving silero's real
/// segmenter over vadkit's 256 ms probabilities at silero's 512-sample stride —
/// collapses the sample-level agreement and blows out the envelope boundary
/// deltas, tripping the mask/span bounds. The grid metric (frame-decision based)
/// is deliberately untouched, so this proves a DIFFERENT family of bound from
/// mutation 1.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn mutation_geometry_lie_breaks_gate() {
  for &clip in GATE_FIXTURES {
    let samples = load_fixture(clip);
    let total = samples.len();
    let silero = silero_segments(&samples);
    let probs = vadkit_probs(&samples);
    let vadkit_lied = silero_segmenter_segments(&probs, MUTANT_FRAME_SAMPLES);

    let agreement = characterize(&silero, &vadkit_lied, &probs, GRID_THRESHOLD_HEALTHY, total);
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
