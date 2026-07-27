//! **The shipping-configuration DER gate**: does the diarizer configuration
//! speakerkit actually ships — the fp32 `wespeaker.mlmodelc` embedder with
//! both CoreML models on [`ComputeUnits::All`] — diarize multi-speaker audio
//! the way dia's own ONNX reference does, and what does each compute
//! placement cost against that?
//!
//! # History: why the shipping embedder is fp32 (issue #15)
//!
//! Until issue #15 this crate shipped the int8-palettized
//! `wespeaker_v2.mlmodelc`, and this suite's job was the precision axis. That
//! configuration SILENTLY COLLAPSED 8-speaker audio: 5 of 8 speakers at
//! 16.5904 % DER (100 % confusion) on clip 09, where dia-ort is
//! frame-perfect. The attribution chain lives in
//! `tests/speaker/backend_factorial.rs`:
//!
//! - the embedding conversion itself is EXONERATED (fp32/`CpuOnly` over dia's
//!   reference segmentation reproduces dia-ort frame-perfectly, min cosine
//!   1.000000 over all 2 114 rows);
//! - the int8 palettization costs 2 speakers at either placement, and its
//!   perturbation is a COHERENT shared displacement (half the delta mass in
//!   one direction, 23x the isotropic null) that compresses between-speaker
//!   margins in the frozen community-1 PLDA space while leaving
//!   within-cluster tightness untouched — `quantization_error_structure`
//!   pins that mechanism;
//! - the `All` placement's fp16 scatter costs 1 further speaker on that clip
//!   at either precision.
//!
//! int8 offered no stable speed edge to lose: across two warm
//! [`shipping_embedder_cost_int8_vs_fp32`] runs the int8-vs-fp32 extraction
//! difference stayed ≤ ~15 % on every placement with the SIGN flipping
//! between runs — inside scheduler variability, which is why that bench
//! prints rather than asserts a winner (regimes and numbers in
//! `model_io.rs`'s DECISION). What palettization bought was ~21 MB of
//! footprint. The palettization is also
//! structurally aggressive: 38 per-tensor 8-bit LUTs, one flat 256-entry
//! codebook per whole tensor, covering even the deterministic DSP constants
//! (the STFT cos/sin bases and the mel filterbank) and the 5120→256 embedding
//! head. Issue #15 retired it from the shipping selection
//! (`FluidAudioArtifacts::resolve`); it stays on disk as a tested,
//! non-shipping sibling so the factorial's record remains reproducible.
//!
//! With precision fixed at fp32, the axis that remains is PLACEMENT, and that
//! is what this suite now measures and gates.
//!
//! # The arms (all fed ONE audio buffer — the input-identity proof)
//!
//! Per clip, on the identical `Vec<f32>` (FNV-1a fingerprinted before and
//! after every arm, asserted unchanged — a divergence caused by a different
//! input is a harness bug, not a finding; this exact trap produced a fake
//! "86 % divergence" in a sibling crate):
//!
//! | arm | segmentation | embedder | role |
//! |---|---|---|---|
//! | `dia-ort` | dia ONNX | dia fp32 ONNX | the oracle |
//! | `sAll+eAll` | CoreML `All` | CoreML `All` | the **literal shipping default** |
//! | `sAll+eCpu` | CoreML `All` | CoreML `CpuOnly` | the embedder-placement control |
//! | `sCpu+eCpu` | CoreML `CpuOnly` | CoreML `CpuOnly` | the all-CPU deterministic fallback |
//!
//! Every speakerkit arm loads the embedder through the SAME
//! `FluidAudioArtifacts` resolver production uses, so the shipping arm IS the
//! shipping selection by construction. Grid geometry (`num_chunks` /
//! `num_output_frames`) is asserted equal across every arm AND against
//! dia-ort's own pipeline, so no comparison is made across a misaligned
//! framing.
//!
//! # The gates
//!
//! [`gate`] (clips 06 / 14 / 10): **G0** every arm clusters; **G1** every
//! arm's speaker count equals dia-ort's (the decision metric — the one argmax
//! violated 7→8, and the one the retired int8 embedder violated 8→5);
//! **G2** the shipping placement stays within [`SHIPPING_ABS_DELTA_MAX`] of
//! the CPU-embedder control on reference agreement; **G3** arm-vs-control
//! confusion under [`SHIPPING_CONFUSION_TRIPWIRE`].
//!
//! Clip 09 (8 speakers) cannot run the plain gate: its two CONTROL arms sit
//! on a real, pinned segmentation knife edge — a spurious 9th speaker with
//! the embedder on CPU, and diaric's alive-band `Err` refusal with everything
//! on CPU, each pinned as its own observed outcome (reading both as one
//! near-threshold cluster is the interpretation the pattern supports, not an
//! asserted cross-arm identity — [`assert_clip09_record`]'s doc carries the
//! distinction). Its SHIPPING arm is gated at 8 of 8 speakers with a pinned
//! DER band, and the control states are pinned fail-if-changed.
//!
//! # The diagnostic: confusion, not DER
//!
//! DER decomposes into miss + false-alarm + confusion. Miss/FA move with
//! *speech/non-speech* boundaries — benign jitter the 0.25 s collar exists to
//! absorb. **Confusion** means speech was attributed to the WRONG speaker.
//! Argmax was caught by confusion (3.33 % DER, all of it confusion, plus a
//! speaker-count flip); the int8 collapse was 100 % confusion too. But
//! confusion between two of our own arms is an AGREEMENT statistic, not a
//! correctness one, so it carries only the gross-regression tripwire
//! ([`SHIPPING_CONFUSION_TRIPWIRE`]); the tight gates are the speaker count
//! (exact) and reference agreement ([`SHIPPING_ABS_DELTA_MAX`]).
//!
//! # The clips
//!
//! [`MULTI_SPEAKER_CLIPS`] — a SELECTED subset of dia's parity corpus: four of
//! its EIGHT ≥ 3-speaker clips (06 = 3, 14 = 4, 10 = 7, 09 = 8), spanning the
//! speaker-count ladder at the lowest runtime — the regime where clustering can
//! actually fail. The full ≥ 3-speaker membership and this selection are pinned
//! against silent drift by [`shipping_clip_selection_is_the_documented_subset`].
//! `parity_e2e`'s fixtures (2 speakers, ≤ 30 s) cannot express this failure:
//! argmax scored 0.0000 % on them and still broke at 7 speakers, and the int8
//! collapse needed 8. **The gate must run on audio hard enough to fail.**
//!
//! # The reference (pyannote's output, not ground truth)
//!
//! `reference.rttm` is **pyannote.audio 4.0.4's own output** on the clip (dia's
//! `manifest.json`), not human labels — the upstream reference implementation
//! the stack targets. Absolute DER here means "distance to pyannote 4.0.4",
//! reported honestly as such. The *decision* gate is against dia-ort and the
//! placement controls, which are apples-to-apples.
//!
//! `#[ignore]`d (needs the gitignored `Models/speakerkit`, the sibling
//! `diarization` ONNX + fixtures, and `ort`). Run with:
//!
//! ```text
//! cargo test -p coremlit --features speaker-oracle --test speaker_parity_shipping_der -- --ignored --nocapture
//! ```
//!
//! When swapping a `Models/speakerkit` artifact, finish staging the WHOLE
//! bundle before any model-gated run starts — stage aside and rename, rather
//! than copying files into a live bundle one by one. The byte pins read every
//! file of a bundle, so a partially-staged swap fails them deterministically
//! with hashes that look plausible (pre-repair siblings share bytes), and the
//! failure looks like a code regression until the mtimes are checked. With
//! multiple checkouts on one machine, `SPEAKERKIT_TEST_MODELS` additionally
//! pins WHICH tree a run reads (the default resolves from the building
//! checkout's manifest path).
#![cfg(feature = "speaker-oracle")]

mod common;
mod der_calc;

use std::{path::Path, time::Instant};

use coremlit::{
  ComputeUnits,
  audio::speaker::{
    embed::{EmbedModel, EmbedModelOptions},
    extract::{Extraction, Options},
    segment::{SegmentModel, SegmentModelOptions},
    source::{AnySource, FluidAudioArtifacts, FluidAudioSource, ModelSource},
  },
};
use der_calc::{
  Der, Seg, const_str_eq, der_std, der_strict, distinct_speakers, fmt_der, parse_rttm,
};

// ══════════════════════════════════════════════════════════════════════
// The gate bounds
// ══════════════════════════════════════════════════════════════════════

/// **The decision gate.** Ceiling on |DER(arm vs pyannote) − DER(control vs
/// pyannote)| (standard, 0.25 s collar) for the shipping placement
/// (`sAll+eAll`) and the all-CPU fallback (`sCpu+eCpu`), each against the
/// CPU-embedder control (`sAll+eCpu`): how much *agreement with the
/// independent reference* a compute placement may cost.
///
/// Derivation (issue #15 re-derivation; Apple M1 Max, macOS 26.5 build
/// 25F71, arm64, release harness, the byte-pinned fp16-safe artifacts).
/// Measured placement deltas on the three count-stable gated clips:
///
/// ```text
/// clip |  Δ(sAll+eAll − ctrl) | Δ(sCpu+eCpu − ctrl)
///   06 |            +0.3069pp |           +0.0508pp
///   14 |            +0.2843pp |           +0.3220pp
///   10 |            +0.0369pp |           +0.0000pp
/// ```
///
/// Worst measured |Δ| = 0.3220 pp. The bound is 1 pp: ≥ 3x the worst
/// measured placement drift (headroom for scheduler/host variation without
/// absorbing a real regression), and 3.3x BELOW the smallest known real
/// clustering failure on this axis — the argmax source's +3.33 pp — so that
/// failure class cannot pass. Clip 09 is excluded from this gate: its
/// control arms sit on the pinned segmentation knife edge
/// ([`assert_clip09_record`]), so its shipping arm carries its own pinned
/// band instead. Never loosened.
const SHIPPING_ABS_DELTA_MAX: f64 = 0.01;

/// Gross-regression tripwire on the **confusion** component of the
/// arm-vs-control parity DER — speech attributed to a DIFFERENT speaker than
/// the CPU-embedder control put it under.
///
/// # Why this is a tripwire and not a tight bound (read before changing it)
///
/// Arm-vs-control confusion is a cross-placement AGREEMENT proxy, not a
/// correctness metric: clip 14's clustering sits near a decision boundary
/// where ANY perturbation (the conversion itself, a placement, formerly the
/// int8 quantization) moves a marginal assignment, so a tight bound here
/// would gate something no placement pair achieves. The decision metrics are
/// the speaker count (exact equality, G1) and [`SHIPPING_ABS_DELTA_MAX`]
/// (agreement vs the reference); this tripwire only guards against a
/// CATASTROPHIC clustering divergence between placements. It sits well above
/// the placement-drift scale the deltas above imply (every arm-vs-control
/// DER is ≤ ~0.4 pp on the count-stable clips, and confusion is bounded by
/// that DER) and below the one known real failure on this axis — the argmax
/// source's 3.33 % DER, 100 % of it confusion — so that failure still trips
/// it. It is a controller decision, and it is never raised to hide a
/// regression.
const SHIPPING_CONFUSION_TRIPWIRE: f64 = 0.02;

/// Two-sided tolerance (±0.05 pp) on the clip-09 known-defect DER pins
/// ([`assert_clip09_known_defect`]) — the SAME band `parity_e2e`'s `DER_PIN_TOL`
/// uses, for the same reason: the pipeline is deterministic per placement and
/// the int8-era clip-09 values reproduced EXACTLY across two independent full
/// runs (spec §5.9), so the band exists only to absorb a stray flipped frame
/// on a different CoreML build, not to hide movement. The knife-edge states
/// this band pins are speaker-count and error-mode changes far larger than
/// it, so any clustering-decision change fires immediately. Never widened to
/// make a pin pass.
const DER_PIN_TOL: f64 = 0.000_5;

/// A dia parity-corpus clip with ≥ 3 reference speakers — the regime where
/// clustering can actually fail. (`parity_e2e`'s own fixtures top out at 2
/// speakers; argmax scored 0.0000 % on them and still broke at 7.)
struct MultiSpkClip {
  /// Fixture directory name under dia's `tests/parity/fixtures/`.
  name: &'static str,
  /// Distinct speakers in `reference.rttm` — asserted, so a corpus change that
  /// silently drops the multi-speaker coverage fails loudly instead of turning
  /// this suite into a 2-speaker no-op.
  ref_spk: usize,
  /// Decoded 16 kHz-mono sample count of `clip_16k.wav`, pinned so the clip's
  /// identity is its AUDIO, not just its directory name. Asserted in
  /// [`measure`] immediately after load: a same-name swap (a re-encode, a
  /// truncation, a wrong file dropped in) changes this and fails BEFORE any DER
  /// is scored. F4 — the clip-09 defect pin ([`assert_clip09_known_defect`])
  /// checks only `o.clip == "09..."`, the manifest string, which a same-name
  /// swap sails through.
  samples: usize,
  /// [`common::fnv1a_f32`] of those decoded samples — the content half of the
  /// identity pin, catching a same-LENGTH content change the sample count
  /// alone cannot. Asserted alongside [`Self::samples`] in [`measure`].
  audio_fnv: u64,
}

/// The gated clips: a SELECTED subset — four of the eight ≥ 3-speaker clips in
/// dia's parity corpus ([`MULTISPK_CORPUS`]) — ordered by speaker count. 10
/// (7 spk) is the clip that caught argmax. The selection and its "four of eight"
/// denominator are pinned by [`shipping_clip_selection_is_the_documented_subset`].
const MULTI_SPEAKER_CLIPS: &[MultiSpkClip] = &[
  MultiSpkClip {
    name: "06_long_recording",
    ref_spk: 3,
    samples: 15_643_627,
    audio_fnv: 6_813_989_898_382_736_122,
  },
  MultiSpkClip {
    name: "14_mrbeast_strongman_robot",
    ref_spk: 4,
    samples: 17_648_640,
    audio_fnv: 8_962_622_122_443_019_965,
  },
  MultiSpkClip {
    name: "10_mrbeast_clean_water",
    ref_spk: 7,
    samples: 9_911_979,
    audio_fnv: 3_229_612_773_310_046_830,
  },
  MultiSpkClip {
    name: "09_mrbeast_dollar_date",
    ref_spk: 8,
    samples: 16_671_744,
    audio_fnv: 8_657_240_795_675_234_981,
  },
];

/// The FULL ≥ 3-speaker membership of dia's parity corpus (eight clips), as
/// `(name, ref_spk)`, from which [`MULTI_SPEAKER_CLIPS`] selects four.
/// Documented here so the "four of eight" claim and the SELECTION are pinned
/// against silent corpus drift by
/// [`shipping_clip_selection_is_the_documented_subset`] (which re-derives the
/// counts from the RTTMs), not asserted in prose alone. This is a MEMBERSHIP
/// manifest, not a load manifest — only the selected four are decoded and
/// scored, so only they carry the audio content-identity pin (F4,
/// [`MultiSpkClip::audio_fnv`]); the corpus needs only name + speaker count.
/// The parallel `parity_e2e::FIXTURE_FACTS` independently pins all 14 clips.
const MULTISPK_CORPUS: &[(&str, usize)] = &[
  ("06_long_recording", 3),
  ("08_luyu_jinjing_freedom", 3),
  ("09_mrbeast_dollar_date", 8),
  ("10_mrbeast_clean_water", 7),
  ("11_mrbeast_age_race", 6),
  ("12_mrbeast_schools", 15),
  ("13_mrbeast_saved_animals", 11),
  ("14_mrbeast_strongman_robot", 4),
];

/// The [`MultiSpkClip`] row for `name`. Resolving a gated clip BY NAME kills the
/// positional-index coupling the wrappers used to carry (codex r7 F1):
/// `MULTI_SPEAKER_CLIPS[2]` says nothing about which clip index 2 is, so a table
/// reorder silently retargeted a gate to different audio.
///
/// # Panics
/// If `name` is not one of the gated clips.
fn clip_by_name(name: &str) -> &'static MultiSpkClip {
  MULTI_SPEAKER_CLIPS
    .iter()
    .find(|c| c.name == name)
    .unwrap_or_else(|| panic!("{name}: not in MULTI_SPEAKER_CLIPS"))
}

/// The reference speaker count [`MULTI_SPEAKER_CLIPS`] records for `name`,
/// evaluated in `const` context. The `shipping_der_gate!` count assertion uses it
/// to tie each wrapper's `@ <count>` to the table — and thus, via
/// [`shipping_clip_selection_is_the_documented_subset`], to the RTTM corpus — at
/// compile time.
///
/// # Panics
/// If `name` is not a gated clip: a wrapper for an un-selected clip is a build
/// error, never a silently-skipped gate.
const fn clip_ref_spk(name: &str) -> usize {
  let mut i = 0;
  while i < MULTI_SPEAKER_CLIPS.len() {
    if const_str_eq(MULTI_SPEAKER_CLIPS[i].name, name) {
      return MULTI_SPEAKER_CLIPS[i].ref_spk;
    }
    i += 1;
  }
  panic!("clip_ref_spk: name is not in MULTI_SPEAKER_CLIPS");
}

// ══════════════════════════════════════════════════════════════════════
// Fixture / model resolution
// ══════════════════════════════════════════════════════════════════════

/// dia's parity-fixture root (override with `DIA_PARITY_FIXTURES`) — same
/// convention as `parity_e2e.rs`.
fn fixtures_root() -> std::path::PathBuf {
  std::env::var_os("DIA_PARITY_FIXTURES").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../diarization/tests/parity/fixtures")
    },
    std::path::PathBuf::from,
  )
}

fn reference_rttm_path(name: &str) -> std::path::PathBuf {
  fixtures_root().join(name).join("reference.rttm")
}

fn clip_audio_path(name: &str) -> std::path::PathBuf {
  fixtures_root().join(name).join("clip_16k.wav")
}

/// dia's fp32 WeSpeaker ONNX (override with `DIA_EMBED_MODEL_PATH`) — same
/// convention as `parity_e2e.rs` / `generate_goldens.rs`.
fn dia_wespeaker_onnx() -> std::path::PathBuf {
  std::env::var_os("DIA_EMBED_MODEL_PATH").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../diarization/models/wespeaker_resnet34_lm.onnx")
    },
    std::path::PathBuf::from,
  )
}

// ══════════════════════════════════════════════════════════════════════
// Pipeline runners
// ══════════════════════════════════════════════════════════════════════

/// dia's in-crate community-1 PLDA — the REFERENCE (dia-ort) side's projection.
/// Its measured-side twin is [`load_plda_diaric`]; both `include_bytes!` the
/// same community-1 blobs, and the `plda_cross_crate_equivalence` gate asserts
/// their transforms are bit-identical, so the two crate-typed instances cannot
/// diverge on the projection. NB `new()` takes NO data: a frozen, pretrained
/// LDA+PLDA (see the module doc's rotation-invariance note).
fn load_plda() -> dia::plda::PldaTransform {
  dia::plda::PldaTransform::new().expect("load community-1 PldaTransform")
}

/// diaric's in-crate community-1 PLDA — the MEASURED (speakerkit) side's
/// projection, the one `Extraction::diarize` consumes. Bit-identical to
/// [`load_plda`]'s dia instance (asserted by `plda_cross_crate_equivalence`),
/// so measuring through diaric while the oracle runs through dia does not move
/// the projection.
fn load_plda_diaric() -> diaric::plda::PldaTransform {
  diaric::plda::PldaTransform::new().expect("load community-1 PldaTransform (diaric)")
}

/// Boundary adapter: diaric's `OfflineOutput` (the MEASURED clustering result)
/// → dia's `RttmSpan`s (the REFERENCE output type), so both sides funnel their
/// spans through the one dia-typed [`output_segs`] for the DER comparison.
///
/// Total at the span level: every diaric span's full public state
/// (`cluster`/`start`/`duration`) is mapped through dia's positional
/// `RttmSpan::new`, which fails to compile if that constructor's shape changes.
/// A whole-`OfflineOutput` → `OfflineOutput` map is impossible — dia's
/// `SpillBytes::from_vec` is `pub(crate)`, so `dia::offline::OfflineOutput`
/// cannot be constructed outside `dia` — and the DER comparison consumes only
/// spans, so the span vector is the complete DER-relevant output.
fn to_dia_spans(out: &diaric::offline::OfflineOutput) -> Vec<dia::reconstruct::RttmSpan> {
  out
    .spans_slice()
    .iter()
    .map(|s| dia::reconstruct::RttmSpan::new(s.cluster(), s.start(), s.duration()))
    .collect()
}

/// dia `RttmSpan`s → [`Seg`]s — the single DER span-extractor both the reference
/// (dia-ort) and the measured (speakerkit, via [`to_dia_spans`]) sides share.
fn output_segs(spans: &[dia::reconstruct::RttmSpan]) -> Vec<Seg> {
  spans
    .iter()
    .map(|s| Seg {
      start: s.start(),
      end: s.end(),
      spk: s.cluster(),
    })
    .collect()
}

/// dia's OWN ort path — the oracle. dia-ort seg (bundled `segmentation-3.0`) +
/// dia-ort embed (fp32 `wespeaker_resnet34_lm.onnx`) → the SAME
/// `diarize_offline` clustering.
struct DiaOrtRun {
  segs: Vec<Seg>,
  num_chunks: usize,
  num_output_frames: usize,
}

fn dia_ort_run(samples: &[f32], plda: &dia::plda::PldaTransform) -> DiaOrtRun {
  let mut seg = dia::segment::SegmentModel::bundled().expect("dia bundled segmentation-3.0");
  let onnx = dia_wespeaker_onnx();
  assert!(
    onnx.exists(),
    "dia WeSpeaker ONNX not found at {}; set DIA_EMBED_MODEL_PATH",
    onnx.display()
  );
  let mut embed = dia::embed::EmbedModel::from_file(&onnx).expect("dia WeSpeaker fp32 ONNX");
  let pipeline = dia::offline::OwnedDiarizationPipeline::new();
  let out = pipeline
    .run(&mut seg, &mut embed, plda, samples)
    .expect("dia OwnedDiarizationPipeline::run");
  let num_clusters = out.num_clusters();
  let num_chunks = out.hard_clusters_slice().len();
  let num_output_frames = out
    .discrete_diarization_slice()
    .len()
    .checked_div(num_clusters)
    .unwrap_or(0);
  DiaOrtRun {
    segs: output_segs(out.spans_slice()),
    num_chunks,
    num_output_frames,
  }
}

/// speakerkit's FluidAudio source over an explicitly-chosen embedder artifact
/// and per-model placements — the knobs this suite varies. Segmentation and
/// embedder placements are independent in production
/// ([`coremlit::audio::speaker::extract::ComputeOptions`] carries one
/// [`ComputeUnits`] per model), so every arm names both.
fn fluidaudio_extraction(
  samples: &[f32],
  embed_path: &Path,
  seg_cu: ComputeUnits,
  emb_cu: ComputeUnits,
) -> Extraction {
  let seg = SegmentModel::from_file_with(
    common::seg_path(),
    SegmentModelOptions::new().with_compute(seg_cu),
  )
  .expect("load pyannote_segmentation.mlmodelc");
  let embed = EmbedModel::from_file_with(embed_path, EmbedModelOptions::new().with_compute(emb_cu))
    .expect("load wespeaker embedder");
  FluidAudioSource::with_options(seg, embed, Options::new())
    .extract(samples)
    .expect("FluidAudioSource::extract")
}

/// Run the public `Extraction::diarize` clustering path on an `Extraction` +
/// the measured-side (diaric) PLDA → its spans (one code path with the runtime
/// API), converting diaric's output to dia spans at the boundary via
/// [`to_dia_spans`].
///
/// Returns diaric's TYPED error rather than unwrapping or stringifying it:
/// diaric's clustering can REFUSE to produce an answer (e.g.
/// `Pipeline(Centroid(AmbiguousAliveCluster { .. }))`, its deliberate bail-out
/// when a cluster's alive-value lands in the SIMD guard band around the
/// threshold). Whether a given arm hits that is itself a first-class
/// measurement — an arm that errors where the oracle clusters means that
/// configuration cannot diarize the audio AT ALL, which is a far worse defect
/// than any DER. Unwrapping would report it as an opaque harness panic; and
/// keeping the TYPED [`diaric::offline::Error`] (not a `String`) is what lets
/// [`assert_clip09_record`] match the exact `AmbiguousAliveCluster` variant,
/// not merely assert `is_err`.
fn diarize_extraction_segs(
  ext: &Extraction,
  plda: &diaric::plda::PldaTransform,
) -> Result<Vec<Seg>, diaric::offline::Error> {
  ext
    .diarize(plda)
    .map(|out| output_segs(&to_dia_spans(&out)))
}

/// One measured speakerkit arm. `segs` is `Err` when dia's clustering refused
/// to produce an answer for this arm's embeddings (see
/// [`diarize_extraction_segs`]).
struct Arm {
  tag: &'static str,
  segs: Result<Vec<Seg>, diaric::offline::Error>,
  spk: Option<usize>,
  extract_s: f64,
}

impl Arm {
  /// The arm's spans, or a panic naming the arm — call only after the
  /// clustering-outcome gate has proven every arm clustered.
  fn segs(&self) -> &[Seg] {
    match &self.segs {
      Ok(s) => s,
      Err(e) => panic!("{}: clustering failed: {e}", self.tag),
    }
  }

  /// The arm's speaker count, or a panic naming the arm.
  fn spk(&self) -> usize {
    self
      .spk
      .unwrap_or_else(|| panic!("{}: no speaker count (clustering failed)", self.tag))
  }

  /// The arm's speaker count for the report, or `ERR` if it could not cluster —
  /// so the report line prints for every arm, including a failed one.
  fn spk_str(&self) -> String {
    self
      .spk
      .map_or_else(|| "ERR".to_string(), |n| n.to_string())
  }
}

/// Everything an arm must hold CONSTANT: the one audio buffer, its
/// fingerprint, the measured-side (diaric) PLDA, the oracle's grid geometry,
/// and the clip name. Bundled so the only things an arm varies are the embedder
/// artifact and the compute placement — which is exactly the experiment.
struct ClipCtx<'a> {
  clip: &'a str,
  samples: &'a [f32],
  audio_fnv: u64,
  plda: &'a diaric::plda::PldaTransform,
  dia: &'a DiaOrtRun,
}

/// Runs one arm end-to-end and re-proves it consumed the untouched buffer on
/// the untouched grid.
fn run_arm(
  ctx: &ClipCtx<'_>,
  tag: &'static str,
  embed_path: &Path,
  seg_cu: ComputeUnits,
  emb_cu: ComputeUnits,
) -> Arm {
  let ClipCtx {
    clip,
    samples,
    audio_fnv,
    plda,
    dia,
  } = *ctx;

  let t0 = Instant::now();
  let ext = fluidaudio_extraction(samples, embed_path, seg_cu, emb_cu);
  let extract_s = t0.elapsed().as_secs_f64();

  // ── INPUT-IDENTITY PROOF. The buffer every arm consumed must still be the
  // buffer it started as: a divergence caused by a different input is a
  // harness bug, not a finding (the alignkit fake-86 % lesson).
  assert_eq!(
    common::fnv1a_f32(samples),
    audio_fnv,
    "{clip}/{tag}: the audio buffer changed under the arm — comparison invalid"
  );
  // ── FRAMING PROOF. Same sliding-window grid as dia-ort's own pipeline, so
  // no DER is scored across a misaligned framing.
  assert_eq!(
    ext.num_chunks(),
    dia.num_chunks,
    "{clip}/{tag}: grid num_chunks mismatch (speakerkit {} vs dia-ort {}) — framing diverged",
    ext.num_chunks(),
    dia.num_chunks
  );
  assert_eq!(
    ext.num_output_frames(),
    dia.num_output_frames,
    "{clip}/{tag}: grid num_output_frames mismatch (speakerkit {} vs dia-ort {}) — framing diverged",
    ext.num_output_frames(),
    dia.num_output_frames
  );

  let segs = diarize_extraction_segs(&ext, plda);
  let spk = segs.as_ref().ok().map(|s| distinct_speakers(s).len());
  match &segs {
    Ok(_) => println!(
      "[{clip}] {tag}: clustered OK ({} speakers)",
      spk.unwrap_or(0)
    ),
    Err(e) => println!("[{clip}] {tag}: CLUSTERING FAILED — {e}"),
  }
  Arm {
    tag,
    segs,
    spk,
    extract_s,
  }
}

// ══════════════════════════════════════════════════════════════════════
// The measurement + gate
// ══════════════════════════════════════════════════════════════════════

/// Everything one clip's four arms produced. Returned by [`measure`] so the
/// gate ([`gate`]) is a separate, purely-asserting step: the full report is
/// printed BEFORE any assertion fires, so a gate failure never hides the
/// numbers that explain it (the clip-09 lesson — its first run panicked inside
/// an arm and reported nothing at all).
struct Measurement {
  clip: &'static str,
  ref_spk: usize,
  /// dia-ort's speaker count — the oracle decision every arm is held to.
  /// (dia-ort's spans themselves are not carried: every DER involving them is
  /// computed and printed inside [`measure`], and the gate needs only the
  /// count.)
  dia_spk: usize,
  reference: Vec<Seg>,
  /// `seg@All + fp32@All` — the literal shipping default, through the
  /// production resolver.
  shipping: Arm,
  /// `seg@All + fp32@CpuOnly` — the EMBEDDER-placement control: identical
  /// segmentation tensors, embedder moved to IEEE CPU kernels.
  emb_cpu: Arm,
  /// `seg@CpuOnly + fp32@CpuOnly` — the all-CPU deterministic fallback, the
  /// configuration `parity_e2e`'s Part A gates on its own corpus.
  all_cpu: Arm,
}

/// Measures one clip across the oracle + three fp32 arms and prints the full
/// report. Asserts only the things that make the measurement *meaningful at
/// all* (audio identity, grid identity, reference speaker count); the product
/// gate is [`gate`].
///
/// Split per-clip (rather than one loop over the clip table) because these are
/// 10-24 minute recordings: each clip is ~4 full pipeline passes, so per-clip
/// tests keep any single invocation tractable and let a failure name the clip
/// that broke.
fn measure(clip: &MultiSpkClip) -> Measurement {
  let audio = clip_audio_path(clip.name);
  assert!(
    audio.exists(),
    "clip audio not found at {} (set DIA_PARITY_FIXTURES)",
    audio.display()
  );
  assert!(
    common::embed_fp32_path().exists(),
    "need wespeaker.mlmodelc (fp32, shipping) under {} (set SPEAKERKIT_TEST_MODELS)",
    common::models_dir().display()
  );

  // dia's PLDA drives the dia-ort oracle; diaric's drives the measured
  // speakerkit arms. The two are bit-identical (asserted by
  // `plda_cross_crate_equivalence`), so the split does not move the projection.
  let plda = load_plda();
  let plda_dc = load_plda_diaric();

  // ── ONE audio buffer. Every arm gets this exact slice; its fingerprint is
  // re-asserted after each arm.
  let samples = common::load_wav_16k_mono(&audio);
  let audio_fnv = common::fnv1a_f32(&samples);

  // ── CONTENT-IDENTITY PIN (F4). The clip's identity is its DECODED AUDIO, not
  // its directory name: a same-name swap (a re-encode, a truncation, a wrong
  // file dropped in) changes the sample count or the FNV and fails HERE, before
  // any DER is scored. The downstream clip-09 defect pin asserts only
  // `o.clip == "09..."` (the manifest string); this is the check that watches
  // the bytes. Mutation-proven by `clip09_content_pin_catches_an_audio_swap`.
  assert_eq!(
    samples.len(),
    clip.samples,
    "{}: decoded {} samples, pinned {} — the audio identity changed (a same-name swap?)",
    clip.name,
    samples.len(),
    clip.samples
  );
  assert_eq!(
    audio_fnv, clip.audio_fnv,
    "{}: audio content hash {audio_fnv} != pinned {} — a same-length content change the \
     sample count alone would miss",
    clip.name, clip.audio_fnv
  );

  let reference = parse_rttm(&reference_rttm_path(clip.name));
  let ref_spk = distinct_speakers(&reference).len();

  println!(
    "\n╔══ [{}] {:.2} s, {} samples, fnv1a={} ══",
    clip.name,
    samples.len() as f64 / 16_000.0,
    samples.len(),
    common::fnv_hex(audio_fnv)
  );
  // Pin the corpus: if the fixture's reference ever loses its multi-speaker
  // character, this suite must fail rather than silently become a no-op.
  assert_eq!(
    ref_spk, clip.ref_spk,
    "{}: reference.rttm has {ref_spk} speakers, expected {} — the multi-speaker \
     coverage this suite depends on changed",
    clip.name, clip.ref_spk
  );

  // ── The oracle.
  let t0 = Instant::now();
  let dia = dia_ort_run(&samples, &plda);
  let dia_s = t0.elapsed().as_secs_f64();
  assert_eq!(
    common::fnv1a_f32(&samples),
    audio_fnv,
    "{}: dia-ort mutated the audio buffer — comparison invalid",
    clip.name
  );
  let dia_spk = distinct_speakers(&dia.segs).len();

  // ── The three speakerkit arms, all on the SAME buffer, the SAME (measured,
  // diaric) PLDA, the SAME shipping artifact and the SAME grid. Only the two
  // model placements vary.
  let ctx = ClipCtx {
    clip: clip.name,
    samples: &samples,
    audio_fnv,
    plda: &plda_dc,
    dia: &dia,
  };
  // The shipping embedder path comes from the SAME resolver production uses
  // (`AnySource::load`'s FluidAudio arm), so the shipping arm IS the shipping
  // selection by construction — not a second copy of the path via
  // `common::embed_fp32_path()` that could drift from production (finding 3).
  // The control arms move ONE placement at a time off that default.
  let artifacts = FluidAudioArtifacts::resolve(common::models_dir());
  let shipping = run_arm(
    &ctx,
    "sAll+eAll",
    artifacts.embedder(),
    ComputeUnits::All,
    ComputeUnits::All,
  );
  let emb_cpu = run_arm(
    &ctx,
    "sAll+eCpu",
    artifacts.embedder(),
    ComputeUnits::All,
    ComputeUnits::CpuOnly,
  );
  let all_cpu = run_arm(
    &ctx,
    "sCpu+eCpu",
    artifacts.embedder(),
    ComputeUnits::CpuOnly,
    ComputeUnits::CpuOnly,
  );

  // ══ REPORT (unconditional — printed BEFORE any assertion) ══
  //
  // An arm whose clustering FAILED has no spans, so it contributes its error
  // instead of a DER row. Everything that can be computed, is — a gate failure
  // must never hide the numbers that explain it.
  println!(
    "[{}] speaker counts: reference={ref_spk} dia-ort={dia_spk} {}={} {}={} {}={}",
    clip.name,
    shipping.tag,
    shipping.spk_str(),
    emb_cpu.tag,
    emb_cpu.spk_str(),
    all_cpu.tag,
    all_cpu.spk_str(),
  );
  println!(
    "[{}] extract wall-clock (CONTENDED when clips run in parallel — NOT a latency \
     measurement; see shipping_embedder_cost_int8_vs_fp32): dia-ort={dia_s:.1}s {}={:.1}s \
     {}={:.1}s {}={:.1}s",
    clip.name,
    shipping.tag,
    shipping.extract_s,
    emb_cpu.tag,
    emb_cpu.extract_s,
    all_cpu.tag,
    all_cpu.extract_s,
  );

  println!(
    "[{}] {}",
    clip.name,
    fmt_der("ABS dia-ort      std   ", &der_std(&reference, &dia.segs))
  );
  for arm in [&shipping, &emb_cpu, &all_cpu] {
    match &arm.segs {
      Ok(segs) => {
        println!(
          "[{}] {}",
          clip.name,
          fmt_der(
            &format!("ABS {:<12} std   ", arm.tag),
            &der_std(&reference, segs)
          )
        );
        println!(
          "[{}] {}",
          clip.name,
          fmt_der(
            &format!("ABS {:<12} strict", arm.tag),
            &der_strict(&reference, segs)
          )
        );
      }
      Err(e) => println!("[{}] ABS {:<12} — NO SPANS: {e}", clip.name, arm.tag),
    }
  }

  // The placement axes, ISOLATED: the embedder placement (shipping vs emb_cpu:
  // identical segmentation tensors, embedder All -> CpuOnly) and the
  // segmentation placement on top of a CPU embedder (emb_cpu vs all_cpu). Same
  // audio, same artifact, same clustering — ONLY a placement differs per pair.
  for (tag, pair) in [
    (
      "EMB-PLACE  sAll+eAll vs sAll+eCpu std",
      (&shipping, &emb_cpu),
    ),
    (
      "SEG-PLACE  sAll+eCpu vs sCpu+eCpu std",
      (&emb_cpu, &all_cpu),
    ),
  ] {
    let (r, h) = pair;
    if let (Ok(rs), Ok(hs)) = (&r.segs, &h.segs) {
      println!("[{}] {}", clip.name, fmt_der(tag, &der_std(rs, hs)));
      println!(
        "[{}] {}",
        clip.name,
        fmt_der(&format!("{tag} (strict)"), &der_strict(rs, hs))
      );
    } else {
      println!(
        "[{}] {tag} — not computable (an arm failed to cluster)",
        clip.name
      );
    }
  }
  for (tag, arm) in [
    ("SHIPPING   sAll+eAll vs dia-ort std", &shipping),
    ("CONTROL    sAll+eCpu vs dia-ort std", &emb_cpu),
    ("ALL-CPU    sCpu+eCpu vs dia-ort std", &all_cpu),
  ] {
    match &arm.segs {
      Ok(s) => println!("[{}] {}", clip.name, fmt_der(tag, &der_std(&dia.segs, s))),
      Err(_) => println!(
        "[{}] {tag} — not computable (arm failed to cluster)",
        clip.name
      ),
    }
  }

  // The one-line verdict: what the shipping placement costs against the
  // CPU-embedder control, vs the independent reference. Clip 09 is EXCLUDED
  // from the two bounds printed here (its control arms sit on the pinned
  // segmentation knife edge, so cross-count deltas are not placement noise —
  // see `assert_clip09_record`); its line says so instead of printing bounds
  // it is not held to, so a value above a bound next to a green result cannot
  // be misread as a gate that failed to fire.
  if let Ok(f) = &emb_cpu.segs {
    let abs_control = der_std(&reference, f).der;
    let d = |a: &Arm| {
      a.segs
        .as_ref()
        .map_or(f64::NAN, |s| der_std(&reference, s).der - abs_control)
    };
    let conf = |a: &Arm| {
      a.segs
        .as_ref()
        .map_or(f64::NAN, |s| der_std(f, s).confusion)
    };
    let bounds = if clip.name == "09_mrbeast_dollar_date" {
      "[clip 09: informational only — EXCLUDED from the placement gates; governed by its own \
       pinned record]"
        .to_string()
    } else {
      format!(
        "[GATE: ±{:.4}%]  ||  tripwire {:.4}%",
        SHIPPING_ABS_DELTA_MAX * 100.0,
        SHIPPING_CONFUSION_TRIPWIRE * 100.0
      )
    };
    println!(
      "[{}] ΔDER(sAll+eAll − sAll+eCpu) vs pyannote = {:+.4}%  ||  shipping-vs-control \
       CONFUSION = {:.4}%  {bounds}",
      clip.name,
      d(&shipping) * 100.0,
      conf(&shipping) * 100.0,
    );
  } else {
    println!(
      "[{}] ΔDER not computable — the CPU-embedder CONTROL failed to cluster (see the clip-09 \
       record if this is clip 09)",
      clip.name
    );
  }

  Measurement {
    clip: clip.name,
    ref_spk,
    dia_spk,
    reference,
    shipping,
    emb_cpu,
    all_cpu,
  }
}

// ══════════════════════════════════════════════════════════════════════
// The gate (pure assertions over a completed Measurement)
// ══════════════════════════════════════════════════════════════════════

/// The product gate for a clip on which every placement is expected to hold
/// the oracle's clustering decision.
///
/// Asserts, in order:
/// - **G0** every arm clustered at all (dia-ort did, so a speakerkit-arm failure
///   is a CoreML-path defect, not a dia limitation on this audio);
/// - **G1** the speaker-count decision is identical to dia-ort's across the
///   shipping configuration and both placement controls — the metric argmax
///   violated (7→8) and the metric the retired int8 embedder violated on
///   8-speaker audio (8→5);
/// - **G2** the shipping placement's agreement with the independent pyannote
///   reference is within [`SHIPPING_ABS_DELTA_MAX`] of the CPU-embedder
///   control's (and likewise the all-CPU fallback's);
/// - **G3** the arm-vs-control confusion stays under
///   [`SHIPPING_CONFUSION_TRIPWIRE`] (gross-regression guard only — read that
///   constant's doc for why it is a tripwire and not a tight bound).
fn gate(m: &Measurement) {
  let clip = m.clip;

  // ── G0: every arm produced an answer.
  let failed: Vec<String> = [&m.shipping, &m.emb_cpu, &m.all_cpu]
    .into_iter()
    .filter_map(|a| a.segs.as_ref().err().map(|e| format!("{} → {e}", a.tag)))
    .collect();
  assert!(
    failed.is_empty(),
    "{clip}: dia-ort clustered this clip ({} speakers), but {} speakerkit arm(s) could NOT: \
     {}. A CoreML pipeline whose embeddings dia cannot cluster is a HARD product failure — \
     the pipeline returns Err on real {}-speaker audio. Do NOT paper over this.",
    m.dia_spk,
    failed.len(),
    failed.join("; "),
    m.ref_spk,
  );

  let (shipping, emb_cpu, all_cpu) = (m.shipping.segs(), m.emb_cpu.segs(), m.all_cpu.segs());
  let (ship_spk, emb_cpu_spk, all_cpu_spk) = (m.shipping.spk(), m.emb_cpu.spk(), m.all_cpu.spk());

  // ── G1 (THE DECISION METRIC). No placement may change how many speakers the
  // pipeline finds. Exact equality against the ORACLE's count, no tolerance: a
  // speaker-count flip is never boundary jitter. This assertion, on THESE
  // clips, is the one that would have caught argmax (it invented a spurious
  // 8th speaker on the 7-speaker clip while the pre-rework Part B only
  // `println!`ed the count and exited 0).
  assert_eq!(
    ship_spk, m.dia_spk,
    "{clip}: the SHIPPING configuration (seg@All + fp32@All) disagrees with dia-ort on speaker \
     count ({ship_spk} vs {}) — a product defect in the configuration we actually ship.",
    m.dia_spk
  );
  assert_eq!(
    emb_cpu_spk, m.dia_spk,
    "{clip}: the CPU-embedder control (seg@All + fp32@CpuOnly) disagrees with dia-ort on speaker \
     count ({emb_cpu_spk} vs {}) — the embedder-placement axis changed a clustering decision.",
    m.dia_spk
  );
  assert_eq!(
    all_cpu_spk, m.dia_spk,
    "{clip}: the all-CPU fallback (seg@CpuOnly + fp32@CpuOnly) disagrees with dia-ort on speaker \
     count ({all_cpu_spk} vs {}) — the deterministic fallback configuration is broken.",
    m.dia_spk
  );

  // ── G2 (REFERENCE AGREEMENT, the tight bound). The shipping placement may
  // not cost measurable agreement with the independent reference relative to
  // the CPU-embedder control. A bound against the independent reference, not
  // bit-agreement with another of our own artifacts. argmax cost +3.33 points
  // here and would blow straight through it. Never loosened.
  let abs_control = der_std(&m.reference, emb_cpu).der;
  for (tag, hyp) in [("sAll+eAll", shipping), ("sCpu+eCpu", all_cpu)] {
    let delta = der_std(&m.reference, hyp).der - abs_control;
    assert!(
      delta.abs() <= SHIPPING_ABS_DELTA_MAX,
      "{clip}: ΔDER({tag} − sAll+eCpu) vs pyannote = {:+.4}%, over the ±{:.4}% bound — a \
       placement measurably degrades agreement with the reference. Do NOT loosen.",
      delta * 100.0,
      SHIPPING_ABS_DELTA_MAX * 100.0
    );
  }

  // ── G3 (GROSS CLUSTERING REGRESSION). Tripwire only — see the constant's doc.
  for (tag, hyp) in [("sAll+eAll", shipping), ("sCpu+eCpu", all_cpu)] {
    let conf = der_std(emb_cpu, hyp).confusion;
    assert!(
      conf <= SHIPPING_CONFUSION_TRIPWIRE,
      "{clip}: {tag}-vs-control DER confusion {:.4}% exceeds the gross-regression tripwire \
       {:.4}% — far past marginal-assignment drift, indicating a placement is genuinely breaking \
       clustering. Investigate; do NOT raise the tripwire to pass.",
      conf * 100.0,
      SHIPPING_CONFUSION_TRIPWIRE * 100.0
    );
  }
}

/// Declares one shipping DER gate, binding the wrapper's NAME to the clip it
/// LOADS (codex r7 F1). Two compile-time assertions gate each wrapper:
/// [`const_str_eq`] proves the function name is exactly
/// `shipping_der_<fixture>_<count>spk[suffix]`, and [`clip_ref_spk`] proves
/// `<count>` equals the fixture's speaker count in [`MULTI_SPEAKER_CLIPS`] (whose
/// membership and counts are pinned to the RTTM corpus by
/// [`shipping_clip_selection_is_the_documented_subset`]). The body resolves the
/// clip BY NAME via [`clip_by_name`] — the fixture literal is the ONLY place the
/// clip appears — so neither a positional-index reorder nor a name/fixture
/// mismatch can slip through as a green gate over the wrong audio.
///
/// A bare row `name : "fixture" @ count` gets the default `gate(&measure(…))`
/// body. `=> |m| { … }` supplies a custom body with the [`Measurement`] bound to
/// `m`; `+ "suffix"` extends the checked name.
macro_rules! shipping_der_gate {
  ( $(#[$meta:meta])* $name:ident : $fixture:literal @ $count:literal ) => {
    shipping_der_gate! {
      $(#[$meta])* $name : $fixture @ $count + "" => |m| { gate(&m); }
    }
  };
  ( $(#[$meta:meta])* $name:ident : $fixture:literal @ $count:literal
    $(+ $suffix:literal)? => |$m:ident| $body:block ) => {
    const _: () = assert!(
      const_str_eq(
        stringify!($name),
        concat!("shipping_der_", $fixture, "_", $count, "spk" $(, $suffix)?),
      ),
      concat!(
        "shipping gate `", stringify!($name), "` disagrees with its fixture/count `",
        $fixture, "` @ ", stringify!($count),
        " — a wrapper name must be shipping_der_<fixture>_<count>spk",
      ),
    );
    const _: () = assert!(
      clip_ref_spk($fixture) == $count,
      concat!(
        "shipping gate `", stringify!($name), "` encodes ", stringify!($count),
        " speakers but MULTI_SPEAKER_CLIPS records a different count for `", $fixture, "`",
      ),
    );
    $(#[$meta])*
    #[test]
    #[ignore = "requires Models/speakerkit + sibling diarization ONNX/fixtures + ort"]
    fn $name() {
      let $m = measure(clip_by_name($fixture));
      $body
    }
  };
}

shipping_der_gate! {
  /// 3 speakers, 977.7 s.
  shipping_der_06_long_recording_3spk : "06_long_recording" @ 3
}

shipping_der_gate! {
  /// 4 speakers, 1103.0 s. The clip where clustering sits nearest a decision
  /// boundary: the fp32 CoreML pipeline already carries ~0.39 % confusion
  /// against dia-ort here with no placement or precision perturbation
  /// involved, so any placement drift shows up here first — see
  /// [`SHIPPING_CONFUSION_TRIPWIRE`].
  shipping_der_14_mrbeast_strongman_robot_4spk : "14_mrbeast_strongman_robot" @ 4
}

shipping_der_gate! {
  /// 7 speakers, 619.5 s — **the clip that caught the argmax source** (spurious
  /// 8th speaker, 3.33 % DER, 100 % confusion). [`gate`]'s G1 count equality
  /// on this clip is the assertion that failure class cannot pass.
  shipping_der_10_mrbeast_clean_water_7spk : "10_mrbeast_clean_water" @ 7
}

// ══════════════════════════════════════════════════════════════════════
// Clip 09 — the 8-speaker clip: gated SHIPPING arm, pinned placement knife
// edge
// ══════════════════════════════════════════════════════════════════════

/// Everything clip 09's record must satisfy, extracted from a [`Measurement`]
/// (or synthesized by the mutation test) so [`assert_clip09_record`] is a pure
/// function both can call.
#[derive(Clone, Copy)]
struct Clip09Observed<'a> {
  /// The fixture this was measured on — asserted, so the pin cannot silently
  /// re-target a different clip.
  clip: &'a str,
  /// Distinct speakers in `reference.rttm`.
  ref_spk: usize,
  /// dia-ort's (the ONNX oracle's) speaker count.
  dia_spk: usize,
  /// The SHIPPING arm's (seg@All + fp32@All) speaker count.
  ship_spk: usize,
  /// The SHIPPING arm's standard DER vs the pyannote reference.
  ship_der: Der,
  /// The CPU-embedder control's (seg@All + fp32@CpuOnly) speaker count.
  emb_cpu_spk: usize,
  /// The CPU-embedder control's standard DER vs the pyannote reference.
  emb_cpu_der: Der,
  /// The all-CPU arm's (seg@CpuOnly + fp32@CpuOnly) clustering outcome,
  /// carrying diaric's TYPED error so the pin can match the exact
  /// `AmbiguousAliveCluster` variant, not just `is_err`.
  all_cpu: &'a Result<Vec<Seg>, diaric::offline::Error>,
}

/// The measured clip-09 record, pinned field by field so a single-value change
/// in EITHER direction fails. Every field is exercised in both directions,
/// hermetically, by [`clip09_record_pins_every_field`].
///
/// Measured (issue #15 remedy matrix + this suite, Apple M1 Max, macOS 26.5
/// build 25F71, arm64; fp16-safe `pyannote_segmentation` + fp16-safe fp32
/// `wespeaker`, `Models/speakerkit` at the byte-pinned revisions):
///
/// - reference (pyannote 4.0.4) and `dia-ort` (the ONNX oracle): **8**
///   speakers each; dia-ort is frame-perfect against the reference.
/// - **SHIPPING (seg@All + fp32@All): 8 of 8 speakers, 2.9810 % DER** —
///   the count is RIGHT and the residual is confusion against the oracle's
///   assignment, not a lost speaker. (The retired int8 embedder returned 5
///   of 8 at 16.5904 % on this exact composition; the swap is the issue-#15
///   fix.)
/// - **seg@All + fp32@CpuOnly: 9 speakers, 1.3011 %** — with the embedder on
///   IEEE CPU kernels (bit-near dia's own ONNX), a spurious 9th cluster
///   survives clustering. This is the segmentation conversion's defect class:
///   `backend_factorial`'s `COREML-seg + ONNX-emb` cell overcounts the same
///   way (9 speakers, 1.3011 %). Lower DER, wrong count — and the count
///   decision outranks the DER value.
/// - **seg@CpuOnly + fp32@CpuOnly: `Err(Pipeline(Centroid(
///   AmbiguousAliveCluster)))`** — a cluster's VBx prior (recorded
///   1.700e-7 on this host) lands inside diaric's deliberate ±2x guard band
///   around the 1e-7 alive-cutoff, and diaric refuses to make a
///   CPU-backend-dependent call. The all-CPU fallback cannot diarize this
///   clip at all.
///
/// The three placement outcomes {8, 9, Err} are pinned SEPARATELY — each is
/// its own observed fact. Reading them as one spurious near-threshold
/// cluster whose alive/ambiguous/absorbed state flips with placement is the
/// INTERPRETATION the pattern supports (one extra cluster when clustering
/// resolves, an alive-band refusal when it cannot), but no assertion here
/// ties the three outcomes to the same cluster — VBx cluster identities are
/// not stable across arms, so the `Err` pin deliberately matches the VARIANT
/// and not its `cluster`/`value` fields. What the pins enforce: the
/// segmentation conversion's residual clip-09 defect — real, an order of
/// magnitude smaller than the fixed embedder collapse — cannot be forgotten
/// and cannot silently change shape in any arm.
fn assert_clip09_record(o: &Clip09Observed<'_>) {
  // Identity: the pin is clip-09-specific, so a fixture-resolution drift onto
  // another clip must fail loudly, never silently re-target.
  assert_eq!(
    o.clip, "09_mrbeast_dollar_date",
    "clip-09 record ran on {} — the fixture selection drifted",
    o.clip
  );
  // Reference AND oracle both see 8 speakers, so the arms below are held to a
  // real count, not a reference artifact.
  assert_eq!(
    o.ref_spk, 8,
    "09: reference.rttm no longer holds 8 speakers ({})",
    o.ref_spk
  );
  assert_eq!(
    o.dia_spk, 8,
    "09: dia-ort (the ONNX oracle) no longer clusters this clip at 8 speakers ({}) — the oracle \
     moved; re-establish it before trusting this record",
    o.dia_spk
  );
  // THE GATE: the shipping configuration finds all 8 speakers. This is the
  // issue-#15 fix; a count below 8 is the collapse returning, a count above
  // is the segmentation knife edge escaping into the shipping placement.
  assert_eq!(
    o.ship_spk, 8,
    "09: the SHIPPING configuration (seg@All + fp32@All) no longer finds 8 of 8 speakers ({}). \
     Below 8 is the issue-#15 collapse class returning; above 8 is a spurious cluster surfacing \
     at the shipping placement (the overcount class the segmentation conversion produces — \
     backend_factorial's COREML-seg + ONNX-emb cell). Either way: investigate, do not re-pin \
     without attribution.",
    o.ship_spk
  );
  assert_clip09_der_decomposed("sAll+eAll (SHIPPING)", o.ship_der, 0.029_810);
  // The CPU-embedder control: a spurious 9th cluster survives (the
  // segmentation conversion's overcount class). If this becomes 8, the
  // seg-side knife edge is gone and clip 09 belongs in the plain `gate(..)`
  // set; if it becomes an Err, the ambiguity refusal moved into this
  // placement. Both are deliberate re-baselines.
  assert_eq!(
    o.emb_cpu_spk, 9,
    "09: seg@All + fp32@CpuOnly moved from the pinned 9 speakers to {} — this arm's spurious \
     cluster changed state; re-measure all three placements and re-pin deliberately",
    o.emb_cpu_spk
  );
  assert_clip09_der_value("sAll+eCpu", o.emb_cpu_der, 0.013_011);
  // The all-CPU arm: diaric's typed refusal. Matching the VARIANT (not just
  // `is_err`, and not the unstable `cluster`/`value` fields) is what makes
  // "this arm stopped refusing" the only way this can go green — and is why
  // the record's doc presents the one-cluster reading as interpretation, not
  // an asserted identity across arms.
  assert!(
    matches!(
      o.all_cpu,
      Err(diaric::offline::Error::Pipeline(
        diaric::pipeline::Error::Centroid(
          diaric::cluster::centroid::Error::AmbiguousAliveCluster { .. }
        )
      ))
    ),
    "09: seg@CpuOnly + fp32@CpuOnly no longer fails with \
     Pipeline(Centroid(AmbiguousAliveCluster)). Either it now CLUSTERS (the segmentation knife \
     edge moved — re-measure and re-pin all three placements) or it fails a DIFFERENT way (a new \
     defect — investigate). Got: {:?}",
    o.all_cpu.as_ref().err()
  );
}

/// A clip-09 arm's DER, pinned two-sided (±[`DER_PIN_TOL`]) with its full
/// miss/FA/confusion decomposition: all the error is confusion.
fn assert_clip09_der_decomposed(tag: &str, d: Der, pinned: f64) {
  assert_clip09_der_value(tag, d, pinned);
  assert_eq!(
    d.miss_units, 0,
    "09 {tag}: DER decomposition changed — {} miss units, pinned at 0 (undercounting a \
     single-speaker-per-scored-frame reference is pure confusion, never miss)",
    d.miss_units
  );
  assert_eq!(
    d.fa_units, 0,
    "09 {tag}: DER decomposition changed — {} false-alarm units, pinned at 0",
    d.fa_units
  );
  assert!(
    (d.confusion - pinned).abs() <= DER_PIN_TOL,
    "09 {tag}: confusion {:.4}% moved from the pinned {:.4}% (±{:.4}%) — with miss/FA pinned at 0 \
     the confusion IS the DER; the clustering divergence changed character",
    d.confusion * 100.0,
    pinned * 100.0,
    DER_PIN_TOL * 100.0
  );
}

/// A clip-09 arm's standard DER, pinned two-sided (±[`DER_PIN_TOL`]).
fn assert_clip09_der_value(tag: &str, d: Der, pinned: f64) {
  assert!(
    (d.der - pinned).abs() <= DER_PIN_TOL,
    "09 {tag}: standard DER {:.4}% moved from the pinned {:.4}% (±{:.4}%). Worse is a regression; \
     better means the CoreML path changed — re-baseline deliberately, do NOT widen the band.",
    d.der * 100.0,
    pinned * 100.0,
    DER_PIN_TOL * 100.0
  );
}

shipping_der_gate! {
  /// **Clip 09 (8 speakers, 1042.0 s) — the issue-#15 clip: the SHIPPING arm is
  /// gated at 8 of 8 speakers; the two placement controls are pinned at their
  /// measured knife-edge states.**
  ///
  /// This clip cannot run the plain [`gate`] body, because its two control
  /// arms do not hold the oracle's count — a real, pinned property of the
  /// segmentation conversion on this clip, not a suite defect: a spurious
  /// cluster lands ALIVE (9 speakers) with the embedder on CPU kernels, and
  /// diaric refuses inside its ambiguity band (`Err`) with the whole
  /// pipeline on CPU — two separately-pinned outcomes.
  /// [`assert_clip09_record`] carries the full measured record and what may
  /// be read as shared cause versus what is pinned; every field's
  /// both-directions sensitivity is proven hermetically by
  /// [`clip09_record_pins_every_field`].
  ///
  /// History: the shipping arm of this clip returned 5 of 8 speakers at
  /// 16.5904 % DER (100 % confusion) until the int8-palettized embedder was
  /// retired — the collapse `backend_factorial.rs` attributes to the
  /// palettization's coherent embedding-space displacement, with the `All`
  /// placement contributing one further lost speaker. The mechanism and the
  /// factorial are pinned there; the DECISION record is
  /// `tests/speaker/model_io.rs`.
  shipping_der_09_mrbeast_dollar_date_8spk
    : "09_mrbeast_dollar_date" @ 8 => |m| {
    // The shipping and CPU-embedder arms must ANSWER to score their DER; a
    // failure is itself a regression (the configuration went from "answers"
    // to "cannot answer"), reported before any pin so the numbers still print.
    let (shipping, emb_cpu) = match (&m.shipping.segs, &m.emb_cpu.segs) {
      (Ok(_), Ok(_)) => (m.shipping.segs(), m.emb_cpu.segs()),
      _ => panic!(
        "09: an fp32 arm now fails to cluster (sAll+eAll={:?}, sAll+eCpu={:?}) — a regression \
         from 'answers' to 'cannot answer' on 8-speaker audio.",
        m.shipping.segs.as_ref().err(),
        m.emb_cpu.segs.as_ref().err(),
      ),
    };

    assert_clip09_record(&Clip09Observed {
      clip: m.clip,
      ref_spk: m.ref_spk,
      dia_spk: m.dia_spk,
      ship_spk: m.shipping.spk(),
      ship_der: der_std(&m.reference, shipping),
      emb_cpu_spk: m.emb_cpu.spk(),
      emb_cpu_der: der_std(&m.reference, emb_cpu),
      all_cpu: &m.all_cpu.segs,
    });
  }
}

/// [`assert_clip09_record`] pins EVERY field, in BOTH directions — proven
/// here hermetically (no models, no fixtures): the measured record passes, and
/// every single-field perturbation fails. Without this a field could silently
/// go unpinned, and a real clip-09 change in it would pass green.
#[test]
fn clip09_record_pins_every_field() {
  // diaric's typed bail-out — the exact variant the all-CPU arm hits.
  let all_cpu_err: Result<Vec<Seg>, diaric::offline::Error> =
    Err(diaric::offline::Error::Pipeline(
      diaric::pipeline::Error::Centroid(diaric::cluster::centroid::Error::AmbiguousAliveCluster {
        cluster: 13,
        value: 1.70e-7,
        threshold: 1e-7,
        lo: 5e-8,
        hi: 2e-7,
      }),
    ));

  let ship_der = clip09_synth_der(0.029_810);
  let emb_cpu_der = clip09_synth_der(0.013_011);
  let good = Clip09Observed {
    clip: "09_mrbeast_dollar_date",
    ref_spk: 8,
    dia_spk: 8,
    ship_spk: 8,
    ship_der,
    emb_cpu_spk: 9,
    emb_cpu_der,
    all_cpu: &all_cpu_err,
  };
  // The measured record passes.
  assert_clip09_record(&good);

  // The VARIANT is matched, not its field values: a DIFFERENT
  // AmbiguousAliveCluster still passes, so the pin does not over-fit the
  // fragile `cluster: 13` / `value: 1.70e-7`.
  let all_cpu_other_fields: Result<Vec<Seg>, diaric::offline::Error> =
    Err(diaric::offline::Error::Pipeline(
      diaric::pipeline::Error::Centroid(diaric::cluster::centroid::Error::AmbiguousAliveCluster {
        cluster: 99,
        value: -1.0,
        threshold: 5e-8,
        lo: 1e-8,
        hi: 9e-8,
      }),
    ));
  assert_clip09_record(&Clip09Observed {
    all_cpu: &all_cpu_other_fields,
    ..good
  });

  // ── Every single-field perturbation must FAIL. ──
  fn reject(label: &str, o: &Clip09Observed<'_>) {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      assert_clip09_record(o);
    }));
    assert!(
      res.is_err(),
      "mutation '{label}' did NOT fail the clip-09 record — that field is unpinned, so a real \
       change in it would pass silently"
    );
  }

  // Identity + counts (both directions on every scalar).
  reject(
    "clip",
    &Clip09Observed {
      clip: "10_mrbeast_clean_water",
      ..good
    },
  );
  reject("ref_spk hi", &Clip09Observed { ref_spk: 9, ..good });
  reject("ref_spk lo", &Clip09Observed { ref_spk: 7, ..good });
  reject("dia_spk hi", &Clip09Observed { dia_spk: 9, ..good });
  reject("dia_spk lo", &Clip09Observed { dia_spk: 7, ..good });
  reject(
    "ship_spk lo (the collapse class)",
    &Clip09Observed {
      ship_spk: 7,
      ..good
    },
  );
  reject(
    "ship_spk hi (the knife edge escaping)",
    &Clip09Observed {
      ship_spk: 9,
      ..good
    },
  );
  reject(
    "emb_cpu_spk lo (knife edge resolved)",
    &Clip09Observed {
      emb_cpu_spk: 8,
      ..good
    },
  );
  reject(
    "emb_cpu_spk hi",
    &Clip09Observed {
      emb_cpu_spk: 10,
      ..good
    },
  );

  // all-CPU arm: "resolved" (clusters) and a DIFFERENT error variant both fail.
  let all_cpu_fixed: Result<Vec<Seg>, diaric::offline::Error> = Ok(Vec::new());
  reject(
    "all_cpu resolved (Ok)",
    &Clip09Observed {
      all_cpu: &all_cpu_fixed,
      ..good
    },
  );
  let all_cpu_wrong_variant: Result<Vec<Seg>, diaric::offline::Error> = Err(
    diaric::offline::Error::Pipeline(diaric::pipeline::Error::InvalidActiveRatio(0.5)),
  );
  reject(
    "all_cpu wrong variant",
    &Clip09Observed {
      all_cpu: &all_cpu_wrong_variant,
      ..good
    },
  );

  // SHIPPING DER + FULL decomposition (both directions).
  reject(
    "ship der hi",
    &Clip09Observed {
      ship_der: clip09_synth_der(0.029_810 + 2.0 * DER_PIN_TOL),
      ..good
    },
  );
  reject(
    "ship der lo",
    &Clip09Observed {
      ship_der: clip09_synth_der(0.029_810 - 2.0 * DER_PIN_TOL),
      ..good
    },
  );
  reject(
    "ship miss_units",
    &Clip09Observed {
      ship_der: Der {
        miss_units: 1,
        ..ship_der
      },
      ..good
    },
  );
  reject(
    "ship fa_units",
    &Clip09Observed {
      ship_der: Der {
        fa_units: 1,
        ..ship_der
      },
      ..good
    },
  );
  reject(
    "ship confusion hi",
    &Clip09Observed {
      ship_der: Der {
        confusion: ship_der.confusion + 2.0 * DER_PIN_TOL,
        ..ship_der
      },
      ..good
    },
  );
  reject(
    "ship confusion lo",
    &Clip09Observed {
      ship_der: Der {
        confusion: ship_der.confusion - 2.0 * DER_PIN_TOL,
        ..ship_der
      },
      ..good
    },
  );

  // CPU-embedder control DER value (both directions; its decomposition is not
  // separately pinned).
  reject(
    "emb_cpu der hi",
    &Clip09Observed {
      emb_cpu_der: clip09_synth_der(0.013_011 + 2.0 * DER_PIN_TOL),
      ..good
    },
  );
  reject(
    "emb_cpu der lo",
    &Clip09Observed {
      emb_cpu_der: clip09_synth_der(0.013_011 - 2.0 * DER_PIN_TOL),
      ..good
    },
  );
}

/// A synthetic [`Der`] for the clip-09 mutation test: standard DER = confusion =
/// `der`, zero miss/FA (both pinned clip-09 arms' error is pure confusion). Only
/// the fields [`assert_clip09_record`] reads need be meaningful; the rest are
/// inert.
fn clip09_synth_der(der: f64) -> Der {
  Der {
    der,
    miss: 0.0,
    fa: 0.0,
    confusion: der,
    miss_units: 0,
    fa_units: 0,
    conf_units: 1,
    ref_units: 1,
    scored_frames: 1,
    err_frames: 1,
    num_ref_spk: 8,
    num_hyp_spk: 8,
  }
}

/// What each candidate shipping configuration COSTS: model load time and
/// steady-state extraction latency, measured cleanly, with the segmentation
/// and embedder placements varied INDEPENDENTLY (they are independent knobs in
/// production — [`coremlit::audio::speaker::extract::ComputeOptions`] carries one
/// [`ComputeUnits`] per model).
///
/// The per-arm `extract_s` printed by [`measure`] conflates model LOAD
/// (a one-off, and on `All` a first-run ANE compile that can take minutes) with
/// INFERENCE, and those runs are deliberately concurrent, so neither number is a
/// usable latency. This test measures the two phases separately, one config at a
/// time, with a warm-up pass so no ANE compile lands inside the timed region.
///
/// Rows sharing a segmentation placement differ only in the embedder's
/// artifact/placement, so their `extract_s` difference is the embedder's own
/// contribution.
///
/// Reported, not gated: latency is hardware-dependent, and a wall-clock bound
/// would be a flaky gate. The DER gates above are the ones that must hold.
#[test]
#[ignore = "requires Models/speakerkit; latency benchmark (reported, not gated)"]
fn shipping_embedder_cost_int8_vs_fp32() {
  // A 120 s slice of the 7-speaker clip: long enough that per-chunk steady-state
  // cost dominates fixed overhead, short enough to run every config twice.
  const BENCH_S: usize = 120;
  let all = common::load_wav_16k_mono(&clip_audio_path(
    clip_by_name("10_mrbeast_clean_water").name,
  ));
  let samples = &all[..(BENCH_S * 16_000).min(all.len())];
  let audio_s = samples.len() as f64 / 16_000.0;

  println!("\n══ embedder cost: {audio_s:.1} s of 10_mrbeast_clean_water ══");
  println!(
    "{:<32} {:>10} {:>12} {:>10} {:>12}",
    "config (seg / embedder)", "load_s", "extract_s", "RTF", "per-chunk_ms"
  );

  for (tag, embed_path, seg_cu, emb_cu) in [
    (
      "seg@All + int8@All",
      common::embed_path(),
      ComputeUnits::All,
      ComputeUnits::All,
    ),
    (
      "seg@All + fp32@All",
      common::embed_fp32_path(),
      ComputeUnits::All,
      ComputeUnits::All,
    ),
    (
      "seg@All + fp32@CpuOnly",
      common::embed_fp32_path(),
      ComputeUnits::All,
      ComputeUnits::CpuOnly,
    ),
    (
      "seg@CpuOnly + fp32@CpuOnly",
      common::embed_fp32_path(),
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuOnly,
    ),
    (
      "seg@CpuOnly + int8@CpuOnly",
      common::embed_path(),
      ComputeUnits::CpuOnly,
      ComputeUnits::CpuOnly,
    ),
  ] {
    let t0 = Instant::now();
    let seg = SegmentModel::from_file_with(
      common::seg_path(),
      SegmentModelOptions::new().with_compute(seg_cu),
    )
    .expect("load segmentation");
    let embed =
      EmbedModel::from_file_with(&embed_path, EmbedModelOptions::new().with_compute(emb_cu))
        .expect("load embedder");
    let load_s = t0.elapsed().as_secs_f64();

    let source = FluidAudioSource::with_options(seg, embed, Options::new());
    // Warm-up: forces any lazy CoreML/ANE specialization OUT of the timed region.
    let warm = source.extract(samples).expect("warm-up extract");
    let num_chunks = warm.num_chunks();
    drop(warm);

    let t1 = Instant::now();
    let ext = source.extract(samples).expect("timed extract");
    let extract_s = t1.elapsed().as_secs_f64();
    assert_eq!(ext.num_chunks(), num_chunks, "{tag}: chunk count unstable");

    println!(
      "{tag:<32} {load_s:>10.2} {extract_s:>12.2} {:>9.1}× {:>12.1}",
      audio_s / extract_s,
      extract_s * 1000.0 / num_chunks as f64,
    );
  }
  println!(
    "(RTF = audio seconds processed per wall-clock second; higher is faster. Rows sharing a seg \
     placement differ only in the embedder, so their extract_s delta is the embedder's own cost.)"
  );
}

/// The shipping default really is the fp32 artifact — the premise this whole
/// suite rests on since issue #15 retired the int8 embedder. It pins the exact
/// selection at its source of truth: the pure [`FluidAudioArtifacts::resolve`]
/// that `AnySource::load` itself uses. If that resolver is ever repointed back
/// at the int8 `wespeaker_v2.mlmodelc`, this fails — and so does the hermetic
/// `FluidAudioArtifacts` unit test — because both read production's own
/// selection, not a parallel copy of the path (finding 3: an earlier version
/// asserted only the enum variant and directory sizes, so repointing the
/// loader passed everything while the DER arms went on loading their own
/// path).
///
/// Needs only the model directory (no audio, no ort), so it runs cheaply — but
/// it is still `#[ignore]`d because it loads the real artifacts.
#[test]
#[ignore = "requires Models/speakerkit"]
fn shipping_default_is_the_fp32_embedder() {
  let root = common::models_dir();
  assert!(
    root.join("wespeaker.mlmodelc").exists(),
    "wespeaker.mlmodelc (fp32) missing under {}",
    root.display()
  );
  // The selection, pinned at production's source of truth. `AnySource::load`
  // resolves the FluidAudio embedder through exactly this, and the DER arms in
  // `measure()` load through it too, so this assertion covers what actually
  // ships — not a re-encoding of it.
  let artifacts = FluidAudioArtifacts::resolve(&root);
  assert!(
    artifacts.embedder().ends_with("wespeaker.mlmodelc"),
    "AnySource::load's FluidAudio embedder resolves to {}, not the fp32 wespeaker.mlmodelc — \
     the shipping default moved, so the DER gates in this suite are now measuring the wrong \
     artifact",
    artifacts.embedder().display()
  );
  // `AnySource::load` (the shipping entry point) must also succeed against the
  // real directory and build the FluidAudio variant.
  let source = AnySource::load(&root, Options::new()).expect("AnySource::load shipping default");
  assert!(
    matches!(source, AnySource::FluidAudio(_)),
    "the default Source is no longer FluidAudio — re-derive which embedder ships"
  );
  // Documented sizes, so the accepted footprint cost of the fp32 default is
  // visible right where the selection is gated (fp32 ≈ 29.4 MB vs the retired
  // int8's ≈ 8.0 MB — issue #15 accepted the +21 MB to fix the 8-speaker
  // collapse; the int8 artifact stays on disk as a tested, non-shipping
  // sibling).
  let du = |p: &Path| -> u64 {
    fn walk(p: &Path) -> u64 {
      std::fs::read_dir(p).map_or(0, |rd| {
        rd.flatten()
          .map(|e| {
            let m = e.metadata().expect("metadata");
            if m.is_dir() { walk(&e.path()) } else { m.len() }
          })
          .sum()
      })
    }
    walk(p)
  };
  let fp32 = du(artifacts.embedder());
  let int8 = du(&root.join("wespeaker_v2.mlmodelc"));
  println!(
    "shipping embedder wespeaker (fp32) = {:.1} MB | retired wespeaker_v2 (int8) = {:.1} MB | \
     fp32 costs {:+.1} MB ({:.1}×)",
    fp32 as f64 / 1e6,
    int8 as f64 / 1e6,
    (fp32 as f64 - int8 as f64) / 1e6,
    fp32 as f64 / int8 as f64,
  );
  assert!(
    int8 < fp32,
    "wespeaker_v2 ({int8} B) is not smaller than wespeaker ({fp32} B) — the int8/fp32 \
     identification is wrong"
  );
}

// ══════════════════════════════════════════════════════════════════════
// Corpus guard — the shipping matrix gates a documented SUBSET, not all of it
// ══════════════════════════════════════════════════════════════════════

/// The shipping matrix ([`MULTI_SPEAKER_CLIPS`]) gates a SELECTED subset — four
/// of the eight ≥ 3-speaker clips in dia's parity corpus ([`MULTISPK_CORPUS`]),
/// NOT all of them. This pins that selection against silent drift instead of
/// trusting the prose: it re-derives the ≥ 3-speaker membership straight from the
/// RTTMs on disk and asserts both the denominator (eight) and the selected four.
///
/// Needs only the sibling `diarization` fixtures (no models), like
/// `parity_e2e`'s `fixture_facts_match_the_corpus_on_disk` (which independently
/// pins all 14 clips).
///
/// # Why only four are gated (the excluded 08 / 11 / 12 / 13)
///
/// The selected four span the speaker-count ladder — 3 / 4 / 7 / 8 — at the
/// lowest runtime that covers it. The other four ≥ 3-speaker clips are excluded
/// deliberately: 08 (3 spk) duplicates 06's minimal-≥3 case, and 11 (6), 13 (11),
/// 12 (15) are higher counts whose shipping-configuration behaviour is
/// UNMEASURED in this suite (§5.9
/// measured only 06 / 14 / 10 / 09), each adding ~10 min to an already ~65-min
/// suite (~+40 min for all four). Adding one is NOT free here: this suite runs
/// once and pins measured values, so a new gated clip needs a measured
/// baseline FIRST — a blind gate on an unmeasured clip could fail with no way
/// to tell a real defect from a missing expectation. Clip 12 (the known
/// FluidAudio-breach clip, whose fp32 all-CPU arm parity_e2e's stress set
/// already measures) has the highest marginal value and is first to add once
/// measured on all three placements.
#[test]
#[ignore = "requires the sibling diarization parity fixtures (no models needed)"]
fn shipping_clip_selection_is_the_documented_subset() {
  // Derive the ≥ 3-speaker membership from the RTTMs on disk — the denominator of
  // "four of eight", not trusted from a table.
  let mut found: Vec<(String, usize)> = Vec::new();
  for entry in std::fs::read_dir(fixtures_root()).expect("read dia parity fixtures root") {
    let dir = entry.expect("dir entry").path();
    let rttm = dir.join("reference.rttm");
    if !rttm.is_file() {
      continue;
    }
    let n = distinct_speakers(&parse_rttm(&rttm)).len();
    if n >= 3 {
      let name = dir
        .file_name()
        .expect("clip dir name")
        .to_string_lossy()
        .into_owned();
      found.push((name, n));
    }
  }
  found.sort();

  let mut documented: Vec<(String, usize)> = MULTISPK_CORPUS
    .iter()
    .map(|(name, spk)| ((*name).to_string(), *spk))
    .collect();
  documented.sort();
  assert_eq!(
    found, documented,
    "the corpus's ≥ 3-speaker membership on disk differs from MULTISPK_CORPUS — re-derive it \
     (and the module doc's 'four of eight' denominator) from the RTTMs"
  );
  assert_eq!(
    MULTISPK_CORPUS.len(),
    8,
    "the ≥ 3-speaker corpus is no longer eight clips — the 'four of eight' denominator moved"
  );

  // The gated selection is exactly the documented four, each a member of the
  // ≥ 3-speaker corpus above. If MULTI_SPEAKER_CLIPS drifts, this fails.
  let selected: Vec<&str> = MULTI_SPEAKER_CLIPS.iter().map(|c| c.name).collect();
  assert_eq!(
    selected,
    [
      "06_long_recording",
      "14_mrbeast_strongman_robot",
      "10_mrbeast_clean_water",
      "09_mrbeast_dollar_date",
    ],
    "the shipping stress selection moved — re-derive it and update the module doc's 'four of \
     eight' claim"
  );
  for c in MULTI_SPEAKER_CLIPS {
    assert!(
      MULTISPK_CORPUS
        .iter()
        .any(|(name, spk)| *name == c.name && *spk == c.ref_spk),
      "{}: gated but not in the ≥ 3-speaker corpus manifest (or its count disagrees)",
      c.name
    );
  }
}

/// F4 mutation proof: the content-identity pin (`samples` + `audio_fnv`) that
/// [`measure`] asserts actually catches a same-name audio swap on clip 09.
/// Loads the real clip (no models — only the sibling fixture, like
/// [`shipping_clip_selection_is_the_documented_subset`]) and shows the pinned
/// values match the bytes on disk, then that BOTH a one-sample perturbation
/// (the FNV half) and a length change (the count half) break the pin. Without
/// this the pinned numbers could silently drift from the clip and a swap would
/// still pass `measure`.
#[test]
#[ignore = "requires the sibling diarization parity fixtures (no models needed)"]
fn clip09_content_pin_catches_an_audio_swap() {
  let clip = clip_by_name("09_mrbeast_dollar_date");

  let samples = common::load_wav_16k_mono(&clip_audio_path(clip.name));
  let fnv = common::fnv1a_f32(&samples);

  // The pinned values ARE the real clip's — the same check `measure` runs. If a
  // future capture drifts from the bytes, this fails here rather than letting a
  // stale pin wave a swapped clip through.
  assert_eq!(
    samples.len(),
    clip.samples,
    "pinned sample count {} != the clip's {}",
    clip.samples,
    samples.len()
  );
  assert_eq!(fnv, clip.audio_fnv, "pinned FNV != the clip's");

  // (a) One perturbed sample, SAME length: the count is unchanged but the FNV
  // must move — so `measure`'s pin fires on a same-length swap the count misses.
  let mut swapped = samples.clone();
  let i = swapped.len() / 2;
  swapped[i] = f32::from_bits(swapped[i].to_bits() ^ 1); // flip one mantissa bit
  assert_eq!(
    swapped.len(),
    clip.samples,
    "the perturbation keeps the count"
  );
  assert_ne!(
    common::fnv1a_f32(&swapped),
    clip.audio_fnv,
    "a one-sample perturbation must break the FNV pin"
  );

  // (b) A truncated clip: the sample-count half must fire.
  assert_ne!(
    samples.len() - 1,
    clip.samples,
    "a length change must break the sample-count pin"
  );
}
