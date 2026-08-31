//! **Where does the clip-09 collapse come from — the segmentation conversion,
//! the embedding conversion, or both — AT THE CONFIGURATION THAT SHIPPED?**
//!
//! This file is the ATTRIBUTION RECORD for issue #15. The configuration it
//! dissects — the int8 `wespeaker_v2.mlmodelc` on [`ComputeUnits::All`] —
//! shipped until that issue retired it for the collapse pinned here (5 of 8
//! speakers, 16.5904 % DER, 100 % confusion on clip 09), and the retirement
//! decision (`model_io.rs`'s DECISION, `parity_shipping_der.rs`'s gates)
//! cites these experiments as its evidence. The int8 artifact stays on disk,
//! byte-pinned (`model_io.rs`), exactly so this record keeps reproducing.
//!
//! `parity_shipping_der.rs` measured that default end to end; it could not
//! say WHICH of the two CoreML conversions produced the collapse, because it
//! never varies them independently: every one of its speakerkit arms runs
//! CoreML segmentation AND CoreML embedding.
//!
//! This suite varies them independently — the 2x2 cross-product of
//! `{ONNX, CoreML}` segmentation x `{ONNX, CoreML}` embedding — with every
//! other factor held constant, and it does so **at that configuration**: the
//! int8 embedder, both CoreML models on [`ComputeUnits::All`].
//!
//! # Why this is a distinct experiment from the one already on record
//!
//! An earlier cross-product (issue #15, recorded in `model_io.rs`'s module doc)
//! ran the **fp32** embedder on **`CpuOnly`**. That configuration has a
//! different symptom — dia's clustering returns
//! `Err(AmbiguousAliveCluster { .. })`, i.e. NO diarization at all — from the
//! one this crate ships, which silently returns 5 of 8 speakers. A result
//! measured on the erroring configuration cannot exonerate the answering one:
//! they differ in both the embedder artifact (fp32 vs int8-palettized) and the
//! placement (`CpuOnly`'s IEEE CPU kernels vs `All`'s fp16 ANE/GPU kernels),
//! and the failure they exhibit is not even the same kind of failure. Hence
//! this suite, which runs the identical design where the defect actually lives.
//!
//! # The design (one variable at a time)
//!
//! Held constant across all four cells, so the ONLY difference between any two
//! is which conversion computed a stage:
//!
//! - **one audio buffer** — FNV-1a fingerprinted before and after every cell
//!   and asserted unchanged (a divergence caused by different input is a
//!   harness bug, not a finding);
//! - **the chunk grid** — speakerkit's production
//!   [`chunk_starts`]/[`chunk_sliding_window`]/[`frame_sliding_window`]
//!   geometry, from the production [`Options`] defaults;
//! - **the powerset decode** — speakerkit's shipping [`multilabel`] (direct
//!   argmax over the log-probabilities) on BOTH segmentation backends. Holding
//!   it fixed is what makes the seg factor a *backend* factor and not a
//!   backend-plus-decode factor; see [`assemble`]'s doc for the consequence
//!   this has for the all-ONNX cell;
//! - **the overlap-exclusion mask rule** — `common::derive_expected_slot_masks`,
//!   the same port of `owned.rs:507-591` that speakerkit's private
//!   `extract::derive_slot_plans` and `generate_goldens.rs`'s
//!   `derive_slot_masks` implement;
//! - **the clustering** — `diaric::offline::diarize_offline` over the measured
//!   side's community-1 PLDA, for every cell including the all-ONNX one.
//!
//! # Harness validity: both corners are anchored to the REAL pipelines
//!
//! The cells are assembled by this file, not by `Extractor::extract` — no
//! public entry point runs a MIXED backend end to end (`Extraction::from_parts`
//! is private to the `extract` module, reachable only through
//! `Extraction::assemble_checked` or the public `Extraction::try_from_parts`,
//! and the latter takes an already-assembled tensor set, which is exactly what
//! this file builds by hand). A hand-assembled pipeline is only worth as much as its agreement with
//! the real one:
//!
//! - the all-ONNX corner must reproduce dia-ort's pinned clip-09 speaker
//!   count (8) at 0.0000 % DER against `reference.rttm`
//!   ([`assert_factorial_verdict`]);
//! - the all-CoreML corner is checked against an IN-SUITE control:
//!   [`shipping_config_backend_factorial`] also runs the identical
//!   configuration through the PRODUCTION path — `FluidAudioSource::extract`
//!   (the real `Extractor::extract`) + the public `Extraction::diarize` —
//!   and asserts the production run and the hand-assembled corner agree on
//!   the count and to ±[`CORNER_DER_TOL`] / ±[`ERR_UNITS_TOL`] on the
//!   error mass, both landing on the retired shipping record (5 speakers,
//!   16.5904 %). This control used to live in `parity_shipping_der` as its
//!   int8/All pin; that suite's arms moved to the fp32 shipping
//!   configuration with issue #15, so the anchor now runs here, where the
//!   retired artifact still has its record to hold.
//!
//! Both hold on the measured run, which is what licenses reading the hybrids.
//!
//! # What it found: the two configurations do NOT agree
//!
//! The recorded fp32/`CpuOnly` cross-product concluded "the segmentation
//! conversion is guilty; the embedder is exonerated". At the shipping
//! configuration that is **not** what happens (full table and per-cell
//! rationale in [`assert_factorial_verdict`]): swapping only the EMBEDDING
//! conversion reproduces the collapse exactly (5 of 8 speakers, 16.5904 %,
//! the all-CoreML corner's own number), while swapping only the SEGMENTATION
//! conversion produces a different and much smaller defect — one spurious
//! speaker, 1.3011 %. A conclusion measured on the fp32/`CpuOnly`
//! configuration did not transfer to the one this crate ships, which is
//! precisely why it had to be re-measured here.
//!
//! # The follow-up this file also carries
//!
//! The cross-product above varies the whole BACKEND, so its finding implicates
//! the CoreML embedding path as shipped — int8-palettized artifact **plus**
//! `All` placement **plus** that conversion — as one bundle.
//! [`embedding_precision_x_placement`] separates the three: same harness, same
//! clip, dia's reference segmentation held fixed for every arm, and the
//! embedding arm run across precision x placement. Its verdict
//! ([`assert_precision_placement_verdict`]) carries the measured table; in
//! short, the conversion alone is frame-perfect, the int8 palettization costs 2
//! speakers at either placement, and `All` costs 1 at either precision.
//!
//! # What this suite does NOT establish
//!
//! It localizes the collapse to a STAGE, on ONE clip, at ONE configuration, on
//! ONE host. Three limits, all load-bearing:
//!
//! - **It does not separate int8 from `All` from the conversion itself.** The
//!   factor varied is the BACKEND; the CoreML embedding arm is the shipping
//!   bundle (int8-palettized artifact + `All` placement + that conversion) as
//!   one unit. See [`assert_factorial_verdict`]'s "What it does NOT pin" — and
//!   [`embedding_precision_x_placement`], which is the experiment that does
//!   separate them.
//! - **It does not say which op inside a stage is responsible, and it cannot.**
//!   Both segmentation graphs expose only their post-tail output
//!   (`z - logsumexp(z)`), never the pre-tail logits `z`, so no measurement
//!   here separates a divergence created in the graph's trunk from one created
//!   in its log-softmax tail. [`seg_divergence`] reports the one decomposition
//!   the observable outputs DO support, and names what would settle the rest.
//! - **It is one clip.** Clip 09 is the clip with the defect; nothing here
//!   extends to the rest of the corpus.
//!
//! `#[ignore]`d and `speaker-oracle`-gated (needs the gitignored
//! `Models/speakerkit`, the sibling `diarization` ONNX + parity fixtures, and
//! `ort`). Run with:
//!
//! ```text
//! cargo test -p coremlit-parity --features speaker-oracle --test speaker_backend_factorial -- --ignored --nocapture
//! ```
//!
//! **`--features speaker-oracle`, never `--all-features`.** Under
//! `--all-features` this binary HANGS instead of running: `align-oracle` pulls
//! `asry`, which enables `ort/load-dynamic`, and Cargo unifies that onto the
//! single `ort` in the build — so `dia`'s first `Session` tries to `dlopen` an
//! ONNX Runtime dylib that is not there, and `ort`'s failure path re-enters the
//! same `OnceLock` it is initializing (`setup_api` -> error construction ->
//! `ort::api()` -> `Once::wait`) and deadlocks forever. A plain
//! `cargo test -p coremlit --all-features` does not notice, because without
//! `--ignored` it never opens an `ort` session. The same applies to every
//! `speaker-oracle` DER binary.
#![cfg(feature = "speaker-oracle")]

// The shared speaker test-support module lives in the `coremlit` package (13 of
// its test binaries include it as a plain `mod common;`); this oracle binary
// pulls in that ONE copy rather than a fork that could drift.
#[path = "../../../coremlit/tests/speaker/common/mod.rs"]
mod common;
#[path = "../../../coremlit/tests/speaker/der_calc/mod.rs"]
mod der_calc;

use std::path::PathBuf;

use coremlit::{
  ComputeUnits,
  audio::speaker::{
    embed::{EMBED_SLOTS, EMBEDDING_DIM, EmbedModel, EmbedModelOptions},
    extract::Options,
    segment::{
      POWERSET_CLASSES, SEG_CHUNK_SAMPLES, SEG_NUM_SLOTS, SegmentModel, SegmentModelOptions,
      multilabel,
    },
    source::{FluidAudioSource, ModelSource},
    window::{
      WindowOptions, chunk_sliding_window, chunk_starts, count_from_segmentations,
      frame_sliding_window,
    },
  },
};
use der_calc::{Der, Seg, der_std, distinct_speakers, fmt_der, parse_rttm};

// ══════════════════════════════════════════════════════════════════════
// The clip, the pinned corners, and the fixture/model resolution
// ══════════════════════════════════════════════════════════════════════

/// The clip this experiment runs on: dia's 8-speaker parity fixture — the
/// clip whose int8-era shipping collapse this suite dissects and whose
/// post-fix record `parity_shipping_der`'s
/// `shipping_der_09_mrbeast_dollar_date_8spk` now gates.
const CLIP: &str = "09_mrbeast_dollar_date";

/// Decoded 16 kHz-mono sample count of `09_mrbeast_dollar_date/clip_16k.wav`,
/// and [`common::fnv1a_f32`] of those samples — the same two-part content pin
/// `parity_shipping_der::MULTI_SPEAKER_CLIPS` carries, repeated here so a
/// same-name audio swap fails before any cell is measured rather than
/// producing a comparison across different audio.
const CLIP_SAMPLES: usize = 16_671_744;
const CLIP_AUDIO_FNV: u64 = 8_657_240_795_675_234_981;

/// `reference.rttm`'s distinct-speaker count for [`CLIP`] (pyannote 4.0.4's own
/// output, not human labels — see `parity_shipping_der`'s module doc).
const CLIP_REF_SPK: usize = 8;

/// The all-CoreML corner's expected state: the retired int8/All shipping
/// record (formerly `parity_shipping_der`'s clip-09 pin, retired there with
/// the artifact). [`shipping_config_backend_factorial`]'s in-suite production
/// control re-measures the same configuration through the real
/// `FluidAudioSource::extract` + `Extraction::diarize` path, so this corner
/// reproduces these numbers or the hand-assembly in [`assemble`] has
/// diverged from `Extractor::extract`.
const SHIPPING_CORNER_SPK: usize = 5;
const SHIPPING_CORNER_DER: f64 = 0.165_904;

/// The all-ONNX corner's expected speaker count: dia-ort's pinned clip-09
/// oracle count. Its DER against `reference.rttm` is 0.0000 % (dia-ort is
/// frame-perfect against pyannote here), asserted as
/// `<= `[`CORNER_DER_TOL`].
const REFERENCE_CORNER_SPK: usize = 8;

/// `ONNX-seg + COREML-emb` — the measured speaker count when ONLY the
/// embedding conversion is swapped, over dia's own reference segmentation.
/// It equals [`SHIPPING_CORNER_SPK`], at [`SHIPPING_CORNER_DER`]: the shipping
/// collapse, reproduced by the embedder alone. See
/// [`assert_factorial_verdict`].
const EMBED_ONLY_SPK: usize = 5;

/// `COREML-seg + ONNX-emb` — the measured speaker count and standard DER when
/// ONLY the segmentation conversion is swapped. An OVERcount by one, an order
/// of magnitude smaller than the collapse: a separate defect, not this one.
const SEG_ONLY_SPK: usize = 9;
const SEG_ONLY_DER: f64 = 0.013_011;

/// Two-sided band on the corner DER reproductions — `parity_shipping_der`'s own
/// `DER_PIN_TOL`, for the same reason: the pipeline is deterministic and these
/// values reproduced exactly across runs, so the band absorbs a stray flipped
/// frame on a different CoreML build, not real movement. The distances this
/// experiment resolves are ~16 pp, ~330x this band.
const CORNER_DER_TOL: f64 = 0.000_5;

/// dia's parity-fixture root (override with `DIA_PARITY_FIXTURES`) — the same
/// convention `parity_e2e.rs` / `parity_shipping_der.rs` use.
fn fixtures_root() -> PathBuf {
  std::env::var_os("DIA_PARITY_FIXTURES").map_or_else(
    || common::checkout_parent().join("diarization/tests/parity/fixtures"),
    PathBuf::from,
  )
}

/// dia's fp32 WeSpeaker ONNX (override with `DIA_EMBED_MODEL_PATH`) — the same
/// convention `parity_e2e.rs` / `generate_goldens.rs` use.
fn dia_wespeaker_onnx() -> PathBuf {
  std::env::var_os("DIA_EMBED_MODEL_PATH").map_or_else(
    || common::checkout_parent().join("diarization/models/wespeaker_resnet34_lm.onnx"),
    PathBuf::from,
  )
}

// ══════════════════════════════════════════════════════════════════════
// The two factors
// ══════════════════════════════════════════════════════════════════════

/// Which conversion computes a stage. The whole experiment is this value,
/// chosen independently for segmentation and for embedding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
  /// dia's ONNX graph on `ort`'s CPU EP — the reference implementation.
  Onnx,
  /// speakerkit's CoreML conversion at the placement under test.
  CoreMl,
}

impl Backend {
  const fn tag(self) -> &'static str {
    match self {
      Self::Onnx => "ONNX",
      Self::CoreMl => "COREML",
    }
  }
}

/// The compute placement both CoreML models run on. `All` is the shipping
/// default ([`coremlit::audio::speaker::segment::DEFAULT_SEGMENT_COMPUTE`] and
/// its embed twin), which is the entire point of running the factorial here:
/// the recorded prior cross-product used `CpuOnly`.
const PLACEMENT: ComputeUnits = ComputeUnits::All;

/// The one embedding artifact this experiment's CoreML side uses:
/// `wespeaker_v2.mlmodelc`, i.e. the **int8-palettized** model `ModelSource`
/// actually loads (`common::embed_path`). The prior cross-product used the
/// fp32 `wespeaker.mlmodelc` instead.
fn coreml_embed_path() -> PathBuf {
  common::embed_path()
}

/// The CoreML embedding artifact's **weight precision** — the quantization
/// axis of [`embedding_precision_x_placement`].
///
/// Both artifacts are contract-equal (`model_io`'s
/// `wespeaker_fp32_io_matches_spec`), so they are
/// substitutable for each other in [`EmbedSide`] and differ only in whether the
/// weights were palettized.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Precision {
  /// `wespeaker.mlmodelc` — 27 MB of unquantized float32 weights.
  Fp32,
  /// `wespeaker_v2.mlmodelc` — the int8-palettized artifact `ModelSource`
  /// ships.
  Int8,
}

impl Precision {
  fn path(self) -> PathBuf {
    match self {
      Self::Fp32 => common::embed_fp32_path(),
      Self::Int8 => coreml_embed_path(),
    }
  }
  const fn artifact(self) -> &'static str {
    match self {
      Self::Fp32 => "wespeaker.mlmodelc",
      Self::Int8 => "wespeaker_v2.mlmodelc",
    }
  }
  const fn tag(self) -> &'static str {
    match self {
      Self::Fp32 => "fp32",
      Self::Int8 => "int8",
    }
  }
}

/// speakerkit's CoreML conversion at a chosen precision and placement.
/// Payload of [`EmbedArm::CoreMl`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CoreMl {
  precision: Precision,
  placement: ComputeUnits,
}

/// **One embedding arm**: which conversion computes the embeddings and, for the
/// CoreML conversion, at which precision and on which compute placement.
///
/// [`shipping_config_backend_factorial`] only ever needs the two ends of the
/// [`Backend`] axis, so it uses [`Self::SHIPPING`] for its CoreML cells and
/// nothing else. [`embedding_precision_x_placement`] is the suite that opens
/// the other two dimensions up.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EmbedArm {
  /// dia's fp32 `wespeaker_resnet34_lm.onnx` on `ort`'s CPU EP — the reference
  /// implementation, and the only arm that is not a CoreML conversion.
  Onnx,
  /// speakerkit's CoreML conversion at a chosen precision and placement.
  CoreMl(CoreMl),
}

impl EmbedArm {
  /// The int8-palettized artifact on [`ComputeUnits::All`] — the embedding
  /// path that SHIPPED until issue #15 retired it. This is the bundle
  /// [`shipping_config_backend_factorial`] varies as a single unit.
  const SHIPPING: Self = Self::CoreMl(CoreMl {
    precision: Precision::Int8,
    placement: PLACEMENT,
  });

  /// The [`Backend`] this arm belongs to — the coarse factor the 2x2
  /// cross-product varies.
  const fn backend(self) -> Backend {
    match self {
      Self::Onnx => Backend::Onnx,
      Self::CoreMl(_) => Backend::CoreMl,
    }
  }

  /// `"ONNX (fp32) / CPU"`-style label for the report tables.
  fn label(self) -> String {
    match self {
      Self::Onnx => "ONNX fp32 / ort CPU EP".to_string(),
      Self::CoreMl(CoreMl {
        precision,
        placement,
      }) => format!("CoreML {} / {placement:?}", precision.tag()),
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
// Stage 1 — segmentation log-probabilities, one backend at a time
// ══════════════════════════════════════════════════════════════════════

/// Every chunk's flattened `[num_frames * POWERSET_CLASSES]` powerset
/// log-probabilities from ONE segmentation backend, plus the frame count both
/// backends must agree on.
struct SegRun {
  slabs: Vec<Vec<f32>>,
  num_frames: usize,
  elapsed_s: f64,
}

/// Fills `padded` with the chunk starting at `start`, zero-padding any overhang
/// past the end of `samples`. Byte-identical to speakerkit's private
/// `extract::fill_padded_chunk` (`extract/mod.rs`), itself a port of
/// `owned.rs:469-475`: the two models are contracted to exactly
/// [`SEG_CHUNK_SAMPLES`] samples and neither pads internally.
fn fill_padded_chunk(padded: &mut [f32], samples: &[f32], start: usize) {
  padded.fill(0.0);
  let end = (start + SEG_CHUNK_SAMPLES).min(samples.len());
  let lo = start.min(samples.len());
  let n = end - lo;
  if n > 0 {
    padded[..n].copy_from_slice(&samples[lo..end]);
  }
}

/// Runs one segmentation backend over the whole chunk grid.
///
/// Both backends emit the same quantity — per-frame powerset
/// log-probabilities, `z - logsumexp(z)` (dia's ONNX through `softmax` ->
/// `log`, the CoreML conversion through the fused `reduce_log_sum_exp` ->
/// `sub`; see `coremlit::audio::speaker::segment`'s module doc) — which is what
/// makes them comparable element-for-element in [`seg_divergence`] and
/// substitutable for each other here.
fn run_seg(backend: Backend, samples: &[f32], starts: &[usize]) -> SegRun {
  let mut padded = vec![0.0f32; SEG_CHUNK_SAMPLES];
  let mut slabs: Vec<Vec<f32>> = Vec::with_capacity(starts.len());
  let t0 = std::time::Instant::now();

  let num_frames = match backend {
    Backend::CoreMl => {
      let seg = SegmentModel::from_file_with(
        common::seg_path(),
        SegmentModelOptions::new().with_compute(PLACEMENT),
      )
      .expect("load pyannote_segmentation.mlmodelc");
      for &start in starts {
        fill_padded_chunk(&mut padded, samples, start);
        slabs.push(seg.infer(&padded).expect("CoreML segmentation infer"));
      }
      seg.num_frames()
    }
    Backend::Onnx => {
      let mut seg = dia::segment::SegmentModel::bundled().expect("dia bundled segmentation-3.0");
      for &start in starts {
        fill_padded_chunk(&mut padded, samples, start);
        slabs.push(seg.infer(&padded).expect("dia-ort segmentation infer"));
      }
      slabs[0].len() / POWERSET_CLASSES
    }
  };

  for (c, slab) in slabs.iter().enumerate() {
    assert_eq!(
      slab.len(),
      num_frames * POWERSET_CLASSES,
      "{}-seg chunk {c}: {} values, expected {num_frames} frames x {POWERSET_CLASSES} classes",
      backend.tag(),
      slab.len()
    );
  }
  SegRun {
    slabs,
    num_frames,
    elapsed_s: t0.elapsed().as_secs_f64(),
  }
}

// ══════════════════════════════════════════════════════════════════════
// Stage 2 — assembly: seg slabs + an embedding backend -> clustered spans
// ══════════════════════════════════════════════════════════════════════

/// The embedding backend for one cell, holding its loaded model. dia's is
/// `&mut self` per call; speakerkit's is `&self` and batches all three slots.
enum EmbedSide {
  CoreMl(EmbedModel),
  Onnx(Box<dia::embed::EmbedModel>),
}

impl EmbedSide {
  fn load(arm: EmbedArm) -> Self {
    match arm {
      EmbedArm::CoreMl(CoreMl {
        precision,
        placement,
      }) => Self::CoreMl(
        EmbedModel::from_file_with(
          precision.path(),
          EmbedModelOptions::new().with_compute(placement),
        )
        .unwrap_or_else(|e| {
          panic!(
            "load {} ({}) on {placement:?}: {e}",
            precision.artifact(),
            precision.tag()
          )
        }),
      ),
      EmbedArm::Onnx => {
        let onnx = dia_wespeaker_onnx();
        assert!(
          onnx.exists(),
          "dia WeSpeaker ONNX not found at {}; set DIA_EMBED_MODEL_PATH",
          onnx.display()
        );
        Self::Onnx(Box::new(
          dia::embed::EmbedModel::from_file(&onnx).expect("dia WeSpeaker fp32 ONNX"),
        ))
      }
    }
  }

  /// This chunk's raw 256-d embedding for every PLANNED slot, in slot order.
  /// `None` entries are slots the mask rule skipped — no embed call is made for
  /// them on either backend.
  ///
  /// The two arms mirror how each pipeline actually calls its model: dia's
  /// offline pipeline makes one `embed_chunk_with_frame_mask` call per planned
  /// slot (`owned.rs:593-617`), speakerkit's `Extractor` makes ONE batched
  /// `embed_chunk` call whose skipped rows borrow a planned slot's mask as a
  /// non-degenerate placeholder and are discarded (design spec §4). Calling
  /// each backend the way its own pipeline does is what lets the corners
  /// reproduce the pinned numbers.
  fn embed_chunk(
    &mut self,
    padded: &[f32],
    plans: &[Option<Vec<bool>>; SEG_NUM_SLOTS],
  ) -> [Option<[f32; EMBEDDING_DIM]>; SEG_NUM_SLOTS] {
    let Some(placeholder) = plans.iter().flatten().next() else {
      return [const { None }; SEG_NUM_SLOTS];
    };
    match self {
      Self::CoreMl(model) => {
        let masks: [&[bool]; EMBED_SLOTS] =
          core::array::from_fn(|s| plans[s].as_deref().unwrap_or(placeholder.as_slice()));
        let rows = model
          .embed_chunk(padded, &masks)
          .expect("CoreML embed_chunk");
        core::array::from_fn(|s| plans[s].as_ref().map(|_| rows[s]))
      }
      Self::Onnx(model) => core::array::from_fn(|s| {
        plans[s].as_ref().map(|mask| {
          model
            .embed_chunk_with_frame_mask(padded, mask)
            .expect("dia-ort embed_chunk_with_frame_mask")
        })
      }),
    }
  }
}

/// One cell's assembled diaric offline-input tensors.
struct Cell {
  segmentations: Vec<f64>,
  raw_embeddings: Vec<f32>,
  num_chunks: usize,
  num_frames: usize,
  embed_s: f64,
}

/// Assembles one cell: a fixed set of segmentation slabs + a chosen embedding
/// backend -> the `segmentations` / `raw_embeddings` tensor pair
/// `diaric::offline::OfflineInput` consumes.
///
/// This is `Extractor::extract`'s stage 8 (`extract/mod.rs`) with the two model
/// calls made pluggable, and it must stay a faithful copy of it — the
/// all-CoreML corner assertion in [`shipping_config_backend_factorial`] is what
/// enforces that. Reproduced in order: the shipping [`multilabel`] decode, the
/// overlap-exclusion mask rule, the Skip-slot column zeroing, and the ROW
/// PREDICATE drop (which also zeroes the slot's column).
///
/// **The decode is speakerkit's on BOTH segmentation backends.** dia's own
/// pipeline decodes `softmax`-then-argmax instead; the two agree over the reals
/// (a per-row constant shift) and on every committed golden row
/// (`parity_seg::golden_direct_and_dia_decode_agree`) but can differ on an f32
/// near-tie. Holding the decode fixed is required for the factorial to isolate
/// the BACKEND — the cost is that the all-ONNX cell is this crate's decode over
/// dia's logits, not a byte-for-byte rerun of dia's pipeline, which is why that
/// corner is checked against dia-ort's pinned speaker count rather than
/// asserted identical to it.
fn assemble(
  seg: &SegRun,
  embed: &mut EmbedSide,
  samples: &[f32],
  starts: &[usize],
  plda: &diaric::plda::PldaTransform,
) -> Cell {
  let num_frames = seg.num_frames;
  let num_chunks = starts.len();
  let mut segmentations = vec![0.0f64; num_chunks * num_frames * SEG_NUM_SLOTS];
  let mut raw_embeddings = vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM];
  let mut padded = vec![0.0f32; SEG_CHUNK_SAMPLES];
  let t0 = std::time::Instant::now();

  for (c, &start) in starts.iter().enumerate() {
    fill_padded_chunk(&mut padded, samples, start);
    let lo = c * num_frames * SEG_NUM_SLOTS;
    let hi = lo + num_frames * SEG_NUM_SLOTS;
    let slab = multilabel(&seg.slabs[c], num_frames);
    segmentations[lo..hi].copy_from_slice(&slab);

    // The overlap-exclusion rule, over the PRE-zeroing values, exactly as
    // `derive_slot_plans` runs it. `None` = Skip.
    let plans = common::derive_expected_slot_masks(&seg.slabs[c], num_frames);
    for (s, plan) in plans.iter().enumerate() {
      if plan.is_none() {
        zero_slot_column(&mut segmentations[lo..hi], num_frames, s);
      }
    }

    let rows = embed.embed_chunk(&padded, &plans);
    for (s, row) in rows.iter().enumerate() {
      let Some(row) = row else { continue };
      // dia's per-slot pre-check (`owned.rs:619-630`): a row neither backend
      // can consume is dropped and its column zeroed rather than fed to PLDA.
      //
      // This cell's contract is that it drops EXACTLY the rows the crate's own
      // producers drop, so the test is theirs, not a restatement of it.
      // `coremlit`'s `raw_embedding_reaches_plda` is `pub(crate)` and out of
      // reach from this integration test, so the three functions it composes
      // are called directly, in its order — `Embedding::normalize_from` (the
      // online engine's `f32`-narrowed norm and `1e-12` floor),
      // `RawEmbedding::from_wespeaker` (PLDA's raw boundary: a finiteness scan
      // and the `f64` `0.01` floor `PLDA_MIN_NORM` names), and
      // `PldaTransform::project` (the centered-norm rejection the offline route
      // runs immediately after that boundary). A local `norm_sq.sqrt() < 0.01`
      // here was a fourth copy of the FIRST clause only: it carried neither the
      // finiteness scan nor the `f32` narrowing nor the projection, so it could
      // keep a row the producers drop and the contract was not provable.
      let keeps = diaric::embed::Embedding::normalize_from(*row).is_some()
        && diaric::plda::RawEmbedding::from_wespeaker(*row)
          .is_ok_and(|raw| plda.project(&raw).is_ok());
      if !keeps {
        zero_slot_column(&mut segmentations[lo..hi], num_frames, s);
      } else {
        let e = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
        raw_embeddings[e..e + EMBEDDING_DIM].copy_from_slice(row);
      }
    }
  }

  Cell {
    segmentations,
    raw_embeddings,
    num_chunks,
    num_frames,
    embed_s: t0.elapsed().as_secs_f64(),
  }
}

/// Zeroes exactly slot `s`'s column across one chunk's `[f][s]` slab — dia's
/// column-zero on a dropped `(chunk, slot)` (`owned.rs:567-569,626-628`).
fn zero_slot_column(chunk_segs: &mut [f64], num_frames: usize, s: usize) {
  for f in 0..num_frames {
    chunk_segs[f * SEG_NUM_SLOTS + s] = 0.0;
  }
}

/// Clusters one assembled cell through the SAME `diaric::offline::diarize_offline`
/// every measured arm in this repo runs, returning diaric's TYPED error rather
/// than unwrapping: whether a cell can cluster AT ALL is a first-class result
/// here (the prior cross-product's every CoreML-seg cell could not).
fn cluster(
  cell: &Cell,
  window: &WindowOptions,
  plda: &diaric::plda::PldaTransform,
) -> Result<Vec<Seg>, diaric::offline::Error> {
  let count = count_from_segmentations(
    &cell.segmentations,
    cell.num_chunks,
    cell.num_frames,
    SEG_NUM_SLOTS,
    window.onset(),
    chunk_sliding_window(window),
    frame_sliding_window(),
  );
  let input = diaric::offline::OfflineInput::new(
    &cell.raw_embeddings,
    cell.num_chunks,
    SEG_NUM_SLOTS,
    &cell.segmentations,
    cell.num_frames,
    &count,
    count.len(),
    chunk_sliding_window(window).into(),
    frame_sliding_window().into(),
    plda,
  );
  diaric::offline::diarize_offline(&input).map(|out| {
    out
      .spans_slice()
      .iter()
      .map(|s| Seg {
        start: s.start(),
        end: s.start() + s.duration(),
        spk: s.cluster(),
      })
      .collect()
  })
}

// ══════════════════════════════════════════════════════════════════════
// The segmentation-divergence decomposition
// ══════════════════════════════════════════════════════════════════════

/// What the two segmentation backends' outputs differ by, and how much of that
/// difference the log-softmax TAIL could account for.
struct SegDivergence {
  /// Worst per-element `|CoreML - ONNX|` over every `(chunk, frame, class)`.
  worst_abs: f64,
  /// Total frames compared.
  frames: usize,
  /// Frames whose shipping [`multilabel`] argmax class differs.
  flips: usize,
  /// Of [`Self::flips`], those where the CoreML row has more than one element
  /// EQUAL to its maximum. See this struct's doc for why that is the only
  /// subset of flips the tail can be responsible for.
  flips_with_coreml_tie: usize,
  /// Minimum and maximum log-probability observed on each side — the observed
  /// value range Finding 3 asks for in place of a universal no-saturation claim.
  coreml_min: f64,
  coreml_max: f64,
  onnx_min: f64,
  onnx_max: f64,
  /// Non-finite cells on the CoreML side (the `ort` CoreML-EP corruption mode
  /// this crate exists to replace).
  coreml_nonfinite: usize,
}

/// Compares the two backends' log-probability slabs element for element and
/// splits the argmax flips into the two populations the observable outputs can
/// distinguish.
///
/// # What this can and cannot attribute
///
/// Neither graph exposes its PRE-TAIL logits `z`: both emit only
/// `z - logsumexp(z)`. So no measurement here can say whether a given
/// divergence was created in the graph's trunk (the convolutional/LSTM/linear
/// stack producing `z` in fp16) or in its log-softmax tail. What the outputs
/// DO support is a bound in one direction, for ARGMAX flips specifically:
///
/// - the tail subtracts ONE per-row scalar from every element of that row;
/// - correctly-rounded IEEE arithmetic is monotone, so for a common `L`,
///   `z_i >= z_j` implies `fl(z_i - L) >= fl(z_j - L)`;
/// - therefore the tail can never INVERT two elements' order. The only argmax
///   change it can produce is collapsing a strict inequality into an exact tie,
///   which [`multilabel`]'s lowest-index rule then breaks toward the lower
///   class.
///
/// So a flipped frame whose CoreML row has NO tie at its maximum cannot have
/// been flipped by the tail; its ordering already differed upstream.
/// [`Self::flips_with_coreml_tie`] is the entire population of flips the tail
/// could be responsible for.
///
/// Two caveats, both load-bearing: monotonicity is a property of
/// correctly-rounded operations, which CoreML does not contract for its ANE/GPU
/// kernels, so on [`ComputeUnits::All`] this is an argument about IEEE
/// semantics rather than a guarantee about the silicon; and it bounds only
/// argmax flips, not [`Self::worst_abs`] — the tail can contribute freely to
/// magnitude divergence. **Settling the trunk-vs-tail question needs a
/// re-conversion that emits the pre-tail activation as a second output**, on
/// both sides, compared on identical input; nothing short of that isolates it.
fn seg_divergence(coreml: &SegRun, onnx: &SegRun) -> SegDivergence {
  assert_eq!(
    coreml.num_frames, onnx.num_frames,
    "backends disagree on the per-chunk frame count ({} vs {})",
    coreml.num_frames, onnx.num_frames
  );
  assert_eq!(
    coreml.slabs.len(),
    onnx.slabs.len(),
    "backends ran a different number of chunks"
  );

  let mut d = SegDivergence {
    worst_abs: 0.0,
    frames: 0,
    flips: 0,
    flips_with_coreml_tie: 0,
    coreml_min: f64::INFINITY,
    coreml_max: f64::NEG_INFINITY,
    onnx_min: f64::INFINITY,
    onnx_max: f64::NEG_INFINITY,
    coreml_nonfinite: 0,
  };

  for (a, b) in coreml.slabs.iter().zip(&onnx.slabs) {
    for (ra, rb) in a
      .as_chunks::<POWERSET_CLASSES>()
      .0
      .iter()
      .zip(b.as_chunks::<POWERSET_CLASSES>().0)
    {
      d.frames += 1;
      for (&x, &y) in ra.iter().zip(rb) {
        if x.is_finite() {
          d.coreml_min = d.coreml_min.min(f64::from(x));
          d.coreml_max = d.coreml_max.max(f64::from(x));
        } else {
          d.coreml_nonfinite += 1;
        }
        d.onnx_min = d.onnx_min.min(f64::from(y));
        d.onnx_max = d.onnx_max.max(f64::from(y));
        d.worst_abs = d.worst_abs.max((f64::from(x) - f64::from(y)).abs());
      }
      if common::powerset_argmax(ra) != common::powerset_argmax(rb) {
        d.flips += 1;
        let max = ra.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if ra.iter().filter(|v| **v == max).count() > 1 {
          d.flips_with_coreml_tie += 1;
        }
      }
    }
  }
  d
}

// ══════════════════════════════════════════════════════════════════════
// The experiment
// ══════════════════════════════════════════════════════════════════════

/// One cell's result, for the printed table.
struct CellResult {
  seg: Backend,
  embed: Backend,
  outcome: Result<Vec<Seg>, diaric::offline::Error>,
  embed_s: f64,
}

impl CellResult {
  fn spk(&self) -> Option<usize> {
    self
      .outcome
      .as_ref()
      .ok()
      .map(|s| distinct_speakers(s).len())
  }
  fn der(&self, reference: &[Seg]) -> Option<Der> {
    self.outcome.as_ref().ok().map(|s| der_std(reference, s))
  }
}

/// **The shipping-configuration backend cross-product on clip 09.**
///
/// Runs all four `{ONNX, CoreML}` x `{ONNX, CoreML}` cells with the int8
/// embedder on [`ComputeUnits::All`] — the literal shipping default — prints
/// the full table with each cell's speaker count, standard-collar DER against
/// `reference.rttm` and clustering outcome, then asserts only:
///
/// 1. the audio and grid are the ones every cell was supposed to share;
/// 2. the two CORNERS reproduce the numbers the real pipelines pinned
///    elsewhere (harness validity — see the module doc);
/// 3. the localization VERDICT the table supports, pinned two-sided so that a
///    change in which stage is implicated fails here rather than silently
///    rewriting the record.
///
/// The report prints in full BEFORE any assertion fires, so a failure never
/// hides the numbers that explain it.
#[test]
#[ignore = "requires speakerkit models, dia parity fixtures + WeSpeaker ONNX (17 min of audio, 6 model passes)"]
fn shipping_config_backend_factorial() {
  let audio = fixtures_root().join(CLIP).join("clip_16k.wav");
  assert!(
    audio.exists(),
    "clip audio not found at {} (set DIA_PARITY_FIXTURES)",
    audio.display()
  );
  assert!(
    common::seg_path().exists() && coreml_embed_path().exists(),
    "need pyannote_segmentation.mlmodelc + wespeaker_v2.mlmodelc under {} (set \
     SPEAKERKIT_TEST_MODELS)",
    common::models_dir().display()
  );

  let samples = common::load_wav_16k_mono(&audio);
  let audio_fnv = common::fnv1a_f32(&samples);
  assert_eq!(
    samples.len(),
    CLIP_SAMPLES,
    "{CLIP}: decoded {} samples, pinned {CLIP_SAMPLES} — the audio identity changed",
    samples.len()
  );
  assert_eq!(
    audio_fnv, CLIP_AUDIO_FNV,
    "{CLIP}: audio content hash {audio_fnv} != pinned {CLIP_AUDIO_FNV}"
  );

  let reference = parse_rttm(&fixtures_root().join(CLIP).join("reference.rttm"));
  let ref_spk = distinct_speakers(&reference).len();
  assert_eq!(
    ref_spk, CLIP_REF_SPK,
    "{CLIP}: reference.rttm has {ref_spk} speakers, expected {CLIP_REF_SPK}"
  );

  let window = Options::new().window();
  let starts = chunk_starts(samples.len(), &window);
  let plda = diaric::plda::PldaTransform::new().expect("load community-1 PldaTransform (diaric)");

  println!(
    "\n╔══ backend factorial @ SHIPPING CONFIG — {CLIP} ══\n║ {:.2} s, {} samples, fnv1a={}\n║ \
     embedder: wespeaker_v2.mlmodelc (int8) | placement: {PLACEMENT:?} | {} chunks",
    samples.len() as f64 / 16_000.0,
    samples.len(),
    common::fnv_hex(audio_fnv),
    starts.len()
  );

  // ── Stage 1: each segmentation backend once, cached, so the four cells
  // differ ONLY in which conversion produced the tensors they consume.
  let seg_coreml = run_seg(Backend::CoreMl, &samples, &starts);
  let seg_onnx = run_seg(Backend::Onnx, &samples, &starts);
  assert_eq!(
    common::fnv1a_f32(&samples),
    audio_fnv,
    "the audio buffer changed under segmentation — comparison invalid"
  );
  println!(
    "║ seg: COREML/{PLACEMENT:?} {:.1} s, ONNX {:.1} s, {} frames/chunk",
    seg_coreml.elapsed_s, seg_onnx.elapsed_s, seg_coreml.num_frames
  );

  let div = seg_divergence(&seg_coreml, &seg_onnx);
  println!(
    "║\n║ segmentation divergence (COREML/{PLACEMENT:?} vs ONNX), {} frames:\n║   worst \
     |Δlog-prob| {:.4} | argmax flips {} ({:.4} %)\n║   of those flips, with an exact tie at the \
     CoreML max: {} — the ONLY ones the log-softmax tail could have caused\n║   observed range: \
     CoreML [{:.4}, {:.4}] ({} non-finite) | ONNX [{:.4}, {:.4}]",
    div.frames,
    div.worst_abs,
    div.flips,
    100.0 * div.flips as f64 / div.frames as f64,
    div.flips_with_coreml_tie,
    div.coreml_min,
    div.coreml_max,
    div.coreml_nonfinite,
    div.onnx_min,
    div.onnx_max,
  );

  // ── Stage 2: the four cells.
  let mut cells: Vec<CellResult> = Vec::with_capacity(4);
  for (seg_backend, seg_run) in [(Backend::Onnx, &seg_onnx), (Backend::CoreMl, &seg_coreml)] {
    for embed_arm in [EmbedArm::Onnx, EmbedArm::SHIPPING] {
      let embed_backend = embed_arm.backend();
      let mut embed = EmbedSide::load(embed_arm);
      let cell = assemble(seg_run, &mut embed, &samples, &starts, &plda);
      assert_eq!(
        common::fnv1a_f32(&samples),
        audio_fnv,
        "{}-seg + {}-emb: the audio buffer changed under the cell — comparison invalid",
        seg_backend.tag(),
        embed_backend.tag()
      );
      assert_eq!(
        cell.num_chunks,
        starts.len(),
        "{}-seg + {}-emb: chunk grid diverged",
        seg_backend.tag(),
        embed_backend.tag()
      );
      let outcome = cluster(&cell, &window, &plda);
      println!(
        "║ ran {:>6}-seg + {:>6}-emb ({:.1} s embed): {}",
        seg_backend.tag(),
        embed_backend.tag(),
        cell.embed_s,
        match &outcome {
          Ok(s) => format!("{} speakers", distinct_speakers(s).len()),
          Err(e) => format!("CLUSTERING FAILED — {e}"),
        }
      );
      cells.push(CellResult {
        seg: seg_backend,
        embed: embed_backend,
        outcome,
        embed_s: cell.embed_s,
      });
    }
  }

  // ── The table.
  println!(
    "║\n║ {:>11} | {:>11} | {:>4} | {:>9} | {:>9} | {:>9} | outcome",
    "segmentation", "embedding", "spk", "DER", "miss", "conf"
  );
  println!(
    "║ {:-<11}-+-{:-<11}-+-{:-<4}-+-{:-<9}-+-{:-<9}-+-{:-<9}-+--------",
    "", "", "", "", "", ""
  );
  for c in &cells {
    match (c.spk(), c.der(&reference)) {
      (Some(spk), Some(d)) => println!(
        "║ {:>11} | {:>11} | {spk:>4} | {:>8.4}% | {:>8.4}% | {:>8.4}% | clustered ({:.1} s embed)",
        c.seg.tag(),
        c.embed.tag(),
        d.der * 100.0,
        d.miss * 100.0,
        d.confusion * 100.0,
        c.embed_s,
      ),
      _ => println!(
        "║ {:>11} | {:>11} | {:>4} | {:>9} | {:>9} | {:>9} | Err: {}",
        c.seg.tag(),
        c.embed.tag(),
        "ERR",
        "-",
        "-",
        "-",
        c.outcome
          .as_ref()
          .expect_err("no speaker count implies Err"),
      ),
    }
  }
  for c in &cells {
    if let Some(d) = c.der(&reference) {
      println!(
        "║   {}",
        fmt_der(&format!("{}-seg + {}-emb", c.seg.tag(), c.embed.tag()), &d)
      );
    }
  }
  println!("╚══ reference (pyannote 4.0.4): {ref_spk} speakers\n");

  let corner = |seg: Backend, embed: Backend| -> &CellResult {
    cells
      .iter()
      .find(|c| c.seg == seg && c.embed == embed)
      .expect("every cell ran")
  };
  let cell = |seg: Backend, embed: Backend| -> Option<(usize, f64)> {
    let c = corner(seg, embed);
    Some((c.spk()?, c.der(&reference)?.der))
  };

  // ── The in-suite PRODUCTION control (module doc, "Harness validity"): the
  // identical retired-int8 configuration through the real
  // `FluidAudioSource::extract` (`Extractor::extract`) + public
  // `Extraction::diarize` path. The all-CoreML corner above is a hand
  // assembly; this run is what anchors it to the production pipeline now
  // that `parity_shipping_der`'s int8 arms are retired.
  let control_seg = SegmentModel::from_file_with(
    common::seg_path(),
    SegmentModelOptions::new().with_compute(PLACEMENT),
  )
  .expect("load pyannote_segmentation.mlmodelc (production control)");
  let control_embed = EmbedModel::from_file_with(
    coreml_embed_path(),
    EmbedModelOptions::new().with_compute(PLACEMENT),
  )
  .expect("load wespeaker_v2.mlmodelc (production control)");
  let control_ext = FluidAudioSource::with_options(control_seg, control_embed, Options::new())
    .extract(&samples)
    .expect("production-control extract");
  assert_eq!(
    common::fnv1a_f32(&samples),
    audio_fnv,
    "the audio buffer changed under the production control — comparison invalid"
  );
  let control_segs: Vec<Seg> = control_ext
    .diarize(&plda)
    .map(|out| {
      out
        .spans_slice()
        .iter()
        .map(|s| Seg {
          start: s.start(),
          end: s.start() + s.duration(),
          spk: s.cluster(),
        })
        .collect()
    })
    .expect(
      "the production control (FluidAudioSource, retired int8/All) must cluster — its record \
       ANSWERS with 5 of 8 speakers",
    );
  let control_spk = distinct_speakers(&control_segs).len();
  let control_der = der_std(&reference, &control_segs);
  println!(
    "║ production control (FluidAudioSource + Extraction::diarize, int8/All): {control_spk} \
     speakers, {:.4} % DER, {} err units",
    control_der.der * 100.0,
    control_der.err_units()
  );
  assert_eq!(
    control_spk, SHIPPING_CORNER_SPK,
    "the production control found {control_spk} speakers, not the retired shipping record's \
     {SHIPPING_CORNER_SPK} — the real pipeline no longer reproduces the record every corner and \
     hybrid here is read against"
  );
  assert!(
    (control_der.der - SHIPPING_CORNER_DER).abs() <= CORNER_DER_TOL,
    "the production control scored {:.4} % DER, off the retired shipping record's {:.4} % \
     (±{:.4} %)",
    control_der.der * 100.0,
    SHIPPING_CORNER_DER * 100.0,
    CORNER_DER_TOL * 100.0
  );
  // The equivalence gate: the hand assembly and the production pipeline must
  // agree — count exactly, DER and error mass to the stray-frame bands.
  let corner_cell = corner(Backend::CoreMl, Backend::CoreMl);
  let corner_der = corner_cell
    .der(&reference)
    .expect("the all-CoreML corner clustered (checked above)");
  assert_eq!(
    control_spk,
    corner_cell.spk().expect("corner clustered"),
    "the production control and the hand-assembled all-CoreML corner disagree on the speaker \
     count — `assemble` has diverged from `Extractor::extract`, and no hybrid cell can be read \
     until that is fixed"
  );
  assert!(
    (control_der.der - corner_der.der).abs() <= CORNER_DER_TOL,
    "production control at {:.4} % DER vs the hand-assembled corner's {:.4} % — beyond the \
     ±{:.4} % stray-frame band; `assemble` has diverged from `Extractor::extract`",
    control_der.der * 100.0,
    corner_der.der * 100.0,
    CORNER_DER_TOL * 100.0
  );
  assert!(
    control_der.err_units().abs_diff(corner_der.err_units()) <= ERR_UNITS_TOL,
    "production control at {} err units vs the hand-assembled corner's {} — beyond \
     ±{ERR_UNITS_TOL}; `assemble` has diverged from `Extractor::extract` at unit precision",
    control_der.err_units(),
    corner_der.err_units()
  );

  assert_factorial_verdict(&FactorialObserved {
    onnx_onnx: cell(Backend::Onnx, Backend::Onnx),
    coreml_onnx: cell(Backend::CoreMl, Backend::Onnx),
    onnx_coreml: cell(Backend::Onnx, Backend::CoreMl),
    coreml_coreml: cell(Backend::CoreMl, Backend::CoreMl),
  });
}

// ══════════════════════════════════════════════════════════════════════
// The verdict, as a pure function + its hermetic falsifiability guard
// ══════════════════════════════════════════════════════════════════════

/// The four cells' decision-relevant outcomes — `(speaker count, standard
/// DER)`, or `None` when that cell's clustering refused to answer. Extracted
/// from the measurement (or synthesized by
/// [`factorial_verdict_pins_every_cell`]) so [`assert_factorial_verdict`] is a
/// pure function both can call, exactly as `parity_shipping_der`'s
/// `Clip09Observed` is.
#[derive(Clone, Copy, Debug)]
struct FactorialObserved {
  onnx_onnx: Option<(usize, f64)>,
  coreml_onnx: Option<(usize, f64)>,
  onnx_coreml: Option<(usize, f64)>,
  coreml_coreml: Option<(usize, f64)>,
}

/// The MEASURED shipping-configuration cross-product on clip 09, pinned cell by
/// cell (Apple M1 Max, macOS 26.5 build 25F71, arm64; int8 `wespeaker_v2` and
/// the fp16-safe `pyannote_segmentation` re-conversion, both on
/// [`ComputeUnits::All`]):
///
/// ```text
/// segmentation | embedding | spk |      DER | conf
/// -------------+-----------+-----+----------+---------
///         ONNX |      ONNX |   8 |  0.0000% |  0.0000%   the dia-ort reference, reproduced
///         ONNX |    COREML |   5 | 16.5904% | 16.5904%   the shipping collapse, from the EMBEDDER alone
///       COREML |      ONNX |   9 |  1.3011% |  1.3011%   a DIFFERENT defect: one spurious speaker
///       COREML |    COREML |   5 | 16.5904% | 16.5904%   the shipping default
/// ```
///
/// # What this pins, and why each direction matters
///
/// - **Both corners** are harness-validity checks against the REAL pipelines:
///   all-ONNX must reproduce dia-ort's 8 speakers at 0.0000 %, and all-CoreML
///   must reproduce the retired int8/All shipping record (5 speakers,
///   16.5904 %, 11 999 confusion units) that
///   [`shipping_config_backend_factorial`]'s in-suite production control
///   re-measures through `FluidAudioSource::extract` + `Extraction::diarize`
///   on every run. The corner lands on that record to the printed precision,
///   which is what licenses reading the two hybrid cells at all.
/// - **`ONNX-seg + COREML-emb` is the finding.** Swapping ONLY the embedding
///   conversion, over dia's own reference segmentation, reproduces the
///   shipping failure exactly: 5 of 8 speakers at 16.5904 %, identical to the
///   all-CoreML corner. The recorded fp32/`CpuOnly` cross-product concluded
///   the embedder was *exonerated*; at the configuration this crate ships it
///   is sufficient on its own. Pinned so that conclusion cannot silently
///   revert.
/// - **`COREML-seg + ONNX-emb` is a different defect, not this one.** The
///   segmentation conversion alone OVERcounts by one (9 speakers, 1.3011 %) —
///   the same "spurious extra cluster" signature that, with the fp32 embedder
///   on `CpuOnly`, tripped diaric's `AmbiguousAliveCluster` bail-out. It is
///   real, it is an order of magnitude smaller, and it is masked in the
///   shipping arm. Pinned so it is not conflated with the collapse again.
///
/// # What it does NOT pin
///
/// The factor varied here is the BACKEND (dia's ONNX vs this crate's CoreML),
/// not precision and not placement. `ONNX-seg + COREML-emb` therefore
/// implicates "the CoreML embedding path as shipped" — int8-palettized
/// artifact, `All` placement, that conversion — as one bundle. Which of those
/// three properties carries the failure is NOT isolated here.
///
/// [`embedding_precision_x_placement`] is the experiment that isolates them,
/// and it re-measures this very cell as its own cell B. Its finding, in one
/// line: the conversion carries none of it (fp32 on `CpuOnly` is frame-perfect
/// against dia-ort), the palettization carries 2 of the 3 lost speakers, and
/// the `All` placement carries 1 — so this cell's "the CoreML embedding path"
/// must not be read as "the CoreML embedding conversion".
///
/// # Panics
/// On any divergence from the pinned record, naming the cell and what the
/// divergence means for the attribution recorded in `model_io.rs`.
fn assert_factorial_verdict(o: &FactorialObserved) {
  // ── Corner 1: the harness reproduces the dia-ort reference.
  let (onnx_spk, onnx_der) = o.onnx_onnx.unwrap_or_else(|| {
    panic!(
      "ONNX-seg + ONNX-emb did not cluster. The reference corner must reproduce dia-ort; that is \
       a harness failure, not a finding about the CoreML path — no other cell can be trusted."
    )
  });
  assert_eq!(
    onnx_spk, REFERENCE_CORNER_SPK,
    "ONNX-seg + ONNX-emb found {onnx_spk} speakers, not dia-ort's pinned {REFERENCE_CORNER_SPK} — \
     the harness does not reproduce the reference, so no cell it produced can be trusted"
  );
  assert!(
    onnx_der <= CORNER_DER_TOL,
    "ONNX-seg + ONNX-emb scored {:.4} % DER against reference.rttm; dia-ort is frame-perfect on \
     this clip (0.0000 %), so the harness has diverged from the reference pipeline",
    onnx_der * 100.0
  );

  // ── Corner 2: the harness reproduces `Extractor::extract` at the retired
  // int8/All configuration, to the record the in-suite production control
  // re-measures.
  let (ship_spk, ship_der) = o.coreml_coreml.unwrap_or_else(|| {
    panic!(
      "COREML-seg + COREML-emb did not cluster. The in-suite production control pins this exact \
       configuration as ANSWERING with {SHIPPING_CORNER_SPK} of 8 speakers; either that path \
       regressed to 'cannot answer at all' or this harness diverged from `Extractor::extract`."
    )
  });
  assert_eq!(
    ship_spk, SHIPPING_CORNER_SPK,
    "COREML-seg + COREML-emb found {ship_spk} speakers; the in-suite production control pins the \
     identical artifacts/placement/decode at {SHIPPING_CORNER_SPK}. This corner is the harness's \
     equivalence proof against `Extractor::extract` — investigate before reading any other cell."
  );
  assert!(
    (ship_der - SHIPPING_CORNER_DER).abs() <= CORNER_DER_TOL,
    "COREML-seg + COREML-emb scored {:.4} % DER; the in-suite production control pins the \
     identical configuration at {:.4} % (±{:.4} %)",
    ship_der * 100.0,
    SHIPPING_CORNER_DER * 100.0,
    CORNER_DER_TOL * 100.0
  );

  // ── THE FINDING: the embedding conversion alone reproduces the collapse.
  let (emb_spk, emb_der) = o.onnx_coreml.unwrap_or_else(|| {
    panic!(
      "ONNX-seg + COREML-emb did not cluster. It is pinned as REPRODUCING the shipping collapse \
       (an answer of {EMBED_ONLY_SPK} of 8 speakers), not as failing to answer — a change of \
       failure MODE is a new finding; re-measure before rewriting `model_io.rs`."
    )
  });
  assert_eq!(
    emb_spk, EMBED_ONLY_SPK,
    "ONNX-seg + COREML-emb found {emb_spk} speakers, not the pinned {EMBED_ONLY_SPK}. This cell \
     is the whole reason `model_io.rs` attributes the clip-09 collapse at the SHIPPING \
     configuration to the embedding conversion rather than the segmentation one. If it now \
     recovers all {REFERENCE_CORNER_SPK}, that attribution is stale and must be rewritten — do \
     NOT delete this assertion."
  );
  assert!(
    (emb_der - SHIPPING_CORNER_DER).abs() <= CORNER_DER_TOL,
    "ONNX-seg + COREML-emb scored {:.4} % DER, off the pinned {:.4} % (±{:.4} %). Its landing on \
     the SAME DER as the all-CoreML corner is what says the embedding conversion accounts for the \
     shipping collapse rather than merely contributing to it.",
    emb_der * 100.0,
    SHIPPING_CORNER_DER * 100.0,
    CORNER_DER_TOL * 100.0
  );

  // ── The segmentation conversion's own, DIFFERENT defect.
  let (seg_spk, seg_der) = o.coreml_onnx.unwrap_or_else(|| {
    panic!(
      "COREML-seg + ONNX-emb did not cluster. It is pinned as ANSWERING with {SEG_ONLY_SPK} \
       speakers (one spurious cluster); a refusal to answer is the fp32/`CpuOnly` symptom \
       appearing at a placement where it was not seen before — a new finding, not a flake."
    )
  });
  assert_eq!(
    seg_spk, SEG_ONLY_SPK,
    "COREML-seg + ONNX-emb found {seg_spk} speakers, not the pinned {SEG_ONLY_SPK}. The \
     segmentation conversion's own defect on this clip is an OVERcount by one, an order of \
     magnitude smaller than the collapse; if that changed, `model_io.rs`'s two-defect account is \
     stale."
  );
  assert!(
    (seg_der - SEG_ONLY_DER).abs() <= CORNER_DER_TOL,
    "COREML-seg + ONNX-emb scored {:.4} % DER, off the pinned {:.4} % (±{:.4} %)",
    seg_der * 100.0,
    SEG_ONLY_DER * 100.0,
    CORNER_DER_TOL * 100.0
  );
}

/// [`assert_factorial_verdict`] pins EVERY cell, in BOTH directions — proven
/// here hermetically (no models, no fixtures, no audio): the measured record
/// passes, and every single-cell perturbation fails. Without this, a cell could
/// silently go unpinned and a real change in the localization would pass green
/// — the same falsifiability contract `parity_shipping_der`'s
/// `clip09_known_defect_pins_every_field` carries.
#[test]
fn factorial_verdict_pins_every_cell() {
  let good = FactorialObserved {
    onnx_onnx: Some((REFERENCE_CORNER_SPK, 0.0)),
    coreml_onnx: Some((SEG_ONLY_SPK, SEG_ONLY_DER)),
    onnx_coreml: Some((EMBED_ONLY_SPK, SHIPPING_CORNER_DER)),
    coreml_coreml: Some((SHIPPING_CORNER_SPK, SHIPPING_CORNER_DER)),
  };
  assert_factorial_verdict(&good);

  let fails = |o: FactorialObserved, what: &str| {
    assert!(
      std::panic::catch_unwind(move || assert_factorial_verdict(&o)).is_err(),
      "assert_factorial_verdict accepted a record with {what} — that cell is NOT pinned"
    );
  };

  // Every cell must be pinned on BOTH axes (count and DER) and against a
  // refusal to cluster.
  for (mutate, what) in [
    (
      (|o: &mut FactorialObserved| o.onnx_onnx = Some((9, 0.0))) as fn(&mut FactorialObserved),
      "a moved all-ONNX speaker count",
    ),
    (
      |o: &mut FactorialObserved| o.onnx_onnx = Some((REFERENCE_CORNER_SPK, 0.05)),
      "a moved all-ONNX DER",
    ),
    (
      |o: &mut FactorialObserved| o.onnx_onnx = None,
      "an all-ONNX clustering refusal",
    ),
    (
      |o: &mut FactorialObserved| o.coreml_coreml = Some((6, SHIPPING_CORNER_DER)),
      "a moved all-CoreML speaker count",
    ),
    (
      |o: &mut FactorialObserved| o.coreml_coreml = Some((SHIPPING_CORNER_SPK, 0.17)),
      "a moved all-CoreML DER",
    ),
    (
      |o: &mut FactorialObserved| o.coreml_coreml = None,
      "an all-CoreML clustering refusal",
    ),
    (
      |o: &mut FactorialObserved| o.onnx_coreml = Some((REFERENCE_CORNER_SPK, 0.0)),
      "an embedder-only cell that recovers every speaker",
    ),
    (
      |o: &mut FactorialObserved| o.onnx_coreml = Some((EMBED_ONLY_SPK, 0.17)),
      "a moved embedder-only DER",
    ),
    (
      |o: &mut FactorialObserved| o.onnx_coreml = None,
      "an embedder-only clustering refusal",
    ),
    (
      |o: &mut FactorialObserved| o.coreml_onnx = Some((REFERENCE_CORNER_SPK, SEG_ONLY_DER)),
      "a moved segmenter-only speaker count",
    ),
    (
      |o: &mut FactorialObserved| o.coreml_onnx = Some((SEG_ONLY_SPK, 0.05)),
      "a moved segmenter-only DER",
    ),
    (
      |o: &mut FactorialObserved| o.coreml_onnx = None,
      "a segmenter-only clustering refusal",
    ),
  ] {
    let mut o = good;
    mutate(&mut o);
    fails(o, what);
  }
}

// ══════════════════════════════════════════════════════════════════════
// The disambiguating experiment: precision x placement, reference
// segmentation held fixed
// ══════════════════════════════════════════════════════════════════════

/// The five embedding arms, in report order. Reference segmentation is held
/// fixed for all of them, so the ONLY thing that varies down this list is a
/// property of the embedding path.
///
/// | cell | arm | what its outcome establishes |
/// |---|---|---|
/// | A | ONNX fp32 on `ort`'s CPU EP | the reference; harness validity |
/// | B | CoreML int8 on `All` | the shipping bundle; reproduces `shipping_config_backend_factorial`'s `ONNX-seg + COREML-emb` cell |
/// | C | CoreML **fp32** on `All` | placement + conversion, quantization removed |
/// | D | CoreML int8 on **`CpuOnly`** | quantization + conversion, placement removed |
/// | E | CoreML fp32 on `CpuOnly` | the conversion alone, both other factors removed |
const PRECISION_PLACEMENT_ARMS: [EmbedArm; 5] = [
  EmbedArm::Onnx,
  EmbedArm::SHIPPING,
  EmbedArm::CoreMl(CoreMl {
    precision: Precision::Fp32,
    placement: ComputeUnits::All,
  }),
  EmbedArm::CoreMl(CoreMl {
    precision: Precision::Int8,
    placement: ComputeUnits::CpuOnly,
  }),
  EmbedArm::CoreMl(CoreMl {
    precision: Precision::Fp32,
    placement: ComputeUnits::CpuOnly,
  }),
];

/// One embedding arm's measured result over the fixed reference segmentation.
struct ArmResult {
  cell: char,
  arm: EmbedArm,
  outcome: Result<Vec<Seg>, diaric::offline::Error>,
  embed_s: f64,
  /// The arm's `[num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM]` raw embeddings,
  /// kept so [`embed_agreement`] can report how far each arm's embedding SPACE
  /// moved — the intermediate the DER table's outcomes are downstream of.
  embeddings: Vec<f32>,
}

impl ArmResult {
  fn spk(&self) -> Option<usize> {
    self
      .outcome
      .as_ref()
      .ok()
      .map(|s| distinct_speakers(s).len())
  }
  fn der(&self, reference: &[Seg]) -> Option<Der> {
    self.outcome.as_ref().ok().map(|s| der_std(reference, s))
  }
}

/// How far one arm's embeddings sit from the ONNX reference arm's, over the
/// `(chunk, slot)` rows both arms actually produced.
///
/// Reported, not asserted: it is the mechanism-side companion to the DER table,
/// and its job is to say whether two arms that land on the same DER got there
/// through the same size of embedding perturbation. The DER outcomes are what
/// the verdict is pinned on.
struct EmbedAgreement {
  /// Rows non-zero on BOTH sides — the ones a cosine is defined on.
  rows: usize,
  mean_cos: f64,
  min_cos: f64,
  /// Rows non-zero on exactly one side: the row predicate dropped the
  /// `(chunk, slot)` on one arm and kept it on the other. A cosine
  /// is undefined against a zeroed row, so these are counted rather than
  /// folded in — and they are themselves a divergence, since a dropped slot
  /// also zeroes that slot's segmentation column.
  drop_disagreements: usize,
}

/// Compares one arm's `raw_embeddings` against the reference arm's, row by row.
///
/// An all-zero row means the slot was never embedded: either the
/// overlap-exclusion rule skipped it (identical across arms — the plans derive
/// from the ONE fixed reference segmentation) or the row predicate dropped it
/// (which CAN differ per arm). Rows zero on both sides carry no information.
fn embed_agreement(arm: &[f32], reference: &[f32]) -> EmbedAgreement {
  assert_eq!(
    arm.len(),
    reference.len(),
    "embedding tensors have different lengths ({} vs {})",
    arm.len(),
    reference.len()
  );
  let mut d = EmbedAgreement {
    rows: 0,
    mean_cos: 0.0,
    min_cos: f64::INFINITY,
    drop_disagreements: 0,
  };
  let mut total = 0.0f64;
  for (a, r) in arm
    .as_chunks::<EMBEDDING_DIM>()
    .0
    .iter()
    .zip(reference.as_chunks::<EMBEDDING_DIM>().0)
  {
    let a_live = a.iter().any(|v| *v != 0.0);
    let r_live = r.iter().any(|v| *v != 0.0);
    match (a_live, r_live) {
      (true, true) => {
        let c = common::cosine(a, r);
        d.rows += 1;
        total += c;
        d.min_cos = d.min_cos.min(c);
      }
      (false, false) => {}
      _ => d.drop_disagreements += 1,
    }
  }
  if d.rows > 0 {
    d.mean_cos = total / d.rows as f64;
  } else {
    d.min_cos = f64::NAN;
  }
  d
}

/// **Which property of the CoreML embedding path carries the clip-09 collapse:
/// the int8 palettization, the `ComputeUnits::All` placement, or the conversion
/// itself?**
///
/// [`shipping_config_backend_factorial`] established that swapping ONLY the
/// embedding conversion, over dia's own reference segmentation, reproduces the
/// shipping collapse exactly (5 of 8 speakers, 16.5904 %). But the factor it
/// varied is the whole BACKEND, so three properties were implicated as one
/// bundle. This suite separates them: same harness, same clip, same reference
/// segmentation slabs, five embedding arms.
///
/// The DER verdict is [`assert_precision_placement_verdict`]. The report also
/// prints an unpinned embedding-space companion ([`embed_agreement`]), and it
/// is worth reading against the verdict because the two disagree about which
/// factor is "bigger". Measured on the same run as the pinned table, mean and
/// minimum cosine against cell A over all 2 114 `(chunk, slot)` rows, with zero
/// row-predicate drop disagreements on any arm:
///
/// ```text
/// B  CoreML int8 / All        mean 0.985777 | min 0.042112
/// C  CoreML fp32 / All        mean 0.987201 | min 0.035367
/// D  CoreML int8 / CpuOnly    mean 0.998545 | min 0.963721
/// E  CoreML fp32 / CpuOnly    mean 1.000000 | min 1.000000
/// ```
///
/// Two things to take from it. First, cell E is not merely "clean at the DER
/// level": the CoreML embedding conversion on `CpuOnly` agrees with dia's fp32
/// ONNX to 1.000000 on EVERY row, minimum included — the conversion is
/// numerically the same function, which is the strongest form the exoneration
/// could take.
///
/// Second, **the embedding perturbation is anti-correlated with the clustering
/// damage here.** The `All` placement moves the embeddings roughly 9x further
/// in mean cosine distance than the palettization does (~0.013 vs ~0.0015) and
/// drags some rows to near-orthogonality (min cosine 0.035), yet it costs 1
/// speaker and 1 839 error units; int8's much smaller, much more uniform
/// perturbation costs 2 speakers and 11 835. So the size of the embedding
/// perturbation does not predict the clustering outcome. The KIND of
/// perturbation is what matters — the int8-era `parity_shipping_der` module
/// doc argued that shape too, but with the roles reversed (quantization was
/// expected to be the benign, roughly isotropic one; on this clip it is the
/// harmful one), and that rationale was retired with the artifact. WHAT
/// structure the frozen community-1 LDA/PLDA basis is sensitive to is
/// established by [`quantization_error_structure`]: the palettization delta
/// is a coherent shared displacement that compresses between-cluster margins
/// in that basis, while the placement scatter averages out.
///
/// **The `embed_s` column is not a latency measurement.** It is wall time for
/// one un-warmed pass, so it carries each arm's cold CoreML/ANE specialization,
/// and it is not stable: across two runs of this suite the identical int8/`All`
/// arm measured 31.7 s and 51.9 s, and fp32/`CpuOnly` 61.1 s and 100.3 s. It is
/// printed to show an arm ran and roughly how long to expect, nothing else. The
/// int8-vs-fp32 cost question is answered by
/// `parity_shipping_der::shipping_embedder_cost_int8_vs_fp32`, which warms up
/// first and times the whole extract.
///
/// The report prints in full BEFORE any assertion fires.
#[test]
#[ignore = "requires speakerkit models, dia parity fixtures + WeSpeaker ONNX (17 min of audio, 1 seg + 5 embed passes)"]
fn embedding_precision_x_placement() {
  let audio = fixtures_root().join(CLIP).join("clip_16k.wav");
  assert!(
    audio.exists(),
    "clip audio not found at {} (set DIA_PARITY_FIXTURES)",
    audio.display()
  );
  assert!(
    common::embed_path().exists() && common::embed_fp32_path().exists(),
    "need BOTH wespeaker_v2.mlmodelc (int8) and wespeaker.mlmodelc (fp32) under {} (set \
     SPEAKERKIT_TEST_MODELS) — this experiment's whole point is varying the precision, so a \
     missing artifact must fail loudly rather than be substituted",
    common::models_dir().display()
  );

  let samples = common::load_wav_16k_mono(&audio);
  let audio_fnv = common::fnv1a_f32(&samples);
  assert_eq!(
    samples.len(),
    CLIP_SAMPLES,
    "{CLIP}: decoded {} samples, pinned {CLIP_SAMPLES} — the audio identity changed",
    samples.len()
  );
  assert_eq!(
    audio_fnv, CLIP_AUDIO_FNV,
    "{CLIP}: audio content hash {audio_fnv} != pinned {CLIP_AUDIO_FNV}"
  );

  let reference = parse_rttm(&fixtures_root().join(CLIP).join("reference.rttm"));
  let ref_spk = distinct_speakers(&reference).len();
  assert_eq!(
    ref_spk, CLIP_REF_SPK,
    "{CLIP}: reference.rttm has {ref_spk} speakers, expected {CLIP_REF_SPK}"
  );

  let window = Options::new().window();
  let starts = chunk_starts(samples.len(), &window);
  let plda = diaric::plda::PldaTransform::new().expect("load community-1 PldaTransform (diaric)");

  println!(
    "\n╔══ embedding precision x placement — {CLIP} ══\n║ {:.2} s, {} samples, fnv1a={}\n║ \
     segmentation: dia ONNX (the REFERENCE), held fixed for every arm | {} chunks",
    samples.len() as f64 / 16_000.0,
    samples.len(),
    common::fnv_hex(audio_fnv),
    starts.len()
  );

  // ── The reference segmentation, computed ONCE. Every arm consumes these
  // exact slabs, so nothing downstream of the embedder can differ for a
  // segmentation reason.
  let seg = run_seg(Backend::Onnx, &samples, &starts);
  assert_eq!(
    common::fnv1a_f32(&samples),
    audio_fnv,
    "the audio buffer changed under segmentation — comparison invalid"
  );
  println!(
    "║ seg: ONNX {:.1} s, {} frames/chunk\n║",
    seg.elapsed_s, seg.num_frames
  );

  let mut arms: Vec<ArmResult> = Vec::with_capacity(PRECISION_PLACEMENT_ARMS.len());
  for (i, arm) in PRECISION_PLACEMENT_ARMS.into_iter().enumerate() {
    let cell = char::from(b'A' + u8::try_from(i).expect("five arms"));
    let mut embed = EmbedSide::load(arm);
    let assembled = assemble(&seg, &mut embed, &samples, &starts, &plda);
    assert_eq!(
      common::fnv1a_f32(&samples),
      audio_fnv,
      "cell {cell} ({}): the audio buffer changed under the arm — comparison invalid",
      arm.label()
    );
    assert_eq!(
      assembled.num_chunks,
      starts.len(),
      "cell {cell} ({}): chunk grid diverged",
      arm.label()
    );
    let outcome = cluster(&assembled, &window, &plda);
    println!(
      "║ ran {cell}: {:<24} ({:>5.1} s embed): {}",
      arm.label(),
      assembled.embed_s,
      match &outcome {
        Ok(s) => format!("{} speakers", distinct_speakers(s).len()),
        Err(e) => format!("CLUSTERING FAILED — {e}"),
      }
    );
    arms.push(ArmResult {
      cell,
      arm,
      outcome,
      embed_s: assembled.embed_s,
      embeddings: assembled.raw_embeddings,
    });
  }

  // ── The table.
  println!(
    "║\n║ {:>4} | {:>24} | {:>4} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | outcome",
    "cell", "embedding arm", "spk", "DER", "miss", "fa", "conf", "err units"
  );
  println!(
    "║ {:-<4}-+-{:-<24}-+-{:-<4}-+-{:-<9}-+-{:-<9}-+-{:-<9}-+-{:-<9}-+-{:-<9}-+--------",
    "", "", "", "", "", "", "", ""
  );
  for a in &arms {
    match (a.spk(), a.der(&reference)) {
      (Some(spk), Some(d)) => println!(
        "║ {:>4} | {:>24} | {spk:>4} | {:>8.4}% | {:>8.4}% | {:>8.4}% | {:>8.4}% | {:>9} | \
         clustered ({:.1} s embed)",
        a.cell,
        a.arm.label(),
        d.der * 100.0,
        d.miss * 100.0,
        d.fa * 100.0,
        d.confusion * 100.0,
        d.err_units(),
        a.embed_s,
      ),
      _ => println!(
        "║ {:>4} | {:>24} | {:>4} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | Err: {}",
        a.cell,
        a.arm.label(),
        "ERR",
        "-",
        "-",
        "-",
        "-",
        "-",
        a.outcome
          .as_ref()
          .expect_err("no speaker count implies Err"),
      ),
    }
  }
  for a in &arms {
    if let Some(d) = a.der(&reference) {
      println!(
        "║   {}",
        fmt_der(&format!("{} {}", a.cell, a.arm.label()), &d)
      );
    }
  }

  // ── The embedding space each arm handed to clustering, against cell A's.
  // Reported, not pinned: it says whether arms that land on the same DER got
  // there through the same size of perturbation.
  let reference_embeddings: &[f32] = &arms
    .iter()
    .find(|a| a.arm == EmbedArm::Onnx)
    .expect("cell A ran")
    .embeddings;
  println!("║\n║ embedding agreement vs cell A (ONNX fp32), per (chunk, slot) row:");
  for a in arms.iter().filter(|a| a.arm != EmbedArm::Onnx) {
    let g = embed_agreement(&a.embeddings, reference_embeddings);
    println!(
      "║   {} {:<24} mean cos {:.6} | min cos {:.6} | {} rows | {} row-predicate drop \
       disagreements",
      a.cell,
      a.arm.label(),
      g.mean_cos,
      g.min_cos,
      g.rows,
      g.drop_disagreements,
    );
  }
  println!("╚══ reference (pyannote 4.0.4): {ref_spk} speakers\n");

  let at = |arm: EmbedArm| -> Option<CellOutcome> {
    let a = arms
      .iter()
      .find(|a| a.arm == arm)
      .unwrap_or_else(|| panic!("every arm ran ({})", arm.label()));
    let d = a.der(&reference)?;
    Some(CellOutcome {
      spk: a.spk()?,
      der: d.der,
      err_units: d.err_units(),
    })
  };

  assert_precision_placement_verdict(&PrecisionPlacementObserved {
    onnx_cpu: at(EmbedArm::Onnx),
    int8_all: at(EmbedArm::SHIPPING),
    fp32_all: at(EmbedArm::CoreMl(CoreMl {
      precision: Precision::Fp32,
      placement: ComputeUnits::All,
    })),
    int8_cpu: at(EmbedArm::CoreMl(CoreMl {
      precision: Precision::Int8,
      placement: ComputeUnits::CpuOnly,
    })),
    fp32_cpu: at(EmbedArm::CoreMl(CoreMl {
      precision: Precision::Fp32,
      placement: ComputeUnits::CpuOnly,
    })),
  });
}

/// One arm's decision-relevant outcome. `err_units` is the DER numerator in raw
/// speaker-frames — the axis on which "0" means *not one scored frame differs*
/// rather than "rounds to 0.0000 %", which is the whole strength of cells A
/// and E.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CellOutcome {
  spk: usize,
  der: f64,
  err_units: u64,
}

/// The five arms' outcomes, or `None` where an arm's clustering refused to
/// answer. Extracted from the measurement (or synthesized by
/// [`precision_placement_verdict_pins_every_cell`]) so
/// [`assert_precision_placement_verdict`] is a pure function both can call.
#[derive(Clone, Copy, Debug)]
struct PrecisionPlacementObserved {
  onnx_cpu: Option<CellOutcome>,
  int8_all: Option<CellOutcome>,
  fp32_all: Option<CellOutcome>,
  int8_cpu: Option<CellOutcome>,
  fp32_cpu: Option<CellOutcome>,
}

/// `CoreML fp32 / All` (cell C) — the placement + conversion, quantization
/// removed.
const CELL_C_SPK: usize = 7;
const CELL_C_DER: f64 = 0.025_427;
const CELL_C_ERR_UNITS: u64 = 1_839;

/// `CoreML int8 / CpuOnly` (cell D) — the quantization + conversion, placement
/// removed.
const CELL_D_SPK: usize = 6;
const CELL_D_DER: f64 = 0.163_636;
const CELL_D_ERR_UNITS: u64 = 11_835;

/// Cell B's pinned error units — `parity_shipping_der`'s retired int8/All
/// record in raw speaker-frames.
const CELL_B_ERR_UNITS: u64 = 11_999;

/// Band on the B/C/D error-unit pins. Each unit is one 10 ms scored frame; a
/// cross-CoreML-build stray flip moves a boundary by a few frames, so ±10
/// absorbs strays at far finer precision than the DER bands alone hold
/// (±[`CORNER_DER_TOL`] admits ±36 units at this clip's 72 325 reference
/// units).
///
/// What is GUARDED is each CELL's unit count. Derived cross-cell figures —
/// the 11 999 − 11 835 = 164-unit placement contribution, the 98.6 %
/// quantization share — are REPORTED measurements of the pinned run, not
/// separately asserted invariants: within these cell bands the difference
/// can drift by up to ±20 units and the share by ~±0.15 pp, and pinning the
/// relations too would only push the same gap one level up. Every place this
/// record cites 164 or 98.6 % labels them as measured values of that run.
const ERR_UNITS_TOL: u64 = 10;

/// `CoreML fp32 / CpuOnly` (cell E) — the conversion ALONE. It reproduces the
/// dia-ort reference exactly, which is why its pin is `err_units == 0` rather
/// than a DER band.
const CELL_E_SPK: usize = 8;

/// **The measured precision x placement record on clip 09**, reference
/// segmentation held fixed (Apple M1 Max, macOS 26.5 build 25F71, arm64):
///
/// ```text
/// cell | embedding arm            | spk |      DER |     conf | err units
/// -----+--------------------------+-----+----------+----------+----------
///    A | ONNX fp32 / ort CPU EP   |   8 |  0.0000% |  0.0000% |         0
///    B | CoreML int8 / All        |   5 | 16.5904% | 16.5904% |     11999
///    C | CoreML fp32 / All        |   7 |  2.5427% |  2.5427% |      1839
///    D | CoreML int8 / CpuOnly    |   6 | 16.3636% | 16.3636% |     11835
///    E | CoreML fp32 / CpuOnly    |   8 |  0.0000% |  0.0000% |         0
/// ```
///
/// Laid out as the 2x2 it is, speakers / err units:
///
/// ```text
///          | CpuOnly        | All
/// ---------+----------------+---------------
///     fp32 | 8 /      0 (E) | 7 /  1 839 (C)
///     int8 | 6 / 11 835 (D) | 5 / 11 999 (B)
/// ```
///
/// # What this establishes
///
/// - **The conversion itself is exonerated.** Cell E — the CoreML embedding
///   conversion with BOTH other factors removed — reproduces dia-ort's answer
///   frame-perfectly: 8 of 8 speakers and `err_units == 0`, not one
///   collar-scored speaker-frame different from the ONNX reference. Whatever
///   the clip-09 embedding defect is, it is not "speakerkit converted this
///   graph wrong".
/// - **Quantization is the dominant term, and it is placement-independent.**
///   Holding placement fixed, int8 costs exactly 2 speakers at BOTH
///   placements (E 8 -> D 6 on `CpuOnly`; C 7 -> B 5 on `All`); the measured
///   run's cell values put 11 835 of cell B's 11 999 error units — 98.6 % —
///   on `CpuOnly` alone (derived figures REPORTED from the pinned run;
///   [`ERR_UNITS_TOL`]'s doc states what is guarded vs reported).
/// - **Placement is a real but minority term, and it too is
///   precision-independent.** Holding precision fixed, `All` costs exactly 1
///   speaker at BOTH precisions (E 8 -> C 7 on fp32; D 6 -> B 5 on int8);
///   the measured run's error-unit cost is 1 839 in the fp32 arm (E 0 ->
///   C 1 839) and 164 in the int8 arm (D 11 835 -> B 11 999) — the latter
///   1.4 % of the shipping arm's total, against quantization's 98.6 %
///   (same reported-not-asserted status).
/// - **Neither factor alone reproduces the shipping collapse.** Cell B is 5
///   speakers at 16.5904 %; the best either single factor manages is D's 6
///   speakers at 16.3636 % (nearly all the DER, one speaker short) or C's 7
///   speakers at 2.5427 % (one speaker short of the reference, and 6.5x
///   smaller in DER). On speaker count the two factors are cleanly ADDITIVE:
///   -2 for quantization, -1 for placement, at every level of the other.
///
/// # What it does NOT establish
///
/// - **It is one clip, one host, one segmentation backend.** Every arm here
///   runs over dia's ONNX reference segmentation, which is NOT what this
///   crate ships. Real-pipeline numbers (CoreML segmentation) land nearby
///   but not identically — the retired int8-era record had the CoreML-seg
///   int8/`CpuOnly` composition at 6 speakers / 16.4590 % against cell D's
///   16.3636 % here — so no cell in this table doubles as a real-pipeline
///   number.
/// - **It does not price the remedy; `parity_shipping_der` does.** That fp32
///   recovers speakers over the REFERENCE segmentation says nothing by
///   itself about the shipped composition. The remedy's real-pipeline
///   pricing is that suite's job today: it resolves the production fp32
///   embedder and gates `seg@All + fp32@All` (with both CPU placement
///   controls) on all four clips — 06 / 14 / 10 under [`gate`], clip 09
///   under its pinned record.
/// - **It does not say WHY int8 costs two speakers** —
///   [`quantization_error_structure`] does. Palettization changes the
///   weights, not the graph: the two artifacts' op histograms are identical
///   apart from 38 `constexpr_lut_to_dense` decompressions replacing 36
///   `const` weight tensors (and one dropped no-op `identity`). Which LAYER's
///   perturbation the frozen LDA/PLDA basis is sensitive to is still
///   unmeasured; the probe pins the embedding-space mechanism, not a
///   per-layer attribution.
///
/// # Panics
/// On any divergence from the pinned record, naming the cell and what the
/// divergence means for the attribution recorded in `model_io.rs`.
fn assert_precision_placement_verdict(o: &PrecisionPlacementObserved) {
  // ── Cell A: the harness reproduces the dia-ort reference.
  let a = o.onnx_cpu.unwrap_or_else(|| {
    panic!(
      "cell A (ONNX fp32 / ort CPU EP) did not cluster. The reference arm must reproduce dia-ort; \
       that is a harness failure, not a finding about the CoreML path — no other cell can be \
       trusted."
    )
  });
  assert_eq!(
    a.spk, REFERENCE_CORNER_SPK,
    "cell A found {} speakers, not dia-ort's pinned {REFERENCE_CORNER_SPK} — the harness does not \
     reproduce the reference, so no cell it produced can be trusted",
    a.spk
  );
  assert_eq!(
    a.err_units, 0,
    "cell A scored {} error units against reference.rttm; dia-ort is frame-perfect on this clip, \
     so the harness has diverged from the reference pipeline",
    a.err_units
  );
  assert!(
    a.der == 0.0,
    "cell A carries a DER of {:.4} % alongside zero error units — in a real measurement the two \
     derive from one scoring pass and cannot disagree; this record is internally inconsistent",
    a.der * 100.0
  );

  // ── Cell B: this suite's arm B is `shipping_config_backend_factorial`'s
  // `ONNX-seg + COREML-emb` cell, re-measured. Same segmentation backend, same
  // artifact, same placement — so it lands on that suite's pin or the two
  // suites are not measuring the same thing.
  let b = o.int8_all.unwrap_or_else(|| {
    panic!(
      "cell B (CoreML int8 / All) did not cluster. It is pinned as REPRODUCING the shipping \
       collapse (an answer of {EMBED_ONLY_SPK} of 8 speakers), which is also \
       `shipping_config_backend_factorial`'s pinned `ONNX-seg + COREML-emb` cell — a change of \
       failure MODE is a new finding, not a flake."
    )
  });
  assert_eq!(
    b.spk, EMBED_ONLY_SPK,
    "cell B found {} speakers, not the {EMBED_ONLY_SPK} that \
     `shipping_config_backend_factorial` pins for the identical arm. These two suites share a \
     cell on purpose; if it moved, they are no longer measuring the same configuration and this \
     experiment's baseline is gone.",
    b.spk
  );
  assert!(
    (b.der - SHIPPING_CORNER_DER).abs() <= CORNER_DER_TOL,
    "cell B scored {:.4} % DER; the identical arm is pinned at {:.4} % (±{:.4} %) by \
     `shipping_config_backend_factorial`'s corner and its retired-int8 production control",
    b.der * 100.0,
    SHIPPING_CORNER_DER * 100.0,
    CORNER_DER_TOL * 100.0
  );
  assert!(
    b.err_units.abs_diff(CELL_B_ERR_UNITS) <= ERR_UNITS_TOL,
    "cell B scored {} error units, off the pinned {CELL_B_ERR_UNITS} (±{ERR_UNITS_TOL}) — the \
     collapse's error mass moved at unit precision; the reported figures derived from it \
     (see ERR_UNITS_TOL's doc) need re-deriving from the new run",
    b.err_units
  );

  // ── THE FINDING, part 1: the conversion alone is CLEAN.
  let e = o.fp32_cpu.unwrap_or_else(|| {
    panic!(
      "cell E (CoreML fp32 / CpuOnly) did not cluster. It is pinned as reproducing the reference \
       EXACTLY; a refusal to answer would mean the CoreML embedding conversion is defective on \
       its own, overturning this suite's central result."
    )
  });
  assert_eq!(
    e.spk, CELL_E_SPK,
    "cell E found {} speakers, not the pinned {CELL_E_SPK}. This cell is the whole reason \
     `model_io.rs` records the CoreML embedding CONVERSION as exonerated and attributes clip 09 \
     to the int8 palettization plus the `All` placement instead. If it moved, that attribution is \
     stale and must be rewritten — do NOT delete this assertion.",
    e.spk
  );
  assert_eq!(
    e.err_units, 0,
    "cell E scored {} error units. The pinned claim is not 'small DER' but frame-perfect identity \
     with the dia-ort reference — with quantization and the `All` placement both removed, the \
     CoreML embedding conversion does not move one collar-scored speaker-frame. A non-zero value \
     here weakens that to 'nearly clean', which is a different finding.",
    e.err_units
  );
  assert!(
    e.der == 0.0,
    "cell E carries a DER of {:.4} % alongside zero error units — in a real measurement the two \
     derive from one scoring pass and cannot disagree; this record is internally inconsistent",
    e.der * 100.0
  );

  // ── THE FINDING, part 2: quantization, placement held at CpuOnly.
  let d = o.int8_cpu.unwrap_or_else(|| {
    panic!(
      "cell D (CoreML int8 / CpuOnly) did not cluster. It is pinned as ANSWERING with \
       {CELL_D_SPK} speakers; a refusal is a new failure mode, not a flake."
    )
  });
  assert_eq!(
    d.spk, CELL_D_SPK,
    "cell D found {} speakers, not the pinned {CELL_D_SPK}. Against cell E's {CELL_E_SPK}, this \
     cell is what prices the int8 palettization on its own: 2 speakers, with the placement held \
     at `CpuOnly`.",
    d.spk
  );
  assert!(
    (d.der - CELL_D_DER).abs() <= CORNER_DER_TOL,
    "cell D scored {:.4} % DER, off the pinned {:.4} % (±{:.4} %). Its landing within 0.23 pp of \
     the shipping arm's 16.5904 % is what says quantization carries almost the whole error mass.",
    d.der * 100.0,
    CELL_D_DER * 100.0,
    CORNER_DER_TOL * 100.0
  );
  assert!(
    d.err_units.abs_diff(CELL_D_ERR_UNITS) <= ERR_UNITS_TOL,
    "cell D scored {} error units, off the pinned {CELL_D_ERR_UNITS} (±{ERR_UNITS_TOL}) — the \
     quantization arm's error mass moved at unit precision; the reported '98.6 %' share derived \
     from it (see ERR_UNITS_TOL's doc) needs re-deriving from the new run",
    d.err_units
  );

  // ── THE FINDING, part 3: placement, precision held at fp32.
  let c = o.fp32_all.unwrap_or_else(|| {
    panic!(
      "cell C (CoreML fp32 / All) did not cluster. It is pinned as ANSWERING with {CELL_C_SPK} \
       speakers; a refusal is a new failure mode, not a flake."
    )
  });
  assert_eq!(
    c.spk, CELL_C_SPK,
    "cell C found {} speakers, not the pinned {CELL_C_SPK}. Against cell E's {CELL_E_SPK}, this \
     cell is what prices the `All` placement on its own: 1 speaker, with the precision held at \
     fp32. If it now recovers all {CELL_E_SPK}, the placement is no longer implicated at all and \
     the record must say so.",
    c.spk
  );
  assert!(
    (c.der - CELL_C_DER).abs() <= CORNER_DER_TOL,
    "cell C scored {:.4} % DER, off the pinned {:.4} % (±{:.4} %). Its being 6.5x SMALLER than \
     the shipping arm's 16.5904 % is what says the placement is the minority term.",
    c.der * 100.0,
    CELL_C_DER * 100.0,
    CORNER_DER_TOL * 100.0
  );
  assert!(
    c.err_units.abs_diff(CELL_C_ERR_UNITS) <= ERR_UNITS_TOL,
    "cell C scored {} error units, off the pinned {CELL_C_ERR_UNITS} (±{ERR_UNITS_TOL}) — the \
     placement arm's error mass moved at unit precision; the reported figures derived from it \
     (see ERR_UNITS_TOL's doc) need re-deriving from the new run",
    c.err_units
  );
}

/// [`assert_precision_placement_verdict`] pins EVERY cell, in BOTH directions —
/// proven here hermetically (no models, no fixtures, no audio): the measured
/// record passes, and every single-cell perturbation fails. Same falsifiability
/// contract as [`factorial_verdict_pins_every_cell`]; without it a cell could
/// silently go unpinned and a real change in which factor carries the collapse
/// would pass green.
#[test]
fn precision_placement_verdict_pins_every_cell() {
  /// A frame-perfect outcome: `spk` speakers and not one differing scored
  /// speaker-frame. Free-standing rather than a closure so the mutation table
  /// below can still coerce to `fn(&mut _)`.
  const fn clean(spk: usize) -> CellOutcome {
    CellOutcome {
      spk,
      der: 0.0,
      err_units: 0,
    }
  }
  let good = PrecisionPlacementObserved {
    onnx_cpu: Some(clean(REFERENCE_CORNER_SPK)),
    int8_all: Some(CellOutcome {
      spk: EMBED_ONLY_SPK,
      der: SHIPPING_CORNER_DER,
      err_units: 11_999,
    }),
    fp32_all: Some(CellOutcome {
      spk: CELL_C_SPK,
      der: CELL_C_DER,
      err_units: 1_839,
    }),
    int8_cpu: Some(CellOutcome {
      spk: CELL_D_SPK,
      der: CELL_D_DER,
      err_units: 11_835,
    }),
    fp32_cpu: Some(clean(CELL_E_SPK)),
  };
  assert_precision_placement_verdict(&good);

  let fails = |o: PrecisionPlacementObserved, what: &str| {
    assert!(
      std::panic::catch_unwind(move || assert_precision_placement_verdict(&o)).is_err(),
      "assert_precision_placement_verdict accepted a record with {what} — that cell is NOT pinned"
    );
  };

  for (mutate, what) in [
    (
      (|o: &mut PrecisionPlacementObserved| o.onnx_cpu = Some(clean(9)))
        as fn(&mut PrecisionPlacementObserved),
      "a moved cell-A speaker count",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.onnx_cpu = Some(CellOutcome {
          spk: REFERENCE_CORNER_SPK,
          der: 0.0,
          err_units: 1,
        });
      },
      "a cell A that is no longer frame-perfect",
    ),
    (
      |o: &mut PrecisionPlacementObserved| o.onnx_cpu = None,
      "a cell-A clustering refusal",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.onnx_cpu = Some(CellOutcome {
          spk: REFERENCE_CORNER_SPK,
          der: 0.05,
          err_units: 0,
        });
      },
      "a cell-A DER that disagrees with its own zero error units",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.int8_all = Some(CellOutcome {
          spk: 6,
          der: SHIPPING_CORNER_DER,
          err_units: 11_999,
        });
      },
      "a moved cell-B speaker count",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.int8_all = Some(CellOutcome {
          spk: EMBED_ONLY_SPK,
          der: 0.17,
          err_units: 11_999,
        });
      },
      "a moved cell-B DER",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.int8_all = Some(CellOutcome {
          spk: EMBED_ONLY_SPK,
          der: SHIPPING_CORNER_DER,
          err_units: 12_035,
        });
      },
      "a cell-B error mass drifted 36 units inside the DER band",
    ),
    (
      |o: &mut PrecisionPlacementObserved| o.int8_all = None,
      "a cell-B clustering refusal",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.fp32_all = Some(CellOutcome {
          spk: CELL_E_SPK,
          der: CELL_C_DER,
          err_units: 1_839,
        });
      },
      "a cell C that recovers every speaker (the placement no longer implicated)",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.fp32_all = Some(CellOutcome {
          spk: CELL_C_SPK,
          der: SHIPPING_CORNER_DER,
          err_units: 11_999,
        });
      },
      "a cell-C DER that moved to the shipping arm's",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.fp32_all = Some(CellOutcome {
          spk: CELL_C_SPK,
          der: CELL_C_DER,
          err_units: 1_875,
        });
      },
      "a cell-C error mass drifted 36 units inside the DER band",
    ),
    (
      |o: &mut PrecisionPlacementObserved| o.fp32_all = None,
      "a cell-C clustering refusal",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.int8_cpu = Some(CellOutcome {
          spk: CELL_E_SPK,
          der: CELL_D_DER,
          err_units: 11_835,
        });
      },
      "a cell D that recovers every speaker (quantization no longer implicated)",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.int8_cpu = Some(CellOutcome {
          spk: CELL_D_SPK,
          der: 0.0,
          err_units: 0,
        });
      },
      "a moved cell-D DER",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.int8_cpu = Some(CellOutcome {
          spk: CELL_D_SPK,
          der: CELL_D_DER,
          err_units: 11_799,
        });
      },
      "a cell-D error mass drifted 36 units inside the DER band (the 98.6 % share silently \
       becoming 98.0 %)",
    ),
    (
      |o: &mut PrecisionPlacementObserved| o.int8_cpu = None,
      "a cell-D clustering refusal",
    ),
    (
      |o: &mut PrecisionPlacementObserved| o.fp32_cpu = Some(clean(CELL_C_SPK)),
      "a cell E that loses a speaker (the conversion no longer exonerated)",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.fp32_cpu = Some(CellOutcome {
          spk: CELL_E_SPK,
          der: 0.0,
          err_units: 1,
        });
      },
      "a cell E that is no longer frame-perfect",
    ),
    (
      |o: &mut PrecisionPlacementObserved| o.fp32_cpu = None,
      "a cell-E clustering refusal",
    ),
    (
      |o: &mut PrecisionPlacementObserved| {
        o.fp32_cpu = Some(CellOutcome {
          spk: CELL_E_SPK,
          der: 0.05,
          err_units: 0,
        });
      },
      "a cell-E DER that disagrees with its own zero error units",
    ),
  ] {
    let mut o = good;
    mutate(&mut o);
    fails(o, what);
  }
}

// ══════════════════════════════════════════════════════════════════════
// The mechanism probe: WHAT KIND of perturbation each factor applies
// ══════════════════════════════════════════════════════════════════════

/// One arm's per-row 256-d f64 embeddings, `None` for dead rows.
type ArmRows = Vec<Option<[f64; EMBEDDING_DIM]>>;

/// Per-row 256-d f64 copy of one arm's raw embeddings, `None` for dead rows.
fn arm_rows(raw: &[f32]) -> ArmRows {
  raw
    .as_chunks::<EMBEDDING_DIM>()
    .0
    .iter()
    .map(|r| {
      if r.iter().any(|v| *v != 0.0) {
        Some(core::array::from_fn(|d| f64::from(r[d])))
      } else {
        None
      }
    })
    .collect()
}

fn norm(v: &[f64]) -> f64 {
  v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cos64(a: &[f64], b: &[f64]) -> f64 {
  let (na, nb) = (norm(a), norm(b));
  if na == 0.0 || nb == 0.0 {
    return f64::NAN;
  }
  a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / (na * nb)
}

/// How one arm's per-row perturbation against the fp32/`CpuOnly` base is
/// SHAPED, in the raw 256-d space: its size, and how much of it is one shared
/// direction versus independent per-row scatter.
struct DeltaShape {
  rows: usize,
  /// Mean per-row `‖Δ‖`.
  mean_norm: f64,
  /// `‖mean(Δ)‖` — the shared (coherent) component's size.
  bias_norm: f64,
  /// `mean(Δ)` itself — the shared component's DIRECTION, kept so the verdict
  /// can compare arms' bias directions rather than only their sizes.
  bias: [f64; EMBEDDING_DIM],
  /// `‖mean(Δ)‖ / mean(‖Δ‖)`. Independent zero-mean scatter drives this
  /// toward `1/sqrt(rows)` (~0.02 at 2 114 rows); a shared bias holds it up.
  coherence: f64,
  /// Mean `cos(Δ_i, mean(Δ))` — how aligned individual rows' perturbations
  /// are with the shared direction.
  mean_cos_to_bias: f64,
  /// Fraction of rows with `cos(Δ_i, mean(Δ)) > 0`.
  frac_pos: f64,
  /// Rows whose delta is exactly zero — bit-identical in both arms. Excluded
  /// from the two alignment stats above (no direction to compare).
  zero_deltas: usize,
}

fn delta_shape(
  arm: &[Option<[f64; EMBEDDING_DIM]>],
  base: &[Option<[f64; EMBEDDING_DIM]>],
) -> DeltaShape {
  let deltas: Vec<[f64; EMBEDDING_DIM]> = arm
    .iter()
    .zip(base)
    .filter_map(|(a, b)| match (a, b) {
      (Some(a), Some(b)) => Some(core::array::from_fn(|d| a[d] - b[d])),
      _ => None,
    })
    .collect();
  let rows = deltas.len();
  let mut bias = [0.0f64; EMBEDDING_DIM];
  for d in &deltas {
    for (m, v) in bias.iter_mut().zip(d) {
      *m += v;
    }
  }
  for m in &mut bias {
    *m /= rows as f64;
  }
  let mean_norm = deltas.iter().map(|d| norm(d)).sum::<f64>() / rows as f64;
  let bias_norm = norm(&bias);
  // A zero delta (a row identical in both arms) has no direction, so a cosine
  // against the bias is undefined for it; such rows are excluded from the
  // alignment stats rather than poisoning the mean with NaN. They still count
  // toward `rows`/`mean_norm`/`coherence` — an exactly-agreeing row is real
  // agreement data.
  let cosines: Vec<f64> = deltas
    .iter()
    .filter(|d| d.iter().any(|v| *v != 0.0))
    .map(|d| cos64(d, &bias))
    .collect();
  let directed = cosines.len().max(1);
  DeltaShape {
    rows,
    mean_norm,
    bias_norm,
    bias,
    coherence: bias_norm / mean_norm,
    mean_cos_to_bias: cosines.iter().sum::<f64>() / directed as f64,
    frac_pos: cosines.iter().filter(|c| **c > 0.0).count() as f64 / directed as f64,
    zero_deltas: rows - cosines.len(),
  }
}

/// Clusters one cell keeping diaric's full output (hard per-(chunk, slot)
/// assignments included), unlike [`cluster`] which keeps only the spans.
fn cluster_full(
  cell: &Cell,
  window: &WindowOptions,
  plda: &diaric::plda::PldaTransform,
) -> Result<diaric::offline::OfflineOutput, diaric::offline::Error> {
  let count = count_from_segmentations(
    &cell.segmentations,
    cell.num_chunks,
    cell.num_frames,
    SEG_NUM_SLOTS,
    window.onset(),
    chunk_sliding_window(window),
    frame_sliding_window(),
  );
  let input = diaric::offline::OfflineInput::new(
    &cell.raw_embeddings,
    cell.num_chunks,
    SEG_NUM_SLOTS,
    &cell.segmentations,
    cell.num_frames,
    &count,
    count.len(),
    chunk_sliding_window(window).into(),
    frame_sliding_window().into(),
    plda,
  );
  diaric::offline::diarize_offline(&input)
}

/// The decision-relevant shape statistics of one arm's perturbation and its
/// effect in the clustering space, extracted from the measurement (or
/// synthesized by [`mechanism_verdict_pins_every_field`]) so
/// [`assert_mechanism_verdict`] is a pure function both can call.
#[derive(Clone, Copy, Debug)]
struct ArmMechanism {
  /// `‖mean(Δ)‖ / mean(‖Δ‖)` vs cell E ([`DeltaShape::coherence`]).
  coherence: f64,
  /// Fraction of rows whose delta has positive projection on the shared bias.
  frac_pos: f64,
  /// Mean per-row `‖Δ‖` vs cell E ([`DeltaShape::mean_norm`]) — the arm's
  /// per-row perturbation SIZE.
  mean_norm: f64,
  /// `mean(Δ)` vs cell E ([`DeltaShape::bias`]) — the shared (coherent)
  /// component, direction AND size. A scalar norm alone admits a bias
  /// pointing anywhere; the vector is what lets the verdict hold two arms to
  /// the SAME direction.
  bias: [f64; EMBEDDING_DIM],
  /// Rows paired live in both this arm and cell E ([`DeltaShape::rows`]).
  rows: usize,
  /// Rows whose delta is exactly zero ([`DeltaShape::zero_deltas`]). A
  /// summary over mostly-zero deltas can reproduce every ratio above from a
  /// single moved embedding; this counter is what forbids that record.
  zero_deltas: usize,
  /// Mean cos-to-own-centroid of the arm's PLDA-projected rows, grouped by
  /// cell E's labels — within-cluster tightness.
  within: f64,
  /// The LARGEST positive between-E-cluster centroid-cosine movement vs cell
  /// E's own geometry — margin compression toward an AHC merge.
  max_pair_gain: f64,
}

/// The mechanism record: cells D (quantization isolated), C (placement
/// isolated) and B (the int8-era shipping bundle) against base E, plus E's
/// own within-cluster tightness and mean raw embedding norm.
#[derive(Clone, Copy, Debug)]
struct MechanismObserved {
  base_within: f64,
  /// Mean raw `‖x‖` of cell E's live rows — the scale the bias sizes are
  /// read against.
  base_mean_norm: f64,
  d: ArmMechanism,
  c: ArmMechanism,
  b: ArmMechanism,
}

/// **The measured mechanism, pinned** (clip 09, dia reference segmentation
/// fixed, Apple M1 Max, macOS 26.5 build 25F71, arm64; the independent-scatter
/// null for `coherence` is `1/sqrt(2114)` ≈ 0.022; cell E's mean raw `‖x‖` is
/// 2.5182):
///
/// ```text
/// cell |            arm | rows |  Δ=0 | coherence | frac>0 | mean|Δ| |  |bias| | within (E 0.6813) | max pair Δcos
/// -----+----------------+------+------+-----------+--------+---------+--------+-------------------+--------------
///    D | int8 / CpuOnly | 2114 |    9 |    0.5039 | 0.9868 | 0.12223 | 0.0616 |            0.6812 |       +0.0504
///    C | fp32 / All     | 2114 |    4 |    0.0624 | 0.6026 | 0.18897 | 0.0118 |            0.6811 |       +0.0196
///    B | int8 / All     | 2114 |    4 |    0.2473 | 0.9669 | 0.25264 | 0.0625 |            0.6809 |       +0.0524
/// ```
///
/// Every causal claim the record documents is guarded below — a claim this
/// gate cannot catch is a claim this gate does not support:
///
/// - **The statistics summarize the whole population, not an outlier.** Every
///   arm pairs all 2 114 live rows against cell E, and at most a handful are
///   bit-identical (measured 4-9). Without these two guards a single moved
///   embedding among zero deltas reproduces every ratio below. Guards:
///   `rows == 2114` and `zero_deltas <= 20`, per arm.
/// - **Quantization error is COHERENT** (D: half the perturbation mass is one
///   shared direction, 23x the independent-scatter null; 98.7 % of rows
///   aligned with it). The palettization applies a near-constant displacement
///   to every embedding — NOT the "roughly isotropic, unbiased noise" the
///   int8 DECISION's original rationale assumed. Guards: `d.coherence`,
///   `d.frac_pos`.
/// - **The int8-era shipping bundle B carries D's displacement — the same
///   DIRECTION, not merely a same-sized one** (B keeps 96.7 % alignment
///   against its own bias, and its bias vector must lie in D's half-space
///   cone). Guards: `b.coherence`, `b.frac_pos`, and
///   `cos(bias_B, bias_D) >= 0.70` — the floor separates the shared-direction
///   mechanism from the orthogonal/opposite counterexample class, the way
///   `NULL_COHERENCE` separates coherence from scatter. B's per-row
///   magnitude carries no causal claim of its own, but its table row is part
///   of this pinned record, so it is held to the record band
///   `b.mean_norm ∈ [0.20, 0.31]` around the measured 0.25264 — the bundle
///   keeps its stacked-perturbation scale (a collapse to D's ~0.12 would
///   mean the placement scatter vanished from the bundle).
/// - **Placement error is NEAR-ISOTROPIC and LARGER per row** (C: coherence
///   within ~3x of the null, alignment near chance, per-row magnitude 1.5x
///   D's — the anti-correlation of perturbation size with clustering
///   damage). Guards: `c.coherence`, `c.frac_pos`, and the two-sided ratio
///   `c.mean_norm / d.mean_norm ∈ [1.25, 1.85]` around the measured 1.55.
/// - **The coherent component is what distinguishes the arms**: D's shared
///   bias is ~2.4 % of the mean embedding norm and ~5x C's. Guards:
///   `‖bias_D‖ / base_mean_norm ∈ [0.015, 0.035]` and
///   `‖bias_D‖ / ‖bias_C‖ ∈ [4.0, 7.0]` around the measured 5.2.
/// - **The damage is between-class compression, not added scatter**: every
///   arm's within-cluster tightness equals E's to three decimals — guarded at
///   that stated precision (±0.0005) — while D and B move their worst
///   between-centroid pair by at least +0.03 cosine (measured +0.05) toward
///   AHC's merge region and C stays below that. Which specific clusters then
///   lose their identity is the printed merge contingency's report, not an
///   asserted claim. Guards: `within` per arm, `max_pair_gain` per arm.
///
/// # Panics
/// On any divergence from the pinned mechanism, naming what changed and what
/// record it invalidates.
fn assert_mechanism_verdict(o: &MechanismObserved) {
  /// The independent-scatter null for the coherence statistic at 2 114 rows,
  /// named in the panic messages.
  const NULL_COHERENCE: f64 = 0.022;
  /// Every arm pairs this many live `(chunk, slot)` rows against cell E —
  /// the probe's full population, with zero row-predicate drop
  /// disagreements.
  const MECHANISM_ROWS: usize = 2_114;
  /// Ceiling on bit-identical rows per arm (measured 4-9). A mostly-zero
  /// delta field is the degenerate record in which one moved embedding
  /// reproduces every summary ratio.
  const MAX_ZERO_DELTAS: usize = 20;

  for (name, arm) in [("D", &o.d), ("C", &o.c), ("B", &o.b)] {
    assert_eq!(
      arm.rows, MECHANISM_ROWS,
      "cell {name}: {} rows paired against cell E, not the probe's full {MECHANISM_ROWS} — the \
       population these statistics summarize changed; nothing below can be read until the \
       coverage is re-established",
      arm.rows
    );
    assert!(
      arm.zero_deltas <= MAX_ZERO_DELTAS,
      "cell {name}: {} of {MECHANISM_ROWS} rows have an exactly-zero delta (measured 4-9). A \
       mostly-zero delta field lets a handful of moved embeddings reproduce every summary ratio \
       this verdict checks; the statistics no longer describe the population",
      arm.zero_deltas
    );
  }

  assert!(
    o.d.coherence >= 0.35,
    "int8/CpuOnly delta coherence {:.4} is no longer far above the isotropic null \
     ({NULL_COHERENCE}) — the 'palettization error is a coherent shared displacement' mechanism \
     (model_io.rs) is stale; re-derive it before trusting any conclusion built on it",
    o.d.coherence
  );
  assert!(
    o.d.frac_pos >= 0.95,
    "int8/CpuOnly bias alignment {:.4} dropped below 0.95 — the shared-direction claim weakened",
    o.d.frac_pos
  );
  assert!(
    o.c.coherence <= 0.15,
    "fp32/All delta coherence {:.4} is no longer near-isotropic — the placement axis developed a \
     systematic component; the C-vs-D mechanism contrast is stale",
    o.c.coherence
  );
  assert!(
    o.c.frac_pos <= 0.75,
    "fp32/All bias alignment {:.4} rose above 0.75 — placement error is no longer directionless",
    o.c.frac_pos
  );
  assert!(
    o.b.coherence >= 0.20,
    "int8/All delta coherence {:.4} lost the quantization bias component (0.2473 measured for \
     the int8-era shipping bundle) — B no longer carries D's coherent displacement",
    o.b.coherence
  );
  assert!(
    o.b.frac_pos >= 0.90,
    "int8/All bias alignment {:.4} dropped below 0.90 (0.9669 measured) — the shipping bundle no \
     longer shares the quantization arm's one-signed displacement; the mechanism's claim that B \
     carries D's bias is stale",
    o.b.frac_pos
  );
  assert!(
    (0.20..=0.31).contains(&o.b.mean_norm),
    "int8/All per-row perturbation {:.5} left the record band [0.20, 0.31] around the measured \
     0.25264 — the bundle no longer carries its stacked quantization-plus-placement scale; \
     re-derive the record",
    o.b.mean_norm
  );
  let bd_alignment = cos64(&o.b.bias, &o.d.bias);
  assert!(
    bd_alignment >= 0.70,
    "cos(bias_B, bias_D) = {bd_alignment:.4}: the int8/All bundle's shared displacement no \
     longer points where int8/CpuOnly's does. Equal-sized biases in different directions are a \
     DIFFERENT mechanism; the record's 'B carries D's displacement' claim is stale"
  );
  let size_ratio = o.c.mean_norm / o.d.mean_norm;
  assert!(
    (1.25..=1.85).contains(&size_ratio),
    "fp32/All per-row perturbation is {size_ratio:.3}x int8/CpuOnly's ({:.5} / {:.5}), outside \
     the pinned [1.25, 1.85] band around the measured 1.55 — the anti-correlation record (the \
     LARGER perturbation does LESS damage) no longer matches its measurement; re-derive before \
     citing it",
    o.c.mean_norm,
    o.d.mean_norm
  );
  let bias_ratio = norm(&o.d.bias) / norm(&o.c.bias);
  assert!(
    (4.0..=7.0).contains(&bias_ratio),
    "int8/CpuOnly shared bias is {bias_ratio:.2}x fp32/All's ({:.4} / {:.4}), outside the pinned \
     [4.0, 7.0] band around the measured 5.2 — the '|bias| ~5x' record no longer matches its \
     measurement; re-derive before citing it",
    norm(&o.d.bias),
    norm(&o.c.bias)
  );
  let bias_frac = norm(&o.d.bias) / o.base_mean_norm;
  assert!(
    (0.015..=0.035).contains(&bias_frac),
    "int8/CpuOnly shared bias is {:.4} of the mean embedding norm ({:.4} / {:.4}), outside the \
     pinned [0.015, 0.035] band around the measured 0.024 — the bias magnitude story changed in \
     one direction or the other; re-derive before citing '~2.4 % of norm'",
    bias_frac,
    norm(&o.d.bias),
    o.base_mean_norm
  );
  for (name, arm) in [("D", &o.d), ("C", &o.c), ("B", &o.b)] {
    assert!(
      (arm.within - o.base_within).abs() <= 0.000_5,
      "cell {name}: within-cluster tightness {:.4} moved from E's {:.4} beyond ±0.0005 — the \
       'unchanged to three decimals' record no longer holds; the damage may have stopped being \
       pure between-class compression",
      arm.within,
      o.base_within
    );
  }
  for (name, arm) in [("D", &o.d), ("B", &o.b)] {
    assert!(
      arm.max_pair_gain >= 0.03,
      "cell {name}: worst between-centroid margin compression {:+.4} fell below +0.03 — int8 no \
       longer visibly compresses a speaker-pair margin, so the collapse must have another cause; \
       re-derive the mechanism",
      arm.max_pair_gain
    );
  }
  assert!(
    o.c.max_pair_gain <= 0.03,
    "cell C: worst between-centroid margin compression {:+.4} exceeds +0.03 — the placement now \
     compresses margins as much as quantization does; the minority-factor record is stale",
    o.c.max_pair_gain
  );
}

/// [`assert_mechanism_verdict`] pins EVERY field in the direction that
/// invalidates the recorded mechanism — proven hermetically: the measured
/// record passes, and each single-field perturbation fails, including the
/// degenerate-record classes (a bias pointing elsewhere, a mostly-zero delta
/// field, lost or phantom row coverage) and values just inside the former,
/// looser bounds. Same falsifiability contract as
/// [`precision_placement_verdict_pins_every_cell`].
#[test]
fn mechanism_verdict_pins_every_field() {
  /// A vector with `v` at index `i`, zero elsewhere — the synthetic bias
  /// directions the hermetic record is built from.
  fn axis(i: usize, v: f64) -> [f64; EMBEDDING_DIM] {
    let mut a = [0.0; EMBEDDING_DIM];
    a[i] = v;
    a
  }
  /// The measured record (bias directions synthesized: D and B share one
  /// axis, as the live probe's B·D alignment guard demands; C on another).
  fn good() -> MechanismObserved {
    MechanismObserved {
      base_within: 0.6813,
      base_mean_norm: 2.5182,
      d: ArmMechanism {
        coherence: 0.5039,
        frac_pos: 0.9868,
        mean_norm: 0.12223,
        bias: axis(0, 0.0616),
        rows: 2_114,
        zero_deltas: 9,
        within: 0.6812,
        max_pair_gain: 0.0504,
      },
      c: ArmMechanism {
        coherence: 0.0624,
        frac_pos: 0.6026,
        mean_norm: 0.18897,
        bias: axis(1, 0.0118),
        rows: 2_114,
        zero_deltas: 4,
        within: 0.6811,
        max_pair_gain: 0.0196,
      },
      b: ArmMechanism {
        coherence: 0.2473,
        frac_pos: 0.9669,
        mean_norm: 0.25264,
        bias: axis(0, 0.0625),
        rows: 2_114,
        zero_deltas: 4,
        within: 0.6809,
        max_pair_gain: 0.0524,
      },
    }
  }
  assert_mechanism_verdict(&good());

  let fails = |mutate: fn(&mut MechanismObserved), what: &str| {
    let mut o = good();
    mutate(&mut o);
    assert!(
      std::panic::catch_unwind(move || assert_mechanism_verdict(&o)).is_err(),
      "assert_mechanism_verdict accepted a record with {what} — that field is NOT pinned"
    );
  };
  fails(
    |o| o.d.coherence = 0.05,
    "an isotropic int8 delta (the coherence claim gone)",
  );
  fails(|o| o.d.frac_pos = 0.6, "an unaligned int8 delta");
  fails(
    |o| o.c.coherence = 0.4,
    "a systematic placement delta (the isotropy contrast gone)",
  );
  fails(|o| o.c.frac_pos = 0.95, "a directional placement delta");
  fails(
    |o| o.b.coherence = 0.05,
    "a shipping bundle without the bias",
  );
  fails(
    |o| o.b.frac_pos = 0.5,
    "a shipping bundle whose rows no longer align with the shared bias",
  );
  fails(
    |o| o.b.mean_norm = f64::NAN,
    "a NaN shipping-bundle magnitude",
  );
  fails(
    |o| o.b.mean_norm = 0.13,
    "a shipping bundle collapsed to the quantization arm's scale (placement scatter gone)",
  );
  fails(
    |o| o.b.mean_norm = 0.40,
    "a shipping bundle far above its recorded scale",
  );
  // The direction class: equal-sized biases pointing elsewhere.
  fails(
    |o| o.b.bias = axis(1, 0.0625),
    "a shipping-bundle bias orthogonal to quantization's",
  );
  fails(
    |o| o.b.bias = axis(0, -0.0625),
    "a shipping-bundle bias opposite to quantization's",
  );
  // The coverage class: statistics no longer describing the population.
  fails(
    |o| o.d.zero_deltas = 2_113,
    "a mostly-zero delta field (one moved embedding faking every ratio)",
  );
  fails(|o| o.d.rows = 2_113, "a probe that lost a row of coverage");
  fails(|o| o.b.rows = 2_115, "a probe that grew phantom coverage");
  // The size-ordering band, wild and just inside the former ordering-only
  // guard.
  fails(
    |o| o.c.mean_norm = 0.05,
    "an inverted size ordering (placement no longer perturbs more per row)",
  );
  fails(
    |o| o.c.mean_norm = 0.1234,
    "a size ratio of 1.01x — ordered, but far off the measured 1.55x",
  );
  fails(
    |o| o.c.mean_norm = 0.245,
    "a size ratio of 2.0x, past the pinned band",
  );
  // The bias-ratio band, floor and ceiling.
  fails(
    |o| o.c.bias = axis(1, 0.02),
    "a placement bias grown to within 4x of quantization's",
  );
  fails(
    |o| o.c.bias = axis(1, 0.001),
    "a placement bias so small the coherence contrast reads 61.6x — far off its measurement",
  );
  // The fraction-of-norm band, both sides.
  fails(
    |o| o.d.bias = axis(0, 0.09),
    "a quantization bias above the pinned fraction-of-norm band",
  );
  fails(
    |o| o.base_mean_norm = 5.0,
    "a quantization bias below the pinned fraction-of-norm band",
  );
  // Within-cluster tightness, wild and just inside the former tolerance.
  fails(
    |o| o.d.within = 0.60,
    "a within-cluster collapse under quantization",
  );
  fails(
    |o| o.d.within = 0.6773,
    "a within-cluster drift of 0.004 — outside the stated three-decimals equality",
  );
  fails(
    |o| o.c.within = 0.75,
    "a within-cluster change under placement",
  );
  fails(
    |o| o.b.within = 0.60,
    "a within-cluster collapse under the shipping bundle",
  );
  fails(
    |o| o.d.max_pair_gain = 0.001,
    "an int8 arm that no longer compresses any margin",
  );
  fails(
    |o| o.b.max_pair_gain = 0.001,
    "a shipping bundle that no longer compresses any margin",
  );
  fails(
    |o| o.c.max_pair_gain = 0.06,
    "a placement arm compressing margins like quantization",
  );
}

/// **WHY does the int8 palettization cost speakers when its per-row
/// perturbation is ~9x SMALLER than the `All` placement's?** The DER factorial
/// ([`embedding_precision_x_placement`]) pinned the outcome; this probe pins
/// the mechanism, on the same clip, same host, same fixed reference
/// segmentation.
///
/// Three measurements per arm, each against cell E (fp32/`CpuOnly`, the
/// frame-perfect conversion base):
///
/// 1. **Raw-space delta SHAPE** ([`DeltaShape`]): how much of the arm's
///    perturbation is ONE shared direction (a bias all rows move along)
///    versus independent per-row scatter.
/// 2. **Clustering-space between-speaker margins**: every live row projected
///    through diaric's own frozen community-1 transform
///    (`PldaTransform::project` — center on the FROZEN mean, L2-normalize,
///    LDA, re-center, re-normalize, PLDA-whiten), grouped by cell E's own
///    hard-cluster labels, and the pairwise centroid cosines compared arm
///    vs E. AHC merges at cosine distance `1 - max(0, cos) < 0.6`, so a
///    between-centroid cosine climbing toward and past 0.4 is the margin
///    that decides a merge.
/// 3. **Merge contingency**: which of E's 8 clusters land in the same
///    cluster of the arm's own diarization — naming the speakers that
///    collapse rather than inferring them from a count.
///
/// The report prints in full before any assertion fires.
#[test]
#[ignore = "requires speakerkit models, dia parity fixtures + WeSpeaker ONNX (17 min of audio, 1 seg + 5 embed passes)"]
fn quantization_error_structure() {
  let audio = fixtures_root().join(CLIP).join("clip_16k.wav");
  assert!(
    audio.exists(),
    "clip audio not found (set DIA_PARITY_FIXTURES)"
  );
  assert!(
    common::embed_path().exists() && common::embed_fp32_path().exists(),
    "need BOTH wespeaker_v2.mlmodelc and wespeaker.mlmodelc (set SPEAKERKIT_TEST_MODELS)"
  );
  let samples = common::load_wav_16k_mono(&audio);
  assert_eq!(
    samples.len(),
    CLIP_SAMPLES,
    "{CLIP}: audio identity changed"
  );
  assert_eq!(common::fnv1a_f32(&samples), CLIP_AUDIO_FNV);

  let window = Options::new().window();
  let starts = chunk_starts(samples.len(), &window);
  let plda = diaric::plda::PldaTransform::new().expect("load community-1 PldaTransform (diaric)");
  let seg = run_seg(Backend::Onnx, &samples, &starts);

  println!(
    "\n╔══ quantization error structure — {CLIP} ══\n║ reference segmentation: dia ONNX, fixed | \
     {} chunks x {SEG_NUM_SLOTS} slots",
    starts.len()
  );

  // ── The five arms' raw embeddings + full clustering outputs.
  let mut rows_by_cell: Vec<(char, EmbedArm, ArmRows)> = Vec::new();
  let mut outs_by_cell: Vec<(
    char,
    Result<diaric::offline::OfflineOutput, diaric::offline::Error>,
  )> = Vec::new();
  for (i, arm) in PRECISION_PLACEMENT_ARMS.into_iter().enumerate() {
    let cell_tag = char::from(b'A' + u8::try_from(i).expect("five arms"));
    let mut embed = EmbedSide::load(arm);
    let assembled = assemble(&seg, &mut embed, &samples, &starts, &plda);
    let out = cluster_full(&assembled, &window, &plda);
    println!(
      "║ ran {cell_tag}: {:<24} -> {}",
      arm.label(),
      match &out {
        Ok(o) => format!(
          "{} speakers",
          distinct_speakers(
            &o.spans_slice()
              .iter()
              .map(|s| Seg {
                start: s.start(),
                end: s.start() + s.duration(),
                spk: s.cluster(),
              })
              .collect::<Vec<_>>()
          )
          .len()
        ),
        Err(e) => format!("CLUSTERING FAILED — {e}"),
      }
    );
    rows_by_cell.push((cell_tag, arm, arm_rows(&assembled.raw_embeddings)));
    outs_by_cell.push((cell_tag, out));
  }

  let rows_of = |tag: char| -> &ArmRows {
    &rows_by_cell
      .iter()
      .find(|(c, ..)| *c == tag)
      .expect("cell ran")
      .2
  };
  let out_of = |tag: char| -> &diaric::offline::OfflineOutput {
    match &outs_by_cell
      .iter()
      .find(|(c, _)| *c == tag)
      .expect("cell ran")
      .1
    {
      Ok(o) => o,
      Err(e) => panic!("cell {tag} failed to cluster ({e}); the probe needs its labels"),
    }
  };

  // ── 1. Delta shape vs cell E, raw 256-d space.
  println!(
    "║\n║ raw-space perturbation vs cell E (fp32/CpuOnly):\n║ {:>4} | {:>24} | {:>5} | {:>5} | \
     {:>9} | {:>9} | {:>9} | {:>11} | {:>8}",
    "cell", "arm", "rows", "Δ=0", "mean|Δ|", "|bias|", "coherence", "cos(Δ,bias)", "frac>0"
  );
  let base = rows_of('E');
  let mut shapes: std::collections::BTreeMap<char, DeltaShape> = std::collections::BTreeMap::new();
  for (tag, arm, rows) in rows_by_cell.iter().filter(|(c, ..)| *c != 'E') {
    let s = delta_shape(rows, base);
    println!(
      "║ {tag:>4} | {:>24} | {:>5} | {:>5} | {:>9.5} | {:>9.5} | {:>9.4} | {:>11.4} | {:>8.4}",
      arm.label(),
      s.rows,
      s.zero_deltas,
      s.mean_norm,
      s.bias_norm,
      s.coherence,
      s.mean_cos_to_bias,
      s.frac_pos,
    );
    shapes.insert(*tag, s);
  }
  let mean_norm_of = |tag: char| -> f64 {
    let rows = rows_of(tag);
    let live: Vec<f64> = rows.iter().flatten().map(|r| norm(r)).collect();
    live.iter().sum::<f64>() / live.len() as f64
  };
  println!(
    "║ mean raw ‖x‖: A {:.4} | B {:.4} | C {:.4} | D {:.4} | E {:.4}",
    mean_norm_of('A'),
    mean_norm_of('B'),
    mean_norm_of('C'),
    mean_norm_of('D'),
    mean_norm_of('E'),
  );
  println!(
    "║ bias-direction alignment: cos(bias_B, bias_D) {:.4} | cos(bias_C, bias_D) {:.4}",
    cos64(&shapes[&'B'].bias, &shapes[&'D'].bias),
    cos64(&shapes[&'C'].bias, &shapes[&'D'].bias),
  );

  // ── 2. Between-speaker margins in diaric's own clustering space, grouped
  // by cell E's hard labels (E reproduces dia-ort frame-perfectly, so its
  // labels are the reference partition).
  let e_hard = out_of('E').hard_clusters();
  let e_label = |row: usize| -> i32 { e_hard[row / SEG_NUM_SLOTS][row % SEG_NUM_SLOTS] };
  let n_clusters = 1
    + e_hard
      .iter()
      .flatten()
      .copied()
      .max()
      .expect("nonempty hard clusters");
  assert!(
    n_clusters >= 2,
    "need at least two E-clusters to measure margins"
  );

  // Per arm: per-E-cluster centroid of the PLDA-projected live rows.
  let projected_centroids = |tag: char| -> (Vec<Vec<f64>>, usize, f64) {
    let rows = rows_of(tag);
    let mut sums = vec![vec![0.0f64; 128]; usize::try_from(n_clusters).expect("cluster count")];
    let mut counts = vec![0usize; sums.len()];
    let mut proj_fail = 0usize;
    let mut projected: Vec<(usize, Vec<f64>)> = Vec::new();
    for (row, x) in rows.iter().enumerate() {
      let Some(x) = x else { continue };
      let label = e_label(row);
      let Ok(k) = usize::try_from(label) else {
        continue;
      };
      let arr: [f32; 256] = core::array::from_fn(|d| x[d] as f32);
      let Ok(raw) = diaric::plda::RawEmbedding::from_wespeaker(arr) else {
        proj_fail += 1;
        continue;
      };
      let Ok(p) = plda.project(&raw) else {
        proj_fail += 1;
        continue;
      };
      for (sum, v) in sums[k].iter_mut().zip(p.iter()) {
        *sum += v;
      }
      counts[k] += 1;
      projected.push((k, p.to_vec()));
    }
    for (sum, n) in sums.iter_mut().zip(&counts) {
      for v in sum.iter_mut() {
        *v /= *n as f64;
      }
    }
    // Within-cluster tightness: mean cos of a row's projection to its own
    // centroid.
    let within: f64 = projected
      .iter()
      .map(|(k, p)| cos64(p, &sums[*k]))
      .sum::<f64>()
      / projected.len() as f64;
    (sums, proj_fail, within)
  };

  println!(
    "║\n║ between-E-cluster centroid cosines in the PLDA clustering space (AHC merges when \
     pairwise cosine distance 1-max(0,cos) < 0.6):"
  );
  let (base_cent, base_fail, base_within) = projected_centroids('E');
  println!(
    "║   E within-cluster mean cos-to-centroid {base_within:.4} ({base_fail} projection failures)"
  );
  let mut margins: std::collections::BTreeMap<char, (f64, f64)> = std::collections::BTreeMap::new();
  for tag in ['D', 'C', 'B'] {
    let (cent, fail, within) = projected_centroids(tag);
    let mut pair_cos_e: Vec<f64> = Vec::new();
    let mut pair_cos_x: Vec<f64> = Vec::new();
    for i in 0..cent.len() {
      for j in (i + 1)..cent.len() {
        pair_cos_e.push(cos64(&base_cent[i], &base_cent[j]));
        pair_cos_x.push(cos64(&cent[i], &cent[j]));
      }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
      "║   {tag}: between-centroid cos mean {:.4} (E {:.4}) | max {:.4} (E {:.4}) | \
       within-cluster {within:.4} | {fail} projection failures",
      mean(&pair_cos_x),
      mean(&pair_cos_e),
      max(&pair_cos_x),
      max(&pair_cos_e),
    );
    // The per-pair movement, worst five wideners toward a merge.
    let mut moved: Vec<(usize, f64, f64)> = pair_cos_e
      .iter()
      .zip(&pair_cos_x)
      .enumerate()
      .map(|(idx, (e, x))| (idx, *e, *x))
      .collect();
    moved.sort_by(|a, b| (b.2 - b.1).total_cmp(&(a.2 - a.1)));
    for (idx, e, x) in moved.iter().take(5) {
      // Recover (i, j) from the flattened upper-triangle index.
      let mut k = *idx;
      let mut i = 0usize;
      let n = cent.len();
      while k >= n - 1 - i {
        k -= n - 1 - i;
        i += 1;
      }
      let j = i + 1 + k;
      println!(
        "║      pair ({i},{j}): cos {e:+.4} -> {x:+.4}  (Δ {:+.4})",
        x - e
      );
    }
    let max_pair_gain = pair_cos_e
      .iter()
      .zip(&pair_cos_x)
      .map(|(e, x)| x - e)
      .fold(f64::NEG_INFINITY, f64::max);
    margins.insert(tag, (within, max_pair_gain));
  }

  // ── 3. Merge contingency: which E-clusters share an arm cluster.
  for tag in ['D', 'B', 'C'] {
    let hard = out_of(tag).hard_clusters();
    let x_label = |row: usize| -> i32 { hard[row / SEG_NUM_SLOTS][row % SEG_NUM_SLOTS] };
    let rows = rows_of(tag);
    let mut table: std::collections::BTreeMap<(i32, i32), usize> =
      std::collections::BTreeMap::new();
    for (row, x) in rows.iter().enumerate() {
      if x.is_none() {
        continue;
      }
      let (e, l) = (e_label(row), x_label(row));
      if e >= 0 && l >= 0 {
        *table.entry((e, l)).or_default() += 1;
      }
    }
    println!("║\n║ E-cluster -> {tag}-cluster contingency (live rows):");
    let mut by_e: std::collections::BTreeMap<i32, Vec<(i32, usize)>> =
      std::collections::BTreeMap::new();
    for ((e, l), n) in table {
      by_e.entry(e).or_default().push((l, n));
    }
    for (e, mut ls) in by_e {
      ls.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
      let s: Vec<String> = ls.iter().map(|(l, n)| format!("{tag}{l}:{n}")).collect();
      println!("║   E{e} -> {}", s.join(" "));
    }
  }
  println!("╚══");

  // ── The verdict, after the full report has printed.
  let mech = |tag: char| -> ArmMechanism {
    let s = &shapes[&tag];
    let (within, max_pair_gain) = margins[&tag];
    ArmMechanism {
      coherence: s.coherence,
      frac_pos: s.frac_pos,
      mean_norm: s.mean_norm,
      bias: s.bias,
      rows: s.rows,
      zero_deltas: s.zero_deltas,
      within,
      max_pair_gain,
    }
  };
  assert_mechanism_verdict(&MechanismObserved {
    base_within,
    base_mean_norm: mean_norm_of('E'),
    d: mech('D'),
    c: mech('C'),
    b: mech('B'),
  });
}
