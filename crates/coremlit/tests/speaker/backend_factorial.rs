//! **Where does the clip-09 collapse come from — the segmentation conversion,
//! the embedding conversion, or both — AT THE CONFIGURATION WE SHIP?**
//!
//! `parity_shipping_der.rs` measures the shipping default end to end and pins
//! its known-bad clip-09 state (int8 `wespeaker_v2.mlmodelc` on
//! [`ComputeUnits::All`]: 5 of 8 speakers, 16.5904 % DER, 100 % confusion). It
//! cannot say WHICH of the two CoreML conversions produces that, because it
//! never varies them independently: every one of its speakerkit arms runs
//! CoreML segmentation AND CoreML embedding.
//!
//! This suite varies them independently — the 2x2 cross-product of
//! `{ONNX, CoreML}` segmentation x `{ONNX, CoreML}` embedding — with every
//! other factor held constant, and it does so **at the shipping
//! configuration**: the int8 embedder, both CoreML models on
//! [`ComputeUnits::All`].
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
//! # Harness validity: the two corners are checked against pinned numbers
//!
//! The cells are assembled by this file, not by `Extractor::extract`
//! ([`Extraction::from_parts`] is crate-private, so a mixed-backend
//! `Extraction` cannot be built from outside the crate). A hand-assembled
//! pipeline is only worth as much as its agreement with the real one, so
//! [`assert_factorial_verdict`] checks both corners against numbers that were
//! pinned elsewhere by the real pipelines:
//!
//! - the all-CoreML corner must reproduce `parity_shipping_der`'s pinned
//!   `int8/All` clip-09 state (5 speakers, 16.5904 % DER) — same artifacts,
//!   same placement, same decode, so this is an EXACT reproduction and any
//!   drift means the assembly diverged from `Extractor::extract`;
//! - the all-ONNX corner must reproduce dia-ort's pinned clip-09 speaker count
//!   (8) at 0.0000 % DER against `reference.rttm`.
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
//! # What this suite does NOT establish
//!
//! It localizes the collapse to a STAGE, on ONE clip, at ONE configuration, on
//! ONE host. Three limits, all load-bearing:
//!
//! - **It does not separate int8 from `All` from the conversion itself.** The
//!   factor varied is the BACKEND; the CoreML embedding arm is the shipping
//!   bundle (int8-palettized artifact + `All` placement + that conversion) as
//!   one unit. See [`assert_factorial_verdict`]'s "What it does NOT pin".
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
//! cargo test -p coremlit --features speaker-oracle --test speaker_backend_factorial -- --ignored --nocapture
//! ```
#![cfg(feature = "speaker-oracle")]

mod common;
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

/// The clip this experiment runs on: dia's 8-speaker parity fixture, the one
/// whose shipping-configuration collapse `parity_shipping_der`'s
/// `shipping_int8_der_09_mrbeast_dollar_date_8spk_known_defect` pins.
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

/// The all-CoreML corner's expected state: `parity_shipping_der`'s pinned
/// `int8/All` clip-09 numbers. Same artifacts, same placement, same decode as
/// that suite's shipping arm, so this corner reproduces them exactly or the
/// hand-assembly in [`assemble`] has diverged from `Extractor::extract`.
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
    || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../diarization/tests/parity/fixtures"),
    PathBuf::from,
  )
}

/// dia's fp32 WeSpeaker ONNX (override with `DIA_EMBED_MODEL_PATH`) — the same
/// convention `parity_e2e.rs` / `generate_goldens.rs` use.
fn dia_wespeaker_onnx() -> PathBuf {
  std::env::var_os("DIA_EMBED_MODEL_PATH").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../diarization/models/wespeaker_resnet34_lm.onnx")
    },
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

/// The compute placement both CoreML models run on. `All` IS the shipping
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

/// dia's minimum pre-PLDA embedding norm (`owned.rs:619-630`; speakerkit's
/// private `extract::PLDA_MIN_NORM`): a slot whose raw embedding is shorter
/// than this is dropped, its segmentation column zeroed, exactly as if it had
/// never been active.
const PLDA_MIN_NORM: f64 = 0.01;

/// The embedding backend for one cell, holding its loaded model. dia's is
/// `&mut self` per call; speakerkit's is `&self` and batches all three slots.
enum EmbedSide {
  CoreMl(EmbedModel),
  Onnx(Box<dia::embed::EmbedModel>),
}

impl EmbedSide {
  fn load(backend: Backend) -> Self {
    match backend {
      Backend::CoreMl => Self::CoreMl(
        EmbedModel::from_file_with(
          coreml_embed_path(),
          EmbedModelOptions::new().with_compute(PLACEMENT),
        )
        .expect("load wespeaker_v2.mlmodelc (int8, shipping)"),
      ),
      Backend::Onnx => {
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
/// overlap-exclusion mask rule, the Skip-slot column zeroing, and the
/// [`PLDA_MIN_NORM`] drop (which also zeroes the slot's column).
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
fn assemble(seg: &SegRun, embed: &mut EmbedSide, samples: &[f32], starts: &[usize]) -> Cell {
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
      // dia's f64 norm pre-check (`owned.rs:619-630`): a degenerate embedding
      // is dropped and its column zeroed rather than fed to PLDA.
      let norm_sq: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
      if norm_sq.sqrt() < PLDA_MIN_NORM {
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
    for embed_backend in [Backend::Onnx, Backend::CoreMl] {
      let mut embed = EmbedSide::load(embed_backend);
      let cell = assemble(seg_run, &mut embed, &samples, &starts);
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
/// - **Both corners** are harness-validity checks against numbers the REAL
///   pipelines pinned elsewhere: all-ONNX must reproduce dia-ort's 8 speakers
///   at 0.0000 %, and all-CoreML must reproduce `parity_shipping_der`'s
///   `int8/All` clip-09 pin (5 speakers, 16.5904 %). The all-CoreML corner
///   lands on that pin to the printed precision — same DER, same 11 999
///   confusion units — which is what licenses reading the two hybrid cells at
///   all.
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
/// three properties carries the failure is NOT isolated here; separating them
/// needs the same hybrid harness run with the fp32 CoreML embedder on `All`
/// and with the int8 CoreML embedder on `CpuOnly`, reference segmentation held
/// fixed.
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

  // ── Corner 2: the harness reproduces `Extractor::extract` at the shipping
  // configuration, to `parity_shipping_der`'s own pinned numbers.
  let (ship_spk, ship_der) = o.coreml_coreml.unwrap_or_else(|| {
    panic!(
      "COREML-seg + COREML-emb did not cluster. `parity_shipping_der` pins this exact \
       configuration as ANSWERING with {SHIPPING_CORNER_SPK} of 8 speakers; either the shipping \
       path regressed to 'cannot answer at all' or this harness diverged from `Extractor::extract`."
    )
  });
  assert_eq!(
    ship_spk, SHIPPING_CORNER_SPK,
    "COREML-seg + COREML-emb found {ship_spk} speakers; `parity_shipping_der` pins the identical \
     artifacts/placement/decode at {SHIPPING_CORNER_SPK}. This corner is the harness's \
     equivalence proof against `Extractor::extract` — investigate before reading any other cell."
  );
  assert!(
    (ship_der - SHIPPING_CORNER_DER).abs() <= CORNER_DER_TOL,
    "COREML-seg + COREML-emb scored {:.4} % DER; `parity_shipping_der` pins the identical \
     configuration at {:.4} % (±{:.4} %)",
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
