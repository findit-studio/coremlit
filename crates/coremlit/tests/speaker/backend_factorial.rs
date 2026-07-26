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
//! cargo test -p coremlit --features speaker-oracle --test speaker_backend_factorial -- --ignored --nocapture
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

/// The CoreML embedding artifact's **weight precision** — the quantization
/// axis of [`embedding_precision_x_placement`].
///
/// Both artifacts are contract-equal (`model_io`'s
/// `wespeaker_fp32_io_contract_equal_but_not_targeted`), so they are
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
  CoreMl {
    precision: Precision,
    placement: ComputeUnits,
  },
}

impl EmbedArm {
  /// The literal shipping embedding path: the int8-palettized artifact on
  /// [`ComputeUnits::All`]. This is the bundle
  /// [`shipping_config_backend_factorial`] varies as a single unit.
  const SHIPPING: Self = Self::CoreMl {
    precision: Precision::Int8,
    placement: PLACEMENT,
  };

  /// The [`Backend`] this arm belongs to — the coarse factor the 2x2
  /// cross-product varies.
  const fn backend(self) -> Backend {
    match self {
      Self::Onnx => Backend::Onnx,
      Self::CoreMl { .. } => Backend::CoreMl,
    }
  }

  /// `"ONNX (fp32) / CPU"`-style label for the report tables.
  fn label(self) -> String {
    match self {
      Self::Onnx => "ONNX fp32 / ort CPU EP".to_string(),
      Self::CoreMl {
        precision,
        placement,
      } => format!("CoreML {} / {placement:?}", precision.tag()),
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
  fn load(arm: EmbedArm) -> Self {
    match arm {
      EmbedArm::CoreMl {
        precision,
        placement,
      } => Self::CoreMl(
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
    for embed_arm in [EmbedArm::Onnx, EmbedArm::SHIPPING] {
      let embed_backend = embed_arm.backend();
      let mut embed = EmbedSide::load(embed_arm);
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
  EmbedArm::CoreMl {
    precision: Precision::Fp32,
    placement: ComputeUnits::All,
  },
  EmbedArm::CoreMl {
    precision: Precision::Int8,
    placement: ComputeUnits::CpuOnly,
  },
  EmbedArm::CoreMl {
    precision: Precision::Fp32,
    placement: ComputeUnits::CpuOnly,
  },
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
  /// Rows non-zero on exactly one side: dia's [`PLDA_MIN_NORM`] pre-check
  /// dropped the `(chunk, slot)` on one arm and kept it on the other. A cosine
  /// is undefined against a zeroed row, so these are counted rather than
  /// folded in — and they are themselves a divergence, since a dropped slot
  /// also zeroes that slot's segmentation column.
  drop_disagreements: usize,
}

/// Compares one arm's `raw_embeddings` against the reference arm's, row by row.
///
/// An all-zero row means the slot was never embedded: either the
/// overlap-exclusion rule skipped it (identical across arms — the plans derive
/// from the ONE fixed reference segmentation) or [`PLDA_MIN_NORM`] dropped it
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
/// [`PLDA_MIN_NORM`] drop disagreements on any arm:
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
/// perturbation does not predict the clustering outcome — consistent with
/// `parity_shipping_der`'s module doc, which already argues that the KIND of
/// perturbation matters rather than its magnitude, but pointing the opposite
/// way from that doc's specific rationale: there, quantization was expected to
/// be the benign, roughly isotropic one. On this clip it is the harmful one.
/// What structure in the palettization error the frozen community-1 LDA/PLDA
/// basis is sensitive to is NOT established here; it would need the
/// perturbation decomposed against that basis, which nothing in this repo
/// currently does.
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
    let assembled = assemble(&seg, &mut embed, &samples, &starts);
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
      "║   {} {:<24} mean cos {:.6} | min cos {:.6} | {} rows | {} PLDA_MIN_NORM drop \
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
    fp32_all: at(EmbedArm::CoreMl {
      precision: Precision::Fp32,
      placement: ComputeUnits::All,
    }),
    int8_cpu: at(EmbedArm::CoreMl {
      precision: Precision::Int8,
      placement: ComputeUnits::CpuOnly,
    }),
    fp32_cpu: at(EmbedArm::CoreMl {
      precision: Precision::Fp32,
      placement: ComputeUnits::CpuOnly,
    }),
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

/// `CoreML int8 / CpuOnly` (cell D) — the quantization + conversion, placement
/// removed.
const CELL_D_SPK: usize = 6;
const CELL_D_DER: f64 = 0.163_636;

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
///   placements (E 8 -> D 6 on `CpuOnly`; C 7 -> B 5 on `All`) and it moves
///   11 835 of cell B's 11 999 error units — 98.6 % of the shipping arm's
///   entire error mass — on `CpuOnly` alone.
/// - **Placement is a real but minority term, and it too is
///   precision-independent.** Holding precision fixed, `All` costs exactly 1
///   speaker at BOTH precisions (E 8 -> C 7 on fp32; D 6 -> B 5 on int8). Its
///   error-unit cost is 1 839 in the fp32 arm (E 0 -> C 1 839) and 164 in the
///   int8 arm (D 11 835 -> B 11 999) — the latter 1.4 % of the shipping arm's
///   total, against quantization's 98.6 %.
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
///   runs over dia's ONNX reference segmentation, which is NOT what this crate
///   ships; `parity_shipping_der`'s arms run CoreML segmentation and land
///   nearby but not identically (its int8/`CpuOnly` clip-09 arm is 6 speakers
///   at 16.4590 %, against cell D's 16.3636 %).
/// - **It does not price the remedy.** That fp32 recovers speakers HERE says
///   nothing about what fp32 does to the other corpus clips, and no fp32/`All`
///   DER arm exists for the gated clips (06 / 14 / 10) — `parity_shipping_der`
///   measures fp32 only on `CpuOnly`. Deciding to ship fp32 needs that arm
///   measured, clip 14 especially.
/// - **It does not say WHY int8 costs two speakers.** Palettization changes the
///   weights, not the graph: the two artifacts' op histograms are identical
///   apart from 38 `constexpr_lut_to_dense` decompressions replacing 36 `const`
///   weight tensors (and one dropped no-op `identity`). Which layer's
///   perturbation the frozen LDA/PLDA basis is sensitive to is unmeasured.
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
     `shipping_config_backend_factorial` and by `parity_shipping_der`'s int8/All clip-09 pin",
    b.der * 100.0,
    SHIPPING_CORNER_DER * 100.0,
    CORNER_DER_TOL * 100.0
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
  ] {
    let mut o = good;
    mutate(&mut o);
    fails(o, what);
  }
}
