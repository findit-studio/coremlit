//! The argmax [`ModelSource`]: `argmaxinc/speakerkit-coreml`'s
//! **in-graph-decoded** segmenter + its two-stage embedder, mapped onto the
//! same [`Extraction`] [`crate::source::FluidAudioSource`] produces (design
//! spec §3-§4,
//! `docs/superpowers/specs/2026-07-13-speakerkit-multisource-diarizer-backend-design.md`).
//!
//! # The fundamental difference: argmax decodes IN-GRAPH
//!
//! FluidAudio's segmenter emits raw powerset logits `[1, 589, 7]` and this
//! crate decodes them host-side with dia's exact semantics
//! ([`crate::segment::multilabel`], [`crate::extract`]'s overlap-exclusion
//! mask derivation). argmax's segmenter emits **already-decoded** tensors:
//! it takes 30 s of waveform and returns per-window, per-frame, per-speaker
//! activity having done the windowing, the powerset decode, the overlap
//! detection and a VAD *inside the CoreML graph*, with **its own**
//! semantics.
//!
//! So this source reuses **none** of [`crate::segment::multilabel`] /
//! [`crate::extract`]'s masking / [`crate::window::chunk_starts`]: there is
//! nothing left to decode. Its whole job is to READ argmax's decoded output
//! exactly the way argmax's own Swift reads it, and to place those values
//! into [`Extraction`]'s layout. The two sources can therefore diarize the
//! same audio differently — that is by design (spec §4), and each is
//! validated against its own oracle (spec §5).
//!
//! Every ported behavior below cites `argmax-oss-swift`'s
//! `Sources/SpeakerKit/Pyannote/SpeakerSegmenterModel.swift` (`Seg.swift`)
//! and `SpeakerEmbedderModel.swift` (`Emb.swift`) by line.
//!
//! # What argmax's Swift actually reads (and what it ignores)
//!
//! The segmenter declares six outputs (`tests/argmax_model_io.rs`). argmax's
//! Swift consumes **three** of them, plus one for its SHAPE only:
//!
//! | Output | Shape | argmax's Swift | This port |
//! |---|---|---|---|
//! | `speaker_ids` | `[21,589,3]` | `Emb.swift:242,290,360` | → `segmentations` + mask values |
//! | `speaker_activity` | `[21,3]` | `Emb.swift:101` | → the per-slot activity gate |
//! | `overlapped_speaker_activity` | `[21,589]` | `Emb.swift:111,244` | → mask overlap-exclusion |
//! | `sliding_window_waveform` | `[21,1,160000]` | **shape only** (`Seg.swift:57-63,289-295`) | shape only |
//! | `speaker_probs` | `[21,589,3]` | **never read** | never read |
//! | `voice_activity` | `[1767]` | **never read** | never read |
//!
//! `speaker_probs` and `voice_activity` appear **nowhere** in
//! `argmax-oss-swift/Sources` (verified by grep across the whole tree; the
//! only `voiceActivity` hits are WhisperKit's unrelated energy VAD,
//! `Sources/WhisperKit/Core/Audio/VoiceActivityDetector.swift`). They are
//! computed by the graph and dropped on the floor. This port does the same,
//! for the same reason: `Extraction` has no field either could fill, and
//! inventing one would change the tensor set `dia` consumes.
//!
//! `sliding_window_waveform`'s **data** is likewise never read: the embedder
//! is fed the *host's own* 30 s padded chunk waveform
//! (`Emb.swift:255-256,348` reads `segmenterOutput.audioChunk`, which
//! `Seg.swift:192-193` set to the padded input the host supplied), not the
//! model's windowed copy of it. The shapes confirm it — the preprocessor's
//! input is `waveforms [1, 480000]` (a whole 30 s chunk), and
//! `Emb.swift:330-334` *hard-errors* unless the preprocessor's waveform size
//! **exceeds** the sliding window's, explicitly asserting the preprocessor is
//! per-CHUNK, not per-window. Only `.shape[0]` (window count = 21) and
//! `.shape[2]` (window length = 160 000 samples) are used.
//!
//! # Value semantics (verified against the real models, not just the Swift)
//!
//! The Swift says *which* tensors; only running the model says *what is in
//! them*. Probed on `tests/fixtures/audio/02_pyannote_sample.wav` through
//! the W32A32/W16A16 variants, and re-asserted by
//! `tests`'s model-gated `argmax_decoded_output_value_semantics`:
//!
//! - `speaker_ids` is **binary** — its value set is exactly `{0.0, 1.0}`. It
//!   is the graph's hard powerset→multilabel decode (the analog of dia's
//!   `powerset_to_speakers_hard`), NOT a threshold on `speaker_probs`:
//!   `speaker_ids == (speaker_probs > 0.5)` held for only 37 001 of 37 107
//!   cells. This is the tensor that becomes `segmentations`.
//! - `speaker_activity[w][s]` is a **frame COUNT** — it equals
//!   `Σ_f speaker_ids[w][f][s]` exactly (63/63 window-speaker pairs). That
//!   is what makes `Emb.swift:99-102`'s `activity * secondsPerFrame > 2.0 *
//!   secondsPerFrame` a "more than two active frames" gate.
//! - `overlapped_speaker_activity[w][f]` is a **binary overlap indicator** —
//!   it equals `(#{s : speaker_ids[w][f][s] == 1} >= 2)` exactly (12 369 /
//!   12 369 frames).
//! - `voice_activity` is a continuous per-frame score over the 1767-frame
//!   chunk timeline (634 distinct values on the probe). Unconsumed, as above.
//!
//! # F16: every argmax input AND output is Float16
//!
//! Every input and output of every argmax artifact — **including the ones
//! whose directory is named `W32A32`** — declares `MLMultiArrayDataType`
//! `Float16` (`tests/argmax_model_io.rs`, which introspects the real
//! `.mlmodelc`; its module doc, delta 1). `W32A32`/`W16A16`/`W8A16` name the
//! internal weight/activation *storage* precision, not the external
//! `MLFeature` dtype. Allocating an F32 buffer for any of them would hand
//! CoreML a mis-typed tensor.
//!
//! [`coremlit`] supports this natively — [`coremlit::f16`] implements
//! [`coremlit::Element`] with `DATA_TYPE = DataType::F16`, so
//! [`coremlit::MultiArray::from_slice`] / [`coremlit::MultiArray::copy_into`]
//! are typed F16 end to end. No coremlit change was needed. Every tensor this
//! module builds or reads is `f16`, converted at the host boundary
//! (`f16::from_f32` in, [`f32::from`] out) — the same conversion argmax's
//! Swift performs via `MLMultiArray`'s `.floatValue`, so this port reads
//! exactly the values its Swift reads.
//!
//! # The index mapping: argmax's (chunk, window) grid IS dia's chunk grid
//!
//! This is the load-bearing structural fact the whole mapping rests on.
//!
//! argmax slices audio into **30 s chunks with a 21 s hop** (`Seg.swift:
//! 147-153`: `start_k = end_{k-1} - chunkStrideOffset`, with
//! `chunkStrideOffset = 144 000` derived at `Seg.swift:110-116`; hop =
//! 480 000 − 144 000 = 336 000 = 21 s, matching `Seg.swift:168`'s
//! `chunkStride`). Inside each chunk the *graph* slides **21 windows of 10 s
//! at a 1 s stride** (`(480 000 − 160 000) / (21 − 1) = 16 000`,
//! `Seg.swift:110-112`).
//!
//! So argmax's window `(k, w)` covers absolute samples
//! `[k·336 000 + w·16 000, +160 000)`. Because `336 000 / 16 000 = 21`
//! **exactly**, consecutive chunks' window grids abut with no gap and no
//! overlap, and the union over all `(k, w)` is precisely the arithmetic
//! sequence `c · 16 000` — i.e. **exactly** dia's own offline chunk grid
//! ([`crate::window::chunk_starts`] at its default
//! [`crate::window::DEFAULT_STEP_SAMPLES`] = 16 000, over the same 10 s
//! [`crate::segment::SEG_CHUNK_SAMPLES`] window). The global chunk index is
//!
//! ```text
//! c = k * ARGMAX_WINDOWS_PER_CHUNK + w        (= k * 21 + w)
//! ```
//!
//! which is simultaneously the flattened `(k, w)` index AND dia's chunk
//! index — the two coincide only because the hop is an exact multiple of the
//! stride. The trailing windows that would run past the real audio are
//! dropped by argmax's own `bounded(windowIdx:)` filter (`Emb.swift:120-130`),
//! and — provably, and swept exhaustively by
//! `tests`'s `bounded_window_grid_equals_dia_chunk_grid` — the surviving
//! set is **exactly** `chunk_starts`'s. `num_chunks` therefore agrees between
//! the two sources, and so do `chunks_sw`, `frames_sw` and `count`.
//!
//! Because that stride is baked into the compiled graph, this source cannot
//! honor a different [`crate::window::WindowOptions::step_samples`]; it
//! rejects one with [`ExtractError::UnsupportedStepSamples`] rather than
//! silently producing a grid that does not mean what the caller asked for.
//!
//! ## The 64 embedder slots are `(window, speaker)` flattened — not a reduction
//!
//! `speaker_masks [1, 64, 1767]` / `speaker_embeddings [1, 64, 256]` index
//! their 64-slot axis as `chunkSpeakerIdx = windowIdx * speakersCount +
//! speakerIdx` (`Emb.swift:240`, read back at `Emb.swift:288,291`) — so 63 of
//! the 64 slots are the 21 windows × 3 speakers of ONE 30 s chunk, and slot
//! 63 is unused padding (argmax's own zero-init loop only covers
//! `0..totalSpeakers = 63`, `Emb.swift:219-224`; nothing ever reads slot 63,
//! whose max consumed index is `20*3+2 = 62`). Since each argmax *window*
//! becomes one Extraction *chunk*, those 63 slots do not "reduce" to 3 — they
//! **un-flatten** to 21 chunks × 3 slots:
//!
//! ```text
//! raw_embeddings[c = k*21 + w][s][..]  <-  speaker_embeddings[0][w*3 + s][..]
//! ```
//!
//! This port zeroes all 64 mask rows, not just the first 63 — a deliberate,
//! provably inconsequential divergence from `Emb.swift:219-224` (slot 63's
//! output is never read, and its row cannot affect any other: see the
//! row-independence finding below).
//!
//! The 1767-frame mask timeline is the chunk-level frame grid the *graph*
//! uses (589 frames per 10 s window ⇒ 58.9 frames/s ⇒ 1767 frames per 30 s
//! chunk, matching `preprocessor_output_1`'s 2998 fbank frames at ~100 fps).
//! Window `w`'s 589 frames are scattered into it at
//! `windowFrameIdx = startFrame[w] + f` (`Emb.swift:58-75,247`), where
//! `startFrame[w] = trunc(w * 58.9f)` (see this module's `window_start_frame`).
//! It is argmax's own grid and is entirely internal to the embedder call; it
//! never meets [`Extraction`]'s frame timeline, which stays dia's
//! ([`crate::window::frame_sliding_window`]).
//!
//! # The degenerate-mask guard (a real divergence, and why)
//!
//! An "active" speaker (`> 2` active frames) whose every active frame is also
//! an *overlap* frame gets an all-zero mask row, because the mask value is
//! `speaker_ids * (1 - overlapped)` (`Emb.swift:245`) — argmax has no
//! fallback here (dia does: `owned.rs:573-591` reverts to the raw mask when
//! too few clean frames remain; that fallback is dia's host-side decode,
//! which this source deliberately does not import).
//!
//! Probing the real embedder with an all-zero mask row shows it returns a
//! **finite, non-zero, constant** embedding (L2 norm ≈ 0.5356, bit-identical
//! across every all-zero row) — *not* the NaN that FluidAudio's WeSpeaker
//! produces from an empty mask ([`crate::error::InferError::EmptyMask`]). Two
//! consequences, both load-bearing:
//!
//! 1. No NaN hazard, so this source needs none of [`crate::extract`]'s
//!    placeholder-mask machinery: inactive rows are simply left zero, exactly
//!    as argmax leaves them. Safe because the 64 slots are **independent** —
//!    filling one row versus filling all 41 active rows returns that row
//!    bit-for-bit identically (`max|diff| = 0.0`, verified on the real model).
//! 2. That degenerate constant's norm (≈ 0.5356) is far ABOVE
//!    `PLDA_MIN_NORM` (0.01), so the norm guard **cannot** catch it. Emitting
//!    it would feed `dia`'s clustering a plausible-looking vector carrying no
//!    speaker information.
//!
//! So a consumed slot whose mask row is all-zero is **dropped** — embedding
//! row left zero, segmentation column zeroed — rather than emitted. That is
//! `dia`'s own drop semantics for an unusable slot (`owned.rs:561-571`), and
//! it upholds [`Extraction`]'s documented invariant (see below). argmax's
//! Swift would emit the degenerate vector; this port will not.
//!
//! # The `Extraction` invariant this source upholds
//!
//! `Extraction`'s contract couples the two tensors: a dropped `(chunk, slot)`
//! has an all-zero `raw_embeddings` row **and** an all-zero `segmentations`
//! column (`crate::extract`'s module doc; dia's `owned.rs:561-571,619-630`).
//! A nonzero column with a zero row would hand `dia` a chunk-slot that is
//! active in the count tensor but has no embedding to cluster.
//!
//! Every slot this source leaves un-embedded therefore gets its segmentation
//! column zeroed: the ones argmax's activity gate rejects (`<= 2` active
//! frames — strictly more aggressive than dia's "no active frame at all"),
//! the degenerate-mask ones above, and the PLDA-norm ones. `count` is then
//! computed over the POST-zeroing buffer, preserving dia's own ordering
//! (`owned.rs:663-673`; [`crate::extract`]'s "Count runs after all zeroing").

use std::path::{Path, PathBuf};

use coremlit::{ComputeUnits, DataType, Features, Model, MultiArray, f16};

use crate::{
  embed::{EMBED_SLOTS, EMBEDDING_DIM},
  error::{ExtractError, InferError, ModelError},
  extract::Extraction,
  segment::{SEG_CHUNK_SAMPLES, SEG_NUM_SLOTS},
  source::ModelSource,
  window::WindowOptions,
};

// ─────────────────────────────────────────────────────────────────────────
// The pinned argmax contract (tests/argmax_model_io.rs)
// ─────────────────────────────────────────────────────────────────────────

/// Samples the argmax segmenter consumes per call — `waveform [480000]`,
/// 30 s at 16 kHz (`Seg.swift:20`'s `chunkLengthInSeconds = 30.0`).
pub const ARGMAX_CHUNK_SAMPLES: usize = 480_000;

/// Sliding windows the segmenter's graph produces per 30 s chunk — the `21`
/// of `speaker_ids [21, 589, 3]` / `sliding_window_waveform [21, 1, 160000]`
/// (`Seg.swift:289-291`'s `windowsCount`).
pub const ARGMAX_WINDOWS_PER_CHUNK: usize = 21;

/// Segmentation frames per window — the `589` of `speaker_ids [21, 589, 3]`
/// (`Emb.swift:49`'s `framesPerWindowCount`). Same pyannote frame count
/// FluidAudio's segmenter declares.
pub const ARGMAX_FRAMES_PER_WINDOW: usize = 589;

/// Speaker slots per window — the `3` of `speaker_ids [21, 589, 3]`
/// (`Emb.swift:48`'s `speakersCount`). Equals [`SEG_NUM_SLOTS`].
pub const ARGMAX_NUM_SPEAKERS: usize = SEG_NUM_SLOTS;

/// Mask/embedding slots the embedder exposes — the `64` of
/// `speaker_masks [1, 64, 1767]` (`Emb.swift:388-395`'s
/// `speakerDimensionForMasks`). Only `ARGMAX_WINDOWS_PER_CHUNK *
/// ARGMAX_NUM_SPEAKERS` = 63 are ever used; slot 63 is padding.
pub const ARGMAX_MASK_SLOTS: usize = 64;

/// Frames on the embedder's chunk-level mask timeline — the `1767` of
/// `speaker_masks [1, 64, 1767]`, i.e. `Emb.swift:56`'s `framesPerChunk =
/// framesPerWindowCount * secondsPerChunk / secondsPerWindow = 589 * 3`.
pub const ARGMAX_MASK_FRAMES: usize = ARGMAX_FRAMES_PER_WINDOW * 3;

/// Fbank frames the preprocessor emits per 30 s chunk — the `2998` of
/// `preprocessor_output_1 [1, 2998, 80]`.
pub const ARGMAX_FBANK_FRAMES: usize = 2998;

/// Mel bins the preprocessor emits — the `80` of
/// `preprocessor_output_1 [1, 2998, 80]`.
pub const ARGMAX_FBANK_BINS: usize = 80;

/// Samples in one in-graph sliding window — `sliding_window_waveform`'s
/// `.shape[2]` (`Seg.swift:61`). Equals [`SEG_CHUNK_SAMPLES`] (10 s): argmax
/// and FluidAudio window the audio identically, argmax just does it inside
/// the graph.
pub const ARGMAX_WINDOW_SAMPLES: usize = SEG_CHUNK_SAMPLES;

/// Stride between the graph's sliding windows, in samples:
/// `(480 000 − 160 000) / (21 − 1) = 16 000` (1 s) — `Seg.swift:110-112`'s
/// `windowStride`. Fixed by the compiled graph, hence
/// [`ExtractError::UnsupportedStepSamples`].
pub const ARGMAX_WINDOW_STRIDE_SAMPLES: usize =
  (ARGMAX_CHUNK_SAMPLES - ARGMAX_WINDOW_SAMPLES) / (ARGMAX_WINDOWS_PER_CHUNK - 1);

/// Overlap between consecutive 30 s chunks, in samples: `windowLength −
/// windowStride = 160 000 − 16 000 = 144 000` (9 s) — `Seg.swift:114-116`'s
/// `chunkStrideOffset`.
pub const ARGMAX_CHUNK_STRIDE_OFFSET: usize = ARGMAX_WINDOW_SAMPLES - ARGMAX_WINDOW_STRIDE_SAMPLES;

/// Hop between consecutive 30 s chunk starts: `480 000 − 144 000 = 336 000`
/// (21 s) — `Seg.swift:168`'s `chunkStride`, in samples rather than seconds.
/// Exactly `ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_WINDOW_STRIDE_SAMPLES`, which
/// is what makes the global chunk index `k * 21 + w` (module doc).
pub const ARGMAX_CHUNK_HOP_SAMPLES: usize = ARGMAX_CHUNK_SAMPLES - ARGMAX_CHUNK_STRIDE_OFFSET;

/// argmax's per-slot activity gate: a speaker is "active" in a window only
/// with STRICTLY MORE than this many active frames (`Emb.swift:99-102`, whose
/// `activity * secondsPerFrame > 2.0 * secondsPerFrame` reduces to
/// `activity > 2.0` — `speaker_activity` IS a frame count, module doc).
pub const ARGMAX_MIN_ACTIVE_FRAMES: f32 = 2.0;

/// PLDA minimum raw-embedding L2 norm — the same guard
/// [`crate::extract::Extractor::extract`] applies (dia's inline `0.01`,
/// `diarization/src/offline/owned.rs:619-630`), re-applied here because
/// `Extraction` feeds the same `dia` clustering either way.
///
/// It does NOT subsume the degenerate-mask guard: the all-zero-mask
/// embedding's norm is ≈ 0.5356, far above this (module doc).
const PLDA_MIN_NORM: f64 = 0.01;

mod names {
  /// Segmenter input: the 30 s waveform, `[480000]`, F16.
  pub const WAVEFORM: &str = "waveform";
  /// Segmenter output: hard per-`(window, frame, speaker)` activity.
  pub const SPEAKER_IDS: &str = "speaker_ids";
  /// Segmenter output: per-`(window, speaker)` active-frame COUNT.
  pub const SPEAKER_ACTIVITY: &str = "speaker_activity";
  /// Segmenter output: per-`(window, frame)` binary overlap indicator.
  pub const OVERLAPPED: &str = "overlapped_speaker_activity";
  /// Segmenter output read for its SHAPE only (`Seg.swift:57-63,289-295`).
  pub const SLIDING_WINDOW_WAVEFORM: &str = "sliding_window_waveform";
  /// Segmenter output computed by the graph and never read (module doc).
  pub const SPEAKER_PROBS: &str = "speaker_probs";
  /// Segmenter output computed by the graph and never read (module doc).
  pub const VOICE_ACTIVITY: &str = "voice_activity";
  /// Preprocessor input: the same 30 s waveform, `[1, 480000]`, F16.
  pub const WAVEFORMS: &str = "waveforms";
  /// Preprocessor output / embedder input: the 80-mel fbank.
  pub const PREPROCESSOR_OUTPUT: &str = "preprocessor_output_1";
  /// Embedder input: the `[1, 64, 1767]` per-slot pooling masks.
  pub const SPEAKER_MASKS: &str = "speaker_masks";
  /// Embedder output: the `[1, 64, 256]` raw embeddings.
  pub const SPEAKER_EMBEDDINGS: &str = "speaker_embeddings";
}

// ─────────────────────────────────────────────────────────────────────────
// Options (rust-options-pattern)
// ─────────────────────────────────────────────────────────────────────────

/// Which of argmax's quantization tiers to load.
///
/// The tier names are argmax's own directory names (weight-bits ×
/// activation-bits) and describe INTERNAL storage precision only — every
/// variant's external `MLMultiArray` I/O is F16 either way (module doc's
/// "F16" section). Note the baseline tier is spelled differently per model:
/// `W32A32` for the segmenter, `W16A16` for the embedder pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ArgmaxVariant {
  /// argmax's un-palettized baseline: `W32A32` segmenter + `W16A16`
  /// embedder/preprocessor. The default.
  Baseline,
  /// argmax's 8-bit-palettized tier: `W8A16` for all three models.
  /// (`storagePrecision: "Mixed (Float16, Palettized (8 bits))"`.)
  W8A16,
}

impl Default for ArgmaxVariant {
  fn default() -> Self {
    DEFAULT_ARGMAX_VARIANT
  }
}

/// Default [`ArgmaxOptions::variant`] — [`ArgmaxVariant::Baseline`], the
/// un-palettized tier.
pub const DEFAULT_ARGMAX_VARIANT: ArgmaxVariant = ArgmaxVariant::Baseline;

impl ArgmaxVariant {
  /// This variant's directory name under `speaker_segmenter/pyannote-v3/`.
  #[inline(always)]
  pub const fn segmenter_dir(self) -> &'static str {
    match self {
      Self::Baseline => "W32A32",
      Self::W8A16 => "W8A16",
    }
  }

  /// This variant's directory name under `speaker_embedder/pyannote-v3/` —
  /// note the baseline embedder is `W16A16`, not the segmenter's `W32A32`.
  #[inline(always)]
  pub const fn embedder_dir(self) -> &'static str {
    match self {
      Self::Baseline => "W16A16",
      Self::W8A16 => "W8A16",
    }
  }
}

#[cfg(feature = "serde")]
fn default_compute() -> ComputeUnits {
  DEFAULT_ARGMAX_COMPUTE
}

/// Default compute placement for each of argmax's three models.
///
/// [`ComputeUnits::All`], matching this crate's own
/// [`crate::segment::DEFAULT_SEGMENT_COMPUTE`] /
/// [`crate::embed::DEFAULT_EMBED_COMPUTE`]. argmax's Swift instead hardcodes
/// `.cpuOnly` for the preprocessor (`SpeakerPreEmbedderModel.swift:14`) and
/// defaults the segmenter to `.cpuOnly` (`Seg.swift:27`) — placement is a
/// scheduling choice, not a semantic one, so this crate keeps its own
/// convention and exposes all three as knobs.
pub const DEFAULT_ARGMAX_COMPUTE: ComputeUnits = ComputeUnits::All;

/// Which hardware CoreML may schedule each of argmax's THREE models on
/// (rust-options-pattern).
///
/// Distinct from [`crate::extract::ComputeOptions`]'s two knobs because
/// argmax splits embedding across two artifacts — a fbank
/// `SpeakerEmbedderPreprocessor` and the `SpeakerEmbedder` proper — so it has
/// a third model to place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArgmaxComputeOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_compute", with = "crate::compute_units_serde")
  )]
  segmenter: ComputeUnits,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_compute", with = "crate::compute_units_serde")
  )]
  preprocessor: ComputeUnits,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_compute", with = "crate::compute_units_serde")
  )]
  embedder: ComputeUnits,
}

impl Default for ArgmaxComputeOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ArgmaxComputeOptions {
  /// All three models on [`DEFAULT_ARGMAX_COMPUTE`].
  pub const fn new() -> Self {
    Self {
      segmenter: DEFAULT_ARGMAX_COMPUTE,
      preprocessor: DEFAULT_ARGMAX_COMPUTE,
      embedder: DEFAULT_ARGMAX_COMPUTE,
    }
  }

  /// Hardware `SpeakerSegmenter.mlmodelc` may be scheduled on.
  #[inline(always)]
  pub const fn segmenter(&self) -> ComputeUnits {
    self.segmenter
  }
  /// Hardware `SpeakerEmbedderPreprocessor.mlmodelc` may be scheduled on.
  #[inline(always)]
  pub const fn preprocessor(&self) -> ComputeUnits {
    self.preprocessor
  }
  /// Hardware `SpeakerEmbedder.mlmodelc` may be scheduled on.
  #[inline(always)]
  pub const fn embedder(&self) -> ComputeUnits {
    self.embedder
  }

  /// Builder form of [`Self::set_segmenter`].
  #[must_use]
  #[inline(always)]
  pub const fn with_segmenter(mut self, segmenter: ComputeUnits) -> Self {
    self.set_segmenter(segmenter);
    self
  }
  /// Sets [`Self::segmenter`] in place.
  #[inline(always)]
  pub const fn set_segmenter(&mut self, segmenter: ComputeUnits) -> &mut Self {
    self.segmenter = segmenter;
    self
  }
  /// Builder form of [`Self::set_preprocessor`].
  #[must_use]
  #[inline(always)]
  pub const fn with_preprocessor(mut self, preprocessor: ComputeUnits) -> Self {
    self.set_preprocessor(preprocessor);
    self
  }
  /// Sets [`Self::preprocessor`] in place.
  #[inline(always)]
  pub const fn set_preprocessor(&mut self, preprocessor: ComputeUnits) -> &mut Self {
    self.preprocessor = preprocessor;
    self
  }
  /// Builder form of [`Self::set_embedder`].
  #[must_use]
  #[inline(always)]
  pub const fn with_embedder(mut self, embedder: ComputeUnits) -> Self {
    self.set_embedder(embedder);
    self
  }
  /// Sets [`Self::embedder`] in place.
  #[inline(always)]
  pub const fn set_embedder(&mut self, embedder: ComputeUnits) -> &mut Self {
    self.embedder = embedder;
    self
  }
}

/// Full [`ArgmaxSource`] configuration (rust-options-pattern): the
/// quantization [`ArgmaxVariant`], the per-model [`ArgmaxComputeOptions`],
/// and the [`WindowOptions`] geometry.
///
/// No `Eq`: [`WindowOptions`] carries an `f32` `onset`.
///
/// Unlike [`crate::extract::Options`], `window.step_samples` is not a free
/// knob here — argmax's window stride is compiled into its graph, so
/// [`ArgmaxSource::extract`] rejects anything but
/// [`ARGMAX_WINDOW_STRIDE_SAMPLES`] (module doc).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArgmaxOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  window: WindowOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  compute: ArgmaxComputeOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  variant: ArgmaxVariant,
}

impl Default for ArgmaxOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ArgmaxOptions {
  /// Options composing [`WindowOptions::new`], [`ArgmaxComputeOptions::new`]
  /// and [`DEFAULT_ARGMAX_VARIANT`] — each component's own default is the
  /// single source of truth.
  pub const fn new() -> Self {
    Self {
      window: WindowOptions::new(),
      compute: ArgmaxComputeOptions::new(),
      variant: DEFAULT_ARGMAX_VARIANT,
    }
  }

  /// The sliding-window geometry. Only its `onset` is a free knob; see the
  /// struct doc for `step_samples`.
  #[inline(always)]
  pub const fn window(&self) -> WindowOptions {
    self.window
  }
  /// The per-model compute placement.
  #[inline(always)]
  pub const fn compute(&self) -> ArgmaxComputeOptions {
    self.compute
  }
  /// The selected quantization tier.
  #[inline(always)]
  pub const fn variant(&self) -> ArgmaxVariant {
    self.variant
  }

  /// Builder form of [`Self::set_window`].
  #[must_use]
  #[inline(always)]
  pub const fn with_window(mut self, window: WindowOptions) -> Self {
    self.set_window(window);
    self
  }
  /// Sets [`Self::window`] in place.
  #[inline(always)]
  pub const fn set_window(&mut self, window: WindowOptions) -> &mut Self {
    self.window = window;
    self
  }
  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compute(mut self, compute: ArgmaxComputeOptions) -> Self {
    self.set_compute(compute);
    self
  }
  /// Sets [`Self::compute`] in place.
  #[inline(always)]
  pub const fn set_compute(&mut self, compute: ArgmaxComputeOptions) -> &mut Self {
    self.compute = compute;
    self
  }
  /// Builder form of [`Self::set_variant`].
  #[must_use]
  #[inline(always)]
  pub const fn with_variant(mut self, variant: ArgmaxVariant) -> Self {
    self.set_variant(variant);
    self
  }
  /// Sets [`Self::variant`] in place.
  #[inline(always)]
  pub const fn set_variant(&mut self, variant: ArgmaxVariant) -> &mut Self {
    self.variant = variant;
    self
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Pure geometry (hermetically testable — no CoreML)
// ─────────────────────────────────────────────────────────────────────────

/// argmax's 30 s chunk-start offsets over `total_samples` — the exact
/// sample sequence `Seg.swift:147-153`'s loop produces:
///
/// ```text
/// chunkEnd = 0
/// while chunkEnd < total:
///     start = max(chunkEnd - ARGMAX_CHUNK_STRIDE_OFFSET, 0)
///     chunkEnd = min(start + ARGMAX_CHUNK_SAMPLES, total)
/// ```
///
/// Because the loop stops the moment a chunk's end reaches `total`, a
/// truncated chunk is always the LAST one, so every start is exactly
/// `k * ARGMAX_CHUNK_HOP_SAMPLES` (module doc). Returns `[0]` for
/// `total_samples == 0` — argmax's loop would yield nothing there, but
/// [`ArgmaxSource::extract`] rejects empty input one layer up
/// ([`ExtractError::EmptySamples`], mirroring `owned.rs:369-371`), so that
/// divergence is unreachable; this function stays total over its domain like
/// its FluidAudio sibling [`crate::window::chunk_starts`].
fn argmax_chunk_starts(total_samples: usize) -> Vec<usize> {
  let mut starts = Vec::new();
  let mut chunk_end = 0usize;
  loop {
    let start = chunk_end.saturating_sub(ARGMAX_CHUNK_STRIDE_OFFSET);
    starts.push(start);
    chunk_end = (start + ARGMAX_CHUNK_SAMPLES).min(total_samples);
    if chunk_end >= total_samples {
      break;
    }
  }
  starts
}

/// The first frame of window `w` on the embedder's 1767-frame chunk-level
/// mask timeline — `Emb.swift:58-75`'s `chunkIndices[w][0]`, i.e.
/// `Int(windowIdx * strideInFrames)` where `strideInFrames = secondsPerStride
/// * (framesPerWindow / secondsPerWindow) = 1.0 * 58.9`.
///
/// Ported in `f32` with truncation-toward-zero, matching Swift's `Float`
/// arithmetic and its `Int(_:)` conversion bit-for-bit: Rust's `as usize` on
/// a float truncates exactly as Swift's `Int(_:)` does, and both are IEEE
/// binary32. The sequence is `[0, 58, 117, 176, …, 1178]` — verified against
/// the real model's own geometry, and pinned by `tests`'s
/// `window_start_frames_match_argmax_timeline`.
///
/// `58.9` is not representable in binary32; the nearest value is
/// `58.900001525878906`, i.e. slightly ABOVE. That direction is what keeps
/// `trunc(10 * s)` at 589 and `trunc(20 * s)` at 1178 rather than 588/1177 —
/// had it rounded DOWN, this would be off by one at both.
///
/// Because it rounds up, the `f32` form happens to agree EXACTLY with the
/// integer form `w * 589 / 10` across the whole used domain `0..21` — pinned
/// by `tests`'s `window_start_frame_agrees_with_the_integer_form`, which
/// exists precisely so that this equivalence is asserted rather than assumed
/// (a mutation swapping one for the other is genuinely equivalent HERE, and
/// would stop being so if argmax's window/frame geometry ever changed). The
/// `f32` form is kept as the definition anyway: it is what argmax's Swift
/// evaluates, so it stays right by construction if those constants move.
///
/// The last window's last frame is `1178 + 588 = 1766`, so the timeline is
/// exactly covered and never overrun.
fn window_start_frame(w: usize) -> usize {
  let frames_per_second = ARGMAX_FRAMES_PER_WINDOW as f32 / seconds_per_window();
  let stride_in_frames = seconds_per_stride() * frames_per_second;
  (w as f32 * stride_in_frames) as usize
}

/// `Emb.swift:295`'s `secondsPerWindow` — `sliding_window_waveform.shape[2] /
/// audioSampleRate` = 160 000 / 16 000 = 10.0 s.
fn seconds_per_window() -> f32 {
  ARGMAX_WINDOW_SAMPLES as f32 / crate::window::SAMPLE_RATE_HZ as f32
}

/// `Emb.swift:50-53`'s `secondsPerStride` — `(secondsPerChunk −
/// secondsPerWindow) / (windowsCount − 1)` = (30 − 10) / 20 = 1.0 s.
fn seconds_per_stride() -> f32 {
  let seconds_per_chunk = ARGMAX_CHUNK_SAMPLES as f32 / crate::window::SAMPLE_RATE_HZ as f32;
  (seconds_per_chunk - seconds_per_window()) / (ARGMAX_WINDOWS_PER_CHUNK - 1) as f32
}

/// argmax's `bounded(windowIdx:)` (`Emb.swift:120-130`): whether window `w`'s
/// 10 s span lies within the chunk's REAL (unpadded) audio, rather than
/// running out into the zero padding. Window 0 is always bounded.
///
/// argmax phrases it in `Float` seconds:
/// `(secondsPerStride * w + secondsPerWindow) < (waveformLength +
/// secondsPerStride)`. This port evaluates the SAME predicate in exact
/// integer SAMPLE arithmetic, which the seconds form is an approximation of
/// (every constant in it is an exact sample count divided by 16 000):
///
/// ```text
/// w * STRIDE + WINDOW < len + STRIDE   ⟺   w * 16_000 + 144_000 < len
/// ```
///
/// Deliberate: `Float` has ~24 bits of mantissa, so past roughly an hour of
/// audio its resolution at `waveformLength` degrades to the same order as the
/// 1 s stride and the comparison can flip at a boundary. The integer form
/// cannot, which is what makes the grid identity in the module doc a theorem
/// rather than a floating-point coincidence.
fn window_bounded(w: usize, chunk_len_samples: usize) -> bool {
  if w == 0 {
    return true;
  }
  w * ARGMAX_WINDOW_STRIDE_SAMPLES + ARGMAX_CHUNK_STRIDE_OFFSET < chunk_len_samples
}

/// The global [`Extraction`] chunk index of argmax's window `(k, w)`:
/// `k * ARGMAX_WINDOWS_PER_CHUNK + w`.
///
/// Simultaneously the flattened `(chunk, window)` index and dia's own chunk
/// index — the two coincide because argmax's 30 s chunk hop
/// ([`ARGMAX_CHUNK_HOP_SAMPLES`], 336 000) is exactly
/// [`ARGMAX_WINDOWS_PER_CHUNK`] × [`ARGMAX_WINDOW_STRIDE_SAMPLES`]
/// (21 × 16 000). See the module doc.
fn global_chunk(k: usize, w: usize) -> usize {
  k * ARGMAX_WINDOWS_PER_CHUNK + w
}

/// argmax's `activeSpeakerIndices(for:)` (`Emb.swift:98-103`): which of the
/// window's [`ARGMAX_NUM_SPEAKERS`] slots cleared the activity gate.
///
/// argmax writes it as `activity[w][s] * secondsPerFrame > 2.0 *
/// secondsPerFrame`; that is ported literally (rather than pre-cancelled to
/// `> 2.0`) so the comparison is bit-identical to the Swift's. Since
/// `speaker_activity` is an exact small-integer frame count in `f32` and
/// `secondsPerFrame > 0`, the two forms agree on every reachable value —
/// pinned at the boundary by `tests`'s
/// `activity_gate_excludes_two_frames_includes_three`.
fn active_speakers(activity_row: &[f32]) -> [bool; ARGMAX_NUM_SPEAKERS] {
  // `Emb.swift:54`: secondsPerWindow / framesPerWindowCount.
  let seconds_per_frame = seconds_per_window() / ARGMAX_FRAMES_PER_WINDOW as f32;
  let min_active_duration = ARGMAX_MIN_ACTIVE_FRAMES * seconds_per_frame;
  core::array::from_fn(|s| activity_row[s] * seconds_per_frame > min_active_duration)
}

/// One argmax window's decode plan: whether it maps to an [`Extraction`]
/// chunk at all ([`window_bounded`]) and which of its slots cleared the
/// activity gate ([`active_speakers`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WindowPlan {
  bounded: bool,
  active: [bool; ARGMAX_NUM_SPEAKERS],
}

impl WindowPlan {
  /// Whether any slot is active — argmax's `activeSpeakersIndex.isEmpty`
  /// early-out (`Emb.swift:233-236`), which leaves the whole window's mask
  /// rows zero.
  fn any_active(&self) -> bool {
    self.active.iter().any(|&a| a)
  }
}

/// Every window's [`WindowPlan`] for one 30 s chunk, from its unpadded length
/// and its `speaker_activity [21, 3]` tensor.
fn window_plans(
  chunk_len_samples: usize,
  activity: &[f32],
) -> [WindowPlan; ARGMAX_WINDOWS_PER_CHUNK] {
  core::array::from_fn(|w| WindowPlan {
    bounded: window_bounded(w, chunk_len_samples),
    active: active_speakers(&activity[w * ARGMAX_NUM_SPEAKERS..(w + 1) * ARGMAX_NUM_SPEAKERS]),
  })
}

/// Builds the embedder's `speaker_masks [1, 64, 1767]` buffer for one 30 s
/// chunk — the exact scatter of `Emb.swift:216-253`.
///
/// Row `w * 3 + s` (`Emb.swift:240`) holds, at timeline position
/// `window_start_frame(w) + f` (`Emb.swift:247`), the overlap-excluded mask
/// value `speaker_ids[w][f][s] * (1 - overlapped[w][f])` (`Emb.swift:245`) —
/// but ONLY where slot `s` cleared the activity gate (`Emb.swift:248`), and
/// only for windows with at least one active slot (`Emb.swift:233-236`).
/// Everything else stays zero.
///
/// Note this uses each plan's `active` but NOT its `bounded`: argmax builds
/// masks for ALL 21 windows (`Emb.swift:230`) and applies `bounded` only when
/// reading embeddings back out (`Emb.swift:285`). Ported faithfully — an
/// unbounded window's row is computed and then discarded.
///
/// All [`ARGMAX_MASK_SLOTS`] (64) rows are zero-initialized, where argmax
/// zeroes only the first 63 (`Emb.swift:219-224`): see the module doc for why
/// that divergence cannot change any consumed value.
fn build_speaker_masks(
  ids: &[f32],
  overlapped: &[f32],
  plans: &[WindowPlan; ARGMAX_WINDOWS_PER_CHUNK],
) -> Vec<f16> {
  let mut masks = vec![f16::ZERO; ARGMAX_MASK_SLOTS * ARGMAX_MASK_FRAMES];
  for (w, plan) in plans.iter().enumerate() {
    if !plan.any_active() {
      continue; // Emb.swift:233-236
    }
    let start = window_start_frame(w);
    for (s, &is_active) in plan.active.iter().enumerate() {
      if !is_active {
        continue; // Emb.swift:248 — inactive slots never written
      }
      let row = w * ARGMAX_NUM_SPEAKERS + s; // Emb.swift:240
      for f in 0..ARGMAX_FRAMES_PER_WINDOW {
        let id = ids[ids_index(w, f, s)];
        let overlap = overlapped[overlapped_index(w, f)];
        let value = id * (1.0 - overlap); // Emb.swift:245
        masks[row * ARGMAX_MASK_FRAMES + start + f] = f16::from_f32(value);
      }
    }
  }
  masks
}

/// Flat index into `speaker_ids [21, 589, 3]`.
fn ids_index(w: usize, f: usize, s: usize) -> usize {
  (w * ARGMAX_FRAMES_PER_WINDOW + f) * ARGMAX_NUM_SPEAKERS + s
}

/// Flat index into `overlapped_speaker_activity [21, 589]`.
fn overlapped_index(w: usize, f: usize) -> usize {
  w * ARGMAX_FRAMES_PER_WINDOW + f
}

/// Whether mask row `row` of a [`build_speaker_masks`] buffer is entirely
/// zero — the degenerate-mask test (module doc). Reachable for an ACTIVE slot
/// whose every active frame is also an overlap frame.
fn mask_row_is_zero(masks: &[f16], row: usize) -> bool {
  masks[row * ARGMAX_MASK_FRAMES..(row + 1) * ARGMAX_MASK_FRAMES]
    .iter()
    .all(|&v| v == f16::ZERO)
}

/// The flat `Extraction::segmentations` sub-slice for chunk `c` — the same
/// `[c][f][s]` layout `crate::extract` writes (dia's `owned.rs:496`).
fn chunk_segmentation_range(c: usize) -> core::ops::Range<usize> {
  let stride = ARGMAX_FRAMES_PER_WINDOW * SEG_NUM_SLOTS;
  c * stride..(c + 1) * stride
}

/// The flat `Extraction::raw_embeddings` sub-slice for `(chunk c, slot s)` —
/// dia's `owned.rs:631` offset.
fn embedding_range(c: usize, s: usize) -> core::ops::Range<usize> {
  let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
  base..base + EMBEDDING_DIM
}

/// Zeroes slot `s`'s column across one chunk's `[f][s]` slab — dia's
/// column-zero on a dropped `(chunk, slot)` (`owned.rs:567-569,626-628`).
fn zero_slot_column(chunk_segs: &mut [f64], s: usize) {
  for f in 0..ARGMAX_FRAMES_PER_WINDOW {
    chunk_segs[f * SEG_NUM_SLOTS + s] = 0.0;
  }
}

/// Copies the 30 s window starting at `start` into `padded`, zero-clearing
/// first and leaving any out-of-range tail zero — argmax's
/// `AudioProcessor.padOrTrimAudio(.., toLength: maxChunkLength)`
/// (`Seg.swift:178-182`), which pads the final short chunk to a full 480 000
/// samples. Returns the UNPADDED length, which drives [`window_bounded`]
/// (argmax's own `waveformLength`, `Seg.swift:176`).
fn fill_padded_chunk(padded: &mut [f16], samples: &[f32], start: usize) -> usize {
  padded.fill(f16::ZERO);
  let lo = start.min(samples.len());
  let end = (start + ARGMAX_CHUNK_SAMPLES).min(samples.len());
  let n = end - lo;
  for (dst, &src) in padded[..n].iter_mut().zip(&samples[lo..end]) {
    *dst = f16::from_f32(src);
  }
  n
}

// ─────────────────────────────────────────────────────────────────────────
// Model I/O
// ─────────────────────────────────────────────────────────────────────────

/// Renders a shape/dtype pair for [`ModelError::ContractMismatch`], matching
/// `crate::segment::describe` / `crate::embed::describe`.
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  match dtype {
    Some(dtype) => format!("{shape:?} {dtype:?}"),
    None => format!("{shape:?} (unknown dtype)"),
  }
}

/// Validates one feature's shape and F16 dtype against the pinned contract.
fn check_feature(
  description: &coremlit::ModelDescription,
  feature: &'static str,
  expected: &[usize],
  input: bool,
) -> Result<(), ModelError> {
  let info = if input {
    description.input(feature)
  } else {
    description.output(feature)
  };
  let Some(info) = info else {
    return Err(ModelError::ContractMismatch {
      feature,
      expected: describe(expected, Some(DataType::F16)),
      actual: "absent".to_string(),
    });
  };
  // Every argmax I/O is F16 on EVERY variant, W32A32 included — the one
  // fact that silently corrupts everything if got wrong (module doc).
  if info.shape() != expected || info.data_type() != Some(DataType::F16) {
    return Err(ModelError::ContractMismatch {
      feature,
      expected: describe(expected, Some(DataType::F16)),
      actual: describe(info.shape(), info.data_type()),
    });
  }
  Ok(())
}

/// Reads an F16 output tensor into `f32`, scanning for non-finite values.
///
/// The scan covers only tensors this source CONSUMES (module doc: three of
/// the segmenter's six outputs, plus the fbank). A NaN here would otherwise
/// compare `false` against every threshold and masquerade as "inactive
/// speaker" rather than surfacing the corrupted inference — the same
/// rationale as `crate::window::count_from_segmentations`'s finiteness
/// assert.
fn read_f16_output(
  features: &Features,
  name: &'static str,
  expected: &[usize],
) -> Result<Vec<f32>, InferError> {
  let array = features.get(name).ok_or(InferError::OutputShape {
    got: Vec::new(),
    expected: expected.to_vec(),
  })?;
  if array.shape() != expected {
    return Err(InferError::OutputShape {
      got: array.shape().to_vec(),
      expected: expected.to_vec(),
    });
  }
  let mut raw = vec![f16::ZERO; array.count()];
  array.copy_into(&mut raw)?;
  let values: Vec<f32> = raw.into_iter().map(f32::from).collect();
  if let Some(index) = values.iter().position(|v| !v.is_finite()) {
    return Err(InferError::NonFiniteOutput { index });
  }
  Ok(values)
}

/// The argmax model source: `SpeakerSegmenter` (in-graph decode) +
/// `SpeakerEmbedderPreprocessor` (fbank) + `SpeakerEmbedder`, mapped onto
/// [`Extraction`]. See the module doc for the decode semantics, the index
/// mapping, and every deliberate divergence from argmax's Swift.
#[derive(Debug)]
pub struct ArgmaxSource {
  seg: Model,
  preprocessor: Model,
  embed: Model,
  options: ArgmaxOptions,
}

impl ArgmaxSource {
  /// Loads all three models from an `argmaxinc/speakerkit-coreml` root, using
  /// default [`ArgmaxOptions`].
  ///
  /// # Errors
  /// As [`Self::from_dir_with`].
  pub fn from_dir(root: impl AsRef<Path>) -> Result<Self, ModelError> {
    Self::from_dir_with(root, ArgmaxOptions::new())
  }

  /// Loads all three models from an `argmaxinc/speakerkit-coreml` root with
  /// explicit [`ArgmaxOptions`], resolving the variant's own subdirectories:
  ///
  /// ```text
  /// <root>/speaker_segmenter/pyannote-v3/<variant.segmenter_dir()>/SpeakerSegmenter.mlmodelc
  /// <root>/speaker_embedder/pyannote-v3/<variant.embedder_dir()>/SpeakerEmbedderPreprocessor.mlmodelc
  /// <root>/speaker_embedder/pyannote-v3/<variant.embedder_dir()>/SpeakerEmbedder.mlmodelc
  /// ```
  ///
  /// Every input and output of all three is validated against the pinned
  /// shape/F16 contract (`tests/argmax_model_io.rs`) before returning, so a
  /// model revision that changed a dtype fails at LOAD rather than silently
  /// corrupting a buffer.
  ///
  /// # Errors
  /// [`ModelError::Load`] if CoreML cannot load an artifact;
  /// [`ModelError::ContractMismatch`] if any feature's shape or dtype
  /// diverges from the pinned contract.
  pub fn from_dir_with(root: impl AsRef<Path>, options: ArgmaxOptions) -> Result<Self, ModelError> {
    let root = root.as_ref();
    let variant = options.variant();
    let compute = options.compute();

    let seg = Model::load(seg_path(root, variant), compute.segmenter())?;
    let d = seg.description();
    check_feature(d, names::WAVEFORM, &[ARGMAX_CHUNK_SAMPLES], true)?;
    let per_frame = [
      ARGMAX_WINDOWS_PER_CHUNK,
      ARGMAX_FRAMES_PER_WINDOW,
      ARGMAX_NUM_SPEAKERS,
    ];
    check_feature(d, names::SPEAKER_IDS, &per_frame, false)?;
    check_feature(
      d,
      names::SPEAKER_ACTIVITY,
      &[ARGMAX_WINDOWS_PER_CHUNK, ARGMAX_NUM_SPEAKERS],
      false,
    )?;
    check_feature(
      d,
      names::OVERLAPPED,
      &[ARGMAX_WINDOWS_PER_CHUNK, ARGMAX_FRAMES_PER_WINDOW],
      false,
    )?;
    // Validated but never read (module doc) — pinning them here is what makes
    // "we deliberately ignore these three" checkable rather than an omission.
    check_feature(d, names::SPEAKER_PROBS, &per_frame, false)?;
    check_feature(d, names::VOICE_ACTIVITY, &[ARGMAX_MASK_FRAMES], false)?;
    check_feature(
      d,
      names::SLIDING_WINDOW_WAVEFORM,
      &[ARGMAX_WINDOWS_PER_CHUNK, 1, ARGMAX_WINDOW_SAMPLES],
      false,
    )?;

    let preprocessor = Model::load(preprocessor_path(root, variant), compute.preprocessor())?;
    let d = preprocessor.description();
    let fbank = [1, ARGMAX_FBANK_FRAMES, ARGMAX_FBANK_BINS];
    check_feature(d, names::WAVEFORMS, &[1, ARGMAX_CHUNK_SAMPLES], true)?;
    check_feature(d, names::PREPROCESSOR_OUTPUT, &fbank, false)?;

    let embed = Model::load(embed_path(root, variant), compute.embedder())?;
    let d = embed.description();
    check_feature(d, names::PREPROCESSOR_OUTPUT, &fbank, true)?;
    check_feature(
      d,
      names::SPEAKER_MASKS,
      &[1, ARGMAX_MASK_SLOTS, ARGMAX_MASK_FRAMES],
      true,
    )?;
    check_feature(
      d,
      names::SPEAKER_EMBEDDINGS,
      &[1, ARGMAX_MASK_SLOTS, EMBEDDING_DIM],
      false,
    )?;

    Ok(Self {
      seg,
      preprocessor,
      embed,
      options,
    })
  }

  /// The source's [`ArgmaxOptions`].
  #[inline(always)]
  pub const fn options_ref(&self) -> &ArgmaxOptions {
    &self.options
  }

  /// Runs the segmenter over one padded 30 s chunk, returning the three
  /// decoded tensors this source consumes.
  fn segment_chunk(&self, padded: &MultiArray) -> Result<DecodedChunk, InferError> {
    let out = self.seg.predict_with(&[(names::WAVEFORM, padded)])?;
    Ok(DecodedChunk {
      ids: read_f16_output(
        &out,
        names::SPEAKER_IDS,
        &[
          ARGMAX_WINDOWS_PER_CHUNK,
          ARGMAX_FRAMES_PER_WINDOW,
          ARGMAX_NUM_SPEAKERS,
        ],
      )?,
      activity: read_f16_output(
        &out,
        names::SPEAKER_ACTIVITY,
        &[ARGMAX_WINDOWS_PER_CHUNK, ARGMAX_NUM_SPEAKERS],
      )?,
      overlapped: read_f16_output(
        &out,
        names::OVERLAPPED,
        &[ARGMAX_WINDOWS_PER_CHUNK, ARGMAX_FRAMES_PER_WINDOW],
      )?,
    })
  }

  /// Runs the preprocessor then the embedder over one chunk's padded 30 s
  /// waveform and its `[64, 1767]` mask buffer, returning the raw
  /// `[64, 256]` embedding rows.
  ///
  /// The preprocessor is fed the padded CHUNK waveform, not
  /// `sliding_window_waveform` — see the module doc (`Emb.swift:255-256,348`).
  fn embed_chunk(&self, padded: &[f16], masks: &[f16]) -> Result<Vec<f32>, InferError> {
    let waveforms = MultiArray::from_slice(&[1, ARGMAX_CHUNK_SAMPLES], padded)?;
    let pre_out = self
      .preprocessor
      .predict_with(&[(names::WAVEFORMS, &waveforms)])?;
    let fbank_shape = [1, ARGMAX_FBANK_FRAMES, ARGMAX_FBANK_BINS];
    let fbank = read_f16_output(&pre_out, names::PREPROCESSOR_OUTPUT, &fbank_shape)?;

    let fbank: Vec<f16> = fbank.into_iter().map(f16::from_f32).collect();
    let fbank = MultiArray::from_slice(&fbank_shape, &fbank)?;
    let masks = MultiArray::from_slice(&[1, ARGMAX_MASK_SLOTS, ARGMAX_MASK_FRAMES], masks)?;
    let out = self.embed.predict_with(&[
      (names::PREPROCESSOR_OUTPUT, &fbank),
      (names::SPEAKER_MASKS, &masks),
    ])?;

    // NOT scanned for non-finite values here: the rows this source discards
    // (inactive slots, unbounded windows, the unused 64th slot) are outside
    // the Extraction entirely and dia computes no analog of them
    // (`crate::extract`'s "NonFinite-output scan scope" makes the same
    // distinction). `place_embeddings` scans exactly the CONSUMED rows.
    let array = out
      .get(names::SPEAKER_EMBEDDINGS)
      .ok_or(InferError::OutputShape {
        got: Vec::new(),
        expected: vec![1, ARGMAX_MASK_SLOTS, EMBEDDING_DIM],
      })?;
    let expected = [1, ARGMAX_MASK_SLOTS, EMBEDDING_DIM];
    if array.shape() != expected {
      return Err(InferError::OutputShape {
        got: array.shape().to_vec(),
        expected: expected.to_vec(),
      });
    }
    let mut raw = vec![f16::ZERO; array.count()];
    array.copy_into(&mut raw)?;
    Ok(raw.into_iter().map(f32::from).collect())
  }
}

/// The three decoded segmenter tensors this source consumes, as `f32`.
struct DecodedChunk {
  /// `speaker_ids [21, 589, 3]` — hard 0/1 per-`(window, frame, speaker)`.
  ids: Vec<f32>,
  /// `speaker_activity [21, 3]` — per-`(window, speaker)` active-frame count.
  activity: Vec<f32>,
  /// `overlapped_speaker_activity [21, 589]` — binary overlap indicator.
  overlapped: Vec<f32>,
}

fn seg_path(root: &Path, variant: ArgmaxVariant) -> PathBuf {
  root
    .join("speaker_segmenter/pyannote-v3")
    .join(variant.segmenter_dir())
    .join("SpeakerSegmenter.mlmodelc")
}

fn preprocessor_path(root: &Path, variant: ArgmaxVariant) -> PathBuf {
  root
    .join("speaker_embedder/pyannote-v3")
    .join(variant.embedder_dir())
    .join("SpeakerEmbedderPreprocessor.mlmodelc")
}

fn embed_path(root: &Path, variant: ArgmaxVariant) -> PathBuf {
  root
    .join("speaker_embedder/pyannote-v3")
    .join(variant.embedder_dir())
    .join("SpeakerEmbedder.mlmodelc")
}

/// Writes one bounded window's `speaker_ids` into `segmentations` at global
/// chunk `c` — but ONLY the columns of slots that cleared the activity gate.
///
/// An inactive slot's column is left zero, upholding [`Extraction`]'s
/// invariant that a slot with an all-zero embedding row has an all-zero
/// segmentation column (module doc). argmax's activity gate is strictly more
/// aggressive than dia's ("more than 2 active frames" vs "any active frame"),
/// so a slot with 1-2 active frames is dropped here where dia would keep it.
fn write_segmentations(
  c: usize,
  w: usize,
  ids: &[f32],
  plan: &WindowPlan,
  segmentations: &mut [f64],
) {
  let slab = &mut segmentations[chunk_segmentation_range(c)];
  for (s, &is_active) in plan.active.iter().enumerate() {
    if !is_active {
      continue;
    }
    for f in 0..ARGMAX_FRAMES_PER_WINDOW {
      slab[f * SEG_NUM_SLOTS + s] = f64::from(ids[ids_index(w, f, s)]);
    }
  }
}

/// Places one bounded window's consumed embedding rows into `raw_embeddings`,
/// dropping (row left zero + segmentation column zeroed) any slot that fails
/// the degenerate-mask or PLDA-norm guard.
///
/// `embeddings` is the embedder's flat `[64, 256]` output; the row for
/// `(w, s)` is `w * 3 + s` (`Emb.swift:288,291`).
///
/// # Errors
/// [`InferError::NonFiniteOutput`] if a CONSUMED row holds a NaN/infinity —
/// a hard error, matching dia (`owned.rs:611-618`) and this crate's policy
/// that `NonFiniteOutput` is never a silent drop. The flat index reported is
/// into the model's own `speaker_embeddings` output.
fn place_embeddings(
  c: usize,
  w: usize,
  plan: &WindowPlan,
  masks: &[f16],
  embeddings: &[f32],
  raw_embeddings: &mut [f32],
  segmentations: &mut [f64],
) -> Result<(), InferError> {
  for (s, &is_active) in plan.active.iter().enumerate() {
    if !is_active {
      continue;
    }
    let row = w * ARGMAX_NUM_SPEAKERS + s;

    // Degenerate-mask guard (module doc): an all-zero mask row yields a
    // FINITE constant (L2 ≈ 0.5356) that the norm guard below cannot catch.
    if mask_row_is_zero(masks, row) {
      zero_slot_column(&mut segmentations[chunk_segmentation_range(c)], s);
      continue;
    }

    let embedding = &embeddings[row * EMBEDDING_DIM..(row + 1) * EMBEDDING_DIM];
    if let Some(offset) = embedding.iter().position(|v| !v.is_finite()) {
      return Err(InferError::NonFiniteOutput {
        index: row * EMBEDDING_DIM + offset,
      });
    }

    // Same f64 norm pre-check dia applies (`owned.rs:619-630`).
    let norm_sq: f64 = embedding
      .iter()
      .map(|v| f64::from(*v) * f64::from(*v))
      .sum();
    if norm_sq.sqrt() < PLDA_MIN_NORM {
      zero_slot_column(&mut segmentations[chunk_segmentation_range(c)], s);
    } else {
      raw_embeddings[embedding_range(c, s)].copy_from_slice(embedding);
    }
  }
  Ok(())
}

impl ModelSource for ArgmaxSource {
  /// Maps argmax's in-graph-decoded output onto [`Extraction`]. See the
  /// module doc for the decode semantics, the `(k, w) → c = k*21 + w` index
  /// mapping, and every deliberate divergence from argmax's Swift.
  ///
  /// # Errors
  /// - [`ExtractError::EmptySamples`] if `samples` is empty (mirrors
  ///   `owned.rs:369-371`, and argmax's own chunk loop would produce no
  ///   chunk at all).
  /// - [`ExtractError::UnsupportedStepSamples`] if the configured
  ///   `step_samples` is not [`ARGMAX_WINDOW_STRIDE_SAMPLES`] — argmax's
  ///   window stride is compiled into its graph and cannot be varied (module
  ///   doc). Rejected rather than silently ignored.
  /// - [`ExtractError::OnsetOutOfRange`] if `onset` is not finite in
  ///   `(0.0, 1.0]` (same guard as [`crate::extract::Extractor::extract`]).
  /// - [`ExtractError::Infer`] if any of the three models fails, an output's
  ///   shape diverges, or a CONSUMED tensor holds a non-finite value.
  /// - [`ExtractError::OutputFrameCountOverflow`] if the derived
  ///   `num_output_frames` would not fit in `usize` (unreachable through this
  ///   geometry; kept typed, as in [`crate::extract`]).
  fn extract(&self, samples: &[f32]) -> Result<Extraction, ExtractError> {
    if samples.is_empty() {
      return Err(ExtractError::EmptySamples);
    }
    let w_opts = self.options.window();
    if w_opts.step_samples() as usize != ARGMAX_WINDOW_STRIDE_SAMPLES {
      return Err(ExtractError::UnsupportedStepSamples {
        step: w_opts.step_samples(),
        required: ARGMAX_WINDOW_STRIDE_SAMPLES as u32,
      });
    }
    if !crate::window::check_onset(w_opts.onset()) {
      return Err(ExtractError::OnsetOutOfRange {
        onset: w_opts.onset(),
      });
    }

    // The Extraction chunk grid IS dia's (module doc's theorem), so it is
    // computed from the very same function FluidAudioSource uses — the two
    // sources' geometry agrees by construction, not by coincidence.
    let num_chunks = crate::window::chunk_starts(samples.len(), &w_opts).len();
    let mut segmentations = vec![0.0f64; num_chunks * ARGMAX_FRAMES_PER_WINDOW * SEG_NUM_SLOTS];
    let mut raw_embeddings = vec![0.0f32; num_chunks * SEG_NUM_SLOTS * EMBEDDING_DIM];
    let mut padded = vec![f16::ZERO; ARGMAX_CHUNK_SAMPLES];

    for (k, &start) in argmax_chunk_starts(samples.len()).iter().enumerate() {
      // The UNPADDED length drives `bounded()` (argmax's `waveformLength`).
      let chunk_len = fill_padded_chunk(&mut padded, samples, start);
      let waveform =
        MultiArray::from_slice(&[ARGMAX_CHUNK_SAMPLES], &padded).map_err(InferError::from)?;
      let decoded = self.segment_chunk(&waveform)?;

      let plans = window_plans(chunk_len, &decoded.activity);
      for (w, plan) in plans.iter().enumerate() {
        if !plan.bounded {
          continue; // Emb.swift:285 — this window runs past the real audio.
        }
        let c = global_chunk(k, w);
        debug_assert!(
          c < num_chunks,
          "bounded window (k={k}, w={w}) -> c={c} must index the dia chunk grid \
           (num_chunks={num_chunks}); the module doc's grid theorem is violated"
        );
        write_segmentations(c, w, &decoded.ids, plan, &mut segmentations);
      }

      // A chunk in which no window has an active speaker has no consumed
      // embedding row, so it makes no preprocessor/embedder call at all —
      // mirroring dia's "no planned slot -> zero embed calls"
      // (`crate::extract`'s module doc).
      if !plans.iter().any(WindowPlan::any_active) {
        continue;
      }
      let masks = build_speaker_masks(&decoded.ids, &decoded.overlapped, &plans);
      let embeddings = self.embed_chunk(&padded, &masks)?;
      for (w, plan) in plans.iter().enumerate() {
        if !plan.bounded {
          continue;
        }
        place_embeddings(
          global_chunk(k, w),
          w,
          plan,
          &masks,
          &embeddings,
          &mut raw_embeddings,
          &mut segmentations,
        )?;
      }
    }

    // `count` over the POST-zeroing buffer, on dia's own grid — identical to
    // FluidAudioSource's (`crate::extract`'s "Count runs after all zeroing").
    let chunks_sw = crate::window::chunk_sliding_window(&w_opts);
    let frames_sw = crate::window::frame_sliding_window();
    let count = crate::window::try_count_from_segmentations(
      &segmentations,
      num_chunks,
      ARGMAX_FRAMES_PER_WINDOW,
      SEG_NUM_SLOTS,
      w_opts.onset(),
      chunks_sw,
      frames_sw,
    )
    .map_err(|e| match e {
      crate::window::WindowError::OutputFrameCountOverflow => {
        ExtractError::OutputFrameCountOverflow
      }
    })?;

    Ok(Extraction::from_parts(
      raw_embeddings,
      segmentations,
      count,
      num_chunks,
      ARGMAX_FRAMES_PER_WINDOW,
      chunks_sw,
      frames_sw,
    ))
  }
}

/// Compile-time proof that argmax's speaker/embedding dimensions ARE the ones
/// [`Extraction`]'s layout is built from — if a future argmax revision moved
/// either, the mapping's index arithmetic would silently mean something else.
const _: () = {
  assert!(ARGMAX_NUM_SPEAKERS == SEG_NUM_SLOTS);
  assert!(ARGMAX_NUM_SPEAKERS == EMBED_SLOTS);
  assert!(ARGMAX_WINDOW_SAMPLES == SEG_CHUNK_SAMPLES);
  assert!(ARGMAX_WINDOW_STRIDE_SAMPLES == crate::window::DEFAULT_STEP_SAMPLES as usize);
  // The identity the whole `c = k*21 + w` mapping rests on (module doc).
  assert!(ARGMAX_CHUNK_HOP_SAMPLES == ARGMAX_WINDOWS_PER_CHUNK * ARGMAX_WINDOW_STRIDE_SAMPLES);
  assert!(ARGMAX_MASK_FRAMES == 1767);
};

#[cfg(test)]
mod tests;
