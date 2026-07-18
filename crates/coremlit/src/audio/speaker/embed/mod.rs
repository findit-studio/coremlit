//! CoreML wrapper for `wespeaker_v2.mlmodelc` (spec §4): raw waveform +
//! per-frame speaker-activity mask in, raw (un-normalized) 256-d WeSpeaker
//! embeddings out, batched across all 3 pyannote speaker slots per call.
//!
//! Ports the model-facing half of dia's `embed` stage —
//! `EmbedModel::embed_chunk_with_frame_mask`
//! (`diarization/src/embed/model.rs:611-667`) — over `coremlit` instead of
//! `ort`, plus FluidAudio's raw-WeSpeaker CoreML wrapper,
//! `EmbeddingExtractor.swift` (permalinks below), for the CoreML-specific
//! input-preparation scheme dia's ONNX path has no equivalent of. Ground
//! truth: `tests/model_io.rs`'s `wespeaker_v2_io_matches_spec`
//! introspection test (`wespeaker_v2.mlmodelc`: `waveform [3, 160_000]` f32
//! plus `mask [3, 589]` f32 in, `embedding [3, 256]` f32 out, plus an
//! undocumented scalar `constant` output T1 already established this crate
//! ignores) and the design spec §4/§5.
//!
//! # dia contract match
//!
//! - **Un-normalized, raw output.** dia's `embed_chunk_with_frame_mask`
//!   returns the backend's raw embedding directly — "Returns the raw
//!   (un-normalized) 256-d embedding for the speaker whose activity is in
//!   `frame_mask`" (`model.rs:599-601`) — L2 normalization is a
//!   HIGHER-level concern (`Embedding::normalize_from`, only reachable from
//!   `embed`/`embed_weighted`/`embed_masked`, never from
//!   `embed_chunk_with_frame_mask`). [`EmbedModel::embed_chunk`] and
//!   [`EmbedModel::embed_chunk_with_frame_mask`] do the same: no
//!   normalization anywhere in this module. Model-gated tests assert
//!   real-audio output is NOT unit-norm, so a future accidental
//!   normalization regresses visibly.
//! - **Mask dtype: `f32` 0.0/1.0, not boolean.** dia's ORT and tch embed
//!   backends both convert the boolean `frame_mask` the identical way
//!   before handing it to the model — `frame_mask.iter().map(|&b| if b {
//!   1.0 } else { 0.0 }).collect()` (`model.rs:296-299` ORT,
//!   `model.rs:374-377` tch) — because the model's `weights`/`mask` input
//!   is `f32`, matching this crate's own introspected `mask [3, 589]` f32
//!   contract. `mask_row_f32` (private) performs the identical conversion.
//! - **Empty-mask rejection.** dia's `embed_chunk_with_frame_mask` rejects
//!   a `frame_mask` with no active (`true`) entry at all —
//!   `if !frame_mask.iter().any(|&b| b) { return
//!   Err(Error::EmptyOrInactiveMask); }` (`model.rs:646-649`) — because an
//!   all-zero mask means all-zero pooling weights, which divides by zero
//!   inside WeSpeaker's statistics-pooling layer and yields NaN/Inf.
//!   [`EmbedModel::embed_chunk_with_frame_mask`] mirrors this exactly via
//!   `check_mask_active` (private) and [`InferError::EmptyMask`] — see the
//!   "Scope" section below for why this is the ONLY mask-validity check
//!   dia's ported function performs (no cross-slot "clean" logic here).
//! - **`EMBEDDING_DIM = 256`** matches dia's `EMBEDDING_DIM`
//!   (`diarization/src/embed/options.rs:25`) and the introspected
//!   `embedding` output's trailing dimension.
//! - **`EMBED_SLOTS = 3`** matches dia's `SLOTS_PER_CHUNK`
//!   (`diarization/src/offline/owned.rs:41`) / `MAX_SPEAKER_SLOTS`
//!   (`diarization/src/segment/options.rs:43`, already cited by
//!   [`crate::audio::speaker::segment::SEG_NUM_SLOTS`]) and the introspected `waveform`/
//!   `mask`/`embedding` tensors' shared leading dimension.
//! - **`&self`, not `&mut self`.** dia's `embed_chunk_with_frame_mask` is
//!   `&mut self` over a `!Sync` ort session with input scratch
//!   (`model.rs:611-615`). `crate::Model` is `Send` (but deliberately
//!   NOT `Sync` — Apple documents `MLModel` prediction as
//!   one-thread-at-a-time; `coremlit/src/model/mod.rs` carries only
//!   `unsafe impl Send`, with a `compile_fail` doctest pinning `!Sync`)
//!   and predicts from borrowed inputs with no mutable scratch, so this
//!   module's methods take `&self` — the same documented divergence
//!   [`crate::audio::speaker::segment::SegmentModel::infer`] already makes. Fan-out
//!   therefore means one [`EmbedModel`] per worker (or external
//!   synchronization), not a shared `Arc`.
//!
//! # Deliberate divergence from dia: pad, don't reject
//!
//! Unlike [`crate::audio::speaker::segment::SegmentModel::infer`] (which REJECTS any
//! non-`SEG_CHUNK_SAMPLES` input, matching dia's own segment-side
//! reject-not-pad contract) — and unlike dia's OWN
//! `embed_chunk_with_frame_mask`, which ALSO rejects on exact-length
//! mismatch (`ChunkSamplesShapeMismatch`/`FrameMaskShapeMismatch`,
//! `model.rs:630-643`) — [`EmbedModel::embed_chunk`] and
//! [`EmbedModel::embed_chunk_with_frame_mask`] accept `samples`/masks of
//! ANY length and repeat-pad (or truncate) internally. This is an
//! intentional, spec-mandated divergence sourced from FluidAudio, not a
//! dia-parity behavior: design spec §4 states plainly, "WeSpeaker padding
//! is loop/repeat-doubling (not zero) for waveform and mask" — see the
//! next section for the source and the exact scheme.
//!
//! # FluidAudio's repeat-padding scheme (waveform + mask)
//!
//! Source: FluidAudio's raw WeSpeaker CoreML wrapper, `EmbeddingExtractor`
//! — `fillWaveformBuffer` and `fillMaskBufferOptimized` — pinned at commit
//! `d2937a81747c20ce76476a66d18c80de7e537d78` (FluidAudio tracks `main`
//! with no revision pinning of its own; SHA-pinned here per the design
//! spec's own instruction to do so at read time):
//! <https://github.com/FluidInference/FluidAudio/blob/d2937a81747c20ce76476a66d18c80de7e537d78/Sources/FluidAudio/Diarizer/Extraction/EmbeddingExtractor.swift#L117-L199>
//!
//! Both Swift functions run the identical "doubling-copy" loop: copy the
//! source into the start of a (zero-cleared,
//! `ANEMemoryOptimizer.swift#L18-L33`, `zeroClear: true`) destination
//! buffer, then repeatedly `vDSP_mmov` `min(filled, remaining)` elements
//! from the START of the buffer to its current END, doubling the filled
//! region each iteration until full:
//!
//! ```swift
//! while sampleCount < requiredCount {
//!     let copyCount = min(sampleCount, requiredCount - sampleCount)
//!     vDSP_mmov(ptr, ptr.advanced(by: sampleCount), vDSP_Length(copyCount), ...)
//!     sampleCount += copyCount
//! }
//! ```
//!
//! `repeat_pad_f32` (private) implements the mathematically equivalent
//! closed form, `out[i] = source[i % source.len()]` (periodic tiling) —
//! see its own doc comment for the equivalence proof (by induction, the
//! filled length stays a multiple of `source.len()` at every step before
//! the final, possibly-partial one) — and this equivalence is additionally
//! cross-checked empirically in a test-only `doubling_copy_simulation` (a
//! direct Rust transliteration of the Swift loop above) against several
//! non-power-of-2 lengths, per this task's brief instruction to verify the
//! loop-pad behavior empirically, not just by reading the source.
//!
//! Two edge cases are this crate's OWN choice, not read off FluidAudio
//! (documented precisely on `repeat_pad_f32`'s own doc comment): an empty
//! source pads to all-zero (the NET EFFECT of FluidAudio's own zero-length
//! guard on an already-zero-cleared buffer), and a source at-or-past the
//! target length truncates (FluidAudio's own handling of audio longer
//! than one 10 s chunk is not a clean per-row analog — `optimizedCopy` bounds against
//! the FULL 3-row destination, not one row,
//! `ANEMemoryOptimizer.swift#L116-138` — and the mask-side analog of this
//! exact formula had a documented heap-overread bug for long audio before
//! FluidAudio clamped it, PR #191,
//! `Tests/FluidAudioTests/Diarizer/Extraction/EmbeddingExtractorOverflowTests.swift`
//! at the pinned SHA — so FluidAudio itself has no single clean contract
//! here to match).
//!
//! # Batching design: diverges from FluidAudio's "wasted" batch dim
//!
//! FluidAudio's `EmbeddingExtractor.getEmbeddings` processes one speaker
//! at a time: it fills the shared `[3, 160_000]` waveform buffer's row 0
//! ONCE per chunk with the real (repeat-padded) audio and never writes
//! rows 1-2 (left zero from allocation) — "Fill shared waveform buffer
//! once; reused across speakers"
//! (`EmbeddingExtractor.swift#L54-58`) — then, per speaker, zero-fills the
//! WHOLE mask buffer and writes only row 0 with that speaker's
//! (repeat-padded) mask (`EmbeddingExtractor.swift#L160-179`), runs
//! inference, and reads back only row 0 of the output —
//! `extractEmbeddingOptimized(from: embeddingArray, speakerIndex: 0)`
//! (`EmbeddingExtractor.swift#L99-111`) — discarding whatever the model
//! computed for rows 1-2. That's the "wastes the batch dim" the design
//! spec refers to: 2 of every 3 batch slots compute output nobody reads,
//! on every one of the 3 per-chunk calls.
//!
//! [`EmbedModel::embed_chunk`] instead computes all 3 slots' REAL
//! embeddings in a single call (design spec §4: "dia-coreml batches all 3
//! slots per call") — the whole reason this crate's `embed_chunk` exists
//! as a batched primitive dia has no equivalent of. Consequently the
//! private `build_waveform` fills EVERY row with the same repeat-padded
//! `samples` (`embed_chunk`'s signature takes exactly one shared `samples`
//! buffer, matching how dia's OWN pipeline reuses one `padded_chunk` audio
//! buffer across all 3 speaker slots and varies only the per-slot mask,
//! `diarization/src/offline/owned.rs:524-534`) — a deliberate,
//! FluidAudio-diverging choice, not an oversight: this module's waveform
//! input has no way to carry 3 independent audios even if it wanted to.
//!
//! # Scope: the `< 2` clean-frames overlap exclusion is NOT here
//!
//! Both this crate's design spec (§2 item 2) and this task's brief flag
//! "dia's `< 2` clean-frames fallback semantics" as something to read and
//! match. Having read it in both reference implementations end to end,
//! the precise finding is: **this concept does not live inside the
//! function this module ports, in either reference implementation.**
//!
//! - **dia**: `EmbedModel::embed_chunk_with_frame_mask` itself
//!   (`model.rs:611-667`) takes a single, ALREADY-DECIDED `frame_mask` and
//!   rejects only the fully-degenerate all-inactive case (`model.rs:
//!   646-649`, see "dia contract match" above) — it has no "clean" vs
//!   "overlapping" concept, and structurally cannot: that requires
//!   knowing whether OTHER speakers are active at the same frame, which a
//!   single boolean mask parameter cannot carry. The actual `< 2` logic —
//!   pyannote's `embedding_exclude_overlap` (`min_num_frames = 2`) — lives
//!   one layer up, in dia's OFFLINE PIPELINE, which holds all 3 slots'
//!   segmentation simultaneously and builds a cross-slot "clean" mask
//!   BEFORE ever calling `embed_chunk_with_frame_mask`:
//!   `clean_frame[f] = active_count < 2` (fewer than 2 of the 3 slots
//!   concurrently active at frame `f`), `EXCLUDE_OVERLAP_MIN_FRAMES = 2`,
//!   fall back to the raw mask when `clean_count <= EXCLUDE_OVERLAP_MIN_FRAMES`
//!   (`diarization/src/offline/owned.rs:507-591`). dia's streaming path
//!   (`streaming/offline_diarizer.rs`, `build_range`) notably does NOT
//!   apply this exclusion — it embeds with the raw any-active mask, a
//!   real offline-vs-streaming asymmetry inside dia itself, which is why
//!   only `owned.rs` is citable as the exclusion's source of truth.
//! - **FluidAudio corroborates the identical layering, independently.**
//!   Its raw WeSpeaker wrapper — `EmbeddingExtractor.getEmbeddings`, the
//!   direct Swift analog of [`EmbedModel`] and the file cited throughout
//!   this module doc — takes pre-built `masks: [[Float]]` as given and has
//!   no overlap concept at all (it has a different, simpler per-speaker
//!   activity-sum floor instead — see the note below on why this crate
//!   does not adopt it). The `isClean`/overlap-exclusion logic
//!   (`overlapFrames` from `active > 1`, `cleanMask`, a
//!   `minFramesForEmbedding` fallback) lives in a SEPARATE, higher-level
//!   file: `OfflineEmbeddingExtractor.swift`, `processChunk`
//!   (<https://github.com/FluidInference/FluidAudio/blob/d2937a81747c20ce76476a66d18c80de7e537d78/Sources/FluidAudio/Diarizer/Offline/Extraction/OfflineEmbeddingExtractor.swift#L421-L534>)
//!   — the orchestrator that already has every slot's segmentation in
//!   hand, exactly mirroring dia's own layering.
//!
//! Per this crate's plan (`docs/superpowers/plans/2026-07-12-dia-coreml.md`,
//! Task 5), deriving that cross-slot "clean" mask is a future
//! `Extractor::extract`'s job, not [`EmbedModel`]'s: `Extractor` will hold
//! all 3 slots' `multilabel` output at the call site — the same
//! information dia's `offline/owned.rs` and FluidAudio's
//! `OfflineEmbeddingExtractor.swift` both require, and which
//! [`EmbedModel::embed_chunk_with_frame_mask`]'s single-mask signature
//! cannot carry. This module's `embed_chunk_with_frame_mask` takes
//! whatever mask its caller has already decided on (raw or
//! overlap-excluded) — exactly dia's own ported function's contract.
//!
//! **On FluidAudio's `minActivityThreshold` guard specifically**: this
//! crate does NOT adopt it. dia is this task's adjudicated parity oracle
//! for mask-VALIDITY semantics specifically (FluidAudio is cited for the
//! input-prep padding SCHEME); dia's `embed_chunk_with_frame_mask` has no
//! activity floor beyond "not literally all-inactive"
//! (`EmptyOrInactiveMask`), and silently returning an all-zero embedding
//! for a low-but-nonzero-activity mask (FluidAudio's
//! `speakerActivity < minActivityThreshold` behavior,
//! `EmbeddingExtractor.swift#L69-77`) is a materially different contract
//! from dia's hard error — adopting it here would silently diverge from
//! the function this module claims to port.
//!
//! # NonFinite-output scan scope: 768 vs. 256
//!
//! [`EmbedModel::embed_chunk`] scans ALL `EMBED_SLOTS * EMBEDDING_DIM =
//! 768` output values — the gate-2 failure mode this crate exists to
//! catch (spec §6 gate 2: CoreML-EP NaN/Inf corruption on legitimate
//! input). [`EmbedModel::embed_chunk_with_frame_mask`] scans only its OWN
//! returned 256-element row, matching dia's `embed_chunk_with_frame_mask`
//! exactly: dia's ONNX call for this function is `n = 1` (`model.rs:301`,
//! `run_inference(&mut self.session, 1, ...)`) — dia's ported function has
//! no "other slots" concept whatsoever, so it only ever checks the one
//! output it computes (`model.rs:663-665`). This split matters
//! operationally, not just for citation-fidelity: slots 1-2 of
//! `embed_chunk_with_frame_mask`'s internal batched call are deliberately
//! fed an EMPTY mask (see its own doc), which — per the SAME
//! divide-by-zero mechanism [`InferError::EmptyMask`] exists to prevent —
//! is expected to make WeSpeaker's statistics-pooling layer emit NaN/Inf
//! for those UNUSED rows. A blanket 768-wide scan would make
//! `embed_chunk_with_frame_mask` fail on every call for a reason that has
//! nothing to do with the one embedding it actually returns; scanning
//! only row 0 avoids that while `embed_chunk`'s own contract (3 REAL,
//! caller-supplied masks) keeps the blanket scan meaningful there.

use std::path::Path;

use crate::{ComputeUnits, DataType, Model, MultiArray};

use crate::audio::speaker::error::{InferError, ModelError};

/// Output dimensionality of the WeSpeaker embedding. Matches dia's
/// `EMBEDDING_DIM` (`diarization/src/embed/options.rs:25`) and the
/// introspected `wespeaker_v2.mlmodelc` `embedding` output's trailing
/// dimension (`tests/model_io.rs::wespeaker_v2_io_matches_spec`).
pub const EMBEDDING_DIM: usize = 256;

/// Fixed pyannote speaker-slot count `wespeaker_v2.mlmodelc`'s `waveform`/
/// `mask`/`embedding` tensors all share as their leading dimension
/// (`[3, 160_000]` / `[3, 589]` / `[3, 256]`,
/// `tests/model_io.rs::wespeaker_v2_io_matches_spec`). Matches dia's
/// `SLOTS_PER_CHUNK` (`diarization/src/offline/owned.rs:41`) /
/// `MAX_SPEAKER_SLOTS` (`diarization/src/segment/options.rs:43`) and this
/// crate's own [`crate::audio::speaker::segment::SEG_NUM_SLOTS`].
pub const EMBED_SLOTS: usize = 3;

/// Declared feature names on `wespeaker_v2.mlmodelc`
/// (`tests/model_io.rs::wespeaker_v2_io_matches_spec`). The model's second
/// output, `constant` (an undocumented fixed-shape scalar — T1's module
/// doc, `tests/model_io.rs` items 2-3), is intentionally absent here: this
/// module never reads or validates it, matching T1/T2 precedent.
mod names {
  pub const WAVEFORM: &str = "waveform";
  pub const MASK: &str = "mask";
  pub const EMBEDDING: &str = "embedding";
}

/// Default [`EmbedModelOptions::compute`]. `ComputeUnits::All` lets CoreML
/// schedule across ANE/GPU/CPU (design spec §1's ~30x embedding uplift
/// target). Model-gated tests in this module instead load with
/// `ComputeUnits::CpuOnly` for determinism, matching
/// [`crate::audio::speaker::segment::DEFAULT_SEGMENT_COMPUTE`]'s and `tests/model_io.rs`'s
/// convention — production code keeps this default.
pub const DEFAULT_EMBED_COMPUTE: ComputeUnits = ComputeUnits::All;

#[cfg(feature = "serde")]
fn default_embed_compute() -> ComputeUnits {
  DEFAULT_EMBED_COMPUTE
}

/// Construction options for [`EmbedModel`] (rust-options-pattern). Mirrors
/// [`crate::audio::speaker::segment::SegmentModelOptions`] exactly — a single `compute`
/// knob, `const new`/`Default` sharing one source of truth, `with_`/`set_`
/// pair — down to sharing its `ComputeUnits` serde bridge
/// (the crate-private `compute_units_serde` module, factored out of
/// `segment` during this task rather than copied a third time — see that
/// module's doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbedModelOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_embed_compute",
      with = "crate::audio::speaker::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for EmbedModelOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl EmbedModelOptions {
  /// Options matching the crate's default: [`DEFAULT_EMBED_COMPUTE`]
  /// (`ComputeUnits::All`).
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_EMBED_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the embedding model on.
  #[inline(always)]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }
  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline(always)]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }
  /// Sets [`Self::compute`] in place.
  #[inline(always)]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// Human-readable `shape dtype` rendering for
/// [`ModelError::ContractMismatch`]'s `actual`/`expected` fields. Same
/// tiny helper as `crate::audio::speaker::segment`'s private `describe` — deliberately
/// NOT unified with it: this task's review queue named only the
/// `ComputeUnits` serde bridge (the crate-private `compute_units_serde`
/// module) for sharing, so this one-off duplication is left as a future
/// maintenance-pass candidate rather than an unrequested refactor of
/// already-shipped, already-tested code.
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

/// CoreML wrapper over `wespeaker_v2.mlmodelc`: batched `[EMBED_SLOTS,
/// SEG_CHUNK_SAMPLES]` waveform + `[EMBED_SLOTS, num_mask_frames]` mask in,
/// `[EMBED_SLOTS, EMBEDDING_DIM]` raw embeddings out — see the module doc
/// for the full dia/FluidAudio contract match.
#[derive(Debug)]
pub struct EmbedModel {
  model: Model,
  num_mask_frames: usize,
}

impl EmbedModel {
  /// Loads the model with [`EmbedModelOptions::new`] (`ComputeUnits::All`).
  ///
  /// # Errors
  /// As [`Self::from_file_with`].
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
    Self::from_file_with(path, EmbedModelOptions::new())
  }

  /// Loads the model with custom options, introspecting and validating its
  /// I/O contract against the ground truth pinned by
  /// `tests/model_io.rs::wespeaker_v2_io_matches_spec`.
  ///
  /// # Errors
  /// [`ModelError::Load`] if CoreML rejects the model.
  /// [`ModelError::ContractMismatch`] if the loaded model's `waveform`
  /// input isn't `[EMBED_SLOTS, SEG_CHUNK_SAMPLES]` f32, its `mask` input
  /// isn't rank 2 with a leading dimension of `EMBED_SLOTS` and at least
  /// one frame, or its `embedding` output isn't `[EMBED_SLOTS,
  /// EMBEDDING_DIM]` f32. The mask frame count (`shape[1]`) is read
  /// dynamically, not hardcoded — see [`Self::num_mask_frames`] and this
  /// task's brief ("F comes from the model's declared contract — discover
  /// it dynamically at load ... NEVER hardcode 589").
  pub fn from_file_with(
    path: impl AsRef<Path>,
    options: EmbedModelOptions,
  ) -> Result<Self, ModelError> {
    let model = Model::load(path, options.compute())?;
    let description = model.description();

    let waveform_expected = format!(
      "[{EMBED_SLOTS}, {}] float32",
      crate::audio::speaker::segment::SEG_CHUNK_SAMPLES
    );
    let waveform =
      description
        .input(names::WAVEFORM)
        .ok_or_else(|| ModelError::ContractMismatch {
          feature: names::WAVEFORM,
          expected: waveform_expected.clone(),
          actual: "missing".to_string(),
        })?;
    if waveform.shape()
      != [
        EMBED_SLOTS,
        crate::audio::speaker::segment::SEG_CHUNK_SAMPLES,
      ]
      || waveform.data_type() != Some(DataType::F32)
    {
      return Err(ModelError::ContractMismatch {
        feature: names::WAVEFORM,
        expected: waveform_expected,
        actual: describe(waveform.shape(), waveform.data_type()),
      });
    }

    let mask_expected = format!("[{EMBED_SLOTS}, >=1] float32");
    let mask = description
      .input(names::MASK)
      .ok_or_else(|| ModelError::ContractMismatch {
        feature: names::MASK,
        expected: mask_expected.clone(),
        actual: "missing".to_string(),
      })?;
    let mask_shape = mask.shape();
    // `mask_shape[1] >= 1`: a zero-frame contract would "load fine" and
    // then make every embed call build a zero-length mask row — reject
    // the degenerate contract at construction instead, mirroring
    // `crate::audio::speaker::segment::SegmentModel::from_file_with`'s identical
    // `shape[1] >= 1` guard on `segments`.
    let mask_shape_ok = mask_shape.len() == 2 && mask_shape[0] == EMBED_SLOTS && mask_shape[1] >= 1;
    if !mask_shape_ok || mask.data_type() != Some(DataType::F32) {
      return Err(ModelError::ContractMismatch {
        feature: names::MASK,
        expected: mask_expected,
        actual: describe(mask_shape, mask.data_type()),
      });
    }
    let num_mask_frames = mask_shape[1];

    let embedding_expected = format!("[{EMBED_SLOTS}, {EMBEDDING_DIM}] float32");
    let embedding =
      description
        .output(names::EMBEDDING)
        .ok_or_else(|| ModelError::ContractMismatch {
          feature: names::EMBEDDING,
          expected: embedding_expected.clone(),
          actual: "missing".to_string(),
        })?;
    if embedding.shape() != [EMBED_SLOTS, EMBEDDING_DIM]
      || embedding.data_type() != Some(DataType::F32)
    {
      return Err(ModelError::ContractMismatch {
        feature: names::EMBEDDING,
        expected: embedding_expected,
        actual: describe(embedding.shape(), embedding.data_type()),
      });
    }

    Ok(Self {
      model,
      num_mask_frames,
    })
  }

  /// Mask frame count (`F`) — the introspected `mask` shape's trailing
  /// dimension (589 for `wespeaker_v2.mlmodelc`, pinned by
  /// `tests/model_io.rs::wespeaker_v2_io_matches_spec`; read dynamically at
  /// construction, never hardcoded).
  #[inline(always)]
  pub const fn num_mask_frames(&self) -> usize {
    self.num_mask_frames
  }

  /// Batched call: one shared chunk of audio, three independent per-slot
  /// speaker-activity masks in, three raw (un-normalized) embeddings out —
  /// design spec §4's "dia-coreml batches all 3 slots per call" (dia has
  /// no equivalent; see the module doc's "Batching design" section).
  ///
  /// `samples` is repeat-padded (or truncated) to `SEG_CHUNK_SAMPLES` and
  /// used identically for every slot's waveform row; each `masks[i]` is
  /// independently converted to `f32` and repeat-padded (or truncated) to
  /// [`Self::num_mask_frames`] — see the private `repeat_pad_f32` and the
  /// module doc's "FluidAudio's repeat-padding scheme" section. Unlike
  /// [`Self::embed_chunk_with_frame_mask`], an individual `masks[i]` with
  /// no active frame is NOT rejected here — a genuinely empty per-slot
  /// mask is a legitimate input at this permissive, general-purpose layer
  /// (the caller may deliberately want an unused slot).
  ///
  /// # Errors
  /// [`InferError::NonFiniteInput`] if `samples` contains NaN/infinity.
  /// [`InferError::Tensor`] / [`InferError::Prediction`] on a
  /// tensor-construction or CoreML failure. [`InferError::OutputShape`] if
  /// the predict-time `embedding` tensor's shape diverges from
  /// `[EMBED_SLOTS, EMBEDDING_DIM]` — re-checked on every call for the
  /// same CoreML-runtime-is-a-trust-boundary reason
  /// [`crate::audio::speaker::segment::SegmentModel::infer`] re-checks its own output
  /// shape (see that module's doc). [`InferError::NonFiniteOutput`] if ANY
  /// of the `EMBED_SLOTS * EMBEDDING_DIM` output values is NaN/infinite —
  /// see the module doc's "NonFinite-output scan scope" section for why
  /// this scans the FULL batched output, unlike
  /// [`Self::embed_chunk_with_frame_mask`].
  pub fn embed_chunk(
    &self,
    samples: &[f32],
    masks: &[&[bool]; EMBED_SLOTS],
  ) -> Result<[[f32; EMBEDDING_DIM]; EMBED_SLOTS], InferError> {
    let flat = self.run_batched(samples, masks)?;
    check_finite_output(&flat)?;
    let mut out = [[0.0f32; EMBEDDING_DIM]; EMBED_SLOTS];
    for (row, chunk) in out.iter_mut().zip(flat.as_chunks::<EMBEDDING_DIM>().0) {
      row.copy_from_slice(chunk);
    }
    Ok(out)
  }

  /// dia's single-slot `embed_chunk_with_frame_mask` contract
  /// (`diarization/src/embed/model.rs:611-667`) as a veneer over
  /// [`Self::embed_chunk`]: `frame_mask` becomes slot 0's mask, slots 1-2
  /// get an empty mask (which the private `repeat_pad_f32` zero-fills —
  /// see its own doc), and only slot 0's embedding is returned. See the
  /// module doc's "Scope" section for exactly which parts of dia's contract this
  /// mirrors (the empty-mask rejection, the un-normalized raw output) and
  /// which parts it deliberately does NOT (the cross-slot `< 2`
  /// clean-frames overlap exclusion — out of scope for a single-mask
  /// function in both dia and FluidAudio).
  ///
  /// # Errors
  /// [`InferError::EmptyMask`] if `frame_mask` has no active (`true`)
  /// frame — mirrors dia's `Error::EmptyOrInactiveMask` (`model.rs:
  /// 646-649`) exactly, checked BEFORE any padding or inference.
  /// [`InferError::NonFiniteInput`], [`InferError::Tensor`],
  /// [`InferError::Prediction`], [`InferError::OutputShape`] as
  /// [`Self::embed_chunk`]. [`InferError::NonFiniteOutput`] if any of the
  /// returned `EMBEDDING_DIM` values is NaN/infinite — scanning only
  /// slot 0's row, not the full batched output (module doc, "NonFinite-
  /// output scan scope"), matching dia's own function: its backend call is
  /// `n = 1` (`model.rs:301`), so it only ever has one row to check
  /// (`model.rs:663-665`).
  pub fn embed_chunk_with_frame_mask(
    &self,
    samples: &[f32],
    frame_mask: &[bool],
  ) -> Result<[f32; EMBEDDING_DIM], InferError> {
    check_mask_active(frame_mask)?;
    let flat = self.run_batched(samples, &[frame_mask, &[], &[]])?;
    let row0 = &flat[..EMBEDDING_DIM];
    check_finite_output(row0)?;
    let mut out = [0.0f32; EMBEDDING_DIM];
    out.copy_from_slice(row0);
    Ok(out)
  }

  /// Shared batched-inference core for [`Self::embed_chunk`] and
  /// [`Self::embed_chunk_with_frame_mask`]: builds the padded waveform/mask
  /// tensors, predicts, validates the output shape, and extracts the flat
  /// `[EMBED_SLOTS * EMBEDDING_DIM]` row-major buffer — WITHOUT scanning
  /// it for non-finite values, because the two callers need different
  /// scan scopes (module doc, "NonFinite-output scan scope").
  fn run_batched(
    &self,
    samples: &[f32],
    masks: &[&[bool]; EMBED_SLOTS],
  ) -> Result<[f32; EMBED_SLOTS * EMBEDDING_DIM], InferError> {
    check_finite_input(samples)?;

    let waveform_flat = build_waveform(samples);
    let mask_flat = build_masks(masks, self.num_mask_frames);

    let waveform = MultiArray::from_slice(
      &[
        EMBED_SLOTS,
        crate::audio::speaker::segment::SEG_CHUNK_SAMPLES,
      ],
      &waveform_flat,
    )?;
    let mask = MultiArray::from_slice(&[EMBED_SLOTS, self.num_mask_frames], &mask_flat)?;

    let mut outputs = self
      .model
      .predict_with(&[(names::WAVEFORM, &waveform), (names::MASK, &mask)])?;
    let embedding =
      outputs
        .take(names::EMBEDDING)
        .ok_or_else(|| crate::PredictionError::MissingOutput {
          name: names::EMBEDDING.to_string(),
        })?;
    // Construction validated the DECLARED contract; the CoreML runtime
    // producing this specific prediction's tensor is a separate trust
    // boundary, re-checked on every call — same rationale as
    // `crate::audio::speaker::segment::SegmentModel::infer`'s `check_output_shape` (see
    // that module's doc, "Layout re-validation").
    check_output_shape(embedding.shape())?;

    let mut flat = [0.0f32; EMBED_SLOTS * EMBEDDING_DIM];
    embedding.copy_into::<f32>(&mut flat)?;
    Ok(flat)
  }
}

/// Repeat-pads (or truncates) `source` to exactly `target_len` elements by
/// periodic tiling: `out[i] = source[i % source.len()]`.
///
/// Empirically equivalent to FluidAudio's Swift doubling-copy loop (module
/// doc, "FluidAudio's repeat-padding scheme") for the pad case
/// (`0 < source.len() < target_len`): writing `n` for the filled length at
/// each step of the Swift loop's recurrence `n' = n + min(n, target_len -
/// n)`, `n` starts at `source.len()` and DOUBLES at every step
/// (`n' = 2n`) until the LAST, possibly-partial step. By induction, `n`
/// stays a multiple of `source.len()` at every step BEFORE that last one,
/// so the buffer already satisfies `buf[j] == source[j % source.len()]`
/// for `j < n` at the start of each iteration, and copying `buf[0..c]` to
/// `buf[n..n+c]` (for `c <= n`) preserves it: `buf[n+j] = buf[j] =
/// source[j % source.len()] = source[(n+j) % source.len()]` (the last
/// equality needs `n ≡ 0 (mod source.len())`, which the induction
/// establishes). This holds through the final partial step too, since it
/// only depends on `n` being a multiple of `source.len()` going INTO that
/// step, not coming out of it. Cross-checked empirically (not just by this
/// proof) in [`tests::doubling_copy_simulation`], a literal Rust
/// transliteration of the Swift loop, for several non-power-of-2 lengths.
///
/// Two cases beyond FluidAudio's own documented contract, both this
/// crate's own choice (not read off FluidAudio — see the module doc for
/// why FluidAudio itself has no single clean contract for either):
/// - `source.is_empty()`: returns `target_len` zeros — the buffer is
///   simply left as its (zero-cleared) allocation, so this crate
///   synthesizes that same result directly rather than replicating an
///   infinite-loop guard that has nothing left to do.
/// - `source.len() >= target_len`: truncates to the first `target_len`
///   elements (the `i % source.len() == i` case of the same formula,
///   since `i < target_len <= source.len()`).
fn repeat_pad_f32(source: &[f32], target_len: usize) -> Vec<f32> {
  if source.is_empty() {
    return vec![0.0; target_len];
  }
  (0..target_len).map(|i| source[i % source.len()]).collect()
}

/// Converts a per-frame boolean activity mask to WeSpeaker's expected
/// 0.0/1.0 `f32` pooling weights — the identical conversion dia's ORT and
/// tch embed backends both perform (`diarization/src/embed/model.rs:
/// 296-299` and `:374-377`: `|&b| if b { 1.0 } else { 0.0 }`), because the
/// model's declared `mask` input is `f32`, never boolean
/// (`tests/model_io.rs::wespeaker_v2_io_matches_spec`).
fn mask_row_f32(mask: &[bool]) -> Vec<f32> {
  mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}

/// Builds the `[EMBED_SLOTS, SEG_CHUNK_SAMPLES]` waveform tensor's flat
/// row-major backing buffer: `samples`, repeat-padded to
/// `SEG_CHUNK_SAMPLES` (see [`repeat_pad_f32`]), identically in EVERY
/// slot. See the module doc's "Batching design" section for why every row
/// is identical here (in contrast to FluidAudio, which only ever fills
/// one real row per call).
fn build_waveform(samples: &[f32]) -> Vec<f32> {
  let row = repeat_pad_f32(samples, crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  let mut out = Vec::with_capacity(EMBED_SLOTS * crate::audio::speaker::segment::SEG_CHUNK_SAMPLES);
  for _ in 0..EMBED_SLOTS {
    out.extend_from_slice(&row);
  }
  out
}

/// Builds the `[EMBED_SLOTS, num_mask_frames]` mask tensor's flat
/// row-major backing buffer: each slot's `masks[i]` independently
/// converted to `f32` ([`mask_row_f32`]) and repeat-padded to
/// `num_mask_frames` ([`repeat_pad_f32`]). An empty `masks[i]` (as
/// [`EmbedModel::embed_chunk_with_frame_mask`] passes for its two unused
/// slots) repeat-pads to all-zero, matching FluidAudio's own zero-masked
/// unused rows (module doc, "Batching design").
fn build_masks(masks: &[&[bool]; EMBED_SLOTS], num_mask_frames: usize) -> Vec<f32> {
  let mut out = Vec::with_capacity(EMBED_SLOTS * num_mask_frames);
  for &mask in masks {
    out.extend(repeat_pad_f32(&mask_row_f32(mask), num_mask_frames));
  }
  out
}

/// Validates that a per-frame mask has at least one active (`true`) entry
/// — hermetically testable without a loaded model. Mirrors dia's
/// `embed_chunk_with_frame_mask` exactly: `!frame_mask.iter().any(|&b| b)`
/// (`diarization/src/embed/model.rs:647`) — see the module doc's "dia
/// contract match" section.
fn check_mask_active(mask: &[bool]) -> Result<(), InferError> {
  if !mask.iter().any(|&b| b) {
    return Err(InferError::EmptyMask);
  }
  Ok(())
}

/// Scans `samples` for the first non-finite value, BEFORE any padding or
/// inference — hermetically testable without a loaded model. A NaN sample
/// would otherwise repeat-pad and propagate into a finite-looking but
/// garbage embedding no output-side check would catch (review-queue
/// rationale: this crate's `InferError::NonFiniteInput` review-queue
/// item).
fn check_finite_input(samples: &[f32]) -> Result<(), InferError> {
  if let Some(index) = samples.iter().position(|v| !v.is_finite()) {
    return Err(InferError::NonFiniteInput { index });
  }
  Ok(())
}

/// Scans `values` for the first non-finite value — "the exact `ort`
/// CoreML-EP corruption mode this crate exists to replace" for the
/// embedding stage (spec §6 gate 2), mirroring
/// [`crate::audio::speaker::segment`]'s identical-shaped `check_finite` but over embed's
/// own output buffers (either the full `EMBED_SLOTS * EMBEDDING_DIM` batch
/// or a single `EMBEDDING_DIM` row — see the module doc's "NonFinite-
/// output scan scope"). Extracted so it is hermetically testable without a
/// loaded model.
fn check_finite_output(values: &[f32]) -> Result<(), InferError> {
  if let Some(index) = values.iter().position(|v| !v.is_finite()) {
    return Err(InferError::NonFiniteOutput { index });
  }
  Ok(())
}

/// Validates a predict-time `embedding` tensor's shape against the fixed
/// `[EMBED_SLOTS, EMBEDDING_DIM]` contract — hermetically testable without
/// a loaded model. Same structure as
/// [`crate::audio::speaker::segment`]'s `check_output_shape` (commit `fcbce74`'s
/// precedent: a per-call, every-profile check, not a `debug_assert`),
/// catching what [`crate::MultiArray::copy_into`] cannot: it validates
/// only total element count, so an axes-swapped `[EMBEDDING_DIM,
/// EMBED_SLOTS]` tensor (identical element count) would otherwise pass
/// silently and transpose slots and dimensions.
fn check_output_shape(shape: &[usize]) -> Result<(), InferError> {
  if shape != [EMBED_SLOTS, EMBEDDING_DIM] {
    return Err(InferError::OutputShape {
      got: shape.to_vec(),
      expected: vec![EMBED_SLOTS, EMBEDDING_DIM],
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests;
