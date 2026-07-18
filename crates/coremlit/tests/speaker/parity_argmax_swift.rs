//! Tier-1 **execution fidelity** for [`ArgmaxSource`] (design spec §5, and
//! §5.1's pinned surface): our [`Extraction`]'s contents versus **argmax's
//! own Swift** reading of the **same model outputs** on the **same input**.
//!
//! It answers exactly one question — *did we interpret argmax's decoded
//! tensors correctly?* — and nothing else. It is deliberately **not** an
//! end-to-end RTTM/diarization comparison: everything downstream of the
//! embeddings is `dia`'s clustering (spec §4, §7), so an end-to-end gate
//! would be measuring `dia`'s clusterer against argmax's, which is not what
//! this source is being validated for. In particular argmax's
//! `minActiveRatio` cluster-formation filter is **not** ported (spec §5.2, a
//! recorded decision); it lives *after* the embeddings, so it is out of this
//! gate's surface. This suite REPORTS how often it would have fired and gates
//! nothing on it.
//!
//! # Why a bespoke Swift oracle
//!
//! argmax ships a `DiarizeCLI`, but it emits only post-clustering **RTTM** —
//! no flag, anywhere in it, exposes an intermediate tensor. So the oracle is
//! `tests/swift/`: an out-of-tree SwiftPM package that `@testable import`s
//! `SpeakerKit` and drives argmax's own `SpeakerSegmenterModel.predict` +
//! `SpeakerEmbedderModel.embed`, dumping the `[SpeakerEmbedding]` they
//! produce. That array IS what [`Extraction`] carries, per `(chunk, slot)`:
//! `windowIndex` → chunk `c`, `speakerIndex` → slot `s`, `activeFrames` →
//! `segmentations[c][..][s]`, `embedding` → `raw_embeddings[c][s][..]`.
//!
//! Regenerate with `tests/swift/regen_goldens.sh` (documented there); the
//! goldens live in `tests/speaker/fixtures/golden_argmax_swift/`.
//!
//! # The inputs are PROVEN identical, not assumed (the alignkit lesson)
//!
//! A padding/length mismatch once produced a fake 86% divergence in alignkit.
//! So before any tensor is compared:
//!
//! - the golden records the FNV-1a-64 of the exact `[f32]` array handed to
//!   `SpeakerSegmenterModel.predict`, and this suite recomputes it over the
//!   array it hands to [`ArgmaxSource::extract`] — a mismatch fails as a
//!   HARNESS bug before a single fidelity number is read;
//! - compute placement is pinned on both sides. argmax's fbank preprocessor
//!   HARDCODES `.cpuOnly` (`SpeakerPreEmbedderModel.swift:14`), so the golden
//!   is generated with all three models on `cpu_only` and this suite asserts
//!   its own [`ArgmaxComputeOptions`] matches what the golden was generated
//!   with. Comparing a `.all`-placed Rust preprocessor against a `.cpuOnly`
//!   Swift one would measure CoreML's scheduler and blame the port.
//!
//! (As a free bonus the goldens record that WhisperKit's own AVFoundation
//! loader — the one `DiarizeCLI` uses — decodes these WAVs to the *identical*
//! samples `common::load_wav_16k_mono` does, so there is no audio-front-end
//! confound either.)
//!
//! # What is measured, and what "expected" means for each
//!
//! - **Segmentations: EXACT.** Both sides read the same hard `speaker_ids`
//!   tensor; any mismatch is a real indexing bug, so the bound is zero
//!   mismatching cells — not a tolerance.
//! - **The consumed `(chunk, slot)` SET: EXACT.** Equality in both
//!   directions, which is what makes the activity gate, the `bounded()`
//!   filter, the PLDA-norm guard and [`Extraction`]'s zero-column invariant
//!   checkable rather than assumed.
//! - **The chunk GRID: EXACT.** argmax's own `bounded(windowIdx:)` verdict and
//!   its own `windowIndex` arithmetic (`k*21 + w`) are dumped per chunk; this
//!   suite asserts their union IS `0..num_chunks` of `dia`'s grid. The port's
//!   central grid theorem, checked against argmax's arithmetic rather than
//!   restated in ours. `ted_60` is in the set purely to reach `k >= 1` — the
//!   two speakerkit fixtures are both under 30 s, i.e. a single argmax chunk.
//! - **Embeddings: near-exact,** bounded EMPIRICALLY ([`EMBED_MAX_ABS_TOL`],
//!   [`EMBED_COS_TOL`]) — same three `.mlmodelc`s, same placement, same
//!   inputs, and (below) provably the same masks, so the only slack is host
//!   float→f16 conversion versus CoreML's own.
//!
//! # The two KNOWN divergences from argmax's Swift, and why neither exempts a slot
//!
//! 1. **`dia`'s exclude-overlap fallback** (`clean_count <= 2` ⇒ drop the
//!    overlap exclusion for that slot) has no counterpart in argmax's mask
//!    (`SpeakerEmbedderModel.swift:245` excludes unconditionally). It would
//!    make our mask differ — so this suite proves it never fires here:
//!    [`assert_fallback_inert`] reconstructs each slot's clean-frame count
//!    from the golden's own `nonOverlappedFrameRatio` and asserts it exceeds
//!    2. Measured minima: 75 (`02`), 45 (`07`), 8 (`ted_60`) — never close.
//!    Masks are therefore IDENTICAL to argmax's on every consumed slot, and
//!    no slot needs exempting. Were it ever to fire, that slot's embedding is
//!    legitimately ours-not-theirs and would have to be exempted explicitly.
//! 2. **Mask row 63.** argmax zero-fills only rows `0..<63` and leaves the
//!    64th uninitialized (`SpeakerEmbedderModel.swift:219-224`); we zero all
//!    64. The golden records that a FRESH, separately-allocated `[1, 64,
//!    1767]` `MLMultiArray` does **not** come back zeroed
//!    (`freshMaskAllocAllZero: false`) — but that probe inspects a
//!    DIFFERENT allocation than the one `processChunk` itself used on the
//!    runs below, so on its own it is strong measured evidence that such
//!    allocations aren't zero-initialized *in general* on this platform, not
//!    a measurement that row 63 was nonzero *on this run*. The stronger leg
//!    is `determinismVerified: true`: two independent full-pipeline runs
//!    (separate allocations throughout) produced bit-identical embeddings,
//!    which bears directly on the buffers the compared run actually used.
//!    Together they are strong measured evidence — short of a formal proof —
//!    that argmax's row 63 really does differ from ours here too. That makes
//!    the embedding comparison below a stronger check than the
//!    row-independence probe it replaces: if the 41/24/58 consumed rows
//!    match despite a probably-differing row 63, the embedder's 64 slots are
//!    independent as a matter of strong measured evidence.
//!
//! `#[ignore]` (needs the gitignored `Models/argmax-speakerkit/` artifacts and
//! the committed goldens); run via
//! `ARGMAX_TEST_MODELS=… cargo test -p coremlit -- --ignored`.

mod common;

use std::{collections::BTreeSet, path::PathBuf};

use coremlit::{
  ComputeUnits,
  audio::speaker::{
    embed::EMBEDDING_DIM,
    extract::{EXCLUDE_OVERLAP_MIN_FRAMES, Extraction},
    segment::SEG_NUM_SLOTS,
    source::{
      ArgmaxComputeOptions, ArgmaxOptions, ArgmaxSource, ArgmaxVariant, ModelSource,
      argmax::{ARGMAX_FRAMES_PER_WINDOW, ARGMAX_WINDOWS_PER_CHUNK},
    },
  },
};

/// Max per-element absolute difference tolerated between our raw embedding
/// and argmax's Swift, per consumed `(chunk, slot)`.
///
/// **Measured, not chosen:** with placement matched and masks provably
/// identical (module doc), the observed worst case across all three fixtures
/// is `0.0` — every one of the 123 consumed rows is BIT-IDENTICAL. The bound
/// is kept as a bound rather than `assert_eq!(diff, 0.0)` only because the
/// two sides convert `f32 -> f16` in different places (we call
/// `f16::from_f32` on the host; argmax hands CoreML a `.float32`
/// `MLMultiArray` for the segmenter's `.float16` `waveform` input and lets it
/// convert), and both being IEEE round-to-nearest-even is a fact about the
/// implementations rather than a contract. A value materially above zero is a
/// FINDING (our mask or index mapping), never a tolerance to raise.
const EMBED_MAX_ABS_TOL: f64 = 1e-6;

/// Min cosine between our raw embedding and argmax's, per consumed
/// `(chunk, slot)`. Measured worst case: exactly `1.0` (see
/// [`EMBED_MAX_ABS_TOL`]).
const EMBED_COS_TOL: f64 = 0.999_999;

/// Max fraction of hard `speaker_ids` cells the DEFAULT [`ComputeUnits::All`]
/// placement may flip relative to the `cpu_only` reference.
///
/// **Measured: `66 / 72_447` = 0.0911%.** Bounded at ~2x that, because unlike
/// the fidelity tolerances above this is not a quantity that *should* be zero
/// — it is CoreML electing a different (fp16, differently-accumulating)
/// backend for the same graph, and it can legitimately shift a little with
/// macOS/hardware. The bound is a tripwire for a MATERIAL scheduling
/// regression, not a pin on the exact rate. See
/// [`argmax_default_placement_vs_the_cpu_only_reference`], which is
/// deliberately NOT the fidelity gate.
const PLACEMENT_SEG_MISMATCH_TOL: f64 = 0.002;

/// Min per-slot embedding cosine between the DEFAULT [`ComputeUnits::All`]
/// placement and the `cpu_only` reference. **Measured worst: 0.9241**
/// (`ted_60`, chunk 18, slot 2); bounded with headroom, per
/// [`PLACEMENT_SEG_MISMATCH_TOL`]'s rationale.
const PLACEMENT_COS_TOL: f64 = 0.90;

/// The fixtures this gate replays, and where their audio lives.
///
/// The first two are speakerkit's own committed parity clips
/// (`common::FIXTURES`). `ted_60` is **whisperkit's**, borrowed rather than
/// copied (1.9 MB): both speakerkit fixtures are under 30 s and so produce a
/// SINGLE argmax chunk, leaving `c = k * 21 + w`'s `k >= 1` branch — the port's
/// central index claim — untested by either. `ted_60` is 60.0 s: three chunks,
/// windows abutting at `c = 21` and `c = 42`.
fn fixture_audio(name: &str) -> PathBuf {
  match name {
    "ted_60" => {
      PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/whisper/fixtures/audio/ted_60.wav")
    }
    other => common::audio_path(other),
  }
}

/// The fixture names, in golden order.
const GATE_FIXTURES: &[&str] = &["02_pyannote_sample", "07_yuhewei_dongbei_english", "ted_60"];

// ─────────────────────────────────────────────────────────────────────────
// The golden (written by tests/swift/Tests/ArgmaxTensorDump)
// ─────────────────────────────────────────────────────────────────────────

/// One consumed `(chunk, slot)` as argmax's own Swift produced it.
struct SwiftSlot {
  /// `SpeakerEmbedding.windowIndex` — argmax's own `k * 21 + w`, which IS the
  /// [`Extraction`] chunk index.
  chunk: usize,
  /// `SpeakerEmbedding.speakerIndex`.
  slot: usize,
  /// `SpeakerEmbedding.activeFrames`: the window's 589 `speaker_ids` values
  /// for this slot, hard-binary.
  active_frames: Vec<bool>,
  /// `SpeakerEmbedding.nonOverlappedFrameRatio` — `clean_frames / 589`. The
  /// input to argmax's NOT-ported `minActiveRatio` filter (spec §5.2), used
  /// here only to (a) prove `dia`'s fallback stays inert and (b) report how
  /// often argmax would have withheld the slot from cluster formation.
  non_overlapped_frame_ratio: f64,
  /// `SpeakerEmbedding.embedding` — the raw 256-d WeSpeaker vector.
  embedding: Vec<f32>,
}

/// argmax's own verdict on one 30 s chunk's window grid.
struct SwiftChunk {
  /// `k`.
  chunk_index: usize,
  /// The chunk's UNPADDED sample count (argmax's `waveformLength`).
  unpadded_samples: usize,
  /// `chunkOffset(k) + round(w * secondsPerStride)` over the windows argmax's
  /// own `bounded(windowIdx:)` admits — i.e. argmax's own answer for which
  /// [`Extraction`] chunks this 30 s chunk contributes.
  global_chunks: Vec<usize>,
}

/// A committed argmax-Swift reference for one fixture.
struct SwiftGolden {
  input_samples: usize,
  input_fnv1a: String,
  compute_units: Vec<(String, String)>,
  windows_count: usize,
  frames_per_window: usize,
  speakers_count: usize,
  embedding_dim: usize,
  fresh_mask_alloc_all_zero: bool,
  chunks: Vec<SwiftChunk>,
  slots: Vec<SwiftSlot>,
}

/// Decodes one slot's `activeFrames` bit string, hard-failing unless it is
/// EXACTLY [`ARGMAX_FRAMES_PER_WINDOW`] characters and every character is
/// `'0'` or `'1'` (`activeFrames` is `speaker_ids`, which is hard-binary — a
/// non-binary or wrong-length value is a malformed golden, never a slot with
/// fewer active frames).
///
/// Delegates to the SINGLE strict decoder [`common::parse_bit_mask`] shared with
/// the embedding golden's mask loader — one leniency killer, not two. The old
/// lenient decode (`c == '1'`, any length, every non-`'1'` char → `false`) is
/// what let this gate pass vacuously: an empty or truncated `activeFrames`
/// decoded to a short/empty `Vec<bool>`, the comparison below iterated only that
/// length, yet [`Fidelity::seg_cells`] still REPORTED `slots × 589` cells
/// "compared".
fn parse_active_frames(fixture: &str, chunk: usize, slot: usize, raw: &str) -> Vec<bool> {
  common::parse_bit_mask(
    &format!("{fixture}: chunk {chunk} slot {slot}: activeFrames"),
    ARGMAX_FRAMES_PER_WINDOW,
    raw,
  )
}

/// Loads and parses a committed argmax-Swift golden.
///
/// # Panics
/// If it is missing or malformed — the oracle is a hard dependency of this
/// gate, never an optional input.
fn load_swift_golden(name: &str) -> SwiftGolden {
  let path = common::fixtures_dir()
    .join("golden_argmax_swift")
    .join(format!("{name}.json"));
  let bytes = std::fs::read(&path).unwrap_or_else(|e| {
    panic!(
      "read argmax-swift golden {}: {e}\n  regenerate: crates/speakerkit/tests/swift/regen_goldens.sh",
      path.display()
    )
  });
  let v: serde_json::Value =
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse golden {name}: {e}"));

  let usize_at = |key: &str| v[key].as_u64().unwrap_or_else(|| panic!("{name}: {key}")) as usize;
  let compute_units = v["computeUnits"]
    .as_object()
    .expect("computeUnits object")
    .iter()
    .map(|(k, v)| {
      (
        k.clone(),
        v.as_str().expect("compute unit string").to_string(),
      )
    })
    .collect();

  let chunks = v["chunks"]
    .as_array()
    .expect("chunks array")
    .iter()
    .map(|c| SwiftChunk {
      chunk_index: c["chunkIndex"].as_u64().expect("chunkIndex") as usize,
      unpadded_samples: c["unpaddedSamples"].as_u64().expect("unpaddedSamples") as usize,
      global_chunks: c["globalChunks"]
        .as_array()
        .expect("globalChunks")
        .iter()
        .map(|x| x.as_u64().expect("global chunk") as usize)
        .collect(),
    })
    .collect();

  let slots = v["slots"]
    .as_array()
    .expect("slots array")
    .iter()
    .map(|s| {
      let chunk = s["chunk"].as_u64().expect("chunk") as usize;
      let slot = s["slot"].as_u64().expect("slot") as usize;
      SwiftSlot {
        chunk,
        slot,
        active_frames: parse_active_frames(
          name,
          chunk,
          slot,
          s["activeFrames"].as_str().expect("activeFrames string"),
        ),
        non_overlapped_frame_ratio: s["nonOverlappedFrameRatio"].as_f64().expect("ratio"),
        embedding: s["embedding"]
          .as_array()
          .expect("embedding array")
          .iter()
          .map(|x| x.as_f64().expect("embed f64") as f32)
          .collect(),
      }
    })
    .collect();

  SwiftGolden {
    input_samples: usize_at("inputSamples"),
    input_fnv1a: v["inputFnv1a"].as_str().expect("inputFnv1a").to_string(),
    compute_units,
    windows_count: usize_at("windowsCount"),
    frames_per_window: usize_at("framesPerWindow"),
    speakers_count: usize_at("speakersCount"),
    embedding_dim: usize_at("embeddingDim"),
    fresh_mask_alloc_all_zero: v["freshMaskAllocAllZero"]
      .as_bool()
      .expect("freshMaskAllocAllZero"),
    chunks,
    slots,
  }
}

// ─────────────────────────────────────────────────────────────────────────
// The comparison
// ─────────────────────────────────────────────────────────────────────────

/// Every `(chunk, slot)` our [`Extraction`] actually carries: a nonzero
/// segmentation column OR a nonzero embedding row.
///
/// Both halves are checked, not just one, precisely because [`Extraction`]'s
/// contract couples them (a dropped slot has BOTH zeroed): reading only the
/// embeddings would miss an over-emitted segmentation column, and vice versa.
fn consumed_slots(extraction: &Extraction) -> BTreeSet<(usize, usize)> {
  let segmentations = extraction.segmentations();
  let embeddings = extraction.raw_embeddings();
  let frames = extraction.num_frames_per_chunk();
  let mut consumed = BTreeSet::new();
  for c in 0..extraction.num_chunks() {
    for s in 0..SEG_NUM_SLOTS {
      let column_active =
        (0..frames).any(|f| segmentations[(c * frames + f) * SEG_NUM_SLOTS + s] != 0.0);
      let base = (c * SEG_NUM_SLOTS + s) * EMBEDDING_DIM;
      let row_active = embeddings[base..base + EMBEDDING_DIM]
        .iter()
        .any(|&v| v != 0.0);
      if column_active || row_active {
        consumed.insert((c, s));
      }
    }
  }
  consumed
}

/// Proves `dia`'s exclude-overlap fallback — the one rule in our mask that
/// argmax's mask does not have (module doc's divergence 1) — cannot have
/// fired on any consumed slot of this fixture, so both sides pooled over the
/// SAME mask and the embedding comparison below is a fidelity measurement
/// rather than a comparison of two different maskings.
///
/// The clean-frame count is reconstructed from argmax's own
/// `nonOverlappedFrameRatio` (`clean_frames / framesPerWindowCount`,
/// `SpeakerEmbedderModel.swift:105-118`), so this is argmax's count, not a
/// recomputation of it.
///
/// # Panics
/// If any consumed slot's clean-frame count is `<= EXCLUDE_OVERLAP_MIN_FRAMES`
/// — that slot's mask WOULD legitimately differ, and it must then be exempted
/// from the embedding assertion explicitly and reported, never folded into a
/// loosened tolerance.
fn assert_fallback_inert(fixture: &str, golden: &SwiftGolden) -> usize {
  let mut min_clean = usize::MAX;
  for slot in &golden.slots {
    let clean =
      (slot.non_overlapped_frame_ratio * golden.frames_per_window as f64).round() as usize;
    min_clean = min_clean.min(clean);
    assert!(
      clean > EXCLUDE_OVERLAP_MIN_FRAMES,
      "{fixture}: chunk {} slot {}: clean_count {clean} <= {EXCLUDE_OVERLAP_MIN_FRAMES}, so dia's \
       exclude-overlap fallback FIRES here and our mask is not argmax's. Exempt this slot from the \
       embedding comparison explicitly and report it — do not loosen the bound.",
      slot.chunk,
      slot.slot
    );
  }
  min_clean
}

/// One fixture's measured agreement with argmax's Swift.
///
/// The HARNESS invariants (input match, placement match, model contract, the
/// chunk grid, fallback inertness) are asserted where they are measured —
/// they are preconditions, and a violated one makes every number below
/// meaningless. The FIDELITY numbers are returned, not asserted, so each
/// caller can hold them to its own bound: exactness for the matched-placement
/// gate, an empirical bound for the placement study.
#[derive(Debug)]
struct Fidelity {
  /// Segmentation cells compared (`consumed slots × 589`).
  seg_cells: usize,
  /// Cells where our `segmentations` disagrees with argmax's `speaker_ids`.
  seg_mismatches: usize,
  /// Consumed `(chunk, slot)`s we carry and argmax does not.
  only_ours: Vec<(usize, usize)>,
  /// Consumed `(chunk, slot)`s argmax carries and we do not.
  only_theirs: Vec<(usize, usize)>,
  /// Worst per-element absolute embedding difference over the slots BOTH
  /// sides carry.
  worst_abs: f64,
  /// Worst per-slot embedding cosine over the slots BOTH sides carry.
  worst_cos: f64,
  /// Embedding rows that are BIT-identical, of [`Self::compared_rows`].
  exact_rows: usize,
  /// Rows compared (`|ours ∩ theirs|`).
  compared_rows: usize,
  /// `golden.slots.len()` — the raw count of `(chunk, slot)` entries argmax's
  /// Swift emitted for this fixture, read straight off the golden and
  /// independent of [`Self::only_ours`]/[`Self::only_theirs`]/
  /// [`Self::compared_rows`]'s own bookkeeping. A caller that also proves the
  /// slot SET matches (`only_ours`/`only_theirs` both empty, or their summed
  /// length zero) can assert this against [`Self::compared_rows`] to catch
  /// the embedding loop's `ours.contains(..)` filter silently shrinking
  /// coverage instead of failing — a bug that check would share with neither.
  golden_slots: usize,
}

/// Replays one fixture through [`ArgmaxSource`] at `compute` and measures it
/// against the committed argmax-Swift golden.
fn measure(
  fixture: &str,
  compute: ArgmaxComputeOptions,
  expect_golden_placement: bool,
) -> Fidelity {
  let golden = load_swift_golden(fixture);

  // ── Input-match proof, BEFORE any tensor is read (the alignkit lesson) ──
  let samples = common::load_wav_16k_mono(&fixture_audio(fixture));
  assert_eq!(
    samples.len(),
    golden.input_samples,
    "{fixture}: sample COUNT differs from the golden's — the two sides are not being fed the same \
     audio; fix the harness before reading any fidelity number"
  );
  assert_eq!(
    common::fnv_hex(common::fnv1a_f32(&samples)),
    golden.input_fnv1a,
    "{fixture}: sample BYTES differ from the golden's (same count, different values) — the two \
     sides are not being fed the same audio"
  );

  // ── Placement-match proof: argmax's preprocessor is hardcoded .cpuOnly ──
  if expect_golden_placement {
    let ours = [
      ("segmenter", compute.segmenter()),
      ("preprocessor", compute.preprocessor()),
      ("embedder", compute.embedder()),
    ];
    for (name, units) in ours {
      let theirs = golden
        .compute_units
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("{fixture}: golden has no compute unit for {name}"));
      assert_eq!(
        units.as_str(),
        theirs,
        "{fixture}: the golden was generated with {name} on {theirs}, but this run places it on \
         {}. A placement mismatch measures CoreML's scheduler, not our tensor reading.",
        units.as_str()
      );
    }
  }

  // ── The pinned model contract the golden was dumped through ──
  assert_eq!(golden.windows_count, ARGMAX_WINDOWS_PER_CHUNK);
  assert_eq!(golden.frames_per_window, ARGMAX_FRAMES_PER_WINDOW);
  assert_eq!(golden.speakers_count, SEG_NUM_SLOTS);
  assert_eq!(golden.embedding_dim, EMBEDDING_DIM);

  let min_clean = assert_fallback_inert(fixture, &golden);

  let source = ArgmaxSource::from_dir_with(
    common::argmax_models_dir(),
    ArgmaxOptions::new()
      .with_variant(ArgmaxVariant::Baseline)
      .with_compute(compute),
  )
  .expect("load argmax models");
  let extraction = source.extract(&samples).expect("argmax extract");

  // ── The grid theorem, against argmax's OWN bounded()/windowIndex ──
  let swift_grid: Vec<usize> = {
    let mut all: Vec<usize> = golden
      .chunks
      .iter()
      .flat_map(|c| c.global_chunks.iter().copied())
      .collect();
    all.sort_unstable();
    all.dedup();
    all
  };
  assert_eq!(
    swift_grid,
    (0..extraction.num_chunks()).collect::<Vec<_>>(),
    "{fixture}: the chunks argmax's own bounded()/windowIndex admit are NOT dia's chunk grid — \
     the `c = k * {ARGMAX_WINDOWS_PER_CHUNK} + w` theorem is violated"
  );
  assert_eq!(extraction.num_frames_per_chunk(), golden.frames_per_window);

  // Sanity: no chunk may claim more than the 30 s the segmenter consumes.
  for chunk in &golden.chunks {
    assert!(
      chunk.unpadded_samples <= 480_000,
      "{fixture}: chunk {} claims {} unpadded samples",
      chunk.chunk_index,
      chunk.unpadded_samples
    );
  }

  // ── The consumed (chunk, slot) set, both directions ──
  let ours = consumed_slots(&extraction);
  let theirs: BTreeSet<(usize, usize)> = golden.slots.iter().map(|s| (s.chunk, s.slot)).collect();
  let only_ours: Vec<(usize, usize)> = ours.difference(&theirs).copied().collect();
  let only_theirs: Vec<(usize, usize)> = theirs.difference(&ours).copied().collect();

  // ── Segmentations: cell-for-cell against argmax's hard speaker_ids ──
  let segmentations = extraction.segmentations();
  let frames = extraction.num_frames_per_chunk();
  let mut seg_mismatches = 0usize;
  let mut compared_seg_cells = 0usize;
  let mut first_seg_mismatch = None;
  for slot in &golden.slots {
    for (f, &active) in slot.active_frames.iter().enumerate() {
      let ours = segmentations[(slot.chunk * frames + f) * SEG_NUM_SLOTS + slot.slot];
      let theirs = f64::from(u8::from(active));
      if ours != theirs {
        seg_mismatches += 1;
        first_seg_mismatch.get_or_insert((slot.chunk, slot.slot, f, ours, theirs));
      }
      compared_seg_cells += 1;
    }
  }
  // Coverage guard — the other half of the vacuity fix, paired with the strict
  // loader: the number of cells actually compared MUST equal `slots × frames`.
  // The loader already guarantees every slot carries exactly `frames` bits, but
  // asserting the realized count here — rather than trusting the loop bound —
  // is what turns `seg_cells`'s reported coverage into a checked fact instead
  // of an unverified claim, and it fails if the compared surface ever shrinks
  // below what the report advertises.
  assert_eq!(
    compared_seg_cells,
    golden.slots.len() * frames,
    "{fixture}: compared {compared_seg_cells} segmentation cells, but the golden has {} slots × \
     {frames} frames = {}. The comparison covered fewer cells than it reports — the vacuous pass \
     this gate exists to prevent.",
    golden.slots.len(),
    golden.slots.len() * frames
  );

  // ── Embeddings, over the slots BOTH sides carry ──
  let embeddings = extraction.raw_embeddings();
  let (mut worst_abs, mut worst_cos) = (0.0f64, 1.0f64);
  let (mut worst_abs_slot, mut worst_cos_slot) = ((0, 0), (0, 0));
  let (mut exact_rows, mut compared_rows) = (0usize, 0usize);
  for slot in golden
    .slots
    .iter()
    .filter(|s| ours.contains(&(s.chunk, s.slot)))
  {
    let base = (slot.chunk * SEG_NUM_SLOTS + slot.slot) * EMBEDDING_DIM;
    let row = &embeddings[base..base + EMBEDDING_DIM];
    let max_abs = common::max_abs_diff(row, &slot.embedding);
    let cos = common::cosine(row, &slot.embedding);
    compared_rows += 1;
    if max_abs == 0.0 {
      exact_rows += 1;
    }
    if max_abs > worst_abs {
      worst_abs = max_abs;
      worst_abs_slot = (slot.chunk, slot.slot);
    }
    if cos < worst_cos {
      worst_cos = cos;
      worst_cos_slot = (slot.chunk, slot.slot);
    }
  }

  let seg_cells = golden.slots.len() * frames;
  let would_withhold = golden
    .slots
    .iter()
    .filter(|s| s.non_overlapped_frame_ratio <= 0.2)
    .count();
  println!(
    "[{fixture}] samples={} chunks={} slots={} | seg: {}/{seg_cells} cells agree ({seg_mismatches} \
     mismatched, first {first_seg_mismatch:?}) | slot set: +{} / -{} | embed: max|diff|={worst_abs:.3e} \
     (chunk {}, slot {}) cos_min={worst_cos:.9} (chunk {}, slot {}) bit-identical {exact_rows}/{compared_rows} \
     rows | min clean_count={min_clean} (dia's fallback fires 0x) | argmax's minActiveRatio would \
     withhold {would_withhold}/{} slots from cluster FORMATION (spec §5.2: not ported, not gated) | \
     argmax's fresh mask alloc all-zero: {}",
    samples.len(),
    extraction.num_chunks(),
    golden.slots.len(),
    seg_cells - seg_mismatches,
    only_ours.len(),
    only_theirs.len(),
    worst_abs_slot.0,
    worst_abs_slot.1,
    worst_cos_slot.0,
    worst_cos_slot.1,
    golden.slots.len(),
    golden.fresh_mask_alloc_all_zero,
  );

  Fidelity {
    seg_cells,
    seg_mismatches,
    only_ours,
    only_theirs,
    worst_abs,
    worst_cos,
    exact_rows,
    compared_rows,
    golden_slots: golden.slots.len(),
  }
}

// ─────────────────────────────────────────────────────────────────────────
// The gates
// ─────────────────────────────────────────────────────────────────────────

/// Hermetic (no models): every committed golden loads through the STRICT
/// [`parse_active_frames`] loader. This is what exercises the `activeFrames`
/// validation — exactly [`ARGMAX_FRAMES_PER_WINDOW`] characters, each `'0'` or
/// `'1'` — on every `cargo test` run, and pins the committed goldens as
/// well-formed. An empty, truncated, or non-binary `activeFrames` in any of
/// them (the mutation that used to make the segmentation comparison vacuous)
/// fails HERE, without the model-gated fidelity run below. The model-gated
/// gates additionally assert the realized compared-cell count in [`measure`].
#[test]
fn committed_goldens_load_through_the_strict_loader() {
  for fixture in GATE_FIXTURES {
    let golden = load_swift_golden(fixture);
    assert_eq!(
      golden.frames_per_window, ARGMAX_FRAMES_PER_WINDOW,
      "{fixture}: golden declares framesPerWindow {}, expected {ARGMAX_FRAMES_PER_WINDOW}",
      golden.frames_per_window
    );
    assert!(
      !golden.slots.is_empty(),
      "{fixture}: golden has no consumed slots — nothing to compare"
    );
    for slot in &golden.slots {
      // `load_swift_golden` already rejects a non-589 activeFrames; re-assert
      // the realized length so this test would fail even if the loader's
      // length check were weakened.
      assert_eq!(
        slot.active_frames.len(),
        ARGMAX_FRAMES_PER_WINDOW,
        "{fixture}: chunk {} slot {}: activeFrames decoded to {} frames, expected \
         {ARGMAX_FRAMES_PER_WINDOW}",
        slot.chunk,
        slot.slot,
        slot.active_frames.len()
      );
    }
  }
}

/// **THE GATE.** `ArgmaxSource` versus argmax's own Swift — matched
/// placement, identical (hash-proven) input, provably identical masks — over
/// all three fixtures. See the module doc.
///
/// Every bound here is exactness, because every bound here CAN be: the two
/// sides run the same three `.mlmodelc`s on the same bytes.
#[test]
#[ignore = "needs Models/argmax-speakerkit (ARGMAX_TEST_MODELS)"]
fn argmax_execution_fidelity_vs_swift() {
  // The placement the goldens were dumped at. argmax's fbank preprocessor is
  // hardcoded `.cpuOnly` (`SpeakerPreEmbedderModel.swift:14`), so all three
  // follow it: matched, and reproducible on any machine (an ANE-scheduled
  // golden would not be — see the placement study below).
  let compute = ArgmaxComputeOptions::new()
    .with_segmenter(ComputeUnits::CpuOnly)
    .with_preprocessor(ComputeUnits::CpuOnly)
    .with_embedder(ComputeUnits::CpuOnly);

  let (mut worst_abs, mut worst_cos) = (0.0f64, 1.0f64);
  let (mut exact_rows, mut compared_rows, mut seg_cells) = (0usize, 0usize, 0usize);
  for fixture in GATE_FIXTURES {
    let f = measure(fixture, compute, true);

    assert!(
      f.only_ours.is_empty() && f.only_theirs.is_empty(),
      "{fixture}: the consumed (chunk, slot) set differs from argmax's.\n  only ours: {:?}\n  \
       only theirs: {:?}\nThe activity gate, the bounded() filter and Extraction's zero-column \
       invariant all land here — this is a mapping bug.",
      f.only_ours,
      f.only_theirs
    );
    assert_eq!(
      f.seg_mismatches, 0,
      "{fixture}: {} of {} segmentation cells disagree with argmax's speaker_ids. Both sides read \
       the SAME hard tensor, so this is an indexing bug, not a tolerance.",
      f.seg_mismatches, f.seg_cells
    );
    assert!(
      f.worst_abs <= EMBED_MAX_ABS_TOL,
      "{fixture}: embedding max|diff| {:.6e} exceeds {EMBED_MAX_ABS_TOL:e}. Placement is matched \
       and the masks are provably identical (dia's fallback fires 0x), so this is OUR mask or \
       index mapping — a FINDING, not a bound to raise.",
      f.worst_abs
    );
    assert!(
      f.worst_cos >= EMBED_COS_TOL,
      "{fixture}: embedding cosine {:.9} below {EMBED_COS_TOL}. See above — this is a FINDING.",
      f.worst_cos
    );

    worst_abs = worst_abs.max(f.worst_abs);
    worst_cos = worst_cos.min(f.worst_cos);
    exact_rows += f.exact_rows;
    compared_rows += f.compared_rows;
    seg_cells += f.seg_cells;
  }

  println!(
    "[gate] {seg_cells} segmentation cells EXACT; embeddings: {exact_rows}/{compared_rows} rows \
     BIT-identical, worst max|diff|={worst_abs:.3e}, worst cos={worst_cos:.9}"
  );
  // Not a redundant restatement of the per-fixture bounds: this pins the
  // measured result — bit-identity — which is strictly stronger than the
  // tolerances above, and which is the actual claim the report makes.
  assert_eq!(
    exact_rows,
    compared_rows,
    "every consumed embedding row was bit-identical when this gate was written ({compared_rows} of \
     them); {} now differ within tolerance. Investigate before relaxing this: same models, same \
     placement, same input and same mask should be bit-reproducible.",
    compared_rows - exact_rows
  );
}

/// The placement study: the source's DEFAULT [`ComputeUnits::All`] — what a
/// user actually gets — against the same `cpu_only` golden.
///
/// **Deliberately not part of the gate.** A divergence here is CoreML electing
/// a different backend for the same graph, i.e. a *scheduling* property; it is
/// not evidence about whether we read argmax's tensors correctly, and folding
/// it into the gate's bound is exactly how a fidelity tolerance gets silently
/// loosened. argmax's own Swift is subject to the identical effect (its
/// segmenter defaults to `.cpuOnly` but `PyannoteModelManager` places it
/// wherever its `ModelInfo` says), so this is a property of the MODEL on this
/// hardware, not of the port.
///
/// **What it measures (and it is not nothing):** ANE/GPU placement flips
/// **66 of 72 447** hard `speaker_ids` cells (**0.0911%**) and moves EVERY
/// embedding row — 0 of 123 stay bit-identical, worst max|diff| 0.248, worst
/// cosine **0.9241**. The consumed `(chunk, slot)` set is nevertheless
/// unchanged (+0/-0 on all three fixtures), so no slot appears or vanishes;
/// the decode just wobbles at frame boundaries the way fp16 arithmetic with a
/// different accumulation order does. For scale, T6 measured int8-vs-fp32
/// WeSpeaker at ~0.90-0.92 cosine — i.e. the shipping default's placement
/// noise is of the same order as a quantization tier, which is worth knowing
/// and is recorded in the task report as a concern.
///
/// It is asserted, at bounds settled from that measurement
/// ([`PLACEMENT_SEG_MISMATCH_TOL`], [`PLACEMENT_COS_TOL`]), so a future
/// macOS/CoreML change that makes scheduling *materially* worse fails here —
/// in its own test, with its own message — instead of silently degrading the
/// shipping default.
#[test]
#[ignore = "needs Models/argmax-speakerkit (ARGMAX_TEST_MODELS)"]
fn argmax_default_placement_vs_the_cpu_only_reference() {
  let compute = ArgmaxComputeOptions::new(); // ComputeUnits::All ×3

  let (mut worst_abs, mut worst_cos) = (0.0f64, 1.0f64);
  let (mut mismatches, mut cells, mut slot_diffs) = (0usize, 0usize, 0usize);
  let (mut compared_rows, mut golden_slots) = (0usize, 0usize);
  for fixture in GATE_FIXTURES {
    let f = measure(fixture, compute, false);
    worst_abs = worst_abs.max(f.worst_abs);
    worst_cos = worst_cos.min(f.worst_cos);
    mismatches += f.seg_mismatches;
    cells += f.seg_cells;
    slot_diffs += f.only_ours.len() + f.only_theirs.len();
    compared_rows += f.compared_rows;
    golden_slots += f.golden_slots;
  }
  let seg_rate = mismatches as f64 / cells as f64;
  println!(
    "[placement] ComputeUnits::All vs the cpu_only reference: {mismatches}/{cells} segmentation \
     cells differ ({:.4}%), {slot_diffs} slot-set differences, embedding worst \
     max|diff|={worst_abs:.3e} worst cos={worst_cos:.9}",
    seg_rate * 100.0
  );

  // The consumed (chunk, slot) SET must be IDENTICAL between `All` and
  // `CpuOnly` — no speaker may appear or vanish under a pure scheduling
  // change, only boundary jitter within slots both sides already agree
  // exist. Per spec §5.3 ("RECORDED DECISION — compute-unit placement and
  // what the gates actually prove", decision #4): that invariant is the
  // ENTIRE reason the divergence measured below (worst cosine 0.9241, every
  // embedding row moved) is a BOUNDED placement-noise risk rather than an
  // open "we might be losing speakers" one. It must be enforced here, not
  // merely printed above.
  assert_eq!(
    slot_diffs, 0,
    "ANE/GPU placement changed the consumed (chunk, slot) set relative to the cpu_only \
     reference — a speaker appeared or vanished under a scheduling change, not just boundary \
     jitter. That breaks the load-bearing premise of this whole study (spec §5.3, decision #4) \
     and must never be papered over by loosening a tolerance."
  );
  // Compounding trap: `measure()`'s embedding loop filters `golden.slots` to
  // `ours.contains(..)`, so if the slot sets ever diverged despite
  // `slot_diffs == 0` above (a bug shared with that bookkeeping), the filter
  // would silently SHRINK `compared_rows` instead of failing anything.
  // `golden_slots` is read straight off the golden, independent of
  // `only_ours`/`only_theirs`, so this closes that gap rather than
  // rephrasing the same check.
  assert_eq!(
    compared_rows, golden_slots,
    "only {compared_rows} of {golden_slots} golden-consumed slots were actually compared — the \
     embedding loop silently shrank its coverage instead of comparing every slot argmax \
     consumed."
  );

  assert!(
    seg_rate <= PLACEMENT_SEG_MISMATCH_TOL,
    "ANE/GPU placement flips {:.4}% of the hard speaker_ids decode, above the measured \
     {:.4}%. That is a CoreML scheduling change, not a port defect — but it moves the shipping \
     default's output, so record it rather than raising this.",
    seg_rate * 100.0,
    PLACEMENT_SEG_MISMATCH_TOL * 100.0
  );
  assert!(
    worst_cos >= PLACEMENT_COS_TOL,
    "ANE/GPU placement degrades an embedding to cosine {worst_cos:.9}, below the measured \
     {PLACEMENT_COS_TOL}. See above."
  );
}
