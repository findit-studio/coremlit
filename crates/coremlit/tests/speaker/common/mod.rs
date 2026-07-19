// Not every integration-test binary that includes `mod common;` uses every
// helper below (`model_io.rs` uses only the model-path helpers; the parity
// suites use the audio/golden/metric helpers). Each test file is its own
// crate, so an unused helper is dead code *in that crate* — allow it here so
// the shared module compiles clean under the workspace's `-D warnings` gate.
#![allow(dead_code)]

use std::{collections::BTreeMap, path::PathBuf};

use coremlit::audio::speaker::{
  extract::EXCLUDE_OVERLAP_MIN_FRAMES,
  segment::{POWERSET_CLASSES, SEG_CHUNK_SAMPLES, SEG_NUM_SLOTS, multilabel},
  window::DEFAULT_ONSET,
};
use sha2::{Digest, Sha256};

/// Directory containing the downloaded speakerkit model artifacts.
///
/// Overridable via `SPEAKERKIT_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/speakerkit` — gitignored, fetched dev-time per the
/// design spec §4 (mirrors whisperkit's `WHISPERKIT_TEST_MODELS`/`Models/`
/// convention, one directory level down for this crate's own model set).
pub fn models_dir() -> PathBuf {
  std::env::var_os("SPEAKERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("speakerkit")
    },
    PathBuf::from,
  )
}

/// Path to the decided segmentation artifact.
///
/// See `tests/model_io.rs`'s `// DECISION:` comment for the introspection
/// that picked `pyannote_segmentation.mlmodelc` over `Segmentation.mlmodelc`.
pub fn seg_path() -> PathBuf {
  models_dir().join("pyannote_segmentation.mlmodelc")
}

/// Directory containing the downloaded argmax `speakerkit-coreml` model
/// artifacts (`argmaxinc/speakerkit-coreml` on HuggingFace) — the SECOND
/// `ModelSource` this crate targets (Task 3's `ArgmaxSource`), acquired and
/// pinned independently of the FluidAudio-sourced artifacts [`models_dir`]
/// resolves.
///
/// Overridable via `ARGMAX_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/argmax-speakerkit` — gitignored, fetched dev-time via
/// `hf download argmaxinc/speakerkit-coreml --local-dir
/// Models/argmax-speakerkit`. Sibling convention to `models_dir`'s
/// `SPEAKERKIT_TEST_MODELS`/`Models/speakerkit`: a distinct env var and a
/// distinct default directory, since Task 3 loads both sources side by side
/// and each needs its own independently overridable path. See
/// `tests/argmax_model_io.rs`'s module doc for the pinned revision and the
/// full artifact/SHA-256 table.
pub fn argmax_models_dir() -> PathBuf {
  std::env::var_os("ARGMAX_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("argmax-speakerkit")
    },
    PathBuf::from,
  )
}

/// Path to the decided embedding artifact: the raw-waveform, in-graph-fbank
/// WeSpeaker v2 model (spec §2.4 — no separate fbank stage needed).
///
/// See `tests/model_io.rs`'s `// DECISION:` comment for why this is
/// `wespeaker_v2.mlmodelc` and not `wespeaker.mlmodelc`/`wespeaker_int8.mlmodelc`.
/// This is the **int8-palettized** shipping artifact; Gate 2's
/// conversion-fidelity comparison uses [`embed_fp32_path`] instead (matched
/// against dia-ort's fp32 ONNX — see `tests/parity_embed.rs`).
pub fn embed_path() -> PathBuf {
  models_dir().join("wespeaker_v2.mlmodelc")
}

/// Path to the **true fp32** embedding artifact, `wespeaker.mlmodelc`
/// (27 MB uncompressed float32 weights — `tests/model_io.rs`'s
/// `wespeaker_fp32_io_contract_equal_but_not_targeted`). Contract-equal to
/// the shipping int8 `wespeaker_v2.mlmodelc` but not quantized.
///
/// Gate 2 (embedding conversion fidelity, cosine ≥ 0.9999) is only
/// meaningful at MATCHED precision: dia-ort runs the fp32
/// `wespeaker_resnet34_lm.onnx` (26.7 MB float32), so the precision-matched
/// CoreML side is THIS fp32 artifact, not the int8 shipping one. The int8
/// path is measured separately for context (T3 recorded ~0.90-0.92 int8 vs
/// fp32, i.e. quantization cost, NOT a conversion defect).
pub fn embed_fp32_path() -> PathBuf {
  models_dir().join("wespeaker.mlmodelc")
}

/// A committed parity fixture: a short 16 kHz mono clip copied verbatim from
/// the `diarization` (dia) oracle repo's parity corpus, plus its provenance.
pub struct Fixture {
  /// Basename (no extension) of the committed WAV under `fixtures/audio/`
  /// and the golden JSON under `fixtures/golden/`.
  pub name: &'static str,
  /// Path within the dia repo this clip was copied from (provenance).
  pub source: &'static str,
  /// SHA-256 of the committed WAV, matching the dia-repo source byte-for-byte.
  pub sha256: &'static str,
  /// Human note on why this clip is in the set (coverage rationale).
  pub note: &'static str,
}

/// The parity fixture set (spec §6 Gates 1-2). Two short clips reused verbatim
/// from dia's `tests/parity/fixtures/*/clip_16k.wav`, chosen for ≤ 30 s length
/// (commit budget ~ `whisperkit/tests/fixtures/audio/ted_60.wav`) and to
/// exercise both the whole-window and the zero-padded final-chunk paths.
pub const FIXTURES: &[Fixture] = &[
  Fixture {
    name: "02_pyannote_sample",
    source: "diarization/tests/parity/fixtures/02_pyannote_sample/clip_16k.wav",
    sha256: "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e",
    note: "pyannote's canonical 30.0 s sample → exactly 3 full 10 s chunks (no padding)",
  },
  Fixture {
    name: "07_yuhewei_dongbei_english",
    source: "diarization/tests/parity/fixtures/07_yuhewei_dongbei_english/clip_16k.wav",
    sha256: "096890ba8ffbaf10ca770c5373bf6c6664777f9421595c2cb7780af8cb2e46ff",
    note: "25.26 s clip → 2 full chunks + 1 partial (exercises final-chunk zero-padding)",
  },
];

/// The exact `seg_model` provenance string frozen into every committed golden
/// (`tests/speaker/fixtures/golden/*.json`) AND the single source of truth that
/// `tests/generate_goldens.rs` writes when it regenerates one. Pinned here so
/// the two can never silently drift apart; `tests/golden_metadata.rs` asserts
/// each committed golden still carries this exact string in the ordinary
/// `cargo test` suite.
///
/// The phrase **"raw powerset logits" is a legacy misnomer kept verbatim** to
/// match the committed oracle — the identical decision `tests/parity_seg.rs`
/// documents for the golden's `seg_logits` field name: renaming it would churn
/// every committed golden (each ~270 KB) for zero behavioral gain, since no
/// gate reads this string. The values are in fact powerset **log-probabilities**
/// (`pyannote_segmentation.mlmodelc`'s MIL ends `softmax` → `log`, see
/// `coremlit::audio::speaker::segment`'s module doc; the committed ORT golden agrees — every
/// value `<= 0`, every 7-class row `sum(exp(row)) == 1`). Read the string as a
/// provenance tag, not a claim about the tensor's calibration.
pub const SEG_MODEL_LABEL: &str =
  "segmentation-3.0.onnx (dia bundled, ort CPU EP, raw powerset logits)";

/// Directory holding the committed parity fixtures (`audio/` + `golden/`).
pub fn fixtures_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/speaker/fixtures")
}

/// Committed WAV path for a fixture `name`.
pub fn audio_path(name: &str) -> PathBuf {
  fixtures_dir().join("audio").join(format!("{name}.wav"))
}

/// Committed golden-JSON path for a fixture `name`.
pub fn golden_path(name: &str) -> PathBuf {
  fixtures_dir().join("golden").join(format!("{name}.json"))
}

/// Loads a 16 kHz mono WAV as `f32` samples, the single source of truth for
/// both golden generation and parity replay so the two sides feed the models
/// byte-identical audio (the alignkit Gate-1 lesson: prove the inputs match).
///
/// 16-bit PCM is scaled by `1 / 32768`; float WAVs pass through. Asserts the
/// 16 kHz mono contract (pyannote/segmentation-3.0 and WeSpeaker are 16 kHz).
///
/// # Panics
/// If the file is missing, not 16 kHz mono, or an unsupported bit depth.
pub fn load_wav_16k_mono(path: &std::path::Path) -> Vec<f32> {
  let mut reader =
    hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 16_000, "{}: not 16 kHz", path.display());
  assert_eq!(spec.channels, 1, "{}: not mono", path.display());
  match spec.sample_format {
    hound::SampleFormat::Int => {
      assert_eq!(
        spec.bits_per_sample,
        16,
        "{}: only 16-bit int PCM supported",
        path.display()
      );
      reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("read i16 sample")) / 32_768.0)
        .collect()
    }
    hound::SampleFormat::Float => reader
      .samples::<f32>()
      .map(|s| s.expect("read f32 sample"))
      .collect(),
  }
}

/// Splits `samples` into non-overlapping [`SEG_CHUNK_SAMPLES`]-sample windows,
/// zero-padding the final partial chunk to a full window.
///
/// This is the exact per-chunk contract both models require (dia's
/// `SegmentModel::infer` / `EmbedModel::embed_chunk_with_frame_mask` both take
/// exactly `WINDOW_SAMPLES = 160_000` samples and reject other lengths; dia's
/// offline pipeline zero-pads the short tail, `owned.rs:469-475`). It is a
/// deliberate simplification of dia's *overlapping* sliding-window GRID
/// (`step_samples = 16_000`, `crate::window`) down to `step = window`: the
/// grid geometry is a pipeline concern (speakerkit's `window`/`extract`
/// modules and their own parity), whereas Gates 1-2 isolate the MODELS on
/// identical per-chunk inputs — a 160 000-sample window is processed
/// bit-identically by each model regardless of how the grid spaces windows.
///
/// Used verbatim by BOTH `tests/generate_goldens.rs` (feeding dia-ort) and the
/// parity suites (feeding CoreML), so the two sides are input-identical by
/// construction; [`fnv1a_f32`] hashes recorded in the golden re-prove it at
/// replay time.
pub fn chunk_and_pad(samples: &[f32]) -> Vec<Vec<f32>> {
  let n = samples.len().div_ceil(SEG_CHUNK_SAMPLES).max(1);
  (0..n)
    .map(|c| {
      let start = c * SEG_CHUNK_SAMPLES;
      let mut chunk = vec![0.0f32; SEG_CHUNK_SAMPLES];
      if start < samples.len() {
        let end = (start + SEG_CHUNK_SAMPLES).min(samples.len());
        chunk[..end - start].copy_from_slice(&samples[start..end]);
      }
      chunk
    })
    .collect()
}

/// FNV-1a-64 over the little-endian bytes of `samples` — a stable,
/// platform-independent fingerprint of an exact `f32` buffer.
///
/// The input-match proof: golden generation records this hash of the precise
/// slice it hands to `ort::Session::run`; the parity suites recompute it on
/// the slice they hand to CoreML `predict` and assert equality, proving both
/// sides fed the model element-identical audio (a divergence on mismatched
/// inputs is a harness bug, not a model finding — the alignkit Gate-1 lesson).
pub fn fnv1a_f32(samples: &[f32]) -> u64 {
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  for &s in samples {
    for b in s.to_le_bytes() {
      h ^= u64::from(b);
      h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
  }
  h
}

/// Lowercase 16-hex-digit rendering of a [`fnv1a_f32`] hash for JSON storage.
pub fn fnv_hex(h: u64) -> String {
  format!("{h:016x}")
}

/// SHA-256 (FIPS 180-4) of a byte slice via the `sha2` crate, rendered as a
/// lowercase 64-hex-digit digest.
///
/// Used by `tests/argmax_model_io.rs` to pin the exact bytes of the
/// downloaded argmax model artifacts. Previously hand-rolled here on the
/// belief that a dependency addition was out of scope for that task; it
/// wasn't (root `Cargo.toml`'s `[workspace.dependencies]` is this
/// workspace's one place every dependency is declared, and editing it for
/// this is in scope), and shipping a hand-rolled cryptographic primitive as
/// reusable test infrastructure was unnecessary risk/maintenance for no
/// benefit over the well-tested upstream crate. Unlike [`fnv1a_f32`] below —
/// a non-cryptographic fingerprint with no standard-crate equivalent this
/// concise, so it stays hand-rolled.
pub fn sha256_hex(data: &[u8]) -> String {
  Sha256::digest(data)
    .iter()
    .map(|byte| format!("{byte:02x}"))
    .collect()
}

/// Cosine similarity of two equal-length vectors, accumulated in `f64` for
/// precision (Gate 2's per-`(chunk, slot)` metric).
///
/// Rejects the two degenerate inputs that silently poison the metric rather
/// than returning a `NaN` a downstream fold would discard: a non-finite
/// element (which propagates `NaN` straight into the result) and a zero-norm
/// vector (`0 / 0 == NaN`). Gate 2 folds per-slot cosines with
/// `worst.min(cos)`, and `f64::min` KEEPS the non-`NaN` operand — so a
/// shape-compatible all-zero (or otherwise degenerate) embedder would leave
/// `worst == 1.0` and report PERFECT parity from garbage (M1). A loud panic
/// naming the offending vector turns that silent pass into a failure.
///
/// # Panics
/// If the lengths differ, either vector contains a non-finite element, or
/// either vector has a zero L2 norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
  assert_eq!(a.len(), b.len(), "cosine: length mismatch");
  assert!(
    a.iter().all(|v| v.is_finite()),
    "cosine: vector `a` contains a non-finite element"
  );
  assert!(
    b.iter().all(|v| v.is_finite()),
    "cosine: vector `b` contains a non-finite element"
  );
  let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
  for (&x, &y) in a.iter().zip(b) {
    let (x, y) = (f64::from(x), f64::from(y));
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  assert!(na > 0.0, "cosine: vector `a` has zero norm");
  assert!(nb > 0.0, "cosine: vector `b` has zero norm");
  dot / (na.sqrt() * nb.sqrt())
}

/// Numerically stable softmax over one [`POWERSET_CLASSES`] logit row (the
/// same shape dia applies downstream, `segment::powerset::softmax_row`).
/// Diagnostic only: powerset segmentation's pipeline-relevant output is the
/// softmax probability (then onset/argmax), so softmax max-abs characterizes a
/// raw-logit divergence's actual downstream impact.
///
/// # Panics
/// If `row.len() != POWERSET_CLASSES`.
pub fn softmax_row(row: &[f32]) -> [f32; POWERSET_CLASSES] {
  assert_eq!(row.len(), POWERSET_CLASSES, "softmax_row: bad row length");
  let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let mut out = [0f32; POWERSET_CLASSES];
  let mut sum = 0f32;
  for (o, &l) in out.iter_mut().zip(row) {
    *o = (l - max).exp();
    sum += *o;
  }
  for o in &mut out {
    *o /= sum;
  }
  out
}

/// Maximum absolute per-element difference between two equal-length vectors,
/// in `f64` (Gate 1's segmentation-logit metric).
///
/// # Panics
/// If the lengths differ.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
  assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch");
  a.iter()
    .zip(b)
    .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
    .fold(0.0, f64::max)
}

/// Max `|Σexp(row) − 1|` tolerated when validating a powerset log-softmax row.
/// The committed goldens sit at ≤ 2.3e-7 (an f32 `softmax → log` round-trip);
/// raw logits miss by many orders of magnitude, so this distinguishes the two
/// with a wide margin while never flaking on f32 rounding.
pub const SEG_ROW_SUM_EXP_TOL: f64 = 1e-4;

/// Validates one chunk's flattened `[num_frames * POWERSET_CLASSES]` powerset
/// segmentation output as LOG-PROBABILITIES: every element finite and `≤ 0`, and
/// each [`POWERSET_CLASSES`]-wide row normalized so `Σ exp = 1` (within
/// [`SEG_ROW_SUM_EXP_TOL`]).
///
/// dia-ort's segmentation MIL ends `softmax → log` and the CoreML side matches;
/// the committed goldens store that quantity under the legacy `seg_logits` name.
/// A future model emitting RAW logits (positive values, rows that do not sum-exp
/// to 1) with the argmax ORDERING preserved would decode to the same speakers yet
/// break this invariant — which `generate_goldens.rs`'s prose used to only
/// assert, never check (codex r7 F4). This runs BOTH in the generator before
/// serialization and against the committed goldens in the ordinary suite, so that
/// drift cannot land silently.
///
/// # Errors
/// Returns the first violation as a message: a length not matching
/// `num_frames * POWERSET_CLASSES`, a non-finite or positive element, or a row
/// whose `Σ exp` departs from 1 by more than [`SEG_ROW_SUM_EXP_TOL`].
pub fn check_seg_log_probs(seg_logits: &[f32], num_frames: usize) -> Result<(), String> {
  let expected = num_frames * POWERSET_CLASSES;
  if seg_logits.len() != expected {
    return Err(format!(
      "seg_logits length {} != num_frames*POWERSET_CLASSES ({num_frames}*{POWERSET_CLASSES}={expected})",
      seg_logits.len()
    ));
  }
  for (f, row) in seg_logits
    .as_chunks::<POWERSET_CLASSES>()
    .0
    .iter()
    .enumerate()
  {
    let mut sum_exp = 0.0_f64;
    for (k, &v) in row.iter().enumerate() {
      if !v.is_finite() {
        return Err(format!("frame {f} class {k}: non-finite log-prob {v}"));
      }
      if v > 0.0 {
        return Err(format!(
          "frame {f} class {k}: log-prob {v} > 0 — a probability's log is ≤ 0; this looks like a \
           raw logit"
        ));
      }
      sum_exp += f64::from(v).exp();
    }
    let dev = (sum_exp - 1.0).abs();
    if dev > SEG_ROW_SUM_EXP_TOL {
      return Err(format!(
        "frame {f}: Σexp(row) = {sum_exp:.9}, off 1.0 by {dev:.3e} (> {SEG_ROW_SUM_EXP_TOL:.0e}) — \
         the row is not a normalized log-softmax (raw logits, or a broken normalization)"
      ));
    }
  }
  Ok(())
}

/// Hard argmax over one frame's [`POWERSET_CLASSES`] logits, ties toward the
/// lowest index (`>` seeded at class 0) — the exact rule speakerkit's shipping
/// `segment::multilabel` and dia's `powerset_to_speakers_hard` both use. Gate
/// 1's multilabel-flip check argmaxes both models' logits with this and counts
/// per-frame disagreements.
///
/// # Panics
/// If `row.len() != POWERSET_CLASSES`.
pub fn powerset_argmax(row: &[f32]) -> usize {
  assert_eq!(
    row.len(),
    POWERSET_CLASSES,
    "powerset_argmax: bad row length"
  );
  let mut argmax = 0usize;
  let mut max = row[0];
  for (k, &v) in row.iter().enumerate().skip(1) {
    if v > max {
      max = v;
      argmax = k;
    }
  }
  argmax
}

/// Encodes a per-frame boolean mask as a compact `'0'`/`'1'` string for JSON.
pub fn mask_to_string(mask: &[bool]) -> String {
  mask.iter().map(|&b| if b { '1' } else { '0' }).collect()
}

/// Strictly decodes a [`mask_to_string`] bit string into `Vec<bool>`,
/// hard-failing unless it is EXACTLY `expected_len` characters and every
/// character is `'0'` or `'1'`. `context` names the offending subject (e.g.
/// `"<fixture>: chunk C slot S: mask"`) for the panic.
///
/// This is the SINGLE strict decoder for both committed golden-mask families —
/// the embedding golden's per-frame pooling `mask` ([`load_golden`]) and the
/// argmax-Swift golden's `activeFrames` (`parity_argmax_swift::parse_active_frames`).
/// The old lenient decode (`c == '1'`, any length, every non-`'1'` char →
/// `false`) is what let a gate pass vacuously: a truncated, empty, or junk mask
/// decoded to a short or all-`false` `Vec<bool>` while the comparison still
/// REPORTED full coverage — and an OVER-long mask was silently truncated back by
/// the model's frame padding (`embed::repeat_pad_f32`), producing byte-identical
/// model input. A wrong length or an unexpected character is a MALFORMED golden,
/// never a slot with fewer active frames, so it is rejected here before a single
/// fidelity number is read.
///
/// # Panics
/// If any character is not `'0'`/`'1'`, or the decoded length is not `expected_len`.
pub fn parse_bit_mask(context: &str, expected_len: usize, raw: &str) -> Vec<bool> {
  let bits: Vec<bool> = raw
    .chars()
    .map(|c| match c {
      '0' => false,
      '1' => true,
      other => panic!(
        "{context} contains {other:?} — a golden bit mask is hard-binary, so only '0'/'1' are \
         valid; an unknown character is a malformed golden, not an inactive frame."
      ),
    })
    .collect();
  assert_eq!(
    bits.len(),
    expected_len,
    "{context} has {} characters, expected exactly {expected_len}. A short, empty, or over-long \
     mask makes the comparison vacuous (or is truncated back by the model's frame padding) while \
     the gate still reports full coverage.",
    bits.len()
  );
  bits
}

/// One embedded speaker slot within a golden chunk: the per-frame mask fed to
/// the embedding model and the resulting dia-ort reference embedding.
pub struct GoldenSlot {
  /// Speaker-slot index (0..[`coremlit::audio::speaker::segment::SEG_NUM_SLOTS`]).
  pub slot: usize,
  /// The per-frame pooling mask (length = segmentation frame count) fed to
  /// BOTH backends — stored verbatim so the parity side is mask-identical.
  pub mask: Vec<bool>,
  /// dia-ort's raw (un-normalized) 256-d WeSpeaker embedding for this slot.
  pub embedding: Vec<f32>,
}

/// One chunk's dia-ort reference outputs plus its input fingerprint.
pub struct GoldenChunk {
  /// Sample count of the (zero-padded) chunk fed to the models.
  pub input_len: usize,
  /// [`fnv1a_f32`] of the exact chunk samples dia-ort was run on.
  pub input_fnv1a: u64,
  /// dia-ort's flattened `[num_frames * POWERSET_CLASSES]` powerset segmentation
  /// LOG-PROBABILITIES (frame-major; its MIL ends `softmax → log`, kept under the
  /// legacy `seg_logits` name) — the Gate 1 reference. Validated as log-probs by
  /// [`check_seg_log_probs`].
  pub seg_logits: Vec<f32>,
  /// The embedded slots for this chunk (skipped/degenerate slots omitted).
  pub slots: Vec<GoldenSlot>,
}

/// A parity golden: dia-ort reference tensors for one fixture, produced by
/// `tests/generate_goldens.rs` and committed as the pinned oracle.
pub struct Golden {
  /// Fixture name (matches [`Fixture::name`]).
  pub fixture: String,
  /// Number of non-overlapping chunks (`== chunks.len()`).
  pub num_chunks: usize,
  /// Segmentation frames per chunk (589 for pyannote/segmentation-3.0).
  pub num_frames: usize,
  /// Per-chunk reference outputs.
  pub chunks: Vec<GoldenChunk>,
}

/// Loads and parses a committed golden JSON for fixture `name`.
///
/// # Panics
/// If the file is missing or malformed — a committed golden is a hard
/// dependency of the parity suites, not an optional input.
pub fn load_golden(name: &str) -> Golden {
  let path = golden_path(name);
  let bytes =
    std::fs::read(&path).unwrap_or_else(|e| panic!("read golden {}: {e}", path.display()));
  let v: serde_json::Value =
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse golden {name}: {e}"));

  let num_frames = v["num_frames"].as_u64().expect("num_frames") as usize;
  let chunks = v["chunks"]
    .as_array()
    .expect("chunks array")
    .iter()
    .enumerate()
    .map(|(c_idx, c)| {
      let seg_logits = c["seg_logits"]
        .as_array()
        .expect("seg_logits array")
        .iter()
        .map(|x| x.as_f64().expect("logit f64") as f32)
        .collect();
      let slots = c["slots"]
        .as_array()
        .expect("slots array")
        .iter()
        .map(|s| {
          let slot = s["slot"].as_u64().expect("slot") as usize;
          GoldenSlot {
            slot,
            // Strict decode: exactly `num_frames` chars, `{'0','1'}` only. A
            // malformed mask is a hard error, not a lenient reinterpretation.
            mask: parse_bit_mask(
              &format!("{name}: chunk {c_idx} slot {slot}: mask"),
              num_frames,
              s["mask"].as_str().expect("mask string"),
            ),
            embedding: s["embedding"]
              .as_array()
              .expect("embedding array")
              .iter()
              .map(|x| x.as_f64().expect("embed f64") as f32)
              .collect(),
          }
        })
        .collect();
      let hex = c["input_fnv1a"].as_str().expect("input_fnv1a hex");
      GoldenChunk {
        input_len: c["input_len"].as_u64().expect("input_len") as usize,
        input_fnv1a: u64::from_str_radix(hex, 16).expect("parse fnv hex"),
        seg_logits,
        slots,
      }
    })
    .collect();

  Golden {
    fixture: v["fixture"].as_str().expect("fixture").to_string(),
    num_chunks: v["num_chunks"].as_u64().expect("num_chunks") as usize,
    num_frames,
    chunks,
  }
}

/// Independently re-derives one chunk's expected per-slot embedding masks from
/// its committed powerset segmentation log-probabilities, reproducing the golden
/// generator's rule (`generate_goldens::derive_slot_masks`) WITHOUT reading the
/// golden's stored slot list. `None` for a slot = it has no active frame, so the
/// generator emitted no golden slot for it.
///
/// The decode is speakerkit's shipping [`multilabel`] (direct argmax over the
/// log-probs). The generator decodes via dia's `softmax`-then-argmax; the two
/// coincide on every committed golden row (pinned by
/// `parity_seg::golden_direct_and_dia_decode_agree`), so this reproduces the
/// generator's exact roster and masks on committed data while depending on
/// neither dia nor the stored slots. The overlap-exclusion below is the same
/// `embedding_exclude_overlap` port speakerkit's `extract::derive_slot_plans`
/// runs, keyed on the SAME production [`DEFAULT_ONSET`] and
/// [`EXCLUDE_OVERLAP_MIN_FRAMES`] (a slot is active at a frame iff its hard 0/1
/// value is `>= DEFAULT_ONSET`; the overlap-excluded mask falls back to raw when
/// `<= EXCLUDE_OVERLAP_MIN_FRAMES` clean frames remain).
///
/// # Panics
/// If `seg_logits.len() != num_frames * POWERSET_CLASSES` (via [`multilabel`]).
pub fn derive_expected_slot_masks(
  seg_logits: &[f32],
  num_frames: usize,
) -> [Option<Vec<bool>>; SEG_NUM_SLOTS] {
  let slab = multilabel(seg_logits, num_frames);
  let onset = f64::from(DEFAULT_ONSET);

  // Per-frame "clean" indicator: fewer than 2 of the slots active (dia's
  // `owned.rs:536-549`). Computed once over all slots, before the per-slot loop.
  let mut clean = vec![false; num_frames];
  for (f, clean_f) in clean.iter_mut().enumerate() {
    let active = (0..SEG_NUM_SLOTS)
      .filter(|&s| slab[f * SEG_NUM_SLOTS + s] >= onset)
      .count();
    *clean_f = active < 2;
  }

  core::array::from_fn(|s| {
    let mut frame_mask = vec![false; num_frames];
    let mut any = false;
    for (f, m) in frame_mask.iter_mut().enumerate() {
      *m = slab[f * SEG_NUM_SLOTS + s] >= onset;
      any |= *m;
    }
    if !any {
      return None; // dia drops a slot with no active frame (owned.rs:561-571).
    }
    // Overlap-excluded mask = raw AND clean; fall back to raw when too few clean
    // frames remain (`<=`, owned.rs:573-591).
    let mut used = vec![false; num_frames];
    let mut clean_count = 0usize;
    for (f, u) in used.iter_mut().enumerate() {
      *u = frame_mask[f] && clean[f];
      if *u {
        clean_count += 1;
      }
    }
    if clean_count <= EXCLUDE_OVERLAP_MIN_FRAMES {
      used = frame_mask;
    }
    Some(used)
  })
}

/// Asserts a golden's stored `(chunk, slot)` roster and every per-frame mask
/// EXACTLY match the roster independently re-derived from its committed
/// `seg_logits` (via [`derive_expected_slot_masks`]), spanning EVERY chunk, and
/// returns the total number of `(chunk, slot)` slots in that roster — the count
/// Gate 2 must fold in.
///
/// This closes the embedding half of the golden-loader leniency class. The Gate
/// 2 replay used to compare only the slots the golden happened to carry, gated
/// on a merely global `n > 0`, so deleting one fixture's slots (or an entire
/// fixture) left the other fixture's count non-zero and stayed green. Here the
/// expected roster is derived from a DIFFERENT part of the golden (the
/// segmentation tensor) than the part under test (the slot list), so a missing
/// slot, an extra slot, a moved mask bit, a duplicate slot, or a dropped fixture
/// all fail — per chunk, named.
///
/// # Panics
/// On any roster/mask divergence, or a duplicate slot within a chunk.
pub fn assert_golden_roster(golden: &Golden) -> usize {
  let mut total = 0usize;
  for (c_idx, chunk) in golden.chunks.iter().enumerate() {
    let derived = derive_expected_slot_masks(&chunk.seg_logits, golden.num_frames);
    let expected: BTreeMap<usize, &Vec<bool>> = derived
      .iter()
      .enumerate()
      .filter_map(|(s, m)| m.as_ref().map(|mask| (s, mask)))
      .collect();

    let mut stored: BTreeMap<usize, &Vec<bool>> = BTreeMap::new();
    for slot in &chunk.slots {
      assert!(
        stored.insert(slot.slot, &slot.mask).is_none(),
        "{}: chunk {c_idx}: slot {} appears more than once in the golden",
        golden.fixture,
        slot.slot
      );
    }

    let stored_roster: Vec<usize> = stored.keys().copied().collect();
    let expected_roster: Vec<usize> = expected.keys().copied().collect();
    assert_eq!(
      stored_roster, expected_roster,
      "{}: chunk {c_idx}: stored (chunk, slot) roster {stored_roster:?} != roster \
       {expected_roster:?} independently derived from seg_logits — a golden slot was added or \
       dropped",
      golden.fixture
    );

    for (s, exp_mask) in &expected {
      let got = stored
        .get(s)
        .expect("roster equality checked directly above");
      assert_eq!(
        got,
        exp_mask,
        "{}: chunk {c_idx} slot {s}: stored mask ({} active frame(s)) != mask independently \
         derived from seg_logits ({} active frame(s))",
        golden.fixture,
        got.iter().filter(|&&b| b).count(),
        exp_mask.iter().filter(|&&b| b).count()
      );
    }
    total += expected.len();
  }
  total
}

// ── Host-class provenance for the committed-Swift-golden parity gate ────────
//
// CoreML `CpuOnly` kernels ship with the OS and are not contracted to produce
// identical floats across macOS builds or chip generations (#36), so the parity
// gate stamps each golden with the host-class it was generated on and enforces
// its tight bound only against a matching host (see `check_host_class`). This
// block is duplicated verbatim in the vad and speaker `common/mod.rs` — the
// repo's self-contained-`common` convention, the same one that already
// duplicates `fnv1a_f32`/`load_wav_16k_mono`; each suite's hermetic tests drive
// its own copy, which is what guards the two copies against drift.

/// The host-class identity that determines CoreML `CpuOnly` float
/// reproducibility: macOS build (the OS binary set every CPU kernel ships
/// in), chip (kernel dispatch varies by microarchitecture), process arch
/// (Rosetta), plus the human-readable product version (fully determined by
/// the build; carried for diagnostics). Deliberately NOT included: Xcode /
/// Swift toolchain (compiles the dumper, not the runtime kernels — inputs
/// are FNV-pinned and model bytes SHA-pinned), `hw.model` (same chip + build
/// ⇒ same CPU floats), RAM/core counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostClass {
  pub os_build: String,
  pub os_product_version: String,
  pub chip: String,
  pub arch: String,
}

impl HostClass {
  /// The RUNNING host's class. Reads the sysctl keys the Swift dumpers read
  /// (`kern.osversion`, `kern.osproductversion`, `machdep.cpu.brand_string`) by
  /// shelling out to `/usr/sbin/sysctl`, and normalizes the process arch to the
  /// `arm64`/`x86_64` spelling the dumpers record. Called only from model-gated
  /// tests and the `running_host_class_is_well_formed` smoke test — hermetic
  /// predicate tests use synthetic values.
  ///
  /// Production code (`src/audio/whisper/model/mod.rs::device_identifier`)
  /// deliberately uses `libc::sysctlbyname`; this test-side reader deliberately
  /// shells out instead — spawn cost is irrelevant in a model-gated test and it
  /// keeps the test tree free of `unsafe`.
  ///
  /// # Panics
  /// With a `host-class gate:` message if a sysctl read fails or is empty
  /// (model-gated dev machines only — sysctl always exists on macOS).
  pub fn running() -> Self {
    HostClass {
      os_build: sysctl_string("kern.osversion"),
      os_product_version: sysctl_string("kern.osproductversion"),
      chip: sysctl_string("machdep.cpu.brand_string"),
      // The aarch64 -> arm64 normalization is load-bearing: without it every
      // Apple-Silicon run would mismatch every golden (the dumpers record
      // `arm64` via compile-time `#if arch(arm64)`, and compile-time arch IS
      // the process arch that governs the in-process `CpuOnly` kernels).
      arch: match std::env::consts::ARCH {
        "aarch64" => "arm64".to_string(),
        other => other.to_string(),
      },
    }
  }

  /// Reads the OPTIONAL `generationHost` object from a golden's JSON. Absent or
  /// `null` → `Ok(None)` (a legacy golden, pre-host-provenance). Present → must
  /// be an object whose `osBuild`/`osProductVersion`/`chip`/`arch` are all
  /// non-empty strings; unknown extra keys inside the object are tolerated.
  ///
  /// FORM only — the host MATCH is [`check_host_class`]'s job, never the
  /// parser's: the hermetic loader tests and the committed goldens must parse on
  /// EVERY host (this one included).
  ///
  /// # Errors
  /// If `generationHost` is present but not an object, or is missing any of the
  /// four fields as a non-empty string. Every message contains `generationHost`.
  pub fn from_golden(name: &str, v: &serde_json::Value) -> Result<Option<Self>, String> {
    let host = match v.get("generationHost") {
      None | Some(serde_json::Value::Null) => return Ok(None),
      Some(serde_json::Value::Object(map)) => map,
      Some(other) => {
        return Err(format!(
          "{name}: `generationHost` is {other} — expected an object with osBuild / \
           osProductVersion / chip / arch string fields"
        ));
      }
    };
    let field = |key: &str| -> Result<String, String> {
      host
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{name}: `generationHost.{key}` is missing, not a string, or empty"))
    };
    Ok(Some(HostClass {
      os_build: field("osBuild")?,
      os_product_version: field("osProductVersion")?,
      chip: field("chip")?,
      arch: field("arch")?,
    }))
  }
}

impl std::fmt::Display for HostClass {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "macOS {} (build {}), {}, {}",
      self.os_product_version, self.os_build, self.chip, self.arch
    )
  }
}

/// Reads one sysctl value as a trimmed string via `/usr/sbin/sysctl -n`
/// (absolute path — PATH-independent).
///
/// # Panics
/// With a `host-class gate:` message on spawn failure, a non-zero exit, or
/// empty output.
fn sysctl_string(key: &str) -> String {
  let output = std::process::Command::new("/usr/sbin/sysctl")
    .args(["-n", key])
    .output()
    .unwrap_or_else(|e| panic!("host-class gate: cannot spawn /usr/sbin/sysctl for `{key}`: {e}"));
  assert!(
    output.status.success(),
    "host-class gate: `/usr/sbin/sysctl -n {key}` exited {}",
    output.status
  );
  let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
  assert!(
    !value.is_empty(),
    "host-class gate: `/usr/sbin/sysctl -n {key}` produced empty output"
  );
  value
}

/// Verdict of the host-class gate for one golden.
#[derive(Debug, PartialEq, Eq)]
pub enum HostVerdict {
  /// `generationHost` recorded and equal to the running host-class: the tight
  /// parity bounds apply and a failure is cleanly attributable to the port.
  Match,
  /// The golden predates host provenance (no `generationHost`): the tight
  /// bounds still apply (a PASS is sound evidence on any host — a port bug
  /// exactly cancelling host drift under these bounds is not a real risk),
  /// but a FAILURE is ambiguous between a port defect and host-CoreML drift —
  /// append [`legacy_failure_note`] to fidelity failure messages.
  LegacyUnknown,
}

/// THE host-class match predicate + mismatch diagnosis. Pure — no I/O — so the
/// hermetic tests drive it with synthetic host-class values.
///
/// # Errors
/// The full actionable diagnosis for a recorded-but-different host; the caller
/// panics with it BEFORE any CoreML number is produced.
pub fn check_host_class(
  fixture: &str,
  recorded: Option<&HostClass>,
  running: &HostClass,
  regen_script: &str,
) -> Result<HostVerdict, String> {
  match recorded {
    None => Ok(HostVerdict::LegacyUnknown),
    Some(r) if r == running => Ok(HostVerdict::Match),
    Some(r) => Err(format!(
      "{fixture}: committed golden was generated on a DIFFERENT host-class.\n  \
       golden host : {r}\n  this host   : {running}\n\
       CoreML CpuOnly floating point is not contracted portable across macOS builds or chips,\n\
       so the tight parity bound would misattribute host float drift to the port. This failure\n\
       is NOT evidence of a port defect. To test the port on this machine, regenerate a\n\
       same-host oracle and re-run:\n  {regen_script}\n\
       Do NOT widen the parity tolerances instead — the tight bounds are what catch real\n\
       stitching/index-mapping regressions on a matching host."
    )),
  }
}

/// The ambiguity note appended to a fidelity failure when the golden predates
/// host-class provenance (a [`HostVerdict::LegacyUnknown`] golden): the failure
/// cannot be cleanly attributed to the port versus host-CoreML drift, and the
/// fix is a same-host regeneration, never a widened tolerance.
pub fn legacy_failure_note(regen_script: &str) -> String {
  format!(
    "\nNOTE: this golden predates host-class provenance (no `generationHost` field), so this\n\
     failure is AMBIGUOUS between a port defect and host-CoreML CpuOnly float drift (CpuOnly\n\
     floats are not contracted portable across macOS builds/chips). Regenerate a same-host\n\
     oracle via {regen_script} to disambiguate — regeneration also stamps `generationHost`.\n\
     Do NOT widen the tolerance."
  )
}
