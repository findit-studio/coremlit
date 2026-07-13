// Not every integration-test binary that includes `mod common;` uses every
// helper below (`model_io.rs` uses only the model-path helpers; the parity
// suites use the audio/golden/metric helpers). Each test file is its own
// crate, so an unused helper is dead code *in that crate* — allow it here so
// the shared module compiles clean under the workspace's `-D warnings` gate.
#![allow(dead_code)]

use std::path::PathBuf;

use dia_coreml::segment::{POWERSET_CLASSES, SEG_CHUNK_SAMPLES};

/// Directory containing the downloaded dia-coreml model artifacts.
///
/// Overridable via `DIA_COREML_TEST_MODELS`; otherwise falls back to
/// `<workspace>/Models/dia-coreml` — gitignored, fetched dev-time per the
/// design spec §4 (mirrors whisperkit's `WHISPERKIT_TEST_MODELS`/`Models/`
/// convention, one directory level down for this crate's own model set).
pub fn models_dir() -> PathBuf {
  std::env::var_os("DIA_COREML_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("dia-coreml")
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

/// Directory holding the committed parity fixtures (`audio/` + `golden/`).
pub fn fixtures_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
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
/// grid geometry is a pipeline concern (dia-coreml's `window`/`extract`
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

/// Cosine similarity of two equal-length vectors, accumulated in `f64` for
/// precision (Gate 2's per-`(chunk, slot)` metric).
///
/// # Panics
/// If the lengths differ.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
  assert_eq!(a.len(), b.len(), "cosine: length mismatch");
  let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
  for (&x, &y) in a.iter().zip(b) {
    let (x, y) = (f64::from(x), f64::from(y));
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
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

/// Hard argmax over one frame's [`POWERSET_CLASSES`] logits, ties toward the
/// lowest index (`>` seeded at class 0) — the exact rule dia-coreml's shipping
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

/// Decodes a [`mask_to_string`] `'0'`/`'1'` string back to `Vec<bool>`.
pub fn mask_from_string(s: &str) -> Vec<bool> {
  s.chars().map(|c| c == '1').collect()
}

/// One embedded speaker slot within a golden chunk: the per-frame mask fed to
/// the embedding model and the resulting dia-ort reference embedding.
pub struct GoldenSlot {
  /// Speaker-slot index (0..[`dia_coreml::segment::SEG_NUM_SLOTS`]).
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
  /// dia-ort's flattened `[num_frames * POWERSET_CLASSES]` raw segmentation
  /// logits (frame-major) — the Gate 1 reference.
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
    .map(|c| {
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
        .map(|s| GoldenSlot {
          slot: s["slot"].as_u64().expect("slot") as usize,
          mask: mask_from_string(s["mask"].as_str().expect("mask string")),
          embedding: s["embedding"]
            .as_array()
            .expect("embedding array")
            .iter()
            .map(|x| x.as_f64().expect("embed f64") as f32)
            .collect(),
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
