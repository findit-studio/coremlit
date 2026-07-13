// Not every integration-test binary that includes `mod common;` uses every
// helper below (`model_io.rs` uses only the model-path helpers; the parity
// suites use the audio/golden/metric helpers). Each test file is its own
// crate, so an unused helper is dead code *in that crate* — allow it here so
// the shared module compiles clean under the workspace's `-D warnings` gate.
#![allow(dead_code)]

use std::path::PathBuf;

use speakerkit::segment::{POWERSET_CLASSES, SEG_CHUNK_SAMPLES};

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

/// Minimal, dependency-free SHA-256 (FIPS 180-4) over a byte slice, rendered
/// as a lowercase 64-hex-digit digest.
///
/// Hand-rolled rather than pulling in a `sha2`-style crate dependency for
/// one test-only integrity check — the same rationale as [`fnv1a_f32`]
/// above (a different hand-rolled fingerprint for the same class of
/// problem: prove two byte sequences are identical), and it keeps this
/// task's entire diff inside `crates/speakerkit/`. Used by
/// `tests/argmax_model_io.rs` to pin the exact bytes of the downloaded
/// argmax model artifacts; verified against the FIPS 180-4 / RFC 6234
/// known-answer vectors by that file's (non-ignored, hermetic)
/// `sha256_hex_matches_known_vectors`.
pub fn sha256_hex(data: &[u8]) -> String {
  #[rustfmt::skip]
  const K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5,
    0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
    0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc,
    0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
    0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3,
    0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5,
    0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
    0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
  ];
  let mut h: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
  ];

  // Padding: a single `1` bit (byte 0x80, since the input is byte-aligned),
  // zero bytes up to the last 8 bytes of a 64-byte block, then the original
  // bit length as a big-endian u64 — the standard merkle-damgard tail.
  let bit_len = (data.len() as u64) * 8;
  let mut msg = data.to_vec();
  msg.push(0x80);
  while msg.len() % 64 != 56 {
    msg.push(0);
  }
  msg.extend_from_slice(&bit_len.to_be_bytes());

  let (chunks, remainder) = msg.as_chunks::<64>();
  debug_assert!(remainder.is_empty(), "padding always rounds up to 64 bytes");
  for chunk in chunks {
    let mut w = [0u32; 64];
    for (i, word) in w.iter_mut().take(16).enumerate() {
      *word = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().expect("4 bytes"));
    }
    for i in 16..64 {
      let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
      let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16]
        .wrapping_add(s0)
        .wrapping_add(w[i - 7])
        .wrapping_add(s1);
    }

    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
      (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    for i in 0..64 {
      let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
      let ch = (e & f) ^ ((!e) & g);
      let temp1 = hh
        .wrapping_add(s1)
        .wrapping_add(ch)
        .wrapping_add(K[i])
        .wrapping_add(w[i]);
      let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
      let maj = (a & b) ^ (a & c) ^ (b & c);
      let temp2 = s0.wrapping_add(maj);

      hh = g;
      g = f;
      f = e;
      e = d.wrapping_add(temp1);
      d = c;
      c = b;
      b = a;
      a = temp1.wrapping_add(temp2);
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
  }

  h.iter().map(|word| format!("{word:08x}")).collect()
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

/// Decodes a [`mask_to_string`] `'0'`/`'1'` string back to `Vec<bool>`.
pub fn mask_from_string(s: &str) -> Vec<bool> {
  s.chars().map(|c| c == '1').collect()
}

/// One embedded speaker slot within a golden chunk: the per-frame mask fed to
/// the embedding model and the resulting dia-ort reference embedding.
pub struct GoldenSlot {
  /// Speaker-slot index (0..[`speakerkit::segment::SEG_NUM_SLOTS`]).
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
