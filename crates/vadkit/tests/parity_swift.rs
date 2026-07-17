//! The FluidAudio Swift trace gate (design spec §6 model-layer oracle). Pins
//! `vadkit::VadModel`'s per-chunk speech probabilities against committed traces
//! from FluidAudio's OWN `VadManager.process` — the reference implementation of
//! the 256 ms chunking, 64-sample context stitching, repeat-last final-chunk
//! padding and LSTM state carry-forward this crate ports. Both sides run the
//! SAME `Models/vadkit` artifact on `ComputeUnits::CpuOnly`, so the port is
//! held to a near-bit-exact bound (measured, then pinned two-sided;
//! [`TRACE_TOL`]).
//!
//! Regenerate the traces with `tests/swift/regen_goldens.sh`.
//!
//! # Why this catches a stitching bug
//!
//! A skewed context, a zero-padded (instead of repeat-last) final chunk, a
//! dropped recurrent-state field, or a non-zero first-chunk context all change
//! the model input on at least some chunks, so the committed trace diverges
//! past [`TRACE_TOL`] (`tests/model_state.rs` measures a one-sample context
//! skew moving ~10 % of chunks by up to ~5e-3, an order above the bound). The
//! `strict_loader_*` hermetic tests additionally prove a MALFORMED golden is
//! rejected before a single fidelity number is read.

mod common;

use coremlit::ComputeUnits;
use vadkit::{
  CHUNK_SAMPLES, CONTEXT_SAMPLES, MODEL_INPUT_SAMPLES, STATE_SIZE, VadModel, VadModelOptions,
};

/// The two committed fixtures (spec §6: real speech, one multi-speaker, ≥ 2
/// clips, ≥ 40 chunks). `02_pyannote_sample` → 118 chunks, `07_yuhewei_dongbei_
/// english` → 99 chunks (its short final chunk exercises the repeat-last
/// padding path): 217 chunks total.
const GATE_FIXTURES: &[&str] = &["02_pyannote_sample", "07_yuhewei_dongbei_english"];

/// Worst tolerated per-chunk |Δ| between `vadkit::VadModel` and FluidAudio's
/// `VadManager`, both on `cpu_only` over the SAME artifact. **Measured worst:
/// `0.000e0`** — bit-identical across all 217 committed chunks
/// (`vad_probabilities_match_fluidaudio_swift_trace`), as expected when both
/// sides feed the same `.mlmodelc` byte-identical f32 windows. Pinned at `1e-4`
/// — a hair of headroom for any future cross-toolchain fp16 rounding, yet an
/// order of magnitude below the ~5e-3 a one-sample context skew moves the
/// output (`tests/model_state.rs`), and 10x below the ±1e-3 probability
/// perturbation the trace-mutation gate injects, so neither can hide under it.
const TRACE_TOL: f64 = 1e-4;

/// One committed Swift chunk: its index, the unpadded sample count fed for it,
/// and FluidAudio's speech probability.
#[derive(Debug)]
struct SwiftChunk {
  chunk_index: usize,
  unpadded_samples: usize,
  probability: f32,
}

/// A parsed, VALIDATED Swift trace golden for one fixture.
#[derive(Debug)]
struct SwiftGolden {
  compute_units: String,
  sample_rate: usize,
  chunk_size: usize,
  context_size: usize,
  state_size: usize,
  model_input_size: usize,
  input_samples: usize,
  input_fnv1a: String,
  chunks: Vec<SwiftChunk>,
}

/// STRICTLY parses a Swift trace golden from its JSON, hard-erroring on ANY
/// malformation before a single fidelity number is compared (the speakerkit
/// strict-loader lesson). Split out of [`load_swift_golden`] so the
/// `strict_loader_*` tests can drive it with synthetic malformed JSON and no
/// model. Guards, in order: every top-level field present and well-typed; a
/// NON-EMPTY `chunks` array; `chunkCount == chunks.len()`; every chunk's
/// `chunkIndex` equal to its position (contiguous `0..n`); every `probability`
/// finite and in `[0, 1]` (the noisy-OR output range); every `unpaddedSamples`
/// in `1..=chunkSize`, and exactly `chunkSize` for every chunk BUT the last
/// (only the final chunk may be short) — a structural pin on the 4096-stride
/// chunking itself.
fn parse_golden(name: &str, v: &serde_json::Value) -> Result<SwiftGolden, String> {
  let str_field = |key: &str| -> Result<String, String> {
    v[key]
      .as_str()
      .map(str::to_string)
      .ok_or_else(|| format!("{name}: `{key}` missing or not a string"))
  };
  let usize_field = |key: &str| -> Result<usize, String> {
    v[key]
      .as_u64()
      .map(|n| n as usize)
      .ok_or_else(|| format!("{name}: `{key}` missing or not a non-negative integer"))
  };

  let chunk_size = usize_field("chunkSize")?;
  let chunk_count = usize_field("chunkCount")?;

  let raw_chunks = v["chunks"]
    .as_array()
    .ok_or_else(|| format!("{name}: `chunks` missing or not an array"))?;
  if raw_chunks.is_empty() {
    return Err(format!(
      "{name}: `chunks` is empty — a trace with no chunks compares nothing"
    ));
  }
  if raw_chunks.len() != chunk_count {
    return Err(format!(
      "{name}: chunkCount {chunk_count} != chunks.len() {} — truncated or padded trace",
      raw_chunks.len()
    ));
  }

  let last = raw_chunks.len() - 1;
  let mut chunks = Vec::with_capacity(raw_chunks.len());
  for (i, c) in raw_chunks.iter().enumerate() {
    let chunk_index = c["chunkIndex"]
      .as_u64()
      .ok_or_else(|| format!("{name}: chunk {i}: `chunkIndex` missing or not an integer"))?
      as usize;
    if chunk_index != i {
      return Err(format!(
        "{name}: chunk at position {i} has chunkIndex {chunk_index} — chunks must be contiguous 0..n"
      ));
    }
    let unpadded_samples = c["unpaddedSamples"]
      .as_u64()
      .ok_or_else(|| format!("{name}: chunk {i}: `unpaddedSamples` missing or not an integer"))?
      as usize;
    if unpadded_samples == 0 || unpadded_samples > chunk_size {
      return Err(format!(
        "{name}: chunk {i}: unpaddedSamples {unpadded_samples} not in 1..={chunk_size}"
      ));
    }
    if i != last && unpadded_samples != chunk_size {
      return Err(format!(
        "{name}: chunk {i} (not the last) has unpaddedSamples {unpadded_samples} != chunkSize \
         {chunk_size} — only the final chunk may be short"
      ));
    }
    // A JSON whole number (`1`) is a valid probability; `as_f64` accepts both.
    let probability = c["probability"]
      .as_f64()
      .ok_or_else(|| format!("{name}: chunk {i}: `probability` missing or not a number"))?;
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
      return Err(format!(
        "{name}: chunk {i}: probability {probability} is not a finite value in [0, 1]"
      ));
    }
    chunks.push(SwiftChunk {
      chunk_index,
      unpadded_samples,
      probability: probability as f32,
    });
  }

  Ok(SwiftGolden {
    compute_units: str_field("computeUnits")?,
    sample_rate: usize_field("sampleRate")?,
    chunk_size,
    context_size: usize_field("contextSize")?,
    state_size: usize_field("stateSize")?,
    model_input_size: usize_field("modelInputSize")?,
    input_samples: usize_field("inputSamples")?,
    input_fnv1a: str_field("inputFnv1a")?,
    chunks,
  })
}

/// Loads and strictly validates the committed Swift golden for `name`.
///
/// # Panics
/// If the file is missing or [`parse_golden`] rejects it — a committed golden
/// is a hard dependency of this gate, not an optional input.
fn load_swift_golden(name: &str) -> SwiftGolden {
  let path = common::golden_swift_dir().join(format!("{name}.json"));
  let bytes = std::fs::read(&path).unwrap_or_else(|e| {
    panic!(
      "read swift golden {}: {e}\n  regenerate: crates/vadkit/tests/swift/regen_goldens.sh",
      path.display()
    )
  });
  let v: serde_json::Value =
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse golden {name}: {e}"));
  parse_golden(name, &v).unwrap_or_else(|e| panic!("malformed golden: {e}"))
}

/// **THE TRACE GATE** (model-gated). For each fixture: prove the input is
/// byte-identical to what the Swift oracle saw (FNV-1a), replay
/// `vadkit::VadModel` over the same 4096-stride chunking on `cpu_only`, and
/// require every per-chunk probability within [`TRACE_TOL`] of FluidAudio's.
///
/// Mutation: perturbing any one committed probability by ±1e-3
/// (`strict_loader` fixtures aside) blows past the `1e-4` bound and turns this
/// red — the inputs reach the failure regime (the campaign rule).
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn vad_probabilities_match_fluidaudio_swift_trace() {
  let mut overall_worst = 0.0f64;
  let mut total_chunks = 0usize;

  for &fixture in GATE_FIXTURES {
    let golden = load_swift_golden(fixture);

    // Placement + geometry the golden was generated with must match ours.
    assert_eq!(
      golden.compute_units, "cpu_only",
      "{fixture}: golden placement"
    );
    assert_eq!(golden.sample_rate, 16_000, "{fixture}: sample rate");
    assert_eq!(golden.chunk_size, CHUNK_SAMPLES, "{fixture}: chunk size");
    assert_eq!(
      golden.context_size, CONTEXT_SAMPLES,
      "{fixture}: context size"
    );
    assert_eq!(golden.state_size, STATE_SIZE, "{fixture}: state size");
    assert_eq!(
      golden.model_input_size, MODEL_INPUT_SAMPLES,
      "{fixture}: window size"
    );

    // Input-identity proof: both sides fed the model the SAME f32 samples.
    let samples = common::load_wav_16k_mono(&common::fixture_wav_path(fixture));
    assert_eq!(
      samples.len(),
      golden.input_samples,
      "{fixture}: sample count differs from the golden"
    );
    assert_eq!(
      common::fnv_hex(common::fnv1a_f32(&samples)),
      golden.input_fnv1a,
      "{fixture}: input FNV-1a mismatch — the gate and the oracle saw different audio"
    );

    // Replay vadkit over the SAME 4096-stride chunking (the short final chunk
    // is included and padded inside predict_chunk, exactly as VadManager does).
    let mut model = VadModel::load_with(
      common::model_path(),
      VadModelOptions::new().with_compute(ComputeUnits::CpuOnly),
    )
    .expect("load vad model");

    let rust_chunks: Vec<&[f32]> = samples.chunks(CHUNK_SAMPLES).collect();
    assert_eq!(
      rust_chunks.len(),
      golden.chunks.len(),
      "{fixture}: chunk count differs from the golden"
    );

    let mut worst = 0.0f64;
    for (chunk, gold) in rust_chunks.iter().zip(&golden.chunks) {
      // Cross-check the chunking agrees sample-for-sample on this chunk.
      assert_eq!(
        chunk.len(),
        gold.unpadded_samples,
        "{fixture}: chunk {}: length {} != golden unpaddedSamples {}",
        gold.chunk_index,
        chunk.len(),
        gold.unpadded_samples
      );
      let p_rust = model
        .predict_chunk(chunk)
        .unwrap_or_else(|e| panic!("{fixture}: chunk {}: {e}", gold.chunk_index));
      let delta = (f64::from(p_rust) - f64::from(gold.probability)).abs();
      assert!(
        delta <= TRACE_TOL,
        "{fixture}: chunk {}: |Δ| {delta:.3e} exceeds {TRACE_TOL:.0e} \
         (vadkit={p_rust}, swift={})",
        gold.chunk_index,
        gold.probability
      );
      worst = worst.max(delta);
    }
    total_chunks += golden.chunks.len();
    overall_worst = overall_worst.max(worst);
    println!(
      "[trace] {fixture}: {} chunks, worst |Δ| {worst:.3e}",
      golden.chunks.len()
    );
  }

  assert!(
    total_chunks >= 40,
    "gate must span >= 40 chunks, got {total_chunks}"
  );
  println!(
    "[trace] {total_chunks} chunks across {} clips, overall worst |Δ| {overall_worst:.3e}",
    GATE_FIXTURES.len()
  );
}

// ── Hermetic strict-loader malformation tests (no model) ───────────────────

/// A minimal well-formed golden `Value` the malformation tests each corrupt in
/// exactly one way — so a passing `parse_golden` on THIS proves the negatives
/// below fail for the reason named, not because the base was already broken.
fn well_formed() -> serde_json::Value {
  serde_json::json!({
    "computeUnits": "cpu_only",
    "sampleRate": 16000,
    "chunkSize": 4096,
    "contextSize": 64,
    "stateSize": 128,
    "modelInputSize": 4160,
    "inputSamples": 8192,
    "inputFnv1a": "0000000000000000",
    "chunkCount": 2,
    "chunks": [
      { "chunkIndex": 0, "unpaddedSamples": 4096, "probability": 1 },
      { "chunkIndex": 1, "unpaddedSamples": 2805, "probability": 0.5 }
    ]
  })
}

#[test]
fn strict_loader_accepts_a_well_formed_golden() {
  let g = parse_golden("wf", &well_formed()).expect("well-formed must parse");
  assert_eq!(g.chunks.len(), 2);
  assert_eq!(g.chunks[0].probability, 1.0);
  assert_eq!(g.chunks[1].unpadded_samples, 2805);
}

#[test]
fn strict_loader_rejects_chunk_count_mismatch() {
  let mut v = well_formed();
  v["chunkCount"] = serde_json::json!(3); // says 3, has 2
  assert!(parse_golden("bad", &v).unwrap_err().contains("chunkCount"));
}

#[test]
fn strict_loader_rejects_empty_chunks() {
  let mut v = well_formed();
  v["chunks"] = serde_json::json!([]);
  v["chunkCount"] = serde_json::json!(0);
  assert!(parse_golden("bad", &v).unwrap_err().contains("empty"));
}

#[test]
fn strict_loader_rejects_non_contiguous_chunk_index() {
  let mut v = well_formed();
  v["chunks"][1]["chunkIndex"] = serde_json::json!(5); // should be 1
  assert!(parse_golden("bad", &v).unwrap_err().contains("contiguous"));
}

#[test]
fn strict_loader_rejects_probability_out_of_range() {
  let mut v = well_formed();
  v["chunks"][0]["probability"] = serde_json::json!(1.5);
  assert!(parse_golden("bad", &v).unwrap_err().contains("[0, 1]"));
}

#[test]
fn strict_loader_rejects_non_finite_probability() {
  let mut v = well_formed();
  // JSON has no NaN literal; a string is simply "not a number" here.
  v["chunks"][0]["probability"] = serde_json::json!("NaN");
  assert!(
    parse_golden("bad", &v)
      .unwrap_err()
      .contains("not a number")
  );
}

#[test]
fn strict_loader_rejects_short_non_final_chunk() {
  let mut v = well_formed();
  v["chunks"][0]["unpaddedSamples"] = serde_json::json!(100); // non-final, must be full
  assert!(
    parse_golden("bad", &v)
      .unwrap_err()
      .contains("only the final chunk may be short")
  );
}

#[test]
fn strict_loader_rejects_oversize_unpadded_samples() {
  let mut v = well_formed();
  v["chunks"][1]["unpaddedSamples"] = serde_json::json!(4097); // > chunkSize
  assert!(parse_golden("bad", &v).unwrap_err().contains("1..="));
}

#[test]
fn strict_loader_rejects_missing_field() {
  let mut v = well_formed();
  v["inputFnv1a"] = serde_json::Value::Null;
  assert!(parse_golden("bad", &v).unwrap_err().contains("inputFnv1a"));
}
