//! The FluidAudio Swift trace gate (design spec §6 model-layer oracle). Pins
//! `coremlit::audio::vad::VadModel`'s per-chunk speech probabilities against committed traces
//! from FluidAudio's OWN `VadManager.process` — the reference implementation of
//! the 256 ms chunking, 64-sample context stitching, repeat-last final-chunk
//! padding and LSTM state carry-forward this crate ports. Both sides run the
//! SAME `Models/vadkit` artifact on `ComputeUnits::CpuOnly`, so the port is
//! held to a near-bit-exact bound (measured, then pinned two-sided;
//! [`TRACE_TOL`]).
//!
//! Regenerate the traces with `tests/vad/swift/regen_goldens.sh`.
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
//!
//! # The bound is a SAME-HOST-CLASS bound (host-aware gate)
//!
//! CoreML `CpuOnly` kernels ship with the OS (BNNS/Accelerate/Espresso) and are
//! not contracted to produce identical floats across macOS builds or chip
//! generations — cross-host drift up to ~1.2e-2 has been measured on this very
//! trace after LSTM recurrence amplification (#36). Each golden therefore records
//! the `generationHost` it was dumped on; the gate enforces [`TRACE_TOL`]
//! unchanged when the running host-class matches, and on a mismatch fails with a
//! regenerate-a-same-host-oracle diagnosis instead of blaming the port. The bound
//! itself is NEVER widened for cross-host runs — a wide band would blind the gate
//! to the stitching regressions it exists to catch. Goldens predating the
//! `generationHost` field get the tight bounds with an ambiguity note on failure.

mod common;

use coremlit::{
  ComputeUnits,
  audio::vad::{
    CHUNK_SAMPLES, CONTEXT_SAMPLES, MODEL_INPUT_SAMPLES, STATE_SIZE, VadModel, VadModelOptions,
  },
};

/// The two committed fixtures (spec §6: real speech, one multi-speaker, ≥ 2
/// clips, ≥ 40 chunks). `02_pyannote_sample` → 118 chunks, `07_yuhewei_dongbei_
/// english` → 99 chunks (its short final chunk exercises the repeat-last
/// padding path): 217 chunks total.
const GATE_FIXTURES: &[&str] = &["02_pyannote_sample", "07_yuhewei_dongbei_english"];

/// Worst tolerated per-chunk |Δ| between `coremlit::audio::vad::VadModel` and FluidAudio's
/// `VadManager`, both on `cpu_only` over the SAME artifact. **Measured worst:
/// `0.000e0`** — bit-identical across all 217 committed chunks
/// (`vad_probabilities_match_fluidaudio_swift_trace`), as expected when both
/// sides feed the same `.mlmodelc` byte-identical f32 windows. Pinned at `1e-4`
/// — a hair of headroom for any future cross-toolchain fp16 rounding, yet an
/// order of magnitude below the ~5e-3 a one-sample context skew moves the
/// output (`tests/model_state.rs`), and 10x below the ±1e-3 probability
/// perturbation the trace-mutation gate injects, so neither can hide under it.
/// The bound is enforced only against a same-host-class golden (see the module
/// doc's host-aware-gate section); a host-class mismatch fails toward
/// regeneration, never toward a wider tolerance.
const TRACE_TOL: f64 = 1e-4;

/// The regeneration script the host-aware gate points a mismatched (or legacy,
/// unstamped) host at. Same-host regeneration IS the port-correctness test off
/// the golden's generation host, so the diagnosis names it rather than blaming
/// the port or inviting a widened tolerance.
const VAD_REGEN_SCRIPT: &str = "crates/coremlit/tests/vad/swift/regen_goldens.sh";

/// The pinned generator identity every committed golden must record: the Swift
/// dumper that produced it (`DumpVadTraces.swift`). A golden whose `generator`
/// differs was not produced by this crate's oracle harness.
const GENERATOR: &str = "crates/coremlit/tests/vad/swift/Tests/VadTraceDump/DumpVadTraces.swift";

/// The pinned FluidAudio revision the committed goldens were generated against
/// (`regen_goldens.sh` stamps `FLUIDAUDIO_REVISION` into each golden). The
/// oracle's semantics are only guaranteed at this exact revision, so a golden
/// regenerated from a different (or dirty/unknown) FluidAudio checkout is
/// rejected rather than silently trusted.
const FLUIDAUDIO_REVISION: &str = "1a2da18";

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
  /// The host-class the golden was generated on, or `None` for a legacy golden
  /// that predates host provenance. FORM-validated by
  /// [`common::HostClass::from_golden`]; the host MATCH is the model-gated
  /// gate's job (`check_host_class`), never the parser's.
  generation_host: Option<common::HostClass>,
}

/// STRICTLY parses a Swift trace golden from its JSON, hard-erroring on ANY
/// malformation before a single fidelity number is compared (the speakerkit
/// strict-loader lesson). Split out of [`load_swift_golden`] so the
/// `strict_loader_*` tests can drive it with synthetic malformed JSON and no
/// model. Guards, in order: every top-level field present and well-typed;
/// ORACLE PROVENANCE — `fixture` matches the fixture being loaded, `generator`
/// is the pinned [`GENERATOR`] dumper, `fluidAudioRevision` is the pinned
/// [`FLUIDAUDIO_REVISION`], and `determinismVerified` (the generator's `Bool?`:
/// present-and-true on the first fixture where determinism was measured, absent
/// on the rest) is never present-but-false; a NON-EMPTY `chunks` array;
/// `chunkCount == chunks.len()`; every chunk's `chunkIndex` equal to its
/// position (contiguous `0..n`); every `probability` finite and in `[0, 1]`
/// (the noisy-OR output range); every `unpaddedSamples` in `1..=chunkSize`, and
/// exactly `chunkSize` for every chunk BUT the last (only the final chunk may
/// be short) — a structural pin on the 4096-stride chunking itself.
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

  // Oracle provenance: this golden must be the RIGHT fixture's, from the pinned
  // dumper, against the pinned FluidAudio revision — otherwise the fidelity
  // numbers below are being trusted from an unknown source.
  let fixture = str_field("fixture")?;
  if fixture != name {
    return Err(format!(
      "{name}: `fixture` is {fixture:?}, expected {name:?} — golden loaded for the wrong fixture"
    ));
  }
  let generator = str_field("generator")?;
  if generator != GENERATOR {
    return Err(format!(
      "{name}: `generator` is {generator:?}, expected {GENERATOR:?} — golden not from the pinned dumper"
    ));
  }
  let fluid_audio_revision = str_field("fluidAudioRevision")?;
  if fluid_audio_revision != FLUIDAUDIO_REVISION {
    return Err(format!(
      "{name}: `fluidAudioRevision` is {fluid_audio_revision:?}, expected {FLUIDAUDIO_REVISION:?} \
       — golden regenerated from an unintended FluidAudio revision"
    ));
  }
  // `determinismVerified` is the generator's `Bool?`: present-and-true on the
  // first fixture (where reproducibility was measured), absent (`null`) on the
  // rest. Tolerate absence; reject a golden that records it present-but-not-true
  // (the oracle's own reproducibility check did not pass).
  match v.get("determinismVerified") {
    None | Some(serde_json::Value::Null) | Some(serde_json::Value::Bool(true)) => {}
    Some(other) => {
      return Err(format!(
        "{name}: `determinismVerified` is {other} — present but not `true`; the oracle's own \
         reproducibility check did not pass"
      ));
    }
  }

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
    // FORM only — parses `generationHost` if present, tolerates its absence
    // (legacy golden). The host MATCH happens in the model-gated gate, so the
    // strict-loader and committed-golden hermetic tests parse on every host.
    generation_host: common::HostClass::from_golden(name, v)?,
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
      "read swift golden {}: {e}\n  regenerate: crates/coremlit/tests/vad/swift/regen_goldens.sh",
      path.display()
    )
  });
  let v: serde_json::Value =
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse golden {name}: {e}"));
  parse_golden(name, &v).unwrap_or_else(|e| panic!("malformed golden: {e}"))
}

/// **THE TRACE GATE** (model-gated). For each fixture: prove the input is
/// byte-identical to what the Swift oracle saw (FNV-1a), replay
/// `coremlit::audio::vad::VadModel` over the same 4096-stride chunking on `cpu_only`, and
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

  // The running host-class, read once (all fixtures share this machine).
  let running = common::HostClass::running();

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

    // Host-class gate — the LAST precondition before the first CoreML-produced
    // number. The fixture/generator/revision/geometry/input guards above are
    // host-independent harness-validity checks and must keep failing first (a
    // harness bug must never be reported as a host mismatch); only once they
    // pass does host-class attribution become the question. A recorded-but-
    // different host panics here with the regenerate diagnosis before any
    // probability is compared; a legacy (unstamped) golden yields the ambiguity
    // note appended to the fidelity failure below.
    let host_note = match common::check_host_class(
      fixture,
      golden.generation_host.as_ref(),
      &running,
      VAD_REGEN_SCRIPT,
    ) {
      Ok(common::HostVerdict::Match) => {
        println!("[host] {fixture}: golden generationHost matches this host: {running}");
        String::new()
      }
      Ok(common::HostVerdict::LegacyUnknown) => {
        println!(
          "[host] {fixture}: golden has no generationHost (pre-host-provenance); tight \
           bounds enforced — a failure would be ambiguous between port defect and host \
           drift"
        );
        common::legacy_failure_note(VAD_REGEN_SCRIPT)
      }
      Err(diagnosis) => panic!("{diagnosis}"),
    };

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
         (vadkit={p_rust}, swift={}){host_note}",
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
    "fixture": "wf",
    "generator": "crates/coremlit/tests/vad/swift/Tests/VadTraceDump/DumpVadTraces.swift",
    "fluidAudioRevision": "1a2da18",
    // A deliberately UNREAL host: it can never equal a running host-class, so
    // any accidental parse-time host enforcement would fail the hermetic suite
    // on every machine (the D4 trap guard).
    "generationHost": {
      "osBuild": "99Z999",
      "osProductVersion": "99.9",
      "chip": "Synthetic Chip",
      "arch": "arm64"
    },
    "determinismVerified": true,
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
  assert_eq!(
    g.generation_host,
    Some(common::HostClass {
      os_build: "99Z999".to_string(),
      os_product_version: "99.9".to_string(),
      chip: "Synthetic Chip".to_string(),
      arch: "arm64".to_string(),
    })
  );
}

#[test]
fn strict_loader_rejects_chunk_count_mismatch() {
  let mut v = well_formed();
  v["chunkCount"] = serde_json::json!(3); // says 3, has 2
  assert!(parse_golden("wf", &v).unwrap_err().contains("chunkCount"));
}

#[test]
fn strict_loader_rejects_empty_chunks() {
  let mut v = well_formed();
  v["chunks"] = serde_json::json!([]);
  v["chunkCount"] = serde_json::json!(0);
  assert!(parse_golden("wf", &v).unwrap_err().contains("empty"));
}

#[test]
fn strict_loader_rejects_non_contiguous_chunk_index() {
  let mut v = well_formed();
  v["chunks"][1]["chunkIndex"] = serde_json::json!(5); // should be 1
  assert!(parse_golden("wf", &v).unwrap_err().contains("contiguous"));
}

#[test]
fn strict_loader_rejects_probability_out_of_range() {
  let mut v = well_formed();
  v["chunks"][0]["probability"] = serde_json::json!(1.5);
  assert!(parse_golden("wf", &v).unwrap_err().contains("[0, 1]"));
}

#[test]
fn strict_loader_rejects_non_finite_probability() {
  let mut v = well_formed();
  // JSON has no NaN literal; a string is simply "not a number" here.
  v["chunks"][0]["probability"] = serde_json::json!("NaN");
  assert!(parse_golden("wf", &v).unwrap_err().contains("not a number"));
}

#[test]
fn strict_loader_rejects_short_non_final_chunk() {
  let mut v = well_formed();
  v["chunks"][0]["unpaddedSamples"] = serde_json::json!(100); // non-final, must be full
  assert!(
    parse_golden("wf", &v)
      .unwrap_err()
      .contains("only the final chunk may be short")
  );
}

#[test]
fn strict_loader_rejects_oversize_unpadded_samples() {
  let mut v = well_formed();
  v["chunks"][1]["unpaddedSamples"] = serde_json::json!(4097); // > chunkSize
  assert!(parse_golden("wf", &v).unwrap_err().contains("1..="));
}

#[test]
fn strict_loader_rejects_missing_field() {
  let mut v = well_formed();
  v["inputFnv1a"] = serde_json::Value::Null;
  assert!(parse_golden("wf", &v).unwrap_err().contains("inputFnv1a"));
}

#[test]
fn strict_loader_rejects_wrong_fixture() {
  // The golden must be the one for the fixture being loaded: a copy-pasted or
  // misnamed golden (right shape, wrong clip) is rejected.
  let v = well_formed(); // its `fixture` is "wf"
  assert!(
    parse_golden("02_pyannote_sample", &v)
      .unwrap_err()
      .contains("wrong fixture")
  );
}

#[test]
fn strict_loader_rejects_wrong_generator() {
  let mut v = well_formed();
  v["generator"] = serde_json::json!("some/other/tool.py");
  assert!(
    parse_golden("wf", &v)
      .unwrap_err()
      .contains("pinned dumper")
  );
}

#[test]
fn strict_loader_rejects_wrong_fluidaudio_revision() {
  // Regenerated from an unintended (or dirty/unknown) FluidAudio checkout.
  let mut v = well_formed();
  v["fluidAudioRevision"] = serde_json::json!("deadbee");
  assert!(
    parse_golden("wf", &v)
      .unwrap_err()
      .contains("FluidAudio revision")
  );
}

#[test]
fn strict_loader_rejects_determinism_verified_false() {
  // Absent is tolerated (non-first fixtures), but a golden that records the
  // oracle's own reproducibility check as FAILED must be rejected.
  let mut v = well_formed();
  v["determinismVerified"] = serde_json::json!(false);
  assert!(
    parse_golden("wf", &v)
      .unwrap_err()
      .contains("determinismVerified")
  );
}

#[test]
fn strict_loader_tolerates_absent_determinism_verified() {
  // The generator emits `determinismVerified` only on the FIRST fixture and
  // leaves it `null`/absent on the rest (e.g. `07_yuhewei_dongbei_english`), so
  // absence must parse — only a present-but-not-true value is a violation.
  let mut absent = well_formed();
  absent["determinismVerified"] = serde_json::Value::Null;
  parse_golden("wf", &absent).expect("absent determinismVerified must be tolerated");

  let mut missing = well_formed();
  missing
    .as_object_mut()
    .unwrap()
    .remove("determinismVerified");
  parse_golden("wf", &missing).expect("missing determinismVerified must be tolerated");
}

#[test]
fn strict_loader_tolerates_absent_generation_host() {
  // Mirror the determinismVerified absent-tolerance: both an explicit `null`
  // and a removed key parse to `generation_host == None` (a legacy golden), so
  // committed unstamped goldens keep loading on every host.
  let mut null = well_formed();
  null["generationHost"] = serde_json::Value::Null;
  assert_eq!(
    parse_golden("wf", &null)
      .expect("null generationHost must be tolerated")
      .generation_host,
    None
  );

  let mut missing = well_formed();
  missing.as_object_mut().unwrap().remove("generationHost");
  assert_eq!(
    parse_golden("wf", &missing)
      .expect("absent generationHost must be tolerated")
      .generation_host,
    None
  );
}

#[test]
fn strict_loader_rejects_malformed_generation_host() {
  // A string, not an object.
  let mut a_string = well_formed();
  a_string["generationHost"] = serde_json::json!("Apple M1");
  assert!(
    parse_golden("wf", &a_string)
      .unwrap_err()
      .contains("generationHost")
  );

  // An object missing `chip`.
  let mut missing_chip = well_formed();
  missing_chip["generationHost"]
    .as_object_mut()
    .unwrap()
    .remove("chip");
  assert!(
    parse_golden("wf", &missing_chip)
      .unwrap_err()
      .contains("generationHost")
  );

  // An object with an empty `osBuild`.
  let mut empty_build = well_formed();
  empty_build["generationHost"]["osBuild"] = serde_json::json!("");
  assert!(
    parse_golden("wf", &empty_build)
      .unwrap_err()
      .contains("generationHost")
  );
}

// ── Hermetic host-class gate tests (no model; synthetic host-classes) ───────

/// A synthetic host-class for the pure-predicate tests below.
fn synthetic_host() -> common::HostClass {
  common::HostClass {
    os_build: "24F74".to_string(),
    os_product_version: "15.5".to_string(),
    chip: "Apple M1".to_string(),
    arch: "arm64".to_string(),
  }
}

#[test]
fn host_gate_matches_identical_host_class() {
  let h = synthetic_host();
  assert_eq!(
    common::check_host_class("wf", Some(&h), &h, VAD_REGEN_SCRIPT),
    Ok(common::HostVerdict::Match)
  );
}

#[test]
fn host_gate_mismatch_diagnoses_regeneration_not_port_defect() {
  let base = synthetic_host();
  // One pair differs in osBuild only; the other in chip only.
  let other_build = common::HostClass {
    os_build: "24G84".to_string(),
    ..base.clone()
  };
  let other_chip = common::HostClass {
    chip: "Apple M1 Pro".to_string(),
    ..base.clone()
  };
  for (recorded, running) in [(&base, &other_build), (&base, &other_chip)] {
    let diagnosis = common::check_host_class("wf", Some(recorded), running, VAD_REGEN_SCRIPT)
      .expect_err("a differing host-class must be diagnosed, not matched");
    let golden_render = recorded.to_string();
    let running_render = running.to_string();
    assert!(
      diagnosis.contains(golden_render.as_str()),
      "diagnosis must name the golden host: {diagnosis}"
    );
    assert!(
      diagnosis.contains(running_render.as_str()),
      "diagnosis must name the running host: {diagnosis}"
    );
    assert!(
      diagnosis.contains("regen_goldens.sh"),
      "diagnosis must point at regeneration: {diagnosis}"
    );
    assert!(
      diagnosis.contains("NOT evidence of a port defect"),
      "diagnosis must not blame the port: {diagnosis}"
    );
  }
}

#[test]
fn host_gate_treats_unstamped_golden_as_legacy_ambiguous() {
  let running = synthetic_host();
  assert_eq!(
    common::check_host_class("wf", None, &running, VAD_REGEN_SCRIPT),
    Ok(common::HostVerdict::LegacyUnknown)
  );
  let note = common::legacy_failure_note(VAD_REGEN_SCRIPT);
  assert!(
    note.contains("AMBIGUOUS"),
    "legacy note must flag ambiguity"
  );
  assert!(
    note.contains("regen_goldens.sh"),
    "legacy note must point at regeneration"
  );
}

#[test]
fn running_host_class_is_well_formed() {
  // The only hermetic test that shells out to sysctl; CI is macos-15 on every
  // job, so this is safe and catches a sysctl-key typo without any model.
  let h = common::HostClass::running();
  assert!(!h.os_build.is_empty(), "osBuild empty");
  assert!(!h.os_product_version.is_empty(), "osProductVersion empty");
  assert!(!h.chip.is_empty(), "chip empty");
  assert!(
    h.arch == "arm64" || h.arch == "x86_64",
    "arch {:?} is neither arm64 nor x86_64",
    h.arch
  );
}

/// Hermetic: every committed VAD golden parses through the strict
/// [`load_swift_golden`] loader (reads committed JSON only, no model). Pins the
/// interim legacy behavior — goldens without `generationHost` still load, on
/// every host — and is the designated flip-site for the post-regen
/// host-presence assert (the owner-gated regen runbook). It also gives this
/// suite the committed-golden hermetic coverage the speaker suite already has.
#[test]
fn committed_goldens_parse_through_the_strict_loader() {
  for &fixture in GATE_FIXTURES {
    let golden = load_swift_golden(fixture);
    assert_eq!(golden.chunk_size, CHUNK_SAMPLES, "{fixture}: chunk size");
    assert!(!golden.chunks.is_empty(), "{fixture}: golden has no chunks");
  }
}
