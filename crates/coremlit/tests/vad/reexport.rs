//! The re-export layer (design spec §2-§4): proves vadkit's public detection
//! surface is zuoer's, wired over the CoreML backend, with ZERO segmentation
//! logic authored in vadkit.
//!
//! Four gates:
//!
//! 1. **`src_authors_no_detection_logic`** (hermetic) — the no-duplication
//!    proof. Greps every `crates/coremlit/src/audio/vad/**/*.rs` file for the
//!    zuoer-segmenter vocabulary (thresholding, hysteresis, `min_speech`/
//!    `min_silence`/`speech_pad`, driving/constructing a segmenter). vadkit's
//!    `src/` contains NONE of it; a re-implementation of any segment-assembly
//!    step drags at least one token in and turns this red.
//! 2. **The mock-backend scenarios** (hermetic) — zuoer's own 4096-geometry
//!    detector-test scenarios, replayed over a CoreML-SHAPED mock
//!    ([`MockVadBackend`]: `frame_hop() == 4096`, the same buffer-and-
//!    zero-pad-the-tail windowing [`CoreMlBackend`] implements, one canned
//!    probability per completed frame) driven through the re-exported
//!    [`coremlit::audio::vad::detect_speech_with`]. Same inputs, same pinned
//!    segment boundaries zuoer pins internally — so the re-export provably
//!    drives zuoer's real segmenter, not a copy. Plus the error-bridge shape
//!    an out-of-tree backend uses.
//! 3. **The trailing-frame gates** — the seam's end-of-stream policy, which
//!    moved from the detector into the backend when zuoer replaced the
//!    batch `predict` seam with the push/`finish` one. Hermetically over the
//!    mock ([`finish_zero_pads_a_partial_trailing_frame`],
//!    [`finish_on_an_exact_frame_boundary_emits_nothing`],
//!    [`push_blocking_does_not_change_the_frame_stream`]) and, model-gated,
//!    over the real [`CoreMlBackend`]
//!    ([`coreml_backend_frames_match_a_hand_chunked_zero_padded_reference`]),
//!    which pins vadkit's probability stream against a hand-chunked
//!    zero-padded reference driven straight through [`VadModel`] — the exact
//!    frame sequence the pre-zuoer detector produced.
//! 4. **`detect_speech_on_real_audio_is_pinned`** (model-gated) — the end-to-
//!    end path: [`coremlit::audio::vad::detect_speech`] over a real [`CoreMlBackend`] on a
//!    committed fixture, segment starts/ends pinned two-sided.

mod common;

use coremlit::audio::vad::{CHUNK_SAMPLES, detect_speech_with};
use zuoer::{SampleRate, SpeechOptions, VadBackend};

// ── 1. No-duplication grep gate ─────────────────────────────────────────────

/// Silero-segmenter identifiers and thresholding vocabulary that would appear
/// in vadkit's `src/` ONLY if it re-implemented some part of the segment
/// assembly the `zuoer` crate single-homes (spec §2-§3). Re-exporting the
/// segmenter types by name (`SpeechSegmenter`, `SpeechSegment` — no `::new`)
/// does not match any of these, so the gate stays green on a pure re-export and
/// red on any authored detection logic.
const FORBIDDEN_DETECTION_TOKENS: &[&str] = &[
  "push_probability",
  "SpeechSegment::new",
  "SpeechSegmenter::new",
  "start_threshold",
  "end_threshold",
  "min_silence",
  "min_speech",
  "speech_pad",
  "tentative_end",
  "hysteresis",
  ">= threshold",
  "> threshold",
];

/// Structural float-comparison shapes a hand-rolled probability threshold would
/// introduce — `p >= 0.5`, `p > 0.35`, `p < 0.35` — which are harder to alias
/// than the vocabulary above (they catch the fence's `p >= 0.5` bypass that
/// [`FORBIDDEN_DETECTION_TOKENS`] misses). The leading space keeps `->` return
/// arrows from matching. Forbidden in vadkit's backend / re-export production
/// layer, where any float comparison IS authored thresholding; the model module
/// (its finite / contract checks) and test files (probability assertions)
/// legitimately compare floats and are allowlisted by
/// [`may_compare_floats`].
const FORBIDDEN_FLOAT_COMPARISONS: &[&str] = &[" >= 0.", " > 0.", " <= 0.", " < 0."];

/// Whether `rel` (a `vadkit/src`-relative path) is allowed to compare floats:
/// the model module's finite / contract checks and any test file's probability
/// assertions. Everywhere else — `backend.rs`, `lib.rs`, `error/mod.rs`, the
/// re-export/backend layer — a float comparison would be smuggled-in detection
/// logic and is forbidden by [`FORBIDDEN_FLOAT_COMPARISONS`].
fn may_compare_floats(rel: &std::path::Path) -> bool {
  let s = rel.to_string_lossy();
  s.contains("model") || s.ends_with("tests.rs")
}

/// Collects every `.rs` file under `dir`, recursively.
fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
  for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
    let path = entry.expect("dir entry").path();
    if path.is_dir() {
      rust_files(&path, out);
    } else if path.extension().is_some_and(|ext| ext == "rs") {
      out.push(path);
    }
  }
}

/// **THE NO-DUPLICATION GATE** (spec §2-§3, plan T5): vadkit's `src/` authors no
/// thresholding / hysteresis / segment-assembly logic — all of it stays
/// single-homed in `zuoer`, and vadkit only implements the backend seam and
/// re-exports the detector surface. Two layers per source file: the
/// [`FORBIDDEN_DETECTION_TOKENS`] vocabulary, plus the
/// [`FORBIDDEN_FLOAT_COMPARISONS`] structural check (a probability threshold
/// outside the model module — harder to alias than the vocabulary). This grep
/// is the FIRST, cheap layer; [`reexport_detect_speech_with_is_bit_identical_to_zuoer`]
/// is the complementary equivalence proof that catches an aliased
/// re-implementation this grep could still miss.
#[test]
fn src_authors_no_detection_logic() {
  let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/audio/vad");
  let mut files = Vec::new();
  rust_files(&src, &mut files);
  assert!(
    files.len() >= 4,
    "expected to scan vadkit's src tree, found only {} files under {src:?}",
    files.len()
  );

  let mut violations = Vec::new();
  for file in &files {
    let rel = file.strip_prefix(&src).unwrap_or(file);
    let floats_allowed = may_compare_floats(rel);
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("read {file:?}: {e}"));
    for (lineno, line) in text.lines().enumerate() {
      // Scan CODE only, not prose: the claim is that vadkit AUTHORS no
      // segmentation logic, so a doc comment that DESCRIBES what zuoer owns
      // (as this crate's own module docs do) is not a violation. Everything
      // from the first `//` to end-of-line is comment text and is dropped
      // (`///` / `//!` doc lines drop whole; a trailing `// ...` drops its
      // tail). vadkit's `src` has no string literals carrying these tokens, so
      // this cannot mask a real re-implementation.
      let code = line.split("//").next().unwrap_or("");
      for token in FORBIDDEN_DETECTION_TOKENS {
        if code.contains(token) {
          violations.push(format!(
            "{}:{} authors segmentation logic (token `{token}`): {}",
            rel.display(),
            lineno + 1,
            line.trim(),
          ));
        }
      }
      // Second layer: a float comparison outside the model module is a
      // hand-rolled probability threshold the vocabulary list can miss.
      if !floats_allowed {
        for token in FORBIDDEN_FLOAT_COMPARISONS {
          if code.contains(token) {
            violations.push(format!(
              "{}:{} compares a float outside the model module (token `{token}`) — a \
               hand-rolled probability threshold belongs single-homed in `zuoer`: {}",
              rel.display(),
              lineno + 1,
              line.trim(),
            ));
          }
        }
      }
    }
  }

  assert!(
    violations.is_empty(),
    "vadkit/src must author NO detection logic — it lives single-homed in \
     `zuoer` (spec §2-§3). Found {} violation(s):\n{}",
    violations.len(),
    violations.join("\n"),
  );
}

// ── 2. CoreML-shaped mock backend + zuoer's detector-test scenarios ──────────

/// A backend error distinct from `vadkit`'s own [`coremlit::audio::vad::InferError`], present
/// to exercise the out-of-tree error bridge the [`VadBackend::Error`] contract
/// prescribes — the same shape [`coremlit::audio::vad::CoreMlBackend`] uses for real
/// (`impl From<TheirError> for zuoer::Error` wrapping in
/// [`zuoer::Error::Backend`]).
#[derive(Debug)]
struct MockError(&'static str);

impl std::fmt::Display for MockError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.0)
  }
}

impl std::error::Error for MockError {}

impl From<MockError> for zuoer::Error {
  fn from(error: MockError) -> Self {
    zuoer::Error::Backend(Box::new(error))
  }
}

/// A [`VadBackend`] shaped like [`coremlit::audio::vad::CoreMlBackend`] — 4096-sample
/// (256 ms) frames at 16 kHz, the same buffer-across-`push` /
/// zero-pad-the-tail-on-`finish` windowing, an inert "model" returning one
/// canned probability per completed frame. It authors no detection logic; it
/// exists to replay zuoer's detector scenarios over the re-exported
/// [`detect_speech_with`] at vadkit's exact geometry, proving the re-export
/// drives zuoer's real segmenter.
struct MockVadBackend {
  probabilities: Vec<f32>,
  cursor: usize,
  fail_at: Option<usize>,
  /// PCM that does not yet complete a frame — [`CoreMlBackend`]'s buffer,
  /// mirrored so the mock's frame stream is the production one.
  pending: Vec<f32>,
  /// Every frame the mock "ran", in order — the record the trailing-frame
  /// gates read to see WHAT was handed to the model, not just how many times.
  frames: Vec<Vec<f32>>,
}

impl MockVadBackend {
  fn new(probabilities: Vec<f32>) -> Self {
    Self {
      probabilities,
      cursor: 0,
      fail_at: None,
      pending: Vec::new(),
      frames: Vec::new(),
    }
  }

  fn failing_at(index: usize, probabilities: Vec<f32>) -> Self {
    Self {
      fail_at: Some(index),
      ..Self::new(probabilities)
    }
  }

  /// Records one completed frame and returns its canned probability — the
  /// mock's stand-in for `VadModel::predict_chunk`.
  fn run_frame(&mut self, frame: &[f32]) -> Result<f32, MockError> {
    assert_eq!(
      frame.len(),
      CHUNK_SAMPLES,
      "a CoreML-shaped backend must run exactly frame_hop samples per frame"
    );
    if self.fail_at == Some(self.cursor) {
      return Err(MockError("mock predict failure"));
    }
    self.frames.push(frame.to_vec());
    let probability = self.probabilities.get(self.cursor).copied().unwrap_or(0.0);
    self.cursor += 1;
    Ok(probability)
  }
}

impl VadBackend for MockVadBackend {
  type Error = MockError;

  fn frame_hop(&self) -> usize {
    CHUNK_SAMPLES
  }

  fn sample_rate(&self) -> SampleRate {
    SampleRate::Rate16k
  }

  fn push(&mut self, samples: &[f32], sink: &mut dyn FnMut(f32)) -> Result<(), MockError> {
    let mut rest = samples;
    if !self.pending.is_empty() {
      let take = (CHUNK_SAMPLES - self.pending.len()).min(rest.len());
      self.pending.extend_from_slice(&rest[..take]);
      rest = &rest[take..];
      if self.pending.len() < CHUNK_SAMPLES {
        return Ok(());
      }
      let frame = std::mem::take(&mut self.pending);
      let probability = self.run_frame(&frame)?;
      sink(probability);
    }
    let (frames, remainder) = rest.as_chunks::<CHUNK_SAMPLES>();
    for frame in frames {
      let probability = self.run_frame(frame)?;
      sink(probability);
    }
    self.pending.extend_from_slice(remainder);
    Ok(())
  }

  fn finish(&mut self, sink: &mut dyn FnMut(f32)) -> Result<(), MockError> {
    if self.pending.is_empty() {
      return Ok(());
    }
    let mut frame = std::mem::take(&mut self.pending);
    frame.resize(CHUNK_SAMPLES, 0.0);
    let probability = self.run_frame(&frame)?;
    sink(probability);
    Ok(())
  }

  fn reset(&mut self) {
    self.cursor = 0;
    self.pending.clear();
    self.frames.clear();
  }
}

#[test]
fn reexport_closes_after_two_256ms_low_frames() {
  // zuoer's `mock_geometry_closes_after_two_256ms_low_frames`, replayed over
  // the re-export: three speech frames then two 256 ms low frames close one
  // segment. The default `min_silence_duration_ms = 100` (1600 samples) is
  // crossed on the SECOND low frame (the silence counter is read before the
  // frame is consumed), so the segment closes at the silence-start sample
  // 3 * 4096, plus 30 ms (480) speech_pad.
  let mut backend = MockVadBackend::new(vec![0.9, 0.9, 0.9, 0.0, 0.0]);
  let samples = vec![0.0_f32; 5 * CHUNK_SAMPLES];
  let segments =
    detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");

  assert_eq!(
    segments.len(),
    1,
    "two 256 ms low frames must close one segment"
  );
  assert_eq!(segments[0].start_sample(), 0);
  assert_eq!(segments[0].end_sample(), 3 * CHUNK_SAMPLES as u64 + 480);
  assert_eq!(
    backend.cursor, 5,
    "every frame consumed at the 4096 geometry"
  );
}

#[test]
fn reexport_holds_open_through_one_256ms_low_frame() {
  // zuoer's `mock_geometry_holds_open_through_one_256ms_low_frame`: a single
  // 256 ms low frame only establishes the silence start (counter 0 < 1600), so
  // no segment closes mid-stream; the open segment is emitted by the end-of-
  // stream flush, spanning to the raw current sample with no trailing pad.
  let mut backend = MockVadBackend::new(vec![0.9, 0.9, 0.9, 0.0]);
  let samples = vec![0.0_f32; 4 * CHUNK_SAMPLES];
  let segments =
    detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");

  assert_eq!(
    segments.len(),
    1,
    "one 256 ms low frame must not close the segment"
  );
  assert_eq!(segments[0].start_sample(), 0);
  assert_eq!(segments[0].end_sample(), 4 * CHUNK_SAMPLES as u64);
}

#[test]
fn reexport_bridges_backend_error_through_backend_variant() {
  // A backend failure must reach the caller of the re-exported
  // `detect_speech_with` through the transparent `zuoer::Error::Backend`
  // variant, delegating its `Display` to the wrapped error — the exact bridge
  // `CoreMlBackend`'s `InferError` uses (proven to compile; here proven to
  // propagate).
  let mut backend = MockVadBackend::failing_at(1, vec![0.9, 0.9, 0.9]);
  let samples = vec![0.0_f32; 3 * CHUNK_SAMPLES];
  let error = detect_speech_with(&mut backend, &samples, SpeechOptions::default())
    .expect_err("backend failure must propagate");
  assert!(
    matches!(error, zuoer::Error::Backend(_)),
    "backend error must bridge through zuoer::Error::Backend, got {error:?}"
  );
  assert_eq!(error.to_string(), "mock predict failure");
}

/// **THE EQUIVALENCE GATE** (hermetic, spec §2-§3): the behavioural complement
/// to the no-duplication grep. Drives the SAME scripted inputs through vadkit's
/// re-exported [`detect_speech_with`] and zuoer's OWN
/// [`zuoer::detect_speech_with`], requiring the segment vectors bit-for-bit
/// identical. It pins the re-export BY BEHAVIOUR, not vocabulary: today
/// `coremlit::audio::vad::detect_speech_with` IS `zuoer::detect_speech_with` (a `pub use`),
/// so they agree trivially — but if a future change replaced the re-export with
/// a hand-rolled threshold loop in vadkit (e.g. `p >= 0.5` plus an aliased
/// `S::new(..)` that the grep's vocabulary could miss), the two paths would
/// diverge and this turns red.
///
/// Mutation: shadow the `detect_speech_with` re-export in `src/audio/vad/mod.rs`
/// with any locally-authored function → these comparisons go red.
#[test]
fn reexport_detect_speech_with_is_bit_identical_to_zuoer() {
  let scenarios: &[Vec<f32>] = &[
    vec![0.9, 0.9, 0.9, 0.0, 0.0],           // closes one segment
    vec![0.9, 0.9, 0.9, 0.0],                // holds open to end-of-stream
    vec![0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.0], // starts mid-stream
    vec![0.0; 6],                            // all silence
    vec![0.9; 6],                            // all speech
    vec![0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9], // alternating
  ];
  for probs in scenarios {
    let samples = vec![0.0_f32; probs.len() * CHUNK_SAMPLES];
    let mut backend_vadkit = MockVadBackend::new(probs.clone());
    let mut backend_zuoer = MockVadBackend::new(probs.clone());
    let via_vadkit = detect_speech_with(&mut backend_vadkit, &samples, SpeechOptions::default())
      .expect("vadkit re-export");
    let via_zuoer =
      zuoer::detect_speech_with(&mut backend_zuoer, &samples, SpeechOptions::default())
        .expect("zuoer direct");
    assert_eq!(
      via_vadkit, via_zuoer,
      "coremlit::audio::vad::detect_speech_with must equal zuoer::detect_speech_with bit-for-bit on {probs:?}"
    );
  }
}

// ── 3. The trailing-frame / input-windowing policy the backend now owns ──────

/// Collects every probability a backend emits over one full stream —
/// `push(samples)` then `finish()` — the exact two-call sequence
/// [`detect_speech_with`] drives.
fn stream_probabilities<B: VadBackend>(backend: &mut B, samples: &[f32]) -> Vec<f32>
where
  B::Error: std::fmt::Debug,
{
  let mut probabilities = Vec::new();
  backend
    .push(samples, &mut |p| probabilities.push(p))
    .expect("push");
  backend
    .finish(&mut |p| probabilities.push(p))
    .expect("finish");
  probabilities
}

/// **THE TRAILING-FRAME GATE** (hermetic). Under zuoer's push-based seam the
/// BACKEND owns the end-of-stream policy that the pre-zuoer detector applied on
/// every backend's behalf: zero-pad a non-empty trailing partial frame and run
/// it. This pins that policy structurally — not "one extra probability appeared"
/// but "the frame handed to the model is the remainder followed by zeros" —
/// which is the only formulation a silently-changed padding value or a dropped
/// tail cannot both satisfy.
///
/// Mutation: delete the `finish` body (drop the tail) → the frame count and the
/// probability count both fall to 2. Pad with anything but zero → the recorded
/// frame's tail comparison goes red.
#[test]
fn finish_zero_pads_a_partial_trailing_frame() {
  let partial = 1_234;
  let samples: Vec<f32> = (0..2 * CHUNK_SAMPLES + partial)
    .map(|i| (i % 97) as f32 / 97.0)
    .collect();

  let mut backend = MockVadBackend::new(vec![0.1, 0.2, 0.3]);
  let probabilities = stream_probabilities(&mut backend, &samples);

  assert_eq!(
    probabilities,
    vec![0.1, 0.2, 0.3],
    "2 whole frames from push + 1 zero-padded tail frame from finish"
  );
  assert_eq!(backend.frames.len(), 3, "three frames reached the model");
  let tail = &backend.frames[2];
  assert_eq!(tail.len(), CHUNK_SAMPLES, "the tail frame is a FULL frame");
  assert_eq!(
    &tail[..partial],
    &samples[2 * CHUNK_SAMPLES..],
    "the tail frame opens with the leftover samples, in order"
  );
  assert!(
    tail[partial..].iter().all(|&x| x == 0.0),
    "the tail frame is padded with ZEROS, not repeats or garbage"
  );
}

/// The other half of the policy: a stream that ends exactly on a frame boundary
/// has nothing buffered, so `finish` runs no model and emits nothing. (The
/// pre-zuoer detector's `if offset < samples.len()` guard, relocated.)
///
/// Mutation: make `finish` unconditional → a fourth, all-zero frame appears and
/// the counts go red.
#[test]
fn finish_on_an_exact_frame_boundary_emits_nothing() {
  let samples = vec![0.5_f32; 3 * CHUNK_SAMPLES];
  let mut backend = MockVadBackend::new(vec![0.1, 0.2, 0.3, 0.4]);
  let probabilities = stream_probabilities(&mut backend, &samples);

  assert_eq!(probabilities, vec![0.1, 0.2, 0.3]);
  assert_eq!(
    backend.frames.len(),
    3,
    "no phantom trailing frame on an exact boundary"
  );
}

/// Input windowing is the backend's too: the frame stream must depend on the
/// STREAM, not on how the caller happened to block its `push` calls. One
/// 2.75-frame buffer pushed whole and the same buffer pushed in ragged blocks
/// must produce byte-identical frames.
///
/// Mutation: drop the buffer and run `samples.chunks(CHUNK_SAMPLES)` per call →
/// the ragged run emits a frame per short block and this goes red.
#[test]
fn push_blocking_does_not_change_the_frame_stream() {
  let samples: Vec<f32> = (0..2 * CHUNK_SAMPLES + 3_072)
    .map(|i| (i % 31) as f32 / 31.0)
    .collect();
  let probabilities = vec![0.9, 0.1, 0.8];

  let mut whole = MockVadBackend::new(probabilities.clone());
  let via_whole = stream_probabilities(&mut whole, &samples);

  let mut ragged = MockVadBackend::new(probabilities);
  let mut cursor = 0;
  for block in [1_usize, CHUNK_SAMPLES - 1, 5, 4 * CHUNK_SAMPLES] {
    let end = (cursor + block).min(samples.len());
    ragged
      .push(&samples[cursor..end], &mut |_| {})
      .expect("ragged push");
    cursor = end;
  }
  assert_eq!(cursor, samples.len(), "the ragged blocks cover the buffer");
  ragged.finish(&mut |_| {}).expect("ragged finish");

  assert_eq!(
    whole.frames, ragged.frames,
    "frame boundaries must follow the stream, not the caller's block sizes"
  );
  assert_eq!(via_whole.len(), 3);
}

/// **THE MODEL-GATED TRAILING-FRAME EQUIVALENCE GATE.** The hermetic gates above
/// pin the policy the mock implements; this one pins the policy the REAL
/// [`CoreMlBackend`] implements, against a reference that reproduces the
/// pre-zuoer detector's loop verbatim — whole frames off the buffer, then the
/// zero-padded remainder — driven straight through [`VadModel::predict_chunk`],
/// the same call the backend makes. Bit-for-bit equality of the two probability
/// streams is the evidence that moving the tail policy from the detector into
/// the backend changed nothing observable.
///
/// Runs on the same 30 s fixture the end-to-end pin uses (480_000 samples =
/// 117 whole 4096-frames + a 768-sample remainder), so the padded tail is
/// genuinely exercised. `CpuOnly` for bit-determinism.
///
/// Mutation: drop the `finish` tail → 117 probabilities vs the reference's 118.
/// Pad with the last sample instead of zero → the 118th probability diverges.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn coreml_backend_frames_match_a_hand_chunked_zero_padded_reference() {
  let samples = common::load_wav_16k_mono(&common::fixture_wav_path(E2E_FIXTURE));
  assert_ne!(
    samples.len() % CHUNK_SAMPLES,
    0,
    "the fixture must NOT be a whole number of frames, or the tail policy is untested"
  );
  let options = VadModelOptions::new().with_compute(ComputeUnits::CpuOnly);

  // The pre-zuoer detector's loop, verbatim, against the model directly.
  let mut model = VadModel::load_with(common::model_path(), options).expect("load reference model");
  let mut reference = Vec::new();
  let mut offset = 0;
  while offset + CHUNK_SAMPLES <= samples.len() {
    reference.push(
      model
        .predict_chunk(&samples[offset..offset + CHUNK_SAMPLES])
        .expect("reference whole frame"),
    );
    offset += CHUNK_SAMPLES;
  }
  if offset < samples.len() {
    let mut tail = vec![0.0_f32; CHUNK_SAMPLES];
    tail[..samples.len() - offset].copy_from_slice(&samples[offset..]);
    reference.push(model.predict_chunk(&tail).expect("reference tail frame"));
  }
  assert_eq!(
    reference.len(),
    samples.len().div_ceil(CHUNK_SAMPLES),
    "the reference must run one frame per 256 ms plus the padded tail"
  );

  let mut backend =
    CoreMlBackend::load_with(common::model_path(), options).expect("load vadkit CoreML backend");
  let whole = stream_probabilities(&mut backend, &samples);
  assert_eq!(
    whole, reference,
    "CoreMlBackend's push+finish stream must equal the hand-chunked zero-padded reference"
  );

  // And the same stream delivered in ragged blocks, to pin the buffering.
  backend.reset();
  let mut ragged = Vec::new();
  let mut cursor = 0;
  let mut block = 1_usize;
  while cursor < samples.len() {
    let end = (cursor + block).min(samples.len());
    backend
      .push(&samples[cursor..end], &mut |p| ragged.push(p))
      .expect("ragged push");
    cursor = end;
    block = block * 3 + 7;
  }
  backend.finish(&mut |p| ragged.push(p)).expect("finish");
  assert_eq!(
    ragged, reference,
    "block sizes must not change the frame stream"
  );
}

// ── 4. End-to-end model-gated detect on real audio (two-sided pins) ──────────

use coremlit::{
  ComputeUnits,
  audio::vad::{CoreMlBackend, VadModel, VadModelOptions, detect_speech},
};

/// The committed fixture the e2e runs on (`common::FIXTURES`): pyannote's
/// canonical 30 s multi-speaker demo, 118 full 256 ms chunks — the same clip
/// the Swift-trace and cross-backend gates use.
const E2E_FIXTURE: &str = "02_pyannote_sample";

// Measured on `cpu_only` (bit-deterministic — T2 recorded identical output on
// all four compute units), then pinned two-sided. The ± band is one 256 ms
// frame (`CHUNK_SAMPLES`): the only thing that can move a boundary is a single
// probability crossing the 0.5 start-threshold under cross-silicon fp16 drift,
// and these are high-confidence clips (measured margin 0), so the band is
// T3's `TRACE_TOL`-style headroom over a measured-exact value, never slack that
// hides a regression (a real change moves a boundary by many frames or changes
// the segment count).
// Measured: one segment [106_016, 483_328) = 6.626 s .. 30.208 s. The start is
// the raw first-speech frame boundary 106_496 (frame 26 × 4096) minus the 30 ms
// (480-sample) `speech_pad`. The clip is 480_000 samples (30.0 s) = 117 full
// 4096-frames + one partial; speech runs to the end, so the trailing segment
// closes at the padded frame boundary 483_328 (118 × 4096) — one frame past the
// raw length: the backend's zero-padded-tail policy, unchanged from the
// pre-zuoer detector's. NOTE the ±1-frame `E2E_BOUNDARY_TOL_SAMPLES` band is
// exactly wide enough to swallow a DROPPED tail (479_232 vs 483_328), so this
// pin does not gate the trailing-frame policy —
// `coreml_backend_frames_match_a_hand_chunked_zero_padded_reference` does.
// Measured: mutating `CoreMlBackend::finish` to drop the tail leaves this test
// green and turns that one red.
const E2E_EXPECTED_SEGMENTS: usize = 1;
const E2E_FIRST_START_SAMPLE: u64 = 106_016;
const E2E_LAST_END_SAMPLE: u64 = 483_328;
const E2E_BOUNDARY_TOL_SAMPLES: u64 = CHUNK_SAMPLES as u64;

/// **THE END-TO-END RE-EXPORT GATE** (model-gated). Runs [`detect_speech`] over
/// a real [`CoreMlBackend`] on the fixture and pins the segment count and the
/// outer envelope (first start, last end) two-sided against the measured
/// values. Proves the whole public path — CoreML model → seam → zuoer
/// segmenter → segments — works on real audio, not just canned probabilities.
#[test]
#[ignore = "requires local vadkit models (VADKIT_TEST_MODELS)"]
fn detect_speech_on_real_audio_is_pinned() {
  let path = common::fixture_wav_path(E2E_FIXTURE);
  let fixture = common::FIXTURES
    .iter()
    .find(|f| f.name == E2E_FIXTURE)
    .expect("fixture entry");
  assert_eq!(
    common::sha256_hex(&path),
    fixture.sha256,
    "{E2E_FIXTURE}: fixture audio SHA-256 changed"
  );
  let samples = common::load_wav_16k_mono(&path);

  let mut backend = CoreMlBackend::load_with(
    common::model_path(),
    VadModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load vadkit CoreML backend");

  let segments = detect_speech(&mut backend, &samples, SpeechOptions::default())
    .expect("detect_speech over the CoreML backend");

  for (i, seg) in segments.iter().enumerate() {
    println!(
      "[reexport] {E2E_FIXTURE} seg {i}: [{}, {}) = {:.3}s..{:.3}s",
      seg.start_sample(),
      seg.end_sample(),
      seg.start_seconds(),
      seg.end_seconds(),
    );
  }

  assert_eq!(
    segments.len(),
    E2E_EXPECTED_SEGMENTS,
    "{E2E_FIXTURE}: segment count changed"
  );

  // Structural: non-empty, monotone, in-bounds.
  let total = samples.len() as u64;
  let mut prev_end = 0;
  for seg in &segments {
    assert!(
      seg.end_sample() > seg.start_sample(),
      "empty/inverted segment"
    );
    // `CoreMlBackend::finish` zero-pads a trailing PARTIAL frame and closes the
    // segment at the padded frame boundary (`n_frames * CHUNK_SAMPLES`), which
    // overhangs a clip that is not a whole number of frames by up to one frame
    // (02 is 480_000 samples, so the trailing segment ends at 483_328 = 118 ×
    // 4096). Tolerate up to that boundary, not the raw sample count.
    assert!(
      seg.end_sample() <= total.next_multiple_of(CHUNK_SAMPLES as u64),
      "segment past the padded trailing-frame boundary"
    );
    assert!(
      seg.start_sample() >= prev_end,
      "segments overlap / out of order"
    );
    prev_end = seg.end_sample();
  }

  // Two-sided envelope pins.
  let first_start = segments
    .first()
    .expect("at least one segment")
    .start_sample();
  let last_end = segments.last().expect("at least one segment").end_sample();
  assert!(
    first_start.abs_diff(E2E_FIRST_START_SAMPLE) <= E2E_BOUNDARY_TOL_SAMPLES,
    "{E2E_FIXTURE}: first start {first_start} outside {E2E_FIRST_START_SAMPLE} \
     ± {E2E_BOUNDARY_TOL_SAMPLES}"
  );
  assert!(
    last_end.abs_diff(E2E_LAST_END_SAMPLE) <= E2E_BOUNDARY_TOL_SAMPLES,
    "{E2E_FIXTURE}: last end {last_end} outside {E2E_LAST_END_SAMPLE} \
     ± {E2E_BOUNDARY_TOL_SAMPLES}"
  );
}
