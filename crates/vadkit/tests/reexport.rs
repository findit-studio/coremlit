//! The re-export layer (design spec §2-§4): proves vadkit's public detection
//! surface is silero's, wired over the CoreML backend, with ZERO segmentation
//! logic authored in vadkit.
//!
//! Three gates:
//!
//! 1. **`src_authors_no_detection_logic`** (hermetic) — the no-duplication
//!    proof. Greps every `crates/vadkit/src/**/*.rs` file for the
//!    silero-segmenter vocabulary (thresholding, hysteresis, `min_speech`/
//!    `min_silence`/`speech_pad`, driving/constructing a segmenter). vadkit's
//!    `src/` contains NONE of it; a re-implementation of any segment-assembly
//!    step drags at least one token in and turns this red.
//! 2. **The mock-backend scenarios** (hermetic) — silero's own 4096-geometry
//!    detector-test scenarios, replayed over a CoreML-SHAPED mock
//!    ([`MockVadBackend`]: `frame_samples() == 4096`, one canned probability
//!    per frame) driven through the re-exported [`vadkit::detect_speech_with`].
//!    Same inputs, same pinned segment boundaries silero pins internally — so
//!    the re-export provably drives silero's real segmenter, not a copy. Plus
//!    the error-bridge shape an out-of-tree backend uses.
//! 3. **`detect_speech_on_real_audio_is_pinned`** (model-gated) — the end-to-
//!    end path: [`vadkit::detect_speech`] over a real [`CoreMlBackend`] on a
//!    committed fixture, segment starts/ends pinned two-sided.

mod common;

use silero::{SampleRate, SpeechOptions, VadBackend};
use vadkit::{CHUNK_SAMPLES, detect_speech_with};

// ── 1. No-duplication grep gate ─────────────────────────────────────────────

/// Silero-segmenter identifiers and thresholding vocabulary that would appear
/// in vadkit's `src/` ONLY if it re-implemented some part of the segment
/// assembly the `silero` crate single-homes (spec §2-§3). Re-exporting the
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
/// single-homed in `silero`, and vadkit only implements the backend seam and
/// re-exports the detector surface. Scans every source file for
/// [`FORBIDDEN_DETECTION_TOKENS`].
#[test]
fn src_authors_no_detection_logic() {
  let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
  let mut files = Vec::new();
  rust_files(&src, &mut files);
  assert!(
    files.len() >= 4,
    "expected to scan vadkit's src tree, found only {} files under {src:?}",
    files.len()
  );

  let mut violations = Vec::new();
  for file in &files {
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("read {file:?}: {e}"));
    for (lineno, line) in text.lines().enumerate() {
      // Scan CODE only, not prose: the claim is that vadkit AUTHORS no
      // segmentation logic, so a doc comment that DESCRIBES what silero owns
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
            file.strip_prefix(&src).unwrap_or(file).display(),
            lineno + 1,
            line.trim(),
          ));
        }
      }
    }
  }

  assert!(
    violations.is_empty(),
    "vadkit/src must author NO detection logic — it lives single-homed in \
     `silero` (spec §2-§3). Found {} violation(s):\n{}",
    violations.len(),
    violations.join("\n"),
  );
}

// ── 2. CoreML-shaped mock backend + silero's detector-test scenarios ─────────

/// A backend error distinct from `vadkit`'s own [`vadkit::InferError`], present
/// to exercise the out-of-tree error bridge the [`VadBackend::Error`] contract
/// prescribes — the same shape [`vadkit::CoreMlBackend`] uses for real
/// (`impl From<TheirError> for silero::Error` wrapping in
/// [`silero::Error::Backend`]).
#[derive(Debug)]
struct MockError(&'static str);

impl std::fmt::Display for MockError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.0)
  }
}

impl std::error::Error for MockError {}

impl From<MockError> for silero::Error {
  fn from(error: MockError) -> Self {
    silero::Error::Backend(Box::new(error))
  }
}

/// A [`VadBackend`] shaped like [`vadkit::CoreMlBackend`] — 4096-sample
/// (256 ms) frames at 16 kHz — returning one canned probability per frame. It
/// authors no detection logic; it exists to replay silero's detector scenarios
/// over the re-exported [`detect_speech_with`] at vadkit's exact geometry,
/// proving the re-export drives silero's real segmenter.
struct MockVadBackend {
  probabilities: Vec<f32>,
  cursor: usize,
  fail_at: Option<usize>,
}

impl MockVadBackend {
  fn new(probabilities: Vec<f32>) -> Self {
    Self {
      probabilities,
      cursor: 0,
      fail_at: None,
    }
  }

  fn failing_at(index: usize, probabilities: Vec<f32>) -> Self {
    Self {
      probabilities,
      cursor: 0,
      fail_at: Some(index),
    }
  }
}

impl VadBackend for MockVadBackend {
  type Error = MockError;

  fn frame_samples(&self) -> usize {
    CHUNK_SAMPLES
  }

  fn sample_rate(&self) -> SampleRate {
    SampleRate::Rate16k
  }

  fn predict(&mut self, frame: &[f32]) -> Result<f32, MockError> {
    assert_eq!(
      frame.len(),
      CHUNK_SAMPLES,
      "the detector must hand a CoreML-shaped backend exactly frame_samples per frame"
    );
    if self.fail_at == Some(self.cursor) {
      return Err(MockError("mock predict failure"));
    }
    let probability = self.probabilities.get(self.cursor).copied().unwrap_or(0.0);
    self.cursor += 1;
    Ok(probability)
  }

  fn reset(&mut self) {
    self.cursor = 0;
  }
}

#[test]
fn reexport_closes_after_two_256ms_low_frames() {
  // silero's `mock_geometry_closes_after_two_256ms_low_frames`, replayed over
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
  // silero's `mock_geometry_holds_open_through_one_256ms_low_frame`: a single
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
  // `detect_speech_with` through the transparent `silero::Error::Backend`
  // variant, delegating its `Display` to the wrapped error — the exact bridge
  // `CoreMlBackend`'s `InferError` uses (proven to compile; here proven to
  // propagate).
  let mut backend = MockVadBackend::failing_at(1, vec![0.9, 0.9, 0.9]);
  let samples = vec![0.0_f32; 3 * CHUNK_SAMPLES];
  let error = detect_speech_with(&mut backend, &samples, SpeechOptions::default())
    .expect_err("backend failure must propagate");
  assert!(
    matches!(error, silero::Error::Backend(_)),
    "backend error must bridge through silero::Error::Backend, got {error:?}"
  );
  assert_eq!(error.to_string(), "mock predict failure");
}

// ── 3. End-to-end model-gated detect on real audio (two-sided pins) ──────────

use coremlit::ComputeUnits;
use vadkit::{CoreMlBackend, VadModelOptions, detect_speech};

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
// raw length, silero's zero-padded-tail semantics inherited faithfully.
const E2E_EXPECTED_SEGMENTS: usize = 1;
const E2E_FIRST_START_SAMPLE: u64 = 106_016;
const E2E_LAST_END_SAMPLE: u64 = 483_328;
const E2E_BOUNDARY_TOL_SAMPLES: u64 = CHUNK_SAMPLES as u64;

/// **THE END-TO-END RE-EXPORT GATE** (model-gated). Runs [`detect_speech`] over
/// a real [`CoreMlBackend`] on the fixture and pins the segment count and the
/// outer envelope (first start, last end) two-sided against the measured
/// values. Proves the whole public path — CoreML model → seam → silero
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
    // `detect_speech_with` zero-pads a trailing PARTIAL frame and closes the
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
