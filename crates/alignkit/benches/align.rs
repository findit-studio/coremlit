//! Forced-alignment throughput on the shipping configuration (model-gated).
//!
//! Reports the two numbers desktop#120's acceptance criteria ask for, and
//! reports them **separately**:
//!
//! - `encode` — the CoreML wav2vec2 forward pass alone. This is the only stage
//!   alignkit owns; everything downstream is asry's parity-tested algorithm,
//!   shared byte-for-byte with the ONNX path.
//! - `align_chunk` — the whole pipeline (`prepare` → encode → `finish`), which
//!   is what a caller pays.
//!
//! Their difference is the CPU cost of the trellis/beam/compose stages, so a
//! reader can attribute any regression to the encoder or to the algorithm
//! without re-running anything.
//!
//! **Alignment runtime is NOT ASR runtime.** desktop#120 exists because those
//! two were once added together and reported as one figure. This bench never
//! loads an ASR model and never transcribes; the transcript is a constant. RTF
//! here is *alignment* RTF over a chunk of known duration.
//!
//! Run: `cargo bench -p alignkit --bench align`
//!
//! # This bench does not skip
//!
//! A missing model is a hard failure, exactly as in the crate's model-gated
//! tests. A benchmark that exits 0 having measured nothing looks, in a CI log,
//! indistinguishable from one that measured something — and this crate has
//! already shipped one gate that reported `ok. 1 passed` with no model loaded.
//! Set `ALIGNKIT_TEST_MODELS`, or put the model at `Models/alignkit/`.
//!
//! `criterion_group!` expands to a `pub fn` with no doc comment of its own, and
//! an outer `#[allow]` on a `macro_rules!` invocation does not reach the item
//! it expands to — hence the crate-level allow (whisperkit's benches carry the
//! same one, for the same reason).
#![allow(missing_docs)]

use core::{sync::atomic::AtomicBool, time::Duration};
use std::{
  hint::black_box,
  path::{Path, PathBuf},
};

use alignkit::{
  ANALYSIS_TIMEBASE, Aligner, EnglishNormalizer, Lang, OutputClock, default_oov_decisions,
  encode::{Encoder, EncoderInput},
};
use criterion::{Criterion, criterion_group, criterion_main};

/// The known transcript for `jfk.wav` (whisperkit's
/// `tests/fixtures/golden/jfk_tiny_golden.json`). Duplicated from
/// `tests/common/mod.rs` because a bench target is its own crate and cannot
/// reach a `tests/` module.
const JFK_TRANSCRIPT: &str = "And so my fellow Americans ask not what your country can do for you, \
                              ask what you can do for your country.";

/// `jfk.wav`'s duration, for the real-time factor. 176,000 samples @ 16 kHz.
const JFK_SECONDS: f64 = 11.0;

fn models_dir() -> PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("alignkit")
    },
    PathBuf::from,
  )
}

fn load_wav_mono_f32(path: &Path) -> Vec<f32> {
  let mut reader = hound::WavReader::open(path).expect("fixture wav opens");
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

fn bench_align(c: &mut Criterion) {
  let model = models_dir().join("base960h_aligner.mlmodelc");
  let samples = load_wav_mono_f32(
    &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../whisperkit/tests/fixtures/audio/jfk.wav"),
  );

  // `from_paths` / `from_file` → the SHIPPING defaults, never a hardcoded
  // compute placement: a benchmark pinned to a compute unit measures only that
  // compute unit, and a number measured on a configuration the crate does not
  // ship is worse than no number.
  let aligner = Aligner::from_paths(Lang::En, &model, Box::new(EnglishNormalizer::new()))
    .expect("build the En aligner (set ALIGNKIT_TEST_MODELS to the model directory)");
  let encoder = Encoder::from_file(&model).expect("load the CoreML encoder");

  let events = aligner.detect_oov(JFK_TRANSCRIPT).expect("detect_oov");
  let decisions = default_oov_decisions(&events);
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);

  let mut group = c.benchmark_group("alignkit");
  // One CoreML forward pass over the fixed 960,000-sample window is ~0.7 s on
  // the CpuOnly default, so criterion's 5 s default measurement window would
  // collect too few samples to be stable.
  group
    .sample_size(10)
    .measurement_time(Duration::from_secs(30))
    .throughput(criterion::Throughput::Elements(JFK_SECONDS as u64));

  // Stage 1 of 2: the CoreML forward pass alone — the ONLY stage alignkit owns.
  // Fed the raw samples rather than a `PreparedChunk::encoder_input()`: the
  // encoder's cost is a function of the fixed window, not of the content, and
  // the silence mask that distinguishes them is measured as part of
  // `align_chunk` below.
  group.bench_function("encode", |b| {
    b.iter(|| {
      black_box(
        encoder
          .emissions(EncoderInput::from_samples(black_box(&samples)).expect("jfk fits the window"))
          .expect("encode"),
      );
    });
  });

  // Stage 2 of 2: prepare → encode → finish, i.e. what a caller pays for one
  // chunk. Subtract `encode` to get the algorithm's share.
  group.bench_function("align_chunk", |b| {
    b.iter(|| {
      black_box(
        aligner
          .align_chunk(
            black_box(&samples),
            &[],
            black_box(JFK_TRANSCRIPT),
            clock,
            &abort,
            &decisions,
          )
          .expect("align_chunk"),
      );
    });
  });

  group.finish();

  // Criterion reports throughput in elements/sec; with one element per second
  // of audio that IS the speed factor, and RTF is its reciprocal. Spelled out
  // here so the acceptance criteria can be read straight off the bench output
  // without deriving anything.
  println!(
    "\nalignment RTF = (wall time reported above) / {JFK_SECONDS:.1} s of audio; speed factor = \
     its reciprocal. ASR runtime is NOT included — no ASR model is loaded by this bench.\n"
  );
}

criterion_group!(benches, bench_align);
criterion_main!(benches);
