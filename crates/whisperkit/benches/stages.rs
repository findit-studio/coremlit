//! Hermetic criterion benches over the pure pipeline stages (no models).
//!
//! These are the spec §9.4 micro-benches: logits filters at real vocab
//! geometry, DTW at word-alignment shape, energy-VAD chunking on 60 s of
//! synthetic audio, and token compression ratio. Run:
//! `cargo bench -p whisperkit --bench stages`
//!
//! `criterion_group!` below expands to a `pub fn` with no doc comment of
//! its own; outer `#[allow]` on a `macro_rules!` invocation does not
//! forward into its expansion (confirmed: rustc still denies it and flags
//! the attribute itself as unused), so `missing_docs` is silenced for this
//! whole bench binary instead — it has no external API for the lint to
//! protect.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use whisperkit::{
  audio::{
    chunker::VadChunker,
    vad::{EnergyVad, VoiceActivityDetector},
  },
  backend::AlignmentView,
  decode::filter::{
    LanguageLogitsFilter, LogitsFilter, SuppressBlankFilter, SuppressTokensFilter,
    TimestampRulesFilter,
  },
  segment::dynamic_time_warping,
  text::compression_ratio_of_tokens,
  tokenizer::SpecialTokens,
};

/// Tiny-model vocabulary size — the buffer every filter scans.
const VOCAB: usize = 51_865;

fn logits_fixture() -> Vec<f32> {
  (0..VOCAB).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect()
}

fn bench_logits_filters(c: &mut Criterion) {
  let s = SpecialTokens::whisper_defaults();
  let mut group = c.benchmark_group("decode/filter");

  // Mid-decode prompt: [sot, lang, task] prefix + a timestamped word.
  let tokens = [
    s.start_of_transcript_token(),
    s.english_token(),
    s.transcribe_token(),
    s.time_token_begin(),
    1_000,
    2_000,
  ];
  let timestamp = TimestampRulesFilter::new(&s, 3, None, true);
  group.bench_function("timestamp_rules", |b| {
    b.iter_batched_ref(
      logits_fixture,
      |logits| timestamp.filter(black_box(logits), black_box(&tokens)),
      BatchSize::LargeInput,
    )
  });

  // ~90 suppressed ids — the order of Swift's non-speech token list.
  let suppress = SuppressTokensFilter::new((0..90u32).map(|i| i * 379).collect());
  group.bench_function("suppress_tokens", |b| {
    b.iter_batched_ref(
      logits_fixture,
      |logits| suppress.filter(black_box(logits), black_box(&tokens)),
      BatchSize::LargeInput,
    )
  });

  let blank = SuppressBlankFilter::new(&s, 1);
  let at_sample_begin = [s.start_of_transcript_token()];
  group.bench_function("suppress_blank", |b| {
    b.iter_batched_ref(
      logits_fixture,
      |logits| blank.filter(black_box(logits), black_box(&at_sample_begin)),
      BatchSize::LargeInput,
    )
  });

  // All 99 language tokens; the filter masks the rest of the vocab.
  let language_tokens: Vec<u32> = (0..99).map(|i| s.english_token() + i).collect();
  let language = LanguageLogitsFilter::new(&language_tokens, 1);
  group.bench_function("language", |b| {
    b.iter_batched_ref(
      logits_fixture,
      |logits| language.filter(black_box(logits), black_box(&at_sample_begin)),
      BatchSize::LargeInput,
    )
  });
  group.finish();
}

fn bench_dtw(c: &mut Criterion) {
  // Word-alignment shape: ~32 text tokens x 1500 encoder frames (30 s).
  let rows: usize = 32;
  let cols: usize = 1_500;
  let data: Vec<f32> = (0..rows * cols)
    .map(|i| ((i.wrapping_mul(2_654_435_761)) % 1_000) as f32 / 1_000.0)
    .collect();
  let view = AlignmentView::new(&data, rows, cols);
  c.bench_function("segment/dtw_32x1500", |b| {
    b.iter(|| dynamic_time_warping(black_box(&view)).unwrap())
  });
}

fn bench_vad_chunker(c: &mut Criterion) {
  // 60 s @ 16 kHz alternating 2 s speech-ish tone / 2 s near-silence.
  let samples: Vec<f32> = (0..960_000)
    .map(|i| {
      if (i / 32_000) % 2 == 0 {
        0.3 * ((i % 160) as f32 / 160.0 - 0.5)
      } else {
        0.001
      }
    })
    .collect();
  let vad = EnergyVad::new();
  c.bench_function("audio/energy_vad_60s", |b| {
    b.iter(|| vad.voice_activity(black_box(&samples)))
  });
  let chunker = VadChunker::new();
  let clips = [(0usize, samples.len())];
  c.bench_function("audio/vad_chunk_all_60s", |b| {
    b.iter(|| chunker.chunk_all(&vad, black_box(&samples), 480_000, black_box(&clips)))
  });
}

fn bench_compression_ratio(c: &mut Criterion) {
  // 240 tokens: the worst repetitive case vs a varied transcript —
  // the two poles the fallback ladder distinguishes.
  let repetitive: Vec<u32> = std::iter::repeat_n([220u32, 220, 220, 220], 60)
    .flatten()
    .collect();
  let varied: Vec<u32> = (0..240u32).map(|i| 1_000 + (i * 37) % 40_000).collect();
  c.bench_function("text/compression_ratio_repetitive_240", |b| {
    b.iter(|| compression_ratio_of_tokens(black_box(&repetitive)))
  });
  c.bench_function("text/compression_ratio_varied_240", |b| {
    b.iter(|| compression_ratio_of_tokens(black_box(&varied)))
  });
}

criterion_group!(
  stages,
  bench_logits_filters,
  bench_dtw,
  bench_vad_chunker,
  bench_compression_ratio
);
criterion_main!(stages);
