//! CLAP dual-tower encode harness: the four cost phases the #30 audit called
//! out, measured PER TOWER (audio, text) × PER [`ComputeUnits`], model-gated.
//!
//! This replaces the audit's ad-hoc one-off numbers with a committed,
//! reproducible measurement. `harness = false` (a custom `main`, like
//! `benches/whisper/rtf.rs`): the load phases are **one-shot** costs — criterion's
//! statistical resampling would reload each model dozens of times, and every load
//! after the first hits the OS specialization cache, so it cannot see a
//! first-observed-load or first-inference cost at all.
//!
//! # The phases (spec §; audit #30)
//!
//! - **first-observed load** — the first [`crate::Model::load`] of a
//!   `(tower, placement)` in THIS fresh process. On a host that has never
//!   specialized this model for this device it folds in the **one-time OS
//!   specialization** cost; on a host that already has (e.g. after running the
//!   clap tests) it is already a cache hit. Either way it is the honest,
//!   real-world first-load number — no shared system state is touched to force
//!   it, so read it as an upper bound that still carries any not-yet-paid
//!   specialization.
//! - **cached load** — a second [`crate::Model::load`] in the SAME process, after
//!   the first-observed load has populated the OS specialization cache. It
//!   isolates the specialization-cache-HIT load path (any cache-MISS cost is paid
//!   by the first-observed load above). Being a back-to-back in-process call it
//!   also inherits process-local CoreML/framework init and warm allocator state
//!   that a fresh process does NOT, so read it as a lower bound on — not a
//!   measurement of — a later process's cached-start latency.
//! - **first inference** — the first `embed` after a load (the prediction path's
//!   own MPSGraph/ANE program specialization).
//! - **warm inference** — median / p90 over [`WARM_RUNS`] (≥ 20) steady-state
//!   `embed` calls.
//!
//! For each measured configuration it emits the latency, the output embedding's
//! hash (an 8-hex SHA-256 prefix over the warm embedding's bytes — a perf change
//! that alters numerics changes the hash) AND its cosine against the
//! [`ComputeUnits::CpuOnly`] reference (the semantic drift check the placement
//! gate uses), and a current process resident-memory snapshot.
//!
//! # True cold specialization is NOT measured here
//!
//! Genuine cold specialization — the first-ever load of a model the OS has never
//! specialized for this device — is a one-time OS cost. Measuring it reliably
//! requires a CLEAN, disposable environment (a fresh user profile or VM whose ANE
//! specialization cache, `~/Library/Caches/com.apple.e5rt.e5bundlecache`, has
//! never seen this model), NOT evicting the shared on-device cache, which is a
//! resource other CoreML processes on the host rely on. This harness therefore
//! never touches that cache and does not force a cold load; run it once in a
//! disposable profile if you need the cold number. The `first-observed load`
//! column already captures the cold cost whenever the host running the bench has
//! not previously specialized the model.
//!
//! Note on iteration order: the [`UNITS`] are measured in sequence, so a later
//! placement's `first-observed load` can be partly warmed by an earlier one. OS
//! specialization is per-model, so this cross-placement warming is limited, but
//! the first-observed column should be read with the iteration order in mind.
//!
//! Run (model-gated — set `CLAPKIT_TEST_MODELS`, or place models at
//! `Models/clapkit/`):
//! `cargo bench -p coremlit --features clap --bench clap_encode`
//!
//! `criterion_group!`-style benches carry a crate-level `#![allow(missing_docs)]`
//! because the macro expands to an undocumented `pub fn`; this custom-`main`
//! bench has no such expansion and no external API, so it needs no such allow.

use std::{
  hint::black_box,
  path::{Path, PathBuf},
  time::Instant,
};

use coremlit::{
  ComputeUnits,
  embeddings::clap::{
    AudioEncoder, AudioEncoderOptions, Embedding, TextEncoder, TextEncoderOptions,
    audio::TARGET_SAMPLES,
  },
};

/// Steady-state `embed` calls timed for the warm median / p90 (≥ 20 per the
/// spec; 25 for a stable p90 index).
const WARM_RUNS: usize = 25;

/// The public compute matrix both towers are measured over — the same four units
/// as `tests/clap/placement.rs`, but **`CpuOnly` first**: it is the cosine
/// reference, so measuring it first lets every other unit (`All` included) report
/// a real cross-placement cosine against it rather than a self-reference.
const UNITS: [ComputeUnits; 4] = [
  ComputeUnits::CpuOnly,
  ComputeUnits::All,
  ComputeUnits::CpuAndGpu,
  ComputeUnits::CpuAndNeuralEngine,
];

/// A representative text query (the placement gate's prompt).
const TEXT_PROMPT: &str = "a violin playing a slow melody in a concert hall";

fn main() {
  let models = models_dir();
  let audio_model = models.join("clap_audio.mlmodelc");
  let text_model = models.join("clap_text.mlmodelc");
  if !audio_model.is_dir() || !text_model.is_dir() {
    eprintln!(
      "clap_encode bench skipped: clapkit models not found under {} \
       (set CLAPKIT_TEST_MODELS, or fetch to Models/clapkit — see the crate README). \
       MEASURED NOTHING.",
      models.display()
    );
    return;
  }

  println!(
    "# clap_encode — CLAP dual-tower encode phases (measured, never marketed)\n\
     # models: {}\n\
     # warm runs: {WARM_RUNS}\n\
     # latency in ms; cos = cosine vs CpuOnly reference (same tower/input); \
     rss = current process resident MB (a snapshot, not a peak)\n",
    models.display(),
  );

  bench_audio(&audio_model);
  bench_text(&text_model);

  println!("\n# current process resident memory: {:.1} MB", rss_mb());
  println!(
    "# note: rss is a single current snapshot (task resident_size at the moment \
     each row prints), NOT a peak and NOT a cumulative running total — encoders \
     are dropped between configurations, so a row is neither monotonic nor that \
     configuration's isolated footprint. A true peak would need sampling across \
     load + inference."
  );
}

/// Audio tower: first-observed + cached load, first + warm `embed_window`, per
/// unit.
fn bench_audio(model: &Path) {
  println!("## audio tower (HTSAT) — {}", model.display());
  print_header();

  let samples = deterministic_window(TARGET_SAMPLES);
  let mut reference: Option<Embedding> = None;

  for unit in UNITS {
    // First-observed load: the first load of this (tower, unit) in this fresh
    // process — folds in the one-time OS specialization on a host that has not
    // yet specialized this model. Dropped before the cached measurement.
    let first_load_start = Instant::now();
    let priming =
      AudioEncoder::from_file_with(model, AudioEncoderOptions::new().with_compute(unit))
        .unwrap_or_else(|e| panic!("first-observed load audio [{unit}]: {e}"));
    let first_load_ms = ms(first_load_start);
    drop(priming);

    // Cached load: a second in-process load after the first populated the OS
    // specialization cache — isolates the specialization-cache-hit path. Being
    // back-to-back in-process, it is a lower bound on (not a measurement of) a
    // fresh process's cached start.
    let load_start = Instant::now();
    let encoder =
      AudioEncoder::from_file_with(model, AudioEncoderOptions::new().with_compute(unit))
        .unwrap_or_else(|e| panic!("cached load audio [{unit}]: {e}"));
    let cached_load_ms = ms(load_start);

    let first_start = Instant::now();
    let first = encoder
      .embed_window(&samples)
      .unwrap_or_else(|e| panic!("first audio embed [{unit}]: {e}"));
    let first_ms = ms(first_start);

    let mut warm = Vec::with_capacity(WARM_RUNS);
    let mut last = first;
    for _ in 0..WARM_RUNS {
      let t = Instant::now();
      last = black_box(
        encoder
          .embed_window(black_box(&samples))
          .unwrap_or_else(|e| panic!("warm audio embed [{unit}]: {e}")),
      );
      warm.push(ms(t));
    }

    let cos = record_cosine(unit, &last, &mut reference);
    print_row(
      unit,
      first_load_ms,
      cached_load_ms,
      first_ms,
      &mut warm,
      cos,
      &last,
    );
  }
  println!();
}

/// Text tower: first-observed + cached load, first + warm `embed`, per unit.
fn bench_text(model: &Path) {
  println!("## text tower (RoBERTa) — {}", model.display());
  print_header();

  let mut reference: Option<Embedding> = None;

  for unit in UNITS {
    // First-observed load: the first load of this (tower, unit) in this fresh
    // process — folds in the one-time OS specialization on a host that has not
    // yet specialized this model. Dropped before the cached measurement.
    let first_load_start = Instant::now();
    let priming =
      TextEncoder::from_bundled_tokenizer(model, TextEncoderOptions::new().with_compute(unit))
        .unwrap_or_else(|e| panic!("first-observed load text [{unit}]: {e}"));
    let first_load_ms = ms(first_load_start);
    drop(priming);

    // Cached load: a second in-process load after the first populated the OS
    // specialization cache — isolates the specialization-cache-hit path. Being
    // back-to-back in-process, it is a lower bound on (not a measurement of) a
    // fresh process's cached start.
    let load_start = Instant::now();
    let encoder =
      TextEncoder::from_bundled_tokenizer(model, TextEncoderOptions::new().with_compute(unit))
        .unwrap_or_else(|e| panic!("cached load text [{unit}]: {e}"));
    let cached_load_ms = ms(load_start);

    let first_start = Instant::now();
    let first = encoder
      .embed(TEXT_PROMPT)
      .unwrap_or_else(|e| panic!("first text embed [{unit}]: {e}"));
    let first_ms = ms(first_start);

    let mut warm = Vec::with_capacity(WARM_RUNS);
    let mut last = first;
    for _ in 0..WARM_RUNS {
      let t = Instant::now();
      last = black_box(
        encoder
          .embed(black_box(TEXT_PROMPT))
          .unwrap_or_else(|e| panic!("warm text embed [{unit}]: {e}")),
      );
      warm.push(ms(t));
    }

    let cos = record_cosine(unit, &last, &mut reference);
    print_row(
      unit,
      first_load_ms,
      cached_load_ms,
      first_ms,
      &mut warm,
      cos,
      &last,
    );
  }
  println!();
}

/// Cosine of `emb` against the `CpuOnly` reference for this tower. `CpuOnly` is
/// measured first in [`UNITS`] and seeds the reference (self-scoring to 1.0);
/// every subsequent unit is scored against it — the same drift check
/// `tests/clap/placement.rs` pins.
fn record_cosine(unit: ComputeUnits, emb: &Embedding, reference: &mut Option<Embedding>) -> f32 {
  if unit == ComputeUnits::CpuOnly {
    *reference = Some(emb.clone());
  }
  reference
    .as_ref()
    .map_or_else(|| emb.cosine(emb), |r| emb.cosine(r))
}

fn print_header() {
  println!(
    "{:<26} {:>12} {:>11} {:>10} {:>11} {:>10} {:>12} {:>9} {:>10}",
    "unit",
    "first_load",
    "cached_load",
    "first_inf",
    "warm_median",
    "warm_p90",
    "cos_vs_cpu",
    "rss_MB",
    "emb_hash"
  );
}

#[allow(clippy::too_many_arguments)]
fn print_row(
  unit: ComputeUnits,
  first_load_ms: f64,
  cached_load_ms: f64,
  first_ms: f64,
  warm: &mut [f64],
  cos: f32,
  emb: &Embedding,
) {
  let (median, p90) = median_p90(warm);
  println!(
    "{:<26} {:>12.1} {:>11.1} {:>10.2} {:>11.2} {:>10.2} {:>12.6} {:>9.1} {:>10}",
    unit.as_str(),
    first_load_ms,
    cached_load_ms,
    first_ms,
    median,
    p90,
    cos,
    rss_mb(),
    hash8(emb),
  );
}

/// Milliseconds elapsed since `start`.
fn ms(start: Instant) -> f64 {
  start.elapsed().as_secs_f64() * 1e3
}

/// Median and p90 of `samples` (sorted in place). p90 = the `ceil(0.9·n)-1`
/// order statistic (nearest-rank), matching a small-sample percentile.
fn median_p90(samples: &mut [f64]) -> (f64, f64) {
  samples.sort_by(f64::total_cmp);
  let n = samples.len();
  let median = samples[n / 2];
  let p90_idx = ((n as f64 * 0.9).ceil() as usize)
    .saturating_sub(1)
    .min(n - 1);
  (median, samples[p90_idx])
}

/// First 8 hex of the SHA-256 over the embedding's raw f32 little-endian bytes —
/// a stable fingerprint of the numeric output (any drift changes it).
fn hash8(emb: &Embedding) -> String {
  use sha2::{Digest, Sha256};
  let mut hasher = Sha256::new();
  for &v in emb.as_slice() {
    hasher.update(v.to_le_bytes());
  }
  hasher
    .finalize()
    .iter()
    .take(4)
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// This process's resident memory footprint, in MB. Replicates
/// `coremlit::audio::whisper::log::resident_memory_bytes` (that helper rides the
/// `whisper` feature, which a `clap` bench does not enable) via the `libc` /
/// `mach2` dev-deps.
fn rss_mb() -> f64 {
  // SAFETY: `mach_task_basic_info` is plain-old-data; the all-zero bit pattern
  // is a valid value for every field.
  let mut info: libc::mach_task_basic_info = unsafe { core::mem::zeroed() };
  let mut count = (core::mem::size_of::<libc::mach_task_basic_info>()
    / core::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
  // SAFETY: `mach_task_self()` names the current task; `info` is a zeroed,
  // properly sized/aligned MACH_TASK_BASIC_INFO out-struct and `count` carries
  // its capacity in words per the in-out contract, so the kernel writes only
  // within `info`. The result code is checked before `info` is read.
  let result = unsafe {
    libc::task_info(
      mach2::traps::mach_task_self(),
      libc::MACH_TASK_BASIC_INFO,
      (&raw mut info).cast(),
      &mut count,
    )
  };
  if result == libc::KERN_SUCCESS {
    info.resident_size as f64 / (1024.0 * 1024.0)
  } else {
    f64::NAN
  }
}

/// The clapkit model directory: `CLAPKIT_TEST_MODELS`, else
/// `<workspace>/Models/clapkit` — mirrors `tests/clap/common::models_dir`. A
/// bench target is its own crate and cannot reach a `tests/` module, so this is
/// duplicated (as the align bench duplicates its constants for the same reason).
fn models_dir() -> PathBuf {
  std::env::var_os("CLAPKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("clapkit")
    },
    PathBuf::from,
  )
}

/// A deterministic 48 kHz mono window: a sum of a few fixed sinusoids (no RNG),
/// giving both towers a stable, non-trivial input. Mirrors
/// `tests/clap/common::deterministic_window` (unreachable from a bench crate).
fn deterministic_window(len: usize) -> Vec<f32> {
  const SR: f32 = 48_000.0;
  (0..len)
    .map(|i| {
      let t = i as f32 / SR;
      let two_pi = std::f32::consts::TAU;
      0.5 * (two_pi * 220.0 * t).sin()
        + 0.3 * (two_pi * 440.0 * t).sin()
        + 0.2 * (two_pi * 1760.0 * t).sin()
    })
    .collect()
}
