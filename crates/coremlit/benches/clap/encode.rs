//! CLAP dual-tower encode harness: the four cost phases the #30 audit called
//! out, measured PER TOWER (audio, text) × PER [`ComputeUnits`], model-gated.
//!
//! This replaces the audit's ad-hoc one-off numbers with a committed,
//! reproducible measurement. `harness = false` (a custom `main`, like
//! `benches/whisper/rtf.rs`): the load phases are **one-shot** costs — criterion's
//! statistical resampling would reload each model dozens of times, and every load
//! after the first hits the OS specialization cache, so it cannot see a cold or a
//! first-inference cost at all.
//!
//! # The four phases (spec §; audit #30)
//!
//! - **cold specialization** — the first-ever load of a model the OS has not yet
//!   specialized for this device. A **one-time OS cost**, distinct from a cached
//!   load. Measured only in the opt-in, reversible cold mode (see below), because
//!   on any host that has already run the clap tests the model is already
//!   specialized and every load is a cache hit.
//! - **cached load** — [`crate::Model::load`] when the specialized artifact
//!   already exists in the OS cache. This is what every production process start
//!   pays after the one-time cold specialization; it is the steady-state load.
//! - **first inference** — the first `embed` after a load (the prediction path's
//!   own MPSGraph/ANE program specialization).
//! - **warm inference** — median / p90 over [`WARM_RUNS`] (≥ 20) steady-state
//!   `embed` calls.
//!
//! For each measured configuration it emits the latency, the output embedding's
//! hash (an 8-hex SHA-256 prefix over the warm embedding's bytes — a perf change
//! that alters numerics changes the hash) AND its cosine against the
//! [`ComputeUnits::CpuOnly`] reference (the semantic drift check the placement
//! gate uses), and the process resident-memory footprint.
//!
//! # Cold mode (opt-in, reversible)
//!
//! The default run measures **cached load / first inference / warm** for the full
//! 2 × 4 matrix — always safe, never touches state outside the target dir. To
//! also measure genuine **cold specialization**, set `CLAPKIT_BENCH_COLD=1`: for
//! `ComputeUnits::All` on each tower, the harness renames the on-device ANE
//! specialization cache (`~/Library/Caches/com.apple.e5rt.e5bundlecache`) aside,
//! times a cold [`crate::Model::load`], then **restores** it — a cache the OS
//! rebuilds transparently, left byte-for-byte as it was. Caveat: clearing the
//! e5rt cache yields a genuine cold number for the ANE-compiling **text** tower;
//! the **audio** (HTSAT) tower falls back to GPU/CPU (`ANECCompile` fails — see
//! `tests/clap/placement.rs`), whose MPSGraph specialization is cached partly
//! elsewhere, so the audio cold figure here is a lower bound on the true
//! first-ever specialization.
//!
//! Run (model-gated — set `CLAPKIT_TEST_MODELS`, or place models at
//! `Models/clapkit/`):
//! `cargo bench -p coremlit --features clap --bench clap_encode`
//! Add `CLAPKIT_BENCH_COLD=1` to also measure cold specialization.
//!
//! `criterion_group!`-style benches carry a crate-level `#![allow(missing_docs)]`
//! because the macro expands to an undocumented `pub fn`; this custom-`main`
//! bench has no such expansion and no external API, so it needs no such allow.

use std::{
  fs,
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

  let cold = std::env::var_os("CLAPKIT_BENCH_COLD").is_some_and(|v| v == "1");
  println!(
    "# clap_encode — CLAP dual-tower encode phases (measured, never marketed)\n\
     # models: {}\n\
     # warm runs: {WARM_RUNS}   cold mode: {}\n\
     # latency in ms; cos = cosine vs CpuOnly reference (same tower/input); \
     rss = cumulative process resident MB\n",
    models.display(),
    if cold {
      "ON (genuine cold specialization, e5rt cache reversibly cleared)"
    } else {
      "off (set CLAPKIT_BENCH_COLD=1 for cold specialization)"
    },
  );

  bench_audio(&audio_model, cold);
  bench_text(&text_model, cold);

  println!("\n# peak process resident memory: {:.1} MB", rss_mb());
  println!(
    "# note: rss is cumulative — every configuration above is loaded in this one \
     process, so each rss column is a running total, not that configuration's \
     isolated footprint."
  );
}

/// Audio tower: cold / cached load, first + warm `embed_window`, per unit.
fn bench_audio(model: &Path, cold: bool) {
  println!("## audio tower (HTSAT) — {}", model.display());
  print_header();

  let samples = deterministic_window(TARGET_SAMPLES);
  let mut reference: Option<Embedding> = None;

  for unit in UNITS {
    let cold_ms = (cold && unit == ComputeUnits::All)
      .then(|| {
        measure_cold(model, "audio", || {
          AudioEncoder::from_file_with(model, AudioEncoderOptions::new().with_compute(unit))
            .map(drop)
        })
      })
      .flatten();

    let load_start = Instant::now();
    let encoder =
      AudioEncoder::from_file_with(model, AudioEncoderOptions::new().with_compute(unit))
        .unwrap_or_else(|e| panic!("load audio [{unit}]: {e}"));
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
      cold_ms,
      cached_load_ms,
      first_ms,
      &mut warm,
      cos,
      &last,
    );
  }
  println!();
}

/// Text tower: cold / cached load, first + warm `embed`, per unit.
fn bench_text(model: &Path, cold: bool) {
  println!("## text tower (RoBERTa) — {}", model.display());
  print_header();

  let mut reference: Option<Embedding> = None;

  for unit in UNITS {
    let cold_ms = (cold && unit == ComputeUnits::All)
      .then(|| {
        measure_cold(model, "text", || {
          TextEncoder::from_bundled_tokenizer(model, TextEncoderOptions::new().with_compute(unit))
            .map(drop)
        })
      })
      .flatten();

    let load_start = Instant::now();
    let encoder =
      TextEncoder::from_bundled_tokenizer(model, TextEncoderOptions::new().with_compute(unit))
        .unwrap_or_else(|e| panic!("load text [{unit}]: {e}"));
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
      cold_ms,
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

/// Times a cold [`crate::Model::load`] with the OS ANE specialization cache
/// reversibly cleared, then restores the cache. Returns the cold load in ms, or
/// `None` if the cache could not be safely moved aside (reported, never faked).
fn measure_cold(
  _model: &Path,
  tower: &str,
  load: impl Fn() -> Result<(), coremlit::embeddings::clap::Error>,
) -> Option<f64> {
  let cache = e5rt_cache_dir()?;
  let backup = cache.with_extension(format!("clapbench-bak-{}", std::process::id()));

  // Rename the specialization cache aside so the next load is genuinely cold.
  // Atomic on the same filesystem; skip (report None) if it is absent or the
  // move fails, rather than fabricate a cold number.
  if !cache.exists() {
    eprintln!(
      "[cold] {tower}: no e5rt cache present — cannot force a cold load; reporting cached only"
    );
    return None;
  }
  if let Err(e) = fs::rename(&cache, &backup) {
    eprintln!("[cold] {tower}: could not move e5rt cache aside ({e}); reporting cached only");
    return None;
  }

  let start = Instant::now();
  let result = load();
  let elapsed = ms(start);

  // Restore: discard whatever the cold load repopulated, then move the original
  // cache back so the user's on-device cache is left exactly as it was.
  let _ = fs::remove_dir_all(&cache);
  if let Err(e) = fs::rename(&backup, &cache) {
    eprintln!(
      "[cold] {tower}: WARNING failed to restore e5rt cache from {} ({e}) — the OS will \
       transparently rebuild it on next use",
      backup.display()
    );
  }

  match result {
    Ok(()) => Some(elapsed),
    Err(e) => {
      eprintln!("[cold] {tower}: cold load failed: {e}");
      None
    }
  }
}

/// `~/Library/Caches/com.apple.e5rt.e5bundlecache`, the on-device ANE
/// specialization cache, if `HOME` is set.
fn e5rt_cache_dir() -> Option<PathBuf> {
  std::env::var_os("HOME").map(|home| {
    PathBuf::from(home)
      .join("Library")
      .join("Caches")
      .join("com.apple.e5rt.e5bundlecache")
  })
}

fn print_header() {
  println!(
    "{:<26} {:>12} {:>11} {:>10} {:>11} {:>10} {:>12} {:>9} {:>10}",
    "unit",
    "cold_load",
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
  cold_ms: Option<f64>,
  cached_load_ms: f64,
  first_ms: f64,
  warm: &mut [f64],
  cos: f32,
  emb: &Embedding,
) {
  let (median, p90) = median_p90(warm);
  let cold = cold_ms.map_or_else(|| "—".to_string(), |v| format!("{v:.1}"));
  println!(
    "{:<26} {:>12} {:>11.1} {:>10.2} {:>11.2} {:>10.2} {:>12.6} {:>9.1} {:>10}",
    unit.as_str(),
    cold,
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
