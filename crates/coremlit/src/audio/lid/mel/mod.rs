//! Log-mel front end for the language-identification graph — the same
//! Rust-mel pattern as the ced/clap/whisper front ends (`audio/ced/mel/` is the
//! implementation template): precomputed window + filterbank + `rustfft` plan,
//! scalar `unsafe`-free reductions, `&self` extraction.
//!
//! # Why a Rust front end at all
//!
//! The exported graph starts AT the mel: its first ops are the fused input
//! normalization (see [`Self::extract_into`]'s closing note), not an STFT. An
//! in-graph STFT is the fragility class this crate avoids everywhere.
//!
//! # These constants are MEASURED, not believed
//!
//! Every value here was read off the producing pipeline and checked against
//! two independent oracles: SpeechBrain's own `processing_features.py`
//! (`STFT` / `Filterbank`) and the artifact author's `frontend.py`, whose
//! window and filterbank tables this module reproduces (`tests.rs` carries the
//! SHA-256 anchors). The graph's own `MLModelCreatorDefinedKey` metadata
//! independently states `n_fft = win_length = 400`, `hop_length = 160`,
//! `n_mels = 60`, `sample_rate = 16000`.
//!
//! Four conventions here are ones a generic mel implementation gets WRONG, and
//! each has a dedicated red-if-violated test in the sibling `tests.rs`:
//!
//! 1. **Periodic Hamming, not Hann and not symmetric.**
//!    `w[k] = 0.54 - 0.46·cos(2πk/400)` over `k ∈ [0, 400)`. Hann would give
//!    `w[0] = 0`; symmetric Hamming would divide by `399` and give
//!    `w[399] = 0.08` exactly. Periodic Hamming gives `w[0] = 0.08`,
//!    `w[399] ≈ 0.080057`, and an exact sum of `216.0 = 0.54 × 400` (the
//!    cosine sums to zero over a whole period — the symmetric window's sum is
//!    ~216.46).
//! 2. **`center = True` with CONSTANT-ZERO padding**, 200 samples each side.
//!    SpeechBrain sets `pad_mode="constant"`, overriding torch's `"reflect"`
//!    default — `torch.stft`'s own default would reflect, and ced's front end
//!    (which follows torchaudio) does reflect. Copying ced here would be
//!    wrong.
//! 3. **Symmetric triangles from the LOWER-side band, unnormalized.**
//!    SpeechBrain computes `band[i] = hz[i+1] - hz[i]` and uses that ONE width
//!    for both sides of triangle `i`
//!    (`processing_features.py::_triangular_filters`:
//!    `slope = (all_freqs - f_central) / band`, then
//!    `max(0, min(slope + 1, -slope + 1))`). That is not the usual
//!    HTK/librosa asymmetric construction, which uses each neighbour's own
//!    spacing. There is also NO Slaney/area normalization: triangles peak at
//!    ~1.0 and column sums run 0.645..8.452, growing with frequency.
//! 4. **One global `top_db = 80` over the WHOLE utterance.**
//!    `10·log10(max(x, 1e-10))`, then `max(db, db_max - 80)` where `db_max` is
//!    the maximum over time AND mel of the entire clip — not per frame, and
//!    not a fixed floor. (`db_multiplier` is `log10(max(amin, ref_value)) =
//!    log10(1.0) = 0`, so SpeechBrain's `x_db -= multiplier·db_multiplier` is
//!    a no-op and is not reproduced here.)
//!
//! The spectrum is plain power, `re² + im²`, with no `1/√n_fft` scaling.
//!
//! # Layout: TIME-major
//!
//! Output is `out[t · N_MELS + m]`, matching the graph's `[1, frames, 60]`
//! input. This is the OPPOSITE of ced's freq-major `[1, n_mels, T]`, and the
//! two produce identically-shaped buffers for a square-ish clip — hence the
//! sibling `layout_is_time_major_rows` mutation test.
//!
//! # Precision: f32 tables, f64 signal path
//!
//! The window and filterbank are built with **f32 arithmetic**, mirroring the
//! producing toolchain (SpeechBrain builds them in torch f32) so the tables are
//! bit-comparable against its published digests — an f64 build narrowed to f32
//! differs in 380 of 12 060 filterbank weights and would not be. Everything
//! downstream — framing, FFT, power, filterbank accumulation, dB — runs in f64
//! and narrows once at the write, the crate's convention and strictly tighter
//! than the f32 reference.

use core::fmt;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::audio::lid::error::Result;

/// Mel-frequency bin count — the graph's input width (`n_mels`).
pub(crate) const N_MELS: usize = 60;

/// FFT size, equal to the window length: 400 samples = 25 ms at 16 kHz.
pub(crate) const N_FFT: usize = 400;

/// STFT hop: 160 samples = 10 ms at 16 kHz.
pub(crate) const HOP: usize = 160;

/// Positive-frequency bin count of a real FFT of [`N_FFT`] points.
const N_FREQ: usize = N_FFT / 2 + 1;

/// Upper edge of the mel filterbank: Nyquist at 16 kHz.
const F_MAX: f64 = 8_000.0;

/// SpeechBrain `Filterbank`'s `amin`: the power floor before `log10`, giving
/// −100 dB on silence.
const AMIN: f64 = 1e-10;

/// SpeechBrain `Filterbank`'s `top_db`: the whole-utterance dynamic-range
/// clamp, in dB (trap 4 in the module docs).
const TOP_DB: f64 = 80.0;

/// Log-mel extractor. Owns the Hamming window, mel filterbank, and FFT plan
/// (all immutable after construction); per-call scratch is allocated locally so
/// [`Self::extract_into`] takes `&self`.
pub(crate) struct MelExtractor {
  window: [f32; N_FFT],
  /// `[N_FREQ × N_MELS]`, **freq-major** rows (`filterbank[k · N_MELS + m]`) —
  /// the orientation SpeechBrain's `fbank_matrix` has, so one frame's power
  /// spectrum walks it contiguously.
  filterbank: Vec<f32>,
  fft: Arc<dyn Fft<f64>>,
}

impl fmt::Debug for MelExtractor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // `Arc<dyn Fft>` is not `Debug`; the window/filterbank are fixed tables.
    f.debug_struct("MelExtractor").finish_non_exhaustive()
  }
}

impl MelExtractor {
  /// Periodic Hamming window (trap 1): `w[k] = 0.54 - 0.46·cos(2πk/n)`.
  ///
  /// The division is by `n`, NOT `n - 1`; a symmetric Hamming would use the
  /// latter and end at exactly `0.08` instead of `0.080057`.
  ///
  /// The cosine is evaluated in f64 and narrowed, which is the
  /// correctly-rounded f32 cosine of the f32 argument for all 400 samples;
  /// building the whole expression in f32 would instead inherit whichever
  /// `cosf` the host libm ships.
  fn periodic_hamming() -> [f32; N_FFT] {
    let mut window = [0.0f32; N_FFT];
    for (k, slot) in window.iter_mut().enumerate() {
      let phase = core::f32::consts::TAU * (k as f32) / (N_FFT as f32);
      let cosine = f64::from(phase).cos() as f32;
      *slot = 0.54f32 - 0.46f32 * cosine;
    }
    window
  }

  /// Hz → mel, SpeechBrain's scale: `2595·log10(1 + f/700)`.
  fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
  }

  /// mel → Hz, the inverse of [`Self::hz_to_mel`], evaluated on an f32 mel
  /// point. The `powf` runs in f64 and narrows — the f32 power loop of the
  /// producing toolchain agrees with that on every one of these 62 points.
  fn mel_to_hz(mel: f32) -> f32 {
    let exponent = mel / 2595.0f32;
    let power = 10.0f64.powf(f64::from(exponent)) as f32;
    700.0f32 * (power - 1.0f32)
  }

  /// Build SpeechBrain's `[N_FREQ × N_MELS]` freq-major triangular filterbank
  /// (trap 3): symmetric triangles whose half-width is the LOWER-side mel
  /// spacing, and no area normalization.
  ///
  /// The mel grid is laid out in f64 and narrowed per point (the producing
  /// toolchain's `linspace(..., dtype=float32)`, which likewise interpolates in
  /// double and casts); everything after that is f32, so the table is
  /// bit-comparable against the published digest.
  fn build_filterbank() -> Vec<f32> {
    let mel_min = Self::hz_to_mel(0.0);
    let mel_max = Self::hz_to_mel(F_MAX);
    let step = (mel_max - mel_min) / (N_MELS + 1) as f64;

    // N_MELS + 2 mel-spaced edges: edge[m] .. edge[m + 2] bracket triangle m.
    let mut edges_hz = [0.0f32; N_MELS + 2];
    for (m, slot) in edges_hz.iter_mut().enumerate() {
      // Deliberately NOT `mul_add`: a fused multiply-add rounds once where the
      // reference grid rounds twice, which moves a handful of edges by an ulp
      // and takes the table off its published digest.
      let mel = ((m as f64) * step + mel_min) as f32;
      *slot = Self::mel_to_hz(mel);
    }
    // The grid's last point is the endpoint exactly, not `mel_min + 61·step`.
    edges_hz[N_MELS + 1] = Self::mel_to_hz(mel_max as f32);

    let mut filterbank = vec![0.0f32; N_FREQ * N_MELS];
    for k in 0..N_FREQ {
      // `linspace(0, 8000, 201)`: every point is an exact multiple of 40 Hz.
      let freq = (k as f64 * (F_MAX / (N_FREQ - 1) as f64)) as f32;
      for m in 0..N_MELS {
        let center = edges_hz[m + 1];
        // ONE width for both sides — see trap 3.
        let band = edges_hz[m + 1] - edges_hz[m];
        let slope = (freq - center) / band;
        filterbank[k * N_MELS + m] = f32::max(0.0, f32::min(slope + 1.0, -slope + 1.0));
      }
    }
    filterbank
  }

  pub(crate) fn new() -> Self {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    Self {
      window: Self::periodic_hamming(),
      filterbank: Self::build_filterbank(),
      fft,
    }
  }

  /// Compute the log-mel features for `samples` and write them into `out`,
  /// which must be exactly `frame_count(samples.len()) · N_MELS` long,
  /// **time-major**: `out[t · N_MELS + m]`.
  ///
  /// The caller is responsible for the frame-count range check
  /// (`Identifier::log_probabilities`); this function only requires that
  /// `samples` be long enough to fill one frame after centre padding, which
  /// every in-range clip is by a wide margin.
  ///
  /// # Errors
  /// [`Error::NonFiniteInput`](crate::audio::lid::Error::NonFiniteInput) if any
  /// sample is NaN or infinite — it would silently poison every frame it
  /// touches and then, through the whole-utterance `top_db` floor, every frame
  /// it does not.
  pub(crate) fn extract_into(&self, samples: &[f32], out: &mut [f32]) -> Result<()> {
    let frames = super::frame_count(samples.len());
    debug_assert_eq!(out.len(), frames * N_MELS);
    super::check_finite_samples(samples)?;

    // 1. center = True with CONSTANT-ZERO padding (trap 2): N_FFT/2 zeros on
    //    each side, so frame t is centred on sample t·HOP. NOT reflection —
    //    SpeechBrain overrides torch's `pad_mode` default, and reflecting here
    //    would change every frame within 200 samples of either edge.
    let half = N_FFT / 2;
    let mut centered = vec![0.0f64; samples.len() + N_FFT];
    for (dst, &src) in centered[half..half + samples.len()]
      .iter_mut()
      .zip(samples.iter())
    {
      *dst = f64::from(src);
    }

    let mut fft_buffer = vec![Complex::new(0.0f64, 0.0); N_FFT];
    let mut fft_scratch = vec![Complex::new(0.0f64, 0.0); self.fft.get_inplace_scratch_len()];
    let mut power = vec![0.0f64; N_FREQ];
    let mut db = vec![0.0f64; frames * N_MELS];
    let mut db_max = f64::NEG_INFINITY;

    for t in 0..frames {
      let start = t * HOP;
      for ((slot, &sample), &weight) in fft_buffer
        .iter_mut()
        .zip(centered[start..start + N_FFT].iter())
        .zip(self.window.iter())
      {
        *slot = Complex::new(sample * f64::from(weight), 0.0);
      }
      self
        .fft
        .process_with_scratch(&mut fft_buffer, &mut fft_scratch);

      // 2. Power spectrum: re² + im², no 1/√n_fft scaling.
      for (dst, bin) in power.iter_mut().zip(fft_buffer.iter().take(N_FREQ)) {
        *dst = bin.re * bin.re + bin.im * bin.im;
      }

      // 3. Filterbank multiply (trap 3's table) + 4. 10·log10 with the amin
      //    floor. The whole-utterance top_db clamp needs every frame's dB
      //    first, so it runs after this loop.
      let row = &mut db[t * N_MELS..(t + 1) * N_MELS];
      for (m, slot) in row.iter_mut().enumerate() {
        let mut energy = 0.0f64;
        for (k, &bin_power) in power.iter().enumerate() {
          energy += f64::from(self.filterbank[k * N_MELS + m]) * bin_power;
        }
        let decibels = 10.0 * energy.max(AMIN).log10();
        *slot = decibels;
        db_max = db_max.max(decibels);
      }
    }

    // 5. top_db (trap 4): ONE floor for the whole utterance, taken from the
    //    clip's own peak over time AND mel — not a per-frame floor, and not a
    //    constant.
    let floor = db_max - TOP_DB;
    for (dst, &value) in out.iter_mut().zip(db.iter()) {
      *dst = value.max(floor) as f32;
    }

    // ── DO NOT ADD SENTENCE MEAN NORMALIZATION HERE ──────────────────────
    //
    // This is exactly where a faithful port of SpeechBrain's *pipeline* would
    // subtract the per-mel mean over time — `InputNormalization(norm_type=
    // "sentence", std_norm=False)`, and the upstream helper
    // `frontend.py::sentence_mean_normalize` is right there inviting it.
    //
    // It is ALREADY FUSED INTO THE GRAPH. The exported `.mlmodelc` opens with
    // `reduce_mean(axes=[1], keep_dims=True)` followed by `sub`, and its own
    // metadata records `sentence_mean_normalization = included`; the artifact
    // author's README example likewise, and deliberately, never calls that
    // helper. Subtracting here would mean subtracting twice: the output keeps
    // the right shape and stays plausible-looking, and accuracy silently
    // degrades. The features leaving this function must be raw log-mel dB.
    Ok(())
  }
}

#[cfg(test)]
mod tests;
