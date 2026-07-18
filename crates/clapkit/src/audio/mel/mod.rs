//! Log-mel spectrogram front-end for the CLAP audio tower — a Rust port of
//! textclap's `mel.rs`, adapted to a scalar (`unsafe`-free) implementation.
//!
//! # Why a Rust front-end at all
//!
//! T1's conversion rejected an in-graph mel/STFT: a faithful HF
//! `ClapFeatureExtractor` needs **float64** STFT numerics, which an fp16 ANE
//! graph cannot carry (worst end-to-end cosine 0.5546 for an in-graph f32 STFT),
//! and `power_to_db`'s `amin = 1e-10` floor is the exact fp16-vanishing-guard
//! class the campaign forbids. So the graph takes a spectrogram
//! (`input_features [1, 1, 1001, 64]`, perfect fp32 parity) and the mel is a
//! Rust port, bit-validated against textclap's mel (the spec's stated oracle).
//!
//! # Numerics (== textclap; == HF `ClapFeatureExtractor`)
//!
//! `n_fft 1024`, `hop 480`, `n_mels 64`, `fmin 50`, `fmax 14000`, periodic Hann,
//! Slaney-scale Slaney-norm filterbank, `center=True` reflection padding,
//! `repeatpad` to 480 000 samples, `10·log10(max(·, 1e-10))`, time-major
//! `[1001, 64]` output, HTSAT input-norm `none`. The FFT runs in **f64** (via
//! `rustfft`, the crate textclap uses) because HF promotes to float64 before the
//! STFT; f32 leaves ~1.24e-4 drift that exceeds the mel budget.
//!
//! # Divergence from textclap: scalar only
//!
//! textclap dispatches the power-spectrum and filterbank-dot kernels to SIMD
//! backends. clapkit keeps the crate `unsafe`-free and uses the scalar reductions
//! — which are **bit-identical to textclap's own `scalar` backend** (the naive
//! `re² + im²` and the left-to-right f64 sum). textclap's default aarch64 path is
//! its NEON backend (FMA + 2× ILP), which reassociates the filterbank sum and so
//! differs from this scalar port by at most `~1e-10·scale` — far below the
//! `1e-4` mel budget and utterly below the fp16 resolution of the downstream
//! graph. `tests/mel_parity.rs` measures and pins the actual agreement against
//! textclap's committed golden mel.

use core::fmt;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::error::{Error, Result};

/// Mel time-frame count for a 480 000-sample window at hop 480 with
/// `center=True` (`1 + 480000/480 = 1001`). The audio graph's `input_features`
/// leading spatial dim. Re-exported from [`crate::audio`].
pub const T_FRAMES: usize = 1001;

/// Mel-frequency bin count — the audio graph's `input_features` trailing dim.
/// Re-exported from [`crate::audio`].
pub const N_MELS: usize = 64;

/// Fixed audio window: 10 s at 48 kHz. Re-exported from [`crate::audio`].
pub const TARGET_SAMPLES: usize = 480_000;

const N_FFT: usize = 1024;
const HOP: usize = 480;
const SR: u32 = 48_000;
const FMIN: f64 = 50.0;
const FMAX: f64 = 14_000.0;
const POWER_TO_DB_AMIN: f64 = 1e-10;
const N_FREQ: usize = N_FFT / 2 + 1;

/// Mel-spectrogram extractor. Owns the Hann window, mel filterbank, and FFT
/// plan (all immutable after construction); per-call scratch is allocated
/// locally so [`Self::extract_into`] takes `&self`.
pub(crate) struct MelExtractor {
  window: Vec<f64>,       // length N_FFT
  filterbank: Vec<f64>,   // length N_MELS × N_FREQ, row-major
  fft: Arc<dyn Fft<f64>>, // forward FFT for N_FFT
}

impl fmt::Debug for MelExtractor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // `Arc<dyn Fft>` is not `Debug`; the window/filterbank are fixed tables.
    f.debug_struct("MelExtractor").finish_non_exhaustive()
  }
}

impl MelExtractor {
  /// Periodic Hann window: `w[k] = 0.5 − 0.5·cos(2π·k / n)` for `k ∈ [0, n)` —
  /// the `torch.hann_window(n, periodic=True)` / `numpy.hanning(n+1)[:-1]`
  /// convention HF uses.
  fn periodic_hann(n: usize) -> Vec<f64> {
    let denom = n as f64;
    (0..n)
      .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (k as f64) / denom).cos())
      .collect()
  }

  /// Hz → Slaney mel (linear below 1 kHz, logarithmic above). Matches librosa
  /// `mel_frequencies(htk=False)`.
  fn hz_to_slaney_mel(hz: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f64).ln() / 27.0;
    if hz < MIN_LOG_HZ {
      (hz - F_MIN) / F_SP
    } else {
      MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    }
  }

  /// Slaney mel → Hz (inverse of [`Self::hz_to_slaney_mel`]).
  fn slaney_mel_to_hz(mel: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f64).ln() / 27.0;
    if mel < MIN_LOG_MEL {
      F_MIN + F_SP * mel
    } else {
      MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    }
  }

  /// Build a `[n_mels × n_freq]` Slaney-norm Slaney-scale mel filterbank
  /// (row-major, f64). Matches
  /// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False,
  /// norm='slaney')`.
  fn build_mel_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let n_freq = n_fft / 2 + 1;
    let mel_min = Self::hz_to_slaney_mel(fmin);
    let mel_max = Self::hz_to_slaney_mel(fmax);
    let mel_points: Vec<f64> = (0..n_mels + 2)
      .map(|i| mel_min + (mel_max - mel_min) * (i as f64) / (n_mels + 1) as f64)
      .collect();
    let hz_points: Vec<f64> = mel_points
      .iter()
      .map(|&m| Self::slaney_mel_to_hz(m))
      .collect();
    let bin_hz: Vec<f64> = (0..n_freq)
      .map(|k| (k as f64) * (sr as f64) / (n_fft as f64))
      .collect();

    let mut fb = vec![0.0f64; n_mels * n_freq];
    for m in 0..n_mels {
      let left = hz_points[m];
      let center = hz_points[m + 1];
      let right = hz_points[m + 2];
      let inv_left_diff = 1.0 / (center - left);
      let inv_right_diff = 1.0 / (right - center);
      let slaney_norm = 2.0 / (right - left);
      for (k, &f) in bin_hz.iter().enumerate() {
        let weight = if f >= left && f <= center {
          (f - left) * inv_left_diff
        } else if f >= center && f <= right {
          (right - f) * inv_right_diff
        } else {
          0.0
        };
        fb[m * n_freq + k] = weight * slaney_norm;
      }
    }
    fb
  }

  pub(crate) fn new() -> Self {
    let window = Self::periodic_hann(N_FFT);
    let filterbank = Self::build_mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    Self {
      window,
      filterbank,
      fft,
    }
  }

  /// `|X[k]|² = re² + im²` for the first `N_FREQ` bins (real-FFT identity) —
  /// the naive form, bit-identical to textclap's scalar backend.
  fn power_spectrum(fft_input: &[Complex<f64>], power: &mut [f64]) {
    for (dst, c) in power.iter_mut().zip(fft_input.iter().take(N_FREQ)) {
      *dst = c.re * c.re + c.im * c.im;
    }
  }

  /// `Σ weights[i]·power[i]`, left-to-right f64 accumulation — textclap's scalar
  /// filterbank dot.
  fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
    weights.iter().zip(power.iter()).map(|(w, p)| w * p).sum()
  }

  /// Window one `N_FFT`-sample frame, forward-FFT it (f64), and write its power
  /// spectrum into `power` (length `N_FREQ`). `fft_input` / `fft_scratch` are
  /// caller-owned reusable buffers.
  fn stft_one_frame_power(
    &self,
    frame: &[f64],
    fft_input: &mut [Complex<f64>],
    fft_scratch: &mut [Complex<f64>],
    power: &mut [f64],
  ) {
    for ((dst, &s), &w) in fft_input
      .iter_mut()
      .zip(frame.iter())
      .zip(self.window.iter())
    {
      *dst = Complex::new(s * w, 0.0);
    }
    self.fft.process_with_scratch(fft_input, fft_scratch);
    Self::power_spectrum(fft_input, power);
  }

  /// Compute the log-mel features for `samples` and write them into `out`
  /// (length exactly `N_MELS × T_FRAMES`, time-major: `out[t·64 + mel]`).
  ///
  /// `samples` is `repeatpad`ed (or head-truncated) to [`TARGET_SAMPLES`],
  /// then `center=True` reflection-padded — exactly HF's `ClapFeatureExtractor`.
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples` is empty (the repeatpad branch would
  /// otherwise divide by zero).
  pub(crate) fn extract_into(&self, samples: &[f32], out: &mut [f32]) -> Result<()> {
    debug_assert_eq!(out.len(), N_MELS * T_FRAMES);
    if samples.is_empty() {
      return Err(Error::EmptyAudio);
    }

    // 1. repeatpad / head-truncate to TARGET_SAMPLES (HF `repeatpad`):
    //    tile the clip an integer number of times, then zero-pad the remainder.
    let mut padded: Vec<f64> = Vec::with_capacity(TARGET_SAMPLES);
    if samples.len() >= TARGET_SAMPLES {
      padded.extend(samples[..TARGET_SAMPLES].iter().map(|&s| s as f64));
    } else {
      let n_repeat = TARGET_SAMPLES / samples.len();
      for _ in 0..n_repeat {
        padded.extend(samples.iter().map(|&s| s as f64));
      }
      padded.resize(TARGET_SAMPLES, 0.0);
    }

    // 2. center=True reflection padding: prepend + append N_FFT/2 reflected
    //    samples so the first frame is centered at sample 0.
    let half_fft = N_FFT / 2;
    let mut centered: Vec<f64> = Vec::with_capacity(TARGET_SAMPLES + 2 * half_fft);
    for i in 0..half_fft {
      centered.push(padded[half_fft - i]);
    }
    centered.extend_from_slice(&padded);
    for i in 0..half_fft {
      centered.push(padded[TARGET_SAMPLES - 2 - i]);
    }
    debug_assert_eq!(centered.len(), TARGET_SAMPLES + 2 * half_fft);

    // 3. STFT loop → 4. filterbank multiply → 5. power_to_db floor.
    let mut frame = vec![0.0f64; N_FFT];
    let mut power = vec![0.0f64; N_FREQ];
    let mut fft_input = vec![Complex::new(0.0f64, 0.0); N_FFT];
    let mut fft_scratch = vec![Complex::new(0.0f64, 0.0); self.fft.get_inplace_scratch_len()];

    for t in 0..T_FRAMES {
      let start = t * HOP;
      // Last frame ends at 1000·480 + 1024 = 481024 = TARGET_SAMPLES + N_FFT.
      frame.copy_from_slice(&centered[start..start + N_FFT]);
      self.stft_one_frame_power(&frame, &mut fft_input, &mut fft_scratch, &mut power);

      for mel_bin in 0..N_MELS {
        let row = &self.filterbank[mel_bin * N_FREQ..(mel_bin + 1) * N_FREQ];
        let acc = Self::mel_filterbank_dot(row, &power);
        // Single 10·log10 application, ref=1.0, amin=1e-10 ⇒ −100 dB floor.
        let db = 10.0 * acc.max(POWER_TO_DB_AMIN).log10();
        out[t * N_MELS + mel_bin] = db as f32;
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests;
