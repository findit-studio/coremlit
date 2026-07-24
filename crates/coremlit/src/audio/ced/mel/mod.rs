//! Log-mel spectrogram front-end for the CED transformer — the same Rust-mel
//! pattern as the clap/whisper front-ends (`embeddings/clap/audio/mel/` is the
//! implementation template): precomputed Hann window + filterbank + `rustfft`
//! plan, scalar `unsafe`-free reductions, `&self` extraction.
//!
//! # Why a Rust front-end at all
//!
//! The ANE handles the ViT-class transformer well and FFT/mel poorly; an
//! in-graph STFT is the exact fragility class behind the ORT CoreML EP
//! zeroed-logits bug and clap's rejected in-graph mel (spec decision 2). The
//! CoreML graph therefore starts AT the mel.
//!
//! # BELIEVED numerics — probe-pinned in Wave B (spec §2 / §12 item 1)
//!
//! Every constant below is a believed value from upstream source reading
//! (torchaudio `MelSpectrogram` + `AmplitudeToDB` family, RicherMans/CED):
//! 16 kHz, `n_fft = win_length = 512` (32 ms), `hop = 160` (10 ms),
//! `n_mels = 64`, `f_min = 0` / `f_max = 8000`, HTK mel scale, **no**
//! filterbank norm, power 2.0, `center=True` reflection padding, periodic
//! Hann, `AmplitudeToDB(stype="power", top_db=120)`:
//! `10·log10(max(x, 1e-10))` then the per-window clamp `max(db, db_max − 120)`
//! — the floor is coupled to the window's own peak. They are implementation
//! guidance ONLY until the §6 conversion probe pins them; a probe divergence
//! is a constants + golden change by design (the recorded rework seam). The
//! §8 structural tests (sibling `tests.rs`) are the Wave-A guards; the
//! committed-golden mel parity lands in Wave B.
//!
//! Output is **freq-major** `[N_MELS, N_FRAMES]` (`out[mel * N_FRAMES + t]`)
//! — the believed `[1, n_mels, T]` torchaudio layout the graph consumes; the
//! probe decides the final orientation. Compute runs in f64 (the clap mel
//! infrastructure; CED's front-end is f32-native upstream — the Wave-B parity
//! measurement decides whether f32 is required for the budget).
//!
//! # `N_FRAMES = 1001` vs upstream `target_length = 1012`
//!
//! Upstream CED's `target_length = 1012` (RicherMans/CED `audiotransformer.py`)
//! is NOT the input length: it is the transformer's time positional-embedding
//! *capacity* (`AudioPatchEmbed(input_size=(64, 1012))`, 16×16 patches ⇒ 63
//! time-patch columns) and its long-form mel *chunk size* (mels longer than
//! 1012 frames split into 1012-frame chunks, last padded/dropped, logits
//! averaged). The fixed 10 s window is 160 000 samples ⇒ `1 + 160_000/160 =
//! 1001` frames (`center=True`) ⇒ `(1001−16)/16 + 1 = 62` of those 63 patch
//! columns; upstream runs a ≤ 1012-frame mel **unpadded** with the pos embed
//! sliced to the actual patch count, dropping the trailing 9 mel frames past
//! `62×16 = 992` in the patch conv — exactly what any 10 s clip does. Padding
//! 1001 → 1012 would add a 63rd column and compute a *different* function, so
//! [`N_FRAMES`] stays 1001; the relation `N_FRAMES <= 1012` (pinned in the
//! sibling `tests.rs`) is what makes a fixed `[1, 64, 1001]` export a faithful
//! evaluation of the upstream model. Verified against RicherMans/CED
//! `audiotransformer.py` and the mispeech feature extractor (which never pads a
//! 10 s clip to 1012). Shared, unchanged, across all four CED sizes.

use core::fmt;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::audio::ced::{
  WINDOW_SAMPLES,
  error::{Error, Result},
};

/// Mel-frequency bin count — the graph's believed input height. BELIEVED —
/// probe-pinned (Wave B).
pub(crate) const N_MELS: usize = 64;

/// Mel time-frame count for the fixed window at hop 160 with `center=True`
/// (`1 + 160_000/160 = 1001`) — the graph's believed input width. BELIEVED —
/// probe-pinned (Wave B).
pub(crate) const N_FRAMES: usize = 1 + WINDOW_SAMPLES / HOP;

// BELIEVED front-end constants — probe-pinned (Wave B); see the module docs.
const N_FFT: usize = 512; // == win_length (32 ms at 16 kHz)
const HOP: usize = 160; // 10 ms
const SR: u32 = 16_000;
const FMIN: f64 = 0.0;
const FMAX: f64 = 8_000.0; // sr/2, the torchaudio f_max=None default
const AMIN: f64 = 1e-10; // AmplitudeToDB power floor ⇒ −100 dB
const TOP_DB: f64 = 120.0; // per-window dynamic-range clamp
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
  /// the `torch.hann_window(n, periodic=True)` convention.
  fn periodic_hann(n: usize) -> Vec<f64> {
    let denom = n as f64;
    (0..n)
      .map(|k| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (k as f64) / denom).cos())
      .collect()
  }

  /// Hz → HTK mel: `2595·log10(1 + f/700)` — torchaudio `mel_scale="htk"`.
  fn hz_to_htk_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
  }

  /// HTK mel → Hz (inverse of [`Self::hz_to_htk_mel`]).
  fn htk_mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
  }

  /// Build a `[n_mels × n_freq]` HTK-scale, UNNORMALIZED (`norm=None`)
  /// triangular mel filterbank (row-major, f64) — torchaudio
  /// `melscale_fbanks(n_freqs, f_min, f_max, n_mels, sr, norm=None,
  /// mel_scale="htk")`, transposed to mel-major rows.
  fn build_htk_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let n_freq = n_fft / 2 + 1;
    let mel_min = Self::hz_to_htk_mel(fmin);
    let mel_max = Self::hz_to_htk_mel(fmax);
    let mel_points: Vec<f64> = (0..n_mels + 2)
      .map(|i| mel_min + (mel_max - mel_min) * (i as f64) / (n_mels + 1) as f64)
      .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| Self::htk_mel_to_hz(m)).collect();
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
      for (k, &f) in bin_hz.iter().enumerate() {
        let weight = if f >= left && f <= center {
          (f - left) * inv_left_diff
        } else if f >= center && f <= right {
          (right - f) * inv_right_diff
        } else {
          0.0
        };
        fb[m * n_freq + k] = weight;
      }
    }
    fb
  }

  pub(crate) fn new() -> Self {
    let window = Self::periodic_hann(N_FFT);
    let filterbank = Self::build_htk_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    Self {
      window,
      filterbank,
      fft,
    }
  }

  /// `|X[k]|² = re² + im²` for the first `N_FREQ` bins (real-FFT identity) —
  /// the scalar naive form (the clap mel convention).
  fn power_spectrum(fft_input: &[Complex<f64>], power: &mut [f64]) {
    for (dst, c) in power.iter_mut().zip(fft_input.iter().take(N_FREQ)) {
      *dst = c.re * c.re + c.im * c.im;
    }
  }

  /// `Σ weights[i]·power[i]`, left-to-right f64 accumulation.
  fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
    weights.iter().zip(power.iter()).map(|(w, p)| w * p).sum()
  }

  /// Window one `N_FFT`-sample frame, forward-FFT it (f64), and write its
  /// power spectrum into `power` (length `N_FREQ`). `fft_input` /
  /// `fft_scratch` are caller-owned reusable buffers.
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
  /// (length exactly `N_MELS × N_FRAMES`, **freq-major**:
  /// `out[mel · N_FRAMES + t]`).
  ///
  /// `samples` must be `1..=`[`WINDOW_SAMPLES`] long; a shorter input is
  /// **zero-padded at the waveform** up to the fixed window (the believed
  /// sub-window policy, probe-pinned), then `center=True` reflection-padded.
  ///
  /// # Errors
  /// [`Error::EmptyAudio`] if `samples` is empty; [`Error::AudioTooLong`] if
  /// it exceeds [`WINDOW_SAMPLES`] (never silently truncated — the classifier
  /// windows long clips explicitly).
  pub(crate) fn extract_into(&self, samples: &[f32], out: &mut [f32]) -> Result<()> {
    debug_assert_eq!(out.len(), N_MELS * N_FRAMES);
    if samples.is_empty() {
      return Err(Error::EmptyAudio);
    }
    if samples.len() > WINDOW_SAMPLES {
      return Err(Error::AudioTooLong {
        len: samples.len(),
        max: WINDOW_SAMPLES,
      });
    }

    // 1. Zero-pad to the fixed window (believed policy; probe-ratified or
    //    replaced in Wave B).
    let mut padded: Vec<f64> = Vec::with_capacity(WINDOW_SAMPLES);
    padded.extend(samples.iter().map(|&s| s as f64));
    padded.resize(WINDOW_SAMPLES, 0.0);

    // 2. center=True reflection padding: prepend + append N_FFT/2 reflected
    //    samples so frame t is centered at sample t·HOP.
    let half_fft = N_FFT / 2;
    let mut centered: Vec<f64> = Vec::with_capacity(WINDOW_SAMPLES + 2 * half_fft);
    for i in 0..half_fft {
      centered.push(padded[half_fft - i]);
    }
    centered.extend_from_slice(&padded);
    for i in 0..half_fft {
      centered.push(padded[WINDOW_SAMPLES - 2 - i]);
    }
    debug_assert_eq!(centered.len(), WINDOW_SAMPLES + 2 * half_fft);

    // 3. STFT loop → 4. filterbank multiply → 5. power_to_db floor, into an
    //    f64 scratch so the per-window top_db clamp (6.) applies before the
    //    f32 narrowing.
    let mut frame = vec![0.0f64; N_FFT];
    let mut power = vec![0.0f64; N_FREQ];
    let mut fft_input = vec![Complex::new(0.0f64, 0.0); N_FFT];
    let mut fft_scratch = vec![Complex::new(0.0f64, 0.0); self.fft.get_inplace_scratch_len()];
    let mut db = vec![0.0f64; N_MELS * N_FRAMES];
    let mut db_max = f64::MIN;

    for t in 0..N_FRAMES {
      let start = t * HOP;
      // Last frame ends at 1000·160 + 512 = 160_512 = WINDOW_SAMPLES + N_FFT.
      frame.copy_from_slice(&centered[start..start + N_FFT]);
      self.stft_one_frame_power(&frame, &mut fft_input, &mut fft_scratch, &mut power);

      for mel_bin in 0..N_MELS {
        let row = &self.filterbank[mel_bin * N_FREQ..(mel_bin + 1) * N_FREQ];
        let acc = Self::mel_filterbank_dot(row, &power);
        // AmplitudeToDB power form: 10·log10(max(·, amin)), ref 1.0.
        let v = 10.0 * acc.max(AMIN).log10();
        db[mel_bin * N_FRAMES + t] = v;
        db_max = db_max.max(v);
      }
    }

    // 6. top_db: clamp the floor to the window's own peak minus 120 dB — the
    //    per-example max coupling (believed; probe-pinned).
    let floor = db_max - TOP_DB;
    for (dst, &v) in out.iter_mut().zip(db.iter()) {
      *dst = v.max(floor) as f32;
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests;
