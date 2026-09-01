//! The identity lane's log-mel front end — the caller-side half of a contract
//! whose CoreML graph deliberately starts one module later.
//!
//! # Why this is in Rust, and why that was measured rather than assumed
//!
//! The natural export is `waveform [1, 96000] -> embedding`: one graph, the
//! whole published function. It converts cleanly and is exact in fp32 (worst
//! cosine 0.99999994 against PyTorch). **It is still wrong in fp16 on every
//! compute unit** — worst cosine 0.9306 `CpuOnly`, 0.9470 `CpuAndGpu`, 0.2770
//! `All`, 0.2769 `CpuAndNeuralEngine` — and it was rejected on that evidence.
//! Isolating the mel alone localizes the damage: the front end is what breaks
//! (0.0463 on the ANE arms), and the network itself is fp16-clean everywhere.
//!
//! The cause is dynamic range failing at both ends of a `power = 2.0`
//! spectrogram. A full-scale tone concentrates ~400 samples of energy into one
//! bin, and the squared magnitude summed across a mel filter passes fp16's
//! 65504 ceiling; at the other end the log guard is `+1e-6`, which is
//! **subnormal** in fp16 (smallest normal 6.10e-5), so hardware that flushes
//! subnormals turns it into `log(0)`. That is the defect class
//! `tests/fp16_guards.rs` and issue #15 already exist for. A `FP16ComputePrecision`
//! op_selector pinning the front end to fp32 was tried and did not move the
//! numbers — coremltools casts the graph *input* to fp16 before the fp32 island
//! — and is recorded as a dead end, not an option. `conversion/ced` and
//! `embeddings::clap` made the same call for the same reason.
//!
//! The full table is in `conversion/redimnet/README.md`.
//!
//! # The parameters, and where they come from
//!
//! Not transcribed from a paper: every one was read out of the checkpoint's
//! live `MelBanks` by `conversion/redimnet/scripts/_redimnet_common.py::assert_front_end`,
//! which fails the conversion run on any mismatch. `MEL_FRONT_END` there is the
//! specification this module implements.
//!
//! | stage | parameters |
//! |---|---|
//! | input | [`WINDOW_SAMPLES`] samples, 16 kHz, mono, `f32` — exactly one window |
//! | pre-emphasis | reflect-pad 1 sample on the left, `y[n] = x[n] − 0.97·x[n−1]` |
//! | STFT | `n_fft 512`, `win_length 400`, `hop 240`, `hamming_window(400, periodic=True)` **zero-padded to 512**, `center=True`, `pad_mode reflect`, `power 2.0` |
//! | mel filterbank | `n_mels 72`, `f_min 20`, `f_max 7600`, `norm=None`, **htk** |
//! | log | `ln(power + 1e-6)` — natural log, ADDITIVE guard |
//! | spec-norm | subtract the per-mel-bin mean over all [`N_FRAMES`] frames |
//! | output | freq-major `[`[`N_MELS`]`, `[`N_FRAMES`]`]` |
//!
//! Four of those are easy to get subtly wrong and are each pinned by a named
//! test in the sibling `tests.rs`: `periodic=True` versus symmetric, **htk**
//! versus slaney *mel scale*, `norm=None` versus slaney *filter normalization*
//! (a different knob with a similar name), and the fact that the window is 400
//! long but sits **zero-padded, centred at offset 56**, inside a 512-point FFT.
//! Two more have no analogue in this crate's other front ends at all: the
//! pre-emphasis filter, and the final per-bin mean subtraction — which is what
//! makes this front end non-local, and so why the door refuses a short clip
//! rather than padding it.
//!
//! # Where this came from, and what was deliberately not shared
//!
//! Structurally this is `embeddings::clap`'s mel — a precomputed window,
//! filterbank and `rustfft` plan built once, scalar `unsafe`-free reductions,
//! `&self` extraction into a caller-owned buffer, f64 throughout — by way of
//! `audio::ced`'s, which is the same shape with an **htk, `norm=None`**
//! filterbank, i.e. this one's. What was NOT taken is the arithmetic that is
//! CLAP's own: its Slaney mel scale, its Slaney filter normalization, its Hann
//! window, and its `10·log10(max(x, 1e-10))` dB conversion — every one of which
//! is a mutation this module's goldens catch, so importing any of them "for
//! consistency" would be a defect rather than reuse.
//!
//! Nothing is shared as *code*, and that is a recorded non-goal rather than an
//! oversight: the three modules' common ground is four small pure functions
//! (`hz_to_htk_mel` and its inverse, `re² + im²`, and a left-to-right dot),
//! each private to a different feature-gated module, so hoisting them would be
//! a cross-cutting refactor of two shipped front ends for a few dozen lines.
//! What IS shared is the shape, which is what keeps the three readable side by
//! side.
//!
//! # Precision
//!
//! f64 throughout, like both of the above. The oracle these goldens come from
//! computes the whole front end in **fp32** and uses the checkpoint's *saved*
//! fp32 hamming window, whose taps sit up to 2.3e-7 from the exact analytic
//! window. So agreement with it is close but not exact, the residual is
//! dominated by the oracle's own precision rather than by this port, and the
//! sibling `tests.rs` pins the MEASURED number and says so rather than choosing
//! a threshold in advance.

use core::fmt;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::audio::identity::{
  N_FRAMES, N_MELS, SAMPLE_RATE_HZ, WINDOW_SAMPLES,
  error::{Error, Result, WindowLength},
};

/// FFT size. Larger than [`WIN_LENGTH`]: the window is zero-padded into it, so
/// the transform interpolates the 400-sample window's spectrum onto 257 bins
/// rather than analysing 512 samples.
const N_FFT: usize = 512;

/// Analysis-window length in samples (25 ms at 16 kHz) — **not** the FFT size.
const WIN_LENGTH: usize = 400;

/// Hop between frame starts in samples (15 ms at 16 kHz). [`N_FRAMES`] is
/// `1 + WINDOW_SAMPLES / HOP`, derived from this rather than written down
/// twice.
pub(super) const HOP: usize = 240;

/// Offset of the analysis window inside the zero-padded FFT frame:
/// `(512 − 400) / 2`, the centring `torch.stft` applies when
/// `win_length < n_fft`.
const WINDOW_OFFSET: usize = (N_FFT - WIN_LENGTH) / 2;

/// Low edge of the mel filterbank, in Hz.
const F_MIN: f64 = 20.0;

/// High edge of the mel filterbank, in Hz. Below Nyquist (8000) on purpose —
/// the top 400 Hz carries no filter.
const F_MAX: f64 = 7600.0;

/// Pre-emphasis coefficient: `y[n] = x[n] − PRE_EMPHASIS · x[n−1]`.
const PRE_EMPHASIS: f64 = 0.97;

/// The log guard, ADDED to the power rather than used as a floor:
/// `ln(power + LOG_EPSILON)`. A `max(power, ε)` floor would be a different
/// function, and one of the mutations the goldens catch.
const LOG_EPSILON: f64 = 1e-6;

/// One-sided spectrum length for [`N_FFT`].
const N_FREQ: usize = N_FFT / 2 + 1;

/// Samples of reflection padding added at each end by `center=True`.
const CENTER_PAD: usize = N_FFT / 2;

/// Mel-spectrogram extractor. Owns the zero-padded hamming window, the mel
/// filterbank and the FFT plan (all immutable after construction); per-call
/// scratch is allocated locally so [`Self::extract_into`] takes `&self`.
pub(crate) struct MelExtractor {
  /// Length [`N_FFT`]: the [`WIN_LENGTH`]-tap hamming window centred at
  /// [`WINDOW_OFFSET`], zeros elsewhere.
  window: Vec<f64>,
  /// `[N_MELS × N_FREQ]`, mel-major row-major.
  filterbank: Vec<f64>,
  /// Forward FFT for [`N_FFT`].
  fft: Arc<dyn Fft<f64>>,
}

impl fmt::Debug for MelExtractor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // `Arc<dyn Fft>` is not `Debug`; the window/filterbank are fixed tables.
    f.debug_struct("MelExtractor").finish_non_exhaustive()
  }
}

impl MelExtractor {
  /// Periodic Hamming window: `w[k] = 0.54 − 0.46·cos(2π·k / n)` for
  /// `k ∈ [0, n)` — `torch.hamming_window(n, periodic=True)`, which is what
  /// `MelBanks` passes as its `window_fn` with torchaudio's default `wkwargs`.
  ///
  /// The `periodic` half of that is load-bearing and not cosmetic: the
  /// symmetric form divides by `n − 1`, which moves every tap and is one of the
  /// mutations the goldens catch. Hamming rather than Hann is the other half —
  /// same shape, different coefficients, and a far lower first sidelobe.
  fn periodic_hamming(n: usize) -> Vec<f64> {
    let denom = n as f64;
    (0..n)
      .map(|k| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * (k as f64) / denom).cos())
      .collect()
  }

  /// The [`N_FFT`]-long analysis window: [`Self::periodic_hamming`] of
  /// [`WIN_LENGTH`] taps written at [`WINDOW_OFFSET`], zeros elsewhere.
  ///
  /// This is `torch.stft`'s own handling of `win_length < n_fft` — it centres
  /// the window in an `n_fft`-long buffer of zeros — and it is a genuinely
  /// different computation from a 512-tap window, not an approximation of one:
  /// the effective analysis length stays 25 ms while the transform's frequency
  /// grid is that of a 512-point FFT.
  fn padded_window() -> Vec<f64> {
    let taps = Self::periodic_hamming(WIN_LENGTH);
    let mut window = vec![0.0f64; N_FFT];
    window[WINDOW_OFFSET..WINDOW_OFFSET + WIN_LENGTH].copy_from_slice(&taps);
    window
  }

  /// Hz → HTK mel: `2595·log10(1 + f/700)` — torchaudio `mel_scale="htk"`.
  fn hz_to_htk_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
  }

  /// HTK mel → Hz (inverse of [`Self::hz_to_htk_mel`]).
  fn htk_mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
  }

  /// Build the `[n_mels × n_freq]` HTK-scale, **unnormalized** (`norm=None`)
  /// triangular mel filterbank, mel-major row-major.
  ///
  /// This is torchaudio's `melscale_fbanks(n_freqs, f_min, f_max, n_mels, sr,
  /// norm=None, mel_scale="htk")` transposed to mel-major rows, written in its
  /// own `max(0, min(down, up))` form so the triangle's edges are the same
  /// expression rather than an equivalent-looking rewrite.
  fn build_htk_filterbank(sr: u32, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let n_freq = n_fft / 2 + 1;
    // torchaudio spells this `linspace(0, sr // 2, n_freqs)`; on this geometry
    // that is exactly `k · sr / n_fft`, and both are exact in binary here.
    let all_freqs: Vec<f64> = (0..n_freq)
      .map(|k| (k as f64) * (f64::from(sr) / 2.0) / ((n_freq - 1) as f64))
      .collect();

    let mel_min = Self::hz_to_htk_mel(fmin);
    let mel_max = Self::hz_to_htk_mel(fmax);
    // n_mels + 2 mel-equispaced points: one per filter plus the two outer edges.
    let f_pts: Vec<f64> = (0..n_mels + 2)
      .map(|i| {
        let mel = mel_min + (mel_max - mel_min) * (i as f64) / ((n_mels + 1) as f64);
        Self::htk_mel_to_hz(mel)
      })
      .collect();

    let mut fb = vec![0.0f64; n_mels * n_freq];
    for m in 0..n_mels {
      let left_diff = f_pts[m + 1] - f_pts[m];
      let right_diff = f_pts[m + 2] - f_pts[m + 1];
      for (k, &f) in all_freqs.iter().enumerate() {
        let down = (f - f_pts[m]) / left_diff;
        let up = (f_pts[m + 2] - f) / right_diff;
        // NO `norm='slaney'` 2/(right − left) scaling: `MelBanks` builds this
        // filterbank with `norm=None`, so the triangles peak at 1.0 and a wide
        // high-frequency filter contributes proportionally more energy than a
        // narrow low-frequency one. Adding the scaling is a mutation the
        // goldens catch.
        fb[m * n_freq + k] = down.min(up).max(0.0);
      }
    }
    fb
  }

  pub(crate) fn new() -> Self {
    let window = Self::padded_window();
    let filterbank = Self::build_htk_filterbank(SAMPLE_RATE_HZ, N_FFT, N_MELS, F_MIN, F_MAX);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    Self {
      window,
      filterbank,
      fft,
    }
  }

  /// `|X[k]|² = re² + im²` for the first [`N_FREQ`] bins (real-FFT identity) —
  /// the scalar naive form, as in the clap/ced front ends.
  fn power_spectrum(fft_input: &[Complex<f64>], power: &mut [f64]) {
    for (dst, c) in power.iter_mut().zip(fft_input.iter().take(N_FREQ)) {
      *dst = c.re * c.re + c.im * c.im;
    }
  }

  /// `Σ weights[i]·power[i]`, left-to-right f64 accumulation.
  fn mel_filterbank_dot(weights: &[f64], power: &[f64]) -> f64 {
    weights.iter().zip(power.iter()).map(|(w, p)| w * p).sum()
  }

  /// Window one [`N_FFT`]-sample frame, forward-FFT it (f64), and write its
  /// power spectrum into `power` (length [`N_FREQ`]). `fft_input` /
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

  /// Pre-emphasis with a 1-sample reflection pad on the left, in f64.
  ///
  /// `PreEmphasis` pads with `F.pad(x, (1, 0), 'reflect')` and then correlates
  /// with `[-0.97, 1.0]`, so the FIRST output is `x[0] − 0.97·x[1]` — the
  /// reflected sample is `x[1]`, not a repeat of `x[0]`. Everything after it is
  /// the ordinary `x[n] − 0.97·x[n−1]`. Getting that first sample wrong is
  /// invisible in a spectrogram and shifts one frame's worth of low-frequency
  /// energy, so it is spelled out rather than folded into the loop.
  fn pre_emphasize(samples: &[f32]) -> Vec<f64> {
    debug_assert!(samples.len() >= 2, "the fixed window is far longer than 2");
    let mut out = Vec::with_capacity(samples.len());
    out.push(f64::from(samples[0]) - PRE_EMPHASIS * f64::from(samples[1]));
    for pair in samples.windows(2) {
      out.push(f64::from(pair[1]) - PRE_EMPHASIS * f64::from(pair[0]));
    }
    out
  }

  /// `center=True` reflection padding: [`CENTER_PAD`] samples mirrored onto
  /// each end, so frame `t` is centred at sample `t · HOP` of the input.
  ///
  /// The reflection excludes the edge sample itself (`torch`'s `'reflect'`, not
  /// `'replicate'`): the left pad reads `y[CENTER_PAD] … y[1]` and the right
  /// pad reads `y[len − 2] … y[len − 1 − CENTER_PAD]`.
  fn center_pad(signal: &[f64]) -> Vec<f64> {
    let len = signal.len();
    debug_assert!(len > CENTER_PAD, "reflect padding needs len > pad");
    let mut padded = Vec::with_capacity(len + 2 * CENTER_PAD);
    for i in 0..CENTER_PAD {
      padded.push(signal[CENTER_PAD - i]);
    }
    padded.extend_from_slice(signal);
    for i in 0..CENTER_PAD {
      padded.push(signal[len - 2 - i]);
    }
    padded
  }

  /// Compute the log-mel features for `samples` and write them into `out`
  /// (length exactly `N_MELS × N_FRAMES`, **freq-major**:
  /// `out[mel · N_FRAMES + t]`, which is the graph's `mel [1, 72, 401]`
  /// row-major layout with the batch axis dropped).
  ///
  /// `samples` must be exactly [`WINDOW_SAMPLES`] long. It is not padded and
  /// not truncated — see [`WindowLength`] for why this front end, unlike the
  /// crate's others, cannot offer either.
  ///
  /// # Errors
  /// [`Error::WindowLength`] if `samples` is not exactly [`WINDOW_SAMPLES`]
  /// samples.
  pub(crate) fn extract_into(&self, samples: &[f32], out: &mut [f32]) -> Result<()> {
    debug_assert_eq!(out.len(), N_MELS * N_FRAMES);
    if samples.len() != WINDOW_SAMPLES {
      return Err(Error::WindowLength(WindowLength::new(
        samples.len(),
        WINDOW_SAMPLES,
      )));
    }

    // 1. pre-emphasis, then 2. center=True reflection padding. The order is the
    //    checkpoint's: `MelBanks` pre-emphasizes the whole window and hands the
    //    RESULT to `MelSpectrogram`, which does its own centring — so the
    //    reflected pad mirrors pre-emphasized samples, not raw ones.
    let emphasized = Self::pre_emphasize(samples);
    let padded = Self::center_pad(&emphasized);
    debug_assert_eq!(padded.len(), WINDOW_SAMPLES + 2 * CENTER_PAD);

    // 3. STFT -> 4. filterbank -> 5. ln(power + eps), into an f64 scratch so
    //    the per-bin mean (6.) is taken before the f32 narrowing.
    let mut frame = vec![0.0f64; N_FFT];
    let mut power = vec![0.0f64; N_FREQ];
    let mut fft_input = vec![Complex::new(0.0f64, 0.0); N_FFT];
    let mut fft_scratch = vec![Complex::new(0.0f64, 0.0); self.fft.get_inplace_scratch_len()];
    let mut log_mel = vec![0.0f64; N_MELS * N_FRAMES];

    for t in 0..N_FRAMES {
      let start = t * HOP;
      // The last frame ends at 400·240 + 512 = 96 512 = the padded length.
      frame.copy_from_slice(&padded[start..start + N_FFT]);
      self.stft_one_frame_power(&frame, &mut fft_input, &mut fft_scratch, &mut power);

      for mel_bin in 0..N_MELS {
        let row = &self.filterbank[mel_bin * N_FREQ..(mel_bin + 1) * N_FREQ];
        let acc = Self::mel_filterbank_dot(row, &power);
        // ADDITIVE guard and a NATURAL log: `MelBanks` computes
        // `(fbank(x) + 1e-6).log()`. Not `10·log10`, and not `max(x, eps)`.
        log_mel[mel_bin * N_FRAMES + t] = (acc + LOG_EPSILON).ln();
      }
    }

    // 6. spec_norm='mn': subtract each mel bin's own mean over ALL frames.
    //    Over the TIME axis, per bin — the transposed reading (a per-frame mean
    //    over the mel axis) is a different function and one the goldens catch.
    //    This is also what makes the front end non-local, and so why a short
    //    clip is refused above rather than zero-padded.
    for mel_bin in 0..N_MELS {
      let row = &log_mel[mel_bin * N_FRAMES..(mel_bin + 1) * N_FRAMES];
      let mean = row.iter().sum::<f64>() / (N_FRAMES as f64);
      let dst = &mut out[mel_bin * N_FRAMES..(mel_bin + 1) * N_FRAMES];
      for (o, &v) in dst.iter_mut().zip(row.iter()) {
        *o = (v - mean) as f32;
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests;
