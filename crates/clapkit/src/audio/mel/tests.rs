use rustfft::num_complex::Complex;

use super::*;

/// Read a committed `.npy` of f32 into a flat `Vec<f32>`.
fn read_npy_f32(path: &str) -> Vec<f32> {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
  npyz::NpyFile::new(&bytes[..])
    .unwrap()
    .into_vec::<f32>()
    .unwrap()
}

/// Read `tests/fixtures/mel/sample.wav` (48 kHz mono) into normalized f32.
fn read_sample_wav() -> Vec<f32> {
  let mut reader =
    hound::WavReader::open("tests/fixtures/mel/sample.wav").expect("open sample.wav");
  match reader.spec().sample_format {
    hound::SampleFormat::Int => {
      let bits = reader.spec().bits_per_sample;
      let scale = 1.0 / (1_i64 << (bits - 1)) as f32;
      reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 * scale)
        .collect()
    }
    hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
  }
}

/// Periodic Hann at n=1024: peak (1.0) exactly at index n/2, last sample small
/// but POSITIVE (distinguishing periodic from symmetric, which would be 0).
#[test]
fn hann_window_periodic_length_1024() {
  let win = MelExtractor::periodic_hann(1024);
  assert_eq!(win.len(), 1024);
  assert_eq!(win[0], 0.0);
  assert!(
    win[1023] > 0.0 && win[1023] < 1e-3,
    "periodic Hann last sample should be positive but small; got {}",
    win[1023]
  );
  for &v in &win {
    assert!((0.0..=1.0 + 1e-7).contains(&v));
  }
  let max_idx = win
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .unwrap()
    .0;
  assert_eq!(max_idx, 512, "peak must be exactly at index N/2 = 512");
  assert_eq!(win[512], 1.0);
  assert!(win[513] < 1.0 && win[513] > 0.999);
}

/// STFT of a 1 kHz sine at 48 kHz should peak at the bin closest to
/// `1000 / (48000/1024) = 21.33` → bin 21 (or 22).
#[test]
fn stft_peaks_at_expected_bin() {
  let mel = MelExtractor::new();
  let sr = 48_000_f64;
  let freq = 1000.0_f64;
  let frame: Vec<f64> = (0..N_FFT)
    .map(|k| (2.0 * std::f64::consts::PI * freq * (k as f64) / sr).sin())
    .collect();
  let mut power = vec![0.0f64; N_FREQ];
  let mut fft_input = vec![Complex::new(0.0f64, 0.0); N_FFT];
  let mut fft_scratch = vec![Complex::new(0.0f64, 0.0); mel.fft.get_inplace_scratch_len()];
  mel.stft_one_frame_power(&frame, &mut fft_input, &mut fft_scratch, &mut power);
  let (peak_bin, _) = power
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .unwrap();
  assert!(
    peak_bin == 21 || peak_bin == 22,
    "expected peak at bin 21 or 22, got {peak_bin}"
  );
}

/// `power_spectrum` is the exact `re² + im²` (3-4-5 triangle → 25).
#[test]
fn power_spectrum_is_exact_magnitude_squared() {
  let mut input = vec![Complex::new(0.0f64, 0.0); N_FFT];
  input[0] = Complex::new(3.0, 4.0);
  let mut power = vec![0.0f64; N_FREQ];
  MelExtractor::power_spectrum(&input, &mut power);
  assert_eq!(power[0], 25.0);
}

/// A single 10·log10 application: a unit 1 kHz sine peaks near 29.3 dB with a
/// −100 dB floor. A double application would compress the peak to ~14.6 dB or
/// NaN; a missing log10 would push it above 50 (raw power reaches ~1e5).
#[test]
fn power_to_db_applied_once() {
  let mel = MelExtractor::new();
  let sr = 48_000_f32;
  let samples: Vec<f32> = (0..TARGET_SAMPLES)
    .map(|k| (2.0 * std::f32::consts::PI * 1000.0 * (k as f32) / sr).sin())
    .collect();
  let mut out = vec![0.0f32; N_MELS * T_FRAMES];
  mel.extract_into(&samples, &mut out).unwrap();
  let max = out.iter().fold(f32::MIN, |a, &b| a.max(b));
  let min = out.iter().fold(f32::MAX, |a, &b| a.min(b));
  assert!(
    max > 20.0 && max < 50.0,
    "unit-sine mel should peak near 29.3 dB; got max = {max}"
  );
  assert!(
    (-100.0 - 1e-3..-50.0).contains(&min),
    "amin floor should clip silent bins to -100 dB; got min = {min}"
  );
}

/// Empty input is rejected explicitly (the repeatpad branch would otherwise
/// divide by `samples.len() == 0`).
#[test]
fn extract_into_rejects_empty_input() {
  let mel = MelExtractor::new();
  let mut out = vec![0.0f32; N_MELS * T_FRAMES];
  let err = mel.extract_into(&[], &mut out).unwrap_err();
  assert!(matches!(err, Error::EmptyAudio), "got {err:?}");
}

/// A clip shorter than the window is repeat-tiled (not zero-padded to the end):
/// a constant-value short clip must produce identical mel rows across time
/// (repeatpad makes the padded signal periodic within the tiled region).
#[test]
fn short_clip_is_repeat_padded() {
  let mel = MelExtractor::new();
  // 1 s of a constant → tiles 10× exactly into 10 s (480000 / 48000 = 10, no
  // remainder), so the padded signal is a pure constant and every interior
  // frame's mel row is identical.
  let samples = vec![0.25f32; 48_000];
  let mut out = vec![0.0f32; N_MELS * T_FRAMES];
  mel.extract_into(&samples, &mut out).unwrap();
  let mid_a = &out[500 * N_MELS..501 * N_MELS];
  let mid_b = &out[600 * N_MELS..601 * N_MELS];
  let max_diff = mid_a
    .iter()
    .zip(mid_b.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
  assert!(
    max_diff < 1e-3,
    "repeat-padded constant clip should give stable interior rows; diff = {max_diff}"
  );
}

/// Filterbank rows 0, 10, 32 match committed librosa references (< 1e-6). Row 10
/// is the discriminator: it straddles the ~1 kHz Slaney inflection where an HTK
/// construction would diverge. The `.npy` files are textclap's committed librosa
/// references, so this also pins clapkit's filterbank == textclap's.
#[test]
fn filterbank_rows_match_librosa() {
  let fb = MelExtractor::build_mel_filterbank(48_000, 1024, 64, 50.0, 14_000.0);
  for &row_idx in &[0usize, 10, 32] {
    let expected = read_npy_f32(&format!("tests/fixtures/mel/filterbank_row_{row_idx}.npy"));
    assert_eq!(expected.len(), N_FREQ);
    let actual = &fb[row_idx * N_FREQ..(row_idx + 1) * N_FREQ];
    let max_diff = actual
      .iter()
      .zip(expected.iter())
      .map(|(a, b)| (*a as f32 - b).abs())
      .fold(0.0f32, f32::max);
    assert!(
      max_diff < 1e-6,
      "filterbank row {row_idx} max_abs_diff = {max_diff:.3e}"
    );
  }
}

/// **The mel-parity gate.** `tests/fixtures/mel/golden_mel.npy` is textclap's
/// committed mel oracle (`textclap/tests/fixtures/golden_mel.npy`, the HF
/// `ClapFeatureExtractor` reference textclap's own `mel.rs` validates against at
/// its documented 1e-4 budget). clapkit ports the same f64 algorithm, so on the
/// same `sample.wav` it must reproduce that pinned mel — measure-then-pin, tighter
/// than textclap's 1e-4 (the actual agreement is far below it because both are
/// f64 and clapkit's scalar reductions are bit-identical to textclap's `scalar`
/// backend; textclap's default aarch64 path is its NEON backend, ≤~1e-10·scale
/// away — so the residual here is the shared f64-STFT-vs-numpy difference, not a
/// SIMD reassociation).
///
/// A regression that widens this (wrong `fmax`/`hop`/window, f32 STFT, a
/// missing center pad) blows past the pin — the mel-internal mutation gate.
#[test]
fn extract_into_matches_textclap_golden_mel() {
  // MEASURED max-abs-diff of clapkit's mel vs textclap's golden = 7.629e-6
  // (exactly 2^-17, one f32 ULP at the largest mel magnitudes — essentially
  // bit-level agreement). Pinned with a small margin, still ~10× under
  // textclap's own 1e-4 budget and far below the fp16 resolution of the graph.
  const PARITY_MAX_ABS_DIFF: f32 = 1e-5;

  let golden = read_npy_f32("tests/fixtures/mel/golden_mel.npy");
  assert_eq!(golden.len(), N_MELS * T_FRAMES, "golden mel shape");
  let samples = read_sample_wav();

  let mel = MelExtractor::new();
  let mut out = vec![0.0f32; N_MELS * T_FRAMES];
  mel.extract_into(&samples, &mut out).expect("extract_into");

  let max_diff = out
    .iter()
    .zip(golden.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
  eprintln!("[mel] clapkit-vs-textclap-golden max_abs_diff = {max_diff:.6e}");
  assert!(
    max_diff <= PARITY_MAX_ABS_DIFF,
    "mel parity vs textclap golden regressed: max_abs_diff = {max_diff:.3e} > {PARITY_MAX_ABS_DIFF:.3e}"
  );
}
