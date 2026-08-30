use sha2::{Digest, Sha256};

use super::*;
use crate::audio::lid::{Error, frame_count};

/// SHA-256 of the producing toolchain's `[201, 60]` freq-major filterbank as
/// f32 little-endian bytes. Reproduced here EXACTLY — all 12 060 weights.
const FILTERBANK_SHA256: &str = "264347e3ef9068b6f1951c63c6b0f8344584d55182879195edf021bee42f7328";

/// SHA-256 of THIS module's 400-sample periodic Hamming window as f32
/// little-endian bytes.
///
/// # Why this is not the producing toolchain's digest
///
/// The reference implementation's window is
/// `ed725a5ae3cb12845c2ac89ed13104641ee640be9af8c026e773b90dabf52c82`, and it
/// is not reproducible from the formula alone: it was evaluated with NumPy
/// 2.5.1's vectorized **float32** cosine, which is not correctly rounded — it
/// disagrees with the exactly-rounded f32 cosine at 66 of the 400 sample
/// positions (checked against a 60-digit series evaluation). This module
/// evaluates the cosine in f64 and narrows, which IS correctly rounded at all
/// 400 positions, so the two windows differ at 23 of 400 samples by exactly one
/// ulp — a maximum absolute difference of 5.96e-8 on a coefficient of order
/// 0.5, i.e. ~1e-7 relative, five orders of magnitude below anything the
/// log-mel output can express.
///
/// So the anchor below pins OUR table, which is the more accurate of the two;
/// the reference digest is recorded here rather than asserted, because
/// asserting it would mean reproducing one NumPy build's rounding noise.
/// Everything about the window that is a real convention rather than a rounding
/// artifact — periodic not symmetric, Hamming not Hann, the exact sum — is
/// pinned by `window_is_periodic_hamming_not_hann_and_not_symmetric` below,
/// which no ulp can move.
const WINDOW_SHA256: &str = "82a72e618e6d19dbf72699d2a0246693cba1cf5f122bb88c92cb80842dba6b30";

fn sha256_f32_le(values: impl IntoIterator<Item = f32>) -> String {
  use core::fmt::Write;

  let mut hasher = Sha256::new();
  for value in values {
    hasher.update(value.to_le_bytes());
  }
  hasher.finalize().iter().fold(String::new(), |mut acc, b| {
    let _ = write!(acc, "{b:02x}");
    acc
  })
}

/// `seconds` of a 1 kHz tone at 16 kHz, amplitude 0.5.
fn tone(seconds: f32) -> Vec<f32> {
  let n = (seconds * 16_000.0) as usize;
  (0..n)
    .map(|i| 0.5 * (core::f32::consts::TAU * 1_000.0 * (i as f32 / 16_000.0)).sin())
    .collect()
}

fn extract(samples: &[f32]) -> Vec<f32> {
  let mel = MelExtractor::new();
  let mut out = vec![f32::NAN; frame_count(samples.len()) * N_MELS];
  mel.extract_into(samples, &mut out).expect("extract");
  out
}

/// One frame's 60 mel values, at time-major offsets.
fn frame(out: &[f32], t: usize) -> &[f32] {
  &out[t * N_MELS..(t + 1) * N_MELS]
}

// ── The two table anchors ───────────────────────────────────────────────────

/// The filterbank reproduces the producing toolchain's published table
/// BIT-EXACTLY: same 12 060 f32 weights, same freq-major order.
///
/// This is the strongest single statement available about the front end
/// without a model, because the filterbank is where all three interesting
/// conventions live at once — the mel scale, the non-standard symmetric
/// lower-side bands, and the absence of any area normalization. An f64
/// construction narrowed at the end would NOT pass: it moves 380 of the 12 060
/// weights.
#[test]
fn filterbank_reproduces_the_reference_digest_bit_exactly() {
  let filterbank = MelExtractor::build_filterbank();
  assert_eq!(filterbank.len(), N_FREQ * N_MELS);
  assert_eq!(sha256_f32_le(filterbank.iter().copied()), FILTERBANK_SHA256);
}

/// The window's digest, pinned as a byte-level regression anchor. See
/// [`WINDOW_SHA256`] for why this is not the reference's own digest.
#[test]
fn window_digest_is_pinned() {
  let window = MelExtractor::periodic_hamming();
  assert_eq!(window.len(), N_FFT);
  assert_eq!(sha256_f32_le(window.iter().copied()), WINDOW_SHA256);
}

// ── Trap 1: periodic Hamming ────────────────────────────────────────────────

/// Rejects all three ways this window is commonly got wrong, in the terms that
/// distinguish them and that no rounding difference can perturb:
///
/// - **Hann** would open at exactly `0.0`; Hamming opens at `0.54 - 0.46`.
/// - **Symmetric** Hamming divides the phase by `n - 1`, which makes the last
///   sample equal the first (`0.08`); periodic divides by `n`, so the last
///   sample is `0.54 - 0.46·cos(2π·399/400) ≈ 0.080057` — strictly greater.
/// - The periodic window's coefficients sum to exactly `0.54 · 400 = 216`
///   (a whole period of cosine sums to zero); the symmetric one sums to
///   ~216.46, which is 0.46 away and far outside any rounding slack.
#[test]
fn window_is_periodic_hamming_not_hann_and_not_symmetric() {
  let window = MelExtractor::periodic_hamming();

  // Hamming, not Hann: cos(0) = 1 exactly, so the opening coefficient is the
  // f32 difference of the two Hamming constants (0.08 to f32 precision).
  assert_eq!(window[0], 0.54f32 - 0.46f32);
  assert!(
    window[0] > 0.079 && window[0] < 0.081,
    "Hann would open at 0.0, got {}",
    window[0]
  );

  // Periodic, not symmetric: the tail does NOT come back to the head.
  assert!(
    window[N_FFT - 1] > window[0],
    "a symmetric window ends where it starts ({} vs {})",
    window[N_FFT - 1],
    window[0]
  );
  assert!(
    (window[N_FFT - 1] - 0.080_057).abs() < 1e-6,
    "expected the periodic tail 0.080057, got {}",
    window[N_FFT - 1]
  );

  // The sum separates the two conventions by 0.46, a huge margin.
  let sum: f64 = window.iter().map(|&w| f64::from(w)).sum();
  assert!(
    (sum - 216.0).abs() < 1e-4,
    "a periodic Hamming window sums to 216.0 (symmetric sums to ~216.46), got {sum}"
  );

  // The peak sits at the half-way sample and reaches 1.0 to f32 precision.
  assert!((window[N_FFT / 2] - 1.0).abs() < 1e-6);
}

// ── Trap 2: center=True with CONSTANT-ZERO padding ──────────────────────────

/// Centre padding is CONSTANT ZERO, not reflection.
///
/// The falsifier is exact rather than statistical. Constant-zero centring makes
/// the extractor equivalent to unpadded framing of `[0; 200] ++ x ++ [0; 200]`,
/// so prepending `2 · HOP = 320` real zero samples must shift the whole
/// spectrogram down by exactly two frames and change nothing else:
/// `extract(x)[frame t] == extract([0; 320] ++ x)[frame t + 2]`, bit for bit.
///
/// Under reflection padding that equality breaks at the head and only at the
/// head: `x`'s first frames would be filled with a mirror image of `x`'s own
/// opening samples, whereas the zero-prefixed signal reflects its own leading
/// zeros and so still sees silence there. The fixture starts at full amplitude
/// precisely so that a mirrored copy would be loud, not negligible.
#[test]
fn center_padding_is_constant_zero_not_reflection() {
  // Full amplitude within the first few samples: a reflection would duplicate
  // real energy into frame 0 instead of leaving it half empty.
  let signal: Vec<f32> = (0..8_000)
    .map(|i| 0.5 * (core::f32::consts::TAU * 1_000.0 * (i as f32 / 16_000.0) + 1.2).sin())
    .collect();
  assert!(
    signal[0].abs() > 0.2,
    "fixture must open loud, got {}",
    signal[0]
  );

  let mut prefixed = vec![0.0f32; 2 * HOP];
  prefixed.extend_from_slice(&signal);

  let plain = extract(&signal);
  let shifted = extract(&prefixed);
  assert_eq!(frame_count(prefixed.len()), frame_count(signal.len()) + 2);

  for t in 0..frame_count(signal.len()) {
    assert_eq!(
      frame(&plain, t),
      frame(&shifted, t + 2),
      "frame {t} must be identical under a whole-hop zero prefix"
    );
  }
}

/// Second, independent falsifier for the same convention, on a signal that
/// makes reflection a NO-OP: a constant.
///
/// Reflecting a constant reproduces the constant, so under reflection padding
/// EVERY frame of a DC clip is identical. Under constant-zero padding, frame 0
/// straddles the silence-to-DC step, and that step is broadband — it splashes
/// energy across the whole mel range, where an interior frame of pure DC has
/// nothing above the window's main lobe and sits on the `top_db` floor. The
/// gap is tens of dB, so the two conventions are not close.
#[test]
fn constant_signal_shows_the_zero_pad_step_that_reflection_would_hide() {
  let out = extract(&vec![0.3f32; 16_000]);
  let head = frame(&out, 0);
  let interior = frame(&out, 50);

  let widest = head
    .iter()
    .zip(interior.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
  assert!(
    widest > 30.0,
    "reflection would make every frame of a constant clip identical; \
     constant-zero padding must leave frame 0 tens of dB apart, got {widest}"
  );
}

// ── Trap 3: symmetric lower-side triangles, unnormalized ────────────────────

/// The triangles are SYMMETRIC about their centre with the LOWER-side mel
/// spacing as their half-width — not the usual asymmetric HTK/librosa shape —
/// and they carry no area normalization.
///
/// The falsifier exploits the fact that mel spacing GROWS with frequency, so
/// for any interior triangle `upper_band > lower_band`. SpeechBrain's shape
/// therefore closes at `centre + lower_band`, while the asymmetric
/// construction would still be strictly positive there (it runs out to
/// `centre + upper_band`). Sampling a frequency bin strictly between those two
/// feet separates the two constructions with no tolerance at all.
#[test]
fn triangles_are_symmetric_lower_side_bands_without_area_norm() {
  let filterbank = MelExtractor::build_filterbank();

  // Rebuild the mel edge grid the same way the filterbank does.
  let mel_max = MelExtractor::hz_to_mel(F_MAX);
  let step = mel_max / (N_MELS + 1) as f64;
  let edge = |m: usize| MelExtractor::mel_to_hz(((m as f64) * step) as f32);
  let bin_hz = |k: usize| (k as f64 * (F_MAX / (N_FREQ - 1) as f64)) as f32;

  // The standard asymmetric HTK/librosa triangle from the SAME edges: each
  // side scaled by its own neighbour spacing.
  let asymmetric = |k: usize, m: usize| {
    let (low, center, high) = (edge(m), edge(m + 1), edge(m + 2));
    let freq = bin_hz(k);
    let left = (freq - low) / (center - low);
    let right = (high - freq) / (high - center);
    f32::max(0.0, f32::min(left, right))
  };

  // The zero-tolerance falsifier: a frequency bin that falls strictly BETWEEN
  // the two constructions' right feet. SpeechBrain's triangle has already
  // closed there (exactly 0.0); the asymmetric one is still open, because mel
  // spacing widens so `upper_band > lower_band` at every triangle.
  let mut separated = 0;
  for m in 0..N_MELS {
    let center = edge(m + 1);
    let lower_band = edge(m + 1) - edge(m);
    let upper_band = edge(m + 2) - edge(m + 1);
    assert!(
      upper_band > lower_band,
      "mel spacing must widen with frequency at triangle {m} ({lower_band} -> {upper_band})"
    );
    for k in 0..N_FREQ {
      if bin_hz(k) > center + lower_band && bin_hz(k) < center + upper_band {
        assert_eq!(
          filterbank[k * N_MELS + m],
          0.0,
          "triangle {m} must close at centre + LOWER band ({} Hz), but bin {k} \
           ({} Hz) is non-zero — that is the asymmetric construction",
          center + lower_band,
          bin_hz(k)
        );
        assert!(
          asymmetric(k, m) > 0.0,
          "fixture assumption: the asymmetric construction must still be open \
           at bin {k} for triangle {m}"
        );
        // It really is open one bin earlier, so the closure is the foot and
        // not an all-zero column.
        assert!(filterbank[(k - 1) * N_MELS + m] > 0.0);
        separated += 1;
      }
    }
  }
  assert!(
    separated >= 4,
    "the 40 Hz bin grid must resolve the two right feet somewhere; found {separated}"
  );

  // Away from the feet the two constructions differ by up to ~0.04 in weight —
  // recorded so a silent swap to the standard shape is a measurable change,
  // not a rounding argument.
  let widest = (0..N_FREQ)
    .flat_map(|k| (0..N_MELS).map(move |m| (k, m)))
    .map(|(k, m)| (filterbank[k * N_MELS + m] - asymmetric(k, m)).abs())
    .fold(0.0f32, f32::max);
  assert!(
    (widest - 0.0404).abs() < 1e-3,
    "expected a peak divergence of ~0.0404 from the asymmetric construction, got {widest}"
  );

  // No Slaney/area normalization: peaks stay at ~1, and column mass grows with
  // frequency instead of being equalized away.
  let peak = filterbank.iter().copied().fold(0.0f32, f32::max);
  assert!(
    (0.99..=1.0).contains(&peak),
    "unnormalized triangles peak at ~1.0, got {peak}"
  );
  let column_sum = |m: usize| (0..N_FREQ).map(|k| filterbank[k * N_MELS + m]).sum::<f32>();
  let sums: Vec<f32> = (0..N_MELS).map(column_sum).collect();
  let min = sums.iter().copied().fold(f32::INFINITY, f32::min);
  let max = sums.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  assert!(
    (min - 0.645).abs() < 1e-3 && (max - 8.452).abs() < 1e-3,
    "column sums must run 0.645..8.452 (area normalization would flatten them), got {min}..{max}"
  );
}

// ── Trap 4: one global top_db over the whole utterance ──────────────────────

/// The `top_db` floor is taken from the WHOLE clip's peak, over time and mel
/// together — not per frame, and not a constant.
///
/// Silence followed by a tone separates the three readings cleanly. Raw silence
/// floors at `10·log10(1e-10) = -100` dB. With a global floor the silent frames
/// are lifted to `clip_max - 80`, which for this fixture is well above -100. A
/// PER-FRAME floor would leave them at -100 (a silent frame's own peak is its
/// own floor), and a fixed floor would leave them at whatever constant was
/// chosen. Asserting that the minimum is both exactly `max - 80` and strictly
/// above -100 rejects both alternatives.
#[test]
fn top_db_floor_is_global_over_the_whole_utterance() {
  let mut samples = vec![0.0f32; 16_000];
  samples.extend_from_slice(&tone(1.0));
  let out = extract(&samples);

  let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let min = out.iter().copied().fold(f32::INFINITY, f32::min);
  assert!(
    (min - (max - 80.0)).abs() < 1e-3,
    "the floor must be the clip's own peak minus 80 dB (max {max}, min {min})"
  );
  assert!(
    min > -99.0,
    "a per-frame floor would leave the silent half at the -100 dB amin floor, got {min}"
  );

  // The silent half really is sitting on that floor.
  let silent = frame(&out, 10);
  assert!(silent.iter().all(|&v| (v - min).abs() < 1e-3));
}

// ── Layout: TIME-major, the transpose killer ────────────────────────────────

/// The write is TIME-major (`out[t · N_MELS + m]`), the graph's
/// `[1, frames, 60]` contract — NOT ced's freq-major `[1, n_mels, T]`.
///
/// A transposed write produces a correctly-sized, entirely finite, plausible
/// buffer, so only a test that reads the axes apart can catch it. This one
/// makes the two axes carry different structure — silence for the first half of
/// the clip, a tone for the second — and then asserts the strong, per-frame
/// form of that structure: EVERY frame in the first half sits exactly on the
/// `top_db` floor and EVERY frame in the second half is far above it.
///
/// Under `out[m · frames + t]` the same buffer, read as frames, interleaves the
/// two halves: frame index `t` maps to flat offset `60·t`, which walks the
/// transposed buffer's own time axis 60 elements at a time and so reaches loud
/// content within the first handful of "frames". A mean-versus-mean comparison
/// could survive that; the all-frames form cannot.
#[test]
fn layout_is_time_major_rows() {
  let mut samples = vec![0.0f32; 48_000];
  samples.extend_from_slice(&tone(3.0));
  let out = extract(&samples);
  let frames = frame_count(samples.len());
  let quiet_frames = 48_000 / HOP; // frames fully inside the silent half

  let floor = out.iter().copied().fold(f32::INFINITY, f32::min);
  for t in 0..quiet_frames - 2 {
    let row = frame(&out, t);
    assert!(
      row.iter().all(|&v| (v - floor).abs() < 1e-3),
      "frame {t} is in the silent half, so every one of its {N_MELS} mel values \
       must sit on the global floor {floor}"
    );
  }
  for t in quiet_frames + 2..frames {
    let row = frame(&out, t);
    let loudest = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
      loudest > floor + 30.0,
      "frame {t} is in the tone half, so it must rise far above the floor \
       ({loudest} vs {floor})"
    );
  }
}

// ── Geometry and guards ─────────────────────────────────────────────────────

/// The written frame count follows the centre-padded hop arithmetic, and every
/// element is written (the NaN seed proves no under-run leaves a hole).
#[test]
fn extraction_writes_exactly_the_expected_frame_count() {
  for samples in [1_440usize, 1_600, 16_000, 48_000] {
    let signal = vec![0.25f32; samples];
    let out = extract(&signal);
    assert_eq!(out.len(), frame_count(samples) * N_MELS);
    assert_eq!(frame_count(samples), 1 + samples / HOP);
    assert!(out.iter().all(|v| v.is_finite()), "{samples} samples");
  }
}

/// A NaN or infinity anywhere in the clip is a typed error, not a poisoned
/// spectrogram. It matters more here than in a per-window front end: the
/// `top_db` floor is global, so one bad sample would drag EVERY frame's floor
/// to NaN, not only the frames it touches.
#[test]
fn non_finite_samples_are_a_typed_error() {
  let mel = MelExtractor::new();
  let mut signal = vec![0.25f32; 1_600];
  signal[900] = f32::NAN;
  let mut out = vec![0.0f32; frame_count(signal.len()) * N_MELS];
  assert!(matches!(
    mel.extract_into(&signal, &mut out),
    Err(Error::NonFiniteInput(900))
  ));

  signal[900] = f32::NEG_INFINITY;
  assert!(matches!(
    mel.extract_into(&signal, &mut out),
    Err(Error::NonFiniteInput(900))
  ));
}

/// Silence alone floors at the `amin` power floor exactly: `10·log10(1e-10)` is
/// -100 dB, and with zero dynamic range the `top_db` clamp cannot engage.
#[test]
fn silence_floors_at_the_amin_power_floor() {
  let out = extract(&vec![0.0f32; 16_000]);
  assert!(
    out.iter().all(|&v| (v - (-100.0)).abs() < 1e-4),
    "silence must land on 10·log10(amin) = -100 dB"
  );
}

/// The mel scale is SpeechBrain's `2595·log10(1 + f/700)` and round-trips.
#[test]
fn mel_scale_anchors_and_round_trips() {
  assert_eq!(MelExtractor::hz_to_mel(0.0), 0.0);
  assert!((MelExtractor::hz_to_mel(1_000.0) - 999.99).abs() < 0.05);
  assert!((MelExtractor::hz_to_mel(F_MAX) - 2_840.023).abs() < 1e-3);
  for hz in [0.0f32, 125.0, 440.0, 1_000.0, 4_000.0, 8_000.0] {
    let mel = MelExtractor::hz_to_mel(f64::from(hz)) as f32;
    let back = MelExtractor::mel_to_hz(mel);
    assert!(
      (back - hz).abs() < 0.01,
      "round trip at {hz} Hz gave {back} Hz"
    );
  }
}

/// A pure tone's energy lands in the mel bin the scale puts it in, and stands
/// far above a distant bin — the end-to-end check that the window, FFT,
/// filterbank and dB stages are wired to each other in the right order.
#[test]
fn tone_energy_lands_in_the_expected_mel_bin() {
  let out = extract(&tone(1.0));
  let frames = frame_count(16_000);
  let column_mean =
    |m: usize| (0..frames).map(|t| out[t * N_MELS + m]).sum::<f32>() / frames as f32;
  let peak = (0..N_MELS)
    .max_by(|&a, &b| column_mean(a).total_cmp(&column_mean(b)))
    .expect("N_MELS > 0");

  // 1 kHz sits at 2595·log10(1 + 1000/700) = 999.99 mel; the grid step is
  // 2840.023 / 61 = 46.558 mel, so the nearest centre is index
  // round(999.99 / 46.558) - 1 = 20.
  assert_eq!(peak, 20, "1 kHz must peak in mel bin 20");
  assert!(
    column_mean(20) > column_mean(55) + 30.0,
    "the tone bin must dominate a far bin by more than 30 dB"
  );
}
