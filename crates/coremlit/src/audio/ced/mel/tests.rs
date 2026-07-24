use super::*;
use crate::audio::ced::{Error, WINDOW_SAMPLES};

/// A full 10 s window of a 1 kHz sine at 16 kHz (period 16 samples — an exact
/// FFT-bin frequency: 1000 / (16000/512) = bin 32.0, no leakage smear).
fn tone_window() -> Vec<f32> {
  (0..WINDOW_SAMPLES)
    .map(|i| 0.5 * (std::f32::consts::TAU * 1_000.0 * (i as f32 / 16_000.0)).sin())
    .collect()
}

fn extract(samples: &[f32]) -> Vec<f32> {
  let mel = MelExtractor::new();
  let mut out = vec![f32::NAN; N_MELS * N_FRAMES];
  mel.extract_into(samples, &mut out).expect("extract");
  out
}

/// STRUCTURAL GATE 1 (spec §8): frame/bin count for the fixed window.
/// center=True at hop 160 over 160_000 samples ⇒ 1 + 160_000/160 = 1001
/// frames. The NaN seed on `out` only proves the buffer is fully WRITTEN (no
/// length/zip under-run leaves a stale NaN): `db` is zero-initialized in
/// `extract_into` and the final `top_db` clamp loop writes every element
/// unconditionally, so a skipped STFT frame/bin (still 0.0, still finite)
/// would NOT surface here — GATE 2/3/4 are the real numeric catchers.
#[test]
fn frame_count_matches_the_believed_hop_geometry() {
  assert_eq!(N_FRAMES, 1 + WINDOW_SAMPLES / 160);
  assert_eq!(N_FRAMES, 1001);
  assert_eq!(N_MELS, 64);
  let out = extract(&tone_window());
  assert!(
    out.iter().all(|v| v.is_finite()),
    "every mel element must be written and finite"
  );
}

/// STRUCTURAL GATE 2 (spec §8): dB floor on silence. All-zero input ⇒ power 0
/// everywhere ⇒ the amin floor: 10·log10(1e-10) = −100 dB exactly, and with
/// zero dynamic range the top_db clamp never engages.
#[test]
fn silence_floors_at_amin_db() {
  let out = extract(&vec![0.0f32; WINDOW_SAMPLES]);
  for (i, &v) in out.iter().enumerate() {
    assert_eq!(v, -100.0, "element {i} must sit exactly at the amin floor");
  }
}

/// STRUCTURAL GATE 3 (spec §8): energy placement of a pure tone in the
/// expected HTK mel bin. Derivation (believed constants): HTK
/// m(f) = 2595·log10(1 + f/700) ⇒ m(1000) ≈ 1000.0, m(8000) ≈ 2840.0; centers
/// sit at (i+1)·2840/65 mel, so 1 kHz falls between center 21 (~942.8 Hz) and
/// center 22 (~1006.7 Hz) with triangle weights ≈ 0.105 vs 0.895 — decisively
/// bin 22.
#[test]
fn tone_energy_lands_in_the_expected_htk_mel_bin() {
  let out = extract(&tone_window());
  // Mean dB per mel row over all frames (freq-major rows).
  let row_mean =
    |m: usize| out[m * N_FRAMES..(m + 1) * N_FRAMES].iter().sum::<f32>() / N_FRAMES as f32;
  let peak = (0..N_MELS)
    .max_by(|&a, &b| row_mean(a).total_cmp(&row_mean(b)))
    .unwrap();
  assert_eq!(peak, 22, "1 kHz must peak in HTK mel bin 22");
  // The peak stands far above a distant quiet bin.
  assert!(
    row_mean(22) > row_mean(60) + 30.0,
    "tone bin must dominate a far bin by > 30 dB"
  );
}

/// STRUCTURAL GATE 4 (spec §8): layout orientation. The believed layout is
/// freq-major rows (`out[mel * N_FRAMES + t]`, the torchaudio `[1, n_mels, T]`
/// contract). A silence-then-tone signal makes each peak-bin ROW time-varying:
/// its first-half mean sits at the clamped floor, its second half is loud. A
/// deliberately transposed write (`out[t * N_MELS + mel]`) scrambles rows
/// across time and mel, collapsing the half-difference to ≈ 0 — this test
/// reds.
#[test]
fn layout_is_freq_major_rows() {
  let mut samples = vec![0.0f32; WINDOW_SAMPLES];
  let tone = tone_window();
  samples[WINDOW_SAMPLES / 2..].copy_from_slice(&tone[WINDOW_SAMPLES / 2..]);
  let out = extract(&samples);
  let row = &out[22 * N_FRAMES..23 * N_FRAMES];
  let first_half: f32 = row[..N_FRAMES / 2].iter().sum::<f32>() / (N_FRAMES / 2) as f32;
  let second_half: f32 = row[N_FRAMES / 2..].iter().sum::<f32>() / (N_FRAMES - N_FRAMES / 2) as f32;
  assert!(
    second_half > first_half + 30.0,
    "peak-bin row must be quiet-then-loud along the frame axis \
     (first {first_half} dB, second {second_half} dB)"
  );
}

/// STRUCTURAL GATE (top_db coupling, spec §12 item 1): with the believed
/// AmplitudeToDB(top_db=120), the floor is coupled to the WINDOW'S OWN peak:
/// min == max − 120 whenever raw dynamic range exceeds 120 dB. Silence-then-
/// tone spans (−100 raw floor) … (tone peak ≫ +0 dB), > 120 dB apart, so the
/// clamp must engage exactly at max − 120.
#[test]
fn top_db_couples_the_floor_to_the_window_peak() {
  let mut samples = vec![0.0f32; WINDOW_SAMPLES];
  let tone = tone_window();
  samples[WINDOW_SAMPLES / 2..].copy_from_slice(&tone[WINDOW_SAMPLES / 2..]);
  let out = extract(&samples);
  let max = out.iter().copied().fold(f32::MIN, f32::max);
  let min = out.iter().copied().fold(f32::MAX, f32::min);
  assert!(
    max - (-100.0) > 120.0,
    "the fixture must span more than 120 dB raw (max {max})"
  );
  assert!(
    (min - (max - 120.0)).abs() < 1e-3,
    "floor must clamp to max − 120 (max {max}, min {min})"
  );
}

/// Believed sub-window policy: zero-pad AT THE WAVEFORM (spec §2 data flow).
/// Extracting a half-window input must equal extracting the explicitly
/// zero-padded full window, bit for bit.
#[test]
fn short_input_is_zero_padded_at_the_waveform() {
  let half = WINDOW_SAMPLES / 2;
  let tone = tone_window();
  let short = &tone[..half];
  let mut padded = vec![0.0f32; WINDOW_SAMPLES];
  padded[..half].copy_from_slice(short);
  assert_eq!(extract(short), extract(&padded));
}

/// Input guards: empty and over-length are typed errors, never a panic or a
/// silent truncation.
#[test]
fn empty_and_overlong_inputs_are_typed_errors() {
  let mel = MelExtractor::new();
  let mut out = vec![0.0f32; N_MELS * N_FRAMES];
  assert!(matches!(
    mel.extract_into(&[], &mut out),
    Err(Error::EmptyAudio)
  ));
  let long = vec![0.0f32; WINDOW_SAMPLES + 1];
  assert!(matches!(
    mel.extract_into(&long, &mut out),
    Err(Error::AudioTooLong { len, max }) if len == WINDOW_SAMPLES + 1 && max == WINDOW_SAMPLES
  ));
}

/// The HTK mel map's anchor property: m(1000 Hz) ≈ 1000 mel, and the
/// hz↔mel pair round-trips.
#[test]
fn htk_mel_map_anchors_and_round_trips() {
  assert!((MelExtractor::hz_to_htk_mel(1_000.0) - 999.99).abs() < 0.05);
  for hz in [0.0f64, 125.0, 440.0, 1_000.0, 4_000.0, 8_000.0] {
    let back = MelExtractor::htk_mel_to_hz(MelExtractor::hz_to_htk_mel(hz));
    assert!((back - hz).abs() < 1e-6, "round trip at {hz} gave {back}");
  }
}

/// Periodic Hann at n=512: 0 at index 0, peak 1.0 exactly at n/2, last sample
/// small but POSITIVE (distinguishing periodic from symmetric).
#[test]
fn hann_window_is_periodic_length_512() {
  let win = MelExtractor::periodic_hann(512);
  assert_eq!(win.len(), 512);
  assert_eq!(win[0], 0.0);
  assert_eq!(win[256], 1.0);
  assert!(win[511] > 0.0 && win[511] < 1e-3);
}

/// The filterbank is [N_MELS × N_FREQ] row-major, non-negative, with every row
/// carrying some mass (no empty triangle at this geometry) and NO Slaney
/// normalization (peak weight of an interior triangle is 1.0, not 2/(right−left)).
#[test]
fn filterbank_rows_are_unnormalized_triangles() {
  let fb = MelExtractor::build_htk_filterbank(16_000, 512, 64, 0.0, 8_000.0);
  assert_eq!(fb.len(), 64 * 257);
  assert!(fb.iter().all(|&w| w >= 0.0));
  for m in 0..64 {
    let row = &fb[m * 257..(m + 1) * 257];
    assert!(row.iter().sum::<f64>() > 0.0, "row {m} must carry mass");
    let peak = row.iter().copied().fold(0.0f64, f64::max);
    assert!(
      peak <= 1.0 + 1e-9,
      "norm=None peaks at ≤ 1.0, row {m} = {peak}"
    );
  }
}
