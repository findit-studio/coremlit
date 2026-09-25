use core::sync::atomic::AtomicBool;

use asry::emissions::{EmissionsAligner, EnglishNormalizer, SpeechSpans};

use super::*;
use crate::audio::align::{Lang, vocab::tokenizer_json_bytes};

fn nonzero(value: u32) -> NonZeroU32 {
  NonZeroU32::new(value).expect("nonzero")
}

/// The staged artifact's contract is the three facts its tests and its
/// measurements pinned: its table's `-` at id 0, wav2vec2's front end, and the
/// fp16 saturation band its tail saturates into on the Neural Engine.
#[test]
fn the_staged_contract_is_the_staged_artifacts() {
  let contract = AcousticContract::BASE960H;
  assert_eq!(contract.blank(), 0);
  assert_eq!(contract.blank(), BLANK_ID);
  assert_eq!(contract.geometry(), AcousticGeometry::WAV2VEC2);
  assert_eq!(contract.sentinel_band(), Some(SentinelBand::Fp16Saturation));

  let geometry = AcousticGeometry::WAV2VEC2;
  assert_eq!(geometry.sample_rate().get(), 16_000);
  assert_eq!(geometry.receptive_field().get(), 400);
  assert_eq!(geometry.stride().get(), 320);
  assert_eq!(
    geometry.frames(960_000),
    2_999,
    "the staged window's frames"
  );
}

/// **A contract of a model's own carries no band.** It states the blank and the
/// geometry, both the caller's, and nothing measured on another artifact: no
/// floor under the log-probabilities is inferred for it.
#[test]
fn a_models_own_contract_carries_no_band() {
  let geometry = AcousticGeometry::new(16_000, nonzero(640), nonzero(320)).expect("a geometry");
  let contract = AcousticContract::new(3, geometry);
  assert_eq!(contract.blank(), 3);
  assert_eq!(contract.geometry(), geometry);
  assert_eq!(contract.sentinel_band(), None);
}

/// **One declaration fits more than one front end.** A 640-sample receptive
/// field at a 320-sample stride makes the same 2999 frames of the staged
/// 960,000-sample window as wav2vec2's 400 — so a load cannot infer the
/// geometry from the declared shapes, and a contract states it.
#[test]
fn a_declared_frame_count_fits_more_than_one_front_end() {
  let wide = AcousticGeometry::new(16_000, nonzero(640), nonzero(320)).expect("a geometry");
  assert_eq!(
    wide.frames(960_000),
    AcousticGeometry::WAV2VEC2.frames(960_000)
  );
  assert_ne!(wide, AcousticGeometry::WAV2VEC2);
  // ...and they part on real audio: 720 samples are one complete 640-sample
  // frame, but two 400-sample ones.
  assert_eq!(wide.frames(720), 1);
  assert_eq!(AcousticGeometry::WAV2VEC2.frames(720), 2);
}

/// The frame count is the conv stack's own output length, and an input shorter
/// than the receptive field makes none.
#[test]
fn frames_is_the_conv_output_length() {
  let geometry = AcousticGeometry::WAV2VEC2;
  for (samples, frames) in [
    (0, 0),
    (399, 0),
    (400, 1),
    (719, 1),
    (720, 2),
    (48_000, 149),
  ] {
    assert_eq!(geometry.frames(samples), frames, "{samples} samples");
  }
}

/// **A rate asry's seam cannot time is refused by name.** asry analyses 16 kHz
/// audio only; a geometry at any other rate — or at none — would time every
/// word in the wrong unit.
#[test]
fn a_rate_other_than_16_khz_is_refused_by_name() {
  for rate in [0u32, 8_000, 22_050, 44_100, 48_000] {
    assert_eq!(
      AcousticGeometry::new(rate, nonzero(400), nonzero(320)),
      Err(GeometryError::SampleRate(rate)),
      "{rate} Hz"
    );
  }
}

/// **A geometry under which a chunk asry pads spans two frames is refused by
/// name.** asry pads a chunk shorter than 400 samples up to 400 and spreads its
/// frames over the padded length, so a receptive field and a stride summing to
/// less than 400 would give such a chunk two frames timed over samples it does
/// not have. The boundary, a sum of exactly 400, passes.
#[test]
fn a_geometry_whose_padded_chunk_spans_two_frames_is_refused_by_name() {
  for (receptive_field, stride) in [(200u32, 100u32), (299, 100), (1, 1), (320, 79)] {
    assert_eq!(
      AcousticGeometry::new(16_000, nonzero(receptive_field), nonzero(stride)),
      Err(GeometryError::PaddedChunk(PaddedChunk::new(
        receptive_field,
        stride
      ))),
      "{receptive_field}/{stride}"
    );
  }
  for (receptive_field, stride) in [(300u32, 100u32), (80, 320), (640, 320)] {
    assert!(
      AcousticGeometry::new(16_000, nonzero(receptive_field), nonzero(stride)).is_ok(),
      "{receptive_field}/{stride} sums to at least 400"
    );
  }
  // Two `u32`s at their largest do not overflow the check.
  assert!(AcousticGeometry::new(16_000, nonzero(u32::MAX), nonzero(u32::MAX)).is_ok());
}

/// The pad the geometry is checked against is the one asry's own `prepare`
/// pads with: a 100-sample chunk comes back 400 samples long, a 400-sample one
/// unpadded. If an asry release moves its pad, this fails before a geometry is
/// checked against a stale number.
#[test]
fn the_pad_is_the_one_asry_prepares_with() {
  let seam = EmissionsAligner::builder(Lang::En, tokenizer_json_bytes())
    .normalizer(Box::new(EnglishNormalizer::new()))
    .blank_token_id(BLANK_ID)
    .build()
    .expect("the bundled seam builds");
  let abort = AtomicBool::new(false);
  for (real, padded) in [
    (100usize, ASRY_PREPARE_PAD_SAMPLES as usize),
    (400, 400),
    (500, 500),
  ] {
    let samples = vec![0.1f32; real];
    let prepared = seam
      .prepare(&samples, &SpeechSpans::all_speech(), "A", &[], &abort)
      .expect("prepare");
    assert!(!prepared.is_trivial(), "`A` is alignable");
    assert_eq!(prepared.encoder_input().len(), padded, "{real} samples");
    assert_eq!(prepared.real_samples(), real);
  }
}

/// **The band holds what the staged model emits in place of a log-probability,
/// and nothing it computes correctly.** `-45440`, the ANE's saturated fp16
/// `log(0)`, and every value down to `-inf` are in it; the band's ceiling,
/// `-32768`, is its top; the staged model's legitimate minimum (`-30.81`) and
/// anything above `-32768` are not; `NaN` is no value at all.
#[test]
fn the_band_holds_the_saturated_log_zero_and_nothing_computed() {
  let band = SentinelBand::Fp16Saturation;
  assert_eq!(band.ceiling(), -32_768.0);
  for value in [-45_440.0f32, -65_504.0, -32_768.0, f32::NEG_INFINITY] {
    assert!(band.holds(value), "{value}");
  }
  for value in [-32_767.0f32, -30.81, -1.0, 0.0, f32::NAN] {
    assert!(!band.holds(value), "{value}");
  }
}
