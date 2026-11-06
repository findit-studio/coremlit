use super::*;

/// K2: the precisions' spellings and package stems.
#[test]
fn the_precision_spellings_are_pinned() {
  let spelled: Vec<String> = [AudioPrecision::Fp16, AudioPrecision::Fp32]
    .iter()
    .map(|p| format!("{p}={}:{}", p.as_str(), p.package_stem()))
    .collect();
  assert_eq!(
    spelled,
    [
      "fp16=fp16:EmbeddingGemma2Audio",
      "fp32=fp32:EmbeddingGemma2AudioFp32"
    ]
  );
}

/// K2: the serde form is the snake_case name.
#[cfg(feature = "serde")]
#[test]
fn the_precisions_round_trip_through_json_and_postcard() {
  for p in [AudioPrecision::Fp16, AudioPrecision::Fp32] {
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(json, format!("\"{}\"", p.as_str()));
    assert_eq!(serde_json::from_str::<AudioPrecision>(&json).unwrap(), p);
    let bytes = postcard::to_allocvec(&p).unwrap();
    assert_eq!(postcard::from_bytes::<AudioPrecision>(&bytes).unwrap(), p);
  }
}

/// A1: the closed form of the valid frames is the reference's rule, frame
/// `i` valid when `i·160 + 321 ≤ 160 + s`, counted frame by frame for every
/// window length up to a full window; the tokens are four frames each; and
/// §4.3's rows.
#[test]
fn the_frame_and_token_counts_are_the_references() {
  for samples in 0..=WINDOW_SAMPLES {
    let counted = (0..FRAMES)
      .filter(|i| i * 160 + 321 <= 160 + samples)
      .count();
    assert_eq!(valid_frames(samples), counted, "{samples} samples");
  }
  for (samples, frames, tokens) in [
    (160_000, 999, 250),
    (55_680, 347, 87),
    (8_000, 49, 13),
    (161, 1, 1),
    (100, 0, 0),
  ] {
    assert_eq!(
      (valid_frames(samples), token_count(samples)),
      (frames, tokens),
      "{samples} samples"
    );
  }
}

/// A2: a window's `waveform` is 160 zeros, its samples, then zeros; its
/// `frame_mask` is 1 over the valid frames and 0 after, every bit written.
#[test]
fn a_window_is_laid_out_as_the_reference_lays_it() {
  let window: Vec<f32> = (1..=1_000u16).map(f32::from).collect();
  let mut waveform = vec![f32::NAN; WAVEFORM_SAMPLES];
  fill_waveform(&window, &mut waveform);
  assert!(waveform[..PAD_LEFT].iter().all(|v| v.to_bits() == 0));
  assert_eq!(&waveform[PAD_LEFT..PAD_LEFT + 1_000], &window[..]);
  assert!(
    waveform[PAD_LEFT + 1_000..]
      .iter()
      .all(|v| v.to_bits() == 0)
  );
  let full = vec![0.5f32; WINDOW_SAMPLES];
  fill_waveform(&full, &mut waveform);
  assert!(waveform[..PAD_LEFT].iter().all(|v| v.to_bits() == 0));
  assert_eq!(&waveform[PAD_LEFT..], &full[..]);
  let mut mask = vec![f16::NAN; FRAMES];
  fill_mask(valid_frames(1_000), &mut mask);
  assert_eq!(valid_frames(1_000), 6);
  assert!(mask[..6].iter().all(|v| v.to_bits() == f16::ONE.to_bits()));
  assert!(mask[6..].iter().all(|v| v.to_bits() == 0));
}

/// Condition 4: a plan no allocation can hold is refused by name, as the
/// audio plan's reservation.
#[test]
fn an_unservable_plan_is_refused_by_name() {
  let options = AudioOptions::new().with_hop_samples(core::num::NonZeroU32::MIN);
  match plan(usize::MAX, &options) {
    Err(Error::Allocation(refusal)) => {
      assert_eq!(refusal.site(), AllocationSite::AudioPlan);
      assert_eq!(refusal.bytes(), usize::MAX);
    }
    other => panic!("expected Allocation(AudioPlan), got {other:?}"),
  }
}

fn spans(samples: usize, options: &AudioOptions) -> Result<Vec<(usize, usize)>> {
  plan(samples, options).map(|spans| spans.iter().map(|s| (s.start(), s.len())).collect())
}

/// A3: the plan is windit's at a 160,000-sample window and the options' hop
/// and cap, then the reference's tail rule: a final window under the minimum
/// is dropped only when it is not the only one, and a sole window is kept
/// from 161 samples.
#[test]
fn a_clip_is_planned_as_the_reference_plans_it() {
  let options = AudioOptions::new();
  assert_eq!(plan(57_600_000, &options).expect("plans").len(), 360);
  assert_eq!(
    spans(176_000, &options).expect("plans"),
    [(0, 160_000), (160_000, 16_000)]
  );
  assert_eq!(spans(167_999, &options).expect("plans"), [(0, 160_000)]);
  assert_eq!(
    spans(168_000, &options).expect("plans"),
    [(0, 160_000), (160_000, 8_000)]
  );
  assert_eq!(spans(7_999, &options).expect("plans"), [(0, 7_999)]);
  assert_eq!(spans(161, &options).expect("plans"), [(0, 161)]);
  match plan(160, &options) {
    Err(Error::AudioTooShort(e)) => assert_eq!((e.samples(), e.min()), (160, 161)),
    other => panic!("expected AudioTooShort, got {other:?}"),
  }
  let hop = options
    .clone()
    .with_hop_samples(core::num::NonZeroU32::new(80_000).expect("non-zero"));
  let planned = plan(57_600_000, &hop).expect("plans");
  assert_eq!(planned.len(), 719);
  assert_eq!(planned.last().map(Span::start), Some(3_590 * 16_000));
  let capped = options
    .clone()
    .with_max_windows(core::num::NonZeroU32::new(2).expect("non-zero"));
  assert!(matches!(plan(480_000, &capped), Err(Error::Windowing(_))));
  assert_eq!(plan(320_000, &capped).expect("plans").len(), 2);
}

/// The staged model directory: `EMBEDDINGGEMMA2_TEST_MODELS`, else the
/// workspace's `Models/embeddinggemma2`.
fn staged_dir() -> std::path::PathBuf {
  std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../Models/embeddinggemma2"
      ))
    },
    std::path::PathBuf::from,
  )
}

fn sha256_hex(bytes: &[u8]) -> String {
  use sha2::{Digest, Sha256};
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// M3 (audio), the inputs half: for every window of the R2 probe's six clips,
/// the audio inputs the kit builds hash to what FluidUse handed Core ML,
/// `waveform` and `frame_mask` byte for byte; and the text function's inputs
/// over the audio tokens hash to the probe's, when the audio model's output
/// repeats across processes.
#[test]
#[ignore = "needs the staged EmbeddingGemma 2 tree (EMBEDDINGGEMMA2_TEST_MODELS)"]
fn the_staged_audio_inputs_are_the_swift_probes_bytes() {
  use crate::embeddings::embeddinggemma2::{
    config::{BOA_ID, EOA_ID},
    text::{
      TextArtifacts,
      layout::{MediaSequence, Sequence, fill_unpacked, unpacked},
    },
    tokenizer::bpe::TokenId,
  };

  let dir = staged_dir();
  let artifacts = TextArtifacts::open(&dir).expect("the staged text part opens");
  let mut encoder = AudioEncoder::new(&dir, &AudioOptions::new()).expect("an instance");
  let probe: serde_json::Value = serde_json::from_slice(
    &std::fs::read(
      std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/embeddinggemma2/fixtures/goldens/swift_probe.json"),
    )
    .expect("the probe reads"),
  )
  .expect("parses");
  let (mut windows, mut audio_inputs) = (0, 0);
  let (mut text_inputs, mut text_differing) = (0, Vec::new());
  let mut rows = Vec::new();
  for clip in probe["audio"].as_array().expect("audio") {
    let id = clip["id"].as_str().expect("id");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join(clip["file"].as_str().expect("file"));
    let samples: Vec<f32> = hound::WavReader::open(&path)
      .expect("the clip opens")
      .samples::<i16>()
      .map(|s| f32::from(s.expect("a sample")) / 32_768.0)
      .collect();
    let samples = &samples[..clip["samples"].as_u64().expect("samples") as usize];
    for window in clip["windows"].as_array().expect("windows") {
      windows += 1;
      let start = window["start"].as_u64().expect("start") as usize;
      let len = window["samples"].as_u64().expect("samples") as usize;
      let audio = &samples[start..start + len];
      let mut waveform = vec![0f32; WAVEFORM_SAMPLES];
      fill_waveform(audio, &mut waveform);
      let mut mask = vec![f16::ZERO; FRAMES];
      fill_mask(valid_frames(len), &mut mask);
      let captures = window["captures"].as_array().expect("captures");
      for input in captures[0]["inputs"].as_array().expect("inputs") {
        let got = match input["name"].as_str() {
          Some("waveform") => sha256_hex(
            &waveform
              .iter()
              .flat_map(|v| v.to_le_bytes())
              .collect::<Vec<u8>>(),
          ),
          Some("frame_mask") => sha256_hex(
            &mask
              .iter()
              .flat_map(|v| v.to_bits().to_le_bytes())
              .collect::<Vec<u8>>(),
          ),
          other => panic!("{id}@{start}: an unexpected input {other:?}"),
        };
        assert_eq!(
          Some(got.as_str()),
          input["sha256"].as_str(),
          "{id}@{start}: {}",
          input["name"]
        );
        audio_inputs += 1;
      }
      let tokens = encoder.encode_window(audio).expect("encodes");
      tokens.rows_into(&mut rows).expect("copies");
      let sequence = Sequence::Media(MediaSequence {
        open: TokenId::from_vocab(BOA_ID),
        rows: &rows,
        close: TokenId::from_vocab(EOA_ID),
      });
      let function = unpacked(sequence.len());
      assert_eq!(
        captures[1]["function"].as_str(),
        Some(function.as_str()),
        "{id}@{start}"
      );
      let mut inputs = vec![f16::ZERO; function.tokens() * HIDDEN_SIZE];
      let mut text_mask = vec![f16::ZERO; function.tokens()];
      fill_unpacked(artifacts.table(), &sequence, &mut inputs, &mut text_mask).expect("fits");
      let mut same = true;
      for input in captures[1]["inputs"].as_array().expect("inputs") {
        let values = match input["name"].as_str() {
          Some("inputs_embeds") => &inputs,
          Some("attention_mask") => &text_mask,
          other => panic!("{id}@{start}: an unexpected input {other:?}"),
        };
        let got = sha256_hex(
          &values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<u8>>(),
        );
        same &= Some(got.as_str()) == input["sha256"].as_str();
      }
      if same {
        text_inputs += 1;
      } else {
        text_differing.push(format!("{id}@{start}"));
      }
    }
  }
  eprintln!(
    "M3 audio inputs: {audio_inputs} audio inputs byte-identical; text inputs byte-identical on {text_inputs} windows, differing on {text_differing:?}"
  );
  assert_eq!(audio_inputs, 2 * windows);
  assert!(windows >= 12, "{windows}");
}
