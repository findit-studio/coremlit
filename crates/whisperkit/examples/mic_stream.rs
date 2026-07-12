//! Live microphone transcription over the push-based stream state machine.
//!
//! Everything the library refuses to do lives *here*: device capture (cpal),
//! downmix to mono, resampling to 16 kHz (rubato). The library only ever
//! sees 16 kHz mono `&[f32]` via `AudioStreamTranscriber::push_samples` —
//! this example is the reference implementation of a caller on the other
//! side of the sans-I/O boundary.
//!
//! Usage:
//! `cargo run -p whisperkit --example mic_stream -- [model-folder] [tokenizer-folder]`
//! Speak into the default input device; Ctrl-C to stop.

use std::{path::PathBuf, sync::mpsc};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedIn, Resampler};
use whisperkit::{
  options::{DecodingOptions, Options},
  transcribe::WhisperKit,
};

/// Resampler input chunk: 64 ms at 48 kHz — small enough for sub-100 ms
/// push latency, large enough to keep the FFT resampler efficient.
const RESAMPLE_CHUNK: usize = 3_072;

fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
    },
    PathBuf::from,
  )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = std::env::args().skip(1);
  let model_folder = args.next().map_or_else(
    || {
      models_dir()
        .join("whisperkit-coreml")
        .join("openai_whisper-tiny")
    },
    PathBuf::from,
  );
  let tokenizer_folder = args.next().map_or_else(
    || models_dir().join("tokenizers").join("whisper-tiny"),
    PathBuf::from,
  );

  let kit = WhisperKit::new(&Options::new(model_folder, tokenizer_folder))?;
  let mut streamer = kit.audio_stream_transcriber(DecodingOptions::new());

  let host = cpal::default_host();
  let device = host
    .default_input_device()
    .ok_or("no default input device — check microphone permissions")?;
  let supported = device.default_input_config()?;
  if supported.sample_format() != cpal::SampleFormat::F32 {
    // CoreAudio input is f32; anything else means an exotic device.
    return Err(format!("unsupported input format {:?}", supported.sample_format()).into());
  }
  let channels = usize::from(supported.channels());
  let device_rate = supported.sample_rate().0 as usize;
  eprintln!(
    "capturing from {} at {device_rate} Hz, {channels} ch (Ctrl-C to stop)",
    device.name().unwrap_or_else(|_| "<unnamed>".into()),
  );

  // The cpal callback runs on a realtime thread: downmix and ship the
  // samples off immediately; all real work happens on this thread.
  let (sender, receiver) = mpsc::channel::<Vec<f32>>();
  let config: cpal::StreamConfig = supported.into();
  let stream = device.build_input_stream(
    &config,
    move |data: &[f32], _: &cpal::InputCallbackInfo| {
      let mono: Vec<f32> = data
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
      let _ = sender.send(mono);
    },
    |err| eprintln!("input stream error: {err}"),
    None,
  )?;
  stream.play()?;

  let mut resampler = FftFixedIn::<f32>::new(device_rate, 16_000, RESAMPLE_CHUNK, 2, 1)?;
  let mut pending: Vec<f32> = Vec::new();
  let mut resampled: Vec<f32> = Vec::new();

  for captured in receiver {
    pending.extend_from_slice(&captured);
    while pending.len() >= RESAMPLE_CHUNK {
      let chunk: Vec<f32> = pending.drain(..RESAMPLE_CHUNK).collect();
      let output = resampler.process(&[chunk], None)?;
      resampled.extend_from_slice(&output[0]);
    }
    if resampled.is_empty() {
      continue;
    }
    let update = streamer.push_samples(&resampled)?;
    resampled.clear();
    if update.is_transcribed() {
      print!("\x1B[2J\x1B[H"); // clear screen: rolling transcript view
      let state = streamer.state();
      for segment in state.confirmed_segments_slice() {
        println!("{}", segment.text());
      }
      for segment in state.unconfirmed_segments_slice() {
        println!("~ {}", segment.text());
      }
    }
  }
  Ok(())
}
