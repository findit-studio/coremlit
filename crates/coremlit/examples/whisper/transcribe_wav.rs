//! Transcribe a 16 kHz mono WAV file.
//!
//! The library is sans-I/O: WAV decoding happens *here* (via `hound`, a
//! dev-dependency) and samples cross the boundary as 16 kHz mono `&[f32]`.
//! Convert other audio first, e.g.
//! `afconvert -f WAVE -d LEI16@16000 -c 1 input.m4a output.wav`.
//!
//! Usage:
//! `cargo run -p coremlit --features whisper --example whisper_transcribe_wav -- [wav] [model-folder] [tokenizer-folder]`
//!
//! Defaults: the committed jfk fixture and the `Models/` layout from the
//! README's "Getting models" section (honoring `WHISPERKIT_TEST_MODELS`).

use std::path::PathBuf;

use coremlit::audio::whisper::{
  options::{DecodingOptions, Options},
  result::format_segments,
  transcribe::WhisperKit,
};

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
  let wav = args.next().map_or_else(
    || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/whisper/fixtures/audio/jfk.wav"),
    PathBuf::from,
  );
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

  let mut reader = hound::WavReader::open(&wav)?;
  let spec = reader.spec();
  if spec.channels != 1
    || spec.sample_rate != 16_000
    || spec.sample_format != hound::SampleFormat::Int
    || spec.bits_per_sample != 16
  {
    return Err(
      format!("expected 16 kHz mono 16-bit PCM, got {spec:?} — convert with afconvert first")
        .into(),
    );
  }
  let audio = reader
    .samples::<i16>()
    .map(|sample| Ok(f32::from(sample?) / 32_768.0))
    .collect::<Result<Vec<f32>, hound::Error>>()?;

  eprintln!("loading models from {}", model_folder.display());
  let kit = WhisperKit::new(&Options::new(model_folder, tokenizer_folder))?;
  let result = kit.transcribe(&audio, &DecodingOptions::new())?;

  println!("language: {}", result.language());
  for line in format_segments(result.segments_slice(), true) {
    println!("{line}");
  }
  let timings = result.timings();
  println!(
    "\n{:.2} s of audio in {:.2} s — rtf {:.3}, {:.1} tokens/s",
    timings.input_audio_seconds(),
    timings.full_pipeline(),
    timings.real_time_factor(),
    timings.tokens_per_second(),
  );
  Ok(())
}
