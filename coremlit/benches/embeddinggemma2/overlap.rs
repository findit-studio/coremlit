//! EmbeddingGemma 2 with the GPU and Neural Engine stages overlapped, against
//! the facade's one stage after another, model-gated.
//!
//! `harness = false`, like the other EmbeddingGemma 2 benches. Every lane on
//! at its defaults: audio and vision on the GPU, text on the Neural Engine.
//! The work is [`FRAMES`] 1280 × 720 frames at the default budget and a clip
//! of [`CLIP_SECONDS`] seconds, in [`ROUNDS`] rounds:
//! - **one after another:** the facade, `embed_image` per frame, then
//!   `embed_audio` over the clip;
//! - **overlapped:** the facade's parts on two threads. The GPU thread
//!   prepares and encodes each frame and each window and sends the tokens over
//!   a bounded channel; the Neural Engine thread embeds them as they arrive.
//!
//! It prints the median of each, items per second and the ratio. The work is
//! the same either way (one image or one window per text call), so the ratio
//! is what the overlap buys on this host.
//!
//! Run with the staged tree in `EMBEDDINGGEMMA2_TEST_MODELS` (or
//! `Models/embeddinggemma2`):
//! `cargo bench -p coremlit --features embeddinggemma2 --bench embeddinggemma2_overlap`

use std::{
  path::PathBuf,
  sync::mpsc,
  time::{Duration, Instant},
};

use coremlit::embeddings::embeddinggemma2::{
  AudioOptions, AudioTokens, Embedder, EmbedderOptions, ImageOptions, ImageTokens, PreparedImage,
  Rgb8Image, compile_artifacts,
};

/// Frames per round.
const FRAMES: usize = 24;

/// The clip's length per round, in seconds: six full windows.
const CLIP_SECONDS: usize = 60;

/// Rounds per mode.
const ROUNDS: usize = 5;

/// The frame's geometry.
const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

/// Tokens in flight from the GPU thread to the Neural Engine thread.
const IN_FLIGHT: usize = 4;

fn staged() -> Option<PathBuf> {
  let dir = std::env::var_os("EMBEDDINGGEMMA2_TEST_MODELS").map_or_else(
    || {
      PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../Models/embeddinggemma2"
      ))
    },
    PathBuf::from,
  );
  dir.is_dir().then_some(dir)
}

/// The median of `times`.
fn median(mut times: Vec<Duration>) -> Duration {
  times.sort();
  times[times.len() / 2]
}

/// What the GPU thread hands the Neural Engine thread.
enum Tokens {
  Image(ImageTokens),
  Audio(AudioTokens),
}

fn main() {
  let Some(dir) = staged() else {
    eprintln!("SKIP: no staged EmbeddingGemma 2 tree (set EMBEDDINGGEMMA2_TEST_MODELS)");
    return;
  };
  compile_artifacts(&dir).expect("the staged tree compiles");
  let options = EmbedderOptions::new()
    .with_audio(AudioOptions::new())
    .with_vision(ImageOptions::new());
  let mut embedder = Embedder::load(&dir, &options).expect("every lane loads");
  embedder.prewarm().expect("warms");

  // Frames that differ, so no stage sees the same input twice in a row.
  let frames: Vec<Vec<u8>> = (0..FRAMES)
    .map(|i| {
      (0..HEIGHT)
        .flat_map(|y| {
          (0..WIDTH).flat_map(move |x| {
            [
              ((x + i * 9) / 5) as u8,
              (y / 3) as u8,
              ((x + y + i) / 8) as u8,
            ]
          })
        })
        .collect()
    })
    .collect();
  let images: Vec<Rgb8Image<'_>> = frames
    .iter()
    .map(|pixels| Rgb8Image::new(pixels, WIDTH, HEIGHT).expect("a frame"))
    .collect();
  let clip: Vec<f32> = (0..CLIP_SECONDS * 16_000)
    .map(|i| 0.3 * (i as f32 * 440.0 * core::f32::consts::TAU / 16_000.0).sin())
    .collect();

  let mut sequential = Vec::new();
  let mut items = 0;
  for _ in 0..ROUNDS {
    let started = Instant::now();
    for image in &images {
      embedder.embed_image(*image, None).expect("embeds");
    }
    items = images.len() + embedder.embed_audio(&clip).expect("embeds").len();
    sequential.push(started.elapsed());
  }

  let (mut text, audio, vision) = embedder.into_parts().into_inner();
  let mut audio = audio.expect("the audio part");
  let (artifacts, mut encoder) = vision.expect("the vision part").into_inner();
  let image_options = encoder.options().clone();
  let mut prepared = PreparedImage::new();
  let mut overlapped = Vec::new();
  for _ in 0..ROUNDS {
    let started = Instant::now();
    std::thread::scope(|scope| {
      let (send, receive) = mpsc::sync_channel::<Tokens>(IN_FLIGHT);
      let (images, clip) = (&images, &clip);
      let (artifacts, encoder, audio, prepared) =
        (&artifacts, &mut encoder, &mut audio, &mut prepared);
      let image_options = &image_options;
      scope.spawn(move || {
        for image in images {
          artifacts
            .prepare_into(*image, None, image_options, prepared)
            .expect("prepares");
          let tokens = encoder.encode(prepared).expect("encodes");
          send
            .send(Tokens::Image(tokens))
            .expect("the receiver waits");
        }
        for span in audio.plan(clip.len()).expect("plans") {
          let tokens = audio
            .encode_window(&clip[span.start()..span.end()])
            .expect("encodes");
          send
            .send(Tokens::Audio(tokens))
            .expect("the receiver waits");
        }
      });
      for tokens in receive {
        match tokens {
          Tokens::Image(tokens) => text.embed_image_tokens(&tokens),
          Tokens::Audio(tokens) => text.embed_audio_tokens(&tokens),
        }
        .expect("embeds");
      }
    });
    overlapped.push(started.elapsed());
  }

  let (one_after_another, both) = (median(sequential), median(overlapped));
  println!(
    "{FRAMES} frames + {CLIP_SECONDS} s of audio ({items} items), median of {ROUNDS} rounds: one after another {one_after_another:>9.1?} ({:>6.2} items/s); overlapped {both:>9.1?} ({:>6.2} items/s); ratio {:.2}",
    items as f64 / one_after_another.as_secs_f64(),
    items as f64 / both.as_secs_f64(),
    one_after_another.as_secs_f64() / both.as_secs_f64()
  );
}
