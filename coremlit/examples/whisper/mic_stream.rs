//! Live microphone transcription over the push-based stream state machine.
//!
//! Everything the library refuses to do lives *here*: device capture (cpal),
//! downmix to mono, resampling to 16 kHz (rubato). The library only ever
//! sees 16 kHz mono `&[f32]` via `AudioStreamTranscriber::push_samples` —
//! this example is the reference implementation of a caller on the other
//! side of the sans-I/O boundary.
//!
//! Usage:
//! `cargo run -p coremlit --features whisper --example whisper_mic_stream -- [model-folder] [tokenizer-folder]`
//! Speak into the default input device; Ctrl-C to stop.

// The workspace-root anchor, FOUND by searching upward for the `[workspace]`
// manifest rather than counted in `../` hops — see its module doc.
#[path = "../../tests/support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;

use std::{
  collections::VecDeque,
  path::PathBuf,
  sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicUsize, Ordering},
  },
  time::Duration,
};

use coremlit::audio::whisper::{
  options::{DecodingOptions, Options},
  transcribe::WhisperKit,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{Fft, FixedSync, Indexing, Resampler, audioadapter_buffers::direct::InterleavedSlice};

/// Resampler input chunk: 64 ms at 48 kHz — small enough for sub-100 ms
/// push latency, large enough to keep the FFT resampler efficient.
const RESAMPLE_CHUNK: usize = 3_072;

/// How long the consumer sleeps between polls of an empty
/// [`BoundedAudioQueue`]. There is no condvar/wake-up signal here (see the
/// queue's own doc) — a short poll keeps added latency well under the
/// resampler chunk's own budget without busy-spinning.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Seconds of raw, pre-resample audio [`BoundedAudioQueue`] holds before it
/// starts dropping the oldest samples. Sized generously — a stalled decode
/// or a temporarily slower-than-realtime model should not lose audio the
/// user is about to see transcribed — while still bounding worst-case
/// queued memory to a fixed amount regardless of how long the consumer
/// stalls, unlike the unbounded channel this queue replaces.
const QUEUE_CAPACITY_SECONDS: usize = 30;

/// Bounded queue between the realtime CPAL callback and the consumer
/// thread that resamples and feeds the pipeline. Holds samples at the
/// *device's* native rate: resampling to 16 kHz happens only on the
/// consumer side, never in the callback (an FFT resampler is not
/// realtime-safe).
///
/// **Overrun policy — drop oldest, not incoming:** once full, a push makes
/// room by discarding the *oldest* queued audio rather than rejecting the
/// new arrival. For a live "what's being said right now" transcript, the
/// most recently captured audio is the useful half to keep; dropping
/// incoming audio instead would freeze the visible transcript on stale
/// backlog while silently discarding everything the user is currently
/// saying. Either policy loses audio once genuinely overrun — this only
/// decides which half, and [`Self::overruns`] makes the loss observable
/// either way.
///
/// **Lock tradeoff, stated plainly:** the callback still takes a `Mutex`
/// to append, so this is not wait-free — a true lock-free SPSC ring would
/// remove even the bounded-wait risk of a bounded-priority-inversion
/// scenario on CoreAudio's realtime thread. That's out of scope for a
/// reference example. What this type does provide: no heap allocation on
/// the realtime path (backing storage is preallocated to `capacity` once,
/// in [`Self::new`], and never grows past it) and a hard cap on queued
/// memory with observable overrun accounting.
struct BoundedAudioQueue {
  inner: Mutex<VecDeque<f32>>,
  capacity: usize,
  overruns: AtomicUsize,
}

impl BoundedAudioQueue {
  fn new(capacity: usize) -> Self {
    Self {
      inner: Mutex::new(VecDeque::with_capacity(capacity)),
      capacity,
      overruns: AtomicUsize::new(0),
    }
  }

  /// Appends `samples` — the realtime callback's entry point. Takes an
  /// iterator rather than a slice so the caller (the downmix step) can
  /// feed computed samples straight in without collecting them into a
  /// temporary `Vec` first; nothing on this path allocates.
  ///
  /// Drops the oldest queued samples first if `samples` would overflow
  /// `capacity` (see the type's own doc for why oldest-not-incoming), and
  /// counts every push that had to drop anything in [`Self::overruns`].
  fn push_overwriting(&self, samples: impl ExactSizeIterator<Item = f32>) {
    let incoming = samples.len();
    let mut queue = self.inner.lock().unwrap_or_else(PoisonError::into_inner);

    if incoming >= self.capacity {
      // A single callback chunk already fills (or exceeds) the whole
      // queue on its own: keep only its most recent `capacity` samples.
      // Count an overrun only when samples were actually discarded —
      // queued audio dropped by the clear, or the chunk's own excess
      // beyond capacity. `incoming == capacity` into an empty queue
      // keeps every sample and is NOT an overrun (phase-gate finding).
      let dropped = queue.len() + (incoming - self.capacity);
      queue.clear();
      queue.extend(samples.skip(incoming - self.capacity));
      if dropped > 0 {
        self.overruns.fetch_add(1, Ordering::Relaxed);
      }
      return;
    }

    let overflow = (queue.len() + incoming).saturating_sub(self.capacity);
    if overflow > 0 {
      queue.drain(..overflow);
      self.overruns.fetch_add(1, Ordering::Relaxed);
    }
    queue.extend(samples);
  }

  /// Moves every currently queued sample into `into`, appending rather
  /// than replacing its contents — `into` is the consumer's reusable
  /// scratch buffer, so this never allocates on the consumer's steady
  /// state either (only ever grows `into` up to the high-water mark it
  /// already reached once).
  fn drain_into(&self, into: &mut Vec<f32>) {
    let mut queue = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
    into.extend(queue.drain(..));
  }

  /// Total pushes that dropped queued audio to stay within capacity.
  /// Monotonically increasing for the life of the queue.
  fn overruns(&self) -> usize {
    self.overruns.load(Ordering::Relaxed)
  }

  #[cfg(test)]
  fn len(&self) -> usize {
    self
      .inner
      .lock()
      .unwrap_or_else(PoisonError::into_inner)
      .len()
  }
}

fn models_dir() -> PathBuf {
  std::env::var_os("WHISPERKIT_TEST_MODELS").map_or_else(workspace_root::models_root, PathBuf::from)
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
  // cpal 0.18: `SampleRate` is a `u32` alias (no `.0` newtype field), and the
  // device label now comes from the `Display` impl (the fallible `name()` is
  // gone).
  let device_rate = supported.sample_rate() as usize;
  eprintln!(
    "capturing from {device} at {device_rate} Hz, {channels} ch, {QUEUE_CAPACITY_SECONDS}s buffer \
     (Ctrl-C to stop)",
  );

  // The cpal callback runs on a realtime thread: downmix and copy straight
  // into the bounded queue; all real work (resampling, transcription)
  // happens on this thread instead. See `BoundedAudioQueue`'s own doc for
  // the capacity/overrun/locking tradeoffs.
  let queue = Arc::new(BoundedAudioQueue::new(device_rate * QUEUE_CAPACITY_SECONDS));
  let callback_queue = Arc::clone(&queue);
  let config: cpal::StreamConfig = supported.into();
  let stream = device.build_input_stream(
    config,
    move |data: &[f32], _: &cpal::InputCallbackInfo| {
      let mono = data
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32);
      callback_queue.push_overwriting(mono);
    },
    |err| eprintln!("input stream error: {err}"),
    None,
  )?;
  stream.play()?;

  // rubato 5.0: the unified `Fft` resampler (`FixedSync::Input` consumes a fixed
  // RESAMPLE_CHUNK input frames per call — the former `FftFixedIn` role) that
  // writes into a preallocated audioadapter buffer instead of returning a fresh
  // Vec. `Fft::new` picks its sub-chunk count as `chunk / 256` (12 here for the
  // 3072-frame RESAMPLE_CHUNK), replacing the old `FftFixedIn` explicit argument
  // `2` — the larger count slightly reduces leading latency. `Fft::new_custom`
  // is the knob if that ever matters, but a live-mic reference stream pins no
  // resampled output.
  let mut resampler = Fft::<f32>::new(device_rate, 16_000, RESAMPLE_CHUNK, 1, FixedSync::Input)?;
  let mut resample_out = vec![0.0f32; resampler.output_frames_max()];
  let indexing = Indexing::new();
  let mut captured: Vec<f32> = Vec::new();
  let mut pending: Vec<f32> = Vec::new();
  let mut resampled: Vec<f32> = Vec::new();

  loop {
    queue.drain_into(&mut captured);
    if captured.is_empty() {
      std::thread::sleep(POLL_INTERVAL);
      continue;
    }
    pending.extend_from_slice(&captured);
    captured.clear();

    while pending.len() >= RESAMPLE_CHUNK {
      let chunk: Vec<f32> = pending.drain(..RESAMPLE_CHUNK).collect();
      // Mono (1 channel) so interleaved == sequential: wrap the fixed input
      // chunk and the reusable output buffer as audioadapter views, resample
      // into the buffer, then copy out only the frames actually written.
      let input = InterleavedSlice::new(&chunk, 1, RESAMPLE_CHUNK)?;
      let out_capacity = resample_out.len();
      let frames_written = {
        let mut output = InterleavedSlice::new_mut(&mut resample_out, 1, out_capacity)?;
        let (_, written) = resampler.process_into_buffer(&input, &mut output, Some(&indexing))?;
        written
      };
      resampled.extend_from_slice(&resample_out[..frames_written]);
    }
    if resampled.is_empty() {
      continue;
    }
    let update = streamer.push_samples(&resampled)?;
    resampled.clear();
    if update.is_transcribed() {
      print!("\x1B[2J\x1B[H"); // clear screen: rolling transcript view
      println!("[overruns: {}]", queue.overruns());
      let state = streamer.state();
      for segment in state.confirmed_segments_slice() {
        println!("{}", segment.text());
      }
      for segment in state.unconfirmed_segments_slice() {
        println!("~ {}", segment.text());
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::BoundedAudioQueue;

  #[test]
  fn stalled_consumer_never_exceeds_capacity_and_overruns_are_observable() {
    let capacity = 100;
    let queue = BoundedAudioQueue::new(capacity);
    let chunk = [1.0_f32; 10];

    // A stalled consumer: push 50 callback-sized chunks (5x capacity) with
    // nothing draining the queue in between.
    for _ in 0..50 {
      queue.push_overwriting(chunk.iter().copied());
      assert!(
        queue.len() <= capacity,
        "queue grew past capacity: {} > {capacity}",
        queue.len()
      );
    }

    assert_eq!(
      queue.len(),
      capacity,
      "queue should be saturated at capacity"
    );
    let overruns_after_stall = queue.overruns();
    assert!(
      overruns_after_stall > 0,
      "sustained overflow must be observable via overruns()"
    );

    // The counter must never decrease, including once more overflow
    // happens after it was already observed.
    queue.push_overwriting(chunk.iter().copied());
    assert!(
      queue.overruns() >= overruns_after_stall,
      "overruns() must be monotonically non-decreasing"
    );
  }

  #[test]
  fn drain_into_empties_the_queue_and_appends_to_the_destination() {
    let queue = BoundedAudioQueue::new(16);
    queue.push_overwriting([1.0_f32, 2.0, 3.0].into_iter());

    let mut into = vec![0.0_f32];
    queue.drain_into(&mut into);

    assert_eq!(into, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(queue.len(), 0);
  }

  #[test]
  fn oversized_single_push_keeps_only_the_most_recent_capacity_samples() {
    let queue = BoundedAudioQueue::new(4);
    // One callback chunk bigger than the whole queue.
    queue.push_overwriting((0..10).map(|n| n as f32));

    let mut into = Vec::new();
    queue.drain_into(&mut into);
    assert_eq!(
      into,
      vec![6.0, 7.0, 8.0, 9.0],
      "must keep the newest samples"
    );
    assert_eq!(queue.overruns(), 1);
  }
}

#[cfg(test)]
mod overrun_edge_tests {
  use super::BoundedAudioQueue;

  #[test]
  fn exactly_capacity_into_an_empty_queue_is_not_an_overrun() {
    // Regression (phase-gate round 2): the oversized branch incremented
    // the counter even when nothing was discarded.
    let queue = BoundedAudioQueue::new(8);
    queue.push_overwriting([0.0f32; 8].into_iter());
    assert_eq!(queue.overruns(), 0, "all samples kept: no overrun");
    let mut out = Vec::new();
    queue.drain_into(&mut out);
    assert_eq!(out.len(), 8);
  }

  #[test]
  fn overruns_count_exactly_the_dropping_pushes() {
    let queue = BoundedAudioQueue::new(8);
    queue.push_overwriting([0.0f32; 4].into_iter()); // fits
    queue.push_overwriting([0.0f32; 4].into_iter()); // fits exactly
    assert_eq!(queue.overruns(), 0);
    queue.push_overwriting([0.0f32; 1].into_iter()); // drops one -> 1
    queue.push_overwriting([0.0f32; 12].into_iter()); // oversized, drops -> 2
    assert_eq!(queue.overruns(), 2, "exactly the two dropping pushes");
  }
}
