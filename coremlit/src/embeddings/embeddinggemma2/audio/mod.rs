//! The audio part: 16 kHz mono windows of up to 10 s, each to at most 250
//! audio tokens through the audio package, and the plan that cuts a clip into
//! those windows.
//!
//! A window is fed as the reference feeds it: the waveform with 160 zeros on
//! the left and zeros after it, and a frame mask over the 1,000 frames that is
//! 1 where a frame's 321 samples end inside the padded window. Its tokens are
//! the frames over four, rounded up, and a text function embeds them.

use core::{num::NonZeroU16, time::Duration};
use std::{path::Path, time::Instant};

use half::f16;
use windit::{
  TailPolicy,
  plan::{Span, WindowOptions, WindowPlan},
};

use super::{
  artifacts::{ArtifactLabel, Part, stamp::Stamp},
  config::HIDDEN_SIZE,
  error::{
    Allocation, AllocationSite, Artifact, AudioTooShort, AudioWindowTooLong, ContractRefusal,
    Error, LoadRefusal, PredictionFailure, Result, WinditError,
  },
  media::{AUDIO_ROWS, AudioTokens},
  options::{AudioOptions, DerivationTerms},
  prewarm::{PrewarmReport, Warmed},
};
use crate::{
  DataType, Model, MultiArray,
  model::contract::{Checked, Dim, FeatureContract, LoadContract, StateContract},
};

/// Which audio package runs: FluidInference's fp16 conversion, or this
/// crate's own fp32 conversion of the same audio tower.
///
/// Both keep one I/O contract (`waveform` F32 `[1, 160160]`, `frame_mask` F16
/// `[1, 1000]` → `audio_tokens` F16 `[1, 250, 512]`); only the precision the
/// graph computes in differs. The text function that pools the tokens is fp16
/// either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum AudioPrecision {
  /// `EmbeddingGemma2Audio.mlpackage`, FluidInference's fp16 conversion. The
  /// default.
  Fp16,
  /// `EmbeddingGemma2AudioFp32.mlpackage`, converted by
  /// `coremlit/conversion/embeddinggemma2` under `compute_precision=FLOAT32`.
  Fp32,
}

impl AudioPrecision {
  /// Stable snake_case name of the precision: `"fp16"` or `"fp32"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Fp16 => "fp16",
      Self::Fp32 => "fp32",
    }
  }

  /// The package's name without its extension: `"EmbeddingGemma2Audio"` or
  /// `"EmbeddingGemma2AudioFp32"`.
  #[inline]
  pub const fn package_stem(&self) -> &'static str {
    match self {
      Self::Fp16 => "EmbeddingGemma2Audio",
      Self::Fp32 => "EmbeddingGemma2AudioFp32",
    }
  }
}

/// The samples one window holds: 10 s at 16 kHz.
pub(crate) const WINDOW_SAMPLES: usize = 160_000;
/// The zeros the reference puts before a window's samples.
pub(crate) const PAD_LEFT: usize = 160;
/// The `waveform` input's samples: the pad, a window, and the zeros after.
pub(crate) const WAVEFORM_SAMPLES: usize = 160_160;
/// The `frame_mask` input's frames.
pub(crate) const FRAMES: usize = 1_000;
/// The samples one frame reads.
const FRAME_SAMPLES: usize = 321;
/// The samples between two frames' starts.
const FRAME_HOP: usize = 160;
/// The fewest samples that complete a frame, so the fewest a window may hold.
pub(crate) const MIN_SAMPLES: usize = FRAME_SAMPLES - PAD_LEFT;

/// The frames of a window of `samples` whose mask is 1: frame `i` reads the
/// padded samples `i·160 .. i·160 + 321`, so it is valid when that span ends
/// inside the window, `i·160 + 321 ≤ 160 + samples`.
#[inline]
pub(crate) const fn valid_frames(samples: usize) -> usize {
  if PAD_LEFT + samples < FRAME_SAMPLES {
    0
  } else {
    let frames = (PAD_LEFT + samples - FRAME_SAMPLES) / FRAME_HOP + 1;
    if frames < FRAMES { frames } else { FRAMES }
  }
}

/// The audio tokens of a window of `samples`: four frames to a token, the
/// last one partial. Token `k` is valid when frame `4k` is.
#[inline]
pub(crate) const fn token_count(samples: usize) -> usize {
  valid_frames(samples).div_ceil(4)
}

/// Writes a window into the `waveform` input: 160 zeros, the samples, then
/// zeros to the end.
pub(crate) fn fill_waveform(window: &[f32], waveform: &mut [f32]) {
  let (pad, rest) = waveform.split_at_mut(PAD_LEFT);
  pad.fill(0.0);
  let (body, tail) = rest.split_at_mut(window.len());
  body.copy_from_slice(window);
  tail.fill(0.0);
}

/// Writes the `frame_mask` input: 1 over the first `frames`, 0 after.
pub(crate) fn fill_mask(frames: usize, mask: &mut [f16]) {
  let (valid, rest) = mask.split_at_mut(frames);
  valid.fill(f16::ONE);
  rest.fill(f16::ZERO);
}

/// The windows of a clip of `samples` under `options`: windit's spans of
/// 160,000 samples at the options' hop and cap, then the reference's tail
/// rule, which drops a final window shorter than `min_tail_samples` only when
/// it is not the only window. A sole window is kept from 161 samples, the
/// first that complete a frame.
///
/// # Errors
/// [`Error::AudioTooShort`] below 161 samples; [`Error::Windowing`] for a
/// plan past `max_windows`; [`Error::Allocation`] for a plan no allocation
/// can hold.
pub(crate) fn plan(samples: usize, options: &AudioOptions) -> Result<Vec<Span>> {
  if samples < MIN_SAMPLES {
    return Err(Error::AudioTooShort(AudioTooShort::new(
      samples,
      MIN_SAMPLES,
    )));
  }
  let mut geometry = WindowOptions::new(WINDOW_SAMPLES)
    .with_hop(options.hop_samples().get() as usize)
    .with_tail(TailPolicy::KeepWithCoverage);
  if let Some(max) = options.max_windows() {
    geometry = geometry.with_max_windows(max.get() as usize);
  }
  let mut spans = WindowPlan::spans(&geometry, samples).map_err(|e| match e {
    WinditError::AllocFailed { elements } => Error::Allocation(Allocation::new(
      AllocationSite::AudioPlan,
      elements.saturating_mul(size_of::<Span>()),
    )),
    e => Error::Windowing(e),
  })?;
  if spans.len() > 1
    && spans
      .last()
      .is_some_and(|last| last.len() < options.min_tail_samples() as usize)
  {
    spans.pop();
  }
  Ok(spans)
}

/// The audio package's contract, one for both precisions: `waveform` F32
/// `[1, 160160]` and `frame_mask` F16 `[1, 1000]` in, `audio_tokens` F16
/// `[1, 250, 512]` out.
fn contract() -> LoadContract {
  LoadContract::new(
    vec![
      FeatureContract::new(
        "waveform",
        DataType::F32,
        vec![Dim::Exactly(1), Dim::Exactly(WAVEFORM_SAMPLES)],
      ),
      FeatureContract::new(
        "frame_mask",
        DataType::F16,
        vec![Dim::Exactly(1), Dim::Exactly(FRAMES)],
      ),
    ],
    vec![FeatureContract::new(
      "audio_tokens",
      DataType::F16,
      vec![
        Dim::Exactly(1),
        Dim::Exactly(AUDIO_ROWS),
        Dim::Exactly(HIDDEN_SIZE),
      ],
    )],
    StateContract::None,
  )
}

/// One audio instance: the bundle of the configured precision, loaded under
/// the kit's contract, and its two inputs, allocated once and refilled.
///
/// `Send` and not `Sync`: encoding takes `&mut self`. N instances each load
/// their own bundle; the audio part maps nothing.
#[derive(Debug)]
pub struct AudioEncoder {
  model: Checked,
  options: AudioOptions,
  label: ArtifactLabel,
  waveform: MultiArray,
  frame_mask: MultiArray,
  load: Duration,
}

impl AudioEncoder {
  /// The sample rate a window is read at.
  pub const SAMPLE_RATE_HZ: u32 = 16_000;
  /// The samples one window holds: 10 s.
  pub const WINDOW_SAMPLES: usize = WINDOW_SAMPLES;

  /// An instance over the stamped model directory `dir`: the audio bundle of
  /// `options`' precision, as the stamp names it, loaded under the kit's
  /// contract with `options`' Core ML choices.
  ///
  /// # Errors
  /// [`Error::Stamp`], [`Error::Unstamped`] (naming the precision's bundle
  /// when the directory has none) or [`Error::StaleStamp`] from the stamp;
  /// [`Error::ArtifactIdentity`] when `options` pins another label;
  /// [`Error::Load`] or [`Error::Contract`] for the bundle.
  pub fn new(dir: impl AsRef<Path>, options: &AudioOptions) -> Result<Self> {
    let dir = dir.as_ref();
    let artifact = Artifact::AudioBundle(options.precision());
    let view = Stamp::read(dir)?.part(dir, Part::Audio, options.precision())?;
    view.expect(options.expect_artifacts())?;
    let started = Instant::now();
    let model = Model::load_with(view.bundle(), &options.coreml())
      .map_err(|e| Error::Load(LoadRefusal::new(artifact, None, e)))?;
    let model = Checked::new(model, &contract())
      .map_err(|violation| ContractRefusal::of(artifact, None, violation))?;
    let load = started.elapsed();
    Ok(Self {
      model,
      options: options.clone(),
      label: view.label(),
      waveform: MultiArray::zeros(&[1, WAVEFORM_SAMPLES], DataType::F32)?,
      frame_mask: MultiArray::zeros(&[1, FRAMES], DataType::F16)?,
      load,
    })
  }

  /// Runs one throwaway prediction, so the first window pays no device
  /// specialization. The report holds the bundle's load and that
  /// prediction's duration.
  ///
  /// # Errors
  /// [`Error::Prediction`] or [`Error::Tensor`].
  pub fn prewarm(&mut self) -> Result<PrewarmReport> {
    let started = Instant::now();
    self.encode_window(&[0.0; MIN_SAMPLES])?;
    let mut report = PrewarmReport::default();
    report.push(Warmed::new(None, self.load, started.elapsed()));
    Ok(report)
  }

  /// The windows of a clip of `samples`, as windit spans over its samples:
  /// see the options' hop, tail rule and cap.
  ///
  /// # Errors
  /// [`Error::AudioTooShort`] below 161 samples; [`Error::Windowing`] past
  /// `max_windows`; [`Error::Allocation`] for a plan no allocation can hold.
  pub fn plan(&self, samples: usize) -> Result<Vec<Span>> {
    plan(samples, &self.options)
  }

  /// Encodes one window of 161 to 160,000 samples, 16 kHz mono, into its
  /// audio tokens: the model's own output, and how many of its 250 rows are
  /// real.
  ///
  /// # Errors
  /// [`Error::AudioTooShort`] below 161 samples;
  /// [`Error::AudioWindowTooLong`] past 160,000; [`Error::Prediction`] or
  /// [`Error::Tensor`].
  pub fn encode_window(&mut self, window: &[f32]) -> Result<AudioTokens> {
    if window.len() < MIN_SAMPLES {
      return Err(Error::AudioTooShort(AudioTooShort::new(
        window.len(),
        MIN_SAMPLES,
      )));
    }
    if window.len() > WINDOW_SAMPLES {
      return Err(Error::AudioWindowTooLong(AudioWindowTooLong::new(
        window.len(),
        WINDOW_SAMPLES,
      )));
    }
    fill_waveform(window, self.waveform.as_slice_mut::<f32>()?);
    fill_mask(
      valid_frames(window.len()),
      self.frame_mask.as_slice_mut::<f16>()?,
    );
    let artifact = Artifact::AudioBundle(self.options.precision());
    let failure = |e| Error::Prediction(PredictionFailure::new(artifact, None, e));
    let mut outputs = self
      .model
      .predict_with(&[
        ("waveform", &self.waveform),
        ("frame_mask", &self.frame_mask),
      ])
      .map_err(failure)?;
    let tokens = outputs.take("audio_tokens").ok_or_else(|| {
      failure(crate::PredictionError::MissingOutput(
        "audio_tokens".to_owned(),
      ))
    })?;
    let valid = u16::try_from(token_count(window.len()))
      .ok()
      .and_then(NonZeroU16::new)
      .expect("161 to 160,000 samples make 1 to 250 tokens");
    Ok(AudioTokens::new(tokens, valid))
  }

  /// The precision of the bundle the instance runs.
  #[inline]
  pub const fn precision(&self) -> AudioPrecision {
    self.options.precision()
  }

  /// The options the instance runs under.
  #[inline]
  pub const fn options(&self) -> &AudioOptions {
    &self.options
  }

  /// What the instance's tokens derive from: its bundle's label and every
  /// option that changes them.
  #[inline]
  pub fn derivation_terms(&self) -> DerivationTerms<'_> {
    DerivationTerms::audio(self.label, &self.options)
  }
}

#[cfg(test)]
mod tests;
