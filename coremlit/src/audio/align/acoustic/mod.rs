//! What an aligner must know about a CTC acoustic model and cannot read from
//! it: which column of its head is the CTC blank, how its front end turns
//! samples into frames, and which values, if any, it is measured to emit in
//! place of a log-probability.
//!
//! A load reads what a model DECLARES (its window, its frame count, its head
//! width) and checks it. None of the three facts here is declared anywhere a
//! load can read. A flat `{token: id}` table does not say which class is the
//! blank. A `[1, 960000]` window and a `[1, 2999, V]` head are what a
//! 400-sample receptive field and a 640-sample one both produce at a
//! 320-sample stride. And a saturated fp16 `log(0)` is a finite negative number
//! like any log-probability. So they are an [`AcousticContract`] the caller
//! states, and the aligner checks what it can of it against the model and the
//! vocabulary, refusing a disagreement by name at load: the blank against the
//! vocabulary ([`AlignerError::BlankOutOfVocabulary`]), the geometry against
//! the declared frame count ([`AlignerError::FrameCountMismatch`]).
//!
//! [`AcousticContract::BASE960H`] is the staged artifact's own contract, the
//! one [`Aligner::from_paths`](crate::audio::align::aligner::Aligner::from_paths)
//! binds. A model loaded with its own vocabulary states its own through
//! [`AcousticContract::new`], which carries no sentinel band: a band is a
//! measurement of one artifact, and no finite threshold separates a sentinel
//! from a valid log-probability for every model.
//!
//! [`AlignerError::BlankOutOfVocabulary`]: crate::audio::align::error::AlignerError::BlankOutOfVocabulary
//! [`AlignerError::FrameCountMismatch`]: crate::audio::align::error::AlignerError::FrameCountMismatch

use core::num::NonZeroU32;

use crate::audio::align::{
  error::{GeometryError, PaddedChunk},
  vocab::BLANK_ID,
};

/// asry's `prepare` pads a chunk shorter than this many samples up to exactly
/// this many (`asry/src/runner/aligner/core.rs`, `prepare`'s `< 400` arm), and
/// `finish` spreads the chunk's frames over that padded length. This is a law
/// of the seam this crate is built on, not of a model: asry fixes it at
/// wav2vec2's receptive field, and no contract can move it. So
/// [`AcousticGeometry::new`] refuses a geometry under which a chunk that short
/// spans two frames. `tests::the_pad_is_the_one_asry_prepares_with` holds the
/// number to asry's own `prepare`.
pub(crate) const ASRY_PREPARE_PAD_SAMPLES: u32 = 400;

/// How a model's acoustic front end turns samples into frames: the audio rate
/// it takes, the samples its first output frame spans (its receptive field),
/// and the samples between consecutive frames (its stride).
///
/// Every number the aligner derives from time comes from this geometry. It
/// gives the frames a chunk of real audio yields, which the encoder truncates
/// its output to so that frames computed from padding never reach the trellis,
/// and the stride handed to asry's seam. A model declares none of it: a CoreML
/// graph declares its window and its frame count, and several geometries
/// produce one frame count from one window. A 400-sample and a 640-sample
/// receptive field both make 2999 frames of a 960,000-sample window at a
/// 320-sample stride, yet on 720 samples of real audio the first yields two
/// frames and the second one. So the geometry is stated, and the encoder
/// checks it against the declared window and frame count at load.
///
/// # What a geometry must be
///
/// [`Self::new`] refuses, by name, the geometries under which asry's seam would
/// time words wrong: audio at any rate but asry's 16 kHz analysis rate
/// ([`GeometryError::SampleRate`]), and a receptive field and a stride summing
/// to less than the 400 samples asry pads a short chunk to
/// ([`GeometryError::PaddedChunk`]). No value of this type mis-times a chunk.
/// asry's own per-chunk stride check refuses more, by name and per chunk: it
/// requires a chunk's frames to span its length give or take two strides, and
/// a receptive field wider than two strides fails that on some chunk lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcousticGeometry {
  /// The audio rate the front end takes, in Hz.
  sample_rate: NonZeroU32,
  /// The samples the first output frame spans.
  receptive_field: NonZeroU32,
  /// The samples between consecutive frames.
  stride: NonZeroU32,
}

impl AcousticGeometry {
  /// The wav2vec2 family's convolutional front end at 16 kHz, which wav2vec2,
  /// HuBERT, WavLM, data2vec, XLS-R and MMS share. Seven strided convolutions
  /// (kernels `[10, 3, 3, 3, 3, 2, 2]`, strides `[5, 2, 2, 2, 2, 2, 2]`)
  /// compose to a 400-sample (25 ms) receptive field and a 320-sample (20 ms)
  /// stride. It is the staged `base960h`'s, and
  /// [`AcousticContract::BASE960H`]'s.
  pub const WAV2VEC2: Self = match Self::new(
    asry::time::SAMPLE_RATE_HZ,
    NonZeroU32::new(400).unwrap(),
    NonZeroU32::new(320).unwrap(),
  ) {
    Ok(geometry) => geometry,
    Err(_) => panic!("wav2vec2's front end is one asry's seam times"),
  };

  /// A front end taking audio at `sample_rate` Hz, whose first frame spans
  /// `receptive_field` samples and whose frames are `stride` samples apart.
  ///
  /// # Errors
  /// [`GeometryError::SampleRate`] unless `sample_rate` is asry's 16 kHz
  /// analysis rate; [`GeometryError::PaddedChunk`] if `receptive_field` and
  /// `stride` sum to less than the 400 samples asry pads a short chunk to.
  pub const fn new(
    sample_rate: u32,
    receptive_field: NonZeroU32,
    stride: NonZeroU32,
  ) -> Result<Self, GeometryError> {
    let Some(rate) = NonZeroU32::new(sample_rate) else {
      return Err(GeometryError::SampleRate(sample_rate));
    };
    if rate.get() != asry::time::SAMPLE_RATE_HZ {
      return Err(GeometryError::SampleRate(sample_rate));
    }
    // In `u64`: two `u32`s cannot overflow it.
    if (receptive_field.get() as u64) + (stride.get() as u64) < ASRY_PREPARE_PAD_SAMPLES as u64 {
      return Err(GeometryError::PaddedChunk(PaddedChunk::new(
        receptive_field.get(),
        stride.get(),
      )));
    }
    Ok(Self {
      sample_rate: rate,
      receptive_field,
      stride,
    })
  }

  /// The audio rate the front end takes, in Hz: asry's 16 kHz analysis rate,
  /// the only one [`Self::new`] accepts.
  #[inline]
  pub const fn sample_rate(&self) -> NonZeroU32 {
    self.sample_rate
  }

  /// The samples the first output frame spans.
  #[inline]
  pub const fn receptive_field(&self) -> NonZeroU32 {
    self.receptive_field
  }

  /// The samples between consecutive frames, which is also the stride asry's
  /// seam is handed.
  #[inline]
  pub const fn stride(&self) -> NonZeroU32 {
    self.stride
  }

  /// The frames the front end yields over an input of `samples` samples: the
  /// conv stack's own output length, `floor((samples - receptive_field) /
  /// stride) + 1`, and none when the input is shorter than the receptive
  /// field.
  pub(crate) const fn frames(&self, samples: usize) -> usize {
    let receptive_field = self.receptive_field.get() as usize;
    if samples < receptive_field {
      0
    } else {
      (samples - receptive_field) / self.stride.get() as usize + 1
    }
  }
}

/// Values an artifact is measured to emit IN PLACE OF a log-probability: a
/// band that a correctly computed log-probability of that artifact never
/// reaches and its failure mode does.
///
/// Only [`AcousticContract::BASE960H`] carries one. A band is a measurement of
/// one artifact, not a property of log-probabilities: no law bounds a
/// log-probability from below, so for a model nobody measured a band can only
/// refuse valid values. A normalized row such as `[0, -40000]` is a valid row
/// of some model, and it sits inside the band below. [`AcousticContract::new`]
/// therefore carries none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SentinelBand {
  /// fp16's saturation binade, `[-65504, -32768]`, and everything below it.
  /// An fp16 `log` of an underflowed softmax output saturates there: `-45440`
  /// on the Apple Neural Engine (the staged `base960h`, and pyannote's
  /// segmentation graph, issue #15). The binade is reached only by overflow or
  /// saturation: the fp16 `log` of fp16's smallest subnormal is `-16.6`.
  ///
  /// Measured on the staged model over `jfk.wav` (549 frames × 29 = 15,921
  /// cells):
  ///
  /// | compute | `min(emissions)` | cells in the band |
  /// |---|---|---|
  /// | `CpuOnly` (the default) | **-30.81** | 0 |
  /// | `CpuAndGpu` | **-30.02** | 0 |
  /// | `All` / `CpuAndNeuralEngine` (ANE) | **-45440** | **2,667 of 15,921 (16.7%)** |
  ///
  /// Every sentinel cell is `-45440` exactly, bit-identical run to run, and
  /// nothing the model computes correctly lands in the band.
  ///
  /// # Why a band of VALUES, never of placements
  ///
  /// The corruption is a property of the *artifact*, not of the ANE: a
  /// re-converted `base960h_aligner` with a fused (or fp32) `log_softmax` tail
  /// would be correct on the ANE, and a placement-keyed guard ("reject `All`")
  /// would forbid it forever while still failing to describe what is actually
  /// wrong. A value band is placement-agnostic in both directions: it fails the
  /// corrupt artifact wherever it runs, and passes it on
  /// [`ComputeUnits::CpuAndGpu`](crate::ComputeUnits::CpuAndGpu), a legitimate
  /// non-default placement measured clean above.
  Fp16Saturation,
}

impl SentinelBand {
  /// The band's top: a cell at or below it is in the band. `-32768` (`-2^15`,
  /// the bottom of fp16's top binade) for [`Self::Fp16Saturation`].
  #[inline]
  pub const fn ceiling(&self) -> f32 {
    match self {
      Self::Fp16Saturation => -32_768.0,
    }
  }

  /// Whether `value` is in the band. A `NaN` is not: it is no number at all,
  /// and `Emissions::from_log_probs`'s finite scan refuses it.
  pub(crate) fn holds(&self, value: f32) -> bool {
    value <= self.ceiling()
  }
}

/// [`SentinelBand::Fp16Saturation`]'s place, asserted at **compile time**: at
/// or below the top of fp16's saturation binade, so the band holds no value
/// the staged model computes correctly (its minimum is `-30.81`), and above the
/// measured sentinel (`-45440`), so it holds the one it exists for. Moving the
/// ceiling out of that interval is then a BUILD failure: above the binade the
/// band would refuse correct audio, below the sentinel it would silently stop
/// catching the corruption.
const _: () = {
  let ceiling = SentinelBand::Fp16Saturation.ceiling();
  assert!(
    ceiling <= -32_768.0,
    "the fp16 saturation band must not reach above fp16's top binade"
  );
  assert!(
    ceiling > -45_440.0,
    "the fp16 saturation band must hold the measured fp16 log(0) sentinel (-45440)"
  );
};

/// A CTC aligner model's contract: what the aligner must know about the model
/// and cannot read from it.
///
/// - [`Self::blank`]: the id of the head's CTC blank. The aligner checks it
///   against the vocabulary at load.
/// - [`Self::geometry`]: the front end's [`AcousticGeometry`]. The encoder
///   checks it against the model's declared window and frame count at load,
///   then truncates by it; the seam is handed its stride.
/// - [`Self::sentinel_band`]: values the model is measured to emit in place of
///   a log-probability, refused wherever they appear. Only the staged artifact
///   has one.
///
/// [`Self::BASE960H`] is the staged artifact's. Any other model states its own
/// with [`Self::new`], and no part of the aligner guesses one: a blank, a
/// geometry and a band are each the caller's statement or the staged
/// artifact's measurement, never an inference from the table or the declared
/// shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcousticContract {
  /// The id of the CTC blank.
  blank: u32,
  /// The front end's geometry.
  geometry: AcousticGeometry,
  /// Values measured in place of a log-probability, if the artifact has any.
  sentinel_band: Option<SentinelBand>,
}

impl AcousticContract {
  /// The staged `base960h_aligner.mlmodelc`'s contract. Its blank is id 0, the
  /// `-` of its own table ([`BLANK_ID`]).
  /// Its geometry is [`AcousticGeometry::WAV2VEC2`]. Its sentinel band is
  /// [`SentinelBand::Fp16Saturation`], where its fp16 tail saturates on the
  /// Neural Engine.
  ///
  /// It describes that artifact and its own 29-class table
  /// ([`Vocabulary::bundled`](crate::audio::align::vocab::Vocabulary::bundled),
  /// or the `base960h_dict.json` the table was derived from), and nothing
  /// else: the band would refuse valid values of another model, and id 0 is
  /// the blank only in that table.
  pub const BASE960H: Self = Self {
    blank: BLANK_ID,
    geometry: AcousticGeometry::WAV2VEC2,
    sentinel_band: Some(SentinelBand::Fp16Saturation),
  };

  /// The contract of a model loaded with its own vocabulary.
  ///
  /// `blank` is the id of its CTC blank: the class its head scores as "no
  /// token here", named by the model rather than by its table. A HuggingFace
  /// `config.json` calls it `pad_token_id`, and chordai's and torchaudio's
  /// exports put a `-` at id 0. `geometry` is its front end's. The contract
  /// carries no sentinel band.
  #[must_use]
  pub const fn new(blank: u32, geometry: AcousticGeometry) -> Self {
    Self {
      blank,
      geometry,
      sentinel_band: None,
    }
  }

  /// The id of the CTC blank.
  #[inline]
  pub const fn blank(&self) -> u32 {
    self.blank
  }

  /// The front end's geometry.
  #[inline]
  pub const fn geometry(&self) -> AcousticGeometry {
    self.geometry
  }

  /// Values the model is measured to emit in place of a log-probability:
  /// [`SentinelBand::Fp16Saturation`] for [`Self::BASE960H`], none for a
  /// contract made with [`Self::new`].
  #[inline]
  pub const fn sentinel_band(&self) -> Option<SentinelBand> {
    self.sentinel_band
  }
}

#[cfg(test)]
mod tests;
