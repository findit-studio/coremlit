//! What an aligner must know about a CTC acoustic model and cannot read from
//! it: which column of its head is the CTC blank, how its front end turns
//! samples into frames, how its head spells a word, what its head emits, and
//! which values, if any, it is measured to emit in place of a log-probability.
//!
//! A load reads what a model DECLARES (its window, its frame count, its head
//! width) and checks it. None of the facts here is declared anywhere a load can
//! read. A flat `{token: id}` table does not say which class is the blank,
//! which token delimits words, or whether its letters are cased. A
//! `[1, 960000]` window and a `[1, 2999, V]` head are what a 400-sample
//! receptive field and a 640-sample one both produce at a 320-sample stride. A
//! head's last op, a log-softmax or a bare linear layer, leaves no trace in its
//! declared shape. And a saturated fp16 `log(0)` is a finite negative number
//! like any log-probability. So they are an [`AcousticContract`] the caller
//! states, and the aligner checks what it can of it against the model, the
//! vocabulary and the normalizer, refusing a disagreement by name at load: the
//! blank against the vocabulary ([`AlignerError::BlankOutOfVocabulary`]), the
//! tokenization against the vocabulary, the normalizer and the one policy
//! asry's seam implements ([`AlignerError::Tokenization`]), the geometry
//! against the declared frame count ([`AlignerError::FrameCountMismatch`]), and
//! the output kind against the head's width
//! ([`AlignerError::UnprovableNormalization`]).
//!
//! [`AcousticContract::BASE960H`] is the staged artifact's own contract, the
//! one [`Aligner::from_paths`](crate::audio::align::aligner::Aligner::from_paths)
//! binds. A model loaded with its own vocabulary states its own through
//! [`AcousticContract::new`], which carries no sentinel band: a band is a
//! measurement of one artifact, and no finite threshold separates a sentinel
//! from a valid log-probability for every model.
//!
//! Every field of a contract is the caller's own ASSERTION about one specific
//! model, vocabulary and normalizer, checked for agreement among THEMSELVES —
//! never against the model's actual weights or tokenizer, which nothing here
//! can see. A caller who states [`Tokenization`]'s
//! [`Granularity::Character`] falsely, for a model whose vocabulary genuinely
//! holds subword classes, gets no defense here beyond what the table itself
//! can show: once a class like `AB` sits beside `A` and `B`,
//! [`AlignerError::Tokenization`] refuses it by name; a table that hides the
//! disagreement is not caught. That is this crate's trust model for every
//! statement an [`AcousticContract`] carries, not a gap particular to
//! tokenization: the model, the vocabulary and the contract are the caller's
//! own, and are trusted to describe one real, consistent artifact.
//!
//! [`AlignerError::BlankOutOfVocabulary`]: crate::audio::align::error::AlignerError::BlankOutOfVocabulary
//! [`AlignerError::Tokenization`]: crate::audio::align::error::AlignerError::Tokenization
//! [`AlignerError::FrameCountMismatch`]: crate::audio::align::error::AlignerError::FrameCountMismatch
//! [`AlignerError::UnprovableNormalization`]: crate::audio::align::error::AlignerError::UnprovableNormalization

use core::num::NonZeroU32;

use crate::audio::align::{
  error::{GeometryError, PaddedChunk, TokenizationError},
  vocab::{BLANK_ID, Vocabulary},
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
/// ([`GeometryError::PaddedChunk`]). So no value of this type lets asry's
/// preparation and the encoder's truncation disagree about a chunk's frames.
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
  /// the magnitude at which fp16's top binade begins) for
  /// [`Self::Fp16Saturation`].
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

/// The token that delimits words in a CTC head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WordDelimiter {
  /// Words are delimited by the `|` token: the wav2vec2 convention (a
  /// HuggingFace tokenizer's `word_delimiter_token`) and the one delimiter
  /// asry's seam inserts, between the words of a word-delimiting normalizer.
  Pipe,
  /// The model has no word delimiter: a character-segmented script (Chinese,
  /// Japanese), whose normalizer inserts none.
  Absent,
}

impl WordDelimiter {
  /// The delimiter a model's own configuration names (a HuggingFace
  /// tokenizer's `word_delimiter_token`).
  ///
  /// # Errors
  /// [`TokenizationError::UnsupportedDelimiter`] for any token but `|`: asry's
  /// seam delimits words with `|` only, so a model that delimits them with a
  /// space or another token cannot be aligned through it.
  pub fn from_token(token: &str) -> Result<Self, TokenizationError> {
    if token == "|" {
      Ok(Self::Pipe)
    } else {
      Err(TokenizationError::UnsupportedDelimiter(token.to_owned()))
    }
  }
}

/// How a CTC head's table cases ASCII letters, and so how a text's letters
/// are looked up in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LetterCase {
  /// The table spells ASCII letters in upper case only, and a text's ASCII
  /// letters are projected to upper case before they are looked up (the
  /// wav2vec2 English checkpoints, the staged `base960h` among them).
  Upper,
  /// Letters are looked up as the normalizer writes them, with no projection
  /// (a lowercase table, or one that spells both cases).
  AsWritten,
}

/// How finely a CTC head's vocabulary segments text into tokens.
///
/// asry's seam always resolves a text's tokens with a per-character
/// `token_to_id` lookup: it never runs a real subword tokenizer. So
/// [`Self::Character`] is the only variant this seam can execute, and the one
/// a contract is checked against at load. A subword table — say, one
/// truthfully spelling `A`, `B` and the class `AB` a model's own tokenizer
/// actually emits for "AB" — passes every other check here and still reads
/// the wrong columns: asry would look `A` and `B` up separately and never
/// touch the `AB` column the model scored. Aligning that needs a seam that
/// calls the model's own tokenizer instead of looking characters up one at a
/// time; this type grows no `Subword` variant until such a seam exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Granularity {
  /// Every LEXICAL token — one that is not the blank, the delimiter, or a
  /// declared special ([`Tokenization::specials`]) — is exactly one Unicode
  /// scalar value: what asry's per-character lookup requires.
  Character,
}

/// How a CTC head spells a word: the token that delimits words, the case its
/// letters are spelled in, how finely it segments text, and the tokens that
/// are never letters however they are spelled.
///
/// asry 0.2's seam takes neither a delimiter nor a case policy. It inserts
/// `|` between the words of a word-delimiting normalizer, and projects ASCII
/// letters to upper case exactly when the table spells `A` and not `a`. So
/// this is the model's own statement, checked at load against the table, the
/// normalizer and that one policy, and a disagreement is refused by name
/// ([`AlignerError::Tokenization`](crate::audio::align::error::AlignerError::Tokenization))
/// rather than aligned against the wrong columns.
///
/// [`Self::specials`] names the table's NON-LEXICAL tokens beyond the blank
/// and the delimiter — a pad, a bos, an eos, an unk, or whatever else the
/// vocabulary holds that is not a letter — explicitly, by spelling. They are
/// never inferred from spelling the other way (a token is not a special
/// because it LOOKS like one): a declared special is exempt from
/// [`Self::granularity`]'s one-scalar check whatever it spells, and an
/// undeclared token is checked whatever it spells, blank and delimiter
/// aside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tokenization {
  /// The token that delimits words.
  delimiter: WordDelimiter,
  /// The case the table spells letters in.
  case: LetterCase,
  /// How finely the table segments text into tokens.
  granularity: Granularity,
  /// Non-lexical tokens beyond the blank and the delimiter, named explicitly
  /// by spelling: a pad, a bos, an eos, an unk, or whatever else the
  /// vocabulary holds. A declared special need not appear in the vocabulary;
  /// one that does not is simply never matched.
  specials: &'static [&'static str],
}

impl Tokenization {
  /// A head delimiting words with `delimiter`, spelling letters in `case`,
  /// segmenting text at `granularity`, and naming `specials` as its
  /// non-lexical tokens beyond the blank and the delimiter.
  #[must_use]
  pub const fn new(
    delimiter: WordDelimiter,
    case: LetterCase,
    granularity: Granularity,
    specials: &'static [&'static str],
  ) -> Self {
    Self {
      delimiter,
      case,
      granularity,
      specials,
    }
  }

  /// The token that delimits words.
  #[inline]
  pub const fn delimiter(&self) -> WordDelimiter {
    self.delimiter
  }

  /// The case the table spells letters in.
  #[inline]
  pub const fn case(&self) -> LetterCase {
    self.case
  }

  /// How finely the table segments text into tokens.
  #[inline]
  pub const fn granularity(&self) -> Granularity {
    self.granularity
  }

  /// Non-lexical tokens beyond the blank and the delimiter, named explicitly
  /// by spelling.
  #[inline]
  pub const fn specials(&self) -> &'static [&'static str] {
    self.specials
  }
}

/// Checks `tokenization` against `vocabulary`, against whether the normalizer
/// delimits words (`word_delimited`), against `blank` (the contract's CTC
/// blank id), and against the one policy asry's seam implements.
///
/// - A whitespace token is refused whatever the statement: asry splits words
///   at whitespace and never looks whitespace up, so such a table delimits its
///   words by something asry cannot insert, and beside a `|` does not say
///   which of the two is its delimiter.
/// - [`WordDelimiter::Pipe`] needs the table to spell `|` and the normalizer
///   to insert it; [`WordDelimiter::Absent`] needs the normalizer to insert
///   none.
/// - asry projects ASCII letters to upper case exactly when the table spells
///   `A` and not `a`. [`LetterCase::Upper`] therefore needs a table that
///   spells `A` and no lowercase ASCII letter, for which the projection reads
///   every letter's own column; [`LetterCase::AsWritten`] needs a table asry
///   does not project for.
/// - Under [`Granularity::Character`], every LEXICAL token must be exactly
///   one Unicode scalar value: asry looks a text up one character at a time,
///   so a token of another length can never be the column it reads. The
///   blank (`vocabulary`'s entry at `blank`) and a token named in
///   [`Tokenization::specials`] are non-lexical and exempt whatever they
///   spell; every other token is checked.
///
/// Only single-character tokens are letters here: asry looks a text up one
/// character at a time, so a `<pad>` or an `<unk>` is never a letter.
///
/// # Errors
/// The first [`TokenizationError`] these disagree by.
pub(crate) fn check_tokenization(
  blank: u32,
  tokenization: Tokenization,
  vocabulary: &Vocabulary,
  word_delimited: bool,
) -> Result<(), TokenizationError> {
  if let Some(whitespace) = vocabulary
    .tokens()
    .find(|token| !token.is_empty() && token.chars().all(char::is_whitespace))
  {
    return Err(TokenizationError::WhitespaceToken(whitespace.to_owned()));
  }

  match (tokenization.delimiter(), word_delimited) {
    (WordDelimiter::Pipe, true) if !vocabulary.contains("|") => {
      return Err(TokenizationError::DelimiterMissing);
    }
    (WordDelimiter::Pipe, false) => return Err(TokenizationError::DelimiterUnused),
    (WordDelimiter::Absent, true) => return Err(TokenizationError::DelimiterRequired),
    _ => {}
  }

  let projects = vocabulary.contains("A") && !vocabulary.contains("a");
  match tokenization.case() {
    LetterCase::Upper => {
      if !projects {
        return Err(if vocabulary.contains("A") {
          TokenizationError::UpperWithLowercase('a')
        } else {
          TokenizationError::UpperWithoutA
        });
      }
      if let Some(lowercase) = vocabulary.tokens().find_map(single_lowercase_letter) {
        return Err(TokenizationError::UpperWithLowercase(lowercase));
      }
    }
    LetterCase::AsWritten => {
      if projects {
        return Err(TokenizationError::ProjectedAsWritten);
      }
    }
  }

  match tokenization.granularity() {
    Granularity::Character => {
      // The blank is the contract's statement, never inferred from spelling
      // (`check_blank` — not this function — refuses `blank` itself if it
      // names no column); an out-of-range id here just names nothing, and
      // exempts nothing.
      let blank_token = usize::try_from(blank)
        .ok()
        .and_then(|blank| vocabulary.tokens().nth(blank));
      if let Some(token) = vocabulary.tokens().find(|&token| {
        Some(token) != blank_token
          && !tokenization.specials().contains(&token)
          && !is_one_scalar(token)
      }) {
        return Err(TokenizationError::NotCharacterLevel(token.to_owned()));
      }
    }
  }

  Ok(())
}

/// `token`'s one character, when it is exactly one lowercase ASCII letter.
fn single_lowercase_letter(token: &str) -> Option<char> {
  let mut chars = token.chars();
  match (chars.next(), chars.next()) {
    (Some(letter), None) if letter.is_ascii_lowercase() => Some(letter),
    _ => None,
  }
}

/// Whether `token` is exactly one Unicode scalar value: what
/// [`Granularity::Character`] requires of every lexical token, since asry
/// looks a text up one `char` at a time.
fn is_one_scalar(token: &str) -> bool {
  let mut chars = token.chars();
  matches!((chars.next(), chars.next()), (Some(_), None))
}

/// What a CTC head emits, and so how the aligner normalizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputKind {
  /// The head ends in a log-softmax: its emissions are log-probabilities. The
  /// encoder checks each frame is normalized (a logsumexp within
  /// [`log_prob_sum_tolerance`](crate::audio::align::encode::log_prob_sum_tolerance)
  /// of 0) and hands the values on as they are. Only a head no wider than
  /// [`MAX_LOG_PROB_WIDTH`](crate::audio::align::encode::MAX_LOG_PROB_WIDTH)
  /// classes can be checked apart from an unnormalized one; a wider head is
  /// refused at load
  /// ([`AlignerError::UnprovableNormalization`](crate::audio::align::error::AlignerError::UnprovableNormalization)).
  LogProbabilities,
  /// The head ends in its last linear layer: raw logits. asry normalizes them
  /// (a log-softmax), so there is nothing to check. A head that does end in a
  /// log-softmax may be stated as this too: normalizing log-probabilities again
  /// changes nothing, and that is the road for a head too wide to check.
  Logits,
}

/// A CTC aligner model's contract: what the aligner must know about the model
/// and cannot read from it.
///
/// - [`Self::blank`]: the id of the head's CTC blank. The aligner checks it
///   against the vocabulary at load.
/// - [`Self::geometry`]: the front end's [`AcousticGeometry`]. The encoder
///   checks it against the model's declared window and frame count at load,
///   then truncates by it; the seam is handed its stride.
/// - [`Self::tokenization`]: how the head spells a word. The aligner checks it
///   against the table, the normalizer and asry's one policy at load.
/// - [`Self::output`]: what the head emits, which decides how its emissions
///   are normalized. The encoder checks a log-probability head's width at load.
/// - [`Self::sentinel_band`]: values the model is measured to emit in place of
///   a log-probability, refused wherever they appear. Only the staged artifact
///   has one.
///
/// [`Self::BASE960H`] is the staged artifact's. Any other model states its own
/// with [`Self::new`], and no part of the aligner guesses one: a blank, a
/// geometry, a tokenization, an output kind and a band are each the caller's
/// statement or the staged artifact's measurement, never an inference from the
/// table or the declared shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AcousticContract {
  /// The id of the CTC blank.
  blank: u32,
  /// The front end's geometry.
  geometry: AcousticGeometry,
  /// How the head spells a word.
  tokenization: Tokenization,
  /// What the head emits.
  output: OutputKind,
  /// Values measured in place of a log-probability, if the artifact has any.
  sentinel_band: Option<SentinelBand>,
}

impl AcousticContract {
  /// The staged `base960h_aligner.mlmodelc`'s contract. Its blank is id 0, the
  /// `-` of its own table ([`BLANK_ID`]). Its geometry is
  /// [`AcousticGeometry::WAV2VEC2`]. It delimits words with `|` and spells
  /// letters in upper case, one character at a time
  /// ([`Granularity::Character`]) — its 29-class table's only non-lexical
  /// entries are that blank and that delimiter, both already named, so it
  /// declares no further specials. Its graph ends in `softmax` then `log`, so
  /// it emits [`OutputKind::LogProbabilities`]. Its sentinel band is
  /// [`SentinelBand::Fp16Saturation`], where that fp16 tail saturates on the
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
    tokenization: Tokenization::new(
      WordDelimiter::Pipe,
      LetterCase::Upper,
      Granularity::Character,
      &[],
    ),
    output: OutputKind::LogProbabilities,
    sentinel_band: Some(SentinelBand::Fp16Saturation),
  };

  /// The contract of a model loaded with its own vocabulary.
  ///
  /// `blank` is the id of its CTC blank: the class its head scores as "no
  /// token here", named by the model rather than by its table. A HuggingFace
  /// `config.json` calls it `pad_token_id`, and chordai's and torchaudio's
  /// exports put a `-` at id 0. `geometry` is its front end's, `tokenization`
  /// how its head spells a word (a HuggingFace `config.json` names the
  /// delimiter `word_delimiter_token`), and `output` what its head emits. The
  /// contract carries no sentinel band.
  #[must_use]
  pub const fn new(
    blank: u32,
    geometry: AcousticGeometry,
    tokenization: Tokenization,
    output: OutputKind,
  ) -> Self {
    Self {
      blank,
      geometry,
      tokenization,
      output,
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

  /// How the head spells a word.
  #[inline]
  pub const fn tokenization(&self) -> Tokenization {
    self.tokenization
  }

  /// What the head emits.
  #[inline]
  pub const fn output(&self) -> OutputKind {
    self.output
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
