use core::sync::atomic::AtomicBool;

use asry::emissions::{EmissionsAligner, EnglishNormalizer, SpeechSpans};

use super::*;
use crate::audio::align::{Lang, vocab::tokenizer_json_bytes};

fn nonzero(value: u32) -> NonZeroU32 {
  NonZeroU32::new(value).expect("nonzero")
}

/// The staged model's tokenization: `|` between words, letters in upper case,
/// one character at a time, no specials beyond the blank and the delimiter.
const PIPE_UPPER: Tokenization = Tokenization::new(
  WordDelimiter::Pipe,
  LetterCase::Upper,
  Granularity::Character,
  &[],
);

/// A table from `tokens`, each at its index.
fn table(tokens: &[&str]) -> Vocabulary {
  let entries: Vec<String> = tokens
    .iter()
    .enumerate()
    .map(|(id, token)| format!("{}: {id}", serde_json::to_string(token).expect("a token")))
    .collect();
  Vocabulary::from_json(format!("{{{}}}", entries.join(", ")).as_bytes()).expect("a table")
}

/// The staged artifact's contract is the facts its tests and its measurements
/// pinned: its table's `-` at id 0, wav2vec2's front end, `|` between words
/// and upper-case letters, a head ending in `softmax` then `log`, and the fp16
/// saturation band that tail saturates into on the Neural Engine.
#[test]
fn the_staged_contract_is_the_staged_artifacts() {
  let contract = AcousticContract::BASE960H;
  assert_eq!(contract.blank(), 0);
  assert_eq!(contract.blank(), BLANK_ID);
  assert_eq!(contract.geometry(), AcousticGeometry::WAV2VEC2);
  assert_eq!(contract.tokenization(), PIPE_UPPER);
  assert_eq!(contract.output(), OutputKind::LogProbabilities);
  assert_eq!(contract.sentinel_band(), Some(SentinelBand::Fp16Saturation));
  // ...and the staged table and the English normalizer satisfy it.
  assert_eq!(
    check_tokenization(
      contract.blank(),
      contract.tokenization(),
      &Vocabulary::bundled(),
      true
    ),
    Ok(())
  );

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
  let contract = AcousticContract::new(3, geometry, PIPE_UPPER, OutputKind::Logits);
  assert_eq!(contract.blank(), 3);
  assert_eq!(contract.geometry(), geometry);
  assert_eq!(contract.tokenization(), PIPE_UPPER);
  assert_eq!(contract.output(), OutputKind::Logits);
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

// ---------------------------------------------------------------------
// Tokenization: the model's statement, checked at load against the table, the
// normalizer and the one policy asry's seam implements (`|` between the words
// of a word-delimiting normalizer; ASCII upper-case projection exactly when
// the table spells `A` and not `a`).
// ---------------------------------------------------------------------

/// **A `|`-containing, space-delimited vocabulary is refused by name.** asry
/// splits words at whitespace and never looks whitespace up, so a table that
/// spells a space delimits its words by something asry cannot insert, and
/// beside a `|` it does not say which of the two is its delimiter. Refused
/// whatever the contract states — and stating the space as the delimiter is
/// refused when the contract is made.
///
/// Mutation check: deleting the whitespace clause of `check_tokenization`
/// turns the first assertion green-for-the-wrong-reason no longer: the table
/// is accepted, and this test fails.
#[test]
fn a_pipe_containing_space_delimited_table_is_refused_by_name() {
  let spaced = table(&["<pad>", " ", "|", "A", "B"]);
  for tokenization in [
    PIPE_UPPER,
    Tokenization::new(
      WordDelimiter::Absent,
      LetterCase::Upper,
      Granularity::Character,
      &[],
    ),
  ] {
    for word_delimited in [true, false] {
      assert_eq!(
        check_tokenization(0, tokenization, &spaced, word_delimited),
        Err(TokenizationError::WhitespaceToken(" ".to_owned())),
        "{tokenization:?}, word_delimited {word_delimited}"
      );
    }
  }
  assert_eq!(
    WordDelimiter::from_token(" "),
    Err(TokenizationError::UnsupportedDelimiter(" ".to_owned()))
  );
  assert_eq!(WordDelimiter::from_token("|"), Ok(WordDelimiter::Pipe));
}

/// **The case table with `A`, `B` and `b` but no `a` is refused under either
/// statement.** asry projects every ASCII letter to upper case exactly when
/// the table spells `A` and not `a`. Stated upper case, the table's own `b`
/// would never be read; stated as written, asry would project anyway and read
/// `B` for every `b`.
///
/// Mutation checks: deleting the lowercase-letter clause lets the upper-case
/// statement through; deleting the projection clause lets the as-written one
/// through. Either fails this test.
#[test]
fn the_a_b_b_table_without_a_is_refused_under_either_case() {
  let mixed = table(&["<pad>", "|", "A", "B", "b"]);
  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &mixed, true),
    Err(TokenizationError::UpperWithLowercase('b'))
  );
  assert_eq!(
    check_tokenization(
      0,
      Tokenization::new(
        WordDelimiter::Pipe,
        LetterCase::AsWritten,
        Granularity::Character,
        &[],
      ),
      &mixed,
      true
    ),
    Err(TokenizationError::ProjectedAsWritten)
  );
}

/// The case statements asry honours pass, and every other one is named: an
/// upper-case table (projected), a lowercase one (as written), one that spells
/// both cases (as written), and a table with no Latin letter at all (as
/// written) — and each of those stated the other way is refused.
#[test]
fn every_case_statement_is_checked_against_asrys_projection() {
  let as_written = Tokenization::new(
    WordDelimiter::Pipe,
    LetterCase::AsWritten,
    Granularity::Character,
    &[],
  );
  let upper = table(&["<pad>", "|", "A", "B"]);
  let lower = table(&["<pad>", "|", "a", "b"]);
  let both = table(&["<pad>", "|", "A", "a", "B", "b"]);
  let han = table(&["<pad>", "|", "中", "文"]);

  assert_eq!(check_tokenization(0, PIPE_UPPER, &upper, true), Ok(()));
  assert_eq!(check_tokenization(0, as_written, &lower, true), Ok(()));
  assert_eq!(check_tokenization(0, as_written, &both, true), Ok(()));
  assert_eq!(check_tokenization(0, as_written, &han, true), Ok(()));

  assert_eq!(
    check_tokenization(0, as_written, &upper, true),
    Err(TokenizationError::ProjectedAsWritten)
  );
  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &lower, true),
    Err(TokenizationError::UpperWithoutA)
  );
  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &both, true),
    Err(TokenizationError::UpperWithLowercase('a'))
  );
  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &han, true),
    Err(TokenizationError::UpperWithoutA)
  );
}

/// The delimiter statement must agree with the table and the normalizer: `|`
/// needs a table that spells it and a normalizer that inserts it; no delimiter
/// needs a normalizer that inserts none. A multi-character token such as
/// `<pad>` is never a letter.
#[test]
fn the_delimiter_statement_is_checked_against_the_table_and_the_normalizer() {
  let absent = Tokenization::new(
    WordDelimiter::Absent,
    LetterCase::Upper,
    Granularity::Character,
    &[],
  );
  let with_pipe = table(&["<pad>", "|", "A"]);
  let without_pipe = table(&["<pad>", "A", "B"]);

  assert_eq!(check_tokenization(0, PIPE_UPPER, &with_pipe, true), Ok(()));
  assert_eq!(check_tokenization(0, absent, &without_pipe, false), Ok(()));
  assert_eq!(check_tokenization(0, absent, &with_pipe, false), Ok(()));

  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &without_pipe, true),
    Err(TokenizationError::DelimiterMissing)
  );
  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &with_pipe, false),
    Err(TokenizationError::DelimiterUnused)
  );
  assert_eq!(
    check_tokenization(0, absent, &without_pipe, true),
    Err(TokenizationError::DelimiterRequired)
  );
}

// ---------------------------------------------------------------------
// Granularity: asry looks a text up one Unicode character at a time
// (`Granularity::Character`), so a LEXICAL token of any other length is
// refused by name; the blank and a declared special are exempt whatever they
// spell, because neither is a letter this seam looks up.
// ---------------------------------------------------------------------

/// **A subword class beside its own characters is refused by name.** A table
/// can truthfully hold `A`, `B` and the class `AB` a model's own tokenizer
/// emits for "AB" — nothing about the table is malformed — and asry would
/// still look `A` and `B` up separately and never read the `AB` column: the
/// alignment would be silently built from the wrong classes.
///
/// Mutation check: disabling `check_tokenization`'s one-scalar clause
/// (`is_one_scalar`, forced to always return `true`) turns this
/// green-for-the-wrong-reason no longer — the table is accepted and this test
/// fails. Verified by hand and reverted; not left in the tree.
#[test]
fn a_subword_class_beside_its_own_characters_is_refused_by_name() {
  let subword = table(&["<pad>", "|", "A", "B", "AB"]);
  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &subword, true),
    Err(TokenizationError::NotCharacterLevel("AB".to_owned()))
  );
}

/// **A multi-character token is accepted once, and only once, it is declared
/// a special.** The same table refuses `<pad>` — here NOT the blank, so its
/// exemption can only come from [`Tokenization::specials`] — when nothing
/// names it special, and passes once the contract does. Spelling like a
/// special is never enough on its own.
#[test]
fn a_multicharacter_token_is_accepted_only_when_declared_special() {
  let with_pad = table(&["-", "|", "A", "B", "<pad>"]);
  let undeclared = Tokenization::new(
    WordDelimiter::Pipe,
    LetterCase::Upper,
    Granularity::Character,
    &[],
  );
  let declared = Tokenization::new(
    WordDelimiter::Pipe,
    LetterCase::Upper,
    Granularity::Character,
    &["<pad>"],
  );
  assert_eq!(
    check_tokenization(0, undeclared, &with_pad, true),
    Err(TokenizationError::NotCharacterLevel("<pad>".to_owned()))
  );
  assert_eq!(check_tokenization(0, declared, &with_pad, true), Ok(()));
}

/// **The blank is exempt by id, not by a `specials` declaration.** A blank
/// spelled `<pad>` — the HuggingFace convention, unlike the staged model's
/// own `-` — is never checked against the one-scalar rule: `blank` names it
/// by id, the same way [`AlignerError::BlankOutOfVocabulary`] does, never by
/// spelling.
#[test]
fn a_multicharacter_blank_is_exempt_without_being_declared_special() {
  let hf_style = table(&["<pad>", "|", "A", "B"]);
  assert_eq!(
    check_tokenization(0, PIPE_UPPER, &hf_style, true),
    Ok(()),
    "id 0, `<pad>`, is the stated blank and needs no `specials` entry"
  );
}

/// **A single-scalar non-ASCII letter passes as lexical.** `é` and `ß` are
/// each one Unicode scalar value (precomposed, not a base letter plus a
/// combining mark), so [`Granularity::Character`] reads them as ordinary
/// letters — the same as any ASCII one — with no need to declare either a
/// special.
#[test]
fn a_single_scalar_non_ascii_letter_passes_as_lexical() {
  let accented = table(&["-", "|", "\u{e9}", "\u{df}"]);
  let as_written = Tokenization::new(
    WordDelimiter::Pipe,
    LetterCase::AsWritten,
    Granularity::Character,
    &[],
  );
  assert_eq!(check_tokenization(0, as_written, &accented, true), Ok(()));
}
